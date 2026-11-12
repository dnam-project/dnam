#![forbid(unsafe_code)]
//! `A1..A6` Deterministic Excitation Waveforms, `Theorem 4.7` Operator-Aligned Primitives
//! ($\mathcal{P}_1\dots\mathcal{P}_5$), `Theorem 4.8` Two-Episode "Pump-and-Probe" Bloom,
//! and `INPC` / `STMC` Coreset & Spectral Summary Encoders (`stimulus::waveforms`).
//!
//! Canonical References:
//! - `docs/specs/training/capture_and_alignment.md` (`ref[train.capture.excitation.codesigned]`)
//! - `docs/specs/containers/project_container_format.md` (`ref[container.proj.inpc-capsule]`, `ref[container.proj.stmc-capsule]`)

use std::f64::consts::PI;

use crate::solver::bezout_three_band_weights;
use crate::{CORESET_FP16_FRAMES_PER_STEM, f32_to_fp16_bits};

use super::fft::{
    ADAPTIVE_FRAME_SAMPLES, ADC_PEAK_GOVERNOR_FS, CYCLIC_WINDOW_SAMPLES, HANN_CROSSFADE_SAMPLES,
    irfft_from_sparse_tones, rfft_bins, synthesize_cyclic_comb_frame_with_hint,
};
use super::laws::{Disambiguated2To1Mode, TopologyProbeMode};

/// Canonical `500 ms` (`24,000` samples at `48 kHz`) `A5` musical transient coreset length.
pub const A5_CORESET_SAMPLES_500MS: usize = 24_000;

/// Canonical `250 ms` (`12,000` samples at `48 kHz`) `A3` monophonic glissando length (`Law 0.E`).
pub const A3_GLISSANDO_SAMPLES_250MS: usize = 12_000;

/// Canonical `512`-sample (`10.667 ms`, $f_0 = 93.75\text{ Hz}$) periodic reverse-biased comb probe
/// length for `Theorem 4.8` Pump-and-Probe Bloom recovery tracking.
pub const PUMP_PROBE_COMB_PERIOD: usize = 512;

/// `Theorem 4.8` Episode 1 -> Episode 2 gain-compression/bloom trigger threshold (`0.5 dB`).
pub const BLOOM_EPISODE2_TRIGGER_DB: f64 = 0.5;

/// `Theorem 4.7` 5 Operator-Aligned Active-Learning Primitives ($\mathcal{P}_1\dots\mathcal{P}_5$).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivePrimitiveKind {
    /// $\mathcal{P}_1$: Monophonic octave-crossing glide + sharp release targeting Stage I `T1`.
    P1BranchCutChirp,
    /// $\mathcal{P}_2$: 5-knee multi-drive asymmetric/wavefolder burst targeting Stage II `V1A/V1B/C^2`.
    P2MultiDriveBurst,
    /// $\mathcal{P}_3$: Constant-$V/\omega$ (`40–220 Hz`) LF flux sweep targeting `QuadratureHysteresis2D`.
    P3ConstantFluxLfSweep,
    /// $\mathcal{P}_4$: `24 ms` dense chordal burst + `72 ms` ducking release gap targeting Stage IV FDN.
    P4TankDuckingBurst,
    /// $\mathcal{P}_5$: `Theorem 4.8` Two-Episode "Pump-and-Probe" Bloom & Sag probe.
    P5PumpAndProbeBloom,
}

impl ActivePrimitiveKind {
    /// All 5 operator-aligned primitives in canonical order.
    pub const ALL: [Self; 5] = [
        Self::P1BranchCutChirp,
        Self::P2MultiDriveBurst,
        Self::P3ConstantFluxLfSweep,
        Self::P4TankDuckingBurst,
        Self::P5PumpAndProbeBloom,
    ];
}

/// Telemetry extracted from a `Theorem 4.8` Pump-and-Probe Bloom recovery probe sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct BloomEpisodeTelemetry {
    /// Instantaneous post-pump gain shift $\Delta g_{\text{dB}}(0^+)$ in the first `512`-sample window.
    pub initial_delta_gain_db: f64,
    /// True if $|\Delta g_{\text{dB}}(0^+)| > 0.5\text{ dB}$, triggering Episode 2 (`576 ms + 576 ms`).
    pub trigger_episode_2: bool,
    /// Estimated exponential sag/thermal recovery time constant $\hat{\tau}_{\text{sag}}$ in ms.
    pub recovery_tau_ms: f64,
    /// Peak even-harmonic bias shift ratio observed immediately after pump cutoff.
    pub bias_asymmetry_ratio: f64,
    /// Per-`512`-sample-window gain envelope trajectory in dB relative to steady reference.
    pub window_gain_db: Vec<f32>,
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn seed_to_u64(seed_256: &[u8; 256], stream_idx: usize) -> u64 {
    let off = (stream_idx * 8) % 248;
    u64::from_le_bytes(seed_256[off..off + 8].try_into().unwrap_or([0; 8]))
        ^ ((stream_idx as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93))
}

/// Generates a deterministic `256 B` stimulus seed $\boldsymbol{\Xi}$ from a `u64` master seed.
#[must_use]
pub fn deterministic_seed_256(master_seed: u64) -> [u8; 256] {
    let mut st = master_seed ^ 0x444E_414D_5345_4544; // "DNAMSEED"
    let mut out = [0u8; 256];
    for chunk in out.as_chunks_mut::<8>().0 {
        chunk.copy_from_slice(&splitmix64(&mut st).to_le_bytes());
    }
    out
}

/// Synthesizes `A1`: Phase-continuous logarithmic swept sine (`22 Hz -> min(20 kHz, 0.45 * fs)`).
#[must_use]
pub fn synthesize_a1_log_chirp(num_samples: usize, sample_rate_hz: u32, peak_fs: f64) -> Vec<f32> {
    if num_samples == 0 || sample_rate_hz == 0 {
        return Vec::new();
    }
    let sr = f64::from(sample_rate_hz);
    let amp = peak_fs.clamp(0.0, ADC_PEAK_GOVERNOR_FS);
    let f0 = 22.0_f64;
    let f1 = (0.45 * sr).clamp(120.0, 20_000.0);
    let ln_ratio = (f1 / f0).ln();
    let dur = (num_samples as f64) / sr;
    let fade = (num_samples / 16).max(4);
    (0..num_samples)
        .map(|n| {
            let t = (n as f64) / sr;
            let ph = 2.0 * PI * f0 * dur * ((t / dur * ln_ratio).exp() - 1.0) / ln_ratio;
            let env = if n < fade {
                0.5 * (1.0 - (PI * (n as f64) / (fade as f64)).cos())
            } else if n + fade >= num_samples {
                let rem = num_samples - 1 - n;
                0.5 * (1.0 - (PI * (rem as f64) / (fade as f64)).cos())
            } else {
                1.0
            };
            (amp * env * ph.sin()) as f32
        })
        .collect()
}

/// Synthesizes `A2`: Multi-level dynamic step ramp (`-48 dBFS` up to `peak_fs` across 6 steps).
#[must_use]
pub fn synthesize_a2_dynamic_staircase(num_samples: usize, peak_fs: f64) -> Vec<f32> {
    if num_samples == 0 {
        return Vec::new();
    }
    let top = peak_fs.clamp(0.01, ADC_PEAK_GOVERNOR_FS);
    let levels_db = [-48.0_f64, -36.0, -24.0, -14.0, -6.0, 0.0];
    let step_len = (num_samples / levels_db.len()).max(1);
    (0..num_samples)
        .map(|n| {
            let idx = (n / step_len).min(levels_db.len() - 1);
            let gain = top * 10.0_f64.powf(levels_db[idx] / 20.0);
            let t = (n as f64) / 48_000.0;
            let carrier = 0.65 * (2.0 * PI * 196.0 * t).sin() + 0.35 * (2.0 * PI * 588.0 * t).sin();
            (gain * carrier) as f32
        })
        .collect()
}

/// Synthesizes `A3`: `250 ms` (`12,000`-sample at `48 kHz`) continuous single-voice monophonic
/// pitch glissando (`65 Hz -> 880 Hz`) with zero polyphonic beating (`Law 0.E` `OC-2` handoff).
#[must_use]
pub fn synthesize_a3_monophonic_glissando(sample_rate_hz: u32, peak_fs: f64) -> Vec<f32> {
    if sample_rate_hz == 0 || !peak_fs.is_finite() || peak_fs <= 0.0 {
        return Vec::new();
    }
    let sr = f64::from(sample_rate_hz.clamp(8_000, 384_000));
    let n = ((sr * 0.25).round() as usize).max(64);
    let amp = peak_fs.clamp(0.01, ADC_PEAK_GOVERNOR_FS);
    let f0 = 65.406_39_f64; // C2
    let f1 = 880.0_f64; // A5
    let ln_r = (f1 / f0).ln();
    let dur = (n as f64) / sr;
    let fade = (n / 20).max(8);
    (0..n)
        .map(|i| {
            let t = (i as f64) / sr;
            let ph = 2.0 * PI * f0 * dur * ((t / dur * ln_r).exp() - 1.0) / ln_r;
            let env = if i < fade {
                0.5 * (1.0 - (PI * (i as f64) / (fade as f64)).cos())
            } else if i + fade >= n {
                let rem = n - 1 - i;
                0.5 * (1.0 - (PI * (rem as f64) / (fade as f64)).cos())
            } else {
                1.0
            };
            (amp * env * ph.sin()) as f32
        })
        .collect()
}

/// Synthesizes `A5`: Deterministic musical transient & chordal coreset (`num_frames` frames, e.g.
/// `24,000` for `500 ms` capture or `12,000` for `250 ms` `INPC` coreset; `channels` in `1..=2`).
/// Returns `(ch_l, Option<ch_r>)` governed strictly below `ADC_PEAK_GOVERNOR_FS`.
#[must_use]
pub fn synthesize_a5_musical_transient_coreset(
    seed_256: &[u8; 256],
    num_frames: usize,
    channels: u16,
    sample_rate_hz: u32,
) -> (Vec<f32>, Option<Vec<f32>>) {
    let sr = f64::from(sample_rate_hz.max(8_000));
    let mut rng_l = seed_to_u64(seed_256, 0);
    let mut rng_r = seed_to_u64(seed_256, 1);

    let build_channel = |rng: &mut u64, pan_skew: f64| -> Vec<f32> {
        if num_frames == 0 {
            return Vec::new();
        }
        let mut buf = vec![0.0_f64; num_frames];
        // 4 musical phrases across num_frames:
        // [0..25%]: Crisp pick-attack single-note plucks (E2=82.41 Hz, D3=146.83 Hz, B3=246.94 Hz)
        // [25..50%]: Palm-muted low-end chugs + ghost transients (stages I/II sag & hysteresis)
        // [50..75%]: Open power-chord / triad dyad (A2=110 Hz + E3=164.81 Hz + A3=220 Hz + C#4=277.18 Hz)
        // [75..100%]: Broadband percussive impulse + natural release decay gap (Stage III/IV tank)
        let q1 = num_frames / 4;
        let q2 = num_frames / 2;
        let q3 = (3 * num_frames) / 4;

        let jitter = (((splitmix64(rng) >> 16) as f64) / ((1u64 << 48) as f64) - 0.5) * 0.02;
        let freqs_pluck = [82.406_89 * (1.0 + jitter), 146.832_38, 246.941_65];
        let sub_len = (q1 / 3).max(1);
        for (i, slot) in buf.iter_mut().enumerate().take(q1) {
            let note_idx = (i / sub_len).min(2);
            let local_n = i % sub_len;
            let t = (local_n as f64) / sr;
            let f = freqs_pluck[note_idx] * pan_skew;
            let attack = 1.0 - (-t / 0.0012).exp();
            let decay = (-t / 0.045).exp();
            let mut s = 0.0_f64;
            for h in 1..=6 {
                let hf = f * (h as f64);
                if hf < 0.45 * sr {
                    let h_decay = (-t * (h as f64) * 9.0).exp();
                    s += (1.0 / (h as f64)) * h_decay * (2.0 * PI * hf * t).sin();
                }
            }
            *slot = 0.65 * attack * decay * s;
        }

        let chug_len = ((q2 - q1) / 3).max(1);
        for i in q1..q2 {
            let local = (i - q1) % chug_len;
            let t = (local as f64) / sr;
            let env = (1.0 - (-t / 0.0008).exp()) * (-t / 0.018).exp();
            let f = 92.498_6 * pan_skew;
            let noise_u = (((splitmix64(rng) >> 16) as f64) / ((1u64 << 48) as f64)) - 0.5;
            let pick = noise_u * (-t / 0.0025).exp();
            buf[i] = env
                * (0.75 * (2.0 * PI * f * t).sin() + 0.35 * (4.0 * PI * f * t).sin() + 0.35 * pick);
        }

        let chord_freqs = [110.0, 164.813_78, 220.0, 277.182_63, 440.0, 659.255_1];
        for (idx, slot) in buf[q2..q3].iter_mut().enumerate() {
            let t = (idx as f64) / sr;
            let env = (1.0 - (-t / 0.002).exp()) * (-t / 0.095).exp();
            let mut s = 0.0_f64;
            for (k, &cf) in chord_freqs.iter().enumerate() {
                let f = cf * (1.0 + 0.003 * ((k as f64) - 2.5) * (pan_skew - 1.0));
                if f < 0.45 * sr {
                    s += (1.0 / ((k + 1) as f64).sqrt())
                        * (2.0 * PI * f * t + (k as f64) * 0.7).sin();
                }
            }
            *slot = 0.48 * env * s;
        }

        // Final quarter: crisp transient burst in first 30% followed by clean decay gap
        let q4_len = num_frames.saturating_sub(q3);
        let burst_end = (q4_len * 3) / 10;
        for idx in 0..burst_end {
            let t = (idx as f64) / sr;
            let env = (1.0 - (-t / 0.0006).exp()) * (-t / 0.012).exp();
            let noise_u = (((splitmix64(rng) >> 16) as f64) / ((1u64 << 48) as f64)) - 0.5;
            buf[q3 + idx] = env * (0.8 * noise_u + 0.5 * (2.0 * PI * 330.0 * pan_skew * t).sin());
        }

        // Two-pass zero-phase [0.25, 0.5, 0.25] Nyquist-nulling anti-imaging filter
        for _ in 0..2 {
            let prev = buf.clone();
            for i in 1..num_frames.saturating_sub(1) {
                buf[i] = 0.25 * prev[i - 1] + 0.5 * prev[i] + 0.25 * prev[i + 1];
            }
        }

        let peak = buf.iter().fold(1.0e-12_f64, |acc, &v| acc.max(v.abs()));
        let scale = (0.78 * ADC_PEAK_GOVERNOR_FS) / peak;
        buf.into_iter().map(|v| (v * scale) as f32).collect()
    };

    let ch_l = build_channel(&mut rng_l, 1.0);
    let ch_r = (channels >= 2).then(|| build_channel(&mut rng_r, 1.007));
    (ch_l, ch_r)
}

/// Applies `Law 4C` (`3 x 32 ms` = `3 x 1,536` samples at `{-18 dB, -6 dB, 0 dB}`) micro-dynamic
/// staircase envelope to a `4,608`-sample frame with `32`-sample $C^1$ cosine transitions.
#[must_use]
pub fn apply_law4c_micro_staircase(base_frame: &[f32], peak_ceiling_fs: f64) -> Vec<f32> {
    let gains = [0.125_892_54_f64, 0.501_187_23_f64, 1.0_f64]; // -18 dB, -6 dB, 0 dB
    let n = base_frame.len();
    if n == 0 {
        return Vec::new();
    }
    let sub_len = (n / 3).max(1);
    let xfade = 32usize.min(sub_len / 2);
    let ceiling = peak_ceiling_fs.clamp(0.01, ADC_PEAK_GOVERNOR_FS) as f32;
    (0..n)
        .map(|i| {
            let seg = (i / sub_len).min(2);
            let local = i - seg * sub_len;
            let g = if seg > 0 && local < xfade {
                let w = 0.5 * (1.0 - (PI * (local as f64) / (xfade as f64)).cos());
                (1.0 - w) * gains[seg - 1] + w * gains[seg]
            } else {
                gains[seg]
            };
            ((f64::from(base_frame[i]) * g) as f32).clamp(-ceiling, ceiling)
        })
        .collect()
}

/// Synthesizes a single `512`-sample (`10.667 ms`) `-30 dBFS` reverse-biased odd-comb probe tile
/// for `Theorem 4.8` Pump-and-Probe Bloom recovery tracking.
#[must_use]
pub fn synthesize_512_bloom_comb_probe(probe_rms_fs: f64) -> [f32; PUMP_PROBE_COMB_PERIOD] {
    let odd_bins = [1usize, 3, 5, 9, 15, 25, 41, 67];
    let tones: Vec<(usize, f64, f64)> = odd_bins
        .iter()
        .enumerate()
        .map(|(idx, &b)| {
            let ph = -PI * ((idx * idx) as f64) / (odd_bins.len() as f64);
            (b, 1.0, ph)
        })
        .collect();
    let raw = irfft_from_sparse_tones(PUMP_PROBE_COMB_PERIOD, &tones);
    let rms = (raw.iter().map(|&x| x * x).sum::<f64>() / (PUMP_PROBE_COMB_PERIOD as f64)).sqrt();
    let scale = if rms > 1.0e-18 {
        probe_rms_fs.clamp(1.0e-4, 0.25) / rms
    } else {
        0.0
    };
    let mut out = [0.0_f32; PUMP_PROBE_COMB_PERIOD];
    for (dst, &v) in out.iter_mut().zip(&raw) {
        *dst = (v * scale) as f32;
    }
    out
}

/// Synthesizes one `4,608`-sample (`96 ms`) frame of one of the 5 `Theorem 4.7` active-learning
/// primitives ($\mathcal{P}_1\dots\mathcal{P}_5$, including `Theorem 4.8` Episode 1 & Episode 2).
#[must_use]
pub fn synthesize_active_primitive_frame(
    kind: ActivePrimitiveKind,
    episode_frame_offset: usize,
    is_episode2: bool,
    peak_ceiling_fs: f64,
) -> Vec<f32> {
    let ceiling = peak_ceiling_fs.clamp(0.01, ADC_PEAK_GOVERNOR_FS);
    let sr = 48_000.0_f64;
    let mut out = vec![0.0_f32; ADAPTIVE_FRAME_SAMPLES];

    match kind {
        ActivePrimitiveKind::P1BranchCutChirp => {
            // 64 ms octave-crossing pitch glide (82 Hz -> 330 Hz) + 32 ms abrupt envelope drop
            let active_n = 3_072usize;
            for (n, slot) in out.iter_mut().enumerate().take(active_n) {
                let u = (n as f64) / (active_n as f64);
                let ph = 2.0 * PI * 82.4 * ((active_n as f64) / sr) * (4.0_f64.powf(u) - 1.0)
                    / 4.0_f64.ln();
                let env = (1.0 - 0.75 * u) * (PI * u).sin().sqrt();
                *slot = (0.85 * ceiling * env * ph.sin()) as f32;
            }
        }
        ActivePrimitiveKind::P2MultiDriveBurst => {
            // 5-knee multi-amplitude asymmetric crest burst across 4,608 samples
            let knees = [0.12_f64, 0.28, 0.52, 0.78, 1.0];
            let seg = ADAPTIVE_FRAME_SAMPLES / knees.len();
            for (n, slot) in out.iter_mut().enumerate() {
                let k_idx = (n / seg).min(knees.len() - 1);
                let t = (n as f64) / sr;
                let wave =
                    0.70 * (2.0 * PI * 187.5 * t).sin() + 0.30 * (2.0 * PI * 375.0 * t).cos();
                *slot = (ceiling * knees[k_idx] * wave) as f32;
            }
        }
        ActivePrimitiveKind::P3ConstantFluxLfSweep => {
            // Constant V/omega (amplitude proportional to frequency) over 40 -> 220 Hz
            let f0 = 40.0_f64;
            let f1 = 220.0_f64;
            let dur = (ADAPTIVE_FRAME_SAMPLES as f64) / sr;
            let ln_r = (f1 / f0).ln();
            for (n, slot) in out.iter_mut().enumerate() {
                let u = (n as f64) / (ADAPTIVE_FRAME_SAMPLES as f64);
                let f_inst = f0 * (f1 / f0).powf(u);
                let ph = 2.0 * PI * f0 * dur * ((u * ln_r).exp() - 1.0) / ln_r;
                let amp_v_over_w = ceiling * (f_inst / f1).clamp(0.18, 1.0);
                *slot = (amp_v_over_w * ph.sin()) as f32;
            }
        }
        ActivePrimitiveKind::P4TankDuckingBurst => {
            // 24 ms (1,152 samples) dense transient chord burst + 72 ms (3,456 samples) quiet gap
            let burst_n = 1_152usize;
            for (n, slot) in out.iter_mut().enumerate().take(burst_n) {
                let t = (n as f64) / sr;
                let win = (PI * (n as f64) / (burst_n as f64)).sin();
                let s = 0.45 * (2.0 * PI * 146.83 * t).sin()
                    + 0.35 * (2.0 * PI * 220.0 * t).sin()
                    + 0.20 * (2.0 * PI * 587.33 * t).sin();
                *slot = (ceiling * win * s) as f32;
            }
        }
        ActivePrimitiveKind::P5PumpAndProbeBloom => {
            // Episode 1: frame 0 = pump (96 ms), frames 1..4 = probe (288 ms)
            // Episode 2: frames 0..6 = pump (576 ms), frames 6..12 = probe (576 ms)
            let pump_frames = if is_episode2 { 6usize } else { 1usize };
            if episode_frame_offset < pump_frames {
                for (n, slot) in out.iter_mut().enumerate() {
                    let t = ((episode_frame_offset * ADAPTIVE_FRAME_SAMPLES + n) as f64) / sr;
                    let s =
                        0.65 * (2.0 * PI * 93.75 * t).sin() + 0.35 * (2.0 * PI * 281.25 * t).sin();
                    let fade_out = if episode_frame_offset + 1 == pump_frames
                        && n + 32 >= ADAPTIVE_FRAME_SAMPLES
                    {
                        let rem = ADAPTIVE_FRAME_SAMPLES - 1 - n;
                        0.5 * (1.0 - (PI * (rem as f64) / 32.0).cos())
                    } else {
                        1.0
                    };
                    *slot = (ceiling * fade_out * s) as f32;
                }
            } else {
                let tile = synthesize_512_bloom_comb_probe(0.031_622_776_6); // -30 dBFS
                for (n, slot) in out.iter_mut().enumerate() {
                    *slot = tile[n % PUMP_PROBE_COMB_PERIOD];
                }
            }
        }
    }
    out
}

/// Analyzes `Theorem 4.8` `512`-sample periodic comb probe windows (`probe_wet`) following a
/// high-amplitude pump burst, extracting $\Delta g_{\text{dB}}(0^+)$, Episode-2 trigger, and $\hat{\tau}_{\text{sag}}$.
#[must_use]
pub fn analyze_pump_and_probe_recovery(
    probe_wet: &[f32],
    steady_ref_odd_rms: Option<f64>,
) -> BloomEpisodeTelemetry {
    let odd_bins = [1usize, 3, 5, 9, 15, 25, 41, 67];
    let even_bins = [2usize, 6, 10, 18, 30, 50];
    let num_win = probe_wet.len() / PUMP_PROBE_COMB_PERIOD;
    if num_win == 0 {
        return BloomEpisodeTelemetry {
            initial_delta_gain_db: 0.0,
            trigger_episode_2: false,
            recovery_tau_ms: 25.0,
            bias_asymmetry_ratio: 0.0,
            window_gain_db: Vec::new(),
        };
    }

    let mut odd_rms_vec = Vec::with_capacity(num_win);
    let mut max_bias_ratio = 0.0_f64;
    let inv_n = 2.0 / (PUMP_PROBE_COMB_PERIOD as f64);
    for w in 0..num_win {
        let slice = &probe_wet[w * PUMP_PROBE_COMB_PERIOD..(w + 1) * PUMP_PROBE_COMB_PERIOD];
        let spec = rfft_bins(slice);
        let odd_e: f64 = odd_bins
            .iter()
            .map(|&b| (spec[b].abs() * inv_n).powi(2))
            .sum();
        let even_e: f64 = even_bins
            .iter()
            .map(|&b| (spec[b].abs() * inv_n).powi(2))
            .sum();
        let odd_rms = (0.5 * odd_e).sqrt().max(1.0e-12);
        let even_rms = (0.5 * even_e).sqrt();
        max_bias_ratio = max_bias_ratio.max(even_rms / odd_rms);
        odd_rms_vec.push(odd_rms);
    }

    let ref_rms = steady_ref_odd_rms
        .filter(|&r| r.is_finite() && r > 1.0e-9)
        .unwrap_or_else(|| *odd_rms_vec.last().unwrap_or(&1.0e-9));
    let window_gain_db: Vec<f32> = odd_rms_vec
        .iter()
        .map(|&r| (20.0 * (r / ref_rms.max(1.0e-12)).log10()) as f32)
        .collect();
    let g0 = f64::from(window_gain_db[0]);
    let trigger_ep2 = g0.abs() > BLOOM_EPISODE2_TRIGGER_DB;

    // Estimate 1/e recovery time constant across the 10.667 ms windows
    let dt_ms = 1_000.0 * (PUMP_PROBE_COMB_PERIOD as f64) / 48_000.0;
    let mut tau_ms = 35.0_f64;
    if g0.abs() > 0.05 && window_gain_db.len() >= 2 {
        let target_mag = g0.abs() * (-1.0_f64).exp();
        for (idx, &gdb) in window_gain_db.iter().enumerate().skip(1) {
            if f64::from(gdb).abs() <= target_mag {
                tau_ms = ((idx as f64) * dt_ms).clamp(5.0, 2_500.0);
                break;
            }
            if idx + 1 == window_gain_db.len() {
                let ratio = (f64::from(gdb).abs() / g0.abs()).clamp(0.05, 0.98);
                tau_ms = (-(idx as f64) * dt_ms / ratio.ln()).clamp(5.0, 4_000.0);
            }
        }
    }

    BloomEpisodeTelemetry {
        initial_delta_gain_db: g0,
        trigger_episode_2: trigger_ep2,
        recovery_tau_ms: tau_ms,
        bias_asymmetry_ratio: max_bias_ratio,
        window_gain_db,
    }
}

/// Encodes `12,000` frames (`250 ms` at `48 kHz`) of mono (`24,000 B`) or stereo (`48,000 B`)
/// audio into RNE `fp16` bytes for `INPC` or `STMC` capsules (`r[container.proj.stmc]`).
#[must_use]
pub fn encode_coreset_12k_fp16(ch_l: &[f32], ch_r: Option<&[f32]>) -> Vec<u8> {
    let is_stereo = ch_r.is_some();
    let mut out = Vec::with_capacity(CORESET_FP16_FRAMES_PER_STEM * if is_stereo { 4 } else { 2 });
    for i in 0..CORESET_FP16_FRAMES_PER_STEM {
        let sl = ch_l
            .get(i)
            .copied()
            .filter(|v| v.is_finite())
            .unwrap_or(0.0)
            .clamp(-65_500.0, 65_500.0);
        out.extend_from_slice(&f32_to_fp16_bits(sl).to_le_bytes());
        if let Some(r_slice) = ch_r {
            let sr = r_slice
                .get(i)
                .copied()
                .filter(|v| v.is_finite())
                .unwrap_or(0.0)
                .clamp(-65_500.0, 65_500.0);
            out.extend_from_slice(&f32_to_fp16_bits(sr).to_le_bytes());
        }
    }
    out
}

/// Computes the `88 B` `uplo='U'` 4-band PSD summary (`10 x f64` `uplo='U'` + `1 x f64` total power)
/// and `4,096 B` (`1,024 x f32`) spectral/ballistic summary block for `STMC` capsules.
#[must_use]
pub fn compute_stmc_psd_88_and_spectral_4096(
    wet_l: &[f32],
    wet_r: Option<&[f32]>,
) -> ([u8; 88], [u8; 4_096]) {
    let mut win = [0.0_f32; CYCLIC_WINDOW_SAMPLES];
    for (i, dst) in win.iter_mut().enumerate() {
        let l = wet_l
            .get(i)
            .copied()
            .filter(|v| v.is_finite())
            .unwrap_or(0.0);
        let r = wet_r
            .and_then(|rs| rs.get(i).copied())
            .filter(|v| v.is_finite())
            .unwrap_or(l);
        *dst = 0.5 * (l + r);
    }
    let spec = rfft_bins(&win);
    let inv_n = 2.0 / (CYCLIC_WINDOW_SAMPLES as f64);

    let mut band_vec = [0.0_f64; 4];
    let mut total_power = 0.0_f64;
    for (k, c) in spec
        .iter()
        .enumerate()
        .skip(1)
        .take(CYCLIC_WINDOW_SAMPLES / 2 - 1)
    {
        let p = c.norm_sqr() * inv_n * inv_n;
        total_power += p;
        let s = (k as f64) / ((CYCLIC_WINDOW_SAMPLES / 2) as f64);
        let (w_lo, w_mid, w_hi) = bezout_three_band_weights(s);
        band_vec[0] += w_lo * p;
        band_vec[1] += 0.5 * w_mid * p;
        band_vec[2] += 0.5 * w_mid * p;
        band_vec[3] += w_hi * p;
    }

    // 4x4 uplo='U' (10 f64s = 80 B) + total_power (8 B) = 88 B
    let mut psd_88 = [0u8; 88];
    let mut cursor = 0usize;
    for j in 0..4 {
        for i in 0..=j {
            let val = band_vec[i].sqrt() * band_vec[j].sqrt();
            psd_88[cursor..cursor + 8].copy_from_slice(&val.to_le_bytes());
            cursor += 8;
        }
    }
    psd_88[80..88].copy_from_slice(&total_power.to_le_bytes());

    // 4,096 B = 1,024 f32 bins (downsampled 2:1 magnitude spectrum from 2,048 positive bins)
    let mut spec_4096 = [0u8; 4_096];
    for idx in 0..1_024 {
        let b0 = idx * 2;
        let b1 = (b0 + 1).min(spec.len() - 1);
        let mag = (0.5 * (spec[b0].abs() + spec[b1].abs()) * inv_n) as f32;
        let off = idx * 4;
        spec_4096[off..off + 4].copy_from_slice(&mag.to_le_bytes());
    }

    (psd_88, spec_4096)
}

/// Synthesizes companion Right (`in_2`) and Return/Aux (`in_3`) `48 kHz` channels for `Theorem 4.5`
/// multi-port / multi-channel topology probing (`2->2`, `2->1`, `4CM`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn synthesize_topology_companion_48k(
    topology_mode: TopologyProbeMode,
    valid_frames: usize,
    disambiguated_2to1: Option<Disambiguated2To1Mode>,
    dry_l: &mut Vec<f32>,
    bin_amps_r: &[(usize, f64)],
    pedestal_bin_r: Option<usize>,
    ped_amp: f64,
    target_carrier_rms: f64,
    ceiling: f64,
    carrier_tail_r: &mut [f32; HANN_CROSSFADE_SAMPLES],
    phases_r: &mut Vec<f64>,
) -> (Option<Vec<f32>>, Option<Vec<f32>>) {
    let mut dry_r = None;
    let mut aux_l = None;
    match topology_mode {
        TopologyProbeMode::TrueStereo2To2 => match valid_frames {
            0 => dry_r = Some(dry_l.clone()),
            1 => dry_r = Some(dry_l.iter().map(|&x| -x).collect()),
            2 => {
                let witness_r = synthesize_cyclic_comb_frame_with_hint(
                    bin_amps_r,
                    None,
                    0.0,
                    0.25 * target_carrier_rms,
                    0.25 * ceiling,
                    carrier_tail_r,
                    Some(phases_r),
                );
                *carrier_tail_r = witness_r.carrier_tail_64;
                *phases_r = witness_r.phases;
                dry_r = Some(witness_r.samples);
            }
            _ => {
                let comb_r = synthesize_cyclic_comb_frame_with_hint(
                    bin_amps_r,
                    pedestal_bin_r,
                    ped_amp,
                    target_carrier_rms,
                    ceiling,
                    carrier_tail_r,
                    Some(phases_r),
                );
                *carrier_tail_r = comb_r.carrier_tail_64;
                *phases_r = comb_r.phases;
                dry_r = Some(comb_r.samples);
            }
        },
        TopologyProbeMode::DualInToMono2To1 => {
            if valid_frames == 0 {
                let am_sig: Vec<f32> = (0..ADAPTIVE_FRAME_SAMPLES)
                    .map(|n| {
                        let w0 = 2.0 * PI / (CYCLIC_WINDOW_SAMPLES as f64);
                        let env = 0.5 * (1.0 - (2.0 * w0 * (n as f64)).cos());
                        (0.65 * ceiling * env * (127.0 * w0 * (n as f64)).sin()) as f32
                    })
                    .collect();
                dry_r = Some(am_sig.clone());
                aux_l = Some(am_sig);
            } else if disambiguated_2to1 == Some(Disambiguated2To1Mode::ExternalAuxKeySidechain) {
                let sc_key = synthesize_a2_dynamic_staircase(ADAPTIVE_FRAME_SAMPLES, ceiling);
                dry_r = Some(sc_key.clone());
                aux_l = Some(sc_key);
            } else {
                let comb_r = synthesize_cyclic_comb_frame_with_hint(
                    bin_amps_r,
                    pedestal_bin_r,
                    ped_amp,
                    target_carrier_rms,
                    ceiling,
                    carrier_tail_r,
                    Some(phases_r),
                );
                *carrier_tail_r = comb_r.carrier_tail_64;
                *phases_r = comb_r.phases;
                dry_r = Some(comb_r.samples);
            }
        }
        TopologyProbeMode::FourCableMethod4Cm => {
            let ret_comb = synthesize_cyclic_comb_frame_with_hint(
                bin_amps_r,
                pedestal_bin_r,
                ped_amp,
                target_carrier_rms,
                ceiling,
                carrier_tail_r,
                Some(phases_r),
            );
            *carrier_tail_r = ret_comb.carrier_tail_64;
            *phases_r = ret_comb.phases;
            aux_l = Some(ret_comb.samples);
            if valid_frames == 1 {
                *dry_l = synthesize_active_primitive_frame(
                    ActivePrimitiveKind::P2MultiDriveBurst,
                    0,
                    false,
                    ceiling,
                );
            }
        }
        TopologyProbeMode::Mono1To1
        | TopologyProbeMode::MonoToStereo1To2
        | TopologyProbeMode::WetDryWet1To3 => {}
    }
    (dry_r, aux_l)
}
