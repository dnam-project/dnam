//! Deterministic 3-pulse asymmetric alignment burst, 2-step calibration probe synthesis,
//! probe system-ID analyzer (`TargetExcitationProfile`), and tailored dry excitation generation.

use std::f32::consts::{PI, SQRT_2, TAU};

use ndarray::ArrayView1;
use realfft::RealFftPlanner;
use serde::{Deserialize, Serialize};

/// Duration of the 3-pulse asymmetric chirp burst in seconds (`0.35s`).
pub const ALIGNMENT_PULSE_DURATION_SEC: f32 = 0.35;
/// Duration of the post-pulse nonlinear settling silence in seconds (`0.10s`).
pub const ALIGNMENT_SETTLE_SILENCE_SEC: f32 = 0.10;
/// Total duration of the leading alignment header (`0.45s`).
pub const ALIGNMENT_HEADER_TOTAL_SEC: f32 =
    ALIGNMENT_PULSE_DURATION_SEC + ALIGNMENT_SETTLE_SILENCE_SEC;
/// Duration of the baseline trailing silent window for delay/reverb tail decay and noise-floor estimation (`3.00s`).
pub const TRAILING_NOISE_WINDOW_SEC: f32 = 3.00;
/// Maximum additional live recording tail duration in seconds when a long reverb/delay tail (`> 10s`) is still decaying (`15.00s`).
pub const MAX_DYNAMIC_TAIL_EXTRA_SEC: f32 = 15.00;
/// Canonical duration in seconds for hardware stem excitation (`30.0s` = `1,440,000` samples @ `48 kHz`).
pub const DEFAULT_HARDWARE_EXCITATION_SEC: usize = 30;
/// Canonical duration in seconds for Step-1/Step-2 hardware calibration probes (`6.0s` = `288,000` samples @ `48 kHz`).
pub const DEFAULT_HARDWARE_PROBE_SEC: usize = 6;

/// Probed target device system-identification profile used to synthesize an optimal dry excitation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TargetExcitationProfile {
    /// Measured nonlinear soft-knee onset amplitude in linear full-scale `[0.01, 0.95]`.
    pub knee_amplitude_lin: f32,
    /// Measured hard saturation ceiling in linear full-scale `[0.05, 1.0]`.
    pub saturation_ceiling_lin: f32,
    /// Whether an active delay or reverb tail was detected in the silent listening gap.
    pub has_delay_or_reverb_tail: bool,
    /// Estimated primary echo delay time in milliseconds.
    pub estimated_max_delay_ms: f32,
    /// Estimated reverb/delay tail $T_{60}$ decay time in milliseconds.
    pub estimated_tail_t60_ms: f32,
    /// Detected LFO modulation rate in Hz (`0.0` if time-invariant).
    pub detected_lfo_hz: f32,
    /// Dominant resonant peak frequency in Hz.
    pub dominant_resonance_hz: f32,
}

impl Default for TargetExcitationProfile {
    fn default() -> Self {
        Self {
            knee_amplitude_lin: 0.15,
            saturation_ceiling_lin: 0.85,
            has_delay_or_reverb_tail: false,
            estimated_max_delay_ms: 0.0,
            estimated_tail_t60_ms: 0.0,
            detected_lfo_hz: 0.0,
            dominant_resonance_hz: 800.0,
        }
    }
}

/// Synthesizes the 3-pulse asymmetric alignment burst (`800 Hz -> 6 kHz` Hann-tapered log chirps
/// with polarities `+0.90, -0.90, +0.675` followed by settling silence).
pub fn synth_alignment_pulses(sample_rate_hz: u32, header_samples: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; header_samples];
    if header_samples == 0 {
        return out;
    }
    let fs = sample_rate_hz.max(8_000) as f32;
    let chirp_len = ((0.012 * fs).round() as usize)
        .max(16)
        .min((header_samples / 5).max(8));
    let f0 = 800.0_f32;
    let f1 = (0.4 * fs).min(6_000.0);
    let dur_c = (chirp_len as f32) / fs;
    let k_rate = (f1 / f0).ln() / dur_c.max(1e-6);

    let pulses = [
        (header_samples / 8, 0.90_f32),
        ((header_samples * 3) / 10, -0.90_f32),
        ((header_samples * 11) / 20, 0.675_f32),
    ];
    for &(idx0, gain) in &pulses {
        for k in 0..chirp_len {
            if idx0 + k < header_samples {
                let t = (k as f32) / fs;
                let phase = TAU * f0 * (((k_rate * t).exp() - 1.0) / k_rate);
                let window = 0.5 * (1.0 - (TAU * (k as f32) / (chirp_len.max(1) as f32)).cos());
                out[idx0 + k] += gain * window * phase.sin();
            }
        }
    }
    out
}

/// Generates a deterministic 3-pulse + log-amplitude ramp + sweep + plosive transient calibration probe.
pub fn generate_calibration_probe(sample_rate_hz: u32, num_samples: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; num_samples];
    if num_samples == 0 {
        return out;
    }
    let fs = sample_rate_hz.max(1) as f32;

    // Leading silent lead-in (capped at 100 ms) + 3-pulse asymmetric burst (capped at 0.45s)
    let max_lead = ((0.10 * fs).round() as usize).max(1);
    let max_header = ((ALIGNMENT_HEADER_TOTAL_SEC * fs).round() as usize).max(16);
    let lead_silence = (num_samples / 16).min(max_lead).max(1);
    let raw_header_len = ((num_samples * 6) / 16).min(max_header).max(1);
    let header_end = (lead_silence + raw_header_len).min(num_samples);
    let header_len = header_end - lead_silence;
    let pulses = synth_alignment_pulses(sample_rate_hz, header_len);
    out[lead_silence..header_end].copy_from_slice(&pulses);

    let tail_silence = (num_samples / 16).max(1);
    let body_start = header_end;
    let body_end = num_samples.saturating_sub(tail_silence).max(body_start);
    let body_len = body_end - body_start;
    if body_len == 0 {
        return out;
    }

    // Part A (0..35% of body): Log-amplitude nonlinearity ramp (-40 dBFS -> 0 dBFS) on 3-decade triad (40 Hz + 330 Hz + 2640 Hz)
    let a_end = (body_len * 35) / 100;
    for i in 0..a_end {
        let frac = (i as f32) / (a_end.max(1) as f32);
        let amp = 10.0_f32.powf(-2.0 + 2.0 * frac);
        let t = (i as f32) / fs;
        let triad = 0.45 * (TAU * 40.0 * t).sin()
            + 0.35 * (TAU * 330.0 * t).sin()
            + 0.20 * (TAU * 2_640.0 * t).sin();
        out[body_start + i] = amp * triad;
    }

    // Part B (35..70% of body): Steady harmonic carrier + full-spectrum log sweep (20 Hz -> 20 kHz)
    let b_end = ((body_len * 70) / 100).min(
        ((num_samples * 78) / 100)
            .saturating_sub(body_start)
            .max(a_end),
    );
    let nb = b_end.saturating_sub(a_end).max(1);
    let dur_b = ((nb as f32) / fs).max(1e-4);
    let f0 = 20.0_f32;
    let f1 = (0.45 * fs).min(20_000.0);
    let k_b = (f1 / f0).ln() / dur_b;
    for j in 0..(b_end.saturating_sub(a_end)) {
        let t = (j as f32) / fs;
        let phi = TAU * f0 * (((k_b * t).exp() - 1.0) / k_b);
        let carrier = 0.35 * ((TAU * 220.0 * t).sin() + 0.5 * (TAU * 880.0 * t).sin());
        out[body_start + a_end + j] = carrier + 0.35 * phi.sin();
    }

    // Part C (70..100% of body): Plosive transient burst before 80% + silent echo listening gap (82..95%)
    let nc = body_len.saturating_sub(b_end);
    let max_burst_before_gap = ((num_samples * 80) / 100).saturating_sub(body_start + b_end);
    let burst_samples = ((0.015 * fs).round() as usize)
        .max(8)
        .min(nc / 2)
        .min(max_burst_before_gap);
    for j in 0..burst_samples {
        let t = (j as f32) / fs;
        let win = 0.5 * (1.0 - (TAU * (j as f32) / (burst_samples.max(1) as f32)).cos());
        let sig = 0.5 * (TAU * 500.0 * t).sin() + 0.5 * (TAU * 1_800.0 * t).sin();
        out[body_start + b_end + j] = 0.92 * win * sig;
    }

    out
}

/// Analyzes a dry calibration probe and aligned wet return to extract [`TargetExcitationProfile`].
pub fn analyze_probe_for_excitation(
    dry_probe: &[f32],
    aligned_wet: &[f32],
    sample_rate_hz: u32,
) -> TargetExcitationProfile {
    let n = dry_probe.len().min(aligned_wet.len());
    if n < 64 {
        return TargetExcitationProfile::default();
    }
    let fs = sample_rate_hz.max(8_000) as f32;
    let wet_view = ArrayView1::from(&aligned_wet[..n]);
    let wet_peak = wet_view.fold(1e-6_f32, |acc, &x| acc.max(x.abs()));

    // 1. Windowed RMS envelope curve (10 ms windows) to estimate soft-knee onset & input saturation/clipping ceiling
    let win = ((0.010 * fs).round() as usize).clamp(16, (n / 8).max(16));
    let mut active_pairs: Vec<(f32, f32, f32)> = Vec::new();
    for (d_chunk, w_chunk) in dry_probe[..n].chunks(win).zip(aligned_wet[..n].chunks(win)) {
        if d_chunk.is_empty() {
            continue;
        }
        let inv_len = 1.0 / (d_chunk.len() as f32);
        let d_rms = (d_chunk.iter().map(|&x| x * x).sum::<f32>() * inv_len).sqrt();
        let w_rms = (w_chunk.iter().map(|&x| x * x).sum::<f32>() * inv_len).sqrt();
        let w_max_win = w_chunk.iter().map(|&x| x.abs()).fold(0.0_f32, f32::max);
        if d_rms > 0.015 {
            active_pairs.push((d_rms, w_rms, w_max_win));
        }
    }

    let mut knee_amplitude_lin = 0.15_f32;
    let mut saturation_ceiling_lin = 0.85_f32;
    if active_pairs.len() >= 4 {
        active_pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let q_len = (active_pairs.len() / 4).max(1);
        let mut small_gains: Vec<f32> = active_pairs[..q_len]
            .iter()
            .map(|&(d, w, _)| w / d.max(1e-6))
            .collect();
        small_gains.sort_by(f32::total_cmp);
        let small_sig_gain = small_gains[small_gains.len() / 2].max(1e-6);

        let sqrt2 = 2.0_f32.sqrt();
        for &(d_rms, w_rms, _) in &active_pairs {
            if (w_rms / d_rms.max(1e-6)) < 0.85 * small_sig_gain {
                knee_amplitude_lin = (d_rms * sqrt2).clamp(0.02, 0.90);
                break;
            }
        }

        let w_rms_max = active_pairs
            .iter()
            .map(|&(_, w, _)| w)
            .fold(1e-6_f32, f32::max);
        for &(d_rms, w_rms, w_max_win) in &active_pairs {
            if w_rms >= 0.92 * w_rms_max || w_max_win >= 0.994 {
                saturation_ceiling_lin =
                    (d_rms * sqrt2).clamp((knee_amplitude_lin + 0.05).max(0.10), 1.0);
                break;
            }
        }
    }

    // 2. Check Part C echo listening gap for delay/reverb energy and T60 decay rate
    let gap_start = (n * 82) / 100;
    let gap_end = (n * 95) / 100;
    let mut gap_rms = 0.0_f32;
    let mut peak_gap_idx = gap_start;
    let mut peak_gap_val = 0.0_f32;
    if gap_end > gap_start {
        for (i, &w) in aligned_wet[gap_start..gap_end].iter().enumerate() {
            let a = w.abs();
            gap_rms += a * a;
            if a > peak_gap_val {
                peak_gap_val = a;
                peak_gap_idx = gap_start + i;
            }
        }
        gap_rms = (gap_rms / ((gap_end - gap_start) as f32)).sqrt();
    }
    let has_delay_or_reverb_tail = gap_rms > 0.015 * wet_peak || peak_gap_val > 0.02;
    let estimated_max_delay_ms = if has_delay_or_reverb_tail {
        (((peak_gap_idx.saturating_sub((n * 80) / 100)) as f32) * 1_000.0 / fs).clamp(15.0, 2_000.0)
    } else {
        0.0
    };
    let estimated_tail_t60_ms = if has_delay_or_reverb_tail && gap_end > gap_start + 8 {
        let quiet_slice = &aligned_wet[gap_start..gap_end];
        let half = (quiet_slice.len() / 2).max(1);
        let rms_early = (quiet_slice[..half].iter().map(|&x| x * x).sum::<f32>() / (half as f32))
            .sqrt()
            .max(1e-6);
        let rms_late = (quiet_slice[half..].iter().map(|&x| x * x).sum::<f32>()
            / (quiet_slice[half..].len().max(1) as f32))
            .sqrt()
            .max(1e-7);
        let dt_ms = 1_000.0 * (half as f32) / fs;
        let decay_db = (20.0 * (rms_early / rms_late).max(1.02).log10()).max(0.4);
        (dt_ms * (60.0 / decay_db))
            .max(estimated_max_delay_ms * 3.5)
            .clamp(80.0, 20_000.0)
    } else {
        0.0
    };

    // 3. Estimate dominant transfer-function resonance H(f) = |W(f)| / (|D(f)| + eps) in Part B via realfft
    let b_start = (n * 45) / 100;
    let b_end = (n * 78) / 100;
    let b_len = b_end.saturating_sub(b_start);
    let mut dominant_resonance_hz = 800.0_f32;
    if b_len >= 64 {
        let n_fft = b_len.next_power_of_two();
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(n_fft);
        let mut dry_seg = vec![0.0_f32; n_fft];
        let mut wet_seg = vec![0.0_f32; n_fft];
        for i in 0..b_len {
            let win_v = 0.5 * (1.0 - (TAU * (i as f32) / (b_len as f32)).cos());
            dry_seg[i] = dry_probe[b_start + i] * win_v;
            wet_seg[i] = aligned_wet[b_start + i] * win_v;
        }
        let mut d_spec = r2c.make_output_vec();
        let mut w_spec = r2c.make_output_vec();
        let _ = r2c.process(&mut dry_seg, &mut d_spec);
        let _ = r2c.process(&mut wet_seg, &mut w_spec);

        let max_d_mag = d_spec.iter().map(|c| c.norm()).fold(1e-6_f32, f32::max);
        let eps = 0.02 * max_d_mag;
        let bin_hz = fs / (n_fft as f32);
        let mut best_gain = 0.0_f32;
        for (k, (w_bin, d_bin)) in w_spec.iter().zip(d_spec.iter()).enumerate() {
            let freq = (k as f32) * bin_hz;
            if (25.0..=16_000.0).contains(&freq) {
                let gain = w_bin.norm() / (d_bin.norm() + eps);
                if gain > best_gain {
                    best_gain = gain;
                    dominant_resonance_hz = freq;
                }
            }
        }
    }

    // 4. Detect coherent LFO envelope modulation in [0.25 Hz, 12.0 Hz] via decimated envelope realfft
    let mut detected_lfo_hz = 0.0_f32;
    let decim = ((fs / 200.0).round() as usize).max(1);
    let env_samples: Vec<f32> = aligned_wet[..n]
        .chunks(decim)
        .map(|c| c.iter().map(|&x| x.abs()).sum::<f32>() / (c.len().max(1) as f32))
        .collect();
    if env_samples.len() >= 32 {
        let env_fs = fs / (decim as f32);
        let env_view = ArrayView1::from(&env_samples);
        let env_mean = env_view.mean().unwrap_or(0.0);
        let env_fft_len = env_samples.len().next_power_of_two();
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(env_fft_len);
        let mut env_buf = vec![0.0_f32; env_fft_len];
        for (i, &v) in env_samples.iter().enumerate() {
            env_buf[i] = v - env_mean;
        }
        let mut env_spec = r2c.make_output_vec();
        let _ = r2c.process(&mut env_buf, &mut env_spec);
        let bin_hz = env_fs / (env_fft_len as f32);
        let mut max_lfo_mag = 0.0_f32;
        let mut total_lfo_mag = 1e-9_f32;
        let mut best_lfo_freq = 0.0_f32;
        for (k, bin) in env_spec.iter().enumerate().skip(1) {
            let freq = (k as f32) * bin_hz;
            let mag = bin.norm();
            total_lfo_mag += mag;
            if (0.25..=12.0).contains(&freq) && mag > max_lfo_mag {
                max_lfo_mag = mag;
                best_lfo_freq = freq;
            }
        }
        if max_lfo_mag > 0.35 * total_lfo_mag {
            detected_lfo_hz = best_lfo_freq;
        }
    }

    TargetExcitationProfile {
        knee_amplitude_lin,
        saturation_ceiling_lin,
        has_delay_or_reverb_tail,
        estimated_max_delay_ms,
        estimated_tail_t60_ms,
        detected_lfo_hz,
        dominant_resonance_hz,
    }
}

/// Generates a universal 5-section multi-band dry excitation signal (`30.0s` = `1,440,000` samples
/// at `48 kHz` via [`DEFAULT_HARDWARE_EXCITATION_SEC`]) tailored to an optional
/// [`TargetExcitationProfile`] that simultaneously excites nonlinear amplifiers, preamps,
/// overdrives, distortions, fuzzes, multi-band crossovers, compressors, envelope/resonant filters,
/// LFO modulation effects, delays, reverbs, pitch dividers, and speaker cabinets—or complete
/// multi-effect signal chains—without relying on control-name heuristics or dynamic length branching.
///
/// Structure (`30.0s` @ `48 kHz`):
/// 1. **Leading Header (`0.00s – 0.55s`):** `100 ms` silence + `0.45s` 3-pulse asymmetric chirp
///    alignment burst and settling silence ([`synth_alignment_pulses`]).
/// 2. **Core 5-Section Excitation Body (`0.55s – 27.00s`, `26.45s`):**
///    - **Section 1 (`0..20%`, `~5.29s`):** Dual-regime logarithmic sine sweeps (`20 Hz -> 20 kHz`)
///      comprising a clean/Zone-A sweep (`knee -> sat`, `0..45%`), an inter-sweep settling gap
///      (`45..52%`), and a hot/Zone-B sweep (`sat -> 0.98`, `52..100%`).
///    - **Section 2 (`20..48%`, `~7.41s`):** 12-step multi-register ballistic attack/release &
///      staccato re-trigger ladder (`-42 dBFS -> 0.98`, `~617 ms/step`) across low/mid/high
///      registers (`41.2 Hz E1 .. 659.3 Hz E5`) with simultaneous sub/mid/high crossover anchors
///      (`65 Hz`, `330 Hz`, `1,320 Hz`), a 12-tone Schroeder harmonic cluster (`120 Hz .. 7.2 kHz`),
///      rapid double-pluck re-triggers on top velocity steps, and `-46 dBFS` release-tracking probes.
///    - **Section 3 (`48..70%`, `~5.82s`):** 16-burst polyphonic dyad/triad saturation and
///      intermodulation matrix (`sat -> 0.98`) spanning octaves, fifths, fourths, major/minor thirds,
///      and tritones across `41.2 Hz .. 784 Hz`.
///    - **Section 4 (`70..88%`, `~4.76s`):** Strictly constant-envelope harmonic + multitone sustain
///      (`0..48%`, $\dot{e}_{\text{in}}[n] \equiv 0$, isolating autonomous `0.25–10 Hz` LFOs from
///      envelope followers) followed by a dual-regime multitone/CCIF dynamic hysteresis swell
///      (`48..100%`, `-48 dBFS -> knee -> 0.98 -> knee -> -48 dBFS`).
///    - **Section 5 (`88..100%`, `~3.17s`):** Broadband high-crest harmonic cluster (`0..60%`,
///      `sat -> 0.98`) + isolated full-scale (`0.98`) chord/transient tail probe (`60..68%`) +
///      pre-tail decay silence (`68..100%`, `~1.02s`).
/// 3. **Trailing Decay & Silent Noise-Floor Window (`27.00s – 30.00s`, `3.00s`):** Baseline
///    uninterrupted decay window (`~4.02s` total after the Section 5B impulse); if a stem's wet
///    reverb/delay return is still ringing above the noise floor at `30.0s`, live hardware capture
///    (`cpal_play_and_record`) automatically continues recording up to [`MAX_DYNAMIC_TAIL_EXTRA_SEC`]
///    without inflating the dry `input.wav` file on disk.
pub fn generate_excitation_signal_with_profile(
    sample_rate_hz: u32,
    num_samples: usize,
    profile: Option<&TargetExcitationProfile>,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; num_samples];
    if num_samples == 0 {
        return out;
    }
    let fs = sample_rate_hz.max(8_000) as f32;
    let default_prof = TargetExcitationProfile::default();
    let prof = profile.unwrap_or(&default_prof);

    let max_lead = ((0.10 * fs).round() as usize).max(1);
    let max_header = ((ALIGNMENT_HEADER_TOTAL_SEC * fs).round() as usize).max(16);
    let max_base_tail = ((TRAILING_NOISE_WINDOW_SEC * fs).round() as usize).max(1);

    let lead_silence = (num_samples / 16).min(max_lead).max(1);
    let raw_header_len = ((num_samples * 6) / 16).min(max_header).max(1);
    let header_end = (lead_silence + raw_header_len).min(num_samples);
    let header_len = header_end - lead_silence;
    let pulses = synth_alignment_pulses(sample_rate_hz, header_len);
    out[lead_silence..header_end].copy_from_slice(&pulses);

    let tail_silence = (num_samples / 10).min(max_base_tail).max(1);
    let body_start = header_end;
    let body_end = num_samples.saturating_sub(tail_silence).max(body_start);
    let n_body = body_end - body_start;
    if n_body == 0 {
        return out;
    }

    let knee = prof.knee_amplitude_lin.clamp(0.02, 0.90);
    let sat = prof
        .saturation_ceiling_lin
        .clamp((knee + 0.05).min(0.95), 0.98);
    let res_hz = prof.dominant_resonance_hz.clamp(80.0, 8_000.0);
    let nyq_guard = 0.45 * fs;

    let mut body = vec![0.0_f32; n_body];
    let s1_end = (n_body * 20) / 100;
    let s2_end = (n_body * 48) / 100;
    let s3_end = (n_body * 70) / 100;
    let s4_end = (n_body * 88) / 100;

    // Section 1 (0..20%): Dual-regime logarithmic sweeps (20 Hz -> 20 kHz) with inter-sweep settling gap
    if s1_end > 0 {
        let sw1_end = (s1_end * 45) / 100;
        let sw2_start = (s1_end * 52) / 100;
        let f_start = 20.0_f32;
        let f_stop = nyq_guard.min(20_000.0);

        if sw1_end > 0 {
            let d1a = ((sw1_end as f32) / fs).max(1e-4);
            let k_sw1 = (f_stop / f_start).ln() / d1a;
            for (i, sample) in body.iter_mut().enumerate().take(sw1_end) {
                let t = (i as f32) / fs;
                let u = (i as f32) / (sw1_end.max(1) as f32);
                let env = knee + (sat - knee) * (0.5 - 0.5 * (TAU * u).cos());
                let phi = TAU * f_start * (((k_sw1 * t).exp() - 1.0) / k_sw1);
                *sample = env * phi.sin();
            }
        }
        if s1_end > sw2_start {
            let sw2_len = s1_end - sw2_start;
            let d1b = ((sw2_len as f32) / fs).max(1e-4);
            let k_sw2 = (f_stop / f_start).ln() / d1b;
            let f_res = res_hz.clamp(80.0, 2_400.0).min(nyq_guard);
            for j in 0..sw2_len {
                let t = (j as f32) / fs;
                let u = (j as f32) / (sw2_len.max(1) as f32);
                let env = sat + (0.98 - sat) * (0.5 - 0.5 * (TAU * u).cos());
                let phi = TAU * f_start * (((k_sw2 * t).exp() - 1.0) / k_sw2);
                let mut val = env * phi.sin();
                if prof.detected_lfo_hz > 0.15 {
                    let lfo_carrier = knee
                        * (0.65 * (TAU * f_res * t).sin() + 0.35 * (TAU * (1.5 * f_res) * t).sin());
                    val = 0.60 * val + 0.40 * lfo_carrier;
                }
                body[sw2_start + j] = val;
            }
        }
    }

    // Section 2 (20..48%): 12-step multi-register ballistic attack/release & staccato re-trigger ladder (-42 dBFS .. 0.98)
    let n2 = s2_end.saturating_sub(s1_end);
    if n2 > 0 {
        let notes_hz: [f32; 12] = [
            41.20, 55.00, 73.42, 82.41, 110.00, 146.83, 196.00, 246.94, 329.63, 440.00, 587.33,
            659.25,
        ];
        let step_levels: [f32; 12] = [
            0.008,
            0.020,
            0.045,
            (0.55 * knee).max(0.080),
            knee,
            0.35 * sat + 0.65 * knee,
            0.70 * sat + 0.30 * knee,
            sat,
            0.35 * 0.98 + 0.65 * sat,
            0.70 * 0.98 + 0.30 * sat,
            0.92,
            0.98,
        ];
        let n_steps = step_levels.len();
        let stride = (n2 / n_steps).max(1);
        let burst_len = if prof.has_delay_or_reverb_tail {
            ((stride * 52) / 100).max(1)
        } else {
            ((stride * 62) / 100).max(1)
        };
        let f_hi_max = nyq_guard.clamp(240.0, 7_200.0);
        let n_harm = 12_usize;
        let harm_log_ratio = (f_hi_max / 120.0_f32).ln();

        for (b_idx, &vel) in step_levels.iter().enumerate() {
            let st = b_idx * stride;
            if st >= n2 {
                break;
            }
            let en = (st + stride).min(n2);
            let span = en - st;
            let act = burst_len.min(span);
            let plat_len = (act * 60) / 100;
            let f_root = notes_hz[b_idx % notes_hz.len()].min(nyq_guard * 0.45);
            let f_fifth = (f_root * 1.498_307_1).min(nyq_guard);
            let f_oct = (f_root * 2.0).min(nyq_guard);
            let f_sub_anchor = 65.41_f32.min(nyq_guard);
            let f_mid_anchor = 329.63_f32.min(nyq_guard);
            let f_hi_anchor = 1_318.5_f32.min(nyq_guard);

            for k in 0..span {
                let tb = (k as f32) / fs;
                let reg_triad = 0.45 * (TAU * f_root * tb).sin()
                    + 0.25 * (TAU * f_fifth * tb + 0.3).sin()
                    + 0.15 * (TAU * f_oct * tb + 0.7).sin();
                let xover_anchors = 0.06 * (TAU * f_sub_anchor * tb + 0.2).sin()
                    + 0.05 * (TAU * f_mid_anchor * tb + 0.9).sin()
                    + 0.04 * (TAU * f_hi_anchor * tb + 1.7).sin();
                let mut sch_sum = 0.0_f32;
                let mut sch_cnt = 0.0_f32;
                for h in 1..=n_harm {
                    let frac_h = ((h - 1) as f32) / ((n_harm - 1).max(1) as f32);
                    let fh = 120.0 * (frac_h * harm_log_ratio).exp();
                    if fh >= nyq_guard {
                        break;
                    }
                    let ph = PI * (h as f32) * ((h - 1) as f32) / (n_harm as f32);
                    sch_sum += (TAU * fh * tb + ph).cos();
                    sch_cnt += 1.0;
                }
                let sch_norm = if sch_cnt > 0.0 {
                    (sch_sum / sch_cnt.sqrt()).clamp(-1.0, 1.0)
                } else {
                    0.0
                };
                let carrier = 0.78 * (reg_triad + xover_anchors) + 0.22 * sch_norm;
                let env = if k < act {
                    // Top 4 velocity steps include a mid-plateau staccato re-trigger dip + 1.2 ms re-attack
                    let retrig_mod = if b_idx >= 8 && plat_len > 16 {
                        let half_p = plat_len / 2;
                        if k >= half_p.saturating_sub(8) && k < half_p {
                            0.18
                        } else if k >= half_p {
                            let t_re = ((k - half_p) as f32) / fs;
                            1.0 - 0.82 * (-t_re / 0.0012).exp()
                        } else {
                            1.0
                        }
                    } else {
                        1.0
                    };
                    let atk = 1.0 - (-tb / 0.0012).exp();
                    let rel_tail = if k < plat_len {
                        1.0
                    } else {
                        let t_dec = ((k - plat_len) as f32) / fs;
                        (-t_dec / 0.032).exp()
                    };
                    vel * atk * retrig_mod * rel_tail
                } else if !prof.has_delay_or_reverb_tail {
                    0.005
                } else {
                    0.0
                };
                body[s1_end + st + k] = env * carrier;
            }
        }
    }

    // Section 3 (48..70%): 16-burst polyphonic dyad/triad saturation & intermodulation matrix (sat .. 0.98)
    let n3 = s3_end.saturating_sub(s2_end);
    if n3 > 0 {
        let chord_roots_hz: [f32; 16] = [
            41.20, 55.00, 65.41, 73.42, 82.41, 98.00, 110.00, 130.81, 146.83, 164.81, 196.00,
            220.00, 261.63, 329.63, 440.00, 587.33,
        ];
        let interval_ratios: [(f32, f32); 8] = [
            (1.498_307_1, 2.000_000_0), // Power 5th + Octave
            (1.334_839_8, 2.000_000_0), // Perfect 4th + Octave
            (1.259_921_1, 1.498_307_1), // Major Triad (1, 3, 5)
            (1.189_207_1, 1.498_307_1), // Minor Triad (1, b3, 5)
            (SQRT_2, 2.000_000_0),      // Tritone IMD + Octave
            (1.498_307_1, 3.000_000_0), // 5th + 12th
            (2.000_000_0, 4.000_000_0), // Multi-Octave stack
            (1.259_921_1, 1.887_748_6), // Major 7th chord voicing
        ];
        let n_chords = chord_roots_hz.len();
        let c_stride = (n3 / n_chords).max(1);
        for (c_idx, &f_r0) in chord_roots_hz.iter().enumerate() {
            let st = c_idx * c_stride;
            if st >= n3 {
                break;
            }
            let en = (st + c_stride).min(n3);
            let span = en - st;
            let f1 = f_r0.min(nyq_guard * 0.45);
            let (r2, r3) = interval_ratios[c_idx % interval_ratios.len()];
            let f2 = (f1 * r2).min(nyq_guard);
            let f3 = (f1 * r3).min(nyq_guard);
            let f4 = (f1 * 4.0).min(nyq_guard);
            let f_pres = (f1 * 9.0).clamp(900.0, nyq_guard.min(6_500.0));
            let drive_amp = if c_idx % 2 == 0 {
                0.98
            } else {
                0.5 * (sat + 0.98)
            };

            for k in 0..span {
                let tc = (k as f32) / fs;
                let u = (k as f32) / (span.max(1) as f32);
                let atk = 1.0 - (-tc / 0.0010).exp();
                let sus_win = if u < 0.85 {
                    1.0 - 0.12 * u
                } else {
                    let r = (1.0 - u) / 0.15;
                    0.898 * r * r
                };
                let chord = 0.42 * (TAU * f1 * tc).sin()
                    + 0.30 * (TAU * f2 * tc + 0.4).sin()
                    + 0.18 * (TAU * f3 * tc + 1.1).sin()
                    + 0.12 * (TAU * f4 * tc + 1.9).sin()
                    + 0.08 * (TAU * f_pres * tc + 0.6).cos();
                body[s2_end + st + k] = drive_amp * atk * sus_win * chord;
            }
        }
    }

    // Section 4 (70..88%): Constant-envelope autonomous LFO sustain (0..48%) + Dual-regime multitone/CCIF swell (48..100%)
    let n4 = s4_end.saturating_sub(s3_end);
    if n4 > 0 {
        let lfo_sus_end = (n4 * 48) / 100;
        let swell_len = n4.saturating_sub(lfo_sus_end);
        let f_max_mt = nyq_guard.clamp(40.0, 20_000.0);
        let f_ccif = res_hz.clamp(120.0, (0.40 * fs).min(8_000.0));
        let f_air = (0.30 * fs).min(11_000.0);
        let n_tones = 24_usize;
        let log_ratio = (f_max_mt / 20.0_f32).ln();
        let mut multitone = vec![0.0_f32; n4];
        let mut max_mt = 1e-6_f32;
        for (j, mt_val) in multitone.iter_mut().enumerate() {
            let t4 = (j as f32) / fs;
            let mut sum_t = 0.0_f32;
            for k_tone in 1..=n_tones {
                let frac_k = ((k_tone - 1) as f32) / ((n_tones - 1) as f32);
                let fk = 20.0 * (frac_k * log_ratio).exp();
                if fk >= nyq_guard {
                    break;
                }
                let schroeder_phase =
                    PI * (k_tone as f32) * ((k_tone - 1) as f32) / (n_tones as f32);
                sum_t += (TAU * fk * t4 + schroeder_phase).cos();
            }
            *mt_val = sum_t;
            max_mt = max_mt.max(sum_t.abs());
        }

        // Part 4A: Strictly constant-amplitude harmonic + multitone sustain for autonomous LFO identification
        let const_env = (0.55 * (knee + sat)).clamp(0.25, 0.85);
        for (j, &mt_raw) in multitone.iter().enumerate().take(lfo_sus_end) {
            let t4 = (j as f32) / fs;
            let chord_6 = 0.30 * (TAU * 82.41 * t4).sin()
                + 0.22 * (TAU * 123.47 * t4 + 0.5).sin()
                + 0.18 * (TAU * 164.81 * t4 + 1.2).sin()
                + 0.14 * (TAU * 246.94 * t4 + 2.0).sin()
                + 0.10 * (TAU * 329.63 * t4 + 2.7).sin()
                + 0.06 * (TAU * f_ccif * t4 + 0.8).sin();
            let mt_norm = mt_raw / max_mt;
            body[s3_end + j] = const_env * (0.65 * chord_6 + 0.35 * mt_norm);
        }

        // Part 4B: Dual-regime dynamic hysteresis swell (-48 dBFS -> knee -> 0.98 -> knee -> -48 dBFS)
        if swell_len > 0 {
            let low_floor = 0.004_f32;
            let log_knee_ratio = (knee.max(low_floor) / low_floor).ln();
            for j in 0..swell_len {
                let idx = lfo_sus_end + j;
                let t4 = (idx as f32) / fs;
                let u = (j as f32) / (swell_len.max(1) as f32);
                let swell_env = if u < 0.15 {
                    let r = u / 0.15;
                    low_floor * (r * log_knee_ratio).exp()
                } else if u <= 0.85 {
                    let r = (u - 0.15) / 0.70;
                    let s_val = (PI * r).sin();
                    knee + (0.98 - knee) * (s_val * s_val)
                } else {
                    let r = ((1.0 - u) / 0.15).clamp(0.0, 1.0);
                    low_floor * (r * log_knee_ratio).exp()
                };
                let ccif_res =
                    0.30 * ((TAU * f_ccif * t4).sin() + (TAU * (f_ccif + 80.0) * t4).sin());
                let ccif_wide = 0.15 * ((TAU * 60.0 * t4).sin() + (TAU * f_air * t4).sin());
                let mt_norm = multitone[idx] / max_mt;
                body[s3_end + idx] = swell_env * (0.55 * mt_norm + ccif_res + ccif_wide);
            }
        }
    }

    // Section 5 (88..100%): Broadband high-crest cluster (0..60%) + Isolated chord/impulse tail probe (60..68%) + Pre-tail decay gap (68..100%)
    let n5 = n_body.saturating_sub(s4_end);
    if n5 > 0 {
        let cluster_end = (n5 * 60) / 100;
        let impulse_start = (n5 * 62) / 100;
        let impulse_end = (n5 * 68) / 100;
        let f_max_c = nyq_guard.clamp(80.0, 18_000.0);
        let n_c_tones = 20_usize;
        let c_log_ratio = (f_max_c / 25.0_f32).ln();

        for j in 0..cluster_end {
            let t5 = (j as f32) / fs;
            let u = (j as f32) / (cluster_end.max(1) as f32);
            let env5 = sat + (0.98 - sat) * u;
            let mut sum_c = 0.0_f32;
            for m in 1..=n_c_tones {
                let frac_m = ((m - 1) as f32) / ((n_c_tones - 1) as f32);
                let fm = 25.0 * (frac_m * c_log_ratio).exp();
                if fm >= nyq_guard {
                    break;
                }
                let ph = PI * (m as f32) * ((m + 1) as f32) / (n_c_tones as f32);
                sum_c += (TAU * fm * t5 + ph).sin();
            }
            body[s4_end + j] = env5 * (sum_c / (n_c_tones as f32).sqrt()).clamp(-1.0, 1.0);
        }

        if impulse_end > impulse_start {
            let imp_len = impulse_end - impulse_start;
            let burst_s = ((0.080 * fs).round() as usize).max(1).min(imp_len);
            for k in 0..burst_s {
                let ti = (k as f32) / fs;
                let win = 0.5 * (1.0 - (TAU * (k as f32) / (burst_s.max(1) as f32)).cos());
                let chord_imp = 0.35 * (TAU * 110.0 * ti).sin()
                    + 0.30 * (TAU * 220.0 * ti + 0.4).sin()
                    + 0.20 * (TAU * 660.0 * ti + 1.1).sin()
                    + 0.15 * (TAU * 2_200.0 * ti + 0.7).sin();
                body[s4_end + impulse_start + k] = 0.98 * win * chord_imp;
            }
        }
    }

    // Smooth C^1 fade-in at body start and fade-out before the Section 5B tail gap
    let fade_len = ((0.005 * fs).round() as usize)
        .max(4)
        .min((n_body / 8).max(1));
    for i in 0..fade_len {
        let w = 0.5 * (1.0 - (PI * (i as f32) / (fade_len as f32)).cos());
        body[i] *= w;
    }
    let peak_body = body.iter().map(|&x| x.abs()).fold(0.0_f32, f32::max);
    if peak_body > 1e-6 {
        let scale = 0.98 / peak_body;
        for sample in &mut body {
            *sample *= scale;
        }
    }
    out[body_start..body_end].copy_from_slice(&body);
    out
}

/// Generates a deterministic multi-band dry excitation signal for re-amp system identification.
pub fn generate_excitation_signal(sample_rate_hz: u32, num_samples: usize) -> Vec<f32> {
    generate_excitation_signal_with_profile(sample_rate_hz, num_samples, None)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(24))]

        #[test]
        fn prop_generate_excitation_signal_with_profile_bounds_and_finiteness(
            sample_rate_hz in 8_000_u32..=96_000,
            num_samples in 64_usize..=2_048,
            knee in 0.02_f32..=0.80,
            sat_offset in 0.05_f32..=0.18,
            has_tail in proptest::bool::ANY,
            max_delay_ms in 0.0_f32..=500.0,
            tail_t60_ms in 0.0_f32..=2_000.0,
            lfo_hz in 0.0_f32..=10.0,
            res_hz in 80.0_f32..=6_000.0,
        ) {
            let prof = TargetExcitationProfile {
                knee_amplitude_lin: knee,
                saturation_ceiling_lin: (knee + sat_offset).min(0.98),
                has_delay_or_reverb_tail: has_tail,
                estimated_max_delay_ms: max_delay_ms,
                estimated_tail_t60_ms: tail_t60_ms,
                detected_lfo_hz: lfo_hz,
                dominant_resonance_hz: res_hz,
            };
            let sig = generate_excitation_signal_with_profile(sample_rate_hz, num_samples, Some(&prof));
            prop_assert_eq!(sig.len(), num_samples);
            prop_assert!(sig.iter().all(|&x| x.is_finite() && (-1.0..=1.0).contains(&x)));
        }
    }

    #[test]
    fn test_synth_alignment_pulses_probe_and_excitation_analysis() {
        assert!(synth_alignment_pulses(48_000, 0).is_empty());
        assert!(generate_calibration_probe(48_000, 0).is_empty());
        assert!(generate_excitation_signal_with_profile(48_000, 0, None).is_empty());

        // Short slice fallback in analyze_probe_for_excitation
        let short_prof = analyze_probe_for_excitation(&[0.0; 16], &[0.0; 16], 48_000);
        assert_eq!(short_prof, TargetExcitationProfile::default());

        // 8 kHz low sample-rate excitation exercises Nyquist multitone cutoff guard
        let sig_8k = generate_excitation_signal(8_000, 2_048);
        assert_eq!(sig_8k.len(), 2_048);

        // Probe synthesis + delay/LFO/soft-clip analysis (ported from test_excitation.py)
        let n = 48_000_usize;
        let probe = generate_calibration_probe(48_000, n);
        let mut wet_delay = vec![0.0_f32; n];
        for i in 0..n {
            let t = (i as f32) / 48_000.0;
            let lfo_mod = 1.0 + 0.65 * (TAU * 3.0 * t).cos();
            let driven = (probe[i] * 4.5).tanh() * lfo_mod;
            wet_delay[i] = driven;
        }
        let echo_samples = 1_200_usize;
        for i in echo_samples..n {
            wet_delay[i] += 0.45 * wet_delay[i - echo_samples];
        }
        // Inject decaying echo energy in the Part C silent listening gap (82..95%)
        let gap_start = (n * 82) / 100;
        let gap_end = (n * 95) / 100;
        for (k, sample) in wet_delay[gap_start..gap_end].iter_mut().enumerate() {
            let decay = (-((k as f32) / 1_500.0)).exp();
            *sample = 0.18 * decay * (TAU * 600.0 * (k as f32) / 48_000.0).sin();
        }

        let prof = analyze_probe_for_excitation(&probe, &wet_delay, 48_000);
        assert!(prof.has_delay_or_reverb_tail);
        assert!(prof.estimated_max_delay_ms >= 15.0);
        assert!(prof.estimated_tail_t60_ms >= 80.0);
        assert!(prof.knee_amplitude_lin < prof.saturation_ceiling_lin);

        let tailored = generate_excitation_signal_with_profile(48_000, 8_192, Some(&prof));
        assert_eq!(tailored.len(), 8_192);
    }

    #[test]
    fn test_universal_excitation_saturation_density_and_envelope_ballistic_invariants() {
        let sr = 48_000_u32;
        // 1. Header pulses must be smooth Hann-windowed chirps (zero at boundaries, no step-tick spike)
        let hdr = synth_alignment_pulses(sr, 21_600);
        assert_eq!(hdr[0], 0.0);
        assert_eq!(*hdr.last().expect("non-empty"), 0.0);
        let max_hdr_diff = hdr
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_hdr_diff < 0.75,
            "Alignment header must not contain a 2-sample +0.95/-0.95 step-tick discontinuity (got max diff {max_hdr_diff})"
        );

        assert_eq!(DEFAULT_HARDWARE_EXCITATION_SEC, 30);
        assert_eq!(DEFAULT_HARDWARE_PROBE_SEC, 6);
        assert!((TRAILING_NOISE_WINDOW_SEC - 3.00).abs() < 1e-6);

        // 2. Universal 5-section excitation body must simultaneously deliver full-scale saturation drive
        // (peak >= 0.97, RMS >= -13.5 dBFS, >= 5% samples >= -6 dBFS, >= 1.5% samples >= -3 dBFS),
        // sub-threshold / release-probe envelope ballistics coverage ([-54 dBFS, -24 dBFS]),
        // constant-envelope LFO sustain, and an uninterrupted trailing decay window.
        let n = (sr as usize) * 10;
        let sig = generate_excitation_signal_with_profile(sr, n, None);
        let body_start = (0.55 * (sr as f32)) as usize;
        let tail_len = n / 10; // 1.0s on 10s signal, 3.0s on 30s signal
        let body_end = n - tail_len;
        assert!(
            sig[body_end..].iter().all(|&x| x == 0.0),
            "Trailing decay/noise-floor window must be strictly silent in dry excitation"
        );
        let body = &sig[body_start..body_end];
        let body_peak = body.iter().map(|&x| x.abs()).fold(0.0_f32, f32::max);
        let body_rms = (body.iter().map(|&x| x * x).sum::<f32>() / (body.len() as f32)).sqrt();
        let body_rms_dbfs = 20.0 * body_rms.log10();
        let frac_ge_minus_6dbfs =
            (body.iter().filter(|&&x| x.abs() >= 0.50).count() as f32) / (body.len() as f32);
        let frac_ge_minus_3dbfs =
            (body.iter().filter(|&&x| x.abs() >= 0.7079).count() as f32) / (body.len() as f32);
        let frac_low_ballistic = (body
            .iter()
            .filter(|&&x| (0.002..=0.063).contains(&x.abs()))
            .count() as f32)
            / (body.len() as f32);

        assert!(
            body_peak >= 0.97,
            "Expected full-scale Zone-B peak >= 0.97, got {body_peak}"
        );
        assert!(
            body_rms_dbfs >= -13.5,
            "Expected high-drive body RMS >= -13.5 dBFS, got {body_rms_dbfs} dBFS"
        );
        assert!(
            frac_ge_minus_6dbfs >= 0.05,
            "Expected >= 5% of body samples >= -6 dBFS for amp/fuzz saturation, got {}",
            frac_ge_minus_6dbfs
        );
        assert!(
            frac_ge_minus_3dbfs >= 0.015,
            "Expected >= 1.5% of body samples >= -3 dBFS for top-end clipping, got {}",
            frac_ge_minus_3dbfs
        );
        assert!(
            frac_low_ballistic >= 0.05,
            "Expected >= 5% of body samples in [-54 dBFS, -24 dBFS] for envelope-filter threshold & release probing, got {}",
            frac_low_ballistic
        );
        assert!((MAX_DYNAMIC_TAIL_EXTRA_SEC - 15.00).abs() < 1e-6);
    }
}
