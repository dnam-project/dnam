//! Deterministic 3-pulse asymmetric alignment burst, 2-step calibration probe synthesis,
//! probe system-ID analyzer (`TargetExcitationProfile`), and tailored dry excitation generation.

use std::f32::consts::{PI, TAU};

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
/// Duration of the trailing silent window for noise-floor estimation (`0.50s`).
pub const TRAILING_NOISE_WINDOW_SEC: f32 = 0.50;

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

    // Leading 1/16 silent lead-in + 3-pulse asymmetric burst in first 3/8 of buffer
    let lead_silence = (num_samples / 16).max(1);
    let header_end = ((num_samples * 7) / 16)
        .max(lead_silence + 1)
        .min(num_samples);
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

    // Part A (0..40% of body): Log-amplitude nonlinearity ramp (-40 dBFS -> 0 dBFS) on 220+440 Hz dyad
    let a_end = (body_len * 4) / 10;
    for i in 0..a_end {
        let frac = (i as f32) / (a_end.max(1) as f32);
        let amp = 10.0_f32.powf(-2.0 + 2.0 * frac);
        let t = (i as f32) / fs;
        let dyad = 0.70 * (TAU * 220.0 * t).sin() + 0.30 * (TAU * 440.0 * t).sin();
        out[body_start + i] = amp * dyad;
    }

    // Part B (40..80% of body): Steady harmonic carrier + fast log sweep
    let b_end = (body_len * 8) / 10;
    let nb = b_end.saturating_sub(a_end).max(1);
    let dur_b = ((nb as f32) / fs).max(1e-4);
    let f0 = 60.0_f32;
    let f1 = (0.42 * fs).min(10_000.0);
    let k_b = (f1 / f0).ln() / dur_b;
    for j in 0..(b_end.saturating_sub(a_end)) {
        let t = (j as f32) / fs;
        let phi = TAU * f0 * (((k_b * t).exp() - 1.0) / k_b);
        let carrier = 0.35 * ((TAU * 330.0 * t).sin() + 0.5 * (TAU * 660.0 * t).sin());
        out[body_start + a_end + j] = carrier + 0.35 * phi.sin();
    }

    // Part C (80..100% of body): Plosive transient burst + silent echo listening gap
    let nc = body_len.saturating_sub(b_end);
    let burst_samples = ((0.015 * fs).round() as usize).max(8).min(nc / 2);
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
        let decay_db = (20.0 * (rms_early / rms_late).max(1.05).log10()).max(1.5);
        (dt_ms * (60.0 / decay_db))
            .max(estimated_max_delay_ms * 3.5)
            .clamp(80.0, 5_000.0)
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
            if (60.0..=6_000.0).contains(&freq) {
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

/// Generates a multi-band dry excitation signal tailored to an optional [`TargetExcitationProfile`].
///
/// Structure:
/// 1. Leading 3-pulse asymmetric alignment burst + settling silence ([`synth_alignment_pulses`]).
/// 2. Core 3-section excitation body spanning both Zone A (`[knee_amplitude_lin, saturation_ceiling_lin]`)
///    and Zone B (`[saturation_ceiling_lin, 0.98]` post-clipping overdrive & full-scale headroom):
///    - **Section 1 (0..35%):** Logarithmic sweeps (`25 Hz -> 14 kHz`) modulated across Zone A and Zone B,
///      plus a steady phase-locking harmonic carrier at `dominant_resonance_hz` when `detected_lfo_hz > 0.15`.
///    - **Section 2 (35..70%):** Transient velocity ladder of root+fifth+octave dyads (`55 Hz .. 440 Hz`)
///      stepping across `[0.45 * knee_amplitude_lin, saturation_ceiling_lin]` and `[saturation_ceiling_lin, 0.98]`,
///      with `2x` stride silent echo-isolation gaps when `has_delay_or_reverb_tail` is true.
///    - **Section 3 (70..100%):** Dynamic `sin^2` swell (`0 -> 0.98 -> 0`) combining a resonance-centered
///      CCIF two-tone (`f_res`, `f_res + 80 Hz`) and a 16-tone Schroeder-phase multitone.
/// 3. Trailing silent window for per-stem `noise_floor_dbfs` and tail decay measurement.
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

    let lead_silence = (num_samples / 16).max(1);
    let header_end = ((num_samples * 7) / 16)
        .max(lead_silence + 1)
        .min(num_samples);
    let header_len = header_end - lead_silence;
    let pulses = synth_alignment_pulses(sample_rate_hz, header_len);
    out[lead_silence..header_end].copy_from_slice(&pulses);

    let extra_tail = if prof.has_delay_or_reverb_tail {
        ((prof.estimated_tail_t60_ms * 0.001 * fs).round() as usize).min(num_samples / 8)
    } else {
        0
    };
    let tail_silence = ((num_samples / 16) + extra_tail)
        .max(1)
        .min((num_samples / 4).max(1));
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

    let mut body = vec![0.0_f32; n_body];
    let s1_end = (n_body * 35) / 100;
    let s2_end = (n_body * 70) / 100;

    // Section 1 (0..35%): Logarithmic sweeps across Zone A (knee..sat) and Zone B (sat..0.98)
    if s1_end > 0 {
        let d1 = ((s1_end as f32) / fs).max(1e-4);
        let f_start = 25.0_f32;
        let f_stop = (0.42 * fs).min(14_000.0);
        let k_sw = (f_stop / f_start).ln() / d1;
        let half_s1 = s1_end / 2;
        for i in 0..s1_end {
            let t1 = (i as f32) / fs;
            let frac = (i as f32) / (s1_end.max(1) as f32);
            // Zone A in first half (knee -> sat), Zone B in second half (sat -> 0.98 full scale)
            let env_sw = if frac < 0.5 {
                let u = frac * 2.0;
                knee + (sat - knee) * (0.5 - 0.5 * (TAU * u).cos())
            } else {
                let u = (frac - 0.5) * 2.0;
                sat + (0.98 - sat) * (0.5 - 0.5 * (PI * u).cos())
            };
            let phi_sw = TAU * f_start * (((k_sw * t1).exp() - 1.0) / k_sw);
            body[i] = env_sw * phi_sw.sin();
        }
        if prof.detected_lfo_hz > 0.15 && half_s1 > 8 {
            let f_res = res_hz.clamp(110.0, 1_200.0);
            for (i, sample) in body.iter_mut().enumerate().take(s1_end).skip(half_s1) {
                let tl = (i as f32) / fs;
                let lfo_carrier = knee
                    * (0.65 * (TAU * f_res * tl).sin() + 0.35 * (TAU * (1.5 * f_res) * tl).sin());
                *sample = 0.55 * *sample + 0.45 * lfo_carrier;
            }
        }
    }

    // Section 2 (35..70%): Transient velocity ladder across Zone A (0.45*knee..sat) & Zone B (sat..0.98)
    let n2 = s2_end.saturating_sub(s1_end);
    if n2 > 0 {
        let notes_hz: [f32; 8] = [55.0, 82.41, 110.0, 146.83, 196.0, 246.94, 329.63, 440.0];
        let burst_len = ((0.18 * fs).round() as usize).max(16).min((n2 / 4).max(16));
        let stride = if prof.has_delay_or_reverb_tail {
            (burst_len * 2).max(1)
        } else {
            burst_len.max(1)
        };
        let vel_lo = (0.45 * knee).max(0.04);
        let mut b_idx = 0_usize;
        let mut start_idx = 0_usize;
        while start_idx < n2 {
            let end_idx = (start_idx + burst_len).min(n2);
            let f_root = notes_hz[b_idx % notes_hz.len()];
            let f_fifth = f_root * 1.498_307_1;
            let step_pos = b_idx % 8;
            // First 5 steps densely resolve Zone A [vel_lo, sat]; steps 5..7 drive Zone B [sat, 0.98]
            let vel = if step_pos < 5 {
                vel_lo + (sat - vel_lo) * ((step_pos as f32) / 4.0)
            } else {
                sat + (0.98 - sat) * (((step_pos - 4) as f32) / 3.0)
            };
            let decay_tau = if prof.has_delay_or_reverb_tail {
                0.030
            } else if b_idx.is_multiple_of(3) {
                0.035
            } else {
                0.120
            };
            for k in 0..(end_idx - start_idx) {
                let tb = (k as f32) / fs;
                let env_b = (1.0 - (-tb / 0.002).exp()) * (-tb / decay_tau).exp();
                let wave_b = 0.60 * (TAU * f_root * tb).sin()
                    + 0.30 * (TAU * f_fifth * tb + 0.3).sin()
                    + 0.10 * (TAU * (2.0 * f_root) * tb + 0.7).sin();
                body[s1_end + start_idx + k] = vel * env_b * wave_b;
            }
            b_idx += 1;
            start_idx += stride;
        }
    }

    // Section 3 (70..100%): Dynamic sin^2 swell + resonance-centered CCIF two-tone + 16-tone Schroeder multitone
    let n3 = n_body.saturating_sub(s2_end);
    if n3 > 0 {
        let d3 = ((n3 as f32) / fs).max(1e-4);
        let f_ccif = res_hz.clamp(250.0, 3_500.0);
        let n_tones = 16_usize;
        let mut multitone = vec![0.0_f32; n3];
        let mut max_mt = 1e-6_f32;
        for (j, mt_val) in multitone.iter_mut().enumerate() {
            let t3 = (j as f32) / fs;
            let mut sum_t = 0.0_f32;
            for k_tone in 1..=n_tones {
                let fk = 40.0 * 1.38_f32.powi((k_tone as i32) - 1);
                if fk >= 0.45 * fs {
                    break;
                }
                let schroeder_phase =
                    PI * (k_tone as f32) * ((k_tone - 1) as f32) / (n_tones as f32);
                sum_t += (TAU * fk * t3 + schroeder_phase).cos();
            }
            *mt_val = sum_t;
            max_mt = max_mt.max(sum_t.abs());
        }
        for (j, &mt_raw) in multitone.iter().enumerate() {
            let t3 = (j as f32) / fs;
            let s_val = (PI * t3 / d3).sin();
            let swell_env = s_val * s_val;
            let ccif = 0.45 * ((TAU * f_ccif * t3).sin() + (TAU * (f_ccif + 80.0) * t3).sin());
            let mt_norm = mt_raw / max_mt;
            body[s2_end + j] = 0.98 * swell_env * (0.55 * mt_norm + 0.45 * ccif);
        }
    }

    // Smooth C^1 fade-in and fade-out at body boundaries
    let fade_len = ((0.005 * fs).round() as usize)
        .max(4)
        .min((n_body / 8).max(1));
    for i in 0..fade_len {
        let w = 0.5 * (1.0 - (PI * (i as f32) / (fade_len as f32)).cos());
        body[i] *= w;
        body[n_body - 1 - i] *= w;
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
}
