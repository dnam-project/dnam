#![forbid(unsafe_code)]
//! 4-Regime Sub-Sample / Clock-Skew Alignment (`CleanLinear`, `PolarityInverted`,
//! `DispersiveAllPass`, `NonlinearEnvelopeFallback`) and `64`-Tap Kaiser-Sinc Fractional
//! Realignment (`align::regimes`).
//!
//! Canonical Reference: `docs/specs/training/capture_and_alignment.md`
//! (`ref[train.capture.alignment.multi-regime]`, `ref[train.capture.invariants]`).

use std::f64::consts::PI;

pub use super::calibration::{
    ACOUSTIC_NULL_HOMING_MAX_DB, CALB_STIMULUS_SAMPLES_48K, DecayTailDecision,
    DifferentialPadCalibration, LIVE_DECAY_TAIL_MARGIN_DB, LoopbackCalibrationReport,
    MIC_PAD_PROBE_SAMPLES_48K, NullHomingReport, THERMAL_RECOVERY_MAX_DB, ThermalRecoveryReport,
    calibrate_loopback_calb, evaluate_acoustic_null_homing, evaluate_live_decay_tail,
    evaluate_thermal_recovery_gate, generate_calb_loopback_stimulus, generate_mic_pad_probe_400ms,
    measure_differential_pad_ratio, rescale_padded_stem,
};
use super::{AlignError, estimate_subsample_lag_and_gain};

/// Detected physical signal-path regime during sub-sample alignment
/// (`ref[train.capture.alignment.multi-regime]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentRegime {
    /// High linear coherence ($\gamma_{\text{lin}} \ge 0.35$), positive polarity, non-dispersive peak.
    CleanLinear,
    /// High linear coherence ($\gamma_{\text{lin}} \ge 0.35$) with inverted polarity (`signed_gain < 0.0`).
    PolarityInverted,
    /// Dispersive phase/all-pass response (crossover, wah, spring) locked to causal HF wavefront.
    DispersiveAllPass,
    /// Suppressed fundamental linear coherence ($\gamma_{\text{lin}} < 0.35$ from hard fuzz, rectifier,
    /// or octave divider) locked via broadband analytic Hilbert envelope cross-correlation.
    NonlinearEnvelopeFallback,
}

/// Multi-regime sub-sample latency, polarity, coherence, and clock-skew estimate
/// (`ref[train.capture.alignment.multi-regime]`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiRegimeAlignment {
    /// Active physical alignment regime selected for this stem.
    pub regime: AlignmentRegime,
    /// Total sub-sample lag in device-rate samples (`wet` delayed relative to `dry`).
    pub total_lag_samples: f64,
    /// Signed linear gain (or signed RMS gain in `NonlinearEnvelopeFallback`).
    pub signed_gain: f64,
    /// True if wet return is polarity-inverted (`signed_gain < 0.0`).
    pub polarity_inverted: bool,
    /// Normalized broadband linear coherence $\gamma_{\text{lin}} \in [0.0, 1.0]$.
    pub linear_coherence: f64,
    /// Estimated sample-clock skew / drift in parts-per-million (`ppm`).
    pub clock_drift_ppm: f64,
}

#[inline]
fn bessel_i0_f64(x: f64) -> f64 {
    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;
    let half_sq = 0.25 * x * x;
    for k in 1..=24 {
        let kf = f64::from(k);
        term *= half_sq / (kf * kf);
        sum += term;
        if term < sum * 1.0e-16 {
            break;
        }
    }
    sum
}

/// Realigns `wet` by advancing it by `lag_samples + (clock_drift_ppm * 1e-6) * n` using a `64`-tap
/// (`radius = 32`) partition-of-unity Kaiser-sinc interpolator ($\beta = 8.6$) and optional polarity flip.
#[must_use]
pub fn realign_fractional_kaiser64(
    wet: &[f32],
    lag_samples: f64,
    clock_drift_ppm: f64,
    invert_polarity: bool,
) -> Vec<f32> {
    if wet.is_empty() {
        return Vec::new();
    }
    let clean: Vec<f64> = wet
        .iter()
        .map(|&v| if v.is_finite() { f64::from(v) } else { 0.0 })
        .collect();
    let sign = if invert_polarity { -1.0_f64 } else { 1.0_f64 };
    let safe_lag = if lag_samples.is_finite() {
        lag_samples
    } else {
        0.0
    };
    let drift_rate = if clock_drift_ppm.is_finite() {
        (clock_drift_ppm * 1.0e-6).clamp(-0.05, 0.05)
    } else {
        0.0
    };

    if safe_lag.abs() < 1.0e-12 && drift_rate.abs() < 1.0e-15 {
        return clean.into_iter().map(|v| (sign * v) as f32).collect();
    }

    const RADIUS: isize = 32;
    const BETA: f64 = 8.6;
    let inv_i0 = 1.0 / bessel_i0_f64(BETA);
    let n_len = clean.len() as isize;
    let mut out = Vec::with_capacity(clean.len());

    let build_taps = |frac: f64| -> [f64; 64] {
        let mut taps = [0.0_f64; 64];
        let mut sum_w = 0.0_f64;
        for (t, slot) in taps.iter_mut().enumerate() {
            let u = frac - (((t as isize) - RADIUS + 1) as f64);
            let abs_u = u.abs();
            if abs_u >= (RADIUS as f64) {
                continue;
            }
            let sinc = if abs_u < 1.0e-12 {
                1.0
            } else {
                (PI * u).sin() / (PI * u)
            };
            let r = abs_u / (RADIUS as f64);
            let w = sinc * bessel_i0_f64(BETA * (1.0 - r * r).max(0.0).sqrt()) * inv_i0;
            *slot = w;
            sum_w += w;
        }
        let sc = if sum_w.abs() > 1.0e-15 {
            sign / sum_w
        } else {
            0.0
        };
        for slot in &mut taps {
            *slot *= sc;
        }
        taps
    };

    let dot_at = |base: isize, taps: &[f64; 64]| -> f32 {
        let mut acc = 0.0_f64;
        if base >= 0 && base + 64 <= n_len {
            for (&w, &x) in taps.iter().zip(&clean[base as usize..(base as usize + 64)]) {
                acc += w * x;
            }
        } else {
            for (t, &w) in taps.iter().enumerate() {
                let m = base + (t as isize);
                if m >= 0 && m < n_len {
                    acc += w * clean[m as usize];
                }
            }
        }
        if acc.is_finite() { acc as f32 } else { 0.0 }
    };

    if drift_rate.abs() < 1.0e-15 {
        let lag_floor = safe_lag.floor() as isize;
        let taps = build_taps(safe_lag - (lag_floor as f64));
        for n in 0..clean.len() {
            out.push(dot_at((n as isize) + lag_floor - RADIUS + 1, &taps));
        }
        return out;
    }

    let table: Vec<[f64; 64]> = (0..=512).map(|p| build_taps((p as f64) / 512.0)).collect();
    for n in 0..clean.len() {
        let center = (n as f64) + safe_lag + drift_rate * (n as f64);
        let k_base = center.floor() as isize;
        let phase_f = ((center - (k_base as f64)) * 512.0).clamp(0.0, 512.0);
        let p0 = phase_f.floor() as usize;
        let alpha = (phase_f - (p0 as f64)) as f32;
        let y0 = dot_at(k_base - RADIUS + 1, &table[p0]);
        let y1 = dot_at(k_base - RADIUS + 1, &table[(p0 + 1).min(512)]);
        out.push(y0 + alpha * (y1 - y0));
    }
    out
}

fn compute_analytic_envelope_zero_mean(sig: &[f32]) -> Vec<f32> {
    let n = sig.len();
    if n == 0 {
        return Vec::new();
    }
    let mean = sig.iter().map(|&x| f64::from(x)).sum::<f64>() / (n as f64);
    let centered: Vec<f64> = sig.iter().map(|&x| f64::from(x) - mean).collect();
    let mut env = Vec::with_capacity(n);
    let mut env_sum = 0.0_f64;
    for i in 0..n {
        let x0 = centered[i];
        let xm1 = if i >= 1 { centered[i - 1] } else { 0.0 };
        let xp1 = if i + 1 < n { centered[i + 1] } else { 0.0 };
        let xm3 = if i >= 3 { centered[i - 3] } else { 0.0 };
        let xp3 = if i + 3 < n { centered[i + 3] } else { 0.0 };
        let hilb = core::f64::consts::FRAC_2_PI * (xm1 - xp1) + (0.9 / (1.5 * PI)) * (xm3 - xp3);
        let mag = x0.hypot(hilb);
        env_sum += mag;
        env.push(mag);
    }
    let env_mean = env_sum / (n as f64);
    env.into_iter().map(|e| (e - env_mean) as f32).collect()
}

fn normalized_corr_at_fractional_lag(dry: &[f32], wet: &[f32], lag: f64) -> (f64, f64) {
    let shifted = realign_fractional_kaiser64(wet, lag, 0.0, false);
    let n = dry.len().min(shifted.len());
    let margin = ((lag.abs().ceil() as usize) + 34).min(n / 4);
    let start = margin;
    let end = n.saturating_sub(margin).max(start);
    let (slice_d, slice_w) = if end > start + 8 {
        (&dry[start..end], &shifted[start..end])
    } else {
        (&dry[..n], &shifted[..n])
    };
    let mut dot = 0.0_f64;
    let mut ex = 0.0_f64;
    let mut ey = 0.0_f64;
    for (&d, &w) in slice_d.iter().zip(slice_w) {
        let d64 = f64::from(d);
        let w64 = f64::from(w);
        dot += d64 * w64;
        ex += d64 * d64;
        ey += w64 * w64;
    }
    let denom = (ex * ey).sqrt().max(1.0e-30);
    let gain = if ex > 1.0e-30 { dot / ex } else { 0.0 };
    (dot / denom, gain)
}

fn refine_fractional_lag_golden(dry: &[f32], wet: &[f32], init_lag: f64) -> (f64, f64, f64) {
    let mut a = init_lag - 0.65;
    let mut b = init_lag + 0.65;
    let inv_phi = 0.618_033_988_749_894_9_f64;
    let inv_phi2 = 1.0 - inv_phi;
    let mut c = a + inv_phi2 * (b - a);
    let mut d = a + inv_phi * (b - a);
    let mut fc = normalized_corr_at_fractional_lag(dry, wet, c).0.abs();
    let mut fd = normalized_corr_at_fractional_lag(dry, wet, d).0.abs();

    for _ in 0..20 {
        if fc >= fd {
            b = d;
            d = c;
            fd = fc;
            c = a + inv_phi2 * (b - a);
            fc = normalized_corr_at_fractional_lag(dry, wet, c).0.abs();
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + inv_phi * (b - a);
            fd = normalized_corr_at_fractional_lag(dry, wet, d).0.abs();
        }
    }
    let best_lag = 0.5 * (a + b);
    let (signed_coh, signed_gain) = normalized_corr_at_fractional_lag(dry, wet, best_lag);
    (best_lag, signed_coh, signed_gain)
}

fn estimate_leading_edge_polarity(dry: &[f32], wet: &[f32], lag_int: isize) -> bool {
    let peak_dry = dry.iter().fold(0.0_f32, |acc, &x| acc.max(x.abs()));
    if peak_dry <= 1.0e-6 {
        return false;
    }
    let thresh = 0.25 * peak_dry;
    for (i, &d) in dry.iter().enumerate() {
        if d.abs() >= thresh {
            let j = i as isize + lag_int;
            if j >= 0 && (j as usize) < wet.len() {
                let w = wet[j as usize];
                if w.abs() > 1.0e-6 {
                    return (d * w) < 0.0;
                }
            }
        }
    }
    false
}

fn second_diff_hp_wavefront(sig: &[f32]) -> Vec<f32> {
    (0..sig.len())
        .map(|i| {
            let x0 = sig[i];
            let x1 = if i >= 1 { sig[i - 1] } else { 0.0 };
            let x2 = if i >= 2 { sig[i - 2] } else { 0.0 };
            x0 - 1.92 * x1 + 0.9216 * x2
        })
        .collect()
}

fn eighth_diff_wavefront(sig: &[f32]) -> Vec<f32> {
    const BINOM_8: [f32; 9] = [1.0, -8.0, 28.0, -56.0, 70.0, -56.0, 28.0, -8.0, 1.0];
    (0..sig.len())
        .map(|i| {
            let mut acc = 0.0_f32;
            for (k, &c) in BINOM_8.iter().enumerate() {
                if i >= k {
                    acc += c * sig[i - k];
                }
            }
            acc
        })
        .collect()
}

/// Estimates sub-sample latency, signed gain, polarity, physical regime (`CleanLinear`,
/// `PolarityInverted`, `DispersiveAllPass`, or `NonlinearEnvelopeFallback`), and linear clock
/// drift (`ppm`) (`ref[train.capture.alignment.multi-regime]`).
pub fn estimate_multiregime_alignment(
    dry: &[f32],
    wet: &[f32],
    sample_rate_hz: u32,
    max_lag_samples: usize,
) -> Result<MultiRegimeAlignment, AlignError> {
    if sample_rate_hz == 0 {
        return Err(AlignError::InvalidAlignmentInput(
            "sample_rate_hz must be >= 1".to_string(),
        ));
    }
    if dry.len() < 8 || dry.len() != wet.len() {
        return Err(AlignError::InvalidAlignmentInput(
            "dry and wet must have equal length >= 8".to_string(),
        ));
    }
    if dry.iter().any(|v| !v.is_finite()) || wet.iter().any(|v| !v.is_finite()) {
        return Err(AlignError::InvalidAlignmentInput(
            "dry and wet must contain finite samples".to_string(),
        ));
    }
    let inv_len = 1.0 / (dry.len() as f64);
    let mean_d = dry.iter().map(|&x| f64::from(x)).sum::<f64>() * inv_len;
    let mean_w = wet.iter().map(|&x| f64::from(x)).sum::<f64>() * inv_len;
    let dry_centered: Vec<f32> = dry
        .iter()
        .map(|&x| (f64::from(x) - mean_d) as f32)
        .collect();
    let wet_centered: Vec<f32> = wet
        .iter()
        .map(|&x| (f64::from(x) - mean_w) as f32)
        .collect();
    let dry = dry_centered.as_slice();
    let wet = wet_centered.as_slice();

    let coarse = estimate_subsample_lag_and_gain(dry, wet, max_lag_samples)?;
    let (lin_lag, signed_coh, lin_gain) =
        refine_fractional_lag_golden(dry, wet, coarse.total_lag_samples);
    let lin_coherence = signed_coh.abs().clamp(0.0, 1.0);

    // Evaluate interior Nyquist-band 8th-difference causal wavefront BEFORE linear/envelope
    // classification: on bass-heavy musical stimuli (`A5`), a 16-stage all-pass cascade can retain
    // high LF linear coherence (~0.97 at tau_LF ~ +68 samples) while the causal HF wavefront
    // arrives ~62 samples earlier (`lin_lag - hf_lag > 4.0`). Trimming 64 boundary samples
    // eliminates step-edge transients from finite-window fractional delay filters.
    let mut dispersive_hit: Option<(f64, f64, bool)> = None;
    if dry.len() >= 256 {
        let hf_dry_full = eighth_diff_wavefront(dry);
        let hf_wet_full = eighth_diff_wavefront(wet);
        let hf_dry = &hf_dry_full[64..hf_dry_full.len() - 64];
        let hf_wet = &hf_wet_full[64..hf_wet_full.len() - 64];
        let dry_e: f64 = dry[64..dry.len() - 64]
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum();
        let hf_dry_e: f64 = hf_dry.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        if hf_dry_e > 1.0e-10 * dry_e.max(1.0e-24) {
            let search_span = max_lag_samples.max((lin_lag.abs() as usize) + 16);
            if let Ok(hc) = estimate_subsample_lag_and_gain(hf_dry, hf_wet, search_span) {
                let (hf_lag, hf_coh, hf_gain) =
                    refine_fractional_lag_golden(hf_dry, hf_wet, hc.total_lag_samples);
                let hf_abs = hf_coh.abs();
                let strong_disp = (lin_coherence >= 0.35
                    && hf_abs >= 0.55
                    && (lin_lag - hf_lag) > 3.5)
                    || (lin_coherence < 0.35 && hf_abs >= 0.30 && hf_abs > lin_coherence + 0.12);
                if strong_disp {
                    let inv = if lin_coherence >= 0.50 {
                        lin_gain < 0.0
                    } else {
                        hf_coh < 0.0
                    };
                    let eff_gain = if lin_coherence >= 0.35 {
                        lin_gain
                    } else {
                        hf_gain
                    };
                    dispersive_hit = Some((hf_lag, eff_gain, inv));
                }
            }
        }
        if dispersive_hit.is_none() && lin_coherence >= 0.35 && lin_lag > 4.0 {
            let hp_dry_full = second_diff_hp_wavefront(dry);
            let hp_wet_full = second_diff_hp_wavefront(wet);
            let hp_dry = &hp_dry_full[64..hp_dry_full.len() - 64];
            let hp_wet = &hp_wet_full[64..hp_wet_full.len() - 64];
            let hp_dry_e: f64 = hp_dry.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
            if hp_dry_e > 1.0e-8 * dry_e.max(1.0e-24) {
                let search_span = max_lag_samples.max((lin_lag.abs() as usize) + 16);
                if let Ok(hc2) = estimate_subsample_lag_and_gain(hp_dry, hp_wet, search_span) {
                    let (hp_lag, hp_coh, _) =
                        refine_fractional_lag_golden(hp_dry, hp_wet, hc2.total_lag_samples);
                    if hp_coh.abs() >= 0.36 && hp_lag >= -1.0 && (lin_lag - hp_lag) > 3.5 {
                        let inv = if lin_coherence >= 0.50 {
                            lin_gain < 0.0
                        } else {
                            hp_coh < 0.0
                        };
                        dispersive_hit = Some((hp_lag, lin_gain, inv));
                    }
                }
            }
        }
    }

    let (regime, chosen_lag, chosen_gain, polarity_inverted) =
        if let Some((hf_lag, eff_gain, inv)) = dispersive_hit {
            (AlignmentRegime::DispersiveAllPass, hf_lag, eff_gain, inv)
        } else if lin_coherence < 0.35 {
            let env_dry = compute_analytic_envelope_zero_mean(dry);
            let env_wet = compute_analytic_envelope_zero_mean(wet);
            let (env_lag, env_coh) =
                match estimate_subsample_lag_and_gain(&env_dry, &env_wet, max_lag_samples) {
                    Ok(env_est) => {
                        let (elag, ecoh, _) = refine_fractional_lag_golden(
                            &env_dry,
                            &env_wet,
                            env_est.total_lag_samples,
                        );
                        (elag, ecoh.abs())
                    }
                    Err(_) => (lin_lag, 0.0),
                };
            let use_lag = if env_coh > lin_coherence {
                env_lag
            } else {
                lin_lag
            };
            let rms_dry = dry
                .iter()
                .map(|&x| f64::from(x) * f64::from(x))
                .sum::<f64>()
                .sqrt();
            let rms_wet = wet
                .iter()
                .map(|&x| f64::from(x) * f64::from(x))
                .sum::<f64>()
                .sqrt();
            let inv = if lin_coherence >= 0.12 {
                signed_coh < 0.0
            } else {
                estimate_leading_edge_polarity(dry, wet, use_lag.round() as isize)
            };
            let mag_gain = if rms_dry > 1.0e-15 {
                rms_wet / rms_dry
            } else {
                0.0
            };
            (
                AlignmentRegime::NonlinearEnvelopeFallback,
                use_lag,
                if inv { -mag_gain } else { mag_gain },
                inv,
            )
        } else if lin_gain < 0.0 {
            (AlignmentRegime::PolarityInverted, lin_lag, lin_gain, true)
        } else {
            (AlignmentRegime::CleanLinear, lin_lag, lin_gain, false)
        };

    let mut clock_drift_ppm = 0.0_f64;
    if dry.len() >= 24_000 {
        let seg_len = dry.len() / 4;
        let late_start = dry.len() - seg_len;
        let mut e_early = 0.0_f64;
        let mut n_early_num = 0.0_f64;
        for (i, &x) in dry[..seg_len].iter().enumerate() {
            let p = f64::from(x) * f64::from(x);
            e_early += p;
            n_early_num += (i as f64) * p;
        }
        let mut e_late = 0.0_f64;
        let mut n_late_num = 0.0_f64;
        for (i, &x) in dry[late_start..].iter().enumerate() {
            let p = f64::from(x) * f64::from(x);
            e_late += p;
            n_late_num += ((late_start + i) as f64) * p;
        }
        if e_early > 1.0e-5 && e_late > 1.0e-5 {
            let search = max_lag_samples.min(seg_len / 3);
            if let (Ok(c_early), Ok(c_late)) = (
                estimate_subsample_lag_and_gain(&dry[..seg_len], &wet[..seg_len], search),
                estimate_subsample_lag_and_gain(&dry[late_start..], &wet[late_start..], search),
            ) {
                let (l_early, coh_e, _) = refine_fractional_lag_golden(
                    &dry[..seg_len],
                    &wet[..seg_len],
                    c_early.total_lag_samples,
                );
                let (l_late, coh_l, _) = refine_fractional_lag_golden(
                    &dry[late_start..],
                    &wet[late_start..],
                    c_late.total_lag_samples,
                );
                if coh_e.abs() >= 0.40 && coh_l.abs() >= 0.40 {
                    let centroid_early = n_early_num / e_early;
                    let centroid_late = n_late_num / e_late;
                    let span = (centroid_late - centroid_early).max(1_000.0);
                    let ppm = ((l_late - l_early) / span) * 1.0e6;
                    if ppm.abs() <= 5_000.0 {
                        clock_drift_ppm = ppm;
                    }
                }
            }
        }
    }

    Ok(MultiRegimeAlignment {
        regime,
        total_lag_samples: chosen_lag,
        signed_gain: chosen_gain,
        polarity_inverted,
        linear_coherence: lin_coherence,
        clock_drift_ppm,
    })
}
