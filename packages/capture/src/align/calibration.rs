#![forbid(unsafe_code)]
//! Phase-0 `0.5s` `CALB` Loopback Calibration, `400 ms` Differential Mic-Pad Ratio,
//! `< 0.05 dB` Acoustic Null Homing, `<= 0.15 dB` Thermal Recovery Gate, and Live Decay-Tail
//! Monitor (`align::calibration`).
//!
//! Canonical Reference: `docs/specs/training/capture_and_alignment.md`
//! (`ref[train.capture.calib.loopback]`, `ref[train.capture.calib.clean-pad]`,
//! `ref[train.capture.planner.null-homing]`, `ref[train.capture.excitation.live-tail]`).

use std::f64::consts::PI;

use crate::proj::CalbChunk64B;
use crate::targets::CaptureBitDepth;

use super::AlignError;
use super::regimes::{
    MultiRegimeAlignment, estimate_multiregime_alignment, realign_fractional_kaiser64,
};

/// Canonical `0.5s` (`24,000` samples at `48 kHz`) Phase-0 `CALB` loopback length.
pub const CALB_STIMULUS_SAMPLES_48K: usize = 24_000;

/// Canonical `400 ms` (`19,200` samples at `48 kHz`) `-18 dBFS` differential mic-pad probe length.
pub const MIC_PAD_PROBE_SAMPLES_48K: usize = 19_200;

/// Maximum acoustic null-homing error (`0.05 dB`, `ref[train.capture.planner.null-homing]`).
pub const ACOUSTIC_NULL_HOMING_MAX_DB: f64 = 0.05;

/// Maximum pre-stem thermal/sag drift gate (`0.15 dB`, `ref[train.capture.planner.null-homing]`).
pub const THERMAL_RECOVERY_MAX_DB: f64 = 0.15;

/// Live decay-tail termination margin (`+1.0 dB` above calibrated noise floor $\epsilon_{\text{floor}}^2$).
pub const LIVE_DECAY_TAIL_MARGIN_DB: f64 = 1.0;

/// Phase-0 `0.5s` `CALB` converter loopback calibration report (`ref[train.capture.calib.loopback]`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoopbackCalibrationReport {
    /// Validated `64 B` `CALB` container chunk ready for `.dnam-proj` sealing.
    pub calb: CalbChunk64B,
    /// True if loopback return clipped at or above `-0.1 dBFS` (`0.988553` FS).
    pub converter_clipped: bool,
    /// Measured wet loopback peak level in dBFS.
    pub peak_dbfs: f32,
    /// Underlying 4-regime alignment telemetry at `sample_rate_device_hz`.
    pub alignment: MultiRegimeAlignment,
}

/// Differential `400 ms` `-18 dB` mic-pad ratio calibration (`ref[train.capture.calib.clean-pad]`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DifferentialPadCalibration {
    /// Linear pad gain $\hat{G}_{\text{pad}} \in (0, 2.0]$ (`padded / unpadded`).
    pub pad_gain_linear: f64,
    /// Measured pad attenuation in dB (`20 * log10(pad_gain_linear)`, e.g. `-10.0` or `-20.0`).
    pub pad_attenuation_db: f64,
    /// Sub-sample shift between unpadded and padded probes (compensates mic bump when toggling pad).
    pub relative_lag_samples: f64,
    /// Normalized correlation coherence $\gamma^2 \in [0.0, 1.0]$ between probes.
    pub coherence_sq: f64,
}

/// Live `< 0.05 dB` acoustic null-homing verification report (`ref[train.capture.planner.null-homing]`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NullHomingReport {
    /// True when `error_db < ACOUSTIC_NULL_HOMING_MAX_DB` (`0.05 dB`).
    pub homed: bool,
    /// Noise-qualified composite amplitude/null deviation in dB (`>= 0.0`).
    pub error_db: f64,
    /// Signed RMS level difference `live - ref` in dB.
    pub gain_delta_db: f64,
    /// Noise-subtracted null residual ESR in dB.
    pub residual_esr_db: f64,
}

/// Pre-stem `<= 0.15 dB` thermal/sag recovery gate report (`ref[train.capture.planner.null-homing]`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermalRecoveryReport {
    /// True when `drift_db <= THERMAL_RECOVERY_MAX_DB` (`0.15 dB`).
    pub recovered: bool,
    /// Absolute RMS/spectral return drift relative to `StarCenter` in dB.
    pub drift_db: f64,
}

/// Live post-stimulus decay-tail decision (`ref[train.capture.excitation.live-tail]`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecayTailDecision {
    /// Wet tail decayed within `+1.0 dB` of $\epsilon_{\text{floor}}^2$ (or reached ceiling).
    Complete {
        /// Trailing window RMS in dBFS.
        tail_rms_dbfs: f32,
        /// True if stopped by `max_tail_samples` ceiling rather than floor convergence.
        ceiling_reached: bool,
    },
    /// Wet tail remains above `noise_floor_dbfs + 1.0 dB`; continue recording silence tail.
    Extend {
        /// Trailing window RMS in dBFS.
        tail_rms_dbfs: f32,
        /// Target termination threshold (`noise_floor_dbfs + 1.0 dB`).
        target_dbfs: f32,
    },
}

#[inline]
fn bit_depth_to_calb_code(bit_depth: CaptureBitDepth) -> u8 {
    match bit_depth {
        CaptureBitDepth::Int12 => 0,
        CaptureBitDepth::Int16 => 1,
        CaptureBitDepth::Int24 => 2,
        CaptureBitDepth::Int32 => 3,
        CaptureBitDepth::Float32 => 4,
    }
}

/// Synthesizes a deterministic `0.5s` Phase-0 `CALB` converter loopback stimulus (`70%` Hann-tapered
/// `-3 dBFS` log chirp `25 Hz -> min(20 kHz, 0.45 * fs)` + `30%` silent quiescent-floor tail).
pub fn generate_calb_loopback_stimulus(sample_rate_hz: u32) -> Result<Vec<f32>, AlignError> {
    if !(8_000..=384_000).contains(&sample_rate_hz) {
        return Err(AlignError::InvalidAlignmentInput(format!(
            "unsupported CALB sample_rate_hz {sample_rate_hz}"
        )));
    }
    let total_n = ((sample_rate_hz as usize) / 2).max(64);
    let active_n = (total_n * 7) / 10;
    let f0 = 25.0_f64;
    let f1 = (0.45 * f64::from(sample_rate_hz)).clamp(300.0, 20_000.0);
    let dur = (active_n as f64) / f64::from(sample_rate_hz);
    let ln_ratio = (f1 / f0).ln().max(1.0e-6);
    let taper_n = (active_n / 12).max(4);
    let amp = 0.707_945_76_f64; // -3.0 dBFS

    let mut out = vec![0.0_f32; total_n];
    for (i, slot) in out.iter_mut().enumerate().take(active_n) {
        let t = (i as f64) / f64::from(sample_rate_hz);
        let phase = 2.0 * PI * f0 * dur * (((t / dur) * ln_ratio).exp() - 1.0) / ln_ratio;
        let win = if i < taper_n {
            0.5 * (1.0 - (PI * (i as f64) / (taper_n as f64)).cos())
        } else if i + taper_n >= active_n {
            let rem = active_n - 1 - i;
            0.5 * (1.0 - (PI * (rem as f64) / (taper_n as f64)).cos())
        } else {
            1.0
        };
        *slot = (amp * win * phase.sin()) as f32;
    }
    Ok(out)
}

/// Executes Phase-0 `0.5s` `CALB` converter loopback calibration (`ref[train.capture.calib.loopback]`),
/// producing a validated [`CalbChunk64B`] chunk and converter clip status.
pub fn calibrate_loopback_calb(
    dry: &[f32],
    wet: &[f32],
    sample_rate_device_hz: u32,
    bit_depth: CaptureBitDepth,
    input_reference_dbfs: f32,
    max_lag_samples: usize,
) -> Result<LoopbackCalibrationReport, AlignError> {
    if !(8_000..=384_000).contains(&sample_rate_device_hz) {
        return Err(AlignError::InvalidAlignmentInput(format!(
            "CALB sample_rate_device_hz {sample_rate_device_hz} outside 8000..=384000 Hz"
        )));
    }
    if !input_reference_dbfs.is_finite() {
        return Err(AlignError::InvalidAlignmentInput(
            "CALB input_reference_dbfs must be finite".to_string(),
        ));
    }
    if dry.len() < 256 || dry.len() != wet.len() {
        return Err(AlignError::InvalidAlignmentInput(
            "CALB loopback requires equal-length dry and wet buffers >= 256 samples".to_string(),
        ));
    }
    let alignment =
        estimate_multiregime_alignment(dry, wet, sample_rate_device_hz, max_lag_samples)?;
    let peak_abs = wet.iter().fold(0.0_f32, |acc, &x| acc.max(x.abs()));
    let peak_dbfs = if peak_abs > 1.0e-12 {
        20.0 * peak_abs.log10()
    } else {
        -160.0
    };
    let converter_clipped = peak_dbfs >= -0.1_f32;

    let settle = (alignment.total_lag_samples.ceil().max(0.0) as usize) + 64;
    let mut quiet_sum_sq = 0.0_f64;
    let mut quiet_cnt = 0usize;
    for i in settle..dry.len() {
        let src_end_ok = dry[i.saturating_sub(settle)..=i]
            .iter()
            .all(|&d| d.abs() <= 1.0e-6);
        if src_end_ok {
            let w = f64::from(wet[i]);
            quiet_sum_sq += w * w;
            quiet_cnt += 1;
        }
    }
    let floor_rms = if quiet_cnt >= 32 {
        (quiet_sum_sq / (quiet_cnt as f64)).sqrt()
    } else {
        let aligned = realign_fractional_kaiser64(
            wet,
            alignment.total_lag_samples,
            alignment.clock_drift_ppm,
            false,
        );
        let gain = alignment.signed_gain;
        let err_sq: f64 = dry
            .iter()
            .zip(&aligned)
            .map(|(&d, &w)| {
                let r = f64::from(w) - gain * f64::from(d);
                r * r
            })
            .sum();
        (err_sq / (dry.len() as f64)).sqrt()
    };
    let noise_floor_dbfs = if floor_rms > 1.0e-8 {
        (20.0 * floor_rms.log10() as f32).clamp(-160.0, 0.0)
    } else {
        -160.0_f32
    };

    let lag_48k = alignment.total_lag_samples * (48_000.0 / f64::from(sample_rate_device_hz));
    let calb = CalbChunk64B {
        sample_rate_device_hz,
        bit_depth_code: bit_depth_to_calb_code(bit_depth),
        polarity_inverted: alignment.polarity_inverted,
        loopback_lag_samples: lag_48k,
        signed_gain: alignment.signed_gain,
        input_reference_dbfs,
        noise_floor_dbfs,
    };

    Ok(LoopbackCalibrationReport {
        calb,
        converter_clipped,
        peak_dbfs,
        alignment,
    })
}

/// Synthesizes the canonical `400 ms` (`19,200` samples at `48 kHz`) `-18.0 dBFS` differential
/// mic-pad reference probe (`Law 0.F`, `ref[train.capture.calib.clean-pad]`).
#[must_use]
pub fn generate_mic_pad_probe_400ms(sample_rate_hz: u32) -> Vec<f32> {
    let sr = sample_rate_hz.clamp(8_000, 384_000);
    let n = ((sr as usize) * 2) / 5;
    let freqs = [110.0_f64, 247.0, 440.0, 880.0, 1_760.0, 3_320.0];
    let phases = [0.0_f64, 1.05, 2.41, 0.63, 3.88, 5.12];
    let mut raw = vec![0.0_f64; n];
    let mut peak = 1.0e-12_f64;
    let fade = (n / 16).max(8);
    for (i, slot) in raw.iter_mut().enumerate() {
        let t = (i as f64) / f64::from(sr);
        let mut s = 0.0_f64;
        for (k, (&f, &ph)) in freqs.iter().zip(&phases).enumerate() {
            if f < 0.45 * f64::from(sr) {
                let w = 1.0 / ((k + 1) as f64).sqrt();
                s += w * (2.0 * PI * f * t + ph).sin();
            }
        }
        let win = if i < fade {
            0.5 * (1.0 - (PI * (i as f64) / (fade as f64)).cos())
        } else if i + fade >= n {
            let rem = n - 1 - i;
            0.5 * (1.0 - (PI * (rem as f64) / (fade as f64)).cos())
        } else {
            1.0
        };
        let v = s * win;
        peak = peak.max(v.abs());
        *slot = v;
    }
    let target_peak = 0.125_892_54_f64; // -18.0 dBFS
    let scale = target_peak / peak;
    raw.into_iter().map(|v| (v * scale) as f32).collect()
}

/// Measures the exact linear pad attenuation $\hat{G}_{\text{pad}}$ (`padded / unpadded`) from a
/// `400 ms` `-18 dB` differential probe (`ref[train.capture.calib.clean-pad]`).
pub fn measure_differential_pad_ratio(
    prev_unpadded_wet: &[f32],
    new_padded_wet: &[f32],
    max_lag_samples: usize,
) -> Result<DifferentialPadCalibration, AlignError> {
    let align =
        estimate_multiregime_alignment(prev_unpadded_wet, new_padded_wet, 48_000, max_lag_samples)?;
    let aligned_new =
        realign_fractional_kaiser64(new_padded_wet, align.total_lag_samples, 0.0, false);
    let mut dot = 0.0_f64;
    let mut e_prev = 0.0_f64;
    let mut e_new = 0.0_f64;
    for (&p, &n) in prev_unpadded_wet.iter().zip(&aligned_new) {
        let p64 = f64::from(p);
        let n64 = f64::from(n);
        dot += p64 * n64;
        e_prev += p64 * p64;
        e_new += n64 * n64;
    }
    if e_prev <= 1.0e-18 || e_new <= 1.0e-18 {
        return Err(AlignError::InvalidAlignmentInput(
            "differential mic-pad probe has near-zero energy".to_string(),
        ));
    }
    let g_pad = (dot / e_prev).abs();
    if !(1.0e-3..=2.0).contains(&g_pad) {
        return Err(AlignError::InvalidAlignmentInput(format!(
            "differential mic-pad ratio {g_pad:.6} outside valid range [0.001, 2.0]"
        )));
    }
    let coh_sq = ((dot * dot) / (e_prev * e_new)).clamp(0.0, 1.0);
    Ok(DifferentialPadCalibration {
        pad_gain_linear: g_pad,
        pad_attenuation_db: 20.0 * g_pad.log10(),
        relative_lag_samples: align.total_lag_samples,
        coherence_sq: coh_sq,
    })
}

/// Rescales a padded wet capture back into the original unpadded calibration frame by multiplying
/// by $\hat{G}_{\text{pad}}^{-1}$ (`ref[train.capture.calib.clean-pad]`).
pub fn rescale_padded_stem(
    padded_samples: &[f32],
    pad_gain_linear: f64,
) -> Result<Vec<f32>, AlignError> {
    if !pad_gain_linear.is_finite() || !(1.0e-3..=2.0).contains(&pad_gain_linear) {
        return Err(AlignError::InvalidAlignmentInput(format!(
            "invalid pad_gain_linear {pad_gain_linear}"
        )));
    }
    let inv = 1.0 / pad_gain_linear;
    Ok(padded_samples
        .iter()
        .map(|&s| {
            if s.is_finite() {
                (f64::from(s) * inv).clamp(-1.0e4, 1.0e4) as f32
            } else {
                0.0
            }
        })
        .collect())
}

/// Evaluates live `< 0.05 dB` acoustic null homing when returning a physical knob to `StarCenter`
/// (`ref[train.capture.planner.null-homing]`).
pub fn evaluate_acoustic_null_homing(
    star_center_ref_wet: &[f32],
    live_probe_wet: &[f32],
    noise_floor_rms: f64,
) -> Result<NullHomingReport, AlignError> {
    let align = estimate_multiregime_alignment(star_center_ref_wet, live_probe_wet, 48_000, 8)?;
    let aligned_live =
        realign_fractional_kaiser64(live_probe_wet, align.total_lag_samples, 0.0, false);
    let n = star_center_ref_wet.len() as f64;
    let mut e_ref = 0.0_f64;
    let mut e_live = 0.0_f64;
    let mut e_err = 0.0_f64;
    for (&r, &l) in star_center_ref_wet.iter().zip(&aligned_live) {
        let r64 = f64::from(r);
        let l64 = f64::from(l);
        e_ref += r64 * r64;
        e_live += l64 * l64;
        let diff = l64 - r64;
        e_err += diff * diff;
    }
    if e_ref <= 1.0e-18 {
        return Err(AlignError::InvalidAlignmentInput(
            "star_center_ref_wet has near-zero energy".to_string(),
        ));
    }
    let floor_var = if noise_floor_rms.is_finite() && noise_floor_rms > 0.0 {
        2.0 * n * noise_floor_rms * noise_floor_rms
    } else {
        0.0
    };
    let clean_err = (e_err - floor_var).max(0.0);
    let ratio_rms = (clean_err / e_ref).sqrt();
    let null_err_db = 20.0 * (1.0 + ratio_rms).log10();
    let gain_delta_db = 10.0 * (e_live.max(1.0e-30) / e_ref).log10();
    let error_db = null_err_db.max(gain_delta_db.abs());
    let residual_esr_db = if clean_err > 1.0e-30 {
        10.0 * (clean_err / e_ref).log10()
    } else {
        -160.0
    };
    Ok(NullHomingReport {
        homed: error_db < ACOUSTIC_NULL_HOMING_MAX_DB,
        error_db,
        gain_delta_db,
        residual_esr_db,
    })
}

/// Evaluates the pre-stem `<= 0.15 dB` thermal/sag recovery gate back to `StarCenter`
/// (`ref[train.capture.planner.null-homing]`).
pub fn evaluate_thermal_recovery_gate(
    star_center_ref_wet: &[f32],
    current_return_wet: &[f32],
    noise_floor_rms: f64,
) -> Result<ThermalRecoveryReport, AlignError> {
    let report =
        evaluate_acoustic_null_homing(star_center_ref_wet, current_return_wet, noise_floor_rms)?;
    Ok(ThermalRecoveryReport {
        recovered: report.error_db <= THERMAL_RECOVERY_MAX_DB,
        drift_db: report.error_db,
    })
}

/// Evaluates whether a live post-stimulus reverb/echo tail has decayed within `+1.0 dB` of the
/// calibrated noise floor $\epsilon_{\text{floor}}^2$ (`ref[train.capture.excitation.live-tail]`).
#[must_use]
pub fn evaluate_live_decay_tail(
    tail_samples: &[f32],
    noise_floor_dbfs: f32,
    elapsed_tail_samples: usize,
    max_tail_samples: usize,
) -> DecayTailDecision {
    let win_len = tail_samples.len().min(960);
    let tail_rms_dbfs = if win_len == 0 {
        -160.0_f32
    } else {
        let slice = &tail_samples[tail_samples.len() - win_len..];
        let mean_sq = slice
            .iter()
            .map(|&s| {
                let v = if s.is_finite() { f64::from(s) } else { 0.0 };
                v * v
            })
            .sum::<f64>()
            / (win_len as f64);
        if mean_sq > 1.0e-16 {
            (10.0 * mean_sq.log10()) as f32
        } else {
            -160.0
        }
    };
    let safe_floor = if noise_floor_dbfs.is_finite() {
        noise_floor_dbfs.clamp(-160.0, 0.0)
    } else {
        -100.0
    };
    let target_dbfs = safe_floor + (LIVE_DECAY_TAIL_MARGIN_DB as f32);
    if tail_rms_dbfs <= target_dbfs {
        DecayTailDecision::Complete {
            tail_rms_dbfs,
            ceiling_reached: false,
        }
    } else if elapsed_tail_samples >= max_tail_samples {
        DecayTailDecision::Complete {
            tail_rms_dbfs,
            ceiling_reached: true,
        }
    } else {
        DecayTailDecision::Extend {
            tail_rms_dbfs,
            target_dbfs,
        }
    }
}
