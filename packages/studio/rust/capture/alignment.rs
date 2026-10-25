//! $O(N \log N)$ `realfft` + `ndarray` cross-correlation & parabolic sub-sample latency alignment,
//! FFT fractional-sample phase-ramp shift, locked Step-1 bypass delay/reverb alignment, sliding
//! 50 ms quiet-window noise floor telemetry, and hot-corner nonlinearity analysis.

use std::f32::consts::TAU;

use ndarray::ArrayView1;
use realfft::RealFftPlanner;
use realfft::num_complex::Complex;
use serde::{Deserialize, Serialize};

use crate::error::{StudioError, StudioResult};

/// Measured alignment and signal quality metrics for a captured wet stem.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignmentReport {
    /// Integer coarse lag in samples (`wet` delayed relative to `dry`).
    pub coarse_lag_samples: i32,
    /// Parabolic sub-sample refined lag in samples.
    pub subsample_lag_samples: f32,
    /// Normalized peak correlation in `[0.0, 1.0]`.
    pub peak_correlation: f32,
    /// Measured signal-to-noise ratio in dB.
    pub snr_db: f32,
    /// Measured silent-window noise floor in dBFS.
    pub noise_floor_dbfs: f32,
}

/// Telemetry from Step-2 Engaged Hot-Corner Verification (`0.0 dB` send preserved).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HotCornerAnalysis {
    /// Peak return level in dBFS.
    pub measured_return_peak_dbfs: f32,
    /// RMS return level in dBFS.
    pub measured_return_rms_dbfs: f32,
    /// Net RMS gain over Step-1 unity reference in dB.
    pub net_gain_over_unity_db: f32,
    /// Estimated crest-factor compression ratio (`>= 0.1`).
    pub compression_ratio: f32,
    /// Fraction of samples exceeding `-0.05 dBFS` (`[0.0, 1.0]`).
    pub clipped_sample_fraction: f32,
}

#[inline]
fn array_rms(slice: &[f32]) -> f32 {
    if slice.is_empty() {
        return 0.0;
    }
    let view = ArrayView1::from(slice);
    (view.dot(&view) / (slice.len() as f32)).sqrt()
}

#[inline]
fn array_l2_norm(slice: &[f32]) -> f32 {
    if slice.is_empty() {
        return 0.0;
    }
    let view = ArrayView1::from(slice);
    view.dot(&view).sqrt()
}

/// Measures the RMS `noise_floor_dbfs` by scanning sliding `50 ms` sub-windows across silent lead/tail
/// regions using `ndarray` dot products (bounded by `baseline_noise_floor_dbfs + 12 dB` when calibrated
/// so delay/reverb tails are not mistaken for broadband noise floor).
pub fn measure_silent_window_noise_floor(
    wet: &[f32],
    sample_rate_hz: u32,
    baseline_noise_floor_dbfs: Option<f32>,
) -> f32 {
    if wet.is_empty() {
        return -120.0;
    }
    let n = wet.len();
    let edge_win = (n / 16).max(1);
    let sub_win = ((0.050 * (sample_rate_hz.max(8_000) as f32)).round() as usize)
        .max(8)
        .min(edge_win);

    let mut min_rms = f32::INFINITY;
    for region in [&wet[..edge_win], &wet[n.saturating_sub(edge_win)..]] {
        if region.len() >= sub_win {
            let step = (sub_win / 2).max(1);
            let mut start = 0;
            while start + sub_win <= region.len() {
                let rms = array_rms(&region[start..start + sub_win]);
                min_rms = min_rms.min(rms);
                start += step;
            }
        } else {
            min_rms = min_rms.min(array_rms(region));
        }
    }

    let mut dbfs = if !min_rms.is_finite() || min_rms <= 1e-6 {
        -120.0
    } else {
        (20.0 * min_rms.log10()).clamp(-120.0, 0.0)
    };
    if let Some(base) = baseline_noise_floor_dbfs
        && base < -20.0
    {
        dbfs = dbfs.min(base + 12.0);
    }
    dbfs
}

/// Computes normalized cross-correlation $R(\tau) = \frac{\sum_n x[n] y[n + \tau]}{\|x\|_2 \|y\|_2}$
/// for $\tau \in [-\text{max\_lag}, +\text{max\_lag}]$ in $O(N \log N)$ via `realfft`.
fn fft_normalized_cross_correlation(dry: &[f32], wet: &[f32], max_lag: usize) -> Vec<f32> {
    let n = dry.len().min(wet.len());
    let norm_denom = (array_l2_norm(&dry[..n]) * array_l2_norm(&wet[..n])).max(1e-12);
    let n_fft = (2 * n).next_power_of_two().max(64);

    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(n_fft);
    let c2r = planner.plan_fft_inverse(n_fft);

    let mut dry_pad = vec![0.0_f32; n_fft];
    let mut wet_pad = vec![0.0_f32; n_fft];
    dry_pad[..n].copy_from_slice(&dry[..n]);
    wet_pad[..n].copy_from_slice(&wet[..n]);

    let mut dry_spec = r2c.make_output_vec();
    let mut wet_spec = r2c.make_output_vec();
    let _ = r2c.process(&mut dry_pad, &mut dry_spec);
    let _ = r2c.process(&mut wet_pad, &mut wet_spec);

    // Cross-power spectrum: W[k] * conj(D[k]) peaks at index +lag when wet is delayed by +lag
    let mut cross_spec: Vec<Complex<f32>> = wet_spec
        .iter()
        .zip(dry_spec.iter())
        .map(|(&w, &d)| w * d.conj())
        .collect();
    if let Some(first) = cross_spec.first_mut() {
        first.im = 0.0;
    }
    if let Some(last) = cross_spec.last_mut() {
        last.im = 0.0;
    }

    let mut corr_full = c2r.make_output_vec();
    let _ = c2r.process(&mut cross_spec, &mut corr_full);

    let scale = 1.0 / ((n_fft as f32) * norm_denom);
    let mut corr_map = vec![0.0_f32; max_lag * 2 + 1];
    for (idx, lag) in (-(max_lag as i32)..=(max_lag as i32)).enumerate() {
        let fft_idx = if lag >= 0 {
            (lag as usize) % n_fft
        } else {
            (n_fft as isize + (lag as isize)).rem_euclid(n_fft as isize) as usize
        };
        corr_map[idx] = corr_full[fft_idx] * scale;
    }
    corr_map
}

/// Applies an exact sub-sample time shift (`lag_samples`) to `wet` using integer sample shifting
/// followed by a `realfft` linear-phase fractional shift ($e^{+j 2\pi k \delta / N}$).
pub fn apply_subsample_shift_fft(wet: &[f32], lag_samples: f32) -> Vec<f32> {
    let n = wet.len();
    if n == 0 {
        return Vec::new();
    }
    let int_lag = lag_samples.round() as i32;
    let frac = lag_samples - (int_lag as f32);

    let mut shifted = vec![0.0_f32; n];
    for (i, out_s) in shifted.iter_mut().enumerate() {
        let src = (i as i32) + int_lag;
        if src >= 0 && (src as usize) < n {
            *out_s = wet[src as usize];
        }
    }
    if frac.abs() < 1e-4 {
        return shifted;
    }

    let n_fft = n.next_power_of_two().max(64);
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(n_fft);
    let c2r = planner.plan_fft_inverse(n_fft);

    let mut buf = vec![0.0_f32; n_fft];
    buf[..n].copy_from_slice(&shifted);
    let mut spec = r2c.make_output_vec();
    let _ = r2c.process(&mut buf, &mut spec);

    let n_bins = spec.len();
    for (k, bin) in spec.iter_mut().enumerate() {
        // Zero Nyquist imaginary part for valid ComplexToReal input
        if k == 0 || (n_fft.is_multiple_of(2) && k + 1 == n_bins) {
            bin.im = 0.0;
            continue;
        }
        let angle = TAU * (k as f32) * frac / (n_fft as f32);
        let rot = Complex::new(angle.cos(), angle.sin());
        *bin *= rot;
    }

    let mut time_out = c2r.make_output_vec();
    let _ = c2r.process(&mut spec, &mut time_out);
    let inv_n = 1.0 / (n_fft as f32);
    time_out.truncate(n);
    for s in &mut time_out {
        *s *= inv_n;
    }
    time_out
}

/// Aligns `wet` to `dry` using $O(N \log N)$ `realfft` normalized cross-correlation + parabolic
/// sub-sample refinement, or locks to `fixed_latency_samples` when calibrated via Step-1 Bypass Reference.
pub fn align_hardware_latency(
    dry: &[f32],
    wet: &[f32],
    max_lag_samples: usize,
    fixed_latency_samples: Option<i32>,
    sample_rate_hz: u32,
    baseline_noise_floor_dbfs: Option<f32>,
) -> StudioResult<AlignmentReport> {
    if dry.is_empty() || wet.is_empty() || dry.len() != wet.len() {
        return Err(StudioError::Audio(
            "dry and wet signals must be non-empty and equal length".to_string(),
        ));
    }
    let n = dry.len();
    let max_lag = max_lag_samples.min(n / 4).max(1);
    let corr_map = fft_normalized_cross_correlation(dry, wet, max_lag);

    let mut best_lag: i32 = 0;
    let mut best_corr = f32::NEG_INFINITY;
    for (idx, &norm_c) in corr_map.iter().enumerate() {
        let lag = (idx as i32) - (max_lag as i32);
        if norm_c > best_corr {
            best_corr = norm_c;
            best_lag = lag;
        }
    }

    let (resolved_lag, frac_offset, resolved_corr) = if let Some(locked) = fixed_latency_samples {
        let clamped_idx =
            (locked + (max_lag as i32)).clamp(0, (corr_map.len() as i32) - 1) as usize;
        (locked, 0.0_f32, corr_map[clamped_idx].abs().clamp(0.0, 1.0))
    } else {
        let best_idx = (best_lag + (max_lag as i32)) as usize;
        let mut frac = 0.0_f32;
        if best_idx > 0 && best_idx + 1 < corr_map.len() {
            let y_prev = corr_map[best_idx - 1];
            let y_curr = corr_map[best_idx];
            let y_next = corr_map[best_idx + 1];
            let denom = y_prev - 2.0 * y_curr + y_next;
            if denom.abs() > 1e-9 {
                frac = (0.5 * (y_prev - y_next) / denom).clamp(-0.5, 0.5);
            }
        }
        (best_lag, frac, best_corr.abs().clamp(0.0, 1.0))
    };

    let noise_floor_dbfs =
        measure_silent_window_noise_floor(wet, sample_rate_hz, baseline_noise_floor_dbfs);
    let safe_noise_power = 10.0_f32.powf(noise_floor_dbfs / 10.0).max(1e-12);
    let wet_rms = array_rms(wet);
    let signal_power = (wet_rms * wet_rms).max(1e-12);
    let snr_db = 10.0 * (signal_power / safe_noise_power).log10();

    Ok(AlignmentReport {
        coarse_lag_samples: resolved_lag,
        subsample_lag_samples: (resolved_lag as f32) + frac_offset,
        peak_correlation: resolved_corr,
        snr_db,
        noise_floor_dbfs,
    })
}

/// Estimates coarse and parabolic sub-sample lag between `dry` and `wet` signals plus SNR.
pub fn estimate_subsample_lag_and_snr(
    dry: &[f32],
    wet: &[f32],
    max_lag_samples: usize,
) -> StudioResult<AlignmentReport> {
    align_hardware_latency(dry, wet, max_lag_samples, None, 48_000, None)
}

/// Analyzes Step-2 engaged hot-corner return signal without reducing `send_gain_db`.
pub fn analyze_hot_corner_nonlinearity(
    dry_probe: &[f32],
    wet_return: &[f32],
    unity_ref_dbfs: f32,
) -> HotCornerAnalysis {
    let n = dry_probe.len().min(wet_return.len()).max(1);
    let wet_view = ArrayView1::from(&wet_return[..n]);
    let dry_view = ArrayView1::from(&dry_probe[..n]);

    let wet_peak = wet_view.fold(1e-9_f32, |acc, &x| acc.max(x.abs()));
    let wet_rms = (wet_view.dot(&wet_view) / (n as f32)).sqrt().max(1e-9);
    let dry_peak = dry_view.fold(1e-9_f32, |acc, &x| acc.max(x.abs()));
    let dry_rms = (dry_view.dot(&dry_view) / (n as f32)).sqrt().max(1e-9);

    let peak_dbfs = (20.0 * wet_peak.log10()).clamp(-120.0, 6.0);
    let rms_dbfs = (20.0 * wet_rms.log10()).clamp(-120.0, 6.0);
    let dry_rms_dbfs = 20.0 * dry_rms.log10();
    let net_gain_over_unity_db = (rms_dbfs - dry_rms_dbfs) - unity_ref_dbfs;

    let dry_crest = (dry_peak / dry_rms).max(1.0);
    let wet_crest = (wet_peak / wet_rms).max(1.0);
    let compression_ratio = (dry_crest / wet_crest).max(0.1);

    let clip_lin = 10.0_f32.powf(-0.05 / 20.0);
    let clipped_count = wet_return.iter().filter(|&&x| x.abs() >= clip_lin).count();
    let clipped_sample_fraction = (clipped_count as f32) / (wet_return.len().max(1) as f32);

    HotCornerAnalysis {
        measured_return_peak_dbfs: peak_dbfs,
        measured_return_rms_dbfs: rms_dbfs,
        net_gain_over_unity_db,
        compression_ratio,
        clipped_sample_fraction,
    }
}
