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

/// Result of verifying (and auto-polishing if needed) a post-shift aligned wet capture.
#[derive(Debug, Clone, PartialEq)]
pub struct PostAlignmentVerification {
    /// Aligned wet waveform after any residual multi-sample polish.
    pub aligned_wet: Vec<f32>,
    /// Total sub-sample lag applied (`initial_lag_samples + applied_polish_samples`).
    pub total_lag_samples: f32,
    /// Integer residual lag (`samples`) remaining after verification (`0` when aligned).
    pub residual_lag_samples: i32,
    /// Peak normalized cross-correlation in `[-16, +16]` against the best reference.
    pub verification_correlation: f32,
    /// Reference signal that yielded `verification_correlation` (`"dry"`, `"dry_lp"`, `"dry_hp"`, `"fuzz"`, or `"star_center"`).
    pub best_reference: &'static str,
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

/// Builds a causal clipped fuzz proxy `LPF_800(tanh(10 * HPF_80(dry)))` so extreme comparator/fuzz
/// pedals with `< 0.35` linear correlation still lock sub-sample USB stream startup jitter while
/// matching the causal group delay of hardware fuzz tone stacks.
fn build_clipped_fuzz_proxy(dry: &[f32], sample_rate_hz: u32) -> Vec<f32> {
    let sr = (sample_rate_hz.max(8_000)) as f32;
    let a_hp = (-TAU * 80.0 / sr).exp();
    let b_hp = 1.0 - a_hp;
    let a_lp = (-TAU * 800.0 / sr).exp();
    let b_lp = 1.0 - a_lp;
    let mut s_hp = 0.0_f32;
    let mut s_lp = 0.0_f32;
    let mut out = Vec::with_capacity(dry.len());
    for &x in dry {
        s_hp = a_hp * s_hp + b_hp * x;
        let x_hp = x - s_hp;
        let clipped = (10.0 * x_hp).tanh();
        s_lp = a_lp * s_lp + b_lp * clipped;
        out.push(s_lp);
    }
    out
}

/// Builds a zero-phase 1-pole forward-backward (`filtfilt`) low-pass helper at `cutoff_hz`
/// used to construct the complementary zero-phase high-pass proxy (`build_zerophase_hp_proxy`).
fn build_zerophase_lp_proxy(dry: &[f32], sample_rate_hz: u32, cutoff_hz: f32) -> Vec<f32> {
    let sr = (sample_rate_hz.max(8_000)) as f32;
    let a_lp = (-TAU * cutoff_hz / sr).exp();
    let b_lp = 1.0 - a_lp;
    let mut s_fwd = 0.0_f32;
    let mut out = Vec::with_capacity(dry.len());
    for &x in dry {
        s_fwd = a_lp * s_fwd + b_lp * x;
        out.push(s_fwd);
    }
    let mut s_rev = 0.0_f32;
    for y in out.iter_mut().rev() {
        s_rev = a_lp * s_rev + b_lp * *y;
        *y = s_rev;
    }
    out
}

/// Builds a zero-phase high-pass proxy `x[n] - filtfilt_LPF(x[n])` at `cutoff_hz`
/// so steep high-pass/band-pass filter stems (`Mode = HP/BP`) correlate cleanly inside
/// `locked ± 48` without group-delay bias.
fn build_zerophase_hp_proxy(dry: &[f32], sample_rate_hz: u32, cutoff_hz: f32) -> Vec<f32> {
    let lp = build_zerophase_lp_proxy(dry, sample_rate_hz, cutoff_hz);
    dry.iter().zip(lp.iter()).map(|(&x, &l)| x - l).collect()
}

/// Locates the leading Pulse-1 chirp onset window (`~12 ms`) in `dry` so delay/reverb stems with
/// heavy repeats (`Mix < 100%`) can lock sub-sample USB startup jitter on the first direct pulse
/// before any echo tap (`>= 12 ms`) arrives, and so `100% Mix` kill-dry delay stems can be
/// identified by their silent Pulse-1 return window.
fn pulse1_onset_window(dry: &[f32], sample_rate_hz: u32) -> Option<std::ops::Range<usize>> {
    let n = dry.len();
    if n < 128 {
        return None;
    }
    let fs = (sample_rate_hz.max(8_000)) as f32;
    let search_lim = ((0.50 * fs).round() as usize).min(n / 2).max(32);
    let idx0 = dry[..search_lim].iter().position(|&x| x.abs() > 0.35)?;
    let p1_start = idx0.saturating_sub(8);
    let chirp_len = ((0.012 * fs).round() as usize).clamp(32, (n / 8).max(32));
    let p1_end = (p1_start + chirp_len).min(n);
    if p1_end.saturating_sub(p1_start) >= 32 {
        Some(p1_start..p1_end)
    } else {
        None
    }
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
        let abs_c = norm_c.abs();
        if abs_c > best_corr {
            best_corr = abs_c;
            best_lag = lag;
        }
    }

    let best_idx = (best_lag + (max_lag as i32)) as usize;
    let mut best_frac = 0.0_f32;
    if best_idx > 0 && best_idx + 1 < corr_map.len() {
        let y_prev = corr_map[best_idx - 1].abs();
        let y_curr = corr_map[best_idx].abs();
        let y_next = corr_map[best_idx + 1].abs();
        let denom = y_prev - 2.0 * y_curr + y_next;
        if denom.abs() > 1e-9 {
            best_frac = (0.5 * (y_prev - y_next) / denom).clamp(-0.5, 0.5);
        }
    }

    let wet_rms = array_rms(wet);

    let (resolved_lag, frac_offset, resolved_corr) = if let Some(locked) = fixed_latency_samples {
        let search_window = |map: &[f32], radius: i32| -> (i32, f32) {
            let lo = locked - radius;
            let hi = locked + radius;
            let mut best_l = locked;
            let mut best_c = f32::NEG_INFINITY;
            for lag in lo..=hi {
                let idx_i32 = lag + (max_lag as i32);
                if idx_i32 >= 0 && (idx_i32 as usize) < map.len() {
                    let abs_c = map[idx_i32 as usize].abs();
                    if abs_c > best_c {
                        best_c = abs_c;
                        best_l = lag;
                    }
                }
            }
            // If best_l landed on the window boundary (lo or hi) with strong correlation and the
            // slope is still rising outward, follow the monotonic main-lobe slope up to 16 samples
            // to its true local maximum so we never clip on a boundary without jumping to a sidelobe.
            if best_l == lo && best_c > 0.35 {
                for step in 1..=16 {
                    let cand = lo - step;
                    let idx_i32 = cand + (max_lag as i32);
                    if idx_i32 < 0 || (idx_i32 as usize) >= map.len() {
                        break;
                    }
                    let abs_c = map[idx_i32 as usize].abs();
                    if abs_c > best_c {
                        best_c = abs_c;
                        best_l = cand;
                    } else {
                        break;
                    }
                }
            } else if best_l == hi && best_c > 0.35 {
                for step in 1..=16 {
                    let cand = hi + step;
                    let idx_i32 = cand + (max_lag as i32);
                    if idx_i32 < 0 || (idx_i32 as usize) >= map.len() {
                        break;
                    }
                    let abs_c = map[idx_i32 as usize].abs();
                    if abs_c > best_c {
                        best_c = abs_c;
                        best_l = cand;
                    } else {
                        break;
                    }
                }
            }
            (best_l, best_c)
        };

        let (mut local_best_lag, mut local_best_corr) = search_window(&corr_map, 48);
        let mut active_map = corr_map;

        if local_best_corr <= 0.35 {
            for proxy in [
                build_clipped_fuzz_proxy(dry, sample_rate_hz),
                build_zerophase_hp_proxy(dry, sample_rate_hz, 1_500.0),
            ] {
                let p_map = fft_normalized_cross_correlation(&proxy, wet, max_lag);
                let (p_lag, p_corr) = search_window(&p_map, 48);
                if p_corr > local_best_corr {
                    local_best_corr = p_corr;
                    local_best_lag = p_lag;
                    active_map = p_map;
                }
            }
        }

        // Echo-free Pulse-1 onset fallback for delay/reverb pedals with heavy repeats (Mix < 100%)
        // where 11s of echo repeats dilute full-signal correlation below 0.35, while preserving
        // locked latency on 100% Mix kill-dry delay stems (where Pulse-1 wet RMS is silent).
        let mut p1_override: Option<(i32, f32, f32)> = None;
        if local_best_corr <= 0.35
            && let Some(p1_range) = pulse1_onset_window(dry, sample_rate_hz)
        {
            let d_slice = &dry[p1_range.clone()];
            let d_view = ArrayView1::from(d_slice);
            let d_energy = d_view.dot(&d_view);
            let p1_len = p1_range.len();
            let radius = 48_i32;
            let mut p1_corrs = vec![0.0_f32; (2 * radius + 1) as usize];
            let mut best_p1_lag = locked;
            let mut best_p1_c = 0.0_f32;
            for offset in -radius..=radius {
                let cand_lag = locked + offset;
                let w_start_i32 = (p1_range.start as i32) + cand_lag;
                let w_end_i32 = w_start_i32 + (p1_len as i32);
                if w_start_i32 >= 0 && (w_end_i32 as usize) <= n {
                    let w_slice = &wet[(w_start_i32 as usize)..(w_end_i32 as usize)];
                    let w_p1_rms = array_rms(w_slice);
                    if w_p1_rms > 0.02 * wet_rms && w_p1_rms > 1e-4 {
                        let w_view = ArrayView1::from(w_slice);
                        let denom = (d_energy * w_view.dot(&w_view)).sqrt();
                        let c = if denom > 1e-12 {
                            (d_view.dot(&w_view) / denom).abs()
                        } else {
                            0.0
                        };
                        p1_corrs[(offset + radius) as usize] = c;
                        if c > best_p1_c {
                            best_p1_c = c;
                            best_p1_lag = cand_lag;
                        }
                    }
                }
            }
            if best_p1_c > 0.65 {
                let best_off_idx = (best_p1_lag - locked + radius) as usize;
                let mut p1_frac = 0.0_f32;
                if best_off_idx > 0 && best_off_idx + 1 < p1_corrs.len() {
                    let y_prev = p1_corrs[best_off_idx - 1];
                    let y_curr = p1_corrs[best_off_idx];
                    let y_next = p1_corrs[best_off_idx + 1];
                    let denom = y_prev - 2.0 * y_curr + y_next;
                    if denom.abs() > 1e-9 {
                        p1_frac = (0.5 * (y_prev - y_next) / denom).clamp(-0.5, 0.5);
                    }
                }
                p1_override = Some((best_p1_lag, p1_frac, best_p1_c.clamp(0.0, 1.0)));
            }
        }

        if let Some((p1_lag, p1_frac, p1_c)) = p1_override {
            (p1_lag, p1_frac, p1_c)
        } else if local_best_corr > 0.35 {
            let local_idx = (local_best_lag + (max_lag as i32)) as usize;
            let mut local_frac = 0.0_f32;
            if local_idx > 0 && local_idx + 1 < active_map.len() {
                let y_prev = active_map[local_idx - 1].abs();
                let y_curr = active_map[local_idx].abs();
                let y_next = active_map[local_idx + 1].abs();
                let denom = y_prev - 2.0 * y_curr + y_next;
                if denom.abs() > 1e-9 {
                    local_frac = (0.5 * (y_prev - y_next) / denom).clamp(-0.5, 0.5);
                }
            }
            (local_best_lag, local_frac, local_best_corr.clamp(0.0, 1.0))
        } else {
            let clamped_idx =
                (locked + (max_lag as i32)).clamp(0, (active_map.len() as i32) - 1) as usize;
            (
                locked,
                0.0_f32,
                active_map[clamped_idx].abs().clamp(0.0, 1.0),
            )
        }
    } else {
        (best_lag, best_frac, best_corr.clamp(0.0, 1.0))
    };

    let noise_floor_dbfs =
        measure_silent_window_noise_floor(wet, sample_rate_hz, baseline_noise_floor_dbfs);
    let safe_noise_power = 10.0_f32.powf(noise_floor_dbfs / 10.0).max(1e-12);
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

fn local_normalized_xcorr(
    reference: &[f32],
    target: &[f32],
    max_local_lag: i32,
) -> (i32, f32, f32) {
    let n = reference.len().min(target.len());
    if n < 64 {
        return (0, 0.0, 0.0);
    }
    let r_max = max_local_lag.max(1) as usize;
    let mut corrs = vec![0.0_f32; 2 * r_max + 1];
    let mut best_k = 0_i32;
    let mut best_c = 0.0_f32;
    for k in -(r_max as i32)..=(r_max as i32) {
        let (r_slice, t_slice) = if k >= 0 {
            let ku = k as usize;
            if ku >= n {
                continue;
            }
            (&reference[..n - ku], &target[ku..n])
        } else {
            let ku = (-k) as usize;
            if ku >= n {
                continue;
            }
            (&reference[ku..n], &target[..n - ku])
        };
        let r_view = ArrayView1::from(r_slice);
        let t_view = ArrayView1::from(t_slice);
        let dot = r_view.dot(&t_view);
        let denom = (r_view.dot(&r_view) * t_view.dot(&t_view)).sqrt();
        let c = if denom > 1e-12 {
            (dot / denom).abs()
        } else {
            0.0
        };
        corrs[(k + (r_max as i32)) as usize] = c;
        if c > best_c {
            best_c = c;
            best_k = k;
        }
    }
    let idx = (best_k + (r_max as i32)) as usize;
    let mut frac = 0.0_f32;
    if idx > 0 && idx + 1 < corrs.len() {
        let y_prev = corrs[idx - 1];
        let y_curr = corrs[idx];
        let y_next = corrs[idx + 1];
        let denom = y_prev - 2.0 * y_curr + y_next;
        if denom.abs() > 1e-9 {
            frac = (0.5 * (y_prev - y_next) / denom).clamp(-0.5, 0.5);
        }
    }
    (best_k, best_c.clamp(0.0, 1.0), frac)
}

/// Verifies a post-shift aligned wet capture over `[-16, +16]` samples against broadband `dry`
/// first. When broadband `dry` already has strong correlation (`c_dry > 0.40`), uses `dry`
/// exclusively so causal tone-stack phase shifts on amplifiers/preamps never trigger false shifts.
/// When broadband `dry` correlation is low (`c_dry <= 0.40`, e.g., extreme comparator fuzz or
/// steep `Mode = BP / HP` filter stems), evaluates causal clipped fuzz (`fuzz`), zero-phase
/// `1.5 kHz` HP (`dry_hp`), `star_center` (or same-switch reference stem), and the stationary
/// high-level Velocity Step 6 burst window (`step6_dry_hp`, `step6_ref`). If a residual integer
/// misalignment (`|best_k| >= 1`) is detected with high confidence (`best_corr > 0.45`),
/// automatically applies a secondary sub-sample FFT phase-ramp polish.
pub fn verify_and_polish_post_alignment(
    dry: &[f32],
    star_center: Option<&[f32]>,
    aligned_wet: &[f32],
    initial_lag_samples: f32,
    sample_rate_hz: u32,
) -> PostAlignmentVerification {
    let n = dry.len().min(aligned_wet.len());
    let (k_dry, c_dry, frac_dry) = local_normalized_xcorr(&dry[..n], &aligned_wet[..n], 16);

    let mut best_name = "dry";
    let mut best_ref = dry;
    let mut best_range = 0..n;
    let mut best_k = k_dry;
    let mut best_c = c_dry;
    let mut best_frac = frac_dry;

    let hp_proxy;
    let fuzz_proxy;
    if c_dry <= 0.40 {
        let wet_full_rms = array_rms(&aligned_wet[..n]);
        if let Some(p1_range) = pulse1_onset_window(dry, sample_rate_hz)
            && p1_range.end <= n
        {
            let dry_p1_rms = array_rms(&dry[p1_range.clone()]);
            let wet_p1_rms = array_rms(&aligned_wet[p1_range.clone()]);
            // If dry has Pulse 1 (> 0.15 RMS) while aligned_wet is silent during Pulse 1
            // (< 1.5% of full wet RMS), this is a 100% Mix kill-dry delay stem whose output is
            // intentionally delayed beyond Pulse 1—preserve the locked hardware latency untouched.
            if dry_p1_rms > 0.15 && wet_full_rms > 1e-4 && wet_p1_rms < 0.015 * wet_full_rms {
                return PostAlignmentVerification {
                    aligned_wet: aligned_wet.to_vec(),
                    total_lag_samples: initial_lag_samples,
                    residual_lag_samples: 0,
                    verification_correlation: c_dry,
                    best_reference: "kill_dry_locked",
                };
            }
            if wet_p1_rms > 0.02 * wet_full_rms && wet_p1_rms > 1e-4 {
                let (k_p1, c_p1, frac_p1) = local_normalized_xcorr(
                    &dry[p1_range.clone()],
                    &aligned_wet[p1_range.clone()],
                    16,
                );
                if c_p1 > 0.65 {
                    best_name = "pulse1_dry";
                    best_ref = dry;
                    best_range = p1_range;
                    best_k = k_p1;
                    best_c = c_p1;
                    best_frac = frac_p1;
                }
            }
        }

        if best_name != "pulse1_dry" {
            hp_proxy = build_zerophase_hp_proxy(dry, sample_rate_hz, 1_500.0);
            fuzz_proxy = build_clipped_fuzz_proxy(dry, sample_rate_hz);

            let wlen = ((0.24 * (sample_rate_hz as f32)).round() as usize)
                .min(n / 4)
                .max(1);
            let lo = (n * 35) / 100;
            let hi = ((n * 68) / 100).min(n);
            let mut burst_start = lo;
            if hi > lo + wlen {
                let stride = ((sample_rate_hz as usize) / 100).max(1);
                let mut best_energy = -1.0_f64;
                let mut s = lo;
                while s + wlen <= hi {
                    let energy: f64 = dry[s..s + wlen]
                        .iter()
                        .map(|&x| (x as f64) * (x as f64))
                        .sum();
                    if energy > best_energy {
                        best_energy = energy;
                        burst_start = s;
                    }
                    s += stride;
                }
            }
            let burst_end = (burst_start + wlen).min(n);
            let has_burst_win = burst_end.saturating_sub(burst_start) >= 512;

            let mut candidates: Vec<(&'static str, &[f32], std::ops::Range<usize>)> =
                vec![("fuzz", &fuzz_proxy, 0..n), ("dry_hp", &hp_proxy, 0..n)];
            if has_burst_win {
                candidates.push(("burst_dry_hp", &hp_proxy, burst_start..burst_end));
            }
            if let Some(center) = star_center
                && center.len() >= n
            {
                candidates.push(("star_center", center, 0..n));
                if has_burst_win {
                    candidates.push(("burst_ref", center, burst_start..burst_end));
                }
            }

            for (name, ref_sig, range) in candidates {
                let (k, c, frac) = local_normalized_xcorr(
                    &ref_sig[range.clone()],
                    &aligned_wet[range.clone()],
                    16,
                );
                if c > best_c {
                    best_c = c;
                    best_k = k;
                    best_frac = frac;
                    best_name = name;
                    best_ref = ref_sig;
                    best_range = range;
                }
            }
        }
    }

    if best_k != 0 && best_c > 0.45 {
        let polish_shift = (best_k as f32) + best_frac;
        let polished = apply_subsample_shift_fft(aligned_wet, polish_shift);
        let (new_k, new_c, _) =
            local_normalized_xcorr(&best_ref[best_range.clone()], &polished[best_range], 16);
        PostAlignmentVerification {
            aligned_wet: polished,
            total_lag_samples: initial_lag_samples + polish_shift,
            residual_lag_samples: new_k,
            verification_correlation: new_c,
            best_reference: best_name,
        }
    } else {
        PostAlignmentVerification {
            aligned_wet: aligned_wet.to_vec(),
            total_lag_samples: initial_lag_samples,
            residual_lag_samples: best_k,
            verification_correlation: best_c,
            best_reference: best_name,
        }
    }
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

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use crate::capture::excitation::generate_excitation_signal;

    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(24))]

        #[test]
        fn prop_align_hardware_latency_recovers_arbitrary_lag_gain_and_polarity(
            true_lag in 0_usize..=28,
            gain in 0.15_f32..=1.0,
            invert in proptest::bool::ANY,
        ) {
            let dry = generate_excitation_signal(48_000, 2_048);
            let sign = if invert { -1.0_f32 } else { 1.0_f32 };
            let mut wet = vec![0.0_f32; dry.len()];
            for i in true_lag..dry.len() {
                wet[i] = sign * gain * dry[i - true_lag];
            }
            let report = align_hardware_latency(&dry, &wet, 48, None, 48_000, None)
                .expect("align hardware latency");
            prop_assert_eq!(report.coarse_lag_samples, true_lag as i32);
            prop_assert!(report.peak_correlation > 0.85);
            prop_assert!(report.snr_db.is_finite());
        }

        #[test]
        fn prop_apply_subsample_shift_preserves_length_and_finiteness(
            samples in proptest::collection::vec(-1.0_f32..1.0_f32, 16..128),
            shift in -8.0_f32..8.0_f32,
        ) {
            let shifted = apply_subsample_shift_fft(&samples, shift);
            prop_assert_eq!(shifted.len(), samples.len());
            prop_assert!(shifted.iter().all(|s| s.is_finite()));
        }
    }

    #[test]
    fn test_align_hardware_latency_polarity_inversion_and_locked_delay_lag() {
        let dry = generate_excitation_signal(48_000, 4_096);
        let true_lag = 19_usize;
        let mut wet_inv = vec![0.0_f32; dry.len()];
        for i in true_lag..dry.len() {
            wet_inv[i] = -0.85 * dry[i - true_lag];
        }

        let res = align_hardware_latency(&dry, &wet_inv, 64, None, 48_000, None)
            .expect("align inverted wet");
        assert_eq!(res.coarse_lag_samples, true_lag as i32);
        assert!(res.peak_correlation > 0.90);

        // Locked hardware latency on a kill-dry delay stem preserves the echo tap offset
        let hw_lag = 12_i32;
        let echo_tap = 200_usize;
        let mut wet_delay = vec![0.0_f32; dry.len()];
        let total_shift = (hw_lag as usize) + echo_tap;
        for i in total_shift..dry.len() {
            wet_delay[i] = 0.9 * dry[i - total_shift];
        }
        let locked = align_hardware_latency(&dry, &wet_delay, 64, Some(hw_lag), 48_000, None)
            .expect("locked align");
        assert_eq!(locked.coarse_lag_samples, hw_lag);

        // USB stream startup jitter (+/- 20 samples) around locked latency is recovered even under clipping
        let jittered_lag = 21_usize; // +9 samples jitter relative to hw_lag = 12
        let mut wet_jittered = vec![0.0_f32; dry.len()];
        for i in jittered_lag..dry.len() {
            wet_jittered[i] = (6.0 * dry[i - jittered_lag]).tanh();
        }
        let jitter_res =
            align_hardware_latency(&dry, &wet_jittered, 64, Some(hw_lag), 48_000, None)
                .expect("jittered align");
        assert_eq!(jitter_res.coarse_lag_samples, jittered_lag as i32);
        assert!(jitter_res.peak_correlation > 0.35);

        // Extreme comparator fuzz where linear dry correlation < 0.35 falls back to clipped fuzz proxy
        let dry_fuzz: Vec<f32> = (0..4_096)
            .map(|i| {
                let env = 0.5 + 0.5 * ((i as f32) * 0.019).sin() * ((i as f32) * 0.007).cos();
                let carrier = match i % 3 {
                    0 => 0.8_f32,
                    _ => -0.4_f32,
                };
                env * carrier
            })
            .collect();
        let fuzz_proxy = build_clipped_fuzz_proxy(&dry_fuzz, 48_000);
        let mut wet_fuzz = vec![0.0_f32; dry_fuzz.len()];
        wet_fuzz[jittered_lag..dry_fuzz.len()]
            .copy_from_slice(&fuzz_proxy[..(dry_fuzz.len() - jittered_lag)]);
        let fuzz_res = align_hardware_latency(&dry_fuzz, &wet_fuzz, 64, Some(hw_lag), 48_000, None)
            .expect("fuzz proxy align");
        assert_eq!(fuzz_res.coarse_lag_samples, jittered_lag as i32);
        assert!(fuzz_res.peak_correlation > 0.35);

        // Error on empty or mismatched lengths
        assert!(align_hardware_latency(&[], &[], 16, None, 48_000, None).is_err());
        assert!(align_hardware_latency(&[1.0, 2.0], &[1.0], 16, None, 48_000, None).is_err());
    }

    #[test]
    fn test_subsample_shift_silent_window_and_hot_corner_clipping() {
        assert!(apply_subsample_shift_fft(&[], 1.5).is_empty());
        let sig: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.1).sin()).collect();
        let shifted_int = apply_subsample_shift_fft(&sig, 4.0);
        assert_eq!(shifted_int.len(), 128);
        let shifted_frac = apply_subsample_shift_fft(&sig, 4.35);
        assert_eq!(shifted_frac.len(), 128);

        assert_eq!(measure_silent_window_noise_floor(&[], 48_000, None), -120.0);
        let mut ringing = vec![0.0_f32; 4_096];
        for s in &mut ringing[2_000..3_500] {
            *s = 0.25;
        }
        for s in &mut ringing[3_500..] {
            *s = 1e-4;
        }
        let nf = measure_silent_window_noise_floor(&ringing, 48_000, Some(-90.0));
        assert!(nf <= -75.0);

        let dry = generate_excitation_signal(48_000, 1_024);
        let hot_wet: Vec<f32> = dry.iter().map(|&x| (x * 4.0).clamp(-1.0, 1.0)).collect();
        let stats = analyze_hot_corner_nonlinearity(&dry, &hot_wet, 0.0);
        assert!(stats.measured_return_peak_dbfs > -0.2);
        assert!(stats.net_gain_over_unity_db > 2.0);
        assert!(stats.compression_ratio > 1.1);
        assert!(stats.clipped_sample_fraction > 0.01);
    }

    #[test]
    fn test_wide_usb_jitter_and_post_alignment_residual_polish() {
        let dry = generate_excitation_signal(48_000, 4_096);
        let locked_lag = 120_i32;
        let true_lag = 80_usize; // -40 samples relative to locked_lag (outside old +/-32 window)
        let hp_dry = build_zerophase_hp_proxy(&dry, 48_000, 1_500.0);
        let mut wet_hp = vec![0.0_f32; dry.len()];
        wet_hp[true_lag..dry.len()].copy_from_slice(&hp_dry[..(dry.len() - true_lag)]);

        let rep = align_hardware_latency(&dry, &wet_hp, 256, Some(locked_lag), 48_000, None)
            .expect("align wide USB jitter HP stem");
        assert_eq!(rep.coarse_lag_samples, true_lag as i32);
        assert!(rep.peak_correlation > 0.50);

        // Simulate a residual -9 sample misalignment on both broadband (> 0.40) and
        // low-broadband-correlation (<= 0.40) stems and verify post-alignment auto-polish fixes both
        let misshifted_hp = apply_subsample_shift_fft(&hp_dry, 9.0);
        let polished_hp =
            verify_and_polish_post_alignment(&dry, Some(&hp_dry), &misshifted_hp, 89.0, 48_000);
        assert_eq!(polished_hp.residual_lag_samples, 0);
        assert!((polished_hp.total_lag_samples - (true_lag as f32)).abs() < 0.25);
        assert!(polished_hp.verification_correlation > 0.50);

        let dry_low_corr: Vec<f32> = (0..4_096)
            .map(|i| {
                let env = 0.5 + 0.5 * ((i as f32) * 0.019).sin() * ((i as f32) * 0.007).cos();
                let carrier = if i % 3 == 0 { 0.8_f32 } else { -0.4_f32 };
                env * carrier
            })
            .collect();
        let fuzz_ref = build_clipped_fuzz_proxy(&dry_low_corr, 48_000);
        let misshifted_fuzz = apply_subsample_shift_fft(&fuzz_ref, 9.0);
        let polished_fuzz = verify_and_polish_post_alignment(
            &dry_low_corr,
            Some(&fuzz_ref),
            &misshifted_fuzz,
            89.0,
            48_000,
        );
        assert_eq!(polished_fuzz.residual_lag_samples, 0);
        assert!((polished_fuzz.total_lag_samples - (true_lag as f32)).abs() < 0.25);
        assert!(polished_fuzz.verification_correlation > 0.95);
    }

    #[test]
    fn test_delay_pedal_heavy_repeats_slapback_and_kill_dry_alignment() {
        let sr = 48_000_u32;
        let dry = generate_excitation_signal(sr, 96_000); // 2.0s signal
        let locked_lag = 300_i32;
        let true_hw_lag = 314_usize; // +14 samples USB startup jitter

        let sim_delay = |delay_samples: usize, feedback: f32, mix: f32| -> Vec<f32> {
            let n = dry.len();
            let mut x = vec![0.0_f32; n];
            x[true_hw_lag..n].copy_from_slice(&dry[..(n - true_hw_lag)]);
            let mut dline = vec![0.0_f32; n];
            for i in delay_samples..n {
                dline[i] = x[i - delay_samples] + feedback * dline[i - delay_samples];
            }
            x.iter()
                .zip(dline.iter())
                .map(|(&xi, &di)| (1.0 - mix) * xi + mix * di)
                .collect()
        };

        // 1. Heavy-mix / high-feedback delay (Mix = 82%, Delay = 180 ms, Feedback = 0.85)
        // Full-signal dry correlation is diluted below 0.35 by echo repeats, but Pulse-1 onset
        // locks exact hardware latency (314) before the first echo arrives.
        let wet_heavy = sim_delay(8_640, 0.85, 0.82);
        let rep_heavy = align_hardware_latency(&dry, &wet_heavy, 2_048, Some(locked_lag), sr, None)
            .expect("align heavy-repeat delay");
        assert_eq!(rep_heavy.coarse_lag_samples, true_hw_lag as i32);
        assert!(rep_heavy.peak_correlation > 0.65);

        // 2. Short slapback delay (Mix = 88%, Delay = 16 ms = 768 samples, Feedback = 0.70)
        let wet_slap = sim_delay(768, 0.70, 0.88);
        let rep_slap = align_hardware_latency(&dry, &wet_slap, 2_048, Some(locked_lag), sr, None)
            .expect("align slapback delay");
        assert_eq!(rep_slap.coarse_lag_samples, true_hw_lag as i32);
        assert!(rep_slap.peak_correlation > 0.65);

        // 3. 100% Mix kill-dry delay (Delay = 120 ms = 5,760 samples, Feedback = 0.50)
        // Preserves locked hardware latency (300) without stripping the echo tap offset, even if
        // verify_and_polish_post_alignment is called with a different star_center reference.
        let wet_kill_dry = sim_delay(5_760, 0.50, 1.00);
        let rep_kill =
            align_hardware_latency(&dry, &wet_kill_dry, 2_048, Some(locked_lag), sr, None)
                .expect("align kill-dry delay");
        assert_eq!(rep_kill.coarse_lag_samples, locked_lag);
        let aligned_kill = apply_subsample_shift_fft(&wet_kill_dry, rep_kill.subsample_lag_samples);
        let verified_kill = verify_and_polish_post_alignment(
            &dry,
            Some(&wet_heavy),
            &aligned_kill,
            rep_kill.subsample_lag_samples,
            sr,
        );
        assert_eq!(verified_kill.best_reference, "kill_dry_locked");
        assert_eq!(verified_kill.residual_lag_samples, 0);
        assert_eq!(
            verified_kill.total_lag_samples,
            rep_kill.subsample_lag_samples
        );
    }

    #[test]
    fn test_causal_fuzz_group_delay_bass_boost_sidelobe_and_tonestack_dry_gate() {
        let sr = 48_000_u32;
        let dry = generate_excitation_signal(sr, 48_000);
        let locked_lag = 200_i32;
        let true_hw_lag = 211_usize; // +11 samples USB jitter

        // 1. Causal fuzz pedal regression test (`duality_fuzz`):
        // Construct wet via an independent CAUSAL comparator + 1-pole 750 Hz tone stack (not by calling
        // build_clipped_fuzz_proxy) so if build_clipped_fuzz_proxy ever regressed to zero-phase filtfilt,
        // the ~9-sample causal group delay mismatch would fail this test.
        let dry_fuzz_in: Vec<f32> = (0..16_384)
            .map(|i| {
                let env = 0.5 + 0.5 * ((i as f32) * 0.019).sin() * ((i as f32) * 0.007).cos();
                let carrier = if i % 3 == 0 { 0.8_f32 } else { -0.4_f32 };
                env * carrier
            })
            .collect();
        let a_hp = (-TAU * 85.0 / (sr as f32)).exp();
        let a_lp = (-TAU * 750.0 / (sr as f32)).exp();
        let mut s_hp = 0.0_f32;
        let mut s_lp = 0.0_f32;
        let mut wet_causal_fuzz = vec![0.0_f32; dry_fuzz_in.len()];
        for i in true_hw_lag..dry_fuzz_in.len() {
            let x = dry_fuzz_in[i - true_hw_lag];
            s_hp = a_hp * s_hp + (1.0 - a_hp) * x;
            let clipped = (12.0 * (x - s_hp)).tanh();
            s_lp = a_lp * s_lp + (1.0 - a_lp) * clipped;
            wet_causal_fuzz[i] = s_lp;
        }
        let fuzz_rep = align_hardware_latency(
            &dry_fuzz_in,
            &wet_causal_fuzz,
            512,
            Some(locked_lag),
            sr,
            None,
        )
        .expect("align causal hardware fuzz");
        assert!(
            (fuzz_rep.coarse_lag_samples - (true_hw_lag as i32)).abs() <= 1,
            "Causal fuzz alignment must match hardware latency within +/-1 sample (got {} vs {})",
            fuzz_rep.coarse_lag_samples,
            true_hw_lag
        );

        // 2. Heavy bass-boosted amp sidelobe rejection (`swe_tc650` Gain=10, Bass=10):
        // Inject a strong +82-sample periodic bass sidelobe outside +/-48 samples and verify
        // align_hardware_latency never jumps to the +82-sample bass sidelobe.
        let mut wet_bass_sidelobe = vec![0.0_f32; dry.len()];
        let sidelobe_lag = (locked_lag as usize) + 82;
        for i in sidelobe_lag..dry.len() {
            wet_bass_sidelobe[i] =
                0.30 * dry[i - (locked_lag as usize)] + 0.70 * dry[i - sidelobe_lag];
        }
        let bass_rep =
            align_hardware_latency(&dry, &wet_bass_sidelobe, 512, Some(locked_lag), sr, None)
                .expect("align bass sidelobe");
        assert!(
            (bass_rep.coarse_lag_samples - locked_lag).abs() <= 2,
            "Must lock to direct path at locked_lag ({locked_lag}) and reject +82-sample sidelobe (got {})",
            bass_rep.coarse_lag_samples
        );

        // 3. Amp tone-stack broadband dry gate (`peggy_fliptop` Treble=0.0 vs star_center Treble=5.0):
        // When broadband dry has c_dry > 0.40 at lag = 0, verify_and_polish_post_alignment must use
        // "dry" exclusively and NEVER shift the stem toward a star_center reference at lag = +3.
        let mut wet_treble_min = vec![0.0_f32; dry.len()];
        let mut s_tone = 0.0_f32;
        let a_tone = (-TAU * 600.0 / (sr as f32)).exp();
        for (i, &x) in dry.iter().enumerate() {
            s_tone = a_tone * s_tone + (1.0 - a_tone) * x;
            wet_treble_min[i] = 0.55 * x + 0.45 * s_tone;
        }
        let star_center_shifted = apply_subsample_shift_fft(&wet_treble_min, -3.0);
        let v_tone = verify_and_polish_post_alignment(
            &dry,
            Some(&star_center_shifted),
            &wet_treble_min,
            locked_lag as f32,
            sr,
        );
        assert_eq!(v_tone.best_reference, "dry");
        assert_eq!(v_tone.residual_lag_samples, 0);
        assert_eq!(v_tone.total_lag_samples, locked_lag as f32);

        // 4. Boundary slope-following (`best_l == lo` and `best_l == hi` with `best_c > 0.35`):
        // When USB jitter places the smooth main-lobe peak at `locked - 54` or `locked + 54`
        // (6 samples outside `locked ± 48`), slope-following walks outward up to 16 samples to the
        // true local maximum without clipping at the boundary.
        let smooth_dry = build_zerophase_lp_proxy(&dry, sr, 900.0);
        let lag_lo_out = (locked_lag - 54) as usize;
        let mut wet_lo_out = vec![0.0_f32; smooth_dry.len()];
        wet_lo_out[lag_lo_out..].copy_from_slice(&smooth_dry[..smooth_dry.len() - lag_lo_out]);
        let rep_lo =
            align_hardware_latency(&smooth_dry, &wet_lo_out, 512, Some(locked_lag), sr, None)
                .expect("boundary slope follow lo");
        assert_eq!(rep_lo.coarse_lag_samples, lag_lo_out as i32);

        let lag_hi_out = (locked_lag + 54) as usize;
        let mut wet_hi_out = vec![0.0_f32; smooth_dry.len()];
        wet_hi_out[lag_hi_out..].copy_from_slice(&smooth_dry[..smooth_dry.len() - lag_hi_out]);
        let rep_hi =
            align_hardware_latency(&smooth_dry, &wet_hi_out, 512, Some(locked_lag), sr, None)
                .expect("boundary slope follow hi");
        assert_eq!(rep_hi.coarse_lag_samples, lag_hi_out as i32);

        // 5. Post-alignment verification using "pulse1_dry" on a heavy-echo wet signal with residual lag
        let mut wet_echo_misaligned = vec![0.0_f32; dry.len()];
        let res_shift = 5_usize;
        for i in res_shift..dry.len() {
            wet_echo_misaligned[i] = 0.20 * dry[i - res_shift];
        }
        for tap in (1_200..18_000).step_by(900) {
            for i in (tap + res_shift)..dry.len() {
                wet_echo_misaligned[i] += 0.85 * dry[i - tap - res_shift];
            }
        }
        let v_p1 = verify_and_polish_post_alignment(
            &dry,
            None,
            &wet_echo_misaligned,
            locked_lag as f32,
            sr,
        );
        assert_eq!(v_p1.best_reference, "pulse1_dry");
        assert_eq!(v_p1.residual_lag_samples, 0);

        // 6. Short-buffer guard clauses (`pulse1_onset_window` < 128, `local_normalized_xcorr` < 64)
        assert!(pulse1_onset_window(&[0.5; 64], sr).is_none());
        assert_eq!(
            local_normalized_xcorr(&[0.5; 32], &[0.5; 32], 8),
            (0, 0.0, 0.0)
        );
    }
}
