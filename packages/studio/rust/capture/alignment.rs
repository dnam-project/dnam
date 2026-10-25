//! 2-stage coarse + parabolic sub-sample latency alignment and SNR analyzer.

use serde::{Deserialize, Serialize};

use crate::error::{StudioError, StudioResult};

/// Measured alignment and signal quality metrics for a captured wet stem.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignmentReport {
    /// Integer coarse lag in samples (`wet` delayed relative to `dry`).
    pub coarse_lag_samples: i32,
    /// Parabolic sub-sample refined lag in samples.
    pub subsample_lag_samples: f32,
    /// Measured signal-to-noise ratio in dB.
    pub snr_db: f32,
    /// Measured silent-lead noise floor in dBFS.
    pub noise_floor_dbfs: f32,
}

/// Estimates coarse and parabolic sub-sample lag between `dry` and `wet` signals plus SNR.
pub fn estimate_subsample_lag_and_snr(
    dry: &[f32],
    wet: &[f32],
    max_lag_samples: usize,
) -> StudioResult<AlignmentReport> {
    if dry.is_empty() || wet.is_empty() || dry.len() != wet.len() {
        return Err(StudioError::Audio(
            "dry and wet signals must be non-empty and equal length".to_string(),
        ));
    }
    let n = dry.len();
    let max_lag = max_lag_samples.min(n / 4).max(1);
    let mut best_lag: i32 = 0;
    let mut best_corr = f32::NEG_INFINITY;
    let mut corr_map = vec![0.0_f32; max_lag * 2 + 1];

    for (idx, lag) in (-(max_lag as i32)..=(max_lag as i32)).enumerate() {
        let mut acc = 0.0_f32;
        for i in 0..n {
            let j = (i as i32) + lag;
            if j >= 0 && (j as usize) < n {
                acc += dry[i] * wet[j as usize];
            }
        }
        corr_map[idx] = acc;
        if acc > best_corr {
            best_corr = acc;
            best_lag = lag;
        }
    }

    let best_idx = (best_lag + (max_lag as i32)) as usize;
    let mut frac_offset = 0.0_f32;
    if best_idx > 0 && best_idx + 1 < corr_map.len() {
        let y_prev = corr_map[best_idx - 1];
        let y_curr = corr_map[best_idx];
        let y_next = corr_map[best_idx + 1];
        let denom = y_prev - 2.0 * y_curr + y_next;
        if denom.abs() > 1e-9 {
            frac_offset = 0.5 * (y_prev - y_next) / denom;
            frac_offset = frac_offset.clamp(-0.5, 0.5);
        }
    }

    let noise_window = (n / 16).max(1);
    let noise_power =
        wet[..noise_window].iter().map(|&x| x * x).sum::<f32>() / (noise_window as f32);
    let signal_power = wet.iter().map(|&x| x * x).sum::<f32>() / (n as f32);
    let safe_noise = noise_power.max(1e-12);
    let safe_sig = signal_power.max(1e-12);
    let snr_db = 10.0 * (safe_sig / safe_noise).log10();
    let noise_floor_dbfs = 10.0 * safe_noise.log10();

    Ok(AlignmentReport {
        coarse_lag_samples: best_lag,
        subsample_lag_samples: (best_lag as f32) + frac_offset,
        snr_db,
        noise_floor_dbfs,
    })
}
