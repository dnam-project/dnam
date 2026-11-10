//! Zero-Phase Polyphase Kaiser-Sinc Multi-Rate Audio & Control Trajectory Resampler (`8..192 kHz` <-> `48 kHz`).
//!
//! Canonical References:
//! - `docs/specs/training/capture_and_alignment.md` (`r[train.capture.invariants]`)
//! - `docs/specs/training/four_stage_pipeline.md` (`r[train.pipeline.sweep1]`)
//! - `docs/adr/0018-signal-chain-fixtures-crate-and-whitebox-distillation.md`

#![forbid(unsafe_code)]

use std::f64::consts::PI;

/// Canonical internal trainer excitation and parameter-identification sample rate (`48,000 Hz`).
pub const CANONICAL_TRAINER_SAMPLE_RATE_HZ: u32 = 48_000;

#[inline]
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;
    let half_x_sq = 0.25 * x * x;
    for k in 1..=28 {
        let kf = f64::from(k);
        term *= half_x_sq / (kf * kf);
        sum += term;
        if term < sum * 1.0e-16 {
            break;
        }
    }
    sum
}

#[inline]
fn sanitize_f32(x: f32) -> f32 {
    if x.is_finite() { x } else { 0.0 }
}

/// Computes the canonical resampled frame length for `src_len` frames from `src_rate_hz` to `dst_rate_hz`.
#[inline]
#[must_use]
pub fn resampled_frame_len(src_len: usize, src_rate_hz: u32, dst_rate_hz: u32) -> usize {
    if src_len == 0 || src_rate_hz == 0 || dst_rate_hz == 0 {
        return 0;
    }
    if src_rate_hz == dst_rate_hz {
        return src_len;
    }
    let num = (src_len as u128) * u128::from(dst_rate_hz) + u128::from(src_rate_hz) / 2;
    (num / u128::from(src_rate_hz))
        .max(1)
        .min(usize::MAX as u128) as usize
}

/// Resamples `signal` from `src_rate_hz` to `dst_rate_hz` using a zero-phase (`0.00`-sample group delay),
/// partition-of-unity-normalized Kaiser-windowed sinc interpolator ($\beta = 8.6$, `> 85 dB` stopband rejection).
#[must_use]
pub fn resample_arbitrary_rate(
    signal: &[f32],
    src_rate_hz: u32,
    dst_rate_hz: u32,
    dst_len: Option<usize>,
) -> Vec<f32> {
    if signal.is_empty() || src_rate_hz == 0 || dst_rate_hz == 0 || dst_len == Some(0) {
        return Vec::new();
    }
    let out_len =
        dst_len.unwrap_or_else(|| resampled_frame_len(signal.len(), src_rate_hz, dst_rate_hz));
    if src_rate_hz == dst_rate_hz && out_len == signal.len() {
        return signal.iter().map(|&x| sanitize_f32(x)).collect();
    }

    let src_fs = f64::from(src_rate_hz);
    let dst_fs = f64::from(dst_rate_hz);
    let min_fs = src_fs.min(dst_fs);
    // Anti-imaging / anti-aliasing lowpass cutoff centered at 0.477 * min(src_fs, dst_fs) with R >= 64
    // so 44.1 kHz devices maintain < 0.01 dB passband flatness past 20,000 Hz and > 85 dB stopband before 22,050 Hz.
    let fc_norm = 0.477 * (min_fs / src_fs);
    let stretch = (src_fs / min_fs).clamp(1.0, 192.0);
    let radius = (64.0 * stretch).ceil().max(32.0);
    let half_support = radius as isize;

    // Precompute symmetric half-wing table h(u) for u in [0, radius] at fine resolution
    const TABLE_STEPS_PER_UNIT: usize = 512;
    let table_len = (radius as usize) * TABLE_STEPS_PER_UNIT + 2;
    let beta = 8.6_f64;
    let inv_i0_beta = 1.0 / bessel_i0(beta);
    let mut table = vec![0.0_f64; table_len];
    for (idx, slot) in table.iter_mut().enumerate() {
        let u = (idx as f64) / (TABLE_STEPS_PER_UNIT as f64);
        if u >= radius {
            *slot = 0.0;
        } else {
            let arg = 2.0 * fc_norm * u;
            let sinc = if arg.abs() < 1.0e-12 {
                1.0
            } else {
                let pix = PI * arg;
                pix.sin() / pix
            };
            let ratio = u / radius;
            let win = bessel_i0(beta * (1.0 - ratio * ratio).max(0.0).sqrt()) * inv_i0_beta;
            *slot = 2.0 * fc_norm * sinc * win;
        }
    }

    let clean_src: Vec<f64> = signal.iter().map(|&x| f64::from(sanitize_f32(x))).collect();
    let n_src = clean_src.len() as isize;
    let step_src = src_fs / dst_fs;
    let scale_idx = TABLE_STEPS_PER_UNIT as f64;

    let mut out = Vec::with_capacity(out_len);
    for m in 0..out_len {
        let center = (m as f64) * step_src;
        let k_min = (center - radius).ceil() as isize;
        let k_max = (center + radius).floor() as isize;
        let mut acc = 0.0_f64;
        let mut sum_w = 0.0_f64;

        for k in k_min.max(-half_support)..=k_max {
            let dist = (center - (k as f64)).abs();
            if dist >= radius {
                continue;
            }
            let pos = dist * scale_idx;
            let idx = pos as usize;
            let frac = pos - (idx as f64);
            let w = table[idx] + frac * (table[idx + 1] - table[idx]);
            let clamped_k = k.clamp(0, n_src - 1) as usize;
            acc += w * clean_src[clamped_k];
            sum_w += w;
        }
        let y = if sum_w.abs() > 1.0e-15 {
            acc / sum_w
        } else {
            0.0
        };
        let y_f32 = if y.is_finite() {
            y.clamp(f64::from(f32::MIN), f64::from(f32::MAX)) as f32
        } else {
            0.0
        };
        out.push(y_f32);
    }
    out
}

/// Resamples an arbitrary device-rate waveform (`src_rate_hz`) to canonical `48,000 Hz`.
#[inline]
#[must_use]
pub fn resample_arbitrary_rate_to_48k(
    signal: &[f32],
    src_rate_hz: u32,
    dst_len: Option<usize>,
) -> Vec<f32> {
    resample_arbitrary_rate(
        signal,
        src_rate_hz,
        CANONICAL_TRAINER_SAMPLE_RATE_HZ,
        dst_len,
    )
}

/// Resamples a canonical `48,000 Hz` stimulus waveform to a target device's native `dst_rate_hz`.
#[inline]
#[must_use]
pub fn resample_48k_to_arbitrary_rate(
    signal_48k: &[f32],
    dst_rate_hz: u32,
    dst_len: Option<usize>,
) -> Vec<f32> {
    resample_arbitrary_rate(
        signal_48k,
        CANONICAL_TRAINER_SAMPLE_RATE_HZ,
        dst_rate_hz,
        dst_len,
    )
}

/// Resamples a per-sample parameter control trajectory `traj` onto `dst_len` samples using
/// monotone endpoint-preserving piecewise-linear interpolation (zero Gibbs overshoot outside `[min, max]`).
#[must_use]
pub fn resample_control_trajectory(traj: &[f32], dst_len: usize) -> Vec<f32> {
    if dst_len == 0 || traj.is_empty() {
        return vec![0.0_f32; dst_len];
    }
    if traj.len() == dst_len {
        return traj.iter().map(|&v| sanitize_f32(v)).collect();
    }
    if traj.len() == 1 || dst_len == 1 {
        return vec![sanitize_f32(traj[0]); dst_len];
    }
    let src_last = (traj.len() - 1) as f64;
    let dst_last = (dst_len - 1) as f64;
    (0..dst_len)
        .map(|m| {
            let pos = (m as f64) * src_last / dst_last;
            let i0 = (pos.floor() as usize).min(traj.len() - 1);
            let i1 = (i0 + 1).min(traj.len() - 1);
            let frac = pos - (i0 as f64);
            let v0 = f64::from(sanitize_f32(traj[i0]));
            let v1 = f64::from(sanitize_f32(traj[i1]));
            ((1.0 - frac) * v0 + frac * v1).clamp(f64::from(f32::MIN), f64::from(f32::MAX)) as f32
        })
        .collect()
}

/// Scales a `[start, end]` sample-index window proportionally from `src_len` to `dst_len`.
#[inline]
#[must_use]
pub fn scale_freeze_window(window: [usize; 2], src_len: usize, dst_len: usize) -> [usize; 2] {
    if src_len == 0 || dst_len == 0 {
        return [0, 0];
    }
    let scale = |idx: usize| -> usize {
        let clamped = idx.min(src_len) as u128;
        (((clamped * (dst_len as u128)) + (src_len as u128) / 2) / (src_len as u128))
            .min(dst_len as u128) as usize
    };
    let s = scale(window[0]);
    let e = scale(window[1]).max(s);
    [s, e]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_phase_kaiser_sinc_resampler_dc_tone_and_stopband() {
        assert!(resample_arbitrary_rate(&[], 48_000, 96_000, None).is_empty());
        assert_eq!(resampled_frame_len(0, 48_000, 96_000), 0);

        // Identity at 48 kHz
        let id = vec![0.25_f32, -0.5, 0.75, f32::NAN];
        assert_eq!(
            resample_arbitrary_rate_to_48k(&id, 48_000, None),
            vec![0.25, -0.5, 0.75, 0.0]
        );

        // Exact DC unity across non-integer and integer rate ratios
        for &sr in &[32_000_u32, 44_100, 88_200, 96_000, 192_000] {
            let dc_up = resample_48k_to_arbitrary_rate(&vec![0.65_f32; 512], sr, None);
            assert!(dc_up.iter().all(|&v| (v - 0.65).abs() < 1.0e-5));
            let dc_back = resample_arbitrary_rate_to_48k(&dc_up, sr, Some(512));
            assert!(dc_back.iter().all(|&v| (v - 0.65).abs() < 1.0e-5));
        }

        // Round-trip 440 Hz tone 48 kHz -> 44.1 kHz -> 48 kHz and 48 kHz -> 96 kHz -> 48 kHz:
        // verify zero phase lag (< -80 dB residual in interior window!)
        let n_48k = 4_096usize;
        let tone_48k: Vec<f32> = (0..n_48k)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / 48_000.0).sin())
            .collect();
        for &dev_sr in &[32_000_u32, 44_100, 88_200, 96_000, 192_000] {
            let at_dev = resample_48k_to_arbitrary_rate(&tone_48k, dev_sr, None);
            let roundtrip = resample_arbitrary_rate_to_48k(&at_dev, dev_sr, Some(n_48k));
            let interior = &roundtrip[256..n_48k - 256];
            let ref_int = &tone_48k[256..n_48k - 256];
            let err_sq: f64 = interior
                .iter()
                .zip(ref_int.iter())
                .map(|(&a, &b)| {
                    let d = f64::from(a) - f64::from(b);
                    d * d
                })
                .sum();
            let sig_sq: f64 = ref_int.iter().map(|&b| f64::from(b).powi(2)).sum();
            let esr_db = 10.0 * (err_sq / sig_sq).log10();
            assert!(
                esr_db < -85.0,
                "48k -> {dev_sr} -> 48k roundtrip ESR too high: {esr_db:.2} dB"
            );
        }

        // Anti-aliasing rejection: 35 kHz ultrasonic tone @ 96 kHz downsampled to 48 kHz
        let ultra_96k: Vec<f32> = (0..4_096)
            .map(|i| (2.0 * std::f32::consts::PI * 35_000.0 * (i as f32) / 96_000.0).sin())
            .collect();
        let down_48k = resample_arbitrary_rate_to_48k(&ultra_96k, 96_000, None);
        let interior_alias = &down_48k[128..down_48k.len() - 128];
        let alias_rms = (interior_alias
            .iter()
            .map(|&x| f64::from(x).powi(2))
            .sum::<f64>()
            / (interior_alias.len() as f64))
            .sqrt();
        assert!(
            alias_rms < 1.0e-4,
            "35 kHz ultrasonic leaked: {alias_rms:e}"
        );

        // Control trajectory & freeze window scaling
        assert!(
            resample_control_trajectory(&[], 8)
                .iter()
                .all(|&v| v == 0.0)
        );
        assert_eq!(resample_control_trajectory(&[0.5], 4), vec![0.5; 4]);
        let ramp = resample_control_trajectory(&[-1.0, 1.0], 5);
        assert!(
            (ramp[0] - (-1.0)).abs() < 1.0e-6
                && (ramp[2] - 0.0).abs() < 1.0e-6
                && (ramp[4] - 1.0).abs() < 1.0e-6
        );
        assert_eq!(scale_freeze_window([100, 300], 400, 800), [200, 600]);
        assert_eq!(scale_freeze_window([10, 20], 0, 100), [0, 0]);

        // Direct 24x decimation (192 kHz -> 8 kHz): > 85 dB stopband rejection at 4,200 Hz
        let sig_192k: Vec<f32> = (0..16_384)
            .map(|i| (2.0 * std::f32::consts::PI * 4_200.0 * (i as f32) / 192_000.0).sin())
            .collect();
        let dec_8k = resample_arbitrary_rate(&sig_192k, 192_000, 8_000, None);
        let int_8k = &dec_8k[96..dec_8k.len() - 96];
        let peak_8k = int_8k.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        assert!(peak_8k < 5.0e-5, "192k->8k stopband leak: {peak_8k:e}");

        // Near-f32::MAX overflow safety in resample_control_trajectory and resample_arbitrary_rate
        let extreme_traj = resample_control_trajectory(&[-f32::MAX, f32::MAX], 5);
        assert!(extreme_traj.iter().all(|v| v.is_finite()));
        assert_eq!(extreme_traj[0], -f32::MAX);
        assert_eq!(extreme_traj[4], f32::MAX);
        let alt_max: Vec<f32> = (0..64)
            .map(|i| if i % 2 == 0 { f32::MAX } else { -f32::MAX })
            .collect();
        let res_max = resample_arbitrary_rate(&alt_max, 44_100, 48_000, None);
        assert!(res_max.iter().all(|v| v.is_finite()));
        assert_eq!(resampled_frame_len(usize::MAX, 44_100, 192_000), usize::MAX);
    }
}
