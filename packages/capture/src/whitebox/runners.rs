//! GPL-Isolated `SpiceRunner`, Monotonic `PCHIP` Variable-Step Resampling,
//! and `309`-Tap Kaiser-Sinc `4x -> 1x` Anti-Aliasing Decimation.

use super::ipc::validate_runner_stimulus_layout;
use super::{DnwbCommand, DnwbError, DnwbWorkerSession, SPICE_MAXSTEP_SEC};
use crate::targets::{DistillationTarget, StemResponse, StemStimulus, StimulusFrame};
use dnam_engine::schema::DnamError;
use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::time::{Duration, Instant};

const KAISER_HALF_TAPS: isize = 154;
const KAISER_NUM_TAPS: usize = (2 * KAISER_HALF_TAPS + 1) as usize; // 309 taps
const KAISER_HISTORY_LEN: usize = KAISER_NUM_TAPS - 1; // 308 samples
const KAISER_FC_NORM: f64 = 22_000.0 / 96_000.0;
const KAISER_BETA: f64 = 10.06;

/// Exact integer causal group delay (`39` `1x` samples = `156` `4x` substeps, `0.00`-sample fractional skew)
/// of `SpiceRunner`'s streaming `309`-tap Kaiser-sinc `4x -> 1x` decimator.
pub const SPICE_CAUSAL_DELAY_SAMPLES: usize = 39;

/// Generates canonical SPICE `.options maxstep=5.208333e-6` and `.tran` netlist directives at `48 kHz`.
#[must_use]
pub fn format_spice_tran_directives(duration_sec: f64) -> String {
    format_spice_tran_directives_for_rate(duration_sec, 48_000)
}

/// Generates SPICE `.options maxstep` (`1 / (4 * sample_rate_hz)`) and `.tran` directives for arbitrary `sample_rate_hz`.
#[must_use]
pub fn format_spice_tran_directives_for_rate(duration_sec: f64, sample_rate_hz: u32) -> String {
    let sr = f64::from(sample_rate_hz.max(8_000));
    let maxstep = 1.0 / (4.0 * sr);
    format!(
        ".options maxstep={maxstep:.6e} reltol=1e-5 abstol=1e-9\n.tran {maxstep:.6e} {duration_sec:.6e}\n"
    )
}

#[inline]
fn fritsch_carlson_endpoint_slope(h0: f64, h1: f64, d0: f64, d1: f64) -> f64 {
    let s = h0.max(h1);
    if !d0.is_finite() || !d1.is_finite() || s <= 0.0 {
        return 0.0;
    }
    let (u0, u1) = (h0 / s, h1 / s);
    let raw = ((2.0 * u0 + u1) * d0 - u0 * d1) / (u0 + u1);
    if !raw.is_finite() || raw.signum() != d0.signum() || d0 == 0.0 {
        0.0
    } else if d0.signum() != d1.signum() && raw.abs() > 3.0 * d0.abs() {
        (3.0 * d0).clamp(f64::MIN, f64::MAX)
    } else {
        raw
    }
}

/// Resamples a variable-timestep SPICE transient trajectory `(times_sec, values)` onto a uniform
/// `target_rate_hz` grid using Fritsch-Carlson Monotonic Cubic Hermite (`PCHIP`) interpolation
/// (coalescing duplicate/non-increasing event timestamps and enforcing monotonicity at endpoints).
#[must_use]
pub fn pchip_resample_to_rate(
    times_sec: &[f64],
    values: &[f64],
    num_samples: usize,
    target_rate_hz: f64,
) -> Vec<f64> {
    let n_in = times_sec.len().min(values.len());
    if n_in == 0 || num_samples == 0 {
        return Vec::new();
    }
    let rate = if target_rate_hz.is_finite() {
        target_rate_hz.max(1.0)
    } else {
        1.0
    };

    // Coalesce duplicate or non-strictly-increasing SPICE event breakpoints cleanly.
    let mut t_knots: Vec<f64> = Vec::with_capacity(n_in);
    let mut y_knots: Vec<f64> = Vec::with_capacity(n_in);
    for idx in 0..n_in {
        let (t, y) = (times_sec[idx], values[idx]);
        if !t.is_finite() || !y.is_finite() {
            continue;
        }
        if let Some(&last_t) = t_knots.last()
            && 0.25 * t - 0.25 * last_t <= 0.25e-15
        {
            if t >= last_t
                && let Some(last_y) = y_knots.last_mut()
            {
                *last_y = y;
            }
            continue;
        }
        t_knots.push(t);
        y_knots.push(y);
    }

    let n = t_knots.len();
    if n == 0 {
        return vec![0.0; num_samples];
    }
    if n == 1 {
        return vec![y_knots[0]; num_samples];
    }
    if n == 2 {
        let qh0 = 0.25 * t_knots[1] - 0.25 * t_knots[0];
        return (0..num_samples)
            .map(|m| {
                let qt = 0.25 * ((m as f64) / rate) - 0.25 * t_knots[0];
                let s = if qh0 > 0.0 {
                    (qt / qh0).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                ((1.0 - s) * y_knots[0] + s * y_knots[1]).clamp(f64::MIN, f64::MAX)
            })
            .collect();
    }

    let mut qh = Vec::with_capacity(n - 1);
    let mut delta = Vec::with_capacity(n - 1);
    for k in 0..(n - 1) {
        let qdt = 0.25 * t_knots[k + 1] - 0.25 * t_knots[k];
        qh.push(qdt);
        delta.push((0.25 * y_knots[k + 1] - 0.25 * y_knots[k]) / qdt);
    }

    let mut d = vec![0.0_f64; n];
    d[0] = fritsch_carlson_endpoint_slope(qh[0], qh[1], delta[0], delta[1]);
    d[n - 1] = fritsch_carlson_endpoint_slope(qh[n - 2], qh[n - 3], delta[n - 2], delta[n - 3]);
    for k in 1..(n - 1) {
        let (dk_prev, dk_curr) = (delta[k - 1], delta[k]);
        if dk_prev.is_finite()
            && dk_curr.is_finite()
            && dk_prev.signum() == dk_curr.signum()
            && dk_prev.abs() > 0.0
            && dk_curr.abs() > 0.0
        {
            let s = qh[k].max(qh[k - 1]);
            let (uk, up) = (qh[k] / s, qh[k - 1] / s);
            let (w1, w2) = (2.0 * uk + up, uk + 2.0 * up);
            let raw_d = (w1 + w2) / (w1 / dk_prev + w2 / dk_curr);
            if raw_d.is_finite() {
                let max_mag = (3.0 * dk_prev.abs().min(dk_curr.abs())).min(f64::MAX);
                d[k] = raw_d.clamp(-max_mag, max_mag);
            }
        }
    }

    let mut out = Vec::with_capacity(num_samples);
    let mut seg = 0usize;
    for m in 0..num_samples {
        let t = (m as f64) / rate;
        while seg + 2 < n && t_knots[seg + 1] < t {
            seg += 1;
        }
        let qhk = qh[seg];
        let s = if qhk > 0.0 {
            ((0.25 * t - 0.25 * t_knots[seg]) / qhk).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let (s2, s3) = (s * s, s * s * s);
        let (h00, h10) = (2.0 * s3 - 3.0 * s2 + 1.0, s3 - 2.0 * s2 + s);
        let (h01, h11) = (-2.0 * s3 + 3.0 * s2, s3 - s2);
        let td0 = (qhk * d[seg]).clamp(-0.25 * f64::MAX, 0.25 * f64::MAX);
        let td1 = (qhk * d[seg + 1]).clamp(-0.25 * f64::MAX, 0.25 * f64::MAX);
        let base = h00 * y_knots[seg] + h01 * y_knots[seg + 1];
        let tang = 4.0 * (h10 * td0 + h11 * td1);
        out.push((base + tang).clamp(f64::MIN, f64::MAX));
    }
    out
}

/// Resamples a variable-timestep SPICE trajectory onto a uniform `192 kHz` (`4x` at `48 kHz`) grid
/// using Fritsch-Carlson Monotonic Cubic Hermite (`PCHIP`) interpolation.
#[must_use]
pub fn pchip_resample_to_192khz(
    times_sec: &[f64],
    values: &[f64],
    num_samples_192k: usize,
) -> Vec<f64> {
    pchip_resample_to_rate(times_sec, values, num_samples_192k, 1.0 / SPICE_MAXSTEP_SEC)
}

/// Evaluates the modified zeroth-order Bessel function $I_0(x)$ for the Kaiser window.
#[inline]
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;
    let half_x_sq = 0.25 * x * x;
    for k in 1..=48 {
        term *= half_x_sq / ((k * k) as f64);
        sum += term;
        if term < 1.0e-16 * sum {
            break;
        }
    }
    sum
}

fn build_kaiser_309_coeffs() -> [f64; KAISER_NUM_TAPS] {
    let denom_i0 = bessel_i0(KAISER_BETA);
    let mut coeffs = [0.0_f64; KAISER_NUM_TAPS];
    let mut sum_w = 0.0_f64;
    for k in -KAISER_HALF_TAPS..=KAISER_HALF_TAPS {
        let sinc = if k == 0 {
            KAISER_FC_NORM
        } else {
            let x = PI * KAISER_FC_NORM * (k as f64);
            KAISER_FC_NORM * (x.sin() / x)
        };
        let ratio = (k as f64) / (KAISER_HALF_TAPS as f64);
        let win = bessel_i0(KAISER_BETA * (1.0 - ratio * ratio).max(0.0).sqrt()) / denom_i0;
        let c = sinc * win;
        coeffs[(k + KAISER_HALF_TAPS) as usize] = c;
        sum_w += c;
    }
    for c in &mut coeffs {
        *c /= sum_w;
    }
    coeffs
}

#[inline]
fn f64_to_finite_f32(v: f64) -> f32 {
    if v.is_finite() {
        v.clamp(f64::from(f32::MIN), f64::from(f32::MAX)) as f32
    } else {
        0.0
    }
}

/// Decimates a uniform `4x` oversampled signal (e.g. `192 kHz -> 48 kHz`) using a `309`-tap
/// linear-phase Kaiser-windowed sinc anti-aliasing low-pass filter (`f_pass = 0.4167 * f_N`,
/// `f_stop = 0.5000 * f_N`, `beta = 10.06` -> `> 100 dB` stopband attenuation).
#[must_use]
pub fn kaiser_sinc_decimate_4x_to_48khz(signal_192k: &[f64]) -> Vec<f32> {
    let n_192 = signal_192k.len();
    let n_48 = n_192 / 4;
    if n_48 == 0 {
        return Vec::new();
    }
    let coeffs = build_kaiser_309_coeffs();
    let mut out = Vec::with_capacity(n_48);
    for m in 0..n_48 {
        let center = (m * 4) as isize;
        let mut acc = 0.0_f64;
        for k in -KAISER_HALF_TAPS..=KAISER_HALF_TAPS {
            let idx = (center + k).clamp(0, (n_192 as isize) - 1) as usize;
            let s = signal_192k[idx];
            acc += coeffs[(k + KAISER_HALF_TAPS) as usize] * if s.is_finite() { s } else { 0.0 };
        }
        out.push(f64_to_finite_f32(acc));
    }
    out
}

#[derive(Clone, Debug)]
struct CausalKaiserDecimator4x {
    coeffs: Vec<f64>,
    history: Vec<f64>,
}

impl CausalKaiserDecimator4x {
    fn new() -> Self {
        Self {
            coeffs: build_kaiser_309_coeffs().to_vec(),
            history: vec![0.0_f64; KAISER_HISTORY_LEN],
        }
    }

    fn reset(&mut self) {
        self.history.fill(0.0);
    }

    fn process_4x_block(&mut self, signal_4x: &[f64]) -> Vec<f32> {
        let n_1x = signal_4x.len() / 4;
        if n_1x == 0 {
            return Vec::new();
        }
        let used_4x = n_1x * 4;
        let mut ext = Vec::with_capacity(KAISER_HISTORY_LEN + used_4x);
        ext.extend_from_slice(&self.history);
        ext.extend(
            signal_4x[..used_4x]
                .iter()
                .map(|&x| if x.is_finite() { x } else { 0.0 }),
        );

        let mut out = Vec::with_capacity(n_1x);
        for m in 0..n_1x {
            // Center tap 154 sits at ext[155 + 4*m] = ext[311 - 156 + 4*m] -> exact 39.00-sample delay!
            let win_start = 1 + m * 4;
            let mut acc = 0.0_f64;
            for (tap, &c) in self.coeffs.iter().enumerate() {
                acc += c * ext[win_start + tap];
            }
            out.push(f64_to_finite_f32(acc));
        }
        self.history
            .copy_from_slice(&ext[ext.len() - KAISER_HISTORY_LEN..]);
        out
    }
}

#[inline]
fn bounded_sinh_cosh(x: f64) -> (f64, f64) {
    const LIMIT: f64 = 40.0;
    if x > LIMIT {
        let sh = LIMIT.sinh();
        let ch = LIMIT.cosh();
        (sh + (x - LIMIT) * ch, ch)
    } else if x < -LIMIT {
        let sh = (-LIMIT).sinh();
        let ch = LIMIT.cosh();
        (sh + (x + LIMIT) * ch, ch)
    } else {
        (x.sinh(), x.cosh())
    }
}

/// Analytical Rust `f64` / Subprocess SPICE Runner solving a continuous-time nonlinear diode-clipper
/// nodal ODE via Backward-Euler damped Newton-Raphson on an adaptive timestep grid (`dt <= 1 / (4 * fs)`),
/// followed by `PCHIP` `4x` resampling and causal `309`-tap Kaiser-sinc `4x -> 1x` decimation.
#[derive(Clone, Debug)]
pub struct SpiceRunner {
    pub netlist_name: String,
    pub timeout: Duration,
    pub subprocess_cmd: Option<Vec<String>>,
    pub active_switches: Vec<u8>,
    cap_voltage: [f64; 4],
    prev_vin: [f64; 4],
    decim_v: [CausalKaiserDecimator4x; 4],
    decim_i: [CausalKaiserDecimator4x; 4],
    worker_session: DnwbWorkerSession,
}

impl SpiceRunner {
    #[must_use]
    pub fn new(netlist_name: impl Into<String>, timeout_ms: u64) -> Self {
        Self {
            netlist_name: netlist_name.into(),
            timeout: Duration::from_millis(timeout_ms),
            subprocess_cmd: None,
            active_switches: Vec::new(),
            cap_voltage: [0.0; 4],
            prev_vin: [0.0; 4],
            decim_v: std::array::from_fn(|_| CausalKaiserDecimator4x::new()),
            decim_i: std::array::from_fn(|_| CausalKaiserDecimator4x::new()),
            worker_session: DnwbWorkerSession::new(),
        }
    }

    /// Configures an external GPL-isolated SPICE worker binary communicating over `DNWB` pipes.
    #[must_use]
    pub fn with_subprocess_command(mut self, cmd: Vec<String>) -> Self {
        self.worker_session.terminate();
        self.subprocess_cmd = Some(cmd);
        self
    }

    /// Sets discrete hardware switch positions (`0` = silicon symmetrical, `1` = LED high-headroom)
    /// forwarded over `DNWB` IPC and applied during transient ODE integration.
    pub fn set_switches(&mut self, switches: &[u8]) {
        self.active_switches.clear();
        self.active_switches.extend_from_slice(switches);
    }

    /// Simulates adaptive-timestep SPICE transient integration at `48 kHz` (`dt <= 1 / 192,000 s`)
    /// and returns `(decimated_voltage_f32, decimated_diode_current_f64)` both aligned at `48 kHz`.
    pub fn simulate_adaptive_spice_stem(
        &mut self,
        dry_48k: &[f32],
        drive_norm: f32,
    ) -> Result<(Vec<f32>, Vec<f64>), DnwbError> {
        let (v_out, i_out) =
            self.simulate_adaptive_spice_for_rate(dry_48k, 48_000, drive_norm, None)?;
        Ok((v_out, i_out.into_iter().map(f64::from).collect()))
    }

    /// Batch zero-phase SPICE transient simulation compensating [`SPICE_CAUSAL_DELAY_SAMPLES`] (`0`-sample lag).
    pub fn simulate_adaptive_spice_zero_phase(
        &mut self,
        dry_1x: &[f32],
        sample_rate_hz: u32,
        drive_norm: f32,
        drive_traj: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Vec<f32>), DnwbError> {
        self.simulate_channel_zero_phase_ext(
            0,
            dry_1x,
            sample_rate_hz,
            &[drive_norm],
            drive_traj.map(|t| vec![t.to_vec()]).as_ref(),
            None,
        )
    }

    fn simulate_channel_zero_phase_ext(
        &mut self,
        ch: usize,
        dry_1x: &[f32],
        sample_rate_hz: u32,
        control_values: &[f32],
        control_trajectories: Option<&Vec<Vec<f32>>>,
        freeze_window: Option<[usize; 2]>,
    ) -> Result<(Vec<f32>, Vec<f32>), DnwbError> {
        self.cap_voltage[ch] = 0.0;
        self.prev_vin[ch] = 0.0;
        self.decim_v[ch].reset();
        self.decim_i[ch].reset();
        let n = dry_1x.len();
        if n == 0 {
            if !(8_000..=1_536_000).contains(&sample_rate_hz) {
                return Err(DnwbError::InvalidStimulusLayout(format!(
                    "sample_rate_hz {sample_rate_hz} outside [8000, 1536000]"
                )));
            }
            return Ok((Vec::new(), Vec::new()));
        }
        let mut padded = Vec::with_capacity(n + SPICE_CAUSAL_DELAY_SAMPLES);
        padded.extend_from_slice(dry_1x);
        padded.resize(n + SPICE_CAUSAL_DELAY_SAMPLES, 0.0);
        let padded_trajs = control_trajectories.map(|trajs| {
            trajs
                .iter()
                .enumerate()
                .map(|(p, tr)| {
                    let fallback = control_values.get(p).copied().unwrap_or(0.0);
                    let last = tr.last().copied().unwrap_or(fallback);
                    let mut pt = Vec::with_capacity(n + SPICE_CAUSAL_DELAY_SAMPLES);
                    pt.extend_from_slice(tr);
                    pt.resize(n + SPICE_CAUSAL_DELAY_SAMPLES, last);
                    pt
                })
                .collect::<Vec<Vec<f32>>>()
        });
        let ext_fw = freeze_window.map(|[fs, fe]| {
            [
                fs,
                if fs < fe && fe == n {
                    n + SPICE_CAUSAL_DELAY_SAMPLES
                } else {
                    fe
                },
            ]
        });
        let (v_all, i_all) = self.simulate_channel_for_rate_ext(
            ch,
            &padded,
            sample_rate_hz,
            control_values,
            padded_trajs.as_ref(),
            ext_fw,
        )?;
        Ok((
            v_all[SPICE_CAUSAL_DELAY_SAMPLES..SPICE_CAUSAL_DELAY_SAMPLES + n].to_vec(),
            i_all[SPICE_CAUSAL_DELAY_SAMPLES..SPICE_CAUSAL_DELAY_SAMPLES + n].to_vec(),
        ))
    }

    /// Simulates adaptive-timestep SPICE transient integration for arbitrary `sample_rate_hz` (`>= 8 kHz`)
    /// and optional per-sample drive trajectory, returning causally decimated `1x` `(out_1, diode_current)`.
    pub fn simulate_adaptive_spice_for_rate(
        &mut self,
        dry_1x: &[f32],
        sample_rate_hz: u32,
        drive_norm: f32,
        drive_traj: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Vec<f32>), DnwbError> {
        self.simulate_channel_for_rate_ext(
            0,
            dry_1x,
            sample_rate_hz,
            &[drive_norm],
            drive_traj.map(|t| vec![t.to_vec()]).as_ref(),
            None,
        )
    }

    fn simulate_channel_for_rate_ext(
        &mut self,
        ch: usize,
        dry_1x: &[f32],
        sample_rate_hz: u32,
        control_values: &[f32],
        control_trajectories: Option<&Vec<Vec<f32>>>,
        freeze_window: Option<[usize; 2]>,
    ) -> Result<(Vec<f32>, Vec<f32>), DnwbError> {
        if self.timeout.is_zero() {
            return Err(DnwbError::WatchdogTimeout { timeout_ms: 0 });
        }
        if !(8_000..=1_536_000).contains(&sample_rate_hz) {
            return Err(DnwbError::InvalidStimulusLayout(format!(
                "SpiceRunner sample_rate_hz {sample_rate_hz} outside [8000, 1536000]"
            )));
        }
        let n_1x = dry_1x.len();
        if n_1x == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let start_inst = Instant::now();
        let sr = f64::from(sample_rate_hz);
        let dt_1x = 1.0 / sr;
        let dt_max = 1.0 / (4.0 * sr);
        let n_4x = n_1x * 4;
        let drive_norm = control_values.first().copied().unwrap_or(0.0);
        let level_norm_default = control_values.get(1).copied().unwrap_or(0.0);
        let has_level_ctrl =
            control_values.len() >= 2 || control_trajectories.is_some_and(|t| t.len() >= 2);
        let eval_trim = |sample_idx: usize| -> f64 {
            if !has_level_ctrl {
                return 1.0;
            }
            let lvl_u = control_trajectories
                .and_then(|t| t.get(1))
                .and_then(|tr| tr.get(sample_idx).copied())
                .unwrap_or(level_norm_default);
            let u_clean = if lvl_u.is_finite() {
                f64::from(lvl_u).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            1.0 + 0.25 * u_clean
        };

        const TAU_RC_SEC: f64 = SPICE_MAXSTEP_SEC / 0.08;
        const IS_OVER_C_VPS: f64 = 0.025 / SPICE_MAXSTEP_SEC;
        let v_th = if self.active_switches.first() == Some(&1) {
            0.22
        } else {
            0.12
        };

        let mut times = Vec::with_capacity(n_4x * 2 + 4);
        let mut volts = Vec::with_capacity(n_4x * 2 + 4);
        let mut currents = Vec::with_capacity(n_4x * 2 + 4);

        let (sh0, _) = bounded_sinh_cosh(self.cap_voltage[ch] / v_th);
        times.push(-dt_max);
        volts.push(self.cap_voltage[ch] * eval_trim(0));
        currents.push(0.025 * sh0);

        for idx in 0..n_1x {
            if (idx & 63) == 0 && start_inst.elapsed() >= self.timeout {
                return Err(DnwbError::WatchdogTimeout {
                    timeout_ms: self.timeout.as_millis() as u64,
                });
            }
            let frozen = freeze_window.is_some_and(|[fs, fe]| idx >= fs && idx < fe);
            let drv_u = control_trajectories
                .and_then(|t| t.first())
                .and_then(|tr| tr.get(idx).copied())
                .unwrap_or(drive_norm);
            let drv_clean = if drv_u.is_finite() {
                f64::from(drv_u).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            let trim_gain = eval_trim(idx);
            let drive = 1.5 + 8.5 * (0.5 * (drv_clean + 1.0));
            let x_raw = f64::from(dry_1x[idx]);
            let x_sanitized = if x_raw.is_finite() { x_raw } else { 0.0 };
            let target_vin = x_sanitized * drive;

            let t_seg_start = (idx as f64) * dt_1x;
            let t_seg_end = ((idx + 1) as f64) * dt_1x;
            let mut t = t_seg_start;

            while t < t_seg_end - 1.0e-15 {
                let u_mid = ((t - t_seg_start) / dt_1x).clamp(0.0, 1.0);
                let vin_est = self.prev_vin[ch] + u_mid * (target_vin - self.prev_vin[ch]);
                let dt_adapt =
                    dt_max * (0.60 + 0.40 / (1.0 + (vin_est - self.cap_voltage[ch]).abs()));
                let dt = dt_adapt.min(t_seg_end - t).max(1.0e-12);
                let t_next = (t + dt).min(t_seg_end);
                let u_next = ((t_next - t_seg_start) / dt_1x).clamp(0.0, 1.0);
                let vin = self.prev_vin[ch] + u_next * (target_vin - self.prev_vin[ch]);

                if !frozen {
                    let g_rc = dt / TAU_RC_SEC;
                    let g_diode = dt * IS_OVER_C_VPS;
                    let mut v = self.cap_voltage[ch];
                    for _ in 0..10 {
                        let (sh, ch_d) = bounded_sinh_cosh(v / v_th);
                        let res = (v - self.cap_voltage[ch]) + g_rc * (v - vin) + g_diode * sh;
                        let jac = 1.0 + g_rc + (g_diode / v_th) * ch_d;
                        let step = (res / jac).clamp(-2.5, 2.5);
                        v -= step;
                        if step.abs() <= 1.0e-12 {
                            break;
                        }
                    }
                    self.cap_voltage[ch] = v;
                }
                let (sh_final, _) = bounded_sinh_cosh(self.cap_voltage[ch] / v_th);
                t = t_next;
                times.push(t - dt_max);
                volts.push(self.cap_voltage[ch] * trim_gain);
                currents.push(0.025 * sh_final);
            }
            self.prev_vin[ch] = target_vin;
        }

        let uniform_v_4x = pchip_resample_to_rate(&times, &volts, n_4x, 4.0 * sr);
        let uniform_i_4x = pchip_resample_to_rate(&times, &currents, n_4x, 4.0 * sr);
        let dec_v = self.decim_v[ch].process_4x_block(&uniform_v_4x);
        let dec_i = self.decim_i[ch].process_4x_block(&uniform_i_4x);
        Ok((dec_v, dec_i))
    }

    #[allow(clippy::too_many_arguments)]
    fn render_channels_in_process(
        &mut self,
        zero_phase: bool,
        stem_id: String,
        sr: u32,
        in_1: &[f32],
        in_2: Option<&[f32]>,
        in_3: Option<&[f32]>,
        in_4: Option<&[f32]>,
        cv: &[f32],
        ct: Option<&Vec<Vec<f32>>>,
        fw: Option<[usize; 2]>,
    ) -> Result<StemResponse, DnamError> {
        let map_e = |e: DnwbError| DnamError::SchemaValidationError(format!("{e:?}"));
        validate_runner_stimulus_layout(sr, in_1, in_2, in_3, in_4, cv, ct, fw).map_err(map_e)?;
        if zero_phase {
            self.reset_state();
        }
        let mut run_ch = |ch: usize, sig: &[f32]| -> Result<(Vec<f32>, Vec<f32>), DnamError> {
            if zero_phase {
                self.simulate_channel_zero_phase_ext(ch, sig, sr, cv, ct, fw)
                    .map_err(map_e)
            } else {
                self.simulate_channel_for_rate_ext(ch, sig, sr, cv, ct, fw)
                    .map_err(map_e)
            }
        };
        let (out_1, i_1) = run_ch(0, in_1)?;
        let mut internal_nodes = BTreeMap::new();
        internal_nodes.insert(String::from("spice_diode_current"), i_1);
        let out_2 = if let Some(ch2) = in_2 {
            let (v2, i2) = run_ch(1, ch2)?;
            internal_nodes.insert(String::from("spice_diode_current_r"), i2);
            Some(v2)
        } else {
            None
        };
        let out_3 = if let Some(ch3) = in_3 {
            Some(run_ch(2, ch3)?.0)
        } else {
            None
        };
        let out_4 = if let Some(ch4) = in_4 {
            Some(run_ch(3, ch4)?.0)
        } else {
            None
        };
        Ok(StemResponse {
            stem_id,
            sample_rate_hz: sr,
            out_1,
            out_2,
            out_3,
            out_4,
            internal_nodes,
        })
    }
}

impl DistillationTarget for SpiceRunner {
    fn reset_state(&mut self) {
        self.cap_voltage = [0.0; 4];
        self.prev_vin = [0.0; 4];
        for ch in 0..4 {
            self.decim_v[ch].reset();
            self.decim_i[ch].reset();
        }
        if let Some(ref cmd) = self.subprocess_cmd {
            self.worker_session
                .reset_if_live(cmd, &self.active_switches, self.timeout);
        }
    }

    fn render_stem(&mut self, stimulus: &StemStimulus) -> Result<StemResponse, DnamError> {
        if let Some(ref cmd) = self.subprocess_cmd {
            return self
                .worker_session
                .execute(
                    cmd,
                    DnwbCommand::RenderStem,
                    stimulus,
                    &self.active_switches,
                    self.timeout,
                )
                .map_err(|e| DnamError::SchemaValidationError(format!("{e:?}")));
        }
        self.render_channels_in_process(
            true,
            stimulus.stem_id.clone(),
            stimulus.sample_rate_hz,
            &stimulus.in_1,
            stimulus.in_2.as_deref(),
            stimulus.in_3.as_deref(),
            stimulus.in_4.as_deref(),
            &stimulus.control_values,
            stimulus.control_trajectories.as_ref(),
            stimulus.freeze_transition_window,
        )
    }

    fn render_frame(&mut self, frame: &StimulusFrame) -> Result<StemResponse, DnamError> {
        if let Some(ref cmd) = self.subprocess_cmd {
            let stim = StemStimulus {
                stem_id: format!("frame_{}", frame.frame_index),
                sample_rate_hz: frame.sample_rate_hz,
                in_1: frame.in_1.clone(),
                in_2: frame.in_2.clone(),
                in_3: frame.in_3.clone(),
                in_4: frame.in_4.clone(),
                control_values: frame.control_values.clone(),
                control_trajectories: frame.control_trajectories.clone(),
                freeze_transition_window: None,
            };
            return self
                .worker_session
                .execute(
                    cmd,
                    DnwbCommand::RenderFrame,
                    &stim,
                    &self.active_switches,
                    self.timeout,
                )
                .map_err(|e| DnamError::SchemaValidationError(format!("{e:?}")));
        }
        self.render_channels_in_process(
            false,
            format!("frame_{}", frame.frame_index),
            frame.sample_rate_hz,
            &frame.in_1,
            frame.in_2.as_deref(),
            frame.in_3.as_deref(),
            frame.in_4.as_deref(),
            &frame.control_values,
            frame.control_trajectories.as_ref(),
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pchip_monotonicity_duplicate_knots_and_309_tap_kaiser_stopband() {
        let tran_96k = format_spice_tran_directives_for_rate(0.01, 96_000);
        assert!(tran_96k.contains("2.604167e-6"));
        let tran_48k = format_spice_tran_directives(0.01);
        assert!(tran_48k.contains(".options maxstep="));

        assert!(pchip_resample_to_192khz(&[], &[], 10).is_empty());
        assert_eq!(pchip_resample_to_192khz(&[0.0], &[0.5], 4), vec![0.5; 4]);
        assert_eq!(
            pchip_resample_to_rate(&[0.0, 1.0], &[0.0, 1.0], 3, 2.0),
            vec![0.0, 0.5, 1.0]
        );

        // Duplicate / non-increasing SPICE event timestamps + sharp monotone step: zero overshoot!
        let times = [
            0.0,
            0.0,
            SPICE_MAXSTEP_SEC,
            2.0 * SPICE_MAXSTEP_SEC,
            2.0 * SPICE_MAXSTEP_SEC,
            5.0 * SPICE_MAXSTEP_SEC,
            9.0 * SPICE_MAXSTEP_SEC,
        ];
        let vals = [0.0, 0.0, 0.02, 0.95, 0.95, 0.99, 1.0];
        let resampled = pchip_resample_to_192khz(&times, &vals, 10);
        for (idx, &v) in resampled.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "PCHIP overshoot at sample {idx}: {v}"
            );
            if idx > 0 {
                assert!(v + 1.0e-12 >= resampled[idx - 1]);
            }
        }
        // Verify genuine > 100 dB stopband attenuation above 24 kHz and < 0.02 dB passband flatness at 18 kHz
        assert!(kaiser_sinc_decimate_4x_to_48khz(&[0.0, 0.0]).is_empty());
        let n_192 = 4096usize;
        let pass_18k: Vec<f64> = (0..n_192)
            .map(|i| (2.0 * PI * 18_000.0 * (i as f64) / 192_000.0).sin())
            .collect();
        let dec_pass = kaiser_sinc_decimate_4x_to_48khz(&pass_18k);
        let rms_pass = (dec_pass[64..960]
            .iter()
            .map(|&v| f64::from(v * v))
            .sum::<f64>()
            / 896.0)
            .sqrt();
        assert!((20.0 * (rms_pass * 2.0_f64.sqrt()).log10()).abs() < 0.05);
        for f_stop_hz in [25_000.0, 31_000.0, 60_000.0] {
            let stop_sig: Vec<f64> = (0..n_192)
                .map(|i| (2.0 * PI * f_stop_hz * (i as f64) / 192_000.0).sin())
                .collect();
            let dec_stop = kaiser_sinc_decimate_4x_to_48khz(&stop_sig);
            let peak_stop = dec_stop[64..960]
                .iter()
                .map(|&v| f64::from(v).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                peak_stop < 1.0e-5,
                "stopband leak at {f_stop_hz} Hz = {peak_stop:e}"
            );
        }
    }

    #[test]
    fn test_spice_and_faust_multirate_streaming_and_nested_ui() {
        let mut spice = SpiceRunner::new("ts808_clipper.cir", 5_000);
        for &sr in &[44_100u32, 48_000, 96_000] {
            let n = 128usize;
            let dry: Vec<f32> = (0..n)
                .map(|i| {
                    0.45 * (2.0 * std::f32::consts::PI * 330.0 * (i as f32) / (sr as f32)).sin()
                })
                .collect();
            let stim = StemStimulus {
                stem_id: format!("spice_{sr}"),
                sample_rate_hz: sr,
                in_1: dry.clone(),
                in_2: None,
                in_3: None,
                in_4: None,
                control_values: vec![0.5],
                control_trajectories: None,
                freeze_transition_window: None,
            };
            let full_resp = spice.render_stem(&stim).unwrap();
            assert_eq!((full_resp.sample_rate_hz, full_resp.out_1.len()), (sr, n));
            assert_eq!(full_resp.internal_nodes["spice_diode_current"].len(), n);

            // Streaming two 64-sample frames + causal group-delay flush (with active trim_gain != 1.0)
            let mk_fr = |frame_index: usize, in_1: Vec<f32>| StimulusFrame {
                frame_index,
                sample_rate_hz: sr,
                in_1,
                in_2: None,
                in_3: None,
                in_4: None,
                control_values: vec![0.5, 0.8],
                control_trajectories: None,
            };
            let mut stim2 = stim.clone();
            stim2.control_values = vec![0.5, 0.8];
            let full_resp2 = spice.render_stem(&stim2).unwrap();
            spice.reset_state();
            let f0 = spice.render_frame(&mk_fr(0, dry[..64].to_vec())).unwrap();
            let f1 = spice.render_frame(&mk_fr(1, dry[64..].to_vec())).unwrap();
            let f_tail = spice
                .render_frame(&mk_fr(2, vec![0.0; SPICE_CAUSAL_DELAY_SAMPLES]))
                .unwrap();
            assert_eq!(f1.internal_nodes["spice_diode_current"].len(), 64);
            let mut joined = f0.out_1;
            joined.extend_from_slice(&f1.out_1);
            joined.extend_from_slice(&f_tail.out_1);
            let aligned = &joined[SPICE_CAUSAL_DELAY_SAMPLES..SPICE_CAUSAL_DELAY_SAMPLES + n];
            let max_diff = aligned
                .iter()
                .zip(&full_resp2.out_1)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(max_diff < 5.0e-4, "SPICE mismatch at {sr} Hz: {max_diff}");
        }

        // +40 dBFS extreme overdrive + NaN control sanitization + extreme PCHIP finite-knot guard
        let (hot_v, hot_i) = spice
            .simulate_adaptive_spice_stem(&[100.0_f32; 32], f32::NAN)
            .unwrap();
        assert!(hot_v.iter().all(|v| v.is_finite()) && hot_i.iter().all(|i| i.is_finite()));
        let t_tiny = [0.0, 1.0e-12, 2.0e-12, 3.0e-12];
        let y_huge = [-1.0e308, 1.0e308, -1.0e308, 1.0e308];
        assert!(
            pchip_resample_to_rate(&t_tiny, &y_huge, 6, 1.0e12)
                .iter()
                .all(|v| v.is_finite())
        );
        assert!(
            SpiceRunner::new("timeout.cir", 0)
                .simulate_adaptive_spice_stem(&[0.1; 8], 0.0)
                .is_err()
        );
    }
}
