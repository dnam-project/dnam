//! Hardware Interface, Converter Loopback, Human Hand Trajectory, Pot Taper, and
//! Polarity/DC Offset Fault Decorators (`Cases #1..#4, #6, #7, #10, #11`).

#![forbid(unsafe_code)]

use super::{
    impl_output_decorator_common, init_inner_float32, resolve_controls_with_defaults,
    step_inner_frame, write_outputs,
};
use crate::{
    BlackBoxModel, FixtureBitDepth, FixtureDeviceMetadata, clamp_sample_rate_hz, sanitize_norm,
    sanitize_sample,
};
use std::f64::consts::PI;

// Case #1: ADC Hard Clipping & Clean Pad Staging
/// `Case #1` (`AdcClipAndPadDecorator`): Hard-clips wet return at the converter ADC full-scale rail
/// and applies pre-ADC pad attenuation (`clean_pad_db`) with optional inverse post-gain restoration.
#[derive(Clone, Debug)]
pub struct AdcClipAndPadDecorator<M: BlackBoxModel> {
    pub inner: M,
    pub clean_pad_db: f32,
    /// When `true` (default), restores inverse pad gain `10^{+|pad|/20}` digitally after the ADC;
    /// when `false`, leaves the attenuated/uncalibrated pad level intact.
    pub restore_pad_gain: bool,
    adc_bit_depth: FixtureBitDepth,
    /// Clipped sample count observed in the most recent `process_frame` call.
    pub last_frame_clipped_samples: usize,
    /// Cumulative clipped sample count since construction or `reset()`.
    pub last_clipped_samples: usize,
}

impl<M: BlackBoxModel> AdcClipAndPadDecorator<M> {
    #[must_use]
    pub fn new(mut inner: M, clean_pad_db: f32) -> Self {
        let adc_bit_depth = inner.bit_depth();
        inner.set_bit_depth(FixtureBitDepth::Float32);
        Self {
            inner,
            clean_pad_db,
            restore_pad_gain: true,
            adc_bit_depth,
            last_frame_clipped_samples: 0,
            last_clipped_samples: 0,
        }
    }

    /// Creates an uncalibrated pad decorator where pre-ADC attenuation is not digitally compensated.
    #[must_use]
    pub fn new_uncalibrated_pad(inner: M, clean_pad_db: f32) -> Self {
        let mut s = Self::new(inner, clean_pad_db);
        s.restore_pad_gain = false;
        s
    }
}

impl<M: BlackBoxModel + Default> Default for AdcClipAndPadDecorator<M> {
    fn default() -> Self {
        Self::new(M::default(), 0.0)
    }
}

impl<M: BlackBoxModel> BlackBoxModel for AdcClipAndPadDecorator<M> {
    fn reset(&mut self) {
        self.inner.reset();
        self.last_frame_clipped_samples = 0;
        self.last_clipped_samples = 0;
    }

    impl_output_decorator_common!(adc_bit_depth);

    fn process_frame(
        &mut self,
        inputs: &[f32],
        outputs: &mut [f32],
        controls: &[f32],
        switches: &[u8],
    ) {
        let mut out_buf = [0.0_f32; 4];
        let model_out = step_inner_frame(&mut self.inner, inputs, controls, switches, &mut out_buf);
        let pad_db = if self.clean_pad_db.is_finite() {
            self.clean_pad_db.abs().min(80.0)
        } else {
            0.0
        };
        let pad_lin = 10.0_f64.powf(-f64::from(pad_db) / 20.0);
        let inv_pad = if self.restore_pad_gain {
            1.0 / pad_lin.max(1.0e-6)
        } else {
            1.0
        };
        let bit_depth = self.adc_bit_depth;
        let pos_rail = if bit_depth == FixtureBitDepth::Float32 {
            1.0_f64
        } else {
            bit_depth.positive_full_scale_f64()
        };
        self.last_frame_clipped_samples = 0;
        for s in out_buf.iter_mut().take(model_out) {
            let adc_in = f64::from(crate::sanitize_sample(*s)) * pad_lin;
            if adc_in >= pos_rail || adc_in <= -1.0 {
                self.last_frame_clipped_samples += 1;
                self.last_clipped_samples += 1;
            }
            let adc_clipped = bit_depth.quantize_sample_f64(adc_in.clamp(-1.0, pos_rail));
            *s = crate::sanitize_sample((adc_clipped * inv_pad) as f32);
        }
        write_outputs(outputs, &out_buf, model_out);
    }
}

// Case #2: Converter Loopback Latency, 20 Hz HPF Tilt, Clock Drift & USB Drop
const LOOPBACK_RING_LEN: usize = 4_096;
const LOOPBACK_RING_MASK: usize = LOOPBACK_RING_LEN - 1;

/// `Case #2` (`ConverterLoopbackDecorator`): Injects fractional loopback latency (`tau_hw = 143.62`
/// samples at `48 kHz`), `20 Hz` AC-coupling HPF phase tilt, `+45 ppm` clock drift, optional USB drop,
/// and optional converter sample-rate / PCM bit-depth quantization.
#[derive(Clone, Debug)]
pub struct ConverterLoopbackDecorator<M: BlackBoxModel> {
    pub inner: M,
    pub delay_samples: f64,
    pub clock_drift_ppm: f64,
    pub usb_drop_at_sample: Option<usize>,
    converter_sample_rate_hz: Option<u32>,
    converter_bit_depth: Option<FixtureBitDepth>,
    default_bit_depth: FixtureBitDepth,
    anchor_delay_samples: f64,
    anchor_sample_rate_hz: u32,
    cached_delay_samples: f64,
    anchor_usb_drop: Option<(usize, u32)>,
    cached_usb_drop_at_sample: Option<usize>,
    rings: [Vec<f32>; 4],
    wr_idx: usize,
    sample_index: usize,
    hpf_states: [f64; 4],
}

impl<M: BlackBoxModel> ConverterLoopbackDecorator<M> {
    fn new_internal(
        mut inner: M,
        converter_sr: Option<u32>,
        converter_bd: Option<FixtureBitDepth>,
    ) -> Self {
        let orig_bd = inner.bit_depth();
        if let Some(sr) = converter_sr {
            inner.set_sample_rate_hz(sr);
        }
        inner.set_bit_depth(FixtureBitDepth::Float32);
        let eff_sr = clamp_sample_rate_hz(converter_sr.unwrap_or_else(|| inner.sample_rate_hz()));
        let delay_samples = if eff_sr == 48_000 {
            143.62
        } else {
            143.62 * (f64::from(eff_sr) / 48_000.0)
        };
        Self {
            inner,
            delay_samples,
            clock_drift_ppm: 45.0,
            usb_drop_at_sample: None,
            converter_sample_rate_hz: converter_sr,
            converter_bit_depth: converter_bd,
            default_bit_depth: converter_bd.unwrap_or(orig_bd),
            anchor_delay_samples: 143.62,
            anchor_sample_rate_hz: 48_000,
            cached_delay_samples: delay_samples,
            anchor_usb_drop: None,
            cached_usb_drop_at_sample: None,
            rings: std::array::from_fn(|_| vec![0.0; LOOPBACK_RING_LEN]),
            wr_idx: 0,
            sample_index: 0,
            hpf_states: [0.0; 4],
        }
    }

    #[must_use]
    pub fn new(inner: M) -> Self {
        Self::new_internal(inner, None, None)
    }

    /// Creates a loopback converter operating at an explicit sample rate and fixed/float bit depth.
    #[must_use]
    pub fn with_converter_format(
        inner: M,
        sample_rate_hz: u32,
        bit_depth: FixtureBitDepth,
    ) -> Self {
        Self::new_internal(
            inner,
            Some(clamp_sample_rate_hz(sample_rate_hz)),
            Some(bit_depth),
        )
    }
}

impl<M: BlackBoxModel + Default> Default for ConverterLoopbackDecorator<M> {
    fn default() -> Self {
        Self::new(M::default())
    }
}

impl<M: BlackBoxModel> BlackBoxModel for ConverterLoopbackDecorator<M> {
    fn reset(&mut self) {
        self.inner.reset();
        for ring in &mut self.rings {
            ring.fill(0.0);
        }
        self.wr_idx = 0;
        self.sample_index = 0;
        self.hpf_states = [0.0; 4];
    }

    fn metadata(&self) -> FixtureDeviceMetadata {
        let mut meta = self.inner.metadata();
        if let Some(sr) = self.converter_sample_rate_hz {
            meta.sample_rate_hz = clamp_sample_rate_hz(sr);
        }
        meta.bit_depth = self.converter_bit_depth.unwrap_or(self.default_bit_depth);
        meta
    }

    fn set_sample_rate_hz(&mut self, sample_rate_hz: u32) {
        let sr = clamp_sample_rate_hz(sample_rate_hz);
        let prev_sr = clamp_sample_rate_hz(self.sample_rate_hz());
        if self.delay_samples.to_bits() != self.cached_delay_samples.to_bits() {
            self.anchor_delay_samples = if self.delay_samples.is_finite() {
                self.delay_samples.max(0.0)
            } else {
                0.0
            };
            self.anchor_sample_rate_hz = prev_sr;
        }
        if self.usb_drop_at_sample != self.cached_usb_drop_at_sample {
            self.anchor_usb_drop = self.usb_drop_at_sample.map(|idx| (idx, prev_sr));
        }
        self.inner.set_sample_rate_hz(sr);
        if self.converter_sample_rate_hz.is_some() {
            self.converter_sample_rate_hz = Some(sr);
        }
        let actual_sr = clamp_sample_rate_hz(self.sample_rate_hz());
        let scaled = if actual_sr == self.anchor_sample_rate_hz {
            self.anchor_delay_samples
        } else {
            self.anchor_delay_samples
                * (f64::from(actual_sr) / f64::from(self.anchor_sample_rate_hz))
        };
        self.delay_samples = scaled;
        self.cached_delay_samples = scaled;
        self.usb_drop_at_sample = self.anchor_usb_drop.map(|(idx, a_sr)| {
            if actual_sr == a_sr {
                idx
            } else {
                ((idx as f64) * (f64::from(actual_sr) / f64::from(a_sr))).round() as usize
            }
        });
        self.cached_usb_drop_at_sample = self.usb_drop_at_sample;
        self.reset();
    }

    fn set_bit_depth(&mut self, bit_depth: FixtureBitDepth) {
        self.inner.set_bit_depth(FixtureBitDepth::Float32);
        self.default_bit_depth = bit_depth;
        if self.converter_bit_depth.is_some() {
            self.converter_bit_depth = Some(bit_depth);
        }
        self.reset();
    }

    fn num_inputs(&self) -> usize {
        self.inner.num_inputs()
    }

    fn num_outputs(&self) -> usize {
        self.inner.num_outputs()
    }

    fn process_frame(
        &mut self,
        inputs: &[f32],
        outputs: &mut [f32],
        controls: &[f32],
        switches: &[u8],
    ) {
        let mut out_buf = [0.0_f32; 4];
        let model_out = step_inner_frame(&mut self.inner, inputs, controls, switches, &mut out_buf);
        let bit_depth = self.bit_depth();
        for (ch, &s) in out_buf.iter().enumerate().take(model_out) {
            self.rings[ch][self.wr_idx & LOOPBACK_RING_MASK] = sanitize_sample(s);
        }

        let idx = self.sample_index;
        self.sample_index = self.sample_index.wrapping_add(1);
        let fs = f64::from(clamp_sample_rate_hz(self.sample_rate_hz()));
        let drop_len = ((64.0 * fs / 48_000.0).round() as usize).max(1);

        if let Some(drop_idx) = self.usb_drop_at_sample
            && idx >= drop_idx
            && idx < drop_idx.saturating_add(drop_len)
        {
            out_buf[..model_out].fill(0.0);
            self.wr_idx = self.wr_idx.wrapping_add(1);
            write_outputs(outputs, &out_buf, model_out);
            return;
        }

        let max_del = (LOOPBACK_RING_LEN - 4) as f64;
        let drift = if self.clock_drift_ppm.is_finite() {
            self.clock_drift_ppm.clamp(-5_000.0, 5_000.0)
        } else {
            0.0
        };
        let base_del = if self.delay_samples.is_finite() {
            self.delay_samples
        } else {
            0.0
        };
        let eff_delay = (base_del - (idx as f64) * drift * 1.0e-6).clamp(0.0, max_del);
        let d_int = eff_delay.floor() as usize;
        let frac = eff_delay - (d_int as f64);
        let i0 = self.wr_idx.wrapping_sub(d_int) & LOOPBACK_RING_MASK;
        let i1 = self.wr_idx.wrapping_sub(d_int + 1) & LOOPBACK_RING_MASK;
        self.wr_idx = self.wr_idx.wrapping_add(1);

        let a_hpf = 1.0 - (-2.0 * PI * 20.0 / fs).exp();
        for (ch, out) in out_buf.iter_mut().enumerate().take(model_out) {
            let del =
                (1.0 - frac) * f64::from(self.rings[ch][i0]) + frac * f64::from(self.rings[ch][i1]);
            self.hpf_states[ch] += a_hpf * (del - self.hpf_states[ch]);
            if self.hpf_states[ch].abs() < f64::MIN_POSITIVE {
                self.hpf_states[ch] = 0.0;
            }
            *out = bit_depth.quantize_f64_to_f32(del - self.hpf_states[ch]);
        }
        write_outputs(outputs, &out_buf, model_out);
    }
}

// Case #4: Human Hand Trajectory Decorator (`150-400 ms` Lag & Damped Overshoot)
const MAX_HAND_LAG_SAMPLES: usize = 384_000;

/// `Case #4` (`HumanHandTrajectoryDecorator`): Perturbs ideal control trajectories with human reaction
/// lag (`lag_samples`), `~8 Hz` neuromuscular wrist bandwidth, and calibrated mechanical overshoot
/// (`overshoot_ratio`, e.g. `0.02` for `2%` peak overshoot) before settling.
#[derive(Clone, Debug)]
pub struct HumanHandTrajectoryDecorator<M: BlackBoxModel> {
    pub inner: M,
    pub lag_samples: usize,
    pub overshoot_ratio: f32,
    pub wrist_bandwidth_hz: f32,
    lag_seconds: f64,
    cached_lag_samples: usize,
    pos_state: Vec<f64>,
    vel_state: Vec<f64>,
    delay_ring: Vec<Vec<f32>>,
    wr_idx: usize,
    initialized: bool,
}

impl<M: BlackBoxModel> HumanHandTrajectoryDecorator<M> {
    #[must_use]
    pub fn new(inner: M, lag_samples: usize, overshoot_ratio: f32) -> Self {
        let sr = clamp_sample_rate_hz(inner.sample_rate_hz());
        let clamped_lag = lag_samples.min(MAX_HAND_LAG_SAMPLES);
        let n_init = inner.metadata().controls.len();
        Self {
            inner,
            lag_samples: clamped_lag,
            overshoot_ratio,
            wrist_bandwidth_hz: 8.0,
            lag_seconds: (clamped_lag as f64) / f64::from(sr),
            cached_lag_samples: clamped_lag,
            pos_state: vec![0.0; n_init],
            vel_state: vec![0.0; n_init],
            delay_ring: vec![vec![0.0; n_init]; clamped_lag + 1],
            wr_idx: 0,
            initialized: false,
        }
    }

    #[inline]
    fn spring_coefficients(&self) -> (f64, f64) {
        let fs = f64::from(clamp_sample_rate_hz(self.sample_rate_hz()));
        let fn_hz = if self.wrist_bandwidth_hz.is_finite() {
            f64::from(self.wrist_bandwidth_hz).clamp(1.0, 40.0)
        } else {
            8.0
        };
        let omega_n = (2.0 * PI * fn_hz / fs).clamp(1.0e-6, 0.25);
        let mp = if self.overshoot_ratio.is_finite() {
            f64::from(self.overshoot_ratio).clamp(0.0, 0.85)
        } else {
            0.0
        };
        let zeta = if mp <= 1.0e-5 {
            1.0
        } else {
            let ln_mp = mp.ln();
            (-ln_mp / (PI * PI + ln_mp * ln_mp).sqrt()).clamp(0.08, 1.0)
        };
        (omega_n * omega_n, 2.0 * zeta * omega_n)
    }

    #[inline]
    fn step_spring(pos: &mut f64, vel: &mut f64, target: f64, k_spring: f64, c_damp: f64) -> f32 {
        *vel += k_spring * (target - *pos) - c_damp * *vel;
        if vel.abs() < f64::MIN_POSITIVE {
            *vel = 0.0;
        }
        let next = *pos + *vel;
        if next > 1.0 {
            *pos = 1.0;
            *vel = (*vel).min(0.0);
        } else if next < -1.0 {
            *pos = -1.0;
            *vel = (*vel).max(0.0);
        } else {
            *pos = if next.abs() < f64::MIN_POSITIVE {
                0.0
            } else {
                next
            };
        }
        crate::sanitize_norm(*pos as f32, 0.0)
    }

    /// Perturbs an ideal single-knob control trajectory `ideal` with reaction lag and calibrated overshoot.
    #[must_use]
    pub fn perturb_trajectory(&self, ideal: &[f32]) -> Vec<f32> {
        let n = ideal.len();
        if n == 0 {
            return Vec::new();
        }
        let mut out = vec![0.0_f32; n];
        let (k_spring, c_damp) = self.spring_coefficients();
        let eff_lag = self.lag_samples.min(MAX_HAND_LAG_SAMPLES);
        let mut pos = f64::from(sanitize_norm(ideal[0], 0.0));
        let mut vel = 0.0_f64;
        for i in 0..n {
            let target = f64::from(sanitize_norm(ideal[i.saturating_sub(eff_lag)], 0.0));
            out[i] = Self::step_spring(&mut pos, &mut vel, target, k_spring, c_damp);
        }
        out
    }
}

impl<M: BlackBoxModel + Default> Default for HumanHandTrajectoryDecorator<M> {
    fn default() -> Self {
        Self::new(M::default(), 0, 0.02)
    }
}

impl<M: BlackBoxModel> BlackBoxModel for HumanHandTrajectoryDecorator<M> {
    fn reset(&mut self) {
        self.inner.reset();
        self.pos_state.fill(0.0);
        self.vel_state.fill(0.0);
        for slot in &mut self.delay_ring {
            slot.fill(0.0);
        }
        self.wr_idx = 0;
        self.initialized = false;
    }

    fn metadata(&self) -> FixtureDeviceMetadata {
        self.inner.metadata()
    }

    fn set_sample_rate_hz(&mut self, sample_rate_hz: u32) {
        let sr = clamp_sample_rate_hz(sample_rate_hz);
        let prev_sr = clamp_sample_rate_hz(self.sample_rate_hz());
        if self.lag_samples != self.cached_lag_samples {
            self.lag_seconds =
                (self.lag_samples.min(MAX_HAND_LAG_SAMPLES) as f64) / f64::from(prev_sr);
        }
        self.inner.set_sample_rate_hz(sr);
        let actual_sr = clamp_sample_rate_hz(self.sample_rate_hz());
        let scaled =
            ((self.lag_seconds * f64::from(actual_sr)).round() as usize).min(MAX_HAND_LAG_SAMPLES);
        self.lag_samples = scaled;
        self.cached_lag_samples = scaled;
        let n_init = self.pos_state.len();
        self.delay_ring.resize(scaled + 1, vec![0.0; n_init]);
        self.reset();
    }

    fn set_bit_depth(&mut self, bit_depth: FixtureBitDepth) {
        self.inner.set_bit_depth(bit_depth);
        self.reset();
    }

    fn num_inputs(&self) -> usize {
        self.inner.num_inputs()
    }

    fn num_outputs(&self) -> usize {
        self.inner.num_outputs()
    }

    fn process_frame(
        &mut self,
        inputs: &[f32],
        outputs: &mut [f32],
        controls: &[f32],
        switches: &[u8],
    ) {
        let resolved = resolve_controls_with_defaults(&self.inner, controls);
        let n_c = resolved.len();
        let eff_lag = self.lag_samples.min(MAX_HAND_LAG_SAMPLES);
        if !self.initialized {
            self.pos_state = resolved.iter().map(|&v| f64::from(v)).collect();
            self.vel_state = vec![0.0; n_c];
            self.delay_ring = vec![resolved.clone(); eff_lag + 1];
            self.wr_idx = 0;
            self.initialized = true;
        } else {
            if n_c > self.pos_state.len() {
                for &val in &resolved[self.pos_state.len()..n_c] {
                    self.pos_state.push(f64::from(val));
                    self.vel_state.push(0.0);
                    for slot in &mut self.delay_ring {
                        slot.push(val);
                    }
                }
            }
            if self.delay_ring.len() <= eff_lag {
                self.delay_ring.rotate_left(self.wr_idx);
                let oldest = self.delay_ring[0].clone();
                let extra = (eff_lag + 1) - self.delay_ring.len();
                self.delay_ring.resize(eff_lag + 1, oldest);
                self.delay_ring.rotate_right(extra);
                self.wr_idx = 0;
            }
        }
        let ring_len = self.delay_ring.len();
        let state_len = self.pos_state.len();
        self.delay_ring[self.wr_idx].resize(state_len, 0.0);
        self.delay_ring[self.wr_idx][..n_c].copy_from_slice(&resolved);
        self.delay_ring[self.wr_idx][n_c..].fill(0.0);
        let rd_idx = (self.wr_idx + ring_len - eff_lag) % ring_len;
        let (k_spring, c_damp) = self.spring_coefficients();
        let mut smoothed = vec![0.0_f32; n_c];
        for p in 0..state_len {
            let fallback = resolved.get(p).copied().unwrap_or(0.0);
            let target = f64::from(self.delay_ring[rd_idx].get(p).copied().unwrap_or(fallback));
            let val = Self::step_spring(
                &mut self.pos_state[p],
                &mut self.vel_state[p],
                target,
                k_spring,
                c_damp,
            );
            if let Some(slot) = smoothed.get_mut(p) {
                *slot = val;
            }
        }
        self.wr_idx = (self.wr_idx + 1) % ring_len;
        let mut out_buf = [0.0_f32; 4];
        let model_out =
            step_inner_frame(&mut self.inner, inputs, &smoothed, switches, &mut out_buf);
        write_outputs(outputs, &out_buf, model_out);
    }
}

// Case #7: Potentiometer Taper & Multi-Throw Switch Decorator (`15%A`, `85%C`, `W`)
/// Physical potentiometer resistance taper curves (`Case #7`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaperCurveKind {
    /// `15%A` audio log taper: $f(x) = x^{3.2}$.
    Log15A,
    /// `85%C` reverse-log taper: $f(x) = 1 - (1 - x)^{3.2}$.
    RevLog85C,
    /// `W`-taper S-curve tone pot.
    WTaper,
}

impl TaperCurveKind {
    /// Maps physical shaft angle $x \in [0, 1]$ to electrical normalized coordinate $u \in [-1, 1]$.
    #[must_use]
    pub fn map_shaft_to_electrical(self, shaft_01: f32) -> f32 {
        let x = if shaft_01.is_finite() {
            shaft_01.clamp(0.0, 1.0)
        } else {
            0.5
        };
        let elec_01 = match self {
            Self::Log15A => x.powf(3.2),
            Self::RevLog85C => 1.0 - (1.0 - x).powf(3.2),
            Self::WTaper => 0.5 + 0.5 * ((x - 0.5) * 3.5).tanh() / (1.75_f32).tanh(),
        };
        2.0 * elec_01 - 1.0
    }
}

/// `Case #7` (`PotTaperAndSwitchDecorator`): Applies non-linear pot tapers (`15%A`, `85%C`, `W`) before the inner model.
#[derive(Clone, Debug)]
pub struct PotTaperAndSwitchDecorator<M: BlackBoxModel> {
    pub inner: M,
    pub tapers: Vec<TaperCurveKind>,
}

impl<M: BlackBoxModel> PotTaperAndSwitchDecorator<M> {
    #[must_use]
    pub fn new(inner: M, tapers: Vec<TaperCurveKind>) -> Self {
        Self { inner, tapers }
    }
}

impl<M: BlackBoxModel + Default> Default for PotTaperAndSwitchDecorator<M> {
    fn default() -> Self {
        Self::new(M::default(), Vec::new())
    }
}

impl<M: BlackBoxModel> BlackBoxModel for PotTaperAndSwitchDecorator<M> {
    fn reset(&mut self) {
        self.inner.reset();
    }

    fn metadata(&self) -> FixtureDeviceMetadata {
        self.inner.metadata()
    }

    fn set_sample_rate_hz(&mut self, sample_rate_hz: u32) {
        self.inner.set_sample_rate_hz(sample_rate_hz);
        self.reset();
    }

    fn set_bit_depth(&mut self, bit_depth: FixtureBitDepth) {
        self.inner.set_bit_depth(bit_depth);
        self.reset();
    }

    fn num_inputs(&self) -> usize {
        self.inner.num_inputs()
    }

    fn num_outputs(&self) -> usize {
        self.inner.num_outputs()
    }

    fn process_frame(
        &mut self,
        inputs: &[f32],
        outputs: &mut [f32],
        controls: &[f32],
        switches: &[u8],
    ) {
        let mut mapped = resolve_controls_with_defaults(&self.inner, controls);
        for (p, slot) in mapped.iter_mut().enumerate() {
            if let Some(&taper) = self.tapers.get(p) {
                *slot = taper.map_shaft_to_electrical(0.5 * (*slot + 1.0));
            }
        }
        let mut out_buf = [0.0_f32; 4];
        let model_out = step_inner_frame(&mut self.inner, inputs, &mapped, switches, &mut out_buf);
        write_outputs(outputs, &out_buf, model_out);
    }
}

// Case #11: Polarity Inversion, DC Offset & Numeric Envelope Sanitization
/// `Case #11` (`PolarityAndDcOffsetDecorator`): Injects `s_pol = -1` polarity inversion, `+0.015` DC offset,
/// `+-100.0` (`+40 dBFS`) unclipped operating envelope sanitization, `+-10^{30}` / `NaN` fault recovery,
/// and `10^{-38}` denormal flushing.
#[derive(Clone, Debug)]
pub struct PolarityAndDcOffsetDecorator<M: BlackBoxModel> {
    pub inner: M,
    pub invert_polarity: bool,
    pub dc_offset: f32,
    out_bit_depth: FixtureBitDepth,
}

impl<M: BlackBoxModel> PolarityAndDcOffsetDecorator<M> {
    #[must_use]
    pub fn new(inner: M, invert_polarity: bool, dc_offset: f32) -> Self {
        let (inner, out_bit_depth) = init_inner_float32(inner);
        Self {
            inner,
            invert_polarity,
            dc_offset,
            out_bit_depth,
        }
    }

    /// Sanitizes an input sample against `NaN`, `+-Inf`, denormals (`|x| < f32::MIN_POSITIVE`), and `+-100.0` (`+40 dBFS`) rails.
    #[inline]
    #[must_use]
    pub fn sanitize_sample(x: f32) -> (f32, bool) {
        const ENVELOPE_LIMIT: f32 = 100.0;
        const DENORMAL_FLOOR: f32 = f32::MIN_POSITIVE;
        if !x.is_finite() {
            return (0.0, true);
        }
        if x.abs() < DENORMAL_FLOOR {
            return (0.0, false);
        }
        if x.abs() > ENVELOPE_LIMIT {
            return (x.clamp(-ENVELOPE_LIMIT, ENVELOPE_LIMIT), true);
        }
        (x, false)
    }
}

impl<M: BlackBoxModel + Default> Default for PolarityAndDcOffsetDecorator<M> {
    fn default() -> Self {
        Self::new(M::default(), false, 0.0)
    }
}

impl<M: BlackBoxModel> BlackBoxModel for PolarityAndDcOffsetDecorator<M> {
    fn reset(&mut self) {
        self.inner.reset();
    }

    impl_output_decorator_common!(out_bit_depth);

    fn process_frame(
        &mut self,
        inputs: &[f32],
        outputs: &mut [f32],
        controls: &[f32],
        switches: &[u8],
    ) {
        let n_in = inputs.len().min(self.num_inputs()).min(4);
        let mut clean_in = [0.0_f32; 4];
        for (dst, &src) in clean_in.iter_mut().zip(inputs.iter()).take(n_in) {
            *dst = Self::sanitize_sample(src).0;
        }
        let mut out_buf = [0.0_f32; 4];
        let model_out = step_inner_frame(
            &mut self.inner,
            &clean_in[..n_in],
            controls,
            switches,
            &mut out_buf,
        );
        let pol = if self.invert_polarity {
            -1.0_f64
        } else {
            1.0_f64
        };
        let dc = f64::from(sanitize_sample(self.dc_offset));
        for s in out_buf.iter_mut().take(model_out) {
            *s = self
                .out_bit_depth
                .quantize_f64_to_f32(pol * f64::from(sanitize_sample(*s)) + dc);
        }
        write_outputs(outputs, &out_buf, model_out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chains::{Chain01SymmetricOd, Chain07StereoChorusAndRotary};
    use crate::test_utils::{
        exercise_blackbox_metadata_and_format, run_mono_block, run_stereo_block, sine_wave,
        sine_wave_at_rate,
    };

    #[test]
    fn test_interface_and_control_decorators_unit() {
        use FixtureBitDepth::{Int16, Int24};
        use TaperCurveKind::{Log15A, RevLog85C, WTaper};
        let dry = sine_wave(256, 330.0, 0.5);
        let ctrls = [0.6_f32, 0.2, 0.5];
        let mut d1 = AdcClipAndPadDecorator::new(Chain01SymmetricOd::new(), 0.0);
        exercise_blackbox_metadata_and_format(&mut d1);
        assert_eq!(d1.metadata().id, "Chain-01-Symmetric-OD");
        let y_d1 = run_mono_block(&mut d1, &dry, &[1.0, 1.0, 1.0], &[]);
        assert!(y_d1.iter().all(|v| v.abs() <= 1.0));
        let mut d1_cal = AdcClipAndPadDecorator::new(Chain01SymmetricOd::new(), 6.0);
        let od_u = Chain01SymmetricOd::new();
        let mut d1_uncal = AdcClipAndPadDecorator::new_uncalibrated_pad(od_u, 6.0);
        let y_cal = run_mono_block(&mut d1_cal, &dry, &[0.0, 0.0, 0.0], &[]);
        let y_uncal = run_mono_block(&mut d1_uncal, &dry, &[0.0, 0.0, 0.0], &[]);
        let rms_cal: f64 = y_cal.iter().map(|&v| f64::from(v * v)).sum();
        let rms_uncal: f64 = y_uncal.iter().map(|&v| f64::from(v * v)).sum();
        assert!(rms_cal > 3.5 * rms_uncal);
        let hot_inner = PolarityAndDcOffsetDecorator::new(Chain01SymmetricOd::new(), false, 2.0);
        let mut d1_hot = AdcClipAndPadDecorator::new(hot_inner, 0.0);
        let _ = run_mono_block(&mut d1_hot, &dry, &ctrls, &[]);
        assert_eq!(d1_hot.last_clipped_samples, dry.len());
        assert_eq!(d1_hot.last_frame_clipped_samples, 1);
        let mut d2 = ConverterLoopbackDecorator::new(Chain07StereoChorusAndRotary::new());
        exercise_blackbox_metadata_and_format(&mut d2);
        d2.usb_drop_at_sample = Some(32);
        assert_eq!(d2.metadata().id, "Chain-07-Stereo-Chorus-And-Rotary");
        assert_eq!((d2.num_inputs(), d2.num_outputs()), (2, 2));
        let (y_d2_l, y_d2_r) = run_stereo_block(&mut d2, &dry, &dry, &ctrls, &[1]);
        assert!(y_d2_l[40] == 0.0 && y_d2_r[40] == 0.0 && y_d2_l[220].abs() > 0.0);
        let od96 = Chain01SymmetricOd::new();
        let mut d2_96k = ConverterLoopbackDecorator::with_converter_format(od96, 96_000, Int16);
        assert_eq!(
            (d2_96k.sample_rate_hz(), d2_96k.bit_depth()),
            (96_000, Int16)
        );
        let dry96 = sine_wave_at_rate(512, 330.0, 0.5, 96_000);
        let y_96k = run_mono_block(&mut d2_96k, &dry96, &ctrls, &[]);
        assert!(y_96k.iter().all(|&v| v == Int16.quantize_sample(v)));
        let mut d3_low_os = HumanHandTrajectoryDecorator::new(Chain01SymmetricOd::new(), 12, 0.01);
        let mut d3_high_os = HumanHandTrajectoryDecorator::new(Chain01SymmetricOd::new(), 12, 0.25);
        d3_low_os.wrist_bandwidth_hz = 40.0;
        d3_high_os.wrist_bandwidth_hz = 40.0;
        let ideal: Vec<f32> = (0..1_024)
            .map(|i| if i < 16 { -0.5 } else { 0.5 })
            .collect();
        let traj_low = d3_low_os.perturb_trajectory(&ideal);
        let traj_high = d3_high_os.perturb_trajectory(&ideal);
        assert!(traj_low[..28].iter().all(|&v| (v - (-0.5)).abs() < 1.0e-6));
        let peak_low = traj_low.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let peak_high = traj_high.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(peak_low > 0.5 && peak_high > peak_low + 0.08);
        exercise_blackbox_metadata_and_format(&mut d3_low_os);
        let y_d3 = run_mono_block(&mut d3_low_os, &dry, &[], &[]);
        assert!(y_d3.iter().all(|v| v.is_finite()));

        let od5 = Chain01SymmetricOd::new();
        let mut d5 = PotTaperAndSwitchDecorator::new(od5, vec![Log15A, RevLog85C, WTaper]);
        exercise_blackbox_metadata_and_format(&mut d5);
        let y_d5 = run_mono_block(&mut d5, &dry, &[0.1, -0.2, 0.3], &[]);
        assert!(y_d5.iter().all(|v| v.is_finite()));
        type PolOd = PolarityAndDcOffsetDecorator<Chain01SymmetricOd>;
        let mut d6 = PolarityAndDcOffsetDecorator::new(Chain01SymmetricOd::new(), true, 0.015);
        exercise_blackbox_metadata_and_format(&mut d6);
        assert_eq!(PolOd::sanitize_sample(100.0), (100.0, false));
        assert_eq!(PolOd::sanitize_sample(1.0e30), (100.0, true));
        assert_eq!(PolOd::sanitize_sample(f32::NAN), (0.0, true));
        assert_eq!(PolOd::sanitize_sample(1.0e-40), (0.0, false));
        let mut fault_in = dry.clone();
        fault_in[10] = f32::NAN;
        fault_in[20] = 1.0e30;
        let y_fault = run_mono_block(&mut d6, &fault_in, &ctrls, &[]);
        assert!(y_fault.iter().all(|v| v.is_finite()));
        d6.reset();
        let y_d6 = run_mono_block(&mut d6, &dry, &ctrls, &[]);
        let mut d6_pos = PolarityAndDcOffsetDecorator::new(Chain01SymmetricOd::new(), false, 0.0);
        let y_d6_pos = run_mono_block(&mut d6_pos, &dry, &ctrls, &[]);
        assert!((y_d6[64] - (-y_d6_pos[64] + 0.015)).abs() < 1.0e-6);

        struct PassThrough;
        impl BlackBoxModel for PassThrough {
            fn reset(&mut self) {}
            fn metadata(&self) -> FixtureDeviceMetadata {
                Chain01SymmetricOd::new().metadata()
            }
            fn num_inputs(&self) -> usize {
                1
            }
            fn num_outputs(&self) -> usize {
                1
            }
            fn process_frame(&mut self, ins: &[f32], outs: &mut [f32], _: &[f32], _: &[u8]) {
                outs[0] = ins.first().copied().unwrap_or(0.0);
            }
        }
        let mut lb_fixed =
            ConverterLoopbackDecorator::with_converter_format(PassThrough, 96_000, Int24);
        assert!((lb_fixed.delay_samples - 287.24).abs() < 1.0e-9);
        lb_fixed.set_sample_rate_hz(96_000);
        assert!((lb_fixed.delay_samples - 287.24).abs() < 1.0e-9);
        let mut pad50 = AdcClipAndPadDecorator::new(PassThrough, 50.0);
        let mut pad42 = AdcClipAndPadDecorator::new(PassThrough, 42.0);
        let mut out_pad = [0.0_f32; 1];
        pad50.process_frame(&[250.0], &mut out_pad, &[], &[]);
        assert!(pad50.last_clipped_samples == 0 && (out_pad[0] - 250.0).abs() < 0.25);
        pad42.process_frame(&[130.0], &mut out_pad, &[], &[]);
        assert_eq!(pad42.last_clipped_samples, 1);
        let orig_del = d2.delay_samples;
        for &sr in &[44_100, 96_000, 48_000] {
            d2.set_sample_rate_hz(sr);
        }
        assert_eq!(d2.delay_samples, orig_del);
        d2.delay_samples = 27.0;
        d2.set_sample_rate_hz(96_000);
        assert_eq!(d2.delay_samples, 54.0);
        d2.set_sample_rate_hz(48_000);
        assert_eq!(d2.delay_samples.to_bits(), 27.0_f64.to_bits());
        d2.delay_samples = f64::NAN;
        d2.set_sample_rate_hz(44_100);
        assert_eq!(d2.delay_samples, 0.0);
        let (mut extra_out, mut main_out) = ([1.0_f32; 4], [0.0_f32; 16]);
        let ins = [&dry[..16], &[][..]];
        let mut outs = [&mut main_out[..], &mut extra_out[..]];
        Chain01SymmetricOd::new().process_block(&ins, &mut outs, &ctrls, &[]);
        assert!(main_out.iter().any(|&v| v.abs() > 1.0e-3) && extra_out == [0.0; 4]);
    }
}
