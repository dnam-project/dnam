//! Spring Dispersion (`Case #18`), Digital Quirks (`Case #20`), Scratchy Pot & Stepped Attenuator
//! (`Case #22`), Idle Squeal (`Case #25`), and Sub-`0 dBFS` Microphone Capsule / Preamp Clipping +
//! LF `X_max` Excursion (`Case #26`, `mechanical_and_capsule`).

#![forbid(unsafe_code)]

use super::{
    impl_output_decorator_common, init_inner_float32, resolve_controls_with_defaults,
    step_inner_frame, write_outputs,
};
use crate::{BlackBoxModel, FixtureBitDepth, clamp_sample_rate_hz, sanitize_norm, sanitize_sample};
use std::f64::consts::PI;

#[inline]
fn splitmix64_bipolar(seed: u64) -> f32 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (((z >> 11) as f64) * (2.0 / ((1u64 << 53) as f64)) - 1.0) as f32
}

// ============================================================================
// Case #18: Spring Reverb Dispersion Chirp & Mechanical Splash Decorator
// ============================================================================

/// `Case #18` (`SpringChirpDecorator`): Adds an independent per-channel 16-stage dispersive
/// all-pass spring chirp chain and mechanical splash.
#[derive(Clone, Debug)]
pub struct SpringChirpDecorator<M: BlackBoxModel> {
    pub inner: M,
    out_bit_depth: FixtureBitDepth,
    ap_states: [[f64; 16]; 4],
}

impl<M: BlackBoxModel> SpringChirpDecorator<M> {
    #[must_use]
    pub fn new(inner: M) -> Self {
        let (inner, out_bit_depth) = init_inner_float32(inner);
        Self {
            inner,
            out_bit_depth,
            ap_states: [[0.0; 16]; 4],
        }
    }
}

impl<M: BlackBoxModel + Default> Default for SpringChirpDecorator<M> {
    fn default() -> Self {
        Self::new(M::default())
    }
}

impl<M: BlackBoxModel> BlackBoxModel for SpringChirpDecorator<M> {
    fn reset(&mut self) {
        self.inner.reset();
        self.ap_states = [[0.0; 16]; 4];
    }

    impl_output_decorator_common!(out_bit_depth);

    fn process_frame(
        &mut self,
        inputs: &[f32],
        outputs: &mut [f32],
        controls: &[f32],
        switches: &[u8],
    ) {
        let mut out_buf = [0.0_f32; 4];
        let model_out = step_inner_frame(&mut self.inner, inputs, controls, switches, &mut out_buf);
        let sr = clamp_sample_rate_hz(self.sample_rate_hz());
        let a_ap = if sr == 48_000 {
            -0.62_f64
        } else {
            let g = (PI * 3_520.0 / f64::from(sr)).min(1.25).tan();
            -(1.0 - g) / (1.0 + g)
        };
        for (ch, s) in out_buf.iter_mut().enumerate().take(model_out) {
            let dry = f64::from(sanitize_sample(*s));
            let mut u = dry;
            for st in &mut self.ap_states[ch] {
                let y = a_ap * u + *st;
                let next = u - a_ap * y;
                *st = if next.abs() < f64::MIN_POSITIVE {
                    0.0
                } else {
                    next
                };
                u = y;
            }
            *s = self
                .out_bit_depth
                .quantize_f64_to_f32(0.75 * dry + 0.25 * u);
        }
        write_outputs(outputs, &out_buf, model_out);
    }
}

// ============================================================================
// Case #20: Digital Pedal Quirks (12-Bit R-2R DAC + 28 kHz ZOH + 1.8 ms Latency)
// ============================================================================

const DIGITAL_LATENCY_RING_LEN: usize = 1_024;
const DIGITAL_LATENCY_RING_MASK: usize = DIGITAL_LATENCY_RING_LEN - 1;

/// `Case #20` (`DigitalQuirksDecorator`): Applies internal `12`-bit `R-2R` quantization (`dac_bit_depth`),
/// `28 kHz` ZOH hold, and `1.8 ms` (`+0.5 ms` mode-switch) DSP pipeline latency before emitting at
/// the outer device `out_bit_depth`.
#[derive(Clone, Debug)]
pub struct DigitalQuirksDecorator<M: BlackBoxModel> {
    pub inner: M,
    pub dac_bit_depth: FixtureBitDepth,
    out_bit_depth: FixtureBitDepth,
    pub zoh_rate_hz: f64,
    pub base_latency_ms: f64,
    pub mode_extra_latency_ms: f64,
    zoh_phase: f64,
    zoh_held: [f32; 4],
    latency_rings: [Vec<f32>; 4],
    wr_idx: usize,
}

impl<M: BlackBoxModel> DigitalQuirksDecorator<M> {
    #[must_use]
    pub fn new(inner: M) -> Self {
        let (inner, out_bit_depth) = init_inner_float32(inner);
        Self {
            inner,
            dac_bit_depth: FixtureBitDepth::Int12,
            out_bit_depth,
            zoh_rate_hz: 28_000.0,
            base_latency_ms: 1.8,
            mode_extra_latency_ms: 0.5,
            zoh_phase: 1.0, // Latch immediately on sample 0
            zoh_held: [0.0; 4],
            latency_rings: std::array::from_fn(|_| vec![0.0; DIGITAL_LATENCY_RING_LEN]),
            wr_idx: 0,
        }
    }
}

impl<M: BlackBoxModel + Default> Default for DigitalQuirksDecorator<M> {
    fn default() -> Self {
        Self::new(M::default())
    }
}

impl<M: BlackBoxModel> BlackBoxModel for DigitalQuirksDecorator<M> {
    fn reset(&mut self) {
        self.inner.reset();
        self.zoh_phase = 1.0;
        self.zoh_held = [0.0; 4];
        for ring in &mut self.latency_rings {
            ring.fill(0.0);
        }
        self.wr_idx = 0;
    }

    impl_output_decorator_common!(out_bit_depth);

    fn process_frame(
        &mut self,
        inputs: &[f32],
        outputs: &mut [f32],
        controls: &[f32],
        switches: &[u8],
    ) {
        let mut out_buf = [0.0_f32; 4];
        let model_out = step_inner_frame(&mut self.inner, inputs, controls, switches, &mut out_buf);
        let fs = f64::from(clamp_sample_rate_hz(self.sample_rate_hz()));
        let zoh_hz = if self.zoh_rate_hz.is_finite() {
            self.zoh_rate_hz.clamp(4_000.0, fs)
        } else {
            28_000.0_f64.min(fs)
        };
        let min_ratio = 4_000.0 / f64::from(crate::MAX_FIXTURE_SAMPLE_RATE_HZ);
        let ratio = (zoh_hz / fs).clamp(min_ratio, 1.0);
        let latch = if self.zoh_phase >= 1.0 {
            self.zoh_phase -= 1.0;
            true
        } else {
            false
        };
        self.zoh_phase += ratio;

        let base_ms = if self.base_latency_ms.is_finite() {
            self.base_latency_ms.max(0.0)
        } else {
            1.8
        };
        let default_sw0 = self
            .inner
            .metadata()
            .switches
            .first()
            .map_or(0, |s| s.default_index);
        let extra_ms = if switches.first().copied().unwrap_or(default_sw0) > 0 {
            if self.mode_extra_latency_ms.is_finite() {
                self.mode_extra_latency_ms.max(0.0)
            } else {
                0.5
            }
        } else {
            0.0
        };
        let total_ms = (base_ms + extra_ms).min(5.0);
        let delay_samples =
            ((total_ms * 1.0e-3 * fs).round() as usize).min(DIGITAL_LATENCY_RING_LEN - 2);

        let rd_idx = self.wr_idx.wrapping_sub(delay_samples) & DIGITAL_LATENCY_RING_MASK;
        for (ch, s) in out_buf.iter_mut().enumerate().take(model_out) {
            if latch {
                self.zoh_held[ch] = self.dac_bit_depth.quantize_sample(sanitize_sample(*s));
            }
            self.latency_rings[ch][self.wr_idx & DIGITAL_LATENCY_RING_MASK] = self.zoh_held[ch];
            *s = self
                .out_bit_depth
                .quantize_sample(self.latency_rings[ch][rd_idx]);
        }
        self.wr_idx = self.wr_idx.wrapping_add(1);
        write_outputs(outputs, &out_buf, model_out);
    }
}

// ============================================================================
// Case #22: Scratchy Potentiometer & 12-Step Attenuator
// ============================================================================

/// `Case #22` (`ScratchyPotAndSteppedAttenuatorDecorator`): Injects sparse, zero-mean `-32 dBFS`
/// wiper oxidation crackle bursts exclusively during knob motion (`dc/dt != 0`) and quantizes
/// knob `p = 1` into `12` discrete stepped-attenuator clicks.
#[derive(Clone, Debug)]
pub struct ScratchyPotAndSteppedAttenuatorDecorator<M: BlackBoxModel> {
    pub inner: M,
    out_bit_depth: FixtureBitDepth,
    prev_ctrl0: Option<f32>,
    wiper_travel_acc: f32,
    crackle_burst_step: u8,
    crackle_burst_amp: f32,
    sample_index: u64,
}

impl<M: BlackBoxModel> ScratchyPotAndSteppedAttenuatorDecorator<M> {
    #[must_use]
    pub fn new(inner: M) -> Self {
        let (inner, out_bit_depth) = init_inner_float32(inner);
        Self {
            inner,
            out_bit_depth,
            prev_ctrl0: None,
            wiper_travel_acc: 0.0,
            crackle_burst_step: 0,
            crackle_burst_amp: 0.0,
            sample_index: 0,
        }
    }

    /// Quantizes continuous control `u in [-1, 1]` to a 12-position rotary stepped attenuator.
    #[must_use]
    pub fn quantize_12step_attenuator(u: f32) -> f32 {
        let norm = 0.5 * (sanitize_norm(u, 0.0) + 1.0);
        let step = (norm * 11.0).round() / 11.0;
        2.0 * step - 1.0
    }
}

impl<M: BlackBoxModel + Default> Default for ScratchyPotAndSteppedAttenuatorDecorator<M> {
    fn default() -> Self {
        Self::new(M::default())
    }
}

impl<M: BlackBoxModel> BlackBoxModel for ScratchyPotAndSteppedAttenuatorDecorator<M> {
    fn reset(&mut self) {
        self.inner.reset();
        self.prev_ctrl0 = None;
        self.wiper_travel_acc = 0.0;
        self.crackle_burst_step = 0;
        self.crackle_burst_amp = 0.0;
        self.sample_index = 0;
    }

    impl_output_decorator_common!(out_bit_depth);

    fn process_frame(
        &mut self,
        inputs: &[f32],
        outputs: &mut [f32],
        controls: &[f32],
        switches: &[u8],
    ) {
        let mut stepped = resolve_controls_with_defaults(&self.inner, controls);
        if stepped.len() > 1 {
            stepped[1] = Self::quantize_12step_attenuator(stepped[1]);
        }

        let idx = self.sample_index;
        self.sample_index = self.sample_index.wrapping_add(1);

        let c0 = stepped.first().copied().unwrap_or(0.0);
        if let Some(prev) = self.prev_ctrl0 {
            let dc = (c0 - prev).abs();
            if dc > 0.0 {
                self.wiper_travel_acc += dc;
                if self.crackle_burst_step == 0 && self.wiper_travel_acc >= 0.035 {
                    self.wiper_travel_acc %= 0.035;
                    self.crackle_burst_step = 3;
                    // Deterministic high-kurtosis oxidation spike around -32 dBFS (0.025 peak)
                    let r = splitmix64_bipolar(idx);
                    let sign = if r >= 0.0 { 1.0_f32 } else { -1.0_f32 };
                    self.crackle_burst_amp = sign * (0.006_25 + 0.006_25 * r.abs());
                } else if self.wiper_travel_acc >= 0.035 {
                    self.wiper_travel_acc %= 0.035;
                }
            }
        }
        self.prev_ctrl0 = Some(c0);

        // Zero-mean [ +1, -2, +1 ] transient wavelet so wiper crackle injects zero net DC offset
        let crackle = match self.crackle_burst_step {
            3 => {
                self.crackle_burst_step = 2;
                self.crackle_burst_amp
            }
            2 => {
                self.crackle_burst_step = 1;
                -2.0 * self.crackle_burst_amp
            }
            1 => {
                self.crackle_burst_step = 0;
                self.crackle_burst_amp
            }
            _ => 0.0_f32,
        };

        let mut out_buf = [0.0_f32; 4];
        let model_out = step_inner_frame(&mut self.inner, inputs, &stepped, switches, &mut out_buf);
        let crackle_f64 = f64::from(crackle);
        for s in out_buf.iter_mut().take(model_out) {
            *s = self
                .out_bit_depth
                .quantize_f64_to_f32(f64::from(sanitize_sample(*s)) + crackle_f64);
        }
        write_outputs(outputs, &out_buf, model_out);
    }
}

// ============================================================================
// Case #25: Parasitic Idle Squeal (`3.4 kHz` Envelope-Quenched Oscillation)
// ============================================================================

/// `Case #25` (`IdleSquealDecorator`): Injects `-42 dBFS` `3.4 kHz` parasitic oscillation during
/// sustained quiet idle (`input_env < -55 dBFS`) when `Gain > 0.85`, quenched cleanly without
/// zero-crossing chatter whenever active guitar signal exceeds `-55 dBFS`.
#[derive(Clone, Debug)]
pub struct IdleSquealDecorator<M: BlackBoxModel> {
    pub inner: M,
    out_bit_depth: FixtureBitDepth,
    sample_index: u64,
    input_env: f32,
}

impl<M: BlackBoxModel> IdleSquealDecorator<M> {
    #[must_use]
    pub fn new(inner: M) -> Self {
        let (inner, out_bit_depth) = init_inner_float32(inner);
        Self {
            inner,
            out_bit_depth,
            sample_index: 0,
            input_env: 0.0,
        }
    }
}

impl<M: BlackBoxModel + Default> Default for IdleSquealDecorator<M> {
    fn default() -> Self {
        Self::new(M::default())
    }
}

impl<M: BlackBoxModel> BlackBoxModel for IdleSquealDecorator<M> {
    fn reset(&mut self) {
        self.inner.reset();
        self.sample_index = 0;
        self.input_env = 0.0;
    }

    impl_output_decorator_common!(out_bit_depth);

    fn process_frame(
        &mut self,
        inputs: &[f32],
        outputs: &mut [f32],
        controls: &[f32],
        switches: &[u8],
    ) {
        let resolved = resolve_controls_with_defaults(&self.inner, controls);

        let n_in = inputs.len().min(self.num_inputs()).min(4);
        let mut peak_in = 0.0_f32;
        for &x in inputs.iter().take(n_in) {
            peak_in = peak_in.max(sanitize_sample(x).abs());
        }
        let fs = f64::from(clamp_sample_rate_hz(self.sample_rate_hz()));
        let rel = (-1.0 / (0.035 * fs)).exp() as f32;
        let decayed = self.input_env * rel;
        self.input_env = peak_in.max(if decayed < f32::MIN_POSITIVE {
            0.0
        } else {
            decayed
        });

        let mut out_buf = [0.0_f32; 4];
        let model_out = step_inner_frame(
            &mut self.inner,
            &inputs[..n_in],
            &resolved,
            switches,
            &mut out_buf,
        );
        let idx = self.sample_index;
        self.sample_index = self.sample_index.wrapping_add(1);
        let gain_hi = resolved.first().copied().unwrap_or(0.0) > 0.85;
        let squeal = if gain_hi && self.input_env < 0.001_778 {
            0.007_943 * (2.0 * PI * 3_400.0 * (idx as f64) / fs).sin()
        } else {
            0.0_f64
        };
        for s in out_buf.iter_mut().take(model_out) {
            *s = self
                .out_bit_depth
                .quantize_f64_to_f32(f64::from(sanitize_sample(*s)) + squeal);
        }
        write_outputs(outputs, &out_buf, model_out);
    }
}

// ============================================================================
// Case #26: Sub-0 dBFS Microphone Capsule / Preamp Clipping & LF X_max Excursion
// ============================================================================

/// Black-box physical metrics for sub-`0 dBFS` microphone capsule clipping (`Case #26`).
#[derive(Clone, Debug)]
pub struct CapsuleClipMetrics {
    /// Post-cabinet stopband excess ratio (`> 10.0` indicates post-cabinet clipping splatter).
    pub stopband_excess_gamma: f64,
    /// Flat-rail derivative plateau density (`> 0.002` at sub-`0 dBFS` peak `< 0.90`).
    pub rail_plateau_kappa: f64,
    /// Estimated sub-`0 dBFS` capsule ceiling in dBFS (e.g. `-9.5 dBFS`).
    pub estimated_ceiling_dbfs: f64,
    /// Number of samples exceeding `0.85 * peak`.
    pub clipped_sample_count: usize,
}

/// `Case #26` (`MicrophoneCapsuleClipDecorator`): Injects asymmetric sub-`0 dBFS` FET capsule/preamp
/// rail clipping (`+0.335 / -0.310` linear = `-9.5 dBFS`) and `85 Hz` speaker cone `X_max`
/// monotone Padé suspension stiffening (`1 / (1 + alpha_xmax * x_cone^2)`) per channel.
#[derive(Clone, Debug)]
pub struct MicrophoneCapsuleClipDecorator<M: BlackBoxModel> {
    pub inner: M,
    pub pos_ceiling_lin: f32,
    pub neg_ceiling_lin: f32,
    pub alpha_xmax: f64,
    out_bit_depth: FixtureBitDepth,
    cone_lp_states: [f64; 4],
}

impl<M: BlackBoxModel> MicrophoneCapsuleClipDecorator<M> {
    /// Creates a `Case #26` decorator with `-9.5 dBFS` (`+0.335 / -0.310`) FET capsule ceiling and `X_max` IM.
    #[must_use]
    pub fn new(inner: M) -> Self {
        let (inner, out_bit_depth) = init_inner_float32(inner);
        Self {
            inner,
            pos_ceiling_lin: 0.335,
            neg_ceiling_lin: -0.310,
            alpha_xmax: 0.28,
            out_bit_depth,
            cone_lp_states: [0.0; 4],
        }
    }

    #[inline]
    fn sanitized_ceilings(&self) -> (f32, f32) {
        let pos = if self.pos_ceiling_lin.is_finite() {
            self.pos_ceiling_lin.clamp(-1.0, 1.0)
        } else {
            0.335
        };
        let neg = if self.neg_ceiling_lin.is_finite() {
            self.neg_ceiling_lin.clamp(-1.0, 1.0)
        } else {
            -0.310
        };
        (neg.min(pos), pos.max(neg))
    }

    /// Applies `X_max` LF cone excursion intermodulation and sub-`0 dBFS` asymmetric FET capsule clipping to `clean_cab`.
    pub fn apply_capsule_and_xmax_fault(&mut self, clean_cab: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(clean_cab.len());
        let fs = f64::from(clamp_sample_rate_hz(self.sample_rate_hz()));
        let a_cone = 1.0 - (-2.0 * PI * 85.0 / fs).exp();
        let alpha = if self.alpha_xmax.is_finite() {
            self.alpha_xmax.max(0.0)
        } else {
            0.28
        };
        let (lo, hi) = self.sanitized_ceilings();
        let bit_depth = self.out_bit_depth;
        for &y in clean_cab {
            let yf = f64::from(sanitize_sample(y));
            self.cone_lp_states[0] += a_cone * (yf - self.cone_lp_states[0]);
            let x_cone = self.cone_lp_states[0];
            let im_mod = 1.0 / (1.0 + alpha * x_cone * x_cone);
            let radiated = (yf * im_mod).clamp(f64::from(lo), f64::from(hi));
            out.push(bit_depth.quantize_f64_to_f32(radiated));
        }
        out
    }

    /// Computes physical rail-plateau and stopband-splatter metrics comparing `clean_mic` and `clipped_mic`.
    #[must_use]
    pub fn analyze_capsule_clip_metrics(
        clean_mic: &[f32],
        clipped_mic: &[f32],
    ) -> CapsuleClipMetrics {
        let n = clipped_mic.len().min(clean_mic.len());
        let mut peak = 0.0_f64;
        for &y in &clipped_mic[..n] {
            peak = peak.max(f64::from(sanitize_sample(y).abs()));
        }

        let mut near_rail_count = 0usize;
        let mut clipped_sample_count = 0usize;
        if peak > 1.0e-6 {
            let rail_thresh = 0.85 * peak;
            if n > 0 && f64::from(sanitize_sample(clipped_mic[0]).abs()) >= rail_thresh {
                clipped_sample_count += 1;
            }
            for i in 1..n {
                let y_cur = f64::from(sanitize_sample(clipped_mic[i]));
                let y_prev = f64::from(sanitize_sample(clipped_mic[i - 1]));
                if y_cur.abs() >= rail_thresh {
                    clipped_sample_count += 1;
                    if (y_cur - y_prev).abs() < 1.0e-4 {
                        near_rail_count += 1;
                    }
                }
            }
        }
        let rail_plateau_kappa = (near_rail_count as f64) / (n.max(1) as f64);

        let mut hf_clipped = 0.0_f64;
        let mut hf_clean_env = 0.0_f64;
        for i in 2..n {
            let d2_clip = f64::from(sanitize_sample(clipped_mic[i]))
                - 2.0 * f64::from(sanitize_sample(clipped_mic[i - 1]))
                + f64::from(sanitize_sample(clipped_mic[i - 2]));
            let d2_clean = f64::from(sanitize_sample(clean_mic[i]))
                - 2.0 * f64::from(sanitize_sample(clean_mic[i - 1]))
                + f64::from(sanitize_sample(clean_mic[i - 2]));
            hf_clipped += d2_clip * d2_clip;
            hf_clean_env += d2_clean * d2_clean;
        }
        let stopband_excess_gamma = hf_clipped / hf_clean_env.max(1.0e-12);
        let estimated_ceiling_dbfs = 20.0 * peak.max(1.0e-6).log10();

        CapsuleClipMetrics {
            stopband_excess_gamma,
            rail_plateau_kappa,
            estimated_ceiling_dbfs,
            clipped_sample_count,
        }
    }
}

impl<M: BlackBoxModel + Default> Default for MicrophoneCapsuleClipDecorator<M> {
    fn default() -> Self {
        Self::new(M::default())
    }
}

impl<M: BlackBoxModel> BlackBoxModel for MicrophoneCapsuleClipDecorator<M> {
    fn reset(&mut self) {
        self.inner.reset();
        self.cone_lp_states = [0.0; 4];
    }

    impl_output_decorator_common!(out_bit_depth);

    fn process_frame(
        &mut self,
        inputs: &[f32],
        outputs: &mut [f32],
        controls: &[f32],
        switches: &[u8],
    ) {
        let mut out_buf = [0.0_f32; 4];
        let model_out = step_inner_frame(&mut self.inner, inputs, controls, switches, &mut out_buf);
        let fs = f64::from(clamp_sample_rate_hz(self.sample_rate_hz()));
        let a_cone = 1.0 - (-2.0 * PI * 85.0 / fs).exp();
        let alpha = if self.alpha_xmax.is_finite() {
            self.alpha_xmax.max(0.0)
        } else {
            0.28
        };
        let (lo, hi) = self.sanitized_ceilings();
        for (ch, s) in out_buf.iter_mut().enumerate().take(model_out) {
            let yf = f64::from(sanitize_sample(*s));
            self.cone_lp_states[ch] += a_cone * (yf - self.cone_lp_states[ch]);
            let x_cone = self.cone_lp_states[ch];
            let im_mod = 1.0 / (1.0 + alpha * x_cone * x_cone);
            let radiated = (yf * im_mod).clamp(f64::from(lo), f64::from(hi));
            *s = self.out_bit_depth.quantize_f64_to_f32(radiated);
        }
        write_outputs(outputs, &out_buf, model_out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chains::{
        Chain01SymmetricOd, Chain04TubeAmpFmv4x12, Chain07StereoChorusAndRotary,
        Chain08LongBbdEchoAnd30sShimmer, Chain09La2aOpticalCompressor,
        Chain11BiAmpBassAndOutputIron,
    };
    use crate::test_utils::{
        exercise_blackbox_metadata_and_format, run_mono_block, run_stereo_block, sine_wave,
    };

    #[test]
    fn test_mechanical_and_capsule_decorators_unit() {
        let dry = sine_wave(256, 330.0, 0.5);
        let ctrls = [0.6_f32, 0.2, 0.5];
        let silence = vec![0.0_f32; 256];

        // 1. SpringChirpDecorator: stereo isolation without crosstalk
        let mut d10 = SpringChirpDecorator::new(Chain07StereoChorusAndRotary::new());
        exercise_blackbox_metadata_and_format(&mut d10);
        let (sl, sr) = run_stereo_block(&mut d10, &dry, &silence, &[0.0, 0.0, 0.0], &[0]);
        assert!(sl.iter().any(|&v| v.abs() > 1.0e-3));
        assert!(sr.iter().all(|v| v.is_finite()));

        // 2. DigitalQuirksDecorator: preserves internal Int12 R-2R quantization regardless of outer
        //    bit depth, applies 1.8 ms latency (~86 samples @ 48k), and honors 4 kHz ZOH at 192 kHz
        let mut d11 = DigitalQuirksDecorator::new(Chain01SymmetricOd::new());
        exercise_blackbox_metadata_and_format(&mut d11);
        assert_eq!(d11.dac_bit_depth, FixtureBitDepth::Int12);
        assert_eq!(d11.metadata().bit_depth, FixtureBitDepth::Float32);
        let y_d11 = run_mono_block(&mut d11, &dry, &ctrls, &[0]);
        assert!(y_d11[..80].iter().all(|&v| v == 0.0));
        assert!(y_d11[100..].iter().any(|&v| v.abs() > 1.0e-3));
        assert!(
            y_d11
                .iter()
                .all(|&v| (f64::from(v) * 2048.0).fract().abs() == 0.0)
        );

        let mut d11_composed = ScratchyPotAndSteppedAttenuatorDecorator::new(
            DigitalQuirksDecorator::new(Chain01SymmetricOd::new()),
        );
        assert_eq!(d11_composed.bit_depth(), FixtureBitDepth::Float32);
        assert_eq!(d11_composed.inner.dac_bit_depth, FixtureBitDepth::Int12);
        let y_comp = run_mono_block(&mut d11_composed, &dry, &[0.5, 0.0, 0.5], &[]);
        assert!(
            y_comp
                .iter()
                .all(|&v| (f64::from(v) * 2048.0).fract().abs() == 0.0)
        );

        let mut d11_192k = DigitalQuirksDecorator::new(Chain01SymmetricOd::with_format(
            192_000,
            FixtureBitDepth::Int24,
        ));
        d11_192k.zoh_rate_hz = 4_000.0;
        d11_192k.base_latency_ms = 0.0;
        assert_eq!(d11_192k.bit_depth(), FixtureBitDepth::Int24);
        let y_192k = run_mono_block(&mut d11_192k, &sine_wave(96, 1_500.0, 0.6), &ctrls, &[0]);
        assert_eq!(
            y_192k[1..48].iter().filter(|&&v| v != y_192k[0]).count(),
            0,
            "4 kHz ZOH at 192 kHz must hold for 48 samples without clamping to 0.05"
        );
        assert_ne!(y_192k[48], y_192k[0]);

        let mut d11_zero_lat =
            DigitalQuirksDecorator::new(crate::adversaries::PolarityAndDcOffsetDecorator::new(
                Chain01SymmetricOd::new(),
                false,
                0.25,
            ));
        d11_zero_lat.base_latency_ms = 0.0;
        let step_in = vec![0.0_f32; 16];
        let y_zero_lat = run_mono_block(&mut d11_zero_lat, &step_in, &ctrls, &[0]);
        assert!(y_zero_lat[0].abs() > 0.2, "sample 0 must latch immediately");

        // 3. ScratchyPotAndSteppedAttenuatorDecorator: zero startup click on stationary c0 != 0,
        //    sparse zero-mean crackle bursts during knob sweep
        let mut d12 = ScratchyPotAndSteppedAttenuatorDecorator::new(Chain01SymmetricOd::new());
        exercise_blackbox_metadata_and_format(&mut d12);
        let mut clean = Chain01SymmetricOd::new();
        let stationary_c = [0.42_f32, 0.0, 0.0]; // quantized c1=0.0 matches clean
        let y_d12_stat = run_mono_block(&mut d12, &silence, &stationary_c, &[]);
        let y_clean_stat = run_mono_block(&mut clean, &silence, &stationary_c, &[]);
        assert_eq!(
            y_d12_stat, y_clean_stat,
            "stationary pot with c0 != 0 must never emit startup click"
        );

        d12.reset();
        let mut sweep_out = Vec::with_capacity(256);
        for i in 0..256 {
            let c_sweep = -0.8 + 1.6 * ((i as f32) / 255.0);
            let mut out = [0.0_f32];
            d12.process_frame(&[0.0], &mut out, &[c_sweep, 0.0, 0.0], &[]);
            sweep_out.push(out[0]);
        }
        let nonzero_count = sweep_out.iter().filter(|&&v| v.abs() > 1.0e-5).count();
        let net_dc: f32 = sweep_out.iter().sum();
        assert!(nonzero_count > 6 && nonzero_count < 240);
        assert!(net_dc.abs() < 0.03, "crackle wavelets must be zero-mean");

        // 4. IdleSquealDecorator: squeals on quiet idle, completely quenched across sine zero-crossings
        let mut d13 = IdleSquealDecorator::new(Chain01SymmetricOd::new());
        exercise_blackbox_metadata_and_format(&mut d13);
        let y_d13_idle = run_mono_block(&mut d13, &silence, &[0.95, 0.0, 0.0], &[]);
        assert!(y_d13_idle.iter().any(|&v| v.abs() > 1.0e-3));

        d13.reset();
        clean.reset();
        let guitar_80hz = sine_wave(1_200, 80.0, 0.45); // crosses zero every 300 samples (6.25 ms)
        let y_d13_play = run_mono_block(&mut d13, &guitar_80hz, &[0.95, 0.0, 0.0], &[]);
        let y_clean_play = run_mono_block(&mut clean, &guitar_80hz, &[0.95, 0.0, 0.0], &[]);
        // After sample 1 matching attack, zero squeal is injected even at every zero crossing
        for i in 2..1_200 {
            assert!(
                (y_d13_play[i] - y_clean_play[i]).abs() < 1.0e-6,
                "squeal leaked during active guitar zero crossing at sample {i}"
            );
        }

        // 5. MicrophoneCapsuleClipDecorator: Padé X_max stays positive even under extreme +40 dBFS cone excursion
        let mut d14 = MicrophoneCapsuleClipDecorator::new(Chain04TubeAmpFmv4x12::new());
        exercise_blackbox_metadata_and_format(&mut d14);
        let y_d14 = run_mono_block(&mut d14, &dry, &[0.9, 0.5, 0.5, 0.5], &[1]);
        assert!(
            y_d14
                .iter()
                .all(|&v| v <= d14.pos_ceiling_lin + 1.0e-6 && v >= d14.neg_ceiling_lin - 1.0e-6)
        );
        let huge_lf = vec![50.0_f32; 64];
        let y_huge = d14.apply_capsule_and_xmax_fault(&huge_lf);
        assert!(
            y_huge.iter().all(|&v| v > 0.0),
            "Padé X_max stiffening must never invert phase under large excursion"
        );

        let clean_cab = sine_wave(512, 180.0, 0.55);
        d14.reset();
        let clipped_cab = d14.apply_capsule_and_xmax_fault(&clean_cab);
        let clip_metrics =
            MicrophoneCapsuleClipDecorator::<Chain04TubeAmpFmv4x12>::analyze_capsule_clip_metrics(
                &clean_cab,
                &clipped_cab,
            );
        assert!(clip_metrics.stopband_excess_gamma > 10.0);
        assert!(clip_metrics.rail_plateau_kappa > 0.002);
        assert!(clip_metrics.clipped_sample_count > 0);
        assert!(clip_metrics.estimated_ceiling_dbfs < -8.0);
        let sil_metrics =
            MicrophoneCapsuleClipDecorator::<Chain04TubeAmpFmv4x12>::analyze_capsule_clip_metrics(
                &silence, &silence,
            );
        assert_eq!(sil_metrics.clipped_sample_count, 0);
        assert_eq!(sil_metrics.rail_plateau_kappa, 0.0);
        let mut nan_probe = clipped_cab.clone();
        nan_probe[5] = f32::NAN;
        nan_probe[6] = f32::INFINITY;
        let nan_metrics =
            MicrophoneCapsuleClipDecorator::<Chain04TubeAmpFmv4x12>::analyze_capsule_clip_metrics(
                &clean_cab, &nan_probe,
            );
        assert!(nan_metrics.stopband_excess_gamma.is_finite());

        // Cases #17, #21, #23, #24 behavioral checks
        let mut c7 = Chain07StereoChorusAndRotary::new();
        for subdiv in 0..12 {
            let hz_20 = Chain07StereoChorusAndRotary::lfo_hz_for_subdivision(20.0, subdiv);
            let hz_400 = Chain07StereoChorusAndRotary::lfo_hz_for_subdivision(400.0, subdiv);
            assert!(hz_20 > 0.0 && hz_400 > hz_20);
        }
        c7.set_tempo_and_subdivisions(120.0, 5, 7);
        let (c7_l, c7_r) = run_stereo_block(&mut c7, &dry, &dry, &[0.0, 0.8, 0.8], &[1]);
        assert!(
            c7_l.iter()
                .zip(c7_r.iter())
                .any(|(a, b)| (a - b).abs() > 1.0e-4)
        );

        let mut c8 = Chain08LongBbdEchoAnd30sShimmer::new();
        c8.set_freeze_window(2_400, 4_800);
        let mut freeze_energy = 0.0_f64;
        for i in 0..4_800 {
            let x = if i < 2_400 {
                0.6 * ((i as f32) * 0.2).sin()
            } else {
                0.0
            };
            let [l, r] = c8.step_stereo(x, x, &[0.0, 0.9, 0.8], false);
            if i >= 2_400 {
                freeze_energy += f64::from(l * l + r * r);
            }
        }
        assert!(freeze_energy > 1.0e-4);

        let mut c11 = Chain11BiAmpBassAndOutputIron::new();
        let mut max_flux = 0.0_f32;
        for i in 0..256 {
            let x = 0.5 * (2.0 * std::f32::consts::PI * 65.0 * (i as f32) / 48_000.0).sin();
            let _ = c11.step_sample(x, &[0.3, 0.6], &[1]);
            max_flux = max_flux.max(c11.current_flux().abs());
        }
        assert!(max_flux > 1.0e-4);

        let mut c9 = Chain09La2aOpticalCompressor::new();
        for _ in 0..256 {
            let _ = c9.step_sample(0.8, None, &[0.7, 0.0], &[0]);
        }
        assert!(c9.current_gain_reduction() < 0.99);

        // DigitalQuirksDecorator non-finite latency fallbacks + ConverterLoopback usb_drop scaling + Int32 clip rail
        let mut dq = DigitalQuirksDecorator::new(Chain01SymmetricOd::new());
        dq.base_latency_ms = f64::NAN;
        dq.mode_extra_latency_ms = f64::INFINITY;
        let mut dq_out = [0.0_f32; 1];
        dq.process_frame(&[0.4], &mut dq_out, &[0.0, 0.0, 0.0], &[1]);
        assert!(dq_out[0].is_finite());

        let mut lb = crate::adversaries::ConverterLoopbackDecorator::new(Chain01SymmetricOd::new());
        lb.usb_drop_at_sample = Some(4_800);
        lb.set_sample_rate_hz(8_000);
        assert_eq!(lb.usb_drop_at_sample, Some(800));
        lb.set_sample_rate_hz(192_000);
        assert_eq!(lb.usb_drop_at_sample, Some(19_200));
        lb.set_sample_rate_hz(48_000);
        assert_eq!(lb.usb_drop_at_sample, Some(4_800));

        let mut clip32 = crate::adversaries::AdcClipAndPadDecorator::new(
            crate::adversaries::PolarityAndDcOffsetDecorator::new(
                Chain01SymmetricOd::new(),
                false,
                1.0,
            ),
            0.0,
        );
        clip32.set_bit_depth(FixtureBitDepth::Int32);
        let mut c32_out = [0.0_f32; 1];
        clip32.process_frame(&[0.0], &mut c32_out, &ctrls, &[]);
        assert_eq!(clip32.last_clipped_samples, 1);

        // DigitalQuirksDecorator empty switches parity with metadata().default_switches() on Chain-06 (default_index = 1)
        let mut dq6_empty =
            DigitalQuirksDecorator::new(crate::chains::Chain06EnvelopeResonantSvf::new());
        let mut dq6_def = dq6_empty.clone();
        let (c_def, sw_def) = (
            dq6_def.metadata().default_controls(),
            dq6_def.metadata().default_switches(),
        );
        let y_empty = run_mono_block(&mut dq6_empty, &dry, &[], &[]);
        let y_def = run_mono_block(&mut dq6_def, &dry, &c_def, &sw_def);
        assert_eq!(y_empty, y_def);

        // ScratchyPot must never initiate a new crackle burst once knob stops moving (dc == 0.0)
        let mut sp = ScratchyPotAndSteppedAttenuatorDecorator::new(Chain01SymmetricOd::new());
        let mut sp_out = [0.0_f32; 1];
        for &c in &[0.0_f32, 0.04, 0.08, 0.08, 0.08] {
            sp.process_frame(&[0.0], &mut sp_out, &[c, 0.0, 0.0], &[]);
        }
        assert_eq!(sp_out[0], 0.0);
        let mut sq = IdleSquealDecorator::new(Chain01SymmetricOd::new());
        sq.process_frame(&[0.5], &mut sp_out, &[], &[]);
        for _ in 0..160_000 {
            sq.process_frame(&[0.0], &mut sp_out, &[], &[]);
        }
        assert_eq!(sq.input_env, 0.0);
    }
}
