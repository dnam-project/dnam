//! Tube Sag & Thermal Hysteresis (`Case #5`), Germanium Temperature Drift (`Case #14`),
//! Mains Hum & Noise Floor (`Case #15`), and Hardware Noise Gate (`Case #16`) Decorators
//! (`thermal_and_noise`).

#![forbid(unsafe_code)]

use super::{
    init_inner_float32, metadata_with_bit_depth, resolve_controls_with_defaults, step_inner_frame,
    write_outputs,
};
use crate::{
    BlackBoxModel, FixtureBitDepth, FixtureDeviceMetadata, clamp_sample_rate_hz, sanitize_sample,
};
use std::f64::consts::PI;

#[inline]
fn splitmix64_bipolar(sample_idx: u64, channel: u64) -> f64 {
    let mut z = sample_idx
        .wrapping_add(channel.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 11) as f64) * (2.0 / ((1u64 << 53) as f64)) - 1.0
}

// ============================================================================
// Case #5: Tube Sag & Thermal Hysteresis (`tau_sag = 1.2s`, `tau_therm = 45.0s`)
// ============================================================================

/// `Case #5` (`TubeThermalHysteresisDecorator`): Applies `tau_sag = 1.2s` shared B+ supply sag and
/// `tau_therm = 45.0s` cathode thermal drift once per frame across all channels.
#[derive(Clone, Debug)]
pub struct TubeThermalHysteresisDecorator<M: BlackBoxModel> {
    pub inner: M,
    pub sag_state: f64,
    pub thermal_state: f64,
    out_bit_depth: FixtureBitDepth,
}

impl<M: BlackBoxModel> TubeThermalHysteresisDecorator<M> {
    #[must_use]
    pub fn new(inner: M) -> Self {
        let (inner, out_bit_depth) = init_inner_float32(inner);
        Self {
            inner,
            sag_state: 0.0,
            thermal_state: 0.0,
            out_bit_depth,
        }
    }
}

impl<M: BlackBoxModel + Default> Default for TubeThermalHysteresisDecorator<M> {
    fn default() -> Self {
        Self::new(M::default())
    }
}

impl<M: BlackBoxModel> BlackBoxModel for TubeThermalHysteresisDecorator<M> {
    fn reset(&mut self) {
        self.inner.reset();
        self.sag_state = 0.0;
        self.thermal_state = 0.0;
    }

    fn metadata(&self) -> FixtureDeviceMetadata {
        metadata_with_bit_depth(&self.inner, self.out_bit_depth)
    }

    fn set_sample_rate_hz(&mut self, sample_rate_hz: u32) {
        self.inner.set_sample_rate_hz(sample_rate_hz);
        self.reset();
    }

    fn set_bit_depth(&mut self, bit_depth: FixtureBitDepth) {
        self.out_bit_depth = bit_depth;
        self.inner.set_bit_depth(FixtureBitDepth::Float32);
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
        if model_out == 0 {
            write_outputs(outputs, &out_buf, 0);
            return;
        }
        let mut env_sum = 0.0_f64;
        for &s in out_buf.iter().take(model_out) {
            env_sum += f64::from(sanitize_sample(s).abs());
        }
        let e = env_sum / (model_out as f64);

        let fs = f64::from(clamp_sample_rate_hz(self.sample_rate_hz()));
        let a_sag = 1.0 - (-1.0 / (1.2 * fs)).exp();
        let a_therm = 1.0 - (-1.0 / (45.0 * fs)).exp();
        self.sag_state = if self.sag_state.is_finite() {
            self.sag_state.clamp(0.0, 1.0e4)
        } else {
            0.0
        };
        self.thermal_state = if self.thermal_state.is_finite() {
            self.thermal_state.clamp(0.0, 1.0e4)
        } else {
            0.0
        };
        self.sag_state += a_sag * (e - self.sag_state);
        self.thermal_state += a_therm * (e - self.thermal_state);
        let gain = 1.0 / (1.0 + 0.25 * self.sag_state + 0.15 * self.thermal_state);
        for s in out_buf.iter_mut().take(model_out) {
            *s = self
                .out_bit_depth
                .quantize_f64_to_f32(f64::from(sanitize_sample(*s)) * gain);
        }
        write_outputs(outputs, &out_buf, model_out);
    }
}

// ============================================================================
// Case #14: Germanium Temperature Drift (`22 C -> 31 C`)
// ============================================================================

/// `Case #14` (`GermaniumTempDriftDecorator`): Simulates ambient temperature drift (`22 C -> 31 C`) shifting bias & leakage.
#[derive(Clone, Debug)]
pub struct GermaniumTempDriftDecorator<M: BlackBoxModel> {
    pub inner: M,
    pub temp_celsius: f32,
    out_bit_depth: FixtureBitDepth,
}

impl<M: BlackBoxModel> GermaniumTempDriftDecorator<M> {
    #[must_use]
    pub fn new(inner: M, temp_celsius: f32) -> Self {
        let (inner, out_bit_depth) = init_inner_float32(inner);
        Self {
            inner,
            temp_celsius,
            out_bit_depth,
        }
    }
}

impl<M: BlackBoxModel + Default> Default for GermaniumTempDriftDecorator<M> {
    fn default() -> Self {
        Self::new(M::default(), 25.0)
    }
}

impl<M: BlackBoxModel> BlackBoxModel for GermaniumTempDriftDecorator<M> {
    fn reset(&mut self) {
        self.inner.reset();
    }

    fn metadata(&self) -> FixtureDeviceMetadata {
        metadata_with_bit_depth(&self.inner, self.out_bit_depth)
    }

    fn set_sample_rate_hz(&mut self, sample_rate_hz: u32) {
        self.inner.set_sample_rate_hz(sample_rate_hz);
        self.reset();
    }

    fn set_bit_depth(&mut self, bit_depth: FixtureBitDepth) {
        self.out_bit_depth = bit_depth;
        self.inner.set_bit_depth(FixtureBitDepth::Float32);
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
        let mut adj = resolve_controls_with_defaults(&self.inner, controls);
        let temp = if self.temp_celsius.is_finite() {
            self.temp_celsius.clamp(-40.0, 85.0)
        } else {
            25.0
        };
        let delta_c = temp - 25.0;
        let bias_shift = delta_c * 0.015;
        if adj.len() > 1 {
            adj[1] = (adj[1] + bias_shift).clamp(-1.0, 1.0);
        } else if adj.len() == 1 {
            adj[0] = (adj[0] + bias_shift).clamp(-1.0, 1.0);
        }
        let mut out_buf = [0.0_f32; 4];
        let model_out = step_inner_frame(&mut self.inner, inputs, &adj, switches, &mut out_buf);
        let leakage_gain = 1.0 + 0.002 * f64::from(delta_c);
        for s in out_buf.iter_mut().take(model_out) {
            *s = self
                .out_bit_depth
                .quantize_f64_to_f32(f64::from(sanitize_sample(*s)) * leakage_gain);
        }
        write_outputs(outputs, &out_buf, model_out);
    }
}

// ============================================================================
// Case #15: Mains Hum & Uncorrelated Wideband Noise Floor
// ============================================================================

/// `Case #15` (`MainsHumAndNoiseFloorDecorator`): Injects `60/120/180 Hz` mains hum (`-58 dBFS`)
/// plus uncorrelated SplitMix64 wideband white noise floor (`-65 dBFS` per channel).
#[derive(Clone, Debug)]
pub struct MainsHumAndNoiseFloorDecorator<M: BlackBoxModel> {
    pub inner: M,
    pub mains_freq_hz: f64,
    pub hum_amp: f64,
    pub noise_amp: f64,
    out_bit_depth: FixtureBitDepth,
    sample_index: u64,
}

impl<M: BlackBoxModel> MainsHumAndNoiseFloorDecorator<M> {
    #[must_use]
    pub fn new(inner: M, mains_freq_hz: f64) -> Self {
        let (inner, out_bit_depth) = init_inner_float32(inner);
        Self {
            inner,
            mains_freq_hz,
            hum_amp: 0.001_258_9,   // -58 dBFS
            noise_amp: 0.000_562_3, // -65 dBFS
            out_bit_depth,
            sample_index: 0,
        }
    }
}

impl<M: BlackBoxModel + Default> Default for MainsHumAndNoiseFloorDecorator<M> {
    fn default() -> Self {
        Self::new(M::default(), 60.0)
    }
}

impl<M: BlackBoxModel> BlackBoxModel for MainsHumAndNoiseFloorDecorator<M> {
    fn reset(&mut self) {
        self.inner.reset();
        self.sample_index = 0;
    }

    fn metadata(&self) -> FixtureDeviceMetadata {
        metadata_with_bit_depth(&self.inner, self.out_bit_depth)
    }

    fn set_sample_rate_hz(&mut self, sample_rate_hz: u32) {
        self.inner.set_sample_rate_hz(sample_rate_hz);
        self.reset();
    }

    fn set_bit_depth(&mut self, bit_depth: FixtureBitDepth) {
        self.out_bit_depth = bit_depth;
        self.inner.set_bit_depth(FixtureBitDepth::Float32);
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
        let i = self.sample_index;
        self.sample_index = self.sample_index.wrapping_add(1);
        let fs = f64::from(clamp_sample_rate_hz(self.sample_rate_hz()));
        let t = (i as f64) / fs;
        let f0 = if self.mains_freq_hz.is_finite() {
            self.mains_freq_hz.clamp(40.0, 400.0)
        } else {
            60.0
        };
        let hum_a = if self.hum_amp.is_finite() {
            self.hum_amp.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let noise_a = if self.noise_amp.is_finite() {
            self.noise_amp.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let hum = hum_a
            * ((2.0 * PI * f0 * t).sin()
                + 0.5 * (4.0 * PI * f0 * t).sin()
                + 0.25 * (6.0 * PI * f0 * t).sin());
        for (ch, s) in out_buf.iter_mut().enumerate().take(model_out) {
            let white = splitmix64_bipolar(i, ch as u64);
            let y = f64::from(sanitize_sample(*s)) + hum + noise_a * white;
            *s = self.out_bit_depth.quantize_f64_to_f32(y);
        }
        write_outputs(outputs, &out_buf, model_out);
    }
}

// ============================================================================
// Case #16: Per-Channel Hysteretic Hardware Noise Gate (`-48 dBFS`)
// ============================================================================

/// `Case #16` (`HardwareNoiseGateDecorator`): Per-channel hysteretic noise gate opening above
/// `-48 dBFS` (`0.003981`) and closing below `-51 dBFS` (`0.707 * threshold_lin`).
#[derive(Clone, Debug)]
pub struct HardwareNoiseGateDecorator<M: BlackBoxModel> {
    pub inner: M,
    pub threshold_lin: f32,
    env: [f32; 4],
    gate_open: [bool; 4],
    gate_gain: [f32; 4],
}

impl<M: BlackBoxModel> HardwareNoiseGateDecorator<M> {
    #[must_use]
    pub fn new(inner: M) -> Self {
        Self {
            inner,
            threshold_lin: 0.003_981_07, // -48 dBFS
            env: [0.0; 4],
            gate_open: [false; 4],
            gate_gain: [0.0; 4],
        }
    }
}

impl<M: BlackBoxModel + Default> Default for HardwareNoiseGateDecorator<M> {
    fn default() -> Self {
        Self::new(M::default())
    }
}

impl<M: BlackBoxModel> BlackBoxModel for HardwareNoiseGateDecorator<M> {
    fn reset(&mut self) {
        self.inner.reset();
        self.env = [0.0; 4];
        self.gate_open = [false; 4];
        self.gate_gain = [0.0; 4];
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
        let fs = clamp_sample_rate_hz(self.sample_rate_hz()) as f32;
        let a_att = 1.0 - (-1.0 / (0.000_5 * fs)).exp();
        let a_rel = 1.0 - (-1.0 / (0.040 * fs)).exp();
        let a_smooth = 1.0 - (-1.0 / (0.001 * fs)).exp();
        let open_th = if self.threshold_lin.is_finite() {
            self.threshold_lin.max(0.0)
        } else {
            0.003_981_07
        };
        let close_th = 0.707_106_77 * open_th;

        let mut gated = [0.0_f32; 4];
        let n_in = inputs.len().min(self.num_inputs()).min(4);
        for (ch, &raw_x) in inputs.iter().enumerate().take(n_in) {
            let x = sanitize_sample(raw_x);
            let ax = x.abs();
            let coeff = if ax > self.env[ch] { a_att } else { a_rel };
            self.env[ch] += coeff * (ax - self.env[ch]);
            if self.env[ch] < f32::MIN_POSITIVE {
                self.env[ch] = 0.0;
            }
            if !self.gate_open[ch] && self.env[ch] >= open_th {
                self.gate_open[ch] = true;
            } else if self.gate_open[ch] && self.env[ch] < close_th {
                self.gate_open[ch] = false;
            }
            let target_gain = if self.gate_open[ch] { 1.0_f32 } else { 0.0_f32 };
            if target_gain == 0.0 && self.env[ch] < open_th && self.gate_gain[ch] == 0.0 {
                gated[ch] = 0.0;
            } else {
                self.gate_gain[ch] += a_smooth * (target_gain - self.gate_gain[ch]);
                if self.gate_gain[ch] < f32::MIN_POSITIVE {
                    self.gate_gain[ch] = 0.0;
                }
                gated[ch] = sanitize_sample(x * self.gate_gain[ch]);
            }
        }
        let mut out_buf = [0.0_f32; 4];
        let model_out = step_inner_frame(
            &mut self.inner,
            &gated[..n_in],
            controls,
            switches,
            &mut out_buf,
        );
        write_outputs(outputs, &out_buf, model_out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chains::{
        Chain01SymmetricOd, Chain02GatedAsymmetricFuzz, Chain03ParallelCleanBlend,
        Chain04TubeAmpFmv4x12, Chain05PureLtiCabAndRoom, Chain07StereoChorusAndRotary,
    };
    use crate::test_utils::{
        exercise_blackbox_metadata_and_format, run_mono_block, run_stereo_block, sine_wave,
    };

    #[test]
    fn test_thermal_and_noise_decorators_unit() {
        let dry = sine_wave(128, 330.0, 0.5);
        let silence = vec![0.0_f32; 128];

        // 1. TubeThermalHysteresisDecorator (mono + stereo shared supply check)
        let mut d4 = TubeThermalHysteresisDecorator::new(Chain04TubeAmpFmv4x12::new());
        exercise_blackbox_metadata_and_format(&mut d4);
        assert_eq!(d4.metadata().id, "Chain-04-Tube-Amp-FMV-4x12");
        let y_d4 = run_mono_block(&mut d4, &dry, &[0.5, 0.2, 0.1, 0.3], &[1]);
        assert!(d4.sag_state > 0.0 && d4.thermal_state > 0.0);
        assert!(y_d4.iter().all(|v| v.is_finite()));

        // 2. GermaniumTempDriftDecorator: responds even when empty controls slice is passed
        let mut d7_cold = GermaniumTempDriftDecorator::new(Chain02GatedAsymmetricFuzz::new(), 18.0);
        exercise_blackbox_metadata_and_format(&mut d7_cold);
        let mut d7_hot = GermaniumTempDriftDecorator::new(Chain02GatedAsymmetricFuzz::new(), 36.0);
        let y_cold = run_mono_block(&mut d7_cold, &dry, &[], &[]);
        let y_hot = run_mono_block(&mut d7_hot, &dry, &[], &[]);
        assert!(
            y_cold
                .iter()
                .zip(y_hot.iter())
                .any(|(a, b)| (a - b).abs() > 1.0e-3)
        );

        // 3. MainsHumAndNoiseFloorDecorator: uncorrelated per-channel hiss + lag-1 autocorrelation near 0
        let mut d8 = MainsHumAndNoiseFloorDecorator::new(Chain07StereoChorusAndRotary::new(), 60.0);
        exercise_blackbox_metadata_and_format(&mut d8);
        d8.hum_amp = 0.0; // isolate white noise floor
        let sil_long = vec![0.0_f32; 2_048];
        let (nl, nr) = run_stereo_block(&mut d8, &sil_long, &sil_long, &[0.0, 0.0, 0.0], &[0]);
        let var_l: f64 = nl.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let cov_lr: f64 = nl
            .iter()
            .zip(nr.iter())
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum();
        let lag1_l: f64 = nl
            .windows(2)
            .map(|w| f64::from(w[0]) * f64::from(w[1]))
            .sum();
        assert!(var_l > 1.0e-6);
        assert!(
            (cov_lr / var_l).abs() < 0.08,
            "stereo noise channels must be uncorrelated"
        );
        assert!(
            (lag1_l / var_l).abs() < 0.08,
            "white noise must not be a periodic sawtooth"
        );

        // 4. HardwareNoiseGateDecorator: independent per-channel stereo gating
        let mut d9 = HardwareNoiseGateDecorator::new(Chain07StereoChorusAndRotary::new());
        exercise_blackbox_metadata_and_format(&mut d9);
        let quiet = vec![0.000_1_f32; 128];
        let (gl_q, gr_q) = run_stereo_block(&mut d9, &quiet, &quiet, &[0.0, 0.0, 0.0], &[0]);
        assert!(gl_q.iter().all(|&v| v == 0.0) && gr_q.iter().all(|&v| v == 0.0));
        d9.reset();
        let (gl_mix, _gr_mix) = run_stereo_block(&mut d9, &dry, &silence, &[0.0, 0.0, 0.0], &[0]);
        assert!(gl_mix.iter().any(|&v| v.abs() > 1.0e-3));

        // Cases #12 & #13 topology variance and parallel clean blend checks
        let mut lti = Chain05PureLtiCabAndRoom::new();
        let mut od = Chain01SymmetricOd::new();
        let probe_lo = sine_wave(64, 330.0, 0.25);
        let probe_hi = sine_wave(64, 330.0, 0.50);
        let (lti_1, _) = run_stereo_block(&mut lti, &probe_lo, &probe_lo, &[], &[]);
        let (lti_2, _) = run_stereo_block(&mut lti, &probe_hi, &probe_hi, &[], &[]);
        assert!((lti_2[48] - 2.0 * lti_1[48]).abs() < 1.0e-5);
        let od_1 = run_mono_block(&mut od, &probe_lo, &[0.8, 0.0, 0.0], &[]);
        let od_2 = run_mono_block(&mut od, &probe_hi, &[0.8, 0.0, 0.0], &[]);
        assert!((od_2[48] - 2.0 * od_1[48]).abs() > 1.0e-2);

        let (clean_lo, dirty_lo) = Chain03ParallelCleanBlend::dual_gang_weights(-0.9);
        let (clean_hi, dirty_hi) = Chain03ParallelCleanBlend::dual_gang_weights(0.9);
        assert!(clean_lo > clean_hi && dirty_hi > dirty_lo);

        // Mono slice process_frame vs process_block parity on asymmetric stereo sag + single quantization
        let mut sag_blk = TubeThermalHysteresisDecorator::new(
            Chain05PureLtiCabAndRoom::with_format(48_000, FixtureBitDepth::Int12),
        );
        assert_eq!(sag_blk.bit_depth(), FixtureBitDepth::Int12);
        assert_eq!(sag_blk.inner.bit_depth(), FixtureBitDepth::Float32);
        let mut sag_frm = sag_blk.clone();
        let mut blk_ch0 = vec![0.0_f32; 32];
        sag_blk.process_block(&[&dry[..32], &silence[..32]], &mut [&mut blk_ch0], &[], &[]);
        let mut frm_ch0 = vec![0.0_f32; 32];
        for i in 0..32 {
            sag_frm.process_frame(&[dry[i], silence[i]], &mut frm_ch0[i..=i], &[], &[]);
        }
        assert_eq!(blk_ch0, frm_ch0);

        // Sanitize corrupted public sag_state / thermal_state (-4.0 divide-by-zero & NaN)
        sag_blk.sag_state = -4.0;
        sag_blk.thermal_state = f64::NAN;
        let mut rec_out = [0.0_f32; 2];
        sag_blk.process_frame(&[0.5, -0.5], &mut rec_out, &[], &[]);
        assert!(rec_out[0].is_finite() && rec_out[1].is_finite());
        assert!(sag_blk.sag_state.is_finite() && sag_blk.thermal_state.is_finite());

        // Verify HardwareNoiseGateDecorator flushes subnormals to exact 0.0 after burst -> silence
        let mut gate_dn = HardwareNoiseGateDecorator::new(Chain01SymmetricOd::new());
        let mut g_out = [0.0_f32; 1];
        for _ in 0..128 {
            gate_dn.process_frame(&[0.5], &mut g_out, &[], &[]);
        }
        for _ in 0..180_000 {
            gate_dn.process_frame(&[0.0], &mut g_out, &[], &[]);
        }
        assert_eq!((gate_dn.env[0], gate_dn.gate_gain[0]), (0.0, 0.0));
        gate_dn.process_frame(&[1.0e-3], &mut g_out, &[], &[]);
        assert_eq!(g_out[0], 0.0);
    }
}
