//! `Chain-02`: Gated Asymmetric Silicon/Germanium Fuzz (`-28 dBFS` Bias Gate).

#![forbid(unsafe_code)]

use crate::chains::{Oversampler4xF64, scale_alpha_from_192k, substep_rate_4x};
use crate::{
    BlackBoxModel, DEFAULT_FIXTURE_SAMPLE_RATE_HZ, FixtureBitDepth, FixtureControlSpec,
    FixtureDeviceMetadata, clamp_sample_rate_hz, sanitize_norm,
};

const CHAIN02_METADATA: FixtureDeviceMetadata = FixtureDeviceMetadata {
    id: "Chain-02-Gated-Asymmetric-Fuzz",
    name: "Gated Asymmetric Silicon/Germanium Fuzz",
    category: "fuzz",
    sample_rate_hz: DEFAULT_FIXTURE_SAMPLE_RATE_HZ,
    bit_depth: FixtureBitDepth::Float32,
    expected_tail_ms: 40.0,
    supports_tempo_sync: false,
    controls: &[
        FixtureControlSpec {
            id: "fuzz",
            name: "Fuzz",
            unit: "%",
            default_norm: 0.0,
            min_physical: 0.0,
            max_physical: 100.0,
        },
        FixtureControlSpec {
            id: "bias",
            name: "Bias / Gate",
            unit: "V",
            default_norm: 0.0,
            min_physical: 3.0,
            max_physical: 7.5,
        },
        FixtureControlSpec {
            id: "volume",
            name: "Volume",
            unit: "%",
            default_norm: 0.0,
            min_physical: 0.0,
            max_physical: 100.0,
        },
    ],
    switches: &[],
};

/// `Chain-02-Gated-Asymmetric-Fuzz`: `1 -> 1` `4x`-oversampled asymmetric fuzz with `-28 dBFS` bias pinch-off.
#[derive(Clone, Debug)]
pub struct Chain02GatedAsymmetricFuzz {
    sample_rate_hz: u32,
    bit_depth: FixtureBitDepth,
    os: Oversampler4xF64,
    env_state: f64,
    dc_block_x1: f64,
    dc_block_y1: f64,
}

impl Default for Chain02GatedAsymmetricFuzz {
    fn default() -> Self {
        Self::new()
    }
}

impl Chain02GatedAsymmetricFuzz {
    #[must_use]
    pub fn new() -> Self {
        Self::with_format(DEFAULT_FIXTURE_SAMPLE_RATE_HZ, FixtureBitDepth::Float32)
    }

    #[must_use]
    pub fn with_format(sample_rate_hz: u32, bit_depth: FixtureBitDepth) -> Self {
        Self {
            sample_rate_hz: clamp_sample_rate_hz(sample_rate_hz),
            bit_depth,
            os: Oversampler4xF64::default(),
            env_state: 0.0,
            dc_block_x1: 0.0,
            dc_block_y1: 0.0,
        }
    }

    #[inline]
    pub fn step_sample(&mut self, x: f32, controls: &[f32]) -> f32 {
        let x_clean = self.bit_depth.quantize_sample(x);
        let fuzz_u = f64::from(sanitize_norm(controls.first().copied().unwrap_or(0.0), 0.0));
        let bias_u = f64::from(sanitize_norm(controls.get(1).copied().unwrap_or(0.0), 0.0));
        let vol_u = f64::from(sanitize_norm(controls.get(2).copied().unwrap_or(0.0), 0.0));

        let fs_4x = substep_rate_4x(self.sample_rate_hz);
        let a_atk = scale_alpha_from_192k(0.02, fs_4x);
        let a_rel = scale_alpha_from_192k(0.0008, fs_4x);
        let dc_pole = (0.995_f64).powf(48_000.0 / f64::from(self.sample_rate_hz));

        let drive = 18.0 * 2.0_f64.powf(1.2 * fuzz_u);
        // -28 dBFS nominal gate threshold (~0.03981), adjustable via Bias/Gate knob
        let gate_thresh = 0.039_810_7 * (1.0 + 0.5 * bias_u).max(0.1);
        let asym_offset = 0.28 + 0.15 * bias_u;
        let vol = 0.70 * (0.5 * (vol_u + 1.0));

        let [y_raw, _] = self
            .os
            .process_frame_4x(f64::from(x_clean), 0.0, |_, u, _| {
                let abs_u = u.abs();
                let alpha = if abs_u > self.env_state { a_atk } else { a_rel };
                self.env_state += alpha * (abs_u - self.env_state);
                let ratio = self.env_state / gate_thresh;
                let gate_gain = (ratio * ratio) / (1.0 + ratio * ratio);
                let biased = drive * u * gate_gain + asym_offset;
                let nl = if biased >= 0.0 {
                    biased.tanh()
                } else {
                    0.55 * (1.6 * biased).tanh()
                } - asym_offset.tanh();
                [nl * vol, 0.0]
            });

        let y_dc = y_raw - self.dc_block_x1 + dc_pole * self.dc_block_y1;
        self.dc_block_x1 = y_raw;
        self.dc_block_y1 = y_dc;
        self.bit_depth.quantize_f64_to_f32(y_dc)
    }
}

impl BlackBoxModel for Chain02GatedAsymmetricFuzz {
    fn reset(&mut self) {
        self.os.reset();
        self.env_state = 0.0;
        self.dc_block_x1 = 0.0;
        self.dc_block_y1 = 0.0;
    }

    fn metadata(&self) -> FixtureDeviceMetadata {
        FixtureDeviceMetadata {
            sample_rate_hz: self.sample_rate_hz,
            bit_depth: self.bit_depth,
            ..CHAIN02_METADATA
        }
    }

    fn set_sample_rate_hz(&mut self, sample_rate_hz: u32) {
        self.sample_rate_hz = clamp_sample_rate_hz(sample_rate_hz);
        self.reset();
    }

    fn set_bit_depth(&mut self, bit_depth: FixtureBitDepth) {
        self.bit_depth = bit_depth;
        self.reset();
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }

    fn process_frame(
        &mut self,
        inputs: &[f32],
        outputs: &mut [f32],
        controls: &[f32],
        _switches: &[u8],
    ) {
        let x = inputs.first().copied().unwrap_or(0.0);
        crate::chains::write_mono_output(outputs, self.step_sample(x, controls));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{run_mono_block, sine_wave};

    #[test]
    fn test_chain02_gated_fuzz_metadata_reset_and_bias_gate() {
        let mut c2 = Chain02GatedAsymmetricFuzz::new();
        assert_eq!(c2.metadata().id, "Chain-02-Gated-Asymmetric-Fuzz");
        assert_eq!(
            (
                c2.num_inputs(),
                c2.num_outputs(),
                c2.num_controls(),
                c2.num_switches()
            ),
            (1, 1, 3, 0)
        );
        let dry = sine_wave(128, 220.0, 0.45);
        let y_a = run_mono_block(&mut c2, &dry, &[0.5, 0.2, 0.0], &[]);
        let y_b = run_mono_block(&mut c2, &dry, &[0.5, 0.2, 0.0], &[]);
        let y_bias = run_mono_block(&mut c2, &dry, &[0.5, -0.8, 0.0], &[]);
        assert_eq!(y_a, y_b);
        assert_ne!(y_a, y_bias);
    }
}
