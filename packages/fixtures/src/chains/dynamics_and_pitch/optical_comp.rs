//! `Chain-09`: Program-Dependent Optical Compressor (`LA-2A` `T4B` Dual-Release & `Aux-Key`).

#![forbid(unsafe_code)]

use crate::chains::{Oversampler4xF64, substep_rate_4x};
use crate::{
    BlackBoxModel, DEFAULT_FIXTURE_SAMPLE_RATE_HZ, FixtureBitDepth, FixtureControlSpec,
    FixtureDeviceMetadata, FixtureSwitchSpec, clamp_sample_rate_hz, sanitize_norm,
};

const CHAIN09_METADATA: FixtureDeviceMetadata = FixtureDeviceMetadata {
    id: "Chain-09-LA2A-Optical-Compressor",
    name: "Program-Dependent T4B Optical Compressor",
    category: "compressor",
    sample_rate_hz: DEFAULT_FIXTURE_SAMPLE_RATE_HZ,
    bit_depth: FixtureBitDepth::Float32,
    expected_tail_ms: 1500.0,
    supports_tempo_sync: false,
    controls: &[
        FixtureControlSpec {
            id: "peak_reduction",
            name: "Peak Reduction",
            unit: "%",
            default_norm: 0.0,
            min_physical: 0.0,
            max_physical: 100.0,
        },
        FixtureControlSpec {
            id: "gain",
            name: "Makeup Gain",
            unit: "dB",
            default_norm: 0.0,
            min_physical: 0.0,
            max_physical: 24.0,
        },
    ],
    switches: &[FixtureSwitchSpec {
        id: "mode",
        name: "Compress / Limit",
        options: &["Compress (3:1)", "Limit (Inf:1)"],
        default_index: 0,
    }],
};

/// `Chain-09-LA2A-Optical-Compressor`: Program-dependent `T4B` optical compressor (`60 ms` fast +
/// `1.5s` slow dual-release electroluminescent cell) supporting `1 -> 1` internal sidechain and
/// `2 -> 1` external `Aux-Key` sidechain (`Peak Reduction`, `Gain`, `Comp/Limit`).
#[derive(Clone, Debug)]
pub struct Chain09La2aOpticalCompressor {
    sample_rate_hz: u32,
    bit_depth: FixtureBitDepth,
    os: Oversampler4xF64,
    fast_env: f64,
    slow_env: f64,
    last_gr_lin: f32,
}

impl Default for Chain09La2aOpticalCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Chain09La2aOpticalCompressor {
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
            fast_env: 0.0,
            slow_env: 0.0,
            last_gr_lin: 1.0,
        }
    }

    /// Returns the instantaneous linear gain reduction factor in `(0.0, 1.0]`.
    #[must_use]
    pub fn current_gain_reduction(&self) -> f32 {
        self.last_gr_lin
    }

    #[inline]
    pub fn step_sample(
        &mut self,
        main_in: f32,
        aux_key_in: Option<f32>,
        controls: &[f32],
        switches: &[u8],
    ) -> f32 {
        let main_clean = self.bit_depth.quantize_sample(main_in);
        let key_clean = self
            .bit_depth
            .quantize_sample(aux_key_in.unwrap_or(main_clean));
        let peak_red_u = f64::from(sanitize_norm(controls.first().copied().unwrap_or(0.0), 0.0));
        let gain_u = f64::from(sanitize_norm(controls.get(1).copied().unwrap_or(0.0), 0.0));
        let is_limit = switches.first().copied().unwrap_or(0) != 0;

        let thresh = 0.35 * (1.0 - 0.65 * (0.5 * (peak_red_u + 1.0)));
        let ratio_slope = if is_limit { 0.88 } else { 0.65 };
        // Maps u in [-1, 1] accurately to [0.0 dB, +24.0 dB] linear makeup gain (1.0 .. 15.85)
        let makeup = 10.0_f64.powf((12.0 * (gain_u + 1.0)) / 20.0);

        let fs_4x = substep_rate_4x(self.sample_rate_hz);
        let alpha_atk = 1.0 / (0.010 * fs_4x);
        let alpha_rel_fast = 1.0 / (0.060 * fs_4x);
        let alpha_rel_slow = 1.0 / (1.500 * fs_4x);

        let mut gr_lin_out = 1.0_f64;
        let [y, _] = self.os.process_frame_4x(
            f64::from(main_clean),
            f64::from(key_clean),
            |_, u_main, u_key| {
                let over = (u_key.abs() - thresh).max(0.0);
                if over > self.fast_env {
                    self.fast_env += alpha_atk * (over - self.fast_env);
                    self.slow_env += (0.25 * alpha_atk) * (over - self.slow_env);
                } else {
                    self.fast_env += alpha_rel_fast * (over - self.fast_env);
                    self.slow_env += alpha_rel_slow * (over - self.slow_env);
                }
                let cell_env = 0.65 * self.fast_env + 0.35 * self.slow_env;
                let gr = 1.0 / (1.0 + 4.5 * ratio_slope * cell_env);
                gr_lin_out = gr;
                let tube_makeup = (u_main * gr * makeup).tanh();
                [tube_makeup, 0.0]
            },
        );
        self.last_gr_lin = gr_lin_out as f32;
        self.bit_depth.quantize_f64_to_f32(y)
    }
}

impl BlackBoxModel for Chain09La2aOpticalCompressor {
    fn reset(&mut self) {
        self.os.reset();
        self.fast_env = 0.0;
        self.slow_env = 0.0;
        self.last_gr_lin = 1.0;
    }

    fn metadata(&self) -> FixtureDeviceMetadata {
        FixtureDeviceMetadata {
            sample_rate_hz: self.sample_rate_hz,
            bit_depth: self.bit_depth,
            ..CHAIN09_METADATA
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
        2
    }

    fn num_outputs(&self) -> usize {
        1
    }

    fn process_frame(
        &mut self,
        inputs: &[f32],
        outputs: &mut [f32],
        controls: &[f32],
        switches: &[u8],
    ) {
        let main_in = inputs.first().copied().unwrap_or(0.0);
        let aux_in = inputs.get(1).copied();
        crate::chains::write_mono_output(
            outputs,
            self.step_sample(main_in, aux_in, controls, switches),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::sine_wave;

    #[test]
    fn test_chain09_optical_compressor_default_gr_and_aux_sidechain() {
        let mut c9 = Chain09La2aOpticalCompressor::default();
        assert_eq!(c9.current_gain_reduction(), 1.0);
        assert_eq!(c9.metadata().id, "Chain-09-LA2A-Optical-Compressor");
        assert_eq!(
            (
                c9.num_inputs(),
                c9.num_outputs(),
                c9.num_controls(),
                c9.num_switches()
            ),
            (2, 1, 2, 1)
        );
        let dry_l = sine_wave(128, 220.0, 0.45);
        let dry_r = sine_wave(128, 330.0, 0.45);
        c9.reset();
        let mut y_comp = vec![0.0_f32; dry_l.len()];
        c9.process_block(
            &[&dry_l, &dry_r],
            &mut [y_comp.as_mut_slice()],
            &[0.6, -0.5],
            &[0],
        );
        c9.reset();
        let mut y_lim = vec![0.0_f32; dry_l.len()];
        c9.process_block(
            &[&dry_l, &dry_r],
            &mut [y_lim.as_mut_slice()],
            &[0.6, -0.5],
            &[1],
        );
        assert_ne!(y_comp, y_lim);
        assert!(c9.current_gain_reduction() < 1.0);
    }
}
