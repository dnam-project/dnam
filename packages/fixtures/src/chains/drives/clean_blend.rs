//! `Chain-03`: Parallel Clean-Blend Drive (`Klon`-style dual-ganged Gain pot).

#![forbid(unsafe_code)]

use crate::chains::{Oversampler4xF64, scale_alpha_from_192k, substep_rate_4x};
use crate::{
    BlackBoxModel, DEFAULT_FIXTURE_SAMPLE_RATE_HZ, FixtureBitDepth, FixtureControlSpec,
    FixtureDeviceMetadata, clamp_sample_rate_hz, sanitize_norm,
};

const CHAIN03_METADATA: FixtureDeviceMetadata = FixtureDeviceMetadata {
    id: "Chain-03-Parallel-Clean-Blend",
    name: "Parallel Clean-Blend Drive",
    category: "overdrive",
    sample_rate_hz: DEFAULT_FIXTURE_SAMPLE_RATE_HZ,
    bit_depth: FixtureBitDepth::Float32,
    expected_tail_ms: 20.0,
    supports_tempo_sync: false,
    controls: &[
        FixtureControlSpec {
            id: "gain",
            name: "Gain (Dual-Gang)",
            unit: "%",
            default_norm: 0.0,
            min_physical: 0.0,
            max_physical: 100.0,
        },
        FixtureControlSpec {
            id: "treble",
            name: "Treble",
            unit: "dB",
            default_norm: 0.0,
            min_physical: -12.0,
            max_physical: 12.0,
        },
        FixtureControlSpec {
            id: "output",
            name: "Output",
            unit: "%",
            default_norm: 0.0,
            min_physical: 0.0,
            max_physical: 100.0,
        },
    ],
    switches: &[],
};

/// `Chain-03-Parallel-Clean-Blend`: `1 -> 1` dual-ganged `Gain` pot crossfading mid-filtered clean + germanium clip.
#[derive(Clone, Debug)]
pub struct Chain03ParallelCleanBlend {
    sample_rate_hz: u32,
    bit_depth: FixtureBitDepth,
    os: Oversampler4xF64,
    clean_mid_state: f64,
    drive_hp_state: f64,
    treble_state: f64,
}

impl Default for Chain03ParallelCleanBlend {
    fn default() -> Self {
        Self::new()
    }
}

impl Chain03ParallelCleanBlend {
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
            clean_mid_state: 0.0,
            drive_hp_state: 0.0,
            treble_state: 0.0,
        }
    }

    /// Returns `(clean_weight, dirty_weight)` for the dual-ganged `Gain` pot at `gain_u in [-1.0, 1.0]`.
    #[must_use]
    pub fn dual_gang_weights(gain_u: f32) -> (f32, f32) {
        let g_norm = 0.5 * (sanitize_norm(gain_u, 0.0) + 1.0);
        (1.0 - 0.85 * g_norm, 0.15 + 0.85 * g_norm)
    }

    #[inline]
    pub fn step_sample(&mut self, x: f32, controls: &[f32]) -> f32 {
        let x_clean = self.bit_depth.quantize_sample(x);
        let gain_u = f64::from(sanitize_norm(controls.first().copied().unwrap_or(0.0), 0.0));
        let treble_u = f64::from(sanitize_norm(controls.get(1).copied().unwrap_or(0.0), 0.0));
        let out_u = f64::from(sanitize_norm(controls.get(2).copied().unwrap_or(0.0), 0.0));

        let fs_4x = substep_rate_4x(self.sample_rate_hz);
        let a_clean_mid = scale_alpha_from_192k(0.025, fs_4x);
        let a_drive_hp = scale_alpha_from_192k(0.015, fs_4x);
        let a_treble = scale_alpha_from_192k(0.08, fs_4x);

        let g_norm = 0.5 * (gain_u + 1.0);
        let w_clean = 1.0 - 0.85 * g_norm;
        let w_dirty = 0.15 + 0.85 * g_norm;
        let pre_drive = 1.5 + 14.0 * g_norm;
        let out_gain = 0.90 * (0.5 * (out_u + 1.0));

        let [y, _] = self
            .os
            .process_frame_4x(f64::from(x_clean), 0.0, |_, u, _| {
                self.clean_mid_state += a_clean_mid * (u - self.clean_mid_state);
                let clean_branch = w_clean * (0.65 * u + 0.35 * self.clean_mid_state);

                self.drive_hp_state += a_drive_hp * (u - self.drive_hp_state);
                let hp = u - self.drive_hp_state;
                let clipped = ((pre_drive * hp).clamp(-1.8, 1.8) / 1.8).tanh();
                let dirty_branch = w_dirty * clipped;

                let sum = clean_branch + dirty_branch;
                self.treble_state += a_treble * (sum - self.treble_state);
                let hi = sum - self.treble_state;
                [(sum + 0.45 * treble_u * hi) * out_gain, 0.0]
            });
        self.bit_depth.quantize_f64_to_f32(y)
    }
}

impl BlackBoxModel for Chain03ParallelCleanBlend {
    fn reset(&mut self) {
        self.os.reset();
        self.clean_mid_state = 0.0;
        self.drive_hp_state = 0.0;
        self.treble_state = 0.0;
    }

    fn metadata(&self) -> FixtureDeviceMetadata {
        FixtureDeviceMetadata {
            sample_rate_hz: self.sample_rate_hz,
            bit_depth: self.bit_depth,
            ..CHAIN03_METADATA
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
    fn test_chain03_parallel_clean_blend_weights_and_response() {
        let mut c3 = Chain03ParallelCleanBlend::new();
        assert_eq!(c3.metadata().id, "Chain-03-Parallel-Clean-Blend");
        assert_eq!(
            (
                c3.num_inputs(),
                c3.num_outputs(),
                c3.num_controls(),
                c3.num_switches()
            ),
            (1, 1, 3, 0)
        );
        let (clean_lo, dirty_lo) = Chain03ParallelCleanBlend::dual_gang_weights(-0.9);
        let (clean_hi, dirty_hi) = Chain03ParallelCleanBlend::dual_gang_weights(0.9);
        assert!(clean_lo > clean_hi && dirty_hi > dirty_lo);

        let dry = sine_wave(128, 220.0, 0.45);
        let y_lo = run_mono_block(&mut c3, &dry, &[-0.8, -0.4, 0.0], &[]);
        let y_hi = run_mono_block(&mut c3, &dry, &[0.8, 0.6, 0.2], &[]);
        assert_ne!(y_lo, y_hi);
    }
}
