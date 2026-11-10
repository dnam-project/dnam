//! `Chain-01`: Symmetric Soft-Clipping Overdrive (`720 Hz` HPF -> `tanh` -> Post-EQ).

#![forbid(unsafe_code)]

use crate::chains::{Oversampler4xF64, substep_rate_4x};
use crate::{
    BlackBoxModel, DEFAULT_FIXTURE_SAMPLE_RATE_HZ, FixtureBitDepth, FixtureControlSpec,
    FixtureDeviceMetadata, clamp_sample_rate_hz, sanitize_norm,
};
use std::f64::consts::PI;

const CHAIN01_METADATA: FixtureDeviceMetadata = FixtureDeviceMetadata {
    id: "Chain-01-Symmetric-OD",
    name: "Symmetric Soft-Clipping Overdrive",
    category: "overdrive",
    sample_rate_hz: DEFAULT_FIXTURE_SAMPLE_RATE_HZ,
    bit_depth: FixtureBitDepth::Float32,
    expected_tail_ms: 15.0,
    supports_tempo_sync: false,
    controls: &[
        FixtureControlSpec {
            id: "drive",
            name: "Drive",
            unit: "dB",
            default_norm: 0.0,
            min_physical: -6.0,
            max_physical: 30.0,
        },
        FixtureControlSpec {
            id: "tone",
            name: "Tone",
            unit: "Hz",
            default_norm: 0.0,
            min_physical: 800.0,
            max_physical: 6400.0,
        },
        FixtureControlSpec {
            id: "level",
            name: "Level",
            unit: "%",
            default_norm: 0.0,
            min_physical: 0.0,
            max_physical: 100.0,
        },
    ],
    switches: &[],
};

/// `Chain-01-Symmetric-OD`: `1 -> 1` `4x`-oversampled symmetric overdrive (`Drive`, `Tone`, `Level`).
#[derive(Clone, Debug)]
pub struct Chain01SymmetricOd {
    sample_rate_hz: u32,
    bit_depth: FixtureBitDepth,
    os: Oversampler4xF64,
    hpf_state: f64,
    lpf_state: f64,
}

impl Default for Chain01SymmetricOd {
    fn default() -> Self {
        Self::new()
    }
}

impl Chain01SymmetricOd {
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
            hpf_state: 0.0,
            lpf_state: 0.0,
        }
    }

    #[inline]
    pub fn step_sample(&mut self, x: f32, controls: &[f32]) -> f32 {
        let x_clean = self.bit_depth.quantize_sample(x);
        let drive_u = f64::from(sanitize_norm(controls.first().copied().unwrap_or(0.0), 0.0));
        let tone_u = f64::from(sanitize_norm(controls.get(1).copied().unwrap_or(0.0), 0.0));
        let level_u = f64::from(sanitize_norm(controls.get(2).copied().unwrap_or(0.0), 0.0));

        let fs_4x = substep_rate_4x(self.sample_rate_hz);
        let drive_gain = 10.0_f64.powf((12.0 + 18.0 * drive_u) / 20.0);
        let hpf_a = (1.0 - (-2.0 * PI * 720.0 / fs_4x).exp()).clamp(0.0005, 0.45);
        let fc_tone = 800.0 * 2.0_f64.powf(1.5 * (tone_u + 1.0));
        let lpf_a = (1.0 - (-2.0 * PI * fc_tone / fs_4x).exp()).clamp(0.001, 0.75);
        let out_gain = 0.70 * (0.5 * (level_u + 1.0));

        let [y, _] = self
            .os
            .process_frame_4x(f64::from(x_clean), 0.0, |_, u, _| {
                self.hpf_state += hpf_a * (u - self.hpf_state);
                let hp = u - self.hpf_state;
                let nl = (drive_gain * hp).tanh();
                self.lpf_state += lpf_a * (nl - self.lpf_state);
                [self.lpf_state * out_gain, 0.0]
            });
        self.bit_depth.quantize_f64_to_f32(y)
    }
}

impl BlackBoxModel for Chain01SymmetricOd {
    fn reset(&mut self) {
        self.os.reset();
        self.hpf_state = 0.0;
        self.lpf_state = 0.0;
    }

    fn metadata(&self) -> FixtureDeviceMetadata {
        FixtureDeviceMetadata {
            sample_rate_hz: self.sample_rate_hz,
            bit_depth: self.bit_depth,
            ..CHAIN01_METADATA
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
    use crate::test_utils::{run_mono_block, sine_wave, sine_wave_at_rate};

    #[test]
    fn test_chain01_symmetric_od_metadata_reset_controls_and_formats() {
        let mut c1 = Chain01SymmetricOd::new();
        assert_eq!(c1.metadata().id, "Chain-01-Symmetric-OD");
        assert_eq!(c1.sample_rate_hz(), 48_000);
        assert_eq!(c1.bit_depth(), FixtureBitDepth::Float32);
        assert_eq!(
            (
                c1.num_inputs(),
                c1.num_outputs(),
                c1.num_controls(),
                c1.num_switches()
            ),
            (1, 1, 3, 0)
        );
        let dry = sine_wave(128, 220.0, 0.45);
        let y_a = run_mono_block(&mut c1, &dry, &[-0.6, 0.0, 0.0], &[]);
        let y_b = run_mono_block(&mut c1, &dry, &[-0.6, 0.0, 0.0], &[]);
        let y_hi = run_mono_block(&mut c1, &dry, &[0.8, 0.5, 0.4], &[]);
        let y_muted = run_mono_block(&mut c1, &dry, &[0.0, 0.0, -1.0], &[]);
        assert_eq!(y_a, y_b);
        assert_ne!(y_a, y_hi);
        assert!(y_muted.iter().all(|v| v.abs() < 1.0e-9));

        let mut c1_96k_16b = Chain01SymmetricOd::with_format(96_000, FixtureBitDepth::Int16);
        assert_eq!(c1_96k_16b.sample_rate_hz(), 96_000);
        assert_eq!(c1_96k_16b.bit_depth(), FixtureBitDepth::Int16);
        let dry_96k = sine_wave_at_rate(256, 220.0, 0.45, 96_000);
        let y_96k = run_mono_block(&mut c1_96k_16b, &dry_96k, &[0.2, 0.0, 0.5], &[]);
        assert!(
            y_96k
                .iter()
                .all(|&v| v == FixtureBitDepth::Int16.quantize_sample(v))
        );
    }
}
