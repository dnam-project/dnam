//! `Chain-10`: Monophonic Octave Divider + Ring Modulator (`OC-2` + Octavia).

#![forbid(unsafe_code)]

use crate::chains::{Oversampler4xF64, scale_alpha_from_192k, substep_rate_4x};
use crate::{
    BlackBoxModel, DEFAULT_FIXTURE_SAMPLE_RATE_HZ, FixtureBitDepth, FixtureControlSpec,
    FixtureDeviceMetadata, FixtureSwitchSpec, clamp_sample_rate_hz, sanitize_norm,
};
use std::f64::consts::PI;

const CHAIN10_METADATA: FixtureDeviceMetadata = FixtureDeviceMetadata {
    id: "Chain-10-OC2-Octave-Divider-And-RingMod",
    name: "Analog Octave Divider & Ring Modulator",
    category: "octave_fuzz",
    sample_rate_hz: DEFAULT_FIXTURE_SAMPLE_RATE_HZ,
    bit_depth: FixtureBitDepth::Float32,
    expected_tail_ms: 35.0,
    supports_tempo_sync: false,
    controls: &[
        FixtureControlSpec {
            id: "direct",
            name: "Direct Level",
            unit: "%",
            default_norm: 0.0,
            min_physical: 0.0,
            max_physical: 100.0,
        },
        FixtureControlSpec {
            id: "oct1",
            name: "Octave -1",
            unit: "%",
            default_norm: 0.2,
            min_physical: 0.0,
            max_physical: 100.0,
        },
        FixtureControlSpec {
            id: "oct2",
            name: "Octave -2",
            unit: "%",
            default_norm: -0.2,
            min_physical: 0.0,
            max_physical: 100.0,
        },
    ],
    switches: &[FixtureSwitchSpec {
        id: "upper_mode",
        name: "Upper Harmonic Mode",
        options: &["Octavia Doubler", "Ring Modulator"],
        default_index: 0,
    }],
};

/// `Chain-10-OC2-Octave-Divider-And-RingMod`: Zero-crossing flip-flop `-1 Oct` and `-2 Oct`
/// dividers + diode ring-modulator / Octavia full-wave frequency doubler (`Direct`, `Oct 1`, `Oct 2`, `Ring`).
#[derive(Clone, Debug)]
pub struct Chain10Oc2OctaveDividerAndRingMod {
    sample_rate_hz: u32,
    bit_depth: FixtureBitDepth,
    os: Oversampler4xF64,
    tracker_lp: f64,
    prev_sign: f64,
    ff_oct1: f64,
    ff_oct2: f64,
    oct1_count: u32,
    env_follow: f64,
    carrier_phase: f64,
}

impl Default for Chain10Oc2OctaveDividerAndRingMod {
    fn default() -> Self {
        Self::new()
    }
}

impl Chain10Oc2OctaveDividerAndRingMod {
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
            tracker_lp: 0.0,
            prev_sign: 1.0,
            ff_oct1: 1.0,
            ff_oct2: 1.0,
            oct1_count: 0,
            env_follow: 0.0,
            carrier_phase: 0.0,
        }
    }

    #[inline]
    pub fn step_sample(&mut self, x: f32, controls: &[f32], switches: &[u8]) -> f32 {
        let x_clean = self.bit_depth.quantize_sample(x);
        let dir_u = f64::from(sanitize_norm(controls.first().copied().unwrap_or(0.0), 0.0));
        let oct1_u = f64::from(sanitize_norm(controls.get(1).copied().unwrap_or(0.2), 0.2));
        let oct2_u = f64::from(sanitize_norm(
            controls.get(2).copied().unwrap_or(-0.2),
            -0.2,
        ));
        let ring_mode = switches.first().copied().unwrap_or(0) != 0;

        let fs_4x = substep_rate_4x(self.sample_rate_hz);
        let a_trk = scale_alpha_from_192k(0.012, fs_4x);
        let a_env = scale_alpha_from_192k(0.005, fs_4x);

        let g_dir = 0.5 * (dir_u + 1.0);
        let g_oct1 = 0.5 * (oct1_u + 1.0);
        let g_oct2 = 0.5 * (oct2_u + 1.0);

        let [y, _] = self
            .os
            .process_frame_4x(f64::from(x_clean), 0.0, |_, u, _| {
                self.tracker_lp += a_trk * (u - self.tracker_lp);
                self.env_follow += a_env * (u.abs() - self.env_follow);
                let cur_sign = if self.tracker_lp >= 0.0 { 1.0 } else { -1.0 };
                if cur_sign > 0.0 && self.prev_sign < 0.0 {
                    self.ff_oct1 = -self.ff_oct1;
                    self.oct1_count = self.oct1_count.wrapping_add(1);
                    if (self.oct1_count & 1) == 0 {
                        self.ff_oct2 = -self.ff_oct2;
                    }
                }
                self.prev_sign = cur_sign;

                let sub1 = self.ff_oct1 * self.env_follow;
                let sub2 = self.ff_oct2 * self.env_follow;

                self.carrier_phase = (self.carrier_phase + 220.0 / fs_4x).fract();
                let ring_or_octavia = if ring_mode {
                    u * (2.0 * PI * self.carrier_phase).sin()
                } else {
                    (2.2 * u).abs() - self.env_follow
                };
                let mix = 0.5 * g_dir * u
                    + 0.6 * g_oct1 * sub1
                    + 0.4 * g_oct2 * sub2
                    + 0.35 * ring_or_octavia;
                [mix.tanh(), 0.0]
            });
        self.bit_depth.quantize_f64_to_f32(y)
    }
}

impl BlackBoxModel for Chain10Oc2OctaveDividerAndRingMod {
    fn reset(&mut self) {
        self.os.reset();
        self.tracker_lp = 0.0;
        self.prev_sign = 1.0;
        self.ff_oct1 = 1.0;
        self.ff_oct2 = 1.0;
        self.oct1_count = 0;
        self.env_follow = 0.0;
        self.carrier_phase = 0.0;
    }

    fn metadata(&self) -> FixtureDeviceMetadata {
        FixtureDeviceMetadata {
            sample_rate_hz: self.sample_rate_hz,
            bit_depth: self.bit_depth,
            ..CHAIN10_METADATA
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
        switches: &[u8],
    ) {
        let x = inputs.first().copied().unwrap_or(0.0);
        crate::chains::write_mono_output(outputs, self.step_sample(x, controls, switches));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{run_mono_block, sine_wave};

    #[test]
    fn test_chain10_octave_divider_and_ringmod_modes() {
        let mut c10 = Chain10Oc2OctaveDividerAndRingMod::default();
        assert_eq!(c10.metadata().id, "Chain-10-OC2-Octave-Divider-And-RingMod");
        assert_eq!(
            (
                c10.num_inputs(),
                c10.num_outputs(),
                c10.num_controls(),
                c10.num_switches()
            ),
            (1, 1, 3, 1)
        );
        let dry = sine_wave(128, 220.0, 0.45);
        let y_oct = run_mono_block(&mut c10, &dry, &[0.2, 0.7, 0.5], &[0]);
        let y_ring = run_mono_block(&mut c10, &dry, &[0.2, 0.7, 0.5], &[1]);
        assert_ne!(y_oct, y_ring);
    }
}
