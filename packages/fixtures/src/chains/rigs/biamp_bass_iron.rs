//! `Chain-11`: Bi-Amp Bass Crossover (`200 Hz` `LR4`) + Output Transformer Iron (`B-H` Hysteresis).

#![forbid(unsafe_code)]

use crate::chains::{Oversampler4xF64, scale_alpha_from_192k, substep_rate_4x};
use crate::{
    BlackBoxModel, DEFAULT_FIXTURE_SAMPLE_RATE_HZ, FixtureBitDepth, FixtureControlSpec,
    FixtureDeviceMetadata, FixtureSwitchSpec, clamp_sample_rate_hz, sanitize_norm,
};
use std::f64::consts::PI;

const CHAIN11_METADATA: FixtureDeviceMetadata = FixtureDeviceMetadata {
    id: "Chain-11-BiAmp-Bass-And-Output-Iron",
    name: "Bi-Amp Bass Crossover (200 Hz LR4) + Output Iron",
    category: "bass_rig",
    sample_rate_hz: DEFAULT_FIXTURE_SAMPLE_RATE_HZ,
    bit_depth: FixtureBitDepth::Float32,
    expected_tail_ms: 95.0,
    supports_tempo_sync: false,
    controls: &[
        FixtureControlSpec {
            id: "blend",
            name: "Clean / CMOS Blend",
            unit: "%",
            default_norm: 0.0,
            min_physical: 0.0,
            max_physical: 100.0,
        },
        FixtureControlSpec {
            id: "drive",
            name: "CMOS Drive",
            unit: "%",
            default_norm: 0.0,
            min_physical: 0.0,
            max_physical: 100.0,
        },
    ],
    switches: &[FixtureSwitchSpec {
        id: "grunt",
        name: "LF Iron Grunt Boost",
        options: &["Normal", "Grunt (+2.6 dB)"],
        default_index: 0,
    }],
};

/// Internal `f64` 4th-order Linkwitz-Riley (`LR4`) crossover at `200 Hz`.
#[derive(Clone, Debug, Default)]
struct Lr4Crossover200HzF64 {
    lp_ic1: [f64; 2],
    lp_ic2: [f64; 2],
    hp_ic1: [f64; 2],
    hp_ic2: [f64; 2],
}

impl Lr4Crossover200HzF64 {
    #[inline]
    fn step(&mut self, x: f64, sample_rate_hz: u32) -> (f64, f64) {
        let fs = f64::from(clamp_sample_rate_hz(sample_rate_hz));
        let g = (PI * 200.0 / fs).tan().min(0.95);
        let k = std::f64::consts::SQRT_2;
        let a1 = 1.0 / (1.0 + g * (g + k));
        let a2 = g * a1;
        let a3 = g * a2;

        let mut u_lp = x;
        for s in 0..2 {
            let v3 = u_lp - self.lp_ic2[s];
            let v1 = a1 * self.lp_ic1[s] + a2 * v3;
            let v2 = self.lp_ic2[s] + a2 * self.lp_ic1[s] + a3 * v3;
            self.lp_ic1[s] = 2.0 * v1 - self.lp_ic1[s];
            self.lp_ic2[s] = 2.0 * v2 - self.lp_ic2[s];
            u_lp = v2;
        }

        let mut u_hp = x;
        for s in 0..2 {
            let v3 = u_hp - self.hp_ic2[s];
            let v1 = a1 * self.hp_ic1[s] + a2 * v3;
            let v2 = self.hp_ic2[s] + a2 * self.hp_ic1[s] + a3 * v3;
            self.hp_ic1[s] = 2.0 * v1 - self.hp_ic1[s];
            self.hp_ic2[s] = 2.0 * v2 - self.hp_ic2[s];
            u_hp = u_hp - k * v1 - v2;
        }

        (u_lp, u_hp)
    }
}

/// `Chain-11-BiAmp-Bass-And-Output-Iron`: `200 Hz` `LR4` crossover clean sub-bass path +
/// high-pass CMOS drive + `B-H` quadrature hysteresis LF output transformer core saturation.
#[derive(Clone, Debug)]
pub struct Chain11BiAmpBassAndOutputIron {
    sample_rate_hz: u32,
    bit_depth: FixtureBitDepth,
    crossover: Lr4Crossover200HzF64,
    os: Oversampler4xF64,
    bh_flux_state: f64,
}

impl Default for Chain11BiAmpBassAndOutputIron {
    fn default() -> Self {
        Self::new()
    }
}

impl Chain11BiAmpBassAndOutputIron {
    #[must_use]
    pub fn new() -> Self {
        Self::with_format(DEFAULT_FIXTURE_SAMPLE_RATE_HZ, FixtureBitDepth::Float32)
    }

    #[must_use]
    pub fn with_format(sample_rate_hz: u32, bit_depth: FixtureBitDepth) -> Self {
        Self {
            sample_rate_hz: clamp_sample_rate_hz(sample_rate_hz),
            bit_depth,
            crossover: Lr4Crossover200HzF64::default(),
            os: Oversampler4xF64::default(),
            bh_flux_state: 0.0,
        }
    }

    /// Returns the instantaneous LF output transformer core magnetic flux state.
    #[must_use]
    pub fn current_flux(&self) -> f32 {
        crate::sanitize_sample(self.bh_flux_state as f32)
    }

    #[inline]
    pub fn step_sample(&mut self, x: f32, controls: &[f32], switches: &[u8]) -> f32 {
        let x_clean = self.bit_depth.quantize_sample(x);
        let blend_u = f64::from(sanitize_norm(controls.first().copied().unwrap_or(0.0), 0.0));
        let drive_u = f64::from(sanitize_norm(controls.get(1).copied().unwrap_or(0.0), 0.0));
        let grunt_boost = if switches.first().copied().unwrap_or(0) != 0 {
            1.35
        } else {
            1.0
        };

        let (sub_clean, hp_raw) = self.crossover.step(f64::from(x_clean), self.sample_rate_hz);
        let w_blend = 0.5 * (blend_u + 1.0);
        let cmos_drive = 2.5 + 9.0 * (0.5 * (drive_u + 1.0));
        let fs_4x = substep_rate_4x(self.sample_rate_hz);
        let a_flux = scale_alpha_from_192k(0.004, fs_4x);

        let [y, _] = self
            .os
            .process_frame_4x(sub_clean, hp_raw, |_, u_sub, u_hp| {
                self.bh_flux_state += a_flux * (u_sub * grunt_boost - self.bh_flux_state);
                let iron_sat = (u_sub + 0.45 * (2.5 * self.bh_flux_state).tanh()
                    - 0.15 * self.bh_flux_state)
                    .tanh();
                let cmos_dirt = (cmos_drive * u_hp).tanh();
                let combined = (1.0 - 0.65 * w_blend) * iron_sat + 0.65 * w_blend * cmos_dirt;
                [combined, 0.0]
            });
        self.bit_depth.quantize_f64_to_f32(y)
    }
}

impl BlackBoxModel for Chain11BiAmpBassAndOutputIron {
    fn reset(&mut self) {
        self.crossover = Lr4Crossover200HzF64::default();
        self.os.reset();
        self.bh_flux_state = 0.0;
    }

    fn metadata(&self) -> FixtureDeviceMetadata {
        FixtureDeviceMetadata {
            sample_rate_hz: self.sample_rate_hz,
            bit_depth: self.bit_depth,
            ..CHAIN11_METADATA
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
    fn test_chain11_biamp_bass_crossover_and_output_iron() {
        let mut c11 = Chain11BiAmpBassAndOutputIron::default();
        assert_eq!(c11.metadata().id, "Chain-11-BiAmp-Bass-And-Output-Iron");
        assert_eq!(
            (
                c11.num_inputs(),
                c11.num_outputs(),
                c11.num_controls(),
                c11.num_switches()
            ),
            (1, 1, 2, 1)
        );
        let dry = sine_wave(128, 80.0, 0.45);
        let y_norm = run_mono_block(&mut c11, &dry, &[0.1, 0.5], &[0]);
        let y_grunt = run_mono_block(&mut c11, &dry, &[0.1, 0.5], &[1]);
        assert_ne!(y_norm, y_grunt);
        assert!(c11.current_flux().abs() > 1.0e-5);
    }
}
