//! `Chain-06`: Envelope-Modulated Resonant SVF (`LP`/`BP`/`HP` 3-Way Switch, `Q <= 12`).

#![forbid(unsafe_code)]

use crate::chains::{Oversampler4xF64, scale_alpha_from_192k, substep_rate_4x};
use crate::{
    BlackBoxModel, DEFAULT_FIXTURE_SAMPLE_RATE_HZ, FixtureBitDepth, FixtureControlSpec,
    FixtureDeviceMetadata, FixtureSwitchSpec, clamp_sample_rate_hz, sanitize_norm,
};
use std::f64::consts::PI;

const CHAIN06_METADATA: FixtureDeviceMetadata = FixtureDeviceMetadata {
    id: "Chain-06-Envelope-Resonant-SVF",
    name: "Envelope-Modulated Resonant SVF",
    category: "envelope_filter",
    sample_rate_hz: DEFAULT_FIXTURE_SAMPLE_RATE_HZ,
    bit_depth: FixtureBitDepth::Float32,
    expected_tail_ms: 45.0,
    supports_tempo_sync: false,
    controls: &[
        FixtureControlSpec {
            id: "sensitivity",
            name: "Sensitivity",
            unit: "%",
            default_norm: 0.0,
            min_physical: 0.0,
            max_physical: 100.0,
        },
        FixtureControlSpec {
            id: "resonance",
            name: "Resonance (Q)",
            unit: "",
            default_norm: 0.0,
            min_physical: 1.5,
            max_physical: 12.0,
        },
    ],
    switches: &[FixtureSwitchSpec {
        id: "mode",
        name: "Filter Mode",
        options: &["LP", "BP", "HP"],
        default_index: 1,
    }],
};

/// `Chain-06-Envelope-Resonant-SVF`: `1 -> 1` envelope-followed resonant TPT SVF (`Q <= 12`)
/// ahead of soft saturation (`Sensitivity`, `Resonance`, `Mode` `0=LP, 1=BP, 2=HP`).
#[derive(Clone, Debug)]
pub struct Chain06EnvelopeResonantSvf {
    sample_rate_hz: u32,
    bit_depth: FixtureBitDepth,
    os: Oversampler4xF64,
    env_state: f64,
    ic1eq: f64,
    ic2eq: f64,
}

impl Default for Chain06EnvelopeResonantSvf {
    fn default() -> Self {
        Self::new()
    }
}

impl Chain06EnvelopeResonantSvf {
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
            ic1eq: 0.0,
            ic2eq: 0.0,
        }
    }

    #[inline]
    pub fn step_sample(&mut self, x: f32, controls: &[f32], switches: &[u8]) -> f32 {
        let x_clean = self.bit_depth.quantize_sample(x);
        let sens_u = f64::from(sanitize_norm(controls.first().copied().unwrap_or(0.0), 0.0));
        let res_u = f64::from(sanitize_norm(controls.get(1).copied().unwrap_or(0.0), 0.0));
        let mode = switches.first().copied().unwrap_or(1) % 3;

        let fs_4x = substep_rate_4x(self.sample_rate_hz);
        let a_atk = scale_alpha_from_192k(0.008, fs_4x);
        let a_rel = scale_alpha_from_192k(0.0003, fs_4x);
        let max_fc = (0.42 * f64::from(self.sample_rate_hz)).min(4200.0);

        let sens = 1.5 + 3.5 * (0.5 * (sens_u + 1.0));
        let q = (1.5 + 10.5 * (0.5 * (res_u + 1.0))).clamp(0.707, 12.0);
        let k_damp = 1.0 / q;

        let [y, _] = self
            .os
            .process_frame_4x(f64::from(x_clean), 0.0, |_, u, _| {
                let rect = (u * sens).abs();
                let alpha = if rect > self.env_state { a_atk } else { a_rel };
                self.env_state += alpha * (rect - self.env_state);
                let fc = (320.0 * 2.0_f64.powf((2.6 * self.env_state).clamp(0.0, 3.2)))
                    .clamp(250.0, max_fc);

                let g = (PI * fc / fs_4x).tan();
                let a1 = 1.0 / (1.0 + g * (g + k_damp));
                let a2 = g * a1;
                let a3 = g * a2;

                let v3 = u - self.ic2eq;
                let v1 = a1 * self.ic1eq + a2 * v3;
                let v2 = self.ic2eq + a2 * self.ic1eq + a3 * v3;
                self.ic1eq = 2.0 * v1 - self.ic1eq;
                self.ic2eq = 2.0 * v2 - self.ic2eq;

                let lp = v2;
                let bp = v1;
                let hp = u - k_damp * v1 - v2;
                let svf_out = match mode {
                    0 => lp,
                    1 => bp * 2.2,
                    _ => hp,
                };
                [(1.4 * svf_out).tanh() * 0.7, 0.0]
            });
        self.bit_depth.quantize_f64_to_f32(y)
    }
}

impl BlackBoxModel for Chain06EnvelopeResonantSvf {
    fn reset(&mut self) {
        self.os.reset();
        self.env_state = 0.0;
        self.ic1eq = 0.0;
        self.ic2eq = 0.0;
    }

    fn metadata(&self) -> FixtureDeviceMetadata {
        FixtureDeviceMetadata {
            sample_rate_hz: self.sample_rate_hz,
            bit_depth: self.bit_depth,
            ..CHAIN06_METADATA
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
    fn test_chain06_envelope_svf_lp_bp_hp_modes() {
        let mut c6 = Chain06EnvelopeResonantSvf::new();
        assert_eq!(c6.metadata().id, "Chain-06-Envelope-Resonant-SVF");
        assert_eq!(
            (
                c6.num_inputs(),
                c6.num_outputs(),
                c6.num_controls(),
                c6.num_switches()
            ),
            (1, 1, 2, 1)
        );
        let dry = sine_wave(128, 220.0, 0.45);
        let y_lp = run_mono_block(&mut c6, &dry, &[0.5, 0.6], &[0]);
        let y_bp = run_mono_block(&mut c6, &dry, &[0.5, 0.6], &[1]);
        let y_hp = run_mono_block(&mut c6, &dry, &[0.5, 0.6], &[2]);
        assert_ne!(y_lp, y_bp);
        assert_ne!(y_bp, y_hp);
    }
}
