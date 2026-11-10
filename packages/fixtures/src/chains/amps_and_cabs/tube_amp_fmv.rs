//! `Chain-04`: Multi-Stage Tube Amp + Coupled `FMV` Tone Stack + `4x12` Cabinet.

#![forbid(unsafe_code)]

use crate::chains::{
    Oversampler4xF64, scale_alpha_from_192k, scale_modal_pole_from_192k, substep_rate_4x,
};
use crate::{
    BlackBoxModel, DEFAULT_FIXTURE_SAMPLE_RATE_HZ, FixtureBitDepth, FixtureControlSpec,
    FixtureDeviceMetadata, FixtureSwitchSpec, clamp_sample_rate_hz, sanitize_norm,
};

const CHAIN04_METADATA: FixtureDeviceMetadata = FixtureDeviceMetadata {
    id: "Chain-04-Tube-Amp-FMV-4x12",
    name: "3-Triode Tube Amp + Coupled FMV + 4x12 Cab",
    category: "tube_amp",
    sample_rate_hz: DEFAULT_FIXTURE_SAMPLE_RATE_HZ,
    bit_depth: FixtureBitDepth::Float32,
    expected_tail_ms: 120.0,
    supports_tempo_sync: false,
    controls: &[
        FixtureControlSpec {
            id: "gain",
            name: "Gain",
            unit: "%",
            default_norm: 0.0,
            min_physical: 0.0,
            max_physical: 100.0,
        },
        FixtureControlSpec {
            id: "bass",
            name: "Bass",
            unit: "%",
            default_norm: 0.0,
            min_physical: 0.0,
            max_physical: 100.0,
        },
        FixtureControlSpec {
            id: "mid",
            name: "Middle",
            unit: "%",
            default_norm: 0.0,
            min_physical: 0.0,
            max_physical: 100.0,
        },
        FixtureControlSpec {
            id: "treble",
            name: "Treble",
            unit: "%",
            default_norm: 0.0,
            min_physical: 0.0,
            max_physical: 100.0,
        },
    ],
    switches: &[FixtureSwitchSpec {
        id: "bright",
        name: "Bright Cap",
        options: &["Off", "On"],
        default_index: 0,
    }],
};

/// `Chain-04-Tube-Amp-FMV-4x12`: `1 -> 1` 3-triode preamp (`V1A -> RC bias -> V1B`) ->
/// coupled 4-knob `FMV` tone stack (`Gain`, `Bass`, `Mid`, `Treble`, `Bright` switch) -> push-pull
/// power stage with multi-second screen/supply sag -> dual-mic (`SM57 + R121`) `4x12` cabinet.
#[derive(Clone, Debug)]
pub struct Chain04TubeAmpFmv4x12 {
    sample_rate_hz: u32,
    bit_depth: FixtureBitDepth,
    os: Oversampler4xF64,
    v1a_bias_rc: f64,
    interstage_lp: f64,
    interstage_hp: f64,
    fmv_low_state: f64,
    fmv_mid_state: f64,
    fmv_high_state: f64,
    power_sag_env: f64,
    cab_modal_re: [f64; 4],
    cab_modal_im: [f64; 4],
    cab_cone_lp: f64,
    cab_dc_x1: f64,
    cab_dc_y1: f64,
}

impl Default for Chain04TubeAmpFmv4x12 {
    fn default() -> Self {
        Self::new()
    }
}

impl Chain04TubeAmpFmv4x12 {
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
            v1a_bias_rc: 0.0,
            interstage_lp: 0.0,
            interstage_hp: 0.0,
            fmv_low_state: 0.0,
            fmv_mid_state: 0.0,
            fmv_high_state: 0.0,
            power_sag_env: 0.0,
            cab_modal_re: [0.0; 4],
            cab_modal_im: [0.0; 4],
            cab_cone_lp: 0.0,
            cab_dc_x1: 0.0,
            cab_dc_y1: 0.0,
        }
    }

    #[inline]
    pub fn step_sample(&mut self, x: f32, controls: &[f32], switches: &[u8]) -> f32 {
        let x_clean = self.bit_depth.quantize_sample(x);
        let gain_u = f64::from(sanitize_norm(controls.first().copied().unwrap_or(0.0), 0.0));
        let bass_u = f64::from(sanitize_norm(controls.get(1).copied().unwrap_or(0.0), 0.0));
        let mid_u = f64::from(sanitize_norm(controls.get(2).copied().unwrap_or(0.0), 0.0));
        let treb_u = f64::from(sanitize_norm(controls.get(3).copied().unwrap_or(0.0), 0.0));
        let bright_on = switches.first().copied().unwrap_or_else(|| {
            if controls.get(4).copied().unwrap_or(0.0) > 0.5 {
                1
            } else {
                0
            }
        }) != 0;

        let fs_4x = substep_rate_4x(self.sample_rate_hz);
        let a_grid_charge = scale_alpha_from_192k(0.008, fs_4x);
        let a_grid_bleed = scale_alpha_from_192k(0.0004, fs_4x);
        let a_hp = scale_alpha_from_192k(0.004, fs_4x);
        let a_lp = scale_alpha_from_192k(0.14, fs_4x);
        let a_fmv_lo = scale_alpha_from_192k(0.006, fs_4x);
        let a_fmv_md = scale_alpha_from_192k(0.032, fs_4x);
        let a_fmv_hi = scale_alpha_from_192k(0.115, fs_4x);
        let a_sag_atk = scale_alpha_from_192k(0.0015, fs_4x);
        let a_sag_rel = scale_alpha_from_192k(4.0e-6, fs_4x);
        let a_cone = scale_alpha_from_192k(0.165, fs_4x);
        let dc_pole = (0.9992_f64).powf(192_000.0 / fs_4x);

        const RADII_192: [f64; 4] = [0.9965, 0.9920, 0.9840, 0.9750];
        const FREQS_192: [f64; 4] = [0.0036, 0.0150, 0.0520, 0.0880];
        const GAINS: [f64; 4] = [0.22, -0.16, 0.30, -0.14];
        let mut poles = [(0.0_f64, 0.0_f64, 0.0_f64); 4];
        for m in 0..4 {
            let (r, w) = scale_modal_pole_from_192k(RADII_192[m], FREQS_192[m], fs_4x);
            let (s, c) = w.sin_cos();
            poles[m] = (r, s, c);
        }

        let g1 = 2.5 + 8.0 * (0.5 * (gain_u + 1.0));
        let bright_boost = if bright_on {
            0.35 * (1.0 - 0.5 * (gain_u + 1.0))
        } else {
            0.0
        };

        let b_norm = 0.5 * (bass_u + 1.0);
        let m_norm = 0.5 * (mid_u + 1.0);
        let t_norm = 0.5 * (treb_u + 1.0);
        let z_load = 1.0 + 0.45 * b_norm + 0.55 * m_norm + 0.35 * t_norm;
        let w_b = (0.3 + 0.9 * b_norm) / z_load;
        let w_m = (0.2 + 0.8 * m_norm) / z_load;
        let w_t = (0.3 + 1.1 * t_norm + bright_boost) / z_load;
        let v1b_quiescent = (-0.12_f64).tanh();

        let [y_cab, _] = self
            .os
            .process_frame_4x(f64::from(x_clean), 0.0, |_, u, _| {
                let v1a_in = g1 * u - 0.35 * self.v1a_bias_rc;
                let v1a = (v1a_in + 0.18 * v1a_in * v1a_in).tanh();
                let grid_cond = (v1a_in - 0.4).max(0.0);
                self.v1a_bias_rc += a_grid_charge * grid_cond - a_grid_bleed * self.v1a_bias_rc;

                self.interstage_hp += a_hp * (v1a - self.interstage_hp);
                let coupled = v1a - self.interstage_hp;
                self.interstage_lp += a_lp * (coupled - self.interstage_lp);

                // AC-coupled cathode follower / plate stage at exact 0.0 quiescent DC equilibrium
                let v1b_out = (2.8 * self.interstage_lp - 0.12).tanh() - v1b_quiescent;

                self.fmv_low_state += a_fmv_lo * (v1b_out - self.fmv_low_state);
                self.fmv_mid_state += a_fmv_md * (v1b_out - self.fmv_mid_state);
                self.fmv_high_state += a_fmv_hi * (v1b_out - self.fmv_high_state);
                let fmv_lo = self.fmv_low_state;
                let fmv_md = 0.65 * (self.fmv_mid_state - self.fmv_low_state)
                    + 0.35 * (self.fmv_high_state - self.fmv_mid_state);
                let fmv_hi = v1b_out - self.fmv_high_state;
                let fmv_sig = w_b * fmv_lo + w_m * fmv_md + w_t * fmv_hi;

                let pwr_drive = 2.2 * fmv_sig;
                let pwr_env = pwr_drive.abs();
                // Fast rectifier droop attack (~3.5 ms) + multi-second filter-cap recovery (~1.3 s)
                let sag_alpha = if pwr_env > self.power_sag_env {
                    a_sag_atk
                } else {
                    a_sag_rel
                };
                self.power_sag_env += sag_alpha * (pwr_env - self.power_sag_env);
                let headroom = (1.0 / (1.0 + 0.45 * self.power_sag_env)).clamp(0.45, 1.0);
                let pwr_out = headroom * (pwr_drive / headroom).tanh();

                self.cab_cone_lp += a_cone * (pwr_out - self.cab_cone_lp);
                let mut cab_sum = 0.25 * self.cab_cone_lp;
                for m in 0..4 {
                    let (r, s, c) = poles[m];
                    let nr = r * (c * self.cab_modal_re[m] - s * self.cab_modal_im[m])
                        + self.cab_cone_lp;
                    let ni = r * (s * self.cab_modal_re[m] + c * self.cab_modal_im[m]);
                    self.cab_modal_re[m] = nr;
                    self.cab_modal_im[m] = ni;
                    cab_sum += (1.0 - r) * GAINS[m] * nr;
                }
                let cab_ac = cab_sum - self.cab_dc_x1 + dc_pole * self.cab_dc_y1;
                self.cab_dc_x1 = cab_sum;
                self.cab_dc_y1 = cab_ac;
                [cab_ac * 0.48, 0.0]
            });
        self.bit_depth.quantize_f64_to_f32(y_cab)
    }
}

impl BlackBoxModel for Chain04TubeAmpFmv4x12 {
    fn reset(&mut self) {
        self.os.reset();
        self.v1a_bias_rc = 0.0;
        self.interstage_lp = 0.0;
        self.interstage_hp = 0.0;
        self.fmv_low_state = 0.0;
        self.fmv_mid_state = 0.0;
        self.fmv_high_state = 0.0;
        self.power_sag_env = 0.0;
        self.cab_modal_re = [0.0; 4];
        self.cab_modal_im = [0.0; 4];
        self.cab_cone_lp = 0.0;
        self.cab_dc_x1 = 0.0;
        self.cab_dc_y1 = 0.0;
    }

    fn metadata(&self) -> FixtureDeviceMetadata {
        FixtureDeviceMetadata {
            sample_rate_hz: self.sample_rate_hz,
            bit_depth: self.bit_depth,
            ..CHAIN04_METADATA
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
    fn test_chain04_tube_amp_fmv_zero_dc_equilibrium_and_bright_switch() {
        let mut c4 = Chain04TubeAmpFmv4x12::new();
        assert_eq!(c4.metadata().id, "Chain-04-Tube-Amp-FMV-4x12");
        assert_eq!(
            (
                c4.num_inputs(),
                c4.num_outputs(),
                c4.num_controls(),
                c4.num_switches()
            ),
            (1, 1, 4, 1)
        );
        // Zero input must produce exact 0.0 output without resting DC transient
        let silence = vec![0.0_f32; 64];
        let y_silent = run_mono_block(&mut c4, &silence, &[0.0, 0.0, 0.0, 0.0], &[0]);
        assert!(y_silent.iter().all(|&s| s == 0.0));

        let dry = sine_wave(128, 220.0, 0.45);
        let y_off = run_mono_block(&mut c4, &dry, &[-0.3, 0.1, 0.2, 0.3], &[0]);
        let y_on = run_mono_block(&mut c4, &dry, &[-0.3, 0.1, 0.2, 0.3], &[1]);
        let y_ctrl_bright = run_mono_block(&mut c4, &dry, &[-0.3, 0.1, 0.2, 0.3, 0.9], &[]);
        assert_ne!(y_off, y_on);
        assert_eq!(y_on, y_ctrl_bright);
    }
}
