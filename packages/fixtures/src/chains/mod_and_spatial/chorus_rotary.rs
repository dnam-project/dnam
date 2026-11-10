//! `Chain-07`: Stereo Chorus & Rotary Speaker (`2 -> 2`, 12 Tempo Subdivisions & Rotor Inertia).

#![forbid(unsafe_code)]

use crate::chains::{Oversampler4xF64, substep_rate_4x};
use crate::{
    BlackBoxModel, DEFAULT_FIXTURE_SAMPLE_RATE_HZ, FixtureBitDepth, FixtureControlSpec,
    FixtureDeviceMetadata, FixtureSwitchSpec, clamp_sample_rate_hz, sanitize_norm,
};
use std::f64::consts::PI;

/// Canonical 12 rhythmic subdivision period ratios `(a_s, b_s)` in quarter-note beats.
pub const CANONICAL_SUBDIVISION_RATIOS: [(u32, u32); 12] = [
    (3, 1), // 0: 1/2.
    (2, 1), // 1: 1/2
    (4, 3), // 2: 1/2T
    (3, 2), // 3: 1/4.
    (1, 1), // 4: 1/4
    (2, 3), // 5: 1/4T
    (3, 4), // 6: 1/8.
    (1, 2), // 7: 1/8
    (1, 3), // 8: 1/8T
    (3, 8), // 9: 1/16.
    (1, 4), // 10: 1/16
    (1, 6), // 11: 1/16T
];

const CHAIN07_METADATA: FixtureDeviceMetadata = FixtureDeviceMetadata {
    id: "Chain-07-Stereo-Chorus-And-Rotary",
    name: "Stereo Chorus & Rotary Speaker",
    category: "modulation",
    sample_rate_hz: DEFAULT_FIXTURE_SAMPLE_RATE_HZ,
    bit_depth: FixtureBitDepth::Float32,
    expected_tail_ms: 50.0,
    supports_tempo_sync: true,
    controls: &[
        FixtureControlSpec {
            id: "rate",
            name: "Rate",
            unit: "Hz",
            default_norm: 0.0,
            min_physical: 0.5,
            max_physical: 7.6,
        },
        FixtureControlSpec {
            id: "depth",
            name: "Depth",
            unit: "%",
            default_norm: 0.0,
            min_physical: 5.0,
            max_physical: 95.0,
        },
        FixtureControlSpec {
            id: "mix",
            name: "Mix",
            unit: "%",
            default_norm: 0.0,
            min_physical: 0.0,
            max_physical: 100.0,
        },
    ],
    switches: &[FixtureSwitchSpec {
        id: "speed",
        name: "Rotor Speed",
        options: &["Chorale (Slow)", "Tremolo (Fast)"],
        default_index: 0,
    }],
};

const CHORUS_RING_LEN: usize = 16_384;

/// `Chain-07-Stereo-Chorus-And-Rotary`: `2 -> 2` stereo chorus (`~8.5 ms` base delay with independent
/// polyrhythmic tempo-synced `phase_chorus_l` / `phase_chorus_r`) cascaded into a Leslie rotary horn
/// stage (`phase_horn`) with asymmetric mechanical inertia (`tau_accel = 1.2s`, `tau_decel = 4.0s`).
#[derive(Clone, Debug)]
pub struct Chain07StereoChorusAndRotary {
    sample_rate_hz: u32,
    bit_depth: FixtureBitDepth,
    os: Oversampler4xF64,
    delay_l: Vec<f64>,
    delay_r: Vec<f64>,
    wr_idx: usize,
    phase_chorus_l: f64,
    phase_chorus_r: f64,
    phase_horn: f64,
    current_horn_hz: f64,
    bpm: f64,
    subdiv_l: usize,
    subdiv_r: usize,
}

impl Default for Chain07StereoChorusAndRotary {
    fn default() -> Self {
        Self::new()
    }
}

impl Chain07StereoChorusAndRotary {
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
            delay_l: vec![0.0; CHORUS_RING_LEN],
            delay_r: vec![0.0; CHORUS_RING_LEN],
            wr_idx: 0,
            phase_chorus_l: 0.0,
            phase_chorus_r: 0.25,
            phase_horn: 0.0,
            current_horn_hz: 0.8,
            bpm: 120.0,
            subdiv_l: 4, // 1/4
            subdiv_r: 5, // 1/4T polyrhythmic
        }
    }

    /// Sets host tempo (`20..=400 BPM`) and L/R subdivision indices (`0..12`).
    pub fn set_tempo_and_subdivisions(&mut self, bpm: f64, subdiv_l: usize, subdiv_r: usize) {
        if bpm.is_finite() {
            self.bpm = bpm.clamp(20.0, 400.0);
        }
        self.subdiv_l = subdiv_l % 12;
        self.subdiv_r = subdiv_r % 12;
    }

    /// Returns the instantaneous mechanical horn rotation frequency in Hz.
    #[must_use]
    pub fn current_horn_hz(&self) -> f32 {
        self.current_horn_hz as f32
    }

    /// Computes exact LFO frequency in Hz for `(bpm, subdivision_idx)` across all 12 subdivisions.
    #[must_use]
    pub fn lfo_hz_for_subdivision(bpm: f64, subdiv_idx: usize) -> f64 {
        let clamped_bpm = if bpm.is_finite() {
            bpm.clamp(20.0, 400.0)
        } else {
            120.0
        };
        let (num, den) = CANONICAL_SUBDIVISION_RATIOS[subdiv_idx % 12];
        let ratio = f64::from(num) / f64::from(den.max(1));
        (clamped_bpm / 60.0) / ratio
    }

    #[inline]
    pub fn step_stereo(
        &mut self,
        in_l: f32,
        in_r: f32,
        controls: &[f32],
        switches: &[u8],
    ) -> [f32; 2] {
        let xl = self.bit_depth.quantize_sample(in_l);
        let xr = self.bit_depth.quantize_sample(in_r);
        let rate_u = f64::from(sanitize_norm(controls.first().copied().unwrap_or(0.0), 0.0));
        let depth_u = f64::from(sanitize_norm(controls.get(1).copied().unwrap_or(0.0), 0.0));
        let mix_u = f64::from(sanitize_norm(controls.get(2).copied().unwrap_or(0.0), 0.0));
        let fast_tremolo = switches.first().copied().unwrap_or(0) != 0;

        let fs_4x = substep_rate_4x(self.sample_rate_hz);
        let scale = fs_4x / 192_000.0;

        let target_horn_hz = if fast_tremolo {
            6.4 + 1.2 * rate_u
        } else {
            0.8 + 0.3 * rate_u
        };
        let alpha_inertia = if target_horn_hz > self.current_horn_hz {
            1.0 / (1.2 * fs_4x)
        } else {
            1.0 / (4.0 * fs_4x)
        };

        let f_l = Self::lfo_hz_for_subdivision(self.bpm, self.subdiv_l);
        let f_r = Self::lfo_hz_for_subdivision(self.bpm, self.subdiv_r);
        let base_dly = (1632.0 * scale).min((CHORUS_RING_LEN as f64) * 0.65);
        let chorus_depth = 280.0 * scale * (0.5 + 0.45 * depth_u);
        let rotary_doppler = 65.0 * scale * (0.5 + 0.45 * depth_u);
        let wet_mix = 0.5 * (mix_u + 1.0);

        let [yl, yr] = self
            .os
            .process_frame_4x(f64::from(xl), f64::from(xr), |_, ul, ur| {
                self.current_horn_hz += alpha_inertia * (target_horn_hz - self.current_horn_hz);
                self.phase_chorus_l = (self.phase_chorus_l + f_l / fs_4x).fract();
                self.phase_chorus_r = (self.phase_chorus_r + f_r / fs_4x).fract();
                self.phase_horn = (self.phase_horn + self.current_horn_hz / fs_4x).fract();

                let mask = CHORUS_RING_LEN - 1;
                self.delay_l[self.wr_idx] = ul;
                self.delay_r[self.wr_idx] = ur;

                let horn_sin = (2.0 * PI * self.phase_horn).sin();
                let horn_cos = (2.0 * PI * self.phase_horn).cos();
                let d_l = (base_dly
                    + chorus_depth * (2.0 * PI * self.phase_chorus_l).sin()
                    + rotary_doppler * horn_sin)
                    .clamp(2.0, (CHORUS_RING_LEN - 4) as f64);
                let d_r = (base_dly
                    + chorus_depth * (2.0 * PI * self.phase_chorus_r).cos()
                    + rotary_doppler * horn_cos)
                    .clamp(2.0, (CHORUS_RING_LEN - 4) as f64);

                let idx_l = (self.wr_idx as f64 - d_l).rem_euclid(CHORUS_RING_LEN as f64);
                let i0_l = idx_l.floor() as usize;
                let frac_l = idx_l - (i0_l as f64);
                let wet_l = (1.0 - frac_l) * self.delay_l[i0_l & mask]
                    + frac_l * self.delay_l[(i0_l + 1) & mask];

                let idx_r = (self.wr_idx as f64 - d_r).rem_euclid(CHORUS_RING_LEN as f64);
                let i0_r = idx_r.floor() as usize;
                let frac_r = idx_r - (i0_r as f64);
                let wet_r = (1.0 - frac_r) * self.delay_r[i0_r & mask]
                    + frac_r * self.delay_r[(i0_r + 1) & mask];

                self.wr_idx = (self.wr_idx + 1) & mask;

                let am_l = 1.0 + 0.22 * horn_cos;
                let am_r = 1.0 + 0.22 * horn_sin;
                [
                    ((1.0 - 0.55 * wet_mix) * ul + 0.55 * wet_mix * wet_l) * am_l,
                    ((1.0 - 0.55 * wet_mix) * ur + 0.55 * wet_mix * wet_r) * am_r,
                ]
            });
        [
            self.bit_depth.quantize_f64_to_f32(yl),
            self.bit_depth.quantize_f64_to_f32(yr),
        ]
    }
}

impl BlackBoxModel for Chain07StereoChorusAndRotary {
    fn reset(&mut self) {
        self.os.reset();
        self.delay_l.fill(0.0);
        self.delay_r.fill(0.0);
        self.wr_idx = 0;
        self.phase_chorus_l = 0.0;
        self.phase_chorus_r = 0.25;
        self.phase_horn = 0.0;
        self.current_horn_hz = 0.8;
    }

    fn metadata(&self) -> FixtureDeviceMetadata {
        FixtureDeviceMetadata {
            sample_rate_hz: self.sample_rate_hz,
            bit_depth: self.bit_depth,
            ..CHAIN07_METADATA
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
        2
    }

    fn process_frame(
        &mut self,
        inputs: &[f32],
        outputs: &mut [f32],
        controls: &[f32],
        switches: &[u8],
    ) {
        let xl = inputs.first().copied().unwrap_or(0.0);
        let xr = inputs.get(1).copied().unwrap_or(xl);
        let [yl, yr] = self.step_stereo(xl, xr, controls, switches);
        crate::chains::write_stereo_output(outputs, yl, yr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{run_stereo_block, sine_wave};

    #[test]
    fn test_chain07_chorus_rotary_tempo_subdivisions_and_inertia() {
        let mut c7 = Chain07StereoChorusAndRotary::default();
        assert_eq!(c7.metadata().id, "Chain-07-Stereo-Chorus-And-Rotary");
        assert!(c7.metadata().supports_tempo_sync);
        assert_eq!(
            (
                c7.num_inputs(),
                c7.num_outputs(),
                c7.num_controls(),
                c7.num_switches()
            ),
            (2, 2, 3, 1)
        );
        let f_q = Chain07StereoChorusAndRotary::lfo_hz_for_subdivision(120.0, 4);
        let f_qt = Chain07StereoChorusAndRotary::lfo_hz_for_subdivision(120.0, 5);
        assert!((f_qt / f_q - 1.5).abs() < 1.0e-12);

        let dry_l = sine_wave(1_024, 220.0, 0.45);
        let dry_r = sine_wave(1_024, 330.0, 0.45);
        c7.set_tempo_and_subdivisions(128.0, 3, 6);
        let (y_l, y_r) = run_stereo_block(&mut c7, &dry_l, &dry_r, &[0.2, 0.7, 0.8], &[1]);
        let fast_horn = c7.current_horn_hz();
        let (y_slow_l, _) = run_stereo_block(&mut c7, &dry_l, &dry_r, &[0.2, 0.7, 0.8], &[0]);
        assert!(fast_horn > 0.8);
        assert_ne!(y_l, y_r);
        assert_ne!(y_l, y_slow_l);
    }
}
