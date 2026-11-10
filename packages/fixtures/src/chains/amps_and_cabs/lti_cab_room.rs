//! `Chain-05`: Pure LTI Speaker Cabinet + Room Reflections (`1 -> 2` Stereo).

#![forbid(unsafe_code)]

use crate::chains::{Oversampler4xF64, scale_modal_pole_from_192k, substep_rate_4x};
use crate::{
    BlackBoxModel, DEFAULT_FIXTURE_SAMPLE_RATE_HZ, FixtureBitDepth, FixtureDeviceMetadata,
    clamp_sample_rate_hz,
};
use std::f64::consts::PI;

const CHAIN05_METADATA: FixtureDeviceMetadata = FixtureDeviceMetadata {
    id: "Chain-05-Pure-LTI-Cab-And-Room",
    name: "Pure LTI Speaker Cabinet + Room Reflections",
    category: "cabinet",
    sample_rate_hz: DEFAULT_FIXTURE_SAMPLE_RATE_HZ,
    bit_depth: FixtureBitDepth::Float32,
    expected_tail_ms: 85.0,
    supports_tempo_sync: false,
    controls: &[],
    switches: &[],
};

const ROOM_RING_LEN: usize = 8_192;

/// `Chain-05-Pure-LTI-Cab-And-Room`: `1 -> 2` stereo LTI cabinet (`55 Hz` bass-reflex port +
/// `1.2–4.5 kHz` cone breakup modes + `1.85–7.85 ms` stereo early room reflections).
#[derive(Clone, Debug)]
pub struct Chain05PureLtiCabAndRoom {
    sample_rate_hz: u32,
    bit_depth: FixtureBitDepth,
    os: Oversampler4xF64,
    modal_re: [f64; 6],
    modal_im: [f64; 6],
    room_delay_l: Vec<f64>,
    room_delay_r: Vec<f64>,
    wr_idx: usize,
}

impl Default for Chain05PureLtiCabAndRoom {
    fn default() -> Self {
        Self::new()
    }
}

impl Chain05PureLtiCabAndRoom {
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
            modal_re: [0.0; 6],
            modal_im: [0.0; 6],
            room_delay_l: vec![0.0; ROOM_RING_LEN],
            room_delay_r: vec![0.0; ROOM_RING_LEN],
            wr_idx: 0,
        }
    }

    #[inline]
    pub fn step_sample(&mut self, x: f32) -> [f32; 2] {
        const RADII_192: [f64; 6] = [0.9985, 0.9940, 0.9890, 0.9850, 0.9810, 0.9760];
        const FREQS_HZ: [f64; 6] = [55.0, 140.0, 1250.0, 2150.0, 3300.0, 4450.0];
        const W_L: [f64; 6] = [0.35, 0.20, 0.28, -0.18, 0.22, -0.15];
        const W_R: [f64; 6] = [0.35, 0.18, -0.22, 0.25, -0.19, 0.17];

        let fs_4x = substep_rate_4x(self.sample_rate_hz);
        let scale = fs_4x / 192_000.0;
        // Studio early reflections: L = 1.854 ms & 4.854 ms, R = 3.000 ms & 7.854 ms
        let tap_l1 = ((356.0 * scale).round() as usize).clamp(1, ROOM_RING_LEN - 2);
        let tap_l2 = ((932.0 * scale).round() as usize).clamp(2, ROOM_RING_LEN - 2);
        let tap_r1 = ((576.0 * scale).round() as usize).clamp(1, ROOM_RING_LEN - 2);
        let tap_r2 = ((1508.0 * scale).round() as usize).clamp(2, ROOM_RING_LEN - 2);

        let mut poles = [(0.0_f64, 0.0_f64, 0.0_f64); 6];
        for m in 0..6 {
            let omega_192 = 2.0 * PI * FREQS_HZ[m] / 192_000.0;
            let (r, omega) = scale_modal_pole_from_192k(RADII_192[m], omega_192, fs_4x);
            let (s, c) = omega.sin_cos();
            poles[m] = (r, s, c);
        }

        let x_clean = self.bit_depth.quantize_sample(x);
        let [yl, yr] = self
            .os
            .process_frame_4x(f64::from(x_clean), 0.0, |_, u, _| {
                let mut acc_l = 0.20 * u;
                let mut acc_r = 0.20 * u;
                for m in 0..6 {
                    let (r, s, c) = poles[m];
                    let nr = r * (c * self.modal_re[m] - s * self.modal_im[m]) + u;
                    let ni = r * (s * self.modal_re[m] + c * self.modal_im[m]);
                    self.modal_re[m] = nr;
                    self.modal_im[m] = ni;
                    let norm_nr = (1.0 - r) * nr;
                    acc_l += W_L[m] * norm_nr;
                    acc_r += W_R[m] * norm_nr;
                }
                let mask = ROOM_RING_LEN - 1;
                let early_l = self.room_delay_l[(self.wr_idx.wrapping_sub(tap_l1)) & mask] * 0.24
                    - self.room_delay_l[(self.wr_idx.wrapping_sub(tap_l2)) & mask] * 0.14;
                let early_r = self.room_delay_r[(self.wr_idx.wrapping_sub(tap_r1)) & mask] * 0.24
                    + self.room_delay_r[(self.wr_idx.wrapping_sub(tap_r2)) & mask] * 0.14;
                self.room_delay_l[self.wr_idx] = acc_l;
                self.room_delay_r[self.wr_idx] = acc_r;
                self.wr_idx = (self.wr_idx + 1) & mask;
                [acc_l + early_l, acc_r + early_r]
            });
        [
            self.bit_depth.quantize_f64_to_f32(yl),
            self.bit_depth.quantize_f64_to_f32(yr),
        ]
    }
}

impl BlackBoxModel for Chain05PureLtiCabAndRoom {
    fn reset(&mut self) {
        self.os.reset();
        self.modal_re = [0.0; 6];
        self.modal_im = [0.0; 6];
        self.room_delay_l.fill(0.0);
        self.room_delay_r.fill(0.0);
        self.wr_idx = 0;
    }

    fn metadata(&self) -> FixtureDeviceMetadata {
        FixtureDeviceMetadata {
            sample_rate_hz: self.sample_rate_hz,
            bit_depth: self.bit_depth,
            ..CHAIN05_METADATA
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
        2
    }

    fn process_frame(
        &mut self,
        inputs: &[f32],
        outputs: &mut [f32],
        _controls: &[f32],
        _switches: &[u8],
    ) {
        let x = inputs.first().copied().unwrap_or(0.0);
        let [l, r] = self.step_sample(x);
        crate::chains::write_stereo_output(outputs, l, r);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::sine_wave;

    #[test]
    fn test_chain05_lti_cab_and_room_superposition_and_stereo_spread() {
        let mut c5 = Chain05PureLtiCabAndRoom::default();
        assert_eq!(c5.metadata().id, "Chain-05-Pure-LTI-Cab-And-Room");
        assert_eq!(
            (
                c5.num_inputs(),
                c5.num_outputs(),
                c5.num_controls(),
                c5.num_switches()
            ),
            (1, 2, 0, 0)
        );
        let x1 = sine_wave(512, 220.0, 0.40);
        let x2 = sine_wave(512, 660.0, 0.25);
        let a = 0.65_f32;
        let b = -0.35_f32;
        let x_comb: Vec<f32> = x1.iter().zip(&x2).map(|(&u, &v)| a * u + b * v).collect();

        let mut y1_l = vec![0.0_f32; 512];
        let mut y1_r = vec![0.0_f32; 512];
        c5.reset();
        c5.process_block(&[&x1], &mut [&mut y1_l, &mut y1_r], &[], &[]);

        let mut y2_l = vec![0.0_f32; 512];
        let mut y2_r = vec![0.0_f32; 512];
        c5.reset();
        c5.process_block(&[&x2], &mut [&mut y2_l, &mut y2_r], &[], &[]);

        let mut yc_l = vec![0.0_f32; 512];
        let mut yc_r = vec![0.0_f32; 512];
        c5.reset();
        c5.process_block(&[&x_comb], &mut [&mut yc_l, &mut yc_r], &[], &[]);

        assert_ne!(yc_l, yc_r);
        for i in 0..512 {
            assert!((yc_l[i] - (a * y1_l[i] + b * y2_l[i])).abs() < 1.0e-6);
            assert!((yc_r[i] - (a * y1_r[i] + b * y2_r[i])).abs() < 1.0e-6);
        }
    }
}
