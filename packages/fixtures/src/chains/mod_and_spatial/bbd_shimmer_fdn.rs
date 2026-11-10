//! `Chain-08`: Long BBD Echo + `30s` Shimmer FDN + Timed Freeze Latch.

#![forbid(unsafe_code)]

use crate::chains::{Oversampler4xF64, scale_alpha_from_192k, substep_rate_4x};
use crate::{
    BlackBoxModel, DEFAULT_FIXTURE_SAMPLE_RATE_HZ, FixtureBitDepth, FixtureControlSpec,
    FixtureDeviceMetadata, FixtureSwitchSpec, clamp_sample_rate_hz, sanitize_norm,
};

const CHAIN08_METADATA: FixtureDeviceMetadata = FixtureDeviceMetadata {
    id: "Chain-08-Long-BBD-Echo-And-30s-Shimmer",
    name: "Long BBD Echo + 30s Shimmer FDN + Freeze Latch",
    category: "delay_reverb",
    sample_rate_hz: DEFAULT_FIXTURE_SAMPLE_RATE_HZ,
    bit_depth: FixtureBitDepth::Float32,
    expected_tail_ms: 30_000.0,
    supports_tempo_sync: true,
    controls: &[
        FixtureControlSpec {
            id: "time",
            name: "Delay Time",
            unit: "ms",
            default_norm: 0.0,
            min_physical: 40.0,
            max_physical: 800.0,
        },
        FixtureControlSpec {
            id: "feedback",
            name: "Feedback / Shimmer",
            unit: "%",
            default_norm: 0.0,
            min_physical: 20.0,
            max_physical: 98.5,
        },
        FixtureControlSpec {
            id: "mix",
            name: "Wet Mix",
            unit: "%",
            default_norm: 0.0,
            min_physical: 0.0,
            max_physical: 100.0,
        },
    ],
    switches: &[FixtureSwitchSpec {
        id: "freeze",
        name: "Freeze Latch",
        options: &["Normal", "Frozen"],
        default_index: 0,
    }],
};

const FDN_RING_LEN: usize = 131_072; // Covers 33..88 ms hall lines up to 192 kHz base rate
const FDN_TAPS_192: [usize; 4] = [6_421, 9_199, 12_853, 16_901]; // Coprime 33.4..88.0 ms @ 192 kHz

#[inline]
fn bbd_ring_len_for_rate(sample_rate_hz: u32) -> usize {
    let fs_4x = substep_rate_4x(sample_rate_hz);
    let needed = ((0.800 * fs_4x).ceil() as usize) + 64;
    needed.next_power_of_two().max(65_536)
}

/// `Chain-08-Long-BBD-Echo-And-30s-Shimmer`: `2 -> 2` dispersive BBD echo (`40–800 ms`) + `30s`
/// octave-shimmer Hadamard FDN tank + lossless timed `freeze_latch` window `[n_earliest, n_latest)`.
#[derive(Clone, Debug)]
pub struct Chain08LongBbdEchoAnd30sShimmer {
    sample_rate_hz: u32,
    bit_depth: FixtureBitDepth,
    os: Oversampler4xF64,
    bbd_ring: Vec<f64>,
    fdn_rings: [Vec<f64>; 4],
    wr_idx: usize,
    bbd_lp: f64,
    shimmer_dc: [f64; 4],
    sample_counter: usize,
    freeze_window: Option<(usize, usize)>,
    anchor_freeze_window: Option<(usize, usize, u32)>,
    tempo_bpm: Option<f64>,
}

impl Default for Chain08LongBbdEchoAnd30sShimmer {
    fn default() -> Self {
        Self::new()
    }
}

impl Chain08LongBbdEchoAnd30sShimmer {
    #[must_use]
    pub fn new() -> Self {
        Self::with_format(DEFAULT_FIXTURE_SAMPLE_RATE_HZ, FixtureBitDepth::Float32)
    }

    #[must_use]
    pub fn with_format(sample_rate_hz: u32, bit_depth: FixtureBitDepth) -> Self {
        let sr = clamp_sample_rate_hz(sample_rate_hz);
        Self {
            sample_rate_hz: sr,
            bit_depth,
            os: Oversampler4xF64::default(),
            bbd_ring: vec![0.0; bbd_ring_len_for_rate(sr)],
            fdn_rings: [
                vec![0.0; FDN_RING_LEN],
                vec![0.0; FDN_RING_LEN],
                vec![0.0; FDN_RING_LEN],
                vec![0.0; FDN_RING_LEN],
            ],
            wr_idx: 0,
            bbd_lp: 0.0,
            shimmer_dc: [0.0; 4],
            sample_counter: 0,
            freeze_window: None,
            anchor_freeze_window: None,
            tempo_bpm: None,
        }
    }

    /// Configures a deterministic sample-accurate freeze window `[n_earliest, n_latest)`.
    pub fn set_freeze_window(&mut self, earliest_sample: usize, latest_sample: usize) {
        self.freeze_window = Some((earliest_sample, latest_sample));
        self.anchor_freeze_window = Some((earliest_sample, latest_sample, self.sample_rate_hz));
    }

    /// Configures optional host tempo (`BPM`) quarter-note sync (`40..800 ms`) for the BBD delay line.
    pub fn set_tempo_bpm(&mut self, bpm: Option<f64>) {
        self.tempo_bpm = bpm.filter(|b| b.is_finite()).map(|b| b.clamp(20.0, 400.0));
    }

    #[inline]
    pub fn step_stereo(
        &mut self,
        in_l: f32,
        in_r: f32,
        controls: &[f32],
        freeze_active: bool,
    ) -> [f32; 2] {
        let xl = self.bit_depth.quantize_sample(in_l);
        let xr = self.bit_depth.quantize_sample(in_r);
        let time_u = f64::from(sanitize_norm(controls.first().copied().unwrap_or(0.0), 0.0));
        let fb_u = f64::from(sanitize_norm(controls.get(1).copied().unwrap_or(0.0), 0.0));
        let mix_u = f64::from(sanitize_norm(controls.get(2).copied().unwrap_or(0.0), 0.0));

        let in_window = self
            .freeze_window
            .is_some_and(|(s, e)| self.sample_counter >= s && self.sample_counter < e);
        let is_frozen = freeze_active || in_window;
        self.sample_counter += 1;

        let fs_4x = substep_rate_4x(self.sample_rate_hz);
        let scale = fs_4x / 192_000.0;
        let a_bbd_lp = scale_alpha_from_192k(0.08, fs_4x);
        let a_shim_dc = scale_alpha_from_192k(0.004, fs_4x);

        let mut fdn_taps = [0_usize; 4];
        let mut mean_fdn_delay = 0.0_f64;
        for m in 0..4 {
            let t = (((FDN_TAPS_192[m] as f64) * scale).round() as usize)
                .clamp(31, FDN_RING_LEN - 16)
                | 1;
            fdn_taps[m] = t;
            mean_fdn_delay += (t as f64) * 0.25;
        }

        let fb_01 = 0.5 * (fb_u + 1.0);
        // Calibrate FDN loop gain so RT60 spans 1.5s (fb_u = -1) to 30.0s (fb_u = +1) across all sample rates
        let rt60_sec = 1.5 * 20.0_f64.powf(fb_01);
        let g_fdn = if is_frozen {
            1.0_f64
        } else {
            10.0_f64
                .powf(-3.0 * mean_fdn_delay / (rt60_sec * fs_4x))
                .clamp(0.75, 0.999_8)
        };
        let g_bbd = if is_frozen {
            1.0_f64
        } else {
            0.25 + 0.62 * fb_01
        };
        let shimmer_amt = if is_frozen { 0.0 } else { 0.14 * fb_01 };
        let in_scale = if is_frozen { 0.0 } else { 1.0 };

        let delay_ms = if let Some(bpm) = self.tempo_bpm {
            (60_000.0 / bpm * (0.5 + 0.5 * (time_u + 1.0))).clamp(40.0, 800.0)
        } else {
            40.0 + 760.0 * (0.5 * (time_u + 1.0))
        };
        let bbd_ring_len = self.bbd_ring.len();
        let bbd_delay = ((delay_ms * 1.0e-3 * fs_4x).round() as usize).clamp(16, bbd_ring_len - 16);
        let wet_mix = 0.5 * (mix_u + 1.0);

        let [wet_l, wet_r] = self
            .os
            .process_frame_4x(f64::from(xl), f64::from(xr), |_, ul, ur| {
                let bbd_mask = bbd_ring_len - 1;
                let fdn_mask = FDN_RING_LEN - 1;
                let mono_in = 0.5 * (ul + ur) * in_scale;

                let bbd_rd = self.bbd_ring[(self.wr_idx.wrapping_sub(bbd_delay)) & bbd_mask];
                if is_frozen {
                    self.bbd_ring[self.wr_idx & bbd_mask] = bbd_rd;
                } else {
                    self.bbd_lp += a_bbd_lp * (bbd_rd - self.bbd_lp);
                    self.bbd_ring[self.wr_idx & bbd_mask] = (mono_in + g_bbd * self.bbd_lp).tanh();
                }
                let bbd_tap = if is_frozen { bbd_rd } else { self.bbd_lp };

                let mut r = [0.0_f64; 4];
                for m in 0..4 {
                    r[m] = self.fdn_rings[m][(self.wr_idx.wrapping_sub(fdn_taps[m])) & fdn_mask];
                }
                // Orthogonal 4x4 Householder / Hadamard feedback matrix (||H||_2 == 1.0)
                let s = 0.5 * (r[0] + r[1] + r[2] + r[3]);
                for m in 0..4 {
                    let h = s - r[m];
                    let next_v = if is_frozen {
                        // Strictly lossless unitary circulation when freeze latch is engaged
                        h
                    } else {
                        // Full-wave zero-mean octave-doubled shimmer harmonic injection
                        let oct_raw = h.abs();
                        self.shimmer_dc[m] += a_shim_dc * (oct_raw - self.shimmer_dc[m]);
                        let oct_up = oct_raw - self.shimmer_dc[m];
                        let drive_in = 0.25 * mono_in
                            + 0.18 * bbd_tap
                            + g_fdn * ((1.0 - shimmer_amt) * h + shimmer_amt * oct_up);
                        drive_in.clamp(-4.0, 4.0)
                    };
                    self.fdn_rings[m][self.wr_idx & fdn_mask] = next_v;
                }
                self.wr_idx = self.wr_idx.wrapping_add(1);

                let tank_l = (0.5 * (r[0] - r[2]) + 0.45 * bbd_tap).tanh();
                let tank_r = (0.5 * (r[1] - r[3]) + 0.45 * bbd_tap).tanh();
                [tank_l, tank_r]
            });

        let out_l = (1.0 - wet_mix) * f64::from(xl) + wet_mix * wet_l;
        let out_r = (1.0 - wet_mix) * f64::from(xr) + wet_mix * wet_r;
        [
            self.bit_depth.quantize_f64_to_f32(out_l),
            self.bit_depth.quantize_f64_to_f32(out_r),
        ]
    }
}

impl BlackBoxModel for Chain08LongBbdEchoAnd30sShimmer {
    fn reset(&mut self) {
        self.os.reset();
        self.bbd_ring.fill(0.0);
        for ring in &mut self.fdn_rings {
            ring.fill(0.0);
        }
        self.wr_idx = 0;
        self.bbd_lp = 0.0;
        self.shimmer_dc = [0.0; 4];
        self.sample_counter = 0;
    }

    fn metadata(&self) -> FixtureDeviceMetadata {
        FixtureDeviceMetadata {
            sample_rate_hz: self.sample_rate_hz,
            bit_depth: self.bit_depth,
            ..CHAIN08_METADATA
        }
    }

    fn set_sample_rate_hz(&mut self, sample_rate_hz: u32) {
        let sr = clamp_sample_rate_hz(sample_rate_hz);
        if let Some((a_start, a_end, a_sr)) = self.anchor_freeze_window
            && a_sr > 0
        {
            if sr == a_sr {
                self.freeze_window = Some((a_start, a_end));
            } else {
                let ratio = f64::from(sr) / f64::from(a_sr);
                let scaled_start = ((a_start as f64) * ratio).round() as usize;
                let mut scaled_end = ((a_end as f64) * ratio).round() as usize;
                if a_start < a_end {
                    scaled_end = scaled_end.max(scaled_start.saturating_add(1));
                }
                self.freeze_window = Some((scaled_start, scaled_end));
            }
        }
        self.sample_rate_hz = sr;
        self.bbd_ring.resize(bbd_ring_len_for_rate(sr), 0.0);
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
        let freeze_active = switches.first().copied().unwrap_or(0) != 0;
        let [yl, yr] = self.step_stereo(xl, xr, controls, freeze_active);
        crate::chains::write_stereo_output(outputs, yl, yr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{run_stereo_block, sine_wave};

    #[test]
    fn test_chain08_bbd_40ms_delay_shimmer_fdn_and_lossless_freeze() {
        let mut c8 = Chain08LongBbdEchoAnd30sShimmer::default();
        assert_eq!(c8.metadata().id, "Chain-08-Long-BBD-Echo-And-30s-Shimmer");
        assert!((c8.metadata().expected_tail_ms - 30_000.0).abs() < 1.0e-3);
        assert_eq!(
            (
                c8.num_inputs(),
                c8.num_outputs(),
                c8.num_controls(),
                c8.num_switches()
            ),
            (2, 2, 3, 1)
        );

        let dry_l = sine_wave(4096, 220.0, 0.5);
        let dry_r = sine_wave(4096, 330.0, 0.5);
        let (y_l, y_r) = run_stereo_block(&mut c8, &dry_l, &dry_r, &[-1.0, 0.8, 1.0], &[0]);
        assert!(y_l.iter().chain(&y_r).all(|v| v.is_finite()));

        let silence = vec![0.0_f32; 4096];
        let mut frz_1_l = vec![0.0_f32; 4096];
        let mut frz_1_r = vec![0.0_f32; 4096];
        c8.process_block(
            &[&silence, &silence],
            &mut [&mut frz_1_l, &mut frz_1_r],
            &[-1.0, 0.8, 1.0],
            &[1],
        );
        let mut frz_2_l = vec![0.0_f32; 4096];
        let mut frz_2_r = vec![0.0_f32; 4096];
        c8.process_block(
            &[&silence, &silence],
            &mut [&mut frz_2_l, &mut frz_2_r],
            &[-1.0, 0.8, 1.0],
            &[1],
        );
        let e1: f64 = frz_1_l.iter().map(|&x| f64::from(x * x)).sum();
        let e2: f64 = frz_2_l.iter().map(|&x| f64::from(x * x)).sum();
        assert!(e1 > 1.0e-3 && (e2 / e1) > 0.90);

        c8.set_freeze_window(4_800, 9_600);
        c8.set_sample_rate_hz(96_000);
        assert_eq!(c8.freeze_window, Some((9_600, 19_200)));
        c8.set_sample_rate_hz(48_000);
        c8.set_freeze_window(10, 25);
        c8.set_sample_rate_hz(8_000);
        c8.set_sample_rate_hz(48_000);
        assert_eq!(c8.freeze_window, Some((10, 25)));
        c8.set_freeze_window(1, 2);
        c8.set_sample_rate_hz(8_000);
        let (s8, e8) = c8.freeze_window.unwrap();
        assert!(s8 < e8);
    }
}
