//! `Chain-12`: Full End-to-End Stereo Rig (`Fixed 0-Knob` & `P = 16`, `1->2` & `2->2`).

#![forbid(unsafe_code)]

use crate::chains::{
    Oversampler4xF64, scale_alpha_from_192k, scale_modal_pole_from_192k, substep_rate_4x,
};
use crate::{
    BlackBoxModel, DEFAULT_FIXTURE_SAMPLE_RATE_HZ, FixtureBitDepth, FixtureControlSpec,
    FixtureDeviceMetadata, FixtureSwitchSpec, clamp_sample_rate_hz, sanitize_norm, sanitize_sample,
};
use std::f64::consts::PI;

const CHAIN12_FIXED_METADATA: FixtureDeviceMetadata = FixtureDeviceMetadata {
    id: "Chain-12-Full-EndToEnd-Stereo-Rig-Fixed",
    name: "Full End-to-End Stereo Rig (Fixed 0-Knob Snapshot)",
    category: "full_rig",
    sample_rate_hz: DEFAULT_FIXTURE_SAMPLE_RATE_HZ,
    bit_depth: FixtureBitDepth::Float32,
    expected_tail_ms: 6_250.0,
    supports_tempo_sync: true,
    controls: &[],
    switches: &[],
};

const CHAIN12_P16_CONTROLS: [FixtureControlSpec; 12] = [
    FixtureControlSpec {
        id: "drive",
        name: "Preamp Drive",
        unit: "%",
        default_norm: 0.0,
        min_physical: 0.0,
        max_physical: 100.0,
    },
    FixtureControlSpec {
        id: "tone",
        name: "Tone Tilt",
        unit: "%",
        default_norm: 0.0,
        min_physical: 0.0,
        max_physical: 100.0,
    },
    FixtureControlSpec {
        id: "space_mix",
        name: "Ping-Pong & Reverb Mix",
        unit: "%",
        default_norm: 0.0,
        min_physical: 0.0,
        max_physical: 100.0,
    },
    FixtureControlSpec {
        id: "comp_sustain",
        name: "Compressor Sustain",
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
    FixtureControlSpec {
        id: "presence",
        name: "Presence",
        unit: "%",
        default_norm: 0.0,
        min_physical: 0.0,
        max_physical: 100.0,
    },
    FixtureControlSpec {
        id: "master",
        name: "Master Volume",
        unit: "%",
        default_norm: 0.0,
        min_physical: 0.0,
        max_physical: 100.0,
    },
    FixtureControlSpec {
        id: "delay_time",
        name: "Delay Time",
        unit: "ms",
        default_norm: 0.0,
        min_physical: 50.0,
        max_physical: 750.0,
    },
    FixtureControlSpec {
        id: "delay_feedback",
        name: "Delay Feedback",
        unit: "%",
        default_norm: 0.0,
        min_physical: 0.0,
        max_physical: 95.0,
    },
    FixtureControlSpec {
        id: "reverb_decay",
        name: "Reverb Decay",
        unit: "s",
        default_norm: 0.0,
        min_physical: 0.5,
        max_physical: 12.0,
    },
];

const CHAIN12_P16_SWITCHES: [FixtureSwitchSpec; 4] = [
    FixtureSwitchSpec {
        id: "bright",
        name: "Bright Switch",
        options: &["Off", "On"],
        default_index: 0,
    },
    FixtureSwitchSpec {
        id: "channel",
        name: "Amp Channel",
        options: &["Clean", "Crunch", "Lead"],
        default_index: 1,
    },
    FixtureSwitchSpec {
        id: "cab_voice",
        name: "Cabinet Voicing",
        options: &["4x12 V30", "2x12 Alnico"],
        default_index: 0,
    },
    FixtureSwitchSpec {
        id: "routing",
        name: "Rig Routing",
        options: &["Stereo 2->2", "4CM", "WDW"],
        default_index: 0,
    },
];

const CHAIN12_P16_METADATA: FixtureDeviceMetadata = FixtureDeviceMetadata {
    id: "Chain-12-Full-EndToEnd-Stereo-Rig-P16",
    name: "Full End-to-End Stereo Rig (P=16 Parametric Mega-Rig)",
    category: "full_rig",
    sample_rate_hz: DEFAULT_FIXTURE_SAMPLE_RATE_HZ,
    bit_depth: FixtureBitDepth::Float32,
    expected_tail_ms: 28_000.0,
    supports_tempo_sync: true,
    controls: &CHAIN12_P16_CONTROLS,
    switches: &CHAIN12_P16_SWITCHES,
};

const RIG_FDN_LEN: usize = 32_768;
const RIG_FDN_TAPS_192: [usize; 4] = [1_493, 2_267, 3_181, 4_099]; // 7.8..21.3 ms @ 192 kHz

#[inline]
fn ping_pong_ring_len_for_rate(sample_rate_hz: u32) -> usize {
    let fs_4x = substep_rate_4x(sample_rate_hz);
    let needed = ((0.750 * fs_4x).ceil() as usize) + 128;
    needed.next_power_of_two().max(65_536)
}

/// Parameterization mode for [`Chain12FullEndToEndStereoRig`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chain12RigMode {
    /// Fixed `0`-knob snapshot (`P = 0`).
    FixedZeroKnob,
    /// Full `P = 16` (`12` continuous controls + `4` multi-throw switches) mega-rig.
    Parametric16,
}

/// `Chain-12-Full-EndToEnd-Stereo-Rig`: Full 4-stage rig (`Compressor -> 3-Ch Preamp + FMV + Presence/Master ->
/// Dual-Voiced Stereo Cabinet -> Ping-Pong Delay + 4-Line FDN Reverb`) supporting `FixedZeroKnob` (`P = 0`)
/// and `Parametric16` (`P = 16`) across `1->2` and `2->2`.
#[derive(Clone, Debug)]
pub struct Chain12FullEndToEndStereoRig {
    pub mode: Chain12RigMode,
    sample_rate_hz: u32,
    bit_depth: FixtureBitDepth,
    os: Oversampler4xF64,
    comp_env: f64,
    bright_hp_l: f64,
    bright_hp_r: f64,
    preamp_lp_l: f64,
    preamp_lp_r: f64,
    eq_low_l: f64,
    eq_low_r: f64,
    eq_mid_l: f64,
    eq_mid_r: f64,
    pres_lp_l: f64,
    pres_lp_r: f64,
    cab_lp_l: f64,
    cab_lp_r: f64,
    cab_res_re: [f64; 2],
    cab_res_im: [f64; 2],
    ping_pong_l: Vec<f64>,
    ping_pong_r: Vec<f64>,
    fdn_rings: [Vec<f64>; 4],
    wr_idx: usize,
    mod_phase: f64,
    tempo_bpm: f64,
}

impl Chain12FullEndToEndStereoRig {
    #[must_use]
    pub fn new(mode: Chain12RigMode) -> Self {
        Self::with_format(
            mode,
            DEFAULT_FIXTURE_SAMPLE_RATE_HZ,
            FixtureBitDepth::Float32,
        )
    }

    #[must_use]
    pub fn with_format(
        mode: Chain12RigMode,
        sample_rate_hz: u32,
        bit_depth: FixtureBitDepth,
    ) -> Self {
        let sr = clamp_sample_rate_hz(sample_rate_hz);
        let pp_len = ping_pong_ring_len_for_rate(sr);
        Self {
            mode,
            sample_rate_hz: sr,
            bit_depth,
            os: Oversampler4xF64::default(),
            comp_env: 0.0,
            bright_hp_l: 0.0,
            bright_hp_r: 0.0,
            preamp_lp_l: 0.0,
            preamp_lp_r: 0.0,
            eq_low_l: 0.0,
            eq_low_r: 0.0,
            eq_mid_l: 0.0,
            eq_mid_r: 0.0,
            pres_lp_l: 0.0,
            pres_lp_r: 0.0,
            cab_lp_l: 0.0,
            cab_lp_r: 0.0,
            cab_res_re: [0.0; 2],
            cab_res_im: [0.0; 2],
            ping_pong_l: vec![0.0; pp_len],
            ping_pong_r: vec![0.0; pp_len],
            fdn_rings: [
                vec![0.0; RIG_FDN_LEN],
                vec![0.0; RIG_FDN_LEN],
                vec![0.0; RIG_FDN_LEN],
                vec![0.0; RIG_FDN_LEN],
            ],
            wr_idx: 0,
            mod_phase: 0.0,
            tempo_bpm: 120.0,
        }
    }

    /// Sets host tempo (`20..=400 BPM`) for tempo-synced spatial modulation.
    pub fn set_tempo_bpm(&mut self, bpm: f64) {
        if bpm.is_finite() {
            self.tempo_bpm = bpm.clamp(20.0, 400.0);
        }
    }

    /// Evaluates symmetric stereo-linked power detector `0.5 * (in_l^2 + in_r^2)`.
    #[inline]
    #[must_use]
    pub fn stereo_linked_detector_power(in_l: f32, in_r: f32) -> f32 {
        let xl = sanitize_sample(in_l);
        let xr = sanitize_sample(in_r);
        0.5 * (xl * xl + xr * xr)
    }

    #[inline]
    fn ctrl_01(controls: &[f32], idx: usize) -> f64 {
        0.5 * (f64::from(sanitize_norm(
            controls.get(idx).copied().unwrap_or(0.0),
            0.0,
        )) + 1.0)
    }

    /// Steps one stereo frame through the full rig (`1->2` when `in_l == in_r`, `2->2` true-stereo
    /// with stereo-linked detector so L/R gain reduction is identical and pan shift is `0.00 deg`).
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

        let (
            drive_01,
            tone_01,
            space_01,
            comp_01,
            bass_01,
            mid_01,
            treb_01,
            pres_01,
            master_01,
            dly_time_01,
            dly_fb_01,
            rev_dec_01,
            bright_on,
            ch_mode,
            cab_alnico,
            routing_mode,
        ) = match self.mode {
            Chain12RigMode::FixedZeroKnob => (
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, false, 1u8, false, 0u8,
            ),
            Chain12RigMode::Parametric16 => (
                Self::ctrl_01(controls, 0),
                Self::ctrl_01(controls, 1),
                Self::ctrl_01(controls, 2),
                Self::ctrl_01(controls, 3),
                Self::ctrl_01(controls, 4),
                Self::ctrl_01(controls, 5),
                Self::ctrl_01(controls, 6),
                Self::ctrl_01(controls, 7),
                Self::ctrl_01(controls, 8),
                Self::ctrl_01(controls, 9),
                Self::ctrl_01(controls, 10),
                Self::ctrl_01(controls, 11),
                switches.first().copied().unwrap_or(0) != 0,
                switches.get(1).copied().unwrap_or(1) % 3,
                switches.get(2).copied().unwrap_or(0) != 0,
                switches.get(3).copied().unwrap_or(0) % 3,
            ),
        };

        let fs_4x = substep_rate_4x(self.sample_rate_hz);
        let scale = fs_4x / 192_000.0;
        let a_comp = scale_alpha_from_192k(0.004, fs_4x);
        let a_bright = scale_alpha_from_192k(0.085, fs_4x);
        let tone_a = scale_alpha_from_192k(0.045 + 0.14 * tone_01, fs_4x);
        let a_eq_lo = scale_alpha_from_192k(0.008, fs_4x);
        let a_eq_md = scale_alpha_from_192k(0.042, fs_4x);
        let a_pres = scale_alpha_from_192k(0.095, fs_4x);

        let linked_det = f64::from(Self::stereo_linked_detector_power(xl, xr)).sqrt();
        let comp_slope = 0.6 + 2.8 * comp_01;
        let ch_gain = match ch_mode {
            0 => 0.45, // Clean
            1 => 1.00, // Crunch
            _ => 2.15, // Lead
        };
        let drive = (1.4 + 4.2 * drive_01) * ch_gain;
        let bright_amt = if bright_on { 0.42 } else { 0.0 };
        let w_bass = 0.45 + 1.10 * bass_01;
        let w_mid = 0.40 + 1.20 * mid_01;
        let w_treb = 0.45 + 1.15 * treb_01;
        let pres_gain = -0.30 + 0.75 * pres_01;
        let master_gain = 0.18 + 0.82 * master_01;

        let (cab_alpha_192, cab_omega_192, cab_r_192) = if cab_alnico {
            (0.165, 2.0 * PI * 165.0 / 192_000.0, 0.9965)
        } else {
            (0.115, 2.0 * PI * 105.0 / 192_000.0, 0.9980)
        };
        let cab_alpha = scale_alpha_from_192k(cab_alpha_192, fs_4x);
        let (cab_r, cab_omega) = scale_modal_pole_from_192k(cab_r_192, cab_omega_192, fs_4x);
        let (cab_sin, cab_cos) = cab_omega.sin_cos();

        let pp_ring_len = self.ping_pong_l.len();
        let max_dly_taps = (pp_ring_len - 96) as f64;
        let dly_ms = 50.0 + 700.0 * dly_time_01;
        let dly_l_base = (dly_ms * 1.0e-3 * fs_4x).clamp(32.0, max_dly_taps);
        let dly_r_base = ((dly_ms * 0.75) * 1.0e-3 * fs_4x).clamp(24.0, max_dly_taps);
        let dly_fb = 0.85 * dly_fb_01;

        let mut fdn_taps = [0_usize; 4];
        let mut mean_fdn_tap = 0.0_f64;
        for m in 0..4 {
            let t = (((RIG_FDN_TAPS_192[m] as f64) * scale).round() as usize)
                .clamp(15, RIG_FDN_LEN - 16)
                | 1;
            fdn_taps[m] = t;
            mean_fdn_tap += (t as f64) * 0.25;
        }
        let rev_rt60 = 0.5 + 11.5 * rev_dec_01;
        let g_fdn = 10.0_f64
            .powf(-3.0 * mean_fdn_tap / (rev_rt60 * fs_4x))
            .clamp(0.65, 0.998);
        let space_mix = 0.08 + 0.56 * space_01;
        let mod_inc = (self.tempo_bpm / 60.0) * 0.25 / fs_4x;
        let mod_depth = 12.0 * scale;

        let [yl, yr] = self
            .os
            .process_frame_4x(f64::from(xl), f64::from(xr), |_, ul, ur| {
                // Stage 1: Stereo-linked compressor
                self.comp_env += a_comp * (linked_det - self.comp_env);
                let gr = 1.0 / (1.0 + comp_slope * self.comp_env);
                let c_l = ul * gr;
                let c_r = ur * gr;

                // Stage 2: Bright cap + 3-channel preamp + FMV + Presence + Master
                self.bright_hp_l += a_bright * (c_l - self.bright_hp_l);
                self.bright_hp_r += a_bright * (c_r - self.bright_hp_r);
                let in_pre_l = c_l + bright_amt * (c_l - self.bright_hp_l);
                let in_pre_r = c_r + bright_amt * (c_r - self.bright_hp_r);

                let pre_l = if ch_mode == 2 {
                    let s1 = (0.75 * drive * in_pre_l).tanh();
                    (1.6 * s1).tanh()
                } else {
                    (drive * in_pre_l).tanh()
                };
                let pre_r = if ch_mode == 2 {
                    let s1 = (0.75 * drive * in_pre_r).tanh();
                    (1.6 * s1).tanh()
                } else {
                    (drive * in_pre_r).tanh()
                };

                self.preamp_lp_l += tone_a * (pre_l - self.preamp_lp_l);
                self.preamp_lp_r += tone_a * (pre_r - self.preamp_lp_r);

                self.eq_low_l += a_eq_lo * (self.preamp_lp_l - self.eq_low_l);
                self.eq_low_r += a_eq_lo * (self.preamp_lp_r - self.eq_low_r);
                self.eq_mid_l += a_eq_md * (self.preamp_lp_l - self.eq_mid_l);
                self.eq_mid_r += a_eq_md * (self.preamp_lp_r - self.eq_mid_r);

                let eq_l = 0.34
                    * (w_bass * self.eq_low_l
                        + w_mid * (self.eq_mid_l - self.eq_low_l)
                        + w_treb * (self.preamp_lp_l - self.eq_mid_l));
                let eq_r = 0.34
                    * (w_bass * self.eq_low_r
                        + w_mid * (self.eq_mid_r - self.eq_low_r)
                        + w_treb * (self.preamp_lp_r - self.eq_mid_r));

                self.pres_lp_l += a_pres * (eq_l - self.pres_lp_l);
                self.pres_lp_r += a_pres * (eq_r - self.pres_lp_r);
                let pwr_l =
                    ((eq_l + pres_gain * (eq_l - self.pres_lp_l)) * master_gain * 1.6).tanh();
                let pwr_r =
                    ((eq_r + pres_gain * (eq_r - self.pres_lp_r)) * master_gain * 1.6).tanh();

                // Stage 3: Dual-voiced stereo cabinet
                self.cab_lp_l += cab_alpha * (pwr_l - self.cab_lp_l);
                self.cab_lp_r += cab_alpha * (pwr_r - self.cab_lp_r);
                let nr_l = cab_r * (cab_cos * self.cab_res_re[0] - cab_sin * self.cab_res_im[0])
                    + self.cab_lp_l;
                let ni_l = cab_r * (cab_sin * self.cab_res_re[0] + cab_cos * self.cab_res_im[0]);
                self.cab_res_re[0] = nr_l;
                self.cab_res_im[0] = ni_l;
                let nr_r = cab_r * (cab_cos * self.cab_res_re[1] - cab_sin * self.cab_res_im[1])
                    + self.cab_lp_r;
                let ni_r = cab_r * (cab_sin * self.cab_res_re[1] + cab_cos * self.cab_res_im[1]);
                self.cab_res_re[1] = nr_r;
                self.cab_res_im[1] = ni_r;

                let cab_l = 0.72 * self.cab_lp_l + 0.28 * (1.0 - cab_r) * nr_l;
                let cab_r_out = 0.72 * self.cab_lp_r + 0.28 * (1.0 - cab_r) * nr_r;

                // Stage 4: Stereo Ping-Pong Delay + 4-Line Hadamard FDN Reverb + Routing Matrix
                self.mod_phase = (self.mod_phase + mod_inc).fract();
                let mod_ofs = mod_depth * (2.0 * PI * self.mod_phase).sin();
                let pp_mask = pp_ring_len - 1;
                let lag_l = (dly_l_base - mod_ofs).clamp(4.0, max_dly_taps + 32.0);
                let lag_l_int = lag_l.floor() as usize;
                let lag_l_frac = lag_l - (lag_l_int as f64);
                let dly_l = (1.0 - lag_l_frac)
                    * self.ping_pong_l[self.wr_idx.wrapping_sub(lag_l_int) & pp_mask]
                    + lag_l_frac
                        * self.ping_pong_l[self.wr_idx.wrapping_sub(lag_l_int + 1) & pp_mask];

                let lag_r = (dly_r_base + mod_ofs).clamp(4.0, max_dly_taps + 32.0);
                let lag_r_int = lag_r.floor() as usize;
                let lag_r_frac = lag_r - (lag_r_int as f64);
                let dly_r = (1.0 - lag_r_frac)
                    * self.ping_pong_r[self.wr_idx.wrapping_sub(lag_r_int) & pp_mask]
                    + lag_r_frac
                        * self.ping_pong_r[self.wr_idx.wrapping_sub(lag_r_int + 1) & pp_mask];
                self.ping_pong_l[self.wr_idx & pp_mask] = cab_l + dly_fb * dly_r;
                self.ping_pong_r[self.wr_idx & pp_mask] = cab_r_out + dly_fb * dly_l;

                let fdn_mask = RIG_FDN_LEN - 1;
                let mut fr = [0.0_f64; 4];
                for m in 0..4 {
                    fr[m] = self.fdn_rings[m][(self.wr_idx.wrapping_sub(fdn_taps[m])) & fdn_mask];
                }
                let fs = 0.5 * (fr[0] + fr[1] + fr[2] + fr[3]);
                let mono_cab = 0.5 * (cab_l + cab_r_out);
                for m in 0..4 {
                    let h = fs - fr[m];
                    self.fdn_rings[m][self.wr_idx & fdn_mask] = 0.25 * mono_cab + g_fdn * h;
                }
                self.wr_idx = self.wr_idx.wrapping_add(1);

                let wet_l = 0.55 * dly_l + 0.45 * (fr[0] - fr[2]);
                let wet_r = 0.55 * dly_r + 0.45 * (fr[1] - fr[3]);

                match routing_mode {
                    0 => [
                        (1.0 - space_mix) * cab_l + space_mix * wet_l,
                        (1.0 - space_mix) * cab_r_out + space_mix * wet_r,
                    ],
                    1 => [
                        // 4CM: series insert blend + post-loop return
                        0.78 * cab_l + space_mix * (0.75 * wet_l + 0.25 * dly_r),
                        0.78 * cab_r_out + space_mix * (0.75 * wet_r + 0.25 * dly_l),
                    ],
                    _ => {
                        // WDW: centered mono dry core + wide stereo wet spread
                        let center_dry = 0.65 * mono_cab;
                        [
                            center_dry + space_mix * (wet_l - 0.25 * wet_r),
                            center_dry + space_mix * (wet_r - 0.25 * wet_l),
                        ]
                    }
                }
            });
        [
            self.bit_depth.quantize_f64_to_f32(yl),
            self.bit_depth.quantize_f64_to_f32(yr),
        ]
    }
}

impl BlackBoxModel for Chain12FullEndToEndStereoRig {
    fn reset(&mut self) {
        self.os.reset();
        self.comp_env = 0.0;
        self.bright_hp_l = 0.0;
        self.bright_hp_r = 0.0;
        self.preamp_lp_l = 0.0;
        self.preamp_lp_r = 0.0;
        self.eq_low_l = 0.0;
        self.eq_low_r = 0.0;
        self.eq_mid_l = 0.0;
        self.eq_mid_r = 0.0;
        self.pres_lp_l = 0.0;
        self.pres_lp_r = 0.0;
        self.cab_lp_l = 0.0;
        self.cab_lp_r = 0.0;
        self.cab_res_re = [0.0; 2];
        self.cab_res_im = [0.0; 2];
        self.ping_pong_l.fill(0.0);
        self.ping_pong_r.fill(0.0);
        for ring in &mut self.fdn_rings {
            ring.fill(0.0);
        }
        self.wr_idx = 0;
        self.mod_phase = 0.0;
    }

    fn metadata(&self) -> FixtureDeviceMetadata {
        let base = match self.mode {
            Chain12RigMode::FixedZeroKnob => CHAIN12_FIXED_METADATA,
            Chain12RigMode::Parametric16 => CHAIN12_P16_METADATA,
        };
        FixtureDeviceMetadata {
            sample_rate_hz: self.sample_rate_hz,
            bit_depth: self.bit_depth,
            ..base
        }
    }

    fn set_sample_rate_hz(&mut self, sample_rate_hz: u32) {
        let sr = clamp_sample_rate_hz(sample_rate_hz);
        let pp_len = ping_pong_ring_len_for_rate(sr);
        self.sample_rate_hz = sr;
        self.ping_pong_l.resize(pp_len, 0.0);
        self.ping_pong_r.resize(pp_len, 0.0);
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
    fn test_chain12_full_stereo_rig_all_16_params_and_mono_stereo_parity() {
        let dry_l = sine_wave(4_800, 220.0, 0.45);
        let dry_r = sine_wave(4_800, 330.0, 0.45);

        let mut c12_fixed = Chain12FullEndToEndStereoRig::new(Chain12RigMode::FixedZeroKnob);
        assert_eq!(
            c12_fixed.metadata().id,
            "Chain-12-Full-EndToEnd-Stereo-Rig-Fixed"
        );
        assert!((c12_fixed.metadata().expected_tail_ms - 6_250.0).abs() < 1.0e-3);
        assert_eq!(
            (
                c12_fixed.num_inputs(),
                c12_fixed.num_outputs(),
                c12_fixed.num_controls(),
                c12_fixed.num_switches()
            ),
            (2, 2, 0, 0)
        );
        let (y_f_l, y_f_r) = run_stereo_block(&mut c12_fixed, &dry_l, &dry_r, &[], &[]);
        assert!(y_f_l.iter().chain(&y_f_r).all(|v| v.is_finite()));

        let mut c12_p16 = Chain12FullEndToEndStereoRig::new(Chain12RigMode::Parametric16);
        assert_eq!(
            c12_p16.metadata().id,
            "Chain-12-Full-EndToEnd-Stereo-Rig-P16"
        );
        assert!((c12_p16.metadata().expected_tail_ms - 28_000.0).abs() < 1.0e-3);
        assert_eq!(
            (
                c12_p16.num_inputs(),
                c12_p16.num_outputs(),
                c12_p16.num_controls(),
                c12_p16.num_switches()
            ),
            (2, 2, 12, 4)
        );
        c12_p16.set_tempo_bpm(132.0);

        // Verify genuine 1->2 mono-in vs 2->2 symmetric stereo-in parity via process_frame(&[x]) vs process_frame(&[x, x])
        let mut mono_out = Vec::with_capacity(256);
        c12_p16.reset();
        for &x in &dry_l[..256] {
            let mut y = [0.0_f32; 2];
            c12_p16.process_frame(&[x], &mut y, &[0.2; 12], &[0, 1, 0, 0]);
            mono_out.push(y);
        }
        let mut stereo_out = Vec::with_capacity(256);
        c12_p16.reset();
        for &x in &dry_l[..256] {
            let mut y = [0.0_f32; 2];
            c12_p16.process_frame(&[x, x], &mut y, &[0.2; 12], &[0, 1, 0, 0]);
            stereo_out.push(y);
        }
        assert_eq!(mono_out, stereo_out);
        assert_eq!(
            Chain12FullEndToEndStereoRig::stereo_linked_detector_power(0.6, 0.2),
            Chain12FullEndToEndStereoRig::stereo_linked_detector_power(0.2, 0.6)
        );

        // Verify EVERY ONE of the 12 controls and 4 switches produces a non-zero output change
        let mut base_ctrls = c12_p16.metadata().default_controls();
        base_ctrls[9] = -1.0;
        let base_sw = c12_p16.metadata().default_switches();
        let (base_l, base_r) =
            run_stereo_block(&mut c12_p16, &dry_l, &dry_r, &base_ctrls, &base_sw);

        for p in 0..12 {
            let mut alt_ctrls = base_ctrls.clone();
            alt_ctrls[p] = if p == 9 { -0.6 } else { 0.75 };
            let (alt_l, alt_r) =
                run_stereo_block(&mut c12_p16, &dry_l, &dry_r, &alt_ctrls, &base_sw);
            assert!(
                alt_l != base_l || alt_r != base_r,
                "Chain-12 control {p} ({}) had zero effect",
                c12_p16.metadata().controls[p].id
            );
        }
        for s in 0..4 {
            let n_opts = c12_p16.metadata().switches[s].options.len() as u8;
            for opt in 0..n_opts {
                if opt == base_sw[s] {
                    continue;
                }
                let mut alt_sw = base_sw.clone();
                alt_sw[s] = opt;
                let (alt_l, alt_r) =
                    run_stereo_block(&mut c12_p16, &dry_l, &dry_r, &base_ctrls, &alt_sw);
                assert!(
                    alt_l != base_l || alt_r != base_r,
                    "Chain-12 switch {s} ({}) option {opt} had zero effect",
                    c12_p16.metadata().switches[s].id
                );
            }
        }
    }
}
