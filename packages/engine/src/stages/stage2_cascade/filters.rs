//! Stage II Tone-Basis Filters, Arena Parameter Loading, ZOH Rescaling & Inter-Stage Helpers.
//!
//! Canonical Reference: `docs/specs/dsp/op_n_commuted_cascade.md` §4 & `docs/specs/dsp/router_and_stages_topology.md` §2.2.

use super::Stage2CommutedCascade;
use crate::math::flush_recursive_state_f32;

/// 4-Filter Tone Basis Filter State: `[Direct, LPF_80, LPF_400, HPF_1200]`.
#[derive(Clone, Copy, Debug)]
pub struct ToneBasisFilters {
    pub lpf80_state: f32,
    pub lpf400_state: f32,
    pub hpf1200_state: f32,
    pub hpf1200_x1: f32,
    pub base_alpha_80: f32,
    pub base_alpha_400: f32,
    pub base_alpha_1200: f32,
    pub alpha_80: f32,
    pub alpha_400: f32,
    pub alpha_1200: f32,
}

impl Default for ToneBasisFilters {
    fn default() -> Self {
        Self::new()
    }
}

impl ToneBasisFilters {
    // 1-pole filter coefficients at 48 kHz
    // 80 Hz LPF: alpha = exp(-2*pi*80/48000) ~ 0.98959
    pub const ALPHA_80: f32 = 0.98959;
    // 400 Hz LPF: alpha = exp(-2*pi*400/48000) ~ 0.94916
    pub const ALPHA_400: f32 = 0.94916;
    // 1200 Hz HPF: alpha = exp(-2*pi*1200/48000) ~ 0.85463
    pub const ALPHA_1200: f32 = 0.85463;

    pub const fn new() -> Self {
        Self {
            lpf80_state: 0.0,
            lpf400_state: 0.0,
            hpf1200_state: 0.0,
            hpf1200_x1: 0.0,
            base_alpha_80: Self::ALPHA_80,
            base_alpha_400: Self::ALPHA_400,
            base_alpha_1200: Self::ALPHA_1200,
            alpha_80: Self::ALPHA_80,
            alpha_400: Self::ALPHA_400,
            alpha_1200: Self::ALPHA_1200,
        }
    }

    /// Initializes the 4-filter tone basis (`Direct`, `LPF_80`, `LPF_400`, `HPF_1200`) to exact
    /// resting DC equilibrium for constant input `v_rest` (`LPF y[-1] = v_rest`,
    /// `HPF y[-1] = 0, x[-1] = v_rest`, `ref[dsp.op-n.commuted.dc-equilibrium]`).
    pub fn init_dc_equilibrium(&mut self, v_rest: f32) {
        let vr = flush_recursive_state_f32(v_rest);
        self.lpf80_state = vr;
        self.lpf400_state = vr;
        self.hpf1200_state = 0.0;
        self.hpf1200_x1 = vr;
    }

    pub fn reset(&mut self) {
        self.lpf80_state = 0.0;
        self.lpf400_state = 0.0;
        self.hpf1200_state = 0.0;
        self.hpf1200_x1 = 0.0;
    }

    /// Reconfigures 1-pole tone basis cutoffs (`80 Hz`, `400 Hz`, `1200 Hz`) under continuous-time ZOH.
    pub fn reconfigure_sample_rate(&mut self, zoh: &crate::container::ZohRescaledParams) {
        self.alpha_80 = zoh.rescale_real_pole(self.base_alpha_80);
        self.alpha_400 = zoh.rescale_real_pole(self.base_alpha_400);
        self.alpha_1200 = zoh.rescale_real_pole(self.base_alpha_1200);
    }

    /// Advances the tone-basis filters by one sample. Direct callers must hold a
    /// [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, x: f32) -> [f32; 4] {
        // Direct
        let s_direct = x;

        // LPF 80 Hz
        let s_lpf80 = self.alpha_80 * self.lpf80_state + (1.0 - self.alpha_80) * x;

        // LPF 400 Hz
        let s_lpf400 = self.alpha_400 * self.lpf400_state + (1.0 - self.alpha_400) * x;

        // HPF 1200 Hz (y[n] = alpha * y[n-1] + alpha * (x[n] - x[n-1]))
        let s_hpf1200 =
            self.alpha_1200 * self.hpf1200_state + self.alpha_1200 * (x - self.hpf1200_x1);
        self.lpf80_state = flush_recursive_state_f32(s_lpf80);
        self.lpf400_state = flush_recursive_state_f32(s_lpf400);
        self.hpf1200_state = flush_recursive_state_f32(s_hpf1200);
        self.hpf1200_x1 = flush_recursive_state_f32(x);

        [s_direct, s_lpf80, s_lpf400, s_hpf1200]
    }

    /// Advances a stereo pair (`self` = Left, `right` = Right) of tone-basis filters by one sample.
    #[inline(always)]
    pub fn step_stereo(&mut self, right: &mut Self, x_l: f32, x_r: f32) -> ([f32; 4], [f32; 4]) {
        (self.step(x_l), right.step(x_r))
    }
}

impl Stage2CommutedCascade {
    #[inline]
    pub(crate) fn valid_pole(p: f32, default: f32) -> f32 {
        if p.is_finite() && (0.5..=0.999995).contains(&p) {
            p
        } else {
            default
        }
    }

    /// Loads Stage II 48-float parameter block from `Arena 0 [4,096..4,288]` and Role A (`b = 0`)
    /// inter-stage coupling weights from `CASCADE_W_BASES[0..4][0..16]`.
    pub fn load_from_arena(&mut self, arena: &[u8]) {
        if arena.len() < 4_160 {
            return;
        }
        use crate::provenance::watermark::read_f32_le;
        let a80 = Self::valid_pole(read_f32_le(arena, 4_096), ToneBasisFilters::ALPHA_80);
        let a400 = Self::valid_pole(read_f32_le(arena, 4_100), ToneBasisFilters::ALPHA_400);
        let a1200 = Self::valid_pole(read_f32_le(arena, 4_104), ToneBasisFilters::ALPHA_1200);
        let rsag = Self::valid_pole(read_f32_le(arena, 4_108), 0.995);
        self.tone_basis_l.base_alpha_80 = a80;
        self.tone_basis_r.base_alpha_80 = a80;
        self.tone_basis_l.base_alpha_400 = a400;
        self.tone_basis_r.base_alpha_400 = a400;
        self.tone_basis_l.base_alpha_1200 = a1200;
        self.tone_basis_r.base_alpha_1200 = a1200;
        self.base_r_sag = rsag;

        // Stereo V1A (4,112..4,136): [drive_l, drive_r, asym_l, asym_r, comp_l, comp_r]
        // Also supports legacy 3-float [drive, asym, comp] layout when bytes 4,124..4,136 are zero.
        let v1a_f0 = read_f32_le(arena, 4_112);
        let v1a_f1 = read_f32_le(arena, 4_116);
        let v1a_f2 = read_f32_le(arena, 4_120);
        let v1a_f3 = read_f32_le(arena, 4_124);
        let v1a_f4 = read_f32_le(arena, 4_128);
        let v1a_f5 = read_f32_le(arena, 4_132);
        let v1a_has_stereo_tail = v1a_f3.abs() > 1e-6 || v1a_f4.abs() > 1e-6 || v1a_f5.abs() > 1e-6;
        if v1a_f0.is_finite() && v1a_f0 > 1e-6 {
            self.v1a_beta = v1a_f0.clamp(0.1, 8.0);
            if v1a_has_stereo_tail {
                self.v1a_delta = v1a_f2.clamp(-0.9, 0.9);
                self.v1a_comp = if v1a_f4.is_finite() && v1a_f4 > 1e-6 {
                    v1a_f4.clamp(0.1, 2.0)
                } else {
                    0.5
                };
                if v1a_f1.is_finite() && v1a_f1 > 1e-6 {
                    self.v1a_beta_r = v1a_f1.clamp(0.1, 8.0);
                    self.v1a_delta_r = v1a_f3.clamp(-0.9, 0.9);
                    self.v1a_comp_r = if v1a_f5.is_finite() && v1a_f5 > 1e-6 {
                        v1a_f5.clamp(0.1, 2.0)
                    } else {
                        self.v1a_comp
                    };
                } else {
                    self.v1a_beta_r = self.v1a_beta;
                    self.v1a_delta_r = self.v1a_delta;
                    self.v1a_comp_r = self.v1a_comp;
                }
            } else {
                self.v1a_delta = v1a_f1.clamp(-0.9, 0.9);
                self.v1a_comp = if v1a_f2.is_finite() && v1a_f2 > 1e-6 {
                    v1a_f2.clamp(0.1, 2.0)
                } else {
                    0.5
                };
                self.v1a_beta_r = self.v1a_beta;
                self.v1a_delta_r = self.v1a_delta;
                self.v1a_comp_r = self.v1a_comp;
            }
        } else {
            self.v1a_beta = 1.0;
            self.v1a_delta = 0.25;
            self.v1a_comp = 0.5;
            self.v1a_beta_r = 1.0;
            self.v1a_delta_r = 0.25;
            self.v1a_comp_r = 0.5;
        }

        // Stereo V1B (4,136..4,160): [drive_l, drive_r, asym_l, asym_r, comp_l, comp_r]
        let v1b_f0 = read_f32_le(arena, 4_136);
        let v1b_f1 = read_f32_le(arena, 4_140);
        let v1b_f2 = read_f32_le(arena, 4_144);
        let v1b_f3 = read_f32_le(arena, 4_148);
        let v1b_f4 = read_f32_le(arena, 4_152);
        let v1b_f5 = read_f32_le(arena, 4_156);
        let v1b_has_stereo_tail = v1b_f3.abs() > 1e-6 || v1b_f4.abs() > 1e-6 || v1b_f5.abs() > 1e-6;
        if v1b_f0.is_finite() && v1b_f0 > 1e-6 {
            self.v1b_beta = v1b_f0.clamp(0.1, 8.0);
            if v1b_has_stereo_tail {
                self.v1b_delta = v1b_f2.clamp(-0.9, 0.9);
                self.v1b_comp = if v1b_f4.is_finite() && v1b_f4 > 1e-6 {
                    v1b_f4.clamp(0.1, 2.0)
                } else {
                    0.5
                };
                if v1b_f1.is_finite() && v1b_f1 > 1e-6 {
                    self.v1b_beta_r = v1b_f1.clamp(0.1, 8.0);
                    self.v1b_delta_r = v1b_f3.clamp(-0.9, 0.9);
                    self.v1b_comp_r = if v1b_f5.is_finite() && v1b_f5 > 1e-6 {
                        v1b_f5.clamp(0.1, 2.0)
                    } else {
                        self.v1b_comp
                    };
                } else {
                    self.v1b_beta_r = self.v1b_beta;
                    self.v1b_delta_r = self.v1b_delta;
                    self.v1b_comp_r = self.v1b_comp;
                }
            } else {
                self.v1b_delta = v1b_f1.clamp(-0.9, 0.9);
                self.v1b_comp = if v1b_f2.is_finite() && v1b_f2 > 1e-6 {
                    v1b_f2.clamp(0.1, 2.0)
                } else {
                    0.5
                };
                self.v1b_beta_r = self.v1b_beta;
                self.v1b_delta_r = self.v1b_delta;
                self.v1b_comp_r = self.v1b_comp;
            }
        } else {
            self.v1b_beta = 1.0;
            self.v1b_delta = 0.20;
            self.v1b_comp = 0.5;
            self.v1b_beta_r = 1.0;
            self.v1b_delta_r = 0.20;
            self.v1b_comp_r = 0.5;
        }

        if arena.len() >= 4_288 {
            // Inter-stage L^(1) (4,160..4,192)
            self.base_l1_poles[0] =
                Self::valid_pole(read_f32_le(arena, 4_160), Self::DEFAULT_L1_POLES[0]);
            self.base_l1_poles[1] =
                Self::valid_pole(read_f32_le(arena, 4_164), Self::DEFAULT_L1_POLES[1]);
            let g_sh1 = read_f32_le(arena, 4_168);
            self.base_l1_gains[0] = if g_sh1.is_finite() {
                g_sh1.clamp(-2.0, 2.0)
            } else {
                0.0
            };
            self.base_l1_poles[2] =
                Self::valid_pole(read_f32_le(arena, 4_172), Self::DEFAULT_L1_POLES[2]);
            let r_m1 = read_f32_le(arena, 4_176);
            self.base_l1_poles[3] = if r_m1.is_finite() && (0.25..=0.9995).contains(&r_m1) {
                r_m1
            } else {
                Self::DEFAULT_L1_POLES[3]
            };
            let th_m1 = read_f32_le(arena, 4_180);
            self.base_l1_poles[4] =
                if th_m1.is_finite() && (1e-4..=core::f32::consts::PI - 1e-4).contains(&th_m1) {
                    th_m1
                } else {
                    Self::DEFAULT_L1_POLES[4]
                };
            let g_m1 = read_f32_le(arena, 4_184);
            self.base_l1_gains[1] = if g_m1.is_finite() {
                g_m1.clamp(-2.0, 2.0)
            } else {
                0.0
            };
            let g_bs1 = read_f32_le(arena, 4_188);
            self.base_l1_gains[2] = if g_bs1.is_finite() {
                g_bs1.clamp(-2.0, 2.0)
            } else {
                0.0
            };

            // Inter-stage L^(2) (4,192..4,224)
            self.base_l2_poles[0] =
                Self::valid_pole(read_f32_le(arena, 4_192), Self::DEFAULT_L2_POLES[0]);
            self.base_l2_poles[1] =
                Self::valid_pole(read_f32_le(arena, 4_196), Self::DEFAULT_L2_POLES[1]);
            let g_sh2 = read_f32_le(arena, 4_200);
            self.base_l2_gains[0] = if g_sh2.is_finite() {
                g_sh2.clamp(-2.0, 2.0)
            } else {
                0.0
            };
            self.base_l2_poles[2] =
                Self::valid_pole(read_f32_le(arena, 4_204), Self::DEFAULT_L2_POLES[2]);
            let r_m2 = read_f32_le(arena, 4_208);
            self.base_l2_poles[3] = if r_m2.is_finite() && (0.25..=0.9995).contains(&r_m2) {
                r_m2
            } else {
                Self::DEFAULT_L2_POLES[3]
            };
            let th_m2 = read_f32_le(arena, 4_212);
            self.base_l2_poles[4] =
                if th_m2.is_finite() && (1e-4..=core::f32::consts::PI - 1e-4).contains(&th_m2) {
                    th_m2
                } else {
                    Self::DEFAULT_L2_POLES[4]
                };
            let g_m2 = read_f32_le(arena, 4_216);
            self.base_l2_gains[1] = if g_m2.is_finite() {
                g_m2.clamp(-2.0, 2.0)
            } else {
                0.0
            };
            let g_bs2 = read_f32_le(arena, 4_220);
            self.base_l2_gains[2] = if g_bs2.is_finite() {
                g_bs2.clamp(-2.0, 2.0)
            } else {
                0.0
            };

            // Full V2A (4,224..4,240), Pro V2B (4,240..4,256), Ultra V3 (4,256..4,272)
            for b in 0..3 {
                let off = 4_224 + b * 16;
                let defaults = Self::DEFAULT_BRANCH_PARAMS[b];
                let alpf = Self::valid_pole(read_f32_le(arena, off), defaults[0]);
                let ahpf = Self::valid_pole(read_f32_le(arena, off + 4), defaults[1]);
                let drv = read_f32_le(arena, off + 8);
                let asy = read_f32_le(arena, off + 12);
                self.base_branch_params[b][0] = alpf;
                self.base_branch_params[b][1] = ahpf;
                self.base_branch_params[b][2] = if drv.is_finite() && drv > 1e-6 {
                    drv.clamp(0.1, 8.0)
                } else {
                    defaults[2]
                };
                self.base_branch_params[b][3] = if asy.is_finite() && asy.abs() > 1e-6 {
                    asy.clamp(-0.9, 0.9)
                } else {
                    defaults[3]
                };
            }

            // Global cascade scaling (4,272..4,288)
            let h_drv = read_f32_le(arena, 4_272);
            let h_dif = read_f32_le(arena, 4_276);
            let s_atk = read_f32_le(arena, 4_280);
            let s_rel = read_f32_le(arena, 4_284);
            self.base_global_scales[0] = if h_drv.is_finite() && h_drv > 1e-6 {
                h_drv.clamp(0.1, 4.0)
            } else {
                Self::DEFAULT_GLOBAL_SCALES[0]
            };
            self.base_global_scales[1] = if h_dif.is_finite() && h_dif > 1e-6 {
                h_dif.clamp(0.25, 4.0)
            } else {
                Self::DEFAULT_GLOBAL_SCALES[1]
            };
            self.base_global_scales[2] = Self::valid_pole(s_atk, Self::DEFAULT_GLOBAL_SCALES[2]);
            self.base_global_scales[3] = Self::valid_pole(s_rel, Self::DEFAULT_GLOBAL_SCALES[3]);
        }

        // Load Role A (b = 0, 16 floats/tier at CASCADE_W_BASES[t][0..16])
        const CASCADE_W_BASES: [usize; 4] = [1_024, 11_264, 22_336, 36_224];
        for (t, &base) in CASCADE_W_BASES.iter().enumerate() {
            if arena.len() >= base + 64 {
                for i in 0..16 {
                    let delta_a = read_f32_le(arena, base + i * 4);
                    self.role_a_scales[t][i] = if delta_a.is_finite() {
                        (1.0 + delta_a).clamp(0.5, 1.5)
                    } else {
                        1.0
                    };
                }
            } else {
                self.role_a_scales[t] = [1.0; 16];
            }
        }
    }

    /// Reconfigures Stage II AC-coupling pole, sag integrator poles, hysteresis rate scale,
    /// inter-stage `L^{(1)}`/`L^{(2)}` poles, branch poles, and 4-filter tone basis cutoffs
    /// for host sample rate under continuous-time ZOH.
    pub fn reconfigure_sample_rate(&mut self, zoh: &crate::container::ZohRescaledParams) {
        self.r_ac = zoh.r_ac_host;
        self.r_sag = zoh.rescale_real_pole(self.base_r_sag);
        self.hyst_diff_scale = zoh.hysteresis_diff_scale() * self.base_global_scales[1];
        self.sag_attack_alpha = zoh.rescale_real_pole(self.base_global_scales[2]);
        self.sag_release_alpha = zoh.rescale_real_pole(self.base_global_scales[3]);
        self.tone_basis_l.reconfigure_sample_rate(zoh);
        self.tone_basis_r.reconfigure_sample_rate(zoh);

        let (r1_h, _, th1_h, _) =
            zoh.rescale_op_m_channel(self.base_l1_poles[3], 0.0, self.base_l1_poles[4], 1.0);
        let (c1, s1) = crate::operators::op_m::algebraic_unit_rotor_op_m(th1_h);
        let omr1 = (1.0 - r1_h * r1_h).max(0.0);
        let bg1 = omr1 * crate::operators::op_m::fast_rsqrt_nr2(omr1);
        self.l1_poles = [
            zoh.rescale_real_pole(self.base_l1_poles[0]),
            zoh.rescale_real_pole(self.base_l1_poles[1]),
            zoh.rescale_real_pole(self.base_l1_poles[2]),
            r1_h * c1,
            r1_h * s1,
            bg1,
        ];

        let (r2_h, _, th2_h, _) =
            zoh.rescale_op_m_channel(self.base_l2_poles[3], 0.0, self.base_l2_poles[4], 1.0);
        let (c2, s2) = crate::operators::op_m::algebraic_unit_rotor_op_m(th2_h);
        let omr2 = (1.0 - r2_h * r2_h).max(0.0);
        let bg2 = omr2 * crate::operators::op_m::fast_rsqrt_nr2(omr2);
        self.l2_poles = [
            zoh.rescale_real_pole(self.base_l2_poles[0]),
            zoh.rescale_real_pole(self.base_l2_poles[1]),
            zoh.rescale_real_pole(self.base_l2_poles[2]),
            r2_h * c2,
            r2_h * s2,
            bg2,
        ];

        for b in 0..3 {
            self.branch_poles[b][0] = zoh.rescale_real_pole(self.base_branch_params[b][0]);
            self.branch_poles[b][1] = zoh.rescale_real_pole(self.base_branch_params[b][1]);
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub(crate) fn step_interstage_filter_stereo(
        state_l: &mut [f32; 6],
        state_r: &mut [f32; 6],
        poles: &[f32; 6],
        gains: &[f32; 3],
        x_l: f32,
        a_even_l: f32,
        x_r: f32,
        a_even_r: f32,
    ) -> (f32, f32) {
        let a_hpf = poles[0];
        let a_sh = poles[1];
        let a_rc = poles[2];
        let m_re = poles[3];
        let m_im = poles[4];
        let m_bg = poles[5];
        let one_minus_sh = 1.0 - a_sh;
        let one_minus_rc = 1.0 - a_rc;
        let g0 = gains[0];
        let g1 = gains[1];
        let g2 = gains[2];

        let hpf_y_l = a_hpf * (state_l[1] + x_l - state_l[0]);
        let lpf_sh_l = a_sh * state_l[2] + one_minus_sh * x_l;
        let shelf_hi_l = x_l - lpf_sh_l;
        let prev_re_l = state_l[3];
        let prev_im_l = state_l[4];
        let next_re_l = m_re * prev_re_l - m_im * prev_im_l + m_bg * x_l;
        let next_im_l = m_im * prev_re_l + m_re * prev_im_l;
        let next_rc_l = a_rc * state_l[5] + one_minus_rc * a_even_l;

        let hpf_y_r = a_hpf * (state_r[1] + x_r - state_r[0]);
        let lpf_sh_r = a_sh * state_r[2] + one_minus_sh * x_r;
        let shelf_hi_r = x_r - lpf_sh_r;
        let prev_re_r = state_r[3];
        let prev_im_r = state_r[4];
        let next_re_r = m_re * prev_re_r - m_im * prev_im_r + m_bg * x_r;
        let next_im_r = m_im * prev_re_r + m_re * prev_im_r;
        let next_rc_r = a_rc * state_r[5] + one_minus_rc * a_even_r;

        state_l[0] = flush_recursive_state_f32(x_l);
        state_l[1] = flush_recursive_state_f32(hpf_y_l);
        state_l[2] = flush_recursive_state_f32(lpf_sh_l);
        state_l[3] = flush_recursive_state_f32(next_re_l);
        state_l[4] = flush_recursive_state_f32(next_im_l);
        state_l[5] = flush_recursive_state_f32(next_rc_l);

        state_r[0] = flush_recursive_state_f32(x_r);
        state_r[1] = flush_recursive_state_f32(hpf_y_r);
        state_r[2] = flush_recursive_state_f32(lpf_sh_r);
        state_r[3] = flush_recursive_state_f32(next_re_r);
        state_r[4] = flush_recursive_state_f32(next_im_r);
        state_r[5] = flush_recursive_state_f32(next_rc_r);

        let y_l = x_l + g0 * shelf_hi_l + g1 * next_re_l - g2 * next_rc_l;
        let y_r = x_r + g0 * shelf_hi_r + g1 * next_re_r - g2 * next_rc_r;
        (y_l, y_r)
    }

    #[inline(always)]
    pub(crate) fn step_branch_shaper_stereo(
        state_l: &mut [f32; 3],
        state_r: &mut [f32; 3],
        poles: &[f32; 2],
        params: &[f32; 4],
        x_l: f32,
        x_r: f32,
    ) -> ((f32, f32, f32), (f32, f32, f32)) {
        let a_lpf = poles[0];
        let a_hpf = poles[1];
        let drive = params[2];
        let asym = params[3];
        let one_minus_lpf = 1.0 - a_lpf;

        let lpf_y_l = a_lpf * state_l[0] + one_minus_lpf * x_l;
        let lpf_y_r = a_lpf * state_r[0] + one_minus_lpf * x_r;
        let hpf_y_l = a_hpf * (state_l[2] + x_l - state_l[1]);
        let hpf_y_r = a_hpf * (state_r[2] + x_r - state_r[1]);
        state_l[0] = flush_recursive_state_f32(lpf_y_l);
        state_l[1] = flush_recursive_state_f32(x_l);
        state_l[2] = flush_recursive_state_f32(hpf_y_l);
        state_r[0] = flush_recursive_state_f32(lpf_y_r);
        state_r[1] = flush_recursive_state_f32(x_r);
        state_r[2] = flush_recursive_state_f32(hpf_y_r);

        let u_bp_l = ((hpf_y_l - 0.35 * lpf_y_l) * drive).clamp(-1.0, 1.0);
        let u_bp_r = ((hpf_y_r - 0.35 * lpf_y_r) * drive).clamp(-1.0, 1.0);
        let u2_l = u_bp_l * u_bp_l;
        let u2_r = u_bp_r * u_bp_r;
        let nl_bp_l = u_bp_l * (1.5 - 0.5 * u2_l) + asym * u2_l * (1.0 - u2_l);
        let nl_bp_r = u_bp_r * (1.5 - 0.5 * u2_r) + asym * u2_r * (1.0 - u2_r);
        ((nl_bp_l, lpf_y_l, hpf_y_l), (nl_bp_r, lpf_y_r, hpf_y_r))
    }
}
