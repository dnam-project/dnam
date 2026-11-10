//! TORA Stages Pipeline Core: Stage 0 8-Port Router & Stages I–IV Topology.
//!
//! Canonical Reference: `docs/dsp/router_and_stages_topology.md` and `docs/overview/tora_architecture.md`.
//!
//! Direct per-sample calls require a caller-held [`crate::FtzDazGuard`] for the
//! enclosing batch; engine block methods provide the guard automatically.

pub mod head;
pub mod router;
pub mod stage1_pre;
pub mod stage2_cascade;
pub mod stage3_cabinet;
pub mod stage4_tank;
pub mod topology;
pub mod weights;

pub use topology::{
    BALLISTIC_ENVELOPE_BANK_TAUS_SEC, GAIN_BOUNDED_FIRST_TAP_64_BITS, ParallelAlignmentMode,
    STAGE4_MAX_G_LIN_BITS, STAGE4_MAX_V_CEIL_BITS, TptLr4Crossover, WdwDryTapMode,
    eval_bounce_6point_lagrange_tap, eval_live_first_tap_difference,
    eval_min_latency_4tap_alignment, eval_stage_modulation_laws, eval_stage4_blended_feedback_knee,
    eval_stage4_loop_saturator, eval_stereo_linked_aux_detector, eval_wdw_routing_outputs,
    pack_stage3_modal_block_features, prepare_ballistic_band_alphas,
    prepare_bounce_6point_lagrange_tap, prepare_kaiser_sinc_alignment_weights,
    prepare_live_first_tap_difference_coeffs, stage3_feature_bus_slot_ranges,
    validate_tensor_directory_slice,
};

pub use head::{HEAD_RAM_BYTES, MidSideKroneckerHead};
pub use router::{BezoutCleanCrossover, ROUTER_STATE_RAM_BYTES, ReturnAligner, Stage0Router};
pub use stage1_pre::{STAGE1_PRE_RAM_BYTES, Stage1PreDynamics};
pub use stage2_cascade::{
    STAGE2_CASCADE_RAM_BYTES, Stage2Backend, Stage2CascadeFrameOut, Stage2CommutedCascade,
    ToneBasisFilters, conv6_taps_1group_vec4, conv6_taps_2groups_vec4, conv6_taps_3groups_vec4,
    conv6_taps_4groups_vec4,
};
pub use stage3_cabinet::{
    STAGE3_CABINET_RAM_BYTES, Stage3Backend, Stage3CabinetModals, dot16_stereo_vec4,
};
pub use stage4_tank::{STAGE4_TANK_RAM_BYTES, Stage4SpatialTank, TankDcBlocker};
pub use weights::{
    ARENA_FULL_BYTES, ARENA_LITE_BYTES, ARENA_PRO_BYTES, ARENA_ULTRA_BYTES, ArenaDirectoryView,
};

/// Composite SIMD backend trait for the TORA Stages 0–IV pipeline.
pub trait StageBackend:
    crate::math::MathBackend
    + crate::operators::op_m::OpMBackend
    + crate::operators::op_n::OpNBackend<Self::Vec4>
    + crate::operators::op_t::OpTBackend<Self::Vec4>
    + Stage2Backend<Self::Vec4>
    + Stage3Backend
{
}

impl<B> StageBackend for B where
    B: crate::math::MathBackend
        + crate::operators::op_m::OpMBackend
        + crate::operators::op_n::OpNBackend<B::Vec4>
        + crate::operators::op_t::OpTBackend<B::Vec4>
        + Stage2Backend<B::Vec4>
        + Stage3Backend
{
}

/// 3 kHz Control Plane state RAM footprint.
pub const CONTROL_PLANE_RAM_BYTES: usize = 256;

/// Total Lite State RAM: strictly 8,656 B (8.45 KB, 24.52% headroom under 11.2 KB L1D budget).
pub const TOTAL_LITE_STATE_RAM_BYTES: usize = ROUTER_STATE_RAM_BYTES
    + STAGE1_PRE_RAM_BYTES
    + STAGE2_CASCADE_RAM_BYTES
    + STAGE3_CABINET_RAM_BYTES
    + STAGE4_TANK_RAM_BYTES
    + HEAD_RAM_BYTES
    + CONTROL_PLANE_RAM_BYTES;

const _: () = assert!(TOTAL_LITE_STATE_RAM_BYTES == 8656);

/// TORA Multi-Stage Rig Pipeline combining Stages 0 through IV and Mid/Side Head.
#[derive(Clone, Debug)]
pub struct StagePipeline {
    pub stage0: Stage0Router,
    pub stage1: Stage1PreDynamics,
    pub stage2: Stage2CommutedCascade,
    pub stage3: Stage3CabinetModals,
    pub stage3_r: Stage3CabinetModals,
    pub kautz_tiers_l: [[crate::operators::op_m::KautzModalBank<4>; 4]; 4],
    pub kautz_tiers_r: [[crate::operators::op_m::KautzModalBank<4>; 4]; 4],
    pub kautz_base_r_low: [[[f32; 4]; 4]; 4],
    pub kautz_base_dr: [[[f32; 4]; 4]; 4],
    pub kautz_base_theta: [[[f32; 4]; 4]; 4],
    pub kautz_base_b_gamma: [[[f32; 4]; 4]; 4],
    pub kautz_host_r_low: [[[f32; 4]; 4]; 4],
    pub kautz_host_dr: [[[f32; 4]; 4]; 4],
    pub kautz_host_theta: [[[f32; 4]; 4]; 4],
    pub kautz_host_cos: [[[f32; 4]; 4]; 4],
    pub kautz_host_sin: [[[f32; 4]; 4]; 4],
    pub last_dtheta_kautz: f32,
    pub last_g_kautz: f32,
    pub base_fir_taps: [f32; 16],
    pub fir_taps: [f32; 16],
    pub base_fir_taps_l: [[f32; 16]; 4],
    pub base_fir_taps_r: [[f32; 16]; 4],
    pub fir_taps_l: [[f32; 16]; 4],
    pub fir_taps_r: [[f32; 16]; 4],
    pub stage4: Stage4SpatialTank,
    pub head: MidSideKroneckerHead,
}

impl Default for StagePipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl StagePipeline {
    pub const fn new() -> Self {
        const EMPTY_BANK: crate::operators::op_m::KautzModalBank<4> =
            crate::operators::op_m::KautzModalBank::new();
        const EMPTY_TIER: [crate::operators::op_m::KautzModalBank<4>; 4] = [EMPTY_BANK; 4];
        Self {
            stage0: Stage0Router::new(),
            stage1: Stage1PreDynamics::new(),
            stage2: Stage2CommutedCascade::new(),
            stage3: Stage3CabinetModals::new(),
            stage3_r: Stage3CabinetModals::new(),
            kautz_tiers_l: [EMPTY_TIER; 4],
            kautz_tiers_r: [EMPTY_TIER; 4],
            kautz_base_r_low: [[[0.0; 4]; 4]; 4],
            kautz_base_dr: [[[0.0; 4]; 4]; 4],
            kautz_base_theta: [[[0.0; 4]; 4]; 4],
            kautz_base_b_gamma: [[[0.0; 4]; 4]; 4],
            kautz_host_r_low: [[[0.0; 4]; 4]; 4],
            kautz_host_dr: [[[0.0; 4]; 4]; 4],
            kautz_host_theta: [[[0.0; 4]; 4]; 4],
            kautz_host_cos: [[[1.0; 4]; 4]; 4],
            kautz_host_sin: [[[0.0; 4]; 4]; 4],
            last_dtheta_kautz: 0.0,
            last_g_kautz: 0.5,
            base_fir_taps: [0.0625; 16],
            fir_taps: [0.0625; 16],
            base_fir_taps_l: [[0.0625; 16], [0.0; 16], [0.0; 16], [0.0; 16]],
            base_fir_taps_r: [[0.0625; 16], [0.0; 16], [0.0; 16], [0.0; 16]],
            fir_taps_l: [[0.0625; 16], [0.0; 16], [0.0; 16], [0.0; 16]],
            fir_taps_r: [[0.0625; 16], [0.0; 16], [0.0; 16], [0.0; 16]],
            stage4: Stage4SpatialTank::new(),
            head: MidSideKroneckerHead::new(),
        }
    }

    pub fn reset(&mut self) {
        self.stage0.reset();
        self.stage1.reset();
        self.stage2.reset();
        self.stage3.reset();
        self.stage3_r.reset();
        for t in 0..4 {
            for b in 0..4 {
                self.kautz_tiers_l[t][b].reset();
                self.kautz_tiers_r[t][b].reset();
            }
        }
        self.stage4.reset();
        self.head.reset();
    }

    /// Resets all stage states to analytical DC resting equilibrium fixed points.
    /// Initializes both commuted LTI filter states and the Op-N PolyADAA states.
    pub fn reset_to_dc_equilibrium(&mut self) {
        self.stage0.reset();
        self.stage1.reset();
        self.stage2.init_dc_equilibrium(0.0, 1.0, 0.25);
        self.stage3.reset();
        self.stage3_r.reset();
        for t in 0..4 {
            for b in 0..4 {
                self.kautz_tiers_l[t][b].reset();
                self.kautz_tiers_r[t][b].reset();
            }
        }
        self.stage4.reset();
        self.head.reset();
    }

    /// Loads per-tier Orthonormal Kautz modal bank parameters and early FIR taps from `Arena 0..3`.
    pub fn load_from_arena(&mut self, arena: &[u8]) {
        let zoh_id = crate::container::ZohRescaledParams::new(48_000, 48_000);
        self.load_from_arena_with_zoh(arena, &zoh_id);
    }

    /// Loads per-tier Orthonormal Kautz modal bank parameters, early FIR taps, and Stage 0–IV base
    /// parameters from `Arena 0..3`, rescaling modal poles, angles, $\mathcal{H}_2$-norm input scalars,
    /// and FIR taps under continuous-time ZOH without resetting recursive state (`s_re, s_im`, delay lines).
    pub fn load_kautz_and_fir_from_arena_with_zoh(
        &mut self,
        arena: &[u8],
        zoh: &crate::container::ZohRescaledParams,
    ) {
        if arena.len() >= 132 {
            let fc = crate::provenance::watermark::read_f32_le(arena, 128);
            if fc.is_finite() && (20.0..=1000.0).contains(&fc) {
                self.stage0.set_crossover_hz(fc);
            }
        }
        self.stage1.load_from_arena(arena);
        self.stage2.load_from_arena(arena);
        self.stage4.load_from_arena(arena);
        self.last_dtheta_kautz = 0.0;
        self.last_g_kautz = 0.5;

        let max_tiers = crate::provenance::watermark::max_tiers_for_arena_len(arena.len());
        for tier in 0..max_tiers {
            let n_blocks = crate::provenance::STAGE3_KAUTZ_BLOCKS_PER_TIER[tier];
            for blk in 0..n_blocks {
                let base = crate::provenance::kautz_soa_byte_offset(tier, blk);
                for lane in 0..4 {
                    let r_low = crate::provenance::watermark::read_f32_le(arena, base + lane * 4);
                    let dr =
                        crate::provenance::watermark::read_f32_le(arena, base + (4 + lane) * 4);
                    let theta =
                        crate::provenance::watermark::read_f32_le(arena, base + (8 + lane) * 4);
                    let b_gamma =
                        crate::provenance::watermark::read_f32_le(arena, base + (12 + lane) * 4);
                    let c_re_l =
                        crate::provenance::watermark::read_f32_le(arena, base + (16 + lane) * 4);
                    let c_im_l =
                        crate::provenance::watermark::read_f32_le(arena, base + (20 + lane) * 4);
                    let raw_c_re_r =
                        crate::provenance::watermark::read_f32_le(arena, base + (24 + lane) * 4);
                    let raw_c_im_r =
                        crate::provenance::watermark::read_f32_le(arena, base + (28 + lane) * 4);
                    let (c_re_r, c_im_r) = if raw_c_re_r.abs() <= 1e-12 && raw_c_im_r.abs() <= 1e-12
                    {
                        (c_re_l, c_im_l)
                    } else {
                        (raw_c_re_r, raw_c_im_r)
                    };
                    self.kautz_base_r_low[tier][blk][lane] = r_low;
                    self.kautz_base_dr[tier][blk][lane] = dr;
                    self.kautz_base_theta[tier][blk][lane] = theta;
                    self.kautz_base_b_gamma[tier][blk][lane] = b_gamma;
                    let is_active =
                        r_low.abs() > 1e-12 || dr.abs() > 1e-12 || b_gamma.abs() > 1e-12;
                    let (r, theta_eff, b_gamma_eff) = if is_active {
                        let (rl_h, dr_h, th_h, bg_h) =
                            zoh.rescale_op_m_channel(r_low, dr, theta, b_gamma);
                        self.kautz_host_r_low[tier][blk][lane] = rl_h;
                        self.kautz_host_dr[tier][blk][lane] = dr_h;
                        self.kautz_host_theta[tier][blk][lane] = th_h;
                        ((rl_h + 0.5 * dr_h).clamp(0.0, 0.99995), th_h, bg_h)
                    } else {
                        self.kautz_host_r_low[tier][blk][lane] = r_low;
                        self.kautz_host_dr[tier][blk][lane] = dr;
                        self.kautz_host_theta[tier][blk][lane] = theta;
                        ((r_low + 0.5 * dr).clamp(0.0, 0.9995), theta, b_gamma)
                    };
                    let (ch, sh) = crate::operators::op_m::algebraic_unit_rotor_op_m(
                        theta_eff.clamp(0.0, core::f32::consts::PI - 1e-4),
                    );
                    self.kautz_host_cos[tier][blk][lane] = ch;
                    self.kautz_host_sin[tier][blk][lane] = sh;
                    self.kautz_tiers_l[tier][blk].set_pole_with_b_gamma(
                        lane,
                        r,
                        theta_eff,
                        0.0,
                        b_gamma_eff,
                        c_re_l,
                        c_im_l,
                    );
                    self.kautz_tiers_r[tier][blk].set_pole_with_b_gamma(
                        lane,
                        r,
                        theta_eff,
                        0.0,
                        b_gamma_eff,
                        c_re_r,
                        c_im_r,
                    );
                }
            }
            for blk in n_blocks..4 {
                self.kautz_base_r_low[tier][blk] = [0.0; 4];
                self.kautz_base_dr[tier][blk] = [0.0; 4];
                self.kautz_base_theta[tier][blk] = [0.0; 4];
                self.kautz_base_b_gamma[tier][blk] = [0.0; 4];
                self.kautz_host_r_low[tier][blk] = [0.0; 4];
                self.kautz_host_dr[tier][blk] = [0.0; 4];
                self.kautz_host_theta[tier][blk] = [0.0; 4];
                self.kautz_host_cos[tier][blk] = [1.0; 4];
                self.kautz_host_sin[tier][blk] = [0.0; 4];
                for lane in 0..4 {
                    self.kautz_tiers_l[tier][blk]
                        .set_pole_with_b_gamma(lane, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                    self.kautz_tiers_r[tier][blk]
                        .set_pole_with_b_gamma(lane, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                }
            }

            let fir_base = crate::provenance::STAGE3_EARLY_FIR_BASES[tier];
            if arena.len() >= fir_base + 128 {
                let mut has_l = false;
                let mut has_r = false;
                let mut taps_l = [0.0f32; 16];
                let mut taps_r = [0.0f32; 16];
                for i in 0..16 {
                    let vl = crate::provenance::watermark::read_f32_le(arena, fir_base + i * 4);
                    let vr =
                        crate::provenance::watermark::read_f32_le(arena, fir_base + 64 + i * 4);
                    if vl.abs() > 1e-12 {
                        has_l = true;
                    }
                    if vr.abs() > 1e-12 {
                        has_r = true;
                    }
                    taps_l[i] = vl;
                    taps_r[i] = vr;
                }
                let default_taps = if tier == 0 { [0.0625; 16] } else { [0.0; 16] };
                let eff_l = if has_l { taps_l } else { default_taps };
                let eff_r = if has_r {
                    taps_r
                } else if has_l {
                    taps_l
                } else {
                    default_taps
                };
                self.base_fir_taps_l[tier] = eff_l;
                self.base_fir_taps_r[tier] = eff_r;
                zoh.rescale_fir_taps(&eff_l, &mut self.fir_taps_l[tier]);
                zoh.rescale_fir_taps(&eff_r, &mut self.fir_taps_r[tier]);
            }
        }
        self.base_fir_taps = self.base_fir_taps_l[0];
        self.fir_taps = self.fir_taps_l[0];
    }

    /// Loads per-tier Orthonormal Kautz modal bank parameters and early FIR taps from `Arena 0..3`,
    /// rescaling modal poles, angles, $\mathcal{H}_2$-norm input scalars, and FIR taps under continuous-time ZOH.
    pub fn load_from_arena_with_zoh(
        &mut self,
        arena: &[u8],
        zoh: &crate::container::ZohRescaledParams,
    ) {
        self.load_kautz_and_fir_from_arena_with_zoh(arena, zoh);
        self.reset_to_dc_equilibrium();
        self.reconfigure_sample_rate(zoh);
    }

    /// Reconfigures physical stage parameters for host sample rate under continuous-time ZOH.
    pub fn reconfigure_sample_rate(&mut self, zoh: &crate::container::ZohRescaledParams) {
        self.stage0.reconfigure_sample_rate(zoh);
        self.stage1.reconfigure_sample_rate(zoh);
        self.stage2.reconfigure_sample_rate(zoh);
        self.stage3.reconfigure_sample_rate(zoh);
        self.stage3_r.reconfigure_sample_rate(zoh);
        self.last_dtheta_kautz = 0.0;
        self.last_g_kautz = 0.5;
        for tier in 0..4 {
            let n_blocks = crate::provenance::STAGE3_KAUTZ_BLOCKS_PER_TIER[tier];
            for blk in 0..n_blocks {
                for lane in 0..4 {
                    let r_low = self.kautz_base_r_low[tier][blk][lane];
                    let dr = self.kautz_base_dr[tier][blk][lane];
                    let theta = self.kautz_base_theta[tier][blk][lane];
                    let b_gamma = self.kautz_base_b_gamma[tier][blk][lane];
                    if r_low.abs() > 1e-12 || dr.abs() > 1e-12 || b_gamma.abs() > 1e-12 {
                        let (rl_h, dr_h, th_h, bg_h) =
                            zoh.rescale_op_m_channel(r_low, dr, theta, b_gamma);
                        self.kautz_host_r_low[tier][blk][lane] = rl_h;
                        self.kautz_host_dr[tier][blk][lane] = dr_h;
                        self.kautz_host_theta[tier][blk][lane] = th_h;
                        let (ch, sh) = crate::operators::op_m::algebraic_unit_rotor_op_m(
                            th_h.clamp(0.0, core::f32::consts::PI - 1e-4),
                        );
                        self.kautz_host_cos[tier][blk][lane] = ch;
                        self.kautz_host_sin[tier][blk][lane] = sh;
                        let r = (rl_h + 0.5 * dr_h).clamp(0.0, 0.99995);
                        let c_re_l = self.kautz_tiers_l[tier][blk].c_re[lane];
                        let c_im_l = self.kautz_tiers_l[tier][blk].c_im[lane];
                        let c_re_r = self.kautz_tiers_r[tier][blk].c_re[lane];
                        let c_im_r = self.kautz_tiers_r[tier][blk].c_im[lane];
                        self.kautz_tiers_l[tier][blk]
                            .set_pole_with_b_gamma(lane, r, th_h, 0.0, bg_h, c_re_l, c_im_l);
                        self.kautz_tiers_r[tier][blk]
                            .set_pole_with_b_gamma(lane, r, th_h, 0.0, bg_h, c_re_r, c_im_r);
                    }
                }
            }
            let bl = self.base_fir_taps_l[tier];
            let br = self.base_fir_taps_r[tier];
            zoh.rescale_fir_taps(&bl, &mut self.fir_taps_l[tier]);
            zoh.rescale_fir_taps(&br, &mut self.fir_taps_r[tier]);
        }
        let base_taps = self.base_fir_taps;
        zoh.rescale_fir_taps(&base_taps, &mut self.fir_taps);
        self.stage4.reconfigure_sample_rate(zoh);
    }

    /// Retunes Stage III Orthonormal Kautz modal poles at the 3 kHz control rate (`0 FSQRT/FDIV/libm`).
    #[inline]
    pub fn retune_kautz_formant_3khz(
        &mut self,
        active_tiers: usize,
        dtheta_kautz: f32,
        g_kautz: f32,
    ) {
        use crate::math::{DefaultMathBackend, MathBackend, SimdVec4};
        type V = <DefaultMathBackend as MathBackend>::Vec4;

        let g_clamped = g_kautz.clamp(0.0, 1.0);
        if (dtheta_kautz - self.last_dtheta_kautz).abs() <= 1e-9
            && (g_clamped - self.last_g_kautz).abs() <= 1e-9
        {
            return;
        }
        self.last_dtheta_kautz = dtheta_kautz;
        self.last_g_kautz = g_clamped;

        const MAX_THETA: f32 = core::f32::consts::PI - 1e-4;
        let has_dtheta = dtheta_kautz.abs() > 1e-9;
        let (cos_d, sin_d) = if has_dtheta {
            crate::operators::op_m::algebraic_unit_rotor_op_m(dtheta_kautz)
        } else {
            (1.0, 0.0)
        };
        let r_sweep_cap = (0.9950 - 0.15 * dtheta_kautz.abs()).clamp(0.0, 0.9950);

        let g_v = V::splat(g_clamped);
        let zero_v = V::zero();
        let max_r_v = V::splat(0.99995);
        let r_sweep_v = V::splat(r_sweep_cap);
        let cos_d_v = V::splat(cos_d);
        let sin_d_v = V::splat(sin_d);
        let one_v = V::splat(1.0);
        let one_pt_five_v = V::splat(1.5);
        let half_v = V::splat(0.5);
        let dth_v = V::splat(dtheta_kautz);
        let max_th_v = V::splat(MAX_THETA);

        let max_t = active_tiers.min(4);
        for tier in 0..max_t {
            let n_blocks = crate::provenance::STAGE3_KAUTZ_BLOCKS_PER_TIER[tier];
            for blk in 0..n_blocks {
                let rl_v = V::load(&self.kautz_host_r_low[tier][blk]);
                let dr_v = V::load(&self.kautz_host_dr[tier][blk]);
                let r_base_v = rl_v.fma(g_v, dr_v).max(zero_v).min(max_r_v);
                let r_eff_v = if has_dtheta {
                    r_base_v.min(r_sweep_v)
                } else {
                    r_base_v
                };
                let ch_v = V::load(&self.kautz_host_cos[tier][blk]);
                let sh_v = V::load(&self.kautz_host_sin[tier][blk]);
                let (cos_v, sin_v) = if !has_dtheta {
                    (ch_v, sh_v)
                } else {
                    let th_v = V::load(&self.kautz_host_theta[tier][blk]);
                    let th_u_v = th_v.add(dth_v);
                    let clamped_v = th_u_v.max(zero_v).min(max_th_v);
                    if th_u_v.sub(clamped_v).abs().horizontal_sum() == 0.0 {
                        let c_raw = ch_v.mul(cos_d_v).fms(sh_v, sin_d_v);
                        let s_raw = sh_v.mul(cos_d_v).fma(ch_v, sin_d_v);
                        let norm_sq = c_raw.mul(c_raw).fma(s_raw, s_raw);
                        let scale = one_pt_five_v.fms(half_v, norm_sq.max(one_v));
                        (c_raw.mul(scale), s_raw.mul(scale))
                    } else {
                        let mut c_arr = [0.0f32; 4];
                        let mut s_arr = [0.0f32; 4];
                        for lane in 0..4 {
                            let th_u = self.kautz_host_theta[tier][blk][lane] + dtheta_kautz;
                            let (c_t, s_t) = crate::operators::op_m::algebraic_unit_rotor_op_m(
                                th_u.clamp(0.0, MAX_THETA),
                            );
                            c_arr[lane] = c_t;
                            s_arr[lane] = s_t;
                        }
                        (V::load(&c_arr), V::load(&s_arr))
                    }
                };
                let l_re_v = r_eff_v.mul(cos_v);
                let l_im_v = r_eff_v.mul(sin_v);
                l_re_v.store(&mut self.kautz_tiers_l[tier][blk].lambda_re);
                l_im_v.store(&mut self.kautz_tiers_l[tier][blk].lambda_im);
                l_re_v.store(&mut self.kautz_tiers_r[tier][blk].lambda_re);
                l_im_v.store(&mut self.kautz_tiers_r[tier][blk].lambda_im);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{ARENA_OFFSET_ULTRA_END, ZohRescaledParams};
    use alloc::vec;

    #[test]
    fn test_stage_pipeline_lifecycle() {
        assert_eq!(TOTAL_LITE_STATE_RAM_BYTES, 8656);
        let mut pipe = StagePipeline::default();
        pipe.reset();
        pipe.reset_to_dc_equilibrium();

        // Verify load_from_arena with default (zero) FIR taps falls back to [0.0625; 16]
        let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        pipe.load_from_arena(&arena);
        assert_eq!(pipe.fir_taps, [0.0625; 16]);

        // Populate custom FIR taps at STAGE3_EARLY_FIR_BASES[0] (4,288) and Kautz pole at tier 0, blk 0, lane 0
        let fir0 = crate::provenance::STAGE3_EARLY_FIR_BASES[0];
        arena[fir0..fir0 + 4].copy_from_slice(&0.5_f32.to_le_bytes());
        let k_base = crate::provenance::kautz_soa_byte_offset(0, 0);
        arena[k_base..k_base + 4].copy_from_slice(&0.90_f32.to_le_bytes());
        arena[k_base + 16..k_base + 20].copy_from_slice(&0.04_f32.to_le_bytes());
        arena[k_base + 32..k_base + 36].copy_from_slice(&0.25_f32.to_le_bytes());
        arena[k_base + 48..k_base + 52].copy_from_slice(&0.40_f32.to_le_bytes());
        arena[k_base + 64..k_base + 68].copy_from_slice(&0.30_f32.to_le_bytes());
        arena[k_base + 80..k_base + 84].copy_from_slice(&(-0.10_f32).to_le_bytes());

        pipe.load_from_arena(&arena);
        assert_eq!(pipe.fir_taps[0], 0.5);
        assert!((pipe.kautz_tiers_l[0][0].b_gamma[0] - 0.40).abs() < 1e-6);
        assert!((pipe.kautz_tiers_r[0][0].c_re[0] - 0.30).abs() < 1e-6);

        let zoh = ZohRescaledParams::compute(48_000, 96_000, 0.9993, 0.995, 100.0, 128).unwrap();
        pipe.load_from_arena_with_zoh(&arena, &zoh);
        assert!((pipe.stage0.return_aligner_l.delay_samples - 8.0).abs() < 1e-4);
        assert!((pipe.stage1.op_t_pre.current_delay - 200.0).abs() < 1e-4);
        assert!((pipe.stage2.r_ac - zoh.r_ac_host).abs() < 1e-6);
        assert!((pipe.stage2.hyst_diff_scale - 2.0).abs() < 1e-6);
        assert!(
            (pipe.stage2.tone_basis_l.alpha_80 - zoh.rescale_real_pole(ToneBasisFilters::ALPHA_80))
                .abs()
                < 1e-6
        );
        assert!((pipe.stage4.dc_blockers[0].r_dc - zoh.r_dc_host).abs() < 1e-6);
        assert_eq!(pipe.stage4.delay_lengths[0], 298);
        assert_eq!(pipe.stage3.onset_delay_samples, 32);
        // Verify DC sum of FIR taps is preserved across ZOH resampling (sum == 0.5)
        let fir_sum: f32 = pipe.fir_taps.iter().sum();
        assert!((fir_sum - 0.5).abs() < 1e-5);
    }
}
