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
pub mod weights;

pub use head::{HEAD_RAM_BYTES, MidSideKroneckerHead};
pub use router::{BezoutCleanCrossover, ROUTER_STATE_RAM_BYTES, ReturnAligner, Stage0Router};
pub use stage1_pre::{STAGE1_PRE_RAM_BYTES, Stage1PreDynamics};
pub use stage2_cascade::{STAGE2_CASCADE_RAM_BYTES, Stage2CommutedCascade, ToneBasisFilters};
pub use stage3_cabinet::{STAGE3_CABINET_RAM_BYTES, Stage3CabinetModals};
pub use stage4_tank::{STAGE4_TANK_RAM_BYTES, Stage4SpatialTank, TankDcBlocker};
pub use weights::{
    ARENA_FULL_BYTES, ARENA_LITE_BYTES, ARENA_PRO_BYTES, ARENA_ULTRA_BYTES, ArenaDirectoryView,
};

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
    pub base_fir_taps: [f32; 16],
    pub fir_taps: [f32; 16],
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
            base_fir_taps: [0.0625; 16],
            fir_taps: [0.0625; 16],
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

    /// Loads per-tier Orthonormal Kautz modal bank parameters and early FIR taps from `Arena 0..3`,
    /// rescaling modal poles, angles, $\mathcal{H}_2$-norm input scalars, and FIR taps under continuous-time ZOH.
    pub fn load_from_arena_with_zoh(
        &mut self,
        arena: &[u8],
        zoh: &crate::container::ZohRescaledParams,
    ) {
        let max_tiers = crate::provenance::watermark::max_tiers_for_arena_len(arena.len());
        for tier in 0..max_tiers {
            for blk in 0..4 {
                let base = crate::provenance::kautz_soa_byte_offset(tier, blk);
                for lane in 0..4 {
                    let r_low = crate::provenance::watermark::read_f32_le(arena, base + lane * 4);
                    let dr =
                        crate::provenance::watermark::read_f32_le(arena, base + (4 + lane) * 4);
                    let theta =
                        crate::provenance::watermark::read_f32_le(arena, base + (8 + lane) * 4);
                    let b_gamma =
                        crate::provenance::watermark::read_f32_le(arena, base + (12 + lane) * 4);
                    let c_re =
                        crate::provenance::watermark::read_f32_le(arena, base + (16 + lane) * 4);
                    let c_im =
                        crate::provenance::watermark::read_f32_le(arena, base + (20 + lane) * 4);
                    let is_active =
                        r_low.abs() > 1e-12 || dr.abs() > 1e-12 || b_gamma.abs() > 1e-12;
                    let (r, theta_eff, b_gamma_eff) = if is_active {
                        let (rl_h, dr_h, th_h, bg_h) =
                            zoh.rescale_op_m_channel(r_low, dr, theta, b_gamma);
                        ((rl_h + 0.5 * dr_h).clamp(0.0, 0.99995), th_h, bg_h)
                    } else {
                        ((r_low + 0.5 * dr).clamp(0.0, 0.9995), theta, b_gamma)
                    };
                    self.kautz_tiers_l[tier][blk].set_pole_with_b_gamma(
                        lane,
                        r,
                        theta_eff,
                        0.0,
                        b_gamma_eff,
                        c_re,
                        c_im,
                    );
                    self.kautz_tiers_r[tier][blk].set_pole_with_b_gamma(
                        lane,
                        r,
                        theta_eff,
                        0.0,
                        b_gamma_eff,
                        c_re,
                        c_im,
                    );
                }
            }
        }
        if arena.len() >= 4_224 {
            let mut has_nonzero = false;
            let mut taps = [0.0f32; 16];
            for (i, tap) in taps.iter_mut().enumerate() {
                let v = crate::provenance::watermark::read_f32_le(arena, 4_160 + i * 4);
                if v.abs() > 1e-12 {
                    has_nonzero = true;
                }
                *tap = v;
            }
            self.base_fir_taps = if has_nonzero { taps } else { [0.0625; 16] };
            let base_taps = self.base_fir_taps;
            zoh.rescale_fir_taps(&base_taps, &mut self.fir_taps);
        }
        self.reset_to_dc_equilibrium();
        self.reconfigure_sample_rate(zoh);
    }

    /// Reconfigures physical stage parameters for host sample rate under continuous-time ZOH.
    pub fn reconfigure_sample_rate(&mut self, zoh: &crate::container::ZohRescaledParams) {
        let align_delay = zoh.rescale_alignment_delay(4.0);
        self.stage0.return_aligner_l.set_delay(align_delay);
        self.stage0.return_aligner_r.set_delay(align_delay);
        self.stage0.sub_crossover_l.reconfigure_sample_rate(zoh);
        self.stage0.sub_crossover_r.reconfigure_sample_rate(zoh);
        self.stage1
            .op_t_pre
            .set_target_delay(zoh.tau_ref_samples_host, true);
        self.stage1.op_t_pre.current_delay = self.stage1.op_t_pre.target_delay;
        self.stage1.op_t_pre.xfade_active = false;
        self.stage2.reconfigure_sample_rate(zoh);
        let base_taps = self.base_fir_taps;
        zoh.rescale_fir_taps(&base_taps, &mut self.fir_taps);
        self.stage4.reconfigure_sample_rate(zoh);
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

        // Populate custom FIR taps at Arena 0 offset 4,160 and Kautz pole at tier 0, blk 0, lane 0
        arena[4_160..4_164].copy_from_slice(&0.5_f32.to_le_bytes());
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
        // Verify DC sum of FIR taps is preserved across ZOH resampling (sum == 0.5)
        let fir_sum: f32 = pipe.fir_taps.iter().sum();
        assert!((fir_sum - 0.5).abs() < 1e-5);
    }
}
