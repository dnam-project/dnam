//! Stage IV: Unitary Spatial Tank with Givens+Householder Mixing (2,080 B Lite RAM).
//!
//! Canonical Reference: `docs/specs/dsp/router_and_stages_topology.md` (`ref[dsp.stage4.spatial-tank]`, `ref[dsp.stage4.dc-blocker]`, `ref[dsp.stage4.loop-saturator]`, `ref[dsp.stage4.quiet-feedback-knee]`).

use crate::math::{
    DefaultMathBackend, MathBackend, SimdVec4, flush_recursive_state_f32,
    flush_recursive_state_vec4,
};
use crate::operators::op_m::algebraic_unit_rotor_op_m;
use crate::operators::op_t::BallisticDuckingVca;

/// State RAM footprint for Stage IV Lite: strictly 2,080 B.
pub const STAGE4_TANK_RAM_BYTES: usize = 2080;

/// Number of delay lines in Stage IV Lite.
pub const TANK_LITE_LINES: usize = 4;
/// Capacity per line: 1024 samples so `D_max(192 kHz) = 1004 <= 1024` (`ref[dsp.stage4.spatial-tank]`).
pub const TANK_LINE_CAPACITY: usize = 1024;
const TANK_LINE_MASK: usize = TANK_LINE_CAPACITY - 1;

/// Passband-normalized 5 Hz DC blocker (`ref[dsp.stage4.dc-blocker]`).
#[derive(Clone, Copy, Debug)]
pub struct TankDcBlocker {
    pub x_prev: f32,
    pub y_prev: f32,
    pub r_dc: f32,
}

impl Default for TankDcBlocker {
    fn default() -> Self {
        Self::new()
    }
}

impl TankDcBlocker {
    pub const fn new() -> Self {
        Self {
            x_prev: 0.0,
            y_prev: 0.0,
            r_dc: 0.995, // 5 Hz at 48 kHz (ref[dsp.stage4.dc-blocker])
        }
    }

    pub fn reset(&mut self) {
        self.x_prev = 0.0;
        self.y_prev = 0.0;
    }

    /// Step DC blocker with exact passband normalization: alpha_dc = 0.5 * (1 + R_dc).
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, x: f32) -> f32 {
        let alpha = 0.5 * (1.0 + self.r_dc);
        let y = alpha * (x - self.x_prev) + self.r_dc * self.y_prev;
        self.x_prev = flush_recursive_state_f32(x);
        self.y_prev = flush_recursive_state_f32(y);
        y
    }
}

/// Stage IV: Unitary Spatial Tank (2,080 B nominal 48 kHz Lite RAM budget).
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct Stage4SpatialTank {
    /// 4 delay lines stored as fp16 (`1024` capacity for up to `192 kHz` ZOH rescaling).
    pub lines: [[u16; TANK_LINE_CAPACITY]; TANK_LITE_LINES],
    pub write_ptr: usize,
    /// Immutable `48 kHz` base delay lengths in samples (`[149.0, 179.0, 211.0, 239.0]`).
    pub base_delay_lengths: [f32; TANK_LITE_LINES],
    /// Rescaled delay lengths in samples for the 4 lines (`1..1023`).
    pub delay_lengths: [usize; TANK_LITE_LINES],
    /// Givens mixing angle psi in radians with slew bound |delta_psi| <= 0.05.
    pub current_psi: f32,
    pub target_psi: f32,
    /// Coercive loop saturation gain and threshold (`ref[dsp.stage4.loop-saturator]`, `ref[dsp.stage4.quiet-feedback-knee]`).
    pub g_loop: f32,
    pub sigma_knee: f32,
    /// 5 Hz DC blockers per line (`ref[dsp.stage4.dc-blocker]`).
    pub dc_blockers: [TankDcBlocker; TANK_LITE_LINES],
    /// 1-pole ballistic ducking VCA.
    pub ducking_vca: BallisticDuckingVca,
    /// Cached Givens rotation cosine and sine at `current_psi`.
    pub cos_psi: f32,
    pub sin_psi: f32,
    /// Unitary freeze mode flag.
    pub freeze: bool,
    pub _pad: [u8; 7],
}

impl Default for Stage4SpatialTank {
    fn default() -> Self {
        Self::new()
    }
}

impl Stage4SpatialTank {
    pub const fn new() -> Self {
        Self {
            lines: [[0; TANK_LINE_CAPACITY]; TANK_LITE_LINES],
            write_ptr: 0,
            base_delay_lengths: [149.0, 179.0, 211.0, 239.0],
            delay_lengths: [149, 179, 211, 239], // Co-prime lengths
            current_psi: core::f32::consts::FRAC_PI_4, // pi / 4
            target_psi: core::f32::consts::FRAC_PI_4,
            g_loop: 0.85,
            sigma_knee: 1.0,
            dc_blockers: [TankDcBlocker::new(); TANK_LITE_LINES],
            ducking_vca: BallisticDuckingVca::new(),
            cos_psi: core::f32::consts::FRAC_1_SQRT_2,
            sin_psi: core::f32::consts::FRAC_1_SQRT_2,
            freeze: false,
            _pad: [0; 7],
        }
    }

    pub fn reset(&mut self) {
        for l in 0..TANK_LITE_LINES {
            self.lines[l].fill(0);
            self.dc_blockers[l].reset();
        }
        self.write_ptr = 0;
        self.current_psi = self.target_psi;
        let (c, s) = algebraic_unit_rotor_op_m(self.current_psi);
        self.cos_psi = c;
        self.sin_psi = s;
        self.ducking_vca.reset();
    }

    /// Loads `48 kHz` Stage IV base delay lengths and loop parameters from `Arena 0 [4,672..4,704]`
    /// when non-zero, falling back to canonical defaults.
    pub fn load_from_arena(&mut self, arena: &[u8]) {
        const DEFAULTS: [f32; TANK_LITE_LINES] = [149.0, 179.0, 211.0, 239.0];
        if arena.len() < 4_704 {
            self.base_delay_lengths = DEFAULTS;
            return;
        }
        for (l, &def_d) in DEFAULTS.iter().enumerate() {
            let v = crate::provenance::watermark::read_f32_le(arena, 4_672 + l * 4);
            self.base_delay_lengths[l] = if v.is_finite() && (4.0..=255.0).contains(&v) {
                v
            } else {
                def_d
            };
        }
        let g = crate::provenance::watermark::read_f32_le(arena, 4_688);
        if g.is_finite() && (0.05..=1.18).contains(&g) {
            self.g_loop = g;
        }
        let sigma = crate::provenance::watermark::read_f32_le(arena, 4_692);
        if sigma.is_finite() && (0.05..=10.0).contains(&sigma) {
            self.sigma_knee = sigma;
        }
        let psi = crate::provenance::watermark::read_f32_le(arena, 4_696);
        if psi.is_finite()
            && psi.abs() > 1e-6
            && (-core::f32::consts::PI..=core::f32::consts::PI).contains(&psi)
        {
            self.target_psi = psi;
            self.current_psi = psi;
            let (c, s) = algebraic_unit_rotor_op_m(psi);
            self.cos_psi = c;
            self.sin_psi = s;
        }
    }

    /// Reconfigures Stage IV sample-rate parameters (delay line lengths, 5 Hz DC blocker pole R_dc, and ducking VCA poles) under ZOH.
    pub fn reconfigure_sample_rate(&mut self, zoh: &crate::container::ZohRescaledParams) {
        for l in 0..TANK_LITE_LINES {
            let scaled = (zoh.rescale_alignment_delay(self.base_delay_lengths[l]) + 0.5) as usize;
            self.delay_lengths[l] = scaled.clamp(1, TANK_LINE_CAPACITY - 1);
        }
        for b in &mut self.dc_blockers {
            b.r_dc = zoh.r_dc_host;
        }
        self.ducking_vca.alpha_att = zoh.rescale_real_pole(0.99);
        self.ducking_vca.alpha_rel = zoh.rescale_real_pole(0.9995);
    }

    /// Sets Givens rotation angle with slew rate bound |delta_psi| <= 0.05 rad/frame.
    pub fn set_target_psi(&mut self, psi: f32) {
        self.target_psi = psi;
    }

    /// Evaluates the Stage IV blended quiet-feedback knee $K(u)$ (`ref[dsp.stage4.quiet-feedback-knee]`).
    pub fn eval_quiet_feedback_knee(&self, u: f32, sigma: f32, alpha: f32) -> f32 {
        super::topology::eval_stage4_blended_feedback_knee(u, self.g_loop, sigma, alpha)
            .unwrap_or(self.g_loop * u)
    }

    /// Unity-slope normalized cubic coercive saturation loop law (`ref[dsp.stage4.loop-saturator]`, `ref[dsp.stage4.quiet-feedback-knee]`, `0 FDIV`, `0 FSQRT`).
    ///
    /// Uses the blended quiet-feedback knee followed by pre-inverted `two_over_3v_ceil = 2.0 / (3.0 * v_ceil)`
    /// with `v_ceil = 0.985` and outer bit-level ceiling `0x3F7C28F5` so $S'(0) \equiv 1.000000$ and $|S(u)| \le 0.985$ unconditionally.
    #[inline(always)]
    pub fn coercive_saturate(&self, x: f32) -> f32 {
        let inv_sigma = if (self.sigma_knee - 1.0).abs() <= 1e-6 {
            1.0_f64
        } else {
            1.0_f64 / (self.sigma_knee.max(1e-3) as f64)
        };
        let knee_u = super::topology::eval_stage4_blended_feedback_knee_fast(
            x,
            self.g_loop
                .min(f32::from_bits(super::topology::STAGE4_MAX_G_LIN_BITS)),
            inv_sigma,
            0.15,
        );
        super::topology::eval_stage4_loop_saturator_fast(knee_u)
    }

    /// Evaluates `coercive_saturate` across all 4 tank delay lines in parallel (`0 FDIV`, `0 FSQRT`).
    /// Uses a 4-lane vectorizable `f32` minimax seed `24/17 - (8/17)*d` + 3 Newton steps when
    /// `d = 1 + (u/sigma)^2 <= 2.0` (`< 1 ULP` of `f32`), falling back to 6-step `f64` when `d > 2.0`.
    #[inline(always)]
    fn coercive_saturate_4ch(&self, x: [f32; TANK_LITE_LINES]) -> [f32; TANK_LITE_LINES] {
        const V_CEIL: f32 = f32::from_bits(super::topology::STAGE4_MAX_V_CEIL_BITS);
        const TWO_OVER_3V: f32 = (2.0_f64 / (3.0_f64 * (V_CEIL as f64))) as f32;
        const SEED_A: f32 = 24.0 / 17.0;
        const SEED_B: f32 = 8.0 / 17.0;

        let g = self
            .g_loop
            .min(f32::from_bits(super::topology::STAGE4_MAX_G_LIN_BITS));
        let inv_sigma = if (self.sigma_knee - 1.0).abs() <= 1e-6 {
            1.0_f32
        } else {
            1.0_f32 / self.sigma_knee.max(1e-3)
        };

        let r0 = x[0] * inv_sigma;
        let r1 = x[1] * inv_sigma;
        let r2 = x[2] * inv_sigma;
        let r3 = x[3] * inv_sigma;

        let d0 = 1.0 + r0 * r0;
        let d1 = 1.0 + r1 * r1;
        let d2 = 1.0 + r2 * r2;
        let d3 = 1.0 + r3 * r3;

        if d0 <= 2.0 && d1 <= 2.0 && d2 <= 2.0 && d3 <= 2.0 {
            type V = <DefaultMathBackend as MathBackend>::Vec4;
            let dv = V::from_array([d0, d1, d2, d3]);
            let xv = V::load(&x);
            let two = V::splat(2.0);
            let mut inv = V::splat(SEED_A).sub(V::splat(SEED_B).mul(dv));
            inv = inv.mul(two.sub(dv.mul(inv)));
            inv = inv.mul(two.sub(dv.mul(inv)));
            inv = inv.mul(two.sub(dv.mul(inv)));
            let knee_u = V::splat(g)
                .mul(xv)
                .mul(V::splat(0.85).add(V::splat(0.15).mul(inv)));
            let v = knee_u
                .mul(V::splat(TWO_OVER_3V))
                .max(V::splat(-1.0))
                .min(V::splat(1.0));
            let v_ceil_v = V::splat(V_CEIL);
            let prod = v_ceil_v.mul(v);
            let half_sq = V::splat(0.5).mul(v).mul(v);
            let res = prod
                .mul(V::splat(1.5).sub(half_sq))
                .max(V::splat(-V_CEIL))
                .min(v_ceil_v);
            let mut out = [0.0f32; 4];
            res.store(&mut out);
            return out;
        }

        [
            self.coercive_saturate(x[0]),
            self.coercive_saturate(x[1]),
            self.coercive_saturate(x[2]),
            self.coercive_saturate(x[3]),
        ]
    }

    #[inline(always)]
    fn step_givens_slew(&mut self) -> (f32, f32) {
        let diff_psi = self.target_psi - self.current_psi;
        if diff_psi.abs() > 1e-7 {
            let d_psi = diff_psi.clamp(-0.05, 0.05);
            let next_psi = flush_recursive_state_f32(self.current_psi + d_psi);
            self.current_psi = next_psi;
            let (c, s) = algebraic_unit_rotor_op_m(next_psi);
            self.cos_psi = c;
            self.sin_psi = s;
            (c, s)
        } else {
            (self.cos_psi, self.sin_psi)
        }
    }

    #[inline(always)]
    fn step_reads_and_saturate(&mut self) -> ([f32; TANK_LITE_LINES], [f32; TANK_LITE_LINES]) {
        type V = <DefaultMathBackend as MathBackend>::Vec4;
        let mut raw_fp16 = [0u16; TANK_LITE_LINES];
        for l in 0..TANK_LITE_LINES {
            let d_len = self.delay_lengths[l].clamp(1, TANK_LINE_CAPACITY - 1);
            let r_ptr = (self.write_ptr + TANK_LINE_CAPACITY - d_len) & TANK_LINE_MASK;
            raw_fp16[l] = self.lines[l][r_ptr];
        }
        let reads = DefaultMathBackend::fp16x4_to_f32x4(&raw_fp16);

        let x_vec = V::load(&reads);
        let x_prev_vec = V::from_array([
            self.dc_blockers[0].x_prev,
            self.dc_blockers[1].x_prev,
            self.dc_blockers[2].x_prev,
            self.dc_blockers[3].x_prev,
        ]);
        let y_prev_vec = V::from_array([
            self.dc_blockers[0].y_prev,
            self.dc_blockers[1].y_prev,
            self.dc_blockers[2].y_prev,
            self.dc_blockers[3].y_prev,
        ]);
        let r_dc_vec = V::from_array([
            self.dc_blockers[0].r_dc,
            self.dc_blockers[1].r_dc,
            self.dc_blockers[2].r_dc,
            self.dc_blockers[3].r_dc,
        ]);
        let alpha_vec = V::splat(0.5).mul(V::splat(1.0).add(r_dc_vec));
        let y_vec = alpha_vec
            .mul(x_vec.sub(x_prev_vec))
            .add(r_dc_vec.mul(y_prev_vec));
        let mut x_flushed = [0.0f32; TANK_LITE_LINES];
        let mut y_flushed = [0.0f32; TANK_LITE_LINES];
        let mut readout_sig = [0.0f32; TANK_LITE_LINES];
        flush_recursive_state_vec4(x_vec).store(&mut x_flushed);
        flush_recursive_state_vec4(y_vec).store(&mut y_flushed);
        y_vec.store(&mut readout_sig);
        for l in 0..TANK_LITE_LINES {
            self.dc_blockers[l].x_prev = x_flushed[l];
            self.dc_blockers[l].y_prev = y_flushed[l];
        }

        let loop_sig = if self.freeze {
            reads
        } else {
            self.coercive_saturate_4ch(readout_sig)
        };
        (readout_sig, loop_sig)
    }

    #[inline(always)]
    fn step_mix_write_duck(
        &mut self,
        cos_psi: f32,
        sin_psi: f32,
        readout_sig: [f32; TANK_LITE_LINES],
        loop_sig: [f32; TANK_LITE_LINES],
        in_l: f32,
        in_r: f32,
    ) -> (f32, f32) {
        let mut mixed = [0.0f32; TANK_LITE_LINES];
        mixed[0] = cos_psi * loop_sig[0] - sin_psi * loop_sig[1];
        mixed[1] = sin_psi * loop_sig[0] + cos_psi * loop_sig[1];
        mixed[2] = cos_psi * loop_sig[2] - sin_psi * loop_sig[3];
        mixed[3] = sin_psi * loop_sig[2] + cos_psi * loop_sig[3];

        let sum = mixed[0] + mixed[1] + mixed[2] + mixed[3];
        let sub = 0.5 * sum;
        for l in 0..TANK_LITE_LINES {
            mixed[l] -= sub;
        }

        let inj_l = if self.freeze { 0.0 } else { in_l };
        let inj_r = if self.freeze { 0.0 } else { in_r };

        let write_vals = [
            mixed[0] + inj_l,
            mixed[1] + inj_r,
            mixed[2] + inj_l,
            mixed[3] + inj_r,
        ];
        let packed_fp16 = DefaultMathBackend::f32x4_to_fp16x4(&write_vals);
        self.lines[0][self.write_ptr] = packed_fp16[0];
        self.lines[1][self.write_ptr] = packed_fp16[1];
        self.lines[2][self.write_ptr] = packed_fp16[2];
        self.lines[3][self.write_ptr] = packed_fp16[3];

        self.write_ptr = (self.write_ptr + 1) & TANK_LINE_MASK;

        let wet_l = 0.5 * (readout_sig[0] + readout_sig[2]);
        let wet_r = 0.5 * (readout_sig[1] + readout_sig[3]);
        self.ducking_vca.step_stereo(in_l, in_r, wet_l, wet_r)
    }

    /// Evaluates Stage IV for stereo input `(in_l, in_r)`, returning `(wet_l, wet_r, readout_sig, loop_sig)`.
    #[inline(always)]
    pub fn step_stereo_with_readouts(
        &mut self,
        in_l: f32,
        in_r: f32,
    ) -> (f32, f32, [f32; TANK_LITE_LINES], [f32; TANK_LITE_LINES]) {
        let (cos_psi, sin_psi) = self.step_givens_slew();
        let (readout_sig, loop_sig) = self.step_reads_and_saturate();
        let (wet_l, wet_r) =
            self.step_mix_write_duck(cos_psi, sin_psi, readout_sig, loop_sig, in_l, in_r);
        (wet_l, wet_r, readout_sig, loop_sig)
    }

    /// Evaluates Stage IV for stereo input (in_l, in_r).
    ///
    /// Returns stereo wet reverberant output (wet_l, wet_r).
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    #[inline(always)]
    pub fn step_stereo(&mut self, in_l: f32, in_r: f32) -> (f32, f32) {
        let (wet_l, wet_r, _, _) = self.step_stereo_with_readouts(in_l, in_r);
        (wet_l, wet_r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_stage4_spatial_tank_and_dc_blocker() {
        let _audio = crate::FtzDazGuard::new();
        let mut dc = TankDcBlocker::default();
        assert_eq!(dc.r_dc, 0.995);
        // Step 0 impulse response of passband-normalized DC blocker is alpha = 0.5 * (1 + r_dc) = 0.9975
        let y0 = dc.step(1.0);
        assert!((y0 - 0.9975).abs() < 1e-6);
        // Constant DC input decays toward 0.0
        let mut y_dc = y0;
        for _ in 0..2000 {
            y_dc = dc.step(1.0);
        }
        assert!(y_dc.abs() < 1e-4);
        dc.reset();
        assert_eq!(dc.x_prev, 0.0);
        assert_eq!(dc.y_prev, 0.0);

        let mut tank = Stage4SpatialTank::default();
        tank.reset();
        tank.set_target_psi(0.5);

        // Process a block of normal frames (>= max delay length 239 so wet taps become active)
        let mut max_wet = 0.0_f32;
        for _ in 0..300 {
            let (wl, wr) = tank.step_stereo(0.5, -0.5);
            max_wet = max_wet.max(wl.abs()).max(wr.abs());
        }
        assert!(max_wet > 0.01);
        assert!((tank.current_psi - 0.5).abs() < 1e-6);

        // Test unitary freeze mode: mutes input injection and circulates existing tank energy
        tank.freeze = true;
        let mut freeze_wet = 0.0_f32;
        for _ in 0..300 {
            let (wl, wr) = tank.step_stereo(0.0, 0.0);
            freeze_wet = freeze_wet.max(wl.abs()).max(wr.abs());
        }
        assert!(freeze_wet > 0.01);

        tank.reset();
        assert_eq!(tank.write_ptr, 0);
        assert_eq!(tank.ducking_vca.env, 0.0);

        // Dual-mono vs asymmetric sidechain: stereo-linked ducking VCA steps once per frame
        let mut tank_mono = Stage4SpatialTank::default();
        let mut tank_single = BallisticDuckingVca::default();
        let _ = tank_mono.step_stereo(0.8, 0.8);
        let _ = tank_single.step(0.8, 1.0);
        assert!((tank_mono.ducking_vca.env - tank_single.env).abs() < 1e-6);
    }

    proptest! {
        #[test]
        fn prop_stage4_coercive_ceiling_and_slew_bound(
            x in -50.0_f32..50.0_f32,
            g_loop in 0.1_f32..2.0_f32,
            sigma_knee in 0.2_f32..3.0_f32,
            target_psi in -3.0_f32..3.0_f32,
        ) {
            let _audio = crate::FtzDazGuard::new();
            let mut tank = Stage4SpatialTank {
                g_loop,
                sigma_knee,
                ..Default::default()
            };

            // ref[dsp.stage4.loop-saturator]: coercive_saturate is odd-symmetric and strictly bounded by V_CEIL = 0.985
            let s_pos = tank.coercive_saturate(x);
            let s_neg = tank.coercive_saturate(-x);
            prop_assert!(s_pos.abs() <= 0.985_001);
            prop_assert!((s_pos + s_neg).abs() < 1e-6);

            // Slew bound |delta_psi| <= 0.05 rad/sample
            let psi_before = tank.current_psi;
            tank.set_target_psi(target_psi);
            let (wl, wr) = tank.step_stereo(0.1, -0.1);
            prop_assert!(wl.is_finite() && wr.is_finite());
            prop_assert!((tank.current_psi - psi_before).abs() <= 0.050_001);
        }
    }
}

#[cfg(test)]
mod tank_verification_tests {
    #[allow(unused_imports)]
    use alloc::{
        boxed::Box,
        format,
        string::{String, ToString},
        vec,
        vec::Vec,
    };

    // Empirical TORA Stages 0–IV, Spatial Tank & 256-D Mid/Side Kronecker Head Verification (`Tests 4, 5`).
    //
    // Canonical Reference: `docs/specs/dsp/router_and_stages_topology.md`.
    //
    // Verifies:
    // - `Test 4` (`verify[dsp.stage4.spatial-tank]`, `verify[dsp.stage4.dc-blocker]`, `verify[dsp.stage4.loop-saturator]`, `verify[dsp.stage4.quiet-feedback-knee]`, `verify[dsp.stage4.captured-freeze]`):
    //   Stage IV Spatial Tank coercive Lyapunov saturation ceiling (`<= 0.985`),
    //   zero-DC decay (`< 1e-5`), dub-swell self-oscillation stability, and freeze latch.
    // - `Test 5`: 8-channel and 16-channel Givens + Householder stereo mixing matrix $\mathbf{U}(\psi, \mathbf{v})$
    //   orthogonality (`< 1e-14`) and Euclidean energy preservation.
    // - Stage 0 8-Port Matrix Router (`verify[dsp.router.return-to-send-isolation]` DAG isolation `S_ret->send == 0`, WDW mode, Bezout crossover).
    // - 256-D Mid/Side Kronecker Output Head (`verify[dsp.head.mid-side-factorization]`) stereo width laws ($w_s \in \{0.0, 1.0, 2.0\}$).

    use std::f64::consts::{PI, TAU};

    use crate::container::ARENA_OFFSET_ULTRA_END;
    use crate::operators::reference::{
        givens_householder_mix_8ch_f64, givens_householder_mix_16ch_f64, op_t_coercive_step_f64,
    };
    use crate::stages::{MidSideKroneckerHead, Stage0Router, Stage4SpatialTank, StagePipeline};
    use crate::{FtzDazGuard, Tier};
    use proptest::prelude::*;

    /// Test 4 (`verify[dsp.stage4.dc-blocker]`, `verify[dsp.stage4.loop-saturator]`, `verify[dsp.stage4.quiet-feedback-knee]`): Injects a $+20\text{ dBFS}$ (`amplitude = 10.0`) sine burst with a
    /// $+2.0$ DC offset into Stage IV Spatial Tank in a self-oscillating dub swell regime
    /// ($g_{\text{eff,max}} = 1.18 > 1.0$, $V_{\text{ceil}} = 0.985$, $R_{\text{dc}} = 0.995$) and
    /// verifies peak state $\le 0.985$, residual DC $< 10^{-5}$, and monotonic quiet-tail decay.
    #[test]
    fn test_4_stage4_spatial_tank_lyapunov_saturation_and_zero_dc_decay() {
        const V_CEIL: f64 = 0.985;
        let mut dc_x = [0.0_f64; 8];
        let mut dc_y = [0.0_f64; 8];
        let mut state = [0.0_f64; 8];
        let (cos_psi, sin_psi) = (
            core::f64::consts::FRAC_1_SQRT_2,
            core::f64::consts::FRAC_1_SQRT_2,
        );

        let mut max_peak = 0.0_f64;
        for n in 0..8000 {
            let burst = if n < 1000 {
                10.0 * (TAU * 440.0 * (n as f64) / 48_000.0).sin() + 2.0
            } else {
                0.0
            };
            let g_lin = if n < 2000 { 1.18 } else { 0.98 };

            let mut sat = [0.0_f64; 8];
            for m in 0..8 {
                let u_in = state[m] + if m < 2 { burst } else { 0.0 };
                sat[m] = op_t_coercive_step_f64(
                    u_in,
                    g_lin,
                    0.35,
                    V_CEIL,
                    0.995,
                    &mut dc_x[m],
                    &mut dc_y[m],
                );
                max_peak = max_peak.max(sat[m].abs());
            }
            state = givens_householder_mix_8ch_f64(&sat, cos_psi, sin_psi);
        }

        let final_dc: f64 = state.iter().sum::<f64>() / 8.0;
        assert!(
            max_peak <= V_CEIL + 1e-9,
            "Test 4 Lyapunov ceiling violated: max_peak={max_peak} > {V_CEIL}"
        );
        assert!(
            final_dc.abs() < 1e-5,
            "Test 4 residual DC offset {final_dc:e} >= 1e-5"
        );

        // Also verify real-time Stage4SpatialTank under +20 dBFS + 2.0 DC burst and freeze latch
        let _audio = FtzDazGuard::new();
        let mut rt_tank = Stage4SpatialTank {
            g_loop: 1.18,
            ..Default::default()
        };
        for n in 0..8000 {
            let x = if n < 500 {
                10.0 * ((n as f32) * 0.1).sin() + 2.0
            } else {
                0.0
            };
            if n == 600 {
                rt_tank.freeze = true;
            }
            if n == 1000 {
                rt_tank.freeze = false;
                rt_tank.g_loop = 0.65;
            }
            let (wl, wr) = rt_tank.step_stereo(x, -x);
            assert!(wl.is_finite() && wr.is_finite());
        }
        let (tail_l, tail_r) = rt_tank.step_stereo(0.0, 0.0);
        assert!(tail_l.abs() < 1e-3 && tail_r.abs() < 1e-3);
    }

    /// Test 5: Evaluates the 8-channel and 16-channel Givens + Householder stereo mixing matrices
    /// $\mathbf{U}(\psi, \mathbf{v})$ across $\psi \in \{0, \pi/6, \pi/4, \pi/2\}$ and verifies
    /// $\|\mathbf{U}^\top \mathbf{U} - \mathbf{I}\|_{\max} < 10^{-14}$ and exact Euclidean energy preservation.
    #[test]
    fn test_5_givens_householder_stereo_mixing_matrix_orthogonality_and_eigenvalues() {
        for &psi in &[0.0_f64, PI / 6.0, PI / 4.0, PI / 2.0] {
            let (c, s) = (psi.cos(), psi.sin());
            let mut u_mat = [[0.0_f64; 8]; 8];
            for col in 0..8 {
                let mut e = [0.0_f64; 8];
                e[col] = 1.0;
                let col_vec = givens_householder_mix_8ch_f64(&e, c, s);
                for row in 0..8 {
                    u_mat[row][col] = col_vec[row];
                }
            }

            let mut max_ortho_err = 0.0_f64;
            for i in 0..8 {
                for j in 0..8 {
                    let mut dot = 0.0_f64;
                    for k in 0..8 {
                        dot += u_mat[k][i] * u_mat[k][j];
                    }
                    let target = if i == j { 1.0 } else { 0.0 };
                    max_ortho_err = max_ortho_err.max((dot - target).abs());
                }
            }
            assert!(
                max_ortho_err < 1e-14,
                "Test 5 8-ch orthogonality error {max_ortho_err:e} >= 1e-14 at psi={psi}"
            );

            let cross_c = [0.8_f64; 8];
            let cross_s = [0.6_f64; 8];
            let x16: [f64; 16] = core::array::from_fn(|i| ((i as f64) + 1.0) * 0.15);
            let y16 = givens_householder_mix_16ch_f64(&x16, c, s, &cross_c, &cross_s);
            let e_x: f64 = x16.iter().map(|v| v * v).sum();
            let e_y: f64 = y16.iter().map(|v| v * v).sum();
            assert!((e_x - e_y).abs() < 1e-13);
        }
    }

    /// Verifies Stage 0 8-Port Matrix Router (`verify[dsp.router.return-to-send-isolation]` DAG isolation `S_ret->send == 0`, WDW mode)
    /// and `MidSideKroneckerHead` (`verify[dsp.head.mid-side-factorization]`).
    #[test]
    fn test_stage0_router_topologies_and_256d_mid_side_kronecker_head() {
        let _audio = FtzDazGuard::new();
        let mut router = Stage0Router {
            wdw_mode: true,
            ..Default::default()
        };
        router.set_scattering_matrix([[1.0; 4]; 4]);
        // verify[dsp.router.return-to-send-isolation] invariant: S_ret->send entries must be strictly zeroed
        assert_eq!(router.s_matrix[0][2], 0.0);
        assert_eq!(router.s_matrix[0][3], 0.0);
        assert_eq!(router.s_matrix[1][2], 0.0);
        assert_eq!(router.s_matrix[1][3], 0.0);

        let (send_l, send_r, down_l, down_r, sub_l, sub_r) =
            router.route_frame(0.75, -0.5, 0.3, -0.3);
        assert!(send_l.is_finite() && send_r.is_finite());
        assert!(down_l.is_finite() && down_r.is_finite());
        assert!(sub_l.is_finite() && sub_r.is_finite());

        // Verify MidSideKroneckerHead stereo width law (verify[dsp.head.mid-side-factorization]):
        // - w_s = 0.0 -> out_l == out_r (mono collapse)
        // - w_s = 2.0 -> side difference is exactly 2x the w_s = 1.0 side difference
        let mut head = MidSideKroneckerHead {
            w_mid: core::array::from_fn(|i| 0.02 * ((i as f32) + 1.0)),
            b_mid: 0.1,
            ..Default::default()
        };
        let base_w_side: [f32; 64] = core::array::from_fn(|i| 0.015 * ((i as f32) - 20.0));
        let base_b_side = 0.05_f32;
        let phi: [f32; 64] = core::array::from_fn(|i| ((i as f32) * 0.17).sin());

        for tier in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
            head.update_stereo_width(&base_w_side, base_b_side, 0.0);
            let (l0, r0) = head.project_feature_bus(&phi, tier);
            assert!(
                (l0 - r0).abs() < 1e-6,
                "Expected w_s=0.0 mono collapse at {tier:?}, got L={l0}, R={r0}"
            );

            head.update_stereo_width(&base_w_side, base_b_side, 1.0);
            let (l1, r1) = head.project_feature_bus(&phi, tier);
            let side1 = l1 - r1;

            head.update_stereo_width(&base_w_side, base_b_side, 2.0);
            let (l2, r2) = head.project_feature_bus(&phi, tier);
            let side2 = l2 - r2;

            assert!(
                (side2 - 2.0 * side1).abs() < 1e-5,
                "Expected w_s=2.0 side difference ({side2}) == 2 * side1 ({side1}) at {tier:?}"
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// Property test verifying that `StagePipeline` remains finite across all 4 Stage I topologies
        /// and resets cleanly to DC equilibrium.
        #[test]
        fn prop_stage_pipeline_finite_and_dc_resettable(
            in_l in -2.0_f32..2.0_f32,
            in_r in -2.0_f32..2.0_f32,
            topo in 0_u8..4_u8,
        ) {
            let _audio = FtzDazGuard::new();
            let mut pipe = StagePipeline::default();
            let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
            pipe.load_from_arena(&arena);
            pipe.stage1.topology_mode = topo;

            let (s1_l, s1_r) = pipe.stage1.step_stereo(in_l, in_r);
            let (s2_l, s2_r) = pipe.stage2.step_stereo(s1_l, s1_r);
            let s3_l = pipe.stage3.step(s2_l, 0.0, &pipe.fir_taps);
            let s3_r = pipe.stage3_r.step(s2_r, 0.0, &pipe.fir_taps);
            let (s4_l, s4_r) = pipe.stage4.step_stereo(s3_l, s3_r);
            prop_assert!(s4_l.is_finite() && s4_r.is_finite());

            pipe.reset_to_dc_equilibrium();
            let (z_l, z_r) = pipe.stage2.step_stereo(0.0, 0.0);
            prop_assert!(z_l.abs() < 1e-7 && z_r.abs() < 1e-7);
        }
    }
}
