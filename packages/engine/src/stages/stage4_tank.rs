//! Stage IV: Unitary Spatial Tank with Givens+Householder Mixing (2,080 B Lite RAM).
//!
//! Canonical Reference: `docs/dsp/router_and_stages_topology.md` §2.4 (`CL-08`, `CL-09`, `CL-10`, `CL-11`).

use crate::math::{
    flush_recursive_state_f32,
    fp16::{f32_to_fp16, fp16_to_f32},
};
use crate::operators::op_m::algebraic_unit_rotor_cl03;
use crate::operators::op_t::BallisticDuckingVca;

/// State RAM footprint for Stage IV Lite: strictly 2,080 B.
pub const STAGE4_TANK_RAM_BYTES: usize = 2080;

/// Number of delay lines in Stage IV Lite.
pub const TANK_LITE_LINES: usize = 4;
/// Capacity per line in Lite: 256 samples (512 bytes fp16).
pub const TANK_LINE_CAPACITY: usize = 256;
const TANK_LINE_MASK: usize = TANK_LINE_CAPACITY - 1;

/// Passband-normalized 5 Hz DC blocker (CL-09).
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
            r_dc: 0.995, // 5 Hz at 48 kHz (CL-09)
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

/// Stage IV: Unitary Spatial Tank (2,080 B Lite RAM).
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct Stage4SpatialTank {
    /// 4 delay lines stored as fp16: 4 * 256 * 2 B = 2,048 B.
    pub lines: [[u16; TANK_LINE_CAPACITY]; TANK_LITE_LINES],
    pub write_ptr: usize,
    /// Delay lengths in samples for the 4 lines.
    pub delay_lengths: [usize; TANK_LITE_LINES],
    /// Givens mixing angle psi in radians with slew bound |delta_psi| <= 0.05.
    pub current_psi: f32,
    pub target_psi: f32,
    /// Coercive loop saturation gain and threshold (CL-10, CL-11).
    pub g_loop: f32,
    pub sigma_knee: f32,
    /// 5 Hz DC blockers per line (CL-09).
    pub dc_blockers: [TankDcBlocker; TANK_LITE_LINES],
    /// 1-pole ballistic ducking VCA.
    pub ducking_vca: BallisticDuckingVca,
    /// Unitary freeze mode flag.
    pub freeze: bool,
    pub _pad: [u8; 15],
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
            delay_lengths: [149, 179, 211, 239], // Co-prime lengths
            current_psi: core::f32::consts::FRAC_PI_4, // pi / 4
            target_psi: core::f32::consts::FRAC_PI_4,
            g_loop: 0.85,
            sigma_knee: 1.0,
            dc_blockers: [TankDcBlocker::new(); TANK_LITE_LINES],
            ducking_vca: BallisticDuckingVca::new(),
            freeze: false,
            _pad: [0; 15],
        }
    }

    pub fn reset(&mut self) {
        for l in 0..TANK_LITE_LINES {
            self.lines[l].fill(0);
            self.dc_blockers[l].reset();
        }
        self.write_ptr = 0;
        self.current_psi = self.target_psi;
        self.ducking_vca.reset();
    }

    /// Reconfigures Stage IV sample-rate parameters (e.g. 5 Hz DC blocker pole R_dc and ducking VCA poles) under ZOH.
    pub fn reconfigure_sample_rate(&mut self, zoh: &crate::container::ZohRescaledParams) {
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

    /// Unity-slope normalized cubic coercive saturation loop law (`CL-10`, `CL-11`, `0 FDIV`, `0 FSQRT`).
    ///
    /// Uses pre-inverted `two_over_3v_ceil = 2.0 / (3.0 * v_ceil)` with `v_ceil = 0.985` so
    /// $S'(0) \equiv 1.000000$ and $|S(u)| \le 0.985$ unconditionally.
    #[inline(always)]
    fn coercive_saturate(&self, x: f32) -> f32 {
        const V_CEIL: f32 = 0.985;
        const TWO_OVER_3V_CEIL: f32 = 2.0 / (3.0 * V_CEIL);
        let s_pre = self.g_loop * x * self.sigma_knee;
        let u_norm = (s_pre * TWO_OVER_3V_CEIL).clamp(-1.0, 1.0);
        V_CEIL * u_norm * (1.5 - 0.5 * u_norm * u_norm)
    }

    /// Evaluates Stage IV for stereo input (in_l, in_r).
    ///
    /// Returns stereo wet reverberant output (wet_l, wet_r).
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_stereo(&mut self, in_l: f32, in_r: f32) -> (f32, f32) {
        // Update Givens rotation angle with slew limit
        let d_psi = (self.target_psi - self.current_psi).clamp(-0.05, 0.05);
        let next_psi = self.current_psi + d_psi;
        self.current_psi = flush_recursive_state_f32(next_psi);
        let (cos_psi, sin_psi) = algebraic_unit_rotor_cl03(next_psi);

        // 1. Read delayed samples from the 4 lines
        let mut reads = [0.0f32; TANK_LITE_LINES];
        for l in 0..TANK_LITE_LINES {
            let d_len = self.delay_lengths[l].clamp(1, TANK_LINE_CAPACITY - 1);
            let r_ptr = (self.write_ptr + TANK_LINE_CAPACITY - d_len) & TANK_LINE_MASK;
            reads[l] = fp16_to_f32(self.lines[l][r_ptr]);
        }

        // 2. 5 Hz DC blocker + Coercive-Saturating Lyapunov loop law (CL-09, CL-10)
        let mut loop_sig = [0.0f32; TANK_LITE_LINES];
        for l in 0..TANK_LITE_LINES {
            let dc_clean = self.dc_blockers[l].step(reads[l]);
            loop_sig[l] = if self.freeze {
                dc_clean
            } else {
                self.coercive_saturate(dc_clean)
            };
        }

        // 3. Lossless Givens stereo rotation on pairs (0,1) and (2,3)
        let mut mixed = [0.0f32; TANK_LITE_LINES];
        mixed[0] = cos_psi * loop_sig[0] - sin_psi * loop_sig[1];
        mixed[1] = sin_psi * loop_sig[0] + cos_psi * loop_sig[1];
        mixed[2] = cos_psi * loop_sig[2] - sin_psi * loop_sig[3];
        mixed[3] = sin_psi * loop_sig[2] + cos_psi * loop_sig[3];

        // 4. Householder reflection: H_4 = I - 0.5 * 1 * 1^T
        let sum = mixed[0] + mixed[1] + mixed[2] + mixed[3];
        let sub = 0.5 * sum;
        for l in 0..TANK_LITE_LINES {
            mixed[l] -= sub;
        }

        // 5. Inject inputs into lines (freeze mutes input injection)
        let inj_l = if self.freeze { 0.0 } else { in_l };
        let inj_r = if self.freeze { 0.0 } else { in_r };

        let write_val0 = mixed[0] + inj_l;
        let write_val1 = mixed[1] + inj_r;
        let write_val2 = mixed[2] + inj_l;
        let write_val3 = mixed[3] + inj_r;

        self.lines[0][self.write_ptr] = f32_to_fp16(write_val0);
        self.lines[1][self.write_ptr] = f32_to_fp16(write_val1);
        self.lines[2][self.write_ptr] = f32_to_fp16(write_val2);
        self.lines[3][self.write_ptr] = f32_to_fp16(write_val3);

        self.write_ptr = (self.write_ptr + 1) & TANK_LINE_MASK;

        // 6. Direct stereo tap mix + stereo-linked sidechain ducking VCA
        let wet_l = 0.5 * (reads[0] + reads[2]);
        let wet_r = 0.5 * (reads[1] + reads[3]);
        self.ducking_vca.step_stereo(in_l, in_r, wet_l, wet_r)
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

            // CL-10 / CL-11: coercive_saturate is odd-symmetric and strictly bounded by V_CEIL = 0.985
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
