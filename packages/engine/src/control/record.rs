//! Unified 128-Byte `ControlChannelRecord[0..P]` Table & O(1) 64-Byte Latent Accumulator.
//!
//! Canonical Reference: `docs/specs/dsp/parametric_control_bus.md` (`ref[dsp.ctrl.accum.cp1-scaling]`, `ref[dsp.ctrl.accum.anova-fm]`).

use core::mem::{align_of, size_of};

use crate::math::flush_recursive_state_f32;

use super::descriptor::ControlClampDescriptor;

/// Cache Line 1 (64 B): ControlLatentProjection inside [`ControlChannelRecord`].
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControlLatentProjection {
    /// Additive projection W_main,p (8 x f32 = 32 B) into 8-D static latent z_main.
    pub w_main: [f32; 8],
    /// Rank-4/8 FM low-rank interaction factor v_p (8 x f32 = 32 B).
    pub v_factors: [f32; 8],
}

const _: () = assert!(size_of::<ControlLatentProjection>() == 64);
const _: () = assert!(align_of::<ControlLatentProjection>() == 64);

impl Default for ControlLatentProjection {
    fn default() -> Self {
        Self::new()
    }
}

impl ControlLatentProjection {
    pub const fn new() -> Self {
        Self {
            w_main: [0.0; 8],
            v_factors: [0.0; 8],
        }
    }

    /// Serializes the 64-byte [`ControlLatentProjection`] into little-endian bytes.
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        for (i, &w) in self.w_main.iter().enumerate() {
            out[i * 4..(i + 1) * 4].copy_from_slice(&w.to_le_bytes());
        }
        for (i, &v) in self.v_factors.iter().enumerate() {
            out[32 + i * 4..32 + (i + 1) * 4].copy_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Deserializes a 64-byte [`ControlLatentProjection`] from little-endian bytes.
    pub fn from_bytes(bytes: &[u8; 64]) -> Self {
        let mut w_main = [0.0_f32; 8];
        let mut v_factors = [0.0_f32; 8];
        for i in 0..8 {
            w_main[i] = f32::from_le_bytes([
                bytes[i * 4],
                bytes[i * 4 + 1],
                bytes[i * 4 + 2],
                bytes[i * 4 + 3],
            ]);
            v_factors[i] = f32::from_le_bytes([
                bytes[32 + i * 4],
                bytes[32 + i * 4 + 1],
                bytes[32 + i * 4 + 2],
                bytes[32 + i * 4 + 3],
            ]);
        }
        Self { w_main, v_factors }
    }
}

/// Unified 128-byte ControlChannelRecord framing between JSON header and tensor arena.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControlChannelRecord {
    /// Cache Line 0 (64 B): Control normalization and safe clamp descriptor.
    pub clamp: ControlClampDescriptor,
    /// Cache Line 1 (64 B): Additive projection and FM interaction factors.
    pub projection: ControlLatentProjection,
}

const _: () = assert!(size_of::<ControlChannelRecord>() == 128);
const _: () = assert!(align_of::<ControlChannelRecord>() == 64);

impl Default for ControlChannelRecord {
    fn default() -> Self {
        Self::new()
    }
}

impl ControlChannelRecord {
    pub const fn new() -> Self {
        Self {
            clamp: ControlClampDescriptor::empty(),
            projection: ControlLatentProjection::new(),
        }
    }

    /// Serializes the 128-byte [`ControlChannelRecord`] into little-endian bytes.
    pub fn to_bytes(&self) -> [u8; 128] {
        let mut out = [0u8; 128];
        out[0..64].copy_from_slice(&self.clamp.to_bytes());
        out[64..128].copy_from_slice(&self.projection.to_bytes());
        out
    }

    /// Deserializes a 128-byte [`ControlChannelRecord`] from little-endian bytes.
    pub fn from_bytes(bytes: &[u8; 128]) -> Self {
        let mut cl0 = [0u8; 64];
        let mut cl1 = [0u8; 64];
        cl0.copy_from_slice(&bytes[0..64]);
        cl1.copy_from_slice(&bytes[64..128]);
        Self {
            clamp: ControlClampDescriptor::from_bytes(&cl0),
            projection: ControlLatentProjection::from_bytes(&cl1),
        }
    }
}

/// O(1) 64-Byte Latent Accumulator S_lat = (z_main, s_vx, s_v2x2).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatentAccumulator64B {
    /// 8-D Main static latent coordinates (32 B).
    pub z_main: [f32; 8],
    /// 4-D FM linear sum accumulator sum_i v_i * u_i (16 B).
    pub s_vx: [f32; 4],
    /// 4-D FM quadratic sum accumulator sum_i v_i^2 * u_i^2 (16 B).
    pub s_v2x2: [f32; 4],
}

const _: () = assert!(size_of::<LatentAccumulator64B>() == 64);
const _: () = assert!(align_of::<LatentAccumulator64B>() == 64);

impl Default for LatentAccumulator64B {
    fn default() -> Self {
        Self::new()
    }
}

impl LatentAccumulator64B {
    pub const fn new() -> Self {
        Self {
            z_main: [0.0; 8],
            s_vx: [0.0; 4],
            s_v2x2: [0.0; 4],
        }
    }

    pub fn reset(&mut self) {
        self.z_main.fill(0.0);
        self.s_vx.fill(0.0);
        self.s_v2x2.fill(0.0);
    }

    /// O(1) Delta Update Law: executes in ~12 FMAs for knob p when moving from u_old to u_new.
    #[inline(always)]
    pub fn update_delta(
        &mut self,
        proj: &ControlLatentProjection,
        u_old: f32,
        u_new: f32,
        delta_psi: f32,
    ) {
        // 1. Update z_main via delta_psi
        for d in 0..8 {
            self.z_main[d] = flush_recursive_state_f32(self.z_main[d] + proj.w_main[d] * delta_psi);
        }

        // 2. Update FM accumulators
        let delta_u = u_new - u_old;
        let delta_u2 = u_new * u_new - u_old * u_old;

        for f in 0..4 {
            let vf = proj.v_factors[f];
            self.s_vx[f] = flush_recursive_state_f32(self.s_vx[f] + vf * delta_u);
            self.s_v2x2[f] = flush_recursive_state_f32(self.s_v2x2[f] + vf * vf * delta_u2);
        }
    }

    /// Evaluates current 8-D static control latents z_ctrl = [z_head^(4), z_stage^(4)]^T.
    #[inline(always)]
    pub fn evaluate_z_ctrl(&self) -> [f32; 8] {
        let mut z = self.z_main;
        // Fold FM interaction into head modes 0..4
        for f in 0..4 {
            let fm_inter = 0.5 * (self.s_vx[f] * self.s_vx[f] - self.s_v2x2[f]);
            z[f] += fm_inter;
        }
        z
    }

    /// Updates the Ultra-tier ($R = 8$, modes `5..8`) FM accumulators `(s_vx_ultra, s_v2x2_ultra)`
    /// from `proj.v_factors[4..8]` in $\mathcal{O}(1)$.
    #[inline(always)]
    pub fn update_delta_ultra(
        proj: &ControlLatentProjection,
        u_old: f32,
        u_new: f32,
        s_vx_ultra: &mut [f32; 4],
        s_v2x2_ultra: &mut [f32; 4],
    ) {
        let delta_u = u_new - u_old;
        let delta_u2 = u_new * u_new - u_old * u_old;
        for f in 0..4 {
            let vf = proj.v_factors[4 + f];
            s_vx_ultra[f] = flush_recursive_state_f32(s_vx_ultra[f] + vf * delta_u);
            s_v2x2_ultra[f] = flush_recursive_state_f32(s_v2x2_ultra[f] + vf * vf * delta_u2);
        }
    }

    /// Evaluates the 4 Ultra-tier FM interaction terms (`modes 5..8` of Rank-8 FM).
    #[inline(always)]
    pub fn evaluate_z_ultra_fm(s_vx_ultra: &[f32; 4], s_v2x2_ultra: &[f32; 4]) -> [f32; 4] {
        let mut z_u = [0.0_f32; 4];
        for f in 0..4 {
            z_u[f] = 0.5 * (s_vx_ultra[f] * s_vx_ultra[f] - s_v2x2_ultra[f]);
        }
        z_u
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_control_channel_record_and_accumulator() {
        let proj_def = ControlLatentProjection::default();
        assert_eq!(proj_def.w_main, [0.0; 8]);
        assert_eq!(proj_def.v_factors, [0.0; 8]);

        let ccr_def = ControlChannelRecord::default();
        assert_eq!(ccr_def.projection, proj_def);

        let mut acc = LatentAccumulator64B::default();
        assert_eq!(acc.z_main, [0.0; 8]);
        assert_eq!(acc.s_vx, [0.0; 4]);
        assert_eq!(acc.s_v2x2, [0.0; 4]);

        let mut proj = ControlLatentProjection::new();
        proj.w_main = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        proj.v_factors = [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];

        acc.update_delta(&proj, 0.0, 0.5, 0.25);
        assert!((acc.z_main[0] - 0.025).abs() < 1e-6);
        assert!((acc.s_vx[0] - 0.5).abs() < 1e-6);
        assert!((acc.s_v2x2[0] - 0.25).abs() < 1e-6);

        let z_eval = acc.evaluate_z_ctrl();
        for &val in &z_eval {
            assert!(val.is_finite());
        }

        acc.reset();
        assert_eq!(acc.z_main, [0.0; 8]);
        assert_eq!(acc.s_vx, [0.0; 4]);
        assert_eq!(acc.s_v2x2, [0.0; 4]);
    }

    proptest! {
        #[test]
        fn prop_latent_accumulator_o1_incremental_delta_equivalence(
            w0 in -2.0_f32..2.0_f32,
            v0 in -2.0_f32..2.0_f32,
            u1 in -1.0_f32..1.0_f32,
            u2 in -1.0_f32..1.0_f32,
            psi1 in -1.0_f32..1.0_f32,
            psi2 in -1.0_f32..1.0_f32,
        ) {
            let mut proj = ControlLatentProjection::new();
            proj.w_main = [w0; 8];
            proj.v_factors = [v0; 8];

            // Two-step incremental update: 0 -> (u1, psi1) -> (u2, psi2)
            let mut acc_inc = LatentAccumulator64B::new();
            acc_inc.update_delta(&proj, 0.0, u1, psi1);
            acc_inc.update_delta(&proj, u1, u2, psi2 - psi1);

            // Direct one-step update: 0 -> (u2, psi2)
            let mut acc_dir = LatentAccumulator64B::new();
            acc_dir.update_delta(&proj, 0.0, u2, psi2);

            let z_inc = acc_inc.evaluate_z_ctrl();
            let z_dir = acc_dir.evaluate_z_ctrl();
            for d in 0..8 {
                prop_assert!((z_inc[d] - z_dir[d]).abs() < 1e-4);
            }
        }
    }
}
