//! ARM64 128-bit NEON hardware backend (`NeonBackend`) for Rockchip RK3582 (Darkglass Anagram)
//! and Apple Silicon.
//!
//! Uses `core::arch::aarch64::float32x4_t` dual-pipe `FMLA` (`vfmaq_f32`), `FMLS` (`vfmsq_f32`),
//! branchless `FMAXNM` / `FMINNM` (`vmaxnmq_f32`, `vminnmq_f32`), across-vector reduction
//! `FADDP` (`vaddvq_f32`), and ARMv8-A hardware Castagnoli `CRC32C` (`__crc32cd`, `__crc32cb`).

use core::arch::aarch64::{
    __crc32cb, __crc32cd, float32x4_t, vabsq_f32, vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32,
    vfmaq_laneq_f32, vfmsq_f32, vmaxnmq_f32, vminnmq_f32, vmulq_f32, vsubq_f32,
};

use super::{BackendKind, MathBackend, SimdVec4};

/// 128-bit ARM64 NEON 4-lane `f32` register wrapper (`float32x4_t`).
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct NeonVec4(pub float32x4_t);

impl SimdVec4 for NeonVec4 {
    #[inline(always)]
    fn splat(val: f32) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { Self(vdupq_n_f32(val)) }
    }

    #[inline(always)]
    fn load(src: &[f32; 4]) -> Self {
        Self(zerocopy::transmute!(*src))
    }

    #[inline(always)]
    fn store(self, dst: &mut [f32; 4]) {
        *dst = zerocopy::transmute!(self.0);
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { Self(vaddq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { Self(vsubq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { Self(vmulq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { Self(vfmaq_f32(self.0, a.0, b.0)) }
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { Self(vfmsq_f32(self.0, a.0, b.0)) }
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { Self(vmaxnmq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { Self(vminnmq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { Self(vabsq_f32(self.0)) }
    }

    #[inline(always)]
    fn horizontal_sum(self) -> f32 {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { vaddvq_f32(self.0) }
    }
}

/// ARM64 NEON hardware math backend targeting Rockchip RK3582 (Cortex-A76/A55) and Apple Silicon.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NeonBackend;

impl MathBackend for NeonBackend {
    type Vec4 = NeonVec4;
    const NAME: &'static str = "ARM64 NEON (128-bit float32x4_t)";
    const KIND: BackendKind = BackendKind::ArmNeon;

    #[inline(always)]
    fn fma_mat4x4_vec4(acc: NeonVec4, mat: &[[f32; 4]; 4], x: &[f32; 4]) -> NeonVec4 {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { neon_fma_mat4x4_vec4(acc, mat, x) }
    }

    #[inline(always)]
    fn conv6_taps_1group(
        bias: NeonVec4,
        w_tiles: &[[[f32; 4]; 4]; 6],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> NeonVec4 {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { neon_conv6_1group(bias, w_tiles, h_taps) }
    }

    #[inline(always)]
    fn conv6_taps_2groups(
        bias: NeonVec4,
        w_tiles: &[[[f32; 4]; 4]; 12],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> NeonVec4 {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { neon_conv6_2groups(bias, w_tiles, h_taps) }
    }

    #[inline(always)]
    fn conv6_taps_3groups(
        bias: NeonVec4,
        w_tiles: &[[[f32; 4]; 4]; 18],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> NeonVec4 {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { neon_conv6_3groups(bias, w_tiles, h_taps) }
    }

    #[inline(always)]
    fn conv6_taps_4groups(
        bias: NeonVec4,
        w_tiles: &[[[f32; 4]; 4]; 24],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> NeonVec4 {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { neon_conv6_4groups(bias, w_tiles, h_taps) }
    }

    #[inline(always)]
    fn f32x4_to_fp16x4(src: &[f32; 4]) -> [u16; 4] {
        // Clamp to finite fp16 range [-65504.0, 65504.0] via branchless FMAXNM/FMINNM
        // so values >= 65520.0, ±Inf, or NaN never produce non-finite fp16 bits.
        let v_raw = NeonVec4::load(src);
        let v_clamped = v_raw
            .min(NeonVec4::splat(65504.0))
            .max(NeonVec4::splat(-65504.0));
        let v_in = v_clamped.0;
        let bits_u64: u64;
        // SAFETY: `fcvtn` is a mandatory baseline ARMv8.0-A NEON instruction converting
        // 4 IEEE-754 `f32` lanes to 4 `fp16` lanes in a single cycle.
        unsafe {
            core::arch::asm!(
                "fcvtn {tmp:v}.4h, {inp:v}.4s",
                "umov {out:x}, {tmp:v}.d[0]",
                inp = in(vreg) v_in,
                tmp = out(vreg) _,
                out = out(reg) bits_u64,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        zerocopy::transmute!(bits_u64.to_le())
    }

    #[inline(always)]
    fn fp16x4_to_f32x4(src: &[u16; 4]) -> [f32; 4] {
        let bytes: [u8; 8] = zerocopy::transmute!(*src);
        let bits_u64 = u64::from_le_bytes(bytes);
        let v_out: float32x4_t;
        // SAFETY: `fcvtl` is a mandatory baseline ARMv8.0-A NEON instruction converting
        // 4 IEEE-754 `fp16` lanes to 4 `f32` lanes in a single cycle.
        unsafe {
            core::arch::asm!(
                "fmov {tmp:d}, {inp:x}",
                "fcvtl {out:v}.4s, {tmp:v}.4h",
                inp = in(reg) bits_u64,
                tmp = out(vreg) _,
                out = out(vreg) v_out,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        zerocopy::transmute!(v_out)
    }

    #[inline]
    fn crc32c(data: &[u8]) -> u32 {
        crc32c_arm64_update(0, data)
    }
}

/// Safe `#[target_feature(enable = "neon")]` $4 \times 4$ column-major matrix-vector FMA.
#[target_feature(enable = "neon")]
#[inline]
pub(crate) fn neon_fma_mat4x4_vec4(acc: NeonVec4, mat: &[[f32; 4]; 4], x: &[f32; 4]) -> NeonVec4 {
    let vx = NeonVec4::load(x).0;
    let mut out = acc.0;
    out = vfmaq_laneq_f32::<0>(out, NeonVec4::load(&mat[0]).0, vx);
    out = vfmaq_laneq_f32::<1>(out, NeonVec4::load(&mat[1]).0, vx);
    out = vfmaq_laneq_f32::<2>(out, NeonVec4::load(&mat[2]).0, vx);
    out = vfmaq_laneq_f32::<3>(out, NeonVec4::load(&mat[3]).0, vx);
    NeonVec4(out)
}

/// Evaluates a 6-tap dilated convolution over 1 input group (`6 × 4×4` tiles = 24 FMAs)
/// using 4 independent NEON lane accumulators (`za0..za3`) with zero `unsafe` blocks.
#[target_feature(enable = "neon")]
#[inline]
pub(crate) fn neon_conv6_1group(
    bias: NeonVec4,
    w_tiles: &[[[f32; 4]; 4]; 6],
    h_taps: &[&[[f32; 4]; 4]; 6],
) -> NeonVec4 {
    let mut za0 = bias.0;
    let mut za1 = vdupq_n_f32(0.0);
    let mut za2 = vdupq_n_f32(0.0);
    let mut za3 = vdupq_n_f32(0.0);

    for tap in 0..6 {
        let vx = NeonVec4::load(&h_taps[tap][0]).0;
        let wt = &w_tiles[tap];
        za0 = vfmaq_laneq_f32::<0>(za0, NeonVec4::load(&wt[0]).0, vx);
        za1 = vfmaq_laneq_f32::<1>(za1, NeonVec4::load(&wt[1]).0, vx);
        za2 = vfmaq_laneq_f32::<2>(za2, NeonVec4::load(&wt[2]).0, vx);
        za3 = vfmaq_laneq_f32::<3>(za3, NeonVec4::load(&wt[3]).0, vx);
    }

    NeonVec4(vaddq_f32(vaddq_f32(za0, za1), vaddq_f32(za2, za3)))
}

/// Evaluates a 6-tap dilated convolution over 2 input groups (`12 × 4×4` tiles = 48 FMAs)
/// using 4 independent NEON lane accumulators (`za0..za3`) with zero `unsafe` blocks.
#[target_feature(enable = "neon")]
#[inline]
pub(crate) fn neon_conv6_2groups(
    bias: NeonVec4,
    w_tiles: &[[[f32; 4]; 4]; 12],
    h_taps: &[&[[f32; 4]; 4]; 6],
) -> NeonVec4 {
    let mut za0 = bias.0;
    let mut za1 = vdupq_n_f32(0.0);
    let mut za2 = vdupq_n_f32(0.0);
    let mut za3 = vdupq_n_f32(0.0);

    for tap in 0..6 {
        let ht = h_taps[tap];
        let vx0 = NeonVec4::load(&ht[0]).0;
        let vx1 = NeonVec4::load(&ht[1]).0;
        let wt0 = &w_tiles[tap * 2];
        let wt1 = &w_tiles[tap * 2 + 1];

        za0 = vfmaq_laneq_f32::<0>(za0, NeonVec4::load(&wt0[0]).0, vx0);
        za1 = vfmaq_laneq_f32::<1>(za1, NeonVec4::load(&wt0[1]).0, vx0);
        za2 = vfmaq_laneq_f32::<2>(za2, NeonVec4::load(&wt0[2]).0, vx0);
        za3 = vfmaq_laneq_f32::<3>(za3, NeonVec4::load(&wt0[3]).0, vx0);

        za0 = vfmaq_laneq_f32::<0>(za0, NeonVec4::load(&wt1[0]).0, vx1);
        za1 = vfmaq_laneq_f32::<1>(za1, NeonVec4::load(&wt1[1]).0, vx1);
        za2 = vfmaq_laneq_f32::<2>(za2, NeonVec4::load(&wt1[2]).0, vx1);
        za3 = vfmaq_laneq_f32::<3>(za3, NeonVec4::load(&wt1[3]).0, vx1);
    }

    NeonVec4(vaddq_f32(vaddq_f32(za0, za1), vaddq_f32(za2, za3)))
}

/// Evaluates a 6-tap dilated convolution over 3 input groups (`18 × 4×4` tiles = 72 FMAs)
/// using 4 independent NEON lane accumulators (`za0..za3`) with zero `unsafe` blocks.
#[target_feature(enable = "neon")]
#[inline]
pub(crate) fn neon_conv6_3groups(
    bias: NeonVec4,
    w_tiles: &[[[f32; 4]; 4]; 18],
    h_taps: &[&[[f32; 4]; 4]; 6],
) -> NeonVec4 {
    let mut za0 = bias.0;
    let mut za1 = vdupq_n_f32(0.0);
    let mut za2 = vdupq_n_f32(0.0);
    let mut za3 = vdupq_n_f32(0.0);

    for tap in 0..6 {
        let ht = h_taps[tap];
        let vx0 = NeonVec4::load(&ht[0]).0;
        let vx1 = NeonVec4::load(&ht[1]).0;
        let vx2 = NeonVec4::load(&ht[2]).0;
        let base = tap * 3;
        let wt0 = &w_tiles[base];
        let wt1 = &w_tiles[base + 1];
        let wt2 = &w_tiles[base + 2];

        za0 = vfmaq_laneq_f32::<0>(za0, NeonVec4::load(&wt0[0]).0, vx0);
        za1 = vfmaq_laneq_f32::<1>(za1, NeonVec4::load(&wt0[1]).0, vx0);
        za2 = vfmaq_laneq_f32::<2>(za2, NeonVec4::load(&wt0[2]).0, vx0);
        za3 = vfmaq_laneq_f32::<3>(za3, NeonVec4::load(&wt0[3]).0, vx0);

        za0 = vfmaq_laneq_f32::<0>(za0, NeonVec4::load(&wt1[0]).0, vx1);
        za1 = vfmaq_laneq_f32::<1>(za1, NeonVec4::load(&wt1[1]).0, vx1);
        za2 = vfmaq_laneq_f32::<2>(za2, NeonVec4::load(&wt1[2]).0, vx1);
        za3 = vfmaq_laneq_f32::<3>(za3, NeonVec4::load(&wt1[3]).0, vx1);

        za0 = vfmaq_laneq_f32::<0>(za0, NeonVec4::load(&wt2[0]).0, vx2);
        za1 = vfmaq_laneq_f32::<1>(za1, NeonVec4::load(&wt2[1]).0, vx2);
        za2 = vfmaq_laneq_f32::<2>(za2, NeonVec4::load(&wt2[2]).0, vx2);
        za3 = vfmaq_laneq_f32::<3>(za3, NeonVec4::load(&wt2[3]).0, vx2);
    }

    NeonVec4(vaddq_f32(vaddq_f32(za0, za1), vaddq_f32(za2, za3)))
}

/// Evaluates a 6-tap dilated convolution over 4 input groups (`24 × 4×4` tiles = 96 FMAs)
/// using 4 independent NEON lane accumulators (`za0..za3`) with zero `unsafe` blocks.
#[target_feature(enable = "neon")]
#[inline]
pub(crate) fn neon_conv6_4groups(
    bias: NeonVec4,
    w_tiles: &[[[f32; 4]; 4]; 24],
    h_taps: &[&[[f32; 4]; 4]; 6],
) -> NeonVec4 {
    let mut za0 = bias.0;
    let mut za1 = vdupq_n_f32(0.0);
    let mut za2 = vdupq_n_f32(0.0);
    let mut za3 = vdupq_n_f32(0.0);

    for tap in 0..6 {
        let ht = h_taps[tap];
        let base = tap * 4;
        for g in 0..4 {
            let vxg = NeonVec4::load(&ht[g]).0;
            let wtg = &w_tiles[base + g];
            za0 = vfmaq_laneq_f32::<0>(za0, NeonVec4::load(&wtg[0]).0, vxg);
            za1 = vfmaq_laneq_f32::<1>(za1, NeonVec4::load(&wtg[1]).0, vxg);
            za2 = vfmaq_laneq_f32::<2>(za2, NeonVec4::load(&wtg[2]).0, vxg);
            za3 = vfmaq_laneq_f32::<3>(za3, NeonVec4::load(&wtg[3]).0, vxg);
        }
    }

    NeonVec4(vaddq_f32(vaddq_f32(za0, za1), vaddq_f32(za2, za3)))
}

/// Evaluates a 64-element dot product using 4 independent NEON accumulators (`za0..za3`)
/// with zero `unsafe` blocks.
#[allow(dead_code)]
#[target_feature(enable = "neon")]
#[inline]
pub(crate) fn neon_dot64(a: &[f32; 64], b: &[f32; 64]) -> f32 {
    let a_blocks: &[[[f32; 4]; 4]; 4] = zerocopy::transmute_ref!(a);
    let b_blocks: &[[[f32; 4]; 4]; 4] = zerocopy::transmute_ref!(b);

    let mut za0 = vdupq_n_f32(0.0);
    let mut za1 = vdupq_n_f32(0.0);
    let mut za2 = vdupq_n_f32(0.0);
    let mut za3 = vdupq_n_f32(0.0);

    for blk in 0..4 {
        let ab = &a_blocks[blk];
        let bb = &b_blocks[blk];
        za0 = vfmaq_f32(za0, NeonVec4::load(&ab[0]).0, NeonVec4::load(&bb[0]).0);
        za1 = vfmaq_f32(za1, NeonVec4::load(&ab[1]).0, NeonVec4::load(&bb[1]).0);
        za2 = vfmaq_f32(za2, NeonVec4::load(&ab[2]).0, NeonVec4::load(&bb[2]).0);
        za3 = vfmaq_f32(za3, NeonVec4::load(&ab[3]).0, NeonVec4::load(&bb[3]).0);
    }

    let sum01 = vaddq_f32(za0, za1);
    let sum23 = vaddq_f32(za2, za3);
    vaddvq_f32(vaddq_f32(sum01, sum23))
}

/// Evaluates a fused stereo (`L` and `R`) 64-element dot product in a single pass over `taps_64`
/// with zero `unsafe` blocks.
#[allow(dead_code)]
#[target_feature(enable = "neon")]
#[inline]
pub(crate) fn neon_dot64_stereo(taps: &[f32; 64], w_l: &[f32; 64], w_r: &[f32; 64]) -> [f32; 2] {
    let t_blocks: &[[[f32; 4]; 4]; 4] = zerocopy::transmute_ref!(taps);
    let wl_blocks: &[[[f32; 4]; 4]; 4] = zerocopy::transmute_ref!(w_l);
    let wr_blocks: &[[[f32; 4]; 4]; 4] = zerocopy::transmute_ref!(w_r);

    let mut l0 = vdupq_n_f32(0.0);
    let mut l1 = vdupq_n_f32(0.0);
    let mut l2 = vdupq_n_f32(0.0);
    let mut l3 = vdupq_n_f32(0.0);
    let mut r0 = vdupq_n_f32(0.0);
    let mut r1 = vdupq_n_f32(0.0);
    let mut r2 = vdupq_n_f32(0.0);
    let mut r3 = vdupq_n_f32(0.0);

    for blk in 0..4 {
        let tb = &t_blocks[blk];
        let wlb = &wl_blocks[blk];
        let wrb = &wr_blocks[blk];

        let vt0 = NeonVec4::load(&tb[0]).0;
        l0 = vfmaq_f32(l0, vt0, NeonVec4::load(&wlb[0]).0);
        r0 = vfmaq_f32(r0, vt0, NeonVec4::load(&wrb[0]).0);

        let vt1 = NeonVec4::load(&tb[1]).0;
        l1 = vfmaq_f32(l1, vt1, NeonVec4::load(&wlb[1]).0);
        r1 = vfmaq_f32(r1, vt1, NeonVec4::load(&wrb[1]).0);

        let vt2 = NeonVec4::load(&tb[2]).0;
        l2 = vfmaq_f32(l2, vt2, NeonVec4::load(&wlb[2]).0);
        r2 = vfmaq_f32(r2, vt2, NeonVec4::load(&wrb[2]).0);

        let vt3 = NeonVec4::load(&tb[3]).0;
        l3 = vfmaq_f32(l3, vt3, NeonVec4::load(&wlb[3]).0);
        r3 = vfmaq_f32(r3, vt3, NeonVec4::load(&wrb[3]).0);
    }

    let out_l = vaddvq_f32(vaddq_f32(vaddq_f32(l0, l1), vaddq_f32(l2, l3)));
    let out_r = vaddvq_f32(vaddq_f32(vaddq_f32(r0, r1), vaddq_f32(r2, r3)));
    [out_l, out_r]
}

/// Hardware ARMv8-A Castagnoli `CRC32C` (`crc32cx` / `crc32cb`) incremental update.
#[inline]
pub(crate) fn crc32c_arm64_update(crc_init: u32, data: &[u8]) -> u32 {
    // SAFETY: ARMv8.1+ (Apple Silicon M1..M4 and Rockchip RK3582 Cortex-A76/A55) implement
    // mandatory hardware CRC32C instructions.
    unsafe { crc32c_arm64_hw_inner(crc_init, data) }
}

/// Safe `#[target_feature(enable = "crc")]` hardware `CRC32C` inner loop with zero `unsafe` blocks.
#[target_feature(enable = "crc")]
#[inline]
fn crc32c_arm64_hw_inner(crc_init: u32, data: &[u8]) -> u32 {
    let mut crc = !crc_init;
    let (chunks, remainder) = data.as_chunks::<8>();
    for chunk in chunks {
        let val = u64::from_le_bytes(*chunk);
        crc = __crc32cd(crc, val);
    }
    for &byte in remainder {
        crc = __crc32cb(crc, byte);
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neon_vec4_operations() {
        let v1 = NeonVec4::splat(2.0);
        let v2 = NeonVec4::splat(3.0);
        let mut dst = [0.0f32; 4];
        v1.store(&mut dst);
        assert_eq!(dst, [2.0; 4]);

        let v_add = v1.add(v2);
        v_add.store(&mut dst);
        assert_eq!(dst, [5.0; 4]);

        let v_sub = v2.sub(v1);
        v_sub.store(&mut dst);
        assert_eq!(dst, [1.0; 4]);

        let v_mul = v1.mul(v2);
        v_mul.store(&mut dst);
        assert_eq!(dst, [6.0; 4]);

        let v_fma = v1.fma(v2, v1); // 2 + 3 * 2 = 8
        v_fma.store(&mut dst);
        assert_eq!(dst, [8.0; 4]);

        let v_fms = v1.fms(v2, v1); // 2 - 3 * 2 = -4
        v_fms.store(&mut dst);
        assert_eq!(dst, [-4.0; 4]);

        let v_max = v1.max(v2);
        v_max.store(&mut dst);
        assert_eq!(dst, [3.0; 4]);

        let v_min = v1.min(v2);
        v_min.store(&mut dst);
        assert_eq!(dst, [2.0; 4]);

        let v_neg = NeonVec4::splat(-5.0);
        let v_abs = v_neg.abs();
        v_abs.store(&mut dst);
        assert_eq!(dst, [5.0; 4]);

        let sum = v1.horizontal_sum();
        assert_eq!(sum, 8.0);
    }

    #[test]
    fn test_neon_backend_fp16_conversions_and_crc() {
        let f = [1.0f32, -2.5, 3.25, 0.0];
        let h = NeonBackend::f32x4_to_fp16x4(&f);
        let back = NeonBackend::fp16x4_to_f32x4(&h);
        for i in 0..4 {
            assert!((f[i] - back[i]).abs() < 1e-3);
        }

        // Verify finite saturation at [65520.0, 100000.0] never overflows to ±Inf
        let extreme = [65520.0f32, -65520.0, 100_000.0, -100_000.0];
        let h_ext = NeonBackend::f32x4_to_fp16x4(&extreme);
        assert_eq!(h_ext, [0x7BFF, 0xFBFF, 0x7BFF, 0xFBFF]);
        let back_ext = NeonBackend::fp16x4_to_f32x4(&h_ext);
        for &v in &back_ext {
            assert!(v.is_finite());
        }

        let data = b"123456789";
        let crc = NeonBackend::crc32c(data);
        assert_eq!(crc, 0xe3069283);
    }

    #[test]
    fn test_neon_backend_convolutions_and_dots() {
        let bias = NeonVec4::splat(0.1);
        let w_tile = [[0.1f32; 4]; 4];
        let h_tap = [[0.2f32; 4]; 4];

        // Mat-vec FMA
        let mv = NeonBackend::fma_mat4x4_vec4(bias, &w_tile, &[0.5; 4]);
        let mut dst = [0.0f32; 4];
        mv.store(&mut dst);
        assert!(dst[0].is_finite());

        // 1 group
        let w1 = [w_tile; 6];
        let h1 = [&h_tap; 6];
        let r1 = NeonBackend::conv6_taps_1group(bias, &w1, &h1);
        r1.store(&mut dst);
        assert!(dst[0].is_finite());

        // 2 groups
        let w2 = [w_tile; 12];
        let r2 = NeonBackend::conv6_taps_2groups(bias, &w2, &h1);
        r2.store(&mut dst);
        assert!(dst[0].is_finite());

        // 3 groups
        let w3 = [w_tile; 18];
        let r3 = NeonBackend::conv6_taps_3groups(bias, &w3, &h1);
        r3.store(&mut dst);
        assert!(dst[0].is_finite());

        // 4 groups
        let w4 = [w_tile; 24];
        let r4 = NeonBackend::conv6_taps_4groups(bias, &w4, &h1);
        r4.store(&mut dst);
        assert!(dst[0].is_finite());

        // dot64 and dot64_stereo
        let taps = [0.1f32; 64];
        let wl = [0.2f32; 64];
        let wr = [0.3f32; 64];
        // SAFETY: Unit test inputs are 64-element aligned buffers matching neon target requirements.
        let d = unsafe { neon_dot64(&taps, &wl) };
        assert!(d.is_finite());
        // SAFETY: Unit test inputs are 64-element aligned buffers matching neon target requirements.
        let stereo = unsafe { neon_dot64_stereo(&taps, &wl, &wr) };
        assert!(stereo[0].is_finite());
        assert!(stereo[1].is_finite());
    }
}
