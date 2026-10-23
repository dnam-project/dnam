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
        h_taps: &[&[[f32; 4]; 3]; 6],
    ) -> NeonVec4 {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { neon_conv6_1group(bias, w_tiles, h_taps) }
    }

    #[inline(always)]
    fn conv6_taps_2groups(
        bias: NeonVec4,
        w_tiles: &[[[f32; 4]; 4]; 12],
        h_taps: &[&[[f32; 4]; 3]; 6],
    ) -> NeonVec4 {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { neon_conv6_2groups(bias, w_tiles, h_taps) }
    }

    #[inline(always)]
    fn conv6_taps_3groups(
        bias: NeonVec4,
        w_tiles: &[[[f32; 4]; 4]; 18],
        h_taps: &[&[[f32; 4]; 3]; 6],
    ) -> NeonVec4 {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { neon_conv6_3groups(bias, w_tiles, h_taps) }
    }

    #[inline(always)]
    fn dot_product_16taps_4ch(taps_64: &[f32; 64], weights_64: &[f32; 64]) -> f32 {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { neon_dot64(taps_64, weights_64) }
    }

    #[inline(always)]
    fn dot_product_16taps_4ch_stereo(
        taps_64: &[f32; 64],
        weights_l: &[f32; 64],
        weights_r: &[f32; 64],
    ) -> [f32; 2] {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { neon_dot64_stereo(taps_64, weights_l, weights_r) }
    }

    #[inline(always)]
    fn f32x4_to_fp16x4(src: &[f32; 4]) -> [u16; 4] {
        let v_in = NeonVec4::load(src).0;
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
    h_taps: &[&[[f32; 4]; 3]; 6],
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
    h_taps: &[&[[f32; 4]; 3]; 6],
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
    h_taps: &[&[[f32; 4]; 3]; 6],
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

/// Evaluates a 64-element dot product using 4 independent NEON accumulators (`za0..za3`)
/// with zero `unsafe` blocks.
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
