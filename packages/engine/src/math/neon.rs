//! ARM64 128-bit NEON hardware backend (`NeonBackend`) for Rockchip RK3582 (Darkglass Anagram)
//! and Apple Silicon.
//!
//! Uses `core::arch::aarch64::float32x4_t` dual-pipe `FMLA` (`vfmaq_f32`), `FMLS` (`vfmsq_f32`),
//! branchless `FMAXNM` / `FMINNM` (`vmaxnmq_f32`, `vminnmq_f32`), across-vector reduction
//! `FADDP` (`vaddvq_f32`), and ARMv8-A hardware Castagnoli `CRC32C` (`__crc32cd`, `__crc32cb`).

use core::arch::aarch64::{
    __crc32cb, __crc32cd, float32x4_t, vabsq_f32, vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32,
    vfmaq_laneq_f32, vfmsq_f32, vld1q_f32, vmaxnmq_f32, vminnmq_f32, vmulq_f32, vst1q_f32,
    vsubq_f32,
};

use super::{BackendKind, MathBackend, SimdVec4};

/// 128-bit ARM64 NEON 4-lane `f32` register wrapper (`float32x4_t`).
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct NeonVec4(pub float32x4_t);

impl SimdVec4 for NeonVec4 {
    #[inline(always)]
    fn splat(val: f32) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability; `vdupq_n_f32` is a pure
        // register broadcast with no memory access.
        unsafe { Self(vdupq_n_f32(val)) }
    }

    #[inline(always)]
    fn load(src: &[f32; 4]) -> Self {
        // SAFETY: `src` is a valid reference to 4 contiguous `f32` values (`16` bytes).
        unsafe { Self(vld1q_f32(src.as_ptr())) }
    }

    #[inline(always)]
    fn store(self, dst: &mut [f32; 4]) {
        // SAFETY: `dst` is a valid mutable reference to 4 contiguous `f32` values (`16` bytes).
        unsafe { vst1q_f32(dst.as_mut_ptr(), self.0) }
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        // SAFETY: Baseline `aarch64` NEON register operation.
        unsafe { Self(vaddq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        // SAFETY: Baseline `aarch64` NEON register operation.
        unsafe { Self(vsubq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // SAFETY: Baseline `aarch64` NEON register operation.
        unsafe { Self(vmulq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        // SAFETY: `vfmaq_f32(acc, a, b)` computes `acc + a * b` in a single fused instruction.
        unsafe { Self(vfmaq_f32(self.0, a.0, b.0)) }
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        // SAFETY: `vfmsq_f32(acc, a, b)` computes `acc - a * b` in a single fused instruction.
        unsafe { Self(vfmsq_f32(self.0, a.0, b.0)) }
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        // SAFETY: `vmaxnmq_f32` is branchless IEEE-754 maxNum across 4 lanes.
        unsafe { Self(vmaxnmq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        // SAFETY: `vminnmq_f32` is branchless IEEE-754 minNum across 4 lanes.
        unsafe { Self(vminnmq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        // SAFETY: `vabsq_f32` clears sign bits across 4 lanes in 1 cycle.
        unsafe { Self(vabsq_f32(self.0)) }
    }

    #[inline(always)]
    fn horizontal_sum(self) -> f32 {
        // SAFETY: `vaddvq_f32` reduces 4 lanes to a scalar `f32` via hardware `FADDP`.
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
        // SAFETY: `mat` contains 4 contiguous `[f32; 4]` vectors and `x` contains 4 `f32` lanes.
        unsafe {
            let vx = vld1q_f32(x.as_ptr());
            let mut out = acc.0;
            out = vfmaq_laneq_f32::<0>(out, vld1q_f32(mat[0].as_ptr()), vx);
            out = vfmaq_laneq_f32::<1>(out, vld1q_f32(mat[1].as_ptr()), vx);
            out = vfmaq_laneq_f32::<2>(out, vld1q_f32(mat[2].as_ptr()), vx);
            out = vfmaq_laneq_f32::<3>(out, vld1q_f32(mat[3].as_ptr()), vx);
            NeonVec4(out)
        }
    }

    #[inline(always)]
    fn conv6_taps_1group(
        bias: NeonVec4,
        w_tiles: &[[[f32; 4]; 4]; 6],
        h_taps: &[&[[f32; 4]; 3]; 6],
    ) -> NeonVec4 {
        // SAFETY: All 6 tiles and 6 history tap references are fixed-size arrays.
        unsafe { neon_conv6_1group(bias, w_tiles, h_taps) }
    }

    #[inline(always)]
    fn conv6_taps_2groups(
        bias: NeonVec4,
        w_tiles: &[[[f32; 4]; 4]; 12],
        h_taps: &[&[[f32; 4]; 3]; 6],
    ) -> NeonVec4 {
        // SAFETY: All 12 tiles and 6 history tap references are fixed-size arrays.
        unsafe { neon_conv6_2groups(bias, w_tiles, h_taps) }
    }

    #[inline(always)]
    fn conv6_taps_3groups(
        bias: NeonVec4,
        w_tiles: &[[[f32; 4]; 4]; 18],
        h_taps: &[&[[f32; 4]; 3]; 6],
    ) -> NeonVec4 {
        // SAFETY: All 18 tiles and 6 history tap references are fixed-size arrays.
        unsafe { neon_conv6_3groups(bias, w_tiles, h_taps) }
    }

    #[inline(always)]
    fn dot_product_16taps_4ch(taps_64: &[f32; 64], weights_64: &[f32; 64]) -> f32 {
        // SAFETY: `taps_64` and `weights_64` are valid `[f32; 64]` arrays.
        unsafe { neon_dot64(taps_64, weights_64) }
    }

    #[inline(always)]
    fn dot_product_16taps_4ch_stereo(
        taps_64: &[f32; 64],
        weights_l: &[f32; 64],
        weights_r: &[f32; 64],
    ) -> [f32; 2] {
        // SAFETY: `taps_64`, `weights_l`, and `weights_r` are valid `[f32; 64]` arrays.
        unsafe { neon_dot64_stereo(taps_64, weights_l, weights_r) }
    }

    #[inline(always)]
    fn f32x4_to_fp16x4(src: &[f32; 4]) -> [u16; 4] {
        // SAFETY: `fcvtn` is a mandatory baseline ARMv8.0-A NEON instruction converting
        // 4 IEEE-754 `f32` lanes to 4 `fp16` lanes in a single cycle.
        unsafe {
            let v_in = vld1q_f32(src.as_ptr());
            let bits_u64: u64;
            core::arch::asm!(
                "fcvtn {tmp:v}.4h, {inp:v}.4s",
                "umov {out:x}, {tmp:v}.d[0]",
                inp = in(vreg) v_in,
                tmp = out(vreg) _,
                out = out(reg) bits_u64,
                options(pure, nomem, nostack, preserves_flags),
            );
            core::mem::transmute::<u64, [u16; 4]>(bits_u64.to_le())
        }
    }

    #[inline(always)]
    fn fp16x4_to_f32x4(src: &[u16; 4]) -> [f32; 4] {
        // SAFETY: `fcvtl` is a mandatory baseline ARMv8.0-A NEON instruction converting
        // 4 IEEE-754 `fp16` lanes to 4 `f32` lanes in a single cycle.
        unsafe {
            let bits_u64 = u64::from_le_bytes(core::mem::transmute::<[u16; 4], [u8; 8]>(*src));
            let v_out: float32x4_t;
            core::arch::asm!(
                "fmov {tmp:d}, {inp:x}",
                "fcvtl {out:v}.4s, {tmp:v}.4h",
                inp = in(reg) bits_u64,
                tmp = out(vreg) _,
                out = out(vreg) v_out,
                options(pure, nomem, nostack, preserves_flags),
            );
            let mut dst = [0.0_f32; 4];
            vst1q_f32(dst.as_mut_ptr(), v_out);
            dst
        }
    }

    #[inline]
    fn crc32c(data: &[u8]) -> u32 {
        crc32c_arm64_update(0, data)
    }
}

/// Evaluates a 6-tap dilated convolution over 1 input group (`6 × 4×4` tiles = 24 FMAs)
/// using 4 independent NEON lane accumulators (`za0..za3`) to eliminate FMA pipeline latency stalls.
///
/// # Safety
/// Caller must ensure `aarch64` NEON availability.
#[inline(always)]
pub(crate) unsafe fn neon_conv6_1group(
    bias: NeonVec4,
    w_tiles: &[[[f32; 4]; 4]; 6],
    h_taps: &[&[[f32; 4]; 3]; 6],
) -> NeonVec4 {
    // SAFETY: Fixed-size references with in-bounds static indexing.
    unsafe {
        let mut za0 = bias.0;
        let mut za1 = vdupq_n_f32(0.0);
        let mut za2 = vdupq_n_f32(0.0);
        let mut za3 = vdupq_n_f32(0.0);

        for tap in 0..6 {
            let vx = vld1q_f32(h_taps[tap][0].as_ptr());
            let wt = &w_tiles[tap];
            za0 = vfmaq_laneq_f32::<0>(za0, vld1q_f32(wt[0].as_ptr()), vx);
            za1 = vfmaq_laneq_f32::<1>(za1, vld1q_f32(wt[1].as_ptr()), vx);
            za2 = vfmaq_laneq_f32::<2>(za2, vld1q_f32(wt[2].as_ptr()), vx);
            za3 = vfmaq_laneq_f32::<3>(za3, vld1q_f32(wt[3].as_ptr()), vx);
        }

        NeonVec4(vaddq_f32(vaddq_f32(za0, za1), vaddq_f32(za2, za3)))
    }
}

/// Evaluates a 6-tap dilated convolution over 2 input groups (`12 × 4×4` tiles = 48 FMAs)
/// using 4 independent NEON lane accumulators (`za0..za3`).
///
/// # Safety
/// Caller must ensure `aarch64` NEON availability.
#[inline(always)]
pub(crate) unsafe fn neon_conv6_2groups(
    bias: NeonVec4,
    w_tiles: &[[[f32; 4]; 4]; 12],
    h_taps: &[&[[f32; 4]; 3]; 6],
) -> NeonVec4 {
    // SAFETY: Fixed-size references with in-bounds static indexing.
    unsafe {
        let mut za0 = bias.0;
        let mut za1 = vdupq_n_f32(0.0);
        let mut za2 = vdupq_n_f32(0.0);
        let mut za3 = vdupq_n_f32(0.0);

        for tap in 0..6 {
            let ht = h_taps[tap];
            let vx0 = vld1q_f32(ht[0].as_ptr());
            let vx1 = vld1q_f32(ht[1].as_ptr());
            let wt0 = &w_tiles[tap * 2];
            let wt1 = &w_tiles[tap * 2 + 1];

            za0 = vfmaq_laneq_f32::<0>(za0, vld1q_f32(wt0[0].as_ptr()), vx0);
            za1 = vfmaq_laneq_f32::<1>(za1, vld1q_f32(wt0[1].as_ptr()), vx0);
            za2 = vfmaq_laneq_f32::<2>(za2, vld1q_f32(wt0[2].as_ptr()), vx0);
            za3 = vfmaq_laneq_f32::<3>(za3, vld1q_f32(wt0[3].as_ptr()), vx0);

            za0 = vfmaq_laneq_f32::<0>(za0, vld1q_f32(wt1[0].as_ptr()), vx1);
            za1 = vfmaq_laneq_f32::<1>(za1, vld1q_f32(wt1[1].as_ptr()), vx1);
            za2 = vfmaq_laneq_f32::<2>(za2, vld1q_f32(wt1[2].as_ptr()), vx1);
            za3 = vfmaq_laneq_f32::<3>(za3, vld1q_f32(wt1[3].as_ptr()), vx1);
        }

        NeonVec4(vaddq_f32(vaddq_f32(za0, za1), vaddq_f32(za2, za3)))
    }
}

/// Evaluates a 6-tap dilated convolution over 3 input groups (`18 × 4×4` tiles = 72 FMAs)
/// using 4 independent NEON lane accumulators (`za0..za3`).
///
/// # Safety
/// Caller must ensure `aarch64` NEON availability.
#[inline(always)]
pub(crate) unsafe fn neon_conv6_3groups(
    bias: NeonVec4,
    w_tiles: &[[[f32; 4]; 4]; 18],
    h_taps: &[&[[f32; 4]; 3]; 6],
) -> NeonVec4 {
    // SAFETY: Fixed-size references with in-bounds static indexing.
    unsafe {
        let mut za0 = bias.0;
        let mut za1 = vdupq_n_f32(0.0);
        let mut za2 = vdupq_n_f32(0.0);
        let mut za3 = vdupq_n_f32(0.0);

        for tap in 0..6 {
            let ht = h_taps[tap];
            let vx0 = vld1q_f32(ht[0].as_ptr());
            let vx1 = vld1q_f32(ht[1].as_ptr());
            let vx2 = vld1q_f32(ht[2].as_ptr());
            let base = tap * 3;
            let wt0 = &w_tiles[base];
            let wt1 = &w_tiles[base + 1];
            let wt2 = &w_tiles[base + 2];

            za0 = vfmaq_laneq_f32::<0>(za0, vld1q_f32(wt0[0].as_ptr()), vx0);
            za1 = vfmaq_laneq_f32::<1>(za1, vld1q_f32(wt0[1].as_ptr()), vx0);
            za2 = vfmaq_laneq_f32::<2>(za2, vld1q_f32(wt0[2].as_ptr()), vx0);
            za3 = vfmaq_laneq_f32::<3>(za3, vld1q_f32(wt0[3].as_ptr()), vx0);

            za0 = vfmaq_laneq_f32::<0>(za0, vld1q_f32(wt1[0].as_ptr()), vx1);
            za1 = vfmaq_laneq_f32::<1>(za1, vld1q_f32(wt1[1].as_ptr()), vx1);
            za2 = vfmaq_laneq_f32::<2>(za2, vld1q_f32(wt1[2].as_ptr()), vx1);
            za3 = vfmaq_laneq_f32::<3>(za3, vld1q_f32(wt1[3].as_ptr()), vx1);

            za0 = vfmaq_laneq_f32::<0>(za0, vld1q_f32(wt2[0].as_ptr()), vx2);
            za1 = vfmaq_laneq_f32::<1>(za1, vld1q_f32(wt2[1].as_ptr()), vx2);
            za2 = vfmaq_laneq_f32::<2>(za2, vld1q_f32(wt2[2].as_ptr()), vx2);
            za3 = vfmaq_laneq_f32::<3>(za3, vld1q_f32(wt2[3].as_ptr()), vx2);
        }

        NeonVec4(vaddq_f32(vaddq_f32(za0, za1), vaddq_f32(za2, za3)))
    }
}

/// Evaluates a 64-element dot product using 4 independent NEON accumulators (`za0..za3`).
///
/// # Safety
/// Caller must ensure `a` and `b` are valid `[f32; 64]` slices on `aarch64`.
#[inline(always)]
pub(crate) unsafe fn neon_dot64(a: &[f32; 64], b: &[f32; 64]) -> f32 {
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    // SAFETY: Offsets `0..64` in steps of 16 are in-bounds for `[f32; 64]`.
    unsafe {
        let mut za0 = vdupq_n_f32(0.0);
        let mut za1 = vdupq_n_f32(0.0);
        let mut za2 = vdupq_n_f32(0.0);
        let mut za3 = vdupq_n_f32(0.0);

        let mut off = 0_usize;
        while off < 64 {
            za0 = vfmaq_f32(za0, vld1q_f32(a_ptr.add(off)), vld1q_f32(b_ptr.add(off)));
            za1 = vfmaq_f32(
                za1,
                vld1q_f32(a_ptr.add(off + 4)),
                vld1q_f32(b_ptr.add(off + 4)),
            );
            za2 = vfmaq_f32(
                za2,
                vld1q_f32(a_ptr.add(off + 8)),
                vld1q_f32(b_ptr.add(off + 8)),
            );
            za3 = vfmaq_f32(
                za3,
                vld1q_f32(a_ptr.add(off + 12)),
                vld1q_f32(b_ptr.add(off + 12)),
            );
            off += 16;
        }

        let sum01 = vaddq_f32(za0, za1);
        let sum23 = vaddq_f32(za2, za3);
        vaddvq_f32(vaddq_f32(sum01, sum23))
    }
}

/// Evaluates a fused stereo (`L` and `R`) 64-element dot product in a single pass over `taps_64`,
/// halving feature-buffer memory bandwidth.
///
/// # Safety
/// Caller must ensure `taps`, `w_l`, and `w_r` are valid `[f32; 64]` slices on `aarch64`.
#[inline(always)]
pub(crate) unsafe fn neon_dot64_stereo(
    taps: &[f32; 64],
    w_l: &[f32; 64],
    w_r: &[f32; 64],
) -> [f32; 2] {
    let t_ptr = taps.as_ptr();
    let wl_ptr = w_l.as_ptr();
    let wr_ptr = w_r.as_ptr();
    // SAFETY: Offsets `0..64` in steps of 16 are in-bounds for `[f32; 64]`.
    unsafe {
        let mut l0 = vdupq_n_f32(0.0);
        let mut l1 = vdupq_n_f32(0.0);
        let mut l2 = vdupq_n_f32(0.0);
        let mut l3 = vdupq_n_f32(0.0);
        let mut r0 = vdupq_n_f32(0.0);
        let mut r1 = vdupq_n_f32(0.0);
        let mut r2 = vdupq_n_f32(0.0);
        let mut r3 = vdupq_n_f32(0.0);

        let mut off = 0_usize;
        while off < 64 {
            let vt0 = vld1q_f32(t_ptr.add(off));
            l0 = vfmaq_f32(l0, vt0, vld1q_f32(wl_ptr.add(off)));
            r0 = vfmaq_f32(r0, vt0, vld1q_f32(wr_ptr.add(off)));

            let vt1 = vld1q_f32(t_ptr.add(off + 4));
            l1 = vfmaq_f32(l1, vt1, vld1q_f32(wl_ptr.add(off + 4)));
            r1 = vfmaq_f32(r1, vt1, vld1q_f32(wr_ptr.add(off + 4)));

            let vt2 = vld1q_f32(t_ptr.add(off + 8));
            l2 = vfmaq_f32(l2, vt2, vld1q_f32(wl_ptr.add(off + 8)));
            r2 = vfmaq_f32(r2, vt2, vld1q_f32(wr_ptr.add(off + 8)));

            let vt3 = vld1q_f32(t_ptr.add(off + 12));
            l3 = vfmaq_f32(l3, vt3, vld1q_f32(wl_ptr.add(off + 12)));
            r3 = vfmaq_f32(r3, vt3, vld1q_f32(wr_ptr.add(off + 12)));

            off += 16;
        }

        let out_l = vaddvq_f32(vaddq_f32(vaddq_f32(l0, l1), vaddq_f32(l2, l3)));
        let out_r = vaddvq_f32(vaddq_f32(vaddq_f32(r0, r1), vaddq_f32(r2, r3)));
        [out_l, out_r]
    }
}

/// Hardware ARMv8-A Castagnoli `CRC32C` (`crc32cx` / `crc32cb`) incremental update.
#[inline]
pub(crate) fn crc32c_arm64_update(crc_init: u32, data: &[u8]) -> u32 {
    // SAFETY: ARMv8.1+ (Apple Silicon M1..M4 and Rockchip RK3582 Cortex-A76/A55) implement
    // mandatory hardware CRC32C instructions.
    unsafe { crc32c_arm64_hw_inner(crc_init, data) }
}

#[target_feature(enable = "crc")]
#[inline]
unsafe fn crc32c_arm64_hw_inner(crc_init: u32, data: &[u8]) -> u32 {
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
