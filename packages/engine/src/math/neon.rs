//! ARM64 128-bit NEON hardware backend (`NeonBackend`) for Rockchip RK3582 (Darkglass Anagram)
//! and Apple Silicon.
//!
//! Uses `core::arch::aarch64::float32x4_t` dual-pipe `FMLA` (`vfmaq_f32`), `FMLS` (`vfmsq_f32`),
//! branchless `FMAXNM` / `FMINNM` (`vmaxnmq_f32`, `vminnmq_f32`), across-vector reduction
//! `FADDP` (`vaddvq_f32`), and ARMv8-A hardware Castagnoli `CRC32C` (`__crc32cd`, `__crc32cb`).

use core::arch::aarch64::{
    float32x4_t, vabsq_f32, vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vfmsq_f32, vld1q_f32,
    vmaxnmq_f32, vminnmq_f32, vmulq_f32, vst1q_f32, vsubq_f32,
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

    #[inline]
    fn crc32c(data: &[u8]) -> u32 {
        #[cfg(target_feature = "crc")]
        {
            crc32c_arm64_hw(0, data)
        }
        #[cfg(not(target_feature = "crc"))]
        {
            super::crc32c_slice(0, data)
        }
    }
}

#[cfg(target_feature = "crc")]
#[inline]
fn crc32c_arm64_hw(crc_init: u32, data: &[u8]) -> u32 {
    use core::arch::aarch64::{__crc32cb, __crc32cd};

    let mut crc = !crc_init;
    let (chunks, remainder) = data.as_chunks::<8>();
    for chunk in chunks {
        let val = u64::from_le_bytes(*chunk);
        // SAFETY: `target_feature = "crc"` is enabled at compile time.
        crc = unsafe { __crc32cd(crc, val) };
    }
    for &byte in remainder {
        // SAFETY: `target_feature = "crc"` is enabled at compile time.
        crc = unsafe { __crc32cb(crc, byte) };
    }
    !crc
}
