//! x86-64 SSE4.2 / AVX2 + FMA hardware math backend (`X86SimdBackend`).

use core::arch::x86_64::{
    __m128, _mm_add_ps, _mm_andnot_ps, _mm_cvtss_f32, _mm_hadd_ps, _mm_loadu_ps, _mm_max_ps,
    _mm_min_ps, _mm_mul_ps, _mm_set1_ps, _mm_storeu_ps, _mm_sub_ps,
};

use super::{BackendKind, MathBackend, SimdVec4};

/// 128-bit x86-64 SSE/AVX 4-lane `f32` register wrapper (`__m128`).
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct X86Vec4(pub __m128);

impl SimdVec4 for X86Vec4 {
    #[inline(always)]
    fn splat(val: f32) -> Self {
        // SAFETY: Baseline `x86_64` guarantees SSE2 `_mm_set1_ps`.
        unsafe { Self(_mm_set1_ps(val)) }
    }

    #[inline(always)]
    fn load(src: &[f32; 4]) -> Self {
        // SAFETY: `src` is a valid slice of 4 `f32` elements (`16` bytes).
        unsafe { Self(_mm_loadu_ps(src.as_ptr())) }
    }

    #[inline(always)]
    fn store(self, dst: &mut [f32; 4]) {
        // SAFETY: `dst` is a valid mutable slice of 4 `f32` elements (`16` bytes).
        unsafe { _mm_storeu_ps(dst.as_mut_ptr(), self.0) }
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        // SAFETY: Baseline `x86_64` SSE operation.
        unsafe { Self(_mm_add_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        // SAFETY: Baseline `x86_64` SSE operation.
        unsafe { Self(_mm_sub_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // SAFETY: Baseline `x86_64` SSE operation.
        unsafe { Self(_mm_mul_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        #[cfg(target_feature = "fma")]
        {
            // SAFETY: `target_feature = "fma"` is enabled at compile time.
            unsafe { Self(core::arch::x86_64::_mm_fmadd_ps(a.0, b.0, self.0)) }
        }
        #[cfg(not(target_feature = "fma"))]
        {
            self.add(a.mul(b))
        }
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        #[cfg(target_feature = "fma")]
        {
            // SAFETY: `target_feature = "fma"` is enabled at compile time.
            unsafe { Self(core::arch::x86_64::_mm_fnmadd_ps(a.0, b.0, self.0)) }
        }
        #[cfg(not(target_feature = "fma"))]
        {
            self.sub(a.mul(b))
        }
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        // SAFETY: Baseline `x86_64` SSE operation.
        unsafe { Self(_mm_max_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        // SAFETY: Baseline `x86_64` SSE operation.
        unsafe { Self(_mm_min_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        // SAFETY: Clears the sign bit (`-0.0f32`) across all 4 lanes via `_mm_andnot_ps`.
        unsafe {
            let sign_mask = _mm_set1_ps(-0.0f32);
            Self(_mm_andnot_ps(sign_mask, self.0))
        }
    }

    #[inline(always)]
    fn horizontal_sum(self) -> f32 {
        #[cfg(target_feature = "sse3")]
        {
            // SAFETY: `target_feature = "sse3"` is enabled.
            unsafe {
                let s1 = _mm_hadd_ps(self.0, self.0);
                let s2 = _mm_hadd_ps(s1, s1);
                _mm_cvtss_f32(s2)
            }
        }
        #[cfg(not(target_feature = "sse3"))]
        {
            let mut arr = [0.0f32; 4];
            self.store(&mut arr);
            (arr[0] + arr[1]) + (arr[2] + arr[3])
        }
    }
}

/// x86-64 SSE4.2 / AVX2 + FMA hardware math backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct X86SimdBackend;

impl MathBackend for X86SimdBackend {
    type Vec4 = X86Vec4;
    const NAME: &'static str = "x86-64 SIMD (SSE4.2 / AVX2 + FMA)";
    const KIND: BackendKind = if cfg!(all(target_feature = "avx2", target_feature = "fma")) {
        BackendKind::X86Avx2Fma
    } else {
        BackendKind::X86Sse42
    };

    #[inline]
    fn crc32c(data: &[u8]) -> u32 {
        #[cfg(target_feature = "sse4.2")]
        {
            use core::arch::x86_64::{_mm_crc32_u8, _mm_crc32_u64};
            let mut crc = !0u64;
            let (chunks, remainder) = data.as_chunks::<8>();
            for chunk in chunks {
                let val = u64::from_le_bytes(*chunk);
                // SAFETY: `target_feature = "sse4.2"` is enabled.
                crc = unsafe { _mm_crc32_u64(crc, val) };
            }
            let mut crc32 = crc as u32;
            for &byte in remainder {
                // SAFETY: `target_feature = "sse4.2"` is enabled.
                crc32 = unsafe { _mm_crc32_u8(crc32, byte) };
            }
            return !crc32;
        }
        #[cfg(not(target_feature = "sse4.2"))]
        {
            super::crc32c_slice(0, data)
        }
    }
}
