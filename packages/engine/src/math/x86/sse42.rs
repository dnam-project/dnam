//! x86-64 SSE4.2 (`x86-64-v2`) hardware math backend (`Sse42Vec4` & `X86Sse42Backend`).

use core::arch::x86_64::{
    __m128, _mm_add_ps, _mm_add_ss, _mm_and_si128, _mm_castps_si128, _mm_castsi128_ps,
    _mm_crc32_u8, _mm_crc32_u64, _mm_cvtss_f32, _mm_hadd_ps, _mm_min_epi32, _mm_movehdup_ps,
    _mm_movehl_ps, _mm_set1_epi32, _mm_setr_ps, _mm_shuffle_ps, _mm_sub_epi32,
};
#[cfg(not(any(test, feature = "std")))]
use core::sync::atomic::{AtomicU8, Ordering};

use super::sse2::{Sse2Vec4, X86Sse2Backend, sse2_tracker_exp_scales_f32};
use crate::math::crc32c::Crc32cBackend;
use crate::math::{BackendKind, MathBackend, SimdVec4};

/// 128-bit x86-64 SSE4.2 4-lane `f32` register wrapper (`__m128`).
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct Sse42Vec4(pub __m128);

#[target_feature(enable = "sse3")]
#[inline]
pub(crate) fn sse3_horizontal_sum(v: __m128) -> f32 {
    let shuf = _mm_movehdup_ps(v);
    let pairs = _mm_add_ps(v, shuf);
    let hi = _mm_movehl_ps(pairs, pairs);
    _mm_cvtss_f32(_mm_add_ss(pairs, hi))
}

#[target_feature(enable = "sse3")]
#[inline]
pub(crate) fn sse3_horizontal_sum_pair(lhs: __m128, rhs: __m128) -> (f32, f32) {
    let pairs = _mm_hadd_ps(lhs, rhs);
    let totals = _mm_hadd_ps(pairs, pairs);
    (
        _mm_cvtss_f32(totals),
        _mm_cvtss_f32(_mm_shuffle_ps::<0x55>(totals, totals)),
    )
}

#[target_feature(enable = "sse4.1")]
#[inline]
fn sse41_tracker_exp_scales_hw(max_l: f32, max_r: f32) -> (f32, f32, f32, f32) {
    let max_v = _mm_setr_ps(max_l, max_r, 1.0, 1.0);
    let exp_bits = _mm_and_si128(_mm_castps_si128(max_v), _mm_set1_epi32(0x7f80_0000));
    let su_v = _mm_min_epi32(exp_bits, _mm_set1_epi32(253 << 23));
    let sd_v = _mm_sub_epi32(_mm_set1_epi32(254 << 23), su_v);
    let (sd_f, su_f) = (_mm_castsi128_ps(sd_v), _mm_castsi128_ps(su_v));
    (
        _mm_cvtss_f32(sd_f),
        _mm_cvtss_f32(su_f),
        _mm_cvtss_f32(_mm_shuffle_ps::<0x55>(sd_f, sd_f)),
        _mm_cvtss_f32(_mm_shuffle_ps::<0x55>(su_f, su_f)),
    )
}

/// Evaluates `T1` tracker exponent-split scales using SSE4.1 `_mm_min_epi32` when supported,
/// or SSE2 fallback otherwise.
#[inline(always)]
pub(crate) fn sse42_tracker_exp_scales_f32(max_l: f32, max_r: f32) -> (f32, f32, f32, f32) {
    if X86Sse42Backend::is_supported() {
        // SAFETY: Guarded by `X86Sse42Backend::is_supported()`.
        unsafe { sse41_tracker_exp_scales_hw(max_l, max_r) }
    } else {
        sse2_tracker_exp_scales_f32(max_l, max_r)
    }
}

#[target_feature(enable = "sse4.2")]
#[inline]
fn sse42_crc32c_update_hw(crc: u32, data: &[u8]) -> u32 {
    let mut state = u64::from(!crc);
    let (chunks, remainder) = data.as_chunks::<8>();
    for chunk in chunks {
        state = _mm_crc32_u64(state, u64::from_le_bytes(*chunk));
    }
    let mut crc32 = state as u32;
    for &byte in remainder {
        crc32 = _mm_crc32_u8(crc32, byte);
    }
    !crc32
}

#[cfg(not(any(test, feature = "std")))]
#[inline]
pub(crate) fn cached_cpuid_support(cache: &AtomicU8, detect: fn() -> bool) -> bool {
    let cached = cache.load(Ordering::Relaxed);
    if cached != 0 {
        return cached == 2;
    }
    let supported = detect();
    cache.store(if supported { 2 } else { 1 }, Ordering::Relaxed);
    supported
}

impl SimdVec4 for Sse42Vec4 {
    #[inline(always)]
    fn splat(val: f32) -> Self {
        Self(Sse2Vec4::splat(val).0)
    }

    #[inline(always)]
    fn splat_pair(a: f32, b: f32) -> Self {
        Self(Sse2Vec4::splat_pair(a, b).0)
    }

    #[inline(always)]
    fn load(src: &[f32; 4]) -> Self {
        Self(Sse2Vec4::load(src).0)
    }

    #[inline(always)]
    fn store(self, dst: &mut [f32; 4]) {
        Sse2Vec4(self.0).store(dst);
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(Sse2Vec4(self.0).add(Sse2Vec4(rhs.0)).0)
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(Sse2Vec4(self.0).sub(Sse2Vec4(rhs.0)).0)
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(Sse2Vec4(self.0).mul(Sse2Vec4(rhs.0)).0)
    }

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        Self(Sse2Vec4(self.0).fma(Sse2Vec4(a.0), Sse2Vec4(b.0)).0)
    }

    #[inline(always)]
    fn fma_lane<const LANE: usize>(self, a: Self, v: Self) -> Self {
        Self(
            Sse2Vec4(self.0)
                .fma_lane::<LANE>(Sse2Vec4(a.0), Sse2Vec4(v.0))
                .0,
        )
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        Self(Sse2Vec4(self.0).fms(Sse2Vec4(a.0), Sse2Vec4(b.0)).0)
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        Self(Sse2Vec4(self.0).max(Sse2Vec4(rhs.0)).0)
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        Self(Sse2Vec4(self.0).min(Sse2Vec4(rhs.0)).0)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        Self(Sse2Vec4(self.0).abs().0)
    }

    #[inline(always)]
    fn pair_swap_neg(self) -> Self {
        Self(Sse2Vec4(self.0).pair_swap_neg().0)
    }

    #[inline(always)]
    fn flush_subnormal(self) -> Self {
        Self(Sse2Vec4(self.0).flush_subnormal().0)
    }

    #[inline(always)]
    fn horizontal_sum(self) -> f32 {
        if X86Sse42Backend::is_supported() {
            // SAFETY: Guarded by `X86Sse42Backend::is_supported()`.
            unsafe { sse3_horizontal_sum(self.0) }
        } else {
            Sse2Vec4(self.0).horizontal_sum()
        }
    }

    #[inline(always)]
    fn horizontal_sum_pair(self, rhs: Self) -> (f32, f32) {
        if X86Sse42Backend::is_supported() {
            // SAFETY: Guarded by `X86Sse42Backend::is_supported()`.
            unsafe { sse3_horizontal_sum_pair(self.0, rhs.0) }
        } else {
            Sse2Vec4(self.0).horizontal_sum_pair(Sse2Vec4(rhs.0))
        }
    }
}

/// x86-64 SSE4.2 (`x86-64-v2`) hardware math backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct X86Sse42Backend;

#[cfg(not(any(test, feature = "std")))]
static SSE42_SUPPORT_CACHE: AtomicU8 = AtomicU8::new(0);

impl X86Sse42Backend {
    /// Returns `true` if the host x86-64 CPU supports `SSE3`, `SSSE3`, `SSE4.1`, and `SSE4.2`
    /// via [`std::arch::is_x86_feature_detected!`].
    #[inline]
    pub fn is_supported() -> bool {
        #[cfg(any(test, feature = "std"))]
        {
            std::arch::is_x86_feature_detected!("sse3")
                && std::arch::is_x86_feature_detected!("ssse3")
                && std::arch::is_x86_feature_detected!("sse4.1")
                && std::arch::is_x86_feature_detected!("sse4.2")
        }
        #[cfg(not(any(test, feature = "std")))]
        {
            cached_cpuid_support(&SSE42_SUPPORT_CACHE, || {
                // SAFETY: `__cpuid` is always supported on 64-bit x86-64 processors.
                unsafe {
                    let mask = (1u32 << 0) | (1u32 << 9) | (1u32 << 19) | (1u32 << 20);
                    (core::arch::x86_64::__cpuid(1).ecx & mask) == mask
                }
            })
        }
    }
}

impl Crc32cBackend for X86Sse42Backend {
    #[inline]
    fn crc32c_update(crc: u32, data: &[u8]) -> u32 {
        if Self::is_supported() {
            // SAFETY: Guarded by `Self::is_supported()`.
            unsafe { sse42_crc32c_update_hw(crc, data) }
        } else {
            X86Sse2Backend::crc32c_update(crc, data)
        }
    }
}

impl MathBackend for X86Sse42Backend {
    type Vec4 = Sse42Vec4;
    const NAME: &'static str = "x86-64 SSE4.2 (128-bit __m128 + CRC32C)";
    const KIND: BackendKind = BackendKind::X86Sse42;

    #[inline(always)]
    fn rsqrt_nr2_f32(v0: f32) -> f32 {
        X86Sse2Backend::rsqrt_nr2_f32(v0)
    }

    #[inline(always)]
    fn rsqrt_nr2_pair_f32(v0: f32, v1: f32) -> (f32, f32) {
        X86Sse2Backend::rsqrt_nr2_pair_f32(v0, v1)
    }
}
