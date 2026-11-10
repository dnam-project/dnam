//! x86-64 AVX2 + FMA + F16C (`x86-64-v3`) hardware math backend (`Avx2FmaVec4` & `X86Avx2FmaBackend`).

use core::arch::x86_64::{
    __m128, __m128i, _MM_FROUND_TO_NEAREST_INT, _mm_add_ps, _mm_and_si128, _mm_broadcast_ss,
    _mm_castps_si128, _mm_castsi128_ps, _mm_cvtph_ps, _mm_cvtps_ph, _mm_cvtss_f32, _mm_fmadd_ps,
    _mm_fnmadd_ps, _mm_max_ps, _mm_min_ps, _mm_mul_ps, _mm_permute_ps, _mm_set1_epi32, _mm_set1_ps,
    _mm_setr_ps, _mm_shuffle_ps, _mm_srli_epi32, _mm_sub_epi32, _mm_sub_ps, _mm_xor_ps,
    _mm256_add_ps, _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps,
    _mm256_mul_ps,
};
#[cfg(not(any(test, feature = "std")))]
use core::sync::atomic::AtomicU8;

use super::sse2::Sse2Vec4;
use super::sse42::{Sse42Vec4, X86Sse42Backend, sse3_horizontal_sum, sse3_horizontal_sum_pair};
use crate::math::crc32c::Crc32cBackend;
use crate::math::{BackendKind, MathBackend, SimdVec4};

/// 128-bit x86-64 AVX2/FMA 4-lane `f32` register wrapper (`__m128`).
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct Avx2FmaVec4(pub __m128);

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_fma(acc: __m128, a: __m128, b: __m128) -> __m128 {
    _mm_fmadd_ps(a, b, acc)
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_fma_lane<const LANE: usize>(acc: __m128, a: __m128, v: __m128) -> __m128 {
    let lane_v = match LANE {
        0 => _mm_permute_ps::<0x00>(v),
        1 => _mm_permute_ps::<0x55>(v),
        2 => _mm_permute_ps::<0xAA>(v),
        _ => _mm_permute_ps::<0xFF>(v),
    };
    _mm_fmadd_ps(a, lane_v, acc)
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_fms(acc: __m128, a: __m128, b: __m128) -> __m128 {
    _mm_fnmadd_ps(a, b, acc)
}

#[target_feature(enable = "avx")]
#[inline]
fn avx2_pair_swap_neg(v: __m128) -> __m128 {
    _mm_xor_ps(_mm_permute_ps::<0xB1>(v), _mm_setr_ps(-0.0, 0.0, -0.0, 0.0))
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_fma_mat4x4_vec4(acc: __m128, mat: &[[f32; 4]; 4], x: &[f32; 4]) -> __m128 {
    let mut out = _mm_fmadd_ps(_mm_broadcast_ss(&x[0]), Avx2FmaVec4::load(&mat[0]).0, acc);
    out = _mm_fmadd_ps(_mm_broadcast_ss(&x[1]), Avx2FmaVec4::load(&mat[1]).0, out);
    out = _mm_fmadd_ps(_mm_broadcast_ss(&x[2]), Avx2FmaVec4::load(&mat[2]).0, out);
    _mm_fmadd_ps(_mm_broadcast_ss(&x[3]), Avx2FmaVec4::load(&mat[3]).0, out)
}

#[target_feature(enable = "avx")]
#[inline]
fn avx_reduce256(v: core::arch::x86_64::__m256) -> __m128 {
    _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v))
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_dot16_1ch(x: &[[f32; 4]; 4], w: &[[f32; 4]; 4]) -> __m128 {
    // SAFETY: `[[f32; 4]; 4]` is 16 contiguous `f32` elements (`2 x __m256`).
    unsafe {
        let (x0, w0) = (
            _mm256_loadu_ps(x[0].as_ptr()),
            _mm256_loadu_ps(w[0].as_ptr()),
        );
        let (x1, w1) = (
            _mm256_loadu_ps(x[2].as_ptr()),
            _mm256_loadu_ps(w[2].as_ptr()),
        );
        avx_reduce256(_mm256_fmadd_ps(x1, w1, _mm256_mul_ps(x0, w0)))
    }
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_dot16_stereo_hw(
    x_l: &[[f32; 4]; 4],
    w_l: &[[f32; 4]; 4],
    x_r: &[[f32; 4]; 4],
    w_r: &[[f32; 4]; 4],
) -> (f32, f32) {
    sse3_horizontal_sum_pair(avx2_dot16_1ch(x_l, w_l), avx2_dot16_1ch(x_r, w_r))
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_tile_dot64_hw(a: &[f32; 64], b: &[f32; 64]) -> f32 {
    // SAFETY: `[f32; 64]` has 8 contiguous 8-lane `f32` slices.
    unsafe {
        let mut acc0 = _mm256_mul_ps(_mm256_loadu_ps(a.as_ptr()), _mm256_loadu_ps(b.as_ptr()));
        let mut acc1 = _mm256_mul_ps(
            _mm256_loadu_ps(a.as_ptr().add(8)),
            _mm256_loadu_ps(b.as_ptr().add(8)),
        );
        for k in (16..64).step_by(16) {
            acc0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(k)),
                _mm256_loadu_ps(b.as_ptr().add(k)),
                acc0,
            );
            acc1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(k + 8)),
                _mm256_loadu_ps(b.as_ptr().add(k + 8)),
                acc1,
            );
        }
        sse3_horizontal_sum(avx_reduce256(_mm256_add_ps(acc0, acc1)))
    }
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_tile_dot64_stereo_hw(taps: &[f32; 64], w_l: &[f32; 64], w_r: &[f32; 64]) -> [f32; 2] {
    // SAFETY: `[f32; 64]` has 8 contiguous 8-lane `f32` slices.
    unsafe {
        let (t0, t1) = (
            _mm256_loadu_ps(taps.as_ptr()),
            _mm256_loadu_ps(taps.as_ptr().add(8)),
        );
        let mut l0 = _mm256_mul_ps(t0, _mm256_loadu_ps(w_l.as_ptr()));
        let mut r0 = _mm256_mul_ps(t0, _mm256_loadu_ps(w_r.as_ptr()));
        let mut l1 = _mm256_mul_ps(t1, _mm256_loadu_ps(w_l.as_ptr().add(8)));
        let mut r1 = _mm256_mul_ps(t1, _mm256_loadu_ps(w_r.as_ptr().add(8)));
        for k in (16..64).step_by(16) {
            let (tk0, tk1) = (
                _mm256_loadu_ps(taps.as_ptr().add(k)),
                _mm256_loadu_ps(taps.as_ptr().add(k + 8)),
            );
            l0 = _mm256_fmadd_ps(tk0, _mm256_loadu_ps(w_l.as_ptr().add(k)), l0);
            r0 = _mm256_fmadd_ps(tk0, _mm256_loadu_ps(w_r.as_ptr().add(k)), r0);
            l1 = _mm256_fmadd_ps(tk1, _mm256_loadu_ps(w_l.as_ptr().add(k + 8)), l1);
            r1 = _mm256_fmadd_ps(tk1, _mm256_loadu_ps(w_r.as_ptr().add(k + 8)), r1);
        }
        let (sl128, sr128) = (
            avx_reduce256(_mm256_add_ps(l0, l1)),
            avx_reduce256(_mm256_add_ps(r0, r1)),
        );
        let (l, r) = sse3_horizontal_sum_pair(sl128, sr128);
        [l, r]
    }
}

/// Evaluates a 16-element stereo dot product using 256-bit AVX2/FMA when supported at runtime.
#[inline(always)]
pub(crate) fn avx2_dot16_stereo(
    x_l: &[[f32; 4]; 4],
    w_l: &[[f32; 4]; 4],
    x_r: &[[f32; 4]; 4],
    w_r: &[[f32; 4]; 4],
) -> Option<(f32, f32)> {
    if X86Avx2FmaBackend::is_supported() {
        // SAFETY: Guarded by `X86Avx2FmaBackend::is_supported()`.
        Some(unsafe { avx2_dot16_stereo_hw(x_l, w_l, x_r, w_r) })
    } else {
        None
    }
}

/// Evaluates a 64-element dot product using 256-bit AVX2/FMA when supported at runtime.
#[inline(always)]
pub(crate) fn avx2_tile_dot64(a: &[f32; 64], b: &[f32; 64]) -> Option<f32> {
    if X86Avx2FmaBackend::is_supported() {
        // SAFETY: Guarded by `X86Avx2FmaBackend::is_supported()`.
        Some(unsafe { avx2_tile_dot64_hw(a, b) })
    } else {
        None
    }
}

/// Evaluates a dual 64-element stereo dot product using 256-bit AVX2/FMA when supported at runtime.
#[inline(always)]
pub(crate) fn avx2_tile_dot64_stereo(
    taps: &[f32; 64],
    w_l: &[f32; 64],
    w_r: &[f32; 64],
) -> Option<[f32; 2]> {
    if X86Avx2FmaBackend::is_supported() {
        // SAFETY: Guarded by `X86Avx2FmaBackend::is_supported()`.
        Some(unsafe { avx2_tile_dot64_stereo_hw(taps, w_l, w_r) })
    } else {
        None
    }
}

#[target_feature(enable = "avx,f16c")]
#[inline]
fn avx2_f32x4_to_fp16x4(src: &[f32; 4]) -> [u16; 4] {
    let v = _mm_max_ps(
        _mm_min_ps(Avx2FmaVec4::load(src).0, _mm_set1_ps(65504.0)),
        _mm_set1_ps(-65504.0),
    );
    let words: [u16; 8] = zerocopy::transmute!(_mm_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(v));
    [words[0], words[1], words[2], words[3]]
}

#[target_feature(enable = "avx,f16c")]
#[inline]
fn avx2_fp16x4_to_f32x4(src: &[u16; 4]) -> [f32; 4] {
    let ph: __m128i = zerocopy::transmute!([src[0], src[1], src[2], src[3], 0, 0, 0, 0]);
    zerocopy::transmute!(_mm_cvtph_ps(ph))
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_rsqrt_nr2_pair_f32(v0: f32, v1: f32) -> (f32, f32) {
    let v = _mm_setr_ps(v0, v1, 1.0, 1.0);
    let magic = _mm_set1_epi32(0x5f37_59df);
    let (half, three_halves) = (_mm_set1_ps(0.5), _mm_set1_ps(1.5));
    let hv = _mm_mul_ps(half, v);
    let mut est = _mm_castsi128_ps(_mm_sub_epi32(
        magic,
        _mm_and_si128(
            _mm_srli_epi32::<1>(_mm_castps_si128(v)),
            _mm_set1_epi32(i32::MAX),
        ),
    ));
    for _ in 0..2 {
        est = _mm_mul_ps(
            est,
            _mm_sub_ps(three_halves, _mm_mul_ps(_mm_mul_ps(hv, est), est)),
        );
    }
    (
        _mm_cvtss_f32(est),
        _mm_cvtss_f32(_mm_shuffle_ps::<0x55>(est, est)),
    )
}

impl SimdVec4 for Avx2FmaVec4 {
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
        if X86Avx2FmaBackend::is_supported() {
            // SAFETY: Guarded by `X86Avx2FmaBackend::is_supported()`.
            unsafe { Self(avx2_fma(self.0, a.0, b.0)) }
        } else {
            Self(Sse42Vec4(self.0).fma(Sse42Vec4(a.0), Sse42Vec4(b.0)).0)
        }
    }

    #[inline(always)]
    fn fma_lane<const LANE: usize>(self, a: Self, v: Self) -> Self {
        const { assert!(LANE < 4) };
        if X86Avx2FmaBackend::is_supported() {
            // SAFETY: Guarded by `X86Avx2FmaBackend::is_supported()`.
            unsafe { Self(avx2_fma_lane::<LANE>(self.0, a.0, v.0)) }
        } else {
            Self(
                Sse42Vec4(self.0)
                    .fma_lane::<LANE>(Sse42Vec4(a.0), Sse42Vec4(v.0))
                    .0,
            )
        }
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        if X86Avx2FmaBackend::is_supported() {
            // SAFETY: Guarded by `X86Avx2FmaBackend::is_supported()`.
            unsafe { Self(avx2_fms(self.0, a.0, b.0)) }
        } else {
            Self(Sse42Vec4(self.0).fms(Sse42Vec4(a.0), Sse42Vec4(b.0)).0)
        }
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
        if X86Avx2FmaBackend::is_supported() {
            // SAFETY: Guarded by `X86Avx2FmaBackend::is_supported()`.
            unsafe { Self(avx2_pair_swap_neg(self.0)) }
        } else {
            Self(Sse42Vec4(self.0).pair_swap_neg().0)
        }
    }

    #[inline(always)]
    fn flush_subnormal(self) -> Self {
        Self(Sse2Vec4(self.0).flush_subnormal().0)
    }

    #[inline(always)]
    fn horizontal_sum(self) -> f32 {
        Sse42Vec4(self.0).horizontal_sum()
    }

    #[inline(always)]
    fn horizontal_sum_pair(self, rhs: Self) -> (f32, f32) {
        Sse42Vec4(self.0).horizontal_sum_pair(Sse42Vec4(rhs.0))
    }
}

/// x86-64 AVX2 + FMA + F16C (`x86-64-v3`) hardware math backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct X86Avx2FmaBackend;

#[cfg(not(any(test, feature = "std")))]
static AVX2_SUPPORT_CACHE: AtomicU8 = AtomicU8::new(0);

impl X86Avx2FmaBackend {
    /// Returns `true` if the host x86-64 CPU and OS support `AVX`, `AVX2`, `FMA`, and `F16C`
    /// via [`std::arch::is_x86_feature_detected!`].
    #[inline]
    pub fn is_supported() -> bool {
        #[cfg(any(test, feature = "std"))]
        {
            std::arch::is_x86_feature_detected!("avx")
                && std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
                && std::arch::is_x86_feature_detected!("f16c")
        }
        #[cfg(not(any(test, feature = "std")))]
        {
            super::sse42::cached_cpuid_support(&AVX2_SUPPORT_CACHE, Self::detect_cpuid)
        }
    }

    #[cfg(not(any(test, feature = "std")))]
    fn detect_cpuid() -> bool {
        // SAFETY: `__cpuid` is always available on x86-64; `_xgetbv` is guarded by `OSXSAVE` (bit 27).
        unsafe {
            if core::arch::x86_64::__cpuid(0).eax < 7 {
                return false;
            }
            let cpuid1 = core::arch::x86_64::__cpuid(1);
            let req = (1u32 << 12) | (1u32 << 27) | (1u32 << 28) | (1u32 << 29);
            if (cpuid1.ecx & req) != req || (core::arch::x86_64::_xgetbv(0) & 0x6) != 0x6 {
                return false;
            }
            (core::arch::x86_64::__cpuid_count(7, 0).ebx & (1u32 << 5)) != 0
        }
    }
}

impl Crc32cBackend for X86Avx2FmaBackend {
    #[inline]
    fn crc32c_update(crc: u32, data: &[u8]) -> u32 {
        X86Sse42Backend::crc32c_update(crc, data)
    }
}

impl MathBackend for X86Avx2FmaBackend {
    type Vec4 = Avx2FmaVec4;
    const NAME: &'static str = "x86-64 AVX2 + FMA + F16C";
    const KIND: BackendKind = BackendKind::X86Avx2Fma;

    #[inline(always)]
    fn fma_mat4x4_vec4(acc: Avx2FmaVec4, mat: &[[f32; 4]; 4], x: &[f32; 4]) -> Avx2FmaVec4 {
        if Self::is_supported() {
            // SAFETY: Guarded by `Self::is_supported()`.
            unsafe { Avx2FmaVec4(avx2_fma_mat4x4_vec4(acc.0, mat, x)) }
        } else {
            Avx2FmaVec4(X86Sse42Backend::fma_mat4x4_vec4(Sse42Vec4(acc.0), mat, x).0)
        }
    }

    #[inline(always)]
    fn f32x4_to_fp16x4(src: &[f32; 4]) -> [u16; 4] {
        if Self::is_supported() {
            // SAFETY: Guarded by `Self::is_supported()`.
            unsafe { avx2_f32x4_to_fp16x4(src) }
        } else {
            X86Sse42Backend::f32x4_to_fp16x4(src)
        }
    }

    #[inline(always)]
    fn fp16x4_to_f32x4(src: &[u16; 4]) -> [f32; 4] {
        if Self::is_supported() {
            // SAFETY: Guarded by `Self::is_supported()`.
            unsafe { avx2_fp16x4_to_f32x4(src) }
        } else {
            X86Sse42Backend::fp16x4_to_f32x4(src)
        }
    }

    #[inline(always)]
    fn rsqrt_nr2_f32(v0: f32) -> f32 {
        Self::rsqrt_nr2_pair_f32(v0, 1.0).0
    }

    #[inline(always)]
    fn rsqrt_nr2_pair_f32(v0: f32, v1: f32) -> (f32, f32) {
        if Self::is_supported() {
            // SAFETY: Guarded by `Self::is_supported()`.
            unsafe { avx2_rsqrt_nr2_pair_f32(v0, v1) }
        } else {
            X86Sse42Backend::rsqrt_nr2_pair_f32(v0, v1)
        }
    }
}
