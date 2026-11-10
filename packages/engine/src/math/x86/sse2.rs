//! x86-64 SSE2 (`x86-64-v1`) baseline hardware math backend (`Sse2Vec4` & `X86Sse2Backend`).

use core::arch::x86_64::{
    __m128, _mm_add_ps, _mm_and_si128, _mm_andnot_ps, _mm_andnot_si128, _mm_castps_si128,
    _mm_castsi128_ps, _mm_cmpeq_epi32, _mm_cmpgt_epi32, _mm_cvtss_f32, _mm_max_ps, _mm_min_ps,
    _mm_movehl_ps, _mm_mul_ps, _mm_or_si128, _mm_set1_epi32, _mm_set1_ps, _mm_setr_ps,
    _mm_shuffle_ps, _mm_srli_epi32, _mm_sub_epi32, _mm_sub_ps, _mm_xor_ps,
};

use crate::math::crc32c::{Crc32cBackend, crc32c_software_slice};
use crate::math::{BackendKind, MathBackend, SimdVec4};

/// 128-bit x86-64 SSE2 4-lane `f32` register wrapper (`__m128`).
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct Sse2Vec4(pub __m128);

impl SimdVec4 for Sse2Vec4 {
    #[inline(always)]
    fn splat(val: f32) -> Self {
        // SAFETY: SSE is part of the x86-64 architectural baseline.
        unsafe { Self(_mm_set1_ps(val)) }
    }

    #[inline(always)]
    fn splat_pair(a: f32, b: f32) -> Self {
        // SAFETY: SSE is part of the x86-64 architectural baseline.
        unsafe { Self(_mm_setr_ps(a, b, a, b)) }
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
        // SAFETY: SSE is part of the x86-64 architectural baseline.
        unsafe { Self(_mm_add_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        // SAFETY: SSE is part of the x86-64 architectural baseline.
        unsafe { Self(_mm_sub_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // SAFETY: SSE is part of the x86-64 architectural baseline.
        unsafe { Self(_mm_mul_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        self.add(a.mul(b))
    }

    #[inline(always)]
    fn fma_lane<const LANE: usize>(self, a: Self, v: Self) -> Self {
        const { assert!(LANE < 4) };
        // SAFETY: SSE is part of the x86-64 architectural baseline.
        let lane_v = unsafe {
            match LANE {
                0 => _mm_shuffle_ps::<0x00>(v.0, v.0),
                1 => _mm_shuffle_ps::<0x55>(v.0, v.0),
                2 => _mm_shuffle_ps::<0xAA>(v.0, v.0),
                _ => _mm_shuffle_ps::<0xFF>(v.0, v.0),
            }
        };
        self.fma(a, Self(lane_v))
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        self.sub(a.mul(b))
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        // SAFETY: SSE is part of the x86-64 architectural baseline.
        unsafe { Self(_mm_max_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        // SAFETY: SSE is part of the x86-64 architectural baseline.
        unsafe { Self(_mm_min_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        // SAFETY: SSE is part of the x86-64 architectural baseline.
        unsafe { Self(_mm_andnot_ps(_mm_set1_ps(-0.0f32), self.0)) }
    }

    #[inline(always)]
    fn pair_swap_neg(self) -> Self {
        // SAFETY: SSE is part of the x86-64 architectural baseline.
        unsafe {
            let swapped = _mm_shuffle_ps::<0xB1>(self.0, self.0);
            Self(_mm_xor_ps(swapped, _mm_setr_ps(-0.0, 0.0, -0.0, 0.0)))
        }
    }

    #[inline(always)]
    fn flush_subnormal(self) -> Self {
        // SAFETY: SSE2 is part of the x86-64 architectural baseline.
        unsafe {
            let bits = _mm_castps_si128(self.0);
            let exp = _mm_and_si128(bits, _mm_set1_epi32(0x7f80_0000));
            let is_zero_exp = _mm_cmpeq_epi32(exp, _mm_set1_epi32(0));
            let clear_mant = _mm_and_si128(is_zero_exp, _mm_set1_epi32(0x007f_ffff));
            Self(_mm_castsi128_ps(_mm_andnot_si128(clear_mant, bits)))
        }
    }

    #[inline(always)]
    fn horizontal_sum(self) -> f32 {
        // SAFETY: SSE is part of the x86-64 architectural baseline; pairwise `(s0 + s1) + (s2 + s3)`.
        unsafe {
            let odd = _mm_shuffle_ps::<0xB1>(self.0, self.0);
            let pair = _mm_add_ps(self.0, odd);
            let hi = _mm_movehl_ps(pair, pair);
            _mm_cvtss_f32(_mm_add_ps(pair, hi))
        }
    }

    #[inline(always)]
    fn horizontal_sum_pair(self, rhs: Self) -> (f32, f32) {
        // SAFETY: SSE is part of the x86-64 architectural baseline; pairwise `(s0 + s1) + (s2 + s3)`.
        unsafe {
            let evens = _mm_shuffle_ps::<0x88>(self.0, rhs.0);
            let odds = _mm_shuffle_ps::<0xDD>(self.0, rhs.0);
            let pairs = _mm_add_ps(evens, odds);
            let swapped = _mm_shuffle_ps::<0xB1>(pairs, pairs);
            let totals = _mm_add_ps(pairs, swapped);
            (
                _mm_cvtss_f32(totals),
                _mm_cvtss_f32(_mm_movehl_ps(totals, totals)),
            )
        }
    }
}

/// Evaluates `T1` tracker exponent-split scales `(sd_l, su_l, sd_r, su_r)` in SSE2 registers.
#[inline(always)]
pub(crate) fn sse2_tracker_exp_scales_f32(max_l: f32, max_r: f32) -> (f32, f32, f32, f32) {
    // SAFETY: SSE2 is part of the x86-64 architectural baseline; `exp_bits` is non-negative.
    unsafe {
        let max_v = _mm_setr_ps(max_l, max_r, 1.0, 1.0);
        let exp_bits = _mm_and_si128(_mm_castps_si128(max_v), _mm_set1_epi32(0x7f80_0000));
        let cap = _mm_set1_epi32(253 << 23);
        let gt_mask = _mm_cmpgt_epi32(exp_bits, cap);
        let su_v = _mm_or_si128(
            _mm_and_si128(gt_mask, cap),
            _mm_andnot_si128(gt_mask, exp_bits),
        );
        let sd_v = _mm_sub_epi32(_mm_set1_epi32(254 << 23), su_v);
        let (sd_f, su_f) = (_mm_castsi128_ps(sd_v), _mm_castsi128_ps(su_v));
        (
            _mm_cvtss_f32(sd_f),
            _mm_cvtss_f32(su_f),
            _mm_cvtss_f32(_mm_shuffle_ps::<0x55>(sd_f, sd_f)),
            _mm_cvtss_f32(_mm_shuffle_ps::<0x55>(su_f, su_f)),
        )
    }
}

/// x86-64 architectural baseline (`SSE` + `SSE2`) hardware math backend (`x86-64-v1`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct X86Sse2Backend;

impl X86Sse2Backend {
    /// SSE2 is guaranteed by the 64-bit x86-64 ABI.
    #[inline(always)]
    pub const fn is_supported() -> bool {
        true
    }
}

impl Crc32cBackend for X86Sse2Backend {
    #[inline]
    fn crc32c_update(crc: u32, data: &[u8]) -> u32 {
        crc32c_software_slice(crc, data)
    }
}

impl MathBackend for X86Sse2Backend {
    type Vec4 = Sse2Vec4;
    const NAME: &'static str = "x86-64 SSE2 (128-bit __m128)";
    const KIND: BackendKind = BackendKind::X86Sse2;

    #[inline(always)]
    fn rsqrt_nr2_f32(v0: f32) -> f32 {
        Self::rsqrt_nr2_pair_f32(v0, 1.0).0
    }

    #[inline(always)]
    fn rsqrt_nr2_pair_f32(v0: f32, v1: f32) -> (f32, f32) {
        // SAFETY: SSE2 is part of the x86-64 architectural baseline.
        unsafe {
            let v = _mm_setr_ps(v0, v1, 1.0, 1.0);
            let magic = _mm_set1_epi32(0x5f37_59df);
            let (half, three_halves) = (_mm_set1_ps(0.5), _mm_set1_ps(1.5));
            let hv = _mm_mul_ps(half, v);
            let mut est = _mm_castsi128_ps(_mm_sub_epi32(
                magic,
                _mm_srli_epi32::<1>(_mm_castps_si128(v)),
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
    }
}
