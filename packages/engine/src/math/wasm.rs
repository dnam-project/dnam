//! Unified WebAssembly (`wasm32`) Hardware Math Library (`WasmSimd128Backend` & `WasmVec4`).
//!
//! Uses `core::arch::wasm32` 128-bit `v128` SIMD intrinsics when `target_feature = "simd128"` is
//! enabled, with a portable `#[repr(C, align(16))]` fallback when `simd128` is not enabled.

use super::{BackendKind, MathBackend, SimdVec4};
use crate::math::crc32c::{Crc32cBackend, crc32c_software_slice};

#[cfg(target_feature = "simd128")]
use core::arch::wasm32::{
    f32x4_abs, f32x4_add, f32x4_extract_lane, f32x4_mul, f32x4_pmax, f32x4_pmin,
    f32x4_replace_lane, f32x4_splat, f32x4_sub, i32x4_add, i32x4_all_true, i32x4_eq, i32x4_shuffle,
    i32x4_splat, i32x4_sub, u32x4, u32x4_ge, u32x4_lt, u32x4_min, u32x4_shl, u32x4_shr,
    u32x4_splat, u64x2_splat, v128, v128_and, v128_andnot, v128_any_true, v128_or, v128_xor,
};

/// Detects the WebAssembly backend (`WasmSimd128` when `target_feature = "simd128"` is active,
/// or `Scalar` otherwise).
#[inline]
pub const fn detect_wasm_backend() -> BackendKind {
    if cfg!(target_feature = "simd128") {
        BackendKind::WasmSimd128
    } else {
        BackendKind::Scalar
    }
}

/// WebAssembly 4-lane `f32` vector (`v128` when `simd128` is enabled, or `[f32; 4]` fallback).
#[cfg(target_feature = "simd128")]
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct WasmVec4(pub v128);

/// Fallback 4-lane vector on `wasm32` when `simd128` is not enabled at compile time.
#[cfg(not(target_feature = "simd128"))]
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct WasmVec4(pub [f32; 4]);

#[cfg(target_feature = "simd128")]
impl SimdVec4 for WasmVec4 {
    #[inline(always)]
    fn splat(val: f32) -> Self {
        Self(f32x4_splat(val))
    }

    #[inline(always)]
    fn splat_pair(a: f32, b: f32) -> Self {
        let pair = f32x4_replace_lane::<1>(f32x4_splat(a), b);
        Self(i32x4_shuffle::<0, 1, 0, 1>(pair, pair))
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
        Self(f32x4_add(self.0, rhs.0))
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(f32x4_sub(self.0, rhs.0))
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(f32x4_mul(self.0, rhs.0))
    }

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        Self(f32x4_add(self.0, f32x4_mul(a.0, b.0)))
    }

    #[inline(always)]
    fn fma_lane<const LANE: usize>(self, a: Self, v: Self) -> Self {
        const { assert!(LANE < 4) };
        let lane_v = match LANE {
            0 => i32x4_shuffle::<0, 0, 0, 0>(v.0, v.0),
            1 => i32x4_shuffle::<1, 1, 1, 1>(v.0, v.0),
            2 => i32x4_shuffle::<2, 2, 2, 2>(v.0, v.0),
            _ => i32x4_shuffle::<3, 3, 3, 3>(v.0, v.0),
        };
        Self(f32x4_add(self.0, f32x4_mul(a.0, lane_v)))
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        Self(f32x4_sub(self.0, f32x4_mul(a.0, b.0)))
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        Self(f32x4_pmax(rhs.0, self.0))
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        Self(f32x4_pmin(rhs.0, self.0))
    }

    #[inline(always)]
    fn abs(self) -> Self {
        Self(f32x4_abs(self.0))
    }

    #[inline(always)]
    fn pair_swap_neg(self) -> Self {
        let swapped = i32x4_shuffle::<1, 0, 3, 2>(self.0, self.0);
        Self(v128_xor(swapped, u64x2_splat(0x0000_0000_8000_0000)))
    }

    #[inline(always)]
    fn flush_subnormal(self) -> Self {
        let is_sub = u32x4_lt(f32x4_abs(self.0), u32x4_splat(0x0080_0000));
        Self(v128_andnot(self.0, u32x4_shr(is_sub, 1)))
    }

    #[inline(always)]
    fn horizontal_sum(self) -> f32 {
        let pair = f32x4_add(self.0, i32x4_shuffle::<1, 0, 3, 2>(self.0, self.0));
        let total = f32x4_add(pair, i32x4_shuffle::<2, 3, 0, 1>(pair, pair));
        f32x4_extract_lane::<0>(total)
    }

    #[inline(always)]
    fn horizontal_sum_pair(self, rhs: Self) -> (f32, f32) {
        let evens = i32x4_shuffle::<0, 2, 4, 6>(self.0, rhs.0);
        let odds = i32x4_shuffle::<1, 3, 5, 7>(self.0, rhs.0);
        let pairs = f32x4_add(evens, odds);
        let swapped = i32x4_shuffle::<1, 0, 3, 2>(pairs, pairs);
        let totals = f32x4_add(pairs, swapped);
        (
            f32x4_extract_lane::<0>(totals),
            f32x4_extract_lane::<2>(totals),
        )
    }
}

#[cfg(not(target_feature = "simd128"))]
impl SimdVec4 for WasmVec4 {
    #[inline(always)]
    fn splat(val: f32) -> Self {
        Self([val; 4])
    }

    #[inline(always)]
    fn load(src: &[f32; 4]) -> Self {
        Self(*src)
    }

    #[inline(always)]
    fn store(self, dst: &mut [f32; 4]) {
        *dst = self.0;
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(
            super::scalar::ScalarVec4(self.0)
                .add(super::scalar::ScalarVec4(rhs.0))
                .0,
        )
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(
            super::scalar::ScalarVec4(self.0)
                .sub(super::scalar::ScalarVec4(rhs.0))
                .0,
        )
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(
            super::scalar::ScalarVec4(self.0)
                .mul(super::scalar::ScalarVec4(rhs.0))
                .0,
        )
    }

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        Self(
            super::scalar::ScalarVec4(self.0)
                .fma(
                    super::scalar::ScalarVec4(a.0),
                    super::scalar::ScalarVec4(b.0),
                )
                .0,
        )
    }

    #[inline(always)]
    fn fma_lane<const LANE: usize>(self, a: Self, v: Self) -> Self {
        Self(
            super::scalar::ScalarVec4(self.0)
                .fma_lane::<LANE>(
                    super::scalar::ScalarVec4(a.0),
                    super::scalar::ScalarVec4(v.0),
                )
                .0,
        )
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        Self(
            super::scalar::ScalarVec4(self.0)
                .fms(
                    super::scalar::ScalarVec4(a.0),
                    super::scalar::ScalarVec4(b.0),
                )
                .0,
        )
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        Self(
            super::scalar::ScalarVec4(self.0)
                .max(super::scalar::ScalarVec4(rhs.0))
                .0,
        )
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        Self(
            super::scalar::ScalarVec4(self.0)
                .min(super::scalar::ScalarVec4(rhs.0))
                .0,
        )
    }

    #[inline(always)]
    fn abs(self) -> Self {
        Self(super::scalar::ScalarVec4(self.0).abs().0)
    }

    #[inline(always)]
    fn horizontal_sum(self) -> f32 {
        super::scalar::ScalarVec4(self.0).horizontal_sum()
    }
}

/// WebAssembly SIMD128 hardware math backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WasmSimd128Backend;

impl Crc32cBackend for WasmSimd128Backend {
    #[inline]
    fn crc32c_update(crc: u32, data: &[u8]) -> u32 {
        crc32c_software_slice(crc, data)
    }
}

impl MathBackend for WasmSimd128Backend {
    type Vec4 = WasmVec4;
    const NAME: &'static str = "WebAssembly SIMD128 (v128 f32x4)";
    const KIND: BackendKind = BackendKind::WasmSimd128;

    #[cfg(target_feature = "simd128")]
    #[inline(always)]
    fn fma_mat4x4_vec4(acc: WasmVec4, mat: &[[f32; 4]; 4], x: &[f32; 4]) -> WasmVec4 {
        let vx = WasmVec4::load(x);
        acc.fma_lane::<0>(WasmVec4::load(&mat[0]), vx)
            .fma_lane::<1>(WasmVec4::load(&mat[1]), vx)
            .fma_lane::<2>(WasmVec4::load(&mat[2]), vx)
            .fma_lane::<3>(WasmVec4::load(&mat[3]), vx)
    }

    #[cfg(target_feature = "simd128")]
    #[inline(always)]
    fn f32x4_to_fp16x4(src: &[f32; 4]) -> [u16; 4] {
        let bits: v128 = zerocopy::transmute!(*src);
        let sign = u32x4_shr(v128_and(bits, u32x4_splat(0x8000_0000)), 16);
        let abs_bits = v128_and(bits, u32x4_splat(0x7fff_ffff));
        let is_tiny = u32x4_lt(abs_bits, u32x4_splat(0x3300_0000));
        let is_normal = v128_and(
            u32x4_ge(abs_bits, u32x4_splat(0x3880_0000)),
            u32x4_lt(abs_bits, u32x4_splat(0x477f_e000)),
        );
        if i32x4_all_true(v128_or(is_tiny, is_normal)) {
            let odd_bit = v128_and(u32x4_shr(abs_bits, 13), u32x4_splat(1));
            let rounded = i32x4_add(abs_bits, i32x4_add(u32x4_splat(0x0000_0fff), odd_bit));
            let biased = i32x4_sub(rounded, u32x4_splat(112 << 23));
            let half_mag = v128_andnot(u32x4_shr(biased, 13), is_tiny);
            let combined = v128_or(sign, half_mag);
            let lanes: [u32; 4] = zerocopy::transmute!(combined);
            return [
                lanes[0] as u16,
                lanes[1] as u16,
                lanes[2] as u16,
                lanes[3] as u16,
            ];
        }
        [
            super::f32_to_fp16(src[0]),
            super::f32_to_fp16(src[1]),
            super::f32_to_fp16(src[2]),
            super::f32_to_fp16(src[3]),
        ]
    }

    #[cfg(target_feature = "simd128")]
    #[inline(always)]
    fn fp16x4_to_f32x4(src: &[u16; 4]) -> [f32; 4] {
        let h = u32x4(
            u32::from(src[0]),
            u32::from(src[1]),
            u32::from(src[2]),
            u32::from(src[3]),
        );
        let sign = u32x4_shl(v128_and(h, u32x4_splat(0x8000)), 16);
        let exp = v128_and(u32x4_shr(h, 10), u32x4_splat(0x1f));
        let mant = v128_and(h, u32x4_splat(0x03ff));
        let is_zero_exp = i32x4_eq(exp, i32x4_splat(0));
        let is_inf_nan = i32x4_eq(exp, i32x4_splat(31));
        let is_subnormal = v128_andnot(is_zero_exp, i32x4_eq(mant, i32x4_splat(0)));
        if !v128_any_true(v128_or(is_subnormal, is_inf_nan)) {
            let norm_bits = v128_or(
                u32x4_shl(i32x4_add(exp, i32x4_splat(112)), 23),
                u32x4_shl(mant, 13),
            );
            let f32_bits = v128_or(sign, v128_andnot(norm_bits, is_zero_exp));
            return zerocopy::transmute!(f32_bits);
        }
        [
            super::fp16_to_f32(src[0]),
            super::fp16_to_f32(src[1]),
            super::fp16_to_f32(src[2]),
            super::fp16_to_f32(src[3]),
        ]
    }

    #[cfg(target_feature = "simd128")]
    #[inline(always)]
    fn rsqrt_nr2_f32(v0: f32) -> f32 {
        wasm_rsqrt_nr2_pair_f32(v0, 1.0).0
    }

    #[cfg(target_feature = "simd128")]
    #[inline(always)]
    fn rsqrt_nr2_pair_f32(v0: f32, v1: f32) -> (f32, f32) {
        wasm_rsqrt_nr2_pair_f32(v0, v1)
    }
}

/// Evaluates 2-step Newton `rsqrt` (`0x5f37_59df - (v >> 1)`) for a stereo pair `(v0, v1)`
/// entirely inside `v128` registers without scalar `i32.reinterpret_f32` transfers (`0 ULP` difference).
#[cfg(target_feature = "simd128")]
#[inline(always)]
pub(crate) fn wasm_rsqrt_nr2_pair_f32(v0: f32, v1: f32) -> (f32, f32) {
    let v = f32x4_replace_lane::<1>(f32x4_splat(v0), v1);
    let magic = u32x4_splat(0x5f37_59df);
    let half = f32x4_splat(0.5);
    let three_halves = f32x4_splat(1.5);
    let hv = f32x4_mul(half, v);
    let mut est = i32x4_sub(magic, u32x4_shr(v, 1));
    for _ in 0..2 {
        let corr = f32x4_sub(three_halves, f32x4_mul(f32x4_mul(hv, est), est));
        est = f32x4_mul(est, corr);
    }
    (f32x4_extract_lane::<0>(est), f32x4_extract_lane::<1>(est))
}

/// Evaluates `T1` tracker exponent-split scales `(sd_l, su_l, sd_r, su_r)` for normal positive
/// maxima `(max_l, max_r)` entirely inside `v128` registers without scalar `i32.reinterpret_f32` transfers.
#[cfg(target_feature = "simd128")]
#[inline(always)]
pub(crate) fn wasm_tracker_exp_scales_f32(max_l: f32, max_r: f32) -> (f32, f32, f32, f32) {
    let max_v = f32x4_replace_lane::<1>(f32x4_splat(max_l), max_r);
    let exp_bits = v128_and(max_v, u32x4_splat(0x7f80_0000));
    let su_v = u32x4_min(exp_bits, u32x4_splat(253 << 23));
    let sd_v = i32x4_sub(u32x4_splat(254 << 23), su_v);
    (
        f32x4_extract_lane::<0>(sd_v),
        f32x4_extract_lane::<0>(su_v),
        f32x4_extract_lane::<1>(sd_v),
        f32x4_extract_lane::<1>(su_v),
    )
}
