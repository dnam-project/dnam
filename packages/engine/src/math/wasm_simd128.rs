//! WebAssembly 128-bit SIMD (`v128`) hardware math backend (`WasmSimd128Backend`).

use super::{BackendKind, MathBackend, SimdVec4};

#[cfg(target_feature = "simd128")]
use core::arch::wasm32::{
    f32x4_abs, f32x4_add, f32x4_extract_lane, f32x4_mul, f32x4_pmax, f32x4_pmin, f32x4_splat,
    f32x4_sub, v128,
};

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
    fn horizontal_sum(self) -> f32 {
        (f32x4_extract_lane::<0>(self.0) + f32x4_extract_lane::<1>(self.0))
            + (f32x4_extract_lane::<2>(self.0) + f32x4_extract_lane::<3>(self.0))
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

impl MathBackend for WasmSimd128Backend {
    type Vec4 = WasmVec4;
    const NAME: &'static str = "WebAssembly SIMD128 (v128 f32x4)";
    const KIND: BackendKind = BackendKind::WasmSimd128;
}
