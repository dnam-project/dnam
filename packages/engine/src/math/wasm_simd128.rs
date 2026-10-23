//! WebAssembly 128-bit SIMD (`v128`) hardware math backend (`WasmSimd128Backend`).

use super::{BackendKind, MathBackend, SimdVec4};

#[cfg(target_feature = "simd128")]
use core::arch::wasm32::{
    f32x4_abs, f32x4_add, f32x4_extract_lane, f32x4_mul, f32x4_pmax, f32x4_pmin, f32x4_splat,
    f32x4_sub, i32x4_shuffle, v128,
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
        let pair = f32x4_add(self.0, i32x4_shuffle::<1, 0, 3, 2>(self.0, self.0));
        let total = f32x4_add(pair, i32x4_shuffle::<2, 3, 0, 1>(pair, pair));
        f32x4_extract_lane::<0>(total)
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

    #[cfg(target_feature = "simd128")]
    #[inline(always)]
    fn fma_mat4x4_vec4(acc: WasmVec4, mat: &[[f32; 4]; 4], x: &[f32; 4]) -> WasmVec4 {
        let vx = WasmVec4::load(x).0;
        let x0 = i32x4_shuffle::<0, 0, 0, 0>(vx, vx);
        let x1 = i32x4_shuffle::<1, 1, 1, 1>(vx, vx);
        let x2 = i32x4_shuffle::<2, 2, 2, 2>(vx, vx);
        let x3 = i32x4_shuffle::<3, 3, 3, 3>(vx, vx);
        let p0 = f32x4_add(acc.0, f32x4_mul(WasmVec4::load(&mat[0]).0, x0));
        let p1 = f32x4_mul(WasmVec4::load(&mat[1]).0, x1);
        let p2 = f32x4_mul(WasmVec4::load(&mat[2]).0, x2);
        let p3 = f32x4_mul(WasmVec4::load(&mat[3]).0, x3);
        WasmVec4(f32x4_add(f32x4_add(p0, p1), f32x4_add(p2, p3)))
    }

    #[cfg(target_feature = "simd128")]
    #[inline(always)]
    fn conv6_taps_1group(
        bias: WasmVec4,
        w_tiles: &[[[f32; 4]; 4]; 6],
        h_taps: &[&[[f32; 4]; 3]; 6],
    ) -> WasmVec4 {
        let mut za0 = bias.0;
        let mut za1 = f32x4_splat(0.0);
        let mut za2 = f32x4_splat(0.0);
        let mut za3 = f32x4_splat(0.0);

        for tap in 0..6 {
            let vx = WasmVec4::load(&h_taps[tap][0]).0;
            let wt = &w_tiles[tap];
            za0 = f32x4_add(
                za0,
                f32x4_mul(
                    WasmVec4::load(&wt[0]).0,
                    i32x4_shuffle::<0, 0, 0, 0>(vx, vx),
                ),
            );
            za1 = f32x4_add(
                za1,
                f32x4_mul(
                    WasmVec4::load(&wt[1]).0,
                    i32x4_shuffle::<1, 1, 1, 1>(vx, vx),
                ),
            );
            za2 = f32x4_add(
                za2,
                f32x4_mul(
                    WasmVec4::load(&wt[2]).0,
                    i32x4_shuffle::<2, 2, 2, 2>(vx, vx),
                ),
            );
            za3 = f32x4_add(
                za3,
                f32x4_mul(
                    WasmVec4::load(&wt[3]).0,
                    i32x4_shuffle::<3, 3, 3, 3>(vx, vx),
                ),
            );
        }

        WasmVec4(f32x4_add(f32x4_add(za0, za1), f32x4_add(za2, za3)))
    }

    #[cfg(target_feature = "simd128")]
    #[inline(always)]
    fn conv6_taps_2groups(
        bias: WasmVec4,
        w_tiles: &[[[f32; 4]; 4]; 12],
        h_taps: &[&[[f32; 4]; 3]; 6],
    ) -> WasmVec4 {
        let mut za0 = bias.0;
        let mut za1 = f32x4_splat(0.0);
        let mut za2 = f32x4_splat(0.0);
        let mut za3 = f32x4_splat(0.0);

        for tap in 0..6 {
            let ht = h_taps[tap];
            let vx0 = WasmVec4::load(&ht[0]).0;
            let vx1 = WasmVec4::load(&ht[1]).0;
            let wt0 = &w_tiles[tap * 2];
            let wt1 = &w_tiles[tap * 2 + 1];

            za0 = f32x4_add(
                za0,
                f32x4_mul(
                    WasmVec4::load(&wt0[0]).0,
                    i32x4_shuffle::<0, 0, 0, 0>(vx0, vx0),
                ),
            );
            za1 = f32x4_add(
                za1,
                f32x4_mul(
                    WasmVec4::load(&wt0[1]).0,
                    i32x4_shuffle::<1, 1, 1, 1>(vx0, vx0),
                ),
            );
            za2 = f32x4_add(
                za2,
                f32x4_mul(
                    WasmVec4::load(&wt0[2]).0,
                    i32x4_shuffle::<2, 2, 2, 2>(vx0, vx0),
                ),
            );
            za3 = f32x4_add(
                za3,
                f32x4_mul(
                    WasmVec4::load(&wt0[3]).0,
                    i32x4_shuffle::<3, 3, 3, 3>(vx0, vx0),
                ),
            );

            za0 = f32x4_add(
                za0,
                f32x4_mul(
                    WasmVec4::load(&wt1[0]).0,
                    i32x4_shuffle::<0, 0, 0, 0>(vx1, vx1),
                ),
            );
            za1 = f32x4_add(
                za1,
                f32x4_mul(
                    WasmVec4::load(&wt1[1]).0,
                    i32x4_shuffle::<1, 1, 1, 1>(vx1, vx1),
                ),
            );
            za2 = f32x4_add(
                za2,
                f32x4_mul(
                    WasmVec4::load(&wt1[2]).0,
                    i32x4_shuffle::<2, 2, 2, 2>(vx1, vx1),
                ),
            );
            za3 = f32x4_add(
                za3,
                f32x4_mul(
                    WasmVec4::load(&wt1[3]).0,
                    i32x4_shuffle::<3, 3, 3, 3>(vx1, vx1),
                ),
            );
        }

        WasmVec4(f32x4_add(f32x4_add(za0, za1), f32x4_add(za2, za3)))
    }

    #[cfg(target_feature = "simd128")]
    #[inline(always)]
    fn conv6_taps_3groups(
        bias: WasmVec4,
        w_tiles: &[[[f32; 4]; 4]; 18],
        h_taps: &[&[[f32; 4]; 3]; 6],
    ) -> WasmVec4 {
        let mut za0 = bias.0;
        let mut za1 = f32x4_splat(0.0);
        let mut za2 = f32x4_splat(0.0);
        let mut za3 = f32x4_splat(0.0);

        for tap in 0..6 {
            let ht = h_taps[tap];
            let vx0 = WasmVec4::load(&ht[0]).0;
            let vx1 = WasmVec4::load(&ht[1]).0;
            let vx2 = WasmVec4::load(&ht[2]).0;
            let base = tap * 3;
            let wt0 = &w_tiles[base];
            let wt1 = &w_tiles[base + 1];
            let wt2 = &w_tiles[base + 2];

            za0 = f32x4_add(
                za0,
                f32x4_mul(
                    WasmVec4::load(&wt0[0]).0,
                    i32x4_shuffle::<0, 0, 0, 0>(vx0, vx0),
                ),
            );
            za1 = f32x4_add(
                za1,
                f32x4_mul(
                    WasmVec4::load(&wt0[1]).0,
                    i32x4_shuffle::<1, 1, 1, 1>(vx0, vx0),
                ),
            );
            za2 = f32x4_add(
                za2,
                f32x4_mul(
                    WasmVec4::load(&wt0[2]).0,
                    i32x4_shuffle::<2, 2, 2, 2>(vx0, vx0),
                ),
            );
            za3 = f32x4_add(
                za3,
                f32x4_mul(
                    WasmVec4::load(&wt0[3]).0,
                    i32x4_shuffle::<3, 3, 3, 3>(vx0, vx0),
                ),
            );

            za0 = f32x4_add(
                za0,
                f32x4_mul(
                    WasmVec4::load(&wt1[0]).0,
                    i32x4_shuffle::<0, 0, 0, 0>(vx1, vx1),
                ),
            );
            za1 = f32x4_add(
                za1,
                f32x4_mul(
                    WasmVec4::load(&wt1[1]).0,
                    i32x4_shuffle::<1, 1, 1, 1>(vx1, vx1),
                ),
            );
            za2 = f32x4_add(
                za2,
                f32x4_mul(
                    WasmVec4::load(&wt1[2]).0,
                    i32x4_shuffle::<2, 2, 2, 2>(vx1, vx1),
                ),
            );
            za3 = f32x4_add(
                za3,
                f32x4_mul(
                    WasmVec4::load(&wt1[3]).0,
                    i32x4_shuffle::<3, 3, 3, 3>(vx1, vx1),
                ),
            );

            za0 = f32x4_add(
                za0,
                f32x4_mul(
                    WasmVec4::load(&wt2[0]).0,
                    i32x4_shuffle::<0, 0, 0, 0>(vx2, vx2),
                ),
            );
            za1 = f32x4_add(
                za1,
                f32x4_mul(
                    WasmVec4::load(&wt2[1]).0,
                    i32x4_shuffle::<1, 1, 1, 1>(vx2, vx2),
                ),
            );
            za2 = f32x4_add(
                za2,
                f32x4_mul(
                    WasmVec4::load(&wt2[2]).0,
                    i32x4_shuffle::<2, 2, 2, 2>(vx2, vx2),
                ),
            );
            za3 = f32x4_add(
                za3,
                f32x4_mul(
                    WasmVec4::load(&wt2[3]).0,
                    i32x4_shuffle::<3, 3, 3, 3>(vx2, vx2),
                ),
            );
        }

        WasmVec4(f32x4_add(f32x4_add(za0, za1), f32x4_add(za2, za3)))
    }
}
