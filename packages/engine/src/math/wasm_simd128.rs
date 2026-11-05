//! WebAssembly 128-bit SIMD (`v128`) hardware math backend (`WasmSimd128Backend`).

use super::{BackendKind, MathBackend, SimdVec4};

#[cfg(target_feature = "simd128")]
use core::arch::wasm32::{
    f32x4_abs, f32x4_add, f32x4_extract_lane, f32x4_mul, f32x4_pmax, f32x4_pmin, f32x4_splat,
    f32x4_sub, i32x4_add, i32x4_all_true, i32x4_eq, i32x4_shuffle, i32x4_splat, i32x4_sub, u32x4,
    u32x4_ge, u32x4_lt, u32x4_shl, u32x4_shr, u32x4_splat, v128, v128_and, v128_andnot,
    v128_any_true, v128_or,
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
    fn flush_subnormal(self) -> Self {
        let exponent = v128_and(self.0, i32x4_splat(0x7f80_0000));
        let zero_exponent = i32x4_eq(exponent, i32x4_splat(0));
        let mantissa_to_clear = v128_and(zero_exponent, i32x4_splat(0x007f_ffff));
        Self(v128_andnot(self.0, mantissa_to_clear))
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
        h_taps: &[&[[f32; 4]; 4]; 6],
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
        h_taps: &[&[[f32; 4]; 4]; 6],
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
        h_taps: &[&[[f32; 4]; 4]; 6],
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

    #[cfg(target_feature = "simd128")]
    #[inline(always)]
    fn conv6_taps_4groups(
        bias: WasmVec4,
        w_tiles: &[[[f32; 4]; 4]; 24],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> WasmVec4 {
        let mut za0 = bias.0;
        let mut za1 = f32x4_splat(0.0);
        let mut za2 = f32x4_splat(0.0);
        let mut za3 = f32x4_splat(0.0);

        for tap in 0..6 {
            let ht = h_taps[tap];
            let base = tap * 4;
            for g in 0..4 {
                let vxg = WasmVec4::load(&ht[g]).0;
                let wtg = &w_tiles[base + g];
                za0 = f32x4_add(
                    za0,
                    f32x4_mul(
                        WasmVec4::load(&wtg[0]).0,
                        i32x4_shuffle::<0, 0, 0, 0>(vxg, vxg),
                    ),
                );
                za1 = f32x4_add(
                    za1,
                    f32x4_mul(
                        WasmVec4::load(&wtg[1]).0,
                        i32x4_shuffle::<1, 1, 1, 1>(vxg, vxg),
                    ),
                );
                za2 = f32x4_add(
                    za2,
                    f32x4_mul(
                        WasmVec4::load(&wtg[2]).0,
                        i32x4_shuffle::<2, 2, 2, 2>(vxg, vxg),
                    ),
                );
                za3 = f32x4_add(
                    za3,
                    f32x4_mul(
                        WasmVec4::load(&wtg[3]).0,
                        i32x4_shuffle::<3, 3, 3, 3>(vxg, vxg),
                    ),
                );
            }
        }

        WasmVec4(f32x4_add(f32x4_add(za0, za1), f32x4_add(za2, za3)))
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
}
