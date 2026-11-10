//! Portable 16-byte aligned scalar/auto-vectorized math backend (`ScalarBackend`).
//!
//! Serves both as the universal fallback on any CPU architecture and as the deterministic
//! reference oracle in cross-backend parity tests (`tests/backend_parity.rs`).

use super::{BackendKind, MathBackend, SimdVec4};
use crate::math::crc32c::{Crc32cBackend, crc32c_software_slice};

/// 16-byte SIMD-aligned 4-lane scalar vector (`#[repr(C, align(16))]`).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ScalarVec4(pub [f32; 4]);

impl SimdVec4 for ScalarVec4 {
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
        Self([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
            self.0[3] + rhs.0[3],
        ])
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
            self.0[3] - rhs.0[3],
        ])
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self([
            self.0[0] * rhs.0[0],
            self.0[1] * rhs.0[1],
            self.0[2] * rhs.0[2],
            self.0[3] * rhs.0[3],
        ])
    }

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        Self([
            self.0[0] + a.0[0] * b.0[0],
            self.0[1] + a.0[1] * b.0[1],
            self.0[2] + a.0[2] * b.0[2],
            self.0[3] + a.0[3] * b.0[3],
        ])
    }

    #[inline(always)]
    fn fma_lane<const LANE: usize>(self, a: Self, v: Self) -> Self {
        const { assert!(LANE < 4) };
        let s = v.0[LANE];
        Self([
            self.0[0] + a.0[0] * s,
            self.0[1] + a.0[1] * s,
            self.0[2] + a.0[2] * s,
            self.0[3] + a.0[3] * s,
        ])
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        Self([
            self.0[0] - a.0[0] * b.0[0],
            self.0[1] - a.0[1] * b.0[1],
            self.0[2] - a.0[2] * b.0[2],
            self.0[3] - a.0[3] * b.0[3],
        ])
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        Self([
            self.0[0].max(rhs.0[0]),
            self.0[1].max(rhs.0[1]),
            self.0[2].max(rhs.0[2]),
            self.0[3].max(rhs.0[3]),
        ])
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        Self([
            self.0[0].min(rhs.0[0]),
            self.0[1].min(rhs.0[1]),
            self.0[2].min(rhs.0[2]),
            self.0[3].min(rhs.0[3]),
        ])
    }

    #[inline(always)]
    fn abs(self) -> Self {
        Self([
            self.0[0].abs(),
            self.0[1].abs(),
            self.0[2].abs(),
            self.0[3].abs(),
        ])
    }

    #[inline(always)]
    fn horizontal_sum(self) -> f32 {
        (self.0[0] + self.0[1]) + (self.0[2] + self.0[3])
    }
}

/// Portable scalar reference backend implementing [`MathBackend`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScalarBackend;

impl Crc32cBackend for ScalarBackend {
    #[inline]
    fn crc32c_update(crc_init: u32, data: &[u8]) -> u32 {
        crc32c_software_slice(crc_init, data)
    }
}

impl MathBackend for ScalarBackend {
    type Vec4 = ScalarVec4;
    const NAME: &'static str = "Scalar (16B Aligned Reference)";
    const KIND: BackendKind = BackendKind::Scalar;
}
