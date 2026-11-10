//! Hardware abstraction traits (`SimdVec4` and `MathBackend`) for 4-lane SIMD registers
//! and core division-free math primitives.
//!
//! Higher-level domain modules define their own `<Domain>Backend` traits in their respective
//! modules (`OpMBackend`, `OpNBackend`, `OpTBackend`, `Stage2Backend`, `Stage3Backend`,
//! `FeatureBusBackend`), keeping the dependency DAG strictly one-way (`engine -> stages -> operators -> math`).

use super::{BackendKind, algebraic_unit_rotor, f32_to_fp16, fp16_to_f32};
use crate::math::crc32c::Crc32cBackend;

/// Flushes `f32` subnormals to signed zero without changing any other bit pattern.
#[inline(always)]
pub const fn flush_subnormal_f32_bits(value: f32) -> f32 {
    let bits = value.to_bits();
    let mask = if bits & 0x7f80_0000 == 0 {
        0x8000_0000
    } else {
        u32::MAX
    };
    f32::from_bits(bits & mask)
}

/// Applies the portable fallback on targets without a native block-level denormal guard.
#[inline(always)]
pub(crate) const fn flush_recursive_state_f32(value: f32) -> f32 {
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    {
        value
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        flush_subnormal_f32_bits(value)
    }
}

#[inline(always)]
pub(crate) fn flush_recursive_state_vec4<V: SimdVec4>(value: V) -> V {
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    {
        value
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        value.flush_subnormal()
    }
}

/// Portable scalar 2-step Newton `rsqrt` (`0x5f37_59df - (v >> 1)`) for a single positive `f32`.
#[inline(always)]
pub fn rsqrt_nr2_scalar_f32(value: f32) -> f32 {
    let mut est = f32::from_bits(0x5f37_59df - (value.to_bits() >> 1));
    let half_v = 0.5 * value;
    for _ in 0..2 {
        let corr = 1.5 - (half_v * est) * est;
        est *= corr;
    }
    est
}

/// Portable scalar 2-step Newton `rsqrt` (`0x5f37_59df - (v >> 1)`) for a stereo pair `(v0, v1)`.
#[inline(always)]
pub fn rsqrt_nr2_scalar_pair_f32(v0: f32, v1: f32) -> (f32, f32) {
    let mut e0 = f32::from_bits(0x5f37_59df - (v0.to_bits() >> 1));
    let mut e1 = f32::from_bits(0x5f37_59df - (v1.to_bits() >> 1));
    let hv0 = 0.5 * v0;
    let hv1 = 0.5 * v1;
    for _ in 0..2 {
        let c0 = 1.5 - (hv0 * e0) * e0;
        let c1 = 1.5 - (hv1 * e1) * e1;
        e0 *= c0;
        e1 *= c1;
    }
    (e0, e1)
}

/// Level 1 Hardware Abstraction: 128-bit 4-lane `f32` SIMD register (`float32x4_t` / `__m128` / `v128`).
pub trait SimdVec4: Copy + Clone + Send + Sync + 'static {
    /// Broadcasts `val` across all 4 lanes.
    fn splat(val: f32) -> Self;
    /// Broadcasts the stereo pair `(a, b)` into `[a, b, a, b]`.
    #[inline(always)]
    fn splat_pair(a: f32, b: f32) -> Self {
        Self::from_array([a, b, a, b])
    }
    /// Loads 4 `f32` lanes from a slice.
    fn load(src: &[f32; 4]) -> Self;
    /// Stores 4 `f32` lanes into a slice.
    fn store(self, dst: &mut [f32; 4]);
    /// Lane-wise addition (`self + rhs`).
    fn add(self, rhs: Self) -> Self;
    /// Lane-wise subtraction (`self - rhs`).
    fn sub(self, rhs: Self) -> Self;
    /// Lane-wise multiplication (`self * rhs`).
    fn mul(self, rhs: Self) -> Self;
    /// Fused multiply-add (`self + a * b`).
    fn fma(self, a: Self, b: Self) -> Self;
    /// Fused multiply-add with indexed scalar lane broadcast (`self + a * v[LANE]`, `LANE < 4`).
    #[inline(always)]
    fn fma_lane<const LANE: usize>(self, a: Self, v: Self) -> Self {
        const { assert!(LANE < 4) };
        self.fma(a, Self::splat(v.to_array()[LANE]))
    }
    /// Fused multiply-subtract (`self - a * b`).
    fn fms(self, a: Self, b: Self) -> Self;
    /// Lane-wise maximum (`max(self, rhs)`).
    fn max(self, rhs: Self) -> Self;
    /// Lane-wise minimum (`min(self, rhs)`).
    fn min(self, rhs: Self) -> Self;
    /// Branchless lane-wise clamp to `[lo, hi]`.
    #[inline(always)]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }
    /// Lane-wise absolute value (`|self|`).
    fn abs(self) -> Self;
    /// Swaps adjacent stereo pairs and negates the even lanes (`[-v1, v0, -v3, v2]`).
    #[inline(always)]
    fn pair_swap_neg(self) -> Self {
        let a = self.to_array();
        Self::from_array([-a[1], a[0], -a[3], a[2]])
    }
    /// Flushes subnormal lanes to signed zero without altering other bit patterns.
    #[inline(always)]
    fn flush_subnormal(self) -> Self {
        Self::from_array(self.to_array().map(flush_subnormal_f32_bits))
    }
    /// Horizontal sum across all 4 lanes (`lane0 + lane1 + lane2 + lane3`).
    fn horizontal_sum(self) -> f32;
    /// Simultaneously reduces two 4-lane vectors into `(self.horizontal_sum(), rhs.horizontal_sum())`.
    #[inline(always)]
    fn horizontal_sum_pair(self, rhs: Self) -> (f32, f32) {
        (self.horizontal_sum(), rhs.horizontal_sum())
    }
    /// 4-lane dot product (`sum(self * rhs)`).
    #[inline(always)]
    fn dot(self, rhs: Self) -> f32 {
        self.mul(rhs).horizontal_sum()
    }
    /// Returns a 4-lane vector of zeros (`0.0`).
    #[inline(always)]
    fn zero() -> Self {
        Self::splat(0.0)
    }
    /// Constructs a 4-lane vector from a `[f32; 4]` value.
    #[inline(always)]
    fn from_array(arr: [f32; 4]) -> Self {
        Self::load(&arr)
    }
    /// Converts this 4-lane vector into a `[f32; 4]` array.
    #[inline(always)]
    fn to_array(self) -> [f32; 4] {
        let mut dst = [0.0; 4];
        self.store(&mut dst);
        dst
    }
}

/// Level 2 Hardware Abstraction: Core SIMD math primitives across every architecture
/// (`aarch64`, `x86_64`, `wasm32`, `scalar`).
pub trait MathBackend: Copy + Clone + Send + Sync + 'static + Crc32cBackend {
    /// Underlying 128-bit 4-lane vector register type.
    type Vec4: SimdVec4;
    /// Human-readable backend name for telemetry and benchmarks.
    const NAME: &'static str;
    /// Enum identifier for this backend.
    const KIND: BackendKind;

    /// 4x4 matrix-vector FMA tile `acc + sum_{c=0..3} x[c] * mat[c]`.
    #[inline(always)]
    fn fma_mat4x4_vec4(acc: Self::Vec4, mat: &[[f32; 4]; 4], x: &[f32; 4]) -> Self::Vec4 {
        let xv = Self::Vec4::load(x);
        acc.fma_lane::<0>(Self::Vec4::load(&mat[0]), xv)
            .fma_lane::<1>(Self::Vec4::load(&mat[1]), xv)
            .fma_lane::<2>(Self::Vec4::load(&mat[2]), xv)
            .fma_lane::<3>(Self::Vec4::load(&mat[3]), xv)
    }

    /// Converts 4 IEEE-754 single-precision `f32` samples to `fp16` (`u16`).
    #[inline(always)]
    fn f32x4_to_fp16x4(src: &[f32; 4]) -> [u16; 4] {
        [
            f32_to_fp16(src[0]),
            f32_to_fp16(src[1]),
            f32_to_fp16(src[2]),
            f32_to_fp16(src[3]),
        ]
    }

    /// Converts 4 IEEE-754 half-precision `fp16` (`u16`) samples to `f32`.
    #[inline(always)]
    fn fp16x4_to_f32x4(src: &[u16; 4]) -> [f32; 4] {
        [
            fp16_to_f32(src[0]),
            fp16_to_f32(src[1]),
            fp16_to_f32(src[2]),
            fp16_to_f32(src[3]),
        ]
    }

    /// Evaluates 2-step Newton `rsqrt` (`0x5f37_59df - (v >> 1)`) for a single positive `f32` (`0 FSQRT`, `0 FDIV`).
    #[inline(always)]
    fn rsqrt_nr2_f32(v: f32) -> f32 {
        rsqrt_nr2_scalar_f32(v)
    }

    /// Evaluates 2-step Newton `rsqrt` (`0x5f37_59df - (v >> 1)`) for a stereo pair `(v0, v1)` (`0 FSQRT`, `0 FDIV`).
    #[inline(always)]
    fn rsqrt_nr2_pair_f32(v0: f32, v1: f32) -> (f32, f32) {
        rsqrt_nr2_scalar_pair_f32(v0, v1)
    }

    /// Evaluates the division-free algebraic unit-circle phasor `(cos(theta), sin(theta))` (`ref[dsp.op-m.rotor.contractive]`).
    #[inline(always)]
    fn fast_unit_phasor(theta_rad: f32) -> (f32, f32) {
        algebraic_unit_rotor(theta_rad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::scalar::{ScalarBackend, ScalarVec4};
    use proptest::prelude::*;

    fn check_flush_subnormal<V: SimdVec4>() {
        let cases = [
            (
                [0x0000_0000, 0x8000_0000, 0x0000_0001, 0x8000_0001],
                [0x0000_0000, 0x8000_0000, 0x0000_0000, 0x8000_0000],
            ),
            (
                [0x007f_ffff, 0x807f_ffff, 0x0080_0000, 0x8080_0000],
                [0x0000_0000, 0x8000_0000, 0x0080_0000, 0x8080_0000],
            ),
            (
                [0x7f80_0000, 0xff80_0000, 0x7f80_0001, 0xffc1_2345],
                [0x7f80_0000, 0xff80_0000, 0x7f80_0001, 0xffc1_2345],
            ),
        ];
        for (input, expected) in cases {
            let values = input.map(f32::from_bits);
            assert_eq!(
                values.map(flush_subnormal_f32_bits).map(f32::to_bits),
                expected
            );
            assert_eq!(
                V::from_array(values)
                    .flush_subnormal()
                    .to_array()
                    .map(f32::to_bits),
                expected
            );
        }
    }

    #[test]
    fn test_flush_subnormal_preserves_bits() {
        check_flush_subnormal::<ScalarVec4>();
        #[cfg(target_arch = "aarch64")]
        check_flush_subnormal::<crate::math::NeonVec4>();
        #[cfg(target_arch = "x86_64")]
        check_flush_subnormal::<crate::math::X86Vec4>();
        #[cfg(target_arch = "wasm32")]
        check_flush_subnormal::<crate::math::WasmVec4>();
    }

    proptest! {
        #[test]
        fn prop_flush_subnormal_preserves_non_subnormal_bits(bits in any::<u32>()) {
            let expected = if bits & 0x7f80_0000 == 0 {
                bits & 0x8000_0000
            } else {
                bits
            };
            prop_assert_eq!(flush_subnormal_f32_bits(f32::from_bits(bits)).to_bits(), expected);
            let vector = ScalarVec4::splat(f32::from_bits(bits)).flush_subnormal();
            prop_assert_eq!(vector.to_array().map(f32::to_bits), [expected; 4]);
        }
    }

    #[test]
    fn test_simd_vec4_defaults() {
        let z = ScalarVec4::zero();
        assert_eq!(z.to_array(), [0.0; 4]);

        let arr = [1.0, 2.0, 3.0, 4.0];
        let v = ScalarVec4::from_array(arr);
        assert_eq!(v.to_array(), arr);

        let d = v.dot(v);
        assert_eq!(d, 1.0 + 4.0 + 9.0 + 16.0);

        let lo = ScalarVec4::splat(1.5);
        let hi = ScalarVec4::splat(3.5);
        let clamped = v.clamp(lo, hi);
        assert_eq!(clamped.to_array(), [1.5, 2.0, 3.0, 3.5]);
    }

    #[test]
    fn test_math_backend_defaults() {
        let (c, s) = ScalarBackend::fast_unit_phasor(0.5);
        assert!(c.is_finite());
        assert!(s.is_finite());

        let crc = ScalarBackend::crc32c(b"123456789");
        assert_eq!(crc, 0xe3069283);

        let fp16 = ScalarBackend::f32x4_to_fp16x4(&[1.0, 2.0, 3.0, 4.0]);
        let back = ScalarBackend::fp16x4_to_f32x4(&fp16);
        assert!((back[0] - 1.0).abs() < 1e-3);

        let v = ScalarVec4::splat(0.5);
        let mat = [[0.1f32; 4]; 4];
        let _ = ScalarBackend::fma_mat4x4_vec4(v, &mat, &[0.5; 4]);
    }
}
