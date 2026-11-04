//! Hardware abstraction traits (`SimdVec4` and `MathBackend`) for 4-lane SIMD registers
//! and domain-specific DSP and matrix kernels.
//!
//! Canonical Reference: `docs/dsp/op_n_commuted_cascade.md` and `docs/overview/tora_architecture.md`.

use super::{
    BackendKind, algebraic_unit_rotor, crc32c_slice, default_ops, f32_to_fp16, fp16_to_f32,
};

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

/// Level 1 Hardware Abstraction: 128-bit 4-lane `f32` SIMD register (`float32x4_t` / `__m128` / `v128`).
pub trait SimdVec4: Copy + Clone + Send + Sync + 'static {
    /// Broadcasts `val` across all 4 lanes.
    fn splat(val: f32) -> Self;
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
    /// Flushes subnormal lanes to signed zero without altering other bit patterns.
    #[inline(always)]
    fn flush_subnormal(self) -> Self {
        Self::from_array(self.to_array().map(flush_subnormal_f32_bits))
    }
    /// Horizontal sum across all 4 lanes (`lane0 + lane1 + lane2 + lane3`).
    fn horizontal_sum(self) -> f32;
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

/// Level 2 Hardware Abstraction: Domain-specific DSP, tile-matrix, and container kernels.
pub trait MathBackend: Copy + Clone + Send + Sync + 'static {
    /// Underlying 128-bit 4-lane vector register type.
    type Vec4: SimdVec4;
    /// Human-readable backend name for telemetry and benchmarks.
    const NAME: &'static str;
    /// Enum identifier for this backend.
    const KIND: BackendKind;

    /// Evaluates the exact division-free algebraic `PolyADAA` kernel across 4 channels (`ref[dsp.op-n.polyadaa.midpoint]`).
    #[inline(always)]
    fn poly_adaa_algebraic_4ch(
        u_prev: Self::Vec4,
        u_curr: Self::Vec4,
        beta: Self::Vec4,
        delta: Self::Vec4,
    ) -> Self::Vec4 {
        default_ops::poly_adaa_algebraic_4ch::<Self>(u_prev, u_curr, beta, delta)
    }

    /// 4x4 matrix-vector FMA tile `acc + sum_{c=0..3} x[c] * mat[c]`.
    #[inline(always)]
    fn fma_mat4x4_vec4(acc: Self::Vec4, mat: &[[f32; 4]; 4], x: &[f32; 4]) -> Self::Vec4 {
        default_ops::fma_mat4x4_vec4::<Self>(acc, mat, x)
    }

    /// 6-tap dilated convolution over 1 input group (`6 × 4×4` tiles).
    #[inline(always)]
    fn conv6_taps_1group(
        bias: Self::Vec4,
        w_tiles: &[[[f32; 4]; 4]; 6],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> Self::Vec4 {
        default_ops::conv6_taps_1group::<Self>(bias, w_tiles, h_taps)
    }

    /// 6-tap dilated convolution over 2 input groups (`12 × 4×4` tiles).
    #[inline(always)]
    fn conv6_taps_2groups(
        bias: Self::Vec4,
        w_tiles: &[[[f32; 4]; 4]; 12],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> Self::Vec4 {
        default_ops::conv6_taps_2groups::<Self>(bias, w_tiles, h_taps)
    }

    /// 6-tap dilated convolution over 3 input groups (`18 × 4×4` tiles).
    #[inline(always)]
    fn conv6_taps_3groups(
        bias: Self::Vec4,
        w_tiles: &[[[f32; 4]; 4]; 18],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> Self::Vec4 {
        default_ops::conv6_taps_3groups::<Self>(bias, w_tiles, h_taps)
    }

    /// 6-tap dilated convolution over 4 input groups (`24 × 4×4` tiles).
    #[inline(always)]
    fn conv6_taps_4groups(
        bias: Self::Vec4,
        w_tiles: &[[[f32; 4]; 4]; 24],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> Self::Vec4 {
        default_ops::conv6_taps_4groups::<Self>(bias, w_tiles, h_taps)
    }

    /// Evaluates the 4-tap Cubic Lagrange basis weights across 4 lanes (`ref[dsp.op-t.reader.catmull-rom]`).
    #[inline(always)]
    fn cubic_lagrange_weights_4ch(
        vmu: Self::Vec4,
    ) -> (Self::Vec4, Self::Vec4, Self::Vec4, Self::Vec4) {
        default_ops::cubic_lagrange_weights_4ch::<Self>(vmu)
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

    /// Computes Castagnoli CRC32C over `data`.
    #[inline]
    fn crc32c(data: &[u8]) -> u32 {
        crc32c_slice(0, data)
    }

    /// Evaluates the division-free algebraic unit-circle phasor (cos(theta), sin(theta)) (`ref[dsp.op-m.rotor.contractive]`).
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
        let _ = ScalarBackend::poly_adaa_algebraic_4ch(v, v, v, v);
        let mat = [[0.1f32; 4]; 4];
        let _ = ScalarBackend::fma_mat4x4_vec4(v, &mat, &[0.5; 4]);

        let h_tap = [[0.2f32; 4]; 4];
        let w1 = [mat; 6];
        let h1 = [&h_tap; 6];
        let _ = ScalarBackend::conv6_taps_1group(v, &w1, &h1);

        let w2 = [mat; 12];
        let _ = ScalarBackend::conv6_taps_2groups(v, &w2, &h1);

        let w3 = [mat; 18];
        let _ = ScalarBackend::conv6_taps_3groups(v, &w3, &h1);

        let w4 = [mat; 24];
        let _ = ScalarBackend::conv6_taps_4groups(v, &w4, &h1);

        let _ = ScalarBackend::cubic_lagrange_weights_4ch(v);
    }
}
