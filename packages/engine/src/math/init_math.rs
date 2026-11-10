//! Pure `#![no_std]` IEEE-754 cold-initialization math helpers (`ln_f64`, `exp_f64`, `powf_f64`,
//! `sqrt_f64`, `floor_f32`, `round_f32`, `cos_pi_unit_f64`).
//!
//! Used exclusively during cold setup, ZOH `set_sample_rate`, and UI skew calculation so zero
//! hardware `FSQRT` or `libm` calls appear in the crate binary.

use super::kernels::algebraic_unit_rotor_f64;

/// Pure `#![no_std]` floor for `f32`.
#[inline(always)]
pub fn floor_f32(x: f32) -> f32 {
    if x.is_nan() || x.abs() >= 8_388_608.0 {
        return x;
    }
    let i = x as i32;
    let f = i as f32;
    if x < f { f - 1.0 } else { f }
}

/// Pure `#![no_std]` floor for `f64`.
#[inline(always)]
pub fn floor_f64(x: f64) -> f64 {
    if x.is_nan() || x.abs() >= 4_503_599_627_370_496.0 {
        return x;
    }
    let i = x as i64;
    let f = i as f64;
    if x < f { f - 1.0 } else { f }
}

/// Pure `#![no_std]` round-half-away-from-zero for `f32`.
#[inline(always)]
pub fn round_f32(x: f32) -> f32 {
    if x >= 0.0 {
        floor_f32(x + 0.5)
    } else {
        -floor_f32(-x + 0.5)
    }
}

/// Pure `#![no_std]` round-half-away-from-zero for `f64`.
#[inline(always)]
pub fn round_f64(x: f64) -> f64 {
    if x >= 0.0 {
        floor_f64(x + 0.5)
    } else {
        -floor_f64(-x + 0.5)
    }
}

/// Pure `#![no_std]` IEEE-754 + division-free Newton-Goldschmidt square root in `f64` (`0 FDIV`, `0 FSQRT`).
#[inline]
pub fn sqrt_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    let (scaled_x, scale_out) = if x < f64::MIN_POSITIVE {
        (
            x * 18_014_398_509_481_984.0,
            f64::from_bits(0x3E40_0000_0000_0000),
        )
    } else {
        (x, 1.0)
    };
    let bits = scaled_x.to_bits();
    let rsqrt_bits = 0x5FE6_EB50_C7B5_37A9_u64.wrapping_sub(bits >> 1);
    let mut z = f64::from_bits(rsqrt_bits);
    let half_x = 0.5 * scaled_x;
    for _ in 0..6 {
        z *= 1.5 - half_x * z * z;
    }
    let y = scaled_x * z;
    (y + 0.5 * (scaled_x - y * y) * z) * scale_out
}

/// Pure `#![no_std]` natural logarithm $\ln(x)$ in `f64` via IEEE-754 exponent extraction +
/// Padé/atanh series (accurate to $< 10^{-15}$ across all positive normals; `0 FDIV`).
#[inline]
pub fn ln_f64(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7FF) as i64 - 1023;
    let mut mant = f64::from_bits((bits & 0x000F_FFFF_FFFF_FFFF) | (1023u64 << 52));
    let mut k = exp;
    if mant > core::f64::consts::SQRT_2 {
        mant *= 0.5;
        k += 1;
    }
    let u = (mant - 1.0) * super::newton6_reciprocal_f64(mant + 1.0);
    let u2 = u * u;
    const INV_ODD: [f64; 10] = [
        1.0 / 3.0,
        1.0 / 5.0,
        1.0 / 7.0,
        1.0 / 9.0,
        1.0 / 11.0,
        1.0 / 13.0,
        1.0 / 15.0,
        1.0 / 17.0,
        1.0 / 19.0,
        1.0 / 21.0,
    ];
    let mut term = u;
    let mut sum = u;
    for &inv_n in &INV_ODD {
        term *= u2;
        sum += term * inv_n;
    }
    2.0 * sum + (k as f64) * core::f64::consts::LN_2
}

/// Pure `#![no_std]` exponential $\exp(x)$ in `f64` via range reduction $x = k \ln 2 + r$
/// ($|r| \le \frac{\ln 2}{2}$) and 14-term Taylor polynomial (accurate to $< 10^{-15}$; `0 FDIV`).
#[inline]
pub fn exp_f64(x: f64) -> f64 {
    if x < -708.0 {
        return 0.0;
    }
    if x > 709.0 {
        return f64::INFINITY;
    }
    let k = round_f64(x * core::f64::consts::LOG2_E) as i64;
    let r = x - (k as f64) * core::f64::consts::LN_2;
    const INV_STEP: [f64; 14] = [
        1.0,
        1.0 / 2.0,
        1.0 / 3.0,
        1.0 / 4.0,
        1.0 / 5.0,
        1.0 / 6.0,
        1.0 / 7.0,
        1.0 / 8.0,
        1.0 / 9.0,
        1.0 / 10.0,
        1.0 / 11.0,
        1.0 / 12.0,
        1.0 / 13.0,
        1.0 / 14.0,
    ];
    let mut term = 1.0;
    let mut sum = 1.0;
    for &inv_i in &INV_STEP {
        term *= r * inv_i;
        sum += term;
    }
    let scale_bits = ((k + 1023) as u64) << 52;
    sum * f64::from_bits(scale_bits)
}

/// Pure `#![no_std]` power function $x^y = \exp(y \ln x)$ in `f64` (cold init / ZOH rescaling only).
#[inline]
pub fn powf_f64(base: f64, exp: f64) -> f64 {
    if base <= 0.0 {
        return 0.0;
    }
    if (exp - 1.0).abs() < 1.0e-15 {
        return base;
    }
    exp_f64(exp * ln_f64(base))
}

/// Pure `#![no_std]` cosine $\cos(\theta)$ for $\theta \in [0, \pi]$ via our algebraic rotor in `f64`.
#[inline]
pub fn cos_pi_unit_f64(w: f64) -> f64 {
    let theta = (w.clamp(0.0, 1.0)) * core::f64::consts::PI;
    algebraic_unit_rotor_f64(theta).0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_std_f64_init_math_accuracy_and_bounds() {
        for &x in &[0.25f64, 0.5, 0.995, 1.0, 2.0, 10.0, 640.0] {
            let s = sqrt_f64(x);
            assert!((s * s - x).abs() < 1.0e-12);
            let l = ln_f64(x);
            let e = exp_f64(l);
            assert!((e - x).abs() < 1.0e-12);
        }
        assert!((powf_f64(0.995, 48000.0 / 96000.0) - sqrt_f64(0.995)).abs() < 1.0e-14);

        assert_eq!(floor_f32(2.7), 2.0);
        assert_eq!(floor_f32(-2.7), -3.0);
        assert!(floor_f32(f32::NAN).is_nan());
        assert_eq!(floor_f32(10_000_000.0), 10_000_000.0);
        assert_eq!(floor_f64(2.7), 2.0);
        assert_eq!(floor_f64(-2.7), -3.0);
        assert!(floor_f64(f64::NAN).is_nan());
        assert_eq!(floor_f64(1.0e16), 1.0e16);
        assert_eq!(round_f32(2.6), 3.0);
        assert_eq!(round_f32(-2.6), -3.0);
        assert_eq!(round_f64(2.6), 3.0);
        assert_eq!(round_f64(-2.6), -3.0);
        assert_eq!(sqrt_f64(0.0), 0.0);
        assert_eq!(sqrt_f64(-5.0), 0.0);
        assert!(sqrt_f64(f64::NAN).is_nan());
        assert_eq!(sqrt_f64(f64::INFINITY), f64::INFINITY);
        let sub = 1.0e-312_f64;
        assert!((sqrt_f64(sub) - 1.0e-156).abs() < 1.0e-168);
        assert_eq!(ln_f64(0.0), f64::NEG_INFINITY);
        assert_eq!(exp_f64(-800.0), 0.0);
        assert_eq!(exp_f64(800.0), f64::INFINITY);
        assert_eq!(powf_f64(0.0, 2.0), 0.0);
        assert_eq!(powf_f64(3.5, 1.0), 3.5);
        assert!((cos_pi_unit_f64(0.5)).abs() < 1e-5);
    }
}
