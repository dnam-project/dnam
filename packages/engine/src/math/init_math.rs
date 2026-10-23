//! Pure `#![no_std]` IEEE-754 cold-initialization math helpers (`ln_f64`, `exp_f64`, `powf_f64`,
//! `sqrt_f64`, `floor_f32`, `round_f32`, `cos_pi_unit_f64`).
//!
//! Used exclusively during cold setup, ZOH `set_sample_rate`, and UI skew calculation so zero
//! hardware `FSQRT` or `libm` calls appear in the crate binary.

use super::kernels::algebraic_unit_rotor_f64;

/// Pure `#![no_std]` floor for `f32`.
#[inline(always)]
pub fn floor_f32(x: f32) -> f32 {
    let i = x as i32;
    let f = i as f32;
    if x < f { f - 1.0 } else { f }
}

/// Pure `#![no_std]` floor for `f64`.
#[inline(always)]
pub fn floor_f64(x: f64) -> f64 {
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

/// Pure `#![no_std]` IEEE-754 + 6-step Newton-Raphson square root in `f64` (used only at cold init / ZOH).
#[inline]
pub fn sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let bits = x.to_bits();
    let approx_bits = (bits >> 1) + (0x3FFu64 << 51);
    let mut y = f64::from_bits(approx_bits);
    for _ in 0..6 {
        y = 0.5 * (y + x / y);
    }
    y
}

/// Pure `#![no_std]` natural logarithm $\ln(x)$ in `f64` via IEEE-754 exponent extraction +
/// Padé/atanh series (accurate to $< 10^{-15}$ across all positive normals; cold init only).
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
    let u = (mant - 1.0) / (mant + 1.0);
    let u2 = u * u;
    let mut term = u;
    let mut sum = u;
    let mut n = 3;
    while n <= 21 {
        term *= u2;
        sum += term / (n as f64);
        n += 2;
    }
    2.0 * sum + (k as f64) * core::f64::consts::LN_2
}

/// Pure `#![no_std]` exponential $\exp(x)$ in `f64` via range reduction $x = k \ln 2 + r$
/// ($|r| \le \frac{\ln 2}{2}$) and 14-term Taylor polynomial (accurate to $< 10^{-15}$; cold init only).
#[inline]
pub fn exp_f64(x: f64) -> f64 {
    if x < -708.0 {
        return 0.0;
    }
    if x > 709.0 {
        return f64::INFINITY;
    }
    let k = round_f64(x / core::f64::consts::LN_2) as i64;
    let r = x - (k as f64) * core::f64::consts::LN_2;
    let mut term = 1.0;
    let mut sum = 1.0;
    for i in 1..=14 {
        term *= r / (i as f64);
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
        assert_eq!(floor_f64(2.7), 2.0);
        assert_eq!(floor_f64(-2.7), -3.0);
        assert_eq!(round_f32(2.6), 3.0);
        assert_eq!(round_f32(-2.6), -3.0);
        assert_eq!(round_f64(2.6), 3.0);
        assert_eq!(round_f64(-2.6), -3.0);
        assert_eq!(sqrt_f64(0.0), 0.0);
        assert_eq!(sqrt_f64(-5.0), 0.0);
        assert_eq!(ln_f64(0.0), f64::NEG_INFINITY);
        assert_eq!(exp_f64(-800.0), 0.0);
        assert_eq!(exp_f64(800.0), f64::INFINITY);
        assert_eq!(powf_f64(0.0, 2.0), 0.0);
        assert_eq!(powf_f64(3.5, 1.0), 3.5);
        assert!((cos_pi_unit_f64(0.5)).abs() < 1e-5);
    }
}
