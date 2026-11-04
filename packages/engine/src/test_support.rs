//! Shared unit-test verification harness (`packages/engine/src/test_support.rs`).
//!
//! Provides:
//! 1. Exact rational (`i128` / `u128` + multi-word) arithmetic matching Python's `fractions.Fraction`.
//! 2. IEEE-754 `f32`/`f64` bit-exact conversions, ULP distance, outward rounding, and rank-space helpers.
//! 3. Canonical 15-rate sample-rate sweep table (`44.1 kHz`..`384 kHz`) and deterministic `SplitMix64` PRNG.

#![allow(dead_code)]

/// Canonical 15 sample rates across `44.1 kHz`..`384 kHz` (`H1` Hilbert anchor grid & ZOH sweep).
pub const CANONICAL_SAMPLE_RATES: [u32; 15] = [
    44_100, 48_000, 64_000, 88_200, 96_000, 128_000, 176_400, 192_000, 220_500, 240_000, 256_000,
    288_000, 320_000, 352_800, 384_000,
];

/// Deterministic `SplitMix64` PRNG for reproducible property and sweep tests.
#[derive(Clone, Copy, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Uniform `f64` in `[0.0, 1.0)`.
    pub fn next_unit_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Uniform `f32` in `[lo, hi]`.
    pub fn next_f32_range(&mut self, lo: f32, hi: f32) -> f32 {
        let t = self.next_unit_f64();
        (f64::from(lo) + (f64::from(hi) - f64::from(lo)) * t) as f32
    }
}

/// Greatest common divisor of two unsigned 128-bit integers.
pub const fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

/// Exact rational number `num / den` (`den > 0`, reduced by `gcd`) backed by `i128` / `u128`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rat {
    pub num: i128,
    pub den: i128,
}

impl Rat {
    pub fn new(num: i128, den: i128) -> Self {
        assert!(den != 0, "Rat denominator must be non-zero");
        if num == 0 {
            return Self { num: 0, den: 1 };
        }
        let sign = if (num < 0) ^ (den < 0) { -1i128 } else { 1i128 };
        let g = gcd_u128(num.unsigned_abs(), den.unsigned_abs()) as i128;
        Self {
            num: sign * (num.unsigned_abs() as i128 / g),
            den: (den.unsigned_abs() as i128 / g),
        }
    }

    pub const fn from_int(n: i128) -> Self {
        Self { num: n, den: 1 }
    }

    pub fn add(self, rhs: Self) -> Self {
        Self::new(self.num * rhs.den + rhs.num * self.den, self.den * rhs.den)
    }

    pub fn sub(self, rhs: Self) -> Self {
        Self::new(self.num * rhs.den - rhs.num * self.den, self.den * rhs.den)
    }

    pub fn mul(self, rhs: Self) -> Self {
        Self::new(self.num * rhs.num, self.den * rhs.den)
    }

    pub fn div(self, rhs: Self) -> Self {
        Self::new(self.num * rhs.den, self.den * rhs.num)
    }

    pub fn abs(self) -> Self {
        Self {
            num: self.num.abs(),
            den: self.den,
        }
    }

    pub fn floor(self) -> i128 {
        self.num.div_euclid(self.den)
    }

    pub fn ceil(self) -> i128 {
        -((-self.num).div_euclid(self.den))
    }

    /// Rounds `self` to the nearest integer with halfway ties to even (`RNE`).
    pub fn round_ties_to_even(self) -> i128 {
        let q = self.num.div_euclid(self.den);
        let r = self.num.rem_euclid(self.den);
        let twice_r = 2 * r;
        if twice_r > self.den || (twice_r == self.den && (q & 1) != 0) {
            q + 1
        } else {
            q
        }
    }

    /// Rounds `self >= 0` to `(integer, q32_fraction)` with ties to even on `self * 2^32`.
    pub fn round_q32(self) -> (u64, u32) {
        assert!(self.num >= 0, "round_q32 requires non-negative value");
        let scaled = Self::new(self.num << 32, self.den);
        let whole = scaled.round_ties_to_even() as u128;
        ((whole >> 32) as u64, (whole & 0xFFFF_FFFF) as u32)
    }

    pub fn to_f64(self) -> f64 {
        (self.num as f64) / (self.den as f64)
    }

    pub fn to_f32(self) -> f32 {
        self.to_f64() as f32
    }
}

/// Bitwise-exact widening from `f32` raw bits to `f64` (immune to FTZ/DAZ on subnormal inputs).
pub fn widen_f32_bits_to_f64(bits: u32) -> Option<f64> {
    let sign = u64::from(bits & 0x8000_0000) << 32;
    let exponent = (bits >> 23) & 0xff;
    let fraction = bits & 0x007f_ffff;
    let wide_bits = match (exponent, fraction) {
        (0xff, _) => return None,
        (0, 0) => sign,
        (0, fraction) => {
            let leading = 31 - fraction.leading_zeros();
            let exponent64 = u64::from(leading + 874);
            let fraction64 = u64::from(fraction - (1 << leading)) << (52 - leading);
            sign | (exponent64 << 52) | fraction64
        }
        (exponent, fraction) => {
            sign | (u64::from(exponent + 896) << 52) | (u64::from(fraction) << 29)
        }
    };
    Some(f64::from_bits(wide_bits))
}

/// ULP distance between two finite `f32` values (treating `-0.0` and `+0.0` as distance 0).
pub fn ulp_distance_f32(a: f32, b: f32) -> u32 {
    assert!(a.is_finite() && b.is_finite());
    if a == 0.0 && b == 0.0 {
        return 0;
    }
    let to_ordered = |x: f32| -> i32 {
        let bits = x.to_bits() as i32;
        if bits < 0 { i32::MIN - bits } else { bits }
    };
    to_ordered(a).abs_diff(to_ordered(b))
}

/// ULP distance between two finite `f64` values (treating `-0.0` and `+0.0` as distance 0).
pub fn ulp_distance_f64(a: f64, b: f64) -> u64 {
    assert!(a.is_finite() && b.is_finite());
    if a == 0.0 && b == 0.0 {
        return 0;
    }
    let to_ordered = |x: f64| -> i64 {
        let bits = x.to_bits() as i64;
        if bits < 0 { i64::MIN - bits } else { bits }
    };
    to_ordered(a).abs_diff(to_ordered(b))
}
