//! Normative bit-exact and finite-precision mathematical kernels (`src/math/spec_math.rs`).
//!
//! Implements the mathematical foundation rules across:
//! - `ref[arch.fp.range-saturation]`: `±100.0` operating envelope and `±1e9` / `fp16` saturation flags.
//! - `ref[dsp.op-m.rotor.contractive]`, `ref[dsp.op-m.rotor.zero-hold]`,
//!   `ref[dsp.op-m.rotor.quadrant-horner]`, `ref[dsp.op-m.rotor.ulp-guard]`:
//!   Quadrant-reduced degree-13/14 Horner contractive rotor with adaptive 1–2 ULP norm guard.
//! - `ref[dsp.op-m.rotor.q0-192-lfo]`: `Q0.192` fixed-point LFO phase accumulator (`[u64; 3]`).
//! - `ref[dsp.ctrl.math.pitch-profile]`: Degree-14 `f64` Horner pitch-ratio & time-warp profile.
//! - `ref[dsp.ctrl.math.log2-lut]`: 129-entry monotone `f32` `log2` lookup table (`516 B`).
//! - `ref[dsp.ctrl.op-n.trust-clip]` & `ref[dsp.op-n.ctrl.trust-clip]`: Bit-preserving `f32 <-> f64`,
//!   outward-rounded energy check, 6-step Newton fast path & 32-step bisection trust clip.
//! - `ref[dsp.ctrl.range.normalize-clamp]` & `ref[dsp.ctrl.range.f32-rank-bounds]`:
//!   `binary64` declared-range normalization and ordered admissible `f32` rank-space intersection.
//! - `ref[dsp.op-n.polyadaa.rail-crossing]`: 6-step division-free Newton reciprocal in `f64`.

/// Error returned when a non-finite or out-of-domain floating-point bit pattern is rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidFloatInput;

/// Event bitflags for `overflow_or_saturation_flags` (`ref[arch.fp.range-saturation]`).
pub mod saturation_flags {
    /// Bit 0: A processed audio input exceeded the `±1e9` safety ceiling.
    pub const INPUT_SATURATED: u32 = 1 << 0;
    /// Bit 1: A finite delay write exceeded `±65,504.0` (`0x7BFF` / `0xFBFF`).
    pub const FP16_SATURATED: u32 = 1 << 1;
    /// Bit 2: An intermediate or recursive-state safety guard clamped a value to `±1e9`.
    pub const INTERNAL_SATURATED: u32 = 1 << 2;
    /// Bit 3: The processed output guard clamped a value to `±1e9`.
    pub const OUTPUT_SATURATED: u32 = 1 << 3;
    /// Bit 4: An input exceeded the model's certified input envelope (`>= ±100.0`).
    pub const OUTSIDE_CERTIFIED_INPUT: u32 = 1 << 4;
    /// Bit 5: A non-finite external sample, control, or parameter was detected and rejected.
    pub const INVALID_NONFINITE: u32 = 1 << 5;
}

/// Hard safety ceiling `S = 1e9` (`ref[arch.fp.range-saturation]`).
pub const SAFETY_CEILING_F32: f32 = 1.0e9;

/// Minimum certified unclipped input envelope `±100.0` (`+40 dBFS`, `ref[arch.fp.range-saturation]`).
pub const CERTIFIED_INPUT_ENVELOPE_MIN_F32: f32 = 100.0;

/// Maximum finite `fp16` magnitude `65,504.0` (`0x7BFF`, `ref[arch.fp.range-saturation]`).
pub const FP16_MAX_FINITE_F32: f32 = 65_504.0;

/// Sanitizes and clamps an input sample against `certified_limit >= 100.0` and `±1e9` (`ref[arch.fp.range-saturation]`).
#[inline]
pub fn sanitize_input_sample(x: f32, certified_limit: f32, flags: &mut u32) -> f32 {
    if !x.is_finite() {
        *flags |= saturation_flags::INVALID_NONFINITE;
        return 0.0;
    }
    let cert = certified_limit.max(CERTIFIED_INPUT_ENVELOPE_MIN_F32);
    if x.abs() > cert {
        *flags |= saturation_flags::OUTSIDE_CERTIFIED_INPUT;
    }
    if x.abs() > SAFETY_CEILING_F32 {
        *flags |= saturation_flags::INPUT_SATURATED;
        return x.clamp(-SAFETY_CEILING_F32, SAFETY_CEILING_F32);
    }
    x
}

/// Sanitizes and clamps an internal state value against `±1e9` (`ref[arch.fp.range-saturation]`).
#[inline]
pub fn sanitize_internal_sample(x: f32, flags: &mut u32) -> f32 {
    if !x.is_finite() {
        *flags |= saturation_flags::INVALID_NONFINITE;
        return 0.0;
    }
    if x.abs() > SAFETY_CEILING_F32 {
        *flags |= saturation_flags::INTERNAL_SATURATED;
        return x.clamp(-SAFETY_CEILING_F32, SAFETY_CEILING_F32);
    }
    x
}

/// Sanitizes and clamps an output sample against `±1e9` (`ref[arch.fp.range-saturation]`).
#[inline]
pub fn sanitize_output_sample(x: f32, flags: &mut u32) -> f32 {
    if !x.is_finite() {
        *flags |= saturation_flags::INVALID_NONFINITE;
        return 0.0;
    }
    if x.abs() > SAFETY_CEILING_F32 {
        *flags |= saturation_flags::OUTPUT_SATURATED;
        return x.clamp(-SAFETY_CEILING_F32, SAFETY_CEILING_F32);
    }
    x
}

/// Converts a finite `f32` sample to `fp16` bits while recording `FP16_SATURATED` if `|x| > 65504.0`
/// (`ref[arch.fp.range-saturation]`, `ref[dsp.op-t.ring.fp16-saturation]`).
#[inline]
pub fn f32_to_fp16_with_telemetry(x: f32, flags: &mut u32) -> u16 {
    if !x.is_finite() {
        *flags |= saturation_flags::INVALID_NONFINITE;
        return 0;
    }
    if x.abs() > FP16_MAX_FINITE_F32 {
        *flags |= saturation_flags::FP16_SATURATED;
    }
    super::fp16::f32_to_fp16(x.clamp(-FP16_MAX_FINITE_F32, FP16_MAX_FINITE_F32))
}

// ============================================================================
// Bit-Preserving f32 <-> f64 Conversions (`ref[dsp.ctrl.op-n.trust-clip]`)
// ============================================================================

/// Widens an IEEE-754 `f32` bit pattern (including subnormals) directly to `f64`
/// without executing hardware `f32` instructions subject to FTZ/DAZ (`ref[dsp.ctrl.op-n.trust-clip]`).
/// Returns `None` if `bits` encodes `±Inf` or `NaN`.
#[inline(always)]
pub const fn widen_f32_bits(bits: u32) -> Option<f64> {
    let exponent = (bits >> 23) & 0xff;
    if exponent != 0 && exponent != 0xff {
        let sign = ((bits & 0x8000_0000) as u64) << 32;
        let frac = (bits & 0x007f_ffff) as u64;
        return Some(f64::from_bits(
            sign | (((exponent + 896) as u64) << 52) | (frac << 29),
        ));
    }
    let sign = ((bits & 0x8000_0000) as u64) << 32;
    let fraction = bits & 0x007f_ffff;
    let wide_bits = match (exponent, fraction) {
        (0xff, _) => return None,
        (0, 0) => sign,
        (_, frac) => {
            let leading = 31 - frac.leading_zeros();
            let exponent64 = (leading + 874) as u64;
            let fraction64 = ((frac - (1 << leading)) as u64) << (52 - leading);
            sign | (exponent64 << 52) | fraction64
        }
    };
    Some(f64::from_bits(wide_bits))
}

#[inline]
const fn round_right_even_u64(mantissa: u64, shift: u32) -> u64 {
    if shift > 53 {
        return 0;
    }
    let truncated = mantissa >> shift;
    let remainder = mantissa & ((1u64 << shift) - 1);
    let halfway = 1u64 << (shift - 1);
    let bump = if remainder > halfway || (remainder == halfway && (truncated & 1) != 0) {
        1
    } else {
        0
    };
    truncated + bump
}

/// Rounds a finite `f64` value to an IEEE-754 `f32` bit pattern using integer significand shifts
/// (`M >> 29` for normals, `M >> (-E - 97)` for subnormals, ties-to-even) without hardware `f32`
/// instructions (`ref[dsp.ctrl.op-n.trust-clip]`).
/// Returns `None` on `f64` subnormal, non-finite, or `f32` overflow.
#[inline]
pub const fn round_f64_to_f32_bits(value: f64) -> Option<u32> {
    let bits = value.to_bits();
    let sign = ((bits >> 32) as u32) & 0x8000_0000;
    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & 0x000f_ffff_ffff_ffff;
    if exponent_bits == 0 {
        return if fraction == 0 { Some(sign) } else { None };
    }
    if exponent_bits == 0x7ff {
        return None;
    }
    let mut exponent = exponent_bits - 1023;
    let mantissa = (1_u64 << 52) | fraction;
    if exponent < -126 {
        let shift = (-exponent - 97) as u32;
        return Some(sign | (round_right_even_u64(mantissa, shift) as u32));
    }
    if exponent > 127 {
        return None;
    }
    let mut rounded = round_right_even_u64(mantissa, 29);
    if rounded == (1 << 24) {
        exponent += 1;
        rounded >>= 1;
    }
    if exponent > 127 {
        return None;
    }
    Some(sign | (((exponent + 127) as u32) << 23) | ((rounded as u32) & 0x007f_ffff))
}

/// Returns the unbiased base-2 exponent of a non-zero finite `f32` bit pattern (`-149..=127`).
#[inline]
pub const fn f32_unbiased_exponent(bits: u32) -> Option<i32> {
    let magnitude = bits & 0x7fff_ffff;
    let exponent = magnitude >> 23;
    if magnitude == 0 || exponent == 0xff {
        return None;
    }
    if exponent != 0 {
        Some((exponent as i32) - 127)
    } else {
        let bit_len = 32 - (magnitude & 0x007f_ffff).leading_zeros();
        Some((bit_len as i32) - 150)
    }
}

// ============================================================================
// Outward-Checked Trust-Region Clip (`ref[dsp.ctrl.op-n.trust-clip]`)
// ============================================================================

/// Outward step (+1 ULP) on a strictly positive finite `f64` value (`ref[dsp.ctrl.op-n.trust-clip]`).
#[inline]
pub const fn next_up_pos_f64(value: f64) -> f64 {
    f64::from_bits(value.to_bits() + 1)
}

/// Outward-rounded product of two non-negative `f64` values (`ref[dsp.ctrl.op-n.trust-clip]`).
#[inline]
pub fn upper_product_f64(left: f64, right: f64) -> f64 {
    if left == 0.0 || right == 0.0 {
        0.0
    } else {
        next_up_pos_f64(left * right)
    }
}

/// Outward-rounded sum of squares $\mathrm{add\text{-}up}(\mathrm{initial}, \sum_i x_i^2)$ (`ref[dsp.ctrl.op-n.trust-clip]`).
#[inline]
pub fn upper_sum_squares_f64(values: &[f64], initial: f64) -> f64 {
    let mut total = initial;
    for &v in values {
        let sq = v * v;
        if sq != 0.0 {
            total = if total == 0.0 {
                sq
            } else {
                next_up_pos_f64(total + sq)
            };
        }
    }
    total
}

#[inline]
fn trust_clip_candidate<const D: usize>(
    x: &[f64; D],
    radius_squared: f64,
    energy_upper: f64,
    scale: f64,
) -> Option<(bool, [u32; D])> {
    let mut out_bits = [0u32; D];
    let mut rounded_f64 = [0.0f64; D];
    for i in 0..D {
        let b = round_f64_to_f32_bits(x[i] * scale)?;
        out_bits[i] = b;
        rounded_f64[i] = widen_f32_bits(b)?;
    }
    let scale_sq_upper = upper_product_f64(scale, scale);
    let ideal_upper = upper_product_f64(scale_sq_upper, energy_upper);
    let rounded_upper = upper_sum_squares_f64(&rounded_f64, 0.0);
    let accepted = ideal_upper <= radius_squared && rounded_upper <= radius_squared;
    Some((accepted, out_bits))
}

#[inline]
const fn inverse_sqrt_seed_f64(energy_upper: f64) -> f64 {
    let bits = energy_upper.to_bits();
    let exponent = (((bits >> 52) & 0x7ff) as i32) - 1023;
    let half_exp = (exponent + 2).div_euclid(2);
    let inv_bits = ((1023 - half_exp) as u64) << 52;
    f64::from_bits(inv_bits)
}

/// Evaluates the 6-step Newton fast path for outward-checked trust clipping (`ref[dsp.ctrl.op-n.trust-clip]`).
pub fn trust_clip_fast_candidate<const D: usize>(
    values_bits: &[u32; D],
    radius_bits: u32,
    epsilon_bits: u32,
) -> Option<(bool, [u32; D], f64)> {
    let mut x = [0.0f64; D];
    for i in 0..D {
        x[i] = widen_f32_bits(values_bits[i])?;
    }
    let radius = widen_f32_bits(radius_bits)?;
    let epsilon = widen_f32_bits(epsilon_bits)?;
    if radius <= 0.0 || epsilon <= 0.0 {
        return None;
    }
    let energy_upper = upper_sum_squares_f64(&x, epsilon);
    let mut inverse = inverse_sqrt_seed_f64(energy_upper);
    for _ in 0..6 {
        let sq = (energy_upper * inverse) * inverse;
        inverse *= 1.5 - 0.5 * sq;
    }
    let guard = 1.0 - f64::from_bits((1023 - 20) << 52);
    let scale = guard.min((radius * inverse) * guard);
    let (accepted, out) = trust_clip_candidate(&x, radius * radius, energy_upper, scale)?;
    Some((accepted, out, scale))
}

/// Evaluates the outward-checked trust-region clip on `D` (`4` or `8`) `f32` bit patterns
/// using the 6-step Newton fast path and 32-step bisection fallback (`ref[dsp.ctrl.op-n.trust-clip]`).
pub fn trust_clip_outward_checked_bits<const D: usize>(
    values_bits: &[u32; D],
    radius_bits: u32,
    epsilon_bits: u32,
) -> Option<([u32; D], f64, f64)> {
    if D != 4 && D != 8 {
        return None;
    }
    let mut x = [0.0f64; D];
    for i in 0..D {
        x[i] = widen_f32_bits(values_bits[i])?;
    }
    let radius = widen_f32_bits(radius_bits)?;
    let epsilon = widen_f32_bits(epsilon_bits)?;
    if radius <= 0.0 || epsilon <= 0.0 {
        return None;
    }
    let ideal_energy_upper = upper_sum_squares_f64(&x, epsilon);
    let radius_squared = radius * radius;

    let (accepted_one, orig_bits) =
        trust_clip_candidate(&x, radius_squared, ideal_energy_upper, 1.0)?;
    if accepted_one {
        return Some((orig_bits, 1.0, 1.0));
    }

    if let Some((true, fast_bits, fast_scale)) =
        trust_clip_fast_candidate(values_bits, radius_bits, epsilon_bits)
    {
        return Some((fast_bits, fast_scale, 1.0));
    }

    trust_clip_reference_bisection_bits(values_bits, radius_bits, epsilon_bits)
}

/// Evaluates the canonical 32-step reference bisection clip (`_clip_reference` in spec tests)
/// returning `(clipped_bits, accepted_scale, initial_high)`.
pub fn trust_clip_reference_bisection_bits<const D: usize>(
    values_bits: &[u32; D],
    radius_bits: u32,
    epsilon_bits: u32,
) -> Option<([u32; D], f64, f64)> {
    if D != 4 && D != 8 {
        return None;
    }
    let mut x = [0.0f64; D];
    for i in 0..D {
        x[i] = widen_f32_bits(values_bits[i])?;
    }
    let radius = widen_f32_bits(radius_bits)?;
    let epsilon = widen_f32_bits(epsilon_bits)?;
    if radius <= 0.0 || epsilon <= 0.0 {
        return None;
    }
    let ideal_energy_upper = upper_sum_squares_f64(&x, epsilon);
    let radius_squared = radius * radius;

    let (accepted_one, orig_bits) =
        trust_clip_candidate(&x, radius_squared, ideal_energy_upper, 1.0)?;
    if accepted_one {
        return Some((orig_bits, 1.0, 1.0));
    }

    let eps_exp_half = f32_unbiased_exponent(epsilon_bits)?.div_euclid(2);
    let mut norm_exp = eps_exp_half;
    for &b in values_bits {
        if let Some(e) = f32_unbiased_exponent(b)
            && e > norm_exp
        {
            norm_exp = e;
        }
    }
    let rad_exp = f32_unbiased_exponent(radius_bits)?;
    let high_exp = rad_exp + 1 - norm_exp;
    let mut high = if high_exp >= 0 {
        1.0
    } else {
        f64::from_bits(((high_exp + 1023) as u64) << 52)
    };
    let initial_high = high;
    let mut low = 0.0_f64;
    let mut result = [0u32; D];
    for i in 0..D {
        result[i] = values_bits[i] & 0x8000_0000;
    }
    for _ in 0..32 {
        let mid = (low + high) * 0.5;
        let (acc, cand) = trust_clip_candidate(&x, radius_squared, ideal_energy_upper, mid)?;
        if acc {
            low = mid;
            result = cand;
        } else {
            high = mid;
        }
    }
    Some((result, low, initial_high))
}

/// Applies outward-checked trust-region clipping in place on a 4-D `f32` group (`ref[dsp.ctrl.op-n.trust-clip]`).
#[inline]
pub fn trust_clip_outward_4d(z: &mut [f32; 4], rho_max: f32) {
    const DEFAULT_EPS_BITS: u32 = 0x0080_0000; // f32::MIN_POSITIVE
    let in_bits = [
        z[0].to_bits(),
        z[1].to_bits(),
        z[2].to_bits(),
        z[3].to_bits(),
    ];
    if let Some((out_bits, _, _)) =
        trust_clip_outward_checked_bits(&in_bits, rho_max.to_bits(), DEFAULT_EPS_BITS)
    {
        for i in 0..4 {
            z[i] = f32::from_bits(out_bits[i]);
        }
    }
}

/// Applies outward-checked trust-region clipping in place on an 8-D `f32` group (`ref[dsp.ctrl.op-n.trust-clip]`).
#[inline]
pub fn trust_clip_outward_8d(z: &mut [f32; 8], rho_max: f32) {
    const DEFAULT_EPS_BITS: u32 = 0x0080_0000; // f32::MIN_POSITIVE
    let in_bits: [u32; 8] = core::array::from_fn(|i| z[i].to_bits());
    if let Some((out_bits, _, _)) =
        trust_clip_outward_checked_bits(&in_bits, rho_max.to_bits(), DEFAULT_EPS_BITS)
    {
        for i in 0..8 {
            z[i] = f32::from_bits(out_bits[i]);
        }
    }
}

// ============================================================================
// Quadrant-Reduced Contractive Rotor & Q0.192 LFO (`ref[dsp.op-m.rotor.*]`)
// ============================================================================

/// `pi` in IEEE-754 `f64` (`0x400921FB54442D18`, `ref[dsp.op-m.rotor.quadrant-horner]`).
pub const PI_F64_BITS: u64 = 0x4009_21FB_5444_2D18;
/// `pi / 2` in IEEE-754 `f64` (`0x3FF921FB54442D18`, `ref[dsp.op-m.rotor.quadrant-horner]`).
pub const HALF_PI_F64_BITS: u64 = 0x3FF9_21FB_5444_2D18;
/// `pi / 4` in IEEE-754 `f64` (`0x3FE921FB54442D18`, `ref[dsp.op-m.rotor.quadrant-horner]`).
pub const QUARTER_PI_F64_BITS: u64 = 0x3FE9_21FB_5444_2D18;
/// Largest admissible `f32` angle magnitude strictly below `pi` (`0x40490FDA`).
pub const MAX_ROTOR_ANGLE_F32_BITS: u32 = 0x4049_0FDA;

/// Applies the adaptive 1–2 ULP norm guard to `(c_bits, s_bits)` (`ref[dsp.op-m.rotor.ulp-guard]`).
#[inline]
pub fn project_rotor_components_ulp_guard(
    mut c_bits: u32,
    mut s_bits: u32,
    active: bool,
) -> (u32, u32) {
    if !active {
        return (c_bits, s_bits);
    }
    let reduce_cosine = (c_bits & 0x7fff_ffff) >= (s_bits & 0x7fff_ffff);
    for _ in 0..2 {
        let c = widen_f32_bits(c_bits).unwrap_or(0.0);
        let s = widen_f32_bits(s_bits).unwrap_or(0.0);
        if c * c + s * s >= 1.0 {
            if reduce_cosine {
                c_bits -= 1;
            } else {
                s_bits -= 1;
            }
        }
    }
    (c_bits, s_bits)
}

/// Evaluates the quadrant-reduced degree-13/14 Horner contractive rotor on a signed `f32` bit
/// pattern `theta_bits` (`-pi < theta < pi`), returning `(cos_bits, sin_bits)` (`ref[dsp.op-m.rotor.contractive]`,
/// `ref[dsp.op-m.rotor.zero-hold]`, `ref[dsp.op-m.rotor.quadrant-horner]`, `ref[dsp.op-m.rotor.ulp-guard]`).
/// Returns `None` if `theta` is non-finite or `|theta| >= pi`.
#[inline]
pub fn contractive_rotor_f32_bits(theta_bits: u32) -> Option<(u32, u32)> {
    let mag_bits = theta_bits & 0x7fff_ffff;
    if mag_bits == 0 {
        return Some((1.0_f32.to_bits(), 0.0_f32.to_bits()));
    }
    let theta = widen_f32_bits(theta_bits)?;
    let magnitude = theta.abs();
    let pi64 = f64::from_bits(PI_F64_BITS);
    let half_pi64 = f64::from_bits(HALF_PI_F64_BITS);
    let quarter_pi64 = f64::from_bits(QUARTER_PI_F64_BITS);
    if !theta.is_finite() || magnitude >= pi64 {
        return None;
    }

    let reflected = magnitude > half_pi64;
    let quadrant = if reflected {
        pi64 - magnitude
    } else {
        magnitude
    };
    let swapped = quadrant > quarter_pi64;
    let reduced = if swapped {
        half_pi64 - quadrant
    } else {
        quadrant
    };
    let sq = reduced * reduced;

    let sine = reduced
        * (1.0
            + sq * (-1.0 / 6.0
                + sq * (1.0 / 120.0
                    + sq * (-1.0 / 5040.0
                        + sq * (1.0 / 362_880.0
                            + sq * (-1.0 / 39_916_800.0 + sq * (1.0 / 6_227_020_800.0)))))));
    let cosine = 1.0
        + sq * (-0.5
            + sq * (1.0 / 24.0
                + sq * (-1.0 / 720.0
                    + sq * (1.0 / 40_320.0
                        + sq * (-1.0 / 3_628_800.0
                            + sq * (1.0 / 479_001_600.0 - sq * (1.0 / 87_178_291_200.0)))))));

    let quad_cos = if swapped { sine } else { cosine };
    let quad_sin = if swapped { cosine } else { sine };
    let res_cos_f64 = if reflected { -quad_cos } else { quad_cos };
    let res_sin_f64 = if (theta_bits & 0x8000_0000) != 0 {
        -quad_sin
    } else {
        quad_sin
    };

    let c_bits = round_f64_to_f32_bits(res_cos_f64)?;
    let s_bits = round_f64_to_f32_bits(res_sin_f64)?;
    Some(project_rotor_components_ulp_guard(c_bits, s_bits, true))
}

/// Evaluates `contractive_rotor_f32_bits` on `theta: f32`, returning `Some((cos, sin))` for `|theta| < pi`.
#[inline]
pub fn contractive_rotor_f32(theta: f32) -> Option<(f32, f32)> {
    let (c_bits, s_bits) = contractive_rotor_f32_bits(theta.to_bits())?;
    Some((f32::from_bits(c_bits), f32::from_bits(s_bits)))
}

/// Steps an `f32` LFO state `(re, im)` with zero-increment hold (`ref[dsp.op-m.rotor.zero-hold]`):
/// when `theta == ±0.0`, `(re, im)` is returned unchanged bit-for-bit.
#[inline]
pub fn step_lfo_phasor_zero_hold(re: f32, im: f32, theta: f32) -> Option<(f32, f32)> {
    if (theta.to_bits() & 0x7fff_ffff) == 0 {
        return Some((re, im));
    }
    let (c, s) = contractive_rotor_f32(theta)?;
    let next_re = c * re - s * im;
    let next_im = s * re + c * im;
    let corr = 1.5 - 0.5 * (next_re * next_re + next_im * next_im);
    Some((next_re * corr, next_im * corr))
}

/// 256-bit constant $K = \lfloor 2^{256} / (2\pi) \rfloor$ in little-endian `u64` words (`ref[dsp.op-m.rotor.q0-192-lfo]`).
pub const INV_TAU_256_LE: [u64; 4] = [
    0x7f94_58ea_f7ae_f158,
    0x36d8_a566_4f10_e410,
    0x7f09_d5f4_7d4d_3770,
    0x28be_60db_9391_054a,
];

/// `Q0.192` fixed-point LFO phase accumulator (`3 × u64` little-endian words modulo $2^{192}$,
/// `ref[dsp.op-m.rotor.q0-192-lfo]`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Q0_192Phase {
    /// Little-endian limbs: `words[0]` = bits `0..64`, `words[1]` = bits `64..128`, `words[2]` = bits `128..192`.
    pub words: [u64; 3],
}

impl Q0_192Phase {
    pub const ZERO: Self = Self { words: [0; 3] };

    /// Advances `self` modulo $2^{192}$ by a signed increment `(sign_negative, mag_words)`.
    #[inline]
    pub fn advance_signed(&mut self, negative: bool, mag: [u64; 3]) {
        if mag == [0, 0, 0] {
            return;
        }
        if !negative {
            let (w0, c0) = self.words[0].overflowing_add(mag[0]);
            let (w1a, c1a) = self.words[1].overflowing_add(mag[1]);
            let (w1, c1b) = w1a.overflowing_add(u64::from(c0));
            let w2 = self.words[2]
                .wrapping_add(mag[2])
                .wrapping_add(u64::from(c1a || c1b));
            self.words = [w0, w1, w2];
        } else {
            let (w0, b0) = self.words[0].overflowing_sub(mag[0]);
            let (w1a, b1a) = self.words[1].overflowing_sub(mag[1]);
            let (w1, b1b) = w1a.overflowing_sub(u64::from(b0));
            let w2 = self.words[2]
                .wrapping_sub(mag[2])
                .wrapping_sub(u64::from(b1a || b1b));
            self.words = [w0, w1, w2];
        }
    }

    /// Advances `self` by the increment corresponding to `theta_bits` (`f32` bit pattern in `(-pi, pi)`).
    #[inline]
    pub fn step_angle_bits(&mut self, theta_bits: u32) -> Option<()> {
        let (neg, mag) = q0_192_phase_increment_from_f32_bits(theta_bits)?;
        self.advance_signed(neg, mag);
        Some(())
    }

    /// Maps the top 32 bits of the `Q0.192` phase to a unit phasor `(cos, sin)` (`ref[dsp.op-m.rotor.q0-192-lfo]`).
    #[inline]
    pub fn to_phasor(self) -> (f32, f32) {
        let q32 = (self.words[2] >> 32) as u32;
        q32_phase_to_phasor(q32)
    }
}

/// Computes the signed `Q0.192` increment `(negative, [u64; 3])` for an `f32` angle bit pattern
/// `theta_bits` with `|theta| < pi` (`ref[dsp.op-m.rotor.q0-192-lfo]`).
pub fn q0_192_phase_increment_from_f32_bits(theta_bits: u32) -> Option<(bool, [u64; 3])> {
    let negative = (theta_bits >> 31) != 0;
    let magnitude = theta_bits & 0x7fff_ffff;
    if magnitude > MAX_ROTOR_ANGLE_F32_BITS {
        return None;
    }
    if magnitude == 0 {
        return Some((false, [0, 0, 0]));
    }
    let exponent = magnitude >> 23;
    let mut mantissa = (magnitude & 0x007f_ffff) as u64;
    let power: i32 = if exponent != 0 {
        mantissa |= 1 << 23;
        (exponent as i32) - 150
    } else {
        -149
    };

    // Multiply 24-bit `mantissa` by 256-bit `INV_TAU_256_LE` -> 5 x u64 little-endian limbs (`prod[0..5]`).
    let mut prod = [0u64; 5];
    let mut carry = 0u128;
    for i in 0..4 {
        let cur = (mantissa as u128) * (INV_TAU_256_LE[i] as u128) + carry;
        prod[i] = cur as u64;
        carry = cur >> 64;
    }
    prod[4] = carry as u64;

    // Shift right by `shift = 64 - power` with ties-to-even rounding.
    let shift = (64 - power) as u32;
    let word_shift = (shift / 64) as usize;
    let bit_shift = shift % 64;

    let mut out = [0u64; 3];
    for (i, slot) in out.iter_mut().enumerate() {
        let src = i + word_shift;
        if src < 5 {
            let lo = prod[src] >> bit_shift;
            let hi = if bit_shift > 0 && src + 1 < 5 {
                prod[src + 1] << (64 - bit_shift)
            } else {
                0
            };
            *slot = lo | hi;
        }
    }

    // Check rounding bit at `shift - 1` and sticky bits below `shift - 1`.
    let half_pos = shift - 1;
    let half_word = (half_pos / 64) as usize;
    let half_bit = half_pos % 64;
    let half_set = half_word < 5 && ((prod[half_word] >> half_bit) & 1) != 0;

    let mut sticky = false;
    for &w in prod.iter().take(half_word.min(5)) {
        if w != 0 {
            sticky = true;
            break;
        }
    }
    if !sticky && half_word < 5 && half_bit > 0 {
        let mask = (1u64 << half_bit) - 1;
        if (prod[half_word] & mask) != 0 {
            sticky = true;
        }
    }

    if half_set && (sticky || (out[0] & 1) != 0) {
        let (w0, c0) = out[0].overflowing_add(1);
        let (w1, c1) = out[1].overflowing_add(u64::from(c0));
        let w2 = out[2].wrapping_add(u64::from(c1));
        out = [w0, w1, w2];
    }

    Some((negative, out))
}

/// Maps a 32-bit `Q0.32` turn phase `q32` into a unit phasor `(cos, sin)` by splitting quadrant
/// `j = q32 >> 30` and residual `n = q32 & 0x3FFF_FFFF`, multiplying by `0x30C90FDB` ($2\pi/2^{32}$),
/// and applying the quadrant-reduced contractive rotor (`ref[dsp.op-m.rotor.q0-192-lfo]`, `ref[dsp.tempo.q64-q32-phase]`).
#[inline]
pub fn q32_phase_to_phasor(q32: u32) -> (f32, f32) {
    let quadrant = q32 >> 30;
    let remainder = q32 & 0x3fff_ffff;
    let scale = f32::from_bits(0x30c9_0fdb);
    let angle = (remainder as f32) * scale;
    let (c, s) = contractive_rotor_f32(angle).unwrap_or((1.0, 0.0));
    match quadrant {
        0 => (c, s),
        1 => (-s, c),
        2 => (-c, -s),
        _ => (s, -c),
    }
}

// ============================================================================
// Pitch-Ratio and Time-Warp Profile (`ref[dsp.ctrl.math.pitch-profile]`)
// ============================================================================

/// Scale factor $\ln(2)/12$ in IEEE-754 `f64` (`0x3FAD9303FEA2F7EA`, `ref[dsp.ctrl.math.pitch-profile]`).
pub const PITCH_LN2_OVER_12_F64_BITS: u64 = 0x3FAD_9303_FEA2_F7EA;

/// Evaluates the degree-14 `f64` Horner pitch-ratio $R = 2^{p/12}$ and time-warp $\nu = 1 - R$
/// for a finite `f32` semitone shift $p \in [-24.0, +24.0]$ (`ref[dsp.ctrl.math.pitch-profile]`).
/// Returns `None` if `semitones` is non-finite or `|semitones| > 24.0`.
#[inline]
pub fn eval_pitch_profile_f32(semitones: f32) -> Option<(f32, f32)> {
    let p = widen_f32_bits(semitones.to_bits())?;
    if !p.is_finite() || p.abs() > 24.0 {
        return None;
    }
    if p == 0.0 {
        return Some((1.0, 0.0));
    }
    let scale = f64::from_bits(PITCH_LN2_OVER_12_F64_BITS);
    let x = p * scale;
    const INV_FACT: [f64; 14] = [
        1.0,
        1.0 / 2.0,
        1.0 / 6.0,
        1.0 / 24.0,
        1.0 / 120.0,
        1.0 / 720.0,
        1.0 / 5_040.0,
        1.0 / 40_320.0,
        1.0 / 362_880.0,
        1.0 / 3_628_800.0,
        1.0 / 39_916_800.0,
        1.0 / 479_001_600.0,
        1.0 / 6_227_020_800.0,
        1.0 / 87_178_291_200.0,
    ];
    let mut poly = INV_FACT[13];
    let mut idx = 13;
    while idx > 0 {
        idx -= 1;
        poly = INV_FACT[idx] + x * poly;
    }
    let delta = x * poly;
    let ratio_bits = round_f64_to_f32_bits(1.0 + delta)?;
    let warp_bits = round_f64_to_f32_bits(-delta)?;
    Some((f32::from_bits(ratio_bits), f32::from_bits(warp_bits)))
}

// ============================================================================
// Monotone f32 Log-Envelope Lookup Profile (`ref[dsp.ctrl.math.log2-lut]`)
// ============================================================================

/// 129-entry read-only `f32` bit table $T_j = \mathrm{RNE}_{f32}(\log_2(1 + j/128))$ (`516 B`,
/// SHA-256 `4d0d7c0ec194dd9d761bfe2982ca4847001f333c58a3f12f423ed2053de3a5f1`, `ref[dsp.ctrl.math.log2-lut]`).
pub const LOG2_MANTISSA_LUT_129_BITS: [u32; 129] = [
    0x00000000, 0x3c37f286, 0x3cb73cb4, 0x3d08e68f, 0x3d35d69c, 0x3d626fd6, 0x3d8759c5, 0x3d9d517f,
    0x3db31fb8, 0x3dc8c50b, 0x3dde4212, 0x3df39761, 0x3e0462c4, 0x3e0ee68d, 0x3e19574f, 0x3e23b550,
    0x3e2e00d2, 0x3e383a17, 0x3e42615f, 0x3e4c76e8, 0x3e567af1, 0x3e606db6, 0x3e6a4f72, 0x3e74205f,
    0x3e7de0b6, 0x3e83c857, 0x3e88983f, 0x3e8d602e, 0x3e92203d, 0x3e96d888, 0x3e9b8926, 0x3ea03232,
    0x3ea4d3c2, 0x3ea96df0, 0x3eae00d2, 0x3eb28c7f, 0x3eb7110e, 0x3ebb8e96, 0x3ec0052b, 0x3ec474e4,
    0x3ec8ddd4, 0x3ecd4012, 0x3ed19bb0, 0x3ed5f0c4, 0x3eda3f60, 0x3ede8797, 0x3ee2c97d, 0x3ee70525,
    0x3eeb3a9f, 0x3eef69ff, 0x3ef39355, 0x3ef7b6b4, 0x3efbd42b, 0x3effebcd, 0x3f01fed4, 0x3f0404e8,
    0x3f060828, 0x3f08089e, 0x3f0a0650, 0x3f0c0146, 0x3f0df989, 0x3f0fef1f, 0x3f11e20f, 0x3f13d260,
    0x3f15c01a, 0x3f17ab44, 0x3f1993e3, 0x3f1b7a00, 0x3f1d5da0, 0x3f1f3eca, 0x3f211d84, 0x3f22f9d5,
    0x3f24d3c2, 0x3f26ab53, 0x3f28808c, 0x3f2a5374, 0x3f2c2411, 0x3f2df268, 0x3f2fbe80, 0x3f31885c,
    0x3f335004, 0x3f35157d, 0x3f36d8cb, 0x3f3899f5, 0x3f3a58ff, 0x3f3c15ee, 0x3f3dd0c8, 0x3f3f8991,
    0x3f41404f, 0x3f42f506, 0x3f44a7ba, 0x3f465872, 0x3f480731, 0x3f49b3fb, 0x3f4b5ed7, 0x3f4d07c7,
    0x3f4eaed0, 0x3f5053f7, 0x3f51f740, 0x3f5398af, 0x3f553848, 0x3f56d60f, 0x3f587209, 0x3f5a0c3a,
    0x3f5ba4a4, 0x3f5d3b4e, 0x3f5ed039, 0x3f60636a, 0x3f61f4e5, 0x3f6384ad, 0x3f6512c7, 0x3f669f35,
    0x3f6829fb, 0x3f69b31e, 0x3f6b3a9f, 0x3f6cc083, 0x3f6e44cd, 0x3f6fc781, 0x3f7148a1, 0x3f72c832,
    0x3f744636, 0x3f75c2b0, 0x3f773da4, 0x3f78b714, 0x3f7a2f04, 0x3f7ba578, 0x3f7d1a71, 0x3f7e8df2,
    0x3f800000,
];

/// Status of decomposing an `f32` bit pattern for log-envelope lookup (`ref[dsp.ctrl.math.log2-lut]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Log2EnvelopeDecomp {
    /// Positive finite envelope $x = 2^E(1 + m/2^{23})$ with $E \in [-149, 127]$ and $m \in [0, 2^{23})$.
    Positive { exponent: i32, fraction: u32 },
    /// Exact zero (`+0.0` or `-0.0`): silent status.
    ZeroSilent,
    /// Negative or non-finite (`NaN`/`Inf`): rejected.
    Invalid,
}

/// Decomposes an `f32` bit pattern into `(exponent, 23-bit normalized fraction)` using leading-zero
/// normalization on subnormals (`ref[dsp.ctrl.math.log2-lut]`).
#[inline]
pub const fn decompose_positive_f32_bits(bits: u32) -> Log2EnvelopeDecomp {
    let magnitude = bits & 0x7fff_ffff;
    if magnitude == 0 {
        return Log2EnvelopeDecomp::ZeroSilent;
    }
    let exponent_bits = (bits >> 23) & 0xff;
    if (bits >> 31) != 0 || exponent_bits == 0xff {
        return Log2EnvelopeDecomp::Invalid;
    }
    let fraction = bits & 0x007f_ffff;
    if exponent_bits != 0 {
        Log2EnvelopeDecomp::Positive {
            exponent: (exponent_bits as i32) - 127,
            fraction,
        }
    } else {
        let top_bit = 31 - fraction.leading_zeros();
        let norm_frac = (fraction << (23 - top_bit)) & 0x007f_ffff;
        Log2EnvelopeDecomp::Positive {
            exponent: (top_bit as i32) - 149,
            fraction: norm_frac,
        }
    }
}

/// Evaluates the monotone `f32` fractional mantissa interpolation $p \in [0, 1]$ for `fraction < 2^23`
/// (`ref[dsp.ctrl.math.log2-lut]`).
#[inline]
pub fn lookup_log2_mantissa_f32(fraction: u32) -> f32 {
    let index = ((fraction >> 16) & 0x7f) as usize;
    let alpha = ((fraction & 0xffff) as f32) * f32::from_bits(0x3780_0000); // 2^-16
    let low = f32::from_bits(LOG2_MANTISSA_LUT_129_BITS[index]);
    let high = f32::from_bits(LOG2_MANTISSA_LUT_129_BITS[index + 1]);
    let d = high - low;
    high.min(low + alpha * d)
}

/// Evaluates $\widehat{\log_2(x)}$ for a positive finite `f32` bit pattern `bits` (`ref[dsp.ctrl.math.log2-lut]`).
/// Returns `Ok(Some(log2_x))` for positive finite `x`, `Ok(None)` for `±0.0`, and `Err(InvalidFloatInput)` for negative/non-finite.
#[inline]
pub fn eval_monotone_log2_f32_bits(bits: u32) -> Result<Option<f32>, InvalidFloatInput> {
    match decompose_positive_f32_bits(bits) {
        Log2EnvelopeDecomp::ZeroSilent => Ok(None),
        Log2EnvelopeDecomp::Invalid => Err(InvalidFloatInput),
        Log2EnvelopeDecomp::Positive { exponent, fraction } => {
            let p = lookup_log2_mantissa_f32(fraction);
            Ok(Some((exponent as f32) + p))
        }
    }
}

// ============================================================================
// Declared-Range Normalization & Ordered Admissible f32 Ranks
// (`ref[dsp.ctrl.range.normalize-clamp]`, `ref[dsp.ctrl.range.f32-rank-bounds]`)
// ============================================================================

/// Normalizes a physical `f32` value `x` into `[-1.0, +1.0]` against declared `f32` endpoints
/// `low < high` using cold-path `f64` reciprocal arithmetic (`ref[dsp.ctrl.range.normalize-clamp]`).
#[inline]
pub fn normalize_declared_range_bits(
    low_bits: u32,
    high_bits: u32,
    value_bits: u32,
) -> Option<f32> {
    let low = widen_f32_bits(low_bits)?;
    let high = widen_f32_bits(high_bits)?;
    let value = widen_f32_bits(value_bits)?;
    if !(low < high && value >= low && value <= high) {
        return None;
    }
    let span = high - low;
    let reciprocal = 1.0 / span;
    let distance = value - low;
    let product = distance * reciprocal;
    let coord = (2.0 * product - 1.0).clamp(-1.0, 1.0);
    let bits = round_f64_to_f32_bits(coord)?;
    Some(f32::from_bits(bits))
}

const F32_NEG_FIRST_KEY: u64 = 0x0080_0000;
const F32_NEG_NORMAL_END_KEY: u64 = 0x7f7f_ffff;
const F32_NEG_ZERO_KEY: u64 = 0x7fff_ffff;
const F32_POS_ZERO_KEY: u64 = 0x8000_0000;
const F32_POS_NORMAL_START_KEY: u64 = 0x8080_0000;
const F32_NORMALS_PER_SIGN: u64 = 254 * (1 << 23);

/// Returns the total cardinality of admissible `f32` physical knob values (`4,278,190,079` under
/// gradual underflow or `4,261,412,865` under FTZ/DAZ, `ref[dsp.ctrl.range.f32-rank-bounds]`).
#[inline]
pub const fn f32_coordinate_count(flush_subnormals: bool) -> u64 {
    if flush_subnormals {
        2 * F32_NORMALS_PER_SIGN + 1
    } else {
        2 * 255 * (1 << 23) - 1
    }
}

/// Maps an ordered rank `0 <= rank < f32_coordinate_count(flush_subnormals)` to its canonical `f32`
/// bit pattern (`+0.0` at the origin, `ref[dsp.ctrl.range.f32-rank-bounds]`).
#[inline]
pub const fn f32_bits_at_rank(rank: u64, flush_subnormals: bool) -> Option<u32> {
    let count = f32_coordinate_count(flush_subnormals);
    if rank >= count {
        return None;
    }
    let key = if flush_subnormals {
        if rank < F32_NORMALS_PER_SIGN {
            F32_NEG_FIRST_KEY + rank
        } else if rank == F32_NORMALS_PER_SIGN {
            F32_POS_ZERO_KEY
        } else {
            F32_POS_NORMAL_START_KEY + rank - F32_NORMALS_PER_SIGN - 1
        }
    } else {
        let mut k = F32_NEG_FIRST_KEY + rank;
        if k >= F32_NEG_ZERO_KEY {
            k += 1;
        }
        k
    };
    let bits = if (key & F32_POS_ZERO_KEY) != 0 {
        (key ^ F32_POS_ZERO_KEY) as u32
    } else {
        !(key as u32)
    };
    Some(bits)
}

/// Maps a finite `f32` bit pattern to its ordered rank in `[0, f32_coordinate_count(flush_subnormals))`
/// (`ref[dsp.ctrl.range.f32-rank-bounds]`). Returns `Ok(None)` if `bits` is a subnormal when
/// `flush_subnormals == true`, or `Err(InvalidFloatInput)` if `bits` is non-finite.
#[inline]
pub const fn f32_rank_of_bits(
    bits: u32,
    flush_subnormals: bool,
) -> Result<Option<u64>, InvalidFloatInput> {
    if ((bits >> 23) & 0xff) == 0xff {
        return Err(InvalidFloatInput);
    }
    let magnitude = bits & 0x7fff_ffff;
    let key = if magnitude == 0 {
        F32_POS_ZERO_KEY
    } else {
        if flush_subnormals && magnitude < 0x0080_0000 {
            return Ok(None);
        }
        if (bits & 0x8000_0000) != 0 {
            (!bits) as u64
        } else {
            (bits as u64) ^ F32_POS_ZERO_KEY
        }
    };
    let rank = if flush_subnormals {
        if key <= F32_NEG_NORMAL_END_KEY {
            key - F32_NEG_FIRST_KEY
        } else if key == F32_POS_ZERO_KEY {
            F32_NORMALS_PER_SIGN
        } else {
            F32_NORMALS_PER_SIGN + 1 + key - F32_POS_NORMAL_START_KEY
        }
    } else {
        key - F32_NEG_FIRST_KEY - if key >= F32_POS_ZERO_KEY { 1 } else { 0 }
    };
    Ok(Some(rank))
}

/// Binary-searches the rank boundary for a finite `binary64` bound (`upper == false` -> lower bound
/// `first >= value`; `upper == true` -> upper exclusive bound `first > value`, `ref[dsp.ctrl.range.f32-rank-bounds]`).
pub fn f32_rank_boundary_for_f64(value: f64, flush_subnormals: bool, upper: bool) -> Option<u64> {
    if !value.is_finite() {
        return None;
    }
    let mut low = 0u64;
    let mut high = f32_coordinate_count(flush_subnormals);
    while low < high {
        let mid = low + (high - low) / 2;
        let current = widen_f32_bits(f32_bits_at_rank(mid, flush_subnormals)?)?;
        if current < value || (upper && current == value) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    Some(low)
}

/// Intersects slices of `binary64` lower and upper bounds onto the inward admissible `f32` rank
/// interval `[first, end)` (`first < end`, `ref[dsp.ctrl.range.f32-rank-bounds]`).
pub fn f32_rank_intersection(
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    flush_subnormals: bool,
) -> Option<(u64, u64)> {
    let count = f32_coordinate_count(flush_subnormals);
    let mut first = 0u64;
    for &lo in lower_bounds {
        let b = f32_rank_boundary_for_f64(lo, flush_subnormals, false)?;
        if b > first {
            first = b;
        }
    }
    let mut end = count;
    for &hi in upper_bounds {
        let b = f32_rank_boundary_for_f64(hi, flush_subnormals, true)?;
        if b < end {
            end = b;
        }
    }
    if first >= end {
        return None;
    }
    Some((first, end))
}

/// Maps any finite `f32` request bit pattern to the nearest admissible `f32` bit pattern in `[first, end)`
/// (ties toward zero, `ref[dsp.ctrl.range.f32-rank-bounds]`).
pub fn f32_nearest_admissible_bits(
    requested_bits: u32,
    first: u64,
    end: u64,
    flush_subnormals: bool,
) -> Option<u32> {
    let count = f32_coordinate_count(flush_subnormals);
    if !(first < end && end <= count) {
        return None;
    }
    let req_f64 = widen_f32_bits(requested_bits)?;
    let at = f32_rank_boundary_for_f64(req_f64, flush_subnormals, false)?;
    let idx0 = at.saturating_sub(1).clamp(first, end - 1);
    let idx1 = at.clamp(first, end - 1);
    let b0 = f32_bits_at_rank(idx0, flush_subnormals)?;
    if idx0 == idx1 {
        return Some(b0);
    }
    let b1 = f32_bits_at_rank(idx1, flush_subnormals)?;
    let v0 = widen_f32_bits(b0)?;
    let v1 = widen_f32_bits(b1)?;
    let d0 = (v0 - req_f64).abs();
    let d1 = (v1 - req_f64).abs();
    if d0 < d1 {
        Some(b0)
    } else if d1 < d0 {
        Some(b1)
    } else if v0.abs() < v1.abs() {
        Some(b0)
    } else if v1.abs() < v0.abs() {
        Some(b1)
    } else if v0 <= v1 {
        Some(b0)
    } else {
        Some(b1)
    }
}

// ============================================================================
// 6-Step Newton Reciprocal for Saturated PolyADAA (`ref[dsp.op-n.polyadaa.rail-crossing]`)
// ============================================================================

/// Evaluates the 6-step division-free Newton reciprocal $1 / \Delta$ for $\Delta > 0$ in `f64`
/// (`< 2^-44` relative error, `ref[dsp.op-n.polyadaa.rail-crossing]`).
#[inline]
pub fn newton6_reciprocal_f64(distance: f64) -> f64 {
    let exponent = (((distance.to_bits() >> 52) & 0x7ff) as i32) - 1023;
    let mut inverse = f64::from_bits(((1023 - exponent - 1) as u64) << 52);
    for _ in 0..6 {
        inverse *= 2.0 - distance * inverse;
    }
    inverse
}

#[cfg(test)]
mod spec_tests {
    #[allow(unused_imports)]
    use alloc::{
        boxed::Box,
        format,
        string::{String, ToString},
        vec,
        vec::Vec,
    };

    // Specification Verification Suite: Phase 1 — Mathematical & Floating-Point Foundation.
    //
    // Verifies:
    // - `verify[arch.fp.callback-env]`
    // - `verify[arch.fp.range-saturation]`
    // - `verify[arch.simd.lane-packing]`
    // - `verify[dsp.op-m.rotor.contractive]`
    // - `verify[dsp.op-m.rotor.zero-hold]`
    // - `verify[dsp.op-m.rotor.quadrant-horner]`
    // - `verify[dsp.op-m.rotor.ulp-guard]`
    // - `verify[dsp.op-m.rotor.q0-192-lfo]`
    // - `verify[dsp.ctrl.math.pitch-profile]`
    // - `verify[dsp.ctrl.math.log2-lut]`
    // - `verify[dsp.ctrl.op-n.trust-clip]`
    // - `verify[dsp.op-n.ctrl.trust-clip]`
    // - `verify[dsp.ctrl.range.normalize-clamp]`
    // - `verify[dsp.ctrl.range.f32-rank-bounds]`
    // - `verify[dsp.op-n.polyadaa.rail-crossing]`

    use crate::test_support as support;

    use crate::math::{
        CERTIFIED_INPUT_ENVELOPE_MIN_F32, DefaultMathBackend, FP16_MAX_FINITE_F32,
        LOG2_MANTISSA_LUT_129_BITS, Log2EnvelopeDecomp, MathBackend, Q0_192Phase,
        SAFETY_CEILING_F32, ScalarBackend, SimdVec4, contractive_rotor_f32,
        contractive_rotor_f32_bits, decompose_positive_f32_bits, eval_monotone_log2_f32_bits,
        eval_pitch_profile_f32, f32_bits_at_rank, f32_coordinate_count,
        f32_nearest_admissible_bits, f32_rank_boundary_for_f64, f32_rank_intersection,
        f32_rank_of_bits, f32_to_fp16_with_telemetry, flush_subnormal_f32_bits,
        lookup_log2_mantissa_f32, newton6_reciprocal_f64, normalize_declared_range_bits,
        project_rotor_components_ulp_guard, q0_192_phase_increment_from_f32_bits,
        q32_phase_to_phasor, sanitize_input_sample, sanitize_internal_sample,
        sanitize_output_sample, saturation_flags, step_lfo_phasor_zero_hold,
        trust_clip_fast_candidate, trust_clip_outward_4d, trust_clip_outward_8d,
        trust_clip_outward_checked_bits, trust_clip_reference_bisection_bits, widen_f32_bits,
    };
    use crate::provenance::sha256_digest;
    use crate::{FtzDazGuard, Tier};
    use support::{Rat, SplitMix64};

    /// verify[arch.fp.callback-env]
    /// Verifies `FtzDazGuard` hardware control register policy (RNE + FTZ/DAZ) and portable subnormal flushing.
    #[test]
    fn test_fp_callback_environment_and_subnormal_flushing() {
        let _guard = FtzDazGuard::new();
        let one = std::hint::black_box(1.0_f32);
        let half_ulp = std::hint::black_box(f32::from_bits(0x3380_0000));
        assert_eq!((one + half_ulp).to_bits(), 1.0_f32.to_bits());

        // Portable branchless exponent-mask zeroing preserves sign, normals, infinities, and NaN payloads
        assert_eq!(
            flush_subnormal_f32_bits(f32::from_bits(0x0000_0001)).to_bits(),
            0x0000_0000
        );
        assert_eq!(
            flush_subnormal_f32_bits(f32::from_bits(0x807f_ffff)).to_bits(),
            0x8000_0000
        );
        assert_eq!(
            flush_subnormal_f32_bits(f32::from_bits(0x0080_0000)).to_bits(),
            0x0080_0000
        );
        assert_eq!(
            flush_subnormal_f32_bits(f32::from_bits(0x7f80_0000)).to_bits(),
            0x7f80_0000
        );
        assert_eq!(
            flush_subnormal_f32_bits(f32::from_bits(0x7fc0_1234)).to_bits(),
            0x7fc0_1234
        );
    }

    /// verify[arch.fp.range-saturation]
    /// Verifies `±100.0` operating envelope, `±1e9` and `fp16` (`±65504.0`) safety guards, and all 6 telemetry flags.
    #[test]
    fn test_range_saturation_policy_and_all_six_flags() {
        assert_eq!(CERTIFIED_INPUT_ENVELOPE_MIN_F32, 100.0);
        assert_eq!(SAFETY_CEILING_F32, 1.0e9);
        assert_eq!(FP16_MAX_FINITE_F32, 65_504.0);

        let mut flags = 0u32;
        assert_eq!(sanitize_input_sample(100.0, 100.0, &mut flags), 100.0);
        assert_eq!(flags, 0);

        assert_eq!(sanitize_input_sample(100.5, 100.0, &mut flags), 100.5);
        assert_eq!(flags, saturation_flags::OUTSIDE_CERTIFIED_INPUT);

        flags = 0;
        assert_eq!(
            sanitize_input_sample(2.0e9, 100.0, &mut flags),
            SAFETY_CEILING_F32
        );
        assert_eq!(
            flags,
            saturation_flags::OUTSIDE_CERTIFIED_INPUT | saturation_flags::INPUT_SATURATED
        );

        flags = 0;
        assert_eq!(
            sanitize_internal_sample(-2.0e9, &mut flags),
            -SAFETY_CEILING_F32
        );
        assert_eq!(flags, saturation_flags::INTERNAL_SATURATED);

        flags = 0;
        assert_eq!(
            sanitize_output_sample(5.0e9, &mut flags),
            SAFETY_CEILING_F32
        );
        assert_eq!(flags, saturation_flags::OUTPUT_SATURATED);

        flags = 0;
        let h_ok = f32_to_fp16_with_telemetry(65_504.0, &mut flags);
        assert_eq!(h_ok, 0x7bff);
        assert_eq!(flags, 0);

        let h_sat = f32_to_fp16_with_telemetry(70_000.0, &mut flags);
        assert_eq!(h_sat, 0x7bff);
        assert_eq!(flags, saturation_flags::FP16_SATURATED);

        flags = 0;
        assert_eq!(sanitize_input_sample(f32::NAN, 100.0, &mut flags), 0.0);
        assert_eq!(flags, saturation_flags::INVALID_NONFINITE);
    }

    /// verify[arch.simd.lane-packing]
    /// Verifies 4-lane block packing across `Lite` (4 ch), `Full` (8 ch), `Pro` (12 ch), and `Ultra` (16 ch),
    /// and bit parity between `ScalarBackend` and `DefaultMathBackend`.
    #[test]
    fn test_simd_4lane_packing_and_tier_hierarchy() {
        assert_eq!((Tier::Lite.channels(), Tier::Lite.simd_groups()), (4, 1));
        assert_eq!((Tier::Full.channels(), Tier::Full.simd_groups()), (8, 2));
        assert_eq!((Tier::Pro.channels(), Tier::Pro.simd_groups()), (12, 3));
        assert_eq!((Tier::Ultra.channels(), Tier::Ultra.simd_groups()), (16, 4));

        let _guard = FtzDazGuard::new();
        let u_prev = [-0.5_f32, 0.0, 0.25, 0.75];
        let u_curr = [-0.25_f32, 0.125, 0.5, -0.5];
        let beta = [0.8_f32, 0.9, 1.0, 0.7];
        let delta = [-0.2_f32, 0.1, -0.15, 0.25];

        let s_out = ScalarBackend::poly_adaa_algebraic_4ch(
            <ScalarBackend as MathBackend>::Vec4::from_array(u_prev),
            <ScalarBackend as MathBackend>::Vec4::from_array(u_curr),
            <ScalarBackend as MathBackend>::Vec4::from_array(beta),
            <ScalarBackend as MathBackend>::Vec4::from_array(delta),
        )
        .to_array();
        let d_out = DefaultMathBackend::poly_adaa_algebraic_4ch(
            <DefaultMathBackend as MathBackend>::Vec4::from_array(u_prev),
            <DefaultMathBackend as MathBackend>::Vec4::from_array(u_curr),
            <DefaultMathBackend as MathBackend>::Vec4::from_array(beta),
            <DefaultMathBackend as MathBackend>::Vec4::from_array(delta),
        )
        .to_array();
        for lane in 0..4 {
            assert!((s_out[lane] - d_out[lane]).abs() < 1e-6);
        }
    }

    /// verify[dsp.op-m.rotor.contractive]
    /// verify[dsp.op-m.rotor.zero-hold]
    /// verify[dsp.op-m.rotor.quadrant-horner]
    /// verify[dsp.op-m.rotor.ulp-guard]
    /// Verifies quadrant-reduced degree-13/14 Horner rotor accuracy (`< 1.6e-7`), strict contractive
    /// norm (`c^2 + s^2 < 1` for non-zero angles), signed symmetry, quadrant boundaries, and `±0.0` hold.
    #[test]
    fn test_contractive_rotor_zero_hold_quadrant_horner_and_ulp_guard() {
        let _guard = FtzDazGuard::new();

        // 1. Exact zero-increment hold on +0.0 and -0.0
        assert_eq!(
            contractive_rotor_f32_bits(0x0000_0000),
            Some((1.0_f32.to_bits(), 0.0_f32.to_bits()))
        );
        assert_eq!(
            contractive_rotor_f32_bits(0x8000_0000),
            Some((1.0_f32.to_bits(), 0.0_f32.to_bits()))
        );
        let held = (0.6_f32, 0.8_f32);
        assert_eq!(step_lfo_phasor_zero_hold(held.0, held.1, 0.0), Some(held));
        assert_eq!(step_lfo_phasor_zero_hold(held.0, held.1, -0.0), Some(held));

        // 2. Rejects |theta| >= pi or non-finite
        for bad in [
            core::f32::consts::PI,
            -core::f32::consts::PI,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ] {
            assert!(contractive_rotor_f32(bad).is_none());
        }

        // 3. Signed angles & subnormal/boundary angles
        for bits in [1_u32, 0x007f_ffff, 0x0080_0000, 0x3f80_0000, 0x4049_0fda] {
            let (cp, sp) = contractive_rotor_f32_bits(bits).unwrap();
            let (cn, sn) = contractive_rotor_f32_bits(bits | 0x8000_0000).unwrap();
            assert_eq!(cp, cn);
            assert_eq!(sp ^ 0x8000_0000, sn);
            let c = widen_f32_bits(cp).unwrap();
            let s = widen_f32_bits(sp).unwrap();
            assert!(c * c + s * s < 1.0);
        }

        // 4. Small cosine near pi/2 quadrant boundary
        let half_pi_bits = (core::f64::consts::FRAC_PI_2 as f32).to_bits();
        for bits in (half_pi_bits - 2)..=(half_pi_bits + 2) {
            let theta64 = widen_f32_bits(bits).unwrap();
            let (c_bits, _) = contractive_rotor_f32_bits(bits).unwrap();
            let c64 = widen_f32_bits(c_bits).unwrap();
            assert!((c64 - theta64.cos()).abs() < 1e-14);
        }

        // 5. ULP guard two-step limit verification
        let small_bits = (-30.0_f64).exp2() as f32;
        let (cg, sg) =
            project_rotor_components_ulp_guard(1.0_f32.to_bits(), small_bits.to_bits(), true);
        let cg64 = widen_f32_bits(cg).unwrap();
        let sg64 = widen_f32_bits(sg).unwrap();
        assert!(cg64 < 1.0);
        assert!(cg64 * cg64 + sg64 * sg64 < 1.0);

        let above_sqrt_half = f32::from_bits((0.5_f64.sqrt() as f32).to_bits() + 1);
        let (c2, s2) = project_rotor_components_ulp_guard(
            above_sqrt_half.to_bits(),
            above_sqrt_half.to_bits(),
            true,
        );
        let c2_64 = widen_f32_bits(c2).unwrap();
        let s2_64 = widen_f32_bits(s2).unwrap();
        assert_eq!(c2, above_sqrt_half.to_bits() - 2);
        assert_eq!(s2, above_sqrt_half.to_bits());
        assert!(c2_64 * c2_64 + s2_64 * s2_64 < 1.0);

        // 6. Sweep across 200,000 angles in (0, pi)
        let last_f32 = f32::from_bits(0x4049_0fda);
        let steps = 200_000usize;
        let mut max_comp_err = 0.0_f64;
        let mut max_ang_err = 0.0_f64;
        let mut min_norm_sq = 1.0_f64;
        for i in 1..=steps {
            let theta = (last_f32 as f64 * (i as f64) / (steps as f64)) as f32;
            let (c, s) = contractive_rotor_f32(theta).unwrap();
            let c64 = c as f64;
            let s64 = s as f64;
            let t64 = theta as f64;
            max_comp_err = max_comp_err
                .max((c64 - t64.cos()).abs())
                .max((s64 - t64.sin()).abs());
            max_ang_err = max_ang_err.max((s64.atan2(c64) - t64).abs());
            let n2 = c64 * c64 + s64 * s64;
            assert!(n2 < 1.0, "Non-contractive norm at theta={theta}: n2={n2}");
            min_norm_sq = min_norm_sq.min(n2);
        }
        assert!(max_comp_err < 1.6e-7);
        assert!(max_ang_err < 2.3e-7);
        assert!(min_norm_sq > 0.999_999_8);
    }

    /// verify[dsp.op-m.rotor.q0-192-lfo]
    /// Verifies `Q0.192` fixed-point LFO phase increment and slow-motion retention at `2^-30 rad/tick`.
    #[test]
    fn test_q0_192_lfo_phase_accumulator_and_slow_motion() {
        let _guard = FtzDazGuard::new();
        // Slow LFO angle 2^-30 rad/tick over 300,000 ticks
        let theta = f32::from_bits((127 - 30) << 23);
        let mut phase = Q0_192Phase {
            words: [0, 0, 1u64 << 61], // pi / 4 initial phase (1 << 189)
        };
        let initial = phase;
        assert_eq!(
            q0_192_phase_increment_from_f32_bits(0),
            Some((false, [0, 0, 0]))
        );
        phase.step_angle_bits(0).unwrap();
        assert_eq!(phase, initial);

        let (neg, step_words) = q0_192_phase_increment_from_f32_bits(theta.to_bits()).unwrap();
        assert!(!neg);
        assert!(step_words[0] >= (1u64 << 40));
        let (neg_sign, neg_words) =
            q0_192_phase_increment_from_f32_bits(theta.to_bits() | 0x8000_0000).unwrap();
        assert!(neg_sign);
        assert_eq!(neg_words, step_words);

        for _ in 0..300_000 {
            phase.step_angle_bits(theta.to_bits()).unwrap();
        }
        let (c, s) = phase.to_phasor();
        let actual_angle = (s as f64).atan2(c as f64);
        let expected_angle = core::f64::consts::FRAC_PI_4 + (theta as f64) * 300_000.0;
        assert!((actual_angle - expected_angle).abs() < 4e-7);

        // Verify q32_phase_to_phasor quadrant boundaries
        assert_eq!(q32_phase_to_phasor(0), (1.0, 0.0));
        assert_eq!(q32_phase_to_phasor(0x4000_0000), (0.0, 1.0));
        assert_eq!(q32_phase_to_phasor(0x8000_0000), (-1.0, 0.0));
        assert_eq!(q32_phase_to_phasor(0xc000_0000), (0.0, -1.0));
    }

    /// verify[dsp.ctrl.math.pitch-profile]
    /// Verifies degree-14 `f64` Horner pitch-ratio and time-warp profile accuracy, endpoints, and monotonicity.
    #[test]
    fn test_pitch_profile_accuracy_endpoints_and_monotonicity() {
        let _guard = FtzDazGuard::new();
        for &(st, exp_r, exp_w) in &[
            (-24.0_f32, 0.25_f32, 0.75_f32),
            (-12.0, 0.5, 0.5),
            (-0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (12.0, 2.0, -1.0),
            (24.0, 4.0, -3.0),
        ] {
            let (r, w) = eval_pitch_profile_f32(st).unwrap();
            assert_eq!(r, exp_r);
            assert_eq!(w, exp_w);
            if st == 0.0 {
                assert_eq!(w.to_bits(), 0);
            }
        }

        // Small non-zero semitone shift preserves non-zero warp even when R rounds to 1.0f32
        let (r_pos, w_pos) = eval_pitch_profile_f32(1.0e-8).unwrap();
        let (r_neg, w_neg) = eval_pitch_profile_f32(-1.0e-8).unwrap();
        assert_eq!(r_pos, 1.0);
        assert_eq!(r_neg, 1.0);
        assert!(w_pos < 0.0);
        assert!(w_neg > 0.0);

        // Rejects out-of-range or non-finite semitones
        for bad in [
            -24.0001_f32,
            24.0001,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ] {
            assert!(eval_pitch_profile_f32(bad).is_none());
        }

        // Accuracy and monotonicity across [-24, 24] and adjacent f32 bit patterns
        let steps = 200_000usize;
        let mut prev_r = 0.0_f32;
        let mut prev_w = f32::INFINITY;
        for i in 0..=steps {
            let p = -24.0_f32 + 48.0_f32 * (i as f32) / (steps as f32);
            let (r, w) = eval_pitch_profile_f32(p).unwrap();
            let ideal_r = ((p as f64) / 12.0).exp2();
            let rel_err = ((r as f64) / ideal_r - 1.0).abs();
            let warp_err = ((w as f64) - (1.0 - ideal_r)).abs();
            assert!(rel_err < 6.2e-8, "Pitch ratio rel_err={rel_err} at p={p}");
            assert!(warp_err < 1.2e-7, "Time warp err={warp_err} at p={p}");
            assert!(r >= prev_r);
            assert!(w <= prev_w);
            prev_r = r;
            prev_w = w;
        }

        // Adjacent f32 inputs around -16.0 st where naive f32 reverses monotonicity
        let (_, w0) = eval_pitch_profile_f32(-f32::from_bits(0x4180_0010)).unwrap();
        let (_, w1) = eval_pitch_profile_f32(-f32::from_bits(0x4180_0011)).unwrap();
        assert!(w1 >= w0);
    }

    /// verify[dsp.ctrl.math.log2-lut]
    /// Verifies the 129-entry `log2` lookup table SHA-256, subnormal decomposition, monotonicity, and `< 1.9e-5` error.
    #[test]
    fn test_monotone_log2_lut_sha256_subnormals_and_accuracy() {
        let _guard = FtzDazGuard::new();
        let mut table_bytes = [0u8; 129 * 4];
        for (i, &bits) in LOG2_MANTISSA_LUT_129_BITS.iter().enumerate() {
            table_bytes[i * 4..(i + 1) * 4].copy_from_slice(&bits.to_le_bytes());
            if i > 0 {
                assert!(f32::from_bits(bits) > f32::from_bits(LOG2_MANTISSA_LUT_129_BITS[i - 1]));
            }
        }
        let expected_sha256: [u8; 32] = [
            0x4d, 0x0d, 0x7c, 0x0e, 0xc1, 0x94, 0xdd, 0x9d, 0x76, 0x1b, 0xfe, 0x29, 0x82, 0xca,
            0x48, 0x47, 0x00, 0x1f, 0x33, 0x3c, 0x58, 0xa3, 0xf1, 0x2f, 0x42, 0x3e, 0xd2, 0x05,
            0x3d, 0xe3, 0xa5, 0xf1,
        ];
        assert_eq!(sha256_digest(&table_bytes), expected_sha256);

        // Zero silent and negative/non-finite rejection
        assert_eq!(eval_monotone_log2_f32_bits(0x0000_0000), Ok(None));
        assert_eq!(eval_monotone_log2_f32_bits(0x8000_0000), Ok(None));
        for bad in [0x8000_0001_u32, 0xbf80_0000, 0x7f80_0000, 0x7fc0_0000] {
            assert!(eval_monotone_log2_f32_bits(bad).is_err());
        }

        // Subnormal decomposition
        assert_eq!(
            decompose_positive_f32_bits(0x0000_0001),
            Log2EnvelopeDecomp::Positive {
                exponent: -149,
                fraction: 0
            }
        );
        assert_eq!(
            decompose_positive_f32_bits(0x0080_0000),
            Log2EnvelopeDecomp::Positive {
                exponent: -126,
                fraction: 0
            }
        );

        // Monotonicity and < 1.9e-5 error across all 128 * 1024 sampled mantissas & subnormals
        let mut prev = -f64::INFINITY;
        for step in 0..=(1u32 << 17) {
            let frac = (step << 6).min((1 << 23) - 1);
            let p = lookup_log2_mantissa_f32(frac) as f64;
            let exact = (1.0 + (frac as f64) * (-23.0_f64).exp2()).log2();
            assert!((p - exact).abs() < 1.9e-5);
            assert!(p >= prev);
            prev = p;
        }

        let mut prev_sub = -f64::INFINITY;
        for top_bit in 0..23u32 {
            let low = 1u32 << top_bit;
            let high = (1u32 << (top_bit + 1)) - 1;
            for bits in [low, (low + high) / 2, high] {
                let val = eval_monotone_log2_f32_bits(bits).unwrap().unwrap() as f64;
                let exact = (bits as f64).log2() - 149.0;
                assert!((val - exact).abs() < 1.9e-5);
                assert!(val >= prev_sub);
                prev_sub = val;
            }
        }
    }

    /// verify[dsp.ctrl.op-n.trust-clip]
    /// verify[dsp.op-n.ctrl.trust-clip]
    /// Verifies bit-preserving `f32 <-> f64`, outward energy bounds, 6-step Newton fast path, and
    /// 32-step bisection fallback across 4-D and 8-D latent vectors under active `FtzDazGuard`.
    #[test]
    fn test_outward_checked_trust_region_clip_4d_and_8d() {
        let _guard = FtzDazGuard::new();
        let eps_bits = 0x0080_0000_u32;
        let radius_bits = 2.5_f32.to_bits();
        let r_f64 = widen_f32_bits(radius_bits).unwrap();

        // 1. Inside-ball vector is preserved bit-for-bit (including subnormals!)
        let inside = [1_u32, 0x8000_0001, 0x3f00_0000, 0xbf00_0000];
        let (out_in, scale_in, _) =
            trust_clip_outward_checked_bits(&inside, radius_bits, eps_bits).unwrap();
        assert_eq!(out_in, inside);
        assert_eq!(scale_in, 1.0);

        // 2. Outside-ball 4-D and 8-D vectors are strictly inside R^2 in exact rational arithmetic
        let mut rng = SplitMix64::new(0x7257_c119);
        for _ in 0..500 {
            let v4: [u32; 4] = core::array::from_fn(|_| {
                let f = rng.next_f32_range(-20.0, 20.0);
                f.to_bits()
            });
            let (fast_acc, fast_out, _) =
                trust_clip_fast_candidate(&v4, radius_bits, eps_bits).unwrap();
            assert!(fast_acc);
            let (out4, scale4, _) =
                trust_clip_outward_checked_bits(&v4, radius_bits, eps_bits).unwrap();
            if scale4 == 1.0 {
                assert_eq!(out4, v4);
            } else {
                assert_eq!(out4, fast_out);
            }
            let mut sq_sum = 0.0_f64;
            for &b in &out4 {
                let w = widen_f32_bits(b).unwrap();
                sq_sum += w * w;
            }
            assert!(sq_sum <= r_f64 * r_f64);

            let v8: [u32; 8] = core::array::from_fn(|_| rng.next_f32_range(-20.0, 20.0).to_bits());
            let (out8, _, _) = trust_clip_outward_checked_bits(&v8, radius_bits, eps_bits).unwrap();
            let (ref8, _, _) =
                trust_clip_reference_bisection_bits(&v8, radius_bits, eps_bits).unwrap();
            let mut sq8 = 0.0_f64;
            let mut sq_ref8 = 0.0_f64;
            for i in 0..8 {
                let w = widen_f32_bits(out8[i]).unwrap();
                let wr = widen_f32_bits(ref8[i]).unwrap();
                sq8 += w * w;
                sq_ref8 += wr * wr;
            }
            assert!(sq8 <= r_f64 * r_f64);
            assert!(sq_ref8 <= r_f64 * r_f64);
        }

        // 3. In-place 4-D and 8-D wrappers
        let mut z4 = [3.0_f32, -4.0, 2.0, -1.0];
        trust_clip_outward_4d(&mut z4, 2.5);
        let n4: f64 = z4.iter().map(|&v| (v as f64) * (v as f64)).sum();
        assert!(n4 <= 2.5 * 2.5);

        let mut z8 = [3.0_f32, -4.0, 2.0, -1.0, 1.5, -2.5, 0.5, -3.5];
        trust_clip_outward_8d(&mut z8, 2.5);
        let n8: f64 = z8.iter().map(|&v| (v as f64) * (v as f64)).sum();
        assert!(n8 <= 2.5 * 2.5);
    }

    /// verify[dsp.ctrl.range.normalize-clamp]
    /// verify[dsp.ctrl.range.f32-rank-bounds]
    /// Verifies `binary64` declared-range normalization (closely spaced `2^30` endpoints & wide `±f32::MAX` spans)
    /// and ordered admissible `f32` rank bounds (`4,278,190,079` vs `4,261,412,865`).
    #[test]
    fn test_declared_range_normalization_and_f32_rank_bounds() {
        let _guard = FtzDazGuard::new();

        // 1. Closely spaced endpoints 2^30 and 2^30 + 128
        let low = (1u64 << 30) as f32;
        let high = f32::from_bits(low.to_bits() + 1);
        assert_eq!(
            normalize_declared_range_bits(low.to_bits(), high.to_bits(), low.to_bits()),
            Some(-1.0)
        );
        assert_eq!(
            normalize_declared_range_bits(low.to_bits(), high.to_bits(), high.to_bits()),
            Some(1.0)
        );

        // 2. Extreme span [-f32::MAX, +f32::MAX]
        let max_bits = 0x7f7f_ffff_u32;
        let min_bits = 0xff7f_ffff_u32;
        assert_eq!(
            normalize_declared_range_bits(min_bits, max_bits, min_bits),
            Some(-1.0)
        );
        assert_eq!(
            normalize_declared_range_bits(min_bits, max_bits, 0),
            Some(0.0)
        );
        assert_eq!(
            normalize_declared_range_bits(min_bits, max_bits, max_bits),
            Some(1.0)
        );

        // 3. Cardinalities and rank round-trips
        assert_eq!(f32_coordinate_count(false), 4_278_190_079);
        assert_eq!(f32_coordinate_count(true), 4_261_412_865);
        for flush in [false, true] {
            let count = f32_coordinate_count(flush);
            assert_eq!(f32_bits_at_rank(0, flush), Some(0xff7f_ffff));
            assert_eq!(f32_bits_at_rank(count - 1, flush), Some(0x7f7f_ffff));
            let zero_rank = f32_rank_of_bits(0, flush).unwrap().unwrap();
            assert_eq!(f32_rank_of_bits(0x8000_0000, flush), Ok(Some(zero_rank)));
            assert_eq!(f32_bits_at_rank(zero_rank, flush), Some(0));
        }

        // 4. Inward binary64 bound intersection and halfway ties toward zero
        assert!(f32_rank_intersection(&[0.1], &[0.1], false).is_none());
        let (first, end) = f32_rank_intersection(&[0.1], &[0.2], false).unwrap();
        let lo_v = widen_f32_bits(f32_bits_at_rank(first, false).unwrap()).unwrap();
        let hi_v = widen_f32_bits(f32_bits_at_rank(end - 1, false).unwrap()).unwrap();
        assert!(lo_v >= 0.1 && hi_v <= 0.2);

        let min_norm_f64 = widen_f32_bits(0x0080_0000).unwrap();
        let (f_ftz, e_ftz) =
            f32_rank_intersection(&[-min_norm_f64], &[min_norm_f64], true).unwrap();
        assert_eq!(
            f32_nearest_admissible_bits(0x0040_0000, f_ftz, e_ftz, true),
            Some(0)
        );
        assert_eq!(
            f32_nearest_admissible_bits(0x8040_0000, f_ftz, e_ftz, true),
            Some(0)
        );
        assert_eq!(
            f32_nearest_admissible_bits(0x0040_0001, f_ftz, e_ftz, true),
            Some(0x0080_0000)
        );
        assert_eq!(
            f32_nearest_admissible_bits(0x8040_0001, f_ftz, e_ftz, true),
            Some(0x8080_0000)
        );
        assert!(f32_rank_boundary_for_f64(f64::NAN, false, false).is_none());
    }

    /// verify[dsp.op-n.polyadaa.rail-crossing]
    /// Verifies the 6-step division-free Newton reciprocal accuracy (`< 2^-44` relative error) across wide exponents.
    #[test]
    fn test_newton6_reciprocal_f64_accuracy() {
        let _guard = FtzDazGuard::new();
        let tol = f64::from_bits((1023 - 44) << 52);
        let mut rng = SplitMix64::new(0x1234_5678);
        for exp in -140..=128 {
            let base = (exp as f64).exp2();
            let d = base * (1.0 + rng.next_unit_f64());
            let inv = newton6_reciprocal_f64(d);
            let rel_err = (d * inv - 1.0).abs();
            assert!(rel_err < tol, "rel_err={rel_err} at d={d}");
        }
        let r = Rat::new(3, 8);
        assert_eq!(r.round_ties_to_even(), 0);
    }

    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// verify[arch.fp.range-saturation]
        /// verify[dsp.op-m.rotor.contractive]
        /// verify[dsp.op-m.rotor.ulp-guard]
        /// verify[dsp.ctrl.math.pitch-profile]
        /// verify[dsp.ctrl.math.log2-lut]
        /// verify[dsp.ctrl.op-n.trust-clip]
        #[test]
        fn prop_spec_math_rotor_pitch_log2_trust_clip_and_saturation(
            raw_x in prop::num::f32::ANY,
            theta in -3.141_592_5_f32..3.141_592_5_f32,
            semitones in -24.0_f32..=24.0_f32,
            pos_x in 1.0e-30_f32..1.0e30_f32,
            v4 in prop::array::uniform4(-25.0_f32..25.0_f32),
            radius in 0.25_f32..20.0_f32,
        ) {
            let _guard = FtzDazGuard::new();

            // 1. Saturation & sanitization
            let mut flags = 0u32;
            let s_in = sanitize_input_sample(raw_x, 100.0, &mut flags);
            prop_assert!(s_in.is_finite() && s_in.abs() <= SAFETY_CEILING_F32);
            let s_out = sanitize_output_sample(raw_x, &mut flags);
            prop_assert!(s_out.is_finite() && s_out.abs() <= SAFETY_CEILING_F32);

            // 2. Contractive rotor
            let (c, s) = contractive_rotor_f32(theta).unwrap();
            let norm_sq = (c as f64) * (c as f64) + (s as f64) * (s as f64);
            prop_assert!(norm_sq <= 1.0);
            prop_assert!((c as f64 - (theta as f64).cos()).abs() < 2e-7);
            prop_assert!((s as f64 - (theta as f64).sin()).abs() < 2e-7);

            // 3. Pitch profile & log2 LUT
            let (ratio, warp) = eval_pitch_profile_f32(semitones).unwrap();
            let exact_ratio = ((semitones as f64) / 12.0).exp2();
            prop_assert!(((ratio as f64 - exact_ratio) / exact_ratio).abs() < 6.2e-8);
            prop_assert!((warp as f64 - (1.0 - exact_ratio)).abs() < 1.2e-7);

            let approx_log2 = eval_monotone_log2_f32_bits(pos_x.to_bits()).unwrap().unwrap();
            let exact_log2 = (pos_x as f64).log2();
            prop_assert!((approx_log2 as f64 - exact_log2).abs() < 1.9e-5);

            // 4. Outward-checked 4-D trust clip
            let v4_bits = v4.map(f32::to_bits);
            let (out_bits, _, _) = trust_clip_outward_checked_bits(
                &v4_bits,
                radius.to_bits(),
                1.0e-12_f32.to_bits(),
            ).unwrap();
            let mut sq_sum = 0.0_f64;
            for b in out_bits {
                let w = widen_f32_bits(b).unwrap();
                sq_sum += w * w;
            }
            let r64 = widen_f32_bits(radius.to_bits()).unwrap();
            prop_assert!(sq_sum <= r64 * r64);
        }
    }
}

#[cfg(test)]
mod trust_clip_reference_tests {
    #[allow(unused_imports)]
    use alloc::{
        boxed::Box,
        format,
        string::{String, ToString},
        vec,
        vec::Vec,
    };

    // Bit-level trust-clipping reference checks, not production-engine tests.

    use crate::FtzDazGuard;

    fn widen_f32_bits(bits: u32) -> Option<f64> {
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

    fn round_right_even(mantissa: u64, shift: u32) -> u64 {
        if shift > 53 {
            return 0;
        }
        let truncated = mantissa >> shift;
        let remainder = mantissa & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        truncated + u64::from(remainder > halfway || (remainder == halfway && truncated & 1 != 0))
    }

    fn round_f64_to_f32_bits(value: f64) -> Option<u32> {
        let bits = value.to_bits();
        let sign = ((bits >> 32) as u32) & 0x8000_0000;
        let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
        let fraction = bits & 0x000f_ffff_ffff_ffff;
        if exponent_bits == 0 {
            return (fraction == 0).then_some(sign);
        }
        if exponent_bits == 0x7ff {
            return None;
        }
        let mut exponent = exponent_bits - 1023;
        let mantissa = (1_u64 << 52) | fraction;
        if exponent < -126 {
            let shift = (-exponent - 97) as u32;
            return Some(sign | round_right_even(mantissa, shift) as u32);
        }
        if exponent > 127 {
            return None;
        }
        let mut rounded = round_right_even(mantissa, 29);
        if rounded == 1 << 24 {
            exponent += 1;
            rounded >>= 1;
        }
        if exponent > 127 {
            return None;
        }
        Some(sign | (((exponent + 127) as u32) << 23) | (rounded as u32 & 0x007f_ffff))
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn native_guard_flushes_actual_f32_arithmetic() {
        let _guard = FtzDazGuard::new();
        let subnormal = std::hint::black_box(f32::from_bits(1));
        let one = std::hint::black_box(1.0_f32);
        let normal = std::hint::black_box(f32::MIN_POSITIVE);
        let half = std::hint::black_box(0.5_f32);
        assert_eq!((subnormal * one).to_bits(), 0);
        assert_eq!((normal * half).to_bits(), 0);
    }

    #[test]
    fn bitwise_widening_covers_every_subnormal_with_native_ftz_daz() {
        let _guard = FtzDazGuard::new();
        let quantum = f64::from_bits(((1023 - 149) as u64) << 52);

        for fraction in 0..=0x007f_ffff {
            let expected = (f64::from(fraction) * quantum).to_bits();
            let actual = widen_f32_bits(fraction).expect("finite input").to_bits();
            assert_eq!(actual, expected);
            let negative = widen_f32_bits(fraction | 0x8000_0000)
                .expect("finite input")
                .to_bits();
            assert_eq!(negative, expected | (1_u64 << 63));
        }

        for exponent in 1..=254 {
            for fraction in [0, 1, 0x003f_ffff, 0x007f_ffff] {
                for sign in [0, 0x8000_0000] {
                    let bits = sign | (exponent << 23) | fraction;
                    let value = std::hint::black_box(f32::from_bits(bits));
                    assert_eq!(
                        widen_f32_bits(bits).expect("finite input").to_bits(),
                        f64::from(value).to_bits()
                    );
                    assert_eq!(value.to_bits(), bits);
                }
            }
        }
        for bits in [1, 0x007f_ffff, 0x8000_0001, 0x807f_ffff] {
            let value = std::hint::black_box(f32::from_bits(bits));
            assert_eq!(value.to_bits(), bits);
        }
        assert!(widen_f32_bits(0x7f80_0000).is_none());
        assert!(widen_f32_bits(0x7fc0_0001).is_none());
    }

    #[test]
    fn bitwise_rounding_covers_every_subnormal_midpoint_with_native_ftz_daz() {
        let _guard = FtzDazGuard::new();
        let half_quantum = f64::from_bits(((1023 - 150) as u64) << 52);

        for lower in 0..0x0080_0000_u32 {
            let midpoint = f64::from(2 * lower + 1) * half_quantum;
            let midpoint_bits = midpoint.to_bits();
            let expected = lower + (lower & 1);
            assert_eq!(round_f64_to_f32_bits(midpoint), Some(expected));
            assert_eq!(
                round_f64_to_f32_bits(-midpoint),
                Some(expected | 0x8000_0000)
            );
            if lower % 127 == 0 {
                assert_eq!(
                    round_f64_to_f32_bits(f64::from_bits(midpoint_bits - 1)),
                    Some(lower)
                );
                assert_eq!(
                    round_f64_to_f32_bits(f64::from_bits(midpoint_bits + 1)),
                    Some(lower + 1)
                );
            }
        }

        assert_eq!(round_f64_to_f32_bits(0.0), Some(0));
        assert_eq!(round_f64_to_f32_bits(-0.0), Some(0x8000_0000));
        assert!(round_f64_to_f32_bits(f64::from_bits(1)).is_none());
        assert!(round_f64_to_f32_bits(f64::INFINITY).is_none());
        assert!(round_f64_to_f32_bits(f64::from_bits((1151_u64) << 52)).is_none());
        for exponent in 1..=254 {
            for fraction in [0, 1, 0x003f_ffff, 0x007f_fffe] {
                let lower = (exponent << 23) | fraction;
                let left = widen_f32_bits(lower).expect("finite input");
                let right = widen_f32_bits(lower + 1).expect("finite input");
                let midpoint = (left + right) * 0.5;
                let expected = lower + (lower & 1);
                assert_eq!(round_f64_to_f32_bits(midpoint), Some(expected));
                assert_eq!(
                    round_f64_to_f32_bits(-midpoint),
                    Some(expected | 0x8000_0000)
                );
                assert_eq!(
                    round_f64_to_f32_bits(f64::from_bits(midpoint.to_bits() - 1)),
                    Some(lower)
                );
                assert_eq!(
                    round_f64_to_f32_bits(f64::from_bits(midpoint.to_bits() + 1)),
                    Some(lower + 1)
                );
            }
        }
    }
}
