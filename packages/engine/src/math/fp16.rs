//! Branchless IEEE-754 `f32` $\leftrightarrow$ `fp16` (`u16`) software conversion primitives.

/// Branchless IEEE-754 `f32` to `fp16` (`u16`) conversion (round-to-nearest-even, flushes subnormals).
#[inline(always)]
pub fn f32_to_fp16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;

    if exp == 255 {
        // Inf or NaN
        return sign | 0x7C00 | ((mant != 0) as u16);
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        // Clamp overflow to max finite fp16
        return sign | 0x7BFF;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign;
        }
        let m = mant | 0x0080_0000;
        let shift = (14 - new_exp) as u32;
        let half_mant = (m >> shift) as u16;
        return sign | half_mant;
    }
    let round_bit = (mant >> 12) & 1;
    let sticky = (mant & 0x0FFF) != 0;
    let mut half_mant = (mant >> 13) as u16;
    if round_bit != 0 && (sticky || (half_mant & 1) != 0) {
        half_mant += 1;
    }
    sign + (((new_exp as u16) << 10) + half_mant)
}

/// Branchless IEEE-754 `fp16` (`u16`) to `f32` conversion.
#[inline(always)]
pub fn fp16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x03FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal fp16
        let mut m = mant;
        let mut e = 127u32 - 14;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03FF;
        return f32::from_bits(sign | (e << 23) | (m << 13));
    }
    if exp == 31 {
        return f32::from_bits(sign | 0x7F80_0000 | (mant << 13));
    }
    let f32_exp = (exp + 127 - 15) << 23;
    f32::from_bits(sign | f32_exp | (mant << 13))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp16_roundtrip_and_special_values() {
        for &val in &[0.0f32, -0.0, 1.0, -1.0, 0.5, -0.25, 0.985, -0.985] {
            let half = f32_to_fp16(val);
            let back = fp16_to_f32(half);
            assert!((back - val).abs() < 1.0e-3);
        }

        let inf_h = f32_to_fp16(f32::INFINITY);
        assert!(fp16_to_f32(inf_h).is_infinite());

        let nan_h = f32_to_fp16(f32::NAN);
        assert!(fp16_to_f32(nan_h).is_nan());

        let ovf_h = f32_to_fp16(100_000.0);
        assert_eq!(ovf_h, 0x7BFF);

        let und_h = f32_to_fp16(1.0e-10);
        assert_eq!(fp16_to_f32(und_h), 0.0);

        let sub_h = f32_to_fp16(3.0e-5);
        assert!(fp16_to_f32(sub_h) > 0.0);
    }
}
