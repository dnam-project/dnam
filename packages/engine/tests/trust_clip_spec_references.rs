//! Bit-level trust-clipping reference checks, not production-engine tests.

use dnam_engine::FtzDazGuard;

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
