//! Saturated-curve ADAA arithmetic references, not production-engine tests.

use dnam_engine::FtzDazGuard;

#[inline(never)]
fn saturated_adaa_reference(previous: f64, current: f64) -> (f64, f64) {
    let lower = previous.min(current);
    let upper = previous.max(current);
    if lower >= 1.0 {
        return (1.0, 0.0);
    }
    if upper <= -1.0 {
        return (-1.0, 0.0);
    }

    let clamped_lower = lower.max(-1.0);
    let clamped_upper = upper.min(1.0);
    let midpoint = (clamped_upper + clamped_lower) * 0.5;
    let half_difference = (clamped_upper - clamped_lower) * 0.5;
    let m2 = midpoint * midpoint;
    let d2 = half_difference * half_difference;
    let odd = midpoint * (1.5 - 0.5 * (m2 + d2));
    let even = m2 + d2 * 0.3333333333333333 - (m2 * m2 + 2.0 * m2 * d2 + d2 * d2 * 0.2);
    if lower >= -1.0 && upper <= 1.0 {
        return (odd, even);
    }

    let distance = upper - lower;
    let exponent = ((distance.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let mut inverse = f64::from_bits(((1023 - exponent - 1) as u64) << 52);
    for _ in 0..6 {
        inverse *= 2.0 - distance * inverse;
    }
    let inside = (clamped_upper - clamped_lower).max(0.0);
    let below = (upper.min(-1.0) - lower).max(0.0);
    let above = (upper - lower.max(1.0)).max(0.0);
    (
        (inside * odd + above - below) * inverse,
        (inside * even) * inverse,
    )
}

#[test]
fn saturated_adaa_covers_rail_crossings_under_native_guard() {
    let _guard = FtzDazGuard::new();
    let cases = [
        (0.0, 1.0, 0.625, 0.13333333333333333),
        (0.0, 2.0, 0.8125, 0.06666666666666667),
        (-2.0, 0.0, -0.8125, 0.06666666666666667),
        (-2.0, 2.0, 0.0, 0.06666666666666667),
        (-1.0, 1.0, 0.0, 0.13333333333333333),
        (2.0, 4.0, 1.0, 0.0),
        (-4.0, -2.0, -1.0, 0.0),
        (0.25, 0.25, 0.3671875, 0.05859375),
    ];
    let tolerance = f64::from_bits(((1023 - 44) as u64) << 52);
    for (previous, current, expected_odd, expected_even) in cases {
        for (left, right) in [(previous, current), (current, previous)] {
            let (odd, even) = saturated_adaa_reference(left, right);
            assert!((odd - expected_odd).abs() < tolerance);
            assert!((even - expected_even).abs() < tolerance);
        }
    }
}

#[test]
fn saturated_adaa_preserves_tiny_even_result_in_guarded_f64() {
    let _guard = FtzDazGuard::new();
    let minimum_f32 = f64::from_bits(((1023 - 149) as u64) << 52);
    let (tiny_odd, tiny_even) = saturated_adaa_reference(minimum_f32, minimum_f32);
    assert_eq!(tiny_odd, 1.5 * minimum_f32);
    assert_eq!(tiny_even, minimum_f32 * minimum_f32);

    let maximum = f64::from(f32::MAX);
    let (odd, even) = saturated_adaa_reference(-maximum, maximum);
    let expected = 2.0 / (15.0 * maximum);
    let relative_tolerance = f64::from_bits(((1023 - 44) as u64) << 52);
    assert_eq!(odd, 0.0);
    assert!(even > 0.0);
    assert!((even / expected - 1.0).abs() < relative_tolerance);
}
