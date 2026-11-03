//! Fixed-point tempo feasibility examples, not production tempo tests.

use proptest::prelude::*;

const RATIOS: [(u32, u32); 12] = [
    (3, 1),
    (2, 1),
    (4, 3),
    (3, 2),
    (1, 1),
    (2, 3),
    (3, 4),
    (1, 2),
    (1, 3),
    (3, 8),
    (1, 4),
    (1, 6),
];

const REM3: [u32; 24] = [
    0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
];

fn fixed_subdivision_phase(beat: u32, fraction: u32, ratio: (u32, u32), offset: u32) -> u32 {
    let (a, b) = ratio;
    let beat_remainder = match a {
        1 => 0,
        2 => beat & 1,
        3 => REM3[beat as usize],
        4 => beat & 3,
        _ => panic!("unsupported subdivision"),
    };
    let product = u64::from(b) * ((u64::from(beat_remainder) << 32) + u64::from(fraction));
    let high = (product >> 32) as usize;
    let low = product & u64::from(u32::MAX);
    let phase = match a {
        1 => low,
        2 => ((u64::from((high as u32) & 1) << 32) + low) >> 1,
        3 => {
            let remainder = u64::from(REM3[high]);
            remainder * 0x5555_5555 + (((low + remainder) * 0xaaaa_aaab) >> 33)
        }
        4 => ((u64::from((high as u32) & 3) << 32) + low) >> 2,
        _ => unreachable!(),
    };
    assert!(phase <= u64::from(u32::MAX));
    (phase as u32).wrapping_add(offset)
}

fn tick_beat(beat: u8, fraction: u64, step: u64) -> (u8, u64) {
    let (next, carry) = fraction.overflowing_add(step);
    let next_beat = if carry && beat == 11 {
        0
    } else {
        beat + u8::from(carry)
    };
    (next_beat, next)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4096))]

    #[test]
    fn division_free_phase_matches_integer_reference(
        beat in 0_u32..12,
        index in 0_usize..12,
        fraction: u32,
        offset: u32,
    ) {
        let (a, b) = RATIOS[index];
        let product = u64::from(b) * ((u64::from(beat % a) << 32) + u64::from(fraction));
        prop_assert!(product < 24_u64 << 32);
        let reference = ((product / u64::from(a)) as u32).wrapping_add(offset);
        prop_assert_eq!(fixed_subdivision_phase(beat, fraction, (a, b), offset), reference);
    }

    #[test]
    fn f32_phase_conversion_preserves_circular_accuracy(phase: u32) {
        let mut rounded = (phase as f32) * f32::from_bits(0x2f80_0000);
        if rounded == 1.0 {
            rounded = 0.0;
        }
        let exact = f64::from(phase) * f64::from(f32::from_bits(0x2f80_0000));
        let mut error = f64::from(rounded) - exact;
        if error < -0.5 {
            error += 1.0;
        } else if error >= 0.5 {
            error -= 1.0;
        }
        prop_assert!(error.abs() <= 2.0_f64.powi(-25));
    }

    #[test]
    fn q64_tick_matches_wide_integer_carry(beat in 0_u8..12, fraction: u64, step: u64) {
        let total = u128::from(fraction) + u128::from(step);
        let expected_fraction = total as u64;
        let expected_beat = if total > u128::from(u64::MAX) {
            (beat + 1) % 12
        } else {
            beat
        };
        prop_assert_eq!(tick_beat(beat, fraction, step), (expected_beat, expected_fraction));
    }
}

#[test]
fn fixed_phase_extremes_match_the_integer_reference() {
    for (a, b) in RATIOS {
        for beat in 0..12 {
            for fraction in [0, 1, u32::MAX / 2, u32::MAX - 1, u32::MAX] {
                for offset in [0, 1, u32::MAX / 2, u32::MAX] {
                    let product =
                        u64::from(b) * ((u64::from(beat % a) << 32) + u64::from(fraction));
                    let reference = ((product / u64::from(a)) as u32).wrapping_add(offset);
                    assert_eq!(
                        fixed_subdivision_phase(beat, fraction, (a, b), offset),
                        reference
                    );
                }
            }
        }
    }
}

#[test]
fn q64_tick_tracks_carries_and_twelve_beat_wrap() {
    assert_eq!(tick_beat(11, u64::MAX - 1, 3), (0, 1));
    assert_eq!(tick_beat(7, 42, 0), (7, 42));
    assert_eq!(tick_beat(2, 1, u64::MAX), (3, 0));
}

#[test]
fn largest_u32_phase_rounds_to_one_before_canonicalization() {
    let rounded = (u32::MAX as f32) * f32::from_bits(0x2f80_0000);
    assert_eq!(rounded, 1.0);
    let normalized = if rounded == 1.0 { 0.0 } else { rounded };
    assert_eq!(normalized, 0.0);
}
