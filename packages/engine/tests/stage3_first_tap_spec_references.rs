//! Native first-tap feasibility examples, not production Stage III tests.

use dnam_engine::FtzDazGuard;
use std::hint::black_box;

const COEFFICIENT_BITS: [u32; 64] = [
    0x3d6c_9268,
    0x3f83_725f,
    0xbdf0_a4f1,
    0x3dad_b502,
    0xbd9d_a17d,
    0x3d95_380c,
    0xbd94_9c5f,
    0x3d91_792e,
    0xbd91_ce40,
    0x3d8d_d068,
    0xbd8b_eae0,
    0x3d85_3dac,
    0xbd80_189f,
    0x3d6c_7fa0,
    0xbd5b_a083,
    0x3d42_31cb,
    0xbd2b_f99b,
    0x3d0e_88f9,
    0xbcea_4a93,
    0x3cac_58c1,
    0xbc71_4876,
    0x3bf4_c5f0,
    0xbae2_2743,
    0xbb96_2d07,
    0x3c16_34d0,
    0xbc65_add4,
    0x3c8b_7e54,
    0xbca4_d3dc,
    0x3caf_0e76,
    0xbcb9_82b1,
    0x3cb5_d3c2,
    0xbcb2_e27a,
    0x3ca3_932b,
    0xbc96_3f77,
    0x3c7d_c51e,
    0xbc56_5369,
    0x3c1f_d21c,
    0xbbe7_f768,
    0x3b77_5394,
    0xbaa9_8a21,
    0xbacf_4513,
    0x3b66_e3c0,
    0xbbba_af04,
    0x3be2_53a0,
    0xbc07_70ed,
    0x3c0e_a839,
    0xbc18_1cd3,
    0x3c14_420d,
    0xbc13_3bf5,
    0x3c07_4843,
    0xbbfe_526f,
    0x3bdd_6c26,
    0xbbc6_2d29,
    0x3ba3_1dc4,
    0xbb8b_1039,
    0x3b56_8017,
    0xbb2d_823b,
    0x3af6_3e3f,
    0xbaba_d495,
    0x3a6a_e4c8,
    0xba23_1f37,
    0x39a6_0fed,
    0xb943_301a,
    0x386d_c990,
];

fn difference_coefficients() -> [f32; 63] {
    let mut prefix = 0.0_f64;
    let mut result = [0.0; 63];
    for (difference, bits) in result.iter_mut().zip(COEFFICIENT_BITS) {
        prefix += f64::from(f32::from_bits(bits));
        *difference = (prefix - 1.0) as f32;
    }
    result
}

fn step_difference(history: &[f32; 64], coefficients: &[f32; 63]) -> f32 {
    let mut output = history[0];
    for (offset, coefficient) in coefficients.iter().enumerate() {
        let delta = black_box(history[offset] - history[offset + 1]);
        let product = black_box(coefficient * delta);
        output = black_box(output + product);
    }
    output
}

fn step_previous_anchored(history: &[f32; 64], coefficients: &[f32; 63]) -> f32 {
    let mut output = history[1];
    for (offset, coefficient) in coefficients.iter().enumerate() {
        let delta = black_box(history[offset] - history[offset + 1]);
        let product = black_box(coefficient * delta);
        output = black_box(output + product);
    }
    output
}

fn step_six_point(
    history: &[f32; 8],
    integer: usize,
    anchor: usize,
    coefficients: &[f32; 5],
) -> f32 {
    let mut output = history[anchor];
    for (offset, coefficient) in coefficients.iter().enumerate() {
        let delta = black_box(history[integer - 2 + offset] - history[integer - 1 + offset]);
        let product = black_box(coefficient * delta);
        output = black_box(output + product);
    }
    output
}

fn step_six_point_half_delay(history: &[f32; 8]) -> f32 {
    let coefficients = [
        3.0_f32 / 256.0,
        -22.0 / 256.0,
        -0.5,
        22.0 / 256.0,
        -3.0 / 256.0,
    ];
    step_six_point(history, 4, 4, &coefficients)
}

fn exact_bounce_delay(rate: f64, tap: u8) -> (u64, u64, u64) {
    assert!((176_400.0..=384_000.0).contains(&rate) && (1..=63).contains(&tap));
    let bits = rate.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as u32 - 1023;
    let significand = (bits & ((1_u64 << 52) - 1)) | (1_u64 << 52);
    let denominator = 48_000_u64 << (52 - exponent);
    let numerator = u64::from(tap) * significand;
    (
        numerator / denominator,
        numerator % denominator,
        denominator,
    )
}

#[test]
fn difference_form_preserves_constant_normal_samples() {
    let _guard = FtzDazGuard::new();
    let coefficients = difference_coefficients();
    let mut direct = 0.0_f32;
    for bits in COEFFICIENT_BITS {
        direct = black_box(direct + f32::from_bits(bits));
    }
    assert_eq!(direct.to_bits(), 0x3f80_0002);

    for sample in [0.125_f32, -0.125, 1.0, -1.0, 100.0, -100.0] {
        let history = [sample; 64];
        assert_eq!(
            step_difference(&history, &coefficients).to_bits(),
            sample.to_bits()
        );
    }
}

#[test]
fn previous_sample_anchor_avoids_reference_rate_cancellation() {
    let _guard = FtzDazGuard::new();
    let mut history = [0.0_f32; 64];
    history[0] = black_box(1.0);
    history[1] = black_box(f32::from_bits(0x3300_0000));
    let mut current_anchored = [0.0_f32; 63];
    current_anchored[0] = -1.0;
    assert_eq!(step_difference(&history, &current_anchored).to_bits(), 0);
    assert_eq!(
        step_previous_anchored(&history, &[0.0; 63]).to_bits(),
        history[1].to_bits()
    );
}

#[test]
fn previous_sample_anchor_tracks_tested_20khz_tones() {
    let _guard = FtzDazGuard::new();
    let differences = difference_coefficients();
    for rate in [44_100_usize, 46_000, 48_000] {
        let weight = (48_000.0 - rate as f64) / 3_900.0;
        let mut residual = differences;
        residual[0] += 1.0;
        let coefficients = residual.map(|value| (f64::from(value) * weight) as f32);
        let mut history = [0.0; 64];
        let omega = core::f64::consts::TAU * 20_000.0 / rate as f64;
        let delay = rate as f64 / 48_000.0;
        let mut residual_energy = 0.0;
        let mut target_energy = 0.0;
        for index in 0..64 + rate / 100 {
            history.copy_within(0..63, 1);
            history[0] = (omega * index as f64).sin() as f32;
            let output = step_previous_anchored(&history, &coefficients);
            if index >= 64 {
                let target = (omega * (index as f64 - delay)).sin();
                residual_energy += (f64::from(output) - target).powi(2);
                target_energy += target.powi(2);
            }
        }
        assert!(10.0 * (residual_energy / target_energy).log10() < -35.0);
    }
}

#[test]
fn six_point_half_delay_preserves_constant_normal_samples() {
    let _guard = FtzDazGuard::new();
    for sample in [0.125_f32, -0.125, 1.0, -1.0, 100.0, -100.0] {
        assert_eq!(
            step_six_point_half_delay(&[sample; 8]).to_bits(),
            sample.to_bits()
        );
    }
}

#[test]
fn six_point_nearest_anchor_avoids_integer_boundary_cancellation() {
    let _guard = FtzDazGuard::new();
    let small = f32::from_bits(0x3300_0000);
    let history = [0.0, 1.0, 1.0, 1.0, small, small, small, 0.0];
    let nearest = [
        0x23ba_69dc,
        0xa517_7603,
        0x2608_e5bd,
        0x259d_4952,
        0xa40b_cf65,
    ]
    .map(f32::from_bits);
    let mut previous = nearest;
    previous[2] = -1.0;
    assert_eq!(step_six_point(&history, 3, 3, &previous), 0.0);
    assert!((step_six_point(&history, 3, 4, &nearest) - small).abs() < 1e-14);
}

#[test]
fn bounce_rate_decomposition_preserves_integer_boundary() {
    let rate = f64::from_bits(0x4106_99b6_db6d_b6db);
    assert_eq!((rate * 7.0 / 48_000.0).floor(), 27.0);
    assert_eq!(
        exact_bounce_delay(rate, 7),
        (26, 1_649_267_441_663_997, 1_649_267_441_664_000)
    );
    let (integer, remainder, denominator) = exact_bounce_delay(176_400.0, 16);
    assert_eq!(integer, 58);
    assert_eq!(remainder * 5, denominator * 4);
    assert_eq!(exact_bounce_delay(384_000.0, 63).0, 504);
    assert_eq!(exact_bounce_delay(384_000.0, 63).1, 0);
}

#[test]
fn six_point_half_delay_tracks_one_20khz_bounce_tone() {
    let _guard = FtzDazGuard::new();
    let mut history = [0.0_f32; 8];
    let omega = core::f64::consts::TAU * 20_000.0 / 216_000.0;
    let mut residual_energy = 0.0;
    let mut target_energy = 0.0;
    for index in 0..8 + 2_160 {
        history.copy_within(0..7, 1);
        history[0] = (omega * index as f64).sin() as f32;
        let output = step_six_point_half_delay(&history);
        if index >= 8 {
            let target = (omega * (index as f64 - 4.5)).sin();
            residual_energy += (f64::from(output) - target).powi(2);
            target_energy += target.powi(2);
        }
    }
    assert!(10.0 * (residual_energy / target_energy).log10() < -63.8);
}

#[test]
fn difference_form_tracks_one_20khz_tone() {
    let _guard = FtzDazGuard::new();
    let coefficients = difference_coefficients();
    let mut history = [0.0; 64];
    let omega = core::f64::consts::TAU * 20_000.0 / 44_100.0;
    let delay = 44_100.0 / 48_000.0;
    let mut residual_energy = 0.0;
    let mut target_energy = 0.0;
    for index in 0..64 + 441 {
        history.copy_within(0..63, 1);
        history[0] = (omega * index as f64).sin() as f32;
        let output = step_difference(&history, &coefficients);
        if index >= 64 {
            let target = (omega * (index as f64 - delay)).sin();
            residual_energy += (f64::from(output) - target).powi(2);
            target_energy += target.powi(2);
        }
    }
    assert!(10.0 * (residual_energy / target_energy).log10() < -35.5);
}
