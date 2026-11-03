//! Finite-precision Stage I reference examples, not production-path tests.

use dnam_engine::FtzDazGuard;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use std::hint::black_box;

const RATE: usize = 48_000;
const TONE_PERIOD: usize = 48;
const WARMUP: usize = RATE;
const OBSERVATION: usize = 4_800;
const AMPLITUDE: f64 = 1.0 / 4_194_304.0;
const I_BRANCH_BITS: [u32; 6] = [
    0x3d91_bee6,
    0x3eec_b8e1,
    0x3f48_de0d,
    0x3f6d_91c9,
    0x3f7a_5939,
    0x3f7e_8e9d,
];

#[derive(Default)]
struct HilbertBranch {
    state: [[f32; 2]; 6],
}

impl HilbertBranch {
    fn step(&mut self, mut value: f32) -> f32 {
        for (state, bits) in self.state.iter_mut().zip(I_BRANCH_BITS) {
            let coefficient = f32::from_bits(bits);
            let output = coefficient * value + state[1];
            state[1] = state[0];
            state[0] = coefficient * output - value;
            value = output;
        }
        value
    }
}

struct Blocker {
    pole: f32,
    gain: f32,
    previous_input: f32,
    previous_output: f32,
}

impl Blocker {
    fn new() -> Self {
        let pole = 0.999_345_7_f32;
        let gain = ((1.0_f64 + f64::from(pole)) * 0.5) as f32;
        Self {
            pole,
            gain,
            previous_input: 0.0,
            previous_output: 0.0,
        }
    }

    fn step(&mut self, value: f32) -> f32 {
        let output = self.gain * (value - self.previous_input) + self.pole * self.previous_output;
        self.previous_input = value;
        self.previous_output = output;
        output
    }
}

#[derive(Default)]
struct Projection {
    real: f64,
    imag: f64,
}

impl Projection {
    fn add(&mut self, sample: f32, index: usize) {
        let angle = core::f64::consts::TAU * (index % TONE_PERIOD) as f64 / TONE_PERIOD as f64;
        self.real += f64::from(sample) * angle.cos();
        self.imag -= f64::from(sample) * angle.sin();
    }

    fn magnitude(&self) -> f64 {
        self.real.hypot(self.imag)
    }
}

#[test]
fn preconditioned_tracker_preserves_a_quiet_tone_better_than_postconditioning_one_example() {
    let _guard = FtzDazGuard::new();
    let mut before_filter = Blocker::new();
    let mut before_branch = HilbertBranch::default();
    let mut after_branch = HilbertBranch::default();
    let mut after_filter = Blocker::new();
    assert_eq!(before_filter.pole.to_bits(), 0x3f7f_d51f);
    assert_eq!(before_filter.gain.to_bits(), 0x3f7f_ea90);
    let mut input_projection = Projection::default();
    let mut before_projection = Projection::default();
    let mut after_projection = Projection::default();

    for index in 0..WARMUP + OBSERVATION {
        let angle = core::f64::consts::TAU * (index % TONE_PERIOD) as f64 / TONE_PERIOD as f64;
        let input = (0.5 + AMPLITUDE * angle.cos()) as f32;
        let before = before_branch.step(before_filter.step(input));
        let after = after_filter.step(after_branch.step(input));
        if index >= WARMUP {
            input_projection.add(input, index);
            before_projection.add(before, index);
            after_projection.add(after, index);
        }
    }

    let input = input_projection.magnitude();
    let before = before_projection.magnitude();
    let difference = (after_projection.real - before_projection.real)
        .hypot(after_projection.imag - before_projection.imag);
    assert!((before / input - 1.0).abs() < 0.01);
    assert!(difference / before > 0.2);
}

#[test]
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn flushed_normalized_i_does_not_preserve_the_raw_detector_sign() {
    let _guard = FtzDazGuard::new();
    let squared_norm = black_box(1.0_f32);
    let mut inverse = f32::from_bits(0x5f37_59df - (squared_norm.to_bits() >> 1));
    for _ in 0..2 {
        let half_norm = black_box(0.5 * squared_norm);
        let product = black_box(black_box(half_norm * inverse) * inverse);
        inverse = black_box(inverse * black_box(1.5 - product));
    }

    let raw_i = black_box(f32::MIN_POSITIVE);
    let normalized_i = black_box(raw_i * inverse);
    let later_positive = black_box(1.0_f32);
    let earlier_negative = black_box(-1.0_f32);
    assert!(raw_i > 0.0);
    assert!(inverse < 1.0);
    assert_eq!(normalized_i.to_bits() & 0x7fff_ffff, 0);
    assert!(!(raw_i <= 0.0 && later_positive > 0.0));
    assert!(normalized_i <= 0.0 && later_positive > 0.0);
    assert!(earlier_negative <= 0.0 && raw_i > 0.0);
    assert!(!(earlier_negative <= 0.0 && normalized_i > 0.0));
}

#[test]
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn equal_transposed_cells_can_require_different_history_consistent_retunes() {
    fn step(coefficient: f32, state: &mut [f32; 2], input: f32) -> f32 {
        let output = black_box(black_box(coefficient * input) + state[1]);
        state[1] = state[0];
        state[0] = black_box(black_box(coefficient * output) - input);
        output
    }

    let _guard = FtzDazGuard::new();
    let old = black_box(f32::from_bits(0x3d91_bee6));
    let new = black_box(f32::from_bits(0x3db0_73c5));
    let mut driven_state = [0.0; 2];
    let mut outputs = [0.0; 3];
    for (sample, output) in [1.0, 0.0, -old].into_iter().zip(&mut outputs) {
        *output = step(old, &mut driven_state, sample);
    }
    assert_eq!(driven_state.map(f32::to_bits), [0; 2]);
    assert_eq!(&outputs[1..], &[0.0, -1.0]);

    let mut empty_state = [0.0; 2];
    let change = black_box(new - old);
    driven_state[0] = black_box(driven_state[0] + black_box(change * outputs[2]));
    driven_state[1] = black_box(driven_state[1] + black_box(change * outputs[1]));
    assert_ne!(driven_state, empty_state);

    assert_eq!(step(new, &mut driven_state, 0.0), 0.0);
    assert_eq!(step(new, &mut empty_state, 0.0), 0.0);
    let driven_second = step(new, &mut driven_state, 0.0);
    let empty_second = step(new, &mut empty_state, 0.0);
    assert!((-0.015..-0.014).contains(&driven_second));
    assert_eq!(empty_second, 0.0);
}
