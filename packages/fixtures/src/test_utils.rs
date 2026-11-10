//! Shared test-only helper functions for `dnam-fixtures` module unit tests.

#![forbid(unsafe_code)]

use crate::{BlackBoxModel, FixtureBitDepth};
use std::f32::consts::PI;

/// Exercises all `BlackBoxModel` format, metadata, channel/control count, and `reset` trait methods
/// on `model`, restoring its original `sample_rate_hz` and `bit_depth` before returning.
pub fn exercise_blackbox_metadata_and_format<M: BlackBoxModel>(model: &mut M) {
    let orig_sr = model.sample_rate_hz();
    let orig_bd = model.bit_depth();
    let meta = model.metadata();
    assert!(!meta.id.is_empty() && !meta.name.is_empty());
    assert!(model.num_inputs() >= 1);
    assert!(model.num_outputs() >= 1);
    assert_eq!(meta.controls.len(), model.num_controls());
    assert_eq!(meta.switches.len(), model.num_switches());
    let ctrls = model.metadata().default_controls();
    let sws = model.metadata().default_switches();
    let warm_in = [0.42_f32; 4];
    let mut initial_ref = [0.0_f32; 4];
    model.reset();
    let mut wide_out = [f32::NAN; 4];
    model.process_frame(
        &warm_in[..model.num_inputs().min(4)],
        &mut wide_out,
        &ctrls,
        &sws,
    );
    assert!(
        wide_out[model.num_outputs().min(4)..]
            .iter()
            .all(|&v| v == 0.0)
    );
    model.reset();
    model.process_frame(
        &warm_in[..model.num_inputs().min(4)],
        &mut initial_ref[..model.num_outputs().min(4)],
        &ctrls,
        &sws,
    );
    model.set_sample_rate_hz(96_000);
    assert!(model.sample_rate_hz() >= 8_000);
    model.set_bit_depth(FixtureBitDepth::Int16);
    let _ = model.bit_depth();
    model.set_sample_rate_hz(orig_sr);
    model.set_bit_depth(orig_bd);
    // Verify set_bit_depth resets internal DSP state and preserves 0-ULP parity with new()
    let mut warm_out = [0.0_f32; 4];
    for _ in 0..8 {
        model.process_frame(
            &warm_in[..model.num_inputs().min(4)],
            &mut warm_out[..model.num_outputs().min(4)],
            &ctrls,
            &sws,
        );
    }
    model.set_bit_depth(orig_bd);
    let mut after_set_bd = [0.0_f32; 4];
    model.process_frame(
        &warm_in[..model.num_inputs().min(4)],
        &mut after_set_bd[..model.num_outputs().min(4)],
        &ctrls,
        &sws,
    );
    assert_eq!(after_set_bd, initial_ref);
    model.reset();
    let mut after_reset = [0.0_f32; 4];
    model.process_frame(
        &warm_in[..model.num_inputs().min(4)],
        &mut after_reset[..model.num_outputs().min(4)],
        &ctrls,
        &sws,
    );
    assert_eq!(after_set_bd, after_reset);
    for _ in 0..4 {
        model.process_frame(
            &warm_in[..model.num_inputs().min(4)],
            &mut warm_out[..model.num_outputs().min(4)],
            &ctrls,
            &sws,
        );
    }
    model.set_sample_rate_hz(orig_sr);
    let mut after_set_sr = [0.0_f32; 4];
    model.process_frame(
        &warm_in[..model.num_inputs().min(4)],
        &mut after_set_sr[..model.num_outputs().min(4)],
        &ctrls,
        &sws,
    );
    assert_eq!(after_set_sr, after_reset);
    model.reset();
}

/// Generates a deterministic mono sine wave at `freq_hz` (`48 kHz` canonical sample rate).
#[must_use]
pub fn sine_wave(num_samples: usize, freq_hz: f32, amplitude: f32) -> Vec<f32> {
    sine_wave_at_rate(num_samples, freq_hz, amplitude, 48_000)
}

/// Generates a deterministic mono sine wave at `freq_hz` for an arbitrary `sample_rate_hz`.
#[must_use]
pub fn sine_wave_at_rate(
    num_samples: usize,
    freq_hz: f32,
    amplitude: f32,
    sample_rate_hz: u32,
) -> Vec<f32> {
    let sr = (sample_rate_hz.max(1)) as f32;
    (0..num_samples)
        .map(|i| amplitude * (2.0 * PI * freq_hz * (i as f32) / sr).sin())
        .collect()
}

/// Resets `model` and runs a mono block through `BlackBoxModel::process_block`.
#[must_use]
pub fn run_mono_block<M: BlackBoxModel>(
    model: &mut M,
    input: &[f32],
    controls: &[f32],
    switches: &[u8],
) -> Vec<f32> {
    model.reset();
    let mut out = vec![0.0_f32; input.len()];
    let inputs = [input];
    let mut outputs = [out.as_mut_slice()];
    model.process_block(&inputs, &mut outputs, controls, switches);
    out
}

/// Resets `model` and runs a stereo block through `BlackBoxModel::process_block`.
#[must_use]
pub fn run_stereo_block<M: BlackBoxModel>(
    model: &mut M,
    in_l: &[f32],
    in_r: &[f32],
    controls: &[f32],
    switches: &[u8],
) -> (Vec<f32>, Vec<f32>) {
    model.reset();
    let n = in_l.len().min(in_r.len());
    let mut out_l = vec![0.0_f32; n];
    let mut out_r = vec![0.0_f32; n];
    let inputs = [&in_l[..n], &in_r[..n]];
    let mut outputs = [out_l.as_mut_slice(), out_r.as_mut_slice()];
    model.process_block(&inputs, &mut outputs, controls, switches);
    (out_l, out_r)
}
