//! Deterministic excitation signal and calibration probe synthesis.

use std::f32::consts::TAU;

/// Generates a deterministic 3-pulse + logarithmic chirp calibration probe.
pub fn generate_calibration_probe(sample_rate_hz: u32, num_samples: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; num_samples];
    let fs = sample_rate_hz.max(1) as f32;
    let pulse_positions = [
        num_samples / 8,
        (num_samples * 2) / 8,
        (num_samples * 3) / 8,
    ];
    for &pos in &pulse_positions {
        for k in 0..32 {
            if pos + k < num_samples {
                let w = 0.5 * (1.0 - ((TAU * (k as f32)) / 32.0).cos());
                let carrier = ((TAU * 2_400.0 * (k as f32)) / fs).sin();
                out[pos + k] = 0.85 * w * carrier;
            }
        }
    }
    let start_chirp = num_samples / 2;
    for i in start_chirp..num_samples {
        let t = ((i - start_chirp) as f32) / fs;
        let freq = 120.0 + 1_800.0 * t;
        out[i] = 0.35 * (TAU * freq * t).sin();
    }
    out
}

/// Generates a deterministic multi-band dry excitation signal for re-amp system identification.
pub fn generate_excitation_signal(sample_rate_hz: u32, num_samples: usize) -> Vec<f32> {
    let mut out = generate_calibration_probe(sample_rate_hz, num_samples);
    let fs = sample_rate_hz.max(1) as f32;
    for (idx, sample) in out.iter_mut().enumerate() {
        let t = (idx as f32) / fs;
        let env = 0.5 + 0.45 * (TAU * 1.7 * t).sin();
        let harmonic = 0.4 * (TAU * 110.0 * t).sin()
            + 0.25 * (TAU * 220.0 * t).sin()
            + 0.15 * (TAU * 440.0 * t).cos();
        *sample = (*sample + env * harmonic).clamp(-0.98, 0.98);
    }
    out
}
