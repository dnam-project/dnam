#![forbid(unsafe_code)]
//! `dnam-capture` — Pure IO-Buffer Capture, Alignment, Single-File `.dnam-proj` (`DNPJ`) Container,
//! and Analytical Variable-Projection (`VarPro`) System-ID Engine.
//!
//! Shares the exact `#![no_std]` + `alloc` DSP operators (`Op-T`, `Op-M`, `Op-N`),
//! `3 kHz` Parametric Control Bus, `uplo='U'` Gram matrix packer, and `.dnam`
//! container primitives with [`dnam_engine`] (`ref[arch.isomorphism.two-sweep]`,
//! `ref[arch.isomorphism.audit-wiring]`).

pub mod align;
pub mod audit;
pub mod metrics;
pub mod proj;
pub mod resampler;
pub mod solver;
pub mod stimulus;
pub mod targets;
#[cfg(not(target_arch = "wasm32"))]
pub mod whitebox;

pub use align::{
    AlignmentRegime, DifferentialPadCalibration, LoopbackCalibrationReport, MultiRegimeAlignment,
    NullHomingReport, ThermalRecoveryReport, WavData, calibrate_loopback_calb, compute_pcm_sha256,
    decode_wav_bytes, encode_wav_12bit, encode_wav_16bit, encode_wav_24bit, encode_wav_32bit_float,
    encode_wav_32bit_int, encode_wav_with_bit_depth, estimate_multiregime_alignment,
    estimate_subsample_lag_and_gain, evaluate_acoustic_null_homing, evaluate_live_decay_tail,
    evaluate_thermal_recovery_gate, generate_calb_loopback_stimulus, generate_mic_pad_probe_400ms,
    measure_differential_pad_ratio, realign_fractional_kaiser64, rescale_padded_stem,
};
pub use resampler::{
    CANONICAL_TRAINER_SAMPLE_RATE_HZ, resample_48k_to_arbitrary_rate, resample_arbitrary_rate,
    resample_arbitrary_rate_to_48k, resample_control_trajectory, resampled_frame_len,
    scale_freeze_window,
};
pub use solver::{
    BuyerWatermarkConfig, SolverOptions, StageAConfig, StageAReport, StageBHarvest,
    StageCSolveOutput, StageDSolveOutput, StageESolveOutput, Wide192Accum,
    accumulate_uplo_u_wide192, compute_integrated_lufs, contract_256d_gram_tiled_wide192,
    derive_buyer_gauge_params, derive_buyer_phase_seed, derive_buyer_tardos_codebook,
    export_runtime_dnam_watermarked, extract_arena_pole_angles_34, finalize_stage_e_and_telemetry,
    harvest_star_center_256d, identify_pole_angles_blackbox_from_audio, lift_stage_a_backbone,
    render_blackbox_probe_response, solve_stage_c_nested_schur_4tier, solve_stage_d_cp1_and_audit,
};
pub use stimulus::{
    AdaptiveStimulusController, AdaptiveTrainingOptions, AdaptiveTrainingReport, BatchSolveReport,
    BatchStimulusOptions, CaptureWarning, LiveNullTestReport, StimulusConfig, TopologyProbeMode,
    generate_refinement_stimulus_batch, generate_stimulus_batch, recapture_clipped_stem,
    run_live_null_test_and_autopatch, solve_from_responses, train_adaptive_target,
};
pub use targets::{
    CaptureBitDepth, DistillationTarget, StemResponse, StemStimulus, StemStimulusBatch,
    StimulusFrame,
};

use dnam_engine::Tier;
use sha2::{Digest, Sha256};

/// Canonical reference sample rate (`48,000 Hz`) for Two-Sweep / `VarPro` system identification.
pub const REFERENCE_SAMPLE_RATE_HZ: u32 = 48_000;

/// Canonical `250 ms` `fp16` coreset window length (`12,000` frames at `48 kHz`, `ref[container.proj.stmc-capsule]`).
pub const CORESET_FP16_FRAMES_PER_STEM: usize = 12_000;

/// Canonical mono (`C = 1`) `fp16` coreset byte size (`12,000 * 1 * 2 B = 24,000 B`, `64 B`-aligned).
pub const CORESET_FP16_MONO_BYTES: usize = CORESET_FP16_FRAMES_PER_STEM * 2;

/// Canonical stereo (`C = 2`) `fp16` coreset byte size (`12,000 * 2 * 2 B = 48,000 B`, `64 B`-aligned).
pub const CORESET_FP16_STEREO_BYTES: usize = CORESET_FP16_FRAMES_PER_STEM * 4;

/// Parses a case-insensitive tier identifier (`lite`, `full`, `pro`, `ultra`) into [`Tier`].
pub fn parse_tier_name(name: &str) -> Result<Tier, String> {
    match name.trim().to_ascii_lowercase().as_str() {
        "lite" => Ok(Tier::Lite),
        "full" => Ok(Tier::Full),
        "pro" => Ok(Tier::Pro),
        "ultra" => Ok(Tier::Ultra),
        other => Err(format!("unknown slimmable tier '{other}'")),
    }
}

/// Computes the deterministic `DNAM-GRAM-INPUT-v2` SHA-256 fingerprint over ordered frames
/// (`ref[container.proj.stat-uplo-u]`).
#[must_use]
pub fn compute_gram_input_fingerprint(feature_dim: u32, frames: &[(u64, f64, &[f64])]) -> [u8; 32] {
    let expected_len = feature_dim as usize;
    let mut prev_frame_idx: Option<u64> = None;
    let mut hasher = Sha256::new();
    hasher.update(b"DNAM-GRAM-INPUT-v2\0");
    hasher.update(feature_dim.to_le_bytes());
    hasher.update([2u8, 2u8]); // feature_type=2 (f64), weight_type=2 (f64)
    for (frame_idx, weight, row) in frames {
        assert_eq!(
            row.len(),
            expected_len,
            "compute_gram_input_fingerprint requires row.len() ({}) == feature_dim ({expected_len})",
            row.len()
        );
        assert!(
            weight.is_finite() && *weight >= 0.0,
            "compute_gram_input_fingerprint requires finite nonnegative weight, got {weight}"
        );
        if let Some(prev) = prev_frame_idx {
            assert!(
                *frame_idx > prev,
                "compute_gram_input_fingerprint requires strictly increasing frame_idx ({frame_idx} <= {prev})"
            );
        }
        prev_frame_idx = Some(*frame_idx);
        let canon_w = if *weight == 0.0 { 0.0_f64 } else { *weight };
        hasher.update([1u8]);
        hasher.update(frame_idx.to_le_bytes());
        hasher.update(canon_w.to_le_bytes());
        for &val in *row {
            assert!(
                val.is_finite(),
                "compute_gram_input_fingerprint requires finite feature values, got {val}"
            );
            let canon_v = if val == 0.0 { 0.0_f64 } else { val };
            hasher.update(canon_v.to_le_bytes());
        }
    }
    hasher.update([0u8]);
    hasher.update((frames.len() as u64).to_le_bytes());
    hasher.finalize().into()
}

/// Quantizes an `f32` scalar to IEEE-754 binary16 (`fp16`) bits using round-to-nearest-even (`RNE`)
/// and full 10-bit subnormal encoding (`ref[container.proj.stmc-capsule]`).
#[must_use]
pub fn f32_to_fp16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 0xff {
        if mant == 0 {
            return sign | 0x7c00;
        }
        return sign | 0x7e00 | ((mant >> 13) as u16);
    }
    if exp == 0 {
        return sign;
    }
    let half_exp = exp - 127 + 15;
    if half_exp >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let sig = mant | 0x0080_0000;
        let shift = (14 - half_exp) as u32;
        let half_way = 1_u32 << (shift - 1);
        let lsb = (sig >> shift) & 1;
        let sub_mant = ((sig + (half_way - 1) + lsb) >> shift) as u16;
        return sign | sub_mant;
    }
    let lsb = (mant >> 13) & 1;
    let rounded_mant = ((mant + 0x0000_0fff + lsb) >> 13) as u16;
    sign | (((half_exp as u16) << 10) + rounded_mant)
}

/// Converts IEEE-754 binary16 (`fp16`) bits back to `f32` (`ref[container.proj.stmc-capsule]`).
#[must_use]
pub fn fp16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 0x001f);
    let mant = u32::from(bits & 0x03ff);
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let val = (mant as f32) * 2.0_f32.powi(-24);
        return if sign != 0 { -val } else { val };
    }
    if exp == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (mant << 13));
    }
    let f32_exp = (exp + 127 - 15) << 23;
    f32::from_bits(sign | f32_exp | (mant << 13))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_tier_parsing() {
        for (name, tier, dim) in [
            ("lite", Tier::Lite, 64),
            ("full", Tier::Full, 128),
            ("pro", Tier::Pro, 192),
            ("ultra", Tier::Ultra, 256),
        ] {
            let parsed = parse_tier_name(name).expect("tier parse should succeed");
            assert_eq!(parsed, tier);
            assert_eq!(solver::tier_feature_dimension(parsed), dim);
        }
        assert!(parse_tier_name("unknown").is_err());
    }

    #[test]
    fn test_gram_fingerprint_and_fp16_edge_cases() {
        let row0 = [1.0_f64, -0.5_f64];
        let row1 = [0.25_f64, 0.75_f64];
        let fp_a = compute_gram_input_fingerprint(2, &[(0, 1.0, &row0), (1, 0.5, &row1)]);
        let fp_b = compute_gram_input_fingerprint(2, &[(0, 1.0, &row0), (1, 0.5, &row1)]);
        let fp_c = compute_gram_input_fingerprint(2, &[(0, 1.0, &row0), (2, 0.5, &row1)]);
        assert_eq!(fp_a, fp_b);
        assert_ne!(fp_a, fp_c);

        assert_eq!(fp16_bits_to_f32(f32_to_fp16_bits(0.0)), 0.0);
        assert_eq!(
            fp16_bits_to_f32(f32_to_fp16_bits(-0.0)).to_bits(),
            (-0.0_f32).to_bits()
        );
        assert!(fp16_bits_to_f32(f32_to_fp16_bits(f32::INFINITY)).is_infinite());
        assert!(fp16_bits_to_f32(f32_to_fp16_bits(f32::NAN)).is_nan());
        assert!(fp16_bits_to_f32(f32_to_fp16_bits(1.0e6)).is_infinite());
        assert_eq!(f32_to_fp16_bits(2.0_f32.powi(-24)), 0x0001);
        assert_eq!(f32_to_fp16_bits(-2.0_f32.powi(-24)), 0x8001);
        assert_eq!(
            fp16_bits_to_f32(f32_to_fp16_bits(2.0_f32.powi(-20))),
            2.0_f32.powi(-20)
        );
        assert_eq!(f32_to_fp16_bits(2.0_f32.powi(-25)), 0x0000);
        assert_eq!(f32_to_fp16_bits(3.0 * 2.0_f32.powi(-25)), 0x0002);
        assert_eq!(f32_to_fp16_bits(1.0 + 2.0_f32.powi(-11)), 0x3c00);
        assert_eq!(f32_to_fp16_bits(1.0 + 3.0 * 2.0_f32.powi(-11)), 0x3c02);
        assert_eq!(fp16_bits_to_f32(0x0001), 2.0_f32.powi(-24));
        assert_eq!(fp16_bits_to_f32(0x8001), -2.0_f32.powi(-24));
    }

    #[test]
    #[should_panic(expected = "finite feature values")]
    fn test_gram_fingerprint_rejects_nan_feature() {
        let bad_row = [f64::NAN];
        let _ = compute_gram_input_fingerprint(1, &[(0, 1.0, &bad_row)]);
    }

    proptest! {
        #[test]
        fn prop_fp16_roundtrip_bounded_relative_error(x in -100.0_f32..100.0_f32) {
            let bits = f32_to_fp16_bits(x);
            let y = fp16_bits_to_f32(bits);
            if x.abs() >= 1.0e-3 {
                let rel_err = ((y - x) / x).abs();
                prop_assert!(rel_err <= 1.0e-3, "rel_err={} for x={}", rel_err, x);
            } else {
                prop_assert!((y - x).abs() <= 1.0e-3);
            }
        }
    }
}
