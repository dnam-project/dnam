#![forbid(unsafe_code)]
//! `dnam-trainer` — Pure-Rust Analytical Variable-Projection (`VarPro`) Two-Sweep System-ID Trainer.
//!
//! Shares the exact `#![no_std]` + `alloc` DSP operators (`Op-T`, `Op-M`, `Op-N`),
//! `3 kHz` Parametric Control Bus, `uplo='U'` Gram matrix packer, and `.dnam`/`.dnam-ckpt`
//! container primitives with [`dnam_engine`] (`ref[arch.isomorphism.two-sweep]`,
//! `ref[arch.isomorphism.audit-wiring]`).

pub mod audit;
pub mod ckpt;
pub mod metrics;
pub mod resampler;
pub mod solver;
pub mod targets;
#[cfg(not(target_arch = "wasm32"))]
pub mod whitebox;

pub use resampler::{
    CANONICAL_TRAINER_SAMPLE_RATE_HZ, resample_48k_to_arbitrary_rate, resample_arbitrary_rate,
    resample_arbitrary_rate_to_48k, resample_control_trajectory, resampled_frame_len,
    scale_freeze_window,
};
pub use targets::{
    CaptureBitDepth, DistillationTarget, StemResponse, StemStimulus, StemStimulusBatch,
    StimulusFrame,
};

#[cfg(feature = "cli")]
use clap::{Parser, Subcommand};
use dnam_engine::Tier;
use sha2::{Digest, Sha256};

/// Canonical reference sample rate (`48,000 Hz`) for Two-Sweep system identification.
pub const REFERENCE_SAMPLE_RATE_HZ: u32 = 48_000;

/// Canonical `250 ms` `fp16` coreset window length (`12,000` frames at `48 kHz`, `ref[container.dnca.coreset-fp16]`).
pub const CORESET_FP16_FRAMES_PER_STEM: usize = 12_000;

/// Canonical mono (`C = 1`) `fp16` coreset byte size (`12,000 * 1 * 2 B = 24,000 B`, `64 B`-aligned).
pub const CORESET_FP16_MONO_BYTES: usize = CORESET_FP16_FRAMES_PER_STEM * 2;

/// Canonical stereo (`C = 2`) `fp16` coreset byte size (`12,000 * 2 * 2 B = 48,000 B`, `64 B`-aligned).
pub const CORESET_FP16_STEREO_BYTES: usize = CORESET_FP16_FRAMES_PER_STEM * 4;

/// Command-line interface for the `dnam-train` analytical Two-Sweep system-identification binary.
#[cfg(feature = "cli")]
#[derive(Debug, Clone, Parser, PartialEq)]
#[command(
    name = "dnam-train",
    version,
    about = "Pure-Rust Analytical Variable-Projection (VarPro) Two-Sweep System-ID Trainer for DNAM"
)]
pub struct TrainerCli {
    /// Subcommand to execute.
    #[command(subcommand)]
    pub command: Option<TrainerCommand>,
}

/// Supported subcommands for `dnam-train`.
#[cfg(feature = "cli")]
#[derive(Debug, Clone, Subcommand, PartialEq, Eq)]
pub enum TrainerCommand {
    /// Inspect and validate trainer configuration and target slimmable tier.
    Check {
        /// Maximum slimmable tier (`lite`, `full`, `pro`, `ultra`).
        #[arg(long, default_value = "ultra")]
        max_tier: String,
    },
    /// Serve a persistent stream of out-of-process `64 B` `DNWB` white-box requests over `stdin`/`stdout`.
    DnwbServe {
        /// Worker engine (`spice` or `faust`).
        #[arg(long, default_value = "faust")]
        runner: String,
    },
}

/// Summary returned by executing a `TrainerCli` invocation.
#[cfg_attr(not(feature = "cli"), allow(dead_code))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliExecutionReport {
    /// Selected maximum slimmable tier.
    pub max_tier: Tier,
    /// Canonical feature-bus dimension for `max_tier`.
    pub feature_dim: usize,
    /// Packed `uplo='U'` Gram matrix element count `N * (N + 1) / 2`.
    pub packed_gram_len: usize,
    /// Whether `stdout` was already consumed by a binary `DNWB` pipe response.
    pub binary_pipe_mode: bool,
}

/// Parses and executes a `dnam-train` CLI argument sequence.
#[cfg(feature = "cli")]
pub fn run_cli_from_args<I, T>(args: I) -> Result<CliExecutionReport, String>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let cli = TrainerCli::try_parse_from(args).map_err(|err| err.to_string())?;
    run_cli_parsed(cli)
}

#[cfg(feature = "cli")]
fn run_cli_parsed(cli: TrainerCli) -> Result<CliExecutionReport, String> {
    let mut binary_pipe_mode = false;
    let tier_str = match &cli.command {
        Some(TrainerCommand::Check { max_tier }) => max_tier.as_str(),
        #[cfg(not(target_arch = "wasm32"))]
        Some(TrainerCommand::DnwbServe { runner }) => {
            binary_pipe_mode = true;
            let mut stdin = std::io::stdin().lock();
            let mut stdout = std::io::stdout().lock();
            run_dnwb_serve_io(runner, &mut stdin, &mut stdout)?
        }
        #[cfg(target_arch = "wasm32")]
        Some(TrainerCommand::DnwbServe { .. }) => {
            return Err(String::from("dnwb-serve is not supported on wasm32"));
        }
        None => "ultra",
    };
    let max_tier = parse_tier_name(tier_str)?;
    let feature_dim = solver::tier_feature_dimension(max_tier);
    let packed_gram_len = solver::uplo_u_packed_len(feature_dim);
    Ok(CliExecutionReport {
        max_tier,
        feature_dim,
        packed_gram_len,
        binary_pipe_mode,
    })
}

#[cfg(all(feature = "cli", not(target_arch = "wasm32")))]
fn run_dnwb_serve_io<R: std::io::Read, W: std::io::Write>(
    runner: &str,
    reader: &mut R,
    writer: &mut W,
) -> Result<&'static str, String> {
    match runner.trim().to_ascii_lowercase().as_str() {
        "spice" => {
            let mut target = whitebox::SpiceRunner::new("cli_worker.cir", 5_000);
            whitebox::serve_dnwb_stream_with_switches(reader, writer, &mut target, |t, sw| {
                t.set_switches(sw)
            })
            .map_err(|e| format!("{e:?}"))?;
        }
        "faust" => {
            let json = r#"{"ui":[{"type":"vgroup","label":"W","items":[{"type":"hslider","label":"Drive","init":0.5,"min":0.0,"max":1.0,"step":0.01},{"type":"hslider","label":"Tone","init":0.5,"min":0.0,"max":1.0,"step":0.01}]}]}"#;
            let mut target = whitebox::FaustRunner::from_faust_json("cli_worker.dsp", json, 5_000)
                .map_err(|e| format!("{e:?}"))?;
            whitebox::serve_dnwb_stream_with_switches(reader, writer, &mut target, |t, sw| {
                t.set_switches(sw)
            })
            .map_err(|e| format!("{e:?}"))?;
        }
        other => {
            return Err(format!(
                "unknown dnwb-serve runner '{other}' (expected 'spice' or 'faust')"
            ));
        }
    }
    Ok("ultra")
}

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

/// Computes the deterministic `DNAM-GRAM-INPUT-v1` SHA-256 fingerprint over ordered frames
/// (`ref[container.ckpt.gram-fingerprint]`).
#[must_use]
pub fn compute_gram_input_fingerprint(feature_dim: u32, frames: &[(u64, f64, &[f64])]) -> [u8; 32] {
    let expected_len = feature_dim as usize;
    let mut prev_frame_idx: Option<u64> = None;
    let mut hasher = Sha256::new();
    hasher.update(b"DNAM-GRAM-INPUT-v1\0");
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
/// and full 10-bit subnormal encoding (`ref[container.dnca.coreset-fp16]`).
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

/// Converts IEEE-754 binary16 (`fp16`) bits back to `f32` (`ref[container.dnca.coreset-fp16]`).
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

    #[cfg(feature = "cli")]
    #[test]
    fn test_cli_execution_and_tier_parsing() {
        let r_default = run_cli_from_args(["dnam-train"]).expect("default CLI run should succeed");
        assert_eq!(r_default.max_tier, Tier::Ultra);
        assert_eq!(r_default.feature_dim, 256);
        assert_eq!(r_default.packed_gram_len, 256 * 257 / 2);

        for (name, tier, dim) in [
            ("lite", Tier::Lite, 64),
            ("full", Tier::Full, 128),
            ("pro", Tier::Pro, 192),
            ("ultra", Tier::Ultra, 256),
        ] {
            let rep = run_cli_from_args(["dnam-train", "check", "--max-tier", name])
                .expect("tier check should succeed");
            assert_eq!(rep.max_tier, tier);
            assert_eq!(rep.feature_dim, dim);
        }

        assert!(parse_tier_name("unknown").is_err());
        assert!(run_cli_from_args(["dnam-train", "--invalid-flag"]).is_err());
        assert!(run_cli_from_args(["dnam-train", "dnwb-serve", "--runner", "bogus"]).is_err());

        #[cfg(all(feature = "cli", not(target_arch = "wasm32")))]
        {
            let stim = StemStimulus {
                stem_id: String::from("cli_io"),
                sample_rate_hz: 48_000,
                in_1: vec![0.1, -0.1],
                in_2: None,
                in_3: None,
                in_4: None,
                control_values: vec![0.5, 0.5],
                control_trajectories: None,
                freeze_transition_window: None,
            };
            let mut req = Vec::new();
            whitebox::write_dnwb_stimulus_request_with_switches(
                &mut req,
                whitebox::DnwbCommand::RenderStem,
                &stim,
                &[1],
            )
            .unwrap();
            for r_kind in ["spice", "faust"] {
                let mut out = Vec::new();
                assert_eq!(
                    run_dnwb_serve_io(r_kind, &mut std::io::Cursor::new(&req), &mut out).unwrap(),
                    "ultra"
                );
                assert!(!out.is_empty());
            }
        }
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
        // Subnormal encoding & round-to-nearest-even (RNE) verification
        assert_eq!(f32_to_fp16_bits(2.0_f32.powi(-24)), 0x0001);
        assert_eq!(f32_to_fp16_bits(-2.0_f32.powi(-24)), 0x8001);
        assert_eq!(
            fp16_bits_to_f32(f32_to_fp16_bits(2.0_f32.powi(-20))),
            2.0_f32.powi(-20)
        );
        assert_eq!(f32_to_fp16_bits(2.0_f32.powi(-25)), 0x0000); // exact halfway -> even 0
        assert_eq!(f32_to_fp16_bits(3.0 * 2.0_f32.powi(-25)), 0x0002); // exact halfway -> even 2
        // Normal RNE ties: 1.0 + 2^-11 rounds to even 1.0 (0x3c00); 1.0 + 3 * 2^-11 rounds to even 1.0 + 2^-9 (0x3c02)
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
