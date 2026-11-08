#![forbid(unsafe_code)]
//! `dnam-trainer` — Pure-Rust Analytical Variable-Projection (`VarPro`) Two-Sweep System-ID Trainer.
//!
//! Shares the exact `#![no_std]` + `alloc` DSP operators (`Op-T`, `Op-M`, `Op-N`),
//! `3 kHz` Parametric Control Bus, `uplo='U'` Gram matrix packer, and `.dnam`/`.dnam-ckpt`
//! container primitives with [`dnam_engine`] (`ref[arch.isomorphism.two-sweep]`,
//! `ref[arch.isomorphism.audit-wiring]`).

pub mod solver;

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
#[derive(Debug, Clone, Parser, PartialEq, Eq)]
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
#[derive(Debug, Clone, Subcommand, PartialEq, Eq)]
pub enum TrainerCommand {
    /// Inspect and validate trainer configuration and target slimmable tier.
    Check {
        /// Maximum slimmable tier (`lite`, `full`, `pro`, `ultra`).
        #[arg(long, default_value = "ultra")]
        max_tier: String,
    },
}

/// Summary returned by executing a `TrainerCli` invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliExecutionReport {
    /// Selected maximum slimmable tier.
    pub max_tier: Tier,
    /// Canonical feature-bus dimension for `max_tier`.
    pub feature_dim: usize,
    /// Packed `uplo='U'` Gram matrix element count `N * (N + 1) / 2`.
    pub packed_gram_len: usize,
}

/// Parses and executes a `dnam-train` CLI argument sequence.
pub fn run_cli_from_args<I, T>(args: I) -> Result<CliExecutionReport, String>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let cli = TrainerCli::try_parse_from(args).map_err(|err| err.to_string())?;
    let tier_str = match &cli.command {
        Some(TrainerCommand::Check { max_tier }) => max_tier.as_str(),
        None => "ultra",
    };
    let max_tier = parse_tier_name(tier_str)?;
    let feature_dim = solver::tier_feature_dimension(max_tier);
    let packed_gram_len = solver::uplo_u_packed_len(feature_dim);
    Ok(CliExecutionReport {
        max_tier,
        feature_dim,
        packed_gram_len,
    })
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
    let mut hasher = Sha256::new();
    hasher.update(b"DNAM-GRAM-INPUT-v1\0");
    hasher.update(feature_dim.to_le_bytes());
    hasher.update([2u8, 2u8]); // feature_type=2 (f64), weight_type=2 (f64)
    for (frame_idx, weight, row) in frames {
        hasher.update([1u8]);
        hasher.update(frame_idx.to_le_bytes());
        hasher.update(weight.to_le_bytes());
        for &val in *row {
            hasher.update(val.to_le_bytes());
        }
    }
    hasher.update([0u8]);
    hasher.update((frames.len() as u64).to_le_bytes());
    hasher.finalize().into()
}

/// Quantizes an `f32` scalar to IEEE-754 binary16 (`fp16`) bits and back for `DNCA` coreset storage
/// (`ref[container.dnca.coreset-fp16]`).
#[must_use]
pub fn f32_to_fp16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | u16::from(mant != 0);
    }
    let half_exp = exp - 127 + 15;
    if half_exp >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        return sign;
    }
    let rounded_mant = ((mant + 0x0000_1000) >> 13) as u16;
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
        assert_eq!(fp16_bits_to_f32(0x0001), 2.0_f32.powi(-24));
        assert_eq!(fp16_bits_to_f32(0x8001), -2.0_f32.powi(-24));
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
