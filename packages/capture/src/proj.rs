#![forbid(unsafe_code)]
//! Single-File Cloud-Ready `.dnam-proj` (`DNPJ`, `v2.0`-Only) Project & Capture Vault Container (`dnam_capture::proj`).
//!
//! Unifies capture projects, stem capsules, sufficient statistics, and runtime `.dnam` weights
//! inside a single `64 B`-aligned cloud-ready container featuring:
//! - Front-loaded `64 B` Preamble + `64 B * N` Chunk Directory (`crc32c` + `sha256`) + RFC 8785 JCS
//!   `DnamProjManifest` at byte `0` for `HTTP Range: bytes=0-65535` inspection (`parse_proj_manifest_range`).
//! - `CALB`, `SPCM`, `INPC` (`24,640 B` / `48,640 B`), `STMC` (`28,800 B` / `52,800 B`), `STAT`
//!   (`267,264 B` `StarCenter` `uplo='U'`), `WGHT` (`[STAB-ZERO-DC]` runtime `.dnam`), and `AUTH` chunks.
//! - `< 1 ms` `compact_to_capsules`, `< 0.2 ms` `export_runtime_dnam`, and Two-Path Capture Vault
//!   editing (`add_capture`, `remove_capture`, `replace_capture`, `set_capture_enabled`).
//!
//! Canonical Reference: `docs/specs/containers/project_container_format.md` (`ref[container.proj.preamble-and-directory]`).

pub mod bundle;
pub mod capsules;
pub mod directory;
pub mod schema;
pub mod vault;

pub use bundle::{
    DnamProjView, ProjChunkInput, attach_proj_auth_signature, build_dnam_proj_container,
    build_dnam_proj_container_with_cp1, build_dnam_proj_container_with_crypto, compact_to_capsules,
    export_runtime_dnam, export_runtime_dnam_with_crypto, verify_arena_b_head_zero_dc,
    zero_arena_b_head_dc_bias,
};
pub use directory::{
    CalbChunk64B, DnamProjPreamble, InpcCapsuleView, ParsedProjManifestRange, ProjAuthRecord,
    ProjChunkDirEntry, ProjChunkKind, StmcCapsuleView, StmcTrajectoryWindow,
    build_proj_auth_signing_message, build_proj_inpc_capsule, build_proj_stmc_capsule,
    compute_proj_header_crc32c, parse_proj_manifest_range, set_stmc_capsule_enabled_flag,
    set_stmc_capsule_star_center_flag, verify_proj_auth_chunk, verify_proj_inpc_capsule,
    verify_proj_stat_uplo_u, verify_proj_stmc_capsule,
};
pub use schema::{
    AuditReport, CalibrationSummary, DnamProjManifest, ExtractedPhysics, ProjectMeta,
    RigRoutingConfig, SolverTelemetrySummary, StemProvenance, StemRegistryEntry,
};
pub use vault::{
    VaultCaptureSpec, add_capture, add_capture_with_options,
    recompute_vault_phase1_audit_and_clamps, recompute_vault_phase1_audit_and_clamps_with_windows,
    remove_capture, remove_capture_with_options, replace_capture, replace_capture_with_options,
    set_capture_enabled, set_capture_enabled_with_options, solve_captures_batch_into_proj,
    solve_captures_batch_with_options,
};

/// Magic bytes for a `.dnam-proj` authoring project & Capture Vault container (`b"DNPJ"`).
pub const DNPJ_MAGIC: [u8; 4] = *b"DNPJ";

/// Magic bytes for an `INPC` dry stimulus capsule (`b"INPC"`).
pub const INPC_MAGIC: [u8; 4] = *b"INPC";

/// Magic bytes for an `STMC` `StemCaptureRecord` capsule (`b"STMC"`).
pub const STMC_MAGIC: [u8; 4] = *b"STMC";

/// Canonical `.dnam-proj` major wire version (`2`).
pub const PROJ_VERSION_MAJOR: u16 = 2;

/// Canonical `.dnam-proj` minor wire version (`0`).
pub const PROJ_VERSION_MINOR: u16 = 0;

/// Byte length of Region 0 [`DnamProjPreamble`] (`64 B`).
pub const PROJ_PREAMBLE_SIZE: usize = 64;

/// Maximum chunk directory entries permitted in Region 1 (`4,096`).
pub const MAX_PROJ_CHUNKS: usize = 4_096;

/// Byte length of the `AUTH` `Ed25519` creator attestation chunk (`128 B` = `2 × 64 B`).
pub const PROJ_AUTH_RECORD_BYTES: usize = 128;

/// Byte length of a mono `INPC` dry stimulus capsule (`385 × 64 B = 24,640 B`).
pub const INPC_MONO_BYTES: usize = 24_640;

/// Byte length of a stereo `INPC` dry stimulus capsule (`760 × 64 B = 48,640 B`).
pub const INPC_STEREO_BYTES: usize = 48_640;

/// Byte length of a mono `STMC` `StemCaptureRecord` capsule (`450 × 64 B = 28,800 B`).
pub const STMC_MONO_BYTES: usize = 28_800;

/// Byte length of a stereo `STMC` `StemCaptureRecord` capsule (`825 × 64 B = 52,800 B`).
pub const STMC_STEREO_BYTES: usize = 52_800;

/// Maximum trajectory windows in an `STMC` capsule (`36 × 16 B = 576 B`).
pub const STMC_MAX_TRAJECTORY_WINDOWS: usize = 36;

// ============================================================================
// DnamProjPreamble `lifecycle_flags` (`u32`)
// ============================================================================

/// Bit 0: `CALB` loopback calibration chunk is present.
pub const PROJ_FLAG_HAS_CALIBRATION: u32 = 1 << 0;
/// Bit 1: Optional `SPCM` raw aligned stem audio chunks are present.
pub const PROJ_FLAG_HAS_RAW_STEMS: u32 = 1 << 1;
/// Bit 2: `INPC` + `STMC` non-destructive Capture Vault capsules are present.
pub const PROJ_FLAG_HAS_CAPSULES: u32 = 1 << 2;
/// Bit 3: `STAT` `StarCenter` `uplo='U'` sufficient-statistics chunk is present.
pub const PROJ_FLAG_HAS_STAT: u32 = 1 << 3;
/// Bit 4: `WGHT` solved runtime `.dnam` weights chunk is present.
pub const PROJ_FLAG_HAS_WGHT: u32 = 1 << 4;
/// Bit 5: `AUTH` `Ed25519` creator provenance chunk is present.
pub const PROJ_FLAG_HAS_AUTH: u32 = 1 << 5;
/// Bit 6: Project output head is configured for stereo (`base.io_config.output_mode == "stereo"`).
pub const PROJ_FLAG_STEREO_OUT: u32 = 1 << 6;
/// Bit 7: Project was compacted via [`compact_to_capsules`] (`0` `SPCM` chunks retained).
pub const PROJ_FLAG_COMPACTED_CAPSULES: u32 = 1 << 7;

/// Bitmask of all defined `PROJ_FLAG_*` bits in `v2.0`.
pub const PROJ_FLAG_KNOWN_MASK: u32 = PROJ_FLAG_HAS_CALIBRATION
    | PROJ_FLAG_HAS_RAW_STEMS
    | PROJ_FLAG_HAS_CAPSULES
    | PROJ_FLAG_HAS_STAT
    | PROJ_FLAG_HAS_WGHT
    | PROJ_FLAG_HAS_AUTH
    | PROJ_FLAG_STEREO_OUT
    | PROJ_FLAG_COMPACTED_CAPSULES;

// ============================================================================
// STMC Capsule `flags` (`u16`)
// ============================================================================

/// Bit 0: Stem is enabled (`true` = included in active solve and 3-Phase audit).
pub const STMC_FLAG_ENABLED: u16 = 1 << 0;
/// Bit 1: Stem is the `StarCenter` (`s = 0`) anchor backing `STAT`.
pub const STMC_FLAG_STAR_CENTER: u16 = 1 << 1;
/// Bit 2: Stem is a time-varying Mode A continuous sweep (`R > 1` windows).
pub const STMC_FLAG_MODE_A_SWEEP: u16 = 1 << 2;
/// Bit 3: Stem wet coreset has stereo (`2`) channels (`48,000 B`).
pub const STMC_FLAG_STEREO: u16 = 1 << 3;
/// Bit 4: Wet return was polarity-inverted prior to alignment normalization.
pub const STMC_FLAG_WET_POLARITY_INVERTED: u16 = 1 << 4;

/// Bitmask of all defined `STMC_FLAG_*` bits in `v2.0`.
pub const STMC_FLAG_KNOWN_MASK: u16 = STMC_FLAG_ENABLED
    | STMC_FLAG_STAR_CENTER
    | STMC_FLAG_MODE_A_SWEEP
    | STMC_FLAG_STEREO
    | STMC_FLAG_WET_POLARITY_INVERTED;

// ============================================================================
// `STAT` `StarCenter` `uplo='U'` Byte Lengths (`r[container.proj.stat-uplo-u]`)
// ============================================================================

/// `Lite` (`N = 64`): `2,080 * 8 B` (`16,640 B`) `G_0` + `1,024 B` stereo `C_0` = `17,664 B` (`276 × 64 B`).
pub const PROJ_STAT_BYTES_LITE: usize = 17_664;

/// `Full` (`N = 128`): `8,256 * 8 B` (`66,048 B`) `G_0` + `2,048 B` stereo `C_0` = `68,096 B` (`1,064 × 64 B`).
pub const PROJ_STAT_BYTES_FULL: usize = 68_096;

/// `Pro` (`N = 192`): `18,528 * 8 B` (`148,224 B`) `G_0` + `3,072 B` stereo `C_0` = `151,296 B` (`2,364 × 64 B`).
pub const PROJ_STAT_BYTES_PRO: usize = 151_296;

/// `Ultra` (`N = 256`): `32,896 * 8 B` (`263,168 B`) `G_0` + `4,096 B` stereo `C_0` = `267,264 B` (`4,176 × 64 B`).
pub const PROJ_STAT_BYTES_ULTRA: usize = 267_264;

/// Returns the packed upper-triangular `uplo='U'` element count $N(N + 1)/2$ (`r[container.proj.stat-uplo-u]`).
#[inline]
#[must_use]
pub const fn uplo_u_packed_len(n_dim: usize) -> usize {
    crate::solver::uplo_u_packed_len(n_dim)
}

/// Packs a dense row-major symmetric `n_dim × n_dim` `f64` matrix into column-major `uplo='U'`
/// upper-triangular order so every tier prefix $N_T \in \{64, 128, 192, 256\}$ is contiguous
/// (`r[container.proj.stat-uplo-u]`).
pub fn pack_uplo_u_f64(dense_flat: &[f64], n_dim: usize, out_packed: &mut [f64]) {
    assert_eq!(dense_flat.len(), n_dim.saturating_mul(n_dim));
    let req = uplo_u_packed_len(n_dim);
    assert!(out_packed.len() >= req);
    let mut k = 0usize;
    for j in 0..n_dim {
        for i in 0..=j {
            out_packed[k] = 0.5 * (dense_flat[i * n_dim + j] + dense_flat[j * n_dim + i]);
            k += 1;
        }
    }
}

/// Unpacks the leading `n_tier × n_tier` principal submatrix from a column-major `uplo='U'`
/// packed buffer into a dense symmetric row-major `n_tier × n_tier` `f64` slice
/// (`r[container.proj.stat-uplo-u]`).
pub fn unpack_uplo_u_f64(packed_u: &[f64], n_tier: usize, out_dense: &mut [f64]) {
    let req = uplo_u_packed_len(n_tier);
    assert!(packed_u.len() >= req);
    assert_eq!(out_dense.len(), n_tier.saturating_mul(n_tier));
    let mut k = 0usize;
    for j in 0..n_tier {
        for i in 0..=j {
            let v = packed_u[k];
            out_dense[i * n_tier + j] = v;
            out_dense[j * n_tier + i] = v;
            k += 1;
        }
    }
}

#[inline]
fn f32_to_fp16_rne_bits(val: f32) -> u16 {
    if !val.is_finite() {
        return 0;
    }
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;
    if exp == 0 {
        return sign;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        // Clamp finite overflow to max finite binary16 (0x7BFF / 0xFBFF) so coresets stay strictly finite
        return sign | 0x7BFF;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign;
        }
        let full_mant = mant | 0x0080_0000;
        let shift = (14 - new_exp) as u32;
        let half_bit = 1u32 << (shift - 1);
        let mask = half_bit - 1;
        let mut sub = (full_mant >> shift) as u16;
        let rem = full_mant & (half_bit | mask);
        if rem > half_bit || (rem == half_bit && (sub & 1) != 0) {
            sub += 1;
        }
        return sign | sub;
    }
    let round_bit = 1u32 << 12;
    let mask = round_bit - 1;
    let mut half_mant = (mant >> 13) as u16;
    let rem = mant & (round_bit | mask);
    let mut half_exp = (new_exp as u16) << 10;
    if rem > round_bit || (rem == round_bit && (half_mant & 1) != 0) {
        half_mant += 1;
        if half_mant >= 0x0400 {
            half_mant = 0;
            half_exp += 0x0400;
            if half_exp >= 0x7C00 {
                return sign | 0x7BFF;
            }
        }
    }
    sign | half_exp | half_mant
}

#[inline]
pub(crate) fn fp16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 0x1F);
    let mant = u32::from(bits & 0x03FF);
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let val = (mant as f32) * (1.0 / 16_777_216.0);
        return if sign != 0 { -val } else { val };
    }
    if exp == 31 {
        return 0.0;
    }
    let f32_bits = sign | ((exp + 112) << 23) | (mant << 13);
    f32::from_bits(f32_bits)
}

/// Encodes `f32` samples into little-endian IEEE-754 round-to-nearest-even (`RNE`) `fp16` bytes
/// (clamping non-finite or out-of-range inputs to finite `binary16`).
#[must_use]
pub fn f32_to_fp16_rne_bytes(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        out.extend_from_slice(&f32_to_fp16_rne_bits(s).to_le_bytes());
    }
    out
}

/// Decodes little-endian IEEE-754 `fp16` bytes into `f32` samples.
#[must_use]
pub fn fp16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| fp16_bits_to_f32(u16::from_le_bytes(*pair)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_stat_byte_sizes_and_fp16_roundtrip() {
        assert_eq!(uplo_u_packed_len(64) * 8 + 1_024, PROJ_STAT_BYTES_LITE);
        assert_eq!(uplo_u_packed_len(128) * 8 + 2_048, PROJ_STAT_BYTES_FULL);
        assert_eq!(uplo_u_packed_len(192) * 8 + 3_072, PROJ_STAT_BYTES_PRO);
        assert_eq!(uplo_u_packed_len(256) * 8 + 4_096, PROJ_STAT_BYTES_ULTRA);
        assert_eq!(PROJ_STAT_BYTES_ULTRA % 64, 0);

        let vals = [0.0_f32, -0.0, 1.0, -1.0, 0.25, 1.0e-6, 99_999.0, f32::NAN];
        let enc = f32_to_fp16_rne_bytes(&vals);
        let dec = fp16_bytes_to_f32(&enc);
        assert_eq!(dec[0], 0.0);
        assert_eq!(dec[2], 1.0);
        assert_eq!(dec[3], -1.0);
        assert_eq!(dec[4], 0.25);
        assert!(dec[6].is_finite() && dec[6] > 65_000.0);
        assert_eq!(dec[7], 0.0);
    }

    proptest! {
        #[test]
        fn prop_uplo_u_tier_prefix_submatrix_exactness(seed in 0u64..1000u64) {
            let n = 16usize;
            let mut dense = vec![0.0_f64; n * n];
            for i in 0..n {
                for j in i..n {
                    let v = (((i * 31 + j * 17) as u64 ^ seed) as f64) * 0.01 - 0.5;
                    dense[i * n + j] = v;
                    dense[j * n + i] = v;
                }
            }
            let mut packed = vec![0.0_f64; uplo_u_packed_len(n)];
            pack_uplo_u_f64(&dense, n, &mut packed);

            // Unpack leading 8x8 submatrix directly from prefix
            let n_sub = 8usize;
            let mut sub_dense = vec![0.0_f64; n_sub * n_sub];
            unpack_uplo_u_f64(&packed[..uplo_u_packed_len(n_sub)], n_sub, &mut sub_dense);
            for i in 0..n_sub {
                for j in 0..n_sub {
                    prop_assert!((sub_dense[i * n_sub + j] - dense[i * n + j]).abs() <= 1.0e-15);
                }
            }
        }
    }
}
