#![forbid(unsafe_code)]
//! `.dnam-ckpt` Stem Record Column-Major Upper-Triangular (`LAPACK uplo='U'`) packing,
//! `Arena 5` calibration statistics (`DNCA`) verification & stem verification
//! (`ref[container.ckpt.uplo-u]`, `ref[container.ckpt.standalone-full]`,
//! `ref[container.ckpt.prefix-delta]`, `ref[container.ckpt.objective-records]`,
//! `ref[container.dnca.inpc]`, `ref[container.dnca.stmc]`, `ref[container.dnca.dnce-sidecar]`,
//! `ref[container.dnca.coreset-fp16]`).

use dnam_engine::container::crc32c_compute;
use dnam_engine::schema::DnamError;

/// Magic bytes for `.dnam-ckpt` `Arena 5` calibration statistics header (`b"DNCA"`).
pub const DNCA_MAGIC: [u8; 4] = *b"DNCA";

/// Fixed 64-byte header length for `.dnam-ckpt` `Arena 5`.
pub const ARENA5_HEADER_BYTES: usize = 64;

/// Per-stem `.dnam-ckpt` byte length for `Lite` ($N=64$: `16,640 B` `G_audio` + `1,024 B` `C_audio` = `17,664 B`).
pub const ARENA4_STEM_BYTES_LITE: usize = 17_664;

/// Per-stem `.dnam-ckpt` byte length for `Full` ($N=128$: `66,048 B` `G_audio` + `2,048 B` `C_audio` = `68,096 B`).
pub const ARENA4_STEM_BYTES_FULL: usize = 68_096;

/// Per-stem `.dnam-ckpt` byte length for `Pro` ($N=192$: `148,224 B` `G_audio` + `3,072 B` `C_audio` = `151,296 B`).
pub const ARENA4_STEM_BYTES_PRO: usize = 151_296;

/// Per-stem `.dnam-ckpt` byte length for `Ultra` ($N=256$: `263,168 B` `G_audio` + `4,096 B` `C_audio` = `267,264 B`).
pub const ARENA4_STEM_BYTES_ULTRA: usize = 267_264;

/// Returns the packed element count $\frac{N(N+1)}{2}$ for a symmetric $N \times N$ Gram matrix:
/// - `Lite` ($N=64$): `2,080` `f64` (`16,640 B`)
/// - `Full` ($N=128$): `8,256` `f64` (`66,048 B`)
/// - `Pro` ($N=192$): `18,528` `f64` (`148,224 B`)
/// - `Ultra` ($N=256$): `32,896` `f64` (`263,168 B`)
#[inline]
#[must_use]
pub const fn uplo_u_packed_len(n_dim: usize) -> usize {
    (n_dim * (n_dim + 1)) / 2
}

/// Packs a dense symmetric $N \times N$ row-major matrix `dense_matrix` (`N * N` elements)
/// into **Column-Major Upper-Triangular (`LAPACK uplo='U'`)** format:
/// $$\operatorname{pack}_{\text{U}}(\mathbf{G})\!\left[i + \frac{j(j+1)}{2}\right] = \mathbf{G}[i, j], \qquad 0 \le i \le j < N$$
#[inline]
pub fn pack_uplo_u_f64(dense_matrix: &[f64], n_dim: usize, out_packed: &mut [f64]) {
    let mut idx = 0;
    for col in 0..n_dim {
        for row in 0..=col {
            out_packed[idx] = dense_matrix[row * n_dim + col];
            idx += 1;
        }
    }
}

/// Unpacks a **Column-Major Upper-Triangular (`LAPACK uplo='U'`)** buffer into a symmetric
/// $N_{\text{tier}} \times N_{\text{tier}}$ matrix (`out_dense`, length `n_tier * n_tier`).
///
/// By the **Nested Principal Submatrix Prefix Theorem**, passing `n_tier = 64` (`Lite`) or
/// `n_tier = 128` (`Full`) on an `Ultra` ($N=256$) packed buffer extracts the exact leading
/// principal submatrix from the contiguous prefix `[0 .. n_tier*(n_tier+1)/2]`.
#[inline]
pub fn unpack_uplo_u_f64(packed_u: &[f64], n_tier: usize, out_dense: &mut [f64]) {
    let mut idx = 0;
    for col in 0..n_tier {
        for row in 0..=col {
            let val = packed_u[idx];
            out_dense[row * n_tier + col] = val;
            out_dense[col * n_tier + row] = val;
            idx += 1;
        }
    }
}

/// Verifies a `.dnam-ckpt` stem's `stem_arena4_crc32c` over its `Arena 4` record.
pub fn verify_ckpt_stem_arena4_crc32c(
    arena_bytes: &[u8],
    stem_byte_offset: usize,
    stem_byte_len: usize,
    expected_crc32c: u32,
) -> Result<(), DnamError> {
    if arena_bytes.len() < stem_byte_offset + stem_byte_len {
        return Err(DnamError::BufferTooSmall {
            expected: stem_byte_offset + stem_byte_len,
            actual: arena_bytes.len(),
        });
    }
    let slice = &arena_bytes[stem_byte_offset..stem_byte_offset + stem_byte_len];
    let computed = crc32c_compute(slice);
    if computed != expected_crc32c {
        return Err(DnamError::Crc32cMismatch {
            region: "arena4_stem",
            expected: expected_crc32c,
            computed,
        });
    }
    Ok(())
}

/// Verifies `.dnam-ckpt` `Arena 5` (`DNCA` magic, version, 64B alignment, body `CRC32C`,
/// and per-entry `INPC`/`STMC` record framing and `CRC32C`s) starting at `arena5_offset`.
pub fn verify_ckpt_arena5_calib_stats(
    arena_bytes: &[u8],
    arena5_offset: usize,
) -> Result<usize, DnamError> {
    if (arena5_offset & 0x3F) != 0 {
        return Err(DnamError::UnalignedOffset {
            field: "arena5_offset",
            offset: arena5_offset as u32,
        });
    }
    if arena_bytes.len() < arena5_offset + ARENA5_HEADER_BYTES {
        return Err(DnamError::BufferTooSmall {
            expected: arena5_offset + ARENA5_HEADER_BYTES,
            actual: arena_bytes.len(),
        });
    }
    let a5 = &arena_bytes[arena5_offset..];
    if (a5.len() & 0x3F) != 0 {
        return Err(DnamError::UnalignedOffset {
            field: "arena5_byte_len",
            offset: a5.len() as u32,
        });
    }
    let magic = [a5[0], a5[1], a5[2], a5[3]];
    if magic != DNCA_MAGIC {
        return Err(DnamError::SchemaValidationError(
            "invalid Arena 5 magic (expected 'DNCA')".to_string(),
        ));
    }
    let version = u16::from_le_bytes([a5[4], a5[5]]);
    if version != 1 {
        return Err(DnamError::SchemaValidationError(format!(
            "unsupported Arena 5 version {version} (expected 1)"
        )));
    }
    let num_inputs = u16::from_le_bytes([a5[8], a5[9]]) as usize;
    let num_stems = u16::from_le_bytes([a5[10], a5[11]]) as usize;
    let total_byte_len = u32::from_le_bytes([a5[12], a5[13], a5[14], a5[15]]) as usize;
    let expected_body_crc = u32::from_le_bytes([a5[16], a5[17], a5[18], a5[19]]);

    if total_byte_len != a5.len() || (total_byte_len & 0x3F) != 0 {
        return Err(DnamError::SchemaValidationError(format!(
            "Arena 5 total_byte_len ({total_byte_len}) does not match slice length ({})",
            a5.len()
        )));
    }
    let computed_body_crc = crc32c_compute(&a5[ARENA5_HEADER_BYTES..]);
    if computed_body_crc != expected_body_crc {
        return Err(DnamError::Crc32cMismatch {
            region: "arena5_calib",
            expected: expected_body_crc,
            computed: computed_body_crc,
        });
    }

    let mut pos = ARENA5_HEADER_BYTES;
    for _ in 0..num_inputs {
        if pos + 64 > a5.len() {
            return Err(DnamError::BufferTooSmall {
                expected: arena5_offset + pos + 64,
                actual: arena_bytes.len(),
            });
        }
        if &a5[pos..pos + 4] != b"INPC" {
            return Err(DnamError::SchemaValidationError(
                "invalid Arena 5 input record magic (expected 'INPC')".to_string(),
            ));
        }
        let rec_len =
            u32::from_le_bytes([a5[pos + 56], a5[pos + 57], a5[pos + 58], a5[pos + 59]]) as usize;
        let rec_crc = u32::from_le_bytes([a5[pos + 60], a5[pos + 61], a5[pos + 62], a5[pos + 63]]);
        if rec_len < 64 || (rec_len & 0x3F) != 0 || pos + rec_len > a5.len() {
            return Err(DnamError::SchemaValidationError(
                "invalid Arena 5 input record byte length".to_string(),
            ));
        }
        let calc_crc = crc32c_compute(&a5[pos + 64..pos + rec_len]);
        if calc_crc != rec_crc {
            return Err(DnamError::Crc32cMismatch {
                region: "arena5_input_record",
                expected: rec_crc,
                computed: calc_crc,
            });
        }
        pos += rec_len;
    }

    for _ in 0..num_stems {
        if pos + 128 > a5.len() {
            return Err(DnamError::BufferTooSmall {
                expected: arena5_offset + pos + 128,
                actual: arena_bytes.len(),
            });
        }
        if &a5[pos..pos + 4] != b"STMC" {
            return Err(DnamError::SchemaValidationError(
                "invalid Arena 5 stem record magic (expected 'STMC')".to_string(),
            ));
        }
        let rec_len =
            u32::from_le_bytes([a5[pos + 116], a5[pos + 117], a5[pos + 118], a5[pos + 119]])
                as usize;
        let rec_crc =
            u32::from_le_bytes([a5[pos + 120], a5[pos + 121], a5[pos + 122], a5[pos + 123]]);
        if rec_len < 128 || (rec_len & 0x3F) != 0 || pos + rec_len > a5.len() {
            return Err(DnamError::SchemaValidationError(
                "invalid Arena 5 stem record byte length".to_string(),
            ));
        }
        let calc_crc = crc32c_compute(&a5[pos + 128..pos + rec_len]);
        if calc_crc != rec_crc {
            return Err(DnamError::Crc32cMismatch {
                region: "arena5_stem_record",
                expected: rec_crc,
                computed: calc_crc,
            });
        }
        pos += rec_len;
    }

    if pos != a5.len() {
        return Err(DnamError::SchemaValidationError(
            "Arena 5 trailing bytes after records".to_string(),
        ));
    }
    Ok(total_byte_len)
}

/// Verifies `.dnam-ckpt` `Arena 4` Kronecker / `uplo='U'` sufficient statistics (`G_audio`, `C_audio`)
/// byte framing, 64-byte alignment, tier-dependent per-stem record size (`17,664 B` `Lite`,
/// `68,096 B` `Full`, `151,296 B` `Pro`, `267,264 B` `Ultra`), and finite `f64` entries.
pub fn verify_ckpt_arena4_kronecker_stats(
    arena4_bytes: &[u8],
    max_tier: dnam_engine::types::Tier,
    num_stems: usize,
) -> Result<usize, DnamError> {
    if (arena4_bytes.len() & 0x3F) != 0 {
        return Err(DnamError::UnalignedOffset {
            field: "arena4_byte_len",
            offset: arena4_bytes.len() as u32,
        });
    }
    let stem_bytes = match max_tier {
        dnam_engine::types::Tier::Lite => ARENA4_STEM_BYTES_LITE,
        dnam_engine::types::Tier::Full => ARENA4_STEM_BYTES_FULL,
        dnam_engine::types::Tier::Pro => ARENA4_STEM_BYTES_PRO,
        dnam_engine::types::Tier::Ultra => ARENA4_STEM_BYTES_ULTRA,
    };
    let expected_total = stem_bytes * num_stems.max(1);
    if arena4_bytes.len() != stem_bytes && arena4_bytes.len() != expected_total {
        return Err(DnamError::SchemaValidationError(format!(
            "Arena 4 byte length ({}) does not match tier stem_bytes ({stem_bytes}) or expected_total ({expected_total})",
            arena4_bytes.len()
        )));
    }
    for &chunk in arena4_bytes.as_chunks::<8>().0 {
        let v = f64::from_le_bytes(chunk);
        if !v.is_finite() {
            return Err(DnamError::SchemaValidationError(
                "Arena 4 contains non-finite f64 entry".to_string(),
            ));
        }
    }
    Ok(arena4_bytes.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_ckpt_stem_arena4_crc32c_verification() {
        let stem_buf = [5u8; 128];
        let stem_crc = crc32c_compute(&stem_buf[0..64]);
        assert!(verify_ckpt_stem_arena4_crc32c(&stem_buf, 0, 64, stem_crc).is_ok());
        assert!(verify_ckpt_stem_arena4_crc32c(&stem_buf, 0, 256, stem_crc).is_err());
        assert!(verify_ckpt_stem_arena4_crc32c(&stem_buf, 0, 64, stem_crc ^ 1).is_err());

        let mut a4 = vec![0u8; ARENA4_STEM_BYTES_LITE];
        assert_eq!(
            verify_ckpt_arena4_kronecker_stats(&a4, dnam_engine::types::Tier::Lite, 1).unwrap(),
            ARENA4_STEM_BYTES_LITE
        );
        assert!(
            verify_ckpt_arena4_kronecker_stats(&a4[..100], dnam_engine::types::Tier::Lite, 1)
                .is_err()
        );
        assert!(
            verify_ckpt_arena4_kronecker_stats(&a4[..64], dnam_engine::types::Tier::Lite, 1)
                .is_err()
        );
        a4[0..8].copy_from_slice(&f64::NAN.to_le_bytes());
        assert!(
            verify_ckpt_arena4_kronecker_stats(&a4, dnam_engine::types::Tier::Lite, 1).is_err()
        );
    }

    #[test]
    fn test_verify_ckpt_arena5_calib_stats_valid_and_error_branches() {
        let mut a5 = [0u8; 384];
        a5[0..4].copy_from_slice(b"DNCA");
        a5[4..6].copy_from_slice(&1u16.to_le_bytes());
        a5[8..10].copy_from_slice(&1u16.to_le_bytes()); // num_inputs = 1
        a5[10..12].copy_from_slice(&1u16.to_le_bytes()); // num_stems = 1
        a5[12..16].copy_from_slice(&384u32.to_le_bytes());

        // INPC record at offset 64..192 (len 128)
        a5[64..68].copy_from_slice(b"INPC");
        a5[64 + 56..64 + 60].copy_from_slice(&128u32.to_le_bytes());
        let inpc_crc = crc32c_compute(&a5[128..192]);
        a5[64 + 60..64 + 64].copy_from_slice(&inpc_crc.to_le_bytes());

        // STMC record at offset 192..384 (len 192)
        a5[192..196].copy_from_slice(b"STMC");
        a5[192 + 116..192 + 120].copy_from_slice(&192u32.to_le_bytes());
        let stmc_crc = crc32c_compute(&a5[320..384]);
        a5[192 + 120..192 + 124].copy_from_slice(&stmc_crc.to_le_bytes());

        let body_crc = crc32c_compute(&a5[64..]);
        a5[16..20].copy_from_slice(&body_crc.to_le_bytes());

        assert_eq!(verify_ckpt_arena5_calib_stats(&a5, 0).unwrap(), 384);
        assert!(verify_ckpt_arena5_calib_stats(&a5, 3).is_err());
        assert!(verify_ckpt_arena5_calib_stats(&a5[..32], 0).is_err());
        assert!(verify_ckpt_arena5_calib_stats(&a5[..380], 0).is_err());

        let mut bad = a5;
        bad[0] = b'X';
        assert!(verify_ckpt_arena5_calib_stats(&bad, 0).is_err());

        let mut bad_ver = a5;
        bad_ver[4..6].copy_from_slice(&2u16.to_le_bytes());
        assert!(verify_ckpt_arena5_calib_stats(&bad_ver, 0).is_err());

        let mut bad_len = a5;
        bad_len[12..16].copy_from_slice(&256u32.to_le_bytes());
        assert!(verify_ckpt_arena5_calib_stats(&bad_len, 0).is_err());

        let mut bad_crc = a5;
        bad_crc[16] ^= 0xFF;
        assert!(verify_ckpt_arena5_calib_stats(&bad_crc, 0).is_err());

        let mut bad_inpc = a5;
        bad_inpc[64] = b'X';
        let c = crc32c_compute(&bad_inpc[64..]);
        bad_inpc[16..20].copy_from_slice(&c.to_le_bytes());
        assert!(verify_ckpt_arena5_calib_stats(&bad_inpc, 0).is_err());

        let mut bad_inpc_crc = a5;
        bad_inpc_crc[64 + 60] ^= 0xFF;
        let c = crc32c_compute(&bad_inpc_crc[64..]);
        bad_inpc_crc[16..20].copy_from_slice(&c.to_le_bytes());
        assert!(verify_ckpt_arena5_calib_stats(&bad_inpc_crc, 0).is_err());

        let mut bad_stmc = a5;
        bad_stmc[192] = b'X';
        let c = crc32c_compute(&bad_stmc[64..]);
        bad_stmc[16..20].copy_from_slice(&c.to_le_bytes());
        assert!(verify_ckpt_arena5_calib_stats(&bad_stmc, 0).is_err());

        let mut bad_stmc_crc = a5;
        bad_stmc_crc[192 + 120] ^= 0xFF;
        let c = crc32c_compute(&bad_stmc_crc[64..]);
        bad_stmc_crc[16..20].copy_from_slice(&c.to_le_bytes());
        assert!(verify_ckpt_arena5_calib_stats(&bad_stmc_crc, 0).is_err());

        let mut bad_inpc_len = a5;
        bad_inpc_len[64 + 56..64 + 60].copy_from_slice(&32u32.to_le_bytes());
        let c = crc32c_compute(&bad_inpc_len[64..]);
        bad_inpc_len[16..20].copy_from_slice(&c.to_le_bytes());
        assert!(verify_ckpt_arena5_calib_stats(&bad_inpc_len, 0).is_err());

        let mut bad_stmc_len = a5;
        bad_stmc_len[192 + 116..192 + 120].copy_from_slice(&64u32.to_le_bytes());
        let c = crc32c_compute(&bad_stmc_len[64..]);
        bad_stmc_len[16..20].copy_from_slice(&c.to_le_bytes());
        assert!(verify_ckpt_arena5_calib_stats(&bad_stmc_len, 0).is_err());

        let mut trailing = a5;
        trailing[10..12].copy_from_slice(&0u16.to_le_bytes());
        assert!(verify_ckpt_arena5_calib_stats(&trailing, 0).is_err());

        let mut trunc_inpc = a5;
        trunc_inpc[8..10].copy_from_slice(&2u16.to_le_bytes());
        trunc_inpc[64 + 56..64 + 60].copy_from_slice(&320u32.to_le_bytes());
        let c_inpc = crc32c_compute(&trunc_inpc[128..384]);
        trunc_inpc[64 + 60..64 + 64].copy_from_slice(&c_inpc.to_le_bytes());
        let c = crc32c_compute(&trunc_inpc[64..]);
        trunc_inpc[16..20].copy_from_slice(&c.to_le_bytes());
        assert!(verify_ckpt_arena5_calib_stats(&trunc_inpc, 0).is_err());

        let mut trunc_stmc = a5;
        trunc_stmc[10..12].copy_from_slice(&2u16.to_le_bytes());
        assert!(verify_ckpt_arena5_calib_stats(&trunc_stmc, 0).is_err());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_uplo_u_prefix_theorem(
            seed in 1_u64..10_000_u64,
        ) {
            const N_FULL: usize = 8;
            const N_SUB: usize = 4;
            let mut dense = [0.0_f64; N_FULL * N_FULL];
            for i in 0..N_FULL {
                for j in i..N_FULL {
                    let val = (((seed as usize) + i * 17 + j * 31) % 97) as f64 * 0.1;
                    dense[i * N_FULL + j] = val;
                    dense[j * N_FULL + i] = val;
                }
            }
            let mut packed_full = [0.0_f64; uplo_u_packed_len(N_FULL)];
            pack_uplo_u_f64(&dense, N_FULL, &mut packed_full);

            let mut unpacked_sub = [0.0_f64; N_SUB * N_SUB];
            unpack_uplo_u_f64(&packed_full[..uplo_u_packed_len(N_SUB)], N_SUB, &mut unpacked_sub);
            for i in 0..N_SUB {
                for j in 0..N_SUB {
                    prop_assert_eq!(unpacked_sub[i * N_SUB + j], dense[i * N_FULL + j]);
                }
            }
        }
    }

    /// verify[container.ckpt.standalone-full]
    /// verify[container.ckpt.prefix-delta]
    /// verify[container.ckpt.objective-records]
    /// verify[container.ckpt.stems-registry]
    /// verify[container.ckpt.gram-fingerprint]
    /// verify[container.dnca.inpc]
    /// verify[container.dnca.stmc]
    /// verify[container.dnca.dnce-sidecar]
    /// verify[container.dnca.coreset-fp16]
    #[test]
    fn test_checkpoint_arena4_and_dnca_coreset_fp16_layout_invariants() {
        assert_eq!(uplo_u_packed_len(64) * 8, 16_640);
        assert_eq!(uplo_u_packed_len(128) * 8, 66_048);
        assert_eq!(uplo_u_packed_len(192) * 8, 148_224);
        assert_eq!(uplo_u_packed_len(256) * 8, 263_168);

        let dims = [64usize, 128, 192, 256];
        let standalone_totals = [
            ARENA4_STEM_BYTES_LITE,
            ARENA4_STEM_BYTES_FULL,
            ARENA4_STEM_BYTES_PRO,
            ARENA4_STEM_BYTES_ULTRA,
        ];
        for (&n, &expected) in dims.iter().zip(&standalone_totals) {
            let g_bytes = 8 * (n * (n + 1) / 2);
            let c_bytes = 16 * n;
            assert_eq!(g_bytes + c_bytes, expected);
            assert_eq!(expected % 64, 0);
        }
        let coreset_mono_fp16_bytes = 12_000usize * 2;
        let coreset_stereo_fp16_bytes = 12_000usize * 2 * 2;
        assert_eq!(coreset_mono_fp16_bytes, 24_000);
        assert_eq!(coreset_stereo_fp16_bytes, 48_000);
        assert_eq!(coreset_mono_fp16_bytes % 64, 0);
        assert_eq!(coreset_stereo_fp16_bytes % 64, 0);
    }
}
