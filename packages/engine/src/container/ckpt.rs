//! `.dnam-ckpt` `Arena 3` Column-Major Upper-Triangular (`LAPACK uplo='U'`) packing,
//! `Arena 4` calibration statistics (`DNCA`) verification & stem verification.

use alloc::string::ToString;

use crate::schema::DnamError;

use super::crc32c_compute;

/// Magic bytes for `.dnam-ckpt` `Arena 4` calibration statistics header (`b"DNCA"`).
pub const DNCA_MAGIC: [u8; 4] = *b"DNCA";

/// Fixed 64-byte header length for `.dnam-ckpt` `Arena 4`.
pub const ARENA4_HEADER_BYTES: usize = 64;

/// Returns the packed element count $\frac{N(N+1)}{2}$ for a symmetric $N \times N$ Gram matrix:
/// - `Lite` ($N=64$): `2,080` `f64` (`16,640 B`)
/// - `Full` ($N=128$): `8,256` `f64` (`66,048 B`)
/// - `Ultra` ($N=192$): `18,528` `f64` (`148,224 B`)
#[inline]
pub const fn uplo_u_packed_len(n_dim: usize) -> usize {
    (n_dim * (n_dim + 1)) / 2
}

/// Packs a dense symmetric $N \times N$ row-major matrix `dense_matrix` (`N * N` elements)
/// into **Column-Major Upper-Triangular (`LAPACK uplo='U'`)** format:
/// $$\operatorname{pack}_{\text{U}}(\mathbf{G})\!\left[i + \frac{j(j+1)}{2}\right] = \mathbf{G}[i, j], \qquad 0 \le i \le j < N$$
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
/// `n_tier = 128` (`Full`) on an `Ultra` ($N=192$) packed buffer extracts the exact leading
/// principal submatrix from the contiguous prefix `[0 .. n_tier*(n_tier+1)/2]`!
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

/// Verifies a `.dnam-ckpt` stem's `stem_arena3_crc32c` over its `Arena 3` slice (`151,296 B` Ultra).
pub fn verify_ckpt_stem_crc32c(
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
            region: "arena3_stem",
            expected: expected_crc32c,
            computed,
        });
    }
    Ok(())
}

/// Verifies `.dnam-ckpt` `Arena 4` (`DNCA` magic, version, 64B alignment, body `CRC32C`,
/// and per-entry `INPC`/`STMC` record framing and `CRC32C`s) starting at `arena4_offset`.
pub fn verify_ckpt_arena4_calib_stats(
    arena_bytes: &[u8],
    arena4_offset: usize,
) -> Result<usize, DnamError> {
    if (arena4_offset & 0x3F) != 0 {
        return Err(DnamError::UnalignedOffset {
            field: "arena4_offset",
            offset: arena4_offset as u32,
        });
    }
    if arena_bytes.len() < arena4_offset + ARENA4_HEADER_BYTES {
        return Err(DnamError::BufferTooSmall {
            expected: arena4_offset + ARENA4_HEADER_BYTES,
            actual: arena_bytes.len(),
        });
    }
    let a4 = &arena_bytes[arena4_offset..];
    if (a4.len() & 0x3F) != 0 {
        return Err(DnamError::UnalignedOffset {
            field: "arena4_byte_len",
            offset: a4.len() as u32,
        });
    }
    let magic = [a4[0], a4[1], a4[2], a4[3]];
    if magic != DNCA_MAGIC {
        return Err(DnamError::SchemaValidationError(
            "invalid Arena 4 magic (expected 'DNCA')".to_string(),
        ));
    }
    let version = u16::from_le_bytes([a4[4], a4[5]]);
    if version != 1 {
        return Err(DnamError::SchemaValidationError(alloc::format!(
            "unsupported Arena 4 version {version} (expected 1)"
        )));
    }
    let num_inputs = u16::from_le_bytes([a4[8], a4[9]]) as usize;
    let num_stems = u16::from_le_bytes([a4[10], a4[11]]) as usize;
    let total_byte_len = u32::from_le_bytes([a4[12], a4[13], a4[14], a4[15]]) as usize;
    let expected_body_crc = u32::from_le_bytes([a4[16], a4[17], a4[18], a4[19]]);

    if total_byte_len != a4.len() || (total_byte_len & 0x3F) != 0 {
        return Err(DnamError::SchemaValidationError(alloc::format!(
            "Arena 4 total_byte_len ({total_byte_len}) does not match slice length ({})",
            a4.len()
        )));
    }
    let computed_body_crc = crc32c_compute(&a4[ARENA4_HEADER_BYTES..]);
    if computed_body_crc != expected_body_crc {
        return Err(DnamError::Crc32cMismatch {
            region: "arena4_calib",
            expected: expected_body_crc,
            computed: computed_body_crc,
        });
    }

    let mut pos = ARENA4_HEADER_BYTES;
    for _ in 0..num_inputs {
        if pos + 64 > a4.len() {
            return Err(DnamError::BufferTooSmall {
                expected: arena4_offset + pos + 64,
                actual: arena_bytes.len(),
            });
        }
        if &a4[pos..pos + 4] != b"INPC" {
            return Err(DnamError::SchemaValidationError(
                "invalid Arena 4 input record magic (expected 'INPC')".to_string(),
            ));
        }
        let rec_len =
            u32::from_le_bytes([a4[pos + 56], a4[pos + 57], a4[pos + 58], a4[pos + 59]]) as usize;
        let rec_crc = u32::from_le_bytes([a4[pos + 60], a4[pos + 61], a4[pos + 62], a4[pos + 63]]);
        if rec_len < 64 || (rec_len & 0x3F) != 0 || pos + rec_len > a4.len() {
            return Err(DnamError::SchemaValidationError(
                "invalid Arena 4 input record byte length".to_string(),
            ));
        }
        let calc_crc = crc32c_compute(&a4[pos + 64..pos + rec_len]);
        if calc_crc != rec_crc {
            return Err(DnamError::Crc32cMismatch {
                region: "arena4_input_record",
                expected: rec_crc,
                computed: calc_crc,
            });
        }
        pos += rec_len;
    }

    for _ in 0..num_stems {
        if pos + 128 > a4.len() {
            return Err(DnamError::BufferTooSmall {
                expected: arena4_offset + pos + 128,
                actual: arena_bytes.len(),
            });
        }
        if &a4[pos..pos + 4] != b"STMC" {
            return Err(DnamError::SchemaValidationError(
                "invalid Arena 4 stem record magic (expected 'STMC')".to_string(),
            ));
        }
        let rec_len =
            u32::from_le_bytes([a4[pos + 116], a4[pos + 117], a4[pos + 118], a4[pos + 119]])
                as usize;
        let rec_crc =
            u32::from_le_bytes([a4[pos + 120], a4[pos + 121], a4[pos + 122], a4[pos + 123]]);
        if rec_len < 128 || (rec_len & 0x3F) != 0 || pos + rec_len > a4.len() {
            return Err(DnamError::SchemaValidationError(
                "invalid Arena 4 stem record byte length".to_string(),
            ));
        }
        let calc_crc = crc32c_compute(&a4[pos + 128..pos + rec_len]);
        if calc_crc != rec_crc {
            return Err(DnamError::Crc32cMismatch {
                region: "arena4_stem_record",
                expected: rec_crc,
                computed: calc_crc,
            });
        }
        pos += rec_len;
    }

    if pos != a4.len() {
        return Err(DnamError::SchemaValidationError(
            "Arena 4 trailing bytes after records".to_string(),
        ));
    }
    Ok(total_byte_len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_ckpt_stem_crc32c_verification() {
        let stem_buf = [5u8; 128];
        let stem_crc = crc32c_compute(&stem_buf[0..64]);
        assert!(verify_ckpt_stem_crc32c(&stem_buf, 0, 64, stem_crc).is_ok());
        assert!(verify_ckpt_stem_crc32c(&stem_buf, 0, 256, stem_crc).is_err());
        assert!(verify_ckpt_stem_crc32c(&stem_buf, 0, 64, stem_crc ^ 1).is_err());
    }

    #[test]
    fn test_verify_ckpt_arena4_calib_stats_valid_and_error_branches() {
        // Build a valid Arena 4 buffer with 1 input record (128B) and 1 stem record (192B)
        // Total = 64 (header) + 128 (INPC) + 192 (STMC) = 384 bytes
        let mut a4 = [0u8; 384];
        a4[0..4].copy_from_slice(b"DNCA");
        a4[4..6].copy_from_slice(&1u16.to_le_bytes());
        a4[8..10].copy_from_slice(&1u16.to_le_bytes()); // num_inputs = 1
        a4[10..12].copy_from_slice(&1u16.to_le_bytes()); // num_stems = 1
        a4[12..16].copy_from_slice(&384u32.to_le_bytes());

        // INPC record at offset 64..192 (len 128)
        a4[64..68].copy_from_slice(b"INPC");
        a4[64 + 56..64 + 60].copy_from_slice(&128u32.to_le_bytes());
        let inpc_crc = crc32c_compute(&a4[128..192]);
        a4[64 + 60..64 + 64].copy_from_slice(&inpc_crc.to_le_bytes());

        // STMC record at offset 192..384 (len 192)
        a4[192..196].copy_from_slice(b"STMC");
        a4[192 + 116..192 + 120].copy_from_slice(&192u32.to_le_bytes());
        let stmc_crc = crc32c_compute(&a4[320..384]);
        a4[192 + 120..192 + 124].copy_from_slice(&stmc_crc.to_le_bytes());

        let body_crc = crc32c_compute(&a4[64..]);
        a4[16..20].copy_from_slice(&body_crc.to_le_bytes());

        assert_eq!(verify_ckpt_arena4_calib_stats(&a4, 0).unwrap(), 384);
        assert!(verify_ckpt_arena4_calib_stats(&a4, 3).is_err());
        assert!(verify_ckpt_arena4_calib_stats(&a4[..32], 0).is_err());
        assert!(verify_ckpt_arena4_calib_stats(&a4[..380], 0).is_err());

        // Corrupt header fields
        let mut bad = a4;
        bad[0] = b'X';
        assert!(verify_ckpt_arena4_calib_stats(&bad, 0).is_err());

        let mut bad_ver = a4;
        bad_ver[4..6].copy_from_slice(&2u16.to_le_bytes());
        assert!(verify_ckpt_arena4_calib_stats(&bad_ver, 0).is_err());

        let mut bad_len = a4;
        bad_len[12..16].copy_from_slice(&256u32.to_le_bytes());
        assert!(verify_ckpt_arena4_calib_stats(&bad_len, 0).is_err());

        let mut bad_crc = a4;
        bad_crc[16] ^= 0xFF;
        assert!(verify_ckpt_arena4_calib_stats(&bad_crc, 0).is_err());

        // Corrupt INPC magic & CRC
        let mut bad_inpc = a4;
        bad_inpc[64] = b'X';
        let c = crc32c_compute(&bad_inpc[64..]);
        bad_inpc[16..20].copy_from_slice(&c.to_le_bytes());
        assert!(verify_ckpt_arena4_calib_stats(&bad_inpc, 0).is_err());

        let mut bad_inpc_crc = a4;
        bad_inpc_crc[64 + 60] ^= 0xFF;
        let c = crc32c_compute(&bad_inpc_crc[64..]);
        bad_inpc_crc[16..20].copy_from_slice(&c.to_le_bytes());
        assert!(verify_ckpt_arena4_calib_stats(&bad_inpc_crc, 0).is_err());

        // Corrupt STMC magic & CRC
        let mut bad_stmc = a4;
        bad_stmc[192] = b'X';
        let c = crc32c_compute(&bad_stmc[64..]);
        bad_stmc[16..20].copy_from_slice(&c.to_le_bytes());
        assert!(verify_ckpt_arena4_calib_stats(&bad_stmc, 0).is_err());

        let mut bad_stmc_crc = a4;
        bad_stmc_crc[192 + 120] ^= 0xFF;
        let c = crc32c_compute(&bad_stmc_crc[64..]);
        bad_stmc_crc[16..20].copy_from_slice(&c.to_le_bytes());
        assert!(verify_ckpt_arena4_calib_stats(&bad_stmc_crc, 0).is_err());
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
}
