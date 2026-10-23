//! `.dnam-ckpt` `Arena 3` Column-Major Upper-Triangular (`LAPACK uplo='U'`) packing & stem verification.

use crate::schema::DnamError;

use super::crc32c_compute;

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
