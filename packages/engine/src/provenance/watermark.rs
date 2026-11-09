//! 3-Layer Buyer Watermarking (`Test 8H`: Layer 1 Coupled Bilinear Gauge, Layer 2 Pole-Angle Phase Coding,
//! Layer 3 Two-Seed Arcsine Tardos) & Prefix-Composable `4 × 56-D = 224-D` Gauge-Invariant Structural Hash.

use crate::schema::{DnamError, cold_sqrt_f64};

use super::sha256::sha256_digest;

pub mod tardos;
pub use tardos::{
    TARDOS_CASCADE_RANGES, TardosCodebook7680, apply_cascade_head_gauge,
    apply_tardos_watermark_7680, apply_tardos_watermark_7680_envelopes,
    compute_emr_masked_tardos_envelopes, generate_tardos_code_7680, generate_tardos_codebook_7680,
    is_role_a_cascade_index, role_b_head_slot_for_index, score_tardos_watermark_7680,
};

#[inline]
pub(crate) fn read_f32_le(arena: &[u8], byte_offset: usize) -> f32 {
    f32::from_le_bytes([
        arena[byte_offset],
        arena[byte_offset + 1],
        arena[byte_offset + 2],
        arena[byte_offset + 3],
    ])
}

#[inline]
pub(crate) fn write_f32_le(arena: &mut [u8], byte_offset: usize, val: f32) {
    let b = val.to_le_bytes();
    arena[byte_offset..byte_offset + 4].copy_from_slice(&b);
}

#[inline]
pub(crate) const fn max_tiers_for_arena_len(arena_len: usize) -> usize {
    if arena_len >= 64_000 {
        4
    } else if arena_len >= 36_224 {
        3
    } else if arena_len >= 22_336 {
        2
    } else if arena_len >= 10_752 {
        1
    } else {
        0
    }
}

/// Per-tier byte offsets of the 128-byte Stage III Early Reflection FIR block (`16` Left + `16` Right `f32` taps)
/// inside `Arena 0..3`:
/// - `tier == 0` (`Lite`): `4_288..4_416` (`128 B`)
/// - `tier == 1` (`Full`): `17_408..17_536` (`128 B`)
/// - `tier == 2` (`Pro`): `31_552..31_680` (`128 B`)
/// - `tier == 3` (`Ultra`): `48_512..48_640` (`128 B`)
pub const STAGE3_EARLY_FIR_BASES: [usize; 4] = [4_288, 17_408, 31_552, 48_512];

/// Per-tier base byte offsets of the Stage III Orthonormal Kautz 128-byte (`[8, 4]` `f32` SoA) modal blocks
/// inside `Arena 0..3`:
/// - `tier == 0` (`Lite`, `1` block = `4` pole pairs, `128 B`): `4_416..4_544`
/// - `tier == 1` (`Full`, `1` block = `+4` pole pairs, `128 B`): `17_536..17_664`
/// - `tier == 2` (`Pro`, `2` blocks = `+8` pole pairs, `256 B`): `31_680..31_936`
/// - `tier == 3` (`Ultra`, `4` blocks = `+16` pole pairs, `512 B`): `60_160..60_672`
pub const STAGE3_KAUTZ_BASES: [usize; 4] = [4_416, 17_536, 31_680, 60_160];

/// Number of 128-byte (`4`-pole-pair) Kautz SoA blocks stored in each tier delta (`Lite=1`, `Full=1`, `Pro=2`, `Ultra=4`).
pub const STAGE3_KAUTZ_BLOCKS_PER_TIER: [usize; 4] = [1, 1, 2, 4];

/// Cumulative Kautz pole-pair channel base index per tier (`[0, 4, 8, 16]`, total `32` pole pairs across all 4 tiers).
pub const STAGE3_KAUTZ_TIER_CHANNEL_BASE: [usize; 4] = [0, 4, 8, 16];

/// Returns the byte offset of the 128-byte modal/Kautz SoA struct (`[8, 4]` floats:
/// `[r_low, dr, theta, b_gamma, c_re_l, c_im_l, c_re_r, c_im_r]`) for `(tier, block_idx)` inside `Arena 0..3`.
#[inline]
pub const fn kautz_soa_byte_offset(tier: usize, block_idx: usize) -> usize {
    let t = if tier < 4 { tier } else { 3 };
    STAGE3_KAUTZ_BASES[t] + block_idx * 128
}

/// Applies **Mechanism A (Exact Isometric Gauge Watermark)** in-place on a prefix-composable
/// `Arena 0..3` buffer (`10,752 B`, `22,336 B`, `36,224 B`, or `64,000 B`):
/// 1. Scales $b_{\gamma,b,c}$ by $\kappa_{b,c} \in [0.985, 1.015]$ and $(c_{\text{real},L/R,b,c}, c_{\text{imag},L/R,b,c})$
///    by $\kappa_{b,c}^{-1}$ across all present Stage III Kautz modal blocks (`[1, 1, 2, 4]` blocks per tier = `32` pole pairs in `Ultra`; `0.0000 dB` audio distortion).
/// 2. Rotates the SVD stereo output head basis $\mathbf{W}_{\text{head},1..4}$ (and `Ultra` modes `5..8` when present)
///    by $\mathbf{Q}_{\text{buyer}}^\top \in SO(4)$ and the control subspace projections (`CP1` contributors, $\mathbf{U}_{\text{int}}$, and $\mathbf{W}_{\text{ctrl}}$ columns `0..4`)
///    by $\mathbf{Q}_{\text{buyer}}$.
pub fn apply_isometric_gauge_watermark(
    arena: &mut [u8],
    kappa: &[f32],
    q_buyer_4x4: &[[f32; 4]; 4],
) -> Result<(), DnamError> {
    let max_tiers = max_tiers_for_arena_len(arena.len());
    if max_tiers == 0 {
        return Err(DnamError::BufferTooSmall {
            expected: 10_752,
            actual: arena.len(),
        });
    }
    let required_channels = STAGE3_KAUTZ_TIER_CHANNEL_BASE[max_tiers - 1]
        + STAGE3_KAUTZ_BLOCKS_PER_TIER[max_tiers - 1] * 4;
    if kappa.len() < required_channels {
        return Err(DnamError::BufferTooSmall {
            expected: required_channels,
            actual: kappa.len(),
        });
    }

    // 1. Modal/Kautz reciprocal scaling gauge (b_gamma * kappa, c_re_l..c_im_r / kappa)
    for tier in 0..max_tiers {
        let blocks = STAGE3_KAUTZ_BLOCKS_PER_TIER[tier];
        let ch_tier_base = STAGE3_KAUTZ_TIER_CHANNEL_BASE[tier];
        for blk in 0..blocks {
            let base = kautz_soa_byte_offset(tier, blk);
            for lane in 0..4 {
                let ch_global = ch_tier_base + blk * 4 + lane;
                let k_val = kappa[ch_global];
                let inv_kappa = 1.0 / k_val;

                let b_gamma_off = base + (12 + lane) * 4;
                let c_re_l_off = base + (16 + lane) * 4;
                let c_im_l_off = base + (20 + lane) * 4;
                let c_re_r_off = base + (24 + lane) * 4;
                let c_im_r_off = base + (28 + lane) * 4;

                let b_gamma = read_f32_le(arena, b_gamma_off);
                let c_re_l = read_f32_le(arena, c_re_l_off);
                let c_im_l = read_f32_le(arena, c_im_l_off);
                let c_re_r = read_f32_le(arena, c_re_r_off);
                let c_im_r = read_f32_le(arena, c_im_r_off);

                write_f32_le(arena, b_gamma_off, b_gamma * k_val);
                write_f32_le(arena, c_re_l_off, c_re_l * inv_kappa);
                write_f32_le(arena, c_im_l_off, c_im_l * inv_kappa);
                write_f32_le(arena, c_re_r_off, c_re_r * inv_kappa);
                write_f32_le(arena, c_im_r_off, c_im_r * inv_kappa);
            }
        }
    }

    // 2. Rotate SVD stereo head basis W_head[1..5] (each mode has 2*16*4 = 128 floats = 512 B)
    // across Arena 0 (4928..7488), Arena 1 (18048..20608), Arena 2 (31936..34496), Arena 3 (57600..60160):
    let head_arena_offsets = [4_928_usize, 18_048_usize, 31_936_usize, 57_600_usize];
    for &head_base in head_arena_offsets.iter().take(max_tiers) {
        for elem_idx in 0..128 {
            let mut old_vals = [0.0_f32; 4];
            for m in 0..4 {
                let off = head_base + (m + 1) * 512 + elem_idx * 4;
                old_vals[m] = read_f32_le(arena, off);
            }
            for k in 0..4 {
                let mut rotated = 0.0_f32;
                for m in 0..4 {
                    rotated += q_buyer_4x4[m][k] * old_vals[m];
                }
                let off = head_base + (k + 1) * 512 + elem_idx * 4;
                write_f32_le(arena, off, rotated);
            }
        }
    }

    // Also rotate b_head[1..5] in Arena 0 CL24 (bytes 9024..9064: 5 modes * 2 stereo channels):
    let b_head_base = 9_024_usize;
    for lr in 0..2 {
        let mut old_b = [0.0_f32; 4];
        for m in 0..4 {
            let off = b_head_base + ((m + 1) * 2 + lr) * 4;
            old_b[m] = read_f32_le(arena, off);
        }
        for k in 0..4 {
            let mut rotated = 0.0_f32;
            for m in 0..4 {
                rotated += q_buyer_4x4[m][k] * old_b[m];
            }
            let off = b_head_base + ((k + 1) * 2 + lr) * 4;
            write_f32_le(arena, off, rotated);
        }
    }

    // Helper closure to post-multiply a 4-float row at `row_byte_off` by `q_buyer_4x4`:
    let rotate_row_4 = |buf: &mut [u8], row_byte_off: usize| {
        let mut old_row = [0.0_f32; 4];
        for m in 0..4 {
            old_row[m] = read_f32_le(buf, row_byte_off + m * 4);
        }
        for k in 0..4 {
            let mut rotated = 0.0_f32;
            for m in 0..4 {
                rotated += old_row[m] * q_buyer_4x4[m][k];
            }
            write_f32_le(buf, row_byte_off + k * 4, rotated);
        }
    };

    // 3. Rotate inline CP1 contributors (7,680..8,512) + U_int^base (7,552..7,680) + W_ctrl^base[:, 0..4] (9,536..10,560)
    //    when cp1_lanes_dir (7,524..7,526) has count > 0, or legacy P_svd (8,256..8,512) when count == 0:
    let cp1_count = u16::from_le_bytes([arena[7_524], arena[7_525]]) as usize;
    if (1..=26).contains(&cp1_count) {
        let use_3b_dir = cp1_count <= 8
            && arena[7_526] == 0
            && arena[7_527] == 0
            && ((1..=3).contains(&(arena[7_529] & 0x0F))
                || (1..=3).contains(&(arena[7_530] & 0x0F)));
        let mut b_cursor = 7_680_usize;
        let mut u_cursor = 61_312_usize;
        for c in 0..cp1_count {
            let packed = if use_3b_dir {
                let b1 = arena[7_528 + c * 3 + 1];
                let b2 = arena[7_528 + c * 3 + 2];
                if (1..=3).contains(&(b1 & 0x0F)) {
                    b1
                } else {
                    b2
                }
            } else {
                arena[7_526 + c]
            };
            let lanes = (packed & 0x0F) as usize;
            let b_bytes = 4 * (9 * lanes + 4);
            if !(1..=3).contains(&lanes) || b_cursor + b_bytes > 8_512 {
                break;
            }
            for l in 0..lanes {
                rotate_row_4(arena, b_cursor + l * 32);
            }
            b_cursor += b_bytes;
            if max_tiers >= 4 && u_cursor + (32 * lanes + 16) <= 63_424 {
                for l in 0..lanes {
                    rotate_row_4(arena, u_cursor + l * 32);
                }
                u_cursor += 32 * lanes + 16;
            }
        }
        // Rotate U_int^base[0..4][0..4] at 7,552..7,680 (128 B = 4 rows x 8 floats)
        for r in 0..4 {
            rotate_row_4(arena, 7_552 + r * 32);
        }
        // Rotate W_ctrl^base[row, 0..4] at 9,536..10,560 so W_ctrl * q_16 is gauge-invariant under z_head rotation
        for row in 0..16 {
            rotate_row_4(arena, 9_536 + row * 64);
        }
        if max_tiers >= 4 {
            // Rotate 3 x U_int,Delta[0..12][0..4] at 60,928..61,312 (384 B = 12 rows x 8 floats)
            for r in 0..12 {
                rotate_row_4(arena, 60_928 + r * 32);
            }
        }
    } else {
        let p_svd_base = 8_256_usize;
        for row in 0..16 {
            rotate_row_4(arena, p_svd_base + row * 16);
        }
        if max_tiers >= 4 {
            let p_svd_ultra_base = 63_168_usize;
            for row in 0..16 {
                rotate_row_4(arena, p_svd_ultra_base + row * 16);
            }
        }
    }

    // 4. When Arena 3 (Tier::Ultra, 64,000 B) is present, also rotate modes 5..8 (49,408..57,600)
    //    and b_head_modes_5_to_8 (63,424..63,456):
    if max_tiers >= 4 {
        for tier in 0..4 {
            let ultra_mode_base = 49_408 + tier * 2048;
            for elem_idx in 0..128 {
                let mut old_vals = [0.0_f32; 4];
                for m in 0..4 {
                    let off = ultra_mode_base + m * 512 + elem_idx * 4;
                    old_vals[m] = read_f32_le(arena, off);
                }
                for k in 0..4 {
                    let mut rotated = 0.0_f32;
                    for m in 0..4 {
                        rotated += q_buyer_4x4[m][k] * old_vals[m];
                    }
                    let off = ultra_mode_base + k * 512 + elem_idx * 4;
                    write_f32_le(arena, off, rotated);
                }
            }
        }
        for lr in 0..2 {
            let mut old_b = [0.0_f32; 4];
            for m in 0..4 {
                let off = 63_424 + (m * 2 + lr) * 4;
                old_b[m] = read_f32_le(arena, off);
            }
            for k in 0..4 {
                let mut rotated = 0.0_f32;
                for m in 0..4 {
                    rotated += q_buyer_4x4[m][k] * old_b[m];
                }
                let off = 63_424 + (k * 2 + lr) * 4;
                write_f32_le(arena, off, rotated);
            }
        }
    }

    Ok(())
}

pub use super::fingerprint::{extract_structural_hash_224d, structural_cosine_similarity};

/// Maximum Layer 2 pole-angle phase perturbation (`0.05 deg = 8.726646e-4 rad`, `0.00 dB` magnitude change).
pub const PHASE_CODING_MAX_RAD: f32 = 8.726_646e-4;

#[inline]
fn derive_phase_coding_tier_digest(buyer_seed: &[u8; 32], tier: u8) -> [u8; 32] {
    let mut msg = [0u8; 17 + 32 + 1];
    msg[0..17].copy_from_slice(b"DNAM-WM-PHASE-V1:");
    msg[17..49].copy_from_slice(buyer_seed);
    msg[49] = tier;
    sha256_digest(&msg)
}

#[inline]
fn phase_chip_from_digest(digest: &[u8; 32], chip_idx: usize) -> f32 {
    let byte = digest[(chip_idx / 8) & 31];
    let bit = (byte >> (chip_idx & 7)) & 1;
    if bit == 1 { 1.0_f32 } else { -1.0_f32 }
}

/// Applies **Layer 2 (Kautz & Inter-Stage Pole-Angle Phase Coding)** in-place on `Arena 0..3`
/// using domain tag `"DNAM-WM-PHASE-V1"` (`delta_rad <= PHASE_CODING_MAX_RAD = 8.726646e-4 rad` = `0.05 deg`):
/// - Perturbs each tier's Stage III Kautz modal pole angles $\theta_m$ (`STAGE3_KAUTZ_BASES`, `+32..+48` in each `128 B` block)
///   by $\pm\Delta\theta$ (`0.00 dB` pole-radius change).
/// - Perturbs active Stage II lifted inter-stage resonator pole angles (`theta_mid_1` at `4,180` when `r_mid_1 > 0.0`,
///   and `theta_mid_2` at `4,212` when `r_mid_2 > 0.0`) by $\pm\Delta\theta$.
pub fn apply_phase_coding_watermark(
    arena: &mut [u8],
    buyer_seed: &[u8; 32],
    delta_rad: f32,
) -> Result<(), DnamError> {
    let max_tiers = max_tiers_for_arena_len(arena.len());
    if max_tiers == 0 {
        return Err(DnamError::BufferTooSmall {
            expected: 10_752,
            actual: arena.len(),
        });
    }
    let step_rad = delta_rad.clamp(0.0, PHASE_CODING_MAX_RAD);

    for tier in 0..max_tiers {
        let digest = derive_phase_coding_tier_digest(buyer_seed, tier as u8);
        let blocks = STAGE3_KAUTZ_BLOCKS_PER_TIER[tier];
        for blk in 0..blocks {
            let base = kautz_soa_byte_offset(tier, blk);
            for lane in 0..4 {
                let theta_off = base + (8 + lane) * 4;
                let theta = read_f32_le(arena, theta_off);
                if (1e-3..=core::f32::consts::PI - 1e-3).contains(&theta) {
                    let chip_idx = blk * 4 + lane;
                    let chip = phase_chip_from_digest(&digest, chip_idx);
                    let perturbed =
                        (theta + step_rad * chip).clamp(1e-4, core::f32::consts::PI - 1e-4);
                    write_f32_le(arena, theta_off, perturbed);
                }
            }
        }
        if tier == 0 {
            let r_mid_1 = read_f32_le(arena, 4_176);
            let th1 = read_f32_le(arena, 4_180);
            if r_mid_1 > 0.0 && (1e-3..=core::f32::consts::PI - 1e-3).contains(&th1) {
                let chip = phase_chip_from_digest(&digest, 16);
                write_f32_le(
                    arena,
                    4_180,
                    (th1 + step_rad * chip).clamp(1e-4, core::f32::consts::PI - 1e-4),
                );
            }
            let r_mid_2 = read_f32_le(arena, 4_208);
            let th2 = read_f32_le(arena, 4_212);
            if r_mid_2 > 0.0 && (1e-3..=core::f32::consts::PI - 1e-3).contains(&th2) {
                let chip = phase_chip_from_digest(&digest, 17);
                write_f32_le(
                    arena,
                    4_212,
                    (th2 + step_rad * chip).clamp(1e-4, core::f32::consts::PI - 1e-4),
                );
            }
        }
    }
    Ok(())
}

/// Computes the Layer 2 pole-angle phase-coding Z-score ($\sigma$) of a suspect `Arena 0..3`
/// against `master_arena` and a candidate `buyer_seed` (`"DNAM-WM-PHASE-V1"`).
pub fn score_phase_coding_arena(
    master_arena: &[u8],
    suspect_arena: &[u8],
    buyer_seed: &[u8; 32],
) -> Result<f64, DnamError> {
    let shared_len = master_arena.len().min(suspect_arena.len());
    let max_tiers = max_tiers_for_arena_len(shared_len);
    if max_tiers == 0 {
        return Err(DnamError::BufferTooSmall {
            expected: 10_752,
            actual: shared_len,
        });
    }
    let mut corr = 0.0_f64;
    let mut active = 0.0_f64;

    for tier in 0..max_tiers {
        let digest = derive_phase_coding_tier_digest(buyer_seed, tier as u8);
        let blocks = STAGE3_KAUTZ_BLOCKS_PER_TIER[tier];
        for blk in 0..blocks {
            let base = kautz_soa_byte_offset(tier, blk);
            for lane in 0..4 {
                let theta_off = base + (8 + lane) * 4;
                let theta_m = read_f32_le(master_arena, theta_off);
                if !(1e-3..=core::f32::consts::PI - 1e-3).contains(&theta_m) {
                    continue;
                }
                let chip_idx = blk * 4 + lane;
                let chip = phase_chip_from_digest(&digest, chip_idx) as f64;
                let diff = (read_f32_le(suspect_arena, theta_off) - theta_m) as f64;
                let y = if diff > 1e-6 {
                    1.0
                } else if diff < -1e-6 {
                    -1.0
                } else {
                    0.0
                };
                corr += chip * y;
                active += y * y;
            }
        }
        if tier == 0 {
            for (r_off, th_off, chip_idx) in
                [(4_176_usize, 4_180_usize, 16_usize), (4_208, 4_212, 17)]
            {
                let theta_m = read_f32_le(master_arena, th_off);
                if read_f32_le(master_arena, r_off) > 0.0
                    && (1e-3..=core::f32::consts::PI - 1e-3).contains(&theta_m)
                {
                    let chip = phase_chip_from_digest(&digest, chip_idx) as f64;
                    let diff = (read_f32_le(suspect_arena, th_off) - theta_m) as f64;
                    let y = if diff > 1e-6 {
                        1.0
                    } else if diff < -1e-6 {
                        -1.0
                    } else {
                        0.0
                    };
                    corr += chip * y;
                    active += y * y;
                }
            }
        }
    }
    Ok(corr / cold_sqrt_f64(active.max(1.0)))
}

/// Computes the black-box Stage C pole-angle phase correlation Z-score $Z_{\text{phase},j}$
/// from extracted/identified complex pole angles (`master_pole_angles` vs `suspect_pole_angles`)
/// against a candidate `buyer_seed` (`"DNAM-WM-PHASE-V1"`).
pub fn score_phase_coding_blackbox(
    master_pole_angles: &[f32],
    suspect_pole_angles: &[f32],
    buyer_seed: &[u8; 32],
) -> Result<f64, DnamError> {
    let n = master_pole_angles.len().min(suspect_pole_angles.len());
    if n < 4 {
        return Err(DnamError::BufferTooSmall {
            expected: 4,
            actual: n,
        });
    }
    let tier_counts = [4_usize, 4, 8, 16];
    let mut corr = 0.0_f64;
    let mut active = 0.0_f64;
    let mut global_idx = 0_usize;

    for (tier, &cnt) in tier_counts.iter().enumerate() {
        let digest = derive_phase_coding_tier_digest(buyer_seed, tier as u8);
        for c in 0..cnt {
            if global_idx >= n {
                break;
            }
            let chip = phase_chip_from_digest(&digest, c) as f64;
            let diff = (suspect_pole_angles[global_idx] - master_pole_angles[global_idx]) as f64;
            let y = if diff > 1e-6 {
                1.0
            } else if diff < -1e-6 {
                -1.0
            } else {
                0.0
            };
            corr += chip * y;
            active += y * y;
            global_idx += 1;
        }
    }
    if n > 32 {
        let digest0 = derive_phase_coding_tier_digest(buyer_seed, 0);
        for extra in 0..(n - 32).min(2) {
            let idx = 32 + extra;
            let chip = phase_chip_from_digest(&digest0, 16 + extra) as f64;
            let diff = (suspect_pole_angles[idx] - master_pole_angles[idx]) as f64;
            let y = if diff > 1e-6 {
                1.0
            } else if diff < -1e-6 {
                -1.0
            } else {
                0.0
            };
            corr += chip * y;
            active += y * y;
        }
    }
    Ok(corr / cold_sqrt_f64(active.max(1.0)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::KAUTZ_TOTAL_CHANNELS;
    use crate::provenance::security::extract_stage3_tier_fingerprint_component;
    use proptest::prelude::*;

    #[test]
    fn test_tardos_4buyer_collusion_role_a_exclusion_and_phase_coding() {
        let mut short = [0u8; 1024];
        assert!(apply_isometric_gauge_watermark(&mut short, &[1.0; 48], &[[0.0; 4]; 4]).is_err());
        assert!(extract_structural_hash_224d(&short).is_err());
        assert!(apply_tardos_watermark_7680(&mut short, &[1.0; 7680], 1e-4).is_err());
        assert!(score_tardos_watermark_7680(&short, &short, &[1.0; 7680]).is_err());
        assert!(
            apply_phase_coding_watermark(&mut short, &[0x11; 32], PHASE_CODING_MAX_RAD).is_err()
        );
        assert!(score_phase_coding_arena(&short, &short, &[0x11; 32]).is_err());
        assert!(score_phase_coding_blackbox(&[0.1; 2], &[0.1; 2], &[0x11; 32]).is_err());

        let mut valid_lite = [0u8; 10_752];
        assert!(
            apply_isometric_gauge_watermark(&mut valid_lite, &[1.0; 2], &[[0.0; 4]; 4]).is_err()
        );
        assert!(apply_tardos_watermark_7680(&mut valid_lite, &[1.0; 100], 1e-4).is_err());
        assert!(score_tardos_watermark_7680(&valid_lite, &valid_lite, &[1.0; 100]).is_err());

        let mut master = alloc::vec![0u8; 64_000];
        for i in 0..16_000 {
            let v = (((i * 37) % 101) as f32 - 50.0) * 0.01;
            master[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }

        let mp_secret = [0x5Au8; 32];
        let c1 = generate_tardos_code_7680(&mp_secret, "mdl-1", "buyer-1");
        let c2 = generate_tardos_code_7680(&mp_secret, "mdl-1", "buyer-2");
        let c3 = generate_tardos_code_7680(&mp_secret, "mdl-1", "buyer-3");
        let c4 = generate_tardos_code_7680(&mp_secret, "mdl-1", "buyer-4");
        let c_innocent = generate_tardos_code_7680(&mp_secret, "mdl-1", "buyer-innocent");

        // Verify all 64 Role A indices have chips[i] == 0.0 and accusation_weights[i] == 0.0:
        for &(start, end, _) in &TARDOS_CASCADE_RANGES {
            for idx in start..start + 16 {
                assert_eq!(c1.chips[idx], 0.0);
                assert_eq!(c1.accusation_weights[idx], 0.0);
            }
            for idx in start + 16..end {
                assert!(c1.chips[idx] == 1.0 || c1.chips[idx] == -1.0);
                assert_eq!(c1.chips[idx].signum(), c1.accusation_weights[idx].signum());
            }
        }

        let mut b1 = master.clone();
        let mut b2 = master.clone();
        let mut b3 = master.clone();
        let mut b4 = master.clone();
        apply_tardos_watermark_7680(&mut b1, &c1, 2e-4).unwrap();
        apply_tardos_watermark_7680(&mut b2, &c2, 2e-4).unwrap();
        apply_tardos_watermark_7680(&mut b3, &c3, 2e-4).unwrap();
        apply_tardos_watermark_7680(&mut b4, &c4, 2e-4).unwrap();

        // Verify Role A bytes in b1 are 100% identical to master:
        for &(_, _, byte_base) in &TARDOS_CASCADE_RANGES {
            assert_eq!(
                b1[byte_base..byte_base + 64],
                master[byte_base..byte_base + 64]
            );
        }

        // 1. 4-buyer linear averaging collusion:
        let mut pirated_avg = master.clone();
        // 2. 4-buyer nonlinear majority-vote / interleaving collusion:
        let mut pirated_maj = master.clone();
        for i in 0..16_000 {
            let off = i * 4;
            let (v1, v2, v3, v4) = (
                read_f32_le(&b1, off),
                read_f32_le(&b2, off),
                read_f32_le(&b3, off),
                read_f32_le(&b4, off),
            );
            let avg = 0.25 * (v1 + v2 + v3 + v4);
            write_f32_le(&mut pirated_avg, off, avg);

            let vm = read_f32_le(&master, off);
            let votes = [v1 - vm, v2 - vm, v3 - vm, v4 - vm]
                .iter()
                .fold(0_i32, |acc, &d| {
                    acc + if d > 1e-7 {
                        1
                    } else if d < -1e-7 {
                        -1
                    } else {
                        0
                    }
                });
            let maj = if votes > 0 {
                v1.max(v2).max(v3).max(v4)
            } else if votes < 0 {
                v1.min(v2).min(v3).min(v4)
            } else {
                [v1, v2, v3, v4][i & 3]
            };
            write_f32_le(&mut pirated_maj, off, maj);
        }

        for code in [&c1, &c2, &c3, &c4] {
            let z_avg = score_tardos_watermark_7680(&master, &pirated_avg, code).unwrap();
            assert!(
                z_avg > 12.0,
                "Linear-avg colluder Z-score too low: {z_avg:.2}"
            );
            let z_maj = score_tardos_watermark_7680(&master, &pirated_maj, code).unwrap();
            assert!(
                z_maj > 10.0,
                "Majority-vote colluder Z-score too low: {z_maj:.2}"
            );
        }
        let z_innocent = score_tardos_watermark_7680(&master, &pirated_avg, &c_innocent).unwrap();
        assert!(z_innocent.abs() < 4.5);

        // 3. Verify Layer 2 Phase Coding & 1e-2 rad fingerprint quantization invariance:
        for tier in 0..4 {
            for blk in 0..STAGE3_KAUTZ_BLOCKS_PER_TIER[tier] {
                let base = kautz_soa_byte_offset(tier, blk);
                for lane in 0..4 {
                    write_f32_le(
                        &mut master,
                        base + (8 + lane) * 4,
                        0.15 + 0.05 * (lane as f32),
                    );
                }
            }
        }
        write_f32_le(&mut master, 4_176, 0.85);
        write_f32_le(&mut master, 4_180, 0.35);
        write_f32_le(&mut master, 4_208, 0.85);
        write_f32_le(&mut master, 4_212, 0.65);

        let hash_pre_phase = extract_structural_hash_224d(&master).unwrap();
        let fp_pre_phase = extract_stage3_tier_fingerprint_component(
            &master[4_288..4_416],
            &master[4_416..4_544],
            4,
        )
        .unwrap();

        let mut phase_arena = master.clone();
        let buyer_phase_seed = [0x3C_u8; 32];
        apply_phase_coding_watermark(&mut phase_arena, &buyer_phase_seed, PHASE_CODING_MAX_RAD)
            .unwrap();

        let hash_post_phase = extract_structural_hash_224d(&phase_arena).unwrap();
        let fp_post_phase = extract_stage3_tier_fingerprint_component(
            &phase_arena[4_288..4_416],
            &phase_arena[4_416..4_544],
            4,
        )
        .unwrap();
        assert_eq!(fp_pre_phase, fp_post_phase);
        assert!((1.0 - structural_cosine_similarity(&hash_pre_phase, &hash_post_phase)) < 1e-12);

        let z_phase = score_phase_coding_arena(&master, &phase_arena, &buyer_phase_seed).unwrap();
        assert!(z_phase > 5.0);
        let z_phase_wrong = score_phase_coding_arena(&master, &phase_arena, &[0x99; 32]).unwrap();
        assert!(z_phase_wrong.abs() < 3.5);

        // Extract all 34 pole angles (32 Kautz + 2 Stage II inter-stage) and verify black-box Z-score:
        let mut master_angles_34 = alloc::vec::Vec::with_capacity(34);
        let mut suspect_angles_34 = alloc::vec::Vec::with_capacity(34);
        for tier in 0..4 {
            for blk in 0..STAGE3_KAUTZ_BLOCKS_PER_TIER[tier] {
                let base = kautz_soa_byte_offset(tier, blk);
                for lane in 0..4 {
                    master_angles_34.push(read_f32_le(&master, base + (8 + lane) * 4));
                    suspect_angles_34.push(read_f32_le(&phase_arena, base + (8 + lane) * 4));
                }
            }
        }
        master_angles_34.push(read_f32_le(&master, 4_180));
        master_angles_34.push(read_f32_le(&master, 4_212));
        suspect_angles_34.push(read_f32_le(&phase_arena, 4_180));
        suspect_angles_34.push(read_f32_le(&phase_arena, 4_212));
        let z_bb_34 =
            score_phase_coding_blackbox(&master_angles_34, &suspect_angles_34, &buyer_phase_seed)
                .unwrap();
        assert!((z_bb_34 - z_phase).abs() < 1e-9);

        // Verify inactive/zero Kautz blocks (theta == 0.0) are not clamped into false +1e-4 chips:
        let mut sparse_master = alloc::vec![0u8; 64_000];
        for lane in 0..4 {
            write_f32_le(
                &mut sparse_master,
                4_416 + (8 + lane) * 4,
                0.20 + 0.05 * (lane as f32),
            );
        }
        let mut sparse_suspect = sparse_master.clone();
        apply_phase_coding_watermark(&mut sparse_suspect, &buyer_phase_seed, PHASE_CODING_MAX_RAD)
            .unwrap();
        let z_sparse =
            score_phase_coding_arena(&sparse_master, &sparse_suspect, &buyer_phase_seed).unwrap();
        assert!((z_sparse - 2.0).abs() < 1e-6);

        // 4. Verify EMR-masked Tardos envelopes and apply_cascade_head_gauge composition:
        let gram_diag = [4.0_f64; 256];
        let envs = compute_emr_masked_tardos_envelopes(&gram_diag, [2e-4, 2e-4]);
        assert_eq!(envs[0], 0.0); // Role A
        assert!((envs[16] - 1e-4).abs() < 1e-7); // Role B: 2e-4 / sqrt(4) = 1e-4
        let mut env_arena = master.clone();
        apply_tardos_watermark_7680_envelopes(&mut env_arena, &c1, &envs).unwrap();
        assert!(score_tardos_watermark_7680(&master, &env_arena, &c1).unwrap() > 20.0);

        let mut gauge_arena = alloc::vec![0u8; 64_000];
        for off in (4_928..7_488).step_by(4) {
            write_f32_le(&mut gauge_arena, off, 0.5);
        }
        for off in (49_408..51_456).step_by(4) {
            write_f32_le(&mut gauge_arena, off, 0.25);
        }
        let delta_dyadic = [[1.0_f32; 4]; 32]; // m: 1.0 -> 2.0
        apply_cascade_head_gauge(&mut gauge_arena, 0, &delta_dyadic).unwrap();
        assert_eq!(read_f32_le(&gauge_arena, 4_928), 0.25);
        assert_eq!(read_f32_le(&gauge_arena, 49_408), 0.125);
        // Compose a second dyadic gauge (m: 2.0 -> 4.0) on already non-zero Role B deltas:
        let delta_second = [[2.0_f32; 4]; 32];
        apply_cascade_head_gauge(&mut gauge_arena, 0, &delta_second).unwrap();
        assert_eq!(read_f32_le(&gauge_arena, 4_928), 0.125);
        assert_eq!(read_f32_le(&gauge_arena, 49_408), 0.0625);
        assert!(apply_cascade_head_gauge(&mut gauge_arena, 0, &[[-4.0_f32; 4]; 32]).is_err());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_isometric_gauge_preserves_224d_structural_hash(
            kappa in prop::array::uniform32(0.985f32..1.015f32),
            angle1 in -3.0f32..3.0f32,
            angle2 in -3.0f32..3.0f32,
        ) {
            let mut master = alloc::vec![0u8; 64_000];
            for i in 0..16_000 {
                let v = (((i * 29 + 7) % 97) as f32 - 48.0) * 0.015;
                master[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            master[7_524..7_526].fill(0);
            for tier in 0..4 {
                for blk in 0..STAGE3_KAUTZ_BLOCKS_PER_TIER[tier] {
                    let base = kautz_soa_byte_offset(tier, blk);
                    for lane in 0..4 {
                        write_f32_le(&mut master, base + lane * 4, 0.85);
                        write_f32_le(&mut master, base + (4 + lane) * 4, 0.10);
                        write_f32_le(&mut master, base + (8 + lane) * 4, 0.25);
                        write_f32_le(&mut master, base + (12 + lane) * 4, 0.30);
                        write_f32_le(&mut master, base + (16 + lane) * 4, 0.45);
                        write_f32_le(&mut master, base + (20 + lane) * 4, -0.20);
                        write_f32_le(&mut master, base + (24 + lane) * 4, 0.35);
                        write_f32_le(&mut master, base + (28 + lane) * 4, -0.15);
                    }
                }
            }

            let mut kappa_64 = [1.0f32; KAUTZ_TOTAL_CHANNELS];
            for i in 0..KAUTZ_TOTAL_CHANNELS {
                kappa_64[i] = kappa[i % 32];
            }
            let (c1, s1) = (angle1.cos(), angle1.sin());
            let (c2, s2) = (angle2.cos(), angle2.sin());
            let q = [
                [c1, -s1, 0.0, 0.0],
                [s1, c1, 0.0, 0.0],
                [0.0, 0.0, c2, -s2],
                [0.0, 0.0, s2, c2],
            ];

            let mut watermarked = master.clone();
            apply_isometric_gauge_watermark(&mut watermarked, &kappa_64, &q).unwrap();

            let f_master = extract_structural_hash_224d(&master).unwrap();
            let f_watermarked = extract_structural_hash_224d(&watermarked).unwrap();
            let cos_sim = structural_cosine_similarity(&f_master, &f_watermarked);
            prop_assert!((cos_sim - 1.0).abs() < 1e-11);
        }
    }
}
