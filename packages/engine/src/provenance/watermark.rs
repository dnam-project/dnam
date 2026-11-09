//! Dual-Mechanism Buyer Watermarking (`Test 8H`) & Prefix-Composable `4 × 56-D = 224-D` Gauge-Invariant Structural Hash.

use alloc::vec::Vec;

use crate::schema::DnamError;

use super::TARDOS_CASCADE_WEIGHTS;
use super::chacha20::chacha20_block_64b;
use super::sha256::sha256_digest;

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
fn write_f32_le(arena: &mut [u8], byte_offset: usize, val: f32) {
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

/// Visits the byte offsets of all Stage II commuted nonlinear cascade projection weights (`stage2_cascade_w`)
/// across `arena_len`:
/// - `Arena 0` (`Lite`, `10,752 B`): `768` weights (`0..768`) at `1,024..4,096`
/// - `Arena 1` (`Full`, `22,336 B`): `+1,536` weights (`768..2,304`) at `11,264..17,408`
/// - `Arena 2` (`Pro`, `36,224 B`): `+2,304` weights (`2,304..4,608`) at `22,336..31,552`
/// - `Arena 3` (`Ultra`, `64,000 B`): `+3,072` weights (`4,608..7,680`) at `36,224..48,512`
fn for_each_conv_weight_offset_up_to<F: FnMut(usize, usize)>(arena_len: usize, mut f: F) -> usize {
    let max_tiers = max_tiers_for_arena_len(arena_len);
    let mut idx = 0_usize;
    if max_tiers >= 1 {
        // Arena 0 (Lite): stage2_cascade_w [3, 16, 4, 4] = 768 f32 words starting at 1024
        let base = 1_024_usize;
        for w in 0..768 {
            f(idx, base + w * 4);
            idx += 1;
        }
    }
    if max_tiers >= 2 {
        // Arena 1 (Full Delta): stage2_cascade_w_delta [3, 16, 8, 4] = 1536 f32 words starting at 11264
        let base = 11_264_usize;
        for w in 0..1536 {
            f(idx, base + w * 4);
            idx += 1;
        }
    }
    if max_tiers >= 3 {
        // Arena 2 (Pro Delta): stage2_cascade_w_delta [3, 16, 12, 4] = 2304 f32 words starting at 22336
        let base = 22_336_usize;
        for w in 0..2304 {
            f(idx, base + w * 4);
            idx += 1;
        }
    }
    if max_tiers >= 4 {
        // Arena 3 (Ultra Delta): stage2_cascade_w_delta [3, 16, 16, 4] = 3072 f32 words starting at 36224
        let base = 36_224_usize;
        for w in 0..3072 {
            f(idx, base + w * 4);
            idx += 1;
        }
    }
    idx
}

/// Deterministically generates a 7,680-chip Tardos binary spread-spectrum watermark code
/// $\mathbf{c}^{(\text{buyer})} \in \{-1.0, +1.0\}^{7680}$ from `(marketplace_secret, model_id, license_id)`.
pub fn generate_tardos_code_7680(
    marketplace_secret: &[u8; 32],
    model_id: &str,
    license_id: &str,
) -> [f32; TARDOS_CASCADE_WEIGHTS] {
    let mut seed_input = Vec::with_capacity(64 + model_id.len() + license_id.len());
    seed_input.extend_from_slice(b"DNAM-TARDOS-V1:");
    seed_input.extend_from_slice(marketplace_secret);
    seed_input.extend_from_slice(model_id.as_bytes());
    seed_input.push(b':');
    seed_input.extend_from_slice(license_id.as_bytes());
    let buyer_seed = sha256_digest(&seed_input);
    let nonce = [0u8; 12];

    let mut code = [1.0_f32; TARDOS_CASCADE_WEIGHTS];
    let mut bit_idx = 0_usize;
    let mut blk_counter = 0_u32;

    while bit_idx < TARDOS_CASCADE_WEIGHTS {
        let stream = chacha20_block_64b(&buyer_seed, &nonce, blk_counter);
        blk_counter += 1;
        for &byte in &stream {
            for b in 0..8 {
                if bit_idx < TARDOS_CASCADE_WEIGHTS {
                    code[bit_idx] = if ((byte >> b) & 1) == 1 { 1.0 } else { -1.0 };
                    bit_idx += 1;
                }
            }
        }
    }
    code
}

/// Embeds **Mechanism B (Prefix-Composable Spread-Spectrum Tardos Watermark)** into `Arena 0..3`
/// (`768`, `2,304`, `4,608`, or `7,680` weights depending on container tier):
/// $$W_{\text{conv},i}^{(\text{buyer})} = W_{\text{conv},i}^{(\text{master})} + \epsilon \, c_i^{(\text{buyer})}$$
pub fn apply_tardos_watermark_7680(
    arena: &mut [u8],
    code: &[f32],
    epsilon: f32,
) -> Result<(), DnamError> {
    let max_tiers = max_tiers_for_arena_len(arena.len());
    if max_tiers == 0 {
        return Err(DnamError::BufferTooSmall {
            expected: 10_752,
            actual: arena.len(),
        });
    }
    let needed_chips = match max_tiers {
        1 => 768,
        2 => 2304,
        3 => 4608,
        _ => TARDOS_CASCADE_WEIGHTS,
    };
    if code.len() < needed_chips {
        return Err(DnamError::BufferTooSmall {
            expected: needed_chips,
            actual: code.len(),
        });
    }
    for_each_conv_weight_offset_up_to(arena.len(), |idx, byte_off| {
        let w = read_f32_le(arena, byte_off);
        write_f32_le(arena, byte_off, w + epsilon * code[idx]);
    });
    Ok(())
}

/// Computes the Tardos matched-filter Z-score ($\sigma$) of a suspect `Arena 0..3` (including
/// tier-stripped containers down to `10,752 B` `Lite`) against `master_arena` and a candidate buyer's `code`:
/// $$Z = \frac{1}{\sqrt{N_{\text{prefix}}}} \sum_{i=0}^{N_{\text{prefix}}-1} c_i^{(\text{buyer})} \operatorname{sign}\!\big(W_{\text{conv},i}^{(\text{suspect})} - W_{\text{conv},i}^{(\text{master})}\big)$$
pub fn score_tardos_watermark_7680(
    master_arena: &[u8],
    suspect_arena: &[u8],
    code: &[f32],
) -> Result<f64, DnamError> {
    let shared_len = master_arena.len().min(suspect_arena.len());
    let max_tiers = max_tiers_for_arena_len(shared_len);
    if max_tiers == 0 {
        return Err(DnamError::BufferTooSmall {
            expected: 10_752,
            actual: shared_len,
        });
    }
    let needed_chips = match max_tiers {
        1 => 768,
        2 => 2304,
        3 => 4608,
        _ => TARDOS_CASCADE_WEIGHTS,
    };
    if code.len() < needed_chips {
        return Err(DnamError::BufferTooSmall {
            expected: needed_chips,
            actual: code.len(),
        });
    }
    let mut corr = 0.0_f64;
    let count = for_each_conv_weight_offset_up_to(shared_len, |idx, byte_off| {
        let w_m = read_f32_le(master_arena, byte_off);
        let w_s = read_f32_le(suspect_arena, byte_off);
        let diff = w_s - w_m;
        let sign = if diff > 0.0 {
            1.0_f64
        } else if diff < 0.0 {
            -1.0_f64
        } else {
            0.0_f64
        };
        corr += (code[idx] as f64) * sign;
    });
    Ok(corr / crate::schema::cold_sqrt_f64(count as f64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::KAUTZ_TOTAL_CHANNELS;
    use proptest::prelude::*;

    #[test]
    fn test_tardos_4buyer_collusion_and_short_buffer_errors() {
        let mut short = [0u8; 1024];
        assert!(apply_isometric_gauge_watermark(&mut short, &[1.0; 48], &[[0.0; 4]; 4]).is_err());
        assert!(extract_structural_hash_224d(&short).is_err());
        assert!(apply_tardos_watermark_7680(&mut short, &[1.0; 7680], 1e-4).is_err());
        assert!(score_tardos_watermark_7680(&short, &short, &[1.0; 7680]).is_err());

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

        let mut b1 = master.clone();
        let mut b2 = master.clone();
        let mut b3 = master.clone();
        let mut b4 = master.clone();
        apply_tardos_watermark_7680(&mut b1, &c1, 2e-4).unwrap();
        apply_tardos_watermark_7680(&mut b2, &c2, 2e-4).unwrap();
        apply_tardos_watermark_7680(&mut b3, &c3, 2e-4).unwrap();
        apply_tardos_watermark_7680(&mut b4, &c4, 2e-4).unwrap();

        let mut pirated = master.clone();
        for i in 0..16_000 {
            let off = i * 4;
            let avg = 0.25
                * (read_f32_le(&b1, off)
                    + read_f32_le(&b2, off)
                    + read_f32_le(&b3, off)
                    + read_f32_le(&b4, off));
            write_f32_le(&mut pirated, off, avg);
        }

        for code in [&c1, &c2, &c3, &c4] {
            let z = score_tardos_watermark_7680(&master, &pirated, code).unwrap();
            assert!(z > 20.0, "Colluder Z-score too low: {z:.2}");
        }
        let z_innocent = score_tardos_watermark_7680(&master, &pirated, &c_innocent).unwrap();
        assert!(z_innocent.abs() < 4.5);
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
