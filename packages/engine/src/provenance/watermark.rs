//! Dual-Mechanism Buyer Watermarking (`Test 8H`) & Prefix-Composable `4 × 56-D = 224-D` Gauge-Invariant Structural Hash.

use alloc::vec::Vec;

use crate::schema::DnamError;

use super::chacha20::chacha20_block_64b;
use super::sha256::sha256_digest;
use super::{STRUCTURAL_HASH_DIM, STRUCTURAL_HASH_TIER_DIM, TARDOS_CONV_WEIGHTS};

#[inline]
fn read_f32_le(arena: &[u8], byte_offset: usize) -> f32 {
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
const fn max_tiers_for_arena_len(arena_len: usize) -> usize {
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

/// Returns the byte offset of the 128-byte `SelectiveRgLru` SoA struct for `(tier, block_idx)`
/// inside `Arena 0..3`:
/// - `tier == 0` (`Lite`, `ch 0..3`): `128 + block_idx * 1024 + 896`
/// - `tier == 1` (`Full`, `ch 4..7`): `10816 + block_idx * 1792 + 1664`
/// - `tier == 2` (`Pro`, `ch 8..11`): `22400 + block_idx * 2560 + 2432`
/// - `tier == 3` (`Ultra`, `ch 12..15`): `36288 + block_idx * 3328 + 3200`
#[inline]
pub const fn rglru_soa_byte_offset(tier: usize, block_idx: usize) -> usize {
    match tier {
        0 => 128 + block_idx * 1024 + 896,
        1 => 10_816 + block_idx * 1792 + 1664,
        2 => 22_400 + block_idx * 2560 + 2432,
        _ => 36_288 + block_idx * 3328 + 3200,
    }
}

/// Applies **Mechanism A (Exact Isometric Gauge Watermark)** in-place on a prefix-composable
/// `Arena 0..3` buffer (`10,752 B`, `22,336 B`, `36,224 B`, or `64,000 B`):
/// 1. Scales $b_{\gamma,b,c}$ by $\kappa_{b,c} \in [0.985, 1.015]$ and $(c_{\text{real},b,c}, c_{\text{imag},b,c})$
///    by $\kappa_{b,c}^{-1}$ across all present RG-LRU channels (`16` per tier, up to `64` in `Ultra`; `0.0000 dB` audio distortion).
/// 2. Rotates the SVD stereo output head basis $\mathbf{W}_{\text{head},1..4}$ (and `Ultra` modes `5..8` when present)
///    by $\mathbf{Q}_{\text{buyer}}^\top \in SO(4)$ and the SVD control subspace projection matrix $\mathbf{P}_{\text{svd}}$
///    (and $\mathbf{P}_{\text{svd,ultra}}$) by $\mathbf{Q}_{\text{buyer}}$.
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
    let required_channels = max_tiers * 16;
    if kappa.len() < required_channels {
        return Err(DnamError::BufferTooSmall {
            expected: required_channels,
            actual: kappa.len(),
        });
    }

    // 1. RG-LRU reciprocal scaling gauge (b_gamma * kappa, c_real / kappa, c_imag / kappa)
    for tier in 0..max_tiers {
        for blk in 0..4 {
            let base = rglru_soa_byte_offset(tier, blk);
            for lane in 0..4 {
                let ch_global = tier * 16 + blk * 4 + lane;
                let k_val = kappa[ch_global];
                let inv_kappa = 1.0 / k_val;

                let b_gamma_off = base + (12 + lane) * 4;
                let c_real_off = base + (20 + lane) * 4;
                let c_imag_off = base + (24 + lane) * 4;

                let b_gamma = read_f32_le(arena, b_gamma_off);
                let c_real = read_f32_le(arena, c_real_off);
                let c_imag = read_f32_le(arena, c_imag_off);

                write_f32_le(arena, b_gamma_off, b_gamma * k_val);
                write_f32_le(arena, c_real_off, c_real * inv_kappa);
                write_f32_le(arena, c_imag_off, c_imag * inv_kappa);
            }
        }
    }

    // 2. Rotate SVD stereo head basis W_head[1..5] (each mode has 2*16*4 = 128 floats = 512 B)
    // across Arena 0 (4224..6784), Arena 1 (17984..20544), Arena 2 (32640..35200), Arena 3 (49600..52160):
    let head_arena_offsets = [4224_usize, 17_984_usize, 32_640_usize, 49_600_usize];
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

    // Also rotate b_head[1..5] in Arena 0 CL1 (bytes 64..104: 5 modes * 2 stereo channels):
    for lr in 0..2 {
        let mut old_b = [0.0_f32; 4];
        for m in 0..4 {
            let off = 64 + ((m + 1) * 2 + lr) * 4;
            old_b[m] = read_f32_le(arena, off);
        }
        for k in 0..4 {
            let mut rotated = 0.0_f32;
            for m in 0..4 {
                rotated += q_buyer_4x4[m][k] * old_b[m];
            }
            let off = 64 + ((k + 1) * 2 + lr) * 4;
            write_f32_le(arena, off, rotated);
        }
    }

    // 3. Rotate P_svd (16 rows x 4 cols = 64 floats = 256 B) in Arena 0 (8960..9216):
    let p_svd_base = 8960_usize;
    for row in 0..16 {
        let mut old_row = [0.0_f32; 4];
        for m in 0..4 {
            old_row[m] = read_f32_le(arena, p_svd_base + (row * 4 + m) * 4);
        }
        for k in 0..4 {
            let mut rotated = 0.0_f32;
            for m in 0..4 {
                rotated += old_row[m] * q_buyer_4x4[m][k];
            }
            write_f32_le(arena, p_svd_base + (row * 4 + k) * 4, rotated);
        }
    }

    // 4. When Arena 3 (Tier::Ultra, 64,000 B) is present, also rotate modes 5..8 (52,160..60,352),
    //    b_head_modes_5_to_8 (63,424..63,456), and P_svd_ultra (63,168..63,424):
    if max_tiers >= 4 {
        for tier in 0..4 {
            let ultra_mode_base = 52_160 + tier * 2048;
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
        let p_svd_ultra_base = 63_168_usize;
        for row in 0..16 {
            let mut old_row = [0.0_f32; 4];
            for m in 0..4 {
                old_row[m] = read_f32_le(arena, p_svd_ultra_base + (row * 4 + m) * 4);
            }
            for k in 0..4 {
                let mut rotated = 0.0_f32;
                for m in 0..4 {
                    rotated += old_row[m] * q_buyer_4x4[m][k];
                }
                write_f32_le(arena, p_svd_ultra_base + (row * 4 + k) * 4, rotated);
            }
        }
    }

    Ok(())
}

/// Extracts the **Prefix-Composable `4 × 56-D = 224-D` Gauge-Invariant Structural Vector**
/// $\mathbf{f}_{\text{struct}} \in \mathbb{R}^{224}$ directly from an `Arena 0..3` buffer (`>= 10,752 B`).
///
/// Ordered strictly by tier $t \in \{0, 1, 2, 3\}$ (`[56*t .. 56*(t+1)]`):
/// - `[0..16]`: 16 RG-LRU invariant composite pole signatures $r_{\text{low},b,c} + 0.5\,\Delta r_{b,c} + 0.1\,\theta_{0,b,c}$,
/// - `[16..32]`: 16 gauge-invariant real products $I_{\text{real},b,c} = b_{\gamma,b,c} \cdot c_{\text{real},b,c}$,
/// - `[32..48]`: 16 gauge-invariant imaginary products $I_{\text{imag},b,c} = b_{\gamma,b,c} \cdot c_{\text{imag},b,c}$,
/// - `[48..52]`: 4 Core 2 `M-UFDN` delay invariants for tier $t$,
/// - `[52..56]`: 4 base stereo output head $\mathbf{W}_{\text{head},0}$ block energy invariants for tier $t$.
///
/// Any stripped higher tier remains zero-filled, enabling exact prefix cosine similarity across tier-stripped containers.
pub fn extract_structural_hash_224d(arena: &[u8]) -> Result<[f64; STRUCTURAL_HASH_DIM], DnamError> {
    let max_tiers = max_tiers_for_arena_len(arena.len());
    if max_tiers == 0 {
        return Err(DnamError::BufferTooSmall {
            expected: 10_752,
            actual: arena.len(),
        });
    }
    let mut vec224 = [0.0_f64; STRUCTURAL_HASH_DIM];
    let head_arena_offsets = [4224_usize, 17_984_usize, 32_640_usize, 49_600_usize];

    for tier in 0..max_tiers {
        let t_base = tier * STRUCTURAL_HASH_TIER_DIM;
        for blk in 0..4 {
            let base = rglru_soa_byte_offset(tier, blk);
            for lane in 0..4 {
                let ch_in_tier = blk * 4 + lane;
                let r_low = read_f32_le(arena, base + lane * 4) as f64;
                let dr = read_f32_le(arena, base + (4 + lane) * 4) as f64;
                let theta = read_f32_le(arena, base + (8 + lane) * 4) as f64;
                let b_gamma = read_f32_le(arena, base + (12 + lane) * 4) as f64;
                let c_real = read_f32_le(arena, base + (20 + lane) * 4) as f64;
                let c_imag = read_f32_le(arena, base + (24 + lane) * 4) as f64;

                vec224[t_base + ch_in_tier] = r_low + 0.5 * dr + 0.1 * theta;
                vec224[t_base + 16 + ch_in_tier] = b_gamma * c_real;
                vec224[t_base + 32 + ch_in_tier] = b_gamma * c_imag;
            }
        }

        // 4 Core 2 delay invariants for this tier
        let delay_base = match tier {
            0 => 6784_usize,
            1 => 20_544_usize,
            2 => 35_200_usize,
            _ => 62_400_usize,
        };
        for lane in 0..4 {
            vec224[t_base + 48 + lane] = read_f32_le(arena, delay_base + lane * 4) as f64;
        }

        // 4 Base stereo head W_head,0 energy norms for this tier (32 floats per segment × 4 segments = 128 floats)
        let head_base = head_arena_offsets[tier];
        for seg in 0..4 {
            let mut energy = 0.0_f64;
            for i in 0..32 {
                let v = read_f32_le(arena, head_base + (seg * 32 + i) * 4) as f64;
                energy += v * v;
            }
            vec224[t_base + 52 + seg] = energy;
        }
    }

    Ok(vec224)
}

/// Alias for [`extract_structural_hash_224d`].
#[inline]
pub fn extract_structural_hash_160d(arena: &[u8]) -> Result<[f64; STRUCTURAL_HASH_DIM], DnamError> {
    extract_structural_hash_224d(arena)
}

#[inline]
fn populated_tier_count(v: &[f64; STRUCTURAL_HASH_DIM]) -> usize {
    let mut tiers = 1_usize;
    for t in 1..4 {
        let slice = &v[t * STRUCTURAL_HASH_TIER_DIM..(t + 1) * STRUCTURAL_HASH_TIER_DIM];
        if slice.iter().any(|&x| x.abs() > 1e-30) {
            tiers = t + 1;
        }
    }
    tiers
}

/// Computes the prefix-composable cosine similarity $\cos(\mathbf{f}_1, \mathbf{f}_2) \in [-1, +1]$
/// over the shared populated tier prefix (`56 * min(tiers_a, tiers_b)` dimensions).
pub fn structural_cosine_similarity(
    a: &[f64; STRUCTURAL_HASH_DIM],
    b: &[f64; STRUCTURAL_HASH_DIM],
) -> f64 {
    let shared_tiers = populated_tier_count(a).min(populated_tier_count(b));
    let active_dim = shared_tiers * STRUCTURAL_HASH_TIER_DIM;
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    let mut i = 0;
    while i < active_dim {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
        i += 1;
    }
    let denom = crate::schema::cold_sqrt_f64(na * nb);
    if denom <= 1e-30 { 0.0 } else { dot / denom }
}

/// Visits the byte offsets of all nonlinear dilated convolution `f32` weights present in `arena_len`:
/// - `Arena 0` (`Lite`, `10,752 B`): `768` weights (`0..768`)
/// - `Arena 1` (`Full`, `22,336 B`): `+1,536` weights (`768..2,304`)
/// - `Arena 2` (`Pro`, `36,224 B`): `+2,304` weights (`2,304..4,608`)
/// - `Arena 3` (`Ultra`, `64,000 B`): `+3,072` weights (`4,608..7,680`)
fn for_each_conv_weight_offset_up_to<F: FnMut(usize, usize)>(arena_len: usize, mut f: F) -> usize {
    let max_tiers = max_tiers_for_arena_len(arena_len);
    let mut idx = 0_usize;
    if max_tiers >= 1 {
        // Arena 0 (Lite): 4 blocks * 1024 B/block starting at 128; Conv0 is 0..384, Conv1 is 448..832
        for blk in 0..4 {
            let base = 128 + blk * 1024;
            for w in 0..96 {
                f(idx, base + w * 4);
                idx += 1;
            }
            for w in 0..96 {
                f(idx, base + 448 + w * 4);
                idx += 1;
            }
        }
    }
    if max_tiers >= 2 {
        // Arena 1 (Full Delta): 4 blocks * 1792 B/block starting at 10816; Conv0 is 0..768, Conv1 is 832..1600
        for blk in 0..4 {
            let base = 10_816 + blk * 1792;
            for w in 0..192 {
                f(idx, base + w * 4);
                idx += 1;
            }
            for w in 0..192 {
                f(idx, base + 832 + w * 4);
                idx += 1;
            }
        }
    }
    if max_tiers >= 3 {
        // Arena 2 (Pro Delta): 4 blocks * 2560 B/block starting at 22400; Conv0 is 0..1152, Conv1 is 1216..2368
        for blk in 0..4 {
            let base = 22_400 + blk * 2560;
            for w in 0..288 {
                f(idx, base + w * 4);
                idx += 1;
            }
            for w in 0..288 {
                f(idx, base + 1216 + w * 4);
                idx += 1;
            }
        }
    }
    if max_tiers >= 4 {
        // Arena 3 (Ultra Delta): 4 blocks * 3328 B/block starting at 36288; Conv0 is 0..1536, Conv1 is 1600..3136
        for blk in 0..4 {
            let base = 36_288 + blk * 3328;
            for w in 0..384 {
                f(idx, base + w * 4);
                idx += 1;
            }
            for w in 0..384 {
                f(idx, base + 1600 + w * 4);
                idx += 1;
            }
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
) -> [f32; TARDOS_CONV_WEIGHTS] {
    let mut seed_input = Vec::with_capacity(64 + model_id.len() + license_id.len());
    seed_input.extend_from_slice(b"DNAM-TARDOS-V1:");
    seed_input.extend_from_slice(marketplace_secret);
    seed_input.extend_from_slice(model_id.as_bytes());
    seed_input.push(b':');
    seed_input.extend_from_slice(license_id.as_bytes());
    let buyer_seed = sha256_digest(&seed_input);
    let nonce = [0u8; 12];

    let mut code = [1.0_f32; TARDOS_CONV_WEIGHTS];
    let mut bit_idx = 0_usize;
    let mut blk_counter = 0_u32;

    while bit_idx < TARDOS_CONV_WEIGHTS {
        let stream = chacha20_block_64b(&buyer_seed, &nonce, blk_counter);
        blk_counter += 1;
        for &byte in &stream {
            for b in 0..8 {
                if bit_idx < TARDOS_CONV_WEIGHTS {
                    code[bit_idx] = if ((byte >> b) & 1) == 1 { 1.0 } else { -1.0 };
                    bit_idx += 1;
                }
            }
        }
    }
    code
}

/// Alias for [`generate_tardos_code_7680`].
#[inline]
pub fn generate_tardos_code_4608(
    marketplace_secret: &[u8; 32],
    model_id: &str,
    license_id: &str,
) -> [f32; TARDOS_CONV_WEIGHTS] {
    generate_tardos_code_7680(marketplace_secret, model_id, license_id)
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
        _ => 7680,
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

/// Alias for [`apply_tardos_watermark_7680`].
#[inline]
pub fn apply_tardos_watermark_4608(
    arena: &mut [u8],
    code: &[f32],
    epsilon: f32,
) -> Result<(), DnamError> {
    apply_tardos_watermark_7680(arena, code, epsilon)
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
        _ => 7680,
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

/// Alias for [`score_tardos_watermark_7680`].
#[inline]
pub fn score_tardos_watermark_4608(
    master_arena: &[u8],
    suspect_arena: &[u8],
    code: &[f32],
) -> Result<f64, DnamError> {
    score_tardos_watermark_7680(master_arena, suspect_arena, code)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::RGLRU_TOTAL_CHANNELS;
    use proptest::prelude::*;

    #[test]
    fn test_tardos_4buyer_collusion_and_short_buffer_errors() {
        let mut short = [0u8; 1024];
        assert!(apply_isometric_gauge_watermark(&mut short, &[1.0; 48], &[[0.0; 4]; 4]).is_err());
        assert!(extract_structural_hash_160d(&short).is_err());
        assert!(apply_tardos_watermark_4608(&mut short, &[1.0; 4608], 1e-4).is_err());
        assert!(score_tardos_watermark_4608(&short, &short, &[1.0; 4608]).is_err());

        let mut master = alloc::vec![0u8; 36_224];
        for i in 0..9_056 {
            let v = (((i * 37) % 101) as f32 - 50.0) * 0.01;
            master[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }

        let mp_secret = [0x5Au8; 32];
        let c1 = generate_tardos_code_4608(&mp_secret, "mdl-1", "buyer-1");
        let c2 = generate_tardos_code_4608(&mp_secret, "mdl-1", "buyer-2");
        let c3 = generate_tardos_code_4608(&mp_secret, "mdl-1", "buyer-3");
        let c4 = generate_tardos_code_4608(&mp_secret, "mdl-1", "buyer-4");
        let c_innocent = generate_tardos_code_4608(&mp_secret, "mdl-1", "buyer-innocent");

        let mut b1 = master.clone();
        let mut b2 = master.clone();
        let mut b3 = master.clone();
        let mut b4 = master.clone();
        apply_tardos_watermark_4608(&mut b1, &c1, 2e-4).unwrap();
        apply_tardos_watermark_4608(&mut b2, &c2, 2e-4).unwrap();
        apply_tardos_watermark_4608(&mut b3, &c3, 2e-4).unwrap();
        apply_tardos_watermark_4608(&mut b4, &c4, 2e-4).unwrap();

        let mut pirated = master.clone();
        for i in 0..9_056 {
            let off = i * 4;
            let avg = 0.25
                * (read_f32_le(&b1, off)
                    + read_f32_le(&b2, off)
                    + read_f32_le(&b3, off)
                    + read_f32_le(&b4, off));
            write_f32_le(&mut pirated, off, avg);
        }

        for code in [&c1, &c2, &c3, &c4] {
            let z = score_tardos_watermark_4608(&master, &pirated, code).unwrap();
            assert!(z > 20.0, "Colluder Z-score too low: {z:.2}");
        }
        let z_innocent = score_tardos_watermark_4608(&master, &pirated, &c_innocent).unwrap();
        assert!(z_innocent.abs() < 4.5);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_isometric_gauge_preserves_160d_structural_hash(
            kappa in prop::array::uniform32(0.985f32..1.015f32),
            angle1 in -3.0f32..3.0f32,
            angle2 in -3.0f32..3.0f32,
        ) {
            let mut master = alloc::vec![0u8; 36_224];
            for i in 0..9_056 {
                let v = (((i * 29 + 7) % 97) as f32 - 48.0) * 0.015;
                master[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            for tier in 0..3 {
                for blk in 0..4 {
                    let base = rglru_soa_byte_offset(tier, blk);
                    for lane in 0..4 {
                        write_f32_le(&mut master, base + lane * 4, 0.85);
                        write_f32_le(&mut master, base + (4 + lane) * 4, 0.10);
                        write_f32_le(&mut master, base + (8 + lane) * 4, 0.25);
                        write_f32_le(&mut master, base + (12 + lane) * 4, 0.30);
                        write_f32_le(&mut master, base + (20 + lane) * 4, 0.45);
                        write_f32_le(&mut master, base + (24 + lane) * 4, -0.20);
                    }
                }
            }

            let mut kappa_48 = [1.0f32; RGLRU_TOTAL_CHANNELS];
            for i in 0..RGLRU_TOTAL_CHANNELS {
                kappa_48[i] = kappa[i % 32];
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
            apply_isometric_gauge_watermark(&mut watermarked, &kappa_48, &q).unwrap();

            let f_master = extract_structural_hash_160d(&master).unwrap();
            let f_watermarked = extract_structural_hash_160d(&watermarked).unwrap();
            let cos_sim = structural_cosine_similarity(&f_master, &f_watermarked);
            prop_assert!((cos_sim - 1.0).abs() < 1e-11);
        }
    }
}
