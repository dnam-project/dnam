//! Dual-Mechanism Buyer Watermarking (`Test 8H`) & 160-D Gauge-Invariant Structural Hash.

use alloc::vec::Vec;

use crate::schema::DnamError;

use super::chacha20::chacha20_block_64b;
use super::sha256::sha256_digest;
use super::{RGLRU_TOTAL_CHANNELS, STRUCTURAL_HASH_DIM, TARDOS_CONV_WEIGHTS};

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

/// Returns the byte offset of the 128-byte `SelectiveRgLru` SoA struct for `(tier, block_idx)`
/// inside `Arena 0..2`:
/// - `tier == 0` (`Lite`, `ch 0..3`): `128 + block_idx * 1024 + 896`
/// - `tier == 1` (`Full`, `ch 4..7`): `10816 + block_idx * 1792 + 1664`
/// - `tier == 2` (`Ultra`, `ch 8..11`): `22400 + block_idx * 2560 + 2432`
#[inline]
pub const fn rglru_soa_byte_offset(tier: usize, block_idx: usize) -> usize {
    match tier {
        0 => 128 + block_idx * 1024 + 896,
        1 => 10816 + block_idx * 1792 + 1664,
        _ => 22400 + block_idx * 2560 + 2432,
    }
}

/// Applies **Mechanism A (Exact Isometric Gauge Watermark)** in-place on a `36,224 B` `Arena 0..2` buffer:
/// 1. Scales $b_{\gamma,b,c}$ by $\kappa_{b,c} \in [0.985, 1.015]$ and $(c_{\text{real},b,c}, c_{\text{imag},b,c})$
///    by $\kappa_{b,c}^{-1}$ across all 48 RG-LRU channels (`0.0000 dB` audio distortion).
/// 2. Rotates the SVD stereo output head basis $\mathbf{W}_{\text{head},1..4}$ by $\mathbf{Q}_{\text{buyer}}^\top \in SO(4)$
///    and the SVD control subspace projection matrix $\mathbf{P}_{\text{svd}}$ by $\mathbf{Q}_{\text{buyer}}$.
pub fn apply_isometric_gauge_watermark(
    arena: &mut [u8],
    kappa_48: &[f32; RGLRU_TOTAL_CHANNELS],
    q_buyer_4x4: &[[f32; 4]; 4],
) -> Result<(), DnamError> {
    if arena.len() < 36_224 {
        return Err(DnamError::BufferTooSmall {
            expected: 36_224,
            actual: arena.len(),
        });
    }

    // 1. 48-channel RG-LRU reciprocal scaling gauge (b_gamma * kappa, c_real / kappa, c_imag / kappa)
    for tier in 0..3 {
        for blk in 0..4 {
            let base = rglru_soa_byte_offset(tier, blk);
            for lane in 0..4 {
                let ch_global = tier * 16 + blk * 4 + lane;
                let kappa = kappa_48[ch_global];
                let inv_kappa = 1.0 / kappa;

                // In SelectiveRgLru SoA (32 floats = 128 B):
                // [0..4]: r_low, [4..8]: dr_span, [8..12]: theta0_rad, [12..16]: b_gamma_norm,
                // [16..20]: w_sel, [20..24]: c_real, [24..28]: c_imag, [28..32]: u_rot_init
                let b_gamma_off = base + (12 + lane) * 4;
                let c_real_off = base + (20 + lane) * 4;
                let c_imag_off = base + (24 + lane) * 4;

                let b_gamma = read_f32_le(arena, b_gamma_off);
                let c_real = read_f32_le(arena, c_real_off);
                let c_imag = read_f32_le(arena, c_imag_off);

                write_f32_le(arena, b_gamma_off, b_gamma * kappa);
                write_f32_le(arena, c_real_off, c_real * inv_kappa);
                write_f32_le(arena, c_imag_off, c_imag * inv_kappa);
            }
        }
    }

    // 2. Rotate SVD stereo head basis W_head[1..5] (each mode has 2*16*4 = 128 floats = 512 B)
    // across Arena 0 (4224..6784), Arena 1 (17984..20544), and Arena 2 (32640..35200):
    // W_head_buyer[k, i] = sum_{m=0..3} Q_buyer[m][k] * W_head_master[m, i]  (i.e. Q^T @ W_head)
    let head_arena_offsets = [4224_usize, 17984_usize, 32640_usize];
    for &head_base in &head_arena_offsets {
        // Mode 0 (base head W_head,0) is at head_base + 0..512; Modes 1..=4 are at head_base + 512..2560
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

    // 3. Rotate P_svd (16 rows x 4 cols = 64 floats = 256 B) in Arena 0 control_bus_and_clamp_table
    // Global control bus weights start at 7040 + 1024 = 8064; P_svd occupies 8064 + 896 .. 8064 + 1152 (256 B):
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

    Ok(())
}

/// Extracts the **160-Dimensional Gauge-Invariant Structural Vector** $\mathbf{f}_{\text{struct}} \in \mathbb{R}^{160}$
/// directly from a `36,224 B` `Arena 0..2` buffer:
/// - `[0..48]`: 48 RG-LRU invariant composite pole signatures $r_{\text{low},b,c} + 0.5\,\Delta r_{b,c} + 0.1\,\theta_{0,b,c}$,
/// - `[48..96]`: 48 gauge-invariant real products $I_{\text{real},b,c} = b_{\gamma,b,c} \cdot c_{\text{real},b,c}$,
/// - `[96..144]`: 48 gauge-invariant imaginary products $I_{\text{imag},b,c} = b_{\gamma,b,c} \cdot c_{\text{imag},b,c}$,
/// - `[144..152]`: 8 Core 2 `M-UFDN` base delay anchors $\tau_{0,m}$,
/// - `[152..160]`: 8 base stereo output head $\mathbf{W}_{\text{head},0}$ block energy invariants.
pub fn extract_structural_hash_160d(arena: &[u8]) -> Result<[f64; STRUCTURAL_HASH_DIM], DnamError> {
    if arena.len() < 36_224 {
        return Err(DnamError::BufferTooSmall {
            expected: 36_224,
            actual: arena.len(),
        });
    }
    let mut vec160 = [0.0_f64; STRUCTURAL_HASH_DIM];

    for tier in 0..3 {
        for blk in 0..4 {
            let base = rglru_soa_byte_offset(tier, blk);
            for lane in 0..4 {
                let idx = tier * 16 + blk * 4 + lane;
                let r_low = read_f32_le(arena, base + lane * 4) as f64;
                let dr = read_f32_le(arena, base + (4 + lane) * 4) as f64;
                let theta = read_f32_le(arena, base + (8 + lane) * 4) as f64;
                let b_gamma = read_f32_le(arena, base + (12 + lane) * 4) as f64;
                let c_real = read_f32_le(arena, base + (20 + lane) * 4) as f64;
                let c_imag = read_f32_le(arena, base + (24 + lane) * 4) as f64;

                vec160[idx] = r_low + 0.5 * dr + 0.1 * theta;
                vec160[48 + idx] = b_gamma * c_real;
                vec160[96 + idx] = b_gamma * c_imag;
            }
        }
    }

    // 8 M-UFDN base delays (4 in Arena 0 at 6784, 4 in Arena 1 at 20544)
    for lane in 0..4 {
        vec160[144 + lane] = read_f32_le(arena, 6784 + lane * 4) as f64;
        vec160[148 + lane] = read_f32_le(arena, 20544 + lane * 4) as f64;
    }

    // 8 Base stereo head W_head,0 energy norms (unaffected by Q_buyer rotation on modes 1..4)
    for seg in 0..8 {
        let mut energy = 0.0_f64;
        for i in 0..16 {
            let v = read_f32_le(arena, 4224 + (seg * 16 + i) * 4) as f64;
            energy += v * v;
        }
        vec160[152 + seg] = energy;
    }

    Ok(vec160)
}

/// Computes the cosine similarity $\cos(\mathbf{f}_1, \mathbf{f}_2) \in [-1, +1]$ between two 160-D structural hashes.
pub fn structural_cosine_similarity(
    a: &[f64; STRUCTURAL_HASH_DIM],
    b: &[f64; STRUCTURAL_HASH_DIM],
) -> f64 {
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    let mut i = 0;
    while i < STRUCTURAL_HASH_DIM {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
        i += 1;
    }
    let denom = crate::schema::cold_sqrt_f64(na * nb);
    if denom <= 1e-30 { 0.0 } else { dot / denom }
}

/// Visits the byte offsets of all $N_{\text{conv}} = 4,608$ nonlinear dilated convolution `f32` weights
/// across `Arena 0` (`768` floats), `Arena 1` (`1,536` floats), and `Arena 2` (`2,304` floats).
fn for_each_conv_weight_offset<F: FnMut(usize, usize)>(mut f: F) {
    let mut idx = 0_usize;
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
    // Arena 1 (Full Delta): 4 blocks * 1792 B/block starting at 10816; Conv0 is 0..768, Conv1 is 832..1600
    for blk in 0..4 {
        let base = 10816 + blk * 1792;
        for w in 0..192 {
            f(idx, base + w * 4);
            idx += 1;
        }
        for w in 0..192 {
            f(idx, base + 832 + w * 4);
            idx += 1;
        }
    }
    // Arena 2 (Ultra Delta): 4 blocks * 2560 B/block starting at 22400; Conv0 is 0..1152, Conv1 is 1216..2368
    for blk in 0..4 {
        let base = 22400 + blk * 2560;
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

/// Deterministically generates a 4,608-chip Tardos binary spread-spectrum watermark code
/// $\mathbf{c}^{(\text{buyer})} \in \{-1.0, +1.0\}^{4608}$ from `(marketplace_secret, model_id, license_id)`.
pub fn generate_tardos_code_4608(
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

/// Embeds **Mechanism B (4,608-Weight Spread-Spectrum Tardos Watermark)** into `Arena 0..2`:
/// $$W_{\text{conv},i}^{(\text{buyer})} = W_{\text{conv},i}^{(\text{master})} + \epsilon \, c_i^{(\text{buyer})}$$
pub fn apply_tardos_watermark_4608(
    arena: &mut [u8],
    code: &[f32; TARDOS_CONV_WEIGHTS],
    epsilon: f32,
) -> Result<(), DnamError> {
    if arena.len() < 36_224 {
        return Err(DnamError::BufferTooSmall {
            expected: 36_224,
            actual: arena.len(),
        });
    }
    for_each_conv_weight_offset(|idx, byte_off| {
        let w = read_f32_le(arena, byte_off);
        write_f32_le(arena, byte_off, w + epsilon * code[idx]);
    });
    Ok(())
}

/// Computes the Tardos matched-filter Z-score ($\sigma$) of a suspect `Arena 0..2` against
/// `master_arena` and a candidate buyer's `code`:
/// $$Z = \frac{1}{\sqrt{4608}} \sum_{i=0}^{4607} c_i^{(\text{buyer})} \operatorname{sign}\!\big(W_{\text{conv},i}^{(\text{suspect})} - W_{\text{conv},i}^{(\text{master})}\big)$$
pub fn score_tardos_watermark_4608(
    master_arena: &[u8],
    suspect_arena: &[u8],
    code: &[f32; TARDOS_CONV_WEIGHTS],
) -> Result<f64, DnamError> {
    if master_arena.len() < 36_224 || suspect_arena.len() < 36_224 {
        return Err(DnamError::BufferTooSmall {
            expected: 36_224,
            actual: master_arena.len().min(suspect_arena.len()),
        });
    }
    let mut corr = 0.0_f64;
    for_each_conv_weight_offset(|idx, byte_off| {
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
    // 1.0 / sqrt(4608.0) = 1.0 / 67.88225099390857
    Ok(corr * (1.0 / 67.882_250_993_908_57))
}

#[cfg(test)]
mod tests {
    use super::*;
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
