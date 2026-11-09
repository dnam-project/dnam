//! 224-D gauge-invariant structural fingerprinting and cosine similarity.
//!
//! Canonical Reference: `docs/security/watermarking_and_marketplace.md`.

use crate::schema::DnamError;

use super::watermark::{
    STAGE3_KAUTZ_BLOCKS_PER_TIER, kautz_soa_byte_offset, max_tiers_for_arena_len, read_f32_le,
};
use super::{STRUCTURAL_HASH_DIM, STRUCTURAL_HASH_TIER_DIM};

#[inline]
pub(crate) fn quantize_pole_angle_1e2(theta: f64) -> f64 {
    let scaled = theta * 100.0;
    let rounded = (scaled + if scaled >= 0.0 { 0.5 } else { -0.5 }) as i64;
    (rounded as f64) * 0.01
}

/// Extracts the 224-dimensional prefix-composable gauge-invariant structural hash vector $\mathbf{f}_{\text{struct}} \in \mathbb{R}^{224}$.
///
/// Ordered strictly by tier $t \in \{0, 1, 2, 3\}$ (`[56*t .. 56*(t+1)]`):
/// - `[0..16]`: modal/Kautz invariant composite pole signatures $r_{\text{low},b,c} + 0.5\,\Delta r_{b,c} + 0.1\,\lfloor\theta_{0,b,c}\rceil_{10^{-2}}$,
/// - `[16..32]`: gauge-invariant real products $I_{\text{real},b,c} = b_{\gamma,b,c} \cdot (c_{\text{real},L,b,c} + c_{\text{real},R,b,c})$,
/// - `[32..48]`: gauge-invariant imaginary products $I_{\text{imag},b,c} = b_{\gamma,b,c} \cdot (c_{\text{imag},L,b,c} + c_{\text{imag},R,b,c})$,
/// - `[48..52]`: 4 Stage IV Spatial Tank delay invariants for tier $t$,
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
    let head_arena_offsets = [4_928_usize, 18_048_usize, 31_936_usize, 57_600_usize];

    for tier in 0..max_tiers {
        let t_base = tier * STRUCTURAL_HASH_TIER_DIM;
        let blocks = STAGE3_KAUTZ_BLOCKS_PER_TIER[tier];
        for blk in 0..blocks {
            let base = kautz_soa_byte_offset(tier, blk);
            for lane in 0..4 {
                let ch_in_tier = blk * 4 + lane;
                let r_low = read_f32_le(arena, base + lane * 4) as f64;
                let dr = read_f32_le(arena, base + (4 + lane) * 4) as f64;
                let theta =
                    quantize_pole_angle_1e2(read_f32_le(arena, base + (8 + lane) * 4) as f64);
                let b_gamma = read_f32_le(arena, base + (12 + lane) * 4) as f64;
                let c_re_l = read_f32_le(arena, base + (16 + lane) * 4) as f64;
                let c_im_l = read_f32_le(arena, base + (20 + lane) * 4) as f64;
                let c_re_r = read_f32_le(arena, base + (24 + lane) * 4) as f64;
                let c_im_r = read_f32_le(arena, base + (28 + lane) * 4) as f64;

                vec224[t_base + ch_in_tier] = r_low + 0.5 * dr + 0.1 * theta;
                vec224[t_base + 16 + ch_in_tier] = b_gamma * (c_re_l + c_re_r);
                vec224[t_base + 32 + ch_in_tier] = b_gamma * (c_im_l + c_im_r);
            }
        }

        // 4 Stage IV Spatial Tank delay invariants for this tier
        let delay_base = match tier {
            0 => 4_672_usize,
            1 => 17_792_usize,
            2 => 34_496_usize,
            _ => 48_896_usize,
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

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_structural_cosine_similarity_scale_and_prefix_invariance(
            base_val in 0.05_f64..10.0_f64,
            scale in 0.1_f64..20.0_f64,
            extra_val in 0.05_f64..10.0_f64,
        ) {
            let mut f_lite = [0.0_f64; STRUCTURAL_HASH_DIM];
            let mut f_full = [0.0_f64; STRUCTURAL_HASH_DIM];
            for i in 0..STRUCTURAL_HASH_TIER_DIM {
                f_lite[i] = base_val * ((i + 1) as f64);
                f_full[i] = f_lite[i] * scale;
                f_full[STRUCTURAL_HASH_TIER_DIM + i] = extra_val * ((i + 1) as f64);
            }
            // Prefix-composable cosine similarity between Lite (1 tier) and scaled Full (2 tiers) must be 1.0
            let sim = structural_cosine_similarity(&f_lite, &f_full);
            prop_assert!((sim - 1.0).abs() < 1e-12, "Expected 1.0, got {}", sim);
        }
    }
}
