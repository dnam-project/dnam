//! Layer 3 Two-Seed Arcsine Tardos Traitor-Tracing Codebook (`TardosCodebook7680`),
//! Coordinate-Space `EMR`-Masked Envelopes, and Standalone Stage II Role B $\leftrightarrow$ `W_head`
//! Bilinear Gauge (`packages/engine/src/provenance/watermark/tardos.rs`).
//!
//! Canonical Reference: `docs/specs/security/watermarking_and_marketplace.md` (§1.1–§1.3).

use core::ops::Deref;

use crate::provenance::TARDOS_CASCADE_WEIGHTS;
use crate::provenance::chacha20::chacha20_block_64b;
use crate::provenance::security::{WatermarkSeedScope, derive_watermark_seed_hmac_sha256};
use crate::provenance::sha256::sha256_digest;
use crate::schema::{DnamError, cold_sqrt_f64};

use super::{max_tiers_for_arena_len, read_f32_le, write_f32_le};

/// Cumulative global float index boundaries `[start, end)` of `stage2_cascade_w` across `Arena 0..3`:
/// - `Tier 0` (`Lite`): `0..768` at `1_024..4_096`
/// - `Tier 1` (`Full`): `768..2304` at `11_264..17_408`
/// - `Tier 2` (`Pro`): `2304..4608` at `22_336..31_552`
/// - `Tier 3` (`Ultra`): `4608..7680` at `36_224..48_512`
pub const TARDOS_CASCADE_RANGES: [(usize, usize, usize); 4] = [
    (0, 768, 1_024),
    (768, 2_304, 11_264),
    (2_304, 4_608, 22_336),
    (4_608, 7_680, 36_224),
];

/// Returns `true` if `global_idx` (`0..7680`) belongs to the `64` Role A coordinates
/// (`word_in_tier < 16` in each tier's `CASCADE_W_BASES[t]`: `0..16`, `768..784`, `2304..2320`, `4608..4624`).
///
/// Role A coordinates feed the `48 kHz` Stage II inter-stage `L^{(1)}` / `L^{(2)}` nonlinear coupling
/// and are strictly excluded from Layer 3 Tardos perturbation (`chips[i] == 0.0`, `accusation_weights[i] == 0.0`,
/// `epsilon_i == 0.0`).
#[inline]
pub const fn is_role_a_cascade_index(global_idx: usize) -> bool {
    global_idx < 16
        || (global_idx >= 768 && global_idx < 784)
        || (global_idx >= 2_304 && global_idx < 2_320)
        || (global_idx >= 4_608 && global_idx < 4_624)
}

/// Visits the `(global_idx, byte_offset)` of all Stage II commuted nonlinear cascade projection weights
/// (`stage2_cascade_w`) present in `arena_len`.
pub(crate) fn for_each_conv_weight_offset_up_to<F: FnMut(usize, usize)>(
    arena_len: usize,
    mut f: F,
) -> usize {
    let max_tiers = max_tiers_for_arena_len(arena_len);
    let mut count = 0_usize;
    for &(start, end, byte_base) in TARDOS_CASCADE_RANGES.iter().take(max_tiers) {
        let len = end - start;
        for w in 0..len {
            f(start + w, byte_base + w * 4);
            count += 1;
        }
    }
    count
}

/// Maps a Role B `stage2_cascade_w` global float index `global_idx` (`0..7680`) to its `(tier, slot_32, lane_4)`
/// target in Tier `tier`'s collapsed head weights (`slot_32 = 16 * c + g = (4 * r + q) % 32`).
/// Returns `None` if `global_idx` is a Role A index (`word_in_tier < 16`) or out of range.
#[inline]
pub const fn role_b_head_slot_for_index(global_idx: usize) -> Option<(usize, usize, usize)> {
    let (tier, tier_start) = if global_idx < 768 {
        (0, 0)
    } else if global_idx < 2_304 {
        (1, 768)
    } else if global_idx < 4_608 {
        (2, 2_304)
    } else if global_idx < TARDOS_CASCADE_WEIGHTS {
        (3, 4_608)
    } else {
        return None;
    };
    let word_in_tier = global_idx - tier_start;
    if word_in_tier < 16 {
        return None;
    }
    let role_b_word = word_in_tier - 16;
    let vec4_idx = role_b_word / 4; // 4 * r + q
    let lane = role_b_word % 4;
    let slot_32 = vec4_idx % 32;
    Some((tier, slot_32, lane))
}

/// Two-Seed Arcsine Tardos Traitor-Tracing Codebook (`7,680` coordinates across `Arena 0..3`:
/// `64` Role A coordinates zeroed + `7,616` active Role B coordinates, `ref[sec.watermark.tardos-code]`).
#[derive(Clone, Debug, PartialEq)]
pub struct TardosCodebook7680 {
    /// Ternary embedding chips (`0.0` on the `64` Role A coordinates, `±1.0` on the `7,616` Role B coordinates).
    pub chips: [f32; TARDOS_CASCADE_WEIGHTS],
    /// Symmetric Škorić–Tardos accusation weights (`0.0` on Role A, `+sqrt((1-p_i)/p_i)` or `-sqrt(p_i/(1-p_i))` on Role B).
    pub accusation_weights: [f32; TARDOS_CASCADE_WEIGHTS],
}

impl Deref for TardosCodebook7680 {
    type Target = [f32];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.accusation_weights
    }
}

impl AsRef<[f32]> for TardosCodebook7680 {
    #[inline]
    fn as_ref(&self) -> &[f32] {
        &self.accusation_weights
    }
}

/// High-precision `#![no_std]` $\sin(\phi)$ on $\phi \in [0, \pi/2]$ via 11th-order minimax/Taylor polynomial.
#[inline]
fn cold_sin_half_pi_f64(phi: f64) -> f64 {
    let x = phi.clamp(0.0, core::f64::consts::FRAC_PI_2);
    let x2 = x * x;
    x * (1.0
        - x2 * (1.0 / 6.0
            - x2 * (1.0 / 120.0
                - x2 * (1.0 / 5_040.0 - x2 * (1.0 / 362_880.0 - x2 / 39_916_800.0)))))
}

/// High-precision `#![no_std]` $\arcsin(x)$ for small cutoff $x = \sqrt{\tau} \in (0, 0.25]$.
#[inline]
fn cold_asin_small_f64(x: f64) -> f64 {
    let x2 = x * x;
    x * (1.0 + x2 * (1.0 / 6.0 + x2 * (3.0 / 40.0 + x2 * (5.0 / 112.0))))
}

#[inline]
fn draw_uniform_01_from_chacha_stream(
    seed: &[u8; 32],
    stream_buf: &mut [u8; 64],
    word_cursor: &mut usize,
    blk_counter: &mut u32,
) -> f64 {
    if *word_cursor >= 16 {
        *stream_buf = chacha20_block_64b(seed, &[0u8; 12], *blk_counter);
        *blk_counter = blk_counter.wrapping_add(1);
        *word_cursor = 0;
    }
    let off = *word_cursor * 4;
    *word_cursor += 1;
    let u = u32::from_le_bytes([
        stream_buf[off],
        stream_buf[off + 1],
        stream_buf[off + 2],
        stream_buf[off + 3],
    ]);
    ((u as f64) + 0.5) * (1.0 / 4_294_967_296.0)
}

/// Generates a Two-Seed Arcsine Tardos Codebook (`TardosCodebook7680`) across `Arena 0..3`
/// using per-tier `(shared_seed[t], buyer_seed[t])` (`owning_tier = t as u8`) derived via
/// [`derive_watermark_seed_hmac_sha256`] (`ref[sec.watermark.seed-derivation]`, `ref[sec.watermark.tardos-code]`):
/// - Sets `chips[i] = 0.0` and `accusation_weights[i] = 0.0` on all `64` Role A coordinates (`word_in_tier < 16`).
/// - Draws Arcsine bias $p_i = \sin^2(\tau' + (\frac{\pi}{2} - 2\tau')u_i) \in [\tau, 1-\tau]$ with
///   $\tau = \frac{1}{300 c_{\max}^2}$ from `shared_seed[t]`, and draws $v_{j,i} \in (0, 1)$ from `buyer_seed[t]`
///   to populate binary $X_{j,i} = \mathbf{1}[v_{j,i} < p_i]$, `chips[i] = 2 X_{j,i} - 1`, and symmetric
///   Škorić–Tardos accusation weight $U_{j,i} = X_{j,i}\sqrt{(1-p_i)/p_i} - (1-X_{j,i})\sqrt{p_i/(1-p_i)}$
///   across all `7,616` Role B coordinates.
pub fn generate_tardos_codebook_7680(
    wm_key: &[u8; 32],
    scope_base: &WatermarkSeedScope,
    license_id: &[u8; 16],
    buyer_id: &[u8; 16],
    c_max: usize,
) -> Result<TardosCodebook7680, DnamError> {
    let c_eff = c_max.max(1) as f64;
    let tau = (1.0 / (300.0 * c_eff * c_eff)).clamp(1e-6, 0.25);
    let tau_prime = cold_asin_small_f64(cold_sqrt_f64(tau));
    let span = core::f64::consts::FRAC_PI_2 - 2.0 * tau_prime;

    let mut chips = [0.0_f32; TARDOS_CASCADE_WEIGHTS];
    let mut accusation_weights = [0.0_f32; TARDOS_CASCADE_WEIGHTS];

    for (tier, &(start, end, _)) in TARDOS_CASCADE_RANGES.iter().enumerate() {
        let tier_scope = WatermarkSeedScope {
            owning_tier: tier as u8,
            ..*scope_base
        };
        let shared_seed = derive_watermark_seed_hmac_sha256(wm_key, &tier_scope, None)?;
        let buyer_seed =
            derive_watermark_seed_hmac_sha256(wm_key, &tier_scope, Some((license_id, buyer_id)))?;

        let mut shared_buf = [0u8; 64];
        let mut buyer_buf = [0u8; 64];
        let mut shared_cursor = 16_usize;
        let mut buyer_cursor = 16_usize;
        let mut shared_blk = 0_u32;
        let mut buyer_blk = 0_u32;

        for idx in start..end {
            if idx - start < 16 {
                chips[idx] = 0.0;
                accusation_weights[idx] = 0.0;
                continue;
            }
            let u_i = draw_uniform_01_from_chacha_stream(
                &shared_seed,
                &mut shared_buf,
                &mut shared_cursor,
                &mut shared_blk,
            );
            let v_ji = draw_uniform_01_from_chacha_stream(
                &buyer_seed,
                &mut buyer_buf,
                &mut buyer_cursor,
                &mut buyer_blk,
            );
            let phi = tau_prime + span * u_i;
            let s = cold_sin_half_pi_f64(phi);
            let p_i = (s * s).clamp(tau, 1.0 - tau);

            if v_ji < p_i {
                chips[idx] = 1.0;
                accusation_weights[idx] = cold_sqrt_f64((1.0 - p_i) / p_i) as f32;
            } else {
                chips[idx] = -1.0;
                accusation_weights[idx] = -(cold_sqrt_f64(p_i / (1.0 - p_i)) as f32);
            }
        }
    }

    Ok(TardosCodebook7680 {
        chips,
        accusation_weights,
    })
}

/// Deterministically generates a 7,680-coordinate Two-Seed Arcsine Tardos codebook
/// ([`TardosCodebook7680`], dereferencing to `[f32; 7680]`) from `(marketplace_secret, model_id, license_id)`
/// with default collusion resistance $c_{\max} = 4$ ($\tau = 1/4800$).
pub fn generate_tardos_code_7680(
    marketplace_secret: &[u8; 32],
    model_id: &str,
    license_id: &str,
) -> TardosCodebook7680 {
    let model_digest = sha256_digest(model_id.as_bytes());
    let lic_digest = sha256_digest(license_id.as_bytes());
    let mut model_id_16 = [0u8; 16];
    let mut lic_id_16 = [0u8; 16];
    let mut buyer_id_16 = [0u8; 16];
    model_id_16.copy_from_slice(&model_digest[0..16]);
    lic_id_16.copy_from_slice(&lic_digest[0..16]);
    buyer_id_16.copy_from_slice(&lic_digest[16..32]);
    // Ensure non-all-zero 16B identifiers so `derive_watermark_seed_hmac_sha256` never rejects arbitrary string IDs
    model_id_16[0] |= 1;
    lic_id_16[0] |= 1;
    buyer_id_16[0] |= 1;

    let mut manifest_sha = model_digest;
    manifest_sha[0] |= 1;
    let mut mkt_pk = *marketplace_secret;
    if mkt_pk.iter().all(|&b| b == 0) {
        mkt_pk[0] = 1;
    }

    let scope_base = WatermarkSeedScope {
        marketplace_pubkey: mkt_pk,
        watermark_key_id: model_id_16,
        creator_signed_manifest_sha256: manifest_sha,
        model_id: model_id_16,
        watermark_profile_sha256: [0x55; 32],
        owning_tier: 0,
    };

    generate_tardos_codebook_7680(marketplace_secret, &scope_base, &lic_id_16, &buyer_id_16, 4)
        .expect("non-zero scope IDs are always valid")
}

/// Computes the strictly positive coordinate-space `EMR`-masked perturbation envelope
/// $\epsilon_i \in [5 \times 10^{-5},\, 2 \times 10^{-4}]$ across all `7,616` Role B coordinates
/// (and $\epsilon_i = 0.0$ on the `64` Role A coordinates):
/// $$\epsilon_i = \operatorname{clamp}\!\left(\min_{c \in \{0,1\}} \frac{\epsilon_{\text{mask},c}}{\sqrt{\max([\widehat{\mathbf{G}}_{\text{EMR}}]_{s(i),s(i)},\, 10^{-12})}},\, 5\times 10^{-5},\, 2\times 10^{-4}\right)$$
pub fn compute_emr_masked_tardos_envelopes(
    gram_diag_256: &[f64; 256],
    epsilon_mask_lr: [f64; 2],
) -> [f32; TARDOS_CASCADE_WEIGHTS] {
    let mut envelopes = [0.0_f32; TARDOS_CASCADE_WEIGHTS];
    let min_mask = epsilon_mask_lr[0].min(epsilon_mask_lr[1]).max(0.0);
    for idx in 0..TARDOS_CASCADE_WEIGHTS {
        if let Some((tier, slot_32, lane)) = role_b_head_slot_for_index(idx) {
            let group_16 = slot_32 % 16;
            let feat_idx = tier * 64 + group_16 * 4 + lane;
            let g_diag = gram_diag_256[feat_idx].max(1e-12);
            let raw_eps = min_mask / cold_sqrt_f64(g_diag);
            envelopes[idx] = raw_eps.clamp(5e-5, 2e-4) as f32;
        }
    }
    envelopes
}

/// Embeds **Layer 3 (Prefix-Composable Spread-Spectrum Tardos Watermark)** into the `7,616` Role B
/// coordinates of `Arena 0..3` (`64` Role A coordinates at `word_in_tier < 16` are strictly untouched):
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
        2 => 2_304,
        3 => 4_608,
        _ => TARDOS_CASCADE_WEIGHTS,
    };
    if code.len() < needed_chips {
        return Err(DnamError::BufferTooSmall {
            expected: needed_chips,
            actual: code.len(),
        });
    }
    for_each_conv_weight_offset_up_to(arena.len(), |idx, byte_off| {
        if is_role_a_cascade_index(idx) {
            return;
        }
        let chip = if code[idx] > 0.0 {
            1.0_f32
        } else if code[idx] < 0.0 {
            -1.0_f32
        } else {
            0.0_f32
        };
        let w = read_f32_le(arena, byte_off);
        write_f32_le(arena, byte_off, w + epsilon * chip);
    });
    Ok(())
}

/// Embeds **Layer 3 (`EMR`-Masked Spread-Spectrum Tardos Watermark)** using per-coordinate
/// positive envelopes $\epsilon_i \in [5 \times 10^{-5},\, 2 \times 10^{-4}]$ on Role B (`0.0` on Role A).
pub fn apply_tardos_watermark_7680_envelopes(
    arena: &mut [u8],
    code: &[f32],
    envelopes: &[f32; TARDOS_CASCADE_WEIGHTS],
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
        2 => 2_304,
        3 => 4_608,
        _ => TARDOS_CASCADE_WEIGHTS,
    };
    if code.len() < needed_chips {
        return Err(DnamError::BufferTooSmall {
            expected: needed_chips,
            actual: code.len(),
        });
    }
    for_each_conv_weight_offset_up_to(arena.len(), |idx, byte_off| {
        if is_role_a_cascade_index(idx) {
            return;
        }
        let chip = if code[idx] > 0.0 {
            1.0_f32
        } else if code[idx] < 0.0 {
            -1.0_f32
        } else {
            0.0_f32
        };
        let eps = envelopes[idx].max(0.0);
        let w = read_f32_le(arena, byte_off);
        write_f32_le(arena, byte_off, w + eps * chip);
    });
    Ok(())
}

/// Computes the symmetric Škorić–Tardos Z-score ($\sigma$) of a suspect `Arena 0..3` (including
/// tier-stripped containers down to `10,752 B` `Lite`) against `master_arena` and a candidate buyer's `code`:
/// $$Z_j = \frac{1}{\sqrt{\max\!\big(\sum_{i=0}^{N_{\text{prefix}}-1} y_i^2,\, 1\big)}} \sum_{i=0}^{N_{\text{prefix}}-1} U_{j,i} \, y_i, \qquad y_i = \operatorname{sign}_{\text{deadzone}}\!\big(W_i^{(\text{suspect})} - W_i^{(\text{master})},\, 10^{-7}\big)$$
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
        2 => 2_304,
        3 => 4_608,
        _ => TARDOS_CASCADE_WEIGHTS,
    };
    if code.len() < needed_chips {
        return Err(DnamError::BufferTooSmall {
            expected: needed_chips,
            actual: code.len(),
        });
    }
    let mut corr = 0.0_f64;
    let mut active_y2 = 0.0_f64;
    for_each_conv_weight_offset_up_to(shared_len, |idx, byte_off| {
        if is_role_a_cascade_index(idx) {
            return;
        }
        let w_m = read_f32_le(master_arena, byte_off);
        let w_s = read_f32_le(suspect_arena, byte_off);
        let diff = (w_s - w_m) as f64;
        let y_i = if diff > 1e-7 {
            1.0_f64
        } else if diff < -1e-7 {
            -1.0_f64
        } else {
            0.0_f64
        };
        corr += (code[idx] as f64) * y_i;
        active_y2 += y_i * y_i;
    });
    Ok(corr / cold_sqrt_f64(active_y2.max(1.0)))
}

/// Standalone Role B $\leftrightarrow$ `W_head` reciprocal bilinear gauge helper (`Oracle-01`):
/// Adds `role_b_delta_32` to the first 32 Role B `Vec4` slots (`b = 1..9` at `CASCADE_W_BASES[tier] + 64..576`,
/// updating Tier `tier`'s runtime multiplier $\mathbf{m}_{t,c,g} \gets \mathbf{m}_{t,c,g} + \boldsymbol{\delta}_{c,g}$)
/// and reciprocally scales Tier `tier`'s `W_head` slices across all modes (`0..5` in `HEAD_BASES[tier]`,
/// plus `Ultra` modes `5..8` at `49,408 + tier * 2,048` when `arena.len() >= 64,000`) so that
/// $\mathbf{m}_{t,c,g} \odot \widetilde{\mathbf{w}}_{t,m,c,g}$ is invariant (`0.000000 ULP` for dyadic
/// $1 + \delta \in \{0.5, 1.0, 2.0\}$, and `max_audio_diff < 1e-5` for continuous $1 + \delta \in [0.985, 1.015]$).
pub fn apply_cascade_head_gauge(
    arena: &mut [u8],
    tier: usize,
    role_b_delta_32: &[[f32; 4]; 32],
) -> Result<(), DnamError> {
    let max_tiers = max_tiers_for_arena_len(arena.len());
    if max_tiers == 0 || tier >= max_tiers {
        return Err(DnamError::BufferTooSmall {
            expected: 10_752,
            actual: arena.len(),
        });
    }
    let cascade_base = TARDOS_CASCADE_RANGES[tier].2;
    let cascade_words = TARDOS_CASCADE_RANGES[tier].1 - TARDOS_CASCADE_RANGES[tier].0;
    let role_b_vec4_count = (cascade_words - 16) / 4;
    let head_arena_offsets = [4_928_usize, 18_048_usize, 31_936_usize, 57_600_usize];
    let head_base = head_arena_offsets[tier];

    for (s, delta_vec) in role_b_delta_32.iter().enumerate() {
        for lane in 0..4 {
            // Compute existing Role B sum for (s, lane) before perturbation:
            let mut sum_before = 1.0_f32;
            let mut v_idx = s;
            while v_idx < role_b_vec4_count {
                let off = cascade_base + 64 + (v_idx * 4 + lane) * 4;
                sum_before += read_f32_le(arena, off);
                v_idx += 32;
            }
            let delta = delta_vec[lane];
            let sum_after = sum_before + delta;
            if sum_after.abs() < 1e-6 {
                return Err(DnamError::SchemaValidationError(
                    "Role B gauge multiplier sum_after must be non-zero".into(),
                ));
            }
            // Write delta into the first Role B block for slot `s`:
            let first_off = cascade_base + 64 + (s * 4 + lane) * 4;
            let cur_first = read_f32_le(arena, first_off);
            write_f32_le(arena, first_off, cur_first + delta);

            let scale_ratio = sum_before / sum_after;
            for mode in 0..5 {
                let w_off = head_base + mode * 512 + (s * 4 + lane) * 4;
                let w_val = read_f32_le(arena, w_off);
                write_f32_le(arena, w_off, w_val * scale_ratio);
            }
            if max_tiers >= 4 {
                let ultra_base = 49_408 + tier * 2_048;
                for m_u in 0..4 {
                    let w_off = ultra_base + m_u * 512 + (s * 4 + lane) * 4;
                    let w_val = read_f32_le(arena, w_off);
                    write_f32_le(arena, w_off, w_val * scale_ratio);
                }
            }
        }
    }
    Ok(())
}
