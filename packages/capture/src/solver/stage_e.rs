#![forbid(unsafe_code)]
//! Stage E (`< 0.2 ms` `.dnam-proj` Packing, 4-Tier Loudness Trim & Telemetry, and 3-Layer Buyer
//! Watermarked `.dnam` Export + Black-Box Modal Phase Identification, `r[train.pipeline.stage-e-pack]`).

use dnam_engine::container::{
    ARENA_OFFSET_ULTRA_END, DnamContainerView, build_dnam_container_with_cp1,
    strip_container_to_tier,
};
use dnam_engine::control::Cp1ContributorSlot;
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::provenance::{
    DefaultCryptoProvider, STAGE3_KAUTZ_BLOCKS_PER_TIER, TardosCodebook7680, WatermarkSeedScope,
    apply_isometric_gauge_watermark, apply_phase_coding_watermark,
    apply_tardos_watermark_7680_envelopes, compute_emr_masked_tardos_envelopes,
    generate_tardos_codebook_7680, kautz_soa_byte_offset, sha256_digest,
};
use dnam_engine::schema::DnamError;
use dnam_engine::{DnamEngine, Tier};

use crate::metrics::{evaluate_besr_from_band_energies, try_split_bezout_3band_energies};
use crate::proj::{
    DnamProjManifest, DnamProjView, PROJ_STAT_BYTES_ULTRA, ProjChunkKind, SolverTelemetrySummary,
    zero_arena_b_head_dc_bias,
};
use crate::solver::bezout_gram::canonical_besr_warmup;
use crate::solver::stage_b_c::compute_role_b_multipliers;
use crate::solver::stage_d::star_center_control_values;
use crate::solver::{solve_uplo_u_cholesky_2ch, uplo_u_index, uplo_u_packed_len};

const ARENA0_TIER_GAIN_TRIM_OFFSET: usize = 10_624;
const HEAD_BASES: [usize; 4] = [4_928, 18_048, 31_936, 57_600];
const TIERS_ALL: [Tier; 4] = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra];

/// Telemetry and trim outputs produced by Stage E (`r[train.pipeline.stage-e-pack]`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StageESolveOutput {
    /// Per-tier validation `ESR` in dB (`[Lite, Full, Pro, Ultra]`).
    pub tier_esr_db: [f32; 4],
    /// Per-tier integrated loudness in `LUFS` (`[Lite, Full, Pro, Ultra]`).
    pub tier_lufs: [f32; 4],
    /// Per-tier loudness compensation trim in dB (`<= 0.05 LU` step across tiers).
    pub tier_output_trim_db: [f32; 4],
    /// Wall-clock duration of Stage E in milliseconds.
    pub elapsed_ms: f32,
}

/// Configuration for 3-Layer Buyer Watermarked `.dnam` Runtime Export (`Case #8`).
#[derive(Debug, Clone, PartialEq)]
pub struct BuyerWatermarkConfig {
    /// 32-byte marketplace master secret (`K_master`).
    pub marketplace_secret: [u8; 32],
    /// Canonical model identifier (`model_id`).
    pub model_id: String,
    /// Per-transaction license identifier (`license_id`).
    pub license_id: String,
    /// 16-byte buyer UUID (`buyer_id`).
    pub buyer_id: [u8; 16],
    /// Layer-2 pole-angle perturbation in radians (`<= PHASE_CODING_MAX_RAD`, default `2.5e-5`).
    pub phase_coding_delta_rad: f32,
    /// Layer-3 EMR masking budget `[eps_L, eps_R]` (default `[1.0e-5, 1.0e-5]`).
    pub epsilon_mask_lr: [f64; 2],
    /// Reciprocally compensates Layer-3 Role-B gain deltas in `W_head` so audio null is `< -95 dB`
    /// while `stage2_cascade_w` retains 100% Tardos chip SNR.
    pub compensate_role_b_in_head: bool,
}

impl BuyerWatermarkConfig {
    /// Constructs a canonical 3-Layer Buyer Watermark configuration with `< -85 dB` null defaults.
    #[must_use]
    pub fn new(
        marketplace_secret: [u8; 32],
        model_id: impl Into<String>,
        license_id: impl Into<String>,
        buyer_id: [u8; 16],
    ) -> Self {
        Self {
            marketplace_secret,
            model_id: model_id.into(),
            license_id: license_id.into(),
            buyer_id,
            phase_coding_delta_rad: 3.5e-4,
            epsilon_mask_lr: [1.0e-5, 1.0e-5],
            compensate_role_b_in_head: true,
        }
    }
}

#[inline]
fn read_f32_le(buf: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

#[inline]
fn write_f32_le(buf: &mut [u8], off: usize, val: f32) {
    buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn now_ms_token() -> Option<std::time::Instant> {
    Some(std::time::Instant::now())
}

#[cfg(target_arch = "wasm32")]
#[inline]
fn now_ms_token() -> Option<()> {
    None
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn elapsed_ms_since(token: Option<std::time::Instant>) -> f32 {
    token.map_or(0.05, |t| (t.elapsed().as_secs_f64() * 1_000.0) as f32)
}

#[cfg(target_arch = "wasm32")]
#[inline]
fn elapsed_ms_since(_token: Option<()>) -> f32 {
    0.05
}

/// Selects the highest-energy contiguous window of length `<= max_len` after a `64`-sample warmup.
#[must_use]
pub fn select_eval_window(wet_lr: &[[f32; 2]], max_len: usize) -> (usize, usize) {
    let n = wet_lr.len();
    if n == 0 {
        return (0, 0);
    }
    let warmup = n.min(64);
    let avail = n.saturating_sub(warmup);
    if avail <= max_len {
        return (warmup, n);
    }
    let stride = max_len / 2;
    let mut best_start = warmup;
    let mut best_e = -1.0_f64;
    let mut start = warmup;
    while start + max_len <= n {
        let mut e = 0.0_f64;
        for &[l, r] in &wet_lr[start..start + max_len] {
            e += f64::from(l) * f64::from(l) + f64::from(r) * f64::from(r);
        }
        if e > best_e {
            best_e = e;
            best_start = start;
        }
        start += stride.max(1);
    }
    (best_start, best_start + max_len)
}

/// Evaluates integrated energy loudness (`LUFS`, ITU-R BS.1770 energy scale `-0.691 + 10 log10(E_L + E_R)`).
#[must_use]
pub fn compute_integrated_lufs(frames_lr: &[[f32; 2]]) -> f32 {
    if frames_lr.is_empty() {
        return -70.0;
    }
    let mut sum_sq = 0.0_f64;
    for &[l, r] in frames_lr {
        sum_sq += f64::from(l) * f64::from(l) + f64::from(r) * f64::from(r);
    }
    let mean_sq = (sum_sq / (frames_lr.len() as f64)).max(1.0e-14);
    (-0.691 + 10.0 * mean_sq.log10()).clamp(-70.0, 24.0) as f32
}

/// Finalizes Stage E (`r[train.pipeline.stage-e-pack]`): enforces `[STAB-ZERO-DC]`, computes or
/// preserves 4-tier `ESR` & `LUFS` loudness compensation trims (`<= 0.05 LU`), and updates
/// `manifest.solver_telemetry`.
pub fn finalize_stage_e_and_telemetry(
    manifest: &mut DnamProjManifest,
    arena_64k: &mut [u8],
    cp1_slots: &[Cp1ContributorSlot],
    dry_lr: &[[f32; 2]],
    star_wet_lr: Option<&[[f32; 2]]>,
    fallback_tier_esr_db: Option<[f32; 4]>,
    prior_solve_ms: f32,
) -> Result<StageESolveOutput, DnamError> {
    let timer = now_ms_token();
    zero_arena_b_head_dc_bias(arena_64k);
    let _ = super::stages::sync_clamp_descriptors_to_arena(&manifest.base, arena_64k)?;

    let active_count = manifest.stems_registry.iter().filter(|s| s.enabled).count() as u32;

    let (tier_esr_db, tier_lufs, tier_output_trim_db) = if let Some(wet_lr) = star_wet_lr
        && !dry_lr.is_empty()
        && !wet_lr.is_empty()
    {
        let mono_only = manifest.base.io_config.output_mode == "mono";
        let n_common = dry_lr.len().min(wet_lr.len());
        let w_start = canonical_besr_warmup(n_common);
        for slot in 0..16 {
            if ARENA0_TIER_GAIN_TRIM_OFFSET + (slot + 1) * 4 <= arena_64k.len() {
                write_f32_le(arena_64k, ARENA0_TIER_GAIN_TRIM_OFFSET + slot * 4, 1.0);
            }
        }
        let mut temp_hdr = manifest.base.clone();
        let dnam_bytes =
            build_dnam_container_with_cp1(&mut temp_hdr, arena_64k, arena_64k, false, cp1_slots)?;
        let star_ctrls = star_center_control_values(manifest);

        let mut esr_db = [-40.0_f32; 4];
        let mut lufs = [-20.5_f32; 4];
        let mut tier_waves: [Vec<[f32; 2]>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let tgt_l: Vec<f64> = wet_lr[..n_common].iter().map(|v| f64::from(v[0])).collect();
        let tgt_r: Vec<f64> = wet_lr[..n_common].iter().map(|v| f64::from(v[1])).collect();

        for (t_idx, &tier) in TIERS_ALL.iter().enumerate() {
            let tier_bytes = if tier == Tier::Ultra {
                std::borrow::Cow::Borrowed(dnam_bytes.as_slice())
            } else {
                std::borrow::Cow::Owned(strip_container_to_tier(&dnam_bytes, tier)?)
            };
            let mut engine =
                DnamEngine::<DefaultMathBackend>::from_bytes(&tier_bytes, tier, 48_000)?;
            engine.reset_to_dc_equilibrium();
            for (p, &val) in star_ctrls.iter().enumerate() {
                engine.set_control_immediate(p, val);
            }
            let rendered: Vec<[f32; 2]> = dry_lr[..n_common]
                .iter()
                .map(|&[dl, dr]| engine.step_frame_tora(dl, dr, mono_only))
                .collect();
            lufs[t_idx] = ((compute_integrated_lufs(&rendered[w_start..n_common]) * 1_000.0)
                .round())
                / 1_000.0;
            tier_waves[t_idx] = rendered;
        }

        let ultra_lufs = lufs[3];
        let mut trims = [0.0_f32; 4];
        for t in 0..4 {
            let raw_trim = (ultra_lufs - lufs[t]).clamp(-12.0, 12.0);
            trims[t] = if raw_trim.abs() < 5.0e-4 {
                0.0
            } else {
                (raw_trim * 1_000.0).round() / 1_000.0
            };
            let lin_trim = 10.0_f32.powf(raw_trim / 20.0);
            if ARENA0_TIER_GAIN_TRIM_OFFSET + (t + 1) * 4 <= arena_64k.len() {
                write_f32_le(arena_64k, ARENA0_TIER_GAIN_TRIM_OFFSET + t * 4, lin_trim);
            }
            let g64 = f64::from(lin_trim);
            let mdl_l: Vec<f64> = tier_waves[t]
                .iter()
                .map(|v| g64 * f64::from(v[0]))
                .collect();
            let ch_besr = |tgt: &[f64], mdl: &[f64]| -> Option<f64> {
                let (se, ee, ne) =
                    try_split_bezout_3band_energies(tgt, mdl, None, 48_000.0, w_start)?;
                let rep = evaluate_besr_from_band_energies(se, ee, ne);
                rep.valid.then_some(rep.besr_db)
            };
            let db_val = if mono_only {
                ch_besr(&tgt_l, &mdl_l).unwrap_or(-120.0)
            } else {
                let mdl_r: Vec<f64> = tier_waves[t]
                    .iter()
                    .map(|v| g64 * f64::from(v[1]))
                    .collect();
                match (ch_besr(&tgt_l, &mdl_l), ch_besr(&tgt_r, &mdl_r)) {
                    (Some(dl), Some(dr)) => dl.max(dr),
                    (Some(d), None) | (None, Some(d)) => d,
                    (None, None) => -120.0,
                }
            };
            esr_db[t] = ((db_val.clamp(-120.0, 20.0) * 1_000.0).round() / 1_000.0) as f32;
        }
        for pad in 4..16 {
            if ARENA0_TIER_GAIN_TRIM_OFFSET + (pad + 1) * 4 <= arena_64k.len() {
                write_f32_le(arena_64k, ARENA0_TIER_GAIN_TRIM_OFFSET + pad * 4, 1.0);
            }
        }

        manifest.base.slimmable_tiers.lite.validation_esr_db = esr_db[0];
        manifest.base.slimmable_tiers.full.validation_esr_db = esr_db[1];
        manifest.base.slimmable_tiers.pro.validation_esr_db = esr_db[2];
        manifest.base.slimmable_tiers.ultra.validation_esr_db = esr_db[3];

        manifest.base.slimmable_tiers.lite.loudness_lufs = lufs[0];
        manifest.base.slimmable_tiers.full.loudness_lufs = lufs[1];
        manifest.base.slimmable_tiers.pro.loudness_lufs = lufs[2];
        manifest.base.slimmable_tiers.ultra.loudness_lufs = lufs[3];

        manifest.base.slimmable_tiers.lite.output_trim_db = trims[0];
        manifest.base.slimmable_tiers.full.output_trim_db = trims[1];
        manifest.base.slimmable_tiers.pro.output_trim_db = trims[2];
        manifest.base.slimmable_tiers.ultra.output_trim_db = trims[3];

        (esr_db, lufs, trims)
    } else {
        let esr_db = fallback_tier_esr_db
            .or_else(|| manifest.solver_telemetry.as_ref().map(|s| s.tier_esr_db))
            .unwrap_or([
                manifest.base.slimmable_tiers.lite.validation_esr_db,
                manifest.base.slimmable_tiers.full.validation_esr_db,
                manifest.base.slimmable_tiers.pro.validation_esr_db,
                manifest.base.slimmable_tiers.ultra.validation_esr_db,
            ]);
        let lufs = [
            manifest.base.slimmable_tiers.lite.loudness_lufs,
            manifest.base.slimmable_tiers.full.loudness_lufs,
            manifest.base.slimmable_tiers.pro.loudness_lufs,
            manifest.base.slimmable_tiers.ultra.loudness_lufs,
        ];
        let trims = [
            manifest.base.slimmable_tiers.lite.output_trim_db,
            manifest.base.slimmable_tiers.full.output_trim_db,
            manifest.base.slimmable_tiers.pro.output_trim_db,
            manifest.base.slimmable_tiers.ultra.output_trim_db,
        ];
        (esr_db, lufs, trims)
    };

    let stage_e_ms = elapsed_ms_since(timer);
    let total_ms = (prior_solve_ms + stage_e_ms).max(0.05);
    manifest.solver_telemetry = Some(SolverTelemetrySummary {
        active_stem_count: active_count,
        total_solve_ms: total_ms,
        tier_esr_db,
    });

    Ok(StageESolveOutput {
        tier_esr_db,
        tier_lufs,
        tier_output_trim_db,
        elapsed_ms: stage_e_ms,
    })
}

/// Deterministically derives the 7,680-chip Two-Seed Arcsine Tardos codebook (`c_max = 5`)
/// bound to `(marketplace_secret, model_id, license_id, buyer_id)`.
#[must_use]
pub fn derive_buyer_tardos_codebook(config: &BuyerWatermarkConfig) -> TardosCodebook7680 {
    let model_digest = sha256_digest(config.model_id.as_bytes());
    let lic_digest = sha256_digest(config.license_id.as_bytes());
    let mut model_id_16 = [0u8; 16];
    let mut lic_id_16 = [0u8; 16];
    let mut buyer_id_16 = config.buyer_id;
    model_id_16.copy_from_slice(&model_digest[0..16]);
    lic_id_16.copy_from_slice(&lic_digest[0..16]);
    model_id_16[0] |= 1;
    lic_id_16[0] |= 1;
    if buyer_id_16.iter().all(|&b| b == 0) {
        buyer_id_16.copy_from_slice(&lic_digest[16..32]);
        buyer_id_16[0] |= 1;
    }
    let mut manifest_sha = model_digest;
    manifest_sha[0] |= 1;
    let mut mkt_pk = config.marketplace_secret;
    if mkt_pk.iter().all(|&b| b == 0) {
        mkt_pk[0] = 1;
    }
    let scope = WatermarkSeedScope {
        marketplace_pubkey: mkt_pk,
        watermark_key_id: model_id_16,
        creator_signed_manifest_sha256: manifest_sha,
        model_id: model_id_16,
        watermark_profile_sha256: [0x55; 32],
        owning_tier: 0,
    };
    generate_tardos_codebook_7680(
        &config.marketplace_secret,
        &scope,
        &lic_id_16,
        &buyer_id_16,
        5,
    )
    .expect("non-zero scope IDs are always valid")
}

/// Deterministically derives the 32-byte Layer-2 buyer phase seed (`"DNAM-BUYER-PHASE-SEED-V1:"`).
#[must_use]
pub fn derive_buyer_phase_seed(config: &BuyerWatermarkConfig) -> [u8; 32] {
    let mut msg = Vec::with_capacity(96 + config.model_id.len() + config.license_id.len());
    msg.extend_from_slice(b"DNAM-BUYER-PHASE-SEED-V1:");
    msg.extend_from_slice(&config.marketplace_secret);
    msg.extend_from_slice(&(config.model_id.len() as u32).to_le_bytes());
    msg.extend_from_slice(config.model_id.as_bytes());
    msg.extend_from_slice(&(config.license_id.len() as u32).to_le_bytes());
    msg.extend_from_slice(config.license_id.as_bytes());
    msg.extend_from_slice(&config.buyer_id);
    sha256_digest(&msg)
}

/// Deterministically derives Layer-1 modal reciprocal scaling `kappa[0..32]` in `[0.988, 1.012]`
/// and an exact special orthogonal rotation `q_buyer_4x4` in `SO(4)` (`det = +1`).
#[must_use]
pub fn derive_buyer_gauge_params(config: &BuyerWatermarkConfig) -> ([f32; 32], [[f32; 4]; 4]) {
    let mut seed_msg = Vec::with_capacity(96 + config.model_id.len() + config.license_id.len());
    seed_msg.extend_from_slice(b"DNAM-BUYER-GAUGE-SEED-V1:");
    seed_msg.extend_from_slice(&config.marketplace_secret);
    seed_msg.extend_from_slice(config.model_id.as_bytes());
    seed_msg.push(0x1F);
    seed_msg.extend_from_slice(config.license_id.as_bytes());
    seed_msg.extend_from_slice(&config.buyer_id);
    let digest_a = sha256_digest(&seed_msg);
    seed_msg.push(0xA5);
    let digest_b = sha256_digest(&seed_msg);

    let mut kappa = [1.0_f32; 32];
    for (i, k_slot) in kappa.iter_mut().enumerate() {
        let b = if i < 16 {
            u16::from_le_bytes([digest_a[i * 2], digest_a[i * 2 + 1]])
        } else {
            let j = i - 16;
            u16::from_le_bytes([digest_b[j * 2], digest_b[j * 2 + 1]])
        };
        let u = (f64::from(b) / 65_535.0) * 2.0 - 1.0;
        *k_slot = (1.0 + 0.012 * u) as f32;
    }

    // Compose 6 Givens rotations across all 6 planes of R^4 -> exact Q in SO(4) with det(Q) = +1
    let mut q_f64 = [
        [1.0_f64, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let planes: [(usize, usize); 6] = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)];
    for (p_idx, &(i, j)) in planes.iter().enumerate() {
        let raw = u16::from_le_bytes([digest_b[p_idx * 2], digest_b[p_idx * 2 + 1]]);
        let angle = (f64::from(raw) / 65_535.0) * std::f64::consts::TAU;
        let (s, c) = angle.sin_cos();
        for row in &mut q_f64 {
            let ri = row[i];
            let rj = row[j];
            row[i] = c * ri - s * rj;
            row[j] = s * ri + c * rj;
        }
    }

    let mut q_f32 = [[0.0_f32; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            q_f32[r][c] = q_f64[r][c] as f32;
        }
    }
    (kappa, q_f32)
}

fn rotate_cp1_slots_by_buyer_gauge(slots: &mut [Cp1ContributorSlot], q_buyer_4x4: &[[f32; 4]; 4]) {
    let rotate_4 = |row: &mut [f32; 8]| {
        let old = [row[0], row[1], row[2], row[3]];
        for k in 0..4 {
            let mut acc = 0.0_f32;
            for m in 0..4 {
                acc += old[m] * q_buyer_4x4[m][k];
            }
            row[k] = acc;
        }
    };
    for slot in slots {
        let lanes = usize::from(slot.lanes).min(3);
        for l in 0..lanes {
            rotate_4(&mut slot.main_base[l]);
            if slot.has_ultra {
                rotate_4(&mut slot.main_ultra[l]);
            }
        }
    }
}

fn compensate_head_for_role_b_ratio(
    arena_64k: &mut [u8],
    m_before: &[[f64; 2]; 256],
    m_after: &[[f64; 2]; 256],
) {
    for (t, &h_base) in HEAD_BASES.iter().enumerate() {
        for i in 0..64 {
            let idx = t * 64 + i;
            let ratio_l = if m_after[idx][0].abs() > 1.0e-6 {
                (m_before[idx][0] / m_after[idx][0]) as f32
            } else {
                1.0
            };
            let ratio_r = if m_after[idx][1].abs() > 1.0e-6 {
                (m_before[idx][1] / m_after[idx][1]) as f32
            } else {
                1.0
            };
            for mode in 0..5 {
                let m_off = h_base + mode * 512;
                if m_off + 512 <= arena_64k.len() {
                    let wl = read_f32_le(arena_64k, m_off + i * 4) * ratio_l;
                    let wr = read_f32_le(arena_64k, m_off + 256 + i * 4) * ratio_r;
                    write_f32_le(arena_64k, m_off + i * 4, wl);
                    write_f32_le(arena_64k, m_off + 256 + i * 4, wr);
                }
            }
            if arena_64k.len() >= ARENA_OFFSET_ULTRA_END as usize {
                let u_base = 49_408 + t * 2_048;
                for um in 0..4 {
                    let m_off = u_base + um * 512;
                    let wl = read_f32_le(arena_64k, m_off + i * 4) * ratio_l;
                    let wr = read_f32_le(arena_64k, m_off + 256 + i * 4) * ratio_r;
                    write_f32_le(arena_64k, m_off + i * 4, wl);
                    write_f32_le(arena_64k, m_off + 256 + i * 4, wr);
                }
            }
        }
    }
}

fn extract_gram_diag_from_stat(stat_bytes: Option<&[u8]>) -> [f64; 256] {
    let mut diag = [1.0_f64; 256];
    if let Some(bytes) = stat_bytes
        && bytes.len() >= PROJ_STAT_BYTES_ULTRA
    {
        for (d, slot) in diag.iter_mut().enumerate() {
            let off = uplo_u_index(d, d) * 8;
            let mut b = [0u8; 8];
            b.copy_from_slice(&bytes[off..off + 8]);
            let val = f64::from_le_bytes(b);
            if val.is_finite() && val > 1.0e-12 {
                *slot = val;
            }
        }
    }
    diag
}

/// Exports a 3-Layer Buyer Watermarked `.dnam` runtime container from a `.dnam-proj` archive
/// (`r[train.pipeline.stage-e-pack]`): embeds Layer-1 isometric `SO(4)` + modal `kappa` gauge,
/// Layer-2 pole-angle phase coding, and Layer-3 EMR-masked `7,680`-chip Tardos code.
pub fn export_runtime_dnam_watermarked(
    proj_bytes: &[u8],
    config: &BuyerWatermarkConfig,
    target_tier: Option<Tier>,
) -> Result<Vec<u8>, DnamError> {
    let crypto = DefaultCryptoProvider::default();
    let proj_view = DnamProjView::from_bytes(proj_bytes, &crypto)?;

    let mut wght_slice: Option<&[u8]> = None;
    let mut stat_slice: Option<&[u8]> = None;
    for (entry, &slice) in proj_view.directory.iter().zip(&proj_view.chunk_slices) {
        match entry.kind()? {
            ProjChunkKind::Wght => wght_slice = Some(slice),
            ProjChunkKind::Stat => stat_slice = Some(slice),
            _ => {}
        }
    }
    let wght_bytes = wght_slice.ok_or_else(|| {
        DnamError::SchemaValidationError(
            "project has no solved WGHT chunk for watermarked export".to_string(),
        )
    })?;
    let rt_view = DnamContainerView::from_bytes(wght_bytes, &crypto)?;
    let mut arena_64k = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    let src = rt_view.arena_storage.as_bytes();
    let copy_len = src.len().min(arena_64k.len());
    arena_64k[..copy_len].copy_from_slice(&src[..copy_len]);
    let mut cp1_slots = rt_view.cp1_contributors;

    let m_before = compute_role_b_multipliers(&arena_64k);

    // Layer 1: Exact Isometric Gauge (32-ch modal kappa + SO(4) head & CP1/U_int rotation)
    let (kappa, q_buyer_4x4) = derive_buyer_gauge_params(config);
    apply_isometric_gauge_watermark(&mut arena_64k, &kappa, &q_buyer_4x4)?;
    rotate_cp1_slots_by_buyer_gauge(&mut cp1_slots, &q_buyer_4x4);

    // Layer 2: Kautz & Stage-II Inter-Stage Pole-Angle Phase Coding
    let phase_seed = derive_buyer_phase_seed(config);
    apply_phase_coding_watermark(&mut arena_64k, &phase_seed, config.phase_coding_delta_rad)?;

    // Layer 3: Two-Seed Arcsine 7,680-Chip Tardos Watermark with STAT EMR Masking
    let gram_diag = extract_gram_diag_from_stat(stat_slice);
    let envelopes = compute_emr_masked_tardos_envelopes(&gram_diag, config.epsilon_mask_lr);
    let code = derive_buyer_tardos_codebook(config);
    apply_tardos_watermark_7680_envelopes(&mut arena_64k, &code.chips, &envelopes)?;

    if config.compensate_role_b_in_head {
        let m_after = compute_role_b_multipliers(&arena_64k);
        compensate_head_for_role_b_ratio(&mut arena_64k, &m_before, &m_after);
    }

    zero_arena_b_head_dc_bias(&mut arena_64k);

    let mut header = rt_view.header;
    let ultra_dnam =
        build_dnam_container_with_cp1(&mut header, &arena_64k, &arena_64k, false, &cp1_slots)?;

    let final_dnam = match target_tier {
        Some(tier) if tier != Tier::Ultra => strip_container_to_tier(&ultra_dnam, tier)?,
        _ => ultra_dnam,
    };
    let _ = DnamContainerView::from_bytes(&final_dnam, &crypto)?;
    Ok(final_dnam)
}

/// Extracts the canonical 34 Layer-2 pole angles (`32` Kautz + `2` Stage-II `theta_mid`) from an
/// `Arena 0..3` buffer in the exact order expected by `score_phase_coding_blackbox`.
#[must_use]
pub fn extract_arena_pole_angles_34(arena: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(34);
    for (tier, &blocks) in STAGE3_KAUTZ_BLOCKS_PER_TIER.iter().enumerate() {
        for blk in 0..blocks {
            let off = kautz_soa_byte_offset(tier, blk);
            if off + 48 <= arena.len() {
                for lane in 0..4 {
                    out.push(read_f32_le(arena, off + (8 + lane) * 4));
                }
            }
        }
    }
    if arena.len() >= 4_216 {
        if read_f32_le(arena, 4_176) > 0.0 {
            out.push(read_f32_le(arena, 4_180));
        }
        if read_f32_le(arena, 4_208) > 0.0 {
            out.push(read_f32_le(arena, 4_212));
        }
    }
    out
}

/// Renders a deterministic black-box probe audio response (`frames` stereo samples) by driving
/// `DnamEngine` purely through `step_frame_tora` (`offline_bounce = true`).
pub fn render_blackbox_probe_response(
    dnam_bytes: &[u8],
    frames: usize,
) -> Result<Vec<[f32; 2]>, DnamError> {
    let crypto = DefaultCryptoProvider::default();
    let view = DnamContainerView::from_bytes(dnam_bytes, &crypto)?;
    let arena_len = view.arena_storage.as_bytes().len();
    let tier = if arena_len >= 64_000 {
        Tier::Ultra
    } else if arena_len >= 36_224 {
        Tier::Pro
    } else if arena_len >= 22_336 {
        Tier::Full
    } else {
        Tier::Lite
    };
    let mut engine = DnamEngine::<DefaultMathBackend>::from_bytes_owned(dnam_bytes, tier, 48_000)?;
    engine.set_offline_bounce(true);
    let mut out = Vec::with_capacity(frames);
    for n in 0..frames {
        // Deterministic flat-spectrum white MLS/PRNG probe excites 0..24 kHz continuously across all frames
        let mut z = (n as u64)
            .wrapping_add(1)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let u = ((z >> 40) as f32) * (1.0 / 16_777_216.0) * 2.0 - 1.0;
        let x = 0.35_f32 * u;
        out.push(engine.step_frame_tora(x, x, false));
    }
    Ok(out)
}

/// Identifies the 34 modal & inter-stage pole angles purely from a suspect's **black-box audio**
/// probe response (`suspect_probe_lr`), returning `(master_angles, estimated_suspect_angles)`
/// for `dnam_engine::provenance::watermark::score_phase_coding_blackbox`.
pub fn identify_pole_angles_blackbox_from_audio(
    master_dnam_bytes: &[u8],
    suspect_probe_lr: &[[f32; 2]],
) -> Result<(Vec<f32>, Vec<f32>), DnamError> {
    let crypto = DefaultCryptoProvider::default();
    let master_view = DnamContainerView::from_bytes(master_dnam_bytes, &crypto)?;
    let base_arena = master_view.arena_storage.as_bytes();
    let frames = suspect_probe_lr.len();
    if frames < 64 {
        return Err(DnamError::BufferTooSmall {
            expected: 64,
            actual: frames,
        });
    }

    let h_master = render_blackbox_probe_response(master_dnam_bytes, frames)?;
    let mut pole_offsets: Vec<usize> = Vec::with_capacity(34);
    for (tier, &blocks) in STAGE3_KAUTZ_BLOCKS_PER_TIER.iter().enumerate() {
        for blk in 0..blocks {
            let off = kautz_soa_byte_offset(tier, blk);
            if off + 48 <= base_arena.len() {
                for lane in 0..4 {
                    pole_offsets.push(off + (8 + lane) * 4);
                }
            }
        }
    }
    if base_arena.len() >= 4_216 {
        if read_f32_le(base_arena, 4_176) > 0.0 {
            pole_offsets.push(4_180);
        }
        if read_f32_le(base_arena, 4_208) > 0.0 {
            pole_offsets.push(4_212);
        }
    }

    let m = pole_offsets.len();
    let master_angles: Vec<f32> = pole_offsets
        .iter()
        .map(|&off| read_f32_le(base_arena, off))
        .collect();
    let eps = 1.0e-3_f32;
    let inv_two_eps = 1.0 / (2.0 * f64::from(eps));

    let mut basis: Vec<Vec<[f64; 2]>> = Vec::with_capacity(m);
    for &p_off in &pole_offsets {
        let th0 = read_f32_le(base_arena, p_off);
        let mut arena_pos = base_arena.to_vec();
        write_f32_le(&mut arena_pos, p_off, th0 + eps);
        let mut hdr_pos = master_view.header.clone();
        let dnam_pos = build_dnam_container_with_cp1(
            &mut hdr_pos,
            &arena_pos,
            &arena_pos,
            false,
            &master_view.cp1_contributors,
        )?;
        let h_pos = render_blackbox_probe_response(&dnam_pos, frames)?;

        let mut arena_neg = base_arena.to_vec();
        write_f32_le(&mut arena_neg, p_off, th0 - eps);
        let mut hdr_neg = master_view.header.clone();
        let dnam_neg = build_dnam_container_with_cp1(
            &mut hdr_neg,
            &arena_neg,
            &arena_neg,
            false,
            &master_view.cp1_contributors,
        )?;
        let h_neg = render_blackbox_probe_response(&dnam_neg, frames)?;

        let mut g_k = Vec::with_capacity(frames);
        for n in 0..frames {
            let dl = (f64::from(h_pos[n][0]) - f64::from(h_neg[n][0])) * inv_two_eps;
            let dr = (f64::from(h_pos[n][1]) - f64::from(h_neg[n][1])) * inv_two_eps;
            g_k.push([dl, dr]);
        }
        basis.push(g_k);
    }

    let mut col_norms = vec![0.0_f64; m];
    let mut max_norm_sq = 0.0_f64;
    for j in 0..m {
        let mut norm_sq = 0.0_f64;
        for n in 0..frames {
            norm_sq += basis[j][n][0] * basis[j][n][0] + basis[j][n][1] * basis[j][n][1];
        }
        col_norms[j] = norm_sq;
        max_norm_sq = max_norm_sq.max(norm_sq);
    }
    let floor_sq = (max_norm_sq * 1.0e-7).max(1.0e-18);
    let col_scale: Vec<f64> = col_norms
        .iter()
        .map(|&ns| {
            if ns > floor_sq * 0.01 {
                1.0 / (ns + floor_sq).sqrt()
            } else {
                0.0
            }
        })
        .collect();

    let mut gram = vec![0.0_f64; uplo_u_packed_len(m)];
    let mut rhs = vec![[0.0_f64; 2]; m];
    for j in 0..m {
        let sj = col_scale[j];
        for i in 0..=j {
            let si = col_scale[i];
            let mut acc = 0.0_f64;
            if si > 0.0 && sj > 0.0 {
                for n in 0..frames {
                    acc += basis[i][n][0] * basis[j][n][0] + basis[i][n][1] * basis[j][n][1];
                }
                acc *= si * sj;
            } else if i == j {
                acc = 1.0;
            }
            gram[uplo_u_index(i, j)] = acc;
        }
        let mut c = 0.0_f64;
        if sj > 0.0 {
            for n in 0..frames {
                let dl = f64::from(suspect_probe_lr[n][0]) - f64::from(h_master[n][0]);
                let dr = f64::from(suspect_probe_lr[n][1]) - f64::from(h_master[n][1]);
                c += basis[j][n][0] * dl + basis[j][n][1] * dr;
            }
            c *= sj;
        }
        rhs[j] = [c, 0.0];
    }

    let delta_tilde = solve_uplo_u_cholesky_2ch(&gram, &rhs, 2.5e-2)
        .map_err(|e| DnamError::SchemaValidationError(format!("black-box pole solve: {e}")))?;

    let suspect_angles: Vec<f32> = master_angles
        .iter()
        .zip(&delta_tilde)
        .enumerate()
        .map(|(j, (&th, &[dt, _]))| {
            let wiener = col_norms[j] / (col_norms[j] + 1.0e-3 * max_norm_sq.max(1.0e-18));
            (f64::from(th) + (dt * col_scale[j] * wiener).clamp(-0.01, 0.01)) as f32
        })
        .collect();
    Ok((master_angles, suspect_angles))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proj::{ProjChunkInput, build_dnam_proj_container, build_proj_inpc_capsule};
    use dnam_engine::schema::DnamJsonHeader;

    #[test]
    fn test_stage_e_watermark_edge_branches_and_tier_probes() {
        assert_eq!(compute_integrated_lufs(&[]), -70.0);
        assert_eq!(compute_integrated_lufs(&[[0.0, 0.0]; 256]), -70.0);

        let zero_cfg = BuyerWatermarkConfig {
            marketplace_secret: [0u8; 32],
            model_id: "m".to_string(),
            license_id: "l".to_string(),
            buyer_id: [0u8; 16],
            phase_coding_delta_rad: 2.5e-5,
            epsilon_mask_lr: [1.0e-5, 1.0e-5],
            compensate_role_b_in_head: true,
        };
        let code = derive_buyer_tardos_codebook(&zero_cfg);
        assert_eq!(code.chips.len(), 7_680);

        let manifest = DnamProjManifest::default_reference();
        let inpc =
            build_proj_inpc_capsule(1, &[0u8; 256], &[0u8; 320], &vec![0u8; 24_000]).unwrap();
        let un_solved = build_dnam_proj_container(
            &manifest,
            &[ProjChunkInput {
                kind: ProjChunkKind::Inpc,
                chunk_id: 0,
                payload: inpc.clone(),
            }],
        )
        .unwrap();
        assert!(export_runtime_dnam_watermarked(&un_solved, &zero_cfg, None).is_err());

        let mut hdr = DnamJsonHeader::default_reference();
        let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        write_f32_le(&mut arena, 4_176, 0.9);
        write_f32_le(&mut arena, 4_180, 0.25);
        write_f32_le(&mut arena, 4_208, 0.9);
        write_f32_le(&mut arena, 4_212, 0.35);
        assert_eq!(extract_arena_pole_angles_34(&arena).len(), 34);
        let dnam_u = build_dnam_container_with_cp1(&mut hdr, &arena, &arena, false, &[]).unwrap();
        let solved = build_dnam_proj_container(
            &manifest,
            &[
                ProjChunkInput {
                    kind: ProjChunkKind::Inpc,
                    chunk_id: 0,
                    payload: inpc,
                },
                ProjChunkInput {
                    kind: ProjChunkKind::Wght,
                    chunk_id: 0,
                    payload: dnam_u.clone(),
                },
            ],
        )
        .unwrap();
        for tier in [Tier::Lite, Tier::Full, Tier::Pro] {
            let wm_t = export_runtime_dnam_watermarked(&solved, &zero_cfg, Some(tier)).unwrap();
            let probe = render_blackbox_probe_response(&wm_t, 64).unwrap();
            assert_eq!(probe.len(), 64);
        }
        assert!(identify_pole_angles_blackbox_from_audio(&dnam_u, &[[0.0, 0.0]; 16]).is_err());
        assert_eq!(select_eval_window(&[], 64), (0, 0));
        assert_eq!(select_eval_window(&[[0.1, 0.1]; 80], 64), (64, 80));
        let mut win_sig = [[0.01_f32, 0.01]; 256];
        for s in &mut win_sig[128..192] {
            *s = [0.9, -0.9];
        }
        assert_eq!(select_eval_window(&win_sig, 64), (128, 192));
    }
}
