#![forbid(unsafe_code)]
//! Stage B (`< 10 ms` Full-Rank `256-D` Feature Harvest & 3-Band Bézout `Wide192Accum` Contraction)
//! and Stage C (`< 5 ms` 3-Band Bézout Multi-Tier Nested SPD Normal Equations, Theorem 5 Selective
//! `1/12`-Octave `> 1.5 kHz` `MM-Cholesky` & `33`-Band `LTAS` KKT Polish).
//!
//! Canonical References:
//! - `docs/specs/training/four_stage_pipeline.md`
//!   (`ref[train.pipeline.stage-b-harvest]`, `ref[train.pipeline.stage-c-schur]`)
//! - `docs/specs/containers/project_container_format.md` (`ref[container.proj.stat-uplo-u]`)

use dnam_engine::container::build_dnam_container_with_cp1;
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::schema::DnamError;
use dnam_engine::types::FtzDazGuard;
use dnam_engine::{DnamEngine, Tier};

use crate::proj::PROJ_STAT_BYTES_ULTRA;
use crate::proj::bundle::zero_arena_b_head_dc_bias;
use crate::proj::schema::DnamProjManifest;

use super::bezout_gram::{
    BezoutHarvestStats, contract_bezout_3band_stats_4tier, eval_4tier_post_trim_besr_lin,
};
use super::bezout_schur::{apply_selective_hf_mm_and_ltas_polish, solve_multitier_besr_4tier};
use super::stage_a::{CASCADE_BLOCKS_PER_TIER, CASCADE_W_BASES};
use super::wide_gram::contract_256d_gram_tiled_wide192;
use super::{
    bezout_three_band_weights, factorize_uplo_u_cholesky, solve_factorized_uplo_u_2ch,
    solve_uplo_u_cholesky_2ch, uplo_u_index, uplo_u_packed_len,
};

/// Mode 0..4 stereo output head base offsets across `Arena 0..3`.
pub const HEAD_BASES: [usize; 4] = [4_928, 18_048, 31_936, 57_600];

/// Ultra Mode 5..8 stereo output head base offset in `Arena 3` (`4 * 2,048 B = 8,192 B`).
pub const ULTRA_MODES_5_TO_8_BASE: usize = 49_408;

/// Harvested `256-D` `StarCenter` trajectory and contracted `Wide192Accum` sufficient statistics.
#[derive(Debug, Clone)]
pub struct StageBHarvest {
    /// Per-sample `256-D` feature bus across `StarCenter` coreset frames (full or strided).
    pub features: Vec<[f32; 256]>,
    /// Per-sample stereo non-homogeneous Stage 0–IV backbone output `y_backbone[n]` in `f32`.
    pub backbone_out_lr: Vec<[f32; 2]>,
    /// Per-sample stereo residual $y_{\text{target}}[n] - y_{\text{backbone}}[n]$ in `f64`.
    pub residual_lr: Vec<[f64; 2]>,
    /// Per-sample psychoacoustic `EMR` + transient weights (normalized to mean `1.0`).
    pub frame_weights: Vec<f64>,
    /// Contracted broadband `uplo='U'` `256 × 256` Gram matrix (`32,896 × f64`) for `STAT`.
    pub packed_gram_256: Vec<f64>,
    /// Contracted broadband `256 × 2` cross-correlation matrix.
    pub cross_256: Vec<[f64; 2]>,
    /// Weighted stereo residual energy $\sum_n w_n \|\mathbf{r}[n]\|_2^2$.
    pub weighted_residual_energy: f64,
    /// Weighted stereo target energy $\sum_n w_n \|\mathbf{y}[n]\|_2^2$.
    pub weighted_target_energy: f64,
    /// Static Role B (`b = 1..K_t-1`) multiplicative gauge factors per `(feature_idx, ch)`.
    pub role_b_multipliers: [[f64; 2]; 256],
    /// Exact 3-band Binomial/Bézout (`LF`, `MF`, `HF`) contracted sufficient statistics across 4 tiers.
    pub besr_stats: Option<BezoutHarvestStats>,
    /// Whether stereo Right output channel is distinct (`output_mode != "mono"`).
    pub eval_stereo: bool,
    /// Contiguous `256`-frame feature window `[win_start..win_start + 256]` when `features` is strided.
    pub(crate) polish_win_features: Vec<[f32; 256]>,
    /// Wall-clock Stage B duration in milliseconds.
    pub elapsed_ms: f32,
}

/// Telemetry and `STAT` payload emitted by [`solve_stage_c_nested_schur_4tier`].
#[derive(Debug, Clone)]
pub struct StageCSolveOutput {
    /// Effective `256 × 2` Mode-0 weights (`w_stored * role_b_multipliers`).
    pub effective_w0_256: Vec<[f64; 2]>,
    /// Serialized `267,264 B` `STAT` chunk (`G_0` + pre-baked `C_{0,eff}^{MM-KKT}`).
    pub stat_bytes: Vec<u8>,
    /// Schur-complement homotopy blend factor $\eta \in [0, 1]$ (`0.0` = pure `256-D` joint).
    pub schur_homotopy_eta: f64,
    /// Number of `> 1.5 kHz` `1/12`-octave bands relaxed by Theorem 5 (`|\Delta L_b| > 0.25 dB`).
    pub mm_relaxed_hf_bands: usize,
    /// Closed-form prefix `bESR` in dB across `[Lite, Full, Pro, Ultra]`.
    pub prefix_esr_db: [f32; 4],
    /// Wall-clock Stage C duration in milliseconds.
    pub elapsed_ms: f32,
}

#[inline]
fn now_ms_token() -> Option<std::time::Instant> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        Some(std::time::Instant::now())
    }
    #[cfg(target_arch = "wasm32")]
    {
        None
    }
}

#[inline]
fn elapsed_ms_since(token: Option<std::time::Instant>) -> f32 {
    token.map_or(0.0, |t| t.elapsed().as_secs_f32() * 1_000.0)
}

#[inline]
fn read_f32_le(buf: &[u8], off: usize) -> f32 {
    if off + 4 <= buf.len() {
        f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
    } else {
        0.0
    }
}

/// Computes the exact static Role B (`b = 1..K_t-1`) multiplicative gauge factors cached by
/// `DnamEngine::refresh_cached_role_b_slices` (`1.0` everywhere when Role B is zeroed).
#[must_use]
pub fn compute_role_b_multipliers(arena_64k: &[u8]) -> [[f64; 2]; 256] {
    let mut out = [[1.0_f64; 2]; 256];
    for (t, (&cw_base, &k_t)) in CASCADE_W_BASES
        .iter()
        .zip(&CASCADE_BLOCKS_PER_TIER)
        .enumerate()
    {
        let mut m_slices = [[1.0_f32; 4]; 32];
        if arena_64k.len() >= cw_base + k_t * 64 {
            for r in 0..(k_t - 1) {
                let blk_off = cw_base + (r + 1) * 64;
                for q in 0..4 {
                    let s = (4 * r + q) & 31;
                    let q_off = blk_off + q * 16;
                    for lane in 0..4 {
                        m_slices[s][lane] += read_f32_le(arena_64k, q_off + lane * 4);
                    }
                }
            }
        }
        for i in 0..64 {
            let g = i / 4;
            let lane = i & 3;
            out[t * 64 + i][0] = f64::from(m_slices[g][lane]);
            out[t * 64 + i][1] = f64::from(m_slices[16 + g][lane]);
        }
    }
    out
}

/// Zeroes all Mode 0..8 output head weights, inline/Ultra `CP1` & `U_int` tables, and `b_head`
/// DC biases on `backbone_arena` so `step_frame_with_feature_bus` emits the pure Stage 0–IV signal.
pub fn strip_all_head_modes_on_arena(backbone_arena: &mut [u8]) {
    for &h_base in &HEAD_BASES {
        let end = (h_base + 5 * 512).min(backbone_arena.len());
        if h_base < end {
            backbone_arena[h_base..end].fill(0);
        }
    }
    if backbone_arena.len() >= ULTRA_MODES_5_TO_8_BASE + 8_192 {
        backbone_arena[ULTRA_MODES_5_TO_8_BASE..ULTRA_MODES_5_TO_8_BASE + 8_192].fill(0);
    }
    if backbone_arena.len() >= 8_512 {
        backbone_arena[7_524..8_512].fill(0);
    }
    if backbone_arena.len() >= 63_424 {
        backbone_arena[60_928..63_424].fill(0);
    }
    // Reset tier output trims to unity while harvesting backbone trajectories
    if backbone_arena.len() >= 10_688 {
        for slot in 0..16 {
            let off = 10_624 + slot * 4;
            backbone_arena[off..off + 4].copy_from_slice(&1.0_f32.to_le_bytes());
        }
    }
    zero_arena_b_head_dc_bias(backbone_arena);
}

/// Computes psychoacoustic `EMR` + transient onset sample weights across `(dry_lr, wet_lr)`,
/// normalized so $\frac{1}{N}\sum_n w_n = 1.0$.
pub(crate) fn compute_psychoacoustic_emr_weights(
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
) -> Vec<f64> {
    let n = dry_lr.len().min(wet_lr.len());
    if n == 0 {
        return Vec::new();
    }
    let mut weights = vec![1.0_f64; n];
    let mut fast_env = 0.0_f64;
    let mut slow_env = 1.0e-4_f64;
    let mut w_sum = 0.0_f64;

    for i in 0..n {
        let mag = 0.5 * (f64::from(wet_lr[i][0]).abs() + f64::from(wet_lr[i][1]).abs())
            + 0.25 * (f64::from(dry_lr[i][0]).abs() + f64::from(dry_lr[i][1]).abs());
        fast_env = 0.88 * fast_env + 0.12 * mag;
        slow_env = 0.996 * slow_env + 0.004 * mag;
        let crest_s = ((fast_env / slow_env.max(1.0e-5)) - 0.5).clamp(0.0, 2.0) * 0.5;
        let (h_low, h_mid, h_high) = bezout_three_band_weights(crest_s);
        let w = (0.85 * h_low + 1.05 * h_mid + 1.35 * h_high).clamp(0.5, 2.0);
        weights[i] = w;
        w_sum += w;
    }

    let inv_mean = (n as f64) / w_sum.max(1.0e-12);
    for w in &mut weights {
        *w *= inv_mean;
    }
    weights
}

/// Executes Stage B (`< 10 ms` Full-Rank `256-D` Feature Harvest & 3-Band Bézout `Wide192Accum`
/// Contraction, `r[train.pipeline.stage-b-harvest]`): steps `DnamEngine` across all `StarCenter`
/// frames and contracts both broadband `STAT` ($\mathbf{G}_0, \mathbf{C}_0$) and 3-band Bézout
/// normal equations across `[Lite, Full, Pro, Ultra]`.
pub fn harvest_star_center_256d(
    manifest: &DnamProjManifest,
    arena_64k: &[u8],
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
    star_controls: &[f32],
    gram_frame_stride: usize,
) -> Result<StageBHarvest, DnamError> {
    let timer = now_ms_token();
    let n = dry_lr.len().min(wet_lr.len());
    let mut backbone_arena = arena_64k.to_vec();
    strip_all_head_modes_on_arena(&mut backbone_arena);
    let _ = super::stages::sync_clamp_descriptors_to_arena(&manifest.base, &mut backbone_arena)?;

    let mut temp_hdr = manifest.base.clone();
    let dnam_bytes =
        build_dnam_container_with_cp1(&mut temp_hdr, &backbone_arena, &backbone_arena, false, &[])?;
    let mut engine =
        DnamEngine::<DefaultMathBackend>::from_bytes(&dnam_bytes, Tier::Ultra, 48_000)?;
    for (p, &val) in star_controls.iter().enumerate() {
        engine.set_control_immediate(p, val);
    }

    let eval_stereo = manifest.base.io_config.output_mode != "mono";
    let mono_only = !eval_stereo;
    let frame_weights = compute_psychoacoustic_emr_weights(&dry_lr[..n], &wet_lr[..n]);
    let stride = gram_frame_stride.max(1);
    let mut residual_lr = Vec::with_capacity(n);
    let mut backbone_out_lr = Vec::with_capacity(n);
    let mut full_features = vec![[0.0_f32; 256]; n];
    let mut weighted_res_e = 0.0_f64;
    let mut weighted_tgt_e = 0.0_f64;

    let mut tier_backbone_lr: [Vec<[f32; 2]>; 4] = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::new(),
    ];
    let mut lower_tier_features: [Vec<[f32; 256]>; 3] = [
        vec![[0.0_f32; 256]; n],
        vec![[0.0_f32; 256]; n],
        vec![[0.0_f32; 256]; n],
    ];

    {
        let _ftz = FtzDazGuard::new();
        for (i, &[dl, dr]) in dry_lr[..n].iter().enumerate() {
            let [bl, br] = engine.step_frame_with_feature_bus(dl, dr, &mut full_features[i]);
            backbone_out_lr.push([bl, br]);
            let yl = f64::from(wet_lr[i][0]);
            let yr = if eval_stereo {
                f64::from(wet_lr[i][1])
            } else {
                yl
            };
            let rl = yl - f64::from(bl);
            let rr = if eval_stereo { yr - f64::from(br) } else { rl };
            residual_lr.push([rl, rr]);
            let w = frame_weights[i];
            weighted_res_e += w * (rl * rl + rr * rr);
            weighted_tgt_e += w * (yl * yl + yr * yr);
        }

        // Harvest exact zero-head backbone AND exact per-tier feature-bus trajectories at
        // Lite (64-D), Full (128-D), and Pro (192-D) so Stage IV input coupling across tiers is exact.
        for (t_idx, tier) in [Tier::Lite, Tier::Full, Tier::Pro].into_iter().enumerate() {
            engine.set_tier(tier);
            engine.reset_to_dc_equilibrium();
            for (p, &val) in star_controls.iter().enumerate() {
                engine.set_control_immediate(p, val);
            }
            for (i, &[dl, dr]) in dry_lr[..n].iter().enumerate() {
                let [bl, br] =
                    engine.step_frame_with_feature_bus(dl, dr, &mut lower_tier_features[t_idx][i]);
                tier_backbone_lr[t_idx].push([bl, if mono_only { bl } else { br }]);
            }
        }
    }
    tier_backbone_lr[3].clone_from(&backbone_out_lr);

    let besr_stats = Some(contract_bezout_3band_stats_4tier(
        &[
            &lower_tier_features[0],
            &lower_tier_features[1],
            &lower_tier_features[2],
            &full_features,
        ],
        &wet_lr[..n],
        &tier_backbone_lr,
        eval_stereo,
        stride,
    ));

    let (features, polish_win_features, packed_gram_256, cross_256) = if stride == 1 {
        let (g, c) =
            contract_256d_gram_tiled_wide192(&full_features, &residual_lr, &frame_weights, 1);
        (full_features, Vec::new(), g, c)
    } else {
        const WIN_N: usize = 256;
        let win_start = if n >= WIN_N {
            (n / 3).min(n - WIN_N)
        } else {
            n
        };
        let win_end = (win_start + WIN_N).min(n);
        let m_sub = n.div_ceil(stride);
        let scale = if m_sub > 0 {
            (n as f64) / (m_sub as f64)
        } else {
            1.0
        };
        let mut strided_features = Vec::with_capacity(m_sub);
        let mut strided_res = Vec::with_capacity(m_sub);
        let mut strided_weights = Vec::with_capacity(m_sub);
        for i in (0..n).step_by(stride) {
            strided_features.push(full_features[i]);
            strided_res.push(residual_lr[i]);
            strided_weights.push(frame_weights[i] * scale);
        }
        let polish_win = if win_end > win_start {
            full_features[win_start..win_end].to_vec()
        } else {
            Vec::new()
        };
        let (g, c) =
            contract_256d_gram_tiled_wide192(&strided_features, &strided_res, &strided_weights, 1);
        (strided_features, polish_win, g, c)
    };
    let role_b_multipliers = compute_role_b_multipliers(arena_64k);

    Ok(StageBHarvest {
        features,
        backbone_out_lr,
        residual_lr,
        frame_weights,
        packed_gram_256,
        cross_256,
        weighted_residual_energy: weighted_res_e,
        weighted_target_energy: weighted_tgt_e,
        role_b_multipliers,
        besr_stats,
        eval_stereo,
        polish_win_features,
        elapsed_ms: elapsed_ms_since(timer),
    })
}

pub(super) fn solve_cholesky_robust(
    packed_gram: &[f64],
    cross: &[[f64; 2]],
    base_ridge: f64,
) -> Result<Vec<[f64; 2]>, DnamError> {
    let mut ridge = base_ridge.max(1.0e-12);
    for _ in 0..6 {
        if let Ok(sol) = solve_uplo_u_cholesky_2ch(packed_gram, cross, ridge) {
            return Ok(sol);
        }
        ridge *= 10.0;
    }
    solve_uplo_u_cholesky_2ch(packed_gram, cross, ridge.max(1.0e-3)).map_err(|e| {
        DnamError::SchemaValidationError(format!("Cholesky factorization failed: {e}"))
    })
}

fn factorize_cholesky_robust(
    dim: usize,
    packed_gram: &[f64],
    base_ridge: f64,
) -> Result<Vec<f64>, DnamError> {
    let mut ridge = base_ridge.max(1.0e-12);
    for _ in 0..6 {
        if let Ok(r_up) = factorize_uplo_u_cholesky(dim, packed_gram, ridge) {
            return Ok(r_up);
        }
        ridge *= 10.0;
    }
    factorize_uplo_u_cholesky(dim, packed_gram, ridge.max(1.0e-3)).map_err(|e| {
        DnamError::SchemaValidationError(format!("Cholesky factorization failed: {e}"))
    })
}

fn extract_block_uplo_u(packed_256: &[f64], start: usize, len: usize) -> Vec<f64> {
    let mut sub = vec![0.0_f64; uplo_u_packed_len(len)];
    for j in 0..len {
        let src_col = uplo_u_index(0, start + j) + start;
        let dst_col = uplo_u_index(0, j);
        sub[dst_col..dst_col + j + 1].copy_from_slice(&packed_256[src_col..src_col + j + 1]);
    }
    sub
}

/// Evaluates closed-form quadratic residual energies across all 4 tier prefixes (`64, 128, 192, 256`).
pub(super) fn eval_all_4_prefix_residuals(
    packed_gram_256: &[f64],
    cross_256: &[[f64; 2]],
    w_256: &[[f64; 2]],
    base_res_energy: f64,
) -> [f64; 4] {
    let mut quad = base_res_energy;
    let mut out = [0.0_f64; 4];
    for j in 0..256 {
        let wj0 = w_256[j][0];
        let wj1 = w_256[j][1];
        quad -= 2.0 * (wj0 * cross_256[j][0] + wj1 * cross_256[j][1]);
        let col_base = uplo_u_index(0, j);
        let g_jj = packed_gram_256[col_base + j];
        quad += g_jj * (wj0 * wj0 + wj1 * wj1);
        for (i, &g_ij) in packed_gram_256[col_base..col_base + j].iter().enumerate() {
            quad += 2.0 * g_ij * (w_256[i][0] * wj0 + w_256[i][1] * wj1);
        }
        match j {
            63 => out[0] = quad.max(0.0),
            127 => out[1] = quad.max(0.0),
            191 => out[2] = quad.max(0.0),
            255 => out[3] = quad.max(0.0),
            _ => {}
        }
    }
    out
}

/// Serializes a `256 × 256` `uplo='U'` Gram matrix and `256 × 2` cross-correlation matrix into
/// the canonical `267,264 B` `STAT` chunk (`r[container.proj.stat-uplo-u]`).
#[must_use]
pub fn serialize_stat_chunk_ultra(packed_gram_256: &[f64], cross_256: &[[f64; 2]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(PROJ_STAT_BYTES_ULTRA);
    for &g in packed_gram_256 {
        out.extend_from_slice(&g.to_le_bytes());
    }
    for &[cl, cr] in cross_256 {
        out.extend_from_slice(&cl.to_le_bytes());
        out.extend_from_slice(&cr.to_le_bytes());
    }
    debug_assert_eq!(out.len(), PROJ_STAT_BYTES_ULTRA);
    out
}

/// Writes effective `256 × 2` Mode-0 weights into `HEAD_BASES[0..4]`, dividing by `role_b_multipliers`
/// so non-zero Role B arenas (`Oracle-01`) reconstruct `w_eff` to `0–1 ULP`.
pub fn write_effective_mode0_weights(
    arena_64k: &mut [u8],
    w_eff_256: &[[f64; 2]],
    role_b_multipliers: &[[f64; 2]; 256],
) {
    for (t, &h_base) in HEAD_BASES.iter().enumerate() {
        for i in 0..64 {
            let idx = t * 64 + i;
            let ml = role_b_multipliers[idx][0];
            let mr = role_b_multipliers[idx][1];
            let inv_ml = if ml.abs() > 1.0e-6 { 1.0 / ml } else { 1.0 };
            let inv_mr = if mr.abs() > 1.0e-6 { 1.0 / mr } else { 1.0 };
            let wl = ((w_eff_256[idx][0] * inv_ml) as f32).clamp(-16.0, 16.0);
            let wr = ((w_eff_256[idx][1] * inv_mr) as f32).clamp(-16.0, 16.0);
            let l_off = h_base + i * 4;
            let r_off = h_base + 256 + i * 4;
            if r_off + 4 <= arena_64k.len() {
                arena_64k[l_off..l_off + 4].copy_from_slice(&wl.to_le_bytes());
                arena_64k[r_off..r_off + 4].copy_from_slice(&wr.to_le_bytes());
            }
        }
    }
    zero_arena_b_head_dc_bias(arena_64k);
}

/// Executes Stage C (`< 5 ms` 3-Band Bézout Multi-Tier Nested SPD Normal Equations,
/// `r[train.pipeline.stage-c-schur]`): solves `Lite (0..64) -> Full (64..128) -> Pro (128..192) -> Ultra (192..256)`
/// directly minimizing 3-band `bESR` with guaranteed monotone prefix `bESR`, writes `HEAD_BASES[0..4]`,
/// and serializes `267,264 B` `STAT`.
pub fn solve_stage_c_nested_schur_4tier(
    arena_64k: &mut [u8],
    harvest: Option<&StageBHarvest>,
    packed_gram_256: &[f64],
    cross_256: &[[f64; 2]],
    role_b_multipliers: &[[f64; 2]; 256],
) -> Result<StageCSolveOutput, DnamError> {
    let timer = now_ms_token();
    let mut trace_g = 0.0_f64;
    for d in 0..256 {
        trace_g += packed_gram_256[uplo_u_index(d, d)];
    }
    let mean_diag = (trace_g / 256.0).max(1.0e-9);
    let base_ridge = (1.0e-7 * mean_diag).max(1.0e-10);
    let cond_ridge = (1.0e-5 * mean_diag).max(1.0e-9);

    let (chosen_w, chosen_eta, mm_relaxed_hf_bands, prefix_esr_db) = if let Some(h) = harvest
        && let Some(ref besr_stats) = h.besr_stats
    {
        let mut sol = solve_multitier_besr_4tier(besr_stats, role_b_multipliers, h.eval_stereo)?;
        let mm_bands = apply_selective_hf_mm_and_ltas_polish(
            h,
            packed_gram_256,
            base_ridge,
            role_b_multipliers,
            &mut sol.effective_w0_256,
        );
        if mm_bands > 0 {
            let mut w_arr = [[0.0_f64; 2]; 256];
            w_arr.copy_from_slice(&sol.effective_w0_256[..256]);
            let (b_lin, _) = eval_4tier_post_trim_besr_lin(besr_stats, &w_arr, h.eval_stereo);
            sol.prefix_besr_db = b_lin.map(|v| (10.0 * v.max(1.0e-16).log10()) as f32);
        }
        (
            sol.effective_w0_256,
            sol.max_homotopy_eta,
            mm_bands,
            sol.prefix_besr_db,
        )
    } else {
        // Matrix-only fallback when StageBHarvest is unavailable
        let mut w_joint = solve_cholesky_robust(packed_gram_256, cross_256, base_ridge)?;
        let mm_bands = if let Some(h) = harvest {
            apply_selective_hf_mm_and_ltas_polish(
                h,
                packed_gram_256,
                base_ridge,
                role_b_multipliers,
                &mut w_joint,
            )
        } else {
            0
        };

        let base_res_e = harvest.map_or_else(
            || {
                let mut y_est = 1.0_f64;
                for d in 0..256 {
                    y_est += (w_joint[d][0] * cross_256[d][0] + w_joint[d][1] * cross_256[d][1])
                        .max(0.0);
                }
                y_est
            },
            |h| h.weighted_residual_energy,
        );
        let target_e = harvest.map_or(base_res_e.max(1.0), |h| {
            h.weighted_target_energy.max(1.0e-12)
        });

        let tol = 1.0e-12 * target_e;
        let tier_ceil_e = base_res_e.min(target_e) * 1.02 + tol;
        let [e1_0, e2_0, e3_0, e4_0] =
            eval_all_4_prefix_residuals(packed_gram_256, cross_256, &w_joint, base_res_e);
        let mono_0 = e4_0 <= e3_0 + tol && e3_0 <= e2_0 + tol && e2_0 <= e1_0 + tol;
        let well_0 = e1_0 <= tier_ceil_e && e2_0 <= tier_ceil_e && e3_0 <= tier_ceil_e;

        let (cw, c_eta, chosen_e) = if mono_0 && well_0 {
            (w_joint, 0.0_f64, [e1_0, e2_0, e3_0, e4_0])
        } else {
            let sub_r_up: [Vec<f64>; 4] = [
                factorize_cholesky_robust(
                    64,
                    &extract_block_uplo_u(packed_gram_256, 0, 64),
                    cond_ridge,
                )?,
                factorize_cholesky_robust(
                    64,
                    &extract_block_uplo_u(packed_gram_256, 64, 64),
                    cond_ridge,
                )?,
                factorize_cholesky_robust(
                    64,
                    &extract_block_uplo_u(packed_gram_256, 128, 64),
                    cond_ridge,
                )?,
                factorize_cholesky_robust(
                    64,
                    &extract_block_uplo_u(packed_gram_256, 192, 64),
                    cond_ridge,
                )?,
            ];

            let mut w_seq = vec![[0.0_f64; 2]; 256];
            let mut sub_c = [[0.0_f64; 2]; 64];
            for t in 0..4 {
                let start = t * 64;
                for i in 0..64 {
                    let row = start + i;
                    let col_base = uplo_u_index(0, row);
                    let mut cl = cross_256[row][0];
                    let mut cr = cross_256[row][1];
                    for (prev, &g) in packed_gram_256[col_base..col_base + start]
                        .iter()
                        .enumerate()
                    {
                        cl -= g * w_seq[prev][0];
                        cr -= g * w_seq[prev][1];
                    }
                    sub_c[i] = [cl, cr];
                }
                let w_t = solve_factorized_uplo_u_2ch(64, &sub_r_up[t], &sub_c).map_err(|e| {
                    DnamError::SchemaValidationError(format!("64-D block backsolve failed: {e}"))
                })?;
                w_seq[start..start + 64].copy_from_slice(&w_t);
            }

            let mut best_w = w_seq.clone();
            let mut best_eta = 1.0_f64;
            let mut best_e = [e1_0, e2_0, e3_0, e4_0];
            for (attempt, eta) in [0.02_f64, 0.08, 0.20, 0.45, 1.0].into_iter().enumerate() {
                let mut cand = vec![[0.0_f64; 2]; 256];
                if eta >= 1.0 {
                    cand.clone_from(&w_seq);
                } else {
                    for t in 0..4 {
                        let start = t * 64;
                        for i in 0..64 {
                            let row = start + i;
                            let col_base = uplo_u_index(0, row);
                            let mut cl = cross_256[row][0];
                            let mut cr = cross_256[row][1];
                            for (prev, &g) in packed_gram_256[col_base..col_base + start]
                                .iter()
                                .enumerate()
                            {
                                cl -= g * cand[prev][0];
                                cr -= g * cand[prev][1];
                            }
                            sub_c[i] = [cl, cr];
                        }
                        let w_cond = solve_factorized_uplo_u_2ch(64, &sub_r_up[t], &sub_c)
                            .map_err(|e| {
                                DnamError::SchemaValidationError(format!(
                                    "64-D block backsolve failed: {e}"
                                ))
                            })?;
                        if t < 3 {
                            for i in 0..64 {
                                let d = start + i;
                                cand[d][0] = (1.0 - eta) * w_joint[d][0] + eta * w_cond[i][0];
                                cand[d][1] = (1.0 - eta) * w_joint[d][1] + eta * w_cond[i][1];
                            }
                        } else {
                            cand[192..256].copy_from_slice(&w_cond);
                        }
                    }
                }
                let [e1, e2, e3, e4] =
                    eval_all_4_prefix_residuals(packed_gram_256, cross_256, &cand, base_res_e);
                let monotone = e4 <= e3 + tol && e3 <= e2 + tol && e2 <= e1 + tol;
                let lite_well_behaved = e1 <= tier_ceil_e && e2 <= tier_ceil_e && e3 <= tier_ceil_e;
                if (monotone && lite_well_behaved) || attempt == 4 {
                    best_w = cand;
                    best_eta = eta;
                    best_e = [e1, e2, e3, e4];
                    break;
                }
            }
            (best_w, best_eta, best_e)
        };

        let mut esr_db = [0.0_f32; 4];
        for t in 0..4 {
            let ratio = (chosen_e[t] / target_e).max(1.0e-16);
            esr_db[t] = (10.0 * ratio.log10()) as f32;
        }
        (cw, c_eta, mm_bands, esr_db)
    };

    write_effective_mode0_weights(arena_64k, &chosen_w, role_b_multipliers);

    // Pre-bake C_{0,eff}^{MM-KKT} = (G_0 + base_ridge * I) * w_final into STAT (267,264 B)
    let mut c_eff = vec![[0.0_f64; 2]; 256];
    for j in 0..256 {
        let col_base = uplo_u_index(0, j);
        let wj0 = chosen_w[j][0];
        let wj1 = chosen_w[j][1];
        let g_jj = packed_gram_256[col_base + j];
        let mut sl_j = (base_ridge + g_jj) * wj0;
        let mut sr_j = (base_ridge + g_jj) * wj1;
        for (i, &g_ij) in packed_gram_256[col_base..col_base + j].iter().enumerate() {
            c_eff[i][0] += g_ij * wj0;
            c_eff[i][1] += g_ij * wj1;
            sl_j += g_ij * chosen_w[i][0];
            sr_j += g_ij * chosen_w[i][1];
        }
        c_eff[j][0] += sl_j;
        c_eff[j][1] += sr_j;
    }
    let stat_bytes = serialize_stat_chunk_ultra(packed_gram_256, &c_eff);

    Ok(StageCSolveOutput {
        effective_w0_256: chosen_w,
        stat_bytes,
        schur_homotopy_eta: chosen_eta,
        mm_relaxed_hf_bands,
        prefix_esr_db,
        elapsed_ms: elapsed_ms_since(timer),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_c_schur_homotopy_fallback_when_joint_prefix_blows_up() {
        let mut gram = vec![0.0_f64; uplo_u_packed_len(256)];
        for d in 0..256 {
            gram[uplo_u_index(d, d)] = 1.0;
        }
        // Strong cross-tier coupling between Tier 0 (d=0) and Tier 1 (d=64) causes eta=0 prefix blowup
        gram[uplo_u_index(0, 64)] = 0.99;
        let mut cross = vec![[0.0_f64; 2]; 256];
        cross[64] = [1.0, 1.0];
        let role_b = [[1.0_f64; 2]; 256];
        let mut arena_64k = vec![0u8; 64_000];

        let out = solve_stage_c_nested_schur_4tier(&mut arena_64k, None, &gram, &cross, &role_b)
            .expect("Stage C Schur homotopy should converge");
        assert!(out.schur_homotopy_eta > 0.0);
        assert_eq!(out.stat_bytes.len(), 267_264);
        assert!(out.prefix_esr_db[3] <= out.prefix_esr_db[0] + 1.0e-5);
    }

    #[test]
    fn test_bezout_gram_matches_canonical_3band_splitter_and_enforces_post_trim_monotonicity() {
        use super::super::bezout_gram::contract_bezout_3band_stats;
        use crate::metrics::try_split_bezout_3band_energies;
        let n = 1_024;
        let mut feats = vec![[0.0_f32; 256]; n];
        let mut wet_lr = vec![[0.0_f32; 2]; n];
        let zero_bb = [
            vec![[0.0_f32; 2]; n],
            vec![[0.0_f32; 2]; n],
            vec![[0.0_f32; 2]; n],
            vec![[0.0_f32; 2]; n],
        ];
        for i in 0..n {
            let t = (i as f64) / 48_000.0;
            let lf = (std::f64::consts::TAU * 110.0 * t).sin() as f32;
            let mf = (std::f64::consts::TAU * 900.0 * t).cos() as f32;
            let hf = (std::f64::consts::TAU * 6_000.0 * t).sin() as f32;
            feats[i][0] = lf + 0.98 * hf;
            feats[i][64] = hf;
            feats[i][128] = lf;
            feats[i][1] = mf;
            let y = 0.8 * lf + 0.5 * mf + 0.025 * hf;
            wet_lr[i] = [y, y];
        }

        let stats = contract_bezout_3band_stats(&feats, &wet_lr, &zero_bb, false, 1);
        let tgt_f64: Vec<f64> = wet_lr.iter().map(|v| f64::from(v[0])).collect();
        let zeros = vec![0.0_f64; n];
        let (sig_ref, _, _) =
            try_split_bezout_3band_energies(&tgt_f64, &zeros, None, 48_000.0, stats.warmup_samples)
                .unwrap();
        for b in 0..3 {
            assert!((stats.target_band_energy_lr[0][b] - sig_ref[b]).abs() < 1.0e-9);
        }

        let role_b = [[1.0_f64; 2]; 256];
        let besr_db = solve_multitier_besr_4tier(&stats, &role_b, false)
            .unwrap()
            .prefix_besr_db;
        assert!(besr_db[3] <= besr_db[2] + 1.0e-5);
        assert!(besr_db[2] <= besr_db[1] + 1.0e-5);
        assert!(besr_db[1] <= besr_db[0] + 1.0e-5);
        assert!(besr_db[3] < -40.0);
    }
}
