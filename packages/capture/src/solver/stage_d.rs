#![forbid(unsafe_code)]
//! Stage D (`< 5 ms` Arbitrary-`P` `CP1` Control Projections, Kronecker SVD Modes `1..8`,
//! Walsh-Hadamard `U_int` Factorization Machines, Vault Fast-Path Re-Solve & 3-Phase Audit).
//!
//! Canonical References:
//! - `docs/specs/training/four_stage_pipeline.md`
//!   (`ref[train.pipeline.stage-d-control]`, `ref[train.pipeline.dispatch-solve]`)
//! - `docs/specs/training/iterative_resume_and_caching.md`
//!   (`ref[train.resume.kronecker-reassembly]`, `ref[train.cache.proj-vault]`)
//! - `docs/specs/dsp/parametric_control_bus.md` (`ref[dsp.ctrl.accum.anova-fm]`)

use std::collections::BTreeMap;

use dnam_engine::control::{ControlClampDescriptor, Cp1ContributorSlot};
use dnam_engine::schema::{ControlKind, ControlSchemaDef, DnamError};

use crate::audit::{run_full_3phase_arena_audit, run_head_only_3phase_arena_audit};
use crate::proj::bundle::zero_arena_b_head_dc_bias;
use crate::proj::directory::{StmcTrajectoryWindow, verify_proj_stmc_capsule};
use crate::proj::schema::{AuditReport, DnamProjManifest, StemRegistryEntry};
use crate::proj::{STMC_FLAG_MODE_A_SWEEP, STMC_FLAG_STEREO, fp16_bytes_to_f32};
use crate::solver::{
    factorize_uplo_u_cholesky, solve_factorized_uplo_u_2ch, uplo_u_index, uplo_u_packed_len,
};

use super::stage_b_c::{
    HEAD_BASES, ULTRA_MODES_5_TO_8_BASE, compute_psychoacoustic_emr_weights,
    compute_role_b_multipliers,
};
use super::wide_gram::extract_rank4_svd_modes;

/// Telemetry and `CP1` slots emitted by [`solve_stage_d_cp1_and_audit`].
#[derive(Debug, Clone)]
pub struct StageDSolveOutput {
    /// Compiled `Cp1ContributorSlot` list for Region 4 (`build_dnam_container_with_cp1`).
    pub cp1_slots: Vec<Cp1ContributorSlot>,
    /// Number of active SVD head modes populated (`0..=8`).
    pub active_head_modes: usize,
    /// Full 3-Phase Control-Space Audit report.
    pub audit_report: AuditReport,
    /// Wall-clock Stage D execution time in milliseconds.
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
fn write_f32_le(buf: &mut [u8], off: usize, val: f32) {
    if off + 4 <= buf.len() {
        let clean = if val.is_finite() {
            if val == 0.0 { 0.0 } else { val }
        } else {
            0.0
        };
        buf[off..off + 4].copy_from_slice(&clean.to_le_bytes());
    }
}

pub(crate) fn decode_coreset_lr(fp16_bytes: &[u8], channels: usize) -> Vec<[f32; 2]> {
    let samples = fp16_bytes_to_f32(fp16_bytes);
    if channels == 2 {
        samples.as_chunks::<2>().0.to_vec()
    } else {
        samples.into_iter().map(|s| [s, s]).collect()
    }
}

/// Resolves the canonical `StarCenter` physical control vector (`default` for `Range`, `default_index` for `Switch`).
#[must_use]
pub fn star_center_control_values(manifest: &DnamProjManifest) -> Vec<f32> {
    manifest
        .base
        .controls_schema
        .iter()
        .map(|ctrl| match ctrl.kind {
            ControlKind::Range => {
                let [lo, hi] = ctrl.declared_range.unwrap_or([0.0, 1.0]);
                ctrl.default
                    .as_f64()
                    .map_or(0.5 * (lo + hi), |v| v as f32)
                    .clamp(lo, hi)
            }
            ControlKind::Switch => {
                let opts = ctrl.declared_options.as_deref().unwrap_or(&[]);
                let def_idx = ctrl
                    .default
                    .as_str()
                    .and_then(|s| opts.iter().position(|o| o == s))
                    .or_else(|| ctrl.default.as_u64().map(|u| u as usize))
                    .unwrap_or(0);
                def_idx.min(opts.len().saturating_sub(1)) as f32
            }
        })
        .collect()
}

fn resolve_stationary_range_value(
    ctrl: &ControlSchemaDef,
    ctrl_idx: usize,
    stem: &StemRegistryEntry,
    windows: &[StmcTrajectoryWindow],
    default_val: f32,
) -> f32 {
    if let Some(v) = stem
        .controls_meta
        .get(&ctrl.id)
        .and_then(serde_json::Value::as_f64)
    {
        return v as f32;
    }
    if let Some(w) = windows
        .iter()
        .find(|w| usize::from(w.control_id) == ctrl_idx)
    {
        return w.u_phys;
    }
    default_val
}

fn resolve_stationary_switch_option(
    ctrl: &ControlSchemaDef,
    ctrl_idx: usize,
    stem: &StemRegistryEntry,
    windows: &[StmcTrajectoryWindow],
    default_opt: u8,
) -> u8 {
    let opts = ctrl.declared_options.as_deref().unwrap_or(&[]);
    if let Some(val) = stem.controls_meta.get(&ctrl.id) {
        if let Some(s) = val.as_str()
            && let Some(pos) = opts.iter().position(|o| o == s)
        {
            return pos.min(15) as u8;
        }
        if let Some(u) = val.as_u64() {
            return (u as usize).min(15) as u8;
        }
    }
    if let Some(w) = windows
        .iter()
        .find(|w| usize::from(w.control_id) == ctrl_idx)
    {
        return w.active_opt.min(15);
    }
    default_opt
}

fn interpolate_sweep_windows(knots: &[f32], frac_01: f32) -> f32 {
    if knots.is_empty() {
        return 0.0;
    }
    if knots.len() == 1 {
        return knots[0];
    }
    let pos = frac_01.clamp(0.0, 1.0) * ((knots.len() - 1) as f32);
    let idx = (pos.floor() as usize).min(knots.len() - 2);
    let alpha = pos - (idx as f32);
    knots[idx] + alpha * (knots[idx + 1] - knots[idx])
}

#[derive(Clone, Copy)]
enum LaneSource {
    SlotLane { slot_idx: usize, lane_k: usize },
    FmFactor { factor_idx: usize },
}

pub use super::stages::write_inline_cp1_if_fits;
use super::stages::{decode_coreset_frame_lr_at, harvest_and_resolve_mode0_for_stage_d};

/// Executes Stage D (`< 5 ms` Arbitrary-`P` `CP1` Control Projections, Kronecker SVD Modes `1..8`,
/// Walsh-Hadamard `U_int` Factorization Machines & Full 3-Phase Audit, `r[train.pipeline.stage-d-control]`).
///
/// Guarantees:
/// - **Strict `0.000000 ULP` `StarCenter` Mode-0 Invariance**: `HEAD_BASES[t][0..512]` is never
///   modified by Stage D, and `z_head == z_head_ultra == [0.0; 4]` at `StarCenter` even under
///   asymmetric `anchor_nodes_norm[0] != 0` or non-zero `Switch` `default_index`.
/// - **Strict `0.000000 ULP` `Ultra -> Pro` Stripping Invariance**: `Arena 3 [60,928..61,056]`
///   (`U_int,Delta` Block 0) is kept strictly `0.0`.
#[allow(clippy::too_many_arguments)]
pub fn solve_stage_d_cp1_and_audit(
    manifest: &mut DnamProjManifest,
    arena_64k: &mut [u8],
    stmc_by_id: &BTreeMap<u32, Vec<u8>>,
    dry_lr: &[[f32; 2]],
    cached_packed_gram_256: Option<&[f64]>,
    gram_frame_stride: usize,
    clear_dynamic_w_ctrl: bool,
) -> Result<StageDSolveOutput, DnamError> {
    solve_stage_d_cp1_and_audit_with_harvest(
        manifest,
        arena_64k,
        stmc_by_id,
        dry_lr,
        cached_packed_gram_256,
        gram_frame_stride,
        clear_dynamic_w_ctrl,
        None,
        true,
    )
}

/// Variant of [`solve_stage_d_cp1_and_audit`] accepting pre-harvested `StageBHarvest`
/// `(features, backbone_out_lr)` on Path 2 and `clamp_backbone` (`false` when the `StarCenter`
/// physical backbone was already audited in Stage A or frozen in Path 1 / `preserve_backbone()`).
/// Borrowed `(features, backbone_out_lr)` harvest pair passed from Stage B to Stage D.
pub type BackboneHarvestSlices<'a> = (&'a [[f32; 256]], &'a [[f32; 2]]);

#[allow(clippy::too_many_arguments)]
pub fn solve_stage_d_cp1_and_audit_with_harvest(
    manifest: &mut DnamProjManifest,
    arena_64k: &mut [u8],
    stmc_by_id: &BTreeMap<u32, Vec<u8>>,
    dry_lr: &[[f32; 2]],
    cached_packed_gram_256: Option<&[f64]>,
    gram_frame_stride: usize,
    clear_dynamic_w_ctrl: bool,
    precomputed_backbone_harvest: Option<BackboneHarvestSlices<'_>>,
    clamp_backbone: bool,
) -> Result<StageDSolveOutput, DnamError> {
    let timer = now_ms_token();

    // Clear Modes 1..4, Ultra Modes 5..8, U_int^base, and U_int,Delta WITHOUT touching Mode 0!
    for &h_base in &HEAD_BASES {
        if arena_64k.len() >= h_base + 5 * 512 {
            arena_64k[h_base + 512..h_base + 5 * 512].fill(0);
        }
    }
    if arena_64k.len() >= ULTRA_MODES_5_TO_8_BASE + 8_192 {
        arena_64k[ULTRA_MODES_5_TO_8_BASE..ULTRA_MODES_5_TO_8_BASE + 8_192].fill(0);
    }
    if arena_64k.len() >= 7_680 {
        arena_64k[7_552..7_680].fill(0);
    }
    if arena_64k.len() >= 61_312 {
        arena_64k[60_928..61_312].fill(0);
    }
    if clear_dynamic_w_ctrl {
        if arena_64k.len() >= 10_560 {
            arena_64k[9_536..10_560].fill(0);
        }
        if arena_64k.len() >= 60_928 {
            arena_64k[60_672..60_928].fill(0);
        }
    }
    write_f32_le(arena_64k, 7_520, 10.0);
    write_f32_le(arena_64k, 10_560, 10.0);
    write_f32_le(arena_64k, 10_688, 0.0625);
    zero_arena_b_head_dc_bias(arena_64k);

    let descriptors: Vec<ControlClampDescriptor> =
        super::stages::sync_clamp_descriptors_to_arena(&manifest.base, arena_64k)?;
    let star_ctrls = star_center_control_values(manifest);

    // Build baseline main-effect Cp1ContributorSlots (Range 1..=3 lanes + Switch non-default options)
    let mut cp1_slots: Vec<Cp1ContributorSlot> = Vec::new();
    let mut lane_sources: Vec<LaneSource> = Vec::new();
    let mut range_slot_indices: Vec<usize> = Vec::new();

    for (p_idx, (ctrl, desc)) in manifest
        .base
        .controls_schema
        .iter()
        .zip(&descriptors)
        .enumerate()
    {
        match ctrl.kind {
            ControlKind::Range => {
                if desc.lobatto_degree == 0 {
                    continue;
                }
                let deg = desc.lobatto_degree.clamp(1, 3);
                let slot_idx = cp1_slots.len();
                cp1_slots.push(Cp1ContributorSlot {
                    lanes: deg,
                    ctrl_kind: 0,
                    control_idx: p_idx as u16,
                    option_idx: 0xFF,
                    partner_control_id: 0,
                    main_base: [[0.0; 8]; 3],
                    emb: [1.0, 0.0, 0.0], // Guarantees z_c == lob[0] == 0.0 at StarCenter!
                    v_base: [0.0; 4],
                    main_ultra: [[0.0; 8]; 3],
                    v_ultra: [0.0; 4],
                    has_ultra: true,
                });
                range_slot_indices.push(slot_idx);
                for lane_k in 0..usize::from(deg) {
                    lane_sources.push(LaneSource::SlotLane { slot_idx, lane_k });
                }
            }
            ControlKind::Switch => {
                let decl_opts = ctrl.declared_options.as_deref().unwrap_or(&[]);
                let def_opt = (desc.default_val.max(0.0) + 0.5) as u8;
                for opt_idx in 0..decl_opts.len().min(16) {
                    if (desc.active_options_mask & (1_u16 << opt_idx)) == 0 {
                        continue;
                    }
                    let opt_u8 = opt_idx as u8;
                    let slot_idx = cp1_slots.len();
                    cp1_slots.push(Cp1ContributorSlot {
                        lanes: 1,
                        ctrl_kind: 1,
                        control_idx: p_idx as u16,
                        option_idx: opt_u8,
                        partner_control_id: 0,
                        main_base: [[0.0; 8]; 3],
                        emb: [1.0, 0.0, 0.0],
                        v_base: [0.0; 4],
                        main_ultra: [[0.0; 8]; 3],
                        v_ultra: [0.0; 4],
                        has_ultra: true,
                    });
                    if opt_u8 != def_opt {
                        lane_sources.push(LaneSource::SlotLane {
                            slot_idx,
                            lane_k: 0,
                        });
                    }
                }
            }
        }
    }

    // Collect enabled off-center stems
    let off_center_stems: Vec<&StemRegistryEntry> = manifest
        .stems_registry
        .iter()
        .filter(|s| s.enabled && !s.is_star_center)
        .collect();

    if cp1_slots.is_empty() || off_center_stems.is_empty() || dry_lr.is_empty() {
        write_inline_cp1_if_fits(arena_64k, &cp1_slots);
        if clamp_backbone {
            run_full_3phase_arena_audit(manifest, arena_64k);
        } else {
            run_head_only_3phase_arena_audit(manifest, arena_64k);
        }
        let audit_report = manifest.audit_report.clone().unwrap_or_default();
        zero_arena_b_head_dc_bias(arena_64k);
        return Ok(StageDSolveOutput {
            cp1_slots,
            active_head_modes: 0,
            audit_report,
            elapsed_ms: elapsed_ms_since(timer),
        });
    }

    // Assign 3 Walsh-Hadamard FM factor vectors across active Range slots when >= 2 Range controls exist
    let use_fm_factors = range_slot_indices.len() >= 2;
    if use_fm_factors {
        let scale = (1.0 / (range_slot_indices.len() as f64).sqrt()) as f32;
        for (rank_pos, &slot_idx) in range_slot_indices.iter().enumerate() {
            let w0 = scale;
            let w1 = if rank_pos % 2 == 0 { scale } else { -scale };
            let w2 = if (rank_pos / 2) % 2 == 0 {
                scale
            } else {
                -scale
            };
            let vf = [w0, w1, w2, 0.0]; // Factor 3 strictly 0.0 so U_int row 3 stays 0.0!
            cp1_slots[slot_idx].v_base = vf;
            cp1_slots[slot_idx].v_ultra = vf;
        }
        for factor_idx in 0..3 {
            lane_sources.push(LaneSource::FmFactor { factor_idx });
        }
    }

    let star_emr_w: Option<Vec<f64>> = manifest
        .stems_registry
        .iter()
        .find(|s| s.enabled && s.is_star_center)
        .and_then(|s| stmc_by_id.get(&s.chunk_id))
        .and_then(|b| verify_proj_stmc_capsule(b).ok())
        .map(|st| {
            let ch = if (st.flags & STMC_FLAG_STEREO) != 0 {
                2
            } else {
                1
            };
            let sw = decode_coreset_lr(st.wet_coreset_fp16, ch);
            compute_psychoacoustic_emr_weights(dry_lr, &sw)
        });
    let resolved = harvest_and_resolve_mode0_for_stage_d(
        manifest,
        arena_64k,
        dry_lr,
        &star_ctrls,
        precomputed_backbone_harvest,
        cached_packed_gram_256,
        star_emr_w.as_deref(),
        gram_frame_stride,
    )?;
    let n_frames = resolved.n_total_frames;
    let gram_256 = resolved.gram_256;
    let sample_stride = resolved.sample_stride;
    let eval_stereo = manifest.base.io_config.output_mode != "mono";
    let sampled_frames = n_frames.div_ceil(sample_stride);
    let stride_scale = (n_frames as f64) / (sampled_frames.max(1) as f64);

    let l_total = lane_sources.len();
    let mut h_psi = vec![vec![0.0_f64; l_total]; l_total];
    let mut rhs_lanes = vec![vec![[0.0_f64; 2]; 256]; l_total];
    let mut total_weight_frames = 0.0_f64;

    for stem in &off_center_stems {
        let Some(stmc_bytes) = stmc_by_id.get(&stem.chunk_id) else {
            continue;
        };
        let stmc = verify_proj_stmc_capsule(stmc_bytes)?;
        let stmc_ch = if (stmc.flags & STMC_FLAG_STEREO) != 0 {
            2
        } else {
            1
        };
        let stmc_frames = stmc.wet_coreset_fp16.len() / (stmc_ch * 2);
        let m_len = stmc_frames.min(n_frames);
        if m_len == 0 {
            continue;
        }

        let is_freeze_probe = stem.user_metadata.contains_key("freeze_transition_window");
        let is_mode_a_sweep = !is_freeze_probe
            && (stmc.flags & STMC_FLAG_MODE_A_SWEEP) != 0
            && stmc.windows.len() >= 2;

        // Precompute per-control sweep knots (if sweeping) or stationary values
        let p_cnt = manifest.base.controls_schema.len();
        let mut sweep_knots: Vec<Vec<f32>> = vec![Vec::new(); p_cnt];
        let mut stat_range = vec![0.0_f32; p_cnt];
        let mut stat_switch = vec![0_u8; p_cnt];

        for (p, (ctrl, desc)) in manifest
            .base
            .controls_schema
            .iter()
            .zip(&descriptors)
            .enumerate()
        {
            match ctrl.kind {
                ControlKind::Range => {
                    if is_mode_a_sweep {
                        let knots: Vec<f32> = stmc
                            .windows
                            .iter()
                            .filter(|w| usize::from(w.control_id) == p)
                            .map(|w| w.u_phys)
                            .collect();
                        if knots.len() >= 2 && (knots.last().unwrap() - knots[0]).abs() > 1.0e-5 {
                            sweep_knots[p] = knots;
                        }
                    }
                    stat_range[p] = if is_freeze_probe {
                        star_ctrls[p]
                    } else {
                        resolve_stationary_range_value(ctrl, p, stem, &stmc.windows, star_ctrls[p])
                    };
                }
                ControlKind::Switch => {
                    let def_opt = (desc.default_val.max(0.0) + 0.5) as u8;
                    stat_switch[p] = if is_freeze_probe {
                        def_opt
                    } else {
                        resolve_stationary_switch_option(ctrl, p, stem, &stmc.windows, def_opt)
                    };
                }
            }
        }

        let eval_psi_at = |frac: f32, psi_out: &mut [f64], z_c_out: &mut [f64]| {
            psi_out.fill(0.0);
            z_c_out.fill(0.0);
            for (slot_idx, slot) in cp1_slots.iter().enumerate() {
                let p = usize::from(slot.control_idx);
                let desc = &descriptors[p];
                if slot.ctrl_kind == 0 {
                    let u_phys = if !sweep_knots[p].is_empty() {
                        interpolate_sweep_windows(&sweep_knots[p], frac)
                    } else {
                        stat_range[p]
                    };
                    let (_, u_norm, _) = desc.clamp_and_normalize(u_phys, &stat_switch);
                    let lob = desc.eval_lobatto_basis(u_norm);
                    z_c_out[slot_idx] = f64::from(lob[0]);
                    for (lane_pos, src) in lane_sources.iter().enumerate() {
                        if let LaneSource::SlotLane {
                            slot_idx: si,
                            lane_k,
                        } = *src
                            && si == slot_idx
                        {
                            psi_out[lane_pos] = f64::from(lob[lane_k]);
                        }
                    }
                } else if slot.ctrl_kind == 1 {
                    let s = if stat_switch[p] == slot.option_idx {
                        1.0_f64
                    } else {
                        0.0_f64
                    };
                    z_c_out[slot_idx] = s;
                    for (lane_pos, src) in lane_sources.iter().enumerate() {
                        if let LaneSource::SlotLane { slot_idx: si, .. } = *src
                            && si == slot_idx
                        {
                            psi_out[lane_pos] = s;
                        }
                    }
                }
            }
            if use_fm_factors {
                for (lane_pos, src) in lane_sources.iter().enumerate() {
                    if let LaneSource::FmFactor { factor_idx } = *src {
                        let mut sum_vz = 0.0_f64;
                        let mut sum_sq = 0.0_f64;
                        for &si in &range_slot_indices {
                            let vz = f64::from(cp1_slots[si].v_base[factor_idx]) * z_c_out[si];
                            sum_vz += vz;
                            sum_sq += vz * vz;
                        }
                        psi_out[lane_pos] = 0.5 * (sum_vz * sum_vz - sum_sq);
                    }
                }
            }
        };

        let mut psi_vec = vec![0.0_f64; l_total];
        let mut z_c_by_slot = vec![0.0_f64; cp1_slots.len()];

        if !is_mode_a_sweep {
            eval_psi_at(0.0, &mut psi_vec, &mut z_c_by_slot);
            let mut stem_rhs = [[0.0_f64; 2]; 256];
            let mut stem_w_sum = 0.0_f64;
            let mut step_idx = 0usize;
            while step_idx * sample_stride < m_len {
                let n = step_idx * sample_stride;
                let f_idx = if resolved.compact_strided {
                    step_idx
                } else {
                    n
                };
                let wet_n = decode_coreset_frame_lr_at(stmc.wet_coreset_fp16, stmc_ch, n);
                let pred_n = resolved.pred0_lr[f_idx];
                let yl = f64::from(wet_n[0]);
                let yr = if eval_stereo { f64::from(wet_n[1]) } else { yl };
                let dl = yl - f64::from(pred_n[0]);
                let dr = if eval_stereo {
                    yr - f64::from(pred_n[1])
                } else {
                    dl
                };
                let fb = &resolved.features[f_idx];
                let sw = star_emr_w
                    .as_ref()
                    .and_then(|w| w.get(n))
                    .copied()
                    .unwrap_or(1.0);
                let w_frame = stride_scale * sw;
                let w_dl = w_frame * dl;
                let w_dr = w_frame * dr;
                for d in 0..256 {
                    let fd = f64::from(fb[d]);
                    stem_rhs[d][0] += fd * w_dl;
                    stem_rhs[d][1] += fd * w_dr;
                }
                stem_w_sum += w_frame;
                step_idx += 1;
            }
            for i in 0..l_total {
                let pi = psi_vec[i];
                if pi.abs() <= 1.0e-12 {
                    continue;
                }
                for j in i..l_total {
                    h_psi[i][j] += stem_w_sum * pi * psi_vec[j];
                }
                let rhs_i = &mut rhs_lanes[i];
                for d in 0..256 {
                    rhs_i[d][0] += pi * stem_rhs[d][0];
                    rhs_i[d][1] += pi * stem_rhs[d][1];
                }
            }
            total_weight_frames += stem_w_sum;
        } else {
            let mut step_idx = 0usize;
            while step_idx * sample_stride < m_len {
                let n = step_idx * sample_stride;
                let f_idx = if resolved.compact_strided {
                    step_idx
                } else {
                    n
                };
                let frac = if m_len > 1 {
                    (n as f32) / ((m_len - 1) as f32)
                } else {
                    0.0
                };
                eval_psi_at(frac, &mut psi_vec, &mut z_c_by_slot);
                let wet_n = decode_coreset_frame_lr_at(stmc.wet_coreset_fp16, stmc_ch, n);
                let pred_n = resolved.pred0_lr[f_idx];
                let yl = f64::from(wet_n[0]);
                let yr = if eval_stereo { f64::from(wet_n[1]) } else { yl };
                let dl = yl - f64::from(pred_n[0]);
                let dr = if eval_stereo {
                    yr - f64::from(pred_n[1])
                } else {
                    dl
                };
                let fb = &resolved.features[f_idx];
                let sw = star_emr_w
                    .as_ref()
                    .and_then(|w| w.get(n))
                    .copied()
                    .unwrap_or(1.0);
                let w_frame = stride_scale * sw;
                for i in 0..l_total {
                    let pi = psi_vec[i];
                    if pi.abs() <= 1.0e-12 {
                        continue;
                    }
                    let w_pi = w_frame * pi;
                    for j in i..l_total {
                        h_psi[i][j] += w_pi * psi_vec[j];
                    }
                    let rhs_i = &mut rhs_lanes[i];
                    let pdl = w_pi * dl;
                    let pdr = w_pi * dr;
                    for d in 0..256 {
                        let fd = f64::from(fb[d]);
                        rhs_i[d][0] += fd * pdl;
                        rhs_i[d][1] += fd * pdr;
                    }
                }
                total_weight_frames += w_frame;
                step_idx += 1;
            }
        }
    }

    if total_weight_frames <= 0.0 {
        write_inline_cp1_if_fits(arena_64k, &cp1_slots);
        if clamp_backbone {
            run_full_3phase_arena_audit(manifest, arena_64k);
        } else {
            run_head_only_3phase_arena_audit(manifest, arena_64k);
        }
        let audit_report = manifest.audit_report.clone().unwrap_or_default();
        zero_arena_b_head_dc_bias(arena_64k);
        return Ok(StageDSolveOutput {
            cp1_slots,
            active_head_modes: 0,
            audit_report,
            elapsed_ms: elapsed_ms_since(timer),
        });
    }

    // Normalize H_psi by N_frames (since gram_256 integrates over N_frames!)
    let norm_n = n_frames as f64;
    let mut h_packed = vec![0.0_f64; uplo_u_packed_len(l_total)];
    let mut tr_h = 0.0_f64;
    for j in 0..l_total {
        for i in 0..=j {
            let val = h_psi[i][j] / norm_n;
            h_packed[uplo_u_index(i, j)] = val;
            if i == j {
                tr_h += val;
            }
        }
    }

    // Factorize G_0 once and solve G_0^{-1} * R_l for each active lane l in 0..l_total
    let tr_g: f64 = (0..256).map(|d| gram_256[uplo_u_index(d, d)]).sum();
    let mean_diag_g = (tr_g / 256.0).max(1.0e-9);
    let g_ridge = (5.0e-5 * mean_diag_g).max(1.0e-8);
    let mut g_inv_r = vec![[0.0_f64; 512]; l_total];
    if let Ok(r_g) = factorize_uplo_u_cholesky(256, &gram_256, g_ridge) {
        for l in 0..l_total {
            if rhs_lanes[l]
                .iter()
                .any(|c| c[0].abs() > 1.0e-12 || c[1].abs() > 1.0e-12)
                && let Ok(sol) = solve_factorized_uplo_u_2ch(256, &r_g, &rhs_lanes[l])
            {
                for d in 0..256 {
                    g_inv_r[l][d * 2] = sol[d][0];
                    g_inv_r[l][d * 2 + 1] = sol[d][1];
                }
            }
        }
    }

    // Factorize H_psi once and solve H_psi^{-1} * (G_0^{-1} R) across all 256 stereo pairs
    let h_ridge = (1.0e-6 * (tr_h / (l_total.max(1) as f64))).max(1.0e-7);
    let mut u_full = vec![vec![0.0_f64; 512]; l_total];
    let mut col_pair = vec![[0.0_f64; 2]; l_total];
    let mut feat_rms = [1.0_f64; 256];
    let rms_floor = (1.0e-4 * mean_diag_g).sqrt().max(1.0e-6);
    let r_h_opt = factorize_uplo_u_cholesky(l_total, &h_packed, h_ridge).ok();
    for d in 0..256 {
        feat_rms[d] = (gram_256[uplo_u_index(d, d)] / norm_n.max(1.0))
            .max(0.0)
            .sqrt()
            .max(rms_floor);
        for l in 0..l_total {
            col_pair[l] = [g_inv_r[l][d * 2], g_inv_r[l][d * 2 + 1]];
        }
        if let Some(ref r_h) = r_h_opt
            && let Ok(sol_l) = solve_factorized_uplo_u_2ch(l_total, r_h, &col_pair)
        {
            let s_d = feat_rms[d];
            for l in 0..l_total {
                u_full[l][d * 2] = sol_l[l][0] * s_d;
                u_full[l][d * 2 + 1] = sol_l[l][1] * s_d;
            }
        }
    }

    // Factor energy-weighted U_full into Modes 1..4 (base) + Modes 5..8 (Ultra residual)
    let (mut modes_base, mut coords_base) = extract_rank4_svd_modes(&u_full);
    let mut u_res = u_full;
    for l in 0..l_total {
        for (k, mode) in modes_base.iter().enumerate() {
            let c_lk = coords_base[l][k];
            for d in 0..512 {
                u_res[l][d] -= c_lk * mode[d];
            }
        }
    }
    let (mut modes_ultra, mut coords_ultra) = extract_rank4_svd_modes(&u_res);
    let active_head_modes = modes_base.len() + modes_ultra.len();

    // Unscale feature RMS and gauge-normalize each mode so stored arena weights stay well inside
    // audit.rs's +6 dB (rms_gain <= 2.0) [CLAMP-HEAD-GAIN] ceiling while preserving coords * mode!
    let role_b_mult = compute_role_b_multipliers(arena_64k);
    let unscale_and_cap = |modes: &mut [[f64; 512]], coords: &mut [[f64; 4]]| {
        for (k, mode) in modes.iter_mut().enumerate() {
            let mut max_tier_rms = 0.0_f64;
            for t in 0..4 {
                let mut sq_l = 0.0_f64;
                let mut sq_r = 0.0_f64;
                for i in 0..64 {
                    let idx = t * 64 + i;
                    let inv_s = 1.0 / feat_rms[idx];
                    let inv_ml = if role_b_mult[idx][0].abs() > 1.0e-6 {
                        1.0 / role_b_mult[idx][0]
                    } else {
                        1.0
                    };
                    let inv_mr = if role_b_mult[idx][1].abs() > 1.0e-6 {
                        1.0 / role_b_mult[idx][1]
                    } else {
                        1.0
                    };
                    mode[idx * 2] *= inv_s * inv_ml;
                    mode[idx * 2 + 1] *= inv_s * inv_mr;
                    sq_l += mode[idx * 2] * mode[idx * 2];
                    sq_r += mode[idx * 2 + 1] * mode[idx * 2 + 1];
                }
                max_tier_rms = max_tier_rms.max((sq_l.max(sq_r) / 16.0).sqrt());
            }
            if max_tier_rms > 1.25 {
                let shrink = 1.25 / max_tier_rms;
                let grow = 1.0 / shrink;
                for v in mode.iter_mut() {
                    *v *= shrink;
                }
                for row in coords.iter_mut() {
                    row[k] *= grow;
                }
            }
        }
    };
    unscale_and_cap(&mut modes_base, &mut coords_base);
    for (k, mode) in modes_base.iter().enumerate() {
        for (t, &h_base) in HEAD_BASES.iter().enumerate() {
            let mode_off = h_base + (k + 1) * 512;
            for i in 0..64 {
                let idx = t * 64 + i;
                write_f32_le(
                    arena_64k,
                    mode_off + i * 4,
                    (mode[idx * 2] as f32).clamp(-64.0, 64.0),
                );
                write_f32_le(
                    arena_64k,
                    mode_off + 256 + i * 4,
                    (mode[idx * 2 + 1] as f32).clamp(-64.0, 64.0),
                );
            }
        }
    }
    if arena_64k.len() >= ULTRA_MODES_5_TO_8_BASE + 8_192 {
        unscale_and_cap(&mut modes_ultra, &mut coords_ultra);
        for (k, mode) in modes_ultra.iter().enumerate() {
            for t in 0..4 {
                let mode_off = ULTRA_MODES_5_TO_8_BASE + t * 2_048 + k * 512;
                for i in 0..64 {
                    let idx = t * 64 + i;
                    write_f32_le(
                        arena_64k,
                        mode_off + i * 4,
                        (mode[idx * 2] as f32).clamp(-64.0, 64.0),
                    );
                    write_f32_le(
                        arena_64k,
                        mode_off + 256 + i * 4,
                        (mode[idx * 2 + 1] as f32).clamp(-64.0, 64.0),
                    );
                }
            }
        }
    }

    // Populate Cp1ContributorSlot projections and U_int^base (7,552..7,680) + U_int,Delta Block 2 (61,184..61,312)
    // while keeping U_int,Delta Block 0 (60,928..61,056) strictly 0.0 for 0.000000 ULP Ultra->Pro stripping!
    let mut has_fm_base = false;
    let mut has_fm_ultra = false;
    for (lane_pos, &src) in lane_sources.iter().enumerate() {
        match src {
            LaneSource::SlotLane { slot_idx, lane_k } => {
                for d in 0..4 {
                    cp1_slots[slot_idx].main_base[lane_k][d] =
                        (coords_base[lane_pos][d] as f32).clamp(-9.5, 9.5);
                    cp1_slots[slot_idx].main_ultra[lane_k][d] =
                        (coords_ultra[lane_pos][d] as f32).clamp(-9.5, 9.5);
                }
            }
            LaneSource::FmFactor { factor_idx } => {
                let b_row = 7_552 + factor_idx * 32;
                let u_row = 60_928 + 256 + factor_idx * 32; // Block 2: z_fm_ultra -> z_raw[8..12]
                for d in 0..4 {
                    let vb = (coords_base[lane_pos][d] as f32).clamp(-9.5, 9.5);
                    if vb.abs() > 1.0e-7 {
                        has_fm_base = true;
                        write_f32_le(arena_64k, b_row + d * 4, vb);
                    }
                    if arena_64k.len() >= 61_312 {
                        let vu = (coords_ultra[lane_pos][d] as f32).clamp(-9.5, 9.5);
                        if vu.abs() > 1.0e-7 {
                            has_fm_ultra = true;
                            write_f32_le(arena_64k, u_row + d * 4, vu);
                        }
                    }
                }
            }
        }
    }
    if !has_fm_base {
        for slot in &mut cp1_slots {
            slot.v_base = [0.0; 4];
        }
    }
    if !has_fm_ultra {
        for slot in &mut cp1_slots {
            slot.v_ultra = [0.0; 4];
        }
    }

    write_inline_cp1_if_fits(arena_64k, &cp1_slots);
    if clamp_backbone {
        run_full_3phase_arena_audit(manifest, arena_64k);
    } else {
        run_head_only_3phase_arena_audit(manifest, arena_64k);
    }
    let audit_report = manifest.audit_report.clone().unwrap_or_default();
    zero_arena_b_head_dc_bias(arena_64k);

    Ok(StageDSolveOutput {
        cp1_slots,
        active_head_modes,
        audit_report,
        elapsed_ms: elapsed_ms_since(timer),
    })
}
