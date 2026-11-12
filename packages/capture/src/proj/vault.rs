#![forbid(unsafe_code)]
//! Non-Destructive `.dnam-proj` (`DNPJ`) Capture Vault Editing (`proj::vault`):
//! `add_capture`, `remove_capture`, `replace_capture`, `set_capture_enabled`,
//! `solve_captures_batch_into_proj`, and Two-Path (`< 5–10 ms` Fast-Path vs. `< 15–25 ms` Full
//! Re-Pool) 5-Stage Compositional Re-Solve.
//!
//! Canonical References:
//! - `docs/specs/containers/project_container_format.md` (`ref[container.proj.lifecycle-states]`)
//! - `docs/specs/training/four_stage_pipeline.md` (`ref[train.pipeline.dispatch-solve]`)
//! - `docs/specs/training/iterative_resume_and_caching.md` (`ref[train.resume.kronecker-reassembly]`, `ref[train.cache.proj-vault]`)

use std::collections::{BTreeMap, BTreeSet};

use dnam_engine::container::ARENA_OFFSET_ULTRA_END;
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::schema::DnamError;

pub use crate::audit::{
    recompute_vault_phase1_audit_and_clamps, recompute_vault_phase1_audit_and_clamps_with_windows,
};
use crate::solver::{
    SolverOptions, compute_role_b_multipliers, finalize_stage_e_and_telemetry,
    harvest_star_center_256d, lift_stage_a_backbone, solve_stage_c_nested_schur_4tier,
    solve_stage_d_cp1_and_audit_with_harvest, star_center_control_values, uplo_u_packed_len,
};

use super::bundle::{
    DnamProjView, VaultChunkInput, build_dnam_proj_container_from_vault, zero_arena_b_head_dc_bias,
};
use super::directory::{
    ProjChunkKind, StmcTrajectoryWindow, build_proj_stmc_capsule, verify_proj_inpc_capsule,
    verify_proj_stmc_capsule,
};
use super::schema::{DnamProjManifest, ExtractedPhysics, StemProvenance, StemRegistryEntry};
use super::{
    PROJ_FLAG_COMPACTED_CAPSULES, STMC_FLAG_ENABLED, STMC_FLAG_MODE_A_SWEEP, STMC_FLAG_STAR_CENTER,
    STMC_FLAG_STEREO, fp16_bytes_to_f32,
};

const HEAD_BASES: [usize; 4] = [4_928, 18_048, 31_936, 57_600];
const STAGE3_EARLY_FIR_BASES: [usize; 4] = [4_288, 17_408, 31_552, 48_512];

/// Specification of a single captured stem inserted or replaced in the `.dnam-proj` Capture Vault.
#[derive(Debug, Clone, PartialEq)]
pub struct VaultCaptureSpec {
    /// Unique stem identifier within the project.
    pub stem_id: String,
    /// True if this stem is the `StarCenter` (`s = 0`) anchor backing `STAT`.
    pub is_star_center: bool,
    /// Initial enable state (`true` = active in solve & Phase-1 audit).
    pub enabled: bool,
    /// Input pad linear gain applied during capture (`1.0` default).
    pub pad_gain_f32: f32,
    /// Sub-sample alignment lag in samples at `48 kHz`.
    pub fractional_lag_samples_f32: f32,
    /// Active physical control coordinates (`control_id -> f64 | String`).
    pub controls_meta: BTreeMap<String, serde_json::Value>,
    /// Dry/wet PCM SHA-256 provenance and frame count.
    pub provenance: StemProvenance,
    /// Creator stem notes/tags (excluded from immutable JCS digest).
    pub user_metadata: BTreeMap<String, serde_json::Value>,
    /// Optional white-box / blind extracted physical descriptors.
    pub extracted_physics: Option<ExtractedPhysics>,
    /// `1..=36` trajectory windows (`R = 1` for stationary stops, `R > 1` for Mode A sweeps).
    pub windows: Vec<StmcTrajectoryWindow>,
    /// `88 B` `uplo='U'` band-energy PSD summary.
    pub band_psd_summary_88: [u8; 88],
    /// Disjoint-comb carrier count recorded in `STMC`.
    pub n_comb_bins: u32,
    /// `4,096 B` spectral, sideband, and ballistic summary block.
    pub spectral_summary_4096: [u8; 4_096],
    /// `24,000 B` (mono) or `48,000 B` (stereo) RNE `fp16` wet coreset (`12,000` frames).
    pub wet_coreset_fp16: Vec<u8>,
    /// Optional raw `SPCM` chunk (`b"SPCM"` header + aligned PCM samples).
    pub optional_spcm: Option<Vec<u8>>,
}

impl VaultCaptureSpec {
    fn to_stmc_bytes(&self, stem_index: u16) -> Result<Vec<u8>, DnamError> {
        let mut flags = 0u16;
        if self.enabled {
            flags |= STMC_FLAG_ENABLED;
        }
        if self.is_star_center {
            flags |= STMC_FLAG_STAR_CENTER;
        }
        if self.windows.len() > 1 {
            flags |= STMC_FLAG_MODE_A_SWEEP;
        }
        if self.wet_coreset_fp16.len() == 48_000 {
            flags |= STMC_FLAG_STEREO;
        }
        build_proj_stmc_capsule(
            stem_index,
            flags,
            self.pad_gain_f32,
            self.fractional_lag_samples_f32,
            &self.band_psd_summary_88,
            self.n_comb_bins,
            &self.windows,
            &self.spectral_summary_4096,
            &self.wet_coreset_fp16,
        )
    }
}

pub(crate) fn initialize_backbone_arena_if_needed(existing_arena: Option<&[u8]>) -> Vec<u8> {
    if let Some(bytes) = existing_arena
        && bytes.len() == ARENA_OFFSET_ULTRA_END as usize
        && bytes.iter().any(|&b| b != 0)
    {
        return bytes.to_vec();
    }
    let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    if let Some(bytes) = existing_arena
        && bytes.iter().any(|&b| b != 0)
    {
        let copy_len = bytes.len().min(arena.len());
        arena[..copy_len].copy_from_slice(&bytes[..copy_len]);
    } else {
        for (t, &fir_base) in STAGE3_EARLY_FIR_BASES.iter().enumerate() {
            let l_off = fir_base + (t * 4);
            let r_off = fir_base + 64 + (t * 4);
            arena[l_off..l_off + 4].copy_from_slice(&1.0_f32.to_le_bytes());
            arena[r_off..r_off + 4].copy_from_slice(&1.0_f32.to_le_bytes());
        }
    }
    zero_arena_b_head_dc_bias(&mut arena);
    arena
}

fn decode_coreset_frames_lr(fp16_bytes: &[u8], channels: usize) -> Vec<[f32; 2]> {
    let samples = fp16_bytes_to_f32(fp16_bytes);
    if channels == 2 {
        samples.as_chunks::<2>().0.to_vec()
    } else {
        samples.into_iter().map(|s| [s, s]).collect()
    }
}

type ChunkDigest = (u32, [u8; 32]);

struct VaultState {
    manifest: DnamProjManifest,
    calb: Option<(Vec<u8>, Option<ChunkDigest>)>,
    inpc: Option<(Vec<u8>, Option<ChunkDigest>)>,
    spcm_by_id: BTreeMap<u32, (Vec<u8>, Option<ChunkDigest>)>,
    stmc_by_id: BTreeMap<u32, Vec<u8>>,
    stmc_digests: BTreeMap<u32, ChunkDigest>,
    stat: Option<(Vec<u8>, Option<ChunkDigest>)>,
    wght_arena: Option<Vec<u8>>,
    compacted: bool,
}

fn parse_validated_stmc_windows(stmc_buf: &[u8]) -> Result<Vec<StmcTrajectoryWindow>, DnamError> {
    if stmc_buf.len() < 704 {
        return Ok(verify_proj_stmc_capsule(stmc_buf)?.windows);
    }
    let n_windows = usize::from(u16::from_le_bytes([stmc_buf[10], stmc_buf[11]]));
    let mut windows = Vec::with_capacity(n_windows);
    for idx in 0..n_windows {
        let off = 128 + idx * 16;
        windows.push(StmcTrajectoryWindow::from_bytes(&stmc_buf[off..off + 16])?);
    }
    Ok(windows)
}

fn find_stem_pos(manifest: &DnamProjManifest, stem_id: &str) -> Result<usize, DnamError> {
    manifest
        .stems_registry
        .iter()
        .position(|s| s.stem_id == stem_id)
        .ok_or_else(|| {
            DnamError::SchemaValidationError(format!(
                "stem_id '{stem_id}' not found in stems_registry"
            ))
        })
}

fn promote_first_enabled_star(stems: &mut [StemRegistryEntry]) {
    for entry in stems.iter_mut() {
        entry.is_star_center = false;
    }
    if let Some(first_enabled) = stems.iter_mut().find(|s| s.enabled) {
        first_enabled.is_star_center = true;
    }
}

fn rebuild_vault_container(
    mut state: VaultState,
    inpc_entry: (Vec<u8>, Option<ChunkDigest>),
    force_full_repool: bool,
    touched_controls: Option<BTreeSet<u16>>,
    options: SolverOptions,
) -> Result<Vec<u8>, DnamError> {
    let (inpc_bytes, inpc_digest) = inpc_entry;
    let inpc_channels = if inpc_digest.is_some() && inpc_bytes.len() >= 640 {
        usize::from(u16::from_le_bytes([inpc_bytes[6], inpc_bytes[7]]))
    } else {
        usize::from(verify_proj_inpc_capsule(&inpc_bytes)?.channels)
    };

    let mut enabled_windows = BTreeMap::new();
    for stem in &state.manifest.stems_registry {
        if let Some(stmc_buf) = state.stmc_by_id.get_mut(&stem.chunk_id) {
            if stmc_buf.len() < 4_800 {
                let _ = verify_proj_stmc_capsule(stmc_buf)?;
            }
            let cur_flags = u16::from_le_bytes([stmc_buf[8], stmc_buf[9]]);
            let mut new_flags = cur_flags;
            if stem.enabled {
                new_flags |= STMC_FLAG_ENABLED;
            } else {
                new_flags &= !STMC_FLAG_ENABLED;
            }
            if stem.is_star_center {
                new_flags |= STMC_FLAG_STAR_CENTER;
            } else {
                new_flags &= !STMC_FLAG_STAR_CENTER;
            }
            if new_flags != cur_flags {
                stmc_buf[8..10].copy_from_slice(&new_flags.to_le_bytes());
                state.stmc_digests.remove(&stem.chunk_id);
            }
            if stem.enabled {
                enabled_windows.insert(stem.chunk_id, parse_validated_stmc_windows(stmc_buf)?);
            }
        }
    }
    recompute_vault_phase1_audit_and_clamps_with_windows(&mut state.manifest, &enabled_windows);

    let has_prior_wght = state.wght_arena.is_some();
    let mut arena_64k = initialize_backbone_arena_if_needed(state.wght_arena.as_deref());
    let _ = crate::solver::stages::sync_clamp_descriptors_to_arena(
        &state.manifest.base,
        &mut arena_64k,
    )?;

    let active_star = state
        .manifest
        .stems_registry
        .iter()
        .find(|s| s.enabled && s.is_star_center)
        .cloned();

    let need_path2 =
        force_full_repool || !has_prior_wght || (active_star.is_some() != state.stat.is_some());

    let dry_lr = decode_coreset_frames_lr(&inpc_bytes[640..], inpc_channels);
    let mut stat_out = state.stat;
    let mut cached_gram_256: Option<Vec<f64>> = None;
    let mut cached_backbone_harvest = None;
    let mut star_wet_opt: Option<Vec<[f32; 2]>> = None;
    let mut fallback_esr: Option<[f32; 4]> = None;
    let mut accum_solve_ms = 0.0_f32;
    let _ = touched_controls;

    if need_path2 {
        crate::solver::stage_b_c::strip_all_head_modes_on_arena(&mut arena_64k);
        if options.clear_dynamic_w_ctrl {
            arena_64k[9_536..10_560].fill(0);
            arena_64k[60_672..60_928].fill(0);
        }
        if let Some(ref star_entry) = active_star
            && let Some(star_stmc_bytes) = state.stmc_by_id.get(&star_entry.chunk_id)
        {
            let star_flags = u16::from_le_bytes([star_stmc_bytes[8], star_stmc_bytes[9]]);
            let wet_lr = decode_coreset_frames_lr(
                &star_stmc_bytes[4_800..],
                if (star_flags & STMC_FLAG_STEREO) != 0 {
                    2
                } else {
                    1
                },
            );
            let spec_arr: Option<&[u8; 4_096]> = star_stmc_bytes[704..4_800].try_into().ok();

            // Stage A: Anchor-Stem Lifting (< 5 ms; Phase-2 rail lockup clamped inside Stage A)
            let stage_a_rep = lift_stage_a_backbone(
                &state.manifest,
                &mut arena_64k,
                &dry_lr,
                &wet_lr,
                spec_arr,
                options.stage_a,
            );
            accum_solve_ms += stage_a_rep.elapsed_ms;
            if let Some(warn) = stage_a_rep.phase2_rail_warning {
                let ar = state
                    .manifest
                    .audit_report
                    .get_or_insert_with(Default::default);
                if !ar.phase2_rail_warnings.contains(&warn) {
                    ar.phase2_rail_warnings.push(warn);
                }
            }

            // Stage B: Full-Rank 256-D Feature Harvest & Wide192Accum Contraction (< 10 ms)
            let role_b = compute_role_b_multipliers(&arena_64k);
            let star_ctrls = star_center_control_values(&state.manifest);
            let harvest = harvest_star_center_256d(
                &state.manifest,
                &arena_64k,
                &dry_lr,
                &wet_lr,
                &star_ctrls,
                options.gram_frame_stride,
            )?;
            accum_solve_ms += harvest.elapsed_ms;

            // Stage C: Multi-Tier Non-Homogeneous 64-D Schur-Complement Re-Balancing (< 5 ms)
            let stage_c = solve_stage_c_nested_schur_4tier(
                &mut arena_64k,
                Some(&harvest),
                &harvest.packed_gram_256,
                &harvest.cross_256,
                &role_b,
            )?;
            accum_solve_ms += stage_c.elapsed_ms;
            stat_out = Some((stage_c.stat_bytes, None));
            fallback_esr = Some(stage_c.prefix_esr_db);
            cached_gram_256 = Some(harvest.packed_gram_256);
            cached_backbone_harvest = Some((harvest.features, harvest.backbone_out_lr));
            star_wet_opt = Some(wet_lr);
        } else {
            stat_out = None;
            for &h_base in &HEAD_BASES {
                arena_64k[h_base..h_base + 512].fill(0);
            }
            zero_arena_b_head_dc_bias(&mut arena_64k);
        }
    } else if let Some((ref stat_bytes, _)) = stat_out {
        // Path 1 (< 5–10 ms): reuse cached G_0 from STAT; Mode 0 stays 0.000000 ULP untouched.
        let n_pack = uplo_u_packed_len(256);
        if stat_bytes.len() >= n_pack * 8 {
            let mut g = Vec::with_capacity(n_pack);
            for chunk in stat_bytes[..n_pack * 8].as_chunks::<8>().0 {
                g.push(f64::from_le_bytes(*chunk));
            }
            cached_gram_256 = Some(g);
        }
    }

    // Stage D: Arbitrary-P CP1 Projections, Kronecker SVD Modes 1..8, Hadamard FM & 3-Phase Audit
    let pre_harvest_ref = cached_backbone_harvest
        .as_ref()
        .map(|(f, b)| (f.as_slice(), b.as_slice()));
    let stage_d = solve_stage_d_cp1_and_audit_with_harvest(
        &mut state.manifest,
        &mut arena_64k,
        &state.stmc_by_id,
        &dry_lr,
        cached_gram_256.as_deref(),
        options.gram_frame_stride,
        need_path2 && options.clear_dynamic_w_ctrl,
        pre_harvest_ref,
        need_path2 && active_star.is_none() && !options.stage_a.preserve_existing_backbone,
    )?;
    accum_solve_ms += stage_d.elapsed_ms;

    // Stage E: 4-Tier Telemetry, Loudness Trim (on Path 2), and [STAB-ZERO-DC] Packing
    let _stage_e = finalize_stage_e_and_telemetry(
        &mut state.manifest,
        &mut arena_64k,
        &stage_d.cp1_slots,
        &dry_lr,
        star_wet_opt.as_deref(),
        fallback_esr,
        accum_solve_ms,
    )?;

    let mut chunks = Vec::new();
    if let Some((calb, calb_dig)) = state.calb {
        chunks.push(VaultChunkInput {
            kind: ProjChunkKind::Calb,
            chunk_id: 0,
            payload: calb,
            cached_digest: calb_dig,
        });
    }
    for (&cid, (spcm, spcm_dig)) in &state.spcm_by_id {
        chunks.push(VaultChunkInput {
            kind: ProjChunkKind::Spcm,
            chunk_id: cid,
            payload: spcm.clone(),
            cached_digest: *spcm_dig,
        });
    }
    if !state.stmc_by_id.is_empty() {
        chunks.push(VaultChunkInput {
            kind: ProjChunkKind::Inpc,
            chunk_id: 0,
            payload: inpc_bytes,
            cached_digest: inpc_digest,
        });
        for stem in &state.manifest.stems_registry {
            if let Some(stmc) = state.stmc_by_id.get(&stem.chunk_id) {
                chunks.push(VaultChunkInput {
                    kind: ProjChunkKind::Stmc,
                    chunk_id: stem.chunk_id,
                    payload: stmc.clone(),
                    cached_digest: state.stmc_digests.get(&stem.chunk_id).copied(),
                });
            }
        }
    }
    if let Some((stat, stat_dig)) = stat_out {
        chunks.push(VaultChunkInput {
            kind: ProjChunkKind::Stat,
            chunk_id: 0,
            payload: stat,
            cached_digest: stat_dig,
        });
    }

    let keep_compacted =
        state.compacted && state.spcm_by_id.is_empty() && !state.stmc_by_id.is_empty();
    build_dnam_proj_container_from_vault(
        &mut state.manifest,
        chunks,
        &arena_64k,
        &stage_d.cp1_slots,
        keep_compacted,
    )
}

fn extract_vault_state(proj_bytes: &[u8]) -> Result<VaultState, DnamError> {
    let crypto = DefaultCryptoProvider::default();
    let (view, wght_pair_opt) = DnamProjView::from_bytes_with_wght_view(proj_bytes, &crypto, true)?;
    let compacted = (view.preamble.lifecycle_flags & PROJ_FLAG_COMPACTED_CAPSULES) != 0;
    let mut calb = None;
    let mut inpc = None;
    let mut spcm_by_id = BTreeMap::new();
    let mut stmc_by_id = BTreeMap::new();
    let mut stmc_digests = BTreeMap::new();
    let mut stat = None;
    let wght_arena = wght_pair_opt.map(|(_, rv)| rv.arena_storage.as_bytes().to_vec());

    for (entry, &slice) in view.directory.iter().zip(&view.chunk_slices) {
        let dig = Some((entry.crc32c, entry.sha256));
        match entry.kind()? {
            ProjChunkKind::Calb => calb = Some((slice.to_vec(), dig)),
            ProjChunkKind::Inpc => inpc = Some((slice.to_vec(), dig)),
            ProjChunkKind::Spcm => {
                spcm_by_id.insert(entry.chunk_id, (slice.to_vec(), dig));
            }
            ProjChunkKind::Stmc => {
                stmc_by_id.insert(entry.chunk_id, slice.to_vec());
                stmc_digests.insert(entry.chunk_id, (entry.crc32c, entry.sha256));
            }
            ProjChunkKind::Stat => stat = Some((slice.to_vec(), dig)),
            ProjChunkKind::Wght | ProjChunkKind::Auth => {}
        }
    }

    Ok(VaultState {
        manifest: view.manifest,
        calb,
        inpc,
        spcm_by_id,
        stmc_by_id,
        stmc_digests,
        stat,
        wght_arena,
        compacted,
    })
}

fn collect_touched_control_indices(
    manifest: &DnamProjManifest,
    controls_meta: &BTreeMap<String, serde_json::Value>,
    windows: &[StmcTrajectoryWindow],
) -> BTreeSet<u16> {
    let mut out = BTreeSet::new();
    for w in windows {
        out.insert(w.control_id);
    }
    for (idx, ctrl) in manifest.base.controls_schema.iter().enumerate() {
        if controls_meta.contains_key(&ctrl.id) {
            out.insert(idx as u16);
        }
    }
    out
}

fn push_capture_into_vault_state(
    state: &mut VaultState,
    capture: VaultCaptureSpec,
) -> Result<(bool, BTreeSet<u16>), DnamError> {
    if state
        .manifest
        .stems_registry
        .iter()
        .any(|s| s.stem_id == capture.stem_id)
    {
        return Err(DnamError::SchemaValidationError(format!(
            "stem_id '{}' already exists in stems_registry",
            capture.stem_id
        )));
    }
    let next_chunk_id = state
        .manifest
        .stems_registry
        .iter()
        .map(|s| s.chunk_id)
        .max()
        .map_or(Ok(0_u32), |m| {
            m.checked_add(1).ok_or_else(|| {
                DnamError::SchemaValidationError("stems_registry chunk_id overflow".to_string())
            })
        })?;
    let stem_idx = u16::try_from(next_chunk_id).map_err(|_| {
        DnamError::SchemaValidationError("chunk_id exceeds u16 STMC stem_index".to_string())
    })?;
    let is_new_star = capture.is_star_center && capture.enabled;
    if is_new_star {
        for prev in &mut state.manifest.stems_registry {
            prev.is_star_center = false;
        }
    }
    let stmc_bytes = capture.to_stmc_bytes(stem_idx)?;
    let touched =
        collect_touched_control_indices(&state.manifest, &capture.controls_meta, &capture.windows);

    if let Some(spcm) = capture.optional_spcm {
        if spcm.len() < 64 || spcm[0..4] != *b"SPCM" {
            return Err(DnamError::SchemaValidationError(
                "invalid SPCM header".to_string(),
            ));
        }
        state.spcm_by_id.insert(next_chunk_id, (spcm, None));
    }
    state.stmc_digests.remove(&next_chunk_id);
    state.stmc_by_id.insert(next_chunk_id, stmc_bytes);
    state.manifest.stems_registry.push(StemRegistryEntry {
        stem_id: capture.stem_id,
        chunk_id: next_chunk_id,
        enabled: capture.enabled,
        is_star_center: capture.is_star_center,
        controls_meta: capture.controls_meta,
        provenance: capture.provenance,
        user_metadata: capture.user_metadata,
        extracted_physics: capture.extracted_physics,
    });
    Ok((is_new_star, touched))
}

/// Inserts a new capture into the `.dnam-proj` Capture Vault (`add_capture`), executing Path 1
/// (`< 5–10 ms`, `0.000000 ULP` on `StarCenter` Mode 0) when `StarCenter` is unchanged or Path 2
/// (`< 15–25 ms` full `STAT` + Mode 0 re-pool) when adding a new `StarCenter` (`r[train.cache.proj-vault]`).
pub fn add_capture(
    proj_bytes: &[u8],
    capture: VaultCaptureSpec,
    inpc_if_missing: Option<&[u8]>,
) -> Result<Vec<u8>, DnamError> {
    add_capture_with_options(
        proj_bytes,
        capture,
        inpc_if_missing,
        SolverOptions::default(),
    )
}

/// Variant of [`add_capture`] accepting explicit [`SolverOptions`] (e.g. `SolverOptions::preserve_backbone()`).
pub fn add_capture_with_options(
    proj_bytes: &[u8],
    capture: VaultCaptureSpec,
    inpc_if_missing: Option<&[u8]>,
    options: SolverOptions,
) -> Result<Vec<u8>, DnamError> {
    let mut state = extract_vault_state(proj_bytes)?;
    let inpc_entry = match (state.inpc.take(), inpc_if_missing) {
        (Some(existing), _) => existing,
        (None, Some(provided)) => (provided.to_vec(), None),
        (None, None) => {
            return Err(DnamError::SchemaValidationError(
                "add_capture requires an INPC dry stimulus capsule".to_string(),
            ));
        }
    };
    let (is_new_star, touched) = push_capture_into_vault_state(&mut state, capture)?;
    rebuild_vault_container(state, inpc_entry, is_new_star, Some(touched), options)
}

/// Inserts an entire batch of captured stems into `.dnam-proj` and executes the 5-Stage
/// `< 25 ms` Closed-Form + `VarPro` Solver in a single pass (`r[train.pipeline.dispatch-solve]`).
pub fn solve_captures_batch_into_proj(
    proj_bytes: &[u8],
    captures: Vec<VaultCaptureSpec>,
    inpc_if_missing: Option<&[u8]>,
) -> Result<Vec<u8>, DnamError> {
    solve_captures_batch_with_options(
        proj_bytes,
        captures,
        inpc_if_missing,
        SolverOptions::default(),
    )
}

/// Variant of [`solve_captures_batch_into_proj`] accepting explicit [`SolverOptions`].
pub fn solve_captures_batch_with_options(
    proj_bytes: &[u8],
    captures: Vec<VaultCaptureSpec>,
    inpc_if_missing: Option<&[u8]>,
    options: SolverOptions,
) -> Result<Vec<u8>, DnamError> {
    if captures.is_empty() {
        return Ok(proj_bytes.to_vec());
    }
    let mut state = extract_vault_state(proj_bytes)?;
    let inpc_entry = match (state.inpc.take(), inpc_if_missing) {
        (Some(existing), _) => existing,
        (None, Some(provided)) => (provided.to_vec(), None),
        (None, None) => {
            return Err(DnamError::SchemaValidationError(
                "solve_captures_batch requires an INPC dry stimulus capsule".to_string(),
            ));
        }
    };
    let mut any_new_star = false;
    let mut all_touched = BTreeSet::new();
    for cap in captures {
        let (is_star, touched) = push_capture_into_vault_state(&mut state, cap)?;
        any_new_star |= is_star;
        all_touched.extend(touched);
    }
    if !state
        .manifest
        .stems_registry
        .iter()
        .any(|s| s.enabled && s.is_star_center)
    {
        promote_first_enabled_star(&mut state.manifest.stems_registry);
        any_new_star = true;
    }
    rebuild_vault_container(state, inpc_entry, any_new_star, Some(all_touched), options)
}

/// Removes `stem_id` from the `.dnam-proj` Capture Vault (`remove_capture`), automatically
/// tightening Phase-1 `[CLAMP-RANGE]` / `[CLAMP-SWITCH]` bounds (`r[train.cache.proj-vault]`).
pub fn remove_capture(proj_bytes: &[u8], stem_id: &str) -> Result<Vec<u8>, DnamError> {
    remove_capture_with_options(proj_bytes, stem_id, SolverOptions::default())
}

/// Variant of [`remove_capture`] accepting explicit [`SolverOptions`].
pub fn remove_capture_with_options(
    proj_bytes: &[u8],
    stem_id: &str,
    options: SolverOptions,
) -> Result<Vec<u8>, DnamError> {
    let mut state = extract_vault_state(proj_bytes)?;
    let pos = find_stem_pos(&state.manifest, stem_id)?;
    let removed = state.manifest.stems_registry.remove(pos);
    state.spcm_by_id.remove(&removed.chunk_id);
    state.stmc_digests.remove(&removed.chunk_id);
    let windows = state
        .stmc_by_id
        .remove(&removed.chunk_id)
        .as_deref()
        .and_then(|b| parse_validated_stmc_windows(b).ok())
        .unwrap_or_default();
    let touched =
        collect_touched_control_indices(&state.manifest, &removed.controls_meta, &windows);

    let was_active_star = removed.is_star_center && removed.enabled;
    if was_active_star {
        state.stat = None;
        promote_first_enabled_star(&mut state.manifest.stems_registry);
    }

    if state.manifest.stems_registry.is_empty() {
        recompute_vault_phase1_audit_and_clamps(&mut state.manifest);
        let mut chunks = Vec::new();
        if let Some((c, c_dig)) = state.calb {
            chunks.push(VaultChunkInput {
                kind: ProjChunkKind::Calb,
                chunk_id: 0,
                payload: c,
                cached_digest: c_dig,
            });
        }
        let arena_64k = initialize_backbone_arena_if_needed(None);
        return build_dnam_proj_container_from_vault(
            &mut state.manifest,
            chunks,
            &arena_64k,
            &[],
            false,
        );
    }

    let inpc_entry = state.inpc.take().ok_or_else(|| {
        DnamError::SchemaValidationError("missing INPC chunk in non-empty vault".to_string())
    })?;
    rebuild_vault_container(state, inpc_entry, was_active_star, Some(touched), options)
}

fn clear_other_stars(stems: &mut [StemRegistryEntry], keep_pos: usize) {
    for (i, entry) in stems.iter_mut().enumerate() {
        entry.is_star_center = i == keep_pos;
    }
}

/// Replaces `stem_id` in-place inside the `.dnam-proj` Capture Vault (`replace_capture`), running
/// Path 1 when `StarCenter` is unchanged or Path 2 when `StarCenter` is replaced (`r[train.cache.proj-vault]`).
pub fn replace_capture(
    proj_bytes: &[u8],
    stem_id: &str,
    replacement: VaultCaptureSpec,
) -> Result<Vec<u8>, DnamError> {
    replace_capture_with_options(proj_bytes, stem_id, replacement, SolverOptions::default())
}

/// Variant of [`replace_capture`] accepting explicit [`SolverOptions`].
pub fn replace_capture_with_options(
    proj_bytes: &[u8],
    stem_id: &str,
    replacement: VaultCaptureSpec,
    options: SolverOptions,
) -> Result<Vec<u8>, DnamError> {
    let mut state = extract_vault_state(proj_bytes)?;
    let pos = find_stem_pos(&state.manifest, stem_id)?;
    if state
        .manifest
        .stems_registry
        .iter()
        .enumerate()
        .any(|(i, s)| i != pos && s.stem_id == replacement.stem_id)
    {
        return Err(DnamError::SchemaValidationError(format!(
            "stem_id '{}' already exists in stems_registry",
            replacement.stem_id
        )));
    }
    let inpc_bytes = state.inpc.take().ok_or_else(|| {
        DnamError::SchemaValidationError("missing INPC chunk in replace_capture".to_string())
    })?;
    let chunk_id = state.manifest.stems_registry[pos].chunk_id;
    let stem_idx = u16::try_from(chunk_id).map_err(|_| {
        DnamError::SchemaValidationError("chunk_id exceeds u16 STMC stem_index".to_string())
    })?;
    let old_was_star = state.manifest.stems_registry[pos].is_star_center
        && state.manifest.stems_registry[pos].enabled;
    let new_is_star = replacement.is_star_center && replacement.enabled;
    let old_windows = state
        .stmc_by_id
        .get(&chunk_id)
        .and_then(|b| verify_proj_stmc_capsule(b).ok().map(|v| v.windows))
        .unwrap_or_default();

    let mut touched = collect_touched_control_indices(
        &state.manifest,
        &state.manifest.stems_registry[pos].controls_meta,
        &old_windows,
    );
    touched.extend(collect_touched_control_indices(
        &state.manifest,
        &replacement.controls_meta,
        &replacement.windows,
    ));

    state.stmc_digests.remove(&chunk_id);
    state
        .stmc_by_id
        .insert(chunk_id, replacement.to_stmc_bytes(stem_idx)?);
    match replacement.optional_spcm {
        Some(spcm) => {
            if spcm.len() < 64 || spcm[0..4] != *b"SPCM" {
                return Err(DnamError::SchemaValidationError(
                    "invalid SPCM header".to_string(),
                ));
            }
            state.spcm_by_id.insert(chunk_id, (spcm, None));
        }
        None => {
            state.spcm_by_id.remove(&chunk_id);
        }
    }
    state.manifest.stems_registry[pos] = StemRegistryEntry {
        stem_id: replacement.stem_id,
        chunk_id,
        enabled: replacement.enabled,
        is_star_center: replacement.is_star_center,
        controls_meta: replacement.controls_meta,
        provenance: replacement.provenance,
        user_metadata: replacement.user_metadata,
        extracted_physics: replacement.extracted_physics,
    };
    if new_is_star {
        clear_other_stars(&mut state.manifest.stems_registry, pos);
    } else if old_was_star {
        state.stat = None;
        promote_first_enabled_star(&mut state.manifest.stems_registry);
    }

    rebuild_vault_container(
        state,
        inpc_bytes,
        old_was_star || new_is_star,
        Some(touched),
        options,
    )
}

/// Non-destructively mutes (`enabled = false`) or unmutes (`enabled = true`) a capture in the
/// `.dnam-proj` Capture Vault (`set_capture_enabled`), keeping its `STMC` capsule in Region 3 while
/// updating `STMC_FLAG_ENABLED`, Phase-1 clamps, and `WGHT` (`r[train.cache.proj-vault]`).
pub fn set_capture_enabled(
    proj_bytes: &[u8],
    stem_id: &str,
    enabled: bool,
) -> Result<Vec<u8>, DnamError> {
    set_capture_enabled_with_options(proj_bytes, stem_id, enabled, SolverOptions::default())
}

/// Variant of [`set_capture_enabled`] accepting explicit [`SolverOptions`].
pub fn set_capture_enabled_with_options(
    proj_bytes: &[u8],
    stem_id: &str,
    enabled: bool,
    options: SolverOptions,
) -> Result<Vec<u8>, DnamError> {
    let mut state = extract_vault_state(proj_bytes)?;
    let pos = find_stem_pos(&state.manifest, stem_id)?;
    let inpc_bytes = state.inpc.take().ok_or_else(|| {
        DnamError::SchemaValidationError("missing INPC chunk in set_capture_enabled".to_string())
    })?;
    let chunk_id = state.manifest.stems_registry[pos].chunk_id;
    let stmc_buf = state.stmc_by_id.get(&chunk_id).ok_or_else(|| {
        DnamError::SchemaValidationError(format!(
            "missing STMC chunk {chunk_id} for stem '{stem_id}'"
        ))
    })?;
    let stmc_view = verify_proj_stmc_capsule(stmc_buf)?;
    let touched = collect_touched_control_indices(
        &state.manifest,
        &state.manifest.stems_registry[pos].controls_meta,
        &stmc_view.windows,
    );

    let was_active_star = state.manifest.stems_registry[pos].is_star_center
        && state.manifest.stems_registry[pos].enabled;
    state.manifest.stems_registry[pos].enabled = enabled;
    let mut force_full_repool = false;
    if was_active_star && !enabled {
        state.stat = None;
        force_full_repool = true;
        promote_first_enabled_star(&mut state.manifest.stems_registry);
    } else if enabled {
        let other_active_star = state
            .manifest
            .stems_registry
            .iter()
            .enumerate()
            .any(|(i, s)| i != pos && s.enabled && s.is_star_center);
        if state.manifest.stems_registry[pos].is_star_center || !other_active_star {
            clear_other_stars(&mut state.manifest.stems_registry, pos);
            state.stat = None;
            force_full_repool = true;
        }
    }

    rebuild_vault_container(state, inpc_bytes, force_full_repool, Some(touched), options)
}
