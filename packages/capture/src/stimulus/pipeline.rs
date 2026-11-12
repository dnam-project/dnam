#![forbid(unsafe_code)]
//! Mode 1 Closed-Loop Streaming Distillation (`train_adaptive_target`), Signal-Measured
//! Endpoint Probing (`REQ-1.6`), and Freeze-Transition Capture (`stimulus::pipeline`).
//!
//! Canonical References:
//! - `docs/specs/training/capture_and_alignment.md`
//!   (`ref[train.capture.excitation.codesigned]`, `ref[train.capture.planner.schedule]`,
//!   `ref[train.capture.planner.freeze-transition]`, `ref[train.capture.invariants]`)
//! - `docs/specs/training/iterative_resume_and_caching.md` (`ref[train.cache.proj-vault]`)

use std::collections::{BTreeMap, BTreeSet};

use dnam_engine::schema::{ControlCalibrationRole, ControlKind, DnamError};

use crate::proj::bundle::{
    ProjChunkInput, build_dnam_proj_container_with_cp1_and_compacted, export_runtime_dnam,
};
use crate::proj::capsules::CalbChunk64B;
use crate::proj::directory::{ProjChunkKind, StmcTrajectoryWindow};
use crate::proj::schema::DnamProjManifest;
use crate::proj::vault::{
    VaultCaptureSpec, initialize_backbone_arena_if_needed, solve_captures_batch_into_proj,
};
use crate::resampler::{resample_arbitrary_rate_to_48k, resampled_frame_len};
use crate::targets::{DistillationTarget, StemResponse, StemStimulus};

use super::batch::{
    align_full_stem_and_seal_vault, build_controls_meta_map, build_freeze_transition_stem_stimulus,
    control_declared_bounds, default_control_vector, resample_48k_channels_to_sr,
};
use super::fft::{
    ADAPTIVE_FRAME_SAMPLES, ADC_PEAK_GOVERNOR_FS, CYCLIC_WINDOW_SAMPLES, PEDESTAL_BIN_MONO,
    build_disjoint_comb_schedule, synthesize_cyclic_comb_frame,
};
use super::laws::{
    HadamardLaw4dTracker, KnobAcousticClass, SweepExecutionMode, TopologyProbeMode,
    build_orthogonal_hadamard_diagonals, classify_knob_from_endpoint_probe,
    compute_hadamard_sfm_ratio, select_sweep_execution_mode,
};
pub use super::null_test::{LiveNullTestReport, run_live_null_test_and_autopatch};
use super::safeguards::CaptureWarning;
use super::waveforms::deterministic_seed_256;
use super::{AdaptiveStimulusController, StimulusConfig};

/// Options controlling Mode-1 closed-loop adaptive capture and distillation (`train_adaptive_target`).
#[derive(Debug, Clone)]
pub struct AdaptiveTrainingOptions {
    /// Sample rate in Hz (`48_000` canonical; `8_000..=384_000` supported).
    pub sample_rate_hz: u32,
    /// Multi-port / multi-channel probe topology.
    pub topology_mode: TopologyProbeMode,
    /// `--fixed-duration`: disables 4-Gate Cramér-Rao early stopping when `true`.
    pub fixed_duration: bool,
    /// `"Prefer Stationary Stops (Mode B)"` override toggle (`r[train.capture.control-sweeps]`).
    pub prefer_mode_b: bool,
    /// Maximum `96 ms` frames per stem (`default = 16`).
    pub max_frames_per_stem: usize,
    /// Automatically execute the `1.0s` Live Null-Test & `< 5 ms` Auto-Patch at capture end.
    pub run_live_null_test: bool,
    /// Null-test ESR threshold in dB above which the `< 5 ms` sufficient-statistic auto-patch runs.
    pub null_test_autopatch_threshold_db: f32,
    /// Optional Phase-0 `CALB` loopback calibration chunk to embed in `.dnam-proj`.
    pub calb: Option<CalbChunk64B>,
}

impl Default for AdaptiveTrainingOptions {
    fn default() -> Self {
        Self {
            sample_rate_hz: 48_000,
            topology_mode: TopologyProbeMode::Mono1To1,
            fixed_duration: false,
            prefer_mode_b: false,
            max_frames_per_stem: 16,
            run_live_null_test: true,
            null_test_autopatch_threshold_db: -36.0,
            calb: None,
        }
    }
}

/// Complete output of [`train_adaptive_target`].
#[derive(Debug, Clone)]
pub struct AdaptiveTrainingReport {
    /// Validated single-file `.dnam-proj` (`DNPJ`) container bytes.
    pub proj_bytes: Vec<u8>,
    /// Exported runtime `.dnam` (`v2.0`) container bytes.
    pub runtime_dnam_bytes: Vec<u8>,
    /// Accumulated non-blocking hardware safeguard warnings across all stems.
    pub warnings: Vec<CaptureWarning>,
    /// Optional `1.0s` Live Null-Test & `< 5 ms` Auto-Patch report.
    pub null_test: Option<LiveNullTestReport>,
    /// Total stems sealed into the `.dnam-proj` vault.
    pub stem_count: usize,
    /// Number of Hadamard cross-term diagonal sweeps executed (`<= 2` orthogonal diagonals).
    pub hadamard_diagonals_run: usize,
    /// Number of decoupled Hadamard diagonals skipped via `Law 4D` commutator pruning.
    pub hadamard_diagonals_pruned: usize,
}

fn spectral_tilt_from_stmc_4096(spec_4096: &[u8; 4_096]) -> f64 {
    let mut weighted_e = 0.0_f64;
    let mut total_e = 0.0_f64;
    for i in 1..512 {
        let off = i * 4;
        let m = f64::from(f32::from_le_bytes(
            spec_4096[off..off + 4].try_into().unwrap(),
        ));
        let p = m * m;
        weighted_e += ((i as f64) / 32.0) * p;
        total_e += p;
    }
    weighted_e / (total_e + 1.0e-18)
}

/// Renders the deterministic endpoint probe (`1` `96 ms` comb frame + `9` `96 ms` silent tail
/// frames = `864 ms` tail) and normalizes the wet response to `48 kHz` (`r[train.capture.invariants]`).
fn render_endpoint_probe_48k(
    target: &mut dyn DistillationTarget,
    probe_dry_l_48k: &[f32],
    sample_rate_hz: u32,
    topology_mode: TopologyProbeMode,
    control_values: &[f32],
) -> Result<Vec<f32>, DnamError> {
    let sr = sample_rate_hz.clamp(8_000, 384_000);
    let need_stereo = matches!(
        topology_mode,
        TopologyProbeMode::TrueStereo2To2 | TopologyProbeMode::DualInToMono2To1
    );
    let need_aux = matches!(
        topology_mode,
        TopologyProbeMode::FourCableMethod4Cm | TopologyProbeMode::DualInToMono2To1
    );
    let dry_r_48k = need_stereo.then_some(probe_dry_l_48k);
    let aux_48k = need_aux.then_some(probe_dry_l_48k);
    let (in_1, in_2, in_3) = resample_48k_channels_to_sr(sr, probe_dry_l_48k, dry_r_48k, aux_48k);

    target.reset_state();
    let resp = target.render_stem(&StemStimulus {
        stem_id: "endpoint_probe".to_string(),
        sample_rate_hz: sr,
        in_1,
        in_2,
        in_3,
        in_4: None,
        control_values: control_values.to_vec(),
        control_trajectories: None,
        freeze_transition_window: None,
    })?;

    let resp_sr = resp.sample_rate_hz.clamp(8_000, 384_000);
    if resp_sr == 48_000 {
        Ok(resp.out_1)
    } else {
        let dst = resampled_frame_len(resp.out_1.len(), resp_sr, 48_000);
        Ok(resample_arbitrary_rate_to_48k(
            &resp.out_1,
            resp_sr,
            Some(dst),
        ))
    }
}

fn run_single_closed_loop_stem(
    target: &mut dyn DistillationTarget,
    cfg: StimulusConfig,
    is_star_center: bool,
    controls_meta: BTreeMap<String, serde_json::Value>,
) -> Result<(Vec<u8>, VaultCaptureSpec, Vec<CaptureWarning>, f64), DnamError> {
    target.reset_state();
    let mut ctrl = AdaptiveStimulusController::new(cfg);
    let mut last_resp: Option<StemResponse> = None;
    while let Some(frame) = ctrl.next_frame(last_resp.as_ref()) {
        last_resp = Some(target.render_frame(&frame)?);
    }
    let warnings = ctrl.warnings().to_vec();
    let (inpc, spec) = ctrl.seal_inpc_and_vault_spec(is_star_center, controls_meta)?;
    let tilt = spectral_tilt_from_stmc_4096(&spec.spectral_summary_4096);
    Ok((inpc, spec, warnings, tilt))
}

fn capture_freeze_transition_stem_mode1(
    manifest: &DnamProjManifest,
    target: &mut dyn DistillationTarget,
    freeze_ctrl_idx: usize,
    options: &AdaptiveTrainingOptions,
    seed_256: &[u8; 256],
    carrier_bins: &[usize],
) -> Result<VaultCaptureSpec, DnamError> {
    let stim = build_freeze_transition_stem_stimulus(
        manifest,
        freeze_ctrl_idx,
        options.sample_rate_hz,
        options.topology_mode,
        seed_256,
    );
    target.reset_state();
    let resp = target.render_stem(&stim)?;

    let to_48k = |ch: &[f32], sr: u32| -> Vec<f32> {
        let csr = sr.clamp(8_000, 384_000);
        if csr == 48_000 {
            ch.to_vec()
        } else {
            let dst = resampled_frame_len(ch.len(), csr, 48_000);
            resample_arbitrary_rate_to_48k(ch, csr, Some(dst))
        }
    };
    let dry_48k_l = to_48k(&stim.in_1, stim.sample_rate_hz);
    let dry_48k_r = stim.in_2.as_ref().map(|r| to_48k(r, stim.sample_rate_hz));
    let wet_48k_l = to_48k(&resp.out_1, resp.sample_rate_hz);
    let wet_48k_r = resp.out_2.as_ref().map(|r| to_48k(r, resp.sample_rate_hz));

    let ctrl_def = &manifest.base.controls_schema[freeze_ctrl_idx];
    let is_sw = ctrl_def.kind == ControlKind::Switch;
    let mut sw_map = BTreeMap::new();
    if is_sw && let Some(opt_name) = ctrl_def.declared_options.as_ref().and_then(|o| o.get(1)) {
        sw_map.insert(freeze_ctrl_idx, opt_name.clone());
    }
    let meta = build_controls_meta_map(manifest, &stim.control_values, &sw_map);
    let mut user_meta = BTreeMap::new();
    if let Some([n0, n1]) = stim.freeze_transition_window {
        user_meta.insert(
            "freeze_transition_window".to_string(),
            serde_json::json!([n0, n1]),
        );
    }
    let hi_val = stim
        .control_values
        .get(freeze_ctrl_idx)
        .copied()
        .unwrap_or(1.0);
    let windows = vec![
        StmcTrajectoryWindow {
            control_id: freeze_ctrl_idx as u16,
            ctrl_kind: u8::from(is_sw),
            active_opt: 0,
            u_phys: 0.0,
            window_weight: 1.0,
            emr_velocity: 0.1,
        },
        StmcTrajectoryWindow {
            control_id: freeze_ctrl_idx as u16,
            ctrl_kind: u8::from(is_sw),
            active_opt: u8::from(is_sw),
            u_phys: hi_val,
            window_weight: 1.0,
            emr_velocity: 0.1,
        },
    ];
    let (_, spec) = align_full_stem_and_seal_vault(
        &stim.stem_id,
        false,
        seed_256,
        carrier_bins,
        5,
        &dry_48k_l,
        dry_48k_r.as_deref(),
        &wet_48k_l,
        wet_48k_r.as_deref(),
        windows,
        meta,
        user_meta,
    )?;
    Ok(spec)
}

/// Mode 1 (Streaming Closed-Loop): Drives `target` frame-by-frame via [`AdaptiveStimulusController`],
/// sealing `INPC`, `STMC`, `STAT`, and `WGHT` into a single-file `.dnam-proj` container and
/// optionally executing [`run_live_null_test_and_autopatch`].
pub fn train_adaptive_target(
    mut manifest: DnamProjManifest,
    target: &mut dyn DistillationTarget,
    options: &AdaptiveTrainingOptions,
) -> Result<AdaptiveTrainingReport, DnamError> {
    if !(8_000..=384_000).contains(&options.sample_rate_hz) {
        return Err(DnamError::InvalidSampleRate(options.sample_rate_hz));
    }
    manifest.stems_registry.clear();
    let mut init_chunks = Vec::new();
    if let Some(calb) = options.calb {
        init_chunks.push(ProjChunkInput {
            kind: ProjChunkKind::Calb,
            chunk_id: 0,
            payload: calb.to_bytes().to_vec(),
        });
    }
    let seeded_arena = initialize_backbone_arena_if_needed(None);
    let mut proj_bytes = build_dnam_proj_container_with_cp1_and_compacted(
        &mut manifest,
        &init_chunks,
        &seeded_arena,
        &[],
        false,
    )?;

    let seed_256 = deterministic_seed_256(0xACE5_2026_0001);
    let noise_floor_dbfs = options.calb.map_or(-100.0, |c| c.noise_floor_dbfs);
    let base_ctrls = default_control_vector(&manifest);
    let empty_switches = BTreeMap::new();
    let mut all_warnings = Vec::new();

    let empty_mask = BTreeSet::new();
    let probe_sched =
        build_disjoint_comb_schedule(48_000, false, Some(PEDESTAL_BIN_MONO), &empty_mask, &[]);
    let probe_amps: Vec<(usize, f64)> =
        probe_sched.carrier_bins.iter().map(|&b| (b, 1.0)).collect();
    let probe_comb = synthesize_cyclic_comb_frame(
        &probe_amps,
        Some(PEDESTAL_BIN_MONO),
        0.05,
        0.16,
        ADC_PEAK_GOVERNOR_FS,
        &[0.0; 64],
    );
    let mut probe_dry_l_48k = probe_comb.samples;
    probe_dry_l_48k.resize(10 * ADAPTIVE_FRAME_SAMPLES, 0.0);

    let has_range_controls = manifest
        .base
        .controls_schema
        .iter()
        .any(|c| c.kind == ControlKind::Range);
    let star_probe_wet = if has_range_controls {
        Some(render_endpoint_probe_48k(
            target,
            &probe_dry_l_48k,
            options.sample_rate_hz,
            options.topology_mode,
            &base_ctrls,
        )?)
    } else {
        None
    };

    // 1. Capture StarCenter (s = 0)
    let star_meta = build_controls_meta_map(&manifest, &base_ctrls, &empty_switches);
    let star_cfg = StimulusConfig {
        stem_id: "star_center".to_string(),
        sample_rate_hz: options.sample_rate_hz,
        topology_mode: options.topology_mode,
        control_values: base_ctrls.clone(),
        sweep_end_control_values: None,
        active_control_id: 0,
        is_master_volume_sweep: false,
        fixed_duration: options.fixed_duration,
        max_frames: options.max_frames_per_stem.max(6),
        max_tail_frames: 3,
        noise_floor_dbfs,
        seed_256,
    };
    let (inpc_bytes, star_spec, star_warns, star_tilt) =
        run_single_closed_loop_stem(target, star_cfg, true, star_meta)?;
    all_warnings.extend(star_warns);
    let mut captured_specs = vec![star_spec];

    // 2. Capture FreezeLatch transitions, 1D Range sweeps/stops & Switch options
    let mut active_range_indices = Vec::new();
    let mut shape_spread_by_ctrl = vec![0.0_f64; manifest.base.controls_schema.len()];
    let controls_snapshot = manifest.base.controls_schema.clone();

    for (p_idx, ctrl) in controls_snapshot.iter().enumerate() {
        if ctrl
            .calibration
            .as_ref()
            .is_some_and(|c| c.role == ControlCalibrationRole::FreezeLatch)
        {
            let freeze_spec = capture_freeze_transition_stem_mode1(
                &manifest,
                target,
                p_idx,
                options,
                &seed_256,
                &probe_sched.carrier_bins,
            )?;
            captured_specs.push(freeze_spec);
        }

        match ctrl.kind {
            ControlKind::Range => {
                active_range_indices.push(p_idx);
                let (lo, hi) = control_declared_bounds(ctrl);
                let probe_u = if (lo - base_ctrls[p_idx]).abs() >= (hi - base_ctrls[p_idx]).abs() {
                    lo
                } else {
                    hi
                };
                let mut probe_v = base_ctrls.clone();
                probe_v[p_idx] = probe_u;
                let end_probe_wet = render_endpoint_probe_48k(
                    target,
                    &probe_dry_l_48k,
                    options.sample_rate_hz,
                    options.topology_mode,
                    &probe_v,
                )?;
                let ref_wet = star_probe_wet.as_deref().unwrap_or(&end_probe_wet);
                let mut ep = classify_knob_from_endpoint_probe(
                    ref_wet,
                    &end_probe_wet,
                    ADAPTIVE_FRAME_SAMPLES,
                    &probe_sched,
                );
                if ctrl.calibration.as_ref().is_some_and(|c| {
                    matches!(
                        c.role,
                        ControlCalibrationRole::DelayTimeMs
                            | ControlCalibrationRole::GrainSizeMs
                            | ControlCalibrationRole::LfoRateHz
                    )
                }) {
                    ep.knob_class = KnobAcousticClass::ClassIIIPitchDelayTime;
                    ep.delta_t_mem_ms = ep.delta_t_mem_ms.max(50.0);
                }
                let is_master = ep.is_master_volume_stage;
                let probe_spread = ep.spectral_shape_spread;
                let mode = select_sweep_execution_mode(
                    ep.delta_t_mem_ms,
                    ep.knob_class,
                    options.prefer_mode_b,
                );

                if mode == SweepExecutionMode::ModeAContinuousSweep {
                    let mut start_v = base_ctrls.clone();
                    let mut end_v = base_ctrls.clone();
                    start_v[p_idx] = lo;
                    end_v[p_idx] = hi;
                    let meta = build_controls_meta_map(&manifest, &end_v, &empty_switches);
                    let cfg = StimulusConfig {
                        stem_id: format!("sweep_ctrl_{p_idx}"),
                        sample_rate_hz: options.sample_rate_hz,
                        topology_mode: options.topology_mode,
                        control_values: start_v,
                        sweep_end_control_values: Some(end_v),
                        active_control_id: p_idx as u16,
                        is_master_volume_sweep: is_master,
                        fixed_duration: options.fixed_duration,
                        max_frames: options.max_frames_per_stem.max(6),
                        max_tail_frames: 2,
                        noise_floor_dbfs,
                        seed_256,
                    };
                    let (_, mut spec, warns, tilt) =
                        run_single_closed_loop_stem(target, cfg, false, meta)?;
                    if spec.windows.len() < 2 {
                        spec.windows = vec![
                            StmcTrajectoryWindow {
                                control_id: p_idx as u16,
                                ctrl_kind: 0,
                                active_opt: 0,
                                u_phys: lo,
                                window_weight: 1.0,
                                emr_velocity: 0.1,
                            },
                            StmcTrajectoryWindow {
                                control_id: p_idx as u16,
                                ctrl_kind: 0,
                                active_opt: 0,
                                u_phys: hi,
                                window_weight: 1.0,
                                emr_velocity: 0.1,
                            },
                        ];
                    } else {
                        spec.windows[0].u_phys = lo;
                        if let Some(last_w) = spec.windows.last_mut() {
                            last_w.u_phys = hi;
                        }
                    }
                    all_warnings.extend(warns);
                    shape_spread_by_ctrl[p_idx] = (tilt - star_tilt).abs().max(probe_spread);
                    captured_specs.push(spec);
                } else {
                    let mut tilts = Vec::new();
                    for (s_idx, &stop_u) in [lo, hi].iter().enumerate() {
                        let mut v = base_ctrls.clone();
                        v[p_idx] = stop_u;
                        let meta = build_controls_meta_map(&manifest, &v, &empty_switches);
                        let cfg = StimulusConfig {
                            stem_id: format!("stop_ctrl_{p_idx}_{s_idx}"),
                            sample_rate_hz: options.sample_rate_hz,
                            topology_mode: options.topology_mode,
                            control_values: v,
                            sweep_end_control_values: None,
                            active_control_id: p_idx as u16,
                            is_master_volume_sweep: is_master,
                            fixed_duration: options.fixed_duration,
                            max_frames: 8,
                            max_tail_frames: 2,
                            noise_floor_dbfs,
                            seed_256,
                        };
                        let (_, spec, warns, tilt) =
                            run_single_closed_loop_stem(target, cfg, false, meta)?;
                        all_warnings.extend(warns);
                        tilts.push(tilt);
                        captured_specs.push(spec);
                    }
                    shape_spread_by_ctrl[p_idx] = (tilts[1] - tilts[0]).abs().max(probe_spread);
                }
            }
            ControlKind::Switch => {
                let opts = ctrl.declared_options.clone().unwrap_or_default();
                for (opt_idx, opt_name) in opts.iter().enumerate().skip(1) {
                    let mut sw_map = BTreeMap::new();
                    sw_map.insert(p_idx, opt_name.clone());
                    let mut sw_ctrls = base_ctrls.clone();
                    sw_ctrls[p_idx] = opt_idx as f32;
                    let meta = build_controls_meta_map(&manifest, &sw_ctrls, &sw_map);
                    let cfg = StimulusConfig {
                        stem_id: format!("switch_{p_idx}_opt_{opt_idx}"),
                        sample_rate_hz: options.sample_rate_hz,
                        topology_mode: options.topology_mode,
                        control_values: sw_ctrls,
                        sweep_end_control_values: None,
                        active_control_id: p_idx as u16,
                        is_master_volume_sweep: false,
                        fixed_duration: options.fixed_duration,
                        max_frames: 6,
                        max_tail_frames: 2,
                        noise_floor_dbfs,
                        seed_256,
                    };
                    let (_, mut spec, warns, _) =
                        run_single_closed_loop_stem(target, cfg, false, meta)?;
                    spec.windows = vec![StmcTrajectoryWindow {
                        control_id: p_idx as u16,
                        ctrl_kind: 1,
                        active_opt: opt_idx as u8,
                        u_phys: opt_idx as f32,
                        window_weight: 1.0,
                        emr_velocity: 0.0,
                    }];
                    all_warnings.extend(warns);
                    captured_specs.push(spec);
                }
            }
        }
    }

    // 3. Law 4D Commutator-Pruned Orthogonal Hadamard Diagonals (`<= 2` diagonals, ADR-0016 §3.1)
    let total_possible_pairs = if active_range_indices.len() >= 2 {
        (active_range_indices.len() * (active_range_indices.len() - 1)) / 2
    } else {
        0
    };
    let orth_diags =
        build_orthogonal_hadamard_diagonals(&active_range_indices, &shape_spread_by_ctrl);
    let pruned_count = total_possible_pairs.saturating_sub(orth_diags.len());
    let mut hadamard_run = 0usize;

    for (diag_idx, signs) in orth_diags.iter().enumerate() {
        let mut v0 = base_ctrls.clone();
        let mut v1 = base_ctrls.clone();
        let first_p = signs.first().map_or(0, |&(p, _)| p);
        for &(p, sign) in signs {
            let (lo, hi) = control_declared_bounds(&controls_snapshot[p]);
            if sign >= 0.0 {
                v0[p] = lo;
                v1[p] = hi;
            } else {
                v0[p] = hi;
                v1[p] = lo;
            }
        }

        target.reset_state();
        let meta = build_controls_meta_map(&manifest, &v1, &empty_switches);
        let mut ctrl = AdaptiveStimulusController::new(StimulusConfig {
            stem_id: format!("hadamard_diag_{diag_idx}"),
            sample_rate_hz: options.sample_rate_hz,
            topology_mode: options.topology_mode,
            control_values: v0,
            sweep_end_control_values: Some(v1),
            active_control_id: first_p as u16,
            is_master_volume_sweep: false,
            fixed_duration: options.fixed_duration,
            max_frames: options.max_frames_per_stem.max(12),
            max_tail_frames: 1,
            noise_floor_dbfs,
            seed_256,
        });
        let mut law4d = HadamardLaw4dTracker::default();
        let mut sfm_hist: [Option<Vec<f64>>; 2] = [None, None];
        let mut last_resp: Option<StemResponse> = None;
        while let Some(frame) = ctrl.next_frame(last_resp.as_ref()) {
            let r = target.render_frame(&frame)?;
            if ctrl.is_in_active_learning() && !options.fixed_duration {
                let (wl_48k, _) = ctrl.normalize_wet_to_48k(&r);
                let wet_win = &wl_48k[wl_48k.len().saturating_sub(CYCLIC_WINDOW_SAMPLES)..];
                let sfm = compute_hadamard_sfm_ratio(wet_win, ctrl.schedule_l(), &mut sfm_hist);
                if law4d.observe_frame(sfm) {
                    ctrl.request_coreset_handoff();
                }
            }
            last_resp = Some(r);
        }
        all_warnings.extend_from_slice(ctrl.warnings());
        let (_, spec) = ctrl.seal_inpc_and_vault_spec(false, meta)?;
        captured_specs.push(spec);
        hadamard_run += 1;
    }

    let stem_count = captured_specs.len();
    proj_bytes = solve_captures_batch_into_proj(&proj_bytes, captured_specs, Some(&inpc_bytes))?;

    let mut runtime_dnam_bytes = export_runtime_dnam(&proj_bytes, None)?;
    let null_test = if options.run_live_null_test {
        let (updated_proj, updated_dnam, report) = run_live_null_test_and_autopatch(
            &proj_bytes,
            target,
            options.null_test_autopatch_threshold_db,
        )?;
        proj_bytes = updated_proj;
        runtime_dnam_bytes = updated_dnam;
        Some(report)
    } else {
        None
    };

    Ok(AdaptiveTrainingReport {
        proj_bytes,
        runtime_dnam_bytes,
        warnings: all_warnings,
        null_test,
        stem_count,
        hadamard_diagonals_run: hadamard_run,
        hadamard_diagonals_pruned: pruned_count,
    })
}
