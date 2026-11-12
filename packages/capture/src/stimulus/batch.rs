#![forbid(unsafe_code)]
//! Mode 2 (1-or-2-Pass Sans-IO Batch Distillation): `generate_stimulus_batch`,
//! `generate_refinement_stimulus_batch`, and `solve_from_responses` (`stimulus::batch`).
//!
//! Canonical References:
//! - `docs/specs/training/capture_and_alignment.md`
//!   (`ref[train.capture.excitation.codesigned]`, `ref[train.capture.planner.freeze-transition]`,
//!   `ref[train.capture.invariants]`)
//! - `docs/specs/training/iterative_resume_and_caching.md` (`ref[train.cache.proj-vault]`)

use std::collections::{BTreeMap, BTreeSet};

use dnam_engine::schema::{ControlCalibrationRole, ControlKind, DnamError};

use crate::CORESET_FP16_FRAMES_PER_STEM;
use crate::proj::bundle::{
    ProjChunkInput, build_dnam_proj_container_with_cp1_and_compacted, export_runtime_dnam,
};
use crate::proj::capsules::CalbChunk64B;
use crate::proj::directory::{ProjChunkKind, StmcTrajectoryWindow};
use crate::proj::schema::DnamProjManifest;
use crate::proj::vault::{initialize_backbone_arena_if_needed, solve_captures_batch_into_proj};
use crate::targets::{StemResponse, StemStimulus, StemStimulusBatch};

use super::laws::TopologyProbeMode;
use super::safeguards::{CaptureWarning, SafeguardState, evaluate_frame_safeguards};
pub use super::sealing::recapture_clipped_stem;
pub(crate) use super::sealing::{
    CORESET_WITH_SETTLE_48K, align_full_stem_and_seal_vault, build_controls_meta_map,
    build_freeze_transition_stem_stimulus, control_declared_bounds, default_control_vector,
    normalize_stem_channel_to_48k, resample_48k_channels_to_sr, seal_coresets_to_inpc_and_vault,
};
use super::waveforms::{
    ActivePrimitiveKind, deterministic_seed_256, synthesize_a3_monophonic_glissando,
    synthesize_a5_musical_transient_coreset, synthesize_active_primitive_frame,
};
use super::{
    ADAPTIVE_FRAME_SAMPLES, ADC_PEAK_GOVERNOR_FS, PEDESTAL_BIN_MONO, PEDESTAL_BIN_STEREO_R,
    build_disjoint_comb_schedule, synthesize_cyclic_comb_frame,
};

/// Options for Mode-2 Sans-IO batch stimulus generation (`generate_stimulus_batch`).
#[derive(Debug, Clone)]
pub struct BatchStimulusOptions {
    /// Sample rate in Hz (`48_000` canonical; `8_000..=384_000` supported).
    pub sample_rate_hz: u32,
    /// Multi-port / multi-channel probe topology.
    pub topology_mode: TopologyProbeMode,
    /// Force stationary `Mode B` stops instead of `Mode A` continuous sweeps.
    pub prefer_mode_b: bool,
    /// Deterministic master seed.
    pub master_seed: u64,
}

impl Default for BatchStimulusOptions {
    fn default() -> Self {
        Self {
            sample_rate_hz: 48_000,
            topology_mode: TopologyProbeMode::Mono1To1,
            prefer_mode_b: false,
            master_seed: 0xACE5_2026_BA7C,
        }
    }
}

/// Output of Mode-2 Sans-IO [`solve_from_responses`].
#[derive(Debug, Clone)]
pub struct BatchSolveReport {
    /// Validated single-file `.dnam-proj` (`DNPJ`) container bytes.
    pub proj_bytes: Vec<u8>,
    /// Exported runtime `.dnam` (`v2.0`) container bytes.
    pub runtime_dnam_bytes: Vec<u8>,
    /// Accumulated hardware safeguard warnings detected across batch stems.
    pub warnings: Vec<CaptureWarning>,
    /// Total stems sealed into `.dnam-proj`.
    pub stem_count: usize,
}

/// Mode 2 (Pass 1): Synthesizes a deterministic [`StemStimulusBatch`] (`StarCenter` + 1D sweeps/stops
/// + discrete `Switch` options + timed `FreezeLatch` transitions) for offline/parallel runners.
pub fn generate_stimulus_batch(
    manifest: &DnamProjManifest,
    options: &BatchStimulusOptions,
) -> Result<StemStimulusBatch, DnamError> {
    let sr = options.sample_rate_hz;
    if !(8_000..=384_000).contains(&sr) {
        return Err(DnamError::InvalidSampleRate(sr));
    }
    let seed_256 = deterministic_seed_256(options.master_seed);
    let empty_mask = BTreeSet::new();
    let sched_l =
        build_disjoint_comb_schedule(48_000, false, Some(PEDESTAL_BIN_MONO), &empty_mask, &[]);
    let bin_amps_l: Vec<(usize, f64)> = sched_l.carrier_bins.iter().map(|&b| (b, 1.0)).collect();
    let comb_l = synthesize_cyclic_comb_frame(
        &bin_amps_l,
        Some(PEDESTAL_BIN_MONO),
        0.05,
        0.15,
        ADC_PEAK_GOVERNOR_FS,
        &[0.0; 64],
    );
    let need_stereo = matches!(
        options.topology_mode,
        TopologyProbeMode::TrueStereo2To2 | TopologyProbeMode::DualInToMono2To1
    );
    let need_aux = matches!(
        options.topology_mode,
        TopologyProbeMode::FourCableMethod4Cm | TopologyProbeMode::DualInToMono2To1
    );
    let (mut coreset_l, mut coreset_r) = synthesize_a5_musical_transient_coreset(
        &seed_256,
        CORESET_FP16_FRAMES_PER_STEM,
        if need_stereo { 2 } else { 1 },
        48_000,
    );
    coreset_l.resize(CORESET_WITH_SETTLE_48K, 0.0);
    if let Some(ref mut cr) = coreset_r {
        cr.resize(CORESET_WITH_SETTLE_48K, 0.0);
    }

    let mut base_wave_l_48k =
        Vec::with_capacity(6 * ADAPTIVE_FRAME_SAMPLES + CORESET_WITH_SETTLE_48K);
    base_wave_l_48k.extend_from_slice(&comb_l.samples);
    base_wave_l_48k.extend_from_slice(&comb_l.samples);
    for ep_f in 0..4 {
        base_wave_l_48k.extend_from_slice(&synthesize_active_primitive_frame(
            ActivePrimitiveKind::P5PumpAndProbeBloom,
            ep_f,
            false,
            ADC_PEAK_GOVERNOR_FS,
        ));
    }
    let active_prefix_48k = base_wave_l_48k.len();
    base_wave_l_48k.extend_from_slice(&coreset_l);

    let base_wave_r_48k = coreset_r.map(|cr| {
        let mut wr = base_wave_l_48k[..active_prefix_48k].to_vec();
        wr.extend_from_slice(&cr);
        wr
    });
    let base_aux_48k = if need_aux {
        let sched_r = build_disjoint_comb_schedule(
            48_000,
            true,
            Some(PEDESTAL_BIN_STEREO_R),
            &empty_mask,
            &[],
        );
        let bin_amps_r: Vec<(usize, f64)> =
            sched_r.carrier_bins.iter().map(|&b| (b, 1.0)).collect();
        let comb_aux = synthesize_cyclic_comb_frame(
            &bin_amps_r,
            Some(PEDESTAL_BIN_STEREO_R),
            0.05,
            0.15,
            ADC_PEAK_GOVERNOR_FS,
            &[0.0; 64],
        );
        let mut aux = Vec::with_capacity(base_wave_l_48k.len());
        aux.extend_from_slice(&comb_aux.samples);
        aux.extend_from_slice(&comb_aux.samples);
        aux.extend_from_slice(&base_wave_l_48k[2 * ADAPTIVE_FRAME_SAMPLES..]);
        Some(aux)
    } else {
        None
    };

    let (base_wave_l, base_wave_r, base_aux) = resample_48k_channels_to_sr(
        sr,
        &base_wave_l_48k,
        base_wave_r_48k.as_deref(),
        base_aux_48k.as_deref(),
    );

    let base_ctrls = default_control_vector(manifest);
    let mut stems = vec![StemStimulus {
        stem_id: "star_center".to_string(),
        sample_rate_hz: sr,
        in_1: base_wave_l.clone(),
        in_2: base_wave_r.clone(),
        in_3: base_aux.clone(),
        in_4: None,
        control_values: base_ctrls.clone(),
        control_trajectories: None,
        freeze_transition_window: None,
    }];

    for (p_idx, ctrl) in manifest.base.controls_schema.iter().enumerate() {
        let is_freeze_latch = ctrl
            .calibration
            .as_ref()
            .is_some_and(|c| c.role == ControlCalibrationRole::FreezeLatch);
        if is_freeze_latch {
            stems.push(build_freeze_transition_stem_stimulus(
                manifest,
                p_idx,
                sr,
                options.topology_mode,
                &seed_256,
            ));
        }

        match ctrl.kind {
            ControlKind::Range => {
                let (lo, hi) = control_declared_bounds(ctrl);
                let force_mode_b = options.prefer_mode_b
                    || ctrl.calibration.as_ref().is_some_and(|c| {
                        matches!(
                            c.role,
                            ControlCalibrationRole::DelayTimeMs
                                | ControlCalibrationRole::GrainSizeMs
                                | ControlCalibrationRole::LfoRateHz
                        )
                    });
                if force_mode_b {
                    for (si, &u) in [lo, hi].iter().enumerate() {
                        let mut v = base_ctrls.clone();
                        v[p_idx] = u;
                        stems.push(StemStimulus {
                            stem_id: format!("stop_ctrl_{p_idx}_{si}"),
                            sample_rate_hz: sr,
                            in_1: base_wave_l.clone(),
                            in_2: base_wave_r.clone(),
                            in_3: base_aux.clone(),
                            in_4: None,
                            control_values: v,
                            control_trajectories: None,
                            freeze_transition_window: None,
                        });
                    }
                } else {
                    let n = base_wave_l.len();
                    let prefix_sr = (active_prefix_48k as u64 * u64::from(sr) / 48_000) as usize;
                    let coreset_sr = ((CORESET_FP16_FRAMES_PER_STEM as u64 * u64::from(sr))
                        / 48_000)
                        .max(1) as usize;
                    let mut trajs = Vec::with_capacity(base_ctrls.len());
                    for (q, &def_v) in base_ctrls.iter().enumerate() {
                        if q == p_idx {
                            trajs.push(
                                (0..n)
                                    .map(|i| {
                                        let frac = if i < prefix_sr {
                                            i as f32 / (prefix_sr.max(1) as f32)
                                        } else {
                                            ((i - prefix_sr) as f32 / (coreset_sr as f32))
                                                .clamp(0.0, 1.0)
                                        };
                                        lo + (hi - lo) * frac
                                    })
                                    .collect(),
                            );
                        } else {
                            trajs.push(vec![def_v; n]);
                        }
                    }
                    let mut end_v = base_ctrls.clone();
                    end_v[p_idx] = hi;
                    stems.push(StemStimulus {
                        stem_id: format!("sweep_ctrl_{p_idx}"),
                        sample_rate_hz: sr,
                        in_1: base_wave_l.clone(),
                        in_2: base_wave_r.clone(),
                        in_3: base_aux.clone(),
                        in_4: None,
                        control_values: end_v,
                        control_trajectories: Some(trajs),
                        freeze_transition_window: None,
                    });
                }
            }
            ControlKind::Switch => {
                let opts = ctrl.declared_options.as_deref().unwrap_or(&[]);
                for opt_idx in 1..opts.len() {
                    let mut v = base_ctrls.clone();
                    v[p_idx] = opt_idx as f32;
                    stems.push(StemStimulus {
                        stem_id: format!("switch_{p_idx}_opt_{opt_idx}"),
                        sample_rate_hz: sr,
                        in_1: base_wave_l.clone(),
                        in_2: base_wave_r.clone(),
                        in_3: base_aux.clone(),
                        in_4: None,
                        control_values: v,
                        control_trajectories: None,
                        freeze_transition_window: None,
                    });
                }
            }
        }
    }

    Ok(StemStimulusBatch {
        sample_rate_hz: sr,
        stems,
    })
}

/// Mode 2 (Optional Pass 2): Analyzes all Pass-1 responses using `Laws 0.A–0.F` & `Laws 1–3` and emits
/// a targeted refinement [`StemStimulusBatch`] if sub-harmonic (`Law 0.E`), gated (`Branch 1A`), or
/// clipped (`Law 0.D`/`0.F`) behavior is detected (`None` if Pass 1 already converged).
pub fn generate_refinement_stimulus_batch(
    manifest: &DnamProjManifest,
    pass1_batch: &StemStimulusBatch,
    pass1_responses: &[StemResponse],
) -> Result<Option<StemStimulusBatch>, DnamError> {
    if pass1_batch.stems.is_empty() || pass1_responses.is_empty() {
        return Ok(None);
    }
    let sr = pass1_batch.sample_rate_hz.clamp(8_000, 384_000);
    let empty_mask = BTreeSet::new();
    let sched =
        build_disjoint_comb_schedule(48_000, false, Some(PEDESTAL_BIN_MONO), &empty_mask, &[]);
    let seed_256 = deterministic_seed_256(0xACE5_2026_BA7C);
    let (mut coreset_l_48k, _) =
        synthesize_a5_musical_transient_coreset(&seed_256, CORESET_FP16_FRAMES_PER_STEM, 1, 48_000);
    coreset_l_48k.resize(CORESET_WITH_SETTLE_48K, 0.0);

    let default_ctrls = default_control_vector(manifest);
    let mut oc2_ctrls: Option<Vec<f32>> = None;
    let mut gate_ctrls: Option<Vec<f32>> = None;
    let mut clip_backoff: Option<(Vec<f32>, f64)> = None;

    for (stim, resp) in pass1_batch.stems.iter().zip(pass1_responses) {
        let dry_48k = normalize_stem_channel_to_48k(&stim.in_1, stim.sample_rate_hz);
        let wet_48k = normalize_stem_channel_to_48k(&resp.out_1, resp.sample_rate_hz);
        if dry_48k.len() < ADAPTIVE_FRAME_SAMPLES || wet_48k.len() < ADAPTIVE_FRAME_SAMPLES {
            continue;
        }
        let wet_r_48k = resp
            .out_2
            .as_ref()
            .map(|r| normalize_stem_channel_to_48k(r, resp.sample_rate_hz));

        let mut sg = SafeguardState::new(stim.stem_id.clone(), false, -100.0);
        let _ = evaluate_frame_safeguards(
            &mut sg,
            0,
            &dry_48k[..ADAPTIVE_FRAME_SAMPLES],
            &wet_48k[..ADAPTIVE_FRAME_SAMPLES],
            wet_r_48k
                .as_deref()
                .map(|r| &r[..ADAPTIVE_FRAME_SAMPLES.min(r.len())]),
            &sched,
        );

        let check_len = wet_48k.len().min(2 * ADAPTIVE_FRAME_SAMPLES);
        let out_rms = (wet_48k[..check_len]
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum::<f64>()
            / (check_len as f64))
            .sqrt();

        let ctrls = if stim.control_values.is_empty() {
            default_ctrls.clone()
        } else {
            stim.control_values.clone()
        };
        if sg.oc2_glissando_handoff && oc2_ctrls.is_none() {
            oc2_ctrls = Some(ctrls.clone());
        }
        if out_rms < 1.0e-3 && gate_ctrls.is_none() {
            gate_ctrls = Some(ctrls.clone());
        }
        if sg.excitation_gain_scale < 0.99 && clip_backoff.is_none() {
            clip_backoff = Some((ctrls, sg.excitation_gain_scale));
        }
    }

    let mut ref_stems = Vec::new();
    if let Some(ctrls) = oc2_ctrls {
        let mut wave_48k = synthesize_a3_monophonic_glissando(48_000, 0.75 * ADC_PEAK_GOVERNOR_FS);
        wave_48k.extend_from_slice(&coreset_l_48k);
        let (in_1, _, _) = resample_48k_channels_to_sr(sr, &wave_48k, None, None);
        ref_stems.push(StemStimulus {
            stem_id: "refinement_oc2_glissando".to_string(),
            sample_rate_hz: sr,
            in_1,
            in_2: None,
            in_3: None,
            in_4: None,
            control_values: ctrls,
            control_trajectories: None,
            freeze_transition_window: None,
        });
    } else if let Some(ctrls) = gate_ctrls {
        let mut wave_48k = synthesize_active_primitive_frame(
            ActivePrimitiveKind::P2MultiDriveBurst,
            0,
            false,
            ADC_PEAK_GOVERNOR_FS,
        );
        wave_48k.extend_from_slice(&coreset_l_48k);
        let (in_1, _, _) = resample_48k_channels_to_sr(sr, &wave_48k, None, None);
        ref_stems.push(StemStimulus {
            stem_id: "refinement_gate_open".to_string(),
            sample_rate_hz: sr,
            in_1,
            in_2: None,
            in_3: None,
            in_4: None,
            control_values: ctrls,
            control_trajectories: None,
            freeze_transition_window: None,
        });
    } else if let Some((ctrls, scale)) = clip_backoff {
        let bin_amps: Vec<(usize, f64)> = sched.carrier_bins.iter().map(|&b| (b, 1.0)).collect();
        let comb = synthesize_cyclic_comb_frame(
            &bin_amps,
            Some(PEDESTAL_BIN_MONO),
            0.05 * scale,
            0.15 * scale,
            ADC_PEAK_GOVERNOR_FS * scale,
            &[0.0; 64],
        );
        let mut wave_48k = comb.samples;
        wave_48k.extend(coreset_l_48k.iter().map(|&s| s * (scale as f32)));
        let (in_1, _, _) = resample_48k_channels_to_sr(sr, &wave_48k, None, None);
        ref_stems.push(StemStimulus {
            stem_id: "refinement_clip_backoff".to_string(),
            sample_rate_hz: sr,
            in_1,
            in_2: None,
            in_3: None,
            in_4: None,
            control_values: ctrls,
            control_trajectories: None,
            freeze_transition_window: None,
        });
    }

    if ref_stems.is_empty() {
        Ok(None)
    } else {
        Ok(Some(StemStimulusBatch {
            sample_rate_hz: sr,
            stems: ref_stems,
        }))
    }
}

fn infer_batch_stem_windows_and_switches(
    manifest: &DnamProjManifest,
    base_ctrls: &[f32],
    stim: &StemStimulus,
) -> (
    Vec<StmcTrajectoryWindow>,
    BTreeMap<usize, String>,
    BTreeMap<String, serde_json::Value>,
) {
    let mut switch_overrides = BTreeMap::new();
    let mut user_meta = BTreeMap::new();

    if let Some([n0, n1]) = stim.freeze_transition_window {
        user_meta.insert(
            "freeze_transition_window".to_string(),
            serde_json::json!([n0, n1]),
        );
        let freeze_p = manifest
            .base
            .controls_schema
            .iter()
            .position(|c| {
                c.calibration
                    .as_ref()
                    .is_some_and(|cal| cal.role == ControlCalibrationRole::FreezeLatch)
            })
            .unwrap_or(0);
        let is_sw = manifest
            .base
            .controls_schema
            .get(freeze_p)
            .is_some_and(|c| c.kind == ControlKind::Switch);
        if is_sw
            && let Some(opt_name) = manifest.base.controls_schema[freeze_p]
                .declared_options
                .as_ref()
                .and_then(|o| o.get(1))
        {
            switch_overrides.insert(freeze_p, opt_name.clone());
        }
        let hi_val = stim.control_values.get(freeze_p).copied().unwrap_or(1.0);
        let windows = vec![
            StmcTrajectoryWindow {
                control_id: freeze_p as u16,
                ctrl_kind: u8::from(is_sw),
                active_opt: 0,
                u_phys: 0.0,
                window_weight: 1.0,
                emr_velocity: 0.1,
            },
            StmcTrajectoryWindow {
                control_id: freeze_p as u16,
                ctrl_kind: u8::from(is_sw),
                active_opt: u8::from(is_sw),
                u_phys: hi_val,
                window_weight: 1.0,
                emr_velocity: 0.1,
            },
        ];
        return (windows, switch_overrides, user_meta);
    }

    if let Some(ref trajs) = stim.control_trajectories
        && !trajs.is_empty()
    {
        let swept: Vec<(usize, f32, f32)> = trajs
            .iter()
            .enumerate()
            .filter_map(|(p, tr)| {
                let (&a, &b) = tr.first().zip(tr.last())?;
                ((b - a).abs() > 1.0e-5).then_some((p, a, b))
            })
            .collect();
        if !swept.is_empty() {
            let knots = (36 / swept.len()).clamp(2, 6);
            let mut windows = Vec::with_capacity(knots * swept.len());
            for k in 0..knots {
                let frac = (k as f32) / ((knots - 1) as f32);
                for &(p, a, b) in &swept {
                    windows.push(StmcTrajectoryWindow {
                        control_id: p as u16,
                        ctrl_kind: 0,
                        active_opt: 0,
                        u_phys: a + (b - a) * frac,
                        window_weight: 1.0,
                        emr_velocity: 1.0 / (knots as f32),
                    });
                }
            }
            return (windows, switch_overrides, user_meta);
        }
    }

    for (p, ctrl) in manifest.base.controls_schema.iter().enumerate() {
        if ctrl.kind == ControlKind::Switch {
            let opt_idx = stim
                .control_values
                .get(p)
                .map(|&v| v.round().max(0.0) as usize)
                .unwrap_or(0);
            if opt_idx > 0 {
                if let Some(opt_name) = ctrl
                    .declared_options
                    .as_ref()
                    .and_then(|opts| opts.get(opt_idx))
                {
                    switch_overrides.insert(p, opt_name.clone());
                }
                return (
                    vec![StmcTrajectoryWindow {
                        control_id: p as u16,
                        ctrl_kind: 1,
                        active_opt: opt_idx as u8,
                        u_phys: opt_idx as f32,
                        window_weight: 1.0,
                        emr_velocity: 0.0,
                    }],
                    switch_overrides,
                    user_meta,
                );
            }
        }
    }

    let parsed_stop_p = stim
        .stem_id
        .strip_prefix("stop_ctrl_")
        .and_then(|rest| rest.split('_').next())
        .and_then(|tok| tok.parse::<usize>().ok())
        .filter(|&p| p < manifest.base.controls_schema.len());
    let active_p = parsed_stop_p.or_else(|| {
        manifest
            .base
            .controls_schema
            .iter()
            .enumerate()
            .find_map(|(p, ctrl)| {
                (ctrl.kind == ControlKind::Range
                    && stim
                        .control_values
                        .get(p)
                        .zip(base_ctrls.get(p))
                        .is_some_and(|(&u, &def_u)| (u - def_u).abs() > 1.0e-5))
                .then_some(p)
            })
    });
    let p_id = active_p.unwrap_or(0);
    let u_val = stim
        .control_values
        .get(p_id)
        .or_else(|| stim.control_values.first())
        .copied()
        .unwrap_or(0.0);
    (
        vec![StmcTrajectoryWindow {
            control_id: p_id as u16,
            ctrl_kind: 0,
            active_opt: 0,
            u_phys: u_val,
            window_weight: 1.0,
            emr_velocity: 0.0,
        }],
        switch_overrides,
        user_meta,
    )
}

/// Mode 2 (Sans-IO Solve): Aligns & safeguards 1-or-2-pass batch responses across arbitrary
/// `8..384 kHz` device sample rates and seals `.dnam-proj` (`INPC`, `STMC`, `STAT`, `WGHT`) + `.dnam`.
pub fn solve_from_responses(
    mut manifest: DnamProjManifest,
    batches: &[(&StemStimulusBatch, &[StemResponse])],
    calb: Option<CalbChunk64B>,
) -> Result<BatchSolveReport, DnamError> {
    if batches.is_empty() {
        return Err(DnamError::SchemaValidationError(
            "solve_from_responses requires at least one stimulus/response batch".to_string(),
        ));
    }
    manifest.stems_registry.clear();
    let mut init_chunks = Vec::new();
    if let Some(c) = calb {
        init_chunks.push(ProjChunkInput {
            kind: ProjChunkKind::Calb,
            chunk_id: 0,
            payload: c.to_bytes().to_vec(),
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

    let seed_256 = deterministic_seed_256(0xACE5_2026_BA7C);
    let empty_mask = BTreeSet::new();
    let sched =
        build_disjoint_comb_schedule(48_000, false, Some(PEDESTAL_BIN_MONO), &empty_mask, &[]);
    let base_ctrls = default_control_vector(&manifest);
    let mut inpc_built: Option<Vec<u8>> = None;
    let mut warnings = Vec::new();
    let mut specs = Vec::new();

    for &(batch, resps) in batches {
        if batch.stems.len() != resps.len() {
            return Err(DnamError::SchemaValidationError(format!(
                "batch stimulus count ({}) != response count ({})",
                batch.stems.len(),
                resps.len()
            )));
        }
        for (stim, resp) in batch.stems.iter().zip(resps) {
            let dry_48k_l = normalize_stem_channel_to_48k(&stim.in_1, stim.sample_rate_hz);
            let dry_48k_r = stim
                .in_2
                .as_ref()
                .map(|r| normalize_stem_channel_to_48k(r, stim.sample_rate_hz));
            let wet_48k_l = normalize_stem_channel_to_48k(&resp.out_1, resp.sample_rate_hz);
            let wet_48k_r = resp
                .out_2
                .as_ref()
                .map(|r| normalize_stem_channel_to_48k(r, resp.sample_rate_hz));

            if dry_48k_l.len() < CORESET_FP16_FRAMES_PER_STEM
                || wet_48k_l.len() < CORESET_FP16_FRAMES_PER_STEM
            {
                return Err(DnamError::SchemaValidationError(format!(
                    "stem '{}' shorter than {} coreset frames at 48 kHz",
                    stim.stem_id, CORESET_FP16_FRAMES_PER_STEM
                )));
            }
            let mut sg = SafeguardState::new(
                stim.stem_id.clone(),
                false,
                calb.map_or(-100.0, |c| c.noise_floor_dbfs),
            );
            if dry_48k_l.len() >= ADAPTIVE_FRAME_SAMPLES
                && wet_48k_l.len() >= ADAPTIVE_FRAME_SAMPLES
            {
                let _ = evaluate_frame_safeguards(
                    &mut sg,
                    0,
                    &dry_48k_l[..ADAPTIVE_FRAME_SAMPLES],
                    &wet_48k_l[..ADAPTIVE_FRAME_SAMPLES],
                    wet_48k_r
                        .as_deref()
                        .map(|r| &r[..ADAPTIVE_FRAME_SAMPLES.min(r.len())]),
                    &sched,
                );
            }
            warnings.extend(sg.warnings);

            let (windows, sw_overrides, user_meta) =
                infer_batch_stem_windows_and_switches(&manifest, &base_ctrls, stim);
            let is_star = specs.is_empty();
            let meta = build_controls_meta_map(&manifest, &stim.control_values, &sw_overrides);
            let (inpc, spec) = align_full_stem_and_seal_vault(
                &stim.stem_id,
                is_star,
                &seed_256,
                &sched.carrier_bins,
                2,
                &dry_48k_l,
                dry_48k_r.as_deref(),
                &wet_48k_l,
                wet_48k_r.as_deref(),
                windows,
                meta,
                user_meta,
            )?;
            if inpc_built.is_none() {
                inpc_built = Some(inpc);
            }
            specs.push(spec);
        }
    }

    let stem_count = specs.len();
    proj_bytes = solve_captures_batch_into_proj(&proj_bytes, specs, inpc_built.as_deref())?;
    let runtime_dnam_bytes = export_runtime_dnam(&proj_bytes, None)?;
    Ok(BatchSolveReport {
        proj_bytes,
        runtime_dnam_bytes,
        warnings,
        stem_count,
    })
}
