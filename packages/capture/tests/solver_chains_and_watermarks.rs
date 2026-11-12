#![forbid(unsafe_code)]
//! Step 2.3 End-to-End Distillation of All 12 Physical Signal Chains (`Chain-01..12`) &
//! `Case #8` 3-Layer Buyer Watermarking, `N = 5` Collusion Attacks, and Black-Box Audio-Only
//! Pole-Angle Phase Identification (`dnam_capture::solver`).
//!
//! Canonical References:
//! - `docs/specs/training/four_stage_pipeline.md` (`r[train.pipeline.stage-a-poles]`,
//!   `r[train.pipeline.stage-e-pack]`, `r[train.pipeline.fixtures-matrix]`)
//! - `docs/specs/security/watermarking_and_marketplace.md`

use std::collections::BTreeMap;

use dnam_capture::fp16_bits_to_f32;
use dnam_capture::metrics::{evaluate_besr_from_band_energies, split_bezout_3band_energies};
use dnam_capture::proj::{
    DnamProjManifest, DnamProjView, PROJ_STAT_BYTES_ULTRA, ProjChunkKind, STMC_FLAG_STEREO,
    export_runtime_dnam, verify_proj_inpc_capsule, verify_proj_stmc_capsule,
};
use dnam_capture::solver::{
    BuyerWatermarkConfig, canonical_besr_warmup, compute_integrated_lufs, derive_buyer_phase_seed,
    derive_buyer_tardos_codebook, export_runtime_dnam_watermarked, extract_arena_pole_angles_34,
    identify_pole_angles_blackbox_from_audio, render_blackbox_probe_response,
    star_center_control_values,
};
use dnam_capture::stimulus::{AdaptiveTrainingOptions, TopologyProbeMode, train_adaptive_target};
use dnam_capture::{
    DistillationTarget, StemResponse, StemStimulus, StimulusFrame, resample_arbitrary_rate,
    resample_control_trajectory, resampled_frame_len, scale_freeze_window,
};
use dnam_engine::container::DnamContainerView;
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::provenance::{
    DefaultCryptoProvider, TARDOS_CASCADE_WEIGHTS, extract_structural_hash_224d,
    is_role_a_cascade_index, score_phase_coding_arena, score_phase_coding_blackbox,
    score_tardos_watermark_7680, structural_cosine_similarity,
};
use dnam_engine::schema::{ControlSchemaDef, DnamError};
use dnam_engine::{DnamEngine, Tier};
use dnam_fixtures::{
    BlackBoxModel, Chain01SymmetricOd, Chain02GatedAsymmetricFuzz, Chain03ParallelCleanBlend,
    Chain04TubeAmpFmv4x12, Chain05PureLtiCabAndRoom, Chain06EnvelopeResonantSvf,
    Chain07StereoChorusAndRotary, Chain08LongBbdEchoAnd30sShimmer, Chain09La2aOpticalCompressor,
    Chain10Oc2OctaveDividerAndRingMod, Chain11BiAmpBassAndOutputIron, Chain12FullEndToEndStereoRig,
    Chain12RigMode,
};

#[derive(Clone, Debug)]
struct BlackBoxDistillationAdapter<M: BlackBoxModel> {
    inner: M,
    switches: Vec<u8>,
}

impl<M: BlackBoxModel> BlackBoxDistillationAdapter<M> {
    fn from_model(inner: M) -> Self {
        let switches = inner.metadata().default_switches();
        Self { inner, switches }
    }

    #[allow(clippy::too_many_arguments)]
    fn render_internal(
        &mut self,
        stem_id: String,
        stim_sr: u32,
        in_1: &[f32],
        in_2: Option<&[f32]>,
        in_3: Option<&[f32]>,
        in_4: Option<&[f32]>,
        control_values: &[f32],
        control_trajectories: Option<&Vec<Vec<f32>>>,
        freeze_window: Option<[usize; 2]>,
    ) -> Result<StemResponse, DnamError> {
        let dev_sr = self.inner.sample_rate_hz();
        let orig_n = in_1.len();
        if orig_n == 0 {
            let n_out = self.inner.num_outputs().clamp(1, 4);
            return Ok(StemResponse {
                stem_id,
                sample_rate_hz: stim_sr,
                out_1: Vec::new(),
                out_2: (n_out >= 2).then(Vec::new),
                out_3: (n_out >= 3).then(Vec::new),
                out_4: (n_out >= 4).then(Vec::new),
                internal_nodes: BTreeMap::new(),
            });
        }
        let needs_resample = stim_sr != dev_sr;
        let dev_n = if needs_resample {
            resampled_frame_len(orig_n, stim_sr, dev_sr)
        } else {
            orig_n
        };
        let rs_in1;
        let rs_in2;
        let rs_in3;
        let rs_in4;
        let rs_trajs;
        let (dev_in1, dev_in2, dev_in3, dev_in4, dev_trajs, dev_freeze) = if needs_resample {
            rs_in1 = resample_arbitrary_rate(in_1, stim_sr, dev_sr, Some(dev_n));
            rs_in2 = in_2.map(|ch| resample_arbitrary_rate(ch, stim_sr, dev_sr, Some(dev_n)));
            rs_in3 = in_3.map(|ch| resample_arbitrary_rate(ch, stim_sr, dev_sr, Some(dev_n)));
            rs_in4 = in_4.map(|ch| resample_arbitrary_rate(ch, stim_sr, dev_sr, Some(dev_n)));
            rs_trajs = control_trajectories.map(|trajs| {
                trajs
                    .iter()
                    .map(|tr| resample_control_trajectory(tr, dev_n))
                    .collect::<Vec<Vec<f32>>>()
            });
            (
                rs_in1.as_slice(),
                rs_in2.as_deref(),
                rs_in3.as_deref(),
                rs_in4.as_deref(),
                rs_trajs.as_ref(),
                freeze_window.map(|w| scale_freeze_window(w, orig_n, dev_n)),
            )
        } else {
            (in_1, in_2, in_3, in_4, control_trajectories, freeze_window)
        };

        let meta = self.inner.metadata();
        let default_controls = meta.default_controls();
        let freeze_sw_idx = meta
            .switches
            .iter()
            .position(|sw| sw.id == "freeze")
            .unwrap_or(0);
        let n_sw = self
            .switches
            .len()
            .max(meta.switches.len())
            .max(if dev_freeze.is_some() {
                freeze_sw_idx + 1
            } else {
                0
            });
        let mut sw_scratch: Vec<u8> = (0..n_sw)
            .map(|k| {
                self.switches
                    .get(k)
                    .copied()
                    .or_else(|| meta.switches.get(k).map(|s| s.default_index))
                    .unwrap_or(0)
            })
            .collect();
        let base_freeze_sw = sw_scratch.get(freeze_sw_idx).copied().unwrap_or(0);
        let n_in = self.inner.num_inputs().clamp(1, 4);
        let n_out = self.inner.num_outputs().clamp(1, 4);
        let n_ctrl = meta
            .controls
            .len()
            .max(control_values.len())
            .max(dev_trajs.map_or(0, Vec::len));
        let mut ctrl_scratch = vec![0.0_f32; n_ctrl];
        let mut dev_out1 = vec![0.0_f32; dev_n];
        let mut dev_out2 = (n_out >= 2).then(|| vec![0.0_f32; dev_n]);
        let mut dev_out3 = (n_out >= 3).then(|| vec![0.0_f32; dev_n]);
        let mut dev_out4 = (n_out >= 4).then(|| vec![0.0_f32; dev_n]);
        let sidechain_uses_in3 = dev_in2.is_none() && dev_in3.is_some();

        for i in 0..dev_n {
            let x1 = dev_in1[i];
            let x2 = if sidechain_uses_in3 {
                dev_in3.and_then(|v| v.get(i).copied()).unwrap_or(x1)
            } else {
                dev_in2.and_then(|v| v.get(i).copied()).unwrap_or(x1)
            };
            let x3 = dev_in3.and_then(|v| v.get(i).copied()).unwrap_or(0.0);
            let x4 = dev_in4.and_then(|v| v.get(i).copied()).unwrap_or(0.0);
            for p in 0..n_ctrl {
                let fb = control_values
                    .get(p)
                    .copied()
                    .or_else(|| default_controls.get(p).copied())
                    .unwrap_or(0.0);
                ctrl_scratch[p] = dev_trajs
                    .and_then(|t| t.get(p))
                    .and_then(|t| t.get(i).copied())
                    .unwrap_or(fb);
            }
            if let Some([fs, fe]) = dev_freeze
                && let Some(slot) = sw_scratch.get_mut(freeze_sw_idx)
            {
                *slot = if i >= fs && i < fe { 1 } else { base_freeze_sw };
            }
            let in_f = [x1, x2, x3, x4];
            let mut y = [0.0_f32; 4];
            self.inner
                .process_frame(&in_f[..n_in], &mut y[..n_out], &ctrl_scratch, &sw_scratch);
            dev_out1[i] = y[0];
            if let Some(ref mut c2) = dev_out2 {
                c2[i] = y[1];
            }
            if let Some(ref mut c3) = dev_out3 {
                c3[i] = y[2];
            }
            if let Some(ref mut c4) = dev_out4 {
                c4[i] = y[3];
            }
        }
        let (out_1, out_2, out_3, out_4) = if needs_resample {
            (
                resample_arbitrary_rate(&dev_out1, dev_sr, stim_sr, Some(orig_n)),
                dev_out2.map(|c| resample_arbitrary_rate(&c, dev_sr, stim_sr, Some(orig_n))),
                dev_out3.map(|c| resample_arbitrary_rate(&c, dev_sr, stim_sr, Some(orig_n))),
                dev_out4.map(|c| resample_arbitrary_rate(&c, dev_sr, stim_sr, Some(orig_n))),
            )
        } else {
            (dev_out1, dev_out2, dev_out3, dev_out4)
        };
        Ok(StemResponse {
            stem_id,
            sample_rate_hz: stim_sr,
            out_1,
            out_2,
            out_3,
            out_4,
            internal_nodes: BTreeMap::new(),
        })
    }
}

impl<M: BlackBoxModel> DistillationTarget for BlackBoxDistillationAdapter<M> {
    fn reset_state(&mut self) {
        self.inner.reset();
    }
    fn render_stem(&mut self, s: &StemStimulus) -> Result<StemResponse, DnamError> {
        self.reset_state();
        self.render_internal(
            s.stem_id.clone(),
            s.sample_rate_hz,
            &s.in_1,
            s.in_2.as_deref(),
            s.in_3.as_deref(),
            s.in_4.as_deref(),
            &s.control_values,
            s.control_trajectories.as_ref(),
            s.freeze_transition_window,
        )
    }
    fn render_frame(&mut self, f: &StimulusFrame) -> Result<StemResponse, DnamError> {
        self.render_internal(
            format!("frame_{}", f.frame_index),
            f.sample_rate_hz,
            &f.in_1,
            f.in_2.as_deref(),
            f.in_3.as_deref(),
            f.in_4.as_deref(),
            &f.control_values,
            f.control_trajectories.as_ref(),
            None,
        )
    }
}

fn manifest_for_fixture<M: BlackBoxModel>(model: &M) -> DnamProjManifest {
    let meta = model.metadata();
    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.metadata.name = meta.name.to_string();
    manifest.base.io_config.output_mode = if model.num_outputs() >= 2 {
        "stereo".to_string()
    } else {
        "mono".to_string()
    };
    let mut schema = Vec::new();
    for c in meta.controls.iter().take(2) {
        let def: ControlSchemaDef = serde_json::from_value(serde_json::json!({
            "id": c.id,
            "name": c.name,
            "type": "range",
            "lobatto_degree": 2,
            "declared_range": [-1.0, 1.0],
            "safe_clamped_range": [-1.0, 1.0],
            "default": c.default_norm
        }))
        .unwrap();
        schema.push(def);
    }
    for sw in meta.switches.iter().take(1) {
        let def: ControlSchemaDef = serde_json::from_value(serde_json::json!({
            "id": sw.id,
            "name": sw.name,
            "type": "switch",
            "declared_options": sw.options,
            "active_options": sw.options,
            "default": sw.options[usize::from(sw.default_index)]
        }))
        .unwrap();
        schema.push(def);
    }
    manifest.base.controls_schema = schema;
    manifest
}

fn read_f32_le(buf: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

fn write_f32_le(buf: &mut [u8], off: usize, v: f32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

fn verify_chain_distillation<M: BlackBoxModel>(model: M, run_null_test: bool) -> Vec<u8> {
    let crypto = DefaultCryptoProvider::default();
    let chain_id = model.metadata().id;
    let n_in = model.num_inputs();
    let n_out = model.num_outputs();
    let topo = match (n_in >= 2, n_out >= 2) {
        (true, true) => TopologyProbeMode::TrueStereo2To2,
        (true, false) => TopologyProbeMode::DualInToMono2To1,
        (false, true) => TopologyProbeMode::MonoToStereo1To2,
        (false, false) => TopologyProbeMode::Mono1To1,
    };
    let manifest = manifest_for_fixture(&model);
    let mut adapter = BlackBoxDistillationAdapter::from_model(model);
    let opts = AdaptiveTrainingOptions {
        sample_rate_hz: 48_000,
        topology_mode: topo,
        fixed_duration: true,
        prefer_mode_b: true,
        max_frames_per_stem: 3,
        run_live_null_test: run_null_test,
        null_test_autopatch_threshold_db: -42.0,
        calb: None,
    };
    let report = train_adaptive_target(manifest, &mut adapter, &opts)
        .unwrap_or_else(|e| panic!("{chain_id} train_adaptive_target failed: {e:?}"));
    if run_null_test {
        assert!(
            report.null_test.is_some(),
            "{chain_id} must produce LiveNullTestReport"
        );
    }

    let proj_view = DnamProjView::from_bytes(&report.proj_bytes, &crypto)
        .unwrap_or_else(|e| panic!("{chain_id} invalid .dnam-proj: {e:?}"));
    let stat = proj_view
        .chunk_slice(ProjChunkKind::Stat, 0)
        .expect("STAT chunk must exist");
    assert_eq!(
        stat.len(),
        PROJ_STAT_BYTES_ULTRA,
        "{chain_id} STAT must be 267,264 B"
    );

    let telem = proj_view
        .manifest
        .solver_telemetry
        .as_ref()
        .expect("solver_telemetry must be populated");
    assert!(
        telem.tier_esr_db[3] <= telem.tier_esr_db[2] + 1.0e-3
            && telem.tier_esr_db[2] <= telem.tier_esr_db[1] + 1.0e-3
            && telem.tier_esr_db[1] <= telem.tier_esr_db[0] + 1.0e-3,
        "{chain_id} tier_esr_db must be monotone: {:?}",
        telem.tier_esr_db
    );
    assert!(
        telem.tier_esr_db[3] < -0.5,
        "{chain_id} Ultra tier_esr_db must beat -0.5 dB across all chains, got {:?}",
        telem.tier_esr_db
    );
    if matches!(
        chain_id,
        "Chain-01-Symmetric-OD"
            | "Chain-03-Parallel-Clean-Blend"
            | "Chain-05-Pure-LTI-Cab-And-Room"
    ) {
        assert!(
            telem.tier_esr_db[3] < -10.0,
            "{chain_id} stationary LTI/drive Ultra tier_esr_db must be < -10.0 dB, got {:?}",
            telem.tier_esr_db
        );
    }
    let audit = proj_view
        .manifest
        .audit_report
        .as_ref()
        .expect("audit_report must be populated");
    assert!(
        audit.prescriptions.len() <= 64,
        "{chain_id} audit report valid"
    );

    let rt_view = DnamContainerView::from_bytes(&report.runtime_dnam_bytes, &crypto)
        .unwrap_or_else(|e| panic!("{chain_id} invalid runtime .dnam: {e:?}"));
    let arena = rt_view.arena_storage.as_bytes();
    for off in (9_024..9_064).step_by(4).chain((63_424..63_456).step_by(4)) {
        assert_eq!(
            read_f32_le(arena, off),
            0.0,
            "{chain_id} b_head at {off} must be 0.0"
        );
    }

    // Silence test ([STAB-ZERO-DC]) + broadband 4-tier loudness parity (<= 0.05 LU)
    let mut tier_lufs = [0.0_f32; 4];
    for (t_idx, tier) in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra]
        .into_iter()
        .enumerate()
    {
        let tier_dnam = export_runtime_dnam(&report.proj_bytes, Some(tier)).unwrap();
        let mut eng =
            DnamEngine::<DefaultMathBackend>::from_bytes_owned(&tier_dnam, tier, 48_000).unwrap();
        for _ in 0..256 {
            let [yl, yr] = eng.step_frame_tora(0.0, 0.0, false);
            assert!(
                yl.abs() <= 1.0e-7 && yr.abs() <= 1.0e-7,
                "{chain_id} {tier:?} non-zero DC on silence: [{yl}, {yr}]"
            );
        }
        eng.reset_state();
        let mut rendered = Vec::with_capacity(2_048);
        for n in 0..2_048 {
            let t = (n as f32) / 48_000.0;
            let x = 0.25 * (std::f32::consts::TAU * 220.0 * t).sin()
                + 0.15 * (std::f32::consts::TAU * 880.0 * t).cos();
            let y = eng.step_frame_tora(x, x, false);
            assert!(
                y[0].is_finite() && y[1].is_finite() && y[0].abs() < 16.0 && y[1].abs() < 16.0,
                "{chain_id} {tier:?} unbounded output: {y:?}"
            );
            if n >= 256 {
                rendered.push(y);
            }
        }
        tier_lufs[t_idx] = compute_integrated_lufs(&rendered);
    }

    // Verify LIVE DnamEngine StarCenter 4-tier loudness achieves <= 0.05 LU parity across Lite..Ultra
    let inpc_view =
        verify_proj_inpc_capsule(proj_view.chunk_slice(ProjChunkKind::Inpc, 0).unwrap()).unwrap();
    let decode_fp16 = |bytes: &[u8]| -> Vec<f32> {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| fp16_bits_to_f32(u16::from_le_bytes(*b)))
            .collect()
    };
    let inpc_vals = decode_fp16(inpc_view.dry_coreset_fp16);
    let inpc_ch = usize::from(inpc_view.channels).max(1);
    let dry_coreset_lr: Vec<[f32; 2]> = inpc_vals
        .chunks_exact(inpc_ch)
        .map(|c| {
            if inpc_ch >= 2 {
                [c[0], c[1]]
            } else {
                [c[0], c[0]]
            }
        })
        .collect();
    let star_entry = proj_view
        .manifest
        .stems_registry
        .iter()
        .find(|s| s.enabled && s.is_star_center)
        .unwrap();
    let stmc_view = verify_proj_stmc_capsule(
        proj_view
            .chunk_slice(ProjChunkKind::Stmc, star_entry.chunk_id)
            .unwrap(),
    )
    .unwrap();
    let stmc_vals = decode_fp16(stmc_view.wet_coreset_fp16);
    let stmc_ch = if (stmc_view.flags & STMC_FLAG_STEREO) != 0 {
        2
    } else {
        1
    };
    let wet_coreset_lr: Vec<[f32; 2]> = stmc_vals
        .chunks_exact(stmc_ch)
        .map(|c| {
            if stmc_ch >= 2 {
                [c[0], c[1]]
            } else {
                [c[0], c[0]]
            }
        })
        .collect();
    let n_common = dry_coreset_lr.len().min(wet_coreset_lr.len());
    let warmup = canonical_besr_warmup(n_common);
    let star_ctrls = star_center_control_values(&proj_view.manifest);
    let mono_only = proj_view.manifest.base.io_config.output_mode == "mono";
    let tgt_l_f64: Vec<f64> = wet_coreset_lr[..n_common]
        .iter()
        .map(|v| f64::from(v[0]))
        .collect();
    let tgt_r_f64: Vec<f64> = wet_coreset_lr[..n_common]
        .iter()
        .map(|v| f64::from(v[1]))
        .collect();
    let mut live_coreset_lufs = [0.0_f32; 4];
    let mut live_tier_besr_db = [0.0_f64; 4];
    let mut live_tier_bands_db = [[0.0_f64; 3]; 4];
    for (t_idx, tier) in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra]
        .into_iter()
        .enumerate()
    {
        let tier_dnam = export_runtime_dnam(&report.proj_bytes, Some(tier)).unwrap();
        let mut eng =
            DnamEngine::<DefaultMathBackend>::from_bytes_owned(&tier_dnam, tier, 48_000).unwrap();
        eng.reset_to_dc_equilibrium();
        for (p, &val) in star_ctrls.iter().enumerate() {
            eng.set_control_immediate(p, val);
        }
        let rendered_all: Vec<[f32; 2]> = dry_coreset_lr[..n_common]
            .iter()
            .map(|&[dl, dr]| eng.step_frame_tora(dl, dr, mono_only))
            .collect();
        live_coreset_lufs[t_idx] = compute_integrated_lufs(&rendered_all[warmup..n_common]);

        let mdl_l: Vec<f64> = rendered_all.iter().map(|v| f64::from(v[0])).collect();
        let (sl, el, nl) = split_bezout_3band_energies(&tgt_l_f64, &mdl_l, None, 48_000.0, warmup);
        let rep_l = evaluate_besr_from_band_energies(sl, el, nl);
        assert!(rep_l.valid, "{chain_id} {tier:?} Left 3-band bESR invalid");
        if mono_only {
            live_tier_besr_db[t_idx] = rep_l.besr_db;
            live_tier_bands_db[t_idx] = rep_l.band_esr_db;
        } else {
            let mdl_r: Vec<f64> = rendered_all.iter().map(|v| f64::from(v[1])).collect();
            let (sr, er, nr) =
                split_bezout_3band_energies(&tgt_r_f64, &mdl_r, None, 48_000.0, warmup);
            let rep_r = evaluate_besr_from_band_energies(sr, er, nr);
            assert!(rep_r.valid, "{chain_id} {tier:?} Right 3-band bESR invalid");
            live_tier_besr_db[t_idx] = rep_l.besr_db.max(rep_r.besr_db);
            live_tier_bands_db[t_idx] = [
                rep_l.band_esr_db[0].max(rep_r.band_esr_db[0]),
                rep_l.band_esr_db[1].max(rep_r.band_esr_db[1]),
                rep_l.band_esr_db[2].max(rep_r.band_esr_db[2]),
            ];
        }
    }
    for t_idx in 0..4 {
        assert!(
            live_tier_besr_db[t_idx] < 0.0 && telem.tier_esr_db[t_idx] < 0.0,
            "{chain_id} tier {t_idx} 3-band bESR must be strictly negative: live={live_tier_besr_db:?}, telem={:?}",
            telem.tier_esr_db
        );
        assert!(
            (f64::from(telem.tier_esr_db[t_idx]) - live_tier_besr_db[t_idx]).abs() <= 0.05,
            "{chain_id} tier {t_idx} telemetry ({}) drifted from live 12k-coreset 3-band bESR ({})",
            telem.tier_esr_db[t_idx],
            live_tier_besr_db[t_idx]
        );
    }
    assert!(
        live_tier_besr_db[3] <= live_tier_besr_db[2] + 1.0e-3
            && live_tier_besr_db[2] <= live_tier_besr_db[1] + 1.0e-3
            && live_tier_besr_db[1] <= live_tier_besr_db[0] + 1.0e-3,
        "{chain_id} live 12k-coreset 3-band bESR must be monotone across [Lite, Full, Pro, Ultra]: {live_tier_besr_db:?} (bands={live_tier_bands_db:?})"
    );
    assert!(
        live_tier_besr_db[3] < -0.5,
        "{chain_id} live Ultra 3-band bESR must be negative (< -0.5 dB), got {live_tier_besr_db:?}"
    );
    for (b_idx, &band_db) in live_tier_bands_db[3].iter().enumerate() {
        assert!(
            band_db < 0.0,
            "{chain_id} Ultra band {b_idx} (0=LF,1=MF,2=HF) must be strictly negative, got {:?}",
            live_tier_bands_db[3]
        );
    }
    if chain_id == "Chain-05-Pure-LTI-Cab-And-Room" {
        for (b_idx, &band_db) in live_tier_bands_db[3].iter().enumerate() {
            assert!(
                band_db < -10.0,
                "Chain-05 pure LTI cab Ultra band {b_idx} must achieve < -10.0 dB across LF/MF/HF, got {:?}",
                live_tier_bands_db[3]
            );
        }
    }
    let tiers_hdr = &proj_view.manifest.base.slimmable_tiers;
    assert!(
        tiers_hdr.ultra.loudness_lufs > -55.0 && live_coreset_lufs[3] > -55.0,
        "{chain_id} distilled model must be non-silent (hdr={} LUFS, live={} LUFS)",
        tiers_hdr.ultra.loudness_lufs,
        live_coreset_lufs[3]
    );
    for (t_idx, &live_lu) in live_coreset_lufs.iter().enumerate() {
        assert!(
            (live_lu - live_coreset_lufs[3]).abs() <= 0.05,
            "{chain_id} tier {t_idx} LIVE DnamEngine loudness {live_lu} LUFS exceeds 0.05 LU vs Ultra {} (all={live_coreset_lufs:?}, broadband={tier_lufs:?})",
            live_coreset_lufs[3]
        );
    }
    for trim_db in [
        tiers_hdr.lite.output_trim_db,
        tiers_hdr.full.output_trim_db,
        tiers_hdr.pro.output_trim_db,
        tiers_hdr.ultra.output_trim_db,
    ] {
        assert!(
            trim_db.is_finite() && (-12.0..=12.0).contains(&trim_db),
            "{chain_id} invalid output_trim_db {trim_db}"
        );
    }
    println!(
        "[CERT-12K] {chain_id}: besr={live_tier_besr_db:.3?} ultra_bands={:.3?} lufs={live_coreset_lufs:.3?}",
        live_tier_bands_db[3]
    );

    report.proj_bytes
}

#[test]
fn test_all_12_physical_signal_chains_end_to_end_varpro_distillation() {
    // verify[train.pipeline.stage-a-poles]
    // verify[train.pipeline.fixtures-matrix]
    let _ = verify_chain_distillation(Chain01SymmetricOd::new(), true);
    let _ = verify_chain_distillation(Chain02GatedAsymmetricFuzz::new(), true);
    let _ = verify_chain_distillation(Chain03ParallelCleanBlend::new(), true);
    let _ = verify_chain_distillation(Chain04TubeAmpFmv4x12::new(), true);
    let _ = verify_chain_distillation(Chain05PureLtiCabAndRoom::new(), true);
    let _ = verify_chain_distillation(Chain06EnvelopeResonantSvf::new(), true);
    let _ = verify_chain_distillation(Chain07StereoChorusAndRotary::new(), true);
    let _ = verify_chain_distillation(Chain08LongBbdEchoAnd30sShimmer::new(), true);
    let _ = verify_chain_distillation(Chain09La2aOpticalCompressor::new(), true);
    let _ = verify_chain_distillation(Chain10Oc2OctaveDividerAndRingMod::new(), true);
    let _ = verify_chain_distillation(Chain11BiAmpBassAndOutputIron::new(), true);
    let _ = verify_chain_distillation(
        Chain12FullEndToEndStereoRig::new(Chain12RigMode::Parametric16),
        true,
    );
}

#[test]
fn test_case_08_three_layer_buyer_watermark_collusion_and_blackbox_phase_id() {
    // verify[train.pipeline.stage-e-pack]
    let crypto = DefaultCryptoProvider::default();
    let proj_bytes = verify_chain_distillation(Chain04TubeAmpFmv4x12::new(), false);
    let master_dnam = export_runtime_dnam(&proj_bytes, Some(Tier::Ultra)).unwrap();
    let master_view = DnamContainerView::from_bytes(&master_dnam, &crypto).unwrap();
    let master_arena = master_view.arena_storage.as_bytes().to_vec();
    let master_hash = extract_structural_hash_224d(&master_arena).unwrap();

    let mk_cfg = |buyer_idx: u8| {
        let mut buyer_id = [0u8; 16];
        buyer_id[0] = buyer_idx;
        buyer_id[15] = buyer_idx.wrapping_mul(37).wrapping_add(11);
        BuyerWatermarkConfig::new(
            [0xA5; 32],
            "chain04_tube_amp_v2",
            format!("lic_2026_buyer_{buyer_idx:02}"),
            buyer_id,
        )
    };

    // Export 5 colluding buyer copies + 1 innocent buyer codebook/seed
    let buyer_cfgs: Vec<BuyerWatermarkConfig> = (1..=5).map(mk_cfg).collect();
    let innocent_cfg = mk_cfg(99);
    let innocent_code = derive_buyer_tardos_codebook(&innocent_cfg);
    let innocent_phase_seed = derive_buyer_phase_seed(&innocent_cfg);

    let mut buyer_dnams = Vec::with_capacity(5);
    let mut buyer_arenas = Vec::with_capacity(5);
    let mut buyer_codes = Vec::with_capacity(5);

    for cfg in &buyer_cfgs {
        let code = derive_buyer_tardos_codebook(cfg);
        // Role A (64 floats) must be 0.0; 7,616 Role B chips must be +-1.0
        let mut role_a_zeros = 0usize;
        let mut role_b_pm1 = 0usize;
        for idx in 0..TARDOS_CASCADE_WEIGHTS {
            if is_role_a_cascade_index(idx) {
                assert_eq!(
                    code.chips[idx], 0.0,
                    "Role A chip {idx} must be untouched (0.0)"
                );
                role_a_zeros += 1;
            } else {
                assert!(
                    (code.chips[idx].abs() - 1.0).abs() < 1.0e-6,
                    "Role B chip {idx} must be +-1.0"
                );
                role_b_pm1 += 1;
            }
        }
        assert_eq!(role_a_zeros, 64);
        assert_eq!(role_b_pm1, 7_616);

        let wm_dnam = export_runtime_dnam_watermarked(&proj_bytes, cfg, Some(Tier::Ultra)).unwrap();
        let wm_view = DnamContainerView::from_bytes(&wm_dnam, &crypto).unwrap();
        let wm_arena = wm_view.arena_storage.as_bytes().to_vec();

        // 224-D gauge-invariant structural hash similarity > 0.9999 vs master
        let wm_hash = extract_structural_hash_224d(&wm_arena).unwrap();
        let cos_sim = structural_cosine_similarity(&master_hash, &wm_hash);
        assert!(
            cos_sim > 0.9999,
            "224-D structural hash cosine similarity must exceed 0.9999, got {cos_sim:.8}"
        );

        // Layer-2 white-box arena phase coding Z-score > 4.0 (innocent < 2.5)
        let phase_seed = derive_buyer_phase_seed(cfg);
        let z_phase_wb = score_phase_coding_arena(&master_arena, &wm_arena, &phase_seed).unwrap();
        let z_phase_inn =
            score_phase_coding_arena(&master_arena, &wm_arena, &innocent_phase_seed).unwrap();
        assert!(
            z_phase_wb > 4.0,
            "Layer-2 white-box Z must beat 4.0, got {z_phase_wb:.2}"
        );
        assert!(
            z_phase_inn.abs() < 3.0,
            "Innocent phase Z too high: {z_phase_inn:.2}"
        );

        buyer_dnams.push(wm_dnam);
        buyer_arenas.push(wm_arena);
        buyer_codes.push(code);
    }

    // Perceptual audio transparency between master and Buyer #1 watermarked copy
    let mut eng_m =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&master_dnam, Tier::Ultra, 48_000)
            .unwrap();
    let mut eng_w =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&buyer_dnams[0], Tier::Ultra, 48_000)
            .unwrap();
    let mut max_diff = 0.0_f32;
    for n in 0..1_024 {
        let x = 0.25 * (std::f32::consts::TAU * 330.0 * (n as f32) / 48_000.0).sin();
        let ym = eng_m.step_frame_tora(x, x, false);
        let yw = eng_w.step_frame_tora(x, x, false);
        max_diff = max_diff.max((ym[0] - yw[0]).abs().max((ym[1] - yw[1]).abs()));
    }
    assert!(
        max_diff < 2.0e-2,
        "Watermarked copy must remain perceptually transparent: {max_diff}"
    );

    // N = 5 Collusion Attacks: (1) Linear Average, (2) Coordinate Median, (3) 64B Block Interleave
    let arena_len = master_arena.len();
    let mut colluded_avg = master_arena.clone();
    let mut colluded_med = master_arena.clone();
    let mut colluded_int = master_arena.clone();

    for off in (0..arena_len).step_by(4) {
        let vals: [f32; 5] = std::array::from_fn(|b| read_f32_le(&buyer_arenas[b], off));
        if vals.iter().all(|v| v.is_finite()) {
            let mean = vals.iter().sum::<f32>() / 5.0;
            write_f32_le(&mut colluded_avg, off, mean);
            let mut sorted = vals;
            sorted.sort_by(|a, b| a.total_cmp(b));
            write_f32_le(&mut colluded_med, off, sorted[2]);
        }
    }
    for blk in 0..(arena_len / 64) {
        let src = &buyer_arenas[blk % 5][blk * 64..(blk + 1) * 64];
        colluded_int[blk * 64..(blk + 1) * 64].copy_from_slice(src);
    }

    for (attack_name, suspect) in [
        ("linear_average", &colluded_avg),
        ("coordinate_median", &colluded_med),
        ("block_interleave", &colluded_int),
    ] {
        let max_colluder_z = buyer_codes
            .iter()
            .map(|c| {
                score_tardos_watermark_7680(&master_arena, suspect, &c.accusation_weights).unwrap()
            })
            .fold(f64::NEG_INFINITY, f64::max);
        let innocent_z =
            score_tardos_watermark_7680(&master_arena, suspect, &innocent_code.accusation_weights)
                .unwrap();
        assert!(
            max_colluder_z > 8.0,
            "{attack_name}: N=5 Tardos max colluder Z must exceed 8.0, got {max_colluder_z:.2}"
        );
        assert!(
            innocent_z.abs() < 3.5,
            "{attack_name}: innocent buyer |Z| must stay < 3.5, got {innocent_z:.2}"
        );
    }

    // Black-box audio-only Layer-2 pole-angle phase identification (no arena inspection of suspect!)
    let suspect_probe_audio = render_blackbox_probe_response(&buyer_dnams[0], 4_096).unwrap();
    let master_angles_gt = extract_arena_pole_angles_34(&master_arena);
    assert!(master_angles_gt.len() >= 32);
    let (master_angles, est_suspect_angles) =
        identify_pole_angles_blackbox_from_audio(&master_dnam, &suspect_probe_audio).unwrap();
    let buyer0_seed = derive_buyer_phase_seed(&buyer_cfgs[0]);
    let z_bb_guilty =
        score_phase_coding_blackbox(&master_angles, &est_suspect_angles, &buyer0_seed).unwrap();
    let z_bb_innocent =
        score_phase_coding_blackbox(&master_angles, &est_suspect_angles, &innocent_phase_seed)
            .unwrap();
    assert!(
        z_bb_guilty > 4.0,
        "Black-box audio-only pole-angle phase Z must exceed 4.0, got {z_bb_guilty:.2}"
    );
    assert!(
        z_bb_innocent.abs() < 3.0,
        "Black-box innocent buyer |Z| must stay < 3.0, got {z_bb_innocent:.2}"
    );
}
