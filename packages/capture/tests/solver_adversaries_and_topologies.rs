#![forbid(unsafe_code)]
//! Step 2.3 End-to-End Certification of All 14 Hardware Fault Decorators (`dnam_fixtures::adversaries`),
//! `Case #9` Multi-Rate ZOH + `4x` Bounce + Inter-Tier Loudness Parity (`<= 0.05 LU`),
//! `Case #11` Polarity/DC Offset + Extreme Overdrive Numerical Envelope (`±100`, `±1e30`, `NaN`, `±Inf`),
//! and `Case #19` All Six Routing Topologies (`1->1`, `2->1`, `1->2`, `2->2`, `4CM`, `WDW`).
//!
//! Canonical References:
//! - `docs/specs/training/four_stage_pipeline.md` (`r[train.pipeline.fixtures-matrix]`)
//! - `docs/specs/training/control_space_audit.md` (`[STAB-ZERO-DC]`, 3-phase audit)
//! - `docs/specs/dsp/router_and_stages_topology.md` (`S_{ret->send} == 0` causal `4CM` DAG)

use std::collections::BTreeMap;

use dnam_capture::fp16_bits_to_f32;
use dnam_capture::proj::{
    DnamProjManifest, DnamProjView, PROJ_STAT_BYTES_ULTRA, ProjChunkKind, STMC_FLAG_STEREO,
    export_runtime_dnam, verify_proj_inpc_capsule, verify_proj_stmc_capsule,
};
use dnam_capture::solver::{
    canonical_besr_warmup, compute_integrated_lufs, star_center_control_values,
};
use dnam_capture::stimulus::{
    AdaptiveTrainingOptions, BatchStimulusOptions, TopologyProbeMode, generate_stimulus_batch,
    solve_from_responses, train_adaptive_target,
};
use dnam_capture::{
    DistillationTarget, StemResponse, StemStimulus, StimulusFrame, resample_arbitrary_rate,
    resample_control_trajectory, resampled_frame_len, scale_freeze_window,
};
use dnam_engine::container::DnamContainerView;
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::schema::{ControlSchemaDef, DnamError};
use dnam_engine::{DnamAudioIo, DnamEngine, Tier};
use dnam_fixtures::{
    AdcClipAndPadDecorator, BlackBoxModel, Chain01SymmetricOd, Chain02GatedAsymmetricFuzz,
    Chain03ParallelCleanBlend, Chain04TubeAmpFmv4x12, Chain05PureLtiCabAndRoom,
    Chain07StereoChorusAndRotary, Chain08LongBbdEchoAnd30sShimmer, Chain09La2aOpticalCompressor,
    ConverterLoopbackDecorator, DigitalQuirksDecorator, FixtureBitDepth, FixtureDeviceMetadata,
    GermaniumTempDriftDecorator, HardwareNoiseGateDecorator, HumanHandTrajectoryDecorator,
    IdleSquealDecorator, MainsHumAndNoiseFloorDecorator, MicrophoneCapsuleClipDecorator,
    PolarityAndDcOffsetDecorator, PotTaperAndSwitchDecorator,
    ScratchyPotAndSteppedAttenuatorDecorator, SpringChirpDecorator, TaperCurveKind,
    TubeThermalHysteresisDecorator,
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

fn manifest_for_fixture<M: BlackBoxModel>(model: &M, max_ctrls: usize) -> DnamProjManifest {
    let meta = model.metadata();
    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.metadata.name = meta.name.to_string();
    manifest.base.io_config.inputs = (model.num_inputs() as u8).clamp(1, 4);
    manifest.base.io_config.output_mode = if model.num_outputs() >= 2 {
        "stereo".to_string()
    } else {
        "mono".to_string()
    };
    let mut schema = Vec::new();
    for c in meta.controls.iter().take(max_ctrls) {
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
    manifest.base.controls_schema = schema;
    manifest
}

fn read_f32_le(buf: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

fn verify_decorator_resilience<M: BlackBoxModel>(label: &str, model: M) -> Vec<u8> {
    let crypto = DefaultCryptoProvider::default();
    let topo = match (model.num_inputs() >= 2, model.num_outputs() >= 2) {
        (true, true) => TopologyProbeMode::TrueStereo2To2,
        (true, false) => TopologyProbeMode::DualInToMono2To1,
        (false, true) => TopologyProbeMode::MonoToStereo1To2,
        (false, false) => TopologyProbeMode::Mono1To1,
    };
    let manifest = manifest_for_fixture(&model, 1);
    let mut adapter = BlackBoxDistillationAdapter::from_model(model);
    let report = train_adaptive_target(
        manifest,
        &mut adapter,
        &AdaptiveTrainingOptions {
            sample_rate_hz: 48_000,
            topology_mode: topo,
            fixed_duration: true,
            prefer_mode_b: false,
            max_frames_per_stem: 3,
            run_live_null_test: false,
            null_test_autopatch_threshold_db: -36.0,
            calb: None,
        },
    )
    .unwrap_or_else(|e| panic!("decorator '{label}' distillation failed: {e:?}"));

    let proj_view = DnamProjView::from_bytes(&report.proj_bytes, &crypto).unwrap();
    assert_eq!(
        proj_view.chunk_slice(ProjChunkKind::Stat, 0).unwrap().len(),
        PROJ_STAT_BYTES_ULTRA
    );
    assert!(proj_view.manifest.audit_report.is_some());

    let rt_view = DnamContainerView::from_bytes(&report.runtime_dnam_bytes, &crypto).unwrap();
    let arena = rt_view.arena_storage.as_bytes();
    for off in (9_024..9_064).step_by(4).chain((63_424..63_456).step_by(4)) {
        assert_eq!(read_f32_le(arena, off), 0.0, "{label}: b_head must be 0.0");
    }

    let mut eng = DnamEngine::<DefaultMathBackend>::from_bytes_owned(
        &report.runtime_dnam_bytes,
        Tier::Ultra,
        48_000,
    )
    .unwrap();
    for _ in 0..128 {
        let [yl, yr] = eng.step_frame_tora(0.0, 0.0, false);
        assert!(
            yl.abs() <= 1.0e-7 && yr.abs() <= 1.0e-7,
            "{label}: DC leak on silence"
        );
    }
    for n in 0..256 {
        let x = 0.3 * (std::f32::consts::TAU * 330.0 * (n as f32) / 48_000.0).sin();
        let [yl, yr] = eng.step_frame_tora(x, x, false);
        assert!(yl.is_finite() && yr.is_finite() && yl.abs() < 16.0 && yr.abs() < 16.0);
    }
    report.proj_bytes
}

#[test]
fn test_all_14_hardware_fault_decorators_end_to_end_solver_resilience() {
    // verify[train.pipeline.fixtures-matrix]
    let _ = verify_decorator_resilience(
        "01_tube_thermal_hysteresis",
        TubeThermalHysteresisDecorator::new(Chain04TubeAmpFmv4x12::new()),
    );
    let _ = verify_decorator_resilience(
        "02_germanium_temp_drift",
        GermaniumTempDriftDecorator::new(Chain02GatedAsymmetricFuzz::new(), 36.0),
    );
    let _ = verify_decorator_resilience(
        "03_mains_hum_and_noise_floor",
        MainsHumAndNoiseFloorDecorator::new(Chain01SymmetricOd::new(), 60.0),
    );
    let _ = verify_decorator_resilience(
        "04_hardware_noise_gate",
        HardwareNoiseGateDecorator::new(Chain04TubeAmpFmv4x12::new()),
    );
    let _ = verify_decorator_resilience(
        "05_spring_chirp",
        SpringChirpDecorator::new(Chain05PureLtiCabAndRoom::new()),
    );
    let _ = verify_decorator_resilience(
        "06_digital_quirks",
        DigitalQuirksDecorator::new(Chain08LongBbdEchoAnd30sShimmer::new()),
    );
    let _ = verify_decorator_resilience(
        "07_scratchy_pot_and_stepped_attenuator",
        ScratchyPotAndSteppedAttenuatorDecorator::new(Chain01SymmetricOd::new()),
    );
    let _ = verify_decorator_resilience(
        "08_idle_squeal",
        IdleSquealDecorator::new(Chain02GatedAsymmetricFuzz::new()),
    );
    let _ = verify_decorator_resilience(
        "09_microphone_capsule_clip",
        MicrophoneCapsuleClipDecorator::new(Chain04TubeAmpFmv4x12::new()),
    );
    let _ = verify_decorator_resilience(
        "10_adc_clip_and_pad",
        AdcClipAndPadDecorator::new(Chain01SymmetricOd::new(), -12.0),
    );
    let _ = verify_decorator_resilience(
        "11_converter_loopback",
        ConverterLoopbackDecorator::new(Chain03ParallelCleanBlend::new()),
    );
    let _ = verify_decorator_resilience(
        "12_human_hand_trajectory",
        HumanHandTrajectoryDecorator::new(Chain01SymmetricOd::new(), 240, 0.08),
    );
    let _ = verify_decorator_resilience(
        "13_pot_taper_and_switch",
        PotTaperAndSwitchDecorator::new(
            Chain01SymmetricOd::new(),
            vec![
                TaperCurveKind::Log15A,
                TaperCurveKind::RevLog85C,
                TaperCurveKind::WTaper,
            ],
        ),
    );
    let _ = verify_decorator_resilience(
        "14_polarity_and_dc_offset",
        PolarityAndDcOffsetDecorator::new(Chain02GatedAsymmetricFuzz::new(), true, 0.08),
    );
}

#[test]
fn test_case_09_multirate_zoh_4x_bounce_and_intertier_loudness_parity() {
    let crypto = DefaultCryptoProvider::default();
    let model = Chain03ParallelCleanBlend::new();
    let manifest = manifest_for_fixture(&model, 1);

    // Distill via Mode 2 Sans-IO batch capture at 44.1 kHz AND 96 kHz
    for capture_sr in [44_100_u32, 96_000] {
        let batch = generate_stimulus_batch(
            &manifest,
            &BatchStimulusOptions {
                sample_rate_hz: capture_sr,
                topology_mode: TopologyProbeMode::Mono1To1,
                prefer_mode_b: true,
                master_seed: 0x2026_0900 + u64::from(capture_sr),
            },
        )
        .unwrap();
        let mut adapter = BlackBoxDistillationAdapter::from_model(model.clone());
        let resps: Vec<StemResponse> = batch
            .stems
            .iter()
            .map(|s| adapter.render_stem(s).unwrap())
            .collect();
        let solved = solve_from_responses(manifest.clone(), &[(&batch, &resps)], None).unwrap();
        let view = DnamProjView::from_bytes(&solved.proj_bytes, &crypto).unwrap();
        let th = &view.manifest.base.slimmable_tiers;
        let inpc_view =
            verify_proj_inpc_capsule(view.chunk_slice(ProjChunkKind::Inpc, 0).unwrap()).unwrap();
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
        let dry_core_lr: Vec<[f32; 2]> = inpc_vals
            .chunks_exact(inpc_ch)
            .map(|c| {
                if inpc_ch >= 2 {
                    [c[0], c[1]]
                } else {
                    [c[0], c[0]]
                }
            })
            .collect();
        let star_entry = view
            .manifest
            .stems_registry
            .iter()
            .find(|s| s.enabled && s.is_star_center)
            .unwrap();
        let stmc_view = verify_proj_stmc_capsule(
            view.chunk_slice(ProjChunkKind::Stmc, star_entry.chunk_id)
                .unwrap(),
        )
        .unwrap();
        let stmc_vals = decode_fp16(stmc_view.wet_coreset_fp16);
        let stmc_ch = if (stmc_view.flags & STMC_FLAG_STEREO) != 0 {
            2
        } else {
            1
        };
        let wet_core_lr: Vec<[f32; 2]> = stmc_vals
            .chunks_exact(stmc_ch)
            .map(|c| {
                if stmc_ch >= 2 {
                    [c[0], c[1]]
                } else {
                    [c[0], c[0]]
                }
            })
            .collect();
        let n_common = dry_core_lr.len().min(wet_core_lr.len());
        let w_start = canonical_besr_warmup(n_common);
        let w_end = n_common;
        let warmup_start = 0usize;
        let star_ctrls = star_center_control_values(&view.manifest);
        let mono_only = view.manifest.base.io_config.output_mode == "mono";
        let mut live_lu_arr = [0.0_f32; 4];
        for (t_idx, tier) in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra]
            .into_iter()
            .enumerate()
        {
            let dnam_t = export_runtime_dnam(&solved.proj_bytes, Some(tier)).unwrap();
            let mut eng =
                DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_t, tier, 48_000).unwrap();
            eng.reset_to_dc_equilibrium();
            for (p, &val) in star_ctrls.iter().enumerate() {
                eng.set_control_immediate(p, val);
            }
            for &[dl, dr] in &dry_core_lr[warmup_start..w_start] {
                let _ = eng.step_frame_tora(dl, dr, mono_only);
            }
            let rendered_core: Vec<[f32; 2]> = dry_core_lr[w_start..w_end]
                .iter()
                .map(|&[dl, dr]| eng.step_frame_tora(dl, dr, mono_only))
                .collect();
            live_lu_arr[t_idx] = compute_integrated_lufs(&rendered_core);
        }
        assert!(
            th.ultra.loudness_lufs > -55.0 && live_lu_arr[3] > -55.0,
            "sr={capture_sr} distilled model must be non-silent (hdr={} LUFS, live={} LUFS)",
            th.ultra.loudness_lufs,
            live_lu_arr[3]
        );
        for (t, &lu) in live_lu_arr.iter().enumerate() {
            assert!(
                (lu - live_lu_arr[3]).abs() <= 0.05,
                "sr={capture_sr} tier {t} LIVE DnamEngine loudness step exceeds 0.05 LU: {live_lu_arr:?}"
            );
        }

        // Execute exported .dnam across 44.1 / 48 / 88.2 / 96 kHz in both 1x live and 4x bounce
        for host_sr in [44_100_u32, 48_000, 88_200, 96_000] {
            for tier in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
                let dnam_t = export_runtime_dnam(&solved.proj_bytes, Some(tier)).unwrap();
                for bounce in [false, true] {
                    let mut eng =
                        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_t, tier, host_sr)
                            .unwrap();
                    eng.set_offline_bounce(bounce);
                    let in_l: Vec<f32> = (0..256)
                        .map(|n| {
                            0.25 * (std::f32::consts::TAU * 440.0 * (n as f32) / (host_sr as f32))
                                .sin()
                        })
                        .collect();
                    let mut out_l = vec![0.0_f32; 256];
                    let mut out_r = vec![0.0_f32; 256];
                    eng.process_block(&in_l, &in_l, &mut out_l, &mut out_r);
                    assert!(
                        out_l
                            .iter()
                            .chain(&out_r)
                            .all(|v| v.is_finite() && v.abs() < 16.0),
                        "non-finite multirate output at host_sr={host_sr}, tier={tier:?}, bounce={bounce}"
                    );
                }
            }
        }
    }
}

#[test]
fn test_case_11_polarity_dc_offset_and_extreme_overdrive_numerical_envelope() {
    let crypto = DefaultCryptoProvider::default();
    let proj_bytes = verify_decorator_resilience(
        "case11_polarity_dc_fuzz",
        PolarityAndDcOffsetDecorator::new(Chain02GatedAsymmetricFuzz::new(), true, 0.08),
    );
    let dnam_ultra = export_runtime_dnam(&proj_bytes, Some(Tier::Ultra)).unwrap();
    let rt_view = DnamContainerView::from_bytes(&dnam_ultra, &crypto).unwrap();
    let arena = rt_view.arena_storage.as_bytes();
    for off in (9_024..9_064).step_by(4).chain((63_424..63_456).step_by(4)) {
        assert_eq!(
            read_f32_le(arena, off),
            0.0,
            "[STAB-ZERO-DC] b_head must be 0.0"
        );
    }

    for (tier, bounce) in [
        (Tier::Lite, false),
        (Tier::Ultra, false),
        (Tier::Ultra, true),
    ] {
        let dnam_t = export_runtime_dnam(&proj_bytes, Some(tier)).unwrap();
        let mut eng =
            DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_t, tier, 48_000).unwrap();
        eng.set_offline_bounce(bounce);

        // 1. Certified +-100.0 (+40 dBFS) overdrive envelope
        let mut overdrive_100 = vec![0.0_f32; 256];
        for (i, slot) in overdrive_100.iter_mut().enumerate() {
            *slot = if i % 2 == 0 { 100.0 } else { -100.0 };
        }
        let mut out_l = vec![0.0_f32; 256];
        let mut out_r = vec![0.0_f32; 256];
        eng.process_block(&overdrive_100, &overdrive_100, &mut out_l, &mut out_r);
        assert!(
            out_l
                .iter()
                .chain(&out_r)
                .all(|v| v.is_finite() && v.abs() <= 1.0e4),
            "non-finite or runaway output under +-100.0 overdrive ({tier:?}, bounce={bounce})"
        );

        // 2. Adversarial +-1.0e30, NaN, +Inf, -Inf safety-ceiling guard (SAFETY_CEILING_F32 = 1.0e9)
        let mut stress_in = vec![0.0_f32; 256];
        for (i, slot) in stress_in.iter_mut().enumerate() {
            *slot = match i % 6 {
                0 => 1.0e30,
                1 => -1.0e30,
                2 => f32::NAN,
                3 => f32::INFINITY,
                4 => f32::NEG_INFINITY,
                _ => 0.5,
            };
        }
        eng.process_block(&stress_in, &stress_in, &mut out_l, &mut out_r);
        assert!(
            out_l
                .iter()
                .chain(&out_r)
                .all(|v| v.is_finite() && v.abs() <= 1.0e9),
            "non-finite or >1e9 output under +-1e30/NaN/Inf stress ({tier:?}, bounce={bounce})"
        );

        // 3. Silence recovery: 65,536 samples (1.36s at 48 kHz) decays +180 dBFS RC/modal tails to ~0
        let zeros = vec![0.0_f32; 65_536];
        let mut rec_l = vec![0.0_f32; 65_536];
        let mut rec_r = vec![0.0_f32; 65_536];
        eng.process_block(&zeros, &zeros, &mut rec_l, &mut rec_r);
        assert!(rec_l.iter().chain(&rec_r).all(|v| v.is_finite()));
        assert!(
            rec_l[65_535].abs() < 1.0e-3 && rec_r[65_535].abs() < 1.0e-3,
            "failed to recover after extreme overdrive burst: [{}, {}]",
            rec_l[65_535],
            rec_r[65_535]
        );
    }
}

#[derive(Clone, Debug)]
struct FourCableAndWdwFixture {
    n_in: usize,
    n_out: usize,
}

const ROUTER_META: FixtureDeviceMetadata = FixtureDeviceMetadata {
    id: "Router-4x4-Rig",
    name: "4CM & WDW Multiport Rig",
    category: "routing",
    sample_rate_hz: 48_000,
    bit_depth: FixtureBitDepth::Float32,
    expected_tail_ms: 15.0,
    supports_tempo_sync: false,
    controls: &[],
    switches: &[],
};

impl BlackBoxModel for FourCableAndWdwFixture {
    fn metadata(&self) -> FixtureDeviceMetadata {
        ROUTER_META
    }
    fn num_inputs(&self) -> usize {
        self.n_in
    }
    fn num_outputs(&self) -> usize {
        self.n_out
    }
    fn reset(&mut self) {}
    fn process_frame(&mut self, inputs: &[f32], outputs: &mut [f32], _: &[f32], _: &[u8]) {
        let i0 = inputs.first().copied().unwrap_or(0.0);
        let i1 = inputs.get(1).copied().unwrap_or(i0);
        let r0 = inputs.get(2).copied().unwrap_or(0.0);
        let r1 = inputs.get(3).copied().unwrap_or(r0);
        let send_l = (1.6 * i0).tanh();
        let send_r = (1.5 * i1).tanh();
        if let Some(o0) = outputs.get_mut(0) {
            *o0 = 0.7 * (send_l + 0.6 * r0);
        }
        if let Some(o1) = outputs.get_mut(1) {
            *o1 = 0.7 * (send_r + 0.6 * r1);
        }
        if let Some(o2) = outputs.get_mut(2) {
            *o2 = send_l;
        }
        if let Some(o3) = outputs.get_mut(3) {
            *o3 = send_r;
        }
    }
}

#[test]
fn test_case_19_all_six_routing_topologies_1to1_2to1_1to2_2to2_4cm_and_wdw() {
    let distill_topo =
        |topo: TopologyProbeMode, model_box: &mut dyn DistillationTarget, n_in: u8, n_out: u8| {
            let mut manifest = DnamProjManifest::default_reference();
            manifest.base.io_config.inputs = n_in;
            manifest.base.io_config.output_mode = if n_out >= 2 {
                "stereo".to_string()
            } else {
                "mono".to_string()
            };
            train_adaptive_target(
                manifest,
                model_box,
                &AdaptiveTrainingOptions {
                    sample_rate_hz: 48_000,
                    topology_mode: topo,
                    fixed_duration: true,
                    prefer_mode_b: true,
                    max_frames_per_stem: 3,
                    run_live_null_test: false,
                    null_test_autopatch_threshold_db: -36.0,
                    calb: None,
                },
            )
            .unwrap()
            .runtime_dnam_bytes
        };

    let sig_a: Vec<f32> = (0..256)
        .map(|n| 0.3 * (std::f32::consts::TAU * 220.0 * (n as f32) / 48_000.0).sin())
        .collect();
    let sig_b: Vec<f32> = (0..256)
        .map(|n| 0.25 * (std::f32::consts::TAU * 550.0 * (n as f32) / 48_000.0).cos())
        .collect();

    // 1. 1 -> 1 (Mono)
    let mut m_1to1 = BlackBoxDistillationAdapter::from_model(Chain01SymmetricOd::new());
    let dnam_1to1 = distill_topo(TopologyProbeMode::Mono1To1, &mut m_1to1, 1, 1);
    let mut eng_1to1 =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_1to1, Tier::Ultra, 48_000)
            .unwrap();
    let mut o1 = vec![0.0_f32; 256];
    eng_1to1.process_block_io(DnamAudioIo {
        in_1: &sig_a,
        in_2: None,
        in_3: None,
        in_4: None,
        out_1: &mut o1,
        out_2: None,
        out_3: None,
        out_4: None,
    });
    assert!(o1.iter().all(|v| v.is_finite()) && o1.iter().any(|v| v.abs() > 1.0e-4));

    // 2. 2 -> 1 (Dual-In / Sidechain to Mono)
    let mut m_2to1 = BlackBoxDistillationAdapter::from_model(Chain09La2aOpticalCompressor::new());
    let dnam_2to1 = distill_topo(TopologyProbeMode::DualInToMono2To1, &mut m_2to1, 2, 1);
    let mut eng_2to1 =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_2to1, Tier::Ultra, 48_000)
            .unwrap();
    let mut o2_l = vec![0.0_f32; 256];
    let mut o2_r = vec![0.0_f32; 256];
    eng_2to1.process_block_sidechain(&sig_a, &sig_a, &sig_b, &sig_b, &mut o2_l, &mut o2_r);
    assert!(o2_l.iter().all(|v| v.is_finite()));

    // 3. 1 -> 2 (Mono-to-Stereo)
    let mut m_1to2 = BlackBoxDistillationAdapter::from_model(Chain05PureLtiCabAndRoom::new());
    let dnam_1to2 = distill_topo(TopologyProbeMode::MonoToStereo1To2, &mut m_1to2, 1, 2);
    let mut eng_1to2 =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_1to2, Tier::Ultra, 48_000)
            .unwrap();
    eng_1to2.process_block(&sig_a, &sig_a, &mut o2_l, &mut o2_r);
    assert!(o2_l.iter().chain(&o2_r).all(|v| v.is_finite()));

    // 4. 2 -> 2 (True Stereo)
    let mut m_2to2 = BlackBoxDistillationAdapter::from_model(Chain07StereoChorusAndRotary::new());
    let dnam_2to2 = distill_topo(TopologyProbeMode::TrueStereo2To2, &mut m_2to2, 2, 2);
    let mut eng_2to2 =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_2to2, Tier::Ultra, 48_000)
            .unwrap();
    eng_2to2.process_block(&sig_a, &sig_b, &mut o2_l, &mut o2_r);
    assert!(o2_l.iter().chain(&o2_r).all(|v| v.is_finite()));

    // 5. 4CM (4-Cable Method): verify strict S_{ret -> send} == 0 (0.000000 ULP causal DAG isolation!)
    let mut m_4cm =
        BlackBoxDistillationAdapter::from_model(FourCableAndWdwFixture { n_in: 4, n_out: 4 });
    let dnam_4cm = distill_topo(TopologyProbeMode::FourCableMethod4Cm, &mut m_4cm, 4, 4);
    let zeros = vec![0.0_f32; 256];
    let mut eng_4cm_a =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_4cm, Tier::Ultra, 48_000).unwrap();
    let mut eng_4cm_b =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_4cm, Tier::Ultra, 48_000).unwrap();
    let (mut ml_a, mut mr_a, mut sl_a, mut sr_a) = (
        vec![0.0_f32; 256],
        vec![0.0_f32; 256],
        vec![0.0_f32; 256],
        vec![0.0_f32; 256],
    );
    let (mut ml_b, mut mr_b, mut sl_b, mut sr_b) = (
        vec![0.0_f32; 256],
        vec![0.0_f32; 256],
        vec![0.0_f32; 256],
        vec![0.0_f32; 256],
    );
    eng_4cm_a.process_block_4cm(
        &sig_a, &sig_a, &zeros, &zeros, &mut ml_a, &mut mr_a, &mut sl_a, &mut sr_a,
    );
    eng_4cm_b.process_block_4cm(
        &sig_a, &sig_a, &sig_b, &sig_b, &mut ml_b, &mut mr_b, &mut sl_b, &mut sr_b,
    );
    for i in 0..256 {
        assert_eq!(
            sl_a[i].to_bits(),
            sl_b[i].to_bits(),
            "4CM S_{{ret->send}} L feedback leak at frame {i}"
        );
        assert_eq!(
            sr_a[i].to_bits(),
            sr_b[i].to_bits(),
            "4CM S_{{ret->send}} R feedback leak at frame {i}"
        );
    }

    // 6. WDW (1 -> 3 Wet / Dry / Wet)
    let mut m_wdw =
        BlackBoxDistillationAdapter::from_model(FourCableAndWdwFixture { n_in: 2, n_out: 3 });
    let dnam_wdw = distill_topo(TopologyProbeMode::WetDryWet1To3, &mut m_wdw, 2, 3);
    let mut eng_wdw =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_wdw, Tier::Ultra, 48_000).unwrap();
    let (mut wl, mut wr, mut dc) = (vec![0.0_f32; 256], vec![0.0_f32; 256], vec![0.0_f32; 256]);
    eng_wdw.process_block_wdw(&sig_a, &sig_a, &mut wl, &mut wr, &mut dc);
    assert!(wl.iter().chain(&wr).chain(&dc).all(|v| v.is_finite()));
    assert!(
        dc.iter().any(|v| v.abs() > 1.0e-4),
        "WDW dry_center must carry signal"
    );
}
