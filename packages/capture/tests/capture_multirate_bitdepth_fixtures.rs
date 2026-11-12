#![forbid(unsafe_code)]
//! Multi-sample-rate (`32 / 44.1 / 48 / 88.2 / 96 / 192 kHz`), multi-bit-depth (`Int12`, `Int16`,
//! `Int24`, `Int32`, `Float32`), 12-chain + 14-decorator `BlackBoxModel`, canonical `48 kHz`
//! zero-phase resampling, and live large-payload `DNWB` subprocess verification suite for `dnam-capture`.

use dnam_capture::metrics::{
    ArchetypeFamily, MultiRateSegEsrSlice, certify_4metric_with_bit_depth,
    compute_multirate_seg_esr_db, evaluate_ltas_33band_report,
};
use dnam_capture::whitebox::{FaustRunner, SPICE_CAUSAL_DELAY_SAMPLES, SpiceRunner};
use dnam_capture::{
    CANONICAL_TRAINER_SAMPLE_RATE_HZ, CaptureBitDepth, DistillationTarget, StemResponse,
    StemStimulus, StimulusFrame, resample_arbitrary_rate, resample_control_trajectory,
    resampled_frame_len, scale_freeze_window,
};
use dnam_engine::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::schema::{DnamError, DnamJsonHeader};
use dnam_engine::{DnamEngine, Tier};
use dnam_fixtures::{
    AdcClipAndPadDecorator, BlackBoxModel, Chain01SymmetricOd, Chain02GatedAsymmetricFuzz,
    Chain03ParallelCleanBlend, Chain04TubeAmpFmv4x12, Chain05PureLtiCabAndRoom,
    Chain06EnvelopeResonantSvf, Chain07StereoChorusAndRotary, Chain08LongBbdEchoAnd30sShimmer,
    Chain09La2aOpticalCompressor, Chain10Oc2OctaveDividerAndRingMod, Chain11BiAmpBassAndOutputIron,
    Chain12FullEndToEndStereoRig, Chain12RigMode, ConverterLoopbackDecorator,
    DigitalQuirksDecorator, FixtureBitDepth, FixtureDeviceMetadata, GermaniumTempDriftDecorator,
    HardwareNoiseGateDecorator, HumanHandTrajectoryDecorator, IdleSquealDecorator,
    MainsHumAndNoiseFloorDecorator, MicrophoneCapsuleClipDecorator, PolarityAndDcOffsetDecorator,
    PotTaperAndSwitchDecorator, ScratchyPotAndSteppedAttenuatorDecorator, SpringChirpDecorator,
    TaperCurveKind, TubeThermalHysteresisDecorator,
};
use std::collections::BTreeMap;
use std::f64::consts::PI;

/// Adapter bridging any zero-dependency [`dnam_fixtures::BlackBoxModel`] into [`dnam_capture::DistillationTarget`]:
/// - Preserves `self.inner`'s true native `device_fs` (never mutating `set_sample_rate_hz`), performing
///   zero-phase Kaiser-sinc `stim_fs -> device_fs -> stim_fs` audio resampling, monotone trajectory
///   resampling, and proportional `freeze_transition_window` scaling whenever `stim_fs != device_fs`.
/// - Maps all 5 [`FixtureBitDepth`] variants (`Int12`, `Int16`, `Int24`, `Int32`, `Float32`) losslessly
///   into [`CaptureBitDepth`].
/// - Supports `1..=4` inputs (`in_3` sidechain key routing), `1..=4` outputs (`WDW` / `4CM`), `[P][N]`
///   `control_trajectories`, and sample-accurate `freeze_transition_window` switch latching.
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

    fn new(inner: M, switches: Vec<u8>) -> Self {
        Self { inner, switches }
    }

    fn device_sample_rate_hz(&self) -> u32 {
        self.inner.sample_rate_hz()
    }

    fn device_bit_depth(&self) -> CaptureBitDepth {
        let bd = self.inner.bit_depth();
        CaptureBitDepth::from_format(bd.bits(), bd.is_float()).unwrap_or(CaptureBitDepth::Float32)
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
        if !(8_000..=1_536_000).contains(&stim_sr) {
            return Err(DnamError::InvalidSampleRate(stim_sr));
        }
        if !(8_000..=1_536_000).contains(&dev_sr) {
            return Err(DnamError::InvalidSampleRate(dev_sr));
        }
        let orig_n = in_1.len();
        for (lbl, ch) in [("in_2", in_2), ("in_3", in_3), ("in_4", in_4)] {
            if let Some(buf) = ch
                && buf.len() != orig_n
            {
                return Err(DnamError::SchemaValidationError(format!(
                    "{lbl} len ({}) != in_1 len ({orig_n})",
                    buf.len()
                )));
            }
        }
        if let Some(trajs) = control_trajectories {
            if trajs.len() != control_values.len() {
                return Err(DnamError::SchemaValidationError(String::from(
                    "control_trajectories count mismatch",
                )));
            }
            if trajs.iter().any(|tr| tr.len() != orig_n) {
                return Err(DnamError::SchemaValidationError(String::from(
                    "control_trajectory len != in_1 len",
                )));
            }
        }
        if let Some([fs, fe]) = freeze_window
            && (fs > fe || fe > orig_n)
        {
            return Err(DnamError::SchemaValidationError(String::from(
                "invalid freeze_window",
            )));
        }
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
            let fw = freeze_window.map(|w| scale_freeze_window(w, orig_n, dev_n));
            (
                rs_in1.as_slice(),
                rs_in2.as_deref(),
                rs_in3.as_deref(),
                rs_in4.as_deref(),
                rs_trajs.as_ref(),
                fw,
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
        let mut sw_scratch = vec![0u8; n_sw];
        for (k, slot) in sw_scratch.iter_mut().enumerate() {
            *slot = self
                .switches
                .get(k)
                .copied()
                .or_else(|| meta.switches.get(k).map(|s| s.default_index))
                .unwrap_or(0);
        }
        let base_freeze_sw = sw_scratch.get(freeze_sw_idx).copied().unwrap_or(0);

        let n_in = self.inner.num_inputs().clamp(1, 4);
        let n_out = self.inner.num_outputs().clamp(1, 4);
        let n_ctrl = meta
            .controls
            .len()
            .max(control_values.len())
            .max(dev_trajs.map_or(0, std::vec::Vec::len));
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
                let fallback = control_values
                    .get(p)
                    .copied()
                    .or_else(|| default_controls.get(p).copied())
                    .unwrap_or(0.0);
                ctrl_scratch[p] = dev_trajs
                    .and_then(|trajs| trajs.get(p))
                    .and_then(|t| t.get(i).copied())
                    .unwrap_or(fallback);
            }

            if let Some([fs, fe]) = dev_freeze
                && let Some(slot) = sw_scratch.get_mut(freeze_sw_idx)
            {
                *slot = if i >= fs && i < fe { 1 } else { base_freeze_sw };
            }

            let in_frame = [x1, x2, x3, x4];
            let mut y = [0.0_f32; 4];
            self.inner.process_frame(
                &in_frame[..n_in],
                &mut y[..n_out],
                &ctrl_scratch,
                &sw_scratch,
            );
            dev_out1[i] = y[0];
            if let Some(ref mut ch2) = dev_out2 {
                ch2[i] = y[1];
            }
            if let Some(ref mut ch3) = dev_out3 {
                ch3[i] = y[2];
            }
            if let Some(ref mut ch4) = dev_out4 {
                ch4[i] = y[3];
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

    fn render_stem(&mut self, stimulus: &StemStimulus) -> Result<StemResponse, DnamError> {
        self.reset_state();
        self.render_internal(
            stimulus.stem_id.clone(),
            stimulus.sample_rate_hz,
            &stimulus.in_1,
            stimulus.in_2.as_deref(),
            stimulus.in_3.as_deref(),
            stimulus.in_4.as_deref(),
            &stimulus.control_values,
            stimulus.control_trajectories.as_ref(),
            stimulus.freeze_transition_window,
        )
    }

    fn render_frame(&mut self, frame: &StimulusFrame) -> Result<StemResponse, DnamError> {
        self.render_internal(
            format!("frame_{}", frame.frame_index),
            frame.sample_rate_hz,
            &frame.in_1,
            frame.in_2.as_deref(),
            frame.in_3.as_deref(),
            frame.in_4.as_deref(),
            &frame.control_values,
            frame.control_trajectories.as_ref(),
            None,
        )
    }
}

/// Minimal `4 -> 4` (`WDW` / `4CM`) black-box fixture for verifying `out_3` and `out_4` adapter routing.
#[derive(Clone, Debug)]
struct QuadPortFixture;

const QUAD_META: FixtureDeviceMetadata = FixtureDeviceMetadata {
    id: "Quad-4x4-Router",
    name: "4x4 WDW/4CM Test Router",
    category: "routing",
    sample_rate_hz: 48_000,
    bit_depth: FixtureBitDepth::Int32,
    expected_tail_ms: 0.0,
    supports_tempo_sync: false,
    controls: &[],
    switches: &[],
};

impl BlackBoxModel for QuadPortFixture {
    fn metadata(&self) -> FixtureDeviceMetadata {
        QUAD_META
    }
    fn num_inputs(&self) -> usize {
        4
    }
    fn num_outputs(&self) -> usize {
        4
    }
    fn reset(&mut self) {}
    fn process_frame(&mut self, inputs: &[f32], outputs: &mut [f32], _c: &[f32], _s: &[u8]) {
        for ch in 0..outputs.len().min(4) {
            let x = inputs.get(ch).copied().unwrap_or(0.0);
            outputs[ch] = QUAD_META.bit_depth.quantize_sample(0.5 * x);
        }
    }
}

fn log_chirp_f32(n: usize, sample_rate_hz: u32, amp: f32) -> Vec<f32> {
    let sr = f64::from(sample_rate_hz);
    let half_dur = 0.5 * (n as f64) / sr;
    let f0 = 20.0_f64;
    let f1 = (0.455 * sr).min(20_200.0);
    let ln_ratio = (f1 / f0).ln();
    (0..n)
        .map(|i| {
            // Symmetric up-down log sweep so both 20 Hz and 20 kHz fall inside the peak of the Welch Hann window
            let u = 2.0 * (i as f64) / (n.max(1) as f64);
            let s = if u <= 1.0 { u } else { 2.0 - u };
            let phase = 2.0 * PI * f0 * half_dur * ((s * ln_ratio).exp() - 1.0) / ln_ratio;
            (f64::from(amp) * phase.sin()) as f32
        })
        .collect()
}

#[test]
fn test_all_12_chains_14_decorators_trajectories_freeze_4ch_and_sidechain_key() {
    // verify[train.pipeline.fixtures-matrix]
    let dry = log_chirp_f32(512, 48_000, 0.45);
    let stim = StemStimulus {
        stem_id: String::from("full_matrix_48k"),
        sample_rate_hz: 48_000,
        in_1: dry.clone(),
        in_2: Some(dry.clone()),
        in_3: None,
        in_4: None,
        control_values: vec![0.35, 0.15, -0.15, 0.25],
        control_trajectories: None,
        freeze_transition_window: None,
    };

    macro_rules! check_chain {
        ($model:expr) => {{
            let mut adapter = BlackBoxDistillationAdapter::from_model($model);
            let resp = adapter.render_stem(&stim).unwrap();
            assert_eq!(resp.out_1.len(), 512);
            assert!(resp.out_1.iter().all(|v| v.is_finite()));
            if let Some(ref r) = resp.out_2 {
                assert_eq!(r.len(), 512);
                assert!(r.iter().all(|v| v.is_finite()));
            }
        }};
    }
    check_chain!(Chain01SymmetricOd::new());
    check_chain!(Chain02GatedAsymmetricFuzz::new());
    check_chain!(Chain03ParallelCleanBlend::new());
    check_chain!(Chain04TubeAmpFmv4x12::new());
    check_chain!(Chain05PureLtiCabAndRoom::new());
    check_chain!(Chain06EnvelopeResonantSvf::new());
    check_chain!(Chain07StereoChorusAndRotary::new());
    check_chain!(Chain08LongBbdEchoAnd30sShimmer::new());
    check_chain!(Chain09La2aOpticalCompressor::new());
    check_chain!(Chain10Oc2OctaveDividerAndRingMod::new());
    check_chain!(Chain11BiAmpBassAndOutputIron::new());
    check_chain!(Chain12FullEndToEndStereoRig::new(
        Chain12RigMode::FixedZeroKnob
    ));
    check_chain!(Chain12FullEndToEndStereoRig::new(
        Chain12RigMode::Parametric16
    ));

    // Exercise all 14 composable hardware impairment decorators stacked end-to-end
    let decorated_stack = AdcClipAndPadDecorator::new(
        ConverterLoopbackDecorator::new(MainsHumAndNoiseFloorDecorator::new(
            MicrophoneCapsuleClipDecorator::new(TubeThermalHysteresisDecorator::new(
                HumanHandTrajectoryDecorator::new(
                    PotTaperAndSwitchDecorator::new(
                        PolarityAndDcOffsetDecorator::new(
                            IdleSquealDecorator::new(
                                ScratchyPotAndSteppedAttenuatorDecorator::new(
                                    DigitalQuirksDecorator::new(SpringChirpDecorator::new(
                                        GermaniumTempDriftDecorator::new(
                                            HardwareNoiseGateDecorator::new(
                                                Chain04TubeAmpFmv4x12::new(),
                                            ),
                                            31.0,
                                        ),
                                    )),
                                ),
                            ),
                            true,
                            0.01,
                        ),
                        vec![TaperCurveKind::Log15A, TaperCurveKind::WTaper],
                    ),
                    12,
                    0.05,
                ),
            )),
            60.0,
        )),
        6.0,
    );
    let mut stack_adapter = BlackBoxDistillationAdapter::new(decorated_stack, vec![1]);
    let stack_resp = stack_adapter.render_stem(&stim).unwrap();
    assert_eq!(stack_resp.out_1.len(), 512);
    assert!(stack_resp.out_1.iter().all(|v| v.is_finite()));

    // Verify 4x4 WDW/4CM output routing (`out_3`, `out_4`) + `Int32` bit depth mapping
    let mut quad_adapter = BlackBoxDistillationAdapter::from_model(QuadPortFixture);
    assert_eq!(quad_adapter.device_bit_depth(), CaptureBitDepth::Int32);
    let quad_resp = quad_adapter
        .render_stem(&StemStimulus {
            stem_id: String::from("quad_4x4"),
            sample_rate_hz: 48_000,
            in_1: vec![0.4; 32],
            in_2: Some(vec![-0.4; 32]),
            in_3: Some(vec![0.2; 32]),
            in_4: Some(vec![-0.2; 32]),
            control_values: Vec::new(),
            control_trajectories: None,
            freeze_transition_window: None,
        })
        .unwrap();
    assert_eq!(quad_resp.out_3.as_ref().unwrap().len(), 32);
    assert_eq!(quad_resp.out_4.as_ref().unwrap().len(), 32);
    assert!((quad_resp.out_3.unwrap()[0] - 0.1).abs() < 1.0e-6);
    assert!((quad_resp.out_4.unwrap()[0] - (-0.1)).abs() < 1.0e-6);

    // Verify `freeze_transition_window` switch latching on Chain-08 FDN shimmer
    let mut freeze_adapter =
        BlackBoxDistillationAdapter::from_model(Chain08LongBbdEchoAnd30sShimmer::new());
    let mut burst_in = vec![0.0_f32; 8_192];
    burst_in[..2_048].copy_from_slice(&log_chirp_f32(2_048, 48_000, 0.7));
    let unlatched = freeze_adapter
        .render_stem(&StemStimulus {
            stem_id: String::from("ch08_unlatched"),
            sample_rate_hz: 48_000,
            in_1: burst_in.clone(),
            in_2: Some(burst_in.clone()),
            in_3: None,
            in_4: None,
            control_values: vec![-0.8, -0.8, 1.0],
            control_trajectories: None,
            freeze_transition_window: None,
        })
        .unwrap();
    let latched = freeze_adapter
        .render_stem(&StemStimulus {
            stem_id: String::from("ch08_latched"),
            sample_rate_hz: 48_000,
            in_1: burst_in.clone(),
            in_2: Some(burst_in),
            in_3: None,
            in_4: None,
            control_values: vec![-0.8, -0.8, 1.0],
            control_trajectories: None,
            freeze_transition_window: Some([2_048, 8_192]),
        })
        .unwrap();
    let e_unlatched: f64 = unlatched.out_1[6_000..]
        .iter()
        .map(|&v| f64::from(v).powi(2))
        .sum();
    let e_latched: f64 = latched.out_1[6_000..]
        .iter()
        .map(|&v| f64::from(v).powi(2))
        .sum();
    assert!(
        e_latched > e_unlatched * 1.25,
        "freeze_transition_window failed to sustain FDN energy: latched={e_latched:e} vs unlatched={e_unlatched:e}"
    );

    // Verify [P][N] control_trajectories indexing: sweeping drive [-1 -> +1] increases tail RMS vs static -1
    let mut od_adapter = BlackBoxDistillationAdapter::from_model(Chain01SymmetricOd::new());
    let low_stim = StemStimulus {
        stem_id: String::from("od_low"),
        sample_rate_hz: 48_000,
        in_1: dry.clone(),
        in_2: None,
        in_3: None,
        in_4: None,
        control_values: vec![-1.0, 0.0, 0.0],
        control_trajectories: None,
        freeze_transition_window: None,
    };
    let ramp_stim = StemStimulus {
        stem_id: String::from("od_ramp"),
        sample_rate_hz: 48_000,
        in_1: dry.clone(),
        in_2: None,
        in_3: None,
        in_4: None,
        control_values: vec![-1.0, 0.0, 0.0],
        control_trajectories: Some(vec![
            (0..512).map(|i| -1.0 + 2.0 * (i as f32) / 511.0).collect(),
            vec![0.0; 512],
            vec![0.0; 512],
        ]),
        freeze_transition_window: None,
    };
    let low_out = od_adapter.render_stem(&low_stim).unwrap().out_1;
    let ramp_out = od_adapter.render_stem(&ramp_stim).unwrap().out_1;
    let tail_rms_low: f64 = low_out[256..].iter().map(|&v| f64::from(v).powi(2)).sum();
    let tail_rms_ramp: f64 = ramp_out[256..].iter().map(|&v| f64::from(v).powi(2)).sum();
    assert!(tail_rms_ramp > tail_rms_low * 1.15);

    // Verify external auxiliary sidechain key input on frame.in_3: hot key ducks Chain-09 LA-2A
    let mut comp = BlackBoxDistillationAdapter::new(Chain09La2aOpticalCompressor::new(), vec![1]);
    let prog = log_chirp_f32(1024, 48_000, 0.25);
    let resp_unkeyed = comp
        .render_stem(&StemStimulus {
            stem_id: String::from("comp_silent_key"),
            sample_rate_hz: 48_000,
            in_1: prog.clone(),
            in_2: None,
            in_3: Some(vec![0.0_f32; 1024]),
            in_4: None,
            control_values: vec![0.8, 0.0],
            control_trajectories: None,
            freeze_transition_window: None,
        })
        .unwrap();
    let resp_keyed = comp
        .render_stem(&StemStimulus {
            stem_id: String::from("comp_hot_key"),
            sample_rate_hz: 48_000,
            in_1: prog,
            in_2: None,
            in_3: Some(vec![0.95_f32; 1024]),
            in_4: None,
            control_values: vec![0.8, 0.0],
            control_trajectories: None,
            freeze_transition_window: None,
        })
        .unwrap();
    let e_unkeyed: f64 = resp_unkeyed.out_1[512..]
        .iter()
        .map(|&v| f64::from(v).powi(2))
        .sum();
    let e_keyed: f64 = resp_keyed.out_1[512..]
        .iter()
        .map(|&v| f64::from(v).powi(2))
        .sum();
    assert!(e_keyed < e_unkeyed * 0.75);
}

#[test]
fn test_trainer_multi_sample_rate_and_5_bit_depth_device_certification() {
    // 1. Multi-sample-rate (44.1 / 48 / 88.2 / 96 / 192 kHz) across Fixtures, DnamEngine, SpiceRunner & FaustRunner
    let mut hdr = DnamJsonHeader::default_reference();
    let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    arena[124..128].copy_from_slice(&0.9993_f32.to_le_bytes());
    let container = build_dnam_container(&mut hdr, &arena, &arena, false).unwrap();
    let mut dnam_target =
        DnamEngine::<DefaultMathBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
    let mut spice_target = SpiceRunner::new("multirate.cir", 5_000);
    let faust_json = r#"{"ui":[{"items":[{"type":"hslider","label":"Drive","init":0.5,"min":0.0,"max":1.0,"step":0.01},{"type":"hslider","label":"Tone","init":0.5,"min":0.0,"max":1.0,"step":0.01}]}]}"#;
    let mut faust_target =
        FaustRunner::from_faust_json("multirate.dsp", faust_json, 5_000).unwrap();

    // Invalid sub-8kHz sample rates are rejected cleanly by white-box runners & adapter
    let invalid_sr_stim = StemStimulus {
        stem_id: String::from("bad_sr"),
        sample_rate_hz: 0,
        in_1: vec![0.1; 16],
        in_2: None,
        in_3: None,
        in_4: None,
        control_values: vec![0.0],
        control_trajectories: None,
        freeze_transition_window: None,
    };
    assert!(spice_target.render_stem(&invalid_sr_stim).is_err());
    assert!(faust_target.render_stem(&invalid_sr_stim).is_err());

    for &sr in &[44_100u32, 48_000, 88_200, 96_000, 192_000] {
        // Ensure >= ~340 ms duration so rate-scaled Welch LTAS resolves [20, 22.39 Hz) cleanly up to 192 kHz
        let n = 16_384usize.max((sr as usize) / 3).next_power_of_two();
        let dry = log_chirp_f32(n, sr, 0.55);
        let stim = StemStimulus {
            stem_id: format!("rate_{sr}"),
            sample_rate_hz: sr,
            in_1: dry.clone(),
            in_2: Some(dry),
            control_values: vec![0.3, 0.1, 0.0],
            in_3: None,
            in_4: None,
            control_trajectories: None,
            freeze_transition_window: None,
        };

        if sr <= 96_000 {
            let r_eng = dnam_target.render_stem(&stim).unwrap();
            assert_eq!(r_eng.sample_rate_hz, sr);
            assert_eq!(dnam_target.sample_rate_hz(), sr);
            assert_eq!(r_eng.out_1.len(), n);
        }

        let r_spice = spice_target.render_stem(&stim).unwrap();
        assert_eq!(r_spice.sample_rate_hz, sr);
        assert_eq!(r_spice.out_1.len(), n);
        assert_eq!(r_spice.internal_nodes["spice_diode_current"].len(), n);
        // Zero-phase batch SPICE render_stem must respond within the first 8 samples (zero 39-sample causal lag!)
        assert!(
            r_spice.out_1[..8].iter().any(|&v| v.abs() > 1.0e-5),
            "SpiceRunner::render_stem lagged at sr={sr} (SPICE_CAUSAL_DELAY_SAMPLES={SPICE_CAUSAL_DELAY_SAMPLES})"
        );

        let r_faust = faust_target.render_stem(&stim).unwrap();
        assert_eq!(r_faust.sample_rate_hz, sr);
        assert_eq!(r_faust.out_1.len(), n);

        // Native-rate 5-bit-depth (`Int12`, `Int16`, `Int24`, `Int32`, `Float32`) Pro certification
        let mut clean_ref = BlackBoxDistillationAdapter::from_model(
            Chain01SymmetricOd::with_format(sr, FixtureBitDepth::Float32),
        );
        let clean_out: Vec<f64> = clean_ref
            .render_stem(&stim)
            .unwrap()
            .out_1
            .into_iter()
            .map(f64::from)
            .collect();

        for fx_depth in FixtureBitDepth::ALL {
            let mut dut = BlackBoxDistillationAdapter::from_model(Chain01SymmetricOd::with_format(
                sr, fx_depth,
            ));
            let cap_depth = dut.device_bit_depth();
            assert_eq!(cap_depth.bits(), fx_depth.bits());
            assert_eq!(cap_depth.is_float(), fx_depth.is_float());
            assert_eq!(dut.device_sample_rate_hz(), sr);

            let dut_out: Vec<f64> = dut
                .render_stem(&stim)
                .unwrap()
                .out_1
                .into_iter()
                .map(f64::from)
                .collect();

            let cert = certify_4metric_with_bit_depth(
                ArchetypeFamily::CleanPedalPreamp,
                Tier::Pro,
                &dut_out,
                &clean_out,
                cap_depth,
                0.0,
                f64::from(sr),
                64,
            );
            assert!(
                cert.passes_target && cert.passes_hard_ceiling,
                "Native device sr={sr} depth={fx_depth:?} failed Pro certification: {cert:?}"
            );

            if fx_depth == FixtureBitDepth::Int12 {
                // Genuine +2.5 dB gain mismatch must STILL be rejected even with 12-bit noise qualification!
                let bad_model: Vec<f64> = clean_out.iter().map(|&v| v * 1.3335).collect();
                let bad_cert = certify_4metric_with_bit_depth(
                    ArchetypeFamily::CleanPedalPreamp,
                    Tier::Pro,
                    &dut_out,
                    &bad_model,
                    cap_depth,
                    0.0,
                    f64::from(sr),
                    64,
                );
                assert!(!bad_cert.passes_target);
            }
        }
    }

    // 2. Canonical 48 kHz trainer stimulus rendered against non-48 kHz devices (44.1 / 88.2 / 96 / 192 kHz)
    //    across all 5 bit depths WITHOUT mutating device native sample rate!
    let n_48k = 16_384usize;
    let dry_48k = log_chirp_f32(n_48k, CANONICAL_TRAINER_SAMPLE_RATE_HZ, 0.50);
    let ramp_traj: Vec<f32> = (0..n_48k)
        .map(|i| -0.4 + 0.8 * (i as f32) / ((n_48k - 1) as f32))
        .collect();
    let canonical_stim = StemStimulus {
        stem_id: String::from("canonical_48k_probe"),
        sample_rate_hz: CANONICAL_TRAINER_SAMPLE_RATE_HZ,
        in_1: dry_48k.clone(),
        in_2: Some(dry_48k),
        in_3: None,
        in_4: None,
        control_values: vec![0.0, 0.1, 0.0],
        control_trajectories: Some(vec![ramp_traj, vec![0.1; n_48k], vec![0.0; n_48k]]),
        freeze_transition_window: None,
    };

    for &dev_sr in &[44_100u32, 88_200, 96_000, 192_000] {
        let mut ref_dev = BlackBoxDistillationAdapter::from_model(Chain01SymmetricOd::with_format(
            dev_sr,
            FixtureBitDepth::Float32,
        ));
        let ref_resp = ref_dev.render_stem(&canonical_stim).unwrap();
        assert_eq!(ref_dev.device_sample_rate_hz(), dev_sr);
        assert_eq!(ref_resp.sample_rate_hz, CANONICAL_TRAINER_SAMPLE_RATE_HZ);
        assert_eq!(ref_resp.out_1.len(), n_48k);
        let ref_48k: Vec<f64> = ref_resp.out_1.into_iter().map(f64::from).collect();

        for fx_depth in FixtureBitDepth::ALL {
            let mut dut = BlackBoxDistillationAdapter::from_model(Chain01SymmetricOd::with_format(
                dev_sr, fx_depth,
            ));
            let resp = dut.render_stem(&canonical_stim).unwrap();
            // Device's native sample rate must NEVER be overwritten to 48,000 Hz!
            assert_eq!(dut.device_sample_rate_hz(), dev_sr);
            assert_eq!(resp.sample_rate_hz, CANONICAL_TRAINER_SAMPLE_RATE_HZ);
            assert_eq!(resp.out_1.len(), n_48k);

            let dut_48k: Vec<f64> = resp.out_1.into_iter().map(f64::from).collect();
            let cert = certify_4metric_with_bit_depth(
                ArchetypeFamily::CleanPedalPreamp,
                Tier::Pro,
                &dut_48k,
                &ref_48k,
                dut.device_bit_depth(),
                1.0e-6,
                f64::from(CANONICAL_TRAINER_SAMPLE_RATE_HZ),
                256,
            );
            assert!(
                cert.passes_target && cert.passes_hard_ceiling,
                "Canonical 48 kHz <-> {dev_sr} Hz resampled device ({fx_depth:?}) failed Pro certification: {cert:?}"
            );
        }
    }

    // 3. Sub-40 kHz vintage digital pedal (32 kHz, Int12) via canonical 48 kHz adapter & native passband LTAS/SegESR
    let mut quirk_32k = BlackBoxDistillationAdapter::from_model(DigitalQuirksDecorator::new(
        Chain01SymmetricOd::with_format(32_000, FixtureBitDepth::Int12),
    ));
    let quirk_48k_resp = quirk_32k.render_stem(&canonical_stim).unwrap();
    assert_eq!(quirk_32k.device_sample_rate_hz(), 32_000);
    assert_eq!(
        (quirk_48k_resp.sample_rate_hz, quirk_48k_resp.out_1.len()),
        (CANONICAL_TRAINER_SAMPLE_RATE_HZ, n_48k)
    );
    let sr_vintage = 32_000u32;
    let stim_v = StemStimulus {
        stem_id: String::from("vintage_32k_12bit"),
        sample_rate_hz: sr_vintage,
        in_1: log_chirp_f32(8_192, sr_vintage, 0.5),
        in_2: None,
        in_3: None,
        in_4: None,
        control_values: vec![0.2, 0.0, 0.0],
        control_trajectories: None,
        freeze_transition_window: None,
    };
    let out_v: Vec<f64> = quirk_32k
        .render_stem(&stim_v)
        .unwrap()
        .out_1
        .into_iter()
        .map(f64::from)
        .collect();
    let mdl_v: Vec<f64> = out_v.iter().map(|&v| v * 0.995).collect();
    let ltas_v = evaluate_ltas_33band_report(&out_v, &mdl_v, None, f64::from(sr_vintage), false);
    assert!(ltas_v.available && !ltas_v.excluded_band_guard_failed && ltas_v.active_band_count > 0);
    assert!(ltas_v.max_adj_deviation_db < 0.2);
    let (seg_v_db, seg_v_cnt, seg_v_ok) = compute_multirate_seg_esr_db(&[MultiRateSegEsrSlice {
        sample_rate_hz: f64::from(sr_vintage),
        target: &out_v,
        model: &mdl_v,
        noise: None,
    }]);
    assert!(seg_v_ok && seg_v_cnt > 0 && seg_v_db < -40.0);
}
