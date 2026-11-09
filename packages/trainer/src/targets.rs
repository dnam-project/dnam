#![forbid(unsafe_code)]
//! Sans-IO [`DistillationTarget`] trait and multi-port excitation/response descriptors
//! (`dnam_trainer::targets`, `ref[train.capture.excitation.universal-30s]`).

use std::collections::BTreeMap;

use dnam_engine::math::MathBackend;
use dnam_engine::schema::DnamError;
use dnam_engine::{DnamAudioIo, DnamEngine, DynamicDnamEngine};

/// Single causal excitation frame (e.g. `96 ms` = `4,608` samples at `48 kHz`) for closed-loop
/// `AdaptiveStimulusController` target probing (`ref[train.capture.excitation.universal-30s]`).
#[derive(Clone, Debug, PartialEq)]
pub struct StimulusFrame {
    /// Zero-based sequential frame index within the active stem.
    pub frame_index: usize,
    /// Sample rate in Hz (`48_000` canonical).
    pub sample_rate_hz: u32,
    /// Primary input channel `in_1` (`Inst In L` / `Mono`).
    pub in_1: Vec<f32>,
    /// Optional stereo right input channel `in_2` (`Inst In R`).
    pub in_2: Option<Vec<f32>>,
    /// Optional auxiliary/return left input channel `in_3` (`FX Loop Return L` / `Sidechain Key L`).
    pub in_3: Option<Vec<f32>>,
    /// Optional auxiliary/return right input channel `in_4` (`FX Loop Return R` / `Sidechain Key R`).
    pub in_4: Option<Vec<f32>>,
    /// Physical or normalized control values `[0..P]` at the start of this frame.
    pub control_values: Vec<f32>,
    /// Optional per-control sample-accurate sweep trajectories (`len == P`, each `len == in_1.len()`).
    pub control_trajectories: Option<Vec<Vec<f32>>>,
}

/// Complete excitation stem (`Mode A` continuous sweep or `Mode B` stationary stop) for
/// hardware capture or white-box/grey-box target distillation.
#[derive(Clone, Debug, PartialEq)]
pub struct StemStimulus {
    /// Unique stem identifier (e.g. `"star_center"`, `"sweep_ctrl_0"`, `"hadamard_diag_0"`).
    pub stem_id: String,
    /// Sample rate in Hz (`48_000` canonical).
    pub sample_rate_hz: u32,
    /// Primary input channel `in_1` (`Inst In L` / `Mono`).
    pub in_1: Vec<f32>,
    /// Optional stereo right input channel `in_2` (`Inst In R`).
    pub in_2: Option<Vec<f32>>,
    /// Optional auxiliary/return left input channel `in_3` (`FX Loop Return L` / `Sidechain Key L`).
    pub in_3: Option<Vec<f32>>,
    /// Optional auxiliary/return right input channel `in_4` (`FX Loop Return R` / `Sidechain Key R`).
    pub in_4: Option<Vec<f32>>,
    /// Stationary control values `[0..P]` for this stem.
    pub control_values: Vec<f32>,
    /// Optional per-control continuous sweep trajectories (`len == P`, each `len == in_1.len()`).
    pub control_trajectories: Option<Vec<Vec<f32>>>,
    /// Optional `[n_earliest, n_latest)` sample window for timed `freeze_latch` transitions.
    pub freeze_transition_window: Option<[usize; 2]>,
}

/// Batch of [`StemStimulus`] descriptors for Sans-IO 1-or-2-pass distillation.
#[derive(Clone, Debug, PartialEq)]
pub struct StemStimulusBatch {
    /// Sample rate in Hz (`48_000` canonical).
    pub sample_rate_hz: u32,
    /// Ordered stems in this batch.
    pub stems: Vec<StemStimulus>,
}

/// Rendered multi-port audio response (plus optional white-box/grey-box internal circuit nodes)
/// produced by a [`DistillationTarget`].
#[derive(Clone, Debug, PartialEq)]
pub struct StemResponse {
    /// Stem or frame identifier matching the request.
    pub stem_id: String,
    /// Sample rate in Hz.
    pub sample_rate_hz: u32,
    /// Primary output channel `out_1` (`Main Out L` / `Mono`).
    pub out_1: Vec<f32>,
    /// Optional stereo right output channel `out_2` (`Main Out R`).
    pub out_2: Option<Vec<f32>>,
    /// Optional auxiliary output channel `out_3` (`FX Loop Send L` or `WDW Dry Center`).
    pub out_3: Option<Vec<f32>>,
    /// Optional auxiliary output channel `out_4` (`FX Loop Send R`).
    pub out_4: Option<Vec<f32>>,
    /// Optional tapped internal circuit or stage node waveforms (`grey-box` / `white-box` distillation).
    pub internal_nodes: BTreeMap<String, Vec<f32>>,
}

/// Coherence-safe target trait implemented by [`DnamEngine`], [`DynamicDnamEngine`],
/// `dnam-fixtures` analytical signal chains, and white-box runners.
pub trait DistillationTarget {
    /// Resets all internal recursive filter, delay, and dynamic control states to resting DC equilibrium.
    fn reset_state(&mut self);

    /// Resets state to DC equilibrium and renders a complete [`StemStimulus`].
    fn render_stem(&mut self, stimulus: &StemStimulus) -> Result<StemResponse, DnamError>;

    /// Renders a single causal [`StimulusFrame`] **without** resetting state, enabling closed-loop
    /// frame-by-frame adaptive feedback excitation (`AdaptiveStimulusController::next_frame`).
    fn render_frame(&mut self, frame: &StimulusFrame) -> Result<StemResponse, DnamError> {
        let stem = StemStimulus {
            stem_id: format!("frame_{}", frame.frame_index),
            sample_rate_hz: frame.sample_rate_hz,
            in_1: frame.in_1.clone(),
            in_2: frame.in_2.clone(),
            in_3: frame.in_3.clone(),
            in_4: frame.in_4.clone(),
            control_values: frame.control_values.clone(),
            control_trajectories: frame.control_trajectories.clone(),
            freeze_transition_window: None,
        };
        self.render_stem(&stem)
    }
}

impl<'a, B: MathBackend> DistillationTarget for DnamEngine<'a, B> {
    fn reset_state(&mut self) {
        self.reset_to_dc_equilibrium();
    }

    fn render_stem(&mut self, stimulus: &StemStimulus) -> Result<StemResponse, DnamError> {
        if stimulus.sample_rate_hz != self.sample_rate_hz() {
            self.set_sample_rate(stimulus.sample_rate_hz)?;
        }
        self.reset_to_dc_equilibrium();
        let frame = StimulusFrame {
            frame_index: 0,
            sample_rate_hz: stimulus.sample_rate_hz,
            in_1: stimulus.in_1.clone(),
            in_2: stimulus.in_2.clone(),
            in_3: stimulus.in_3.clone(),
            in_4: stimulus.in_4.clone(),
            control_values: stimulus.control_values.clone(),
            control_trajectories: stimulus.control_trajectories.clone(),
        };
        let mut resp = self.render_frame(&frame)?;
        resp.stem_id.clone_from(&stimulus.stem_id);
        Ok(resp)
    }

    fn render_frame(&mut self, frame: &StimulusFrame) -> Result<StemResponse, DnamError> {
        if frame.sample_rate_hz != self.sample_rate_hz() {
            self.set_sample_rate(frame.sample_rate_hz)?;
        }
        for (idx, &val) in frame.control_values.iter().enumerate() {
            if idx < self.num_controls() {
                self.set_control_immediate(idx, val);
            }
        }
        let n = frame.in_1.len();
        let mut out_1 = vec![0.0_f32; n];
        let mut out_2 = vec![0.0_f32; n];
        let has_aux =
            frame.in_3.is_some() || frame.in_4.is_some() || self.stage_pipeline.stage0.wdw_mode;
        let mut out_3 = if has_aux {
            Some(vec![0.0_f32; n])
        } else {
            None
        };
        let mut out_4 = if frame.in_3.is_some() || frame.in_4.is_some() {
            Some(vec![0.0_f32; n])
        } else {
            None
        };

        if let Some(ref trajs) = frame.control_trajectories {
            for i in 0..n {
                for (p, tr) in trajs.iter().enumerate() {
                    if p < self.num_controls()
                        && let Some(&v) = tr.get(i)
                    {
                        self.set_control_value(p, v);
                    }
                }
                let s1 = &frame.in_1[i..i + 1];
                let s2 = frame.in_2.as_ref().map(|v| &v[i..i + 1]);
                let s3 = frame.in_3.as_ref().map(|v| &v[i..i + 1]);
                let s4 = frame.in_4.as_ref().map(|v| &v[i..i + 1]);
                self.process_block_io(DnamAudioIo {
                    in_1: s1,
                    in_2: s2,
                    in_3: s3,
                    in_4: s4,
                    out_1: &mut out_1[i..i + 1],
                    out_2: Some(&mut out_2[i..i + 1]),
                    out_3: out_3.as_mut().map(|v| &mut v[i..i + 1]),
                    out_4: out_4.as_mut().map(|v| &mut v[i..i + 1]),
                });
            }
        } else {
            self.process_block_io(DnamAudioIo {
                in_1: &frame.in_1,
                in_2: frame.in_2.as_deref(),
                in_3: frame.in_3.as_deref(),
                in_4: frame.in_4.as_deref(),
                out_1: &mut out_1,
                out_2: Some(&mut out_2),
                out_3: out_3.as_deref_mut(),
                out_4: out_4.as_deref_mut(),
            });
        }

        Ok(StemResponse {
            stem_id: format!("frame_{}", frame.frame_index),
            sample_rate_hz: frame.sample_rate_hz,
            out_1,
            out_2: Some(out_2),
            out_3,
            out_4,
            internal_nodes: BTreeMap::new(),
        })
    }
}

impl<'a> DistillationTarget for DynamicDnamEngine<'a> {
    fn reset_state(&mut self) {
        self.reset_to_dc_equilibrium();
    }

    fn render_stem(&mut self, stimulus: &StemStimulus) -> Result<StemResponse, DnamError> {
        match self {
            Self::Scalar(e) => e.render_stem(stimulus),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.render_stem(stimulus),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.render_stem(stimulus),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.render_stem(stimulus),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.render_stem(stimulus),
        }
    }

    fn render_frame(&mut self, frame: &StimulusFrame) -> Result<StemResponse, DnamError> {
        match self {
            Self::Scalar(e) => e.render_frame(frame),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.render_frame(frame),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.render_frame(frame),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.render_frame(frame),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.render_frame(frame),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dnam_engine::Tier;
    use dnam_engine::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
    use dnam_engine::math::{BackendKind, ScalarBackend};
    use dnam_engine::schema::{ControlKind, ControlSchemaDef, ControlUiHints, DnamJsonHeader};

    struct DummyTarget;
    impl DistillationTarget for DummyTarget {
        fn reset_state(&mut self) {}
        fn render_stem(&mut self, stimulus: &StemStimulus) -> Result<StemResponse, DnamError> {
            Ok(StemResponse {
                stem_id: stimulus.stem_id.clone(),
                sample_rate_hz: stimulus.sample_rate_hz,
                out_1: stimulus.in_1.clone(),
                out_2: stimulus.in_2.clone(),
                out_3: stimulus.in_3.clone(),
                out_4: stimulus.in_4.clone(),
                internal_nodes: BTreeMap::new(),
            })
        }
    }

    #[test]
    fn test_distillation_target_trait_implementations() {
        let mut header = DnamJsonHeader::default_reference();
        header.controls_schema = vec![ControlSchemaDef {
            id: String::from("gain"),
            name: String::from("Gain"),
            kind: ControlKind::Range,
            unit: Some(String::from("dB")),
            lobatto_degree: Some(2),
            anchor_nodes_norm: None,
            declared_range: Some([0.0, 10.0]),
            safe_clamped_range: Some([1.0, 9.0]),
            default: serde_json::json!(5.0),
            conditional_min: BTreeMap::new(),
            conditional_max: BTreeMap::new(),
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            calibration: None,
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        }];
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

        let mut dummy = DummyTarget;
        dummy.reset_state();
        let frame = StimulusFrame {
            frame_index: 7,
            sample_rate_hz: 96_000,
            in_1: vec![0.1, -0.1, 0.2, -0.2],
            in_2: Some(vec![0.05; 4]),
            in_3: Some(vec![0.02; 4]),
            in_4: Some(vec![-0.02; 4]),
            control_values: vec![4.0],
            control_trajectories: Some(vec![vec![3.0, 4.0, 5.0, 6.0]]),
        };
        let dummy_resp = dummy.render_frame(&frame).unwrap();
        assert_eq!(dummy_resp.stem_id, "frame_7");

        let mut eng =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        eng.reset_state();
        let stem = StemStimulus {
            stem_id: String::from("stem_0"),
            sample_rate_hz: 96_000,
            in_1: vec![0.1; 8],
            in_2: None,
            in_3: None,
            in_4: None,
            control_values: vec![5.0],
            control_trajectories: None,
            freeze_transition_window: None,
        };
        let batch = StemStimulusBatch {
            sample_rate_hz: 96_000,
            stems: vec![stem.clone()],
        };
        assert_eq!(batch.stems.len(), 1);
        let stem_resp = eng.render_stem(&stem).unwrap();
        assert_eq!(stem_resp.stem_id, "stem_0");
        assert_eq!(stem_resp.out_1.len(), 8);
        assert!(stem_resp.out_3.is_none());

        let frame_resp = eng.render_frame(&frame).unwrap();
        assert_eq!(frame_resp.stem_id, "frame_7");
        assert!(frame_resp.out_3.is_some());
        assert!(frame_resp.out_4.is_some());

        for backend in [BackendKind::Scalar, BackendKind::Neon, BackendKind::ArmSme] {
            let mut dyn_eng = DynamicDnamEngine::from_bytes_owned_with_backend(
                &container,
                Tier::Lite,
                48_000,
                backend,
            )
            .unwrap();
            dyn_eng.reset_state();
            let r1 = dyn_eng.render_stem(&stem).unwrap();
            assert_eq!(r1.out_1.len(), 8);
            let r2 = dyn_eng.render_frame(&frame).unwrap();
            assert_eq!(r2.out_1.len(), 4);
        }
    }
}
