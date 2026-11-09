//! Real-time host concurrency primitives (`dnam_engine::host`):
//! - 3-Lane lock-free SPSC control ingress ([`SpscControlRing`], [`ControlProducer`],
//!   [`ControlConsumer`]) and synchronous producer-side state shadow ([`AcceptedStateShadow`])
//!   (`ref[dsp.ctrl.op-m.spsc-ingress]`).
//! - Wait-free 64B-aligned Seqlock telemetry cell ([`TelemetryCell`], [`TelemetryWriter`],
//!   [`TelemetryReader`], [`RawTelemetryRecord`]) (`ref[arch.telemetry.seqlock-publication]`).
//! - Non-blocking ping-pong engine hot-swap ([`RealtimeEngineSwap`], [`EngineSwapController`],
//!   [`EngineSwapAudioHandle`]).
//! - Unified thread-safe host handle ([`DnamHostHandle`]) shared across Python, JS/Node-API,
//!   WASM, and native hosts.

pub mod spsc;
pub mod telemetry;

#[cfg(any(test, feature = "std"))]
pub mod handle;
#[cfg(any(test, feature = "std"))]
pub mod swap;

pub use spsc::{
    AcceptedStateShadow, ControlCommand, ControlConsumer, ControlProducer, ControlUpdateMode,
    ControlValueKind, FallbackSnapshot, Lane1Entry, MAX_CONTROLS, RingEntry, SPSC_RING_CAPACITY,
    SpscControlRing, pack_lane1_slot, spsc_control_channel, unpack_lane1_slot,
};
pub use telemetry::{
    RawTelemetryRecord, TelemetryCell, TelemetryReader, TelemetryWriter, telemetry_channel,
};

#[cfg(any(test, feature = "std"))]
pub use handle::{DEFAULT_HOST_SUBBLOCK_FRAMES, DnamHostHandle};
#[cfg(any(test, feature = "std"))]
pub use swap::{
    EngineSwapAudioHandle, EngineSwapController, RealtimeEngineSwap, realtime_engine_swap,
};

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::dynamic::DynamicDnamEngine;
use crate::engine::{DnamAudioIo, DnamEngine};
use crate::math::MathBackend;
use crate::schema::DnamError;

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

/// Coherence-safe `#![no_std] + alloc` target trait implemented by [`DnamEngine`],
/// [`DynamicDnamEngine`], `dnam-fixtures` analytical signal chains, and white-box runners.
pub trait DistillationTarget {
    /// Resets all internal recursive filter, delay, and dynamic control states to resting DC equilibrium.
    fn reset_state(&mut self);

    /// Resets state to DC equilibrium and renders a complete [`StemStimulus`].
    fn render_stem(&mut self, stimulus: &StemStimulus) -> Result<StemResponse, DnamError>;

    /// Renders a single causal [`StimulusFrame`] **without** resetting state, enabling closed-loop
    /// frame-by-frame adaptive feedback excitation (`AdaptiveStimulusController::next_frame`).
    fn render_frame(&mut self, frame: &StimulusFrame) -> Result<StemResponse, DnamError> {
        let stem = StemStimulus {
            stem_id: alloc::format!("frame_{}", frame.frame_index),
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
            stem_id: alloc::format!("frame_{}", frame.frame_index),
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
    use crate::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
    use crate::dynamic::DynamicDnamEngine;
    use crate::math::BackendKind;
    use crate::schema::{
        ControlCalibration, ControlCalibrationRole, ControlKind, ControlSchemaDef, ControlUiHints,
        DnamJsonHeader, PresetDef, TempoSubdivision,
    };
    use crate::types::Tier;
    use alloc::collections::BTreeMap;
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;
    use proptest::prelude::*;

    fn build_test_container_with_controls() -> Vec<u8> {
        let mut header = DnamJsonHeader::default_reference();
        header.io_config.capabilities.supports_freeze = true;
        header.controls_schema = vec![
            ControlSchemaDef {
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
            },
            ControlSchemaDef {
                id: String::from("channel"),
                name: String::from("Channel"),
                kind: ControlKind::Switch,
                unit: None,
                lobatto_degree: None,
                anchor_nodes_norm: None,
                declared_range: None,
                safe_clamped_range: None,
                default: serde_json::json!("Clean"),
                conditional_min: BTreeMap::new(),
                conditional_max: BTreeMap::new(),
                declared_options: Some(vec![
                    String::from("Clean"),
                    String::from("Crunch"),
                    String::from("Lead"),
                ]),
                active_options: Some(vec![String::from("Clean"), String::from("Crunch")]),
                clamp_warning: None,
                calibration: None,
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
            ControlSchemaDef {
                id: String::from("delay_ms"),
                name: String::from("Delay Time"),
                kind: ControlKind::Range,
                unit: Some(String::from("ms")),
                lobatto_degree: Some(2),
                anchor_nodes_norm: None,
                declared_range: Some([20.0, 1000.0]),
                safe_clamped_range: Some([50.0, 800.0]),
                default: serde_json::json!(250.0),
                conditional_min: BTreeMap::new(),
                conditional_max: BTreeMap::new(),
                declared_options: None,
                active_options: None,
                clamp_warning: None,
                calibration: Some(ControlCalibration {
                    role: ControlCalibrationRole::DelayTimeMs,
                    lobatto_physical_anchors: Some([20.0, 510.0, 1000.0]),
                    conditional_physical_anchors: BTreeMap::new(),
                    tempo_sync_default_subdivision: Some(TempoSubdivision::Quarter),
                }),
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
        ];
        let mut lead_controls = BTreeMap::new();
        lead_controls.insert(String::from("gain"), serde_json::json!(8.0));
        lead_controls.insert(String::from("channel"), serde_json::json!("Crunch"));
        header.presets = vec![PresetDef {
            id: String::from("crunch_preset"),
            name: String::from("Crunch Preset"),
            description: None,
            controls: lead_controls,
            user_metadata: BTreeMap::new(),
        }];
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        build_dnam_container(&mut header, &arena, &arena, false).unwrap()
    }

    #[test]
    fn test_spsc_3_lane_coalescing_causal_order_and_overflow_latch() {
        // verify[dsp.ctrl.op-m.spsc-ingress]
        let container = build_test_container_with_controls();
        let mut engine = DynamicDnamEngine::from_bytes_owned_with_backend(
            &container,
            Tier::Ultra,
            48_000,
            BackendKind::Scalar,
        )
        .unwrap();
        let (mut producer, mut consumer) = spsc_control_channel(&engine);

        // 1. Synchronous producer validation & clamping in AcceptedStateShadow
        assert!(producer.try_set_control_value(99, 1.0).is_err());
        assert!(producer.try_set_control_value(0, f32::NAN).is_err());
        assert!(producer.try_set_control_immediate(99, 1.0).is_err());
        assert!(
            producer
                .try_set_control_immediate(0, f32::INFINITY)
                .is_err()
        );
        assert!(producer.reset_control(99).is_err());
        assert!(producer.apply_preset("nonexistent_preset").is_err());

        // High-frequency knob drag on control 0 (1,000 updates before a single drain)
        for i in 0..1_000 {
            let v = (i as f32) * 0.01;
            producer.set_control_value(0, v);
        }
        // 9.99 is clamped to safe_clamped_range [1.0, 9.0] -> 9.0 immediately in shadow!
        assert!((producer.shadow().clamped_control_value(0).unwrap() - 9.0).abs() < 1e-6);
        assert!((producer.shadow().raw_control_values()[0] - 9.99).abs() < 1e-5);
        assert_eq!(producer.overflow_latch_count(), 0);
        assert!(consumer.has_pending());

        let applied = consumer.drain_into(&mut engine);
        assert_eq!(applied, 1); // 1,000 updates coalesced into 1 Lane 1 slot!
        assert!(!consumer.has_pending());
        assert_eq!(
            engine.clamped_control_values(),
            producer.shadow().clamped_control_values()
        );

        // 2. Causal epoch ordering: Lane 1 update BEFORE Lane 2 ResetAllControls vs AFTER
        producer.set_control_value(0, 8.5);
        producer.reset_controls_to_defaults();
        assert!((producer.shadow().clamped_control_value(0).unwrap() - 5.0).abs() < 1e-6);
        consumer.drain_into(&mut engine);
        assert!((engine.clamped_control_values()[0] - 5.0).abs() < 1e-6);

        // Now ResetAllControls BEFORE Lane 1 update:
        producer.reset_controls_to_defaults();
        producer.set_control_immediate(0, 7.5);
        assert!((producer.shadow().clamped_control_value(0).unwrap() - 7.5).abs() < 1e-6);
        consumer.drain_into(&mut engine);
        assert!((engine.clamped_control_values()[0] - 7.5).abs() < 1e-6);

        // 3. Preset application & tempo sync via Lane 2
        producer.apply_preset("crunch_preset").unwrap();
        assert!((producer.shadow().clamped_control_value(0).unwrap() - 8.0).abs() < 1e-6);
        assert!((producer.shadow().clamped_control_value(1).unwrap() - 1.0).abs() < 1e-6);
        let synced = producer.sync_tempo(120.0, Some(TempoSubdivision::Eighth), true);
        assert_eq!(synced, 1);
        assert!((producer.shadow().clamped_control_value(2).unwrap() - 250.0).abs() < 1e-4);
        consumer.drain_into(&mut engine);
        assert_eq!(
            engine.clamped_control_values(),
            producer.shadow().clamped_control_values()
        );

        // 4. Lane 3 Overflow-Proof Full-Reset Epoch Latch: push 150 discrete commands (> 64 ring capacity)
        for i in 0..150 {
            producer.set_bypass(i % 2 == 0, i % 3 == 0);
            producer.set_stereo_width(0.5 + (i as f32) * 0.005);
        }
        let _ = producer.sync_tempo(144.0, Some(TempoSubdivision::Quarter), true);
        producer.set_tier(Tier::Pro);
        producer.set_freeze(true);
        producer.set_control_immediate(0, 6.25);
        assert!(producer.overflow_latch_count() > 0);

        consumer.drain_into(&mut engine);
        assert_eq!(consumer.drained_causal_epoch(), producer.causal_epoch());
        assert_eq!(producer.drained_causal_epoch(), producer.causal_epoch());
        assert_eq!(engine.tier(), producer.shadow().tier());
        assert_eq!(engine.bypass_state(), producer.shadow().bypass_state());
        assert_eq!(engine.is_frozen(), producer.shadow().is_frozen());
        assert!((engine.current_bpm() - producer.shadow().current_bpm()).abs() < 1e-5);
        assert!((engine.stereo_width() - producer.shadow().stereo_width()).abs() < 1e-6);
        assert_eq!(
            engine.telemetry_snapshot().active_clamp_mask,
            producer.shadow().active_clamp_mask()
        );
        assert_eq!(
            engine.clamped_control_values(),
            producer.shadow().clamped_control_values()
        );
    }

    #[test]
    fn test_offline_bounce_preserves_state_across_mid_bounce_control_updates_and_custom_zoh_poles()
    {
        // verify[dsp.ctrl.op-m.spsc-ingress]
        // verify[arch.telemetry.seqlock-publication]
        let mut header = DnamJsonHeader::default_reference();
        header.stage2_cascade.ac_coupling_pole = 0.991;
        header.stage4_tank.dc_blocker_pole = 0.985;
        header.stage4_tank.tzf_reference_tap_samples = 48.0;
        header.stage4_tank.switch_xfade_samples = 96;
        header.controls_schema = vec![ControlSchemaDef {
            id: String::from("gain"),
            name: String::from("Gain"),
            kind: ControlKind::Range,
            unit: Some(String::from("dB")),
            lobatto_degree: Some(2),
            anchor_nodes_norm: None,
            declared_range: Some([0.0, 10.0]),
            safe_clamped_range: Some([0.0, 10.0]),
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
        let arena = vec![0_u8; ARENA_OFFSET_ULTRA_END as usize];
        let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

        let host = DnamHostHandle::from_bytes_owned(&container, Tier::Lite, 48_000).unwrap();
        assert_eq!(host.num_controls(), 1);
        assert_eq!(host.backend_kind(), host.telemetry_record().backend_kind);

        // Verify custom ZOH pole rescaling in AcceptedStateShadow matches engine on offline_bounce toggle
        host.set_offline_bounce(true);
        let shadow_zoh_4x = host.zoh();
        host.flush_pending_controls();
        assert_eq!(shadow_zoh_4x, host.zoh());
        host.set_offline_bounce(false);
        host.set_maximize_tier_on_bounce(false);
        host.flush_pending_controls();
        assert!(!host.offline_bounce());
        assert!(!host.maximize_tier_on_bounce());
        assert_eq!(host.latency_samples(), 0);
        assert_eq!(host.effective_tier(), Tier::Lite);

        // Run render_offline_bounce_with_options(maximize_tier=Some(true)) while another thread
        // pushes control updates mid-bounce, and verify post-bounce state is restored cleanly.
        let host_bounce = host.clone();
        let host_ctrl = host.clone();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let barrier_ctrl = std::sync::Arc::clone(&barrier);

        let bounce_thread = std::thread::spawn(move || {
            let input_l = vec![0.1_f32; 8192];
            let input_r = vec![-0.1_f32; 8192];
            let mut out_l = vec![0.0_f32; 8192];
            let mut out_r = vec![0.0_f32; 8192];
            barrier.wait();
            host_bounce.render_offline_bounce_with_options(
                &input_l,
                &input_r,
                &mut out_l,
                &mut out_r,
                Some(true),
            );
        });

        barrier_ctrl.wait();
        for i in 0..30 {
            host_ctrl.set_control_value(0, 2.0 + (i as f32) * 0.2);
            std::thread::yield_now();
        }
        bounce_thread.join().unwrap();

        assert!(!host.offline_bounce());
        assert!(!host.maximize_tier_on_bounce());
        assert_eq!(host.latency_samples(), 0);
        assert_eq!(host.effective_tier(), Tier::Lite);
    }

    #[test]
    fn test_wait_free_seqlock_telemetry_and_concurrent_host_handle() {
        // verify[arch.telemetry.seqlock-publication]
        // verify[dsp.ctrl.op-m.spsc-ingress]
        let container = build_test_container_with_controls();
        let host = DnamHostHandle::from_bytes_owned(&container, Tier::Ultra, 48_000).unwrap();

        // Single-threaded immediate parity check
        host.set_control_value(0, 7.0);
        assert!((host.clamped_control_value(0).unwrap() - 7.0).abs() < 1e-6);
        let snap = host.telemetry_snapshot();
        assert!((snap.clamped_controls[0] - 7.0).abs() < 1e-6);

        // Concurrent multi-threaded audio + control + wait-free telemetry test
        let host_audio = host.clone();
        let host_ctrl = host.clone();
        let host_telem = host.clone();

        let audio_handle = std::thread::spawn(move || {
            let input_l = vec![0.05_f32; 16_384];
            let input_r = vec![-0.05_f32; 16_384];
            let mut out_l = vec![0.0_f32; 16_384];
            let mut out_r = vec![0.0_f32; 16_384];
            for _ in 0..4 {
                host_audio.process_block_chunked(
                    &input_l,
                    &input_r,
                    &mut out_l,
                    &mut out_r,
                    Some(256),
                );
            }
            assert!(out_l.iter().all(|x| x.is_finite()));
            assert!(out_r.iter().all(|x| x.is_finite()));
        });

        let ctrl_handle = std::thread::spawn(move || {
            for i in 0..200 {
                let start = std::time::Instant::now();
                let v = 1.5 + (i % 70) as f32 * 0.1;
                host_ctrl.set_control_value(0, v);
                host_ctrl.set_bypass(i % 5 == 0, true);
                host_ctrl.set_freeze(i % 7 == 0);
                host_ctrl.set_stereo_width(0.8 + (i % 10) as f32 * 0.1);
                let elapsed = start.elapsed();
                assert!(
                    elapsed < std::time::Duration::from_millis(50),
                    "Control update blocked for {elapsed:?}"
                );
                assert!((host_ctrl.clamped_control_value(0).unwrap() - v).abs() < 1e-5);
            }
        });

        let telem_handle = std::thread::spawn(move || {
            let reader = host_telem.telemetry_reader();
            for _ in 0..500 {
                let rec = reader.read();
                assert!(rec.snapshot.max_pole_radius > 0.0 && rec.snapshot.max_pole_radius <= 1.0);
                assert!((0.0..=2.0).contains(&rec.stereo_width));
                let merged = host_telem.telemetry_record();
                assert_eq!(merged.sample_rate_hz, 48_000);
            }
        });

        ctrl_handle.join().unwrap();
        telem_handle.join().unwrap();
        audio_handle.join().unwrap();

        host.flush_pending_controls();
        let final_rec = host.telemetry_record();
        assert!(final_rec.causal_epoch > 0);
    }

    #[test]
    fn test_realtime_engine_swap_off_thread_reclamation() {
        // verify[dsp.ctrl.op-m.spsc-ingress]
        let container = build_test_container_with_controls();
        let initial = DynamicDnamEngine::from_bytes_owned(&container, Tier::Lite, 48_000).unwrap();
        let (mut controller, mut audio_handle) = realtime_engine_swap(initial);

        assert_eq!(audio_handle.engine().tier(), Tier::Lite);
        assert!(!audio_handle.poll_swap());

        controller
            .stage_reload_from_bytes(&container, Tier::Pro, Tier::Ultra, 96_000)
            .unwrap();
        assert!(audio_handle.poll_swap());
        assert_eq!(audio_handle.engine().tier(), Tier::Pro);
        assert_eq!(audio_handle.engine().sample_rate_hz(), 96_000);
        assert_eq!(audio_handle.swap_epoch(), 1);
        assert_eq!(controller.reclaim_retired(), 1);
        assert_eq!(controller.reclaim_retired(), 0);
    }

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
        let container = build_test_container_with_controls();
        let mut dummy = DummyTarget;
        dummy.reset_state();
        let frame = StimulusFrame {
            frame_index: 7,
            sample_rate_hz: 96_000,
            in_1: vec![0.1, -0.1, 0.2, -0.2],
            in_2: Some(vec![0.05; 4]),
            in_3: Some(vec![0.02; 4]),
            in_4: Some(vec![-0.02; 4]),
            control_values: vec![4.0, 1.0, 200.0],
            control_trajectories: Some(vec![vec![3.0, 4.0, 5.0, 6.0]]),
        };
        let dummy_resp = dummy.render_frame(&frame).unwrap();
        assert_eq!(dummy_resp.stem_id, "frame_7");

        let mut eng = crate::engine::DnamEngine::<crate::math::ScalarBackend>::from_bytes(
            &container,
            Tier::Lite,
            48_000,
        )
        .unwrap();
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

    proptest! {
        #[test]
        fn prop_spsc_shadow_and_engine_converge(
            ops in proptest::collection::vec((0u8..8, -5.0f32..15.0f32), 1..100)
        ) {
            // verify[dsp.ctrl.op-m.spsc-ingress]
            // verify[arch.telemetry.seqlock-publication]
            let container = build_test_container_with_controls();
            let mut engine = DynamicDnamEngine::from_bytes_owned_with_backend(
                &container,
                Tier::Ultra,
                48_000,
                BackendKind::Scalar,
            )
            .unwrap();
            let (mut producer, mut consumer) = spsc_control_channel(&engine);

            for (op, val) in ops {
                match op {
                    0 => producer.set_control_value(0, val),
                    1 => producer.set_control_immediate(0, val),
                    2 => producer.set_control_immediate(1, (val.abs() as u32 % 4) as f32),
                    3 => producer.set_bypass(val > 0.0, val > 5.0),
                    4 => producer.set_freeze(val > 2.0),
                    5 => producer.set_stereo_width(val),
                    6 => {
                        let _ = producer.sync_tempo(val.abs().clamp(30.0, 300.0), None, true);
                    }
                    _ => producer.reset_controls_to_defaults(),
                }
            }

            consumer.drain_into(&mut engine);
            prop_assert_eq!(consumer.drained_causal_epoch(), producer.causal_epoch());
            prop_assert_eq!(
                engine.clamped_control_values(),
                producer.shadow().clamped_control_values()
            );
            prop_assert_eq!(engine.bypass_state(), producer.shadow().bypass_state());
            prop_assert_eq!(engine.is_frozen(), producer.shadow().is_frozen());
            prop_assert!((engine.current_bpm() - producer.shadow().current_bpm()).abs() < 1e-5);
            prop_assert!((engine.stereo_width() - producer.shadow().stereo_width()).abs() < 1e-6);
        }
    }
}
