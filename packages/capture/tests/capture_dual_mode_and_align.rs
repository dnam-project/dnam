#![forbid(unsafe_code)]
//! Verification suite for Phase-0 `CALB` loopback deconvolution, 4-regime sub-sample Kaiser-sinc
//! alignment (`<= 0.02` samples), `< 0.05 dB` live acoustic null homing, `<= 0.15 dB` thermal
//! recovery gate, `Theorem 4.5–4.6` multi-port & intra-sweep laws, `Mode 1` streaming + `1.0s`
//! Live Null-Test / `< 5 ms` Auto-Patch, `400 ms` mic-pad `recapture_clipped_stem`, and `Mode 2`
//! 1-or-2-pass Sans-IO batch distillation.
// verify[train.capture.calib.loopback]
// verify[train.capture.calib.clean-pad]
// verify[train.capture.alignment.multi-regime]
// verify[train.capture.planner.null-homing]
// verify[train.capture.planner.schedule]
// verify[train.cache.proj-vault]

use std::collections::BTreeMap;

use dnam_capture::align::{
    AlignmentRegime, CALB_STIMULUS_SAMPLES_48K, DecayTailDecision, MIC_PAD_PROBE_SAMPLES_48K,
    calibrate_loopback_calb, estimate_multiregime_alignment, evaluate_acoustic_null_homing,
    evaluate_live_decay_tail, evaluate_thermal_recovery_gate, generate_calb_loopback_stimulus,
    generate_mic_pad_probe_400ms, measure_differential_pad_ratio, realign_fractional_kaiser64,
    rescale_padded_stem,
};
use dnam_capture::proj::{
    CalbChunk64B, DnamProjManifest, DnamProjView, compact_to_capsules, export_runtime_dnam,
    parse_proj_manifest_range,
};
use dnam_capture::stimulus::{
    AdaptiveStimulusController, AdaptiveTrainingOptions, BatchStimulusOptions,
    Disambiguated2To1Mode, HadamardLaw4dTracker, KinematicAlphaBetaPoleTracker, KnobAcousticClass,
    StimulusConfig, SweepExecutionMode, TopologyProbeMode, classify_knob_acoustic_class,
    deterministic_seed_256, generate_refinement_stimulus_batch, generate_stimulus_batch,
    is_master_or_output_control, recapture_clipped_stem, run_live_null_test_and_autopatch,
    select_coupled_hadamard_pairs, select_sweep_execution_mode, solve_from_responses,
    synthesize_a5_musical_transient_coreset, train_adaptive_target,
};
use dnam_capture::{
    CaptureBitDepth, DistillationTarget, StemResponse, StemStimulus, StemStimulusBatch,
    StimulusFrame,
};
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::schema::{ControlSchemaDef, DnamError};
use dnam_engine::{DnamEngine, Tier};
use dnam_fixtures::{
    BlackBoxModel, Chain01SymmetricOd, Chain05PureLtiCabAndRoom, Chain07StereoChorusAndRotary,
    Chain09La2aOpticalCompressor, Chain10Oc2OctaveDividerAndRingMod, ConverterLoopbackDecorator,
    HardwareNoiseGateDecorator, PolarityAndDcOffsetDecorator, SpringChirpDecorator,
    TubeThermalHysteresisDecorator,
};
use proptest::prelude::*;

struct SimpleFixtureTarget<M: BlackBoxModel> {
    model: M,
    switches: Vec<u8>,
}

impl<M: BlackBoxModel> SimpleFixtureTarget<M> {
    fn new(model: M) -> Self {
        let switches = model.metadata().default_switches();
        Self { model, switches }
    }
}

impl<M: BlackBoxModel> DistillationTarget for SimpleFixtureTarget<M> {
    fn reset_state(&mut self) {
        self.model.reset();
    }

    fn render_stem(&mut self, stim: &StemStimulus) -> Result<StemResponse, DnamError> {
        let n = stim.in_1.len();
        let n_in = self.model.num_inputs().clamp(1, 4);
        let n_out = self.model.num_outputs().clamp(1, 4);
        let mut out_1 = Vec::with_capacity(n);
        let mut out_2 = (n_out >= 2).then(|| Vec::with_capacity(n));
        let mut in_b = [0.0_f32; 4];
        let mut out_b = [0.0_f32; 4];
        let mut cur_c = stim.control_values.clone();
        for i in 0..n {
            in_b[0] = stim.in_1[i];
            in_b[1] = stim.in_2.as_ref().map_or(stim.in_1[i], |r| r[i]);
            in_b[2] = stim.in_3.as_ref().map_or(0.0, |r| r[i]);
            if let Some(ref trajs) = stim.control_trajectories {
                for (p, tr) in trajs.iter().enumerate() {
                    if p < cur_c.len()
                        && let Some(&v) = tr.get(i)
                    {
                        cur_c[p] = v;
                    }
                }
            }
            out_b.fill(0.0);
            self.model
                .process_frame(&in_b[..n_in], &mut out_b[..n_out], &cur_c, &self.switches);
            out_1.push(out_b[0]);
            if let Some(ref mut r) = out_2 {
                r.push(out_b[1]);
            }
        }
        Ok(StemResponse {
            stem_id: stim.stem_id.clone(),
            sample_rate_hz: stim.sample_rate_hz,
            out_1,
            out_2,
            out_3: None,
            out_4: None,
            internal_nodes: BTreeMap::new(),
        })
    }

    fn render_frame(&mut self, frame: &StimulusFrame) -> Result<StemResponse, DnamError> {
        self.render_stem(&StemStimulus {
            stem_id: format!("frame_{}", frame.frame_index),
            sample_rate_hz: frame.sample_rate_hz,
            in_1: frame.in_1.clone(),
            in_2: frame.in_2.clone(),
            in_3: frame.in_3.clone(),
            in_4: frame.in_4.clone(),
            control_values: frame.control_values.clone(),
            control_trajectories: frame.control_trajectories.clone(),
            freeze_transition_window: None,
        })
    }
}

fn render_batch<T: DistillationTarget>(
    target: &mut T,
    batch: &StemStimulusBatch,
) -> Vec<StemResponse> {
    batch
        .stems
        .iter()
        .map(|s| target.render_stem(s).expect("render stem"))
        .collect()
}

fn sample_calb() -> CalbChunk64B {
    CalbChunk64B {
        sample_rate_device_hz: 48_000,
        bit_depth_code: 4,
        polarity_inverted: false,
        loopback_lag_samples: 0.0,
        signed_gain: 1.0,
        input_reference_dbfs: -20.5,
        noise_floor_dbfs: -100.0,
    }
}

fn make_range_control(id: &str, min_v: f32, max_v: f32, def_v: f32) -> ControlSchemaDef {
    serde_json::from_value(serde_json::json!({
        "id": id,
        "name": id,
        "type": "range",
        "lobatto_degree": 2,
        "declared_range": [min_v, max_v],
        "safe_clamped_range": [min_v, max_v],
        "default": def_v
    }))
    .unwrap()
}

fn make_switch_control(id: &str) -> ControlSchemaDef {
    serde_json::from_value(serde_json::json!({
        "id": id,
        "name": id,
        "type": "switch",
        "declared_options": ["Vintage", "Modern"],
        "active_options": ["Vintage", "Modern"],
        "default": "Vintage"
    }))
    .unwrap()
}

#[derive(Default)]
struct IdentityWire;
impl BlackBoxModel for IdentityWire {
    fn reset(&mut self) {}
    fn metadata(&self) -> dnam_fixtures::FixtureDeviceMetadata {
        Chain05PureLtiCabAndRoom::new().metadata()
    }
    fn set_sample_rate_hz(&mut self, _: u32) {}
    fn set_bit_depth(&mut self, _: dnam_fixtures::FixtureBitDepth) {}
    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn process_frame(&mut self, inputs: &[f32], outputs: &mut [f32], _: &[f32], _: &[u8]) {
        if !outputs.is_empty() && !inputs.is_empty() {
            outputs[0] = inputs[0];
        }
    }
}

#[test]
fn test_calb_loopback_deconvolution_and_converter_clip_detection() {
    // verify[train.capture.calib.loopback]
    assert!(generate_calb_loopback_stimulus(1_000).is_err());
    let stim = generate_calb_loopback_stimulus(48_000).expect("valid 48k CALB stimulus");
    assert_eq!(stim.len(), CALB_STIMULUS_SAMPLES_48K);

    let mut conv = ConverterLoopbackDecorator::new(IdentityWire);
    conv.clock_drift_ppm = 0.0;
    let mut wet = Vec::with_capacity(stim.len());
    for &x in &stim {
        let mut out = [0.0_f32];
        conv.process_frame(&[x], &mut out, &[], &[]);
        wet.push(out[0]);
    }

    let report = calibrate_loopback_calb(&stim, &wet, 48_000, CaptureBitDepth::Float32, -20.5, 512)
        .expect("CALB calibration");
    assert!(
        (report.calb.loopback_lag_samples - 143.62).abs() <= 0.15,
        "expected ~143.62 sample loopback lag, got {}",
        report.calb.loopback_lag_samples
    );
    assert!(!report.calb.polarity_inverted);
    assert!(!report.converter_clipped);
    assert!(report.peak_dbfs < -0.1);

    let calb_bytes = report.calb.to_bytes();
    let decoded = CalbChunk64B::from_bytes(&calb_bytes).expect("CALB 64B roundtrip");
    assert_eq!(decoded, report.calb);

    // Verify converter clip warning (>= -0.1 dBFS = 0.98855)
    let mut clipped_wet = wet.clone();
    clipped_wet[500] = 0.995;
    let clip_rep = calibrate_loopback_calb(
        &stim,
        &clipped_wet,
        48_000,
        CaptureBitDepth::Float32,
        -20.5,
        512,
    )
    .expect("clipped CALB");
    assert!(clip_rep.converter_clipped);

    // Error paths
    assert!(
        calibrate_loopback_calb(
            &stim[..32],
            &wet[..32],
            48_000,
            CaptureBitDepth::Float32,
            -20.5,
            64,
        )
        .is_err()
    );
    assert!(
        calibrate_loopback_calb(
            &vec![0.0; 256],
            &wet[..256],
            48_000,
            CaptureBitDepth::Float32,
            -20.5,
            64,
        )
        .is_err()
    );
}

#[test]
fn test_four_regime_subsample_alignment_and_kaiser64_realignment() {
    // verify[train.capture.alignment.multi-regime]
    let seed = deterministic_seed_256(99);
    let (dry, _) = synthesize_a5_musical_transient_coreset(&seed, 16_384, 1, 48_000);

    // 1. Regime 1: CleanLinear with exact fractional delay +37.38 samples (apply -37.38 advance^-1 = delay +37.38)
    let true_lag = 37.38_f64;
    let delayed_clean = realign_fractional_kaiser64(&dry, -true_lag, 0.0, false);
    let al_clean =
        estimate_multiregime_alignment(&dry, &delayed_clean, 48_000, 256).expect("CleanLinear");
    assert_eq!(al_clean.regime, AlignmentRegime::CleanLinear);
    assert!(!al_clean.polarity_inverted);
    assert!(
        (al_clean.total_lag_samples - true_lag).abs() <= 0.02,
        "CleanLinear lag error {:.5} exceeded 0.02 samples (got {})",
        (al_clean.total_lag_samples - true_lag).abs(),
        al_clean.total_lag_samples
    );

    // Realign back and verify sample-level nulling in interior
    let recovered = realign_fractional_kaiser64(
        &delayed_clean,
        al_clean.total_lag_samples,
        al_clean.clock_drift_ppm,
        al_clean.polarity_inverted,
    );
    let mut err_e = 0.0_f64;
    let mut ref_e = 0.0_f64;
    for i in 256..(dry.len() - 256) {
        let d = f64::from(dry[i]) - f64::from(recovered[i]);
        err_e += d * d;
        ref_e += f64::from(dry[i]) * f64::from(dry[i]);
    }
    let null_db = 10.0 * (err_e / ref_e).log10();
    assert!(
        null_db < -45.0,
        "expected < -45 dB Kaiser-64 null, got {null_db:.2} dB"
    );

    // 2. Regime 2: PolarityInverted + DC offset (`PolarityAndDcOffsetDecorator`)
    let mut pol_dec = PolarityAndDcOffsetDecorator::new(IdentityWire, true, 0.08);
    let mut inv_delayed = Vec::with_capacity(delayed_clean.len());
    for &x in &delayed_clean {
        let mut o = [0.0_f32];
        pol_dec.process_frame(&[x], &mut o, &[], &[]);
        inv_delayed.push(o[0]);
    }
    let al_inv =
        estimate_multiregime_alignment(&dry, &inv_delayed, 48_000, 256).expect("PolarityInverted");
    assert_eq!(al_inv.regime, AlignmentRegime::PolarityInverted);
    assert!(al_inv.polarity_inverted);
    assert!((al_inv.total_lag_samples - true_lag).abs() <= 0.02);

    // 3. Regime 3: DispersiveAllPass (`SpringChirpDecorator` 16-stage all-pass cascade)
    let mut spring = SpringChirpDecorator::new(IdentityWire);
    // Use pure all-pass output (`(spring_out - 0.75 * x) / 0.25`) delayed by 24.25 samples
    let delayed_24 = realign_fractional_kaiser64(&dry, -24.25, 0.0, false);
    let mut ap_wet = Vec::with_capacity(delayed_24.len());
    for &x in &delayed_24 {
        let mut o = [0.0_f32];
        spring.process_frame(&[x], &mut o, &[], &[]);
        ap_wet.push((o[0] - 0.75 * x) * 4.0);
    }
    let al_disp =
        estimate_multiregime_alignment(&dry, &ap_wet, 48_000, 256).expect("DispersiveAllPass");
    assert_eq!(al_disp.regime, AlignmentRegime::DispersiveAllPass);
    assert!(
        (23.0..=31.0).contains(&al_disp.total_lag_samples),
        "expected dispersive causal wavefront lag in [23.0, 31.0], got {}",
        al_disp.total_lag_samples
    );

    // 4. Regime 4: NonlinearEnvelopeFallback (full-wave squaring destroys linear cross-correlation!)
    let squared_wet: Vec<f32> = delayed_clean.iter().map(|&x| 3.5 * x * x).collect();
    let al_nl = estimate_multiregime_alignment(&dry, &squared_wet, 48_000, 256)
        .expect("NonlinearEnvelopeFallback");
    assert_eq!(al_nl.regime, AlignmentRegime::NonlinearEnvelopeFallback);
    assert!(
        (al_nl.total_lag_samples - true_lag).abs() <= 1.5,
        "envelope fallback lag {} far from {true_lag}",
        al_nl.total_lag_samples
    );

    // 5. Clock-drift estimation (+45 ppm over 48,000 samples)
    let (long_dry, _) = synthesize_a5_musical_transient_coreset(&seed, 48_000, 1, 48_000);
    let drifted_wet = realign_fractional_kaiser64(&long_dry, -18.4, -45.0, false);
    let al_drift =
        estimate_multiregime_alignment(&long_dry, &drifted_wet, 48_000, 128).expect("clock drift");
    assert!(
        (al_drift.clock_drift_ppm - 45.0).abs() <= 8.0,
        "expected ~+45 ppm clock drift, got {}",
        al_drift.clock_drift_ppm
    );
}

#[test]
fn test_null_homing_thermal_recovery_and_live_decay_tail_gates() {
    // verify[train.capture.planner.null-homing]
    let ref_wave: Vec<f32> = (0..4_096).map(|n| 0.3 * (n as f32 * 0.07).sin()).collect();
    let near_exact: Vec<f32> = ref_wave.iter().map(|&x| x * 1.002).collect(); // +0.017 dB (< 0.05 dB)
    let homed = evaluate_acoustic_null_homing(&ref_wave, &near_exact, 0.0).expect("null homing");
    assert!(homed.homed && homed.error_db < 0.05);

    let mis_homed: Vec<f32> = ref_wave.iter().map(|&x| x * 1.05).collect(); // +0.42 dB (> 0.05 dB)
    let bad_home = evaluate_acoustic_null_homing(&ref_wave, &mis_homed, 0.0).expect("bad homing");
    assert!(!bad_home.homed);

    // Thermal recovery gate (<= 0.15 dB)
    let recovered: Vec<f32> = ref_wave.iter().map(|&x| x * 0.992).collect(); // -0.07 dB
    let th_ok = evaluate_thermal_recovery_gate(&ref_wave, &recovered, 0.0).expect("thermal ok");
    assert!(th_ok.recovered);

    let mut sag_rig = TubeThermalHysteresisDecorator::new(Chain01SymmetricOd::new());
    sag_rig.thermal_state = 0.5; // Hot cathode drift!
    let mut hot_wave = Vec::with_capacity(ref_wave.len());
    let mut cold_rig = Chain01SymmetricOd::new();
    let mut cold_wave = Vec::with_capacity(ref_wave.len());
    for &x in &ref_wave {
        let (mut o_hot, mut o_cold) = ([0.0_f32], [0.0_f32]);
        sag_rig.process_frame(&[x], &mut o_hot, &[0.5, 0.5, 0.5], &[]);
        cold_rig.process_frame(&[x], &mut o_cold, &[0.5, 0.5, 0.5], &[]);
        hot_wave.push(o_hot[0]);
        cold_wave.push(o_cold[0]);
    }
    let th_hot = evaluate_thermal_recovery_gate(&cold_wave, &hot_wave, 0.0).expect("thermal hot");
    assert!(!th_hot.recovered && th_hot.drift_db > 0.15);

    // Live decay tail monitor
    let loud_tail = vec![0.02_f32; 4_608];
    assert!(matches!(
        evaluate_live_decay_tail(&loud_tail, -90.0, 0, 48_000),
        DecayTailDecision::Extend { .. }
    ));
    let quiet_tail = vec![1.0e-6_f32; 4_608];
    assert!(matches!(
        evaluate_live_decay_tail(&quiet_tail, -90.0, 0, 48_000),
        DecayTailDecision::Complete {
            ceiling_reached: false,
            ..
        }
    ));
}

#[test]
fn test_theorem_4_5_multiport_stereo_link_2to1_aux_key_and_4cm() {
    // 1. TrueStereo2To2 (`2 -> 2` Mid / Side / Hard-Pan L + Quiet Witness R -> alpha_link)
    let mut chorus = SimpleFixtureTarget::new(Chain07StereoChorusAndRotary::new());
    let mut ctrl_2to2 = AdaptiveStimulusController::new(StimulusConfig {
        topology_mode: TopologyProbeMode::TrueStereo2To2,
        max_frames: 6,
        max_tail_frames: 0,
        ..StimulusConfig::default()
    });
    let mut r22 = None;
    while let Some(fr) = ctrl_2to2.next_frame(r22.as_ref()) {
        r22 = Some(chorus.render_frame(&fr).expect("render 2->2"));
    }
    let alpha = ctrl_2to2
        .stereo_detector_link()
        .expect("alpha_link estimated");
    assert!((0.0..=1.0).contains(&alpha));

    // 2. DualInToMono2To1 (`2 -> 1` Frame-0 23.44 Hz AM disambiguating Aux-Key Sidechain vs Input Sum)
    let mut la2a = SimpleFixtureTarget::new(Chain09La2aOpticalCompressor::new());
    let mut ctrl_sc = AdaptiveStimulusController::new(StimulusConfig {
        topology_mode: TopologyProbeMode::DualInToMono2To1,
        control_values: vec![0.85, 0.7],
        max_frames: 5,
        max_tail_frames: 0,
        ..StimulusConfig::default()
    });
    // Switch LA-2A external sidechain key ON (`switches[1] = 1`)
    la2a.switches = vec![0, 1];
    let f_sc0 = ctrl_sc.next_frame(None).expect("f_sc0");
    let r_sc0 = la2a.render_frame(&f_sc0).expect("render LA-2A sidechain");
    let _ = ctrl_sc.next_frame(Some(&r_sc0));
    assert_eq!(
        ctrl_sc.disambiguated_2to1_mode(),
        Some(Disambiguated2To1Mode::ExternalAuxKeySidechain)
    );

    // Linear dual-input sum (`y = in_1 + in_2`) -> DualAudioInputSum
    let mut ctrl_sum = AdaptiveStimulusController::new(StimulusConfig {
        topology_mode: TopologyProbeMode::DualInToMono2To1,
        max_frames: 4,
        max_tail_frames: 0,
        ..StimulusConfig::default()
    });
    let f_s0 = ctrl_sum.next_frame(None).expect("f_s0");
    let sum_out: Vec<f32> = f_s0
        .in_1
        .iter()
        .zip(f_s0.in_2.as_ref().unwrap())
        .map(|(&a, &b)| 0.5 * (a + b))
        .collect();
    let _ = ctrl_sum.next_frame(Some(&StemResponse {
        stem_id: "sum".to_string(),
        sample_rate_hz: 48_000,
        out_1: sum_out,
        out_2: None,
        out_3: None,
        out_4: None,
        internal_nodes: BTreeMap::new(),
    }));
    assert_eq!(
        ctrl_sum.disambiguated_2to1_mode(),
        Some(Disambiguated2To1Mode::DualAudioInputSum)
    );

    // 3. FourCableMethod4Cm + 96 kHz non-48k automatic resampling
    let mut ctrl_4cm = AdaptiveStimulusController::new(StimulusConfig {
        sample_rate_hz: 96_000,
        topology_mode: TopologyProbeMode::FourCableMethod4Cm,
        max_frames: 4,
        max_tail_frames: 0,
        ..StimulusConfig::default()
    });
    let f_4cm = ctrl_4cm.next_frame(None).expect("96k 4CM frame");
    assert_eq!(f_4cm.sample_rate_hz, 96_000);
    assert_eq!(f_4cm.in_1.len(), 9_216);
    assert!(f_4cm.in_3.is_some());
    let r_96k = StemResponse {
        stem_id: "96k".to_string(),
        sample_rate_hz: 96_000,
        out_1: f_4cm.in_1.clone(),
        out_2: None,
        out_3: None,
        out_4: None,
        internal_nodes: BTreeMap::new(),
    };
    let f_4cm_1 = ctrl_4cm.next_frame(Some(&r_96k)).expect("96k frame 1");
    assert_eq!(f_4cm_1.in_1.len(), 9_216);
}

#[test]
fn test_theorem_4_6_control_classes_pole_tracker_and_hadamard_pruning() {
    assert_eq!(
        classify_knob_acoustic_class("drive_gain"),
        KnobAcousticClass::ClassIGainDrive
    );
    assert_eq!(
        classify_knob_acoustic_class("treble_eq"),
        KnobAcousticClass::ClassIIToneFilter
    );
    assert_eq!(
        classify_knob_acoustic_class("bbd_delay_ms"),
        KnobAcousticClass::ClassIIIPitchDelayTime
    );
    assert_eq!(
        classify_knob_acoustic_class("master_level"),
        KnobAcousticClass::ClassIVSpatialMixLevel
    );
    assert!(is_master_or_output_control("master_volume"));
    assert!(!is_master_or_output_control("preamp_gain"));

    assert_eq!(
        select_sweep_execution_mode(8.0, KnobAcousticClass::ClassIGainDrive, false),
        SweepExecutionMode::ModeAContinuousSweep
    );
    assert_eq!(
        select_sweep_execution_mode(25.0, KnobAcousticClass::ClassIGainDrive, false),
        SweepExecutionMode::ModeBStationaryStops
    );
    assert_eq!(
        select_sweep_execution_mode(5.0, KnobAcousticClass::ClassIIIPitchDelayTime, false),
        SweepExecutionMode::ModeBStationaryStops
    );
    assert_eq!(
        select_sweep_execution_mode(5.0, KnobAcousticClass::ClassIGainDrive, true),
        SweepExecutionMode::ModeBStationaryStops
    );

    let mut tracker = KinematicAlphaBetaPoleTracker::default();
    let p0 = tracker.update_and_predict(400.0);
    let p1 = tracker.update_and_predict(650.0);
    assert!((p0 - 400.0).abs() < 1.0e-6 && p1 > 450.0 && tracker.velocity_hz_per_frame > 0.0);

    let mut h_sep = HadamardLaw4dTracker::default();
    for f in 0..11 {
        assert!(
            !h_sep.observe_frame(1.0e-5),
            "should not stop before 12 frames (f={f})"
        );
    }
    assert!(
        h_sep.observe_frame(1.0e-5),
        "must stop at 12th frame (1.152s) when separable"
    );

    let mut h_coupled = HadamardLaw4dTracker::default();
    for _ in 0..14 {
        assert!(!h_coupled.observe_frame(0.15));
    }

    let pairs = select_coupled_hadamard_pairs(&[0, 1, 2], &[0.001, 0.20, 0.25]);
    assert_eq!(pairs, vec![(1, 2)]);
}

#[test]
fn test_mode1_streaming_live_null_test_autopatch_and_mic_pad_recapture() {
    // verify[train.capture.planner.schedule]
    // verify[train.capture.calib.clean-pad]
    // verify[train.cache.proj-vault]
    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.controls_schema = vec![
        make_range_control("gain", 0.0, 1.0, 0.5),
        make_range_control("master_volume", 0.0, 1.0, 0.6),
        make_switch_control("voice"),
    ];

    let mut target = SimpleFixtureTarget::new(Chain01SymmetricOd::new());
    let opts = AdaptiveTrainingOptions {
        sample_rate_hz: 48_000,
        topology_mode: TopologyProbeMode::Mono1To1,
        fixed_duration: false,
        prefer_mode_b: false,
        max_frames_per_stem: 6,
        run_live_null_test: true,
        null_test_autopatch_threshold_db: -60.0, // Force < 5 ms STAT auto-patch path!
        calb: Some(sample_calb()),
    };

    let report = train_adaptive_target(manifest.clone(), &mut target, &opts)
        .expect("Mode 1 train_adaptive_target");
    assert!(report.stem_count >= 4);
    let nt = report.null_test.expect("LiveNullTestReport present");
    assert!(nt.reload_in_place_verified);
    let under_llvm_cov = std::env::var_os("LLVM_PROFILE_FILE").is_some();
    let max_patch_ms = if under_llvm_cov { 500.0 } else { 25.0 };
    assert!(
        nt.autopatch_elapsed_ms < max_patch_ms,
        "sufficient-statistic Cholesky patch took {:.3} ms (ceiling {max_patch_ms} ms)",
        nt.autopatch_elapsed_ms
    );
    assert!(nt.post_patch_null_esr_db <= nt.pre_patch_null_esr_db + 0.05);

    // Also verify skip-autopatch fast path when threshold is generous (e.g. +20 dB)
    let (_, _, nt_skip) =
        run_live_null_test_and_autopatch(&report.proj_bytes, &mut target, 20.0).expect("skip path");
    assert!(!nt_skip.autopatch_applied && nt_skip.reload_in_place_verified);

    // Validate .dnam-proj HTTP Range manifest + compact_to_capsules + DnamEngine playback
    let range_manifest =
        parse_proj_manifest_range(&report.proj_bytes[..report.proj_bytes.len().min(65_536)])
            .expect("HTTP Range parse");
    assert_eq!(
        range_manifest.manifest.stems_registry.len(),
        report.stem_count
    );

    let compacted = compact_to_capsules(&report.proj_bytes).expect("compact_to_capsules");
    let dnam_bytes = export_runtime_dnam(&compacted, None).expect("export_runtime_dnam");
    let mut eng =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_bytes, Tier::Ultra, 48_000)
            .expect("DnamEngine load");
    let [yl, yr] = eng.step_frame_tora(0.25, 0.25, false);
    assert!(yl.is_finite() && yr.is_finite());

    // Verify 400 ms -18 dB differential mic-pad calibration & recapture_clipped_stem
    let probe_dry = generate_mic_pad_probe_400ms(48_000);
    assert_eq!(probe_dry.len(), MIC_PAD_PROBE_SAMPLES_48K);
    let pad_gain_true = 0.251_188_64_f32; // -12.0 dB physical pad
    let unpadded_wet: Vec<f32> = probe_dry.iter().map(|&x| 0.8 * x).collect();
    let padded_wet: Vec<f32> = unpadded_wet.iter().map(|&x| x * pad_gain_true).collect();

    let pad_cal =
        measure_differential_pad_ratio(&unpadded_wet, &padded_wet, 64).expect("pad ratio");
    assert!((pad_cal.pad_attenuation_db - (-12.0)).abs() <= 0.05);
    let rescaled = rescale_padded_stem(&padded_wet, pad_cal.pad_gain_linear).expect("rescale");
    let max_err = rescaled
        .iter()
        .zip(&unpadded_wet)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_err < 1.0e-4);

    // Replace stem in .dnam-proj via recapture_clipped_stem
    let mut ctrl_pad = AdaptiveStimulusController::new(StimulusConfig {
        stem_id: "sweep_ctrl_1".to_string(),
        max_frames: 6,
        max_tail_frames: 0,
        ..StimulusConfig::default()
    });
    let mut rp = None;
    while let Some(fr) = ctrl_pad.next_frame(rp.as_ref()) {
        let mut r = target.render_frame(&fr).unwrap();
        for s in &mut r.out_1 {
            *s *= pad_gain_true;
        }
        rp = Some(r);
    }
    let mut meta_map = BTreeMap::new();
    meta_map.insert("gain".to_string(), serde_json::json!(0.5));
    meta_map.insert("master_volume".to_string(), serde_json::json!(1.0));
    meta_map.insert("voice".to_string(), serde_json::json!("Vintage"));
    let (_, padded_spec) = ctrl_pad
        .seal_inpc_and_vault_spec(false, meta_map)
        .expect("padded spec");
    let (re_proj, cal_out) = recapture_clipped_stem(
        &report.proj_bytes,
        "sweep_ctrl_1",
        &unpadded_wet,
        &padded_wet,
        padded_spec,
    )
    .expect("recapture_clipped_stem");
    assert!((cal_out.pad_attenuation_db - (-12.0)).abs() <= 0.05);
    let crypto = DefaultCryptoProvider::default();
    let re_view = DnamProjView::from_bytes(&re_proj, &crypto).expect("valid updated .dnam-proj");
    assert_eq!(re_view.manifest.stems_registry.len(), report.stem_count);

    // Also exercise Mode 1 with `prefer_mode_b: true`
    let mut manifest_b = DnamProjManifest::default_reference();
    manifest_b.base.controls_schema = vec![make_range_control("gain", 0.0, 1.0, 0.5)];
    let rep_b = train_adaptive_target(
        manifest_b,
        &mut target,
        &AdaptiveTrainingOptions {
            prefer_mode_b: true,
            max_frames_per_stem: 6,
            run_live_null_test: false,
            ..opts
        },
    )
    .expect("Mode 1 prefer_mode_b");
    assert_eq!(rep_b.stem_count, 3); // StarCenter + 2 stationary stops
}

#[test]
fn test_mode2_sans_io_batch_1_or_2_pass_and_refinement_triggers() {
    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.controls_schema = vec![make_range_control("drive", 0.0, 1.0, 0.5)];

    let batch1 = generate_stimulus_batch(
        &manifest,
        &BatchStimulusOptions {
            prefer_mode_b: false,
            ..BatchStimulusOptions::default()
        },
    )
    .expect("Mode 2 Pass 1 batch");
    assert_eq!(batch1.stems.len(), 2);

    // 1. Clean 1-pass solve on Chain01SymmetricOd (refinement == None)
    let mut od_target = SimpleFixtureTarget::new(Chain01SymmetricOd::new());
    let pass1_resps = render_batch(&mut od_target, &batch1);
    let ref_od = generate_refinement_stimulus_batch(&manifest, &batch1, &pass1_resps)
        .expect("refinement check");
    assert!(
        ref_od.is_none(),
        "well-excited OD should not require Pass 2"
    );

    let solved_1pass = solve_from_responses(
        manifest.clone(),
        &[(&batch1, &pass1_resps)],
        Some(sample_calb()),
    )
    .expect("solve_from_responses 1-pass");
    assert_eq!(solved_1pass.stem_count, 2);
    assert!(!solved_1pass.runtime_dnam_bytes.is_empty());

    // 2. Pass-2 refinement trigger on Chain10 OC-2 sub-harmonic divider (`Law 0.E`)
    let mut oc2_target = SimpleFixtureTarget::new(Chain10Oc2OctaveDividerAndRingMod::new());
    oc2_target.switches = vec![0];
    let oc2_resps: Vec<StemResponse> = batch1
        .stems
        .iter()
        .map(|s| {
            let mut s_mod = s.clone();
            s_mod.control_values = vec![1.0, 0.8, 0.0];
            oc2_target.render_stem(&s_mod).unwrap()
        })
        .collect();
    let ref_oc2 = generate_refinement_stimulus_batch(&manifest, &batch1, &oc2_resps)
        .expect("OC-2 refinement")
        .expect("OC-2 must request Pass-2 A3 glissando refinement");
    assert_eq!(ref_oc2.stems[0].stem_id, "refinement_oc2_glissando");
    let pass2_resps = render_batch(&mut oc2_target, &ref_oc2);
    let solved_2pass = solve_from_responses(
        manifest.clone(),
        &[(&batch1, &oc2_resps), (&ref_oc2, &pass2_resps)],
        None,
    )
    .expect("solve_from_responses 2-pass");
    assert_eq!(solved_2pass.stem_count, 3);

    // 3. Pass-2 refinement trigger on heavily gated target (`Branch 1A`)
    let mut gate_m = HardwareNoiseGateDecorator::new(Chain01SymmetricOd::new());
    gate_m.threshold_lin = 0.95; // Gate stays closed on Pass 1!
    let mut gate_target = SimpleFixtureTarget::new(gate_m);
    let gate_resps = render_batch(&mut gate_target, &batch1);
    let ref_gate = generate_refinement_stimulus_batch(&manifest, &batch1, &gate_resps)
        .expect("gate refinement")
        .expect("closed gate must request Pass-2 gate-open burst");
    assert_eq!(ref_gate.stems[0].stem_id, "refinement_gate_open");

    // 4. Mode B + 4CM batch options, clip-backoff Pass-2 refinement, and error paths
    let batch_b = generate_stimulus_batch(
        &manifest,
        &BatchStimulusOptions {
            topology_mode: TopologyProbeMode::FourCableMethod4Cm,
            prefer_mode_b: true,
            ..BatchStimulusOptions::default()
        },
    )
    .expect("Mode B + 4CM batch");
    assert_eq!(batch_b.stems.len(), 3);
    assert!(batch_b.stems.iter().all(|s| s.in_3.is_some()));

    let clip_resps: Vec<StemResponse> = batch_b
        .stems
        .iter()
        .map(|s| StemResponse {
            stem_id: s.stem_id.clone(),
            sample_rate_hz: s.sample_rate_hz,
            out_1: s
                .in_1
                .iter()
                .map(|&x| (25.0 * x).clamp(-1.0, 1.0))
                .collect(),
            out_2: None,
            out_3: None,
            out_4: None,
            internal_nodes: BTreeMap::new(),
        })
        .collect();
    let ref_clip = generate_refinement_stimulus_batch(&manifest, &batch_b, &clip_resps)
        .expect("clip refinement")
        .expect("ADC hard-clip must request Pass-2 clip-backoff stem");
    assert_eq!(ref_clip.stems[0].stem_id, "refinement_clip_backoff");
    assert!(solve_from_responses(manifest, &[], None).is_err());
}

#[test]
fn test_nonlinear_envelope_fallback_and_d_optimal_active_primitive_scheduling() {
    // 1. Even-order full-wave squaring destroys linear correlation (`|coh| < 0.35`), triggering
    //    `AlignmentRegime::NonlinearEnvelopeFallback` + `estimate_leading_edge_polarity`.
    let seed = deterministic_seed_256(0xE47E_2026);
    let (dry, _) = synthesize_a5_musical_transient_coreset(&seed, 6_000, 1, 48_000);
    let lag = 22usize;
    let mut wet_rect = vec![0.0_f32; dry.len()];
    for i in lag..dry.len() {
        let s = dry[i - lag];
        wet_rect[i] = -(s * s);
    }
    let al_env = estimate_multiregime_alignment(&dry, &wet_rect, 48_000, 128)
        .expect("NonlinearEnvelopeFallback alignment");
    assert_eq!(al_env.regime, AlignmentRegime::NonlinearEnvelopeFallback);
    assert!((al_env.total_lag_samples - (lag as f64)).abs() <= 4.0);

    // 2. Run `AdaptiveStimulusController` with `max_frames: 15, fixed_duration: true` on nonlinear OD
    //    so after `Bloom` Episode 1 finishes, `Theorem 4.7` schedules $D$-optimal primitives (`P1..P6`).
    let mut od = Chain01SymmetricOd::new();
    let mut ctrl = AdaptiveStimulusController::new(StimulusConfig {
        max_frames: 15,
        max_tail_frames: 0,
        fixed_duration: true,
        ..StimulusConfig::default()
    });
    let mut last_resp: Option<StemResponse> = None;
    let mut frame_count = 0usize;
    while let Some(fr) = ctrl.next_frame(last_resp.as_ref()) {
        frame_count += 1;
        let mut out_1 = Vec::with_capacity(fr.in_1.len());
        for &x in &fr.in_1 {
            let mut ob = [0.0_f32];
            od.process_frame(&[x], &mut ob, &[0.85, 0.5, 0.6], &[0]);
            out_1.push(ob[0]);
        }
        last_resp = Some(StemResponse {
            stem_id: format!("f_{}", fr.frame_index),
            sample_rate_hz: 48_000,
            out_1,
            out_2: None,
            out_3: None,
            out_4: None,
            internal_nodes: BTreeMap::new(),
        });
    }
    assert!(frame_count >= 16);
    assert!(ctrl.bloom_telemetry().is_some());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]
    #[test]
    fn prop_kaiser64_fractional_realignment_finite_and_accurate(
        frac_lag in -12.0_f64..12.0_f64,
        invert in proptest::bool::ANY,
    ) {
        let dry: Vec<f32> = (0..2_048)
            .map(|n| {
                let t = n as f64 / 48_000.0;
                (0.35 * (2.0 * std::f64::consts::PI * 310.0 * t).sin()
                    + 0.25 * (2.0 * std::f64::consts::PI * 1_190.0 * t).cos()) as f32
            })
            .collect();
        let delayed = realign_fractional_kaiser64(&dry, -frac_lag, 0.0, invert);
        prop_assert_eq!(delayed.len(), dry.len());
        prop_assert!(delayed.iter().all(|v| v.is_finite()));

        let est = estimate_multiregime_alignment(&dry, &delayed, 48_000, 64)
            .expect("bandlimited alignment should succeed");
        prop_assert_eq!(est.polarity_inverted, invert);
        prop_assert!(
            (est.total_lag_samples - frac_lag).abs() <= 0.02,
            "lag error {} > 0.02 for frac_lag={frac_lag}",
            (est.total_lag_samples - frac_lag).abs()
        );
    }
}
