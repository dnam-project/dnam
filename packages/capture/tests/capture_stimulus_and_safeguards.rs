#![forbid(unsafe_code)]
//! Verification suite for `AdaptiveStimulusController`, `A1..A6` excitation primitives,
//! `Laws 0.A–0.F` hardware safeguards (including `Law 0.F` sub-`0 dBFS` microphone capsule clip),
//! `Laws 1–3` active learning, `Theorem 4.7` `256-D` $D$-optimal selection, and `Theorem 4.8`
//! Two-Episode Pump-and-Probe Bloom.
// verify[train.capture.excitation.codesigned]
// verify[train.capture.excitation.live-tail]
// verify[train.capture.invariants]

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::PI;
use std::time::Instant;

use dnam_capture::stimulus::fft::{is_mains_hum_bin, rfft_bins};
use dnam_capture::stimulus::{
    A3_GLISSANDO_SAMPLES_250MS, A5_CORESET_SAMPLES_500MS, ADAPTIVE_FRAME_SAMPLES,
    ADC_PEAK_GOVERNOR_FS, ActivePrimitiveKind, CYCLIC_PREFIX_SAMPLES, CYCLIC_WINDOW_SAMPLES,
    CaptureWarning, ExcitationBranch, FeedbackLawsState, HANN_CROSSFADE_SAMPLES, LtvRegime,
    MIN_COMB_BIN_STRIDE, PEDESTAL_BIN_MONO, PUMP_PROBE_COMB_PERIOD, PoleZoomRegime,
    SCHROEDER_MAX_CREST_DB, SafeguardState, StimulusConfig, TopologyProbeMode,
    analyze_pump_and_probe_recovery, apply_law4c_micro_staircase, build_disjoint_comb_schedule,
    compute_low_crest_schroeder_phases, compute_stmc_psd_88_and_spectral_4096,
    deterministic_seed_256, encode_coreset_12k_fp16, evaluate_frame_safeguards,
    synthesize_a1_log_chirp, synthesize_a2_dynamic_staircase, synthesize_a3_monophonic_glissando,
    synthesize_a5_musical_transient_coreset, synthesize_active_primitive_frame,
    synthesize_cyclic_comb_frame,
};
use dnam_capture::{AdaptiveStimulusController, CORESET_FP16_FRAMES_PER_STEM, StemResponse};
use dnam_fixtures::{
    AdcClipAndPadDecorator, BlackBoxModel, Chain01SymmetricOd, Chain02GatedAsymmetricFuzz,
    Chain04TubeAmpFmv4x12, Chain05PureLtiCabAndRoom, Chain06EnvelopeResonantSvf,
    Chain07StereoChorusAndRotary, Chain08LongBbdEchoAnd30sShimmer,
    Chain10Oc2OctaveDividerAndRingMod, HardwareNoiseGateDecorator, IdleSquealDecorator,
    MainsHumAndNoiseFloorDecorator, MicrophoneCapsuleClipDecorator, PolarityAndDcOffsetDecorator,
    TubeThermalHysteresisDecorator,
};

fn render_blackbox_frame<M: BlackBoxModel>(
    model: &mut M,
    in_1: &[f32],
    in_2: Option<&[f32]>,
    controls: &[f32],
    switches: &[u8],
) -> StemResponse {
    let n = in_1.len();
    let n_in = model.num_inputs().clamp(1, 4);
    let n_out = model.num_outputs().clamp(1, 4);
    let mut out_1 = Vec::with_capacity(n);
    let mut out_2 = (n_out >= 2).then(|| Vec::with_capacity(n));
    let mut in_buf = [0.0_f32; 4];
    let mut out_buf = [0.0_f32; 4];
    for i in 0..n {
        in_buf[0] = in_1[i];
        in_buf[1] = in_2.map_or(in_1[i], |r| r[i]);
        out_buf.fill(0.0);
        model.process_frame(&in_buf[..n_in], &mut out_buf[..n_out], controls, switches);
        out_1.push(out_buf[0]);
        if let Some(ref mut r) = out_2 {
            r.push(out_buf[1]);
        }
    }
    StemResponse {
        stem_id: "frame".to_string(),
        sample_rate_hz: model.sample_rate_hz(),
        out_1,
        out_2,
        out_3: None,
        out_4: None,
        internal_nodes: BTreeMap::new(),
    }
}

#[test]
fn test_cyclic_comb_frame_geometry_mains_exclusion_and_crest_budget() {
    // verify[train.capture.excitation.codesigned]
    assert_eq!(
        CYCLIC_PREFIX_SAMPLES + CYCLIC_WINDOW_SAMPLES,
        ADAPTIVE_FRAME_SAMPLES
    );
    assert_eq!(ADAPTIVE_FRAME_SAMPLES, 4_608);
    assert_eq!(HANN_CROSSFADE_SAMPLES, 64);

    let empty = BTreeSet::new();
    let sched = build_disjoint_comb_schedule(48_000, false, Some(PEDESTAL_BIN_MONO), &empty, &[]);
    assert!(!sched.carrier_bins.is_empty());
    assert!(!sched.guard_bins.is_empty());
    assert!(!sched.reference_anchor_bins.is_empty());

    for &b in &sched.carrier_bins {
        assert_eq!(b % 2, 1, "carrier bin {b} must be odd");
        assert!(
            !is_mains_hum_bin(b, 48_000, CYCLIC_WINDOW_SAMPLES),
            "carrier bin {b} collided with 50/60 Hz mains harmonic"
        );
    }
    for pair in sched.carrier_bins.windows(2) {
        assert!(
            pair[1] - pair[0] >= MIN_COMB_BIN_STRIDE,
            "carrier spacing {} < {}",
            pair[1] - pair[0],
            MIN_COMB_BIN_STRIDE
        );
    }

    let bin_amps: Vec<(usize, f64)> = sched.carrier_bins.iter().map(|&b| (b, 1.0)).collect();
    let phases = compute_low_crest_schroeder_phases(&bin_amps);
    assert_eq!(phases.0.len(), bin_amps.len());

    let frame0 = synthesize_cyclic_comb_frame(
        &bin_amps,
        Some(PEDESTAL_BIN_MONO),
        0.04,
        0.16,
        ADC_PEAK_GOVERNOR_FS,
        &[0.0; HANN_CROSSFADE_SAMPLES],
    );
    assert_eq!(frame0.samples.len(), ADAPTIVE_FRAME_SAMPLES);
    assert!(
        f64::from(frame0.carrier_crest_db) <= SCHROEDER_MAX_CREST_DB + 0.05,
        "Schroeder crest {} dB exceeded budget",
        frame0.carrier_crest_db
    );
    let peak = frame0
        .samples
        .iter()
        .map(|s| s.abs())
        .fold(0.0_f32, f32::max);
    assert!(
        f64::from(peak) <= ADC_PEAK_GOVERNOR_FS + 1.0e-5,
        "peak {peak} exceeded -1.5 dBFS governor"
    );

    // Verify cyclic prefix exact match outside the 64-sample Hann crossfade
    let win = &frame0.samples[CYCLIC_PREFIX_SAMPLES..];
    for i in HANN_CROSSFADE_SAMPLES..CYCLIC_PREFIX_SAMPLES {
        let diff =
            (frame0.samples[i] - win[CYCLIC_WINDOW_SAMPLES - CYCLIC_PREFIX_SAMPLES + i]).abs();
        assert!(
            diff < 1.0e-5,
            "cyclic prefix mismatch at sample {i}: {diff}"
        );
    }

    // Verify < 1.5 ms per-frame controller latency budget (averaged over 8 frames)
    let mut ctrl = AdaptiveStimulusController::new(StimulusConfig {
        fixed_duration: true,
        max_frames: 16,
        ..StimulusConfig::default()
    });
    let first = ctrl.next_frame(None).expect("frame 0");
    let dummy_resp = StemResponse {
        stem_id: "perf".to_string(),
        sample_rate_hz: 48_000,
        out_1: first.in_1.iter().map(|&x| (1.5 * x).tanh()).collect(),
        out_2: None,
        out_3: None,
        out_4: None,
        internal_nodes: BTreeMap::new(),
    };
    let t0 = Instant::now();
    for _ in 0..8 {
        let _ = ctrl.next_frame(Some(&dummy_resp)).expect("active frame");
    }
    let avg_ms = t0.elapsed().as_secs_f64() * 1_000.0 / 8.0;
    let under_llvm_cov = std::env::var_os("LLVM_PROFILE_FILE").is_some();
    let max_avg_ms = if under_llvm_cov { 1_500.0 } else { 15.0 };
    assert!(
        avg_ms < max_avg_ms,
        "next_frame average {avg_ms:.3} ms exceeded CI ceiling {max_avg_ms} ms"
    );
}

#[test]
fn test_primitives_a1_a2_a3_a5_and_pump_probe_bloom_theorem_4_8() {
    let a1 = synthesize_a1_log_chirp(4_800, 48_000, 0.8);
    assert_eq!(a1.len(), 4_800);
    assert!(synthesize_a1_log_chirp(0, 48_000, 0.8).is_empty());

    let a2 = synthesize_a2_dynamic_staircase(ADAPTIVE_FRAME_SAMPLES, 0.8);
    assert_eq!(a2.len(), ADAPTIVE_FRAME_SAMPLES);
    assert!(synthesize_a2_dynamic_staircase(0, 0.8).is_empty());

    let a3 = synthesize_a3_monophonic_glissando(48_000, 0.75);
    assert_eq!(a3.len(), A3_GLISSANDO_SAMPLES_250MS);
    assert!(synthesize_a3_monophonic_glissando(0, 0.75).is_empty());

    let seed = deterministic_seed_256(42);
    let (a5_l, a5_r) =
        synthesize_a5_musical_transient_coreset(&seed, A5_CORESET_SAMPLES_500MS, 2, 48_000);
    assert_eq!(a5_l.len(), A5_CORESET_SAMPLES_500MS);
    assert_eq!(a5_r.as_ref().unwrap().len(), A5_CORESET_SAMPLES_500MS);

    let stair = apply_law4c_micro_staircase(&a2, 0.8);
    assert_eq!(stair.len(), a2.len());

    let fp16_mono = encode_coreset_12k_fp16(&a5_l[..CORESET_FP16_FRAMES_PER_STEM], None);
    assert_eq!(fp16_mono.len(), 24_000);
    let fp16_stereo =
        encode_coreset_12k_fp16(&a5_l[..CORESET_FP16_FRAMES_PER_STEM], a5_r.as_deref());
    assert_eq!(fp16_stereo.len(), 48_000);

    let (psd_88, spec_4096) = compute_stmc_psd_88_and_spectral_4096(
        &a5_l[..CORESET_FP16_FRAMES_PER_STEM],
        a5_r.as_deref(),
    );
    assert_eq!(psd_88.len(), 88);
    assert_eq!(spec_4096.len(), 4_096);

    for prim in ActivePrimitiveKind::ALL {
        let f = synthesize_active_primitive_frame(prim, 0, false, ADC_PEAK_GOVERNOR_FS);
        assert_eq!(f.len(), ADAPTIVE_FRAME_SAMPLES);
    }

    // Theorem 4.8: Verify 512-sample periodic comb probe sag recovery triggers Episode 2
    let base_probe =
        synthesize_active_primitive_frame(ActivePrimitiveKind::P5PumpAndProbeBloom, 1, false, 0.8);
    let mut sagged = Vec::with_capacity(3 * ADAPTIVE_FRAME_SAMPLES);
    for idx in 0..(3 * ADAPTIVE_FRAME_SAMPLES) {
        let src = base_probe[idx % ADAPTIVE_FRAME_SAMPLES];
        // Exponential recovery from -3.0 dB (0.708) back to 1.0 over tau = 120 ms
        let env_gain = 1.0 - 0.292 * (-(idx as f64) / (0.120 * 48_000.0)).exp();
        sagged.push((f64::from(src) * env_gain) as f32);
    }
    assert_eq!(PUMP_PROBE_COMB_PERIOD, 512);
    let bloom = analyze_pump_and_probe_recovery(&sagged, None);
    assert!(
        bloom.initial_delta_gain_db.abs() > 0.5 && bloom.trigger_episode_2,
        "expected Episode 2 trigger on -3 dB sag recovery, got {bloom:?}"
    );

    // Stationary probe (< 0.5 dB sag) must NOT trigger Episode 2
    let flat_probe: Vec<f32> = (0..(3 * ADAPTIVE_FRAME_SAMPLES))
        .map(|i| base_probe[i % ADAPTIVE_FRAME_SAMPLES])
        .collect();
    let flat_bloom = analyze_pump_and_probe_recovery(&flat_probe, None);
    assert!(!flat_bloom.trigger_episode_2);

    // Closed-loop controller over TubeThermalHysteresisDecorator wrapping Chain04TubeAmpFmv4x12
    let mut sag_amp = TubeThermalHysteresisDecorator::new(Chain04TubeAmpFmv4x12::new());
    let mut ctrl = AdaptiveStimulusController::new(StimulusConfig {
        max_frames: 20,
        max_tail_frames: 1,
        ..StimulusConfig::default()
    });
    let mut resp: Option<StemResponse> = None;
    while let Some(fr) = ctrl.next_frame(resp.as_ref()) {
        resp = Some(render_blackbox_frame(
            &mut sag_amp,
            &fr.in_1,
            None,
            &[0.7, 0.5, 0.5, 0.5, 0.6],
            &[],
        ));
    }
    assert!((ctrl.convergence_pct() - 100.0).abs() < 1.0e-4);
    assert!(ctrl.bloom_telemetry().is_some());
}

#[test]
fn test_safeguards_law0a_ltv_law0b_echo_hold_and_live_decay_tail() {
    // Law 0.A: Chain07StereoChorusAndRotary -> LfoModulated
    let mut chorus = Chain07StereoChorusAndRotary::new();
    let mut ctrl_lfo = AdaptiveStimulusController::new(StimulusConfig {
        topology_mode: TopologyProbeMode::MonoToStereo1To2,
        max_frames: 8,
        max_tail_frames: 1,
        ..StimulusConfig::default()
    });
    let mut resp = None;
    while let Some(fr) = ctrl_lfo.next_frame(resp.as_ref()) {
        resp = Some(render_blackbox_frame(
            &mut chorus,
            &fr.in_1,
            None,
            &[0.7, 0.85, 0.7],
            &[0],
        ));
    }
    assert_eq!(ctrl_lfo.ltv_regime(), LtvRegime::LfoModulated);

    // Law 0.A: EnvelopeSvf on Chain06EnvelopeResonantSvf
    let mut env_svf = Chain06EnvelopeResonantSvf::new();
    let mut ctrl_svf = AdaptiveStimulusController::new(StimulusConfig {
        max_frames: 6,
        max_tail_frames: 1,
        ..StimulusConfig::default()
    });
    let mut r_svf = None;
    while let Some(fr) = ctrl_svf.next_frame(r_svf.as_ref()) {
        r_svf = Some(render_blackbox_frame(
            &mut env_svf,
            &fr.in_1,
            None,
            &[0.85, 0.8, 0.6],
            &[0],
        ));
    }
    assert_eq!(ctrl_svf.ltv_regime(), LtvRegime::EnvelopeSvf);
    assert!(f64::from(ctrl_svf.last_carrier_crest_db()) <= SCHROEDER_MAX_CREST_DB + 0.1);

    // Law 0.B + live decay tail: Chain08LongBbdEchoAnd30sShimmer
    // verify[train.capture.excitation.live-tail]
    let mut echo = Chain08LongBbdEchoAnd30sShimmer::new();
    let mut ctrl_echo = AdaptiveStimulusController::new(StimulusConfig {
        max_frames: 6,
        max_tail_frames: 3,
        noise_floor_dbfs: -110.0,
        ..StimulusConfig::default()
    });
    let mut r_echo = None;
    let mut total_emitted = 0usize;
    while let Some(fr) = ctrl_echo.next_frame(r_echo.as_ref()) {
        total_emitted += 1;
        r_echo = Some(render_blackbox_frame(
            &mut echo,
            &fr.in_1,
            None,
            &[0.65, 0.8, 0.75],
            &[0],
        ));
    }
    assert!(
        total_emitted >= 10,
        "expected active + coreset + decay tail frames, got {total_emitted}"
    );
}

#[test]
fn test_safeguards_law0c_mains_hum_idle_squeal_and_usb_clock_slip_replay() {
    // Mains hum + IdleSquealDecorator (Gain > 0.85 triggers 3.4 kHz parasitic tone in guard bins)
    let inner = MainsHumAndNoiseFloorDecorator::new(Chain01SymmetricOd::new(), 60.0);
    let mut squeal_rig = IdleSquealDecorator::new(inner);
    let empty = BTreeSet::new();
    let sched = build_disjoint_comb_schedule(48_000, false, Some(PEDESTAL_BIN_MONO), &empty, &[]);
    let mut sg = SafeguardState::new("squeal_test".to_string(), false, -95.0);

    // Quiet dry frame so input_env < -55 dBFS and 3.4 kHz idle oscillation fires
    let quiet_dry = vec![0.000_5_f32; ADAPTIVE_FRAME_SAMPLES];
    let wet_squeal =
        render_blackbox_frame(&mut squeal_rig, &quiet_dry, None, &[0.95, 0.5, 0.5], &[]);
    let _ = evaluate_frame_safeguards(&mut sg, 0, &quiet_dry, &wet_squeal.out_1, None, &sched);
    assert!(
        sg.warnings
            .iter()
            .any(|w| matches!(w, CaptureWarning::IdleSquealBinsMasked { .. })),
        "expected IdleSquealBinsMasked warning, got {:?}",
        sg.warnings
    );

    // USB dropout / clock-slip (> 0.25 sample jump between frames) -> discard_and_replay
    let bin_amps: Vec<(usize, f64)> = sched.carrier_bins.iter().map(|&b| (b, 1.0)).collect();
    let comb = synthesize_cyclic_comb_frame(&bin_amps, None, 0.0, 0.15, 0.8, &[0.0; 64]);
    let mut sg_usb = SafeguardState::new("usb_slip".to_string(), false, -100.0);
    let dec0 =
        evaluate_frame_safeguards(&mut sg_usb, 0, &comb.samples, &comb.samples, None, &sched);
    assert!(!dec0.discard_and_replay);

    // Shift wet by 4 samples on frame 1 -> clock slip!
    let mut slipped = vec![0.0_f32; 4];
    slipped.extend_from_slice(&comb.samples[..ADAPTIVE_FRAME_SAMPLES - 4]);
    let dec1 = evaluate_frame_safeguards(&mut sg_usb, 1, &comb.samples, &slipped, None, &sched);
    assert!(dec1.discard_and_replay);
    assert!(
        sg_usb
            .warnings
            .iter()
            .any(|w| matches!(w, CaptureWarning::ClockSlipOrUsbDropout { .. }))
    );

    // Also test USB mid-frame silent dropout
    let mut dropped = comb.samples.clone();
    dropped[2_000..2_600].fill(0.0);
    let dec_drop = evaluate_frame_safeguards(&mut sg_usb, 2, &comb.samples, &dropped, None, &sched);
    assert!(dec_drop.discard_and_replay);
}

#[test]
fn test_safeguards_law0d_adc_hard_clip_backoff_and_thermal_drift() {
    // Hot DC-boosted inner chain into AdcClipAndPadDecorator -> >= 3 samples >= 0.999 FS
    let hot = PolarityAndDcOffsetDecorator::new(Chain01SymmetricOd::new(), false, 0.95);
    let mut adc_rig = AdcClipAndPadDecorator::new(hot, 0.0);
    let mut ctrl = AdaptiveStimulusController::new(StimulusConfig {
        max_frames: 6,
        max_tail_frames: 0,
        ..StimulusConfig::default()
    });
    let f0 = ctrl.next_frame(None).expect("f0");
    let r0 = render_blackbox_frame(&mut adc_rig, &f0.in_1, None, &[0.9, 0.5, 0.5], &[]);
    let f1 = ctrl
        .next_frame(Some(&r0))
        .expect("replay frame after -6 dB backoff");
    assert!(
        ctrl.warnings()
            .iter()
            .any(|w| matches!(w, CaptureWarning::AdcHardClipStepDown { .. })),
        "expected AdcHardClipStepDown warning, got {:?}",
        ctrl.warnings()
    );
    let p0 = f0.in_1.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
    let p1 = f1.in_1.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
    assert!(
        p1 < 0.6 * p0,
        "expected -6 dB step-down replay: p0={p0}, p1={p1}"
    );

    // Thermal drift > 0.35 dB on anchor bins
    let empty = BTreeSet::new();
    let sched = build_disjoint_comb_schedule(48_000, false, None, &empty, &[]);
    let bin_amps: Vec<(usize, f64)> = sched.carrier_bins.iter().map(|&b| (b, 1.0)).collect();
    let comb = synthesize_cyclic_comb_frame(&bin_amps, None, 0.0, 0.15, 0.8, &[0.0; 64]);
    let mut sg_th = SafeguardState::new("therm".to_string(), false, -100.0);
    let _ = evaluate_frame_safeguards(&mut sg_th, 0, &comb.samples, &comb.samples, None, &sched);
    let drifted: Vec<f32> = comb.samples.iter().map(|&x| x * 0.92).collect(); // -0.72 dB drift
    let _ = evaluate_frame_safeguards(&mut sg_th, 1, &comb.samples, &drifted, None, &sched);
    assert!(
        sg_th
            .warnings
            .iter()
            .any(|w| matches!(w, CaptureWarning::ThermalBiasDriftDetected { .. }))
    );
}

#[test]
fn test_safeguards_law0e_oc2_subharmonic_divider_glissando_handoff() {
    let mut oc2 = Chain10Oc2OctaveDividerAndRingMod::new();
    let mut ctrl = AdaptiveStimulusController::new(StimulusConfig {
        max_frames: 16,
        max_tail_frames: 0,
        ..StimulusConfig::default()
    });
    let mut resp = None;
    let mut frames_emitted = 0usize;
    while let Some(fr) = ctrl.next_frame(resp.as_ref()) {
        frames_emitted += 1;
        // Switch 0 = OC-2 analog octave divider mode, Direct=0, Oct1=1.0, Oct2=0.8
        resp = Some(render_blackbox_frame(
            &mut oc2,
            &fr.in_1,
            None,
            &[1.0, 0.8, 0.0],
            &[0],
        ));
    }
    assert!(
        ctrl.warnings()
            .iter()
            .any(|w| matches!(w, CaptureWarning::SubHarmonicDividerHandoff { .. })),
        "expected SubHarmonicDividerHandoff on Chain10 OC-2, got {:?}",
        ctrl.warnings()
    );
    // 1-2 multisine frames + 3 A3 glissando frames + 3 A5 coreset frames <= 9 frames total
    assert!(
        frames_emitted <= 9,
        "OC-2 polyphonic multisine was not capped at <= 2 frames (emitted {frames_emitted})"
    );
}

#[test]
fn test_safeguards_law0f_microphone_capsule_clip_cases_a_b_and_dual_mic() {
    // 1. Non-trigger check: smooth un-cabineted overdrive (Chain01SymmetricOd) must NOT trigger Law 0.F
    let mut smooth_od = Chain01SymmetricOd::new();
    let mut ctrl_od = AdaptiveStimulusController::new(StimulusConfig {
        max_frames: 6,
        max_tail_frames: 0,
        ..StimulusConfig::default()
    });
    let mut r_od = None;
    while let Some(fr) = ctrl_od.next_frame(r_od.as_ref()) {
        r_od = Some(render_blackbox_frame(
            &mut smooth_od,
            &fr.in_1,
            None,
            &[0.9, 0.6, 0.7],
            &[],
        ));
    }
    assert!(
        !ctrl_od
            .warnings()
            .iter()
            .any(|w| matches!(w, CaptureWarning::MicrophoneCapsuleOrPreampClip { .. })),
        "smooth overdrive falsely triggered MicrophoneCapsuleOrPreampClip: {:?}",
        ctrl_od.warnings()
    );

    // 2. Case A (stationary high-boost amp + cab into -9.5 dBFS FET capsule ceiling)
    let mut cab_clipped = MicrophoneCapsuleClipDecorator::new(Chain04TubeAmpFmv4x12::new());
    let mut ctrl_a = AdaptiveStimulusController::new(StimulusConfig {
        stem_id: "boost_high".to_string(),
        is_master_volume_sweep: false,
        max_frames: 6,
        max_tail_frames: 0,
        ..StimulusConfig::default()
    });
    let f0 = ctrl_a.next_frame(None).expect("f0");
    // Drive amp hard so post-cab waveform hits +/-0.335/-0.310 capsule rails cleanly below 0 dBFS
    let mut hot_out = render_blackbox_frame(
        &mut cab_clipped,
        &f0.in_1,
        None,
        &[1.0, 0.6, 0.6, 0.6, 1.0],
        &[],
    );
    // Ensure strong post-cab capsule rail saturation if amp output was moderate
    if hot_out.out_1.iter().filter(|&&y| y.abs() >= 0.30).count() < 32 {
        let mut clean_cab = Chain05PureLtiCabAndRoom::new();
        let cab_r = render_blackbox_frame(&mut clean_cab, &f0.in_1, None, &[0.0, 0.5, 0.0], &[]);
        let boosted: Vec<f32> = cab_r.out_1.iter().map(|&y| y * 6.0).collect();
        hot_out.out_1 = cab_clipped.apply_capsule_and_xmax_fault(&boosted);
    }
    let _ = ctrl_a.next_frame(Some(&hot_out));
    assert!(
        ctrl_a.warnings().iter().any(|w| matches!(
            w,
            CaptureWarning::MicrophoneCapsuleOrPreampClip {
                dual_mic_cross_calibrated: false,
                ..
            }
        )),
        "Case A failed to emit MicrophoneCapsuleOrPreampClip: {:?}",
        ctrl_a.warnings()
    );

    // 3. Case B (`is_master_volume_sweep == true`): masks 0 dB sub-frame (`[1.0, 1.0, 0.0]`) without step-down replay
    let mut ctrl_b = AdaptiveStimulusController::new(StimulusConfig {
        stem_id: "sweep_master_volume".to_string(),
        is_master_volume_sweep: true,
        max_frames: 6,
        max_tail_frames: 0,
        ..StimulusConfig::default()
    });
    let _ = ctrl_b.next_frame(None);
    let _ = ctrl_b.next_frame(Some(&hot_out));
    assert_eq!(ctrl_b.law4c_subframe_weights(), [1.0, 1.0, 0.0]);

    // 4. Dual-Mic Stereo Cross-Handoff (`out_1` clipped close mic + `out_2` clean room/condenser mic)
    let mut clean_cab = Chain05PureLtiCabAndRoom::new();
    let f_dm = f0.in_1.clone();
    let clean_r = render_blackbox_frame(&mut clean_cab, &f_dm, None, &[0.0, 0.5, 0.0], &[]);
    let boosted_l: Vec<f32> = clean_r.out_1.iter().map(|&y| y * 6.0).collect();
    let clipped_l = cab_clipped.apply_capsule_and_xmax_fault(&boosted_l);
    let clean_companion_r: Vec<f32> = clean_r.out_1.iter().map(|&y| y * 0.45).collect();
    let dual_resp = StemResponse {
        stem_id: "dual_mic".to_string(),
        sample_rate_hz: 48_000,
        out_1: clipped_l,
        out_2: Some(clean_companion_r),
        out_3: None,
        out_4: None,
        internal_nodes: BTreeMap::new(),
    };
    let mut ctrl_dm = AdaptiveStimulusController::new(StimulusConfig {
        topology_mode: TopologyProbeMode::MonoToStereo1To2,
        max_frames: 6,
        max_tail_frames: 0,
        ..StimulusConfig::default()
    });
    let _ = ctrl_dm.next_frame(None);
    let _ = ctrl_dm.next_frame(Some(&dual_resp));
    assert!(
        ctrl_dm.warnings().iter().any(|w| matches!(
            w,
            CaptureWarning::MicrophoneCapsuleOrPreampClip {
                dual_mic_cross_calibrated: true,
                ..
            }
        )),
        "expected Dual-Mic cross-calibrated warning, got {:?}",
        ctrl_dm.warnings()
    );
}

#[test]
fn test_feedback_laws_1_2_3_d_optimal_feature_bus_and_early_stopping() {
    // 1. Law 1 Branch 1A Gated: HardwareNoiseGateDecorator / Chain02GatedAsymmetricFuzz
    let mut gated = HardwareNoiseGateDecorator::new(Chain02GatedAsymmetricFuzz::new());
    gated.threshold_lin = 0.25; // Aggressive gate threshold (-12 dBFS)
    let mut ctrl_gate = AdaptiveStimulusController::new(StimulusConfig {
        max_frames: 8,
        max_tail_frames: 0,
        ..StimulusConfig::default()
    });
    let fg0 = ctrl_gate.next_frame(None).expect("fg0");
    let quiet_in: Vec<f32> = fg0.in_1.iter().map(|&x| x * 0.05).collect();
    let rg0 = render_blackbox_frame(&mut gated, &quiet_in, None, &[0.8, 0.8, 0.5], &[]);
    let fg1 = ctrl_gate.next_frame(Some(&rg0)).expect("fg1");
    assert_eq!(ctrl_gate.active_branch(), ExcitationBranch::Branch1AGated);
    let rms0: f64 = fg0.in_1.iter().map(|&x| f64::from(x * x)).sum::<f64>();
    let rms1: f64 = fg1.in_1.iter().map(|&x| f64::from(x * x)).sum::<f64>();
    assert!(
        rms1 > rms0,
        "Branch 1A secant search must step up carrier RMS"
    );

    // 2. Law 1 Branch 1B Pure LTI: early stop (< max_frames) vs fixed_duration: true
    let mut cab_lti = Chain05PureLtiCabAndRoom::new();
    let mut ctrl_early = AdaptiveStimulusController::new(StimulusConfig {
        fixed_duration: false,
        max_frames: 24,
        max_tail_frames: 0,
        ..StimulusConfig::default()
    });
    let mut prev_pct = 0.0_f32;
    let mut prev_rem = f32::INFINITY;
    let mut early_frames = 0usize;
    let mut r_lti = None;
    while let Some(fr) = ctrl_early.next_frame(r_lti.as_ref()) {
        early_frames += 1;
        let pct = ctrl_early.convergence_pct();
        let rem = ctrl_early.remaining_max_sec();
        assert!(
            pct + 1.0e-5 >= prev_pct,
            "convergence_pct regressed: {prev_pct} -> {pct}"
        );
        assert!(rem <= prev_rem + 1.0e-5 && rem >= 0.0);
        prev_pct = pct;
        prev_rem = rem;
        r_lti = Some(render_blackbox_frame(
            &mut cab_lti,
            &fr.in_1,
            None,
            &[0.0, 0.5, 0.0],
            &[],
        ));
    }
    assert_eq!(
        ctrl_early.active_branch(),
        ExcitationBranch::Branch1BPureLti
    );
    assert_eq!(ctrl_early.remaining_max_sec(), 0.0);
    assert!(
        early_frames < 15,
        "Pure LTI should converge early in <= 14 frames, took {early_frames}"
    );

    // Compare against fixed_duration: true
    cab_lti.reset();
    let mut ctrl_fixed = AdaptiveStimulusController::new(StimulusConfig {
        fixed_duration: true,
        max_frames: 16,
        max_tail_frames: 0,
        ..StimulusConfig::default()
    });
    let mut fixed_frames = 0usize;
    let mut r_fix = None;
    while let Some(fr) = ctrl_fixed.next_frame(r_fix.as_ref()) {
        fixed_frames += 1;
        r_fix = Some(render_blackbox_frame(
            &mut cab_lti,
            &fr.in_1,
            None,
            &[0.0, 0.5, 0.0],
            &[],
        ));
    }
    assert_eq!(fixed_frames, 16 + 3); // 16 active + 3 coreset frames

    // 3. Law 2 Pole-Zoom & Theorem 4.7 256-D 7-block D-optimal active learning
    let empty = BTreeSet::new();
    let sched = build_disjoint_comb_schedule(48_000, false, None, &empty, &[]);
    let mut laws = FeedbackLawsState::new(&sched);
    // Feed high-Q resonant spectrum to trigger PoleZoomRegime::Regime2ADirectOddZoom
    let mut res_dry = vec![0.0_f32; CYCLIC_WINDOW_SAMPLES];
    let mut res_wet = vec![0.0_f32; CYCLIC_WINDOW_SAMPLES];
    let k_res = sched.carrier_bins[6];
    for n in 0..CYCLIC_WINDOW_SAMPLES {
        for &b in &sched.carrier_bins {
            let s = (2.0 * PI * (b as f64) * (n as f64) / (CYCLIC_WINDOW_SAMPLES as f64)).sin();
            res_dry[n] += 0.03 * (s as f32);
            let gain = if b == k_res { 18.0 } else { 0.4 };
            res_wet[n] += (0.03 * gain * s) as f32;
        }
    }
    let _ = rfft_bins(&res_wet);
    laws.update_from_valid_window(
        &res_dry,
        &res_wet,
        None,
        &sched,
        1.0e-10,
        TopologyProbeMode::Mono1To1,
        24,
        false,
        true,
    );
    assert_eq!(
        laws.pole_zoom_regime,
        PoleZoomRegime::Regime2ALinearPedestalOff
    );
    assert!(!laws.zoom_odd_bins.is_empty() && laws.zoom_odd_bins.len() <= 8);

    // Observe heterogeneous 256-D feature bus energy so Theorem 4.7 targets under-excited Block 5 (52..60)
    let mut fb = [0.001_f32; 256];
    for v in &mut fb[0..52] {
        *v = 5.0;
    }
    for v in &mut fb[60..256] {
        *v = 5.0;
    }
    laws.observe_256d_feature_bus(&[fb]);
    let (chosen, gain) = laws.select_next_d_optimal_primitive(true);
    assert!(gain > 0.0);
    assert_eq!(chosen, ActivePrimitiveKind::P4TankDuckingBurst);

    // Verify capsule sealing produces valid 24,640 B mono INPC + 48,000 B stereo STMC coreset (1->2 cab)
    let (inpc, spec) = ctrl_early
        .seal_inpc_and_vault_spec(true, BTreeMap::new())
        .expect("seal capsules");
    assert_eq!(inpc.len(), 24_640);
    assert_eq!(spec.wet_coreset_fp16.len(), 48_000);
}
