//! Integration tests for `dnam_studio::project` and `dnam_studio::capture`.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use dnam_studio::capture::{
    AtomicF32RingBuffer, CalibrationResult, CaptureBackend, DarkglassAnagramBackend,
    SoftwareLoopbackBackend, Step1CalibrationMetrics, TargetExcitationProfile,
    UsbAudioInterfaceBackend, align_hardware_latency, analyze_hot_corner_nonlinearity,
    analyze_probe_for_excitation, apply_subsample_shift_fft, compute_pcm_sha256,
    create_capture_backend, estimate_subsample_lag_and_snr, generate_calibration_probe,
    generate_excitation_signal, generate_excitation_signal_with_profile, list_backend_devices,
    list_capture_backends, measure_silent_window_noise_floor, read_wav, synth_alignment_pulses,
    write_wav_24bit,
};
use dnam_studio::cli::format::{
    format_calibration_step1_panel, format_calibration_step2_panel, format_stem_prompt_panel,
};
use dnam_studio::error::StudioResult;
use dnam_studio::project::{
    CaptureTier, DnamProject, build_capture_plan, build_capture_plan_with_prescriptions,
    compute_control_deltas, parse_compact_range_spec, parse_compact_switch_spec,
};

fn unique_temp_dir(tag: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = env::temp_dir().join(format!("dnam_studio_test_{tag}_{nanos}"));
    let _ = fs::create_dir_all(&dir);
    dir
}

#[test]
fn test_project_manifest_controls_presets_crud_and_capture_plan() {
    let dir = unique_temp_dir("project_plan");
    let proj = DnamProject::init(&dir, "Test Rig", "alice").expect("init workspace");
    assert_eq!(proj.manifest().project.name, "Test Rig");
    assert_eq!(proj.manifest().project.creator_public_key_hex.len(), 64);
    assert!(dir.join(".gitignore").is_file());
    assert!(!dir.join("audio").exists());
    assert!(!dir.join("captures").exists());
    assert!(!dir.join("keys").exists());

    // Verify compact control spec parsers support both Phase 3 (=/:) and legacy (:/@) formats
    let gain = parse_compact_range_spec("gain:Preamp Gain:0..10@5:dB").expect("parse gain");
    let drive = parse_compact_range_spec("drive=0..10:5").expect("parse drive");
    assert_eq!(drive.id, "drive");
    assert!((drive.default_value - 5.0).abs() < 1e-6);

    let voice = parse_compact_switch_spec("voice:Voice:Clean,Crunch,Lead@1").expect("parse switch");
    let attack =
        parse_compact_switch_spec("attack=cut,flat,boost:flat").expect("parse attack switch");
    assert_eq!(attack.id, "attack");
    assert_eq!(attack.default_index, 1);

    let mut proj = proj;
    let model = proj
        .add_model("SLO100", "Soldano Lead", "Full")
        .expect("add model");
    assert_eq!(model.model_id, "SLO100");

    let updated_model = proj
        .update_model(
            "SLO100",
            Some("Soldano SLO-100 Overdrive"),
            Some("Ultra"),
            None,
            None,
            None,
        )
        .expect("update model");
    assert_eq!(updated_model.title, "Soldano SLO-100 Overdrive");
    assert_eq!(updated_model.tier, "Ultra");

    proj.add_range_control("SLO100", gain).expect("add gain");
    proj.add_range_control("SLO100", drive).expect("add drive");
    proj.add_switch_control("SLO100", voice).expect("add voice");

    proj.update_control(
        "SLO100",
        "drive",
        Some("Overdrive"),
        None,
        None,
        Some("6.0"),
        None,
    )
    .expect("update control");

    let mut preset_ranges = BTreeMap::new();
    preset_ranges.insert("gain".to_string(), 7.5);
    let mut preset_switches = BTreeMap::new();
    preset_switches.insert("voice".to_string(), 2);
    proj.add_preset(
        "SLO100",
        "lead_solo",
        "Lead Solo",
        preset_ranges.clone(),
        preset_switches,
    )
    .expect("add preset");

    proj.update_preset(
        "SLO100",
        "lead_solo",
        Some("Lead Solo Boosted"),
        Some(preset_ranges),
        None,
    )
    .expect("update preset");

    let loaded = proj.load_model("SLO100").expect("load model");
    assert_eq!(loaded.ranges.len(), 2);
    assert_eq!(loaded.switches.len(), 1);
    assert_eq!(loaded.presets.len(), 1);
    assert_eq!(loaded.presets[0].label, "Lead Solo Boosted");

    let plan = build_capture_plan(&loaded, true);
    // 1 center + 2*2 axis + 2 switch throws + 1 corner = 8 steps
    assert_eq!(plan.steps.len(), 8);
    assert_eq!(plan.steps[0].tier, CaptureTier::StarCenter);
    assert!(
        plan.steps
            .iter()
            .any(|s| s.tier == CaptureTier::CornerAnchor)
    );

    // Verify control deltas and interactive stem prompt panel formatting
    let deltas = compute_control_deltas(
        &loaded,
        Some(&plan.steps[0].normalized_ranges),
        Some(&plan.steps[0].switches),
        &plan.steps[1].normalized_ranges,
        &plan.steps[1].switches,
    );
    assert_eq!(deltas.len(), 3);
    assert!(deltas.iter().any(|d| d.changed));
    assert!(deltas.iter().any(|d| !d.changed));
    let panel = format_stem_prompt_panel(
        &loaded,
        &plan.steps[1],
        2,
        plan.steps.len(),
        Some(&plan.steps[0].normalized_ranges),
        Some(&plan.steps[0].switches),
    );
    assert!(panel.contains("TURN / FLIP"));
    assert!(panel.contains("Unchanged"));

    let mut presc_ranges = BTreeMap::new();
    presc_ranges.insert("gain".to_string(), 0.9);
    presc_ranges.insert("drive".to_string(), 0.9);
    let plan_with_presc = build_capture_plan_with_prescriptions(
        &loaded,
        true,
        &[(
            presc_ranges,
            BTreeMap::new(),
            "High gain corner".to_string(),
        )],
    );
    assert_eq!(plan_with_presc.steps.len(), 9);
    assert!(
        plan_with_presc
            .steps
            .iter()
            .any(|s| s.tier == CaptureTier::Prescribed)
    );

    proj.remove_preset("SLO100", "lead_solo")
        .expect("remove preset");
    proj.remove_control("SLO100", "drive")
        .expect("remove control");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_wav_24bit_io_alignment_calibration_and_cpal_backends() {
    let dir = unique_temp_dir("wav_capture");
    let wav_path = dir.join("probe.wav");
    let dry = generate_excitation_signal(48_000, 2_048);
    write_wav_24bit(&wav_path, &dry, 48_000, 1).expect("write 24-bit wav");

    let decoded = read_wav(&wav_path).expect("read 24-bit wav");
    assert_eq!(decoded.sample_rate_hz, 48_000);
    assert_eq!(decoded.channels, 1);
    assert_eq!(decoded.samples.len(), dry.len());

    let max_err = dry
        .iter()
        .zip(decoded.samples.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_err < 1e-4,
        "24-bit quantization error too high: {max_err}"
    );

    let sha1 = compute_pcm_sha256(&decoded.samples);
    assert_eq!(sha1.len(), 64);

    let header = synth_alignment_pulses(48_000, 1_024);
    assert_eq!(header.len(), 1_024);
    let cal_probe = generate_calibration_probe(48_000, 4_096);
    assert_eq!(cal_probe.len(), 4_096);

    assert_eq!(list_capture_backends(), vec!["software", "usb", "anagram"]);
    assert!(!list_backend_devices("anagram").is_empty());
    assert!(!list_backend_devices("usb").is_empty());

    let usb_backend = UsbAudioInterfaceBackend::default();
    assert!(usb_backend.setup_instructions(48_000).contains("Output 1"));
    let anagram_backend = DarkglassAnagramBackend::default();
    assert_eq!(anagram_backend.send_channel, 9);
    assert!(
        anagram_backend
            .setup_instructions(48_000)
            .contains("USB Output 9")
    );

    let dyn_anagram = create_capture_backend("anagram", Some("anagram_rk3582_usb"), 48_000);
    assert_eq!(dyn_anagram.backend_name(), "anagram");

    let backend = SoftwareLoopbackBackend {
        simulated_latency_samples: 5,
        send_gain_db: 0.0,
    };
    let mut step1_called = false;
    let mut step2_metrics: Option<Step1CalibrationMetrics> = None;
    let cal: CalibrationResult = backend
        .calibrate_with_prompts(
            "loopback_default",
            48_000,
            &mut |setup| {
                step1_called = true;
                assert!(format_calibration_step1_panel(setup).contains("Step 1/2"));
                Ok(())
            },
            &mut |m| {
                step2_metrics = Some(m.clone());
                assert!(format_calibration_step2_panel(m).contains("Step 2/2"));
                Ok(())
            },
        )
        .expect("calibrate");
    assert!(step1_called);
    assert_eq!(step2_metrics.expect("step2 metrics").latency_samples, 5);
    assert_eq!(cal.latency_samples, 5);
    assert!(cal.snr_db > 10.0);
    assert!(cal.excitation_profile.dominant_resonance_hz >= 20.0);

    // Verify Step 1 automatic -6.0 dB gain/pad stepping when bypass return clips (>= 0.994)
    let hot_bypass_backend = SoftwareLoopbackBackend {
        simulated_latency_samples: 5,
        send_gain_db: 45.0,
    };
    let padded_cal = hot_bypass_backend
        .calibrate("loopback_default", 48_000)
        .expect("auto-padded calibrate");
    assert!(
        padded_cal.send_gain_db < 45.0,
        "Expected Step 1 to automatically pad send_gain_db below 45.0 dB, got {}",
        padded_cal.send_gain_db
    );

    // Verify that even when Hot-Corner clips early (low saturation_ceiling_lin = 0.25),
    // the tailored excitation still drives Zone B all the way to full input scale (> 0.90)
    let early_clip_prof = TargetExcitationProfile {
        knee_amplitude_lin: 0.08,
        saturation_ceiling_lin: 0.25,
        has_delay_or_reverb_tail: true,
        estimated_max_delay_ms: 120.0,
        estimated_tail_t60_ms: 420.0,
        detected_lfo_hz: 2.5,
        dominant_resonance_hz: 950.0,
    };
    let tailored_dry =
        generate_excitation_signal_with_profile(48_000, 4_096, Some(&early_clip_prof));
    let tailored_peak = tailored_dry
        .iter()
        .map(|&s| s.abs())
        .fold(0.0_f32, f32::max);
    assert!(
        tailored_peak > 0.90,
        "Expected full-range Zone B excitation peak > 0.90, got {tailored_peak}"
    );
    let _ = StudioResult::<()>::Ok(());

    let wet = backend
        .capture_stem(&dry, &BTreeMap::new(), &BTreeMap::new())
        .expect("capture stem");
    let align = estimate_subsample_lag_and_snr(&dry, &wet, 32).expect("align");
    assert_eq!(align.coarse_lag_samples, 5);

    let locked_align =
        align_hardware_latency(&dry, &wet, 32, Some(5), 48_000, Some(-90.0)).expect("locked align");
    assert_eq!(locked_align.coarse_lag_samples, 5);

    let shifted = apply_subsample_shift_fft(&wet, align.subsample_lag_samples);
    assert_eq!(shifted.len(), wet.len());

    // Verify wait-free SPSC AtomicF32RingBuffer
    let rb = AtomicF32RingBuffer::with_capacity(64);
    assert_eq!(rb.capacity(), 64);
    let pushed = rb.push_slice(&[0.25, -0.5, 0.75, 1.0]);
    assert_eq!(pushed, 4);
    assert_eq!(rb.available_read(), 4);
    let mut popped_buf = [0.0_f32; 4];
    let popped = rb.pop_slice(&mut popped_buf);
    assert_eq!(popped, 4);
    assert_eq!(popped_buf, [0.25, -0.5, 0.75, 1.0]);
    assert_eq!(rb.available_read(), 0);

    let hot = analyze_hot_corner_nonlinearity(&dry, &wet, 0.0);
    assert!(hot.compression_ratio > 0.0);
    let nf = measure_silent_window_noise_floor(&wet, 48_000, Some(-90.0));
    assert!(nf <= 0.0);
    let prof = analyze_probe_for_excitation(&dry, &wet, 48_000);
    assert!(prof.saturation_ceiling_lin >= 0.05);

    let _ = fs::remove_dir_all(&dir);
}
