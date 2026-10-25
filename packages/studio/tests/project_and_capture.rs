//! Integration tests for `dnam_studio::project` and `dnam_studio::capture`.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use dnam_studio::capture::{
    CaptureBackend, SoftwareLoopbackBackend, compute_pcm_sha256, estimate_subsample_lag_and_snr,
    generate_excitation_signal, list_backend_devices, list_capture_backends, read_wav,
    write_wav_24bit,
};
use dnam_studio::project::{
    CaptureTier, DnamProject, build_capture_plan, parse_compact_range_spec,
    parse_compact_switch_spec,
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
fn test_project_manifest_controls_presets_and_capture_plan() {
    let dir = unique_temp_dir("project_plan");
    let mut proj = DnamProject::init(&dir, "Test Rig", "alice").expect("init workspace");
    assert_eq!(proj.manifest().project.name, "Test Rig");
    assert_eq!(proj.manifest().project.creator_public_key_hex.len(), 64);

    let model = proj
        .add_model("SLO100", "Soldano Lead", "Full")
        .expect("add model");
    assert_eq!(model.model_id, "SLO100");

    let gain = parse_compact_range_spec("gain:Preamp Gain:0..10@5:dB").expect("parse gain");
    let bass = parse_compact_range_spec("bass:Bass:0..10@5").expect("parse bass");
    let voice = parse_compact_switch_spec("voice:Voice:Clean,Crunch,Lead@1").expect("parse switch");

    proj.add_range_control("SLO100", gain).expect("add gain");
    proj.add_range_control("SLO100", bass).expect("add bass");
    proj.add_switch_control("SLO100", voice).expect("add voice");

    let mut preset_ranges = BTreeMap::new();
    preset_ranges.insert("gain".to_string(), 7.5);
    let mut preset_switches = BTreeMap::new();
    preset_switches.insert("voice".to_string(), 2);
    proj.add_preset(
        "SLO100",
        "lead_solo",
        "Lead Solo",
        preset_ranges,
        preset_switches,
    )
    .expect("add preset");

    let loaded = proj.load_model("SLO100").expect("load model");
    assert_eq!(loaded.ranges.len(), 2);
    assert_eq!(loaded.switches.len(), 1);
    assert_eq!(loaded.presets.len(), 1);

    let plan = build_capture_plan(&loaded, true);
    // 1 center + 2*2 axis + 2 switch throws + 1 corner = 8 steps
    assert_eq!(plan.steps.len(), 8);
    assert_eq!(plan.steps[0].tier, CaptureTier::StarCenter);
    assert!(
        plan.steps
            .iter()
            .any(|s| s.tier == CaptureTier::CornerAnchor)
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_wav_24bit_io_alignment_and_software_backend() {
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

    assert_eq!(list_capture_backends(), vec!["software", "usb", "anagram"]);
    assert!(!list_backend_devices("anagram").is_empty());

    let backend = SoftwareLoopbackBackend {
        simulated_latency_samples: 5,
    };
    let cal = backend
        .calibrate("loopback_default", 48_000)
        .expect("calibrate");
    assert_eq!(cal.latency_samples, 5);
    assert!(cal.snr_db > 10.0);

    let wet = backend
        .capture_stem(&dry, &BTreeMap::new(), &BTreeMap::new())
        .expect("capture stem");
    let align = estimate_subsample_lag_and_snr(&dry, &wet, 32).expect("align");
    assert_eq!(align.coarse_lag_samples, 5);

    let _ = fs::remove_dir_all(&dir);
}
