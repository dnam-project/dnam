//! Integration tests for `dnam_studio::project` and `dnam_studio::capture`.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use dnam_studio::capture::{
    AtomicF32RingBuffer, compute_pcm_sha256, list_backend_devices, list_capture_backends, read_wav,
    write_wav_24bit, write_wav_32bit_int,
};
use dnam_studio::project::{DnamProject, parse_compact_range_spec, parse_compact_switch_spec};

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
fn test_project_manifest_controls_and_presets_crud() {
    let dir = unique_temp_dir("project_crud");
    let cfg_dir = dir.join("_test_config").join("dnam");
    let proj = DnamProject::init_with_config_dir(&dir, "Test Rig", "alice", &cfg_dir)
        .expect("init workspace");
    assert_eq!(proj.manifest().project.name, "Test Rig");
    assert_eq!(proj.manifest().project.creator_public_key_hex.len(), 64);
    assert!(dir.join(".gitignore").is_file());

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

    proj.remove_preset("SLO100", "lead_solo")
        .expect("remove preset");
    proj.remove_control("SLO100", "drive")
        .expect("remove control");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_wav_io_and_ring_buffer() {
    let dir = unique_temp_dir("wav_capture");
    let wav24_path = dir.join("probe24.wav");
    let wav32_path = dir.join("probe32.wav");
    let dry: Vec<f32> = (0..2_048)
        .map(|i| ((i as f32) * 0.03).sin() * 0.5)
        .collect();
    write_wav_24bit(&wav24_path, &dry, 48_000, 1).expect("write 24-bit wav");
    write_wav_32bit_int(&wav32_path, &dry, 96_000, 1).expect("write 32-bit int wav");

    let decoded24 = read_wav(&wav24_path).expect("read 24-bit wav");
    assert_eq!(decoded24.sample_rate_hz, 48_000);
    assert_eq!(decoded24.channels, 1);
    assert_eq!(decoded24.samples.len(), dry.len());

    let decoded32 = read_wav(&wav32_path).expect("read 32-bit int wav");
    assert_eq!(decoded32.sample_rate_hz, 96_000);
    assert_eq!(decoded32.channels, 1);
    assert_eq!(decoded32.samples.len(), dry.len());

    let sha1 = compute_pcm_sha256(&decoded24.samples);
    assert_eq!(sha1.len(), 64);

    assert_eq!(list_capture_backends(), vec!["software", "usb", "anagram"]);
    assert!(!list_backend_devices("anagram").is_empty());
    assert!(!list_backend_devices("usb").is_empty());

    let rb = AtomicF32RingBuffer::with_capacity(64);
    assert_eq!(rb.capacity(), 64);
    let pushed = rb.push_slice(&[0.25, -0.5, 0.75, 1.0]);
    assert_eq!(pushed, 4);
    let mut popped_buf = [0.0_f32; 4];
    assert_eq!(rb.pop_slice(&mut popped_buf), 4);
    assert_eq!(popped_buf, [0.25, -0.5, 0.75, 1.0]);

    let _ = fs::remove_dir_all(&dir);
}
