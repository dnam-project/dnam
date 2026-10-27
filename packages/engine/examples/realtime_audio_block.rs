//! Example 1: Real-Time Stereo & Mono Audio Block Processing (`examples/realtime_audio_block.rs`).
//!
//! Demonstrates loading a `.dnam` container into [`DynamicDnamEngine`] (which automatically
//! dispatches to `ARM SME`, `ARM64 NEON`, `x86 AVX2+FMA`, or `WASM SIMD128`), applying presets,
//! automating continuous `range` and discrete `switch` controls, switching quality tiers (`Lite`,
//! `Full`, `Ultra`) on the fly, and reading lock-free real-time telemetry.
//!
//! Run with:
//!   `cargo run -p dnam-engine --example realtime_audio_block`

use std::collections::BTreeMap;

use dnam_engine::container::{
    ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_ULTRA_END, build_dnam_container,
};
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::memory::{AlignedBuffer, CacheLine64};
use dnam_engine::schema::{
    ControlBusConfig, ControlKind, ControlSchemaDef, ControlUiHints, Core1Config, Core2Config,
    DatasetProvenance, DnamJsonHeader, IoConfig, ModelMetadata, PresetDef, SlimmableTiersConfig,
    TierProfile,
};
use dnam_engine::{DnamEngine, DynamicDnamEngine, Tier};

fn build_demo_dnam_container() -> AlignedBuffer<CacheLine64> {
    let mut header = DnamJsonHeader {
        version: "1.0.0".to_string(),
        container_kind: "dnam".to_string(),
        architecture: "DNAM_TriPhysics".to_string(),
        sample_rate_train_hz: 48_000,
        metadata: ModelMetadata {
            name: "Tri-Physics Crunch & Tape Echo".to_string(),
            author: Some("DNAM Studio".to_string()),
            description: Some("Demo stereo amp + modulated diffusive delay".to_string()),
            license: Some("CC-BY-4.0".to_string()),
            created_at: "2026-10-23T00:00:00Z".to_string(),
            input_reference_dbfs: -20.5,
            provenance: DatasetProvenance::default(),
            signatures: None,
        },
        user_metadata: BTreeMap::new(),
        io_config: IoConfig::default(),
        slimmable_tiers: SlimmableTiersConfig {
            lite: TierProfile {
                channels: 4,
                fdn_lines: 4,
                early_taps: 4,
                head_svd_rank: 4,
                loudness_lufs: -20.5,
                output_trim_db: 0.0,
                validation_esr_db: -40.0,
                arena_end_byte_offset: ARENA_OFFSET_LITE_END,
                prefix_crc32c: 0,
            },
            full: TierProfile {
                channels: 8,
                fdn_lines: 8,
                early_taps: 8,
                head_svd_rank: 4,
                loudness_lufs: -20.5,
                output_trim_db: 0.0,
                validation_esr_db: -45.0,
                arena_end_byte_offset: ARENA_OFFSET_FULL_END,
                prefix_crc32c: 0,
            },
            ultra: TierProfile {
                channels: 12,
                fdn_lines: 8,
                early_taps: 8,
                head_svd_rank: 4,
                loudness_lufs: -20.5,
                output_trim_db: 0.0,
                validation_esr_db: -48.0,
                arena_end_byte_offset: ARENA_OFFSET_ULTRA_END,
                prefix_crc32c: 0,
            },
        },
        controls_schema: vec![
            ControlSchemaDef {
                id: "drive".to_string(),
                name: "Drive".to_string(),
                kind: ControlKind::Range,
                unit: Some("%".to_string()),
                lobatto_degree: Some(2),
                anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
                declared_range: Some([0.0, 10.0]),
                safe_clamped_range: Some([0.0, 10.0]),
                default: serde_json::json!(5.0),
                conditional_min: BTreeMap::new(),
                conditional_max: BTreeMap::new(),
                declared_options: None,
                active_options: None,
                clamp_warning: None,
                ui_hints: ControlUiHints::default(),
                calibration: None,
                user_metadata: BTreeMap::new(),
            },
            ControlSchemaDef {
                id: "bright".to_string(),
                name: "Bright Switch".to_string(),
                kind: ControlKind::Switch,
                unit: None,
                lobatto_degree: None,
                anchor_nodes_norm: None,
                declared_range: None,
                safe_clamped_range: None,
                default: serde_json::json!("normal"),
                conditional_min: BTreeMap::new(),
                conditional_max: BTreeMap::new(),
                declared_options: Some(vec!["normal".to_string(), "bright".to_string()]),
                active_options: Some(vec!["normal".to_string(), "bright".to_string()]),
                clamp_warning: None,
                ui_hints: ControlUiHints::default(),
                calibration: None,
                user_metadata: BTreeMap::new(),
            },
        ],
        presets: vec![PresetDef {
            id: "lead_boost".to_string(),
            name: "Lead Boost".to_string(),
            description: Some("High drive with bright cap engaged".to_string()),
            controls: BTreeMap::from([
                ("drive".to_string(), serde_json::json!(8.5)),
                ("bright".to_string(), serde_json::json!("bright")),
            ]),
            user_metadata: BTreeMap::new(),
        }],
        control_bus: ControlBusConfig::default(),
        core1_hybrid_ssm: Core1Config::default(),
        core2_mufdn: Core2Config {
            active: true,
            ring_buffer_pow2_log2: 11,
            ..Default::default()
        },
        tensor_directory: BTreeMap::new(),
        stems_registry: None,
        audit_report: None,
    };

    let mut arena = vec![0u8; 36_224];
    for (i, byte) in arena[128..35_200].iter_mut().enumerate() {
        *byte = ((i * 11 + 5) & 0x3F) as u8;
    }
    let raw = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
    let mut aligned = AlignedBuffer::<CacheLine64>::new_zeroed(raw.len() / 64);
    aligned.as_bytes_mut().copy_from_slice(&raw);
    aligned
}

fn main() {
    let container = build_demo_dnam_container();

    // 1. Load model with automatic hardware SIMD backend selection (ARM SME / NEON / AVX2 / WASM SIMD)
    let dyn_engine = DynamicDnamEngine::from_bytes(container.as_bytes(), Tier::Full, 48_000)
        .expect("valid .dnam container");
    let mut engine =
        DnamEngine::<DefaultMathBackend>::from_bytes(container.as_bytes(), Tier::Full, 48_000)
            .expect("valid .dnam container");

    println!(
        "Loaded model '{}' on hardware backend: {}",
        engine.header().unwrap().metadata.name,
        dyn_engine.backend_kind().name(),
    );

    // 2. Apply a built-in preset ("lead_boost")
    engine
        .apply_preset("lead_boost")
        .expect("preset 'lead_boost' exists");

    // 3. Process a 128-sample stereo buffer at 48 kHz
    let mut in_l = [0.0_f32; 128];
    let mut in_r = [0.0_f32; 128];
    for (n, (l, r)) in in_l.iter_mut().zip(in_r.iter_mut()).enumerate() {
        let phase = (n as f32) * 0.08;
        *l = 0.25 * phase.sin();
        *r = 0.25 * phase.cos();
    }
    let mut out_l = [0.0_f32; 128];
    let mut out_r = [0.0_f32; 128];
    engine.process_block(&in_l, &in_r, &mut out_l, &mut out_r);

    // 4. Automate Drive knob (`ctrl_idx = 0`) and switch to Ultra tier (`C = 12`) with 0 allocations
    engine.set_control_value(0, 9.2);
    engine.set_tier(Tier::Ultra);
    engine.process_block(&in_l, &in_r, &mut out_l, &mut out_r);

    // 5. Inspect lock-free telemetry snapshot
    let snap = engine.telemetry_snapshot();
    println!(
        "Processed 2x128 stereo frames | Active Tier: {:?} | Max PolyADAA Drive: {:.4} | Max Pole Radius: {:.6}",
        Tier::from_u8(snap.active_tier),
        snap.max_adaa_drive,
        snap.max_pole_radius,
    );
}
