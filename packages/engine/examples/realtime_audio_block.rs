//! Example 1: Real-Time Stereo & Mono Audio Block Processing (`examples/realtime_audio_block.rs`).
//!
//! Demonstrates loading a `.dnam` container into [`DynamicDnamEngine`] (which automatically
//! dispatches to `ARM SME`, `ARM64 NEON`, `x86 AVX2+FMA`, or `WASM SIMD128`), applying presets
//! (including disjoint conditional switch crossfading per `ref[dsp.ctrl.switch.conditional-batch]`),
//! tempo-syncing `DelayTimeMs` controls with out-of-range clamp badge reporting (`ref[dsp.tempo.physical-inversion]`),
//! querying `supports_freeze()` (`ref[dsp.stage4.captured-freeze]`), ramping `SpilloverTrails` bypass
//! (`ref[arch.topology.spillover-trails]`), switching quality tiers (`Lite`, `Full`, `Pro`, `Ultra` `C = 16`),
//! and inspecting `OUTSIDE_CERTIFIED_INPUT` (`bit 4`) on [`DnamTelemetrySnapshot`].
//!
//! Run with:
//!   `cargo run -p dnam-engine --example realtime_audio_block`

use std::collections::BTreeMap;

use dnam_engine::container::{
    ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_PRO_END, ARENA_OFFSET_ULTRA_END,
    build_dnam_container,
};
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::math::spec_math::saturation_flags::OUTSIDE_CERTIFIED_INPUT;
use dnam_engine::memory::{AlignedBuffer, CacheLine64};
use dnam_engine::schema::{
    ControlBusConfig, ControlCalibration, ControlCalibrationRole, ControlKind, ControlSchemaDef,
    ControlUiHints, DatasetProvenance, DnamJsonHeader, IoConfig, ModelMetadata, PresetDef,
    RouterConfig, SlimmableTiersConfig, Stage1PreFxConfig, Stage2CascadeConfig,
    Stage3CabinetKautzConfig, Stage4SpatialTankConfig, TempoSubdivision, TierProfile,
};
use dnam_engine::{DnamEngine, DynamicDnamEngine, Tier};

fn build_demo_dnam_container() -> AlignedBuffer<CacheLine64> {
    let mut cond_min = BTreeMap::new();
    cond_min.insert("range_mode=short".to_string(), 0.0);
    cond_min.insert("range_mode=long".to_string(), 0.6);
    let mut cond_max = BTreeMap::new();
    cond_max.insert("range_mode=short".to_string(), 0.4);
    cond_max.insert("range_mode=long".to_string(), 1.0);

    let mut header = DnamJsonHeader {
        version: "1.0.0".to_string(),
        container_kind: "dnam".to_string(),
        architecture: "DNAM_TORA".to_string(),
        sample_rate_train_hz: 48_000,
        metadata: ModelMetadata {
            name: "TORA Crunch & Tape Echo".to_string(),
            author: Some("DNAM Studio".to_string()),
            description: Some("Demo stereo amp + modulated diffusive delay".to_string()),
            license: Some("CC-BY-4.0".to_string()),
            created_at: "2026-10-23T00:00:00Z".to_string(),
            input_reference_dbfs: -20.5,
            provenance: DatasetProvenance::default(),
            distribution_policy: None,
            signatures: None,
        },
        user_metadata: BTreeMap::new(),
        io_config: IoConfig::default(),
        slimmable_tiers: SlimmableTiersConfig {
            lite: TierProfile {
                feature_bus_dim: 64,
                channels: 4,
                early_fir_taps: 16,
                kautz_modes: 4,
                tank_lines: 4,
                head_svd_rank: 4,
                loudness_lufs: -20.5,
                output_trim_db: 0.0,
                validation_esr_db: -40.0,
                arena_end_byte_offset: ARENA_OFFSET_LITE_END,
                prefix_crc32c: 0,
            },
            full: TierProfile {
                feature_bus_dim: 128,
                channels: 8,
                early_fir_taps: 32,
                kautz_modes: 8,
                tank_lines: 8,
                head_svd_rank: 4,
                loudness_lufs: -20.5,
                output_trim_db: 0.0,
                validation_esr_db: -45.0,
                arena_end_byte_offset: ARENA_OFFSET_FULL_END,
                prefix_crc32c: 0,
            },
            pro: TierProfile {
                feature_bus_dim: 192,
                channels: 12,
                early_fir_taps: 48,
                kautz_modes: 16,
                tank_lines: 8,
                head_svd_rank: 4,
                loudness_lufs: -20.5,
                output_trim_db: 0.0,
                validation_esr_db: -48.0,
                arena_end_byte_offset: ARENA_OFFSET_PRO_END,
                prefix_crc32c: 0,
            },
            ultra: TierProfile {
                feature_bus_dim: 256,
                channels: 16,
                early_fir_taps: 64,
                kautz_modes: 32,
                tank_lines: 16,
                head_svd_rank: 8,
                loudness_lufs: -20.5,
                output_trim_db: 0.0,
                validation_esr_db: -51.0,
                arena_end_byte_offset: ARENA_OFFSET_ULTRA_END,
                prefix_crc32c: 0,
            },
        },
        controls_schema: vec![
            ControlSchemaDef {
                id: "delay_time".to_string(),
                name: "Delay Time".to_string(),
                kind: ControlKind::Range,
                unit: Some("ms".to_string()),
                lobatto_degree: Some(2),
                anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
                declared_range: Some([0.0, 1.0]),
                safe_clamped_range: Some([0.0, 1.0]),
                default: serde_json::json!(0.2),
                conditional_min: cond_min,
                conditional_max: cond_max,
                declared_options: None,
                active_options: None,
                clamp_warning: None,
                ui_hints: ControlUiHints::default(),
                calibration: Some(ControlCalibration {
                    role: ControlCalibrationRole::DelayTimeMs,
                    lobatto_physical_anchors: Some([50.0, 300.0, 800.0]),
                    conditional_physical_anchors: BTreeMap::new(),
                    tempo_sync_default_subdivision: Some(TempoSubdivision::EighthDotted),
                }),
                user_metadata: BTreeMap::new(),
            },
            ControlSchemaDef {
                id: "range_mode".to_string(),
                name: "Range Mode".to_string(),
                kind: ControlKind::Switch,
                unit: None,
                lobatto_degree: None,
                anchor_nodes_norm: None,
                declared_range: None,
                safe_clamped_range: None,
                default: serde_json::json!("short"),
                conditional_min: BTreeMap::new(),
                conditional_max: BTreeMap::new(),
                declared_options: Some(vec!["short".to_string(), "long".to_string()]),
                active_options: Some(vec!["short".to_string(), "long".to_string()]),
                clamp_warning: None,
                ui_hints: ControlUiHints::default(),
                calibration: None,
                user_metadata: BTreeMap::new(),
            },
        ],
        presets: vec![
            PresetDef {
                id: "slapback_short".to_string(),
                name: "Slapback Short".to_string(),
                description: Some("Short slapback range [0.0, 0.4]".to_string()),
                controls: BTreeMap::from([
                    ("range_mode".to_string(), serde_json::json!("short")),
                    ("delay_time".to_string(), serde_json::json!(0.25)),
                ]),
                user_metadata: BTreeMap::new(),
            },
            PresetDef {
                id: "ambient_long".to_string(),
                name: "Ambient Long".to_string(),
                description: Some("Disjoint long range [0.6, 1.0]".to_string()),
                controls: BTreeMap::from([
                    ("range_mode".to_string(), serde_json::json!("long")),
                    ("delay_time".to_string(), serde_json::json!(0.85)),
                ]),
                user_metadata: BTreeMap::new(),
            },
        ],
        control_bus: ControlBusConfig::default(),
        stage0_router: RouterConfig::default(),
        stage1_pre: Stage1PreFxConfig::default(),
        stage2_cascade: Stage2CascadeConfig::default(),
        stage3_cabinet: Stage3CabinetKautzConfig::default(),
        stage4_tank: Stage4SpatialTankConfig::default(),
        tensor_directory: BTreeMap::new(),
        stems_registry: None,
        audit_report: None,
    };
    header.io_config.capabilities.supports_freeze = true;

    let mut arena = vec![0u8; 64_000];
    for (i, byte) in arena[128..62_000].iter_mut().enumerate() {
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
        "Loaded model '{}' on hardware backend: {} | supports_freeze = {}",
        engine.header().unwrap().metadata.name,
        dyn_engine.backend_kind().name(),
        engine.supports_freeze(),
    );

    // 2. Apply presets across disjoint conditional switch intervals ("slapback_short" -> "ambient_long")
    engine
        .apply_preset("slapback_short")
        .expect("preset 'slapback_short' exists");
    engine
        .apply_preset("ambient_long")
        .expect("preset 'ambient_long' triggers disjoint-interval crossfade");

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

    // 4. Sync delay time to host tempo (300 BPM 1/16T = 33.3 ms < 50 ms min -> clamped with UI badge bit)
    let updated = engine.sync_tempo(300.0, Some(TempoSubdivision::SixteenthTriplet), false);
    println!(
        "Tempo-synced {} control(s) | active_clamp_mask = 0x{:04x}",
        updated,
        engine.telemetry_snapshot().active_clamp_mask,
    );

    // 5. Engage SpilloverTrails bypass and switch to Ultra tier (`C = 16`) with 0 allocations
    engine.set_bypass(true, true);
    engine.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
    engine.set_bypass(false, true);
    engine.set_tier(Tier::Ultra);
    engine.process_block(&in_l, &in_r, &mut out_l, &mut out_r);

    // 6. Feed a > +40 dBFS (+150.0) peak to verify OUTSIDE_CERTIFIED_INPUT telemetry flag (bit 4)
    in_l[0] = 150.0;
    engine.process_block(&in_l, &in_r, &mut out_l, &mut out_r);

    // 7. Inspect lock-free telemetry snapshot
    let snap = engine.telemetry_snapshot();
    println!(
        "Processed 4x128 stereo frames | Active Tier: {:?} | Max PolyADAA Drive: {:.4} | Max Pole Radius: {:.6} | Outside Certified Envelope Flag: {}",
        Tier::from_u8(snap.active_tier),
        snap.max_op_n_drive,
        snap.max_pole_radius,
        (snap.overflow_or_saturation_flags & OUTSIDE_CERTIFIED_INPUT) != 0,
    );
}
