//! Stage Recurrence & Control Plane Criterion Benchmarks (`benches/stages_and_control.rs`).
//!
//! Benchmarks `TptLr4Crossover`, `Stage4SpatialTank`, Stage IV loop saturator & blended knee,
//! `eval_stage_modulation_laws`, `ControlBus::tick_3khz`, `ExactIntFactorAccumulator`,
//! `Q64`/`Q32` tempo sync & physical calibration inversion, `apply_preset` disjoint switch
//! crossfade, and `sync_tempo`.

use std::collections::BTreeMap;
use std::hint::black_box;

use dnam_engine::container::{
    ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_PRO_END, ARENA_OFFSET_ULTRA_END,
    DnamContainerView, build_dnam_container,
};
use dnam_engine::control::{
    ExactIntFactorAccumulator, eval_q32_subdivision_phase, invert_tempo_physical_calibration,
};
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::memory::{AlignedBuffer, CacheLine64};
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::schema::{
    ControlBusConfig, ControlCalibration, ControlCalibrationRole, ControlKind, ControlSchemaDef,
    ControlUiHints, DatasetProvenance, DnamJsonHeader, IoConfig, ModelMetadata, PresetDef,
    RouterConfig, SlimmableTiersConfig, Stage1PreFxConfig, Stage2CascadeConfig,
    Stage3CabinetKautzConfig, Stage4SpatialTankConfig, TempoSubdivision, TierProfile,
};
use dnam_engine::stages::topology::{
    TptLr4Crossover, eval_stage_modulation_laws, eval_stage4_blended_feedback_knee,
    eval_stage4_loop_saturator,
};
use dnam_engine::{DnamEngine, FtzDazGuard, Tier};
use hotpath::criterion::{Criterion, criterion_group, criterion_main};

fn make_bench_container() -> AlignedBuffer<CacheLine64> {
    let mut cond_min = BTreeMap::new();
    cond_min.insert("mode=a".to_string(), 0.0);
    cond_min.insert("mode=b".to_string(), 0.6);
    let mut cond_max = BTreeMap::new();
    cond_max.insert("mode=a".to_string(), 0.4);
    cond_max.insert("mode=b".to_string(), 1.0);

    let mut header = DnamJsonHeader {
        version: "1.0.0".to_string(),
        container_kind: "dnam".to_string(),
        architecture: "DNAM_TORA".to_string(),
        sample_rate_train_hz: 48_000,
        metadata: ModelMetadata {
            name: "Stages and Control Benchmark Model".to_string(),
            author: None,
            description: None,
            license: None,
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
                id: "gain".to_string(),
                name: "Gain".to_string(),
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
                id: "mode".to_string(),
                name: "Mode".to_string(),
                kind: ControlKind::Switch,
                unit: None,
                lobatto_degree: None,
                anchor_nodes_norm: None,
                declared_range: None,
                safe_clamped_range: None,
                default: serde_json::json!("a"),
                conditional_min: BTreeMap::new(),
                conditional_max: BTreeMap::new(),
                declared_options: Some(vec!["a".to_string(), "b".to_string()]),
                active_options: Some(vec!["a".to_string(), "b".to_string()]),
                clamp_warning: None,
                ui_hints: ControlUiHints::default(),
                calibration: None,
                user_metadata: BTreeMap::new(),
            },
        ],
        presets: vec![
            PresetDef {
                id: "preset_a".to_string(),
                name: "Preset A".to_string(),
                description: None,
                controls: BTreeMap::from([
                    ("mode".to_string(), serde_json::json!("a")),
                    ("gain".to_string(), serde_json::json!(0.2)),
                ]),
                user_metadata: BTreeMap::new(),
            },
            PresetDef {
                id: "preset_b_disjoint".to_string(),
                name: "Preset B Disjoint".to_string(),
                description: None,
                controls: BTreeMap::from([
                    ("mode".to_string(), serde_json::json!("b")),
                    ("gain".to_string(), serde_json::json!(0.8)),
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

    let mut arena = vec![0u8; 64_000];
    for (i, byte) in arena[128..62_000].iter_mut().enumerate() {
        *byte = ((i * 7 + 3) & 0x3F) as u8;
    }
    let raw = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
    let mut aligned = AlignedBuffer::<CacheLine64>::new_zeroed(raw.len() / 64);
    aligned.as_bytes_mut().copy_from_slice(&raw);
    aligned
}

fn bench_stages_and_tank(c: &mut Criterion) {
    let container = make_bench_container();
    let mut engine =
        DnamEngine::<DefaultMathBackend>::from_bytes(container.as_bytes(), Tier::Full, 48_000)
            .unwrap();

    let mut group = c.benchmark_group("stages_and_tank");

    group.bench_function("tora_step_frame_full", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| black_box(engine.step_frame_tora(black_box(0.15), black_box(-0.15), false)));
    });

    group.bench_function("stage0_route_frame_no_return", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| {
            black_box(
                engine
                    .stage_pipeline
                    .stage0
                    .route_frame_no_return(black_box(0.15), black_box(-0.15)),
            )
        });
    });

    group.bench_function("stage1_pre_dynamics_stereo", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| {
            black_box(
                engine
                    .stage_pipeline
                    .stage1
                    .step_stereo(black_box(0.15), black_box(-0.15)),
            )
        });
    });

    group.bench_function("stage2_commuted_cascade_stereo", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| {
            black_box(engine.stage_pipeline.stage2.step_frame(
                black_box(0.15),
                black_box(-0.15),
                black_box(1.0),
                black_box(0.2),
            ))
        });
    });

    let fir_taps = engine.stage_pipeline.fir_taps;
    group.bench_function("stage3_early_fir_stereo", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| {
            let l = engine.stage_pipeline.stage3.step_fir_only_with_delayed_tap(
                black_box(0.15),
                black_box(0.02),
                &fir_taps,
            );
            let r = engine
                .stage_pipeline
                .stage3_r
                .step_fir_only_with_delayed_tap(black_box(-0.15), black_box(-0.02), &fir_taps);
            black_box((l, r))
        });
    });

    group.bench_function("stage4_spatial_tank_stereo", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| {
            black_box(
                engine
                    .stage_pipeline
                    .stage4
                    .step_stereo(black_box(0.15), black_box(-0.15)),
            )
        });
    });

    let mut lr4 = TptLr4Crossover::new_48khz();
    group.bench_function("stage0_tpt_lr4_crossover", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| black_box(lr4.step(black_box(0.25))));
    });

    group.bench_function("stage4_loop_saturator_and_knee", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| {
            let g = eval_stage4_blended_feedback_knee(
                black_box(0.35),
                black_box(0.85),
                black_box(0.25),
                black_box(0.50),
            )
            .unwrap_or(0.0);
            black_box(eval_stage4_loop_saturator(black_box(g), black_box(0.95)))
        });
    });

    group.finish();
}

fn bench_control_plane_and_ux(c: &mut Criterion) {
    let container = make_bench_container();
    let mut engine =
        DnamEngine::<DefaultMathBackend>::from_bytes(container.as_bytes(), Tier::Full, 48_000)
            .unwrap();
    let crypto = DefaultCryptoProvider::default();
    let view = DnamContainerView::from_bytes(container.as_bytes(), &crypto).unwrap();
    let arena = view.arena_storage.as_bytes();

    let mut group = c.benchmark_group("control_plane_and_ux");

    group.bench_function("control_bus_tick_3khz", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| {
            engine.control_bus.set_control_value(0, black_box(0.75));
            let changed = engine.control_bus.tick_3khz(black_box(0.1), arena);
            black_box(changed)
        });
    });

    let theta = [0.25_f32; 16];
    group.bench_function("eval_stage_modulation_laws", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| {
            black_box(eval_stage_modulation_laws(
                black_box(&theta),
                black_box([0.8, 0.7]),
                black_box(24.0),
                black_box(240.0),
                black_box(64.0),
                black_box(1024.0),
                black_box(0.82),
                black_box(0.5),
                black_box(1.0),
            ))
        });
    });

    let mut exact_acc = ExactIntFactorAccumulator::new(4);
    let contrib_bits = [
        0.25_f32.to_bits(),
        (-0.15_f32).to_bits(),
        0.4_f32.to_bits(),
        0.1_f32.to_bits(),
    ];
    group.bench_function("exact_int_factor_accumulator", |b| {
        b.iter(|| {
            exact_acc.reset();
            for (idx, &bits) in contrib_bits.iter().enumerate() {
                exact_acc.update_contributor(idx, black_box(bits)).unwrap();
            }
            black_box(exact_acc.interaction_f32_bits())
        });
    });

    group.bench_function("tempo_sync_q32_and_physical_inversion", |b| {
        b.iter(|| {
            let q32 = eval_q32_subdivision_phase(
                black_box(3),
                black_box(0x4000_0000),
                black_box((3, 4)),
                black_box(0),
            )
            .unwrap();
            let inv = invert_tempo_physical_calibration(
                (50.0, 800.0),
                (50.0, 800.0),
                [50.0, 300.0, 800.0],
                black_box(375.0),
                true,
            )
            .unwrap();
            black_box((q32, inv.normalized_f32))
        });
    });

    group.bench_function("ux_apply_preset_disjoint_switch_xfade", |b| {
        b.iter(|| {
            engine.apply_preset(black_box("preset_a")).unwrap();
            engine.apply_preset(black_box("preset_b_disjoint")).unwrap();
        });
    });

    group.bench_function("ux_sync_tempo_clamped", |b| {
        b.iter(|| {
            black_box(engine.sync_tempo(
                black_box(128.0),
                black_box(Some(TempoSubdivision::Quarter)),
                black_box(true),
            ))
        });
    });

    group.finish();
}

criterion_group!(benches, bench_stages_and_tank, bench_control_plane_and_ux);
criterion_main!(benches);
