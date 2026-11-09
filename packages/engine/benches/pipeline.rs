//! End-to-End Pipeline Criterion Benchmarks (`benches/pipeline.rs`).
//!
//! Benchmarks `DnamEngine` block processing across slimmable tiers (`Lite`, `Full`, `Pro`, `Ultra`),
//! math backends (`DefaultMathBackend`, `ScalarBackend`, `ArmSmeBackend`), I/O topologies,
//! DAW buffer sizes (`32..512`), subnormal/denormal stress, and multi-instance track scaling.

use std::collections::BTreeMap;
use std::hint::black_box;

use dnam_engine::container::{
    ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_PRO_END, ARENA_OFFSET_ULTRA_END,
    build_dnam_container,
};
use dnam_engine::math::{DefaultMathBackend, MathBackend, ScalarBackend};
use dnam_engine::memory::{AlignedBuffer, CacheLine64};
use dnam_engine::schema::{
    ControlBusConfig, ControlKind, ControlSchemaDef, ControlUiHints, DatasetProvenance,
    DnamJsonHeader, IoConfig, ModelMetadata, RouterConfig, SlimmableTiersConfig, Stage1PreFxConfig,
    Stage2CascadeConfig, Stage3CabinetKautzConfig, Stage4SpatialTankConfig, TierProfile,
};
use dnam_engine::{DnamEngine, Tier};
use hotpath::criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};

fn make_bench_container() -> AlignedBuffer<CacheLine64> {
    let mut header = DnamJsonHeader {
        version: "2.0.0".to_string(),
        container_kind: "dnam".to_string(),
        architecture: "DNAM_TORA".to_string(),
        sample_rate_train_hz: 48_000,
        metadata: ModelMetadata {
            name: "Pipeline Benchmark Model".to_string(),
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
        controls_schema: vec![ControlSchemaDef {
            id: "gain".to_string(),
            name: "Gain".to_string(),
            kind: ControlKind::Range,
            unit: Some("%".to_string()),
            lobatto_degree: Some(2),
            anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
            declared_range: Some([0.0, 1.0]),
            safe_clamped_range: Some([0.0, 1.0]),
            default: serde_json::json!(0.5),
            conditional_min: BTreeMap::new(),
            conditional_max: BTreeMap::new(),
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            ui_hints: ControlUiHints::default(),
            calibration: None,
            user_metadata: BTreeMap::new(),
        }],
        presets: Vec::new(),
        control_bus: ControlBusConfig::default(),
        stage0_router: RouterConfig::default(),
        stage1_pre: Stage1PreFxConfig::default(),
        stage2_cascade: Stage2CascadeConfig::default(),
        stage3_cabinet: Stage3CabinetKautzConfig::default(),
        stage4_tank: Stage4SpatialTankConfig::default(),
        tensor_directory: BTreeMap::new(),
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

fn bench_tier_profile<B: MathBackend>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    container: &[u8],
    tier: Tier,
    profile_name: &str,
) {
    let mut engine = DnamEngine::<B>::from_bytes(container, tier, 48_000).unwrap();
    let in_l = [0.15_f32; 128];
    let in_r = [-0.15_f32; 128];
    let mut out_l = [0.0_f32; 128];
    let mut out_r = [0.0_f32; 128];

    let id = BenchmarkId::new(B::KIND.name(), profile_name);
    group.bench_function(id, |b| {
        b.iter(|| {
            engine.process_block(
                black_box(&in_l),
                black_box(&in_r),
                black_box(&mut out_l),
                black_box(&mut out_r),
            );
        });
    });
}

fn bench_process_block_tiers_and_backends(c: &mut Criterion) {
    let container_full = make_bench_container();

    let mut group = c.benchmark_group("process_block_128");
    group.throughput(Throughput::Elements(128));

    for (profile_name, bytes, tier) in [
        ("lite", container_full.as_bytes(), Tier::Lite),
        ("full", container_full.as_bytes(), Tier::Full),
        ("pro", container_full.as_bytes(), Tier::Pro),
        ("ultra", container_full.as_bytes(), Tier::Ultra),
    ] {
        bench_tier_profile::<DefaultMathBackend>(&mut group, bytes, tier, profile_name);
    }

    bench_tier_profile::<ScalarBackend>(&mut group, container_full.as_bytes(), Tier::Full, "full");

    #[cfg(target_arch = "aarch64")]
    {
        use dnam_engine::math::ArmSmeBackend;
        bench_tier_profile::<ArmSmeBackend>(
            &mut group,
            container_full.as_bytes(),
            Tier::Full,
            "full",
        );
    }

    group.finish();
}

fn bench_io_topologies_and_buffer_sizes(c: &mut Criterion) {
    let container = make_bench_container();

    let mut topo_group = c.benchmark_group("io_topologies_full_128");
    topo_group.throughput(Throughput::Elements(128));
    let mut engine =
        DnamEngine::<DefaultMathBackend>::from_bytes(container.as_bytes(), Tier::Full, 48_000)
            .unwrap();
    let in_l = [0.15_f32; 128];
    let in_r = [-0.15_f32; 128];
    let mut out_l = [0.0_f32; 128];
    let mut out_r = [0.0_f32; 128];

    topo_group.bench_function("mono_to_mono", |b| {
        b.iter(|| {
            engine.process_mono(black_box(&in_l), black_box(&mut out_l));
        });
    });
    topo_group.bench_function("mono_to_stereo", |b| {
        b.iter(|| {
            engine.process_mono_to_stereo(
                black_box(&in_l),
                black_box(&mut out_l),
                black_box(&mut out_r),
            );
        });
    });
    topo_group.bench_function("stereo_to_stereo", |b| {
        b.iter(|| {
            engine.process_block(
                black_box(&in_l),
                black_box(&in_r),
                black_box(&mut out_l),
                black_box(&mut out_r),
            );
        });
    });
    topo_group.bench_function("stereo_in_place", |b| {
        b.iter(|| {
            engine.process_block_in_place(black_box(&mut out_l), black_box(&mut out_r));
        });
    });
    topo_group.finish();

    let mut buf_group = c.benchmark_group("daw_buffer_sizes_full");
    let in_l_max = [0.15_f32; 512];
    let in_r_max = [-0.15_f32; 512];
    let mut out_l_max = [0.0_f32; 512];
    let mut out_r_max = [0.0_f32; 512];
    for frames in [32_usize, 64, 128, 256, 512] {
        buf_group.throughput(Throughput::Elements(frames as u64));
        buf_group.bench_with_input(BenchmarkId::from_parameter(frames), &frames, |b, &n| {
            b.iter(|| {
                engine.set_control_value(0, black_box(0.65));
                engine.process_block(
                    black_box(&in_l_max[..n]),
                    black_box(&in_r_max[..n]),
                    black_box(&mut out_l_max[..n]),
                    black_box(&mut out_r_max[..n]),
                );
            });
        });
    }
    buf_group.finish();
}

fn bench_denormal_and_multi_instance(c: &mut Criterion) {
    let container = make_bench_container();

    let mut denorm_group = c.benchmark_group("denormal_subnormal_stress");
    denorm_group.throughput(Throughput::Elements(128));
    let mut engine =
        DnamEngine::<DefaultMathBackend>::from_bytes(container.as_bytes(), Tier::Full, 48_000)
            .unwrap();
    let mut subnormal_in_l = [0.0_f32; 128];
    let mut subnormal_in_r = [0.0_f32; 128];
    for i in 0..128 {
        let val = f32::from_bits(1 + (i as u32));
        subnormal_in_l[i] = val;
        subnormal_in_r[i] = -val;
    }
    let mut out_l = [0.0_f32; 128];
    let mut out_r = [0.0_f32; 128];
    denorm_group.bench_function("full_128_subnormal_decay", |b| {
        b.iter(|| {
            engine.process_block(
                black_box(&subnormal_in_l),
                black_box(&subnormal_in_r),
                black_box(&mut out_l),
                black_box(&mut out_r),
            );
        });
    });
    denorm_group.finish();

    let mut multi_group = c.benchmark_group("multi_instance_track_scaling_full_128");
    let in_l = [0.15_f32; 128];
    let in_r = [-0.15_f32; 128];
    for num_tracks in [1_usize, 8, 16] {
        multi_group.throughput(Throughput::Elements((num_tracks * 128) as u64));
        let mut tracks: Vec<DnamEngine<'_, DefaultMathBackend>> = (0..num_tracks)
            .map(|_| {
                DnamEngine::<DefaultMathBackend>::from_bytes(
                    container.as_bytes(),
                    Tier::Full,
                    48_000,
                )
                .unwrap()
            })
            .collect();
        multi_group.bench_with_input(
            BenchmarkId::from_parameter(num_tracks),
            &num_tracks,
            |b, _| {
                b.iter(|| {
                    for trk in tracks.iter_mut() {
                        trk.process_block(
                            black_box(&in_l),
                            black_box(&in_r),
                            black_box(&mut out_l),
                            black_box(&mut out_r),
                        );
                    }
                });
            },
        );
    }
    multi_group.finish();
}

criterion_group!(
    benches,
    bench_process_block_tiers_and_backends,
    bench_io_topologies_and_buffer_sizes,
    bench_denormal_and_multi_instance
);
criterion_main!(benches);
