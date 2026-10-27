//! Criterion Statistical Benchmark Suite for `dnam-engine`.
//!
//! Run all benchmarks:
//!   `cargo bench -p dnam-engine`
//! Run a specific benchmark group:
//!   `cargo bench -p dnam-engine -- process_block`

use std::collections::BTreeMap;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use dnam_engine::container::{
    ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_ULTRA_END, DnamContainerView,
    ZohRescaledParams, build_dnam_container,
};
use dnam_engine::math::{DefaultMathBackend, MathBackend, ScalarBackend};
use dnam_engine::memory::{AlignedBuffer, CacheLine64};
use dnam_engine::provenance::{DefaultCryptoProvider, extract_structural_hash_160d};
use dnam_engine::schema::{
    ControlBusConfig, ControlKind, ControlSchemaDef, ControlUiHints, Core1Config, Core2Config,
    DatasetProvenance, DnamJsonHeader, IoConfig, ModelMetadata, SlimmableTiersConfig, TierProfile,
};
use dnam_engine::{DnamEngine, Tier};

fn make_bench_container(core2_active: bool) -> AlignedBuffer<CacheLine64> {
    let mut header = DnamJsonHeader {
        version: "1.0.0".to_string(),
        container_kind: "dnam".to_string(),
        architecture: "DNAM_TriPhysics".to_string(),
        sample_rate_train_hz: 48_000,
        metadata: ModelMetadata {
            name: "Criterion Benchmark Model".to_string(),
            author: None,
            description: None,
            license: None,
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
                fdn_lines: if core2_active { 4 } else { 0 },
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
                fdn_lines: if core2_active { 8 } else { 0 },
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
                fdn_lines: if core2_active { 8 } else { 0 },
                early_taps: 8,
                head_svd_rank: 4,
                loudness_lufs: -20.5,
                output_trim_db: 0.0,
                validation_esr_db: -48.0,
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
        core1_hybrid_ssm: Core1Config::default(),
        core2_mufdn: Core2Config {
            active: core2_active,
            ring_buffer_pow2_log2: if core2_active { 11 } else { 0 },
            ..Default::default()
        },
        tensor_directory: BTreeMap::new(),
        stems_registry: None,
        audit_report: None,
    };

    let mut arena = vec![0u8; 36_224];
    for (i, byte) in arena[128..35_200].iter_mut().enumerate() {
        *byte = ((i * 7 + 3) & 0x3F) as u8;
    }
    let raw = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
    let mut aligned = AlignedBuffer::<CacheLine64>::new_zeroed(raw.len() / 64);
    aligned.as_bytes_mut().copy_from_slice(&raw);
    aligned
}

fn bench_container_and_zoh(c: &mut Criterion) {
    let container = make_bench_container(true);
    let crypto = DefaultCryptoProvider::default();
    let mut group = c.benchmark_group("container_and_zoh");
    group.throughput(Throughput::Bytes(container.as_bytes().len() as u64));

    group.bench_function("from_bytes_crc32c", |b| {
        b.iter(|| {
            let view =
                DnamContainerView::from_bytes(black_box(container.as_bytes()), &crypto).unwrap();
            black_box(view.preamble.header_crc32c)
        });
    });

    group.bench_function("zoh_compute_96khz", |b| {
        b.iter(|| {
            let rescaled = ZohRescaledParams::compute(
                black_box(48_000),
                black_box(96_000),
                16,
                0.9993,
                0.995,
                64.0,
                128,
            )
            .unwrap();
            black_box(rescaled.r_dc_host)
        });
    });

    let mut engine = DnamEngine::<DefaultMathBackend>::with_max_capacity(11);
    group.bench_function("hot_swap_reload_in_place", |b| {
        b.iter(|| {
            engine
                .reload_model_in_place(black_box(container.as_bytes()), Tier::Full, 48_000)
                .unwrap();
        });
    });

    group.finish();
}

fn bench_tier_profile<B: MathBackend>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
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
    let container_full = make_bench_container(true);
    let container_core1 = make_bench_container(false);

    let mut group = c.benchmark_group("process_block_128");
    group.throughput(Throughput::Elements(128));

    for (profile_name, bytes, tier) in [
        ("lite_core1_only", container_core1.as_bytes(), Tier::Lite),
        ("lite_core1_core2", container_full.as_bytes(), Tier::Lite),
        ("full_core1_only", container_core1.as_bytes(), Tier::Full),
        ("full_core1_core2", container_full.as_bytes(), Tier::Full),
        ("ultra_core1_only", container_core1.as_bytes(), Tier::Ultra),
        ("ultra_core1_core2", container_full.as_bytes(), Tier::Ultra),
    ] {
        bench_tier_profile::<DefaultMathBackend>(&mut group, bytes, tier, profile_name);
    }

    bench_tier_profile::<ScalarBackend>(
        &mut group,
        container_full.as_bytes(),
        Tier::Full,
        "full_core1_core2",
    );

    #[cfg(target_arch = "aarch64")]
    {
        use dnam_engine::math::ArmSmeBackend;
        bench_tier_profile::<ArmSmeBackend>(
            &mut group,
            container_full.as_bytes(),
            Tier::Full,
            "full_core1_core2",
        );
    }

    group.finish();
}

fn bench_io_topologies_and_buffer_sizes(c: &mut Criterion) {
    let container = make_bench_container(true);

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

fn bench_isolated_subsystems(c: &mut Criterion) {
    let container = make_bench_container(true);
    let mut engine =
        DnamEngine::<DefaultMathBackend>::from_bytes(container.as_bytes(), Tier::Full, 48_000)
            .unwrap();
    let crypto = DefaultCryptoProvider::default();
    let view = DnamContainerView::from_bytes(container.as_bytes(), &crypto).unwrap();
    let arena = view.arena_storage.as_bytes();

    let mut sub_group = c.benchmark_group("isolated_subsystems");

    sub_group.bench_function("core1_sample_full", |b| {
        b.iter(|| {
            black_box(engine.core1.process_sample(
                black_box(0.15),
                black_box(-0.15),
                arena,
                2,
                false,
            ))
        });
    });

    let core1_tap = [[0.1_f32; 4]; 2];
    let delay_mod = [0.5_f32; 8];
    sub_group.bench_function("core2_sample_full_8lines", |b| {
        b.iter(|| {
            black_box(engine.core2.process_sample(
                black_box(0.15),
                black_box(-0.15),
                black_box(&core1_tap),
                black_box(&delay_mod),
                2,
            ))
        });
    });

    sub_group.bench_function("control_bus_tick_3khz", |b| {
        b.iter(|| {
            engine.control_bus.set_control_value(0, black_box(0.75));
            let changed = engine
                .control_bus
                .tick_3khz(black_box(0.1), black_box(true), arena);
            if changed {
                engine
                    .core1
                    .collapse_stereo_head_3khz(arena, &engine.control_bus.z_head, 2);
            }
            black_box(changed)
        });
    });

    sub_group.bench_function("provenance_160d_structural_hash", |b| {
        b.iter(|| black_box(extract_structural_hash_160d(black_box(arena)).unwrap()));
    });

    sub_group.finish();
}

fn bench_denormal_and_multi_instance(c: &mut Criterion) {
    let container = make_bench_container(true);

    let mut denorm_group = c.benchmark_group("denormal_subnormal_stress");
    denorm_group.throughput(Throughput::Elements(128));
    let mut engine =
        DnamEngine::<DefaultMathBackend>::from_bytes(container.as_bytes(), Tier::Full, 48_000)
            .unwrap();
    let mut subnormal_in_l = [0.0_f32; 128];
    let mut subnormal_in_r = [0.0_f32; 128];
    for i in 0..128 {
        let val = f32::from_bits(1 + (i as u32)); // IEEE-754 subnormal range (~1.4e-45 .. 1.8e-43)
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
    bench_container_and_zoh,
    bench_process_block_tiers_and_backends,
    bench_io_topologies_and_buffer_sizes,
    bench_isolated_subsystems,
    bench_denormal_and_multi_instance
);
criterion_main!(benches);
