//! Criterion Statistical Benchmark Suite (`benches/criterion_bench.rs`).

use std::collections::BTreeMap;
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use dnam_engine::container::{
    ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_ULTRA_END, DnamContainerView,
    build_dnam_container,
};
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::memory::{AlignedBuffer, CacheLine64};
use dnam_engine::schema::{
    ControlBusConfig, Core1Config, Core2Config, DatasetProvenance, DnamJsonHeader, IoConfig,
    ModelMetadata, SlimmableTiersConfig, TierProfile,
};
use dnam_engine::{DnamEngine, Tier};

fn make_criterion_container(core2_active: bool) -> AlignedBuffer<CacheLine64> {
    let mut header = DnamJsonHeader {
        version: "1.0.0".to_string(),
        container_kind: "dnam".to_string(),
        architecture: "DNAM_TriPhysics".to_string(),
        sample_rate_train_hz: 48_000,
        metadata: ModelMetadata {
            name: "Criterion Model".to_string(),
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
        controls_schema: Vec::new(),
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

    let arena = vec![0u8; 36_224];
    let raw = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
    let mut aligned = AlignedBuffer::<CacheLine64>::new_zeroed(raw.len() / 64);
    aligned.as_bytes_mut().copy_from_slice(&raw);
    aligned
}

fn criterion_benchmarks(c: &mut Criterion) {
    let container = make_criterion_container(true);
    let in_l = [0.2_f32; 128];
    let in_r = [-0.2_f32; 128];
    let mut out_l = [0.0_f32; 128];
    let mut out_r = [0.0_f32; 128];

    let crypto = dnam_engine::provenance::DefaultCryptoProvider::default();
    c.bench_function("container_from_bytes_no_std_serde", |b| {
        b.iter(|| {
            let v =
                DnamContainerView::from_bytes(black_box(container.as_bytes()), &crypto).unwrap();
            black_box(v.preamble.header_crc32c)
        });
    });

    for (name, tier) in [
        ("process_block_128_lite", Tier::Lite),
        ("process_block_128_full", Tier::Full),
        ("process_block_128_ultra", Tier::Ultra),
    ] {
        let mut engine =
            DnamEngine::<DefaultMathBackend>::from_bytes(container.as_bytes(), tier, 48_000)
                .unwrap();
        c.bench_function(name, |b| {
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
}

criterion_group!(benches, criterion_benchmarks);
criterion_main!(benches);
