//! Container Framing, ZOH Rescaling & Provenance Criterion Benchmarks (`benches/container_and_provenance.rs`).
//!
//! Benchmarks zero-copy `.dnam` container parsing, CRC32C verification, `validate_tensor_directory_slice`,
//! `ZohRescaledParams::compute`, in-place hot-swap model reloading, 224-D structural hashing,
//! Stage III gauge-invariant modal fingerprinting, and `HMAC-SHA256` watermark seed derivation.

use std::collections::BTreeMap;
use std::hint::black_box;

use dnam_engine::container::{
    ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_PRO_END, ARENA_OFFSET_ULTRA_END,
    DnamContainerView, ZohRescaledParams, build_dnam_container,
};
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::memory::{AlignedBuffer, CacheLine64};
use dnam_engine::provenance::security::{
    WatermarkSeedScope, derive_watermark_seed_hmac_sha256,
    extract_stage3_tier_fingerprint_component,
};
use dnam_engine::provenance::{DefaultCryptoProvider, extract_structural_hash_224d};
use dnam_engine::schema::{
    ControlBusConfig, ControlKind, ControlSchemaDef, ControlUiHints, DatasetProvenance,
    DnamJsonHeader, IoConfig, ModelMetadata, RouterConfig, SlimmableTiersConfig, Stage1PreFxConfig,
    Stage2CascadeConfig, Stage3CabinetKautzConfig, Stage4SpatialTankConfig, TierProfile,
};
use dnam_engine::stages::topology::validate_tensor_directory_slice;
use dnam_engine::{DnamEngine, Tier};
use hotpath::criterion::{Criterion, Throughput, criterion_group, criterion_main};

fn make_bench_container() -> AlignedBuffer<CacheLine64> {
    let mut header = DnamJsonHeader {
        version: "2.0.0".to_string(),
        container_kind: "dnam".to_string(),
        architecture: "DNAM_TORA".to_string(),
        sample_rate_train_hz: 48_000,
        metadata: ModelMetadata {
            name: "Container & Provenance Benchmark Model".to_string(),
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

fn bench_container_and_zoh(c: &mut Criterion) {
    let container = make_bench_container();
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

    group.bench_function("validate_tensor_directory_slice", |b| {
        b.iter(|| {
            let valid = validate_tensor_directory_slice(
                black_box(0),
                black_box(4096),
                black_box(ARENA_OFFSET_LITE_END as usize),
            );
            black_box(valid)
        });
    });

    group.bench_function("zoh_compute_96khz", |b| {
        b.iter(|| {
            let rescaled = ZohRescaledParams::compute(
                black_box(48_000),
                black_box(96_000),
                0.9993,
                0.995,
                64.0,
                128,
            )
            .unwrap();
            black_box(rescaled.r_dc_host)
        });
    });

    let mut engine = DnamEngine::<DefaultMathBackend>::with_max_capacity();
    group.bench_function("hot_swap_reload_in_place", |b| {
        b.iter(|| {
            engine
                .reload_model_in_place(black_box(container.as_bytes()), Tier::Full, 48_000)
                .unwrap();
        });
    });

    group.finish();
}

fn bench_provenance_and_watermark(c: &mut Criterion) {
    let container = make_bench_container();
    let crypto = DefaultCryptoProvider::default();
    let view = DnamContainerView::from_bytes(container.as_bytes(), &crypto).unwrap();
    let arena = view.arena_storage.as_bytes();

    let mut group = c.benchmark_group("provenance_and_watermark");

    group.bench_function("extract_structural_hash_224d", |b| {
        b.iter(|| black_box(extract_structural_hash_224d(black_box(arena)).unwrap()));
    });

    let fir_128b = [0u8; 128];
    let kautz_256b = [0u8; 256];
    group.bench_function("extract_stage3_tier_fingerprint", |b| {
        b.iter(|| {
            black_box(
                extract_stage3_tier_fingerprint_component(
                    black_box(&fir_128b),
                    black_box(&kautz_256b),
                    black_box(8),
                )
                .unwrap(),
            )
        });
    });

    let wm_key = [0x5Au8; 32];
    let wm_scope = WatermarkSeedScope {
        marketplace_pubkey: [0x11; 32],
        watermark_key_id: [0x22; 16],
        creator_signed_manifest_sha256: [0x33; 32],
        model_id: [0x44; 16],
        watermark_profile_sha256: [0x55; 32],
        owning_tier: 1,
    };
    let license_id = [0x66u8; 16];
    let buyer_id = [0x77u8; 16];
    group.bench_function("derive_watermark_seed_hmac_sha256", |b| {
        b.iter(|| {
            black_box(
                derive_watermark_seed_hmac_sha256(
                    black_box(&wm_key),
                    black_box(&wm_scope),
                    Some((black_box(&license_id), black_box(&buyer_id))),
                )
                .unwrap(),
            )
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_container_and_zoh,
    bench_provenance_and_watermark
);
criterion_main!(benches);
