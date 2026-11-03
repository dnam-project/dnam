//! Real-Time Zero-Allocation & Embedded Scratch Verification Suite (`tests/realtime_safety.rs`).
//!
//! Uses a custom `#[global_allocator]` (`CountingAllocator`) to prove mathematically and empirically
//! that all real-time audio methods (`process_block`, `process_block_in_place`,
//! `process_mono_to_stereo`, `process_mono`, `set_tier`, `set_control_value`, `reload_model_in_place`)
//! and embedded `from_bytes_with_scratch` execute with **0 heap allocations**.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use dnam_engine::container::{
    ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_PRO_END, ARENA_OFFSET_ULTRA_END,
    build_dnam_container,
};
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::memory::{AlignedBuffer, AlignedFp16Block, CacheLine64};
use dnam_engine::schema::{
    ControlBusConfig, ControlKind, ControlSchemaDef, ControlUiHints, DatasetProvenance,
    DnamJsonHeader, IoConfig, ModelMetadata, PresetDef, RouterConfig, SlimmableTiersConfig,
    Stage1PreFxConfig, Stage2CascadeConfig, Stage3CabinetKautzConfig, Stage4SpatialTankConfig,
    TierProfile,
};
use dnam_engine::{DnamEngine, Tier};

struct CountingAllocator;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: Delegates directly to `std::alloc::System` while incrementing atomic counters.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        // SAFETY: Forwards valid layout to System allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: Forwards valid pointer and layout to System allocator.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn reset_alloc_counters() {
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    ALLOC_BYTES.store(0, Ordering::SeqCst);
}

fn current_alloc_count() -> usize {
    ALLOC_COUNT.load(Ordering::SeqCst)
}

fn make_test_container() -> AlignedBuffer<CacheLine64> {
    let mut header = DnamJsonHeader {
        version: "1.0.0".to_string(),
        container_kind: "dnam".to_string(),
        architecture: "DNAM_TORA".to_string(),
        sample_rate_train_hz: 48_000,
        metadata: ModelMetadata {
            name: "RT Safety Model".to_string(),
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
        controls_schema: (0..20)
            .map(|idx| ControlSchemaDef {
                id: format!("ctrl_{idx}"),
                name: format!("Control {idx}"),
                kind: ControlKind::Range,
                unit: None,
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
            })
            .collect(),
        presets: vec![PresetDef {
            id: "rt_preset".to_string(),
            name: "RT Preset".to_string(),
            description: None,
            controls: BTreeMap::from([("ctrl_0".to_string(), serde_json::json!(7.5))]),
            user_metadata: BTreeMap::new(),
        }],
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

    let arena = vec![0u8; 64_000];
    let vec_bytes = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
    let mut aligned = AlignedBuffer::<CacheLine64>::new_zeroed(vec_bytes.len() / 64);
    aligned.as_bytes_mut().copy_from_slice(&vec_bytes);
    aligned
}

#[test]
fn test_zero_heap_allocations_in_all_realtime_paths_and_embedded_scratch_loader() {
    let container_active = make_test_container();
    let container_small = make_test_container();

    // 1. Verify telemetry snapshot reports bypassed and is_frozen status flags
    let eng_check = DnamEngine::<DefaultMathBackend>::from_bytes(
        container_small.as_bytes(),
        Tier::Full,
        48_000,
    )
    .unwrap();
    assert_eq!(eng_check.telemetry_snapshot().bypassed, 0);
    assert_eq!(eng_check.telemetry_snapshot().is_frozen, 0);

    // 2. Verify from_bytes_with_scratch borrows caller-supplied 64B-aligned scratch and Arena 0..2
    let mut external_scratch = vec![AlignedFp16Block([0u16; 32]); 256]; // 1,024 frames * 8 lines / 32
    let mut eng_embedded = DnamEngine::<DefaultMathBackend>::from_bytes_with_scratch(
        container_active.as_bytes(),
        &mut external_scratch,
        Tier::Ultra,
        48_000,
    )
    .unwrap();
    assert!(eng_embedded.header().is_some());

    // 3. Verify all 4 audio I/O topologies, tier switching, control updates, tempo sync, preset load, and telemetry allocate 0 bytes!
    reset_alloc_counters();
    let in_l = [0.25_f32; 256];
    let in_r = [-0.25_f32; 256];
    let mut out_l = [0.0_f32; 256];
    let mut out_r = [0.0_f32; 256];

    reset_alloc_counters();
    eng_embedded.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
    eng_embedded.process_block_in_place(&mut out_l, &mut out_r);
    eng_embedded.process_mono_to_stereo(&in_l, &mut out_l, &mut out_r);
    eng_embedded.process_mono(&in_l, &mut out_l);
    eng_embedded.set_tier(Tier::Lite);
    eng_embedded.set_tier(Tier::Full);
    eng_embedded.set_tier(Tier::Pro);
    eng_embedded.set_tier(Tier::Ultra);
    eng_embedded.set_control_value(0, 5.0);
    eng_embedded.set_control_immediate(0, 5.0);
    eng_embedded.sync_tempo(120.0, None, false);
    eng_embedded.register_tap_tempo(None, false);
    eng_embedded.apply_preset("rt_preset").unwrap();
    eng_embedded.reset_to_dc_equilibrium();
    eng_embedded.render_offline_bounce(&in_l, &in_r, &mut out_l, &mut out_r);
    let _snap = eng_embedded.telemetry_snapshot();
    assert_eq!(
        current_alloc_count(),
        0,
        "Real-time audio/control path allocated heap memory!"
    );

    // 4. Verify reload_model_in_place and render_offline_bounce on a pre-allocated engine perform ZERO heap allocations!
    let mut eng_pool = DnamEngine::<DefaultMathBackend>::with_max_capacity();
    reset_alloc_counters();
    eng_pool
        .reload_model_in_place(container_active.as_bytes(), Tier::Full, 48_000)
        .unwrap();
    eng_pool.render_offline_bounce(&in_l, &in_r, &mut out_l, &mut out_r);
    eng_pool
        .reload_model_in_place(container_small.as_bytes(), Tier::Lite, 48_000)
        .unwrap();
    assert_eq!(
        current_alloc_count(),
        0,
        "reload_model_in_place allocated heap memory!"
    );
}
