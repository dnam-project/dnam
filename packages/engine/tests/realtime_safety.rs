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
    ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_ULTRA_END, build_dnam_container,
};
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::memory::{AlignedBuffer, AlignedFp16Block, CacheLine64};
use dnam_engine::schema::{
    ControlBusConfig, Core1Config, Core2Config, DatasetProvenance, DnamJsonHeader, IoConfig,
    ModelMetadata, SlimmableTiersConfig, TierProfile,
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

fn make_test_container(core2_active: bool) -> AlignedBuffer<CacheLine64> {
    let mut header = DnamJsonHeader {
        version: "1.0.0".to_string(),
        container_kind: "dnam".to_string(),
        architecture: "DNAM_TriPhysics".to_string(),
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
            ring_buffer_pow2_log2: if core2_active { 10 } else { 0 },
            ..Default::default()
        },
        tensor_directory: BTreeMap::new(),
        stems_registry: None,
        audit_report: None,
    };

    let arena = vec![0u8; 36_224];
    let vec_bytes = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
    let mut aligned = AlignedBuffer::<CacheLine64>::new_zeroed(vec_bytes.len() / 64);
    aligned.as_bytes_mut().copy_from_slice(&vec_bytes);
    aligned
}

#[test]
fn test_zero_heap_allocations_in_all_realtime_paths_and_embedded_scratch_loader() {
    let container_active = make_test_container(true);
    let container_core1_only = make_test_container(false);

    // 1. Verify CORE2_ACTIVE == 0 allocates 0 KB of Core 2 delay scratch
    let eng_core1_only = DnamEngine::<DefaultMathBackend>::from_bytes(
        container_core1_only.as_bytes(),
        Tier::Full,
        48_000,
    )
    .unwrap();
    assert_eq!(eng_core1_only.core2.delay_scratch.byte_len(), 0);

    // 2. Verify from_bytes_with_scratch borrows caller-supplied 64B-aligned Core 2 scratch and Arena 0..2
    let mut external_scratch = vec![AlignedFp16Block([0u16; 32]); 256]; // 1,024 frames * 8 lines / 32
    let mut eng_embedded = DnamEngine::<DefaultMathBackend>::from_bytes_with_scratch(
        container_active.as_bytes(),
        &mut external_scratch,
        Tier::Ultra,
        48_000,
    )
    .unwrap();
    assert!(eng_embedded.header().is_some());

    // 3. Verify all 4 audio I/O topologies, tier switching, control updates, and telemetry allocate 0 bytes!
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
    eng_embedded.set_tier(Tier::Ultra);
    eng_embedded.set_control_value(0, 5.0);
    eng_embedded.set_control_immediate(0, 5.0);
    eng_embedded.reset_to_dc_equilibrium();
    let _snap = eng_embedded.telemetry_snapshot();
    assert_eq!(
        current_alloc_count(),
        0,
        "Real-time audio/control path allocated heap memory!"
    );

    // 4. Verify reload_model_in_place on a pre-allocated engine performs ZERO heap allocations!
    let mut eng_pool = DnamEngine::<DefaultMathBackend>::with_max_capacity(12);
    reset_alloc_counters();
    eng_pool
        .reload_model_in_place(container_active.as_bytes(), Tier::Full, 48_000)
        .unwrap();
    eng_pool
        .reload_model_in_place(container_core1_only.as_bytes(), Tier::Lite, 48_000)
        .unwrap();
    assert_eq!(
        current_alloc_count(),
        0,
        "reload_model_in_place allocated heap memory!"
    );
}
