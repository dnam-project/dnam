//! Example 2: Embedded Pedalboard Zero-Copy Scratch & Zero-Alloc Model Hot-Swap (`examples/embedded_zero_alloc.rs`).
//!
//! Demonstrates two patterns for real-time audio plugins and embedded DSP hardware:
//! 1. [`DnamEngine::from_bytes_with_scratch`]: Borrows both the 64B-aligned `.dnam` tensor arena
//!    (`Arena 0..2`) and a caller-supplied [`AlignedFp16Block`] delay scratch buffer with zero copies.
//! 2. [`DnamEngine::with_max_capacity`] + [`DnamEngine::reload_model_in_place`]: Pre-allocates all
//!    memory once at plugin startup and hot-swaps models on the audio thread with `0` heap allocations.
//!
//! Run with:
//!   `cargo run -p dnam-engine --example embedded_zero_alloc`

use std::collections::BTreeMap;

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

fn build_demo_model(name: &str, seed: u8) -> AlignedBuffer<CacheLine64> {
    let mut header = DnamJsonHeader {
        version: "1.0.0".to_string(),
        container_kind: "dnam".to_string(),
        architecture: "DNAM_TriPhysics".to_string(),
        sample_rate_train_hz: 48_000,
        metadata: ModelMetadata {
            name: name.to_string(),
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
        controls_schema: Vec::new(),
        presets: Vec::new(),
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
        *byte = (((i as u8).wrapping_mul(7)).wrapping_add(seed)) & 0x3F;
    }
    let raw = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
    let mut aligned = AlignedBuffer::<CacheLine64>::new_zeroed(raw.len() / 64);
    aligned.as_bytes_mut().copy_from_slice(&raw);
    aligned
}

fn main() {
    let model_a = build_demo_model("Vintage Clean Twin", 3);
    let model_b = build_demo_model("High-Gain Lead Stack", 19);

    // Pattern 1: Embedded zero-copy scratch buffer (`2 * 2^11 / 16 = 256` 64-byte AlignedFp16Blocks = 16 KB)
    let mut sram_delay_scratch = vec![AlignedFp16Block([0u16; 32]); 256];
    {
        let mut embedded_engine = DnamEngine::<DefaultMathBackend>::from_bytes_with_scratch(
            model_a.as_bytes(),
            &mut sram_delay_scratch,
            Tier::Lite,
            48_000,
        )
        .expect("borrowed arena + borrowed SRAM delay scratch");

        let mono_in = [0.15_f32; 64];
        let mut mono_out = [0.0_f32; 64];
        embedded_engine.process_mono(&mono_in, &mut mono_out);
        println!(
            "[Embedded Scratch] Processed 64 mono samples | Sample[63] = {:.6}",
            mono_out[63]
        );
    }

    // Pattern 2: Pre-allocated max-capacity engine with zero-allocation in-place model hot-swap
    let mut hot_swap_engine = DnamEngine::<DefaultMathBackend>::with_max_capacity(11);
    hot_swap_engine
        .reload_model_in_place(model_a.as_bytes(), Tier::Full, 48_000)
        .expect("hot-swap Model A");

    let mut left_io = [0.2_f32; 64];
    let mut right_io = [-0.2_f32; 64];
    hot_swap_engine.process_block_in_place(&mut left_io, &mut right_io);
    println!(
        "[Hot-Swap Model A] Processed 64 in-place stereo samples | Out L[63] = {:.6}",
        left_io[63]
    );

    // Hot-swap to Model B at 96 kHz (Continuous-Time ZOH rescaling, 0 heap allocations)
    hot_swap_engine
        .reload_model_in_place(model_b.as_bytes(), Tier::Full, 96_000)
        .expect("hot-swap Model B @ 96 kHz");
    hot_swap_engine.process_block_in_place(&mut left_io, &mut right_io);
    println!(
        "[Hot-Swap Model B @ 96 kHz] Processed 64 in-place stereo samples | Out L[63] = {:.6}",
        left_io[63]
    );
}
