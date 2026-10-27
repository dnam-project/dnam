//! Non-Criterion Real-Time Worst-Case Execution Time (WCET), Tail-Latency Jitter (`P50`/`P99`/`P99.9`/`Max`),
//! and `x Real-Time @ 48 kHz` / `% Single-Core CPU` Benchmark (`benches/realtime_wcet.rs`).
//!
//! Run with:
//!   `cargo bench -p dnam-engine --bench realtime_wcet`

use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Instant;

use dnam_engine::container::{
    ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_ULTRA_END, DnamContainerView,
    build_dnam_container,
};
use dnam_engine::math::{DefaultMathBackend, MathBackend, ScalarBackend};
use dnam_engine::memory::{AlignedBuffer, CacheLine64};
use dnam_engine::provenance::DefaultCryptoProvider;
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
            name: "WCET Benchmark Model".to_string(),
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
        controls_schema: vec![
            ControlSchemaDef {
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
                declared_options: Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
                active_options: Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
                clamp_warning: None,
                ui_hints: ControlUiHints::default(),
                calibration: None,
                user_metadata: BTreeMap::new(),
            },
        ],
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

fn bench_tier<B: MathBackend>(container: &[u8], tier: Tier, label: &str) {
    let mut engine = DnamEngine::<B>::from_bytes(container, tier, 48_000).unwrap();
    let block_size = 128_usize;
    let num_blocks = 375_usize; // 48,000 samples = 1.0 second of 48 kHz stereo audio
    let in_l = [0.15_f32; 128];
    let in_r = [-0.15_f32; 128];
    let mut out_l = [0.0_f32; 128];
    let mut out_r = [0.0_f32; 128];

    for _ in 0..32 {
        engine.process_block(
            black_box(&in_l),
            black_box(&in_r),
            black_box(&mut out_l),
            black_box(&mut out_r),
        );
    }

    let t0 = Instant::now();
    for _ in 0..num_blocks {
        engine.process_block(
            black_box(&in_l),
            black_box(&in_r),
            black_box(&mut out_l),
            black_box(&mut out_r),
        );
    }
    let elapsed = t0.elapsed();
    let total_samples = (block_size * num_blocks) as f64;
    let ns_per_sample = (elapsed.as_nanos() as f64) / total_samples;
    let realtime_budget_ns = 1_000_000_000.0 / 48_000.0; // 20,833.33 ns/sample
    let x_realtime = realtime_budget_ns / ns_per_sample;
    let cpu_pct = (ns_per_sample / realtime_budget_ns) * 100.0;

    println!(
        "| {:<32} | {:<11} | {:>10.2} ns | {:>10.1}x | {:>7.2}% |",
        label,
        B::KIND.name(),
        ns_per_sample,
        x_realtime,
        cpu_pct
    );
}

fn bench_wcet_tail_latency<B: MathBackend>(container: &[u8], tier: Tier, label: &str) {
    let mut engine = DnamEngine::<B>::from_bytes(container, tier, 48_000).unwrap();
    let in_l = [0.18_f32; 128];
    let in_r = [-0.18_f32; 128];
    let mut out_l = [0.0_f32; 128];
    let mut out_r = [0.0_f32; 128];

    let num_blocks = 2_000_usize;
    let mut latencies_us = Vec::with_capacity(num_blocks);

    for _ in 0..64 {
        engine.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
    }

    for b_idx in 0..num_blocks {
        let knob_gain = ((b_idx as f32) * 0.013).sin().mul_add(0.5, 0.5);
        let switch_pos = ((b_idx / 125) % 3) as f32;
        engine.set_control_value(0, black_box(knob_gain));
        engine.set_control_value(1, black_box(switch_pos));

        let t0 = Instant::now();
        engine.process_block(
            black_box(&in_l),
            black_box(&in_r),
            black_box(&mut out_l),
            black_box(&mut out_r),
        );
        latencies_us.push(t0.elapsed().as_nanos() as f64 / 1_000.0);
    }

    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = latencies_us[num_blocks / 2];
    let p99 = latencies_us[(num_blocks * 99) / 100];
    let p999 = latencies_us[(num_blocks * 999) / 1_000];
    let max_us = latencies_us[num_blocks - 1];
    let deadline_us = (128.0 / 48_000.0) * 1_000_000.0; // 2,666.67 us
    let max_budget_pct = (max_us / deadline_us) * 100.0;

    println!(
        "| {:<24} | {:>8.2} us | {:>8.2} us | {:>8.2} us | {:>8.2} us | {:>8.2}% |",
        label, p50, p99, p999, max_us, max_budget_pct
    );
}

fn main() {
    let container_full = make_bench_container(true);
    let container_core1 = make_bench_container(false);

    let iters = 1_000;
    let crypto = DefaultCryptoProvider::default();
    let t_load = Instant::now();
    for _ in 0..iters {
        let v =
            DnamContainerView::from_bytes(black_box(container_full.as_bytes()), &crypto).unwrap();
        black_box(v.preamble.header_crc32c);
    }
    let load_us = (t_load.elapsed().as_nanos() as f64) / (iters as f64 * 1000.0);

    println!(
        "=========================================================================================="
    );
    println!(" DNAM Real-Time SIMD Engine Telemetry Benchmark (48 kHz Stereo, 128-Sample Blocks)");
    println!(" #![no_std] serde + alloc Container Load + CRC32C Verification: {load_us:.2} us");
    println!(
        "=========================================================================================="
    );
    println!(
        "| {:<32} | {:<11} | {:>13} | {:>11} | {:>8} |",
        "Tier & Subsystem Profile", "Backend", "Latency/Frame", "x Real-Time", "CPU Load"
    );
    println!(
        "|----------------------------------|-------------|---------------|-------------|----------|"
    );

    bench_tier::<DefaultMathBackend>(
        container_core1.as_bytes(),
        Tier::Lite,
        "Lite (C=4, Core 1 Only)",
    );
    bench_tier::<DefaultMathBackend>(
        container_full.as_bytes(),
        Tier::Lite,
        "Lite (C=4, Core 1 + Core 2)",
    );
    bench_tier::<DefaultMathBackend>(
        container_core1.as_bytes(),
        Tier::Full,
        "Full (C=8, Core 1 Only)",
    );
    bench_tier::<DefaultMathBackend>(
        container_full.as_bytes(),
        Tier::Full,
        "Full (C=8, Core 1 + Core 2)",
    );
    bench_tier::<DefaultMathBackend>(
        container_core1.as_bytes(),
        Tier::Ultra,
        "Ultra (C=12, Core 1 Only)",
    );
    bench_tier::<DefaultMathBackend>(
        container_full.as_bytes(),
        Tier::Ultra,
        "Ultra (C=12, Core 1 + Core 2)",
    );

    #[cfg(target_arch = "aarch64")]
    {
        use dnam_engine::math::ArmSmeBackend;
        bench_tier::<ArmSmeBackend>(
            container_full.as_bytes(),
            Tier::Full,
            "Full (C=8, ARM SME + Core 2)",
        );
    }

    bench_tier::<ScalarBackend>(
        container_full.as_bytes(),
        Tier::Full,
        "Full (C=8, Scalar Baseline)",
    );
    println!(
        "==========================================================================================\n"
    );

    println!(
        "=========================================================================================="
    );
    println!(
        " Real-Time WCET & Tail-Latency Jitter Audit (2,000 x 128-Sample Blocks = 2,666.67 us Budget)"
    );
    println!(" Active Continuous Knob Modulation + Core 2 Switch Crossfading");
    println!(
        "=========================================================================================="
    );
    println!(
        "| {:<24} | {:>11} | {:>11} | {:>11} | {:>11} | {:>9} |",
        "Tier Profile (SIMD)", "P50 Block", "P99 Block", "P99.9 Block", "Max Block", "Max/Budget"
    );
    println!(
        "|--------------------------|-------------|-------------|-------------|-------------|-----------|"
    );
    bench_wcet_tail_latency::<DefaultMathBackend>(
        container_full.as_bytes(),
        Tier::Lite,
        "Lite (C=4, Modulated)",
    );
    bench_wcet_tail_latency::<DefaultMathBackend>(
        container_full.as_bytes(),
        Tier::Full,
        "Full (C=8, Modulated)",
    );
    bench_wcet_tail_latency::<DefaultMathBackend>(
        container_full.as_bytes(),
        Tier::Ultra,
        "Ultra (C=12, Modulated)",
    );
    println!(
        "=========================================================================================="
    );
}
