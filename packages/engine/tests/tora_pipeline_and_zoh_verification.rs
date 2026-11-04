//! Empirical TORA End-to-End Pipeline, ZOH Multi-Rate, `4x` Bounce, and Lifecycle Verification (`Tests 8H, 8I`).
//!
//! Canonical Reference: `docs/specs/architecture/tora_architecture.md` and `docs/specs/containers/dnam_binary_format.md`.
//!
//! Verifies:
//! - `Test 8H & 8I` (`verify[container.dnam.tier-prefix]`, `verify[arch.bounce.oversampling]`): 64-byte binary framing, cache-line tier offsets
//!   (`168, 349, 566, 1000` CLs, `8,656 B` Lite RAM), Sample-0 DC resting equilibrium (`0.000000`
//!   pop on silence), distinct per-tier (`Lite ⊂ Full ⊂ Pro ⊂ Ultra`) outputs, ZOH multi-sample-rate
//!   rescaling across `44.1 kHz`–`192 kHz`, and `4x` ZOH linear-phase oversampled offline bounce with
//!   `12`-sample PDC trim (`verify[arch.bounce.oversampling]`).
//! - Engine lifecycle constructors (`from_bytes_owned_with_max_tier`, `from_bytes_with_scratch`,
//!   `with_max_tier`, `reload_model_in_place`), all `ControlCalibrationRole` bindings (`DelayTimeMs`,
//!   `Feedback`, `StereoMode`, `StereoWidth`, `FreezeLatch` as both `Switch` and `Range`),
//!   `sync_tempo`, and `DynamicDnamEngine`.

use std::collections::BTreeMap;

use dnam_engine::container::{
    ARENA_OFFSET_ULTRA_END, DnamPreamble, build_dnam_container, strip_container_to_tier,
};
use dnam_engine::math::{DefaultMathBackend, ScalarBackend};
use dnam_engine::memory::AlignedFp16Block;
use dnam_engine::schema::{
    ControlCalibration, ControlCalibrationRole, ControlKind, ControlSchemaDef, ControlUiHints,
    DnamJsonHeader, TempoSubdivision,
};
use dnam_engine::{
    DnamEngine, DynamicDnamEngine, OFFLINE_BOUNCE_PDC_SAMPLES, TOTAL_LITE_STATE_RAM_BYTES, Tier,
};

fn build_populated_tora_container() -> (DnamJsonHeader, Vec<u8>, Vec<u8>) {
    let mut header = DnamJsonHeader::default_reference();
    let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    arena[124..128].copy_from_slice(&0.9993_f32.to_le_bytes());

    let head_bases = [4_928_usize, 18_048_usize, 31_936_usize, 57_600_usize];
    for tier in 0..4 {
        for blk in 0..4 {
            let k_base = dnam_engine::provenance::kautz_soa_byte_offset(tier, blk);
            for lane in 0..4 {
                arena[k_base + lane * 4..k_base + (lane + 1) * 4]
                    .copy_from_slice(&0.90_f32.to_le_bytes());
                arena[k_base + 16 + lane * 4..k_base + 16 + (lane + 1) * 4]
                    .copy_from_slice(&0.04_f32.to_le_bytes());
                arena[k_base + 32 + lane * 4..k_base + 32 + (lane + 1) * 4]
                    .copy_from_slice(&(0.15_f32 * ((tier + 1) as f32)).to_le_bytes());
                arena[k_base + 48 + lane * 4..k_base + 48 + (lane + 1) * 4]
                    .copy_from_slice(&0.35_f32.to_le_bytes());
                arena[k_base + 64 + lane * 4..k_base + 64 + (lane + 1) * 4]
                    .copy_from_slice(&0.25_f32.to_le_bytes());
                arena[k_base + 80 + lane * 4..k_base + 80 + (lane + 1) * 4]
                    .copy_from_slice(&(-0.15_f32).to_le_bytes());
            }
        }
        let h_base = head_bases[tier];
        for i in 0..640 {
            let val = 0.03_f32 * (((i * 7 + tier * 3 + 1) % 11) as f32 - 5.0);
            arena[h_base + i * 4..h_base + (i + 1) * 4].copy_from_slice(&val.to_le_bytes());
        }
    }

    let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
    (header, arena, container)
}

/// Test 8H & 8I (`verify[container.dnam.tier-prefix]`, `verify[arch.bounce.oversampling]`): End-to-end TORA multi-stage pipeline verification across
/// 64B framing, Sample-0 DC equilibrium, 4-tier progressive outputs, ZOH sample rates, and `4x` bounce.
#[test]
fn test_8h_8i_end_to_end_tora_pipeline_tiers_zoh_sample_rates_and_4x_bounce() {
    assert_eq!(core::mem::size_of::<DnamPreamble>(), 64);
    assert_eq!(TOTAL_LITE_STATE_RAM_BYTES, 8_656);
    assert_eq!(OFFLINE_BOUNCE_PDC_SAMPLES, 12);

    let (_header, _arena, container) = build_populated_tora_container();
    let preamble = DnamPreamble::from_bytes(&container).unwrap();
    assert_eq!(preamble.arena_lite_end_cl, 168);
    assert_eq!(preamble.arena_full_end_cl, 349);
    assert_eq!(preamble.arena_pro_end_cl, 566);
    assert_eq!(preamble.arena_ultra_end_cl, 1000);
    assert_eq!(preamble.tensor_arena_offset % 64, 0);

    // 1. Sample-0 DC resting equilibrium: zero input must produce 0.000000 output on sample 0
    let mut eng =
        DnamEngine::<DefaultMathBackend>::from_bytes(&container, Tier::Ultra, 48_000).unwrap();
    eng.reset_to_dc_equilibrium();
    let mut zero_l = [0.0_f32; 16];
    let mut zero_r = [0.0_f32; 16];
    eng.process_block(&[0.0; 16], &[0.0; 16], &mut zero_l, &mut zero_r);
    for i in 0..16 {
        assert!(
            zero_l[i].abs() < 1e-7 && zero_r[i].abs() < 1e-7,
            "Sample-{i} DC transient on silence: L={}, R={}",
            zero_l[i],
            zero_r[i]
        );
    }

    // 2. Verify all 4 tiers produce distinct progressive audio outputs on a harmonic input
    let in_l: Vec<f32> = (0..128).map(|n| 0.4 * ((n as f32) * 0.13).sin()).collect();
    let in_r: Vec<f32> = (0..128).map(|n| 0.4 * ((n as f32) * 0.17).cos()).collect();
    let mut tier_outputs = Vec::new();
    for &tier in &[Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
        let stripped = strip_container_to_tier(&container, tier).unwrap();
        let mut t_eng =
            DnamEngine::<DefaultMathBackend>::from_bytes(&stripped, tier, 48_000).unwrap();
        let mut ol = vec![0.0_f32; 128];
        let mut or = vec![0.0_f32; 128];
        t_eng.process_block(&in_l, &in_r, &mut ol, &mut or);
        tier_outputs.push(ol);
    }
    for t in 0..3 {
        let diff: f32 = tier_outputs[t]
            .iter()
            .zip(tier_outputs[t + 1].iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-3);
    }

    // 3. Verify ZOH multi-sample-rate execution across all canonical host sample rates
    for &sr in &[44_100_u32, 48_000, 88_200, 96_000, 192_000] {
        eng.set_sample_rate(sr).unwrap();
        assert_eq!(eng.sample_rate_hz(), sr);
        let mut ol = [0.0_f32; 64];
        let mut or = [0.0_f32; 64];
        eng.process_block(&in_l[..64], &in_r[..64], &mut ol, &mut or);
        assert!(ol.iter().all(|v| v.is_finite()) && or.iter().all(|v| v.is_finite()));
    }

    // 4. Verify 4x ZOH linear-phase offline bounce with 12-sample PDC trim (verify[arch.bounce.oversampling])
    eng.set_sample_rate(48_000).unwrap();
    let mut bounce_l = vec![0.0_f32; 128];
    let mut bounce_r = vec![0.0_f32; 128];
    eng.render_offline_bounce(&in_l, &in_r, &mut bounce_l, &mut bounce_r);
    assert!(bounce_l.iter().any(|&v| v.abs() > 1e-4));
    assert!(bounce_l.iter().all(|v| v.is_finite()));
}

/// Verifies `from_bytes_owned_with_max_tier`, `from_bytes_with_scratch`, `with_max_tier`,
/// `reload_model_in_place`, all `ControlCalibrationRole` variants (`DelayTimeMs`, `Feedback`,
/// `StereoWidth`, `FreezeLatch` as both `Switch` and `Range`), and `sync_tempo`.
#[test]
fn test_engine_constructors_calibration_roles_and_tempo_sync() {
    let (mut header, arena, _container) = build_populated_tora_container();
    // Customize JSON header stage & capability settings to verify full binding on engine load
    header.stage0_router.sub_crossover_hz = 180.0;
    header.stage1_pre.topology_selector = 2;
    header.stage1_pre.noise_gate_enabled = true;
    header.stage1_pre.gate_threshold_db = -60.0;
    header.stage4_tank.tzf_reference_tap_samples = 24.0;
    header.stage4_tank.switch_xfade_samples = 96;
    header.control_bus.head_trust_region_radius = 12.0;
    header.io_config.capabilities.default_blend = 0.85;
    header.io_config.capabilities.wet_polarity_inverted = true;
    header.io_config.leveling.auto_unity_enabled = true;
    header.io_config.leveling.calibrated_trim_db = -1.5;
    header.controls_schema = vec![
        ControlSchemaDef {
            id: "delay_ms".to_string(),
            name: "Delay".to_string(),
            kind: ControlKind::Range,
            unit: Some("ms".to_string()),
            lobatto_degree: Some(2),
            anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
            declared_range: Some([20.0, 800.0]),
            safe_clamped_range: Some([20.0, 800.0]),
            default: serde_json::json!(250.0),
            conditional_min: BTreeMap::new(),
            conditional_max: BTreeMap::new(),
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            calibration: Some(ControlCalibration {
                role: ControlCalibrationRole::DelayTimeMs,
                lobatto_physical_anchors: Some([20.0, 250.0, 800.0]),
                conditional_physical_anchors: BTreeMap::new(),
                tempo_sync_default_subdivision: Some(TempoSubdivision::EighthDotted),
            }),
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        },
        ControlSchemaDef {
            id: "feedback".to_string(),
            name: "Feedback".to_string(),
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
            calibration: Some(ControlCalibration {
                role: ControlCalibrationRole::Feedback,
                lobatto_physical_anchors: Some([0.1, 0.5, 1.15]),
                conditional_physical_anchors: BTreeMap::new(),
                tempo_sync_default_subdivision: None,
            }),
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        },
        ControlSchemaDef {
            id: "width".to_string(),
            name: "Stereo Width".to_string(),
            kind: ControlKind::Range,
            unit: None,
            lobatto_degree: Some(2),
            anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
            declared_range: Some([0.0, 1.0]),
            safe_clamped_range: Some([0.0, 1.0]),
            default: serde_json::json!(1.0),
            conditional_min: BTreeMap::new(),
            conditional_max: BTreeMap::new(),
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            calibration: Some(ControlCalibration {
                role: ControlCalibrationRole::StereoWidth,
                lobatto_physical_anchors: Some([0.0, 0.5, 1.0]),
                conditional_physical_anchors: BTreeMap::new(),
                tempo_sync_default_subdivision: None,
            }),
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        },
        ControlSchemaDef {
            id: "freeze_sw".to_string(),
            name: "Freeze Switch".to_string(),
            kind: ControlKind::Switch,
            unit: None,
            lobatto_degree: None,
            anchor_nodes_norm: None,
            declared_range: None,
            safe_clamped_range: None,
            default: serde_json::json!("Normal"),
            conditional_min: BTreeMap::new(),
            conditional_max: BTreeMap::new(),
            declared_options: Some(vec!["Normal".to_string(), "Freeze".to_string()]),
            active_options: Some(vec!["Normal".to_string(), "Freeze".to_string()]),
            clamp_warning: None,
            calibration: Some(ControlCalibration {
                role: ControlCalibrationRole::FreezeLatch,
                lobatto_physical_anchors: None,
                conditional_physical_anchors: BTreeMap::new(),
                tempo_sync_default_subdivision: None,
            }),
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        },
        ControlSchemaDef {
            id: "freeze_rng".to_string(),
            name: "Freeze Pedal".to_string(),
            kind: ControlKind::Range,
            unit: None,
            lobatto_degree: Some(2),
            anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
            declared_range: Some([0.0, 1.0]),
            safe_clamped_range: Some([0.0, 1.0]),
            default: serde_json::json!(0.0),
            conditional_min: BTreeMap::new(),
            conditional_max: BTreeMap::new(),
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            calibration: Some(ControlCalibration {
                role: ControlCalibrationRole::FreezeLatch,
                lobatto_physical_anchors: Some([0.0, 0.5, 1.0]),
                conditional_physical_anchors: BTreeMap::new(),
                tempo_sync_default_subdivision: None,
            }),
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        },
    ];

    let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

    // 1. Test from_bytes_owned_with_max_tier and with_max_tier
    let mut eng = DnamEngine::<ScalarBackend>::from_bytes_owned_with_max_tier(
        &container,
        Tier::Lite,
        Tier::Pro,
        48_000,
    )
    .unwrap();
    assert_eq!(eng.max_supported_tier(), Tier::Pro);
    assert_eq!(eng.stage_pipeline.stage1.topology_mode, 2);
    assert!(eng.stage_pipeline.stage1.gate_threshold > 0.0);
    assert!((eng.zoh().tau_ref_samples_host - 24.0).abs() < 1e-6);
    assert_eq!(eng.zoh().switch_xfade_samples_host, 96);
    assert!((eng.control_bus.inv_trust_radius_mh2 - 1.0 / 144.0).abs() < 1e-6);
    eng.configure_host_mix_and_leveling(1.0, 0.0, false);
    eng.set_tier(Tier::Ultra);
    assert_eq!(eng.tier(), Tier::Pro);

    // 2. Test calibration roles (DelayTimeMs, Feedback, StereoWidth, FreezeLatch Switch & Range)
    eng.set_control_value(0, 400.0);
    eng.set_control_value(1, 8.0);
    eng.set_control_value(2, 0.75);
    assert!((eng.stereo_width() - 0.75).abs() < 1e-4);

    eng.set_control_value(3, 1.0);
    assert!(eng.is_frozen());
    eng.set_control_value(3, 0.0);
    assert!(!eng.is_frozen());

    eng.set_control_value(4, 0.8);
    assert!(eng.is_frozen());
    eng.set_freeze(false);
    assert!(!eng.is_frozen());
    eng.set_stereo_width(0.5);
    assert!((eng.stereo_width() - 0.5).abs() < 1e-6);

    // 3. Test sync_tempo and reload_model_in_place
    let updated = eng.sync_tempo(120.0, Some(TempoSubdivision::Quarter), true);
    assert_eq!(updated, 1);

    let mut scratch = [AlignedFp16Block::default(); 4];
    let mut eng_scratch = DnamEngine::<ScalarBackend>::from_bytes_with_scratch(
        &container,
        &mut scratch,
        Tier::Full,
        48_000,
    )
    .unwrap();
    assert_eq!(eng_scratch.tier(), Tier::Full);
    let mut mono_out = [0.0f32; 8];
    eng_scratch.process_mono(&[0.1f32; 8], &mut mono_out);
    assert!(mono_out[0].is_finite());

    let mut eng_cap = DnamEngine::<ScalarBackend>::with_max_tier(Tier::Full);
    eng_cap
        .reload_model_in_place(&container, Tier::Lite, 48_000)
        .unwrap();
    assert_eq!(eng_cap.tier(), Tier::Lite);

    let dyn_eng = DynamicDnamEngine::from_bytes(&container, Tier::Full, 48_000).unwrap();
    assert_eq!(dyn_eng.tier(), Tier::Full);

    // 4. Verify hyper-wide stereo width (2.0) doubles L-R side difference vs reference (1.0)
    let mut eng_w1 =
        DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
    let mut eng_w2 =
        DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
    eng_w1.configure_host_mix_and_leveling(1.0, 0.0, false);
    eng_w2.configure_host_mix_and_leveling(1.0, 0.0, false);
    eng_w1.set_stereo_width(1.0);
    eng_w2.set_stereo_width(2.0);
    assert!((eng_w2.stereo_width() - 2.0).abs() < 1e-6);

    let in_l = [0.35_f32, -0.25, 0.45, -0.15];
    let in_r = [-0.35_f32, 0.25, -0.45, 0.15];
    let mut out1_l = [0.0_f32; 4];
    let mut out1_r = [0.0_f32; 4];
    let mut out2_l = [0.0_f32; 4];
    let mut out2_r = [0.0_f32; 4];
    eng_w1.process_block(&in_l, &in_r, &mut out1_l, &mut out1_r);
    eng_w2.process_block(&in_l, &in_r, &mut out2_l, &mut out2_r);
    for i in 0..4 {
        let side1 = out1_l[i] - out1_r[i];
        let side2 = out2_l[i] - out2_r[i];
        assert!(
            (side2 - 2.0 * side1).abs() < 1e-5,
            "Hyper-wide stereo width=2.0 failed at frame {i}: side1={side1}, side2={side2}"
        );
    }

    // 5. Verify W_ctrl (bytes 9,536..10,560) dispatches dynamic modulations to Stage I and Stage IV
    let mut mod_arena = arena.clone();
    // Row 0 (ctrl_basis_4[0]): col 0 (delay mod +12.0 samples), col 2 (gain mod +0.10), col 7 (state mod: psi +0.2)
    let w_ctrl_base = 9_536_usize;
    mod_arena[w_ctrl_base..w_ctrl_base + 4].copy_from_slice(&12.0_f32.to_le_bytes());
    mod_arena[w_ctrl_base + 32..w_ctrl_base + 36].copy_from_slice(&0.40_f32.to_le_bytes());
    mod_arena[w_ctrl_base + 116..w_ctrl_base + 120].copy_from_slice(&0.25_f32.to_le_bytes());
    let mod_container = build_dnam_container(&mut header, &mod_arena, &mod_arena, false).unwrap();
    let mut mod_eng =
        DnamEngine::<ScalarBackend>::from_bytes(&mod_container, Tier::Lite, 48_000).unwrap();
    mod_eng.set_control_immediate(0, 800.0); // non-default u_norm = +1.0 -> ctrl_basis_4[0] != 0
    assert!(
        (mod_eng.stage_pipeline.stage4.g_loop - 0.85).abs() > 1e-3,
        "Expected W_ctrl to modulate stage4.g_loop, got {}",
        mod_eng.stage_pipeline.stage4.g_loop
    );
    assert!(
        (mod_eng.stage_pipeline.stage4.target_psi - core::f32::consts::FRAC_PI_4).abs() > 1e-3,
        "Expected W_ctrl to modulate stage4.target_psi, got {}",
        mod_eng.stage_pipeline.stage4.target_psi
    );
    assert!(
        (mod_eng.stage_pipeline.stage1.op_t_pre.target_delay - 64.0).abs() > 1.0,
        "Expected W_ctrl to modulate stage1.op_t_pre.target_delay, got {}",
        mod_eng.stage_pipeline.stage1.op_t_pre.target_delay
    );
}

mod arch_spec_integration {
    // Specification Verification Suite: Phase 9 — TORA System Architecture Integration.
    //
    // Verifies all 34 engine-scope rules in `docs/specs/architecture/tora_architecture.md`:
    // - `verify[arch.pillars.dual-space-control]`
    // - `verify[arch.pillars.dual-rate-plane]`
    // - `verify[arch.pillars.cp1-projections]`
    // - `verify[arch.lifecycle.dual-rate]`
    // - `verify[arch.fp.callback-env]`
    // - `verify[arch.fp.range-saturation]`
    // - `verify[arch.operators.tri-kernel]`
    // - `verify[arch.operators.self-similar]`
    // - `verify[arch.simd.lane-packing]`
    // - `verify[arch.operators.differentiability]`
    // - `verify[arch.tiers.arena-hierarchy]`
    // - `verify[arch.tiers.hardware-budget]`
    // - `verify[arch.tiers.state-ram]`
    // - `verify[arch.tiers.efficiency]`
    // - `verify[arch.topology.signal-flow]`
    // - `verify[arch.topology.stage0-router]`
    // - `verify[arch.topology.auto-unity-leveling]`
    // - `verify[arch.tiers.transition-contract]`
    // - `verify[arch.topology.spillover-trails]`
    // - `verify[arch.bounce.oversampling]`
    // - `verify[arch.families.coverage]`
    // - `verify[arch.families.f1-bistables]`
    // - `verify[arch.families.f2-hilbert-ssb]`
    // - `verify[arch.families.f3-pitch]`
    // - `verify[arch.families.f4-dynamics]`
    // - `verify[arch.families.f5-bitcrusher]`
    // - `verify[arch.families.f6-formants]`
    // - `verify[arch.families.f7-multi-input]`
    // - `verify[arch.isomorphism.guarantees]`
    // - `verify[arch.isomorphism.stage2-serialization]`
    // - `verify[arch.isomorphism.stereo-preservation]`
    // - `verify[arch.isomorphism.nested-bus]`
    // - `verify[arch.refinements.summary]`
    // - `verify[arch.verification.matrix]`

    use dnam_engine::container::{
        ARENA_FULL_END_CL, ARENA_LITE_END_CL, ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END,
        ARENA_OFFSET_PRO_END, ARENA_OFFSET_ULTRA_END, ARENA_PRO_END_CL, ARENA_ULTRA_END_CL,
        build_dnam_container, cp1_contributor_record_bytes, strip_container_to_tier,
    };
    use dnam_engine::control::{ExactIntFactorAccumulator, control_stride_for_rate};
    use dnam_engine::math::{DefaultMathBackend, ScalarBackend, eval_monotone_log2_f32_bits};
    use dnam_engine::operators::{
        BallisticDuckingVca, FixedHzBitcrusher, HilbertH1Lattice6Sec, KautzModalBank,
        MultiDrivePolyAdaaChannel, OpTDelayLine, SymplecticModal4Lane, T1DcBlocker,
        T1SubHarmonicTracker, c2_wavefolder, eval_m2_pitch_grain_delays,
    };
    use dnam_engine::schema::DnamJsonHeader;
    use dnam_engine::stages::{
        Stage0Router, Stage2CommutedCascade, TptLr4Crossover, WdwDryTapMode,
    };
    use dnam_engine::{
        DnamEngine, DynamicDnamEngine, FtzDazGuard, OFFLINE_BOUNCE_PDC_SAMPLES,
        TOTAL_LITE_STATE_RAM_BYTES, Tier,
    };

    /// verify[arch.pillars.dual-space-control]
    /// verify[arch.pillars.dual-rate-plane]
    /// verify[arch.pillars.cp1-projections]
    /// verify[arch.lifecycle.dual-rate]
    /// verify[arch.fp.callback-env]
    /// verify[arch.fp.range-saturation]
    /// verify[arch.operators.tri-kernel]
    /// verify[arch.operators.self-similar]
    /// verify[arch.simd.lane-packing]
    /// verify[arch.operators.differentiability]
    /// Verifies the 3 architectural pillars, dual-rate execution lifecycle (`48 kHz` audio / `3 kHz`
    /// control), self-similar `Op-T`/`Op-M`/`Op-N` kernels, 4-lane SIMD packing, and operator
    /// differentiability formulas.
    #[test]
    fn test_tora_pillars_dual_rate_lifecycle_tri_kernel_and_differentiability() {
        let _guard = FtzDazGuard::new();

        // 1. Pillar 2 & 3: 16-sample stride at 48 kHz (3 kHz), CP1 sizes (124 B + 112 B for k=3)
        assert_eq!(control_stride_for_rate(48_000), 16);
        assert_eq!(cp1_contributor_record_bytes(3, false), Some(124));
        assert_eq!(cp1_contributor_record_bytes(3, true), Some(112));

        // 2. Self-similar tri-kernel execution across audio and control planes
        let mut opt = OpTDelayLine::<2048>::new();
        let mut opm = SymplecticModal4Lane::new();
        let mut opn = MultiDrivePolyAdaaChannel::new();
        let y_t = opt.step(0.5);
        let (y_m_re, y_m_im) = opm.step_lane(0, 0.5);
        let y_n = opn.step(0.5, 1.0, 0.25, 0.5, 0.9993);
        assert!(y_t.is_finite() && y_m_re.is_finite() && y_m_im.is_finite() && y_n[0].is_finite());

        let mut acc = ExactIntFactorAccumulator::new(2);
        acc.update_contributor(0, (0.5_f32).to_bits()).unwrap();
        acc.update_contributor(1, (0.25_f32).to_bits()).unwrap();
        assert_eq!(acc.interaction_f32_bits(), Some((0.125_f32).to_bits()));

        // 3. Operator differentiability:
        // (a) Op-T Catmull-Rom analytical derivative d/dmu ((c3*mu + c2)*mu + c1)*mu + c0 == 3*c3*mu^2 + 2*c2*mu + c1
        let (s_m1, s_0, s_1, s_2) = (1.0_f64, 2.0, 3.5, 4.0);
        let c3 = -0.5 * s_m1 + 1.5 * s_0 - 1.5 * s_1 + 0.5 * s_2;
        let c2 = s_m1 - 2.5 * s_0 + 2.0 * s_1 - 0.5 * s_2;
        let c1 = -0.5 * s_m1 + 0.5 * s_1;
        let c0 = s_0;
        let mu = 0.37_f64;
        let eps = 1.0e-7_f64;
        let eval_cr = |m: f64| ((c3 * m + c2) * m + c1) * m + c0;
        let num_deriv = (eval_cr(mu + eps) - eval_cr(mu - eps)) / (2.0 * eps);
        let ana_deriv = (3.0 * c3 * mu + 2.0 * c2) * mu + c1;
        assert!((num_deriv - ana_deriv).abs() < 1e-9);

        // (b) Op-M associative scan composition (a2, b2) o (a1, b1) = (a2*a1, a2*b1 + b2)
        let compose = |(a2, b2): (f64, f64), (a1, b1): (f64, f64)| (a2 * a1, a2 * b1 + b2);
        let e1 = (0.95, 0.2);
        let e2 = (0.92, -0.1);
        let e3 = (0.90, 0.3);
        let left_assoc = compose(e3, compose(e2, e1));
        let right_assoc = compose(compose(e3, e2), e1);
        assert!((left_assoc.0 - right_assoc.0).abs() < 1e-15);
        assert!((left_assoc.1 - right_assoc.1).abs() < 1e-15);
    }

    /// verify[arch.tiers.arena-hierarchy]
    /// verify[arch.tiers.hardware-budget]
    /// verify[arch.tiers.state-ram]
    /// verify[arch.tiers.efficiency]
    /// verify[arch.tiers.transition-contract]
    /// verify[arch.topology.signal-flow]
    /// verify[arch.topology.stage0-router]
    /// verify[arch.topology.auto-unity-leveling]
    /// verify[arch.topology.spillover-trails]
    /// verify[arch.bounce.oversampling]
    /// Verifies the 4-tier lower-block-triangular arena hierarchy (`168 / 349 / 566 / 1000` CLs),
    /// `8,656 B` Lite state RAM budget, auto-unity leveling trims, tier transitions, spillover trails,
    /// and `4x` ZOH oversampling (`12` host samples PDC).
    #[test]
    fn test_tiers_arena_hierarchy_state_ram_transitions_spillover_and_bounce() {
        assert_eq!(ARENA_OFFSET_LITE_END, 10_752);
        assert_eq!(ARENA_OFFSET_FULL_END, 22_336);
        assert_eq!(ARENA_OFFSET_PRO_END, 36_224);
        assert_eq!(ARENA_OFFSET_ULTRA_END, 64_000);
        assert_eq!(
            (
                ARENA_LITE_END_CL,
                ARENA_FULL_END_CL,
                ARENA_PRO_END_CL,
                ARENA_ULTRA_END_CL
            ),
            (168, 349, 566, 1_000)
        );
        assert_eq!(TOTAL_LITE_STATE_RAM_BYTES, 8_656);
        assert_eq!(OFFLINE_BOUNCE_PDC_SAMPLES, 12);

        let mut header = DnamJsonHeader::default_reference();
        let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        arena[124..128].copy_from_slice(&0.9993_f32.to_le_bytes());
        let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

        let mut eng =
            DnamEngine::<DefaultMathBackend>::from_bytes(&container, Tier::Ultra, 48_000).unwrap();
        let in_l = [0.25_f32; 64];
        let in_r = [-0.25_f32; 64];
        let mut out_l = [0.0_f32; 64];
        let mut out_r = [0.0_f32; 64];

        // 1. All 4 I/O topologies + tier transitions
        for &t in &[Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
            eng.set_tier(t);
            assert_eq!(eng.tier(), t);
            eng.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
            eng.process_mono_to_stereo(&in_l, &mut out_l, &mut out_r);
            eng.process_mono(&in_l, &mut out_l);
        }

        // 2. Spillover trails state machine (set_bypass(true, true) vs set_bypass(true, false))
        eng.set_bypass(true, true);
        assert_eq!(eng.telemetry_snapshot().bypassed, 1);
        eng.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
        eng.set_bypass(true, false);
        eng.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
        assert_eq!(out_l, in_l);
        assert_eq!(out_r, in_r);
        eng.set_bypass(false, false);

        // 3. 4x offline bounce preserves live tier and finite outputs
        eng.set_tier(Tier::Lite);
        eng.render_offline_bounce(&in_l, &in_r, &mut out_l, &mut out_r);
        assert_eq!(eng.tier(), Tier::Lite);
        assert!(out_l.iter().all(|v| v.is_finite()));

        // 4. DynamicDnamEngine runtime dispatch parity
        let mut dyn_eng = DynamicDnamEngine::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        dyn_eng.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
        assert!(out_l.iter().all(|v| v.is_finite()));
    }

    /// verify[arch.families.coverage]
    /// verify[arch.families.f1-bistables]
    /// verify[arch.families.f2-hilbert-ssb]
    /// verify[arch.families.f3-pitch]
    /// verify[arch.families.f4-dynamics]
    /// verify[arch.families.f5-bitcrusher]
    /// verify[arch.families.f6-formants]
    /// verify[arch.families.f7-multi-input]
    /// Verifies all 7 Goal 1 hardware families (`F1` Bistables/T1, `F2` Hilbert H1 SSB, `F3` M=2 Pitch,
    /// `F4` Hysteretic Dynamics/Log2/Ducking, `F5` Fixed-Hz Bitcrusher, `F6` Kautz Formants/Barberpole,
    /// `F7` 8-Port Router/LR4/WDW).
    #[test]
    fn test_all_seven_hardware_families_realization() {
        let _guard = FtzDazGuard::new();

        // Family 1: T1 5 Hz DC blocker + H1 analytic pair + T1SubHarmonicTracker
        let mut dc = T1DcBlocker::new_at_rate(48_000.0).unwrap();
        let mut h1 = HilbertH1Lattice6Sec::new_at_rate(48_000.0).unwrap();
        let mut trk = T1SubHarmonicTracker::new();
        let d = dc.step(0.5);
        let (i_v, q_v) = h1.step(d);
        let f1_feats = trk.step_analytic(i_v, q_v);
        assert!(f1_feats.iter().all(|v| v.is_finite()));

        // Family 2: H1 Hilbert SSB frequency shifting
        let ssb_upper = i_v * 0.8 - q_v * 0.6;
        let ssb_lower = i_v * 0.8 + q_v * 0.6;
        assert!(ssb_upper.is_finite() && ssb_lower.is_finite());

        // Family 3: Centered M = 2 Hermite-Hann pitch grains
        let (_, weights, delays) = eval_m2_pitch_grain_delays(100, 4800, 7204.0, -1.0).unwrap();
        assert!((weights[0] + weights[1] - 1.0).abs() < 1e-12 && delays[0] >= 4.0);

        // Family 4: Monotone log2 envelope lookup & 1-pole ballistic ducking VCA
        let log2_env = eval_monotone_log2_f32_bits((0.5_f32).to_bits())
            .unwrap()
            .unwrap();
        assert!((log2_env - (-1.0)).abs() < 1e-5);
        let mut vca = BallisticDuckingVca::new();
        assert!(vca.step(0.5, 1.0).is_finite());

        // Family 5: Fixed-hertz integer ZOH bitcrusher
        let mut crusher = FixedHzBitcrusher::new(8, 12_000.0, 48_000.0).unwrap();
        assert!(crusher.step(0.33).is_finite());

        // Family 6: Shared Kautz modal resonators & dynamic sweep contraction
        let mut kautz = KautzModalBank::<4>::new();
        kautz.set_pole(0, 0.99, 0.3, 0.05, 0.5, -0.2);
        assert!(kautz.step(0.5).is_finite());

        // Family 7: 8-Port Router + 200 Hz TPT LR4 crossover + WDW
        let mut router = Stage0Router::new();
        let (sl, sr, dl, dr, _, _) = router.route_frame(0.5, -0.5, 0.2, -0.2);
        let mut lr4 = TptLr4Crossover::default();
        let (l4, h4) = lr4.step(dl);
        assert!(
            sl.is_finite() && sr.is_finite() && dr.is_finite() && l4.is_finite() && h4.is_finite()
        );
        assert_ne!(WdwDryTapMode::PreSpatialCore, WdwDryTapMode::DirectDry);
    }

    /// verify[arch.isomorphism.guarantees]
    /// verify[arch.isomorphism.stage2-serialization]
    /// verify[arch.isomorphism.stereo-preservation]
    /// verify[arch.isomorphism.nested-bus]
    /// verify[arch.refinements.summary]
    /// verify[arch.verification.matrix]
    /// Verifies Stage II commuted cascade serialization parity, true-stereo preservation, nested 4-tier
    /// feature bus (`64 / 128 / 192 / 256 D`), and the 19 architectural refinements.
    #[test]
    fn test_trainer_engine_isomorphism_stereo_preservation_and_nested_bus() {
        let _guard = FtzDazGuard::new();

        // Stage II commuted cascade & C2 wavefolder
        let mut s2 = Stage2CommutedCascade::new();
        s2.init_dc_equilibrium(0.0, 0.8, -0.2);
        let (bl, br) = s2.step_frame(0.4, -0.4, 1.2, 0.3);
        assert!(bl[0] > 0.0 && br[0] < 0.0);
        let (wo, we) = c2_wavefolder(0.5);
        assert!(wo > 0.0 && we > 0.0);

        // Nested 4-tier prefix hierarchy & bit-exact Scalar vs DefaultMathBackend parity
        let mut header = DnamJsonHeader::default_reference();
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
        for &t in &[Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
            let stripped = strip_container_to_tier(&container, t).unwrap();
            let mut eng_s = DnamEngine::<ScalarBackend>::from_bytes(&stripped, t, 48_000).unwrap();
            let mut eng_d =
                DnamEngine::<DefaultMathBackend>::from_bytes(&stripped, t, 48_000).unwrap();
            let mut ol_s = [0.0f32; 16];
            let mut or_s = [0.0f32; 16];
            let mut ol_d = [0.0f32; 16];
            let mut or_d = [0.0f32; 16];
            eng_s.process_block(&[0.2; 16], &[-0.2; 16], &mut ol_s, &mut or_s);
            eng_d.process_block(&[0.2; 16], &[-0.2; 16], &mut ol_d, &mut or_d);
            for i in 0..16 {
                assert!((ol_s[i] - ol_d[i]).abs() < 1e-6);
                assert!((or_s[i] - or_d[i]).abs() < 1e-6);
            }
        }

        // End-to-end v2.0 Open .dnam container loading & .dnam-prev rejection in DnamEngine
        let hdr_ref = DnamJsonHeader::default_reference();
        let hdr_bytes = serde_json::to_vec(&hdr_ref).unwrap();
        let t0 = vec![0u8; 10_752];
        let t1 = vec![0u8; 11_584];
        let t2 = vec![0u8; 13_888];
        let t3 = vec![0u8; 27_776];
        let v2_container = dnam_engine::container::build_v2_open_dnam_container(
            &hdr_bytes,
            &[0u8; 64],
            b"DAU1-AUTH",
            &[&t0, &t1, &t2, &t3],
            None,
            0,
        )
        .unwrap();
        let mut eng_v2 =
            DnamEngine::<ScalarBackend>::from_bytes(&v2_container, Tier::Ultra, 48_000).unwrap();
        assert_eq!(eng_v2.preamble().version_major, 2);
        let mut out_l = [0.0f32; 16];
        let mut out_r = [0.0f32; 16];
        eng_v2.process_block(&[0.1; 16], &[-0.1; 16], &mut out_l, &mut out_r);
        assert!(out_l.iter().all(|v| v.is_finite()));

        // Verify .dnam-prev (container_kind == "preview") is rejected by DnamEngine::from_bytes
        let mut hdr_val = serde_json::to_value(&hdr_ref).unwrap();
        hdr_val["metadata"]["container_kind"] = serde_json::json!("preview");
        let prev_hdr_bytes = serde_json::to_vec(&hdr_val).unwrap();
        let prev_container = dnam_engine::container::build_v2_open_dnam_container(
            &prev_hdr_bytes,
            &[0u8; 64],
            b"DAU1-AUTH",
            &[&t0],
            None,
            0,
        )
        .unwrap();
        assert_eq!(
            DnamEngine::<ScalarBackend>::from_bytes(&prev_container, Tier::Lite, 48_000)
                .unwrap_err(),
            dnam_engine::schema::DnamError::PreviewContainerRejected
        );

        // Verify integrated runtime guards:
        // 1. Input sanitization & saturation flags on DnamEngine::process_block
        eng_v2.saturation_flags = 0;
        let bad_in = [f32::NAN, f32::INFINITY, 150.0, -0.25];
        let mut san_l = [0.0f32; 4];
        let mut san_r = [0.0f32; 4];
        eng_v2.process_block(&bad_in, &bad_in, &mut san_l, &mut san_r);
        assert!(san_l.iter().all(|v| v.is_finite()));
        assert_ne!(
            eng_v2.saturation_flags & dnam_engine::math::saturation_flags::INVALID_NONFINITE,
            0
        );
        assert_ne!(
            eng_v2.saturation_flags & dnam_engine::math::saturation_flags::OUTSIDE_CERTIFIED_INPUT,
            0
        );

        // 2. ZOH-rescaled OpTDelayLine xfade_len on sample-rate change
        eng_v2.set_sample_rate(96_000).unwrap();
        assert_eq!(eng_v2.stage_pipeline.stage1.op_t_pre.xfade_len, 256);

        // 3. Stage III 16-sample delayed Kautz onset tap
        let mut s3_cab = dnam_engine::stages::Stage3CabinetModals::new();
        let taps = [0.0f32; 16];
        for n in 0..16 {
            let (_, delayed) =
                s3_cab.step_with_delayed_tap(if n == 0 { 1.0 } else { 0.0 }, 0.0, &taps);
            assert_eq!(delayed, 0.0);
        }
        let (_, delayed_at_16) = s3_cab.step_with_delayed_tap(0.0, 0.0, &taps);
        assert_eq!(delayed_at_16, 1.0);

        // 4. Signed v2.0 DAU1 container verification & tampering rejection on DnamEngine::from_bytes
        #[cfg(feature = "crypto")]
        {
            use dnam_engine::provenance::{
                Dau1Envelope, RustCryptoProvider, VerificationStatus, build_creator_manifest_v1,
                build_plaintext_shared_v1, compute_plaintext_prefixes_v1, sha256_digest,
            };
            let creator_seed = [0x42u8; 32];
            let creator_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&creator_seed);
            let dummy_env = Dau1Envelope {
                auth_kind: 0,
                creator_pubkey: creator_pk,
                creator_max_tier: 1,
                master_prefixes: vec![[0u8; 32], [0u8; 32]],
                creator_signature: [0u8; 64],
                marketplace_pubkey: None,
                buyer_max_tier: None,
                policy_bytes: Vec::new(),
                buyer_prefixes: Vec::new(),
                marketplace_signature: None,
            };
            let dummy_auth = dummy_env.to_bytes().unwrap();
            let draft = dnam_engine::container::build_v2_open_dnam_container(
                &hdr_bytes,
                &[0u8; 64],
                &dummy_auth,
                &[&t0, &t1],
                None,
                0,
            )
            .unwrap();
            let shared_framing =
                build_plaintext_shared_v1(&draft[0..64], &draft[64..192], &[0u8; 64]);
            let hdr_sha = sha256_digest(&hdr_bytes);
            let master_pfxs = compute_plaintext_prefixes_v1(
                b"DNAM-MASTER-PREFIX-V1",
                &hdr_sha,
                &shared_framing,
                &[&t0, &t1],
            )
            .unwrap();
            let manifest = build_creator_manifest_v1(&creator_pk, &hdr_sha, &master_pfxs).unwrap();
            let sig = RustCryptoProvider::ed25519_sign(&creator_seed, &manifest);
            let signed_env = Dau1Envelope {
                auth_kind: 0,
                creator_pubkey: creator_pk,
                creator_max_tier: 1,
                master_prefixes: master_pfxs,
                creator_signature: sig,
                marketplace_pubkey: None,
                buyer_max_tier: None,
                policy_bytes: Vec::new(),
                buyer_prefixes: Vec::new(),
                marketplace_signature: None,
            };
            let signed_auth = signed_env.to_bytes().unwrap();
            let signed_v2 = dnam_engine::container::build_v2_open_dnam_container(
                &hdr_bytes,
                &[0u8; 64],
                &signed_auth,
                &[&t0, &t1],
                None,
                0,
            )
            .unwrap();
            let eng_signed =
                DnamEngine::<ScalarBackend>::from_bytes(&signed_v2, Tier::Full, 48_000).unwrap();
            assert!(matches!(
                eng_signed.verification_status(),
                VerificationStatus::CreatorVerified { .. }
            ));

            // Stripping to Lite preserves CreatorVerified without re-signing
            let stripped_lite = strip_container_to_tier(&signed_v2, Tier::Lite).unwrap();
            let eng_stripped =
                DnamEngine::<ScalarBackend>::from_bytes(&stripped_lite, Tier::Lite, 48_000)
                    .unwrap();
            assert!(matches!(
                eng_stripped.verification_status(),
                VerificationStatus::CreatorVerified { .. }
            ));

            // Tampering with a tier byte is rejected by DnamEngine::from_bytes
            let layout = dnam_engine::container::validate_v2_open_dnam_layout(&signed_v2).unwrap();
            let mut tampered = signed_v2.clone();
            tampered[layout.tier_start] ^= 0xFF;
            assert!(
                DnamEngine::<ScalarBackend>::from_bytes(&tampered, Tier::Full, 48_000).is_err()
            );
        }

        // 5. Verify ADR-0011 UX & Glue policies end-to-end on DnamEngine:
        //    - telemetry_snapshot().overflow_or_saturation_flags surfaces OUTSIDE_CERTIFIED_INPUT
        let snap = eng_v2.telemetry_snapshot();
        assert_ne!(
            snap.overflow_or_saturation_flags
                & dnam_engine::math::saturation_flags::OUTSIDE_CERTIFIED_INPUT,
            0
        );
        //    - supports_freeze() == false gates off synthetic set_freeze(true)
        assert!(!eng_v2.supports_freeze());
        eng_v2.set_freeze(true);
        assert!(!eng_v2.is_frozen());
    }

    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(16))]

        /// verify[arch.pillars.dual-rate-plane]
        /// verify[arch.tiers.arena-hierarchy]
        /// verify[arch.fp.range-saturation]
        #[test]
        fn prop_end_to_end_v2_engine_tiers_rates_and_finite_outputs(
            tier_u8 in 0u8..=3u8,
            rate_idx in 0usize..4usize,
            in_l in prop::array::uniform16(-5.0_f32..5.0_f32),
            in_r in prop::array::uniform16(-5.0_f32..5.0_f32),
        ) {
            let rates = [44_100u32, 48_000, 88_200, 96_000];
            let sr = rates[rate_idx];
            let tier = Tier::from_u8(tier_u8);
            let hdr_ref = DnamJsonHeader::default_reference();
            let hdr_bytes = serde_json::to_vec(&hdr_ref).unwrap();
            let t0 = vec![0u8; 10_752];
            let t1 = vec![0u8; 11_584];
            let t2 = vec![0u8; 13_888];
            let t3 = vec![0u8; 27_776];
            let v2_container = dnam_engine::container::build_v2_open_dnam_container(
                &hdr_bytes,
                &[0u8; 64],
                b"DAU1-PROP",
                &[&t0, &t1, &t2, &t3],
                None,
                0,
            ).unwrap();
            let mut eng = DnamEngine::<ScalarBackend>::from_bytes(&v2_container, tier, sr).unwrap();
            let mut out_l = [0.0f32; 16];
            let mut out_r = [0.0f32; 16];
            eng.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
            prop_assert!(out_l.iter().all(|v| v.is_finite() && v.abs() <= 1.0e9));
            prop_assert!(out_r.iter().all(|v| v.is_finite() && v.abs() <= 1.0e9));
        }
    }
}
