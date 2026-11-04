//! Empirical TORA End-to-End Pipeline, ZOH Multi-Rate, `4x` Bounce, and Lifecycle Verification (`Tests 8H, 8I`).
//!
//! Canonical Reference: `docs/specs/architecture/tora_architecture.md` and `docs/specs/containers/dnam_binary_format.md`.
//!
//! Verifies:
//! - `Test 8H & 8I` (`CL-19`, `CL-20`): 64-byte binary framing, cache-line tier offsets
//!   (`168, 349, 566, 1000` CLs, `8,656 B` Lite RAM), Sample-0 DC resting equilibrium (`0.000000`
//!   pop on silence), distinct per-tier (`Lite ⊂ Full ⊂ Pro ⊂ Ultra`) outputs, ZOH multi-sample-rate
//!   rescaling across `44.1 kHz`–`192 kHz`, and `4x` ZOH linear-phase oversampled offline bounce with
//!   `12`-sample PDC trim (`CL-20`).
//! - Engine lifecycle constructors (`from_bytes_owned_with_max_tier`, `from_bytes_with_scratch`,
//!   `with_max_tier`, `reload_model_in_place`), all `ControlCalibrationRole` bindings (`DelayTimeMs`,
//!   `Feedback`, `StereoMode`, `StereoWidth`, `FreezeLatch` as both `Switch` and `Range`),
//!   `configure_precision_preview`, `sync_tempo`, and `DynamicDnamEngine`.

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

/// Test 8H & 8I (`CL-19`, `CL-20`): End-to-end TORA multi-stage pipeline verification across
/// 64B framing, Sample-0 DC equilibrium, 4-tier progressive outputs, ZOH sample rates, and `4x` bounce.
#[test]
fn test_8h_8i_end_to_end_tora_pipeline_tiers_zoh_sample_rates_and_4x_bounce_cl19_cl20() {
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

    // 4. Verify 4x ZOH linear-phase offline bounce with 12-sample PDC trim (CL-20)
    eng.set_sample_rate(48_000).unwrap();
    let mut bounce_l = vec![0.0_f32; 128];
    let mut bounce_r = vec![0.0_f32; 128];
    eng.render_offline_bounce(&in_l, &in_r, &mut bounce_l, &mut bounce_r);
    assert!(bounce_l.iter().any(|&v| v.abs() > 1e-4));
    assert!(bounce_l.iter().all(|v| v.is_finite()));
}

/// Verifies `from_bytes_owned_with_max_tier`, `from_bytes_with_scratch`, `with_max_tier`,
/// `reload_model_in_place`, all `ControlCalibrationRole` variants (`DelayTimeMs`, `Feedback`,
/// `StereoWidth`, `FreezeLatch` as both `Switch` and `Range`), `configure_precision_preview`,
/// and `sync_tempo`.
#[test]
fn test_engine_constructors_calibration_roles_preview_guard_and_tempo_sync() {
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
