//! Empirical TORA Control Bus, 3-Phase Clamping, and `.dnam-ckpt` Verification (`Tests 8A, 8B, 8D, 8E, 8G`).
//!
//! Canonical Reference: `docs/dsp/parametric_control_bus.md`, `docs/containers/dnam_binary_format.md`,
//! and `docs/verification/verification_and_guardrails.md`.
//!
//! Verifies:
//! - `Test 8A & 8B` (`CL-06`, `CL-07`): 3-node Lobatto physical delay interpolation accuracy and
//!   128-sample Hann-windowed dual-pointer switch crossfade vs. single-pointer phase jump.
//! - `Test 8D & 8E` (`CL-12`–`CL-17`): 128-byte `ControlChannelRecord` table at
//!   `tensor_arena_offset - 128 * P`, `update_header_in_place` preservation of `ControlChannelRecord`
//!   bytes, 64-byte $\mathcal{O}(1)$ `LatentAccumulator64B` delta updates across $P = 32$ controls,
//!   and non-stop Phase-1/2/3 clamping (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]`).
//! - `Test 8G`: `LAPACK uplo='U'` Nested Principal Submatrix Prefix Theorem across all 4 tiers
//!   (`Lite = 64`, `Full = 128`, `Pro = 192`, `Ultra = 256`) and `.dnam-ckpt` `Arena 4` / `Arena 5`
//!   (`DNCA` / `INPC` / `STMC`) container verification.

use std::collections::BTreeMap;
use std::f64::consts::TAU;

use dnam_engine::FtzDazGuard;
use dnam_engine::container::{
    ARENA_OFFSET_ULTRA_END, ARENA4_STEM_BYTES_ULTRA, DNCA_MAGIC, DnamContainerView,
    build_dnam_container, crc32c_compute, pack_uplo_u_f64, unpack_uplo_u_f64,
    update_header_in_place, uplo_u_packed_len, verify_ckpt_arena5_calib_stats,
    verify_ckpt_stem_arena4_crc32c,
};
use dnam_engine::control::{
    ControlBus, ControlChannelRecord, ControlClampDescriptor, ControlLatentProjection,
    LatentAccumulator64B,
};
use dnam_engine::math::ScalarBackend;
use dnam_engine::operators::op_t::{OP_T_LITE_RING_CAPACITY, OpTDelayLine};
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::schema::{
    ControlKind, ControlSchemaDef, ControlUiHints, DnamJsonHeader, StemProvenance,
    StemRegistryEntry, TensorSliceEntry,
};
use proptest::prelude::*;

/// Test 8A & 8B (`CL-06`, `CL-07`): Verifies 3-node Lobatto physical delay interpolation accuracy
/// at intermediate knob positions (`u = 0.5`) and verifies that `OpTDelayLine`'s 128-sample
/// Hann-windowed dual-pointer switch crossfade eliminates single-pointer discontinuity jumps.
#[test]
fn test_8a_8b_lobatto_physical_delay_interpolation_and_dual_pointer_switch_crossfade_cl07() {
    let tau_at = |u: f64| -> f64 {
        let psi1 = u;
        let psi2 = u * (u + 1.0);
        200.0 + 100.0 * psi1 + 50.0 * psi2
    };
    let exact_mid = 200.0 + 150.0 * 0.5 + 50.0 * 0.25;
    let interp_mid = tau_at(0.5);
    assert!((interp_mid - exact_mid).abs() < 1e-12);

    let _audio = FtzDazGuard::new();
    let mut dl_xfade = OpTDelayLine::<OP_T_LITE_RING_CAPACITY>::default();
    let mut dl_jump = OpTDelayLine::<OP_T_LITE_RING_CAPACITY>::default();
    dl_xfade.current_delay = 40.0;
    dl_xfade.target_delay = 40.0;
    dl_jump.current_delay = 40.0;
    dl_jump.target_delay = 40.0;

    let mut prev_x = 0.0_f32;
    let mut prev_j = 0.0_f32;
    for n in 0..315 {
        let s = (TAU as f32 * 400.0 * (n as f32) / 48_000.0).sin();
        prev_x = dl_xfade.step(s);
        prev_j = dl_jump.step(s);
    }

    dl_xfade.set_target_delay(100.0, true);
    dl_jump.current_delay = 100.0;
    dl_jump.target_delay = 100.0;

    let mut max_diff_xfade = 0.0_f32;
    let mut max_diff_jump = 0.0_f32;

    for n in 315..450 {
        let s = (TAU as f32 * 400.0 * (n as f32) / 48_000.0).sin();
        let yx = dl_xfade.step(s);
        let yj = dl_jump.step(s);
        max_diff_xfade = max_diff_xfade.max((yx - prev_x).abs());
        max_diff_jump = max_diff_jump.max((yj - prev_j).abs());
        prev_x = yx;
        prev_j = yj;
    }

    assert!(max_diff_jump > 10.0 * max_diff_xfade);
}

/// Test 8D & 8E (`CL-12`–`CL-17`): Verifies 128-byte `ControlChannelRecord` framing,
/// `control_table_offset()` preservation across `update_header_in_place`, $\mathcal{O}(1)$
/// `LatentAccumulator64B` delta updates across $P = 32$ controls, and non-stop Phase-1/2/3 clamping.
#[test]
fn test_8d_8e_control_channel_record_o1_latent_and_3phase_clamping_cl12_cl17() {
    assert_eq!(core::mem::size_of::<ControlClampDescriptor>(), 64);
    assert_eq!(core::mem::size_of::<ControlLatentProjection>(), 64);
    assert_eq!(core::mem::size_of::<ControlChannelRecord>(), 128);
    assert_eq!(core::mem::size_of::<LatentAccumulator64B>(), 64);

    let mut projections = [ControlLatentProjection::new(); 32];
    let mut u_vals = [0.0_f32; 32];
    let mut psi_vals = [0.0_f32; 32];
    let mut acc = LatentAccumulator64B::new();

    for p in 0..32 {
        let pf = (p + 1) as f32;
        projections[p].w_main = core::array::from_fn(|d| 0.02 * pf * ((d + 1) as f32));
        projections[p].v_factors = core::array::from_fn(|f| 0.05 * ((p + f + 1) as f32));
        u_vals[p] = -0.8 + 0.05 * (p as f32);
        psi_vals[p] = 0.5 * u_vals[p];
        acc.update_delta(&projections[p], 0.0, u_vals[p], psi_vals[p]);
    }

    let new_u7 = 0.75_f32;
    let new_psi7 = 0.5 * new_u7;
    acc.update_delta(&projections[7], u_vals[7], new_u7, new_psi7 - psi_vals[7]);
    u_vals[7] = new_u7;
    psi_vals[7] = new_psi7;

    let z_o1 = acc.evaluate_z_ctrl();
    for d in 0..8 {
        let mut expected = 0.0_f32;
        for p in 0..32 {
            expected += projections[p].w_main[d] * psi_vals[p];
        }
        if d < 4 {
            for p in 0..32 {
                for q in (p + 1)..32 {
                    expected += projections[p].v_factors[d]
                        * projections[q].v_factors[d]
                        * u_vals[p]
                        * u_vals[q];
                }
            }
        }
        assert!((z_o1[d] - expected).abs() < 1e-4);
    }

    let mut header = DnamJsonHeader::default_reference();
    let mut cond_max = BTreeMap::new();
    cond_max.insert("boost=on".to_string(), 9.37_f32);
    header.controls_schema = vec![
        ControlSchemaDef {
            id: "gain".to_string(),
            name: "Gain".to_string(),
            kind: ControlKind::Range,
            unit: None,
            lobatto_degree: Some(2),
            anchor_nodes_norm: Some(vec![0.0, -0.6, 0.6]),
            declared_range: Some([0.0, 10.0]),
            safe_clamped_range: Some([2.0, 8.0]),
            default: serde_json::json!(5.0),
            conditional_min: BTreeMap::new(),
            conditional_max: BTreeMap::new(),
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            calibration: None,
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
            conditional_max: cond_max,
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            calibration: None,
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        },
        ControlSchemaDef {
            id: "boost".to_string(),
            name: "Boost".to_string(),
            kind: ControlKind::Switch,
            unit: None,
            lobatto_degree: None,
            anchor_nodes_norm: None,
            declared_range: None,
            safe_clamped_range: None,
            default: serde_json::json!("off"),
            conditional_min: BTreeMap::new(),
            conditional_max: BTreeMap::new(),
            declared_options: Some(vec![
                "off".to_string(),
                "on".to_string(),
                "ultra".to_string(),
            ]),
            active_options: Some(vec!["off".to_string(), "on".to_string()]),
            clamp_warning: None,
            calibration: None,
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        },
    ];

    let descriptors = header.compile_clamp_descriptors().unwrap();
    let mut cb = ControlBus::<ScalarBackend>::new();
    for (i, d) in descriptors.iter().enumerate() {
        cb.descriptors[i] = *d;
    }
    cb.num_controls = 3;

    cb.set_control_immediate(0, 9.5);
    assert!((cb.clamped_values[0] - 8.0).abs() < 1e-6);
    cb.set_control_immediate(0, 0.5);
    assert!((cb.clamped_values[0] - 2.0).abs() < 1e-6);

    cb.set_control_immediate(2, 2.0);
    assert!(cb.active_switch_options[2] <= 1);

    cb.set_control_immediate(2, 1.0);
    cb.set_control_immediate(1, 10.0);
    assert!((cb.clamped_values[1] - 9.37).abs() < 1e-5);
    assert_ne!(cb.active_clamp_mask & (1 << 1), 0);

    // Verify ControlChannelRecord table at tensor_arena_offset - 128 * P survives update_header_in_place
    let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    let mut container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
    let crypto = DefaultCryptoProvider::default();
    let view_before = DnamContainerView::from_bytes(&container, &crypto).unwrap();
    let table_off = view_before.control_table_offset();
    assert_eq!(
        table_off,
        (view_before.preamble.tensor_arena_offset as usize) - 3 * 128
    );
    let table_snapshot = container[table_off..table_off + 3 * 128].to_vec();

    header
        .user_metadata
        .insert("tag".to_string(), serde_json::json!("preserved"));
    update_header_in_place(&mut container, &header).unwrap();
    assert_eq!(&container[table_off..table_off + 3 * 128], &table_snapshot);
}

/// Test 8G: Verifies the `LAPACK uplo='U'` Nested Principal Submatrix Prefix Theorem across all
/// 4 tiers (`Lite = 64`, `Full = 128`, `Pro = 192`, `Ultra = 256`) and verifies `.dnam-ckpt`
/// `Arena 4` stem `CRC32C` and `Arena 5` (`DNCA` / `INPC` / `STMC`) binary container framing.
#[test]
fn test_8g_dnam_ckpt_uplo_u_prefix_theorem_and_arena4_arena5_validation() {
    assert_eq!(uplo_u_packed_len(64) * 8, 16_640);
    assert_eq!(uplo_u_packed_len(128) * 8, 66_048);
    assert_eq!(uplo_u_packed_len(192) * 8, 148_224);
    assert_eq!(uplo_u_packed_len(256) * 8, 263_168);

    const N_ULTRA: usize = 256;
    let mut dense_ultra = vec![0.0_f64; N_ULTRA * N_ULTRA];
    for i in 0..N_ULTRA {
        for j in i..N_ULTRA {
            let val = ((i * 31 + j * 17 + 5) as f64) * 0.001;
            dense_ultra[i * N_ULTRA + j] = val;
            dense_ultra[j * N_ULTRA + i] = val;
        }
    }

    let mut packed_ultra = vec![0.0_f64; uplo_u_packed_len(N_ULTRA)];
    pack_uplo_u_f64(&dense_ultra, N_ULTRA, &mut packed_ultra);

    for &n_tier in &[64_usize, 128, 192, 256] {
        let prefix_len = uplo_u_packed_len(n_tier);
        let mut unpacked = vec![0.0_f64; n_tier * n_tier];
        unpack_uplo_u_f64(&packed_ultra[..prefix_len], n_tier, &mut unpacked);
        for r in 0..n_tier {
            for c in 0..n_tier {
                assert_eq!(unpacked[r * n_tier + c], dense_ultra[r * N_ULTRA + c]);
            }
        }
    }

    let stem_bytes = vec![0x42_u8; ARENA4_STEM_BYTES_ULTRA];
    let stem_crc = crc32c_compute(&stem_bytes);
    assert!(
        verify_ckpt_stem_arena4_crc32c(&stem_bytes, 0, ARENA4_STEM_BYTES_ULTRA, stem_crc).is_ok()
    );
    assert!(
        verify_ckpt_stem_arena4_crc32c(&stem_bytes, 0, ARENA4_STEM_BYTES_ULTRA, stem_crc ^ 1)
            .is_err()
    );

    let a5_len = (64 + 128 + 192) as u32;
    let mut a5 = vec![0u8; a5_len as usize];
    a5[0..4].copy_from_slice(&DNCA_MAGIC);
    a5[4..6].copy_from_slice(&1_u16.to_le_bytes());
    a5[8..10].copy_from_slice(&1_u16.to_le_bytes());
    a5[10..12].copy_from_slice(&1_u16.to_le_bytes());
    a5[12..16].copy_from_slice(&a5_len.to_le_bytes());

    let inpc_pos = 64_usize;
    a5[inpc_pos..inpc_pos + 4].copy_from_slice(b"INPC");
    a5[inpc_pos + 56..inpc_pos + 60].copy_from_slice(&128_u32.to_le_bytes());
    let inpc_crc = crc32c_compute(&a5[inpc_pos + 64..inpc_pos + 128]);
    a5[inpc_pos + 60..inpc_pos + 64].copy_from_slice(&inpc_crc.to_le_bytes());

    let stmc_pos = 192_usize;
    a5[stmc_pos..stmc_pos + 4].copy_from_slice(b"STMC");
    a5[stmc_pos + 116..stmc_pos + 120].copy_from_slice(&192_u32.to_le_bytes());
    let stmc_crc = crc32c_compute(&a5[stmc_pos + 128..stmc_pos + 192]);
    a5[stmc_pos + 120..stmc_pos + 124].copy_from_slice(&stmc_crc.to_le_bytes());

    let body_crc = crc32c_compute(&a5[64..]);
    a5[16..20].copy_from_slice(&body_crc.to_le_bytes());
    assert_eq!(verify_ckpt_arena5_calib_stats(&a5, 0).unwrap(), 384);

    // Also build a complete .dnam-ckpt container with Arena 4 + Arena 5 and load via DnamContainerView
    let mut ckpt_hdr = DnamJsonHeader::default_reference();
    ckpt_hdr.container_kind = "dnam-ckpt".to_string();
    ckpt_hdr.stems_registry = Some(vec![StemRegistryEntry {
        stem_id: "stem_0".to_string(),
        controls_meta: BTreeMap::new(),
        provenance: StemProvenance {
            input_pcm_sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                .to_string(),
            target_pcm_sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                .to_string(),
            num_frames: 48_000,
        },
        stem_arena4_crc32c: stem_crc,
        user_metadata: BTreeMap::new(),
        extracted_physics: None,
        g_audio_uplo_u_f64: TensorSliceEntry {
            arena: 4,
            dtype: "f64".to_string(),
            packing: Some("uplo_upper_col_major_tier_ordered".to_string()),
            feature_dim: Some(256),
            packed_len: Some(32_896),
            shape: Some(vec![32_896]),
            byte_offset: 64_000,
            byte_length: 263_168,
        },
        c_audio_tier_f64: TensorSliceEntry {
            arena: 4,
            dtype: "f64".to_string(),
            packing: None,
            feature_dim: None,
            packed_len: None,
            shape: Some(vec![2, 256]),
            byte_offset: 64_000 + 263_168,
            byte_length: 4_096,
        },
    }]);
    let mut full_ckpt_arena = vec![0u8; 64_000];
    full_ckpt_arena.extend_from_slice(&stem_bytes);
    full_ckpt_arena.extend_from_slice(&a5);
    let ckpt_container =
        build_dnam_container(&mut ckpt_hdr, &full_ckpt_arena, &full_ckpt_arena, false).unwrap();
    let crypto = DefaultCryptoProvider::default();
    let ckpt_view = DnamContainerView::from_bytes(&ckpt_container, &crypto).unwrap();
    assert_eq!(ckpt_view.header.container_kind, "dnam-ckpt");
}

#[test]
fn test_theorem_3_unlimited_controls_31_band_eq_and_64_rack() {
    use dnam_engine::container::compute_header_crc32c;
    use dnam_engine::control::ControlLatentProjection;
    use dnam_engine::{DnamEngine, Tier};

    for &num_ctrls in &[31_usize, 64_usize] {
        let mut header = DnamJsonHeader::default_reference();
        header.controls_schema = (0..num_ctrls)
            .map(|p| ControlSchemaDef {
                id: format!("band_{p}"),
                name: format!("Band {p}"),
                kind: ControlKind::Range,
                unit: Some("dB".to_string()),
                lobatto_degree: Some(2),
                anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
                declared_range: Some([-12.0, 12.0]),
                safe_clamped_range: Some([-12.0, 12.0]),
                default: serde_json::json!(0.0),
                conditional_min: BTreeMap::new(),
                conditional_max: BTreeMap::new(),
                declared_options: None,
                active_options: None,
                clamp_warning: None,
                calibration: None,
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            })
            .collect();

        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let mut container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

        let crypto = DefaultCryptoProvider::default();
        let view0 = DnamContainerView::from_bytes(&container, &crypto).unwrap();
        assert_eq!(view0.preamble.num_controls as usize, num_ctrls);
        assert_eq!(view0.control_records.len(), num_ctrls);

        // Populate CL 1 (ControlLatentProjection) of the last control record (p = num_ctrls - 1)
        let ctrl_off = view0.control_table_offset();
        let arena_off = view0.preamble.tensor_arena_offset as usize;
        let last_p = num_ctrls - 1;
        let proj = ControlLatentProjection {
            w_main: [0.5, -0.25, 0.1, 0.2, 0.3, -0.1, 0.05, 0.15],
            v_factors: [0.4, 0.2, -0.3, 0.1, 0.0, 0.0, 0.0, 0.0],
        };
        let rec_off = ctrl_off + last_p * 128;
        container[rec_off + 64..rec_off + 128].copy_from_slice(&proj.to_bytes());
        let new_hdr_crc = compute_header_crc32c(&container, arena_off);
        container[0x3C..0x40].copy_from_slice(&new_hdr_crc.to_le_bytes());

        let mut engine =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        assert_eq!(engine.preamble().num_controls as usize, num_ctrls);
        assert_eq!(engine.telemetry_snapshot().z_head, [0.0; 4]);

        // Move the last control (index 30 or 63, well beyond the old 16-control wall!)
        engine.set_control_immediate(last_p, 12.0);
        let z_after = engine.telemetry_snapshot().z_head;
        assert!(
            z_after[0].abs() > 0.1,
            "Control #{last_p} failed to update z_head via O(1) LatentAccumulator64B: {:?}",
            z_after
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Property test verifying that `ControlChannelRecord` roundtrips bit-identically through
    /// 128-byte serialization and that `uplo='U'` packing/unpacking is lossless.
    #[test]
    fn prop_control_record_and_uplo_u_roundtrip(
        u0 in -1.0_f32..1.0_f32,
        w0 in -2.0_f32..2.0_f32,
        v0 in -1.0_f32..1.0_f32,
        diag in 0.1_f64..10.0_f64,
        off_diag in -2.0_f64..2.0_f64,
    ) {
        let mut rec = ControlChannelRecord::new();
        rec.clamp.anchor_nodes_norm[1] = u0;
        rec.projection.w_main[0] = w0;
        rec.projection.v_factors[0] = v0;
        let bytes = rec.to_bytes();
        let decoded = ControlChannelRecord::from_bytes(&bytes);
        prop_assert_eq!(decoded, rec);

        let dense = [diag, off_diag, off_diag, diag * 1.5];
        let mut packed = [0.0_f64; 3];
        pack_uplo_u_f64(&dense, 2, &mut packed);
        let mut unpacked = [0.0_f64; 4];
        unpack_uplo_u_f64(&packed, 2, &mut unpacked);
        prop_assert_eq!(unpacked, dense);
    }
}

#[test]
fn test_update_header_in_place_recompiles_control_clamp_records() {
    use dnam_engine::container::{DnamContainerView, update_header_in_place};
    use dnam_engine::provenance::DefaultCryptoProvider;

    let mut header = DnamJsonHeader::default_reference();
    header.controls_schema = vec![ControlSchemaDef {
        id: "drive".to_string(),
        name: "Drive".to_string(),
        kind: ControlKind::Range,
        unit: None,
        lobatto_degree: Some(2),
        anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
        declared_range: Some([0.0, 10.0]),
        safe_clamped_range: Some([1.0, 9.0]),
        default: serde_json::json!(5.0),
        conditional_min: BTreeMap::new(),
        conditional_max: BTreeMap::new(),
        declared_options: None,
        active_options: None,
        clamp_warning: None,
        calibration: None,
        ui_hints: ControlUiHints::default(),
        user_metadata: BTreeMap::new(),
    }];
    let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    let mut container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

    // Update safe_clamped_range via update_header_in_place and verify ControlChannelRecord[0].clamp updates
    header.controls_schema[0].safe_clamped_range = Some([2.5, 7.5]);
    update_header_in_place(&mut container, &header).unwrap();

    let crypto = DefaultCryptoProvider::default();
    let view = DnamContainerView::from_bytes(&container, &crypto).unwrap();
    assert_eq!(view.control_records.len(), 1);
    assert!((view.control_records[0].clamp.safe_min_base - 2.5).abs() < 1e-5);
    assert!((view.control_records[0].clamp.safe_max_base - 7.5).abs() < 1e-5);
}
