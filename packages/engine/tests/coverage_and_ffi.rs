//! Comprehensive test suite covering FFI exports (`ffi.rs`), memory management (`memory.rs`),
//! hardware SIMD backends (`math.rs`, `neon.rs`, `arm_sme.rs`, `scalar.rs`), schema validation
//! (`schema.rs`), container error paths (`container.rs`), provenance (`provenance.rs`), and
//! `DynamicDnamEngine` / `DnamEngine` lifecycle methods (`lib.rs`).

use std::collections::BTreeMap;

use dnam_engine::container::{
    ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_ULTRA_END, DnamContainerView,
    DnamPreamble, ZohRescaledParams, build_dnam_container, crc32c_compute, update_header_in_place,
    verify_ckpt_stem_crc32c,
};
use dnam_engine::control::{
    ConditionalClampRule, ControlBus, ControlClampDescriptor, eval_newton_lobatto_basis_f64,
};
use dnam_engine::core1::{Core1Stack, poly_adaa_slice_f64, poly_adaa_step_f64};
use dnam_engine::core2::ModulatedUfdn;
use dnam_engine::ffi::{
    dnam_engine_create_from_bytes, dnam_engine_destroy, dnam_engine_get_telemetry,
    dnam_engine_process_block, dnam_engine_process_mono, dnam_engine_reset,
    dnam_engine_set_control_immediate, dnam_engine_set_control_value, dnam_engine_set_sample_rate,
    dnam_engine_set_tier,
};
use dnam_engine::math::{
    BackendKind, DefaultMathBackend, MathBackend, ScalarBackend, SimdVec4, algebraic_unit_rotor,
    algebraic_unit_rotor_f64, clamp_pre_activation, cos_pi_unit_f64, crc32c,
    cubic_lagrange_weights, cubic_lagrange_weights_f64, detect_best_backend,
    eval_poly_adaa_algebraic, eval_poly_adaa_algebraic_f64, eval_poly_adaa_antiderivative_f64,
    eval_poly_adaa_waveshaper, eval_poly_adaa_waveshaper_f64, exp_f64, f32_to_fp16, floor_f32,
    floor_f64, fp16_to_f32, ln_f64, powf_f64, round_f32, round_f64, sqrt_f64,
    stabilize_complex_rotor,
};
use dnam_engine::memory::{
    AlignedBuffer, AlignedFp16Block, ArenaStorage, CacheLine64, DelayScratchStorage,
};
use dnam_engine::provenance::{
    CryptoProvider, DefaultCryptoProvider, OpenArenaCryptoProvider, RustCryptoProvider,
    VerificationStatus, apply_isometric_gauge_watermark, apply_tardos_watermark_4608,
    chacha20_xor_arena_blocks, compute_immutable_header_sha256, decode_hex_array, encode_hex,
    extract_structural_hash_160d, score_tardos_watermark_4608, sha256_digest,
    structural_cosine_similarity, verify_container_provenance,
};
use dnam_engine::schema::{
    AuthorizedDevice, ControlBusConfig, ControlKind, ControlSchemaDef, ControlUiHints, Core1Config,
    Core2Config, DatasetProvenance, DnamError, DnamJsonHeader, ExtractedPhysics, IoConfig,
    MarketplaceReceipt, ModelMetadata, PresetDef, SlimmableTiersConfig, StemProvenance,
    StemRegistryEntry, TensorSliceEntry, TierProfile, cold_exp_f64, cold_ln_f64, cold_pow_f64,
    cold_sqrt_f64,
};
use dnam_engine::{DnamEngine, DnamTelemetrySnapshot, DynamicDnamEngine, FtzDazGuard, Tier};

fn make_base_header(name: &str) -> DnamJsonHeader {
    DnamJsonHeader {
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
        core2_mufdn: Core2Config::default(),
        tensor_directory: BTreeMap::new(),
        stems_registry: None,
        audit_report: None,
    }
}

fn make_test_container(arena_len: usize, core2_active: bool) -> Vec<u8> {
    let mut header = make_base_header("CoverageTestModel");
    header.core2_mufdn.active = core2_active;
    header.core2_mufdn.ring_buffer_pow2_log2 = 10;
    header.control_bus.envelopes = 2;
    header.control_bus.lfo_oscillators = 2;

    let gain_ctrl = ControlSchemaDef {
        id: "Gain".to_string(),
        name: "Gain".to_string(),
        kind: ControlKind::Range,
        unit: Some("dB".to_string()),
        lobatto_degree: Some(2),
        anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
        declared_range: Some([0.0, 10.0]),
        safe_clamped_range: Some([0.0, 9.0]),
        default: serde_json::json!(5.0),
        conditional_min: BTreeMap::from([("Channel=Clean".to_string(), 1.0)]),
        conditional_max: BTreeMap::from([("Channel=Lead".to_string(), 8.0)]),
        declared_options: None,
        active_options: None,
        clamp_warning: None,
        ui_hints: ControlUiHints {
            midpoint: Some(3.0),
            ..Default::default()
        },
        user_metadata: BTreeMap::new(),
    };
    let sw_ctrl = ControlSchemaDef {
        id: "Channel".to_string(),
        name: "Channel".to_string(),
        kind: ControlKind::Switch,
        unit: None,
        lobatto_degree: None,
        anchor_nodes_norm: None,
        declared_range: None,
        safe_clamped_range: None,
        default: serde_json::json!("Clean"),
        conditional_min: BTreeMap::new(),
        conditional_max: BTreeMap::new(),
        declared_options: Some(vec!["Clean".to_string(), "Lead".to_string()]),
        active_options: Some(vec!["Clean".to_string(), "Lead".to_string()]),
        clamp_warning: None,
        ui_hints: ControlUiHints::default(),
        user_metadata: BTreeMap::new(),
    };
    header.controls_schema = vec![gain_ctrl, sw_ctrl];
    header.presets = vec![PresetDef {
        id: "preset_lead".to_string(),
        name: "Lead Preset".to_string(),
        description: Some("Lead channel preset".to_string()),
        controls: BTreeMap::from([
            ("Gain".to_string(), serde_json::json!(7.5)),
            ("Channel".to_string(), serde_json::json!("Lead")),
        ]),
        user_metadata: BTreeMap::new(),
    }];

    let mut arena = vec![0u8; arena_len];
    arena[0..4].copy_from_slice(&0.5_f32.to_le_bytes());
    arena[64..68].copy_from_slice(&0.01_f32.to_le_bytes());
    arena[104..108].copy_from_slice(&1.0_f32.to_le_bytes());
    arena[108..112].copy_from_slice(&1.0_f32.to_le_bytes());
    arena[112..116].copy_from_slice(&1.0_f32.to_le_bytes());
    arena[116..120].copy_from_slice(&0.2_f32.to_le_bytes());
    arena[120..124].copy_from_slice(&0.8_f32.to_le_bytes());
    arena[124..128].copy_from_slice(&0.9993_f32.to_le_bytes());

    build_dnam_container(&mut header, &arena, &arena, false).unwrap()
}

#[test]
fn test_ffi_exports_and_null_safety() {
    let container_bytes = make_test_container(ARENA_OFFSET_ULTRA_END as usize, true);

    // SAFETY: Testing C-ABI FFI surface with both valid buffers/handles and null pointers.
    unsafe {
        assert!(dnam_engine_create_from_bytes(core::ptr::null(), 100, 0, 48_000).is_null());
        assert!(dnam_engine_create_from_bytes(container_bytes.as_ptr(), 32, 0, 48_000).is_null());
        let bad_magic = [0u8; 128];
        assert!(
            dnam_engine_create_from_bytes(bad_magic.as_ptr(), bad_magic.len(), 0, 48_000).is_null()
        );

        // Null handle calls should be no-ops
        dnam_engine_set_tier(core::ptr::null_mut(), 1);
        assert_eq!(
            dnam_engine_set_sample_rate(core::ptr::null_mut(), 48_000),
            -1
        );
        dnam_engine_set_control_value(core::ptr::null_mut(), 0, 5.0);
        dnam_engine_set_control_immediate(core::ptr::null_mut(), 0, 5.0);
        dnam_engine_reset(core::ptr::null_mut());
        dnam_engine_process_block(
            core::ptr::null_mut(),
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            0,
        );
        dnam_engine_process_mono(
            core::ptr::null_mut(),
            core::ptr::null(),
            core::ptr::null_mut(),
            0,
        );
        dnam_engine_get_telemetry(core::ptr::null(), core::ptr::null_mut());
        dnam_engine_destroy(core::ptr::null_mut());

        // Valid handle lifecycle
        let handle = dnam_engine_create_from_bytes(
            container_bytes.as_ptr(),
            container_bytes.len(),
            1,
            48_000,
        );
        assert!(!handle.is_null());

        dnam_engine_set_tier(handle, 2);
        dnam_engine_set_tier(handle, 0);
        assert_eq!(dnam_engine_set_sample_rate(handle, 96_000), 0);
        assert_eq!(dnam_engine_set_sample_rate(handle, 12_000), -1);
        assert_eq!(dnam_engine_set_sample_rate(handle, 48_000), 0);

        dnam_engine_set_control_value(handle, 0, 6.0);
        dnam_engine_set_control_immediate(handle, 0, 7.0);
        dnam_engine_reset(handle);

        let left_in = [0.25_f32; 32];
        let right_in = [-0.25_f32; 32];
        let mut left_out = [0.0_f32; 32];
        let mut right_out = [0.0_f32; 32];

        dnam_engine_process_block(
            handle,
            left_in.as_ptr(),
            right_in.as_ptr(),
            left_out.as_mut_ptr(),
            right_out.as_mut_ptr(),
            0,
        );
        dnam_engine_process_block(
            handle,
            left_in.as_ptr(),
            right_in.as_ptr(),
            left_out.as_mut_ptr(),
            right_out.as_mut_ptr(),
            32,
        );

        let mut mono_out = [0.0_f32; 32];
        dnam_engine_process_mono(handle, left_in.as_ptr(), mono_out.as_mut_ptr(), 0);
        dnam_engine_process_mono(handle, left_in.as_ptr(), mono_out.as_mut_ptr(), 32);

        let mut snap = core::mem::MaybeUninit::<DnamTelemetrySnapshot>::zeroed().assume_init();
        dnam_engine_get_telemetry(handle, &mut snap);
        assert_eq!(snap.active_tier, 0);

        dnam_engine_destroy(handle);
    }
}

#[test]
fn test_memory_abstractions_complete() {
    let mut cl = CacheLine64::default();
    assert_eq!(cl.as_f32_slice()[0], 0.0);
    cl.as_f32_slice_mut()[0] = 3.5;
    assert_eq!(cl.as_f32_slice()[0], 3.5);

    let fp_blk = AlignedFp16Block::default();
    assert_eq!(fp_blk.0[0], 0);

    let empty_buf = AlignedBuffer::<CacheLine64>::new_zeroed(0);
    assert!(empty_buf.is_empty());
    assert_eq!(empty_buf.len_blocks(), 0);

    let mut buf = AlignedBuffer::<CacheLine64>::new_zeroed(2);
    assert!(!buf.is_empty());
    assert_eq!(buf.len_blocks(), 2);
    buf.as_bytes_mut()[0] = 42;
    assert_eq!(buf.as_bytes()[0], 42);
    assert_eq!(buf[0].0[0], 42);
    buf[1].0[0] = 99;
    assert_eq!(buf[1].0[0], 99);

    let aligned_backing = AlignedBuffer::<CacheLine64>::new_zeroed(2);
    let borrowed = ArenaStorage::from_slice_or_copy(aligned_backing.as_bytes());
    assert!(borrowed.is_borrowed());
    assert_eq!(borrowed.as_f32_slice().len(), 32);

    let unaligned_arr = [0u8; 136];
    let unaligned_slice = &unaligned_arr[1..129];
    let mut owned = ArenaStorage::from_slice_or_copy(unaligned_slice);
    assert!(!owned.is_borrowed());
    assert!(!owned.reload_in_place(&unaligned_arr[0..63]));
    assert!(!owned.reload_in_place(&[0u8; 256]));
    assert!(owned.reload_in_place(&[7u8; 64]));
    assert_eq!(owned.as_bytes().len(), 64);

    let mut max_arena = ArenaStorage::with_max_blocks(4);
    assert_eq!(max_arena.as_bytes().len(), 0);
    assert!(max_arena.reload_in_place(&[3u8; 128]));
    assert_eq!(max_arena.as_bytes().len(), 128);

    let mut borrowed_mut = ArenaStorage::from_slice_or_copy(aligned_backing.as_bytes());
    assert!(!borrowed_mut.reload_in_place(&[0u8; 64]));

    let empty_scratch = DelayScratchStorage::empty();
    assert_eq!(empty_scratch.byte_len(), 0);
    assert_eq!(empty_scratch.capacity_u16(), 0);

    let mut owned_scratch = DelayScratchStorage::new_owned(2);
    assert_eq!(owned_scratch.byte_len(), 128);
    assert_eq!(owned_scratch.capacity_u16(), 64);
    assert_eq!(owned_scratch.as_slice().len(), 2);
    assert_eq!(owned_scratch.as_slice_mut().len(), 2);
    owned_scratch.as_u16_slice_mut()[0] = 0x3C00;
    assert_eq!(owned_scratch.as_u16_slice()[0], 0x3C00);
    owned_scratch.clear();
    assert_eq!(owned_scratch.as_u16_slice()[0], 0);

    let mut backing_fp16 = [AlignedFp16Block::default(); 2];
    let mut borrowed_scratch = DelayScratchStorage::from_borrowed_scratch(&mut backing_fp16);
    assert_eq!(borrowed_scratch.capacity_u16(), 64);
    borrowed_scratch.as_slice_mut()[0].0[0] = 123;
    assert_eq!(borrowed_scratch.as_slice()[0].0[0], 123);
}

fn exercise_math_backend<B: MathBackend>() {
    let v0 = B::Vec4::zero();
    let v1 = B::Vec4::from_array([1.0, -2.0, 3.0, -4.0]);
    let v2 = B::Vec4::splat(2.0);
    assert_eq!(v0.to_array(), [0.0; 4]);
    assert!((v1.dot(v2) - (-4.0)).abs() < 1e-5);
    let clamped = v1.clamp(B::Vec4::splat(-1.0), B::Vec4::splat(1.0)).abs();
    assert!((clamped.horizontal_sum() - 4.0).abs() < 1e-5);

    let x = [0.5, -0.5, 0.25, -0.25];
    let alpha = [0.8; 4];
    let beta = [0.3; 4];
    let delta = [0.05; 4];
    let mut u_prev = [0.0; 4];
    let mut a_prev = [0.0; 4];
    let mut m_dc = [0.0; 4];
    let mut y_out = [0.0; 4];
    B::poly_adaa_layer_4ch(
        &x,
        &alpha,
        &beta,
        &delta,
        0.999,
        &mut u_prev,
        &mut a_prev,
        &mut m_dc,
        &mut y_out,
    );
    assert!(y_out[0].is_finite());

    let taps = [[0.1_f32; 4]; 6];
    let w4 = [[[0.05_f32; 4]; 4]; 6];
    let w8 = [[[0.05_f32; 4]; 8]; 6];
    let w12 = [[[0.05_f32; 4]; 12]; 6];
    let bias = [0.01_f32; 4];
    let mut out = [0.0_f32; 4];
    B::conv6_tile_4x4(&taps, &w4, &bias, &mut out);
    assert!(out[0].is_finite());
    B::conv6_tile_4x8(&taps, &taps, &w8, &bias, &mut out);
    assert!(out[0].is_finite());
    B::conv6_tile_4x12(&taps, &taps, &taps, &w12, &bias, &mut out);
    assert!(out[0].is_finite());

    let h_group = [[0.1_f32; 4]; 3];
    let h_taps = [&h_group; 6];
    let w_tiles6 = [[[0.05_f32; 4]; 4]; 6];
    let w_tiles12 = [[[0.05_f32; 4]; 4]; 12];
    let w_tiles18 = [[[0.05_f32; 4]; 4]; 18];
    let c1 = B::conv6_taps_1group(B::Vec4::load(&bias), &w_tiles6, &h_taps);
    let c2 = B::conv6_taps_2groups(B::Vec4::load(&bias), &w_tiles12, &h_taps);
    let c3 = B::conv6_taps_3groups(B::Vec4::load(&bias), &w_tiles18, &h_taps);
    assert!(c1.horizontal_sum().is_finite());
    assert!(c2.horizontal_sum().is_finite());
    assert!(c3.horizontal_sum().is_finite());

    let theta = [0.2, 0.5, 1.0, 2.0];
    let mut cos_out = [0.0; 4];
    let mut sin_out = [0.0; 4];
    B::algebraic_rotor_4ch(&theta, &mut cos_out, &mut sin_out);
    assert!((cos_out[0] * cos_out[0] + sin_out[0] * sin_out[0] - 1.0).abs() < 1e-4);

    let mut z_re = [0.0; 4];
    let mut z_im = [0.0; 4];
    B::rglru_step_4ch(
        &x, &[0.8; 4], &[0.1; 4], &[0.2; 4], &[0.5; 4], &[0.4; 4], &[0.1; 4], &cos_out, &sin_out,
        &mut z_re, &mut z_im, &mut y_out,
    );
    assert!(y_out[0].is_finite());

    B::selective_rglru_layer_4ch(
        &x, &[0.8; 4], &[0.1; 4], &[0.5; 4], &[0.2; 4], &[0.4; 4], &[0.1; 4], &cos_out, &sin_out,
        &mut z_re, &mut z_im, &mut y_out,
    );
    assert!(y_out[0].is_finite());

    let base_head = [0.1_f32; 16];
    let svd_modes = [0.02_f32; 64];
    let z_head = [0.5, -0.25, 0.1, 0.2];
    let mut collapsed4 = [0.0_f32; 16];
    let mut collapsed2 = [0.0_f32; 16];
    B::collapse_svd_head_slice(&base_head, &svd_modes, &z_head, 4, &mut collapsed4);
    B::collapse_svd_head_slice(&base_head, &svd_modes, &z_head, 2, &mut collapsed2);
    assert!(collapsed4[0].is_finite());
    assert!(collapsed2[0].is_finite());

    let taps16 = [[0.1_f32; 4]; 16];
    let wl16 = [[0.2_f32; 4]; 16];
    let wr16 = [[-0.2_f32; 4]; 16];
    let (sl, sr) = B::stereo_head_dot_16x4(&taps16, &wl16, &wr16);
    let ml = B::mono_head_dot_16x4(&taps16, &wl16);
    assert!((sl - ml).abs() < 1e-5);
    assert!((sr + ml).abs() < 1e-5);

    let taps64 = [0.1_f32; 64];
    let wl64 = [0.2_f32; 64];
    let wr64 = [-0.2_f32; 64];
    let d1 = B::dot_product_16taps_4ch(&taps64, &wl64);
    let [dl, dr] = B::dot_product_16taps_4ch_stereo(&taps64, &wl64, &wr64);
    assert!((d1 - dl).abs() < 1e-5);
    assert!((d1 + dr).abs() < 1e-5);

    let mu = [0.25, 0.5, 0.75, 0.1];
    B::cubic_lagrange_4ch(&mu, &[1.0; 4], &[1.0; 4], &[1.0; 4], &[1.0; 4], &mut out);
    assert!((out[0] - 1.0).abs() < 1e-5);

    let mut q4 = [0.0_f32; 4];
    B::givens_householder_mix_4ch(&[1.0, 0.0, 0.5, -0.5], 0.8, 0.6, &[0.5; 4], &mut q4);
    assert!(q4[0].is_finite());

    let mut q8 = [0.0_f32; 8];
    let v8 = [0.353_553_38_f32; 8];
    B::givens_householder_mix_8ch(&[0.5_f32; 8], 0.8, 0.6, &v8, &mut q8);
    assert!(q8[0].is_finite());

    let mut s_out = [0.0_f32; 4];
    B::lyapunov_loop_4ch(
        &[0.2; 4], &[0.1; 4], &[0.9; 4], &[0.05; 4], &[2.0; 4], &[0.7; 4], &[0.95; 4], &mut s_out,
    );
    assert!(s_out[0].is_finite());

    let mut dc_x = [0.0_f32; 4];
    let mut dc_y = [0.0_f32; 4];
    B::mufdn_coercive_loop_4ch(
        &[0.3; 4], &[0.9; 4], &[2.0; 4], &[0.7; 4], &[0.95; 4], 0.995, &mut dc_x, &mut dc_y,
        &mut s_out,
    );
    assert!(s_out[0].is_finite());

    let fp16_4 = B::f32x4_to_fp16x4(&[0.5, -0.5, 1.0, 0.0]);
    let f32_4 = B::fp16x4_to_f32x4(&fp16_4);
    assert!((f32_4[0] - 0.5).abs() < 1e-3);

    assert_eq!(B::crc32c(b"123456789"), 0xE306_9283);
    let (cp, sp) = B::fast_unit_phasor(0.7);
    assert!((cp * cp + sp * sp - 1.0).abs() < 1e-4);
}

#[test]
fn test_all_math_backends_and_reference_functions() {
    for (kind, expected) in [
        (BackendKind::Scalar, "Scalar"),
        (BackendKind::ArmNeon, "ARM64 NEON"),
        (BackendKind::Neon, "ARM64 NEON"),
        (BackendKind::ArmSme, "ARM SME"),
        (BackendKind::X86Sse42, "x86 SSE4.2"),
        (BackendKind::X86Avx2Fma, "x86 AVX2"),
        (BackendKind::X86Simd, "x86 AVX2"),
        (BackendKind::WasmSimd128, "WASM SIMD"),
    ] {
        assert_eq!(kind.name(), expected);
    }
    let _ = detect_best_backend();

    exercise_math_backend::<ScalarBackend>();
    exercise_math_backend::<DefaultMathBackend>();
    #[cfg(target_arch = "aarch64")]
    {
        exercise_math_backend::<dnam_engine::math::ArmSmeBackend>();
    }

    assert_eq!(clamp_pre_activation(3.0), 1.0);
    assert_eq!(clamp_pre_activation(-3.0), -1.0);
    let ws32 = eval_poly_adaa_waveshaper(0.5, 0.4, 0.1);
    let ws64 = eval_poly_adaa_waveshaper_f64(0.5, 0.4, 0.1);
    assert!((ws32 as f64 - ws64).abs() < 1e-6);
    let anti = eval_poly_adaa_antiderivative_f64(0.5, 0.4, 0.1);
    assert!(anti.is_finite());
    let ad32 = eval_poly_adaa_algebraic(0.2, 0.4, 0.5, 0.1);
    let ad64 = eval_poly_adaa_algebraic_f64(0.2, 0.4, 0.5, 0.1);
    assert!((ad32 as f64 - ad64).abs() < 1e-6);

    let (r32_c, r32_s) = algebraic_unit_rotor(1.0);
    let (r64_c, r64_s) = algebraic_unit_rotor_f64(1.0);
    assert!((r32_c as f64 - r64_c).abs() < 1e-6);
    assert!((r32_s as f64 - r64_s).abs() < 1e-6);
    let (sc, ss) = stabilize_complex_rotor(0.99, 0.0);
    assert!((sc - 1.0).abs() < 1e-3 && ss.abs() < 1e-6);

    let cw32 = cubic_lagrange_weights(0.3);
    let cw64 = cubic_lagrange_weights_f64(0.3);
    assert!((cw32[0] as f64 - cw64[0]).abs() < 1e-6);

    let inf_h = f32_to_fp16(f32::INFINITY);
    assert!(fp16_to_f32(inf_h).is_infinite());
    let nan_h = f32_to_fp16(f32::NAN);
    assert!(fp16_to_f32(nan_h).is_nan());
    let ovf_h = f32_to_fp16(100_000.0);
    assert_eq!(ovf_h, 0x7BFF);
    let und_h = f32_to_fp16(1.0e-10);
    assert_eq!(fp16_to_f32(und_h), 0.0);
    let sub_h = f32_to_fp16(3.0e-5);
    assert!(fp16_to_f32(sub_h) > 0.0);

    assert_eq!(floor_f32(2.7), 2.0);
    assert_eq!(floor_f32(-2.7), -3.0);
    assert_eq!(floor_f64(2.7), 2.0);
    assert_eq!(floor_f64(-2.7), -3.0);
    assert_eq!(round_f32(2.6), 3.0);
    assert_eq!(round_f32(-2.6), -3.0);
    assert_eq!(round_f64(2.6), 3.0);
    assert_eq!(round_f64(-2.6), -3.0);
    assert_eq!(sqrt_f64(0.0), 0.0);
    assert_eq!(sqrt_f64(-5.0), 0.0);
    assert_eq!(ln_f64(0.0), f64::NEG_INFINITY);
    assert_eq!(exp_f64(-800.0), 0.0);
    assert_eq!(exp_f64(800.0), f64::INFINITY);
    assert_eq!(powf_f64(0.0, 2.0), 0.0);
    assert_eq!(powf_f64(3.5, 1.0), 3.5);
    assert!((cos_pi_unit_f64(0.5)).abs() < 1e-5);

    assert_eq!(cold_ln_f64(0.0), f64::NEG_INFINITY);
    assert_eq!(cold_exp_f64(800.0), f64::INFINITY);
    assert_eq!(cold_exp_f64(-800.0), 0.0);
    assert_eq!(cold_pow_f64(0.0, 2.0), 0.0);
    assert_eq!(cold_pow_f64(4.0, 1.0), 4.0);
    assert_eq!(cold_pow_f64(4.0, 0.0), 1.0);
    assert_eq!(cold_sqrt_f64(0.0), 0.0);
    assert_eq!(cold_sqrt_f64(-1.0), 0.0);

    let mut u_p = 0.0;
    let mut a_p = 0.0;
    let mut m_d = 0.0;
    let s1 = poly_adaa_step_f64(0.5, 0.8, 0.2, 0.05, 0.999, &mut u_p, &mut a_p, &mut m_d);
    assert!(s1.is_finite());
    let inp = [0.0, 0.25, -0.25, 0.1];
    let mut outp = [0.0; 4];
    poly_adaa_slice_f64(&inp, &mut outp, 0.0, 0.8, 0.2, 0.05, 0.999);
    assert!(outp[1].is_finite());
}

#[test]
fn test_schema_container_and_error_formatting() {
    let errors = [
        DnamError::BufferTooSmall {
            expected: 64,
            actual: 12,
        },
        DnamError::InvalidMagic(*b"ABCD"),
        DnamError::UnsupportedVersion { major: 9, minor: 0 },
        DnamError::UnalignedOffset {
            field: "arena",
            offset: 65,
        },
        DnamError::Crc32cMismatch {
            region: "header",
            expected: 1,
            computed: 2,
        },
        DnamError::InvalidJsonHeader("bad json".to_string()),
        DnamError::SchemaValidationError("bad schema".to_string()),
        DnamError::HeaderCapacityExceeded {
            required: 5000,
            capacity: 4000,
        },
        DnamError::CryptoVerificationFailed("bad sig"),
        DnamError::EncryptedArenaRequiresDeviceKey,
        DnamError::InvalidSampleRate(12_000),
        DnamError::InvalidControlIndex(99),
    ];
    for err in &errors {
        assert!(!err.to_string().is_empty());
    }

    // ControlUiHints skew & mapping edge cases
    let hints = ControlUiHints {
        midpoint: Some(2.5),
        ..Default::default()
    };
    assert_eq!(hints.skew_exponent(5.0, 5.0), 1.0);
    assert_eq!(hints.normalized_to_physical(0.0, 0.0, 10.0), 0.0);
    assert_eq!(hints.normalized_to_physical(1.0, 0.0, 10.0), 10.0);
    let mid_phys = hints.normalized_to_physical(0.5, 0.0, 10.0);
    assert!((mid_phys - 2.5).abs() < 1e-3);
    assert_eq!(hints.physical_to_normalized(0.0, 5.0, 5.0), 0.0);
    assert_eq!(hints.physical_to_normalized(0.0, 0.0, 10.0), 0.0);
    assert_eq!(hints.physical_to_normalized(10.0, 0.0, 10.0), 1.0);
    let mid_norm = hints.physical_to_normalized(2.5, 0.0, 10.0);
    assert!((mid_norm - 0.5).abs() < 1e-3);

    // Schema validation error branches
    let mut h = make_base_header("Test");
    h.container_kind = "invalid".to_string();
    assert!(h.validate().is_err());

    h.container_kind = "dnam".to_string();
    h.sample_rate_train_hz = 10_000;
    assert!(h.validate().is_err());
    h.sample_rate_train_hz = 48_000;

    let mut bad_slice = TensorSliceEntry {
        arena: 0,
        dtype: "f32".to_string(),
        packing: None,
        feature_dim: None,
        packed_len: None,
        shape: None,
        byte_offset: 3,
        byte_length: 64,
    };
    h.tensor_directory
        .insert("bad_off".to_string(), bad_slice.clone());
    assert!(h.validate().is_err());
    bad_slice.byte_offset = 64;
    bad_slice.byte_length = 63;
    h.tensor_directory.insert("bad_off".to_string(), bad_slice);
    assert!(h.validate().is_err());
    h.tensor_directory.clear();

    // Missing declared_range on range control
    h.controls_schema = vec![ControlSchemaDef {
        id: "Gain".to_string(),
        name: "Gain".to_string(),
        kind: ControlKind::Range,
        unit: None,
        lobatto_degree: None,
        anchor_nodes_norm: None,
        declared_range: None,
        safe_clamped_range: None,
        default: serde_json::json!(5.0),
        conditional_min: BTreeMap::new(),
        conditional_max: BTreeMap::new(),
        declared_options: None,
        active_options: None,
        clamp_warning: None,
        ui_hints: ControlUiHints::default(),
        user_metadata: BTreeMap::new(),
    }];
    assert!(h.validate().is_err());

    // Zero-span declared_range
    h.controls_schema[0].declared_range = Some([5.0, 5.0]);
    assert!(h.validate().is_err());

    // Non-numeric default on range control
    h.controls_schema[0].declared_range = Some([0.0, 10.0]);
    h.controls_schema[0].default = serde_json::json!("bad");
    assert!(h.validate().is_err());

    // Switch missing declared_options
    h.controls_schema[0].kind = ControlKind::Switch;
    assert!(h.validate().is_err());

    // Switch empty declared_options
    h.controls_schema[0].declared_options = Some(vec![]);
    assert!(h.validate().is_err());

    // > 16 controls
    h.controls_schema = vec![h.controls_schema[0].clone(); 17];
    assert!(h.validate().is_err());
    assert!(h.compile_clamp_descriptors().is_err());

    // Preamble & Container error branches
    assert!(DnamPreamble::from_bytes(&[0u8; 10]).is_err());
    let mut bad_preamble = [0u8; 64];
    bad_preamble[0..4].copy_from_slice(b"NOPE");
    assert!(DnamPreamble::from_bytes(&bad_preamble).is_err());
    bad_preamble[0..4].copy_from_slice(b"DNAM");
    bad_preamble[4..6].copy_from_slice(&99_u16.to_le_bytes());
    assert!(DnamPreamble::from_bytes(&bad_preamble).is_err());
    bad_preamble[4..6].copy_from_slice(&1_u16.to_le_bytes());
    bad_preamble[20..24].copy_from_slice(&65_u32.to_le_bytes());
    assert!(DnamPreamble::from_bytes(&bad_preamble).is_err());

    let lite_bytes = make_test_container(ARENA_OFFSET_LITE_END as usize, false);
    let full_bytes = make_test_container(ARENA_OFFSET_FULL_END as usize, true);
    let mut ultra_bytes = make_test_container(ARENA_OFFSET_ULTRA_END as usize, true);

    let crypto = DefaultCryptoProvider::default();
    assert!(DnamContainerView::from_bytes(&lite_bytes, &crypto).is_ok());
    assert!(DnamContainerView::from_bytes(&full_bytes, &crypto).is_ok());

    assert!(DnamContainerView::from_bytes(&ultra_bytes[..100], &crypto).is_err());

    ultra_bytes[60] ^= 0xFF;
    assert!(DnamContainerView::from_bytes(&ultra_bytes, &crypto).is_err());
    ultra_bytes[60] ^= 0xFF;

    let mut preamble = DnamPreamble::from_bytes(&ultra_bytes).unwrap();
    let arena_off = preamble.tensor_arena_offset as usize;
    let raw_arena = &ultra_bytes[arena_off..];
    preamble.arena_crc32c_lite ^= 1;
    assert!(DnamContainerView::verify_arena_prefix_crcs(&preamble, raw_arena).is_err());
    preamble.arena_crc32c_lite ^= 1;
    preamble.arena_crc32c_full ^= 1;
    assert!(DnamContainerView::verify_arena_prefix_crcs(&preamble, raw_arena).is_err());
    preamble.arena_crc32c_full ^= 1;
    preamble.arena_crc32c_ultra ^= 1;
    assert!(DnamContainerView::verify_arena_prefix_crcs(&preamble, raw_arena).is_err());

    let mut hdr_ok = make_base_header("Ok");
    assert!(build_dnam_container(&mut hdr_ok, &[0u8; 128], &[0u8; 128], false).is_err());

    // dnam-ckpt container with stems_registry
    hdr_ok.container_kind = "dnam-ckpt".to_string();
    let dummy_slice = TensorSliceEntry {
        arena: 3,
        dtype: "f64".to_string(),
        packing: Some("uplo_upper_col_major_tier_ordered".to_string()),
        feature_dim: Some(64),
        packed_len: Some(2080),
        shape: None,
        byte_offset: 0,
        byte_length: 64,
    };
    hdr_ok.stems_registry = Some(vec![StemRegistryEntry {
        stem_id: "stem0".to_string(),
        controls_meta: BTreeMap::new(),
        provenance: StemProvenance {
            input_pcm_sha256: "00".repeat(32),
            target_pcm_sha256: "11".repeat(32),
            num_frames: 48_000,
        },
        stem_arena3_crc32c: crc32c(&[0u8; 64]),
        user_metadata: BTreeMap::new(),
        extracted_physics: Some(ExtractedPhysics {
            delays_ms: vec![10.0],
            lfo_freqs_hz: vec![0.5],
            decay_gains_lin: vec![0.7],
        }),
        g_audio_uplo_u_f64: dummy_slice.clone(),
        c_audio_tier_f64: dummy_slice,
    }]);
    let ckpt_bytes = build_dnam_container(
        &mut hdr_ok,
        &vec![0u8; ARENA_OFFSET_LITE_END as usize],
        &vec![0u8; ARENA_OFFSET_LITE_END as usize],
        false,
    )
    .unwrap();
    let ckpt_view = DnamContainerView::from_bytes(&ckpt_bytes, &crypto).unwrap();
    assert_eq!(ckpt_view.preamble.num_ckpt_stems, 1);

    let stem_buf = [5u8; 128];
    let stem_crc = crc32c_compute(&stem_buf[0..64]);
    assert!(verify_ckpt_stem_crc32c(&stem_buf, 0, 64, stem_crc).is_ok());
    assert!(verify_ckpt_stem_crc32c(&stem_buf, 0, 256, stem_crc).is_err());
    assert!(verify_ckpt_stem_crc32c(&stem_buf, 0, 64, stem_crc ^ 1).is_err());

    assert!(update_header_in_place(&mut ultra_bytes[..64], &hdr_ok).is_err());
    let mut huge_hdr = make_base_header("Huge");
    for i in 0..200 {
        huge_hdr
            .user_metadata
            .insert(format!("key_{i}"), serde_json::json!("x".repeat(50)));
    }
    assert!(update_header_in_place(&mut ultra_bytes, &huge_hdr).is_err());

    assert!(ZohRescaledParams::compute(48_000, 12_000, 16, 0.9993, 0.995, 64.0, 128).is_err());
    let zoh48 = ZohRescaledParams::compute(48_000, 48_000, 16, 0.9993, 0.995, 64.0, 128).unwrap();
    let (r_l, dr, th, bg) = zoh48.rescale_rglru_channel(0.9, 0.05, 0.5, 0.2);
    assert_eq!((r_l, dr, th, bg), (0.9, 0.05, 0.5, 0.2));
    let (t0, dap) = zoh48.rescale_core2_delays(100.0, 20.0);
    assert_eq!((t0, dap), (100.0, 20));
}

#[inline(never)]
fn exercise_scalar_engines(lite_bytes: &[u8], full_bytes: &[u8], ultra_bytes: &[u8]) {
    let mono_in = [0.2_f32; 32];
    let mut mono_out = [0.0_f32; 32];

    {
        let mut eng_lite = Box::new(
            DnamEngine::<ScalarBackend>::from_bytes(lite_bytes, Tier::Ultra, 48_000).unwrap(),
        );
        assert_eq!(eng_lite.tier(), Tier::Lite);
        assert_eq!(eng_lite.sample_rate_hz(), 48_000);
        assert_eq!(
            *eng_lite.verification_status(),
            VerificationStatus::Unsigned
        );
        assert!(eng_lite.header().is_some());
        assert_eq!(eng_lite.preamble().num_controls, 2);
        eng_lite.process_mono(&mono_in, &mut mono_out);
    }

    {
        let mut eng_full = Box::new(
            DnamEngine::<ScalarBackend>::from_bytes_owned(full_bytes, Tier::Full, 48_000).unwrap(),
        );
        assert_eq!(eng_full.tier(), Tier::Full);
        assert!(eng_full.apply_preset("preset_lead").is_ok());
        assert!(eng_full.apply_preset("nonexistent_preset").is_err());
        eng_full.process_mono(&mono_in, &mut mono_out);
    }

    {
        let mut eng_ultra_scalar = Box::new(
            DnamEngine::<ScalarBackend>::from_bytes(ultra_bytes, Tier::Ultra, 48_000).unwrap(),
        );
        eng_ultra_scalar.process_mono(&mono_in, &mut mono_out);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn exercise_sme_engine(ultra_bytes: &[u8]) {
    let mono_in = [0.2_f32; 32];
    let mut mono_out = [0.0_f32; 32];
    let mut eng_sme = Box::new(
        DnamEngine::<dnam_engine::math::ArmSmeBackend>::from_bytes(
            ultra_bytes,
            Tier::Ultra,
            48_000,
        )
        .unwrap(),
    );
    eng_sme.set_tier(Tier::Lite);
    eng_sme.process_mono(&mono_in, &mut mono_out);
    eng_sme.set_tier(Tier::Full);
    eng_sme.process_mono(&mono_in, &mut mono_out);
    eng_sme.set_tier(Tier::Ultra);
    eng_sme.process_mono(&mono_in, &mut mono_out);
}

#[inline(never)]
fn exercise_dynamic_engines(ultra_bytes: &[u8]) {
    for kind in [BackendKind::Scalar, BackendKind::Neon, BackendKind::ArmSme] {
        let mut dyn_eng = Box::new(
            DynamicDnamEngine::from_bytes_with_backend(ultra_bytes, Tier::Full, 48_000, kind)
                .unwrap(),
        );
        let _ = dyn_eng.backend_kind();
        dyn_eng.set_tier(Tier::Lite);
        dyn_eng.set_tier(Tier::Ultra);
        assert!(dyn_eng.set_sample_rate(48_000).is_ok());
        dyn_eng.set_control_value(0, 4.0);
        dyn_eng.set_control_immediate(0, 6.0);
        dyn_eng.reset_to_dc_equilibrium();

        let l_in = [0.1_f32; 16];
        let r_in = [-0.1_f32; 16];
        let mut l_out = [0.0_f32; 16];
        let mut r_out = [0.0_f32; 16];
        dyn_eng.process_block(&l_in, &r_in, &mut l_out, &mut r_out);
        dyn_eng.process_mono(&l_in, &mut l_out);
        let snap = dyn_eng.telemetry_snapshot();
        assert_eq!(snap.active_tier, 2);
    }

    let dyn_auto =
        Box::new(DynamicDnamEngine::from_bytes(ultra_bytes, Tier::Full, 48_000).unwrap());
    assert_eq!(dyn_auto.telemetry_snapshot().active_tier, 1);
}

#[test]
fn test_engine_and_dynamic_engine_all_variants() {
    let _ftz = FtzDazGuard::default();
    assert_eq!(Tier::from_u8(0).channels(), 4);
    assert_eq!(Tier::from_u8(1).channels(), 8);
    assert_eq!(Tier::from_u8(2).channels(), 12);
    assert_eq!(Tier::from_u8(99), Tier::Ultra);

    let _c1_default = Box::new(Core1Stack::<ScalarBackend>::default());
    let _c2_default = Box::new(ModulatedUfdn::<ScalarBackend>::new_inactive());
    let _cb_default = Box::new(ControlBus::<ScalarBackend>::default());

    let full_bytes = make_test_container(ARENA_OFFSET_FULL_END as usize, true);
    let lite_bytes = make_test_container(ARENA_OFFSET_LITE_END as usize, false);
    let ultra_bytes = make_test_container(ARENA_OFFSET_ULTRA_END as usize, true);

    exercise_scalar_engines(&lite_bytes, &full_bytes, &ultra_bytes);

    #[cfg(target_arch = "aarch64")]
    {
        exercise_sme_engine(&ultra_bytes);
    }

    {
        let mut prealloc = Box::new(DnamEngine::<DefaultMathBackend>::with_max_capacity(10));
        assert!(prealloc.header().is_none());
        assert!(prealloc.apply_preset("preset_lead").is_err());
        assert!(
            prealloc
                .reload_model_in_place(&ultra_bytes[..100], Tier::Lite, 48_000)
                .is_err()
        );
        let mut bad_crc_bytes = ultra_bytes.clone();
        bad_crc_bytes[60] ^= 0xFF;
        assert!(
            prealloc
                .reload_model_in_place(&bad_crc_bytes, Tier::Lite, 48_000)
                .is_err()
        );
    }

    exercise_dynamic_engines(&ultra_bytes);

    // Provenance hex & OpenArenaCryptoProvider / RustCryptoProvider
    let hex_str = encode_hex(&[0xDE, 0xAD, 0xBE, 0xEF]);
    assert_eq!(hex_str, "deadbeef");
    assert_eq!(
        decode_hex_array::<4>("DEADBEEF").unwrap(),
        [0xDE, 0xAD, 0xBE, 0xEF]
    );
    assert!(decode_hex_array::<4>("bad").is_err());
    assert!(decode_hex_array::<4>("ZZZZZZZZ").is_err());

    let open_prov = OpenArenaCryptoProvider;
    assert!(open_prov.local_device_id_hash().is_none());
    assert!(open_prov.marketplace_pubkey().is_none());
    assert!(
        open_prov
            .ed25519_verify(&[0u8; 32], b"msg", &[0u8; 64])
            .is_err()
    );
    let dummy_receipt = MarketplaceReceipt {
        marketplace_id: "m".to_string(),
        license_id: "l".to_string(),
        buyer_hash: "00".repeat(32),
        license_tier: "personal".to_string(),
        issued_at: "2026-10-23T00:00:00Z".to_string(),
        buyer_arena_sha256: "00".repeat(32),
        encryption_mode: "none".to_string(),
        arena_nonce_96: None,
        ephemeral_pubkey_x25519: None,
        authorized_devices: Vec::new(),
        royalty_splits: Vec::new(),
        receipt_sig_ed25519: "00".repeat(64),
    };
    let dummy_dev = AuthorizedDevice {
        oem_vendor: "darkglass".to_string(),
        model_code: "anagram_rk3582".to_string(),
        device_id_hash: "00".repeat(16),
        wrapped_cek: Some("00".repeat(48)),
    };
    assert!(
        open_prov
            .unwrap_device_cek(&dummy_receipt, &dummy_dev)
            .is_err()
    );

    let rcp = RustCryptoProvider::new()
        .with_device_key("00".repeat(16), [1u8; 32])
        .with_marketplace_pubkey([2u8; 32]);
    assert!(!format!("{rcp:?}").is_empty());

    assert!(
        DnamEngine::<DefaultMathBackend>::from_bytes_with_crypto(
            &lite_bytes,
            Tier::Lite,
            48_000,
            &open_prov,
        )
        .is_ok()
    );

    // ConditionalClampRule min_floor and max_ceil
    let mut cb: ControlBus<ScalarBackend> = ControlBus::new();
    cb.num_controls = 2;
    cb.descriptors[0] = ControlClampDescriptor {
        ctrl_kind: 0,
        lobatto_degree: 3,
        active_options_mask: 1,
        default_val: 5.0,
        safe_min_base: 0.0,
        safe_max_base: 10.0,
        scale_norm: 0.2,
        offset_norm: -1.0,
        anchor_nodes_norm: [0.0, -1.0, 1.0, 0.5],
        cond_rules: [
            ConditionalClampRule {
                switch_ctrl_idx: 1,
                switch_opt_idx: 0,
                rule_op: 1,
                _pad: 0,
                bound_val: 7.0,
            },
            ConditionalClampRule {
                switch_ctrl_idx: 1,
                switch_opt_idx: 0,
                rule_op: 2,
                _pad: 0,
                bound_val: 2.0,
            },
            ConditionalClampRule::empty(),
        ],
    };
    cb.descriptors[1] = ControlClampDescriptor {
        ctrl_kind: 1,
        lobatto_degree: 1,
        active_options_mask: 0b0010,
        default_val: 0.0,
        safe_min_base: 0.0,
        safe_max_base: 1.0,
        scale_norm: 1.0,
        offset_norm: 0.0,
        anchor_nodes_norm: [0.0; 4],
        cond_rules: [ConditionalClampRule::empty(); 3],
    };
    cb.set_control_immediate(1, 0.0);
    assert_eq!(cb.clamped_values[1], 1.0);
}

#[test]
fn test_remaining_edge_cases_and_branches() {
    // 1. Minimal JSON deserialization into DnamJsonHeader to exercise all #[serde(default)] functions
    let minimal_json = r#"{
        "metadata": {
            "name": "Minimal",
            "created_at": "2026-10-23T00:00:00Z",
            "signatures": {
                "immutable_header_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                "master_arena_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                "creator_signature": {
                    "signer_id": "did:web:test",
                    "pubkey_ed25519": "0000000000000000000000000000000000000000000000000000000000000000",
                    "sig_ed25519": "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                },
                "marketplace_receipt": {
                    "marketplace_id": "dnam.store",
                    "license_id": "lic1",
                    "buyer_hash": "0000000000000000000000000000000000000000000000000000000000000000",
                    "license_tier": "personal",
                    "issued_at": "2026-10-23T00:00:00Z",
                    "buyer_arena_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                    "receipt_sig_ed25519": "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                }
            }
        },
        "io_config": {},
        "control_bus": {},
        "core1_hybrid_ssm": {},
        "core2_mufdn": {},
        "slimmable_tiers": {
            "lite": { "channels": 4, "fdn_lines": 4, "arena_end_byte_offset": 10752, "prefix_crc32c": 0 },
            "full": { "channels": 8, "fdn_lines": 8, "arena_end_byte_offset": 22336, "prefix_crc32c": 0 },
            "ultra": { "channels": 12, "fdn_lines": 8, "arena_end_byte_offset": 36224, "prefix_crc32c": 0 }
        },
        "controls_schema": [
            {
                "id": "Gain",
                "name": "Gain",
                "type": "range",
                "declared_range": [0.0, 10.0],
                "default": 5.0,
                "conditional_min": { "UnknownSw=on": 2.0 }
            },
            {
                "id": "Sw",
                "name": "Sw",
                "type": "switch",
                "declared_options": ["off", "on"],
                "active_options": ["nonexistent"],
                "default": "off"
            }
        ]
    }"#;
    let parsed_min: DnamJsonHeader = serde_json::from_str(minimal_json).unwrap();
    assert_eq!(parsed_min.version, "1.0.0");
    assert_eq!(parsed_min.container_kind, "dnam");
    assert_eq!(parsed_min.architecture, "DNAM_TriPhysics");
    assert_eq!(parsed_min.sample_rate_train_hz, 48_000);
    assert_eq!(parsed_min.metadata.input_reference_dbfs, -20.5);
    assert_eq!(parsed_min.io_config.inputs, 2);
    assert_eq!(parsed_min.slimmable_tiers.lite.head_svd_rank, 4);
    assert_eq!(parsed_min.control_bus.decimation, 16);
    assert_eq!(parsed_min.core1_hybrid_ssm.blocks, 4);
    assert_eq!(parsed_min.core2_mufdn.dc_blocker_pole, 0.995);
    let compiled = parsed_min.compile_clamp_descriptors().unwrap();
    assert_eq!(compiled[1].active_options_mask, 1);

    // 2. SHA-256 2-block padding branch (rem.len() >= 56) & JCS canonicalization types
    let msg_60 = [b'a'; 60];
    let digest_60 = sha256_digest(&msg_60);
    assert_ne!(digest_60, [0u8; 32]);

    let mut hdr_jcs = parsed_min.clone();
    hdr_jcs.metadata.signatures = None;
    hdr_jcs.controls_schema[0].user_metadata.insert(
        "misc".to_string(),
        serde_json::json!({"null_v": null, "b_t": true, "b_f": false, "zero": 0.0}),
    );
    let dummy_slice = TensorSliceEntry {
        arena: 3,
        dtype: "f64".to_string(),
        packing: None,
        feature_dim: None,
        packed_len: None,
        shape: None,
        byte_offset: 0,
        byte_length: 64,
    };
    hdr_jcs.stems_registry = Some(vec![StemRegistryEntry {
        stem_id: "s0".to_string(),
        controls_meta: BTreeMap::from([
            ("null_v".to_string(), serde_json::Value::Null),
            ("bool_t".to_string(), serde_json::Value::Bool(true)),
            ("bool_f".to_string(), serde_json::Value::Bool(false)),
            ("zero_f".to_string(), serde_json::json!(0.0)),
        ]),
        provenance: StemProvenance {
            input_pcm_sha256: "00".repeat(32),
            target_pcm_sha256: "11".repeat(32),
            num_frames: 100,
        },
        stem_arena3_crc32c: 0,
        user_metadata: BTreeMap::new(),
        extracted_physics: None,
        g_audio_uplo_u_f64: dummy_slice.clone(),
        c_audio_tier_f64: dummy_slice,
    }]);
    let imm_sha = compute_immutable_header_sha256(&hdr_jcs).unwrap();
    assert_ne!(imm_sha, [0u8; 32]);

    // 3. Provenance error branches & watermarking bounds
    let rcp = RustCryptoProvider::new();
    assert!(verify_container_provenance(&hdr_jcs, true, &[0u8; 64], &rcp).is_err());
    assert!(verify_container_provenance(&parsed_min, true, &[0u8; 64], &rcp).is_err());
    let mut bad_arena_buf = [0u8; 100];
    assert!(chacha20_xor_arena_blocks(&mut bad_arena_buf, &[0u8; 32], &[0u8; 12], 0).is_err());
    assert!(
        apply_isometric_gauge_watermark(&mut bad_arena_buf, &[1.0; 48], &[[0.0; 4]; 4]).is_err()
    );
    assert!(extract_structural_hash_160d(&bad_arena_buf).is_err());
    assert!(apply_tardos_watermark_4608(&mut bad_arena_buf, &[1.0; 4608], 1e-4).is_err());
    assert!(score_tardos_watermark_4608(&bad_arena_buf, &bad_arena_buf, &[1.0; 4608]).is_err());
    assert_eq!(structural_cosine_similarity(&[0.0; 160], &[0.0; 160]), 0.0);

    // 4. Control Clamp & Lobatto Basis degree 0, 1, 2, 3 & anti-zipper slewing + radial soft-clip
    for deg in 0..=3 {
        let mut d = ControlClampDescriptor::empty();
        d.ctrl_kind = 0;
        d.lobatto_degree = deg;
        d.anchor_nodes_norm = [0.0, -0.5, 0.5, 0.0];
        let b32 = d.eval_lobatto_basis(0.25);
        let b64 = eval_newton_lobatto_basis_f64(0.25, [0.0, -0.5, 0.5], deg as usize);
        assert!((b32[0] as f64 - b64[0]).abs() < 1e-6);
    }
    let mut d_cond = ControlClampDescriptor {
        ctrl_kind: 0,
        lobatto_degree: 2,
        active_options_mask: 1,
        default_val: 5.0,
        safe_min_base: 0.0,
        safe_max_base: 10.0,
        scale_norm: 0.2,
        offset_norm: -1.0,
        anchor_nodes_norm: [0.0, -1.0, 1.0, 0.0],
        cond_rules: [
            ConditionalClampRule {
                switch_ctrl_idx: 1,
                switch_opt_idx: 1,
                rule_op: 1, // max_ceil = 2.0
                _pad: 0,
                bound_val: 2.0,
            },
            ConditionalClampRule {
                switch_ctrl_idx: 1,
                switch_opt_idx: 1,
                rule_op: 2, // min_floor = 5.0 (causes eff_max < eff_min -> eff_max = eff_min)
                _pad: 0,
                bound_val: 5.0,
            },
            ConditionalClampRule::empty(),
        ],
    };
    let mut sw_opts = [0u8; 16];
    sw_opts[1] = 1;
    let (clamped_v, _, was_cl) = d_cond.clamp_and_normalize(8.0, &sw_opts);
    assert_eq!(clamped_v, 5.0);
    assert!(was_cl);
    d_cond.cond_rules[0].bound_val = 7.0;
    let (clamped_v2, _, _) = d_cond.clamp_and_normalize(1.0, &sw_opts);
    assert_eq!(clamped_v2, 5.0);

    // Exercise ControlBus load_from_arena0 with mh_val, r_train, and non-zero P_svd / FM weights + slewing
    let mut arena0 = vec![0u8; ARENA_OFFSET_LITE_END as usize];
    arena0[8064 + 16..8064 + 20].copy_from_slice(&0.95_f32.to_le_bytes());
    arena0[8064 + 32..8064 + 36].copy_from_slice(&0.5_f32.to_le_bytes());
    for off in (8128..9216).step_by(4) {
        arena0[off..off + 4].copy_from_slice(&2.0_f32.to_le_bytes());
    }
    let mut cb_slew = Box::new(ControlBus::<ScalarBackend>::new());
    cb_slew.load_from_arena0(&arena0, 2, 1.0);
    cb_slew.descriptors[0] = compiled[0];
    cb_slew.descriptors[1] = compiled[1];
    cb_slew.num_controls = 2;
    cb_slew.set_control_value(0, 8.0);
    cb_slew.set_control_value(99, 10.0);
    cb_slew.set_control_immediate(99, 10.0);
    assert!(cb_slew.tick_3khz(0.5, true, &arena0));
    // Snap branch on anti-zipper when pole == 0.0 and radial soft-clip (z_norm_sq > 0.25)
    cb_slew.anti_zipper_pole = 0.0;
    cb_slew.set_control_value(0, 10.0);
    cb_slew.set_control_immediate(1, 1.0);
    assert!(cb_slew.tick_3khz(0.5, true, &arena0));

    // 5. Core 2 capacity growth, custom globals, Lite-only arena, active_fdn_groups == 0, and 128-sample switch delay crossfade
    let mut c2 = Box::new(ModulatedUfdn::<ScalarBackend>::with_max_capacity(8));
    let mut c2_arena = vec![0u8; ARENA_OFFSET_FULL_END as usize];
    c2_arena[6784 + 192 + 4..6784 + 192 + 8].copy_from_slice(&0.995_f32.to_le_bytes());
    c2_arena[6784 + 192 + 8..6784 + 192 + 12].copy_from_slice(&64.0_f32.to_le_bytes());
    c2_arena[6784 + 192 + 12..6784 + 192 + 16].copy_from_slice(&128.0_f32.to_le_bytes());
    let zoh = ZohRescaledParams::compute(48_000, 48_000, 16, 0.9993, 0.995, 64.0, 128).unwrap();
    c2.load_from_arena(&c2_arena[..ARENA_OFFSET_LITE_END as usize], true, 10, &zoh);
    c2.load_from_arena(&c2_arena, true, 10, &zoh);
    let out_zero_groups = c2.process_sample(0.5, -0.5, &[[0.1; 4]; 2], &[0.0; 8], 0);
    assert_eq!(out_zero_groups, [0.0, 0.0, 0.5, -0.5]);
    c2.set_target_delay_samples(99, 200.0);
    c2.set_target_delay_samples(0, 200.0);
    let out_xfade = c2.process_sample(0.5, -0.5, &[[0.1; 4]; 2], &[0.0; 8], 2);
    assert!(out_xfade[0].is_finite());

    // 6. DynamicDnamEngine fallback, reload_model_in_place on borrowed engine, and step_frame 3kHz head collapse
    let lite_bytes = make_test_container(ARENA_OFFSET_LITE_END as usize, false);
    let dyn_wasm_fallback = Box::new(
        DynamicDnamEngine::from_bytes_with_backend(
            &lite_bytes,
            Tier::Lite,
            48_000,
            BackendKind::WasmSimd128,
        )
        .unwrap(),
    );
    assert_eq!(dyn_wasm_fallback.backend_kind(), BackendKind::Scalar);
    let mut borrowed_eng =
        Box::new(DnamEngine::<ScalarBackend>::from_bytes(&lite_bytes, Tier::Lite, 48_000).unwrap());
    borrowed_eng.set_control_value(0, 9.0);
    let mut out_l = [0.0_f32; 32];
    let mut out_r = [0.0_f32; 32];
    borrowed_eng.process_block(&[0.2; 32], &[0.2; 32], &mut out_l, &mut out_r);
    assert!(
        borrowed_eng
            .reload_model_in_place(&lite_bytes, Tier::Lite, 48_000)
            .is_ok()
    );
    // Encrypted arena flag rejection in reload_model_in_place
    let mut enc_bytes = lite_bytes.clone();
    enc_bytes[6] |= 0x01; // FLAG_ENCRYPTED_ARENA = 1 << 8 (little-endian byte 6 bit 0)
    assert!(
        borrowed_eng
            .reload_model_in_place(&enc_bytes, Tier::Lite, 48_000)
            .is_err()
    );
}
