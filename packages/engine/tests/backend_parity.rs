//! Cross-Backend Parity & 4-Topology Verification Suite (`tests/backend_parity.rs`).
//!
//! - `verify[arch.simd.two-level-traits]`
//! - `verify[arch.simd.single-library-dispatch]`
//! - `verify[dsp.op-m.recurrence.stereo-simd]`
//! - `verify[dsp.op-m.tracker.stereo-batching]`
//! - `verify[dsp.op-n.polyadaa.stereo-and-rail-guard]`
//! - `verify[dsp.op-n.stage2.conv6-simd]`
//! - `verify[dsp.op-t.reader.contiguous-fp16x4]`
//! - `verify[dsp.stage3.mirrored-fir-simd]`
//! - `verify[dsp.stage4.simd-tank-loop]`
//! - `verify[dsp.head.precollapse-and-zerocopy]`
//! - `verify[dsp.ctrl.op-n.static-and-interior-fastpath]`
//! - `verify[dsp.ctrl.dispatch.sparse-and-active-ramps]`
//!
//! Verifies that `ScalarBackend`, `NeonBackend`, `ArmSmeBackend`, and `DynamicDnamEngine`
//! produce matching audio across `Lite`, `Full`, and `Ultra` tiers and all 4 I/O topologies.

use std::collections::BTreeMap;

use dnam_engine::container::{
    ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_PRO_END, ARENA_OFFSET_ULTRA_END,
    build_dnam_container,
};
use dnam_engine::engine::EngineBackend;
use dnam_engine::math::{BackendKind, DefaultMathBackend, ScalarBackend, SimdVec4};
use dnam_engine::schema::{
    ControlBusConfig, DatasetProvenance, DnamJsonHeader, IoConfig, ModelMetadata, RouterConfig,
    SlimmableTiersConfig, Stage1PreFxConfig, Stage2CascadeConfig, Stage3CabinetKautzConfig,
    Stage4SpatialTankConfig, TierProfile,
};
use dnam_engine::{DnamEngine, DynamicDnamEngine, Tier};

fn build_parity_container() -> Vec<u8> {
    let mut header = DnamJsonHeader {
        version: "2.0.0".to_string(),
        container_kind: "dnam".to_string(),
        architecture: "DNAM_TORA".to_string(),
        sample_rate_train_hz: 48_000,
        metadata: ModelMetadata {
            name: "Backend Parity Test Rig".to_string(),
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
        controls_schema: Vec::new(),
        presets: Vec::new(),
        control_bus: ControlBusConfig::default(),
        stage0_router: RouterConfig::default(),
        stage1_pre: Stage1PreFxConfig::default(),
        stage2_cascade: Stage2CascadeConfig::default(),
        stage3_cabinet: Stage3CabinetKautzConfig::default(),
        stage4_tank: Stage4SpatialTankConfig::default(),
        tensor_directory: BTreeMap::new(),
    };

    let mut arena = vec![0u8; 64_000];
    arena[124..128].copy_from_slice(&0.9993_f32.to_le_bytes());
    for off in (4_096..4_288).step_by(4) {
        arena[off..off + 4].copy_from_slice(&0.5_f32.to_le_bytes());
    }

    let cascade_w_bases = [1_024_usize, 11_264_usize, 22_336_usize, 36_224_usize];
    let cascade_w_counts = [768_usize, 1536_usize, 2304_usize, 3072_usize];
    for tier in 0..4 {
        let cw_base = cascade_w_bases[tier];
        for w in 0..cascade_w_counts[tier] {
            let val = 0.015_f32 * (((w + tier * 5) % 9) as f32 - 4.0);
            let off = cw_base + w * 4;
            arena[off..off + 4].copy_from_slice(&val.to_le_bytes());
        }
        for blk in 0..4 {
            let k_base = dnam_engine::provenance::kautz_soa_byte_offset(tier, blk);
            for lane in 0..4 {
                arena[k_base + lane * 4..k_base + (lane + 1) * 4]
                    .copy_from_slice(&0.55_f32.to_le_bytes());
                arena[k_base + 16 + lane * 4..k_base + 16 + (lane + 1) * 4]
                    .copy_from_slice(&0.05_f32.to_le_bytes());
                arena[k_base + 32 + lane * 4..k_base + 32 + (lane + 1) * 4]
                    .copy_from_slice(&0.45_f32.to_le_bytes());
                arena[k_base + 48 + lane * 4..k_base + 48 + (lane + 1) * 4]
                    .copy_from_slice(&0.25_f32.to_le_bytes());
                arena[k_base + 64 + lane * 4..k_base + 64 + (lane + 1) * 4]
                    .copy_from_slice(&0.15_f32.to_le_bytes());
                arena[k_base + 80 + lane * 4..k_base + 80 + (lane + 1) * 4]
                    .copy_from_slice(&(-0.10_f32).to_le_bytes());
            }
        }
    }

    for &hbase in &[4_928_usize, 18_048_usize, 31_936_usize, 57_600_usize] {
        for i in 0..640 {
            let val = 0.04_f32 * (((i * 11 + 3) % 15) as f32 - 7.0);
            let off = hbase + i * 4;
            arena[off..off + 4].copy_from_slice(&val.to_le_bytes());
        }
    }

    build_dnam_container(&mut header, &arena, &arena, false).unwrap()
}

#[test]
fn test_cross_backend_parity_and_4_io_topologies() {
    let container = build_parity_container();
    let in_l: Vec<f32> = (0..512).map(|n| 0.4 * ((n as f32) * 0.09).sin()).collect();
    let in_r: Vec<f32> = (0..512).map(|n| 0.4 * ((n as f32) * 0.13).cos()).collect();

    for &tier in &[Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
        let mut eng_scalar =
            DnamEngine::<ScalarBackend>::from_bytes(&container, tier, 48_000).unwrap();
        let mut eng_default =
            DnamEngine::<DefaultMathBackend>::from_bytes(&container, tier, 48_000).unwrap();

        let (mut out_sl, mut out_sr) = (vec![0.0_f32; 512], vec![0.0_f32; 512]);
        let (mut out_dl, mut out_dr) = (vec![0.0_f32; 512], vec![0.0_f32; 512]);

        eng_scalar.process_block(&in_l, &in_r, &mut out_sl, &mut out_sr);
        eng_default.process_block(&in_l, &in_r, &mut out_dl, &mut out_dr);

        let max_err_l = out_sl
            .iter()
            .zip(out_dl.iter())
            .fold(0.0_f32, |acc, (&a, &b)| acc.max((a - b).abs()));
        let max_err_r = out_sr
            .iter()
            .zip(out_dr.iter())
            .fold(0.0_f32, |acc, (&a, &b)| acc.max((a - b).abs()));
        assert!(
            max_err_l < 1e-5 && max_err_r < 1e-5,
            "Backend parity mismatch on {tier:?}: max_err_l={max_err_l:e}, max_err_r={max_err_r:e}"
        );

        #[cfg(target_arch = "aarch64")]
        {
            let mut eng_sve2 = DnamEngine::<dnam_engine::math::ArmSve2Backend>::from_bytes(
                &container, tier, 48_000,
            )
            .unwrap();
            let (mut out_sve2_l, mut out_sve2_r) = (vec![0.0_f32; 512], vec![0.0_f32; 512]);
            eng_sve2.process_block(&in_l, &in_r, &mut out_sve2_l, &mut out_sve2_r);
            let sve2_err_l = out_sl
                .iter()
                .zip(out_sve2_l.iter())
                .fold(0.0_f32, |acc, (&a, &b)| acc.max((a - b).abs()));
            let sve2_err_r = out_sr
                .iter()
                .zip(out_sve2_r.iter())
                .fold(0.0_f32, |acc, (&a, &b)| acc.max((a - b).abs()));
            assert!(
                sve2_err_l < 1e-5 && sve2_err_r < 1e-5,
                "ARM SVE2 parity mismatch on {tier:?}: sve2_err_l={sve2_err_l:e}, sve2_err_r={sve2_err_r:e}"
            );

            let mut eng_sme = DnamEngine::<dnam_engine::math::ArmSmeBackend>::from_bytes(
                &container, tier, 48_000,
            )
            .unwrap();
            let (mut out_sme_l, mut out_sme_r) = (vec![0.0_f32; 512], vec![0.0_f32; 512]);
            eng_sme.process_block(&in_l, &in_r, &mut out_sme_l, &mut out_sme_r);
            let sme_err_l = out_sl
                .iter()
                .zip(out_sme_l.iter())
                .fold(0.0_f32, |acc, (&a, &b)| acc.max((a - b).abs()));
            let sme_err_r = out_sr
                .iter()
                .zip(out_sme_r.iter())
                .fold(0.0_f32, |acc, (&a, &b)| acc.max((a - b).abs()));
            assert!(
                sme_err_l < 1e-5 && sme_err_r < 1e-5,
                "ARM SME parity mismatch on {tier:?}: sme_err_l={sme_err_l:e}, sme_err_r={sme_err_r:e}"
            );
        }

        #[cfg(target_arch = "x86_64")]
        {
            for (mut eng_x86_l, mut eng_x86_r) in [
                {
                    let mut eng = DnamEngine::<dnam_engine::math::X86Sse2Backend>::from_bytes(
                        &container, tier, 48_000,
                    )
                    .unwrap();
                    let (mut l, mut r) = (vec![0.0_f32; 512], vec![0.0_f32; 512]);
                    eng.process_block(&in_l, &in_r, &mut l, &mut r);
                    (l, r)
                },
                {
                    let mut eng = DnamEngine::<dnam_engine::math::X86Sse42Backend>::from_bytes(
                        &container, tier, 48_000,
                    )
                    .unwrap();
                    let (mut l, mut r) = (vec![0.0_f32; 512], vec![0.0_f32; 512]);
                    eng.process_block(&in_l, &in_r, &mut l, &mut r);
                    (l, r)
                },
                {
                    let mut eng = DnamEngine::<dnam_engine::math::X86Avx2FmaBackend>::from_bytes(
                        &container, tier, 48_000,
                    )
                    .unwrap();
                    let (mut l, mut r) = (vec![0.0_f32; 512], vec![0.0_f32; 512]);
                    eng.process_block(&in_l, &in_r, &mut l, &mut r);
                    (l, r)
                },
            ] {
                let err_l = out_sl
                    .iter()
                    .zip(eng_x86_l.drain(..))
                    .fold(0.0_f32, |acc, (&a, b)| acc.max((a - b).abs()));
                let err_r = out_sr
                    .iter()
                    .zip(eng_x86_r.drain(..))
                    .fold(0.0_f32, |acc, (&a, b)| acc.max((a - b).abs()));
                assert!(err_l < 1e-5 && err_r < 1e-5);
            }
        }

        // Verify Topology 2 (in-place) matches Topology 1 (separate buffers)
        eng_default.reset_to_dc_equilibrium();
        let mut ip_l = in_l.clone();
        let mut ip_r = in_r.clone();
        eng_default.process_block_in_place(&mut ip_l, &mut ip_r);
        assert_eq!(ip_l, out_dl);
        assert_eq!(ip_r, out_dr);

        // Verify Topology 3 (mono_to_stereo) and Topology 4 (mono)
        eng_default.reset_to_dc_equilibrium();
        let (mut m2s_l, mut m2s_r) = (vec![0.0_f32; 512], vec![0.0_f32; 512]);
        eng_default.process_mono_to_stereo(&in_l, &mut m2s_l, &mut m2s_r);

        eng_default.reset_to_dc_equilibrium();
        let mut m_out = vec![0.0_f32; 512];
        eng_default.process_mono(&in_l, &mut m_out);
        assert_eq!(m2s_l, m_out);
    }

    // Verify DynamicDnamEngine across available backends
    let mut dyn_scalar = DynamicDnamEngine::from_bytes_with_backend(
        &container,
        Tier::Full,
        48_000,
        BackendKind::Scalar,
    )
    .unwrap();
    let mut dyn_default = DynamicDnamEngine::from_bytes(&container, Tier::Full, 48_000).unwrap();
    let (mut d_sl, mut d_sr) = (vec![0.0_f32; 128], vec![0.0_f32; 128]);
    let (mut d_dl, mut d_dr) = (vec![0.0_f32; 128], vec![0.0_f32; 128]);
    dyn_scalar.process_block(&in_l[..128], &in_r[..128], &mut d_sl, &mut d_sr);
    dyn_default.process_block(&in_l[..128], &in_r[..128], &mut d_dl, &mut d_dr);

    let dyn_err = d_sl
        .iter()
        .zip(d_dl.iter())
        .fold(0.0_f32, |acc, (&a, &b)| acc.max((a - b).abs()));
    assert!(dyn_err < 1e-5);
}

fn exercise_math_backend<B: EngineBackend>() {
    let _ = B::NAME;
    let _ = B::KIND.name();

    let v0 = B::Vec4::zero();
    assert_eq!(v0.to_array(), [0.0; 4]);
    let va = B::Vec4::load(&[1.0, -2.0, 3.0, -4.0]);
    let mut stored = [0.0_f32; 4];
    va.store(&mut stored);
    assert_eq!(stored, [1.0, -2.0, 3.0, -4.0]);
    let vb = B::Vec4::splat(2.0);
    let vp = B::Vec4::splat_pair(1.5, -1.5);
    assert_eq!(vp.to_array(), [1.5, -1.5, 1.5, -1.5]);
    assert_eq!(va.pair_swap_neg().to_array(), [2.0, 1.0, 4.0, 3.0]);
    assert_eq!(
        B::Vec4::from_array([f32::from_bits(1), -f32::from_bits(1), 1.0, -1.0])
            .flush_subnormal()
            .to_array(),
        [0.0, -0.0, 1.0, -1.0]
    );
    let _ = va.add(vb);
    let _ = va.sub(vb);
    let _ = va.mul(vb);
    let _ = va.fma(vb, vb);
    let _ = va
        .fma_lane::<0>(vb, vp)
        .fma_lane::<1>(vb, vp)
        .fma_lane::<2>(vb, vp)
        .fma_lane::<3>(vb, vp);
    let _ = va.fms(vb, vb);
    let _ = va.max(vb);
    let _ = va.min(vb);
    let _ = va.clamp(B::Vec4::splat(-1.0), B::Vec4::splat(1.0));
    let _ = va.abs();
    let hs_a = va.horizontal_sum();
    let hs_b = vb.horizontal_sum();
    let (hsp_a, hsp_b) = va.horizontal_sum_pair(vb);
    assert!((hs_a - hsp_a).abs() < 1e-6 && (hs_b - hsp_b).abs() < 1e-6);
    let _ = va.dot(vb);

    let mat = [[0.1_f32; 4]; 4];
    let v_mat = B::fma_mat4x4_vec4(B::Vec4::zero(), &mat, &[1.0, 2.0, 3.0, 4.0]);
    assert!(v_mat.horizontal_sum().is_finite());

    let (d16_l, d16_r) = B::dot16_stereo(&mat, &mat, &mat, &mat);
    assert!((d16_l - 0.16).abs() < 1e-6 && (d16_r - 0.16).abs() < 1e-6);
    let (d16_vl, d16_vr) =
        dnam_engine::stages::dot16_stereo_vec4::<B::Vec4>(&mat, &mat, &mat, &mat);
    assert!((d16_vl - 0.16).abs() < 1e-6 && (d16_vr - 0.16).abs() < 1e-6);

    let h_frame = [[0.2_f32; 4]; 4];
    let h_taps = [&h_frame; 6];
    let w1 = [[[0.1_f32; 4]; 4]; 6];
    let w2 = [[[0.1_f32; 4]; 4]; 12];
    let w3 = [[[0.1_f32; 4]; 4]; 18];
    let w4 = [[[0.1_f32; 4]; 4]; 24];
    let _ = B::conv6_taps_1group(B::Vec4::zero(), &w1, &h_taps);
    let _ = B::conv6_taps_2groups(B::Vec4::zero(), &w2, &h_taps);
    let _ = B::conv6_taps_3groups(B::Vec4::zero(), &w3, &h_taps);
    let _ = B::conv6_taps_4groups(B::Vec4::zero(), &w4, &h_taps);
    let _ = dnam_engine::stages::conv6_taps_1group_vec4::<B::Vec4>(B::Vec4::zero(), &w1, &h_taps);
    let _ = dnam_engine::stages::conv6_taps_2groups_vec4::<B::Vec4>(B::Vec4::zero(), &w2, &h_taps);
    let _ = dnam_engine::stages::conv6_taps_3groups_vec4::<B::Vec4>(B::Vec4::zero(), &w3, &h_taps);
    let _ = dnam_engine::stages::conv6_taps_4groups_vec4::<B::Vec4>(B::Vec4::zero(), &w4, &h_taps);

    let _ = B::poly_adaa_algebraic_4ch(va, vb, B::Vec4::splat(1.0), B::Vec4::splat(0.1));
    let _ = B::cubic_lagrange_weights_4ch(B::Vec4::splat(0.25));
    let tile64 = [0.1_f32; 64];
    assert!((B::tile_dot64(&tile64, &tile64) - 0.64).abs() < 1e-5);
    assert!(
        (dnam_engine::engine::tile_dot64_vec4::<B::Vec4>(&tile64, &tile64) - 0.64).abs() < 1e-5
    );
    let stereo64 = B::tile_dot64_stereo(&tile64, &tile64, &tile64);
    assert!((stereo64[0] - 0.64).abs() < 1e-5 && (stereo64[1] - 0.64).abs() < 1e-5);
    let stereo64_v =
        dnam_engine::engine::tile_dot64_stereo_vec4::<B::Vec4>(&tile64, &tile64, &tile64);
    assert!((stereo64_v[0] - 0.64).abs() < 1e-5 && (stereo64_v[1] - 0.64).abs() < 1e-5);

    let svd_base = [0.5_f32; 16];
    let svd_modes = [0.1_f32; 64];
    let mut svd_collapsed = [0.0_f32; 16];
    B::tile_collapse_svd_slice(
        &svd_base,
        &svd_modes,
        &[1.0, 2.0, 3.0, 4.0],
        4,
        &mut svd_collapsed,
    );
    assert!((svd_collapsed[0] - 1.5).abs() < 1e-5);

    let fp16_4 = B::f32x4_to_fp16x4(&[0.5, -0.5, 1.0, 0.0]);
    let f32_4 = B::fp16x4_to_f32x4(&fp16_4);
    assert!((f32_4[0] - 0.5).abs() < 1e-3);

    let r0 = B::rsqrt_nr2_f32(4.0);
    let (rp0, rp1) = B::rsqrt_nr2_pair_f32(4.0, 0.25);
    assert!((r0 - 0.5).abs() < 1e-3 && (rp0 - 0.5).abs() < 1e-3 && (rp1 - 2.0).abs() < 1e-2);
    let (sd_l, su_l, sd_r, su_r) = B::tracker_exp_scales_f32(0.5, 2.0);
    assert_eq!((sd_l * su_l, sd_r * su_r), (1.0, 1.0));

    assert_eq!(B::crc32c(b"123456789"), 0xE306_9283);
    assert_eq!(
        dnam_engine::math::crc32c::crc32c_software_slice(0, b"123456789"),
        0xE306_9283
    );
    let (cp, sp) = B::fast_unit_phasor(0.7);
    assert!((cp * cp + sp * sp - 1.0).abs() < 1e-4);
}

#[test]
fn test_all_math_backends_and_backend_kind_names() {
    for (kind, expected) in [
        (BackendKind::Scalar, "Scalar"),
        (BackendKind::ArmNeon, "ARM64 NEON"),
        (BackendKind::Neon, "ARM64 NEON"),
        (BackendKind::ArmSve2, "ARMv9 SVE2"),
        (BackendKind::ArmSme, "ARM SME"),
        (BackendKind::X86Sse2, "x86 SSE2"),
        (BackendKind::X86Sse42, "x86 SSE4.2"),
        (BackendKind::X86Avx2Fma, "x86 AVX2"),
        (BackendKind::X86Simd, "x86 AVX2"),
        (BackendKind::WasmSimd128, "WASM SIMD"),
    ] {
        assert_eq!(kind.name(), expected);
    }
    let _ = dnam_engine::math::detect_best_backend();

    exercise_math_backend::<ScalarBackend>();
    exercise_math_backend::<DefaultMathBackend>();
    #[cfg(target_arch = "aarch64")]
    {
        exercise_math_backend::<dnam_engine::math::ArmSve2Backend>();
        exercise_math_backend::<dnam_engine::math::ArmSmeBackend>();
    }
    #[cfg(target_arch = "x86_64")]
    {
        exercise_math_backend::<dnam_engine::math::X86Sse2Backend>();
        exercise_math_backend::<dnam_engine::math::X86Sse42Backend>();
        exercise_math_backend::<dnam_engine::math::X86Avx2FmaBackend>();
    }
}
