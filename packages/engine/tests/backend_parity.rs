//! Cross-Backend Parity & 4-Topology Verification Suite (`tests/backend_parity.rs`).
//!
//! Verifies that `ScalarBackend`, `NeonBackend`, `ArmSmeBackend`, and `DynamicDnamEngine`
//! produce matching audio across `Lite`, `Full`, and `Ultra` tiers and all 4 I/O topologies.

use std::collections::BTreeMap;

use dnam_engine::container::{
    ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_ULTRA_END, build_dnam_container,
};
use dnam_engine::math::{BackendKind, DefaultMathBackend, ScalarBackend};
use dnam_engine::schema::{
    ControlBusConfig, Core1Config, Core2Config, DatasetProvenance, DnamJsonHeader, IoConfig,
    ModelMetadata, SlimmableTiersConfig, TierProfile,
};
use dnam_engine::{DnamEngine, DynamicDnamEngine, Tier};

fn build_parity_container() -> Vec<u8> {
    let mut header = DnamJsonHeader {
        version: "1.0.0".to_string(),
        container_kind: "dnam".to_string(),
        architecture: "DNAM_TriPhysics".to_string(),
        sample_rate_train_hz: 48_000,
        metadata: ModelMetadata {
            name: "Backend Parity Test Rig".to_string(),
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
    for i in 0..4 {
        let off = (i * 4 + (i % 2)) * 4;
        arena[off..off + 4].copy_from_slice(&0.6_f32.to_le_bytes());
    }
    for t in 0..3 {
        arena[104 + t * 4..108 + t * 4].copy_from_slice(&1.0_f32.to_le_bytes());
    }
    arena[116..120].copy_from_slice(&0.2_f32.to_le_bytes());
    arena[120..124].copy_from_slice(&0.8_f32.to_le_bytes());
    arena[124..128].copy_from_slice(&0.9993_f32.to_le_bytes());

    for tier in 0..3 {
        for blk in 0..4 {
            let (blk_base, adaa0_off, adaa1_off, rg_off) = match tier {
                0 => (128 + blk * 1024, 384, 832, 896),
                1 => (10_816 + blk * 1792, 768, 1600, 1664),
                _ => (22_400 + blk * 2560, 1152, 2368, 2432),
            };
            for w in 0..96 {
                let val = 0.015_f32 * (((blk * 19 + w + tier * 5) % 9) as f32 - 4.0);
                let off = blk_base + w * 4;
                arena[off..off + 4].copy_from_slice(&val.to_le_bytes());
            }
            for &a_off in &[adaa0_off, adaa1_off] {
                let base = blk_base + a_off;
                for lane in 0..4 {
                    arena[base + 16 + lane * 4..base + 16 + (lane + 1) * 4]
                        .copy_from_slice(&0.6_f32.to_le_bytes());
                    arena[base + 32 + lane * 4..base + 32 + (lane + 1) * 4]
                        .copy_from_slice(&0.8_f32.to_le_bytes());
                    arena[base + 48 + lane * 4..base + 48 + (lane + 1) * 4]
                        .copy_from_slice(&0.25_f32.to_le_bytes());
                }
            }
            let rg_base = blk_base + rg_off;
            for lane in 0..4 {
                arena[rg_base + lane * 4..rg_base + (lane + 1) * 4]
                    .copy_from_slice(&0.91_f32.to_le_bytes());
                arena[rg_base + 16 + lane * 4..rg_base + 16 + (lane + 1) * 4]
                    .copy_from_slice(&0.06_f32.to_le_bytes());
                arena[rg_base + 32 + lane * 4..rg_base + 32 + (lane + 1) * 4]
                    .copy_from_slice(&0.12_f32.to_le_bytes());
                arena[rg_base + 48 + lane * 4..rg_base + 48 + (lane + 1) * 4]
                    .copy_from_slice(&0.35_f32.to_le_bytes());
                arena[rg_base + 64 + lane * 4..rg_base + 64 + (lane + 1) * 4]
                    .copy_from_slice(&0.45_f32.to_le_bytes());
                arena[rg_base + 80 + lane * 4..rg_base + 80 + (lane + 1) * 4]
                    .copy_from_slice(&0.30_f32.to_le_bytes());
                arena[rg_base + 96 + lane * 4..rg_base + 96 + (lane + 1) * 4]
                    .copy_from_slice(&(-0.20_f32).to_le_bytes());
            }
        }
    }

    for &hbase in &[4224_usize, 17_984_usize, 32_640_usize] {
        for i in 0..128 {
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

    for &tier in &[Tier::Lite, Tier::Full, Tier::Ultra] {
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
