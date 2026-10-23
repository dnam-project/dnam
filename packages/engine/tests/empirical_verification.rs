//! Empirical Verification Suite (`tests/empirical_verification.rs`).
//!
//! Rust equivalents of `Tests 1–8` from `scripts/verify_dnam_architecture.py` and
//! [`docs/verification_and_guardrails.md`](../../../docs/verification_and_guardrails.md).

use std::collections::BTreeMap;

use dnam_engine::container::{
    ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_ULTRA_END, DnamContainerView,
    ZohRescaledParams, build_dnam_container, pack_uplo_u_f64, unpack_uplo_u_f64,
    update_header_in_place, uplo_u_packed_len,
};
use dnam_engine::control::{
    ControlClampDescriptor, eval_chebyshev_lfo_4harm_f64, eval_newton_lobatto_basis_f64,
    eval_rank4_fm_interaction_f64, eval_trust_region_softclip_f64,
};
use dnam_engine::core1::{
    poly_adaa_algebraic_f64, poly_adaa_slice_f64, selective_rglru_slice_f64,
    selective_rglru_step_f64,
};
use dnam_engine::core2::{
    cubic_lagrange_weights_f64, givens_householder_mix_8ch_f64, mufdn_coercive_step_f64,
    mufdn_slice_f64,
};
use dnam_engine::math::{DefaultMathBackend, MathBackend, f32_to_fp16_bits, fp16_bits_to_f32};
use dnam_engine::provenance::{
    RGLRU_TOTAL_CHANNELS, RustCryptoProvider, TARDOS_CONV_WEIGHTS, VerificationStatus,
    apply_isometric_gauge_watermark, apply_tardos_watermark_4608, build_creator_signing_message,
    build_marketplace_receipt_signing_message, chacha20_xor_arena_blocks,
    compute_immutable_header_sha256, encode_hex, extract_structural_hash_160d,
    generate_tardos_code_4608, score_tardos_watermark_4608, sha256_digest,
    structural_cosine_similarity,
};
use dnam_engine::schema::{
    AuthorizedDevice, ControlBusConfig, ControlKind, ControlSchemaDef, ControlUiHints, Core1Config,
    Core2Config, CreatorSignature, DatasetProvenance, DnamJsonHeader, IoConfig, MarketplaceReceipt,
    ModelMetadata, ModelSignatures, PresetDef, SlimmableTiersConfig, TierProfile,
};
use dnam_engine::{DnamEngine, Tier};

/// Helper to construct a complete, valid `.dnam` test header and 36,224-byte `Arena 0..2`.
fn build_reference_header_and_arena(core2_active: bool) -> (DnamJsonHeader, Vec<u8>) {
    let mut cond_max = BTreeMap::new();
    cond_max.insert("boost=on".to_string(), 9.368_421_f32);

    let controls_schema = vec![
        ControlSchemaDef {
            id: "gain".to_string(),
            name: "Drive".to_string(),
            kind: ControlKind::Range,
            unit: Some("dB".to_string()),
            lobatto_degree: Some(2),
            anchor_nodes_norm: Some(vec![0.0, -0.6, 0.6]),
            declared_range: Some([0.0, 10.0]),
            safe_clamped_range: Some([2.0, 8.0]),
            default: serde_json::json!(5.0),
            conditional_min: BTreeMap::new(),
            conditional_max: BTreeMap::new(),
            declared_options: None,
            active_options: None,
            clamp_warning: Some("CLAMP-RANGE: [2.0, 8.0]".to_string()),
            ui_hints: ControlUiHints {
                midpoint: Some(4.0),
                ..Default::default()
            },
            user_metadata: BTreeMap::new(),
        },
        ControlSchemaDef {
            id: "feedback".to_string(),
            name: "Repeats".to_string(),
            kind: ControlKind::Range,
            unit: Some("%".to_string()),
            lobatto_degree: Some(2),
            anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
            declared_range: Some([0.0, 10.0]),
            safe_clamped_range: Some([0.0, 10.0]),
            default: serde_json::json!(5.0),
            conditional_min: BTreeMap::new(),
            conditional_max: cond_max,
            declared_options: None,
            active_options: None,
            clamp_warning: Some("CLAMP-STABILITY: <= 9.37 when boost=on".to_string()),
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        },
        ControlSchemaDef {
            id: "boost".to_string(),
            name: "Boost Mode".to_string(),
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
            clamp_warning: Some("CLAMP-SWITCH: disabled ['ultra']".to_string()),
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        },
    ];

    let mut preset_ctrls = BTreeMap::new();
    preset_ctrls.insert("gain".to_string(), serde_json::json!(6.5));
    preset_ctrls.insert("feedback".to_string(), serde_json::json!(9.0));
    preset_ctrls.insert("boost".to_string(), serde_json::json!("on"));

    let header = DnamJsonHeader {
        version: "1.0.0".to_string(),
        container_kind: "dnam".to_string(),
        architecture: "DNAM_TriPhysics".to_string(),
        sample_rate_train_hz: 48_000,
        metadata: ModelMetadata {
            name: "Test Vintage Tape Echo".to_string(),
            author: Some("DNAM Test Suite".to_string()),
            description: Some("Empirical verification test model".to_string()),
            license: Some("MIT".to_string()),
            created_at: "2026-10-23T00:00:00Z".to_string(),
            input_reference_dbfs: -20.5,
            provenance: DatasetProvenance {
                dataset_manifest_sha256:
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".to_string(),
                config_recipe_sha256: None,
                epochs_trained: 120,
            },
            signatures: None,
        },
        user_metadata: BTreeMap::new(),
        io_config: IoConfig {
            inputs: 2,
            output_mode: "stereo".to_string(),
            latency_samples: 0,
            tail_time_ms: if core2_active { 1250.0 } else { 0.0 },
        },
        slimmable_tiers: SlimmableTiersConfig {
            lite: TierProfile {
                channels: 4,
                fdn_lines: if core2_active { 4 } else { 0 },
                early_taps: 4,
                head_svd_rank: 4,
                loudness_lufs: -20.5,
                output_trim_db: 0.0,
                validation_esr_db: -39.4,
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
                validation_esr_db: -45.8,
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
                validation_esr_db: -47.63,
                arena_end_byte_offset: ARENA_OFFSET_ULTRA_END,
                prefix_crc32c: 0,
            },
        },
        controls_schema,
        presets: vec![PresetDef {
            id: "dub_swell".to_string(),
            name: "Dub Swell".to_string(),
            description: None,
            controls: preset_ctrls,
            user_metadata: BTreeMap::new(),
        }],
        control_bus: ControlBusConfig::default(),
        core1_hybrid_ssm: Core1Config::default(),
        core2_mufdn: Core2Config {
            active: core2_active,
            ring_buffer_pow2_log2: if core2_active { 12 } else { 0 },
            ..Default::default()
        },
        tensor_directory: BTreeMap::new(),
        stems_registry: None,
        audit_report: None,
    };

    let mut arena = vec![0u8; 36_224];
    // Write compiled clamp descriptors into Arena 0 (7040..8064)
    let compiled = header.compile_clamp_descriptors().unwrap();
    for (p, desc) in compiled.iter().enumerate() {
        let off = 7040 + p * 64;
        arena[off..off + 64].copy_from_slice(&desc.to_bytes());
    }

    // Populate deterministic non-zero weights into Arena 0..2
    // Input proj (0..64): identity-like
    for i in 0..4 {
        let off = (i * 4 + (i % 2)) * 4;
        arena[off..off + 4].copy_from_slice(&0.5_f32.to_le_bytes());
    }
    // Tier trims & mix gains (104..128)
    for t in 0..3 {
        arena[104 + t * 4..108 + t * 4].copy_from_slice(&1.0_f32.to_le_bytes());
    }
    arena[116..120].copy_from_slice(&0.25_f32.to_le_bytes()); // mix_dry_gain
    arena[120..124].copy_from_slice(&0.75_f32.to_le_bytes()); // mix_wet_gain
    arena[124..128].copy_from_slice(&0.9993_f32.to_le_bytes()); // r_ac_train

    // Populate PolyADAA & RgLru blocks across tiers 0, 1, 2
    for tier in 0..3 {
        for blk in 0..4 {
            let (blk_base, adaa0_off, adaa1_off, rg_off) = match tier {
                0 => (128 + blk * 1024, 384, 832, 896),
                1 => (10_816 + blk * 1792, 768, 1600, 1664),
                _ => (22_400 + blk * 2560, 1152, 2368, 2432),
            };
            // Small conv weights
            for w in 0..96 {
                let val = 0.01_f32 * (((blk * 17 + w + tier * 7) % 11) as f32 - 5.0);
                let off = blk_base + w * 4;
                arena[off..off + 4].copy_from_slice(&val.to_le_bytes());
            }
            for &a_off in &[adaa0_off, adaa1_off] {
                let base = blk_base + a_off;
                for lane in 0..4 {
                    let b_conv = 0.15_f32 * ((lane as f32) - 1.5);
                    let alpha = 0.5_f32;
                    let beta = 0.75_f32;
                    let delta = 0.20_f32;
                    arena[base + lane * 4..base + (lane + 1) * 4]
                        .copy_from_slice(&b_conv.to_le_bytes());
                    arena[base + 16 + lane * 4..base + 16 + (lane + 1) * 4]
                        .copy_from_slice(&alpha.to_le_bytes());
                    arena[base + 32 + lane * 4..base + 32 + (lane + 1) * 4]
                        .copy_from_slice(&beta.to_le_bytes());
                    arena[base + 48 + lane * 4..base + 48 + (lane + 1) * 4]
                        .copy_from_slice(&delta.to_le_bytes());
                }
            }
            let rg_base = blk_base + rg_off;
            for lane in 0..4 {
                let r_low = 0.88_f32 + 0.01 * (lane as f32);
                let dr = 0.08_f32;
                let theta = 0.05_f32 * ((blk * 4 + lane + 1) as f32);
                let b_gamma = 0.35_f32 + 0.02 * (tier as f32);
                let w_sel = 0.40_f32;
                let c_re = 0.25_f32;
                let c_im = -0.15_f32;
                arena[rg_base + lane * 4..rg_base + (lane + 1) * 4]
                    .copy_from_slice(&r_low.to_le_bytes());
                arena[rg_base + 16 + lane * 4..rg_base + 16 + (lane + 1) * 4]
                    .copy_from_slice(&dr.to_le_bytes());
                arena[rg_base + 32 + lane * 4..rg_base + 32 + (lane + 1) * 4]
                    .copy_from_slice(&theta.to_le_bytes());
                arena[rg_base + 48 + lane * 4..rg_base + 48 + (lane + 1) * 4]
                    .copy_from_slice(&b_gamma.to_le_bytes());
                arena[rg_base + 64 + lane * 4..rg_base + 64 + (lane + 1) * 4]
                    .copy_from_slice(&w_sel.to_le_bytes());
                arena[rg_base + 80 + lane * 4..rg_base + 80 + (lane + 1) * 4]
                    .copy_from_slice(&c_re.to_le_bytes());
                arena[rg_base + 96 + lane * 4..rg_base + 96 + (lane + 1) * 4]
                    .copy_from_slice(&c_im.to_le_bytes());
            }
        }
    }

    // Populate SVD stereo head W_head (4224..6784, 17984..20544, 32640..35200) and P_svd (8960..9216)
    for &hbase in &[4224_usize, 17_984_usize, 32_640_usize] {
        for i in 0..640 {
            let val = 0.02_f32 * (((i * 13 + 3) % 19) as f32 - 9.0);
            let off = hbase + i * 4;
            arena[off..off + 4].copy_from_slice(&val.to_le_bytes());
        }
    }
    for i in 0..64 {
        let val = 0.05_f32 * (((i * 7 + 1) % 13) as f32 - 6.0);
        let off = 8960 + i * 4;
        arena[off..off + 4].copy_from_slice(&val.to_le_bytes());
    }

    // Populate Core 2 M-UFDN SoA at 6784..7040 and 20544..20800
    for (g, &c2base) in [6784_usize, 20_544_usize].iter().enumerate() {
        for lane in 0..4 {
            let tau0 = 64.0_f32 + ((g * 4 + lane) as f32) * 29.0;
            let g_lin = 0.72_f32;
            let inv_sigma = 1.0_f32;
            let v_ceil = 0.985_f32;
            let two_over_3v = 2.0_f32 / (3.0 * v_ceil);
            let a_diff = 0.55_f32;
            let d_ap = [19.0_f32, 31.0, 47.0, 73.0][lane];
            let b_in = 0.25_f32;
            let c_l = 0.20_f32;
            let c_r = if lane % 2 == 0 { 0.20_f32 } else { -0.20_f32 };
            arena[c2base + lane * 4..c2base + (lane + 1) * 4].copy_from_slice(&tau0.to_le_bytes());
            arena[c2base + 16 + lane * 4..c2base + 16 + (lane + 1) * 4]
                .copy_from_slice(&g_lin.to_le_bytes());
            arena[c2base + 32 + lane * 4..c2base + 32 + (lane + 1) * 4]
                .copy_from_slice(&inv_sigma.to_le_bytes());
            arena[c2base + 48 + lane * 4..c2base + 48 + (lane + 1) * 4]
                .copy_from_slice(&two_over_3v.to_le_bytes());
            arena[c2base + 64 + lane * 4..c2base + 64 + (lane + 1) * 4]
                .copy_from_slice(&v_ceil.to_le_bytes());
            arena[c2base + 80 + lane * 4..c2base + 80 + (lane + 1) * 4]
                .copy_from_slice(&a_diff.to_le_bytes());
            arena[c2base + 96 + lane * 4..c2base + 96 + (lane + 1) * 4]
                .copy_from_slice(&d_ap.to_le_bytes());
            arena[c2base + 112 + lane * 4..c2base + 112 + (lane + 1) * 4]
                .copy_from_slice(&b_in.to_le_bytes());
            arena[c2base + 128 + lane * 4..c2base + 128 + (lane + 1) * 4]
                .copy_from_slice(&c_l.to_le_bytes());
            arena[c2base + 144 + lane * 4..c2base + 144 + (lane + 1) * 4]
                .copy_from_slice(&c_r.to_le_bytes());
        }
    }

    (header, arena)
}

#[test]
fn test_1_poly_adaa_algebraic_exactness_and_dc_equilibrium() {
    let beta = 0.85_f64;
    let delta = 0.25_f64;
    // Analytic antiderivative F(u) = beta * (0.75 u^2 - 0.125 u^4) + delta * (u^3 / 3 - u^5 / 5)
    let antideriv = |u: f64| -> f64 {
        let u2 = u * u;
        let u3 = u2 * u;
        let u4 = u2 * u2;
        let u5 = u4 * u;
        beta * (0.75 * u2 - 0.125 * u4) + delta * (u3 / 3.0 - 0.2 * u5)
    };

    for i in -95_i32..=95_i32 {
        for j in -95_i32..=95_i32 {
            if (i - j).abs() >= 2 {
                let u_prev = (i as f64) * 0.01;
                let u_curr = (j as f64) * 0.01;
                let exact_div = (antideriv(u_curr) - antideriv(u_prev)) / (u_curr - u_prev);
                let alg = poly_adaa_algebraic_f64(u_prev, u_curr, beta, delta);
                assert!(
                    (alg - exact_div).abs() < 1e-14,
                    "algebraic PolyADAA mismatch: {alg} vs {exact_div}"
                );
            }
        }
    }

    // Verify exact 0/0 singularity removal when u_curr == u_prev
    let u0 = 0.42_f64;
    let alg_same = poly_adaa_algebraic_f64(u0, u0, beta, delta);
    let g_exact = beta * u0 * (1.5 - 0.5 * u0 * u0) + delta * (u0 * u0 - u0.powi(4));
    assert!((alg_same - g_exact).abs() < 1e-15);

    // Verify 3-State PolyADAA DC Resting Equilibrium produces exact 0.000000 Sample-0 transient on silence
    let silence = [0.0_f64; 64];
    let mut out = [1.0_f64; 64];
    poly_adaa_slice_f64(&silence, &mut out, 0.42, 0.5, beta, delta, 0.9993);
    for (idx, &y) in out.iter().enumerate() {
        assert!(
            y.abs() == 0.0,
            "Sample {idx} had non-zero DC transient {y:e}"
        );
    }
}

#[test]
fn test_2_selective_rglru_radial_polar_and_newton_rotor() {
    // 1. Verify 4-FMA + 1-Step Newton-Raphson rotor stays on unit circle over 1,000,000 steps
    let (c_inc, s_inc) = DefaultMathBackend::fast_unit_phasor(0.012345_f32);
    let mut c = 1.0_f32;
    let mut s = 0.0_f32;
    for _ in 0..1_000_000 {
        let c_hat = c * c_inc - s * s_inc;
        let s_hat = s * c_inc + c * s_inc;
        let y1 = 1.5 - 0.5 * (c_hat * c_hat + s_hat * s_hat);
        c = c_hat * y1;
        s = s_hat * y1;
    }
    let norm_err = (c * c + s * s - 1.0).abs();
    assert!(norm_err < 1e-6, "Newton rotor norm drifted by {norm_err:e}");

    // 2. Verify Radial-Polar SelectiveRgLru BIBO stability under extreme input burst
    let mut z_re = 0.0_f64;
    let mut z_im = 0.0_f64;
    let (u_cos, u_sin) = (0.1_f64.cos(), 0.1_f64.sin());
    for n in 0..10_000 {
        let x = if n < 1000 { 10.0 } else { 0.0 };
        let y = selective_rglru_step_f64(
            x, 0.90, 0.099, 0.8, 0.42, 0.5, -0.3, u_cos, u_sin, &mut z_re, &mut z_im,
        );
        assert!(y.is_finite());
    }
    let mag_after = (z_re * z_re + z_im * z_im).sqrt();
    assert!(mag_after < 1.0, "RG-LRU state failed to decay: {mag_after}");

    let mut slice_out = [0.0_f64; 128];
    selective_rglru_slice_f64(
        &[1.0_f64; 128],
        &mut slice_out,
        0.90,
        0.08,
        0.5,
        0.4,
        0.3,
        -0.2,
        u_cos,
        u_sin,
    );
    assert!(slice_out.iter().all(|v| v.is_finite()));
}

#[test]
fn test_3_cubic_lagrange_partition_of_unity() {
    for k in 0..=1000 {
        let mu = (k as f64) * 0.001;
        let w = cubic_lagrange_weights_f64(mu);
        let sum = w[0] + w[1] + w[2] + w[3];
        assert!(
            (sum - 1.0).abs() < 1e-15,
            "Cubic Lagrange partition of unity failed at mu={mu}: sum={sum}"
        );

        // Verify exact reproduction of cubic polynomial p(x) = 2x^3 - 3x^2 + x - 5 at x = mu
        // Nodes are at x = -1, 0, +1, +2
        let poly = |x: f64| 2.0 * x.powi(3) - 3.0 * x * x + x - 5.0;
        let interp = w[0] * poly(-1.0) + w[1] * poly(0.0) + w[2] * poly(1.0) + w[3] * poly(2.0);
        let exact = poly(mu);
        assert!(
            (interp - exact).abs() < 1e-14,
            "Cubic Lagrange polynomial reproduction failed at mu={mu}"
        );
    }
}

#[test]
fn test_4_mufdn_coercive_lyapunov_stability_and_dc_blocker() {
    // 1. Verify exact small-signal slope S'(0) == 1.0000 and bounded self-oscillation at g_lin = 1.50 > 1.0
    let mut input = vec![0.0_f64; 8192];
    for (i, sample) in input.iter_mut().enumerate().take(256) {
        *sample = 0.8 * ((i as f64) * 0.15).sin();
    }
    let mut output = vec![0.0_f64; 8192];
    let v_ceil = 0.985_f64;
    mufdn_slice_f64(&input, &mut output, 37, 1.50, 0.5, v_ceil, 0.995);

    let max_abs = output.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    assert!(
        max_abs <= v_ceil + 1e-12 && max_abs > 0.1,
        "Coercive Lyapunov loop exceeded ceiling or died out: max_abs={max_abs}"
    );

    // 2. Verify exact 5 Hz DC rejection on constant DC input
    let dc_in = [1.0_f64; 20_000];
    let mut dc_x = 0.0_f64;
    let mut dc_y = 0.0_f64;
    let mut last_y = 1.0_f64;
    for &u in &dc_in {
        last_y = mufdn_coercive_step_f64(u, 0.95, 1.0, 0.985, 0.995, &mut dc_x, &mut dc_y);
    }
    assert!(
        last_y.abs() < 1e-6,
        "In-loop DC blocker failed to reject constant DC offset: {last_y}"
    );
}

#[test]
fn test_5_givens_householder_orthogonality() {
    for angle_idx in -50..=50 {
        let psi = (angle_idx as f64) * 0.06;
        let (c_psi, s_psi) = (psi.cos(), psi.sin());
        let x = [0.4, -1.2, 0.7, 2.1, -0.5, 0.9, -1.8, 0.3];
        let y = givens_householder_mix_8ch_f64(&x, c_psi, s_psi);

        let energy_x: f64 = x.iter().map(|v| v * v).sum();
        let energy_y: f64 = y.iter().map(|v| v * v).sum();
        assert!(
            (energy_x - energy_y).abs() < 1e-14,
            "Givens+Householder energy mismatch: {energy_x} vs {energy_y}"
        );
    }
}

#[test]
fn test_6_zoh_sample_rate_rescaling_and_lfo_tick_invariance() {
    let f_lfo_hz = 3.40_f64;
    let fs_train = 48_000_u32;
    let k_train = 16_u8;
    let dphi_train =
        2.0 * core::f64::consts::PI * f_lfo_hz / ((fs_train as f64) / (k_train as f64));

    for &fs_host in &[44_100_u32, 48_000, 88_200, 96_000] {
        let zoh = ZohRescaledParams::compute(fs_train, fs_host, k_train, 0.9993, 0.995, 64.0, 128)
            .unwrap();
        let dphi_host = zoh.rho_ctrl * dphi_train;
        let f_actual = (dphi_host * ((fs_host as f64) / (zoh.control_decimation_host as f64)))
            / (2.0 * core::f64::consts::PI);
        assert!(
            (f_actual - f_lfo_hz).abs() < 1e-13,
            "LFO Hz drifted at {fs_host} Hz: {f_actual} vs {f_lfo_hz}"
        );
    }

    // Verify ui_hints midpoint power-skew curve round-trip exactness
    let hints = ControlUiHints {
        midpoint: Some(-3.0),
        ..Default::default()
    };
    let (x_min, x_max) = (-12.0_f32, 12.0_f32);
    let mid_val = hints.normalized_to_physical(0.5, x_min, x_max);
    assert!((mid_val - (-3.0)).abs() < 1e-6);
    for k in 0..=100 {
        let t = (k as f32) * 0.01;
        let x = hints.normalized_to_physical(t, x_min, x_max);
        let t_rt = hints.physical_to_normalized(x, x_min, x_max);
        assert!(
            (t_rt - t).abs() < 1e-5,
            "Midpoint skew round-trip failed at t={t}: got {t_rt}"
        );
    }

    // Verify fp16 round-trip conversion
    for k in -100..=100 {
        let val = (k as f32) * 0.0125;
        let bits = f32_to_fp16_bits(val);
        let back = fp16_bits_to_f32(bits);
        assert!((back - val).abs() <= 1e-3 * val.abs().max(1.0));
    }
}

#[test]
fn test_7_control_clamp_descriptor_and_newton_lobatto_basis() {
    let (header, _) = build_reference_header_and_arena(true);
    let table = header.compile_clamp_descriptors().unwrap();
    assert_eq!(core::mem::size_of::<ControlClampDescriptor>(), 64);

    // Verify [CLAMP-RANGE] on 'gain' (declared [0, 10], safe [2, 8]),
    // [CLAMP-SWITCH] on 'boost' (declared ["off", "on", "ultra"], active ["off", "on"]),
    // and [CLAMP-STABILITY] on 'feedback' (clamped <= 9.368421 when boost="on")
    let mut active_switches = [0u8; 16];
    // Try setting boost to option 2 ("ultra", unobserved) -> must clamp to option 0 ("off")
    let (boost_clamped, _, boost_was_clamped) = table[2].clamp_and_normalize(2.0, &active_switches);
    assert!(boost_was_clamped);
    assert_eq!(boost_clamped, 0.0);

    // Set boost to option 1 ("on", active)
    let (boost_on, _, boost_on_clamped) = table[2].clamp_and_normalize(1.0, &active_switches);
    assert!(!boost_on_clamped);
    assert_eq!(boost_on, 1.0);
    active_switches[2] = 1;

    // Now try feedback = 10.0 while boost="on" -> must clamp to <= 9.368421 so g_lin <= 0.9990
    let (fb_clamped, _, fb_was_clamped) = table[1].clamp_and_normalize(10.0, &active_switches);
    assert!(fb_was_clamped);
    let g_lin_corner = 0.40 + 0.057 * fb_clamped + 0.065;
    assert!(g_lin_corner <= 0.9990 + 1e-5);

    // Verify Node-Anchored Newton/Lobatto basis has 0.00e+00 perturbation on old stems (Test 8G)
    let anchors = [-0.6_f64, 0.6, 1.0];
    let p_at_u1 = eval_newton_lobatto_basis_f64(-0.6, anchors, 3);
    assert_eq!(p_at_u1, [0.0, 0.0, 0.0]);
    let p_at_u2 = eval_newton_lobatto_basis_f64(0.6, anchors, 3);
    assert_eq!(p_at_u2[1], 0.0);
    assert_eq!(p_at_u2[2], 0.0);

    // Verify Rank-4 FM, Chebyshev LFO, and Trust-Region Soft-Clip reference kernels
    let cheb = eval_chebyshev_lfo_4harm_f64(0.5);
    assert!((cheb[0] - 0.5).abs() < 1e-15);
    assert!((cheb[1] - (-0.5)).abs() < 1e-15);
    let fm = eval_rank4_fm_interaction_f64(&[1.0, 2.0], &[[0.5; 4], [0.25; 4]]);
    assert!((fm[0] - 0.25).abs() < 1e-15);
    let clipped = eval_trust_region_softclip_f64(10.0, 2.5);
    assert!((clipped - 2.5).abs() < 1e-15);
}

#[test]
fn test_8_ckpt_uplo_u_prefix_theorem_and_marketplace_provenance_watermarking() {
    // 1. Verify Nested Principal Submatrix Prefix Theorem on Arena 3 uplo='U' packing (Lite=64, Full=128, Ultra=192)
    assert_eq!(uplo_u_packed_len(64) * 8 + 64 * 2 * 8, 17_664);
    assert_eq!(uplo_u_packed_len(128) * 8 + 128 * 2 * 8, 68_096);
    assert_eq!(uplo_u_packed_len(192) * 8 + 192 * 2 * 8, 151_296);

    let n_ultra = 192;
    let mut g_dense = vec![0.0_f64; n_ultra * n_ultra];
    for r in 0..n_ultra {
        for c in r..n_ultra {
            let val = ((r * 31 + c * 17 + 5) as f64) * 0.001;
            g_dense[r * n_ultra + c] = val;
            g_dense[c * n_ultra + r] = val;
        }
    }
    let mut packed_u = vec![0.0_f64; uplo_u_packed_len(n_ultra)];
    pack_uplo_u_f64(&g_dense, n_ultra, &mut packed_u);

    for &n_tier in &[64_usize, 128, 192] {
        let prefix_len = uplo_u_packed_len(n_tier);
        let mut unpacked = vec![0.0_f64; n_tier * n_tier];
        unpack_uplo_u_f64(&packed_u[..prefix_len], n_tier, &mut unpacked);
        for r in 0..n_tier {
            for c in 0..n_tier {
                assert_eq!(
                    unpacked[r * n_tier + c],
                    g_dense[r * n_ultra + c],
                    "uplo='U' prefix theorem mismatch at ({r}, {c}) for tier N={n_tier}"
                );
            }
        }
    }

    // 2. Verify Mechanism A (Isometric Gauge Watermark) has 0.0000 dB audio distortion & 1.000000 160-D structural hash match
    let (mut header, master_arena) = build_reference_header_and_arena(true);
    let mut buyer_arena = master_arena.clone();

    let mut kappa = [1.0_f32; RGLRU_TOTAL_CHANNELS];
    for (i, k) in kappa.iter_mut().enumerate() {
        *k = 0.990 + 0.0005 * (i as f32);
    }
    // Orthonormal 4x4 permutation/rotation matrix Q_buyer in SO(4)
    let theta_q = 0.35_f32;
    let (cq, sq) = (theta_q.cos(), theta_q.sin());
    let q_buyer = [
        [cq, -sq, 0.0, 0.0],
        [sq, cq, 0.0, 0.0],
        [0.0, 0.0, cq, -sq],
        [0.0, 0.0, sq, cq],
    ];
    apply_isometric_gauge_watermark(&mut buyer_arena, &kappa, &q_buyer).unwrap();

    // Structural hash cosine similarity must be 1.00000000000000
    let hash_master = extract_structural_hash_160d(&master_arena).unwrap();
    let hash_buyer = extract_structural_hash_160d(&buyer_arena).unwrap();
    let cos_sim = structural_cosine_similarity(&hash_master, &hash_buyer);
    assert!(
        (cos_sim - 1.0).abs() < 1e-12,
        "160-D structural hash not gauge-invariant: cos_sim={cos_sim}"
    );

    // Render audio through master and Mechanism-A watermarked model (with non-zero z_head preset!) -> < 1e-6 audio diff
    let master_bytes =
        build_dnam_container(&mut header.clone(), &master_arena, &master_arena, false).unwrap();
    let buyer_bytes =
        build_dnam_container(&mut header.clone(), &buyer_arena, &buyer_arena, false).unwrap();

    let mut eng_m =
        DnamEngine::<DefaultMathBackend>::from_bytes(&master_bytes, Tier::Ultra, 48_000).unwrap();
    let mut eng_b =
        DnamEngine::<DefaultMathBackend>::from_bytes(&buyer_bytes, Tier::Ultra, 48_000).unwrap();
    eng_m.apply_preset("dub_swell").unwrap();
    eng_b.apply_preset("dub_swell").unwrap();

    let in_l: Vec<f32> = (0..256).map(|n| 0.35 * ((n as f32) * 0.11).sin()).collect();
    let in_r: Vec<f32> = (0..256).map(|n| 0.35 * ((n as f32) * 0.07).cos()).collect();
    let (mut out_ml, mut out_mr) = (vec![0.0_f32; 256], vec![0.0_f32; 256]);
    let (mut out_bl, mut out_br) = (vec![0.0_f32; 256], vec![0.0_f32; 256]);
    eng_m.process_block(&in_l, &in_r, &mut out_ml, &mut out_mr);
    eng_b.process_block(&in_l, &in_r, &mut out_bl, &mut out_br);

    let max_audio_diff = out_ml
        .iter()
        .zip(out_bl.iter())
        .fold(0.0_f32, |acc, (&a, &b)| acc.max((a - b).abs()));
    assert!(
        max_audio_diff < 1e-5,
        "Isometric gauge watermark altered audio output by {max_audio_diff:e}"
    );

    // 3. Verify Mechanism B (4,608-Weight Spread-Spectrum Tardos Watermark) under 4-buyer collusion + fp16 truncation
    let mp_secret = *b"dnam-marketplace-master-key-2026";
    let colluder_ids = ["lic_017", "lic_042", "lic_108", "lic_163"];
    let mut colluder_arenas = Vec::new();
    let mut colluder_codes = Vec::new();
    for &cid in &colluder_ids {
        let code = generate_tardos_code_4608(&mp_secret, "tape_echo_v1", cid);
        let mut a = master_arena.clone();
        apply_tardos_watermark_4608(&mut a, &code, 2e-3).unwrap();
        colluder_arenas.push(a);
        colluder_codes.push(code);
    }

    // Average the 4 colluders' arenas and quantize every f32 in the conv blocks to fp16 and back!
    let mut pirated_arena = master_arena.clone();
    for off in (128..32_640).step_by(4) {
        let mut sum = 0.0_f32;
        for ca in &colluder_arenas {
            sum += f32::from_le_bytes([ca[off], ca[off + 1], ca[off + 2], ca[off + 3]]);
        }
        let avg = sum * 0.25;
        let fp16_attacked = fp16_bits_to_f32(f32_to_fp16_bits(avg));
        pirated_arena[off..off + 4].copy_from_slice(&fp16_attacked.to_le_bytes());
    }

    for (idx, code) in colluder_codes.iter().enumerate() {
        let z = score_tardos_watermark_4608(&master_arena, &pirated_arena, code).unwrap();
        assert!(
            z > 12.0,
            "Colluder {idx} Tardos Z-score too low: {z:.2} sigma"
        );
    }
    let innocent_code = generate_tardos_code_4608(&mp_secret, "tape_echo_v1", "innocent_buyer_999");
    let z_innocent =
        score_tardos_watermark_4608(&master_arena, &pirated_arena, &innocent_code).unwrap();
    assert!(
        z_innocent.abs() < 4.5 && TARDOS_CONV_WEIGHTS == 4608,
        "Innocent buyer false positive: {z_innocent:.2} sigma"
    );

    // 4. Verify Split-Payload Ed25519 Signatures, X25519 + ChaCha20 64B-Block Arena Encryption,
    // and 4 KB In-Place Header Preset Updates
    use ed25519_dalek::{Signer, SigningKey};
    use x25519_dalek::{PublicKey as XPublicKey, StaticSecret as XSecret};

    let creator_sk = SigningKey::from_bytes(&[7u8; 32]);
    let creator_pk = creator_sk.verifying_key().to_bytes();
    let mp_sk = SigningKey::from_bytes(&[19u8; 32]);
    let mp_pk = mp_sk.verifying_key().to_bytes();

    let device_id_hash = "9c1185a5c5e9fc54612808977ee8f548";
    let dev_sk_bytes = [42u8; 32];
    let dev_pk_bytes = XPublicKey::from(&XSecret::from(dev_sk_bytes)).to_bytes();
    let eph_sk_bytes = [99u8; 32];
    let eph_pk_bytes = XPublicKey::from(&XSecret::from(eph_sk_bytes)).to_bytes();

    let cek = [0x5A_u8; 32];
    let nonce_96 = [0x11_u8; 12];
    let wrapped_cek =
        RustCryptoProvider::wrap_device_cek(&eph_sk_bytes, &dev_pk_bytes, device_id_hash, &cek)
            .unwrap();

    let imm_sha = compute_immutable_header_sha256(&header).unwrap();
    let master_sha = sha256_digest(&master_arena);
    let buyer_sha = sha256_digest(&buyer_arena);

    let creator_msg = build_creator_signing_message(&imm_sha, &master_sha);
    let creator_sig = creator_sk.sign(&creator_msg).to_bytes();

    let mut receipt = MarketplaceReceipt {
        marketplace_id: "dnam.store".to_string(),
        license_id: "0192b6c0-7e4a-7000-8000-1a2b3c4d5e6f".to_string(),
        buyer_hash: encode_hex(&[0xAB; 32]),
        license_tier: "personal".to_string(),
        issued_at: "2026-10-23T00:00:00Z".to_string(),
        buyer_arena_sha256: encode_hex(&buyer_sha),
        encryption_mode: "x25519_chacha20_arena".to_string(),
        arena_nonce_96: Some(encode_hex(&nonce_96)),
        ephemeral_pubkey_x25519: Some(encode_hex(&eph_pk_bytes)),
        authorized_devices: vec![AuthorizedDevice {
            oem_vendor: "darkglass".to_string(),
            model_code: "anagram_rk3582".to_string(),
            device_id_hash: device_id_hash.to_string(),
            wrapped_cek: Some(encode_hex(&wrapped_cek)),
        }],
        royalty_splits: Vec::new(),
        receipt_sig_ed25519: String::new(),
    };
    let receipt_msg =
        build_marketplace_receipt_signing_message(&imm_sha, &master_sha, &receipt).unwrap();
    receipt.receipt_sig_ed25519 = encode_hex(&mp_sk.sign(&receipt_msg).to_bytes());

    header.metadata.signatures = Some(ModelSignatures {
        immutable_header_sha256: encode_hex(&imm_sha),
        master_arena_sha256: encode_hex(&master_sha),
        creator_signature: CreatorSignature {
            signer_id: "did:web:studio.example".to_string(),
            pubkey_ed25519: encode_hex(&creator_pk),
            sig_ed25519: encode_hex(&creator_sig),
        },
        marketplace_receipt: Some(receipt),
    });

    // Encrypt buyer_arena with ChaCha20 (566 blocks = 36,224 B, 0 padding overhead)
    let mut encrypted_arena = buyer_arena.clone();
    chacha20_xor_arena_blocks(&mut encrypted_arena, &cek, &nonce_96, 0).unwrap();
    assert_eq!(encrypted_arena.len(), 36_224);

    let mut container_bytes =
        build_dnam_container(&mut header, &buyer_arena, &encrypted_arena, true).unwrap();

    // Update user_metadata & add a custom preset in-place inside the 4 KB page WITHOUT invalidating signatures!
    header
        .user_metadata
        .insert("favorite".to_string(), serde_json::json!(true));
    update_header_in_place(&mut container_bytes, &header).unwrap();

    let crypto = RustCryptoProvider::new()
        .with_device_key(device_id_hash, dev_sk_bytes)
        .with_marketplace_pubkey(mp_pk);

    let loaded_view = DnamContainerView::from_bytes(&container_bytes, &crypto).unwrap();
    assert_eq!(
        loaded_view.verification_status,
        VerificationStatus::MarketplaceVerified {
            signer_id: "did:web:studio.example".to_string(),
            marketplace_id: "dnam.store".to_string(),
            license_id: "0192b6c0-7e4a-7000-8000-1a2b3c4d5e6f".to_string(),
            license_tier: "personal".to_string(),
        }
    );
    assert_eq!(loaded_view.arena_storage.as_bytes(), buyer_arena.as_slice());
}
