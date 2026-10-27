//! End-to-End Marketplace Provenance, Encrypted Arena, Isometric Gauge Audio Equivalence,
//! and In-Place Preset Update Integration Suite (`tests/marketplace_integration.rs`).

use std::collections::BTreeMap;

use dnam_engine::container::{
    ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_ULTRA_END, DnamContainerView,
    build_dnam_container, update_header_in_place,
};
use dnam_engine::math::{DefaultMathBackend, f32_to_fp16_bits, fp16_bits_to_f32};
use dnam_engine::provenance::{
    DNSL_MAGIC, RGLRU_TOTAL_CHANNELS, RustCryptoProvider, TARDOS_CONV_WEIGHTS, VerificationStatus,
    apply_isometric_gauge_watermark, apply_tardos_watermark_4608, build_creator_signing_message,
    build_marketplace_receipt_signing_message, chacha20_xor_arena_blocks,
    compute_immutable_header_sha256, encode_hex, extract_structural_hash_160d,
    generate_tardos_code_4608, score_tardos_watermark_4608, seal_dnam_container, sha256_digest,
    structural_cosine_similarity, unseal_dnam_container, verify_oem_device_certificate,
};
use dnam_engine::schema::{
    AuthorizedDevice, ControlBusConfig, ControlKind, ControlSchemaDef, ControlUiHints, Core1Config,
    Core2Config, CreatorSignature, DatasetProvenance, DistributionPolicy, DnamJsonHeader,
    DnamProtectionProfile, IoConfig, LevelingConfig, MarketplaceReceipt, ModelCapabilities,
    ModelMetadata, ModelSignatures, PresetDef, SlimmableTiersConfig, TierProfile,
};
use dnam_engine::{DnamEngine, DynamicDnamEngine, Tier};

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
            calibration: None,
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
            calibration: None,
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
            calibration: None,
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
            distribution_policy: None,
            signatures: None,
        },
        user_metadata: BTreeMap::new(),
        io_config: IoConfig {
            inputs: 2,
            output_mode: "stereo".to_string(),
            latency_samples: 0,
            tail_time_ms: if core2_active { 1250.0 } else { 0.0 },
            capabilities: ModelCapabilities::default(),
            leveling: LevelingConfig::default(),
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
    let compiled = header.compile_clamp_descriptors().unwrap();
    for (p, desc) in compiled.iter().enumerate() {
        let off = 7040 + p * 64;
        arena[off..off + 64].copy_from_slice(&desc.to_bytes());
    }

    for i in 0..4 {
        let off = (i * 4 + (i % 2)) * 4;
        arena[off..off + 4].copy_from_slice(&0.5_f32.to_le_bytes());
    }
    for t in 0..3 {
        arena[104 + t * 4..108 + t * 4].copy_from_slice(&1.0_f32.to_le_bytes());
    }
    arena[116..120].copy_from_slice(&0.25_f32.to_le_bytes());
    arena[120..124].copy_from_slice(&0.75_f32.to_le_bytes());
    arena[124..128].copy_from_slice(&0.9993_f32.to_le_bytes());

    for tier in 0..3 {
        for blk in 0..4 {
            let (blk_base, adaa0_off, adaa1_off, rg_off) = match tier {
                0 => (128 + blk * 1024, 384, 832, 896),
                1 => (10_816 + blk * 1792, 768, 1600, 1664),
                _ => (22_400 + blk * 2560, 1152, 2368, 2432),
            };
            for w in 0..96 {
                let val = 0.01_f32 * (((blk * 17 + w + tier * 7) % 11) as f32 - 5.0);
                let off = blk_base + w * 4;
                arena[off..off + 4].copy_from_slice(&val.to_le_bytes());
            }
            for &a_off in &[adaa0_off, adaa1_off] {
                let base = blk_base + a_off;
                for lane in 0..4 {
                    let b_conv = 0.15_f32 * ((lane as f32) - 1.5);
                    arena[base + lane * 4..base + (lane + 1) * 4]
                        .copy_from_slice(&b_conv.to_le_bytes());
                    arena[base + 16 + lane * 4..base + 16 + (lane + 1) * 4]
                        .copy_from_slice(&0.5_f32.to_le_bytes());
                    arena[base + 32 + lane * 4..base + 32 + (lane + 1) * 4]
                        .copy_from_slice(&0.75_f32.to_le_bytes());
                    arena[base + 48 + lane * 4..base + 48 + (lane + 1) * 4]
                        .copy_from_slice(&0.20_f32.to_le_bytes());
                }
            }
            let rg_base = blk_base + rg_off;
            for lane in 0..4 {
                let r_low = 0.88_f32 + 0.01 * (lane as f32);
                let dr = 0.08_f32;
                let theta = 0.05_f32 * ((blk * 4 + lane + 1) as f32);
                let b_gamma = 0.35_f32 + 0.02 * (tier as f32);
                arena[rg_base + lane * 4..rg_base + (lane + 1) * 4]
                    .copy_from_slice(&r_low.to_le_bytes());
                arena[rg_base + 16 + lane * 4..rg_base + 16 + (lane + 1) * 4]
                    .copy_from_slice(&dr.to_le_bytes());
                arena[rg_base + 32 + lane * 4..rg_base + 32 + (lane + 1) * 4]
                    .copy_from_slice(&theta.to_le_bytes());
                arena[rg_base + 48 + lane * 4..rg_base + 48 + (lane + 1) * 4]
                    .copy_from_slice(&b_gamma.to_le_bytes());
                arena[rg_base + 64 + lane * 4..rg_base + 64 + (lane + 1) * 4]
                    .copy_from_slice(&0.40_f32.to_le_bytes());
                arena[rg_base + 80 + lane * 4..rg_base + 80 + (lane + 1) * 4]
                    .copy_from_slice(&0.25_f32.to_le_bytes());
                arena[rg_base + 96 + lane * 4..rg_base + 96 + (lane + 1) * 4]
                    .copy_from_slice(&(-0.15_f32).to_le_bytes());
            }
        }
    }

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

    for (g, &c2base) in [6784_usize, 20_544_usize].iter().enumerate() {
        for lane in 0..4 {
            let tau0 = 64.0_f32 + ((g * 4 + lane) as f32) * 29.0;
            let v_ceil = 0.985_f32;
            let two_over_3v = 2.0_f32 / (3.0 * v_ceil);
            let d_ap = [19.0_f32, 31.0, 47.0, 73.0][lane];
            let c_r = if lane % 2 == 0 { 0.20_f32 } else { -0.20_f32 };
            arena[c2base + lane * 4..c2base + (lane + 1) * 4].copy_from_slice(&tau0.to_le_bytes());
            arena[c2base + 16 + lane * 4..c2base + 16 + (lane + 1) * 4]
                .copy_from_slice(&0.72_f32.to_le_bytes());
            arena[c2base + 32 + lane * 4..c2base + 32 + (lane + 1) * 4]
                .copy_from_slice(&1.0_f32.to_le_bytes());
            arena[c2base + 48 + lane * 4..c2base + 48 + (lane + 1) * 4]
                .copy_from_slice(&two_over_3v.to_le_bytes());
            arena[c2base + 64 + lane * 4..c2base + 64 + (lane + 1) * 4]
                .copy_from_slice(&v_ceil.to_le_bytes());
            arena[c2base + 80 + lane * 4..c2base + 80 + (lane + 1) * 4]
                .copy_from_slice(&0.55_f32.to_le_bytes());
            arena[c2base + 96 + lane * 4..c2base + 96 + (lane + 1) * 4]
                .copy_from_slice(&d_ap.to_le_bytes());
            arena[c2base + 112 + lane * 4..c2base + 112 + (lane + 1) * 4]
                .copy_from_slice(&0.25_f32.to_le_bytes());
            arena[c2base + 128 + lane * 4..c2base + 128 + (lane + 1) * 4]
                .copy_from_slice(&0.20_f32.to_le_bytes());
            arena[c2base + 144 + lane * 4..c2base + 144 + (lane + 1) * 4]
                .copy_from_slice(&c_r.to_le_bytes());
        }
    }

    (header, arena)
}

#[test]
fn test_marketplace_provenance_enclave_encryption_and_gauge_audio_equivalence() {
    let (mut header, master_arena) = build_reference_header_and_arena(true);
    let mut buyer_arena = master_arena.clone();

    let mut kappa = [1.0_f32; RGLRU_TOTAL_CHANNELS];
    for (i, k) in kappa.iter_mut().enumerate() {
        *k = 0.990 + 0.0005 * (i as f32);
    }
    let theta_q = 0.35_f32;
    let (cq, sq) = (theta_q.cos(), theta_q.sin());
    let q_buyer = [
        [cq, -sq, 0.0, 0.0],
        [sq, cq, 0.0, 0.0],
        [0.0, 0.0, cq, -sq],
        [0.0, 0.0, sq, cq],
    ];
    apply_isometric_gauge_watermark(&mut buyer_arena, &kappa, &q_buyer).unwrap();

    // 1. Structural hash cosine similarity must be 1.00000000000000
    let hash_master = extract_structural_hash_160d(&master_arena).unwrap();
    let hash_buyer = extract_structural_hash_160d(&buyer_arena).unwrap();
    let cos_sim = structural_cosine_similarity(&hash_master, &hash_buyer);
    assert!(
        (cos_sim - 1.0).abs() < 1e-12,
        "160-D structural hash not gauge-invariant: cos_sim={cos_sim}"
    );

    // 2. Render audio through master and Mechanism-A watermarked model (with non-zero z_head preset!) -> < 1e-5 diff
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
        loaded_view.protection_profile,
        DnamProtectionProfile::DeviceAllowlisted
    );
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

#[test]
fn test_dnam_seal_dnsl_envelope_oem_certificate_and_distribution_policy() {
    let (mut header, master_arena) = build_reference_header_and_arena(false);

    // 1. Configure Creator-Signed DistributionPolicy restricting model to EnclaveSealed on Darkglass Anagram
    header.metadata.distribution_policy = Some(DistributionPolicy {
        allowed_protection_profiles: vec![DnamProtectionProfile::EnclaveSealed],
        allowed_oem_vendors: vec!["darkglass:anagram_rk3582".to_string()],
    });

    let creator_seed = [21u8; 32];
    let creator_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&creator_seed);
    let mp_seed = [33u8; 32];
    let mp_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&mp_seed);

    let oem_darkglass_seed = [55u8; 32];
    let oem_darkglass_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&oem_darkglass_seed);
    let oem_other_seed = [66u8; 32];
    let oem_other_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&oem_other_seed);

    let device_id_hash = "9c1185a5c5e9fc54612808977ee8f548";
    let dev_secret_x25519 = [44u8; 32];
    let dev_pubkey_x25519 = RustCryptoProvider::x25519_pubkey_from_secret(&dev_secret_x25519);

    let cert_anagram = RustCryptoProvider::sign_oem_device_certificate(
        &oem_darkglass_seed,
        "darkglass",
        "anagram_rk3582",
        device_id_hash,
        &dev_pubkey_x25519,
    )
    .unwrap();
    let verified_x25519 =
        verify_oem_device_certificate(&cert_anagram, &oem_darkglass_pk, &RustCryptoProvider::new())
            .unwrap();
    assert_eq!(verified_x25519, dev_pubkey_x25519);

    // Wrong OEM root key must fail certificate verification
    assert!(
        verify_oem_device_certificate(&cert_anagram, &oem_other_pk, &RustCryptoProvider::new())
            .is_err()
    );

    // 2. Sign inner .dnam with creator signature + marketplace receipt (ENCRYPTED_ARENA = 1)
    let eph_inner_sk = [77u8; 32];
    let eph_inner_pk = RustCryptoProvider::x25519_pubkey_from_secret(&eph_inner_sk);
    let cek_arena = [0x3C_u8; 32];
    let nonce_arena = [0x22_u8; 12];
    let wrapped_cek_arena = RustCryptoProvider::wrap_device_cek(
        &eph_inner_sk,
        &dev_pubkey_x25519,
        device_id_hash,
        &cek_arena,
    )
    .unwrap();

    let imm_sha = compute_immutable_header_sha256(&header).unwrap();
    let master_sha = sha256_digest(&master_arena);
    let creator_msg = build_creator_signing_message(&imm_sha, &master_sha);
    let creator_sig = RustCryptoProvider::ed25519_sign(&creator_seed, &creator_msg);

    let mut receipt = MarketplaceReceipt {
        marketplace_id: "dnam.store".to_string(),
        license_id: "0192b6c0-7e4a-7000-8000-998877665544".to_string(),
        buyer_hash: encode_hex(&[0xCD; 32]),
        license_tier: "oem_bundle".to_string(),
        issued_at: "2026-10-27T00:00:00Z".to_string(),
        buyer_arena_sha256: encode_hex(&master_sha),
        encryption_mode: "x25519_chacha20_arena".to_string(),
        arena_nonce_96: Some(encode_hex(&nonce_arena)),
        ephemeral_pubkey_x25519: Some(encode_hex(&eph_inner_pk)),
        authorized_devices: vec![AuthorizedDevice {
            oem_vendor: "darkglass".to_string(),
            model_code: "anagram_rk3582".to_string(),
            device_id_hash: device_id_hash.to_string(),
            wrapped_cek: Some(encode_hex(&wrapped_cek_arena)),
        }],
        royalty_splits: Vec::new(),
        receipt_sig_ed25519: String::new(),
    };
    let receipt_msg =
        build_marketplace_receipt_signing_message(&imm_sha, &master_sha, &receipt).unwrap();
    receipt.receipt_sig_ed25519 =
        encode_hex(&RustCryptoProvider::ed25519_sign(&mp_seed, &receipt_msg));

    header.metadata.signatures = Some(ModelSignatures {
        immutable_header_sha256: encode_hex(&imm_sha),
        master_arena_sha256: encode_hex(&master_sha),
        creator_signature: CreatorSignature {
            signer_id: "did:web:darkglass.com".to_string(),
            pubkey_ed25519: encode_hex(&creator_pk),
            sig_ed25519: encode_hex(&creator_sig),
        },
        marketplace_receipt: Some(receipt),
    });

    let mut encrypted_arena = master_arena.clone();
    chacha20_xor_arena_blocks(&mut encrypted_arena, &cek_arena, &nonce_arena, 0).unwrap();
    let inner_dnam_bytes =
        build_dnam_container(&mut header, &master_arena, &encrypted_arena, true).unwrap();

    // 3. Loading the unsealed inner .dnam directly must FAIL because DistributionPolicy requires EnclaveSealed!
    let crypto = RustCryptoProvider::new()
        .with_device_key(device_id_hash, dev_secret_x25519)
        .with_marketplace_pubkey(mp_pk);
    assert!(DnamContainerView::from_bytes(&inner_dnam_bytes, &crypto).is_err());

    // 4. Attempting to seal for a different model ("darkglass:other_pedal") must FAIL DistributionPolicy!
    let cert_wrong_model = RustCryptoProvider::sign_oem_device_certificate(
        &oem_darkglass_seed,
        "darkglass",
        "other_pedal",
        device_id_hash,
        &dev_pubkey_x25519,
    )
    .unwrap();
    let trusted_oems = [("darkglass", oem_darkglass_pk), ("other", oem_other_pk)];
    let eph_outer_sk = [88u8; 32];
    let cek_outer = [0xA5_u8; 32];
    let nonce_outer = [0x77_u8; 12];
    assert!(
        seal_dnam_container(
            &inner_dnam_bytes,
            &[cert_wrong_model],
            &trusted_oems,
            &mp_seed,
            &eph_outer_sk,
            &cek_outer,
            &nonce_outer,
        )
        .is_err()
    );

    // 5. Sealing for the authorized "darkglass:anagram_rk3582" certificate succeeds and produces 64B-aligned DNSL
    let sealed_bytes = seal_dnam_container(
        &inner_dnam_bytes,
        &[cert_anagram],
        &trusted_oems,
        &mp_seed,
        &eph_outer_sk,
        &cek_outer,
        &nonce_outer,
    )
    .unwrap();
    assert_eq!(&sealed_bytes[0..4], &DNSL_MAGIC);
    assert_eq!(sealed_bytes.len() % 64, 0);
    assert_eq!(sealed_bytes.len(), 3 * 64 + inner_dnam_bytes.len());

    // 6. Unsealing recovers the exact inner .dnam bytes (where Arena 0..2 is still ENCRYPTED_ARENA = 1)
    let unsealed_inner = unseal_dnam_container(&sealed_bytes, &crypto).unwrap();
    assert_eq!(unsealed_inner, inner_dnam_bytes);

    // 7. DnamContainerView::from_bytes & DynamicDnamEngine load .dnam-seal directly and report EnclaveSealed
    let sealed_view = DnamContainerView::from_bytes(&sealed_bytes, &crypto).unwrap();
    assert_eq!(
        sealed_view.protection_profile,
        DnamProtectionProfile::EnclaveSealed
    );
    assert_eq!(
        sealed_view.arena_storage.as_bytes(),
        master_arena.as_slice()
    );

    let mut dyn_eng = DynamicDnamEngine::from_bytes_owned_with_crypto(
        &sealed_bytes,
        Tier::Ultra,
        48_000,
        &crypto,
    )
    .unwrap();
    assert_eq!(
        dyn_eng.protection_profile(),
        DnamProtectionProfile::EnclaveSealed
    );
    let in_l = [0.25_f32; 64];
    let in_r = [0.25_f32; 64];
    let mut out_l = [0.0_f32; 64];
    let mut out_r = [0.0_f32; 64];
    dyn_eng.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
    assert!(out_l.iter().any(|&v| v.abs() > 1e-6));

    // 8. Wrong device key or tampered .dnam-seal byte must fail unsealing
    let wrong_crypto = RustCryptoProvider::new()
        .with_device_key(device_id_hash, [99u8; 32])
        .with_marketplace_pubkey(mp_pk);
    assert!(DnamContainerView::from_bytes(&sealed_bytes, &wrong_crypto).is_err());

    let mut tampered_seal = sealed_bytes.clone();
    tampered_seal[200] ^= 0x01;
    assert!(DnamContainerView::from_bytes(&tampered_seal, &crypto).is_err());
}
