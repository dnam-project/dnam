//! Marketplace Provenance, Encrypted Arena, Isometric Gauge Audio Equivalence, Tardos Collusion
//! Tracing, and In-Place Preset Update Integration Suite (`tests/marketplace_provenance_and_watermark.rs`).
//!
//! Canonical Reference: `docs/security/provenance_and_enclave_drm.md` & `docs/security/watermarking_and_marketplace.md`.

use std::collections::BTreeMap;

use dnam_engine::container::{
    ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_PRO_END, ARENA_OFFSET_ULTRA_END,
    DnamContainerView, build_dnam_container, strip_container_to_tier, update_header_in_place,
};
use dnam_engine::math::{DefaultMathBackend, f32_to_fp16_bits, fp16_bits_to_f32};
use dnam_engine::provenance::{
    KAUTZ_TOTAL_CHANNELS, RustCryptoProvider, TARDOS_CASCADE_WEIGHTS, VerificationStatus,
    apply_isometric_gauge_watermark, apply_tardos_watermark_7680, build_creator_signing_message,
    build_marketplace_receipt_signing_message, chacha20_xor_arena_blocks,
    compute_immutable_header_sha256, encode_hex, extract_structural_hash_224d,
    generate_tardos_code_7680, score_tardos_watermark_7680, sha256_digest,
    structural_cosine_similarity,
};
use dnam_engine::schema::{
    AuthorizedDevice, ControlBusConfig, ControlKind, ControlSchemaDef, ControlUiHints,
    CreatorSignature, DatasetProvenance, DnamJsonHeader, DnamProtectionProfile, IoConfig,
    LevelingConfig, MarketplaceReceipt, ModelCapabilities, ModelMetadata, ModelSignatures,
    PresetDef, RouterConfig, SlimmableTiersConfig, Stage1PreFxConfig, Stage2CascadeConfig,
    Stage3CabinetKautzConfig, Stage4SpatialTankConfig, TierProfile,
};
use dnam_engine::{DnamEngine, Tier};

fn build_reference_header_and_arena() -> (DnamJsonHeader, Vec<u8>) {
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
        architecture: "DNAM_TORA".to_string(),
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
            tail_time_ms: 1250.0,
            capabilities: ModelCapabilities::default(),
            leveling: LevelingConfig::default(),
        },
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
                validation_esr_db: -39.4,
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
                validation_esr_db: -45.8,
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
                validation_esr_db: -47.63,
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
                validation_esr_db: -50.8,
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
        stage0_router: RouterConfig::default(),
        stage1_pre: Stage1PreFxConfig::default(),
        stage2_cascade: Stage2CascadeConfig::default(),
        stage3_cabinet: Stage3CabinetKautzConfig::default(),
        stage4_tank: Stage4SpatialTankConfig::default(),
        tensor_directory: BTreeMap::new(),
        stems_registry: None,
        audit_report: None,
    };

    let mut arena = vec![0u8; 64_000];
    let compiled = header.compile_clamp_descriptors().unwrap();
    for (p, desc) in compiled.iter().take(8).enumerate() {
        let off = 8_512 + p * 64;
        arena[off..off + 64].copy_from_slice(&desc.to_bytes());
    }
    arena[124..128].copy_from_slice(&0.9993_f32.to_le_bytes());

    let cascade_w_bases = [1_024_usize, 11_264_usize, 22_336_usize, 36_224_usize];
    let cascade_w_counts = [768_usize, 1536_usize, 2304_usize, 3072_usize];
    for tier in 0..4 {
        let cw_base = cascade_w_bases[tier];
        for w in 0..cascade_w_counts[tier] {
            let val = 0.01_f32 * (((w + tier * 7) % 11) as f32 - 5.0);
            let off = cw_base + w * 4;
            arena[off..off + 4].copy_from_slice(&val.to_le_bytes());
        }
        for blk in 0..4 {
            let k_base = dnam_engine::provenance::kautz_soa_byte_offset(tier, blk);
            for lane in 0..4 {
                let r_low = 0.88_f32 + 0.01 * (lane as f32);
                let dr = 0.08_f32;
                let theta = 0.05_f32 * ((blk * 4 + lane + 1) as f32);
                let b_gamma = 0.35_f32 + 0.02 * (tier as f32);
                arena[k_base + lane * 4..k_base + (lane + 1) * 4]
                    .copy_from_slice(&r_low.to_le_bytes());
                arena[k_base + 16 + lane * 4..k_base + 16 + (lane + 1) * 4]
                    .copy_from_slice(&dr.to_le_bytes());
                arena[k_base + 32 + lane * 4..k_base + 32 + (lane + 1) * 4]
                    .copy_from_slice(&theta.to_le_bytes());
                arena[k_base + 48 + lane * 4..k_base + 48 + (lane + 1) * 4]
                    .copy_from_slice(&b_gamma.to_le_bytes());
                arena[k_base + 64 + lane * 4..k_base + 64 + (lane + 1) * 4]
                    .copy_from_slice(&0.25_f32.to_le_bytes());
                arena[k_base + 80 + lane * 4..k_base + 80 + (lane + 1) * 4]
                    .copy_from_slice(&(-0.15_f32).to_le_bytes());
            }
        }
    }

    for &hbase in &[4_928_usize, 18_048_usize, 31_936_usize, 57_600_usize] {
        for i in 0..640 {
            let val = 0.02_f32 * (((i * 13 + 3) % 19) as f32 - 9.0);
            let off = hbase + i * 4;
            arena[off..off + 4].copy_from_slice(&val.to_le_bytes());
        }
    }
    for i in 0..2048 {
        let val = 0.015_f32 * (((i * 11 + 5) % 17) as f32 - 8.0);
        let off = 49_408 + i * 4;
        arena[off..off + 4].copy_from_slice(&val.to_le_bytes());
    }
    for i in 0..10 {
        let val = 0.03_f32 * ((i as f32) - 4.0);
        let off = 9_024 + i * 4;
        arena[off..off + 4].copy_from_slice(&val.to_le_bytes());
    }
    for i in 0..8 {
        let val = 0.02_f32 * ((i as f32) - 3.5);
        let off = 63_424 + i * 4;
        arena[off..off + 4].copy_from_slice(&val.to_le_bytes());
    }
    for i in 0..64 {
        let val = 0.05_f32 * (((i * 7 + 1) % 13) as f32 - 6.0);
        let off = 8_256 + i * 4;
        arena[off..off + 4].copy_from_slice(&val.to_le_bytes());
        let off_u = 63_168 + i * 4;
        arena[off_u..off_u + 4].copy_from_slice(&val.to_le_bytes());
    }

    for (g, &tbase) in [4_672_usize, 17_792_usize, 34_496_usize, 48_896_usize]
        .iter()
        .enumerate()
    {
        for lane in 0..4 {
            let tau0 = 64.0_f32 + ((g * 4 + lane) as f32) * 29.0;
            arena[tbase + lane * 4..tbase + (lane + 1) * 4].copy_from_slice(&tau0.to_le_bytes());
            arena[tbase + 16 + lane * 4..tbase + 16 + (lane + 1) * 4]
                .copy_from_slice(&0.72_f32.to_le_bytes());
        }
    }

    (header, arena)
}

#[test]
fn test_marketplace_provenance_enclave_encryption_and_gauge_audio_equivalence() {
    let (mut header, master_arena) = build_reference_header_and_arena();
    let mut buyer_arena = master_arena.clone();

    let mut kappa = [1.0_f32; KAUTZ_TOTAL_CHANNELS];
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

    // 1. Structural hash cosine similarity must be 1.00000000000000 across all 4 tiers (224-D) and prefixes!
    let hash_master = extract_structural_hash_224d(&master_arena).unwrap();
    let hash_buyer = extract_structural_hash_224d(&buyer_arena).unwrap();
    let cos_sim = structural_cosine_similarity(&hash_master, &hash_buyer);
    assert!((cos_sim - 1.0).abs() < 1e-12);

    // Also verify tier-sliced isometric gauge watermark and Tardos scoring on Lite, Full, and Pro prefixes
    let mp_secret = *b"dnam-marketplace-master-key-2026";
    let test_code = generate_tardos_code_7680(&mp_secret, "tape_echo_v1", "tier_probe");
    for &tier_len in &[
        ARENA_OFFSET_LITE_END as usize,
        ARENA_OFFSET_FULL_END as usize,
        ARENA_OFFSET_PRO_END as usize,
    ] {
        let mut sliced = master_arena[..tier_len].to_vec();
        apply_isometric_gauge_watermark(&mut sliced, &kappa, &q_buyer).unwrap();
        let h_m = extract_structural_hash_224d(&master_arena[..tier_len]).unwrap();
        let h_s = extract_structural_hash_224d(&sliced).unwrap();
        assert!((structural_cosine_similarity(&h_m, &h_s) - 1.0).abs() < 1e-12);

        let mut tardos_sliced = master_arena[..tier_len].to_vec();
        apply_tardos_watermark_7680(&mut tardos_sliced, &test_code, 2e-3).unwrap();
        let z_tier =
            score_tardos_watermark_7680(&master_arena[..tier_len], &tardos_sliced, &test_code)
                .unwrap();
        assert!(z_tier > 15.0);
    }

    // 2. Render audio through master and Mechanism-A watermarked model -> < 1e-5 diff
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
    assert!(max_audio_diff < 1e-5);

    // 3. Verify Mechanism B (7,680-Weight Spread-Spectrum Tardos Watermark) under 4-buyer collusion + fp16 truncation
    let colluder_ids = ["lic_017", "lic_042", "lic_108", "lic_163"];
    let mut colluder_arenas = Vec::new();
    let mut colluder_codes = Vec::new();
    for &cid in &colluder_ids {
        let code = generate_tardos_code_7680(&mp_secret, "tape_echo_v1", cid);
        let mut a = master_arena.clone();
        apply_tardos_watermark_7680(&mut a, &code, 2e-3).unwrap();
        colluder_arenas.push(a);
        colluder_codes.push(code);
    }

    let mut pirated_arena = master_arena.clone();
    for off in (128..49_600).step_by(4) {
        let mut sum = 0.0_f32;
        for ca in &colluder_arenas {
            sum += f32::from_le_bytes([ca[off], ca[off + 1], ca[off + 2], ca[off + 3]]);
        }
        let avg = sum * 0.25;
        let fp16_attacked = fp16_bits_to_f32(f32_to_fp16_bits(avg));
        pirated_arena[off..off + 4].copy_from_slice(&fp16_attacked.to_le_bytes());
    }

    for code in &colluder_codes {
        let z = score_tardos_watermark_7680(&master_arena, &pirated_arena, code).unwrap();
        assert!(z > 12.0);
    }
    let innocent_code = generate_tardos_code_7680(&mp_secret, "tape_echo_v1", "innocent_buyer_999");
    let z_innocent =
        score_tardos_watermark_7680(&master_arena, &pirated_arena, &innocent_code).unwrap();
    assert!(z_innocent.abs() < 4.5 && TARDOS_CASCADE_WEIGHTS == 7680);

    // 4. Verify Split-Payload Ed25519 Signatures, 4-Tier Prefix Digests, X25519 + ChaCha20 64B-Block Arena Encryption,
    // 4 KB In-Place Header Preset Updates, and Zero-Re-Signing Container Tier Stripping!
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
    let master_tier_shas = [
        encode_hex(&sha256_digest(
            &master_arena[..ARENA_OFFSET_LITE_END as usize],
        )),
        encode_hex(&sha256_digest(
            &master_arena[..ARENA_OFFSET_FULL_END as usize],
        )),
        encode_hex(&sha256_digest(
            &master_arena[..ARENA_OFFSET_PRO_END as usize],
        )),
        encode_hex(&master_sha),
    ];
    let buyer_tier_shas = [
        encode_hex(&sha256_digest(
            &buyer_arena[..ARENA_OFFSET_LITE_END as usize],
        )),
        encode_hex(&sha256_digest(
            &buyer_arena[..ARENA_OFFSET_FULL_END as usize],
        )),
        encode_hex(&sha256_digest(
            &buyer_arena[..ARENA_OFFSET_PRO_END as usize],
        )),
        encode_hex(&buyer_sha),
    ];

    let creator_msg = build_creator_signing_message(&imm_sha, &master_sha);
    let creator_sig = creator_sk.sign(&creator_msg).to_bytes();

    let mut receipt = MarketplaceReceipt {
        marketplace_id: "dnam.store".to_string(),
        license_id: "0192b6c0-7e4a-7000-8000-1a2b3c4d5e6f".to_string(),
        buyer_hash: encode_hex(&[0xAB; 32]),
        license_tier: "personal".to_string(),
        issued_at: "2026-10-23T00:00:00Z".to_string(),
        buyer_arena_sha256: encode_hex(&buyer_sha),
        buyer_tier_arena_sha256: Some(buyer_tier_shas),
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
        tier_arena_sha256: Some(master_tier_shas),
        creator_signature: CreatorSignature {
            signer_id: "did:web:studio.example".to_string(),
            pubkey_ed25519: encode_hex(&creator_pk),
            sig_ed25519: encode_hex(&creator_sig),
        },
        marketplace_receipt: Some(receipt),
    });

    let mut encrypted_arena = buyer_arena.clone();
    chacha20_xor_arena_blocks(&mut encrypted_arena, &cek, &nonce_96, 0).unwrap();

    let mut container_bytes =
        build_dnam_container(&mut header, &buyer_arena, &encrypted_arena, true).unwrap();

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
    assert_eq!(loaded_view.arena_storage.as_bytes(), buyer_arena.as_slice());

    for max_tier in [Tier::Lite, Tier::Full, Tier::Pro] {
        let stripped = strip_container_to_tier(&container_bytes, max_tier).unwrap();
        let expected_arena_len = max_tier.expected_arena_bytes();
        let stripped_view = DnamContainerView::from_bytes(&stripped, &crypto).unwrap();
        assert_eq!(
            stripped_view.verification_status,
            VerificationStatus::MarketplaceVerified {
                signer_id: "did:web:studio.example".to_string(),
                marketplace_id: "dnam.store".to_string(),
                license_id: "0192b6c0-7e4a-7000-8000-1a2b3c4d5e6f".to_string(),
                license_tier: "personal".to_string(),
            }
        );
        assert_eq!(
            stripped_view.arena_storage.as_bytes(),
            &buyer_arena[..expected_arena_len]
        );
    }
}
