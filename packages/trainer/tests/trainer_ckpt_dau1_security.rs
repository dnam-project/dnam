#![forbid(unsafe_code)]
//! Regression tests for v1 `.dnam-ckpt` trailing-byte rejection, v2 `DNCA` multi-marker split
//! resistance against `f64` magic collisions (`ver == 1`), `DAU1` (`auth_kind == 1`) pinned
//! `marketplace_pubkey` enforcement, and `bundle`/`view` error guards.

use dnam_engine::Tier;
use dnam_engine::container::{
    ARENA_OFFSET_LITE_END, DnamContainerView, build_dmu1_mutable_frame,
    build_v2_open_dnam_container, crc32c_compute,
};
use dnam_engine::provenance::{
    Dau1Envelope, DefaultCryptoProvider, RustCryptoProvider, build_buyer_receipt_v1,
    build_creator_manifest_v1, build_plaintext_shared_v1, compute_creator_signed_manifest_sha256,
    compute_license_policy_sha256, compute_plaintext_prefixes_v1, sha256_digest,
};
use dnam_engine::schema::{DnamError, TensorSliceEntry};
use dnam_trainer::ckpt::{
    DnamCkptHeader, DnamCkptView, ExtractedPhysics, StemProvenance, StemRegistryEntry,
    build_dnam_ckpt_container, build_dnck_writer_manifest_message, build_v2_dnck_checkpoint,
    export_runtime_dnam_from_ckpt, export_runtime_dnam_from_ckpt_with_crypto,
    strip_dnck_checkpoint_to_tier_with_crypto, validate_v2_dnck_checkpoint,
};
use std::collections::BTreeMap;

fn make_stem_entry(offset: u32, len: u32, crc: u32) -> StemRegistryEntry {
    let slice = TensorSliceEntry {
        arena: 4,
        dtype: "f64".to_string(),
        packing: Some("uplo_upper_col_major_tier_ordered".to_string()),
        feature_dim: Some(64),
        packed_len: Some(2080),
        shape: None,
        byte_offset: offset,
        byte_length: len,
    };
    StemRegistryEntry {
        stem_id: "stem0".to_string(),
        enabled: true,
        controls_meta: BTreeMap::new(),
        provenance: StemProvenance {
            input_pcm_sha256: "00".repeat(32),
            target_pcm_sha256: "11".repeat(32),
            num_frames: 48_000,
        },
        stem_arena4_crc32c: crc,
        user_metadata: BTreeMap::new(),
        extracted_physics: Some(ExtractedPhysics::default()),
        g_audio_uplo_u_f64: slice.clone(),
        c_audio_tier_f64: slice,
    }
}

#[test]
fn test_ckpt_trailing_bytes_dnca_ver1_f64_collision_and_pinned_marketplace_pubkey() {
    let crypto = DefaultCryptoProvider::default();

    // 1. Canonical v2.0 .dnam-ckpt rejects trailing bytes and short/magic-mismatched buffers
    assert!(DnamCkptView::from_bytes(&[0u8; 32], &crypto).is_err());
    let mut std_hdr = DnamCkptHeader::default_reference();
    let lite_end = ARENA_OFFSET_LITE_END as usize;
    let stems_end = lite_end + 64;
    let mut a5_lite = vec![0u8; stems_end + 384];
    let a5 = &mut a5_lite[stems_end..];
    a5[0..4].copy_from_slice(b"DNCA");
    a5[4..6].copy_from_slice(&1u16.to_le_bytes());
    a5[8..10].copy_from_slice(&1u16.to_le_bytes());
    a5[10..12].copy_from_slice(&1u16.to_le_bytes());
    a5[12..16].copy_from_slice(&384u32.to_le_bytes());
    a5[64..68].copy_from_slice(b"INPC");
    a5[64 + 56..64 + 60].copy_from_slice(&128u32.to_le_bytes());
    let inpc_crc = crc32c_compute(&a5[128..192]);
    a5[64 + 60..64 + 64].copy_from_slice(&inpc_crc.to_le_bytes());
    a5[192..196].copy_from_slice(b"STMC");
    a5[192 + 116..192 + 120].copy_from_slice(&192u32.to_le_bytes());
    let stmc_crc = crc32c_compute(&a5[320..384]);
    a5[192 + 120..192 + 124].copy_from_slice(&stmc_crc.to_le_bytes());
    let body_crc = crc32c_compute(&a5[64..]);
    a5[16..20].copy_from_slice(&body_crc.to_le_bytes());

    std_hdr.stems_registry = Some(vec![make_stem_entry(
        ARENA_OFFSET_LITE_END,
        64,
        crc32c_compute(&[0u8; 64]),
    )]);
    assert!(build_dnam_ckpt_container(&mut std_hdr, &[0u8; 64], &a5_lite, false).is_err());
    let std_ckpt = build_dnam_ckpt_container(&mut std_hdr, &a5_lite, &a5_lite, false).unwrap();
    assert!(DnamCkptView::from_bytes(&std_ckpt, &crypto).is_ok());
    let mut ckpt_trailing = std_ckpt.clone();
    ckpt_trailing.push(0);
    assert!(DnamCkptView::from_bytes(&ckpt_trailing, &crypto).is_err());

    let exported_lite = export_runtime_dnam_from_ckpt(&std_ckpt).unwrap();
    let exported_lite_view = DnamContainerView::from_bytes(&exported_lite, &crypto).unwrap();
    assert_eq!(
        (
            exported_lite_view.preamble.highest_tier(),
            exported_lite_view.arena_storage.as_bytes().len()
        ),
        (Tier::Lite, lite_end)
    );
    assert!(matches!(
        DnamCkptView::from_bytes(&exported_lite, &crypto),
        Err(DnamError::InvalidMagic([b'D', b'N', b'A', b'M']))
    ));

    // 2. v2 f64 weight with bits 0x3FF0_0001_4143_4E44 ("DNCA" + ver=1) or [DNCA, 1, 0, 0, 0]
    //    followed by non-zero f64 weights does NOT false-split Arena 4 as Arena 5 DNCA
    let mut v2_hdr = DnamCkptHeader::default_reference();
    v2_hdr.stems_registry = None;
    v2_hdr.base.tensor_directory = BTreeMap::new();
    let hdr_json = serde_json::to_vec(&v2_hdr).unwrap();
    let hdr_sha = sha256_digest(&hdr_json);
    let rt_t0 = [0u8; 10_752];
    let rt_t1 = [0u8; 11_584];
    let rt_t2 = [0u8; 13_888];
    let rt_t3 = [0u8; 27_776];
    let rt_slices: [&[u8]; 4] = [&rt_t0, &rt_t1, &rt_t2, &rt_t3];

    let c_seed = [0x33u8; 32];
    let m_seed = [0x44u8; 32];
    let w_seed = [0x55u8; 32];
    let c_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&c_seed);
    let m_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&m_seed);
    let w_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&w_seed);
    let policy = b"tier=ultra;offline=1".to_vec();

    let dummy_dau = Dau1Envelope {
        auth_kind: 1,
        creator_pubkey: c_pk,
        creator_max_tier: 3,
        master_prefixes: vec![[0u8; 32]; 4],
        creator_signature: [0u8; 64],
        marketplace_pubkey: Some(m_pk),
        policy_bytes: policy.clone(),
        buyer_max_tier: Some(3),
        buyer_prefixes: vec![[0u8; 32]; 4],
        marketplace_signature: Some([0u8; 64]),
    }
    .to_bytes()
    .unwrap();
    let probe_rt =
        build_v2_open_dnam_container(&hdr_json, &[0u8; 64], &dummy_dau, &rt_slices, None, 1)
            .unwrap();
    let shared_f = build_plaintext_shared_v1(&probe_rt[0..64], &probe_rt[64..192], &[0u8; 64]);
    let m_pfxs =
        compute_plaintext_prefixes_v1(b"DNAM-MASTER-PREFIX-V1", &hdr_sha, &shared_f, &rt_slices)
            .unwrap();
    let b_pfxs =
        compute_plaintext_prefixes_v1(b"DNAM-BUYER-PREFIX-V1", &hdr_sha, &shared_f, &rt_slices)
            .unwrap();
    let c_man = build_creator_manifest_v1(&c_pk, &hdr_sha, &m_pfxs).unwrap();
    let c_sig = RustCryptoProvider::ed25519_sign(&c_seed, &c_man);
    let c_id = compute_creator_signed_manifest_sha256(&c_man, &c_sig);
    let pol_sha = compute_license_policy_sha256(&policy);
    let b_rec = build_buyer_receipt_v1(&m_pk, &c_id, &pol_sha, 0, &b_pfxs, None, &[]).unwrap();
    let m_sig = RustCryptoProvider::ed25519_sign(&m_seed, &b_rec);
    let real_dau = Dau1Envelope {
        auth_kind: 1,
        creator_pubkey: c_pk,
        creator_max_tier: 3,
        master_prefixes: m_pfxs,
        creator_signature: c_sig,
        marketplace_pubkey: Some(m_pk),
        policy_bytes: policy,
        buyer_max_tier: Some(3),
        buyer_prefixes: b_pfxs,
        marketplace_signature: Some(m_sig),
    }
    .to_bytes()
    .unwrap();
    let runtime_dnam =
        build_v2_open_dnam_container(&hdr_json, &[0u8; 64], &real_dau, &rt_slices, None, 1)
            .unwrap();

    let mut b0 = rt_t0.to_vec();
    let mut tail_a4 = vec![0x3F_u8; 256]; // non-zero finite f64 bytes so reserved != 0
    tail_a4[0..8].copy_from_slice(&0x3FF0_0001_4143_4E44_u64.to_le_bytes());
    tail_a4[64..72].copy_from_slice(&[b'D', b'N', b'C', b'A', 1, 0, 0, 0]);
    b0.extend_from_slice(&tail_a4);
    let mut entry_v2 = make_stem_entry(64_000, 128, crc32c_compute(&tail_a4));
    entry_v2.c_audio_tier_f64.byte_offset = 64_128;
    entry_v2.c_audio_tier_f64.byte_length = 128;
    v2_hdr.stems_registry = Some(vec![entry_v2]);
    let ckpt_hdr_json = serde_json::to_vec(&v2_hdr).unwrap();
    let bundle_refs: [&[u8]; 4] = [&b0, &rt_t1, &rt_t2, &rt_t3];

    assert!(
        build_v2_dnck_checkpoint(
            b"",
            &runtime_dnam,
            &[0u8; 64],
            &bundle_refs,
            &w_pk,
            &[0u8; 64],
            None
        )
        .is_err()
    );
    assert!(
        build_v2_dnck_checkpoint(
            &ckpt_hdr_json,
            &runtime_dnam,
            &[0u8; 63],
            &bundle_refs,
            &w_pk,
            &[0u8; 64],
            None
        )
        .is_err()
    );

    let dmu1 = build_dmu1_mutable_frame(br#"{"user_metadata":{"note":"tuned"}}"#, 64).unwrap();
    let probe_ckpt = build_v2_dnck_checkpoint(
        &ckpt_hdr_json,
        &runtime_dnam,
        &[],
        &bundle_refs,
        &w_pk,
        &[0u8; 64],
        Some(&dmu1),
    )
    .unwrap();
    let a_off = u64::from_le_bytes(probe_ckpt[64 + 32..64 + 40].try_into().unwrap()) as usize;
    let pfxs: Vec<[u8; 32]> = (0..4)
        .map(|i| {
            probe_ckpt[a_off + 43 + 33 * i..a_off + 75 + 33 * i]
                .try_into()
                .unwrap()
        })
        .collect();
    let w_msg = build_dnck_writer_manifest_message(&w_pk, &sha256_digest(&ckpt_hdr_json), 3, &pfxs);
    let w_sig = RustCryptoProvider::ed25519_sign(&w_seed, &w_msg);
    let signed_ckpt = build_v2_dnck_checkpoint(
        &ckpt_hdr_json,
        &runtime_dnam,
        &[],
        &bundle_refs,
        &w_pk,
        &w_sig,
        Some(&dmu1),
    )
    .unwrap();

    // 3. Pinned marketplace_pubkey succeeds on matching key and fails closed on wrong key
    let pinned_ok = RustCryptoProvider::new().with_marketplace_pubkey(m_pk);
    let pinned_wrong = RustCryptoProvider::new().with_marketplace_pubkey([0xEEu8; 32]);

    let ok_view = DnamCkptView::from_bytes(&signed_ckpt, &pinned_ok).unwrap();
    assert_eq!(
        ok_view.header.base.user_metadata.get("note"),
        Some(&serde_json::json!("tuned"))
    );
    assert!(export_runtime_dnam_from_ckpt_with_crypto(&signed_ckpt, &pinned_ok).is_ok());
    assert!(
        strip_dnck_checkpoint_to_tier_with_crypto(&signed_ckpt, Tier::Lite, &pinned_ok).is_ok()
    );

    assert!(matches!(
        DnamCkptView::from_bytes(&signed_ckpt, &pinned_wrong),
        Err(DnamError::CryptoVerificationFailed(_))
    ));
    assert!(matches!(
        export_runtime_dnam_from_ckpt_with_crypto(&signed_ckpt, &pinned_wrong),
        Err(DnamError::CryptoVerificationFailed(_))
    ));
    assert!(matches!(
        strip_dnck_checkpoint_to_tier_with_crypto(&signed_ckpt, Tier::Lite, &pinned_wrong),
        Err(DnamError::CryptoVerificationFailed(_))
    ));

    // 4. Out-of-bounds & u64::MAX directory offsets return Err without panicking
    let mut oob_dir = signed_ckpt.clone();
    let huge = (signed_ckpt.len() as u64) + 64 * 1024;
    oob_dir[64 + 64..64 + 72].copy_from_slice(&huge.to_le_bytes());
    for t in 0..4usize {
        let e = huge + 192 * ((t as u64) + 1);
        oob_dir[64 + 72 + t * 8..64 + 80 + t * 8].copy_from_slice(&e.to_le_bytes());
    }
    assert!(validate_v2_dnck_checkpoint(&oob_dir).is_err());

    let mut overflow_dir = signed_ckpt;
    overflow_dir[72..80].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(validate_v2_dnck_checkpoint(&overflow_dir).is_err());
}

#[test]
fn test_v2_dau1_relocated_buyer_tier_distribution_policy_and_dmu1_schema_merges() {
    use dnam_engine::container::{DnamContainerView, append_dmue_mutable_suffix};
    use dnam_engine::provenance::{DefaultCryptoProvider, VerificationStatus};
    use dnam_engine::schema::{
        ControlKind, ControlSchemaDef, ControlUiHints, DistributionPolicy, DnamProtectionProfile,
        PresetDef,
    };
    use std::collections::BTreeMap;

    // 1. Unsigned v2 container does NOT set FLAG_HAS_SIGNATURES; tampered DAU1 or auth_kind=1 garbage fails closed
    let mut base_hdr = DnamCkptHeader::default_reference().base;
    base_hdr.controls_schema = vec![ControlSchemaDef {
        id: "Gain".to_string(),
        name: "Gain".to_string(),
        kind: ControlKind::Range,
        unit: None,
        lobatto_degree: None,
        anchor_nodes_norm: Some(vec![-1.0, 0.0, 1.0]),
        declared_range: Some([0.0, 10.0]),
        safe_clamped_range: Some([0.0, 10.0]),
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
    base_hdr.presets = vec![PresetDef {
        id: "signed_default".to_string(),
        name: "Signed Default".to_string(),
        description: None,
        controls: BTreeMap::new(),
        user_metadata: BTreeMap::new(),
    }];
    base_hdr
        .user_metadata
        .insert("signed_key".to_string(), serde_json::json!("v1"));
    let hdr_json = base_hdr.to_json_vec().unwrap();

    // Build valid 1-control shared extent (64B ControlClampDescriptor + 64B zero CP1 header = 128B)
    let mut shared_1ctrl = vec![0u8; 128];
    shared_1ctrl[4..8].copy_from_slice(&0.0f32.to_le_bytes());
    shared_1ctrl[8..12].copy_from_slice(&10.0f32.to_le_bytes());
    shared_1ctrl[12..16].copy_from_slice(&(-1.0f32).to_le_bytes());
    shared_1ctrl[16..20].copy_from_slice(&1.0f32.to_le_bytes());
    shared_1ctrl[64..68].copy_from_slice(b"CP1\0");

    let t0 = vec![0x11u8; 10_752];
    let t1 = vec![0x22u8; 11_584];
    let t2 = vec![0x33u8; 13_888];
    let t3 = vec![0x44u8; 27_776];

    let unsigned_dnam =
        build_v2_open_dnam_container(&hdr_json, &shared_1ctrl, &[0u8; 64], &[&t0], None, 0)
            .unwrap();
    let unsigned_view =
        DnamContainerView::from_bytes(&unsigned_dnam, &DefaultCryptoProvider::default()).unwrap();
    assert_eq!(
        unsigned_view.verification_status,
        VerificationStatus::Unsigned
    );
    assert!(!unsigned_view.preamble.has_signatures());

    // auth_kind = 1 with non-DAU1 auth_slice must fail closed (never fall back to Unsigned)
    let bad_ak1 =
        build_v2_open_dnam_container(&hdr_json, &shared_1ctrl, &[0u8; 64], &[&t0], None, 1)
            .unwrap();
    assert!(DnamContainerView::from_bytes(&bad_ak1, &DefaultCryptoProvider::default()).is_err());

    // auth_kind = 0 with tampered DAU1 version byte (`DAU1\x09...`) must fail closed
    let mut tampered_dau1 = vec![0u8; 160];
    tampered_dau1[0..8].copy_from_slice(b"DAU1\x09\x00\x00\x00");
    let bad_dau1 =
        build_v2_open_dnam_container(&hdr_json, &shared_1ctrl, &tampered_dau1, &[&t0], None, 0)
            .unwrap();
    assert!(DnamContainerView::from_bytes(&bad_dau1, &DefaultCryptoProvider::default()).is_err());

    // 2. Relocated lower-tier buyer issuance: creator signed Ultra (3), buyer issued Full (1)
    let c_seed = [0x51u8; 32];
    let m_seed = [0x62u8; 32];
    let c_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&c_seed);
    let m_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&m_seed);
    let hdr_sha = sha256_digest(&hdr_json);

    let master_probe = build_v2_open_dnam_container(
        &hdr_json,
        &shared_1ctrl,
        &[0u8; 174],
        &[&t0, &t1, &t2, &t3],
        None,
        0,
    )
    .unwrap();
    let master_shared =
        build_plaintext_shared_v1(&master_probe[..64], &master_probe[64..192], &shared_1ctrl);
    let master_pfxs = compute_plaintext_prefixes_v1(
        b"DNAM-MASTER-PREFIX-V1",
        &hdr_sha,
        &master_shared,
        &[&t0, &t1, &t2, &t3],
    )
    .unwrap();
    let c_manifest = build_creator_manifest_v1(&c_pk, &hdr_sha, &master_pfxs).unwrap();
    let c_sig = RustCryptoProvider::ed25519_sign(&c_seed, &c_manifest);

    // Relocated Full-only buyer container (auth_kind=1, maximum_tier=1, 4 master + 2 buyer prefixes)
    let policy = br#"{"tier":"full"}"#;
    let dummy_buyer_pfxs = vec![[0u8; 32]; 2];
    let probe_env_bytes = Dau1Envelope {
        auth_kind: 1,
        creator_pubkey: c_pk,
        creator_max_tier: 3,
        master_prefixes: master_pfxs.clone(),
        creator_signature: c_sig,
        marketplace_pubkey: Some(m_pk),
        policy_bytes: policy.to_vec(),
        buyer_max_tier: Some(1),
        buyer_prefixes: dummy_buyer_pfxs,
        marketplace_signature: Some([0u8; 64]),
    }
    .to_bytes()
    .unwrap();
    let buyer_probe = build_v2_open_dnam_container(
        &hdr_json,
        &shared_1ctrl,
        &probe_env_bytes,
        &[&t0, &t1],
        None,
        1,
    )
    .unwrap();
    let buyer_shared =
        build_plaintext_shared_v1(&buyer_probe[..64], &buyer_probe[64..192], &shared_1ctrl);
    let buyer_pfxs = compute_plaintext_prefixes_v1(
        b"DNAM-BUYER-PREFIX-V1",
        &hdr_sha,
        &buyer_shared,
        &[&t0, &t1],
    )
    .unwrap();
    let c_signed_sha = compute_creator_signed_manifest_sha256(&c_manifest, &c_sig);
    let pol_sha = compute_license_policy_sha256(policy);
    let b_receipt =
        build_buyer_receipt_v1(&m_pk, &c_signed_sha, &pol_sha, 0, &buyer_pfxs, None, &[]).unwrap();
    let m_sig = RustCryptoProvider::ed25519_sign(&m_seed, &b_receipt);

    let env_reloc = Dau1Envelope {
        auth_kind: 1,
        creator_pubkey: c_pk,
        creator_max_tier: 3,
        master_prefixes: master_pfxs,
        creator_signature: c_sig,
        marketplace_pubkey: Some(m_pk),
        policy_bytes: policy.to_vec(),
        buyer_max_tier: Some(1),
        buyer_prefixes: buyer_pfxs,
        marketplace_signature: Some(m_sig),
    };
    // Empty presets & user_metadata clear signed defaults, and control_user_metadata merges cleanly
    let dmu_clear_and_merge = build_dmu1_mutable_frame(
        br#"{"presets":[],"user_metadata":{},"control_user_metadata":{"Gain":{"midi_cc":7}}}"#,
        128,
    )
    .unwrap();
    let reloc_dnam = build_v2_open_dnam_container(
        &hdr_json,
        &shared_1ctrl,
        &env_reloc.to_bytes().unwrap(),
        &[&t0, &t1],
        Some(&dmu_clear_and_merge),
        1,
    )
    .unwrap();
    let crypto_mp = RustCryptoProvider::new().with_marketplace_pubkey(m_pk);
    let reloc_view = DnamContainerView::from_bytes(&reloc_dnam, &crypto_mp).unwrap();
    assert!(matches!(
        reloc_view.verification_status,
        VerificationStatus::MarketplaceVerified { ref license_tier, .. } if license_tier == "full"
    ));
    assert!(reloc_view.header.presets.is_empty());
    assert!(reloc_view.header.user_metadata.is_empty());
    assert_eq!(
        reloc_view.header.controls_schema[0]
            .user_metadata
            .get("midi_cc"),
        Some(&serde_json::json!(7))
    );

    // 3. Unknown DMU1 schema key rejected fail-closed on both .dnam and .dnam-ckpt; >4096B DMU1 rejected on .dnam-ckpt
    let bad_dmu = build_dmu1_mutable_frame(br#"{"evil_unknown":1}"#, 48).unwrap();
    let bad_dmu_dnam = build_v2_open_dnam_container(
        &hdr_json,
        &shared_1ctrl,
        &[0u8; 64],
        &[&t0],
        Some(&bad_dmu),
        0,
    )
    .unwrap();
    assert!(DnamContainerView::from_bytes(&bad_dmu_dnam, &crypto_mp).is_err());

    let oversized_dmu = build_dmu1_mutable_frame(br#"{}"#, 4_160).unwrap();
    let a4_lite = vec![0u8; 17_664];
    let mut entry_lite = make_stem_entry(ARENA_OFFSET_LITE_END, 16_640, crc32c_compute(&a4_lite));
    entry_lite.c_audio_tier_f64.byte_offset = ARENA_OFFSET_LITE_END + 16_640;
    entry_lite.c_audio_tier_f64.byte_length = 1_024;
    let mut ckpt_hdr = DnamCkptHeader {
        base: base_hdr.clone(),
        stems_registry: Some(vec![entry_lite]),
        audit_report: None,
    };
    ckpt_hdr.base.container_kind = "dnam-ckpt".to_string();
    let ckpt_hdr_bytes = serde_json::to_vec(&ckpt_hdr).unwrap();
    let mut b0_ckpt = Vec::new();
    b0_ckpt.extend_from_slice(&t0);
    b0_ckpt.extend_from_slice(&a4_lite);

    let dmu_stem_um = build_dmu1_mutable_frame(
        br#"{"control_user_metadata":{"Gain":{"color":"red"}},"stem_user_metadata":{"stem0":{"mic":"sm57"}}}"#,
        160,
    )
    .unwrap();
    let w_seed = [0x99u8; 32];
    let w_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&w_seed);
    let probe_c = build_v2_dnck_checkpoint(
        &ckpt_hdr_bytes,
        &unsigned_dnam,
        &[],
        &[&b0_ckpt],
        &w_pk,
        &[0u8; 64],
        Some(&dmu_stem_um),
    )
    .unwrap();
    let ca_off = u64::from_le_bytes(probe_c[96..104].try_into().unwrap()) as usize;
    let pfx0: [u8; 32] = probe_c[ca_off + 43..ca_off + 75].try_into().unwrap();
    let w_sig = RustCryptoProvider::ed25519_sign(
        &w_seed,
        &build_dnck_writer_manifest_message(&w_pk, &sha256_digest(&ckpt_hdr_bytes), 0, &[pfx0]),
    );
    let valid_ckpt = build_v2_dnck_checkpoint(
        &ckpt_hdr_bytes,
        &unsigned_dnam,
        &[],
        &[&b0_ckpt],
        &w_pk,
        &w_sig,
        Some(&dmu_stem_um),
    )
    .unwrap();
    let ckpt_view = DnamCkptView::from_bytes(&valid_ckpt, &crypto_mp).unwrap();
    assert_eq!(
        ckpt_view.header.base.controls_schema[0]
            .user_metadata
            .get("color"),
        Some(&serde_json::json!("red"))
    );
    assert_eq!(
        ckpt_view.header.stems_registry.as_ref().unwrap()[0]
            .user_metadata
            .get("mic"),
        Some(&serde_json::json!("sm57"))
    );

    // Oversized DMU1 (> 4096 B) on .dnam-ckpt fails closed
    let boundary = u64::from_le_bytes(valid_ckpt[64 + 72..64 + 80].try_into().unwrap()) as usize;
    let mut over_ckpt = valid_ckpt[..boundary].to_vec();
    append_dmue_mutable_suffix(&mut over_ckpt, &oversized_dmu);
    assert!(validate_v2_dnck_checkpoint(&over_ckpt).is_err());

    // 4. DistributionPolicy enforcement: enclave_sealed-only rejected on Open .dnam and Open .dnam-ckpt export
    let mut enclave_hdr = base_hdr;
    enclave_hdr.metadata.distribution_policy = Some(DistributionPolicy {
        allowed_protection_profiles: vec![DnamProtectionProfile::EnclaveSealed],
        allowed_oem_vendors: Vec::new(),
    });
    let enc_hdr_bytes = enclave_hdr.to_json_vec().unwrap();
    let illegal_open_dnam =
        build_v2_open_dnam_container(&enc_hdr_bytes, &shared_1ctrl, &[0u8; 64], &[&t0], None, 0)
            .unwrap();
    assert!(matches!(
        DnamContainerView::from_bytes(&illegal_open_dnam, &crypto_mp),
        Err(DnamError::CryptoVerificationFailed(_))
    ));
    let mut enc_ckpt_hdr = ckpt_hdr;
    enc_ckpt_hdr.base = enclave_hdr;
    enc_ckpt_hdr.base.container_kind = "dnam-ckpt".to_string();
    let enc_ckpt_hdr_bytes = serde_json::to_vec(&enc_ckpt_hdr).unwrap();
    let p_enc = build_v2_dnck_checkpoint(
        &enc_ckpt_hdr_bytes,
        &illegal_open_dnam,
        &[],
        &[&b0_ckpt],
        &w_pk,
        &[0u8; 64],
        None,
    )
    .unwrap();
    let ea_off = u64::from_le_bytes(p_enc[96..104].try_into().unwrap()) as usize;
    let epfx0: [u8; 32] = p_enc[ea_off + 43..ea_off + 75].try_into().unwrap();
    let ew_sig = RustCryptoProvider::ed25519_sign(
        &w_seed,
        &build_dnck_writer_manifest_message(
            &w_pk,
            &sha256_digest(&enc_ckpt_hdr_bytes),
            0,
            &[epfx0],
        ),
    );
    let enc_ckpt = build_v2_dnck_checkpoint(
        &enc_ckpt_hdr_bytes,
        &illegal_open_dnam,
        &[],
        &[&b0_ckpt],
        &w_pk,
        &ew_sig,
        None,
    )
    .unwrap();
    assert!(matches!(
        export_runtime_dnam_from_ckpt_with_crypto(&enc_ckpt, &crypto_mp),
        Err(DnamError::CryptoVerificationFailed(_))
    ));

    // Unknown DMU1 key on .dnam-ckpt fails closed
    let mut bad_dmu_ckpt = valid_ckpt[..boundary].to_vec();
    append_dmue_mutable_suffix(&mut bad_dmu_ckpt, &bad_dmu);
    assert!(validate_v2_dnck_checkpoint(&bad_dmu_ckpt).is_err());

    // Relocated Full-only buyer v2 .dnam-ckpt export & strip to Lite succeed with pinned marketplace key
    let p_rel = build_v2_dnck_checkpoint(
        &ckpt_hdr_bytes,
        &reloc_dnam,
        &[],
        &[&b0_ckpt, &t1],
        &w_pk,
        &[0u8; 64],
        None,
    )
    .unwrap();
    let ra_off = u64::from_le_bytes(p_rel[96..104].try_into().unwrap()) as usize;
    let rpfx0: [u8; 32] = p_rel[ra_off + 43..ra_off + 75].try_into().unwrap();
    let rpfx1: [u8; 32] = p_rel[ra_off + 76..ra_off + 108].try_into().unwrap();
    let rw_sig = RustCryptoProvider::ed25519_sign(
        &w_seed,
        &build_dnck_writer_manifest_message(
            &w_pk,
            &sha256_digest(&ckpt_hdr_bytes),
            1,
            &[rpfx0, rpfx1],
        ),
    );
    let reloc_ckpt = build_v2_dnck_checkpoint(
        &ckpt_hdr_bytes,
        &reloc_dnam,
        &[],
        &[&b0_ckpt, &t1],
        &w_pk,
        &rw_sig,
        None,
    )
    .unwrap();
    let exp_reloc = export_runtime_dnam_from_ckpt_with_crypto(&reloc_ckpt, &crypto_mp).unwrap();
    assert!(DnamContainerView::from_bytes(&exp_reloc, &crypto_mp).is_ok());
    let stripped_reloc =
        strip_dnck_checkpoint_to_tier_with_crypto(&reloc_ckpt, Tier::Lite, &crypto_mp).unwrap();
    assert!(export_runtime_dnam_from_ckpt_with_crypto(&stripped_reloc, &crypto_mp).is_ok());

    // Canonical v2.0 .dnam-ckpt Open distribution_policy rejection, preamble guards, and base-header anti-spoofing
    let mut std_enc_hdr = DnamCkptHeader::default_reference();
    std_enc_hdr.base.metadata.distribution_policy = Some(DistributionPolicy {
        allowed_protection_profiles: vec![DnamProtectionProfile::EnclaveSealed],
        allowed_oem_vendors: Vec::new(),
    });
    let std_arena = vec![0u8; ARENA_OFFSET_LITE_END as usize];
    let std_enc_pol =
        build_dnam_ckpt_container(&mut std_enc_hdr, &std_arena, &std_arena, false).unwrap();
    assert!(DnamCkptView::from_bytes(&std_enc_pol, &crypto_mp).is_err());
    assert!(export_runtime_dnam_from_ckpt(&std_enc_pol).is_err());

    let mut std_ok_hdr = DnamCkptHeader::default_reference();
    let std_ok = build_dnam_ckpt_container(&mut std_ok_hdr, &std_arena, &std_arena, false).unwrap();
    let mut std_bad_dir = std_ok.clone();
    std_bad_dir[64] ^= 1;
    assert!(DnamCkptView::from_bytes(&std_bad_dir, &crypto_mp).is_err());

    let mut std_bad_profile = std_ok;
    std_bad_profile[8] = 1;
    assert!(DnamCkptView::from_bytes(&std_bad_profile, &crypto_mp).is_err());

    // Regression guard: DNCK wrapper with forged base.metadata.name around valid MarketplaceVerified DAU1 .dnam must fail closed
    let mut forged_ckpt_hdr: DnamCkptHeader = serde_json::from_slice(&ckpt_hdr_bytes).unwrap();
    forged_ckpt_hdr.base.metadata.name = "Forged Model Name".to_string();
    let forged_hdr_bytes = serde_json::to_vec(&forged_ckpt_hdr).unwrap();
    let p_forg = build_v2_dnck_checkpoint(
        &forged_hdr_bytes,
        &reloc_dnam,
        &[],
        &[&b0_ckpt, &t1],
        &w_pk,
        &[0u8; 64],
        None,
    )
    .unwrap();
    let fa_off = u64::from_le_bytes(p_forg[96..104].try_into().unwrap()) as usize;
    let fpfx0: [u8; 32] = p_forg[fa_off + 43..fa_off + 75].try_into().unwrap();
    let fpfx1: [u8; 32] = p_forg[fa_off + 76..fa_off + 108].try_into().unwrap();
    let fw_sig = RustCryptoProvider::ed25519_sign(
        &w_seed,
        &build_dnck_writer_manifest_message(
            &w_pk,
            &sha256_digest(&forged_hdr_bytes),
            1,
            &[fpfx0, fpfx1],
        ),
    );
    let forged_base_ckpt = build_v2_dnck_checkpoint(
        &forged_hdr_bytes,
        &reloc_dnam,
        &[],
        &[&b0_ckpt, &t1],
        &w_pk,
        &fw_sig,
        None,
    )
    .unwrap();
    assert!(matches!(
        DnamCkptView::from_bytes(&forged_base_ckpt, &crypto_mp),
        Err(DnamError::CryptoVerificationFailed(_))
    ));
}
