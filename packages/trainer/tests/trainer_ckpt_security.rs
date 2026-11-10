#![forbid(unsafe_code)]
//! Checkpoint cryptographic verification (`DCA1`, `DAU1` `auth_kind = 0/1`), `Arena 4` registry
//! validation, and stem-stripping tests.

use dnam_engine::Tier;
use dnam_engine::container::{
    ARENA_OFFSET_ULTRA_END, align64_u64, build_v2_open_dnam_container, crc32c_compute,
    validate_v2_open_dnam_layout,
};
use dnam_engine::provenance::{
    Dau1Envelope, RustCryptoProvider, build_buyer_receipt_v1, build_creator_manifest_v1,
    build_plaintext_shared_v1, compute_creator_signed_manifest_sha256,
    compute_license_policy_sha256, compute_plaintext_prefixes_v1, sha256_digest,
};
use dnam_engine::schema::TensorSliceEntry;
use dnam_trainer::ckpt::{
    ARENA4_STEM_BYTES_ULTRA, DnamCkptHeader, StemProvenance, StemRegistryEntry,
    build_dnam_ckpt_container, build_dnck_writer_manifest_message, build_v2_dnck_checkpoint,
    compute_dnck_bundle_prefixes, export_runtime_dnam_from_ckpt, strip_dnck_checkpoint_to_tier,
    validate_ckpt_arena4_and_stems_registry,
};
use std::collections::BTreeMap;

#[test]
fn test_v2_dnck_dau1_creator_and_buyer_verification_and_v1_stem_strip() {
    let hdr = br#"{"version":1}"#;
    let bundles: Vec<Vec<u8>> = (0..4u8).map(|t| vec![0x20 + t; 128]).collect();
    let rt_slices: Vec<&[u8]> = bundles.iter().map(|b| &b[..64]).collect();
    let b_refs: Vec<&[u8]> = bundles.iter().map(Vec::as_slice).collect();
    let seed = [0x5A_u8; 32];
    let pk = RustCryptoProvider::ed25519_pubkey_from_seed(&seed);

    for auth_kind in [0u8, 1u8] {
        let dummy_dau1 = Dau1Envelope {
            auth_kind,
            creator_pubkey: pk,
            creator_max_tier: 3,
            master_prefixes: vec![[0u8; 32]; 4],
            creator_signature: [0u8; 64],
            marketplace_pubkey: (auth_kind == 1).then_some(pk),
            policy_bytes: if auth_kind == 1 {
                b"p".to_vec()
            } else {
                Vec::new()
            },
            buyer_max_tier: (auth_kind == 1).then_some(3),
            buyer_prefixes: if auth_kind == 1 {
                vec![[0u8; 32]; 4]
            } else {
                Vec::new()
            },
            marketplace_signature: (auth_kind == 1).then_some([0u8; 64]),
        }
        .to_bytes()
        .unwrap();
        let probe = build_v2_open_dnam_container(
            hdr,
            &[b'S'; 64],
            &dummy_dau1,
            &rt_slices,
            None,
            auth_kind,
        )
        .unwrap();
        let p_lay = validate_v2_open_dnam_layout(&probe).unwrap();
        let sh_fr = build_plaintext_shared_v1(&probe[0..64], &probe[64..192], &[b'S'; 64]);
        let h_sha = sha256_digest(hdr);
        let m_pfx =
            compute_plaintext_prefixes_v1(b"DNAM-MASTER-PREFIX-V1", &h_sha, &sh_fr, &rt_slices)
                .unwrap();
        let c_msg = build_creator_manifest_v1(&pk, &h_sha, &m_pfx).unwrap();
        let c_sig = RustCryptoProvider::ed25519_sign(&seed, &c_msg);
        let dau1 = if auth_kind == 0 {
            Dau1Envelope {
                auth_kind: 0,
                creator_pubkey: pk,
                creator_max_tier: 3,
                master_prefixes: m_pfx,
                creator_signature: c_sig,
                marketplace_pubkey: None,
                policy_bytes: Vec::new(),
                buyer_max_tier: None,
                buyer_prefixes: Vec::new(),
                marketplace_signature: None,
            }
        } else {
            let b_pfx =
                compute_plaintext_prefixes_v1(b"DNAM-BUYER-PREFIX-V1", &h_sha, &sh_fr, &rt_slices)
                    .unwrap();
            let cs_sha = compute_creator_signed_manifest_sha256(&c_msg, &c_sig);
            let pol = b"p".to_vec();
            let m_msg = build_buyer_receipt_v1(
                &pk,
                &cs_sha,
                &compute_license_policy_sha256(&pol),
                p_lay.protection_profile,
                &b_pfx,
                None,
                &[],
            )
            .unwrap();
            Dau1Envelope {
                auth_kind: 1,
                creator_pubkey: pk,
                creator_max_tier: 3,
                master_prefixes: m_pfx,
                creator_signature: c_sig,
                marketplace_pubkey: Some(pk),
                policy_bytes: pol,
                buyer_max_tier: Some(3),
                buyer_prefixes: b_pfx,
                marketplace_signature: Some(RustCryptoProvider::ed25519_sign(&seed, &m_msg)),
            }
        }
        .to_bytes()
        .unwrap();
        let rt = build_v2_open_dnam_container(hdr, &[b'S'; 64], &dau1, &rt_slices, None, auth_kind)
            .unwrap();
        let rt_lay = validate_v2_open_dnam_layout(&rt).unwrap();
        let mut sh = rt[..rt_lay.tier_start].to_vec();
        sh.extend_from_slice(&[0u8; 64]);
        let s_off = align64_u64(192 + hdr.len() as u64).unwrap();
        let a_off = align64_u64(s_off + sh.len() as u64).unwrap();
        let a_len = (106 + 33 * 4) as u64;
        let b_start = align64_u64(a_off + a_len).unwrap();
        let mut pre = [0u8; 64];
        pre[0..4].copy_from_slice(b"DNCK");
        pre[4..6].copy_from_slice(&2u16.to_le_bytes());
        pre[10] = 3;
        pre[12..16].copy_from_slice(&48_000u32.to_le_bytes());
        let mut dir = [0u8; 128];
        for (i, v) in [
            192u64,
            hdr.len() as u64,
            s_off,
            sh.len() as u64,
            a_off,
            a_len,
            0,
            0,
            b_start,
            b_start + 128,
            b_start + 256,
            b_start + 384,
            b_start + 512,
        ]
        .into_iter()
        .enumerate()
        {
            dir[i * 8..(i + 1) * 8].copy_from_slice(&v.to_le_bytes());
        }
        let pfx = compute_dnck_bundle_prefixes(&pre, &dir, hdr, &sh, &b_refs);
        let w_msg = build_dnck_writer_manifest_message(&pk, &h_sha, 3, &pfx);
        let w_sig = RustCryptoProvider::ed25519_sign(&seed, &w_msg);
        let ckpt =
            build_v2_dnck_checkpoint(hdr, &rt, &[0u8; 64], &b_refs, &pk, &w_sig, None).unwrap();
        assert!(export_runtime_dnam_from_ckpt(&ckpt).is_ok());

        // Tampering with embedded DAU1 signature inside runtime template must fail export
        let mut bad_rt = rt.clone();
        let dau1_end = rt_lay.auth_extent.0 + rt_lay.auth_extent.1;
        bad_rt[dau1_end - 1] ^= 0xFF;
        let mut bad_sh = bad_rt[..rt_lay.tier_start].to_vec();
        bad_sh.extend_from_slice(&[0u8; 64]);
        let bad_pfx = compute_dnck_bundle_prefixes(&pre, &dir, hdr, &bad_sh, &b_refs);
        let bad_w_msg = build_dnck_writer_manifest_message(&pk, &h_sha, 3, &bad_pfx);
        let bad_w_sig = RustCryptoProvider::ed25519_sign(&seed, &bad_w_msg);
        let bad_ckpt =
            build_v2_dnck_checkpoint(hdr, &bad_rt, &[0u8; 64], &b_refs, &pk, &bad_w_sig, None)
                .unwrap();
        assert!(export_runtime_dnam_from_ckpt(&bad_ckpt).is_err());
        assert!(strip_dnck_checkpoint_to_tier(&bad_ckpt, Tier::Pro).is_err());

        // Malformed DAU1 header byte (`auth_slice[4] = 2`) when auth_kind == 0 must ALSO fail closed
        if auth_kind == 0 {
            let mut mal_rt = rt.clone();
            mal_rt[rt_lay.auth_extent.0 + 4] = 2;
            let mut mal_sh = mal_rt[..rt_lay.tier_start].to_vec();
            mal_sh.extend_from_slice(&[0u8; 64]);
            let mal_pfx = compute_dnck_bundle_prefixes(&pre, &dir, hdr, &mal_sh, &b_refs);
            let mal_sig = RustCryptoProvider::ed25519_sign(
                &seed,
                &build_dnck_writer_manifest_message(&pk, &h_sha, 3, &mal_pfx),
            );
            let mal_ckpt =
                build_v2_dnck_checkpoint(hdr, &mal_rt, &[0u8; 64], &b_refs, &pk, &mal_sig, None)
                    .unwrap();
            assert!(export_runtime_dnam_from_ckpt(&mal_ckpt).is_err());
            assert!(strip_dnck_checkpoint_to_tier(&mal_ckpt, Tier::Lite).is_err());
        }
    }

    // Exercise canonical v2.0 stem-bearing checkpoint strip (`strip_dnck_checkpoint_to_tier` with stems_registry)
    let mut ckpt_hdr = DnamCkptHeader::default_reference();
    let w_end = ARENA_OFFSET_ULTRA_END as usize;
    let s_end = w_end + ARENA4_STEM_BYTES_ULTRA;
    let arena = vec![0u8; s_end];
    let crc = crc32c_compute(&arena[w_end..]);
    ckpt_hdr.stems_registry = Some(vec![StemRegistryEntry {
        stem_id: String::from("s0"),
        enabled: true,
        controls_meta: BTreeMap::new(),
        provenance: StemProvenance {
            input_pcm_sha256: "00".repeat(32),
            target_pcm_sha256: "11".repeat(32),
            num_frames: 48_000,
        },
        stem_arena4_crc32c: crc,
        user_metadata: BTreeMap::new(),
        extracted_physics: None,
        g_audio_uplo_u_f64: TensorSliceEntry {
            arena: 4,
            dtype: String::from("f64"),
            packing: Some(String::from("uplo_upper_col_major_tier_ordered")),
            feature_dim: Some(256),
            packed_len: Some(32_896),
            shape: None,
            byte_offset: ARENA_OFFSET_ULTRA_END,
            byte_length: 263_168,
        },
        c_audio_tier_f64: TensorSliceEntry {
            arena: 4,
            dtype: String::from("f64"),
            packing: None,
            feature_dim: Some(256),
            packed_len: Some(512),
            shape: Some(vec![256, 2]),
            byte_offset: ARENA_OFFSET_ULTRA_END + 263_168,
            byte_length: 4_096,
        },
    }]);
    let std_ckpt = build_dnam_ckpt_container(&mut ckpt_hdr, &arena, &arena, false).unwrap();
    assert_eq!(
        strip_dnck_checkpoint_to_tier(&std_ckpt, Tier::Ultra)
            .unwrap()
            .len(),
        std_ckpt.len()
    );
    let stripped_std = strip_dnck_checkpoint_to_tier(&std_ckpt, Tier::Pro).unwrap();
    assert!(stripped_std.len() < std_ckpt.len());

    // Exercise validate_ckpt_arena4_and_stems_registry error branches
    assert!(
        validate_ckpt_arena4_and_stems_registry(&ckpt_hdr, w_end, s_end, &arena, 1, false).is_ok()
    );
    assert!(
        validate_ckpt_arena4_and_stems_registry(&ckpt_hdr, s_end + 64, s_end, &arena, 1, false)
            .is_err()
    );
    assert!(
        validate_ckpt_arena4_and_stems_registry(&ckpt_hdr, w_end, w_end + 8, &arena, 1, true)
            .is_err()
    );
    let mut trailing = arena.clone();
    trailing.extend_from_slice(&[0u8; 64]);
    assert!(
        validate_ckpt_arena4_and_stems_registry(&ckpt_hdr, w_end, s_end, &trailing, 1, false)
            .is_err()
    );
    assert!(
        validate_ckpt_arena4_and_stems_registry(&ckpt_hdr, w_end, s_end, &arena, 2, false).is_err()
    );

    let mut nan_arena = arena.clone();
    nan_arena[w_end..w_end + 8].copy_from_slice(&f64::NAN.to_le_bytes());
    assert!(
        validate_ckpt_arena4_and_stems_registry(&ckpt_hdr, w_end, s_end, &nan_arena, 1, false)
            .is_err()
    );

    let mut bad_arena_idx = ckpt_hdr.clone();
    bad_arena_idx.stems_registry.as_mut().unwrap()[0]
        .g_audio_uplo_u_f64
        .arena = 0;
    assert!(
        validate_ckpt_arena4_and_stems_registry(&bad_arena_idx, w_end, s_end, &arena, 1, false)
            .is_err()
    );

    let mut bad_align8 = ckpt_hdr.clone();
    bad_align8.stems_registry.as_mut().unwrap()[0]
        .g_audio_uplo_u_f64
        .byte_offset += 4;
    assert!(
        validate_ckpt_arena4_and_stems_registry(&bad_align8, w_end, s_end, &arena, 1, false)
            .is_err()
    );

    let mut bad_bounds = ckpt_hdr.clone();
    bad_bounds.stems_registry.as_mut().unwrap()[0]
        .c_audio_tier_f64
        .byte_length += 64;
    assert!(
        validate_ckpt_arena4_and_stems_registry(&bad_bounds, w_end, s_end, &arena, 1, false)
            .is_err()
    );

    // Reject num_ckpt_stems > 0 when header.stems_registry is None
    let mut none_reg = bad_bounds;
    none_reg.stems_registry = None;
    assert!(
        validate_ckpt_arena4_and_stems_registry(&none_reg, w_end, s_end, &arena, 1, false).is_err()
    );

    // Verify per-bundle DNCA in v2 checkpoint (including bytes matching f64 exponent 0x7FF) & corrupt DNCA rejection
    let mut bundle_dnca = vec![0u8; 128];
    bundle_dnca[0..4].copy_from_slice(b"DNCA");
    bundle_dnca[4..6].copy_from_slice(&1u16.to_le_bytes());
    bundle_dnca[8..10].copy_from_slice(&1u16.to_le_bytes()); // 1 INPC record
    bundle_dnca[12..16].copy_from_slice(&128u32.to_le_bytes());
    bundle_dnca[64..68].copy_from_slice(b"INPC");
    // Embed 0x7F_F8 (NaN exponent if misinterpreted as f64) inside INPC header metadata bytes 8..16
    bundle_dnca[72..80].copy_from_slice(&f64::NAN.to_le_bytes());
    bundle_dnca[120..124].copy_from_slice(&64u32.to_le_bytes()); // rec_len = 64, empty payload -> crc = 0
    let p_crc = crc32c_compute(&bundle_dnca[64..]);
    bundle_dnca[16..20].copy_from_slice(&p_crc.to_le_bytes());

    let ckpt_hdr_json = serde_json::to_vec(&none_reg).unwrap();
    let mut b0 = vec![0u8; 128]; // 64B runtime + 64B Arena 4.0
    b0.extend_from_slice(&bundle_dnca); // + 128B Tier DNCA
    let b_rest: Vec<Vec<u8>> = (0..3).map(|_| vec![0u8; 128]).collect();
    let v2_b_refs = [
        b0.as_slice(),
        b_rest[0].as_slice(),
        b_rest[1].as_slice(),
        b_rest[2].as_slice(),
    ];
    let rt_dnam =
        build_v2_open_dnam_container(&ckpt_hdr_json, &[0u8; 64], &[0u8; 16], &rt_slices, None, 0)
            .unwrap();
    let probe_v2 = build_v2_dnck_checkpoint(
        &ckpt_hdr_json,
        &rt_dnam,
        &[],
        &v2_b_refs,
        &pk,
        &[0u8; 64],
        None,
    )
    .unwrap();
    let a_off_v2 = u64::from_le_bytes(probe_v2[96..104].try_into().unwrap()) as usize;
    let pfxs_v2: Vec<[u8; 32]> = (0..4)
        .map(|i| {
            probe_v2[a_off_v2 + 43 + 33 * i..a_off_v2 + 75 + 33 * i]
                .try_into()
                .unwrap()
        })
        .collect();
    let sig_v2 = RustCryptoProvider::ed25519_sign(
        &seed,
        &build_dnck_writer_manifest_message(&pk, &sha256_digest(&ckpt_hdr_json), 3, &pfxs_v2),
    );
    let signed_v2 = build_v2_dnck_checkpoint(
        &ckpt_hdr_json,
        &rt_dnam,
        &[],
        &v2_b_refs,
        &pk,
        &sig_v2,
        None,
    )
    .unwrap();
    let crypto = RustCryptoProvider::default();
    let v2_loaded = dnam_trainer::ckpt::DnamCkptView::from_bytes(&signed_v2, &crypto).unwrap();
    assert!(v2_loaded.preamble.has_calib_stats());

    // Corrupted per-bundle DNCA payload CRC must fail DnamCkptView::from_bytes even with valid DCA1 sig
    let mut bad_b0 = b0.clone();
    bad_b0[128 + 16] ^= 0xFF;
    let bad_b_refs = [
        bad_b0.as_slice(),
        b_rest[0].as_slice(),
        b_rest[1].as_slice(),
        b_rest[2].as_slice(),
    ];
    let bad_probe = build_v2_dnck_checkpoint(
        &ckpt_hdr_json,
        &rt_dnam,
        &[],
        &bad_b_refs,
        &pk,
        &[0u8; 64],
        None,
    )
    .unwrap();
    let bad_pfxs: Vec<[u8; 32]> = (0..4)
        .map(|i| {
            bad_probe[a_off_v2 + 43 + 33 * i..a_off_v2 + 75 + 33 * i]
                .try_into()
                .unwrap()
        })
        .collect();
    let bad_sig_v2 = RustCryptoProvider::ed25519_sign(
        &seed,
        &build_dnck_writer_manifest_message(&pk, &sha256_digest(&ckpt_hdr_json), 3, &bad_pfxs),
    );
    let signed_bad_dnca = build_v2_dnck_checkpoint(
        &ckpt_hdr_json,
        &rt_dnam,
        &[],
        &bad_b_refs,
        &pk,
        &bad_sig_v2,
        None,
    )
    .unwrap();
    assert!(dnam_trainer::ckpt::DnamCkptView::from_bytes(&signed_bad_dnca, &crypto).is_err());

    // Kaiser 309-tap decimator NaN & f32::MAX overflow guard
    let mut wild_192k = vec![1.0e40_f64; 32];
    wild_192k[4] = f64::NAN;
    let dec = dnam_trainer::whitebox::kaiser_sinc_decimate_4x_to_48khz(&wild_192k);
    assert_eq!(dec.len(), 8);
    assert!(dec.iter().all(|v| v.is_finite()));

    // v2 DnamCkptView controls_schema parity & stems_registry slice validation guards
    let sign_v2_hdr = |hdr_bytes: &[u8]| -> Vec<u8> {
        let p =
            build_v2_dnck_checkpoint(hdr_bytes, &rt_dnam, &[], &v2_b_refs, &pk, &[0u8; 64], None)
                .unwrap();
        let ao = u64::from_le_bytes(p[96..104].try_into().unwrap()) as usize;
        let pf: Vec<[u8; 32]> = (0..4)
            .map(|i| p[ao + 43 + 33 * i..ao + 75 + 33 * i].try_into().unwrap())
            .collect();
        let sg = RustCryptoProvider::ed25519_sign(
            &seed,
            &build_dnck_writer_manifest_message(&pk, &sha256_digest(hdr_bytes), 3, &pf),
        );
        build_v2_dnck_checkpoint(hdr_bytes, &rt_dnam, &[], &v2_b_refs, &pk, &sg, None).unwrap()
    };
    let mut bad_ctrl_hdr = none_reg.clone();
    bad_ctrl_hdr.base.controls_schema.push(
        serde_json::from_value(serde_json::json!({
            "id": "gain", "name": "Gain", "type": "range",
            "lobatto_degree": 0, "declared_range": [0.0, 10.0],
            "safe_clamped_range": [0.0, 10.0], "default": 5.0
        }))
        .unwrap(),
    );
    let bad_ctrl_v2 = sign_v2_hdr(&serde_json::to_vec(&bad_ctrl_hdr).unwrap());
    assert!(dnam_trainer::ckpt::DnamCkptView::from_bytes(&bad_ctrl_v2, &crypto).is_err());

    for (arena_id, off, len) in [(0_u8, 256_u32, 64_u32), (4, 260, 64), (4, 8_192, 64)] {
        let mut bad_s_hdr = ckpt_hdr.clone();
        let e = &mut bad_s_hdr.stems_registry.as_mut().unwrap()[0];
        e.g_audio_uplo_u_f64.arena = arena_id;
        e.g_audio_uplo_u_f64.byte_offset = off;
        e.g_audio_uplo_u_f64.byte_length = len;
        let bad_s_v2 = sign_v2_hdr(&serde_json::to_vec(&bad_s_hdr).unwrap());
        assert!(dnam_trainer::ckpt::DnamCkptView::from_bytes(&bad_s_v2, &crypto).is_err());
    }

    // Valid f64/f32 values whose low 32 bits equal b"DNCA" (0x4143_4E44) must NOT trigger false DNCA splits
    let mut magic_f32_arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    magic_f32_arena[10_752..10_756].copy_from_slice(b"DNCA"); // +12.206608_f32 at Arena 1 start
    let mut stemless_hdr = none_reg.clone();
    let std_magic_ckpt =
        build_dnam_ckpt_container(&mut stemless_hdr, &magic_f32_arena, &magic_f32_arena, false)
            .unwrap();
    let std_magic_view =
        dnam_trainer::ckpt::DnamCkptView::from_bytes(&std_magic_ckpt, &crypto).unwrap();
    assert!(!std_magic_view.preamble.has_calib_stats());

    let mut b0_magic_f64 = vec![0u8; 128];
    b0_magic_f64[64..72].copy_from_slice(&0x3FF0_0000_4143_4E44_u64.to_le_bytes());
    let mf_refs = [
        b0_magic_f64.as_slice(),
        b_rest[0].as_slice(),
        b_rest[1].as_slice(),
        b_rest[2].as_slice(),
    ];
    let mf_probe = build_v2_dnck_checkpoint(
        &ckpt_hdr_json,
        &rt_dnam,
        &[],
        &mf_refs,
        &pk,
        &[0u8; 64],
        None,
    )
    .unwrap();
    let mf_ao = u64::from_le_bytes(mf_probe[96..104].try_into().unwrap()) as usize;
    let mf_pfxs: Vec<[u8; 32]> = (0..4)
        .map(|i| {
            mf_probe[mf_ao + 43 + 33 * i..mf_ao + 75 + 33 * i]
                .try_into()
                .unwrap()
        })
        .collect();
    let mf_sig = RustCryptoProvider::ed25519_sign(
        &seed,
        &build_dnck_writer_manifest_message(&pk, &sha256_digest(&ckpt_hdr_json), 3, &mf_pfxs),
    );
    let mf_ckpt =
        build_v2_dnck_checkpoint(&ckpt_hdr_json, &rt_dnam, &[], &mf_refs, &pk, &mf_sig, None)
            .unwrap();
    let mf_view = dnam_trainer::ckpt::DnamCkptView::from_bytes(&mf_ckpt, &crypto).unwrap();
    assert!(!mf_view.preamble.has_calib_stats());

    // PCHIP extreme finite-span ([-f64::MAX, f64::MAX]) & infinite target_rate_hz finite-output guards
    let pchip_2 = dnam_trainer::whitebox::pchip_resample_to_rate(
        &[f64::MIN, 1.0],
        &[-f64::MAX, f64::MAX],
        4,
        f64::INFINITY,
    );
    let pchip_3 = dnam_trainer::whitebox::pchip_resample_to_rate(
        &[-1.0e308, 1.0e308, 1.5e308],
        &[0.0, 0.9 * f64::MAX, f64::MAX],
        4,
        1.0,
    );
    assert!(pchip_2.iter().chain(&pchip_3).all(|v| v.is_finite()));

    // Stripped v2 .dnam-ckpt rebases stems_registry Arena 4 offsets, drops higher-tier stems,
    // rejects pre-stripped runtime templates, and enforces distribution_policy
    let mut s0_tail = vec![0u8; 128];
    s0_tail[0..8].copy_from_slice(&1.25_f64.to_le_bytes());
    let mut s1_tail = vec![0u8; 128];
    s1_tail[0..8].copy_from_slice(&2.50_f64.to_le_bytes());
    let mk_stem = |id: &str, base_off: u32, payload: &[u8]| StemRegistryEntry {
        stem_id: id.to_string(),
        enabled: true,
        controls_meta: BTreeMap::new(),
        provenance: StemProvenance {
            input_pcm_sha256: "00".repeat(32),
            target_pcm_sha256: "11".repeat(32),
            num_frames: 48_000,
        },
        stem_arena4_crc32c: crc32c_compute(payload),
        user_metadata: BTreeMap::new(),
        extracted_physics: None,
        g_audio_uplo_u_f64: TensorSliceEntry {
            arena: 4,
            dtype: String::from("f64"),
            packing: Some(String::from("uplo_upper_col_major_tier_ordered")),
            feature_dim: Some(8),
            packed_len: Some(8),
            shape: Some(vec![8]),
            byte_offset: base_off,
            byte_length: 64,
        },
        c_audio_tier_f64: TensorSliceEntry {
            arena: 4,
            dtype: String::from("f64"),
            packing: None,
            feature_dim: Some(8),
            packed_len: Some(8),
            shape: Some(vec![8]),
            byte_offset: base_off + 64,
            byte_length: 64,
        },
    };
    let mut two_stem_hdr = DnamCkptHeader::default_reference();
    two_stem_hdr.stems_registry = Some(vec![
        mk_stem("s0_lite", 256, &s0_tail),
        mk_stem("s1_pro", 384, &s1_tail),
    ]);
    let ts_json = serde_json::to_vec(&two_stem_hdr).unwrap();
    let ts_b0: Vec<u8> = [vec![0u8; 64], s0_tail.clone()].concat();
    let ts_b1 = vec![0u8; 64];
    let ts_b2: Vec<u8> = [vec![0u8; 64], s1_tail.clone()].concat();
    let ts_b3 = vec![0u8; 64];
    let ts_refs = [
        ts_b0.as_slice(),
        ts_b1.as_slice(),
        ts_b2.as_slice(),
        ts_b3.as_slice(),
    ];
    let sign_custom = |hdr_bytes: &[u8], refs: &[&[u8]; 4]| -> Vec<u8> {
        let pr = build_v2_dnck_checkpoint(hdr_bytes, &rt_dnam, &[], refs, &pk, &[0u8; 64], None)
            .unwrap();
        let ao = u64::from_le_bytes(pr[96..104].try_into().unwrap()) as usize;
        let pfxs: Vec<[u8; 32]> = (0..4)
            .map(|i| pr[ao + 43 + 33 * i..ao + 75 + 33 * i].try_into().unwrap())
            .collect();
        let s = RustCryptoProvider::ed25519_sign(
            &seed,
            &build_dnck_writer_manifest_message(&pk, &sha256_digest(hdr_bytes), 3, &pfxs),
        );
        build_v2_dnck_checkpoint(hdr_bytes, &rt_dnam, &[], refs, &pk, &s, None).unwrap()
    };
    let ts_ckpt = sign_custom(&ts_json, &ts_refs);
    let ts_stripped_full =
        dnam_trainer::ckpt::strip_dnck_checkpoint_to_tier(&ts_ckpt, dnam_engine::Tier::Full)
            .unwrap();
    let ts_view = dnam_trainer::ckpt::DnamCkptView::from_bytes(&ts_stripped_full, &crypto).unwrap();
    let retained_stems = ts_view.header.stems_registry.as_ref().unwrap();
    assert_eq!(retained_stems.len(), 1);
    assert_eq!(retained_stems[0].stem_id, "s0_lite");
    assert_eq!(retained_stems[0].g_audio_uplo_u_f64.byte_offset, 128);
    assert_eq!(retained_stems[0].c_audio_tier_f64.byte_offset, 192);
    assert_eq!(
        &ts_view.arena_storage.as_bytes()[128..256],
        s0_tail.as_slice()
    );

    // Malformed span_start < max_weights_end must still be rejected even after stripping
    let mut overlap_hdr = DnamCkptHeader::default_reference();
    overlap_hdr.stems_registry = Some(vec![mk_stem("bad_overlap", 192, &s0_tail)]);
    let ov_ckpt = sign_custom(&serde_json::to_vec(&overlap_hdr).unwrap(), &ts_refs);
    let ov_stripped =
        dnam_trainer::ckpt::strip_dnck_checkpoint_to_tier(&ov_ckpt, dnam_engine::Tier::Lite)
            .unwrap();
    assert!(dnam_trainer::ckpt::DnamCkptView::from_bytes(&ov_stripped, &crypto).is_err());

    // Pre-stripped runtime .dnam template must be rejected by build_v2_dnck_checkpoint
    let pre_stripped_rt =
        dnam_engine::container::strip_container_to_tier(&rt_dnam, dnam_engine::Tier::Full).unwrap();
    assert!(
        build_v2_dnck_checkpoint(
            &ts_json,
            &pre_stripped_rt,
            &[],
            &[ts_b0.as_slice(), ts_b1.as_slice()],
            &pk,
            &[0u8; 64],
            None,
        )
        .is_err()
    );

    // EnclaveSealed-only distribution_policy on v2 .dnam-ckpt must be rejected by DnamCkptView
    let mut enclave_hdr = DnamCkptHeader::default_reference();
    enclave_hdr.base.metadata.distribution_policy = Some(dnam_engine::schema::DistributionPolicy {
        allowed_protection_profiles: vec![
            dnam_engine::schema::DnamProtectionProfile::EnclaveSealed,
        ],
        allowed_oem_vendors: Vec::new(),
    });
    let enclave_ckpt = sign_v2_hdr(&serde_json::to_vec(&enclave_hdr).unwrap());
    assert!(matches!(
        dnam_trainer::ckpt::DnamCkptView::from_bytes(&enclave_ckpt, &crypto),
        Err(dnam_engine::schema::DnamError::CryptoVerificationFailed(_))
    ));

    // 1-ULP IEEE-754 binade boundary in evaluate_joint_energy_interval_audit keeps c_minus <= c_plus
    let eps = f64::EPSILON;
    let req_snr = 64.0 + 64.0 * eps;
    let goal_db = -10.0 * req_snr.log10();
    let f_val = 2.0 - 2.0 * eps;
    let y_val = 128.0 - 64.0 * eps;
    let r_val = 2.2;
    let audit = dnam_trainer::metrics::evaluate_joint_energy_interval_audit(
        [[y_val, y_val]; 3],
        [[r_val, r_val]; 3],
        [[f_val, f_val]; 3],
        [goal_db; 3],
        goal_db,
    );
    for b in 0..3 {
        assert!(audit.qualified_bounds_lin[b][0] <= audit.qualified_bounds_lin[b][1]);
    }

    // 1. Canonical v2.0 .dnam-ckpt export & stem-bearing strip must preserve CP1 contributor weights
    let range_ctrl: dnam_engine::schema::ControlSchemaDef =
        serde_json::from_value(serde_json::json!({
            "id": "gain", "name": "Gain", "type": "range",
            "lobatto_degree": 0, "declared_range": [0.0, 10.0],
            "safe_clamped_range": [0.0, 10.0], "default": 5.0
        }))
        .unwrap();
    let mut proj_hdr = ckpt_hdr.clone();
    proj_hdr.base.controls_schema = vec![range_ctrl.clone()];
    let mut expected_cp1 = dnam_engine::control::Cp1ContributorSlot::zero();
    expected_cp1.lanes = 1;
    expected_cp1.ctrl_kind = 0;
    expected_cp1.control_idx = 0;
    expected_cp1.option_idx = 0xFF;
    expected_cp1.main_base[0] = [0.75; 8];
    expected_cp1.emb = [1.0, 0.0, 0.0];
    expected_cp1.v_base = [-0.25; 4];
    let proj_ckpt = dnam_trainer::ckpt::build_dnam_ckpt_container_with_cp1(
        &mut proj_hdr,
        &arena,
        &arena,
        false,
        &[expected_cp1],
    )
    .unwrap();
    let exported_std = export_runtime_dnam_from_ckpt(&proj_ckpt).unwrap();
    let exp_view =
        dnam_engine::container::DnamContainerView::from_bytes(&exported_std, &crypto).unwrap();
    assert_eq!(exp_view.cp1_contributors, vec![expected_cp1]);
    let stripped_proj_std = strip_dnck_checkpoint_to_tier(&proj_ckpt, Tier::Pro).unwrap();
    let strip_view =
        dnam_trainer::ckpt::DnamCkptView::from_bytes(&stripped_proj_std, &crypto).unwrap();
    assert_eq!(strip_view.cp1_contributors, vec![expected_cp1]);

    // 2. stems_registry > 65,535 entries must fail DnamCkptHeader::validate
    let mut huge_stems_hdr = ckpt_hdr.clone();
    let sample_entry = huge_stems_hdr.stems_registry.as_ref().unwrap()[0].clone();
    huge_stems_hdr.stems_registry = Some(vec![sample_entry; 65_536]);
    assert!(huge_stems_hdr.validate().is_err());

    // 3. powf underflow/overflow & infinite ESR in evaluate_joint_energy_interval_audit / noise-qualified bESR
    let pass_underflow = dnam_trainer::metrics::evaluate_joint_energy_interval_audit(
        [[0.0, 0.0]; 3],
        [[1.2e-6, 1.2e-6]; 3],
        [[1.0e-6, 1.0e-6]; 3],
        [3500.0; 3],
        -20.0,
    );
    assert_eq!(
        pass_underflow.decision,
        dnam_trainer::metrics::IntervalAuditDecision::CertifiedPass
    );
    let pass_overflow = dnam_trainer::metrics::evaluate_joint_energy_interval_audit(
        [[1.0e200, 1.0e200]; 3],
        [[1.2e-120, 1.2e-120]; 3],
        [[1.0e-120, 1.0e-120]; 3],
        [-3500.0; 3],
        -3500.0,
    );
    assert_eq!(
        pass_overflow.decision,
        dnam_trainer::metrics::IntervalAuditDecision::CertifiedPass
    );
    let inf_err_fail = dnam_trainer::metrics::evaluate_joint_energy_interval_audit(
        [[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]],
        [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]],
        [[0.0, 0.0]; 3],
        [3500.0; 3],
        3500.0,
    );
    assert_eq!(
        inf_err_fail.decision,
        dnam_trainer::metrics::IntervalAuditDecision::CertifiedFail
    );

    // 4. PCHIP straight-line slope preservation across wide [-0.9 * f64::MAX, 0.9 * f64::MAX] knots
    let wide_lin = dnam_trainer::whitebox::pchip_resample_to_rate(
        &[-0.71 * f64::MAX, 0.0, 0.71 * f64::MAX],
        &[-0.994 * f64::MAX, 0.0, 0.994 * f64::MAX],
        4,
        1.0,
    );
    assert!((wide_lin[1] - 1.4).abs() < 1.0e-9);

    // 5. FreezeLatch calibrated control is preserved across render_stem with None / Some([0, 0]) freeze window
    let mut frz_hdr = dnam_engine::schema::DnamJsonHeader::default_reference();
    frz_hdr.io_config.capabilities.supports_freeze = true;
    frz_hdr.controls_schema = vec![range_ctrl];
    frz_hdr.controls_schema[0].calibration = Some(dnam_engine::schema::ControlCalibration {
        role: dnam_engine::schema::ControlCalibrationRole::FreezeLatch,
        lobatto_physical_anchors: None,
        conditional_physical_anchors: BTreeMap::new(),
        tempo_sync_default_subdivision: None,
    });
    let frz_arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    let frz_dnam =
        dnam_engine::container::build_dnam_container(&mut frz_hdr, &frz_arena, &frz_arena, false)
            .unwrap();
    let frz_view =
        dnam_engine::container::DnamContainerView::from_bytes(&frz_dnam, &crypto).unwrap();
    let mut frz_eng = dnam_engine::DnamEngine::<dnam_engine::math::DefaultMathBackend>::from_bytes(
        &frz_dnam,
        Tier::Lite,
        48_000,
    )
    .unwrap();
    let max_c0 = frz_view.header.controls_schema[0]
        .declared_range
        .unwrap_or([0.0, 10.0])[1];
    let frz_stim = dnam_trainer::StemStimulus {
        stem_id: String::from("frz_latch"),
        sample_rate_hz: 48_000,
        in_1: vec![0.25; 16],
        in_2: None,
        in_3: None,
        in_4: None,
        control_values: vec![max_c0, 5.0],
        control_trajectories: Some(vec![vec![max_c0; 16], vec![5.0; 16]]),
        freeze_transition_window: Some([0, 0]),
    };
    use dnam_trainer::DistillationTarget as _;
    let _ = frz_eng.render_stem(&frz_stim).unwrap();
    assert!(frz_eng.stage_pipeline.stage4.freeze);

    // 6. Canonical v2.0 DnamCkptView rejects tampered preamble sample_rate_train_hz / DCA1 & header-corrupt DNCA
    let mut bad_std_rate = std_ckpt.clone();
    bad_std_rate[12..16].copy_from_slice(&44_100u32.to_le_bytes());
    assert!(dnam_trainer::ckpt::DnamCkptView::from_bytes(&bad_std_rate, &crypto).is_err());
    let mut bad_std_sig = std_ckpt;
    let a_off_std = u64::from_le_bytes(bad_std_sig[64 + 32..64 + 40].try_into().unwrap()) as usize;
    let a_len_std = u64::from_le_bytes(bad_std_sig[64 + 40..64 + 48].try_into().unwrap()) as usize;
    bad_std_sig[a_off_std + a_len_std - 1] ^= 0xFF;
    assert!(dnam_trainer::ckpt::DnamCkptView::from_bytes(&bad_std_sig, &crypto).is_err());

    let mut bad_dnca_hdr_b0 = b0.clone();
    bad_dnca_hdr_b0[128 + 6] = 1; // non-zero DNCA flags in per-bundle Arena 5 header
    let bad_hdr_refs = [
        bad_dnca_hdr_b0.as_slice(),
        b_rest[0].as_slice(),
        b_rest[1].as_slice(),
        b_rest[2].as_slice(),
    ];
    let bad_hdr_probe = build_v2_dnck_checkpoint(
        &ckpt_hdr_json,
        &rt_dnam,
        &[],
        &bad_hdr_refs,
        &pk,
        &[0u8; 64],
        None,
    )
    .unwrap();
    let bad_hdr_pfxs: Vec<[u8; 32]> = (0..4)
        .map(|i| {
            bad_hdr_probe[a_off_v2 + 43 + 33 * i..a_off_v2 + 75 + 33 * i]
                .try_into()
                .unwrap()
        })
        .collect();
    let bad_hdr_sig = RustCryptoProvider::ed25519_sign(
        &seed,
        &build_dnck_writer_manifest_message(&pk, &sha256_digest(&ckpt_hdr_json), 3, &bad_hdr_pfxs),
    );
    let signed_bad_dnca_hdr = build_v2_dnck_checkpoint(
        &ckpt_hdr_json,
        &rt_dnam,
        &[],
        &bad_hdr_refs,
        &pk,
        &bad_hdr_sig,
        None,
    )
    .unwrap();
    assert!(dnam_trainer::ckpt::DnamCkptView::from_bytes(&signed_bad_dnca_hdr, &crypto).is_err());
}
