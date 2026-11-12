#![forbid(unsafe_code)]
//! Fail-fast framing, `AUTH` attestation, `compact_to_capsules`, `export_runtime_dnam`, and
//! stereo / star-promotion Capture Vault mutation edge-case tests (`proj::{directory, bundle, vault}`).

use std::collections::BTreeMap;

use dnam_capture::proj::{
    CalbChunk64B, DnamProjManifest, DnamProjPreamble, DnamProjView, PROJ_AUTH_RECORD_BYTES,
    PROJ_FLAG_COMPACTED_CAPSULES, PROJ_FLAG_HAS_CALIBRATION, ProjAuthRecord, ProjChunkDirEntry,
    ProjChunkInput, ProjChunkKind, StemProvenance, StmcTrajectoryWindow, VaultCaptureSpec,
    add_capture, attach_proj_auth_signature, build_dnam_proj_container,
    build_dnam_proj_container_with_cp1, build_proj_inpc_capsule, compact_to_capsules,
    compute_proj_header_crc32c, export_runtime_dnam, f32_to_fp16_rne_bytes,
    parse_proj_manifest_range, remove_capture, replace_capture, set_capture_enabled,
    verify_arena_b_head_zero_dc, zero_arena_b_head_dc_bias,
};
use dnam_engine::Tier;
use dnam_engine::container::{ARENA_OFFSET_LITE_END, ARENA_OFFSET_ULTRA_END};
use dnam_engine::provenance::{DefaultCryptoProvider, RustCryptoProvider};
use dnam_engine::schema::{ControlSchemaDef, PresetDef};

#[test]
fn test_proj_directory_bundle_auth_and_export_edges() {
    let crypto = DefaultCryptoProvider::default();

    assert!(ProjChunkKind::from_fourcc(*b"ZZZZ").is_err());
    assert!(DnamProjPreamble::from_bytes(&[0; 32]).is_err());
    assert!(ProjChunkDirEntry::from_bytes(&[0; 16]).is_err());
    assert!(ProjAuthRecord::from_bytes(&[0; 32]).is_err());

    let base_proj = build_dnam_proj_container(&DnamProjManifest::default_reference(), &[]).unwrap();
    let valid_pre = DnamProjPreamble::from_bytes(&base_proj[..64]).unwrap();

    let mut pre = valid_pre;
    pre.magic = *b"DNAM";
    assert!(DnamProjPreamble::from_bytes(&pre.to_bytes()).is_err());

    let mut pre = valid_pre;
    pre.version_major = 1;
    assert!(DnamProjPreamble::from_bytes(&pre.to_bytes()).is_err());

    let mut pre = valid_pre;
    pre.lifecycle_flags = 0x8000_0000;
    assert!(DnamProjPreamble::from_bytes(&pre.to_bytes()).is_err());

    let mut pre = valid_pre;
    pre.chunk_count = 9_999;
    assert!(DnamProjPreamble::from_bytes(&pre.to_bytes()).is_err());

    let mut pre = valid_pre;
    pre.reserved[3] = 1;
    assert!(DnamProjPreamble::from_bytes(&pre.to_bytes()).is_err());

    let mut pre = valid_pre;
    pre.chunk_dir_offset = 128;
    assert!(DnamProjPreamble::from_bytes(&pre.to_bytes()).is_err());

    let mut pre = valid_pre;
    pre.manifest_offset = 128;
    assert!(DnamProjPreamble::from_bytes(&pre.to_bytes()).is_err());

    let mut pre = valid_pre;
    pre.manifest_len = 0;
    assert!(DnamProjPreamble::from_bytes(&pre.to_bytes()).is_err());

    let valid_entry = ProjChunkDirEntry {
        fourcc: *b"CALB",
        chunk_id: 0,
        flags: 0,
        crc32c: 0,
        offset: 512,
        byte_len: 64,
        sha256: [0; 32],
    };
    let mut e = valid_entry;
    e.chunk_id = 1;
    assert!(ProjChunkDirEntry::from_bytes(&e.to_bytes()).is_err());

    let mut e = valid_entry;
    e.flags = 1;
    assert!(ProjChunkDirEntry::from_bytes(&e.to_bytes()).is_err());

    let mut e = valid_entry;
    e.offset = 513;
    assert!(ProjChunkDirEntry::from_bytes(&e.to_bytes()).is_err());

    let mut e = valid_entry;
    e.byte_len = 0;
    assert!(ProjChunkDirEntry::from_bytes(&e.to_bytes()).is_err());

    let mut auth_bytes = (ProjAuthRecord {
        signer_pubkey_ed25519: [1; 32],
        signature_ed25519: [2; 64],
    })
    .to_bytes();
    assert_eq!(auth_bytes.len(), PROJ_AUTH_RECORD_BYTES);
    auth_bytes[0..4].copy_from_slice(b"NOPE");
    assert!(ProjAuthRecord::from_bytes(&auth_bytes).is_err());
    auth_bytes[0..4].copy_from_slice(b"AUTH");
    auth_bytes[4..6].copy_from_slice(&1_u16.to_le_bytes());
    assert!(ProjAuthRecord::from_bytes(&auth_bytes).is_err());
    auth_bytes[4..6].copy_from_slice(&2_u16.to_le_bytes());
    auth_bytes[10] = 1;
    assert!(ProjAuthRecord::from_bytes(&auth_bytes).is_err());

    assert!(parse_proj_manifest_range(&base_proj[..80]).is_err());
    let mut pad_corrupt = base_proj.clone();
    let last_idx = pad_corrupt.len() - 1;
    pad_corrupt[last_idx] = 0xFF;
    assert!(parse_proj_manifest_range(&pad_corrupt).is_err());

    let mut crc_corrupt = base_proj;
    crc_corrupt[40] ^= 0xFF;
    assert!(parse_proj_manifest_range(&crc_corrupt).is_err());

    let pretty_json = serde_json::to_vec_pretty(&DnamProjManifest::default_reference()).unwrap();
    let padded_m_end = 64 + ((pretty_json.len() + 63) & !63);
    let mut non_jcs = vec![0u8; padded_m_end];
    let mut p = valid_pre;
    p.manifest_len = pretty_json.len() as u64;
    let p_bytes = p.to_bytes();
    p.header_crc32c = compute_proj_header_crc32c(&p_bytes[..40], &[], &pretty_json);
    non_jcs[..64].copy_from_slice(&p.to_bytes());
    non_jcs[64..64 + pretty_json.len()].copy_from_slice(&pretty_json);
    assert!(parse_proj_manifest_range(&non_jcs).is_err());

    let mut spcm_68 = vec![0u8; 68];
    spcm_68[0..4].copy_from_slice(b"SPCM");
    let proj_with_spcm = build_dnam_proj_container(
        &DnamProjManifest::default_reference(),
        &[ProjChunkInput {
            kind: ProjChunkKind::Spcm,
            chunk_id: 0,
            payload: spcm_68,
        }],
    )
    .unwrap();
    assert!(
        DnamProjView::from_bytes(&proj_with_spcm[..proj_with_spcm.len() - 64], &crypto).is_err()
    );

    let mut extra_tail = proj_with_spcm.clone();
    extra_tail.extend_from_slice(&[0u8; 64]);
    assert!(DnamProjView::from_bytes(&extra_tail, &crypto).is_err());

    let mut bad_chunk_pad = proj_with_spcm.clone();
    let pad_byte = bad_chunk_pad.len() - 1;
    bad_chunk_pad[pad_byte] = 1;
    assert!(DnamProjView::from_bytes(&bad_chunk_pad, &crypto).is_err());

    let mut bad_chunk_crc = proj_with_spcm;
    let payload_off = bad_chunk_crc.len() - 128;
    bad_chunk_crc[payload_off + 16] ^= 0xFF;
    assert!(DnamProjView::from_bytes(&bad_chunk_crc, &crypto).is_err());

    assert!(
        build_dnam_proj_container(
            &DnamProjManifest::default_reference(),
            &[ProjChunkInput {
                kind: ProjChunkKind::Spcm,
                chunk_id: 0,
                payload: Vec::new(),
            }],
        )
        .is_err()
    );
    assert!(
        build_dnam_proj_container(
            &DnamProjManifest::default_reference(),
            &[ProjChunkInput {
                kind: ProjChunkKind::Spcm,
                chunk_id: 0,
                payload: vec![0u8; 64],
            }],
        )
        .is_err()
    );

    assert!(verify_arena_b_head_zero_dc(&[0u8; 1024]).is_err());
    let mut arena_lite = vec![0u8; ARENA_OFFSET_LITE_END as usize];
    arena_lite[9_024] = 1;
    assert!(verify_arena_b_head_zero_dc(&arena_lite).is_err());
    zero_arena_b_head_dc_bias(&mut arena_lite);
    assert!(verify_arena_b_head_zero_dc(&arena_lite).is_ok());

    let mut arena_ultra = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    arena_ultra[63_424] = 1;
    assert!(verify_arena_b_head_zero_dc(&arena_ultra).is_err());
    zero_arena_b_head_dc_bias(&mut arena_ultra);
    assert!(verify_arena_b_head_zero_dc(&arena_ultra).is_ok());

    let seed = [0x42_u8; 32];
    let vk_bytes = RustCryptoProvider::ed25519_pubkey_from_seed(&seed);

    let mut m_wght = DnamProjManifest::default_reference();
    let proj_wght = build_dnam_proj_container_with_cp1(
        &mut m_wght,
        &[ProjChunkInput {
            kind: ProjChunkKind::Spcm,
            chunk_id: 0,
            payload: {
                let mut v = vec![0u8; 64];
                v[0..4].copy_from_slice(b"SPCM");
                v
            },
        }],
        &arena_ultra,
        &[],
    )
    .unwrap();
    let signed_uncompacted = attach_proj_auth_signature(
        &proj_wght,
        vk_bytes,
        |msg| RustCryptoProvider::ed25519_sign(&seed, msg),
        &crypto,
    )
    .unwrap();
    let compacted_once = compact_to_capsules(&signed_uncompacted).unwrap();
    let v_once = DnamProjView::from_bytes(&compacted_once, &crypto).unwrap();
    assert!(v_once.auth_record.is_none());
    assert_ne!(
        v_once.preamble.lifecycle_flags & PROJ_FLAG_COMPACTED_CAPSULES,
        0
    );

    let signed_compacted = attach_proj_auth_signature(
        &compacted_once,
        vk_bytes,
        |msg| RustCryptoProvider::ed25519_sign(&seed, msg),
        &crypto,
    )
    .unwrap();
    let compacted_twice = compact_to_capsules(&signed_compacted).unwrap();
    let v_twice = DnamProjView::from_bytes(&compacted_twice, &crypto).unwrap();
    assert!(v_twice.auth_record.is_some());

    let mut m_updated = v_twice.manifest.clone();
    m_updated.base.user_metadata.insert(
        "artist_tag".to_string(),
        serde_json::json!("Phase 2 Verification"),
    );
    m_updated.base.presets.push(PresetDef {
        id: "crunch_lead".to_string(),
        name: "Crunch Lead".to_string(),
        description: None,
        controls: BTreeMap::new(),
        user_metadata: BTreeMap::new(),
    });
    let wght_slice = v_twice
        .chunk_slice(ProjChunkKind::Wght, 0)
        .unwrap()
        .to_vec();
    let rebuilt_with_preset = build_dnam_proj_container(
        &m_updated,
        &[ProjChunkInput {
            kind: ProjChunkKind::Wght,
            chunk_id: 0,
            payload: wght_slice,
        }],
    )
    .unwrap();
    let slim_lite = export_runtime_dnam(&rebuilt_with_preset, Some(Tier::Lite)).unwrap();
    let rt_view =
        dnam_engine::container::DnamContainerView::from_bytes(&slim_lite, &crypto).unwrap();
    assert_eq!(rt_view.preamble.highest_tier(), Tier::Lite);
    assert_eq!(rt_view.header.presets.len(), 1);
    assert_eq!(
        rt_view.header.user_metadata.get("artist_tag"),
        Some(&serde_json::json!("Phase 2 Verification"))
    );
}

#[test]
fn test_proj_vault_stereo_and_mutation_error_and_star_promotion_edges() {
    let crypto = DefaultCryptoProvider::default();
    let valid_sha = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.io_config.output_mode = "stereo".to_string();
    let drive_ctrl: ControlSchemaDef = serde_json::from_value(serde_json::json!({
        "id": "drive",
        "name": "Drive",
        "type": "range",
        "lobatto_degree": 2,
        "declared_range": [0.0, 10.0],
        "safe_clamped_range": [0.0, 10.0],
        "default": 5.0
    }))
    .unwrap();
    manifest.base.controls_schema = vec![drive_ctrl];

    let calb = (CalbChunk64B {
        sample_rate_device_hz: 48_000,
        bit_depth_code: 2,
        polarity_inverted: false,
        loopback_lag_samples: 0.0,
        signed_gain: 1.0,
        input_reference_dbfs: -20.5,
        noise_floor_dbfs: -96.0,
    })
    .to_bytes()
    .to_vec();
    let empty_proj = build_dnam_proj_container(
        &manifest,
        &[ProjChunkInput {
            kind: ProjChunkKind::Calb,
            chunk_id: 0,
            payload: calb,
        }],
    )
    .unwrap();

    let stereo_samples: Vec<f32> = (0..24_000)
        .map(|i| 0.25 * ((i as f32) * 0.017).sin())
        .collect();
    let stereo_fp16 = f32_to_fp16_rne_bytes(&stereo_samples);
    let inpc_stereo = build_proj_inpc_capsule(2, &[0x33; 256], &[0x44; 320], &stereo_fp16).unwrap();

    let mk_stereo_spec = |id: &str, star: bool, drive_val: f64, with_spcm: bool| {
        let mut meta = BTreeMap::new();
        meta.insert("drive".to_string(), serde_json::json!(drive_val));
        let spcm = if with_spcm {
            let mut s = vec![0u8; 64];
            s[0..4].copy_from_slice(b"SPCM");
            Some(s)
        } else {
            None
        };
        VaultCaptureSpec {
            stem_id: id.to_string(),
            enabled: true,
            is_star_center: star,
            pad_gain_f32: 1.0,
            fractional_lag_samples_f32: 0.0,
            band_psd_summary_88: [0u8; 88],
            n_comb_bins: 0,
            windows: vec![StmcTrajectoryWindow {
                control_id: 0,
                ctrl_kind: 0,
                active_opt: 0,
                u_phys: drive_val as f32,
                window_weight: 1.0,
                emr_velocity: 0.0,
            }],
            spectral_summary_4096: [0u8; 4_096],
            wet_coreset_fp16: stereo_fp16.clone(),
            controls_meta: meta,
            provenance: StemProvenance {
                input_pcm_sha256: valid_sha.to_string(),
                target_pcm_sha256: valid_sha.to_string(),
                num_frames: 12_000,
            },
            user_metadata: BTreeMap::new(),
            extracted_physics: None,
            optional_spcm: spcm,
        }
    };

    assert!(add_capture(&empty_proj, mk_stereo_spec("s0", true, 5.0, true), None).is_err());

    let p1 = add_capture(
        &empty_proj,
        mk_stereo_spec("s0", true, 5.0, true),
        Some(&inpc_stereo),
    )
    .unwrap();

    assert!(add_capture(&p1, mk_stereo_spec("s0", false, 2.0, false), None).is_err());
    assert!(remove_capture(&p1, "missing_id").is_err());
    assert!(replace_capture(&p1, "missing_id", mk_stereo_spec("x", false, 2.0, false)).is_err());
    assert!(set_capture_enabled(&p1, "missing_id", false).is_err());

    let p2 = add_capture(&p1, mk_stereo_spec("s1", false, 2.0, false), None).unwrap();
    let p3 = add_capture(&p2, mk_stereo_spec("s2", true, 8.0, true), None).unwrap();
    let v3 = DnamProjView::from_bytes(&p3, &crypto).unwrap();
    assert!(!v3.manifest.stems_registry[0].is_star_center);
    assert!(v3.manifest.stems_registry[2].is_star_center);

    let p3_muted = set_capture_enabled(&p3, "s2", false).unwrap();
    let p3_unmuted = set_capture_enabled(&p3_muted, "s2", true).unwrap();

    let p4 = replace_capture(
        &p3_unmuted,
        "s2",
        mk_stereo_spec("s2_rep", true, 9.0, false),
    )
    .unwrap();

    let p5 = remove_capture(&p4, "s2_rep").unwrap();
    let v5 = DnamProjView::from_bytes(&p5, &crypto).unwrap();
    assert_eq!(v5.manifest.stems_registry.len(), 2);
    assert!(v5.manifest.stems_registry[0].is_star_center);

    let p6 = remove_capture(&p5, "s0").unwrap();
    let p_empty = remove_capture(&p6, "s1").unwrap();
    let v_empty = DnamProjView::from_bytes(&p_empty, &crypto).unwrap();
    assert!(v_empty.manifest.stems_registry.is_empty());
    assert_ne!(
        v_empty.preamble.lifecycle_flags & PROJ_FLAG_HAS_CALIBRATION,
        0
    );
}

#[test]
fn test_forged_header_crc_and_bijection_tamper_rejections() {
    let crypto = DefaultCryptoProvider::default();

    // Minimal JSON omitting #[serde(default)] fields exercises schema defaults cleanly
    let base_val = serde_json::to_value(&DnamProjManifest::default_reference().base).unwrap();
    let minimal_json = serde_json::json!({ "base": base_val }).to_string();
    let norm = DnamProjManifest::validate_and_normalize_json_str(&minimal_json).unwrap();
    assert!(norm.contains(r#""container_kind":"dnam-proj""#));
    assert!(norm.contains(r#""version":"2.0.0""#));

    let recompute_hdr_crc = |buf: &mut [u8]| {
        let pre = DnamProjPreamble::from_bytes(&buf[..64]).unwrap();
        let dir_start = pre.chunk_dir_offset as usize;
        let m_start = pre.manifest_offset as usize;
        let m_end = m_start + (pre.manifest_len as usize);
        let crc =
            compute_proj_header_crc32c(&buf[..40], &buf[dir_start..m_start], &buf[m_start..m_end]);
        buf[40..44].copy_from_slice(&crc.to_le_bytes());
    };

    // 1. Forged chunk SHA-256 mismatch (valid CRC32C, corrupted directory sha256 + recomputed header CRC)
    let calb = (CalbChunk64B {
        sample_rate_device_hz: 48_000,
        bit_depth_code: 2,
        polarity_inverted: false,
        loopback_lag_samples: 0.0,
        signed_gain: 1.0,
        input_reference_dbfs: -20.5,
        noise_floor_dbfs: -96.0,
    })
    .to_bytes()
    .to_vec();
    let valid_calb_proj = build_dnam_proj_container(
        &DnamProjManifest::default_reference(),
        &[ProjChunkInput {
            kind: ProjChunkKind::Calb,
            chunk_id: 0,
            payload: calb.clone(),
        }],
    )
    .unwrap();

    let mut sha_forged = valid_calb_proj.clone();
    sha_forged[64 + 32] ^= 0xFF;
    recompute_hdr_crc(&mut sha_forged);
    assert!(DnamProjView::from_bytes(&sha_forged, &crypto).is_err());

    // 2. Forged preamble lifecycle_flags (clearing PROJ_FLAG_HAS_CALIBRATION while CALB exists)
    let mut flags_forged = valid_calb_proj.clone();
    flags_forged[8..12].copy_from_slice(&0_u32.to_le_bytes());
    recompute_hdr_crc(&mut flags_forged);
    assert!(DnamProjView::from_bytes(&flags_forged, &crypto).is_err());

    // 3. Forged non-contiguous chunk offset in directory
    let mut gap_forged = valid_calb_proj;
    let cur_off = u64::from_le_bytes(gap_forged[64 + 16..64 + 24].try_into().unwrap());
    gap_forged[64 + 16..64 + 24].copy_from_slice(&(cur_off + 64).to_le_bytes());
    recompute_hdr_crc(&mut gap_forged);
    assert!(parse_proj_manifest_range(&gap_forged).is_err());

    // 4. Duplicate chunk directory key (SPCM chunk_id=0 twice) rejected
    let mut spcm = vec![0u8; 64];
    spcm[0..4].copy_from_slice(b"SPCM");
    assert!(
        build_dnam_proj_container(
            &DnamProjManifest::default_reference(),
            &[
                ProjChunkInput {
                    kind: ProjChunkKind::Spcm,
                    chunk_id: 0,
                    payload: spcm.clone(),
                },
                ProjChunkInput {
                    kind: ProjChunkKind::Spcm,
                    chunk_id: 0,
                    payload: spcm,
                },
            ],
        )
        .is_err()
    );

    // 5. STMC <-> manifest.stems_registry bijection rejections (STMC without INPC, count mismatch,
    //    missing chunk_id, and enabled-bit mismatch)
    let mono_fp16 = vec![0u8; 24_000];
    let inpc = build_proj_inpc_capsule(1, &[1; 256], &[2; 320], &mono_fp16).unwrap();
    let stmc = dnam_capture::proj::build_proj_stmc_capsule(
        0,
        dnam_capture::proj::STMC_FLAG_ENABLED,
        1.0,
        0.0,
        &[0; 88],
        0,
        &[StmcTrajectoryWindow {
            control_id: 0,
            ctrl_kind: 0,
            active_opt: 0,
            u_phys: 0.0,
            window_weight: 1.0,
            emr_velocity: 0.0,
        }],
        &[0; 4_096],
        &mono_fp16,
    )
    .unwrap();

    // STMC without INPC
    assert!(
        build_dnam_proj_container(
            &DnamProjManifest::default_reference(),
            &[ProjChunkInput {
                kind: ProjChunkKind::Stmc,
                chunk_id: 0,
                payload: stmc.clone(),
            }],
        )
        .is_err()
    );

    // STMC + INPC but manifest.stems_registry is empty (count mismatch)
    assert!(
        build_dnam_proj_container(
            &DnamProjManifest::default_reference(),
            &[
                ProjChunkInput {
                    kind: ProjChunkKind::Inpc,
                    chunk_id: 0,
                    payload: inpc.clone(),
                },
                ProjChunkInput {
                    kind: ProjChunkKind::Stmc,
                    chunk_id: 0,
                    payload: stmc.clone(),
                },
            ],
        )
        .is_err()
    );

    let valid_sha = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
    let mut m_wrong_cid = DnamProjManifest::default_reference();
    m_wrong_cid
        .stems_registry
        .push(dnam_capture::proj::StemRegistryEntry {
            stem_id: "s0".to_string(),
            chunk_id: 7,
            enabled: true,
            is_star_center: false,
            controls_meta: BTreeMap::new(),
            provenance: StemProvenance {
                input_pcm_sha256: valid_sha.to_string(),
                target_pcm_sha256: valid_sha.to_string(),
                num_frames: 12_000,
            },
            user_metadata: BTreeMap::new(),
            extracted_physics: None,
        });
    assert!(
        build_dnam_proj_container(
            &m_wrong_cid,
            &[
                ProjChunkInput {
                    kind: ProjChunkKind::Inpc,
                    chunk_id: 0,
                    payload: inpc.clone(),
                },
                ProjChunkInput {
                    kind: ProjChunkKind::Stmc,
                    chunk_id: 0,
                    payload: stmc.clone(),
                },
            ],
        )
        .is_err()
    );

    // Matching chunk_id=0, but manifest.enabled=false while STMC_FLAG_ENABLED=true
    let mut m_en_mismatch = m_wrong_cid;
    m_en_mismatch.stems_registry[0].chunk_id = 0;
    m_en_mismatch.stems_registry[0].enabled = false;
    assert!(
        build_dnam_proj_container(
            &m_en_mismatch,
            &[
                ProjChunkInput {
                    kind: ProjChunkKind::Inpc,
                    chunk_id: 0,
                    payload: inpc,
                },
                ProjChunkInput {
                    kind: ProjChunkKind::Stmc,
                    chunk_id: 0,
                    payload: stmc,
                },
            ],
        )
        .is_err()
    );
}
