#![forbid(unsafe_code)]
//! `.dnam-proj` `AUTH` (`"DNPJ-PROJECT-MANIFEST-V2"`) cryptographic signature verification,
//! mutable vs. immutable JCS manifest separation, runtime `DAU1` / `DMU1` security guards,
//! and live large-payload `DNWB` subprocess pipe worker tests.

use dnam_capture::proj::{
    DnamProjManifest, DnamProjView, PROJ_FLAG_HAS_AUTH, ProjChunkInput, ProjChunkKind,
    attach_proj_auth_signature, build_dnam_proj_container, build_dnam_proj_container_with_cp1,
    export_runtime_dnam_with_crypto,
};
use dnam_capture::whitebox::{DnwbCommand, FaustRunner, SpiceRunner, write_dnwb_stem_response};
use dnam_capture::{DistillationTarget, StemResponse, StemStimulus, StimulusFrame};
use dnam_engine::container::{
    ARENA_OFFSET_LITE_END, DnamContainerView, build_dmu1_mutable_frame, build_dnam_container,
    build_v2_open_dnam_container,
};
use dnam_engine::provenance::{
    Dau1Envelope, DefaultCryptoProvider, RustCryptoProvider, VerificationStatus,
    build_buyer_receipt_v1, build_creator_manifest_v1, build_plaintext_shared_v1,
    compute_creator_signed_manifest_sha256, compute_license_policy_sha256,
    compute_plaintext_prefixes_v1, sha256_digest,
};
use dnam_engine::schema::{
    ControlKind, ControlSchemaDef, ControlUiHints, DistributionPolicy, DnamError, DnamJsonHeader,
    DnamProtectionProfile, PresetDef,
};
use std::collections::BTreeMap;
use std::f64::consts::PI;

fn log_chirp_f32(n: usize, sample_rate_hz: u32, amp: f32) -> Vec<f32> {
    let sr = f64::from(sample_rate_hz);
    let dur = (n as f64) / sr;
    let (f0, f1) = (25.0_f64, (0.42 * sr).min(18_000.0));
    let ln_r = (f1 / f0).ln();
    (0..n)
        .map(|i| {
            let t = (i as f64) / sr;
            let ph = 2.0 * PI * f0 * dur * ((t / dur * ln_r).exp() - 1.0) / ln_r;
            amp * (ph.sin() as f32)
        })
        .collect()
}

#[test]
fn test_proj_auth_signature_mutable_manifest_preservation_and_tamper_rejection() {
    // verify[container.proj.auth-signature]
    let crypto = RustCryptoProvider::default();
    let seed = [0x5A_u8; 32];
    let pk = RustCryptoProvider::ed25519_pubkey_from_seed(&seed);

    let mut manifest = DnamProjManifest::default_reference();
    let arena = vec![0u8; ARENA_OFFSET_LITE_END as usize];
    let unsigned_proj =
        build_dnam_proj_container_with_cp1(&mut manifest, &[], &arena, &[]).unwrap();
    let unsigned_view = DnamProjView::from_bytes(&unsigned_proj, &crypto).unwrap();
    assert_eq!(
        unsigned_view.preamble.lifecycle_flags & PROJ_FLAG_HAS_AUTH,
        0
    );
    let wght = unsigned_view
        .chunk_slice(ProjChunkKind::Wght, 0)
        .unwrap()
        .to_vec();

    let signed_proj = attach_proj_auth_signature(
        &unsigned_proj,
        pk,
        |msg| RustCryptoProvider::ed25519_sign(&seed, msg),
        &crypto,
    )
    .unwrap();
    let signed_view = DnamProjView::from_bytes(&signed_proj, &crypto).unwrap();
    assert_ne!(signed_view.preamble.lifecycle_flags & PROJ_FLAG_HAS_AUTH, 0);
    assert_eq!(signed_view.auth_record.unwrap().signer_pubkey_ed25519, pk);
    assert!(export_runtime_dnam_with_crypto(&signed_proj, None, &crypto).is_ok());

    // Mutable fields (`rig_notes`, `presets`, `user_metadata`) preserve `immutable_jcs_sha256`
    // even when Region 2 grows by > 128 B and shifts Region 3 chunk offsets!
    let mut edited_manifest = signed_view.manifest.clone();
    edited_manifest.project_meta.rig_notes =
        Some("Updated studio mic placement notes — ".repeat(6));
    edited_manifest.base.presets.push(PresetDef {
        id: String::from("user_lead"),
        name: String::from("User Lead"),
        description: Some(String::from("Added without creator private key")),
        controls: BTreeMap::new(),
        user_metadata: BTreeMap::new(),
    });
    edited_manifest
        .base
        .user_metadata
        .insert(String::from("tag"), serde_json::json!("favorite"));
    let auth_bytes = signed_view
        .chunk_slice(ProjChunkKind::Auth, 0)
        .unwrap()
        .to_vec();
    let repacked_mutable = build_dnam_proj_container(
        &edited_manifest,
        &[
            ProjChunkInput {
                kind: ProjChunkKind::Wght,
                chunk_id: 0,
                payload: wght.clone(),
            },
            ProjChunkInput {
                kind: ProjChunkKind::Auth,
                chunk_id: 0,
                payload: auth_bytes.clone(),
            },
        ],
    )
    .unwrap();
    assert!(repacked_mutable.len() > signed_proj.len());
    let repacked_view = DnamProjView::from_bytes(&repacked_mutable, &crypto).unwrap();
    assert_eq!(repacked_view.manifest.base.presets.len(), 1);
    assert_eq!(repacked_view.auth_record.unwrap().signer_pubkey_ed25519, pk);

    // Tampering with an immutable field (`metadata.name`) while keeping old AUTH must fail closed!
    let mut forged_manifest = signed_view.manifest.clone();
    forged_manifest.base.metadata.name = String::from("Forged Model Name");
    let forged_proj_res =
        build_dnam_proj_container_with_crypto_unchecked(&forged_manifest, &wght, &auth_bytes);
    assert!(matches!(
        DnamProjView::from_bytes(&forged_proj_res, &crypto),
        Err(DnamError::CryptoVerificationFailed(_))
    ));
    assert!(matches!(
        export_runtime_dnam_with_crypto(&forged_proj_res, None, &crypto),
        Err(DnamError::CryptoVerificationFailed(_))
    ));

    // Tampering with WGHT payload while keeping old AUTH must also fail closed!
    let mut forged_arena = arena;
    forged_arena[124..128].copy_from_slice(&0.5_f32.to_le_bytes());
    let mut tmp_hdr = signed_view.manifest.base.clone();
    let forged_wght =
        build_dnam_container(&mut tmp_hdr, &forged_arena, &forged_arena, false).unwrap();
    let forged_chunk_proj = build_dnam_proj_container_with_crypto_unchecked(
        &signed_view.manifest,
        &forged_wght,
        &auth_bytes,
    );
    assert!(matches!(
        DnamProjView::from_bytes(&forged_chunk_proj, &crypto),
        Err(DnamError::CryptoVerificationFailed(_))
    ));
}

fn build_dnam_proj_container_with_crypto_unchecked(
    manifest: &DnamProjManifest,
    wght: &[u8],
    auth: &[u8],
) -> Vec<u8> {
    use dnam_capture::proj::{
        DnamProjPreamble, PROJ_PREAMBLE_SIZE, ProjChunkDirEntry, compute_proj_header_crc32c,
    };
    use dnam_engine::container::{align64_u64, crc32c_update};
    let manifest_jcs = manifest.to_jcs_bytes().unwrap();
    let manifest_offset = (PROJ_PREAMBLE_SIZE + 2 * 64) as u64;
    let payload_start = align64_u64(manifest_offset + manifest_jcs.len() as u64).unwrap();
    let wght_padded = align64_u64(payload_start + wght.len() as u64).unwrap();
    let total_end = align64_u64(wght_padded + auth.len() as u64).unwrap();
    let e0 = ProjChunkDirEntry {
        fourcc: ProjChunkKind::Wght.fourcc(),
        chunk_id: 0,
        flags: 0,
        crc32c: crc32c_update(0, wght),
        offset: payload_start,
        byte_len: wght.len() as u64,
        sha256: sha256_digest(wght),
    };
    let e1 = ProjChunkDirEntry {
        fourcc: ProjChunkKind::Auth.fourcc(),
        chunk_id: 0,
        flags: 0,
        crc32c: crc32c_update(0, auth),
        offset: wght_padded,
        byte_len: auth.len() as u64,
        sha256: sha256_digest(auth),
    };
    let mut dir_bytes = Vec::with_capacity(128);
    dir_bytes.extend_from_slice(&e0.to_bytes());
    dir_bytes.extend_from_slice(&e1.to_bytes());
    let mut pre = DnamProjPreamble {
        magic: dnam_capture::proj::DNPJ_MAGIC,
        version_major: 2,
        version_minor: 0,
        lifecycle_flags: (1 << 4) | (1 << 5) | (1 << 6),
        chunk_count: 2,
        chunk_dir_offset: 64,
        manifest_offset,
        manifest_len: manifest_jcs.len() as u64,
        header_crc32c: 0,
        reserved: [0u8; 20],
    };
    let tmp = pre.to_bytes();
    pre.header_crc32c = compute_proj_header_crc32c(&tmp[0..40], &dir_bytes, &manifest_jcs);
    let mut out = vec![0u8; total_end as usize];
    out[0..64].copy_from_slice(&pre.to_bytes());
    out[64..192].copy_from_slice(&dir_bytes);
    let m_off = manifest_offset as usize;
    out[m_off..m_off + manifest_jcs.len()].copy_from_slice(&manifest_jcs);
    out[payload_start as usize..payload_start as usize + wght.len()].copy_from_slice(wght);
    out[wght_padded as usize..wght_padded as usize + auth.len()].copy_from_slice(auth);
    out
}

#[test]
fn test_v2_dau1_relocated_buyer_tier_distribution_policy_and_dmu1_schema_merges() {
    let mut base_hdr = DnamJsonHeader::default_reference();
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

    let bad_ak1 =
        build_v2_open_dnam_container(&hdr_json, &shared_1ctrl, &[0u8; 64], &[&t0], None, 1)
            .unwrap();
    assert!(DnamContainerView::from_bytes(&bad_ak1, &DefaultCryptoProvider::default()).is_err());

    let mut tampered_dau1 = vec![0u8; 160];
    tampered_dau1[0..8].copy_from_slice(b"DAU1\x09\x00\x00\x00");
    let bad_dau1 =
        build_v2_open_dnam_container(&hdr_json, &shared_1ctrl, &tampered_dau1, &[&t0], None, 0)
            .unwrap();
    assert!(DnamContainerView::from_bytes(&bad_dau1, &DefaultCryptoProvider::default()).is_err());

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

    let policy = br#"{"tier":"full"}"#;
    let probe_env_bytes = Dau1Envelope {
        auth_kind: 1,
        creator_pubkey: c_pk,
        creator_max_tier: 3,
        master_prefixes: master_pfxs.clone(),
        creator_signature: c_sig,
        marketplace_pubkey: Some(m_pk),
        policy_bytes: policy.to_vec(),
        buyer_max_tier: Some(1),
        buyer_prefixes: vec![[0u8; 32]; 2],
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

    let pinned_wrong = RustCryptoProvider::new().with_marketplace_pubkey([0xEE_u8; 32]);
    assert!(matches!(
        DnamContainerView::from_bytes(&reloc_dnam, &pinned_wrong),
        Err(DnamError::CryptoVerificationFailed(_))
    ));

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

    let mut enclave_hdr = base_hdr;
    enclave_hdr.metadata.distribution_policy = Some(DistributionPolicy {
        allowed_protection_profiles: vec![DnamProtectionProfile::EnclaveSealed],
        allowed_oem_vendors: Vec::new(),
    });
    let illegal_open_dnam = build_v2_open_dnam_container(
        &enclave_hdr.to_json_vec().unwrap(),
        &shared_1ctrl,
        &[0u8; 64],
        &[&t0],
        None,
        0,
    )
    .unwrap();
    assert!(matches!(
        DnamContainerView::from_bytes(&illegal_open_dnam, &crypto_mp),
        Err(DnamError::CryptoVerificationFailed(_))
    ));
}

#[test]
fn test_live_dnwb_large_payload_subprocess_pipe_workers_faust_and_spice() {
    // verify[train.capture.whitebox.distillation]
    let n_large = 32_768usize;
    let stim = StemStimulus {
        stem_id: String::from("ipc_large_pipe_96k"),
        sample_rate_hz: 96_000,
        in_1: log_chirp_f32(n_large, 96_000, 0.4),
        in_2: Some(log_chirp_f32(n_large, 96_000, 0.3)),
        in_3: None,
        in_4: None,
        control_values: vec![0.5, 0.5],
        control_trajectories: None,
        freeze_transition_window: Some([8_192, 12_288]),
    };
    let json = r#"{"ui":[{"type":"vgroup","label":"W","items":[{"type":"hslider","label":"Drive","init":0.5,"min":0.0,"max":1.0,"step":0.01},{"type":"hslider","label":"Tone","init":0.5,"min":0.0,"max":1.0,"step":0.01}]}]}"#;
    let mut faust_inproc = FaustRunner::from_faust_json("w.dsp", json, 5_000).unwrap();
    faust_inproc.set_switches(&[1]);
    let mut spice_inproc = SpiceRunner::new("w.cir", 5_000);
    spice_inproc.set_switches(&[1]);

    let build_replay_cmd = |resps: &[(DnwbCommand, StemResponse)], tag: &str| -> Vec<String> {
        let mut buf = Vec::new();
        for (cmd, resp) in resps {
            write_dnwb_stem_response(&mut buf, *cmd, resp).unwrap();
        }
        let path = std::env::temp_dir().join(format!("dnwb_{tag}_{}.bin", std::process::id()));
        std::fs::write(&path, &buf).unwrap();
        let js = r#"const fs=require("fs");const out=fs.readFileSync(process.argv[1]);let o=0,b=Buffer.alloc(0);process.stdin.on("data",c=>{b=Buffer.concat([b,c]);while(b.length>=64){const n=64+Number(b.readBigUInt64LE(28));if(b.length<n)break;b=b.subarray(n);const m=64+Number(out.readBigUInt64LE(o+28));process.stdout.write(out.subarray(o,o+m));o+=m;}});"#;
        vec![
            String::from("node"),
            String::from("-e"),
            String::from(js),
            path.to_string_lossy().into_owned(),
        ]
    };

    let ref_faust = faust_inproc.render_stem(&stim).unwrap();
    let ref_spice = spice_inproc.render_stem(&stim).unwrap();
    assert!(
        ref_spice
            .out_2
            .as_ref()
            .unwrap()
            .iter()
            .any(|&v| v.abs() > 1.0e-3)
    );
    faust_inproc.reset_state();
    spice_inproc.reset_state();
    let reset_resp = StemResponse {
        stem_id: String::from("reset"),
        sample_rate_hz: 48_000,
        out_1: Vec::new(),
        out_2: None,
        out_3: None,
        out_4: None,
        internal_nodes: BTreeMap::new(),
    };
    let mut f_seq = vec![
        (DnwbCommand::RenderStem, ref_faust.clone()),
        (DnwbCommand::Reset, reset_resp.clone()),
    ];
    let mut s_seq = vec![
        (DnwbCommand::RenderStem, ref_spice.clone()),
        (DnwbCommand::Reset, reset_resp),
    ];
    let mut frames = Vec::new();
    for idx in 0..3usize {
        let fr = StimulusFrame {
            frame_index: idx,
            sample_rate_hz: 96_000,
            in_1: stim.in_1[idx * 64..(idx + 1) * 64].to_vec(),
            in_2: None,
            in_3: None,
            in_4: None,
            control_values: vec![0.5, 0.5],
            control_trajectories: None,
        };
        let mut rf = faust_inproc.render_frame(&fr).unwrap();
        let mut rs = spice_inproc.render_frame(&fr).unwrap();
        rf.stem_id = format!("frame_{idx}");
        rs.stem_id = format!("frame_{idx}");
        f_seq.push((DnwbCommand::RenderFrame, rf));
        s_seq.push((DnwbCommand::RenderFrame, rs));
        frames.push(fr);
    }

    let mut faust_ipc = FaustRunner::from_faust_json("w.dsp", json, 5_000)
        .unwrap()
        .with_subprocess_command(build_replay_cmd(&f_seq, "faust"));
    faust_ipc.set_switches(&[1]);
    let mut spice_ipc =
        SpiceRunner::new("w.cir", 5_000).with_subprocess_command(build_replay_cmd(&s_seq, "spice"));
    spice_ipc.set_switches(&[1]);

    assert_eq!(ref_faust, faust_ipc.render_stem(&stim).unwrap());
    assert_eq!(ref_spice, spice_ipc.render_stem(&stim).unwrap());
    faust_ipc.reset_state();
    spice_ipc.reset_state();
    for (idx, fr) in frames.iter().enumerate() {
        assert_eq!(
            f_seq[idx + 2].1.out_1,
            faust_ipc.render_frame(fr).unwrap().out_1
        );
        assert_eq!(
            s_seq[idx + 2].1.out_1,
            spice_ipc.render_frame(fr).unwrap().out_1
        );
    }
    let _ = std::fs::remove_file(
        std::env::temp_dir().join(format!("dnwb_faust_{}.bin", std::process::id())),
    );
    let _ = std::fs::remove_file(
        std::env::temp_dir().join(format!("dnwb_spice_{}.bin", std::process::id())),
    );
}
