#![forbid(unsafe_code)]
//! `.dnam-proj` (`DNPJ`, `v2.0`) container layout, `HTTP Range` manifest parsing, capsule
//! verification (`CALB`, `INPC`, `STMC`, `STAT`, `WGHT`), `[STAB-ZERO-DC]` enforcement,
//! `compact_to_capsules`, Two-Path Capture Vault (`add_capture`, `remove_capture`,
//! `replace_capture`, `set_capture_enabled`), and numerical edge-case regression tests.

use dnam_capture::align::{
    compute_pcm_sha256, decode_wav_bytes, encode_wav_with_bit_depth,
    estimate_subsample_lag_and_gain,
};
use dnam_capture::metrics::{IntervalAuditDecision, evaluate_joint_energy_interval_audit};
use dnam_capture::proj::{
    CalbChunk64B, DnamProjManifest, DnamProjView, ExtractedPhysics, PROJ_FLAG_COMPACTED_CAPSULES,
    PROJ_FLAG_HAS_CALIBRATION, PROJ_FLAG_HAS_RAW_STEMS, PROJ_FLAG_HAS_WGHT, PROJ_STAT_BYTES_ULTRA,
    ProjChunkInput, ProjChunkKind, STMC_FLAG_ENABLED, STMC_FLAG_MODE_A_SWEEP,
    STMC_FLAG_STAR_CENTER, STMC_FLAG_STEREO, StemProvenance, StemRegistryEntry,
    StmcTrajectoryWindow, VaultCaptureSpec, add_capture, build_dnam_proj_container,
    build_dnam_proj_container_with_cp1, build_proj_inpc_capsule, build_proj_stmc_capsule,
    compact_to_capsules, export_runtime_dnam, f32_to_fp16_rne_bytes, parse_proj_manifest_range,
    remove_capture, replace_capture, set_capture_enabled, set_stmc_capsule_enabled_flag,
    verify_proj_inpc_capsule, verify_proj_stat_uplo_u, verify_proj_stmc_capsule,
};
use dnam_capture::{
    CORESET_FP16_FRAMES_PER_STEM, CaptureBitDepth, DistillationTarget, StemStimulus,
};
use dnam_engine::Tier;
use dnam_engine::container::{
    ARENA_OFFSET_LITE_END, ARENA_OFFSET_ULTRA_END, DnamContainerView, build_dnam_container,
};
use dnam_engine::control::Cp1ContributorSlot;
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::schema::{
    ControlCalibration, ControlCalibrationRole, ControlSchemaDef, DistributionPolicy, DnamError,
    DnamJsonHeader, DnamProtectionProfile,
};
use std::collections::BTreeMap;

fn make_range_control(id: &str, min_v: f32, max_v: f32, def_v: f32) -> ControlSchemaDef {
    serde_json::from_value(serde_json::json!({
        "id": id,
        "name": id,
        "type": "range",
        "lobatto_degree": 2,
        "declared_range": [min_v, max_v],
        "safe_clamped_range": [min_v, max_v],
        "default": def_v
    }))
    .unwrap()
}

fn make_switch_control(id: &str) -> ControlSchemaDef {
    serde_json::from_value(serde_json::json!({
        "id": id,
        "name": id,
        "type": "switch",
        "declared_options": ["Clean", "Crunch", "Lead"],
        "active_options": ["Clean", "Crunch", "Lead"],
        "default": "Clean"
    }))
    .unwrap()
}

fn make_spcm_chunk(raw_samples: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; 64 + raw_samples.len() * 4];
    out[0..4].copy_from_slice(b"SPCM");
    for (idx, &s) in raw_samples.iter().enumerate() {
        let off = 64 + idx * 4;
        out[off..off + 4].copy_from_slice(&s.to_le_bytes());
    }
    out
}

#[test]
fn test_dnam_proj_lifecycle_range_parse_capsules_and_zero_dc_guards() {
    // verify[container.proj.preamble-and-directory]
    // verify[container.proj.jcs-manifest]
    // verify[container.proj.lifecycle-states]
    // verify[container.proj.inpc-capsule]
    // verify[container.proj.stmc-capsule]
    // verify[container.proj.stat-uplo-u]
    // verify[container.proj.wght-and-export]
    let crypto = DefaultCryptoProvider::default();

    // 1. Setup-only manifest lifecycle state + CALB hardware loopback calibration
    let setup_manifest = DnamProjManifest::default_reference();
    let setup_only = build_dnam_proj_container(&setup_manifest, &[]).unwrap();
    let setup_view = DnamProjView::from_bytes(&setup_only, &crypto).unwrap();
    assert!(setup_view.directory.is_empty());
    assert!(export_runtime_dnam(&setup_only, None).is_err());

    let calb_hdr = CalbChunk64B {
        sample_rate_device_hz: 48_000,
        bit_depth_code: 2,
        polarity_inverted: false,
        loopback_lag_samples: 64.25,
        signed_gain: 0.125,
        input_reference_dbfs: -20.5,
        noise_floor_dbfs: -96.0,
    };
    let calb_bytes = calb_hdr.to_bytes().to_vec();
    let parsed_calb = CalbChunk64B::from_bytes(&calb_bytes).unwrap();
    assert_eq!(parsed_calb, calb_hdr);

    let with_calb = build_dnam_proj_container(
        &setup_manifest,
        &[ProjChunkInput {
            kind: ProjChunkKind::Calb,
            chunk_id: 0,
            payload: calb_bytes.clone(),
        }],
    )
    .unwrap();
    let calb_view = DnamProjView::from_bytes(&with_calb, &crypto).unwrap();
    assert_ne!(
        calb_view.preamble.lifecycle_flags & PROJ_FLAG_HAS_CALIBRATION,
        0
    );

    // 2. Full project with SPCM, INPC, STMC, STAT, WGHT, CP1 & HTTP Range prefix inspection
    let dry: Vec<f32> = (0..CORESET_FP16_FRAMES_PER_STEM)
        .map(|i| 0.35 * ((i as f32) * 0.031).sin())
        .collect();
    let wet: Vec<f32> = dry.iter().map(|&x| (1.6 * x).tanh()).collect();
    let dry_fp16_mono = f32_to_fp16_rne_bytes(&dry);
    let mut dry_stereo = Vec::with_capacity(dry.len() * 2);
    for &x in &dry {
        dry_stereo.push(x);
        dry_stereo.push(0.8 * x);
    }
    let dry_fp16_stereo = f32_to_fp16_rne_bytes(&dry_stereo);
    let wet_fp16_mono = f32_to_fp16_rne_bytes(&wet);
    let mut wet_stereo = Vec::with_capacity(wet.len() * 2);
    for &y in &wet {
        wet_stereo.push(y);
        wet_stereo.push(0.9 * y);
    }
    let wet_fp16_stereo = f32_to_fp16_rne_bytes(&wet_stereo);

    let seed_256 = [0x11_u8; 256];
    let schedule_320 = [0x22_u8; 320];
    let inpc_mono = build_proj_inpc_capsule(1, &seed_256, &schedule_320, &dry_fp16_mono).unwrap();
    let inpc_stereo =
        build_proj_inpc_capsule(2, &seed_256, &schedule_320, &dry_fp16_stereo).unwrap();
    assert_eq!((inpc_mono.len(), inpc_stereo.len()), (24_640, 48_640));
    assert_eq!(verify_proj_inpc_capsule(&inpc_mono).unwrap().channels, 1);
    assert_eq!(verify_proj_inpc_capsule(&inpc_stereo).unwrap().channels, 2);

    let traj_win = StmcTrajectoryWindow {
        control_id: 0,
        ctrl_kind: 0,
        active_opt: 0xFF,
        u_phys: 5.0,
        window_weight: 1.0,
        emr_velocity: 0.05,
    };
    let band_psd = [0u8; 88];
    let spec_4k = [0u8; 4_096];
    let mut stmc_mono = build_proj_stmc_capsule(
        0,
        STMC_FLAG_ENABLED | STMC_FLAG_STAR_CENTER | STMC_FLAG_MODE_A_SWEEP,
        1.0,
        0.25,
        &band_psd,
        24,
        &[traj_win; 36],
        &spec_4k,
        &wet_fp16_mono,
    )
    .unwrap();
    let stmc_stereo = build_proj_stmc_capsule(
        1,
        STMC_FLAG_ENABLED | STMC_FLAG_STEREO,
        1.0,
        0.0,
        &band_psd,
        24,
        &[traj_win],
        &spec_4k,
        &wet_fp16_stereo,
    )
    .unwrap();
    assert_eq!((stmc_mono.len(), stmc_stereo.len()), (28_800, 52_800));
    assert!(verify_proj_stmc_capsule(&stmc_mono).unwrap().is_enabled());
    set_stmc_capsule_enabled_flag(&mut stmc_mono, false).unwrap();
    assert!(!verify_proj_stmc_capsule(&stmc_mono).unwrap().is_enabled());
    set_stmc_capsule_enabled_flag(&mut stmc_mono, true).unwrap();

    let stat = vec![0u8; PROJ_STAT_BYTES_ULTRA];
    verify_proj_stat_uplo_u(&stat).unwrap();

    // 384 KB raw audio beyond 64 KB Range prefix!
    let spcm = make_spcm_chunk(&vec![0.125_f32; 96_000]);
    let arena_64k = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];

    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.controls_schema = vec![make_range_control("gain", 0.0, 10.0, 5.0)];
    manifest.stems_registry = vec![StemRegistryEntry {
        stem_id: String::from("anchor_0"),
        chunk_id: 0,
        enabled: true,
        is_star_center: true,
        controls_meta: BTreeMap::from([(String::from("gain"), serde_json::json!(5.0))]),
        provenance: StemProvenance {
            input_pcm_sha256: compute_pcm_sha256(&dry),
            target_pcm_sha256: compute_pcm_sha256(&wet),
            num_frames: dry.len() as u64,
        },
        user_metadata: BTreeMap::new(),
        extracted_physics: Some(ExtractedPhysics {
            delays_ms: Vec::new(),
            lfo_freqs_hz: Vec::new(),
            decay_gains_lin: Vec::new(),
            detected_grain_size_ms: 0.0,
            detected_pitch_shift_semitones: 0.0,
        }),
    }];

    let mut cp1_slot = Cp1ContributorSlot::zero();
    cp1_slot.lanes = 1;
    cp1_slot.ctrl_kind = 0;
    cp1_slot.control_idx = 0;
    cp1_slot.option_idx = 0xFF;
    cp1_slot.main_base[0] = [0.5; 8];
    cp1_slot.emb = [1.0, 0.0, 0.0];

    let full_proj = build_dnam_proj_container_with_cp1(
        &mut manifest,
        &[
            ProjChunkInput {
                kind: ProjChunkKind::Calb,
                chunk_id: 0,
                payload: calb_bytes,
            },
            ProjChunkInput {
                kind: ProjChunkKind::Spcm,
                chunk_id: 0,
                payload: spcm.clone(),
            },
            ProjChunkInput {
                kind: ProjChunkKind::Inpc,
                chunk_id: 0,
                payload: inpc_mono.clone(),
            },
            ProjChunkInput {
                kind: ProjChunkKind::Stmc,
                chunk_id: 0,
                payload: stmc_mono.clone(),
            },
            ProjChunkInput {
                kind: ProjChunkKind::Stat,
                chunk_id: 0,
                payload: stat,
            },
        ],
        &arena_64k,
        std::slice::from_ref(&cp1_slot),
    )
    .unwrap();
    assert!(full_proj.len() > 65_536);

    // HTTP Range: bytes=0-65535 parses Preamble + Chunk Directory + JCS Manifest without Region 3!
    let range_parsed = parse_proj_manifest_range(&full_proj[..65_536]).unwrap();
    assert_eq!(range_parsed.preamble.chunk_count, 6);
    assert_eq!(range_parsed.directory.len(), 6);
    assert_eq!(range_parsed.manifest.stems_registry[0].stem_id, "anchor_0");
    assert_ne!(
        range_parsed.preamble.lifecycle_flags & PROJ_FLAG_HAS_RAW_STEMS,
        0
    );
    assert_ne!(
        range_parsed.preamble.lifecycle_flags & PROJ_FLAG_HAS_WGHT,
        0
    );

    // Full container parse + runtime .dnam export preserves CP1 contributors
    let full_view = DnamProjView::from_bytes(&full_proj, &crypto).unwrap();
    let wght_in_proj = full_view.chunk_slice(ProjChunkKind::Wght, 0).unwrap();
    let wght_rt_view = DnamContainerView::from_bytes(wght_in_proj, &crypto).unwrap();
    assert_eq!(wght_rt_view.cp1_contributors, vec![cp1_slot]);
    let exported_dnam = export_runtime_dnam(&full_proj, None).unwrap();
    let rt_view = DnamContainerView::from_bytes(&exported_dnam, &crypto).unwrap();
    assert_eq!(rt_view.cp1_contributors, vec![cp1_slot]);

    // Reject .dnam magic passed to DnamProjView, trailing bytes, v1.0, corrupted CRC, non-zero b_head
    assert!(matches!(
        DnamProjView::from_bytes(&exported_dnam, &crypto),
        Err(DnamError::InvalidMagic([b'D', b'N', b'A', b'M']))
    ));
    let mut trailing = full_proj.clone();
    trailing.push(0);
    assert!(DnamProjView::from_bytes(&trailing, &crypto).is_err());

    let mut bad_ver = full_proj.clone();
    bad_ver[4..6].copy_from_slice(&1u16.to_le_bytes());
    assert!(parse_proj_manifest_range(&bad_ver[..4096]).is_err());

    let mut bad_crc = full_proj.clone();
    bad_crc[65] ^= 0xFF;
    assert!(parse_proj_manifest_range(&bad_crc[..4096]).is_err());

    // [STAB-ZERO-DC]: non-zero b_head in Arena 0 [9024..9064] or Arena 3 [63424..63456] must fail closed
    for dc_off in [9_024usize, 63_424usize] {
        let mut bad_arena = arena_64k.clone();
        bad_arena[dc_off..dc_off + 4].copy_from_slice(&0.05_f32.to_le_bytes());
        let mut hdr_copy = manifest.base.clone();
        let bad_wght_dnam =
            build_dnam_container(&mut hdr_copy, &bad_arena, &bad_arena, false).unwrap();
        assert!(
            build_dnam_proj_container(
                &manifest,
                &[
                    ProjChunkInput {
                        kind: ProjChunkKind::Inpc,
                        chunk_id: 0,
                        payload: inpc_mono.clone(),
                    },
                    ProjChunkInput {
                        kind: ProjChunkKind::Stmc,
                        chunk_id: 0,
                        payload: stmc_mono.clone(),
                    },
                    ProjChunkInput {
                        kind: ProjChunkKind::Wght,
                        chunk_id: 0,
                        payload: bad_wght_dnam,
                    },
                ],
            )
            .is_err()
        );
    }

    // compact_to_capsules strips SPCM, sets PROJ_FLAG_COMPACTED_CAPSULES, and keeps export_runtime_dnam identical
    let compacted = compact_to_capsules(&full_proj).unwrap();
    assert!(compacted.len() + spcm.len() <= full_proj.len());
    let comp_view = DnamProjView::from_bytes(&compacted, &crypto).unwrap();
    assert_eq!(
        comp_view.preamble.lifecycle_flags & PROJ_FLAG_HAS_RAW_STEMS,
        0
    );
    assert_ne!(
        comp_view.preamble.lifecycle_flags & PROJ_FLAG_COMPACTED_CAPSULES,
        0
    );
    assert!(comp_view.chunk_slice(ProjChunkKind::Spcm, 0).is_none());
    assert_eq!(
        export_runtime_dnam(&compacted, None).unwrap(),
        exported_dnam
    );

    // EnclaveSealed-only distribution_policy on Open .dnam-proj WGHT must fail closed
    let mut enclave_man = DnamProjManifest::default_reference();
    enclave_man.base.metadata.distribution_policy = Some(DistributionPolicy {
        allowed_protection_profiles: vec![DnamProtectionProfile::EnclaveSealed],
        allowed_oem_vendors: Vec::new(),
    });
    let lite_arena = vec![0u8; ARENA_OFFSET_LITE_END as usize];
    assert!(build_dnam_proj_container_with_cp1(&mut enclave_man, &[], &lite_arena, &[]).is_err());
}

#[test]
fn test_two_path_capture_vault_editing_and_wav_subsample_alignment() {
    // verify[train.cache.proj-vault]
    // verify[train.resume.old-stem-nonregression]
    let crypto = DefaultCryptoProvider::default();

    // 1. 5-Bit-Depth WAV round-trip & sub-sample loopback alignment
    let pulse: Vec<f32> = (0..1024)
        .map(|i| {
            let x = (i as f32) - 120.0;
            0.65 * (-0.02 * x * x).exp() * (0.45 * x).cos()
        })
        .collect();
    for depth in [
        CaptureBitDepth::Int12,
        CaptureBitDepth::Int16,
        CaptureBitDepth::Int24,
        CaptureBitDepth::Int32,
        CaptureBitDepth::Float32,
    ] {
        let wav = encode_wav_with_bit_depth(&pulse, 48_000, 1, depth).unwrap();
        let dec = decode_wav_bytes(&wav).unwrap();
        assert_eq!((dec.sample_rate_hz, dec.channels), (48_000, 1));
        assert_eq!(dec.samples.len(), pulse.len());
    }
    let shifted: Vec<f32> = (0..1024)
        .map(|i| {
            let x = (i as f32) - 127.35;
            0.5 * 0.65 * (-0.02 * x * x).exp() * (0.45 * x).cos()
        })
        .collect();
    let align_est = estimate_subsample_lag_and_gain(&pulse, &shifted, 32).unwrap();
    assert!((align_est.total_lag_samples - 7.35).abs() < 0.15);
    assert!((align_est.signed_gain - 0.5).abs() < 0.05);

    // 2. Two-Path Capture Vault: StarCenter Path 2 -> Off-Center Path 1 (`0.000000 ULP` Mode 0) ->
    //    set_capture_enabled -> remove_capture -> replace_capture
    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.controls_schema = vec![
        make_range_control("gain", 0.0, 10.0, 5.0),
        make_switch_control("channel"),
    ];
    let empty_proj = build_dnam_proj_container(&manifest, &[]).unwrap();

    let n = CORESET_FP16_FRAMES_PER_STEM;
    let dry_0: Vec<f32> = (0..n).map(|i| 0.35 * ((i as f32) * 0.029).sin()).collect();
    let wet_0: Vec<f32> = dry_0.iter().map(|&x| (1.5 * x).tanh()).collect();
    let inpc_0 = build_proj_inpc_capsule(
        1,
        &[0x33; 256],
        &[0x44; 320],
        &f32_to_fp16_rne_bytes(&dry_0),
    )
    .unwrap();
    let stat_win = StmcTrajectoryWindow {
        control_id: 0,
        ctrl_kind: 0,
        active_opt: 0xFF,
        u_phys: 5.0,
        window_weight: 1.0,
        emr_velocity: 0.0,
    };

    let proj_s0 = add_capture(
        &empty_proj,
        VaultCaptureSpec {
            stem_id: String::from("star_0"),
            is_star_center: true,
            enabled: true,
            pad_gain_f32: 1.0,
            fractional_lag_samples_f32: 0.0,
            controls_meta: BTreeMap::from([
                (String::from("gain"), serde_json::json!(5.0)),
                (String::from("channel"), serde_json::json!("Clean")),
            ]),
            provenance: StemProvenance {
                input_pcm_sha256: compute_pcm_sha256(&dry_0),
                target_pcm_sha256: compute_pcm_sha256(&wet_0),
                num_frames: n as u64,
            },
            user_metadata: BTreeMap::new(),
            extracted_physics: None,
            windows: vec![stat_win],
            band_psd_summary_88: [0u8; 88],
            n_comb_bins: 24,
            spectral_summary_4096: [0u8; 4_096],
            wet_coreset_fp16: f32_to_fp16_rne_bytes(&wet_0),
            optional_spcm: Some(make_spcm_chunk(&wet_0[..256])),
        },
        Some(&inpc_0),
    )
    .unwrap();

    let view_s0 = DnamProjView::from_bytes(&proj_s0, &crypto).unwrap();
    // Only gain=5.0 & channel="Clean" observed -> safe_clamped_range = [5.0, 5.0], active_options = ["Clean"]
    assert_eq!(
        view_s0.manifest.base.controls_schema[0].safe_clamped_range,
        Some([5.0, 5.0])
    );
    assert_eq!(
        view_s0.manifest.base.controls_schema[1].active_options,
        Some(vec![String::from("Clean")])
    );
    let wght_s0_rt = DnamContainerView::from_bytes(
        view_s0.chunk_slice(ProjChunkKind::Wght, 0).unwrap(),
        &crypto,
    )
    .unwrap();
    // Extract Arena 0..3 Mode 0 + physical Stage I..IV backbone slices (excluding Modes 1..8, CP1/U_int/W_ctrl, and [8512..9024] clamp descriptors)
    const MODE0_BACKBONE_RANGES: [(usize, usize); 6] = [
        (0, 5_440),
        (9_024, 18_560),
        (20_608, 32_448),
        (34_496, 49_408),
        (57_600, 58_112),
        (63_424, 64_000),
    ];
    let s0_arena = wght_s0_rt.arena_storage.as_bytes();
    let star_mode0_slices: Vec<Vec<u8>> = MODE0_BACKBONE_RANGES
        .iter()
        .map(|&(a, b)| s0_arena[a..b].to_vec())
        .collect();
    let stat_s0_bytes = view_s0
        .chunk_slice(ProjChunkKind::Stat, 0)
        .unwrap()
        .to_vec();

    // Path 1 Fast-Path: add off-center capture at gain=8.0, channel="Crunch"
    // Mode 0 (`WGHT` Arena 0..3 backbone weights) MUST stay 0.000000 ULP identical while safe_clamped_range widens to [5.0, 8.0]!
    let wet_1: Vec<f32> = dry_0.iter().map(|&x| (2.8 * x).tanh()).collect();
    let off_win = StmcTrajectoryWindow {
        control_id: 0,
        ctrl_kind: 0,
        active_opt: 0xFF,
        u_phys: 8.0,
        window_weight: 1.0,
        emr_velocity: 0.0,
    };
    let proj_s0_s1 = add_capture(
        &proj_s0,
        VaultCaptureSpec {
            stem_id: String::from("off_1"),
            is_star_center: false,
            enabled: true,
            pad_gain_f32: 1.0,
            fractional_lag_samples_f32: 0.0,
            controls_meta: BTreeMap::from([
                (String::from("gain"), serde_json::json!(8.0)),
                (String::from("channel"), serde_json::json!("Crunch")),
            ]),
            provenance: StemProvenance {
                input_pcm_sha256: compute_pcm_sha256(&dry_0),
                target_pcm_sha256: compute_pcm_sha256(&wet_1),
                num_frames: n as u64,
            },
            user_metadata: BTreeMap::new(),
            extracted_physics: None,
            windows: vec![off_win],
            band_psd_summary_88: [0u8; 88],
            n_comb_bins: 24,
            spectral_summary_4096: [0u8; 4_096],
            wet_coreset_fp16: f32_to_fp16_rne_bytes(&wet_1),
            optional_spcm: None,
        },
        None,
    )
    .unwrap();
    let view_s0_s1 = DnamProjView::from_bytes(&proj_s0_s1, &crypto).unwrap();
    let wght_s0_s1_rt = DnamContainerView::from_bytes(
        view_s0_s1.chunk_slice(ProjChunkKind::Wght, 0).unwrap(),
        &crypto,
    )
    .unwrap();
    let s0_s1_arena = wght_s0_s1_rt.arena_storage.as_bytes();
    for (&(a, b), expected_slice) in MODE0_BACKBONE_RANGES.iter().zip(&star_mode0_slices) {
        assert_eq!(
            &s0_s1_arena[a..b],
            expected_slice.as_slice(),
            "Mode 0 / physical backbone slice [{a}..{b}] must remain 0.000000 ULP identical on Path 1 edit"
        );
    }
    assert_eq!(
        view_s0_s1.chunk_slice(ProjChunkKind::Stat, 0).unwrap(),
        stat_s0_bytes.as_slice()
    );
    assert!(
        s0_s1_arena[5_440..7_488].iter().any(|&b| b != 0),
        "Stage D must populate non-zero SVD head Modes 1..4 for off_1"
    );
    assert_eq!(
        view_s0_s1.manifest.base.controls_schema[0].safe_clamped_range,
        Some([5.0, 8.0])
    );
    assert_eq!(
        view_s0_s1.manifest.base.controls_schema[1].active_options,
        Some(vec![String::from("Clean"), String::from("Crunch")])
    );

    // Duplicate stem_id or missing stem_id must fail closed
    assert!(remove_capture(&proj_s0_s1, "missing_stem").is_err());

    // Disabling off_1 tightens safe_clamped_range back to [5.0, 5.0] and active_options to ["Clean"]
    let proj_off1_disabled = set_capture_enabled(&proj_s0_s1, "off_1", false).unwrap();
    let view_disabled = DnamProjView::from_bytes(&proj_off1_disabled, &crypto).unwrap();
    assert_eq!(
        view_disabled.manifest.base.controls_schema[0].safe_clamped_range,
        Some([5.0, 5.0])
    );
    assert_eq!(
        view_disabled.manifest.base.controls_schema[1].active_options,
        Some(vec![String::from("Clean")])
    );

    // Re-enabling off_1 widens back to [5.0, 8.0], then removing off_1 leaves star_0 cleanly
    let proj_reenabled = set_capture_enabled(&proj_off1_disabled, "off_1", true).unwrap();
    let proj_removed = remove_capture(&proj_reenabled, "off_1").unwrap();
    let view_removed = DnamProjView::from_bytes(&proj_removed, &crypto).unwrap();
    assert_eq!(view_removed.manifest.stems_registry.len(), 1);

    // Path 2 Full Re-Pool: replacing StarCenter updates STAT & Mode 0 WGHT
    let wet_0_new: Vec<f32> = dry_0.iter().map(|&x| 0.4 * x).collect();
    let new_win = StmcTrajectoryWindow {
        control_id: 0,
        ctrl_kind: 0,
        active_opt: 0xFF,
        u_phys: 4.0,
        window_weight: 1.0,
        emr_velocity: 0.0,
    };
    let proj_replaced = replace_capture(
        &proj_removed,
        "star_0",
        VaultCaptureSpec {
            stem_id: String::from("star_0"),
            is_star_center: true,
            enabled: true,
            pad_gain_f32: 1.0,
            fractional_lag_samples_f32: 0.0,
            controls_meta: BTreeMap::from([
                (String::from("gain"), serde_json::json!(4.0)),
                (String::from("channel"), serde_json::json!("Lead")),
            ]),
            provenance: StemProvenance {
                input_pcm_sha256: compute_pcm_sha256(&dry_0),
                target_pcm_sha256: compute_pcm_sha256(&wet_0_new),
                num_frames: n as u64,
            },
            user_metadata: BTreeMap::new(),
            extracted_physics: None,
            windows: vec![new_win],
            band_psd_summary_88: [0u8; 88],
            n_comb_bins: 24,
            spectral_summary_4096: [0u8; 4_096],
            wet_coreset_fp16: f32_to_fp16_rne_bytes(&wet_0_new),
            optional_spcm: None,
        },
    )
    .unwrap();
    let view_replaced = DnamProjView::from_bytes(&proj_replaced, &crypto).unwrap();
    assert_eq!(
        view_replaced.manifest.base.controls_schema[0].safe_clamped_range,
        Some([4.0, 4.0])
    );
    assert_ne!(
        view_replaced.chunk_slice(ProjChunkKind::Stat, 0).unwrap(),
        view_s0.chunk_slice(ProjChunkKind::Stat, 0).unwrap()
    );
}

#[test]
fn test_decimator_pchip_interval_audit_and_freeze_latch_regressions() {
    let crypto = DefaultCryptoProvider::default();

    // 1. Kaiser 309-tap decimator NaN & f32::MAX overflow guard
    let mut wild_192k = vec![1.0e40_f64; 32];
    wild_192k[4] = f64::NAN;
    let dec = dnam_capture::whitebox::kaiser_sinc_decimate_4x_to_48khz(&wild_192k);
    assert_eq!(dec.len(), 8);
    assert!(dec.iter().all(|v| v.is_finite()));

    // 2. PCHIP extreme finite-span & linear slope preservation guards
    let pchip_2 = dnam_capture::whitebox::pchip_resample_to_rate(
        &[f64::MIN, 1.0],
        &[-f64::MAX, f64::MAX],
        4,
        f64::INFINITY,
    );
    let pchip_3 = dnam_capture::whitebox::pchip_resample_to_rate(
        &[-1.0e308, 1.0e308, 1.5e308],
        &[0.0, 0.9 * f64::MAX, f64::MAX],
        4,
        1.0,
    );
    assert!(pchip_2.iter().chain(&pchip_3).all(|v| v.is_finite()));
    let wide_lin = dnam_capture::whitebox::pchip_resample_to_rate(
        &[-0.71 * f64::MAX, 0.0, 0.71 * f64::MAX],
        &[-0.994 * f64::MAX, 0.0, 0.994 * f64::MAX],
        4,
        1.0,
    );
    assert!((wide_lin[1] - 1.4).abs() < 1.0e-9);

    // 3. 1-ULP IEEE-754 binade boundary & powf underflow/overflow in evaluate_joint_energy_interval_audit
    let eps = f64::EPSILON;
    let req_snr = 64.0 + 64.0 * eps;
    let goal_db = -10.0 * req_snr.log10();
    let audit = evaluate_joint_energy_interval_audit(
        [[128.0 - 64.0 * eps; 2]; 3],
        [[2.2; 2]; 3],
        [[2.0 - 2.0 * eps; 2]; 3],
        [goal_db; 3],
        goal_db,
    );
    for b in 0..3 {
        assert!(audit.qualified_bounds_lin[b][0] <= audit.qualified_bounds_lin[b][1]);
    }
    assert_eq!(
        evaluate_joint_energy_interval_audit(
            [[0.0, 0.0]; 3],
            [[1.2e-6, 1.2e-6]; 3],
            [[1.0e-6, 1.0e-6]; 3],
            [3500.0; 3],
            -20.0,
        )
        .decision,
        IntervalAuditDecision::CertifiedPass
    );
    assert_eq!(
        evaluate_joint_energy_interval_audit(
            [[1.0e200, 1.0e200]; 3],
            [[1.2e-120, 1.2e-120]; 3],
            [[1.0e-120, 1.0e-120]; 3],
            [-3500.0; 3],
            -3500.0,
        )
        .decision,
        IntervalAuditDecision::CertifiedPass
    );
    assert_eq!(
        evaluate_joint_energy_interval_audit(
            [[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]],
            [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0]; 3],
            [3500.0; 3],
            3500.0,
        )
        .decision,
        IntervalAuditDecision::CertifiedFail
    );

    // 4. FreezeLatch calibrated control preserved across render_stem with Some([0, 0]) freeze window
    let mut frz_hdr = DnamJsonHeader::default_reference();
    frz_hdr.io_config.capabilities.supports_freeze = true;
    frz_hdr.controls_schema = vec![make_range_control("gain", 0.0, 10.0, 5.0)];
    frz_hdr.controls_schema[0].calibration = Some(ControlCalibration {
        role: ControlCalibrationRole::FreezeLatch,
        lobatto_physical_anchors: None,
        conditional_physical_anchors: BTreeMap::new(),
        tempo_sync_default_subdivision: None,
    });
    let frz_arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    let frz_dnam = build_dnam_container(&mut frz_hdr, &frz_arena, &frz_arena, false).unwrap();
    let frz_view = DnamContainerView::from_bytes(&frz_dnam, &crypto).unwrap();
    let mut frz_eng =
        dnam_engine::DnamEngine::<DefaultMathBackend>::from_bytes(&frz_dnam, Tier::Lite, 48_000)
            .unwrap();
    let max_c0 = frz_view.header.controls_schema[0]
        .declared_range
        .unwrap_or([0.0, 10.0])[1];
    let frz_stim = StemStimulus {
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
    let _ = frz_eng.render_stem(&frz_stim).unwrap();
    assert!(frz_eng.stage_pipeline.stage4.freeze);
}
