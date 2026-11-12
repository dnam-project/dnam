#![forbid(unsafe_code)]
//! Adversarial boundary & regression suite for `.dnam-proj` (`DNPJ`, `v2.0`) containers,
//! `STAT` PSD diagonal checks, strict JCS duplicate-key rejection, `STMC` bijection guards,
//! `P > 8` multi-control Mode A `R = 36` trajectory sweeps, two-step `> 64 KB` `HTTP Range`
//! header parsing, compacted `StarCenter` mute/promote/all-muted lifecycles, and growable
//! `DMU1` + `control_user_metadata` runtime `.dnam` exports.

use dnam_capture::CORESET_FP16_FRAMES_PER_STEM;
use dnam_capture::align::compute_pcm_sha256;
use dnam_capture::proj::{
    DnamProjManifest, DnamProjView, PROJ_FLAG_COMPACTED_CAPSULES, PROJ_FLAG_HAS_STAT,
    PROJ_STAT_BYTES_LITE, ProjChunkInput, ProjChunkKind, STMC_FLAG_ENABLED, STMC_FLAG_STAR_CENTER,
    StemProvenance, StemRegistryEntry, StmcTrajectoryWindow, VaultCaptureSpec, add_capture,
    build_dnam_proj_container, build_proj_inpc_capsule, build_proj_stmc_capsule,
    compact_to_capsules, export_runtime_dnam, f32_to_fp16_rne_bytes, parse_proj_manifest_range,
    replace_capture, set_capture_enabled, verify_proj_stat_uplo_u, verify_proj_stmc_capsule,
};
use dnam_capture::solver::uplo_u_index;
use dnam_engine::Tier;
use dnam_engine::container::DnamContainerView;
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::schema::{ControlSchemaDef, DnamError, PresetDef};
use std::collections::BTreeMap;

fn make_range_control(id: &str, min_v: f32, max_v: f32, def_v: f32) -> ControlSchemaDef {
    serde_json::from_value(serde_json::json!({
        "id": id,
        "name": id,
        "type": "range",
        "lobatto_degree": 3,
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
fn test_stat_psd_diagonal_duplicate_json_keys_and_stmc_bijection_guards() {
    // verify[container.proj.stat-uplo-u]
    // verify[container.proj.jcs-manifest]
    // verify[container.proj.stmc-capsule]
    // verify[container.proj.lifecycle-states]

    // 1. STAT uplo='U' accepts non-negative diagonals with negative off-diagonal, but rejects G_jj < 0
    let mut stat_buf = vec![0u8; PROJ_STAT_BYTES_LITE];
    for j in 0..64 {
        let diag_off = uplo_u_index(j, j) * 8;
        stat_buf[diag_off..diag_off + 8].copy_from_slice(&1.0_f64.to_le_bytes());
    }
    let off_01 = uplo_u_index(0, 1) * 8;
    stat_buf[off_01..off_01 + 8].copy_from_slice(&(-0.5_f64).to_le_bytes());
    assert_eq!(verify_proj_stat_uplo_u(&stat_buf).unwrap(), 64);

    let bad_diag_off = uplo_u_index(3, 3) * 8;
    stat_buf[bad_diag_off..bad_diag_off + 8].copy_from_slice(&(-1.0e-9_f64).to_le_bytes());
    assert!(verify_proj_stat_uplo_u(&stat_buf).is_err());

    // 2. Strict JSON duplicate-key and UTF-8 BOM rejection in validate_and_normalize_json_str
    let base_json = serde_json::to_string(&DnamProjManifest::default_reference().base).unwrap();
    let dup_key_json = format!(
        r#"{{"version":"2.0.0","version":"2.0.0","container_kind":"dnam-proj","base":{base_json}}}"#
    );
    assert!(DnamProjManifest::validate_and_normalize_json_str(&dup_key_json).is_err());
    let bom_json = format!(
        "\u{FEFF}{{\"version\":\"2.0.0\",\"container_kind\":\"dnam-proj\",\"base\":{base_json}}}"
    );
    assert!(DnamProjManifest::validate_and_normalize_json_str(&bom_json).is_err());

    // 3. STMC stem_index, STMC_FLAG_STAR_CENTER, non-empty stems_registry chunk presence, and
    //    compact_to_capsules raw-only rejection
    let valid_sha = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
    let mono_fp16 = vec![0u8; 24_000];
    let inpc = build_proj_inpc_capsule(1, &[1; 256], &[2; 320], &mono_fp16).unwrap();
    let win = StmcTrajectoryWindow {
        control_id: 0,
        ctrl_kind: 0,
        active_opt: 0xFF,
        u_phys: 0.0,
        window_weight: 1.0,
        emr_velocity: 0.0,
    };

    let mut manifest = DnamProjManifest::default_reference();
    manifest.stems_registry.push(StemRegistryEntry {
        stem_id: String::from("s0"),
        chunk_id: 2,
        enabled: true,
        is_star_center: true,
        controls_meta: BTreeMap::new(),
        provenance: StemProvenance {
            input_pcm_sha256: valid_sha.to_string(),
            target_pcm_sha256: valid_sha.to_string(),
            num_frames: 12_000,
        },
        user_metadata: BTreeMap::new(),
        extracted_physics: None,
    });

    // Non-empty stems_registry with zero STMC/SPCM chunks must fail closed
    assert!(build_dnam_proj_container(&manifest, &[]).is_err());

    // Raw SPCM-only project with non-empty stems_registry is valid in active session,
    // but compact_to_capsules MUST refuse to strip SPCM without INPC + STMC!
    let spcm_only = build_dnam_proj_container(
        &manifest,
        &[ProjChunkInput {
            kind: ProjChunkKind::Spcm,
            chunk_id: 2,
            payload: make_spcm_chunk(&[0.1, -0.1, 0.05, -0.05]),
        }],
    )
    .unwrap();
    assert!(compact_to_capsules(&spcm_only).is_err());

    // STMC stem_index=0 != directory chunk_id=2 must fail closed
    let stmc_wrong_idx = build_proj_stmc_capsule(
        0,
        STMC_FLAG_ENABLED | STMC_FLAG_STAR_CENTER,
        1.0,
        0.0,
        &[0; 88],
        0,
        &[win],
        &[0; 4_096],
        &mono_fp16,
    )
    .unwrap();
    assert!(
        build_dnam_proj_container(
            &manifest,
            &[
                ProjChunkInput {
                    kind: ProjChunkKind::Inpc,
                    chunk_id: 0,
                    payload: inpc.clone(),
                },
                ProjChunkInput {
                    kind: ProjChunkKind::Stmc,
                    chunk_id: 2,
                    payload: stmc_wrong_idx,
                },
            ],
        )
        .is_err()
    );

    // STMC stem_index=2 matches chunk_id=2, but missing STMC_FLAG_STAR_CENTER disagrees with manifest
    let stmc_wrong_star = build_proj_stmc_capsule(
        2,
        STMC_FLAG_ENABLED,
        1.0,
        0.0,
        &[0; 88],
        0,
        &[win],
        &[0; 4_096],
        &mono_fp16,
    )
    .unwrap();
    assert!(
        build_dnam_proj_container(
            &manifest,
            &[
                ProjChunkInput {
                    kind: ProjChunkKind::Inpc,
                    chunk_id: 0,
                    payload: inpc,
                },
                ProjChunkInput {
                    kind: ProjChunkKind::Stmc,
                    chunk_id: 2,
                    payload: stmc_wrong_star,
                },
            ],
        )
        .is_err()
    );
}

#[test]
fn test_p12_mode_a_r36_sweep_and_large_64kb_two_step_http_range() {
    // verify[container.proj.preamble-and-directory]
    // verify[container.proj.stmc-capsule]
    // verify[train.cache.proj-vault]
    let crypto = DefaultCryptoProvider::default();

    // P = 12 controls (9 Range knobs k0..k8 + 3 Switch selectors sw0..sw2)
    let mut manifest = DnamProjManifest::default_reference();
    let mut controls = Vec::with_capacity(12);
    for i in 0..9 {
        let (lo, hi, def) = if i == 8 {
            (-6.0, 6.0, 0.0)
        } else {
            (0.0, 10.0, 5.0)
        };
        controls.push(make_range_control(&format!("k{i}"), lo, hi, def));
    }
    for s in 0..3 {
        controls.push(make_switch_control(&format!("sw{s}")));
    }
    manifest.base.controls_schema = controls;
    let empty_proj = build_dnam_proj_container(&manifest, &[]).unwrap();

    let n = CORESET_FP16_FRAMES_PER_STEM;
    let dry: Vec<f32> = (0..n).map(|i| 0.35 * ((i as f32) * 0.031).sin()).collect();
    let wet: Vec<f32> = dry.iter().map(|&x| (2.2 * x).tanh()).collect();
    let inpc = build_proj_inpc_capsule(1, &[0x11; 256], &[0x22; 320], &f32_to_fp16_rne_bytes(&dry))
        .unwrap();

    // R = 36 Mode A continuous sweep windows:
    // - 18 windows sweep k0 (control_id = 0) across [0.0 .. 10.0]
    // - 15 windows sweep k8 (control_id = 8, beyond first-8 Arena 0 table!) across [-6.0 .. 6.0]
    // - 3 windows toggle sw0 (control_id = 9) across options 0, 1, 2
    // while controls_meta is completely empty!
    let mut windows = Vec::with_capacity(36);
    for r in 0..18 {
        windows.push(StmcTrajectoryWindow {
            control_id: 0,
            ctrl_kind: 0,
            active_opt: 0xFF,
            u_phys: 10.0 * (r as f32) / 17.0,
            window_weight: 1.0,
            emr_velocity: 0.2,
        });
    }
    for r in 0..15 {
        windows.push(StmcTrajectoryWindow {
            control_id: 8,
            ctrl_kind: 0,
            active_opt: 0xFF,
            u_phys: -6.0 + 12.0 * (r as f32) / 14.0,
            window_weight: 1.0,
            emr_velocity: 0.15,
        });
    }
    for opt in 0..3_u8 {
        windows.push(StmcTrajectoryWindow {
            control_id: 9,
            ctrl_kind: 1,
            active_opt: opt,
            u_phys: f32::from(opt),
            window_weight: 1.0,
            emr_velocity: 0.0,
        });
    }
    assert_eq!(windows.len(), 36);

    let proj_p12 = add_capture(
        &empty_proj,
        VaultCaptureSpec {
            stem_id: String::from("mode_a_sweep_0"),
            is_star_center: true,
            enabled: true,
            pad_gain_f32: 1.0,
            fractional_lag_samples_f32: 0.0,
            controls_meta: BTreeMap::new(),
            provenance: StemProvenance {
                input_pcm_sha256: compute_pcm_sha256(&dry),
                target_pcm_sha256: compute_pcm_sha256(&wet),
                num_frames: n as u64,
            },
            user_metadata: BTreeMap::new(),
            extracted_physics: None,
            windows,
            band_psd_summary_88: [0; 88],
            n_comb_bins: 32,
            spectral_summary_4096: [0; 4_096],
            wet_coreset_fp16: f32_to_fp16_rne_bytes(&wet),
            optional_spcm: None,
        },
        Some(&inpc),
    )
    .unwrap();

    let view_p12 = DnamProjView::from_bytes(&proj_p12, &crypto).unwrap();
    // k0 (idx 0) and k8 (idx 8) unlocked full declared ranges via Mode A windows; k1 (idx 1) clamped to default [5.0, 5.0]
    assert_eq!(
        view_p12.manifest.base.controls_schema[0].safe_clamped_range,
        Some([0.0, 10.0])
    );
    assert_eq!(
        view_p12.manifest.base.controls_schema[1].safe_clamped_range,
        Some([5.0, 5.0])
    );
    assert_eq!(
        view_p12.manifest.base.controls_schema[8].safe_clamped_range,
        Some([-6.0, 6.0])
    );
    assert_eq!(
        view_p12.manifest.base.controls_schema[9].active_options,
        Some(vec![
            String::from("Clean"),
            String::from("Crunch"),
            String::from("Lead")
        ])
    );
    assert_eq!(
        view_p12.manifest.base.controls_schema[10].active_options,
        Some(vec![String::from("Clean")])
    );

    let rt_dnam = export_runtime_dnam(&proj_p12, None).unwrap();
    let rt_view = DnamContainerView::from_bytes(&rt_dnam, &crypto).unwrap();
    assert_eq!(rt_view.clamp_descriptors.len(), 12);
    assert_eq!(rt_view.clamp_descriptors[0].lobatto_degree, 3);
    assert_eq!(rt_view.clamp_descriptors[1].lobatto_degree, 0);
    assert_eq!(rt_view.clamp_descriptors[8].lobatto_degree, 3);
    assert!(
        rt_view
            .cp1_contributors
            .iter()
            .any(|c| c.control_idx == 0 && c.lanes == 3)
    );
    assert!(
        rt_view
            .cp1_contributors
            .iter()
            .any(|c| c.control_idx == 8 && c.lanes == 3)
    );
    assert_eq!(
        rt_view
            .cp1_contributors
            .iter()
            .filter(|c| c.control_idx == 9)
            .count(),
        3
    );

    // Large > 64 KB manifest: verify 2-step HTTP Range contract (`bytes=0-65535` -> exact `expected` header end)
    let mut big_manifest = view_p12.manifest.clone();
    big_manifest.project_meta.rig_notes = Some("rig-session-log-".repeat(4_800));
    let mut retained_chunks = Vec::new();
    for (entry, &slice) in view_p12.directory.iter().zip(&view_p12.chunk_slices) {
        retained_chunks.push(ProjChunkInput {
            kind: entry.kind().unwrap(),
            chunk_id: entry.chunk_id,
            payload: slice.to_vec(),
        });
    }
    let big_proj = build_dnam_proj_container(&big_manifest, &retained_chunks).unwrap();
    let err = parse_proj_manifest_range(&big_proj[..65_536]).unwrap_err();
    let expected_hdr_end = match err {
        DnamError::BufferTooSmall {
            expected,
            actual: 65_536,
        } => expected,
        other => panic!("expected BufferTooSmall on 64 KB prefix, got {other:?}"),
    };
    assert!(expected_hdr_end > 65_536);
    assert_eq!(expected_hdr_end % 64, 0);
    assert!(expected_hdr_end < big_proj.len());

    let parsed_2step = parse_proj_manifest_range(&big_proj[..expected_hdr_end]).unwrap();
    assert_eq!(parsed_2step.payload_region_offset, expected_hdr_end);
    assert_eq!(parsed_2step.directory.len(), view_p12.directory.len());
    assert_eq!(
        parsed_2step.manifest.project_meta.rig_notes,
        big_manifest.project_meta.rig_notes
    );
}

#[test]
fn test_compacted_star_center_mute_promote_all_muted_replace_and_grown_dmu1_export() {
    // verify[container.proj.lifecycle-states]
    // verify[container.proj.wght-and-export]
    // verify[train.cache.proj-vault]
    let crypto = DefaultCryptoProvider::default();

    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.controls_schema = vec![
        make_range_control("gain", 0.0, 10.0, 5.0),
        make_range_control("tone", 0.0, 10.0, 5.0),
    ];
    let empty_proj = build_dnam_proj_container(&manifest, &[]).unwrap();

    let n = CORESET_FP16_FRAMES_PER_STEM;
    let dry: Vec<f32> = (0..n).map(|i| 0.32 * ((i as f32) * 0.023).sin()).collect();
    let wet_star: Vec<f32> = dry.iter().map(|&x| (1.2 * x).tanh()).collect();
    let wet_alt: Vec<f32> = dry.iter().map(|&x| (3.4 * x).tanh()).collect();
    let inpc = build_proj_inpc_capsule(1, &[0x55; 256], &[0x66; 320], &f32_to_fp16_rne_bytes(&dry))
        .unwrap();

    let p1 = add_capture(
        &empty_proj,
        VaultCaptureSpec {
            stem_id: String::from("star_0"),
            is_star_center: true,
            enabled: true,
            pad_gain_f32: 1.0,
            fractional_lag_samples_f32: 0.0,
            controls_meta: BTreeMap::from([(String::from("gain"), serde_json::json!(5.0))]),
            provenance: StemProvenance {
                input_pcm_sha256: compute_pcm_sha256(&dry),
                target_pcm_sha256: compute_pcm_sha256(&wet_star),
                num_frames: n as u64,
            },
            user_metadata: BTreeMap::new(),
            extracted_physics: None,
            windows: vec![StmcTrajectoryWindow {
                control_id: 0,
                ctrl_kind: 0,
                active_opt: 0xFF,
                u_phys: 5.0,
                window_weight: 1.0,
                emr_velocity: 0.0,
            }],
            band_psd_summary_88: [0; 88],
            n_comb_bins: 16,
            spectral_summary_4096: [0; 4_096],
            wet_coreset_fp16: f32_to_fp16_rne_bytes(&wet_star),
            optional_spcm: Some(make_spcm_chunk(&wet_star[..256])),
        },
        Some(&inpc),
    )
    .unwrap();

    // sweep_1 carries gain [2.0, 9.0] ONLY in STMC windows (empty controls_meta)
    let p2 = add_capture(
        &p1,
        VaultCaptureSpec {
            stem_id: String::from("sweep_1"),
            is_star_center: false,
            enabled: true,
            pad_gain_f32: 1.0,
            fractional_lag_samples_f32: 0.0,
            controls_meta: BTreeMap::new(),
            provenance: StemProvenance {
                input_pcm_sha256: compute_pcm_sha256(&dry),
                target_pcm_sha256: compute_pcm_sha256(&wet_alt),
                num_frames: n as u64,
            },
            user_metadata: BTreeMap::new(),
            extracted_physics: None,
            windows: vec![
                StmcTrajectoryWindow {
                    control_id: 0,
                    ctrl_kind: 0,
                    active_opt: 0xFF,
                    u_phys: 2.0,
                    window_weight: 1.0,
                    emr_velocity: 0.1,
                },
                StmcTrajectoryWindow {
                    control_id: 0,
                    ctrl_kind: 0,
                    active_opt: 0xFF,
                    u_phys: 9.0,
                    window_weight: 1.0,
                    emr_velocity: 0.1,
                },
            ],
            band_psd_summary_88: [0; 88],
            n_comb_bins: 16,
            spectral_summary_4096: [0; 4_096],
            wet_coreset_fp16: f32_to_fp16_rne_bytes(&wet_alt),
            optional_spcm: Some(make_spcm_chunk(&wet_alt[..256])),
        },
        None,
    )
    .unwrap();

    let compacted = compact_to_capsules(&p2).unwrap();
    let v_compacted = DnamProjView::from_bytes(&compacted, &crypto).unwrap();
    assert_ne!(
        v_compacted.preamble.lifecycle_flags & PROJ_FLAG_COMPACTED_CAPSULES,
        0
    );
    let stat_star0 = v_compacted
        .chunk_slice(ProjChunkKind::Stat, 0)
        .unwrap()
        .to_vec();

    // 1. Muting active StarCenter ("star_0") must preserve PROJ_FLAG_COMPACTED_CAPSULES,
    //    promote "sweep_1" to StarCenter (in both manifest and STMC flags), and recompute STAT!
    let p_star_muted = set_capture_enabled(&compacted, "star_0", false).unwrap();
    let v_star_muted = DnamProjView::from_bytes(&p_star_muted, &crypto).unwrap();
    assert_ne!(
        v_star_muted.preamble.lifecycle_flags & PROJ_FLAG_COMPACTED_CAPSULES,
        0
    );
    assert!(!v_star_muted.manifest.stems_registry[0].is_star_center);
    assert!(v_star_muted.manifest.stems_registry[1].is_star_center);
    let stmc_1 =
        verify_proj_stmc_capsule(v_star_muted.chunk_slice(ProjChunkKind::Stmc, 1).unwrap())
            .unwrap();
    assert!(stmc_1.is_star_center());
    assert_ne!(
        v_star_muted.chunk_slice(ProjChunkKind::Stat, 0).unwrap(),
        stat_star0.as_slice()
    );

    // 2. Muting ALL stems ("sweep_1" as well) must clear STAT and zero Mode-0 WGHT!
    let p_all_muted = set_capture_enabled(&p_star_muted, "sweep_1", false).unwrap();
    let v_all_muted = DnamProjView::from_bytes(&p_all_muted, &crypto).unwrap();
    assert_eq!(v_all_muted.preamble.lifecycle_flags & PROJ_FLAG_HAS_STAT, 0);
    assert!(v_all_muted.chunk_slice(ProjChunkKind::Stat, 0).is_none());
    let wght_muted = DnamContainerView::from_bytes(
        v_all_muted.chunk_slice(ProjChunkKind::Wght, 0).unwrap(),
        &crypto,
    )
    .unwrap();
    assert!(
        wght_muted.arena_storage.as_bytes()[4_928..4_928 + 512]
            .iter()
            .all(|&b| b == 0)
    );

    // 3. Re-enabling "star_0" promotes it back to active StarCenter and restores STAT; then re-enable "sweep_1"
    let p_star_back = set_capture_enabled(&p_all_muted, "star_0", true).unwrap();
    let p_both_back = set_capture_enabled(&p_star_back, "sweep_1", true).unwrap();
    let v_both_back = DnamProjView::from_bytes(&p_both_back, &crypto).unwrap();
    assert_eq!(
        v_both_back.manifest.base.controls_schema[0].safe_clamped_range,
        Some([2.0, 9.0])
    );

    // 4. replace_capture rejects colliding stem_id ("star_0"), and replacing "sweep_1" with a tone-only
    //    stem refreshes gain ("control_id = 0") via old_windows so gain clamps back to [5.0, 5.0] (deg = 0)!
    let mut tone_spec = VaultCaptureSpec {
        stem_id: String::from("star_0"),
        is_star_center: false,
        enabled: true,
        pad_gain_f32: 1.0,
        fractional_lag_samples_f32: 0.0,
        controls_meta: BTreeMap::from([(String::from("tone"), serde_json::json!(8.0))]),
        provenance: StemProvenance {
            input_pcm_sha256: compute_pcm_sha256(&dry),
            target_pcm_sha256: compute_pcm_sha256(&wet_alt),
            num_frames: n as u64,
        },
        user_metadata: BTreeMap::new(),
        extracted_physics: None,
        windows: vec![StmcTrajectoryWindow {
            control_id: 1,
            ctrl_kind: 0,
            active_opt: 0xFF,
            u_phys: 8.0,
            window_weight: 1.0,
            emr_velocity: 0.0,
        }],
        band_psd_summary_88: [0; 88],
        n_comb_bins: 16,
        spectral_summary_4096: [0; 4_096],
        wet_coreset_fp16: f32_to_fp16_rne_bytes(&wet_alt),
        optional_spcm: None,
    };
    assert!(replace_capture(&p_both_back, "sweep_1", tone_spec.clone()).is_err());

    tone_spec.stem_id = String::from("tone_1");
    let p_replaced = replace_capture(&p_both_back, "sweep_1", tone_spec).unwrap();
    let v_replaced = DnamProjView::from_bytes(&p_replaced, &crypto).unwrap();
    assert_eq!(
        v_replaced.manifest.base.controls_schema[0].safe_clamped_range,
        Some([5.0, 5.0])
    );
    let rt_after_rep = DnamContainerView::from_bytes(
        v_replaced.chunk_slice(ProjChunkKind::Wght, 0).unwrap(),
        &crypto,
    )
    .unwrap();
    assert!(
        rt_after_rep
            .cp1_contributors
            .iter()
            .all(|c| c.control_idx != 0)
    );

    // 5. Runtime .dnam export propagates controls_schema[*].user_metadata AND grows a compact 48B DMU1
    //    suffix cleanly up to the 4 KB DMU1 ceiling (while rejecting > 4,080 B with HeaderCapacityExceeded)!
    let orig_wght = v_replaced.chunk_slice(ProjChunkKind::Wght, 0).unwrap();
    let orig_layout =
        dnam_engine::container::validate_v2_dnam_layout_with_max_profile(orig_wght, 1).unwrap();
    let mut compact_wght = orig_wght[..orig_layout.mutable_extent.0].to_vec();
    let dmu_48 = dnam_engine::container::build_dmu1_mutable_frame(b"{}", 48).unwrap();
    dnam_engine::container::append_dmue_mutable_suffix(&mut compact_wght, &dmu_48);

    let mut grown_manifest = v_replaced.manifest.clone();
    grown_manifest.base.controls_schema[0]
        .user_metadata
        .insert(String::from("midi_cc"), serde_json::json!(21));
    for i in 0..12 {
        grown_manifest.base.presets.push(PresetDef {
            id: format!("artist_preset_{i:02}"),
            name: format!("Artist Stage Rig Preset #{i:02}"),
            description: Some(String::from("Extended tour bank preset notes")),
            controls: BTreeMap::from([(String::from("gain"), serde_json::json!(5.0))]),
            user_metadata: BTreeMap::new(),
        });
    }
    let mut chunks = Vec::new();
    for (entry, &slice) in v_replaced.directory.iter().zip(&v_replaced.chunk_slices) {
        let kind = entry.kind().unwrap();
        chunks.push(ProjChunkInput {
            kind,
            chunk_id: entry.chunk_id,
            payload: if kind == ProjChunkKind::Wght {
                compact_wght.clone()
            } else {
                slice.to_vec()
            },
        });
    }
    let grown_proj = build_dnam_proj_container(&grown_manifest, &chunks).unwrap();
    let exported_full = export_runtime_dnam(&grown_proj, Some(Tier::Full)).unwrap();
    let exported_view = DnamContainerView::from_bytes(&exported_full, &crypto).unwrap();
    assert_eq!(exported_view.preamble.highest_tier(), Tier::Full);
    assert_eq!(exported_view.header.presets.len(), 12);
    assert_eq!(
        exported_view.header.controls_schema[0]
            .user_metadata
            .get("midi_cc"),
        Some(&serde_json::json!(21))
    );

    // Overflowing the 4,080 B DMU1 trailer ceiling must return HeaderCapacityExceeded cleanly
    let mut over_manifest = grown_manifest;
    for i in 12..50 {
        over_manifest.base.presets.push(PresetDef {
            id: format!("artist_preset_{i:02}"),
            name: format!("Artist Stage Rig Preset #{i:02}"),
            description: Some("Overflow tour bank notes — ".repeat(6)),
            controls: BTreeMap::from([(String::from("gain"), serde_json::json!(5.0))]),
            user_metadata: BTreeMap::new(),
        });
    }
    let over_proj = build_dnam_proj_container(&over_manifest, &chunks).unwrap();
    assert!(matches!(
        export_runtime_dnam(&over_proj, None),
        Err(DnamError::HeaderCapacityExceeded { .. })
    ));
}
