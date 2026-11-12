#![forbid(unsafe_code)]
//! Step 2.3 Adversarial & Edge-Case Regression Suite (`dnam_capture::solver`, `dnam_capture::proj`,
//! `dnam_capture::stimulus::null_test`, `dnam_capture::whitebox`, and `dnam_engine`).

use std::collections::BTreeMap;

use dnam_capture::audit::{run_full_3phase_arena_audit, run_head_only_3phase_arena_audit};
use dnam_capture::proj::{
    DnamProjManifest, DnamProjView, ProjChunkInput, ProjChunkKind, StemProvenance,
    StmcTrajectoryWindow, VaultCaptureSpec, add_capture, add_capture_with_options,
    build_dnam_proj_container, build_dnam_proj_container_with_cp1, build_proj_inpc_capsule,
    export_runtime_dnam, remove_capture_with_options, replace_capture_with_options,
    set_capture_enabled_with_options,
};
use dnam_capture::solver::stage_b_c::{HEAD_BASES, ULTRA_MODES_5_TO_8_BASE};
use dnam_capture::solver::{
    BuyerWatermarkConfig, SolverOptions, compute_integrated_lufs, contract_256d_gram_tiled_wide192,
    derive_buyer_phase_seed, export_runtime_dnam_watermarked,
    identify_pole_angles_blackbox_from_audio, lift_stage_a_backbone,
    render_blackbox_probe_response, uplo_u_index, write_inline_cp1_if_fits,
};
use dnam_capture::stimulus::{
    AdaptiveTrainingOptions, TopologyProbeMode, run_live_null_test_and_autopatch,
    train_adaptive_target,
};
use dnam_capture::whitebox::{FaustRunner, SpiceRunner, parse_faust_json_controls};
use dnam_capture::{
    CORESET_FP16_FRAMES_PER_STEM, DistillationTarget, StemResponse, StemStimulus, f32_to_fp16_bits,
};
use dnam_engine::container::{
    ARENA_OFFSET_ULTRA_END, DnamContainerView, build_dnam_container_with_cp1,
    strip_container_to_tier,
};
use dnam_engine::control::{ConditionalClampRule, ControlClampDescriptor, Cp1ContributorSlot};
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::provenance::{
    DefaultCryptoProvider, apply_cascade_head_gauge, score_phase_coding_blackbox,
};
use dnam_engine::schema::{ControlSchemaDef, DnamError, DnamJsonHeader};
use dnam_engine::stages::weights::ARENA0_TIER_GAIN_TRIM_OFFSET;
use dnam_engine::{DnamEngine, Tier};

fn write_f32(buf: &mut [u8], off: usize, val: f32) {
    buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
}

fn read_f32(buf: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

fn fp16_bytes(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        out.extend_from_slice(&f32_to_fp16_bits(s).to_le_bytes());
    }
    out
}

fn make_range_ctrl(id: &str, lo: f32, hi: f32, def: f32) -> ControlSchemaDef {
    serde_json::from_value(serde_json::json!({
        "id": id,
        "name": id,
        "type": "range",
        "lobatto_degree": 2,
        "declared_range": [lo, hi],
        "safe_clamped_range": [lo, hi],
        "default": def
    }))
    .unwrap()
}

fn make_spec(
    id: &str,
    is_star: bool,
    meta: BTreeMap<String, serde_json::Value>,
    wet_mono: &[f32],
    windows: Vec<StmcTrajectoryWindow>,
) -> VaultCaptureSpec {
    VaultCaptureSpec {
        stem_id: id.to_string(),
        is_star_center: is_star,
        enabled: true,
        pad_gain_f32: 1.0,
        fractional_lag_samples_f32: 0.0,
        controls_meta: meta,
        provenance: StemProvenance {
            input_pcm_sha256: "0".repeat(64),
            target_pcm_sha256: "1".repeat(64),
            num_frames: CORESET_FP16_FRAMES_PER_STEM as u64,
        },
        user_metadata: BTreeMap::new(),
        extracted_physics: None,
        windows,
        band_psd_summary_88: [0u8; 88],
        n_comb_bins: 32,
        spectral_summary_4096: [0u8; 4_096],
        wet_coreset_fp16: fp16_bytes(wet_mono),
        optional_spcm: None,
    }
}

#[test]
fn test_wide192_tiled_intra_and_cross_batch_cancellation_and_prime_stride() {
    // verify[train.gram.wide192]
    // verify[train.gram.accuracy-contract]
    let tiny_f32 = 2.0_f32.powi(-30);
    let tiny_f64 = f64::from(tiny_f32);
    let big_f32 = 1.0e8_f32;

    let n = 131usize;
    let mut feats = vec![[0.0_f32; 256]; n];
    let mut targets = vec![[0.0_f64; 2]; n];
    let weights = vec![1.0_f64; n];

    // Pair (0, 7) intra-batch [1e8, 2^-30, -1e8]
    feats[0][0] = big_f32;
    feats[0][7] = big_f32;
    feats[1][0] = tiny_f32;
    feats[1][7] = 1.0;
    feats[2][0] = -big_f32;
    feats[2][7] = big_f32;

    // Pair (3, 19) cross-batch [1e8, 2^-30] in batch 0 vs [-1e8] in batch 1
    feats[10][3] = big_f32;
    feats[10][19] = big_f32;
    feats[11][3] = tiny_f32;
    feats[11][19] = 1.0;
    feats[74][3] = -big_f32;
    feats[74][19] = big_f32;

    // Pair (5, 22) intra-batch [1.0, 2^-30, -1.0] + target cross-vector cancellation
    feats[20][5] = 1.0;
    feats[20][22] = 1.0;
    targets[20] = [1.0e8, 1.0];
    feats[21][5] = tiny_f32;
    feats[21][22] = 1.0;
    targets[21] = [1.0, tiny_f64];
    feats[22][5] = 1.0;
    feats[22][22] = -1.0;
    targets[22] = [-1.0e8, 1.0];

    let (g, c) = contract_256d_gram_tiled_wide192(&feats, &targets, &weights, 1);
    assert!((g[uplo_u_index(0, 7)] - tiny_f64).abs() <= 1.0e-20);
    assert!((g[uplo_u_index(3, 19)] - tiny_f64).abs() <= 1.0e-20);
    assert!((g[uplo_u_index(5, 22)] - tiny_f64).abs() <= 1.0e-20);
    assert!((c[5][0] - tiny_f64).abs() <= 1.0e-20);
    assert!((c[22][1] - tiny_f64).abs() <= 1.0e-20);

    // Non-divisible prime stride = 7 integral scaling on constant unit feature
    let const_feats = vec![[1.0_f32; 256]; 100];
    let zero_tgt = vec![[0.0_f64; 2]; 100];
    let unit_w = vec![1.0_f64; 100];
    let (g_s7, _) = contract_256d_gram_tiled_wide192(&const_feats, &zero_tgt, &unit_w, 7);
    assert!((g_s7[uplo_u_index(0, 0)] - 100.0).abs() <= 1.0e-11);

    // Normal-scale intra-batch cancellation (max <= 1.0, exponent span 60 > 53 bits):
    // [1.0, 2^-30, -1.0] * [1.0, 2^-30, 1.0] == 2^-60 and target [1.0, 2^-60, -1.0] == 2^-60.
    let two_neg60 = 2.0_f64.powi(-60);
    let mut norm_feats = vec![[0.0_f32; 256]; 8];
    let mut norm_targets = vec![[0.0_f64; 2]; 8];
    let norm_weights = vec![1.0_f64; 8];
    // Off-diagonal tile (col 0 vs col 7) + diagonal tile (col 1 vs col 2) + non-wide cols 3 & 6 + constant col 12
    norm_feats[0][0] = 1.0;
    norm_feats[0][7] = 1.0;
    norm_feats[0][1] = 1.0;
    norm_feats[0][2] = 1.0;
    norm_feats[0][3] = 2.0;
    norm_feats[0][6] = 3.0;
    norm_feats[0][12] = 1.0;
    norm_targets[0] = [1.0, -1.0];

    norm_feats[1][0] = tiny_f32;
    norm_feats[1][7] = tiny_f32;
    norm_feats[1][1] = tiny_f32;
    norm_feats[1][2] = tiny_f32;
    norm_feats[1][12] = 1.0;
    norm_targets[1] = [two_neg60, two_neg60];

    norm_feats[2][0] = -1.0;
    norm_feats[2][7] = 1.0;
    norm_feats[2][1] = -1.0;
    norm_feats[2][2] = 1.0;
    norm_feats[2][12] = 1.0;
    norm_targets[2] = [-1.0, 1.0];

    let (g_norm, c_norm) =
        contract_256d_gram_tiled_wide192(&norm_feats, &norm_targets, &norm_weights, 1);
    assert_eq!(g_norm[uplo_u_index(0, 7)].to_bits(), two_neg60.to_bits());
    assert_eq!(g_norm[uplo_u_index(1, 2)].to_bits(), two_neg60.to_bits());
    assert_eq!(g_norm[uplo_u_index(3, 3)], 4.0);
    assert_eq!(g_norm[uplo_u_index(3, 6)], 6.0);
    assert_eq!(c_norm[12][0].to_bits(), two_neg60.to_bits());
    assert_eq!(c_norm[12][1].to_bits(), two_neg60.to_bits());

    // Weight-driven intra-batch cancellation: w = [1.0, 2^-60, 1.0] with normal-scale unit features
    let mut w_feats = vec![[0.0_f32; 256]; 3];
    w_feats[0][0] = 1.0;
    w_feats[0][7] = 1.0;
    w_feats[1][0] = 1.0;
    w_feats[1][7] = 1.0;
    w_feats[2][0] = -1.0;
    w_feats[2][7] = 1.0;
    let w_targets = vec![[0.0_f64; 2]; 3];
    let raw_w = vec![1.0_f64, two_neg60, 1.0_f64];
    let (g_w, _) = contract_256d_gram_tiled_wide192(&w_feats, &w_targets, &raw_w, 1);
    assert_eq!(g_w[uplo_u_index(0, 7)].to_bits(), two_neg60.to_bits());

    // Partial intra-batch cancellation (`[1.0, 2^-28, -1.0, 2^-9]` -> `2^-18 + 2^-56` bit-exact)
    // and 3-batch normal-scale cross-batch cancellation (`+1.0` in B0, `+2^-60` in B1, `-1.0` in B2).
    let mut adv_feats = vec![[0.0_f32; 256]; 520];
    let adv_targets = vec![[0.0_f64; 2]; 520];
    let adv_weights = vec![1.0_f64; 520];
    adv_feats[0][0] = 1.0;
    adv_feats[0][7] = 1.0;
    adv_feats[1][0] = 2.0_f32.powi(-28);
    adv_feats[1][7] = 2.0_f32.powi(-28);
    adv_feats[2][0] = -1.0;
    adv_feats[2][7] = 1.0;
    adv_feats[3][0] = 2.0_f32.powi(-9);
    adv_feats[3][7] = 2.0_f32.powi(-9);

    adv_feats[0][4] = 1.0;
    adv_feats[0][11] = 1.0;
    adv_feats[256][4] = tiny_f32;
    adv_feats[256][11] = tiny_f32;
    adv_feats[512][4] = -1.0;
    adv_feats[512][11] = 1.0;

    let (g_adv, _) = contract_256d_gram_tiled_wide192(&adv_feats, &adv_targets, &adv_weights, 1);
    let expected_partial = 2.0_f64.powi(-18) + 2.0_f64.powi(-56);
    assert_eq!(
        g_adv[uplo_u_index(0, 7)].to_bits(),
        expected_partial.to_bits()
    );
    assert_eq!(g_adv[uplo_u_index(4, 11)].to_bits(), two_neg60.to_bits());
}

#[test]
fn test_engine_immediate_control_sync_reload_reset_and_tier_gain_trim() {
    // verify[train.pipeline.stage-e-pack]
    // verify[train.pipeline.isomorphism-parity]
    let mut hdr = DnamJsonHeader::default_reference();
    hdr.io_config.inputs = 4;
    hdr.io_config.output_mode = "stereo".to_string();
    hdr.controls_schema = vec![make_range_ctrl("drive", 0.0, 10.0, 0.0)];

    let mut arena_a = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    let mut seed_m = DnamProjManifest::default_reference();
    seed_m.base = hdr.clone();
    let dry_seed: Vec<[f32; 2]> = (0..1_024)
        .map(|i| {
            let v = 0.35 * ((i as f32) * 0.09).sin();
            [v, v]
        })
        .collect();
    let _ = lift_stage_a_backbone(
        &seed_m,
        &mut arena_a,
        &dry_seed,
        &dry_seed,
        None,
        Default::default(),
    );
    let desc = ControlClampDescriptor {
        ctrl_kind: 0,
        lobatto_degree: 2,
        active_options_mask: 0,
        default_val: 0.0,
        safe_min_base: 0.0,
        safe_max_base: 10.0,
        scale_norm: 0.2,
        offset_norm: -1.0,
        anchor_nodes_norm: [-1.0, 1.0, 0.0],
        _reserved: 0,
        cond_rules: [ConditionalClampRule::empty(); 3],
    };
    arena_a[8_512..8_576].copy_from_slice(&desc.to_bytes());
    write_f32(&mut arena_a, 256, 0.85);
    write_f32(&mut arena_a, 272, 0.12);
    // Row 4 of 16x16 W_ctrl (`9_536 + 4 * 64`) maps `z_head[0]` into `stage_targets[4]` (Stage II drive)
    write_f32(&mut arena_a, 9_536 + 4 * 64, 3.0);

    let mut slot = Cp1ContributorSlot {
        lanes: 2,
        ctrl_kind: 0,
        control_idx: 0,
        ..Cp1ContributorSlot::zero()
    };
    slot.main_base[0][0] = 1.0;
    slot.emb[0] = 1.0;
    slot.v_base[0] = 1.0;
    for d in 0..128 {
        write_f32(&mut arena_a, HEAD_BASES[0] + d * 4, 0.2);
        // Mode 1 L/R head weights (`HEAD_BASES[0] + 512..1_024`) driven by `z_head[0]`
        write_f32(&mut arena_a, HEAD_BASES[0] + 512 + d * 4, 0.8);
    }
    write_inline_cp1_if_fits(&mut arena_a, &[slot]);

    let trims = [2.0_f32, 0.5, 1.25, 1.0];
    for (t, &tr) in trims.iter().enumerate() {
        write_f32(&mut arena_a, ARENA0_TIER_GAIN_TRIM_OFFSET + t * 4, tr);
    }

    let dnam_a =
        build_dnam_container_with_cp1(&mut hdr.clone(), &arena_a, &arena_a, false, &[slot])
            .unwrap();

    // 1. Pre-tick immediate sync (samples 0..4, well before sample 16's first 3 kHz control tick!)
    let mut eng_def =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_a, Tier::Ultra, 48_000).unwrap();
    eng_def.set_control_immediate(0, 0.0);
    let mut y_pre_def = [0.0_f32; 2];
    for _ in 0..4 {
        y_pre_def = eng_def.step_frame_tora(0.35, 0.35, false);
    }

    let mut eng_hi =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_a, Tier::Ultra, 48_000).unwrap();
    eng_hi.set_control_immediate(0, 10.0);
    let mut y_pre_hi = [0.0_f32; 2];
    for _ in 0..4 {
        y_pre_hi = eng_hi.step_frame_tora(0.35, 0.35, false);
    }
    assert!(
        (y_pre_hi[0] - y_pre_def[0]).abs() > 1.0e-3,
        "set_control_immediate must update current_stage_params & Mode weights before sample 16 tick: hi={y_pre_hi:?} vs def={y_pre_def:?}"
    );

    // Verify reset_controls_to_defaults snaps back to default immediately before sample 16 tick
    let mut eng_reset =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_a, Tier::Ultra, 48_000).unwrap();
    eng_reset.set_control_immediate(0, 10.0);
    eng_reset.reset_controls_to_defaults();
    let mut eng_fresh_def =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_a, Tier::Ultra, 48_000).unwrap();
    for _ in 0..4 {
        assert_eq!(
            eng_reset.step_frame_tora(0.35, 0.35, false)[0].to_bits(),
            eng_fresh_def.step_frame_tora(0.35, 0.35, false)[0].to_bits(),
            "reset_controls_to_defaults must immediately clear pre-tick control state"
        );
    }

    // 2. reload_model_in_place must clear stale current_stage_params from eng_hi
    let mut arena_b = arena_a.clone();
    arena_b[9_536..10_560].fill(0);
    write_inline_cp1_if_fits(&mut arena_b, &[]);
    let dnam_b =
        build_dnam_container_with_cp1(&mut hdr.clone(), &arena_b, &arena_b, false, &[]).unwrap();
    eng_hi
        .reload_model_in_place(&dnam_b, Tier::Ultra, 48_000)
        .unwrap();
    let mut eng_b_fresh =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_b, Tier::Ultra, 48_000).unwrap();
    for _ in 0..4 {
        assert_eq!(
            eng_hi.step_frame_tora(0.35, 0.35, false)[0].to_bits(),
            eng_b_fresh.step_frame_tora(0.35, 0.35, false)[0].to_bits(),
            "reload_model_in_place leaked stale current_stage_params during pre-tick window"
        );
    }

    // 3. Verify ARENA0_TIER_GAIN_TRIM_OFFSET scales live 1x & 4x 2-port and 4CM outputs + 0 ULP stripping
    let mut untrimmed_arena = arena_a.clone();
    for t in 0..4 {
        write_f32(
            &mut untrimmed_arena,
            ARENA0_TIER_GAIN_TRIM_OFFSET + t * 4,
            1.0,
        );
    }
    let dnam_untrimmed = build_dnam_container_with_cp1(
        &mut hdr.clone(),
        &untrimmed_arena,
        &untrimmed_arena,
        false,
        &[slot],
    )
    .unwrap();

    for (t, (&tier, &tr)) in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra]
        .iter()
        .zip(&trims)
        .enumerate()
    {
        let mut e_trim =
            DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_a, tier, 48_000).unwrap();
        let mut e_raw =
            DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_untrimmed, tier, 48_000)
                .unwrap();
        let yt = e_trim.step_frame_tora(0.25, 0.25, false);
        let yr = e_raw.step_frame_tora(0.25, 0.25, false);
        assert!(
            (yt[0] - yr[0] * tr).abs() <= 1.0e-6,
            "tier {t} 1x trim mismatch: got {} vs expected {}",
            yt[0],
            yr[0] * tr
        );

        // 4x offline bounce + 4CM 4-port parity
        e_trim.reset_state();
        e_raw.reset_state();
        e_trim.set_offline_bounce(true);
        e_raw.set_offline_bounce(true);
        let (mut ml_t, mut mr_t, mut sl_t, mut sr_t) = ([0.0_f32; 4], [0.0; 4], [0.0; 4], [0.0; 4]);
        let (mut ml_r, mut mr_r, mut sl_r, mut sr_r) = ([0.0_f32; 4], [0.0; 4], [0.0; 4], [0.0; 4]);
        let inp = [0.25_f32; 4];
        let zero = [0.0_f32; 4];
        e_trim.process_block_4cm(
            &inp, &inp, &zero, &zero, &mut ml_t, &mut mr_t, &mut sl_t, &mut sr_t,
        );
        e_raw.process_block_4cm(
            &inp, &inp, &zero, &zero, &mut ml_r, &mut mr_r, &mut sl_r, &mut sr_r,
        );
        assert!(
            (ml_t[3] - ml_r[3] * tr).abs() <= 1.0e-5,
            "tier {t} 4x 4CM trim mismatch: got {} vs expected {}",
            ml_t[3],
            ml_r[3] * tr
        );
    }

    let s_pro = strip_container_to_tier(&dnam_a, Tier::Pro).unwrap();
    let s_full = strip_container_to_tier(&s_pro, Tier::Full).unwrap();
    let s_lite = strip_container_to_tier(&s_full, Tier::Lite).unwrap();
    let mut e_direct_lite =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_a, Tier::Lite, 48_000).unwrap();
    let mut e_strip_lite =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&s_lite, Tier::Lite, 48_000).unwrap();
    for i in 0..64 {
        let x = 0.2 * ((i as f32) * 0.13).sin();
        let a = e_direct_lite.step_frame_tora(x, x, false);
        let b = e_strip_lite.step_frame_tora(x, x, false);
        assert_eq!(a[0].to_bits(), b[0].to_bits());
        assert_eq!(a[1].to_bits(), b[1].to_bits());
    }
}

#[test]
fn test_vault_path1_stride_autodetect_w_ctrl_preservation_and_with_options_apis() {
    // verify[train.cache.proj-vault]
    // verify[train.resume.kronecker-reassembly]
    let crypto = DefaultCryptoProvider::default();
    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.controls_schema = vec![make_range_ctrl("gain", 0.0, 10.0, 5.0)];

    let mut seed_arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    let dry: Vec<f32> = (0..CORESET_FP16_FRAMES_PER_STEM)
        .map(|i| 0.35 * ((i as f32) * 0.071).sin() + 0.2 * ((i as f32) * 0.19).cos())
        .collect();
    let dry_lr: Vec<[f32; 2]> = dry.iter().map(|&x| [x, x]).collect();
    let wet_star: Vec<f32> = dry.iter().map(|&x| (1.5 * x).tanh()).collect();
    let wet_lr: Vec<[f32; 2]> = wet_star.iter().map(|&y| [y, y]).collect();
    let _ = lift_stage_a_backbone(
        &manifest,
        &mut seed_arena,
        &dry_lr,
        &wet_lr,
        None,
        Default::default(),
    );
    write_f32(&mut seed_arena, 9_536 + 8, 0.42);
    write_f32(&mut seed_arena, 60_672 + 12, -0.31);

    let seed_proj =
        build_dnam_proj_container_with_cp1(&mut manifest.clone(), &[], &seed_arena, &[]).unwrap();
    let inpc = build_proj_inpc_capsule(1, &[0x21; 256], &[0x32; 320], &fp16_bytes(&dry)).unwrap();

    let star_spec = make_spec(
        "star_5",
        true,
        BTreeMap::from([("gain".to_string(), serde_json::json!(5.0))]),
        &wet_star,
        vec![StmcTrajectoryWindow {
            control_id: 0,
            ctrl_kind: 0,
            active_opt: 0xFF,
            u_phys: 5.0,
            window_weight: 1.0,
            emr_velocity: 0.0,
        }],
    );
    let proj_p2 = add_capture_with_options(
        &seed_proj,
        star_spec,
        Some(&inpc),
        SolverOptions::preserve_backbone(),
    )
    .unwrap();

    let wet_hi: Vec<f32> = dry.iter().map(|&x| (3.1 * x).tanh()).collect();
    let off_spec = make_spec(
        "off_10",
        false,
        BTreeMap::from([("gain".to_string(), serde_json::json!(10.0))]),
        &wet_hi,
        vec![StmcTrajectoryWindow {
            control_id: 0,
            ctrl_kind: 0,
            active_opt: 0xFF,
            u_phys: 10.0,
            window_weight: 1.0,
            emr_velocity: 0.0,
        }],
    );
    let proj_p1 = add_capture(&proj_p2, off_spec.clone(), None).unwrap();

    let v_p2 = DnamProjView::from_bytes(&proj_p2, &crypto).unwrap();
    let v_p1 = DnamProjView::from_bytes(&proj_p1, &crypto).unwrap();
    let a_p2 =
        DnamContainerView::from_bytes(v_p2.chunk_slice(ProjChunkKind::Wght, 0).unwrap(), &crypto)
            .unwrap()
            .arena_storage
            .as_bytes()
            .to_vec();
    let a_p1 =
        DnamContainerView::from_bytes(v_p1.chunk_slice(ProjChunkKind::Wght, 0).unwrap(), &crypto)
            .unwrap()
            .arena_storage
            .as_bytes()
            .to_vec();

    assert_eq!(read_f32(&a_p1, 9_536 + 8), 0.42);
    assert_eq!(read_f32(&a_p1, 60_672 + 12), -0.31);
    for &hb in &HEAD_BASES {
        assert_eq!(&a_p1[hb..hb + 512], &a_p2[hb..hb + 512]);
    }

    let proj_dis = set_capture_enabled_with_options(
        &proj_p1,
        "off_10",
        false,
        SolverOptions::preserve_backbone(),
    )
    .unwrap();
    let proj_rep = replace_capture_with_options(
        &proj_dis,
        "off_10",
        off_spec,
        SolverOptions::preserve_backbone(),
    )
    .unwrap();
    let proj_rem =
        remove_capture_with_options(&proj_rep, "off_10", SolverOptions::preserve_backbone())
            .unwrap();
    let v_rem = DnamProjView::from_bytes(&proj_rem, &crypto).unwrap();
    assert_eq!(v_rem.manifest.stems_registry.len(), 1);
}

#[test]
fn test_stage_d_p68_region4_cp1_and_collinear_controls_robustness() {
    // verify[train.pipeline.stage-d-control]
    let crypto = DefaultCryptoProvider::default();
    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.controls_schema = (0..68)
        .map(|p| make_range_ctrl(&format!("k_{p:02}"), 0.0, 10.0, 5.0))
        .collect();
    let empty_proj = build_dnam_proj_container(&manifest, &[]).unwrap();

    let dry: Vec<f32> = (0..CORESET_FP16_FRAMES_PER_STEM)
        .map(|i| 0.30 * ((i as f32) * 0.063).sin())
        .collect();
    let inpc = build_proj_inpc_capsule(1, &[0x41; 256], &[0x42; 320], &fp16_bytes(&dry)).unwrap();

    let mut star_meta = BTreeMap::new();
    for p in 0..68 {
        star_meta.insert(format!("k_{p:02}"), serde_json::json!(5.0));
    }
    let star_win = vec![StmcTrajectoryWindow {
        control_id: 0,
        ctrl_kind: 0,
        active_opt: 0xFF,
        u_phys: 5.0,
        window_weight: 1.0,
        emr_velocity: 0.0,
    }];
    let wet_star: Vec<f32> = dry.iter().map(|&x| (1.3 * x).tanh()).collect();
    let proj_star = add_capture(
        &empty_proj,
        make_spec("star", true, star_meta.clone(), &wet_star, star_win),
        Some(&inpc),
    )
    .unwrap();

    let mut collinear_meta = star_meta;
    collinear_meta.insert("k_00".to_string(), serde_json::json!(9.5));
    collinear_meta.insert("k_01".to_string(), serde_json::json!(9.5));
    collinear_meta.insert("k_67".to_string(), serde_json::json!(9.5));
    let off_win = vec![StmcTrajectoryWindow {
        control_id: 67,
        ctrl_kind: 0,
        active_opt: 0xFF,
        u_phys: 9.5,
        window_weight: 1.0,
        emr_velocity: 0.0,
    }];
    let wet_off: Vec<f32> = dry.iter().map(|&x| (2.8 * x).tanh()).collect();
    let proj_p68 = add_capture(
        &proj_star,
        make_spec("off_collinear_67", false, collinear_meta, &wet_off, off_win),
        None,
    )
    .unwrap();

    let dnam_p68 = export_runtime_dnam(&proj_p68, Some(Tier::Ultra)).unwrap();
    let rt_view = DnamContainerView::from_bytes(&dnam_p68, &crypto).unwrap();
    assert!(
        rt_view.cp1_contributors.len() >= 3,
        "P = 68 controls must pack into Region-4 cp1_contributors stream"
    );

    let mut eng =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_p68, Tier::Ultra, 48_000).unwrap();
    eng.set_control_immediate(0, 9.5);
    eng.set_control_immediate(1, 9.5);
    eng.set_control_immediate(67, 9.5);
    let y = eng.step_frame_tora(0.3, 0.3, false);
    assert!(y[0].is_finite() && y[1].is_finite());
}

#[test]
fn test_3phase_audit_ultra_modes_5_to_8_stage1_stage4_and_head_only_mode() {
    // verify[train.audit.phase1.clamp-stability]
    // verify[train.audit.phase2.clamp-rail]
    // verify[train.audit.phase3.clamp-head-gain]
    let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    write_f32(&mut arena, 256, 1.05);
    write_f32(&mut arena, 10_752, 0.99);
    write_f32(&mut arena, 10_768, 0.25);
    write_f32(&mut arena, 4_700, -0.95);
    write_f32(&mut arena, 17_792, 1.5);
    write_f32(&mut arena, 34_496, 2.0);
    write_f32(&mut arena, 48_896, 5_000.0);
    write_f32(&mut arena, 4_112, 25.0);
    let ultra_m7_off = ULTRA_MODES_5_TO_8_BASE + 2 * 2_048 + 2 * 512;
    write_f32(&mut arena, ultra_m7_off, 40.0);

    let mut m_head = DnamProjManifest::default_reference();
    let mut head_only_arena = arena.clone();
    run_head_only_3phase_arena_audit(&mut m_head, &mut head_only_arena);
    assert!(read_f32(&head_only_arena, ultra_m7_off) < 10.0);
    assert_eq!(read_f32(&head_only_arena, 256), 1.05);
    assert_eq!(read_f32(&head_only_arena, 4_700), -0.95);

    let mut m_full = DnamProjManifest::default_reference();
    run_full_3phase_arena_audit(&mut m_full, &mut arena);
    assert!(read_f32(&arena, 256) <= 0.9995);
    assert!(read_f32(&arena, 10_752) + read_f32(&arena, 10_768) <= 0.9995 + 1.0e-6);
    assert!(read_f32(&arena, 4_700).abs() <= 0.75);
    assert!(read_f32(&arena, 17_792) >= 4.0);
    assert!(read_f32(&arena, 34_496) >= 4.0);
    assert!(read_f32(&arena, 48_896) <= 2_048.0);
    assert!(read_f32(&arena, 4_112) < 25.0);
    assert!(read_f32(&arena, ultra_m7_off) < 10.0);
}

#[derive(Clone)]
struct SimpleTanhTarget {
    gain: f32,
}

impl DistillationTarget for SimpleTanhTarget {
    fn reset_state(&mut self) {}

    fn render_stem(&mut self, stim: &StemStimulus) -> Result<StemResponse, DnamError> {
        let g = stim.control_values.first().copied().unwrap_or(5.0) * 0.35 + self.gain;
        let out_1: Vec<f32> = stim.in_1.iter().map(|&x| (g * x).tanh()).collect();
        Ok(StemResponse {
            stem_id: stim.stem_id.clone(),
            sample_rate_hz: stim.sample_rate_hz,
            out_1,
            out_2: None,
            out_3: None,
            out_4: None,
            internal_nodes: BTreeMap::new(),
        })
    }
}

#[test]
fn test_live_null_test_autopatch_with_role_b_gauge_and_noisy_blackbox_phase_id() {
    // verify[train.pipeline.stage-e-pack]
    let crypto = DefaultCryptoProvider::default();
    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.controls_schema = vec![make_range_ctrl("drive", 0.0, 10.0, 6.0)];
    let mut target = SimpleTanhTarget { gain: 0.4 };
    let opts = AdaptiveTrainingOptions {
        sample_rate_hz: 48_000,
        topology_mode: TopologyProbeMode::Mono1To1,
        fixed_duration: true,
        prefer_mode_b: true,
        max_frames_per_stem: 2,
        run_live_null_test: false,
        null_test_autopatch_threshold_db: -42.0,
        calb: None,
    };
    let trained = train_adaptive_target(manifest, &mut target, &opts).unwrap();

    let view = DnamProjView::from_bytes(&trained.proj_bytes, &crypto).unwrap();
    let wght = view.chunk_slice(ProjChunkKind::Wght, 0).unwrap();
    let rt_view = DnamContainerView::from_bytes(wght, &crypto).unwrap();
    let mut gauge_arena = rt_view.arena_storage.as_bytes().to_vec();
    apply_cascade_head_gauge(&mut gauge_arena, 0, &[[1.35_f32; 4]; 32]).unwrap();
    for i in 0..16 {
        let cur = read_f32(&gauge_arena, HEAD_BASES[0] + i * 4);
        write_f32(&mut gauge_arena, HEAD_BASES[0] + i * 4, cur * 0.55);
    }
    let mut chunks = Vec::new();
    for (entry, &slice) in view.directory.iter().zip(&view.chunk_slices) {
        let kind = entry.kind().unwrap();
        if kind != ProjChunkKind::Wght && kind != ProjChunkKind::Auth {
            chunks.push(ProjChunkInput {
                kind,
                chunk_id: entry.chunk_id,
                payload: slice.to_vec(),
            });
        }
    }
    let mut patched_manifest = view.manifest.clone();
    let gauged_proj =
        build_dnam_proj_container_with_cp1(&mut patched_manifest, &chunks, &gauge_arena, &[])
            .unwrap();

    let (_, _, null_rep) =
        run_live_null_test_and_autopatch(&gauged_proj, &mut target, -60.0).unwrap();
    assert!(null_rep.reload_in_place_verified);
    assert!(
        null_rep.post_patch_null_esr_db <= null_rep.pre_patch_null_esr_db,
        "Role-B gauge auto-patch must not degrade ESR: pre={} post={}",
        null_rep.pre_patch_null_esr_db,
        null_rep.post_patch_null_esr_db
    );

    // Noisy (-45 dBFS) black-box audio-only pole-angle phase watermark identification
    let wm_cfg = BuyerWatermarkConfig::new([0x77; 32], "tanh_v2", "lic_buyer_03", [0x33; 16]);
    let inn_cfg = BuyerWatermarkConfig::new([0x77; 32], "tanh_v2", "lic_buyer_99", [0x99; 16]);
    let wm_dnam =
        export_runtime_dnam_watermarked(&trained.proj_bytes, &wm_cfg, Some(Tier::Ultra)).unwrap();
    let mut wm_audio = render_blackbox_probe_response(&wm_dnam, 8_192).unwrap();
    for (i, y) in wm_audio.iter_mut().enumerate() {
        y[0] += 0.0025_f32 * ((i as f32 * 0.713).sin() + 0.5 * (i as f32 * 1.917).cos());
        y[1] += 0.0025_f32 * ((i as f32 * 0.519).cos());
    }
    let (master_angles, est_wm_angles) =
        identify_pole_angles_blackbox_from_audio(&trained.runtime_dnam_bytes, &wm_audio).unwrap();
    let z_target = score_phase_coding_blackbox(
        &master_angles,
        &est_wm_angles,
        &derive_buyer_phase_seed(&wm_cfg),
    )
    .unwrap();
    let z_innocent = score_phase_coding_blackbox(
        &master_angles,
        &est_wm_angles,
        &derive_buyer_phase_seed(&inn_cfg),
    )
    .unwrap();
    assert!(
        z_target >= 4.0,
        "noisy (-45 dBFS) black-box phase Z-score must be >= 4.0, got {z_target}"
    );
    assert!(
        z_innocent.abs() < 3.5,
        "innocent buyer noisy black-box phase |Z| must be < 3.5, got {z_innocent}"
    );
}

#[test]
fn test_tier4_whitebox_faust_and_spice_runners_end_to_end_varpro_distillation() {
    // verify[train.pipeline.stage-a-poles]
    // verify[train.pipeline.fixtures-matrix]
    let faust_json = r#"{"ui":[{"type":"vgroup","label":"Preamp","items":[{"type":"hslider","label":"Drive","init":0.5,"min":0.0,"max":1.0,"step":0.01},{"type":"hslider","label":"Tone","init":0.5,"min":0.0,"max":1.0,"step":0.01}]}]}"#;
    let parsed_specs = parse_faust_json_controls(faust_json).unwrap();
    let mut faust = FaustRunner::from_faust_json("preamp.dsp", faust_json, 5_000).unwrap();
    let mut m_faust = DnamProjManifest::default_reference();
    m_faust.base.controls_schema = parsed_specs
        .iter()
        .map(|s| make_range_ctrl(&s.label.to_lowercase(), s.min, s.max, s.init))
        .collect();

    let opts_mono = AdaptiveTrainingOptions {
        sample_rate_hz: 48_000,
        topology_mode: TopologyProbeMode::Mono1To1,
        fixed_duration: true,
        prefer_mode_b: true,
        max_frames_per_stem: 2,
        run_live_null_test: false,
        null_test_autopatch_threshold_db: -42.0,
        calb: None,
    };
    let rep_faust = train_adaptive_target(m_faust, &mut faust, &opts_mono).unwrap();

    let mut spice = SpiceRunner::new("triode_stage.cir", 5_000);
    let mut m_spice = DnamProjManifest::default_reference();
    m_spice.base.io_config.inputs = 2;
    m_spice.base.io_config.output_mode = "stereo".to_string();
    m_spice.base.controls_schema = vec![
        make_range_ctrl("drive", 0.0, 1.0, 0.5),
        make_range_ctrl("bias", 0.0, 1.0, 0.5),
    ];
    let opts_stereo = AdaptiveTrainingOptions {
        topology_mode: TopologyProbeMode::TrueStereo2To2,
        ..opts_mono
    };
    let rep_spice = train_adaptive_target(m_spice, &mut spice, &opts_stereo).unwrap();

    for (label, proj_bytes) in [
        ("faust", &rep_faust.proj_bytes),
        ("spice", &rep_spice.proj_bytes),
    ] {
        let mut lufs = [0.0_f32; 4];
        for (t_idx, tier) in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra]
            .into_iter()
            .enumerate()
        {
            let dnam_b = export_runtime_dnam(proj_bytes, Some(tier)).unwrap();
            let mut eng =
                DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_b, tier, 48_000).unwrap();
            let [s0, s1] = eng.step_frame_tora(0.0, 0.0, false);
            assert!(
                s0.abs() <= 1.0e-7 && s1.abs() <= 1.0e-7,
                "{label} [STAB-ZERO-DC] failed"
            );
            eng.reset_to_dc_equilibrium();
            let mut out = Vec::with_capacity(512);
            for i in 0..640 {
                let x = 0.3 * ((i as f32) * 0.085).sin() + 0.15 * ((i as f32) * 0.23).cos();
                let y = eng.step_frame_tora(x, x, false);
                assert!(y[0].is_finite() && y[1].is_finite());
                if i >= 128 {
                    out.push(y);
                }
            }
            lufs[t_idx] = compute_integrated_lufs(&out);
        }
        assert!(
            lufs[3] > -60.0,
            "{label} white-box model output unexpectedly silent: {lufs:?}"
        );
    }
}
