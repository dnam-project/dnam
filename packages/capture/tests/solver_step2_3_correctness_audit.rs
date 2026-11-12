#![forbid(unsafe_code)]
//! Step 2.3 Correctness & Isomorphism Audit Regressions (`Wide192Accum` narrow-span/mantissa
//! cancellation, subnormal/extreme-exponent `two_prod_dekker`, Boldo-Melquiond `round-to-odd`,
//! `ControlClampDescriptor` arena synchronization for `u_norm != 0` `StarCenter`s, Path-2 `W_ctrl`
//! & `CP1`/`U_int` clearing order, and `Mode 0` shortcut/fallback isomorphism).

use std::collections::BTreeMap;

use dnam_capture::proj::{
    DnamProjManifest, StemProvenance, StmcTrajectoryWindow, VaultCaptureSpec,
    add_capture_with_options, build_dnam_proj_container_with_cp1, build_proj_inpc_capsule,
    export_runtime_dnam, replace_capture_with_options,
};
use dnam_capture::solver::stage_a::StageAConfig;
use dnam_capture::solver::{
    SolverOptions, Wide192Accum, accumulate_uplo_u_wide192, compute_integrated_lufs,
    contract_256d_gram_tiled_wide192, harvest_star_center_256d, lift_stage_a_backbone,
    solve_stage_d_cp1_and_audit_with_harvest, uplo_u_index, uplo_u_packed_len,
};
use dnam_capture::{CORESET_FP16_FRAMES_PER_STEM, f32_to_fp16_bits};
use dnam_engine::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container_with_cp1};
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::schema::ControlSchemaDef;
use dnam_engine::{DnamEngine, Tier};

fn write_f32(buf: &mut [u8], off: usize, val: f32) {
    buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
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
fn test_wide192_c_and_g_narrow_span_and_f32_mantissa_intra_batch_cancellation() {
    // verify[train.gram.wide192]
    // verify[train.gram.accuracy-contract]
    let n = 64usize;
    let mut feats = vec![[0.0_f32; 256]; n];
    let mut targets = vec![[0.0_f64; 2]; n];
    let weights = vec![1.0_f64; n];

    // 1. Cross-correlation wide_c at <= 65,536x span (2^16):
    // +1.0 + 2^-16 * (2^-16 + 2^-40) - 1.0 - 2^-32 == 2^-56
    let s16 = 2.0_f32.powi(-16);
    feats[0][2] = 1.0;
    targets[0][0] = 1.0;
    feats[1][2] = s16;
    targets[1][0] = 2.0_f64.powi(-16) + 2.0_f64.powi(-40);
    feats[2][2] = -1.0;
    targets[2][0] = 1.0;
    feats[3][2] = -s16;
    targets[3][0] = 2.0_f64.powi(-16);

    // 2. Cross-correlation wide_c at 0-bit exponent span (all magnitudes in [1.0, 2.0)):
    // 1*1 + (1 + 2^-23)*(1 + 2^-40) - 1*(1 + 2^-40) - (1 + 2^-23)*1 == 2^-63
    let u23 = 1.0_f32 + 2.0_f32.powi(-23);
    let t40 = 1.0_f64 + 2.0_f64.powi(-40);
    feats[8][5] = 1.0;
    targets[8][1] = 1.0;
    feats[9][5] = u23;
    targets[9][1] = t40;
    feats[10][5] = -1.0;
    targets[10][1] = t40;
    feats[11][5] = -u23;
    targets[11][1] = 1.0;

    // 3. Off-diagonal Gram wide_g at span = 32 (<< 65,536) with 24-bit f32 significands:
    // 1*1 + (2^-5*(1+2^-23))^2 - 1*1 - 2^-5*(2^-5*(1+2^-22)) == 2^-56
    let a_small = 2.0_f32.powi(-5) * (1.0_f32 + 2.0_f32.powi(-23));
    let b_small = 2.0_f32.powi(-5) * (1.0_f32 + 2.0_f32.powi(-22));
    feats[16][0] = 1.0;
    feats[16][9] = 1.0;
    feats[17][0] = a_small;
    feats[17][9] = a_small;
    feats[18][0] = -1.0;
    feats[18][9] = 1.0;
    feats[19][0] = -2.0_f32.powi(-5);
    feats[19][9] = b_small;

    // Same 24-bit f32 cancellation on diagonal block (col 12 vs col 13 inside same 4-lane block)
    feats[24][12] = 1.0;
    feats[24][13] = 1.0;
    feats[25][12] = a_small;
    feats[25][13] = a_small;
    feats[26][12] = -1.0;
    feats[26][13] = 1.0;
    feats[27][12] = -2.0_f32.powi(-5);
    feats[27][13] = b_small;

    let (g, c) = contract_256d_gram_tiled_wide192(&feats, &targets, &weights, 1);
    assert_eq!(c[2][0].to_bits(), (2.0_f64.powi(-56)).to_bits());
    assert_eq!(c[5][1].to_bits(), (2.0_f64.powi(-63)).to_bits());
    assert_eq!(
        g[uplo_u_index(0, 9)].to_bits(),
        (2.0_f64.powi(-56)).to_bits()
    );
    assert_eq!(
        g[uplo_u_index(12, 13)].to_bits(),
        (2.0_f64.powi(-56)).to_bits()
    );
}

#[test]
fn test_wide192_subnormal_extreme_exponent_and_half_ulp_tie_round_to_odd() {
    // verify[train.gram.wide192]
    // 1. Extreme exponent Dekker split (> 1.34e299 where 134_217_729.0 * a would overflow)
    let mut acc_hi = Wide192Accum::zero();
    let a_big = 2.0_f64.powi(996) * (1.0 + 2.0_f64.powi(-30));
    let b_tiny = 2.0_f64.powi(-996) * (1.0 - 2.0_f64.powi(-30));
    acc_hi.add_product(a_big, b_tiny);
    acc_hi.add_f64(-1.0);
    assert_eq!(acc_hi.to_f64().to_bits(), (-2.0_f64.powi(-60)).to_bits());

    // 2. Deep subnormal product + cancellation: +1.0 + (2^-1040 * 2^-20) - 1.0 == 2^-1060
    let mut acc_sub = Wide192Accum::zero();
    acc_sub.add_f64(1.0);
    acc_sub.add_product(2.0_f64.powi(-1040), 2.0_f64.powi(-20));
    acc_sub.add_f64(-1.0);
    assert_eq!(acc_sub.to_f64().to_bits(), (2.0_f64.powi(-1060)).to_bits());

    // 3. Boldo-Melquiond round-to-odd prevents IEEE-754 double-rounding at half-ULP ties
    let mut tie_up = Wide192Accum::zero();
    tie_up.add_f64(1.0);
    tie_up.add_f64(2.0_f64.powi(-53));
    tie_up.add_f64(2.0_f64.powi(-108));
    assert_eq!(
        tie_up.to_f64().to_bits(),
        (1.0 + 2.0_f64.powi(-52)).to_bits()
    );

    let mut tie_down = Wide192Accum::zero();
    tie_down.add_f64(1.0 + 2.0_f64.powi(-52));
    tie_down.add_f64(2.0_f64.powi(-53));
    tie_down.add_f64(-2.0_f64.powi(-108));
    assert_eq!(
        tie_down.to_f64().to_bits(),
        (1.0 + 2.0_f64.powi(-52)).to_bits()
    );

    // 4. Multi-scale triple w * fi * fj without intermediate w * fj underflow/overflow
    let mut fb = [0.0_f32; 256];
    fb[0] = 1.0e35;
    fb[3] = 1.0e-35;
    fb[1] = 1.0e-35;
    fb[4] = 1.0e35;

    let fb64: Vec<f64> = fb[..8].iter().map(|&v| f64::from(v)).collect();
    let mut acc_vec = vec![Wide192Accum::zero(); uplo_u_packed_len(8)];
    let mut acc_c = vec![[Wide192Accum::zero(); 2]; 8];
    accumulate_uplo_u_wide192(
        &mut acc_vec,
        &mut acc_c,
        &fb64,
        [f64::from(1.0e35_f32), 0.0],
        1.0e-300,
    );
    let expected_lo = 1.0e-300_f64 * (f64::from(1.0e35_f32) * f64::from(1.0e-35_f32));
    assert!((acc_vec[uplo_u_index(0, 3)].to_f64() - expected_lo).abs() <= 1.0e-314);
    assert!((acc_c[3][0].to_f64() - expected_lo).abs() <= 1.0e-314);

    let feats = vec![fb; 2];
    let targets = vec![[0.0_f64; 2]; 2];
    let weights = vec![1.0e-300_f64, 1.0e300_f64];
    let (g, _) = contract_256d_gram_tiled_wide192(&feats, &targets, &weights, 1);
    let expected_hi = 1.0e300_f64 * (f64::from(1.0e-35_f32) * f64::from(1.0e35_f32));
    assert!(
        g[uplo_u_index(1, 4)].is_finite() && (g[uplo_u_index(1, 4)] - expected_hi).abs() <= 1.0e285
    );

    let feats_lo_only = vec![fb];
    let (g_lo, _) = contract_256d_gram_tiled_wide192(&feats_lo_only, &[[0.0; 2]], &[1.0e-300], 1);
    assert!((g_lo[uplo_u_index(0, 3)] - expected_lo).abs() <= 1.0e-314);
}

#[test]
fn test_non_centered_star_control_and_dynamic_w_ctrl_isomorphism_and_repool() {
    // verify[train.pipeline.isomorphism-parity]
    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.io_config.inputs = 1;
    manifest.base.io_config.output_mode = "mono".to_string();
    manifest.base.controls_schema = vec![make_range_ctrl("drive", 0.0, 10.0, 8.0)];

    let n = CORESET_FP16_FRAMES_PER_STEM;
    let dry_mono: Vec<f32> = (0..n)
        .map(|i| 0.30 * ((i as f32) * 0.071).sin() + 0.18 * ((i as f32) * 0.233).cos())
        .collect();
    let dry_lr_seed: Vec<[f32; 2]> = dry_mono.iter().map(|&x| [x, x]).collect();

    // Populate a valid Stage A physical backbone so `has_populated_physical_backbone` is true!
    let mut gt_arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    let _ = lift_stage_a_backbone(
        &manifest,
        &mut gt_arena,
        &dry_lr_seed,
        &dry_lr_seed,
        None,
        StageAConfig::default(),
    );
    // Non-zero W_ctrl row 4 (Stage II drive) driven by z_head[0] (must be preserved when clear_dynamic_w_ctrl = false,
    // and cleared BEFORE Stage B when clear_dynamic_w_ctrl = true!)
    write_f32(&mut gt_arena, 9_536 + 4 * 64, 1.25);
    for (t, &hb) in [4_928usize, 18_048, 31_936, 57_600].iter().enumerate() {
        let g = 0.28 - 0.04 * (t as f32);
        for d in 0..16 {
            let s = if d % 2 == 0 { g } else { -0.5 * g };
            write_f32(&mut gt_arena, hb + d * 4, s);
            write_f32(&mut gt_arena, hb + 256 + d * 4, s);
        }
    }

    let mut gt_hdr = manifest.base.clone();
    gt_hdr.controls_schema[0].safe_clamped_range = Some([6.0, 8.0]);
    let gt_dnam = build_dnam_container_with_cp1(&mut gt_hdr, &gt_arena, &gt_arena, false, &[])
        .expect("valid gt dnam");
    let mut gt_eng =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&gt_dnam, Tier::Ultra, 48_000).unwrap();

    let render_at = |eng: &mut DnamEngine<'_, DefaultMathBackend>, ctrl_val: f32| -> Vec<f32> {
        eng.reset_to_dc_equilibrium();
        eng.set_control_immediate(0, ctrl_val);
        dry_mono
            .iter()
            .map(|&x| eng.step_frame_tora(x, x, true)[0])
            .collect()
    };
    let wet_star = render_at(&mut gt_eng, 8.0);
    let wet_lo = render_at(&mut gt_eng, 6.0);

    let mut seed_arena = gt_arena.clone();
    for &hb in &[4_928usize, 18_048, 31_936, 57_600] {
        seed_arena[hb..hb + 512].fill(0);
    }
    // Note: seed_proj starts with safe_clamped_range = [0.0, 10.0] (u_norm(8.0) = +0.6), whereas
    // Phase-1 [CLAMP-RANGE] clamps [6.0, 8.0] (u_norm(8.0) = +1.0); vault + Stage B/D/E must sync!
    let seed_proj =
        build_dnam_proj_container_with_cp1(&mut manifest.clone(), &[], &seed_arena, &[]).unwrap();
    let inpc =
        build_proj_inpc_capsule(1, &[0x21; 256], &[0x32; 320], &fp16_bytes(&dry_mono)).unwrap();

    let make_win = |u_phys: f32| {
        vec![StmcTrajectoryWindow {
            control_id: 0,
            ctrl_kind: 0,
            active_opt: 0xFF,
            u_phys,
            window_weight: 1.0,
            emr_velocity: 0.0,
        }]
    };
    let star_meta = BTreeMap::from([("drive".to_string(), serde_json::json!(8.0))]);
    let lo_meta = BTreeMap::from([("drive".to_string(), serde_json::json!(6.0))]);

    let opts_preserve = SolverOptions {
        stage_a: StageAConfig {
            preserve_existing_backbone: true,
            vp_gn_steps: 1,
        },
        gram_frame_stride: 1,
        clear_dynamic_w_ctrl: false,
    };
    let proj1 = add_capture_with_options(
        &seed_proj,
        make_spec("star", true, star_meta.clone(), &wet_star, make_win(8.0)),
        Some(&inpc),
        opts_preserve,
    )
    .unwrap();
    let proj2 = add_capture_with_options(
        &proj1,
        make_spec("lo", false, lo_meta, &wet_lo, make_win(6.0)),
        None,
        opts_preserve,
    )
    .unwrap();
    // Force Path 2 re-pool under the newly clamped [6.0, 8.0] range so Stage B harvests u_norm = +1.0
    let proj2 = replace_capture_with_options(
        &proj2,
        "star",
        make_spec("star", true, star_meta.clone(), &wet_star, make_win(8.0)),
        opts_preserve,
    )
    .unwrap();

    let exported = export_runtime_dnam(&proj2, Some(Tier::Ultra)).unwrap();
    let mut dist_eng =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&exported, Tier::Ultra, 48_000).unwrap();
    let pred_star = render_at(&mut dist_eng, 8.0);
    let err_e: f64 = pred_star[256..]
        .iter()
        .zip(&wet_star[256..])
        .map(|(&a, &b)| {
            let d = f64::from(a) - f64::from(b);
            d * d
        })
        .sum();
    let sig_e: f64 = wet_star[256..]
        .iter()
        .map(|&b| f64::from(b) * f64::from(b))
        .sum();
    let esr_db = 10.0 * (err_e / sig_e).log10();
    assert!(
        esr_db < -50.0,
        "non-centered StarCenter (u_norm = +0.6) isomorphism ESR must be < -50 dB, got {esr_db} dB"
    );

    // Replace StarCenter with clear_dynamic_w_ctrl = true: stale W_ctrl/CP1 must be cleared BEFORE Stage B!
    let wet_scaled: Vec<f32> = dry_mono.iter().map(|&x| 0.42 * x).collect();
    let proj3 = replace_capture_with_options(
        &proj2,
        "star",
        make_spec("star", true, star_meta, &wet_scaled, make_win(8.0)),
        SolverOptions::default(),
    )
    .unwrap();
    let mut lufs = [0.0_f32; 4];
    for (t_idx, tier) in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra]
        .into_iter()
        .enumerate()
    {
        let b = export_runtime_dnam(&proj3, Some(tier)).unwrap();
        let mut e = DnamEngine::<DefaultMathBackend>::from_bytes_owned(&b, tier, 48_000).unwrap();
        e.reset_to_dc_equilibrium();
        e.set_control_immediate(0, 8.0);
        for &x in &dry_mono[..128] {
            let _ = e.step_frame_tora(x, x, true);
        }
        let out: Vec<[f32; 2]> = dry_mono[128..640]
            .iter()
            .map(|&x| e.step_frame_tora(x, x, true))
            .collect();
        lufs[t_idx] = compute_integrated_lufs(&out);
    }
    assert!(lufs[3] > -55.0);
    for &lu in &lufs {
        assert!((lu - lufs[3]).abs() <= 0.05, "loudness mismatch: {lufs:?}");
    }
}

#[test]
fn test_mode0_shortcut_wdw_binary_flag_and_b_head_guards() {
    // verify[train.pipeline.isomorphism-parity]
    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.controls_schema = vec![make_range_ctrl("tone", 0.0, 10.0, 7.5)];
    let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    write_f32(&mut arena, 0, 1.0);
    write_f32(&mut arena, 20, 1.0);
    write_f32(&mut arena, 32, 1.0);
    write_f32(&mut arena, 52, 1.0);
    write_f32(&mut arena, 120, 1.0);
    write_f32(&mut arena, 124, 0.9993);
    write_f32(&mut arena, 128, 200.0);
    write_f32(&mut arena, 4_288, 0.35);
    write_f32(&mut arena, 4_928, 0.25);
    // Inject non-zero b_head and binary WDW Stage-0 flag (arena[112] = 1) while output_ports == 4
    write_f32(&mut arena, 9_024, 0.125);
    arena[112] = 1;

    let dry_lr: Vec<[f32; 2]> = (0..512)
        .map(|i| {
            let v = 0.2 * ((i as f32) * 0.11).sin();
            [v, -0.5 * v]
        })
        .collect();
    let harvest = harvest_star_center_256d(&manifest, &arena, &dry_lr, &dry_lr, &[7.5], 1).unwrap();
    let stmc_map = BTreeMap::new();
    let out = solve_stage_d_cp1_and_audit_with_harvest(
        &mut manifest,
        &mut arena,
        &stmc_map,
        &dry_lr,
        Some(&harvest.packed_gram_256),
        1,
        false,
        Some((&harvest.features, &harvest.backbone_out_lr)),
        false,
    )
    .unwrap();
    assert_eq!(out.active_head_modes, 0);
    // Confirm b_head was hard-zeroed ([STAB-ZERO-DC]) and descriptors were synced into [8_512..9_024]
    assert_eq!(&arena[9_024..9_028], &[0, 0, 0, 0]);
    assert!(arena[8_512..8_544].iter().any(|&b| b != 0));
}

#[test]
fn test_strided_bezout_gram_inner_product_consistency_and_48k_window_lifting() {
    // verify[train.pipeline.stage-c-schur]
    // verify[train.pipeline.stage-a-poles]
    use dnam_capture::fp16_bits_to_f32;
    use dnam_capture::solver::bezout_gram::eval_4tier_post_trim_besr_lin;
    use dnam_capture::solver::{
        compute_role_b_multipliers, finalize_stage_e_and_telemetry,
        solve_stage_c_nested_schur_4tier, write_effective_mode0_weights,
    };
    use dnam_capture::stimulus::waveforms::{
        deterministic_seed_256, synthesize_a5_musical_transient_coreset,
    };
    use dnam_fixtures::{BlackBoxModel, Chain05PureLtiCabAndRoom};

    // 1. Strided 60,000-sample skewed-amplitude stem (`gram_frame_stride = 4`):
    // `eval_4tier_post_trim_besr_lin` must share the exact sampled inner product of `tier_gram_lr`.
    for phase_hit in [true, false] {
        let n = 60_000usize;
        let mut pulse_dry = vec![[0.0_f32; 2]; n];
        let mut pulse_wet = vec![[0.0_f32; 2]; n];
        for i in 512..n {
            let on_stride = (i - 512) % 4 == 0;
            let amp = if on_stride == phase_hit {
                0.75_f32
            } else {
                0.06_f32
            };
            let x = amp * ((i as f32) * 0.13).sin();
            pulse_dry[i] = [x, x];
            pulse_wet[i] = [0.68 * x, 0.68 * x];
        }
        let mut man = DnamProjManifest::default_reference();
        man.base.io_config.inputs = 1;
        man.base.io_config.output_mode = "mono".to_string();
        let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let _ = lift_stage_a_backbone(
            &man,
            &mut arena,
            &pulse_dry,
            &pulse_wet,
            None,
            StageAConfig::default(),
        );
        let rb = compute_role_b_multipliers(&arena);
        let h = harvest_star_center_256d(&man, &arena, &pulse_dry, &pulse_wet, &[], 4).unwrap();
        let sc = solve_stage_c_nested_schur_4tier(
            &mut arena,
            Some(&h),
            &h.packed_gram_256,
            &h.cross_256,
            &rb,
        )
        .unwrap();
        let mut w_skew = [[0.0_f64; 2]; 256];
        for d in 0..64 {
            w_skew[d] = [
                0.25 * sc.effective_w0_256[d][0],
                0.25 * sc.effective_w0_256[d][1],
            ];
        }
        w_skew[64..256].copy_from_slice(&sc.effective_w0_256[64..256]);
        let (skew_lin, _) =
            eval_4tier_post_trim_besr_lin(h.besr_stats.as_ref().unwrap(), &w_skew, h.eval_stereo);
        assert!(
            skew_lin.iter().all(|&v| v.is_finite() && v >= 0.0),
            "strided post-trim quadratic form must remain PSD and finite: {skew_lin:?}"
        );
        write_effective_mode0_weights(&mut arena, &w_skew, &rb);
        let _ = finalize_stage_e_and_telemetry(
            &mut man,
            &mut arena,
            &[],
            &pulse_dry,
            Some(&pulse_wet),
            None,
            0.0,
        )
        .unwrap();
    }

    // 2. 48,000-sample (`1.0 s`) Chain-05 capture where `[0..12_000]` is low-note `q1` (`< 1.5 kHz`)
    // and `[36_000..48_000]` is broadband `q4`: Stage A must select the broadband 12k window!
    let seed = deterministic_seed_256(0x48_000_C05);
    let (dry_48k, _) = synthesize_a5_musical_transient_coreset(&seed, 48_000, 1, 48_000);
    let mut cab = Chain05PureLtiCabAndRoom::new();
    cab.reset();
    let ctrls: Vec<f32> = cab
        .metadata()
        .controls
        .iter()
        .map(|c| c.default_norm)
        .collect();
    let sws: Vec<u8> = cab
        .metadata()
        .switches
        .iter()
        .map(|s| s.default_index)
        .collect();
    let mut out_l = vec![0.0_f32; 48_000];
    let mut out_r = vec![0.0_f32; 48_000];
    for (i, &x) in dry_48k.iter().enumerate() {
        let mut y = [0.0_f32; 2];
        cab.process_frame(&[x], &mut y, &ctrls, &sws);
        out_l[i] = y[0];
        out_r[i] = y[1];
    }
    let q16 = |v: f32| fp16_bits_to_f32(f32_to_fp16_bits(v));
    let dry_lr: Vec<[f32; 2]> = dry_48k.iter().map(|&x| [q16(x), q16(x)]).collect();
    let wet_lr: Vec<[f32; 2]> = out_l
        .iter()
        .zip(&out_r)
        .map(|(&l, &r)| [q16(l), q16(r)])
        .collect();
    let mut man48 = DnamProjManifest::default_reference();
    man48.base.io_config.inputs = 1;
    man48.base.io_config.output_mode = "stereo".to_string();
    let mut arena48 = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    let _ = lift_stage_a_backbone(
        &man48,
        &mut arena48,
        &dry_lr,
        &wet_lr,
        None,
        StageAConfig::default(),
    );
    let rb48 = compute_role_b_multipliers(&arena48);
    let h48 = harvest_star_center_256d(&man48, &arena48, &dry_lr, &wet_lr, &[], 4).unwrap();
    let sc48 = solve_stage_c_nested_schur_4tier(
        &mut arena48,
        Some(&h48),
        &h48.packed_gram_256,
        &h48.cross_256,
        &rb48,
    )
    .unwrap();
    let se48 = finalize_stage_e_and_telemetry(
        &mut man48,
        &mut arena48,
        &[],
        &dry_lr,
        Some(&wet_lr),
        Some(sc48.prefix_esr_db),
        0.0,
    )
    .unwrap();
    assert!(
        se48.tier_esr_db[3] < -12.0
            && se48.tier_esr_db[3] <= se48.tier_esr_db[2] + 1.0e-3
            && se48.tier_esr_db[2] <= se48.tier_esr_db[1] + 1.0e-3
            && se48.tier_esr_db[1] <= se48.tier_esr_db[0] + 1.0e-3,
        "48k Chain-05 must achieve < -12 dB monotone 4-tier bESR, got {:?}",
        se48.tier_esr_db
    );
}

#[test]
fn test_silent_muted_target_and_steep_single_band_lti_filters_besr_safety() {
    // verify[train.pipeline.stage-a-poles]
    // verify[train.pipeline.stage-c-schur]
    // verify[train.pipeline.stage-e-pack]
    use dnam_capture::fp16_bits_to_f32;
    use dnam_capture::solver::{
        compute_role_b_multipliers, finalize_stage_e_and_telemetry,
        solve_stage_c_nested_schur_4tier,
    };
    use dnam_capture::stimulus::waveforms::{
        deterministic_seed_256, synthesize_a5_musical_transient_coreset,
    };

    let seed = deterministic_seed_256(0x9988_7766);
    let (dry_l, _) = synthesize_a5_musical_transient_coreset(&seed, 12_000, 1, 48_000);
    let q16 = |v: f32| fp16_bits_to_f32(f32_to_fp16_bits(v));
    let dry_q: Vec<[f32; 2]> = dry_l.iter().map(|&x| [q16(x), q16(x)]).collect();

    // 1. Silent target (`wet_lr == 0.0`): all 4 tiers must output `-70.000 LUFS` silence (`0.000 LU` spread)
    let zero_wet = vec![[0.0_f32; 2]; 12_000];
    let mut man0 = DnamProjManifest::default_reference();
    man0.base.io_config.inputs = 1;
    man0.base.io_config.output_mode = "mono".to_string();
    let mut arena0 = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    let _ = lift_stage_a_backbone(
        &man0,
        &mut arena0,
        &dry_q,
        &zero_wet,
        None,
        StageAConfig::default(),
    );
    let rb0 = compute_role_b_multipliers(&arena0);
    let h0 = harvest_star_center_256d(&man0, &arena0, &dry_q, &zero_wet, &[], 1).unwrap();
    let sc0 = solve_stage_c_nested_schur_4tier(
        &mut arena0,
        Some(&h0),
        &h0.packed_gram_256,
        &h0.cross_256,
        &rb0,
    )
    .unwrap();
    let se0 = finalize_stage_e_and_telemetry(
        &mut man0,
        &mut arena0,
        &[],
        &dry_q,
        Some(&zero_wet),
        Some(sc0.prefix_esr_db),
        0.0,
    )
    .unwrap();
    assert_eq!(sc0.mm_relaxed_hf_bands, 0);
    assert_eq!(se0.tier_lufs, [-70.0; 4]);
    assert_eq!(se0.tier_output_trim_db, [0.0; 4]);

    // 2. Steep brickwall band-pass (`450..800 Hz`) and musical 2-pole LTI high-pass (`3,500 Hz`)
    let mut bp_wet = vec![[0.0_f32; 2]; 12_000];
    let mut hp_wet = vec![[0.0_f32; 2]; 12_000];
    let r_hp450 = (-2.0 * std::f64::consts::PI * 450.0 / 48_000.0).exp();
    let r_lp800 = (-2.0 * std::f64::consts::PI * 800.0 / 48_000.0).exp();
    let r_hp3500 = (-2.0 * std::f64::consts::PI * 3_500.0 / 48_000.0).exp();
    let (mut s_hp8, mut s_lp8, mut s_hp2) = ([0.0_f64; 8], [0.0_f64; 8], [0.0_f64; 2]);
    for (i, &x) in dry_l.iter().enumerate() {
        let mut v = f64::from(x);
        for st in &mut s_hp8 {
            let lp = (1.0 - r_hp450) * v + r_hp450 * *st;
            *st = lp;
            v -= lp;
        }
        for st in &mut s_lp8 {
            *st = (1.0 - r_lp800) * v + r_lp800 * *st;
            v = *st;
        }
        let bq = q16((3.5 * v) as f32);
        bp_wet[i] = [bq, bq];

        let mut vh = f64::from(x);
        for st in &mut s_hp2 {
            let lp = (1.0 - r_hp3500) * vh + r_hp3500 * *st;
            *st = lp;
            vh -= lp;
        }
        let hq = q16((1.8 * vh) as f32);
        hp_wet[i] = [hq, hq];
    }

    for (label, wet, check_low_xover) in [
        ("brickwall_bp_450_800", &bp_wet, true),
        ("lti_hp_3500hz", &hp_wet, true),
    ] {
        let mut man = DnamProjManifest::default_reference();
        man.base.io_config.inputs = 1;
        man.base.io_config.output_mode = "mono".to_string();
        let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let sa =
            lift_stage_a_backbone(&man, &mut arena, &dry_q, wet, None, StageAConfig::default());
        if check_low_xover {
            assert!(
                sa.crossover_hz <= 80.0,
                "{label} high-passed target must lower Stage-0 sub crossover <= 80 Hz, got {}",
                sa.crossover_hz
            );
        }
        let rb = compute_role_b_multipliers(&arena);
        let h = harvest_star_center_256d(&man, &arena, &dry_q, wet, &[], 1).unwrap();
        let sc = solve_stage_c_nested_schur_4tier(
            &mut arena,
            Some(&h),
            &h.packed_gram_256,
            &h.cross_256,
            &rb,
        )
        .unwrap();
        let se = finalize_stage_e_and_telemetry(
            &mut man,
            &mut arena,
            &[],
            &dry_q,
            Some(wet),
            Some(sc.prefix_esr_db),
            0.0,
        )
        .unwrap();
        assert!(
            se.tier_esr_db[3] < -5.0
                && se.tier_esr_db[3] <= se.tier_esr_db[2] + 1.0e-2
                && se.tier_esr_db[2] <= se.tier_esr_db[1] + 1.0e-2
                && se.tier_esr_db[1] <= se.tier_esr_db[0] + 1.0e-2,
            "{label} must achieve < -5 dB monotone 4-tier bESR, got {:?}",
            se.tier_esr_db
        );
    }
}
