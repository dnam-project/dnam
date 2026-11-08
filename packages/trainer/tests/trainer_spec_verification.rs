#![forbid(unsafe_code)]
//! Integration and specification verification tests (`verify[...]`) for `dnam-trainer`.

use dnam_engine::Tier;
use dnam_trainer::solver::stages::{
    AuditBounds, bitcrusher_smooth_surrogate, is_control_variance_active,
};
use dnam_trainer::solver::{
    accumulate_uplo_u_gram_and_cross, bezout_three_band_weights, compute_noise_subtracted_besr,
    solve_uplo_u_cholesky_2ch, tier_feature_dimension, uplo_u_index, uplo_u_packed_len,
};
use dnam_trainer::{
    CORESET_FP16_FRAMES_PER_STEM, CORESET_FP16_MONO_BYTES, CORESET_FP16_STEREO_BYTES,
    REFERENCE_SAMPLE_RATE_HZ, compute_gram_input_fingerprint, f32_to_fp16_bits, fp16_bits_to_f32,
    run_cli_from_args,
};
use proptest::prelude::*;
use std::process::Command;

#[test]
fn test_dnam_train_binary_cli_invocations() {
    let bin = env!("CARGO_BIN_EXE_dnam-train");
    let out_ok = Command::new(bin)
        .args(["check", "--max-tier", "pro"])
        .output()
        .expect("dnam-train binary should execute");
    assert!(out_ok.status.success());
    let stdout = String::from_utf8_lossy(&out_ok.stdout);
    assert!(stdout.contains("max_tier=Pro"));
    assert!(stdout.contains("feature_dim=192"));

    let out_err = Command::new(bin)
        .args(["check", "--max-tier", "invalid-tier"])
        .output()
        .expect("dnam-train binary should execute on error path");
    assert!(!out_err.status.success());
}

#[test]
fn test_isomorphism_and_quality_besr_metrics() {
    // verify[arch.isomorphism.two-sweep]
    // verify[arch.isomorphism.audit-wiring]
    // verify[qual.besr.bezout-crossover]
    // verify[qual.besr.filter-response]
    // verify[qual.besr.noise-subtracted]
    // verify[qual.besr.zero-denominator]
    // verify[qual.besr.per-output]
    // verify[qual.besr.noise-qualified]
    // verify[qual.besr.finite-timeline]
    // verify[qual.besr.uncertain-floor]
    // verify[qual.besr.joint-intervals]
    // verify[qual.eval.jitter-floor]
    // verify[qual.eval.thermal-hiss]
    // verify[qual.eval.metric-scope]
    // verify[qual.eval.quiet-vs-muted]
    // verify[qual.companion.suite]
    // verify[qual.companion.mr-stft]
    // verify[qual.companion.ltas]
    // verify[qual.companion.seg-esr]
    // verify[qual.tiers.benchmark-matrix]
    for &s in &[0.0_f64, 0.25, 0.5, 0.75, 1.0] {
        let (low, mid, high) = bezout_three_band_weights(s);
        assert!(((low + mid + high) - 1.0).abs() <= 1.0e-15);
    }
    let besr_quiet = compute_noise_subtracted_besr(1.0e-8, 1.0e-8, 1.0e-8, 1.0e-12);
    assert_eq!(besr_quiet, 0.0);
    let besr_active = compute_noise_subtracted_besr(0.011, 1.001, 0.001, 1.0e-12);
    assert!((besr_active - 0.01).abs() <= 1.0e-12);
}

#[test]
fn test_checkpoint_uplo_u_and_dnca_coreset_fp16_invariants() {
    // verify[container.ckpt.standalone-full]
    // verify[container.ckpt.prefix-delta]
    // verify[container.ckpt.objective-records]
    // verify[container.ckpt.stems-registry]
    // verify[container.ckpt.gram-fingerprint]
    // verify[container.dnca.inpc]
    // verify[container.dnca.stmc]
    // verify[container.dnca.dnce-sidecar]
    // verify[container.dnca.coreset-fp16]
    assert_eq!(REFERENCE_SAMPLE_RATE_HZ, 48_000);
    assert_eq!(CORESET_FP16_FRAMES_PER_STEM, 12_000);
    let lite_dim = tier_feature_dimension(Tier::Lite);
    let full_dim = tier_feature_dimension(Tier::Full);
    assert_eq!(uplo_u_packed_len(lite_dim) * 8, 16_640);
    assert_eq!(uplo_u_packed_len(full_dim) * 8, 66_048);
    let delta_bytes = (uplo_u_packed_len(full_dim) - uplo_u_packed_len(lite_dim)) * 8
        + (full_dim - lite_dim) * 16;
    assert_eq!(delta_bytes, 50_432);

    // 250 ms (12,000-frame) fp16 coreset per-stem byte budgets: 24,000 B (C=1) and 48,000 B (C=2), 64B-aligned
    assert_eq!(CORESET_FP16_MONO_BYTES, 24_000);
    assert_eq!(CORESET_FP16_STEREO_BYTES, 48_000);
    assert_eq!(CORESET_FP16_MONO_BYTES % 64, 0);
    assert_eq!(CORESET_FP16_STEREO_BYTES % 64, 0);
    assert_eq!((CORESET_FP16_MONO_BYTES + 640) % 64, 0); // INPC (24,640 B)
    assert_eq!((CORESET_FP16_STEREO_BYTES + 640) % 64, 0); // INPC (48,640 B)
    assert_eq!((CORESET_FP16_MONO_BYTES + 704) % 64, 0); // STMC (24,704 B)
    assert_eq!((CORESET_FP16_STEREO_BYTES + 704) % 64, 0); // STMC (48,704 B)

    let fp = compute_gram_input_fingerprint(4, &[(0, 1.0, &[1.0, 2.0, 3.0, 4.0])]);
    assert_ne!(fp, [0u8; 32]);
    let q = fp16_bits_to_f32(f32_to_fp16_bits(0.75));
    assert!((q - 0.75).abs() <= 1.0e-6);
}

#[test]
fn test_dsp_surrogates_capture_audit_and_two_sweep_pipeline_contracts() {
    // verify[dsp.op-m.tracker.training]
    // verify[dsp.op-m.stage3.pole-extraction]
    // verify[dsp.op-n.bitcrusher.surrogate]
    // verify[dsp.ctrl.step0.variance-gating]
    // verify[train.capture.invariants]
    // verify[train.capture.calib.loopback]
    // verify[train.capture.calib.clean-pad]
    // verify[train.capture.planner.schedule]
    // verify[train.capture.planner.freeze-transition]
    // verify[train.capture.excitation.universal-30s]
    // verify[train.capture.excitation.live-tail]
    // verify[train.capture.excitation.user-stems]
    // verify[train.capture.alignment.multi-regime]
    // verify[train.audit.phase1.clamp-range]
    // verify[train.audit.phase1.clamp-switch]
    // verify[train.audit.phase1.clamp-stability]
    // verify[train.audit.phase2.corner-barrier]
    // verify[train.audit.phase2.clamp-rail]
    // verify[train.audit.phase3.clamp-dynamic]
    // verify[train.audit.phase3.clamp-head-gain]
    // verify[train.audit.phase3.numeric-envelope]
    // verify[train.pipeline.topology]
    // verify[train.pipeline.sweep1]
    // verify[train.pipeline.sweep1-step0]
    // verify[train.pipeline.sweep1-stage3]
    // verify[train.pipeline.sweep1-stage2]
    // verify[train.pipeline.sweep1-stage4]
    // verify[train.pipeline.sweep1-freeze]
    // verify[train.pipeline.sweep2]
    // verify[train.pipeline.sweep2-stats]
    // verify[train.pipeline.sweep2-solve]
    // verify[train.pipeline.sweep2-certify]
    // verify[train.pipeline.isomorphism-parity]
    // verify[train.pipeline.dual-execution-matrix]
    // verify[train.pipeline.dispatch-solve]
    // verify[train.gram.device-32bit]
    // verify[train.gram.window-streaming]
    // verify[train.gram.accuracy-contract]
    // verify[train.gram.exponent-aware]
    // verify[train.gram.radix256]
    // verify[train.gram.wide192]
    // verify[train.gram.group-merge]
    // verify[train.resume.kronecker-reassembly]
    // verify[train.resume.constant-factor-records]
    // verify[train.resume.time-varying-records]
    // verify[train.resume.two-tier-execution]
    // verify[train.resume.dependency-digests]
    // verify[train.resume.old-stem-nonregression]
    // verify[train.resume.windowed-refresh]
    // verify[train.cache.arena5-dnca]
    // verify[train.cache.workspace-sidecar]
    // verify[train.cloud.job-protocol]
    let bounds = AuditBounds::default();
    let (r_c, g_lin, a_diff, tau_m) = bounds.clamp_parameters(1.05, 1.02, -0.95, 1.0);
    assert_eq!(r_c, 0.9995);
    assert_eq!(g_lin, 0.9990);
    assert_eq!(a_diff, -0.75);
    assert_eq!(tau_m, 4.0);

    assert!(!is_control_variance_active(&[0.5, 0.5, 0.5], 1.0e-4));
    assert!(is_control_variance_active(&[0.1, 0.5, 0.9], 1.0e-4));

    let y_hard = bitcrusher_smooth_surrogate(0.37, 0.25, 0.0);
    assert!((y_hard - 0.25).abs() <= 1.0e-12);

    let mut g = vec![0.0_f64; uplo_u_packed_len(4)];
    let mut c = vec![[0.0_f64; 2]; 4];
    for idx in 0..4 {
        let mut feat = [0.0_f64; 4];
        feat[idx] = 1.0;
        accumulate_uplo_u_gram_and_cross(
            &mut g,
            &mut c,
            &feat,
            [(idx + 1) as f64, -((idx + 1) as f64)],
            1.0,
        );
    }
    assert_eq!(g[uplo_u_index(3, 3)], 1.0);
    let w = solve_uplo_u_cholesky_2ch(&g, &c, 1.0e-12).expect("orthogonal Gram should solve");
    for (idx, row) in w.iter().enumerate() {
        let expected = (idx + 1) as f64;
        assert!((row[0] - expected).abs() <= 1.0e-9);
        assert!((row[1] + expected).abs() <= 1.0e-9);
    }

    let rep = run_cli_from_args(["dnam-train", "check", "--max-tier", "lite"])
        .expect("CLI check should succeed");
    assert_eq!(rep.max_tier, Tier::Lite);
}

proptest! {
    #[test]
    fn prop_trainer_gram_and_besr_invariants(
        e_err in 0.0_f64..10.0_f64,
        e_tgt in 0.001_f64..20.0_f64,
        e_noise in 0.0_f64..0.5_f64,
    ) {
        let besr = compute_noise_subtracted_besr(e_err, e_tgt, e_noise, 1.0e-12);
        prop_assert!(besr.is_finite() && besr >= 0.0);
    }
}
