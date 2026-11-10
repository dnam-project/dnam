#![forbid(unsafe_code)]
//! Integration and specification verification tests (`verify[...]`) for `dnam-trainer`.

use dnam_engine::container::{
    ARENA_OFFSET_ULTRA_END, build_dnam_container, strip_container_to_tier,
};
use dnam_engine::control::{ExactIntFactorAccumulator, eval_newton_lobatto_basis_f64};
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::provenance::{
    PHASE_CODING_MAX_RAD, STAGE3_KAUTZ_BLOCKS_PER_TIER, apply_cascade_head_gauge,
    apply_isometric_gauge_watermark, apply_phase_coding_watermark, apply_tardos_watermark_7680,
    extract_stage3_tier_fingerprint_component, extract_structural_hash_224d,
    generate_tardos_code_7680, is_role_a_cascade_index, kautz_soa_byte_offset,
    score_phase_coding_arena, score_tardos_watermark_7680, structural_cosine_similarity,
};
use dnam_engine::schema::DnamJsonHeader;
use dnam_engine::{DnamAudioIo, DnamEngine, Tier};
use dnam_trainer::ckpt::{
    ARENA4_STEM_BYTES_ULTRA, DnamCkptHeader, build_dnam_ckpt_container,
    export_runtime_dnam_from_ckpt, is_valid_arena4_stem_slice_len, pack_uplo_u_f64,
    unpack_uplo_u_f64,
};
use dnam_trainer::metrics::{
    ArchetypeFamily, BESR_WORST_BAND_OFFSET_DB, IntervalAuditDecision, MultiRateSegEsrSlice,
    StemActivityClass, canonical_ltas_33_centers_hz, certify_4metric_signal_chain,
    check_noise_qualified_besr_acceptance, classify_and_verify_quiet_or_muted_stem,
    classify_and_verify_quiet_or_muted_with_evidence, compute_ltas_33band_max_deviation_db,
    compute_mr_iir_loss, compute_multirate_seg_esr_db, compute_per_output_besr,
    compute_seg_esr_20ms_db, eval_bezout_3band_weights, evaluate_besr_from_band_energies,
    evaluate_joint_energy_interval_audit, evaluate_ltas_33band_report,
    evaluate_noise_qualified_besr, evaluate_uncertain_noise_interval_bounds,
    split_bezout_3band_energies, try_split_bezout_3band_energies,
    verify_cross_tier_loudness_parity, verify_monotone_4tier_summaries, verify_monotone_tier_besr,
};
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
};
use proptest::prelude::*;
use std::f64::consts::PI;

#[cfg(feature = "cli")]
#[test]
fn test_dnam_train_binary_cli_invocations() {
    use std::process::Command;
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
    // verify[qual.companion.mr-iir]
    // verify[qual.companion.ltas]
    // verify[qual.companion.seg-esr]
    // verify[qual.tiers.benchmark-matrix]
    for &s in &[0.0_f64, 0.25, 0.5, 0.75, 1.0] {
        let (low, mid, high) = bezout_three_band_weights(s);
        assert!(((low + mid + high) - 1.0).abs() <= 1.0e-14);
    }
    let besr_quiet = compute_noise_subtracted_besr(1.0e-8, 1.0e-8, 1.0e-8);
    assert_eq!(besr_quiet, 0.0);
    let besr_active = compute_noise_subtracted_besr(0.011, 0.999, 0.001);
    assert!((besr_active - 0.01).abs() <= 1.0e-12);

    // Full 18-rule qual.* suite verification
    for f in [20.0, 250.0, 1000.0, 2500.0, 12_000.0] {
        let w = eval_bezout_3band_weights(f);
        assert!(((w[0] + w[1] + w[2]) - 1.0).abs() <= 1.0e-12);
    }

    let rep_collapse =
        evaluate_besr_from_band_energies([1.0, 0.0, 1.0], [0.001, 0.01, 0.001], [0.0; 3]);
    assert!(rep_collapse.denominator_collapse[1]);
    assert!(!check_noise_qualified_besr_acceptance(&rep_collapse, -20.0));

    // Underflow-erased nonzero energy (`0 < |x| < 1.49e-154`) must render certification unavailable (`r[qual.besr.zero-denominator]`)
    let uf_tgt = vec![1.0e-180_f64; 2048];
    let uf_zero = vec![0.0_f64; 2048];
    let uf_nse = vec![1.0e-220_f64; 2048];
    assert!(
        try_split_bezout_3band_energies(&uf_tgt, &uf_zero, Some(&uf_nse), 48_000.0, 0).is_none()
    );
    let (uf_s, uf_e, uf_n) =
        split_bezout_3band_energies(&uf_tgt, &uf_zero, Some(&uf_nse), 48_000.0, 0);
    let rep_uf = evaluate_besr_from_band_energies(uf_s, uf_e, uf_n);
    assert!(!rep_uf.valid && !evaluate_noise_qualified_besr(&rep_uf, [-40.0; 3], -34.0).passes);
    assert!(compute_mr_iir_loss(&uf_tgt, &uf_nse, None, 48_000.0, 0).is_infinite());
    assert!(
        compute_mr_iir_loss(
            &[1.0_f64; 2048],
            &[1.0_f64; 2048],
            Some(&uf_nse),
            48_000.0,
            0
        )
        .is_infinite()
    );
    assert!(
        evaluate_ltas_33band_report(
            &[1.0_f64; 2048],
            &[1.0_f64; 2048],
            Some(&uf_nse),
            48_000.0,
            true
        )
        .max_adj_deviation_db
        .is_infinite()
    );
    assert!(
        !compute_multirate_seg_esr_db(&[MultiRateSegEsrSlice {
            sample_rate_hz: 48_000.0,
            target: &uf_zero[..960],
            model: &uf_tgt[..960],
            noise: Some(&uf_nse[..960]),
        }])
        .2
    );
    assert_eq!(
        classify_and_verify_quiet_or_muted_with_evidence(
            &uf_nse[..8],
            &uf_tgt[..8],
            Some(0.0),
            false
        ),
        (StemActivityClass::Unavailable, false)
    );
    assert_eq!(
        classify_and_verify_quiet_or_muted_with_evidence(
            &uf_tgt[..8],
            &uf_tgt[..8],
            Some(1.0e-156),
            true
        ),
        (StemActivityClass::Unavailable, false)
    );

    let rep_clean =
        evaluate_besr_from_band_energies([1.0, 1.0, 1.0], [0.0001, 0.0002, 0.0003], [0.0; 3]);
    assert!(rep_clean.worst_band_db <= rep_clean.besr_db + BESR_WORST_BAND_OFFSET_DB + 1.0e-9);
    assert!(check_noise_qualified_besr_acceptance(&rep_clean, -30.0));

    let (_, _, dec_pass) = evaluate_uncertain_noise_interval_bounds(
        [1.0, 1.0, 1.0],
        [0.0001, 0.0001, 0.0001],
        [1.0e-6; 3],
        -30.0,
    );
    assert_eq!(dec_pass, IntervalAuditDecision::CertifiedPass);
    let (_, _, dec_fail) = evaluate_uncertain_noise_interval_bounds(
        [1.0, 1.0, 1.0],
        [0.2, 0.2, 0.2],
        [1.0e-6; 3],
        -30.0,
    );
    assert_eq!(dec_fail, IntervalAuditDecision::CertifiedFail);
    let (_, _, dec_ind) = evaluate_uncertain_noise_interval_bounds(
        [1.0, 1.0, 1.0],
        [0.0015, 0.0015, 0.0015],
        [0.002; 3],
        -30.0,
    );
    assert_eq!(
        dec_ind,
        IntervalAuditDecision::IndeterminateRequiresQuieterCapture
    );

    let sr = 48_000.0;
    let n = 4096usize;
    let sig_l: Vec<f64> = (0..n)
        .map(|i| {
            0.35 * (2.0 * PI * 110.0 * (i as f64) / sr).sin()
                + 0.25 * (2.0 * PI * 1000.0 * (i as f64) / sr).cos()
                + 0.15 * (2.0 * PI * 5000.0 * (i as f64) / sr).sin()
        })
        .collect();
    let mdl_l: Vec<f64> = sig_l.iter().map(|&x| x * 0.999).collect();
    let (per_ch, worst_db) =
        compute_per_output_besr(&[&sig_l, &sig_l], &[&mdl_l, &mdl_l], None, sr, 64);
    assert_eq!(per_ch.len(), 2);
    assert!(worst_db < -50.0);

    // Asymmetric stereo failure (r[qual.besr.per-output]): clean L + corrupted R must fail worst_db
    let bad_r: Vec<f64> = sig_l.iter().map(|&x| x * 0.5).collect();
    let (_asym_ch, asym_worst_db) =
        compute_per_output_besr(&[&sig_l, &sig_l], &[&mdl_l, &bad_r], None, sr, 64);
    assert!(asym_worst_db > -10.0);

    // Zero-denominator +inf behavior (r[qual.besr.zero-denominator]) and full-timeline pre-roll continuity (r[qual.besr.finite-timeline])
    let rep_zero_all = evaluate_besr_from_band_energies([0.0; 3], [1.0e-5; 3], [0.0; 3]);
    assert!(rep_zero_all.besr_lin.is_infinite());
    assert!(rep_zero_all.besr_db.is_infinite() && rep_zero_all.besr_db.is_sign_positive());

    // Filter-before-selection proof: a DC step settled over a 1024-sample pre-roll has ~0 MF/HF energy,
    // whereas zero-initializing at the selection boundary injects startup transient energy into MF/HF.
    let dc_step = vec![1.0_f64; 1536];
    let (dc_preroll_sig, _, _) = split_bezout_3band_energies(&dc_step, &dc_step, None, sr, 1024);
    let (dc_sliced_sig, _, _) =
        split_bezout_3band_energies(&dc_step[1024..], &dc_step[1024..], None, sr, 0);
    assert!(dc_preroll_sig[1] < 1.0e-12 && dc_preroll_sig[2] < 1.0e-12);
    assert!(dc_sliced_sig[1] > 0.1 && dc_sliced_sig[2] > 0.1);

    // 5-row joint-interval audit (r[qual.besr.joint-intervals])
    let joint = evaluate_joint_energy_interval_audit(
        [[1.0, 1.0]; 3],
        [[1.0e-4, 1.0e-4]; 3],
        [[1.0e-6, 1.0e-6]; 3],
        [-30.0; 3],
        -30.0,
    );
    assert_eq!(joint.decision, IntervalAuditDecision::CertifiedPass);

    // Jitter-floor sensitivity (r[qual.eval.jitter-floor]) & thermal-hiss noise-qualified pass (r[qual.eval.thermal-hiss])
    let jitter_mdl: Vec<f64> = (0..n)
        .map(|i| {
            let dt = (i as f64) + 0.05; // 50 milli-sample timing error degrades high-frequency phase
            0.35 * (2.0 * PI * 110.0 * dt / sr).sin()
                + 0.25 * (2.0 * PI * 1000.0 * dt / sr).cos()
                + 0.15 * (2.0 * PI * 5000.0 * dt / sr).sin()
        })
        .collect();
    let (_, jitter_besr_db) = compute_per_output_besr(&[&sig_l], &[&jitter_mdl], None, sr, 64);
    assert!(jitter_besr_db > worst_db + 15.0);

    // Broadband log-chirp (22 Hz -> 20 kHz, 16,384 samples) exciting all 33 1/3-octave LTAS bands
    let n_chirp = 16_384usize;
    let f0 = 22.0_f64;
    let f1 = 20_000.0_f64;
    let ln_ratio = (f1 / f0).ln();
    let dur = (n_chirp as f64) / sr;
    let chirp: Vec<f64> = (0..n_chirp)
        .map(|i| {
            let t = (i as f64) / sr;
            let phase = 2.0 * PI * f0 * dur * ((t / dur * ln_ratio).exp() - 1.0) / ln_ratio;
            0.55 * phase.sin()
        })
        .collect();
    let mdl_chirp: Vec<f64> = chirp.iter().map(|&x| x * 0.9998).collect();

    let pseudo_white = |seed: u64, amp: f64| -> Vec<f64> {
        (0..n_chirp)
            .map(|i| {
                let mut z = seed
                    .wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                    .wrapping_add(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                let u = ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64)) - 0.5;
                amp * 2.0 * u
            })
            .collect()
    };

    // 1. Elevated thermal hiss (~ -27 dBFS): uncalibrated bESR fails Ultra, while noise-qualified bESR passes
    let elev_hiss_tgt = pseudo_white(0x1111_2222, 0.08);
    let elev_hiss_cal = pseudo_white(0x3333_4444, 0.08);
    let noisy_elev: Vec<f64> = chirp
        .iter()
        .zip(&elev_hiss_tgt)
        .map(|(s, h)| s + h)
        .collect();
    let raw_elev = certify_4metric_signal_chain(
        ArchetypeFamily::CleanPedalPreamp,
        Tier::Ultra,
        &noisy_elev,
        &mdl_chirp,
        None,
        sr,
        128,
    );
    assert!(!raw_elev.passes_target);
    let (se_el, ee_el, ne_el) =
        split_bezout_3band_energies(&noisy_elev, &mdl_chirp, Some(&elev_hiss_cal), sr, 128);
    let rep_el = evaluate_besr_from_band_energies(se_el, ee_el, ne_el);
    let qual_el = evaluate_noise_qualified_besr(&rep_el, [-38.0, -35.0, -30.0], -34.0);
    assert!(
        qual_el.qualified_band_mask.iter().any(|&q| q) && qual_el.passes,
        "expected noise-qualified pass: rep={rep_el:?}, qual={qual_el:?}"
    );

    // 2. Realistic studio thermal hiss (~ -74 dBFS): full 4-metric Ultra certification passes with calibrated noise floor
    let hiss = pseudo_white(0x5555_6666, 2.5e-4);
    let hiss_cal = pseudo_white(0x7777_8888, 2.5e-4);
    let noisy_tgt: Vec<f64> = chirp.iter().zip(&hiss).map(|(s, h)| s + h).collect();
    let hiss_summary = certify_4metric_signal_chain(
        ArchetypeFamily::CleanPedalPreamp,
        Tier::Ultra,
        &noisy_tgt,
        &mdl_chirp,
        Some(&hiss_cal),
        sr,
        128,
    );
    assert!(hiss_summary.passes_target);

    // 2b. Transient burst followed by an idle pause on noisy hardware:
    // Every 1/3-octave band has high overall SNR (so bESR and noise-adjusted LTAS pass Ultra cleanly),
    // whereas during the idle pause random target-hiss peaks exceed sigma_{M,k,h} in MR-IIR (> 0.05).
    // Benchmark Matrix certification passes because MR-IIR/SegESR are enforced via monotone non-regression
    // rather than fixed device-agnostic Benchmark Matrix cutoffs.
    let n_burst = 32_768usize;
    let burst_clean: Vec<f64> = (0..n_burst)
        .map(|i| if i < n_chirp { chirp[i] } else { 0.0 })
        .collect();
    let pseudo_white_n = |seed: u64, len: usize, amp: f64| -> Vec<f64> {
        (0..len)
            .map(|i| {
                let mut z = seed
                    .wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                    .wrapping_add(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                let u = ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64)) - 0.5;
                amp * 2.0 * u
            })
            .collect()
    };
    let burst_hiss_tgt = pseudo_white_n(0x9999_AAAA, n_burst, 5.0e-4);
    let burst_hiss_cal = pseudo_white_n(0xBBBB_CCCC, n_burst, 5.0e-4);
    let burst_noisy_tgt: Vec<f64> = burst_clean
        .iter()
        .zip(&burst_hiss_tgt)
        .map(|(s, h)| s + h)
        .collect();
    let burst_ultra_summary = certify_4metric_signal_chain(
        ArchetypeFamily::CleanPedalPreamp,
        Tier::Ultra,
        &burst_noisy_tgt,
        &burst_clean,
        Some(&burst_hiss_cal),
        sr,
        128,
    );
    assert!(
        burst_ultra_summary.mr_iir > 0.05 && burst_ultra_summary.passes_target,
        "exact deterministic model on burst+pause noisy capture must not fail Ultra Benchmark Matrix due to MR-IIR floor-clamping bias: {burst_ultra_summary:?}"
    );

    // Metric-scope orthogonality (r[qual.eval.metric-scope]): narrow 1/3-octave spectral notch passes bESR Lite ceiling but fails LTAS 0.75 dB
    let mr_iir = compute_mr_iir_loss(&chirp, &mdl_chirp, None, sr, 128);
    assert!(mr_iir < 0.01);
    let centers = canonical_ltas_33_centers_hz();
    assert_eq!(centers.len(), 33);
    let (ltas_db, active_bands) =
        compute_ltas_33band_max_deviation_db(&chirp, &mdl_chirp, None, sr);
    assert!(active_bands == 33 && ltas_db < 0.05);

    // Apply a narrowband biquad peaking cut (-2.5 dB, Q = 8.0 at 4 kHz) to chirp:
    // wideband HF bESR easily passes Lite (-18 dB composite / -15 dB HF), while LTAS catches > 0.75 dB deviation!
    let w0 = 2.0 * PI * 4000.0 / sr;
    let alpha = w0.sin() / (2.0 * 8.0);
    let amp_a = 10.0_f64.powf(-2.5 / 40.0);
    let b0 = 1.0 + alpha * amp_a;
    let b1 = -2.0 * w0.cos();
    let b2 = 1.0 - alpha * amp_a;
    let a0 = 1.0 + alpha / amp_a;
    let a1 = -2.0 * w0.cos();
    let a2 = 1.0 - alpha / amp_a;
    let (mut x1, mut x2, mut y1, mut y2) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let mut notched_mdl = Vec::with_capacity(n_chirp);
    for &x in &chirp {
        let y = (b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0;
        x2 = x1;
        x1 = x;
        y2 = y1;
        y1 = y;
        notched_mdl.push(y);
    }
    let (notch_besr_reps, _) = compute_per_output_besr(&[&chirp], &[&notched_mdl], None, sr, 128);
    let notch_lite_besr =
        evaluate_noise_qualified_besr(&notch_besr_reps[0], [-23.0, -20.0, -15.0], -18.0);
    assert!(
        notch_lite_besr.passes,
        "narrow notch should pass wideband Lite bESR: {:?}",
        notch_besr_reps[0].band_esr_db
    );
    let (notch_ltas_db, _) = compute_ltas_33band_max_deviation_db(&chirp, &notched_mdl, None, sr);
    assert!(notch_ltas_db > 1.0);
    let notch_summary = certify_4metric_signal_chain(
        ArchetypeFamily::CleanPedalPreamp,
        Tier::Lite,
        &chirp,
        &notched_mdl,
        None,
        sr,
        128,
    );
    assert!(!notch_summary.passes_target);

    let (seg_db, active_segs) = compute_seg_esr_20ms_db(&chirp, &mdl_chirp, None, sr);
    assert!(active_segs > 0 && seg_db < -50.0);

    // Without independent mute evidence, level alone classifies as QuietTail (never MutedGateClosed)
    let (cls_unconfirmed_zero, ok_unconfirmed_zero) =
        classify_and_verify_quiet_or_muted_stem(&vec![0.0; 256], &vec![1.0e-7; 256], 1.0e-6);
    assert_eq!(cls_unconfirmed_zero, StemActivityClass::QuietTail);
    assert!(ok_unconfirmed_zero);
    let (cls_muted, ok_muted) = classify_and_verify_quiet_or_muted_with_evidence(
        &vec![0.0; 256],
        &vec![1.0e-7; 256],
        Some(1.0e-6),
        true,
    );
    assert_eq!(cls_muted, StemActivityClass::MutedGateClosed);
    assert!(ok_muted);
    let (cls_ev, ok_ev) = classify_and_verify_quiet_or_muted_with_evidence(
        &vec![1.0e-6; 256],
        &vec![1.0e-3; 256],
        Some(1.0e-5),
        true,
    );
    assert_eq!(cls_ev, StemActivityClass::MutedGateClosed);
    assert!(!ok_ev);
    let (cls_quiet, ok_quiet) =
        classify_and_verify_quiet_or_muted_stem(&vec![1.5e-5; 256], &vec![1.5e-5; 256], 1.0e-5);
    assert_eq!(cls_quiet, StemActivityClass::QuietTail);
    assert!(ok_quiet);
    let (cls_act, ok_act) = classify_and_verify_quiet_or_muted_stem(&sig_l, &mdl_l, 1.0e-5);
    assert_eq!(cls_act, StemActivityClass::ActiveSignal);
    assert!(ok_act);

    let (step_lu, lu_ok) = verify_cross_tier_loudness_parity([0.25, 0.2502, 0.2501, 0.2503]);
    assert!(lu_ok && step_lu <= 0.05);
    assert!(verify_monotone_tier_besr([-24.0, -31.0, -36.0, -41.0]));

    for family in [
        ArchetypeFamily::CleanPedalPreamp,
        ArchetypeFamily::CrunchHighGainAmp,
        ArchetypeFamily::FullRigModDelayReverb,
        ArchetypeFamily::FuzzOctaveChaos,
    ] {
        let summaries = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra].map(|tier| {
            certify_4metric_signal_chain(family, tier, &chirp, &mdl_chirp, None, sr, 128)
        });
        for summary in &summaries {
            assert!(summary.passes_target && summary.passes_hard_ceiling);
        }
        assert!(verify_monotone_4tier_summaries(&summaries));
    }
    let _ = split_bezout_3band_energies(&[], &[], None, sr, 0);
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

    #[cfg(feature = "cli")]
    {
        let rep = dnam_trainer::run_cli_from_args(["dnam-train", "check", "--max-tier", "lite"])
            .expect("CLI check should succeed");
        assert_eq!(rep.max_tier, Tier::Lite);
    }
}

#[test]
fn test_oracles_01_02_03_and_cases_08_09_19_engine_trainer_harnesses() {
    // verify[train.pipeline.fixtures-matrix]
    // 1. Oracle-01-256D-Schur-And-Gauge & Case #8 (ProvenanceWatermarkHarness)
    let mut header = DnamJsonHeader::default_reference();
    let mut master_arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    master_arena[124..128].copy_from_slice(&0.9993_f32.to_le_bytes());

    let head_bases = [4_928_usize, 18_048_usize, 31_936_usize, 57_600_usize];
    for tier in 0..4 {
        for blk in 0..STAGE3_KAUTZ_BLOCKS_PER_TIER[tier] {
            let k_base = kautz_soa_byte_offset(tier, blk);
            for lane in 0..4 {
                master_arena[k_base + lane * 4..k_base + (lane + 1) * 4]
                    .copy_from_slice(&0.90_f32.to_le_bytes());
                master_arena[k_base + 16 + lane * 4..k_base + 16 + (lane + 1) * 4]
                    .copy_from_slice(&0.04_f32.to_le_bytes());
                master_arena[k_base + 32 + lane * 4..k_base + 32 + (lane + 1) * 4]
                    .copy_from_slice(&(0.18_f32 + 0.09_f32 * ((tier + blk) as f32)).to_le_bytes());
                master_arena[k_base + 48 + lane * 4..k_base + 48 + (lane + 1) * 4]
                    .copy_from_slice(&0.35_f32.to_le_bytes());
                master_arena[k_base + 64 + lane * 4..k_base + 64 + (lane + 1) * 4]
                    .copy_from_slice(&0.25_f32.to_le_bytes());
                master_arena[k_base + 96 + lane * 4..k_base + 96 + (lane + 1) * 4]
                    .copy_from_slice(&0.20_f32.to_le_bytes());
            }
        }
        let h_base = head_bases[tier];
        for i in 0..640 {
            let val = 0.03_f32 * (((i * 7 + tier * 3 + 1) % 11) as f32 - 5.0);
            master_arena[h_base + i * 4..h_base + (i + 1) * 4].copy_from_slice(&val.to_le_bytes());
        }
    }

    let hash_master = extract_structural_hash_224d(&master_arena).unwrap();
    let fp_master = extract_stage3_tier_fingerprint_component(
        &master_arena[4_288..4_416],
        &master_arena[4_416..4_544],
        4,
    )
    .unwrap();
    assert_eq!(fp_master.len(), 60);

    // Apply dyadic Layer 1 gauge (Oracle-01: 0.000000 ULP audio invariance & fingerprint invariance)
    let mut gauged_arena = master_arena.clone();
    let kappa_dyadic = [2.0_f32; 32];
    let q_ident = [
        [1.0_f32, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    apply_isometric_gauge_watermark(&mut gauged_arena, &kappa_dyadic, &q_ident).unwrap();

    let hash_gauged = extract_structural_hash_224d(&gauged_arena).unwrap();
    assert!(1.0 - structural_cosine_similarity(&hash_master, &hash_gauged) < 1.0e-12);
    let fp_gauged = extract_stage3_tier_fingerprint_component(
        &gauged_arena[4_288..4_416],
        &gauged_arena[4_416..4_544],
        4,
    )
    .unwrap();
    assert_eq!(fp_master, fp_gauged);

    let role_b_dyadic = [[1.0_f32; 4]; 32]; // 1 + delta = 2.0
    apply_cascade_head_gauge(&mut gauged_arena, 0, &role_b_dyadic).unwrap();

    let cont_master =
        build_dnam_container(&mut header, &master_arena, &master_arena, false).unwrap();
    let cont_gauged =
        build_dnam_container(&mut header, &gauged_arena, &gauged_arena, false).unwrap();
    let mut eng_m =
        DnamEngine::<DefaultMathBackend>::from_bytes(&cont_master, Tier::Ultra, 48_000).unwrap();
    let mut eng_g =
        DnamEngine::<DefaultMathBackend>::from_bytes(&cont_gauged, Tier::Ultra, 48_000).unwrap();
    let dry: Vec<f32> = (0..64).map(|i| 0.35 * ((i as f32) * 0.19).sin()).collect();
    let mut out_ml = vec![0.0_f32; 64];
    let mut out_mr = vec![0.0_f32; 64];
    let mut out_gl = vec![0.0_f32; 64];
    let mut out_gr = vec![0.0_f32; 64];
    eng_m.process_block(&dry, &dry, &mut out_ml, &mut out_mr);
    eng_g.process_block(&dry, &dry, &mut out_gl, &mut out_gr);
    for i in 0..64 {
        assert_eq!(out_ml[i].to_bits(), out_gl[i].to_bits());
        assert_eq!(out_mr[i].to_bits(), out_gr[i].to_bits());
    }

    // Case #8: Layer 2 Pole-Angle Phase Coding + Layer 3 TardosCodebook7680 (Role A zeroed, N=5 collusion)
    let buyer_seed = [0x42_u8; 32];
    let mut wm_arena = master_arena.clone();
    apply_phase_coding_watermark(&mut wm_arena, &buyer_seed, PHASE_CODING_MAX_RAD).unwrap();
    let z_phase = score_phase_coding_arena(&master_arena, &wm_arena, &buyer_seed).unwrap();
    assert!(z_phase > 3.0);

    let mkt_secret = [0x77_u8; 32];
    let mut pirate_codes = Vec::new();
    let mut pirate_arenas = Vec::new();
    for p in 0..5 {
        let code = generate_tardos_code_7680(&mkt_secret, "Oracle01Model", &format!("pirate_{p}"));
        for idx in 0..7680 {
            if is_role_a_cascade_index(idx) {
                assert_eq!(code.chips[idx], 0.0);
                assert_eq!(code.accusation_weights[idx], 0.0);
            } else {
                assert!(code.chips[idx].abs() == 1.0);
            }
        }
        let mut p_arena = master_arena.clone();
        apply_tardos_watermark_7680(&mut p_arena, &code.chips, 1.5e-4).unwrap();
        pirate_codes.push(code);
        pirate_arenas.push(p_arena);
    }
    let innocent = generate_tardos_code_7680(&mkt_secret, "Oracle01Model", "innocent_user");
    let z_pirate =
        score_tardos_watermark_7680(&master_arena, &pirate_arenas[0], &pirate_codes[0]).unwrap();
    let z_innocent =
        score_tardos_watermark_7680(&master_arena, &pirate_arenas[0], &innocent).unwrap();
    assert!(z_pirate > 15.0 && z_pirate > z_innocent + 10.0);

    // Genuine N = 5 linear-averaging pirate collusion attack: average all 5 watermarked f32 arenas
    let mut colluded_arena = master_arena.clone();
    for off in (0..colluded_arena.len()).step_by(4) {
        let mut sum = 0.0_f64;
        for pa in &pirate_arenas {
            let val = f32::from_le_bytes([pa[off], pa[off + 1], pa[off + 2], pa[off + 3]]);
            sum += f64::from(val);
        }
        let avg = (sum / (pirate_arenas.len() as f64)) as f32;
        colluded_arena[off..off + 4].copy_from_slice(&avg.to_le_bytes());
    }
    let z_colluded_innocent =
        score_tardos_watermark_7680(&master_arena, &colluded_arena, &innocent).unwrap();
    for p_code in &pirate_codes {
        let z_colluder =
            score_tardos_watermark_7680(&master_arena, &colluded_arena, p_code).unwrap();
        assert!(
            z_colluder > 6.0 && z_colluder > z_colluded_innocent + 5.0,
            "N=5 colluder Z-score {z_colluder} failed separation vs innocent {z_colluded_innocent}"
        );
    }

    // 2. Oracle-02-CP1-ArbitraryP-And-Hadamard-FM (P = 6 and P = 16 controls, StarCenter 0 ULP)
    for num_controls in [6usize, 16usize] {
        let mut acc = ExactIntFactorAccumulator::new(num_controls);
        for p in 0..num_controls {
            let b = eval_newton_lobatto_basis_f64(0.0, [0.0, -1.0, 1.0], 3);
            assert_eq!(b, [0.0, 0.0, 0.0]);
            assert!(acc.update_contributor_rt(p, 0));
        }
        assert_eq!(acc.sum.to_rounded_f32_bits(), Some(0));
        assert_eq!(acc.interaction_f32_bits(), Some(0));
    }

    // 3. Oracle-03-Zero-WAV-Checkpoint-Resume (267,264 B StarCenter Arena 4 + runtime export)
    assert!(is_valid_arena4_stem_slice_len(
        ARENA4_STEM_BYTES_ULTRA,
        4,
        ARENA4_STEM_BYTES_ULTRA
    ));
    let mut dense_g0 = vec![0.0_f64; 64 * 64];
    for i in 0..64 {
        dense_g0[i * 64 + i] = 1.0 + (i as f64) * 0.01;
    }
    let mut packed_g0 = vec![0.0_f64; uplo_u_packed_len(64)];
    pack_uplo_u_f64(&dense_g0, 64, &mut packed_g0);
    let mut unpacked_g0 = vec![0.0_f64; 64 * 64];
    unpack_uplo_u_f64(&packed_g0, 64, &mut unpacked_g0);
    assert_eq!(dense_g0, unpacked_g0);

    let mut ckpt_hdr = DnamCkptHeader::default_reference();
    let ckpt_bytes =
        build_dnam_ckpt_container(&mut ckpt_hdr, &master_arena, &master_arena, false).unwrap();
    let exported_dnam = export_runtime_dnam_from_ckpt(&ckpt_bytes).unwrap();
    let stripped_pro = strip_container_to_tier(&exported_dnam, Tier::Pro).unwrap();
    let mut eng_exp =
        DnamEngine::<DefaultMathBackend>::from_bytes(&stripped_pro, Tier::Pro, 48_000).unwrap();
    let mut eng_ref =
        DnamEngine::<DefaultMathBackend>::from_bytes(&cont_master, Tier::Pro, 48_000).unwrap();
    eng_exp.process_block(&dry, &dry, &mut out_gl, &mut out_gr);
    eng_ref.process_block(&dry, &dry, &mut out_ml, &mut out_mr);
    assert_eq!(out_ml, out_gl);

    // 4. Case #9 (MultiRateZohHarness) & Case #19 (ChannelTopologyMatrixHarness)
    for &sr in &[44_100u32, 48_000, 88_200, 96_000] {
        let mut eng_sr =
            DnamEngine::<DefaultMathBackend>::from_bytes(&cont_master, Tier::Ultra, sr).unwrap();
        eng_sr.set_offline_bounce(sr == 96_000);
        eng_sr.process_block(&dry, &dry, &mut out_ml, &mut out_mr);
        assert!(out_ml.iter().all(|v| v.is_finite()));
    }
    let mut eng_io =
        DnamEngine::<DefaultMathBackend>::from_bytes(&cont_master, Tier::Lite, 48_000).unwrap();
    let mut sc_out_l = vec![0.0_f32; 64];
    let mut sc_out_r = vec![0.0_f32; 64];
    eng_io.process_block_sidechain(&dry, &dry, &dry, &dry, &mut sc_out_l, &mut sc_out_r);
    assert!(sc_out_l.iter().all(|v| v.is_finite()) && sc_out_r.iter().all(|v| v.is_finite()));
    let mut main_l = vec![0.0_f32; 64];
    let mut main_r = vec![0.0_f32; 64];
    let mut send_l = vec![0.0_f32; 64];
    let mut send_r = vec![0.0_f32; 64];
    eng_io.process_block_4cm(
        &dry,
        &dry,
        &dry,
        &dry,
        &mut main_l,
        &mut main_r,
        &mut send_l,
        &mut send_r,
    );
    assert!(send_l.iter().all(|v| v.is_finite()) && main_l.iter().all(|v| v.is_finite()));
    let mut wdw_dry = vec![0.0_f32; 64];
    let mut wdw_l = vec![0.0_f32; 64];
    let mut wdw_r = vec![0.0_f32; 64];
    eng_io.process_block_wdw(&dry, &dry, &mut wdw_l, &mut wdw_r, &mut wdw_dry);
    assert!(wdw_l.iter().all(|v| v.is_finite()) && wdw_r.iter().all(|v| v.is_finite()));
    let mut io_l = vec![0.0_f32; 64];
    let mut io_r = vec![0.0_f32; 64];
    eng_io.process_block_io(DnamAudioIo {
        in_1: &dry,
        in_2: Some(&dry),
        in_3: None,
        in_4: None,
        out_1: &mut io_l,
        out_2: Some(&mut io_r),
        out_3: None,
        out_4: None,
    });
    assert!(io_l.iter().all(|v| v.is_finite()));
}

proptest! {
    #[test]
    fn prop_trainer_gram_and_besr_invariants(
        e_err in 0.0_f64..10.0_f64,
        e_tgt in 0.001_f64..20.0_f64,
        e_noise in 0.0_f64..0.5_f64,
    ) {
        let besr = compute_noise_subtracted_besr(e_err, e_tgt, e_noise);
        prop_assert!(besr.is_finite() && besr >= 0.0);
    }
}
