//! Quiet vs. Muted Stem Classification (`r[qual.eval.quiet-vs-muted]`), Cross-Tier Loudness Parity,
//! and 4-Tier Perceptual Benchmark Certification (`r[qual.companion.suite]`, `r[qual.tiers.benchmark-matrix]`).

use super::{
    ArchetypeFamily, MAX_TIER_LOUDNESS_STEP_LU, MultiRateSegEsrSlice, ONE_DB_POWER_FACTOR,
    SIX_DB_POWER_FACTOR, StemActivityClass, TierQualityThresholds, compute_mr_iir_loss,
    compute_multirate_seg_esr_db, evaluate_besr_from_band_energies, evaluate_ltas_33band_report,
    evaluate_noise_qualified_besr, split_bezout_3band_energies,
};
use crate::targets::CaptureBitDepth;
use dnam_engine::Tier;

/// Classifies a stem output relative to its validated broadband `noise_rms` floor and verifies the
/// muted-output residual rule (`rms_err <= noise_rms`) when genuinely muted, the `+1.0 dB` quiet-tail
/// residual bound when unconfirmed, and bounded residual sanity (`rms_err <= rms_t + n_rms`) when
/// active (`ref[qual.eval.quiet-vs-muted]`; full active-output tier gates run via [`certify_4metric_signal_chain`]).
#[must_use]
pub fn classify_and_verify_quiet_or_muted_with_evidence(
    target: &[f64],
    model: &[f64],
    noise_rms: Option<f64>,
    confirmed_mute_evidence: bool,
) -> (StemActivityClass, bool) {
    let Some(n_rms) = noise_rms else {
        return (StemActivityClass::Unavailable, false);
    };
    if target.is_empty()
        || target.len() != model.len()
        || !n_rms.is_finite()
        || n_rms < 0.0
        || !target.iter().all(|v| v.is_finite())
        || !model.iter().all(|v| v.is_finite())
    {
        return (StemActivityClass::Unavailable, false);
    }

    let n = target.len() as f64;
    let mut sum_sq_t = 0.0_f64;
    let mut sum_sq_err = 0.0_f64;
    let mut nz_t = false;
    let mut nz_err = false;
    for (&yt, &ym) in target.iter().zip(model.iter()) {
        let d = ym - yt;
        sum_sq_t += yt * yt;
        sum_sq_err += d * d;
        nz_t |= yt != 0.0;
        nz_err |= d != 0.0;
    }
    if n_rms == 0.0 && !nz_t {
        return if confirmed_mute_evidence {
            (StemActivityClass::MutedGateClosed, !nz_err)
        } else {
            (StemActivityClass::Unavailable, false)
        };
    }
    if (nz_t && sum_sq_t == 0.0) || (nz_err && sum_sq_err == 0.0) {
        return (StemActivityClass::Unavailable, false);
    }
    let rms_t = (sum_sq_t / n).sqrt();
    let rms_err = (sum_sq_err / n).sqrt();
    if !rms_t.is_finite() || !rms_err.is_finite() {
        return (StemActivityClass::Unavailable, false);
    }

    if n_rms == 0.0 {
        return (
            StemActivityClass::ActiveSignal,
            rms_err.is_finite() && rms_err <= rms_t,
        );
    }

    let six_db_rms_ratio = SIX_DB_POWER_FACTOR.sqrt();
    if rms_t > six_db_rms_ratio * n_rms {
        (
            StemActivityClass::ActiveSignal,
            rms_err.is_finite() && rms_err <= rms_t + n_rms,
        )
    } else if confirmed_mute_evidence {
        (StemActivityClass::MutedGateClosed, rms_err <= n_rms)
    } else {
        (
            StemActivityClass::QuietTail,
            rms_err <= ONE_DB_POWER_FACTOR.sqrt() * n_rms,
        )
    }
}

/// Classifies and verifies a stem given a validated `noise_rms` floor **without** independent mute
/// evidence (`confirmed_mute_evidence = false`, per `r[qual.eval.quiet-vs-muted]`: level alone never
/// establishes a mute). Use [`classify_and_verify_quiet_or_muted_with_evidence`] when independent
/// target-side mute evidence is available.
#[must_use]
pub fn classify_and_verify_quiet_or_muted_stem(
    target: &[f64],
    model: &[f64],
    noise_rms: f64,
) -> (StemActivityClass, bool) {
    classify_and_verify_quiet_or_muted_with_evidence(target, model, Some(noise_rms), false)
}

/// Verifies cross-tier loudness parity (`<= 0.05 LU` step across `Lite`, `Full`, `Pro`, `Ultra`
/// at `StarCenter`, rejecting zero/negative/non-finite RMS levels, `ref[qual.tiers.benchmark-matrix]`).
#[must_use]
pub fn verify_cross_tier_loudness_parity(tier_rms_levels: [f64; 4]) -> (f64, bool) {
    if !tier_rms_levels.iter().all(|&r| r.is_finite() && r > 0.0) {
        return (f64::INFINITY, false);
    }
    let mut max_step_lu = 0.0_f64;
    for t in 0..3 {
        let step_lu = (20.0 * (tier_rms_levels[t + 1] / tier_rms_levels[t]).log10()).abs();
        if step_lu > max_step_lu {
            max_step_lu = step_lu;
        }
    }
    (max_step_lu, max_step_lu <= MAX_TIER_LOUDNESS_STEP_LU)
}

/// Strict monotone non-regression check for raw composite `bESR` across `[Lite, Full, Pro, Ultra]`
/// (`Ultra <= Pro <= Full <= Lite`, zero regression slack, `ref[qual.tiers.benchmark-matrix]`).
#[must_use]
pub fn verify_monotone_tier_besr(tier_besr_db: [f64; 4]) -> bool {
    tier_besr_db
        .iter()
        .all(|v| !v.is_nan() && *v < f64::INFINITY)
        && tier_besr_db[1] <= tier_besr_db[0]
        && tier_besr_db[2] <= tier_besr_db[1]
        && tier_besr_db[3] <= tier_besr_db[2]
}

/// Complete 4-metric perceptual certification summary for a single output channel
/// (`ref[qual.companion.suite]`, `ref[qual.tiers.benchmark-matrix]`).
#[derive(Clone, Debug)]
pub struct PerceptualCertificationSummary {
    pub family: ArchetypeFamily,
    pub tier: Tier,
    pub raw_besr_db: f64,
    pub besr_db: f64,
    pub band_esr_db: [f64; 3],
    pub qualified_band_mask: [bool; 3],
    pub mr_iir: f64,
    pub ltas_max_db: f64,
    pub ltas_unadjusted_max_db: f64,
    pub ltas_excluded_bands: [bool; 33],
    pub ltas_residual_band_power: [f64; 33],
    pub seg_esr_db: f64,
    pub passes_target: bool,
    pub passes_hard_ceiling: bool,
}

/// Verifies full §4 monotone per-output non-regression across `[Lite, Full, Pro, Ultra]`:
/// ordered `[Lite, Full, Pro, Ultra]` tiers, matching `ArchetypeFamily`, finite non-collapsed metrics,
/// and strict zero-slack monotonicity (`higher <= lower`) across raw composite `bESR`, all 3 per-band
/// `bESR`s, `MR-IIR`, max noise-adjusted `LTAS`, excluded-band `LTAS` raw residuals, and `SegESR`
/// (`ref[qual.companion.ltas]`, `ref[qual.tiers.benchmark-matrix]`).
#[must_use]
pub fn verify_monotone_4tier_summaries(tiers: &[PerceptualCertificationSummary; 4]) -> bool {
    const EXPECTED_ORDER: [Tier; 4] = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra];
    for (idx, s) in tiers.iter().enumerate() {
        let valid_db = |x: f64| !x.is_nan() && x < f64::INFINITY;
        let valid_nonneg = |x: f64| x.is_finite() && x >= 0.0;
        if s.tier != EXPECTED_ORDER[idx]
            || s.family != tiers[0].family
            || !valid_db(s.raw_besr_db)
            || !valid_db(s.besr_db)
            || !valid_nonneg(s.mr_iir)
            || !valid_nonneg(s.ltas_max_db)
            || !valid_nonneg(s.ltas_unadjusted_max_db)
            || !valid_db(s.seg_esr_db)
            || !s.band_esr_db.iter().copied().all(valid_db)
            || !s
                .ltas_residual_band_power
                .iter()
                .all(|&p| p.is_finite() && p >= 0.0)
        {
            return false;
        }
    }
    for t in 0..3 {
        let lo = &tiers[t];
        let hi = &tiers[t + 1];
        if hi.raw_besr_db > lo.raw_besr_db
            || hi.mr_iir > lo.mr_iir
            || hi.ltas_max_db > lo.ltas_max_db
            || hi.seg_esr_db > lo.seg_esr_db
            || hi.ltas_excluded_bands != lo.ltas_excluded_bands
        {
            return false;
        }
        for b in 0..3 {
            if hi.band_esr_db[b] > lo.band_esr_db[b] {
                return false;
            }
        }
        for b in 0..33 {
            if hi.ltas_excluded_bands[b]
                && hi.ltas_residual_band_power[b] > lo.ltas_residual_band_power[b]
            {
                return false;
            }
        }
    }
    true
}

/// Evaluates all 4 perceptual metrics (`bESR` composite + per-band, `MR-IIR`, `LTAS`, `SegESR`)
/// against the canonical device-agnostic §4 benchmark matrix (`ref[qual.companion.suite]`,
/// `ref[qual.tiers.benchmark-matrix]`, `ref[qual.besr.finite-timeline]`).
#[must_use]
pub fn certify_4metric_signal_chain(
    family: ArchetypeFamily,
    tier: Tier,
    target: &[f64],
    model: &[f64],
    noise: Option<&[f64]>,
    sample_rate_hz: f64,
    warmup_samples: usize,
) -> PerceptualCertificationSummary {
    let th = TierQualityThresholds::for_tier(tier);
    let (sig_e, err_e, nse_e) =
        split_bezout_3band_energies(target, model, noise, sample_rate_hz, warmup_samples);
    let besr_rep = evaluate_besr_from_band_energies(sig_e, err_e, nse_e);
    let besr_qual =
        evaluate_noise_qualified_besr(&besr_rep, th.per_band_besr_db, th.composite_besr_db);
    let unseen_relax_db = th.unseen_control_besr_db - th.composite_besr_db;
    let unseen_band_goals = th.per_band_besr_db.map(|g| g + unseen_relax_db);
    let unseen_qual =
        evaluate_noise_qualified_besr(&besr_rep, unseen_band_goals, th.unseen_control_besr_db);

    let mr_iir = compute_mr_iir_loss(target, model, noise, sample_rate_hz, warmup_samples);
    let valid_slice = target.len() == model.len()
        && warmup_samples < target.len()
        && !target.is_empty()
        && noise.is_none_or(|ns| ns.len() >= target.len());
    let (tgt_post, mdl_post, nse_post) = if valid_slice {
        (
            &target[warmup_samples..],
            &model[warmup_samples..],
            noise.and_then(|ns| ns.get(warmup_samples..target.len())),
        )
    } else {
        (&[][..], &[][..], None)
    };

    let ltas_rep = evaluate_ltas_33band_report(tgt_post, mdl_post, nse_post, sample_rate_hz, true);
    let (seg_esr_db, seg_active_count, seg_valid) =
        compute_multirate_seg_esr_db(&[MultiRateSegEsrSlice {
            sample_rate_hz,
            target: tgt_post,
            model: mdl_post,
            noise: nse_post,
        }]);

    let companion_ok = mr_iir.is_finite()
        && mr_iir >= 0.0
        && ltas_rep.available
        && !ltas_rep.excluded_band_guard_failed
        && ltas_rep.max_adj_deviation_db <= th.max_ltas_db
        && seg_valid
        && seg_active_count > 0
        && !seg_esr_db.is_nan()
        && seg_esr_db < f64::INFINITY;

    PerceptualCertificationSummary {
        family,
        tier,
        raw_besr_db: besr_rep.besr_db,
        besr_db: besr_qual.qualified_besr_db,
        band_esr_db: besr_rep.band_esr_db,
        qualified_band_mask: besr_qual.qualified_band_mask,
        mr_iir,
        ltas_max_db: ltas_rep.max_adj_deviation_db,
        ltas_unadjusted_max_db: ltas_rep.max_unadjusted_deviation_db,
        ltas_excluded_bands: ltas_rep.excluded_bands,
        ltas_residual_band_power: ltas_rep.residual_band_power,
        seg_esr_db,
        passes_target: besr_qual.passes && companion_ok,
        passes_hard_ceiling: unseen_qual.passes && companion_ok,
    }
}

/// Evaluates all 4 perceptual metrics (`bESR`, `MR-IIR`, `LTAS`, `SegESR`) with a converter-calibrated
/// floor derived from `bit_depth` (`Int12`, `Int16`, `Int24`, `Int32`, `Float32`) and `analog_floor_rms`
/// (`ref[qual.besr.noise-qualified]`, `ref[qual.companion.suite]`).
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn certify_4metric_with_bit_depth(
    family: ArchetypeFamily,
    tier: Tier,
    target: &[f64],
    model: &[f64],
    bit_depth: CaptureBitDepth,
    analog_floor_rms: f64,
    sample_rate_hz: f64,
    warmup_samples: usize,
) -> PerceptualCertificationSummary {
    if !analog_floor_rms.is_finite() || analog_floor_rms < 0.0 {
        return certify_4metric_signal_chain(
            family,
            tier,
            target,
            model,
            None,
            f64::NAN,
            warmup_samples,
        );
    }
    if !bit_depth.is_integer_pcm() && analog_floor_rms == 0.0 {
        return certify_4metric_signal_chain(
            family,
            tier,
            target,
            model,
            None,
            sample_rate_hz,
            warmup_samples,
        );
    }
    let floor_trace = bit_depth.synthesize_undithered_floor_trace(
        target.len(),
        analog_floor_rms,
        0x517C_C1B7_2722_0A95 ^ u64::from(bit_depth.bits()),
    );
    certify_4metric_signal_chain(
        family,
        tier,
        target,
        model,
        Some(&floor_trace),
        sample_rate_hz,
        warmup_samples,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{
        IntervalAuditDecision, compute_per_output_besr, evaluate_joint_energy_interval_audit,
        evaluate_uncertain_noise_interval_bounds,
    };

    #[test]
    fn test_certification_quiet_muted_loudness_and_monotone_guards() {
        assert_eq!(
            classify_and_verify_quiet_or_muted_with_evidence(&[0.0; 8], &[0.0; 8], None, false),
            (StemActivityClass::Unavailable, false)
        );
        assert_eq!(
            classify_and_verify_quiet_or_muted_with_evidence(&[0.0; 8], &[0.0; 8], Some(0.0), true),
            (StemActivityClass::MutedGateClosed, true)
        );
        assert_eq!(
            classify_and_verify_quiet_or_muted_with_evidence(
                &[0.0; 8],
                &[0.0; 8],
                Some(0.0),
                false
            ),
            (StemActivityClass::Unavailable, false)
        );
        assert_eq!(
            classify_and_verify_quiet_or_muted_with_evidence(
                &[0.1; 8],
                &[0.1; 8],
                Some(0.0),
                false
            ),
            (StemActivityClass::ActiveSignal, true)
        );

        let (lu_ok, pass_lu) = verify_cross_tier_loudness_parity([0.5, 0.500_5, 0.501, 0.501_2]);
        assert!(pass_lu && lu_ok <= MAX_TIER_LOUDNESS_STEP_LU);
        assert!(!verify_monotone_tier_besr([-20.0, -18.0, -25.0, -30.0]));

        assert!(
            compute_per_output_besr(&[], &[], None, 48_000.0, 0)
                .1
                .is_infinite()
        );
        let s = [0.1_f64; 64];
        assert!(
            compute_per_output_besr(&[&s], &[&s], Some(&[]), 48_000.0, 0)
                .1
                .is_infinite()
        );

        let bad_iv = evaluate_joint_energy_interval_audit(
            [[1.0, 0.5]; 3],
            [[0.0, 0.0]; 3],
            [[0.0, 0.0]; 3],
            [-20.0; 3],
            -20.0,
        );
        assert_eq!(
            bad_iv.decision,
            IntervalAuditDecision::IndeterminateRequiresQuieterCapture
        );
        let (_, _, dec_pass) = evaluate_uncertain_noise_interval_bounds(
            [1.0, 1.0, 1.0],
            [1.0e-5, 1.0e-5, 1.0e-5],
            [1.0e-6, 1.0e-6, 1.0e-6],
            -30.0,
        );
        assert_eq!(dec_pass, IntervalAuditDecision::CertifiedPass);
        let (_, _, dec_fail) = evaluate_uncertain_noise_interval_bounds(
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
            [1.0e-4, 1.0e-4, 1.0e-4],
            -20.0,
        );
        assert_eq!(dec_fail, IntervalAuditDecision::CertifiedFail);

        let sr = 48_000.0;
        let tgt: Vec<f64> = (0..4_096)
            .map(|i| 0.4 * (2.0 * std::f64::consts::PI * 440.0 * (i as f64) / sr).sin())
            .collect();
        let tiers = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra].map(|tier| {
            let scale = match tier {
                Tier::Lite => 0.96,
                Tier::Full => 0.98,
                Tier::Pro => 0.995,
                Tier::Ultra => 0.999,
            };
            let mdl: Vec<f64> = tgt.iter().map(|&v| v * scale).collect();
            certify_4metric_signal_chain(
                ArchetypeFamily::CleanPedalPreamp,
                tier,
                &tgt,
                &mdl,
                None,
                sr,
                64,
            )
        });
        assert!(verify_monotone_4tier_summaries(&tiers));
        let mut regressed = tiers.clone();
        regressed[3].mr_iir = regressed[0].mr_iir + 0.5;
        assert!(!verify_monotone_4tier_summaries(&regressed));

        // Reject NaN/+inf summaries, reversed tier order, and underflow-erased nonzero mute residuals
        let mut nan_tiers = tiers.clone();
        nan_tiers[2].raw_besr_db = f64::NAN;
        assert!(!verify_monotone_4tier_summaries(&nan_tiers));
        let mut nan_besr = tiers.clone();
        nan_besr[1].besr_db = f64::NAN;
        assert!(!verify_monotone_4tier_summaries(&nan_besr));
        let mut nan_ltas = tiers.clone();
        nan_ltas[1].ltas_unadjusted_max_db = f64::INFINITY;
        assert!(!verify_monotone_4tier_summaries(&nan_ltas));
        let reversed = [
            tiers[3].clone(),
            tiers[2].clone(),
            tiers[1].clone(),
            tiers[0].clone(),
        ];
        assert!(!verify_monotone_4tier_summaries(&reversed));
        assert!(
            crate::metrics::compute_mr_iir_loss(&tgt, &tgt, None, 2_000_000.0, 0).is_infinite()
        );
        let bad_ltas =
            crate::metrics::evaluate_ltas_33band_report(&tgt, &tgt, None, 2_000_000.0, false);
        assert!(!bad_ltas.available && bad_ltas.max_adj_deviation_db.is_infinite());
        let bad_seg =
            crate::metrics::compute_multirate_seg_esr_db(&[crate::metrics::MultiRateSegEsrSlice {
                target: &tgt,
                model: &tgt,
                noise: None,
                sample_rate_hz: 2_000_000.0,
            }]);
        assert_eq!(bad_seg, (f64::INFINITY, 0, false));
        assert_eq!(
            classify_and_verify_quiet_or_muted_with_evidence(
                &[0.0; 8],
                &[1.0e-200; 8],
                Some(0.0),
                true
            ),
            (StemActivityClass::MutedGateClosed, false)
        );
        assert_eq!(
            classify_and_verify_quiet_or_muted_stem(&[0.1; 8], &[10.0; 8], 1.0e-5),
            (StemActivityClass::ActiveSignal, false)
        );
        assert_eq!(
            classify_and_verify_quiet_or_muted_stem(&[1.0e200; 8], &[1.0e200; 8], 1.0e-5),
            (StemActivityClass::Unavailable, false)
        );

        // Unseen-control hard ceiling relaxes per-band goals by +2 dB alongside composite ceiling on Lite:
        // exciting all 33 LTAS 1/3-octave centers with pure phase rotation (-22.5 dB LF, -19.5 dB MF, -14.5 dB HF)
        // preserves 0 dB LTAS power delta while failing captured target and passing unseen hard ceiling.
        let centers = crate::metrics::canonical_ltas_33_centers_hz();
        let mut tgt_33 = vec![0.0_f64; 16_384];
        let mut mdl_unseen = vec![0.0_f64; 16_384];
        for &f_hz in &centers {
            let err_db = if f_hz < 250.0 {
                -22.5_f64
            } else if f_hz < 2_500.0 {
                -19.5_f64
            } else {
                -14.5_f64
            };
            let theta = 2.0 * (0.5 * 10.0_f64.powf(err_db / 20.0)).asin();
            let (c_th, s_th) = (theta.cos(), theta.sin());
            for i in 0..16_384 {
                let ph = 2.0 * std::f64::consts::PI * f_hz * (i as f64) / sr;
                let (s, c) = ph.sin_cos();
                tgt_33[i] += 0.03 * s;
                mdl_unseen[i] += 0.03 * (c_th * s + s_th * c);
            }
        }
        let unseen_cert = certify_4metric_signal_chain(
            ArchetypeFamily::CleanPedalPreamp,
            Tier::Lite,
            &tgt_33,
            &mdl_unseen,
            None,
            sr,
            256,
        );
        assert!(
            !unseen_cert.passes_target && unseen_cert.passes_hard_ceiling,
            "unseen_cert={unseen_cert:?}"
        );
        for bad_floor in [f64::NAN, f64::INFINITY, -1.0e-5] {
            let bad_cert = certify_4metric_with_bit_depth(
                ArchetypeFamily::CleanPedalPreamp,
                Tier::Lite,
                &tgt,
                &tgt,
                CaptureBitDepth::Int24,
                bad_floor,
                sr,
                64,
            );
            assert!(!bad_cert.passes_target && !bad_cert.passes_hard_ceiling);
        }

        // Fail-closed guards on squared-energy overflow, truncated noise, and negative/mismatched monotonicity
        let huge_t = vec![1.0e200_f64; 512];
        let huge_m = vec![1.0e199_f64; 512];
        let fin_n = vec![1.0e-3_f64; 512];
        let ovf_cert = certify_4metric_signal_chain(
            ArchetypeFamily::CleanPedalPreamp,
            Tier::Ultra,
            &huge_t,
            &huge_m,
            Some(&fin_n),
            sr,
            0,
        );
        assert!(!ovf_cert.passes_target && !ovf_cert.passes_hard_ceiling);
        assert!(ovf_cert.mr_iir.is_infinite());
        assert!(ovf_cert.ltas_max_db.is_infinite());
        assert!(ovf_cert.seg_esr_db.is_infinite());

        let short_n = vec![1.0e-5_f64; 16];
        let trunc_cert = certify_4metric_signal_chain(
            ArchetypeFamily::CleanPedalPreamp,
            Tier::Lite,
            &tgt,
            &tgt,
            Some(&short_n),
            sr,
            64,
        );
        assert!(!trunc_cert.passes_target && trunc_cert.ltas_max_db.is_infinite());
        assert!(trunc_cert.seg_esr_db.is_infinite());

        let mut neg_mr = tiers.clone();
        neg_mr[0].mr_iir = -0.1;
        assert!(!verify_monotone_4tier_summaries(&neg_mr));
        let mut neg_ltas = tiers.clone();
        neg_ltas[0].ltas_max_db = f64::NEG_INFINITY;
        assert!(!verify_monotone_4tier_summaries(&neg_ltas));
        let mut excl_mismatch = tiers.clone();
        excl_mismatch[2].ltas_excluded_bands[0] = !excl_mismatch[0].ltas_excluded_bands[0];
        assert!(!verify_monotone_4tier_summaries(&excl_mismatch));

        // Large-energy denom sum without overflow in raw_band_esr_ratio & interval audit extreme goals
        let huge_rep = crate::metrics::evaluate_besr_from_band_energies(
            [1.5e308; 3],
            [1.5e308; 3],
            [1.0e308; 3],
        );
        assert!(huge_rep.valid && (huge_rep.besr_lin - 0.2).abs() < 1.0e-12);
        let iv_rep = crate::metrics::evaluate_joint_energy_interval_audit(
            [[0.0, 0.0]; 3],
            [[0.0, 0.0]; 3],
            [[1.0e-6, 1.0e-6]; 3],
            [3500.0; 3],
            3500.0,
        );
        assert!(iv_rep.composite_c_plus.is_finite());
        assert!(
            (crate::solver::compute_noise_subtracted_besr(1.5e308, 1.5e308, 1.0e308) - 0.2).abs()
                < 1.0e-12
        );
    }

    #[test]
    fn test_companion_multirate_seg_esr_and_ltas_guards() {
        use crate::metrics::{
            MultiRateSegEsrSlice, compute_ltas_33band_max_deviation_db, compute_mr_iir_loss,
            compute_multirate_seg_esr_db, evaluate_ltas_33band_report,
        };
        use std::f64::consts::PI;

        assert_eq!(compute_multirate_seg_esr_db(&[]), (f64::INFINITY, 0, false));
        let silence = [0.0_f64; 1_024];
        let hiss = [1.0e-4_f64; 1_024];
        let loud_err = [1.0e-2_f64; 1_024];
        let seg = |target: &[f64], model: &[f64], noise: Option<&[f64]>| {
            compute_multirate_seg_esr_db(&[MultiRateSegEsrSlice {
                sample_rate_hz: 48_000.0,
                target,
                model,
                noise,
            }])
        };
        // Excluded silent/noise-limited frame residual guard: R > 10^{0.1} F fails closed
        let (db_fail, _, valid_fail) = seg(&silence, &loud_err, Some(&hiss));
        assert!(!valid_fail && db_fail.is_infinite());

        // Noise slice shorter than target fails closed in both SegESR and LTAS
        assert!(
            !evaluate_ltas_33band_report(&hiss, &hiss, Some(&hiss[..16]), 48_000.0, true).available
        );
        assert!(!seg(&hiss, &hiss, Some(&hiss[..16])).2);

        // Underflow-erased nonzero residual on silent frame must fail closed in SegESR
        let mut mixed_tgt = vec![0.0_f64; 2_048];
        for (i, slot) in mixed_tgt[..960].iter_mut().enumerate() {
            *slot = 0.5 * (2.0 * PI * 440.0 * (i as f64) / 48_000.0).sin();
        }
        let mut mixed_mdl = mixed_tgt.clone();
        mixed_mdl[1_000] = 1.0e-200;
        let (db_uf, _, valid_uf) = seg(&mixed_tgt, &mixed_mdl, None);
        assert!(!valid_uf && db_uf.is_infinite());

        // Noiseless LTAS excluded-band guard catches parasitic HF tone despite Hann sidelobe leakage
        let n_ltas = 20_000usize; // also exercises non-multiple-of-hop Welch tail window anchoring
        let pure_440: Vec<f64> = (0..n_ltas)
            .map(|i| 0.5 * (2.0 * PI * 440.0 * (i as f64) / 48_000.0).sin())
            .collect();
        let parasitic_mdl: Vec<f64> = (0..n_ltas)
            .map(|i| pure_440[i] + 0.1 * (2.0 * PI * 15_000.0 * (i as f64) / 48_000.0).sin())
            .collect();
        let rep_p = evaluate_ltas_33band_report(&pure_440, &parasitic_mdl, None, 48_000.0, true);
        assert!(rep_p.excluded_band_guard_failed && rep_p.max_adj_deviation_db.is_infinite());

        // MR-IIR chunked accumulator resists ULP absorption after huge opening transient
        let loud_then_quiet_t: Vec<f64> = (0..24_000)
            .map(|i| {
                let amp = if i < 2_048 { 1.0e3 } else { 1.0e-6 };
                amp * (2.0 * PI * 1_000.0 * (i as f64) / 48_000.0).sin()
            })
            .collect();
        let mr_exact =
            compute_mr_iir_loss(&loud_then_quiet_t, &loud_then_quiet_t, None, 48_000.0, 64);
        assert!(mr_exact.is_finite() && mr_exact <= 1.0e-12);
        // Zero/silent stems & out-of-range sample rates must return +inf on LTAS (never 0.0 dB)
        let (ltas_sil, _) =
            compute_ltas_33band_max_deviation_db(&silence, &silence, None, 48_000.0);
        assert!(ltas_sil.is_infinite());
        assert!(!evaluate_ltas_33band_report(&pure_440, &pure_440, None, 1.0e12, true).available);

        // MR-IIR per-window underflow guards & long autonomous ring-down / DC self-null
        let mut uf_then_burst_t = vec![0.0_f64; 4_096];
        let mut uf_then_burst_m = vec![0.0_f64; 4_096];
        for i in 0..2_048 {
            uf_then_burst_t[i] = 1.0e-200 * (2.0 * PI * 1_000.0 * (i as f64) / 48_000.0).sin();
            uf_then_burst_m[i] = 1.0e-165 * (2.0 * PI * 1_000.0 * (i as f64) / 48_000.0).sin();
        }
        for i in 2_048..4_096 {
            let s = 0.5 * (2.0 * PI * 1_000.0 * (i as f64) / 48_000.0).sin();
            uf_then_burst_t[i] = s;
            uf_then_burst_m[i] = s;
        }
        assert!(
            compute_mr_iir_loss(&uf_then_burst_t, &uf_then_burst_m, None, 48_000.0, 64)
                .is_infinite()
        );
        let sub_sq_t: Vec<f64> = (0..2_048)
            .map(|i| 1.0e-161 * (2.0 * PI * 750.0 * (i as f64) / 48_000.0).sin())
            .collect();
        let zero_2k = vec![0.0_f64; 2_048];
        assert!(compute_mr_iir_loss(&sub_sq_t, &zero_2k, None, 48_000.0, 64).is_infinite());

        let mut long_imp = vec![0.0_f64; 72_000];
        long_imp[0] = 1.0;
        assert_eq!(
            compute_mr_iir_loss(&long_imp, &long_imp, None, 48_000.0, 64),
            0.0
        );
        let long_dc = vec![0.25_f64; 72_000];
        assert_eq!(
            compute_mr_iir_loss(&long_dc, &long_dc, None, 48_000.0, 64),
            0.0
        );

        // Overflowing Bezout 3-band energies (`1e200` -> `+inf`, `1e308` -> `NaN`) must fail closed
        let huge_200 = [1.0e200_f64; 64];
        let huge_308 = [1.0e308_f64; 64];
        let zero_64 = [0.0_f64; 64];
        assert!(
            crate::metrics::try_split_bezout_3band_energies(&huge_200, &zero_64, None, 48_000.0, 8)
                .is_none()
        );
        assert!(
            crate::metrics::try_split_bezout_3band_energies(&huge_308, &zero_64, None, 48_000.0, 8)
                .is_none()
        );
        let (reps_200, worst_200) =
            crate::metrics::compute_per_output_besr(&[&huge_200], &[&zero_64], None, 48_000.0, 8);
        assert!(!reps_200[0].valid && worst_200.is_infinite());
        let cert_200 = certify_4metric_signal_chain(
            ArchetypeFamily::CleanPedalPreamp,
            Tier::Lite,
            &huge_200,
            &huge_200,
            None,
            48_000.0,
            8,
        );
        assert!(!cert_200.passes_target && !cert_200.passes_hard_ceiling);
    }
}
