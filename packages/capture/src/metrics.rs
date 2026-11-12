#![forbid(unsafe_code)]
//! 4-Metric Perceptual Certification Suite (`bESR`, `MR-IIR`, `LTAS`, `SegESR`) & Device-Agnostic
//! 4-Tier Gates (`metrics`).
//!
//! Canonical Reference: `docs/specs/architecture/psychoacoustic_quality_and_tiers.md` (`r[qual.*]`).

pub mod certification;
pub mod companion;

pub use certification::*;
pub use companion::*;

use dnam_engine::Tier;
use std::f64::consts::PI;

/// Worst-band bound offset $10 \log_{10}(3) \approx 4.771212547\text{ dB}$ (`ref[qual.besr.noise-subtracted]`).
pub const BESR_WORST_BAND_OFFSET_DB: f64 = 4.771_212_547_196_624;

/// Maximum allowed loudness step between adjacent tiers at `StarCenter`: `0.05 LU` (`ref[qual.tiers.benchmark-matrix]`).
pub const MAX_TIER_LOUDNESS_STEP_LU: f64 = 0.05;

/// Informational hardware/effect archetype label (never used to relax §4 tier thresholds;
/// DNAM enforces zero subjective carve-outs per `ref[qual.tiers.benchmark-matrix]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArchetypeFamily {
    /// Clean / edge-of-breakup pedals & preamps (`Chain-01`, `Chain-03`, `Chain-05`, `Chain-09`).
    CleanPedalPreamp,
    /// Crunch / high-gain tube amps & bi-amp bass rigs (`Chain-04`, `Chain-06`, `Chain-11`).
    CrunchHighGainAmp,
    /// Full rigs with modulation, BBD/tape delay, FDN shimmer, or stereo routing (`Chain-07`, `Chain-08`, `Chain-12`).
    FullRigModDelayReverb,
    /// Gated fuzz, analog octave dividers, and ring modulators (`Chain-02`, `Chain-10`).
    FuzzOctaveChaos,
}

/// Canonical device-agnostic 4-tier quality benchmark thresholds (`ref[qual.tiers.benchmark-matrix]`).
///
/// Note: `MR-IIR` (`r[qual.companion.mr-iir]`) and `SegESR` (`r[qual.companion.seg-esr]`) are reported
/// companion diagnostics enforced via §4's monotone adjacent-tier and old-stem non-regression rules
/// rather than fixed device-agnostic Benchmark Matrix cutoffs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TierQualityThresholds {
    /// Noise-qualified composite `bESR` acceptance limit in dB.
    pub composite_besr_db: f64,
    /// Per-band `bESR` limits `[LF, MF, HF]` in dB (`or <= NoiseFloor + 1.0 dB`).
    pub per_band_besr_db: [f64; 3],
    /// Maximum absolute noise-adjusted `LTAS` deviation in dB.
    pub max_ltas_db: f64,
    /// Multi-control unseen interpolation `bESR` acceptance limit in dB.
    pub unseen_control_besr_db: f64,
}

impl TierQualityThresholds {
    /// Returns the canonical §4 benchmark thresholds for `tier` (`Lite`, `Full`, `Pro`, `Ultra`).
    #[must_use]
    pub const fn for_tier(tier: Tier) -> Self {
        match tier {
            Tier::Lite => Self {
                composite_besr_db: -18.0,
                per_band_besr_db: [-23.0, -20.0, -15.0],
                max_ltas_db: 0.75,
                unseen_control_besr_db: -16.0,
            },
            Tier::Full => Self {
                composite_besr_db: -23.0,
                per_band_besr_db: [-27.0, -25.0, -20.0],
                max_ltas_db: 0.35,
                unseen_control_besr_db: -21.0,
            },
            Tier::Pro => Self {
                composite_besr_db: -30.0,
                per_band_besr_db: [-34.0, -31.0, -26.0],
                max_ltas_db: 0.15,
                unseen_control_besr_db: -27.0,
            },
            Tier::Ultra => Self {
                composite_besr_db: -34.0,
                per_band_besr_db: [-38.0, -35.0, -30.0],
                max_ltas_db: 0.10,
                unseen_control_besr_db: -32.0,
            },
        }
    }
}

impl ArchetypeFamily {
    /// Device-agnostic `(composite_besr_db, unseen_control_besr_db)` thresholds for `tier` (`ref[qual.tiers.benchmark-matrix]`).
    #[must_use]
    pub const fn besr_thresholds_db(self, tier: Tier) -> (f64, f64) {
        let th = TierQualityThresholds::for_tier(tier);
        (th.composite_besr_db, th.unseen_control_besr_db)
    }

    /// Device-agnostic maximum absolute noise-adjusted `LTAS` threshold in dB for `tier` (`ref[qual.tiers.benchmark-matrix]`).
    #[must_use]
    pub const fn ltas_threshold_db(tier: Tier) -> f64 {
        TierQualityThresholds::for_tier(tier).max_ltas_db
    }
}

/// Stem activity classification (`ref[qual.eval.quiet-vs-muted]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StemActivityClass {
    /// Genuinely muted output (`<= +6.0 dB` above validated floor with independent mute evidence).
    MutedGateClosed,
    /// Active signal below `+6.0 dB` over floor (e.g., quiet tail without mute evidence).
    QuietTail,
    /// Active output (`> +6.0 dB` above validated floor, or positive signal on exact zero floor).
    ActiveSignal,
    /// Missing/invalid noise evidence or empty/non-finite selection (`r[qual.eval.quiet-vs-muted]`).
    Unavailable,
}

/// Uncertain noise-floor interval certification status (`ref[qual.besr.uncertain-floor]`, `ref[qual.besr.joint-intervals]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntervalAuditDecision {
    /// Upper bound $c_+ \le \tau$ and all per-band upper bounds pass (`Guaranteed Pass`).
    CertifiedPass,
    /// Lower bound $c_- > \tau$ or any per-band lower bound fails (`Guaranteed Failure`).
    CertifiedFail,
    /// Bounds straddle a band or composite threshold (`Unavailable / Ambiguous`).
    IndeterminateRequiresQuieterCapture,
}

/// Per-band (`LF`, `MF`, `HF`) and composite `bESR` evaluation report (`ref[qual.besr.noise-subtracted]`,
/// `ref[qual.besr.zero-denominator]`, `ref[qual.besr.noise-qualified]`).
#[derive(Clone, Debug)]
pub struct BesrReport {
    /// Whether the inputs and floor energies were finite, non-negative, excited (`>= 1` active band),
    /// and free of zero-denominator collapse (`denominator_collapse` distinguishes `+inf` failure from unexcited `[0; 3]`).
    pub valid: bool,
    /// Raw target band energies $Y_{b,c} = \|H_b y_c\|_2^2$.
    pub sig_energy: [f64; 3],
    /// Raw residual band energies $R_{b,c} = \|H_b (\hat{y}_c - y_c)\|_2^2$.
    pub err_energy: [f64; 3],
    /// Selection-matched expected noise band energies $F_{b,c} = T \epsilon_{\text{floor},b,c}^2$.
    pub noise_energy: [f64; 3],
    /// Raw noise-subtracted linear `ESR_{b,c}` (`max(0, R - F) / (Y + F)`).
    pub band_esr_lin: [f64; 3],
    /// Raw noise-subtracted `ESR_{b,c}` in dB (`-inf` for `0`, `+inf` for zero-denominator collapse).
    pub band_esr_db: [f64; 3],
    /// Active band mask (`Y_b + F_b > 0`).
    pub active_bands: [bool; 3],
    /// Whether any zero-denominator band (`Y_b + F_b == 0`) had positive residual error `R_b > 0`.
    pub denominator_collapse: [bool; 3],
    /// Raw equal-weight 3-band linear average `(ESR_LF + ESR_MF + ESR_HF) / 3`.
    pub besr_lin: f64,
    /// Raw composite `bESR` in dB ($10 \log_{10}(\text{bESR}_{\text{lin}})$).
    pub besr_db: f64,
    /// Worst single-band raw `ESR` in dB (guaranteed $\le \text{bESR}_{\text{dB}} + 4.7712125\text{ dB}$ when finite).
    pub worst_band_db: f64,
}

/// Tier-specific noise-qualified `bESR` acceptance report (`ref[qual.besr.noise-qualified]`).
#[derive(Clone, Debug)]
pub struct NoiseQualifiedBesrEvaluation {
    /// Which bands qualified for the 3-condition noise-limited exception.
    pub qualified_band_mask: [bool; 3],
    /// Per-band pass status (either `ESR_b <= 10^{g_{b,t}/10}` or noise-qualified).
    pub band_passes: [bool; 3],
    /// Noise-qualified linear composite (qualifying bands zeroed, divisor strictly **3**).
    pub qualified_besr_lin: f64,
    /// Noise-qualified composite in dB.
    pub qualified_besr_db: f64,
    /// Overall tier `bESR` pass (`valid && all band_passes && qualified_besr_lin <= 10^{comp/10}`).
    pub passes: bool,
}

// ============================================================================
// 1. Exact Degree-7 Binomial / Bézout Crossover (`r[qual.besr.bezout-crossover]`)
// ============================================================================

#[inline]
fn eval_bezout_lowpass_complex(freq_hz: f64, cutoff_hz: f64, sample_rate_hz: f64) -> (f64, f64) {
    let r = (-2.0 * PI * cutoff_hz / sample_rate_hz).exp();
    let omega = 2.0 * PI * freq_hz / sample_rate_hz;
    let (sin_w, cos_w) = omega.sin_cos();
    // H_1(e^{jw}) = (1 - r) / (1 - r e^{-jw}) = (1 - r) / ((1 - r cos w) + j r sin w)
    let den_re = 1.0 - r * cos_w;
    let den_im = r * sin_w;
    let den_norm = den_re * den_re + den_im * den_im;
    let num = 1.0 - r;
    let h_re = num * den_re / den_norm;
    let h_im = -num * den_im / den_norm;

    let c_mul = |(ar, ai): (f64, f64), (br, bi): (f64, f64)| (ar * br - ai * bi, ar * bi + ai * br);
    let h2 = c_mul((h_re, h_im), (h_re, h_im));
    let h4 = c_mul(h2, h2);
    let h5 = c_mul(h4, (h_re, h_im));
    let h6 = c_mul(h5, (h_re, h_im));
    let h7 = c_mul(h6, (h_re, h_im));

    (
        35.0 * h4.0 - 84.0 * h5.0 + 70.0 * h6.0 - 20.0 * h7.0,
        35.0 * h4.1 - 84.0 * h5.1 + 70.0 * h6.1 - 20.0 * h7.1,
    )
}

/// Evaluates the exact complex frequency responses `[H_LF(e^{jω}), H_MF(e^{jω}), H_HF(e^{jω})]`
/// of the causal degree-7 Binomial/Bézout crossover $\mathcal{L}_{f_c}(z) = 35 H_1^4 - 84 H_1^5 + 70 H_1^6 - 20 H_1^7$
/// at `freq_hz` and `sample_rate_hz` (`48 kHz` canonical, satisfying $H_{\text{LF}} + H_{\text{MF}} + H_{\text{HF}} \equiv 1$,
/// `ref[qual.besr.bezout-crossover]`, `ref[qual.besr.filter-response]`).
#[must_use]
pub fn eval_bezout_3band_complex(freq_hz: f64, sample_rate_hz: f64) -> [(f64, f64); 3] {
    let fs = if (8_000.0..=1_536_000.0).contains(&sample_rate_hz) {
        sample_rate_hz
    } else {
        48_000.0
    };
    let f = if freq_hz.is_finite() {
        freq_hz.abs().clamp(0.0, 0.5 * fs)
    } else {
        0.0
    };
    let l_250 = eval_bezout_lowpass_complex(f, 250.0, fs);
    let l_2500 = eval_bezout_lowpass_complex(f, 2_500.0, fs);
    let h_lf = l_250;
    let h_mf = (l_2500.0 - l_250.0, l_2500.1 - l_250.1);
    let h_hf = (1.0 - l_2500.0, -l_2500.1);
    [h_lf, h_mf, h_hf]
}

/// Evaluates the real-part coherent partition `[Re(H_LF), Re(H_MF), Re(H_HF)]` (which sums identically to `1.0`
/// at every frequency) and individual squared gains via [`eval_bezout_3band_complex`] at `48 kHz`.
#[must_use]
pub fn eval_bezout_3band_weights(freq_hz: f64) -> [f64; 3] {
    let [lf, mf, hf] = eval_bezout_3band_complex(freq_hz, 48_000.0);
    [lf.0, mf.0, hf.0]
}

#[inline]
fn step_bezout7_lp(x: f64, r: f64, one_minus_r: f64, u: &mut [f64; 7]) -> f64 {
    let mut prev = x;
    for stage in u.iter_mut() {
        *stage = one_minus_r * prev + r * *stage;
        prev = *stage;
    }
    35.0 * u[3] - 84.0 * u[4] + 70.0 * u[5] - 20.0 * u[6]
}

/// Splits a finite-timeline signal into 3 exact complementary Binomial/Bézout bands (`LF`, `MF`, `HF`),
/// stepping zero-initialized filters across the **entire** aligned timeline `0..n_total` before accumulating
/// energies over `warmup_samples..n_total` (`ref[qual.besr.bezout-crossover]`, `ref[qual.besr.finite-timeline]`).
///
/// Returns `None` if inputs are empty, mismatched in length, non-finite, or `warmup_samples >= len`.
#[must_use]
pub fn try_split_bezout_3band_energies(
    target: &[f64],
    model: &[f64],
    noise: Option<&[f64]>,
    sample_rate_hz: f64,
    warmup_samples: usize,
) -> Option<([f64; 3], [f64; 3], [f64; 3])> {
    if !(8_000.0..=1_536_000.0).contains(&sample_rate_hz)
        || target.is_empty()
        || target.len() != model.len()
        || warmup_samples >= target.len()
        || !target.iter().all(|v| v.is_finite())
        || !model.iter().all(|v| v.is_finite())
    {
        return None;
    }
    let n_total = target.len();
    if let Some(ns) = noise
        && (ns.len() < n_total || !ns[..n_total].iter().all(|v| v.is_finite()))
    {
        return None;
    }

    let r1 = (-2.0 * PI * 250.0 / sample_rate_hz).exp();
    let omr1 = 1.0 - r1;
    let r2 = (-2.0 * PI * 2_500.0 / sample_rate_hz).exp();
    let omr2 = 1.0 - r2;

    let (mut u1_t, mut u2_t) = ([0.0_f64; 7], [0.0_f64; 7]);
    let (mut u1_e, mut u2_e) = ([0.0_f64; 7], [0.0_f64; 7]);
    let (mut u1_n, mut u2_n) = ([0.0_f64; 7], [0.0_f64; 7]);

    let mut sig_e = [0.0_f64; 3];
    let mut err_e = [0.0_f64; 3];
    let mut nse_e = [0.0_f64; 3];
    let mut nz_sig = [false; 3];
    let mut nz_err = [false; 3];
    let mut nz_nse = [false; 3];

    for i in 0..n_total {
        let y = target[i];
        let e = model[i] - y;
        let eta = noise.map_or(0.0, |ns| ns[i]);

        let l250_t = step_bezout7_lp(y, r1, omr1, &mut u1_t);
        let l2500_t = step_bezout7_lp(y, r2, omr2, &mut u2_t);
        let (lf_t, mf_t, hf_t) = (l250_t, l2500_t - l250_t, y - l2500_t);

        let l250_e = step_bezout7_lp(e, r1, omr1, &mut u1_e);
        let l2500_e = step_bezout7_lp(e, r2, omr2, &mut u2_e);
        let (lf_e, mf_e, hf_e) = (l250_e, l2500_e - l250_e, e - l2500_e);

        let l250_n = step_bezout7_lp(eta, r1, omr1, &mut u1_n);
        let l2500_n = step_bezout7_lp(eta, r2, omr2, &mut u2_n);
        let (lf_n, mf_n, hf_n) = (l250_n, l2500_n - l250_n, eta - l2500_n);

        if i >= warmup_samples {
            sig_e[0] += lf_t * lf_t;
            sig_e[1] += mf_t * mf_t;
            sig_e[2] += hf_t * hf_t;
            nz_sig[0] |= lf_t != 0.0;
            nz_sig[1] |= mf_t != 0.0;
            nz_sig[2] |= hf_t != 0.0;

            err_e[0] += lf_e * lf_e;
            err_e[1] += mf_e * mf_e;
            err_e[2] += hf_e * hf_e;
            nz_err[0] |= lf_e != 0.0;
            nz_err[1] |= mf_e != 0.0;
            nz_err[2] |= hf_e != 0.0;

            nse_e[0] += lf_n * lf_n;
            nse_e[1] += mf_n * mf_n;
            nse_e[2] += hf_n * hf_n;
            nz_nse[0] |= lf_n != 0.0;
            nz_nse[1] |= mf_n != 0.0;
            nz_nse[2] |= hf_n != 0.0;
        }
    }

    // Reject overflow/NaN (`+inf`/`NaN`) and IEEE-754 underflow-erased nonzero energy
    // (`0 < |x| < 1.5e-154`) as unavailable (`r[qual.besr.zero-denominator]`)
    for b in 0..3 {
        if !sig_e[b].is_finite()
            || !err_e[b].is_finite()
            || !nse_e[b].is_finite()
            || (nz_sig[b] && sig_e[b] == 0.0)
            || (nz_err[b] && err_e[b] == 0.0)
            || (nz_nse[b] && nse_e[b] == 0.0)
        {
            return None;
        }
    }

    Some((sig_e, err_e, nse_e))
}

/// Convenience wrapper around [`try_split_bezout_3band_energies`] returning `[f64::NAN; 3]` on invalid inputs
/// so downstream [`evaluate_besr_from_band_energies`] marks the report `valid = false`.
#[must_use]
pub fn split_bezout_3band_energies(
    target: &[f64],
    model: &[f64],
    noise: Option<&[f64]>,
    sample_rate_hz: f64,
    warmup_samples: usize,
) -> ([f64; 3], [f64; 3], [f64; 3]) {
    try_split_bezout_3band_energies(target, model, noise, sample_rate_hz, warmup_samples)
        .unwrap_or(([f64::NAN; 3], [f64::NAN; 3], [f64::NAN; 3]))
}

// ============================================================================
// 2. Sum-Denominator bESR, Zero-Denominator & Noise-Qualified Acceptance
// ============================================================================

#[inline]
fn raw_band_esr_ratio(y: f64, r: f64, f: f64) -> (f64, bool) {
    if !y.is_finite() || !r.is_finite() || !f.is_finite() || y < 0.0 || r < 0.0 || f < 0.0 {
        return (f64::INFINITY, true);
    }
    let denom = y + f;
    if denom.is_finite() {
        if denom > 0.0 {
            ((r - f).max(0.0) / denom, false)
        } else if r == 0.0 {
            (0.0, false)
        } else {
            (f64::INFINITY, true)
        }
    } else {
        let half_denom = 0.5 * y + 0.5 * f;
        ((0.5 * r - 0.5 * f).max(0.0) / half_denom, false)
    }
}

/// Computes `BesrReport` using the exact §1.2 sum-denominator formula
/// $\text{ESR}_{b,c} = \max(0, R_{b,c} - F_{b,c}) / (Y_{b,c} + F_{b,c})$
/// (`ref[qual.besr.noise-subtracted]`, `ref[qual.besr.zero-denominator]`).
#[must_use]
pub fn evaluate_besr_from_band_energies(
    sig_energy: [f64; 3],
    err_energy: [f64; 3],
    noise_energy: [f64; 3],
) -> BesrReport {
    let inputs_ok = sig_energy.iter().all(|&v| v.is_finite() && v >= 0.0)
        && err_energy.iter().all(|&v| v.is_finite() && v >= 0.0)
        && noise_energy.iter().all(|&v| v.is_finite() && v >= 0.0);

    if !inputs_ok {
        return BesrReport {
            valid: false,
            sig_energy,
            err_energy,
            noise_energy,
            band_esr_lin: [f64::INFINITY; 3],
            band_esr_db: [f64::INFINITY; 3],
            active_bands: [false; 3],
            denominator_collapse: [true; 3],
            besr_lin: f64::INFINITY,
            besr_db: f64::INFINITY,
            worst_band_db: f64::INFINITY,
        };
    }

    let mut band_esr_lin = [0.0_f64; 3];
    let mut band_esr_db = [f64::NEG_INFINITY; 3];
    let mut active_bands = [false; 3];
    let mut denominator_collapse = [false; 3];

    for b in 0..3 {
        let (y, r, f) = (sig_energy[b], err_energy[b], noise_energy[b]);
        active_bands[b] = y > 0.0 || f > 0.0;
        let (esr, collapsed) = raw_band_esr_ratio(y, r, f);
        denominator_collapse[b] = collapsed;
        band_esr_lin[b] = esr;
        band_esr_db[b] = if esr == 0.0 {
            f64::NEG_INFINITY
        } else if esr.is_infinite() {
            f64::INFINITY
        } else {
            10.0 * esr.log10()
        };
    }

    let besr_lin = (band_esr_lin[0] + band_esr_lin[1] + band_esr_lin[2]) / 3.0;
    let besr_db = if besr_lin == 0.0 {
        f64::NEG_INFINITY
    } else if besr_lin.is_infinite() {
        f64::INFINITY
    } else {
        10.0 * besr_lin.log10()
    };
    let worst_band_db = band_esr_db[0].max(band_esr_db[1]).max(band_esr_db[2]);

    BesrReport {
        valid: !denominator_collapse.iter().any(|&c| c) && active_bands.iter().any(|&a| a),
        sig_energy,
        err_energy,
        noise_energy,
        band_esr_lin,
        band_esr_db,
        active_bands,
        denominator_collapse,
        besr_lin,
        besr_db,
        worst_band_db,
    }
}

/// Evaluates §1.3 noise-limited per-band and composite qualification against `per_band_goals_db`
/// and `composite_goal_db` (`ref[qual.besr.noise-qualified]`).
#[must_use]
pub fn evaluate_noise_qualified_besr(
    report: &BesrReport,
    per_band_goals_db: [f64; 3],
    composite_goal_db: f64,
) -> NoiseQualifiedBesrEvaluation {
    if !report.valid
        || !per_band_goals_db.iter().all(|g| g.is_finite())
        || !composite_goal_db.is_finite()
    {
        return NoiseQualifiedBesrEvaluation {
            qualified_band_mask: [false; 3],
            band_passes: [false; 3],
            qualified_besr_lin: f64::INFINITY,
            qualified_besr_db: f64::INFINITY,
            passes: false,
        };
    }

    let mut qualified_band_mask = [false; 3];
    let mut band_passes = [false; 3];
    let mut qual_sum = 0.0_f64;

    for b in 0..3 {
        let g_db = per_band_goals_db[b];
        let g_lin = 10.0_f64.powf(g_db / 10.0);
        let (y, r, f) = (
            report.sig_energy[b],
            report.err_energy[b],
            report.noise_energy[b],
        );

        let qualifies = f > 0.0
            && report.band_esr_lin[b].is_finite()
            && snr_below_neg_goal_db(y, f, g_db, 0.0)
            && r <= ONE_DB_POWER_FACTOR * f;

        if qualifies {
            qualified_band_mask[b] = true;
            band_passes[b] = true;
        } else {
            band_passes[b] = report.band_esr_lin[b].is_finite() && report.band_esr_lin[b] <= g_lin;
            qual_sum += report.band_esr_lin[b];
        }
    }

    // Divisor is strictly 3 (never |non-qualifying|) per r[qual.besr.noise-qualified]
    let qualified_besr_lin = qual_sum / 3.0;
    let qualified_besr_db = if qualified_besr_lin == 0.0 {
        f64::NEG_INFINITY
    } else {
        10.0 * qualified_besr_lin.log10()
    };
    let comp_tau = 10.0_f64.powf(composite_goal_db / 10.0);
    let passes = band_passes.iter().all(|&p| p)
        && qualified_besr_lin.is_finite()
        && qualified_besr_lin <= comp_tau;

    NoiseQualifiedBesrEvaluation {
        qualified_band_mask,
        band_passes,
        qualified_besr_lin,
        qualified_besr_db,
        passes,
    }
}

/// Evaluates whether a `BesrReport` passes a composite threshold `threshold_db` (using uniform
/// per-band goals equal to `threshold_db`; for full 4-tier per-band goals use [`evaluate_noise_qualified_besr`]).
#[must_use]
pub fn check_noise_qualified_besr_acceptance(report: &BesrReport, threshold_db: f64) -> bool {
    evaluate_noise_qualified_besr(report, [threshold_db; 3], threshold_db).passes
}

/// Evaluates per-channel independent `bESR` without cross-channel pooling (`ref[qual.besr.per-output]`).
/// Rejects empty or mismatched channel counts by returning `worst_db = +inf`.
#[must_use]
pub fn compute_per_output_besr(
    target_channels: &[&[f64]],
    model_channels: &[&[f64]],
    noise_channels: Option<&[&[f64]]>,
    sample_rate_hz: f64,
    warmup_samples: usize,
) -> (Vec<BesrReport>, f64) {
    if target_channels.is_empty() || target_channels.len() != model_channels.len() {
        return (Vec::new(), f64::INFINITY);
    }
    if let Some(nc) = noise_channels
        && nc.len() != target_channels.len()
    {
        return (Vec::new(), f64::INFINITY);
    }

    let c_out = target_channels.len();
    let mut reports = Vec::with_capacity(c_out);
    let mut worst_db = f64::NEG_INFINITY;
    for c in 0..c_out {
        let nse = noise_channels.map(|nc| nc[c]);
        let (sig_e, err_e, nse_e) = split_bezout_3band_energies(
            target_channels[c],
            model_channels[c],
            nse,
            sample_rate_hz,
            warmup_samples,
        );
        let rep = evaluate_besr_from_band_energies(sig_e, err_e, nse_e);
        if !rep.valid {
            worst_db = f64::INFINITY;
        } else if rep.besr_db > worst_db {
            worst_db = rep.besr_db;
        }
        reports.push(rep);
    }
    (reports, worst_db)
}

// ============================================================================
// 3. Exact 5-Row Rectangular Joint Energy-Interval Audit (`r[qual.besr.joint-intervals]`)
// ============================================================================

/// Detailed rectangular interval audit bounds (`[r_-, r_+]`, `[c_-, c_+]`, and decision).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointIntervalAuditReport {
    pub raw_bounds_lin: [[f64; 2]; 3],
    pub qualified_bounds_lin: [[f64; 2]; 3],
    pub composite_c_minus: f64,
    pub composite_c_plus: f64,
    pub decision: IntervalAuditDecision,
}

#[inline]
fn snr_below_neg_goal_db(y: f64, f: f64, goal_db: f64, slack_db: f64) -> bool {
    if f <= 0.0 {
        return false;
    }
    if y <= 0.0 {
        return true;
    }
    let req_snr = 10.0_f64.powf(-goal_db / 10.0);
    let ratio = y / f;
    if req_snr >= f64::MIN_POSITIVE
        && req_snr.is_finite()
        && ratio >= f64::MIN_POSITIVE
        && ratio.is_finite()
    {
        ratio < req_snr
    } else {
        10.0 * (y.log10() - f.log10()) < -goal_db + slack_db
    }
}

/// Evaluates the exact §1.5 5-row rectangular joint-interval bounds for $Y \in [Y_-, Y_+]$,
/// $R \in [R_-, R_+]$, and $F \in [F_-, F_+]$ against `per_band_goals_db` and `composite_goal_db`
/// (`ref[qual.besr.uncertain-floor]`, `ref[qual.besr.joint-intervals]`).
#[must_use]
pub fn evaluate_joint_energy_interval_audit(
    y_intervals: [[f64; 2]; 3],
    r_intervals: [[f64; 2]; 3],
    f_intervals: [[f64; 2]; 3],
    per_band_goals_db: [f64; 3],
    composite_goal_db: f64,
) -> JointIntervalAuditReport {
    let valid_iv = |iv: &[[f64; 2]; 3]| {
        iv.iter()
            .all(|&[lo, hi]| lo.is_finite() && hi.is_finite() && lo >= 0.0 && lo <= hi)
    };
    if !valid_iv(&y_intervals)
        || !valid_iv(&r_intervals)
        || !valid_iv(&f_intervals)
        || !per_band_goals_db.iter().all(|g| g.is_finite())
        || !composite_goal_db.is_finite()
    {
        return JointIntervalAuditReport {
            raw_bounds_lin: [[0.0, f64::INFINITY]; 3],
            qualified_bounds_lin: [[0.0, f64::INFINITY]; 3],
            composite_c_minus: 0.0,
            composite_c_plus: f64::INFINITY,
            decision: IntervalAuditDecision::IndeterminateRequiresQuieterCapture,
        };
    }

    let q = ONE_DB_POWER_FACTOR;
    let qual_c = |y: f64, r: f64, f: f64, g_db: f64, slack_db: f64| -> f64 {
        if f > 0.0 && r <= q * f && snr_below_neg_goal_db(y, f, g_db, slack_db) {
            0.0
        } else {
            raw_band_esr_ratio(y, r, f).0
        }
    };

    let mut raw_bounds_lin = [[0.0_f64; 2]; 3];
    let mut qualified_bounds_lin = [[0.0_f64; 2]; 3];
    let mut all_bands_pass_upper = true;
    let mut any_band_fails_lower = false;

    for b in 0..3 {
        let [y_lo, y_hi] = y_intervals[b];
        let [r_lo, r_hi] = r_intervals[b];
        let [f_lo, f_hi] = f_intervals[b];
        let g_db = per_band_goals_db[b];
        let g = 10.0_f64.powf(g_db / 10.0);
        let req_snr = 10.0_f64.powf(-g_db / 10.0);

        let r_minus = raw_band_esr_ratio(y_hi, r_lo, f_hi).0;
        let r_plus = raw_band_esr_ratio(y_lo, r_hi, f_lo).0;
        raw_bounds_lin[b] = [r_minus, r_plus];

        // Upper bound c_+ at (F_-, R_+) — Rows 1..3 of spec table
        let raw_c_plus = if f_lo == 0.0 || r_hi > q * f_lo {
            qual_c(y_lo, r_hi, f_lo, g_db, -1.0e-11)
        } else if snr_below_neg_goal_db(y_hi, f_lo, g_db, -1.0e-11) {
            0.0
        } else {
            let mut y_req = f_lo * req_snr;
            if y_req.is_finite()
                && y_req > 0.0
                && f64::from_bits(y_req.to_bits() - 1) / f_lo >= req_snr
            {
                y_req = f64::from_bits(y_req.to_bits() - 1);
            }
            let y_eff = if y_req.is_finite() {
                y_lo.max(y_req)
            } else {
                y_lo
            };
            raw_band_esr_ratio(y_eff, r_hi, f_lo).0
        };

        // Lower bound c_- at (F_+, R_-) — Rows 4..5 of spec table
        let c_minus =
            if f_hi > 0.0 && r_lo <= q * f_hi && snr_below_neg_goal_db(y_lo, f_hi, g_db, 1.0e-11) {
                0.0
            } else {
                qual_c(y_hi, r_lo, f_hi, g_db, 1.0e-11)
            };
        let c_plus = raw_c_plus.max(c_minus);

        qualified_bounds_lin[b] = [c_minus, c_plus];
        if !c_plus.is_finite() || c_plus > g {
            all_bands_pass_upper = false;
        }
        if !c_minus.is_finite() || c_minus > g {
            any_band_fails_lower = true;
        }
    }

    let comp_c_minus =
        (qualified_bounds_lin[0][0] + qualified_bounds_lin[1][0] + qualified_bounds_lin[2][0])
            / 3.0;
    let comp_c_plus =
        (qualified_bounds_lin[0][1] + qualified_bounds_lin[1][1] + qualified_bounds_lin[2][1])
            / 3.0;
    let tau = 10.0_f64.powf(composite_goal_db / 10.0);
    let comp_fails_lower = !comp_c_minus.is_finite() || comp_c_minus > tau;
    let comp_passes_upper = comp_c_plus.is_finite() && comp_c_plus <= tau;

    let any_band_excited = (0..3).any(|b| y_intervals[b][1] > 0.0 || f_intervals[b][1] > 0.0);
    let decision = if !any_band_excited {
        if any_band_fails_lower || comp_fails_lower {
            IntervalAuditDecision::CertifiedFail
        } else {
            IntervalAuditDecision::IndeterminateRequiresQuieterCapture
        }
    } else if all_bands_pass_upper && comp_passes_upper {
        IntervalAuditDecision::CertifiedPass
    } else if any_band_fails_lower || comp_fails_lower {
        IntervalAuditDecision::CertifiedFail
    } else {
        IntervalAuditDecision::IndeterminateRequiresQuieterCapture
    };

    JointIntervalAuditReport {
        raw_bounds_lin,
        qualified_bounds_lin,
        composite_c_minus: comp_c_minus,
        composite_c_plus: comp_c_plus,
        decision,
    }
}

/// Evaluates uncertain floor bounds `[0, noise_floor_upper[b]]` for fixed observed `(Y_b, R_b)`
/// (`ref[qual.besr.uncertain-floor]`).
#[must_use]
pub fn evaluate_uncertain_noise_interval_bounds(
    noisy_sig_energy: [f64; 3],
    noisy_err_energy: [f64; 3],
    noise_floor_upper: [f64; 3],
    threshold_db: f64,
) -> (f64, f64, IntervalAuditDecision) {
    let y_iv = noisy_sig_energy.map(|y| [y, y]);
    let r_iv = noisy_err_energy.map(|r| [r, r]);
    let f_iv = noise_floor_upper.map(|f_hi| [0.0, f_hi]);
    let rep =
        evaluate_joint_energy_interval_audit(y_iv, r_iv, f_iv, [threshold_db; 3], threshold_db);
    (rep.composite_c_minus, rep.composite_c_plus, rep.decision)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bezout7_exact_complex_sum_crossover_gain_and_finite_timeline() {
        for fs in [44_100.0, 48_000.0, 88_200.0, 96_000.0] {
            for f in [0.0, 20.0, 250.0, 1_000.0, 2_500.0, 10_000.0, 20_000.0] {
                let [lf, mf, hf] = eval_bezout_3band_complex(f, fs);
                assert!((lf.0 + mf.0 + hf.0 - 1.0).abs() < 1.0e-13);
                assert!((lf.1 + mf.1 + hf.1).abs() < 1.0e-13);
            }
        }
        // Verify §1.1 warning box: at fs = 48 kHz and f = 250 Hz, |H_LF| ≈ +9.33 dB
        let [lf_250, _, _] = eval_bezout_3band_complex(250.0, 48_000.0);
        let gain_db_250 = 10.0 * (lf_250.0 * lf_250.0 + lf_250.1 * lf_250.1).log10();
        assert!((gain_db_250 - 9.33).abs() < 0.05);

        // Verify r[qual.besr.finite-timeline]: filtering 0..N before warmup != resetting at warmup
        let sig: Vec<f64> = (0..1_024).map(|i| (0.05 * (i as f64)).sin()).collect();
        let mdl: Vec<f64> = sig.iter().map(|&v| 0.98 * v).collect();
        let full_tl = try_split_bezout_3band_energies(&sig, &mdl, None, 48_000.0, 256).unwrap();
        let reset_at_boundary =
            try_split_bezout_3band_energies(&sig[256..], &mdl[256..], None, 48_000.0, 0).unwrap();
        assert_ne!(full_tl.0, reset_at_boundary.0);
    }

    #[test]
    fn test_four_metrics_edge_cases_and_tier_gates() {
        let sr = 48_000.0;
        let n = 8_192;
        let target: Vec<f64> = (0..n)
            .map(|i| {
                0.4 * (2.0 * PI * 110.0 * (i as f64) / sr).sin()
                    + 0.3 * (2.0 * PI * 880.0 * (i as f64) / sr).cos()
                    + 0.2 * (2.0 * PI * 4_400.0 * (i as f64) / sr).sin()
            })
            .collect();
        let model: Vec<f64> = target.iter().map(|&x| x * 0.9995).collect();

        let summary = certify_4metric_signal_chain(
            ArchetypeFamily::CleanPedalPreamp,
            Tier::Ultra,
            &target,
            &model,
            None,
            sr,
            128,
        );
        assert!(summary.passes_target && summary.passes_hard_ceiling);
        assert!(summary.raw_besr_db < -60.0);
        assert!(summary.mr_iir < 0.01);

        // Invalid / empty / zero-prediction edge cases fail-closed (+inf / false)
        assert!(compute_mr_iir_loss(&[], &[], None, sr, 0).is_infinite());
        assert!(compute_mr_iir_loss(&target[..256], &vec![0.0; 256], None, sr, 0).is_infinite());
        assert!(
            compute_ltas_33band_max_deviation_db(&[], &[], None, sr)
                .0
                .is_infinite()
        );
        assert!(
            compute_ltas_33band_max_deviation_db(&target, &model, None, 32_000.0)
                .0
                .is_infinite()
        );
        assert!(
            evaluate_ltas_33band_report(&target, &model, None, 32_000.0, false).available,
            "device-rate LTAS helper should evaluate sub-40 kHz devices below Nyquist"
        );
        // Partial < 20 ms frame is retained (weighted by exact duration), never silently -100 dB!
        let (partial_db, partial_cnt) =
            compute_seg_esr_20ms_db(&target[..32], &model[..32], None, sr);
        assert_eq!(partial_cnt, 1);
        assert!(partial_db < -60.0 && partial_db > -70.0);

        assert_eq!(
            classify_and_verify_quiet_or_muted_stem(&[], &[], 1.0e-5),
            (StemActivityClass::Unavailable, false)
        );
        assert!(!verify_cross_tier_loudness_parity([0.0, 0.0, 0.0, 0.0]).1);

        // Bezout weights, fallback sample rate, zero-denominator collapse, and per-output bESR
        let w = eval_bezout_3band_weights(1_000.0);
        assert!((w[0] + w[1] + w[2] - 1.0).abs() < 1.0e-12);
        let fallback_c = eval_bezout_3band_complex(1_000.0, 0.0);
        assert!((fallback_c[0].0 + fallback_c[1].0 + fallback_c[2].0 - 1.0).abs() < 1.0e-12);

        let collapse_rep = evaluate_besr_from_band_energies([0.0; 3], [1.0, 0.0, 0.0], [0.0; 3]);
        assert!(!collapse_rep.valid && collapse_rep.denominator_collapse[0]);
        assert!(!evaluate_noise_qualified_besr(&collapse_rep, [-20.0; 3], -20.0).passes);

        let exact_zero_rep = evaluate_besr_from_band_energies([1.0; 3], [0.0; 3], [0.0; 3]);
        assert!(exact_zero_rep.valid && exact_zero_rep.besr_db.is_infinite());
        assert!(check_noise_qualified_besr_acceptance(
            &exact_zero_rep,
            -30.0
        ));

        let (ch_reps, worst_2ch) =
            compute_per_output_besr(&[&target, &target], &[&model, &model], None, sr, 64);
        assert_eq!(ch_reps.len(), 2);
        assert!(worst_2ch < -60.0);

        // TierQualityThresholds + ArchetypeFamily helpers + NaN energy fail-closed guard
        for tier in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
            let (comp_db, unseen_db) = ArchetypeFamily::CrunchHighGainAmp.besr_thresholds_db(tier);
            let ltas_th = ArchetypeFamily::ltas_threshold_db(tier);
            assert!(comp_db < 0.0 && unseen_db < 0.0);
            assert!(ltas_th > 0.0);
        }
        let nan_rep = evaluate_besr_from_band_energies([f64::NAN, 1.0, 1.0], [0.0; 3], [0.0; 3]);
        assert!(!nan_rep.valid && nan_rep.besr_db.is_infinite());
    }
}
