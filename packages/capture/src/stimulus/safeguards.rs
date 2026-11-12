#![forbid(unsafe_code)]
//! Pre-Synthesis Hardware Safeguard `Laws 0.A–0.F` & Structured [`CaptureWarning`] Diagnostics
//! (`stimulus::safeguards`).
//!
//! Canonical Reference: `docs/specs/training/capture_and_alignment.md`
//! (`ref[train.capture.invariants]`, `ref[train.capture.calib.clean-pad]`).

use std::collections::BTreeSet;

use super::fft::{
    ADAPTIVE_FRAME_SAMPLES, ADC_PEAK_GOVERNOR_FS, CYCLIC_PREFIX_SAMPLES, CYCLIC_WINDOW_SAMPLES,
    DisjointCombSchedule, rfft_bins,
};

/// Structured non-blocking hardware safeguard diagnostics emitted during adaptive capture.
#[derive(Debug, Clone, PartialEq)]
pub enum CaptureWarning {
    /// `Law 0.F`: Sub-`0 dBFS` microphone capsule, ribbon, or external mic-preamp rail clipping
    /// (`r[train.capture.safeguards.mic-capsule-clip]`).
    MicrophoneCapsuleOrPreampClip {
        /// Stem identifier where sub-`0 dBFS` capsule/preamp clipping was detected.
        stem_id: String,
        /// Estimated sub-`0 dBFS` rail ceiling in dBFS (e.g. `-9.5 dBFS`).
        ceiling_dbfs: f32,
        /// Post-cabinet `8–20 kHz` stopband harmonic excess in dB (`10 * log10(Gamma_post_cab)`).
        post_cab_excess_db: f32,
        /// True when non-stop soft recovery (`0.80 * A_rail` clamp or `0 dB` sub-frame mask) ran.
        soft_recovery_applied: bool,
        /// True when a clean companion mic channel (`1->2`/`2->2`) cross-calibrated Stage II.
        dual_mic_cross_calibrated: bool,
    },
    /// `Law 0.D`: `>= 3` consecutive ADC samples clipped at `>= 0.999` FS; excitation stepped down `-6 dB`.
    AdcHardClipStepDown {
        /// Stem identifier.
        stem_id: String,
        /// Zero-based frame index where ADC hard clipping occurred.
        frame_index: usize,
        /// Longest run of consecutive `>= 0.999` FS samples.
        consecutive_clipped_samples: usize,
        /// Updated excitation attenuation in dB.
        new_gain_scale_db: f32,
    },
    /// `Law 0.C`: `96 ms` frame discarded (`w_frame = 0`) and queued for replay due to USB dropout,
    /// clock slip (`|delta_tau| > 0.25` samples), or impulsive guard-bin crackle (`crest > 8.0`).
    ClockSlipOrUsbDropout {
        /// Stem identifier.
        stem_id: String,
        /// Discarded frame index.
        frame_index: usize,
        /// Estimated phase-delay jump in samples relative to prior frame.
        delta_lag_samples: f32,
        /// Impulsive crest factor measured on unexcited guard bins.
        guard_crest: f32,
    },
    /// `Law 0.C`: Narrowband parasitic oscillation / idle squeal masked on guard bins ($\kappa_{\text{spec}} > 10$).
    IdleSquealBinsMasked {
        /// Stem identifier.
        stem_id: String,
        /// Total count of DFT bins masked out of comb schedule and residual loss.
        masked_bin_count: usize,
    },
    /// `Law 0.E`: Sub-harmonic flip-flop pitch-divider chatter (`OC-2`) detected; polyphonic multisine
    /// capped at `2` frames (`192 ms`) and handed off to `A3` monophonic glissando + `A5` coreset.
    SubHarmonicDividerHandoff {
        /// Stem identifier.
        stem_id: String,
        /// Frame index triggering handoff (`0` or `1`).
        frame_index: usize,
        /// Sub-harmonic (`f/2`, `f/4`) energy ratio in dB (`> -20 dB`).
        sub_harmonic_ratio_db: f32,
        /// Suppressed fundamental linear coherence $\gamma_{xy}^2 < 0.70$.
        coherence_sq: f32,
    },
    /// `Law 0.D`: Thermal bias/gain drift detected across fixed reference anchors $\Omega_{\text{ref}}$.
    ThermalBiasDriftDetected {
        /// Stem identifier.
        stem_id: String,
        /// Anchor magnitude drift relative to Frame `0` in dB.
        anchor_drift_db: f32,
    },
    /// `Law 0.A`: Autonomous LFO or input-envelope-driven resonant SVF detected.
    LtvModulationDetected {
        /// Stem identifier.
        stem_id: String,
        /// Detected LTV regime (`LfoModulated` or `EnvelopeSvf`).
        regime: LtvRegime,
        /// Spectral centroid or carrier variation metric.
        modulation_metric: f32,
    },
}

/// `Law 0.A` linear time-varying (`LTV`) / envelope-modulated classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LtvRegime {
    /// Stationary time-invariant response ($\rho_{\text{LTV}}^{(1)} \le 0.02$).
    TimeInvariant,
    /// Autonomous LFO chorus/phaser/vibrato/rotary ($\rho_{\text{LTV}}^{(1)} > 0.02$ or $\eta_{\text{Bessel}} > 10^{-3}$).
    LfoModulated,
    /// Input-envelope-driven resonant SVF / auto-wah (triggers 2-level `-18 / -3 dBFS` pole-zoom).
    EnvelopeSvf,
}

/// Persistent safeguard state tracked across `96 ms` frames within a stem.
#[derive(Debug, Clone)]
pub struct SafeguardState {
    /// Stem identifier for warning attribution.
    pub stem_id: String,
    /// True when sweeping an output/master volume control (`Law 0.F` Case B).
    pub is_master_volume_sweep: bool,
    /// Calibrated noise floor power $\epsilon_{\text{floor}}^2$.
    pub noise_floor_power: f64,
    /// Active LTV regime (`Law 0.A`).
    pub ltv_regime: LtvRegime,
    /// True if `Law 0.B` detected long echo/reverb tail ($\hat{T}_{\text{mem}} > 9.33\text{ ms}$) requiring 2-frame hold.
    pub long_memory_2frame_hold: bool,
    /// Estimated causal memory decay tail in ms (`Law 0.B`).
    pub estimated_memory_ms: f64,
    /// Masked narrowband idle-squeal DFT bins (`Law 0.C`).
    pub squeal_mask_bins: BTreeSet<usize>,
    /// Cumulative multiplicative excitation scale (`<= 1.0`, `Law 0.D` & `Law 0.F`).
    pub excitation_gain_scale: f64,
    /// True when `Law 0.E` capped polyphonic multisine and handed off to `A3` glissando.
    pub oc2_glissando_handoff: bool,
    /// Sub-frame quadrature weights `[w_-18dB, w_-6dB, w_0dB]` (`[1.0, 1.0, 0.0]` under `Law 0.F` Case B).
    pub law4c_subframe_weights: [f32; 3],
    /// Measured Frame-0 carrier complex magnitudes on identical schedule (`Law 0.A`).
    pub frame0_carrier_mags: Option<Vec<f64>>,
    /// Measured Frame-0 dry RMS and normalized transfer `[bin, |H[k]|]` (`Law 0.A` `EnvelopeSvf`).
    pub frame0_transfer_ref: Option<(f64, Vec<(usize, f64)>)>,
    /// Measured Frame-0 reference anchor RMS (`Law 0.D`).
    pub frame0_anchor_rms: Option<f64>,
    /// Previous frame carrier phase slope (lag in samples) for `Law 0.C` clock-slip check.
    pub prev_phase_lag_samples: Option<f64>,
    /// Previous `4,096`-sample wet window for `Law 0.B` causal echo deflation.
    pub prev_wet_window: Option<Vec<f32>>,
    /// Whether `MicrophoneCapsuleOrPreampClip` has already been emitted for this stem.
    pub mic_clip_emitted: bool,
    /// Accumulated warnings for this stem.
    pub warnings: Vec<CaptureWarning>,
}

impl SafeguardState {
    /// Initializes safeguard state for `stem_id`.
    #[must_use]
    pub fn new(stem_id: String, is_master_volume_sweep: bool, noise_floor_dbfs: f32) -> Self {
        let floor_db = if noise_floor_dbfs.is_finite() {
            f64::from(noise_floor_dbfs).clamp(-160.0, -20.0)
        } else {
            -100.0
        };
        Self {
            stem_id,
            is_master_volume_sweep,
            noise_floor_power: 10.0_f64.powf(floor_db / 10.0),
            ltv_regime: LtvRegime::TimeInvariant,
            long_memory_2frame_hold: false,
            estimated_memory_ms: 2.0,
            squeal_mask_bins: BTreeSet::new(),
            excitation_gain_scale: 1.0,
            oc2_glissando_handoff: false,
            law4c_subframe_weights: [1.0, 1.0, 1.0],
            frame0_carrier_mags: None,
            frame0_transfer_ref: None,
            frame0_anchor_rms: None,
            prev_phase_lag_samples: None,
            prev_wet_window: None,
            mic_clip_emitted: false,
            warnings: Vec::new(),
        }
    }
}

/// Per-frame evaluation result of `Laws 0.A–0.F`.
#[derive(Debug, Clone, PartialEq)]
pub struct SafeguardFrameDecision {
    /// `Law 0.C` / `Law 0.D`: If `true`, discard this frame (`w_frame = 0`) and replay it immediately.
    pub discard_and_replay: bool,
    /// Echo-deflated `4,096`-sample wet analysis window (`Law 0.B`).
    pub clean_analysis_window: Vec<f32>,
    /// Measured reference-anchor thermal drift in dB (`Law 0.D`).
    pub anchor_drift_db: f32,
    /// Post-cabinet `8–20 kHz` stopband excess ratio $\Gamma_{\text{post-cab}}^{(m)}$ (`Law 0.F`).
    pub gamma_post_cab: f64,
    /// Sub-`0 dBFS` rail plateau concentration $\kappa_{\text{rail}}^{(m)}$ (`Law 0.F`).
    pub rail_plateau_kappa: f64,
}

fn longest_clip_run(samples: &[f32], thresh: f32) -> usize {
    let mut best = 0usize;
    let mut cur = 0usize;
    for &s in samples {
        if s.is_finite() && s.abs() >= thresh {
            cur += 1;
            best = best.max(cur);
        } else {
            cur = 0;
        }
    }
    best
}

fn detect_sub_0dbfs_rail_plateau(samples: &[f32]) -> (usize, f64, f32) {
    if samples.len() < 8 {
        return (0, 0.0, 0.0);
    }
    let mut max_pos = 0.0_f32;
    let mut max_neg = 0.0_f32;
    for &s in samples {
        if s.is_finite() {
            if s > max_pos {
                max_pos = s;
            } else if -s > max_neg {
                max_neg = -s;
            }
        }
    }
    let rail = max_pos.max(max_neg);
    if !(0.06..0.990).contains(&rail) {
        return (0, 0.0, rail);
    }

    let pos_band = 0.988 * max_pos;
    let neg_band = 0.988 * max_neg;
    let eps_flat = (5.0e-4 * rail).max(4.0e-5);
    let mut longest_flat = 0usize;
    let mut cur_flat = 1usize;
    let mut plateau_samples = 0usize;

    for i in 1..samples.len() {
        let y = samples[i];
        let yp = samples[i - 1];
        let near_pos = max_pos >= 0.06 && y >= pos_band && yp >= pos_band;
        let near_neg = max_neg >= 0.06 && (-y) >= neg_band && (-yp) >= neg_band;
        if (near_pos || near_neg) && (y - yp).abs() <= eps_flat {
            cur_flat += 1;
            plateau_samples += 1;
            longest_flat = longest_flat.max(cur_flat);
        } else {
            cur_flat = 1;
        }
    }
    let kappa = (plateau_samples as f64) / (samples.len() as f64);
    (longest_flat, kappa, rail)
}

fn carrier_transfer_centroid_and_peak(transfer: &[(usize, f64)]) -> (f64, usize, f64) {
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    let mut best_bin = 0usize;
    let mut best_h = 0.0_f64;
    for &(b, h) in transfer {
        let w = h * h;
        num += (b.max(1) as f64).log2() * w;
        den += w;
        if h > best_h {
            best_h = h;
            best_bin = b;
        }
    }
    let mut max_prom = 1.0_f64;
    for i in 1..transfer.len().saturating_sub(1) {
        let neigh = 0.5 * (transfer[i - 1].1 + transfer[i + 1].1).max(1.0e-6);
        max_prom = max_prom.max(transfer[i].1 / neigh);
    }
    (num / den.max(1.0e-18), best_bin, max_prom)
}

fn estimate_carrier_phase_lag_samples(
    dry_win: &[f32],
    wet_win: &[f32],
    carriers: &[usize],
) -> (f64, f64) {
    let x_spec = rfft_bins(dry_win);
    let y_spec = rfft_bins(wet_win);
    // Estimate delay from unwrapped phase differences across adjacent carriers
    let mut sum_dphi = 0.0_f64;
    let mut sum_dk = 0.0_f64;
    let mut sum_num_coh = 0.0_f64;
    let mut sum_xx = 0.0_f64;
    let mut sum_yy = 0.0_f64;
    let mut prev: Option<(usize, f64)> = None;

    for &k in carriers {
        if k >= x_spec.len() {
            continue;
        }
        let xk = x_spec[k];
        let yk = y_spec[k];
        let x_p = xk.norm_sqr();
        let y_p = yk.norm_sqr();
        if x_p <= 1.0e-12 || y_p <= 1.0e-14 {
            continue;
        }
        // H[k] = Y[k] * conj(X[k]) / |X[k]|^2
        let h_re = yk.re * xk.re + yk.im * xk.im;
        let h_im = yk.im * xk.re - yk.re * xk.im;
        let phi = h_im.atan2(h_re);
        sum_num_coh += (h_re * h_re + h_im * h_im).sqrt();
        sum_xx += x_p;
        sum_yy += y_p;

        if let Some((pk, p_phi)) = prev {
            let dk = (k - pk) as f64;
            if dk <= 68.0 {
                // Wrap phase difference into [-pi, pi]
                let mut dphi = phi - p_phi;
                while dphi > std::f64::consts::PI {
                    dphi -= 2.0 * std::f64::consts::PI;
                }
                while dphi < -std::f64::consts::PI {
                    dphi += 2.0 * std::f64::consts::PI;
                }
                sum_dphi += -dphi * dk;
                sum_dk += dk * dk;
            }
        }
        prev = Some((k, phi));
    }

    let lag = if sum_dk > 0.0 {
        (sum_dphi / sum_dk) * (CYCLIC_WINDOW_SAMPLES as f64) / (2.0 * std::f64::consts::PI)
    } else {
        0.0
    };
    let coh_sq = if sum_xx > 1.0e-15 && sum_yy > 1.0e-15 {
        ((sum_num_coh * sum_num_coh) / (sum_xx * sum_yy)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (lag, coh_sq)
}

/// Evaluates pre-synthesis hardware safeguard `Laws 0.A–0.F` on a newly captured `96 ms` (`4,608`-sample)
/// wet frame (`wet_l`, optional stereo `wet_r`) against its transmitted `dry_l` and `schedule`.
pub fn evaluate_frame_safeguards(
    state: &mut SafeguardState,
    frame_index: usize,
    dry_l: &[f32],
    wet_l: &[f32],
    wet_r: Option<&[f32]>,
    schedule: &DisjointCombSchedule,
) -> SafeguardFrameDecision {
    let win_start = wet_l.len().saturating_sub(CYCLIC_WINDOW_SAMPLES);
    let raw_win_l = &wet_l[win_start..];
    let dry_win_l = &dry_l[dry_l.len().saturating_sub(CYCLIC_WINDOW_SAMPLES)..];

    // --- Law 0.D (Part 1): Hard ADC Clipping (>= 3 consecutive samples >= 0.999 FS) ---
    let adc_run_l = longest_clip_run(wet_l, 0.999);
    let adc_run_r = wet_r.map_or(0, |r| longest_clip_run(r, 0.999));
    let adc_run = adc_run_l.max(adc_run_r);
    if adc_run >= 3 {
        state.excitation_gain_scale = (state.excitation_gain_scale * 0.5).max(1.0 / 64.0);
        let scale_db = (20.0 * state.excitation_gain_scale.log10()) as f32;
        state.warnings.push(CaptureWarning::AdcHardClipStepDown {
            stem_id: state.stem_id.clone(),
            frame_index,
            consecutive_clipped_samples: adc_run,
            new_gain_scale_db: scale_db,
        });
        return SafeguardFrameDecision {
            discard_and_replay: true,
            clean_analysis_window: raw_win_l.to_vec(),
            anchor_drift_db: 0.0,
            gamma_post_cab: 0.0,
            rail_plateau_kappa: 0.0,
        };
    }

    if raw_win_l.len() < CYCLIC_WINDOW_SAMPLES || dry_win_l.len() < CYCLIC_WINDOW_SAMPLES {
        return SafeguardFrameDecision {
            discard_and_replay: false,
            clean_analysis_window: raw_win_l.to_vec(),
            anchor_drift_db: 0.0,
            gamma_post_cab: 0.0,
            rail_plateau_kappa: 0.0,
        };
    }

    // --- Law 0.B: Causal Known-History Echo Deflation & 2-Frame Hold ---
    let mut clean_win = raw_win_l.to_vec();
    if wet_l.len() >= ADAPTIVE_FRAME_SAMPLES {
        // Compare cyclic prefix arrival [64..256] against periodic wrap [4096+64..4096+256]
        let mut diff_e = 0.0_f64;
        let mut sig_e = 0.0_f64;
        for i in 64..256 {
            let a = f64::from(wet_l[i]);
            let b = f64::from(wet_l[CYCLIC_WINDOW_SAMPLES + i]);
            let d = b - a;
            diff_e += d * d;
            sig_e += b * b;
        }
        let wrap_mismatch = diff_e / (sig_e + 1.0e-12);
        if wrap_mismatch > 0.015 {
            state.estimated_memory_ms = state.estimated_memory_ms.max(18.0);
            state.long_memory_2frame_hold = true;
            if let Some(ref prev_w) = state.prev_wet_window
                && prev_w.len() == CYCLIC_WINDOW_SAMPLES
            {
                // Subtract 1-tap frame-to-frame carryover alpha * prev_w
                let mut dot = 0.0_f64;
                let mut ep = 0.0_f64;
                for i in 0..512 {
                    let cur_tail =
                        f64::from(wet_l[CYCLIC_WINDOW_SAMPLES + i]) - f64::from(wet_l[i]);
                    let pw = f64::from(prev_w[CYCLIC_WINDOW_SAMPLES - 512 + i]);
                    dot += cur_tail * pw;
                    ep += pw * pw;
                }
                if ep > 1.0e-9 {
                    let alpha = (dot / ep).clamp(-0.85, 0.85);
                    if alpha.abs() > 0.02 {
                        for (i, slot) in clean_win.iter_mut().enumerate() {
                            let decay = (-((i as f64) / 2048.0)).exp();
                            *slot =
                                (f64::from(*slot) - alpha * decay * f64::from(prev_w[i])) as f32;
                        }
                    }
                }
            }
        }
    }
    state.prev_wet_window = Some(raw_win_l.to_vec());

    let y_spec = rfft_bins(&clean_win);
    let inv_n = 2.0 / (CYCLIC_WINDOW_SAMPLES as f64);

    // --- Law 0.C: Idle-Squeal Kurtosis Mask & 96 ms Clock-Slip / Crackle Replay ---
    if !schedule.guard_bins.is_empty() {
        let guard_powers: Vec<(usize, f64)> = schedule
            .guard_bins
            .iter()
            .filter_map(|&b| y_spec.get(b).map(|c| (b, c.norm_sqr() * inv_n * inv_n)))
            .collect();
        if guard_powers.len() >= 8 {
            let mut sorted_p: Vec<f64> = guard_powers.iter().map(|&(_, p)| p).collect();
            sorted_p.sort_by(|a, b| a.total_cmp(b));
            // Trim top 25% so a single parasitic squeal bin cannot inflate the background floor
            let trim_len = (sorted_p.len() * 3) / 4;
            let bg_mean = sorted_p[..trim_len].iter().sum::<f64>() / (trim_len.max(1) as f64);
            let mean_p = sorted_p.iter().sum::<f64>() / (sorted_p.len() as f64);
            let m4 = sorted_p.iter().map(|&p| (p - mean_p).powi(4)).sum::<f64>()
                / (sorted_p.len() as f64);
            let var = sorted_p.iter().map(|&p| (p - mean_p).powi(2)).sum::<f64>()
                / (sorted_p.len() as f64);
            let spec_kurtosis = if var > 1.0e-24 { m4 / (var * var) } else { 0.0 };

            if spec_kurtosis > 10.0 && bg_mean > 0.0 {
                let mut newly_masked = false;
                for &(b, p) in &guard_powers {
                    if p > 25.0 * bg_mean.max(state.noise_floor_power) {
                        for nb in b.saturating_sub(1)..=(b + 1) {
                            if state.squeal_mask_bins.insert(nb) {
                                newly_masked = true;
                            }
                        }
                    }
                }
                if newly_masked {
                    state.warnings.push(CaptureWarning::IdleSquealBinsMasked {
                        stem_id: state.stem_id.clone(),
                        masked_bin_count: state.squeal_mask_bins.len(),
                    });
                }
            }

            // Guard-bin bleed check for Law 0.B
            let carrier_mean = schedule
                .carrier_bins
                .iter()
                .filter_map(|&b| y_spec.get(b).map(|c| c.norm_sqr() * inv_n * inv_n))
                .sum::<f64>()
                / (schedule.carrier_bins.len().max(1) as f64);
            if carrier_mean > 1.0e-12 && (bg_mean / carrier_mean) > 1.0e-3 {
                state.long_memory_2frame_hold = true;
            }
        }
    }

    let (cur_lag, coh_sq) =
        estimate_carrier_phase_lag_samples(dry_win_l, &clean_win, &schedule.carrier_bins);

    // --- Law 0.E: Sub-Harmonic Flip-Flop Divider (`OC-2`) Chatter Detector ---
    // Evaluated prior to clock-slip replay so chaotic flip-flop zero-crossing phase chatter on Frame 0/1
    // hands off to `A3` monophonic glissando instead of falsely triggering USB clock-slip replay.
    if frame_index <= 1 && !state.oc2_glissando_handoff {
        let first_carr = schedule.carrier_bins.first().copied().unwrap_or(23);
        let ped = schedule.pedestal_bin.unwrap_or(0);
        // Bins 3..first_carr-2 excluding pedestal harmonics and 2nd/3rd-order Volterra difference
        // products (first_carr - ped, first_carr - 2*ped) are strictly forbidden to memoryless
        // nonlinearities (since MIN_COMB_BIN_STRIDE = 34 > first_carr), uniquely isolating f/2 & f/4 dividers!
        let diff1 = first_carr.saturating_sub(ped);
        let diff2 = first_carr.saturating_sub(2 * ped);
        let mut sub_e = 0.0_f64;
        for (b, c) in y_spec
            .iter()
            .enumerate()
            .take(first_carr.saturating_sub(2))
            .skip(3)
        {
            if b != ped && b != 2 * ped && b != diff1 && b != diff2 {
                sub_e += c.norm_sqr();
            }
        }
        let total_spec_e: f64 = y_spec.iter().skip(2).map(|c| c.norm_sqr()).sum();
        let sub_carr_e: f64 = schedule
            .carrier_bins
            .iter()
            .filter_map(|&b| y_spec.get(b).map(|c| c.norm_sqr()))
            .sum();
        let sub_vs_total_db = 10.0 * (sub_e / total_spec_e.max(1.0e-18)).log10();
        let sub_vs_carr_db = 10.0 * (sub_e / sub_carr_e.max(1.0e-18)).log10();
        if (sub_vs_total_db > -25.0 || (sub_vs_carr_db > -18.0 && coh_sq < 0.85))
            && total_spec_e * inv_n * inv_n > 1.0e-7
        {
            state.oc2_glissando_handoff = true;
            state
                .warnings
                .push(CaptureWarning::SubHarmonicDividerHandoff {
                    stem_id: state.stem_id.clone(),
                    frame_index,
                    sub_harmonic_ratio_db: sub_vs_carr_db as f32,
                    coherence_sq: coh_sq as f32,
                });
        }
    }

    // Impulsive crackle / dropout check via sample-to-sample second-difference spike vs RMS
    let win_rms = (clean_win
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        / (clean_win.len() as f64))
        .sqrt();
    let max_d2 = (2..clean_win.len()).fold(0.0_f64, |acc, i| {
        let d2 = (f64::from(clean_win[i]) - 2.0 * f64::from(clean_win[i - 1])
            + f64::from(clean_win[i - 2]))
        .abs();
        acc.max(d2)
    });
    let guard_crest = if win_rms > 1.0e-6 {
        max_d2 / (4.0 * win_rms)
    } else {
        0.0
    };
    let dropout_run = if win_rms > 0.01 {
        let quiet_thr = (0.02 * win_rms) as f32;
        let mut best = 0usize;
        let mut cur = 0usize;
        for &s in &clean_win {
            if s.abs() <= quiet_thr {
                cur += 1;
                best = best.max(cur);
            } else {
                cur = 0;
            }
        }
        best
    } else {
        0
    };

    let dry_rms = (dry_win_l
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        / (dry_win_l.len().max(1) as f64))
        .sqrt();
    let same_rms_as_ref = state
        .frame0_transfer_ref
        .as_ref()
        .is_none_or(|(r0, _)| (dry_rms / r0.max(1.0e-9) - 1.0).abs() < 0.20);

    if !state.oc2_glissando_handoff {
        let phase_slip = same_rms_as_ref
            && state.prev_phase_lag_samples.is_some_and(|prev_lag| {
                coh_sq >= 0.65
                    && (cur_lag - prev_lag).abs() > 0.25
                    && state.ltv_regime == LtvRegime::TimeInvariant
            });
        if phase_slip || guard_crest > 8.0 || dropout_run >= 48 {
            let delta_lag = state
                .prev_phase_lag_samples
                .map_or(dropout_run as f64, |prev| (cur_lag - prev).abs());
            state.prev_phase_lag_samples = Some(cur_lag);
            state.warnings.push(CaptureWarning::ClockSlipOrUsbDropout {
                stem_id: state.stem_id.clone(),
                frame_index,
                delta_lag_samples: delta_lag as f32,
                guard_crest: guard_crest as f32,
            });
            return SafeguardFrameDecision {
                discard_and_replay: true,
                clean_analysis_window: clean_win,
                anchor_drift_db: 0.0,
                gamma_post_cab: 0.0,
                rail_plateau_kappa: 0.0,
            };
        }
    }
    if coh_sq >= 0.65 && !state.oc2_glissando_handoff && same_rms_as_ref {
        state.prev_phase_lag_samples = Some(cur_lag);
    }

    // --- Law 0.A: Two-Frame LTV / Bessel Sideband & Envelope-SVF Detector ---
    let x_spec_l = rfft_bins(dry_win_l);
    let cur_mags: Vec<f64> = schedule
        .carrier_bins
        .iter()
        .map(|&b| y_spec.get(b).map_or(0.0, |c| c.abs() * inv_n))
        .collect();
    let max_xm = schedule
        .carrier_bins
        .iter()
        .filter_map(|&b| x_spec_l.get(b).map(|c| c.abs() * inv_n))
        .fold(0.0_f64, f64::max);
    let xm_floor = (0.04 * max_xm).max(2.0e-4);
    let cur_transfer: Vec<(usize, f64)> = schedule
        .carrier_bins
        .iter()
        .filter_map(|&b| {
            let xm = x_spec_l.get(b)?.abs() * inv_n;
            let ym = y_spec.get(b)?.abs() * inv_n;
            (xm >= xm_floor).then_some((b, ym / xm))
        })
        .collect();

    let mut bessel_e = 0.0_f64;
    let mut carr_e = 0.0_f64;
    for &b in &schedule.carrier_bins {
        if b + 1 < y_spec.len() && b >= 1 {
            carr_e += y_spec[b].norm_sqr();
            // +/-1 adjacent even bins capture autonomous LFO FM/AM Bessel skirts while avoiding
            // odd Volterra pedestal sidebands (`b +/- 2` when `PEDESTAL_BIN_MONO == 2`).
            bessel_e += y_spec[b - 1].norm_sqr() + y_spec[b + 1].norm_sqr();
        }
    }
    let eta_bessel = bessel_e / carr_e.max(1.0e-18);
    let prev_ltv = state.ltv_regime;
    let mut ltv_metric = 0.0_f32;
    if state.frame0_transfer_ref.is_none() {
        state.frame0_carrier_mags = Some(cur_mags);
        if dry_rms > 1.0e-4 && cur_transfer.len() * 4 >= schedule.carrier_bins.len() * 3 {
            state.frame0_transfer_ref = Some((dry_rms, cur_transfer));
        }
    } else if let Some((r0, ref h0)) = state.frame0_transfer_ref
        && (dry_rms / r0.max(1.0e-9) - 1.0).abs() < 0.20
        && cur_transfer.len() * 4 >= schedule.carrier_bins.len() * 3
    {
        if state.ltv_regime == LtvRegime::TimeInvariant {
            let h0_map: std::collections::BTreeMap<usize, f64> = h0.iter().copied().collect();
            let mut diff_sq = 0.0_f64;
            let mut ref_sq = 0.0_f64;
            for &(b, hm) in &cur_transfer {
                if let Some(&h_ref) = h0_map.get(&b) {
                    diff_sq += (hm - h_ref).powi(2);
                    ref_sq += h_ref * h_ref;
                }
            }
            let rho_ltv = if ref_sq > 1.0e-18 {
                (diff_sq / ref_sq).sqrt()
            } else {
                0.0
            };
            if rho_ltv > 0.06 || eta_bessel > 3.5e-2 {
                state.ltv_regime = LtvRegime::LfoModulated;
                ltv_metric = rho_ltv.max(eta_bessel) as f32;
            }
        }
    } else if state.ltv_regime != LtvRegime::EnvelopeSvf {
        if let Some((r0, ref h0)) = state.frame0_transfer_ref
            && dry_rms > 1.0e-4
            && !(0.75..=1.33).contains(&(dry_rms / r0.max(1.0e-9)))
            && cur_transfer.len() * 4 >= schedule.carrier_bins.len() * 3
            && h0.len() * 4 >= schedule.carrier_bins.len() * 3
        {
            let (c0, pk0, prom0) = carrier_transfer_centroid_and_peak(h0);
            let (cm, pkm, prom_m) = carrier_transfer_centroid_and_peak(&cur_transfer);
            let centroid_shift_oct = (cm - c0).abs();
            let peak_bin_shift = pkm.abs_diff(pk0);
            if prom0.max(prom_m) > 2.8 && (centroid_shift_oct > 0.06 || peak_bin_shift >= 34) {
                state.ltv_regime = LtvRegime::EnvelopeSvf;
                ltv_metric = centroid_shift_oct as f32;
            }
        } else if state.ltv_regime == LtvRegime::TimeInvariant && eta_bessel > 3.5e-2 {
            state.ltv_regime = LtvRegime::LfoModulated;
            ltv_metric = eta_bessel as f32;
        }
    }
    if state.ltv_regime != prev_ltv
        && state.ltv_regime != LtvRegime::TimeInvariant
        && !state.warnings.iter().any(|w| {
            matches!(
                w,
                CaptureWarning::LtvModulationDetected { regime, .. } if *regime == state.ltv_regime
            )
        })
    {
        state.warnings.push(CaptureWarning::LtvModulationDetected {
            stem_id: state.stem_id.clone(),
            regime: state.ltv_regime,
            modulation_metric: ltv_metric,
        });
    }

    // --- Law 0.D (Part 2): Reference-Anchor Thermal Drift Tracker ---
    let anchor_rms = if schedule.reference_anchor_bins.is_empty() {
        0.0
    } else {
        let sum_sq: f64 = schedule
            .reference_anchor_bins
            .iter()
            .filter_map(|&b| y_spec.get(b).map(|c| (c.abs() * inv_n).powi(2)))
            .sum();
        (sum_sq / (schedule.reference_anchor_bins.len() as f64)).sqrt()
    };
    let mut anchor_drift_db = 0.0_f32;
    if anchor_rms > 1.0e-7 {
        if let Some(ref0) = state.frame0_anchor_rms {
            anchor_drift_db = (20.0 * (anchor_rms / ref0.max(1.0e-12)).log10()) as f32;
            if anchor_drift_db.abs() > 0.35
                && !state
                    .warnings
                    .iter()
                    .any(|w| matches!(w, CaptureWarning::ThermalBiasDriftDetected { .. }))
            {
                state
                    .warnings
                    .push(CaptureWarning::ThermalBiasDriftDetected {
                        stem_id: state.stem_id.clone(),
                        anchor_drift_db,
                    });
            }
        } else {
            state.frame0_anchor_rms = Some(anchor_rms);
        }
    }

    let _ = carr_e;

    // --- Law 0.F: Sub-0 dBFS Microphone Capsule / External Preamp Clipping & LF X_max Governor ---
    let (flat_run_l, kappa_l, rail_l) = detect_sub_0dbfs_rail_plateau(wet_l);
    let (flat_run_r, kappa_r, rail_r) = wet_r.map_or((0, 0.0, 0.0), detect_sub_0dbfs_rail_plateau);
    let flat_run = flat_run_l.max(flat_run_r);
    let rail_kappa = kappa_l.max(kappa_r);
    let active_rail = if flat_run_l >= flat_run_r {
        rail_l
    } else {
        rail_r
    };

    // Compute 8–20 kHz post-cabinet stopband excess Gamma_post_cab & mid-to-HF cabinet rolloff
    let bin_800hz = (800 * CYCLIC_WINDOW_SAMPLES) / 48_000;
    let bin_3khz = (3_000 * CYCLIC_WINDOW_SAMPLES) / 48_000;
    let bin_8khz = (8_000 * CYCLIC_WINDOW_SAMPLES) / 48_000;
    let bin_20khz = ((20_000 * CYCLIC_WINDOW_SAMPLES) / 48_000).min(y_spec.len() - 1);
    let carr_set: BTreeSet<usize> = schedule.carrier_bins.iter().copied().collect();

    let mut mid_carr_sum = 0.0_f64;
    let mut mid_cnt = 0usize;
    let mut hf_carr_sum = 0.0_f64;
    let mut hf_carr_cnt = 0usize;
    let mut hf_harm_sum = 0.0_f64;

    for &k in &schedule.carrier_bins {
        let p = y_spec[k].norm_sqr() * inv_n * inv_n;
        if (bin_800hz..=bin_3khz).contains(&k) {
            mid_carr_sum += p;
            mid_cnt += 1;
        } else if (bin_8khz..=bin_20khz).contains(&k) {
            hf_carr_sum += p;
            hf_carr_cnt += 1;
        }
    }
    for k in bin_8khz..=bin_20khz {
        if !carr_set.contains(&k) {
            hf_harm_sum += y_spec[k].norm_sqr() * inv_n * inv_n;
        }
    }

    let mid_mean = mid_carr_sum / (mid_cnt.max(1) as f64);
    let hf_carr_mean = hf_carr_sum / (hf_carr_cnt.max(1) as f64);
    let cab_rolloff_db = 10.0 * (mid_mean.max(1.0e-18) / (hf_carr_mean + 1.0e-18)).log10();
    let gamma_post_cab = hf_harm_sum
        / (hf_carr_sum + (bin_20khz - bin_8khz + 1) as f64 * state.noise_floor_power + 1.0e-18);

    let capsule_clipped = (flat_run >= 4
        && (rail_kappa >= 0.0025 || (cab_rolloff_db >= 10.0 && gamma_post_cab > 10.0)))
        || (flat_run >= 5 && rail_kappa >= 0.0010);

    let mut case_a_replay = false;
    if capsule_clipped {
        let was_already_clipped = state.mic_clip_emitted;
        let dual_mic_clean_companion = wet_r.is_some()
            && ((flat_run_l >= 4 && flat_run_r < 4 && kappa_r < 0.001)
                || (flat_run_r >= 4 && flat_run_l < 4 && kappa_l < 0.001));

        if state.is_master_volume_sweep {
            // Case B: Output-stage Master Volume sweep -> mask clipped 0 dB sub-frame (`w_sub = 0`)
            state.law4c_subframe_weights = [1.0, 1.0, 0.0];
        } else {
            // Case A: Stationary / Boost-induced overload -> scale down to <= 0.80 * A_rail
            let safe_target = (0.80 * f64::from(active_rail)).clamp(0.03, ADC_PEAK_GOVERNOR_FS);
            let cur_peak = ADC_PEAK_GOVERNOR_FS * state.excitation_gain_scale;
            if safe_target < cur_peak {
                state.excitation_gain_scale = (safe_target / ADC_PEAK_GOVERNOR_FS).max(1.0 / 64.0);
                if !was_already_clipped && !dual_mic_clean_companion {
                    case_a_replay = true;
                }
            }
        }

        if !state.mic_clip_emitted {
            state.mic_clip_emitted = true;
            let ceiling_dbfs = 20.0 * active_rail.max(1.0e-6).log10();
            let excess_db = (10.0 * gamma_post_cab.max(1.0).log10()) as f32;
            state
                .warnings
                .push(CaptureWarning::MicrophoneCapsuleOrPreampClip {
                    stem_id: state.stem_id.clone(),
                    ceiling_dbfs,
                    post_cab_excess_db: excess_db,
                    soft_recovery_applied: true,
                    dual_mic_cross_calibrated: dual_mic_clean_companion,
                });
        }
    }

    let _ = CYCLIC_PREFIX_SAMPLES;
    SafeguardFrameDecision {
        discard_and_replay: case_a_replay,
        clean_analysis_window: clean_win,
        anchor_drift_db,
        gamma_post_cab,
        rail_plateau_kappa: rail_kappa,
    }
}
