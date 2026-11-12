#![forbid(unsafe_code)]
//! Closed-Loop Feedback `Laws 1–3`, `Theorem 4.5` Multi-Port / Stereo Disambiguation,
//! `Theorem 4.6` Intra-Sweep `Laws 4A–4D` (Auto Mode A/B + `1.2s` Hadamard Early Stop),
//! and `Theorem 4.7` `256-D` Block-$D$-Optimal Active Learning (`stimulus::laws`).
//!
//! Canonical Reference: `docs/specs/training/capture_and_alignment.md`
//! (`ref[train.capture.excitation.codesigned]`, `ref[train.capture.planner.schedule]`).

use crate::metrics::canonical_ltas_33_centers_hz;

use super::fft::{CYCLIC_WINDOW_SAMPLES, DisjointCombSchedule, PEDESTAL_BIN_MONO, rfft_bins};
use super::waveforms::ActivePrimitiveKind;

/// `Law 1` 3-branch closed-loop excitation regime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExcitationBranch {
    /// `Branch 1A`: Carrier SNR `< 15 dB` (closed hardware noise gate); steps amplitude up via secant search.
    Branch1AGated,
    /// `Branch 1B`: Carrier SNR `>= 15 dB` and sideband SNR `< 10 dB`; pure LTI fast path (`pedestal = off`).
    Branch1BPureLti,
    /// `Branch 1C`: Nonlinear cascade (`SNR_side >= 10 dB`); bounded `[0.25, 4.0]` water-filling + active primitives.
    Branch1CNonlinearCascade,
}

/// `Law 2` sub-Hz resonant pole-zoom spectroscopy regime ($K_{\text{zoom}} \le 8$).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoleZoomRegime {
    /// No sharp high-$Q$ pole zoom currently active.
    None,
    /// `Regime 2A`: Pedestal-off linear zoom ($\Delta k = 2 \implies 23.44\text{ Hz}$ spacing).
    Regime2ALinearPedestalOff,
    /// `Regime 2B`: Nonlinear staggered multi-frame $+2 r \omega_0$ interleaved comb.
    Regime2BNonlinearInterleaved,
}

/// `Theorem 4.5` multi-port / multi-channel excitation topology mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TopologyProbeMode {
    /// `1 -> 1` mono instrument / line signal path.
    #[default]
    Mono1To1,
    /// `1 -> 2` mono-in to stereo-out (chorus, stereo cab/room, dual-mic).
    MonoToStereo1To2,
    /// `2 -> 2` true-stereo Half-Stride Interleaved Dual-Comb + `288 ms` 3-State (`Mid`, `Side`, `Link`).
    TrueStereo2To2,
    /// `2 -> 1` dual-input to mono-out (Frame-0 `23.44 Hz` AM disambiguates Dual-Sum vs Aux-Key Sidechain).
    DualInToMono2To1,
    /// `4CM` 4-Cable Method (`Inst In` + `FX Send` / `FX Return` + `Main Out`).
    FourCableMethod4Cm,
    /// `1 -> 3` Wet / Dry / Wet rig split.
    WetDryWet1To3,
}

/// Disambiguated physical architecture of a `2 -> 1` hardware unit (`Theorem 4.5`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Disambiguated2To1Mode {
    /// Both `in_1` and `in_2` pass audio linearly/nonlinearly to `out_1` (e.g. dual-channel mixer/amp).
    DualAudioInputSum,
    /// `in_2` never feeds `out_1` directly, instead modulating `in_1`'s gain (`External Aux-Key Sidechain`).
    ExternalAuxKeySidechain,
}

/// `Theorem 4.6` acoustic control knob class (`Class I..IV`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnobAcousticClass {
    /// `Class I`: Gain, Drive, Sustain, Fuzz, Threshold, Compression.
    ClassIGainDrive,
    /// `Class II`: Bass, Mid, Treble, Presence, Tone, Cutoff, Resonance, Wah.
    ClassIIToneFilter,
    /// `Class III`: Delay Time, Pitch, Rate, Speed, Detune (often violates $\Delta T_{\text{mem}} \le 15\text{ ms}$).
    ClassIIIPitchDelayTime,
    /// `Class IV`: Level, Master, Output, Mix, Blend, Depth, Decay.
    ClassIVSpatialMixLevel,
}

/// `Theorem 4.6` control trajectory capture mode (`Mode A` vs `Mode B`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SweepExecutionMode {
    /// `Mode A`: `3.5s` continuous bidirectional/monotone sweep ($\Delta T_{\text{mem}}(p) \le 15\text{ ms}$).
    ModeAContinuousSweep,
    /// `Mode B`: `1.0s` stationary Gauss-Lobatto stops (used when $\Delta T_{\text{mem}}(p) > 15\text{ ms}$ or `prefer_mode_b`).
    ModeBStationaryStops,
}

/// Classifies a control identifier/label into [`KnobAcousticClass`] (`Theorem 4.6`).
#[must_use]
pub fn classify_knob_acoustic_class(control_id_or_name: &str) -> KnobAcousticClass {
    let low = control_id_or_name.to_ascii_lowercase();
    if low.contains("time")
        || low.contains("delay")
        || low.contains("rate")
        || low.contains("speed")
        || low.contains("pitch")
    {
        KnobAcousticClass::ClassIIIPitchDelayTime
    } else if low.contains("bass")
        || low.contains("mid")
        || low.contains("treb")
        || low.contains("tone")
        || low.contains("pres")
        || low.contains("cut")
        || low.contains("freq")
        || low.contains("wah")
        || low.contains("res")
    {
        KnobAcousticClass::ClassIIToneFilter
    } else if low.contains("master")
        || low.contains("level")
        || low.contains("vol")
        || low.contains("output")
        || low.starts_with("out_")
        || low.ends_with("_out")
        || low == "out"
        || low.contains("mix")
        || low.contains("blend")
        || low.contains("decay")
    {
        KnobAcousticClass::ClassIVSpatialMixLevel
    } else {
        KnobAcousticClass::ClassIGainDrive
    }
}

/// Returns `true` if `control_id_or_name` is an output/master volume stage (`Law 0.F` Case B).
#[must_use]
pub fn is_master_or_output_control(control_id_or_name: &str) -> bool {
    let low = control_id_or_name.to_ascii_lowercase();
    low.contains("master") || low.contains("output") || low.contains("out_level") || low == "level"
}

/// Signal-measured endpoint probe metrics for zero-heuristic control classification
/// (`ref[train.capture.invariants]`, `Theorem 4.6`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EndpointProbeMetrics {
    /// Signal-measured acoustic control class (`Class I..IV`).
    pub knob_class: KnobAcousticClass,
    /// Measured impulse/tail memory change $\Delta T_{\text{mem}}(p)$ in milliseconds.
    pub delta_t_mem_ms: f64,
    /// Normalized spectral transfer shape spread relative to `StarCenter`.
    pub spectral_shape_spread: f64,
    /// True if signal exhibits pure post-nonlinearity output/master scaling (`Law 0.F` Case B).
    pub is_master_volume_stage: bool,
}

/// Classifies a continuous control purely from signal responses (`active_comb_samples` followed by
/// optional silent tail samples) at `StarCenter` vs an endpoint (`ref[train.capture.invariants]`).
#[must_use]
pub fn classify_knob_from_endpoint_probe(
    star_wet_48k: &[f32],
    probe_wet_48k: &[f32],
    active_comb_samples: usize,
    schedule: &DisjointCombSchedule,
) -> EndpointProbeMetrics {
    let n_total = star_wet_48k.len().min(probe_wet_48k.len());
    let n_act = active_comb_samples.min(n_total);
    if n_act < CYCLIC_WINDOW_SAMPLES {
        return EndpointProbeMetrics {
            knob_class: KnobAcousticClass::ClassIGainDrive,
            delta_t_mem_ms: 0.0,
            spectral_shape_spread: 0.0,
            is_master_volume_stage: false,
        };
    }

    let win_s = &star_wet_48k[n_act - CYCLIC_WINDOW_SAMPLES..n_act];
    let win_p = &probe_wet_48k[n_act - CYCLIC_WINDOW_SAMPLES..n_act];
    let spec_s = rfft_bins(win_s);
    let spec_p = rfft_bins(win_p);
    let kp = schedule.pedestal_bin.unwrap_or(PEDESTAL_BIN_MONO);

    let mut e_carr_s = 0.0_f64;
    let mut e_carr_p = 0.0_f64;
    let mut e_side_s = 0.0_f64;
    let mut e_side_p = 0.0_f64;
    let mut mags_s = Vec::with_capacity(schedule.carrier_bins.len());
    let mut mags_p = Vec::with_capacity(schedule.carrier_bins.len());

    for &b in &schedule.carrier_bins {
        if b < spec_s.len() {
            let ps = spec_s[b].norm_sqr();
            let pp = spec_p[b].norm_sqr();
            e_carr_s += ps;
            e_carr_p += pp;
            mags_s.push(ps.sqrt());
            mags_p.push(pp.sqrt());
            for mult in [1usize, 2usize] {
                if b > mult * kp {
                    e_side_s += spec_s[b - mult * kp].norm_sqr();
                    e_side_p += spec_p[b - mult * kp].norm_sqr();
                }
                if b + mult * kp < spec_s.len() {
                    e_side_s += spec_s[b + mult * kp].norm_sqr();
                    e_side_p += spec_p[b + mult * kp].norm_sqr();
                }
            }
        }
    }

    let norm_s = e_carr_s.sqrt().max(1.0e-12);
    let norm_p = e_carr_p.sqrt().max(1.0e-12);
    let shape_spread = mags_s
        .iter()
        .zip(&mags_p)
        .map(|(&ms, &mp)| (mp / norm_p - ms / norm_s).powi(2))
        .sum::<f64>()
        .sqrt();
    let gain_shift_db = 10.0 * (e_carr_p.max(1.0e-24) / e_carr_s.max(1.0e-24)).log10();
    let thd_s_db = 10.0 * (e_side_s.max(1.0e-24) / e_carr_s.max(1.0e-24)).log10();
    let thd_p_db = 10.0 * (e_side_p.max(1.0e-24) / e_carr_p.max(1.0e-24)).log10();
    let delta_thd_db = (thd_p_db - thd_s_db).abs();

    // Measure post-cutoff tail memory shift (> 15 ms = 720 samples after active_comb_samples)
    let mut delta_t_mem_ms = 0.0_f64;
    let tail_gate = n_act + 720;
    if n_total > tail_gate + 64 {
        let e_act_max = (win_s.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>()
            / (win_s.len() as f64))
            .max(win_p.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / (win_p.len() as f64))
            .max(1.0e-18);
        let tail_s = &star_wet_48k[tail_gate..n_total];
        let tail_p = &probe_wet_48k[tail_gate..n_total];
        let mut es = 0.0_f64;
        let mut ep = 0.0_f64;
        let mut dot = 0.0_f64;
        let mut weighted_t_diff = 0.0_f64;
        let mut diff_e_sum = 0.0_f64;
        let g_scale = (norm_p / norm_s).clamp(0.05, 20.0);
        for (i, (&ys, &yp)) in tail_s.iter().zip(tail_p).enumerate() {
            let fs = f64::from(ys);
            let fp = f64::from(yp);
            es += fs * fs;
            ep += fp * fp;
            dot += fs * fp;
            let resid = fp - g_scale * fs;
            let re = resid * resid;
            diff_e_sum += re;
            weighted_t_diff += ((720 + i) as f64 / 48.0) * re;
        }
        let tail_ms_mean = es.max(ep) / (tail_s.len() as f64);
        let tail_corr = dot.abs() / (es * ep).sqrt().max(1.0e-18);
        if tail_ms_mean > 5.0e-4 * e_act_max
            && (tail_corr < 0.78 || (ep - es).abs() > 0.5 * es.max(ep))
        {
            delta_t_mem_ms = (weighted_t_diff / diff_e_sum.max(1.0e-18)).max(16.0);
        }
    }

    let is_master = delta_t_mem_ms <= 15.0
        && gain_shift_db.abs() > 3.0
        && shape_spread < 0.06
        && delta_thd_db < 1.5
        && (thd_s_db > -45.0 || thd_p_db > -45.0);

    let knob_class = if delta_t_mem_ms > 15.0 {
        KnobAcousticClass::ClassIIIPitchDelayTime
    } else if shape_spread > 0.12 {
        KnobAcousticClass::ClassIIToneFilter
    } else if is_master {
        KnobAcousticClass::ClassIVSpatialMixLevel
    } else {
        KnobAcousticClass::ClassIGainDrive
    };

    EndpointProbeMetrics {
        knob_class,
        delta_t_mem_ms,
        spectral_shape_spread: shape_spread,
        is_master_volume_stage: is_master,
    }
}

/// Selects `Mode A` (`3.5s` continuous sweep) vs `Mode B` (`1.0s` stationary stops) per `Theorem 4.6`
/// (`r[train.capture.control-sweeps]`).
#[must_use]
pub fn select_sweep_execution_mode(
    delta_t_mem_ms: f64,
    knob_class: KnobAcousticClass,
    prefer_mode_b: bool,
) -> SweepExecutionMode {
    if prefer_mode_b
        || !delta_t_mem_ms.is_finite()
        || delta_t_mem_ms > 15.0
        || knob_class == KnobAcousticClass::ClassIIIPitchDelayTime
    {
        SweepExecutionMode::ModeBStationaryStops
    } else {
        SweepExecutionMode::ModeAContinuousSweep
    }
}

/// `Law 4B` kinematic $\alpha$-$\beta$ tracker for moving resonant poles $\hat{\omega}_q(u_p)$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KinematicAlphaBetaPoleTracker {
    /// Smoothed pole frequency in Hz.
    pub freq_hz: f64,
    /// Smoothed pole sweep velocity in Hz/frame.
    pub velocity_hz_per_frame: f64,
    alpha: f64,
    beta: f64,
    initialized: bool,
}

impl Default for KinematicAlphaBetaPoleTracker {
    fn default() -> Self {
        Self::new(0.65, 0.25)
    }
}

impl KinematicAlphaBetaPoleTracker {
    /// Creates an $\alpha$-$\beta$ kinematic pole tracker.
    #[must_use]
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self {
            freq_hz: 0.0,
            velocity_hz_per_frame: 0.0,
            alpha: alpha.clamp(0.05, 0.99),
            beta: beta.clamp(0.01, 0.80),
            initialized: false,
        }
    }

    /// Updates state with `measured_freq_hz` and returns predicted next-frame frequency in Hz.
    pub fn update_and_predict(&mut self, measured_freq_hz: f64) -> f64 {
        if !measured_freq_hz.is_finite() || measured_freq_hz <= 0.0 {
            return (self.freq_hz + self.velocity_hz_per_frame).clamp(20.0, 20_000.0);
        }
        if !self.initialized {
            self.freq_hz = measured_freq_hz;
            self.velocity_hz_per_frame = 0.0;
            self.initialized = true;
            return self.freq_hz;
        }
        let pred = self.freq_hz + self.velocity_hz_per_frame;
        let resid = measured_freq_hz - pred;
        self.freq_hz = (pred + self.alpha * resid).clamp(20.0, 20_000.0);
        self.velocity_hz_per_frame =
            (self.velocity_hz_per_frame + self.beta * resid).clamp(-4_000.0, 4_000.0);
        (self.freq_hz + self.velocity_hz_per_frame).clamp(20.0, 20_000.0)
    }
}

/// `Law 4D` `1.2s` (`12`-frame) Hadamard diagonal early-stopping tracker (`r[train.capture.control-sweeps]`).
#[derive(Debug, Clone, Default)]
pub struct HadamardLaw4dTracker {
    frames_observed: usize,
    consecutive_separable_frames: usize,
    last_sfm_ratio: f64,
}

impl HadamardLaw4dTracker {
    /// Minimum Hadamard probe frames (`12` frames = `1.152s` $\approx 1.2\text{ s}$) before early stop.
    pub const MIN_HADAMARD_FRAMES: usize = 12;

    /// Records the cross-modulation residual ratio $S_{\text{FM}}^{(r)}$ for frame `r` and returns
    /// `true` when the control pair is certified separable ($S_{\text{FM}}^{(r)} < 10^{-4}$ for 4 frames).
    pub fn observe_frame(&mut self, sfm_ratio: f64) -> bool {
        let clean = if sfm_ratio.is_finite() && sfm_ratio >= 0.0 {
            sfm_ratio
        } else {
            1.0
        };
        self.frames_observed += 1;
        self.last_sfm_ratio = clean;
        if clean < 1.0e-4 {
            self.consecutive_separable_frames += 1;
        } else {
            self.consecutive_separable_frames = 0;
        }
        self.frames_observed >= Self::MIN_HADAMARD_FRAMES && self.consecutive_separable_frames >= 4
    }

    /// Last observed $S_{\text{FM}}^{(r)}$ cross-modulation energy ratio.
    #[must_use]
    pub const fn last_sfm_ratio(&self) -> f64 {
        self.last_sfm_ratio
    }
}

/// Prunes Hadamard cross-term diagonals from 1D sweep spectral sensitivities (`Law 4D` commutator pre-pruning):
/// controls whose 1D spectral transfer change is purely scalar gain or inactive (`spread < 0.02`) do not
/// require multi-axis Hadamard cross-sweeps; if $\le 1$ coupled nonlinear/EQ control remains, `0` diagonals run!
#[must_use]
pub fn select_coupled_hadamard_pairs(
    active_range_control_indices: &[usize],
    spectral_shape_spread_by_ctrl: &[f64],
) -> Vec<(usize, usize)> {
    let coupled: Vec<usize> = active_range_control_indices
        .iter()
        .copied()
        .filter(|&p| spectral_shape_spread_by_ctrl.get(p).copied().unwrap_or(1.0) > 0.02)
        .collect();
    if coupled.len() <= 1 {
        return Vec::new();
    }
    let mut pairs = Vec::new();
    for i in 0..coupled.len() {
        for j in (i + 1)..coupled.len() {
            pairs.push((coupled[i], coupled[j]));
            if pairs.len() >= 6 {
                return pairs;
            }
        }
    }
    pairs
}

/// Builds at most `2` mutually orthogonal Walsh-Hadamard sign vectors (`(control_idx, sign)` with
/// `sign in {-1.0, +1.0}`) across all coupled controls (`ADR-0016` §3.1 / `Theorem 3.5`).
#[must_use]
pub fn build_orthogonal_hadamard_diagonals(
    active_range_control_indices: &[usize],
    spectral_shape_spread_by_ctrl: &[f64],
) -> Vec<Vec<(usize, f32)>> {
    let coupled: Vec<usize> = active_range_control_indices
        .iter()
        .copied()
        .filter(|&p| spectral_shape_spread_by_ctrl.get(p).copied().unwrap_or(1.0) > 0.02)
        .collect();
    if coupled.len() <= 1 {
        return Vec::new();
    }
    let diag0: Vec<(usize, f32)> = coupled
        .iter()
        .enumerate()
        .map(|(k, &p)| (p, if k % 2 == 0 { 1.0 } else { -1.0 }))
        .collect();
    if coupled.len() == 2 {
        return vec![diag0];
    }
    let diag1: Vec<(usize, f32)> = coupled
        .iter()
        .enumerate()
        .map(|(k, &p)| (p, if (k / 2) % 2 == 0 { 1.0 } else { -1.0 }))
        .collect();
    vec![diag0, diag1]
}

/// Computes the `Law 4D` spectral cross-modulation / non-separability ratio $S_{\text{FM}}^{(r)}$
/// on a `4,096`-sample cyclic wet window (`< 1e-4` when controls act additively/separably).
pub fn compute_hadamard_sfm_ratio(
    wet_win_48k: &[f32],
    schedule: &DisjointCombSchedule,
    history_mags: &mut [Option<Vec<f64>>; 2],
) -> f64 {
    if wet_win_48k.len() < CYCLIC_WINDOW_SAMPLES {
        return 0.0;
    }
    let y_spec = rfft_bins(&wet_win_48k[wet_win_48k.len() - CYCLIC_WINDOW_SAMPLES..]);
    let kp = schedule.pedestal_bin.unwrap_or(PEDESTAL_BIN_MONO);
    let mut carr_e = 0.0_f64;
    let mut im_e = 0.0_f64;
    let mut cur_mags = Vec::with_capacity(schedule.carrier_bins.len());

    for &b in &schedule.carrier_bins {
        if b < y_spec.len() {
            let p = y_spec[b].norm_sqr();
            carr_e += p;
            cur_mags.push(p.sqrt());
            if b > kp && b + kp < y_spec.len() {
                im_e += y_spec[b - kp].norm_sqr() + y_spec[b + kp].norm_sqr();
            }
        }
    }
    let im_ratio = im_e / carr_e.max(1.0e-18);

    // Second-difference curvature ||m_r - 2*m_{r-1} + m_{r-2}||^2 / ||m_r||^2 vanishes for
    // additively separable linear sweeps u(r) = u0 + r * du and spikes under nonlinear coupling.
    let curv_ratio = if let (Some(m_prev2), Some(m_prev1)) = (&history_mags[0], &history_mags[1])
        && m_prev2.len() == cur_mags.len()
        && m_prev1.len() == cur_mags.len()
    {
        let d2_sq: f64 = cur_mags
            .iter()
            .zip(m_prev1.iter().zip(m_prev2.iter()))
            .map(|(&m0, (&m1, &m2))| (m0 - 2.0 * m1 + m2).powi(2))
            .sum();
        d2_sq / carr_e.max(1.0e-18)
    } else {
        0.0
    };
    history_mags[0] = history_mags[1].take();
    history_mags[1] = Some(cur_mags);

    0.5 * im_ratio + 0.5 * curv_ratio
}

/// Closed-loop `Laws 1–3` + `Theorem 4.7` `256-D` active-learning state machine.
#[derive(Debug, Clone)]
pub struct FeedbackLawsState {
    /// Active `Law 1` branch.
    pub branch: ExcitationBranch,
    /// Active `Law 2` pole-zoom regime.
    pub pole_zoom_regime: PoleZoomRegime,
    /// Selected odd pole-zoom bins ($K_{\text{zoom}} \le 8$).
    pub zoom_odd_bins: Vec<usize>,
    /// Per-bin water-filling inverse pre-emphasis weights (`[0.25, 4.0]`).
    pub bin_weights: Vec<(usize, f64)>,
    /// Target carrier RMS (`Law 1` gate step-up or constant Stage-I power).
    pub target_carrier_rms: f64,
    /// Whether the `93.75 Hz` pedestal is active (`false` in `Branch1BPureLti`).
    pub pedestal_enabled: bool,
    /// 33-band psychoacoustic `EMR` uncertainty weights $M_b^{(m)}$.
    pub emr_band_weights_33: [f64; 33],
    /// Accumulated diagonal Fisher / Gram trace across the 7 intra-tier `256-D` feature blocks.
    pub block_information_7: [f64; 7],
    /// Count of times each of the 5 primitives $\mathcal{P}_1\dots\mathcal{P}_5$ has been fired.
    pub primitive_counts: [usize; 5],
    /// Last peak resonance proxy radius $\hat{r}_{\max}$ for Gate-3 convergence check.
    pub prev_peak_radius_est: f64,
    /// Latest $|\Delta \hat{r}_{\max}|$.
    pub delta_peak_radius: f64,
    /// Monotonically non-decreasing convergence percentage (`0.0..=100.0`).
    pub convergence_pct: f32,
    /// True once all 4 Cramér-Rao convergence gates pass (when `!fixed_duration`).
    pub converged_4gate: bool,
    /// Valid (non-discarded) frames processed.
    pub valid_frames: usize,
    /// Recovered stereo detector link $\hat{\alpha}_{\text{link}} \in [0.0, 1.0]$ (`Theorem 4.5` `2->2`).
    pub stereo_detector_link: Option<f32>,
    /// Quiet-R witness RMS during un-pumped Mid/Side reference (`2->2`).
    stereo_ref_r_rms: Option<f64>,
    /// Disambiguated `2 -> 1` architecture (`DualAudioInputSum` vs `ExternalAuxKeySidechain`).
    pub disambiguated_2to1: Option<Disambiguated2To1Mode>,
}

impl FeedbackLawsState {
    /// Initializes `Laws 1–3` state for `noise_floor_power` and `initial_schedule`.
    #[must_use]
    pub fn new(initial_schedule: &DisjointCombSchedule) -> Self {
        let bin_weights = initial_schedule
            .carrier_bins
            .iter()
            .map(|&b| (b, 1.0_f64))
            .collect();
        Self {
            branch: ExcitationBranch::Branch1CNonlinearCascade,
            pole_zoom_regime: PoleZoomRegime::None,
            zoom_odd_bins: Vec::new(),
            bin_weights,
            target_carrier_rms: 0.14,
            pedestal_enabled: true,
            emr_band_weights_33: [1.0; 33],
            block_information_7: [1.0e-3; 7],
            primitive_counts: [0; 5],
            prev_peak_radius_est: 0.90,
            delta_peak_radius: 1.0,
            convergence_pct: 0.0,
            converged_4gate: false,
            valid_frames: 0,
            stereo_detector_link: None,
            stereo_ref_r_rms: None,
            disambiguated_2to1: None,
        }
    }

    /// Accumulates real `256-D` feature-bus energy into the 7 canonical intra-tier blocks (`Theorem 4.7`).
    pub fn observe_256d_feature_bus(&mut self, features: &[[f32; 256]]) {
        if features.is_empty() {
            return;
        }
        let inv = 1.0 / (features.len() as f64);
        let mut blk_e = [0.0_f64; 7];
        for fb in features {
            // Block 0: Stage III Kautz (0..4, 64..68, 128..136, 192..208)
            for &idx in &[0usize, 1, 2, 3, 64, 65, 128, 192] {
                blk_e[0] += f64::from(fb[idx]).powi(2);
            }
            // Block 1: Stage I T1 Sub-Harmonic Tracker (4..12, 216..224)
            for idx in 4..12 {
                blk_e[1] += f64::from(fb[idx]).powi(2);
            }
            // Block 2: Stage I Op-M[Pre] (12..20, 68..84, 136..152, 208..216)
            for idx in 12..20 {
                blk_e[2] += f64::from(fb[idx]).powi(2);
            }
            // Block 3: Stage II V1A/V1B & Wavefolder (20..44, 84..108, 152..176, 224..240)
            for idx in 20..44 {
                blk_e[3] += f64::from(fb[idx]).powi(2);
            }
            // Block 4: Stage II QuadratureHysteresis2D & Tone/Sag (44..52, 108..116, 176..184, 240..246)
            for idx in 44..52 {
                blk_e[4] += f64::from(fb[idx]).powi(2);
            }
            // Block 5: Stage IV Spatial Tank & Ducking (52..60, 116..126, 184..190, 246..254)
            for idx in 52..60 {
                blk_e[5] += f64::from(fb[idx]).powi(2);
            }
            // Block 6: Stage 0 Clean Sub & Stage III Early FIR (60..64, 126..128, 190..192, 254..256)
            for idx in 60..64 {
                blk_e[6] += f64::from(fb[idx]).powi(2);
            }
        }
        for (dst, &e) in self.block_information_7.iter_mut().zip(&blk_e) {
            *dst += e * inv;
        }
    }

    /// Selects the highest marginal Cholesky log-det gain primitive $\mathcal{P}_1\dots\mathcal{P}_5$
    /// (`Theorem 4.7`) and returns `(best_primitive, max_marginal_gain)`.
    #[must_use]
    pub fn select_next_d_optimal_primitive(&self, bloom_done: bool) -> (ActivePrimitiveKind, f64) {
        // Expected block leverage profile V_c across the 7 blocks for P1..P5
        const PROFILES: [[f64; 7]; 5] = [
            [0.05, 0.85, 0.25, 0.10, 0.05, 0.05, 0.05], // P1 -> T1 tracker & Op-M[Pre]
            [0.10, 0.05, 0.10, 0.90, 0.20, 0.05, 0.10], // P2 -> Stage II V1A/V1B/Wavefolder
            [0.10, 0.05, 0.10, 0.20, 0.90, 0.05, 0.15], // P3 -> QuadratureHysteresis2D LF iron
            [0.15, 0.05, 0.05, 0.10, 0.05, 0.95, 0.20], // P4 -> Stage IV Tank & Ducking
            [0.05, 0.05, 0.15, 0.45, 0.65, 0.25, 0.05], // P5 -> Pump-and-Probe Bloom & Sag
        ];
        let mut best_kind = ActivePrimitiveKind::P2MultiDriveBurst;
        let mut best_gain = -1.0_f64;
        for (idx, kind) in ActivePrimitiveKind::ALL.into_iter().enumerate() {
            if kind == ActivePrimitiveKind::P5PumpAndProbeBloom && bloom_done {
                continue;
            }
            let repeat_penalty = 1.0 / (1.0 + (self.primitive_counts[idx] as f64) * 1.5);
            let mut log_det_gain = 0.0_f64;
            for (prof_w, &g_diag) in PROFILES[idx].iter().zip(&self.block_information_7) {
                log_det_gain += (1.0 + repeat_penalty * prof_w / g_diag.max(1.0e-6)).ln();
            }
            if log_det_gain > best_gain {
                best_gain = log_det_gain;
                best_kind = kind;
            }
        }
        (best_kind, best_gain.max(0.0))
    }

    /// Updates `Laws 1–3`, `Theorem 4.5` topology disambiguation, and 4-Gate convergence on a valid
    /// `4,096`-sample wet analysis window.
    #[allow(clippy::too_many_arguments)]
    pub fn update_from_valid_window(
        &mut self,
        dry_win_l: &[f32],
        wet_win_l: &[f32],
        wet_win_r: Option<&[f32]>,
        schedule: &DisjointCombSchedule,
        noise_floor_power: f64,
        topology_mode: TopologyProbeMode,
        max_frames: usize,
        fixed_duration: bool,
        bloom_completed: bool,
    ) {
        let frame_idx = self.valid_frames;
        self.valid_frames += 1;

        let x_spec = rfft_bins(dry_win_l);
        let y_spec = rfft_bins(wet_win_l);
        let inv_n = 2.0 / (CYCLIC_WINDOW_SAMPLES as f64);
        let floor_p = noise_floor_power.max(1.0e-14);

        // --- Theorem 4.5: 2 -> 1 Frame-0 AM Sidechain vs Dual-Sum Disambiguation ---
        if topology_mode == TopologyProbeMode::DualInToMono2To1 && self.disambiguated_2to1.is_none()
        {
            // Bin 127 (~1,488.3 Hz) is injected exclusively on `in_2` with +/-2 bin (23.44 Hz) AM
            let in2_probe_bin = 127usize;
            let p_in2_direct = y_spec
                .get(in2_probe_bin)
                .map_or(0.0, |c| c.norm_sqr() * inv_n * inv_n);
            let mut am_sideband_p = 0.0_f64;
            let mut main_carr_p = 0.0_f64;
            for &b in schedule.carrier_bins.iter().take(16) {
                if b != in2_probe_bin && b + 2 < y_spec.len() && b >= 2 {
                    main_carr_p += y_spec[b].norm_sqr() * inv_n * inv_n;
                    am_sideband_p +=
                        (y_spec[b - 2].norm_sqr() + y_spec[b + 2].norm_sqr()) * inv_n * inv_n;
                }
            }
            if p_in2_direct > 1.0e-3 * main_carr_p.max(1.0e-12) {
                self.disambiguated_2to1 = Some(Disambiguated2To1Mode::DualAudioInputSum);
            } else if am_sideband_p > 1.0e-5 * main_carr_p.max(1.0e-12) {
                self.disambiguated_2to1 = Some(Disambiguated2To1Mode::ExternalAuxKeySidechain);
            } else {
                self.disambiguated_2to1 = Some(Disambiguated2To1Mode::DualAudioInputSum);
            }
        }

        // --- Theorem 4.5: 2 -> 2 Stereo Detector Coupling alpha_link ---
        if topology_mode == TopologyProbeMode::TrueStereo2To2
            && let Some(wr) = wet_win_r
            && wr.len() >= CYCLIC_WINDOW_SAMPLES
        {
            let r_rms = (wr.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>()
                / (wr.len() as f64))
                .sqrt();
            if frame_idx == 0 {
                // Pure Mid / reference frame
                self.stereo_ref_r_rms = Some(r_rms.max(1.0e-6));
            } else if frame_idx == 2 {
                // Frame 2 is Hard-Pan Loud L + Quiet (-24 dBFS) Witness R:
                // compare witness R output gain against expected linear scaling
                if let Some(ref_r) = self.stereo_ref_r_rms {
                    let expected_quiet_r = 0.25 * ref_r;
                    let comp_ratio = (1.0 - (r_rms / expected_quiet_r.max(1.0e-9))).clamp(0.0, 1.0);
                    self.stereo_detector_link = Some(comp_ratio as f32);
                }
            }
        }

        // --- Law 1: Carrier SNR, Sideband SNR & Water-Filling Inverse Pre-Emphasis ---
        let kp = schedule.pedestal_bin.unwrap_or(PEDESTAL_BIN_MONO);
        let mut carr_p_sum = 0.0_f64;
        let mut side_p_sum = 0.0_f64;
        let mut h_mags = Vec::with_capacity(schedule.carrier_bins.len());

        for &b in &schedule.carrier_bins {
            let xp = x_spec[b].norm_sqr() * inv_n * inv_n;
            let yp = y_spec[b].norm_sqr() * inv_n * inv_n;
            carr_p_sum += yp;
            let h_mag = (yp / xp.max(1.0e-12)).sqrt();
            h_mags.push((b, h_mag));

            for mult in [1usize, 2usize] {
                if b > mult * kp {
                    side_p_sum += y_spec[b - mult * kp].norm_sqr() * inv_n * inv_n;
                }
                if b + mult * kp < y_spec.len() {
                    side_p_sum += y_spec[b + mult * kp].norm_sqr() * inv_n * inv_n;
                }
            }
        }

        let n_c = schedule.carrier_bins.len().max(1) as f64;
        let snr_carrier_db = 10.0 * ((carr_p_sum / n_c) / floor_p).max(1.0e-9).log10();
        let snr_side_db = 10.0 * ((side_p_sum / (4.0 * n_c)) / floor_p).max(1.0e-9).log10();

        let dry_rms = (dry_win_l
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            / (dry_win_l.len().max(1) as f64))
            .sqrt();
        if snr_carrier_db < 15.0 {
            self.branch = ExcitationBranch::Branch1AGated;
            self.target_carrier_rms = (self.target_carrier_rms * 2.0).min(0.42);
            self.pedestal_enabled = true;
        } else if snr_side_db < 10.0
            && !(self.branch == ExcitationBranch::Branch1CNonlinearCascade && dry_rms < 0.10)
        {
            self.branch = ExcitationBranch::Branch1BPureLti;
            self.pedestal_enabled = false;
        } else {
            self.branch = ExcitationBranch::Branch1CNonlinearCascade;
            self.pedestal_enabled = true;
        }

        // Bounded [0.25, 4.0] (±12 dB) water-filling inverse pre-emphasis normalized to unit mean power
        if !h_mags.is_empty() {
            let median_h = {
                let mut tmp: Vec<f64> = h_mags.iter().map(|&(_, h)| h).collect();
                tmp.sort_by(|a, b| a.total_cmp(b));
                tmp[tmp.len() / 2].max(1.0e-4)
            };
            let mut raw_w: Vec<(usize, f64)> = h_mags
                .iter()
                .map(|&(b, h)| {
                    let inv_h = (median_h / h.max(1.0e-4)).sqrt().clamp(0.25, 4.0);
                    (b, inv_h)
                })
                .collect();
            let mean_sq = raw_w.iter().map(|&(_, w)| w * w).sum::<f64>() / (raw_w.len() as f64);
            let norm = 1.0 / mean_sq.sqrt().max(1.0e-9);
            for (_, w) in &mut raw_w {
                *w = (*w * norm).clamp(0.25, 4.0);
            }
            self.bin_weights = raw_w;
        }

        // --- Law 2: Sub-Hz Pole-Zoom Spectroscopy (K_zoom <= 8) ---
        let mut peaks: Vec<(f64, usize)> = Vec::new();
        for i in 1..h_mags.len().saturating_sub(1) {
            let hm1 = h_mags[i - 1].1;
            let h0 = h_mags[i].1;
            let hp1 = h_mags[i + 1].1;
            let neigh = 0.5 * (hm1 + hp1).max(1.0e-6);
            let prominence = h0 / neigh;
            if prominence > 1.28 {
                peaks.push((prominence, h_mags[i].0));
            }
        }
        peaks.sort_by(|a, b| b.0.total_cmp(&a.0));
        let top_prom = peaks.first().map_or(1.0, |p| p.0);
        let est_r = (1.0 - 0.08 / top_prom.max(1.0)).clamp(0.80, 0.999);
        self.delta_peak_radius = (est_r - self.prev_peak_radius_est).abs();
        self.prev_peak_radius_est = est_r;

        let mut new_zoom = Vec::new();
        if !peaks.is_empty() {
            if self.branch == ExcitationBranch::Branch1BPureLti {
                self.pole_zoom_regime = PoleZoomRegime::Regime2ALinearPedestalOff;
                for &(_, kb) in peaks.iter().take(4) {
                    if kb >= 5 {
                        new_zoom.push(kb - 2);
                    }
                    new_zoom.push(kb + 2);
                }
            } else {
                self.pole_zoom_regime = PoleZoomRegime::Regime2BNonlinearInterleaved;
                let step = 2 * ((frame_idx % 3) + 1);
                for &(_, kb) in peaks.iter().take(4) {
                    if kb > step + 3 {
                        new_zoom.push(kb - step);
                    }
                    new_zoom.push(kb + step);
                }
            }
        } else {
            self.pole_zoom_regime = PoleZoomRegime::None;
        }
        new_zoom.truncate(8);
        self.zoom_odd_bins = new_zoom;

        // --- Law 3: 33-Band EMR Update & Spectral Proxy Feature Block Accumulation ---
        let centers = canonical_ltas_33_centers_hz();
        let mut bands_converged = 0usize;
        for (bi, &fc) in centers.iter().enumerate() {
            let b_lo = (((fc * 0.89) * (CYCLIC_WINDOW_SAMPLES as f64) / 48_000.0).floor() as usize)
                .clamp(1, y_spec.len() - 1);
            let b_hi = (((fc * 1.12) * (CYCLIC_WINDOW_SAMPLES as f64) / 48_000.0).ceil() as usize)
                .clamp(b_lo, y_spec.len() - 1);
            let mut band_e = 0.0_f64;
            for c in &y_spec[b_lo..=b_hi] {
                band_e += c.norm_sqr() * inv_n * inv_n;
            }
            let snr_b = band_e / ((b_hi - b_lo + 1) as f64 * floor_p);
            self.emr_band_weights_33[bi] = (1.0 / (1.0 + 0.05 * snr_b.sqrt())).clamp(0.05, 2.0);
            if snr_b >= 4.0 || self.valid_frames >= 6 {
                bands_converged += 1;
            }
        }

        // Update block information proxy smoothly per observed frame so active learning advances
        self.block_information_7[0] += 0.35 * (carr_p_sum / n_c).sqrt().max(0.05);
        self.block_information_7[6] += 0.35;
        if self.branch == ExcitationBranch::Branch1CNonlinearCascade {
            self.block_information_7[3] += 0.25 * (side_p_sum / n_c).sqrt().max(0.04);
        }

        // --- 4-Gate Cramér-Rao Early-Stopping Evaluation ---
        let (_, max_marginal_gain) = self.select_next_d_optimal_primitive(bloom_completed);
        let gate1_emr = bands_converged >= 30;
        let gate2_d_opt = self.branch == ExcitationBranch::Branch1BPureLti
            || max_marginal_gain < 0.55
            || self.valid_frames >= 14;
        let gate3_poles = self.delta_peak_radius < 5.0e-3 || self.valid_frames >= 8;
        let min_req_frames = match self.branch {
            ExcitationBranch::Branch1BPureLti => 6usize, // 0.576s
            ExcitationBranch::Branch1AGated => 8usize,
            ExcitationBranch::Branch1CNonlinearCascade => 10usize,
        };
        let gate4_floor = self.valid_frames >= min_req_frames
            && (self.branch == ExcitationBranch::Branch1BPureLti || bloom_completed);

        let all_gates = gate1_emr && gate2_d_opt && gate3_poles && gate4_floor;
        if all_gates && !fixed_duration {
            self.converged_4gate = true;
            self.convergence_pct = 100.0;
        } else {
            let progress_frames =
                (self.valid_frames as f32) / (max_frames.max(min_req_frames).max(1) as f32);
            let gate_score = 0.25 * (bands_converged as f32 / 33.0)
                + 0.25 * (1.0 / (1.0 + max_marginal_gain as f32))
                + 0.25 * if gate3_poles { 1.0 } else { 0.5 }
                + 0.25 * ((self.valid_frames as f32) / (min_req_frames as f32)).min(1.0);
            let raw_pct = if fixed_duration {
                (progress_frames * 100.0).clamp(0.0, 99.5)
            } else {
                (progress_frames.max(0.85 * gate_score) * 100.0).clamp(0.0, 99.0)
            };
            self.convergence_pct = self.convergence_pct.max(raw_pct);
        }
    }
}
