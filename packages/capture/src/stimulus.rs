#![forbid(unsafe_code)]
//! Closed-Loop `AdaptiveStimulusController` (`< 1.5 ms` per `96.0 ms` Cyclic Frame),
//! `A1..A6` Co-Designed Excitation, `Laws 0.A–0.F` Hardware Safeguards, `Laws 1–3` Active Learning,
//! and Dual-Mode Streaming / Batch Distillation (`capture::stimulus`).
//!
//! Canonical Reference: `docs/specs/training/capture_and_alignment.md`
//! (`ref[train.capture.excitation.codesigned]`, `ref[train.capture.invariants]`).

pub mod batch;
pub mod fft;
pub mod laws;
pub mod null_test;
pub mod pipeline;
pub mod safeguards;
pub mod sealing;
pub mod waveforms;

use std::collections::BTreeMap;

use dnam_engine::schema::DnamError;

use crate::CORESET_FP16_FRAMES_PER_STEM;
use crate::align::{DecayTailDecision, evaluate_live_decay_tail};
use crate::proj::directory::StmcTrajectoryWindow;
use crate::proj::vault::VaultCaptureSpec;
use crate::resampler::{
    resample_48k_to_arbitrary_rate, resample_arbitrary_rate_to_48k, resampled_frame_len,
};
use crate::targets::{StemResponse, StimulusFrame};

pub use batch::{
    BatchSolveReport, BatchStimulusOptions, generate_refinement_stimulus_batch,
    generate_stimulus_batch, recapture_clipped_stem, solve_from_responses,
};
pub use fft::{
    ADAPTIVE_FRAME_SAMPLES, ADC_PEAK_GOVERNOR_FS, CYCLIC_PREFIX_SAMPLES, CYCLIC_WINDOW_SAMPLES,
    HANN_CROSSFADE_SAMPLES, MIN_COMB_BIN_STRIDE, PEDESTAL_BIN_MONO, PEDESTAL_BIN_STEREO_R,
    SCHROEDER_MAX_CREST_DB, build_disjoint_comb_schedule, compute_low_crest_schroeder_phases,
    compute_low_crest_schroeder_phases_with_hint, synthesize_cyclic_comb_frame,
    synthesize_cyclic_comb_frame_with_hint,
};
pub use laws::{
    Disambiguated2To1Mode, EndpointProbeMetrics, ExcitationBranch, FeedbackLawsState,
    HadamardLaw4dTracker, KinematicAlphaBetaPoleTracker, KnobAcousticClass, PoleZoomRegime,
    SweepExecutionMode, TopologyProbeMode, build_orthogonal_hadamard_diagonals,
    classify_knob_acoustic_class, classify_knob_from_endpoint_probe, compute_hadamard_sfm_ratio,
    is_master_or_output_control, select_coupled_hadamard_pairs, select_sweep_execution_mode,
};
pub use null_test::{LiveNullTestReport, run_live_null_test_and_autopatch};
pub use pipeline::{AdaptiveTrainingOptions, AdaptiveTrainingReport, train_adaptive_target};
pub use safeguards::{CaptureWarning, LtvRegime, SafeguardState, evaluate_frame_safeguards};
pub use waveforms::{
    A3_GLISSANDO_SAMPLES_250MS, A5_CORESET_SAMPLES_500MS, ActivePrimitiveKind,
    BLOOM_EPISODE2_TRIGGER_DB, BloomEpisodeTelemetry, PUMP_PROBE_COMB_PERIOD,
    analyze_pump_and_probe_recovery, apply_law4c_micro_staircase,
    compute_stmc_psd_88_and_spectral_4096, deterministic_seed_256, encode_coreset_12k_fp16,
    synthesize_a1_log_chirp, synthesize_a2_dynamic_staircase, synthesize_a3_monophonic_glissando,
    synthesize_a5_musical_transient_coreset, synthesize_active_primitive_frame,
};

/// Configuration for [`AdaptiveStimulusController`].
#[derive(Debug, Clone)]
pub struct StimulusConfig {
    /// Stem identifier (e.g. `"star_center"`, `"sweep_ctrl_0"`).
    pub stem_id: String,
    /// Target device sample rate in Hz (`48_000` canonical; `8_000..=384_000` auto-resampled).
    pub sample_rate_hz: u32,
    /// Multi-port / multi-channel probe topology (`1->1`, `1->2`, `2->2`, `2->1`, `4CM`, `WDW`).
    pub topology_mode: TopologyProbeMode,
    /// Stationary control values `[0..P]` for this stem (or starting values for a Mode A sweep).
    pub control_values: Vec<f32>,
    /// Optional ending control values `[0..P]` when executing a continuous `Mode A` sweep.
    pub sweep_end_control_values: Option<Vec<f32>>,
    /// Active swept control index `p` (`0` for `StarCenter`).
    pub active_control_id: u16,
    /// True if sweeping a Master / Output Volume stage (`Law 0.F` Case B).
    pub is_master_volume_sweep: bool,
    /// `--fixed-duration`: disables 4-Gate Cramér-Rao early stopping and runs all `max_frames`.
    pub fixed_duration: bool,
    /// Maximum active comb/primitive `96 ms` frames (`default = 36` -> `3.456s`).
    pub max_frames: usize,
    /// Maximum post-stimulus live decay-tail `96 ms` frames (`default = 6` -> `0.576s`).
    pub max_tail_frames: usize,
    /// Calibrated hardware noise floor in dBFS (`default = -100.0`).
    pub noise_floor_dbfs: f32,
    /// Deterministic `256 B` stimulus seed $\boldsymbol{\Xi}$.
    pub seed_256: [u8; 256],
}

impl Default for StimulusConfig {
    fn default() -> Self {
        Self {
            stem_id: "star_center".to_string(),
            sample_rate_hz: 48_000,
            topology_mode: TopologyProbeMode::Mono1To1,
            control_values: Vec::new(),
            sweep_end_control_values: None,
            active_control_id: 0,
            is_master_volume_sweep: false,
            fixed_duration: false,
            max_frames: 24,
            max_tail_frames: 4,
            noise_floor_dbfs: -100.0,
            seed_256: deterministic_seed_256(0xACE5_2026_0001),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ControllerPhase {
    ActiveLearning,
    Oc2Glissando { gliss_frame_idx: usize },
    Coreset { coreset_frame_idx: usize },
    DecayTail { tail_frame_idx: usize },
    Finished,
}

/// Pure IO-buffer closed-loop `AdaptiveStimulusController` (`< 1.5 ms` per `96 ms` frame, `wasm32`-safe).
#[derive(Debug, Clone)]
pub struct AdaptiveStimulusController {
    config: StimulusConfig,
    safeguards: SafeguardState,
    laws: FeedbackLawsState,
    schedule_l: fft::DisjointCombSchedule,
    schedule_r: fft::DisjointCombSchedule,
    phase: ControllerPhase,
    emitted_frame_index: usize,
    consecutive_replays: usize,
    hold_counter: usize,
    last_dry_48k_l: Vec<f32>,
    last_dry_48k_r: Option<Vec<f32>>,
    last_aux_48k_l: Option<Vec<f32>>,
    carrier_tail_l: [f32; HANN_CROSSFADE_SAMPLES],
    carrier_tail_r: [f32; HANN_CROSSFADE_SAMPLES],
    phases_l: Vec<f64>,
    phases_r: Vec<f64>,
    last_carrier_crest_db: f32,
    last_emitted_was_primitive: bool,
    // Bloom state (`Theorem 4.8`)
    active_primitive: Option<(ActivePrimitiveKind, usize, bool)>,
    bloom_ep1_done: bool,
    bloom_ep2_done: bool,
    bloom_probe_acc: Vec<f32>,
    bloom_telemetry: Option<BloomEpisodeTelemetry>,
    // Pre-synthesized A3 (12k) and A5 (13,824 = 3 * 4,608) buffers
    glissando_buf_l: Vec<f32>,
    coreset_dry_l: Vec<f32>,
    coreset_dry_r: Option<Vec<f32>>,
    coreset_wet_l: Vec<f32>,
    coreset_wet_r: Vec<f32>,
    recorded_wet_total_l: Vec<f32>,
    trajectory_windows: Vec<StmcTrajectoryWindow>,
}

impl AdaptiveStimulusController {
    /// Creates a new closed-loop [`AdaptiveStimulusController`] from `config`.
    #[must_use]
    pub fn new(config: StimulusConfig) -> Self {
        let empty_mask = std::collections::BTreeSet::new();
        let schedule_l =
            build_disjoint_comb_schedule(48_000, false, Some(PEDESTAL_BIN_MONO), &empty_mask, &[]);
        let schedule_r = build_disjoint_comb_schedule(
            48_000,
            true,
            Some(PEDESTAL_BIN_STEREO_R),
            &empty_mask,
            &[],
        );
        let safeguards = SafeguardState::new(
            config.stem_id.clone(),
            config.is_master_volume_sweep,
            config.noise_floor_dbfs,
        );
        let laws = FeedbackLawsState::new(&schedule_l);
        let need_stereo_dry = matches!(
            config.topology_mode,
            TopologyProbeMode::TrueStereo2To2 | TopologyProbeMode::DualInToMono2To1
        );
        let (mut core_l, mut core_r) = synthesize_a5_musical_transient_coreset(
            &config.seed_256,
            CORESET_FP16_FRAMES_PER_STEM,
            if need_stereo_dry { 2 } else { 1 },
            48_000,
        );
        // Pad to 3 * 4,608 = 13,824 samples (trailing 1,824 samples = 38 ms quiet settle)
        core_l.resize(3 * ADAPTIVE_FRAME_SAMPLES, 0.0);
        if let Some(ref mut cr) = core_r {
            cr.resize(3 * ADAPTIVE_FRAME_SAMPLES, 0.0);
        }
        let mut gliss_l = synthesize_a3_monophonic_glissando(48_000, 0.72 * ADC_PEAK_GOVERNOR_FS);
        gliss_l.resize(3 * ADAPTIVE_FRAME_SAMPLES, 0.0);

        let init_amps_l: Vec<(usize, f64)> =
            schedule_l.carrier_bins.iter().map(|&b| (b, 1.0)).collect();
        let (phases_l, _) = compute_low_crest_schroeder_phases(&init_amps_l);
        let init_amps_r: Vec<(usize, f64)> =
            schedule_r.carrier_bins.iter().map(|&b| (b, 1.0)).collect();
        let (phases_r, _) = compute_low_crest_schroeder_phases(&init_amps_r);

        Self {
            config,
            safeguards,
            laws,
            schedule_l,
            schedule_r,
            phase: ControllerPhase::ActiveLearning,
            emitted_frame_index: 0,
            consecutive_replays: 0,
            hold_counter: 0,
            last_dry_48k_l: Vec::new(),
            last_dry_48k_r: None,
            last_aux_48k_l: None,
            carrier_tail_l: [0.0; HANN_CROSSFADE_SAMPLES],
            carrier_tail_r: [0.0; HANN_CROSSFADE_SAMPLES],
            phases_l,
            phases_r,
            last_carrier_crest_db: 0.0,
            last_emitted_was_primitive: false,
            active_primitive: None,
            bloom_ep1_done: false,
            bloom_ep2_done: false,
            bloom_probe_acc: Vec::new(),
            bloom_telemetry: None,
            glissando_buf_l: gliss_l,
            coreset_dry_l: core_l,
            coreset_dry_r: core_r,
            coreset_wet_l: Vec::new(),
            coreset_wet_r: Vec::new(),
            recorded_wet_total_l: Vec::new(),
            trajectory_windows: Vec::new(),
        }
    }

    /// Monotonically non-decreasing convergence percentage in `[0.0, 100.0]`.
    #[must_use]
    pub fn convergence_pct(&self) -> f32 {
        match self.phase {
            ControllerPhase::Finished => 100.0,
            ControllerPhase::DecayTail { .. } => self.laws.convergence_pct.max(98.0),
            ControllerPhase::Coreset { coreset_frame_idx } => self
                .laws
                .convergence_pct
                .max(92.0 + 2.0 * (coreset_frame_idx as f32)),
            _ => self.laws.convergence_pct,
        }
    }

    /// Estimated maximum remaining capture duration in seconds (`>= 0.0`).
    #[must_use]
    pub fn remaining_max_sec(&self) -> f32 {
        if self.phase == ControllerPhase::Finished {
            return 0.0;
        }
        let rem_active = self
            .config
            .max_frames
            .saturating_sub(self.laws.valid_frames);
        let rem_coreset = match self.phase {
            ControllerPhase::ActiveLearning | ControllerPhase::Oc2Glissando { .. } => 3,
            ControllerPhase::Coreset { coreset_frame_idx } => {
                3usize.saturating_sub(coreset_frame_idx)
            }
            ControllerPhase::DecayTail { .. } | ControllerPhase::Finished => 0,
        };
        ((rem_active + rem_coreset + self.config.max_tail_frames) as f32) * 0.096
    }

    /// Structured hardware safeguard warnings accumulated so far.
    #[must_use]
    pub fn warnings(&self) -> &[CaptureWarning] {
        &self.safeguards.warnings
    }

    /// Active `Law 1` excitation branch (`Gated`, `PureLti`, or `NonlinearCascade`).
    #[must_use]
    pub const fn active_branch(&self) -> ExcitationBranch {
        self.laws.branch
    }

    /// Active `Law 0.A` LTV classification (`TimeInvariant`, `LfoModulated`, or `EnvelopeSvf`).
    #[must_use]
    pub const fn ltv_regime(&self) -> LtvRegime {
        self.safeguards.ltv_regime
    }

    /// Last synthesized Schroeder multisine carrier crest factor in dB ($\le 4.5\text{ dB}$).
    #[must_use]
    pub const fn last_carrier_crest_db(&self) -> f32 {
        self.last_carrier_crest_db
    }

    /// Recovered `2 -> 2` stereo detector coupling $\hat{\alpha}_{\text{link}} \in [0.0, 1.0]$.
    #[must_use]
    pub const fn stereo_detector_link(&self) -> Option<f32> {
        self.laws.stereo_detector_link
    }

    /// Disambiguated `2 -> 1` architecture (`DualAudioInputSum` vs `ExternalAuxKeySidechain`).
    #[must_use]
    pub const fn disambiguated_2to1_mode(&self) -> Option<Disambiguated2To1Mode> {
        self.laws.disambiguated_2to1
    }

    /// Latest `Theorem 4.8` Pump-and-Probe Bloom recovery telemetry.
    #[must_use]
    pub fn bloom_telemetry(&self) -> Option<&BloomEpisodeTelemetry> {
        self.bloom_telemetry.as_ref()
    }

    /// Sub-frame quadrature weights `[w_-18dB, w_-6dB, w_0dB]` (`[1.0, 1.0, 0.0]` under `Law 0.F` Case B).
    #[must_use]
    pub const fn law4c_subframe_weights(&self) -> [f32; 3] {
        self.safeguards.law4c_subframe_weights
    }

    /// Active left-channel disjoint-comb schedule.
    #[must_use]
    pub const fn schedule_l(&self) -> &fft::DisjointCombSchedule {
        &self.schedule_l
    }

    /// True while still in the `ActiveLearning` comb/primitive phase prior to `A5` coreset capture.
    #[must_use]
    pub fn is_in_active_learning(&self) -> bool {
        self.phase == ControllerPhase::ActiveLearning
    }

    /// Requests an immediate transition from `ActiveLearning` into the 3-frame `A5` coreset phase
    /// (used by `Law 4D` `HadamardLaw4dTracker` early stopping so `coreset_wet` is always captured).
    pub fn request_coreset_handoff(&mut self) {
        if self.phase == ControllerPhase::ActiveLearning {
            self.laws.converged_4gate = true;
            self.active_primitive = None;
            self.bloom_probe_acc.clear();
        }
    }

    fn aux_for_non_active_48k(&self, ref_l: &[f32]) -> Option<Vec<f32>> {
        matches!(
            self.config.topology_mode,
            TopologyProbeMode::FourCableMethod4Cm | TopologyProbeMode::DualInToMono2To1
        )
        .then(|| ref_l.to_vec())
    }

    pub(crate) fn normalize_wet_to_48k(&self, resp: &StemResponse) -> (Vec<f32>, Option<Vec<f32>>) {
        let to_48k = |ch: &[f32]| {
            if resp.sample_rate_hz == 48_000 {
                ch.to_vec()
            } else {
                resample_arbitrary_rate_to_48k(
                    ch,
                    resp.sample_rate_hz,
                    Some(ADAPTIVE_FRAME_SAMPLES),
                )
            }
        };
        (to_48k(&resp.out_1), resp.out_2.as_deref().map(to_48k))
    }

    fn emit_coreset_or_silence_48k(&mut self, coreset_idx: Option<usize>) -> StimulusFrame {
        let (cl, cr) = if let Some(idx) = coreset_idx {
            let start = idx * ADAPTIVE_FRAME_SAMPLES;
            (
                self.coreset_dry_l[start..start + ADAPTIVE_FRAME_SAMPLES].to_vec(),
                self.coreset_dry_r
                    .as_ref()
                    .map(|r| r[start..start + ADAPTIVE_FRAME_SAMPLES].to_vec()),
            )
        } else {
            (
                vec![0.0_f32; ADAPTIVE_FRAME_SAMPLES],
                self.coreset_dry_r
                    .as_ref()
                    .map(|_| vec![0.0_f32; ADAPTIVE_FRAME_SAMPLES]),
            )
        };
        let aux = self.aux_for_non_active_48k(&cl);
        self.build_stimulus_frame_from_48k(cl, cr, aux)
    }

    fn build_stimulus_frame_from_48k(
        &mut self,
        dry_l_48k: Vec<f32>,
        dry_r_48k: Option<Vec<f32>>,
        aux_l_48k: Option<Vec<f32>>,
    ) -> StimulusFrame {
        self.last_dry_48k_l.clone_from(&dry_l_48k);
        self.last_dry_48k_r.clone_from(&dry_r_48k);
        self.last_aux_48k_l.clone_from(&aux_l_48k);

        let sr = self.config.sample_rate_hz.clamp(8_000, 384_000);
        let dst_len = resampled_frame_len(ADAPTIVE_FRAME_SAMPLES, 48_000, sr);
        let map_ch = |buf: Vec<f32>| -> Vec<f32> {
            if sr == 48_000 {
                buf
            } else {
                resample_48k_to_arbitrary_rate(&buf, sr, Some(dst_len))
            }
        };

        let in_1 = map_ch(dry_l_48k);
        let in_2 = dry_r_48k.map(map_ch);
        let in_3 = aux_l_48k.map(map_ch);

        // Interpolate per-sample Mode A / Hadamard sweep trajectories if configured.
        // During Coreset frames 1..=3 (0..13,824 samples at 48 kHz, sealing 0..12,000), sweep [0 -> 1]
        // directly across the 12,000-sample sealed coreset so STMC windows align with wet_coreset_fp16.
        let (t0, t1, reset_coreset_knots) = match self.phase {
            ControllerPhase::Coreset { coreset_frame_idx } => {
                let idx0 = coreset_frame_idx.saturating_sub(1);
                let s_start = (idx0 * ADAPTIVE_FRAME_SAMPLES) as f32;
                let s_end = (coreset_frame_idx * ADAPTIVE_FRAME_SAMPLES) as f32;
                let denom = CORESET_FP16_FRAMES_PER_STEM as f32;
                (
                    (s_start / denom).clamp(0.0, 1.0),
                    (s_end / denom).clamp(0.0, 1.0),
                    coreset_frame_idx == 1,
                )
            }
            ControllerPhase::DecayTail { .. } => (1.0_f32, 1.0_f32, false),
            _ => {
                let total_est = self.config.max_frames.max(1) as f32;
                (
                    (self.laws.valid_frames as f32 / total_est).clamp(0.0, 1.0),
                    ((self.laws.valid_frames + 1) as f32 / total_est).clamp(0.0, 1.0),
                    false,
                )
            }
        };

        let is_coreset = matches!(self.phase, ControllerPhase::Coreset { .. });
        let (cur_ctrls, control_trajectories) = sealing::interpolate_frame_trajectories(
            &self.config.control_values,
            self.config.sweep_end_control_values.as_deref(),
            self.config.active_control_id,
            self.safeguards.law4c_subframe_weights[1],
            t0,
            t1,
            reset_coreset_knots,
            is_coreset,
            in_1.len(),
            &mut self.trajectory_windows,
        );

        let idx = self.emitted_frame_index;
        self.emitted_frame_index += 1;
        StimulusFrame {
            frame_index: idx,
            sample_rate_hz: sr,
            in_1,
            in_2,
            in_3,
            in_4: None,
            control_values: cur_ctrls,
            control_trajectories,
        }
    }

    fn synthesize_next_active_48k(&mut self) -> (Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>) {
        let ceiling = ADC_PEAK_GOVERNOR_FS * self.safeguards.excitation_gain_scale;

        // Refresh comb schedule if not in 2-frame long-memory hold
        if !self.safeguards.long_memory_2frame_hold || self.hold_counter == 0 {
            let ped_l = self.laws.pedestal_enabled.then_some(PEDESTAL_BIN_MONO);
            self.schedule_l = build_disjoint_comb_schedule(
                48_000,
                false,
                ped_l,
                &self.safeguards.squeal_mask_bins,
                &self.laws.zoom_odd_bins,
            );
            self.schedule_r = build_disjoint_comb_schedule(
                48_000,
                true,
                self.laws.pedestal_enabled.then_some(PEDESTAL_BIN_STEREO_R),
                &self.safeguards.squeal_mask_bins,
                &[],
            );
            if self.safeguards.long_memory_2frame_hold {
                self.hold_counter = 1;
            }
        } else {
            self.hold_counter = 0;
        }

        // Check if an operator-aligned active primitive (or Theorem 4.8 Bloom episode) is running
        if let Some((prim, offset, is_ep2)) = self.active_primitive {
            self.last_emitted_was_primitive = true;
            let dry_l = synthesize_active_primitive_frame(prim, offset, is_ep2, ceiling);
            let total_len = match (prim, is_ep2) {
                (ActivePrimitiveKind::P5PumpAndProbeBloom, false) => 4,
                (ActivePrimitiveKind::P5PumpAndProbeBloom, true) => 12,
                _ => 1,
            };
            if offset + 1 < total_len {
                self.active_primitive = Some((prim, offset + 1, is_ep2));
            } else {
                self.active_primitive = None;
            }
            let dry_r = matches!(
                self.config.topology_mode,
                TopologyProbeMode::TrueStereo2To2 | TopologyProbeMode::DualInToMono2To1
            )
            .then(|| dry_l.clone());
            let aux_l = self.aux_for_non_active_48k(&dry_l);
            return (dry_l, dry_r, aux_l);
        }
        self.last_emitted_was_primitive = false;

        // Law 0.A: Frame-2 dynamic-level probe for EnvelopeSvf & 2-level (-18 / -6 dBFS) pole-zoom
        let eff_carrier_rms = if self.safeguards.ltv_regime == LtvRegime::EnvelopeSvf {
            self.laws.branch = ExcitationBranch::Branch1CNonlinearCascade;
            if self.laws.valid_frames.is_multiple_of(2) {
                0.085
            } else {
                0.28
            }
        } else if self.laws.valid_frames == 2 {
            (0.38 * self.laws.target_carrier_rms * self.safeguards.excitation_gain_scale).max(0.015)
        } else {
            self.laws.target_carrier_rms
        };

        // Otherwise synthesize disjoint-comb frame (with Law 4C staircase if sweeping or Master Vol)
        let weight_map: BTreeMap<usize, f64> = self.laws.bin_weights.iter().copied().collect();
        let bin_amps_l: Vec<(usize, f64)> = self
            .schedule_l
            .carrier_bins
            .iter()
            .map(|&b| (b, weight_map.get(&b).copied().unwrap_or(1.0)))
            .collect();
        let ped_amp = if self.laws.pedestal_enabled {
            0.35 * eff_carrier_rms
        } else {
            0.0
        };
        let comb_l = synthesize_cyclic_comb_frame_with_hint(
            &bin_amps_l,
            self.schedule_l.pedestal_bin,
            ped_amp,
            eff_carrier_rms,
            ceiling,
            &self.carrier_tail_l,
            Some(&self.phases_l),
        );
        self.carrier_tail_l = comb_l.carrier_tail_64;
        self.phases_l = comb_l.phases;
        self.last_carrier_crest_db = comb_l.carrier_crest_db;

        let mut dry_l = if self.config.is_master_volume_sweep
            || self.config.sweep_end_control_values.is_some()
        {
            apply_law4c_micro_staircase(&comb_l.samples, ceiling)
        } else {
            comb_l.samples
        };

        let bin_amps_r: Vec<(usize, f64)> = self
            .schedule_r
            .carrier_bins
            .iter()
            .map(|&b| (b, 1.0))
            .collect();
        let (dry_r, aux_l) = waveforms::synthesize_topology_companion_48k(
            self.config.topology_mode,
            self.laws.valid_frames,
            self.laws.disambiguated_2to1,
            &mut dry_l,
            &bin_amps_r,
            self.schedule_r.pedestal_bin,
            ped_amp,
            self.laws.target_carrier_rms,
            ceiling,
            &mut self.carrier_tail_r,
            &mut self.phases_r,
        );

        (dry_l, dry_r, aux_l)
    }

    /// Processes the previous `96.0 ms` wet response (`None` on Frame `0`) and synthesizes the next
    /// causal [`StimulusFrame`] in `< 1.5 ms`, returning `None` once converged and coreset/tail complete.
    pub fn next_frame(&mut self, last_wet_frame: Option<&StemResponse>) -> Option<StimulusFrame> {
        if self.phase == ControllerPhase::Finished {
            return None;
        }

        // Frame 0 startup
        let Some(resp) = last_wet_frame else {
            let (dl, dr, aux) = self.synthesize_next_active_48k();
            return Some(self.build_stimulus_frame_from_48k(dl, dr, aux));
        };

        let (wet_l_48k, wet_r_48k) = self.normalize_wet_to_48k(resp);
        self.recorded_wet_total_l.extend_from_slice(&wet_l_48k);

        match self.phase {
            ControllerPhase::ActiveLearning => {
                // Track Bloom Episode 1 / 2 probe windows if currently inside P5 probe frames
                if let Some((ActivePrimitiveKind::P5PumpAndProbeBloom, next_off, is_ep2)) =
                    self.active_primitive
                {
                    let pump_frames = if is_ep2 { 6 } else { 1 };
                    if next_off > pump_frames {
                        self.bloom_probe_acc.extend_from_slice(&wet_l_48k);
                    }
                } else if !self.bloom_probe_acc.is_empty() {
                    self.bloom_probe_acc.extend_from_slice(&wet_l_48k);
                    let telem = analyze_pump_and_probe_recovery(&self.bloom_probe_acc, None);
                    self.bloom_probe_acc.clear();
                    let need_ep2 = telem.trigger_episode_2
                        && !self.bloom_ep2_done
                        && self.laws.valid_frames + 12 <= self.config.max_frames;
                    self.bloom_telemetry = Some(telem);
                    if !self.bloom_ep1_done {
                        self.bloom_ep1_done = true;
                        if need_ep2 {
                            self.active_primitive =
                                Some((ActivePrimitiveKind::P5PumpAndProbeBloom, 0, true));
                        } else {
                            self.bloom_ep2_done = true;
                        }
                    } else {
                        self.bloom_ep2_done = true;
                    }
                }

                let decision = evaluate_frame_safeguards(
                    &mut self.safeguards,
                    self.emitted_frame_index.saturating_sub(1),
                    &self.last_dry_48k_l,
                    &wet_l_48k,
                    wet_r_48k.as_deref(),
                    &self.schedule_l,
                );

                if decision.discard_and_replay && self.consecutive_replays < 3 {
                    self.consecutive_replays += 1;
                    let scale = self.safeguards.excitation_gain_scale as f32;
                    let scale_buf = |b: &[f32]| -> Vec<f32> {
                        b.iter().map(|&s| (s * scale).clamp(-0.84, 0.84)).collect()
                    };
                    let replay_l = scale_buf(&self.last_dry_48k_l);
                    let replay_r = self.last_dry_48k_r.as_deref().map(scale_buf);
                    let replay_aux = self.last_aux_48k_l.as_deref().map(scale_buf);
                    return Some(
                        self.build_stimulus_frame_from_48k(replay_l, replay_r, replay_aux),
                    );
                }
                self.consecutive_replays = 0;

                // Law 0.E: OC-2 sub-harmonic divider handoff after <= 2 multisine frames
                if self.safeguards.oc2_glissando_handoff {
                    let slice = self.glissando_buf_l[0..ADAPTIVE_FRAME_SAMPLES].to_vec();
                    let aux = self.aux_for_non_active_48k(&slice);
                    self.phase = ControllerPhase::Oc2Glissando { gliss_frame_idx: 1 };
                    return Some(self.build_stimulus_frame_from_48k(slice, None, aux));
                }

                let dry_win_l = &self.last_dry_48k_l[self
                    .last_dry_48k_l
                    .len()
                    .saturating_sub(CYCLIC_WINDOW_SAMPLES)..];
                let wet_win_r = wet_r_48k
                    .as_deref()
                    .map(|wr| &wr[wr.len().saturating_sub(CYCLIC_WINDOW_SAMPLES)..]);
                let bloom_done =
                    self.bloom_ep1_done && (self.bloom_ep2_done || self.active_primitive.is_none());

                if self.last_emitted_was_primitive {
                    self.laws.valid_frames += 1;
                } else {
                    self.laws.update_from_valid_window(
                        dry_win_l,
                        &decision.clean_analysis_window,
                        wet_win_r,
                        &self.schedule_l,
                        self.safeguards.noise_floor_power,
                        self.config.topology_mode,
                        self.config.max_frames,
                        self.config.fixed_duration,
                        bloom_done,
                    );
                }

                // Schedule Theorem 4.8 Bloom Episode 1 or Theorem 4.7 D-optimal primitive
                if self.active_primitive.is_none()
                    && self.laws.branch == ExcitationBranch::Branch1CNonlinearCascade
                    && !self.laws.converged_4gate
                    && self.laws.valid_frames + 4 <= self.config.max_frames
                {
                    if !self.bloom_ep1_done && self.laws.valid_frames >= 3 {
                        self.active_primitive =
                            Some((ActivePrimitiveKind::P5PumpAndProbeBloom, 0, false));
                        self.laws.primitive_counts[4] += 1;
                    } else if self.laws.valid_frames >= 4
                        && self.laws.valid_frames.is_multiple_of(3)
                    {
                        let (prim, _) = self
                            .laws
                            .select_next_d_optimal_primitive(self.bloom_ep1_done);
                        let p_idx = ActivePrimitiveKind::ALL
                            .iter()
                            .position(|&k| k == prim)
                            .unwrap_or(1);
                        self.laws.primitive_counts[p_idx] += 1;
                        self.active_primitive = Some((prim, 0, false));
                    }
                }

                let bloom_in_flight =
                    self.active_primitive.is_some() || !self.bloom_probe_acc.is_empty();
                if (!bloom_in_flight && self.laws.converged_4gate)
                    || self.laws.valid_frames >= self.config.max_frames
                {
                    self.phase = ControllerPhase::Coreset {
                        coreset_frame_idx: 1,
                    };
                    return Some(self.emit_coreset_or_silence_48k(Some(0)));
                }

                let (dl, dr, aux) = self.synthesize_next_active_48k();
                Some(self.build_stimulus_frame_from_48k(dl, dr, aux))
            }

            ControllerPhase::Oc2Glissando { gliss_frame_idx } => {
                self.laws.valid_frames += 1;
                if gliss_frame_idx < 3 {
                    let start = gliss_frame_idx * ADAPTIVE_FRAME_SAMPLES;
                    let slice =
                        self.glissando_buf_l[start..start + ADAPTIVE_FRAME_SAMPLES].to_vec();
                    let aux = self.aux_for_non_active_48k(&slice);
                    self.phase = ControllerPhase::Oc2Glissando {
                        gliss_frame_idx: gliss_frame_idx + 1,
                    };
                    Some(self.build_stimulus_frame_from_48k(slice, None, aux))
                } else {
                    self.phase = ControllerPhase::Coreset {
                        coreset_frame_idx: 1,
                    };
                    Some(self.emit_coreset_or_silence_48k(Some(0)))
                }
            }

            ControllerPhase::Coreset { coreset_frame_idx } => {
                self.coreset_wet_l.extend_from_slice(&wet_l_48k);
                if let Some(ref wr) = wet_r_48k {
                    self.coreset_wet_r.extend_from_slice(wr);
                }
                if coreset_frame_idx < 3 {
                    self.phase = ControllerPhase::Coreset {
                        coreset_frame_idx: coreset_frame_idx + 1,
                    };
                    Some(self.emit_coreset_or_silence_48k(Some(coreset_frame_idx)))
                } else if self.config.max_tail_frames > 0 {
                    let dec = evaluate_live_decay_tail(
                        &wet_l_48k,
                        self.config.noise_floor_dbfs,
                        0,
                        self.config.max_tail_frames * ADAPTIVE_FRAME_SAMPLES,
                    );
                    if matches!(dec, DecayTailDecision::Extend { .. }) {
                        self.phase = ControllerPhase::DecayTail { tail_frame_idx: 1 };
                        Some(self.emit_coreset_or_silence_48k(None))
                    } else {
                        self.phase = ControllerPhase::Finished;
                        self.laws.convergence_pct = 100.0;
                        None
                    }
                } else {
                    self.phase = ControllerPhase::Finished;
                    self.laws.convergence_pct = 100.0;
                    None
                }
            }

            ControllerPhase::DecayTail { tail_frame_idx } => {
                let dec = evaluate_live_decay_tail(
                    &wet_l_48k,
                    self.config.noise_floor_dbfs,
                    tail_frame_idx * ADAPTIVE_FRAME_SAMPLES,
                    self.config.max_tail_frames * ADAPTIVE_FRAME_SAMPLES,
                );
                if matches!(dec, DecayTailDecision::Extend { .. })
                    && tail_frame_idx < self.config.max_tail_frames
                {
                    self.phase = ControllerPhase::DecayTail {
                        tail_frame_idx: tail_frame_idx + 1,
                    };
                    Some(self.emit_coreset_or_silence_48k(None))
                } else {
                    self.phase = ControllerPhase::Finished;
                    self.laws.convergence_pct = 100.0;
                    None
                }
            }

            ControllerPhase::Finished => None,
        }
    }

    /// Seals the shared `INPC` dry stimulus capsule (`24,640 B` mono / `48,640 B` stereo) and
    /// sub-sample-aligned [`VaultCaptureSpec`] (`STMC` `28,800 B` / `52,800 B`) from recorded audio.
    pub fn seal_inpc_and_vault_spec(
        &self,
        is_star_center: bool,
        controls_meta: BTreeMap<String, serde_json::Value>,
    ) -> Result<(Vec<u8>, VaultCaptureSpec), DnamError> {
        let raw_wet_l = if self.coreset_wet_l.len() >= CORESET_FP16_FRAMES_PER_STEM {
            &self.coreset_wet_l[..self.coreset_wet_l.len().min(3 * ADAPTIVE_FRAME_SAMPLES)]
        } else {
            &self.recorded_wet_total_l
        };
        let windows = if self.trajectory_windows.is_empty() {
            let u0 = self
                .config
                .control_values
                .get(usize::from(self.config.active_control_id))
                .copied()
                .unwrap_or(0.0);
            vec![StmcTrajectoryWindow {
                control_id: self.config.active_control_id,
                ctrl_kind: 0,
                active_opt: 0,
                u_phys: u0,
                window_weight: 1.0,
                emr_velocity: 0.0,
            }]
        } else {
            self.trajectory_windows.clone()
        };
        batch::seal_coresets_to_inpc_and_vault(
            &self.config.stem_id,
            is_star_center,
            &self.config.seed_256,
            &self.schedule_l.carrier_bins,
            self.laws.valid_frames,
            &self.coreset_dry_l,
            self.coreset_dry_r.as_deref(),
            raw_wet_l,
            (!self.coreset_wet_r.is_empty()).then_some(self.coreset_wet_r.as_slice()),
            windows,
            controls_meta,
        )
    }
}
