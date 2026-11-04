//! $3\text{ kHz}$ Parametric Control Bus, `ControlChannelRecord` & Anti-Zipper Smoother (`control`).
//!
//! Implements Subsystem 3 from `docs/dsp/parametric_control_bus.md`
//! and `docs/training/clamp_and_audit_registry.md`.

pub mod bus;
pub mod descriptor;
pub mod plane;
pub mod record;
pub mod reference;

pub use plane::{
    BoundedPerProducerIngress, CANONICAL_SUBDIVISION_RATIOS, ClockTrackerState, ControlEventStamp,
    EXACT_SQUARE_WORDS, EXACT_SUM_WORDS, ExactIntFactorAccumulator, ExactMagnitude,
    FDN_COPRIME_BASE_PRIMES, IngressDelivery, MAX_EXACT_CONTRIBUTORS, OrderedClockPulseTracker,
    SignedExactInt, TempoPhysicalInversionResult, TransportJumpPolicy,
    advance_q64_beat_accumulator, audit_phase1_clamp_range, audit_phase1_clamp_switch,
    blend_disjoint_interval_paths, clamp_head_spectral_gain_linear, control_stride_for_rate,
    control_tick_ratio, eval_3node_calibration_f64, eval_clamped_micro_ramp_block,
    eval_event_rate_aware_tracker, eval_q32_subdivision_phase, eval_tier_stage_dispatch,
    format_audit_prescription_report, host_frame_q64_increment, invert_tempo_physical_calibration,
    is_late_for_first_boundary, reconstruct_vst3_normalized_curve, retime_control_window,
    smooth_relock_q32_tick, subdivision_q32_to_phasor, subframe_4x_q64_offset,
    verify_coprime_fdn_pair, verify_dynamic_stability_excursion,
};

pub use bus::ControlBus;
pub use descriptor::{ConditionalClampRule, ControlClampDescriptor};
pub use record::{ControlChannelRecord, ControlLatentProjection, LatentAccumulator64B};
pub use reference::{
    eval_chebyshev_lfo_4harm_f64, eval_newton_lobatto_basis_f64, eval_rank4_fm_interaction_f64,
    eval_trust_region_softclip_f64,
};

/// Number of leading control slots captured in the fixed C-ABI [`crate::DnamTelemetrySnapshot`] preview array.
/// Note: TORA supports an arbitrary number of user controls $P \in [0, 255]$ via the unified 128-byte
/// [`ControlChannelRecord`] table and $\mathcal{O}(1)$ [`LatentAccumulator64B`] (`Theorem 3`), while
/// SVD head rank ($D_h \in \{4, 8\}$) and FM interaction rank ($R \in \{4, 8\}$) scale dynamically with
/// [`crate::Tier::svd_control_modes`] and [`crate::Tier::fm_interaction_rank`].
pub const TELEMETRY_CONTROL_SLOTS: usize = 16;

/// Anti-zipper convergence threshold: once $|u_p - u_{p,\text{target}}| < 10^{-6}$, the smoother snaps
/// to `u_target` and re-engages the static `0 MACs/sample` head cache.
pub const ANTI_ZIPPER_SNAP_EPS: f32 = 1e-6;

/// Default $3\text{ kHz}$ 1-pole anti-zipper smoothing pole ($\tau \approx 5\text{ ms} \implies \alpha \approx 0.9355$).
pub const DEFAULT_ANTI_ZIPPER_POLE: f32 = 0.9355;
