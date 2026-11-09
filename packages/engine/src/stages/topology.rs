//! Canonical v2.0 Stage 0 8-Port Router (WDW, Kaiser-Sinc Sub-Sample Alignment, 200 Hz TPT LR4
//! Crossover, Stereo-Linked & Aux-Key Detector), Stage IV Blended Quiet-Feedback Knee & Outer-Clamped
//! Loop Saturator, 256-D Feature Bus Slot Allocation, and Stage III ZOH First-Tap & 6-Point Lagrange
//! Difference Mappings (`src/stages/topology.rs`).
//!
//! Implements:
//! - `ref[dsp.router.signal-flow]`, `ref[dsp.router.ports]`, `ref[dsp.router.wdw-contract]`,
//!   `ref[dsp.router.return-to-send-isolation]`, `ref[dsp.router.parallel-alignment]`,
//!   `ref[dsp.router.lr4-crossover]`, `ref[dsp.router.mid-side-input]`,
//!   `ref[dsp.router.topology-dispatch]`, `ref[dsp.router.stereo-link-detector]`,
//!   `ref[dsp.router.ballistic-envelope-bank]`, `ref[dsp.router.modulation-bus]`
//! - `ref[dsp.stage1.pre-bus]`, `ref[dsp.stage2.commuted-cascade]`, `ref[dsp.stage3.cabinet-bank]`,
//!   `ref[dsp.stage4.spatial-tank]`, `ref[dsp.stage4.captured-freeze]`, `ref[dsp.stage4.dc-blocker]`,
//!   `ref[dsp.stage4.loop-saturator]`, `ref[dsp.stage4.quiet-feedback-knee]`
//! - `ref[dsp.head.mid-side-factorization]`, `ref[dsp.head.feature-bus-slots]`,
//!   `ref[dsp.head.tier-equivalence]`, `ref[dsp.head.lower-triangular]`
//! - `ref[dsp.zoh.router-detector]`, `ref[dsp.zoh.stage3-fir-onset]`

pub mod router_and_head;
pub mod zoh_fir_onset;

pub use router_and_head::{
    BALLISTIC_ENVELOPE_BANK_TAUS_SEC, ParallelAlignmentMode, STAGE4_MAX_G_LIN_BITS,
    STAGE4_MAX_V_CEIL_BITS, TptLr4Crossover, WdwDryTapMode, eval_min_latency_4tap_alignment,
    eval_stage_modulation_laws, eval_stage4_blended_feedback_knee,
    eval_stage4_blended_feedback_knee_fast, eval_stage4_loop_saturator,
    eval_stage4_loop_saturator_fast, eval_stereo_linked_aux_detector, eval_wdw_routing_outputs,
    pack_stage3_modal_block_features, prepare_ballistic_band_alphas,
    prepare_kaiser_sinc_alignment_weights, stage3_feature_bus_slot_ranges,
    validate_tensor_directory_slice,
};
pub use zoh_fir_onset::{
    GAIN_BOUNDED_FIRST_TAP_64_BITS, eval_bounce_6point_lagrange_tap,
    eval_live_first_tap_difference, prepare_bounce_6point_lagrange_tap,
    prepare_live_first_tap_difference_coeffs,
};
