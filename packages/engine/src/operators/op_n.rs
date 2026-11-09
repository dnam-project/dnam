//! Op-N: Universal Nonlinear, Hysteretic & Commuted Cascade Operator.
//!
//! Canonical Reference: `docs/specs/dsp/op_n_commuted_cascade.md` (`ref[dsp.op-n.polyadaa.midpoint]`, `ref[dsp.op-n.polyadaa.bspline-ac]`, `ref[dsp.op-n.ctrl.lobatto-basis]`, `ref[dsp.op-n.ctrl.trust-clip]`).

pub mod commuted_and_ctrl;
pub mod polyadaa;
pub mod wavefolder_and_bitcrusher;

pub use commuted_and_ctrl::{
    eval_chebyshev_4harm, eval_commuted_vs_uncommuted_lti_cascade_f64, eval_lobatto_basis_3node,
    eval_stage2_stereo_coupling_16d, fold_stage2_latent_weights, trust_region_softclip_4d,
};
pub use polyadaa::{
    MULTI_DRIVE_SCALES, MultiDrivePolyAdaaChannel, eval_saturated_polyadaa_rail_crossing_f64,
    poly_adaa_algebraic_eval,
};
pub use wavefolder_and_bitcrusher::{
    FixedHzBitcrusher, QuadratureHysteresis2D, ZohBitcrusher, c2_wavefolder,
    eval_adaa_cubic_c2_wavefolder_f64, eval_adaa_cubic_c2_wavefolder_odd_f32,
    hertz_phase_parameters, hertz_phase_step, quantize_bitcrusher_nearest_ties_pos_inf,
    rescale_bitcrusher_phase_u128,
};
