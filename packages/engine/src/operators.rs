//! TORA Zero-FDIV SIMD Operator Kernels (`Op-T`, `Op-M`, `Op-N`, and `reference`).
//!
//! Subsystems:
//! - [`op_t`]: Universal Transport, Granular Windowing & All-Pass Diffusion Operator.
//! - [`op_m`]: Universal Symplectic Modal, Hilbert, Sub-Harmonic & Kautz Operator.
//! - [`op_n`]: Universal Nonlinear, Hysteretic & Commuted Cascade Operator.
//! - [`reference`]: Bit-accurate scalar `f64` reference implementations.
//!
//! Direct per-sample calls to real-time kernels require a caller-held
//! [`crate::FtzDazGuard`] for the enclosing batch.

pub mod op_m;
pub mod op_n;
pub mod op_t;
pub mod reference;

pub use op_m::{
    HilbertSsbLattice10th, KautzModalBank, SubHarmonicTracker, SymplecticModal4Lane,
    algebraic_unit_rotor_cl03, fast_rsqrt_nr2,
};
pub use op_n::{
    MULTI_DRIVE_SCALES, MultiDrivePolyAdaaChannel, QuadratureHysteresis2D, ZohBitcrusher,
    c2_wavefolder, eval_chebyshev_4harm, eval_lobatto_basis_3node, poly_adaa_algebraic_eval,
    trust_region_softclip_4d,
};
pub use op_t::{
    BallisticDuckingVca, OP_T_FULL_RING_CAPACITY, OP_T_LITE_RING_CAPACITY,
    OP_T_ULTRA_RING_CAPACITY, OpTDelayLine, SCHROEDER_COPRIME_DELAYS, catmull_rom_hermite_cubic,
    hermite_hann_partition_of_unity, raised_cosine_barberpole_3phase,
};
pub use reference::{
    catmull_rom_hermite_f64, cubic_lagrange_weights_f64, eval_chebyshev_lfo_4harm_f64,
    eval_newton_lobatto_basis_f64, eval_rank4_fm_interaction_f64, eval_trust_region_softclip_f64,
    givens_householder_mix_8ch_f64, givens_householder_mix_16ch_f64, op_m_modal_step_f64,
    op_t_coercive_step_f64, op_t_tank_slice_f64, poly_adaa_algebraic_f64, poly_adaa_slice_f64,
    poly_adaa_step_f64,
};
