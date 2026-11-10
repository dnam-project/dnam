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
    HILBERT_H1_ANCHORS_15, HilbertH1Lattice6Sec, HilbertSsbLattice10th, KautzModalBank, OpMBackend,
    SubHarmonicTracker, SymplecticModal4Lane, T1DcBlocker, T1SubHarmonicTracker,
    algebraic_unit_rotor_op_m, compute_t1_dc_blocker_coeffs_bits,
    convert_stage3_kautz_pole_for_rate, encode_hilbert_h1_anchor_table_780b,
    eval_stage3_sweep_contraction, eval_tracker_half_angle_pair, fast_rsqrt_nr2,
    init_log_octave_poles_h2, interpolate_hilbert_h1_between_anchors_bits,
    interpolate_hilbert_h1_profile_bits, normalize_tracker_phasor_exponent_split,
    tracker_crosses_branch_cut, tracker_exp_scales_scalar_f32,
    validate_and_eval_stage3_radial_gate,
};
pub use op_n::{
    FixedHzBitcrusher, MULTI_DRIVE_SCALES, MultiDrivePolyAdaaChannel, OpNBackend,
    QuadratureHysteresis2D, ZohBitcrusher, c2_wavefolder, eval_adaa_cubic_c2_wavefolder_f64,
    eval_chebyshev_4harm, eval_commuted_vs_uncommuted_lti_cascade_f64, eval_lobatto_basis_3node,
    eval_saturated_polyadaa_rail_crossing_f64, eval_stage2_stereo_coupling_16d,
    fold_stage2_latent_weights, hertz_phase_parameters, hertz_phase_step,
    poly_adaa_algebraic_4ch_vec4, poly_adaa_algebraic_eval,
    quantize_bitcrusher_nearest_ties_pos_inf, rescale_bitcrusher_phase_u128,
    trust_region_softclip_4d,
};
pub use op_t::{
    BallisticDuckingVca, FP16_DECAY_GAIN_BITS, OP_T_FULL_RING_CAPACITY, OP_T_LITE_RING_CAPACITY,
    OP_T_ULTRA_RING_CAPACITY, OpTBackend, OpTBlockMaxTreeGuard, OpTDelayLine,
    PartitionedSchroederDiffuserBank, SCHROEDER_COPRIME_DELAYS, admit_op_t_memory_budget,
    bounded_q32_reader_position, bspline_barberpole_3phase, catmull_rom_hermite_cubic,
    convert_schroeder_delay_for_rate, cubic_lagrange_weights_4ch_vec4,
    directed_fp16_decay_write_bits, eval_m2_pitch_grain_delays, hermite_hann_partition_of_unity,
    op_t_branchless_next_write, op_t_branchless_wrap_index, op_t_fractional_history_capacity,
    q32_decode_reader_position, raised_cosine_barberpole_3phase,
};
pub use reference::{
    catmull_rom_hermite_f64, cubic_lagrange_weights_f64, eval_chebyshev_lfo_4harm_f64,
    eval_newton_lobatto_basis_f64, eval_rank4_fm_interaction_f64, eval_trust_region_softclip_f64,
    givens_householder_mix_8ch_f64, givens_householder_mix_16ch_f64, op_m_modal_step_f64,
    op_t_coercive_step_f64, op_t_tank_slice_f64, poly_adaa_algebraic_f64, poly_adaa_slice_f64,
    poly_adaa_step_f64,
};
