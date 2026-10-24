"""Differentiable PyTorch DSP layers & `dnam_engine.dsp` Reference Bridge (`CL-01`..`CL-11`)."""

from __future__ import annotations

from dnam_trainer.dsp.modulated_ufdn import (
    DifferentiableModulatedUfdn,
    coercive_lyapunov_loop_step_torch,
    cubic_lagrange_read_torch,
    cubic_lagrange_weights_torch,
    dual_pointer_switch_crossfade_torch,
    givens_householder_matrix_torch,
)
from dnam_trainer.dsp.poly_adaa import (
    DifferentiablePolyAdaa,
    PolyAdaaState,
    clamp_pre_activation_torch,
    poly_adaa_algebraic_torch,
    poly_adaa_waveshaper_torch,
    resting_dc_equilibrium,
)
from dnam_trainer.dsp.selective_rglru import (
    DifferentiableSelectiveRgLru,
    algebraic_unit_rotor_torch,
    h2_input_norm_gain,
    log_octave_pole_init,
    parallel_associative_scan_complex,
)

__all__ = [
    "DifferentiableModulatedUfdn",
    "DifferentiablePolyAdaa",
    "DifferentiableSelectiveRgLru",
    "PolyAdaaState",
    "algebraic_unit_rotor_torch",
    "clamp_pre_activation_torch",
    "coercive_lyapunov_loop_step_torch",
    "cubic_lagrange_read_torch",
    "cubic_lagrange_weights_torch",
    "dual_pointer_switch_crossfade_torch",
    "givens_householder_matrix_torch",
    "h2_input_norm_gain",
    "log_octave_pole_init",
    "parallel_associative_scan_complex",
    "poly_adaa_algebraic_torch",
    "poly_adaa_waveshaper_torch",
    "resting_dc_equilibrium",
]
