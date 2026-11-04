"""Vectorized TORA Reference DSP Kernels (`op_t`, `op_m`, `op_n`, `control_bus`, `mixing`)
(`ref[sdk.python.dsp.reference-kernels]`).
"""

from dnam.dsp import control_bus, mixing, op_m, op_n, op_t
from dnam.dsp.control_bus import (
    chebyshev_lfo,
    lobatto_basis,
    rank4_fm,
    trust_region_softclip,
)
from dnam.dsp.mixing import (
    cubic_lagrange_weights,
    givens_householder_8ch,
    givens_householder_16ch,
)
from dnam.dsp.op_m import modal_slice, modal_step, unit_rotor
from dnam.dsp.op_n import (
    poly_adaa_algebraic,
    poly_adaa_slice,
    poly_adaa_step,
)
from dnam.dsp.op_t import catmull_rom_hermite, coercive_step, tank_slice

__all__ = [
    "catmull_rom_hermite",
    "chebyshev_lfo",
    "coercive_step",
    "control_bus",
    "cubic_lagrange_weights",
    "givens_householder_8ch",
    "givens_householder_16ch",
    "lobatto_basis",
    "mixing",
    "modal_slice",
    "modal_step",
    "op_m",
    "op_n",
    "op_t",
    "poly_adaa_algebraic",
    "poly_adaa_slice",
    "poly_adaa_step",
    "rank4_fm",
    "tank_slice",
    "trust_region_softclip",
    "unit_rotor",
]
