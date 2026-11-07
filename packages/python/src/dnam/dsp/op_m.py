"""`Op-M` Symplectic Modal Step & Contractive Rotor Reference Kernels
(`ref[sdk.python.dsp.reference-kernels]`).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dnam._arrays import coerce_f64_1d
from dnam._native import (
    native_op_m_modal_slice,
    native_op_m_modal_step,
    native_unit_rotor,
    native_unit_rotor_slice,
)


def modal_step(
    x: float,
    *,
    r_low: float = 0.95,
    dr_span: float = 0.04,
    w_sel: float = 0.5,
    b_gamma_norm: float = 0.1,
    c_real: float = 1.0,
    c_imag: float = 0.0,
    u_rot_cos: float = 0.99,
    u_rot_sin: float = 0.14106735979665894,
    z_re: float = 0.0,
    z_im: float = 0.0,
) -> tuple[float, float, float]:
    """Single-sample Radial-Polar `Op-M` modal step returning `(y, next_z_re, next_z_im)`."""
    y, n_re, n_im = native_op_m_modal_step(
        float(x),
        float(r_low),
        float(dr_span),
        float(w_sel),
        float(b_gamma_norm),
        float(c_real),
        float(c_imag),
        float(u_rot_cos),
        float(u_rot_sin),
        float(z_re),
        float(z_im),
    )
    return float(y), float(n_re), float(n_im)


def modal_slice(
    signal: ArrayLike,
    *,
    r_low: float = 0.95,
    dr_span: float = 0.04,
    w_sel: float = 0.5,
    b_gamma_norm: float = 0.1,
    c_real: float = 1.0,
    c_imag: float = 0.0,
    u_rot_cos: float = 0.99,
    u_rot_sin: float = 0.14106735979665894,
    z_re: float = 0.0,
    z_im: float = 0.0,
    reset: bool = True,
    out: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], float, float]:
    """Processes a 1D `float64` audio slice through the `Op-M` modal resonator, returning `(output,
    next_z_re, next_z_im)`.
    """
    inp = coerce_f64_1d(signal, name="signal")
    if out is not None:
        if np.shares_memory(inp, out):
            inp = inp.copy()
        target = out
    else:
        target = np.empty(inp.shape[0], dtype=np.float64)
    n_re, n_im = native_op_m_modal_slice(
        inp,
        target,
        float(r_low),
        float(dr_span),
        float(w_sel),
        float(b_gamma_norm),
        float(c_real),
        float(c_imag),
        float(u_rot_cos),
        float(u_rot_sin),
        float(z_re),
        float(z_im),
        reset,
    )
    return target, float(n_re), float(n_im)


def unit_rotor(
    phi: float | ArrayLike,
) -> tuple[float, float] | NDArray[np.float64]:
    """Evaluates the contractive unit-circle rotor `(cos_phi, sin_phi)`.

    Returns `(cos_phi, sin_phi)` for scalar `phi` or a `(2, N)` `float64` array for 1D slice `phi`.
    """
    if isinstance(phi, (int, float)):
        c, s = native_unit_rotor(float(phi))
        return float(c), float(s)
    p_arr = coerce_f64_1d(phi, name="phi")
    out = np.empty((2, p_arr.shape[0]), dtype=np.float64)
    native_unit_rotor_slice(p_arr, out)
    return out


__all__ = [
    "modal_slice",
    "modal_step",
    "unit_rotor",
]
