"""`Op-N` Division-Free Algebraic `PolyADAA` Reference Kernels
(`ref[sdk.python.dsp.reference-kernels]`).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dnam._arrays import coerce_f64_1d
from dnam._native import (
    native_poly_adaa_algebraic,
    native_poly_adaa_slice,
    native_poly_adaa_step,
)


def poly_adaa_algebraic(
    u_prev: float,
    u_curr: float,
    *,
    beta: float = 1.0,
    delta: float = 0.0,
) -> float:
    """Division-free algebraic `PolyADAA` segment average (`ref[dsp.op-n.polyadaa.midpoint]`)."""
    return float(
        native_poly_adaa_algebraic(
            float(u_prev),
            float(u_curr),
            float(beta),
            float(delta),
        )
    )


def poly_adaa_step(
    x: float,
    *,
    b_conv: float = 0.0,
    alpha: float = 0.25,
    beta: float = 1.0,
    delta: float = 0.1,
    r_ac: float = 0.9993,
    u_prev: float = 0.0,
    a_prev: float = 0.0,
    m_dc: float = 0.0,
) -> tuple[float, float, float, float]:
    """Single-sample `PolyADAA` step returning `(y, next_u_prev, next_a_prev, next_m_dc)`."""
    y, nu, na, nm = native_poly_adaa_step(
        float(x),
        float(b_conv),
        float(alpha),
        float(beta),
        float(delta),
        float(r_ac),
        float(u_prev),
        float(a_prev),
        float(m_dc),
    )
    return float(y), float(nu), float(na), float(nm)


def poly_adaa_slice(
    signal: ArrayLike,
    *,
    b_conv: float = 0.0,
    alpha: float = 0.25,
    beta: float = 1.0,
    delta: float = 0.1,
    r_ac: float = 0.9993,
    u_prev: float = 0.0,
    a_prev: float = 0.0,
    m_dc: float = 0.0,
    reset: bool = True,
    out: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], float, float, float]:
    """Processes a 1D `float64` audio slice through `PolyADAA`, returning `(output, next_u_prev,
    next_a_prev, next_m_dc)`.
    """
    inp = coerce_f64_1d(signal, name="signal")
    if out is not None:
        if np.shares_memory(inp, out):
            inp = inp.copy()
        target = out
    else:
        target = np.empty(inp.shape[0], dtype=np.float64)
    nu, na, nm = native_poly_adaa_slice(
        inp,
        target,
        float(b_conv),
        float(alpha),
        float(beta),
        float(delta),
        float(r_ac),
        float(u_prev),
        float(a_prev),
        float(m_dc),
        reset,
    )
    return target, float(nu), float(na), float(nm)


__all__ = [
    "poly_adaa_algebraic",
    "poly_adaa_slice",
    "poly_adaa_step",
]
