"""`Op-T` Transport, Catmull-Rom Cubic Hermite & Coercive Tank Reference Kernels
(`ref[sdk.python.dsp.reference-kernels]`).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dnam._arrays import coerce_f64_1d
from dnam._native import (
    native_catmull_rom_hermite,
    native_catmull_rom_hermite_slice,
    native_op_t_coercive_step,
    native_op_t_tank_slice,
)


def catmull_rom_hermite(
    s_m1: float | ArrayLike,
    s_0: float | ArrayLike,
    s_1: float | ArrayLike,
    s_2: float | ArrayLike,
    mu: float | ArrayLike,
) -> float | NDArray[np.float64]:
    """Evaluates 4-tap Catmull-Rom cubic Hermite fractional delay interpolation.

    Accepts either scalar `float` arguments or 1D `ArrayLike` slices.
    """
    if (
        isinstance(s_m1, (int, float))
        and isinstance(s_0, (int, float))
        and isinstance(s_1, (int, float))
        and isinstance(s_2, (int, float))
        and isinstance(mu, (int, float))
    ):
        return float(
            native_catmull_rom_hermite(
                float(s_m1),
                float(s_0),
                float(s_1),
                float(s_2),
                float(mu),
            )
        )
    a_m1 = coerce_f64_1d(s_m1, name="s_m1")
    a_0 = coerce_f64_1d(s_0, name="s_0")
    a_1 = coerce_f64_1d(s_1, name="s_1")
    a_2 = coerce_f64_1d(s_2, name="s_2")
    a_mu = coerce_f64_1d(mu, name="mu")
    out = np.empty(a_0.shape[0], dtype=np.float64)
    native_catmull_rom_hermite_slice(a_m1, a_0, a_1, a_2, a_mu, out)
    return out


def coercive_step(
    u_in: float,
    *,
    g_lin: float = 0.85,
    sigma_knee: float = 1.0,
    v_ceil: float = 4.0,
    r_dc: float = 0.995,
    dc_x_prev: float = 0.0,
    dc_y_prev: float = 0.0,
) -> tuple[float, float, float]:
    """Single-sample Op-T coercive-saturating Lyapunov loop step, returning `(s_next, dc_x_prev,
    dc_y_prev)`.
    """
    s_next, nx, ny = native_op_t_coercive_step(
        float(u_in),
        float(g_lin),
        float(sigma_knee),
        float(v_ceil),
        float(r_dc),
        float(dc_x_prev),
        float(dc_y_prev),
    )
    return float(s_next), float(nx), float(ny)


def tank_slice(
    signal: ArrayLike,
    *,
    g_lin: float = 0.85,
    sigma_knee: float = 1.0,
    v_ceil: float = 4.0,
    r_dc: float = 0.995,
    out: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Processes a 1D `float64` audio slice through the reference Op-T coercive tank loop."""
    inp = coerce_f64_1d(signal, name="signal")
    if out is not None:
        if np.shares_memory(inp, out):
            inp = inp.copy()
        target = out
    else:
        target = np.empty(inp.shape[0], dtype=np.float64)
    native_op_t_tank_slice(
        inp,
        target,
        float(g_lin),
        float(sigma_knee),
        float(v_ceil),
        float(r_dc),
    )
    return target


__all__ = [
    "catmull_rom_hermite",
    "coercive_step",
    "tank_slice",
]
