"""Cubic Lagrange Interpolation & Orthogonal Givens/Householder Mixing Reference Kernels
(`ref[sdk.python.dsp.reference-kernels]`).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dnam._arrays import coerce_f64_1d
from dnam._native import (
    native_cubic_lagrange_weights,
    native_cubic_lagrange_weights_slice,
    native_givens_householder_8ch,
    native_givens_householder_16ch,
)
from dnam.errors import ValidationError


def cubic_lagrange_weights(
    mu: float | ArrayLike,
) -> tuple[float, float, float, float] | NDArray[np.float64]:
    """Evaluates 4-tap Cubic Lagrange fractional delay weights `(w_{-1}, w_0, w_1, w_2)`.

    Returns a 4-tuple for scalar `mu` or an `(N, 4)` `float64` array for 1D slice `mu`.
    """
    if isinstance(mu, (int, float)):
        w0, w1, w2, w3 = native_cubic_lagrange_weights(float(mu))
        return float(w0), float(w1), float(w2), float(w3)
    m_arr = coerce_f64_1d(mu, name="mu")
    out = np.empty((m_arr.shape[0], 4), dtype=np.float64)
    native_cubic_lagrange_weights_slice(m_arr, out)
    return out


def givens_householder_8ch(
    x: ArrayLike,
    *,
    cos_psi: float = 1.0,
    sin_psi: float = 0.0,
) -> NDArray[np.float64]:
    """Applies the 8-channel Orthogonal Givens + Householder mixing matrix
    (`ref[dsp.stage4.spatial-tank]`).
    """
    arr = coerce_f64_1d(x, name="x")
    if arr.shape[0] != 8:
        raise ValidationError(f"givens_householder_8ch requires 8 elements, got {arr.shape[0]}")
    vec8 = (
        float(arr[0]),
        float(arr[1]),
        float(arr[2]),
        float(arr[3]),
        float(arr[4]),
        float(arr[5]),
        float(arr[6]),
        float(arr[7]),
    )
    res = native_givens_householder_8ch(vec8, float(cos_psi), float(sin_psi))
    return np.asarray(res, dtype=np.float64)


def givens_householder_16ch(
    x: ArrayLike,
    *,
    cos_psi: float = 1.0,
    sin_psi: float = 0.0,
    cross_cos: ArrayLike | None = None,
    cross_sin: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Applies the 16-channel Orthogonal Givens + Householder mixing matrix
    (`ref[dsp.stage4.spatial-tank]`).
    """
    arr = coerce_f64_1d(x, name="x")
    if arr.shape[0] != 16:
        raise ValidationError(f"givens_householder_16ch requires 16 elements, got {arr.shape[0]}")
    c_arr = (
        coerce_f64_1d(cross_cos, name="cross_cos")
        if cross_cos is not None
        else np.ones(8, dtype=np.float64)
    )
    s_arr = (
        coerce_f64_1d(cross_sin, name="cross_sin")
        if cross_sin is not None
        else np.zeros(8, dtype=np.float64)
    )
    if c_arr.shape[0] != 8 or s_arr.shape[0] != 8:
        raise ValidationError("cross_cos and cross_sin must each have 8 elements")
    vec16 = (
        float(arr[0]),
        float(arr[1]),
        float(arr[2]),
        float(arr[3]),
        float(arr[4]),
        float(arr[5]),
        float(arr[6]),
        float(arr[7]),
        float(arr[8]),
        float(arr[9]),
        float(arr[10]),
        float(arr[11]),
        float(arr[12]),
        float(arr[13]),
        float(arr[14]),
        float(arr[15]),
    )
    cc8 = (
        float(c_arr[0]),
        float(c_arr[1]),
        float(c_arr[2]),
        float(c_arr[3]),
        float(c_arr[4]),
        float(c_arr[5]),
        float(c_arr[6]),
        float(c_arr[7]),
    )
    cs8 = (
        float(s_arr[0]),
        float(s_arr[1]),
        float(s_arr[2]),
        float(s_arr[3]),
        float(s_arr[4]),
        float(s_arr[5]),
        float(s_arr[6]),
        float(s_arr[7]),
    )
    res = native_givens_householder_16ch(
        vec16,
        float(cos_psi),
        float(sin_psi),
        cc8,
        cs8,
    )
    return np.asarray(res, dtype=np.float64)


__all__ = [
    "cubic_lagrange_weights",
    "givens_householder_8ch",
    "givens_householder_16ch",
]
