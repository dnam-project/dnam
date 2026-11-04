"""$3\\text{ kHz}$ Parametric Control-Bus Reference Kernels (`Lobatto`, `Rank-4 FM`, `Chebyshev
LFO`, `Trust-Region Soft-Clip`) (`ref[sdk.python.dsp.reference-kernels]`).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dnam._arrays import coerce_f64_1d, coerce_f64_2d
from dnam._native import (
    native_eval_chebyshev_lfo,
    native_eval_chebyshev_lfo_slice,
    native_eval_lobatto_basis,
    native_eval_lobatto_basis_slice,
    native_eval_rank4_fm,
    native_eval_trust_region_softclip,
    native_eval_trust_region_softclip_slice,
)
from dnam.errors import ValidationError


def lobatto_basis(
    u: float | ArrayLike,
    anchors: tuple[float, float, float] = (0.0, -1.0, 1.0),
    *,
    degree: int = 2,
) -> tuple[float, float, float] | NDArray[np.float64]:
    """Evaluates Node-Anchored Newton/Lobatto basis polynomials (`ref[dsp.ctrl.basis.lobatto]`).

    Returns `(psi_1, psi_2, psi_3)` for scalar `u` or an `(N, 3)` `float64` array for 1D slice `u`.
    """
    if degree < 0 or degree > 3:
        raise ValidationError(f"lobatto_basis degree must be in 0..=3, got {degree}")
    anc = (float(anchors[0]), float(anchors[1]), float(anchors[2]))
    if isinstance(u, (int, float)):
        p1, p2, p3 = native_eval_lobatto_basis(float(u), anc, degree)
        return float(p1), float(p2), float(p3)
    u_arr = coerce_f64_1d(u, name="u")
    out = np.empty((u_arr.shape[0], 3), dtype=np.float64)
    native_eval_lobatto_basis_slice(u_arr, out, anc, degree)
    return out


def rank4_fm(
    features: ArrayLike,
    v_factors: ArrayLike,
) -> NDArray[np.float64]:
    """Evaluates the Rank-4 Factorization Machine interaction vector (`4-D` `float64` array)."""
    feat_arr = coerce_f64_1d(features, name="features")
    vf_2d = coerce_f64_2d(v_factors, expected_cols=4, name="v_factors")
    if vf_2d.shape[0] != feat_arr.shape[0]:
        raise ValidationError(
            f"v_factors rows ({vf_2d.shape[0]}) != features length ({feat_arr.shape[0]})"
        )
    res = native_eval_rank4_fm(
        feat_arr,
        vf_2d.reshape(feat_arr.shape[0] * 4),
    )
    return np.asarray(res, dtype=np.float64)


def chebyshev_lfo(
    cos_phi: float | ArrayLike,
) -> tuple[float, float, float, float] | NDArray[np.float64]:
    """Evaluates the 4-Harmonic Chebyshev LFO polynomials `(T_1, T_2, T_3, T_4)`.

    Returns a 4-tuple for scalar `cos_phi` or an `(N, 4)` `float64` array for 1D slice `cos_phi`.
    """
    if isinstance(cos_phi, (int, float)):
        t1, t2, t3, t4 = native_eval_chebyshev_lfo(float(cos_phi))
        return float(t1), float(t2), float(t3), float(t4)
    c_arr = coerce_f64_1d(cos_phi, name="cos_phi")
    out = np.empty((c_arr.shape[0], 4), dtype=np.float64)
    native_eval_chebyshev_lfo_slice(c_arr, out)
    return out


def trust_region_softclip(
    z: float | ArrayLike,
    *,
    mh: float = 2.5,
) -> float | NDArray[np.float64]:
    """Evaluates the cubic trust-region soft-clipper `M_h * S_3(z / M_h)`
    (`ref[dsp.ctrl.op-n.trust-clip]`).
    """
    if mh <= 0.0:
        raise ValidationError(f"trust_region_softclip mh must be positive, got {mh}")
    if isinstance(z, (int, float)):
        return float(native_eval_trust_region_softclip(float(z), float(mh)))
    z_arr = coerce_f64_1d(z, name="z")
    out = np.empty(z_arr.shape[0], dtype=np.float64)
    native_eval_trust_region_softclip_slice(z_arr, out, float(mh))
    return out


__all__ = [
    "chebyshev_lfo",
    "lobatto_basis",
    "rank4_fm",
    "trust_region_softclip",
]
