"""Tests for `dnam.dsp` vectorized TORA reference kernels (`op_t`, `op_m`, `op_n`, `control_bus`,
`mixing`).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dnam import dsp
from dnam.errors import ValidationError


@given(mu=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=25, deadline=None)
def test_op_t_and_mixing_kernels_scalar_slice_equivalence(mu: float) -> None:
    """verify[sdk.python.dsp.reference-kernels]
    Verifies Op-T Catmull-Rom, coercive tank step/slice, Cubic Lagrange partition of unity,
    and 8ch/16ch Givens-Householder energy preservation (orthogonality).
    """
    # Catmull-Rom scalar vs slice
    s_scalar = dsp.catmull_rom_hermite(-0.5, 0.2, 0.8, 0.1, mu)
    s_slice = dsp.catmull_rom_hermite([-0.5], [0.2], [0.8], [0.1], [mu])
    assert isinstance(s_scalar, float)
    assert isinstance(s_slice, np.ndarray)
    assert abs(s_scalar - float(s_slice[0])) == 0.0

    # Cubic Lagrange weights sum to 1.0 (partition of unity)
    w_scalar = dsp.cubic_lagrange_weights(mu)
    w_slice = dsp.cubic_lagrange_weights([mu])
    assert isinstance(w_scalar, tuple)
    assert isinstance(w_slice, np.ndarray)
    assert abs(sum(w_scalar) - 1.0) < 1e-12
    np.testing.assert_allclose(w_slice[0], w_scalar, rtol=0.0, atol=0.0)

    # Op-T coercive step vs tank_slice
    sig = np.array([0.2, -0.4, 0.6, -0.1], dtype=np.float64)
    out_slice = dsp.tank_slice(sig, g_lin=0.85, sigma_knee=1.0, v_ceil=4.0, r_dc=0.995)
    dx, dy = 0.0, 0.0
    for i, x in enumerate(sig):
        y, dx, dy = dsp.coercive_step(
            float(x),
            g_lin=0.85,
            sigma_knee=1.0,
            v_ceil=4.0,
            r_dc=0.995,
            dc_x_prev=dx,
            dc_y_prev=dy,
        )
        assert abs(y - float(out_slice[i])) == 0.0

    # Givens-Householder 8ch and 16ch energy preservation (orthogonal matrix ||Ux||_2 == ||x||_2)
    psi = 0.37
    cp, sp = math.cos(psi), math.sin(psi)
    x8 = np.linspace(-1.0, 1.0, 8, dtype=np.float64)
    y8 = dsp.givens_householder_8ch(x8, cos_psi=cp, sin_psi=sp)
    assert abs(float(np.sum(x8 * x8)) - float(np.sum(y8 * y8))) < 1e-12

    x16 = np.linspace(-1.0, 1.0, 16, dtype=np.float64)
    y16 = dsp.givens_householder_16ch(x16, cos_psi=cp, sin_psi=sp)
    assert abs(float(np.sum(x16 * x16)) - float(np.sum(y16 * y16))) < 1e-12


def test_op_m_op_n_and_control_bus_kernels() -> None:
    """verify[sdk.python.dsp.reference-kernels]
    Verifies Op-M modal step/slice & unit_rotor, Op-N PolyADAA step/slice, and Control-Bus
    Lobatto, Rank-4 FM, Chebyshev LFO, and Trust-Region Soft-Clipper kernels.
    """
    sig = np.linspace(-0.8, 0.8, 32, dtype=np.float64)

    # Op-M modal_step vs modal_slice
    m_slice, _, _ = dsp.modal_slice(sig, reset=True)
    zr, zi = 0.0, 0.0
    for i, x in enumerate(sig):
        y, zr, zi = dsp.modal_step(float(x), z_re=zr, z_im=zi)
        assert abs(y - float(m_slice[i])) == 0.0

    # unit_rotor scalar vs slice
    phis = np.linspace(-3.0, 3.0, 16, dtype=np.float64)
    rot_slice = dsp.unit_rotor(phis)
    assert isinstance(rot_slice, np.ndarray)
    assert rot_slice.shape == (2, 16)
    for i, p in enumerate(phis):
        rot_pt = dsp.unit_rotor(float(p))
        assert isinstance(rot_pt, tuple)
        c, s = rot_pt
        assert abs(c - float(rot_slice[0, i])) == 0.0
        assert abs(s - float(rot_slice[1, i])) == 0.0
        assert abs(math.hypot(c, s) - 1.0) < 1e-5

    # Op-N PolyADAA algebraic, step, and slice
    alg = dsp.poly_adaa_algebraic(0.2, 0.2, beta=1.0, delta=0.0)
    assert abs(alg - 0.2 * (1.5 - 0.5 * 0.04)) < 1e-12
    p_slice, _, _, _ = dsp.poly_adaa_slice(sig, b_conv=0.0, reset=True)
    up, ap, mdc = 0.0, 0.0, 0.0
    for i, x in enumerate(sig):
        y, up, ap, mdc = dsp.poly_adaa_step(float(x), b_conv=0.0, u_prev=up, a_prev=ap, m_dc=mdc)
        assert abs(y - float(p_slice[i])) == 0.0

    # Control Bus: Lobatto, Rank-4 FM, Chebyshev LFO, Trust-Region Soft-Clipper
    u_vals = np.linspace(-1.0, 1.0, 9, dtype=np.float64)
    lob_2d = dsp.lobatto_basis(u_vals, (0.0, -1.0, 1.0), degree=3)
    assert isinstance(lob_2d, np.ndarray)
    assert lob_2d.shape == (9, 3)
    lob_0 = dsp.lobatto_basis(float(u_vals[0]), (0.0, -1.0, 1.0), degree=3)
    assert isinstance(lob_0, tuple)
    np.testing.assert_allclose(lob_2d[0], lob_0, rtol=0.0, atol=0.0)

    x_feat = np.array([0.5, -0.25, 0.75, 1.0], dtype=np.float64)
    v_mat = np.ones((4, 4), dtype=np.float64) * 0.5
    fm = dsp.rank4_fm(x_feat, v_mat)
    assert fm.shape == (4,)

    cheb_2d = dsp.chebyshev_lfo(u_vals)
    assert isinstance(cheb_2d, np.ndarray)
    assert cheb_2d.shape == (9, 4)
    cheb_0 = dsp.chebyshev_lfo(float(u_vals[0]))
    np.testing.assert_allclose(cheb_2d[0], cheb_0, rtol=0.0, atol=0.0)

    z_vals = np.array([-10.0, 0.0, 10.0], dtype=np.float64)
    tr_slice = dsp.trust_region_softclip(z_vals, mh=2.5)
    assert isinstance(tr_slice, np.ndarray)
    assert abs(float(tr_slice[0]) - (-2.5)) < 1e-12
    assert abs(float(tr_slice[2]) - 2.5) < 1e-12
    assert abs(float(dsp.trust_region_softclip(10.0, mh=2.5)) - 2.5) < 1e-12

    # Validation errors on bad shapes/params
    with pytest.raises(ValidationError):
        dsp.lobatto_basis(0.0, degree=5)
    with pytest.raises(ValidationError):
        dsp.lobatto_basis(np.zeros((2, 2)))
    with pytest.raises(ValidationError):
        dsp.rank4_fm(np.zeros(4), np.zeros(4))
    with pytest.raises(ValidationError):
        dsp.rank4_fm(np.zeros(4), np.zeros((4, 3)))
    with pytest.raises(ValidationError):
        dsp.rank4_fm(np.zeros(4), np.zeros((2, 4)))
    with pytest.raises(ValidationError):
        dsp.trust_region_softclip(0.0, mh=-1.0)
    with pytest.raises(ValidationError):
        dsp.givens_householder_8ch(np.zeros(4))
    with pytest.raises(ValidationError):
        dsp.givens_householder_16ch(np.zeros(4))
    with pytest.raises(ValidationError):
        dsp.givens_householder_16ch(np.zeros(16), cross_cos=np.zeros(4))
