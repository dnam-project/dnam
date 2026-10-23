"""Unit and property-based tests for `dnam_engine.dsp` (`PolyAdaa`, `SelectiveRgLru`,
`ModulatedUfdn`, `unit_rotor`, `lagrange_weights`, `givens_householder_mix`)."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

from dnam_engine import DnamValidationError, dsp
from dnam_engine.schema import (
    MufdnParams,
    MufdnState,
    PolyAdaaParams,
    PolyAdaaState,
    RotorStep,
    SelectiveRgLruParams,
    SelectiveRgLruState,
)
from strategies import st_audio_1d_f64


@given(x=st_audio_1d_f64(min_len=32, max_len=128))
def test_poly_adaa_streaming_and_dc_equilibrium(x: NDArray[np.float64]) -> None:
    params = PolyAdaaParams(b_conv=0.35, alpha=1.0, beta=0.65, delta=0.25, r_ac=0.9993)
    adaa = dsp.PolyAdaa(params)
    assert adaa.params == params

    # Silence with non-zero bias produces exact 0.000000 Sample-0 transient
    silence = np.zeros(32, dtype=np.float64)
    y_sil = adaa(silence, reset=True)
    np.testing.assert_allclose(y_sil, 0.0, atol=1e-15)

    # Full-block vs chunked streaming parity
    y_full = adaa(x, reset=True)
    adaa.reset()
    y_c1 = adaa(x[:16])
    y_c2 = adaa(x[16:])
    np.testing.assert_allclose(np.concatenate([y_c1, y_c2]), y_full, atol=1e-14)

    # Pre-allocated out= buffer and single-sample step parity
    out_buf = np.empty_like(x)
    adaa(x, reset=True, out=out_buf)
    np.testing.assert_allclose(out_buf, y_full, atol=1e-15)

    adaa.reset()
    step_0 = adaa.step(float(x[0]))
    assert abs(step_0 - float(y_full[0])) < 1e-14

    # State getter/setter and algebraic helper
    saved_state = adaa.state
    adaa.state = saved_state
    alg = dsp.PolyAdaa.algebraic(-0.5, 0.5, beta=0.65, delta=0.25)
    assert np.isfinite(alg)


def test_poly_adaa_validation() -> None:
    adaa = dsp.PolyAdaa(PolyAdaaParams())
    with pytest.raises(DnamValidationError):
        dsp.PolyAdaa(cast(Any, "invalid"))
    with pytest.raises(DnamValidationError):
        adaa.state = cast(Any, "invalid")
    with pytest.raises(DnamValidationError):
        adaa(np.zeros((2, 8)))
    with pytest.raises(DnamValidationError):
        adaa(np.zeros(8), out=np.zeros(4))
    with pytest.raises(DnamValidationError):
        adaa(np.zeros(8), out=cast(Any, np.zeros(8, dtype=np.float32)))
    # Non-float64 input coercion
    y_coerced = adaa([0.1, 0.2, -0.1], reset=True)
    assert y_coerced.shape == (3,)
    assert isinstance(adaa.reset(), PolyAdaaState)


@given(x=st_audio_1d_f64(min_len=32, max_len=128))
def test_selective_rglru_streaming_and_h2_norm(x: NDArray[np.float64]) -> None:
    params = SelectiveRgLruParams.from_polar(r_low=0.92, dr_span=0.04, theta_rad=0.2, w_sel=0.5)
    rglru = dsp.SelectiveRgLru(params)
    assert rglru.params == params

    y_full = rglru(x, reset=True)
    rglru.reset()
    y_c1 = rglru(x[:16])
    y_c2 = rglru(x[16:])
    np.testing.assert_allclose(np.concatenate([y_c1, y_c2]), y_full, atol=1e-14)

    out_buf = np.empty_like(x)
    rglru(x, reset=True, out=out_buf)
    np.testing.assert_allclose(out_buf, y_full, atol=1e-15)

    rglru.reset()
    s0 = rglru.step(float(x[0]))
    assert abs(s0 - float(y_full[0])) < 1e-14
    rglru.state = SelectiveRgLruState(z_re=0.1, z_im=-0.1)
    assert abs(rglru.state.z_re - 0.1) < 1e-12
    y_non_zero_init = rglru(x[:8], reset=True)
    np.testing.assert_allclose(y_non_zero_init, y_full[:8], atol=1e-14)

    with pytest.raises(DnamValidationError):
        dsp.SelectiveRgLru(cast(Any, "invalid"))
    with pytest.raises(DnamValidationError):
        rglru.state = cast(Any, "invalid")
    with pytest.raises(DnamValidationError):
        rglru(np.zeros((2, 8)))


@given(x=st_audio_1d_f64(min_len=64, max_len=128))
def test_modulated_ufdn_coercive_loop(x: NDArray[np.float64]) -> None:
    params = MufdnParams(delay_samples=16, g_lin=0.90, sigma_knee=0.35, v_ceil=0.985, r_dc=0.995)
    ufdn = dsp.ModulatedUfdn(params)
    assert ufdn.params == params

    y = ufdn(x, reset=True)
    assert np.all(np.abs(y) <= params.v_ceil + 1e-12)
    out_buf = np.empty_like(x)
    ufdn(x, out=out_buf)
    np.testing.assert_allclose(out_buf, y, atol=1e-15)

    s_step = ufdn.step(0.5)
    assert abs(s_step) <= params.v_ceil + 1e-12
    ufdn.state = MufdnState(dc_x_prev=0.0, dc_y_prev=0.0)
    assert ufdn.state.dc_x_prev == 0.0

    with pytest.raises(DnamValidationError):
        dsp.ModulatedUfdn(cast(Any, "invalid"))
    with pytest.raises(DnamValidationError):
        ufdn.state = cast(Any, "invalid")
    with pytest.raises(DnamValidationError):
        ufdn(np.zeros((2, 8)))


@given(
    phi=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    mu=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_algebraic_rotor_lagrange_and_givens_householder(phi: float, mu: float) -> None:
    rot = dsp.unit_rotor(phi)
    assert isinstance(rot, RotorStep)
    assert abs(rot.cos * rot.cos + rot.sin * rot.sin - 1.0) < 1e-5

    rot_0d = dsp.unit_rotor(np.array(phi))
    assert isinstance(rot_0d, RotorStep)

    rot_arr = dsp.unit_rotor(np.array([phi, -phi]))
    assert isinstance(rot_arr, np.ndarray)
    assert rot_arr.shape == (2, 2)
    with pytest.raises(DnamValidationError):
        dsp.unit_rotor(np.zeros((2, 2)))

    w = dsp.lagrange_weights(mu)
    assert abs(sum(w) - 1.0) < 1e-14

    vec8 = np.array([0.4, -0.7, 1.1, -0.2, 0.85, -0.55, 0.3, 0.9], dtype=np.float64)
    mixed = dsp.givens_householder_mix(vec8, cos_psi=rot.cos, sin_psi=rot.sin)
    assert abs(float(np.sum(mixed * mixed)) - float(np.sum(vec8 * vec8))) < 1e-5
    with pytest.raises(DnamValidationError):
        dsp.givens_householder_mix(np.zeros(4), cos_psi=1.0, sin_psi=0.0)
