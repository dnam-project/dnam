"""Canonical Reference DSP interface (`dnam_engine.dsp`):
callable stateful `PolyAdaa`, `SelectiveRgLru`, and `ModulatedUfdn` processors configured via
validated Pydantic v2 parameter models, plus stateless algebraic kernels."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dnam_engine import _native
from dnam_engine.exceptions import DnamValidationError, call_native
from dnam_engine.schema.runtime import (
    MufdnParams,
    MufdnState,
    PolyAdaaParams,
    PolyAdaaState,
    RotorStep,
    SelectiveRgLruParams,
    SelectiveRgLruState,
)


def _coerce_f64_input(x: ArrayLike) -> NDArray[np.float64]:
    """Zero-copy fast path for C-contiguous `float64` arrays; coerces other `ArrayLike` inputs."""
    if isinstance(x, np.ndarray) and x.dtype == np.float64 and x.flags.c_contiguous:
        return x
    return np.ascontiguousarray(x, dtype=np.float64)


def _validate_f64_out(out: NDArray[np.float64], expected_shape: tuple[int, ...]) -> None:
    if (
        not isinstance(out, np.ndarray)
        or out.dtype != np.float64
        or not out.flags.c_contiguous
        or not out.flags.writeable
    ):
        raise DnamValidationError("out= buffer must be a writeable C-contiguous float64 ndarray")
    if out.shape != expected_shape:
        raise DnamValidationError(
            f"out= buffer shape {out.shape} does not match expected shape {expected_shape}"
        )


class PolyAdaa:
    """Callable stateful Core 1 $1\\times$ B-Spline `PolyADAA` waveshaper (`CL-01`, `CL-02`)."""

    def __init__(self, params: PolyAdaaParams) -> None:
        if not isinstance(params, PolyAdaaParams):
            raise DnamValidationError("PolyAdaa requires a PolyAdaaParams instance")
        self._params = params
        self._state = PolyAdaaState.resting_equilibrium(params)

    @property
    def params(self) -> PolyAdaaParams:
        """Returns the validated `PolyAdaaParams`."""
        return self._params

    @property
    def state(self) -> PolyAdaaState:
        """Returns the current 3-state `(u_prev, a_prev, m_dc)` memory."""
        return self._state

    @state.setter
    def state(self, new_state: PolyAdaaState) -> None:
        if not isinstance(new_state, PolyAdaaState):
            raise DnamValidationError("state must be a PolyAdaaState instance")
        self._state = new_state

    def reset(self) -> PolyAdaaState:
        """Resets the 3-state memory to exact analytical DC resting equilibrium (`0.000000`)."""
        self._state = PolyAdaaState.resting_equilibrium(self._params)
        return self._state

    def step(self, x: float) -> float:
        """Processes a single `float` sample and advances internal state."""
        p = self._params
        s = self._state
        y, u_next, a_next, m_next = call_native(
            _native.native_poly_adaa_step,
            float(x),
            p.b_conv,
            p.alpha,
            p.beta,
            p.delta,
            p.r_ac,
            s.u_prev,
            s.a_prev,
            s.m_dc,
        )
        self._state = PolyAdaaState._construct_trusted(u_prev=u_next, a_prev=a_next, m_dc=m_next)
        return float(y)

    def process(
        self,
        x: ArrayLike,
        *,
        reset: bool = False,
        out: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Processes a 1D `(N,)` array through the reference `PolyADAA` kernel."""
        inp = _coerce_f64_input(x)
        if inp.ndim != 1:
            raise DnamValidationError(f"PolyAdaa expected 1D input array, got shape {inp.shape}")
        if out is not None:
            _validate_f64_out(out, inp.shape)
            target = out
        else:
            target = np.empty_like(inp, dtype=np.float64)

        p = self._params
        s = self._state
        u_next, a_next, m_next = call_native(
            _native.native_poly_adaa_scan,
            inp,
            target,
            p.b_conv,
            p.alpha,
            p.beta,
            p.delta,
            p.r_ac,
            s.u_prev,
            s.a_prev,
            s.m_dc,
            reset,
        )
        self._state = PolyAdaaState._construct_trusted(u_prev=u_next, a_prev=a_next, m_dc=m_next)
        return target

    def __call__(
        self,
        x: ArrayLike,
        *,
        reset: bool = False,
        out: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        return self.process(x, reset=reset, out=out)

    @staticmethod
    def algebraic(
        u_prev: float,
        u_curr: float,
        *,
        beta: float = 0.65,
        delta: float = 0.25,
    ) -> float:
        """Evaluates the division-free algebraic segment average `A_c[n]` (`CL-01`)."""
        return float(
            call_native(
                _native.native_poly_adaa_algebraic,
                float(u_prev),
                float(u_curr),
                float(beta),
                float(delta),
            )
        )


class SelectiveRgLru:
    """Callable stateful Core 1 Radial-Polar `SelectiveRgLru` resonator (`CL-03`..`CL-05`)."""

    def __init__(self, params: SelectiveRgLruParams) -> None:
        if not isinstance(params, SelectiveRgLruParams):
            raise DnamValidationError("SelectiveRgLru requires a SelectiveRgLruParams instance")
        self._params = params
        self._state = SelectiveRgLruState()

    @property
    def params(self) -> SelectiveRgLruParams:
        """Returns the validated `SelectiveRgLruParams`."""
        return self._params

    @property
    def state(self) -> SelectiveRgLruState:
        """Returns the current complex resonator state `(z_re, z_im)`."""
        return self._state

    @state.setter
    def state(self, new_state: SelectiveRgLruState) -> None:
        if not isinstance(new_state, SelectiveRgLruState):
            raise DnamValidationError("state must be a SelectiveRgLruState instance")
        self._state = new_state

    def reset(self) -> SelectiveRgLruState:
        """Resets the complex resonator state to `(0.0, 0.0)`."""
        self._state = SelectiveRgLruState()
        return self._state

    def step(self, x: float) -> float:
        """Processes a single `float` sample and advances internal complex state."""
        p = self._params
        s = self._state
        y, z_re_next, z_im_next = call_native(
            _native.native_selective_rglru_step,
            float(x),
            p.r_low,
            p.dr_span,
            p.w_sel,
            p.b_gamma_norm,
            p.c_real,
            p.c_imag,
            p.u_rot_cos,
            p.u_rot_sin,
            s.z_re,
            s.z_im,
        )
        self._state = SelectiveRgLruState._construct_trusted(z_re=z_re_next, z_im=z_im_next)
        return float(y)

    def process(
        self,
        x: ArrayLike,
        *,
        reset: bool = False,
        out: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Processes a 1D `(N,)` array through the reference `SelectiveRgLru` resonator."""
        inp = _coerce_f64_input(x)
        if inp.ndim != 1:
            raise DnamValidationError(
                f"SelectiveRgLru expected 1D input array, got shape {inp.shape}"
            )
        if out is not None:
            _validate_f64_out(out, inp.shape)
            target = out
        else:
            target = np.empty_like(inp, dtype=np.float64)

        p = self._params
        s = self._state
        z_re_next, z_im_next = call_native(
            _native.native_selective_rglru_scan,
            inp,
            target,
            p.r_low,
            p.dr_span,
            p.w_sel,
            p.b_gamma_norm,
            p.c_real,
            p.c_imag,
            p.u_rot_cos,
            p.u_rot_sin,
            s.z_re,
            s.z_im,
            reset,
        )
        self._state = SelectiveRgLruState._construct_trusted(z_re=z_re_next, z_im=z_im_next)
        return target

    def __call__(
        self,
        x: ArrayLike,
        *,
        reset: bool = False,
        out: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        return self.process(x, reset=reset, out=out)


class ModulatedUfdn:
    """Callable stateful Core 2 Coercive-Saturating `M-UFDN` processor (`CL-09`..`CL-11`)."""

    def __init__(self, params: MufdnParams) -> None:
        if not isinstance(params, MufdnParams):
            raise DnamValidationError("ModulatedUfdn requires a MufdnParams instance")
        self._params = params
        self._state = MufdnState()

    @property
    def params(self) -> MufdnParams:
        """Returns the validated `MufdnParams`."""
        return self._params

    @property
    def state(self) -> MufdnState:
        """Returns the current 2-state DC blocker memory `(dc_x_prev, dc_y_prev)`."""
        return self._state

    @state.setter
    def state(self, new_state: MufdnState) -> None:
        if not isinstance(new_state, MufdnState):
            raise DnamValidationError("state must be a MufdnState instance")
        self._state = new_state

    def reset(self) -> MufdnState:
        """Resets the DC blocker memory to `(0.0, 0.0)`."""
        self._state = MufdnState()
        return self._state

    def step(self, u_in: float) -> float:
        """Processes a sample through the coercive-saturating Lyapunov loop + DC blocker."""
        p = self._params
        s = self._state
        s_next, x_next, y_next = call_native(
            _native.native_mufdn_coercive_step,
            float(u_in),
            p.g_lin,
            p.sigma_knee,
            p.v_ceil,
            p.r_dc,
            s.dc_x_prev,
            s.dc_y_prev,
        )
        self._state = MufdnState._construct_trusted(dc_x_prev=x_next, dc_y_prev=y_next)
        return float(s_next)

    def process(
        self,
        x: ArrayLike,
        *,
        reset: bool = False,
        out: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Processes a 1D `(N,)` array through the reference `M-UFDN` delay loop."""
        if reset:
            self.reset()
        inp = _coerce_f64_input(x)
        if inp.ndim != 1:
            raise DnamValidationError(
                f"ModulatedUfdn expected 1D input array, got shape {inp.shape}"
            )
        if out is not None:
            _validate_f64_out(out, inp.shape)
            target = out
        else:
            target = np.empty_like(inp, dtype=np.float64)

        p = self._params
        call_native(
            _native.native_mufdn_slice,
            inp,
            target,
            p.delay_samples,
            p.g_lin,
            p.sigma_knee,
            p.v_ceil,
            p.r_dc,
        )
        return target

    def __call__(
        self,
        x: ArrayLike,
        *,
        reset: bool = False,
        out: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        return self.process(x, reset=reset, out=out)


def unit_rotor(phi: float | ArrayLike) -> RotorStep | NDArray[np.float64]:
    """Evaluates the $3\\text{ kHz}$ 4-FMA + Newton unit-circle rotor `(cos_theta, sin_theta)`."""
    if isinstance(phi, int | float):
        c, s = call_native(_native.native_unit_rotor, float(phi))
        return RotorStep(cos=float(c), sin=float(s))
    arr = (
        phi
        if isinstance(phi, np.ndarray) and phi.dtype == np.float64 and phi.flags.c_contiguous
        else np.ascontiguousarray(phi, dtype=np.float64)
    )
    if arr.ndim == 0:
        c, s = call_native(_native.native_unit_rotor, float(arr))
        return RotorStep(cos=float(c), sin=float(s))
    if arr.ndim != 1:
        raise DnamValidationError("unit_rotor input phi must be a scalar or 1D array")
    out = np.empty((2, arr.shape[0]), dtype=np.float64)
    call_native(_native.native_unit_rotor_slice, arr, out)
    return out


def lagrange_weights(mu: float) -> tuple[float, float, float, float]:
    """Computes the 4-tap Cubic Lagrange weights `(w_-1, w_0, w_+1, w_+2)` (`CL-06`)."""
    w = call_native(_native.native_cubic_lagrange_weights, float(mu))
    return (float(w[0]), float(w[1]), float(w[2]), float(w[3]))


def givens_householder_mix(
    x: ArrayLike,
    *,
    cos_psi: float,
    sin_psi: float,
) -> NDArray[np.float64]:
    """Applies the 8-channel Orthogonal Givens + Householder mixing matrix `U(psi)` (`CL-08`)."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape != (8,):
        raise DnamValidationError(
            f"givens_householder_mix requires an 8-element 1D vector, got shape {arr.shape}"
        )
    mixed = call_native(
        _native.native_givens_householder_mix_8ch,
        [float(v) for v in arr],
        float(cos_psi),
        float(sin_psi),
    )
    return np.asarray(mixed, dtype=np.float64)
