"""Runtime settings (`DnamEngineSettings`), telemetry (`DnamTelemetrySnapshot`), ZOH rescaling
(`ZohRescaledParams`), binary preamble (`DnamPreambleInfo`), and Reference DSP models."""

from __future__ import annotations

import math
from typing import NamedTuple

from pydantic import Field, field_validator, model_validator

from dnam_engine.exceptions import DnamSampleRateError, DnamValidationError
from dnam_engine.schema.base import BackendKind, StrictModel, Tier


class DnamEngineSettings(StrictModel):
    """Strongly typed configuration settings for `DnamEngine`."""

    tier: Tier = Tier.FULL
    sample_rate_hz: float = 48_000.0
    backend: BackendKind = BackendKind.AUTO
    buffer_size: int | None = Field(default=None, ge=1)

    @field_validator("sample_rate_hz")
    @classmethod
    def _validate_sample_rate(cls, v: float) -> float:
        if not (44_100.0 <= float(v) <= 96_000.0):
            raise DnamSampleRateError(f"sample_rate_hz ({v} Hz) must lie in [44100, 96000] Hz")
        return float(v)


class ZohRescaledParams(StrictModel):
    """Continuous-Time Zero-Order Hold (ZOH) rescaled parameters (`44.1--96 kHz`)."""

    sample_rate_host_hz: int
    rho_sr: float
    control_decimation_host: int
    rho_ctrl: float
    r_ac_host: float
    r_dc_host: float
    tau_ref_samples_host: float
    switch_xfade_samples_host: int


class DnamPreambleInfo(StrictModel):
    """64-byte binary preamble metadata of a `.dnam` or `.dnam-ckpt` container."""

    magic: str
    version_major: int
    version_minor: int
    flags: int
    num_controls: int
    ring_buffer_pow2_log2: int
    sample_rate_train_hz: int
    json_header_byte_len: int
    tensor_arena_offset: int
    tensor_arena_byte_len: int
    arena_offset_lite_end: int
    arena_offset_full_end: int
    arena_offset_ultra_end: int
    arena_crc32c_lite: int
    arena_crc32c_full: int
    arena_crc32c_ultra: int
    num_ckpt_stems: int
    control_decimation: int
    head_svd_rank: int
    tail_time_ms: float
    header_crc32c: int

    @property
    def header_capacity_bytes(self) -> int:
        """Returns the pre-allocated JSON header capacity (`tensor_arena_offset - 64`)."""
        return max(0, self.tensor_arena_offset - 64)


class DnamTelemetrySnapshot(StrictModel):
    """Lock-free runtime telemetry snapshot from `DnamEngine`."""

    active_tier: Tier
    backend: BackendKind
    sample_rate_hz: int
    core2_active: bool
    lfo_env_active: bool
    stereo_out: bool
    active_clamp_mask: int
    control_decimation_host: int
    max_adaa_drive: float
    max_pole_radius: float
    max_effective_loop_gain: float
    tail_time_ms: float
    z_head: tuple[float, float, float, float]
    env_states: tuple[float, float, float, float]
    lfo_harmonics: tuple[float, float, float, float, float, float, float, float]
    clamped_controls: tuple[float, ...]
    rho_sr: float
    rho_ctrl: float
    r_ac_host: float
    r_dc_host: float
    tau_ref_samples_host: float
    switch_xfade_samples_host: int


class RotorStep(NamedTuple):
    """Unit-circle rotor coordinates `(cos_theta, sin_theta)`."""

    cos: float
    sin: float


class PolyAdaaParams(StrictModel):
    """Validated parameters for Core 1 `PolyADAA` reference waveshaper (`CL-01`, `CL-02`)."""

    b_conv: float = 0.0
    alpha: float = 1.0
    beta: float = 0.65
    delta: float = Field(default=0.25, ge=-0.45, le=0.45)
    r_ac: float = Field(default=0.9993, gt=0.9, lt=1.0)


class PolyAdaaState(StrictModel):
    """3-state memory `(u_prev, a_prev, m_dc)` for Core 1 `PolyADAA` (`CL-02`)."""

    u_prev: float
    a_prev: float
    m_dc: float

    @classmethod
    def resting_equilibrium(cls, params: PolyAdaaParams) -> PolyAdaaState:
        """Computes the exact analytical DC resting equilibrium (`0.000000` Sample-0 transient)."""
        u_rest = max(-1.0, min(1.0, params.b_conv * (2.0 / 3.0)))
        u2 = u_rest * u_rest
        g_rest = params.beta * u_rest * (1.5 - 0.5 * u2) + params.delta * (u2 - u2 * u2)
        return cls(u_prev=u_rest, a_prev=g_rest, m_dc=g_rest)


class SelectiveRgLruParams(StrictModel):
    """Validated parameters for Core 1 Radial-Polar `SelectiveRgLru` (`CL-03`..`CL-05`)."""

    r_low: float = Field(default=0.90, ge=0.25, le=0.9995)
    dr_span: float = Field(default=0.05, ge=0.0, le=0.7495)
    w_sel: float = Field(default=0.5, ge=0.0)
    b_gamma_norm: float = 0.4358898943540673
    c_real: float = 1.0
    c_imag: float = 0.0
    u_rot_cos: float = 1.0
    u_rot_sin: float = 0.0

    @model_validator(mode="after")
    def _validate_pole_radius(self) -> SelectiveRgLruParams:
        if self.r_low + self.dr_span > 0.9995 + 1e-9:
            r_sum = self.r_low + self.dr_span
            raise DnamValidationError(
                f"SelectiveRgLruParams r_low + dr_span ({r_sum}) must be <= 0.9995"
            )
        return self

    @classmethod
    def from_polar(
        cls,
        *,
        r_low: float = 0.90,
        dr_span: float = 0.0,
        theta_rad: float = 0.15,
        w_sel: float = 0.0,
        c_real: float = 1.0,
        c_imag: float = 0.0,
    ) -> SelectiveRgLruParams:
        """Constructs `SelectiveRgLruParams` from polar coordinates `(r_low, dr_span, theta_rad)`
        with exact H2 energy normalization `b_gamma_norm = sqrt(1 - r_bar^2)`."""
        r_bar = min(0.9995, max(0.25, r_low + 0.5 * dr_span))
        return cls(
            r_low=r_low,
            dr_span=dr_span,
            w_sel=w_sel,
            b_gamma_norm=math.sqrt(max(0.0, 1.0 - r_bar * r_bar)),
            c_real=c_real,
            c_imag=c_imag,
            u_rot_cos=math.cos(theta_rad),
            u_rot_sin=math.sin(theta_rad),
        )


class SelectiveRgLruState(StrictModel):
    """Complex resonator state `(z_re, z_im)` for Core 1 `SelectiveRgLru`."""

    z_re: float = 0.0
    z_im: float = 0.0


class MufdnParams(StrictModel):
    """Validated parameters for Core 2 Coercive-Saturating `M-UFDN` loop (`CL-09`..`CL-11`)."""

    delay_samples: int = Field(default=128, ge=4, le=65_536)
    g_lin: float = Field(default=0.85, ge=0.0, le=0.9990)
    sigma_knee: float = Field(default=0.35, gt=0.0)
    v_ceil: float = Field(default=0.985, gt=0.0, le=2.0)
    r_dc: float = Field(default=0.995, ge=-1.0, lt=1.0)


class MufdnState(StrictModel):
    """2-state DC blocker memory `(dc_x_prev, dc_y_prev)` for Core 2 `M-UFDN`."""

    dc_x_prev: float = 0.0
    dc_y_prev: float = 0.0
