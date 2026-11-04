"""
Real-time `Telemetry` snapshot model and `OverflowFlags` (`ref[sdk.python.api.engine-facets]`).
"""

from __future__ import annotations

from enum import IntFlag

from dnam.schema._base import Backend, StrictModel, Tier


class OverflowFlags(IntFlag):
    """Bitflags reported in `Telemetry.overflow_or_saturation_flags`
    (`ref[arch.fp.range-saturation]`).
    """

    NONE = 0
    INPUT_SATURATED = 1 << 0
    INTERNAL_SATURATED = 1 << 1
    OUTPUT_SATURATED = 1 << 2
    FP16_SATURATED = 1 << 3
    OUTSIDE_CERTIFIED_INPUT = 1 << 4


class Telemetry(StrictModel):
    """Lock-free telemetry snapshot from `Engine.telemetry`."""

    active_tier: Tier
    backend: Backend
    sample_rate_hz: int
    stereo_out: bool
    bypassed: bool
    trails_enabled: bool
    is_frozen: bool
    supports_freeze: bool
    stereo_width: float
    current_bpm: float
    active_clamp_mask: int
    control_decimation_host: int
    max_op_n_drive: float
    max_pole_radius: float
    max_effective_loop_gain: float
    tail_time_ms: float
    z_head: tuple[float, float, float, float]
    env_states: tuple[float, float, float, float]
    lfo_harmonics: tuple[float, float, float, float, float, float, float, float]
    clamped_controls: list[float]
    overflow_or_saturation_flags: int
    rho_sr: float
    rho_ctrl: float
    alpha_ramp_host: float
    r_ac_host: float
    r_dc_host: float
    tau_ref_samples_host: float
    switch_xfade_samples_host: int

    @property
    def overflow_flags(self) -> OverflowFlags:
        """Typed `OverflowFlags` bitmask for `overflow_or_saturation_flags`."""
        return OverflowFlags(self.overflow_or_saturation_flags)

    @property
    def outside_certified_input(self) -> bool:
        """`True` if input exceeded the certified `+/- 100.0` (`+40 dBFS`) envelope
        (`OUTSIDE_CERTIFIED_INPUT`).
        """
        return bool(self.overflow_flags & OverflowFlags.OUTSIDE_CERTIFIED_INPUT)
