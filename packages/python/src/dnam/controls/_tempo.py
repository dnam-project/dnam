"""Tempo synchronization, ZOH rescaling, and 32-step physical calibration inversion helpers
(`ref[sdk.python.controls.space-and-tempo]`).
"""

from __future__ import annotations

from dnam._native import (
    native_compute_zoh,
    native_control_stride_for_rate,
    native_control_tick_ratio,
    native_eval_3node_calibration,
    native_eval_subdivision_phase,
    native_invert_tempo_calibration,
)
from dnam.schema._base import StrictModel
from dnam.schema.tempo import TempoInversionResult, TempoSubdivision


class ZohParams(StrictModel):
    """
    Continuous-Time Zero-Order Hold (ZOH) rescaled parameters (`ref[dsp.zoh.ctrl-stride]`).
    """

    sample_rate_host_hz: int
    rho_sr: float
    control_decimation_host: int
    rho_ctrl: float
    alpha_ramp_host: float
    r_ac_host: float
    r_dc_host: float
    tau_ref_samples_host: float
    switch_xfade_samples_host: int


def compute_zoh(
    sample_rate_train_hz: int,
    sample_rate_host_hz: int,
    *,
    r_ac_train: float = 0.9993,
    r_dc_train: float = 0.995,
    tau_ref_train: float = 64.0,
    switch_xfade_train: int = 128,
) -> ZohParams:
    """Computes Continuous-Time ZOH rescaled parameters for `sample_rate_host_hz`."""
    raw = native_compute_zoh(
        sample_rate_train_hz,
        sample_rate_host_hz,
        float(r_ac_train),
        float(r_dc_train),
        float(tau_ref_train),
        switch_xfade_train,
    )
    return ZohParams._construct_trusted(**raw)


def control_stride_for_rate(sample_rate_hz: int) -> int:
    """Returns the control-bus decimation stride `K_ctrl(f_s) = max(8, round(f_s / 3000))`."""
    return native_control_stride_for_rate(sample_rate_hz)


def control_tick_ratio(sample_rate_hz: int) -> float:
    """Returns the exact control-tick ratio `rho_ctrl(f_s)`."""
    return float(native_control_tick_ratio(sample_rate_hz))


def subdivision_phase_q32(
    beat_mod12: int,
    within_beat_q32: int,
    subdivision: TempoSubdivision | str,
    *,
    offset_q32: int = 0,
) -> tuple[int, float, float]:
    """Evaluates `(phase_q32, cos_phi, sin_phi)` for a `Q64`/`Q32` beat position."""
    sub = TempoSubdivision.parse(subdivision)
    q32, c, s = native_eval_subdivision_phase(
        beat_mod12,
        within_beat_q32,
        sub.value,
        offset_q32,
    )
    return q32, float(c), float(s)


def eval_3node_calibration(
    anchors: tuple[float, float, float],
    u: float,
) -> float:
    """Evaluates the 3-node physical calibration curve `v(u)` at `u in [-1, 1]`."""
    a0, a1, a2 = float(anchors[0]), float(anchors[1]), float(anchors[2])
    return float(native_eval_3node_calibration((a0, a1, a2), float(u)))


def invert_tempo_calibration(
    declared_range: tuple[float, float],
    safe_range: tuple[float, float],
    anchors: tuple[float, float, float],
    target: float,
    *,
    flush_subnormals: bool = True,
) -> TempoInversionResult:
    """Inverts a 3-node physical calibration curve over admissible `f32` ranks."""
    raw = native_invert_tempo_calibration(
        (float(declared_range[0]), float(declared_range[1])),
        (float(safe_range[0]), float(safe_range[1])),
        (float(anchors[0]), float(anchors[1]), float(anchors[2])),
        float(target),
        flush_subnormals,
    )
    return TempoInversionResult._construct_trusted(**raw)
