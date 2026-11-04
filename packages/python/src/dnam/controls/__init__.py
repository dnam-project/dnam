"""Offline control-space clamping, ZOH rescaling, and tempo calibration helpers
(`ref[sdk.python.controls.space-and-tempo]`).
"""

from dnam.controls._space import ClampedControl, ClampResult, ControlSpace
from dnam.controls._tempo import (
    ZohParams,
    compute_zoh,
    control_stride_for_rate,
    control_tick_ratio,
    eval_3node_calibration,
    invert_tempo_calibration,
    subdivision_phase_q32,
)
from dnam.schema.tempo import (
    TempoInversionResult,
    TempoSubdivision,
    TempoSyncResolution,
)

__all__ = [
    "ClampResult",
    "ClampedControl",
    "ControlSpace",
    "TempoInversionResult",
    "TempoSubdivision",
    "TempoSyncResolution",
    "ZohParams",
    "compute_zoh",
    "control_stride_for_rate",
    "control_tick_ratio",
    "eval_3node_calibration",
    "invert_tempo_calibration",
    "subdivision_phase_q32",
]
