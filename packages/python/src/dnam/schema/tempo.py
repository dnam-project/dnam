"""Canonical 12-subdivision tempo sync and physical calibration inversion models
(`ref[sdk.python.schema.rust-parity]`).
"""

from __future__ import annotations

from enum import StrEnum

from dnam.errors import ValidationError
from dnam.schema._base import StrictModel


class TempoSubdivision(StrEnum):
    """Canonical 12 rhythmic tempo subdivisions (`1/2.` through `1/16T`)."""

    HALF_DOTTED = "1/2."
    HALF = "1/2"
    HALF_TRIPLET = "1/2T"
    QUARTER_DOTTED = "1/4."
    QUARTER = "1/4"
    QUARTER_TRIPLET = "1/4T"
    EIGHTH_DOTTED = "1/8."
    EIGHTH = "1/8"
    EIGHTH_TRIPLET = "1/8T"
    SIXTEENTH_DOTTED = "1/16."
    SIXTEENTH = "1/16"
    SIXTEENTH_TRIPLET = "1/16T"

    @classmethod
    def parse(cls, value: TempoSubdivision | str) -> TempoSubdivision:
        """Coerces a `TempoSubdivision` or subdivision string (`'1/2.'`..`'1/16T'`)."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            norm = value.strip()
            for member in cls:
                if member.value == norm or member.name.lower() == norm.lower():
                    return member
        raise ValidationError(f"Unknown tempo subdivision: {value!r}")

    @property
    def beat_multiplier(self) -> float:
        """Quarter-note beat multiplier for this subdivision."""
        multipliers: dict[TempoSubdivision, float] = {
            TempoSubdivision.HALF_DOTTED: 3.0,
            TempoSubdivision.HALF: 2.0,
            TempoSubdivision.HALF_TRIPLET: 4.0 / 3.0,
            TempoSubdivision.QUARTER_DOTTED: 1.5,
            TempoSubdivision.QUARTER: 1.0,
            TempoSubdivision.QUARTER_TRIPLET: 2.0 / 3.0,
            TempoSubdivision.EIGHTH_DOTTED: 0.75,
            TempoSubdivision.EIGHTH: 0.5,
            TempoSubdivision.EIGHTH_TRIPLET: 1.0 / 3.0,
            TempoSubdivision.SIXTEENTH_DOTTED: 0.375,
            TempoSubdivision.SIXTEENTH: 0.25,
            TempoSubdivision.SIXTEENTH_TRIPLET: 1.0 / 6.0,
        }
        return multipliers[self]

    def delay_ms_at_bpm(self, bpm: float) -> float:
        """Target delay time in milliseconds at `bpm` (clamped to `[20.0, 400.0]` BPM)."""
        safe_bpm = min(max(float(bpm), 20.0), 400.0)
        return (60_000.0 / safe_bpm) * self.beat_multiplier

    def lfo_hz_at_bpm(self, bpm: float) -> float:
        """Target LFO modulation rate in Hz at `bpm` (clamped to `[20.0, 400.0]` BPM)."""
        safe_bpm = min(max(float(bpm), 20.0), 400.0)
        return safe_bpm / (60.0 * self.beat_multiplier)


class TempoSyncResolution(StrictModel):
    """Result of resolving a physical target or MIDI tempo sync against `ControlCalibration`."""

    switch_selection: tuple[str, str] | None = None
    normalized_u: float
    was_clamped: bool


class TempoInversionResult(StrictModel):
    """Result of the 32-step binary-search physical `f32` calibration inversion
    (`ref[dsp.tempo.physical-inversion]`).
    """

    physical_f32: float
    normalized_f32: float
    realized_value: float
    residual: float
    search_steps: tuple[int, int]
