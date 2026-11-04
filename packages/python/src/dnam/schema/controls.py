"""Parametric `range` and `switch` control schemas, physical calibration, and preset definitions
(`ref[sdk.python.schema.rust-parity]`).
"""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import Field

from dnam.schema._base import StrictModel
from dnam.schema.tempo import TempoSubdivision, TempoSyncResolution
from dnam.schema.ui_hints import ControlUiHints

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


class ControlKind(StrEnum):
    """Parametric control kind (`'range'` vs `'switch'`)."""

    RANGE = "range"
    SWITCH = "switch"


class ControlCalibrationRole(StrEnum):
    """Physical calibration role for a parametric control (`ControlCalibration.role`)."""

    DELAY_TIME_MS = "delay_time_ms"
    LFO_RATE_HZ = "lfo_rate_hz"
    FEEDBACK = "feedback"
    MIX = "mix"
    CROSSOVER_HZ = "crossover_hz"
    STEREO_MODE = "stereo_mode"
    STEREO_WIDTH = "stereo_width"
    FREEZE_LATCH = "freeze_latch"
    GRAIN_SIZE_MS = "grain_size_ms"
    GRAIN_PITCH_SEMITONES = "grain_pitch_semitones"
    GRAIN_SPRAY = "grain_spray"


class ControlCalibration(StrictModel):
    """Physical calibration and MIDI tempo-sync metadata for a single parametric control."""

    role: ControlCalibrationRole
    lobatto_physical_anchors: tuple[float, float, float] | None = None
    conditional_physical_anchors: dict[str, tuple[float, float, float]] = Field(
        default_factory=dict
    )
    tempo_sync_default_subdivision: TempoSubdivision | None = None

    @staticmethod
    def invert_lobatto_anchors(
        anchors: tuple[float, float, float],
        target_val: float,
        *,
        fold_octave: bool = False,
    ) -> tuple[float, bool]:
        """Inverts a 3-point Lobatto quadratic `v(u)` anchored at `[v_min, v_mid, v_max]`,
        returning `(u, was_clamped)`.
        """
        v_min, v_mid, v_max = (
            float(anchors[0]),
            float(anchors[1]),
            float(anchors[2]),
        )
        lo = min(v_min, v_max)
        hi = max(v_min, v_max)
        if (hi - lo) <= 1e-9:
            return (0.0, True)

        t = float(target_val)
        if fold_octave and t > 1e-6 and lo > 1e-6:
            for _ in range(4):
                if t > hi:
                    t *= 0.5
                elif t < lo:
                    t *= 2.0
                else:
                    break

        ascending = v_max >= v_min
        if t <= lo:
            clamped = abs(t - lo) > 1e-6
            return (-1.0 if ascending else 1.0, clamped)
        if t >= hi:
            clamped = abs(t - hi) > 1e-6
            return (1.0 if ascending else -1.0, clamped)

        c1 = 0.5 * (v_max - v_min)
        c2 = 0.5 * (v_max + v_min) - v_mid
        if abs(c2) <= 1e-7 * max(abs(c1), 1.0):
            if abs(c1) <= 1e-12:
                return (0.0, False)
            u = min(max((t - v_mid) / c1, -1.0), 1.0)
            return (u, False)

        disc = max(c1 * c1 - 4.0 * c2 * (v_mid - t), 0.0)
        root_disc = math.sqrt(disc)
        r1 = (-c1 + root_disc) / (2.0 * c2)
        r2 = (-c1 - root_disc) / (2.0 * c2)
        r1_in = (-1.0 - 1e-4) <= r1 <= (1.0 + 1e-4)
        r2_in = (-1.0 - 1e-4) <= r2 <= (1.0 + 1e-4)
        if r1_in and not r2_in:
            u = r1
        elif r2_in and not r1_in:
            u = r2
        elif c1 >= 0.0:
            u = r1
        else:
            u = r2
        return (min(max(u, -1.0), 1.0), False)

    def evaluate_lobatto_anchors(self, u: float) -> float | None:
        """Evaluates the 3-point Lobatto quadratic `v(u)` for `u in [-1, 1]` if anchors are
        present.
        """
        if self.lobatto_physical_anchors is None:
            return None
        v_min, v_mid, v_max = self.lobatto_physical_anchors
        u_clamped = min(max(float(u), -1.0), 1.0)
        c1 = 0.5 * (v_max - v_min)
        c2 = 0.5 * (v_max + v_min) - v_mid
        return v_mid + c1 * u_clamped + c2 * u_clamped * u_clamped

    def resolve_tempo_sync(
        self,
        bpm: float,
        subdivision: TempoSubdivision | str | None = None,
        *,
        fold_octave: bool = False,
    ) -> TempoSyncResolution | None:
        """
        Resolves `(bpm, subdivision)` into a `TempoSyncResolution` for delay/grain/LFO controls.
        """
        sub = (
            TempoSubdivision.parse(subdivision)
            if subdivision is not None
            else (self.tempo_sync_default_subdivision or TempoSubdivision.QUARTER)
        )
        if self.role in (
            ControlCalibrationRole.DELAY_TIME_MS,
            ControlCalibrationRole.GRAIN_SIZE_MS,
        ):
            target = sub.delay_ms_at_bpm(bpm)
        elif self.role == ControlCalibrationRole.LFO_RATE_HZ:
            target = sub.lfo_hz_at_bpm(bpm)
        else:
            return None

        if self.conditional_physical_anchors:
            best_key: str | None = None
            best_anchors: tuple[float, float, float] | None = None
            best_dist = float("inf")
            for key, anchors in self.conditional_physical_anchors.items():
                lo = min(anchors[0], anchors[2])
                hi = max(anchors[0], anchors[2])
                dist = (lo - target) if target < lo else ((target - hi) if target > hi else 0.0)
                if dist < best_dist:
                    best_dist = dist
                    best_key = key
                    best_anchors = anchors
                    if dist == 0.0:
                        break
            if best_key is None or best_anchors is None:
                return None
            u, was_clamped = self.invert_lobatto_anchors(
                best_anchors, target, fold_octave=fold_octave
            )
            sw_sel = tuple(best_key.split("=", 1)) if "=" in best_key else None
            pair = (sw_sel[0], sw_sel[1]) if sw_sel is not None and len(sw_sel) == 2 else None
            return TempoSyncResolution(
                switch_selection=pair,
                normalized_u=u,
                was_clamped=was_clamped,
            )

        if self.lobatto_physical_anchors is None:
            return None
        u, was_clamped = self.invert_lobatto_anchors(
            self.lobatto_physical_anchors, target, fold_octave=fold_octave
        )
        return TempoSyncResolution(
            switch_selection=None,
            normalized_u=u,
            was_clamped=was_clamped,
        )


class ConditionalClampRule(StrictModel):
    """Switch-conditioned safe range bound rule (`ref[container.dnam.conditional-rules]`)."""

    switch_ctrl_idx: int = Field(ge=0, le=255)
    option_mask: int = Field(ge=0, le=65535)
    cond_min: float
    cond_max: float


class ControlSchemaDef(StrictModel):
    """Schema definition for a single parametric control (`range` or `switch`)."""

    id: str
    name: str
    kind: ControlKind = Field(alias="type")
    unit: str | None = None
    lobatto_degree: int | None = Field(default=None, ge=0, le=3)
    anchor_nodes_norm: list[float] | None = None
    declared_range: tuple[float, float] | None = None
    safe_clamped_range: tuple[float, float] | None = None
    default: float | int | str
    conditional_min: dict[str, float] = Field(default_factory=dict)
    conditional_max: dict[str, float] = Field(default_factory=dict)
    declared_options: list[str] | None = None
    active_options: list[str] | None = None
    clamp_warning: str | None = None
    calibration: ControlCalibration | None = None
    ui_hints: ControlUiHints = Field(default_factory=ControlUiHints)
    user_metadata: dict[str, JsonValue] = Field(default_factory=dict)


class PresetDef(StrictModel):
    """Built-in preset snapshot (`presets[*]`)."""

    id: str
    name: str
    description: str | None = None
    controls: dict[str, float | int | str]
    user_metadata: dict[str, JsonValue] = Field(default_factory=dict)
