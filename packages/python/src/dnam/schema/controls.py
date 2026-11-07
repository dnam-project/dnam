"""Parametric `range` and `switch` control schemas, physical calibration, and preset definitions
(`ref[sdk.python.schema.rust-parity]`).
"""

from __future__ import annotations

from enum import StrEnum

from dnam._native import (
    native_evaluate_lobatto_anchors,
    native_invert_lobatto_anchors,
    native_resolve_tempo_sync,
)
from dnam.schema._base import StrictModel, strict_field
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
    conditional_physical_anchors: dict[str, tuple[float, float, float]] = strict_field(
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
        a = (float(anchors[0]), float(anchors[1]), float(anchors[2]))
        u, was_clamped = native_invert_lobatto_anchors(a, float(target_val), fold_octave)
        return float(u), was_clamped

    def evaluate_lobatto_anchors(self, u: float) -> float | None:
        """Evaluates the 3-point Lobatto quadratic `v(u)` for `u in [-1, 1]` if anchors are
        present.
        """
        if self.lobatto_physical_anchors is None:
            return None
        return float(native_evaluate_lobatto_anchors(self.lobatto_physical_anchors, float(u)))

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
        sub_str = TempoSubdivision.parse(subdivision).value if subdivision is not None else None
        res = native_resolve_tempo_sync(
            self.to_json(),
            float(bpm),
            sub_str,
            fold_octave,
        )
        if res is None:
            return None
        sw_sel, norm_u, was_clamped = res
        return TempoSyncResolution._construct_trusted(
            switch_selection=sw_sel,
            normalized_u=float(norm_u),
            was_clamped=was_clamped,
        )


class ConditionalClampRule(StrictModel):
    """Switch-conditioned safe range bound rule (`ref[container.dnam.conditional-rules]`)."""

    switch_ctrl_idx: int = strict_field(ge=0, le=255)
    option_mask: int = strict_field(ge=0, le=65535)
    cond_min: float
    cond_max: float


class ControlSchemaDef(StrictModel):
    """Schema definition for a single parametric control (`range` or `switch`)."""

    id: str
    name: str
    kind: ControlKind = strict_field(alias="type")
    unit: str | None = None
    lobatto_degree: int | None = strict_field(default=None, ge=0, le=3)
    anchor_nodes_norm: list[float] | None = None
    declared_range: tuple[float, float] | None = None
    safe_clamped_range: tuple[float, float] | None = None
    default: float | int | str
    conditional_min: dict[str, float] = strict_field(default_factory=dict)
    conditional_max: dict[str, float] = strict_field(default_factory=dict)
    declared_options: list[str] | None = None
    active_options: list[str] | None = None
    clamp_warning: str | None = None
    calibration: ControlCalibration | None = None
    ui_hints: ControlUiHints = strict_field(default_factory=ControlUiHints)
    user_metadata: dict[str, JsonValue] = strict_field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind == ControlKind.RANGE and type(self.default) is int:
            object.__setattr__(self, "default", float(self.default))


class PresetDef(StrictModel):
    """Built-in preset snapshot (`presets[*]`)."""

    id: str
    name: str
    description: str | None = None
    controls: dict[str, float | int | str]
    user_metadata: dict[str, JsonValue] = strict_field(default_factory=dict)
