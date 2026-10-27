"""Parametric control schemas (`ControlSchemaDef`), UI hints (`ControlUiHints`),
presets (`PresetDef`), clamping results (`ClampedControlsResult`), and `LobattoParams`."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from dnam_engine.exceptions import DnamValidationError
from dnam_engine.schema.base import ControlKind, StrictModel

TempoSubdivision = Literal[
    "1/2.",
    "1/2",
    "1/2T",
    "1/4.",
    "1/4",
    "1/4T",
    "1/8.",
    "1/8",
    "1/8T",
    "1/16.",
    "1/16",
    "1/16T",
]

TEMPO_SUBDIVISION_MULTIPLIERS: dict[TempoSubdivision, float] = {
    "1/2.": 3.0,
    "1/2": 2.0,
    "1/2T": 4.0 / 3.0,
    "1/4.": 1.5,
    "1/4": 1.0,
    "1/4T": 2.0 / 3.0,
    "1/8.": 0.75,
    "1/8": 0.5,
    "1/8T": 1.0 / 3.0,
    "1/16.": 0.375,
    "1/16": 0.25,
    "1/16T": 1.0 / 6.0,
}

ControlCalibrationRole = Literal[
    "delay_time_ms",
    "lfo_rate_hz",
    "feedback",
    "mix",
    "crossover_hz",
    "stereo_mode",
    "stereo_width",
]


def tempo_subdivision_delay_ms(bpm: float, subdivision: TempoSubdivision) -> float:
    """Computes target delay time in milliseconds for `(bpm, subdivision)`."""
    safe_bpm = max(20.0, min(400.0, float(bpm)))
    return (60_000.0 / safe_bpm) * TEMPO_SUBDIVISION_MULTIPLIERS[subdivision]


def tempo_subdivision_freq_hz(bpm: float, subdivision: TempoSubdivision) -> float:
    """Computes target LFO rate in Hz for `(bpm, subdivision)`."""
    return 1000.0 / tempo_subdivision_delay_ms(bpm, subdivision)


def invert_lobatto_anchors(v_target: float, anchors: tuple[float, float, float]) -> float:
    """Inverts quadratic Lobatto anchors `[v_min, v_mid, v_max]` at `u in {-1, 0, +1}`."""
    v_min, v_mid, v_max = float(anchors[0]), float(anchors[1]), float(anchors[2])
    c1 = 0.5 * (v_max - v_min)
    c2 = 0.5 * (v_max + v_min) - v_mid
    dv = float(v_target) - v_mid
    if abs(c2) < 1e-6:
        if abs(c1) < 1e-9:
            return 0.0
        return max(-1.0, min(1.0, dv / c1))
    disc = max(0.0, c1 * c1 + 4.0 * c2 * dv)
    sqrt_disc = math.sqrt(disc)
    denom = c1 + sqrt_disc if c1 >= 0.0 else c1 - sqrt_disc
    if abs(denom) < 1e-9:
        return 0.0
    return max(-1.0, min(1.0, (2.0 * dv) / denom))


class TempoSyncResolution(StrictModel):
    """Result of resolving a tempo-synced target against `ControlCalibration`."""

    target_physical: float
    normalized_u: float
    switch_selection: tuple[str, str] | None = None
    was_clamped: bool = False


class ControlCalibration(StrictModel):
    """Cryptographically signed physical calibration metadata (`controls_schema[*].calibration`)."""

    role: ControlCalibrationRole
    lobatto_physical_anchors: tuple[float, float, float] | None = None
    conditional_physical_anchors: dict[str, tuple[float, float, float]] = Field(
        default_factory=dict
    )
    tempo_sync_default_subdivision: TempoSubdivision | None = None

    @model_validator(mode="after")
    def _validate_calibration(self) -> ControlCalibration:
        if self.lobatto_physical_anchors is not None:
            a0, a1, a2 = self.lobatto_physical_anchors
            if (
                not (math.isfinite(a0) and math.isfinite(a1) and math.isfinite(a2))
                or abs(a2 - a0) <= 1e-6
            ):
                raise DnamValidationError(
                    "lobatto_physical_anchors must be finite with non-zero span"
                )
        for cond_key, (c0, c1, c2) in self.conditional_physical_anchors.items():
            if "=" not in cond_key:
                raise DnamValidationError(
                    f"conditional_physical_anchors key '{cond_key}' must have "
                    "format 'switch_id=option_id'"
                )
            if (
                not (math.isfinite(c0) and math.isfinite(c1) and math.isfinite(c2))
                or abs(c2 - c0) <= 1e-6
            ):
                raise DnamValidationError(
                    f"conditional_physical_anchors['{cond_key}'] must be finite with non-zero span"
                )
        return self

    def resolve_conditional_target(
        self, v_target: float, *, fold_octave: bool = False
    ) -> TempoSyncResolution | None:
        """Resolves a target physical value across conditional or Lobatto anchors."""
        if self.conditional_physical_anchors:
            best_cond: tuple[str, tuple[float, float, float], float, bool] | None = None
            best_dist = float("inf")
            for cond_key, anchors in self.conditional_physical_anchors.items():
                lo = min(anchors[0], anchors[2])
                hi = max(anchors[0], anchors[2])
                v = float(v_target)
                folded = False
                if fold_octave and lo > 0.0:
                    while v > hi and v * 0.5 >= lo:
                        v *= 0.5
                        folded = True
                    while v < lo and v * 2.0 <= hi:
                        v *= 2.0
                        folded = True
                if lo <= v <= hi:
                    dist = 0.0
                    clamped = folded
                    v_eff = v
                elif v < lo:
                    dist = lo - v
                    clamped = True
                    v_eff = lo
                else:
                    dist = v - hi
                    clamped = True
                    v_eff = hi
                if dist < best_dist:
                    best_dist = dist
                    best_cond = (cond_key, anchors, v_eff, clamped)
                    if dist == 0.0 and not folded:
                        break
            if best_cond is not None:
                cond_key, anchors, v_eff, clamped = best_cond
                u = invert_lobatto_anchors(v_eff, anchors)
                sw_id, opt_id = cond_key.split("=", 1)
                return TempoSyncResolution(
                    target_physical=v_eff,
                    normalized_u=u,
                    switch_selection=(sw_id, opt_id),
                    was_clamped=clamped,
                )
        if self.lobatto_physical_anchors is not None:
            anchors = self.lobatto_physical_anchors
            lo = min(anchors[0], anchors[2])
            hi = max(anchors[0], anchors[2])
            v = float(v_target)
            clamped = False
            if fold_octave and lo > 0.0:
                while v > hi and v * 0.5 >= lo:
                    v *= 0.5
                    clamped = True
                while v < lo and v * 2.0 <= hi:
                    v *= 2.0
                    clamped = True
            if v < lo or v > hi:
                clamped = True
                v = max(lo, min(hi, v))
            return TempoSyncResolution(
                target_physical=v,
                normalized_u=invert_lobatto_anchors(v, anchors),
                switch_selection=None,
                was_clamped=clamped,
            )
        return None

    def resolve_tempo_sync(
        self,
        bpm: float,
        subdivision_override: TempoSubdivision | None = None,
        *,
        fold_octave: bool = False,
    ) -> TempoSyncResolution | None:
        """Resolves a host BPM and subdivision into a `TempoSyncResolution`."""
        sub = subdivision_override or self.tempo_sync_default_subdivision or "1/4"
        if self.role == "delay_time_ms":
            target = tempo_subdivision_delay_ms(bpm, sub)
        elif self.role == "lfo_rate_hz":
            target = tempo_subdivision_freq_hz(bpm, sub)
        else:
            return None
        return self.resolve_conditional_target(target, fold_octave=fold_octave)

    def stereo_width_for_option(self, option_label: str) -> float | None:
        """Maps a `stereo_mode` switch option label (`'Mono'` / `'Stereo'`) to `[0.0, 1.0]`."""
        if self.role != "stereo_mode":
            return None
        norm = option_label.strip().lower()
        if norm in ("mono", "m", "0", "sum"):
            return 0.0
        if norm in ("stereo", "st", "1", "wide", "full"):
            return 1.0
        return None


class ControlUiHints(StrictModel):
    """Intrinsic visual and gesture-mapping hints (`ui_hints`) for a parametric control."""

    section: str | None = None
    order: int = 0
    widget: str | None = None
    size: str | None = None
    polarity: str | None = None
    midpoint: float | None = None
    detents: list[float] = Field(default_factory=list)
    step: float | None = None
    decimals: int | None = Field(default=None, ge=0, le=8)
    scale_labels: dict[str, str] = Field(default_factory=dict)
    option_labels: dict[str, str] = Field(default_factory=dict)
    color_hex: str | None = None

    def skew_exponent(self, x_min: float, x_max: float) -> float:
        """Computes the closed-form power-skew exponent `gamma_skew` anchored at `midpoint`."""
        span = float(x_max - x_min)
        if span <= 1e-9:
            return 1.0
        if self.midpoint is not None:
            ratio = (float(self.midpoint) - float(x_min)) / span
            if 1e-4 < ratio < (1.0 - 1e-4) and abs(ratio - 0.5) > 1e-6:
                return math.log(0.5) / math.log(ratio)
        return 1.0

    def normalized_to_physical(self, t: float, x_min: float, x_max: float) -> float:
        """Maps a normalized UI gesture `t in [0, 1]` to physical value `x in [x_min, x_max]`."""
        t_clamped = max(0.0, min(1.0, float(t)))
        if t_clamped <= 0.0:
            return float(x_min)
        if t_clamped >= 1.0:
            return float(x_max)
        gamma = self.skew_exponent(x_min, x_max)
        curved = math.pow(t_clamped, 1.0 / gamma) if abs(gamma - 1.0) > 1e-6 else t_clamped
        return float(x_min) + (float(x_max) - float(x_min)) * curved

    def physical_to_normalized(self, x: float, x_min: float, x_max: float) -> float:
        """Maps a physical value `x in [x_min, x_max]` to normalized UI gesture `t in [0, 1]`."""
        span = float(x_max - x_min)
        if span <= 1e-9:
            return 0.0
        ratio = max(0.0, min(1.0, (float(x) - float(x_min)) / span))
        if ratio <= 0.0:
            return 0.0
        if ratio >= 1.0:
            return 1.0
        gamma = self.skew_exponent(x_min, x_max)
        return math.pow(ratio, gamma) if abs(gamma - 1.0) > 1e-6 else ratio


class ControlSchemaDef(StrictModel):
    """Schema definition for a single parametric control (`range` or `switch`)."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    kind: ControlKind = Field(alias="type")
    unit: str | None = None
    lobatto_degree: int | None = Field(default=None, ge=0, le=3)
    anchor_nodes_norm: list[float] | None = None
    declared_range: tuple[float, float] | None = None
    safe_clamped_range: tuple[float, float] | None = None
    default: float | str
    conditional_min: dict[str, float] = Field(default_factory=dict)
    conditional_max: dict[str, float] = Field(default_factory=dict)
    declared_options: list[str] | None = None
    active_options: list[str] | None = None
    clamp_warning: str | None = None
    calibration: ControlCalibration | None = None
    ui_hints: ControlUiHints = Field(default_factory=ControlUiHints)
    user_metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    @model_validator(mode="after")
    def _validate_control_invariants(self) -> ControlSchemaDef:
        if self.kind == ControlKind.RANGE:
            if self.declared_range is None:
                raise DnamValidationError(f"Range control '{self.id}' requires declared_range")
            if self.declared_range[1] - self.declared_range[0] <= 1e-6:
                raise DnamValidationError(
                    f"Range control '{self.id}' declared_range span must be > 0"
                )
            if not isinstance(self.default, int | float) or isinstance(self.default, bool):
                raise DnamValidationError(
                    f"Range control '{self.id}' default must be numeric, got {self.default!r}"
                )
        else:
            if not self.declared_options or len(self.declared_options) > 16:
                raise DnamValidationError(
                    f"Switch control '{self.id}' requires 1..=16 declared_options"
                )
            if not isinstance(self.default, str) or self.default not in self.declared_options:
                raise DnamValidationError(
                    f"Switch control '{self.id}' default must be one of declared_options"
                )
        return self


class PresetDef(StrictModel):
    """Named parametric control snapshot (`presets[*]`)."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str | None = None
    controls: dict[str, float | str] = Field(default_factory=dict)
    user_metadata: dict[str, Any] = Field(default_factory=dict)


class ClampedControlValue(StrictModel):
    """Clamping diagnostic for a single parametric control."""

    id: str
    clamped_physical: float | str
    clamped_numeric: float
    normalized_u: float
    was_clamped: bool
    clamp_warning: str | None = None


class ClampedControlsResult(StrictModel):
    """Aggregate result of 3-Phase control clamping across all controls in a model."""

    controls: dict[str, ClampedControlValue]

    @property
    def physical_values(self) -> dict[str, float | str]:
        """Returns a mapping of `control_id -> clamped_physical` values."""
        return {k: v.clamped_physical for k, v in self.controls.items()}

    @property
    def any_clamped(self) -> bool:
        """Returns `True` if at least one control value was clamped."""
        return any(v.was_clamped for v in self.controls.values())


class LobattoParams(StrictModel):
    """Validated parameters for Node-Anchored Newton/Lobatto polynomial basis (`CL-13`)."""

    anchors: tuple[float, float, float] = (0.0, -1.0, 1.0)
    degree: int = Field(default=2, ge=0, le=3)

    @field_validator("anchors")
    @classmethod
    def _validate_anchors(cls, v: tuple[float, float, float]) -> tuple[float, float, float]:
        for idx, node in enumerate(v):
            if not (-1.0 <= node <= 1.0):
                raise DnamValidationError(f"Lobatto anchor[{idx}] = {node} must lie in [-1.0, 1.0]")
        return v
