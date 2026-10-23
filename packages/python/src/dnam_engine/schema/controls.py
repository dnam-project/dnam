"""Parametric control schemas (`ControlSchemaDef`), UI hints (`ControlUiHints`),
presets (`PresetDef`), clamping results (`ClampedControlsResult`), and `LobattoParams`."""

from __future__ import annotations

import math
from typing import Any

from pydantic import ConfigDict, Field, field_validator, model_validator

from dnam_engine.exceptions import DnamValidationError
from dnam_engine.schema.base import ControlKind, StrictModel


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
