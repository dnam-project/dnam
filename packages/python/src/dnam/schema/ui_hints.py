"""UI interaction hints and midpoint-anchored power-skew gesture mapping
(`ref[sdk.python.schema.rust-parity]`).
"""

from __future__ import annotations

import math

from pydantic import Field

from dnam.schema._base import StrictModel


class ControlUiHints(StrictModel):
    """Intrinsic visual and interaction hints for a parametric control (`ui_hints`)."""

    section: str | None = None
    order: int = 0
    widget: str | None = None
    size: str | None = None
    polarity: str | None = None
    midpoint: float | None = None
    detents: list[float] = Field(default_factory=list)
    step: float | None = None
    decimals: int | None = Field(default=None, ge=0, le=255)
    scale_labels: dict[str, str] = Field(default_factory=dict)
    option_labels: dict[str, str] = Field(default_factory=dict)
    color_hex: str | None = None

    def skew_exponent(self, x_min: float, x_max: float) -> float:
        """Computes the closed-form power-skew exponent for `[x_min, x_max]` anchored at
        `midpoint`.
        """
        span = float(x_max) - float(x_min)
        if span <= 1e-9:
            return 1.0
        if self.midpoint is not None:
            ratio = (float(self.midpoint) - float(x_min)) / span
            if 1e-4 < ratio < (1.0 - 1e-4) and abs(ratio - 0.5) > 1e-6:
                return math.log(0.5) / math.log(ratio)
        return 1.0

    def normalized_to_physical(self, t: float, x_min: float, x_max: float) -> float:
        """Maps normalized gesture `t in [0.0, 1.0]` to physical value `x in [x_min, x_max]`."""
        t_clamped = min(max(float(t), 0.0), 1.0)
        if t_clamped <= 0.0:
            return float(x_min)
        if t_clamped >= 1.0:
            return float(x_max)
        gamma = self.skew_exponent(x_min, x_max)
        return float(x_min) + (float(x_max) - float(x_min)) * (t_clamped ** (1.0 / gamma))

    def physical_to_normalized(self, x: float, x_min: float, x_max: float) -> float:
        """Maps physical value `x in [x_min, x_max]` to normalized gesture `t in [0.0, 1.0]`."""
        span = float(x_max) - float(x_min)
        if span <= 1e-9:
            return 0.0
        ratio = min(max((float(x) - float(x_min)) / span, 0.0), 1.0)
        if ratio <= 0.0:
            return 0.0
        if ratio >= 1.0:
            return 1.0
        gamma = self.skew_exponent(x_min, x_max)
        return ratio**gamma
