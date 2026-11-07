"""UI interaction hints and midpoint-anchored power-skew gesture mapping
(`ref[sdk.python.schema.rust-parity]`).
"""

from __future__ import annotations

from dnam._native import (
    native_ui_normalized_to_physical,
    native_ui_physical_to_normalized,
    native_ui_skew_exponent,
)
from dnam.schema._base import StrictModel, strict_field


class ControlUiHints(StrictModel):
    """Intrinsic visual and interaction hints for a parametric control (`ui_hints`)."""

    section: str | None = None
    order: int = 0
    widget: str | None = None
    size: str | None = None
    polarity: str | None = None
    midpoint: float | None = None
    detents: list[float] = strict_field(default_factory=list)
    step: float | None = None
    decimals: int | None = strict_field(default=None, ge=0, le=255)
    scale_labels: dict[str, str] = strict_field(default_factory=dict)
    option_labels: dict[str, str] = strict_field(default_factory=dict)
    color_hex: str | None = None

    def skew_exponent(self, x_min: float, x_max: float) -> float:
        """Computes the closed-form power-skew exponent for `[x_min, x_max]` anchored at
        `midpoint`.
        """
        return float(native_ui_skew_exponent(self.midpoint, float(x_min), float(x_max)))

    def normalized_to_physical(self, t: float, x_min: float, x_max: float) -> float:
        """Maps normalized gesture `t in [0.0, 1.0]` to physical value `x in [x_min, x_max]`."""
        return float(
            native_ui_normalized_to_physical(self.midpoint, float(t), float(x_min), float(x_max))
        )

    def physical_to_normalized(self, x: float, x_min: float, x_max: float) -> float:
        """Maps physical value `x in [x_min, x_max]` to normalized gesture `t in [0.0, 1.0]`."""
        return float(
            native_ui_physical_to_normalized(self.midpoint, float(x), float(x_min), float(x_max))
        )
