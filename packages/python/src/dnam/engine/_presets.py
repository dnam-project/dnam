"""Live `Presets` (`Mapping`) facet on `Engine` (`ref[sdk.python.api.engine-facets]`)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from dnam._native import PyEngine
from dnam.controls._space import ClampResult
from dnam.engine._controls import Controls
from dnam.errors import ControlError
from dnam.schema.controls import PresetDef


class Presets(Mapping[str, PresetDef]):
    """Read-only `Mapping` of model presets with atomic `apply(preset_id)` execution."""

    __slots__ = ("_controls", "_native", "_presets")

    def __init__(
        self,
        native: PyEngine,
        controls: Controls,
        presets: Mapping[str, PresetDef],
    ) -> None:
        self._native = native
        self._controls = controls
        self._presets = presets

    def __len__(self) -> int:
        return len(self._presets)

    def __iter__(self) -> Iterator[str]:
        return iter(self._presets)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._presets

    def __getitem__(self, preset_id: str) -> PresetDef:
        try:
            return self._presets[preset_id]
        except KeyError:
            raise ControlError(f"Unknown preset ID: '{preset_id}'") from None

    def apply(self, preset_id: str, *, immediate: bool = False) -> ClampResult:
        """Applies a named preset atomically (`ref[dsp.ctrl.switch.conditional-batch]`)."""
        preset = self[preset_id]
        if immediate:
            self._controls.reset()
            if preset.controls:
                self._controls.snap(preset.controls)
            return self._controls.clamped
        self._native.apply_preset(preset_id)
        self._controls._invalidate()
        return self._controls.clamped

    def __repr__(self) -> str:
        ids = ", ".join(list(self._presets)[:4])
        if len(self._presets) > 4:
            ids += ", ..."
        return f"<Presets ({len(self._presets)}) [{ids}]>"
