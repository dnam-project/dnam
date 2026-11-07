"""Live `Presets` (`Mapping`) facet on `Engine` (`ref[sdk.python.api.engine-facets]`,
`ref[sdk.python.engine.free-threading]`).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING

from dnam.controls._space import ClampResult
from dnam.errors import ControlError
from dnam.schema.controls import PresetDef

if TYPE_CHECKING:
    from dnam.engine._engine import Engine


class Presets(Mapping[str, PresetDef]):
    """Read-only `Mapping` of model presets with atomic `apply(preset_id)` execution."""

    __slots__ = ("_engine",)

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def __len__(self) -> int:
        return len(self._engine._state._presets_map)

    def __iter__(self) -> Iterator[str]:
        return iter(self._engine._state._presets_map)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._engine._state._presets_map

    def __getitem__(self, preset_id: str) -> PresetDef:
        try:
            return self._engine._state._presets_map[preset_id]
        except KeyError:
            raise ControlError(f"Unknown preset ID: '{preset_id}'") from None

    def apply(self, preset_id: str, *, immediate: bool = False) -> ClampResult:
        """Applies a named preset atomically (`ref[dsp.ctrl.switch.conditional-batch]`)."""
        with self._engine._lock:
            state = self._engine._state
            try:
                preset = state._presets_map[preset_id]
            except KeyError:
                raise ControlError(f"Unknown preset ID: '{preset_id}'") from None
            if immediate:
                self._engine._controls.reset()
                if preset.controls:
                    self._engine._controls.snap(preset.controls)
                return self._engine._controls.clamped
            state._clamped_cache = None
            state._current_cache = None
            state._native.apply_preset(preset_id)
            return self._engine._controls.clamped

    def __repr__(self) -> str:
        presets_map = self._engine._state._presets_map
        ids = ", ".join(list(presets_map)[:4])
        if len(presets_map) > 4:
            ids += ", ..."
        return f"<Presets ({len(presets_map)}) [{ids}]>"
