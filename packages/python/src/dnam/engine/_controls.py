"""Live `Controls` (`MutableMapping`) facet on `Engine` (`ref[sdk.python.api.engine-facets]`,
`ref[sdk.python.engine.free-threading]`).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
from typing import TYPE_CHECKING

from dnam.controls._space import ClampResult, ControlSpace
from dnam.errors import ControlError
from dnam.schema.controls import ControlSchemaDef

if TYPE_CHECKING:
    from dnam.engine._engine import Engine


class Controls(MutableMapping[str, float | str]):
    """Live `MutableMapping` proxy over an `Engine`'s $3\\text{ kHz}$ parametric control bus."""

    __slots__ = ("_engine",)

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    @property
    def schema(self) -> Mapping[str, ControlSchemaDef]:
        """Read-only mapping of control ID to `ControlSchemaDef`."""
        return self._engine._state._control_space.schema

    @property
    def space(self) -> ControlSpace:
        """Underlying `ControlSpace` clamper."""
        return self._engine._state._control_space

    @property
    def clamped(self) -> ClampResult:
        """Current 3-Phase clamping diagnostics for the engine's live control values."""
        with self._engine._lock:
            state = self._engine._state
            cached = state._clamped_cache
            if cached is None:
                raw_vals = state._native.raw_control_values()
                cached = state._control_space._clamp_numeric_vec(raw_vals)
                state._clamped_cache = cached
                state._current_cache = {cid: ctrl.value for cid, ctrl in cached.controls.items()}
            return cached

    def __len__(self) -> int:
        return len(self._engine._state._control_space)

    def __iter__(self) -> Iterator[str]:
        return iter(self._engine._state._control_space.schema)

    def __contains__(self, key: object) -> bool:
        return key in self._engine._state._control_space

    def __getitem__(self, control_id: str) -> float | str:
        state = self._engine._state
        cur = state._current_cache
        if cur is not None and control_id in cur:
            return cur[control_id]
        space = state._control_space
        idx = space.index_of(control_id)
        c_val = state._native.clamped_control_value(idx)
        return space.format_value(control_id, c_val)

    def __setitem__(self, control_id: str, value: float | int | str) -> None:
        """Sets a control value with smooth $3\\text{ kHz}$ anti-zipper interpolation."""
        with self._engine._lock:
            state = self._engine._state
            space = state._control_space
            idx = space.index_of(control_id)
            num_val = space.resolve_numeric_value(control_id, value)
            state._clamped_cache = None
            state._current_cache = None
            state._native.set_control_value(idx, num_val)

    def __delitem__(self, control_id: str) -> None:
        """Resets a single control back to its schema default value."""
        with self._engine._lock:
            state = self._engine._state
            space = state._control_space
            idx = space.index_of(control_id)
            def_num = space._default_numeric_tuple[idx]
            state._clamped_cache = None
            state._current_cache = None
            state._native.set_control_immediate(idx, def_num)

    def snap(
        self,
        values: Mapping[str, float | int | str] | None = None,
        /,
        **kwargs: float | int | str,
    ) -> ClampResult:
        """Immediately snaps one or more controls without anti-zipper lag."""
        with self._engine._lock:
            state = self._engine._state
            space = state._control_space
            updates: list[tuple[int, float]] = []
            if values is not None:
                for cid, val in values.items():
                    updates.append((space.index_of(cid), space.resolve_numeric_value(cid, val)))
            if kwargs:
                for cid, val in kwargs.items():
                    updates.append((space.index_of(cid), space.resolve_numeric_value(cid, val)))
            if not updates:
                raise ControlError("Controls.snap requires at least one control assignment")
            state._clamped_cache = None
            state._current_cache = None
            if len(updates) == 1:
                idx, num_val = updates[0]
                state._native.set_control_immediate(idx, num_val)
            else:
                state._native.set_controls_immediate_batch(updates)
            return self.clamped

    def reset(self) -> ClampResult:
        """Immediately resets all controls back to their schema defaults."""
        with self._engine._lock:
            state = self._engine._state
            state._clamped_cache = None
            state._current_cache = None
            state._native.reset_controls()
            return self.clamped

    def to_physical(self, control_id: str, normalized_gesture: float) -> float:
        """Maps a normalized UI gesture `t in [0.0, 1.0]` to physical value."""
        return self._engine._state._control_space.to_physical(control_id, normalized_gesture)

    def to_normalized(self, control_id: str, physical_value: float) -> float:
        """Maps a physical value to normalized UI gesture `t in [0.0, 1.0]`."""
        return self._engine._state._control_space.to_normalized(control_id, physical_value)

    def __repr__(self) -> str:
        state = self._engine._state
        space = state._control_space
        items = ", ".join(f"{k}={self[k]!r}" for k in list(space.schema)[:4])
        if len(space) > 4:
            items += ", ..."
        return f"<Controls ({len(space)}) {{{items}}}>"
