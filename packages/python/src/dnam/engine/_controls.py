"""Live `Controls` (`MutableMapping`) facet on `Engine` (`ref[sdk.python.api.engine-facets]`)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping

from dnam._native import PyEngine
from dnam.controls._space import ClampResult, ControlSpace
from dnam.errors import ControlError
from dnam.schema.controls import ControlSchemaDef


class Controls(MutableMapping[str, float | str]):
    """Live `MutableMapping` proxy over an `Engine`'s $3\\text{ kHz}$ parametric control bus."""

    __slots__ = ("_cached_clamped", "_native", "_space")

    def __init__(self, native: PyEngine, space: ControlSpace) -> None:
        self._native = native
        self._space = space
        self._cached_clamped: ClampResult | None = None

    def _invalidate(self) -> None:
        self._cached_clamped = None

    @property
    def schema(self) -> Mapping[str, ControlSchemaDef]:
        """Read-only mapping of control ID to `ControlSchemaDef`."""
        return self._space.schema

    @property
    def space(self) -> ControlSpace:
        """Underlying `ControlSpace` clamper."""
        return self._space

    @property
    def clamped(self) -> ClampResult:
        """Current 3-Phase clamping diagnostics for the engine's live control values."""
        cached = self._cached_clamped
        if cached is None:
            raw_vals = self._native.raw_control_values()
            cached = self._space._clamp_numeric_vec(raw_vals)
            self._cached_clamped = cached
        return cached

    def __len__(self) -> int:
        return len(self._space)

    def __iter__(self) -> Iterator[str]:
        return iter(self._space.schema)

    def __contains__(self, key: object) -> bool:
        return key in self._space

    def __getitem__(self, control_id: str) -> float | str:
        idx = self._space.index_of(control_id)
        c_val = self._native.clamped_control_value(idx)
        return self._space.format_value(control_id, c_val)

    def __setitem__(self, control_id: str, value: float | int | str) -> None:
        """Sets a control value with smooth $3\\text{ kHz}$ anti-zipper interpolation."""
        idx = self._space.index_of(control_id)
        num_val = self._space.resolve_numeric_value(control_id, value)
        self._native.set_control_value(idx, num_val)
        self._cached_clamped = None

    def __delitem__(self, control_id: str) -> None:
        """Resets a single control back to its schema default value."""
        idx = self._space.index_of(control_id)
        def_num = self._space._default_numeric_tuple[idx]
        self._native.set_control_immediate(idx, def_num)
        self._cached_clamped = None

    def snap(
        self,
        values: Mapping[str, float | int | str] | None = None,
        /,
        **kwargs: float | int | str,
    ) -> ClampResult:
        """Immediately snaps one or more controls without anti-zipper lag."""
        space = self._space
        updates: list[tuple[int, float]] = []
        if values is not None:
            for cid, val in values.items():
                updates.append((space.index_of(cid), space.resolve_numeric_value(cid, val)))
        if kwargs:
            for cid, val in kwargs.items():
                updates.append((space.index_of(cid), space.resolve_numeric_value(cid, val)))
        if not updates:
            raise ControlError("Controls.snap requires at least one control assignment")
        if len(updates) == 1:
            idx, num_val = updates[0]
            self._native.set_control_immediate(idx, num_val)
        else:
            self._native.set_controls_immediate_batch(updates)
        self._cached_clamped = None
        return self.clamped

    def reset(self) -> ClampResult:
        """Immediately resets all controls back to their schema defaults."""
        self._native.reset_controls()
        self._cached_clamped = None
        return self.clamped

    def to_physical(self, control_id: str, normalized_gesture: float) -> float:
        """Maps a normalized UI gesture `t in [0.0, 1.0]` to physical value."""
        return self._space.to_physical(control_id, normalized_gesture)

    def to_normalized(self, control_id: str, physical_value: float) -> float:
        """Maps a physical value to normalized UI gesture `t in [0.0, 1.0]`."""
        return self._space.to_normalized(control_id, physical_value)

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={self[k]!r}" for k in list(self._space.schema)[:4])
        if len(self._space) > 4:
            items += ", ..."
        return f"<Controls ({len(self._space)}) {{{items}}}>"
