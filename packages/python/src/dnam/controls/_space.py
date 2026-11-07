"""Offline 3-Phase control-space clamping (`ControlSpace`, `ClampResult`, `ClampedControl`)
(`ref[sdk.python.controls.space-and-tempo]`).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from dnam._native import PyControlSpace
from dnam.errors import ControlError, ValidationError
from dnam.schema._base import StrictModel
from dnam.schema.controls import ControlKind, ControlSchemaDef, PresetDef
from dnam.schema.header import DnamJsonHeader


class ClampedControl(StrictModel):
    """Clamping result for a single control (`range` or `switch`)."""

    id: str
    value: float | str
    numeric_value: float
    normalized_u: float
    was_clamped: bool
    clamp_warning: str | None = None


class ClampResult(StrictModel):
    """Aggregate 3-Phase clamping result across all controls in a `ControlSpace`."""

    controls: dict[str, ClampedControl]
    active_clamp_mask: int

    @property
    def any_clamped(self) -> bool:
        """`True` if any control was clamped."""
        return self.active_clamp_mask != 0 or any(c.was_clamped for c in self.controls.values())

    @property
    def clamped_ids(self) -> tuple[str, ...]:
        """Tuple of control IDs that were clamped."""
        return tuple(cid for cid, c in self.controls.items() if c.was_clamped)

    def __getitem__(self, control_id: str) -> ClampedControl:
        try:
            return self.controls[control_id]
        except KeyError:
            raise ControlError(f"Unknown control ID: '{control_id}'") from None


class ControlSpace:
    """Pre-compiled 3-Phase control clamping space (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`,
    `[CLAMP-STABILITY]`).
    """

    __slots__ = (
        "_clamp_warnings_by_idx",
        "_default_clamp_result",
        "_default_numeric_tuple",
        "_header_json",
        "_ids_by_idx",
        "_index_by_id",
        "_is_switch_by_idx",
        "_native",
        "_ordered_defs",
        "_presets_proxy",
        "_schema_proxy",
        "_switch_opt_map_by_id",
        "_switch_opts_by_idx",
    )

    def __init__(self, header: DnamJsonHeader, *, header_json: str | None = None) -> None:
        raw_json = header_json if header_json is not None else header.to_json()
        self._header_json = raw_json
        self._native = PyControlSpace(raw_json)
        ordered = tuple(header.controls_schema)
        self._ordered_defs = ordered
        self._ids_by_idx = tuple(c.id for c in ordered)
        self._index_by_id = {c.id: idx for idx, c in enumerate(ordered)}
        self._is_switch_by_idx = tuple(c.kind == ControlKind.SWITCH for c in ordered)
        self._switch_opts_by_idx: tuple[tuple[str, ...] | None, ...] = tuple(
            tuple(c.declared_options)
            if c.kind == ControlKind.SWITCH and c.declared_options
            else None
            for c in ordered
        )
        self._switch_opt_map_by_id: dict[str, dict[str, float]] = {
            c.id: {opt: float(i) for i, opt in enumerate(c.declared_options or ())}
            for c in ordered
            if c.kind == ControlKind.SWITCH
        }
        self._clamp_warnings_by_idx = tuple(c.clamp_warning for c in ordered)
        self._schema_proxy: Mapping[str, ControlSchemaDef] = MappingProxyType(
            {c.id: c for c in ordered}
        )
        self._presets_proxy: Mapping[str, PresetDef] = MappingProxyType(
            {p.id: p for p in header.presets}
        )
        self._default_numeric_tuple: tuple[float, ...] = tuple(
            self.resolve_numeric_value(c.id, c.default) for c in ordered
        )
        self._default_clamp_result: ClampResult | None = None

    @classmethod
    def from_header(cls, header: DnamJsonHeader) -> ControlSpace:
        """Compiles a `ControlSpace` from a validated `DnamJsonHeader`."""
        return cls(header)

    @property
    def schema(self) -> Mapping[str, ControlSchemaDef]:
        """Read-only mapping of control ID to `ControlSchemaDef`."""
        return self._schema_proxy

    @property
    def presets(self) -> Mapping[str, PresetDef]:
        """Read-only mapping of preset ID to `PresetDef`."""
        return self._presets_proxy

    def __len__(self) -> int:
        return len(self._ordered_defs)

    def __contains__(self, control_id: object) -> bool:
        return isinstance(control_id, str) and control_id in self._index_by_id

    def __repr__(self) -> str:
        ids = ", ".join(self._ids_by_idx[:4])
        if len(self._ids_by_idx) > 4:
            ids += ", ..."
        return f"<ControlSpace controls={len(self._ids_by_idx)} [{ids}]>"

    def index_of(self, control_id: str) -> int:
        """Returns the 0-based index of `control_id`, or raises `ControlError`."""
        try:
            return self._index_by_id[control_id]
        except KeyError:
            raise ControlError(f"Unknown control ID: '{control_id}'") from None

    def resolve_numeric_value(self, control_id: str, value: float | int | str) -> float:
        """Converts a physical `float` or switch option `str` into a numeric engine value."""
        idx = self._index_by_id.get(control_id)
        if idx is None:
            raise ControlError(f"Unknown control ID: '{control_id}'")
        if self._is_switch_by_idx[idx]:
            if type(value) is str:
                opt_map = self._switch_opt_map_by_id[control_id]
                found = opt_map.get(value)
                if found is not None:
                    return found
                opts = self._switch_opts_by_idx[idx] or ()
                raise ControlError(
                    f"Unknown option '{value}' for switch '{control_id}' (declared: {list(opts)})"
                )
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                opts = self._switch_opts_by_idx[idx] or ()
                opt_idx = round(float(value))
                if 0 <= opt_idx < len(opts):
                    return float(opt_idx)
                raise ControlError(
                    f"Switch index {opt_idx} out of bounds for '{control_id}' (0..{len(opts)})"
                )
            raise ControlError(f"Invalid switch value {value!r} for '{control_id}'")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ControlError(
                f"Range control '{control_id}' requires a numeric value, got {value!r}"
            )
        val_f = value if type(value) is float else float(value)
        if not (val_f == val_f and abs(val_f) != float("inf")):
            raise ValidationError(f"Control '{control_id}' value must be finite, got {value!r}")
        return val_f

    def format_value(self, control_id: str, numeric_value: float) -> float | str:
        """Formats a numeric engine value as a `str` option label for switches or `float` for
        ranges.
        """
        idx = self._index_by_id.get(control_id)
        if idx is None:
            raise ControlError(f"Unknown control ID: '{control_id}'")
        opts = self._switch_opts_by_idx[idx]
        if opts is not None:
            return opts[max(0, min(round(numeric_value), len(opts) - 1))]
        return float(numeric_value)

    def default_numeric_values(self) -> list[float]:
        """Returns the list of default numeric values across all controls in schema order."""
        return list(self._default_numeric_tuple)

    def clamp(
        self,
        values: Mapping[str, float | int | str] | None = None,
    ) -> ClampResult:
        """Evaluates vectorized 3-Phase control clamping in Rust."""
        if not values:
            if self._default_clamp_result is None:
                self._default_clamp_result = self._clamp_numeric_vec(
                    list(self._default_numeric_tuple)
                )
            return self._default_clamp_result
        raw_vec = list(self._default_numeric_tuple)
        index_by_id = self._index_by_id
        for cid, val in values.items():
            try:
                idx = index_by_id[cid]
            except KeyError:
                raise ControlError(f"Unknown control ID: '{cid}'") from None
            raw_vec[idx] = self.resolve_numeric_value(cid, val)
        return self._clamp_numeric_vec(raw_vec)

    def _clamp_numeric_vec(self, raw_vec: list[float]) -> ClampResult:
        clamped_num, norm_u, was_clamped, mask = self._native.clamp(raw_vec)
        out: dict[str, ClampedControl] = {}
        ids = self._ids_by_idx
        switch_opts = self._switch_opts_by_idx
        warnings = self._clamp_warnings_by_idx
        construct_ctrl = ClampedControl._construct_trusted_dict
        for idx in range(len(ids)):
            cid = ids[idx]
            c_num = clamped_num[idx]
            opts = switch_opts[idx]
            val: float | str = (
                opts[max(0, min(round(c_num), len(opts) - 1))] if opts is not None else c_num
            )
            clamped_flag = was_clamped[idx]
            out[cid] = construct_ctrl(
                {
                    "id": cid,
                    "value": val,
                    "numeric_value": c_num,
                    "normalized_u": norm_u[idx],
                    "was_clamped": clamped_flag,
                    "clamp_warning": warnings[idx] if clamped_flag else None,
                }
            )
        return ClampResult._construct_trusted_dict({"controls": out, "active_clamp_mask": mask})

    def to_physical(self, control_id: str, normalized_gesture: float) -> float:
        """Maps a normalized UI gesture `t in [0.0, 1.0]` to physical value."""
        idx = self.index_of(control_id)
        return float(self._native.to_physical(idx, float(normalized_gesture)))

    def to_normalized(self, control_id: str, physical_value: float) -> float:
        """
        Maps a physical value `x in [x_min, x_max]` to normalized UI gesture `t in [0.0, 1.0]`.
        """
        idx = self.index_of(control_id)
        return float(self._native.to_normalized(idx, float(physical_value)))
