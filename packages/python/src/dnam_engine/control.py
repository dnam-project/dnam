"""$3\\text{ kHz}$ Parametric Control Bus interface: `ControlSpace`, live `ControlRack`,
callable `LobattoBasis` & `Rank4Fm`, and `chebyshev_lfo` / `trust_region_softclip` kernels."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dnam_engine import _native
from dnam_engine.exceptions import DnamControlError, DnamValidationError, call_native
from dnam_engine.schema.base import ControlKind
from dnam_engine.schema.controls import (
    ClampedControlsResult,
    ClampedControlValue,
    ControlSchemaDef,
    LobattoParams,
    PresetDef,
)
from dnam_engine.schema.header import DnamJsonHeader


class ControlSpace:
    """First-class 3-Phase control clamper (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]`)
    and physical <-> normalized gesture mapper for a model's `controls_schema`."""

    def __init__(
        self,
        controls: Sequence[ControlSchemaDef] | DnamJsonHeader,
        presets: Sequence[PresetDef] = (),
        *,
        _native_space: _native.NativeControlSpace | None = None,
    ) -> None:
        if isinstance(controls, DnamJsonHeader):
            self._header = controls
        else:
            self._header = DnamJsonHeader.default_reference().model_copy(
                update={"controls_schema": list(controls), "presets": list(presets)}
            )
        self._schema_list: tuple[ControlSchemaDef, ...] = tuple(self._header.controls_schema)
        self._schema_by_id: dict[str, ControlSchemaDef] = {c.id: c for c in self._schema_list}
        self._index_by_id: dict[str, int] = {c.id: idx for idx, c in enumerate(self._schema_list)}
        self._presets_by_name: dict[str, PresetDef] = {p.id: p for p in self._header.presets}
        if _native_space is not None:
            self._native_space = _native_space
        else:
            self._native_space = call_native(
                _native.NativeControlSpace,
                self._header.model_dump_json(by_alias=True),
            )

    @property
    def schema(self) -> Mapping[str, ControlSchemaDef]:
        """Returns the control schemas keyed by control `id`."""
        return self._schema_by_id

    @property
    def presets(self) -> Mapping[str, PresetDef]:
        """Returns the built-in presets keyed by preset `name`."""
        return self._presets_by_name

    def index_of(self, control_id: str) -> int:
        """Returns the positional slot index (`0..15`) of `control_id`."""
        if control_id not in self._index_by_id:
            raise DnamControlError(f"Unknown control id: '{control_id}'")
        return self._index_by_id[control_id]

    def resolve_numeric_value(self, control_id: str, value: float | str) -> float:
        """Converts a `range` float or `switch` option string/index into a numeric slot value."""
        if control_id not in self._schema_by_id:
            raise DnamControlError(f"Unknown control id: '{control_id}'")
        ctrl = self._schema_by_id[control_id]
        if ctrl.kind == ControlKind.SWITCH:
            opts = ctrl.declared_options or []
            if isinstance(value, str):
                if value not in opts:
                    raise DnamControlError(
                        f"Unknown option '{value}' for switch '{control_id}' (declared: {opts})"
                    )
                return float(opts.index(value))
            idx = round(float(value))
            if not (0 <= idx < len(opts)):
                raise DnamControlError(
                    f"Switch option index {idx} out of range for '{control_id}' (declared: {opts})"
                )
            return float(idx)
        if isinstance(value, str):
            raise DnamControlError(
                f"Range control '{control_id}' requires a numeric value, got {value!r}"
            )
        return float(value)

    def clamp(self, raw_controls: Mapping[str, float | str] | None = None) -> ClampedControlsResult:
        """Evaluates 3-Phase control clamping (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, etc.)."""
        req_map = raw_controls or {}
        for unknown_id in req_map:
            if unknown_id not in self._schema_by_id:
                raise DnamControlError(f"Unknown control id: '{unknown_id}'")
        numeric_vec = [
            self.resolve_numeric_value(ctrl.id, req_map.get(ctrl.id, ctrl.default))
            for ctrl in self._schema_list
        ]

        clamped_nums, norm_us, was_clamped_flags = self._native_space.clamp(numeric_vec)
        out_map: dict[str, ClampedControlValue] = {}
        for idx, ctrl in enumerate(self._schema_list):
            clamped_num = float(clamped_nums[idx])
            if ctrl.kind == ControlKind.SWITCH:
                opts = ctrl.declared_options or []
                opt_idx = max(0, min(len(opts) - 1, round(clamped_num)))
                clamped_phys: float | str = opts[opt_idx]
            else:
                clamped_phys = clamped_num
            out_map[ctrl.id] = ClampedControlValue._construct_trusted(
                id=ctrl.id,
                clamped_physical=clamped_phys,
                clamped_numeric=clamped_num,
                normalized_u=float(norm_us[idx]),
                was_clamped=was_clamped_flags[idx],
                clamp_warning=ctrl.clamp_warning,
            )
        return ClampedControlsResult._construct_trusted(controls=out_map)

    def to_physical(self, control_id: str, normalized_gesture: float) -> float:
        """Maps a UI gesture `t in [0, 1]` to physical value via `ui_hints` power-skew."""
        if control_id not in self._schema_by_id:
            raise DnamControlError(f"Unknown control id: '{control_id}'")
        ctrl = self._schema_by_id[control_id]
        if ctrl.kind != ControlKind.RANGE or ctrl.declared_range is None:
            raise DnamControlError(f"Control '{control_id}' is not a range control")
        return ctrl.ui_hints.normalized_to_physical(
            normalized_gesture, ctrl.declared_range[0], ctrl.declared_range[1]
        )

    def to_normalized(self, control_id: str, physical_value: float) -> float:
        """Maps a physical value to a UI gesture `t in [0, 1]` via `ui_hints` power-skew."""
        if control_id not in self._schema_by_id:
            raise DnamControlError(f"Unknown control id: '{control_id}'")
        ctrl = self._schema_by_id[control_id]
        if ctrl.kind != ControlKind.RANGE or ctrl.declared_range is None:
            raise DnamControlError(f"Control '{control_id}' is not a range control")
        return ctrl.ui_hints.physical_to_normalized(
            physical_value, ctrl.declared_range[0], ctrl.declared_range[1]
        )


class ControlRack(MutableMapping[str, float | str]):
    """Live `MutableMapping[str, float | str]` proxy bound to a running `DnamEngine` instance.

    - Item assignment (`engine.controls["gain"] = 7.5`) applies $3\\text{ kHz}$ anti-zipper slewing.
    - `.snap(gain=7.5, boost="on")` snaps controls immediately with zero slew.
    - `.apply_preset("warm_slapback")` applies a named preset from the container's preset table.
    """

    def __init__(
        self,
        space: ControlSpace,
        *,
        set_slew_fn: Callable[[int, float], None],
        set_snap_fn: Callable[[int, float], None],
    ) -> None:
        self._space = space
        self._set_slew = set_slew_fn
        self._set_snap = set_snap_fn
        self._raw_values: dict[str, float | str] = {
            cid: ctrl.default for cid, ctrl in space.schema.items()
        }
        self._cached_clamped: ClampedControlsResult | None = None

    @property
    def schema(self) -> Mapping[str, ControlSchemaDef]:
        """Returns the control schemas keyed by control `id`."""
        return self._space.schema

    @property
    def clamped(self) -> ClampedControlsResult:
        """Returns the live 3-Phase clamping diagnostics for all current control values."""
        if self._cached_clamped is None:
            self._cached_clamped = self._space.clamp(self._raw_values)
        return self._cached_clamped

    def __getitem__(self, control_id: str) -> float | str:
        if control_id not in self._space.schema:
            raise DnamControlError(f"Unknown control id: '{control_id}'")
        return self.clamped.controls[control_id].clamped_physical

    def __setitem__(self, control_id: str, value: float | str) -> None:
        idx = self._space.index_of(control_id)
        num_val = self._space.resolve_numeric_value(control_id, value)
        self._raw_values[control_id] = value
        self._cached_clamped = None
        self._set_slew(idx, num_val)

    def __delitem__(self, control_id: str) -> None:
        """Deleting a control resets it to its schema default value."""
        if control_id not in self._space.schema:
            raise DnamControlError(f"Unknown control id: '{control_id}'")
        default_val = self._space.schema[control_id].default
        self[control_id] = default_val

    def __iter__(self) -> Iterator[str]:
        return iter(self._space.schema)

    def __len__(self) -> int:
        return len(self._space.schema)

    def snap(
        self,
        mapping: Mapping[str, float | str] | None = None,
        /,
        **kwargs: float | str,
    ) -> ClampedControlsResult:
        """Snaps one or more controls immediately with zero anti-zipper slew."""
        merged: dict[str, float | str] = {}
        if mapping is not None:
            merged.update(mapping)
        merged.update(kwargs)
        for cid, val in merged.items():
            idx = self._space.index_of(cid)
            num_val = self._space.resolve_numeric_value(cid, val)
            self._raw_values[cid] = val
            self._set_snap(idx, num_val)
        self._cached_clamped = None
        return self.clamped

    def apply_preset(self, preset_name: str, *, immediate: bool = True) -> ClampedControlsResult:
        """Applies a named preset from the container's `presets` table."""
        if preset_name not in self._space.presets:
            raise DnamControlError(f"Unknown preset: '{preset_name}'")
        preset = self._space.presets[preset_name]
        if immediate:
            return self.snap(preset.controls)
        for cid, val in preset.controls.items():
            self[cid] = val
        return self.clamped

    def reset_to_defaults(self) -> ClampedControlsResult:
        """Snaps all controls back to their schema defaults."""
        defaults = {cid: ctrl.default for cid, ctrl in self._space.schema.items()}
        return self.snap(defaults)


class LobattoBasis:
    """Callable Node-Anchored Newton/Lobatto polynomial basis evaluator (`CL-13`)."""

    def __init__(self, params: LobattoParams) -> None:
        if not isinstance(params, LobattoParams):
            raise DnamValidationError("LobattoBasis requires a LobattoParams instance")
        self._params = params
        self._anchors_list = list(params.anchors)
        self._degree = params.degree

    @classmethod
    def from_schema(cls, ctrl: ControlSchemaDef) -> LobattoBasis:
        """Constructs a `LobattoBasis` from a `range` `ControlSchemaDef`."""
        if ctrl.kind != ControlKind.RANGE or ctrl.declared_range is None:
            raise DnamValidationError(f"Control '{ctrl.id}' is not a range control")
        deg = ctrl.lobatto_degree if ctrl.lobatto_degree is not None else 2
        if ctrl.anchor_nodes_norm and len(ctrl.anchor_nodes_norm) >= 3:
            anchors = (
                float(ctrl.anchor_nodes_norm[0]),
                float(ctrl.anchor_nodes_norm[1]),
                float(ctrl.anchor_nodes_norm[2]),
            )
        else:
            span = ctrl.declared_range[1] - ctrl.declared_range[0]
            scale_norm = 2.0 / span
            offset_norm = -1.0 - ctrl.declared_range[0] * scale_norm
            u0 = max(-1.0, min(1.0, float(ctrl.default) * scale_norm + offset_norm))
            anchors = (u0, -1.0, 1.0)
        return cls(LobattoParams(anchors=anchors, degree=deg))

    @property
    def params(self) -> LobattoParams:
        """Returns the validated `LobattoParams`."""
        return self._params

    def __call__(self, u: float | ArrayLike) -> NDArray[np.float64]:
        """Evaluates `[psi_1, psi_2, psi_3](u)` for scalar `u` (`(3,)`) or 1D `u` (`(N, 3)`)."""
        if isinstance(u, int | float | np.floating):
            vals = call_native(
                _native.native_eval_lobatto_basis,
                float(u),
                self._anchors_list,
                self._degree,
            )
            return np.asarray(vals, dtype=np.float64)
        arr = (
            u
            if isinstance(u, np.ndarray) and u.dtype == np.float64 and u.flags.c_contiguous
            else np.ascontiguousarray(u, dtype=np.float64)
        )
        if arr.ndim != 1:
            raise DnamValidationError("LobattoBasis input u must be a scalar or 1D array")
        out = np.empty((arr.shape[0], 3), dtype=np.float64)
        call_native(
            _native.native_eval_lobatto_basis_slice,
            arr,
            out,
            self._anchors_list,
            self._degree,
        )
        return out


class Rank4Fm:
    """Callable Rank-4 Factorization Machine interaction evaluator (`CL-14`)."""

    def __init__(self, v_factors: ArrayLike) -> None:
        v_arr = np.ascontiguousarray(v_factors, dtype=np.float64)
        if v_arr.ndim != 2 or v_arr.shape[1] != 4 or v_arr.shape[0] < 1:
            raise DnamValidationError(
                f"Rank4Fm v_factors must have shape (D, 4) with D >= 1; got {v_arr.shape}"
            )
        self._v_factors = v_arr
        self._v_flat = np.ascontiguousarray(v_arr.ravel(), dtype=np.float64)

    @property
    def num_features(self) -> int:
        """Returns the input feature dimension `D`."""
        return int(self._v_factors.shape[0])

    @property
    def v_factors(self) -> NDArray[np.float64]:
        """Returns a read-only view of the `(D, 4)` factorization matrix."""
        view = self._v_factors.view()
        view.flags.writeable = False
        return view

    def __call__(self, features: ArrayLike) -> NDArray[np.float64]:
        """Evaluates the 4-D FM pairwise interaction vector for `features` of shape `(D,)`."""
        feat = (
            features
            if isinstance(features, np.ndarray)
            and features.dtype == np.float64
            and features.flags.c_contiguous
            else np.ascontiguousarray(features, dtype=np.float64)
        )
        if feat.ndim != 1 or feat.shape[0] != self.num_features:
            raise DnamValidationError(
                f"Rank4Fm expected 1D features of length {self.num_features}, got {feat.shape}"
            )
        vals = call_native(_native.native_eval_rank4_fm, feat, self._v_flat)
        return np.asarray(vals, dtype=np.float64)


def chebyshev_lfo(cos_phi: float | ArrayLike) -> NDArray[np.float64]:
    """Evaluates the 4-harmonic Chebyshev LFO `[T_1, T_2, T_3, T_4](cos_phi)` (`CL-16`)."""
    if isinstance(cos_phi, int | float | np.floating):
        vals = call_native(_native.native_eval_chebyshev_lfo, float(cos_phi))
        return np.asarray(vals, dtype=np.float64)
    arr = (
        cos_phi
        if isinstance(cos_phi, np.ndarray)
        and cos_phi.dtype == np.float64
        and cos_phi.flags.c_contiguous
        else np.ascontiguousarray(cos_phi, dtype=np.float64)
    )
    if arr.ndim != 1:
        raise DnamValidationError("chebyshev_lfo input cos_phi must be a scalar or 1D array")
    out = np.empty((arr.shape[0], 4), dtype=np.float64)
    call_native(_native.native_eval_chebyshev_lfo_slice, arr, out)
    return out


def trust_region_softclip(z: float | ArrayLike, *, mh: float = 2.5) -> float | NDArray[np.float64]:
    """Evaluates the cubic trust-region soft-clipper `M_h * S_3(z / M_h)` (`CL-15`)."""
    if mh <= 0.0:
        raise DnamValidationError(f"trust_region_softclip radius mh ({mh}) must be > 0")
    if isinstance(z, int | float):
        return float(call_native(_native.native_eval_trust_region_softclip, float(z), float(mh)))
    arr = (
        z
        if isinstance(z, np.ndarray) and z.dtype == np.float64 and z.flags.c_contiguous
        else np.ascontiguousarray(z, dtype=np.float64)
    )
    if arr.ndim == 0:
        return float(call_native(_native.native_eval_trust_region_softclip, float(arr), float(mh)))
    flat = arr.ravel()
    out = np.empty_like(flat, dtype=np.float64)
    call_native(_native.native_eval_trust_region_softclip_slice, flat, out, float(mh))
    return out.reshape(arr.shape)
