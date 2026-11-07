"""Declarative frozen slotted `StrictModel` and core enumerations
(`ref[sdk.python.schema.rust-parity]`).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import MISSING, Field, dataclass, field, fields
from enum import Enum, IntEnum, StrEnum
from types import UnionType
from typing import (
    Any,
    ClassVar,
    Literal,
    Self,
    TypeAliasType,
    Union,
    cast,
    dataclass_transform,
    get_args,
    get_origin,
    get_type_hints,
)

from dnam.errors import SampleRateError, ValidationError

_MISSING = object()
_OBJ_NEW = object.__new__
_OBJ_SETATTR = object.__setattr__


def strict_field(
    *,
    default: object = _MISSING,
    default_factory: Callable[[], object] | None = None,
    alias: str | None = None,
    ge: float | int | None = None,
    le: float | int | None = None,
    gt: float | int | None = None,
    lt: float | int | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
) -> Any:
    """Declares a `StrictModel` dataclass field with optional alias and numeric/length bounds."""
    meta: dict[str, object] = {}
    if alias is not None:
        meta["alias"] = alias
    if ge is not None:
        meta["ge"] = ge
    if le is not None:
        meta["le"] = le
    if gt is not None:
        meta["gt"] = gt
    if lt is not None:
        meta["lt"] = lt
    if min_length is not None:
        meta["min_length"] = min_length
    if max_length is not None:
        meta["max_length"] = max_length
    if default is not _MISSING:
        return field(default=default, metadata=meta)
    if default_factory is not None:
        return field(default_factory=default_factory, metadata=meta)
    return field(metadata=meta)


def _check_json_value(val: object, field_name: str) -> object:
    if val is None or isinstance(val, (bool, int, str)):
        return val
    if isinstance(val, float):
        if not math.isfinite(val):
            raise ValidationError(f"{field_name}: float must be finite, got {val!r}")
        return val
    if isinstance(val, (list, tuple)):
        return [_check_json_value(item, field_name) for item in val]
    if isinstance(val, Mapping):
        out_dict: dict[str, object] = {}
        for k, v in val.items():
            if not isinstance(k, str):
                raise ValidationError(f"{field_name}: JSON object keys must be str, got {k!r}")
            out_dict[k] = _check_json_value(v, field_name)
        return out_dict
    raise ValidationError(f"{field_name}: invalid JSON value {val!r}")


def _coerce_value(val: object, tp: object, field_name: str) -> object:
    if isinstance(tp, TypeAliasType):
        return _check_json_value(val, field_name)

    origin = get_origin(tp)
    if origin is Union or origin is UnionType:
        args = get_args(tp)
        if val is None and type(None) in args:
            return None
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _coerce_value(val, non_none[0], field_name)
        if isinstance(val, bool) and bool not in non_none:
            raise ValidationError(f"{field_name}: unexpected bool {val!r}")
        if type(val) in non_none:
            return _coerce_value(val, type(val), field_name)
        for branch in non_none:
            try:
                return _coerce_value(val, branch, field_name)
            except ValidationError:
                continue
        raise ValidationError(f"{field_name}: value {val!r} does not match {tp}")

    if origin is Literal:
        allowed = get_args(tp)
        for cand in allowed:
            if val == cand and type(val) is type(cand):
                return val
        raise ValidationError(f"{field_name}: expected one of {allowed!r}, got {val!r}")

    if origin is list:
        if not isinstance(val, (list, tuple)):
            raise ValidationError(f"{field_name}: expected list, got {type(val).__name__}")
        (elem_tp,) = get_args(tp)
        return [_coerce_value(item, elem_tp, field_name) for item in val]

    if origin is tuple:
        if not isinstance(val, (list, tuple)):
            raise ValidationError(f"{field_name}: expected tuple, got {type(val).__name__}")
        args = get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce_value(item, args[0], field_name) for item in val)
        if len(val) != len(args):
            raise ValidationError(f"{field_name}: expected {len(args)} elements, got {len(val)}")
        return tuple(
            _coerce_value(item, arg_tp, field_name) for item, arg_tp in zip(val, args, strict=True)
        )

    if origin is dict:
        if not isinstance(val, Mapping):
            raise ValidationError(f"{field_name}: expected dict, got {type(val).__name__}")
        k_tp, v_tp = get_args(tp)
        return {
            _coerce_value(k, k_tp, field_name): _coerce_value(v, v_tp, field_name)
            for k, v in val.items()
        }

    if isinstance(tp, type):
        if issubclass(tp, StrictModel):
            if isinstance(val, tp):
                return val
            if isinstance(val, Mapping):
                return tp.from_dict(val)
            raise ValidationError(
                f"{field_name}: expected {tp.__name__} or dict, got {type(val).__name__}"
            )
        if issubclass(tp, Enum):
            if isinstance(val, tp):
                return val
            if isinstance(val, bool):
                raise ValidationError(f"{field_name}: invalid {tp.__name__} value {val!r}")
            try:
                return tp(val)
            except ValueError, TypeError:
                parse_fn = getattr(tp, "parse", None)
                if callable(parse_fn):
                    return parse_fn(val)
                raise ValidationError(
                    f"{field_name}: invalid {tp.__name__} value {val!r}"
                ) from None
        if tp is bool:
            if isinstance(val, bool):
                return val
            raise ValidationError(f"{field_name}: expected bool, got {type(val).__name__}")
        if tp is int:
            if isinstance(val, bool):
                raise ValidationError(f"{field_name}: expected int, got bool")
            if isinstance(val, int):
                return val.__int__()
            if isinstance(val, float) and val.is_integer():
                return int(val)
            raise ValidationError(f"{field_name}: expected int, got {val!r}")
        if tp is float:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValidationError(f"{field_name}: expected float, got {val!r}")
            fval = float(val)
            if not math.isfinite(fval):
                raise ValidationError(f"{field_name}: float must be finite, got {val!r}")
            return fval
        if tp is str:
            if isinstance(val, str):
                return val.__str__()
            raise ValidationError(f"{field_name}: expected str, got {type(val).__name__}")

    return val


def _check_field_bounds(val: object, meta: Mapping[str, object], field_name: str) -> None:
    if val is None or not meta:
        return
    err_cls = SampleRateError if "sample_rate" in field_name else ValidationError
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        ge = meta.get("ge")
        if isinstance(ge, (int, float)) and val < ge:
            raise err_cls(f"{field_name} must be >= {ge}, got {val}")
        le = meta.get("le")
        if isinstance(le, (int, float)) and val > le:
            raise err_cls(f"{field_name} must be <= {le}, got {val}")
        gt = meta.get("gt")
        if isinstance(gt, (int, float)) and val <= gt:
            raise err_cls(f"{field_name} must be > {gt}, got {val}")
        lt = meta.get("lt")
        if isinstance(lt, (int, float)) and val >= lt:
            raise err_cls(f"{field_name} must be < {lt}, got {val}")
    if isinstance(val, (list, tuple, str, bytes, dict)):
        min_len = meta.get("min_length")
        if isinstance(min_len, int) and len(val) < min_len:
            raise ValidationError(f"{field_name} length must be >= {min_len}, got {len(val)}")
        max_len = meta.get("max_length")
        if isinstance(max_len, int) and len(val) > max_len:
            raise ValidationError(f"{field_name} length must be <= {max_len}, got {len(val)}")


def _serialize_value(val: object, *, by_alias: bool) -> object:
    if isinstance(val, StrictModel):
        return val.to_dict(by_alias=by_alias)
    if isinstance(val, Enum):
        return val.value
    if isinstance(val, (list, tuple)):
        return [_serialize_value(item, by_alias=by_alias) for item in val]
    if isinstance(val, dict):
        return {str(k): _serialize_value(v, by_alias=by_alias) for k, v in val.items()}
    return val


class _StrictModelMeta(type):
    """Metaclass that automatically applies `@dataclass(frozen=True, slots=True, kw_only=True)`."""

    def __new__(
        mcls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, object],
        **kwargs: object,
    ) -> type:
        if not bases or "__slots__" in namespace:
            return super().__new__(mcls, name, bases, namespace, **kwargs)
        raw_cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        slotted_cls = dataclass(frozen=True, slots=True, kw_only=True)(cast(Any, raw_cls))
        init_hook = getattr(slotted_cls, "_init_strict_model_subclass", None)
        if callable(init_hook):
            init_hook()
        return slotted_cls


@dataclass_transform(
    frozen_default=True,
    kw_only_default=True,
    field_specifiers=(strict_field, field),
)
class StrictModel(metaclass=_StrictModelMeta):
    """Immutable, slotted dataclass model (`frozen=True, slots=True`) with strict validation."""

    __slots__ = ()

    _field_names: ClassVar[tuple[str, ...]] = ()
    _field_names_set: ClassVar[frozenset[str]] = frozenset()
    _required_fields: ClassVar[frozenset[str]] = frozenset()
    _serializable_fields: ClassVar[tuple[tuple[str, str], ...]] = ()
    _alias_to_field: ClassVar[dict[str, str]] = {}
    _valid_input_keys: ClassVar[frozenset[str]] = frozenset()
    _non_init_defaults: ClassVar[tuple[tuple[str, object], ...]] = ()
    _field_specs: ClassVar[tuple[tuple[str, object, Mapping[str, object]], ...] | None] = None

    @classmethod
    def _init_strict_model_subclass(cls) -> None:
        dc_fields: tuple[Field[object], ...] = fields(cast(Any, cls))
        init_names: list[str] = []
        req_names: list[str] = []
        ser_fields: list[tuple[str, str]] = []
        alias_map: dict[str, str] = {}
        valid_keys: set[str] = set()
        non_init_defs: list[tuple[str, object]] = []

        for f in dc_fields:
            if not f.init:
                if f.default is not MISSING:
                    non_init_defs.append((f.name, f.default))
                continue
            init_names.append(f.name)
            valid_keys.add(f.name)
            alias = f.metadata.get("alias")
            alias_str = alias if isinstance(alias, str) else f.name
            if isinstance(alias, str):
                alias_map[alias] = f.name
                valid_keys.add(alias)
            ser_fields.append((f.name, alias_str))
            if f.default is MISSING and f.default_factory is MISSING:
                req_names.append(f.name)

        cls._field_names = tuple(init_names)
        cls._field_names_set = frozenset(init_names)
        cls._required_fields = frozenset(req_names)
        cls._serializable_fields = tuple(ser_fields)
        cls._alias_to_field = alias_map
        cls._valid_input_keys = frozenset(valid_keys)
        cls._non_init_defaults = tuple(non_init_defs)
        cls._field_specs = None

        if alias_map:
            orig_init: Any = cls.__init__

            def _aliased_init(self: StrictModel, **kw: object) -> None:
                for al, fname in alias_map.items():
                    if al in kw:
                        if fname in kw:
                            raise ValidationError(
                                f"Duplicate field '{fname}' (via alias '{al}') for {cls.__name__}"
                            )
                        kw[fname] = kw.pop(al)
                orig_init(self, **kw)

            type.__setattr__(cls, "__init__", _aliased_init)

    @classmethod
    def _ensure_field_specs(
        cls,
    ) -> tuple[tuple[str, object, Mapping[str, object]], ...]:
        specs = cls._field_specs
        if specs is not None:
            return specs
        hints = get_type_hints(cls)
        dc_fields: tuple[Field[object], ...] = fields(cast(Any, cls))
        built: list[tuple[str, object, Mapping[str, object]]] = []
        for f in dc_fields:
            if not f.init:
                continue
            tp = hints.get(f.name, object)
            built.append((f.name, tp, f.metadata))
        resolved = tuple(built)
        cls._field_specs = resolved
        return resolved

    def __post_init__(self) -> None:
        specs = self._ensure_field_specs()
        for fname, tp, meta in specs:
            raw_val = getattr(self, fname)
            coerced = _coerce_value(raw_val, tp, fname)
            _check_field_bounds(coerced, meta, fname)
            if coerced is not raw_val:
                _OBJ_SETATTR(self, fname, coerced)

    @classmethod
    def _construct_trusted(cls, **values: object) -> Self:
        """Fast hot-path constructor for pre-validated Rust keyword values."""
        m = _OBJ_NEW(cls)
        for k, v in cls._non_init_defaults:
            _OBJ_SETATTR(m, k, v)
        for k, v in values.items():
            _OBJ_SETATTR(m, k, v)
        return m

    @classmethod
    def _construct_trusted_dict(cls, values: Mapping[str, object]) -> Self:
        """Fast hot-path constructor taking an existing `Mapping` of pre-validated field values."""
        m = _OBJ_NEW(cls)
        for k, v in cls._non_init_defaults:
            _OBJ_SETATTR(m, k, v)
        for k, v in values.items():
            _OBJ_SETATTR(m, k, v)
        return m

    def replace(self, **changes: object) -> Self:
        """Returns a new validated instance with `changes` applied."""
        if not changes:
            return self
        alias_map = self._alias_to_field
        norm_changes: dict[str, object] = {}
        for k, v in changes.items():
            fname = alias_map.get(k, k)
            if fname not in self._field_names_set:
                raise ValidationError(f"Unknown field '{k}' for {type(self).__name__}.replace")
            if fname in norm_changes:
                raise ValidationError(
                    f"Duplicate field '{fname}' (via '{k}') in {type(self).__name__}.replace"
                )
            norm_changes[fname] = v
        kwargs: dict[str, object] = {
            name: norm_changes[name] if name in norm_changes else getattr(self, name)
            for name in self._field_names
        }
        return type(self)(**kwargs)

    def to_dict(self, *, by_alias: bool = False) -> dict[str, object]:
        """Serializes this model to a JSON-compatible dictionary."""
        out: dict[str, object] = {}
        for field_name, alias_name in self._serializable_fields:
            key = alias_name if by_alias else field_name
            out[key] = _serialize_value(getattr(self, field_name), by_alias=by_alias)
        return out

    def to_json(self, *, by_alias: bool = True) -> str:
        """Serializes this model to a compact UTF-8 JSON string."""
        return json.dumps(
            self.to_dict(by_alias=by_alias), separators=(",", ":"), ensure_ascii=False
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        """Validates and constructs an instance from a mapping, rejecting unknown keys."""
        if not isinstance(data, Mapping):
            raise ValidationError(f"Expected mapping for {cls.__name__}, got {type(data).__name__}")
        alias_map = cls._alias_to_field
        valid_keys = cls._valid_input_keys
        kwargs: dict[str, object] = {}
        for k, v in data.items():
            if not isinstance(k, str) or k not in valid_keys:
                raise ValidationError(f"Unknown field '{k}' for {cls.__name__}")
            fname = alias_map.get(k, k)
            if fname in kwargs:
                raise ValidationError(f"Duplicate field '{fname}' (via '{k}') for {cls.__name__}")
            kwargs[fname] = v
        missing = cls._required_fields - set(kwargs)
        if missing:
            raise ValidationError(
                f"Missing required field(s) for {cls.__name__}: {sorted(missing)}"
            )
        return cls(**kwargs)

    @classmethod
    def from_json(cls, json_data: str | bytes | bytearray | memoryview) -> Self:
        """Parses and validates a UTF-8 JSON string or byte buffer."""
        try:
            raw_str = json_data if isinstance(json_data, str) else bytes(json_data).decode("utf-8")
            parsed = json.loads(raw_str)
        except Exception as exc:
            raise ValidationError(f"Invalid {cls.__name__} JSON: {exc}") from None
        if not isinstance(parsed, Mapping):
            raise ValidationError(
                f"Expected JSON object for {cls.__name__}, got {type(parsed).__name__}"
            )
        return cls.from_dict(parsed)


class Tier(IntEnum):
    """Slimmable model complexity tier (`0 = Lite`, `1 = Full`, `2 = Pro`, `3 = Ultra`)."""

    LITE = 0
    FULL = 1
    PRO = 2
    ULTRA = 3

    @classmethod
    def parse(cls, value: Tier | int | str) -> Tier:
        """Coerces a `Tier`, integer `0..3`, or case-insensitive tier name."""
        if type(value) is cls:
            return value
        if isinstance(value, bool):
            raise ValidationError(f"Invalid tier: {value!r}")
        if isinstance(value, int):
            if 0 <= value <= 3:
                return _TIER_TUPLE[value]
            raise ValidationError(f"Tier index must be 0..3, got {value}")
        if isinstance(value, str):
            found = _TIER_BY_STR.get(value) or _TIER_BY_STR.get(value.strip().lower())
            if found is not None:
                return found
        raise ValidationError(f"Unknown tier: {value!r}")

    @property
    def label(self) -> str:
        """Canonical lowercase tier name (`'lite'`, `'full'`, `'pro'`, `'ultra'`)."""
        return _TIER_LABELS[int(self)]

    @property
    def channels(self) -> int:
        """Active channels `C` (`4`, `8`, `12`, `16`)."""
        return (4, 8, 12, 16)[int(self)]

    @property
    def feature_bus_dim(self) -> int:
        """Active feature-bus dimension (`64`, `128`, `192`, `256`)."""
        return (64, 128, 192, 256)[int(self)]

    @property
    def arena_end_byte_offset(self) -> int:
        """Cumulative arena prefix end byte offset (`10752`, `22336`, `36224`, `64000`)."""
        return (10_752, 22_336, 36_224, 64_000)[int(self)]


_TIER_TUPLE: tuple[Tier, Tier, Tier, Tier] = (
    Tier.LITE,
    Tier.FULL,
    Tier.PRO,
    Tier.ULTRA,
)
_TIER_LABELS: tuple[str, str, str, str] = ("lite", "full", "pro", "ultra")
_TIER_BY_STR: dict[str, Tier] = {
    "lite": Tier.LITE,
    "full": Tier.FULL,
    "pro": Tier.PRO,
    "ultra": Tier.ULTRA,
    "LITE": Tier.LITE,
    "FULL": Tier.FULL,
    "PRO": Tier.PRO,
    "ULTRA": Tier.ULTRA,
}


class Backend(StrEnum):
    """Hardware SIMD / scalar math backend selector."""

    AUTO = "auto"
    SCALAR = "scalar"
    NEON = "neon"
    ARM_SME = "arm_sme"
    X86_SIMD = "x86_simd"
    WASM_SIMD128 = "wasm_simd128"

    @classmethod
    def parse(cls, value: Backend | str) -> Backend:
        """Coerces a `Backend` or case-insensitive backend string."""
        if type(value) is cls:
            return value
        if isinstance(value, str):
            found = _BACKEND_BY_STR.get(value) or _BACKEND_BY_STR.get(value.strip().lower())
            if found is not None:
                return found
        raise ValidationError(f"Unknown backend: {value!r}")


_BACKEND_BY_STR: dict[str, Backend] = {
    "auto": Backend.AUTO,
    "scalar": Backend.SCALAR,
    "neon": Backend.NEON,
    "arm_neon": Backend.NEON,
    "arm64_neon": Backend.NEON,
    "arm_sme": Backend.ARM_SME,
    "armsme": Backend.ARM_SME,
    "sme": Backend.ARM_SME,
    "x86_simd": Backend.X86_SIMD,
    "x86_avx2_fma": Backend.X86_SIMD,
    "avx2": Backend.X86_SIMD,
    "x86_sse42": Backend.X86_SIMD,
    "wasm_simd128": Backend.WASM_SIMD128,
    "wasm": Backend.WASM_SIMD128,
}


class VerificationStatus(StrEnum):
    """Cryptographic provenance verification status of a loaded `.dnam` container."""

    UNSIGNED = "unsigned"
    CREATOR_VERIFIED = "creator_verified"
    MARKETPLACE_VERIFIED = "marketplace_verified"

    @classmethod
    def from_rust_debug(cls, debug_str: str) -> VerificationStatus:
        """Parses the Rust debug/tag representation of `VerificationStatus`."""
        found = _VERIFICATION_FAST.get(debug_str)
        if found is not None:
            return found
        lower = debug_str.strip().lower()
        if lower.startswith("creator"):
            return cls.CREATOR_VERIFIED
        if lower.startswith("marketplace"):
            return cls.MARKETPLACE_VERIFIED
        return cls.UNSIGNED


_VERIFICATION_FAST: dict[str, VerificationStatus] = {
    "Unsigned": VerificationStatus.UNSIGNED,
    "unsigned": VerificationStatus.UNSIGNED,
    "creator_verified": VerificationStatus.CREATOR_VERIFIED,
    "marketplace_verified": VerificationStatus.MARKETPLACE_VERIFIED,
}
