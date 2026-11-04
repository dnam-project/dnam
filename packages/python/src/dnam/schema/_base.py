"""Base Pydantic v2 `StrictModel` and core enumerations (`ref[sdk.python.schema.rust-parity]`)."""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict
from pydantic import ValidationError as PydanticValidationError

from dnam.errors import DnamError, SampleRateError, ValidationError


def _translate_pydantic_error(exc: PydanticValidationError) -> DnamError:
    msg = str(exc)
    if "sample_rate" in msg:
        return SampleRateError(msg)
    return ValidationError(msg)


_BASE_MODEL_INIT = BaseModel.__init__


class StrictModel(BaseModel):
    """Immutable, strictly validated Pydantic v2 model (`frozen=True, extra='forbid'`)."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        populate_by_name=True,
        validate_default=True,
    )

    _trusted_fields_set: ClassVar[set[str]] = set()

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        cls._trusted_fields_set = set(cls.model_fields)

    @classmethod
    def _construct_trusted(cls, **values: object) -> Self:
        """Fast hot-path constructor for pre-validated Rust keyword values."""
        m = cls.__new__(cls)
        object.__setattr__(m, "__dict__", values)
        object.__setattr__(m, "__pydantic_fields_set__", cls._trusted_fields_set or set(values))
        object.__setattr__(m, "__pydantic_extra__", None)
        object.__setattr__(m, "__pydantic_private__", None)
        return m

    @classmethod
    def _construct_trusted_dict(cls, values: dict[str, object]) -> Self:
        """Zero-copy hot-path constructor taking an existing `dict` as `__dict__`."""
        m = cls.__new__(cls)
        object.__setattr__(m, "__dict__", values)
        object.__setattr__(m, "__pydantic_fields_set__", cls._trusted_fields_set)
        object.__setattr__(m, "__pydantic_extra__", None)
        object.__setattr__(m, "__pydantic_private__", None)
        return m

    @classmethod
    def model_validate(
        cls,
        obj: object,
        *,
        strict: bool | None = None,
        extra: Literal["allow", "forbid", "ignore"] | None = None,
        from_attributes: bool | None = None,
        context: object | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        try:
            return super().model_validate(
                obj,
                strict=strict,
                extra=extra,
                from_attributes=from_attributes,
                context=context,
                by_alias=by_alias,
                by_name=by_name,
            )
        except PydanticValidationError as exc:
            raise _translate_pydantic_error(exc) from exc

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        extra: Literal["allow", "forbid", "ignore"] | None = None,
        context: object | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        try:
            return super().model_validate_json(
                json_data,
                strict=strict,
                extra=extra,
                context=context,
                by_alias=by_alias,
                by_name=by_name,
            )
        except PydanticValidationError as exc:
            raise _translate_pydantic_error(exc) from exc


def _strict_model_init(self: StrictModel, **data: object) -> None:
    try:
        _BASE_MODEL_INIT(self, **data)
    except PydanticValidationError as exc:
        raise _translate_pydantic_error(exc) from exc


# Attach after class creation so Pydantic keeps `__pydantic_custom_init__ = False`
# and constructs nested models directly inside Rust `pydantic_core` during JSON parsing.
StrictModel.__init__ = _strict_model_init


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
