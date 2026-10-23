"""Base Pydantic v2 models (`StrictModel`, `OpenMetadataModel`) and core DNAM enums."""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, ValidationError

from dnam_engine.exceptions import DnamError, DnamSampleRateError, DnamValidationError


def _translate_pydantic_error(exc: ValidationError) -> DnamError:
    msg = str(exc)
    if "sample_rate" in msg:
        return DnamSampleRateError(msg)
    return DnamValidationError(msg)


class StrictModel(BaseModel):
    """Immutable Pydantic v2 base model forbidding unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    def __init__(self, **data: Any) -> None:
        try:
            super().__init__(**data)
        except ValidationError as exc:
            raise _translate_pydantic_error(exc) from exc

    @classmethod
    def _construct_trusted(cls, **values: Any) -> Self:
        """Fast internal constructor for pre-validated Rust return values."""
        m = cls.__new__(cls)
        object.__setattr__(m, "__dict__", values)
        object.__setattr__(m, "__pydantic_fields_set__", set(values))
        object.__setattr__(m, "__pydantic_extra__", None)
        object.__setattr__(m, "__pydantic_private__", None)
        return m

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        extra: Literal["allow", "forbid", "ignore"] | None = None,
        from_attributes: bool | None = None,
        context: Any | None = None,
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
        except ValidationError as exc:
            raise _translate_pydantic_error(exc) from exc

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        extra: Literal["allow", "forbid", "ignore"] | None = None,
        context: Any | None = None,
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
        except ValidationError as exc:
            raise _translate_pydantic_error(exc) from exc


class OpenMetadataModel(BaseModel):
    """Immutable Pydantic v2 base model allowing open metadata dictionaries."""

    model_config = ConfigDict(frozen=True, extra="allow")


class Tier(IntEnum):
    """Slimmable execution tier (`Lite = 0`, `Full = 1`, `Ultra = 2`)."""

    LITE = 0
    FULL = 1
    ULTRA = 2

    @property
    def channels(self) -> int:
        """Returns the number of Core 1 channels (`4`, `8`, or `12`)."""
        return (int(self) + 1) * 4

    @property
    def fdn_lines(self) -> int:
        """Returns the number of Core 2 FDN lines (`4` for `Lite`, `8` for `Full`/`Ultra`)."""
        return 4 if self == Tier.LITE else 8


class BackendKind(StrEnum):
    """Hardware SIMD or scalar math backend identifier."""

    AUTO = "auto"
    SCALAR = "scalar"
    ARM_NEON = "arm_neon"
    ARM_SME = "arm_sme"
    X86_SIMD = "x86_simd"
    WASM_SIMD128 = "wasm_simd128"


class ControlKind(StrEnum):
    """Parametric control kind (`range` vs `switch`)."""

    RANGE = "range"
    SWITCH = "switch"


class VerificationStatus(StrEnum):
    """Cryptographic provenance verification status for a `.dnam` or `.dnam-ckpt` container."""

    UNSIGNED = "Unsigned"
    CREATOR_VERIFIED = "CreatorVerified"
    MARKETPLACE_VERIFIED = "MarketplaceVerified"

    @classmethod
    def from_rust_debug(cls, raw: str) -> VerificationStatus:
        """Parses the Rust `VerificationStatus` debug string into a `VerificationStatus` enum."""
        if raw.startswith("MarketplaceVerified"):
            return cls.MARKETPLACE_VERIFIED
        if raw.startswith("CreatorVerified"):
            return cls.CREATOR_VERIFIED
        return cls.UNSIGNED
