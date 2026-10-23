"""Typed exception hierarchy for `dnam_engine`."""

from __future__ import annotations

from collections.abc import Callable


class DnamError(ValueError):
    """Base exception for all Dynamic Neural Audio Modeler (DNAM) errors."""


class DnamValidationError(DnamError):
    """Raised when a schema, parameter bound, alignment invariant, or array layout is invalid."""


class DnamContainerError(DnamError):
    """Raised when a `.dnam` or `.dnam-ckpt` binary preamble, magic, or CRC-32C checksum fails."""


class DnamControlError(DnamError, KeyError):
    """Raised when an unknown control ID, preset name, or switch option is requested."""


class DnamSampleRateError(DnamError):
    """Raised when a sample rate falls outside the supported `[44100, 96000]` Hz range."""


def translate_native_error(exc: ValueError) -> DnamError:
    """Translates a tagged `ValueError` from `dnam_engine._native` into a `DnamError` subclass."""
    msg = str(exc)
    if msg.startswith("[DNAM-CONTAINER] "):
        return DnamContainerError(msg.removeprefix("[DNAM-CONTAINER] "))
    if msg.startswith("[DNAM-SAMPLERATE] "):
        return DnamSampleRateError(msg.removeprefix("[DNAM-SAMPLERATE] "))
    if msg.startswith("[DNAM-VALIDATION] "):
        return DnamValidationError(msg.removeprefix("[DNAM-VALIDATION] "))
    return DnamError(msg)


def call_native[T](fn: Callable[..., T], /, *args: object, **kwargs: object) -> T:
    """Invokes a native `dnam_engine._native` function and maps `ValueError` to `DnamError`."""
    try:
        return fn(*args, **kwargs)
    except ValueError as exc:
        raise translate_native_error(exc) from exc
