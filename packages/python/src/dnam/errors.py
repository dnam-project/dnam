"""Typed exception hierarchy for the `dnam` Python SDK
(`ref[sdk.python.errors.typed-hierarchy]`).
"""

from __future__ import annotations


class DnamError(ValueError):
    """Base exception for all DNAM SDK errors."""


class ContainerError(DnamError):
    """Raised when a `.dnam`, `.dnam-ckpt`, or `.dnam-seal` binary container is malformed
    or fails CRC32C.
    """


class ValidationError(DnamError):
    """Raised when a schema, alignment, buffer shape, or parameter invariant fails validation."""


class SampleRateError(DnamError):
    """Raised when a requested host sample rate is outside the supported `[44100, 192000]` Hz
    range.
    """


class ControlError(DnamError, KeyError):
    """Raised when an unknown control or preset ID, invalid option, or unsupported freeze is
    requested.
    """

    def __str__(self) -> str:
        return ValueError.__str__(self)


class ProvenanceError(DnamError):
    """Raised when cryptographic verification fails, a device key is missing, or a preview
    container is rejected.
    """


__all__ = [
    "ContainerError",
    "ControlError",
    "DnamError",
    "ProvenanceError",
    "SampleRateError",
    "ValidationError",
]
