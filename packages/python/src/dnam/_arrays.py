"""Shared zero-copy NumPy array coercion and shape validation helpers
(`ref[sdk.python.api.flexible-in-precise-out]`).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dnam.errors import ValidationError

_F32_DTYPE = np.dtype(np.float32)
_F64_DTYPE = np.dtype(np.float64)


def coerce_f32_1d(arr: ArrayLike, *, name: str = "audio") -> NDArray[np.float32]:
    """Coerces a 1D `ArrayLike` into a C-contiguous 1D `np.float32` array."""
    if (
        type(arr) is np.ndarray
        and arr.dtype == _F32_DTYPE
        and arr.ndim == 1
        and arr.flags.c_contiguous
    ):
        return arr
    out = np.asarray(arr, dtype=np.float32, order="C")
    if out.ndim != 1:
        raise ValidationError(f"{name} must be a 1D array, got shape {out.shape}")
    return out


def coerce_f64_1d(arr: ArrayLike, *, name: str = "input") -> NDArray[np.float64]:
    """Coerces a 1D `ArrayLike` into a C-contiguous 1D `np.float64` array."""
    if (
        type(arr) is np.ndarray
        and arr.dtype == _F64_DTYPE
        and arr.ndim == 1
        and arr.flags.c_contiguous
    ):
        return arr
    out = np.asarray(arr, dtype=np.float64, order="C")
    if out.ndim != 1:
        raise ValidationError(f"{name} must be a 1D array, got shape {out.shape}")
    return out


def coerce_f64_2d(
    arr: ArrayLike,
    *,
    expected_cols: int | None = None,
    name: str = "matrix",
) -> NDArray[np.float64]:
    """Coerces a 2D `ArrayLike` into a C-contiguous 2D `np.float64` array."""
    if (
        type(arr) is np.ndarray
        and arr.dtype == _F64_DTYPE
        and arr.ndim == 2
        and arr.flags.c_contiguous
    ):
        if expected_cols is not None and arr.shape[1] != expected_cols:
            raise ValidationError(f"{name} must have shape (N, {expected_cols}), got {arr.shape}")
        return arr
    out = np.asarray(arr, dtype=np.float64, order="C")
    if out.ndim != 2:
        raise ValidationError(f"{name} must be a 2D array, got shape {out.shape}")
    if expected_cols is not None and out.shape[1] != expected_cols:
        raise ValidationError(f"{name} must have shape (N, {expected_cols}), got {out.shape}")
    return out


def coerce_audio_f32(
    audio: ArrayLike,
) -> tuple[NDArray[np.float32], int, bool]:
    """Normalizes `audio` into a C-contiguous `float32` array.

    Returns `(contiguous_array, num_input_channels, was_1d)`.
    """
    if type(audio) is np.ndarray and audio.dtype == _F32_DTYPE and audio.flags.c_contiguous:
        ndim = audio.ndim
        if ndim == 1:
            return audio, 1, True
        if ndim == 2:
            ch = audio.shape[0]
            if ch == 2 or ch == 1:
                return audio, ch, False
    arr = np.asarray(audio, dtype=np.float32, order="C")
    if arr.ndim == 1:
        return arr, 1, True
    if arr.ndim == 2 and arr.shape[0] in (1, 2):
        return arr, int(arr.shape[0]), False
    raise ValidationError(f"audio must have shape (N,), (1, N), or (2, N), got {arr.shape}")


def validate_f32_out(
    out: NDArray[np.float32],
    expected_shape: tuple[int, ...],
    *,
    name: str = "out",
) -> None:
    """Verifies that `out` is a writable C-contiguous `float32` array of `expected_shape`."""
    if (
        type(out) is np.ndarray
        and out.dtype == _F32_DTYPE
        and out.shape == expected_shape
        and out.flags.c_contiguous
        and out.flags.writeable
    ):
        return
    if not isinstance(out, np.ndarray):
        raise ValidationError(f"{name} must be a numpy.ndarray")
    if out.dtype != _F32_DTYPE:
        raise ValidationError(f"{name} must have dtype float32, got {out.dtype}")
    if not out.flags.c_contiguous:
        raise ValidationError(f"{name} must be C-contiguous")
    if not out.flags.writeable:
        raise ValidationError(f"{name} must be writeable")
    if out.shape != expected_shape:
        raise ValidationError(f"{name} shape mismatch: expected {expected_shape}, got {out.shape}")
