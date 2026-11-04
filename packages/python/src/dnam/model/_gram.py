"""Column-Major Upper-Triangular (`LAPACK uplo='U'`) Gram Matrix
(`ref[sdk.python.model.containers-and-checkpoints]`).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dnam._arrays import coerce_f64_1d, coerce_f64_2d
from dnam._native import (
    native_crc32c,
    native_pack_uplo_u_f64_into,
    native_unpack_uplo_u_f64_into,
    native_uplo_u_packed_len,
)
from dnam.errors import ValidationError


def crc32c(data: bytes | bytearray | memoryview) -> int:
    """Computes the Castagnoli `CRC32C` checksum over `data`."""
    raw = data if type(data) is bytes else bytes(data)
    return native_crc32c(raw)


class GramMatrix:
    """Symmetric `N x N` Gram matrix stored in canonical `LAPACK uplo='U'` format."""

    __slots__ = ("_dim", "_packed")

    def __init__(
        self,
        packed: NDArray[np.float64],
        dim: int,
        *,
        _trusted: bool = False,
    ) -> None:
        if not _trusted:
            if dim <= 0:
                raise ValidationError(f"GramMatrix dim must be positive, got {dim}")
            expected_len = (dim * (dim + 1)) // 2
            arr = coerce_f64_1d(packed, name="packed")
            if arr.shape[0] != expected_len:
                raise ValidationError(
                    f"Packed uplo='U' buffer for dim={dim} requires {expected_len} elements, "
                    f"got {arr.shape[0]}"
                )
            self._packed = arr
        else:
            self._packed = packed
        self._dim = dim

    @staticmethod
    def packed_length(dim: int) -> int:
        """Returns the packed `uplo='U'` element count `dim * (dim + 1) // 2`."""
        if dim < 0:
            raise ValidationError(f"dim must be non-negative, got {dim}")
        return native_uplo_u_packed_len(dim)

    @classmethod
    def from_dense(cls, matrix: ArrayLike) -> GramMatrix:
        """Packs a dense `N x N` symmetric `float64` matrix into `uplo='U'` format."""
        mat = coerce_f64_2d(matrix, name="matrix")
        n, m = mat.shape
        if n != m or n == 0:
            raise ValidationError(
                f"GramMatrix.from_dense requires a non-empty square (N, N) matrix, got {mat.shape}"
            )
        packed_len = (n * (n + 1)) // 2
        packed = np.empty(packed_len, dtype=np.float64)
        native_pack_uplo_u_f64_into(mat.reshape(n * n), n, packed)
        return cls(packed, n, _trusted=True)

    @classmethod
    def from_packed(cls, packed: ArrayLike, dim: int) -> GramMatrix:
        """Constructs a `GramMatrix` directly from a 1D packed `uplo='U'` array and `dim`."""
        return cls(coerce_f64_1d(packed, name="packed"), dim)

    @property
    def dim(self) -> int:
        """Matrix dimension `N` (`shape == (N, N)`)."""
        return self._dim

    @property
    def packed(self) -> NDArray[np.float64]:
        """1D C-contiguous `float64` array of length `N(N+1)/2` in `uplo='U'` order."""
        return self._packed

    def to_dense(self, *, submatrix_dim: int | None = None) -> NDArray[np.float64]:
        """Unpacks into a symmetric `(k, k)` dense `float64` matrix (`k = submatrix_dim or dim`)."""
        k = self._dim if submatrix_dim is None else submatrix_dim
        if k <= 0 or k > self._dim:
            raise ValidationError(f"submatrix_dim must be in 1..={self._dim}, got {k}")
        out = np.empty((k, k), dtype=np.float64)
        native_unpack_uplo_u_f64_into(self._packed, k, out)
        return out

    def to_bytes(self) -> bytes:
        """
        Returns the little-endian `float64` byte representation of the packed `uplo='U'` buffer.
        """
        return self._packed.tobytes()

    def __repr__(self) -> str:
        return f"<GramMatrix dim={self._dim}x{self._dim} packed_len={self._packed.size}>"
