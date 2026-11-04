"""Hypothesis strategies for `dnam` property-based tests."""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from numpy.typing import NDArray

from dnam.schema import Backend, TempoSubdivision, Tier

tier_strategy: st.SearchStrategy[Tier] = st.sampled_from(list(Tier))
backend_strategy: st.SearchStrategy[Backend] = st.sampled_from([Backend.SCALAR, Backend.AUTO])
sample_rate_strategy: st.SearchStrategy[int] = st.sampled_from(
    [44_100, 48_000, 88_200, 96_000, 192_000]
)
subdivision_strategy: st.SearchStrategy[TempoSubdivision] = st.sampled_from(list(TempoSubdivision))
bpm_strategy: st.SearchStrategy[float] = st.floats(
    min_value=20.0, max_value=400.0, allow_nan=False, allow_infinity=False
)


def stereo_audio_strategy(
    min_frames: int = 1,
    max_frames: int = 256,
) -> st.SearchStrategy[NDArray[np.float32]]:
    """Generates C-contiguous `(2, N)` `float32` stereo audio buffers in `[-1.0, 1.0]`."""
    return st.integers(min_value=min_frames, max_value=max_frames).flatmap(
        lambda n: hnp.arrays(
            dtype=np.float32,
            shape=(2, n),
            elements=st.floats(
                min_value=-1.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
                width=32,
            ),
        )
    )


def mono_audio_strategy(
    min_frames: int = 1,
    max_frames: int = 256,
) -> st.SearchStrategy[NDArray[np.float32]]:
    """Generates C-contiguous `(N,)` `float32` mono audio buffers in `[-1.0, 1.0]`."""
    return st.integers(min_value=min_frames, max_value=max_frames).flatmap(
        lambda n: hnp.arrays(
            dtype=np.float32,
            shape=(n,),
            elements=st.floats(
                min_value=-1.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
                width=32,
            ),
        )
    )


def symmetric_matrix_strategy(
    dim: int = 8,
) -> st.SearchStrategy[NDArray[np.float64]]:
    """Generates a symmetric `(dim, dim)` `float64` matrix."""
    return hnp.arrays(
        dtype=np.float64,
        shape=(dim, dim),
        elements=st.floats(
            min_value=-10.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    ).map(lambda a: np.ascontiguousarray(0.5 * (a + a.T), dtype=np.float64))


__all__ = [
    "backend_strategy",
    "bpm_strategy",
    "mono_audio_strategy",
    "sample_rate_strategy",
    "stereo_audio_strategy",
    "subdivision_strategy",
    "symmetric_matrix_strategy",
    "tier_strategy",
]
