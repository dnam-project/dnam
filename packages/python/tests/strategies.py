"""Reusable Hypothesis, `hypothesis.extra.numpy`, and `hypothesis-jsonschema` strategies."""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from numpy.typing import NDArray

st_sample_rates = st.integers(min_value=44_100, max_value=96_000)


def st_audio_1d_f32(
    min_len: int = 16, max_len: int = 256
) -> st.SearchStrategy[NDArray[np.float32]]:
    """Generates bounded 1D `float32` audio arrays without NaNs or infinities."""
    return hnp.arrays(
        dtype=np.float32,
        shape=st.integers(min_value=min_len, max_value=max_len),
        elements=st.floats(
            min_value=-2.0,
            max_value=2.0,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
    )


def st_audio_2d_stereo_f32(
    min_len: int = 16, max_len: int = 256
) -> st.SearchStrategy[NDArray[np.float32]]:
    """Generates bounded 2D `(2, N)` stereo `float32` audio arrays."""
    return st.integers(min_value=min_len, max_value=max_len).flatmap(
        lambda n: hnp.arrays(
            dtype=np.float32,
            shape=(2, n),
            elements=st.floats(
                min_value=-2.0,
                max_value=2.0,
                allow_nan=False,
                allow_infinity=False,
                width=32,
            ),
        )
    )


def st_audio_1d_f64(
    min_len: int = 16, max_len: int = 256
) -> st.SearchStrategy[NDArray[np.float64]]:
    """Generates bounded 1D `float64` audio arrays for Reference DSP property tests."""
    return hnp.arrays(
        dtype=np.float64,
        shape=st.integers(min_value=min_len, max_value=max_len),
        elements=st.floats(
            min_value=-2.0,
            max_value=2.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )


def st_symmetric_gram_matrix(dim: int = 16) -> st.SearchStrategy[NDArray[np.float64]]:
    """Generates a symmetric `(dim, dim)` `float64` matrix."""
    return hnp.arrays(
        dtype=np.float64,
        shape=(dim, dim),
        elements=st.floats(
            min_value=-100.0,
            max_value=100.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    ).map(lambda m: 0.5 * (m + m.T))
