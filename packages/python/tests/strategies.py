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


st_r_ac = st.floats(min_value=0.990, max_value=0.9999, allow_nan=False, allow_infinity=False)
st_r_dc = st.floats(min_value=0.985, max_value=0.9999, allow_nan=False, allow_infinity=False)
st_control_u = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)


def st_audio_batch_1d(
    batch_size: int = 2,
    channels: int = 4,
    seq_len: int = 64,
) -> st.SearchStrategy[NDArray[np.float32]]:
    """Generates bounded `(B, C, T)` `float32` audio batches without NaNs or infinities."""
    return hnp.arrays(
        dtype=np.float32,
        shape=(batch_size, channels, seq_len),
        elements=st.floats(
            min_value=-1.5,
            max_value=1.5,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
    )


def st_fdn_delays(num_lines: int = 8) -> st.SearchStrategy[list[float]]:
    """Generates bounded FDN base delay lengths in `[32.0, 256.0]` samples."""
    return st.lists(
        st.floats(min_value=32.0, max_value=256.0, allow_nan=False, allow_infinity=False),
        min_size=num_lines,
        max_size=num_lines,
    )
