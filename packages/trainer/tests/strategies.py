"""Hypothesis strategies for property-based testing across `dnam_trainer`."""

from __future__ import annotations

from hypothesis import strategies as st

valid_sample_rates = st.sampled_from([44_100, 48_000, 88_200, 96_000])

valid_pole_radii = st.floats(
    min_value=0.25,
    max_value=0.9995,
    allow_nan=False,
    allow_infinity=False,
)

valid_pole_angles = st.floats(
    min_value=1.0e-3,
    max_value=3.140,
    allow_nan=False,
    allow_infinity=False,
)

valid_fractional_delays = st.floats(
    min_value=4.0,
    max_value=120.0,
    allow_nan=False,
    allow_infinity=False,
)

valid_normalized_coords = st.floats(
    min_value=-1.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)
