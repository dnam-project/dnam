"""Exact boundary examples for the selected LTAS frequency partition."""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import pairwise

import pytest


def _grid_boundaries(precision: int, start: int, end: int) -> tuple[float, ...]:
    with localcontext() as context:
        context.prec = precision
        return tuple(
            float(Decimal(1000) * Decimal(10) ** (Decimal(2 * index + 1) / 20))
            for index in range(start, end)
        )


def _interior_boundaries(precision: int) -> tuple[float, ...]:
    return _grid_boundaries(precision, 3, 13)


def _band_index(frequency: float, boundaries: tuple[float, ...]) -> int:
    if not math.isfinite(frequency) or not 2000 <= frequency <= 18_000:
        raise ValueError("frequency outside the required HF interval")
    return 3 + sum(frequency >= boundary for boundary in boundaries)


def _full_band(frequency: float, boundaries: tuple[float, ...]) -> tuple[int, str]:
    if not math.isfinite(frequency) or not 20 <= frequency <= 20_000:
        raise ValueError("frequency outside the required audible interval")
    index = -17 + sum(frequency >= boundary for boundary in boundaries)
    region = "low" if frequency < 2000 else "hf" if frequency <= 18_000 else "upper"
    return index, region


def _require_full_band_nyquist(sample_rate: float) -> None:
    if not math.isfinite(sample_rate) or sample_rate < 40_000:
        raise ValueError("insufficient or invalid Nyquist frequency")


def test_interior_boundaries_are_correctly_rounded_binary64() -> None:
    boundaries = _interior_boundaries(100)
    assert boundaries == _interior_boundaries(200)
    for index, boundary in enumerate(boundaries, start=3):
        lower = Fraction.from_float(math.nextafter(boundary, -math.inf))
        value = Fraction.from_float(boundary)
        upper = Fraction.from_float(math.nextafter(boundary, math.inf))
        lower_midpoint = (lower + value) / 2
        upper_midpoint = (value + upper) / 2
        exact_twentieth_power = 10 ** (60 + 2 * index + 1)
        assert lower_midpoint**20 < exact_twentieth_power < upper_midpoint**20


def test_partition_covers_required_range_without_boundary_gaps() -> None:
    boundaries = _interior_boundaries(100)
    assert len(boundaries) == 10
    assert 2000 < boundaries[0] < boundaries[-1] < 18_000
    assert tuple(sorted(boundaries)) == boundaries
    assert _band_index(2000, boundaries) == 3
    assert _band_index(18_000, boundaries) == 13
    for index, boundary in enumerate(boundaries, start=3):
        assert _band_index(math.nextafter(boundary, -math.inf), boundaries) == index
        assert _band_index(boundary, boundaries) == index + 1
        assert _band_index(math.nextafter(boundary, math.inf), boundaries) == index + 1


def test_full_grid_boundaries_are_correctly_rounded_binary64() -> None:
    boundaries = _grid_boundaries(100, -17, 13)
    assert len(boundaries) == 30
    assert boundaries == _grid_boundaries(200, -17, 13)
    assert boundaries[20:] == _interior_boundaries(100)
    for index, boundary in enumerate(boundaries, start=-17):
        lower = Fraction.from_float(math.nextafter(boundary, -math.inf))
        value = Fraction.from_float(boundary)
        upper = Fraction.from_float(math.nextafter(boundary, math.inf))
        exact_twentieth_power = 10 ** (60 + 2 * index + 1)
        assert ((lower + value) / 2) ** 20 < exact_twentieth_power < ((value + upper) / 2) ** 20


def test_full_partition_has_33_distinct_intervals_and_preserves_hf_endpoints() -> None:
    boundaries = _grid_boundaries(100, -17, 13)
    cuts = tuple(sorted((*boundaries, 2000.0, 18_000.0)))
    assert len(cuts) == len(set(cuts)) == 32
    edges = (20.0, *cuts, 20_000.0)
    bands = tuple(_full_band((left + right) / 2, boundaries) for left, right in pairwise(edges))
    assert len(set(bands)) == 33
    assert sum(region == "low" for _, region in bands) == 21
    assert sum(region == "hf" for _, region in bands) == 11
    assert sum(region == "upper" for _, region in bands) == 1
    for cut in cuts:
        below = _full_band(math.nextafter(cut, -math.inf), boundaries)
        at = _full_band(cut, boundaries)
        above = _full_band(math.nextafter(cut, math.inf), boundaries)
        if cut == 18_000.0:
            assert below == at != above
        else:
            assert below != at == above
    assert _full_band(20.0, boundaries) == (-17, "low")
    assert _full_band(2000.0, boundaries) == (3, "hf")
    assert _full_band(18_000.0, boundaries) == (13, "hf")
    assert _full_band(20_000.0, boundaries) == (13, "upper")


def test_full_partition_preserves_existing_hf_band_membership() -> None:
    full = _grid_boundaries(100, -17, 13)
    hf = _interior_boundaries(100)
    for frequency in (2000.0, 2100.0, *hf, 16_000.0, 17_900.0, 18_000.0):
        assert _full_band(frequency, full) == (_band_index(frequency, hf), "hf")


def test_full_partition_requires_nyquist_at_or_above_20_khz() -> None:
    for rate in (40_000.0, math.nextafter(40_000.0, math.inf), 44_100.0, 96_000.0):
        _require_full_band_nyquist(rate)
    for rate in (math.nextafter(40_000.0, -math.inf), 0.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="Nyquist"):
            _require_full_band_nyquist(rate)


def test_required_band_examples_include_existing_evaluator_blind_spots() -> None:
    boundaries = _interior_boundaries(100)
    assert _band_index(2100, boundaries) == 3
    assert _band_index(16_000, boundaries) == 12
    assert _band_index(17_900, boundaries) == 13


@pytest.mark.parametrize("frequency", [1999.0, 18_001.0, math.nan, math.inf, -math.inf])
def test_outside_or_invalid_frequency_is_not_a_required_band(frequency: float) -> None:
    with pytest.raises(ValueError, match="outside the required HF interval"):
        _band_index(frequency, _interior_boundaries(100))


@pytest.mark.parametrize("frequency", [19.9, 20_001.0, math.nan, math.inf, -math.inf])
def test_outside_or_invalid_frequency_is_not_an_audible_band(frequency: float) -> None:
    with pytest.raises(ValueError, match="outside the required audible interval"):
        _full_band(frequency, _grid_boundaries(100, -17, 13))
