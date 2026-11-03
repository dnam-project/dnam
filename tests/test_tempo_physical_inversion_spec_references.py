"""Composed physical-control tempo references, not production engine tests."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from fractions import Fraction

import pytest

from test_control_bound_spec_references import _intersection, _rank, _value
from test_control_normalization_spec_references import _correctly_rounded


@dataclass(frozen=True)
class _Domain:
    declared: tuple[float, float]
    safe: tuple[float, float]
    anchors: tuple[float, float, float]
    first: int
    end: int
    ascending: bool
    flush_physical: bool
    flush_normalized: bool
    continuous_values: tuple[Fraction, Fraction]


@dataclass(frozen=True)
class _Inversion:
    physical: float
    normalized: float
    realized: Fraction
    residual: Fraction
    searches: tuple[int, int]


def _real_normalized(declared: tuple[float, float], physical: float) -> Fraction:
    lower, upper = map(Fraction.from_float, declared)
    value = Fraction.from_float(physical)
    return (2 * value - lower - upper) / (upper - lower)


def _normalized(declared: tuple[float, float], physical: float, *, flush: bool) -> float:
    value = _correctly_rounded(_real_normalized(declared, physical))
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    if flush and 0 < (bits & 0x7FFF_FFFF) < 0x0080_0000:
        return 0.0
    return value


def _calibration(anchors: tuple[float, float, float], coordinate: Fraction) -> Fraction:
    low, middle, high = map(Fraction.from_float, anchors)
    return (
        low * coordinate * (coordinate - 1)
        + 2 * middle * (1 - coordinate * coordinate)
        + high * coordinate * (coordinate + 1)
    ) / 2


def _derivative(anchors: tuple[float, float, float], coordinate: Fraction) -> Fraction:
    low, middle, high = map(Fraction.from_float, anchors)
    return (high - low) / 2 + (high + low - 2 * middle) * coordinate


def _nonnegative_on_hull(
    anchors: tuple[float, float, float], hull: tuple[Fraction, Fraction]
) -> bool:
    if min(_calibration(anchors, coordinate) for coordinate in hull) < 0:
        return False
    low, middle, high = map(Fraction.from_float, anchors)
    curvature = high + low - 2 * middle
    if curvature > 0:
        vertex = -(high - low) / (2 * curvature)
        if hull[0] < vertex < hull[1]:
            return _calibration(anchors, vertex) >= 0
    return True


def _domain(
    declared: tuple[float, float],
    safe: tuple[float, float],
    anchors: tuple[float, float, float],
    *,
    flush_physical: bool = False,
    flush_normalized: bool = False,
) -> _Domain:
    if not all(math.isfinite(value) for value in (*declared, *safe, *anchors)):
        raise ValueError("nonfinite model value")
    if declared[0] >= declared[1] or safe[0] > safe[1]:
        raise ValueError("invalid physical interval")
    lower, upper = max(declared[0], safe[0]), min(declared[1], safe[1])
    if lower > upper:
        raise ValueError("empty physical interval")
    first, end = _intersection((lower,), (upper,), flush_subnormals=flush_physical)
    continuous = (_real_normalized(declared, lower), _real_normalized(declared, upper))
    reachable = tuple(
        Fraction.from_float(
            _normalized(
                declared, _value(rank, flush_subnormals=flush_physical), flush=flush_normalized
            )
        )
        for rank in (first, end - 1)
    )
    hull = (min(*continuous, *reachable), max(*continuous, *reachable))
    endpoint_values = tuple(_calibration(anchors, coordinate) for coordinate in continuous)
    if not _nonnegative_on_hull(anchors, hull):
        raise ValueError("negative physical calibration")
    ascending = True
    if lower < upper:
        slopes = tuple(_derivative(anchors, coordinate) for coordinate in hull)
        ascending = min(slopes) >= 0
        descending = max(slopes) <= 0
        if ascending == descending:
            raise ValueError("calibration is not monotone and nonconstant")
    return _Domain(
        declared,
        (lower, upper),
        anchors,
        first,
        end,
        ascending,
        flush_physical,
        flush_normalized,
        (endpoint_values[0], endpoint_values[1]),
    )


def _at(domain: _Domain, rank: int) -> tuple[float, float, Fraction]:
    physical = _value(rank, flush_subnormals=domain.flush_physical)
    normalized = _normalized(domain.declared, physical, flush=domain.flush_normalized)
    return physical, normalized, _calibration(domain.anchors, Fraction.from_float(normalized))


def _invert(domain: _Domain, target: float, *, require_inside: bool = True) -> _Inversion | None:
    if not math.isfinite(target):
        raise ValueError("nonfinite target")
    wanted = Fraction.from_float(target)
    if require_inside:
        if domain.safe[0] == domain.safe[1]:
            if wanted != _at(domain, domain.first)[2]:
                return None
        elif not min(domain.continuous_values) <= wanted <= max(domain.continuous_values):
            return None

    low, high = domain.first, domain.end
    crossing_steps = 0
    while low < high:
        middle = (low + high) // 2
        value = _at(domain, middle)[2]
        crossing_steps += 1
        before = value < wanted if domain.ascending else value > wanted
        if before:
            low = middle + 1
        else:
            high = middle
    candidates = [rank for rank in (low - 1, low) if domain.first <= rank < domain.end]
    best = min(candidates, key=lambda rank: (abs(_at(domain, rank)[2] - wanted), rank))
    chosen_value = _at(domain, best)[2]

    low, high = domain.first, best
    plateau_steps = 0
    while low < high:
        middle = (low + high) // 2
        value = _at(domain, middle)[2]
        plateau_steps += 1
        before = value < chosen_value if domain.ascending else value > chosen_value
        if before:
            low = middle + 1
        else:
            high = middle
    physical, normalized, realized = _at(domain, low)
    return _Inversion(
        physical, normalized, realized, abs(realized - wanted), (crossing_steps, plateau_steps)
    )


def test_normalized_midpoint_can_be_unreachable_from_physical_f32() -> None:
    lower = float(2**30)
    upper = lower + 128.0
    domain = _domain((lower, upper), (lower, upper), (0.0, 1.0, 2.0))
    result = _invert(domain, 1.0)
    assert result is not None
    lower_rank = _rank(lower, flush_subnormals=False)
    assert lower_rank is not None
    assert lower_rank + 1 == _rank(upper, flush_subnormals=False)
    assert result.physical == lower
    assert result.normalized == -1.0
    assert result.realized == 0
    assert result.residual == 1


def test_reachable_option_beats_unreachable_normalized_midpoint() -> None:
    lower = float(2**30)
    coarse = _domain((lower, lower + 128.0), (lower, lower + 128.0), (0.0, 1.0, 2.0))
    fine = _domain((-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0, 2.0))
    results = (_invert(coarse, 1.0), _invert(fine, 1.0))
    assert results[0] is not None
    assert results[0].residual == 1
    assert results[1] is not None
    assert results[1].residual == 0
    assert results[1].physical == results[1].normalized == 0.0


@pytest.mark.parametrize("descending", [False, True])
def test_search_matches_brute_force_for_small_physical_intervals(descending: bool) -> None:
    anchors = (20.0, 10.0, 0.0) if descending else (0.0, 10.0, 20.0)
    start = _rank(0.5, flush_subnormals=False)
    assert start is not None
    for offset in range(8):
        first = start + offset
        last = first + 6
        safe = (
            _value(first, flush_subnormals=False),
            _value(last, flush_subnormals=False),
        )
        domain = _domain((0.0, 1.0), safe, anchors)
        endpoints = (domain.continuous_values[0], domain.continuous_values[1])
        target = float((endpoints[0] + endpoints[1]) / 2)
        wanted = Fraction.from_float(target)
        expected = min(
            range(first, last + 1),
            key=lambda rank: (abs(_at(domain, rank)[2] - wanted), rank),
        )
        result = _invert(domain, target)
        assert result is not None
        assert result.physical == _value(expected, flush_subnormals=False)
        assert result.residual == abs(_at(domain, expected)[2] - wanted)
        assert max(result.searches) <= 32


def test_plateau_residual_tie_selects_lowest_physical_coordinate() -> None:
    scale = float(2**100)
    domain = _domain((-scale, scale), (0.0, 1.0), (-1.0, 0.0, 1.0))
    result = _invert(domain, 2**-150)
    assert result is not None
    assert result.physical == result.normalized == 0.0
    assert result.residual == Fraction(1, 2**150)
    assert _normalized(domain.declared, 2**-50, flush=False) == 0.0
    assert max(result.searches) <= 32


@pytest.mark.parametrize(
    ("flush_physical", "flush_normalized"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_physical_and_normalized_flush_policies_are_distinct(
    flush_physical: bool, flush_normalized: bool
) -> None:
    domain = _domain(
        (-1.0, 1.0),
        (0.0, 1.0),
        (-1.0, 0.0, 1.0),
        flush_physical=flush_physical,
        flush_normalized=flush_normalized,
    )
    result = _invert(domain, 2**-129)
    assert result is not None
    if flush_physical or flush_normalized:
        assert result.physical == result.normalized == 0.0
        assert result.residual == Fraction(1, 2**129)
    else:
        assert result.physical == result.normalized == 2**-129
        assert result.residual == 0


@pytest.mark.parametrize("flush_physical", [False, True])
@pytest.mark.parametrize("flush_normalized", [False, True])
def test_normalized_flush_can_lose_a_normal_physical_knob(
    flush_physical: bool, flush_normalized: bool
) -> None:
    scale = float(2**100)
    domain = _domain(
        (-scale, scale),
        (2**-30, 2**-28),
        (-1.0, 0.0, 1.0),
        flush_physical=flush_physical,
        flush_normalized=flush_normalized,
    )
    result = _invert(domain, 2**-129)
    assert result is not None
    if flush_normalized:
        assert result.physical == 2**-30
        assert result.normalized == 0.0
        assert result.residual == Fraction(1, 2**129)
    else:
        assert result.physical == 2**-29 - 2**-50
        assert result.normalized == 2**-129
        assert result.residual == 0
        rank = _rank(result.physical, flush_subnormals=flush_physical)
        assert rank is not None
        assert _at(domain, rank - 1)[1] < result.normalized


@pytest.mark.parametrize("flush_physical", [False, True])
@pytest.mark.parametrize("flush_normalized", [False, True])
def test_physical_flush_can_lose_a_normalized_normal_value(
    flush_physical: bool, flush_normalized: bool
) -> None:
    minimum_normal = 2**-126
    domain = _domain(
        (-minimum_normal, minimum_normal),
        (0.0, minimum_normal),
        (-1.0, 0.0, 1.0),
        flush_physical=flush_physical,
        flush_normalized=flush_normalized,
    )
    result = _invert(domain, 0.125)
    assert result is not None
    if flush_physical:
        assert result.physical == result.normalized == 0.0
        assert result.residual == Fraction(1, 8)
    else:
        assert result.physical == 2**-129
        assert result.normalized == 0.125
        assert result.residual == 0


def test_empty_and_singleton_physical_intervals_are_reported() -> None:
    with pytest.raises(ValueError, match="empty representable"):
        _domain((0.0, 1.0), (0.1, 0.1), (0.0, 1.0, 2.0))
    fixed = _domain((0.0, 1.0), (0.5, 0.5), (0.0, 1.0, 2.0))
    assert _invert(fixed, 1.0) is not None
    assert _invert(fixed, 1.1) is None
    clamped = _invert(fixed, 1.1, require_inside=False)
    assert clamped is not None
    assert clamped.physical == 0.5


def test_singleton_hull_checks_negative_interior_calibration_vertex() -> None:
    desired_normalized = 0.5 + 2**-25
    declared = (-1.0, (1 - desired_normalized) / (1 + desired_normalized))
    ideal = _real_normalized(declared, 0.0)
    rounded = Fraction.from_float(_normalized(declared, 0.0, flush=False))
    midpoint = float((ideal + rounded) / 2)
    epsilon = 2**-53
    anchors = ((-1 - midpoint) ** 2 - epsilon, midpoint**2 - epsilon, (1 - midpoint) ** 2 - epsilon)
    low, middle, high = map(Fraction.from_float, anchors)
    vertex = -(high - low) / (2 * (high + low - 2 * middle))
    assert min(ideal, rounded) < vertex < max(ideal, rounded)
    assert _calibration(anchors, ideal) > 0
    assert _calibration(anchors, rounded) > 0
    assert _calibration(anchors, vertex) < 0
    with pytest.raises(ValueError, match="negative physical calibration"):
        _domain(declared, (0.0, 0.0), anchors)


def test_nonsingleton_interval_with_one_f32_value_reports_best_effort_residual() -> None:
    safe = (math.nextafter(0.5, -math.inf), math.nextafter(0.5, math.inf))
    domain = _domain((0.0, 1.0), safe, (0.0, 1.0, 2.0))
    assert domain.end - domain.first == 1
    result = _invert(domain, 1.0 + 2**-52)
    assert result is not None
    assert result.physical == 0.5
    assert result.normalized == 0.0
    assert result.residual == Fraction(1, 2**52)


def test_monotonicity_includes_a_rounded_coordinate_outside_the_ideal_interval() -> None:
    critical = -1.0 / 3.0 - 2**-27
    anchors = (
        1.0 + (-1.0 - critical) ** 2,
        1.0 + critical**2,
        1.0 + (1.0 - critical) ** 2,
    )
    ideal = _real_normalized((0.0, 3.0), 1.0)
    rounded = Fraction.from_float(_normalized((0.0, 3.0), 1.0, flush=False))
    assert rounded < ideal
    assert _derivative(anchors, ideal) > 0
    assert _derivative(anchors, rounded) < 0
    rank = _rank(1.0, flush_subnormals=False)
    assert rank is not None
    next_physical = _value(rank + 1, flush_subnormals=False)
    with pytest.raises(ValueError, match="not monotone"):
        _domain((0.0, 3.0), (1.0, next_physical), anchors)


def _integer64(value: float) -> int:
    scaled = Fraction.from_float(value) * 2**1074
    assert scaled.denominator == 1
    return scaled.numerator


def test_composed_exact_comparison_integer_widths_at_binary64_extremes() -> None:
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    minimum = float.fromhex("0x0.0000000000001p-1022")
    intervals = (
        (-maximum, maximum, 0.0),
        (-maximum, maximum, maximum),
        (-minimum, minimum, 0.0),
        (0.0, maximum, minimum),
    )
    anchors_cases = (
        (-maximum, 0.0, maximum),
        (maximum, -maximum, maximum),
        (-maximum, -maximum, maximum),
    )
    for lower, upper, physical in intervals:
        a, b, x = map(_integer64, (lower, upper, physical))
        numerator, denominator = 2 * x - a - b, b - a
        assert abs(numerator) <= denominator < 2**2099
        normalized = Fraction(numerator, denominator)
        for left, right in ((-1.0, 1.0), (0.0, 2**-149), (-(2**-149), 0.0)):
            left_units = Fraction.from_float(left) * 2**149
            right_units = Fraction.from_float(right) * 2**149
            assert left_units.denominator == right_units.denominator == 1
            midpoint_units = left_units.numerator + right_units.numerator
            difference = (numerator << 150) - midpoint_units * denominator
            assert abs(difference) < 2**2250
            assert Fraction(difference, denominator * 2**150) == normalized - Fraction(
                midpoint_units, 2**150
            )
        for anchors in anchors_cases:
            low, middle, high = map(_integer64, anchors)
            derivative = (high - low) * denominator + 2 * (high + low - 2 * middle) * numerator
            assert abs(derivative) < 2**4201
            assert Fraction(derivative, 2**1075 * denominator) == _derivative(anchors, normalized)
            target = _integer64(maximum)
            residual = (
                low * numerator * (numerator - denominator)
                + 2 * middle * (denominator**2 - numerator**2)
                + high * numerator * (numerator + denominator)
                - 2 * target * denominator**2
            )
            assert abs(residual) < 2**6299
            assert Fraction(residual, 2**1075 * denominator**2) == _calibration(
                anchors, normalized
            ) - Fraction.from_float(maximum)
            curvature = high + low - 2 * middle
            if curvature > 0:
                delta = high - low
                vertex_difference = -delta * denominator - 2 * curvature * numerator
                minimum_numerator = 8 * curvature * middle - delta**2
                assert abs(vertex_difference) < 2**4201
                assert abs(minimum_numerator) < 2**4202
                vertex = Fraction(-delta, 2 * curvature)
                assert Fraction(vertex_difference, 2 * curvature * denominator) == (
                    vertex - normalized
                )
                assert _calibration(anchors, vertex) * 2**1074 * 8 * curvature == minimum_numerator


def test_invalid_and_nonmonotone_calibrations_are_rejected() -> None:
    with pytest.raises(ValueError, match="not monotone"):
        _domain((-1.0, 1.0), (-1.0, 1.0), (0.0, 100.0, 10.0))
    with pytest.raises(ValueError, match="negative physical"):
        _domain((-1.0, 1.0), (-1.0, 1.0), (-1.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="nonfinite"):
        _domain((-1.0, math.inf), (0.0, 1.0), (0.0, 1.0, 2.0))
