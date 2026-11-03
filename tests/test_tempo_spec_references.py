"""Independent normalized-space subproblem examples for tempo calibration.

The composed physical-control reference is tested separately; neither file
tests the current engine.
"""

from __future__ import annotations

import math
import struct
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

_F32 = struct.Struct(">f")
_U32 = struct.Struct(">I")

_NEG_ONE_KEY = 0x407FFFFF
_NEG_NORMAL_END_KEY = 0x7F7FFFFF
_NEG_ZERO_KEY = 0x7FFFFFFF
_POS_ZERO_KEY = 0x80000000
_POS_NORMAL_START_KEY = 0x80800000
_POS_ONE_KEY = 0xBF800000


def _f32(value: float) -> float:
    return _F32.unpack(_F32.pack(value))[0]


def _rsqrt_nr2(value: float) -> float:
    bits = _U32.unpack(_F32.pack(value))[0]
    estimate = _F32.unpack(_U32.pack(0x5F3759DF - (bits >> 1)))[0]
    for _ in range(2):
        half_product = _f32(_f32(0.5 * value) * estimate)
        estimate = _f32(estimate * _f32(1.5 - _f32(half_product * estimate)))
    return estimate


def _unstable_previous_formula(c1: float, c2: float, delta: float) -> float:
    discriminant = _f32(_f32(c1 * c1) + _f32(_f32(4 * c2) * delta))
    approximate_sqrt = _f32(discriminant * _rsqrt_nr2(discriminant))
    return _f32(_f32(approximate_sqrt - c1) * _f32(0.5 / c2))


def _coefficients(anchors: tuple[float, float, float]) -> tuple[float, float, float]:
    v_low, v_mid, v_high = anchors
    return v_mid, (v_high - v_low) * 0.5, (v_high + v_low) * 0.5 - v_mid


def _evaluate(anchors: tuple[float, float, float], u: float) -> float:
    midpoint, linear, quadratic = _coefficients(anchors)
    return midpoint + u * (linear + u * quadratic)


def _decimal_residual(anchors: tuple[float, float, float], u: float, target: float) -> Decimal:
    with localcontext() as context:
        context.prec = 200
        low, middle, high = map(Decimal.from_float, anchors)
        coordinate = Decimal.from_float(u)
        linear = (high - low) / 2
        quadratic = (high + low) / 2 - middle
        physical = middle + coordinate * (linear + coordinate * quadratic)
        return abs(physical - Decimal.from_float(target))


def _invert_monotone(
    anchors: tuple[float, float, float],
    target: float,
    low: float = -1.0,
    high: float = 1.0,
    tolerance: float = 1e-5,
) -> float:
    if not all(math.isfinite(value) for value in (*anchors, target, low, high, tolerance)):
        raise ValueError("calibration values must be finite")
    if not -1 <= low <= high <= 1 or tolerance < 0:
        raise ValueError("invalid safe interval")
    safe_low, safe_high = low, high
    _, linear, quadratic = _coefficients(anchors)
    derivative_low = linear + 2 * quadratic * low
    derivative_high = linear + 2 * quadratic * high
    ascending = derivative_low >= 0 and derivative_high >= 0
    descending = derivative_low <= 0 and derivative_high <= 0
    if low != high and ascending == descending:
        raise ValueError("calibration must be monotone and nonconstant")
    endpoint_low = _evaluate(anchors, low)
    endpoint_high = _evaluate(anchors, high)
    if not min(endpoint_low, endpoint_high) <= target <= max(endpoint_low, endpoint_high):
        raise ValueError("target is outside the safe physical range")
    for _ in range(160):
        middle = 0.5 * low + 0.5 * high
        physical = _evaluate(anchors, middle)
        if physical == target:
            low = high = middle
            break
        if (physical < target) == ascending:
            low = middle
        else:
            high = middle
        if _f32(low) == _f32(high):
            break
    candidates = [value for value in (_f32(low), _f32(high)) if safe_low <= value <= safe_high]
    if not candidates:
        raise ValueError("safe interval has no representable coordinate")
    best = min(candidates, key=lambda value: _decimal_residual(anchors, value, target))
    if _decimal_residual(anchors, best, target) > Decimal.from_float(tolerance):
        raise ValueError("target is not representable within tolerance")
    return best


def _decimal_inverse(anchors: tuple[float, float, float], target: float) -> float:
    with localcontext() as context:
        context.prec = 70
        low, middle, high = map(Decimal.from_float, anchors)
        delta = Decimal.from_float(target) - middle
        linear = (high - low) / 2
        quadratic = (high + low) / 2 - middle
        if quadratic == 0:
            return float(delta / linear)
        discriminant = linear * linear + 4 * quadratic * delta
        root = discriminant.sqrt()
        signed_root = root if linear >= 0 else -root
        return float(2 * delta / (linear + signed_root))


def _coordinate_count(flush_subnormals: bool) -> int:
    if flush_subnormals:
        return _NEG_NORMAL_END_KEY - _NEG_ONE_KEY + 1 + 1 + _POS_ONE_KEY - _POS_NORMAL_START_KEY + 1
    return _POS_ONE_KEY - _NEG_ONE_KEY


def _coordinate_at(rank: int, flush_subnormals: bool) -> float:
    count = _coordinate_count(flush_subnormals)
    if not 0 <= rank < count:
        raise ValueError("coordinate rank out of bounds")
    if flush_subnormals:
        negative_count = _NEG_NORMAL_END_KEY - _NEG_ONE_KEY + 1
        if rank < negative_count:
            key = _NEG_ONE_KEY + rank
        elif rank == negative_count:
            key = _POS_ZERO_KEY
        else:
            key = _POS_NORMAL_START_KEY + rank - negative_count - 1
    else:
        key = _NEG_ONE_KEY + rank
        if key >= _NEG_ZERO_KEY:
            key += 1
    bits = key ^ 0x80000000 if key & 0x80000000 else ~key & 0xFFFFFFFF
    return _F32.unpack(_U32.pack(bits))[0]


def _coordinate_bound(value: float, flush_subnormals: bool, *, upper: bool) -> int:
    low, high = 0, _coordinate_count(flush_subnormals)
    while low < high:
        middle = (low + high) // 2
        coordinate = _coordinate_at(middle, flush_subnormals)
        if coordinate < value or (upper and coordinate == value):
            low = middle + 1
        else:
            high = middle
    return low


def _exact_calibration(anchors: tuple[float, float, float], coordinate: float) -> Fraction:
    low, middle, high = map(Fraction.from_float, anchors)
    u = Fraction.from_float(coordinate)
    return (low * u * (u - 1) + 2 * middle * (1 - u * u) + high * u * (u + 1)) / 2


def _exact_derivative(anchors: tuple[float, float, float], coordinate: float) -> Fraction:
    low, middle, high = map(Fraction.from_float, anchors)
    u = Fraction.from_float(coordinate)
    return (high - low) / 2 + (high + low - 2 * middle) * u


def _invert_representable_candidate(
    anchors: tuple[float, float, float],
    target: float,
    low: float,
    high: float,
    *,
    flush_subnormals: bool,
) -> tuple[float, Fraction, int]:
    if not all(math.isfinite(value) for value in (*anchors, target, low, high)):
        raise ValueError("values must be finite")
    if not -1 <= low < high <= 1:
        raise ValueError("safe range must be nonempty")
    low_derivative = _exact_derivative(anchors, low)
    high_derivative = _exact_derivative(anchors, high)
    ascending = low_derivative >= 0 and high_derivative >= 0
    descending = low_derivative <= 0 and high_derivative <= 0
    if ascending == descending:
        raise ValueError("calibration must be monotone and nonconstant")
    desired = Fraction.from_float(target)
    endpoints = (_exact_calibration(anchors, low), _exact_calibration(anchors, high))
    if not min(endpoints) <= desired <= max(endpoints):
        raise ValueError("target is outside the continuous safe range")
    first = _coordinate_bound(low, flush_subnormals, upper=False)
    end = _coordinate_bound(high, flush_subnormals, upper=True)
    if first == end:
        raise ValueError("safe range has no representable coordinate")
    lower, upper = first, end
    comparisons = 0
    while lower < upper:
        middle = (lower + upper) // 2
        value = _exact_calibration(anchors, _coordinate_at(middle, flush_subnormals))
        comparisons += 1
        before_target = value < desired if ascending else value > desired
        if before_target:
            lower = middle + 1
        else:
            upper = middle
    candidates = [
        _coordinate_at(rank, flush_subnormals) for rank in (lower - 1, lower) if first <= rank < end
    ]
    best = min(candidates, key=lambda u: (abs(_exact_calibration(anchors, u) - desired), u))
    return best, abs(_exact_calibration(anchors, best) - desired), comparisons


def test_previous_inverse_is_unstable_for_nearly_linear_calibration() -> None:
    c1 = _f32(1.0)
    c2 = _f32(1e-6)
    target = _f32(0.5 + 0.25 * c2)
    previous = _unstable_previous_formula(c1, c2, target)
    assert abs(previous - 0.5) > 1.0


def test_previous_inverse_chooses_wrong_branch_for_descending_calibration() -> None:
    previous = _unstable_previous_formula(-1.0, _f32(1e-6), -0.5)
    assert previous > 1e5


def test_two_fast_inverse_square_root_steps_are_not_full_f32_precision() -> None:
    assert abs(_rsqrt_nr2(1.0) - 1.0) > 1e-6


@pytest.mark.parametrize(
    ("anchors", "u"),
    [
        ((-0.999999, 0.0, 1.000001), 0.5),
        ((100.0, 475.0, 1000.0), -0.8),
        ((1000.0, 475.0, 100.0), 0.7),
        ((0.0, 1.0, 4.0), -0.99),
    ],
)
def test_bisection_matches_high_precision_inverse(
    anchors: tuple[float, float, float], u: float
) -> None:
    target = _evaluate(anchors, u)
    actual = _invert_monotone(anchors, target)
    expected = _decimal_inverse(anchors, target)
    assert abs(actual - expected) < 1e-7


def test_inversion_rejects_nonmonotone_flat_and_out_of_range_targets() -> None:
    with pytest.raises(ValueError, match="monotone"):
        _invert_monotone((0.0, 100.0, 10.0), 1.0)
    with pytest.raises(ValueError, match="nonconstant"):
        _invert_monotone((20.0, 20.0, 20.0), 20.0)
    with pytest.raises(ValueError, match="safe physical range"):
        _invert_monotone((0.0, 10.0, 20.0), 30.0)


def test_monotonicity_can_be_restricted_to_a_valid_safe_interval() -> None:
    anchors = (0.0, 100.0, 10.0)
    coordinate = _invert_monotone(anchors, 75.0, low=-1.0, high=-0.4)
    assert -1.0 <= coordinate <= -0.4
    assert _decimal_residual(anchors, coordinate, 75.0) <= Decimal("0.00001")


def test_unrepresentable_physical_target_is_not_reported_as_exact() -> None:
    scale = float(2**100)
    with pytest.raises(ValueError, match="not representable"):
        _invert_monotone((-scale, scale, 3 * scale), 1.0, tolerance=1e-6)


def test_subnormal_coordinate_can_be_destroyed_by_runtime_flush_to_zero() -> None:
    scale = float(2**126)
    anchors = (-scale, 0.0, scale)
    coordinate = _invert_monotone(anchors, 0.125, tolerance=1e-6)
    assert 0.0 < coordinate < 2**-126
    assert _decimal_residual(anchors, coordinate, 0.125) == 0
    assert _decimal_residual(anchors, 0.0, 0.125) == Decimal("0.125")


def test_single_safe_coordinate_is_not_treated_as_an_invertible_range() -> None:
    anchors = (0.0, 10.0, 20.0)
    assert _invert_monotone(anchors, 10.0, low=0.0, high=0.0) == 0.0
    with pytest.raises(ValueError, match="safe physical range"):
        _invert_monotone(anchors, 11.0, low=0.0, high=0.0)


def test_representable_coordinate_order_and_subnormal_policy() -> None:
    for flush in (False, True):
        count = _coordinate_count(flush)
        assert count < 2**31
        assert _coordinate_at(0, flush) == -1.0
        assert _coordinate_at(count - 1, flush) == 1.0
        zero_rank = _coordinate_bound(0.0, flush, upper=False)
        assert _coordinate_at(zero_rank, flush) == 0.0
        assert math.copysign(1.0, _coordinate_at(zero_rank, flush)) == 1.0
        assert _coordinate_at(zero_rank - 1, flush) < 0.0
        assert _coordinate_at(zero_rank + 1, flush) > 0.0
        expected = 2**-126 if flush else 2**-149
        assert _coordinate_at(zero_rank + 1, flush) == expected


def test_representable_search_handles_overflowing_f64_intermediates() -> None:
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    anchors = (-maximum, 0.0, maximum)
    assert not math.isfinite(_coefficients(anchors)[1])
    coordinate, residual, comparisons = _invert_representable_candidate(
        anchors, 0.0, -1.0, 1.0, flush_subnormals=False
    )
    assert coordinate == 0.0
    assert residual == 0
    assert comparisons <= 31


def test_representable_search_reports_ftz_loss_and_empty_interval() -> None:
    scale = float(2**126)
    anchors = (-scale, 0.0, scale)
    gradual, gradual_residual, _ = _invert_representable_candidate(
        anchors, 0.125, -1.0, 1.0, flush_subnormals=False
    )
    flushed, flushed_residual, _ = _invert_representable_candidate(
        anchors, 0.125, -1.0, 1.0, flush_subnormals=True
    )
    assert gradual == 2**-129
    assert gradual_residual == 0
    assert flushed == 0.0
    assert flushed_residual == Fraction(1, 8)
    with pytest.raises(ValueError, match="no representable"):
        _invert_representable_candidate(anchors, 0.125, 2**-130, 2**-128, flush_subnormals=True)


def test_representable_search_handles_singletons_ties_and_restricted_monotonicity() -> None:
    linear = (-1.0, 0.0, 1.0)
    only, residual, _ = _invert_representable_candidate(
        linear, 0.0, -(2**-150), 2**-150, flush_subnormals=False
    )
    assert only == 0.0
    assert residual == 0
    tied, residual, _ = _invert_representable_candidate(
        linear, 1.0 - 2**-25, -1.0, 1.0, flush_subnormals=False
    )
    assert tied == 1.0 - 2**-24
    assert residual == Fraction(1, 2**25)
    with pytest.raises(ValueError, match="monotone"):
        _invert_representable_candidate((0.0, 100.0, 10.0), 75.0, -1.0, 1.0, flush_subnormals=False)
    restricted, _, _ = _invert_representable_candidate(
        (0.0, 100.0, 10.0), 75.0, -1.0, -0.4, flush_subnormals=False
    )
    assert -1.0 <= restricted <= -0.4


def test_representable_search_honors_binary64_safe_boundaries() -> None:
    linear = (-1.0, 0.0, 1.0)
    previous = 1.0 - 2**-24
    low = math.nextafter(previous, -math.inf)
    high = math.nextafter(previous, math.inf)
    coordinate, residual, _ = _invert_representable_candidate(
        linear, previous, low, high, flush_subnormals=False
    )
    assert coordinate == previous
    assert residual == 0
    with pytest.raises(ValueError, match="no representable"):
        _invert_representable_candidate(
            linear,
            1.0 - 2**-25,
            math.nextafter(previous, math.inf),
            math.nextafter(1.0, -math.inf),
            flush_subnormals=False,
        )


@pytest.mark.parametrize("flush_subnormals", [False, True])
@pytest.mark.parametrize("anchors", [(-1.0, 0.0, 1.0), (0.0, 1.0, 4.0), (4.0, 1.0, 0.0)])
def test_representable_search_matches_brute_force_near_boundaries(
    flush_subnormals: bool, anchors: tuple[float, float, float]
) -> None:
    count = _coordinate_count(flush_subnormals)
    zero = _coordinate_bound(0.0, flush_subnormals, upper=False)
    for first in (0, count - 5, zero - 2, _coordinate_bound(0.5, flush_subnormals, upper=False)):
        coordinates = [_coordinate_at(rank, flush_subnormals) for rank in range(first, first + 5)]
        low, high = coordinates[0], coordinates[-1]
        midpoint = float(
            (
                _exact_calibration(anchors, coordinates[1])
                + _exact_calibration(anchors, coordinates[2])
            )
            / 2
        )
        desired = Fraction.from_float(midpoint)
        endpoint_values = (_exact_calibration(anchors, low), _exact_calibration(anchors, high))
        if not min(endpoint_values) <= desired <= max(endpoint_values):
            continue
        expected = min(
            coordinates, key=lambda u: (abs(_exact_calibration(anchors, u) - desired), u)
        )
        actual, residual, comparisons = _invert_representable_candidate(
            anchors, midpoint, low, high, flush_subnormals=flush_subnormals
        )
        assert actual == expected
        assert residual == abs(_exact_calibration(anchors, expected) - desired)
        assert comparisons <= 31


def test_exact_binary64_calibration_fits_fixed_dyadic_accumulator() -> None:
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    minimum = float.fromhex("0x0.0000000000001p-1022")
    for anchors in (
        (maximum, -maximum, maximum),
        (-maximum, 0.0, maximum),
        (minimum, -minimum, minimum),
    ):
        for coordinate in (-1.0, -minimum, 0.0, minimum, 1.0):
            for target in (-maximum, minimum, maximum):
                scaled = 2 * (_exact_calibration(anchors, coordinate) - Fraction.from_float(target))
                scaled *= 2**3222
                assert scaled.denominator == 1
                assert abs(scaled.numerator) < 2**4255
            derivative = 2 * _exact_derivative(anchors, coordinate) * 2**3222
            assert derivative.denominator == 1
            assert abs(derivative.numerator) < 2**4255
