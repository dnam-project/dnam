"""Integer-interval enclosure of the ideal H1 Hilbert response.

The box partitioner uses floating-point estimates only to choose subdivisions.
Acceptance uses outward-rounded integer intervals; this does not test the engine.
"""

from __future__ import annotations

import itertools
import math
import random
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

from test_op_m_profile_bounds import _estimate_box
from test_op_m_spec_references import (
    _HILBERT_PROFILE_BITS,
    _HILBERT_RATE_BANDS,
    _coefficients,
    _from_bits,
    _quadrature_error_degrees,
)

_SCALE = 1 << 128


def _ceil_div(numerator: int, denominator: int) -> int:
    return -((-numerator) // denominator)


@dataclass(frozen=True, slots=True)
class _Interval:
    lo: int
    hi: int

    @classmethod
    def rational(cls, value: Fraction | int) -> _Interval:
        fraction = Fraction(value)
        numerator = fraction.numerator * _SCALE
        return cls(numerator // fraction.denominator, _ceil_div(numerator, fraction.denominator))

    @classmethod
    def between(cls, low: Fraction, high: Fraction) -> _Interval:
        assert low <= high
        return cls(cls.rational(low).lo, cls.rational(high).hi)

    def __neg__(self) -> _Interval:
        return _Interval(-self.hi, -self.lo)

    def __add__(self, other: _Interval) -> _Interval:
        return _Interval(self.lo + other.lo, self.hi + other.hi)

    def __sub__(self, other: _Interval) -> _Interval:
        return self + -other

    def __mul__(self, other: _Interval) -> _Interval:
        products = (
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        )
        return _Interval(min(products) // _SCALE, _ceil_div(max(products), _SCALE))

    def __truediv__(self, other: _Interval) -> _Interval:
        assert other.lo > 0
        ratios = tuple(
            (numerator * _SCALE, denominator)
            for numerator in (self.lo, self.hi)
            for denominator in (other.lo, other.hi)
        )
        return _Interval(
            min(numerator // denominator for numerator, denominator in ratios),
            max(_ceil_div(numerator, denominator) for numerator, denominator in ratios),
        )

    def divide_integer(self, value: int) -> _Interval:
        assert value > 0
        return _Interval(self.lo // value, _ceil_div(self.hi, value))

    def times_integer(self, value: int) -> _Interval:
        if value < 0:
            return (-self).times_integer(-value)
        return _Interval(value * self.lo, value * self.hi)

    def square(self) -> _Interval:
        minimum = 0 if self.lo <= 0 <= self.hi else min(self.lo**2, self.hi**2)
        maximum = max(self.lo**2, self.hi**2)
        return _Interval(minimum // _SCALE, _ceil_div(maximum, _SCALE))

    def absolute_upper(self) -> _Interval:
        return _Interval(0, max(abs(self.lo), abs(self.hi)))

    def contains(self, value: Fraction) -> bool:
        return (
            self.lo * value.denominator <= value.numerator * _SCALE <= self.hi * value.denominator
        )


_ZERO = _Interval.rational(0)
_ONE = _Interval.rational(1)


def _arctan_reciprocal_bounds(denominator: int) -> tuple[Fraction, Fraction]:
    terms = 40
    partial = sum(
        (
            Fraction((-1) ** index, (2 * index + 1) * denominator ** (2 * index + 1))
            for index in range(terms)
        ),
        Fraction(),
    )
    next_term = Fraction(1, (2 * terms + 1) * denominator ** (2 * terms + 1))
    return partial, partial + next_term


def _pi_interval() -> _Interval:
    # Machin's identity and alternating-series remainders enclose pi.
    low5, high5 = _arctan_reciprocal_bounds(5)
    low239, high239 = _arctan_reciprocal_bounds(239)
    return _Interval.between(16 * low5 - 4 * high239, 16 * high5 - 4 * low239)


_PI = _pi_interval()
_SERIES_TERMS = 32
_SIN_REMAINDER = _ceil_div(
    4 ** (2 * _SERIES_TERMS + 3) * _SCALE, math.factorial(2 * _SERIES_TERMS + 3)
)
_COS_REMAINDER = _ceil_div(
    4 ** (2 * _SERIES_TERMS + 2) * _SCALE, math.factorial(2 * _SERIES_TERMS + 2)
)


@lru_cache(maxsize=16_384)
def _sin_cos(angle: _Interval) -> tuple[_Interval, _Interval]:
    assert 0 <= angle.lo <= angle.hi <= 4 * _SCALE
    squared = angle.square()
    sine_term = sine = angle
    cosine_term = cosine = _ONE
    for index in range(1, _SERIES_TERMS + 1):
        sine_term = -(sine_term * squared).divide_integer((2 * index) * (2 * index + 1))
        cosine_term = -(cosine_term * squared).divide_integer((2 * index - 1) * (2 * index))
        sine = sine + sine_term
        cosine = cosine + cosine_term
    return (
        _Interval(sine.lo - _SIN_REMAINDER, sine.hi + _SIN_REMAINDER),
        _Interval(cosine.lo - _COS_REMAINDER, cosine.hi + _COS_REMAINDER),
    )


def _omega(frequency: Fraction, rate: Fraction) -> _Interval:
    assert 0 < 2 * frequency < rate
    return _PI * _Interval.rational(2 * frequency / rate)


def _sine_ranges(low: Fraction, high: Fraction) -> tuple[_Interval, _Interval]:
    assert 0 < low <= high < Fraction(1, 2)
    sin_low, cos_low = _sin_cos(_PI * _Interval.rational(2 * low))
    sin_high, cos_high = _sin_cos(_PI * _Interval.rational(2 * high))
    square_low, square_high = sin_low.square(), sin_high.square()
    sine_square = _Interval(min(square_low.lo, square_high.lo), max(square_low.hi, square_high.hi))
    if low <= Fraction(1, 4) <= high:
        sine_square = _Interval(sine_square.lo, _SCALE)
    sine_square = _Interval(max(0, sine_square.lo), min(_SCALE, sine_square.hi))
    double_low = (sin_low * cos_low).times_integer(2)
    double_high = (sin_high * cos_high).times_integer(2)
    sine_double = _Interval(min(double_low.lo, double_high.lo), max(double_low.hi, double_high.hi))
    if low <= Fraction(1, 8) <= high:
        sine_double = _Interval(sine_double.lo, _SCALE)
    if low <= Fraction(3, 8) <= high:
        sine_double = _Interval(-_SCALE, sine_double.hi)
    return sine_square, _Interval(max(-_SCALE, sine_double.lo), min(_SCALE, sine_double.hi))


@lru_cache
def _coefficient_data(low: int, high: int) -> tuple[tuple[Fraction, Fraction], ...]:
    result = []
    for left, right in zip(_HILBERT_PROFILE_BITS[low], _HILBERT_PROFILE_BITS[high], strict=True):
        first = Fraction.from_float(_from_bits(left))
        last = Fraction.from_float(_from_bits(right))
        result.append((first, (last - first) / (high - low)))
    return tuple(result)


def _complex_multiply(
    left: tuple[_Interval, _Interval], right: tuple[_Interval, _Interval]
) -> tuple[_Interval, _Interval]:
    real_left, imaginary_left = left
    real_right, imaginary_right = right
    return (
        real_left * real_right - imaginary_left * imaginary_right,
        real_left * imaginary_right + imaginary_left * real_right,
    )


def _center_error(
    rate: Fraction, frequency: Fraction, anchor_low: int, anchor_high: int
) -> _Interval:
    sine, cosine = _sin_cos(_omega(frequency, rate))
    sine_square = sine.square()
    # Each all-pass has unit magnitude, so i*Q/I equals i*Q*conjugate(I).
    response = (sine, cosine)
    for index, (base, slope) in enumerate(_coefficient_data(anchor_low, anchor_high)):
        alpha = _Interval.rational(base + (rate - anchor_low) * slope)
        denominator = (_ONE - alpha).square() + (alpha * sine_square).times_integer(4)
        assert denominator.lo > 0
        real = ((_ONE + alpha).square() * sine_square).times_integer(2) - denominator
        imaginary = ((_ONE - alpha.square()) * sine * cosine).times_integer(2)
        section = (real / denominator, imaginary / denominator)
        if index < 6:
            section = (section[0], -section[1])
        response = _complex_multiply(response, section)
    assert response[0].lo > 0
    return response[1].absolute_upper() / _Interval(response[0].lo, response[0].lo)


type _Box = tuple[Fraction, Fraction, Fraction, Fraction, int, int, int]


def _enclose_box(box: _Box) -> tuple[int, int, int]:
    rate_low, rate_high, frequency_low, frequency_high, anchor_low, anchor_high, _ = box
    omega_low = _omega(frequency_low, rate_high)
    omega_high = _omega(frequency_high, rate_low)
    omega = _Interval(omega_low.lo, omega_high.hi)
    sine_square, sine_double = _sine_ranges(frequency_low / rate_high, frequency_high / rate_low)
    phase_omega = -_ONE
    rate_coefficient = _ZERO
    rounding_error = _ZERO
    # The largest f32 spacing below one is 2^-24.
    rounding = Fraction(1, 1 << 25)
    for index, (base, slope) in enumerate(_coefficient_data(anchor_low, anchor_high)):
        left = base + (rate_low - anchor_low) * slope
        right = base + (rate_high - anchor_low) * slope
        alpha = _Interval.between(min(left, right) - rounding, max(left, right) + rounding)
        assert 0 < alpha.lo <= alpha.hi < _SCALE
        denominator = (_ONE - alpha).square() + (alpha * sine_square).times_integer(4)
        assert denominator.lo > 0
        derivative_omega = -((_ONE - alpha.square()).times_integer(2) / denominator)
        derivative_alpha = -(sine_double.times_integer(2) / denominator)
        if index < 6:
            derivative_omega = -derivative_omega
            derivative_alpha = -derivative_alpha
        phase_omega = phase_omega + derivative_omega
        rate_coefficient = rate_coefficient + derivative_alpha * _Interval.rational(slope)
        rounding_error = rounding_error + derivative_alpha.absolute_upper() * _Interval.rational(
            rounding
        )
    rate = _Interval.between(rate_low, rate_high)
    frequency_gradient = phase_omega * (_PI.times_integer(2) / rate)
    rate_gradient = phase_omega * -(omega / rate) + rate_coefficient
    frequency_radius = frequency_gradient.absolute_upper() * _Interval.rational(
        (frequency_high - frequency_low) / 2
    )
    rate_radius = rate_gradient.absolute_upper() * _Interval.rational((rate_high - rate_low) / 2)
    center = _center_error(
        (rate_low + rate_high) / 2,
        (frequency_low + frequency_high) / 2,
        anchor_low,
        anchor_high,
    )
    total = center + rounding_error + frequency_radius + rate_radius
    return total.hi, frequency_radius.hi, rate_radius.hi


def _check_enclosures(boxes: deque[_Box]) -> None:
    threshold = (_PI * _Interval.rational(Fraction(17, 9000))).lo
    while boxes:
        box = boxes.popleft()
        rate_low, rate_high, frequency_low, frequency_high, anchor_low, anchor_high, depth = box
        heuristic = _estimate_box(
            (
                float(rate_low),
                float(rate_high),
                float(frequency_low),
                float(frequency_high),
                anchor_low,
                anchor_high,
                depth,
            )
        )
        if heuristic[0] < math.radians(0.34):
            bound, _, _ = _enclose_box(box)
            if bound < threshold:
                continue
        assert depth < 30, box
        if heuristic[1] >= heuristic[2] and frequency_low < frequency_high:
            middle = (frequency_low + frequency_high) / 2
            boxes.append(
                (rate_low, rate_high, frequency_low, middle, anchor_low, anchor_high, depth + 1)
            )
            boxes.append(
                (rate_low, rate_high, middle, frequency_high, anchor_low, anchor_high, depth + 1)
            )
        else:
            assert rate_low < rate_high
            middle = (rate_low + rate_high) / 2
            boxes.append(
                (
                    rate_low,
                    middle,
                    frequency_low,
                    frequency_high,
                    anchor_low,
                    anchor_high,
                    depth + 1,
                )
            )
            boxes.append(
                (
                    middle,
                    rate_high,
                    frequency_low,
                    frequency_high,
                    anchor_low,
                    anchor_high,
                    depth + 1,
                )
            )


def test_fixed_intervals_enclose_exact_arithmetic() -> None:
    rng = random.Random(1601)
    for _ in range(200):
        left = Fraction(rng.randrange(-10_000, 10_000), rng.randrange(1, 10_000))
        right = Fraction(rng.randrange(1, 10_000), rng.randrange(1, 10_000))
        a = _Interval.rational(left)
        b = _Interval.rational(right)
        assert (a + b).contains(left + right)
        assert (a - b).contains(left - right)
        assert (a * b).contains(left * right)
        assert (a / b).contains(left / right)
        assert a.square().contains(left**2)
        lower = min(left, left + right)
        upper = max(left, left + right)
        interval = _Interval.between(lower, upper)
        for value in (lower, (lower + upper) / 2, upper):
            assert (interval * b).contains(value * right)
            assert (interval / b).contains(value / right)
            assert interval.square().contains(value**2)


def test_certified_pi_and_trigonometric_special_points() -> None:
    assert _PI.contains(Fraction("3.141592653589793238462643383279502884197"))
    sine_zero, cosine_zero = _sin_cos(_ZERO)
    assert sine_zero.contains(Fraction(0))
    assert cosine_zero.contains(Fraction(1))
    sine_half, cosine_half = _sin_cos(_PI.divide_integer(2))
    assert sine_half.contains(Fraction(1))
    assert cosine_half.contains(Fraction(0))
    sine_pi, cosine_pi = _sin_cos(_PI)
    assert sine_pi.contains(Fraction(0))
    assert cosine_pi.contains(Fraction(-1))


def test_interval_center_response_matches_an_independent_sample() -> None:
    rate, frequency = Fraction(48_000), Fraction(1000)
    alpha = tuple(_from_bits(bits) for bits in _HILBERT_PROFILE_BITS[48_000])
    expected = math.radians(_quadrature_error_degrees(float(rate), float(frequency), alpha))
    enclosure = _center_error(rate, frequency, 48_000, 64_000)
    assert expected <= enclosure.hi / _SCALE + 1e-14


def test_sine_ranges_include_internal_extrema() -> None:
    square, double = _sine_ranges(Fraction(1, 8), Fraction(3, 8))
    assert square.contains(Fraction(1))
    assert double.contains(Fraction(1))
    assert double.contains(Fraction(-1))


def test_integer_box_bounds_cover_independent_interior_samples() -> None:
    rng = random.Random(718)
    segments = [
        (low, high) for anchors in _HILBERT_RATE_BANDS for low, high in itertools.pairwise(anchors)
    ]
    for _ in range(40):
        low, high = rng.choice(segments)
        rate_low = Fraction(rng.randrange(low, high))
        rate_high = Fraction(rng.randrange(int(rate_low) + 1, high + 1))
        frequency_low = Fraction(rng.randrange(20, 19_000))
        frequency_high = Fraction(rng.randrange(int(frequency_low) + 1, 20_001))
        box = (rate_low, rate_high, frequency_low, frequency_high, low, high, 0)
        bound, _, _ = _enclose_box(box)
        for rate_weight, frequency_weight in (
            (Fraction(0), Fraction(1)),
            (Fraction(1), Fraction(0)),
            (Fraction(1, 3), Fraction(2, 3)),
            (Fraction(1, 2), Fraction(1, 2)),
        ):
            rate = float(rate_low + rate_weight * (rate_high - rate_low))
            frequency = float(frequency_low + frequency_weight * (frequency_high - frequency_low))
            alpha = _coefficients(rate, low, high)
            actual = math.radians(_quadrature_error_degrees(rate, frequency, alpha))
            assert actual <= bound / _SCALE + 1e-10


def test_h1_ideal_response_integer_interval_enclosures() -> None:
    frequencies = (
        20,
        25,
        32,
        40,
        50,
        63,
        80,
        100,
        125,
        160,
        200,
        250,
        315,
        400,
        500,
        630,
        800,
        1000,
        1250,
        1600,
        2000,
        2500,
        3150,
        4000,
        5000,
        6300,
        8000,
        10_000,
        12_500,
        16_000,
        20_000,
    )
    assert frequencies[0] == 20
    assert frequencies[-1] == 20_000
    assert all(low < high for low, high in itertools.pairwise(frequencies))
    boxes: deque[_Box] = deque(
        (Fraction(low), Fraction(high), Fraction(first), Fraction(last), low, high, 0)
        for anchors in _HILBERT_RATE_BANDS
        for low, high in itertools.pairwise(anchors)
        for first, last in itertools.pairwise(frequencies)
    )
    boxes.extend(
        (
            Fraction(48_000),
            Fraction(48_000),
            Fraction(20_000) + Fraction(2050 * index, 16),
            Fraction(20_000) + Fraction(2050 * (index + 1), 16),
            48_000,
            64_000,
            0,
        )
        for index in range(16)
    )
    _check_enclosures(boxes)
