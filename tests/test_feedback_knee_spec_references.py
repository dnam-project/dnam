"""Independent exact and wide-arithmetic references for the Stage IV knee."""

from __future__ import annotations

import math
import random
import struct
from fractions import Fraction

import pytest


def _f32(value: float) -> float:
    return struct.unpack(">f", struct.pack(">f", value))[0]


def _f32_bits(bits: int) -> float:
    return struct.unpack(">f", struct.pack(">I", bits))[0]


_GAIN_LIMIT = _f32_bits(0x3F7FBE76)


def _ideal(value: float, gain: float, width: float, blend: float) -> Fraction:
    u, g, sigma, alpha = map(Fraction.from_float, (value, gain, width, blend))
    return g * u * ((1 - alpha) + alpha / (1 + (u / sigma) ** 2))


def _wide_intermediate(value: float, gain: float, width: float, blend: float) -> float:
    if not all(math.isfinite(x) for x in (value, gain, width, blend)):
        raise ValueError("coefficients and input must be finite")
    if not (0 <= gain <= _GAIN_LIMIT and 0 <= blend <= 1 and width > 0):
        raise ValueError("coefficients outside the permitted domain")

    inverse_width = 1.0 / width  # Prepared outside the real-time loop.
    ratio = abs(value) * inverse_width
    denominator = 1.0 + ratio * ratio
    bits = struct.unpack(">Q", struct.pack(">d", denominator))[0]
    exponent = ((bits >> 52) & 0x7FF) - 1023
    inverse = struct.unpack(">d", struct.pack(">Q", (1023 - exponent - 1) << 52))[0]
    for _ in range(6):
        inverse *= 2.0 - denominator * inverse
    factor = (1.0 - blend) + blend * inverse
    return (gain * value) * factor


def _wide_candidate(value: float, gain: float, width: float, blend: float) -> float:
    return _f32(_wide_intermediate(value, gain, width, blend))


def test_blend_contains_both_previously_incompatible_knees() -> None:
    value, gain, width = 2.0, _f32(0.8), 0.5
    linear = _ideal(value, gain, width, 0.0)
    rational = _ideal(value, gain, width, 1.0)
    assert linear == Fraction.from_float(gain) * 2
    assert rational == Fraction.from_float(gain) * Fraction(2, 17)
    assert _ideal(value, gain, width, 0.25) == (3 * linear + rational) / 4
    assert _ideal(-value, gain, width, 0.25) == -_ideal(value, gain, width, 0.25)
    assert _ideal(0.0, gain, width, 1.0) == 0


def test_exact_derivative_and_nonexpansive_bounds() -> None:
    for q in map(Fraction, (0, 1, 2, 3, 4, 17, 1000)):
        rational_derivative = (1 - q) / (1 + q) ** 2
        assert rational_derivative + Fraction(1, 8) == (q - 3) ** 2 / (8 * (1 + q) ** 2)
        assert rational_derivative <= 1
        for alpha in (Fraction(0), Fraction(1, 2), Fraction(8, 9), Fraction(1)):
            derivative = 1 - alpha + alpha * rational_derivative
            assert 1 - Fraction(9, 8) * alpha <= derivative <= 1
            assert abs(derivative) <= 1
            if alpha <= Fraction(8, 9):
                assert derivative >= 0
    assert (1 - Fraction(3)) / (1 + Fraction(3)) ** 2 == Fraction(-1, 8)


def test_exact_blend_never_amplifies_a_line_input() -> None:
    for value in (0.0, 2**-149, 0.5, 2.0, -100.0, _f32_bits(0x7F7FFFFF)):
        for width in (_f32_bits(1), 0.5, _f32_bits(0x7F7FFFFF)):
            for blend in (0.0, 0.5, 1.0):
                result = _ideal(value, _GAIN_LIMIT, width, blend)
                assert abs(result) <= Fraction.from_float(_GAIN_LIMIT * abs(value))


def test_positive_term_evaluation_avoids_cancellation() -> None:
    value, width, blend = 1e9, _f32(0.05), 1.0
    denominator = 1 + (value / width) ** 2
    reciprocal = 1 / denominator
    assert 1 - blend * (1 - reciprocal) == 0.0
    assert (1 - blend) + blend * reciprocal > 0.0
    assert _wide_candidate(value, 0.5, width, blend) > 0.0


def test_wide_candidate_covers_extreme_and_random_f32_domains() -> None:
    rng = random.Random(4901)
    widths = [_f32_bits(bits) for bits in (1, 0x007FFFFF, 0x00800000, 0x3F000000, 0x7F7FFFFF)]
    values = [_f32_bits(bits) for bits in (0, 1, 0x007FFFFF, 0x00800000, 0x3F800000, 0x7F7FFFFF)]
    gains = [0.0, _f32_bits(1), 0.5, _GAIN_LIMIT]
    blends = [0.0, _f32_bits(1), 0.5, _f32_bits(0x3F7FFFFF), 1.0]
    cases = [
        (u * sign, g, s, a)
        for u in values
        for g in gains
        for s in widths
        for a in blends
        for sign in (-1, 1)
    ]
    for _ in range(5000):
        value = _f32_bits(rng.randrange(0x7F800000)) * rng.choice((-1, 1))
        width = _f32_bits(rng.randrange(1, 0x7F800000))
        gain = _f32(rng.random() * _GAIN_LIMIT)
        blend = _f32(rng.random())
        cases.append((value, gain, width, blend))

    for value, gain, width, blend in cases:
        exact = _ideal(value, gain, width, blend)
        intermediate = _wide_intermediate(value, gain, width, blend)
        if exact == 0:
            assert intermediate == 0
        else:
            assert abs(Fraction.from_float(intermediate) - exact) < abs(exact) / 2**40
        actual = Fraction.from_float(_f32(intermediate))
        assert abs(actual - exact) <= abs(exact) / 2**23 + Fraction(1, 2**150)


@pytest.mark.parametrize(
    ("value", "gain", "width", "blend"),
    [
        (math.inf, 0.5, 0.5, 0.5),
        (1.0, math.nan, 0.5, 0.5),
        (1.0, -0.1, 0.5, 0.5),
        (1.0, _f32(0.999), 0.5, 0.5),
        (1.0, 0.5, 0.0, 0.5),
        (1.0, 0.5, math.inf, 0.5),
        (1.0, 0.5, 0.5, -0.1),
        (1.0, 0.5, 0.5, 1.1),
    ],
)
def test_invalid_or_unsupported_coefficients_are_rejected(
    value: float, gain: float, width: float, blend: float
) -> None:
    with pytest.raises(ValueError, match=r"finite|permitted"):
        _wide_candidate(value, gain, width, blend)
