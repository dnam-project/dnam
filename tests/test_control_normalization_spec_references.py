"""Exact declared-range normalization examples, not production backend tests."""

from __future__ import annotations

import math
import random
import struct
from fractions import Fraction

import pytest

from test_tempo_spec_references import _coordinate_at, _coordinate_count

_F32 = struct.Struct(">f")
_U32 = struct.Struct(">I")


def _from_bits(bits: int) -> float:
    return _F32.unpack(_U32.pack(bits))[0]


def _bits(value: float) -> int:
    return _U32.unpack(_F32.pack(value))[0]


def _f32(value: float) -> float:
    try:
        return _F32.unpack(_F32.pack(value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def _ideal(low: float, high: float, value: float) -> Fraction:
    if not all(math.isfinite(number) for number in (low, high, value)):
        raise ValueError("nonfinite coordinate")
    if not low < high or not low <= value <= high:
        raise ValueError("invalid physical interval or coordinate")
    left, right, x = map(Fraction.from_float, (low, high, value))
    return 2 * (x - left) / (right - left) - 1


def _correctly_rounded(value: Fraction) -> float:
    if not -1 <= value <= 1:
        raise ValueError("invalid normalized coordinate")
    count = _coordinate_count(False)
    low, high = 0, count
    while low < high:
        middle = (low + high) // 2
        if Fraction.from_float(_coordinate_at(middle, False)) < value:
            low = middle + 1
        else:
            high = middle
    candidates = {
        _coordinate_at(max(0, low - 1), False),
        _coordinate_at(min(count - 1, low), False),
    }
    return min(candidates, key=lambda x: (abs(Fraction.from_float(x) - value), _bits(x) & 1))


def _legacy_scale_offset(low: float, high: float, value: float) -> float:
    span = _f32(high - low)
    scale = _f32(2.0 / span)
    offset = _f32(-1.0 - _f32(low * scale))
    return max(-1.0, min(1.0, _f32(_f32(value * scale) + offset)))


def _difference_candidate(low: float, high: float, value: float) -> float:
    if not all(math.isfinite(number) and _f32(number) == number for number in (low, high, value)):
        raise ValueError("not finite f32")
    if not low < high or not low <= value <= high:
        raise ValueError("invalid physical interval or coordinate")
    span = high - low
    reciprocal = 1.0 / span
    distance = value - low
    product = distance * reciprocal
    coordinate = 2.0 * product - 1.0
    return _f32(max(-1.0, min(1.0, coordinate)))


def test_f32_scale_offset_loses_a_valid_endpoint_even_with_fused_last_step() -> None:
    low = float(2**30)
    high = low + 128.0
    assert _bits(high) == _bits(low) + 1
    assert _ideal(low, high, low) == -1
    assert _ideal(low, high, high) == 1
    assert _legacy_scale_offset(low, high, low) == 0
    assert _legacy_scale_offset(low, high, high) == 1
    scale = Fraction(1, 64)
    rounded_offset = Fraction(-(2**24))
    assert Fraction.from_float(low) * scale + rounded_offset == 0
    assert _difference_candidate(low, high, low) == -1
    assert _difference_candidate(low, high, high) == 1


def test_extreme_f32_span_overflows_old_form_but_not_scoped_candidate() -> None:
    maximum = _from_bits(0x7F7F_FFFF)
    assert math.isinf(_f32(maximum - -maximum))
    assert _legacy_scale_offset(-maximum, maximum, maximum) == -1
    assert _difference_candidate(-maximum, maximum, -maximum) == -1
    assert _difference_candidate(-maximum, maximum, 0.0) == 0
    assert _difference_candidate(-maximum, maximum, maximum) == 1


def test_exact_reference_handles_finite_binary64_span_overflow() -> None:
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    assert math.isinf(maximum - -maximum)
    assert _ideal(-maximum, maximum, 0.0) == 0
    assert _correctly_rounded(_ideal(-maximum, maximum, 0.0)) == 0.0


def test_reference_rounds_f32_subnormal_midpoints_to_even() -> None:
    half_subnormal = math.ldexp(1.0, -150)
    assert _bits(_correctly_rounded(_ideal(-1.0, 1.0, half_subnormal))) == 0
    assert _bits(_correctly_rounded(_ideal(-1.0, 1.0, 3 * half_subnormal))) == 2
    assert _bits(_correctly_rounded(_ideal(-1.0, 1.0, -3 * half_subnormal))) == 0x8000_0002


def test_scoped_candidate_can_lose_nonzero_subnormal_near_zero() -> None:
    smallest = _from_bits(1)
    exact = _ideal(-1.0, 1.0, smallest)
    assert _bits(_correctly_rounded(exact)) == 1
    assert _difference_candidate(-1.0, 1.0, smallest) == 0


def test_candidate_absolute_bound_on_reproducible_full_exponent_cases() -> None:
    rng = random.Random(0xC012)
    bound = Fraction(1, 2**25) + Fraction(1, 2**48)
    for _ in range(4000):
        samples: list[float] = []
        while len(samples) < 3:
            value = _from_bits(rng.getrandbits(32))
            if math.isfinite(value):
                samples.append(value)
        low, middle, high = sorted(samples)
        if low == high:
            continue
        actual = [_difference_candidate(low, high, value) for value in (low, middle, high)]
        assert actual == sorted(actual)
        assert actual[0] == -1
        assert actual[-1] == 1
        for value, normalized in zip((low, middle, high), actual, strict=True):
            assert abs(Fraction.from_float(normalized) - _ideal(low, high, value)) < bound


@pytest.mark.parametrize(
    ("low", "high", "value"),
    [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 2.0),
        (0.0, math.inf, 0.0),
        (0.0, 1.0, math.nan),
    ],
)
def test_reference_rejects_invalid_coordinates(low: float, high: float, value: float) -> None:
    with pytest.raises(ValueError, match=r"coordinate|interval"):
        _ideal(low, high, value)
