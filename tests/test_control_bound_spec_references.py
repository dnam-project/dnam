"""Representable physical-bound examples, not production control-bus tests."""

from __future__ import annotations

import math
import random
import struct
from fractions import Fraction

import pytest

_F32 = struct.Struct(">f")
_U32 = struct.Struct(">I")
_NEGATIVE_FIRST = 0x0080_0000
_NEGATIVE_NORMAL_END = 0x7F7F_FFFF
_NEGATIVE_ZERO = 0x7FFF_FFFF
_POSITIVE_ZERO = 0x8000_0000
_POSITIVE_NORMAL_START = 0x8080_0000
_NORMALS_PER_SIGN = 254 * 2**23


def _from_bits(bits: int) -> float:
    return _F32.unpack(_U32.pack(bits))[0]


def _bits(value: float) -> int:
    return _U32.unpack(_F32.pack(value))[0]


def _count(*, flush_subnormals: bool) -> int:
    if flush_subnormals:
        return 2 * _NORMALS_PER_SIGN + 1
    return 2 * 255 * 2**23 - 1


def _value(rank: int, *, flush_subnormals: bool) -> float:
    if not 0 <= rank < _count(flush_subnormals=flush_subnormals):
        raise ValueError("invalid rank")
    if flush_subnormals:
        if rank < _NORMALS_PER_SIGN:
            key = _NEGATIVE_FIRST + rank
        elif rank == _NORMALS_PER_SIGN:
            key = _POSITIVE_ZERO
        else:
            key = _POSITIVE_NORMAL_START + rank - _NORMALS_PER_SIGN - 1
    else:
        key = _NEGATIVE_FIRST + rank
        if key >= _NEGATIVE_ZERO:
            key += 1
    bits = key ^ _POSITIVE_ZERO if key & _POSITIVE_ZERO else ~key & 0xFFFF_FFFF
    return _from_bits(bits)


def _rank(value: float, *, flush_subnormals: bool) -> int | None:
    if not math.isfinite(value):
        raise ValueError("nonfinite coordinate")
    try:
        bits = _bits(value)
    except OverflowError as error:
        raise ValueError("coordinate is not f32") from error
    if _from_bits(bits) != value:
        raise ValueError("coordinate is not f32")
    magnitude = bits & 0x7FFF_FFFF
    if magnitude == 0:
        key = _POSITIVE_ZERO
    else:
        if flush_subnormals and magnitude < 0x0080_0000:
            return None
        key = ~bits & 0xFFFF_FFFF if bits & _POSITIVE_ZERO else bits ^ _POSITIVE_ZERO
    if flush_subnormals:
        if key <= _NEGATIVE_NORMAL_END:
            return key - _NEGATIVE_FIRST
        if key == _POSITIVE_ZERO:
            return _NORMALS_PER_SIGN
        return _NORMALS_PER_SIGN + 1 + key - _POSITIVE_NORMAL_START
    return key - _NEGATIVE_FIRST - int(key >= _POSITIVE_ZERO)


def _boundary(value: float, *, flush_subnormals: bool, upper: bool) -> int:
    if not math.isfinite(value):
        raise ValueError("nonfinite bound")
    target = Fraction.from_float(value)
    low, high = 0, _count(flush_subnormals=flush_subnormals)
    while low < high:
        middle = low + (high - low) // 2
        current = Fraction.from_float(_value(middle, flush_subnormals=flush_subnormals))
        if current < target or (upper and current == target):
            low = middle + 1
        else:
            high = middle
    return low


def _intersection(
    lower: tuple[float, ...], upper: tuple[float, ...], *, flush_subnormals: bool
) -> tuple[int, int]:
    count = _count(flush_subnormals=flush_subnormals)
    first = max(
        (_boundary(value, flush_subnormals=flush_subnormals, upper=False) for value in lower),
        default=0,
    )
    end = min(
        (_boundary(value, flush_subnormals=flush_subnormals, upper=True) for value in upper),
        default=count,
    )
    if first >= end:
        raise ValueError("empty representable interval")
    return first, end


def _nearest(requested: float, first: int, end: int, *, flush_subnormals: bool) -> float:
    if not math.isfinite(requested):
        raise ValueError("invalid f32 request")
    try:
        bits = _bits(requested)
    except OverflowError as error:
        raise ValueError("invalid f32 request") from error
    if _from_bits(bits) != requested:
        raise ValueError("invalid f32 request")
    if not 0 <= first < end <= _count(flush_subnormals=flush_subnormals):
        raise ValueError("invalid interval")
    at = _boundary(requested, flush_subnormals=flush_subnormals, upper=False)
    indices = {max(first, min(end - 1, index)) for index in (at - 1, at)}
    target = Fraction.from_float(requested)
    values = [_value(index, flush_subnormals=flush_subnormals) for index in indices]
    return min(
        values,
        key=lambda value: (
            abs(Fraction.from_float(value) - target),
            abs(Fraction.from_float(value)),
            value,
        ),
    )


def test_f32_coordinate_counts_and_rank_boundaries() -> None:
    assert _count(flush_subnormals=False) == 4_278_190_079
    assert _count(flush_subnormals=True) == 4_261_412_865
    for flush in (False, True):
        count = _count(flush_subnormals=flush)
        assert _bits(_value(0, flush_subnormals=flush)) == 0xFF7F_FFFF
        assert _bits(_value(count - 1, flush_subnormals=flush)) == 0x7F7F_FFFF
        zero_rank = _rank(0.0, flush_subnormals=flush)
        assert zero_rank is not None
        assert _bits(_value(zero_rank, flush_subnormals=flush)) == 0
        assert _rank(-0.0, flush_subnormals=flush) == zero_rank
        assert _value(zero_rank - 1, flush_subnormals=flush) < 0
        assert _value(zero_rank + 1, flush_subnormals=flush) > 0


def test_rank_round_trips_full_exponent_samples() -> None:
    rng = random.Random(0xB01D)
    bits_to_check = [0, 0x8000_0000, 1, 0x8000_0001, 0x0080_0000, 0xFF7F_FFFF]
    bits_to_check.extend(rng.getrandbits(32) for _ in range(4000))
    for bits in bits_to_check:
        coordinate = _from_bits(bits)
        if not math.isfinite(coordinate):
            continue
        for flush in (False, True):
            rank = _rank(coordinate, flush_subnormals=flush)
            magnitude = bits & 0x7FFF_FFFF
            if flush and 0 < magnitude < 0x0080_0000:
                assert rank is None
            else:
                assert rank is not None
                assert _value(rank, flush_subnormals=flush) == coordinate


def test_binary64_bounds_are_rounded_inward_and_singletons_may_be_empty() -> None:
    with pytest.raises(ValueError, match="empty representable"):
        _intersection((0.1,), (0.1,), flush_subnormals=False)
    first, end = _intersection((0.1,), (0.2,), flush_subnormals=False)
    lower = _value(first, flush_subnormals=False)
    upper = _value(end - 1, flush_subnormals=False)
    assert lower >= 0.1
    assert upper <= 0.2
    assert _value(first - 1, flush_subnormals=False) < 0.1
    assert _value(end, flush_subnormals=False) > 0.2


def test_rank_intersections_match_all_signed_conditional_bounds() -> None:
    lower = (-10.0, 0.1, -2.0)
    upper = (10.0, 0.2, 1.0)
    first, end = _intersection(lower, upper, flush_subnormals=False)
    assert first == _boundary(max(lower), flush_subnormals=False, upper=False)
    assert end == _boundary(min(upper), flush_subnormals=False, upper=True)
    with pytest.raises(ValueError, match="empty representable"):
        _intersection((0.2,), (0.1,), flush_subnormals=False)


def test_extreme_binary64_bounds_do_not_overflow_f32_ranks() -> None:
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    max_f32 = _from_bits(0x7F7F_FFFF)
    assert _intersection((-maximum,), (maximum,), flush_subnormals=False) == (
        0,
        _count(flush_subnormals=False),
    )
    with pytest.raises(ValueError, match="empty representable"):
        _intersection((2 * max_f32,), (), flush_subnormals=False)
    with pytest.raises(ValueError, match="empty representable"):
        _intersection((), (-2 * max_f32,), flush_subnormals=False)


def test_flushing_nearest_request_and_halfway_ties_are_explicit() -> None:
    minimum_normal = _from_bits(0x0080_0000)
    halfway = _from_bits(0x0040_0000)
    above_halfway = _from_bits(0x0040_0001)
    first, end = _intersection((-minimum_normal,), (minimum_normal,), flush_subnormals=True)
    assert _nearest(halfway, first, end, flush_subnormals=True) == 0
    assert _nearest(-halfway, first, end, flush_subnormals=True) == 0
    assert _nearest(above_halfway, first, end, flush_subnormals=True) == minimum_normal
    assert _nearest(-above_halfway, first, end, flush_subnormals=True) == -minimum_normal
    first, end = _intersection((halfway,), (minimum_normal,), flush_subnormals=True)
    assert _nearest(halfway, first, end, flush_subnormals=True) == minimum_normal
    with pytest.raises(ValueError, match="empty representable"):
        _intersection((_from_bits(1),), (_from_bits(0x007F_FFFF),), flush_subnormals=True)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_bounds_are_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="nonfinite bound"):
        _boundary(value, flush_subnormals=False, upper=False)


@pytest.mark.parametrize("value", [0.1, float.fromhex("0x1.fffffffffffffp+1023")])
def test_non_f32_requests_are_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="not f32"):
        _rank(value, flush_subnormals=False)
    with pytest.raises(ValueError, match="invalid f32 request"):
        _nearest(value, 0, _count(flush_subnormals=False), flush_subnormals=False)
