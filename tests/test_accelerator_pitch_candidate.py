"""Exploratory 32-bit pitch candidate; this is not a production backend test."""

from __future__ import annotations

import math
import struct
from decimal import Decimal, localcontext
from functools import lru_cache

import numpy as np
import numpy.typing as npt
import pytest

from test_control_math_spec_references import _pitch_profile

_F32 = struct.Struct("<f")
_U32 = struct.Struct("<I")
_SCALE = 16
_LAST_INDEX = 24 * _SCALE
_SMALL_CUTOFF = np.float32(2**-22)
_C3 = np.float32(1 / 6)
_C4 = np.float32(1 / 24)
_C5 = np.float32(1 / 120)
_K = np.float32(math.log(2) / 12)


def _from_bits(bits: int) -> float:
    return _F32.unpack(_U32.pack(bits))[0]


def _to_bits(value: float) -> int:
    return _U32.unpack(_F32.pack(value))[0]


def _as_f32(value: npt.ArrayLike) -> npt.NDArray[np.float32]:
    return np.asarray(value, dtype=np.float32)


def _round_f32(value: Decimal) -> float:
    if not value:
        return 0.0
    sign = -1 if value < 0 else 1
    magnitude = abs(value)
    bits = _to_bits(float(magnitude))
    best = min(
        (candidate for candidate in (bits - 1, bits, bits + 1) if candidate >= 0),
        key=lambda candidate: (
            abs(Decimal.from_float(_from_bits(candidate)) - magnitude),
            candidate & 1,
        ),
    )
    return sign * _from_bits(best)


@lru_cache(maxsize=1)
def _tables() -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], np.float32, np.float32]:
    with localcontext() as context:
        context.prec = 100
        h = Decimal(2).ln() / Decimal(12 * _SCALE)
        highs: list[list[float]] = []
        lows: list[list[float]] = []
        for sign in (-1, 1):
            high_row, low_row = [], []
            for index in range(_LAST_INDEX + 1):
                node = (h * sign * index).exp()
                high = _round_f32(node)
                low = _round_f32(node - Decimal.from_float(high))
                high_row.append(high)
                low_row.append(low)
            highs.append(high_row)
            lows.append(low_row)
        high_h = _round_f32(h)
        low_h = _round_f32(h - Decimal.from_float(high_h))
    return (
        np.asarray(highs, dtype=np.float32),
        np.asarray(lows, dtype=np.float32),
        np.float32(high_h),
        np.float32(low_h),
    )


def _two_sum(
    a: npt.NDArray[np.float32] | np.float32, b: npt.NDArray[np.float32]
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    total = _as_f32(a + b)
    other = _as_f32(total - a)
    error = _as_f32(_as_f32(a - _as_f32(total - other)) + _as_f32(b - other))
    return total, error


def _two_prod(
    a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    product = _as_f32(a * b)
    split_a = _as_f32(np.float32(4097) * a)
    split_b = _as_f32(np.float32(4097) * b)
    high_a = _as_f32(split_a - _as_f32(split_a - a))
    high_b = _as_f32(split_b - _as_f32(split_b - b))
    low_a = _as_f32(a - high_a)
    low_b = _as_f32(b - high_b)
    error = _as_f32(
        _as_f32(
            _as_f32(_as_f32(high_a * high_b) - product)
            + _as_f32(high_a * low_b)
            + _as_f32(low_a * high_b)
        )
        + _as_f32(low_a * low_b)
    )
    return product, error


def _pitch_candidate(
    semitones: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    if np.any(~np.isfinite(semitones)) or np.any(np.abs(semitones) > 24):
        raise ValueError("invalid pitch interval")
    highs, lows, high_h, low_h = _tables()
    magnitude = np.abs(semitones)
    scaled = _as_f32(magnitude * np.float32(_SCALE))
    index = np.minimum(scaled.astype(np.int32), _LAST_INDEX - 1)
    t = _as_f32(scaled - index.astype(np.float32))
    sign_index = (semitones >= 0).astype(np.int32)
    high, low = highs[sign_index, index], lows[sign_index, index]
    h_high = np.where(semitones >= 0, high_h, -high_h)
    h_low = np.where(semitones >= 0, low_h, -low_h)

    with np.errstate(under="ignore"):
        x_high, x_error = _two_prod(t, h_high)
        x_low = _as_f32(x_error + _as_f32(t * h_low))
        polynomial = _as_f32(
            np.float32(0.5) + x_high * _as_f32(_C3 + x_high * _as_f32(_C4 + x_high * _C5))
        )
        correction = _as_f32(_as_f32(x_high * x_high) * polynomial)
        exp_high, exp_error = _two_sum(x_high, correction)
        exp_low = _as_f32(exp_error + x_low + _as_f32(x_high * x_low))
        delta_high, delta_error = _two_prod(high, exp_high)
        delta_low = _as_f32(delta_error + _as_f32(high * exp_low) + _as_f32(low * exp_high))
        ratio_high, ratio_error = _two_sum(high, delta_high)
        ratio = _as_f32(ratio_high + _as_f32(_as_f32(ratio_error + low) + delta_low))
        warp_zero, warp_zero_error = _two_sum(np.float32(1), -high)
        warp_high, warp_error = _two_sum(warp_zero, -delta_high)
        warp = _as_f32(
            warp_high + _as_f32(_as_f32(warp_zero_error - low) + _as_f32(warp_error - delta_low))
        )
        small = magnitude <= _SMALL_CUTOFF
        ratio = np.where(small, np.float32(1), ratio)
        warp = np.where(small, -_as_f32(semitones * _K), warp)
    return ratio, np.where(semitones == 0, np.float32(0), warp)


def _ulp_distance(
    actual: npt.NDArray[np.float32], expected: npt.NDArray[np.float32]
) -> npt.NDArray[np.int64]:
    diff = np.abs(actual.view(np.int32).astype(np.int64) - expected.view(np.int32).astype(np.int64))
    return np.where(actual == expected, np.int64(0), diff)


def test_candidate_table_size_and_pitch_accuracy() -> None:
    highs, lows, _, _ = _tables()
    assert highs.nbytes + lows.nbytes == 6_160
    semitones = np.linspace(-24, 24, 2_000_001, dtype=np.float32)
    ratio, warp = _pitch_candidate(semitones)
    exact = semitones.astype(np.float64) * (math.log(2) / 12)
    assert np.max(np.abs(ratio.astype(np.float64) / np.exp(exact) - 1)) < 6.2e-8
    assert np.max(np.abs(warp.astype(np.float64) + np.expm1(exact))) < 1.2e-7
    assert np.all(ratio[1:] >= ratio[:-1])
    assert np.all(warp[1:] <= warp[:-1])
    reference_ratio, reference_warp = _pitch_profile(semitones)
    assert np.max(_ulp_distance(ratio, reference_ratio)) <= 1
    assert np.max(_ulp_distance(warp, reference_warp)) <= 1


def test_candidate_grid_and_small_branch_boundaries() -> None:
    centers = [2**-22, *[index / _SCALE for index in range(1, _LAST_INDEX + 1)]]
    for center in centers:
        bits = _to_bits(center)
        adjacent = np.arange(max(bits - 64, 1), min(bits + 65, _to_bits(24)), dtype=np.uint32).view(
            np.float32
        )
        for sign in (-1, 1):
            pitch = _as_f32(sign * adjacent)
            ratio, warp = _pitch_candidate(pitch)
            reference_ratio, reference_warp = _pitch_profile(pitch)
            assert np.all(sign * np.diff(ratio.astype(np.float64)) >= 0)
            assert np.all(sign * np.diff(warp.astype(np.float64)) <= 0)
            assert np.max(_ulp_distance(ratio, reference_ratio)) <= 1
            assert np.max(_ulp_distance(warp, reference_warp)) <= 1

    points = np.asarray([-24, -12, -0.0, 0.0, 12, 24, 1e-8], dtype=np.float32)
    ratio, warp = _pitch_candidate(points)
    np.testing.assert_array_equal(ratio[:6], np.asarray([0.25, 0.5, 1, 1, 2, 4], dtype=np.float32))
    np.testing.assert_array_equal(warp[:6], np.asarray([0.75, 0.5, 0, 0, -1, -3], dtype=np.float32))
    assert not np.signbit(warp[2])
    assert warp[6] != 0


def test_candidate_subnormal_warp_is_monotone_within_one_ulp() -> None:
    previous = [0.0, 0.0]
    for start in range(1, 1 << 23, 1 << 18):
        bits = np.arange(start, min(start + (1 << 18), 1 << 23), dtype=np.uint32)
        values = bits.view(np.float32)
        for index, sign in enumerate((1, -1)):
            pitch = _as_f32(sign * values)
            ratio, warp = _pitch_candidate(pitch)
            reference_ratio, reference_warp = _pitch_profile(pitch)
            assert np.all(ratio == 1)
            assert np.all(reference_ratio == 1)
            assert sign * float(warp[0] - previous[index]) <= 0
            assert np.all(sign * np.diff(warp.astype(np.float64)) <= 0)
            assert np.max(_ulp_distance(warp, reference_warp)) <= 1
            previous[index] = float(warp[-1])


def test_candidate_adjacent_random_inputs() -> None:
    rng = np.random.default_rng(42)
    for exponent in (103, 111, 118, 124, 126, 127, 128, 129, 130, 131):
        base = exponent << 23
        bits = rng.integers(base, base + (1 << 23) - 1, 50_000, dtype=np.uint32)
        first = bits.view(np.float32)
        second = (bits + np.uint32(1)).view(np.float32)
        valid = second <= 24
        for sign in (-1, 1):
            low = _as_f32(sign * first[valid])
            high = _as_f32(sign * second[valid])
            r0, w0 = _pitch_candidate(low)
            r1, w1 = _pitch_candidate(high)
            ref_ratio, ref_warp = _pitch_profile(low)
            assert np.all(sign * (r1.astype(np.float64) - r0) >= 0)
            assert np.all(sign * (w1.astype(np.float64) - w0) <= 0)
            assert np.max(_ulp_distance(r0, ref_ratio)) <= 1
            assert np.max(_ulp_distance(w0, ref_warp)) <= 1


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -25.0, 25.0])
def test_candidate_rejects_invalid_pitch(value: float) -> None:
    with pytest.raises(ValueError, match="invalid pitch"):
        _pitch_candidate(np.asarray([value], dtype=np.float32))
