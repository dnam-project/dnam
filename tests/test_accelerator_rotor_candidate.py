"""Exploratory 32-bit rotor candidate; this is not a production backend test."""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache

import numpy as np
import numpy.typing as npt
import pytest

from test_accelerator_pitch_candidate import _as_f32, _round_f32, _two_prod, _two_sum, _ulp_distance
from test_control_math_spec_references import _adaptive_rotor

_MASK = np.int32(0xFFFF)
_TARGET = (0, 0, 0xC000, 0xFFFF, 0xFFFF, 0xFFFF, 0x000F)
_SMALL_CUTOFF = np.float32(2**-12)
_PI_F32 = np.float32(math.pi)
_HALF_PI_F32 = np.float32(math.pi / 2)
_PI_LOW = np.float32(math.pi - float(_PI_F32))
_HALF_PI_LOW = np.float32(math.pi / 2 - float(_HALF_PI_F32))
_PI_TAIL = np.float32(math.pi - float(_PI_F32) - float(_PI_LOW))
_HALF_PI_TAIL = np.float32(math.pi / 2 - float(_HALF_PI_F32) - float(_HALF_PI_LOW))


@lru_cache(maxsize=1)
def _nodes() -> npt.NDArray[np.float32]:
    result: list[list[float]] = [[], [], [], []]
    with localcontext() as context:
        context.prec = 100
        for index in range(202):
            angle = Decimal(index) / 64
            sine, cosine = angle, Decimal(1)
            sine_term, cosine_term = sine, cosine
            square = angle * angle
            for degree in range(1, 36):
                sine_term *= -square / ((2 * degree) * (2 * degree + 1))
                cosine_term *= -square / ((2 * degree - 1) * (2 * degree))
                sine += sine_term
                cosine += cosine_term
            high_cosine = _round_f32(cosine)
            high_sine = _round_f32(sine)
            result[0].append(high_cosine)
            result[1].append(_round_f32(cosine - Decimal.from_float(high_cosine)))
            result[2].append(high_sine)
            result[3].append(_round_f32(sine - Decimal.from_float(high_sine)))
    return np.asarray(result, dtype=np.float32)


def _square_limbs(bits: npt.NDArray[np.int32]) -> list[npt.NDArray[np.int32]]:
    exponent = (bits >> np.int32(23)) & np.int32(0xFF)
    mantissa = (bits & np.int32(0x7F_FFFF)) | np.int32(1 << 23)
    a = mantissa >> np.int32(12)
    b = mantissa & np.int32(0xFFF)
    twice_ab = np.int32(2) * a * b
    low = b * b + ((twice_ab & np.int32(0xFFF)) << np.int32(12))
    high = a * a + (twice_ab >> np.int32(12)) + (low >> np.int32(24))
    low = low & np.int32(0xFF_FFFF)
    pieces = (
        low & _MASK,
        ((low >> np.int32(16)) | ((high & np.int32(0xFF)) << np.int32(8))) & _MASK,
        high >> np.int32(8),
    )
    shift = exponent * np.int32(2) - np.int32(200)
    offset, remainder = shift >> np.int32(4), shift & np.int32(15)
    result = [np.zeros(bits.shape, dtype=np.int32) for _ in range(7)]
    for source in range(3):
        lower = (pieces[source] << remainder) & _MASK
        upper = pieces[source] >> (np.int32(16) - remainder)
        for target in range(7):
            result[target] |= np.where(offset + source == target, lower, np.int32(0))
            result[target] |= np.where(offset + source + 1 == target, upper, np.int32(0))
    return result


def _norm_rounds_to_one_or_more(
    cosine: npt.NDArray[np.float32], sine: npt.NDArray[np.float32]
) -> npt.NDArray[np.bool_]:
    c_bits = np.abs(cosine).view(np.int32)
    s_bits = np.abs(sine).view(np.int32)
    largest, smallest = np.maximum(c_bits, s_bits), np.minimum(c_bits, s_bits)
    obvious = largest >= np.int32(0x3F80_0000)
    active = (largest < np.int32(0x3F80_0000)) & (((smallest >> 23) & 0xFF) >= 100)
    safe_c = np.where(active, c_bits, np.int32(0x3F00_0000))
    safe_s = np.where(active, s_bits, np.int32(0x3F00_0000))
    c_limbs, s_limbs = _square_limbs(safe_c), _square_limbs(safe_s)
    carry = np.zeros(cosine.shape, dtype=np.int32)
    combined = []
    for c_limb, s_limb in zip(c_limbs, s_limbs, strict=True):
        total = c_limb + s_limb + carry
        combined.append(total & _MASK)
        carry = total >> np.int32(16)
    equal = np.ones(cosine.shape, dtype=np.bool_)
    greater = np.zeros(cosine.shape, dtype=np.bool_)
    for index in range(6, -1, -1):
        greater |= equal & (combined[index] > _TARGET[index])
        equal &= combined[index] == _TARGET[index]
    return obvious | (active & (greater | equal))


def _dd_add(
    a_high: npt.NDArray[np.float32],
    a_low: npt.NDArray[np.float32],
    b_high: npt.NDArray[np.float32],
    b_low: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    high, error = _two_sum(a_high, b_high)
    low = _as_f32(_as_f32(error + a_low) + b_low)
    return _two_sum(high, low)


def _dd_mul(
    a_high: npt.NDArray[np.float32],
    a_low: npt.NDArray[np.float32],
    b_high: npt.NDArray[np.float32],
    b_low: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    high, error = _two_prod(a_high, b_high)
    low = _as_f32(error + _as_f32(a_high * b_low) + _as_f32(a_low * b_high))
    return _two_sum(high, low)


def _small_sine_from_split(
    magnitude: npt.NDArray[np.float32],
    center_high: np.float32,
    center_low: np.float32,
    center_tail: np.float32,
) -> npt.NDArray[np.float32]:
    difference = _as_f32(center_high - magnitude)
    high, error = _two_sum(difference, np.full(magnitude.shape, center_low, dtype=np.float32))
    square = _as_f32(high * high)
    correction = _as_f32(-_as_f32(high * square) * np.float32(1 / 6))
    return _as_f32(high + _as_f32(_as_f32(error + center_tail) + correction))


def _rotor_candidate(
    theta: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    if np.any(~np.isfinite(theta)) or np.any(np.abs(theta) >= _PI_F32):
        raise ValueError("invalid rotor angle")
    magnitude = np.abs(theta)
    index = _as_f32(magnitude * np.float32(64)).astype(np.int32)
    local = _as_f32(magnitude - _as_f32(index.astype(np.float32) * np.float32(1 / 64)))
    c_high, c_low, s_high, s_low = (row[index] for row in _nodes())

    with np.errstate(under="ignore"):
        square = _as_f32(local * local)
        correction = _as_f32(
            -_as_f32(local * square) * _as_f32(np.float32(1 / 6) - square * np.float32(1 / 120))
        )
        u_high, u_low = _two_sum(local, correction)
        v = _as_f32(-np.float32(0.5) * square + _as_f32(square * square) * np.float32(1 / 24))
        zero = np.zeros(theta.shape, dtype=np.float32)
        cv_high, cv_low = _dd_mul(c_high, c_low, v, zero)
        su_high, su_low = _dd_mul(s_high, s_low, u_high, u_low)
        cd_high, cd_low = _dd_add(cv_high, cv_low, -su_high, -su_low)
        result_c_high, result_c_low = _dd_add(c_high, c_low, cd_high, cd_low)
        sv_high, sv_low = _dd_mul(s_high, s_low, v, zero)
        cu_high, cu_low = _dd_mul(c_high, c_low, u_high, u_low)
        sd_high, sd_low = _dd_add(sv_high, sv_low, cu_high, cu_low)
        result_s_high, result_s_low = _dd_add(s_high, s_low, sd_high, sd_low)
        cosine = _as_f32(result_c_high + result_c_low)
        sine = _as_f32(result_s_high + result_s_low)
        near_half_pi = np.abs(_as_f32(magnitude - _HALF_PI_F32)) <= np.float32(1 / 64)
        near_pi = magnitude >= _as_f32(_PI_F32 - np.float32(1 / 64))
        cosine = np.where(
            near_half_pi,
            _small_sine_from_split(magnitude, _HALF_PI_F32, _HALF_PI_LOW, _HALF_PI_TAIL),
            cosine,
        )
        sine = np.where(
            near_pi, _small_sine_from_split(magnitude, _PI_F32, _PI_LOW, _PI_TAIL), sine
        )

    small = magnitude <= _SMALL_CUTOFF
    cosine = np.where(small, np.float32(1), cosine)
    sine = np.where(small, magnitude, sine)
    sine = np.where(theta < 0, -sine, sine)
    active = theta != 0
    reduce_cosine = np.abs(cosine) >= np.abs(sine)
    cosine_bits, sine_bits = cosine.view(np.int32), sine.view(np.int32)
    for _ in range(2):
        needs_reduction = active & _norm_rounds_to_one_or_more(cosine, sine)
        cosine_bits[needs_reduction & reduce_cosine] -= np.int32(1)
        sine_bits[needs_reduction & ~reduce_cosine] -= np.int32(1)
    return cosine, np.where(active, sine, np.float32(0))


def test_integer_norm_predicate_matches_exact_and_f64_references() -> None:
    rng = np.random.default_rng(44)
    cosine = rng.uniform(0, 1, 200_000).astype(np.float32)
    sine = np.sqrt(1 - cosine.astype(np.float64) ** 2).astype(np.float32)
    for values in (
        sine,
        np.nextafter(sine, np.float32(0)),
        np.nextafter(sine, np.float32(math.inf)),
    ):
        c, s = cosine.astype(np.float64), values.astype(np.float64)
        assert np.array_equal(_norm_rounds_to_one_or_more(cosine, values), c * c + s * s >= 1)

    cosine = np.asarray(
        [1, 1, np.nextafter(np.float32(1), np.float32(0)), 0, 0.5], dtype=np.float32
    )
    sine = np.asarray([0, 2**-30, 2**-30, 0, 0.5], dtype=np.float32)
    actual = _norm_rounds_to_one_or_more(cosine, sine)
    threshold = Fraction(1) - Fraction(1, 2**54)
    for c, s, selected in zip(cosine, sine, actual, strict=True):
        norm = Fraction.from_float(float(c)) ** 2 + Fraction.from_float(float(s)) ** 2
        assert selected == (norm >= threshold)


def test_candidate_rotor_accuracy_and_contraction() -> None:
    assert _nodes().nbytes == 3_232
    angles = np.linspace(0, np.nextafter(_PI_F32, np.float32(0)), 1_000_001, dtype=np.float32)
    cosine, sine = _rotor_candidate(angles)
    reference_cosine, reference_sine = _adaptive_rotor(angles)
    c, s, theta = cosine.astype(np.float64), sine.astype(np.float64), angles.astype(np.float64)
    assert np.max(np.maximum(np.abs(c - np.cos(theta)), np.abs(s - np.sin(theta)))) < 1e-7
    assert np.max(np.abs(np.arctan2(s, c) - theta)) < 1e-7
    assert np.all((c * c + s * s)[1:] < 1)
    assert np.min((c * c + s * s)[1:]) > 0.9999998
    assert np.max(_ulp_distance(cosine, reference_cosine)) <= 1
    assert np.max(_ulp_distance(sine, reference_sine)) <= 1


def test_candidate_signed_angles_and_boundary_cases() -> None:
    angles = np.asarray(
        [0, 2**-149, 2**-126, 2**-12, math.pi / 4, math.pi / 2, 3 * math.pi / 4],
        dtype=np.float32,
    )
    angles = np.concatenate([angles, [np.nextafter(_PI_F32, np.float32(0))]])
    cosine, sine = _rotor_candidate(angles)
    negative_cosine, negative_sine = _rotor_candidate(-angles)
    assert np.array_equal(cosine, negative_cosine)
    assert np.array_equal(sine, -negative_sine)
    assert (cosine[0], sine[0]) == (1, 0)
    for c, s in zip(cosine[1:], sine[1:], strict=True):
        norm = Fraction.from_float(float(c)) ** 2 + Fraction.from_float(float(s)) ** 2
        assert norm < 1


def test_candidate_adjacent_inputs_at_table_and_quadrant_boundaries() -> None:
    centers = [
        2**-12,
        math.pi / 4,
        math.pi / 2,
        float(_HALF_PI_F32) - 1 / 64,
        float(_HALF_PI_F32) + 1 / 64,
        3 * math.pi / 4,
        float(_PI_F32) - 1 / 64,
        math.pi,
    ]
    centers.extend(index / 64 for index in range(1, 202))
    for center in centers:
        bits = int(np.asarray(center, dtype=np.float32).view(np.uint32))
        angles = np.arange(bits - 64, bits + 65, dtype=np.uint32).view(np.float32)
        angles = angles[angles < _PI_F32]
        for signed_angles in (angles, -angles):
            cosine, sine = _rotor_candidate(signed_angles)
            reference_cosine, reference_sine = _adaptive_rotor(signed_angles)
            assert np.max(_ulp_distance(cosine, reference_cosine)) <= 1
            assert np.max(_ulp_distance(sine, reference_sine)) <= 1
            c, s = cosine.astype(np.float64), sine.astype(np.float64)
            assert np.all(c * c + s * s < 1)


def test_candidate_on_random_binary32_encodings() -> None:
    rng = np.random.default_rng(202)
    maximum = int(np.asarray(_PI_F32).view(np.uint32))
    angles = rng.integers(1, maximum, 400_000, dtype=np.uint32).view(np.float32)
    cosine, sine = _rotor_candidate(angles)
    reference_cosine, reference_sine = _adaptive_rotor(angles)
    assert np.max(_ulp_distance(cosine, reference_cosine)) <= 1
    assert np.max(_ulp_distance(sine, reference_sine)) <= 1
    c, s = cosine.astype(np.float64), sine.astype(np.float64)
    assert np.all(c * c + s * s < 1)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, math.pi, -math.pi])
def test_candidate_rejects_invalid_angle(value: float) -> None:
    with pytest.raises(ValueError, match="invalid rotor"):
        _rotor_candidate(np.asarray([value], dtype=np.float32))
