"""References for slow LFO motion and an unselected fixed-point phase."""

from __future__ import annotations

import math
import random
import struct
from fractions import Fraction

import numpy as np

from test_control_math_spec_references import _adaptive_rotor

_INV_TAU_256 = int("28be60db9391054a7f09d5f47d4d377036d8a5664f10e4107f9458eaf7aef158", 16)
_PHASE_MASK = (1 << 192) - 1


def _float_from_bits(bits: int) -> float:
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def _arctan_bounds(denominator: int) -> tuple[Fraction, Fraction]:
    terms = 80
    lower = sum(
        (
            Fraction((-1) ** index, (2 * index + 1) * denominator ** (2 * index + 1))
            for index in range(terms)
        ),
        Fraction(),
    )
    return lower, lower + Fraction(1, (2 * terms + 1) * denominator ** (2 * terms + 1))


def _pi_bounds() -> tuple[Fraction, Fraction]:
    low5, high5 = _arctan_bounds(5)
    low239, high239 = _arctan_bounds(239)
    return 16 * low5 - 4 * high239, 16 * high5 - 4 * low239


def _phase_increment(bits: int) -> int:
    sign = -1 if bits >> 31 else 1
    magnitude = bits & 0x7FFF_FFFF
    if magnitude > 0x4049_0FDA:
        raise ValueError("angle outside the signed rotor domain")
    exponent = magnitude >> 23
    mantissa = magnitude & 0x7F_FFFF
    if exponent:
        mantissa |= 1 << 23
        power = exponent - 150
    else:
        power = -149
    value = mantissa * _INV_TAU_256
    shift = 64 - power
    whole, remainder = divmod(value, 1 << shift)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and whole & 1):
        whole += 1
    return sign * whole


def _phase_phasor(phase: int) -> tuple[float, float]:
    q32 = (phase & _PHASE_MASK) >> 160
    quadrant = q32 >> 30
    remainder = q32 & 0x3FFF_FFFF
    angle = np.float32(remainder) * np.float32(_float_from_bits(0x30C9_0FDB))
    cosine, sine = _adaptive_rotor(np.asarray([angle], dtype=np.float32))
    c, s = float(cosine[0]), float(sine[0])
    return ((c, s), (-s, c), (-c, -s), (s, -c))[quadrant]


def test_slow_lfo_can_stall_with_the_candidate_f32_state_update() -> None:
    theta = np.float32(2**-30)
    cosine, sine = _adaptive_rotor(np.asarray([theta], dtype=np.float32))
    assert cosine[0] == np.nextafter(np.float32(1), np.float32(0))
    assert sine[0] == theta
    real = np.asarray([_float_from_bits(0x3F35_04F3)], dtype=np.float32)
    imaginary = real.copy()
    next_real = cosine * real - sine * imaginary
    next_imaginary = sine * real + cosine * imaginary
    correction = np.float32(1.5) - np.float32(0.5) * (
        next_real * next_real + next_imaginary * next_imaginary
    )
    assert (next_real * correction)[0] == real[0]
    assert (next_imaginary * correction)[0] == imaginary[0]
    assert 0.000279 < float(theta) * 300_000 < 0.000280


def test_fixed_point_phase_increment_matches_independent_pi_interval() -> None:
    pi_low, pi_high = _pi_bounds()
    k_low = Fraction(1 << 256, 1) / (2 * pi_high)
    k_high = Fraction(1 << 256, 1) / (2 * pi_low)
    assert k_low.numerator // k_low.denominator == _INV_TAU_256
    assert k_high.numerator // k_high.denominator == _INV_TAU_256

    rng = random.Random(0x1F0)
    examples = [0, 1, 0x007F_FFFF, 0x0080_0000, 0x3F80_0000, 0x4049_0FDA]
    examples.extend(rng.randrange(0x4049_0FDB) for _ in range(10_000))
    for bits in examples:
        theta = Fraction.from_float(_float_from_bits(bits))
        low = theta * (1 << 192) / (2 * pi_high)
        high = theta * (1 << 192) / (2 * pi_low)
        step = _phase_increment(bits)
        assert Fraction(step - 1) < low
        assert high < Fraction(step + 1)
        assert _phase_increment(bits | 0x8000_0000) == -step
        if bits:
            assert step >= 1 << 40


def test_fixed_point_phase_retains_the_slow_motion_and_zero_hold() -> None:
    theta = np.float32(2**-30)
    bits = struct.unpack(">I", struct.pack(">f", theta))[0]
    step = _phase_increment(bits)
    initial = 1 << 189
    phase = initial
    for _ in range(300_000):
        phase = (phase + step) & _PHASE_MASK
    assert phase == (initial + 300_000 * step) & _PHASE_MASK
    cosine, sine = _phase_phasor(phase)
    actual = math.atan2(sine, cosine)
    expected = math.pi / 4 + float(theta) * 300_000
    assert abs(actual - expected) < 4e-7
    assert (phase + _phase_increment(0)) & _PHASE_MASK == phase
    assert (phase + step - step) & _PHASE_MASK == phase
