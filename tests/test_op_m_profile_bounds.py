"""Sensitivity bounds for an experimental Op-M Hilbert profile.

The enclosures assume ordinary IEEE-754 binary64 arithmetic and conservative
libm error margins. They are not a directed-rounding proof or a Rust test.
"""

from __future__ import annotations

import cmath
import itertools
import math
import random
from collections import deque

from test_op_m_spec_references import (
    _HILBERT_PROFILE_BITS,
    _coefficients,
    _from_bits,
    _quadrature_error_degrees,
)

type _Interval = tuple[float, float]
type _Box = tuple[float, float, float, float, int, int, int]


def _multiply(left: _Interval, right: _Interval) -> _Interval:
    products = [a * b for a in left for b in right]
    return min(products), max(products)


def _sin_squared_bounds(low: float, high: float) -> _Interval:
    values = [math.sin(low) ** 2, math.sin(high) ** 2]
    if low <= math.pi / 2 <= high:
        values.append(1.0)
    return max(0.0, min(values) - 1e-14), min(1.0, max(values) + 1e-14)


def _sin_twice_bounds(low: float, high: float) -> _Interval:
    values = [math.sin(2 * low), math.sin(2 * high)]
    if low <= math.pi / 4 <= high:
        values.append(1.0)
    if low <= 3 * math.pi / 4 <= high:
        values.append(-1.0)
    return max(-1.0, min(values) - 1e-14), min(1.0, max(values) + 1e-14)


def _derivative_bounds(
    alpha_low: float, alpha_high: float, omega_low: float, omega_high: float
) -> tuple[_Interval, _Interval]:
    # For one all-pass section, d(phi)/d(omega) = -2(1-a^2)/D and
    # d(phi)/d(a) = -2 sin(2 omega)/D, with D = (1-a)^2 + 4a sin^2(omega).
    sin_squared = _sin_squared_bounds(omega_low, omega_high)
    denominator_low = (1 - alpha_high) ** 2 + 4 * alpha_low * sin_squared[0]
    denominator_high = (1 - alpha_low) ** 2 + 4 * alpha_high * sin_squared[1]
    denominator_low *= 1 - 1e-12
    denominator_high *= 1 + 1e-12
    numerator_low = max(0.0, 1 - alpha_high**2 - 1e-14)
    numerator_high = 1 - alpha_low**2 + 1e-14
    phase_omega = (-2 * numerator_high / denominator_low, -2 * numerator_low / denominator_high)
    sine = _sin_twice_bounds(omega_low, omega_high)
    phase_alpha = [
        -2 * value / denominator
        for value in sine
        for denominator in (denominator_low, denominator_high)
    ]
    return phase_omega, (min(phase_alpha), max(phase_alpha))


def _estimate_box(box: _Box) -> tuple[float, float, float]:
    rate_low, rate_high, freq_low, freq_high, anchor_low, anchor_high, _ = box
    omega_low = 2 * math.pi * freq_low / rate_high
    omega_high = 2 * math.pi * freq_high / rate_low
    assert 0 < omega_low <= omega_high < math.pi
    center_rate = 0.5 * (rate_low + rate_high)
    center_freq = 0.5 * (freq_low + freq_high)
    reference = _HILBERT_PROFILE_BITS[anchor_low]
    target = _HILBERT_PROFILE_BITS[anchor_high]
    base = [_from_bits(bits) for bits in reference]
    slopes = [
        (_from_bits(right) - left) / (anchor_high - anchor_low)
        for left, right in zip(base, target, strict=True)
    ]
    center_alpha = []
    phase_omega = (-1.0, -1.0)
    rate_coefficient = (0.0, 0.0)
    rounding_error = 0.0
    # Bound one binary32 rounding after binary64 interpolation.
    coefficient_rounding = 2**-25 + 1e-15
    for index, (coefficient, slope) in enumerate(zip(base, slopes, strict=True)):
        at_low = coefficient + (rate_low - anchor_low) * slope
        at_high = coefficient + (rate_high - anchor_low) * slope
        alpha_low = min(at_low, at_high) - coefficient_rounding
        alpha_high = max(at_low, at_high) + coefficient_rounding
        assert 0 < alpha_low <= alpha_high < 1
        omega_bounds, alpha_bounds = _derivative_bounds(
            alpha_low, alpha_high, omega_low, omega_high
        )
        center_alpha.append(coefficient + (center_rate - anchor_low) * slope)
        sign = -1 if index < 6 else 1
        signed_omega = _multiply(omega_bounds, (sign, sign))
        signed_rate = _multiply(alpha_bounds, (sign * slope, sign * slope))
        phase_omega = (phase_omega[0] + signed_omega[0], phase_omega[1] + signed_omega[1])
        rate_coefficient = (
            rate_coefficient[0] + signed_rate[0],
            rate_coefficient[1] + signed_rate[1],
        )
        rounding_error += max(abs(value) for value in alpha_bounds) * coefficient_rounding
    phase_freq = _multiply(phase_omega, (2 * math.pi / rate_high, 2 * math.pi / rate_low))
    phase_rate = _multiply(phase_omega, (-omega_high / rate_low, -omega_low / rate_high))
    phase_rate = (phase_rate[0] + rate_coefficient[0], phase_rate[1] + rate_coefficient[1])
    frequency_radius = max(abs(value) for value in phase_freq) * (freq_high - freq_low) / 2
    rate_radius = max(abs(value) for value in phase_rate) * (rate_high - rate_low) / 2
    center_error = math.radians(
        _quadrature_error_degrees(center_rate, center_freq, tuple(center_alpha))
    )
    bound = center_error + rounding_error + frequency_radius + rate_radius + 1e-10
    return bound, frequency_radius, rate_radius


def _check_boxes(boxes: deque[_Box]) -> None:
    while boxes:
        box = boxes.popleft()
        rate_low, rate_high, freq_low, freq_high, anchor_low, anchor_high, depth = box
        bound, frequency_radius, rate_radius = _estimate_box(box)
        if bound < math.radians(0.34):
            continue
        assert depth < 24, box
        if frequency_radius >= rate_radius:
            middle = (freq_low + freq_high) / 2
            boxes.append(
                (rate_low, rate_high, freq_low, middle, anchor_low, anchor_high, depth + 1)
            )
            boxes.append(
                (rate_low, rate_high, middle, freq_high, anchor_low, anchor_high, depth + 1)
            )
        else:
            middle = (rate_low + rate_high) / 2
            boxes.append(
                (rate_low, middle, freq_low, freq_high, anchor_low, anchor_high, depth + 1)
            )
            boxes.append(
                (middle, rate_high, freq_low, freq_high, anchor_low, anchor_high, depth + 1)
            )


def test_experimental_profile_sensitivity_enclosures() -> None:
    bands = (
        (44_100, 48_000, 64_000, 80_000, 96_000),
        (176_400, 192_000, 208_000, 224_000, 240_000, 256_000, 288_000, 320_000, 352_000, 384_000),
    )
    for anchors in bands:
        _check_boxes(
            deque(
                (
                    low,
                    high,
                    20 * (1000 ** (index / 32)),
                    20 * (1000 ** ((index + 1) / 32)),
                    low,
                    high,
                    0,
                )
                for low, high in itertools.pairwise(anchors)
                for index in range(32)
            )
        )
    _check_boxes(
        deque(
            (
                48_000,
                48_000,
                20_000 + 2050 * index / 16,
                20_000 + 2050 * (index + 1) / 16,
                48_000,
                64_000,
                0,
            )
            for index in range(16)
        )
    )


def test_section_derivative_bounds_match_independent_finite_differences() -> None:
    def response(alpha: float, omega: float) -> complex:
        z2 = cmath.exp(-2j * omega)
        return (alpha - z2) / (1 - alpha * z2)

    step = 1e-8
    for alpha, omega in ((0.1, 0.3), (0.8, 1.7), (0.99976, 0.000327)):
        omega_bounds, alpha_bounds = _derivative_bounds(alpha, alpha, omega, omega)
        actual_omega = cmath.phase(
            response(alpha, omega + step) / response(alpha, omega - step)
        ) / (2 * step)
        actual_alpha = cmath.phase(
            response(alpha + step, omega) / response(alpha - step, omega)
        ) / (2 * step)
        for bounds, actual in ((omega_bounds, actual_omega), (alpha_bounds, actual_alpha)):
            assert abs(sum(bounds) / 2 - actual) < max(1e-5, abs(actual) * 1e-6)


def test_enclosures_cover_independent_interior_samples() -> None:
    random_generator = random.Random(719)
    anchors = sorted(_HILBERT_PROFILE_BITS)
    segments = [
        (low, high)
        for low, high in itertools.pairwise(anchors)
        if not (low == 96_000 and high == 176_400)
    ]
    for _ in range(200):
        anchor_low, anchor_high = random_generator.choice(segments)
        rate_low = random_generator.uniform(anchor_low, anchor_high)
        rate_high = random_generator.uniform(rate_low, anchor_high)
        freq_low = 20 * (950 ** random_generator.random())
        freq_high = min(20_000.0, freq_low * random_generator.uniform(1, 1.1))
        box = (rate_low, rate_high, freq_low, freq_high, anchor_low, anchor_high, 0)
        bound, _, _ = _estimate_box(box)
        rate = random_generator.uniform(rate_low, rate_high)
        frequency = random_generator.uniform(freq_low, freq_high)
        alpha = _coefficients(rate, anchor_low, anchor_high)
        actual = math.radians(_quadrature_error_degrees(rate, frequency, alpha))
        assert actual <= bound
