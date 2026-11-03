"""Independent control-math and state examples, not runtime backend tests."""

from __future__ import annotations

import hashlib
import math
import struct
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import numpy.typing as npt
import pytest

_F32 = struct.Struct("<f")
_U32 = struct.Struct("<I")
_F64 = struct.Struct("<d")
_U64 = struct.Struct("<Q")
_MANTISSA_MASK = (1 << 23) - 1
_TABLE_SHA256 = "4d0d7c0ec194dd9d761bfe2982ca4847001f333c58a3f12f423ed2053de3a5f1"
_PITCH_SCALE = _F64.unpack(_U64.pack(0x3FAD_9303_FEA2_F7EA))[0]


def _float_from_bits(bits: int) -> float:
    return _F32.unpack(_U32.pack(bits))[0]


def _bits_from_float(value: float) -> int:
    return _U32.unpack(_F32.pack(value))[0]


def _nearest_f32_bits(value: Decimal) -> int:
    if value == 0:
        return 0
    approximate = _bits_from_float(float(value))
    return min(
        (approximate - 1, approximate, approximate + 1),
        key=lambda bits: (abs(Decimal.from_float(_float_from_bits(bits)) - value), bits & 1),
    )


def _reference_table() -> npt.NDArray[np.float32]:
    with localcontext() as context:
        context.prec = 100
        log_two = Decimal(2).ln()
        bits = [
            _nearest_f32_bits((Decimal(128 + index) / 128).ln() / log_two) for index in range(129)
        ]
    return np.asarray(bits, dtype=np.uint32).view(np.float32)


def _decompose_positive_f32(bits: int) -> tuple[int, int] | None:
    if not 0 <= bits <= 0xFFFF_FFFF:
        raise ValueError("not a float32 bit pattern")
    magnitude = bits & 0x7FFF_FFFF
    if magnitude == 0:
        return None
    exponent_bits = (bits >> 23) & 0xFF
    if bits >> 31 or exponent_bits == 0xFF:
        raise ValueError("negative or non-finite envelope")
    fraction = bits & _MANTISSA_MASK
    if exponent_bits:
        return exponent_bits - 127, fraction
    top_bit = fraction.bit_length() - 1
    return top_bit - 149, (fraction << (23 - top_bit)) & _MANTISSA_MASK


def _lookup_mantissas(
    fraction: npt.NDArray[np.uint32], table: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    index = fraction >> np.uint32(16)
    alpha = (fraction & np.uint32(0xFFFF)).astype(np.float32) * np.float32(2**-16)
    low = table[index]
    high = table[index + np.uint32(1)]
    return np.minimum(high, low + alpha * (high - low))


def _current_rotor_f32(theta: float) -> tuple[float, float]:
    single = np.float32
    clamped = np.clip(single(theta), single(0.0001), single(math.pi) - single(0.0001))
    phi = clamped - single(math.pi / 2)
    p2 = phi * phi
    sin_phi = phi * (
        single(0.99999995)
        - p2 * (single(0.1666665) - p2 * (single(0.0083329) - p2 * single(0.00019507)))
    )
    cos_phi = single(0.99999995) - p2 * (
        single(0.4999999)
        - p2 * (single(0.0416663) - p2 * (single(0.0013886) - p2 * single(0.00002438)))
    )
    cosine, sine = -sin_phi, cos_phi
    scale = single(1.5) - single(0.5) * (cosine * cosine + sine * sine)
    return float(cosine * scale), float(sine * scale)


def _project_rotor_components(
    cosine: npt.NDArray[np.float32],
    sine: npt.NDArray[np.float32],
    active: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    reduce_cosine = np.abs(cosine) >= np.abs(sine)
    cosine_bits = cosine.view(np.uint32)
    sine_bits = sine.view(np.uint32)
    for _ in range(2):
        c, s = cosine.astype(np.float64), sine.astype(np.float64)
        needs_reduction = active & (c * c + s * s >= 1)
        cosine_bits[needs_reduction & reduce_cosine] -= np.uint32(1)
        sine_bits[needs_reduction & ~reduce_cosine] -= np.uint32(1)
    return cosine, sine


def _adaptive_rotor(
    angles: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    theta = angles.astype(np.float64)
    magnitude = np.abs(theta)
    if np.any(~np.isfinite(theta)) or np.any(magnitude >= math.pi):
        raise ValueError("invalid rotor angle")
    reflected = magnitude > math.pi / 2
    quadrant = np.where(reflected, math.pi - magnitude, magnitude)
    swapped = quadrant > math.pi / 4
    reduced = np.where(swapped, math.pi / 2 - quadrant, quadrant)
    square = reduced * reduced
    sine = reduced * (
        1
        + square
        * (
            -1 / 6
            + square
            * (
                1 / 120
                + square
                * (
                    -1 / 5040
                    + square * (1 / 362880 + square * (-1 / 39916800 + square * (1 / 6227020800)))
                )
            )
        )
    )
    cosine = 1 + square * (
        -1 / 2
        + square
        * (
            1 / 24
            + square
            * (
                -1 / 720
                + square
                * (
                    1 / 40320
                    + square
                    * (-1 / 3628800 + square * (1 / 479001600 - square * (1 / 87178291200)))
                )
            )
        )
    )
    quadrant_cosine = np.where(swapped, sine, cosine)
    quadrant_sine = np.where(swapped, cosine, sine)
    result_cosine = np.where(reflected, -quadrant_cosine, quadrant_cosine).astype(np.float32)
    result_sine = np.where(theta < 0, -quadrant_sine, quadrant_sine).astype(np.float32)
    result_cosine, result_sine = _project_rotor_components(result_cosine, result_sine, theta != 0)
    return (
        np.where(theta == 0, np.float32(1), result_cosine),
        np.where(theta == 0, np.float32(0), result_sine),
    )


def _pitch_profile(
    semitones: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    p = semitones.astype(np.float64)
    if np.any(~np.isfinite(p)) or np.any(np.abs(p) > 24):
        raise ValueError("invalid pitch interval")
    x = p * _PITCH_SCALE
    poly = np.full(x.shape, 1 / math.factorial(14), dtype=np.float64)
    for degree in range(13, 0, -1):
        poly = 1 / math.factorial(degree) + x * poly
    delta = x * poly
    ratio = (1 + delta).astype(np.float32)
    warp = (-delta).astype(np.float32)
    return ratio, np.where(p == 0, np.float32(0), warp)


def _naive_f32_pitch_warp(semitones: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    x = semitones * np.float32(_PITCH_SCALE)
    poly = np.full(x.shape, np.float32(1 / math.factorial(14)), dtype=np.float32)
    for degree in range(13, 0, -1):
        poly = np.float32(1 / math.factorial(degree)) + x * poly
    return (-x * poly).astype(np.float32)


def test_previous_log_polynomial_misses_its_documented_accuracy() -> None:
    mantissa = _float_from_bits(0x3FFF_FFFF)
    t = mantissa - 1
    approximation = t * (1.442695 - t * (0.72070 - t * (0.4700 - 0.2555 * t)))
    assert 0.0635 < abs(approximation - math.log2(mantissa)) < 0.0636


def test_lookup_table_bits_match_high_precision_reference() -> None:
    table = _reference_table()
    assert table.shape == (129,)
    assert table[0] == 0
    assert table[-1] == 1
    assert np.all(table[1:] > table[:-1])
    assert hashlib.sha256(table.astype("<f4").tobytes()).hexdigest() == _TABLE_SHA256

    nodes = 1.0 + np.arange(129, dtype=np.float64) / 128.0
    assert np.array_equal(table, np.log2(nodes).astype(np.float32))


def test_lookup_is_monotone_and_accurate_for_every_f32_mantissa() -> None:
    table = _reference_table()
    fraction = np.arange(1 << 23, dtype=np.uint32)
    result = _lookup_mantissas(fraction, table)
    assert np.all(result[1:] >= result[:-1])
    exact_mantissa = 1.0 + fraction.astype(np.float64) * 2**-23
    reference = np.log2(exact_mantissa)

    for exponent in (-149, -126, 0, 127):
        rounded = (result + np.float32(exponent)).astype(np.float64)
        assert np.max(np.abs(rounded - (reference + exponent))) < 1.9e-5
        assert np.all(rounded[1:] >= rounded[:-1])

    interpolation = 1 / (8 * math.log(2) * 128**2)
    rounding_allowance = 2**-23 + 2**-30 + 2**-17
    assert interpolation + rounding_allowance < 1.9e-5


def test_subnormal_decomposition_preserves_the_original_bit_pattern() -> None:
    for top_bit in range(23):
        low = 1 << top_bit
        high = min((1 << (top_bit + 1)) - 1, _MANTISSA_MASK)
        for bits in {low, min(low + 1, high), high}:
            decomposed = _decompose_positive_f32(bits)
            assert decomposed is not None
            exponent, fraction = decomposed
            reconstructed = math.ldexp(1.0 + fraction * 2**-23, exponent)
            assert reconstructed == bits * 2**-149
    assert _decompose_positive_f32(0x0000_0001) == (-149, 0)
    assert _decompose_positive_f32(0x0080_0000) == (-126, 0)


def test_lookup_is_monotone_and_accurate_for_every_positive_subnormal() -> None:
    table = _reference_table()
    previous = -math.inf
    for top_bit in range(23):
        stored = np.arange(1 << top_bit, 1 << (top_bit + 1), dtype=np.uint32)
        fraction = (stored << np.uint32(23 - top_bit)) & np.uint32(_MANTISSA_MASK)
        result = (_lookup_mantissas(fraction, table) + np.float32(top_bit - 149)).astype(np.float64)
        reference = np.log2(stored.astype(np.float64)) - 149
        assert np.max(np.abs(result - reference)) < 1.9e-5
        assert result[0] >= previous
        assert np.all(result[1:] >= result[:-1])
        previous = float(result[-1])
    assert previous <= -126


@pytest.mark.parametrize("bits", [0, 0x8000_0000])
def test_zero_envelope_has_a_separate_silent_status(bits: int) -> None:
    assert _decompose_positive_f32(bits) is None


@pytest.mark.parametrize("bits", [0x8000_0001, 0xBF80_0000, 0x7F80_0000, 0x7FC0_0000])
def test_negative_or_nonfinite_envelope_is_rejected(bits: int) -> None:
    with pytest.raises(ValueError, match="negative or non-finite"):
        _decompose_positive_f32(bits)


def test_float32_rounding_alone_refutes_the_previous_rotor_error_claim() -> None:
    exact = math.cos(1.0)
    nearest_bits = _bits_from_float(exact)
    error = min(
        abs(_float_from_bits(bits) - exact) for bits in range(nearest_bits - 1, nearest_bits + 2)
    )
    assert error > 2.9e-8


def test_current_rotor_moves_a_zero_rate_lfo_and_can_exceed_unit_norm() -> None:
    cosine, sine = _current_rotor_f32(0.0)
    frequency = math.atan2(sine, cosine) * 3000 / (2 * math.pi)
    assert 0.053 < frequency < 0.054

    cosine, sine = _current_rotor_f32(2.7439392)
    assert cosine * cosine + sine * sine > 1.0000001


def test_adaptive_rotor_accuracy_and_unit_bound_across_sampled_angles() -> None:
    last = np.nextafter(np.float32(math.pi), np.float32(0))
    theta = np.linspace(0, last, 2_000_001, dtype=np.float32)
    cosine, sine = _adaptive_rotor(theta)
    reference = theta.astype(np.float64)
    c, s = cosine.astype(np.float64), sine.astype(np.float64)
    error = np.maximum(np.abs(c - np.cos(reference)), np.abs(s - np.sin(reference)))
    assert np.max(error) < 1e-7
    assert np.max(np.abs(np.arctan2(s, c) - reference)) < 1e-7
    assert c[0] == 1
    assert s[0] == 0
    assert np.all(c * c + s * s <= 1)
    assert np.all((c * c + s * s)[1:] < 1)
    assert np.min((c * c + s * s)[1:]) > 0.9999998

    sine_remainder = (math.pi / 4) ** 15 / math.factorial(15)
    cosine_remainder = (math.pi / 4) ** 16 / math.factorial(16)
    evaluation_allowance = 2e-14
    component_error = max(sine_remainder, cosine_remainder) + evaluation_allowance + 2**-25
    largest_lower_bound = 1 / math.sqrt(2) - component_error
    two_ulp_reduction = 2**-23
    squared_bound = (1 + math.sqrt(2) * component_error) ** 2
    squared_bound -= 2 * largest_lower_bound * two_ulp_reduction - two_ulp_reduction**2
    assert squared_bound < 1
    assert component_error + two_ulp_reduction < 1.6e-7


def test_adaptive_rotor_signed_angles_and_boundaries() -> None:
    angles = np.asarray(
        [0x0000_0000, 0x8000_0000, 1, 0x007F_FFFF, 0x0080_0000, 0x3F80_0000, 0x4049_0FDA],
        dtype=np.uint32,
    ).view(np.float32)
    cosine, sine = _adaptive_rotor(angles)
    neg_cosine, neg_sine = _adaptive_rotor(-angles)
    assert np.array_equal(cosine, neg_cosine)
    assert np.array_equal(sine, -neg_sine)
    assert cosine[0] == cosine[1] == 1
    assert sine[0] == sine[1] == 0
    c, s = cosine[2:].astype(np.float64), sine[2:].astype(np.float64)
    assert np.all(c * c + s * s < 1)
    for c_value, s_value in zip(c, s, strict=True):
        exact_norm = Fraction.from_float(float(c_value)) ** 2
        exact_norm += Fraction.from_float(float(s_value)) ** 2
        assert exact_norm < 1


def test_adaptive_rotor_preserves_small_cosine_near_quadrant_boundary() -> None:
    center = _bits_from_float(math.pi / 2)
    angles = np.arange(center - 2, center + 3, dtype=np.uint32).view(np.float32)
    cosine, _ = _adaptive_rotor(angles)
    assert np.max(np.abs(cosine.astype(np.float64) - np.cos(angles.astype(np.float64)))) < 1e-14


def test_fixed_phase_target_uses_safe_quadrants_for_the_selected_rotor() -> None:
    boundaries = np.asarray(
        [
            0,
            1,
            0x3FFF_FFFF,
            0x4000_0000,
            0x4000_0001,
            0x7FFF_FFFF,
            0x8000_0000,
            0x8000_0001,
            0xBFFF_FFFF,
            0xC000_0000,
            0xC000_0001,
            0xFFFF_FFFF,
        ],
        dtype=np.uint32,
    )
    phases = np.concatenate((boundaries, np.linspace(0, 0xFFFF_FFFF, 200_001, dtype=np.uint32)))
    quadrant = (phases >> np.uint32(30)).astype(np.intp)
    remainder = phases & np.uint32(0x3FFF_FFFF)
    angle = remainder.astype(np.float32) * np.float32(_float_from_bits(0x30C9_0FDB))
    cosine, sine = _adaptive_rotor(angle)
    real = np.choose(quadrant, (cosine, -sine, -cosine, sine)).astype(np.float64)
    imaginary = np.choose(quadrant, (sine, cosine, -sine, -cosine)).astype(np.float64)
    ideal = phases.astype(np.float64) * (2 * math.pi / 2**32)
    error = np.remainder(np.arctan2(imaginary, real) - ideal + math.pi, 2 * math.pi) - math.pi
    assert np.max(np.abs(error)) < 3.9e-7
    assert np.all(real * real + imaginary * imaginary <= 1)
    assert np.array_equal(real[[0, 3, 6, 9]], np.asarray([1, 0, -1, 0]))
    assert np.array_equal(imaginary[[0, 3, 6, 9]], np.asarray([0, 1, 0, -1]))

    with pytest.raises(ValueError, match="invalid rotor angle"):
        _adaptive_rotor(np.asarray([math.pi], dtype=np.float32))


def test_rotor_projection_checks_rounded_equality_and_has_two_step_limit() -> None:
    small = np.float32(2**-30)
    assert np.float64(1) + np.float64(small) ** 2 == 1
    assert Fraction(1) + Fraction.from_float(float(small)) ** 2 > 1
    c, s = _project_rotor_components(
        np.asarray([1], dtype=np.float32),
        np.asarray([small], dtype=np.float32),
        np.asarray([True], dtype=np.bool_),
    )
    assert float(c[0]) < 1
    exact_norm = Fraction.from_float(float(c[0])) ** 2 + Fraction.from_float(float(s[0])) ** 2
    assert exact_norm < 1

    above = np.nextafter(np.float32(math.sqrt(0.5)), np.float32(math.inf))
    one_down = np.nextafter(above, np.float32(0))
    assert float(one_down) ** 2 + float(above) ** 2 > 1
    c, s = _project_rotor_components(
        np.asarray([above], dtype=np.float32),
        np.asarray([above], dtype=np.float32),
        np.asarray([True], dtype=np.bool_),
    )
    assert c[0] == np.nextafter(one_down, np.float32(0))
    assert s[0] == above
    exact_norm = Fraction.from_float(float(c[0])) ** 2 + Fraction.from_float(float(s[0])) ** 2
    assert exact_norm < 1


def test_adaptive_rotor_decay_cost_is_bounded_but_not_zero() -> None:
    theta = np.linspace(0, np.nextafter(np.float32(math.pi), np.float32(0)), 1_000_001)
    cosine, sine = _adaptive_rotor(theta.astype(np.float32))
    radius = np.sqrt(cosine.astype(np.float64) ** 2 + sine.astype(np.float64) ** 2)
    reference_radius = 0.9995
    effective_radius = reference_radius ** (1 / 8)
    t60_ratio = math.log(effective_radius) / math.log(effective_radius * float(radius.min()))
    assert 0.999 < t60_ratio < 1


@pytest.mark.parametrize("theta", [-math.pi, math.pi, math.inf, -math.inf, math.nan])
def test_adaptive_rotor_rejects_invalid_angles(theta: float) -> None:
    with pytest.raises(ValueError, match="invalid rotor angle"):
        _adaptive_rotor(np.asarray([theta], dtype=np.float32))


def test_adaptive_lfo_example_holds_zero_and_stabilizes_sampled_states() -> None:
    frequency = np.asarray([0, 0.01, 0.1, 0.5, 2.5, 10, 50, 100], dtype=np.float64)
    theta = (2 * math.pi * frequency / 3000).astype(np.float32)
    cosine, sine = _adaptive_rotor(theta)
    real = np.ones(theta.shape, dtype=np.float32)
    imaginary = np.zeros(theta.shape, dtype=np.float32)
    real[0], imaginary[0] = np.float32(0.6), np.float32(0.8)
    held = (real[0], imaginary[0])

    for _ in range(300_000):
        next_real = cosine * real - sine * imaginary
        next_imaginary = sine * real + cosine * imaginary
        correction = np.float32(1.5) - np.float32(0.5) * (
            next_real * next_real + next_imaginary * next_imaginary
        )
        real[1:] = (next_real * correction)[1:]
        imaginary[1:] = (next_imaginary * correction)[1:]

    assert (real[0], imaginary[0]) == held
    r = real[1:].astype(np.float64)
    i = imaginary[1:].astype(np.float64)
    assert np.max(np.abs(np.sqrt(r * r + i * i) - 1)) < 1e-6
    ideal_phase = np.remainder(theta[1:].astype(np.float64) * 300_000 + math.pi, 2 * math.pi)
    ideal_phase -= math.pi
    phase_error = np.remainder(np.arctan2(i, r) - ideal_phase + math.pi, 2 * math.pi)
    assert np.max(np.abs(phase_error - math.pi)) < 0.003


def test_pitch_profile_accuracy_across_sampled_domain() -> None:
    semitones = np.linspace(-24, 24, 2_000_001, dtype=np.float32)
    ratio, warp = _pitch_profile(semitones)
    ideal_ratio = np.exp2(semitones.astype(np.float64) / 12)
    assert np.max(np.abs(ratio.astype(np.float64) / ideal_ratio - 1)) < 6.2e-8
    assert np.max(np.abs(warp.astype(np.float64) - (1 - ideal_ratio))) < 1.2e-7
    assert np.all(ratio[1:] >= ratio[:-1])
    assert np.all(warp[1:] <= warp[:-1])

    max_x = math.log(4)
    tail = 4 * max_x**15 / math.factorial(15)
    arithmetic_allowance = 24e-13
    relative_bound = 2**-24 + (1 + 2**-24) * (tail + arithmetic_allowance) / 0.25
    assert relative_bound < 6.2e-8
    assert tail + arithmetic_allowance + 2**-23 < 1.2e-7


def test_pitch_profile_endpoints_zero_and_near_zero() -> None:
    semitones = np.asarray([-24, -12, -1e-8, -0.0, 0.0, 1e-8, 12, 24], dtype=np.float32)
    ratio, warp = _pitch_profile(semitones)
    assert np.array_equal(ratio[[0, 1, 3, 4, 6, 7]], [0.25, 0.5, 1, 1, 2, 4])
    assert np.array_equal(warp[[0, 1, 3, 4, 6, 7]], [0.75, 0.5, 0, 0, -1, -3])
    assert warp.view(np.uint32)[3] == 0
    assert warp.view(np.uint32)[4] == 0
    assert ratio[2] == ratio[5] == 1
    assert warp[2] > 0
    assert warp[5] < 0
    assert np.float32(1) - ratio[5] == 0

    with localcontext() as context:
        context.prec = 100
        scale = Decimal(2).ln() / Decimal(12)
        assert float(scale) == _PITCH_SCALE
        for index in (0, 1, 2, 5, 6, 7):
            exact = (Decimal.from_float(float(semitones[index])) * scale).exp()
            error = abs(Decimal.from_float(float(ratio[index])) / exact - 1)
            assert error < Decimal("6.2e-8")


@pytest.mark.parametrize("bits", [0x3A00_0000, 0x3F00_0000, 0x4000_0000, 0x4180_0000])
def test_pitch_profile_monotone_for_adjacent_float32_inputs(bits: int) -> None:
    magnitude = np.arange(bits, bits + 100_000, dtype=np.uint32).view(np.float32)
    positive_ratio, positive_warp = _pitch_profile(magnitude)
    negative_ratio, negative_warp = _pitch_profile(-magnitude)
    assert np.all(positive_ratio[1:] >= positive_ratio[:-1])
    assert np.all(positive_warp[1:] <= positive_warp[:-1])
    assert np.all(negative_ratio[1:] <= negative_ratio[:-1])
    assert np.all(negative_warp[1:] >= negative_warp[:-1])


def test_naive_f32_pitch_polynomial_can_reverse_adjacent_inputs() -> None:
    magnitude = np.asarray([0x4180_0010, 0x4180_0011], dtype=np.uint32).view(np.float32)
    naive = _naive_f32_pitch_warp(-magnitude)
    assert naive[1] < naive[0]
    _, reference = _pitch_profile(-magnitude)
    assert reference[1] >= reference[0]


@pytest.mark.parametrize("p", [-24.000002, 24.000002, math.inf, -math.inf, math.nan])
def test_pitch_profile_rejects_invalid_intervals(p: float) -> None:
    with pytest.raises(ValueError, match="invalid pitch interval"):
        _pitch_profile(np.asarray([p], dtype=np.float32))


def test_conditional_switch_can_invalidate_an_omitted_control() -> None:
    wide_range = (Fraction(0), Fraction(1))
    limited_range = (Fraction(0), Fraction(3, 10))
    old_value = Fraction(4, 5)
    assert wide_range[0] <= old_value <= wide_range[1]
    assert old_value > limited_range[1]
    safe_value = min(max(old_value, limited_range[0]), limited_range[1])
    assert safe_value == Fraction(3, 10)


def test_neither_serial_order_is_safe_for_all_conditional_transitions() -> None:
    limited_max = Fraction(3, 10)
    wide = ("wide", Fraction(4, 5))
    limited = ("limited", Fraction(1, 5))
    assert limited[1] <= limited_max < wide[1]

    forward_switch_first = (limited[0], wide[1])
    forward_range_first = (wide[0], limited[1])
    reverse_range_first = (limited[0], wide[1])
    reverse_switch_first = (wide[0], limited[1])
    assert forward_switch_first[0] == "limited"
    assert forward_switch_first[1] > limited_max
    assert forward_range_first[0] == "wide"
    assert forward_range_first[1] <= 1
    assert reverse_range_first[0] == "limited"
    assert reverse_range_first[1] > limited_max
    assert reverse_switch_first[0] == "wide"
    assert reverse_switch_first[1] <= 1


def test_disjoint_switch_intervals_have_no_continuous_common_boundary() -> None:
    previous_range = (Fraction(7, 10), Fraction(1))
    next_range = (Fraction(0), Fraction(3, 10))
    intersection_lower = max(previous_range[0], next_range[0])
    intersection_upper = min(previous_range[1], next_range[1])
    assert intersection_lower > intersection_upper
    assert intersection_lower - intersection_upper == Fraction(2, 5)


def test_complementary_output_blend_preserves_exact_path_bound() -> None:
    for old in range(-4, 5):
        for new in range(-4, 5):
            for step in range(9):
                weight = Fraction(step, 8)
                blended = (1 - weight) * old + weight * new
                assert abs(blended) <= max(abs(old), abs(new))


def test_bounded_output_blend_can_cancel_or_color_valid_paths() -> None:
    weight = Fraction(1, 2)
    assert (1 - weight) * 1 + weight * -1 == 0
    phase = 2 * math.pi * 20_000 / 48_000
    delayed = complex(math.cos(phase), -math.sin(phase))
    frozen_midpoint = 0.5 * (1 + delayed)
    assert 20 * math.log10(abs(frozen_midpoint)) == pytest.approx(-11.740075, abs=1e-6)


def test_conditional_bounds_can_have_an_empty_safe_intersection() -> None:
    base = (Fraction(0), Fraction(1))
    conditional_min = Fraction(7, 10)
    conditional_max = Fraction(3, 10)
    lower = max(base[0], conditional_min)
    upper = min(base[1], conditional_max)
    assert lower > upper
