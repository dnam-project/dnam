"""Feasibility examples for fractional Stage III FIR and modal conversion.

The coefficients and constructions are experimental, not an engine profile.
"""

from __future__ import annotations

import cmath
import itertools
import math
import random
import re
import struct
from fractions import Fraction
from pathlib import Path

_F32 = struct.Struct(">f")
_U32 = struct.Struct(">I")
_F64 = struct.Struct(">d")
_U64 = struct.Struct(">Q")

# A 64-tap first-tap fit at 44.1 kHz, with 0.001 out-of-band fit weight.
_FIRST_TAP_FILTER_BITS = (
    0x3D495965,
    0x3F842AA9,
    0xBDFB2469,
    0x3DBB6A7C,
    0xBDAA241A,
    0x3DA4091B,
    0xBDA15964,
    0x3D9F3B82,
    0xBD9C5D4D,
    0x3D980499,
    0xBD91D439,
    0x3D899EFF,
    0xBD7ED2F2,
    0x3D66A54B,
    0xBD4B3B73,
    0x3D2D3B04,
    0xBD0D7D32,
    0x3CD9C983,
    0xBC98E285,
    0x3C345115,
    0xBB7C482C,
    0xBB2AC0C0,
    0x3C05EC82,
    0xBC5074E9,
    0x3C8449BD,
    0xBC96CAA7,
    0x3C9FAA2A,
    0xBC9F572B,
    0x3C967D9B,
    0xBC8638DA,
    0x3C5F8DCF,
    0xBC2968BC,
    0x3BDA2C81,
    0xBB37DF60,
    0xBA8733FB,
    0x3B98EEE4,
    0xBC0247F8,
    0x3C301D9A,
    0xBC548487,
    0x3C6E5E93,
    0xBC7D6129,
    0x3C80CA1D,
    0xBC7BC12A,
    0x3C6CE47A,
    0xBC5691F3,
    0x3C3A5E81,
    0xBC1A3FC6,
    0x3BEFE97D,
    0xBBAACE9D,
    0x3B50673C,
    0xBAAC7FAC,
    0xB9A74D88,
    0x3ADAF0C6,
    0xBB321B17,
    0x3B61D47E,
    0xBB7DCD46,
    0x3B839D32,
    0xBB808A53,
    0x3B6DECC1,
    0xBB51998F,
    0x3B2F0B81,
    0xBB0A0250,
    0x3AC9DDB1,
    0xBA857949,
)

# A gain- and DC-constrained 64-tap first-tap candidate at 44.1 kHz.
_GAIN_BOUNDED_FIRST_TAP_BITS = (
    0x3D6C9268,
    0x3F83725F,
    0xBDF0A4F1,
    0x3DADB502,
    0xBD9DA17D,
    0x3D95380C,
    0xBD949C5F,
    0x3D91792E,
    0xBD91CE40,
    0x3D8DD068,
    0xBD8BEAE0,
    0x3D853DAC,
    0xBD80189F,
    0x3D6C7FA0,
    0xBD5BA083,
    0x3D4231CB,
    0xBD2BF99B,
    0x3D0E88F9,
    0xBCEA4A93,
    0x3CAC58C1,
    0xBC714876,
    0x3BF4C5F0,
    0xBAE22743,
    0xBB962D07,
    0x3C1634D0,
    0xBC65ADD4,
    0x3C8B7E54,
    0xBCA4D3DC,
    0x3CAF0E76,
    0xBCB982B1,
    0x3CB5D3C2,
    0xBCB2E27A,
    0x3CA3932B,
    0xBC963F77,
    0x3C7DC51E,
    0xBC565369,
    0x3C1FD21C,
    0xBBE7F768,
    0x3B775394,
    0xBAA98A21,
    0xBACF4513,
    0x3B66E3C0,
    0xBBBAAF04,
    0x3BE253A0,
    0xBC0770ED,
    0x3C0EA839,
    0xBC181CD3,
    0x3C14420D,
    0xBC133BF5,
    0x3C074843,
    0xBBFE526F,
    0x3BDD6C26,
    0xBBC62D29,
    0x3BA31DC4,
    0xBB8B1039,
    0x3B568017,
    0xBB2D823B,
    0x3AF63E3F,
    0xBABAD495,
    0x3A6AE4C8,
    0xBA231F37,
    0x39A60FED,
    0xB943301A,
    0x386DC990,
)

# A 30-tap, 2049-frequency weighted least-squares fit to 14.7 samples.
_MODAL_FILTER_BITS = (
    0x3B00D771,
    0xBB4AF619,
    0x3B9637E8,
    0xBBD51306,
    0x3C1267E6,
    0xBC4499D8,
    0x3C81CA65,
    0xBCA986D0,
    0x3CDC59A5,
    0xBD0F9204,
    0x3D3D71EC,
    0xBD809F7F,
    0x3DB93F22,
    0xBE1816A3,
    0x3EBBE0FC,
    0x3F5B7688,
    0xBE479F77,
    0x3DDACF52,
    0xBD9176D7,
    0x3D516097,
    0xBD1C829B,
    0x3CEE1CF9,
    0xBCB621EF,
    0x3C8AE64F,
    0xBC51D8EA,
    0x3C1BFF88,
    0xBBE2C539,
    0x3B9FBF08,
    0xBB57AD81,
    0x3B08C773,
)


def _coefficients() -> tuple[float, ...]:
    return tuple(_F32.unpack(_U32.pack(bits))[0] for bits in _MODAL_FILTER_BITS)


def _round_f32(value: float) -> float:
    return _F32.unpack(_F32.pack(value))[0]


def _first_tap_difference_coefficients() -> tuple[float, ...]:
    coefficients = tuple(_F32.unpack(_U32.pack(bits))[0] for bits in _GAIN_BOUNDED_FIRST_TAP_BITS)
    prefix = Fraction()
    differences = []
    for coefficient in coefficients[:-1]:
        prefix += Fraction.from_float(coefficient)
        exact = prefix - 1
        assert Fraction.from_float(float(exact)) == exact
        differences.append(_round_f32(float(exact)))
    return tuple(differences)


def _first_tap_residual_coefficients() -> tuple[float, ...]:
    differences = _first_tap_difference_coefficients()
    return (_round_f32(1 + differences[0]), *differences[1:])


def _six_point_lagrange(mu: Fraction) -> tuple[Fraction, ...]:
    if not 0 <= mu <= 1:
        raise ValueError("invalid fractional delay")
    nodes = (-2, -1, 0, 1, 2, 3)
    weights = []
    for node in nodes:
        coefficient = Fraction(1)
        for other in nodes:
            if other != node:
                coefficient *= (mu - other) / (node - other)
        weights.append(coefficient)
    return tuple(weights)


def _six_point_residual(mu: Fraction) -> tuple[Fraction, ...]:
    prefix = Fraction()
    residual = []
    anchor = 2 if mu <= Fraction(1, 2) else 3
    for index, coefficient in enumerate(_six_point_lagrange(mu)[:-1]):
        prefix += coefficient
        residual.append(prefix - int(index >= anchor))
    return tuple(residual)


def _round_normal_fraction_f32(value: Fraction) -> float:
    if value == 0:
        return 0.0
    magnitude = abs(value)
    approximate = _round_f32(float(magnitude))
    bits = _U32.unpack(_F32.pack(approximate))[0]
    if not 0x0080_0000 <= bits < 0x7F7F_FFFF:
        raise ValueError("reference expects a normal finite coefficient")
    neighbors = (bits - 1, bits, bits + 1)

    def key(candidate: int) -> tuple[Fraction, int]:
        rounded = _F32.unpack(_U32.pack(candidate))[0]
        return abs(Fraction.from_float(rounded) - magnitude), candidate & 1

    result = _F32.unpack(_U32.pack(min(neighbors, key=key)))[0]
    return -result if value < 0 else result


def _bounce_rate_parts(rate: float, tap: int) -> tuple[int, int, int]:
    if not 176_400 <= rate <= 384_000 or not 1 <= tap <= 63:
        raise ValueError("unsupported effective rate or tap")
    bits = _U64.unpack(_F64.pack(rate))[0]
    exponent = ((bits >> 52) & 0x7FF) - 1023
    significand = (bits & ((1 << 52) - 1)) | (1 << 52)
    denominator = 48_000 << (52 - exponent)
    integer, remainder = divmod(tap * significand, denominator)
    return integer, remainder, denominator


def _six_point_residual_ratio(remainder: int, denominator: int) -> tuple[tuple[int, ...], int]:
    if not 0 <= remainder < denominator:
        raise ValueError("invalid fractional delay")
    nodes = (-2, -1, 0, 1, 2, 3)
    scales = (-1, 5, -10, 10, -5, 1)
    common = 120 * denominator**5
    anchor = 2 if 2 * remainder <= denominator else 3
    prefix = 0
    residual = []
    for index, (node, scale) in enumerate(zip(nodes[:-1], scales[:-1], strict=True)):
        prefix += scale * math.prod(
            remainder - other * denominator for other in nodes if other != node
        )
        residual.append(prefix - int(index >= anchor) * common)
    return tuple(residual), common


def _round_ratio_f32_bits(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator == 0:
        return 0
    sign = int(numerator < 0) << 31
    magnitude = abs(numerator)
    exponent = magnitude.bit_length() - denominator.bit_length()
    if exponent >= 0:
        exponent -= int(magnitude < denominator << exponent)
    else:
        exponent -= int(magnitude << -exponent < denominator)
    shift = 149 if exponent < -126 else 23 - exponent
    if not 0 <= shift <= 149:
        raise ValueError("ratio outside supported rounding range")
    scaled = magnitude << shift
    quotient, remainder = divmod(scaled, denominator)
    quotient += int(2 * remainder > denominator or (2 * remainder == denominator and quotient & 1))
    if exponent < -126:
        return sign | quotient
    if quotient == 1 << 24:
        quotient >>= 1
        exponent += 1
    return sign | ((exponent + 127) << 23) | (quotient - (1 << 23))


def _response(
    coefficients: tuple[float, ...], delay: float, omega: float
) -> tuple[complex, complex]:
    response = 0j
    derivative = 0j
    for index, coefficient in enumerate(coefficients):
        relative = index - delay
        term = coefficient * cmath.exp(-1j * omega * relative)
        response += term
        derivative += -1j * relative * term
    return response, derivative


def test_causal_modal_candidate_matches_its_fractional_delay_on_a_dense_grid() -> None:
    coefficients = _coefficients()
    delay = 16 * 44_100 / 48_000
    edge = 2 * math.pi * 20_000 / 44_100
    steps = 16_384
    derivative_bound = sum(
        abs(coefficient * (index - delay)) for index, coefficient in enumerate(coefficients)
    )
    max_error = 0.0
    max_group_delay_error = 0.0
    for sample in range(steps + 1):
        response, derivative = _response(coefficients, delay, edge * sample / steps)
        max_error = max(max_error, abs(response - 1))
        max_group_delay_error = max(max_group_delay_error, abs((derivative / response).imag))
    between_samples = derivative_bound * edge / (2 * steps)
    assert 0.0065 < max_error < 0.0067
    assert max_error + between_samples + 1e-12 < 0.0071
    assert 0.22 < max_group_delay_error < 0.23


def test_modal_candidate_has_bounded_fullband_gain_and_nontrivial_preringing() -> None:
    coefficients = _coefficients()
    delay = 16 * 44_100 / 48_000
    steps = 16_384
    derivative_bound = sum(
        abs(coefficient * (index - delay)) for index, coefficient in enumerate(coefficients)
    )
    max_gain = max(
        abs(_response(coefficients, delay, math.pi * sample / steps)[0])
        for sample in range(steps + 1)
    )
    assert max_gain + derivative_bound * math.pi / (2 * steps) + 1e-12 < 1.0061
    assert 2.27 < sum(abs(value) for value in coefficients) < 2.28
    assert coefficients[0] != 0.0
    assert 0.017 < sum(value * value for value in coefficients[:13]) < 0.018


def test_real_fractional_delays_cannot_match_complex_nyquist_target() -> None:
    for delay, lower_bound in ((44_100 / 48_000, 0.252), (16 * 44_100 / 48_000, 0.809)):
        target = cmath.exp(-1j * math.pi * delay)
        assert abs(target.imag) > lower_bound
        for coefficients in ((1.0,), (0.25, 0.75), _coefficients()):
            nyquist = sum(
                coefficient * (-1) ** index for index, coefficient in enumerate(coefficients)
            )
            assert abs(nyquist - target) >= abs(target.imag)
            assert math.isclose(abs(target.real - target), abs(target.imag), abs_tol=1e-14)


def test_fixed_first_tap_example_misses_ultra_hf_goal_for_a_steady_tone() -> None:
    rate = 44_100
    omega = 2 * math.pi * 20_000 / rate
    coefficients = tuple(_F32.unpack(_U32.pack(bits))[0] for bits in _FIRST_TAP_FILTER_BITS)
    desired = cmath.exp(-1j * omega * rate / 48_000)
    actual = sum(
        coefficient * cmath.exp(-1j * omega * index)
        for index, coefficient in enumerate(coefficients)
    )
    assert 0.0584 < abs(actual - desired) < 0.0586

    pole = math.exp(-2 * math.pi * 2_500 / rate)
    lowpass = (1 - pole) / (1 - pole * cmath.exp(-1j * omega))
    complement = 1 - lowpass
    hf = 1 - lowpass**4 * (1 + 4 * complement + 10 * complement**2 + 20 * complement**3)
    target_hf = hf * desired
    residual_hf = hf * (actual - desired)
    assert abs(target_hf) > 0.9

    target_energy = 0.0
    residual_energy = 0.0
    for index in range(441):
        carrier = cmath.exp(1j * omega * index)
        target_energy += (target_hf * carrier).real ** 2
        residual_energy += (residual_hf * carrier).real ** 2
    ratio_db = 10 * math.log10(residual_energy / target_energy)
    assert -24.7 < ratio_db < -24.6
    assert ratio_db > -30.0


def test_gain_and_dc_bounded_first_tap_candidate_has_small_band_error() -> None:
    coefficients = tuple(_F32.unpack(_U32.pack(bits))[0] for bits in _GAIN_BOUNDED_FIRST_TAP_BITS)
    delay = 44_100 / 48_000
    edge = 2 * math.pi * 20_000 / 44_100
    steps = 16_384
    derivative_bound = sum(
        abs(coefficient * (index - delay)) for index, coefficient in enumerate(coefficients)
    )
    max_error = max(
        abs(_response(coefficients, delay, edge * sample / steps)[0] - 1)
        for sample in range(steps + 1)
    )
    error_bound = max_error + derivative_bound * edge / (2 * steps) + 1e-12
    max_gain = max(
        abs(_response(coefficients, delay, math.pi * sample / steps)[0])
        for sample in range(steps + 1)
    )
    gain_bound = max_gain + derivative_bound * math.pi / (2 * steps) + 1e-12
    assert 0.0147 < max_error < 0.0149
    assert error_bound < 0.01668
    assert 20 * math.log10(error_bound) < -35.5
    assert gain_bound < 1.983
    assert 2.57 < sum(map(abs, coefficients)) < 2.59
    assert sum(map(Fraction.from_float, coefficients)) == 1


def test_native_first_tap_fixture_matches_the_reference_bits() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "packages/engine/tests/stage3_first_tap_spec_references.rs").read_text(
        encoding="utf-8"
    )
    section = source.split("const COEFFICIENT_BITS: [u32; 64] = [", 1)[1].split("];", 1)[0]
    native_bits = tuple(
        int(word.replace("_", ""), 16) for word in re.findall(r"0x[\da-f_]+", section)
    )
    assert native_bits == _GAIN_BOUNDED_FIRST_TAP_BITS


def test_exact_coefficient_dc_gain_does_not_fix_f32_accumulation_order() -> None:
    coefficients = tuple(_F32.unpack(_U32.pack(bits))[0] for bits in _GAIN_BOUNDED_FIRST_TAP_BITS)

    def accumulate(values: tuple[float, ...]) -> float:
        total = 0.0
        for value in values:
            total = _F32.unpack(_F32.pack(total + value))[0]
        return total

    forward = accumulate(coefficients)
    reverse = accumulate(coefficients[::-1])
    assert _U32.unpack(_F32.pack(forward))[0] == 0x3F800002
    assert _U32.unpack(_F32.pack(reverse))[0] == 0x3F800000
    assert 1.0 - forward == -(2.0**-22)


def test_difference_realization_preserves_normal_dc_and_band_response() -> None:
    differences = _first_tap_difference_coefficients()
    equivalent = (
        1 + differences[0],
        *(current - previous for previous, current in itertools.pairwise(differences)),
        -differences[-1],
    )
    assert sum(map(Fraction.from_float, equivalent)) == 1
    assert 1.746 < sum(map(abs, differences)) < 1.747

    rate = 44_100
    delay = rate / 48_000
    edge = 2 * math.pi * 20_000 / rate
    steps = 16_384
    derivative_bound = sum(
        abs(coefficient * (index - delay)) for index, coefficient in enumerate(equivalent)
    )
    max_error = max(
        abs(_response(equivalent, delay, edge * sample / steps)[0] - 1)
        for sample in range(steps + 1)
    )
    max_gain = max(
        abs(_response(equivalent, delay, math.pi * sample / steps)[0])
        for sample in range(steps + 1)
    )
    assert max_error + derivative_bound * edge / (2 * steps) + 1e-12 < 0.01668
    assert max_gain + derivative_bound * math.pi / (2 * steps) + 1e-12 < 1.983

    for sample in (0.125, -0.125, 1.0, -1.0, 100.0, -100.0):
        rounded = _round_f32(sample)
        output = rounded
        delta = _round_f32(rounded - rounded)
        for coefficient in differences:
            output = _round_f32(output + _round_f32(coefficient * delta))
        assert _U32.unpack(_F32.pack(output))[0] == _U32.unpack(_F32.pack(rounded))[0]


def test_difference_realization_f32_example_tracks_a_20khz_tone() -> None:
    differences = _first_tap_difference_coefficients()
    rate = 44_100
    delay = rate / 48_000
    omega = 2 * math.pi * 20_000 / rate
    history = [0.0] * 64
    residual_energy = 0.0
    target_energy = 0.0
    for index in range(64 + 441):
        sample = _round_f32(math.sin(omega * index))
        history = [sample, *history[:-1]]
        output = sample
        for offset, coefficient in enumerate(differences):
            delta = _round_f32(history[offset] - history[offset + 1])
            output = _round_f32(output + _round_f32(coefficient * delta))
        if index >= 64:
            target = math.sin(omega * (index - delay))
            residual_energy += (output - target) ** 2
            target_energy += target**2
    assert 10 * math.log10(residual_energy / target_energy) < -35.5


def test_previous_sample_anchor_avoids_a_reference_rate_cancellation() -> None:
    current = 1.0
    previous = 2.0**-25
    difference = _round_f32(current - previous)
    current_anchored = _round_f32(current + _round_f32(-difference))
    previous_anchored = _round_f32(previous + _round_f32(0.0 * difference))
    assert current_anchored == 0.0
    assert previous_anchored == previous


def test_first_tap_interpolation_has_a_conditional_continuous_rate_bound() -> None:
    differences = _first_tap_difference_coefficients()
    residual = _first_tap_residual_coefficients()
    assert Fraction.from_float(residual[0]) == 1 + Fraction.from_float(differences[0])
    assert min(abs(value) for value in residual if value) > 2.0**-30
    residual_l1 = sum(map(abs, residual))
    assert 0.85 < residual_l1 < 0.87

    edge = 2 * math.pi * 20_000 / 44_100
    delay_span = 1 - 44_100 / 48_000
    interpolation = edge**2 * delay_span**2 / 2
    rounded_error_slope = 2 * 2.0**-24 * residual_l1
    rounded_error_floor = 126 * 2.0**-150
    endpoint_error = 0.01668 + rounded_error_slope
    assert endpoint_error < interpolation
    error_bound = (endpoint_error + interpolation) ** 2 / (4 * interpolation)
    error_bound += rounded_error_floor
    assert error_bound < 0.0177
    assert 20 * math.log10(error_bound) < -35.0
    assert 1.983 + rounded_error_slope + rounded_error_floor < 1.984
    assert 1 + 2 * (1 + 2.0**-24) * residual_l1 + 126 * 2.0**-150 < 2.741

    for rate in (
        44_100.0,
        44_101.0,
        45_000.0,
        46_000.0,
        47_999.0,
        math.nextafter(48_000.0, 0.0),
        48_000.0,
    ):
        weight = (Fraction(48_000) - Fraction.from_float(rate)) / 3_900
        prepared = []
        for coefficient in residual:
            exact = abs(Fraction.from_float(coefficient) * weight)
            rounded = _round_f32(float(exact))
            if exact:
                bits = _U32.unpack(_F32.pack(rounded))[0]
                assert bits >= 0x0080_0000
                lower = Fraction.from_float(_F32.unpack(_U32.pack(bits - 1))[0])
                upper = Fraction.from_float(_F32.unpack(_U32.pack(bits + 1))[0])
                distance = abs(Fraction.from_float(rounded) - exact)
                assert distance <= abs(lower - exact)
                assert distance <= abs(upper - exact)
                if distance == abs(lower - exact) or distance == abs(upper - exact):
                    assert bits & 1 == 0
            prepared.append(math.copysign(rounded, coefficient))

        edge_at_rate = 2 * math.pi * 20_000 / rate
        for sample in range(257):
            omega = edge_at_rate * sample / 256
            z_inverse = cmath.exp(-1j * omega)
            base = sum(
                value * cmath.exp(-1j * omega * index) for index, value in enumerate(residual)
            )
            scaled = sum(
                value * cmath.exp(-1j * omega * index) for index, value in enumerate(prepared)
            )
            response = z_inverse + (1 - z_inverse) * scaled
            expected = z_inverse + (1 - z_inverse) * float(weight) * base
            assert abs(response - expected) <= float(weight) * rounded_error_slope + 1e-12
            desired = cmath.exp(-1j * omega * rate / 48_000)
            assert abs(response - desired) < error_bound + 1e-12


def test_individual_tap_error_does_not_bound_a_near_null_relative_response() -> None:
    rate = 44_100
    omega = 2 * math.pi * 20_000 / rate
    delay = rate / 48_000
    z_inverse = cmath.exp(-1j * omega)
    residual = _first_tap_residual_coefficients()
    residual_response = sum(
        value * cmath.exp(-1j * omega * index) for index, value in enumerate(residual)
    )
    converted_first_tap = z_inverse + (1 - z_inverse) * residual_response
    ideal_first_tap = cmath.exp(-1j * omega * delay)
    middle = _round_f32(math.sqrt(3) / 4)
    target = 0.25 + middle * ideal_first_tap + 0.25 * ideal_first_tap**2
    converted = 0.25 + middle * converted_first_tap + 0.25 * ideal_first_tap**2
    assert 0.25 + middle + 0.25 < 1
    assert abs(target) < 1e-8
    assert abs(converted - target) > 0.005
    assert abs(converted - target) / abs(target) > 100_000


def test_six_point_lagrange_has_bounded_fullband_and_residual_coefficients() -> None:
    for index in range(129):
        mu = Fraction(index, 128)
        weights = _six_point_lagrange(mu)
        residual = _six_point_residual(mu)
        v = mu * (1 - mu)
        distance = min(mu, 1 - mu)
        assert sum(weights) == 1
        assert sum(map(abs, weights)) == 1 + v * (6 + v) / 4
        assert sum(map(abs, weights)) <= Fraction(89, 64)
        assert sum(map(abs, residual)) == (
            distance * (4 - distance**2) * (23 - 3 * distance**2) / 60
        )
        assert sum(map(abs, residual)) <= Fraction(89, 128)
        product = math.prod(abs(mu - node) for node in (-2, -1, 0, 1, 2, 3))
        assert product == v * (2 + v) * (6 + v)
        assert product <= Fraction(225, 64)

    assert _six_point_residual(Fraction(1, 2)) == tuple(
        Fraction(value, 256) for value in (3, -22, -128, 22, -3)
    )
    assert _six_point_residual(Fraction(0)) == (Fraction(0),) * 5


def test_six_point_bounce_candidate_has_conditional_continuous_error_bound() -> None:
    omega_max = 2 * math.pi * 20_000 / 176_400
    rounding_allowance = 2.0**-23 + 10 * 2.0**-150
    error_bound = 5 * omega_max**6 / 1_024 + rounding_allowance
    assert error_bound < 0.00064
    assert 20 * math.log10(error_bound) < -63.8
    assert 89 / 64 + rounding_allowance < 1.391
    assert 1 + 89 / 64 + rounding_allowance < 2.391

    for rate in (
        176_400.0,
        180_000.0,
        192_000.0,
        216_000.0,
        223_456.25,
        352_800.0,
        math.nextafter(384_000.0, 0.0),
        384_000.0,
    ):
        for tap in (1, 2, 15, 16, 31, 47, 63):
            delay = Fraction.from_float(rate) * tap / 48_000
            integer = delay.numerator // delay.denominator
            mu = delay - integer
            assert integer >= 3
            residual = _six_point_residual(mu)
            rounded = tuple(_round_normal_fraction_f32(value) for value in residual)
            assert sum(
                abs(Fraction.from_float(actual) - exact)
                for actual, exact in zip(rounded, residual, strict=True)
            ) <= Fraction(1, 2**24) * sum(map(abs, residual)) + Fraction(5, 2**150)
            if mu == 0:
                assert rounded == (0.0,) * 5

            for sample in range(129):
                omega = 2 * math.pi * 20_000 / rate * sample / 128
                z_inverse = cmath.exp(-1j * omega)
                correction = sum(
                    coefficient * cmath.exp(-1j * omega * (integer - 2 + offset))
                    for offset, coefficient in enumerate(rounded)
                )
                anchor = integer + int(mu > Fraction(1, 2))
                response = cmath.exp(-1j * omega * anchor) + (1 - z_inverse) * correction
                desired = cmath.exp(-1j * omega * float(delay))
                assert abs(response - desired) < error_bound + 1e-12

            for sample in (0.125, -0.125, 1.0, -1.0):
                history = [_round_f32(sample)] * (integer + 4)
                output = history[integer + int(mu > Fraction(1, 2))]
                for offset, coefficient in enumerate(rounded):
                    start = integer - 2 + offset
                    difference = _round_f32(history[start] - history[start + 1])
                    output = _round_f32(output + _round_f32(coefficient * difference))
                assert _U32.unpack(_F32.pack(output))[0] == _U32.unpack(_F32.pack(sample))[0]


def test_six_point_bounce_candidate_tier_history_limits() -> None:
    rate = Fraction(383_999)
    for tap_count, expected in ((16, 131), (32, 251), (48, 379), (64, 507)):
        offsets = [0]
        for tap in (*range(1, tap_count), 16):
            delay = rate * tap / 48_000
            integer = delay.numerator // delay.denominator
            offsets.append(integer if delay.denominator == 1 else integer + 3)
        assert max(offsets) + 1 == expected


def test_six_point_naive_binary64_delay_can_cross_an_integer_boundary() -> None:
    rate = float.fromhex("0x1.699b6db6db6dbp+17")
    exact_delay = Fraction.from_float(rate) * 7 / 48_000
    assert 26 < exact_delay < 27
    assert math.floor(rate * 7 / 48_000) == 27
    assert _bounce_rate_parts(rate, 7) == (26, 1_649_267_441_663_997, 1_649_267_441_664_000)
    assert any(_six_point_residual(exact_delay - 26))


def test_six_point_integer_preparation_matches_exact_rational_reference() -> None:
    rng = random.Random(0xB0_0CE)
    rates = [
        176_400.0,
        180_000.0,
        math.nextafter(192_000.0, 0.0),
        192_000.0,
        216_000.0,
        223_456.25,
        352_800.0,
        383_999.0,
        math.nextafter(384_000.0, 0.0),
        384_000.0,
        float.fromhex("0x1.699b6db6db6dbp+17"),
    ]
    for exponent in (17, 18):
        for _ in range(32):
            bits = ((exponent + 1023) << 52) | rng.getrandbits(52)
            rate = _F64.unpack(_U64.pack(bits))[0]
            if 176_400 <= rate <= 384_000:
                rates.append(rate)
    for rate in rates:
        for tap in range(1, 64):
            integer, remainder, denominator = _bounce_rate_parts(rate, tap)
            exact = Fraction.from_float(rate) * tap / 48_000
            assert exact == integer + Fraction(remainder, denominator)
            assert denominator < 2**51
            numerators, common = _six_point_residual_ratio(remainder, denominator)
            expected = _six_point_residual(exact - integer)
            assert common < 2**262
            for numerator, reference in zip(numerators, expected, strict=True):
                assert abs(numerator) < 2**269
                assert Fraction(numerator, common) == reference
                bits = _round_ratio_f32_bits(numerator, common)
                expected_bits = _U32.unpack(_F32.pack(_round_normal_fraction_f32(reference)))[0]
                assert bits == expected_bits
                assert (abs(numerator) << 149).bit_length() <= 418


def test_exact_ratio_rounding_handles_f32_ties_and_subnormals() -> None:
    assert _round_ratio_f32_bits((1 << 24) + 1, 1 << 24) == 0x3F800000
    assert _round_ratio_f32_bits((1 << 24) + 3, 1 << 24) == 0x3F800002
    assert _round_ratio_f32_bits(1, 1 << 150) == 0
    assert _round_ratio_f32_bits(-1, 1 << 150) == 0x80000000
    assert _round_ratio_f32_bits(3, 1 << 150) == 2
    assert _round_ratio_f32_bits((1 << 24) - 1, 1 << 150) == 0x00800000


def test_six_point_gradual_f32_arithmetic_has_absolute_error_bound() -> None:
    unit = Fraction(1, 2**24)
    underflow = Fraction(1, 2**150)
    gamma = 5 * unit / (1 - 5 * unit)
    residual_l1 = Fraction(89, 128) * (1 + unit) + 5 * underflow
    slope = residual_l1 * (4 * unit + 2 * unit**2) + gamma * (1 + 2 * residual_l1 * (1 + unit) ** 2)
    intercept = (
        residual_l1 * (1 + unit) * underflow
        + 5 * underflow
        + gamma * ((1 + unit) * residual_l1 * underflow + 5 * underflow)
        + 5 * underflow * (1 + unit) ** 5
    )
    assert slope < Fraction(9, 10_000_000)
    assert intercept < 11 * underflow

    def rounded(value: Fraction) -> Fraction:
        bits = _round_ratio_f32_bits(value.numerator, value.denominator)
        return Fraction.from_float(_F32.unpack(_U32.pack(bits))[0])

    rng = random.Random(0xF32)
    edge_values = (100.0, -100.0, 1.0, -1.0, 2.0**-126, -(2.0**-149), 0.0)
    rates = (176_400.0, 180_000.0, math.nextafter(192_000.0, 0.0), 223_456.25, 383_999.0)
    for case in range(300):
        samples = []
        for offset in range(6):
            if case < len(edge_values):
                sample = edge_values[(case + offset) % len(edge_values)]
            else:
                while True:
                    sample = _F32.unpack(_U32.pack(rng.getrandbits(32)))[0]
                    if math.isfinite(sample) and abs(sample) <= 100:
                        break
            samples.append(Fraction.from_float(sample))
        integer, remainder, denominator = _bounce_rate_parts(
            rng.choice(rates), rng.randrange(1, 64)
        )
        assert integer >= 3
        numerators, common = _six_point_residual_ratio(remainder, denominator)
        coefficients = tuple(rounded(Fraction(value, common)) for value in numerators)
        anchor = 2 if 2 * remainder <= denominator else 3
        ideal = samples[anchor]
        actual = samples[anchor]
        for offset, coefficient in enumerate(coefficients):
            difference = samples[offset] - samples[offset + 1]
            ideal += coefficient * difference
            actual = rounded(actual + rounded(coefficient * rounded(difference)))
        peak = max(map(abs, samples))
        assert abs(actual - ideal) <= Fraction(9, 10_000_000) * peak + 11 * underflow


def test_six_point_nearest_anchor_avoids_integer_boundary_cancellation() -> None:
    rate = math.nextafter(192_000.0, 0.0)
    delay = Fraction.from_float(rate) / 48_000
    integer = delay.numerator // delay.denominator
    mu = delay - integer
    weights = _six_point_lagrange(mu)
    prefix = Fraction()
    previous_anchor = []
    for index, value in enumerate(weights[:-1]):
        prefix += value
        previous_anchor.append(prefix - int(index >= 2))
    nearest_anchor = _six_point_residual(mu)
    history = [0.0, 1.0, 1.0, 1.0, 2.0**-25, 2.0**-25, 2.0**-25]
    assert tuple(
        _U32.unpack(_F32.pack(_round_normal_fraction_f32(value)))[0] for value in nearest_anchor
    ) == (0x23BA69DC, 0xA5177603, 0x2608E5BD, 0x259D4952, 0xA40BCF65)

    def evaluate(anchor: int, residual: list[Fraction] | tuple[Fraction, ...]) -> float:
        output = history[anchor]
        for offset, value in enumerate(residual):
            delta = _round_f32(history[integer - 2 + offset] - history[integer - 1 + offset])
            output = _round_f32(output + _round_f32(_round_normal_fraction_f32(value) * delta))
        return output

    assert mu > Fraction(1, 2)
    assert evaluate(integer, previous_anchor) == 0.0
    assert abs(evaluate(integer + 1, nearest_anchor) - history[integer + 1]) < 1e-14


def test_integer_modal_onset_retains_common_conversion_delay() -> None:
    host_rate = 60_000
    ratio = Fraction(host_rate, 48_000)
    onset = 16 * ratio
    first_fir_tap = ratio
    common_delay = 8
    assert onset == 20
    assert first_fir_tap == Fraction(5, 4)

    fir_zero_tap = common_delay
    aligned_onset = onset + common_delay
    shortcut_without_common_delay = onset
    assert aligned_onset == 28
    assert aligned_onset - fir_zero_tap == onset
    assert shortcut_without_common_delay - fir_zero_tap == 12
    assert 4 * common_delay == 32


def test_stage3_stored_radius_bounds_differ_from_sweep_contraction() -> None:
    invalid_stored_endpoint = 0.5
    assert 0.25 <= invalid_stored_endpoint <= 0.9995
    assert not 0.85 <= invalid_stored_endpoint <= 0.9995

    gated_radius = 0.9
    sweep_ceiling = max(0.25, 0.995 - 0.15 * 1.0)
    contracted_radius = min(gated_radius, sweep_ceiling)
    assert 0.25 <= contracted_radius < 0.85 <= gated_radius
    assert math.sqrt(contracted_radius) > contracted_radius


def test_reference_rate_preserves_valid_stage3_angle_without_conversion_clamp() -> None:
    reference_angle = _round_f32(0.00005)
    assert 0 < reference_angle < 0.0001
    incorrectly_clamped_at_reference = max(reference_angle, 0.0001)
    converted_at_96khz = max(reference_angle / 2, 0.0001)
    assert incorrectly_clamped_at_reference != reference_angle
    assert converted_at_96khz == 0.0001


def test_six_point_bounce_candidate_has_early_modal_support() -> None:
    rate = Fraction(176_400)
    onset = 16 * rate / 48_000
    integer = onset.numerator // onset.denominator
    weights = _six_point_lagrange(onset - integer)
    assert onset == Fraction(294, 5)
    assert integer - 2 == 56
    assert weights[0] != 0


def test_six_point_absolute_error_does_not_certify_near_null_fir_response() -> None:
    rate = 180_000.0
    omega = 2 * math.pi * 20_000 / rate
    coefficients = (0.25, _round_f32(math.sqrt(3) / 4), 0.25)
    desired = 0j
    converted = 0j
    for tap, coefficient in enumerate(coefficients):
        delay = Fraction.from_float(rate) * tap / 48_000
        integer = delay.numerator // delay.denominator
        desired += coefficient * cmath.exp(-1j * omega * float(delay))
        if tap == 0:
            converted += coefficient
            continue
        residual = tuple(
            _round_normal_fraction_f32(value) for value in _six_point_residual(delay - integer)
        )
        correction = sum(
            value * cmath.exp(-1j * omega * (integer - 2 + offset))
            for offset, value in enumerate(residual)
        )
        anchor = integer + int(delay - integer > Fraction(1, 2))
        response = cmath.exp(-1j * omega * anchor) + (1 - cmath.exp(-1j * omega)) * correction
        converted += coefficient * response

    bound = 0.00064 * sum(map(abs, coefficients[1:]))
    assert abs(desired) < 1e-8
    assert abs(converted - desired) < bound
    assert abs(converted - desired) / abs(desired) > 100


def test_small_rational_frequency_error_does_not_prove_a_stable_recursion() -> None:
    epsilon = 2.0**-20
    for index in range(2_049):
        omega = math.pi * index / 2_048
        z_inverse = cmath.exp(-1j * omega)
        ratio = (1 - (2 - epsilon) * z_inverse) / (1 - 2 * z_inverse)
        assert abs(ratio - 1) <= epsilon + 1e-15
        assert abs(ratio) <= 1 + epsilon + 1e-15

    previous_input = 0.0
    previous_output = 0.0
    outputs = []
    for index in range(33):
        sample = 1.0 if index == 0 else 0.0
        output = 2 * previous_output + sample - (2 - epsilon) * previous_input
        outputs.append(output)
        previous_input, previous_output = sample, output
    assert outputs[0] == 1.0
    assert outputs[1] == epsilon
    assert outputs[-1] == 2_048.0


def test_modal_input_filter_l1_bound_and_forced_state() -> None:
    coefficients = _coefficients()
    peak_bound = sum(abs(value) for value in coefficients)
    history = [math.copysign(1.0, value) for value in coefficients]
    assert math.isclose(
        sum(
            coefficient * sample for coefficient, sample in zip(coefficients, history, strict=True)
        ),
        peak_bound,
        rel_tol=1e-14,
    )

    radius = 0.9995
    coupling = 2 * math.sqrt(1 - radius * radius)
    pole = radius * cmath.exp(0.37j)
    samples = [1.0 if ((index * 37) % 101) < 50 else -1.0 for index in range(6_000)]
    state = 0j
    for index, _ in enumerate(samples):
        drive = sum(
            coefficient * samples[index - delay]
            for delay, coefficient in enumerate(coefficients)
            if delay <= index
        )
        state = pole * state + coupling * drive
        bound = coupling * peak_bound * (1 - radius ** (index + 1)) / (1 - radius)
        assert abs(state) <= bound + 1e-9
