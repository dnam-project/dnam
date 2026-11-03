"""Independent Stage IV freeze, DC, and finite-precision counterexamples."""

from __future__ import annotations

import cmath
import math
import random
import struct
from fractions import Fraction


def _householder(values: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    subtraction = sum(values, Fraction()) / 2
    return tuple(value - subtraction for value in values)


def _fp16(value: float) -> float:
    return struct.unpack(">e", struct.pack(">e", value))[0]


def _f32(value: float) -> float:
    return struct.unpack(">f", struct.pack(">f", value))[0]


def _f32_bits(value: int) -> float:
    return struct.unpack(">f", struct.pack(">I", value))[0]


def test_saturation_ceiling_needs_a_positive_representable_profile() -> None:
    ceiling = Fraction(197, 200)
    assert Fraction.from_float(_f32(0.985)) > ceiling
    assert Fraction.from_float(_f32_bits(0x3F7C28F5)) <= ceiling
    minimum_subnormal = Fraction(1, 1 << 149)
    maximum_f32 = Fraction((1 << 24) - 1) * (1 << 104)
    assert 2 / (3 * minimum_subnormal) > maximum_f32


def test_feedback_gain_ceiling_applies_to_the_effective_f32_value() -> None:
    limit = Fraction(999, 1000)
    assert Fraction.from_float(_f32(0.999)) > limit
    assert Fraction.from_float(_f32_bits(0x3F7FBE76)) <= limit
    nominal_factor = Fraction(999, 1000)
    additional_factor = Fraction(3, 2)
    assert nominal_factor <= limit
    assert nominal_factor * additional_factor > 1


def test_legacy_and_current_knee_paths_have_different_signal_shapes() -> None:
    gain = Fraction(4, 5)
    knee = Fraction(1, 2)

    def legacy(value: Fraction) -> Fraction:
        return gain * value / (1 + (value / knee) ** 2)

    def current(value: Fraction) -> Fraction:
        return gain * knee * value

    assert legacy(knee) == current(knee) == Fraction(1, 5)
    assert legacy(4 * knee) == Fraction(8, 85) < legacy(knee)
    assert current(4 * knee) == Fraction(4, 5) > current(knee)
    assert gain * knee == Fraction(2, 5) < gain


def _division_free_knee_candidate(value: float, gain: float, inverse_knee: float) -> float:
    ratio = _f32(abs(value) * inverse_knee)
    square = _f32(ratio * ratio)
    if square < 2.0**-126:
        square = 0.0
    denominator = _f32(1.0 + square)
    bits = struct.unpack(">I", struct.pack(">f", denominator))[0]
    exponent = ((bits >> 23) & 0xFF) - 127
    reciprocal = _f32_bits((127 - exponent - 1) << 23)
    for _ in range(5):
        reciprocal = _f32(reciprocal * _f32(2.0 - _f32(denominator * reciprocal)))
    return _f32(_f32(gain * value) * reciprocal)


def _division_free_wide_knee_candidate(value: float, gain: float, inverse_knee: float) -> float:
    ratio = abs(value) * inverse_knee
    denominator = 1.0 + ratio * ratio
    bits = struct.unpack(">Q", struct.pack(">d", denominator))[0]
    exponent = ((bits >> 52) & 0x7FF) - 1023
    reciprocal = struct.unpack(">d", struct.pack(">Q", (1023 - exponent - 1) << 52))[0]
    for _ in range(6):
        reciprocal *= 2.0 - denominator * reciprocal
    return _f32((gain * value) * reciprocal)


def test_division_free_knee_candidate_tracks_the_legacy_shape() -> None:
    rng = random.Random(0xC110)
    cases = [
        (0.0, _f32_bits(0x3F7FBE76), _f32(0.05)),
        (2.0**-80, 2.0**-32, 2.0),
        (1e9, _f32_bits(0x3F7FBE76), _f32(0.05)),
        (-1e9, _f32_bits(0x3F7FBE76), 2.0),
    ]
    for exponent in range(1, 71):
        value = _f32(0.5 * math.sqrt(2.0**exponent - 1))
        if value > 1e9:
            break
        bits = struct.unpack(">I", struct.pack(">f", value))[0]
        for offset in (-1, 0, 1):
            neighbor = _f32_bits(bits + offset)
            cases.extend(((neighbor, _f32_bits(0x3F7FBE76), 0.5), (-neighbor, 2.0**-32, 0.5)))
    for _ in range(5_000):
        value = _f32(math.ldexp(1.0 + rng.random(), rng.randrange(-80, 29)))
        gain = _f32(2.0 ** rng.uniform(-32, -0.01))
        knee = _f32(0.05 + 1.95 * rng.random())
        cases.append((value * rng.choice((-1, 1)), gain, knee))
    for value, gain, knee in cases:
        actual = _division_free_knee_candidate(value, gain, _f32(1.0 / knee))
        exact_value = Fraction.from_float(value)
        exact_gain = Fraction.from_float(gain)
        exact_knee = Fraction.from_float(knee)
        expected = exact_gain * exact_value / (1 + (exact_value / exact_knee) ** 2)
        assert abs(Fraction.from_float(actual) - expected) <= abs(expected) * Fraction(1, 1 << 17)


def test_wide_knee_candidate_handles_f32_subnormal_inputs_and_outputs() -> None:
    rng = random.Random(0xB110)
    maximum_gain = _f32_bits(0x3F7FBE76)
    minimum = _f32_bits(1)
    cases = [
        (0.0, minimum, _f32(0.05)),
        (minimum, maximum_gain, 2.0),
        (-minimum, maximum_gain, _f32(0.05)),
        (1.0, minimum, 1.0),
        (1e9, minimum, _f32(0.05)),
    ]
    for _ in range(5_000):
        value = _f32(math.ldexp(1.0 + rng.random(), rng.randrange(-149, 29)))
        gain = _f32(math.ldexp(1.0 + rng.random(), rng.randrange(-149, -1)))
        knee = _f32(0.05 + 1.95 * rng.random())
        cases.append((value * rng.choice((-1, 1)), gain, knee))
    for value, gain, knee in cases:
        actual = _division_free_wide_knee_candidate(value, gain, 1.0 / knee)
        exact_value = Fraction.from_float(value)
        exact_gain = Fraction.from_float(gain)
        exact_knee = Fraction.from_float(knee)
        expected = exact_gain * exact_value / (1 + (exact_value / exact_knee) ** 2)
        bound = abs(expected) * Fraction(1, 1 << 23) + Fraction(1, 1 << 150)
        assert abs(Fraction.from_float(actual) - expected) <= bound


def test_f32_saturator_can_exceed_its_ceiling_without_an_output_clamp() -> None:
    ceiling = _f32_bits(0x3F7C28F5)
    normalized = _f32_bits(0x3F7FF4D1)
    product = _f32(ceiling * normalized)
    half_square = _f32(_f32(0.5 * normalized) * normalized)
    rounded = _f32(product * _f32(1.5 - half_square))
    assert rounded == _f32_bits(0x3F7C28F6)
    assert Fraction.from_float(ceiling) < Fraction(197, 200) < Fraction.from_float(rounded)
    assert min(max(rounded, -ceiling), ceiling) == ceiling


def test_orthogonal_householder_can_double_a_single_line_peak() -> None:
    vector = (Fraction(-1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2))
    matrix = tuple(
        tuple(Fraction(i == j) - 2 * vector[i] * vector[j] for j in range(4)) for i in range(4)
    )
    for i in range(4):
        for j in range(4):
            assert sum(matrix[k][i] * matrix[k][j] for k in range(4)) == (i == j)
    input_vector = (Fraction(1),) * 4
    output = tuple(sum(row[j] * input_vector[j] for j in range(4)) for row in matrix)
    assert output == (2, 0, 0, 0)
    assert sum(value**2 for value in output) == sum(value**2 for value in input_vector)


def test_input_conditioned_impulse_can_leave_a_persistent_dc_tank_mode() -> None:
    # Equal, one-sample lines at zero rotation isolate the Householder mode.
    pole = Fraction(1023, 1024)
    alpha = (1 + pole) / 2
    input_frame = (Fraction(1), Fraction(-1), Fraction(1), Fraction(-1))
    state = tuple(alpha * sample for sample in input_frame)
    for _ in range(1_000):
        state = _householder(state)
        assert state == (alpha, -alpha, alpha, -alpha)
    assert (state[0] + state[2]) / 2 == alpha
    assert (state[1] + state[3]) / 2 == -alpha


def test_readout_dc_conditioning_does_not_change_the_held_state() -> None:
    pole = 0.9993457198143005
    alpha = (1.0 + pole) / 2.0
    internal_level = 1.0
    previous_input = 0.0
    previous_output = 0.0
    for _ in range(48_000):
        output = alpha * (internal_level - previous_input) + pole * previous_output
        previous_input, previous_output = internal_level, output
    assert internal_level == 1.0
    assert abs(previous_output) < 3e-14


def test_single_readout_filter_does_not_compound_20_hz_loss() -> None:
    pole = 0.9993457198143005
    delay = cmath.exp(-2j * math.pi * 20.0 / 48_000.0)
    response = ((1.0 + pole) / 2.0) * (1.0 - delay) / (1.0 - pole * delay)
    gain = abs(response)
    gain_db = 20.0 * math.log10(gain)
    assert -0.27 < gain_db < -0.25
    assert 20.0 * math.log10(gain**100) < -25.0


def test_input_and_readout_conditioners_alter_20_hz_magnitude_and_phase() -> None:
    pole = 0.9993457198143005
    delay = cmath.exp(-2j * math.pi * 20.0 / 48_000.0)
    response = ((1.0 + pole) / 2.0) * (1.0 - delay) / (1.0 - pole * delay)
    one_pass_error_db = 20.0 * math.log10(abs(response - 1.0))
    two_pass_error_db = 20.0 * math.log10(abs(response * response - 1.0))
    assert -13.0 < one_pass_error_db < -12.0
    assert -7.0 < two_pass_error_db < -6.0


def test_fp16_rewrites_change_energy_even_with_an_exact_orthogonal_rotation() -> None:
    cosine = math.cos(0.1)
    sine = math.sin(0.1)
    left, right = 1.0, 0.0
    reference_left, reference_right = left, right
    for _ in range(1_000):
        left, right = (
            _fp16(cosine * left - sine * right),
            _fp16(sine * left + cosine * right),
        )
        reference_left, reference_right = (
            cosine * reference_left - sine * reference_right,
            sine * reference_left + cosine * reference_right,
        )
    energy = left * left + right * right
    assert abs(reference_left * reference_left + reference_right * reference_right - 1.0) < 1e-12
    assert 0.96 < energy < 0.99
