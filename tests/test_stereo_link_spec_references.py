"""Independent arithmetic examples for stereo-linked detector guarantees."""

from __future__ import annotations

import math
import random
import struct
from fractions import Fraction


def _f32(value: float) -> float:
    return struct.unpack(">f", struct.pack(">f", value))[0]


def _f32_from_bits(value: int) -> float:
    return struct.unpack(">f", struct.pack(">I", value))[0]


def _flush_subnormal(value: float) -> float:
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    if bits & 0x7F80_0000 == 0:
        bits &= 0x8000_0000
    return _f32_from_bits(bits)


def _detector(
    left: Fraction, right: Fraction, key: Fraction, link: Fraction, key_mix: Fraction
) -> tuple[Fraction, Fraction]:
    linked = max(abs(left), abs(right))
    return (
        (1 - key_mix) * ((1 - link) * abs(left) + link * linked) + key_mix * abs(key),
        (1 - key_mix) * ((1 - link) * abs(right) + link * linked) + key_mix * abs(key),
    )


def test_fully_linked_detector_preserves_nonzero_hard_panned_peak() -> None:
    for amplitude in (Fraction(-3, 2), Fraction(-1, 4), Fraction(1, 4), Fraction(3, 2)):
        left, right = _detector(amplitude, Fraction(), Fraction(), Fraction(1), Fraction())
        assert left == abs(amplitude)
        assert right == abs(amplitude)
    assert _detector(Fraction(), Fraction(), Fraction(), Fraction(1), Fraction()) == (0, 0)


def test_full_linking_uses_one_detector_with_any_shared_aux_mix() -> None:
    for key_mix in (Fraction(), Fraction(1, 4), Fraction(1, 2), Fraction(1)):
        left, right = _detector(
            Fraction(3, 4), Fraction(-1, 2), Fraction(1, 4), Fraction(1), key_mix
        )
        assert left == right
        assert left == (1 - key_mix) * Fraction(3, 4) + key_mix * Fraction(1, 4)


def test_partial_linking_can_produce_unequal_detector_levels() -> None:
    left, right = _detector(Fraction(1), Fraction(1, 2), Fraction(), Fraction(1, 2), Fraction())
    assert left == 1
    assert right == Fraction(3, 4)


def test_equal_detector_inputs_do_not_equalize_unequal_envelope_histories() -> None:
    detector = Fraction(1)
    alpha = Fraction(1, 2)
    left_previous = Fraction()
    right_previous = Fraction(1)
    left_envelope = left_previous + alpha * (detector - left_previous)
    right_envelope = right_previous + alpha * (detector - right_previous)
    assert left_envelope == Fraction(1, 2)
    assert right_envelope == 1
    left_gain = 1 / (1 + left_envelope)
    right_gain = 1 / (1 + right_envelope)
    assert left_gain != right_gain


def test_positive_common_gain_preserves_nonzero_vector_direction_in_real_arithmetic() -> None:
    for left, right in (
        (Fraction(1), Fraction()),
        (Fraction(), Fraction(-1)),
        (Fraction(1, 2), Fraction(1)),
        (Fraction(-1), Fraction(1, 4)),
    ):
        gain = Fraction(3, 8)
        output_left, output_right = gain * left, gain * right
        assert output_left * right - output_right * left == 0
        assert output_left * left + output_right * right > 0


def test_zero_gain_does_not_have_a_well_defined_output_direction() -> None:
    left, right, gain = Fraction(1), Fraction(1, 2), Fraction()
    assert (gain * left, gain * right) == (0, 0)


def test_normal_f32_products_have_a_bounded_linked_gain_direction_error() -> None:
    rng = random.Random(681)
    unit_roundoff = 2.0**-24
    bound = math.asin(unit_roundoff / (1 - unit_roundoff))
    assert 3.41e-6 < math.degrees(bound) < 3.42e-6
    for _ in range(10_000):
        left = _f32(rng.uniform(-1, 1))
        right = _f32(rng.uniform(-1, 1))
        gain = _f32(rng.uniform(0.01, 1))
        output_left = _f32(gain * left)
        output_right = _f32(gain * right)
        cross = abs(output_left * right - output_right * left)
        lengths = math.hypot(left, right) * math.hypot(output_left, output_right)
        assert lengths > 0
        assert cross / lengths <= unit_roundoff / (1 - unit_roundoff) + 1e-15


def test_flushing_one_normal_input_product_can_change_direction() -> None:
    left = _f32_from_bits(0x0080_0000)
    right = _f32_from_bits(0x0100_0000)
    output_left = _f32(0.5 * left)
    output_right = _f32(0.5 * right)
    assert output_left == _f32_from_bits(0x0040_0000)
    assert output_right == left
    flushed_left = _flush_subnormal(output_left)
    flushed_right = _flush_subnormal(output_right)
    assert flushed_left == 0.0
    assert flushed_right == left
    change = math.degrees(math.atan2(left, right) - math.atan2(flushed_left, flushed_right))
    assert 26.56 < change < 26.57


def test_gradual_f32_underflow_can_also_change_direction() -> None:
    left = _f32_from_bits(1)
    right = _f32_from_bits(2)
    output_left = _f32(0.5 * left)
    output_right = _f32(0.5 * right)
    assert output_left == 0.0
    assert output_right == left
    change = math.degrees(math.atan2(left, right) - math.atan2(output_left, output_right))
    assert 26.56 < change < 26.57


def test_disabling_cross_coupling_breaks_centered_topology_parity() -> None:
    signal = Fraction(1, 4)
    diagonal = Fraction(1)
    cross = Fraction(1, 2)
    stereo_left = diagonal * signal + cross * signal
    stereo_right = cross * signal + diagonal * signal
    mono_diagonal_only = diagonal * signal
    assert stereo_left == stereo_right
    assert stereo_left / mono_diagonal_only == Fraction(3, 2)
    assert 3.52 < 20 * math.log10(float(stereo_left / mono_diagonal_only)) < 3.53


def test_mono_normaling_keeps_asymmetric_cross_coupling_equations() -> None:
    def stage(left: Fraction, right: Fraction) -> tuple[Fraction, Fraction]:
        return left + Fraction(1, 4) * right, right - Fraction(1, 2) * left

    signal = Fraction(1, 4)
    mono_to_stereo = stage(signal, signal)
    centered_stereo = stage(signal, signal)
    assert mono_to_stereo == centered_stereo
    assert mono_to_stereo[0] != mono_to_stereo[1]


def test_centered_mid_voltage_identity_includes_silence_without_a_decibel_ratio() -> None:
    for signal in (Fraction(-3, 2), Fraction(-1, 4), Fraction(), Fraction(1, 4), Fraction(3, 2)):
        mid = (signal + signal) / 2
        side = (signal - signal) / 2
        assert mid == signal
        assert side == 0
        if signal:
            assert abs(mid) / abs(signal) == 1
        else:
            assert mid == 0
