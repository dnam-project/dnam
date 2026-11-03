"""Independent bitcrusher timing and rounding examples, not engine tests."""

from __future__ import annotations

import math
import random
import struct
from fractions import Fraction

import pytest

_PHASE_SCALE = 1 << 64
_PHASE_MASK = _PHASE_SCALE - 1
_HERTZ_SCALE = 1 << 32


def _f32(value: float) -> float:
    return struct.unpack(">f", struct.pack(">f", value))[0]


def _f32_neighbor(value: float, *, upward: bool) -> float:
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    if value == 0.0:
        bits = 1 if upward else 0x8000_0001
    elif (value > 0) == upward:
        bits += 1
    else:
        bits -= 1
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def _quantize_exact(value: float, bits: int) -> Fraction:
    scale = 1 << (bits - 1)
    scaled = Fraction.from_float(value) * scale
    return Fraction(math.floor(scaled + Fraction(1, 2)), scale)


def _quantize_floor_residual(value: float, bits: int) -> float:
    scale = 1 << (bits - 1)
    scaled = _f32(value * scale)
    truncated = math.trunc(scaled)
    lower = truncated - (scaled < truncated)
    fraction = _f32(scaled - lower)
    result = lower + (fraction >= 0.5)
    return _f32(result / scale)


def _legacy_phase_step(phase: float, increment: float) -> tuple[float, bool]:
    advanced = _f32(phase + increment)
    wrapped = _f32(advanced - 1.0) if advanced >= 1.0 else advanced
    return wrapped, wrapped < phase


def _fixed_phase_step(phase: int, increment: int) -> tuple[int, bool]:
    wrapped = (phase + increment) & _PHASE_MASK
    return wrapped, wrapped < phase


def _encoded_increment(target_hz: float, sample_rate_hz: float) -> tuple[int, bool]:
    if not math.isfinite(target_hz) or not math.isfinite(sample_rate_hz):
        raise ValueError("rates must be finite")
    if not 0 < target_hz <= sample_rate_hz:
        raise ValueError("rate ratio must be in (0, 1]")
    if target_hz == sample_rate_hz:
        return 0, True
    ratio = target_hz * (1.0 / sample_rate_hz)
    increment = round(ratio * _PHASE_SCALE)
    if not 0 < increment < _PHASE_SCALE:
        raise ValueError("rate is not representable")
    return increment, False


def _hertz_phase_parameters(target_hz: float, sample_rate_hz: float) -> tuple[int, int]:
    if not math.isfinite(target_hz) or not math.isfinite(sample_rate_hz):
        raise ValueError("rates must be finite")
    if sample_rate_hz <= 0 or not 0 <= target_hz <= sample_rate_hz:
        raise ValueError("rate ratio must be in [0, 1]")
    denominator = round(Fraction.from_float(sample_rate_hz) * _HERTZ_SCALE)
    increment = round(Fraction.from_float(target_hz) * _HERTZ_SCALE)
    if (
        denominator <= 0
        or (target_hz > 0 and increment == 0)
        or increment > denominator
        or 2 * denominator > 1 << 64
    ):
        raise ValueError("rate is not representable")
    return increment, denominator


def _hertz_phase_step(phase: int, increment: int, denominator: int) -> tuple[int, bool]:
    advanced = phase + increment
    latch = advanced >= denominator
    return advanced - (denominator if latch else 0), latch


def _rescale_phase_with_prepared_reciprocal(
    phase: int, old_denominator: int, new_denominator: int, reciprocal: int
) -> int:
    numerator = phase * new_denominator
    low, high = numerator & _PHASE_MASK, numerator >> 64
    low_product, high_product = low * reciprocal, high * reciprocal
    middle_sum = (low_product >> 64) + (high_product & _PHASE_MASK)
    top = (high_product >> 64) + (middle_sum >> 64)
    middle = middle_sum & _PHASE_MASK
    quotient = (top << 24) | (middle >> 40)
    remainder = numerator - quotient * old_denominator
    if remainder >= old_denominator:
        quotient += 1
        remainder -= old_denominator
    if 2 * remainder > old_denominator or (2 * remainder == old_denominator and quotient & 1):
        quotient += 1
    return min(quotient, new_denominator - 1)


def test_full_rate_is_not_a_wrap_comparison() -> None:
    for phase in (0.0, 0.25, 0.5):
        wrapped, legacy_latch = _legacy_phase_step(phase, 1.0)
        assert wrapped == phase
        assert not legacy_latch


@pytest.mark.parametrize("phase", [0, 1, _PHASE_SCALE // 2, _PHASE_MASK - 1, _PHASE_MASK])
@pytest.mark.parametrize("increment", [0, 1, _PHASE_SCALE // 2, _PHASE_MASK])
def test_fixed_phase_carry_matches_integer_overflow(phase: int, increment: int) -> None:
    wrapped, latch = _fixed_phase_step(phase, increment)
    exact = phase + increment
    assert wrapped == exact % _PHASE_SCALE
    assert latch == (exact >= _PHASE_SCALE)


def test_fixed_phase_full_rate_latches_every_sample_without_changing_phase() -> None:
    increment, full_rate = _encoded_increment(48_000.0, 48_000.0)
    phase = _PHASE_SCALE // 7
    held = Fraction(0)
    signals = [0.25, -0.25, 0.5, -0.5]
    output = []
    for signal in signals:
        old_phase = phase
        if full_rate:
            latch = True
        else:
            phase, latch = _fixed_phase_step(phase, increment)
        if latch:
            held = _quantize_exact(signal, 4)
        assert phase == old_phase
        output.append(held)
    assert output == [Fraction.from_float(signal) for signal in signals]


def test_fixed_phase_event_count_and_rate_error() -> None:
    ratio = Fraction(40, 147)
    increment = round(ratio * _PHASE_SCALE)
    assert abs(Fraction(increment, _PHASE_SCALE) - ratio) <= Fraction(1, 2 * _PHASE_SCALE)
    phase = 0
    events = 0
    for count in range(1, 14_701):
        phase, latch = _fixed_phase_step(phase, increment)
        events += latch
        assert events == (count * increment) // _PHASE_SCALE
    assert abs(events - math.floor(count * ratio)) <= 1


def test_fixed_phase_rounding_can_move_a_rational_boundary_by_one_sample() -> None:
    increment = round(Fraction(_PHASE_SCALE, 3))
    phase = 0
    latches = []
    for _ in range(4):
        phase, latch = _fixed_phase_step(phase, increment)
        latches.append(latch)
    assert latches == [False, False, False, True]
    assert math.floor(Fraction(3, 3)) == 1


@pytest.mark.parametrize(
    ("target_hz", "period"),
    [(16_000.0, 3), (12_000.0, 4), (48_000.0, 1)],
)
def test_hertz_phase_latches_at_exact_common_rate_boundaries(target_hz: float, period: int) -> None:
    increment, denominator = _hertz_phase_parameters(target_hz, 48_000.0)
    phase = 0
    for sample in range(1, period * 8 + 1):
        phase, latch = _hertz_phase_step(phase, increment, denominator)
        assert latch == (sample % period == 0)
    assert phase == 0


def test_hertz_phase_preserves_phase_for_rate_changes_at_fixed_sample_rate() -> None:
    phase = 0
    events = 0
    total = 0
    for target_hz in (12_000.0, 12_000.0, 24_000.0, 16_000.0, 48_000.0, 24_000.0):
        increment, denominator = _hertz_phase_parameters(target_hz, 48_000.0)
        phase, latch = _hertz_phase_step(phase, increment, denominator)
        total += increment
        events += latch
        assert phase == total % denominator
        assert events == total // denominator


@pytest.mark.parametrize("target_hz", [0.0, -0.0])
def test_hertz_phase_zero_rate_holds_previous_sample_and_phase(target_hz: float) -> None:
    increment, denominator = _hertz_phase_parameters(target_hz, 48_000.0)
    assert increment == 0
    held = Fraction(0)
    phase = 0
    for signal in (0.25, 0.5):
        phase, latch = _hertz_phase_step(phase, increment, denominator)
        if latch:
            held = _quantize_exact(signal, 4)
        assert not latch
        assert phase == 0
        assert held == 0

    increment, _ = _hertz_phase_parameters(12_000.0, 48_000.0)
    for signal in (0.25, 0.25, 0.25, 0.5, 0.75, 0.75):
        phase, latch = _hertz_phase_step(phase, increment, denominator)
        if latch:
            held = _quantize_exact(signal, 4)
    assert held == Fraction(1, 2)
    assert phase == denominator // 2

    increment, _ = _hertz_phase_parameters(target_hz, 48_000.0)
    for _ in range(8):
        phase, latch = _hertz_phase_step(phase, increment, denominator)
        assert not latch
        assert phase == denominator // 2
        assert held == Fraction(1, 2)

    increment, _ = _hertz_phase_parameters(12_000.0, 48_000.0)
    phase, latch = _hertz_phase_step(phase, increment, denominator)
    assert not latch
    phase, latch = _hertz_phase_step(phase, increment, denominator)
    assert latch
    assert phase == 0


@pytest.mark.parametrize(
    ("target_hz", "sample_rate_hz"),
    [(-1.0, 48_000.0), (48_001.0, 48_000.0), (0.0, 0.0), (1.0, 0.0)],
)
def test_hertz_phase_rejects_invalid_rate_domain(target_hz: float, sample_rate_hz: float) -> None:
    with pytest.raises(ValueError, match="rate ratio"):
        _hertz_phase_parameters(target_hz, sample_rate_hz)


@pytest.mark.parametrize("old_rate", [44_100.0, 48_000.0, 96_000.0, 192_000.0, 384_000.0])
@pytest.mark.parametrize("new_rate", [44_100.0, 48_000.0, 96_000.0, 192_000.0, 384_000.0])
def test_hertz_phase_rate_transition_matches_exact_rational_rounding(
    old_rate: float, new_rate: float
) -> None:
    _, old_denominator = _hertz_phase_parameters(1.0, old_rate)
    _, new_denominator = _hertz_phase_parameters(1.0, new_rate)
    reciprocal = (1 << 104) // old_denominator
    assert 0 < reciprocal < 1 << 64
    generator = random.Random((int(old_rate) << 32) | int(new_rate))
    phases = [
        0,
        1,
        old_denominator // 3,
        old_denominator // 2,
        old_denominator - 1,
        *(generator.randrange(old_denominator) for _ in range(64)),
    ]
    for phase in phases:
        result = _rescale_phase_with_prepared_reciprocal(
            phase, old_denominator, new_denominator, reciprocal
        )
        exact = min(round(Fraction(phase * new_denominator, old_denominator)), new_denominator - 1)
        assert result == exact
        assert abs(Fraction(result, new_denominator) - Fraction(phase, old_denominator)) < Fraction(
            1, new_denominator
        )


def test_hertz_phase_rate_transition_rounds_ties_to_even() -> None:
    _, old_denominator = _hertz_phase_parameters(1.0, 48_000.0)
    _, new_denominator = _hertz_phase_parameters(1.0, 44_100.0)
    reciprocal = (1 << 104) // old_denominator
    assert (
        _rescale_phase_with_prepared_reciprocal(80, old_denominator, new_denominator, reciprocal)
        == 74
    )
    assert (
        _rescale_phase_with_prepared_reciprocal(240, old_denominator, new_denominator, reciprocal)
        == 220
    )


def test_hertz_phase_rate_transition_corrects_underestimated_quotient() -> None:
    _, denominator = _hertz_phase_parameters(1.0, 48_000.0)
    reciprocal = (1 << 104) // denominator
    phase = 1
    numerator = phase * denominator
    assert (numerator * reciprocal) >> 104 == 0
    assert _rescale_phase_with_prepared_reciprocal(phase, denominator, denominator, reciprocal) == 1


def test_hertz_phase_rate_transition_preserves_an_imminent_latch() -> None:
    _, old_denominator = _hertz_phase_parameters(1.0, 384_000.0)
    increment, new_denominator = _hertz_phase_parameters(1_000.0, 44_100.0)
    reciprocal = (1 << 104) // old_denominator
    phase = _rescale_phase_with_prepared_reciprocal(
        old_denominator - 1, old_denominator, new_denominator, reciprocal
    )
    assert phase == new_denominator - 1
    _, latch = _hertz_phase_step(phase, increment, new_denominator)
    assert latch


@pytest.mark.parametrize(
    "sample_rate_hz",
    [44_100.0, 48_000.0, math.nextafter(48_000.0, math.inf), 96_000.0, 192_000.0, 384_000.0],
)
@pytest.mark.parametrize("target_hz", [0.0, 1.0, 999.0, 999.999999999, 12_000.0, 44_099.0])
def test_hertz_phase_frequency_error_bound(sample_rate_hz: float, target_hz: float) -> None:
    increment, denominator = _hertz_phase_parameters(target_hz, sample_rate_hz)
    actual = Fraction.from_float(sample_rate_hz) * Fraction(increment, denominator)
    assert abs(actual - Fraction.from_float(target_hz)) <= Fraction(1, _HERTZ_SCALE)
    assert 2 * denominator < 1 << 64


@pytest.mark.parametrize(
    "target_hz",
    [2**-9, _f32_neighbor(2**-9, upward=True), 1.0, _f32(16_000.1), 384_000.0],
)
def test_hertz_scale_exactly_represents_normal_float32_target_rates(target_hz: float) -> None:
    scaled = Fraction.from_float(_f32(target_hz)) * _HERTZ_SCALE
    assert scaled.denominator == 1


def test_hertz_phase_reports_unrepresentably_small_positive_rate() -> None:
    with pytest.raises(ValueError, match="rate is not representable"):
        _hertz_phase_parameters(2**-34, 48_000.0)


@pytest.mark.parametrize("sample_rate_hz", [44_100.0, 48_000.0, 96_000.0, 192_000.0, 384_000.0])
@pytest.mark.parametrize("target_hz", [1.0, 999.0, 12_000.0, 44_099.0])
def test_encoded_rate_error_bound(sample_rate_hz: float, target_hz: float) -> None:
    increment, full_rate = _encoded_increment(target_hz, sample_rate_hz)
    assert not full_rate
    ideal = Fraction.from_float(target_hz) / Fraction.from_float(sample_rate_hz)
    represented = Fraction(increment, _PHASE_SCALE)
    unit_roundoff = Fraction(1, 1 << 53)
    bound = ideal * (2 * unit_roundoff + unit_roundoff**2) + Fraction(1, 2 * _PHASE_SCALE)
    assert abs(represented - ideal) <= bound


@pytest.mark.parametrize(
    ("target_hz", "sample_rate_hz"),
    [(0.0, 48_000.0), (-1.0, 48_000.0), (48_001.0, 48_000.0), (1.0, 0.0)],
)
def test_invalid_rate_candidates_are_rejected(target_hz: float, sample_rate_hz: float) -> None:
    with pytest.raises(ValueError, match="rate ratio"):
        _encoded_increment(target_hz, sample_rate_hz)


@pytest.mark.parametrize("target_hz", [math.nan, math.inf, -math.inf])
def test_nonfinite_rate_candidates_are_rejected(target_hz: float) -> None:
    with pytest.raises(ValueError, match="rates must be finite"):
        _encoded_increment(target_hz, 48_000.0)


def test_rate_below_fixed_phase_resolution_is_rejected() -> None:
    with pytest.raises(ValueError, match="rate is not representable"):
        _encoded_increment(1e-300, 48_000.0)


def test_legacy_float32_half_offset_can_cross_a_quantizer_boundary() -> None:
    below_half = _f32_neighbor(0.5, upward=False)
    sample = _f32(below_half / 2)
    assert Fraction.from_float(below_half) < Fraction(1, 2)
    assert _f32(below_half + 0.5) == 1.0
    assert math.floor(_f32(below_half + 0.5)) == 1
    assert _quantize_exact(sample, 2) == 0


def test_truncating_conversion_is_not_floor_for_negative_values() -> None:
    scaled = -2.0
    assert math.floor(scaled + 0.5) == -2
    assert math.trunc(scaled + 0.5) == -1


def test_subnormal_inputs_quantize_to_zero_in_the_tested_profile() -> None:
    for bits in (2, 8, 16):
        for sample in (_f32_neighbor(0.0, upward=False), _f32_neighbor(0.0, upward=True)):
            assert _quantize_exact(sample, bits) == 0
            assert _quantize_floor_residual(sample, bits) == 0


def test_ungrouped_straight_through_expression_can_change_float32_forward() -> None:
    sample = _f32_neighbor(1.0, upward=True)
    quantized = float(_quantize_exact(sample, 16))
    legacy = _f32(_f32(quantized + sample) - sample)
    grouped = _f32(quantized + _f32(sample - sample))
    assert quantized == 1.0
    assert legacy != quantized
    assert grouped == quantized


def test_fourier_warmup_surrogate_changes_forward_and_can_reverse_gradient() -> None:
    delta = 0.125
    sample = delta / 4
    smooth = sample - delta / math.pi * sum(
        (-1) ** (term - 1) / term * math.sin(2 * math.pi * term * sample / delta)
        for term in range(1, 5)
    )
    assert _quantize_exact(sample, 4) == 0
    assert smooth > 0

    sample = delta / 8
    derivative = 1 - 2 * sum(
        (-1) ** (term - 1) * math.cos(2 * math.pi * term * sample / delta) for term in range(1, 5)
    )
    assert derivative == pytest.approx(-1.0, abs=1e-14)


@pytest.mark.parametrize("bits", [2, 3, 4, 8, 12, 16])
def test_floor_residual_matches_exact_half_up_at_float32_neighbors(bits: int) -> None:
    scale = 1 << (bits - 1)
    indices = {-100 * scale, -255, -31, -3, -1, 0, 1, 3, 31, 255, 100 * scale - 1}
    for index in indices:
        boundary = _f32((index + 0.5) / scale)
        for sample in (
            _f32_neighbor(boundary, upward=False),
            boundary,
            _f32_neighbor(boundary, upward=True),
        ):
            if abs(sample) <= 100:
                result = _quantize_floor_residual(sample, bits)
                assert Fraction.from_float(result) == _quantize_exact(sample, bits)
