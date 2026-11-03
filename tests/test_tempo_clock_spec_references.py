"""Numerical examples for tempo-clock design, not an engine implementation."""

from __future__ import annotations

import itertools
import math
import random
import statistics
import struct
from fractions import Fraction

import pytest

_Q32 = 1 << 32
_Q64 = 1 << 64
_REM3 = (0, 1, 2) * 8
_SUBDIVISION_RATIOS = (
    (3, 1),
    (2, 1),
    (4, 3),
    (3, 2),
    (1, 1),
    (2, 3),
    (3, 4),
    (1, 2),
    (1, 3),
    (3, 8),
    (1, 4),
    (1, 6),
)


def _round_positive_ratio(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    if 2 * remainder > denominator or (2 * remainder == denominator and quotient & 1):
        quotient += 1
    return quotient


def _host_frame_increment(rate: float) -> int:
    if not math.isfinite(rate) or not 44_100 <= rate <= 96_000:
        raise ValueError("unsupported host rate")
    bits = struct.unpack(">Q", struct.pack(">d", rate))[0]
    exponent = ((bits >> 52) & 0x7FF) - 1023
    significand = (1 << 52) | (bits & ((1 << 52) - 1))
    increment = _round_positive_ratio(1 << (116 - exponent), significand)
    assert 0 < increment < _Q64
    return increment


def _subframe_offset(increment: int, index: int) -> int:
    product = index * increment
    quotient, remainder = product >> 2, product & 3
    if remainder > 2 or (remainder == 2 and quotient & 1):
        quotient += 1
    return quotient


def _checked_time_add(clock: int, ticks: int) -> int:
    if not 0 <= clock < 1 << 128 or ticks < 0 or clock + ticks >= 1 << 128:
        raise ValueError("invalid or overflowing time")
    return clock + ticks


def _elapsed_same_epoch(start: tuple[int, int], end: tuple[int, int]) -> int:
    if start[0] != end[0] or end[1] < start[1]:
        raise ValueError("incomparable timestamps")
    return end[1] - start[1]


def _subdivision_phase(
    beat: int, within_beat: Fraction, ratio: tuple[int, int], offset: Fraction = Fraction()
) -> Fraction:
    if not 0 <= within_beat < 1:
        raise ValueError("invalid beat phase")
    numerator, denominator = ratio
    return (denominator * ((beat % numerator) + within_beat) / numerator + offset) % 1


def _wrap_phase(value: Fraction) -> Fraction:
    return (value + Fraction(1, 2)) % 1 - Fraction(1, 2)


def _smooth_relock_tick(
    previous: Fraction, advance: Fraction, target: Fraction, max_correction: Fraction
) -> Fraction:
    if not 0 < max_correction < Fraction(1, 2):
        raise ValueError("invalid correction bound")
    predicted = (previous + advance) % 1
    error = _wrap_phase(target - predicted)
    correction = max(-max_correction, min(error, max_correction))
    return (predicted + correction) % 1


def _fixed_subdivision_phase(
    beat_mod12: int, within_beat: int, ratio: tuple[int, int], offset: int = 0
) -> int:
    if not 0 <= beat_mod12 < 12 or not 0 <= within_beat < _Q32 or not 0 <= offset < _Q32:
        raise ValueError("invalid fixed-point phase")
    if ratio not in _SUBDIVISION_RATIOS:
        raise ValueError("unsupported subdivision")
    numerator, denominator = ratio
    if numerator == 1:
        beat_remainder = 0
    elif numerator == 2:
        beat_remainder = beat_mod12 & 1
    elif numerator == 3:
        beat_remainder = _REM3[beat_mod12]
    else:
        beat_remainder = beat_mod12 & 3

    product = denominator * ((beat_remainder << 32) + within_beat)
    high, low = product >> 32, product & (_Q32 - 1)
    if numerator == 1:
        phase = low
    elif numerator == 2:
        phase = (((high & 1) << 32) + low) >> 1
    elif numerator == 3:
        remainder = _REM3[high]
        phase = remainder * 0x55555555 + ((low + remainder) * 0xAAAAAAAB >> 33)
    else:
        phase = (((high & 3) << 32) + low) >> 2
    return (phase + offset) & (_Q32 - 1)


def _q64_tick(beat_mod12: int, fraction: int, step: int) -> tuple[int, int]:
    if not 0 <= beat_mod12 < 12 or not 0 <= fraction < _Q64 or not 0 <= step < _Q64:
        raise ValueError("invalid fixed-point beat")
    next_fraction = (fraction + step) & (_Q64 - 1)
    if next_fraction < fraction:
        beat_mod12 += 1
        if beat_mod12 == 12:
            beat_mod12 = 0
    return beat_mod12, next_fraction


def _f32(value: float) -> float:
    return struct.unpack(">f", struct.pack(">f", value))[0]


def _candidate_bpm_reciprocal(bpm: float, steps: int) -> float:
    bits = struct.unpack(">I", struct.pack(">f", bpm))[0]
    estimate = struct.unpack(">f", struct.pack(">I", 0x7EF127EA - bits))[0]
    for _ in range(steps):
        estimate = _f32(estimate * _f32(2.0 - _f32(bpm * estimate)))
    return estimate


def _candidate_filter(
    event_times: list[float], initial_period: float, time_constant: float
) -> tuple[list[float], list[float]]:
    period = initial_period
    phase_time = event_times[0]
    periods = [period]
    phases = [phase_time]
    for previous, current in itertools.pairwise(event_times):
        gain = -math.expm1(-period / time_constant)
        predicted = phase_time + period
        phase_time = predicted + gain * (current - predicted)
        period += gain * (current - previous - period)
        periods.append(period)
        phases.append(phase_time)
    return periods, phases


def test_fixed_per_event_gain_has_different_midi_and_tap_time_constants() -> None:
    gain = 0.08
    midi_interval = 0.5 / 24
    tap_interval = 0.5
    midi_tau = -midi_interval / math.log1p(-gain)
    tap_tau = -tap_interval / math.log1p(-gain)
    assert 0.24 < midi_tau < 0.26
    assert 5.9 < tap_tau < 6.1
    assert abs(tap_tau / midi_tau - 24) < 1e-12


def test_single_gap_cannot_always_distinguish_jitter_from_a_missing_pulse() -> None:
    pulse_period = 60 / (360 * 24)
    observed_interval = 0.0105
    jitter_limit = 0.002
    valid_jitter_delta = observed_interval - pulse_period
    missing_jitter_delta = observed_interval - 2 * pulse_period
    assert abs(valid_jitter_delta / 2) <= jitter_limit
    assert abs(missing_jitter_delta / 2) <= jitter_limit
    assert pulse_period + valid_jitter_delta == observed_interval
    assert 2 * pulse_period + missing_jitter_delta == observed_interval


def test_event_rate_candidate_exposes_jitter_and_step_response_tradeoff() -> None:
    pulse_period = 0.5 / 24
    rng = random.Random(1927)
    events = [n * pulse_period + rng.uniform(-0.002, 0.002) for n in range(5000)]
    raw_periods = [later - earlier for earlier, later in itertools.pairwise(events)]
    periods, phases = _candidate_filter(events, pulse_period, 0.5)
    raw_sigma = statistics.pstdev(raw_periods[100:])
    period_sigma = statistics.pstdev(periods[100:])
    phase_sigma = statistics.pstdev(phases[n] - n * pulse_period for n in range(100, len(events)))
    assert 0.0015 < raw_sigma < 0.0018
    assert 0.00004 < period_sigma < 0.00006
    assert 0.0002 < phase_sigma < 0.0003

    switch = 120
    event_times = [0.0]
    for n in range(1, switch + 240):
        event_times.append(event_times[-1] + (0.5 if n < switch else 0.6) / 24)
    periods, phases = _candidate_filter(event_times, pulse_period, 0.5)
    target = 0.6 / 24
    assert all(periods[n] <= periods[n + 1] <= target for n in range(switch, len(periods) - 1))
    settling = next(
        (n - switch) * target
        for n in range(switch, len(periods))
        if all(abs(value - target) <= 0.01 * target for value in periods[n:])
    )
    max_phase_lag = max(abs(phases[n] - event_times[n]) for n in range(switch, len(periods)))
    assert 1.4 < settling < 1.6
    assert 0.03 < max_phase_lag < 0.04


def test_subdivision_phase_uses_whole_beats_and_all_twelve_periods() -> None:
    positions = (Fraction(0), Fraction(1, 4), Fraction(3, 5), Fraction(999, 1000))
    for numerator, denominator in _SUBDIVISION_RATIOS:
        for beat in range(-24, 25):
            for within_beat in positions:
                target = _subdivision_phase(beat, within_beat, (numerator, denominator))
                direct = ((beat + within_beat) * denominator / numerator) % 1
                assert target == direct
                assert target == _subdivision_phase(
                    beat + 12, within_beat, (numerator, denominator)
                )
                next_beat = _subdivision_phase(beat + 1, Fraction(0), (numerator, denominator))
                before_beat = _subdivision_phase(
                    beat, Fraction(999, 1000), (numerator, denominator)
                )
                assert _wrap_phase(next_beat - before_beat) == Fraction(
                    denominator, 1000 * numerator
                )


def test_twelve_beat_counter_is_minimal_for_all_subdivisions() -> None:
    for candidate in range(1, 12):
        assert any(
            _subdivision_phase(0, Fraction(0), ratio)
            != _subdivision_phase(candidate, Fraction(0), ratio)
            for ratio in _SUBDIVISION_RATIOS
        )


def test_quarter_phase_alone_cannot_locate_half_note_or_triplet_phase() -> None:
    for ratio, expected in (((2, 1), Fraction(1, 2)), ((4, 3), Fraction(3, 4))):
        assert _subdivision_phase(0, Fraction(0), ratio) == 0
        assert _subdivision_phase(1, Fraction(0), ratio) == expected


def test_subdivision_phase_repeats_after_its_period_with_offset() -> None:
    for numerator, denominator in _SUBDIVISION_RATIOS:
        for position in (Fraction(-17, 4), Fraction(3, 5), Fraction(121, 8)):
            after = position + Fraction(numerator, denominator)
            beat = position.numerator // position.denominator
            after_beat = after.numerator // after.denominator
            offset = Fraction(2, 7)
            assert _subdivision_phase(
                beat, position - beat, (numerator, denominator), offset
            ) == _subdivision_phase(
                after_beat, after - after_beat, (numerator, denominator), offset
            )


def test_predicted_phase_feedback_does_not_nudge_an_aligned_generator() -> None:
    gain = Fraction(1, 50)
    for previous, increment in (
        (Fraction(1, 7), Fraction(1, 16)),
        (Fraction(99, 100), Fraction(1, 20)),
    ):
        predicted = (previous + increment) % 1
        target = predicted
        old_candidate = (predicted + gain * _wrap_phase(target - previous)) % 1
        corrected = (predicted + gain * _wrap_phase(target - predicted)) % 1
        assert old_candidate != target
        assert corrected == target


def test_cross_product_phase_error_has_an_antipodal_blind_spot() -> None:
    current = (1, 0)
    target = (-1, 0)
    cross = target[1] * current[0] - target[0] * current[1]
    dot = target[0] * current[0] + target[1] * current[1]
    assert cross == 0
    assert dot == -1
    assert _wrap_phase(Fraction(1, 2)) == -Fraction(1, 2)


def test_immediate_half_cycle_relock_cannot_preserve_the_old_modulation_value() -> None:
    old_value = Fraction(1)
    target_value = Fraction(-1)
    assert abs(target_value - old_value) == 2
    assert (old_value + target_value) / 2 == 0


def test_start_position_reference_uses_frame_zero_target_without_relock() -> None:
    position = Fraction(-5, 4)
    offset = Fraction(1, 7)
    beat = position.numerator // position.denominator
    target = _subdivision_phase(beat, position - beat, (2, 1), offset)
    stale_phase = Fraction(1, 8)
    assert target == Fraction(29, 56)
    assert target != stale_phase
    assert _smooth_relock_tick(stale_phase, Fraction(), target, Fraction(1, 16)) != target


def test_smooth_relock_tracks_a_moving_target_on_the_shortest_arc() -> None:
    initial = Fraction(1, 10)
    advance = Fraction(1, 64)
    limit = Fraction(1, 16)
    for direction in (-1, 1):
        phase = initial
        target = (initial + direction * Fraction(1, 3)) % 1
        error = Fraction(1, 3)
        for _ in range(6):
            target = (target + advance) % 1
            predicted = (phase + advance) % 1
            phase = _smooth_relock_tick(phase, advance, target, limit)
            correction = _wrap_phase(phase - predicted)
            assert abs(correction) <= limit
            assert correction * direction > 0
            error = max(Fraction(), error - limit)
            assert abs(_wrap_phase(target - phase)) == error
        assert phase == target


def test_smooth_relock_wraps_and_breaks_half_cycle_ties_negatively() -> None:
    limit = Fraction(1, 8)
    assert _smooth_relock_tick(Fraction(9, 10), Fraction(), Fraction(1, 20), limit) == Fraction(
        1, 40
    )
    phase = Fraction()
    for expected in (Fraction(7, 8), Fraction(3, 4), Fraction(5, 8), Fraction(1, 2)):
        phase = _smooth_relock_tick(phase, Fraction(), Fraction(1, 2), limit)
        assert phase == expected


def test_jump_target_is_advanced_from_event_to_control_boundary() -> None:
    event_position = Fraction(2)
    boundary_position = event_position + Fraction(1, 16)
    event_target = _subdivision_phase(2, Fraction(), (2, 1))
    boundary_target = _subdivision_phase(2, boundary_position - 2, (2, 1))
    assert event_target == 0
    assert boundary_target == Fraction(1, 32)
    assert boundary_target != event_target


def test_immediate_phase_snap_and_parameter_ramp_are_distinct() -> None:
    phase_after_boundary = Fraction(1, 2)
    old_parameter = Fraction(1)
    target_parameter = Fraction(-1)
    increment = (target_parameter - old_parameter) / 16
    samples = [old_parameter + n * increment for n in range(1, 17)]
    assert phase_after_boundary == Fraction(1, 2)
    assert samples[0] == Fraction(7, 8)
    assert samples[-1] == target_parameter
    assert all(samples[index] - samples[index - 1] == increment for index in range(1, 16))


def test_one_dirty_bit_cannot_preserve_timestamped_pulse_multiplicity() -> None:
    def coalesced(events: tuple[int, ...]) -> tuple[int, int]:
        dirty = 0
        last_timestamp = 0
        for timestamp in events:
            dirty |= 0x02
            last_timestamp = timestamp
        return dirty, last_timestamp

    assert coalesced((10,)) == coalesced((0, 10))
    assert (10,) != (0, 10)


def test_atomic_exchange_preserves_a_dirty_update_after_the_consumer_snapshot() -> None:
    dirty = 0x02
    observed = dirty
    dirty |= 0x02
    dirty = 0
    assert observed == 0x02
    assert dirty == 0

    dirty = 0x02
    observed, dirty = dirty, 0
    dirty |= 0x02
    assert observed == 0x02
    assert dirty == 0x02


def test_bounded_event_drain_must_keep_a_hint_for_the_remaining_backlog() -> None:
    pending = [0, 10]
    dirty = 0x02
    observed, dirty = dirty, 0
    assert observed == 0x02
    assert pending.pop(0) == 0
    assert pending
    assert dirty == 0
    dirty |= 0x02
    assert pending == [10]
    assert dirty == 0x02


def test_clamped_bpm_is_not_locked_to_the_original_host_tempo() -> None:
    sample_rate = 48_000
    requested_period = Fraction(60 * sample_rate, 400)
    clipped_period = Fraction(60 * sample_rate, 360)
    assert requested_period == 7200
    assert clipped_period == 8000
    assert clipped_period / requested_period == Fraction(10, 9)


def test_supported_tempo_endpoints_and_subdivision_periods() -> None:
    slow_quarter = Fraction(60, 20)
    fast_quarter = Fraction(60, 400)
    assert slow_quarter == 3
    assert fast_quarter == Fraction(3, 20)
    assert 3 * slow_quarter == 9
    assert fast_quarter / 6 == Fraction(1, 40)
    assert slow_quarter / 24 == Fraction(1, 8)
    assert fast_quarter / 24 == Fraction(1, 160)
    for sample_rate in (44_100, 48_000, 96_000, 176_400, 192_000, 384_000):
        assert slow_quarter * sample_rate == 3 * sample_rate
        assert fast_quarter * sample_rate == Fraction(3 * sample_rate, 20)


def test_candidate_reciprocal_covers_sampled_supported_tempo_range() -> None:
    for index in range(10_001):
        bpm = _f32(20 + index * 380 / 10_000)
        reciprocal = _candidate_bpm_reciprocal(bpm, 3)
        assert abs(bpm * reciprocal - 1) < 2e-7


def test_two_step_reciprocal_can_miss_a_sample_at_low_tempo() -> None:
    bpm = _f32(30.108)
    two_step = _candidate_bpm_reciprocal(bpm, 2)
    three_step = _candidate_bpm_reciprocal(bpm, 3)
    true_period = 60 * 48_000 / bpm
    two_step_error = abs(60 * 48_000 * two_step - true_period)
    three_step_error = abs(60 * 48_000 * three_step - true_period)
    assert 0.99 < two_step_error < 1.01
    assert three_step_error < 0.02


def test_tap_intervals_from_different_rates_need_a_common_timebase() -> None:
    first = Fraction(24_000, 48_000)
    second = Fraction(48_000, 96_000)
    actual_bpm = 60 / ((first + second) / 2)
    raw_sample_average = Fraction(24_000 + 48_000, 2)
    reinterpreted_bpm = 60 * 96_000 / raw_sample_average
    assert actual_bpm == 120
    assert reinterpreted_bpm == 160


def test_rate_change_inside_an_event_interval_requires_segment_times() -> None:
    elapsed = Fraction(12_000, 48_000) + Fraction(24_000, 96_000)
    reinterpreted = Fraction(12_000 + 24_000, 96_000)
    assert elapsed == Fraction(1, 2)
    assert reinterpreted == Fraction(3, 8)
    assert 60 / elapsed == 120
    assert 60 / reinterpreted == 160


@pytest.mark.parametrize(("numerator", "expected"), [(1, 0), (3, 2), (5, 2), (7, 4)])
def test_timebase_quotient_rounds_halfway_to_even(numerator: int, expected: int) -> None:
    assert _round_positive_ratio(numerator, 2) == expected


def test_host_frame_increment_matches_exact_binary64_rate() -> None:
    rng = random.Random(0x71AE)
    rates = [
        44_100.0,
        math.nextafter(44_100.0, math.inf),
        48_000.0,
        math.nextafter(48_000.0, math.inf),
        65_536.0,
        math.nextafter(96_000.0, -math.inf),
        96_000.0,
        *(rng.uniform(44_100.0, 96_000.0) for _ in range(500)),
    ]
    for rate in rates:
        increment = _host_frame_increment(rate)
        exact = Fraction(_Q64, 1) / Fraction.from_float(rate)
        assert increment == round(exact)
        assert abs(Fraction(increment) - exact) <= Fraction(1, 2)


@pytest.mark.parametrize("rate", [0.0, 44_099.0, 96_001.0, math.nan, math.inf, -math.inf])
def test_host_frame_increment_rejects_unsupported_rate(rate: float) -> None:
    with pytest.raises(ValueError, match="unsupported host rate"):
        _host_frame_increment(rate)


def test_timebase_subframes_round_to_even_and_share_the_host_endpoint() -> None:
    for increment in range(100, 108):
        for index in range(5):
            assert _subframe_offset(increment, index) == round(Fraction(index * increment, 4))

    for rate in (44_100.0, 48_000.1, 96_000.0):
        increment = _host_frame_increment(rate)
        offsets = [_subframe_offset(increment, index) for index in range(5)]
        assert offsets[0] == 0
        assert offsets[-1] == increment
        assert all(left < right for left, right in itertools.pairwise(offsets))
        for index, offset in enumerate(offsets[:4]):
            actual = Fraction(offset, _Q64)
            exact = Fraction(index, 4) / Fraction.from_float(rate)
            assert abs(actual - exact) <= Fraction(7, 8 * _Q64)


def test_timebase_intervals_preserve_each_segment_rate() -> None:
    segments = ((48_000.0, 24_000), (96_000.0, 48_000))
    counter = 0
    exact_elapsed = Fraction()
    total_frames = 0
    for rate, frames in segments:
        counter = _checked_time_add(counter, frames * _host_frame_increment(rate))
        exact_elapsed += Fraction(frames, 1) / Fraction.from_float(rate)
        total_frames += frames
    assert exact_elapsed == 1
    assert abs(Fraction(counter, _Q64) - exact_elapsed) <= Fraction(total_frames, 2 * _Q64)


def test_timebase_matches_host_partition_and_oversampling_changes() -> None:
    increment = _host_frame_increment(48_000.0)
    clock = 0
    for factor, frames in ((1, 5), (4, 7), (1, 3)):
        for _ in range(frames):
            four_times = [clock + _subframe_offset(increment, index) for index in range(4)]
            active_times = [clock] if factor == 1 else four_times
            assert active_times[0] == four_times[0] == clock
            assert active_times[-1] < clock + increment
            assert all(left < right for left, right in itertools.pairwise(active_times))
            clock = _checked_time_add(clock, increment)
    assert clock == 15 * increment
    assert clock == _checked_time_add(0, 15 * increment)


def test_timebase_24_hour_rounding_bound_and_checked_overflow() -> None:
    frames = 96_000 * 86_400
    increment = _host_frame_increment(96_000.0)
    actual = Fraction(frames * increment, _Q64)
    exact = Fraction(86_400)
    bound = Fraction(frames, 2 * _Q64)
    assert abs(actual - exact) <= bound
    assert bound < Fraction(225, 10**12)
    assert _checked_time_add(_Q64 - 2, 3) == _Q64 + 1
    with pytest.raises(ValueError, match="overflowing"):
        _checked_time_add((1 << 128) - 1, 1)


def test_timebase_does_not_compare_stale_or_out_of_order_epochs() -> None:
    assert _elapsed_same_epoch((2, 100), (2, 150)) == 50
    with pytest.raises(ValueError, match="incomparable"):
        _elapsed_same_epoch((2, 100), (3, 150))
    with pytest.raises(ValueError, match="incomparable"):
        _elapsed_same_epoch((2, 150), (2, 100))


def test_fixed_subdivision_phase_matches_exact_integer_reference() -> None:
    rng = random.Random(0xD0A)
    fractions = (0, 1, _Q32 // 2, _Q32 - 2, _Q32 - 1)
    offsets = (0, 1, _Q32 // 2, _Q32 - 1)
    for ratio in _SUBDIVISION_RATIOS:
        numerator, denominator = ratio
        for beat in range(12):
            for within_beat in (*fractions, *(rng.getrandbits(32) for _ in range(20))):
                for offset in offsets:
                    full_product = denominator * (((beat % numerator) << 32) + within_beat)
                    assert full_product < 24 * _Q32
                    reference = (full_product // numerator + offset) & (_Q32 - 1)
                    assert _fixed_subdivision_phase(beat, within_beat, ratio, offset) == reference


def test_fixed_subdivision_phase_bounds_fraction_and_offset_quantization() -> None:
    rng = random.Random(0xB34)
    for beat in range(-12, 13):
        for ratio in _SUBDIVISION_RATIOS:
            for _ in range(8):
                within_beat = Fraction(rng.getrandbits(52), 1 << 52)
                offset = Fraction(rng.getrandbits(52), 1 << 52)
                fraction32 = (within_beat.numerator * _Q32) // within_beat.denominator
                offset32 = (offset.numerator * _Q32) // offset.denominator
                fixed = _fixed_subdivision_phase(beat % 12, fraction32, ratio, offset32)
                target = _subdivision_phase(beat, within_beat, ratio, offset)
                assert abs(_wrap_phase(target - Fraction(fixed, _Q32))) < Fraction(8, _Q32)


def test_u32_phase_to_f32_conversion_has_bounded_circular_error() -> None:
    rng = random.Random(0xF32)
    phases = (0, 1, _Q32 // 2, _Q32 - 129, _Q32 - 128, _Q32 - 1)
    for phase in (*phases, *(rng.getrandbits(32) for _ in range(5000))):
        rounded = _f32(_f32(float(phase)) * 2.0**-32)
        if rounded == 1.0:
            rounded = 0.0
        error = _wrap_phase(Fraction.from_float(rounded) - Fraction(phase, _Q32))
        assert abs(error) <= Fraction(1, 1 << 25)


def test_q32_beat_step_drift_and_q64_candidate_bound() -> None:
    ticks = 3000 * 24 * 60 * 60
    exact_increment = Fraction(1, 1500)
    q32_step = round(exact_increment * _Q32)
    q64_step = round(exact_increment * _Q64)
    q32_error = abs(Fraction(q32_step * ticks, _Q32) - exact_increment * ticks)
    q64_error = abs(Fraction(q64_step * ticks, _Q64) - exact_increment * ticks)
    assert Fraction(14, 1000) < q32_error / 2 < Fraction(15, 1000)
    assert q64_error <= Fraction(ticks, 2 * _Q64)
    assert q64_error / 2 < Fraction(4, 10**12)


def test_q64_tick_carries_and_wraps_beat_count() -> None:
    assert _q64_tick(11, _Q64 - 2, 3) == (0, 1)
    assert _q64_tick(7, 42, 0) == (7, 42)
    assert _q64_tick(2, 1, _Q64 - 1) == (3, 0)
