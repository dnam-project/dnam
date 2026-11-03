"""Independent control-clock and envelope examples, not engine tests."""

from __future__ import annotations

import math
import struct
from fractions import Fraction

import pytest


def _stride(sample_rate: int) -> int:
    return max(8, (sample_rate + 1500) // 3000)


def _peak_updates(samples: list[int], stride: int) -> list[tuple[int, int]]:
    updates = []
    peak = 0
    for frame, sample in enumerate(samples):
        peak = max(peak, abs(sample))
        if (frame + 1) % stride == 0:
            updates.append((frame + 1, peak))
            peak = 0
    return updates


def _published_control_states(
    events: list[tuple[int, str, int]], stride: int, last_frame: int
) -> dict[int, dict[str, int]]:
    pending: dict[str, int] = {}
    state: dict[str, int] = {}
    published: dict[int, dict[str, int]] = {}
    next_event = 0
    for frame in range(last_frame + 1):
        while next_event < len(events) and events[next_event][0] == frame:
            _, control, value = events[next_event]
            pending[control] = value
            next_event += 1
        if frame % stride == 0 and pending:
            state.update(pending)
            published[frame] = state.copy()
            pending.clear()
    return published


def _retime_window(old_rate: int, new_rate: int, consumed: int) -> tuple[int, Fraction]:
    old_stride = _stride(old_rate)
    if not 0 <= consumed < old_stride:
        raise ValueError("invalid consumed frame count")
    remaining = (old_stride - consumed) * _stride(new_rate)
    new_frames = (2 * remaining + old_stride) // (2 * old_stride)
    elapsed = Fraction(consumed, old_rate) + Fraction(new_frames, new_rate)
    return new_frames, elapsed


def _integrated_cycles(segments: list[tuple[int, int, Fraction]]) -> Fraction:
    return sum(
        (Fraction(frames, sample_rate) * frequency for frames, sample_rate, frequency in segments),
        Fraction(),
    )


def _peak_update_with_segments(
    envelope: float, peak: float, segments: list[tuple[int, int, Fraction, Fraction]]
) -> float:
    use_attack = peak > envelope
    exposure = sum(
        (
            Fraction(frames, sample_rate) / (attack if use_attack else release)
            for frames, sample_rate, attack, release in segments
        ),
        Fraction(),
    )
    alpha = math.exp(-float(exposure))
    return alpha * envelope + (1 - alpha) * peak


@pytest.mark.parametrize(
    ("sample_rate", "stride", "control_rate"),
    [
        (44_100, 15, Fraction(2940)),
        (46_500, 16, Fraction(46_500, 16)),
        (48_000, 16, Fraction(3000)),
        (88_200, 29, Fraction(88_200, 29)),
        (96_000, 32, Fraction(3000)),
        (176_400, 59, Fraction(176_400, 59)),
        (384_000, 128, Fraction(3000)),
    ],
)
def test_actual_control_rate_uses_the_prepared_rounded_stride(
    sample_rate: int, stride: int, control_rate: Fraction
) -> None:
    assert _stride(sample_rate) == stride
    assert Fraction(sample_rate, stride) == control_rate


def test_effective_tick_ratio_preserves_the_ideal_lfo_frequency() -> None:
    sample_rate = 44_100
    control_rate = Fraction(sample_rate, _stride(sample_rate))
    lfo_frequency = Fraction(10)
    reference_cycles_per_tick = lfo_frequency / 3000
    scaled_cycles_per_tick = reference_cycles_per_tick * 3000 / control_rate
    assert control_rate * scaled_cycles_per_tick == lfo_frequency
    assert control_rate * reference_cycles_per_tick == Fraction(49, 5)


def test_effective_tick_ratio_preserves_a_held_envelope_time_constant() -> None:
    control_rate = 2940
    tau = 0.015
    reference_pole = math.exp(-1 / (3000 * tau))
    effective_pole = reference_pole ** (3000 / control_rate)
    held_decay = effective_pole**294
    assert held_decay == pytest.approx(math.exp(-0.1 / tau), rel=1e-12)
    assert reference_pole**294 != pytest.approx(held_decay, rel=0.01)


def test_legacy_control_vector_cannot_hold_all_distinct_f32_states() -> None:
    oscillator_lanes = 2 * 2
    tempo_lanes = 1
    slew_lanes = 1
    envelope_lanes = 4
    assert oscillator_lanes == 4
    assert oscillator_lanes + tempo_lanes + slew_lanes == 6 > 4
    assert oscillator_lanes + tempo_lanes + slew_lanes + envelope_lanes == 10 > 8


def test_legacy_dynamic_allowance_requires_a_complete_state_ledger() -> None:
    ramp_bytes = (16 + 16) * 4
    oscillator_and_envelope_bytes = (2 * 2 + 2 * 2 + 4) * 4
    assert ramp_bytes == 128
    assert 192 - ramp_bytes - oscillator_and_envelope_bytes == 16


def test_peak_window_is_available_only_after_its_last_sample() -> None:
    samples = [0] * 32
    samples[15] = 3
    samples[16] = -5
    assert _peak_updates(samples, 16) == [(16, 3), (32, 5)]


@pytest.mark.parametrize(
    ("old_rate", "new_rate"),
    [
        (44_100, 48_000),
        (48_000, 44_100),
        (46_500, 88_200),
        (88_200, 46_500),
        (176_400, 384_000),
        (384_000, 44_100),
        (48_000, 48_000),
    ],
)
def test_rate_transition_preserves_window_fraction_to_half_a_new_sample(
    old_rate: int, new_rate: int
) -> None:
    old_stride, new_stride = _stride(old_rate), _stride(new_rate)
    for consumed in range(old_stride):
        remaining, elapsed = _retime_window(old_rate, new_rate, consumed)
        expected = Fraction(old_stride - consumed, old_stride) * Fraction(new_stride, new_rate)
        assert 0 <= remaining <= new_stride
        assert abs(Fraction(remaining, new_rate) - expected) <= Fraction(1, 2 * new_rate)
        assert elapsed == Fraction(consumed, old_rate) + Fraction(remaining, new_rate)
    assert _retime_window(old_rate, new_rate, 0)[0] == new_stride


def test_mixed_window_uses_elapsed_time_for_its_first_control_tick() -> None:
    remaining, elapsed = _retime_window(48_000, 44_100, 8)
    assert remaining == 8
    assert elapsed == Fraction(1, 6000) + Fraction(8, 44_100)
    frequency = Fraction(10)
    assert frequency * elapsed != frequency * Fraction(_stride(44_100), 44_100)

    tau = 0.015
    reference_pole = math.exp(-1 / (3000 * tau))
    partial_pole = reference_pole ** (3000 * float(elapsed))
    assert partial_pole == pytest.approx(math.exp(-float(elapsed) / tau), rel=1e-12)


def test_effective_lfo_frequency_integrates_each_segment_without_retroactive_changes() -> None:
    segments = [(8, 48_000, Fraction(2)), (8, 44_100, Fraction(10))]
    expected = Fraction(16, 48_000) + Fraction(80, 44_100)
    assert _integrated_cycles(segments) == expected

    final_frequency_for_entire_window = Fraction(10) * (Fraction(8, 48_000) + Fraction(8, 44_100))
    assert final_frequency_for_entire_window != expected
    assert _integrated_cycles([(16, 48_000, Fraction(2))]) == Fraction(1, 1500)
    assert _integrated_cycles([(16, 48_000, Fraction(10))]) == Fraction(1, 300)


@pytest.mark.parametrize(("initial", "peak"), [(0.0, 1.0), (1.0, 0.0)])
def test_peak_envelope_uses_each_segment_time_constant(initial: float, peak: float) -> None:
    segments = [
        (8, 48_000, Fraction(1, 5000), Fraction(3, 200)),
        (8, 44_100, Fraction(1, 100), Fraction(1, 1000)),
    ]
    expected = initial
    for frames, sample_rate, attack, release in segments:
        time_constant = attack if peak > initial else release
        alpha = math.exp(-1 / (sample_rate * float(time_constant)))
        for _ in range(frames):
            expected = alpha * expected + (1 - alpha) * peak

    result = _peak_update_with_segments(initial, peak, segments)
    assert result == pytest.approx(expected, abs=1e-14)
    final_only = [
        (8, 48_000, segments[-1][2], segments[-1][3]),
        segments[-1],
    ]
    assert abs(result - _peak_update_with_segments(initial, peak, final_only)) > 0.1


def test_control_boundary_parameters_only_affect_the_next_window() -> None:
    before_boundary = [(16, 48_000, Fraction(1, 5000), Fraction(3, 200))]
    after_boundary = [(16, 48_000, Fraction(1, 100), Fraction(1, 1000))]
    old_result = _peak_update_with_segments(0.0, 1.0, before_boundary)
    new_result = _peak_update_with_segments(0.0, 1.0, after_boundary)
    assert old_result == pytest.approx(1 - math.exp(-16 / 9.6))
    assert old_result > 0.8
    assert new_result < 0.04


@pytest.mark.parametrize("rate", [44_100, 48_000, 88_200, 96_000, 192_000, 384_000])
def test_host_control_event_wait_is_bounded_by_one_fixed_rate_tick(rate: int) -> None:
    stride = _stride(rate)
    waits = []
    for frame in range(3 * stride):
        states = _published_control_states([(frame, "gain", 1)], stride, frame + stride)
        applied = next(iter(states))
        assert applied == frame + (-frame % stride)
        waits.append(applied - frame)
    assert min(waits) == 0
    assert max(waits) == stride - 1
    assert Fraction(max(waits), rate) < Fraction(1, 3000)


def test_timestamp_order_coalesces_control_values_only_at_boundaries() -> None:
    events = [
        (0, "gain", 1),
        (1, "gain", 2),
        (15, "gain", 3),
        (16, "gain", 4),
        (16, "mix", 7),
        (17, "freeze", 1),
        (31, "freeze", 0),
        (32, "gain", 5),
        (32, "gain", 6),
    ]
    assert _published_control_states(events, 16, 32) == {
        0: {"gain": 1},
        16: {"gain": 4, "mix": 7},
        32: {"gain": 6, "mix": 7, "freeze": 0},
    }


def test_fixed_rate_ramp_starts_at_boundary_and_reaches_target_on_last_sample() -> None:
    stride = 16
    ramp = [Fraction(sample + 1, stride) for sample in range(stride)]
    assert ramp[0] == Fraction(1, 16)
    assert ramp[-1] == 1


def test_fixed_rate_96khz_uses_a_32_sample_stride() -> None:
    assert _stride(96_000) == 32
    assert Fraction(31, 96_000) * 1000 == Fraction(31, 96)
    assert Fraction(31, 96) > Fraction(156, 1000)


def test_piecewise_reference_is_independent_of_host_block_partition() -> None:
    frequency = Fraction(7, 2)
    assert _integrated_cycles([(16, 48_000, frequency)]) == _integrated_cycles(
        [(1, 48_000, frequency), (5, 48_000, frequency), (10, 48_000, frequency)]
    )
    attack, release = Fraction(1, 500), Fraction(3, 200)
    whole = _peak_update_with_segments(0.2, 1.0, [(16, 48_000, attack, release)])
    split = _peak_update_with_segments(
        0.2,
        1.0,
        [(1, 48_000, attack, release), (5, 48_000, attack, release), (10, 48_000, attack, release)],
    )
    assert split == whole


def test_rounded_reference_pole_can_lose_a_finite_time_constant() -> None:
    slow_tau = 20_000.0
    fast_tau = 1 / (3000 * 110)
    slow = struct.unpack("<f", struct.pack("<f", math.exp(-1 / (3000 * slow_tau))))[0]
    fast = struct.unpack("<f", struct.pack("<f", math.exp(-1 / (3000 * fast_tau))))[0]
    assert slow == 1
    assert fast == 0
    assert 0 < math.exp(-0.5 / (3000 * fast_tau)) < 1


def test_immediate_rate_boundary_keeps_the_old_partial_peak() -> None:
    remaining, elapsed = _retime_window(384_000, 44_100, 127)
    assert remaining == 0
    assert elapsed == Fraction(127, 384_000)
    old_peak = max(abs(sample) for sample in [0, -3, 1])
    assert old_peak == 3


def test_block_peak_envelope_is_not_the_audio_rate_impulse_envelope() -> None:
    sample_rate = 48_000
    stride = 16
    attack = 0.0002
    release = 0.015
    attack_per_sample = math.exp(-1 / (sample_rate * attack))
    release_per_sample = math.exp(-1 / (sample_rate * release))
    block_peak = 1 - math.exp(-stride / (sample_rate * attack))
    impulse_first = (1 - attack_per_sample) * release_per_sample ** (stride - 1)
    impulse_last = 1 - attack_per_sample

    assert 0.81 < block_peak < 0.82
    assert 0.096 < impulse_first < impulse_last < 0.10
    assert block_peak > 8 * impulse_last


def test_one_peak_cannot_encode_both_a_pulse_and_a_sustained_block() -> None:
    stride = 16
    pulse = [1] + [0] * (stride - 1)
    held = [1] * stride
    assert max(pulse) == max(held)
    attack = math.exp(-1 / (48_000 * 0.0002))
    release = math.exp(-1 / (48_000 * 0.015))
    pulse_envelope = (1 - attack) * release ** (stride - 1)
    held_envelope = 1 - attack**stride
    assert held_envelope > 8 * pulse_envelope


def test_ideal_oversampling_changes_a_windowed_sample_peak() -> None:
    frequency = 20_000
    phase = math.pi / 12
    peak_1x = max(abs(math.cos(2 * math.pi * frequency * n / 48_000 + phase)) for n in range(16))
    peak_4x = max(abs(math.cos(2 * math.pi * frequency * n / 192_000 + phase)) for n in range(64))
    assert peak_1x == pytest.approx(math.cos(math.pi / 12))
    assert peak_4x == pytest.approx(1.0)
    assert 20 * math.log10(peak_4x / peak_1x) == pytest.approx(0.301125, abs=1e-6)
