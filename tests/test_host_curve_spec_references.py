"""Independent host-curve and control-boundary examples, not plug-in tests."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import pytest

from test_preset_spec_references import _candidate, _Event, _merge_daw_and_live, _TimedEvent


@dataclass(frozen=True)
class _CurveUpdate:
    frame: int
    value: Fraction


def _vst3_block(
    previous: Fraction, points: tuple[tuple[int, Fraction], ...], size: int, start: int
) -> tuple[list[Fraction], list[_CurveUpdate]]:
    if size <= 0 or not 0 <= previous <= 1:
        raise ValueError("invalid block")
    offsets = [offset for offset, _ in points]
    if (
        offsets != sorted(set(offsets))
        or any(offset < 0 or offset >= size for offset in offsets)
        or any(not 0 <= value <= 1 for _, value in points)
    ):
        raise ValueError("invalid curve points")
    samples: list[Fraction] = []
    updates: list[_CurveUpdate] = []
    left_offset, left_value = -1, previous
    point_index = 0
    prior_sample = previous
    for frame in range(size):
        while point_index < len(points) and points[point_index][0] < frame:
            left_offset, left_value = points[point_index]
            point_index += 1
        if point_index < len(points):
            right_offset, right_value = points[point_index]
            value = left_value + (right_value - left_value) * Fraction(
                frame - left_offset, right_offset - left_offset
            )
        else:
            value = left_value
        samples.append(value)
        if value != prior_sample or frame in offsets:
            updates.append(_CurveUpdate(start + frame, value))
        prior_sample = value
    return samples, updates


def _latest_at_tick(
    updates: list[_CurveUpdate], effective_frame: int, oversampling: int, initial: Fraction
) -> Fraction:
    selected = initial
    for update in updates:
        if update.frame * oversampling <= effective_frame:
            selected = update.value
    return selected


def test_vst3_single_endpoint_is_a_curve_not_a_step() -> None:
    samples, updates = _vst3_block(Fraction(0), ((63, Fraction(1)),), 64, 0)
    assert [samples[frame] for frame in (0, 15, 16, 32, 63)] == [
        Fraction(1, 64),
        Fraction(1, 4),
        Fraction(17, 64),
        Fraction(33, 64),
        Fraction(1),
    ]
    assert _latest_at_tick(updates, 16, 1, Fraction(0)) == Fraction(17, 64)
    assert samples[16] != 0  # A point-only interpretation would still hold zero.


def test_equivalent_vst3_ramp_has_the_same_samples_across_block_partitions() -> None:
    whole, whole_updates = _vst3_block(Fraction(0), ((63, Fraction(1)),), 64, 0)
    partitioned: list[Fraction] = []
    partitioned_updates: list[_CurveUpdate] = []
    prior = Fraction(0)
    for start in range(0, 64, 16):
        segment, updates = _vst3_block(prior, ((15, Fraction(start + 16, 64)),), 16, start)
        partitioned.extend(segment)
        partitioned_updates.extend(updates)
        prior = segment[-1]
    assert partitioned == whole
    assert partitioned_updates == whole_updates


def test_jump_uses_adjacent_points_and_empty_block_holds_last_value() -> None:
    samples, _ = _vst3_block(Fraction(0), ((10, Fraction(0)), (11, Fraction(1))), 16, 0)
    assert samples[:11] == [Fraction(0)] * 11
    assert samples[11:] == [Fraction(1)] * 5
    held, updates = _vst3_block(samples[-1], (), 16, 16)
    assert held == [Fraction(1)] * 16
    assert updates == []


def test_repeated_explicit_point_is_still_a_request() -> None:
    samples, updates = _vst3_block(Fraction(1, 2), ((3, Fraction(1, 2)),), 8, 0)
    assert samples == [Fraction(1, 2)] * 8
    assert updates == [_CurveUpdate(3, Fraction(1, 2))]


def test_interpolate_in_normalized_domain_before_nonlinear_control_mapping() -> None:
    samples, _ = _vst3_block(Fraction(0), ((3, Fraction(1)),), 4, 0)
    normalized_midpoint = samples[1]
    assert normalized_midpoint == Fraction(1, 2)
    assert normalized_midpoint**2 == Fraction(1, 4)
    assert normalized_midpoint**2 != Fraction(1, 2)


def test_host_samples_map_to_first_subframe_and_are_held_between_them() -> None:
    _, updates = _vst3_block(Fraction(0), ((63, Fraction(1)),), 64, 0)
    assert _latest_at_tick(updates, 59, 4, Fraction(0)) == Fraction(15, 64)
    assert _latest_at_tick(updates, 60, 4, Fraction(0)) == Fraction(16, 64)
    assert _latest_at_tick(updates, 64, 4, Fraction(0)) == Fraction(17, 64)


def test_implicit_daw_samples_participate_in_timestamp_and_tie_order() -> None:
    _, updates = _vst3_block(Fraction(0), ((31, Fraction(1)),), 32, 0)
    current: dict[str, float | str] = {"mode": "lead", "drive": 0.7, "tone": 0.4}
    daw = [
        _TimedEvent(update.frame, "daw", _Event(False, {"drive": float(update.value)}), index)
        for index, update in enumerate(updates)
        if update.frame <= 16
    ]
    live = _TimedEvent(16, "studio", _Event(False, {"drive": 0.9}), len(daw))
    requested, _, _ = _candidate(current, _merge_daw_and_live([*daw, live]))
    assert requested["drive"] == float(Fraction(17, 32))
    live_after = _TimedEvent(17, "midi", _Event(False, {"drive": 0.8}), len(daw) + 1)
    requested, _, _ = _candidate(current, _merge_daw_and_live([*daw, live_after]))
    assert requested["drive"] == 0.8


@pytest.mark.parametrize(
    "points",
    [
        ((-1, Fraction(0)),),
        ((8, Fraction(0)),),
        ((1, Fraction(0)), (1, Fraction(1))),
        ((2, Fraction(0)), (1, Fraction(1))),
        ((1, Fraction(3, 2)),),
    ],
)
def test_invalid_curve_points_are_rejected(points: tuple[tuple[int, Fraction], ...]) -> None:
    with pytest.raises(ValueError, match="invalid curve points"):
        _vst3_block(Fraction(0), points, 8, 0)
