"""Independent preset and same-boundary examples, not production engine tests."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

_DEFAULTS: dict[str, float | str] = {
    "mode": "clean",
    "drive": 0.3,
    "tone": 0.4,
    "ultra_color": 0.5,
}


@dataclass(frozen=True)
class _Event:
    preset: bool
    assignments: dict[str, float | str]


@dataclass(frozen=True)
class _TimedEvent:
    timestamp: int
    source: str
    event: _Event
    receipt_order: int


@dataclass(frozen=True)
class _DelayState:
    mode: str
    delay: int
    tone: float


def _publish_with_live_delays(candidate: _DelayState, live_delays: tuple[int, ...]) -> _DelayState:
    bounds = {"long": (4, 12), "short": (4, 10)}[candidate.mode]
    if any(not bounds[0] <= delay <= bounds[1] for delay in (candidate.delay, *live_delays)):
        raise ValueError("unsafe live delay")
    return candidate


def _merge_daw_and_live(events: list[_TimedEvent]) -> list[_Event]:
    if any(event.source not in {"daw", "studio", "midi"} for event in events):
        raise ValueError("unattributed source")
    orders = [event.receipt_order for event in events]
    if any(order < 0 for order in orders) or len(set(orders)) != len(orders):
        raise ValueError("invalid receipt order")
    ordered = sorted(
        events, key=lambda event: (event.timestamp, event.source == "daw", event.receipt_order)
    )
    return [event.event for event in ordered]


def _candidate(
    current: dict[str, float | str],
    events: list[_Event],
    *,
    ultra: bool = False,
    drive_base: tuple[float, float] = (0.0, 1.0),
) -> tuple[dict[str, float | str], dict[str, float | str], set[str]]:
    available = set(_DEFAULTS) if ultra else set(_DEFAULTS) - {"ultra_color"}
    requested = current.copy()
    for event in events:
        for control, value in event.assignments.items():
            if control not in _DEFAULTS:
                raise ValueError("unknown control")
            if control not in available:
                raise ValueError("unavailable control")
            if control == "mode":
                if not isinstance(value, str) or value not in {"clean", "lead"}:
                    raise ValueError("invalid option")
            elif (
                not isinstance(value, (float, int))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError("invalid range value")
        if event.preset:
            requested = {control: _DEFAULTS[control] for control in available}
        requested.update(event.assignments)

    mode = requested["mode"]
    conditional = (0.0, 0.5) if mode == "clean" else (0.6, 1.0)
    lower = max(drive_base[0], conditional[0])
    upper = min(drive_base[1], conditional[1])
    if lower > upper:
        raise ValueError("empty safe intersection")
    effective = requested.copy()
    drive = requested["drive"]
    if not isinstance(drive, (int, float)):
        raise ValueError("invalid drive")
    effective["drive"] = max(lower, min(upper, drive))
    clamped = {"drive"} if effective["drive"] != drive else set()
    return requested, effective, clamped


def test_partial_preset_resets_omitted_values_then_applies_conditional_clamp() -> None:
    preset = _Event(preset=True, assignments={"mode": "lead"})
    first = {"mode": "clean", "drive": 0.1, "tone": 0.9}
    second = {"mode": "lead", "drive": 0.9, "tone": 0.2}
    requested, effective, clamped = _candidate(first, [preset])
    assert _candidate(second, [preset]) == (requested, effective, clamped)
    assert requested == {"mode": "lead", "drive": 0.3, "tone": 0.4}
    assert effective == {"mode": "lead", "drive": 0.6, "tone": 0.4}
    assert clamped == {"drive"}
    assert first == {"mode": "clean", "drive": 0.1, "tone": 0.9}


def test_preset_position_in_same_stream_controls_which_values_win() -> None:
    current = {"mode": "clean", "drive": 0.2, "tone": 0.8}
    lead = _Event(preset=True, assignments={"mode": "lead"})
    earlier = _Event(preset=False, assignments={"drive": 0.9})
    later = _Event(preset=False, assignments={"drive": 0.8})
    requested, effective, _ = _candidate(current, [earlier, lead, later])
    assert requested["drive"] == effective["drive"] == 0.8
    assert requested["tone"] == 0.4
    requested, effective, clamped = _candidate(current, [earlier, lead])
    assert requested["drive"] == 0.3
    assert effective["drive"] == 0.6
    assert clamped == {"drive"}


@pytest.mark.parametrize("reverse", [False, True])
def test_daw_wins_exact_ties_with_live_controls_and_presets(reverse: bool) -> None:
    current = {"mode": "clean", "drive": 0.2, "tone": 0.8}
    events = [
        _TimedEvent(5, "daw", _Event(preset=False, assignments={"drive": 0.8}), 0),
        _TimedEvent(5, "studio", _Event(preset=True, assignments={"mode": "lead"}), 1),
    ]
    if reverse:
        events.reverse()
    requested, effective, _ = _candidate(current, _merge_daw_and_live(events))
    assert requested == {"mode": "lead", "drive": 0.8, "tone": 0.4}
    assert effective == requested


def test_cross_source_timing_precedes_tie_priority_and_preserves_stream_order() -> None:
    current = {"mode": "clean", "drive": 0.2, "tone": 0.8}
    events = [
        _TimedEvent(4, "daw", _Event(preset=False, assignments={"tone": 0.1}), 0),
        _TimedEvent(4, "daw", _Event(preset=False, assignments={"tone": 0.2}), 1),
        _TimedEvent(5, "studio", _Event(preset=False, assignments={"tone": 0.7}), 2),
    ]
    requested, _, _ = _candidate(current, _merge_daw_and_live(events))
    assert requested["tone"] == 0.7
    requested, _, _ = _candidate(current, _merge_daw_and_live(events[:2]))
    assert requested["tone"] == 0.2
    with pytest.raises(ValueError, match="unattributed source"):
        _merge_daw_and_live([_TimedEvent(5, "unknown", events[0].event, 0)])


@pytest.mark.parametrize("reverse", [False, True])
def test_delay_target_coalescing_uses_the_same_cross_source_order(reverse: bool) -> None:
    events = [
        _TimedEvent(5, "daw", _Event(preset=False, assignments={"delay": 12.0}), 0),
        _TimedEvent(5, "midi", _Event(preset=False, assignments={"delay": 24.0}), 1),
    ]
    if reverse:
        events.reverse()
    ordered = _merge_daw_and_live(events)
    assert ordered[-1].assignments["delay"] == 12.0
    events.append(_TimedEvent(6, "studio", _Event(preset=False, assignments={"delay": 48.0}), 2))
    assert _merge_daw_and_live(events)[-1].assignments["delay"] == 48.0


@pytest.mark.parametrize("reverse", [False, True])
def test_last_received_live_event_wins_exact_ties_not_input_list_order(reverse: bool) -> None:
    current = {"mode": "clean", "drive": 0.2, "tone": 0.8}
    events = [
        _TimedEvent(5, "midi", _Event(preset=False, assignments={"tone": 0.1}), 1),
        _TimedEvent(5, "studio", _Event(preset=False, assignments={"tone": 0.7}), 2),
    ]
    if reverse:
        events.reverse()
    requested, _, _ = _candidate(current, _merge_daw_and_live(events))
    assert requested["tone"] == 0.7
    events.append(_TimedEvent(5, "daw", _Event(preset=False, assignments={"tone": 0.5}), 0))
    requested, _, _ = _candidate(current, _merge_daw_and_live(events))
    assert requested["tone"] == 0.5


def test_live_preset_and_midi_ties_respect_receipt_then_clamp_atomically() -> None:
    current = {"mode": "clean", "drive": 0.2, "tone": 0.8}
    preset = _Event(preset=True, assignments={"mode": "lead"})
    drive = _Event(preset=False, assignments={"drive": 0.9})
    later_preset = [_TimedEvent(5, "midi", drive, 0), _TimedEvent(5, "studio", preset, 1)]
    requested, effective, _ = _candidate(current, _merge_daw_and_live(later_preset))
    assert requested["drive"] == 0.3
    assert effective["drive"] == 0.6
    later_midi = [_TimedEvent(5, "midi", drive, 1), _TimedEvent(5, "studio", preset, 0)]
    requested, effective, _ = _candidate(current, _merge_daw_and_live(later_midi))
    assert requested["drive"] == effective["drive"] == 0.9


def test_timestamps_outrank_receipt_order_and_duplicate_orders_are_rejected() -> None:
    current = {"mode": "clean", "drive": 0.2, "tone": 0.8}
    early_arrival = _TimedEvent(6, "studio", _Event(False, {"tone": 0.7}), 0)
    late_arrival = _TimedEvent(5, "midi", _Event(False, {"tone": 0.1}), 1)
    requested, _, _ = _candidate(current, _merge_daw_and_live([early_arrival, late_arrival]))
    assert requested["tone"] == 0.7
    duplicate = _TimedEvent(5, "midi", _Event(False, {"tone": 0.2}), 0)
    with pytest.raises(ValueError, match="invalid receipt order"):
        _merge_daw_and_live([early_arrival, duplicate])


@pytest.mark.parametrize(
    ("assignment", "error"),
    [
        ({"missing": 0.2}, "unknown control"),
        ({"ultra_color": 0.7}, "unavailable control"),
        ({"mode": "missing"}, "invalid option"),
        ({"drive": float("nan")}, "invalid range value"),
        ({"drive": True}, "invalid range value"),
    ],
)
def test_invalid_or_unavailable_preset_does_not_change_current_state(
    assignment: dict[str, float | str], error: str
) -> None:
    current = {"mode": "clean", "drive": 0.2, "tone": 0.8}
    with pytest.raises(ValueError, match=error):
        _candidate(current, [_Event(preset=True, assignments=assignment)])
    assert current == {"mode": "clean", "drive": 0.2, "tone": 0.8}


def test_empty_safe_intersection_rejects_the_whole_candidate() -> None:
    current = {"mode": "clean", "drive": 0.2, "tone": 0.8}
    with pytest.raises(ValueError, match="empty safe intersection"):
        _candidate(
            current, [_Event(preset=True, assignments={"mode": "lead"})], drive_base=(0.0, 0.4)
        )
    assert current == {"mode": "clean", "drive": 0.2, "tone": 0.8}


def test_conditional_switch_must_not_invalidate_a_live_fade() -> None:
    current = _DelayState(mode="long", delay=12, tone=0.2)
    candidate = _DelayState(mode="short", delay=7, tone=0.8)
    assert _publish_with_live_delays(current, (8, 12)) == current
    assert _publish_with_live_delays(candidate, ()) == candidate
    with pytest.raises(ValueError, match="unsafe live delay"):
        _publish_with_live_delays(candidate, (8, 12))
    with pytest.raises(ValueError, match="unsafe live delay"):
        _publish_with_live_delays(candidate, (12,))
    assert current == _DelayState(mode="long", delay=12, tone=0.2)
    assert _publish_with_live_delays(candidate, (8,)) == candidate


def test_ultra_preset_preserves_extra_control_and_resets_other_defaults() -> None:
    current = {"mode": "lead", "drive": 0.9, "tone": 0.1, "ultra_color": 0.2}
    requested, effective, clamped = _candidate(
        current, [_Event(preset=True, assignments={"ultra_color": 0.7})], ultra=True
    )
    expected = {
        "mode": "clean",
        "drive": 0.3,
        "tone": 0.4,
        "ultra_color": 0.7,
    }
    assert requested == expected
    assert effective == expected
    assert not clamped
