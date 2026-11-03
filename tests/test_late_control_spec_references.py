"""Independent late-control ordering examples, not production queue tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from test_preset_spec_references import (
    _DEFAULTS,
    _candidate,
    _Event,
    _merge_daw_and_live,
    _TimedEvent,
)


@dataclass(frozen=True, order=True)
class _Stamp:
    timestamp: int
    daw_priority: bool
    receipt_order: int


def _stamp(event: _TimedEvent) -> _Stamp:
    return _Stamp(event.timestamp, event.source == "daw", event.receipt_order)


def _apply(
    current: dict[str, float | str],
    published: dict[str, _Stamp],
    events: list[_TimedEvent],
    *,
    drive_base: tuple[float, float] = (0.0, 1.0),
) -> tuple[dict[str, float | str], dict[str, float | str], dict[str, _Stamp], list[int]]:
    # Validate source and receipt uniqueness independently of input storage order.
    _merge_daw_and_live(events)
    available = set(_DEFAULTS) - {"ultra_color"}
    pending = published.copy()
    accepted: list[_Event] = []
    rejected: list[int] = []
    for event in sorted(events, key=_stamp):
        controls = available if event.event.preset else set(event.event.assignments)
        stamp = _stamp(event)
        if any(control in pending and stamp <= pending[control] for control in controls):
            rejected.append(event.receipt_order)
            continue
        accepted.append(event.event)
        pending.update(dict.fromkeys(controls, stamp))
    requested, effective, _ = _candidate(current, accepted, drive_base=drive_base)
    return requested, effective, pending, rejected


def _event(
    timestamp: int,
    source: str,
    receipt_order: int,
    assignments: dict[str, float | str],
    *,
    preset: bool = False,
) -> _TimedEvent:
    return _TimedEvent(timestamp, source, _Event(preset, assignments), receipt_order)


def _late_for_first_boundary(timestamp: int, ingress_frame: int, after_cutoff: bool) -> bool:
    first_boundary = ((timestamp + 15) // 16) * 16
    return ingress_frame > first_boundary or (ingress_frame == first_boundary and after_cutoff)


@pytest.mark.parametrize(
    ("timestamp", "ingress_frame", "after_cutoff", "late"),
    [(0, 0, False, False), (0, 0, True, True), (15, 16, False, False), (15, 16, True, True)],
)
def test_boundary_cutoff_classifies_racing_arrivals(
    timestamp: int, ingress_frame: int, after_cutoff: bool, late: bool
) -> None:
    assert _late_for_first_boundary(timestamp, ingress_frame, after_cutoff) is late


def test_late_update_applies_when_no_newer_request_was_published() -> None:
    current = {"mode": "clean", "drive": 0.3, "tone": 0.2}
    old = {"tone": _Stamp(3, False, 0)}
    requested, effective, stamps, rejected = _apply(
        current, old, [_event(5, "midi", 1, {"tone": 0.7})]
    )
    assert requested["tone"] == effective["tone"] == 0.7
    assert stamps["tone"] == _Stamp(5, False, 1)
    assert not rejected


def test_stale_update_cannot_restore_an_older_value() -> None:
    current = {"mode": "clean", "drive": 0.3, "tone": 0.8}
    newer = {"tone": _Stamp(10, False, 4)}
    requested, _, stamps, rejected = _apply(current, newer, [_event(5, "daw", 5, {"tone": 0.1})])
    assert requested == current
    assert stamps == newer
    assert rejected == [5]


def test_late_exact_ties_respect_daw_then_live_receipt_order() -> None:
    current = {"mode": "clean", "drive": 0.3, "tone": 0.2}
    midi = _Stamp(5, False, 1)
    studio = _event(5, "studio", 2, {"tone": 0.7})
    requested, _, live_stamps, rejected = _apply(current, {"tone": midi}, [studio])
    assert requested["tone"] == 0.7
    assert not rejected

    daw = _event(5, "daw", 3, {"tone": 0.5})
    requested, _, daw_stamps, rejected = _apply(requested, live_stamps, [daw])
    assert requested["tone"] == 0.5
    assert not rejected
    requested, _, _, rejected = _apply(requested, daw_stamps, [_event(5, "midi", 4, {"tone": 0.9})])
    assert requested["tone"] == 0.5
    assert rejected == [4]


def test_a_stale_preset_is_rejected_whole_while_an_independent_update_proceeds() -> None:
    current = {"mode": "clean", "drive": 0.4, "tone": 0.2}
    published = {"drive": _Stamp(10, True, 0)}
    stale_preset = _event(9, "studio", 1, {"mode": "lead"}, preset=True)
    later_tone = _event(11, "midi", 2, {"tone": 0.8})
    requested, _, stamps, rejected = _apply(current, published, [later_tone, stale_preset])
    assert requested == {"mode": "clean", "drive": 0.4, "tone": 0.8}
    assert stamps["drive"] == published["drive"]
    assert "mode" not in stamps
    assert rejected == [1]


def test_stale_indivisible_multi_control_update_is_not_partially_applied() -> None:
    current = {"mode": "clean", "drive": 0.4, "tone": 0.2}
    published = {"drive": _Stamp(10, True, 0)}
    request = _event(9, "studio", 1, {"drive": 0.2, "tone": 0.8})
    requested, _, stamps, rejected = _apply(current, published, [request])
    assert requested == current
    assert stamps == published
    assert rejected == [1]


def test_older_late_request_does_not_replace_a_newer_pending_target() -> None:
    current = {"mode": "clean", "drive": 0.3, "tone": 0.2}
    newer_pending = _event(10, "midi", 0, {"tone": 0.8})
    older_late = _event(9, "studio", 1, {"tone": 0.4})
    requested, _, stamps, rejected = _apply(current, {}, [newer_pending, older_late])
    assert requested["tone"] == 0.8
    assert stamps["tone"] == _Stamp(10, False, 0)
    assert not rejected


def test_clamping_does_not_replace_the_requested_value_stamp() -> None:
    current = {"mode": "clean", "drive": 0.3, "tone": 0.4}
    requested, effective, stamps, _ = _apply(current, {}, [_event(5, "midi", 0, {"drive": 0.9})])
    assert requested["drive"] == 0.9
    assert effective["drive"] == 0.5
    requested, effective, stamps, _ = _apply(
        requested, stamps, [_event(6, "studio", 1, {"mode": "lead"})]
    )
    assert effective["drive"] == 0.9
    assert stamps["drive"] == _Stamp(5, False, 0)
    requested, _, _, rejected = _apply(requested, stamps, [_event(4, "daw", 2, {"drive": 0.2})])
    assert requested["drive"] == 0.9
    assert rejected == [2]


def test_duplicate_identity_is_not_reapplied() -> None:
    current = {"mode": "clean", "drive": 0.3, "tone": 0.2}
    stamp = _Stamp(5, False, 7)
    requested, _, stamps, rejected = _apply(
        current, {"tone": stamp}, [_event(5, "midi", 7, {"tone": 0.8})]
    )
    assert requested == current
    assert stamps["tone"] == stamp
    assert rejected == [7]


def test_invalid_candidate_does_not_publish_values_or_request_stamps() -> None:
    current = {"mode": "clean", "drive": 0.3, "tone": 0.2}
    published = {"tone": _Stamp(1, False, 0)}
    with pytest.raises(ValueError, match="empty safe intersection"):
        _apply(current, published, [_event(2, "midi", 1, {"tone": 0.7})], drive_base=(0.9, 1.0))
    assert current == {"mode": "clean", "drive": 0.3, "tone": 0.2}
    assert published == {"tone": _Stamp(1, False, 0)}
