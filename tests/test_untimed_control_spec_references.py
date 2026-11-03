"""Independent untimestamped-live scheduling examples, not concurrent queue tests."""

from __future__ import annotations

from dataclasses import dataclass

from test_late_control_spec_references import _apply, _event, _Stamp
from test_preset_spec_references import _TimedEvent


@dataclass(frozen=True)
class _Untimed:
    source: str
    order: int
    assignments: dict[str, float | str]


def _schedule(
    records: list[_Untimed], boundary: int, cutoff: int
) -> tuple[list[_TimedEvent], list[_Untimed]]:
    ready = [
        _event(boundary, record.source, record.order, record.assignments)
        for record in records
        if record.order <= cutoff
    ]
    return ready, [record for record in records if record.order > cutoff]


def test_untimed_live_updates_use_cutoff_and_receipt_not_storage_order() -> None:
    records = [_Untimed("midi", 2, {"tone": 0.9}), _Untimed("studio", 1, {"tone": 0.3})]
    ready, pending = _schedule(records, boundary=16, cutoff=1)
    current = {"mode": "clean", "drive": 0.3, "tone": 0.1}
    requested, _, stamps, rejected = _apply(current, {}, ready)
    assert requested["tone"] == 0.3
    assert stamps["tone"] == _Stamp(16, False, 1)
    assert not rejected
    ready, pending = _schedule(pending, boundary=32, cutoff=2)
    requested, _, stamps, rejected = _apply(requested, stamps, ready)
    assert requested["tone"] == 0.9
    assert stamps["tone"] == _Stamp(32, False, 2)
    assert not pending
    assert not rejected


def test_untimed_studio_midi_tie_uses_last_received_at_same_boundary() -> None:
    records = [_Untimed("midi", 2, {"tone": 0.9}), _Untimed("studio", 1, {"tone": 0.3})]
    ready, pending = _schedule(records, boundary=16, cutoff=2)
    current = {"mode": "clean", "drive": 0.3, "tone": 0.1}
    requested, _, _, rejected = _apply(current, {}, ready)
    assert requested["tone"] == 0.9
    assert not pending
    assert not rejected


def test_daw_wins_a_logical_boundary_tie_but_not_an_earlier_timestamp() -> None:
    ready, _ = _schedule([_Untimed("studio", 1, {"tone": 0.8})], boundary=16, cutoff=1)
    current = {"mode": "clean", "drive": 0.3, "tone": 0.1}
    same_boundary = _event(16, "daw", 0, {"tone": 0.4})
    requested, _, _, _ = _apply(current, {}, [*ready, same_boundary])
    assert requested["tone"] == 0.4
    earlier_daw = _event(15, "daw", 0, {"tone": 0.4})
    requested, _, _, _ = _apply(current, {}, [*ready, earlier_daw])
    assert requested["tone"] == 0.8


def test_deferred_untimed_request_keeps_its_first_boundary_and_can_become_stale() -> None:
    ready, _ = _schedule([_Untimed("midi", 1, {"tone": 0.8})], boundary=16, cutoff=1)
    current = {"mode": "clean", "drive": 0.3, "tone": 0.4}
    published = {"tone": _Stamp(20, True, 2)}
    requested, _, stamps, rejected = _apply(current, published, ready)
    assert requested == current
    assert stamps == published
    assert rejected == [1]
    assert ready[0].timestamp == 16
