"""Interleaving examples for the selected bounded ingress ordering reference."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

import pytest


@dataclass
class _Record:
    source: str
    generation: int | None = None
    sequence: int | None = None
    ready: bool = False


@dataclass(frozen=True)
class _Delivery:
    source: str
    receipt: tuple[int, int]
    scheduled: int
    consumed: int
    missed: bool


@dataclass
class _Ingress:
    lane_capacity: int
    lane_names: tuple[str, ...]
    history_capacity: int = 4
    generation: int = 0
    sequence: int = 0
    slots: dict[str, list[_Record]] = field(init=False)
    history: dict[int, int] = field(default_factory=dict)
    expired: list[_Record] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.lane_capacity <= 0 or self.history_capacity <= 0:
            raise ValueError("capacities must be positive")
        if len(set(self.lane_names)) != len(self.lane_names):
            raise ValueError("lanes must be unique")
        if self.lane_capacity * len(self.lane_names) >= 1 << 16:
            raise ValueError("total slots exceed the sequence space")
        self.slots = {source: [] for source in self.lane_names}

    def reserve(self, source: str) -> _Record | None:
        lane = self.slots[source]
        if len(lane) == self.lane_capacity or any(not record.ready for record in lane):
            return None
        record = _Record(source)
        lane.append(record)
        return record

    def accept(self, record: _Record) -> tuple[int, int]:
        if record not in self.slots[record.source] or record.generation is not None:
            raise ValueError("invalid reservation")
        receipt = self.generation, self.sequence
        self.sequence += 1
        record.generation, record.sequence = receipt
        return receipt

    def publish(self, record: _Record) -> None:
        if record.generation is None or record.ready:
            raise ValueError("record must be accepted exactly once")
        record.ready = True

    def cutoff(self, frame: int) -> list[_Delivery]:
        if self.generation == (1 << 48) - 1:
            raise OverflowError("generation exhausted")
        current = self.generation
        self.generation += 1
        self.sequence = 0
        self.history[current] = frame
        if len(self.history) > self.history_capacity:
            del self.history[min(self.history)]

        deliveries = []
        for lane in self.slots.values():
            while lane and lane[0].ready:
                record = lane[0]
                assert record.generation is not None
                assert record.sequence is not None
                if record.generation > current:
                    break
                lane.pop(0)
                if record.generation not in self.history:
                    self.expired.append(record)
                    continue
                scheduled = self.history[record.generation]
                deliveries.append(
                    _Delivery(
                        record.source,
                        (record.generation, record.sequence),
                        scheduled,
                        frame,
                        frame > scheduled,
                    )
                )
        return sorted(deliveries, key=lambda record: record.receipt)


@dataclass
class _Clock:
    previous: tuple[int, int] | None = None
    last_period: int | None = None
    acknowledged_loss: int = 0
    skipped: list[int] = field(default_factory=list)

    def consume(self, records: list[tuple[int, int]], lost: int) -> None:
        if lost > self.acknowledged_loss:
            self.previous = None
            self.acknowledged_loss = lost
        for index, timestamp in records:
            if index <= self.acknowledged_loss:
                self.skipped.append(index)
                continue
            if self.previous is not None:
                previous_index, previous_time = self.previous
                if index == previous_index + 1 and timestamp > previous_time:
                    self.last_period = timestamp - previous_time
                else:
                    self.previous = None
            self.previous = index, timestamp


def test_all_two_producer_cutoff_interleavings_preserve_receipts_and_first_boundary() -> None:
    actions = ("ar", "aa", "ap", "br", "ba", "bp", "cutoff")
    checked = 0
    for ordering in permutations(actions):
        if not (ordering.index("ar") < ordering.index("aa") < ordering.index("ap")):
            continue
        if not (ordering.index("br") < ordering.index("ba") < ordering.index("bp")):
            continue
        ingress = _Ingress(1, ("a", "b"))
        records: dict[str, _Record] = {}
        receipts: dict[str, tuple[int, int]] = {}
        delivered: list[_Delivery] = []
        for action in ordering:
            if action == "cutoff":
                delivered.extend(ingress.cutoff(0))
            elif action[1] == "r":
                record = ingress.reserve(action[0])
                assert record is not None
                records[action[0]] = record
            elif action[1] == "a":
                receipts[action[0]] = ingress.accept(records[action[0]])
            else:
                ingress.publish(records[action[0]])
        delivered.extend(ingress.cutoff(16))
        assert len(delivered) == 2
        assert len(set(receipts.values())) == 2
        for result in delivered:
            assert result.receipt == receipts[result.source]
            assert result.scheduled == 16 * result.receipt[0]
            accepted_before_cutoff = ordering.index(result.source + "a") < ordering.index("cutoff")
            published_before_cutoff = ordering.index(result.source + "p") < ordering.index("cutoff")
            assert (result.receipt[0] == 0) is accepted_before_cutoff
            assert result.missed is (accepted_before_cutoff and not published_before_cutoff)
        checked += 1
    assert checked == 140


def test_slow_publisher_does_not_block_an_independent_lane() -> None:
    ingress = _Ingress(1, ("slow", "ready"))
    slow = ingress.reserve("slow")
    ready = ingress.reserve("ready")
    assert slow is not None
    assert ready is not None
    slow_receipt = ingress.accept(slow)
    ingress.accept(ready)
    ingress.publish(ready)
    assert [result.source for result in ingress.cutoff(0)] == ["ready"]
    ingress.publish(slow)
    delayed = ingress.cutoff(16)
    assert delayed == [_Delivery("slow", slow_receipt, 0, 16, True)]


def test_old_untimed_record_is_rejected_when_cutoff_mapping_expired() -> None:
    ingress = _Ingress(1, ("slow",), history_capacity=2)
    record = ingress.reserve("slow")
    assert record is not None
    ingress.accept(record)
    for frame in (0, 16, 32):
        assert not ingress.cutoff(frame)
    ingress.publish(record)
    assert not ingress.cutoff(48)
    assert ingress.expired == [record]


def test_full_capacity_rejects_before_receipt_and_resets_sequence_at_cutoff() -> None:
    ingress = _Ingress(2, ("one", "two"))
    for source in ("one", "two", "one", "two"):
        record = ingress.reserve(source)
        assert record is not None
        ingress.accept(record)
        ingress.publish(record)
    assert ingress.reserve("one") is None
    assert ingress.sequence == 4
    assert len(ingress.cutoff(0)) == 4
    record = ingress.reserve("one")
    assert record is not None
    assert ingress.accept(record) == (1, 0)


def test_latest_loss_discards_uncertain_backlog_and_requires_fresh_events() -> None:
    clock = _Clock()
    clock.consume([(1, 0), (2, 10)], lost=0)
    assert clock.last_period == 10
    clock.consume([(3, 20), (5, 40)], lost=4)
    assert clock.skipped == [3]
    assert clock.previous == (5, 40)
    assert clock.last_period == 10
    clock.consume([(6, 53)], lost=4)
    assert clock.last_period == 13


def test_capacity_and_generation_exhaustion_are_explicit() -> None:
    with pytest.raises(ValueError, match="sequence space"):
        _Ingress(1 << 15, ("one", "two"))
    ingress = _Ingress(1, ("one",))
    ingress.generation = (1 << 48) - 1
    with pytest.raises(OverflowError, match="generation exhausted"):
        ingress.cutoff(0)
