"""Independent ordered clock-event examples, not a phase-lock implementation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _Clock:
    watermark: int | None = None
    previous: int | None = None
    last_period: int | None = None
    state: str = "unavailable"
    received: int = 0
    unmeasurable: int = 0
    intervals: list[int] = field(default_factory=list)

    def lose_timing(self) -> None:
        self.previous = None
        self.state = "holdover" if self.last_period is not None else "unavailable"

    def pulse(self, timestamp: int) -> bool:
        self.received += 1
        if self.watermark is not None and timestamp <= self.watermark:
            self.unmeasurable += 1
            self.lose_timing()
            return False
        self.watermark = timestamp
        if self.previous is None:
            self.previous = timestamp
            return True
        self.last_period = timestamp - self.previous
        self.previous = timestamp
        self.intervals.append(self.last_period)
        self.state = "candidate"
        return True


def test_backlogged_pulses_use_original_time_not_drain_time() -> None:
    clock = _Clock()
    drain_time = 40
    for timestamp in (0, 10, 20):
        assert timestamp < drain_time
        assert clock.pulse(timestamp)
    assert clock.intervals == [10, 10]
    assert clock.last_period == 10
    assert clock.state == "candidate"


def test_out_of_order_pulse_cannot_rewrite_an_incorporated_interval() -> None:
    clock = _Clock()
    for timestamp in (10, 20, 30):
        assert clock.pulse(timestamp)
    assert not clock.pulse(25)
    assert clock.intervals == [10, 10]
    assert clock.watermark == 30
    assert clock.previous is None
    assert clock.last_period == 10
    assert clock.state == "holdover"


def test_equal_timestamps_are_unmeasurable_not_a_valid_zero_period() -> None:
    clock = _Clock()
    assert clock.pulse(10)
    assert not clock.pulse(10)
    assert clock.intervals == []
    assert clock.received == 2
    assert clock.unmeasurable == 1
    assert clock.last_period is None
    assert clock.state == "unavailable"


def test_loss_requires_two_fresh_events_before_a_new_interval() -> None:
    clock = _Clock()
    assert clock.pulse(0)
    assert clock.pulse(10)
    clock.lose_timing()
    assert clock.state == "holdover"
    assert clock.pulse(30)
    assert clock.intervals == [10]
    assert clock.state == "holdover"
    assert clock.pulse(42)
    assert clock.intervals == [10, 12]
    assert clock.state == "candidate"


def test_two_fresh_pulses_do_not_by_themselves_prove_phase_lock() -> None:
    clock = _Clock()
    assert clock.pulse(100)
    assert clock.pulse(110)
    assert clock.state == "candidate"
