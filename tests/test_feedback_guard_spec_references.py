"""Independent guarded-feedback candidates, not a production tank."""

from __future__ import annotations

import random
import struct
from fractions import Fraction


class _BlockMaxima:
    def __init__(self, histories: list[list[Fraction]], block_size: int) -> None:
        self.histories = [history.copy() for history in histories]
        self.block_size = block_size
        self.positions = [0] * len(histories)
        self.fresh = [Fraction()] * len(histories)
        self.maxima = [
            [
                max(abs(value) for value in history[start : start + block_size])
                for start in range(0, len(history), block_size)
            ]
            for history in histories
        ]

    def bound(self) -> Fraction:
        return max((*self.fresh, *(value for blocks in self.maxima for value in blocks)))

    def write(self, line: int, value: Fraction) -> None:
        position = self.positions[line]
        history = self.histories[line]
        history[position] = value
        self.fresh[line] = max(self.fresh[line], abs(value))
        end_of_block = (position + 1) % self.block_size == 0 or position + 1 == len(history)
        if end_of_block:
            self.maxima[line][position // self.block_size] = self.fresh[line]
            self.fresh[line] = Fraction()
        self.positions[line] = (position + 1) % len(history)


def _hadamard(values: list[Fraction]) -> list[Fraction]:
    a, b, c, d = values
    return [(a + b + c + d) / 2, (a - b + c - d) / 2, (a + b - c - d) / 2, (a - b - c + d) / 2]


def _step(
    state: _BlockMaxima,
    frame: int,
    gain: Fraction,
    inputs: list[Fraction],
    *,
    ceiling: Fraction | None = None,
) -> tuple[Fraction, list[Fraction]]:
    bound = state.bound()
    reads = []
    for history in state.histories:
        weights = (Fraction(-1, 16), Fraction(9, 16), Fraction(9, 16), Fraction(-1, 16))
        taps = [history[(frame * (index + 1) + index) % len(history)] for index in range(4)]
        reads.append(
            sum((weight * tap for weight, tap in zip(weights, taps, strict=True)), Fraction())
        )
    mixed = _hadamard(reads)
    limit = bound if ceiling is None else min(bound, ceiling)
    feedback = [gain * max(-limit, min(limit, value)) for value in mixed]
    for line, (source, value) in enumerate(zip(inputs, feedback, strict=True)):
        state.write(line, source + value)
    return bound, mixed


def test_guarded_variable_reads_and_mixing_are_bibo_bounded_in_exact_arithmetic() -> None:
    rng = random.Random(0xB1B0)
    histories = [[Fraction(rng.randrange(-10, 11), 10) for _ in range(16)] for _ in range(4)]
    state = _BlockMaxima(histories, 4)
    gain, input_bound = Fraction(9, 10), Fraction(1)
    certified_bound = max(state.bound(), input_bound / (1 - gain))
    for frame in range(128):
        source = [Fraction(rng.randrange(-10, 11), 10) for _ in range(4)]
        _step(state, frame, gain, source)
        assert state.bound() <= certified_bound
        assert all(abs(value) <= state.bound() for history in state.histories for value in history)


def test_unforced_block_upper_bound_decreases_after_each_history_replacement() -> None:
    histories = [
        [Fraction((line + position) % 7 - 3, 3) for position in range(16)] for line in range(4)
    ]
    state = _BlockMaxima(histories, 4)
    gain = Fraction(9, 10)
    previous = state.bound()
    for frame in range(64):
        _step(state, frame, gain, [Fraction()] * 4)
        if (frame + 1) % 16 == 0:
            assert state.bound() <= gain * previous
            previous = state.bound()


def test_memoryless_saturator_tightens_the_forced_state_bound() -> None:
    state = _BlockMaxima([[Fraction()] * 12 for _ in range(4)], 4)
    gain, ceiling, input_bound = Fraction(9, 10), Fraction(49, 50), Fraction(1)
    for frame in range(80):
        _step(state, frame, gain, [input_bound] * 4, ceiling=ceiling)
        assert state.bound() <= input_bound + gain * ceiling


def test_guard_changes_a_correlated_feedback_vector_before_quality_is_known() -> None:
    histories = [[Fraction(1, 10)] * 8 for _ in range(4)]
    state = _BlockMaxima(histories, 4)
    bound, mixed = _step(state, 0, Fraction(9, 10), [Fraction()] * 4)
    assert bound == Fraction(1, 10)
    assert mixed == [Fraction(1, 5), Fraction(), Fraction(), Fraction()]
    assert state.histories[0][0] == Fraction(9, 100)
    assert Fraction(9, 10) * mixed[0] == Fraction(18, 100)


def test_unforced_bound_contracts_from_a_partial_block_with_unequal_lengths() -> None:
    histories = [
        [Fraction((line + position) % 7 - 3, 3) for position in range(length)]
        for line, length in enumerate((13, 14, 15, 16))
    ]
    state = _BlockMaxima(histories, 4)
    gain = Fraction(9, 10)
    for frame in range(3):
        _step(state, frame, gain, [Fraction()] * 4)
    prior = state.bound()
    for frame in range(3, 23):
        _step(state, frame, gain, [Fraction()] * 4)
    assert state.bound() <= gain * prior
    assert all(abs(value) <= state.bound() for history in state.histories for value in history)


def test_nearest_fp16_can_prevent_strict_quiet_decay() -> None:
    smallest = 2.0**-24
    stored = struct.unpack(">e", struct.pack(">e", smallest * 0.999))[0]
    assert stored == smallest
    assert 0 < smallest * 0.999 < smallest


def test_candidate_directed_fp16_ceiling_strictly_contracts_every_positive_history_value() -> None:
    gain = struct.unpack(">f", struct.pack(">I", 0x3F7FBE76))[0]
    assert Fraction.from_float(gain) * (1 + Fraction(1, 1 << 24)) < 1
    for bits in range(1, 0x7C00):
        maximum = struct.unpack(">e", struct.pack(">H", bits))[0]
        ceiling = struct.unpack(">f", struct.pack(">f", gain * maximum))[0]
        nearest_bits = struct.unpack(">H", struct.pack(">e", ceiling))[0]
        nearest = struct.unpack(">e", struct.pack(">H", nearest_bits))[0]
        truncated_bits = nearest_bits - (nearest > ceiling)
        truncated = struct.unpack(">e", struct.pack(">H", truncated_bits))[0]
        assert 0 <= truncated < maximum


def test_grouped_maximum_tree_cost_for_a_hypothetical_four_line_tank() -> None:
    length, block_size, lines = 432_000, 64, 4
    blocks = (length + block_size - 1) // block_size
    tree_leaves = 1 << (blocks - 1).bit_length()
    assert blocks == 6_750
    assert lines * 2 * tree_leaves * 4 == 262_144
