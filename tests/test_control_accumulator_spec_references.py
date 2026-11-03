"""Independent finite-precision examples for control accumulation."""

from __future__ import annotations

import math
import struct
from fractions import Fraction
from random import Random

import pytest


def _f32(value: float) -> float:
    return struct.unpack(">f", struct.pack(">f", value))[0]


def _f32_from_bits(bits: int) -> float:
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def _f32_integer_units(value: float) -> int:
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    sign = -1 if bits >> 31 else 1
    exponent = (bits >> 23) & 0xFF
    fraction = bits & 0x7F_FFFF
    if exponent == 0xFF:
        raise ValueError("non-finite contributor")
    if exponent == 0:
        return sign * fraction
    return sign * ((1 << 23) | fraction) * (1 << (exponent - 1))


def _exact_fm(values: list[float]) -> Fraction:
    operands = [Fraction.from_float(value) for value in values]
    return sum(
        (left * right for index, left in enumerate(operands) for right in operands[index + 1 :]),
        Fraction(0),
    )


def _two_stat_fm(values: list[float]) -> float:
    linear = 0.0
    squared = 0.0
    for value in values:
        linear = _f32(linear + value)
        squared = _f32(squared + _f32(value * value))
    return _f32(0.5 * _f32(_f32(linear * linear) - squared))


class _PairTree:
    def __init__(self, count: int) -> None:
        if count < 0:
            raise ValueError("negative contributor count")
        self.count = count
        self.leaves = 1 << (max(count, 1) - 1).bit_length()
        self.linear = [0.0] * (2 * self.leaves)
        self.interaction = [0.0] * (2 * self.leaves)

    def update(self, index: int, value: float) -> None:
        if not 0 <= index < self.count:
            raise IndexError(index)
        node = self.leaves + index
        self.linear[node] = _f32(value)
        while node > 1:
            node //= 2
            left = node * 2
            right = left + 1
            self.linear[node] = _f32(self.linear[left] + self.linear[right])
            existing = _f32(self.interaction[left] + self.interaction[right])
            cross = _f32(self.linear[left] * self.linear[right])
            self.interaction[node] = _f32(existing + cross)

    def result(self) -> tuple[float, float]:
        return self.linear[1], self.interaction[1]


class _IntegerPairStats:
    def __init__(self, count: int) -> None:
        if count < 0:
            raise ValueError("negative contributor count")
        self.values = [0] * count
        self.linear = 0
        self.squared = 0

    def update(self, index: int, value: float) -> None:
        if not 0 <= index < len(self.values):
            raise IndexError(index)
        old = self.values[index]
        new = _f32_integer_units(value)
        self.linear += new - old
        self.squared += new * new - old * old
        self.values[index] = new

    def interaction(self) -> Fraction:
        return Fraction(self.linear * self.linear - self.squared, 2**299)


def _mixed_integer(main: list[float], factors: list[list[float]], mixer: list[float]) -> int:
    if len(factors) != len(mixer) or any(len(factor) != len(main) for factor in factors):
        raise ValueError("inconsistent contributor dimensions")
    mixed = sum(map(_f32_integer_units, main)) << 299
    for factor, coefficient in zip(factors, mixer, strict=True):
        units = list(map(_f32_integer_units, factor))
        difference = sum(units) ** 2 - sum(value * value for value in units)
        mixed += _f32_integer_units(coefficient) * difference
    return mixed


def test_two_stat_subtraction_loses_a_small_pair_interaction() -> None:
    values = [1.0, 2.0**-25]
    tree = _PairTree(len(values))
    exact_stats = _IntegerPairStats(len(values))
    for index, value in enumerate(values):
        tree.update(index, value)
        exact_stats.update(index, value)

    assert _exact_fm(values) == Fraction(1, 2**25)
    assert _two_stat_fm(values) == 0.0
    assert Fraction.from_float(tree.result()[1]) == _exact_fm(values)
    assert exact_stats.interaction() == _exact_fm(values)
    assert _f32(_two_stat_fm(values) * 2.0**25) == 0.0
    assert _f32(tree.result()[1] * 2.0**25) == 1.0


def test_repeated_f32_deltas_accumulate_history_dependent_phantom_interaction() -> None:
    delta = 5.0 * 2.0**-26
    linear = 1.0
    squared = 1.0
    previous = 0.0
    tree = _PairTree(2)
    exact_stats = _IntegerPairStats(2)
    tree.update(0, 1.0)
    exact_stats.update(0, 1.0)

    for _ in range(10_000):
        for current in (delta, 2.0 * delta, 0.0):
            linear = _f32(linear + _f32(current - previous))
            change_squared = _f32(_f32(current * current) - _f32(previous * previous))
            squared = _f32(squared + change_squared)
            tree.update(1, current)
            exact_stats.update(1, current)
            previous = current

    interaction = _f32(0.5 * _f32(_f32(linear * linear) - squared))
    assert linear == 1.0 + 10_000 * 2.0**-23
    assert squared == 1.0
    assert interaction > 0.001
    assert _exact_fm([1.0, previous]) == 0
    assert tree.result() == (1.0, 0.0)
    assert exact_stats.interaction() == 0


def test_f32_integer_units_cover_finite_extremes_and_subnormals() -> None:
    encodings = (
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x007F_FFFF,
        0x0080_0000,
        0x3F80_0000,
        0xBF80_0000,
        0x7F7F_FFFF,
        0xFF7F_FFFF,
    )
    for bits in encodings:
        value = _f32_from_bits(bits)
        assert Fraction(_f32_integer_units(value), 2**149) == Fraction.from_float(value)


@pytest.mark.parametrize("bits", [0x7F80_0000, 0xFF80_0000, 0x7FC0_0000])
def test_f32_integer_units_reject_nonfinite_inputs(bits: int) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        _f32_integer_units(_f32_from_bits(bits))


def test_integer_stats_fit_conditional_255_contributor_widths() -> None:
    count = 255
    maximum = _f32_from_bits(0x7F7F_FFFF)
    values = [maximum] * count
    stats = _IntegerPairStats(count)
    for index, value in enumerate(values):
        stats.update(index, value)
    assert abs(stats.linear) < 2**285
    assert stats.squared < 2**562
    assert stats.linear * stats.linear < 2**570
    assert abs(stats.linear * stats.linear - stats.squared) < 2**570
    assert stats.interaction() == _exact_fm(values)

    for index in range(1, count, 2):
        values[index] = -maximum
        stats.update(index, values[index])
    assert stats.linear * stats.linear - stats.squared < 0
    assert abs(stats.linear * stats.linear - stats.squared) < 2**570
    assert stats.interaction() == _exact_fm(values)


def test_integer_stats_match_exact_pairs_after_random_updates() -> None:
    rng = Random(873_052)
    values = [0.0] * 16
    stats = _IntegerPairStats(len(values))
    for _ in range(500):
        index = rng.randrange(len(values))
        value = _f32(rng.randint(-512, 512) * 2.0 ** rng.randint(-40, 30))
        values[index] = value
        stats.update(index, value)
        assert stats.squared >= 0
        assert stats.interaction() == _exact_fm(values)


def test_accumulator_word_bounds_hold_beyond_one_illustrative_control_limit() -> None:
    maximum = _f32_from_bits(0x7F7F_FFFF)
    for count in (0, 1, 2, 255, 256, 511, 512, 1024):
        stats = _IntegerPairStats(count)
        for index in range(count):
            stats.update(index, maximum if index % 3 else -maximum)
        width = (max(count, 1) - 1).bit_length()
        difference = stats.linear * stats.linear - stats.squared
        assert abs(stats.linear) < 2 ** (277 + width)
        assert abs(difference) < 2 ** (554 + 2 * width)


def test_exact_mixed_raw_reference_allows_large_intermediate_cancellation() -> None:
    maximum = _f32_from_bits(0x7F7F_FFFF)
    factor_a = _IntegerPairStats(2)
    factor_b = _IntegerPairStats(2)
    for index in (0, 1):
        factor_a.update(index, maximum)
        factor_b.update(index, maximum)
    interaction = factor_a.interaction()
    assert interaction > Fraction.from_float(maximum)
    assert interaction == factor_b.interaction()
    exact_raw = Fraction(1) + interaction - factor_b.interaction()
    assert exact_raw == 1
    with pytest.raises(OverflowError):
        _f32(float(interaction))


def test_integer_statistics_and_direct_pairs_give_identical_exact_raw_groups() -> None:
    rng = Random(305_727)
    for _ in range(30):
        factors = [
            [_f32_from_bits(rng.randrange(0x7F80_0000)) * rng.choice((-1, 1)) for _ in range(7)]
            for _ in range(4)
        ]
        main = [
            [_f32_from_bits(rng.randrange(0x7F80_0000)) * rng.choice((-1, 1)) for _ in range(7)]
            for _ in range(8)
        ]
        mixer = [[_f32(rng.randint(-8, 8) / 8) for _ in range(4)] for _ in range(8)]
        integer_interactions = []
        for factor in factors:
            stats = _IntegerPairStats(len(factor))
            for index, value in enumerate(factor):
                stats.update(index, value)
            integer_interactions.append(stats.interaction())
        direct_interactions = [_exact_fm(factor) for factor in factors]
        assert integer_interactions == direct_interactions
        for main_row, mixer_row in zip(main, mixer, strict=True):
            exact_main = sum(map(Fraction.from_float, main_row), Fraction())
            direct = exact_main + sum(
                (
                    Fraction.from_float(u) * pair
                    for u, pair in zip(mixer_row, direct_interactions, strict=True)
                ),
                Fraction(),
            )
            integer = exact_main + sum(
                (
                    Fraction.from_float(u) * pair
                    for u, pair in zip(mixer_row, integer_interactions, strict=True)
                ),
                Fraction(),
            )
            assert direct == integer
            mixed = _mixed_integer(main_row, factors, mixer_row)
            width = (max(len(main_row), 1) - 1).bit_length()
            assert abs(mixed) < 2 ** (835 + 2 * width)
            assert Fraction(mixed, 2**448) == direct


@pytest.mark.parametrize("count", [0, 1, 2, 3, 7, 8, 9, 255, 256, 257])
def test_mixed_integer_bound_handles_worst_sign_coherence(count: int) -> None:
    maximum = _f32_from_bits(0x7F7F_FFFF)
    coefficient = Fraction.from_float(maximum)
    main = [maximum] * count
    factors = [[maximum] * count for _ in range(8)]
    mixed = _mixed_integer(main, factors, [maximum] * 8)
    width = (max(count, 1) - 1).bit_length()
    expected = count * coefficient + 8 * (count * (count - 1) // 2) * coefficient**3
    assert abs(mixed) < 2 ** (835 + 2 * width)
    assert Fraction(mixed, 2**448) == expected


def test_clip_reference_can_receive_raw_value_outside_f32_range() -> None:
    maximum = _f32_from_bits(0x7F7F_FFFF)
    mixed = _mixed_integer([0.0, 0.0], [[maximum, maximum]], [maximum])
    raw = Fraction(mixed, 2**448)
    widened = float(raw)
    assert math.isfinite(widened)
    assert raw > Fraction.from_float(maximum)
    with pytest.raises(OverflowError):
        _f32(widened)
    clipped = widened / math.sqrt(1.0 + widened * widened)
    assert _f32(clipped) == 1.0


def test_pair_tree_returns_to_the_same_state_after_different_update_orders() -> None:
    initial = [1.0, 2.0**-25, -0.25, 0.5, -0.75]
    tree = _PairTree(len(initial))
    for index, value in enumerate(initial):
        tree.update(index, value)
    original = tree.result()

    for index, value in enumerate(reversed(initial)):
        tree.update(index, value)
    for index in (3, 0, 4, 2, 1):
        tree.update(index, initial[index])

    assert tree.result() == original


def test_pair_tree_meets_conditional_absolute_error_bound_on_sampled_inputs() -> None:
    rng = Random(4_080_215)
    unit_roundoff = Fraction(1, 2**24)
    observed_rounding = False
    for count in (0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 255):
        height = (max(count, 1) - 1).bit_length()
        gamma = 2 * height * unit_roundoff / (1 - 2 * height * unit_roundoff)
        for _ in range(8):
            values = [
                _f32(rng.randint(-512, 512) * 2.0 ** rng.randint(-20, 10)) for _ in range(count)
            ]
            tree = _PairTree(count)
            for index, value in enumerate(values):
                tree.update(index, value)
            exact = _exact_fm(values)
            operands = [abs(Fraction.from_float(value)) for value in values]
            absolute_pairs = sum(
                (
                    left * right
                    for index, left in enumerate(operands)
                    for right in operands[index + 1 :]
                ),
                Fraction(0),
            )
            error = abs(Fraction.from_float(tree.result()[1]) - exact)
            observed_rounding |= error > 0
            assert error <= gamma * absolute_pairs
    assert observed_rounding


def test_conditional_raw_latent_error_budget_covers_sampled_mixing() -> None:
    rng = Random(21_540)
    unit_roundoff = Fraction(1, 2**24)
    gamma4 = 4 * unit_roundoff / (1 - 4 * unit_roundoff)
    gamma5 = 5 * unit_roundoff / (1 - 5 * unit_roundoff)
    observed_materialization_error = False

    def sample() -> float:
        return _f32(rng.randint(-(2**20), 2**20) / 2**20)

    def pairs(values: list[Fraction]) -> Fraction:
        return sum(
            (left * right for index, left in enumerate(values) for right in values[index + 1 :]),
            Fraction(),
        )

    for _ in range(100):
        bases = [[sample() for _ in range(4)] for _ in range(8)]
        weights = [[sample() for _ in range(4)] for _ in range(8)]
        embeddings = [sample() for _ in range(8)]
        factors = [[sample() for _ in range(8)] for _ in range(4)]
        mixer = [sample() for _ in range(4)]

        exact_main = Fraction()
        materialized_main = []
        main_error = Fraction()
        main_absolute_terms = Fraction()
        for basis, weight in zip(bases, weights, strict=True):
            exact_terms = [
                Fraction.from_float(x) * Fraction.from_float(w)
                for x, w in zip(basis, weight, strict=True)
            ]
            exact_contribution = sum(exact_terms, Fraction())
            rounded_terms = [_f32(x * w) for x, w in zip(basis, weight, strict=True)]
            contribution = rounded_terms[0]
            for term in rounded_terms[1:]:
                contribution = _f32(contribution + term)
            materialized_main.append(Fraction.from_float(contribution))
            exact_main += exact_contribution
            main_error += abs(Fraction.from_float(contribution) - exact_contribution)
            main_absolute_terms += sum(map(abs, exact_terms), Fraction())
        assert main_error <= gamma4 * main_absolute_terms
        observed_materialization_error |= main_error > 0

        accumulated_main = sum(materialized_main, Fraction())
        main = _f32(float(accumulated_main))
        main_round_error = abs(Fraction.from_float(main) - accumulated_main)
        ideal_raw = exact_main
        rounded_raw = main
        interaction_error_budget = Fraction()
        mixing_magnitude = abs(Fraction.from_float(main))
        for factor, mix in zip(factors, mixer, strict=True):
            exact_leaves = [
                Fraction.from_float(v) * Fraction.from_float(z)
                for v, z in zip(factor, embeddings, strict=True)
            ]
            rounded_leaves = [
                Fraction.from_float(_f32(v * z)) for v, z in zip(factor, embeddings, strict=True)
            ]
            ideal_interaction = pairs(exact_leaves)
            materialized_interaction = pairs(rounded_leaves)
            absolute_pairs = pairs([abs(value) for value in exact_leaves])
            assert (
                abs(materialized_interaction - ideal_interaction)
                <= (2 * unit_roundoff + unit_roundoff**2) * absolute_pairs
            )
            interaction = _f32(float(materialized_interaction))
            interaction_error = abs(Fraction.from_float(interaction) - ideal_interaction)
            exact_mix = Fraction.from_float(mix)
            ideal_raw += exact_mix * ideal_interaction
            interaction_error_budget += abs(exact_mix) * interaction_error
            mixing_magnitude += abs(exact_mix * Fraction.from_float(interaction))
            rounded_raw = _f32(rounded_raw + _f32(mix * interaction))

        bound = main_error + main_round_error + interaction_error_budget + gamma5 * mixing_magnitude
        assert abs(Fraction.from_float(rounded_raw) - ideal_raw) <= bound
    assert observed_materialization_error
