"""Independent Stage II mathematical examples, not production engine tests."""

from __future__ import annotations

import math
import random
import struct
from fractions import Fraction

import pytest

_ONE_THIRD_F64 = 0.3333333333333333
_ONE_FIFTH_F64 = 0.2


def _derivative(coefficients: list[int]) -> list[int]:
    return [degree * value for degree, value in enumerate(coefficients)][1:]


def _evaluate(coefficients: list[int], value: Fraction) -> Fraction:
    result = Fraction(0)
    for coefficient in reversed(coefficients):
        result = result * value + coefficient
    return result


def _rank(matrix: list[list[Fraction]]) -> int:
    rows = [row.copy() for row in matrix]
    pivot_row = 0
    for column in range(len(rows[0])):
        pivot = next((i for i in range(pivot_row, len(rows)) if rows[i][column]), None)
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        scale = rows[pivot_row][column]
        rows[pivot_row] = [value / scale for value in rows[pivot_row]]
        for i in range(pivot_row + 1, len(rows)):
            scale = rows[i][column]
            if scale:
                rows[i] = [
                    value - scale * basis
                    for value, basis in zip(rows[i], rows[pivot_row], strict=True)
                ]
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return pivot_row


def _clamp_unit(value: Fraction) -> Fraction:
    return min(max(value, Fraction(-1)), Fraction(1))


def _drive_features(signal: Fraction) -> list[Fraction]:
    features = []
    for drive in (Fraction(1, 4), Fraction(1), Fraction(4), Fraction(16)):
        u = _clamp_unit(Fraction(2, 3) * drive * signal)
        features.extend((u * (Fraction(3, 2) - u * u / 2), u * u - u**4))
    return features


def _drive_adaa(previous: Fraction, current: Fraction) -> tuple[Fraction, Fraction]:
    midpoint, difference = (current + previous) / 2, (current - previous) / 2
    m2, d2 = midpoint**2, difference**2
    odd = midpoint * (Fraction(3, 2) - (m2 + d2) / 2)
    even = m2 + d2 / 3 - m2**2 - 2 * m2 * d2 - d2**2 / 5
    return odd, even


def _saturated_drive_integrals(value: Fraction) -> tuple[Fraction, Fraction]:
    if value > 1:
        return value - Fraction(3, 8), Fraction(2, 15)
    if value < -1:
        return -value - Fraction(3, 8), Fraction(-2, 15)
    return (
        Fraction(3, 4) * value**2 - value**4 / 8,
        value**3 / 3 - value**5 / 5,
    )


def _saturated_drive_adaa(previous: Fraction, current: Fraction) -> tuple[Fraction, Fraction]:
    if previous == current:
        return _drive_adaa(_clamp_unit(previous), _clamp_unit(current))
    odd_previous, even_previous = _saturated_drive_integrals(previous)
    odd_current, even_current = _saturated_drive_integrals(current)
    return (
        (odd_current - odd_previous) / (current - previous),
        (even_current - even_previous) / (current - previous),
    )


def _rail_partition_adaa(previous: Fraction, current: Fraction) -> tuple[Fraction, Fraction]:
    if previous == current:
        return _drive_adaa(_clamp_unit(previous), _clamp_unit(current))
    lower, upper = sorted((previous, current))
    length = upper - lower
    interior = max(Fraction(), min(upper, Fraction(1)) - max(lower, Fraction(-1)))
    below = max(Fraction(), min(upper, Fraction(-1)) - lower)
    above = max(Fraction(), upper - max(lower, Fraction(1)))
    odd, even = _drive_adaa(_clamp_unit(previous), _clamp_unit(current))
    assert interior + below + above == length
    return (interior * odd + above - below) / length, interior * even / length


def _reciprocal_six_steps(distance: float) -> float:
    bits = struct.unpack("<Q", struct.pack("<d", distance))[0]
    exponent = ((bits >> 52) & 0x7FF) - 1023
    inverse_bits = (1023 - exponent - 1) << 52
    inverse = struct.unpack("<d", struct.pack("<Q", inverse_bits))[0]
    for _ in range(6):
        inverse = inverse * (2 - distance * inverse)
    return inverse


def _rail_partition_adaa_f64(previous: float, current: float) -> tuple[float, float, bool]:
    lower, upper = sorted((previous, current))
    if lower >= 1:
        return 1.0, 0.0, False
    if upper <= -1:
        return -1.0, 0.0, False

    clamped_lower, clamped_upper = max(lower, -1.0), min(upper, 1.0)
    midpoint, difference = (
        (clamped_upper + clamped_lower) * 0.5,
        (clamped_upper - clamped_lower) * 0.5,
    )
    m2, d2 = midpoint * midpoint, difference * difference
    odd = midpoint * (1.5 - 0.5 * (m2 + d2))
    even = m2 + d2 * _ONE_THIRD_F64 - (m2 * m2 + 2 * m2 * d2 + d2 * d2 * _ONE_FIFTH_F64)
    if lower >= -1 and upper <= 1:
        return odd, even, False

    distance = upper - lower
    inverse = _reciprocal_six_steps(distance)
    inside = max(0.0, clamped_upper - clamped_lower)
    below = max(0.0, min(upper, -1.0) - lower)
    above = max(0.0, upper - max(lower, 1.0))
    return (inside * odd + above - below) * inverse, (inside * even) * inverse, True


def _cubic_folds(signal: Fraction) -> tuple[Fraction, Fraction]:
    u = _clamp_unit(Fraction(2, 3) * signal)
    envelope = (1 - u * u) ** 3
    return u * envelope, u * u * envelope


def _adaa_cubic_folds(previous: Fraction, current: Fraction) -> tuple[Fraction, Fraction]:
    midpoint, difference = (current + previous) / 2, (current - previous) / 2
    m2, d2 = midpoint**2, difference**2
    odd = midpoint * (
        1
        - 3 * (m2 + d2)
        + 3 * m2**2
        + 10 * m2 * d2
        + 3 * d2**2
        - m2**3
        - 7 * m2**2 * d2
        - 7 * m2 * d2**2
        - d2**3
    )
    even = (
        m2
        + d2 / 3
        - 3 * (m2**2 + 2 * m2 * d2 + d2**2 / 5)
        + 3 * (m2**3 + 5 * m2**2 * d2 + 3 * m2 * d2**2 + d2**3 / 7)
        - (m2**4 + Fraction(28, 3) * m2**3 * d2 + 14 * m2**2 * d2**2 + 4 * m2 * d2**3 + d2**4 / 9)
    )
    return odd, even


@pytest.mark.parametrize(
    ("coefficients", "left_second", "right_second"),
    [
        ([0, 1, 0, -2, 0, 1], -8, 8),
        ([0, 0, 1, 0, -2, 0, 1], 8, 8),
    ],
)
def test_current_squared_wavefolders_are_c1_not_c2(
    coefficients: list[int], left_second: int, right_second: int
) -> None:
    first = _derivative(coefficients)
    second = _derivative(first)
    assert _evaluate(coefficients, Fraction(-1)) == 0
    assert _evaluate(coefficients, Fraction(1)) == 0
    assert _evaluate(first, Fraction(-1)) == 0
    assert _evaluate(first, Fraction(1)) == 0
    assert _evaluate(second, Fraction(-1)) == left_second
    assert _evaluate(second, Fraction(1)) == right_second


@pytest.mark.parametrize(
    "coefficients",
    [[0, 1, 0, -3, 0, 3, 0, -1], [0, 0, 1, 0, -3, 0, 3, 0, -1]],
)
def test_cubic_envelope_has_c2_but_not_c3_rail_joins(coefficients: list[int]) -> None:
    for _ in range(3):
        assert _evaluate(coefficients, Fraction(-1)) == 0
        assert _evaluate(coefficients, Fraction(1)) == 0
        coefficients = _derivative(coefficients)
    assert abs(_evaluate(coefficients, Fraction(-1))) == 48
    assert abs(_evaluate(coefficients, Fraction(1))) == 48


def test_even_adaa_equal_input_matches_its_antiderivative() -> None:
    u = Fraction(1, 2)
    equal_input = u * u - u**4
    inconsistent_instantaneous_claim = u * u * (1 - u * u / 2)
    assert equal_input == Fraction(3, 16)
    assert inconsistent_instantaneous_claim - equal_input == Fraction(1, 32)

    def integral(value: Fraction) -> Fraction:
        return value**3 / 3 - value**5 / 5

    previous, current = Fraction(-2, 3), Fraction(1, 2)
    midpoint, half_difference = (current + previous) / 2, (current - previous) / 2
    factorized = (
        midpoint**2
        + half_difference**2 / 3
        - midpoint**4
        - 2 * midpoint**2 * half_difference**2
        - half_difference**4 / 5
    )
    assert factorized == (integral(current) - integral(previous)) / (current - previous)


def test_clamped_endpoints_lose_information_needed_by_saturated_adaa() -> None:
    previous, first, second = Fraction(), Fraction(1), Fraction(2)
    clamped_first = _drive_adaa(_clamp_unit(previous), _clamp_unit(first))
    clamped_second = _drive_adaa(_clamp_unit(previous), _clamp_unit(second))
    assert clamped_first == clamped_second == (Fraction(5, 8), Fraction(2, 15))
    assert _saturated_drive_adaa(previous, first) == clamped_first
    assert _saturated_drive_adaa(previous, second) == (Fraction(13, 16), Fraction(1, 15))


def test_piecewise_saturated_adaa_matches_independent_antiderivatives() -> None:
    inputs = [Fraction(index, 4) for index in range(-12, 13)]
    for previous in inputs:
        for current in inputs:
            assert _rail_partition_adaa(previous, current) == _saturated_drive_adaa(
                previous, current
            )
            assert _rail_partition_adaa(previous, current) == _rail_partition_adaa(
                current, previous
            )


def test_division_free_piecewise_candidate_against_exact_saturated_adaa() -> None:
    def unpack(bits: int) -> float:
        return struct.unpack("<f", struct.pack("<I", bits))[0]

    low, high = unpack(0x3F7F_FFFF), unpack(0x3F80_0001)
    finite_max = unpack(0x7F7F_FFFF)
    pairs = [
        (0.0, 2.0),
        (low, high),
        (-high, -low),
        (-finite_max, finite_max),
        (0.0, 0.0),
        (finite_max, finite_max),
    ]
    rail_neighbors = [
        sign * unpack(0x3F80_0000 + offset) for offset in range(-16, 17) for sign in (-1, 1)
    ]
    pairs.extend((left, right) for left in rail_neighbors for right in rail_neighbors)
    rng = random.Random(20260605)
    for _ in range(1_000):
        values = [rng.randrange(0x7F80_0000) | (rng.randrange(2) << 31) for _ in range(2)]
        pairs.append((unpack(values[0]), unpack(values[1])))

    crossings = 0
    for previous, current in pairs:
        odd, even, crossed = _rail_partition_adaa_f64(previous, current)
        expected_odd, expected_even = _saturated_drive_adaa(
            Fraction.from_float(previous), Fraction.from_float(current)
        )
        assert abs(Fraction.from_float(odd) - expected_odd) < Fraction(1, 2**44)
        assert abs(Fraction.from_float(even) - expected_even) < Fraction(1, 2**44)
        output_limit = Fraction(1, 2**24) + Fraction(1, 2**44)
        for actual, expected in ((odd, expected_odd), (even, expected_even)):
            rounded = unpack(struct.unpack("<I", struct.pack("<f", actual))[0])
            assert abs(Fraction.from_float(rounded) - expected) < output_limit
        crossings += crossed
    assert crossings > 100


def test_rail_reciprocal_seed_and_iteration_accuracy() -> None:
    for exponent in range(-23, 129):
        for mantissa in (1.0, 1.5, math.nextafter(2.0, 0.0)):
            distance = math.ldexp(mantissa, exponent)
            inverse = _reciprocal_six_steps(distance)
            relative_error = abs(Fraction.from_float(distance) * Fraction.from_float(inverse) - 1)
            assert relative_error < Fraction(1, 2**45)


def test_tone_filters_at_nonzero_rest_need_their_input_and_output_histories() -> None:
    value = Fraction(1, 2)
    low_pole = Fraction(3, 4)
    high_pole = Fraction(7, 8)
    low_state = value
    high_input = value
    high_state = Fraction(0)

    for _ in range(8):
        low_state = low_pole * low_state + (1 - low_pole) * value
        high_state = high_pole * high_state + high_pole * (value - high_input)
        high_input = value
        assert low_state == value
        assert high_state == 0

    low_from_zero = (1 - low_pole) * value
    high_from_zero = high_pole * value
    assert low_from_zero == Fraction(1, 8)
    assert high_from_zero == Fraction(7, 16)


def test_steady_state_initialization_preserves_constant_weight_commutation() -> None:
    pole = Fraction(7, 8)
    weights = (Fraction(3, 2), Fraction(-1))
    rest = (Fraction(1, 2), Fraction(-1, 4))
    signals = (
        rest,
        (Fraction(1), Fraction(2, 3)),
        (Fraction(-1, 3), Fraction(1, 5)),
        rest,
    )

    def project(values: tuple[Fraction, Fraction]) -> Fraction:
        return sum(
            (value * weight for value, weight in zip(values, weights, strict=True)), Fraction()
        )

    lp_states = list(rest)
    hp_inputs = list(rest)
    hp_states = [Fraction(0), Fraction(0)]
    projected_lp = project(rest)
    projected_hp_input = project(rest)
    projected_hp = Fraction(0)

    for values in signals:
        projected = project(values)
        projected_lp = pole * projected_lp + (1 - pole) * projected
        projected_hp = pole * projected_hp + pole * (projected - projected_hp_input)
        projected_hp_input = projected
        for index, value in enumerate(values):
            lp_states[index] = pole * lp_states[index] + (1 - pole) * value
            hp_states[index] = pole * hp_states[index] + pole * (value - hp_inputs[index])
            hp_inputs[index] = value
        assert projected_lp == project((lp_states[0], lp_states[1]))
        assert projected_hp == project((hp_states[0], hp_states[1]))


def test_adaa_blend_and_ac_mean_have_nonzero_resting_history() -> None:
    rest = Fraction(1, 2)
    adaa_rest = _drive_features(rest)[2]
    previous_adaa = adaa_rest
    previous_mean = adaa_rest
    pole = Fraction(9993, 10_000)

    blended = Fraction(7, 8) * adaa_rest + Fraction(1, 8) * previous_adaa
    mean = pole * previous_mean + (1 - pole) * blended
    assert blended == adaa_rest
    assert blended - mean == 0

    mean_from_zero = (1 - pole) * blended
    assert blended - mean_from_zero == pole * adaa_rest


def test_cubic_wavefolder_adaa_matches_exact_antiderivatives() -> None:
    def integral_odd(value: Fraction) -> Fraction:
        return value**2 / 2 - 3 * value**4 / 4 + value**6 / 2 - value**8 / 8

    def integral_even(value: Fraction) -> Fraction:
        return value**3 / 3 - 3 * value**5 / 5 + 3 * value**7 / 7 - value**9 / 9

    inputs = [Fraction(index, 4) for index in range(-4, 5)]
    for previous in inputs:
        for current in inputs:
            odd, even = _adaa_cubic_folds(previous, current)
            if previous == current:
                envelope = (1 - current**2) ** 3
                assert odd == current * envelope
                assert even == current**2 * envelope
            else:
                assert odd == (integral_odd(current) - integral_odd(previous)) / (
                    current - previous
                )
                assert even == (integral_even(current) - integral_even(previous)) / (
                    current - previous
                )


def test_split_wavefolder_adaa_preserves_static_shape_and_linear_term() -> None:
    for current in (Fraction(-1), Fraction(-1, 2), Fraction(0), Fraction(1, 2), Fraction(1)):
        average, _ = _adaa_cubic_folds(current, current)
        result = current + average - current
        assert result == current * (1 - current**2) ** 3

    for exponent in range(4, 12):
        epsilon = Fraction(1, 1 << exponent)
        previous, current = -epsilon / 2, epsilon
        midpoint = (previous + current) / 2
        average, _ = _adaa_cubic_folds(previous, current)
        result = current + average - midpoint
        assert abs(result - current) < 4 * epsilon**3


def test_drive_bank_and_two_fold_functions_need_ten_independent_coordinates() -> None:
    positive = [
        Fraction(1, 100),
        Fraction(1, 50),
        Fraction(1, 25),
        Fraction(1, 20),
        Fraction(7, 100),
        Fraction(2, 25),
        Fraction(1, 10),
        Fraction(1, 5),
        Fraction(1, 3),
        Fraction(1, 2),
        Fraction(1),
        Fraction(4, 3),
        Fraction(2),
        Fraction(4),
        Fraction(8),
    ]
    signals = [-value for value in reversed(positive)] + positive
    bank = [_drive_features(signal) for signal in signals]
    full = [row + list(_cubic_folds(signal)) for row, signal in zip(bank, signals, strict=True)]
    assert _rank(bank) == 8
    assert _rank(full) == 10
    zeros = [Fraction(0)] * 10
    stereo = [row + zeros for row in full] + [zeros + row for row in full]
    assert _rank(stereo) == 20


def test_direct_path_is_dependent_on_two_unclipped_odd_features_at_rest() -> None:
    for index in range(-24, 25):
        x = Fraction(index, 16)
        bank = _drive_features(x)
        assert 64 * bank[0] - bank[2] == 15 * x

    def odd_adaa(previous: Fraction, current: Fraction, drive: Fraction) -> Fraction:
        p = _clamp_unit(Fraction(2, 3) * drive * previous)
        u = _clamp_unit(Fraction(2, 3) * drive * current)
        midpoint, difference = (u + p) / 2, (u - p) / 2
        return midpoint * (Fraction(3, 2) - (midpoint**2 + difference**2) / 2)

    previous, current = Fraction(-1), Fraction(1)
    low = odd_adaa(previous, current, Fraction(1, 4))
    high = odd_adaa(previous, current, Fraction(1))
    assert 64 * low - high == 15 * (previous + current) / 2
    assert (64 * low - high) / 15 != current


def test_recent_drive_inputs_do_not_determine_independent_sag_history() -> None:
    pole, sample = Fraction(15, 16), Fraction(1, 2)
    histories = ([Fraction(0), sample, sample], [Fraction(1), sample, sample])
    states = []
    for history in histories:
        state = history[0]
        for value in history[-2:]:
            state = pole * state + (1 - pole) * value
        states.append(state)
    assert states[1] - states[0] == pole**2

    assert histories[0][-2:] == histories[1][-2:]
    assert _drive_features(histories[0][-1]) == _drive_features(histories[1][-1])
    outputs = [sample * (1 - state / 2) for state in states]
    assert outputs[0] - outputs[1] == sample * pole**2 / 2


def test_smooth_wavefolder_can_alias_without_antialiasing() -> None:
    size = 48
    samples = [math.sin(2 * math.pi * 5 * index / size) for index in range(size)]
    folded = [sample * (1 - sample * sample) ** 3 for sample in samples]
    alias = (
        2
        / size
        * sum(
            sample * math.sin(2 * math.pi * 13 * index / size)
            for index, sample in enumerate(folded)
        )
    )
    assert alias == pytest.approx(-1 / 64, abs=1e-14)


def test_cubic_wavefolder_adaa_reduces_but_does_not_eliminate_one_alias() -> None:
    size = 48
    samples = [math.sin(2 * math.pi * 5 * index / size) for index in range(size)]
    instant = [sample * (1 - sample * sample) ** 3 for sample in samples]
    adaa = [
        float(
            _adaa_cubic_folds(Fraction.from_float(samples[index - 1]), Fraction.from_float(sample))[
                0
            ]
        )
        for index, sample in enumerate(samples)
    ]
    split = [
        value + (sample - samples[index - 1]) / 2
        for index, (sample, value) in enumerate(zip(samples, adaa, strict=True))
    ]

    def alias_amplitude(values: list[float]) -> float:
        sine = (
            2
            / size
            * sum(
                value * math.sin(2 * math.pi * 13 * index / size)
                for index, value in enumerate(values)
            )
        )
        cosine = (
            2
            / size
            * sum(
                value * math.cos(2 * math.pi * 13 * index / size)
                for index, value in enumerate(values)
            )
        )
        return math.hypot(sine, cosine)

    ratio = alias_amplitude(adaa) / alias_amplitude(instant)
    alpha = 5 * math.pi / 48
    assert ratio == pytest.approx(abs(math.sin(8 * alpha) / (8 * math.sin(alpha))), abs=1e-12)
    assert alias_amplitude(split) == pytest.approx(alias_amplitude(adaa), abs=1e-14)
    assert 0.19 < ratio < 0.20


def test_causal_postfilter_cannot_recover_unseen_future_input() -> None:
    first = [Fraction(0)] * 5
    second = first.copy()
    second[4] = Fraction(1)
    now = 2
    assert first[: now + 1] == second[: now + 1]
    assert first[now + 2] != second[now + 2]
    for taps in ([Fraction(1)], [Fraction(1, 2), Fraction(-2), Fraction(1)]):
        outputs = [
            sum(tap * signal[now - delay] for delay, tap in enumerate(taps))
            for signal in (first, second)
        ]
        assert outputs[0] == outputs[1]
