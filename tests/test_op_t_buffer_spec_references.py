"""Independent references for model-declared Op-T delay storage."""

from __future__ import annotations

import math
import random
import re
import struct
from fractions import Fraction
from itertools import product

import pytest


def _ceil(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def _align64(byte_count: int) -> int:
    return (byte_count + 63) & ~63


def _parse_reference_samples(value: str) -> int:
    if re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise ValueError("noncanonical unsigned count")
    count = int(value)
    if count > (1 << 64) - 1:
        raise ValueError("count exceeds u64")
    return count


def _fractional_history_capacity(reference_delay: int, effective_rate: Fraction) -> tuple[int, int]:
    if reference_delay < 4 or not 44_100 <= effective_rate <= 384_000:
        raise ValueError("invalid delay or effective rate")
    logical_samples = _ceil(reference_delay * effective_rate / 48_000) + 2
    return logical_samples, _align64(2 * logical_samples)


def _wrap(index: int, length: int) -> int:
    if length <= 0 or not -length <= index < length:
        raise ValueError("index outside single-wrap range")
    return index + (length & -int(index < 0))


def _next_write(index: int, length: int) -> int:
    if length <= 0 or not 0 <= index < length:
        raise ValueError("invalid write index")
    next_index = index + 1
    return next_index - (length & -int(next_index == length))


def _catmull_weights(mu: Fraction) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    return (
        -mu * (1 - mu) ** 2 / 2,
        1 - Fraction(5, 2) * mu**2 + Fraction(3, 2) * mu**3,
        mu / 2 + 2 * mu**2 - Fraction(3, 2) * mu**3,
        -(mu**2) * (1 - mu) / 2,
    )


def _f32(value: float) -> float:
    return struct.unpack(">f", struct.pack(">f", value))[0]


def _round_q32(value: Fraction) -> tuple[int, int]:
    scaled = value * (1 << 32)
    whole, remainder = divmod(scaled.numerator, scaled.denominator)
    if 2 * remainder > scaled.denominator or (2 * remainder == scaled.denominator and whole & 1):
        whole += 1
    return divmod(whole, 1 << 32)


def _q32_read_position(integer: int, fraction: int) -> Fraction:
    rounded = _f32(_f32(float(fraction)) * 2.0**-32)
    if rounded == 1:
        integer += 1
        rounded = 0.0
    return Fraction(integer) + Fraction.from_float(rounded)


def _bounded_q32_read_position(
    integer: int, fraction: int, low: Fraction, high: Fraction
) -> Fraction:
    target = Fraction(integer) + Fraction(fraction, 1 << 32)
    if not 0 <= fraction < 1 << 32 or not low <= target <= high:
        raise ValueError("invalid or unsafe Q0.32 target")
    rounded = _f32(_f32(float(fraction)) * 2.0**-32)
    position = Fraction(integer) + Fraction.from_float(rounded)
    if low <= position <= high:
        return position
    bits = struct.unpack(">I", struct.pack(">f", rounded))[0]
    bits += -1 if position > high else 1
    adjacent = struct.unpack(">f", struct.pack(">I", bits))[0]
    position = Fraction(integer) + Fraction.from_float(adjacent)
    if not low <= position <= high:
        raise ValueError("no safe representable reader position")
    return position


@pytest.mark.parametrize(
    ("sample_rate", "logical_samples", "bytes_allocated"),
    [
        (44_100, 396_902, 793_856),
        (48_000, 432_002, 864_064),
        (96_000, 864_002, 1_728_064),
        (384_000, 3_456_002, 6_912_064),
    ],
)
def test_nine_second_history_is_exactly_sized_and_aligned(
    sample_rate: int, logical_samples: int, bytes_allocated: int
) -> None:
    length, allocated = _fractional_history_capacity(432_000, Fraction(sample_rate))
    assert (length, allocated) == (logical_samples, bytes_allocated)
    assert allocated % 64 == 0
    assert allocated - 2 * length < 64


def test_read_before_write_avoids_an_unnecessary_aligned_cache_line() -> None:
    logical, allocated = _fractional_history_capacity(30, Fraction(48_000))
    assert logical == 32
    assert allocated == 64
    assert _align64(2 * (30 + 4)) == 128


def test_catmull_rom_reader_has_a_tight_one_and_quarter_peak_bound() -> None:
    for step in range(257):
        mu = Fraction(step, 256)
        weights = _catmull_weights(mu)
        assert sum(weights) == 1
        assert sum(abs(weight) for weight in weights) == 1 + mu * (1 - mu)
        assert sum(abs(weight) for weight in weights) <= Fraction(5, 4)
        assert sum(
            weight * sample for weight, sample in zip(weights, (-1, 1, 1, -1), strict=True)
        ) == (1 + mu * (1 - mu))
    assert Fraction(999, 1000) * Fraction(5, 4) == Fraction(4995, 4000) > 1


def test_fixed_catmull_reader_is_energy_nonexpansive_in_exact_arithmetic() -> None:
    for phase in range(33):
        mu = Fraction(phase, 32)
        weights = _catmull_weights(mu)
        for step in range(33):
            cosine = Fraction(step, 16) - 1
            harmonics = (Fraction(1), cosine, 2 * cosine**2 - 1, 4 * cosine**3 - 3 * cosine)
            power = sum(
                left * right * harmonics[abs(i - j)]
                for i, left in enumerate(weights)
                for j, right in enumerate(weights)
            )
            deficit = (
                mu**2 * (1 - mu) ** 2 * (1 - cosine) ** 2 * (3 + 2 * mu * (1 - mu) * (1 - cosine))
            )
            assert deficit >= 0
            assert power + deficit == 1


def _alternating_adversarial_read(samples: list[Fraction], frame: int) -> Fraction:
    delay = 7 if frame % 2 == 0 else 4
    offsets = (1, 0, -1, -2)
    weights = (Fraction(-1, 16), Fraction(9, 16), Fraction(9, 16), Fraction(-1, 16))
    result = Fraction()
    for weight, offset in zip(weights, offsets, strict=True):
        result += weight * samples[frame - delay + offset + 9]
    return result


def _repeating_sign(frame: int) -> int:
    return 1 if frame % 4 < 2 else -1


def test_unrestricted_cubic_delay_switching_can_amplify_a_subunity_loop() -> None:
    samples = [Fraction(_repeating_sign(n)) for n in range(-9, 0)]
    gain = Fraction(999, 1000)
    growth = gain * Fraction(5, 4)
    for frame in range(9 * 12):
        sample = gain * _alternating_adversarial_read(samples, frame)
        assert sample * _repeating_sign(frame) > 0
        samples.append(sample)
        if (frame + 1) % 9 == 0:
            assert min(abs(value) for value in samples[-9:]) >= growth ** ((frame + 1) // 9)
    assert min(abs(value) for value in samples[-9:]) > 10


def _scalar_feedback_period_matrix(delays: list[Fraction], gain: Fraction) -> list[list[Fraction]]:
    length = _ceil(max(delays)) + 2
    matrix = [[Fraction(i == j) for j in range(length)] for i in range(length)]
    for delay in delays:
        integer = delay.numerator // delay.denominator
        weights = _catmull_weights(delay - integer)
        row = [
            sum(
                (
                    gain * weight * matrix[integer - offset - 1][column]
                    for offset, weight in zip((1, 0, -1, -2), weights, strict=True)
                ),
                Fraction(),
            )
            for column in range(length)
        ]
        matrix = [row, *matrix[:-1]]
    return matrix


def _characteristic_determinant(matrix: list[list[Fraction]], value: Fraction) -> Fraction:
    length = len(matrix)
    work = [[value * (i == j) - matrix[i][j] for j in range(length)] for i in range(length)]
    determinant = Fraction(1)
    for column in range(length):
        if work[column][column] == 0:
            replacement = next(
                (row for row in range(column + 1, length) if work[row][column] != 0),
                None,
            )
            if replacement is None:
                return Fraction()
            work[column], work[replacement] = work[replacement], work[column]
            determinant = -determinant
        pivot = work[column][column]
        determinant *= pivot
        for row in range(column + 1, length):
            ratio = work[row][column] / pivot
            for j in range(column + 1, length):
                work[row][j] -= ratio * work[column][j]
    return determinant


def _one_pole_reference_delays() -> list[Fraction]:
    low, high = Fraction(403, 32), Fraction(625, 32)
    factor = Fraction(25, 26)
    half_factor = factor**16
    current = (low + half_factor * high) / (1 + half_factor)
    initial = current
    delays = []
    for frame in range(32):
        delays.append(current)
        target = high if frame < 16 else low
        current = factor * current + (1 - factor) * target
    assert current == initial
    return delays


@pytest.mark.parametrize("rounded_slew", [False, True])
def test_one_pole_slew_can_destabilize_subunity_cubic_feedback(rounded_slew: bool) -> None:
    low, high = Fraction(403, 32), Fraction(625, 32)
    if rounded_slew:
        gain = Fraction.from_float(_f32(0.99899995))
        coefficient = _f32(1 / 26)
        current = _f32(15.008489048616386)
        delays: list[Fraction] = []
        for cycle in range(2):
            initial = current
            delays = []
            for frame in range(32):
                delays.append(Fraction.from_float(current))
                target = float(high if frame < 16 else low)
                current = _f32(current + _f32(coefficient * _f32(target - current)))
            if cycle == 1:
                assert current == initial
    else:
        gain = Fraction(999, 1000)
        delays = _one_pole_reference_delays()

    assert 0 < gain <= Fraction(999, 1000)
    assert 4 <= low < high <= 20
    assert all(15 <= delay <= 18 for delay in delays)
    assert max(abs(delays[(i + 1) % 32] - delays[i]) for i in range(32)) < Fraction(87, 500)
    period = _scalar_feedback_period_matrix(delays, gain)
    assert _characteristic_determinant(period, Fraction(51, 50)) < 0
    assert _characteristic_determinant(period, Fraction(103, 100)) > 0


def test_saturator_does_not_prevent_slew_limited_small_signal_growth() -> None:
    initial = (
        -0.505103,
        0.291273,
        -0.028459,
        0.025976,
        0.048684,
        0.052742,
        0.059805,
        0.071664,
        0.087376,
        0.109652,
        0.149468,
        0.232669,
        0.402139,
        0.699915,
        1.0,
        -0.497464,
        0.288252,
        -0.011822,
        0.012326,
        0.043259,
    )
    state = [1e-6 * sample for sample in initial]
    reads = []
    for delay in _one_pole_reference_delays():
        integer = delay.numerator // delay.denominator
        weights = tuple(float(weight) for weight in _catmull_weights(delay - integer))
        reads.append((integer, weights))
    for _ in range(400):
        for integer, weights in reads:
            read = sum(
                weight * state[integer - offset - 1]
                for offset, weight in zip((1, 0, -1, -2), weights, strict=True)
            )
            normalized = max(-1.0, min(1.0, 2 * 0.999 * read / (3 * 0.985)))
            sample = 0.985 * normalized * (1.5 - 0.5 * normalized**2)
            assert abs(sample) <= 0.985
            state = [sample, *state[:-1]]
    assert 0.01 < max(abs(sample) for sample in state) < 0.02


def test_scalar_cubic_loop_small_gain_bound_handles_adversarial_reads() -> None:
    samples = [Fraction(16 * _repeating_sign(n)) for n in range(-9, 0)]
    for frame in range(90):
        sample = _repeating_sign(frame) + Fraction(3, 4) * _alternating_adversarial_read(
            samples, frame
        )
        assert sample == 16 * _repeating_sign(frame)
        samples.append(sample)


def _limited_tangent(left: Fraction, right: Fraction) -> Fraction:
    if left > 0 and right > 0:
        return min((left + right) / 2, 3 * min(left, right))
    if left < 0 and right < 0:
        return max((left + right) / 2, 3 * max(left, right))
    return Fraction()


def _shape_preserving_reader(samples: tuple[Fraction, ...], mu: Fraction) -> Fraction:
    before, left, right, after = samples
    slope_left = _limited_tangent(left - before, right - left)
    slope_right = _limited_tangent(right - left, after - right)
    mu2 = mu * mu
    mu3 = mu2 * mu
    return (
        (2 * mu3 - 3 * mu2 + 1) * left
        + (mu3 - 2 * mu2 + mu) * slope_left
        + (-2 * mu3 + 3 * mu2) * right
        + (mu3 - mu2) * slope_right
    )


def test_shape_preserving_cubic_candidate_is_bounded_and_preserves_lines() -> None:
    for values in product(range(-2, 3), repeat=4):
        samples = tuple(Fraction(value) for value in values)
        for step in range(9):
            output = _shape_preserving_reader(samples, Fraction(step, 8))
            assert min(samples[1:3]) <= output <= max(samples[1:3])
    line = tuple(Fraction(value) for value in (-1, 0, 1, 2))
    for step in range(9):
        mu = Fraction(step, 8)
        assert _shape_preserving_reader(line, mu) == mu
    overshoot = tuple(Fraction(value) for value in (-1, 1, 1, -1))
    assert _shape_preserving_reader(overshoot, Fraction(1, 2)) == 1


@pytest.mark.parametrize(
    ("frequency", "candidate_db", "catmull_db"),
    [(4_000, -55.280381, -55.280381), (10_000, -19.644057, -24.488947)],
)
def test_shape_preserving_candidate_changes_isolated_tone_error(
    frequency: int, candidate_db: float, catmull_db: float
) -> None:
    signal_energy = 0.0
    candidate_error = 0.0
    catmull_error = 0.0
    for frame in range(48 * 16):
        samples = tuple(
            math.sin(2 * math.pi * frequency * (frame - 4 + offset) / 48_000)
            for offset in (1, 0, -1, -2)
        )
        reference = math.sin(2 * math.pi * frequency * (frame - 4.5) / 48_000)
        candidate = float(
            _shape_preserving_reader(tuple(Fraction.from_float(x) for x in samples), Fraction(1, 2))
        )
        catmull = (-samples[0] + 9 * samples[1] + 9 * samples[2] - samples[3]) / 16
        signal_energy += reference**2
        candidate_error += (candidate - reference) ** 2
        catmull_error += (catmull - reference) ** 2
    assert math.isclose(
        10 * math.log10(candidate_error / signal_energy), candidate_db, abs_tol=1e-5
    )
    assert math.isclose(10 * math.log10(catmull_error / signal_energy), catmull_db, abs_tol=1e-5)


def test_shape_preserving_scalar_feedback_decays_under_abrupt_delay_switching() -> None:
    samples = [Fraction(_repeating_sign(n)) for n in range(-9, 0)]
    gain = Fraction(999, 1000)
    for frame in range(9 * 12):
        delay = 7 if frame % 2 == 0 else 4
        history = tuple(samples[frame - delay + offset + 9] for offset in (1, 0, -1, -2))
        samples.append(gain * _shape_preserving_reader(history, Fraction(1, 2)))
        if (frame + 1) % 9 == 0:
            assert max(abs(value) for value in samples[-9:]) <= gain ** ((frame + 1) // 9)


def test_granular_pitch_motion_is_distinct_from_base_delay_slew() -> None:
    for ratio, expected_warp in (
        (Fraction(1, 4), Fraction(3, 4)),
        (Fraction(1, 2), Fraction(1, 2)),
        (Fraction(1), Fraction(0)),
        (Fraction(2), Fraction(-1)),
        (Fraction(4), Fraction(-3)),
    ):
        warp = 1 - ratio
        assert warp == expected_warp
        assert 1 - warp == ratio
        if ratio != 1:
            assert abs(warp) > Fraction(1, 4)
    assert math.isclose(12 * math.log2(0.75), -4.98, abs_tol=0.01)
    assert math.isclose(12 * math.log2(1.25), 3.86, abs_tol=0.01)


def test_four_times_source_advance_can_alias_an_input_tone() -> None:
    for n in range(16):
        shifted = math.cos(2 * math.pi * Fraction(4, 5) * n)
        aliased = math.cos(2 * math.pi * Fraction(1, 5) * n)
        assert math.isclose(shifted, aliased, abs_tol=1e-14)


def _grain_window(phase: Fraction) -> Fraction:
    triangle = 1 - abs(2 * phase - 1)
    return triangle**2 * (3 - 2 * triangle)


@pytest.mark.parametrize("warp", [Fraction(-3), Fraction(0), Fraction(3, 4)])
def test_centered_grain_candidate_has_expected_speed_and_history_range(warp: Fraction) -> None:
    duration = 8
    half_span = abs(warp) * duration / 2
    center = 4 + half_span
    previous_phases: tuple[Fraction, Fraction] | None = None
    previous_delays: tuple[Fraction, Fraction] | None = None
    for n in range(2 * duration):
        phase = Fraction(n % duration, duration)
        phases = (phase, (phase + Fraction(1, 2)) % 1)
        delays = tuple(center + warp * duration * (p - Fraction(1, 2)) for p in phases)
        assert all(4 <= delay <= 4 + 2 * half_span for delay in delays)
        assert _grain_window(phases[0]) + _grain_window(phases[1]) == 1
        if warp == 0:
            assert delays == (center, center)
        if previous_phases is not None and previous_delays is not None:
            for p, prev_p, delay, prev_delay in zip(
                phases, previous_phases, delays, previous_delays, strict=True
            ):
                if p > prev_p:
                    assert delay - prev_delay == warp
                else:
                    assert p == 0
                    assert _grain_window(p) == 0
        previous_phases = phases
        previous_delays = (delays[0], delays[1])

    half_span_100ms = abs(warp) * 4_800 / 2
    if warp == -3:
        assert half_span_100ms == 7_200
        assert 4 + half_span_100ms == 7_204
        assert 4 + 2 * half_span_100ms == 14_404


def test_odd_duration_grain_wrap_can_have_nonzero_window_weight() -> None:
    phases = [(Fraction(n, 7) + Fraction(1, 2)) % 1 for n in range(7)]
    assert phases[3:5] == [Fraction(13, 14), Fraction(1, 14)]
    assert _grain_window(phases[3]) == _grain_window(phases[4]) == Fraction(19, 343)


def test_small_grain_warp_can_vanish_when_added_to_f32_total_delay() -> None:
    center = _f32(7_204.0)
    warp = _f32(-5.78e-10)
    displacement = warp * 4_800 / 2
    assert displacement != 0
    assert _f32(center + displacement) == center
    exact_position = Fraction.from_float(center) + Fraction.from_float(warp) * 2_400
    integer, fraction = _round_q32(exact_position)
    assert (integer, fraction) != (7_204, 0)


def test_decimal_text_retains_u64_values_json_numbers_cannot() -> None:
    count = (1 << 53) + 1
    assert float(count) == float(count - 1)
    assert _parse_reference_samples(str(count)) == count
    assert _parse_reference_samples(str((1 << 64) - 1)) == (1 << 64) - 1


@pytest.mark.parametrize(
    "value",
    ["", "0", "04", "-4", "+4", "4.0", "4e0", " 4", "٤", "18446744073709551616"],
)
def test_reference_sample_counts_reject_noncanonical_or_oversized_text(value: str) -> None:
    with pytest.raises(ValueError, match="count"):
        _parse_reference_samples(value)


def test_fixed_allpass_read_before_write_needs_only_its_logical_delay() -> None:
    delay = 1784
    allocated_fp16 = _align64(2 * delay)
    allocated_f32 = _align64(4 * delay)
    assert allocated_fp16 == 3584
    assert allocated_f32 == 7168
    assert 16 * allocated_fp16 == 56 * 1024
    assert 16 * allocated_f32 == 112 * 1024


def test_converted_allpass_lengths_need_not_remain_coprime() -> None:
    reference = (19, 31, 47, 73, 103, 139, 179, 223)
    assert all(
        math.gcd(left, right) == 1
        for i, left in enumerate(reference)
        for right in reference[i + 1 :]
    )

    def convert(delay: int, rate: int) -> int:
        numerator = delay * rate
        whole, remainder = divmod(numerator, 48_000)
        return max(1, whole + int(2 * remainder >= 48_000))

    at_96k = tuple(convert(delay, 96_000) for delay in reference)
    assert at_96k[:2] == (38, 62)
    assert math.gcd(*at_96k[:2]) == 2
    at_44k1 = tuple(convert(delay, 44_100) for delay in reference)
    assert at_44k1 == (17, 28, 43, 67, 95, 128, 164, 205)
    assert math.gcd(at_44k1[4], at_44k1[7]) == 5
    assert convert(19, 72_000) == 29


@pytest.mark.parametrize("coefficient", [Fraction(-3, 4), Fraction(3, 4)])
def test_allpass_unit_frequency_gain_does_not_limit_sample_peaks(
    coefficient: Fraction,
) -> None:
    count = 64
    history = Fraction()
    output = Fraction()
    for frame in range(count + 1):
        lag = count - frame
        impulse = coefficient if lag == 0 else (1 - coefficient**2) * (-coefficient) ** (lag - 1)
        sample = Fraction(1 if impulse > 0 else -1)
        state = sample - coefficient * history
        output = coefficient * state + history
        history = state
    magnitude = abs(coefficient)
    assert output == magnitude + (1 + magnitude) * (1 - magnitude**count)
    assert Fraction(249, 100) < output < Fraction(5, 2)


def test_allpass_state_can_approach_four_times_bounded_input() -> None:
    coefficient = Fraction(3, 4)
    state = Fraction()
    for frame in range(64):
        sample = Fraction(1 if frame % 2 == 0 else -1)
        state = sample - coefficient * state
        assert abs(state) <= 4
    assert abs(state) == 4 * (1 - coefficient**64)
    assert abs(state) > Fraction(399, 100)


@pytest.mark.parametrize("delay", [1, 3, 7])
def test_time_varying_allpass_respects_nonzero_initial_state_bound(delay: int) -> None:
    history = [Fraction(9 * _repeating_sign(i)) for i in range(-delay, 0)]
    coefficients = (Fraction(3, 4), Fraction(-3, 4), Fraction(0))
    for frame in range(120):
        state = Fraction(_repeating_sign(frame)) - coefficients[frame % 3] * history[frame]
        assert abs(state) <= 9
        history.append(state)


@pytest.mark.parametrize("delay", [1, 19, 223, 1784])
def test_fixed_allpass_ring_preserves_its_exact_delay(delay: int) -> None:
    ring = [0] * delay
    write_index = 0
    for frame in range(2 * delay + 2):
        expected = frame - delay + 1 if frame >= delay else 0
        assert ring[write_index] == expected
        ring[write_index] = frame + 1
        write_index = _next_write(write_index, delay)


def test_single_wrap_matches_modulo_for_arbitrary_ring_lengths() -> None:
    for length in range(8, 80):
        for index in range(-length, length):
            assert _wrap(index, length) == index % length
        for index in range(length):
            assert _next_write(index, length) == (index + 1) % length
    for length in (511, 432_002, 3_456_002):
        for index in (-length, -1, 0, 1, length - 1):
            assert _wrap(index, length) == index % length


def test_all_catmull_rom_neighbors_fit_the_single_wrap_domain() -> None:
    for length in range(8, 80):
        for write_index in range(length):
            for delay in range(4, length - 1):
                for offset in (1, 0, -1, -2):
                    raw = write_index - delay + offset
                    assert -length <= raw < length
                    assert _wrap(raw, length) == raw % length


@pytest.mark.parametrize("length", [13, 37, 100])
def test_wrapped_reads_match_the_unbounded_input_history(length: int) -> None:
    ring = [0] * length
    samples: list[int] = []
    write_index = 0
    for frame in range(3 * length):
        for delay in (4, length - 2):
            for offset in (1, 0, -1, -2):
                source_frame = frame - delay + offset
                expected = samples[source_frame] if source_frame >= 0 else 0
                assert ring[_wrap(write_index - delay + offset, length)] == expected
        value = frame + 1
        samples.append(value)
        ring[write_index] = value
        write_index = _next_write(write_index, length)


def test_fractional_rate_rounding_must_not_undercount_the_ceiling() -> None:
    for reference_delay in range(4, 128):
        for converted_delay in range(1, reference_delay * 8 + 1):
            rate = float(Fraction(converted_delay * 48_000, reference_delay))
            if not 44_100 <= rate <= 384_000:
                continue
            exact = reference_delay * Fraction.from_float(rate) / 48_000
            rounded = reference_delay * rate / 48_000
            if exact > converted_delay and rounded == converted_delay:
                assert _ceil(exact) == converted_delay + 1
                assert math.ceil(rounded) == converted_delay
                return
    raise AssertionError("no finite-precision ceiling counterexample found")


def test_long_f32_delay_cannot_represent_all_small_fractional_offsets() -> None:
    assert _f32(432_000 + 1 / 64) == 432_000
    assert _f32(3_456_000 + 1 / 8) == 3_456_000
    assert _f32(1 / 64) == 1 / 64
    assert _f32(1 / 8) == 1 / 8


def test_long_f32_delay_quantization_can_exceed_ultra_error_target() -> None:
    for sample_rate, integer, fraction in (
        (48_000, 432_000, Fraction(1, 64)),
        (384_000, 3_456_000, Fraction(1, 8)),
    ):
        target = Fraction(integer) + fraction
        rounded = Fraction.from_float(_f32(float(target)))
        phase = math.pi * 20_000 * float(abs(target - rounded)) / sample_rate
        relative_error_db = 20 * math.log10(2 * math.sin(phase))
        assert -28 < relative_error_db < -27


def test_split_q32_position_has_bounded_fraction_and_carry_error() -> None:
    rng = random.Random(0xD31A)
    fractions = [Fraction(0), Fraction(1, 64), Fraction(1, 8), Fraction((1 << 40) - 1, 1 << 40)]
    fractions.extend(Fraction(rng.getrandbits(52), 1 << 52) for _ in range(10_000))
    for integer in (4, 432_000, 3_456_000):
        for fraction in fractions:
            target = Fraction(integer) + fraction
            whole, q32 = _round_q32(target)
            assert 0 <= q32 < 1 << 32
            decoded = _q32_read_position(whole, q32)
            assert abs(decoded - target) <= Fraction(1, 1 << 33) + Fraction(1, 1 << 25)
    assert _q32_read_position(432_000, (1 << 32) - 1) == 432_001


def test_f32_reader_rounding_can_cross_fractional_safety_bounds() -> None:
    integer = 4
    half_word = 1 << 31
    lower_target = Fraction(integer) + Fraction(half_word + 1, 1 << 32)
    upper_target = Fraction(integer) + Fraction(half_word - 1, 1 << 32)
    assert _q32_read_position(integer, half_word + 1) < lower_target
    assert _q32_read_position(integer, half_word - 1) > upper_target
    upward = _bounded_q32_read_position(integer, half_word + 1, lower_target, Fraction(5))
    downward = _bounded_q32_read_position(integer, half_word - 1, Fraction(4), upper_target)
    assert lower_target <= upward <= 5
    assert 4 <= downward <= upper_target
    assert abs(upward - lower_target) < Fraction(1, 1 << 24)
    assert abs(downward - upper_target) < Fraction(1, 1 << 24)


def test_f32_reader_carry_must_respect_fractional_upper_bound() -> None:
    upper = Fraction(5) - Fraction(1, 1 << 32)
    assert _q32_read_position(4, (1 << 32) - 1) == 5 > upper
    safe = _bounded_q32_read_position(4, (1 << 32) - 1, Fraction(4), upper)
    assert safe == Fraction(5) - Fraction(1, 1 << 24)
    assert 4 <= safe <= upper


def test_inward_reader_adjustments_remain_inside_bounds_and_one_f32_step() -> None:
    rng = random.Random(0xB0A1D)
    for fraction in (0, 1, (1 << 31) - 1, 1 << 31, (1 << 32) - 1):
        target = Fraction(4) + Fraction(fraction, 1 << 32)
        if _q32_read_position(4, fraction) == target:
            assert _bounded_q32_read_position(4, fraction, target, target) == target
        else:
            with pytest.raises(ValueError, match="no safe representable reader"):
                _bounded_q32_read_position(4, fraction, target, target)
    for _ in range(4096):
        fraction = rng.getrandbits(32)
        target = Fraction(4) + Fraction(fraction, 1 << 32)
        for low, high in ((Fraction(4), target), (target, Fraction(5))):
            position = _bounded_q32_read_position(4, fraction, low, high)
            assert low <= position <= high
            assert abs(position - target) < Fraction(1, 1 << 24)


def test_safe_q32_interval_can_have_no_safe_f32_reader_position() -> None:
    low = Fraction(4) + Fraction((1 << 31) + 1, 1 << 32)
    high = low + Fraction(1, 1 << 32)
    with pytest.raises(ValueError, match="no safe representable reader"):
        _bounded_q32_read_position(4, (1 << 31) + 1, low, high)
    with pytest.raises(ValueError, match=r"unsafe Q0\.32"):
        _bounded_q32_read_position(4, 0, low, high)


def test_transition_admission_must_account_for_both_live_allocations() -> None:
    old_history = _fractional_history_capacity(432_000, Fraction(48_000))[1]
    new_history = _fractional_history_capacity(432_000, Fraction(96_000))[1]
    ceiling = 2_000_000
    assert new_history <= ceiling
    assert old_history + new_history > ceiling
