"""Repeated-input modulation references, not Studio or hardware tests."""

from __future__ import annotations

import math
from fractions import Fraction
from itertools import combinations, pairwise

import numpy as np
import numpy.typing as npt

from test_excitation_schedule_spec_references import _periodic_multitone_candidate

FloatArray = npt.NDArray[np.float64]

_RATE = 1_000
_STARTS = (0, 1_137, 2_474)
_LENGTH = 1_000


def _carrier() -> FloatArray:
    return np.random.default_rng(20260607).uniform(-0.75, 0.75, _LENGTH)


def _whiten_differences(first: FloatArray, second: FloatArray) -> FloatArray:
    return np.concatenate((first / math.sqrt(2), (2 * second - first) / math.sqrt(6)))


def _design(carrier: FloatArray, frequency: float) -> FloatArray:
    time = np.arange(carrier.size, dtype=np.float64) / _RATE
    phase = 2 * math.pi * frequency * time
    reference = np.column_stack((np.cos(phase), np.sin(phase)))
    differences = []
    for start in _STARTS[1:]:
        shifted = phase + 2 * math.pi * frequency * start / _RATE
        columns = np.column_stack((np.cos(shifted), np.sin(shifted)))
        differences.append(carrier[:, None] * (columns - reference))
    first, second = differences
    return np.vstack((first / math.sqrt(2), (2 * second - first) / math.sqrt(6)))


def _contrast(count: int) -> FloatArray:
    matrix = np.zeros((count - 1, count), dtype=np.float64)
    for row in range(count - 1):
        scale = math.sqrt((row + 1) * (row + 2))
        matrix[row, : row + 1] = -1 / scale
        matrix[row, row + 1] = (row + 1) / scale
    return matrix


def _contrast_design(
    carrier: FloatArray, starts: tuple[int, ...], frequency: float, rate: int = _RATE
) -> FloatArray:
    time = np.arange(carrier.size, dtype=np.float64) / rate
    phase = 2 * math.pi * frequency * (np.asarray(starts)[:, None] / rate + time[None, :])
    samples = np.stack((np.cos(phase), np.sin(phase)), axis=-1) * carrier[None, :, None]
    return (_contrast(len(starts)) @ samples.transpose(1, 0, 2)).transpose(1, 0, 2).reshape(-1, 2)


def _frequency_lipschitz(carrier: FloatArray, starts: tuple[int, ...], rate: int) -> float:
    times = np.arange(carrier.size, dtype=np.float64) / rate
    absolute_times = np.asarray(starts, dtype=np.float64)[:, None] / rate + times[None, :]
    return 2 * math.pi * math.sqrt(float(np.sum(absolute_times**2 * carrier[None, :] ** 2)))


def _contrast_singular_profile(
    starts: tuple[int, ...], length: int, frequencies: FloatArray
) -> FloatArray:
    time = np.arange(length, dtype=np.float64) / _RATE
    temporal = np.sum(np.exp(4j * math.pi * frequencies[:, None] * time[None, :]), axis=1)
    phases = np.exp(2j * math.pi * frequencies[:, None] * np.asarray(starts)[None, :] / _RATE)
    centered = phases - np.mean(phases, axis=1, keepdims=True)
    energy = np.sum(np.abs(centered) ** 2, axis=1)
    doubled = np.abs(np.sum(centered**2, axis=1))
    return np.sqrt(np.maximum(0.25 * (length * energy - np.abs(temporal) * doubled) / 2, 0))


def _minimum_squared_phase_distance(
    starts: tuple[int, ...], rate: int, low: Fraction, high: Fraction
) -> Fraction:
    distances = [abs(right - left) for left, right in combinations(starts, 2)]
    assert all(distance > 0 for distance in distances)
    boundaries = {low, high}
    for distance in distances:
        for index in range(int(high * distance / rate) + 1):
            point = Fraction((2 * index + 1) * rate, 2 * distance)
            if low < point < high:
                boundaries.add(point)

    candidates = []
    ordered = sorted(boundaries)
    for left, right in pairwise(ordered):
        middle = (left + right) / 2
        nearest = [(middle * distance / rate + Fraction(1, 2)) // 1 for distance in distances]
        slopes = [Fraction(distance, rate) for distance in distances]
        vertex = sum(
            (slope * integer for slope, integer in zip(slopes, nearest, strict=True)), Fraction()
        ) / sum((slope**2 for slope in slopes), Fraction())
        position = min(right, max(left, vertex))
        candidates.append(
            sum(
                (
                    (slope * position - integer) ** 2
                    for slope, integer in zip(slopes, nearest, strict=True)
                ),
                Fraction(),
            )
        )
    return min(candidates)


def _static_response(carrier: FloatArray) -> FloatArray:
    return np.clip(
        0.55 * carrier + 0.2 * carrier**3 + 0.15 * carrier**5 + 0.05 * np.sin(9 * carrier),
        -0.8,
        0.8,
    )


def _paired_wet(carrier: FloatArray, frequency: float, phase: float) -> FloatArray:
    time = np.arange(carrier.size, dtype=np.float64) / _RATE
    static = _static_response(carrier)
    return np.stack(
        [
            static
            + 0.12 * carrier * np.cos(2 * math.pi * frequency * (time + start / _RATE) + phase)
            for start in _STARTS
        ]
    )


def test_paired_differences_remove_any_tested_static_waveshaper() -> None:
    carrier = _carrier()
    for frequency in (0.25, 1 / 1.137, 4.0, 10.0):
        for phase in (0.0, 0.7, math.pi / 2):
            wet = _paired_wet(carrier, frequency, phase)
            assert float(np.max(np.abs(wet))) < 1
            observed = _whiten_differences(wet[1] - wet[0], wet[2] - wet[0])
            design = _design(carrier, frequency)
            coefficients, _, rank, _ = np.linalg.lstsq(design, observed, rcond=None)
            assert rank == 2
            np.testing.assert_allclose(
                coefficients, [0.12 * math.cos(phase), -0.12 * math.sin(phase)], atol=1e-12
            )
            np.testing.assert_allclose(design @ coefficients, observed, atol=1e-12)

    static = np.stack([_static_response(carrier) for _ in _STARTS])
    differences = _whiten_differences(static[1] - static[0], static[2] - static[0])
    np.testing.assert_array_equal(differences, np.zeros(2 * carrier.size))


def test_a_second_offset_removes_one_exact_single_pair_blind_spot() -> None:
    carrier = _carrier()
    frequency = _RATE / _STARTS[1]
    time = np.arange(carrier.size, dtype=np.float64) / _RATE
    phase = 2 * math.pi * frequency * time
    baseline = carrier * np.cos(phase)
    first = carrier * np.cos(phase + 2 * math.pi * frequency * _STARTS[1] / _RATE) - baseline
    second = carrier * np.cos(phase + 2 * math.pi * frequency * _STARTS[2] / _RATE) - baseline
    assert float(np.max(np.abs(first))) < 1e-12
    assert float(np.max(np.abs(second))) > 0.5
    assert math.gcd(_STARTS[1], _STARTS[2]) == 1
    assert np.linalg.matrix_rank(_design(carrier, frequency)) == 2


def test_coprime_offsets_can_still_have_a_near_common_blind_spot() -> None:
    frequency = 5.258
    phase_response = math.hypot(
        math.sin(math.pi * frequency * _STARTS[1] / _RATE),
        math.sin(math.pi * frequency * _STARTS[2] / _RATE),
    )
    assert 0.07 < phase_response < 0.08
    weak = np.linalg.svd(_design(_carrier(), frequency), compute_uv=False)[-1]
    comparison = np.linalg.svd(_design(_carrier(), 4.0), compute_uv=False)[-1]
    assert weak < 0.15 * comparison


def test_shared_reference_noise_requires_correlated_pair_whitening() -> None:
    samples = np.random.default_rng(20260608).standard_normal((100_000, 3, 2))
    first = samples[:, 1] - samples[:, 0]
    second = samples[:, 2] - samples[:, 0]
    np.testing.assert_allclose(
        np.cov(np.column_stack((first[:, 0], second[:, 0])), rowvar=False),
        [[2.0, 1.0], [1.0, 2.0]],
        atol=0.02,
    )
    whitened_first = first / math.sqrt(2)
    whitened_second = (2 * second - first) / math.sqrt(6)
    np.testing.assert_allclose(
        np.cov(np.column_stack((whitened_first[:, 0], whitened_second[:, 0])), rowvar=False),
        np.eye(2),
        atol=0.02,
    )

    naive_coordinates = (first + second) / 2
    threshold = 2 * math.log(1 / 0.05)
    naive_alarm = np.mean(np.sum(naive_coordinates**2, axis=1) >= threshold)
    corrected_alarm = np.mean(np.sum(naive_coordinates**2, axis=1) * (2 / 3) >= threshold)
    assert 0.12 < naive_alarm < 0.15
    assert 0.045 < corrected_alarm < 0.055


def test_whitened_fixed_grid_controls_a_conditional_static_null() -> None:
    carrier = _carrier()
    frequencies = (0.25, 0.5, 1 / 1.137, 1.0, 4.0, 10.0)
    alternatives = []
    for frequency in frequencies:
        design = _design(carrier, frequency)
        assert np.linalg.matrix_rank(design) == 2
        basis, _ = np.linalg.qr(design)
        alternatives.append(basis)

    combined = np.column_stack(alternatives)
    assert np.linalg.matrix_rank(combined) == 2 * len(frequencies)
    span, _ = np.linalg.qr(combined)
    draws = np.random.default_rng(20260609).standard_normal((40_000, span.shape[1]))
    scores = np.column_stack(
        [np.sum((draws @ (span.T @ basis)) ** 2, axis=1) for basis in alternatives]
    )
    threshold = 2 * math.log(len(frequencies) / 0.05)
    fraction = float(np.mean(np.max(scores, axis=1) >= threshold))
    assert 0.02 < fraction < 0.06


def test_singular_value_gives_a_conditional_unknown_phase_power_bound() -> None:
    design = _design(_carrier(), 0.25)
    basis, _ = np.linalg.qr(design)
    singular_values = np.linalg.svd(design, compute_uv=False)
    minimum_singular = float(singular_values[-1])
    assert minimum_singular > 0
    alpha, miss_target, count = 0.05, 0.1, 6
    threshold = 2 * math.log(count / alpha)
    required = math.sqrt(threshold) + math.sqrt(2 * math.log(1 / miss_target))
    amplitude = 0.12
    sigma = amplitude * minimum_singular / required
    noise = np.random.default_rng(20260610).standard_normal((40_000, 2))
    for phase in np.linspace(0, 2 * math.pi, 16, endpoint=False):
        coefficients = amplitude * np.array([math.cos(phase), -math.sin(phase)])
        mean = (basis.T @ design @ coefficients) / sigma
        noncentrality = float(np.dot(mean, mean))
        assert math.sqrt(noncentrality) >= required - 1e-12
        miss_bound = math.exp(-((math.sqrt(noncentrality) - math.sqrt(threshold)) ** 2) / 2)
        missed = float(np.mean(np.sum((noise + mean) ** 2, axis=1) < threshold))
        assert miss_bound <= miss_target + 1e-12
        assert missed < miss_bound + 0.002


def test_off_grid_gain_signal_has_a_conditional_projection_and_power_bound() -> None:
    carrier = _carrier()
    grid_frequency = 4.0
    true_frequency = 4.001
    grid_design = _design(carrier, grid_frequency)
    true_design = _design(carrier, true_frequency)
    basis, _ = np.linalg.qr(grid_design)
    minimum_singular = float(np.linalg.svd(grid_design, compute_uv=False)[-1])
    lipschitz = _frequency_lipschitz(carrier, _STARTS, _RATE)
    mismatch = abs(true_frequency - grid_frequency)
    assert float(np.linalg.norm(true_design - grid_design, ord=2)) <= lipschitz * mismatch
    amplitude = 0.12
    lower_mean_norm = amplitude * (minimum_singular - lipschitz * mismatch)
    assert lower_mean_norm > 0

    alpha, miss_target, count = 0.05, 0.1, 6
    threshold = 2 * math.log(count / alpha)
    required = math.sqrt(threshold) + math.sqrt(2 * math.log(1 / miss_target))
    sigma = lower_mean_norm / required
    noise = np.random.default_rng(20260612).standard_normal((30_000, 2))
    for phase in np.linspace(0, 2 * math.pi, 16, endpoint=False):
        coefficients = amplitude * np.array([math.cos(phase), -math.sin(phase)])
        projected = basis.T @ true_design @ coefficients
        assert float(np.linalg.norm(projected)) >= lower_mean_norm - 1e-12
        mean = projected / sigma
        assert float(np.linalg.norm(mean)) >= required - 1e-12
        missed = float(np.mean(np.sum((noise + mean) ** 2, axis=1) < threshold))
        assert missed < miss_target


def test_four_repeats_improve_a_sampled_worst_case_sensitivity() -> None:
    starts = (0, 833, 1_747, 2_747)
    contrast = _contrast(len(starts))
    np.testing.assert_allclose(contrast @ contrast.T, np.eye(3), atol=1e-15)
    np.testing.assert_allclose(contrast @ np.ones(4), np.zeros(3), atol=1e-15)
    np.testing.assert_allclose(contrast.T @ contrast, np.eye(4) - np.ones((4, 4)) / 4, atol=1e-15)
    assert math.gcd(*starts[1:]) == 1
    assert starts[-1] + 750 <= 3_500
    assert min(np.diff(starts)) >= 750

    frequencies = np.linspace(0.25, 10.0, 1_501)
    three_profile = _contrast_singular_profile(_STARTS, _LENGTH, frequencies)
    four_profile = _contrast_singular_profile(starts, 750, frequencies)
    ratio = float(np.min(four_profile) / np.min(three_profile))
    assert ratio > 2, ratio
    carrier = 0.5 * np.random.default_rng(20260611).choice((-1.0, 1.0), size=750)
    for index in (0, 100, 500, 1_000, 1_500):
        design = _contrast_design(carrier, starts, float(frequencies[index]))
        direct = np.linalg.svd(design, compute_uv=False)[-1]
        assert math.isclose(four_profile[index], direct, rel_tol=1e-12, abs_tol=1e-12)
    np.testing.assert_allclose(
        _contrast_design(_carrier(), _STARTS, 4.0), _design(_carrier(), 4.0), atol=1e-14
    )


def test_four_window_toy_has_a_continuous_band_sensitivity_lower_bound() -> None:
    starts = (0, 833, 1_747, 2_747)
    minimum_distance = _minimum_squared_phase_distance(starts, _RATE, Fraction(1, 4), Fraction(10))
    assert minimum_distance > Fraction(1, 100), minimum_distance

    length = 750
    weighted_pairs = sum((length - lag) * lag**2 for lag in range(1, 26))
    assert weighted_pairs == 4_038_125
    temporal_bound = Fraction(weighted_pairs, 2 * length * _RATE**2)
    assert temporal_bound > Fraction(1, 500)
    eigenvalue_bound = 4 * minimum_distance * temporal_bound / 2
    assert eigenvalue_bound > Fraction(1, 160) ** 2

    frequencies = np.linspace(0.25, 10.0, 1_501)
    profile = _contrast_singular_profile(starts, length, frequencies)
    assert float(np.min(profile)) > 1 / 160


def test_uniform_lipschitz_checker_independently_bounds_phase_distances() -> None:
    starts = (0, 833, 1_747, 2_747)
    distances = [abs(right - left) for left, right in combinations(starts, 2)]
    lipschitz = Fraction(sum(distances), _RATE)
    step = Fraction(1, 1_000)
    minimum = None
    for cell in range(9_750):
        middle = Fraction(1, 4) + (cell + Fraction(1, 2)) * step
        value = Fraction()
        for distance in distances:
            phase = middle * distance / _RATE
            nearest = (phase + Fraction(1, 2)) // 1
            value += (phase - nearest) ** 2
        minimum = value if minimum is None else min(minimum, value)
    assert minimum is not None
    assert minimum - lipschitz * step / 2 > Fraction(1, 100)


def test_48khz_multitone_has_a_continuous_band_sensitivity_lower_bound() -> None:
    block = _periodic_multitone_candidate()
    assert block.size == 1_200
    assert float(np.max(np.abs(block))) <= 0.980_001
    energy = sum((Fraction.from_float(float(sample)) ** 2 for sample in block), Fraction())
    assert energy > 100

    starts = (0, 833, 1_747, 2_747)
    minimum_distance = _minimum_squared_phase_distance(starts, _RATE, Fraction(1, 4), Fraction(10))
    temporal_bound = Fraction(58, 57_600_000) * energy
    assert temporal_bound > Fraction(1, 10_000)
    assert 4 * minimum_distance * temporal_bound / 2 > Fraction(1, 1_000) ** 2

    carrier = np.tile(block, 30).astype(np.float64)
    starts_48k = tuple(start * 48 for start in starts)
    assert len(carrier) == 36_000
    assert starts_48k[-1] + carrier.size == 167_856
    for frequency in (0.25, 4.0, 5.258, 10.0):
        singular = np.linalg.svd(
            _contrast_design(carrier, starts_48k, frequency, 48_000), compute_uv=False
        )[-1]
        assert singular > 1 / 1_000


def test_tapered_48khz_multitone_retains_a_conditional_continuous_band_bound() -> None:
    block = _periodic_multitone_candidate()
    original = np.tile(block, 30)
    window = original.copy()
    phase = np.linspace(0, 1, 240, dtype=np.float64)
    fade = phase**2 * (3 - 2 * phase)
    window[:240] = np.asarray(window[:240].astype(np.float64) * fade, dtype=np.float32)
    window[-240:] = np.asarray(window[-240:].astype(np.float64) * fade[::-1], dtype=np.float32)
    assert float(window[0]) == float(window[-1]) == 0.0
    assert np.all(np.abs(window) <= np.abs(original))
    np.testing.assert_array_equal(window[1_200 : 29 * 1_200], original[1_200 : 29 * 1_200])
    assert float(np.sum(window.astype(np.float64) ** 2)) >= (
        28 * float(np.sum(block.astype(np.float64) ** 2))
    )

    energy = sum((Fraction.from_float(float(sample)) ** 2 for sample in block), Fraction())
    assert energy > 100
    minimum_distance = _minimum_squared_phase_distance(
        (0, 833, 1_747, 2_747), _RATE, Fraction(1, 4), Fraction(10)
    )
    temporal_bound = Fraction(54, 57_600_000) * energy
    assert temporal_bound > Fraction(1, 12_000)
    assert 4 * minimum_distance * temporal_bound / 2 > Fraction(1, 1_000) ** 2

    starts_48k = (0, 39_984, 83_856, 131_856)
    for frequency in (0.25, 4.0, 5.258, 10.0):
        singular = np.linalg.svd(
            _contrast_design(window.astype(np.float64), starts_48k, frequency, 48_000),
            compute_uv=False,
        )[-1]
        assert singular > 1 / 1_000


def test_time_invariant_envelope_memory_can_survive_paired_cancellation() -> None:
    carrier = _carrier()
    signal = np.zeros(_STARTS[-1] + carrier.size, dtype=np.float64)
    for start in _STARTS:
        signal[start : start + carrier.size] = carrier

    decay = math.exp(-1 / (_RATE * 0.6))
    envelope = 0.0
    wet = np.empty_like(signal)
    for index, value in enumerate(signal):
        envelope = decay * envelope + (1 - decay) * abs(float(value))
        wet[index] = value * (0.5 + 0.5 * envelope)

    blocks = np.stack([wet[start : start + carrier.size] for start in _STARTS])
    differences = _whiten_differences(blocks[1] - blocks[0], blocks[2] - blocks[0])
    fit, _, rank, _ = np.linalg.lstsq(_design(carrier, 0.25), differences, rcond=None)
    assert rank == 2
    assert float(np.linalg.norm(differences)) > 0.1
    assert float(np.linalg.norm(fit)) > 0.001
