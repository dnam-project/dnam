"""Synthetic Stage II basis comparisons, not device or engine certification."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

type _FloatArray = npt.NDArray[np.float64]


def _quadrature_features(
    current: _FloatArray, previous: _FloatArray
) -> tuple[_FloatArray, _FloatArray]:
    nodes, weights = np.polynomial.legendre.leggauss(5)
    weights = weights * 0.5
    columns = []
    fold_points = None
    for drive in (0.25, 1.0, 4.0, 16.0):
        u = np.clip((2.0 / 3.0) * drive * current, -1.0, 1.0)
        p = np.clip((2.0 / 3.0) * drive * previous, -1.0, 1.0)
        points = ((u + p) * 0.5)[:, None] + ((u - p) * 0.5)[:, None] * nodes
        squared = points * points
        columns.extend(
            (
                np.sum(points * (1.5 - 0.5 * squared) * weights, axis=1),
                np.sum(squared * (1.0 - squared) * weights, axis=1),
            )
        )
        if drive == 1.0:
            fold_points = points
    assert fold_points is not None
    squared = fold_points * fold_points
    envelope = (1.0 - squared) ** 3
    folds = np.column_stack(
        (
            np.sum(fold_points * envelope * weights, axis=1),
            np.sum(squared * envelope * weights, axis=1),
        )
    )
    return np.column_stack(columns), folds


def _blend_and_ac_periodic(signal: _FloatArray) -> _FloatArray:
    count = signal.shape[0]
    delay = np.exp(-2j * np.pi * np.arange(count // 2 + 1) / count)
    pole = 0.9993
    response = (0.875 + 0.125 * delay) * pole * (1.0 - delay) / (1.0 - pole * delay)
    transformed = np.fft.rfft(signal, axis=0)
    return np.fft.irfft(transformed * response[:, None], n=count, axis=0)


def _tone_bank_periodic(signal: _FloatArray) -> _FloatArray:
    count = signal.shape[0]
    delay = np.exp(-2j * np.pi * np.arange(count // 2 + 1) / count)
    transformed = np.fft.rfft(signal, axis=0)
    columns = [signal]
    for cutoff, highpass in ((80.0, False), (400.0, False), (1200.0, True)):
        pole = np.exp(-2.0 * np.pi * cutoff / 48_000.0)
        numerator = pole * (1.0 - delay) if highpass else 1.0 - pole
        response = numerator / (1.0 - pole * delay)
        columns.append(np.fft.irfft(transformed * response[:, None], n=count, axis=0))
    return np.column_stack(columns)


def _fit_error_db(target: _FloatArray, predicted: _FloatArray) -> float:
    error = np.sum((target - predicted) ** 2)
    if error == 0:
        return float("-inf")
    return float(10.0 * np.log10(error / np.sum(target * target)))


def _regime(frequency: float, amplitude: float, phase: float) -> tuple[_FloatArray, _FloatArray]:
    samples = np.arange(48_000, dtype=np.float64)
    angle = 2.0 * np.pi * frequency * samples / 48_000.0 + phase
    current = amplitude * np.sin(angle)
    previous = amplitude * np.sin(angle - 2.0 * np.pi * frequency / 48_000.0)
    drive, folds = _quadrature_features(current, previous)
    return _blend_and_ac_periodic(drive), _blend_and_ac_periodic(folds)


def test_periodic_filter_matches_its_time_domain_recurrence() -> None:
    samples = np.arange(96, dtype=np.float64)
    signal = np.column_stack((np.sin(2 * np.pi * samples / 12), np.cos(2 * np.pi * samples / 8)))
    output = _blend_and_ac_periodic(signal)
    blended = 0.875 * signal + 0.125 * np.roll(signal, 1, axis=0)
    expected = 0.9993 * (np.roll(output, 1, axis=0) + blended - np.roll(blended, 1, axis=0))
    assert float(np.max(np.abs(output - expected))) < 1e-12


def test_synthetic_fold_fit_exposes_restricted_bank_and_generalization_limits() -> None:
    drive_a, folds_a = _regime(997.0, 1.5, 0.0)
    drive_b, folds_b = _regime(3217.0, 1.2, 0.37)
    target_a = folds_a[:, 0] + 0.35 * folds_a[:, 1]
    target_b = folds_b[:, 0] + 0.35 * folds_b[:, 1]

    coefficients, _, _, _ = np.linalg.lstsq(drive_a, target_a, rcond=1e-12)
    assert -16.0 < _fit_error_db(target_a, drive_a @ coefficients) < -15.4
    assert -12.0 < _fit_error_db(target_b, drive_b @ coefficients) < -11.3

    expanded_a = np.column_stack((drive_a, folds_a))
    expanded_b = np.column_stack((drive_b, folds_b))
    coefficients, _, _, _ = np.linalg.lstsq(expanded_a, target_a, rcond=1e-12)
    assert _fit_error_db(target_a, expanded_a @ coefficients) < -200.0
    assert _fit_error_db(target_b, expanded_b @ coefficients) < -200.0

    tone_a = _tone_bank_periodic(drive_a)
    tone_b = _tone_bank_periodic(drive_b)
    coefficients, _, rank, _ = np.linalg.lstsq(tone_a, target_a, rcond=1e-5)
    assert rank == 23
    assert -27.0 < _fit_error_db(target_a, tone_a @ coefficients) < -25.0
    assert _fit_error_db(target_b, tone_b @ coefficients) > 20.0
