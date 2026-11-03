"""Coherent Stage II alias examples, not device or production-engine tests."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

type _FloatArray = npt.NDArray[np.float64]


def _saturated_odd(values: _FloatArray) -> _FloatArray:
    clamped = np.clip(values, -1.0, 1.0)
    return clamped * (1.5 - 0.5 * clamped * clamped)


def _odd_paths(signal: _FloatArray) -> tuple[_FloatArray, _FloatArray, _FloatArray]:
    previous = np.roll(signal, 1)
    lower = np.minimum(previous, signal)
    upper = np.maximum(previous, signal)
    clamped_lower = np.clip(lower, -1.0, 1.0)
    clamped_upper = np.clip(upper, -1.0, 1.0)
    midpoint = (clamped_lower + clamped_upper) * 0.5
    half_difference = (clamped_upper - clamped_lower) * 0.5
    clamped = midpoint * (1.5 - 0.5 * (midpoint * midpoint + half_difference * half_difference))

    inside = clamped_upper - clamped_lower
    below = np.maximum(0.0, np.minimum(upper, -1.0) - lower)
    above = np.maximum(0.0, upper - np.maximum(lower, 1.0))
    distance = upper - lower
    saturated = np.empty_like(signal)
    np.divide(inside * clamped + above - below, distance, out=saturated, where=distance != 0)
    equal = distance == 0
    saturated[equal] = _saturated_odd(lower[equal])
    return _saturated_odd(signal), clamped, saturated


def _tone_metrics(amplitude: float, oversample: int) -> tuple[float, float, float, float, float]:
    count = 1024
    frequency_bin = 73
    samples = np.arange(count, dtype=np.float64)
    signal = amplitude * np.sin(2.0 * np.pi * frequency_bin * samples / count)
    spectra = [np.fft.rfft(path) for path in _odd_paths(signal)]

    alias_bins = np.ones(count // 2 + 1, dtype=np.bool_)
    alias_bins[0] = alias_bins[-1] = False
    for harmonic in range(1, count // (2 * frequency_bin) + 1, 2):
        alias_bins[harmonic * frequency_bin] = False
    alias_energies = [float(np.sum(np.abs(spectrum[alias_bins]) ** 2)) for spectrum in spectra]
    clamped_alias_db = float(10.0 * np.log10(alias_energies[1] / alias_energies[0]))
    saturated_alias_db = float(10.0 * np.log10(alias_energies[2] / alias_energies[0]))

    high_time = np.arange(count * oversample, dtype=np.float64) / oversample
    high_signal = amplitude * np.sin(2.0 * np.pi * frequency_bin * high_time / count)
    ideal_spectrum = np.fft.rfft(_saturated_odd(high_signal))[: count // 2 + 1] / oversample
    ideal_spectrum *= np.exp(-1j * np.pi * np.arange(count // 2 + 1) / count)
    ideal_energy = float(np.sum(np.abs(ideal_spectrum[1:-1]) ** 2))
    ideal_alias_fraction = float(np.sum(np.abs(ideal_spectrum[alias_bins]) ** 2) / ideal_energy)
    errors = [
        float(
            10.0
            * np.log10(np.sum(np.abs(spectrum[1:-1] - ideal_spectrum[1:-1]) ** 2) / ideal_energy)
        )
        for spectrum in spectra[1:]
    ]
    return clamped_alias_db, saturated_alias_db, errors[0], errors[1], ideal_alias_fraction


@pytest.mark.parametrize(
    ("amplitude", "clamped_alias", "saturated_alias", "clamped_error", "saturated_error"),
    [
        (2.0, 0.826, -12.822, -26.785, -29.420),
        (4.0, -15.428, -8.687, -19.452, -24.295),
    ],
)
def test_saturated_adaa_has_no_universal_alias_ordering(
    amplitude: float,
    clamped_alias: float,
    saturated_alias: float,
    clamped_error: float,
    saturated_error: float,
) -> None:
    metrics = _tone_metrics(amplitude, 256)
    coarse = _tone_metrics(amplitude, 128)
    assert metrics[:4] == pytest.approx(
        (clamped_alias, saturated_alias, clamped_error, saturated_error), abs=0.01
    )
    assert metrics[:4] == pytest.approx(coarse[:4], abs=0.001)
    assert max(metrics[4], coarse[4]) < 1e-15
