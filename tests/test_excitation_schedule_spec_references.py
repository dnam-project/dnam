"""Capture-schedule and ideal multitone examples, not Studio integration tests."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import numpy.typing as npt
import pytest


def _body_offsets() -> tuple[int, int, tuple[int, ...]]:
    sample_rate = 48_000
    header = 26_400
    body = 30 * sample_rate - header - 3 * sample_rate
    return header, body, tuple(body * percent // 100 for percent in (20, 48, 70, 88, 100))


def test_canonical_schedule_includes_the_leading_alignment_header() -> None:
    header, body, ends = _body_offsets()
    assert header == 26_400
    assert body == 1_269_600
    assert ends == (253_920, 609_408, 888_720, 1_117_248, 1_269_600)
    assert header + ends[-1] == 27 * 48_000
    assert header + ends[-1] + 3 * 48_000 == 30 * 48_000
    durations = tuple(end - start for start, end in zip((0, *ends[:-1]), ends, strict=True))
    assert durations == (253_920, 355_488, 279_312, 228_528, 152_352)


def test_sustain_and_transient_intervals_fit_the_documented_timeline() -> None:
    header, body, ends = _body_offsets()
    duration_sustain = (ends[3] - ends[2]) * 48 // 100
    assert duration_sustain == 109_693
    assert Fraction(duration_sustain, 48_000) < Fraction(4)
    assert Fraction(duration_sustain, 48_000 * 4) < 1

    section5 = body - ends[3]
    burst_start = header + ends[3] + section5 * 62 // 100
    burst_end = burst_start + min(48_000 * 80 // 1000, section5 * 68 // 100 - section5 * 62 // 100)
    assert burst_start == 1_238_106
    assert burst_end == 1_241_946
    assert Fraction(30 * 48_000 - burst_end, 48_000) == Fraction(33_009, 8_000)


def test_constant_gain_multitone_has_nonconstant_windowed_energy() -> None:
    _, _, ends = _body_offsets()
    n4 = ends[3] - ends[2]
    sustain_length = n4 * 48 // 100
    time = np.arange(n4, dtype=np.float64) / 48_000
    multitone = np.zeros(n4, dtype=np.float64)
    for k in range(1, 25):
        frequency = 20 * math.exp((k - 1) * math.log(1000) / 23)
        phase = math.pi * k * (k - 1) / 24
        multitone += np.cos(2 * math.pi * frequency * time + phase)
    multitone /= np.max(np.abs(multitone))
    chord = sum(
        weight * np.sin(2 * math.pi * frequency * time + phase)
        for weight, frequency, phase in (
            (0.30, 82.41, 0.0),
            (0.22, 123.47, 0.5),
            (0.18, 164.81, 1.2),
            (0.14, 246.94, 2.0),
            (0.10, 329.63, 2.7),
            (0.06, 800.0, 0.8),
        )
    )
    sustain = 0.55 * (0.65 * chord + 0.35 * multitone)
    samples = sustain[: sustain_length // 4_800 * 4_800].reshape(-1, 4_800)
    energies = np.mean(samples * samples, axis=1)
    ratio = float(np.max(energies) / np.min(energies))
    assert 1.28 < ratio < 1.29


def test_post_adc_gain_cannot_recover_clipped_samples() -> None:
    def adc(value: float) -> float:
        return min(1.0, max(-1.0, value))

    gain = 100.0
    assert adc(2.0) * gain == adc(10.0) * gain
    assert math.isclose(adc(10.0 / gain) * gain, 10.0)
    assert adc(10.0) * gain != 10.0


def test_gain_restoration_scales_adc_noise_and_preserves_signal_to_noise_ratio() -> None:
    signal = np.array([0.3, -0.5, 0.1, -0.2], dtype=np.float64)
    adc_noise = np.array([0.002, 0.001, -0.001, -0.002], dtype=np.float64)
    physical_gain = 0.1
    digital_makeup = 10.0
    raw_signal = physical_gain * signal
    raw_ratio = float(np.sum(raw_signal**2) / np.sum(adc_noise**2))
    restored_ratio = float(
        np.sum((digital_makeup * raw_signal) ** 2) / np.sum((digital_makeup * adc_noise) ** 2)
    )
    assert math.isclose(restored_ratio, raw_ratio, rel_tol=1e-15)
    np.testing.assert_allclose(digital_makeup * raw_signal, signal, rtol=1e-15, atol=0)
    np.testing.assert_allclose(digital_makeup * adc_noise, [0.02, 0.01, -0.01, -0.02])


def test_pad_gain_uncertainty_propagates_through_digital_restoration() -> None:
    assumed_gain = 0.1
    measured_gain_bounds = (0.09, 0.11)
    restored_gain_bounds = tuple(gain / assumed_gain for gain in measured_gain_bounds)
    assert math.isclose(restored_gain_bounds[0], 0.9)
    assert math.isclose(restored_gain_bounds[1], 1.1)
    errors_db = tuple(20 * math.log10(gain) for gain in restored_gain_bounds)
    assert -0.916 < errors_db[0] < -0.915
    assert 0.827 < errors_db[1] < 0.829


def test_pad_cannot_always_satisfy_peak_and_quiet_signal_constraints_together() -> None:
    # This counterexample assumes the ADC noise is zero at the peak sample.
    unpadded_peak_bound = 10.0
    quiet_signal_rms_lower = 0.001
    adc_noise_rms_upper = 0.00001
    minimum_snr_db = 40.0
    adc_peak_limit = 10 ** (-0.2 / 20)
    gain_upper = adc_peak_limit / unpadded_peak_bound
    gain_lower = 10 ** (minimum_snr_db / 20) * adc_noise_rms_upper / quiet_signal_rms_lower
    assert gain_upper < 0.1
    assert math.isclose(gain_lower, 1.0)
    assert gain_lower > gain_upper


def test_fractional_alignment_can_hide_a_clipped_raw_sample() -> None:
    raw = np.zeros(256, dtype=np.float64)
    raw[128] = 1.0
    integer_shifted = np.zeros_like(raw)
    integer_shifted[:-1] = raw[1:]
    spectrum = np.fft.rfft(integer_shifted)
    bins = np.arange(1, spectrum.size - 1, dtype=np.float64)
    spectrum[1:-1] *= np.exp(2j * math.pi * bins * -0.5 / raw.size)
    aligned = np.fft.irfft(spectrum, n=raw.size)
    warning_threshold = 10 ** (-0.05 / 20)
    assert float(np.max(np.abs(raw))) >= warning_threshold
    assert float(np.max(np.abs(aligned))) < 0.65 < warning_threshold


def test_skipping_first_200_ms_can_miss_a_raw_clipped_sample() -> None:
    raw = np.zeros(48_000, dtype=np.float64)
    raw[4_800] = 1.0
    assert float(np.max(np.abs(raw))) == 1.0
    assert float(np.max(np.abs(raw[9_600:]))) == 0.0


def _ideal_calibration_probe() -> npt.NDArray[np.float64]:
    sample_rate = 48_000
    sample_count = 6 * sample_rate
    signal = np.zeros(sample_count, dtype=np.float64)
    lead, header_length = 4_800, 21_600
    pulse_length = 576
    pulse_time = np.arange(pulse_length, dtype=np.float64) / sample_rate
    pulse_rate = math.log(6_000 / 800) / (pulse_length / sample_rate)
    pulse_phase = 2 * math.pi * 800 * np.expm1(pulse_rate * pulse_time) / pulse_rate
    pulse_window = 0.5 * (1 - np.cos(2 * math.pi * pulse_time * sample_rate / pulse_length))
    for offset, gain in (
        (header_length // 8, 0.9),
        (3 * header_length // 10, -0.9),
        (11 * header_length // 20, 0.675),
    ):
        start = lead + offset
        signal[start : start + pulse_length] = gain * pulse_window * np.sin(pulse_phase)

    body_start = lead + header_length
    body_length = sample_count - sample_count // 16 - body_start
    ramp_end = body_length * 35 // 100
    sweep_end = body_length * 70 // 100
    ramp_time = np.arange(ramp_end, dtype=np.float64) / sample_rate
    ramp = 10 ** (-2 + 2 * np.arange(ramp_end, dtype=np.float64) / ramp_end)
    signal[body_start : body_start + ramp_end] = ramp * (
        0.45 * np.sin(2 * math.pi * 40 * ramp_time)
        + 0.35 * np.sin(2 * math.pi * 330 * ramp_time)
        + 0.20 * np.sin(2 * math.pi * 2640 * ramp_time)
    )
    sweep_time = np.arange(sweep_end - ramp_end, dtype=np.float64) / sample_rate
    sweep_rate = math.log(20_000 / 20) / ((sweep_end - ramp_end) / sample_rate)
    sweep_phase = 2 * math.pi * 20 * np.expm1(sweep_rate * sweep_time) / sweep_rate
    carrier = 0.35 * (
        np.sin(2 * math.pi * 220 * sweep_time) + 0.5 * np.sin(2 * math.pi * 880 * sweep_time)
    )
    signal[body_start + ramp_end : body_start + sweep_end] = carrier + 0.35 * np.sin(sweep_phase)

    burst_time = np.arange(720, dtype=np.float64) / sample_rate
    burst_window = 0.5 * (1 - np.cos(2 * math.pi * burst_time * sample_rate / 720))
    signal[body_start + sweep_end : body_start + sweep_end + 720] = (
        0.92
        * burst_window
        * (
            0.5 * np.sin(2 * math.pi * 500 * burst_time)
            + 0.5 * np.sin(2 * math.pi * 1_800 * burst_time)
        )
    )
    return signal


def _ideal_wet_envelope_peak(signal: npt.NDArray[np.float64]) -> tuple[float, float]:
    envelope = np.mean(np.abs(signal).reshape(-1, 240), axis=1)
    fft_length = 2 ** (envelope.size - 1).bit_length()
    spectrum = np.abs(np.fft.rfft(envelope - np.mean(envelope), n=fft_length))
    frequencies = np.fft.rfftfreq(fft_length, 1 / 200)
    eligible = (frequencies >= 0.25) & (frequencies <= 12)
    peak = int(np.argmax(np.where(eligible, spectrum, -np.inf)))
    fraction = float(spectrum[peak] / (np.sum(spectrum[1:]) + 1e-9))
    return float(frequencies[peak]), fraction


def _known_frequency_dry_fit(
    dry: npt.NDArray[np.float64], wet: npt.NDArray[np.float64], frequency: float
) -> npt.NDArray[np.float64]:
    time = np.arange(dry.size, dtype=np.float64) / 48_000
    phase = 2 * math.pi * frequency * time
    columns = np.column_stack((dry, dry * np.cos(phase), dry * np.sin(phase)))
    coefficients, _, rank, _ = np.linalg.lstsq(columns, wet, rcond=None)
    if rank != 3:
        raise ValueError("insufficient dry excitation")
    return coefficients


def test_current_envelope_threshold_misses_an_ideal_steady_gain_lfo() -> None:
    time = np.arange(6 * 48_000, dtype=np.float64) / 48_000
    dry = 0.5 * np.sin(2 * math.pi * 1_000 * time)
    wet = dry * (1 + 0.8 * np.cos(2 * math.pi * 4 * time))
    candidate, fraction = _ideal_wet_envelope_peak(wet)
    assert float(np.max(np.abs(wet))) < 1
    assert abs(candidate - 4.0) < 0.01
    assert 0.15 < fraction < 0.17 < 0.35


def test_current_envelope_threshold_misses_lfo_on_ideal_calibration_probe() -> None:
    dry = _ideal_calibration_probe()
    time = np.arange(dry.size, dtype=np.float64) / 48_000
    for frequency in (0.25, 4.0, 10.0):
        for phase in (0.0, math.pi / 2):
            gain = 0.5 * (1 + 0.8 * np.cos(2 * math.pi * frequency * time + phase))
            wet = dry * gain
            candidate, fraction = _ideal_wet_envelope_peak(wet)
            assert float(np.max(np.abs(wet))) < 1
            assert fraction < 0.08 < 0.35
            if frequency >= 4:
                assert 0.29 < candidate < 0.30


def test_dry_aware_fit_recovers_known_frequency_gain_on_ideal_probe() -> None:
    dry = _ideal_calibration_probe()
    time = np.arange(dry.size, dtype=np.float64) / 48_000
    for frequency in (0.25, 4.0, 10.0):
        for phase in (0.0, math.pi / 2):
            wet = dry * (0.5 + 0.4 * np.cos(2 * math.pi * frequency * time + phase))
            fit = _known_frequency_dry_fit(dry, wet, frequency)
            assert float(np.max(np.abs(wet))) < 1
            np.testing.assert_allclose(
                fit, [0.5, 0.4 * math.cos(phase), -0.4 * math.sin(phase)], atol=1e-12
            )
    with pytest.raises(ValueError, match="insufficient dry excitation"):
        _known_frequency_dry_fit(np.zeros(100), np.zeros(100), 0.25)


def test_dry_aware_fit_can_mistake_static_nonlinearity_for_modulation() -> None:
    dry = _ideal_calibration_probe()
    wet = 0.5 * (dry + 0.5 * dry**3)
    fit = _known_frequency_dry_fit(dry, wet, 0.25)
    apparent_amplitude = math.hypot(float(fit[1]), float(fit[2]))
    assert float(np.max(np.abs(wet))) < 1
    assert 0.566 < fit[0] < 0.567
    assert 0.023 < apparent_amplitude < 0.024


def _polynomial_gain_fit(
    dry: npt.NDArray[np.float64],
    wet: npt.NDArray[np.float64],
    sample_rate: float,
    frequency: float,
) -> tuple[npt.NDArray[np.float64], float, float]:
    time = np.arange(dry.size, dtype=np.float64) / sample_rate
    phase = 2 * math.pi * frequency * time
    columns = np.column_stack((dry, dry**3, dry * np.cos(phase), dry * np.sin(phase)))
    coefficients, _, rank, singular_values = np.linalg.lstsq(columns, wet, rcond=None)
    if rank != 4:
        raise ValueError("insufficient independent excitation")
    residual = wet - columns @ coefficients
    return (
        coefficients,
        float(np.dot(residual, residual)),
        float(singular_values[0] / singular_values[-1]),
    )


def test_cubic_nuisance_fit_removes_the_static_cubic_confound() -> None:
    dry = _ideal_calibration_probe()[::48]
    static_wet = 0.4 * dry + 0.2 * dry**3
    coefficients, residual, condition = _polynomial_gain_fit(dry, static_wet, 1_000, 0.25)
    np.testing.assert_allclose(coefficients, [0.4, 0.2, 0.0, 0.0], atol=1e-12)
    assert residual < 1e-20
    assert condition < 100

    time = np.arange(dry.size, dtype=np.float64) / 1_000
    modulated_wet = static_wet + dry * (
        0.12 * np.cos(2 * math.pi * 0.25 * time) - 0.07 * np.sin(2 * math.pi * 0.25 * time)
    )
    coefficients, residual, _ = _polynomial_gain_fit(dry, modulated_wet, 1_000, 0.25)
    np.testing.assert_allclose(coefficients, [0.4, 0.2, 0.12, -0.07], atol=1e-12)
    assert residual < 1e-20
    with pytest.raises(ValueError, match="independent excitation"):
        _polynomial_gain_fit(np.zeros(100), np.zeros(100), 1_000, 0.25)


def test_cubic_nuisance_fit_still_confounds_an_unmodeled_static_nonlinearity() -> None:
    dry = _ideal_calibration_probe()[::48]
    static_wet = 0.4 * dry + 0.2 * dry**3 + 0.2 * dry**5
    coefficients, residual, _ = _polynomial_gain_fit(dry, static_wet, 1_000, 0.25)
    apparent_amplitude = math.hypot(float(coefficients[2]), float(coefficients[3]))
    assert apparent_amplitude > 1e-5
    assert residual > 1e-7


def test_candidate_frequency_grid_recovers_only_the_tested_ideal_frequencies() -> None:
    dry = _ideal_calibration_probe()[::48]
    time = np.arange(dry.size, dtype=np.float64) / 1_000
    candidates = (0.25, 0.5, 1.0, 2.0, 4.0, 7.0, 10.0)
    for frequency in (0.25, 4.0, 10.0):
        wet = 0.4 * dry + 0.2 * dry**3 + 0.12 * dry * np.cos(2 * math.pi * frequency * time + 0.3)
        results = [
            (candidate, _polynomial_gain_fit(dry, wet, 1_000, candidate))
            for candidate in candidates
        ]
        best_frequency, (coefficients, residual, _) = min(results, key=lambda result: result[1][1])
        assert best_frequency == frequency
        np.testing.assert_allclose(
            coefficients, [0.4, 0.2, 0.12 * math.cos(0.3), -0.12 * math.sin(0.3)], atol=1e-12
        )
        assert residual < 1e-20

    off_grid = 3.25
    wet = 0.4 * dry + 0.2 * dry**3 + 0.12 * dry * np.cos(2 * math.pi * off_grid * time)
    results = [_polynomial_gain_fit(dry, wet, 1_000, frequency)[1] for frequency in candidates]
    assert min(results) > 1e-6


def test_fixed_grid_gaussian_null_has_a_familywise_threshold() -> None:
    dry = _ideal_calibration_probe()[::96]
    time = np.arange(dry.size, dtype=np.float64) / 500
    null_columns = np.column_stack((dry, dry**3))
    null_basis, _ = np.linalg.qr(null_columns)
    alternatives = []
    for frequency in (0.25, 1.0, 4.0, 10.0):
        phase = 2 * math.pi * frequency * time
        columns = np.column_stack((dry * np.cos(phase), dry * np.sin(phase)))
        residual_columns = columns - null_basis @ (null_basis.T @ columns)
        assert np.linalg.matrix_rank(residual_columns) == 2
        basis, _ = np.linalg.qr(residual_columns)
        alternatives.append(basis)

    combined = np.column_stack(alternatives)
    assert np.linalg.matrix_rank(combined) == 2 * len(alternatives)
    span, _ = np.linalg.qr(combined)
    draws = np.random.default_rng(20260601).standard_normal((40_000, span.shape[1]))
    scores = np.column_stack(
        [np.sum((draws @ (span.T @ basis)) ** 2, axis=1) for basis in alternatives]
    )
    alpha = 0.05
    threshold = 2 * math.log(len(alternatives) / alpha)
    assert math.isclose(len(alternatives) * math.exp(-threshold / 2), alpha)
    false_alarm_fraction = float(np.mean(np.max(scores, axis=1) >= threshold))
    assert 0.02 < false_alarm_fraction < 0.06

    noise = span @ draws[0]
    wet = null_columns @ np.array([0.4, 0.2]) + noise
    null_fit = np.linalg.lstsq(null_columns, wet, rcond=None)[0]
    null_residual = wet - null_columns @ null_fit
    _, alternative_residual, _ = _polynomial_gain_fit(dry, wet, 500, 4.0)
    improvement = float(np.dot(null_residual, null_residual)) - alternative_residual
    assert math.isclose(improvement, float(scores[0, 2]), rel_tol=1e-11, abs_tol=1e-11)


def _unknown_variance_ratio_threshold(alpha: float, hypotheses: int, dof: int) -> float:
    if not 0 < alpha < 1 or hypotheses < 1 or dof < 1:
        raise ValueError("invalid false-alarm parameters")
    return math.expm1(2 * (math.log(hypotheses) - math.log(alpha)) / dof)


def test_unknown_variance_needs_an_f_threshold_not_a_plug_in_chi_square() -> None:
    alpha, hypotheses, dof = 0.05, 1, 4
    expected = _unknown_variance_ratio_threshold(alpha, hypotheses, dof)
    naive = 2 * math.log(hypotheses / alpha) / dof

    def survival(ratio: float) -> float:
        return math.exp(-dof * math.log1p(ratio) / 2)

    assert math.isclose(survival(expected), alpha, rel_tol=1e-14)
    assert 0.16 < survival(naive) < 0.161
    assert expected > naive

    dof = 1_000_000_000
    threshold = _unknown_variance_ratio_threshold(alpha, 4, dof)
    assert math.isclose(dof * threshold, 2 * math.log(4 / alpha), rel_tol=1e-8)
    for bad in ((0.0, 1, 4), (1.0, 1, 4), (0.05, 0, 4), (0.05, 1, 0)):
        with pytest.raises(ValueError, match="false-alarm"):
            _unknown_variance_ratio_threshold(*bad)


def test_unknown_variance_fixed_grid_controls_simulated_gaussian_null() -> None:
    rng = np.random.default_rng(20260602)
    dry = rng.uniform(-0.9, 0.9, size=48)
    time = np.arange(dry.size, dtype=np.float64) / 12
    null_columns = np.column_stack((dry, dry**3))
    assert np.linalg.matrix_rank(null_columns) == 2
    null_basis, _ = np.linalg.qr(null_columns)
    alternatives = []
    for frequency in (0.25, 1.0, 2.25, 4.0):
        phase = 2 * math.pi * frequency * time
        columns = np.column_stack((dry * np.cos(phase), dry * np.sin(phase)))
        residual_columns = columns - null_basis @ (null_basis.T @ columns)
        assert np.linalg.matrix_rank(residual_columns) == 2
        basis, _ = np.linalg.qr(residual_columns)
        alternatives.append(basis)

    noise = rng.standard_normal((40_000, dry.size))
    null_residual = noise - (noise @ null_basis) @ null_basis.T
    rss_null = np.sum(null_residual**2, axis=1)
    improvements = np.column_stack(
        [np.sum((null_residual @ basis) ** 2, axis=1) for basis in alternatives]
    )
    rss_alternative = rss_null[:, None] - improvements
    assert np.all(rss_alternative > 0)
    ratios = improvements / rss_alternative
    dof = dry.size - null_columns.shape[1] - 2
    single_threshold = _unknown_variance_ratio_threshold(0.05, 1, dof)
    grid_threshold = _unknown_variance_ratio_threshold(0.05, len(alternatives), dof)
    single_false_alarms = float(np.mean(ratios[:, 0] >= single_threshold))
    grid_false_alarms = float(np.mean(np.max(ratios, axis=1) >= grid_threshold))
    assert 0.045 < single_false_alarms < 0.055
    assert 0.02 < grid_false_alarms < 0.06

    wet = null_columns @ np.array([0.4, 0.2]) + noise[0]
    null_fit = np.linalg.lstsq(null_columns, wet, rcond=None)[0]
    direct_null_residual = wet - null_columns @ null_fit
    _, direct_alternative, _ = _polynomial_gain_fit(dry, wet, 12, 2.25)
    direct_improvement = (
        float(np.dot(direct_null_residual, direct_null_residual)) - direct_alternative
    )
    assert math.isclose(direct_improvement, float(improvements[0, 2]), rel_tol=1e-12)
    assert math.isclose(direct_alternative, float(rss_alternative[0, 2]), rel_tol=1e-12)


def test_known_covariance_whitening_restores_the_conditional_f_reference() -> None:
    count = 64
    correlation = 0.9
    innovation_scale = math.sqrt(1 - correlation**2)
    row, col = np.indices((count, count))
    covariance = correlation ** np.abs(row - col)
    whitener = np.zeros((count, count), dtype=np.float64)
    whitener[0, 0] = 1
    indices = np.arange(1, count)
    whitener[indices, indices] = 1 / innovation_scale
    whitener[indices, indices - 1] = -correlation / innovation_scale
    np.testing.assert_allclose(whitener @ covariance @ whitener.T, np.eye(count), atol=1e-14)

    phase = 2 * math.pi * np.arange(count, dtype=np.float64) / count
    columns = np.column_stack((np.ones(count), np.cos(phase), np.sin(phase)))
    raw_basis, _ = np.linalg.qr(columns)
    whitened_basis, _ = np.linalg.qr(whitener @ columns)
    assert np.linalg.matrix_rank(columns) == 3
    assert np.linalg.matrix_rank(whitener @ columns) == 3

    innovations = np.random.default_rng(20260606).standard_normal((25_000, count))
    colored = innovations.copy()
    for index in range(1, count):
        colored[:, index] = (
            correlation * colored[:, index - 1] + innovation_scale * innovations[:, index]
        )
    whitened = colored @ whitener.T
    np.testing.assert_allclose(whitened, innovations, atol=1e-14)

    def false_alarm_fraction(
        observations: npt.NDArray[np.float64], basis: npt.NDArray[np.float64]
    ) -> float:
        coordinates = observations @ basis
        improvement = np.sum(coordinates[:, 1:] ** 2, axis=1)
        residual = np.sum(observations**2, axis=1) - np.sum(coordinates**2, axis=1)
        assert np.all(residual > 0)
        threshold = _unknown_variance_ratio_threshold(0.05, 1, count - 3)
        return float(np.mean(improvement / residual >= threshold))

    raw_false_alarms = false_alarm_fraction(colored, raw_basis)
    whitened_false_alarms = false_alarm_fraction(whitened, whitened_basis)
    assert raw_false_alarms > 0.2
    assert 0.04 < whitened_false_alarms < 0.06


def test_short_lfo_window_can_have_worse_regression_conditioning() -> None:
    conditions = []
    for n_samples in (457, 800):
        time = np.arange(n_samples, dtype=np.float64) / 200
        phase = 2 * math.pi * 0.25 * time
        columns = np.column_stack((np.ones(n_samples), np.cos(phase), np.sin(phase)))
        conditions.append(float(np.linalg.cond(columns.T @ columns)))
    assert 12.2 < conditions[0] < 12.4
    assert math.isclose(conditions[1], 2.0, rel_tol=1e-12)


def _periodic_multitone_candidate() -> npt.NDArray[np.float32]:
    harmonics = np.arange(1, 501, dtype=np.float64)
    phases = math.pi * harmonics * (harmonics - 1) / 500
    time = np.arange(1200, dtype=np.float64) / 1200
    block = np.cos(2 * math.pi * time[:, None] * harmonics + phases).sum(axis=1)
    return np.asarray(0.98 * block / np.max(np.abs(block)), dtype=np.float32)


def test_periodic_candidate_has_no_sub_40_hz_input_energy_modulation() -> None:
    block = _periodic_multitone_candidate()
    spectrum = np.fft.rfft(block.astype(np.float64))
    magnitudes = np.abs(spectrum[1:501])
    assert 20_000 / 500 == 40
    assert np.min(magnitudes) / np.max(magnitudes) > 0.999_999
    assert float(np.max(np.abs(block))) <= 0.980_001
    tone_peak = float(2 * np.max(magnitudes) / block.size)
    assert tone_peak <= 0.98 * math.sqrt(2 / 500) + 1e-6

    signal = np.tile(block, 160).astype(np.float64)
    power = signal * signal
    power_spectrum = np.abs(np.fft.rfft(power - np.mean(power)))
    assert np.max(power_spectrum[1:41]) < 1e-10 * np.max(power_spectrum)
    window_power = np.mean(power.reshape(-1, 1200), axis=1)
    assert np.max(window_power) == np.min(window_power)


def test_periodic_candidate_distinguishes_a_simple_low_frequency_gain_lfo() -> None:
    signal = np.tile(_periodic_multitone_candidate(), 160).astype(np.float64)
    time = np.arange(signal.size, dtype=np.float64) / 48_000
    for frequency in (0.25, 10.0):
        output = signal * (1.0 + 0.2 * np.cos(2 * math.pi * frequency * time + 0.3))
        window_power = np.mean((output * output).reshape(-1, 1200), axis=1)
        assert float(np.max(window_power) / np.min(window_power)) > 1.5


def test_ballistic_follower_warmup_for_periodic_candidate_is_conditional() -> None:
    signal = np.tile(_periodic_multitone_candidate(), 40)
    attack = math.exp(-1 / (48_000 * 0.010))
    release = math.exp(-1 / (48_000 * 0.150))
    first, second = 0.0, 1.0
    for sample in signal:
        detector = abs(float(sample))
        first_alpha = attack if detector > first else release
        second_alpha = attack if detector > second else release
        first = first_alpha * first + (1 - first_alpha) * detector
        second = second_alpha * second + (1 - second_alpha) * detector
    assert abs(first - second) <= math.exp(-1 / 0.150) + 1e-12


def test_periodic_input_alone_cannot_distinguish_an_autonomous_and_locked_oscillator() -> None:
    block = _periodic_multitone_candidate().astype(np.float64)

    def outputs(gap: bool) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        autonomous, input_triggered = [], []
        input_count = -1
        for period in range(17):
            present = not (gap and period == 8)
            if present:
                input_count += 1
            samples = block if present else np.zeros_like(block)
            autonomous.append(samples * (1 + 0.2 * math.cos(2 * math.pi * period / 4)))
            input_triggered.append(samples * (1 + 0.2 * math.cos(2 * math.pi * input_count / 4)))
        return np.concatenate(autonomous), np.concatenate(input_triggered)

    continuous = outputs(gap=False)
    assert np.array_equal(*continuous)
    with_gap = outputs(gap=True)
    assert np.array_equal(with_gap[0][: 9 * block.size], with_gap[1][: 9 * block.size])
    assert (
        float(np.max(np.abs(with_gap[0][9 * block.size :] - with_gap[1][9 * block.size :]))) > 0.19
    )
