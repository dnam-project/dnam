"""Unit tests for `dnam_trainer.capture.alignment` (2-stage latency alignment, noise floor,
hot-corner telemetry)."""

from __future__ import annotations

import numpy as np

from dnam_trainer.capture.alignment import (
    align_hardware_latency,
    analyze_hot_corner_nonlinearity,
    measure_silent_window_noise_floor,
)
from dnam_trainer.capture.excitation import generate_dry_excitation


def test_align_hardware_latency_mono_stereo_and_inverted_polarity() -> None:
    dry = generate_dry_excitation(duration_sec=1.6, sample_rate_hz=48_000, seed=11)
    true_lag = 137

    # Delayed and polarity-inverted mono wet signal
    wet_mono = np.zeros_like(dry)
    wet_mono[true_lag:] = -0.85 * dry[:-true_lag]

    res_mono = align_hardware_latency(dry, wet_mono, sample_rate_hz=48_000)
    assert res_mono.latency_offset_samples == true_lag
    assert res_mono.peak_correlation > 0.95

    # Stereo wet with trim_alignment_header=True
    wet_st = np.stack([wet_mono, 0.9 * wet_mono], axis=0)
    res_st = align_hardware_latency(
        dry,
        wet_st,
        sample_rate_hz=48_000,
        trim_alignment_header=True,
    )
    assert res_st.latency_offset_samples == true_lag
    assert res_st.aligned_audio.ndim == 2
    assert res_st.aligned_audio.shape[1] < dry.shape[0]


def test_silent_window_noise_floor_and_hot_corner_analysis() -> None:
    dry = generate_dry_excitation(duration_sec=1.6, sample_rate_hz=48_000, seed=19)
    # Exact silence tail -> -120.0 dBFS
    assert measure_silent_window_noise_floor(dry, sample_rate_hz=48_000) == -120.0

    rng = np.random.default_rng(99)
    noisy = (dry + rng.standard_normal(dry.shape).astype(np.float32) * 1e-3).astype(np.float32)
    nf = measure_silent_window_noise_floor(noisy, sample_rate_hz=48_000)
    assert -65.0 <= nf <= -55.0

    # Hot-corner hard clipping at +/- 1.0
    hot_wet = np.clip(dry * 4.0, -1.0, 1.0).astype(np.float32)
    stats = analyze_hot_corner_nonlinearity(dry, hot_wet, unity_ref_dbfs=0.0)
    assert stats.measured_return_peak_dbfs > -0.2
    assert stats.net_gain_over_unity_db > 3.0
    assert stats.compression_ratio > 1.2
    assert stats.clipped_sample_fraction > 0.01

    # Negative offset_samples and silent ch0 stereo branch coverage
    from dnam_trainer.capture.alignment import _shift_and_match_length, _to_mono_reference

    s_neg_1d = _shift_and_match_length(np.ones(64, dtype=np.float32), -8, 64)
    assert float(s_neg_1d[0]) == 0.0
    assert float(s_neg_1d[8]) == 1.0
    s_neg_2d = _shift_and_match_length(np.ones((2, 64), dtype=np.float32), -8, 64)
    assert float(s_neg_2d[0, 0]) == 0.0
    assert float(s_neg_2d[1, 8]) == 1.0
    ch1_only = np.stack([np.zeros(32, dtype=np.float32), np.ones(32, dtype=np.float32)], axis=0)
    assert float(_to_mono_reference(ch1_only)[0]) == 1.0

    # Delay pedal 100% wet (kill-dry) stem: fixed_latency_samples strips only hw_lag (96)
    # and preserves the pedal's 100 ms (4,800-sample) echo repeat offset!
    hw_lag = 96
    echo_tap = 4_800
    wet_delay = np.zeros_like(dry)
    total_shift = hw_lag + echo_tap
    wet_delay[total_shift:] = 0.9 * dry[:-total_shift]

    res_locked = align_hardware_latency(
        dry,
        wet_delay,
        sample_rate_hz=48_000,
        fixed_latency_samples=hw_lag,
    )
    assert res_locked.latency_offset_samples == hw_lag
    # After stripping hw_lag, the echo repeat remains at exact offset echo_tap
    assert np.allclose(
        res_locked.aligned_audio[echo_tap:-hw_lag],
        0.9 * dry[:-total_shift],
        atol=1e-6,
    )

    # Ringing delay tail in the first half of the trailing window + baseline noise floor bound
    ringing_tail = dry.copy()
    tail_start = dry.shape[0] - 24_000
    ringing_tail[tail_start : tail_start + 10_000] = 0.25
    ringing_tail[tail_start + 10_000 :] = 1e-4
    nf_sliding = measure_silent_window_noise_floor(
        ringing_tail,
        sample_rate_hz=48_000,
        baseline_noise_floor_dbfs=-90.0,
    )
    assert nf_sliding <= -75.0
