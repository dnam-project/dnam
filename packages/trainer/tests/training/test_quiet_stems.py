"""Unit tests for quiet tonal stem vs silent stem handling (Design §7)."""

from __future__ import annotations

import numpy as np

from dnam_trainer.training.metrics import (
    compute_floor_regularized_besr_db,
    evaluate_psychoacoustic_metrics,
    is_band_goal_satisfied,
    measure_stem_band_noise_floor,
)


def test_quiet_tonal_stem_is_classified_active_while_silent_stem_is_silent() -> None:
    sr = 48_000
    n = 48_000
    rng = np.random.default_rng(2026)

    # Dry input has silent frames in the first 8,000 samples, then active guitar burst
    x = np.zeros(n, dtype=np.float32)
    t_active = np.arange(n - 8_000, dtype=np.float32) / float(sr)
    x[8_000:] = 0.25 * np.sin(2.0 * np.pi * 220.0 * t_active)

    # Analog noise floor at ~ -91.2 dBFS (linear std ~ 2.75e-5)
    noise_std = 10.0 ** (-91.2 / 20.0)
    noise = (noise_std * rng.standard_normal((n, 2))).astype(np.float32)

    # 1. Quiet tonal stem (e.g., peggy_fliptop volume_min at -64.4 dBFS over -91.2 dBFS floor)
    quiet_amp = (10.0 ** (-64.4 / 20.0)) * np.sqrt(2.0)
    y_quiet = noise.copy()
    tone = (
        quiet_amp
        * np.sin(2.0 * np.pi * 220.0 * t_active + 0.3 * np.sin(2.0 * np.pi * 440.0 * t_active))
    ).astype(np.float32)
    y_quiet[8_000:, 0] += tone
    y_quiet[8_000:, 1] += tone

    prof_quiet = measure_stem_band_noise_floor(
        x, y_quiet, sample_rate_hz=sr, min_silent_seconds=0.1
    )
    assert prof_quiet.source == "measured_silent_frames"
    assert prof_quiet.noise_floor_dbfs_broadband < -85.0
    assert prof_quiet.body_rms_dbfs > prof_quiet.noise_floor_dbfs_broadband + 6.0
    assert prof_quiet.is_silent is False

    # 2. Genuinely silent stem (~ -91.0 dBFS pure noise floor)
    y_silent = noise.copy()
    prof_silent = measure_stem_band_noise_floor(
        x, y_silent, sample_rate_hz=sr, min_silent_seconds=0.1
    )
    assert prof_silent.is_silent is True

    # 3. Verify compute_floor_regularized_besr_db and evaluate_psychoacoustic_metrics on quiet stem
    # Accurate model prediction on the quiet tonal stem should achieve < -25 dB bESR
    pred_good = np.zeros_like(y_quiet)
    pred_good[8_000:, 0] = tone
    pred_good[8_000:, 1] = tone

    besr_quiet = compute_floor_regularized_besr_db(
        pred_good,
        y_quiet,
        sample_rate_hz=sr,
        band_floors_dbfs=prof_quiet.band_floors_dbfs,
    )
    assert besr_quiet < -20.0

    rep_quiet = evaluate_psychoacoustic_metrics(
        pred_good,
        y_quiet,
        sample_rate_hz=sr,
        band_floors_dbfs=prof_quiet.band_floors_dbfs,
        is_silent_stem=prof_quiet.is_silent,
    )
    assert rep_quiet.is_silent_stem is False
    assert rep_quiet.besr_db < -20.0

    # Zero prediction on the quiet tonal stem should fail (< -20 dB not met)
    pred_zero = np.zeros_like(y_quiet)
    besr_zero = compute_floor_regularized_besr_db(
        pred_zero,
        y_quiet,
        sample_rate_hz=sr,
        band_floors_dbfs=prof_quiet.band_floors_dbfs,
    )
    assert besr_zero > -10.0


def test_noise_floor_fallbacks_without_silent_gaps() -> None:
    sr = 48_000
    n = 24_000
    t = np.arange(n, dtype=np.float32) / float(sr)
    # Continuous dry input with no silent frames
    x = (0.3 * np.sin(2.0 * np.pi * 110.0 * t)).astype(np.float32)
    y = np.stack([x * 0.5, x * 0.5], axis=1)

    prof_req = measure_stem_band_noise_floor(
        x, y, sample_rate_hz=sr, request_noise_floor_dbfs=-88.0, min_silent_seconds=0.5
    )
    assert prof_req.source == "request_noise_floor"
    assert abs(prof_req.noise_floor_dbfs_broadband - (-88.0)) < 1e-5
    assert prof_req.is_silent is False

    prof_unfloored = measure_stem_band_noise_floor(
        x, y, sample_rate_hz=sr, request_noise_floor_dbfs=None, min_silent_seconds=0.5
    )
    assert prof_unfloored.source == "unfloored"
    assert prof_unfloored.noise_floor_dbfs_broadband <= -119.0
    assert prof_unfloored.is_silent is False


def test_is_band_goal_satisfied_noise_limited_rule() -> None:
    # Case 1: High-SNR band (target -20 dBFS, floor -90 dBFS -> SNR = 70 dB)
    # Goal is -31 dB: -32 dB passes, -25 dB fails
    assert is_band_goal_satisfied(
        -32.0,
        -31.0,
        residual_band_dbfs=-52.0,
        noise_floor_band_dbfs=-90.0,
        target_band_dbfs=-20.0,
    )
    assert not is_band_goal_satisfied(
        -25.0,
        -31.0,
        residual_band_dbfs=-45.0,
        noise_floor_band_dbfs=-90.0,
        target_band_dbfs=-20.0,
    )

    # Case 2: Noise-limited band on a quiet stem (target -68 dBFS, floor -91 dBFS -> SNR = 23 dB)
    # Residual is -90.5 dBFS (within 1.0 dB of the -91.0 dBFS floor) -> passes!
    assert is_band_goal_satisfied(
        -22.5,
        -31.0,
        residual_band_dbfs=-90.5,
        noise_floor_band_dbfs=-91.0,
        target_band_dbfs=-68.0,
    )
    # Residual is -85.0 dBFS (6 dB above the -91.0 dBFS floor) -> fails!
    assert not is_band_goal_satisfied(
        -17.0,
        -31.0,
        residual_band_dbfs=-85.0,
        noise_floor_band_dbfs=-91.0,
        target_band_dbfs=-68.0,
    )

    # Edge cases in metrics.py: Tensor inputs, all-silent x, short < 2 frames, 1-ch vs 2-ch,
    # < 8 samples, silent MR-STFT, LTAS empty bands, evaluate_psychoacoustic_metrics(is_silent=True)
    import torch

    from dnam_trainer.training.metrics import (
        compute_band_split_esr_db,
        compute_mr_stft_loss_db,
        compute_third_octave_ltas_tilt_db,
    )

    p_tensor = measure_stem_band_noise_floor(
        torch.zeros(4096, dtype=torch.float32),
        torch.zeros(4096, dtype=torch.float32) + 1e-5,
        sample_rate_hz=48_000,
        min_silent_seconds=0.04,
    )
    assert p_tensor.is_silent is True
    p_short = measure_stem_band_noise_floor(
        torch.ones(16, dtype=torch.float32),
        torch.ones(16, dtype=torch.float32),
        sample_rate_hz=48_000,
    )
    assert p_short.source == "unfloored"
    assert compute_band_split_esr_db(np.ones((64, 1)), np.ones((64, 2)))[0] <= -60.0
    assert compute_band_split_esr_db(np.ones((64, 2)), np.ones((64, 1)))[0] <= -60.0
    assert compute_floor_regularized_besr_db(np.zeros(4), np.ones(4)) > -5.0
    assert compute_mr_stft_loss_db(np.zeros(256), np.zeros(256)) == 0.0
    assert compute_third_octave_ltas_tilt_db(np.zeros(512), np.zeros(512), min_band_dbfs=10.0) == (
        0.0,
        0.0,
        0.0,
    )
    rep_s1 = evaluate_psychoacoustic_metrics(np.zeros((64, 1)), np.zeros((64, 2)), is_silent=True)
    assert rep_s1.is_silent_stem is True
    rep_s2 = evaluate_psychoacoustic_metrics(np.ones((64, 2)), np.ones((64, 1)), is_silent=False)
    assert rep_s2.is_silent_stem is False
