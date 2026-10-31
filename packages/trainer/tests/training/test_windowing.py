"""Unit and property tests for `dnam_trainer.training.windowing`."""

from __future__ import annotations

import numpy as np
import torch

from dnam_trainer.training.windowing import (
    CausalBsplineWindowSchedule,
    build_orthogonal_fft_windows,
    build_stage_window_grids,
    build_uniform_windows,
    compute_cumulative_energy_state,
    compute_segmentation_window_stats,
    merge_segmentation_stats,
    segment_regimes_dp,
)


def test_causal_bspline_window_schedule_partition_of_unity() -> None:
    x = torch.linspace(-1.0, 1.0, 4096, dtype=torch.float32)
    e_cum = compute_cumulative_energy_state(x, sr=48_000.0)
    win_bounds = build_uniform_windows(4096, win_len=1024)
    assert len(win_bounds) == 4

    for ov in (0, 128):
        sched = CausalBsplineWindowSchedule(e_cum, win_bounds, overlap=ov)
        dense = sched.to_dense()
        assert dense.shape == (4096, 4)
        row_sums = torch.sum(dense, dim=1)
        np.testing.assert_allclose(row_sums.numpy(), np.ones(4096, dtype=np.float32), atol=1e-5)

        # Verify window_weight on [500, 1500) matches dense slice
        w1_slice = sched.window_weight(1, 500, 1500)
        np.testing.assert_allclose(
            w1_slice.squeeze(-1).numpy(), dense[500:1500, 1].numpy(), atol=1e-6
        )


def test_window_grid_builders_for_short_and_long_lengths() -> None:
    for l_tot in (2048, 32_768, 240_000, 1_440_000):
        wb, wb48 = build_stage_window_grids(l_tot)
        assert wb[0][0] == 0
        assert wb[-1][1] == l_tot
        assert wb48[0][0] == 0
        assert wb48[-1][1] == l_tot

        w16, w24, w32, w48 = build_orthogonal_fft_windows(l_tot)
        for grid in (w16, w24, w32, w48):
            assert grid[0][0] == 0
            assert grid[-1][1] == l_tot


def test_segment_regimes_dp_recovers_gain_change_and_merges_stats() -> None:
    rng = np.random.default_rng(1234)
    l_tot = 96_000  # 4 windows of 24,000
    x_np = (0.25 * rng.standard_normal(l_tot)).astype(np.float32)
    y1 = np.zeros(l_tot, dtype=np.float32)
    y2 = np.zeros(l_tot, dtype=np.float32)
    # First half: linear gain 0.5; second half: heavy tanh saturation gain 4.0
    y1[:48_000] = 0.5 * x_np[:48_000]
    y1[48_000:] = np.tanh(4.0 * x_np[48_000:])
    y2[:48_000] = 0.8 * x_np[:48_000]
    y2[48_000:] = np.tanh(3.0 * x_np[48_000:])

    x_t = torch.from_numpy(x_np)
    y_t = torch.from_numpy(np.column_stack([y1, y2]))
    win_bounds = build_uniform_windows(l_tot, win_len=24_000)

    stats_full = compute_segmentation_window_stats(x_t, y_t, win_bounds, sr=48_000.0)
    regimes = segment_regimes_dp(stats_full, min_regimes=2, max_regimes=4)
    assert regimes[0][0] == 0
    assert regimes[-1][1] == l_tot
    assert len(regimes) >= 2
    # Breakpoint at 48,000 should be selected
    boundaries = {r[0] for r in regimes} | {r[1] for r in regimes}
    assert 48_000 in boundaries

    # Test merge_segmentation_stats incremental stem addition
    stats_1 = compute_segmentation_window_stats(x_t, y_t[:, :1], win_bounds, sr=48_000.0)
    stats_2 = compute_segmentation_window_stats(
        x_t,
        y_t[:, 1:],
        win_bounds,
        sr=48_000.0,
        cached_win_grams_f64=stats_1.win_grams_f64,
    )
    merged = merge_segmentation_stats(stats_1, stats_2)
    np.testing.assert_allclose(merged.win_rhs_f64, stats_full.win_rhs_f64, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(merged.win_y2_f64, stats_full.win_y2_f64, rtol=1e-5, atol=1e-5)

    # Edge cases: length-1 signals, out-of-range slices, default window_weight, empty bounds
    import pytest

    from dnam_trainer.training.windowing import (
        SegmentationWindowStats,
        build_segmentation_probe_basis,
    )

    assert compute_cumulative_energy_state(torch.ones(1)).shape == (1,)
    assert build_segmentation_probe_basis(torch.ones(1)).shape == (1, 30)
    sched0 = CausalBsplineWindowSchedule(compute_cumulative_energy_state(x_t), win_bounds)
    assert sched0._raw_weight_on_slice(-1, 0, 16).shape == (16,)
    assert sched0._raw_weight_on_slice(0, 72_000, 96_000).shape == (24_000,)
    assert sched0.window_weight(0).shape[1] == 1
    sched_1w = CausalBsplineWindowSchedule(torch.ones(64), [(0, 64)])
    assert sched_1w.window_weight(0).shape == (64, 1)
    assert sched_1w.to_dense().shape == (64, 1)
    empty_st = SegmentationWindowStats(
        win_bounds=(),
        win_grams_f64=np.zeros((0, 4, 4)),
        win_rhs_f64=np.zeros((0, 4, 1)),
        win_y2_f64=np.zeros((0, 1)),
    )
    assert segment_regimes_dp(empty_st) == [(0, 1)]
    with pytest.raises(ValueError, match="mismatched win_bounds"):
        merge_segmentation_stats(stats_1, empty_st)
