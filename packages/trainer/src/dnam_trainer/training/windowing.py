"""Generic window grids, on-demand per-window $C^0$ B-spline partition-of-unity schedules,
and fit-driven dynamic-programming (`DP`) regime segmentation.

Replaces all hard-coded $L \\ge 500\\text{k}$ sample-index window/regime tables and eliminates
dense `(L, n_wins)` B-spline weight allocations (saving up to `2.07 GB` at 30 s).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from dnam_trainer.training.numerics import ExactGramAccumulator


def build_uniform_windows(
    num_samples: int,
    win_len: int = 24_000,
    *,
    min_last_len: int | None = None,
) -> list[tuple[int, int]]:
    """Tiles `[0, num_samples)` into contiguous windows of length `win_len`.

    If the trailing remainder is shorter than `min_last_len` (default `win_len // 3`), it is
    merged into the final window so no window is pathologically short or unboundedly long.
    """
    l_tot = max(1, num_samples)
    step = max(1, win_len)
    if l_tot <= step:
        return [(0, l_tot)]

    min_rem = step // 3 if min_last_len is None else max(1, min_last_len)
    bounds: list[tuple[int, int]] = []
    pos = 0
    while pos < l_tot:
        nxt = min(l_tot, pos + step)
        if l_tot - nxt < min_rem:
            nxt = l_tot
        bounds.append((pos, nxt))
        pos = nxt
    return bounds


def build_stage_window_grids(
    num_samples: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Builds `(win_bounds_24k, win_bounds_12k)` for the 3-stage selective cascade for any `L`."""
    l_tot = max(1, num_samples)
    if l_tot >= 48_000:
        win_24k = build_uniform_windows(l_tot, 24_000)
        win_12k = build_uniform_windows(l_tot, 12_000)
    else:
        w1 = max(2048, l_tot // 8)
        w2 = max(1536, l_tot // 12)
        win_24k = build_uniform_windows(l_tot, w1)
        win_12k = build_uniform_windows(l_tot, w2)
    return win_24k, win_12k


def build_orthogonal_fft_windows(
    num_samples: int,
) -> tuple[
    list[tuple[int, int]],
    list[tuple[int, int]],
    list[tuple[int, int]],
    list[tuple[int, int]],
]:
    """Builds `(win_bounds_s1, win_bounds_r1, win_bounds_r2, win_bounds_r3)` for the orthogonal-FFT
    cascade for any length `num_samples` (`12k / 8k / 6k / 4k` grids after `36k` lead-in).
    """
    l_tot = max(1, num_samples)
    if l_tot >= 48_000:
        lead = min(36_000, l_tot // 4)

        def _grid_after_lead(step: int) -> list[tuple[int, int]]:
            tail_wins = build_uniform_windows(l_tot - lead, step)
            return [(0, lead), *[(lead + a, lead + b) for a, b in tail_wins]]

        return (
            _grid_after_lead(12_000),
            _grid_after_lead(8_000),
            _grid_after_lead(6_000),
            _grid_after_lead(4_000),
        )

    return (
        build_uniform_windows(l_tot, max(4096, l_tot // 8)),
        build_uniform_windows(l_tot, max(2048, l_tot // 12)),
        build_uniform_windows(l_tot, max(2048, l_tot // 16)),
        build_uniform_windows(l_tot, max(1536, l_tot // 24)),
    )


def compute_cumulative_energy_state(
    x: Tensor,
    *,
    sr: float = 48_000.0,
) -> Tensor:
    """Computes the 1-D causal 1-pole cumulative energy state
    `e_cum[n] = LPF_{0.008 Hz}(1 + |x[n]|)` on `x.device` in `float32`.
    """
    x_1d = x.reshape(-1).to(dtype=torch.float32)
    l_seq = x_1d.shape[0]
    if l_seq <= 1:
        return 1.0 + torch.abs(x_1d)

    drv = 1.0 + torch.abs(x_1d)
    r = float(math.exp(-2.0 * math.pi * 0.008 / float(sr)))
    k_len = min(l_seq, max(64, math.ceil(-16.0 / math.log(max(r, 1e-6)))))
    t_idx = torch.arange(k_len, device=x_1d.device, dtype=torch.float32)
    ir = (1.0 - r) * torch.pow(r, t_idx)
    n_fft = 1 << math.ceil(math.log2(l_seq + k_len))
    h_spec = torch.fft.rfft(ir, n=n_fft)
    x_spec = torch.fft.rfft(drv, n=n_fft)
    return torch.fft.irfft(x_spec * h_spec, n=n_fft)[:l_seq]


class CausalBsplineWindowSchedule:
    """On-demand per-window $C^0$ B-spline trapezoidal partition-of-unity weight generator.

    Stores only the 1-D `e_cum` vector (`(L,)`, `5.76 MB` at 30 s) and evaluates normalized
    window weights over `[s_w, e_w]` on demand without allocating a dense `(L, n_wins)` tensor.
    """

    def __init__(
        self,
        e_cum: Tensor,
        win_bounds: Sequence[tuple[int, int]],
        *,
        overlap: int = 256,
    ) -> None:
        self.e_cum = e_cum.reshape(-1).to(dtype=torch.float32)
        self.l_tot = self.e_cum.shape[0]
        self.win_bounds: list[tuple[int, int]] = [(k0, k1) for k0, k1 in win_bounds]
        self.n_wins = len(self.win_bounds)
        self.overlap = max(0, overlap)

    @classmethod
    def from_signal(
        cls,
        x: Tensor,
        win_bounds: Sequence[tuple[int, int]],
        *,
        overlap: int = 256,
        sr: float = 48_000.0,
    ) -> CausalBsplineWindowSchedule:
        """Constructs a [`CausalBsplineWindowSchedule`] from input signal `x`."""
        e_cum = compute_cumulative_energy_state(x, sr=sr)
        return cls(e_cum, win_bounds, overlap=overlap)

    def support_bounds(self, w_idx: int) -> tuple[int, int]:
        """Returns the overlapped support interval `(s_w, e_w)` for window `w_idx`."""
        k0, k1 = self.win_bounds[w_idx]
        s_w = 0 if w_idx == 0 else max(0, k0 - self.overlap)
        e_w = self.l_tot if w_idx == self.n_wins - 1 else min(self.l_tot, k1 + self.overlap)
        return s_w, e_w

    def _raw_weight_on_slice(self, w_idx: int, s0: int, s1: int) -> Tensor:
        """Evaluates the unnormalized trapezoidal ramp for window `w_idx` on `[s0, s1)`."""
        if w_idx < 0 or w_idx >= self.n_wins or s1 <= s0:
            return torch.zeros(max(0, s1 - s0), device=self.e_cum.device, dtype=torch.float32)
        k0, k1 = self.win_bounds[w_idx]
        ov = self.overlap
        sw_min = 0 if w_idx == 0 else max(0, k0 - ov)
        ew_max = self.l_tot if w_idx == self.n_wins - 1 else min(self.l_tot, k1 + ov)
        a = max(s0, sw_min)
        b = min(s1, ew_max)
        out = torch.zeros(s1 - s0, device=self.e_cum.device, dtype=torch.float32)
        if b <= a:
            return out

        e_sub = self.e_cum[a:b]
        if w_idx == 0 or ov == 0:
            left_ramp = torch.ones_like(e_sub)
        else:
            ka = self.e_cum[max(0, k0 - ov)]
            kb = self.e_cum[min(self.l_tot - 1, k0 + ov)]
            left_ramp = torch.clamp((e_sub - ka) / (kb - ka + 1.0e-12), 0.0, 1.0)

        if w_idx == self.n_wins - 1 or ov == 0:
            right_ramp = torch.ones_like(e_sub)
        else:
            kc = self.e_cum[max(0, k1 - ov)]
            kd = self.e_cum[min(self.l_tot - 1, k1 + ov)]
            right_ramp = torch.clamp((kd - e_sub) / (kd - kc + 1.0e-12), 0.0, 1.0)

        out[a - s0 : b - s0] = torch.minimum(left_ramp, right_ramp)
        return out

    def window_weight(
        self,
        w_idx: int,
        s0: int | None = None,
        s1: int | None = None,
    ) -> Tensor:
        """Evaluates the partition-of-unity normalized weight column `(s1 - s0, 1)` for window
        `w_idx` on `[s0, s1)` (defaults to `support_bounds(w_idx)`).
        """
        if s0 is None or s1 is None:
            s0, s1 = self.support_bounds(w_idx)
        if self.n_wins <= 1:
            return torch.ones((s1 - s0, 1), device=self.e_cum.device, dtype=torch.float32)

        raw_w = self._raw_weight_on_slice(w_idx, s0, s1)
        denom = raw_w.clone()
        # Sum over any overlapping neighbor windows on [s0, s1)
        for nb in range(max(0, w_idx - 2), min(self.n_wins, w_idx + 3)):
            if nb == w_idx:
                continue
            nb_s, nb_e = self.support_bounds(nb)
            if nb_e > s0 and nb_s < s1:
                denom = denom + self._raw_weight_on_slice(nb, s0, s1)

        norm_w = raw_w / denom.clamp_min(1.0e-12)
        return norm_w.unsqueeze(-1)

    def to_dense(self) -> Tensor:
        """Materializes the `(L, n_wins)` matrix (used only for unit-test verification on short
        signals).
        """
        if self.n_wins <= 1:
            return torch.ones((self.l_tot, 1), device=self.e_cum.device, dtype=torch.float32)
        cols = [self.window_weight(w, 0, self.l_tot) for w in range(self.n_wins)]
        return torch.cat(cols, dim=-1)


@dataclass(frozen=True)
class SegmentationWindowStats:
    """Per-window sufficient statistics on the 24k window grid for fit-driven DP regime
    segmentation (cached in Arena 4).
    """

    win_bounds: tuple[tuple[int, int], ...]
    win_grams_f64: NDArray[np.float64]  # (W, P, P)
    win_rhs_f64: NDArray[np.float64]  # (W, P, M)
    win_y2_f64: NDArray[np.float64]  # (W, M)


def build_segmentation_probe_basis(
    x: Tensor,
    *,
    sr: float = 48_000.0,
) -> Tensor:
    """Builds a compact, deterministic 32-column multi-band + envelope + PolyADAA probe basis
    from dry input `x` on `x.device` in `float32` for regime segmentation.
    """
    x_1d = x.reshape(-1).to(dtype=torch.float32)
    l_seq = x_1d.shape[0]
    dev = x_1d.device

    def _lpf_1d(sig: Tensor, fc: float) -> Tensor:
        if l_seq <= 1:
            return sig.clone()
        r = float(math.exp(-2.0 * math.pi * float(fc) / float(sr)))
        k_len = min(l_seq, max(64, math.ceil(-16.0 / math.log(max(r, 1e-6)))))
        t_idx = torch.arange(k_len, device=dev, dtype=torch.float32)
        ir = (1.0 - r) * torch.pow(r, t_idx)
        n_fft = 1 << math.ceil(math.log2(l_seq + k_len))
        return torch.fft.irfft(
            torch.fft.rfft(sig, n=n_fft) * torch.fft.rfft(ir, n=n_fft),
            n=n_fft,
        )[:l_seq]

    l_120 = _lpf_1d(x_1d, 120.0)
    l_600 = _lpf_1d(x_1d, 600.0)
    l_2400 = _lpf_1d(x_1d, 2400.0)
    b_low = l_120
    b_lm = l_600 - l_120
    b_hm = l_2400 - l_600
    b_hi = x_1d - l_2400

    env_fast = _lpf_1d(torch.abs(x_1d), 35.0)
    env_slow = _lpf_1d(torch.abs(x_1d), 4.0)

    cols: list[Tensor] = [
        x_1d,
        b_low,
        b_lm,
        b_hm,
        b_hi,
        x_1d * env_fast,
        x_1d * env_slow,
        b_low * env_slow,
        b_hm * env_fast,
        b_hi * env_fast,
    ]
    for drv in (0.8, 2.5, 8.0, 24.0):
        for band in (x_1d, b_low, b_lm, b_hm):
            u = torch.clamp((2.0 / 3.0) * drv * band, -1.0, 1.0)
            u2 = u * u
            cols.append(u * (1.5 - 0.5 * u2))
        u_ev = torch.clamp((2.0 / 3.0) * drv * x_1d, -1.0, 1.0)
        u_ev2 = u_ev * u_ev
        even = u_ev2 * (1.0 - 0.5 * u_ev2)
        cols.append(even - _lpf_1d(even, 8.0))

    mat = torch.stack(cols, dim=-1)  # (L, 30)
    col_rms = torch.sqrt(torch.mean(mat * mat, dim=0, keepdim=True) + 1.0e-12)
    return mat / col_rms


@torch.no_grad()
def compute_segmentation_window_stats(
    x: Tensor,
    y_targets: Tensor,
    win_bounds: Sequence[tuple[int, int]],
    *,
    stride: int = 4,
    sr: float = 48_000.0,
    cached_win_grams_f64: NDArray[np.float64] | None = None,
) -> SegmentationWindowStats:
    """Computes exact per-window segmentation sufficient statistics `(win_grams, win_rhs, win_y2)`
    over `win_bounds`. If `cached_win_grams_f64` is provided (from Arena 4), skips re-accumulating
    the x-only Gram matrices.
    """
    y_2d = y_targets.unsqueeze(-1) if y_targets.ndim == 1 else y_targets
    y_f32 = y_2d.to(device=x.device, dtype=torch.float32)
    m_stems = y_f32.shape[1]
    w_count = len(win_bounds)
    st = max(1, stride)

    probe_basis = build_segmentation_probe_basis(x, sr=sr)
    p_cols = probe_basis.shape[1]

    have_cached_grams = cached_win_grams_f64 is not None and cached_win_grams_f64.shape == (
        w_count,
        p_cols,
        p_cols,
    )
    grams_list: list[NDArray[np.float64]] = []
    rhs_list: list[NDArray[np.float64]] = []
    y2_list: list[NDArray[np.float64]] = []

    for w_idx, (k0, k1) in enumerate(win_bounds):
        pw = probe_basis[k0:k1:st]
        yw = y_f32[k0:k1:st]

        if have_cached_grams:
            assert cached_win_grams_f64 is not None
            g_w = cached_win_grams_f64[w_idx]
            # Accumulate only X^T Y and Y^T Y
            acc_xy = ExactGramAccumulator(p_cols, m_stems, device=x.device)
            acc_xy.add(pw, yw)
            _, c_w = acc_xy.to_host_f64()
        else:
            acc = ExactGramAccumulator(p_cols, m_stems, device=x.device)
            acc.add(pw, yw)
            g_w, c_w = acc.to_host_f64()

        # Exact target energy per stem in this window via 1xM accumulator
        y2_w = np.empty(m_stems, dtype=np.float64)
        yw_cpu = (
            yw.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .numpy()
            .astype(np.float64)
        )
        np.sum(yw_cpu * yw_cpu, axis=0, out=y2_w)

        grams_list.append(g_w)
        rhs_list.append(c_w)
        y2_list.append(y2_w)

    return SegmentationWindowStats(
        win_bounds=tuple((a, b) for a, b in win_bounds),
        win_grams_f64=np.stack(grams_list, axis=0),
        win_rhs_f64=np.stack(rhs_list, axis=0),
        win_y2_f64=np.stack(y2_list, axis=0),
    )


def segment_regimes_dp(
    stats: SegmentationWindowStats,
    *,
    min_regimes: int = 3,
    max_regimes: int = 8,
    min_windows_per_regime: int = 2,
    max_regime_fraction: float = 0.35,
    ridge: float = 1.0e-3,
    bic_penalty_scale: float = 0.35,
) -> list[tuple[int, int]]:
    """Deterministically segments `stats.win_bounds` into `K in [min_regimes, max_regimes]`
    contiguous regimes via dynamic programming (`DP`) over exact per-window ridge residual
    sufficient statistics.
    """
    bounds = list(stats.win_bounds)
    w_count = len(bounds)
    if w_count == 0:
        return [(0, 1)]
    l_tot = bounds[-1][1]
    if w_count <= 2 or l_tot < 48_000:
        mid = l_tot // 2
        return [(0, l_tot)] if l_tot < 8_192 else [(0, mid), (mid, l_tot)]

    min_w = min_windows_per_regime if w_count >= 2 * min_regimes else 1
    max_k = min(max_regimes, max(1, w_count // min_w))
    min_k = min(min_regimes, max_k)
    max_samples = max(
        int(l_tot * max_regime_fraction),
        max(b - a for a, b in bounds) * min_w,
    )

    p_dim = stats.win_grams_f64.shape[1]
    m_stems = stats.win_rhs_f64.shape[2]

    # Prefix sums for O(1) interval Gram, RHS, and target energy queries
    g_pref = np.zeros((w_count + 1, p_dim, p_dim), dtype=np.float64)
    c_pref = np.zeros((w_count + 1, p_dim, m_stems), dtype=np.float64)
    y2_pref = np.zeros((w_count + 1, m_stems), dtype=np.float64)
    np.cumsum(stats.win_grams_f64, axis=0, out=g_pref[1:])
    np.cumsum(stats.win_rhs_f64, axis=0, out=c_pref[1:])
    np.cumsum(stats.win_y2_f64, axis=0, out=y2_pref[1:])

    tot_y2 = np.maximum(y2_pref[-1], 1.0e-12)
    eye_p = np.eye(p_dim, dtype=np.float64)

    # Precompute interval residual cost C[i, j] for 0 <= i < j <= w_count
    cost_mat = np.full((w_count + 1, w_count + 1), np.inf, dtype=np.float64)
    for i in range(w_count):
        for j in range(i + min_w, w_count + 1):
            seg_len = bounds[j - 1][1] - bounds[i][0]
            if seg_len > max_samples and (j - i) > min_w:
                # Allow slightly longer segments only if K_min requires it; penalized heavily
                pass
            g_ij = 0.5 * ((g_pref[j] - g_pref[i]) + (g_pref[j] - g_pref[i]).T)
            c_ij = c_pref[j] - c_pref[i]
            y2_ij = y2_pref[j] - y2_pref[i]

            diag_mean = max(float(np.trace(g_ij)) / float(max(1, p_dim)), 1.0e-9)
            reg_g = g_ij + (ridge * diag_mean) * eye_p
            try:
                l_chol = np.linalg.cholesky(reg_g)
                w_ij = np.linalg.solve(l_chol.T, np.linalg.solve(l_chol, c_ij))
            except np.linalg.LinAlgError:
                w_ij, _, _, _ = np.linalg.lstsq(reg_g, c_ij, rcond=None)

            explained = np.sum(c_ij * w_ij, axis=0)
            rss_per_stem = np.maximum(0.0, y2_ij - explained) / tot_y2
            rss_norm = float(np.mean(rss_per_stem))
            if seg_len > max_samples and max_k >= 3:
                # Soft barrier on segments exceeding max_samples
                over_ratio = float(seg_len - max_samples) / float(max(1, max_samples))
                rss_norm += 0.5 * over_ratio * over_ratio
            cost_mat[i, j] = rss_norm

    # Dynamic programming table: dp[k, j] = min cost to partition first j windows into k regimes
    dp = np.full((max_k + 1, w_count + 1), np.inf, dtype=np.float64)
    parent = np.zeros((max_k + 1, w_count + 1), dtype=np.int32)
    dp[0, 0] = 0.0

    for k in range(1, max_k + 1):
        for j in range(k * min_w, w_count + 1):
            best_c = np.inf
            best_i = 0
            for i in range((k - 1) * min_w, j - min_w + 1):
                c_val = dp[k - 1, i] + cost_mat[i, j]
                if c_val < best_c:
                    best_c = c_val
                    best_i = i
            dp[k, j] = best_c
            parent[k, j] = best_i

    n_eff = max(64.0, float(w_count * 16))
    best_score = np.inf
    best_k = min_k
    for k in range(min_k, max_k + 1):
        rss_k = dp[k, w_count]
        if not np.isfinite(rss_k):
            continue
        score = n_eff * math.log(max(rss_k, 1.0e-6)) + (
            bic_penalty_scale * float(k) * math.log(n_eff)
        )
        if score < best_score:
            best_score = score
            best_k = k

    # Backtrack optimal regime boundaries
    regimes_rev: list[tuple[int, int]] = []
    cur_j = w_count
    for k in range(best_k, 0, -1):
        prev_i = int(parent[k, cur_j])
        regimes_rev.append((bounds[prev_i][0], bounds[cur_j - 1][1]))
        cur_j = prev_i
    regimes_rev.reverse()
    return regimes_rev


def merge_segmentation_stats(
    existing: SegmentationWindowStats,
    new_stats: SegmentationWindowStats,
) -> SegmentationWindowStats:
    """Merges cached `SegmentationWindowStats` from Arena 4 with newly recorded stems."""
    if existing.win_bounds != new_stats.win_bounds:
        raise ValueError("Cannot merge SegmentationWindowStats with mismatched win_bounds")
    return SegmentationWindowStats(
        win_bounds=existing.win_bounds,
        win_grams_f64=existing.win_grams_f64,
        win_rhs_f64=np.concatenate([existing.win_rhs_f64, new_stats.win_rhs_f64], axis=2),
        win_y2_f64=np.concatenate([existing.win_y2_f64, new_stats.win_y2_f64], axis=1),
    )


__all__ = [
    "CausalBsplineWindowSchedule",
    "SegmentationWindowStats",
    "build_orthogonal_fft_windows",
    "build_segmentation_probe_basis",
    "build_stage_window_grids",
    "build_uniform_windows",
    "compute_cumulative_energy_state",
    "compute_segmentation_window_stats",
    "merge_segmentation_stats",
    "segment_regimes_dp",
]
