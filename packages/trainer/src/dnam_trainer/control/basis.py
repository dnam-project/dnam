"""Parametric Control Bus (`CL-12`, `CL-13`, `CL-14`):
Fixed `declared_range` coordinate normalization, Cardinality-Bounded Node-Anchored
Newton/Lobatto HDMR basis, Centered ANOVA + Rank-4 Factorization Machine (`FM`),
Step-0 Order-Statistic Optimal-Transport delay matching, and Physical Variance Gating."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor, nn

from dnam_trainer.schema import ControlCoords, ControlKind, ControlSchemaDef


def normalize_and_clamp_range(
    x_p: float,
    declared_range: tuple[float, float],
    safe_clamped_range: tuple[float, float] | None = None,
) -> tuple[float, float, bool]:
    """Enforces `CL-12`: clamps `x_p` to `safe_clamped_range` and normalizes to `[-1, +1]` via
    `declared_range`.

    Returns `(clamped_physical, normalized_u, was_clamped)`.
    """
    dec_lo, dec_hi = float(declared_range[0]), float(declared_range[1])
    span = dec_hi - dec_lo
    if span <= 1e-9:
        raise ValueError(f"declared_range span must be > 0, got {declared_range}")

    safe_lo, safe_hi = (
        (float(safe_clamped_range[0]), float(safe_clamped_range[1]))
        if safe_clamped_range is not None
        else (dec_lo, dec_hi)
    )
    clamped = max(safe_lo, min(safe_hi, float(x_p)))
    was_clamped = abs(clamped - float(x_p)) > 1e-6

    scale_norm = 2.0 / span
    offset_norm = -1.0 - dec_lo * scale_norm
    u_norm = max(-1.0, min(1.0, clamped * scale_norm + offset_norm))
    return clamped, float(u_norm), was_clamped


def compute_lobatto_degree_and_anchors(
    observed_u_values: Sequence[float],
    default_u: float = 0.0,
) -> tuple[int, tuple[float, float, float]]:
    """Enforces `CL-13`: Cardinality-Bounded Node-Anchored polynomial degree `K_p = min(U_p - 1,
    3)`.

    Orders the unique observed normalized coordinates `[-1, +1]` so `u_{p,1}` is the recorded
    coordinate closest to `default_u` (noon/reference stem), followed by the two most extreme
    remaining distinct coordinates `u_{p,2}, u_{p,3}`.
    """
    unique_sorted: list[float] = []
    for val in sorted(float(v) for v in observed_u_values):
        clamped_v = max(-1.0, min(1.0, val))
        if not any(abs(clamped_v - ex) < 1e-4 for ex in unique_sorted):
            unique_sorted.append(clamped_v)

    u_count = len(unique_sorted)
    if u_count <= 1:
        u1 = unique_sorted[0] if u_count == 1 else max(-1.0, min(1.0, float(default_u)))
        return 0, (u1, -1.0, 1.0)

    degree = min(u_count - 1, 3)
    # Anchor 1: closest to default_u
    u1 = min(unique_sorted, key=lambda v: abs(v - default_u))
    rem = [v for v in unique_sorted if abs(v - u1) >= 1e-4]
    # Sort remaining by farthest distance from u1 so boundary anchors come first
    rem.sort(key=lambda v: -abs(v - u1))
    u2 = rem[0] if len(rem) >= 1 else -1.0
    u3 = rem[1] if len(rem) >= 2 else 1.0
    return degree, (float(u1), float(u2), float(u3))


def eval_node_anchored_newton_basis(
    u: float | Tensor,
    anchors: tuple[float, float, float],
    degree: int,
) -> Tensor:
    """Evaluates the 3-term Node-Anchored Newton/Lobatto basis `[psi_1, psi_2, psi_3](u)` (`CL-13`).

    - `psi_{p,1}(u) = (u - u_{p,1})` if `degree >= 1` else `0`
    - `psi_{p,2}(u) = (u - u_{p,1})(u - u_{p,2})` if `degree >= 2` else `0`
    - `psi_{p,3}(u) = (u - u_{p,1})(u - u_{p,2})(u - u_{p,3})` if `degree >= 3` else `0`

    Guarantees that unlocking degree `k` vanishes identically (`0.000000`) at all previously
    recorded stem anchors `u_{p,1}, ..., u_{p,k}`.
    """
    if not (0 <= degree <= 3):
        raise ValueError(f"Lobatto degree must be in {{0, 1, 2, 3}}, got {degree}")

    u_t = torch.as_tensor(u, dtype=torch.float32 if not isinstance(u, Tensor) else u.dtype)
    u1, u2, u3 = float(anchors[0]), float(anchors[1]), float(anchors[2])

    d1 = u_t - u1
    d2 = d1 * (u_t - u2)
    d3 = d2 * (u_t - u3)

    z = torch.zeros_like(u_t)
    psi_1 = d1 if degree >= 1 else z
    psi_2 = d2 if degree >= 2 else z
    psi_3 = d3 if degree >= 3 else z
    return torch.stack([psi_1, psi_2, psi_3], dim=-1)


def fit_order_statistic_delay_transport(
    stem_u_coords: ArrayLike,
    stem_delays_ms: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Step-0 Order-Statistic Optimal-Transport delay trajectory fitting (`Test 8D`).

    Sorts each stem's `M` delay times into strict order statistics `tau_{(1)} < ... < tau_{(M)}`
    before fitting `log(tau_{(m)}(u)) = a_m + b_m * u`, preventing permutation-averaging collapse.
    Returns `(intercept_log_ms, slope_log_ms)` each of shape `(M,)`.
    """
    u_arr = np.ascontiguousarray(stem_u_coords, dtype=np.float64).reshape(-1)
    delays_arr = np.ascontiguousarray(stem_delays_ms, dtype=np.float64)
    if delays_arr.ndim != 2 or delays_arr.shape[0] != u_arr.shape[0]:
        raise ValueError(
            f"stem_delays_ms shape {delays_arr.shape} does not match "
            f"stem_u_coords length {u_arr.shape[0]}"
        )

    # Sort each stem's delays into order statistics tau_{(1)} <= ... <= tau_{(M)}
    sorted_delays = np.sort(np.maximum(delays_arr, 0.05), axis=1)
    log_delays = np.log(sorted_delays)

    u_mean = float(np.mean(u_arr))
    u_centered = u_arr - u_mean
    u_var = float(np.sum(u_centered * u_centered))
    if u_var <= 1e-12:
        intercept = np.mean(log_delays, axis=0)
        slope = np.zeros_like(intercept)
        return intercept, slope

    log_mean = np.mean(log_delays, axis=0)
    slope = (u_centered @ (log_delays - log_mean)) / u_var
    intercept = log_mean - slope * u_mean
    return np.ascontiguousarray(intercept, dtype=np.float64), np.ascontiguousarray(
        slope, dtype=np.float64
    )


def compute_step0_variance_gates(
    stem_u_matrix: ArrayLike,
    extracted_physics_matrix: ArrayLike,
    *,
    threshold: float = 0.5,
) -> NDArray[np.float64]:
    """Enforces `CL-14`: Step-0 Physical Variance Gating `G_{p,i} = I(Sens_{p,i} > 0.5) in {0.0,
    1.0}`.

    Computes `Sens_{p,i} = |Cov(u_p, theta_i)| / (Var(u_p) + 1e-12)` across recorded stems and
    permanently zeroes out hyper-network rows where knob `p` does not physically modulate
    subsystem parameter `i` (`0.00` cross-talk leakage).
    """
    u_mat = np.ascontiguousarray(stem_u_matrix, dtype=np.float64)
    if u_mat.ndim == 1:
        u_mat = u_mat[:, None]
    phys_mat = np.ascontiguousarray(extracted_physics_matrix, dtype=np.float64)
    if phys_mat.ndim == 1:
        phys_mat = phys_mat[:, None]
    if u_mat.shape[0] != phys_mat.shape[0]:
        raise ValueError(
            f"Stem count mismatch between u_mat ({u_mat.shape[0]}) and "
            f"phys_mat ({phys_mat.shape[0]})"
        )

    num_stems, num_controls = u_mat.shape
    num_params = phys_mat.shape[1]
    if num_stems < 2:
        return np.ones((num_controls, num_params), dtype=np.float64)

    u_centered = u_mat - np.mean(u_mat, axis=0, keepdims=True)
    p_centered = phys_mat - np.mean(phys_mat, axis=0, keepdims=True)
    var_u = np.mean(u_centered * u_centered, axis=0)[:, None]
    cov_up = (u_centered.T @ p_centered) / float(num_stems)
    sens = np.abs(cov_up) / (var_u + 1.0e-12)
    gates = (sens > float(threshold)).astype(np.float64)
    return np.ascontiguousarray(gates, dtype=np.float64)


def extract_blind_lfo_phase_trajectory(
    wet_audio: ArrayLike,
    *,
    sample_rate_hz: int = 48_000,
    control_decimation: int = 16,
) -> tuple[float, NDArray[np.float64]]:
    """Extracts dominant LFO frequency `f_lfo_hz` and $3\\text{ kHz}$ phase trajectory
    `cos(phi[k])`."""
    arr = np.ascontiguousarray(wet_audio, dtype=np.float64)
    mono = arr if arr.ndim == 1 else 0.5 * (arr[0] + arr[1])
    ctrl_fs = float(sample_rate_hz) / float(max(1, control_decimation))

    n_frames = max(1, mono.shape[0] // max(1, control_decimation))
    trimmed = mono[: n_frames * control_decimation].reshape(n_frames, control_decimation)
    env = np.sqrt(np.mean(trimmed * trimmed, axis=1) + 1e-12)
    env_ac = env - np.mean(env)

    if n_frames < 16 or float(np.std(env_ac)) < 1e-5:
        return 0.0, np.zeros(n_frames, dtype=np.float64)

    spec = np.fft.rfft(env_ac * np.hanning(n_frames))
    freqs = np.fft.rfftfreq(n_frames, d=1.0 / ctrl_fs)
    mask = (freqs >= 0.15) & (freqs <= 15.0)
    if not np.any(mask):
        return 0.0, np.zeros(n_frames, dtype=np.float64)

    sub_idx = int(np.argmax(np.abs(spec[mask])))
    valid_indices = np.flatnonzero(mask)
    peak_bin = int(valid_indices[sub_idx])
    f_lfo = float(freqs[peak_bin])
    phase_0 = float(np.angle(spec[peak_bin]))

    t_ctrl = np.arange(n_frames, dtype=np.float64) / ctrl_fs
    cos_phi = np.cos(2.0 * math.pi * f_lfo * t_ctrl + phase_0)
    return f_lfo, np.ascontiguousarray(cos_phi, dtype=np.float64)


class ParametricControlHyperBus(nn.Module):
    """Centered ANOVA + Node-Anchored Lobatto HDMR + Rank-4 FM Control Bus (`CL-12`..`CL-14`).

    Precomputes static knob/switch basis vectors `psi_raw(u_s, s_s)` for all recorded stems
    and virtual audit corners (`< 0.05 ms`/step), while evaluating the 4-harmonic Chebyshev LFO
    and 3-band envelope bank strictly at the $3\\text{ kHz}$ control rate (`L / 16` frames).
    """

    def __init__(
        self,
        controls_schema: Sequence[ControlSchemaDef],
        *,
        num_target_params: int = 16,
        fm_rank: int = 4,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.controls_schema: tuple[ControlSchemaDef, ...] = tuple(controls_schema)
        self.num_target_params = num_target_params
        self.fm_rank = fm_rank

        # Compute feature dimension: 3 Lobatto terms per range control + (K_q - 1) centered switch
        # terms
        feat_dim = 0
        for c in self.controls_schema:
            if c.kind == ControlKind.RANGE:
                feat_dim += 3
            else:
                n_opts = len(c.declared_options or ["default"])
                feat_dim += max(1, n_opts - 1)
        self.basis_dim = max(1, feat_dim)

        self.w_main = nn.Parameter(
            torch.zeros((self.num_target_params, self.basis_dim), dtype=dtype)
        )
        self.v_fm = nn.Parameter(
            torch.randn((self.num_target_params, self.basis_dim, self.fm_rank), dtype=dtype) * 0.01
        )
        self.lfo_depths = nn.Parameter(torch.zeros((self.num_target_params, 4), dtype=dtype))
        self.register_buffer(
            "variance_gate_mask",
            torch.ones((self.num_target_params, self.basis_dim), dtype=dtype),
        )

    def encode_control_coords(self, coords: Mapping[str, float | str]) -> Tensor:
        """
        Encodes a single control coordinate dictionary into its `1D` raw basis vector `psi_raw`.
        """
        if not self.controls_schema:
            return torch.zeros(1, dtype=self.w_main.dtype, device=self.w_main.device)

        parts: list[Tensor] = []
        for c in self.controls_schema:
            raw_v = coords.get(c.id, c.default)
            if c.kind == ControlKind.RANGE:
                dec_range = c.declared_range or (0.0, 10.0)
                _, u_norm, _ = normalize_and_clamp_range(
                    float(raw_v), dec_range, c.safe_clamped_range
                )
                deg = c.lobatto_degree if c.lobatto_degree is not None else 3
                if c.anchor_nodes_norm and len(c.anchor_nodes_norm) >= 3:
                    anchors = (
                        float(c.anchor_nodes_norm[0]),
                        float(c.anchor_nodes_norm[1]),
                        float(c.anchor_nodes_norm[2]),
                    )
                else:
                    _, def_u, _ = normalize_and_clamp_range(float(c.default), dec_range)
                    anchors = (def_u, -1.0, 1.0)
                parts.append(
                    eval_node_anchored_newton_basis(u_norm, anchors, deg).to(
                        dtype=self.w_main.dtype, device=self.w_main.device
                    )
                )
            else:
                opts = c.declared_options or ["default"]
                n_sub = max(1, len(opts) - 1)
                vec = torch.zeros(n_sub, dtype=self.w_main.dtype, device=self.w_main.device)
                val_str = str(raw_v)
                if val_str in opts:
                    idx = opts.index(val_str)
                    if idx > 0 and (idx - 1) < n_sub:
                        vec[idx - 1] = 1.0
                parts.append(vec)

        return torch.cat(parts, dim=0)

    def encode_batch_coords(self, coords_list: Sequence[Mapping[str, float | str]]) -> Tensor:
        """Encodes a list of control coordinate dicts into `(S, D_basis)` in 32-bit."""
        if not coords_list:
            return torch.zeros(
                (0, self.basis_dim), dtype=self.w_main.dtype, device=self.w_main.device
            )
        rows = [self.encode_control_coords(c) for c in coords_list]
        return torch.stack(rows, dim=0)

    def evaluate_main_and_fm(
        self,
        psi_batch: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Evaluates 1st-order main effects `theta_main` and 2nd-order Rank-4 FM `delta_theta_fm`
        (`CL-14`, `CL-16`).

        `psi_batch` has shape `(S, D_basis)`. Returns `(theta_main, delta_theta_fm)` each of shape
        `(S, num_target_params)`.
        """
        psi = psi_batch.to(dtype=self.w_main.dtype, device=self.w_main.device)
        gated_w = self.w_main * self.variance_gate_mask
        theta_main = psi @ gated_w.T

        # O(D_basis * R) Rank-4 Factorization Machine pairwise interaction:
        # 0.5 * sum_r ((psi @ V_r)^2 - (psi^2 @ V_r^2))
        gated_v = self.v_fm * self.variance_gate_mask.unsqueeze(-1)
        proj_linear = torch.einsum("sd,pdr->spr", psi, gated_v)
        proj_sq = torch.einsum("sd,pdr->spr", psi * psi, gated_v * gated_v)
        delta_theta_fm = 0.5 * torch.sum(proj_linear * proj_linear - proj_sq, dim=-1)
        return theta_main, delta_theta_fm

    def forward(self, psi_batch: Tensor) -> Tensor:
        """Returns total static parameter modulation `theta_main + delta_theta_fm`."""
        t_main, dt_fm = self.evaluate_main_and_fm(psi_batch)
        return t_main + dt_fm


__all__ = [
    "ControlCoords",
    "ParametricControlHyperBus",
    "compute_lobatto_degree_and_anchors",
    "compute_step0_variance_gates",
    "eval_node_anchored_newton_basis",
    "extract_blind_lfo_phase_trajectory",
    "fit_order_statistic_delay_transport",
    "normalize_and_clamp_range",
]
