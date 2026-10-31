"""Multi-Scale Gaussian Derivative Envelope Surrogate for Fractional Delay Optimization (`Test 6`).

Implements the custom `torch.autograd.Function` from
[`docs/training_and_iterative_resume.md`](../../../../../docs/training_and_iterative_resume.md)
§2.1:
- **Forward pass:** Exact 4-tap Cubic Lagrange fractional delay interpolation at $\tau(n)$.
- **Backward pass:** Replaces the high-frequency point-wise finite-difference derivative with a
  Gaussian-derivative smoothed envelope gradient ($-\\dot{g}_{\\sigma_{\text{grad}}} * u$) so delay
  parameters $\tau$ have a wide, convex basin of attraction ($\\ge \\pm 80\text{ samples}$) and
  converge
  to $< 0.05\text{ samples}$ error as $\\sigma_{\text{grad}}$ anneals from $35.0 \to 0.0\text{
  samples}$.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def _sample_cubic_lagrange_1d(
    signal: torch.Tensor,
    delay_samples: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Samples `signal` `(B, T)` at fractional delay `delay_samples` `(B, T)` or `(B, 1)`.

    Returns `(y_delayed, dy_ddelay_exact)` both shaped `(B, T)`.
    """
    bsz, num_frames = signal.shape
    device = signal.device
    dtype = signal.dtype

    max_pad = math.ceil(float(delay_samples.detach().max().item())) + 6
    max_pad = max(max_pad, 8)
    padded = F.pad(signal, (max_pad, 2), mode="constant", value=0.0)

    t_grid = torch.arange(num_frames, device=device, dtype=dtype).unsqueeze(0).expand(bsz, -1)
    tau = delay_samples.expand(bsz, num_frames)
    read_pos = t_grid + float(max_pad) - tau

    d_int = torch.floor(read_pos)
    alpha = read_pos - d_int
    base_idx = d_int.to(dtype=torch.int32)

    pad_len = padded.shape[1]
    idx_m1 = (base_idx + 1).clamp(0, pad_len - 1)
    idx_0 = base_idx.clamp(0, pad_len - 1)
    idx_p1 = (base_idx - 1).clamp(0, pad_len - 1)
    idx_p2 = (base_idx - 2).clamp(0, pad_len - 1)

    row_idx = torch.arange(bsz, device=device, dtype=torch.int32).unsqueeze(1)
    u_m1 = padded[row_idx, idx_m1]
    u_0 = padded[row_idx, idx_0]
    u_p1 = padded[row_idx, idx_p1]
    u_p2 = padded[row_idx, idx_p2]

    a = alpha
    c_m1 = -a * (a - 1.0) * (a - 2.0) * (1.0 / 6.0)
    c_0 = (a + 1.0) * (a - 1.0) * (a - 2.0) * 0.5
    c_p1 = -(a + 1.0) * a * (a - 2.0) * 0.5
    c_p2 = (a + 1.0) * a * (a - 1.0) * (1.0 / 6.0)

    y_delayed = c_m1 * u_m1 + c_0 * u_0 + c_p1 * u_p1 + c_p2 * u_p2

    # Exact derivative w.r.t. delay tau (note read_pos = t - tau, so d(u(t - tau))/d(tau)):
    dc_m1 = -(3.0 * a * a - 6.0 * a + 2.0) * (1.0 / 6.0)
    dc_0 = (3.0 * a * a - 4.0 * a - 1.0) * 0.5
    dc_p1 = -(3.0 * a * a - 2.0 * a - 2.0) * 0.5
    dc_p2 = (3.0 * a * a - 1.0) * (1.0 / 6.0)
    dy_dtau = dc_m1 * u_m1 + dc_0 * u_0 + dc_p1 * u_p1 + dc_p2 * u_p2

    return y_delayed, dy_dtau


def _gaussian_smoothed_delay_derivative(
    signal: torch.Tensor,
    delay_samples: torch.Tensor,
    sigma_grad: float,
) -> torch.Tensor:
    """Computes the Gaussian-smoothed derivative $\\partial y(t - \\tau)/\\partial \\tau$ with
    scale `sigma_grad`."""
    if sigma_grad <= 0.25:
        _, dy_dtau = _sample_cubic_lagrange_1d(signal, delay_samples)
        return dy_dtau

    radius = min(max(math.ceil(3.0 * sigma_grad), 3), 256)
    device = signal.device
    dtype = signal.dtype
    k = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    inv_sigma2 = 1.0 / (sigma_grad * sigma_grad)
    # F.conv1d computes cross-correlation sum_k u[t + k] * w[k].
    # Since d/dtau u(t - tau) = -du/dt(t - tau), w[k] must be negative for k > 0 and positive for k
    # < 0:
    g = torch.exp(-0.5 * k * k * inv_sigma2)
    g = g / g.sum().clamp(min=1e-12)
    weights = -(k * inv_sigma2) * g
    kernel = weights.view(1, 1, -1)

    u_smooth_deriv = F.conv1d(signal.unsqueeze(1), kernel, padding=radius).squeeze(1)
    sampled_deriv, _ = _sample_cubic_lagrange_1d(u_smooth_deriv, delay_samples)
    return sampled_deriv


class SurrogateDelayReadout(torch.autograd.Function):
    """Custom `autograd.Function` pairing exact 4-tap Cubic Lagrange forward delay readout
    with a multi-scale Gaussian derivative envelope surrogate backward pass (`Test 6`).
    """

    @staticmethod
    def forward(
        ctx: Any,
        signal: torch.Tensor,
        delay_samples: torch.Tensor,
        sigma_grad: float = 0.0,
    ) -> torch.Tensor:
        y_delayed, _ = _sample_cubic_lagrange_1d(signal, delay_samples)
        ctx.save_for_backward(signal, delay_samples)
        ctx.sigma_grad = float(sigma_grad)
        return y_delayed

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: Any,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        grad_output: torch.Tensor = grad_outputs[0]
        signal, delay_samples = ctx.saved_tensors
        sigma_grad: float = ctx.sigma_grad

        grad_signal: torch.Tensor | None = None
        grad_delay: torch.Tensor | None = None

        if ctx.needs_input_grad[0]:
            # Adjoint of delay by +tau is advance by -tau (approximated via integer shift for signal
            # grad)
            grad_signal = grad_output.clone()

        if ctx.needs_input_grad[1]:
            dy_dtau = _gaussian_smoothed_delay_derivative(signal, delay_samples, sigma_grad)
            raw_grad = grad_output * dy_dtau
            if delay_samples.shape == signal.shape:
                grad_delay = raw_grad
            else:
                grad_delay = raw_grad.sum(dim=-1, keepdim=True)
                if delay_samples.ndim == 1:
                    grad_delay = grad_delay.squeeze(-1)

        return grad_signal, grad_delay, None


def surrogate_delay_readout(
    signal: torch.Tensor,
    delay_samples: torch.Tensor,
    *,
    sigma_grad: float = 0.0,
) -> torch.Tensor:
    """Applies [`SurrogateDelayReadout`] to `signal` `(B, T)` at `delay_samples`."""
    res: Any = SurrogateDelayReadout.apply(signal, delay_samples, float(sigma_grad))
    assert isinstance(res, torch.Tensor)
    return res


def optimize_delay_with_surrogate(
    input_signal: torch.Tensor,
    target_signal: torch.Tensor,
    *,
    init_delay_samples: float,
    min_delay_samples: float = 1.0,
    max_delay_samples: float = 250.0,
    num_steps: int = 160,
    sigma_start: float = 35.0,
    sigma_end: float = 0.0,
    lr: float = 2.5,
) -> float:
    """Optimizes a scalar fractional delay $\\tau$ from `init_delay_samples` to match
    `target_signal`
    using the Multi-Scale Gaussian Derivative Envelope Surrogate (`Test 6`).

    Combines an envelope correlation warmup over the smoothed energy envelope with a cosine-annealed
    `SurrogateDelayReadout` gradient refinement so initial offsets up to $\\pm 80\\text{ samples}$
    converge to $< 0.05\\text{ samples}$ error.
    """
    x = input_signal.detach().to(dtype=torch.float32)
    y = target_signal.detach().to(dtype=torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if y.ndim == 1:
        y = y.unsqueeze(0)

    tau_param = torch.nn.Parameter(
        torch.tensor([[float(init_delay_samples)]], device=x.device, dtype=torch.float32)
    )
    optimizer = torch.optim.Adam([tau_param], lr=lr)

    # Low-pass envelope of |x| and |y| for wide-basin coarse alignment during the first half of
    # steps
    env_radius = max(math.ceil(2.0 * sigma_start), 8)
    k_env = torch.arange(-env_radius, env_radius + 1, device=x.device, dtype=torch.float32)
    w_env = torch.exp(-0.5 * (k_env / max(sigma_start, 1.0)) ** 2)
    w_env = (w_env / w_env.sum()).view(1, 1, -1)
    x_env = F.conv1d(torch.abs(x).unsqueeze(1), w_env, padding=env_radius).squeeze(1)
    y_env = F.conv1d(torch.abs(y).unsqueeze(1), w_env, padding=env_radius).squeeze(1)

    for step in range(num_steps):
        progress = step / max(num_steps - 1, 1)
        optimizer.zero_grad()
        tau_clamped = tau_param.clamp(min=min_delay_samples, max=max_delay_samples)

        if progress < 0.65:
            # Stage A: Wide-basin Gaussian envelope surrogate
            optimizer.param_groups[0]["lr"] = lr
            sigma_t = max(4.0, sigma_start * (1.0 - (progress / 0.65) * 0.8))
            env_hat = surrogate_delay_readout(x_env, tau_clamped, sigma_grad=sigma_t)
            loss = torch.mean((env_hat - y_env) ** 2)
        else:
            # Stage B: Fine carrier phase alignment with annealed sigma_grad -> sigma_end
            fine_prog = (progress - 0.65) / 0.35
            optimizer.param_groups[0]["lr"] = lr * 0.08 * (1.0 - 0.5 * fine_prog)
            sigma_t = sigma_end + (1.5 - sigma_end) * ((1.0 - fine_prog) ** 2)
            y_hat = surrogate_delay_readout(x, tau_clamped, sigma_grad=sigma_t)
            loss = torch.mean((y_hat - y) ** 2)

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            tau_param.clamp_(min=min_delay_samples, max=max_delay_samples)

    # Final local sub-sample polish around the converged estimate
    center_tau = float(tau_param.detach().item())
    best_tau = center_tau
    best_err = float("inf")
    with torch.no_grad():
        for step_i in range(-400, 401):
            delta = step_i * 0.005
            cand = max(min_delay_samples, min(max_delay_samples, center_tau + delta))
            t_cand = torch.tensor([[cand]], device=x.device, dtype=torch.float32)
            y_cand, _ = _sample_cubic_lagrange_1d(x, t_cand)
            err = float(torch.mean((y_cand - y) ** 2).item())
            if err < best_err:
                best_err = err
                best_tau = cand

    return best_tau
