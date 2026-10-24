"""Unit tests for `dnam_trainer.training.surrogate_delay` (`Test 6` Multi-Scale Gaussian Derivative
Envelope Surrogate)."""

from __future__ import annotations

import torch

from dnam_trainer.training.surrogate_delay import (
    _sample_cubic_lagrange_1d,
    optimize_delay_with_surrogate,
    surrogate_delay_readout,
)


def test_surrogate_delay_readout_forward_backward_and_large_basin_convergence() -> None:
    # 1. Verify forward & backward pass through SurrogateDelayReadout
    t = torch.linspace(0.0, 1.0, 2048, dtype=torch.float32)
    env = torch.exp(-((t - 0.35) ** 2) / (2.0 * (0.04**2)))
    carrier = torch.sin(2.0 * 3.14159265 * 180.0 * t)
    sig = (env * carrier).unsqueeze(0)

    tau_param = torch.tensor([[24.5]], dtype=torch.float32, requires_grad=True)
    y_out = surrogate_delay_readout(sig, tau_param, sigma_grad=15.0)
    assert y_out.shape == sig.shape
    loss = torch.sum(y_out**2)
    loss.backward()
    assert tau_param.grad is not None
    assert torch.isfinite(tau_param.grad).all()

    # 2. Test 6: Optimize fractional delay from a +55 sample initial offset -> < 0.05 sample error
    true_delay = 82.35
    target, _ = _sample_cubic_lagrange_1d(sig, torch.tensor([[true_delay]], dtype=torch.float32))
    est_delay = optimize_delay_with_surrogate(
        sig,
        target,
        init_delay_samples=27.0,  # 55.35 samples away (spanning multiple carrier cycles!)
        min_delay_samples=5.0,
        max_delay_samples=160.0,
        num_steps=140,
        sigma_start=35.0,
        sigma_end=0.0,
        lr=2.5,
    )
    assert abs(est_delay - true_delay) < 0.05, (
        f"Delay error too large: {abs(est_delay - true_delay)}"
    )
