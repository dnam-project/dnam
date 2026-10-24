"""Unit & property tests for `dnam_trainer.dsp.poly_adaa` (`CL-01`, `CL-02`) and
`< 1e-12` parity against `dnam_engine.dsp.PolyAdaa`."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dnam_trainer.dsp.poly_adaa import (
    DifferentiablePolyAdaa,
    clamp_pre_activation_torch,
    poly_adaa_algebraic_torch,
    poly_adaa_waveshaper_torch,
    resting_dc_equilibrium,
)


def test_poly_adaa_zero_sample0_transient_and_rust_reference_parity() -> None:
    layer = DifferentiablePolyAdaa(num_channels=4, r_ac=0.9993, dtype=torch.float64)
    with torch.no_grad():
        layer.b_conv.copy_(torch.tensor([0.35, -0.45, 0.80, -1.20], dtype=torch.float64))
        layer.alpha.copy_(torch.tensor([0.90, 0.75, 1.05, 0.85], dtype=torch.float64))
        layer.beta.copy_(torch.tensor([0.55, 0.40, 0.65, 0.30], dtype=torch.float64))
        layer.delta.copy_(torch.tensor([0.15, -0.10, 0.20, -0.05], dtype=torch.float64))

    # 1. Exact 0.000000 Sample-0 and Sample-1 transient on silence even when b_conv != 0 (CL-01,
    # CL-02)
    silence = torch.zeros((1, 4, 256), dtype=torch.float64)
    y_sil, _ = layer(silence)
    assert float(torch.max(torch.abs(y_sil)).item()) < 1e-15

    # 2. Exact < 1e-12 parity against Rust dnam_engine.dsp.PolyAdaa reference
    t = np.linspace(0.0, 1.0, 512, dtype=np.float64)
    x_np = (0.8 * np.sin(2.0 * np.pi * 17.0 * t) + 0.4 * np.cos(2.0 * np.pi * 53.0 * t)).astype(
        np.float64
    )
    x_t = torch.from_numpy(x_np).view(1, 1, -1).expand(1, 4, -1)
    y_torch, _ = layer(x_t)

    for ch in range(4):
        ref_proc = layer.to_reference_dsp(channel=ch)
        y_rust = ref_proc.process(x_np)
        y_py = y_torch[0, ch].detach().cpu().numpy()
        max_diff = float(np.max(np.abs(y_py - y_rust)))
        assert max_diff < 1e-12, f"Channel {ch} parity mismatch: {max_diff}"


def test_poly_adaa_clamps_gradients_and_state_carry() -> None:
    u = clamp_pre_activation_torch(torch.tensor([-10.0, 0.0, 10.0], dtype=torch.float64))
    assert torch.allclose(u, torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64))

    # Coincident u_prev == u_curr has zero division and matches instantaneous waveshaper
    u_val = torch.tensor([0.42], dtype=torch.float64)
    w_inst = poly_adaa_waveshaper_torch(u_val, 0.5, 0.1)
    w_alg = poly_adaa_algebraic_torch(u_val, u_val, 0.5, 0.1)
    assert float(torch.abs(w_inst - w_alg).item()) < 1e-15

    u_r, a_r, m_r = resting_dc_equilibrium(
        torch.tensor([0.3], dtype=torch.float64),
        torch.tensor([0.5], dtype=torch.float64),
        torch.tensor([0.1], dtype=torch.float64),
    )
    assert torch.allclose(a_r, m_r)
    assert float(torch.abs(u_r - 0.2).item()) < 1e-15

    layer = DifferentiablePolyAdaa(num_channels=2, dtype=torch.float64)
    x_full = torch.randn((2, 2, 300), dtype=torch.float64, requires_grad=True)
    y_full, _ = layer(
        x_full,
        delta_b_conv=torch.full((2, 2, 1), 0.05, dtype=torch.float64),
        delta_beta=torch.full((2, 2, 1), 0.02, dtype=torch.float64),
        delta_delta=torch.full((2, 2, 1), -0.01, dtype=torch.float64),
    )
    loss = torch.sum(y_full**2)
    loss.backward()
    assert x_full.grad is not None
    assert torch.all(torch.isfinite(x_full.grad))

    # State-carry across two halves matches full sequence
    with torch.no_grad():
        x_det = torch.randn((1, 2, 300), dtype=torch.float64)
        y_all, _ = layer(x_det)
        st0 = layer.resting_state(batch_size=1)
        y_p1, st1 = layer(x_det[:, :, :150], state=st0)
        y_p2, _ = layer(x_det[:, :, 150:], state=st1)
        y_cat = torch.cat([y_p1, y_p2], dim=-1)
        assert float(torch.max(torch.abs(y_all - y_cat)).item()) < 1e-13

    with pytest.raises(ValueError, match="num_channels"):
        DifferentiablePolyAdaa(num_channels=0)
    with pytest.raises(ValueError, match="r_ac"):
        DifferentiablePolyAdaa(num_channels=2, r_ac=0.5)
    with pytest.raises(IndexError):
        layer.to_reference_dsp(channel=99)
