"""Unit & property tests for `dnam_trainer.dsp.selective_rglru` (`CL-03`, `CL-04`, `CL-05`)
and `< 1e-12` parity against `dnam_engine.dsp.SelectiveRgLru`."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dnam_trainer.dsp.selective_rglru import (
    DifferentiableSelectiveRgLru,
    algebraic_unit_rotor_torch,
    h2_input_norm_gain,
    log_octave_pole_init,
    parallel_associative_scan_complex,
)


def test_algebraic_unit_rotor_and_h2_norm_bounds() -> None:
    thetas = torch.linspace(1e-4, float(np.pi) - 1e-4, 2048, dtype=torch.float64)
    u_cos, u_sin = algebraic_unit_rotor_torch(thetas)
    mag_sq = u_cos * u_cos + u_sin * u_sin
    # CL-03: unit-circle rotor magnitude <= 1.000000000000 (strictly inside/on unit disk)
    assert float(torch.max(mag_sq).item()) <= 1.0 + 1e-15
    assert float(torch.max(torch.abs(mag_sq - 1.0)).item()) < 2.5e-8

    radii = torch.linspace(0.25, 0.9995, 128, dtype=torch.float64)
    gammas = h2_input_norm_gain(radii)
    assert torch.all(gammas > 0.0)
    assert torch.all(gammas <= 1.0)

    freqs, th_init = log_octave_pole_init(12, sample_rate_hz=48_000.0, dtype=torch.float64)
    assert freqs.shape == (12,)
    assert th_init.shape == (12,)
    assert abs(float(freqs[0].item()) - 25.0) < 1e-9
    assert abs(float(freqs[-1].item()) - 16_000.0) < 1e-6


def test_selective_rglru_rust_reference_parity_and_chunked_scan() -> None:
    layer = DifferentiableSelectiveRgLru(
        num_channels=4, sample_rate_hz=48_000.0, dtype=torch.float64
    )
    t = np.linspace(0.0, 1.0, 600, dtype=np.float64)
    x_np = (0.6 * np.sin(2.0 * np.pi * 11.0 * t) + 0.3 * np.cos(2.0 * np.pi * 37.0 * t)).astype(
        np.float64
    )
    x_t = torch.from_numpy(x_np).view(1, 1, -1).expand(1, 4, -1)
    y_torch, z_final = layer(x_t)
    assert z_final.shape == (1, 4, 1)

    for ch in range(4):
        ref_proc = layer.to_reference_dsp(channel=ch)
        y_rust = ref_proc.process(x_np)
        y_py = y_torch[0, ch].detach().cpu().numpy()
        max_diff = float(np.max(np.abs(y_py - y_rust)))
        assert max_diff < 1e-12, f"SelectiveRgLru channel {ch} parity mismatch: {max_diff}"


def test_selective_rglru_state_carry_and_gradients() -> None:
    layer = DifferentiableSelectiveRgLru(num_channels=3, dtype=torch.float64)
    x = torch.randn((2, 3, 300), dtype=torch.float64, requires_grad=True)
    y, _ = layer(
        x,
        delta_r_low=torch.full((2, 3, 1), 0.01, dtype=torch.float64),
        delta_theta_rad=torch.full((2, 3, 1), 0.02, dtype=torch.float64),
    )
    loss = torch.sum(y**2)
    loss.backward()
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))

    # State carry across two chunks
    with torch.no_grad():
        x_det = torch.randn((1, 3, 320), dtype=torch.float64)
        y_full, _ = layer(x_det)
        y_1, z_1 = layer(x_det[:, :, :160])
        y_2, _ = layer(x_det[:, :, 160:], z_init=z_1)
        y_cat = torch.cat([y_1, y_2], dim=-1)
        assert float(torch.max(torch.abs(y_full - y_cat)).item()) < 1e-12

    with pytest.raises(ValueError, match="num_channels"):
        DifferentiableSelectiveRgLru(num_channels=0)
    with pytest.raises(ValueError, match="a_complex shape"):
        parallel_associative_scan_complex(
            torch.ones((1, 4), dtype=torch.complex128),
            torch.ones((1, 5), dtype=torch.complex128),
        )
    with pytest.raises(IndexError):
        layer.to_reference_dsp(channel=10)
