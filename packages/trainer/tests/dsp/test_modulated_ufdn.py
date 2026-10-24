"""Unit & property tests for `dnam_trainer.dsp.modulated_ufdn` (`CL-06`..`CL-11`)."""

from __future__ import annotations

import math

import pytest
import torch

from dnam_trainer.dsp.modulated_ufdn import (
    DifferentiableModulatedUfdn,
    coercive_lyapunov_loop_step_torch,
    cubic_lagrange_read_torch,
    dual_pointer_switch_crossfade_torch,
    givens_householder_matrix_torch,
)


def test_givens_householder_unitarity_and_coercive_lyapunov_bounds() -> None:
    # CL-08: Exact unitarity U^T @ U = I across Dual-Mono (psi=0, v=0), Reverb, and Ping-Pong
    # (psi=pi/2)
    for psi in (0.0, 0.35, math.pi * 0.5):
        for v_vec in (torch.zeros(8, dtype=torch.float64), torch.ones(8, dtype=torch.float64)):
            u_mat = givens_householder_matrix_torch(psi, v_vec, num_lines=8, dtype=torch.float64)
            ident = u_mat.T @ u_mat
            err = float(torch.max(torch.abs(ident - torch.eye(8, dtype=torch.float64))).item())
            assert err < 1e-14

    # CL-09..CL-11: Coercive Lyapunov loop step bounded by v_ceil <= 0.985 even under +40 dB input
    extreme_in = torch.tensor([100.0, -100.0, 0.0], dtype=torch.float64)
    s_next, _, _ = coercive_lyapunov_loop_step_torch(
        extreme_in,
        g_lin=0.9990,
        sigma_knee=0.45,
        v_ceil=0.985,
        r_dc=0.995,
        dc_x_prev=torch.zeros_like(extreme_in),
        dc_y_prev=torch.zeros_like(extreme_in),
    )
    assert float(torch.max(torch.abs(s_next)).item()) <= 0.985 + 1e-12


def test_cubic_lagrange_and_dual_pointer_switch_crossfade() -> None:
    sig = torch.sin(torch.linspace(0.0, 30.0, 512, dtype=torch.float64))
    y_del = cubic_lagrange_read_torch(sig, 12.35)
    assert y_del.shape == sig.shape
    assert torch.all(torch.isfinite(y_del))

    # CL-07: 128-sample raised-cosine dual-pointer switch crossfade
    y_xf = dual_pointer_switch_crossfade_torch(sig, 10.0, 25.0, trig_idx=128, xfade_len=128)
    assert y_xf.shape == sig.shape
    # Before trig_idx matches tau_old; after trig_idx + 128 matches tau_new
    y_old = cubic_lagrange_read_torch(sig, 10.0)
    y_new = cubic_lagrange_read_torch(sig, 25.0)
    assert torch.allclose(y_xf[:128], y_old[:128])
    assert torch.allclose(y_xf[256:], y_new[256:])


def test_differentiable_modulated_ufdn_forward_bypass_and_reference() -> None:
    ufdn = DifferentiableModulatedUfdn(num_lines=4, active=True, dtype=torch.float64)
    x_st = torch.randn((1, 2, 256), dtype=torch.float64, requires_grad=True)
    y_st = ufdn(
        x_st,
        delta_delay_samples=torch.full((4,), 0.5, dtype=torch.float64),
        delta_g_lin=torch.full((4,), 0.01, dtype=torch.float64),
        lfo_beam_gain=torch.ones((1, 2, 256), dtype=torch.float64),
    )
    assert y_st.shape == (1, 2, 256)
    loss = torch.sum(y_st**2)
    loss.backward()
    assert x_st.grad is not None
    assert torch.all(torch.isfinite(x_st.grad))

    ref_proc = ufdn.to_reference_dsp(line=0)
    assert ref_proc.params.delay_samples >= 4

    # Bypass mode (active=False) returns input unchanged
    ufdn_bypass = DifferentiableModulatedUfdn(num_lines=4, active=False)
    x_raw = torch.randn((2, 64), dtype=torch.float32)
    assert torch.equal(ufdn_bypass(x_raw), x_raw)

    with pytest.raises(ValueError, match="num_lines"):
        DifferentiableModulatedUfdn(num_lines=3)
    with pytest.raises(ValueError, match="r_dc"):
        DifferentiableModulatedUfdn(num_lines=4, r_dc=0.5)
    with pytest.raises(IndexError):
        ufdn.to_reference_dsp(line=99)
