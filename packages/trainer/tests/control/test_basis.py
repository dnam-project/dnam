"""Unit tests for `dnam_trainer.control.basis` (`CL-12`..`CL-14`: Lobatto HDMR, Rank-4 FM,
Order-Statistic Optimal-Transport delay trajectory fitting, and Step-0 Physical Variance Gating)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dnam_trainer.control.basis import (
    ParametricControlHyperBus,
    compute_lobatto_degree_and_anchors,
    compute_step0_variance_gates,
    eval_node_anchored_newton_basis,
    extract_blind_lfo_phase_trajectory,
    fit_order_statistic_delay_transport,
    normalize_and_clamp_range,
)
from dnam_trainer.schema import parse_compact_range_spec, parse_compact_switch_spec


def test_normalize_and_clamp_range_and_lobatto_basis() -> None:
    # CL-12: Normalization anchored to declared_range even when safe_clamped_range is narrower
    clamped, u_norm, was_clamped = normalize_and_clamp_range(9.0, (0.0, 10.0), (2.0, 7.0))
    assert clamped == 7.0
    assert was_clamped
    assert abs(u_norm - 0.4) < 1e-9

    with pytest.raises(ValueError, match="declared_range span"):
        normalize_and_clamp_range(5.0, (5.0, 5.0))

    # CL-13: Cardinality-bounded degree K_p = min(U_p - 1, 3) and zero anchor perturbation
    deg1, anchors1 = compute_lobatto_degree_and_anchors([0.0], default_u=0.0)
    assert deg1 == 0
    assert torch.allclose(
        eval_node_anchored_newton_basis(0.5, anchors1, deg1),
        torch.zeros(3, dtype=torch.float64),
    )

    deg3, anchors3 = compute_lobatto_degree_and_anchors([-1.0, 0.0, 1.0, 0.5], default_u=0.0)
    assert deg3 == 3
    # At anchor u1 (0.0), all 3 terms vanish identically (0.000000)
    b_at_u1 = eval_node_anchored_newton_basis(anchors3[0], anchors3, deg3)
    assert float(torch.max(torch.abs(b_at_u1)).item()) < 1e-15
    # At anchor u2, psi_2 and psi_3 vanish identically
    b_at_u2 = eval_node_anchored_newton_basis(anchors3[1], anchors3, deg3)
    assert abs(float(b_at_u2[1].item())) < 1e-15
    assert abs(float(b_at_u2[2].item())) < 1e-15

    with pytest.raises(ValueError, match="Lobatto degree"):
        eval_node_anchored_newton_basis(0.0, anchors3, 5)


def test_order_statistic_delay_transport_and_variance_gates() -> None:
    # Test 8D: Order-statistic sorting prevents permutation-averaging collapse
    u_coords = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    # Even if raw delays are permuted per stem, sorting into tau_{(1)} < tau_{(2)} recovers exact
    # log-linear law
    delays_ms = np.array(
        [
            [20.0, 10.0],
            [14.1421356, 28.2842712],
            [40.0, 20.0],
        ],
        dtype=np.float64,
    )
    intercept, slope = fit_order_statistic_delay_transport(u_coords, delays_ms)
    pred_u0 = np.exp(intercept)
    assert np.allclose(pred_u0, [14.1421356, 28.2842712], rtol=1e-4)
    assert np.all(slope > 0.0)

    # CL-14: Step-0 Physical Variance Gating zeroes out unmodulated parameter rows
    u_mat = np.array([[-1.0, -1.0], [0.0, 1.0], [1.0, -1.0], [0.5, 0.5]], dtype=np.float64)
    # Param 0 depends only on control 0; Param 1 is constant
    phys_mat = np.column_stack([2.0 * u_mat[:, 0], np.ones(4)])
    gates = compute_step0_variance_gates(u_mat, phys_mat, threshold=0.5)
    assert gates[0, 0] == 1.0
    assert gates[0, 1] == 0.0
    assert gates[1, 1] == 0.0


def test_blind_lfo_extraction_and_parametric_control_hyper_bus() -> None:
    t = np.arange(48_000, dtype=np.float64) / 48_000.0
    carrier = np.sin(2.0 * np.pi * 440.0 * t)
    am_mod = 1.0 + 0.5 * np.cos(2.0 * np.pi * 2.0 * t)
    wet = (carrier * am_mod).astype(np.float32)

    f_lfo, phase_traj = extract_blind_lfo_phase_trajectory(wet, sample_rate_hz=48_000)
    assert 1.5 <= f_lfo <= 2.5
    assert phase_traj.shape == (3000,)

    ctrls = [
        parse_compact_range_spec("drive=0..10:5"),
        parse_compact_switch_spec("attack=cut,flat,boost:flat"),
    ]
    bus = ParametricControlHyperBus(ctrls, num_target_params=8, fm_rank=4)
    psi_batch = bus.encode_batch_coords(
        [
            {"drive": 5.0, "attack": "flat"},
            {"drive": 10.0, "attack": "boost"},
        ]
    )
    assert psi_batch.shape == (2, 5)
    t_main, dt_fm = bus.evaluate_main_and_fm(psi_batch)
    assert t_main.shape == (2, 8)
    assert dt_fm.shape == (2, 8)
    total = bus(psi_batch)
    assert torch.allclose(total, t_main + dt_fm)

    # Also test edge cases: empty batch, empty controls, constant delay transport, 1D variance
    # gates, silent LFO
    assert bus.encode_batch_coords([]).shape == (0, 5)
    empty_bus = ParametricControlHyperBus([], num_target_params=4)
    assert empty_bus.encode_control_coords({}).shape == (1,)

    f_zero, p_zero = extract_blind_lfo_phase_trajectory(
        np.zeros(512, dtype=np.float32), sample_rate_hz=48_000
    )
    assert f_zero == 0.0
    assert p_zero.shape[0] > 0

    ic_const, sl_const = fit_order_statistic_delay_transport([0.0, 0.0], [[1.0, 2.0], [1.0, 2.0]])
    assert np.allclose(sl_const, 0.0)
    assert ic_const.shape == (2,)
    with pytest.raises(ValueError, match="does not match"):
        fit_order_statistic_delay_transport([0.0], [[1.0, 2.0], [1.0, 2.0]])

    g_1d = compute_step0_variance_gates([0.0, 1.0], [0.0, 2.0])
    assert g_1d.shape == (1, 1)
    g_single = compute_step0_variance_gates([[0.0]], [[1.0]])
    assert np.all(g_single == 1.0)
    with pytest.raises(ValueError, match="Stem count mismatch"):
        compute_step0_variance_gates([[0.0], [1.0]], [[1.0]])
