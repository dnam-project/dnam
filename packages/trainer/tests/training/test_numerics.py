"""Unit and property tests for `dnam_trainer.training.numerics`."""

from __future__ import annotations

import numpy as np
import torch

from dnam_trainer.events import SolverEvent, TrainingEvent
from dnam_trainer.training.numerics import (
    ExactGramAccumulator,
    HostDenseSolver,
    HostSolveSpec,
    argmax_i32,
    dd_add,
    pow2_ceil_fp32,
    quick_two_sum,
    recombine_f32_pair_to_f64,
    recombine_f32_trilimb_to_f64,
    split_f64_to_f32_pair,
    split_f64_to_f32_trilimb,
    split_int32_to_f32_pair,
    td_add,
    two_sum,
)


class _SolverEventCollector:
    def __init__(self) -> None:
        self.events: list[TrainingEvent] = []

    def on_training_event(self, event: TrainingEvent) -> None:
        self.events.append(event)


def test_pow2_ceil_and_argmax_i32() -> None:
    vals = torch.tensor([0.0, 1.0, 1.5, 3.2, 8.0, 9.1], dtype=torch.float32)
    p2 = pow2_ceil_fp32(vals)
    assert p2.dtype == torch.float32
    assert float(p2[1].item()) == 1.0
    assert float(p2[2].item()) == 2.0
    assert float(p2[3].item()) == 4.0
    assert float(p2[4].item()) == 8.0
    assert float(p2[5].item()) == 16.0

    mat = torch.tensor([[1.0, 5.0, 3.0], [9.0, 2.0, 4.0]], dtype=torch.float32)
    idx_col = argmax_i32(mat, dim=1)
    assert idx_col.dtype == torch.int32
    assert idx_col.tolist() == [1, 0]

    idx_flat = argmax_i32(mat, dim=None)
    assert idx_flat.dtype == torch.int32
    assert int(idx_flat.item()) == 3


def test_eft_and_limb_roundtrips() -> None:
    rng = np.random.default_rng(42)
    arr_f64 = rng.standard_normal((16, 16)).astype(np.float64) * 123.456789012345

    hi2, lo2 = split_f64_to_f32_pair(arr_f64)
    assert hi2.dtype == torch.float32
    assert lo2.dtype == torch.float32
    rec2 = recombine_f32_pair_to_f64(hi2, lo2)
    np.testing.assert_allclose(rec2, arr_f64, rtol=1e-13, atol=1e-13)

    hi3, mid3, lo3 = split_f64_to_f32_trilimb(arr_f64)
    assert hi3.dtype == torch.float32
    assert mid3.dtype == torch.float32
    assert lo3.dtype == torch.float32
    rec3 = recombine_f32_trilimb_to_f64(hi3, mid3, lo3)
    np.testing.assert_allclose(rec3, arr_f64, rtol=1e-15, atol=1e-15)

    # Int32 split
    i32 = torch.tensor([-1_073_741_823, -123456789, 0, 123456789, 1_073_741_823], dtype=torch.int32)
    ihi, ilo = split_int32_to_f32_pair(i32)
    rec_i = ihi.to(torch.float64) + ilo.to(torch.float64)
    np.testing.assert_array_equal(rec_i.numpy().astype(np.int64), i32.numpy().astype(np.int64))

    # two_sum, quick_two_sum, dd_add, td_add
    a = torch.tensor([1.0e6], dtype=torch.float32)
    b = torch.tensor([1.0e-3], dtype=torch.float32)
    s, e = two_sum(a, b)
    q_s, q_e = quick_two_sum(s, e)
    assert abs((float(q_s.item()) + float(q_e.item())) - (1.0e6 + 1.0e-3)) < 1e-9

    d_hi, d_lo = dd_add(a, torch.zeros_like(a), b, torch.zeros_like(b))
    assert abs((float(d_hi.item()) + float(d_lo.item())) - (1.0e6 + 1.0e-3)) < 1e-9

    t_hi, t_mid, t_lo = td_add(a, torch.zeros_like(a), torch.zeros_like(a), b, torch.zeros_like(b))
    assert (
        abs((float(t_hi.item()) + float(t_mid.item()) + float(t_lo.item())) - (1.0e6 + 1.0e-3))
        < 1e-9
    )


def test_exact_gram_accumulator_matches_fp64_reference() -> None:
    rng = np.random.default_rng(7)
    x_np = rng.standard_normal((20_000, 24)).astype(np.float32)
    # Introduce wide dynamic range across columns
    scales = np.geomspace(1e-3, 1e2, 24).astype(np.float32)
    x_np = x_np * scales[None, :]
    y_np = rng.standard_normal((20_000, 4)).astype(np.float32)

    # 1. Exact 24-bit fixed-point multiples of column power-of-two scale -> <= 1e-13 vs fp64
    q_np = rng.integers(-8_000_000, 8_000_000, size=(20_000, 24)).astype(np.float64) * (2.0**-24)
    scales_p2 = (2.0 ** np.arange(-8, 16, dtype=np.float64)).astype(np.float32)
    x_exact_np = (q_np * scales_p2[None, :]).astype(np.float32)
    y_exact_np = (
        rng.integers(-8_000_000, 8_000_000, size=(20_000, 4)).astype(np.float64) * (2.0**-24)
    ).astype(np.float32)

    x_t = torch.from_numpy(x_exact_np)
    y_t = torch.from_numpy(y_exact_np)

    acc1 = ExactGramAccumulator(24, 4, device="cpu")
    acc1.add(x_t[:12_000], y_t[:12_000], weight=1.25)

    acc2 = ExactGramAccumulator(24, 4, device="cpu")
    acc2.add(x_t[12_000:], y_t[12_000:], weight=1.25)

    acc1.merge(acc2, weight=1.0)
    g_acc, c_acc = acc1.to_host_f64()

    x_64 = x_exact_np.astype(np.float64)
    y_64 = y_exact_np.astype(np.float64)
    g_ref = 1.25 * (x_64.T @ x_64)
    c_ref = 1.25 * (x_64.T @ y_64)

    rel_g = np.linalg.norm(g_acc - g_ref) / np.linalg.norm(g_ref)
    rel_c = np.linalg.norm(c_acc - c_ref) / np.linalg.norm(c_ref)
    assert rel_g <= 1e-13
    assert rel_c <= 1e-13

    # 2. General Gaussian float32 across wide dynamic range -> <= 5e-8 vs fp64
    acc_gauss = ExactGramAccumulator(24, 4, device="cpu")
    acc_gauss.add(torch.from_numpy(x_np), torch.from_numpy(y_np))
    g_gauss, _ = acc_gauss.to_host_f64()
    g_gauss_ref = x_np.astype(np.float64).T @ x_np.astype(np.float64)
    assert np.linalg.norm(g_gauss - g_gauss_ref) / np.linalg.norm(g_gauss_ref) <= 5e-8

    g_hi, g_lo = acc1.gram_hi_lo()
    c_hi, c_lo = acc1.rhs_hi_lo()
    assert g_hi.shape == (24, 24)
    assert g_lo.shape == (24, 24)
    assert c_hi.shape == (24, 4)
    assert c_lo.shape == (24, 4)
    assert len(acc1.gram_trilimb()) == 3
    assert len(acc1.rhs_trilimb()) == 3


def test_host_dense_solver_batch_and_projectors() -> None:
    rng = np.random.default_rng(99)
    x_np = rng.standard_normal((2048, 16)).astype(np.float32)
    w_true = rng.standard_normal((16, 3)).astype(np.float32)
    y_np = x_np @ w_true

    acc_main = ExactGramAccumulator(16, 3, device="cpu")
    acc_main.add(torch.from_numpy(x_np), torch.from_numpy(y_np))

    acc_sec = ExactGramAccumulator(16, 3, device="cpu")
    acc_sec.add(torch.from_numpy(x_np[:512]), torch.from_numpy(y_np[:512]))

    obs = _SolverEventCollector()
    solver = HostDenseSolver(observer=obs)
    w_est = solver.solve_single(acc_main, ridge_scale=1e-9, min_ridge=1e-12)
    np.testing.assert_allclose(w_est.numpy(), w_true, rtol=1e-4, atol=1e-4)

    # Batch solve with secondary accumulator and rhs_col_slice
    specs = [
        HostSolveSpec(
            accumulator=acc_main,
            secondary_accumulator=acc_sec,
            primary_weight=0.8,
            secondary_weight=0.2,
            rhs_col_slice=(0, 2),
        )
    ]
    res_batch = solver.solve_batch(specs, pass_name="test_batch")
    assert len(res_batch) == 1
    assert res_batch[0].shape == (16, 2)
    np.testing.assert_allclose(res_batch[0].numpy(), w_true[:, :2], rtol=1e-4, atol=1e-4)
    assert any(isinstance(e, SolverEvent) for e in obs.events)

    # Orthonormal, interstage, and target manifold projectors
    mat = torch.from_numpy(x_np)
    proj = solver.fit_orthonormal_projector(mat, 8)
    assert proj.shape == (16, 8)
    proj_basis = solver.project_orthonormal_basis(mat, 8)
    assert proj_basis.shape == (2048, 8)

    res_w = torch.ones(16, dtype=torch.float32)
    p_lin, p_nl = solver.fit_interstage_projector(mat, 6, 4, res_weights=res_w)
    assert p_lin.shape == (16, 6)
    assert p_nl.shape == (16, 4)

    p_tm = solver.fit_target_manifold_projector(mat, n_out=10, res_weights=res_w)
    assert p_tm.shape == (16, 10)
