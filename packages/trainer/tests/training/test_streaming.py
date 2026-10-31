"""Unit tests for blocked streaming primitives (`apply_on_block`), block-size invariance, and
planner `max_tensor_bytes` compliance.
"""

from __future__ import annotations

import torch

from dnam_trainer.training.accelerator import resolve_accelerator_runtime
from dnam_trainer.training.cascade_primitives import (
    apply_on_block,
    causal_1pole_halo_samples,
    fast_1pole_lpf,
)
from dnam_trainer.training.numerics import ExactGramAccumulator


def test_apply_on_block_matches_full_length_causal_filter() -> None:
    torch.manual_seed(2026)
    l_seq = 16_384
    c_ch = 4
    src = torch.randn((l_seq, c_ch), dtype=torch.float32)
    f_hz = 250.0
    halo = causal_1pole_halo_samples(f_hz, sr=48_000.0, bezout_order4=True)
    k_len = causal_1pole_halo_samples(f_hz, sr=48_000.0, bezout_order4=False)

    full = fast_1pole_lpf(src, f_hz, sr=48_000.0, bezout_order4=True, k_len=k_len)

    blocks: list[torch.Tensor] = []
    block_len = 4_096
    for t0 in range(0, l_seq, block_len):
        t1 = min(l_seq, t0 + block_len)
        blk = apply_on_block(
            lambda s: fast_1pole_lpf(s, f_hz, sr=48_000.0, bezout_order4=True, k_len=k_len),
            src,
            t0,
            t1,
            left_halo=halo,
        )
        blocks.append(blk)

    recon = torch.cat(blocks, dim=0)
    max_diff = float(torch.max(torch.abs(full - recon)).item())
    assert max_diff < 1e-5, f"Blocked causal LPF diverged from full-length: {max_diff}"


def test_exact_gram_accumulator_block_size_invariance() -> None:
    torch.manual_seed(99)
    n_rows = 4_096
    n_cols = 24
    x = torch.randn((n_rows, n_cols), dtype=torch.float32) * 0.4
    y = torch.randn((n_rows, 2), dtype=torch.float32) * 0.2

    acc_single = ExactGramAccumulator(n_cols, 2, device=torch.device("cpu"))
    acc_single.add(x, y)
    g1, r1 = acc_single.to_host_f64()

    acc_chunked = ExactGramAccumulator(n_cols, 2, device=torch.device("cpu"))
    for s0 in range(0, n_rows, 512):
        s1 = min(n_rows, s0 + 512)
        acc_chunked.add(x[s0:s1], y[s0:s1])
    g2, r2 = acc_chunked.to_host_f64()

    assert float(abs(g1 - g2).max()) <= 1e-6
    assert float(abs(r1 - r2).max()) <= 1e-6


def test_execution_plan_respects_max_tensor_bytes() -> None:
    rt = resolve_accelerator_runtime("cpu", budget_gb_override=8.0, force_reprobe=True)
    plan = rt.plan_2d(
        num_samples=1_440_000,
        num_stems=32,
        max_basis_cols=480,
    )
    assert rt.max_tensor_bytes <= 2 * (1024**3)
    single_basis_block_bytes = plan.block_rows * 480 * 4
    assert single_basis_block_bytes <= rt.max_tensor_bytes


def test_orthogonal_fft_cascade_and_nested_tier_blocks() -> None:
    from dnam_trainer.training.cascade_orthogonal_fft import run_4stage_orthogonal_fft_cascade
    from dnam_trainer.training.numerics import HostDenseSolver
    from dnam_trainer.training.selective_cascade import (
        _build_nested_tier_blocks,
        calibrate_3stage_selective_cascade,
    )

    torch.manual_seed(42)
    l_seq = 4_096
    x = torch.randn(l_seq, dtype=torch.float32) * 0.25
    y = torch.tanh(2.5 * x.unsqueeze(-1).expand(l_seq, 7)).contiguous()
    solver = HostDenseSolver()
    stages: list[int] = []

    def _report(
        step: int, z_lf: torch.Tensor, z_mf: torch.Tensor, z_hf: torch.Tensor
    ) -> torch.Tensor:
        stages.append(step)
        return torch.ones(z_lf.shape[1], dtype=torch.float32)

    z1, _z2, _z3, d4, _z4_lf, _z4_mf, _z4_hf, _z3_hf = run_4stage_orthogonal_fft_cascade(
        x,
        y,
        sr=48_000.0,
        on_stage_progress=lambda s, _m: stages.append(s * 10),
        report_step=_report,
        solver=solver,
    )
    assert z1.shape == (l_seq, 7)
    assert d4.shape == (l_seq, 7)
    assert len(stages) >= 7

    # Also test _build_nested_tier_blocks for m_col = 10 (<= 16) and m_col = 18 (> 16)
    for m_col, exp_w in ((10, 16), (18, 32)):
        t_m = torch.randn((512, m_col), dtype=torch.float32) * 0.1
        bl, bf, bu = _build_nested_tier_blocks(
            t_m,
            t_m * 1.1,
            t_m * 1.2,
            t_m * 0.1,
            t_m * 0.4,
            t_m * 0.4,
            t_m * 0.4,
            t_m * 0.3,
            solver=solver,
        )
        assert bl.shape == (512, exp_w)
        assert bf.shape == (512, exp_w)
        assert bu.shape == (512, exp_w)

    # Test calibrate_3stage_selective_cascade with cached lags and out_calib_records
    import tempfile
    from pathlib import Path

    import numpy as np

    from dnam_trainer.training.calib_cache import SidecarCacheStore
    from dnam_trainer.training.metrics import StemNoiseProfile
    from dnam_trainer.training.pipeline import DnamTriPhysicsModel

    x_np = x.numpy()
    y0_np = y[:, 0].numpy()
    records: dict[str, object] = {}
    step_metrics: list[tuple[int, int, float]] = []
    b_l, b_f, b_u = calibrate_3stage_selective_cascade(
        x_np,
        [y0_np, (y0_np * 1.0e-7).astype(np.float32)],
        sample_rate_hz=48_000,
        silent_stem_flags=[False, True],
        stem_noise_floors_dbfs=[(-90.0, -90.0, -90.0), (-90.0, -90.0, -90.0)],
        cached_lags=[0.0, 0.0],
        out_calib_records=records,
        on_step_metrics=lambda s, tot, bl, _bf, _bu, _lf, _mf, _hf: step_metrics.append(
            (s, tot, bl)
        ),
    )
    assert b_l.shape == (l_seq, 16)
    assert b_f.shape == (l_seq, 16)
    assert b_u.shape == (l_seq, 16)
    assert "all_stem_lags" in records
    assert len(step_metrics) >= 3

    # Reuse cached_seg_stats, cached_s1_proj, cached_lti_irs, non-zero lag, and all-silent fallback
    from typing import Any, cast

    rec_any = cast(dict[str, Any], records)
    s1_p = rec_any.get("s1_proj")
    b_l2, _, _ = calibrate_3stage_selective_cascade(
        x_np,
        [y0_np, (y0_np * 1.0e-7).astype(np.float32)],
        sample_rate_hz=48_000,
        silent_stem_flags=[True, True],
        cached_lags=[1.0, 0.0],
        cached_lti_irs=[rec_any["lti_irs_by_stem_idx"][0], None],
        cached_seg_stats=rec_any.get("seg_stats"),
        cached_s1_proj=torch.from_numpy(s1_p) if s1_p is not None else None,
    )
    assert b_l2.shape == (l_seq, 16)

    # Also test DnamTriPhysicsModel.calibrate_selective_cascade with Arena 4 + sidecar cache
    # and 32-column tier blocks (covering c_knots population in pipeline.py)
    import dnam_trainer.training.pipeline as pipe_mod

    orig_cal = pipe_mod.calibrate_3stage_selective_cascade
    with tempfile.TemporaryDirectory() as tmp_d:
        sc = SidecarCacheStore(Path(tmp_d) / "sc", num_stems=2)
        sc.measured_read_bw_bytes_per_s = 1.0e12  # ensure should_materialize returns True
        sc.measured_write_bw_bytes_per_s = 1.0e12
        mdl = DnamTriPhysicsModel(sample_rate_hz=48_000)
        x_2ch = np.column_stack([x_np, x_np])
        y_2ch = np.column_stack([y0_np, y0_np])
        sha_in = "a1" * 32
        sha_tg = "b2" * 32
        prof = StemNoiseProfile(
            is_silent=False,
            noise_floor_dbfs_broadband=-90.0,
            noise_floor_dbfs_lf=-92.0,
            noise_floor_dbfs_mf=-92.0,
            noise_floor_dbfs_hf=-95.0,
            body_rms_dbfs=-18.0,
            band_energy_broadband=-18.0,
            band_energy_lf=-20.0,
            band_energy_mf=-20.0,
            band_energy_hf=-24.0,
            source="measured_silent_frames",
        )

        def _fast_cal32(*_a: Any, **kw: Any) -> tuple[Any, Any, Any]:
            out_r = kw.get("out_calib_records")
            if isinstance(out_r, dict):
                out_r.update(rec_any)
            blk32 = np.ones((l_seq, 32), dtype=np.float32) * 0.1
            return blk32, blk32, blk32

        pipe_mod.calibrate_3stage_selective_cascade = _fast_cal32
        try:
            mdl.calibrate_selective_cascade(
                [x_2ch],
                [y_2ch],
                in_shas=[sha_in],
                u_ctrl_stems=[(0.0, 0.0, 0.0)],
                silent_stem_flags=[False],
                stem_noise_floors_dbfs=[(-92.0, -92.0, -95.0)],
                stem_noise_profiles=[prof],
                stem_ids=["s0"],
                tgt_shas=[sha_tg],
                sidecar_cache=sc,
            )
            assert mdl._arena4_calib is not None
            assert "s0" in mdl._arena4_calib.stems
            assert sha_in in mdl._calibrated_knot_coords

            # Second pass with existing_arena4 (no sidecar_cache) hits Arena 4 s1_proj reuse
            mdl2 = DnamTriPhysicsModel(sample_rate_hz=48_000)
            mdl2.calibrate_selective_cascade(
                [x_2ch],
                [y_2ch],
                in_shas=[sha_in],
                u_ctrl_stems=[(0.0, 0.0, 0.0)],
                silent_stem_flags=[False],
                stem_noise_floors_dbfs=[(-92.0, -92.0, -95.0)],
                stem_noise_profiles=[prof],
                stem_ids=["s0"],
                tgt_shas=[sha_tg],
                existing_arena4=mdl._arena4_calib,
            )
            assert sha_in in mdl2._calibrated_tier_blocks
        finally:
            pipe_mod.calibrate_3stage_selective_cascade = orig_cal

        # Third pass with sidecar_cache hits tier_blocks sidecar fast-path
        mdl3 = DnamTriPhysicsModel(sample_rate_hz=48_000)
        mdl3.calibrate_selective_cascade(
            [x_2ch],
            [y_2ch],
            in_shas=[sha_in],
            sidecar_cache=sc,
        )
        assert sha_in in mdl3._calibrated_tier_blocks


def test_column_chunked_matrix_and_polish_branches() -> None:
    import pytest

    from dnam_trainer.training.cascade_primitives import build_fast_refinement_gates
    from dnam_trainer.training.cascade_selective_stages import (
        ColumnChunkedMatrix,
        _ensure_chunked,
        _project_chunked_matrix,
        _project_nonlinear_batches_chunked,
        build_interstage_basis,
        solve_fullrate_hf_residual_polish,
    )
    from dnam_trainer.training.numerics import HostDenseSolver
    from dnam_trainer.training.windowing import (
        CausalBsplineWindowSchedule,
        compute_cumulative_energy_state,
    )

    with pytest.raises(ValueError, match="at least one non-empty"):
        ColumnChunkedMatrix([])

    t_empty = torch.zeros((64, 0), dtype=torch.float32)
    t_a = torch.randn((64, 10), dtype=torch.float32)
    t_b = torch.randn((64, 6), dtype=torch.float32)
    t_wide = torch.randn((64, 24), dtype=torch.float32)
    cm = ColumnChunkedMatrix.from_tensors(
        [t_empty, t_a, t_b, t_wide], max_chunk_cols=16, free_inputs=True
    )
    assert cm.shape == (64, 40)
    assert _ensure_chunked(torch.randn((64, 4), dtype=torch.float32)).shape == (64, 4)
    assert _ensure_chunked(cm) is cm

    solver = HostDenseSolver()
    assert _project_chunked_matrix(cm, 64, solver, free_input=False) is cm
    cm_proj = _project_chunked_matrix(cm, 12, solver, free_input=True)
    assert cm_proj.shape == (64, 12)
    out = cm_proj.matmul(torch.ones((12, 2), dtype=torch.float32), free_self=True)
    assert out.shape == (64, 2)
    cm_proj.free()

    # _project_nonlinear_batches_chunked with cached_proj
    bands = torch.randn((128, 12), dtype=torch.float32)
    chunks1, proj1 = _project_nonlinear_batches_chunked(
        bands, (0.8, 2.5), "inst", 16, solver, sr=48_000.0
    )
    chunks2, _ = _project_nonlinear_batches_chunked(
        bands, (0.8, 2.5), "inst", 16, solver, sr=48_000.0, cached_proj=proj1
    )
    assert torch.allclose(chunks1[0], chunks2[0], atol=1e-5)

    # build_interstage_basis with 8 columns (7 < m_col <= 14) and return_chunked=False
    z8 = torch.randn((256, 8), dtype=torch.float32) * 0.2
    u8 = build_interstage_basis(
        z8,
        z8 * 0.9,
        (20.0, 100.0),
        20.0,
        35.0,
        8.0,
        (1.0, 3.0),
        use_power_sag=False,
        solver=solver,
        return_chunked=False,
    )
    assert isinstance(u8, torch.Tensor)
    assert u8.shape[0] == 256

    # solve_fullrate_hf_residual_polish at l_tot = 48_000 (triggers use_lti_polish = True)
    l48 = 48_000
    x48 = torch.randn(l48, dtype=torch.float32) * 0.2
    z3_48 = torch.tanh(1.5 * x48).unsqueeze(-1)
    r3_lf = 0.02 * z3_48
    r3_mf = 0.02 * z3_48
    r3_hf = 0.02 * z3_48
    sig = torch.tensor([10.0], dtype=torch.float32)
    wb48 = [(0, 24_000), (24_000, 48_000)]
    e_cum = compute_cumulative_energy_state(x48, sr=48_000.0)
    sched48 = CausalBsplineWindowSchedule(e_cum, wb48, overlap=256)
    gates48 = build_fast_refinement_gates(x48, sr=48_000.0)
    d4_tot, _d4_lf, _d4_mf, _d4_hf = solve_fullrate_hf_residual_polish(
        z3_48,
        z3_48,
        r3_lf,
        r3_mf,
        r3_hf,
        sig,
        sig,
        sig,
        sched48,
        gates48,
        wb48,
        z1_full=z3_48,
        solver=solver,
        sr=48_000.0,
    )
    assert d4_tot.shape == (l48, 1)
