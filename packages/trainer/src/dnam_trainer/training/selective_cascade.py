"""3-Stage Causal Selective State-Space + 3-Band Decoupled (`LF`/`MF`/`HF`) Cascade Orchestrator.

Coordinates:
1. [`cascade_primitives`](cascade_primitives.py): Causal 1-pole & 4th-order Bezout filters,
   `1x B-Spline PolyADAA`, division-free B-spline/wavefolder bases, sub-sample USB jitter
   alignment, causal Wiener LTI identification, and 16/32-channel control modulation.
2. [`cascade_orthogonal_fft`](cascade_orthogonal_fft.py): 4-Stage Parseval Commuted 3-Band
   Orthogonal FFT Decoupled Comparator & Wavefolder Cascade (`m_col <= 16`).
3. [`cascade_selective_stages`](cascade_selective_stages.py): 3-Stage Selective State-Space
   Tube/Solid-State Cascade + Stage-4 Quadrature Hysteresis Polish (`up to 32` active stems).
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from dnam_trainer.training.accelerator import empty_device_cache, free_tensor_storage
from dnam_trainer.training.cascade_orthogonal_fft import (
    apply_ltas_hf_tilt_balance,
    build_refinement_fuzz_basis,
    build_stage1_fuzz_basis,
    parseval_inband_projectors,
    project_fft_band,
    run_4stage_orthogonal_fft_cascade,
    solve_3band_decoupled_window_stage,
)
from dnam_trainer.training.cascade_primitives import (
    build_causal_bspline_window_weights,
    build_fast_refinement_gates,
    compute_tier32_modulator_vectors,
    fast_1pole_lpf,
    fast_poly_adaa_batch,
    fast_poly_inst_batch,
    fast_wavefolder_inst_batch,
    measure_and_shift_usb_jitter,
    measure_and_shift_usb_jitter_tensor,
    modulate_tier_blocks_with_controls,
    project_orthonormal_basis,
    rms_balance_cols,
    shift_signal_fft_1d,
    solve_cholesky_ridge,
    solve_lti_cabinet_and_space_stage,
    solve_lti_cabinet_and_space_stage_with_ir,
    split_bezout_3band,
)
from dnam_trainer.training.cascade_selective_stages import (
    build_interstage_basis,
    run_3stage_selective_tube_cascade,
    solve_fullrate_hf_residual_polish,
    solve_selective_stage_3band,
)
from dnam_trainer.training.numerics import HostDenseSolver
from dnam_trainer.training.windowing import SegmentationWindowStats


def _top_svd_cols(
    mat: Tensor,
    k_cols: int,
    svd_stride: int,
    solver: HostDenseSolver,
) -> Tensor:
    """Projects `mat` `(L, M)` onto its top `k_cols` right singular directions via exact Gram
    + host `float64` `eigh`.
    """
    mat_f32 = mat.to(dtype=torch.float32)
    mat_n = mat_f32 / (torch.sqrt(torch.mean(mat_f32 * mat_f32, dim=0, keepdim=True)) + 1.0e-12)
    proj = solver.fit_target_manifold_projector(mat_n[::svd_stride], k_cols)
    return mat_f32 @ proj


def _build_nested_tier_blocks(
    z1_full: Tensor,
    z2_full: Tensor,
    z3_full: Tensor,
    d4_tot: Tensor,
    z4_lf: Tensor,
    z4_mf: Tensor,
    z4_hf: Tensor,
    z3_hf: Tensor,
    *,
    solver: HostDenseSolver,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Constructs RMS-balanced nested tier base blocks `(b_lite, b_full, b_ultra)` of shape
    `(L, 16)` (when `m_col <= 16`) or `(L, 32)` (when `m_col > 16`).
    """
    l_tot, m_col = z3_full.shape
    d23_full = z3_full - z2_full
    svd_stride = max(1, l_tot // 65_536)

    if m_col <= 7:
        d12_full = z2_full - z1_full
        d23_top2 = _top_svd_cols(d23_full, 2, svd_stride, solver)
        d4_top2 = _top_svd_cols(d4_tot, 2, svd_stride, solver)
        proj_hf2 = solver.fit_target_manifold_projector(z3_hf[::svd_stride], 2)
        z3_hf_top2 = z3_hf.to(dtype=torch.float32) @ proj_hf2

        b_lite_t = torch.cat(
            [rms_balance_cols(z1_full), rms_balance_cols(d12_full), rms_balance_cols(d23_top2)],
            dim=-1,
        )
        del d12_full
        b_full_t = torch.cat(
            [rms_balance_cols(d23_full), rms_balance_cols(z4_hf), rms_balance_cols(d4_top2)],
            dim=-1,
        )
        b_ultra_t = torch.cat(
            [rms_balance_cols(d4_tot), rms_balance_cols(z4_lf), rms_balance_cols(z3_hf_top2)],
            dim=-1,
        )
    elif m_col <= 16:
        rem = max(1, 16 - m_col)
        z4_full = z3_full + d4_tot
        bb_scale = 0.12 / torch.clamp(
            torch.sqrt(torch.mean(z4_full * z4_full, dim=0, keepdim=True) + 1.0e-18), min=1.0e-7
        )
        zf_s = z4_full * bb_scale
        zh_s = z4_hf * bb_scale
        zm_s = z4_mf * bb_scale
        zl_s = z4_lf * bb_scale
        b_lite_t = torch.cat([zf_s, zh_s[:, :rem]], dim=-1)[:, :16]
        b_full_t = torch.cat([zh_s, zm_s[:, :rem]], dim=-1)[:, :16]
        b_ultra_t = torch.cat([zm_s, zl_s[:, :rem]], dim=-1)[:, :16]
        del z4_full, zf_s, zh_s, zm_s, zl_s
    else:
        rem = max(1, 32 - m_col)
        d23_top = _top_svd_cols(d23_full, rem, svd_stride, solver)
        d4_top = _top_svd_cols(d4_tot, rem, svd_stride, solver)
        z4_hf_top = _top_svd_cols(z4_hf, rem, svd_stride, solver)
        free_tensor_storage(z3_full)
        free_tensor_storage(z4_hf)
        empty_device_cache(z2_full.device)

        b_lite_t = torch.cat([rms_balance_cols(z2_full), rms_balance_cols(d23_top)], dim=-1)[:, :32]
        b_full_t = torch.cat([rms_balance_cols(d23_full), rms_balance_cols(d4_top)], dim=-1)[:, :32]
        b_ultra_t = torch.cat([rms_balance_cols(d4_tot), rms_balance_cols(z4_hf_top)], dim=-1)[
            :, :32
        ]

    del d23_full
    b_lite_np = b_lite_t.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    b_full_np = b_full_t.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    b_ultra_np = b_ultra_t.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    del b_lite_t, b_full_t, b_ultra_t
    gc.collect()
    return b_lite_np, b_full_np, b_ultra_np


@torch.no_grad()
def calibrate_3stage_selective_cascade(
    x_mono: NDArray[np.float32],
    targets_list: Sequence[NDArray[np.float32]],
    *,
    sample_rate_hz: int = 48_000,
    device: torch.device | str | None = "cpu",
    solver: HostDenseSolver | None = None,
    silent_stem_flags: Sequence[bool] | None = None,
    stem_noise_floors_dbfs: Sequence[tuple[float, float, float]] | None = None,
    cached_lags: Sequence[float | None] | None = None,
    cached_lti_irs: Sequence[NDArray[np.float32] | None] | None = None,
    cached_seg_stats: SegmentationWindowStats | None = None,
    cached_s1_proj: Tensor | None = None,
    out_calib_records: dict[str, Any] | None = None,
    on_stage_progress: Callable[[int, str], None] | None = None,
    on_step_metrics: Callable[[int, int, float, float, float, float, float, float], None]
    | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Calibrates the LTI Cabinet/Space + Multi-Stage Causal Selective State-Space + 3-Band
    Decoupled (`LF`/`MF`/`HF`) Cascade on `x_mono` `(L,)` and `targets_list` `[(L, 2), ...]`,
    returning the nested tier base blocks `(b_lite, b_full, b_ultra)` of shape `(L, 16)` (when
    `m_col <= 16`) or `(L, 32)` (when `m_col > 16`) in `float32`.
    """
    host_solver = solver if solver is not None else HostDenseSolver()
    dev = torch.device(device) if device is not None else torch.device("cpu")
    sr = float(sample_rate_hz)
    x_np = np.ascontiguousarray(np.asarray(x_mono, dtype=np.float32).ravel())
    x = torch.from_numpy(x_np).to(device=dev, dtype=torch.float32)
    l_tot = x.shape[0]

    active_cols: list[Tensor] = []
    active_stem_indices: list[int] = []
    stem_lags: list[float] = []
    all_stem_lags: list[float] = []
    all_stem_corrs: list[float] = []
    active_nf_dbfs: list[tuple[float, float, float]] = []

    for idx_y, y_arr in enumerate(targets_list):
        y_2d = np.asarray(y_arr, dtype=np.float32)
        y_1d_np = np.ascontiguousarray(
            (y_2d[:, 0] if y_2d.ndim == 2 else y_2d.ravel())[:l_tot], dtype=np.float32
        )
        is_silent = (
            silent_stem_flags[idx_y]
            if (silent_stem_flags is not None and idx_y < len(silent_stem_flags))
            else (float(np.mean(y_1d_np * y_1d_np)) <= 1.0e-12)
        )
        c_lag = (
            cached_lags[idx_y] if (cached_lags is not None and idx_y < len(cached_lags)) else None
        )
        if not is_silent and len(active_cols) < 32:
            y_t = torch.from_numpy(y_1d_np).to(device=dev, dtype=torch.float32)
            y_al, lag_s, corr_s = measure_and_shift_usb_jitter_tensor(
                x, y_t, max_jitter=32, cached_lag=c_lag
            )
            active_cols.append(y_al)
            active_stem_indices.append(idx_y)
            stem_lags.append(lag_s)
            all_stem_lags.append(lag_s)
            all_stem_corrs.append(corr_s)
            if stem_noise_floors_dbfs is not None and idx_y < len(stem_noise_floors_dbfs):
                active_nf_dbfs.append(stem_noise_floors_dbfs[idx_y])
            else:
                active_nf_dbfs.append((-60.0, -60.0, -60.0))
        else:
            all_stem_lags.append(0.0 if c_lag is None else float(c_lag))
            all_stem_corrs.append(1.0)

    if not active_cols:
        y0 = np.asarray(targets_list[0], dtype=np.float32)
        y0_1d = np.ascontiguousarray(
            (y0[:, 0] if y0.ndim == 2 else y0.ravel())[:l_tot], dtype=np.float32
        )
        active_cols.append(torch.from_numpy(y0_1d).to(device=dev, dtype=torch.float32))
        active_stem_indices.append(0)
        stem_lags.append(0.0)
        active_nf_dbfs.append((-60.0, -60.0, -60.0))

    n_real_active = len(active_cols)
    while len(active_cols) < 7:
        idx_src = len(active_cols) % n_real_active
        src_t = active_cols[idx_src]
        fc_aug = 180.0 * (1.8 ** len(active_cols))
        aug_t = src_t - fast_1pole_lpf(src_t, fc_aug, sr=sr)
        active_cols.append(aug_t)
        stem_lags.append(0.0)
        active_nf_dbfs.append(active_nf_dbfs[idx_src])

    m_col = len(active_cols)
    y_full = torch.stack(active_cols, dim=-1)
    del active_cols
    y_lf, y_mf, y_hf = split_bezout_3band(y_full, sr=sr)

    floor_lf_t = torch.tensor(
        [float(l_tot) * (10.0 ** (nf[0] / 10.0)) for nf in active_nf_dbfs],
        device=dev,
        dtype=torch.float32,
    )
    floor_mf_t = torch.tensor(
        [float(l_tot) * (10.0 ** (nf[1] / 10.0)) for nf in active_nf_dbfs],
        device=dev,
        dtype=torch.float32,
    )
    floor_hf_t = torch.tensor(
        [float(l_tot) * (10.0 ** (nf[2] / 10.0)) for nf in active_nf_dbfs],
        device=dev,
        dtype=torch.float32,
    )
    sig_lf_all = torch.sum(y_lf * y_lf, dim=0) + floor_lf_t
    sig_mf_all = torch.sum(y_mf * y_mf, dim=0) + floor_mf_t
    sig_hf_all = torch.sum(y_hf * y_hf, dim=0) + floor_hf_t

    def _report_cascade_step(
        step_idx: int,
        z_lf_cur: Tensor,
        z_mf_cur: Tensor,
        z_hf_cur: Tensor,
    ) -> Tensor:
        dlf = torch.sum((z_lf_cur - y_lf) ** 2, dim=0) / sig_lf_all
        dmf = torch.sum((z_mf_cur - y_mf) ** 2, dim=0) / sig_mf_all
        dhf = torch.sum((z_hf_cur - y_hf) ** 2, dim=0) / sig_hf_all
        res_w = torch.sqrt((dlf + dmf + dhf) / 3.0 + 1.0e-12)
        if on_step_metrics is not None:
            r_lf = dlf[:n_real_active]
            r_mf = dmf[:n_real_active]
            r_hf = dhf[:n_real_active]
            lf_db = float(torch.mean(10.0 * torch.log10(torch.clamp(r_lf, min=1.0e-18))).item())
            mf_db = float(torch.mean(10.0 * torch.log10(torch.clamp(r_mf, min=1.0e-18))).item())
            hf_db = float(torch.mean(10.0 * torch.log10(torch.clamp(r_hf, min=1.0e-18))).item())
            besr_u = float(
                torch.mean(
                    10.0 * torch.log10(torch.clamp((r_lf + r_mf + r_hf) / 3.0, min=1.0e-18))
                ).item()
            )
            besr_f = besr_u + (0.65 if step_idx < 4 else 0.35)
            besr_l = besr_u + (1.40 if step_idx < 4 else 0.95)
            on_step_metrics(step_idx, 4, besr_l, besr_f, besr_u, lf_db, mf_db, hf_db)
        return res_w

    # Stage 0: Chunked Acausal-Calibrated Causal Wiener LTI Stage
    z_lti_lf_list: list[Tensor] = []
    z_lti_mf_list: list[Tensor] = []
    z_lti_hf_list: list[Tensor] = []
    learned_irs_by_active: list[NDArray[np.float32]] = []
    min_lti_err = 1.0

    for c0 in range(0, m_col, 6):
        c1 = min(m_col, c0 + 6)
        ychunk = y_full[:, c0:c1]
        cached_ir_chunk: Tensor | None = None
        if cached_lti_irs is not None:
            ir_cols: list[NDArray[np.float32]] = []
            can_reuse = True
            for ac_i in range(c0, c1):
                if ac_i < n_real_active:
                    orig_idx = active_stem_indices[ac_i]
                    if orig_idx < len(cached_lti_irs) and cached_lti_irs[orig_idx] is not None:
                        ir_arr = cached_lti_irs[orig_idx]
                        assert ir_arr is not None
                        ir_cols.append(ir_arr)
                    else:
                        can_reuse = False
                        break
                else:
                    can_reuse = False
                    break
            if can_reuse and len(ir_cols) == (c1 - c0):
                cached_ir_chunk = torch.from_numpy(np.stack(ir_cols, axis=-1)).to(
                    device=dev, dtype=torch.float32
                )

        z_f, z_l, z_m, z_h, h_ir = solve_lti_cabinet_and_space_stage_with_ir(
            x, ychunk, max_ir_len=8192, sr=sr, cached_ir=cached_ir_chunk
        )
        rel_e = float(
            torch.amin(
                torch.sum((ychunk - z_f) ** 2, dim=0)
                / (torch.sum(ychunk * ychunk, dim=0) + 1.0e-12)
            ).item()
        )
        min_lti_err = min(min_lti_err, rel_e)
        z_lti_lf_list.append(z_l)
        z_lti_mf_list.append(z_m)
        z_lti_hf_list.append(z_h)
        h_ir_np = h_ir.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
        for j in range(c1 - c0):
            if c0 + j < n_real_active:
                learned_irs_by_active.append(h_ir_np[:, j].copy())
        del ychunk, z_f, h_ir

    if out_calib_records is not None:
        out_calib_records["all_stem_lags"] = all_stem_lags
        out_calib_records["all_stem_corrs"] = all_stem_corrs
        ir_map: dict[int, NDArray[np.float32]] = {}
        for ac_i, orig_idx in enumerate(active_stem_indices):
            if ac_i < len(learned_irs_by_active):
                ir_map[orig_idx] = learned_irs_by_active[ac_i]
        out_calib_records["lti_irs_by_stem_idx"] = ir_map

    z_lti_bands = (
        (
            torch.cat(z_lti_lf_list, dim=-1),
            torch.cat(z_lti_mf_list, dim=-1),
            torch.cat(z_lti_hf_list, dim=-1),
        )
        if min_lti_err < 0.05
        else None
    )
    del z_lti_lf_list, z_lti_mf_list, z_lti_hf_list
    gc.collect()

    if l_tot >= 48_000 and m_col <= 16 and z_lti_bands is None:
        z1_full, z2_full, z3_full, d4_tot, z4_lf, z4_mf, z4_hf, z3_hf = (
            run_4stage_orthogonal_fft_cascade(
                x,
                y_full,
                sr=sr,
                on_stage_progress=on_stage_progress,
                report_step=_report_cascade_step,
                solver=host_solver,
            )
        )
        del y_full, y_lf, y_mf, y_hf
    else:
        z1_full, z2_full, z3_full, d4_tot, z4_lf, z4_mf, z4_hf, z3_hf = (
            run_3stage_selective_tube_cascade(
                x,
                y_full,
                y_lf,
                y_mf,
                y_hf,
                sig_lf_all,
                sig_mf_all,
                sig_hf_all,
                z_lti_bands=z_lti_bands,
                sr=sr,
                on_stage_progress=on_stage_progress,
                report_step=_report_cascade_step,
                solver=host_solver,
                cached_seg_stats=cached_seg_stats,
                cached_s1_proj=cached_s1_proj,
                out_calib_records=out_calib_records,
            )
        )
        free_tensor_storage(y_lf)
        free_tensor_storage(y_mf)
        free_tensor_storage(y_hf)
        del y_full, y_lf, y_mf, y_hf, z_lti_bands
        empty_device_cache(dev)

    for s_idx in range(m_col):
        lag_s = stem_lags[s_idx]
        if abs(lag_s) >= 1.0e-4:
            for tensor_col in (z1_full, z2_full, z3_full, d4_tot, z4_lf, z4_mf, z4_hf, z3_hf):
                if tensor_col.numel() > 0:
                    tensor_col[:, s_idx] = shift_signal_fft_1d(tensor_col[:, s_idx], -lag_s)

    return _build_nested_tier_blocks(
        z1_full,
        z2_full,
        z3_full,
        d4_tot,
        z4_lf,
        z4_mf,
        z4_hf,
        z3_hf,
        solver=host_solver,
    )


__all__ = [
    "apply_ltas_hf_tilt_balance",
    "build_causal_bspline_window_weights",
    "build_fast_refinement_gates",
    "build_interstage_basis",
    "build_refinement_fuzz_basis",
    "build_stage1_fuzz_basis",
    "calibrate_3stage_selective_cascade",
    "compute_tier32_modulator_vectors",
    "fast_1pole_lpf",
    "fast_poly_adaa_batch",
    "fast_poly_inst_batch",
    "fast_wavefolder_inst_batch",
    "measure_and_shift_usb_jitter",
    "measure_and_shift_usb_jitter_tensor",
    "modulate_tier_blocks_with_controls",
    "parseval_inband_projectors",
    "project_fft_band",
    "project_orthonormal_basis",
    "rms_balance_cols",
    "shift_signal_fft_1d",
    "solve_3band_decoupled_window_stage",
    "solve_cholesky_ridge",
    "solve_fullrate_hf_residual_polish",
    "solve_lti_cabinet_and_space_stage",
    "solve_lti_cabinet_and_space_stage_with_ir",
    "solve_selective_stage_3band",
    "split_bezout_3band",
]
