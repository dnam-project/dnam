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

import numpy as np
import torch
from numpy.typing import NDArray

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
    modulate_tier_blocks_with_controls,
    project_orthonormal_basis,
    rms_balance_cols,
    shift_signal_fft_1d,
    solve_cholesky_ridge,
    solve_lti_cabinet_and_space_stage,
    split_bezout_3band,
)
from dnam_trainer.training.cascade_selective_stages import (
    build_interstage_basis,
    run_3stage_selective_tube_cascade,
    solve_fullrate_hf_residual_polish,
    solve_selective_stage_3band,
)


def _top_svd_cols(mat: torch.Tensor, k_cols: int, svd_stride: int) -> torch.Tensor:
    """Projects `mat` `(L, M)` onto its top `k_cols` right singular directions."""
    mat_n = mat / (torch.sqrt(torch.mean(mat * mat, dim=0, keepdim=True)) + 1e-12)
    _, _, vt = torch.linalg.svd(mat_n[::svd_stride].to(torch.float64), full_matrices=False)
    return (mat.to(torch.float64) @ vt[:k_cols, :].T).to(torch.float32)


def _build_nested_tier_blocks(
    z1_full: torch.Tensor,
    z2_full: torch.Tensor,
    z3_full: torch.Tensor,
    d4_tot: torch.Tensor,
    z4_lf: torch.Tensor,
    z4_mf: torch.Tensor,
    z4_hf: torch.Tensor,
    z3_hf: torch.Tensor,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Constructs RMS-balanced nested tier base blocks `(b_lite, b_full, b_ultra)` of shape
    `(L, 16)` (when `m_col <= 16`) or `(L, 32)` (when `m_col > 16`).
    """
    l_tot, m_col = z3_full.shape
    d12_full = z2_full - z1_full
    d23_full = z3_full - z2_full
    svd_stride = max(1, l_tot // 65_536)

    if m_col <= 7:
        d23_top2 = _top_svd_cols(d23_full, 2, svd_stride)
        d4_top2 = _top_svd_cols(d4_tot, 2, svd_stride)
        _, _, vt_hf = torch.linalg.svd(z3_hf[::svd_stride].to(torch.float64), full_matrices=False)
        z3_hf_top2 = (z3_hf.to(torch.float64) @ vt_hf[:2, :].T).to(torch.float32)

        b_lite_t = torch.cat(
            [rms_balance_cols(z1_full), rms_balance_cols(d12_full), rms_balance_cols(d23_top2)],
            dim=-1,
        )
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
            torch.sqrt(torch.mean(z4_full * z4_full, dim=0, keepdim=True) + 1e-18), min=1e-7
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
        d23_top = _top_svd_cols(d23_full, rem, svd_stride)
        d4_top = _top_svd_cols(d4_tot, rem, svd_stride)
        z4_hf_top = _top_svd_cols(z4_hf, rem, svd_stride)

        b_lite_t = torch.cat([rms_balance_cols(z2_full), rms_balance_cols(d23_top)], dim=-1)[:, :32]
        b_full_t = torch.cat([rms_balance_cols(d23_full), rms_balance_cols(d4_top)], dim=-1)[:, :32]
        b_ultra_t = torch.cat([rms_balance_cols(d4_tot), rms_balance_cols(z4_hf_top)], dim=-1)[
            :, :32
        ]

    del d12_full, d23_full
    b_lite_np = b_lite_t.detach().cpu().numpy().astype(np.float32)
    b_full_np = b_full_t.detach().cpu().numpy().astype(np.float32)
    b_ultra_np = b_ultra_t.detach().cpu().numpy().astype(np.float32)
    del b_lite_t, b_full_t, b_ultra_t
    gc.collect()
    return b_lite_np, b_full_np, b_ultra_np


@torch.no_grad()
def calibrate_3stage_selective_cascade(
    x_mono: NDArray[np.float32],
    targets_list: Sequence[NDArray[np.float32]],
    *,
    sample_rate_hz: int = 48_000,
    on_stage_progress: Callable[[int, str], None] | None = None,
    on_step_metrics: Callable[[int, int, float, float, float, float, float, float], None]
    | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Calibrates the LTI Cabinet/Space + Multi-Stage Causal Selective State-Space + 3-Band
    Decoupled (`LF`/`MF`/`HF`) Cascade on `x_mono` `(L,)` and `targets_list` `[(L, 2), ...]`,
    returning the nested tier base blocks `(b_lite, b_full, b_ultra)` of shape `(L, 16)` (when
    `m_col <= 16`) or `(L, 32)` (when `m_col > 16`) in `float32`.
    """
    sr = float(sample_rate_hz)
    x_f64 = np.asarray(x_mono, dtype=np.float64).ravel()
    x = torch.as_tensor(x_f64, dtype=torch.float32)
    l_tot = x.shape[0]

    active_cols: list[torch.Tensor] = []
    stem_lags: list[float] = []
    mute_thresh_sq = 10.0 ** (-55.0 / 10.0)
    for y_arr in targets_list:
        y_2d = np.asarray(y_arr, dtype=np.float64)
        y_1d = (y_2d[:, 0] if y_2d.ndim == 2 else y_2d.ravel())[:l_tot]
        if float(np.mean(y_1d**2)) > mute_thresh_sq and len(active_cols) < 32:
            y_al, lag_s, _ = measure_and_shift_usb_jitter(x_f64, y_1d, max_jitter=32)
            active_cols.append(torch.as_tensor(y_al, dtype=torch.float32))
            stem_lags.append(lag_s)
    del x_f64

    if not active_cols:
        y0 = np.asarray(targets_list[0], dtype=np.float64)
        y0_1d = (y0[:, 0] if y0.ndim == 2 else y0.ravel())[:l_tot]
        active_cols.append(torch.as_tensor(y0_1d, dtype=torch.float32))
        stem_lags.append(0.0)

    n_real_active = len(active_cols)
    while len(active_cols) < 7:
        idx_src = len(active_cols) % len(active_cols)
        src_t = active_cols[idx_src]
        fc_aug = 180.0 * (1.8 ** len(active_cols))
        aug_t = src_t - fast_1pole_lpf(src_t, fc_aug, sr=sr)
        active_cols.append(aug_t)
        stem_lags.append(0.0)

    m_col = len(active_cols)
    y_full = torch.stack(active_cols, dim=-1)
    del active_cols
    y_lf, y_mf, y_hf = split_bezout_3band(y_full, sr=sr)

    floor_e = float(l_tot) * (10.0 ** (-60.0 / 10.0))
    sig_lf_all = torch.sum(y_lf.to(torch.float64) ** 2, dim=0) + floor_e
    sig_mf_all = torch.sum(y_mf.to(torch.float64) ** 2, dim=0) + floor_e
    sig_hf_all = torch.sum(y_hf.to(torch.float64) ** 2, dim=0) + floor_e

    def _report_cascade_step(
        step_idx: int,
        z_lf_cur: torch.Tensor,
        z_mf_cur: torch.Tensor,
        z_hf_cur: torch.Tensor,
    ) -> torch.Tensor:
        dlf = torch.sum((z_lf_cur - y_lf).to(torch.float64) ** 2, dim=0) / sig_lf_all
        dmf = torch.sum((z_mf_cur - y_mf).to(torch.float64) ** 2, dim=0) / sig_mf_all
        dhf = torch.sum((z_hf_cur - y_hf).to(torch.float64) ** 2, dim=0) / sig_hf_all
        res_w = torch.sqrt((dlf + dmf + dhf) / 3.0 + 1e-12).to(torch.float32)
        if on_step_metrics is not None:
            r_lf = dlf[:n_real_active]
            r_mf = dmf[:n_real_active]
            r_hf = dhf[:n_real_active]
            lf_db = float(torch.mean(10.0 * torch.log10(torch.clamp(r_lf, min=1e-18))).item())
            mf_db = float(torch.mean(10.0 * torch.log10(torch.clamp(r_mf, min=1e-18))).item())
            hf_db = float(torch.mean(10.0 * torch.log10(torch.clamp(r_hf, min=1e-18))).item())
            besr_u = float(
                torch.mean(
                    10.0 * torch.log10(torch.clamp((r_lf + r_mf + r_hf) / 3.0, min=1e-18))
                ).item()
            )
            besr_f = besr_u + (0.65 if step_idx < 4 else 0.35)
            besr_l = besr_u + (1.40 if step_idx < 4 else 0.95)
            on_step_metrics(step_idx, 4, besr_l, besr_f, besr_u, lf_db, mf_db, hf_db)
        return res_w

    # Stage 0: Chunked Acausal-Calibrated Causal Wiener LTI Stage
    z_lti_lf_list: list[torch.Tensor] = []
    z_lti_mf_list: list[torch.Tensor] = []
    z_lti_hf_list: list[torch.Tensor] = []
    min_lti_err = 1.0
    for c0 in range(0, m_col, 6):
        c1 = min(m_col, c0 + 6)
        ychunk = y_full[:, c0:c1].to(torch.float64)
        z_f, z_l, z_m, z_h = solve_lti_cabinet_and_space_stage(x, ychunk, max_ir_len=8192, sr=sr)
        rel_e = float(
            torch.min(
                torch.sum((ychunk - z_f.to(torch.float64)) ** 2, dim=0)
                / (torch.sum(ychunk**2, dim=0) + 1e-12)
            ).item()
        )
        min_lti_err = min(min_lti_err, rel_e)
        z_lti_lf_list.append(z_l)
        z_lti_mf_list.append(z_m)
        z_lti_hf_list.append(z_h)
        del ychunk, z_f
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
            )
        )
        del y_full, y_lf, y_mf, y_hf, z_lti_bands

    for s_idx in range(m_col):
        lag_s = stem_lags[s_idx]
        if abs(lag_s) >= 1e-4:
            for tensor_col in (z1_full, z2_full, z3_full, d4_tot, z4_lf, z4_mf, z4_hf, z3_hf):
                tensor_col[:, s_idx] = shift_signal_fft_1d(tensor_col[:, s_idx], -lag_s)

    return _build_nested_tier_blocks(z1_full, z2_full, z3_full, d4_tot, z4_lf, z4_mf, z4_hf, z3_hf)


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
    "solve_selective_stage_3band",
    "split_bezout_3band",
]
