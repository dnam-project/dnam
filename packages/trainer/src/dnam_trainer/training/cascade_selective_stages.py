"""3-Stage Causal Selective State-Space Tube/Solid-State Cascade Solver (`V1A/Pre -> ToneStack ->
V1B/Driver -> PowerTube/OT -> 48 kHz Quadrature Hysteresis Polish`).
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F

from dnam_trainer.training.cascade_primitives import (
    build_causal_bspline_window_weights,
    build_fast_refinement_gates,
    fast_1pole_lpf,
    fast_poly_adaa_batch,
    fast_poly_inst_batch,
    fast_wavefolder_inst_batch,
    project_orthonormal_basis,
    solve_cholesky_ridge,
    solve_lti_cabinet_and_space_stage,
    split_bezout_3band,
)


def _subsampled_lpf_and_pre(
    u_stage: torch.Tensor,
    fc: float,
    stride: int,
    sr: float = 48_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    c_u = u_stage.shape[1]
    chunks_sub: list[torch.Tensor] = []
    chunks_pre: list[torch.Tensor] = []
    for c0 in range(0, c_u, 64):
        c1 = min(c_u, c0 + 64)
        uf = fast_1pole_lpf(u_stage[:, c0:c1], fc, sr=sr)
        chunks_sub.append(uf[::stride].clone())
        chunks_pre.append((uf[1::stride] - 0.85 * uf[:-1:stride]).clone())
        del uf
    return torch.cat(chunks_sub, dim=-1), torch.cat(chunks_pre, dim=-1)


def _eval_lti_commuted_projection(
    u_slice: torch.Tensor,
    wa_3d: torch.Tensor,
    sr: float = 48_000.0,
) -> torch.Tensor:
    l_s = u_slice.shape[0]
    z = (
        (u_slice @ wa_3d[0])
        + fast_1pole_lpf(u_slice @ wa_3d[1], 80.0, sr=sr)
        + fast_1pole_lpf(u_slice @ wa_3d[2], 400.0, sr=sr)
        + fast_1pole_lpf(u_slice @ wa_3d[3], 2000.0, sr=sr)
    )
    if l_s > 2:
        zp2 = u_slice @ wa_3d[4]
        zm2 = u_slice @ wa_3d[5]
        z[: l_s - 2] += zp2[2:]
        z[2:] += zm2[: l_s - 2]
    return z


def _solve_3band_weighted_synthesis(
    phi_s: torch.Tensor,
    y_lf_s: torch.Tensor,
    y_mf_s: torch.Tensor,
    y_hf_s: torch.Tensor,
    w_lf_s: float,
    w_mf_s: float,
    w_hf_s: float,
    *,
    stride: int,
    sr: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Splits `phi_s` into 3 Bezout bands and solves the band-weighted normal equation."""
    p_lf, p_mf, p_hf = split_bezout_3band(phi_s, sr=sr)
    plf_sub = p_lf[::stride].to(torch.float64)
    pmf_sub = p_mf[::stride].to(torch.float64)
    phf_sub = p_hf[::stride].to(torch.float64)
    gm = (
        w_lf_s * (plf_sub.T @ plf_sub)
        + w_mf_s * (pmf_sub.T @ pmf_sub)
        + w_hf_s * (phf_sub.T @ phf_sub)
    )
    cm = (
        w_lf_s * (plf_sub.T @ y_lf_s[::stride].to(torch.float64))
        + w_mf_s * (pmf_sub.T @ y_mf_s[::stride].to(torch.float64))
        + w_hf_s * (phf_sub.T @ y_hf_s[::stride].to(torch.float64))
    )
    del plf_sub, pmf_sub, phf_sub
    alpha_s = solve_cholesky_ridge(gm, cm)
    z_lf_s = (p_lf @ alpha_s).squeeze(-1)
    z_mf_s = (p_mf @ alpha_s).squeeze(-1)
    z_hf_s = (p_hf @ alpha_s).squeeze(-1)
    return z_lf_s + z_mf_s + z_hf_s, z_lf_s, z_mf_s, z_hf_s


@torch.no_grad()
def solve_selective_stage_3band(
    u_stage: torch.Tensor,
    y7_full: torch.Tensor,
    y7_lf: torch.Tensor,
    y7_mf: torch.Tensor,
    y7_hf: torch.Tensor,
    bspline_weights_win: torch.Tensor,
    bspline_weights_reg: torch.Tensor,
    ref_gates: Sequence[torch.Tensor],
    win_bounds: Sequence[tuple[int, int]],
    regimes: Sequence[tuple[int, int]],
    *,
    res_weights: torch.Tensor | None = None,
    z_lti_bands: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    free_u_stage: bool = False,
    sr: float = 48_000.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solves one physical tube/solid-state stage across `L_tot` samples with Pre-Emphasized
    3-Band (`LF`, `MF`, `HF`) Normal Equations in Step A, Step B, and Step C.
    """
    l_tot, c_u = u_stage.shape
    m_col = y7_full.shape[1]
    stride_a = 4 if l_tot >= 32_768 else 1
    stride_b = 2 if l_tot >= 16_384 else 1
    pad = 8

    y_full_f32 = y7_full.to(torch.float32)
    y_lf_f32 = y7_lf.to(torch.float32)
    y_mf_f32 = y7_mf.to(torch.float32)
    y_hf_f32 = y7_hf.to(torch.float32)

    if m_col > 14:
        svd_st = max(1, l_tot // 32_768)
        ylf_std = torch.sqrt(torch.mean(y_lf_f32**2, dim=0, keepdim=True) + 1e-12)
        ymf_std = torch.sqrt(torch.mean(y_mf_f32**2, dim=0, keepdim=True) + 1e-12)
        yhf_std = torch.sqrt(torch.mean(y_hf_f32**2, dim=0, keepdim=True) + 1e-12)
        y_bal = (
            y_lf_f32[::svd_st] / ylf_std
            + y_mf_f32[::svd_st] / ymf_std
            + y_hf_f32[::svd_st] / yhf_std
        ).to(torch.float64)
        _, _, vt_unw = torch.linalg.svd(y_bal, full_matrices=False)
        if res_weights is not None:
            w_row = res_weights.to(torch.float64).view(1, -1)
            _, _, vt_res = torch.linalg.svd(y_bal * w_row, full_matrices=False)
            v_comb = torch.cat([vt_unw[:7, :].T, vt_res[:7, :].T], dim=-1)
            q_mat, _ = torch.linalg.qr(v_comb)
            proj_14 = q_mat[:, :14].to(torch.float32)
            del vt_res, v_comb, q_mat
        else:
            proj_14 = vt_unw[:14, :].T.to(torch.float32)
        y_std = torch.sqrt(torch.mean(y_full_f32**2, dim=0, keepdim=True) + 1e-12)
        ya_full = (y_full_f32 / y_std) @ proj_14
        ya_hf = (y_hf_f32 / yhf_std) @ proj_14
        del y_bal, vt_unw
    else:
        ya_full = y_full_f32
        ya_hf = y_hf_f32

    k_a = ya_full.shape[1]
    u_sub0 = u_stage[::stride_a]
    n_pre_a = (l_tot - 1 + stride_a - 1) // stride_a

    def _chunk_shifted_sub(qs: int, qe: int, shift: int) -> torch.Tensor:
        if shift == 0:
            return u_stage[qs * stride_a : qe * stride_a : stride_a]
        if shift > 0:
            sub = u_stage[qs * stride_a + shift : qe * stride_a + shift : stride_a]
            if sub.shape[0] < (qe - qs):
                sub = F.pad(sub.unsqueeze(0), (0, 0, 0, (qe - qs) - sub.shape[0])).squeeze(0)
            return sub[: qe - qs]
        sh_neg = -shift
        start_idx = qs * stride_a - sh_neg
        end_idx = qe * stride_a - sh_neg
        if start_idx >= 0:
            return u_stage[start_idx:end_idx:stride_a]
        pad_rows = (-start_idx + stride_a - 1) // stride_a
        rem_start = start_idx + pad_rows * stride_a
        sliced = u_stage[rem_start : max(rem_start, end_idx) : stride_a]
        padded = F.pad(sliced.unsqueeze(0), (0, 0, pad_rows, 0)).squeeze(0)
        return padded[: qe - qs]

    def _chunk_shifted_pre(qs: int, qe: int, shift: int) -> torch.Tensor:
        if shift == 0:
            s1 = u_stage[qs * stride_a + 1 : qe * stride_a + 1 : stride_a]
            s0 = u_stage[qs * stride_a : min(l_tot - 1, qe * stride_a) : stride_a]
            m_len = min(s1.shape[0], s0.shape[0], qe - qs)
            if m_len == qe - qs:
                return s1[:m_len] - 0.85 * s0[:m_len]
            pre = torch.zeros((qe - qs, c_u), dtype=u_stage.dtype, device=u_stage.device)
            pre[:m_len] = s1[:m_len] - 0.85 * s0[:m_len]
            return pre
        if shift > 0:
            s1 = u_stage[qs * stride_a + shift + 1 : qe * stride_a + shift + 1 : stride_a]
            s0 = u_stage[qs * stride_a + shift : min(l_tot - 1, qe * stride_a + shift) : stride_a]
            m_len = min(s1.shape[0], s0.shape[0], qe - qs)
            if m_len == qe - qs:
                return s1[:m_len] - 0.85 * s0[:m_len]
            pre = torch.zeros((qe - qs, c_u), dtype=u_stage.dtype, device=u_stage.device)
            if m_len > 0:
                pre[:m_len] = s1[:m_len] - 0.85 * s0[:m_len]
            return pre
        sh_neg = -shift
        start_0 = qs * stride_a - sh_neg
        end_0 = min(l_tot - 1 - sh_neg, qe * stride_a - sh_neg)
        if start_0 >= 0:
            s1 = u_stage[start_0 + 1 : end_0 + 1 : stride_a]
            s0 = u_stage[start_0:end_0:stride_a]
            m_len = min(s1.shape[0], s0.shape[0], qe - qs)
            if m_len == qe - qs:
                return s1[:m_len] - 0.85 * s0[:m_len]
            pre = torch.zeros((qe - qs, c_u), dtype=u_stage.dtype, device=u_stage.device)
            if m_len > 0:
                pre[:m_len] = s1[:m_len] - 0.85 * s0[:m_len]
            return pre
        pad_rows = (-start_0 + stride_a - 1) // stride_a
        rem_0 = start_0 + pad_rows * stride_a
        s1 = u_stage[rem_0 + 1 : max(rem_0 + 1, end_0 + 1) : stride_a]
        s0 = u_stage[rem_0 : max(rem_0, end_0) : stride_a]
        m_len = min(s1.shape[0], s0.shape[0], max(0, (qe - qs) - pad_rows))
        pre = torch.zeros((qe - qs, c_u), dtype=u_stage.dtype, device=u_stage.device)
        if m_len > 0:
            pre[pad_rows : pad_rows + m_len] = s1[:m_len] - 0.85 * s0[:m_len]
        return pre

    u_sub_l80, u_pre_l80 = _subsampled_lpf_and_pre(u_stage, 80.0, stride_a, sr=sr)
    u_sub_l400, u_pre_l400 = _subsampled_lpf_and_pre(u_stage, 400.0, stride_a, sr=sr)
    u_sub_l2000, u_pre_l2000 = _subsampled_lpf_and_pre(u_stage, 2000.0, stride_a, sr=sr)

    ya_hp600 = ya_full - fast_1pole_lpf(ya_full, 600.0, sr=sr)
    ya_pad = F.pad(ya_full.unsqueeze(0), (0, 0, pad, pad)).squeeze(0)
    ya_hp600_pad = F.pad(ya_hp600.unsqueeze(0), (0, 0, pad, pad)).squeeze(0)
    ya_hp2500_pad = F.pad(ya_hf.unsqueeze(0), (0, 0, pad, pad)).squeeze(0)
    del ya_full, ya_hf, ya_hp600

    tgt_a_sub: list[torch.Tensor] = []
    tgt_a_pre: list[torch.Tensor] = []
    for src_pad, shifts_list in (
        (ya_pad, (-3, -1, 0, 1, 3)),
        (ya_hp600_pad, (-1, 0, 1)),
        (ya_hp2500_pad, (0,)),
    ):
        for sh in shifts_list:
            tgt_a_sub.append(src_pad[pad - sh : pad - sh + l_tot : stride_a])
            tgt_a_pre.append(
                (
                    src_pad[pad - sh + 1 : pad - sh + l_tot : stride_a]
                    - 0.85 * src_pad[pad - sh : pad - sh + l_tot - 1 : stride_a]
                )[:n_pre_a]
            )
    y_multi_a_sub = torch.cat(tgt_a_sub, dim=-1)
    y_multi_a_pre = torch.cat(tgt_a_pre, dim=-1)
    del ya_pad, ya_hp600_pad, ya_hp2500_pad, tgt_a_sub, tgt_a_pre

    n_a_tgt = 9 * k_a
    p_a = 6 * c_u
    g_a_gl = torch.zeros((p_a, p_a), dtype=torch.float64)
    c_a_gl = torch.zeros((p_a, n_a_tgt), dtype=torch.float64)
    ov_a = 512 if l_tot >= 32_768 else 0
    burn_a = 2048 if l_tot >= 32_768 else 0
    z_a_loc = torch.zeros((l_tot, n_a_tgt), dtype=torch.float32)

    for r_idx, (r0, r1) in enumerate(regimes):
        sr0 = 0 if r_idx == 0 else max(0, r0 - ov_a)
        er1 = l_tot if r_idx == len(regimes) - 1 else min(l_tot, r1 + ov_a)
        q0 = sr0 // stride_a
        q1 = min(y_multi_a_sub.shape[0], er1 // stride_a)
        q1_pre = min(n_pre_a, q1)

        ga = torch.zeros((p_a, p_a), dtype=torch.float64)
        ca = torch.zeros((p_a, n_a_tgt), dtype=torch.float64)
        for qs in range(q0, q1, 4096):
            qe = min(q1, qs + 4096)
            pa_s = torch.cat(
                [
                    _chunk_shifted_sub(qs, qe, 0),
                    u_sub_l80[qs:qe],
                    u_sub_l400[qs:qe],
                    u_sub_l2000[qs:qe],
                    _chunk_shifted_sub(qs, qe, 2),
                    _chunk_shifted_sub(qs, qe, -2),
                ],
                dim=-1,
            ).to(torch.float64)
            ya_s = y_multi_a_sub[qs:qe].to(torch.float64)
            ga += pa_s.T @ pa_s
            ca += pa_s.T @ ya_s
            del pa_s, ya_s

        for qs in range(q0, q1_pre, 4096):
            qe = min(q1_pre, qs + 4096)
            pa_p = torch.cat(
                [
                    _chunk_shifted_pre(qs, qe, 0),
                    u_pre_l80[qs:qe],
                    u_pre_l400[qs:qe],
                    u_pre_l2000[qs:qe],
                    _chunk_shifted_pre(qs, qe, 2),
                    _chunk_shifted_pre(qs, qe, -2),
                ],
                dim=-1,
            ).to(torch.float64)
            ya_p = y_multi_a_pre[qs:qe].to(torch.float64)
            ga += 0.5 * (pa_p.T @ pa_p)
            ca += 0.5 * (pa_p.T @ ya_p)
            del pa_p, ya_p

        g_a_gl += ga
        c_a_gl += ca
        wa = solve_cholesky_ridge(ga, ca, ridge_scale=5e-6, min_ridge=1e-9)
        del ga

        s_eval = max(0, sr0 - burn_a)
        b_off = sr0 - s_eval
        z_r_slice = _eval_lti_commuted_projection(
            u_stage[s_eval:er1], wa.view(6, c_u, n_a_tgt), sr=sr
        )[b_off:]
        z_a_loc[sr0:er1] += bspline_weights_reg[sr0:er1, r_idx : r_idx + 1] * z_r_slice
        del z_r_slice, wa

    wa_gl = solve_cholesky_ridge(g_a_gl, c_a_gl, ridge_scale=2e-6, min_ridge=1e-9)
    del u_sub0
    del u_sub_l80, u_pre_l80, u_sub_l400, u_pre_l400, u_sub_l2000, u_pre_l2000
    del y_multi_a_sub, y_multi_a_pre, g_a_gl

    z_a_gl = _eval_lti_commuted_projection(u_stage, wa_gl.view(6, c_u, n_a_tgt), sr=sr)
    del wa_gl
    if free_u_stage:
        u_stage.resize_(0)
        gc.collect()

    z_a = torch.cat([z_a_loc, z_a_gl], dim=-1)
    del z_a_loc, z_a_gl

    core_idx: list[int] = []
    for r_i in range(2):
        base = r_i * n_a_tgt
        core_idx.extend(range(base + 2 * k_a, base + 3 * k_a))
        core_idx.extend(range(base + 6 * k_a, base + 7 * k_a))
        core_idx.extend(range(base + 8 * k_a, base + 9 * k_a))
    za_core_raw = z_a[:, core_idx]
    max_core = 48 if m_col > 14 else 42
    if za_core_raw.shape[1] > max_core:
        za_core = project_orthonormal_basis(za_core_raw, max_core)
    else:
        za_core = za_core_raw.clone()
    del za_core_raw
    if m_col > 14 and z_a.shape[1] > 180:
        z_a = project_orthonormal_basis(z_a, 180)
    c_core = za_core.shape[1]

    post_lpfs = (45.0, 150.0, 500.0, 1600.0, 4500.0)
    delays = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8)
    za_lpf_list = [fast_1pole_lpf(za_core, fc, sr=sr) for fc in post_lpfs]
    za_core_pad = F.pad(za_core.unsqueeze(0), (0, 0, pad, pad)).squeeze(0)
    del za_core

    n_b_tgt = 3 * m_col
    y21_bands = torch.cat([y_lf_f32, y_mf_f32, y_hf_f32], dim=-1)

    n_z_a = z_a.shape[1]
    n_b = n_z_a + len(post_lpfs) * c_core + len(delays) * c_core
    g_b_gl = torch.zeros((n_b, n_b), dtype=torch.float64)
    c_b_gl = torch.zeros((n_b, n_b_tgt), dtype=torch.float64)
    n_wins = len(win_bounds)
    overlap = 256 if l_tot >= 32_768 else 0
    z_spline_21 = torch.zeros((l_tot, n_b_tgt), dtype=torch.float32)

    for w_idx, (k0, k1) in enumerate(win_bounds):
        s_w = 0 if w_idx == 0 else max(0, k0 - overlap)
        e_w = l_tot if w_idx == n_wins - 1 else min(l_tot, k1 + overlap)
        b_win_list = [z_a[s_w:e_w]] + [zl[s_w:e_w] for zl in za_lpf_list]
        for d in delays:
            b_win_list.append(za_core_pad[pad + s_w - d : pad + e_w - d])
        pk_full = torch.cat(b_win_list, dim=-1)
        del b_win_list

        pk_sub = pk_full[::stride_b].to(torch.float64)
        yk_sub = y21_bands[s_w:e_w:stride_b].to(torch.float64)
        gw = pk_sub.T @ pk_sub
        cw = pk_sub.T @ yk_sub
        del pk_sub, yk_sub

        pk_pre = (pk_full[1::stride_b] - 0.85 * pk_full[:-1:stride_b]).to(torch.float64)
        yk_pre = (
            y21_bands[s_w + 1 : e_w : stride_b] - 0.85 * y21_bands[s_w : e_w - 1 : stride_b]
        ).to(torch.float64)
        gw += 0.5 * (pk_pre.T @ pk_pre)
        cw += 0.5 * (pk_pre.T @ yk_pre)
        del pk_pre, yk_pre

        g_b_gl += gw
        c_b_gl += cw
        ww = solve_cholesky_ridge(gw, cw, ridge_scale=1e-5, min_ridge=1e-9)

        bw_slice = bspline_weights_win[s_w:e_w, w_idx : w_idx + 1]
        z_spline_21[s_w:e_w] += bw_slice * (pk_full @ ww)
        del pk_full, gw, ww

    del y21_bands
    w_b_gl = solve_cholesky_ridge(g_b_gl, c_b_gl, ridge_scale=2e-6, min_ridge=1e-9)

    idx = 0
    z_glob_21 = z_a @ w_b_gl[idx : idx + n_z_a]
    idx += n_z_a
    del z_a
    for zl in za_lpf_list:
        z_glob_21 = z_glob_21 + (zl @ w_b_gl[idx : idx + c_core])
        idx += c_core
    del za_lpf_list
    for d in delays:
        z_glob_21 = z_glob_21 + (
            za_core_pad[pad - d : pad - d + l_tot] @ w_b_gl[idx : idx + c_core]
        )
        idx += c_core
    del za_core_pad, g_b_gl, c_b_gl, w_b_gl

    # Step C: Per-stem Decoupled 3-Band (`LF`, `MF`, `HF`) Refinement + `bESR` Synthesis
    z_lf_cols: list[torch.Tensor] = []
    z_mf_cols: list[torch.Tensor] = []
    z_hf_cols: list[torch.Tensor] = []
    z_tot_cols: list[torch.Tensor] = []
    floor_e = float(l_tot) * (10.0 ** (-60.0 / 10.0))

    for s_idx in range(m_col):
        b_idx = [s_idx, m_col + s_idx, 2 * m_col + s_idx]
        base_6 = torch.cat([z_spline_21[:, b_idx], z_glob_21[:, b_idx]], dim=-1)
        phi_parts = [base_6 * g.unsqueeze(-1) for g in ref_gates]
        if z_lti_bands is not None:
            phi_parts.append(
                torch.stack(
                    [
                        z_lti_bands[0][:, s_idx],
                        z_lti_bands[1][:, s_idx],
                        z_lti_bands[2][:, s_idx],
                    ],
                    dim=-1,
                )
            )
        phi_ref = torch.cat(phi_parts, dim=-1)
        del phi_parts, base_6
        y_s_3b = torch.stack([y_lf_f32[:, s_idx], y_mf_f32[:, s_idx], y_hf_f32[:, s_idx]], dim=-1)

        phi_sub = phi_ref[::stride_b].to(torch.float64)
        y_sub = y_s_3b[::stride_b].to(torch.float64)
        g_sub = phi_sub.T @ phi_sub
        c_sub = phi_sub.T @ y_sub
        del phi_sub, y_sub

        phi_pre = (phi_ref[1::stride_b] - 0.85 * phi_ref[:-1:stride_b]).to(torch.float64)
        y_pre = (y_s_3b[1::stride_b] - 0.85 * y_s_3b[:-1:stride_b]).to(torch.float64)
        g_pre = phi_pre.T @ phi_pre
        c_pre = phi_pre.T @ y_pre
        del phi_pre, y_pre, y_s_3b

        w_lf = solve_cholesky_ridge(g_sub + 0.05 * g_pre, c_sub[:, 0:1] + 0.05 * c_pre[:, 0:1])
        w_mf = solve_cholesky_ridge(g_sub + 0.50 * g_pre, c_sub[:, 1:2] + 0.50 * c_pre[:, 1:2])
        w_hf = solve_cholesky_ridge(
            0.02 * g_sub + 1.00 * g_pre, 0.02 * c_sub[:, 2:3] + 1.00 * c_pre[:, 2:3]
        )

        wr = torch.cat([w_lf, w_mf, w_hf], dim=-1)
        c3_s = phi_ref @ wr
        del phi_ref, g_sub, g_pre, c_sub, c_pre, wr

        c_l120 = fast_1pole_lpf(c3_s, 120.0, sr=sr)
        c_l250 = fast_1pole_lpf(c3_s, 250.0, sr=sr)
        c_l250_2 = fast_1pole_lpf(c_l250, 250.0, sr=sr)
        c_l700 = fast_1pole_lpf(c3_s, 700.0, sr=sr)
        c_l1600 = fast_1pole_lpf(c3_s, 1600.0, sr=sr)
        c_l2500 = fast_1pole_lpf(c3_s, 2500.0, sr=sr)
        c_l2500_2 = fast_1pole_lpf(c_l2500, 2500.0, sr=sr)
        c_l5500 = fast_1pole_lpf(c3_s, 5500.0, sr=sr)

        m_gated = [c3_s * g.unsqueeze(-1) for g in ref_gates[:5]]
        m_s = torch.cat(
            [
                c3_s,
                c_l120,
                c_l250,
                c_l250_2,
                c_l700,
                c_l1600,
                c_l2500,
                c_l2500_2,
                c_l5500,
                *m_gated,
            ],
            dim=-1,
        )
        del c3_s, c_l120, c_l250, c_l250_2, c_l700, c_l1600, c_l2500, c_l2500_2, c_l5500, m_gated

        ylf_s = y_lf_f32[:, s_idx : s_idx + 1]
        ymf_s = y_mf_f32[:, s_idx : s_idx + 1]
        yhf_s = y_hf_f32[:, s_idx : s_idx + 1]
        w_lf_s = 1.0 / (float(torch.sum(ylf_s.to(torch.float64) ** 2).item()) + floor_e)
        w_mf_s = 1.0 / (float(torch.sum(ymf_s.to(torch.float64) ** 2).item()) + floor_e)
        w_hf_s = 1.0 / (float(torch.sum(yhf_s.to(torch.float64) ** 2).item()) + floor_e)
        z_tot_s, z_lf_s, z_mf_s, z_hf_s = _solve_3band_weighted_synthesis(
            m_s, ylf_s, ymf_s, yhf_s, w_lf_s, w_mf_s, w_hf_s, stride=stride_b, sr=sr
        )
        del m_s
        z_lf_cols.append(z_lf_s)
        z_mf_cols.append(z_mf_s)
        z_hf_cols.append(z_hf_s)
        z_tot_cols.append(z_tot_s)

    del z_spline_21, z_glob_21, y_full_f32, y_lf_f32, y_mf_f32, y_hf_f32
    return (
        torch.stack(z_tot_cols, dim=-1),
        torch.stack(z_lf_cols, dim=-1),
        torch.stack(z_mf_cols, dim=-1),
        torch.stack(z_hf_cols, dim=-1),
    )


def build_interstage_basis(
    z_prev: torch.Tensor,
    z_earlier: torch.Tensor | None,
    cutoffs: Sequence[float],
    bias_fc: float,
    env_fast_fc: float,
    env_slow_fc: float,
    drives: Sequence[float],
    *,
    res_weights: torch.Tensor | None = None,
    use_power_sag: bool = False,
    sr: float = 48_000.0,
) -> torch.Tensor:
    """Vectorized interstage nonlinear basis builder with QR-combined global + residual-weighted
    SVD manifold projection (`< 1.0 GB` peak buffer).
    """
    z_std = torch.sqrt(torch.mean(z_prev**2, dim=0, keepdim=True) + 1e-12)
    z_norm = z_prev / z_std

    if z_norm.shape[1] > 8:
        l_seq = z_norm.shape[0]
        svd_st = max(1, l_seq // 32_768)
        sub_unw = z_norm[::svd_st].to(torch.float64)
        _, _, vt_unw = torch.linalg.svd(sub_unw, full_matrices=False)
        n_lin = min(14, z_norm.shape[1])
        n_nl = 7 if z_norm.shape[1] <= 14 else min(8, z_norm.shape[1])
        if res_weights is not None:
            w_row = res_weights.to(torch.float64).view(1, -1)
            _, _, vt_res = torch.linalg.svd(sub_unw * w_row, full_matrices=False)
            v_comb = torch.cat([vt_unw[:4, :].T, vt_res[:6, :].T, vt_unw[4:8, :].T], dim=-1)
            q_mat, _ = torch.linalg.qr(v_comb)
            proj_lin = q_mat[:, :n_lin].to(torch.float32)
            proj_nl = q_mat[:, :n_nl].to(torch.float32)
            del vt_res, v_comb, q_mat
        else:
            proj_lin = vt_unw[:n_lin, :].T.to(torch.float32)
            proj_nl = vt_unw[:n_nl, :].T.to(torch.float32)
        z_lin = z_norm @ proj_lin
        z_nl = z_norm @ proj_nl
        if z_earlier is not None:
            ze_std = torch.sqrt(torch.mean(z_earlier**2, dim=0, keepdim=True) + 1e-12)
            z_earlier_proj = (z_earlier / ze_std) @ proj_lin
        else:
            z_earlier_proj = None
        del sub_unw, vt_unw, proj_lin, proj_nl
    elif z_norm.shape[1] > 7:
        l_seq = z_norm.shape[0]
        svd_st = max(1, l_seq // 32_768)
        _, _, vt_nl = torch.linalg.svd(z_norm[::svd_st].to(torch.float64), full_matrices=False)
        z_lin = z_prev
        z_nl = (z_norm.to(torch.float64) @ vt_nl[:7, :].T).to(torch.float32)
        z_earlier_proj = z_earlier
        del vt_nl
    else:
        z_lin = z_prev
        z_nl = z_norm
        z_earlier_proj = z_earlier
    del z_norm

    z_hp600 = z_lin - fast_1pole_lpf(z_lin, 600.0, sr=sr)
    z_hp2500 = z_lin - fast_1pole_lpf(z_lin, 2500.0, sr=sr)
    z_nl_hp600 = z_nl - fast_1pole_lpf(z_nl, 600.0, sr=sr)
    z_nl_hp2500 = z_nl - fast_1pole_lpf(z_nl, 2500.0, sr=sr)

    env_s = fast_1pole_lpf(torch.abs(z_nl), env_fast_fc, sr=sr)
    env_slow = fast_1pole_lpf(z_nl * z_nl if use_power_sag else torch.abs(z_nl), env_slow_fc, sr=sr)
    sh_list = [z_nl - fast_1pole_lpf(z_nl, fc, sr=sr) for fc in cutoffs]
    sh_bias = z_nl - fast_1pole_lpf(z_nl, bias_fc, sr=sr)

    c_bias = 0.30 if use_power_sag else 0.35
    c_slow = 0.35 if use_power_sag else 0.45
    c_600 = 0.35 if use_power_sag else 0.40
    c_2k5 = 0.40 if use_power_sag else 0.50

    adaa_bands = torch.cat(
        [*sh_list, sh_bias + c_bias * env_s, sh_bias - c_bias * env_s, sh_bias - c_slow * env_slow],
        dim=-1,
    )
    adaa_out = fast_poly_adaa_batch(adaa_bands, drives, sr=sr)
    del adaa_bands
    if z_prev.shape[1] > 14 and adaa_out.shape[1] > 320:
        adaa_out = project_orthonormal_basis(adaa_out, 320)

    inst_bands = torch.cat(
        [
            *sh_list,
            sh_bias + c_bias * env_s + c_600 * z_nl_hp600,
            sh_bias - c_bias * env_s + c_600 * z_nl_hp600,
            sh_bias + c_2k5 * z_nl_hp2500,
        ],
        dim=-1,
    )
    del sh_list, sh_bias, env_s, env_slow, z_nl_hp600, z_nl_hp2500
    inst_out = fast_poly_inst_batch(inst_bands, drives, sr=sr)
    del inst_bands
    if z_prev.shape[1] > 14 and inst_out.shape[1] > 140:
        inst_out = project_orthonormal_basis(inst_out, 140)

    parts = [z_lin, z_hp600, z_hp2500]
    if z_earlier_proj is not None:
        parts.insert(0, z_earlier_proj)
    parts.append(adaa_out)
    parts.append(inst_out)
    return torch.cat(parts, dim=-1)


@torch.no_grad()
def solve_fullrate_hf_residual_polish(
    z3_full: torch.Tensor,
    z2_full: torch.Tensor,
    r3_lf: torch.Tensor,
    r3_mf: torch.Tensor,
    r3_hf: torch.Tensor,
    sig_lf: torch.Tensor,
    sig_mf: torch.Tensor,
    sig_hf: torch.Tensor,
    bspline_weights_win: torch.Tensor,
    ref_gates: Sequence[torch.Tensor],
    win_bounds: Sequence[tuple[int, int]],
    *,
    z1_full: torch.Tensor | None = None,
    sr: float = 48_000.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """3-Band Quadrature Hysteresis & Wavefolder B-Spline Polish + Step-C Post-Synthesis (`CL-18`
    `float64`).
    """
    l_tot, m_col = z3_full.shape
    stride_s4 = 2 if l_tot >= 16_384 else 1

    r3_lf_f32 = r3_lf.to(torch.float32)
    r3_mf_f32 = r3_mf.to(torch.float32)
    r3_hf_f32 = r3_hf.to(torch.float32)

    dlf3 = torch.sum(r3_lf_f32.to(torch.float64) ** 2, dim=0) / sig_lf
    dmf3 = torch.sum(r3_mf_f32.to(torch.float64) ** 2, dim=0) / sig_mf
    dhf3 = torch.sum(r3_hf_f32.to(torch.float64) ** 2, dim=0) / sig_hf
    res_weight = torch.sqrt((dlf3 + dmf3 + dhf3) / 3.0 + 1e-12).to(torch.float32).unsqueeze(0)

    z1_src = z1_full if z1_full is not None else z2_full
    z3_std = torch.sqrt(torch.mean(z3_full**2, dim=0, keepdim=True) + 1e-12)
    z2_std = torch.sqrt(torch.mean(z2_full**2, dim=0, keepdim=True) + 1e-12)
    z1_std = torch.sqrt(torch.mean(z1_src**2, dim=0, keepdim=True) + 1e-12)
    if m_col > 8:
        svd_st = max(1, l_tot // 32_768)
        z3_w = ((z3_full / z3_std) * res_weight)[::svd_st].to(torch.float64)
        _, _, vt3 = torch.linalg.svd(z3_w, full_matrices=False)
        proj_8 = vt3[:8, :].T.to(torch.float32)
        z3_k = (z3_full / z3_std) @ proj_8
        z2_k = (z2_full / z2_std) @ proj_8
        z1_k = (z1_src / z1_std) @ proj_8
        del z3_w, vt3, proj_8
    else:
        z3_k = z3_full / z3_std
        z2_k = z2_full / z2_std
        z1_k = z1_src / z1_std

    l_80 = fast_1pole_lpf(z3_k, 80.0, sr=sr)
    l_250 = fast_1pole_lpf(z3_k, 250.0, sr=sr)
    h_250 = z3_k - l_250
    h_900 = z3_k - fast_1pole_lpf(z3_k, 900.0, sr=sr)
    h_2500 = z3_k - fast_1pole_lpf(z3_k, 2500.0, sr=sr)
    h_6000 = z3_k - fast_1pole_lpf(z3_k, 6000.0, sr=sr)
    h_s2_1200 = z2_k - fast_1pole_lpf(z2_k, 1200.0, sr=sr)
    h_s1_1200 = z1_k - fast_1pole_lpf(z1_k, 1200.0, sr=sr)
    del z2_k, z1_k

    e_slow = fast_1pole_lpf(torch.abs(z3_k), 12.0, sr=sr)
    e_low = fast_1pole_lpf(torch.abs(z3_k), 40.0, sr=sr)
    e_fast = fast_1pole_lpf(torch.abs(z3_k), 160.0, sr=sr)

    q_low = (l_250 - l_80) / (
        torch.sqrt(torch.mean((l_250 - l_80) ** 2, dim=0, keepdim=True)) + 1e-12
    )
    q_mid = (h_250 - h_2500) / (
        torch.sqrt(torch.mean((h_250 - h_2500) ** 2, dim=0, keepdim=True)) + 1e-12
    )
    q_hi = h_2500 / (torch.sqrt(torch.mean(h_2500**2, dim=0, keepdim=True)) + 1e-12)

    wf_core = fast_wavefolder_inst_batch(
        torch.cat([z3_k, l_250, h_250, h_2500], dim=-1),
        (1.2, 3.0, 7.5, 20.0, 55.0),
        sr=sr,
    )
    q_rep = torch.cat([q_low, q_low, q_mid, q_hi], dim=-1).repeat(1, 10)
    wf_hyst = wf_core * q_rep
    del q_low, q_mid, q_hi, q_rep
    if wf_hyst.shape[1] > 200:
        wf_hyst = project_orthonormal_basis(wf_hyst, 200)

    nl_in = torch.cat(
        [
            z3_k,
            l_80,
            l_250,
            h_250,
            h_900,
            h_2500,
            h_6000,
            h_s2_1200,
            h_s1_1200,
            z3_k * (1.0 - 0.35 * e_slow),
            h_250 * (1.0 + 0.45 * e_low),
            h_900 * (1.0 - 0.40 * e_fast),
            h_2500 * (1.0 + 0.50 * e_low),
        ],
        dim=-1,
    )
    del l_80, l_250, h_250, h_s2_1200, h_s1_1200, e_slow, e_low, e_fast
    nl_inst = fast_poly_inst_batch(nl_in, (0.8, 2.5, 8.0, 25.0), sr=sr)
    del nl_in
    if nl_inst.shape[1] > 360:
        nl_inst = project_orthonormal_basis(nl_inst, 360)

    pad = 16
    h_cat = torch.cat([z3_k, h_900, h_2500, h_6000], dim=-1)
    del z3_k, h_900, h_2500, h_6000
    h_pad = F.pad(h_cat.unsqueeze(0), (0, 0, pad, pad)).squeeze(0)
    shifts = [
        h_pad[pad - d : pad - d + l_tot] for d in (-8, -5, -3, -2, -1, 0, 1, 2, 3, 5, 8, 12, 16)
    ]
    del h_cat, h_pad

    u_s4 = torch.cat([nl_inst, wf_core, wf_hyst, *shifts], dim=-1)
    del nl_inst, wf_core, wf_hyst, shifts
    if u_s4.shape[1] > 800:
        u_s4 = project_orthonormal_basis(u_s4, 800)
    gc.collect()

    n_cols = u_s4.shape[1]
    n_tgt = 3 * m_col
    r3_3b = torch.cat([r3_lf_f32, r3_mf_f32, r3_hf_f32], dim=-1)
    g_gl = torch.zeros((n_cols, n_cols), dtype=torch.float64)
    c_gl = torch.zeros((n_cols, n_tgt), dtype=torch.float64)
    n_wins = len(win_bounds)
    overlap = 256 if l_tot >= 32_768 else 0
    d_win_3b = torch.zeros((l_tot, n_tgt), dtype=torch.float32)

    for w_idx, (k0, k1) in enumerate(win_bounds):
        s_w = 0 if w_idx == 0 else max(0, k0 - overlap)
        e_w = l_tot if w_idx == n_wins - 1 else min(l_tot, k1 + overlap)
        uk = u_s4[s_w:e_w:stride_s4].to(torch.float64)
        rk = r3_3b[s_w:e_w:stride_s4].to(torch.float64)
        gw = uk.T @ uk
        cw = uk.T @ rk
        del uk, rk
        uk_pre = (u_s4[s_w + 1 : e_w : stride_s4] - 0.85 * u_s4[s_w : e_w - 1 : stride_s4]).to(
            torch.float64
        )
        rk_pre = (r3_3b[s_w + 1 : e_w : stride_s4] - 0.85 * r3_3b[s_w : e_w - 1 : stride_s4]).to(
            torch.float64
        )
        gw += 0.5 * (uk_pre.T @ uk_pre)
        cw += 0.5 * (uk_pre.T @ rk_pre)
        del uk_pre, rk_pre
        g_gl += gw
        c_gl += cw
        ww = solve_cholesky_ridge(gw, cw, ridge_scale=1e-5)
        bw_slice = bspline_weights_win[s_w:e_w, w_idx : w_idx + 1]
        d_win_3b[s_w:e_w] += bw_slice * (u_s4[s_w:e_w] @ ww)
        del gw, ww

    del r3_3b
    w_gl = solve_cholesky_ridge(g_gl, c_gl, ridge_scale=2e-6)
    d_glob_3b = u_s4 @ w_gl
    del u_s4, g_gl, c_gl, w_gl
    gc.collect()

    d4_tot_cols: list[torch.Tensor] = []
    d4_lf_cols: list[torch.Tensor] = []
    d4_mf_cols: list[torch.Tensor] = []
    d4_hf_cols: list[torch.Tensor] = []
    use_lti_polish = l_tot >= 48_000
    for s_idx in range(m_col):
        b_idx = [s_idx, m_col + s_idx, 2 * m_col + s_idx]
        w3_s = d_win_3b[:, b_idx]
        g3_s = d_glob_3b[:, b_idx]
        base_6 = torch.cat([w3_s, g3_s], dim=-1)
        phi_parts = [base_6 * g.unsqueeze(-1) for g in ref_gates]
        if use_lti_polish:
            w_l120 = fast_1pole_lpf(w3_s, 120.0, sr=sr)
            w_l400 = fast_1pole_lpf(w3_s, 400.0, sr=sr)
            w_l1500 = fast_1pole_lpf(w3_s, 1500.0, sr=sr)
            w_l4500 = fast_1pole_lpf(w3_s, 4500.0, sr=sr)
            pad_c = 6
            w_pad = F.pad(w3_s.unsqueeze(0), (0, 0, pad_c, pad_c)).squeeze(0)
            w_sh = [w_pad[pad_c - d : pad_c - d + l_tot] for d in (-4, -2, -1, 1, 2, 4)]
            r_tot_s = (
                r3_lf_f32[:, s_idx : s_idx + 1]
                + r3_mf_f32[:, s_idx : s_idx + 1]
                + r3_hf_f32[:, s_idx : s_idx + 1]
            ).to(torch.float64)
            _, r_lti_l, r_lti_m, r_lti_h = solve_lti_cabinet_and_space_stage(
                z3_full[:, s_idx],
                r_tot_s,
                max_ir_len=2048,
                sr=sr,
            )
            phi_parts.extend([w_l120, w_l400, w_l1500, w_l4500, *w_sh])
            phi_parts.append(torch.cat([r_lti_l, r_lti_m, r_lti_h], dim=-1))
            del w_l120, w_l400, w_l1500, w_l4500, w_pad, w_sh, r_tot_s, r_lti_l, r_lti_m, r_lti_h
        phi_s = torch.cat(phi_parts, dim=-1)
        del base_6, phi_parts

        w_lf_s = 1.0 / float(sig_lf[s_idx].item())
        w_mf_s = 1.0 / float(sig_mf[s_idx].item())
        w_hf_s = 1.0 / float(sig_hf[s_idx].item())
        rlf_s = r3_lf_f32[:, s_idx : s_idx + 1]
        rmf_s = r3_mf_f32[:, s_idx : s_idx + 1]
        rhf_s = r3_hf_f32[:, s_idx : s_idx + 1]
        d_tot_s, d_lf_s, d_mf_s, d_hf_s = _solve_3band_weighted_synthesis(
            phi_s, rlf_s, rmf_s, rhf_s, w_lf_s, w_mf_s, w_hf_s, stride=stride_s4, sr=sr
        )
        del phi_s
        d4_lf_cols.append(d_lf_s)
        d4_mf_cols.append(d_mf_s)
        d4_hf_cols.append(d_hf_s)
        d4_tot_cols.append(d_tot_s)
    del d_win_3b, d_glob_3b, r3_lf_f32, r3_mf_f32, r3_hf_f32
    return (
        torch.stack(d4_tot_cols, dim=-1),
        torch.stack(d4_lf_cols, dim=-1),
        torch.stack(d4_mf_cols, dim=-1),
        torch.stack(d4_hf_cols, dim=-1),
    )


@torch.no_grad()
def run_3stage_selective_tube_cascade(
    x: torch.Tensor,
    y_full: torch.Tensor,
    y_lf: torch.Tensor,
    y_mf: torch.Tensor,
    y_hf: torch.Tensor,
    sig_lf_all: torch.Tensor,
    sig_mf_all: torch.Tensor,
    sig_hf_all: torch.Tensor,
    *,
    z_lti_bands: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    sr: float,
    on_stage_progress: Callable[[int, str], None] | None,
    report_step: Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Executes the 3-Stage Selective State-Space Tube/Solid-State Cascade + Stage-4 Quadrature
    Hysteresis Polish and returns `(z1_full, z2_full, z3_full, d4_tot, z4_lf, z4_mf, z4_hf, z3_hf)`.
    """
    l_tot = x.shape[0]
    m_col = y_full.shape[1]

    if l_tot >= 500_000:
        win_bounds = [(w * 24_000, (w + 1) * 24_000 if w < 22 else l_tot) for w in range(23)]
        win_bounds48 = [(w * 12_000, (w + 1) * 12_000 if w < 47 else l_tot) for w in range(48)]
        regimes = [
            (0, 140_000),
            (140_000, 260_000),
            (260_000, 380_000),
            (380_000, 440_000),
            (440_000, 500_000),
            (500_000, l_tot),
        ]
    elif l_tot >= 48_000:
        win_len = 24_000
        n_wins = max(1, (l_tot + win_len - 1) // win_len)
        win_bounds = [(w * win_len, min(l_tot, (w + 1) * win_len)) for w in range(n_wins)]
        win_len48 = 12_000
        n_wins48 = max(1, (l_tot + win_len48 - 1) // win_len48)
        win_bounds48 = [(w * win_len48, min(l_tot, (w + 1) * win_len48)) for w in range(n_wins48)]
        step_r = max(win_len, l_tot // 6)
        regimes = [
            (r * step_r, l_tot if r == 5 else min(l_tot, (r + 1) * step_r)) for r in range(6)
        ]
        regimes = [(r0, r1) for (r0, r1) in regimes if r1 > r0]
    else:
        win_bounds = [(0, l_tot)]
        win_bounds48 = [(0, l_tot)]
        regimes = [(0, l_tot)]

    bspline_weights_win = build_causal_bspline_window_weights(
        x, win_bounds, overlap=256 if l_tot >= 48_000 else 0, sr=sr
    )
    bspline_weights_win48 = (
        build_causal_bspline_window_weights(
            x, win_bounds48, overlap=256 if l_tot >= 48_000 else 0, sr=sr
        )
        if len(win_bounds48) != len(win_bounds)
        else bspline_weights_win
    )
    bspline_weights_reg = build_causal_bspline_window_weights(
        x, regimes, overlap=512 if l_tot >= 48_000 else 0, sr=sr
    )
    ref_gates = build_fast_refinement_gates(x, sr=sr)

    if on_stage_progress is not None:
        on_stage_progress(
            1, "Calibrating Stage 1 (V1A Preamp, Tone-Stack & LTI Cabinet/Space Cascade)"
        )
    cutoffs = (8.0, 18.0, 35.0, 65.0, 120.0, 240.0, 500.0, 1100.0, 2400.0, 5000.0)
    hpfs = [x - fast_1pole_lpf(x, fc, sr=sr) for fc in cutoffs]
    hpfs_t = torch.stack(hpfs, dim=-1)
    hpfs_2nd = [hpfs[i] - fast_1pole_lpf(hpfs[i], cutoffs[i], sr=sr) for i in range(6)]
    bp_src = torch.cat([hpfs_t[:, :8], torch.stack(hpfs_2nd, dim=-1)], dim=-1)
    bp_8k_t = fast_1pole_lpf(bp_src, 9000.0, sr=sr)
    bp_2k2_t = fast_1pole_lpf(hpfs_t[:, :5], 2200.0, sr=sr)

    x_hp12 = hpfs[0]
    x_low150 = fast_1pole_lpf(x_hp12, 150.0, sr=sr)
    x_high900 = x - fast_1pole_lpf(x, 900.0, sr=sr)
    x_high2k = x - fast_1pole_lpf(x, 2200.0, sr=sr)
    bax_pre = [
        2.8 * x_low150 + x_hp12,
        4.5 * x_low150 + 2.0 * x_hp12,
        2.5 * x_low150 + 2.0 * x_hp12 + 2.5 * x_high900,
        0.75 * x_hp12 + 2.5 * x_high900,
        0.50 * x_hp12 + 4.5 * x_high900,
        x_hp12 + 3.0 * x_high2k,
        fast_1pole_lpf(x_hp12, 550.0, sr=sr),
    ]
    env_fast = fast_1pole_lpf(torch.abs(x), 40.0, sr=sr)
    env_med = fast_1pole_lpf(torch.abs(x), 10.0, sr=sr)
    biased_bands: list[torch.Tensor] = []
    for h in (hpfs[0], hpfs[2], bax_pre[0], bax_pre[1], bax_pre[3]):
        biased_bands.append(h + 0.30 * env_fast)
        biased_bands.append(h - 0.30 * env_fast)
        biased_bands.append(h - 0.45 * env_med)

    drives_s1 = (0.6, 2.0, 6.5, 22.0, 80.0)
    adaa_s1_t = torch.cat(
        [
            x.unsqueeze(-1),
            hpfs_t[:, :6],
            bp_8k_t[:, :8],
            bp_2k2_t,
            torch.stack(bax_pre, dim=-1),
            torch.stack(biased_bands, dim=-1),
        ],
        dim=-1,
    )
    inst_s1_t = torch.stack(
        [
            x,
            hpfs[0],
            hpfs[2],
            hpfs[4],
            hpfs[6],
            hpfs[8],
            bax_pre[0],
            bax_pre[1],
            bax_pre[2],
            bax_pre[3],
            bax_pre[4],
            bax_pre[5],
        ],
        dim=-1,
    )
    del hpfs, hpfs_t, hpfs_2nd, bp_src, bp_8k_t, bp_2k2_t, bax_pre, biased_bands
    adaa_s1_out = fast_poly_adaa_batch(adaa_s1_t, drives_s1, sr=sr)
    del adaa_s1_t
    if m_col > 14 and adaa_s1_out.shape[1] > 300:
        adaa_s1_out = project_orthonormal_basis(adaa_s1_out, 300)
    inst_s1_out = fast_poly_inst_batch(inst_s1_t, drives_s1, sr=sr)
    del inst_s1_t
    if m_col > 14 and inst_s1_out.shape[1] > 110:
        inst_s1_out = project_orthonormal_basis(inst_s1_out, 110)
    u_s1 = torch.cat([x.unsqueeze(-1), adaa_s1_out, inst_s1_out], dim=-1)
    del adaa_s1_out, inst_s1_out
    gc.collect()

    z1_full, z1_lf, z1_mf, z1_hf = solve_selective_stage_3band(
        u_s1,
        y_full,
        y_lf,
        y_mf,
        y_hf,
        bspline_weights_win,
        bspline_weights_reg,
        ref_gates,
        win_bounds,
        regimes,
        z_lti_bands=z_lti_bands,
        free_u_stage=True,
        sr=sr,
    )
    del u_s1
    gc.collect()
    res_w1 = report_step(1, z1_lf, z1_mf, z1_hf)
    del z1_lf, z1_mf, z1_hf

    if on_stage_progress is not None:
        on_stage_progress(2, "Calibrating Stage 2 (V1B Interstage Coupling & Grid-Leak Cascade)")
    u_s2 = build_interstage_basis(
        z1_full,
        None,
        (15.0, 60.0, 220.0, 800.0),
        20.0,
        35.0,
        8.0,
        (0.8, 2.5, 8.0, 25.0, 75.0),
        res_weights=res_w1,
        use_power_sag=False,
        sr=sr,
    )
    z2_full, z2_lf, z2_mf, z2_hf = solve_selective_stage_3band(
        u_s2,
        y_full,
        y_lf,
        y_mf,
        y_hf,
        bspline_weights_win,
        bspline_weights_reg,
        ref_gates,
        win_bounds,
        regimes,
        res_weights=res_w1,
        z_lti_bands=z_lti_bands,
        free_u_stage=True,
        sr=sr,
    )
    del u_s2, res_w1
    gc.collect()
    res_w2 = report_step(2, z2_lf, z2_mf, z2_hf)
    del z2_lf, z2_mf, z2_hf

    if on_stage_progress is not None:
        on_stage_progress(
            3, "Calibrating Stage 3 (Power Section, Output Transformer & 48 kHz Reference Polish)"
        )
    u_s3 = build_interstage_basis(
        z2_full,
        z1_full,
        (12.0, 45.0, 180.0, 650.0),
        18.0,
        25.0,
        4.0,
        (0.6, 2.0, 6.5, 20.0, 60.0),
        res_weights=res_w2,
        use_power_sag=True,
        sr=sr,
    )
    z3_full, z3_lf, z3_mf, z3_hf = solve_selective_stage_3band(
        u_s3,
        y_full,
        y_lf,
        y_mf,
        y_hf,
        bspline_weights_win48 if m_col > 14 else bspline_weights_win,
        bspline_weights_reg,
        ref_gates,
        win_bounds48 if m_col > 14 else win_bounds,
        regimes,
        res_weights=res_w2,
        z_lti_bands=z_lti_bands,
        free_u_stage=True,
        sr=sr,
    )
    del u_s3, res_w2
    gc.collect()
    report_step(3, z3_lf, z3_mf, z3_hf)

    r3_lf = y_lf - z3_lf
    r3_mf = y_mf - z3_mf
    r3_hf = y_hf - z3_hf
    d4_tot, d4_lf, d4_mf, d4_hf = solve_fullrate_hf_residual_polish(
        z3_full,
        z2_full,
        r3_lf,
        r3_mf,
        r3_hf,
        sig_lf_all,
        sig_mf_all,
        sig_hf_all,
        bspline_weights_win48,
        ref_gates,
        win_bounds48,
        z1_full=z1_full,
        sr=sr,
    )
    del r3_lf, r3_mf, r3_hf, bspline_weights_win, bspline_weights_win48, bspline_weights_reg
    del ref_gates
    z4_lf = z3_lf + d4_lf
    z4_mf = z3_mf + d4_mf
    z4_hf = z3_hf + d4_hf
    del z3_lf, z3_mf, d4_lf, d4_mf, d4_hf
    report_step(4, z4_lf, z4_mf, z4_hf)
    gc.collect()
    return z1_full, z2_full, z3_full, d4_tot, z4_lf, z4_mf, z4_hf, z3_hf
