"""Parseval Commuted 3-Band Orthogonal FFT Decoupled Comparator & Wavefolder Cascade Solver.

Implements:
1. Exact idempotent orthogonal frequency-band projectors (`project_fft_band`) satisfying
   $P_a^2 = P_a$ and $P_a P_b = 0$ ($a \\neq b$),
2. Parseval commuted in-band orthonormal column reduction (`parseval_inband_projectors`),
3. Window-streamed 3-band decoupled Cholesky solver (`solve_3band_decoupled_window_stage`),
4. 1/3-octave LTAS high-frequency tilt balance (`apply_ltas_hf_tilt_balance`), and
5. Stage-1 and refinement comparator + wavefolder bases (`build_stage1_fuzz_basis`,
   `build_refinement_fuzz_basis`).
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F

from dnam_trainer.training.cascade_primitives import (
    build_causal_bspline_window_weights,
    fast_1pole_lpf,
    solve_cholesky_ridge,
    split_bezout_3band,
)


@torch.no_grad()
def project_fft_band(
    sig: torch.Tensor,
    f_lo: float,
    f_hi: float,
    *,
    sr: float = 48_000.0,
) -> torch.Tensor:
    """Exact idempotent orthogonal frequency-band projector in `float32`, chunked to `< 75 MB`."""
    was_1d = sig.ndim == 1
    if was_1d:
        sig = sig.unsqueeze(-1)
    l_seq, c_ch = sig.shape
    freqs = torch.fft.rfftfreq(l_seq, d=1.0 / sr, device=sig.device)
    if f_lo <= 0.0:
        mask = (freqs <= f_hi).to(torch.float32).unsqueeze(-1)
    elif f_hi >= sr * 0.5:
        mask = (freqs > f_lo).to(torch.float32).unsqueeze(-1)
    else:
        mask = ((freqs > f_lo) & (freqs <= f_hi)).to(torch.float32).unsqueeze(-1)
    del freqs

    out = torch.empty_like(sig)
    for c0 in range(0, c_ch, 16):
        c1 = min(c_ch, c0 + 16)
        spec = torch.fft.rfft(sig[:, c0:c1].to(torch.float32), n=l_seq, dim=0)
        spec.mul_(mask)
        out[:, c0:c1] = torch.fft.irfft(spec, n=l_seq, dim=0).to(sig.dtype)
        del spec
    return out.squeeze(-1) if was_1d else out


@torch.no_grad()
def parseval_inband_projectors(
    u_st: torch.Tensor,
    max_lf: int,
    max_mf: int,
    max_hf: int,
    *,
    sr: float = 48_000.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes in-band orthonormal projection matrices `(P_lf, P_mf, P_hf)` simultaneously via
    Parseval's theorem on 16 multi-window blocks (`< 90 MB` RAM).
    """
    l_seq, c_in = u_st.shape
    blk_len = min(4096, l_seq)
    n_blks = min(16, max(1, l_seq // blk_len))
    step = max(1, (l_seq - blk_len) // max(1, n_blks - 1)) if n_blks > 1 else 0
    blocks = torch.stack([u_st[i * step : i * step + blk_len] for i in range(n_blks)], dim=0)
    spec = torch.fft.rfft(blocks.to(torch.float32), n=blk_len, dim=1)
    del blocks
    freqs = torch.fft.rfftfreq(blk_len, d=1.0 / sr, device=u_st.device)

    def _proj_from_mask(spec_t: torch.Tensor, mask: torch.Tensor, k_max: int) -> torch.Tensor:
        sb = spec_t[:, mask, :].reshape(-1, c_in).to(torch.complex128)
        gram_raw = torch.real(sb.H @ sb)
        del sb
        col_std = torch.sqrt(torch.clamp(torch.diagonal(gram_raw), min=1e-18)).unsqueeze(0)
        gram_n = gram_raw / (col_std.T @ col_std)
        _, eigvecs = torch.linalg.eigh(gram_n)
        k_use = min(k_max, c_in)
        top_v = eigvecs[:, -k_use:]
        return (top_v / col_std.T).to(u_st.dtype)

    p_lf = _proj_from_mask(spec, freqs <= 250.0, max_lf)
    p_mf = _proj_from_mask(spec, (freqs > 250.0) & (freqs <= 2500.0), max_mf)
    p_hf = _proj_from_mask(spec, freqs > 2500.0, max_hf)
    del spec
    return p_lf, p_mf, p_hf


@torch.no_grad()
def solve_3band_decoupled_window_stage(
    u_st: torch.Tensor,
    cur_lf: torch.Tensor,
    cur_mf: torch.Tensor,
    cur_hf: torch.Tensor,
    y_lf_fft: torch.Tensor,
    y_mf_fft: torch.Tensor,
    y_hf_fft: torch.Tensor,
    bspline_weights: torch.Tensor,
    win_bounds: Sequence[tuple[int, int]],
    *,
    shifts: tuple[int, ...] = (-6, -3, -1, 1, 3, 6),
    max_cols_lf: int = 140,
    max_cols_mf: int = 180,
    max_cols_hf: int = 250,
    ridge_scale: float = 1e-5,
    overlap: int = 128,
    free_u_st: bool = True,
    sr: float = 48_000.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Memory-bounded (`< 1.0 GB` working buffer) sequential 3-band orthogonal FFT decoupled
    window solver with Parseval commuted column reduction.
    """
    l_tot, m_col = y_lf_fft.shape
    n_wins = len(win_bounds)
    pad = max(abs(d) for d in shifts) + 2

    p_lf, p_mf, p_hf = parseval_inband_projectors(
        u_st, max_cols_lf, max_cols_mf, max_cols_hf, sr=sr
    )
    u_lf_pre = u_st @ p_lf
    u_mf_pre = u_st @ p_mf
    k_hf = p_hf.shape[1]
    if free_u_st and k_hf <= u_st.shape[1]:
        for r0 in range(0, l_tot, 65_536):
            r1 = min(l_tot, r0 + 65_536)
            u_st[r0:r1, :k_hf] = u_st[r0:r1].clone() @ p_hf
        u_hf_pre = u_st[:, :k_hf].contiguous()
        u_st.resize_(0)
    else:
        u_hf_pre = u_st @ p_hf
        if free_u_st:
            u_st.resize_(0)
    del p_lf, p_mf, p_hf
    gc.collect()

    def _solve_single_band(
        u_band: torch.Tensor,
        cur_band: torch.Tensor,
        y_band: torch.Tensor,
        f_lo: float,
        f_hi: float,
        stride: int,
    ) -> torch.Tensor:
        out_band = torch.zeros((l_tot, m_col), dtype=torch.float32)
        has_cur = float(torch.sum(cur_band**2).item()) > 1e-12
        for w_idx, (k0, k1) in enumerate(win_bounds):
            s_w = 0 if w_idx == 0 else max(0, k0 - overlap)
            e_w = l_tot if w_idx == n_wins - 1 else min(l_tot, k1 + overlap)
            sl = e_w - s_w

            p0 = max(0, s_w - pad)
            p1 = min(l_tot, e_w + pad)
            left_pad = max(0, pad - s_w)
            right_pad = max(0, (e_w + pad) - l_tot)
            u_sh_slice = u_band[p0:p1]
            if left_pad > 0 or right_pad > 0:
                u_sh_pad = F.pad(u_sh_slice.unsqueeze(0), (0, 0, left_pad, right_pad)).squeeze(0)
            else:
                u_sh_pad = u_sh_slice
            base_off = (s_w - p0) + left_pad

            parts = [u_band[s_w:e_w]]
            for d in shifts:
                parts.append(u_sh_pad[base_off - d : base_off - d + sl])
            if has_cur:
                parts.insert(0, cur_band[s_w:e_w])
            b_win = torch.cat(parts, dim=-1)
            del parts, u_sh_pad

            b_sub = b_win[::stride].to(torch.float64)
            y_sub = y_band[s_w:e_w:stride].to(torch.float64)
            g_w = b_sub.T @ b_sub
            c_w = b_sub.T @ y_sub
            del b_sub, y_sub

            w_mat = solve_cholesky_ridge(g_w, c_w, ridge_scale=ridge_scale, min_ridge=1e-9)
            del g_w, c_w

            bw_slice = bspline_weights[s_w:e_w, w_idx : w_idx + 1]
            out_band[s_w:e_w] += bw_slice * (b_win @ w_mat)
            del b_win, w_mat
        return project_fft_band(out_band, f_lo, f_hi, sr=sr)

    u_lf = project_fft_band(u_lf_pre, 0.0, 250.0, sr=sr)
    del u_lf_pre
    new_lf = _solve_single_band(u_lf, cur_lf, y_lf_fft, 0.0, 250.0, stride=4)
    del u_lf

    u_mf = project_fft_band(u_mf_pre, 250.0, 2500.0, sr=sr)
    del u_mf_pre
    new_mf = _solve_single_band(u_mf, cur_mf, y_mf_fft, 250.0, 2500.0, stride=2)
    del u_mf

    u_hf = project_fft_band(u_hf_pre, 2500.0, sr * 0.5, sr=sr)
    del u_hf_pre
    new_hf = _solve_single_band(u_hf, cur_hf, y_hf_fft, 2500.0, sr * 0.5, stride=1)
    del u_hf

    return new_lf, new_mf, new_hf


@torch.no_grad()
def apply_ltas_hf_tilt_balance(
    z_hf: torch.Tensor,
    y_hf: torch.Tensor,
    *,
    sr: float = 48_000.0,
) -> torch.Tensor:
    """Applies a mild 1/3-octave LTAS energy balance (`[0.85, 1.18]`, max +/- 1.4 dB) above 2500 Hz
    to compensate for least-squares attenuation ($r^2 < 1$) in low-energy high-frequency bands.
    """
    l_seq = z_hf.shape[0]
    freqs = torch.fft.rfftfreq(l_seq, d=1.0 / sr, device=z_hf.device)
    centers_hz = (2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0, 12500.0, 16000.0)
    two_1_6 = 2.0 ** (1.0 / 6.0)

    spec_p = torch.fft.rfft(z_hf.to(torch.float32), n=l_seq, dim=0)
    spec_t = torch.fft.rfft(y_hf.to(torch.float32), n=l_seq, dim=0)
    pow_p = torch.abs(spec_p).to(torch.float64) ** 2
    pow_t = torch.abs(spec_t).to(torch.float64) ** 2

    gain_curve = torch.ones_like(pow_p, dtype=torch.float32)
    for fc in centers_hz:
        f_lo = fc / two_1_6
        f_hi = fc * two_1_6
        mask = (freqs >= f_lo) & (freqs < f_hi)
        if not torch.any(mask):
            continue
        ep = torch.sum(pow_p[mask], dim=0) + 1e-18
        et = torch.sum(pow_t[mask], dim=0) + 1e-18
        g = torch.clamp((et / ep) ** 0.45, 0.85, 1.18).to(torch.float32)
        gain_curve[mask] = g.unsqueeze(0)

    spec_p.mul_(gain_curve)
    del spec_t, pow_p, pow_t, gain_curve
    out = torch.fft.irfft(spec_p, n=l_seq, dim=0).to(z_hf.dtype)
    return project_fft_band(out, 2500.0, sr * 0.5, sr=sr)


@torch.no_grad()
def build_stage1_fuzz_basis(x: torch.Tensor, sr: float = 48_000.0) -> torch.Tensor:
    """Builds Stage-1 broadband + envelope-normalized comparator basis on `x`."""
    cutoffs = (12.0, 30.0, 70.0, 160.0, 380.0, 900.0, 2200.0, 5000.0)
    hpfs = [x - fast_1pole_lpf(x, fc, sr=sr) for fc in cutoffs]
    lpfs = [fast_1pole_lpf(hpfs[0], fc, sr=sr) for fc in (100.0, 220.0, 480.0, 1100.0)]
    env_f = fast_1pole_lpf(torch.abs(x), 35.0, sr=sr)
    env_s = fast_1pole_lpf(torch.abs(x), 8.0, sr=sr)

    pre_sigs = [x, *hpfs, *lpfs]
    for h in (hpfs[0], hpfs[2], hpfs[4], lpfs[0], lpfs[1]):
        env_h = fast_1pole_lpf(torch.abs(h), 20.0, sr=sr)
        pre_sigs.append(0.15 * h / (env_h + 0.004))
        pre_sigs.append(0.15 * (h - 0.35 * env_h) / (env_h + 0.004))
        pre_sigs.append(h - 0.35 * env_f)
        pre_sigs.append(h + 0.35 * env_f)
        pre_sigs.append(h - 0.75 * env_s)

    nl_cols = [x, *hpfs[:6], *lpfs]
    for sig in pre_sigs:
        for g in (1.2, 4.0, 14.0, 50.0, 180.0):
            tg = torch.tanh(g * sig)
            nl_cols.append(tg)
            nl_cols.append(tg * tg - fast_1pole_lpf(tg * tg, 10.0, sr=sr))
    return torch.stack(nl_cols, dim=-1)


@torch.no_grad()
def build_refinement_fuzz_basis(
    cur_lf: torch.Tensor,
    cur_mf: torch.Tensor,
    cur_hf: torch.Tensor,
    *,
    include_wavefolder: bool = True,
    n_modes: int = 8,
    sr: float = 48_000.0,
) -> torch.Tensor:
    """Pre-allocated in-place refinement basis builder (avoids doubling RAM in `torch.cat`)."""
    z_prev = cur_lf + cur_mf + cur_hf
    z_lm = cur_lf + cur_mf
    l_seq, m_col = z_prev.shape

    z_glob = z_prev / (torch.sqrt(torch.mean(z_prev**2, dim=0, keepdim=True) + 1e-12)) * 0.20
    z_lm_glob = z_lm / (torch.sqrt(torch.mean(z_lm**2, dim=0, keepdim=True) + 1e-12)) * 0.20
    del z_prev, z_lm

    if m_col > n_modes:
        svd_st = max(1, l_seq // 32_768)
        _, _, vt = torch.linalg.svd(z_glob[::svd_st].to(torch.float64), full_matrices=False)
        proj = vt[:n_modes, :].T.to(torch.float32)
        zg = z_glob @ proj
        z_lmg = z_lm_glob @ proj
        del z_glob, z_lm_glob, proj
    else:
        zg = z_glob
        z_lmg = z_lm_glob

    k_m = zg.shape[1]
    env_loc = torch.sqrt(fast_1pole_lpf(zg * zg, 6.0, sr=sr) + 1e-6)
    z_loc = 0.20 * zg / (env_loc + 0.015)
    del env_loc
    env_lmloc = torch.sqrt(fast_1pole_lpf(z_lmg * z_lmg, 6.0, sr=sr) + 1e-6)
    z_lmloc = 0.20 * z_lmg / (env_lmloc + 0.015)
    del env_lmloc

    z_hp300 = zg - fast_1pole_lpf(zg, 300.0, sr=sr)
    z_hp1200 = zg - fast_1pole_lpf(zg, 1200.0, sr=sr)
    ez = fast_1pole_lpf(torch.abs(zg), 25.0, sr=sr)

    sigs = (zg, z_lmg, z_loc, z_lmloc, z_hp300, zg - 0.35 * ez)
    n_per_sig = 10 + (3 if include_wavefolder else 0)
    n_blocks = 6 + len(sigs) * n_per_sig
    out = torch.empty((l_seq, n_blocks * k_m), dtype=torch.float32)

    idx = 0
    for base_b in (zg, z_lmg, z_loc, z_lmloc, z_hp300, z_hp1200):
        out[:, idx : idx + k_m] = base_b
        idx += k_m
    del z_hp1200

    for sig_s in sigs:
        for g in (1.5, 5.0, 18.0, 65.0, 250.0):
            tg = torch.tanh(g * sig_s)
            out[:, idx : idx + k_m] = tg
            idx += k_m
            tg_sq = tg * tg
            del tg
            out[:, idx : idx + k_m] = tg_sq - fast_1pole_lpf(tg_sq, 15.0, sr=sr)
            del tg_sq
            idx += k_m
        if include_wavefolder:
            for g_wf in (2.5, 8.0, 25.0):
                out[:, idx : idx + k_m] = torch.sin(g_wf * sig_s)
                idx += k_m
    return out


@torch.no_grad()
def run_4stage_orthogonal_fft_cascade(
    x: torch.Tensor,
    y_full: torch.Tensor,
    *,
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
    """Executes the 4-Stage Parseval Commuted 3-Band Orthogonal FFT Decoupled Comparator &
    Wavefolder Cascade, returning `(z1_full, z2_full, z3_full, d4_tot, z4_lf, z4_mf, z4_hf, z3_hf)`.
    """
    l_tot = x.shape[0]

    def _make_active_windows(step_len: int) -> list[tuple[int, int]]:
        if l_tot >= 500_000:
            starts = [0, *range(36_000, 540_000, step_len)]
        else:
            starts = list(range(0, l_tot, step_len))
        bounds_w: list[tuple[int, int]] = []
        for idx_s, s0 in enumerate(starts):
            s1 = l_tot if idx_s == len(starts) - 1 else starts[idx_s + 1]
            if s1 > s0:
                bounds_w.append((s0, s1))
        return bounds_w

    y_lf_fft = project_fft_band(y_full, 0.0, 250.0, sr=sr)
    y_mf_fft = project_fft_band(y_full, 250.0, 2500.0, sr=sr)
    y_hf_fft = project_fft_band(y_full, 2500.0, sr * 0.5, sr=sr)

    win12 = _make_active_windows(12_000)
    win8 = _make_active_windows(8_000)
    win6 = _make_active_windows(6_000)
    win4 = _make_active_windows(4_000)

    bw12 = build_causal_bspline_window_weights(x, win12, overlap=192, sr=sr)
    bw8 = build_causal_bspline_window_weights(x, win8, overlap=160, sr=sr)
    bw6 = build_causal_bspline_window_weights(x, win6, overlap=128, sr=sr)
    bw4 = build_causal_bspline_window_weights(x, win4, overlap=96, sr=sr)

    cur_lf = torch.zeros_like(y_lf_fft)
    cur_mf = torch.zeros_like(y_mf_fft)
    cur_hf = torch.zeros_like(y_hf_fft)

    if on_stage_progress is not None:
        on_stage_progress(
            1, "Calibrating Stage 1 (V1A Preamp, Comparator & 3-Band Orthogonal Cascade)"
        )
    u1 = build_stage1_fuzz_basis(x, sr=sr)
    cur_lf, cur_mf, cur_hf = solve_3band_decoupled_window_stage(
        u1,
        cur_lf,
        cur_mf,
        cur_hf,
        y_lf_fft,
        y_mf_fft,
        y_hf_fft,
        bw12,
        win12,
        max_cols_lf=140,
        max_cols_mf=180,
        max_cols_hf=240,
        overlap=192,
        sr=sr,
    )
    del u1, bw12
    gc.collect()
    z1_full = cur_lf + cur_mf + cur_hf
    z1_b_lf, z1_b_mf, z1_b_hf = split_bezout_3band(z1_full, sr=sr)
    report_step(1, z1_b_lf, z1_b_mf, z1_b_hf)
    del z1_b_lf, z1_b_mf, z1_b_hf

    if on_stage_progress is not None:
        on_stage_progress(2, "Calibrating Stage 2 (Interstage Comparator & Wavefolder Cascade)")
    u2 = build_refinement_fuzz_basis(
        cur_lf, cur_mf, cur_hf, include_wavefolder=True, n_modes=8, sr=sr
    )
    cur_lf, cur_mf, cur_hf = solve_3band_decoupled_window_stage(
        u2,
        cur_lf,
        cur_mf,
        cur_hf,
        y_lf_fft,
        y_mf_fft,
        y_hf_fft,
        bw8,
        win8,
        max_cols_lf=140,
        max_cols_mf=190,
        max_cols_hf=250,
        overlap=160,
        sr=sr,
    )
    del u2, bw8
    gc.collect()
    z2_full = cur_lf + cur_mf + cur_hf
    z2_b_lf, z2_b_mf, z2_b_hf = split_bezout_3band(z2_full, sr=sr)
    report_step(2, z2_b_lf, z2_b_mf, z2_b_hf)
    del z2_b_lf, z2_b_mf, z2_b_hf

    if on_stage_progress is not None:
        on_stage_progress(
            3, "Calibrating Stage 3 (Power Section, Wavefolder & 48 kHz Reference Polish)"
        )
    u3 = build_refinement_fuzz_basis(
        cur_lf, cur_mf, cur_hf, include_wavefolder=True, n_modes=8, sr=sr
    )
    cur_lf, cur_mf, cur_hf = solve_3band_decoupled_window_stage(
        u3,
        cur_lf,
        cur_mf,
        cur_hf,
        y_lf_fft,
        y_mf_fft,
        y_hf_fft,
        bw6,
        win6,
        max_cols_lf=150,
        max_cols_mf=200,
        max_cols_hf=260,
        overlap=128,
        sr=sr,
    )
    del u3, bw6
    gc.collect()
    z3_full = cur_lf + cur_mf + cur_hf
    z3_lf, z3_mf, z3_hf = split_bezout_3band(z3_full, sr=sr)
    report_step(3, z3_lf, z3_mf, z3_hf)
    del z3_lf, z3_mf

    u4 = build_refinement_fuzz_basis(
        cur_lf, cur_mf, cur_hf, include_wavefolder=True, n_modes=8, sr=sr
    )
    cur_lf, cur_mf, cur_hf = solve_3band_decoupled_window_stage(
        u4,
        cur_lf,
        cur_mf,
        cur_hf,
        y_lf_fft,
        y_mf_fft,
        y_hf_fft,
        bw4,
        win4,
        max_cols_lf=150,
        max_cols_mf=210,
        max_cols_hf=270,
        overlap=96,
        sr=sr,
    )
    del u4, bw4, y_lf_fft, y_mf_fft
    gc.collect()
    cur_hf = apply_ltas_hf_tilt_balance(cur_hf, y_hf_fft, sr=sr)
    del y_hf_fft
    z4_full_tot = cur_lf + cur_mf + cur_hf
    del cur_lf, cur_mf, cur_hf
    d4_tot = z4_full_tot - z3_full
    z4_lf, z4_mf, z4_hf = split_bezout_3band(z4_full_tot, sr=sr)
    del z4_full_tot
    report_step(4, z4_lf, z4_mf, z4_hf)
    gc.collect()
    return z1_full, z2_full, z3_full, d4_tot, z4_lf, z4_mf, z4_hf, z3_hf
