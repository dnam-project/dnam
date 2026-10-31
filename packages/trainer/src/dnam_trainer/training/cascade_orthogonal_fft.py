"""Parseval Commuted 3-Band Orthogonal FFT Decoupled Comparator & Wavefolder Cascade Solver.

Implements:
1. Exact idempotent orthogonal frequency-band projectors (`project_fft_band`) satisfying
   $P_a^2 = P_a$ and $P_a P_b = 0$ ($a \\neq b$),
2. Parseval commuted in-band orthonormal column reduction (`parseval_inband_projectors`) via
   on-device `ExactGramAccumulator` + host `float64` `eigh`,
3. Window-streamed 3-band decoupled solver (`solve_3band_decoupled_window_stage`) via batched
   `HostDenseSolver`,
4. 1/3-octave LTAS high-frequency tilt balance (`apply_ltas_hf_tilt_balance`), and
5. Stage-1 and refinement comparator + wavefolder bases (`build_stage1_fuzz_basis`,
   `build_refinement_fuzz_basis`) chunked so no tensor exceeds `max_tensor_bytes` (`1 GiB`).
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from dnam_trainer.training.accelerator import empty_device_cache, free_tensor_storage
from dnam_trainer.training.cascade_primitives import (
    fast_1pole_lpf,
    split_bezout_3band,
)
from dnam_trainer.training.cascade_selective_stages import (
    MAX_CHUNK_COLS,
    ColumnChunkedMatrix,
)
from dnam_trainer.training.numerics import (
    ExactGramAccumulator,
    HostDenseSolver,
    HostSolveSpec,
)
from dnam_trainer.training.windowing import (
    CausalBsplineWindowSchedule,
    build_orthogonal_fft_windows,
    compute_cumulative_energy_state,
)


@torch.no_grad()
def project_fft_band(
    sig: Tensor,
    f_lo: float,
    f_hi: float,
    *,
    sr: float = 48_000.0,
) -> Tensor:
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


def _project_fft_band_chunked(
    cm: ColumnChunkedMatrix,
    f_lo: float,
    f_hi: float,
    *,
    sr: float = 48_000.0,
    free_input: bool = True,
) -> ColumnChunkedMatrix:
    out_chunks: list[Tensor] = []
    for ch in cm.chunks:
        out_chunks.append(project_fft_band(ch, f_lo, f_hi, sr=sr))
        if free_input:
            free_tensor_storage(ch)
    if free_input:
        cm.chunks.clear()
    return ColumnChunkedMatrix(out_chunks)


@torch.no_grad()
def parseval_inband_projectors(
    u_st: Tensor | ColumnChunkedMatrix,
    max_lf: int,
    max_mf: int,
    max_hf: int,
    *,
    solver: HostDenseSolver | None = None,
    sr: float = 48_000.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Computes in-band orthonormal projection matrices `(P_lf, P_mf, P_hf)` simultaneously via
    Parseval's theorem on 16 multi-window blocks using 32-bit `complex64` FFT +
    `ExactGramAccumulator` + host `float64` `eigh`.
    """
    host_solver = solver if solver is not None else HostDenseSolver()
    u_cm = (
        u_st if isinstance(u_st, ColumnChunkedMatrix) else ColumnChunkedMatrix.from_tensors([u_st])
    )
    l_seq, c_in = u_cm.shape
    dev = u_cm.device
    blk_len = min(4096, l_seq)
    n_blks = min(16, max(1, l_seq // blk_len))
    step = max(1, (l_seq - blk_len) // max(1, n_blks - 1)) if n_blks > 1 else 0
    blocks = torch.stack(
        [u_cm.slice_rows(i * step, i * step + blk_len) for i in range(n_blks)],
        dim=0,
    )
    spec = torch.fft.rfft(blocks.to(torch.float32), n=blk_len, dim=1)
    del blocks
    freqs = torch.fft.rfftfreq(blk_len, d=1.0 / sr, device=dev)

    def _proj_from_mask(spec_t: Tensor, mask: Tensor, k_max: int) -> Tensor:
        sb = spec_t[:, mask, :].reshape(-1, c_in)
        sb_re_im = torch.cat([sb.real, sb.imag], dim=0).contiguous()
        del sb
        k_use = min(k_max, c_in)
        return host_solver.fit_orthonormal_projector(sb_re_im, k_use).to(u_cm.dtype)

    p_lf = _proj_from_mask(spec, freqs <= 250.0, max_lf)
    p_mf = _proj_from_mask(spec, (freqs > 250.0) & (freqs <= 2500.0), max_mf)
    p_hf = _proj_from_mask(spec, freqs > 2500.0, max_hf)
    del spec
    return p_lf, p_mf, p_hf


def _matmul_to_chunked(
    u_cm: ColumnChunkedMatrix,
    proj: Tensor,
    *,
    free_u: bool = False,
) -> ColumnChunkedMatrix:
    k_out = proj.shape[1]
    out_chunks: list[Tensor] = []
    out_ranges: list[tuple[int, int]] = []
    for k0 in range(0, k_out, MAX_CHUNK_COLS):
        k1 = min(k_out, k0 + MAX_CHUNK_COLS)
        out_chunks.append(torch.zeros((u_cm.l_tot, k1 - k0), device=u_cm.device, dtype=u_cm.dtype))
        out_ranges.append((k0, k1))

    for ch, (c0, c1) in zip(u_cm.chunks, u_cm.col_ranges, strict=True):
        p_in = proj[c0:c1]
        for ch_out, (k0, k1) in zip(out_chunks, out_ranges, strict=True):
            ch_out.add_(ch @ p_in[:, k0:k1])
        if free_u:
            free_tensor_storage(ch)
    if free_u:
        u_cm.chunks.clear()
    return ColumnChunkedMatrix(out_chunks)


@torch.no_grad()
def solve_3band_decoupled_window_stage(
    u_st: Tensor | ColumnChunkedMatrix,
    cur_lf: Tensor,
    cur_mf: Tensor,
    cur_hf: Tensor,
    y_lf_fft: Tensor,
    y_mf_fft: Tensor,
    y_hf_fft: Tensor,
    bspline_weights: Tensor | CausalBsplineWindowSchedule,
    win_bounds: Sequence[tuple[int, int]],
    *,
    shifts: tuple[int, ...] = (-6, -3, -1, 1, 3, 6),
    max_cols_lf: int = 140,
    max_cols_mf: int = 180,
    max_cols_hf: int = 250,
    ridge_scale: float = 1e-5,
    overlap: int = 128,
    free_u_st: bool = True,
    solver: HostDenseSolver | None = None,
    sr: float = 48_000.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Memory-bounded sequential 3-band orthogonal FFT decoupled window solver with Parseval
    commuted column reduction and batched host-CPU `float64` solves.
    """
    host_solver = solver if solver is not None else HostDenseSolver()
    u_cm = (
        u_st if isinstance(u_st, ColumnChunkedMatrix) else ColumnChunkedMatrix.from_tensors([u_st])
    )
    l_tot, m_col = y_lf_fft.shape
    dev = y_lf_fft.device
    n_wins = len(win_bounds)
    pad = max(abs(d) for d in shifts) + 2

    p_lf, p_mf, p_hf = parseval_inband_projectors(
        u_cm, max_cols_lf, max_cols_mf, max_cols_hf, solver=host_solver, sr=sr
    )

    def _build_b_win(
        u_band_cm: ColumnChunkedMatrix,
        cur_band: Tensor,
        has_cur: bool,
        s_w: int,
        e_w: int,
    ) -> Tensor:
        sl = e_w - s_w
        p0 = max(0, s_w - pad)
        p1 = min(l_tot, e_w + pad)
        left_pad = max(0, pad - s_w)
        right_pad = max(0, (e_w + pad) - l_tot)
        u_sh_slice = u_band_cm.slice_rows(p0, p1)
        if left_pad > 0 or right_pad > 0:
            u_sh_pad = F.pad(u_sh_slice.unsqueeze(0), (0, 0, left_pad, right_pad)).squeeze(0)
        else:
            u_sh_pad = u_sh_slice
        base_off = (s_w - p0) + left_pad

        parts = [u_sh_pad[base_off : base_off + sl]]
        for d in shifts:
            parts.append(u_sh_pad[base_off - d : base_off - d + sl])
        if has_cur:
            parts.insert(0, cur_band[s_w:e_w])
        return torch.cat(parts, dim=-1)

    def _solve_single_band(
        u_band_cm: ColumnChunkedMatrix,
        cur_band: Tensor,
        y_band: Tensor,
        f_lo: float,
        f_hi: float,
        stride: int,
    ) -> Tensor:
        has_cur = float(torch.sum(cur_band * cur_band).item()) > 1.0e-12
        out_band = torch.zeros((l_tot, m_col), device=dev, dtype=torch.float32)
        win_batch_size = 4

        for b0 in range(0, n_wins, win_batch_size):
            b1 = min(n_wins, b0 + win_batch_size)
            specs_batch: list[HostSolveSpec] = []
            b_wins_batch: list[tuple[int, int, int, Tensor]] = []
            for w_idx in range(b0, b1):
                k0, k1 = win_bounds[w_idx]
                s_w = 0 if w_idx == 0 else max(0, k0 - overlap)
                e_w = l_tot if w_idx == n_wins - 1 else min(l_tot, k1 + overlap)
                b_win = _build_b_win(u_band_cm, cur_band, has_cur, s_w, e_w)

                acc_w = ExactGramAccumulator(b_win.shape[1], m_col, device=dev)
                acc_w.add(b_win[::stride], y_band[s_w:e_w:stride], weight=1.0)
                acc_w.offload_to_cpu()
                specs_batch.append(
                    HostSolveSpec(accumulator=acc_w, ridge_scale=ridge_scale, min_ridge=1e-9)
                )
                b_wins_batch.append((w_idx, s_w, e_w, b_win))

            w_mats_batch = host_solver.solve_batch(
                specs_batch, pass_name="ortho_fft_band_win", return_device=dev
            )
            del specs_batch
            for (w_idx, s_w, e_w, b_win), w_mat in zip(b_wins_batch, w_mats_batch, strict=True):
                if isinstance(bspline_weights, CausalBsplineWindowSchedule):
                    bw_slice = bspline_weights.window_weight(w_idx, s_w, e_w)
                else:
                    bw_slice = bspline_weights[s_w:e_w, w_idx : w_idx + 1]
                out_band[s_w:e_w].add_(bw_slice * (b_win @ w_mat))
            del b_wins_batch, w_mats_batch

        u_band_cm.free()
        res_band = project_fft_band(out_band, f_lo, f_hi, sr=sr)
        del out_band
        empty_device_cache(dev)
        return res_band

    u_lf_pre = _matmul_to_chunked(u_cm, p_lf, free_u=False)
    del p_lf
    u_lf = _project_fft_band_chunked(u_lf_pre, 0.0, 250.0, sr=sr, free_input=True)
    del u_lf_pre
    new_lf = _solve_single_band(u_lf, cur_lf, y_lf_fft, 0.0, 250.0, stride=4)
    del u_lf

    u_mf_pre = _matmul_to_chunked(u_cm, p_mf, free_u=False)
    del p_mf
    u_mf = _project_fft_band_chunked(u_mf_pre, 250.0, 2500.0, sr=sr, free_input=True)
    del u_mf_pre
    new_mf = _solve_single_band(u_mf, cur_mf, y_mf_fft, 250.0, 2500.0, stride=2)
    del u_mf

    u_hf_pre = _matmul_to_chunked(u_cm, p_hf, free_u=free_u_st)
    if free_u_st and isinstance(u_st, Tensor):
        free_tensor_storage(u_st)
    del p_hf
    u_hf = _project_fft_band_chunked(u_hf_pre, 2500.0, sr * 0.5, sr=sr, free_input=True)
    del u_hf_pre
    new_hf = _solve_single_band(u_hf, cur_hf, y_hf_fft, 2500.0, sr * 0.5, stride=1)
    del u_hf
    empty_device_cache(dev)

    return new_lf, new_mf, new_hf


@torch.no_grad()
def apply_ltas_hf_tilt_balance(
    z_hf: Tensor,
    y_hf: Tensor,
    *,
    sr: float = 48_000.0,
) -> Tensor:
    """Applies a mild 1/3-octave LTAS energy balance (`[0.85, 1.18]`, max +/- 1.4 dB) above 2500 Hz
    to compensate for least-squares attenuation ($r^2 < 1$) in low-energy high-frequency bands.
    """
    l_seq = z_hf.shape[0]
    freqs = torch.fft.rfftfreq(l_seq, d=1.0 / sr, device=z_hf.device)
    centers_hz = (2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0, 12500.0, 16000.0)
    two_1_6 = 2.0 ** (1.0 / 6.0)

    spec_p = torch.fft.rfft(z_hf.to(torch.float32), n=l_seq, dim=0)
    spec_t = torch.fft.rfft(y_hf.to(torch.float32), n=l_seq, dim=0)
    pow_p = spec_p.real * spec_p.real + spec_p.imag * spec_p.imag
    pow_t = spec_t.real * spec_t.real + spec_t.imag * spec_t.imag

    gain_curve = torch.ones_like(pow_p, dtype=torch.float32)
    for fc in centers_hz:
        f_lo = fc / two_1_6
        f_hi = fc * two_1_6
        mask = (freqs >= f_lo) & (freqs < f_hi)
        if not torch.any(mask):
            continue
        ep = torch.sum(pow_p[mask], dim=0) + 1.0e-18
        et = torch.sum(pow_t[mask], dim=0) + 1.0e-18
        g = torch.clamp((et / ep) ** 0.45, 0.85, 1.18).to(torch.float32)
        gain_curve[mask] = g.unsqueeze(0)

    spec_p.mul_(gain_curve)
    del spec_t, pow_p, pow_t, gain_curve
    out = torch.fft.irfft(spec_p, n=l_seq, dim=0).to(z_hf.dtype)
    return project_fft_band(out, 2500.0, sr * 0.5, sr=sr)


@torch.no_grad()
def build_stage1_fuzz_basis(
    x: Tensor,
    sr: float = 48_000.0,
    *,
    return_chunked: bool = False,
) -> Tensor | ColumnChunkedMatrix:
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

    chunks: list[Tensor] = []
    for c0 in range(0, len(nl_cols), MAX_CHUNK_COLS):
        chunks.append(torch.stack(nl_cols[c0 : c0 + MAX_CHUNK_COLS], dim=-1))
    del nl_cols
    cm = ColumnChunkedMatrix(chunks)
    if return_chunked:
        return cm
    return cm.slice_rows(0, cm.l_tot)


@torch.no_grad()
def build_refinement_fuzz_basis(
    cur_lf: Tensor,
    cur_mf: Tensor,
    cur_hf: Tensor,
    *,
    include_wavefolder: bool = True,
    n_modes: int = 8,
    solver: HostDenseSolver | None = None,
    return_chunked: bool = False,
    sr: float = 48_000.0,
) -> Tensor | ColumnChunkedMatrix:
    """Chunked refinement basis builder (`< 925 MB` max single tensor on `cur_lf.device`)."""
    host_solver = solver if solver is not None else HostDenseSolver()
    z_prev = cur_lf + cur_mf + cur_hf
    z_lm = cur_lf + cur_mf
    l_seq, m_col = z_prev.shape

    z_glob = (
        z_prev / (torch.sqrt(torch.mean(z_prev * z_prev, dim=0, keepdim=True) + 1.0e-12)) * 0.20
    )
    z_lm_glob = z_lm / (torch.sqrt(torch.mean(z_lm * z_lm, dim=0, keepdim=True) + 1.0e-12)) * 0.20
    del z_prev, z_lm

    if m_col > n_modes:
        svd_st = max(1, l_seq // 32_768)
        proj = host_solver.fit_target_manifold_projector(z_glob[::svd_st], n_modes)
        zg = z_glob @ proj
        z_lmg = z_lm_glob @ proj
        del z_glob, z_lm_glob, proj
    else:
        zg = z_glob
        z_lmg = z_lm_glob

    env_loc = torch.sqrt(fast_1pole_lpf(zg * zg, 6.0, sr=sr) + 1.0e-6)
    z_loc = 0.20 * zg / (env_loc + 0.015)
    del env_loc
    env_lmloc = torch.sqrt(fast_1pole_lpf(z_lmg * z_lmg, 6.0, sr=sr) + 1.0e-6)
    z_lmloc = 0.20 * z_lmg / (env_lmloc + 0.015)
    del env_lmloc

    z_hp300 = zg - fast_1pole_lpf(zg, 300.0, sr=sr)
    z_hp1200 = zg - fast_1pole_lpf(zg, 1200.0, sr=sr)
    ez = fast_1pole_lpf(torch.abs(zg), 25.0, sr=sr)

    sigs = (zg, z_lmg, z_loc, z_lmloc, z_hp300, zg - 0.35 * ez)
    blocks_list: list[Tensor] = [zg, z_lmg, z_loc, z_lmloc, z_hp300, z_hp1200]
    del z_hp1200

    for sig_s in sigs:
        for g in (1.5, 5.0, 18.0, 65.0, 250.0):
            tg = torch.tanh(g * sig_s)
            blocks_list.append(tg)
            tg_sq = tg * tg
            del tg
            blocks_list.append(tg_sq - fast_1pole_lpf(tg_sq, 15.0, sr=sr))
            del tg_sq
        if include_wavefolder:
            for g_wf in (2.5, 8.0, 25.0):
                blocks_list.append(torch.sin(g_wf * sig_s))

    cm = ColumnChunkedMatrix.from_tensors(blocks_list)
    del blocks_list
    if return_chunked:
        return cm
    return cm.slice_rows(0, cm.l_tot)


@torch.no_grad()
def run_4stage_orthogonal_fft_cascade(
    x: Tensor,
    y_full: Tensor,
    *,
    sr: float,
    on_stage_progress: Callable[[int, str], None] | None,
    report_step: Callable[[int, Tensor, Tensor, Tensor], Tensor],
    solver: HostDenseSolver | None = None,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]:
    """Executes the 4-Stage Parseval Commuted 3-Band Orthogonal FFT Decoupled Comparator &
    Wavefolder Cascade, returning `(z1_full, z2_full, z3_full, d4_tot, z4_lf, z4_mf, z4_hf, z3_hf)`.
    """
    host_solver = solver if solver is not None else HostDenseSolver()
    l_tot = x.shape[0]

    y_lf_fft = project_fft_band(y_full, 0.0, 250.0, sr=sr)
    y_mf_fft = project_fft_band(y_full, 250.0, 2500.0, sr=sr)
    y_hf_fft = project_fft_band(y_full, 2500.0, sr * 0.5, sr=sr)

    win12, win8, win6, win4 = build_orthogonal_fft_windows(l_tot)
    e_cum = compute_cumulative_energy_state(x, sr=sr)
    bw12 = CausalBsplineWindowSchedule(e_cum, win12, overlap=192)
    bw8 = CausalBsplineWindowSchedule(e_cum, win8, overlap=160)
    bw6 = CausalBsplineWindowSchedule(e_cum, win6, overlap=128)
    bw4 = CausalBsplineWindowSchedule(e_cum, win4, overlap=96)

    cur_lf = torch.zeros_like(y_lf_fft)
    cur_mf = torch.zeros_like(y_mf_fft)
    cur_hf = torch.zeros_like(y_hf_fft)

    if on_stage_progress is not None:
        on_stage_progress(
            1, "Calibrating Stage 1 (V1A Preamp, Comparator & 3-Band Orthogonal Cascade)"
        )
    u1_cm = build_stage1_fuzz_basis(x, sr=sr, return_chunked=True)
    cur_lf, cur_mf, cur_hf = solve_3band_decoupled_window_stage(
        u1_cm,
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
        solver=host_solver,
        sr=sr,
    )
    del u1_cm, bw12
    gc.collect()
    z1_full = cur_lf + cur_mf + cur_hf
    z1_b_lf, z1_b_mf, z1_b_hf = split_bezout_3band(z1_full, sr=sr)
    report_step(1, z1_b_lf, z1_b_mf, z1_b_hf)
    del z1_b_lf, z1_b_mf, z1_b_hf

    if on_stage_progress is not None:
        on_stage_progress(2, "Calibrating Stage 2 (Interstage Comparator & Wavefolder Cascade)")
    u2_cm = build_refinement_fuzz_basis(
        cur_lf,
        cur_mf,
        cur_hf,
        include_wavefolder=True,
        n_modes=8,
        solver=host_solver,
        return_chunked=True,
        sr=sr,
    )
    cur_lf, cur_mf, cur_hf = solve_3band_decoupled_window_stage(
        u2_cm,
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
        solver=host_solver,
        sr=sr,
    )
    del u2_cm, bw8
    gc.collect()
    z2_full = cur_lf + cur_mf + cur_hf
    z2_b_lf, z2_b_mf, z2_b_hf = split_bezout_3band(z2_full, sr=sr)
    report_step(2, z2_b_lf, z2_b_mf, z2_b_hf)
    del z2_b_lf, z2_b_mf, z2_b_hf

    if on_stage_progress is not None:
        on_stage_progress(
            3, "Calibrating Stage 3 (Power Section, Wavefolder & 48 kHz Reference Polish)"
        )
    u3_cm = build_refinement_fuzz_basis(
        cur_lf,
        cur_mf,
        cur_hf,
        include_wavefolder=True,
        n_modes=8,
        solver=host_solver,
        return_chunked=True,
        sr=sr,
    )
    cur_lf, cur_mf, cur_hf = solve_3band_decoupled_window_stage(
        u3_cm,
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
        solver=host_solver,
        sr=sr,
    )
    del u3_cm, bw6
    gc.collect()
    z3_full = cur_lf + cur_mf + cur_hf
    z3_lf, z3_mf, z3_hf = split_bezout_3band(z3_full, sr=sr)
    report_step(3, z3_lf, z3_mf, z3_hf)
    del z3_lf, z3_mf

    u4_cm = build_refinement_fuzz_basis(
        cur_lf,
        cur_mf,
        cur_hf,
        include_wavefolder=True,
        n_modes=8,
        solver=host_solver,
        return_chunked=True,
        sr=sr,
    )
    cur_lf, cur_mf, cur_hf = solve_3band_decoupled_window_stage(
        u4_cm,
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
        solver=host_solver,
        sr=sr,
    )
    del u4_cm, bw4, e_cum, y_lf_fft, y_mf_fft
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
