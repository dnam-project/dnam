"""3-Stage Causal Selective State-Space Tube/Solid-State Cascade Solver (`V1A/Pre -> ToneStack ->
V1B/Driver -> PowerTube/OT -> 48 kHz Quadrature Hysteresis Polish`).

All bulk $O(L)$ filtering, nonlinear basis generation, and split-precision Gram accumulation run
on the accelerator using 32-bit tensors (`float32`, `int32`, `bfloat16`, `complex64`), while small
$N \\times N$ normal equations and subspace eigendecompositions are solved in batched host-CPU
`float64` via [`HostDenseSolver`]. Wide stage bases are stored and projected in column chunks
(`<= 160` columns per tensor) so no single tensor exceeds `max_tensor_bytes` (`1 GiB` on MPS).
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from dnam_trainer.training.accelerator import empty_device_cache, free_tensor_storage
from dnam_trainer.training.cascade_primitives import (
    build_fast_refinement_gates,
    fast_1pole_lpf,
    fast_poly_adaa_batch,
    fast_poly_inst_batch,
    fast_wavefolder_inst_batch,
    solve_lti_cabinet_and_space_stage_with_ir,
    split_bezout_3band,
)
from dnam_trainer.training.numerics import (
    ExactGramAccumulator,
    HostDenseSolver,
    HostSolveSpec,
)
from dnam_trainer.training.windowing import (
    CausalBsplineWindowSchedule,
    SegmentationWindowStats,
    build_stage_window_grids,
    compute_cumulative_energy_state,
    compute_segmentation_window_stats,
    segment_regimes_dp,
)

MAX_CHUNK_COLS: int = 160


class ColumnChunkedMatrix:
    """Stores a `(L, C)` `float32` matrix as a list of contiguous column-chunk tensors, each with
    at most `max_chunk_cols` (`160`) columns (`<= 921.6 MB` at 30 s, strictly `< 1 GiB`).
    """

    def __init__(
        self,
        chunks: Sequence[Tensor],
        *,
        max_chunk_cols: int = MAX_CHUNK_COLS,
    ) -> None:
        self.max_chunk_cols = max(16, max_chunk_cols)
        self.chunks: list[Tensor] = [c for c in chunks if c.shape[1] > 0]
        if not self.chunks:
            raise ValueError("ColumnChunkedMatrix requires at least one non-empty column chunk")
        self.l_tot = self.chunks[0].shape[0]
        self.device = self.chunks[0].device
        self.dtype = self.chunks[0].dtype
        self.col_ranges: list[tuple[int, int]] = []
        cur = 0
        for ch in self.chunks:
            nxt = cur + ch.shape[1]
            self.col_ranges.append((cur, nxt))
            cur = nxt
        self.c_tot = cur

    @property
    def shape(self) -> tuple[int, int]:
        return (self.l_tot, self.c_tot)

    @classmethod
    def from_tensors(
        cls,
        tensors: Sequence[Tensor],
        *,
        max_chunk_cols: int = MAX_CHUNK_COLS,
        free_inputs: bool = False,
    ) -> ColumnChunkedMatrix:
        """Packs `tensors` into column chunks of at most `max_chunk_cols` columns without
        splitting tensors that already fit within `max_chunk_cols`.
        """
        mc = max(16, max_chunk_cols)
        packed: list[Tensor] = []
        pending: list[Tensor] = []
        pending_cols = 0

        def _flush_pending() -> None:
            nonlocal pending_cols
            if not pending:
                return
            if len(pending) == 1:
                packed.append(pending[0].contiguous())
            else:
                cat_t = torch.cat(pending, dim=-1)
                if free_inputs:
                    for p in pending:
                        free_tensor_storage(p)
                packed.append(cat_t)
            pending.clear()
            pending_cols = 0

        for t in tensors:
            w = t.shape[1]
            if w == 0:
                continue
            if w <= mc:
                if pending_cols + w > mc:
                    _flush_pending()
                pending.append(t)
                pending_cols += w
                if pending_cols == mc:
                    _flush_pending()
            else:
                _flush_pending()
                for c0 in range(0, w, mc):
                    c1 = min(w, c0 + mc)
                    packed.append(t[:, c0:c1].contiguous())
                if free_inputs:
                    free_tensor_storage(t)

        _flush_pending()
        return cls(packed, max_chunk_cols=mc)

    def slice_rows(self, s0: int, s1: int, step: int = 1) -> Tensor:
        """Materializes rows `[s0:s1:step]` across all column chunks into a 2-D tensor."""
        if len(self.chunks) == 1:
            return self.chunks[0][s0:s1:step]
        return torch.cat([ch[s0:s1:step] for ch in self.chunks], dim=-1)

    def matmul(self, w: Tensor, *, free_self: bool = False) -> Tensor:
        """Computes `self @ w` where `w` has shape `(c_tot, k_out)`."""
        k_out = w.shape[1]
        out = torch.zeros((self.l_tot, k_out), device=self.device, dtype=self.dtype)
        for ch, (c0, c1) in zip(self.chunks, self.col_ranges, strict=True):
            out.add_(ch @ w[c0:c1])
            if free_self:
                free_tensor_storage(ch)
        if free_self:
            self.chunks.clear()
            empty_device_cache(self.device)
        return out

    def free(self) -> None:
        """Releases device memory of all column chunks."""
        for ch in self.chunks:
            free_tensor_storage(ch)
        self.chunks.clear()
        empty_device_cache(self.device)


def _ensure_chunked(u: Tensor | ColumnChunkedMatrix) -> ColumnChunkedMatrix:
    if isinstance(u, ColumnChunkedMatrix):
        return u
    return ColumnChunkedMatrix.from_tensors([u])


@torch.no_grad()
def _project_nonlinear_batches_chunked(
    bands_tc: Tensor,
    drives: Sequence[float],
    mode: str,
    max_cols: int,
    solver: HostDenseSolver,
    *,
    sr: float = 48_000.0,
    cached_proj: Tensor | None = None,
) -> tuple[list[Tensor], Tensor]:
    """Evaluates `fast_poly_adaa_batch`, `fast_poly_inst_batch`, or `fast_wavefolder_inst_batch`
    in 8-band batches (`<= 80` output columns = `<= 460 MB` at 30 s) and projects onto the top
    `max_cols` orthonormal principal components without ever allocating the full `(L, C_raw)`
    matrix.

    Returns `(projected_chunks, projector_matrix)`.
    """
    l_seq, c_b = bands_tc.shape
    n_d = len(drives)
    c_raw = c_b * n_d * 2
    dev = bands_tc.device

    def _eval_band_slice(b0: int, b1: int) -> Tensor:
        sub_b = bands_tc[:, b0:b1]
        if mode == "adaa":
            return fast_poly_adaa_batch(sub_b, drives, sr=sr)
        if mode == "inst":
            return fast_poly_inst_batch(sub_b, drives, sr=sr)
        if mode == "wavefolder":
            return fast_wavefolder_inst_batch(sub_b, drives, sr=sr)
        raise ValueError(f"Unknown nonlinear batch mode {mode!r}")

    if c_raw <= max_cols and l_seq * c_raw * 4 <= 900_000_000:
        full_out = _eval_band_slice(0, c_b)
        eye = torch.eye(c_raw, device=dev, dtype=torch.float32)
        return ColumnChunkedMatrix.from_tensors([full_out]).chunks, eye

    out_cols = min(c_raw, max_cols)
    if cached_proj is not None and cached_proj.shape == (c_raw, out_cols):
        proj = cached_proj.to(device=dev, dtype=torch.float32)
    else:
        stride = max(1, l_seq // 16_384)
        sub_parts: list[Tensor] = []
        for b0 in range(0, c_b, 8):
            b1 = min(c_b, b0 + 8)
            blk = _eval_band_slice(b0, b1)
            sub_parts.append(blk[::stride].clone())
            del blk
        sub_raw = torch.cat(sub_parts, dim=-1)
        del sub_parts
        proj = solver.fit_orthonormal_projector(sub_raw, out_cols)
        del sub_raw

    # Allocate output chunks of <= MAX_CHUNK_COLS columns
    out_chunks: list[Tensor] = []
    out_ranges: list[tuple[int, int]] = []
    for k0 in range(0, out_cols, MAX_CHUNK_COLS):
        k1 = min(out_cols, k0 + MAX_CHUNK_COLS)
        out_chunks.append(torch.zeros((l_seq, k1 - k0), device=dev, dtype=torch.float32))
        out_ranges.append((k0, k1))

    col_cursor = 0
    for b0 in range(0, c_b, 8):
        b1 = min(c_b, b0 + 8)
        blk = _eval_band_slice(b0, b1)
        w_blk = blk.shape[1]
        p_slice = proj[col_cursor : col_cursor + w_blk]
        for ch_out, (k0, k1) in zip(out_chunks, out_ranges, strict=True):
            ch_out.add_(blk @ p_slice[:, k0:k1])
        col_cursor += w_blk
        del blk

    empty_device_cache(dev)
    return out_chunks, proj


@torch.no_grad()
def _project_chunked_matrix(
    cm: ColumnChunkedMatrix,
    max_cols: int,
    solver: HostDenseSolver,
    *,
    free_input: bool = True,
) -> ColumnChunkedMatrix:
    """Projects a `ColumnChunkedMatrix` onto its top `max_cols` principal components."""
    l_seq, c_in = cm.shape
    if c_in <= max_cols:
        return cm
    stride = max(1, l_seq // 16_384)
    sub_mat = cm.slice_rows(0, l_seq, stride)
    proj = solver.fit_orthonormal_projector(sub_mat, max_cols)
    del sub_mat

    out_chunks: list[Tensor] = []
    out_ranges: list[tuple[int, int]] = []
    for k0 in range(0, max_cols, MAX_CHUNK_COLS):
        k1 = min(max_cols, k0 + MAX_CHUNK_COLS)
        out_chunks.append(torch.zeros((l_seq, k1 - k0), device=cm.device, dtype=cm.dtype))
        out_ranges.append((k0, k1))

    for ch_in, (c0, c1) in zip(cm.chunks, cm.col_ranges, strict=True):
        p_in = proj[c0:c1]
        for ch_out, (k0, k1) in zip(out_chunks, out_ranges, strict=True):
            ch_out.add_(ch_in @ p_in[:, k0:k1])
        if free_input:
            free_tensor_storage(ch_in)
    if free_input:
        cm.chunks.clear()
        empty_device_cache(cm.device)

    return ColumnChunkedMatrix(out_chunks)


def _subsampled_lpf_and_pre_chunked(
    u_cm: ColumnChunkedMatrix,
    fc: float,
    stride: int,
    sr: float = 48_000.0,
) -> tuple[ColumnChunkedMatrix, ColumnChunkedMatrix]:
    chunks_sub: list[Tensor] = []
    chunks_pre: list[Tensor] = []
    for ch in u_cm.chunks:
        c_ch = ch.shape[1]
        for c0 in range(0, c_ch, 64):
            c1 = min(c_ch, c0 + 64)
            uf = fast_1pole_lpf(ch[:, c0:c1], fc, sr=sr)
            chunks_sub.append(uf[::stride].clone())
            chunks_pre.append((uf[1::stride] - 0.85 * uf[:-1:stride]).clone())
            del uf
    return (
        ColumnChunkedMatrix(chunks_sub),
        ColumnChunkedMatrix(chunks_pre),
    )


def _eval_lti_commuted_projection_chunked(
    u_cm: ColumnChunkedMatrix,
    wa_3d: Tensor,
    *,
    s0: int = 0,
    s1: int | None = None,
    sr: float = 48_000.0,
    free_u: bool = False,
) -> Tensor:
    l_end = u_cm.l_tot if s1 is None else s1
    l_s = l_end - s0
    n_tgt = wa_3d.shape[2]
    dev = u_cm.device

    def _mm_slice(w_2d: Tensor, *, free_chunks: bool = False) -> Tensor:
        if s0 == 0 and l_end == u_cm.l_tot:
            return u_cm.matmul(w_2d, free_self=free_chunks)
        out = torch.zeros((l_s, n_tgt), device=dev, dtype=torch.float32)
        for ch, (c0, c1) in zip(u_cm.chunks, u_cm.col_ranges, strict=True):
            out.add_(ch[s0:l_end] @ w_2d[c0:c1])
        return out

    z = _mm_slice(wa_3d[0])
    z.add_(fast_1pole_lpf(_mm_slice(wa_3d[1]), 80.0, sr=sr))
    z.add_(fast_1pole_lpf(_mm_slice(wa_3d[2]), 400.0, sr=sr))
    z.add_(fast_1pole_lpf(_mm_slice(wa_3d[3]), 2000.0, sr=sr))
    if l_s > 2:
        zp2 = _mm_slice(wa_3d[4])
        z[: l_s - 2].add_(zp2[2:])
        del zp2
        zm2 = _mm_slice(wa_3d[5], free_chunks=free_u)
        z[2:].add_(zm2[: l_s - 2])
        del zm2
    elif free_u:
        u_cm.free()
    return z


def _build_3band_synthesis_accumulator(
    phi_s: Tensor,
    y_lf_s: Tensor,
    y_mf_s: Tensor,
    y_hf_s: Tensor,
    w_lf_s: float,
    w_mf_s: float,
    w_hf_s: float,
    *,
    stride: int,
    sr: float,
) -> tuple[ExactGramAccumulator, Tensor | None, Tensor | None, Tensor | None]:
    """Splits `phi_s` into 3 Bezout bands in 16-column chunks and accumulates the band-weighted
    normal equation in `ExactGramAccumulator` without materializing full-length multi-band copies.
    """
    c_tot = phi_s.shape[1]
    lf_subs: list[Tensor] = []
    mf_subs: list[Tensor] = []
    hf_subs: list[Tensor] = []
    for c0 in range(0, c_tot, 16):
        c1 = min(c_tot, c0 + 16)
        lf_c, mf_c, hf_c = split_bezout_3band(phi_s[:, c0:c1], sr=sr)
        lf_subs.append(lf_c[::stride].clone())
        mf_subs.append(mf_c[::stride].clone())
        hf_subs.append(hf_c[::stride].clone())
        del lf_c, mf_c, hf_c
    p_lf_sub = torch.cat(lf_subs, dim=-1)
    p_mf_sub = torch.cat(mf_subs, dim=-1)
    p_hf_sub = torch.cat(hf_subs, dim=-1)
    del lf_subs, mf_subs, hf_subs
    acc = ExactGramAccumulator(c_tot, 1, device=phi_s.device)
    acc.add(p_lf_sub, y_lf_s[::stride], weight=w_lf_s)
    acc.add(p_mf_sub, y_mf_s[::stride], weight=w_mf_s)
    acc.add(p_hf_sub, y_hf_s[::stride], weight=w_hf_s)
    del p_lf_sub, p_mf_sub, p_hf_sub
    return acc, None, None, None


@torch.no_grad()
def solve_selective_stage_3band(
    u_stage: Tensor | ColumnChunkedMatrix,
    y7_full: Tensor,
    y7_lf: Tensor,
    y7_mf: Tensor,
    y7_hf: Tensor,
    bspline_weights_win: Tensor | CausalBsplineWindowSchedule,
    bspline_weights_reg: Tensor | CausalBsplineWindowSchedule,
    ref_gates: Sequence[Tensor],
    win_bounds: Sequence[tuple[int, int]],
    regimes: Sequence[tuple[int, int]],
    *,
    res_weights: Tensor | None = None,
    z_lti_bands: tuple[Tensor, Tensor, Tensor] | None = None,
    sig_lf: Tensor | None = None,
    sig_mf: Tensor | None = None,
    sig_hf: Tensor | None = None,
    solver: HostDenseSolver | None = None,
    free_u_stage: bool = False,
    sr: float = 48_000.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Solves one physical tube/solid-state stage across `L_tot` samples with Pre-Emphasized
    3-Band (`LF`, `MF`, `HF`) Normal Equations in Step A, Step B, and Step C.
    """
    host_solver = solver if solver is not None else HostDenseSolver()
    u_cm = _ensure_chunked(u_stage)
    l_tot, c_u = u_cm.shape
    dev = u_cm.device
    m_col = y7_full.shape[1]
    if l_tot > 600_000:
        stride_scale = max(1, (l_tot + 240_000) // 480_000)
        stride_a = 4 * stride_scale
        stride_b = 2 * stride_scale
    else:
        stride_a = 4 if l_tot >= 32_768 else 1
        stride_b = 2 if l_tot >= 16_384 else 1
    pad = 8

    y_full_f32 = y7_full.to(device=dev, dtype=torch.float32)
    y_lf_f32 = y7_lf.to(device=dev, dtype=torch.float32)
    y_mf_f32 = y7_mf.to(device=dev, dtype=torch.float32)
    y_hf_f32 = y7_hf.to(device=dev, dtype=torch.float32)

    if m_col > 14:
        svd_st = max(1, l_tot // 32_768)
        ylf_std = torch.sqrt(torch.mean(y_lf_f32 * y_lf_f32, dim=0, keepdim=True) + 1.0e-12)
        ymf_std = torch.sqrt(torch.mean(y_mf_f32 * y_mf_f32, dim=0, keepdim=True) + 1.0e-12)
        yhf_std = torch.sqrt(torch.mean(y_hf_f32 * y_hf_f32, dim=0, keepdim=True) + 1.0e-12)
        y_bal = (
            y_lf_f32[::svd_st] / ylf_std
            + y_mf_f32[::svd_st] / ymf_std
            + y_hf_f32[::svd_st] / yhf_std
        )
        proj_14 = host_solver.fit_target_manifold_projector(y_bal, 14, res_weights=res_weights)
        y_std = torch.sqrt(torch.mean(y_full_f32 * y_full_f32, dim=0, keepdim=True) + 1.0e-12)
        ya_full = (y_full_f32 / y_std) @ proj_14
        ya_hf = (y_hf_f32 / yhf_std) @ proj_14
        del y_bal, proj_14
    else:
        ya_full = y_full_f32
        ya_hf = y_hf_f32

    k_a = ya_full.shape[1]
    n_pre_a = (l_tot - 1 + stride_a - 1) // stride_a

    def _chunk_shifted_sub(qs: int, qe: int, shift: int) -> Tensor:
        if shift == 0:
            return u_cm.slice_rows(qs * stride_a, qe * stride_a, stride_a)
        if shift > 0:
            sub = u_cm.slice_rows(qs * stride_a + shift, qe * stride_a + shift, stride_a)
            if sub.shape[0] < (qe - qs):
                sub = F.pad(sub.unsqueeze(0), (0, 0, 0, (qe - qs) - sub.shape[0])).squeeze(0)
            return sub[: qe - qs]
        sh_neg = -shift
        start_idx = qs * stride_a - sh_neg
        end_idx = qe * stride_a - sh_neg
        if start_idx >= 0:
            return u_cm.slice_rows(start_idx, end_idx, stride_a)
        pad_rows = (-start_idx + stride_a - 1) // stride_a
        rem_start = start_idx + pad_rows * stride_a
        sliced = u_cm.slice_rows(rem_start, max(rem_start, end_idx), stride_a)
        padded = F.pad(sliced.unsqueeze(0), (0, 0, pad_rows, 0)).squeeze(0)
        return padded[: qe - qs]

    def _chunk_shifted_pre(qs: int, qe: int, shift: int) -> Tensor:
        if shift == 0:
            s1 = u_cm.slice_rows(qs * stride_a + 1, qe * stride_a + 1, stride_a)
            s0 = u_cm.slice_rows(qs * stride_a, min(l_tot - 1, qe * stride_a), stride_a)
            m_len = min(s1.shape[0], s0.shape[0], qe - qs)
            if m_len == qe - qs:
                return s1[:m_len] - 0.85 * s0[:m_len]
            pre = torch.zeros((qe - qs, c_u), dtype=u_cm.dtype, device=dev)
            pre[:m_len] = s1[:m_len] - 0.85 * s0[:m_len]
            return pre
        if shift > 0:
            s1 = u_cm.slice_rows(qs * stride_a + shift + 1, qe * stride_a + shift + 1, stride_a)
            s0 = u_cm.slice_rows(
                qs * stride_a + shift, min(l_tot - 1, qe * stride_a + shift), stride_a
            )
            m_len = min(s1.shape[0], s0.shape[0], qe - qs)
            if m_len == qe - qs:
                return s1[:m_len] - 0.85 * s0[:m_len]
            pre = torch.zeros((qe - qs, c_u), dtype=u_cm.dtype, device=dev)
            if m_len > 0:
                pre[:m_len] = s1[:m_len] - 0.85 * s0[:m_len]
            return pre
        sh_neg = -shift
        start_0 = qs * stride_a - sh_neg
        end_0 = min(l_tot - 1 - sh_neg, qe * stride_a - sh_neg)
        if start_0 >= 0:
            s1 = u_cm.slice_rows(start_0 + 1, end_0 + 1, stride_a)
            s0 = u_cm.slice_rows(start_0, end_0, stride_a)
            m_len = min(s1.shape[0], s0.shape[0], qe - qs)
            if m_len == qe - qs:
                return s1[:m_len] - 0.85 * s0[:m_len]
            pre = torch.zeros((qe - qs, c_u), dtype=u_cm.dtype, device=dev)
            if m_len > 0:
                pre[:m_len] = s1[:m_len] - 0.85 * s0[:m_len]
            return pre
        pad_rows = (-start_0 + stride_a - 1) // stride_a
        rem_0 = start_0 + pad_rows * stride_a
        s1 = u_cm.slice_rows(rem_0 + 1, max(rem_0 + 1, end_0 + 1), stride_a)
        s0 = u_cm.slice_rows(rem_0, max(rem_0, end_0), stride_a)
        m_len = min(s1.shape[0], s0.shape[0], max(0, (qe - qs) - pad_rows))
        pre = torch.zeros((qe - qs, c_u), dtype=u_cm.dtype, device=dev)
        if m_len > 0:
            pre[pad_rows : pad_rows + m_len] = s1[:m_len] - 0.85 * s0[:m_len]
        return pre

    u_sub_l80, u_pre_l80 = _subsampled_lpf_and_pre_chunked(u_cm, 80.0, stride_a, sr=sr)
    u_sub_l400, u_pre_l400 = _subsampled_lpf_and_pre_chunked(u_cm, 400.0, stride_a, sr=sr)
    u_sub_l2000, u_pre_l2000 = _subsampled_lpf_and_pre_chunked(u_cm, 2000.0, stride_a, sr=sr)

    ya_hp600 = ya_full - fast_1pole_lpf(ya_full, 600.0, sr=sr)
    ya_pad = F.pad(ya_full.unsqueeze(0), (0, 0, pad, pad)).squeeze(0)
    ya_hp600_pad = F.pad(ya_hp600.unsqueeze(0), (0, 0, pad, pad)).squeeze(0)
    ya_hp2500_pad = F.pad(ya_hf.unsqueeze(0), (0, 0, pad, pad)).squeeze(0)
    del ya_full, ya_hf, ya_hp600

    tgt_a_sub: list[Tensor] = []
    tgt_a_pre: list[Tensor] = []
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
    acc_a_gl = ExactGramAccumulator(p_a, n_a_tgt, device=dev)
    ov_a = 512 if l_tot >= 32_768 else 0
    burn_a = 2048 if l_tot >= 32_768 else 0
    step_a_specs: list[HostSolveSpec] = []
    regime_slices: list[tuple[int, int, int]] = []

    for r_idx, (r0, r1) in enumerate(regimes):
        sr0 = 0 if r_idx == 0 else max(0, r0 - ov_a)
        er1 = l_tot if r_idx == len(regimes) - 1 else min(l_tot, r1 + ov_a)
        q0 = sr0 // stride_a
        q1 = min(y_multi_a_sub.shape[0], er1 // stride_a)
        q1_pre = min(n_pre_a, q1)

        acc_a_r = ExactGramAccumulator(p_a, n_a_tgt, device=dev)
        for qs in range(q0, q1, 4096):
            qe = min(q1, qs + 4096)
            pa_s = torch.cat(
                [
                    _chunk_shifted_sub(qs, qe, 0),
                    u_sub_l80.slice_rows(qs, qe),
                    u_sub_l400.slice_rows(qs, qe),
                    u_sub_l2000.slice_rows(qs, qe),
                    _chunk_shifted_sub(qs, qe, 2),
                    _chunk_shifted_sub(qs, qe, -2),
                ],
                dim=-1,
            )
            ya_s = y_multi_a_sub[qs:qe]
            acc_a_r.add(pa_s, ya_s, weight=1.0)
            del pa_s, ya_s

        for qs in range(q0, q1_pre, 4096):
            qe = min(q1_pre, qs + 4096)
            pa_p = torch.cat(
                [
                    _chunk_shifted_pre(qs, qe, 0),
                    u_pre_l80.slice_rows(qs, qe),
                    u_pre_l400.slice_rows(qs, qe),
                    u_pre_l2000.slice_rows(qs, qe),
                    _chunk_shifted_pre(qs, qe, 2),
                    _chunk_shifted_pre(qs, qe, -2),
                ],
                dim=-1,
            )
            ya_p = y_multi_a_pre[qs:qe]
            acc_a_r.add(pa_p, ya_p, weight=0.5)
            del pa_p, ya_p

        acc_a_gl.merge(acc_a_r)
        acc_a_r.offload_to_cpu()
        step_a_specs.append(HostSolveSpec(accumulator=acc_a_r, ridge_scale=5e-6, min_ridge=1e-9))
        regime_slices.append((r_idx, sr0, er1))

    acc_a_gl.offload_to_cpu()
    step_a_specs.append(HostSolveSpec(accumulator=acc_a_gl, ridge_scale=2e-6, min_ridge=1e-9))
    del u_sub_l80, u_pre_l80, u_sub_l400, u_pre_l400, u_sub_l2000, u_pre_l2000
    del y_multi_a_sub, y_multi_a_pre
    empty_device_cache(dev)
    step_a_weights = host_solver.solve_batch(
        step_a_specs, pass_name="stage_step_a", return_device=dev
    )
    del step_a_specs, acc_a_gl

    z_a_loc = torch.zeros((l_tot, n_a_tgt), device=dev, dtype=torch.float32)
    for (r_idx, sr0, er1), wa in zip(regime_slices, step_a_weights[:-1], strict=True):
        s_eval = max(0, sr0 - burn_a)
        b_off = sr0 - s_eval
        z_r_slice = _eval_lti_commuted_projection_chunked(
            u_cm,
            wa.view(6, c_u, n_a_tgt),
            s0=s_eval,
            s1=er1,
            sr=sr,
        )[b_off:]
        if isinstance(bspline_weights_reg, CausalBsplineWindowSchedule):
            reg_w = bspline_weights_reg.window_weight(r_idx, sr0, er1)
        else:
            reg_w = bspline_weights_reg[sr0:er1, r_idx : r_idx + 1]
        z_a_loc[sr0:er1].add_(reg_w * z_r_slice)
        del z_r_slice

    wa_gl = step_a_weights[-1]
    del step_a_weights
    z_a_gl = _eval_lti_commuted_projection_chunked(
        u_cm,
        wa_gl.view(6, c_u, n_a_tgt),
        sr=sr,
        free_u=free_u_stage,
    )
    del wa_gl
    if free_u_stage and isinstance(u_stage, Tensor):
        free_tensor_storage(u_stage)
        gc.collect()

    za_core_raw = torch.cat(
        [
            z_a_loc[:, 2 * k_a : 3 * k_a],
            z_a_loc[:, 6 * k_a : 7 * k_a],
            z_a_loc[:, 8 * k_a : 9 * k_a],
            z_a_gl[:, 2 * k_a : 3 * k_a],
            z_a_gl[:, 6 * k_a : 7 * k_a],
            z_a_gl[:, 8 * k_a : 9 * k_a],
        ],
        dim=-1,
    )
    is_long_seq = l_tot > 600_000
    max_core = 36 if is_long_seq else (48 if m_col > 14 else 42)
    if za_core_raw.shape[1] > max_core:
        za_core = host_solver.project_orthonormal_basis(za_core_raw, max_core)
    else:
        za_core = za_core_raw.clone()
    del za_core_raw

    z_a_cm = ColumnChunkedMatrix.from_tensors([z_a_loc, z_a_gl])
    del z_a_loc, z_a_gl
    max_z_a = 144 if is_long_seq else 180
    if m_col > 14 and z_a_cm.shape[1] > max_z_a:
        z_a_cm = _project_chunked_matrix(z_a_cm, max_z_a, host_solver, free_input=True)
    c_core = za_core.shape[1]

    post_lpfs = (45.0, 150.0, 500.0, 1600.0, 4500.0)
    delays = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8)
    za_lpf_list = [fast_1pole_lpf(za_core, fc, sr=sr) for fc in post_lpfs]
    za_core_pad = F.pad(za_core.unsqueeze(0), (0, 0, pad, pad)).squeeze(0)
    del za_core
    empty_device_cache(dev)

    n_b_tgt = 3 * m_col
    n_z_a = z_a_cm.shape[1]
    n_b = n_z_a + len(post_lpfs) * c_core + len(delays) * c_core
    acc_b_gl = ExactGramAccumulator(n_b, n_b_tgt, device=dev)
    n_wins = len(win_bounds)
    overlap = 256 if l_tot >= 32_768 else 0
    z_spline_21 = torch.zeros((l_tot, n_b_tgt), device=dev, dtype=torch.float32)
    win_batch_size = 4 if l_tot > 600_000 else 8

    def _build_pk_win(s_w: int, e_w: int) -> Tensor:
        b_win_list = [z_a_cm.slice_rows(s_w, e_w), *[zl[s_w:e_w] for zl in za_lpf_list]]
        for d in delays:
            b_win_list.append(za_core_pad[pad + s_w - d : pad + e_w - d])
        return torch.cat(b_win_list, dim=-1)

    for b0 in range(0, n_wins, win_batch_size):
        b1 = min(n_wins, b0 + win_batch_size)
        batch_b_specs: list[HostSolveSpec] = []
        batch_pk: list[tuple[int, int, int, Tensor]] = []
        for w_idx in range(b0, b1):
            k0, k1 = win_bounds[w_idx]
            s_w = 0 if w_idx == 0 else max(0, k0 - overlap)
            e_w = l_tot if w_idx == n_wins - 1 else min(l_tot, k1 + overlap)
            pk_full = _build_pk_win(s_w, e_w)
            y_win_b = torch.cat([y_lf_f32[s_w:e_w], y_mf_f32[s_w:e_w], y_hf_f32[s_w:e_w]], dim=-1)

            acc_b_w = ExactGramAccumulator(n_b, n_b_tgt, device=dev)
            acc_b_w.add(pk_full[::stride_b], y_win_b[::stride_b], weight=1.0)
            pk_pre = pk_full[1::stride_b] - 0.85 * pk_full[:-1:stride_b]
            yk_pre = y_win_b[1::stride_b] - 0.85 * y_win_b[:-1:stride_b]
            acc_b_w.add(pk_pre, yk_pre, weight=0.5)
            del y_win_b, pk_pre, yk_pre

            acc_b_gl.merge(acc_b_w)
            acc_b_w.offload_to_cpu()
            batch_b_specs.append(
                HostSolveSpec(accumulator=acc_b_w, ridge_scale=1e-5, min_ridge=1e-9)
            )
            batch_pk.append((w_idx, s_w, e_w, pk_full))

        batch_b_weights = host_solver.solve_batch(
            batch_b_specs, pass_name="stage_step_b", return_device=dev
        )
        del batch_b_specs
        for (w_idx, s_w, e_w, pk_full), ww in zip(batch_pk, batch_b_weights, strict=True):
            if isinstance(bspline_weights_win, CausalBsplineWindowSchedule):
                bw_slice = bspline_weights_win.window_weight(w_idx, s_w, e_w)
            else:
                bw_slice = bspline_weights_win[s_w:e_w, w_idx : w_idx + 1]
            z_spline_21[s_w:e_w].add_(bw_slice * (pk_full @ ww))
        del batch_pk, batch_b_weights

    acc_b_gl.offload_to_cpu()
    w_b_gl = host_solver.solve_single(
        acc_b_gl, ridge_scale=2e-6, min_ridge=1e-9, pass_name="stage_step_b_gl", return_device=dev
    )
    del acc_b_gl

    idx = 0
    z_glob_21 = z_a_cm.matmul(w_b_gl[idx : idx + n_z_a], free_self=True)
    idx += n_z_a
    for zl in za_lpf_list:
        z_glob_21.add_(zl @ w_b_gl[idx : idx + c_core])
        free_tensor_storage(zl)
        idx += c_core
    za_lpf_list.clear()
    for d in delays:
        z_glob_21.add_(za_core_pad[pad - d : pad - d + l_tot] @ w_b_gl[idx : idx + c_core])
        idx += c_core
    free_tensor_storage(za_core_pad)
    del w_b_gl
    empty_device_cache(dev)

    # Step C: Per-stem Decoupled 3-Band (`LF`, `MF`, `HF`) Refinement + `bESR` Synthesis
    floor_e = float(l_tot) * (10.0 ** (-60.0 / 10.0))

    def _build_phi_ref(s_idx: int) -> Tensor:
        base_6 = torch.cat([z_spline_21[:, s_idx::m_col], z_glob_21[:, s_idx::m_col]], dim=-1)
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
        return torch.cat(phi_parts, dim=-1)

    step_c_ref_specs: list[HostSolveSpec] = []
    for s_idx in range(m_col):
        phi_ref = _build_phi_ref(s_idx)
        y_s_3b = torch.stack([y_lf_f32[:, s_idx], y_mf_f32[:, s_idx], y_hf_f32[:, s_idx]], dim=-1)
        c_phi = phi_ref.shape[1]

        acc_sub = ExactGramAccumulator(c_phi, 3, device=dev)
        acc_sub.add(phi_ref[::stride_b], y_s_3b[::stride_b], weight=1.0)
        acc_sub.offload_to_cpu()

        phi_pre = phi_ref[1::stride_b] - 0.85 * phi_ref[:-1:stride_b]
        y_pre = y_s_3b[1::stride_b] - 0.85 * y_s_3b[:-1:stride_b]
        acc_pre = ExactGramAccumulator(c_phi, 3, device=dev)
        acc_pre.add(phi_pre, y_pre, weight=1.0)
        acc_pre.offload_to_cpu()
        del phi_ref, phi_pre, y_pre, y_s_3b

        step_c_ref_specs.append(
            HostSolveSpec(
                accumulator=acc_sub,
                secondary_accumulator=acc_pre,
                primary_weight=1.0,
                secondary_weight=0.05,
                rhs_col_slice=(0, 1),
            )
        )
        step_c_ref_specs.append(
            HostSolveSpec(
                accumulator=acc_sub,
                secondary_accumulator=acc_pre,
                primary_weight=1.0,
                secondary_weight=0.50,
                rhs_col_slice=(1, 2),
            )
        )
        step_c_ref_specs.append(
            HostSolveSpec(
                accumulator=acc_sub,
                secondary_accumulator=acc_pre,
                primary_weight=0.02,
                secondary_weight=1.00,
                rhs_col_slice=(2, 3),
            )
        )

    ref_weights = host_solver.solve_batch(
        step_c_ref_specs, pass_name="stage_step_c_ref", return_device=dev
    )
    del step_c_ref_specs

    def _build_m_s(s_idx: int, wr: Tensor) -> Tensor:
        phi_ref = _build_phi_ref(s_idx)
        c3_s = phi_ref @ wr
        del phi_ref
        c_l120 = fast_1pole_lpf(c3_s, 120.0, sr=sr)
        c_l250 = fast_1pole_lpf(c3_s, 250.0, sr=sr)
        c_l250_2 = fast_1pole_lpf(c_l250, 250.0, sr=sr)
        c_l700 = fast_1pole_lpf(c3_s, 700.0, sr=sr)
        c_l1600 = fast_1pole_lpf(c3_s, 1600.0, sr=sr)
        c_l2500 = fast_1pole_lpf(c3_s, 2500.0, sr=sr)
        c_l2500_2 = fast_1pole_lpf(c_l2500, 2500.0, sr=sr)
        c_l5500 = fast_1pole_lpf(c3_s, 5500.0, sr=sr)
        m_gated = [c3_s * g.unsqueeze(-1) for g in ref_gates[:5]]
        return torch.cat(
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

    wr_per_stem: list[Tensor] = []
    step_c_syn_specs: list[HostSolveSpec] = []
    for s_idx in range(m_col):
        wr = torch.cat(
            [
                ref_weights[3 * s_idx],
                ref_weights[3 * s_idx + 1],
                ref_weights[3 * s_idx + 2],
            ],
            dim=-1,
        )
        wr_per_stem.append(wr)
        m_s = _build_m_s(s_idx, wr)
        ylf_s = y_lf_f32[:, s_idx : s_idx + 1]
        ymf_s = y_mf_f32[:, s_idx : s_idx + 1]
        yhf_s = y_hf_f32[:, s_idx : s_idx + 1]
        w_lf_s = (
            1.0 / float(sig_lf[s_idx].item())
            if sig_lf is not None
            else 1.0 / (float(torch.sum(ylf_s * ylf_s).item()) + floor_e)
        )
        w_mf_s = (
            1.0 / float(sig_mf[s_idx].item())
            if sig_mf is not None
            else 1.0 / (float(torch.sum(ymf_s * ymf_s).item()) + floor_e)
        )
        w_hf_s = (
            1.0 / float(sig_hf[s_idx].item())
            if sig_hf is not None
            else 1.0 / (float(torch.sum(yhf_s * yhf_s).item()) + floor_e)
        )
        acc_syn, p_lf, p_mf, p_hf = _build_3band_synthesis_accumulator(
            m_s, ylf_s, ymf_s, yhf_s, w_lf_s, w_mf_s, w_hf_s, stride=stride_b, sr=sr
        )
        acc_syn.offload_to_cpu()
        del m_s, p_lf, p_mf, p_hf
        step_c_syn_specs.append(HostSolveSpec(accumulator=acc_syn))

    del ref_weights
    alpha_weights = host_solver.solve_batch(
        step_c_syn_specs, pass_name="stage_step_c_syn", return_device=dev
    )
    del step_c_syn_specs

    z_lf_cols: list[Tensor] = []
    z_mf_cols: list[Tensor] = []
    z_hf_cols: list[Tensor] = []
    z_tot_cols: list[Tensor] = []
    for s_idx, (wr, alpha_s) in enumerate(zip(wr_per_stem, alpha_weights, strict=True)):
        m_s = _build_m_s(s_idx, wr)
        z_tot_s = (m_s @ alpha_s).squeeze(-1)
        del m_s
        z_lf_s, z_mf_s, z_hf_s = split_bezout_3band(z_tot_s, sr=sr)
        z_lf_cols.append(z_lf_s)
        z_mf_cols.append(z_mf_s)
        z_hf_cols.append(z_hf_s)
        z_tot_cols.append(z_lf_s + z_mf_s + z_hf_s)
        del z_tot_s

    free_tensor_storage(z_spline_21)
    free_tensor_storage(z_glob_21)
    del wr_per_stem, alpha_weights
    del y_full_f32, y_lf_f32, y_mf_f32, y_hf_f32
    return (
        torch.stack(z_tot_cols, dim=-1),
        torch.stack(z_lf_cols, dim=-1),
        torch.stack(z_mf_cols, dim=-1),
        torch.stack(z_hf_cols, dim=-1),
    )


def build_interstage_basis(
    z_prev: Tensor,
    z_earlier: Tensor | None,
    cutoffs: Sequence[float],
    bias_fc: float,
    env_fast_fc: float,
    env_slow_fc: float,
    drives: Sequence[float],
    *,
    res_weights: Tensor | None = None,
    use_power_sag: bool = False,
    solver: HostDenseSolver | None = None,
    return_chunked: bool = False,
    sr: float = 48_000.0,
) -> Tensor | ColumnChunkedMatrix:
    """Vectorized interstage nonlinear basis builder with QR-combined global + residual-weighted
    manifold projection (`< 900 MB` max single tensor).
    """
    host_solver = solver if solver is not None else HostDenseSolver()
    z_f32 = z_prev.to(dtype=torch.float32)
    z_std = torch.sqrt(torch.mean(z_f32 * z_f32, dim=0, keepdim=True) + 1.0e-12)
    z_norm = z_f32 / z_std

    if z_norm.shape[1] > 8:
        l_seq = z_norm.shape[0]
        svd_st = max(1, l_seq // 32_768)
        sub_unw = z_norm[::svd_st]
        n_lin = min(14, z_norm.shape[1])
        n_nl = 7 if z_norm.shape[1] <= 14 else min(8, z_norm.shape[1])
        proj_lin, proj_nl = host_solver.fit_interstage_projector(
            sub_unw, n_lin, n_nl, res_weights=res_weights
        )
        z_lin = z_norm @ proj_lin
        z_nl = z_norm @ proj_nl
        if z_earlier is not None:
            ze_f32 = z_earlier.to(dtype=torch.float32)
            ze_std = torch.sqrt(torch.mean(ze_f32 * ze_f32, dim=0, keepdim=True) + 1.0e-12)
            z_earlier_proj = (ze_f32 / ze_std) @ proj_lin
        else:
            z_earlier_proj = None
        del sub_unw, proj_lin, proj_nl
    elif z_norm.shape[1] > 7:
        l_seq = z_norm.shape[0]
        svd_st = max(1, l_seq // 32_768)
        _, proj_nl = host_solver.fit_interstage_projector(z_norm[::svd_st], 7, 7)
        z_lin = z_f32
        z_nl = z_norm @ proj_nl
        z_earlier_proj = z_earlier.to(dtype=torch.float32) if z_earlier is not None else None
        del proj_nl
    else:
        z_lin = z_f32
        z_nl = z_norm
        z_earlier_proj = z_earlier.to(dtype=torch.float32) if z_earlier is not None else None
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
    is_long_seq = z_prev.shape[0] > 600_000
    max_adaa = (
        200
        if is_long_seq
        else (320 if z_prev.shape[1] > 14 else adaa_bands.shape[1] * len(drives) * 2)
    )
    adaa_chunks, _ = _project_nonlinear_batches_chunked(
        adaa_bands, drives, "adaa", max_adaa, host_solver, sr=sr
    )
    del adaa_bands

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
    max_inst = (
        96
        if is_long_seq
        else (140 if z_prev.shape[1] > 14 else inst_bands.shape[1] * len(drives) * 2)
    )
    inst_chunks, _ = _project_nonlinear_batches_chunked(
        inst_bands, drives, "inst", max_inst, host_solver, sr=sr
    )
    del inst_bands

    parts: list[Tensor] = [z_lin, z_hp600, z_hp2500]
    if z_earlier_proj is not None:
        parts.insert(0, z_earlier_proj)
    parts.extend(adaa_chunks)
    parts.extend(inst_chunks)
    cm = ColumnChunkedMatrix.from_tensors(parts)
    if return_chunked:
        return cm
    return cm.slice_rows(0, cm.l_tot)


@torch.no_grad()
def solve_fullrate_hf_residual_polish(
    z3_full: Tensor,
    z2_full: Tensor,
    r3_lf: Tensor,
    r3_mf: Tensor,
    r3_hf: Tensor,
    sig_lf: Tensor,
    sig_mf: Tensor,
    sig_hf: Tensor,
    bspline_weights_win: Tensor | CausalBsplineWindowSchedule,
    ref_gates: Sequence[Tensor],
    win_bounds: Sequence[tuple[int, int]],
    *,
    z1_full: Tensor | None = None,
    solver: HostDenseSolver | None = None,
    sr: float = 48_000.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """3-Band Quadrature Hysteresis & Wavefolder B-Spline Polish + Step-C Post-Synthesis (`CL-18`
    via on-device `ExactGramAccumulator` + batched host `float64` `HostDenseSolver`).
    """
    host_solver = solver if solver is not None else HostDenseSolver()
    l_tot, m_col = z3_full.shape
    dev = z3_full.device
    is_long_seq = l_tot > 600_000
    if is_long_seq:
        stride_s4 = 2 * max(1, (l_tot + 240_000) // 480_000)
    else:
        stride_s4 = 2 if l_tot >= 16_384 else 1

    r3_lf_f32 = r3_lf.to(device=dev, dtype=torch.float32)
    r3_mf_f32 = r3_mf.to(device=dev, dtype=torch.float32)
    r3_hf_f32 = r3_hf.to(device=dev, dtype=torch.float32)
    sig_lf_f32 = sig_lf.to(device=dev, dtype=torch.float32)
    sig_mf_f32 = sig_mf.to(device=dev, dtype=torch.float32)
    sig_hf_f32 = sig_hf.to(device=dev, dtype=torch.float32)

    dlf3 = torch.sum(r3_lf_f32 * r3_lf_f32, dim=0) / sig_lf_f32
    dmf3 = torch.sum(r3_mf_f32 * r3_mf_f32, dim=0) / sig_mf_f32
    dhf3 = torch.sum(r3_hf_f32 * r3_hf_f32, dim=0) / sig_hf_f32
    res_weight = torch.sqrt((dlf3 + dmf3 + dhf3) / 3.0 + 1.0e-12).unsqueeze(0)

    z1_src = (z1_full if z1_full is not None else z2_full).to(device=dev, dtype=torch.float32)
    z3_f32 = z3_full.to(device=dev, dtype=torch.float32)
    z2_f32 = z2_full.to(device=dev, dtype=torch.float32)
    z3_std = torch.sqrt(torch.mean(z3_f32 * z3_f32, dim=0, keepdim=True) + 1.0e-12)
    z2_std = torch.sqrt(torch.mean(z2_f32 * z2_f32, dim=0, keepdim=True) + 1.0e-12)
    z1_std = torch.sqrt(torch.mean(z1_src * z1_src, dim=0, keepdim=True) + 1.0e-12)
    if m_col > 8:
        svd_st = max(1, l_tot // 32_768)
        z3_w = ((z3_f32 / z3_std) * res_weight)[::svd_st]
        proj_8 = host_solver.fit_target_manifold_projector(z3_w, 8)
        z3_k = (z3_f32 / z3_std) @ proj_8
        z2_k = (z2_f32 / z2_std) @ proj_8
        z1_k = (z1_src / z1_std) @ proj_8
        del z3_w, proj_8
    else:
        z3_k = z3_f32 / z3_std
        z2_k = z2_f32 / z2_std
        z1_k = z1_src / z1_std

    l_80 = fast_1pole_lpf(z3_k, 80.0, sr=sr)
    l_250 = fast_1pole_lpf(z3_k, 250.0, sr=sr)
    h_250 = z3_k - l_250
    h_900 = z3_k - fast_1pole_lpf(z3_k, 900.0, sr=sr)
    h_2500 = z3_k - fast_1pole_lpf(z3_k, 2500.0, sr=sr)
    h_6000 = z3_k - fast_1pole_lpf(z3_k, 6000.0, sr=sr)
    h_s2_1200 = z2_k - fast_1pole_lpf(z2_k, 1200.0, sr=sr)
    h_s1_1200 = z1_k - fast_1pole_lpf(z1_k, 1200.0, sr=sr)
    del z2_k, z1_k, z1_src, z2_f32
    if m_col > 7 and z1_full is not None:
        free_tensor_storage(z1_full)

    e_slow = fast_1pole_lpf(torch.abs(z3_k), 12.0, sr=sr)
    e_low = fast_1pole_lpf(torch.abs(z3_k), 40.0, sr=sr)
    e_fast = fast_1pole_lpf(torch.abs(z3_k), 160.0, sr=sr)

    q_low = (l_250 - l_80) / (
        torch.sqrt(torch.mean((l_250 - l_80) ** 2, dim=0, keepdim=True)) + 1.0e-12
    )
    q_mid = (h_250 - h_2500) / (
        torch.sqrt(torch.mean((h_250 - h_2500) ** 2, dim=0, keepdim=True)) + 1.0e-12
    )
    q_hi = h_2500 / (torch.sqrt(torch.mean(h_2500 * h_2500, dim=0, keepdim=True)) + 1.0e-12)

    wf_in = torch.cat([z3_k, l_250, h_250, h_2500], dim=-1)
    wf_drives = (1.2, 3.0, 7.5, 20.0, 55.0)
    wf_core_chunks, _ = _project_nonlinear_batches_chunked(
        wf_in, wf_drives, "wavefolder", wf_in.shape[1] * len(wf_drives) * 2, host_solver, sr=sr
    )
    q_4 = torch.cat([q_low, q_low, q_mid, q_hi], dim=-1)
    del q_low, q_mid, q_hi, wf_in

    # Project wf_hyst in chunks without materializing full (L, 320) q_rep_full or wf_hyst_chunks
    max_wf_hyst = 96 if is_long_seq else 200
    c_wf_tot = sum(ch.shape[1] for ch in wf_core_chunks)
    q_cols = q_4.shape[1]

    def _slice_q_rep(q_mat: Tensor, col_offset: int, width: int) -> Tensor:
        rem = col_offset % q_cols
        reps = ((rem + width + q_cols - 1) // q_cols) + 1
        return q_mat.repeat(1, reps)[:, rem : rem + width]

    if c_wf_tot > max_wf_hyst:
        hyst_stride = max(1, l_tot // 16_384)
        q_sub = q_4[::hyst_stride]
        hyst_sub_parts: list[Tensor] = []
        c_off = 0
        for ch in wf_core_chunks:
            w_ch = ch.shape[1]
            hyst_sub_parts.append(ch[::hyst_stride] * _slice_q_rep(q_sub, c_off, w_ch))
            c_off += w_ch
        hyst_sub = torch.cat(hyst_sub_parts, dim=-1)
        del hyst_sub_parts, q_sub
        proj_hyst = host_solver.fit_orthonormal_projector(hyst_sub, max_wf_hyst)
        del hyst_sub
        hyst_out_chunks: list[Tensor] = []
        hyst_out_ranges: list[tuple[int, int]] = []
        for k0 in range(0, max_wf_hyst, MAX_CHUNK_COLS):
            k1 = min(max_wf_hyst, k0 + MAX_CHUNK_COLS)
            hyst_out_chunks.append(torch.zeros((l_tot, k1 - k0), device=dev, dtype=torch.float32))
            hyst_out_ranges.append((k0, k1))
        c_off = 0
        for ch in wf_core_chunks:
            w_ch = ch.shape[1]
            ch_hyst = ch * _slice_q_rep(q_4, c_off, w_ch)
            p_in = proj_hyst[c_off : c_off + w_ch]
            for ch_out, (k0, k1) in zip(hyst_out_chunks, hyst_out_ranges, strict=True):
                ch_out.add_(ch_hyst @ p_in[:, k0:k1])
            del ch_hyst
            c_off += w_ch
        del proj_hyst, q_4
        wf_hyst_cm = ColumnChunkedMatrix(hyst_out_chunks)
    else:
        wf_hyst_chunks: list[Tensor] = []
        c_off = 0
        for ch in wf_core_chunks:
            w_ch = ch.shape[1]
            wf_hyst_chunks.append(ch * _slice_q_rep(q_4, c_off, w_ch))
            c_off += w_ch
        del q_4
        wf_hyst_cm = ColumnChunkedMatrix(wf_hyst_chunks)
    if is_long_seq:
        wf_core_cm = ColumnChunkedMatrix(wf_core_chunks)
        if wf_core_cm.shape[1] > 96:
            wf_core_cm = _project_chunked_matrix(wf_core_cm, 96, host_solver, free_input=True)
        wf_core_chunks = wf_core_cm.chunks

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
    max_nl_s4 = 128 if is_long_seq else 360
    nl_inst_chunks, _ = _project_nonlinear_batches_chunked(
        nl_in, (0.8, 2.5, 8.0, 25.0), "inst", max_nl_s4, host_solver, sr=sr
    )
    del nl_in

    pad = 16
    h_cat = torch.cat([z3_k, h_900, h_2500, h_6000], dim=-1)
    del z3_k, h_900, h_2500, h_6000
    h_pad = F.pad(h_cat.unsqueeze(0), (0, 0, pad, pad)).squeeze(0)
    del h_cat
    shifts_tup = (
        (-3, -1, 0, 1, 3) if is_long_seq else (-8, -5, -3, -2, -1, 0, 1, 2, 3, 5, 8, 12, 16)
    )
    shift_chunks: list[Tensor] = []
    for s0 in range(0, len(shifts_tup), 5):
        s_sub = shifts_tup[s0 : s0 + 5]
        shift_chunks.append(torch.cat([h_pad[pad - d : pad - d + l_tot] for d in s_sub], dim=-1))
    del h_pad

    u_s4_cm = ColumnChunkedMatrix.from_tensors(
        [*nl_inst_chunks, *wf_core_chunks, *wf_hyst_cm.chunks, *shift_chunks],
        free_inputs=True,
    )
    del nl_inst_chunks, wf_core_chunks, wf_hyst_cm, shift_chunks
    max_s4_tot = 360 if is_long_seq else 800
    if u_s4_cm.shape[1] > max_s4_tot:
        u_s4_cm = _project_chunked_matrix(u_s4_cm, max_s4_tot, host_solver, free_input=True)
    gc.collect()
    empty_device_cache(dev)

    n_cols = u_s4_cm.shape[1]
    n_tgt = 3 * m_col
    acc_gl = ExactGramAccumulator(n_cols, n_tgt, device=dev)
    n_wins = len(win_bounds)
    overlap = 256 if l_tot >= 32_768 else 0
    d_win_3b = torch.zeros((l_tot, n_tgt), device=dev, dtype=torch.float32)
    win_batch_size = 4 if is_long_seq else 8

    for b0 in range(0, n_wins, win_batch_size):
        b1 = min(n_wins, b0 + win_batch_size)
        batch_s4_specs: list[HostSolveSpec] = []
        batch_uw: list[tuple[int, int, int, Tensor]] = []
        for w_idx in range(b0, b1):
            k0, k1 = win_bounds[w_idx]
            s_w = 0 if w_idx == 0 else max(0, k0 - overlap)
            e_w = l_tot if w_idx == n_wins - 1 else min(l_tot, k1 + overlap)
            u_w = u_s4_cm.slice_rows(s_w, e_w)
            r_w = torch.cat([r3_lf_f32[s_w:e_w], r3_mf_f32[s_w:e_w], r3_hf_f32[s_w:e_w]], dim=-1)

            acc_w = ExactGramAccumulator(n_cols, n_tgt, device=dev)
            acc_w.add(u_w[::stride_s4], r_w[::stride_s4], weight=1.0)
            uk_pre = u_w[1::stride_s4] - 0.85 * u_w[:-1:stride_s4]
            rk_pre = r_w[1::stride_s4] - 0.85 * r_w[:-1:stride_s4]
            acc_w.add(uk_pre, rk_pre, weight=0.5)
            del r_w, uk_pre, rk_pre

            acc_gl.merge(acc_w)
            acc_w.offload_to_cpu()
            batch_s4_specs.append(HostSolveSpec(accumulator=acc_w, ridge_scale=1e-5))
            batch_uw.append((w_idx, s_w, e_w, u_w))

        batch_s4_weights = host_solver.solve_batch(
            batch_s4_specs, pass_name="stage4_polish_win", return_device=dev
        )
        del batch_s4_specs
        for (w_idx, s_w, e_w, u_w), ww in zip(batch_uw, batch_s4_weights, strict=True):
            if isinstance(bspline_weights_win, CausalBsplineWindowSchedule):
                bw_slice = bspline_weights_win.window_weight(w_idx, s_w, e_w)
            else:
                bw_slice = bspline_weights_win[s_w:e_w, w_idx : w_idx + 1]
            d_win_3b[s_w:e_w].add_(bw_slice * (u_w @ ww))
        del batch_uw, batch_s4_weights

    acc_gl.offload_to_cpu()
    w_gl = host_solver.solve_single(
        acc_gl, ridge_scale=2e-6, pass_name="stage4_polish_gl", return_device=dev
    )
    del acc_gl
    d_glob_3b = u_s4_cm.matmul(w_gl, free_self=True)
    del u_s4_cm, w_gl
    gc.collect()
    empty_device_cache(dev)

    use_lti_polish = l_tot >= 48_000
    s4_lti_irs: dict[int, Tensor] = {}

    def _build_polish_phi_s(s_idx: int) -> Tensor:
        w3_s = d_win_3b[:, s_idx::m_col]
        g3_s = d_glob_3b[:, s_idx::m_col]
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
            )
            _, r_lti_l, r_lti_m, r_lti_h, h_lti_s = solve_lti_cabinet_and_space_stage_with_ir(
                z3_f32[:, s_idx],
                r_tot_s,
                max_ir_len=2048,
                sr=sr,
                cached_ir=s4_lti_irs.get(s_idx),
            )
            if s_idx not in s4_lti_irs:
                s4_lti_irs[s_idx] = h_lti_s.detach().to(device="cpu")
            phi_parts.extend([w_l120, w_l400, w_l1500, w_l4500, *w_sh])
            phi_parts.append(torch.cat([r_lti_l, r_lti_m, r_lti_h], dim=-1))
            del w_l120, w_l400, w_l1500, w_l4500, w_pad, w_sh
            del r_tot_s, r_lti_l, r_lti_m, r_lti_h, h_lti_s
        return torch.cat(phi_parts, dim=-1)

    s4_syn_specs: list[HostSolveSpec] = []
    for s_idx in range(m_col):
        phi_s = _build_polish_phi_s(s_idx)
        w_lf_s = 1.0 / float(sig_lf_f32[s_idx].item())
        w_mf_s = 1.0 / float(sig_mf_f32[s_idx].item())
        w_hf_s = 1.0 / float(sig_hf_f32[s_idx].item())
        rlf_s = r3_lf_f32[:, s_idx : s_idx + 1]
        rmf_s = r3_mf_f32[:, s_idx : s_idx + 1]
        rhf_s = r3_hf_f32[:, s_idx : s_idx + 1]
        acc_syn, p_lf, p_mf, p_hf = _build_3band_synthesis_accumulator(
            phi_s, rlf_s, rmf_s, rhf_s, w_lf_s, w_mf_s, w_hf_s, stride=stride_s4, sr=sr
        )
        acc_syn.offload_to_cpu()
        del phi_s, p_lf, p_mf, p_hf
        s4_syn_specs.append(HostSolveSpec(accumulator=acc_syn))
    empty_device_cache(dev)

    alpha_s4 = host_solver.solve_batch(
        s4_syn_specs, pass_name="stage4_polish_syn", return_device=dev
    )
    del s4_syn_specs

    d4_tot_cols: list[Tensor] = []
    d4_lf_cols: list[Tensor] = []
    d4_mf_cols: list[Tensor] = []
    d4_hf_cols: list[Tensor] = []
    for s_idx, alpha_s in enumerate(alpha_s4):
        phi_s = _build_polish_phi_s(s_idx)
        d_tot_s = (phi_s @ alpha_s).squeeze(-1)
        del phi_s
        d_lf_s, d_mf_s, d_hf_s = split_bezout_3band(d_tot_s, sr=sr)
        d4_lf_cols.append(d_lf_s)
        d4_mf_cols.append(d_mf_s)
        d4_hf_cols.append(d_hf_s)
        d4_tot_cols.append(d_lf_s + d_mf_s + d_hf_s)
        del d_tot_s

    free_tensor_storage(d_win_3b)
    free_tensor_storage(d_glob_3b)
    s4_lti_irs.clear()
    del alpha_s4
    empty_device_cache(dev)
    return (
        torch.stack(d4_tot_cols, dim=-1),
        torch.stack(d4_lf_cols, dim=-1),
        torch.stack(d4_mf_cols, dim=-1),
        torch.stack(d4_hf_cols, dim=-1),
    )


@torch.no_grad()
def run_3stage_selective_tube_cascade(
    x: Tensor,
    y_full: Tensor,
    y_lf: Tensor,
    y_mf: Tensor,
    y_hf: Tensor,
    sig_lf_all: Tensor,
    sig_mf_all: Tensor,
    sig_hf_all: Tensor,
    *,
    z_lti_bands: tuple[Tensor, Tensor, Tensor] | None,
    sr: float,
    on_stage_progress: Callable[[int, str], None] | None,
    report_step: Callable[[int, Tensor, Tensor, Tensor], Tensor],
    solver: HostDenseSolver | None = None,
    cached_seg_stats: SegmentationWindowStats | None = None,
    cached_s1_proj: Tensor | None = None,
    out_calib_records: dict[str, Any] | None = None,
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
    """Executes the 3-Stage Selective State-Space Tube/Solid-State Cascade + Stage-4 Quadrature
    Hysteresis Polish and returns `(z1_full, z2_full, z3_full, d4_tot, z4_lf, z4_mf, z4_hf, z3_hf)`.
    """
    host_solver = solver if solver is not None else HostDenseSolver()
    l_tot = x.shape[0]
    m_col = y_full.shape[1]

    win_bounds, win_bounds48 = build_stage_window_grids(l_tot)
    seg_stats = (
        cached_seg_stats
        if (cached_seg_stats is not None and list(cached_seg_stats.win_bounds) == win_bounds)
        else compute_segmentation_window_stats(x, y_full, win_bounds, sr=sr)
    )
    regimes = segment_regimes_dp(seg_stats)

    e_cum = compute_cumulative_energy_state(x, sr=sr)
    win_sched = CausalBsplineWindowSchedule(
        e_cum, win_bounds, overlap=256 if l_tot >= 48_000 else 0
    )
    win48_sched = (
        CausalBsplineWindowSchedule(e_cum, win_bounds48, overlap=256 if l_tot >= 48_000 else 0)
        if len(win_bounds48) != len(win_bounds)
        else win_sched
    )
    reg_sched = CausalBsplineWindowSchedule(e_cum, regimes, overlap=512 if l_tot >= 48_000 else 0)
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
    biased_bands: list[Tensor] = []
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
    is_long_seq = l_tot > 600_000
    max_s1_adaa = (
        200 if is_long_seq else (300 if m_col > 14 else adaa_s1_t.shape[1] * len(drives_s1) * 2)
    )
    adaa_s1_chunks, s1_proj_t = _project_nonlinear_batches_chunked(
        adaa_s1_t,
        drives_s1,
        "adaa",
        max_s1_adaa,
        host_solver,
        sr=sr,
        cached_proj=cached_s1_proj,
    )
    if out_calib_records is not None:
        out_calib_records["seg_stats"] = seg_stats
        out_calib_records["s1_proj"] = (
            s1_proj_t.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
            if s1_proj_t is not None
            else None
        )
    del adaa_s1_t, s1_proj_t
    max_s1_inst = (
        80 if is_long_seq else (110 if m_col > 14 else inst_s1_t.shape[1] * len(drives_s1) * 2)
    )
    inst_s1_chunks, _ = _project_nonlinear_batches_chunked(
        inst_s1_t, drives_s1, "inst", max_s1_inst, host_solver, sr=sr
    )
    del inst_s1_t
    u_s1_cm = ColumnChunkedMatrix.from_tensors([x.unsqueeze(-1), *adaa_s1_chunks, *inst_s1_chunks])
    del adaa_s1_chunks, inst_s1_chunks
    gc.collect()

    z1_full, z1_lf, z1_mf, z1_hf = solve_selective_stage_3band(
        u_s1_cm,
        y_full,
        y_lf,
        y_mf,
        y_hf,
        win_sched,
        reg_sched,
        ref_gates,
        win_bounds,
        regimes,
        z_lti_bands=z_lti_bands,
        sig_lf=sig_lf_all,
        sig_mf=sig_mf_all,
        sig_hf=sig_hf_all,
        solver=host_solver,
        free_u_stage=True,
        sr=sr,
    )
    del u_s1_cm
    gc.collect()
    empty_device_cache(x.device)
    res_w1 = report_step(1, z1_lf, z1_mf, z1_hf)
    del z1_lf, z1_mf, z1_hf

    if on_stage_progress is not None:
        on_stage_progress(2, "Calibrating Stage 2 (V1B Interstage Coupling & Grid-Leak Cascade)")
    u_s2_cm = build_interstage_basis(
        z1_full,
        None,
        (15.0, 60.0, 220.0, 800.0),
        20.0,
        35.0,
        8.0,
        (0.8, 2.5, 8.0, 25.0, 75.0),
        res_weights=res_w1,
        use_power_sag=False,
        solver=host_solver,
        return_chunked=True,
        sr=sr,
    )
    z2_full, z2_lf, z2_mf, z2_hf = solve_selective_stage_3band(
        u_s2_cm,
        y_full,
        y_lf,
        y_mf,
        y_hf,
        win_sched,
        reg_sched,
        ref_gates,
        win_bounds,
        regimes,
        res_weights=res_w1,
        z_lti_bands=z_lti_bands,
        sig_lf=sig_lf_all,
        sig_mf=sig_mf_all,
        sig_hf=sig_hf_all,
        solver=host_solver,
        free_u_stage=True,
        sr=sr,
    )
    del u_s2_cm, res_w1
    gc.collect()
    empty_device_cache(x.device)
    res_w2 = report_step(2, z2_lf, z2_mf, z2_hf)
    del z2_lf, z2_mf, z2_hf

    if on_stage_progress is not None:
        on_stage_progress(
            3, "Calibrating Stage 3 (Power Section, Output Transformer & 48 kHz Reference Polish)"
        )
    u_s3_cm = build_interstage_basis(
        z2_full,
        z1_full,
        (12.0, 45.0, 180.0, 650.0),
        18.0,
        25.0,
        4.0,
        (0.6, 2.0, 6.5, 20.0, 60.0),
        res_weights=res_w2,
        use_power_sag=True,
        solver=host_solver,
        return_chunked=True,
        sr=sr,
    )
    z3_full, z3_lf, z3_mf, z3_hf = solve_selective_stage_3band(
        u_s3_cm,
        y_full,
        y_lf,
        y_mf,
        y_hf,
        win48_sched if m_col > 14 else win_sched,
        reg_sched,
        ref_gates,
        win_bounds48 if m_col > 14 else win_bounds,
        regimes,
        res_weights=res_w2,
        z_lti_bands=z_lti_bands,
        sig_lf=sig_lf_all,
        sig_mf=sig_mf_all,
        sig_hf=sig_hf_all,
        solver=host_solver,
        free_u_stage=True,
        sr=sr,
    )
    del u_s3_cm, res_w2
    free_tensor_storage(y_full)
    if z_lti_bands is not None:
        for t_lti in z_lti_bands:
            free_tensor_storage(t_lti)
    gc.collect()
    empty_device_cache(x.device)
    report_step(3, z3_lf, z3_mf, z3_hf)

    r3_lf = y_lf - z3_lf
    r3_mf = y_mf - z3_mf
    r3_hf = y_hf - z3_hf
    free_tensor_storage(z3_lf)
    free_tensor_storage(z3_mf)
    if m_col > 7:
        free_tensor_storage(z3_hf)
    del z3_lf, z3_mf
    gc.collect()
    empty_device_cache(x.device)
    d4_tot, d4_lf, d4_mf, d4_hf = solve_fullrate_hf_residual_polish(
        z3_full,
        z2_full,
        r3_lf,
        r3_mf,
        r3_hf,
        sig_lf_all,
        sig_mf_all,
        sig_hf_all,
        win48_sched,
        ref_gates,
        win_bounds48,
        z1_full=z1_full,
        solver=host_solver,
        sr=sr,
    )
    z4_lf = (y_lf - r3_lf) + d4_lf
    z4_mf = (y_mf - r3_mf) + d4_mf
    z4_hf = (y_hf - r3_hf) + d4_hf
    del r3_lf, r3_mf, r3_hf, d4_lf, d4_mf, d4_hf
    del win_sched, win48_sched, reg_sched, e_cum, ref_gates
    report_step(4, z4_lf, z4_mf, z4_hf)
    if m_col > 16:
        free_tensor_storage(z4_lf)
        free_tensor_storage(z4_mf)
    gc.collect()
    empty_device_cache(x.device)
    return z1_full, z2_full, z3_full, d4_tot, z4_lf, z4_mf, z4_hf, z3_hf
