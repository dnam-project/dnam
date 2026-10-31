"""32-bit-only on-device exact split-precision Gram accumulation (`ExactGramAccumulator`)
and unified batched host-CPU `float64` small dense solver (`HostDenseSolver`).

Architecture:
1. **On-Device (`ExactGramAccumulator`):**
   - Uses only `float32`, `int32`, and `bfloat16` tensors on the accelerator (`MPS`, `TPU`,
     `CUDA`, `ROCm`, or `CPU`).
   - Computes exact per-column power-of-two ceiling scales via IEEE-754 `int32` exponent-bit
     masking (`pow2_ceil_fp32`, avoiding `torch.frexp` and inexact `torch.exp2` on MPS).
   - Splits each sub-block (`<= 16,384` rows) into $S = 3$ 8-bit integer-valued slices
     ($|q| \\le 2^8$), stored compactly in `bfloat16` (exact for integers up to $2^8$).
   - Evaluates slice inner products in chunks of $K = 128$ rows after upcasting `.to(float32)`
     so every chunk sum ($\\le 128 \\times 256^2 = 2^{23}$) is bit-exact in `float32`.
   - Accumulates chunk products in `int32` across up to 128 chunks (`16,384` rows, $\\le 2^{30}$)
     and folds them into a 3-limb `float32` error-free expansion `(hi, mid, lo)` (`72` bits of
     significand) via Knuth `two_sum` / `quick_two_sum`.
2. **On-Host (`HostDenseSolver`):**
   - Transfers all `(hi, mid, lo)` Gram and RHS expansions from a pass to the host in a single
     batched transfer, recombines them losslessly into `float64`, and solves ridge Cholesky
     systems (with `lstsq` fallback + `SolverEvent` telemetry) and `eigh` subspace projections.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from dnam_trainer.events import SolverEvent, TrainingObserver


def pow2_ceil_fp32(v: Tensor) -> Tensor:
    """Computes the exact IEEE-754 power-of-two ceiling $\\ge \\max(|v|, 10^{-30})$ in `float32`
    using `int32` exponent-bit masking (`0` `frexp` / `exp2` calls).
    """
    vc = v.to(dtype=torch.float32).abs().clamp_min(1.0e-30).contiguous()
    bits = vc.view(torch.int32)
    ceil_bits = (bits + 0x007FFFFF) & 0x7F800000
    return ceil_bits.view(torch.float32)


def argmax_i32(x: Tensor, dim: int | None = -1) -> Tensor:
    """Computes `argmax` along `dim` returning a `torch.int32` tensor without allocating `int64`."""
    if dim is None:
        return argmax_i32(x.reshape(-1), dim=0)
    if dim < 0:
        dim = x.ndim + dim
    n = x.shape[dim]
    shape = [1] * x.ndim
    shape[dim] = n
    idx = torch.arange(n, device=x.device, dtype=torch.int32).view(shape).expand_as(x)
    max_val = torch.amax(x, dim=dim, keepdim=True)
    sentinel = torch.full_like(idx, n)
    return torch.amin(torch.where(x == max_val, idx, sentinel), dim=dim)


def two_sum(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """Knuth's error-free transformation `s + e == a + b` in `float32`."""
    s = a + b
    v = s - a
    e = (a - (s - v)) + (b - v)
    return s, e


def quick_two_sum(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """Dekker's fast error-free transformation when `|a| >= |b|` in `float32`."""
    s = a + b
    e = b - (s - a)
    return s, e


def _split_veltkamp_f32(a: Tensor) -> tuple[Tensor, Tensor]:
    """Veltkamp 12-bit splitting of an IEEE-754 `float32` tensor (`2^12 + 1 = 4097.0`)."""
    c = 4097.0 * a
    a_hi = c - (c - a)
    a_lo = a - a_hi
    return a_hi, a_lo


def two_prod(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """Dekker's error-free product `p + err == a * b` in `float32`."""
    p = a * b
    a_hi, a_lo = _split_veltkamp_f32(a)
    b_hi, b_lo = _split_veltkamp_f32(b)
    err = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo
    return p, err


def dd_add(
    hi: Tensor,
    lo: Tensor,
    d_hi: Tensor,
    d_lo: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Adds `(d_hi, d_lo)` into a 2-limb `float32` expansion `(hi, lo)`."""
    s, e = two_sum(hi, d_hi)
    e = e + lo
    if d_lo is not None:
        e = e + d_lo
    return quick_two_sum(s, e)


def td_add(
    hi: Tensor,
    mid: Tensor,
    lo: Tensor,
    d_hi: Tensor,
    d_lo: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Adds `(d_hi, d_lo)` into a 3-limb `float32` expansion `(hi, mid, lo)` (72-bit)."""
    s1, e1 = two_sum(hi, d_hi)
    s2, e2 = two_sum(mid, e1)
    s3, e3 = two_sum(s2, d_lo)
    e_lo = lo + e2 + e3
    r_hi, r_mid = two_sum(s1, s3)
    r_mid, r_lo = two_sum(r_mid, e_lo)
    return r_hi, r_mid, r_lo


def split_int32_to_f32_pair(acc_i32: Tensor) -> tuple[Tensor, Tensor]:
    """Losslessly splits a signed `int32` tensor (`|v| <= 2^30`) into two `float32` tensors
    `(hi, lo)` such that `hi + lo == acc_i32` exactly.
    """
    hi = acc_i32.to(torch.float32)
    lo = (acc_i32 - hi.to(torch.int32)).to(torch.float32)
    return hi, lo


def split_f64_to_f32_trilimb(
    arr_f64: NDArray[np.float64],
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor, Tensor]:
    """Losslessly splits a host `float64` NumPy array into three `float32` tensors `(hi, mid, lo)`
    on `device` (preserving all 53 mantissa bits of `float64`).
    """
    a64 = np.ascontiguousarray(arr_f64, dtype=np.float64)
    hi_np = a64.astype(np.float32)
    rem1 = a64 - hi_np.astype(np.float64)
    mid_np = rem1.astype(np.float32)
    rem2 = rem1 - mid_np.astype(np.float64)
    lo_np = rem2.astype(np.float32)
    return (
        torch.from_numpy(hi_np).to(device=device, dtype=torch.float32),
        torch.from_numpy(mid_np).to(device=device, dtype=torch.float32),
        torch.from_numpy(lo_np).to(device=device, dtype=torch.float32),
    )


def recombine_f32_trilimb_to_f64(hi: Tensor, mid: Tensor, lo: Tensor) -> NDArray[np.float64]:
    """Recombines three `float32` limb tensors `(hi, mid, lo)` into a host `float64` NumPy array."""
    stacked_cpu: Any = (
        torch.stack([hi.detach(), mid.detach(), lo.detach()], dim=0)
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .numpy()
    )
    return np.ascontiguousarray(
        stacked_cpu[0].astype(np.float64)
        + stacked_cpu[1].astype(np.float64)
        + stacked_cpu[2].astype(np.float64),
        dtype=np.float64,
    )


def split_f64_to_f32_pair(
    arr_f64: NDArray[np.float64],
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Splits a host `float64` NumPy array into a `(hi, lo)` `float32` pair on `device`."""
    a64 = np.ascontiguousarray(arr_f64, dtype=np.float64)
    hi_np = a64.astype(np.float32)
    lo_np = (a64 - hi_np.astype(np.float64)).astype(np.float32)
    return (
        torch.from_numpy(hi_np).to(device=device, dtype=torch.float32),
        torch.from_numpy(lo_np).to(device=device, dtype=torch.float32),
    )


def recombine_f32_pair_to_f64(hi: Tensor, lo: Tensor) -> NDArray[np.float64]:
    """Recombines a `(hi, lo)` `float32` pair into a host `float64` NumPy array."""
    stacked_cpu: Any = (
        torch.stack([hi.detach(), lo.detach()], dim=0)
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .numpy()
    )
    return np.ascontiguousarray(
        stacked_cpu[0].astype(np.float64) + stacked_cpu[1].astype(np.float64),
        dtype=np.float64,
    )


def _extract_3slices_bf16(x_sub: Tensor) -> tuple[Tensor, Tensor]:
    """Splits `(R_sub, C)` `float32` tensor into 3 8-bit signed integer slices (`|s_p| <= 128`,
    covering all 24 IEEE-754 `float32` mantissa bits) stored in `(R_sub, 3 * C)` `bfloat16`,
    returning `(s_cat_bf16, col_scale_1xc)`.
    """
    sc = pow2_ceil_fp32(x_sub.abs().amax(dim=0, keepdim=True)) * 2.0
    rem = x_sub / sc
    s0 = torch.round(rem * 256.0)
    rem = rem * 256.0 - s0
    s1 = torch.round(rem * 256.0)
    rem = rem * 256.0 - s1
    s2 = torch.round(rem * 256.0)
    s_cat = torch.cat(
        [
            s0.to(torch.bfloat16),
            s1.to(torch.bfloat16),
            s2.to(torch.bfloat16),
        ],
        dim=1,
    )
    return s_cat, sc


_GRAM_PAIRS_DESC: tuple[tuple[int, int, float], ...] = (
    (2, 2, 2.0 ** (-8 * 6)),
    (1, 2, 2.0 ** (-8 * 5)),
    (1, 1, 2.0 ** (-8 * 4)),
    (0, 2, 2.0 ** (-8 * 4)),
    (0, 1, 2.0 ** (-8 * 3)),
    (0, 0, 2.0 ** (-8 * 2)),
)

_RHS_PAIRS_DESC: tuple[tuple[int, int, float], ...] = (
    (2, 2, 2.0 ** (-8 * 6)),
    (2, 1, 2.0 ** (-8 * 5)),
    (1, 2, 2.0 ** (-8 * 5)),
    (2, 0, 2.0 ** (-8 * 4)),
    (1, 1, 2.0 ** (-8 * 4)),
    (0, 2, 2.0 ** (-8 * 4)),
    (1, 0, 2.0 ** (-8 * 3)),
    (0, 1, 2.0 ** (-8 * 3)),
    (0, 0, 2.0 ** (-8 * 2)),
)


class ExactGramAccumulator:
    """On-device 32-bit Ozaki-style split-precision accumulator for `G = X^T X` and `C = X^T Y`."""

    def __init__(
        self,
        num_cols: int,
        rhs_cols: int = 0,
        *,
        device: torch.device | str = "cpu",
        chunk_rows: int = 1024,
        subblock_rows: int = 16_384,
        max_tile_cols: int = 1024,
    ) -> None:
        if num_cols < 1:
            raise ValueError(f"num_cols must be >= 1, got {num_cols}")
        self.num_cols = num_cols
        self.rhs_cols = max(0, rhs_cols)
        self.device = torch.device(device)
        # With 3 8-bit slices and sc = 2 * pow2_ceil, |s_p| <= 128 = 2^7, so |s_p * s_q| <= 2^14
        # and up to 1024 rows sum to <= 2^24 (100% bit-exact in float32).
        self.chunk_rows = max(16, min(1024, chunk_rows))
        # Ensure subblock_rows <= 64 * chunk_rows so int32 sums stay <= 2^30
        max_sub = 64 * self.chunk_rows
        self.subblock_rows = min(max_sub, max(self.chunk_rows, subblock_rows))
        self.max_tile_cols = max(64, max_tile_cols)

        self.g_hi = torch.zeros(
            (self.num_cols, self.num_cols), device=self.device, dtype=torch.float32
        )
        self.g_mid = torch.zeros(
            (self.num_cols, self.num_cols), device=self.device, dtype=torch.float32
        )
        self.g_lo = torch.zeros(
            (self.num_cols, self.num_cols), device=self.device, dtype=torch.float32
        )

        if self.rhs_cols > 0:
            self.c_hi = torch.zeros(
                (self.num_cols, self.rhs_cols), device=self.device, dtype=torch.float32
            )
            self.c_mid = torch.zeros(
                (self.num_cols, self.rhs_cols), device=self.device, dtype=torch.float32
            )
            self.c_lo = torch.zeros(
                (self.num_cols, self.rhs_cols), device=self.device, dtype=torch.float32
            )
        else:
            self.c_hi = torch.zeros((self.num_cols, 0), device=self.device, dtype=torch.float32)
            self.c_mid = torch.zeros((self.num_cols, 0), device=self.device, dtype=torch.float32)
            self.c_lo = torch.zeros((self.num_cols, 0), device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def add(
        self,
        x: Tensor,
        y: Tensor | None = None,
        *,
        weight: float = 1.0,
    ) -> ExactGramAccumulator:
        """Accumulates `weight * (x^T @ x)` (and `weight * (x^T @ y)` if `y` is provided)."""
        if x.ndim != 2 or x.shape[1] != self.num_cols:
            raise ValueError(f"Expected x shape (R, {self.num_cols}), got {tuple(x.shape)}")
        if y is not None:
            if y.ndim == 1:
                y = y.unsqueeze(-1)
            if y.ndim != 2 or y.shape[0] != x.shape[0] or y.shape[1] != self.rhs_cols:
                raise ValueError(
                    f"Expected y shape ({x.shape[0]}, {self.rhs_cols}), got {tuple(y.shape)}"
                )
        if x.shape[0] == 0 or weight == 0.0:
            return self

        x_f32 = x.to(device=self.device, dtype=torch.float32)
        y_f32 = y.to(device=self.device, dtype=torch.float32) if y is not None else None
        total_rows = x_f32.shape[0]
        w_f32 = float(weight)

        for r0 in range(0, total_rows, self.subblock_rows):
            r1 = min(total_rows, r0 + self.subblock_rows)
            xs = x_f32[r0:r1]
            ys = y_f32[r0:r1] if y_f32 is not None else None
            self._accumulate_subblock(xs, ys, w_f32)

        return self

    def _accumulate_subblock(
        self,
        xs: Tensor,
        ys: Tensor | None,
        weight: float,
    ) -> None:
        c_x = self.num_cols
        c_y = self.rhs_cols if ys is not None else 0
        if c_x <= self.max_tile_cols:
            self._accumulate_subblock_untiled(xs, ys, weight)
            return

        # Column tiling when num_cols > max_tile_cols so no tensor exceeds max_tensor_bytes
        sx_cat, sc_x = _extract_3slices_bf16(xs)
        sy_cat, sc_y = _extract_3slices_bf16(ys) if (ys is not None and c_y > 0) else (None, None)
        tc = self.max_tile_cols
        r_len = xs.shape[0]
        w_t = torch.tensor(weight, device=self.device, dtype=torch.float32)

        for i0 in range(0, c_x, tc):
            i1 = min(c_x, i0 + tc)
            wi = i1 - i0
            sxi = torch.cat(
                [
                    sx_cat[:, i0:i1],
                    sx_cat[:, c_x + i0 : c_x + i1],
                    sx_cat[:, 2 * c_x + i0 : 2 * c_x + i1],
                ],
                dim=1,
            )
            sci = sc_x[:, i0:i1]

            for j0 in range(i0, c_x, tc):
                j1 = min(c_x, j0 + tc)
                wj = j1 - j0
                if i0 == j0:
                    sxj = sxi
                    scj = sci
                else:
                    sxj = torch.cat(
                        [
                            sx_cat[:, j0:j1],
                            sx_cat[:, c_x + j0 : c_x + j1],
                            sx_cat[:, 2 * c_x + j0 : 2 * c_x + j1],
                        ],
                        dim=1,
                    )
                    scj = sc_x[:, j0:j1]
                acc_tile = torch.zeros((3 * wi, 3 * wj), device=self.device, dtype=torch.int32)
                for k0 in range(0, r_len, self.chunk_rows):
                    k1 = min(r_len, k0 + self.chunk_rows)
                    ci = sxi[k0:k1].to(torch.float32)
                    cj = ci if i0 == j0 else sxj[k0:k1].to(torch.float32)
                    acc_tile.add_((ci.T @ cj).to(torch.int32))
                sc_outer_p2 = sci.T * scj
                pairs_to_use = _GRAM_PAIRS_DESC if i0 == j0 else _RHS_PAIRS_DESC
                for p, q, sh in pairs_to_use:
                    blk_i32 = acc_tile[p * wi : (p + 1) * wi, q * wj : (q + 1) * wj]
                    if i0 == j0 and p != q:
                        blk_i32 = blk_i32 + blk_i32.T
                    hi_b, lo_b = split_int32_to_f32_pair(blk_i32)
                    s_p2 = sc_outer_p2 * sh
                    hp = hi_b * s_p2
                    lp = lo_b * s_p2
                    if weight == 1.0:
                        (
                            self.g_hi[i0:i1, j0:j1],
                            self.g_mid[i0:i1, j0:j1],
                            self.g_lo[i0:i1, j0:j1],
                        ) = td_add(
                            self.g_hi[i0:i1, j0:j1],
                            self.g_mid[i0:i1, j0:j1],
                            self.g_lo[i0:i1, j0:j1],
                            hp,
                            lp,
                        )
                    else:
                        d_hi1, d_lo1 = two_prod(hp, w_t)
                        d_hi2, d_lo2 = two_prod(lp, w_t)
                        (
                            self.g_hi[i0:i1, j0:j1],
                            self.g_mid[i0:i1, j0:j1],
                            self.g_lo[i0:i1, j0:j1],
                        ) = td_add(
                            self.g_hi[i0:i1, j0:j1],
                            self.g_mid[i0:i1, j0:j1],
                            self.g_lo[i0:i1, j0:j1],
                            d_hi1,
                            d_lo1,
                        )
                        (
                            self.g_hi[i0:i1, j0:j1],
                            self.g_mid[i0:i1, j0:j1],
                            self.g_lo[i0:i1, j0:j1],
                        ) = td_add(
                            self.g_hi[i0:i1, j0:j1],
                            self.g_mid[i0:i1, j0:j1],
                            self.g_lo[i0:i1, j0:j1],
                            d_hi2,
                            d_lo2,
                        )
                if i0 != j0:
                    self.g_hi[j0:j1, i0:i1].copy_(self.g_hi[i0:i1, j0:j1].T)
                    self.g_mid[j0:j1, i0:i1].copy_(self.g_mid[i0:i1, j0:j1].T)
                    self.g_lo[j0:j1, i0:i1].copy_(self.g_lo[i0:i1, j0:j1].T)

            if sy_cat is not None and sc_y is not None and c_y > 0:
                acc_rhs = torch.zeros((3 * wi, 3 * c_y), device=self.device, dtype=torch.int32)
                for k0 in range(0, r_len, self.chunk_rows):
                    k1 = min(r_len, k0 + self.chunk_rows)
                    ci = sxi[k0:k1].to(torch.float32)
                    cy = sy_cat[k0:k1].to(torch.float32)
                    acc_rhs.add_((ci.T @ cy).to(torch.int32))
                sc_xy_p2 = sci.T * sc_y
                for p, q, sh in _RHS_PAIRS_DESC:
                    blk_i32 = acc_rhs[p * wi : (p + 1) * wi, q * c_y : (q + 1) * c_y]
                    hi_b, lo_b = split_int32_to_f32_pair(blk_i32)
                    s_p2 = sc_xy_p2 * sh
                    hp = hi_b * s_p2
                    lp = lo_b * s_p2
                    if weight == 1.0:
                        (
                            self.c_hi[i0:i1],
                            self.c_mid[i0:i1],
                            self.c_lo[i0:i1],
                        ) = td_add(
                            self.c_hi[i0:i1],
                            self.c_mid[i0:i1],
                            self.c_lo[i0:i1],
                            hp,
                            lp,
                        )
                    else:
                        d_hi1, d_lo1 = two_prod(hp, w_t)
                        d_hi2, d_lo2 = two_prod(lp, w_t)
                        (
                            self.c_hi[i0:i1],
                            self.c_mid[i0:i1],
                            self.c_lo[i0:i1],
                        ) = td_add(
                            self.c_hi[i0:i1],
                            self.c_mid[i0:i1],
                            self.c_lo[i0:i1],
                            d_hi1,
                            d_lo1,
                        )
                        (
                            self.c_hi[i0:i1],
                            self.c_mid[i0:i1],
                            self.c_lo[i0:i1],
                        ) = td_add(
                            self.c_hi[i0:i1],
                            self.c_mid[i0:i1],
                            self.c_lo[i0:i1],
                            d_hi2,
                            d_lo2,
                        )

    def _accumulate_subblock_untiled(
        self,
        xs: Tensor,
        ys: Tensor | None,
        weight: float,
    ) -> None:
        c_x = self.num_cols
        c_y = self.rhs_cols if ys is not None else 0
        sx_cat, sc_x = _extract_3slices_bf16(xs)
        r_len = xs.shape[0]
        w_t = torch.tensor(weight, device=self.device, dtype=torch.float32)

        if ys is not None and c_y > 0:
            sy_cat, sc_y = _extract_3slices_bf16(ys)
            s_both = torch.cat([sx_cat, sy_cat], dim=1)
            if r_len <= self.chunk_rows:
                ck_both = s_both.to(torch.float32)
                ck_x = ck_both[:, : 3 * c_x]
                acc_both = (ck_x.T @ ck_both).to(torch.int32)
            else:
                acc_both = torch.zeros(
                    (3 * c_x, 3 * (c_x + c_y)), device=self.device, dtype=torch.int32
                )
                for k0 in range(0, r_len, self.chunk_rows):
                    k1 = min(r_len, k0 + self.chunk_rows)
                    ck_both = s_both[k0:k1].to(torch.float32)
                    ck_x = ck_both[:, : 3 * c_x]
                    acc_both.add_((ck_x.T @ ck_both).to(torch.int32))
            acc_g = acc_both[:, : 3 * c_x]
            acc_c = acc_both[:, 3 * c_x :]
        else:
            sc_y = None
            acc_c = None
            if r_len <= self.chunk_rows:
                ck_x = sx_cat.to(torch.float32)
                acc_g = (ck_x.T @ ck_x).to(torch.int32)
            else:
                acc_g = torch.zeros((3 * c_x, 3 * c_x), device=self.device, dtype=torch.int32)
                for k0 in range(0, r_len, self.chunk_rows):
                    k1 = min(r_len, k0 + self.chunk_rows)
                    ck_x = sx_cat[k0:k1].to(torch.float32)
                    acc_g.add_((ck_x.T @ ck_x).to(torch.int32))

        sc_xx_p2 = sc_x.T * sc_x
        for p, q, sh in _GRAM_PAIRS_DESC:
            m_i32 = acc_g[p * c_x : (p + 1) * c_x, q * c_x : (q + 1) * c_x]
            if p != q:
                m_i32 = m_i32 + m_i32.T
            hi_i, lo_i = split_int32_to_f32_pair(m_i32)
            s_p2 = sc_xx_p2 * sh
            hp = hi_i * s_p2
            lp = lo_i * s_p2
            if weight == 1.0:
                self.g_hi, self.g_mid, self.g_lo = td_add(
                    self.g_hi,
                    self.g_mid,
                    self.g_lo,
                    hp,
                    lp,
                )
            else:
                d_hi1, d_lo1 = two_prod(hp, w_t)
                d_hi2, d_lo2 = two_prod(lp, w_t)
                self.g_hi, self.g_mid, self.g_lo = td_add(
                    self.g_hi, self.g_mid, self.g_lo, d_hi1, d_lo1
                )
                self.g_hi, self.g_mid, self.g_lo = td_add(
                    self.g_hi, self.g_mid, self.g_lo, d_hi2, d_lo2
                )

        if acc_c is not None and sc_y is not None and c_y > 0:
            sc_xy_p2 = sc_x.T * sc_y
            for p, q, sh in _RHS_PAIRS_DESC:
                m_i32 = acc_c[p * c_x : (p + 1) * c_x, q * c_y : (q + 1) * c_y]
                hi_i, lo_i = split_int32_to_f32_pair(m_i32)
                s_p2 = sc_xy_p2 * sh
                hp = hi_i * s_p2
                lp = lo_i * s_p2
                if weight == 1.0:
                    self.c_hi, self.c_mid, self.c_lo = td_add(
                        self.c_hi,
                        self.c_mid,
                        self.c_lo,
                        hp,
                        lp,
                    )
                else:
                    d_hi1, d_lo1 = two_prod(hp, w_t)
                    d_hi2, d_lo2 = two_prod(lp, w_t)
                    self.c_hi, self.c_mid, self.c_lo = td_add(
                        self.c_hi, self.c_mid, self.c_lo, d_hi1, d_lo1
                    )
                    self.c_hi, self.c_mid, self.c_lo = td_add(
                        self.c_hi, self.c_mid, self.c_lo, d_hi2, d_lo2
                    )

    @torch.no_grad()
    def merge(self, other: ExactGramAccumulator, *, weight: float = 1.0) -> ExactGramAccumulator:
        """Folds `weight * other` into `self` on the accelerator."""
        if other.num_cols != self.num_cols or other.rhs_cols != self.rhs_cols:
            raise ValueError("Cannot merge ExactGramAccumulators with mismatched shapes")
        w = float(weight)
        o_g_hi = other.g_hi.to(self.device) * w
        o_g_mid = other.g_mid.to(self.device) * w
        o_g_lo = other.g_lo.to(self.device) * w
        d_hi, d_lo = quick_two_sum(o_g_mid, o_g_lo)
        self.g_hi, self.g_mid, self.g_lo = td_add(
            self.g_hi, self.g_mid, self.g_lo, o_g_hi, torch.zeros_like(o_g_hi)
        )
        self.g_hi, self.g_mid, self.g_lo = td_add(self.g_hi, self.g_mid, self.g_lo, d_hi, d_lo)

        if self.rhs_cols > 0:
            o_c_hi = other.c_hi.to(self.device) * w
            o_c_mid = other.c_mid.to(self.device) * w
            o_c_lo = other.c_lo.to(self.device) * w
            dc_hi, dc_lo = quick_two_sum(o_c_mid, o_c_lo)
            self.c_hi, self.c_mid, self.c_lo = td_add(
                self.c_hi, self.c_mid, self.c_lo, o_c_hi, torch.zeros_like(o_c_hi)
            )
            self.c_hi, self.c_mid, self.c_lo = td_add(
                self.c_hi, self.c_mid, self.c_lo, dc_hi, dc_lo
            )
        return self

    def gram_hi_lo(self) -> tuple[Tensor, Tensor]:
        """Returns a 2-limb `(hi, lo)` `float32` representation of the accumulated Gram."""
        lo_folded, _ = quick_two_sum(self.g_mid, self.g_lo)
        return self.g_hi, lo_folded

    def rhs_hi_lo(self) -> tuple[Tensor, Tensor]:
        """Returns a 2-limb `(hi, lo)` `float32` representation of the accumulated RHS."""
        lo_folded, _ = quick_two_sum(self.c_mid, self.c_lo)
        return self.c_hi, lo_folded

    def gram_trilimb(self) -> tuple[Tensor, Tensor, Tensor]:
        """Returns the full 3-limb `(hi, mid, lo)` `float32` Gram expansion."""
        return self.g_hi, self.g_mid, self.g_lo

    def rhs_trilimb(self) -> tuple[Tensor, Tensor, Tensor]:
        """Returns the full 3-limb `(hi, mid, lo)` `float32` RHS expansion."""
        return self.c_hi, self.c_mid, self.c_lo

    def to_host_f64(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Transfers and recombines `(gram_f64, rhs_f64)` on the host in a single batch."""
        g_f64 = recombine_f32_trilimb_to_f64(self.g_hi, self.g_mid, self.g_lo)
        if self.rhs_cols > 0:
            c_f64 = recombine_f32_trilimb_to_f64(self.c_hi, self.c_mid, self.c_lo)
        else:
            c_f64 = np.zeros((self.num_cols, 0), dtype=np.float64)
        return g_f64, c_f64

    @torch.no_grad()
    def offload_to_cpu(self) -> ExactGramAccumulator:
        """Moves completed 3-limb Gram and RHS expansion tensors to host CPU RAM to free
        device VRAM before batched `HostDenseSolver` execution.
        """
        if self.g_hi.device.type != "cpu":
            self.g_hi = self.g_hi.detach().to(device="cpu", dtype=torch.float32)
            self.g_mid = self.g_mid.detach().to(device="cpu", dtype=torch.float32)
            self.g_lo = self.g_lo.detach().to(device="cpu", dtype=torch.float32)
            self.c_hi = self.c_hi.detach().to(device="cpu", dtype=torch.float32)
            self.c_mid = self.c_mid.detach().to(device="cpu", dtype=torch.float32)
            self.c_lo = self.c_lo.detach().to(device="cpu", dtype=torch.float32)
            self.device = torch.device("cpu")
        return self


@dataclass(frozen=True)
class HostSolveSpec:
    """Specification for one linear ridge solve inside a batched `HostDenseSolver` pass."""

    accumulator: ExactGramAccumulator
    ridge_scale: float = 1.0e-6
    min_ridge: float = 1.0e-12
    secondary_accumulator: ExactGramAccumulator | None = None
    primary_weight: float = 1.0
    secondary_weight: float = 0.0
    rhs_col_slice: tuple[int, int] | None = None


def _solve_single_f64_numpy(
    gram_f64: NDArray[np.float64],
    rhs_f64: NDArray[np.float64],
    ridge_scale: float,
    min_ridge: float,
) -> tuple[NDArray[np.float64], bool]:
    """Solves `(G + reg * I) W = C` in host `float64` via LAPACK Cholesky with `lstsq` fallback.

    Returns `(weights_f64, used_lstsq_fallback)`.
    """
    n = gram_f64.shape[0]
    g_sym = 0.5 * (gram_f64 + gram_f64.T)
    diag_mean = float(np.trace(g_sym)) / float(max(1, n))
    base_reg = max(float(min_ridge), float(ridge_scale) * max(diag_mean, 0.0))

    for mult in (1.0, 10.0, 100.0):
        g_reg = g_sym.copy()
        g_reg.flat[:: n + 1] += base_reg * mult
        try:
            l_factor = np.linalg.cholesky(g_reg)
            y_inter = np.linalg.solve(l_factor, rhs_f64)
            w_sol = np.linalg.solve(l_factor.T, y_inter)
            if np.all(np.isfinite(w_sol)):
                return np.ascontiguousarray(w_sol, dtype=np.float64), False
        except np.linalg.LinAlgError:
            continue

    g_reg = g_sym.copy()
    g_reg.flat[:: n + 1] += base_reg
    w_lstsq, _, _, _ = np.linalg.lstsq(g_reg, rhs_f64, rcond=None)
    return np.ascontiguousarray(w_lstsq, dtype=np.float64), True


class HostDenseSolver:
    """Unified batched host-CPU `float64` solver for small dense ridge systems, Kronecker heads,
    and eigenspace projections.
    """

    def __init__(self, *, observer: TrainingObserver | None = None) -> None:
        self.observer = observer
        self.transfer_count: int = 0
        self.total_solves: int = 0
        self.total_lstsq_fallbacks: int = 0

    def solve_batch(
        self,
        specs: Sequence[HostSolveSpec],
        *,
        pass_name: str = "stage_pass",
        return_device: torch.device | str | None = None,
    ) -> list[Tensor]:
        """Transfers all `specs` in ONE batched device-to-host transfer, solves in host `float64`,
        and transfers `float32` weights back in ONE host-to-device transfer.
        """
        if not specs:
            return []

        target_dev = torch.device(return_device or specs[0].accumulator.device)
        t0 = time.perf_counter()

        # Pack flat 1D limb views into bounded <= 32 MiB device chunks -> host float32 buffer
        flat_views: list[Tensor] = []
        total_elems = 0
        for sp in specs:
            acc = sp.accumulator
            n = acc.num_cols
            m = acc.rhs_cols
            has_sec = sp.secondary_accumulator is not None
            views_sp = [
                acc.g_hi.reshape(-1),
                acc.g_mid.reshape(-1),
                acc.g_lo.reshape(-1),
                acc.c_hi.reshape(-1),
                acc.c_mid.reshape(-1),
                acc.c_lo.reshape(-1),
            ]
            if has_sec:
                sec = sp.secondary_accumulator
                assert sec is not None
                views_sp.extend(
                    [
                        sec.g_hi.reshape(-1),
                        sec.g_mid.reshape(-1),
                        sec.g_lo.reshape(-1),
                        sec.c_hi.reshape(-1),
                        sec.c_mid.reshape(-1),
                        sec.c_lo.reshape(-1),
                    ]
                )
            flat_views.extend(views_sp)
            total_elems += (6 if not has_sec else 12) * (n * n + n * m)

        max_pack_elems = 8_388_608  # 32 MiB max device staging tensor
        packed_cpu = np.empty(total_elems, dtype=np.float32)
        write_pos = 0
        chunk_views: list[Tensor] = []
        chunk_elems = 0
        for v in flat_views:
            v_len = v.numel()
            if chunk_views and (
                chunk_elems + v_len > max_pack_elems or v.device != chunk_views[0].device
            ):
                packed_dev = torch.cat(chunk_views, dim=0)
                packed_cpu[write_pos : write_pos + chunk_elems] = (
                    packed_dev.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
                )
                del packed_dev
                write_pos += chunk_elems
                chunk_views.clear()
                chunk_elems = 0
                self.transfer_count += 1
            chunk_views.append(v)
            chunk_elems += v_len
        if chunk_views:
            packed_dev = torch.cat(chunk_views, dim=0)
            packed_cpu[write_pos : write_pos + chunk_elems] = (
                packed_dev.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
            )
            del packed_dev
            chunk_views.clear()
            self.transfer_count += 1
        t_transfer_in = (time.perf_counter() - t0) * 1000.0

        t_solve_start = time.perf_counter()
        cursor = 0
        w_f32_list: list[NDArray[np.float32]] = []
        shapes_out: list[tuple[int, int]] = []
        fallbacks = 0

        for sp in specs:
            acc = sp.accumulator
            n = acc.num_cols
            m = acc.rhs_cols
            g_sz = n * n
            c_sz = n * m

            gh = packed_cpu[cursor : cursor + g_sz].reshape(n, n).astype(np.float64)
            cursor += g_sz
            gm = packed_cpu[cursor : cursor + g_sz].reshape(n, n).astype(np.float64)
            cursor += g_sz
            gl = packed_cpu[cursor : cursor + g_sz].reshape(n, n).astype(np.float64)
            cursor += g_sz
            ch = packed_cpu[cursor : cursor + c_sz].reshape(n, m).astype(np.float64)
            cursor += c_sz
            cm = packed_cpu[cursor : cursor + c_sz].reshape(n, m).astype(np.float64)
            cursor += c_sz
            cl = packed_cpu[cursor : cursor + c_sz].reshape(n, m).astype(np.float64)
            cursor += c_sz

            g_f64 = (gh + gm + gl) * float(sp.primary_weight)
            c_f64 = (ch + cm + cl) * float(sp.primary_weight)

            if sp.secondary_accumulator is not None:
                s_gh = packed_cpu[cursor : cursor + g_sz].reshape(n, n).astype(np.float64)
                cursor += g_sz
                s_gm = packed_cpu[cursor : cursor + g_sz].reshape(n, n).astype(np.float64)
                cursor += g_sz
                s_gl = packed_cpu[cursor : cursor + g_sz].reshape(n, n).astype(np.float64)
                cursor += g_sz
                s_ch = packed_cpu[cursor : cursor + c_sz].reshape(n, m).astype(np.float64)
                cursor += c_sz
                s_cm = packed_cpu[cursor : cursor + c_sz].reshape(n, m).astype(np.float64)
                cursor += c_sz
                s_cl = packed_cpu[cursor : cursor + c_sz].reshape(n, m).astype(np.float64)
                cursor += c_sz
                g_f64 = g_f64 + (s_gh + s_gm + s_gl) * float(sp.secondary_weight)
                c_f64 = c_f64 + (s_ch + s_cm + s_cl) * float(sp.secondary_weight)

            if sp.rhs_col_slice is not None:
                c0, c1 = sp.rhs_col_slice
                c_f64 = np.ascontiguousarray(c_f64[:, c0:c1], dtype=np.float64)

            w_64, used_fb = _solve_single_f64_numpy(g_f64, c_f64, sp.ridge_scale, sp.min_ridge)
            if used_fb:
                fallbacks += 1
            w_f32_list.append(w_64.astype(np.float32).reshape(-1))
            shapes_out.append((n, int(c_f64.shape[1])))

        solve_ms = (time.perf_counter() - t_solve_start) * 1000.0

        t_out_start = time.perf_counter()
        packed_w_np = np.concatenate(w_f32_list, axis=0)
        packed_w_dev = torch.from_numpy(packed_w_np).to(device=target_dev, dtype=torch.float32)
        transfer_ms = t_transfer_in + (time.perf_counter() - t_out_start) * 1000.0

        results: list[Tensor] = []
        w_cursor = 0
        for n, m in shapes_out:
            sz = n * m
            results.append(packed_w_dev[w_cursor : w_cursor + sz].view(n, m))
            w_cursor += sz

        self.total_solves += len(specs)
        self.total_lstsq_fallbacks += fallbacks
        if self.observer is not None:
            warn_msg = (
                f"HostDenseSolver used fp64 lstsq fallback on {fallbacks}/{len(specs)} "
                f"systems in pass '{pass_name}'."
                if fallbacks > 0
                else None
            )
            self.observer.on_training_event(
                SolverEvent(
                    pass_name=pass_name,
                    num_solves=len(specs),
                    matrix_dim=max(s.accumulator.num_cols for s in specs),
                    rhs_cols=max(max(1, s.accumulator.rhs_cols) for s in specs),
                    transfer_ms=transfer_ms,
                    solve_ms=solve_ms,
                    lstsq_fallbacks=fallbacks,
                    warning=warn_msg,
                )
            )
        return results

    def solve_single(
        self,
        accumulator: ExactGramAccumulator,
        *,
        ridge_scale: float = 1.0e-6,
        min_ridge: float = 1.0e-12,
        pass_name: str = "single_solve",
        return_device: torch.device | str | None = None,
    ) -> Tensor:
        """Solves a single `ExactGramAccumulator` system via `solve_batch`."""
        res = self.solve_batch(
            [HostSolveSpec(accumulator=accumulator, ridge_scale=ridge_scale, min_ridge=min_ridge)],
            pass_name=pass_name,
            return_device=return_device,
        )
        return res[0]

    @torch.no_grad()
    def fit_orthonormal_projector(
        self,
        sub_mat: Tensor,
        max_cols: int,
    ) -> Tensor:
        """Computes the `(C_in, max_cols)` `float32` principal subspace projection matrix on
        `sub_mat.device` using on-device `ExactGramAccumulator` + host `float64` `eigh`.
        """
        _, c_in = sub_mat.shape
        if c_in <= max_cols:
            return torch.eye(c_in, device=sub_mat.device, dtype=torch.float32)

        sub_f32 = sub_mat.to(dtype=torch.float32)
        col_std = torch.sqrt(torch.mean(sub_f32 * sub_f32, dim=0, keepdim=True) + 1.0e-18)
        sub_n = sub_f32 / col_std

        acc = ExactGramAccumulator(c_in, 0, device=sub_f32.device)
        acc.add(sub_n)
        g_f64, _ = acc.to_host_f64()
        self.transfer_count += 1

        g_sym = 0.5 * (g_f64 + g_f64.T)
        _, eigvecs = np.linalg.eigh(g_sym)
        top_v = np.ascontiguousarray(eigvecs[:, -max_cols:], dtype=np.float32)
        top_v_dev = torch.from_numpy(top_v).to(device=sub_f32.device, dtype=torch.float32)
        proj = top_v_dev / col_std.T
        return proj

    @torch.no_grad()
    def project_orthonormal_basis(
        self,
        mat: Tensor,
        max_cols: int,
    ) -> Tensor:
        """Projects `(L, C_in)` tensor onto its top `max_cols` principal components."""
        l_seq, c_in = mat.shape
        if c_in <= max_cols:
            return mat
        stride = max(1, l_seq // 16_384)
        proj = self.fit_orthonormal_projector(mat[::stride], max_cols)
        return (mat.to(torch.float32) @ proj).to(mat.dtype)

    @torch.no_grad()
    def fit_interstage_projector(
        self,
        sub_norm: Tensor,
        n_lin: int,
        n_nl: int,
        *,
        res_weights: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Computes `(proj_lin, proj_nl)` on `sub_norm.device` from the exact `(M, M)` Gram of
        `sub_norm` via host `float64` `eigh` + `qr` (zero on-device `svd`).
        """
        c_in = sub_norm.shape[1]
        acc = ExactGramAccumulator(c_in, 0, device=sub_norm.device)
        acc.add(sub_norm.to(torch.float32))
        g_unw, _ = acc.to_host_f64()
        self.transfer_count += 1

        g_unw = 0.5 * (g_unw + g_unw.T)
        _, evecs_unw = np.linalg.eigh(g_unw)
        # Descending order of eigenvalues (matching right singular vectors V of SVD)
        v_unw = evecs_unw[:, ::-1]

        if res_weights is not None:
            w_np = (
                res_weights.detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
                .numpy()
                .astype(np.float64)
                .reshape(-1)
            )
            g_res = (w_np[:, None] * g_unw) * w_np[None, :]
            g_res = 0.5 * (g_res + g_res.T)
            _, evecs_res = np.linalg.eigh(g_res)
            v_res = evecs_res[:, ::-1]
            v_comb = np.concatenate([v_unw[:, :4], v_res[:, :6], v_unw[:, 4:8]], axis=1)
            q_mat, _ = np.linalg.qr(v_comb)
            p_lin_np = np.ascontiguousarray(q_mat[:, :n_lin], dtype=np.float32)
            p_nl_np = np.ascontiguousarray(q_mat[:, :n_nl], dtype=np.float32)
        else:
            p_lin_np = np.ascontiguousarray(v_unw[:, :n_lin], dtype=np.float32)
            p_nl_np = np.ascontiguousarray(v_unw[:, :n_nl], dtype=np.float32)

        dev = sub_norm.device
        return (
            torch.from_numpy(p_lin_np).to(device=dev, dtype=torch.float32),
            torch.from_numpy(p_nl_np).to(device=dev, dtype=torch.float32),
        )

    @torch.no_grad()
    def fit_target_manifold_projector(
        self,
        y_bal: Tensor,
        n_out: int = 14,
        *,
        res_weights: Tensor | None = None,
    ) -> Tensor:
        """Computes the `(M, n_out)` target manifold projector on `y_bal.device` via exact Gram
        + host `float64` `eigh` + `qr` (zero on-device `svd`).
        """
        m_col = y_bal.shape[1]
        if m_col <= n_out:
            return torch.eye(m_col, device=y_bal.device, dtype=torch.float32)

        acc = ExactGramAccumulator(m_col, 0, device=y_bal.device)
        acc.add(y_bal.to(torch.float32))
        g_unw, _ = acc.to_host_f64()
        self.transfer_count += 1

        g_unw = 0.5 * (g_unw + g_unw.T)
        _, evecs_unw = np.linalg.eigh(g_unw)
        v_unw = evecs_unw[:, ::-1]

        half_k = max(1, n_out // 2)
        if res_weights is not None:
            w_np = (
                res_weights.detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
                .numpy()
                .astype(np.float64)
                .reshape(-1)
            )
            g_res = (w_np[:, None] * g_unw) * w_np[None, :]
            g_res = 0.5 * (g_res + g_res.T)
            _, evecs_res = np.linalg.eigh(g_res)
            v_res = evecs_res[:, ::-1]
            v_comb = np.concatenate([v_unw[:, :half_k], v_res[:, : n_out - half_k]], axis=1)
            q_mat, _ = np.linalg.qr(v_comb)
            p_np = np.ascontiguousarray(q_mat[:, :n_out], dtype=np.float32)
        else:
            p_np = np.ascontiguousarray(v_unw[:, :n_out], dtype=np.float32)

        return torch.from_numpy(p_np).to(device=y_bal.device, dtype=torch.float32)


__all__ = [
    "ExactGramAccumulator",
    "HostDenseSolver",
    "HostSolveSpec",
    "argmax_i32",
    "dd_add",
    "pow2_ceil_fp32",
    "quick_two_sum",
    "recombine_f32_pair_to_f64",
    "recombine_f32_trilimb_to_f64",
    "split_f64_to_f32_pair",
    "split_f64_to_f32_trilimb",
    "split_int32_to_f32_pair",
    "td_add",
    "two_sum",
]
