"""Core DSP primitives, nonlinear B-spline/wavefolder bases, causal B-spline windows, sub-sample
USB jitter alignment, causal Wiener LTI identification, and control modulation helpers for the
selective cascade solver.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from dnam_trainer.training.numerics import (
    HostDenseSolver,
    _solve_single_f64_numpy,
    argmax_i32,
)
from dnam_trainer.training.windowing import CausalBsplineWindowSchedule


def causal_1pole_halo_samples(
    min_f_hz: float,
    sr: float = 48_000.0,
    *,
    bezout_order4: bool = False,
    decay_nepers: float = 16.0,
) -> int:
    """Returns the causal impulse-response support length (left halo in samples) for a 1-pole
    low-pass filter at `min_f_hz`.
    """
    r = float(math.exp(-2.0 * math.pi * max(float(min_f_hz), 1.0e-4) / float(sr)))
    base_k = max(64, math.ceil(-float(decay_nepers) / math.log(max(r, 1.0e-6))))
    return 7 * base_k if bezout_order4 else base_k


def apply_on_block(
    fn: Callable[[torch.Tensor], torch.Tensor],
    src: torch.Tensor,
    t0: int,
    t1: int,
    left_halo: int,
    right_halo: int = 0,
) -> torch.Tensor:
    """Evaluates causal operator `fn` on `src[max(0, t0 - left_halo) : min(L, t1 + right_halo)]`
    and slices the interior `[t0, t1)` output.
    """
    l_tot = src.shape[0]
    s0 = max(0, t0 - max(0, left_halo))
    s1 = min(l_tot, t1 + max(0, right_halo))
    ext = fn(src[s0:s1])
    cut0 = t0 - s0
    cut1 = cut0 + (t1 - t0)
    return ext[cut0:cut1]


def fast_1pole_lpf(
    sig_tc: torch.Tensor,
    f_hz: float,
    sr: float = 48_000.0,
    *,
    bezout_order4: bool = False,
    k_len: int | None = None,
) -> torch.Tensor:
    """Memory-bounded (`< 64 MB` temp buffer) causal 1-pole LPF writing directly into a
    pre-allocated output tensor. When `bezout_order4=True`, evaluates the 4th-order-flat Bezout
    complementary crossover matching `split_three_bands_np`.
    """
    was_1d = sig_tc.ndim == 1
    if was_1d:
        sig_tc = sig_tc.unsqueeze(-1)
    l_seq, c_ch = sig_tc.shape
    if l_seq <= 1:
        return sig_tc.squeeze(-1) if was_1d else sig_tc.clone()

    r = float(math.exp(-2.0 * math.pi * float(f_hz) / float(sr)))
    full_k = (
        max(64, k_len) if k_len is not None else max(64, math.ceil(-16.0 / math.log(max(r, 1e-6))))
    )
    ir_len = min(l_seq, full_k)
    t_idx = torch.arange(ir_len, device=sig_tc.device, dtype=torch.float32)
    ir = (1.0 - r) * torch.pow(r, t_idx)
    n_fft = 1 << math.ceil(math.log2(l_seq + (7 * ir_len if bezout_order4 else ir_len)))
    h_1 = torch.fft.rfft(ir, n=n_fft)
    if bezout_order4:
        q = 1.0 - h_1
        h_1 = (h_1**4) * (1.0 + q * (4.0 + q * (10.0 + 20.0 * q)))
    h_spec = h_1.unsqueeze(-1)

    out = torch.empty_like(sig_tc)
    for c0 in range(0, c_ch, 8):
        c1 = min(c_ch, c0 + 8)
        x_spec = torch.fft.rfft(sig_tc[:, c0:c1].to(torch.float32), n=n_fft, dim=0)
        out[:, c0:c1] = torch.fft.irfft(x_spec * h_spec, n=n_fft, dim=0)[:l_seq, :].to(sig_tc.dtype)
    return out.squeeze(-1) if was_1d else out


def split_bezout_3band(
    sig_full: torch.Tensor,
    sr: float = 48_000.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Splits `sig_full` into complementary 4th-order Bezout bands (`LF < 250 Hz`,
    `MF 250-2500 Hz`, `HF > 2500 Hz`).
    """
    b_lf = fast_1pole_lpf(sig_full, 250.0, sr=sr, bezout_order4=True)
    b_l2500 = fast_1pole_lpf(sig_full, 2500.0, sr=sr, bezout_order4=True)
    b_mf = b_l2500 - b_lf
    b_hf = sig_full - b_l2500
    return b_lf, b_mf, b_hf


def solve_cholesky_ridge(
    gram: torch.Tensor,
    rhs: torch.Tensor,
    *,
    ridge_scale: float = 1e-6,
    min_ridge: float = 1e-12,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Solves `(gram + reg * I) W = rhs` in host `float64` via Cholesky factorization with `lstsq`
    fallback, returning a `float32` tensor on `gram.device` without allocating 64-bit device
    tensors.
    """
    was_1d = rhs.ndim == 1
    rhs_2d = rhs.unsqueeze(-1) if was_1d else rhs
    g_np = (
        gram.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().astype(np.float64)
    )
    c_np = (
        rhs_2d.detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .numpy()
        .astype(np.float64)
    )
    w_64, _ = _solve_single_f64_numpy(g_np, c_np, ridge_scale, min_ridge)
    w_dev = torch.from_numpy(w_64.astype(np.float32)).to(device=gram.device, dtype=out_dtype)
    return w_dev.squeeze(-1) if was_1d else w_dev


def rms_balance_cols(mat: torch.Tensor, target_rms: float = 0.12) -> torch.Tensor:
    """Scales each column of `mat` to `target_rms` so isotropic ridge regularization in Stage 4
    (`solve_all_tiers_kronecker_head_float64`) treats broadband, LF, MF, HF, and 48 kHz residual
    polish columns with equal weight.
    """
    col_rms = torch.sqrt(torch.mean(mat * mat, dim=0, keepdim=True) + 1e-18)
    scale = torch.where(
        col_rms > 1e-7, target_rms / torch.clamp(col_rms, min=1e-7), torch.zeros_like(col_rms)
    )
    return mat * scale


def fast_poly_adaa_batch(
    bands_tc: torch.Tensor,
    drives: Sequence[float],
    sr: float = 48_000.0,
) -> torch.Tensor:
    """Memory-bounded (`< 120 MB` temp buffer) vectorized `1x B-Spline PolyADAA` (`odd` +
    AC-coupled `even`) on `(L, C_bands)` tensor.
    """
    l_seq, c_b = bands_tc.shape
    n_d = len(drives)
    d_t = torch.tensor(list(drives), dtype=bands_tc.dtype, device=bands_tc.device).view(1, 1, n_d)
    out = torch.empty((l_seq, c_b * n_d * 2), dtype=bands_tc.dtype, device=bands_tc.device)
    half_cols = c_b * n_d
    for b0 in range(0, c_b, 8):
        b1 = min(c_b, b0 + 8)
        w_b = b1 - b0
        u = torch.clamp((2.0 / 3.0) * bands_tc[:, b0:b1].unsqueeze(-1) * d_t, -1.0, 1.0).reshape(
            l_seq, w_b * n_d
        )
        u0 = torch.zeros_like(u)
        if l_seq > 1:
            u0[1:] = u[:-1]
        um = 0.5 * (u0 + u)
        du = u - u0
        del u0, u
        du2 = du * du
        del du
        um2 = um * um
        odd = um * (1.5 - 0.5 * um2 - 0.125 * du2)
        del um
        even = um2 * (1.0 - 0.5 * um2) + du2 * ((1.0 / 12.0) - 0.25 * um2 - (1.0 / 80.0) * du2)
        del um2, du2
        c_start = b0 * n_d
        c_end = b1 * n_d
        out[:, c_start:c_end] = odd
        del odd
        out[:, half_cols + c_start : half_cols + c_end] = even - fast_1pole_lpf(even, 5.0, sr=sr)
        del even
    return out


def fast_poly_inst_batch(
    bands_tc: torch.Tensor,
    drives: Sequence[float],
    sr: float = 48_000.0,
) -> torch.Tensor:
    """Memory-bounded (`< 120 MB` temp buffer) vectorized division-free instantaneous `B-Spline`
    (`odd` + AC-coupled `even`) on `(L, C_bands)` tensor.
    """
    l_seq, c_b = bands_tc.shape
    n_d = len(drives)
    d_t = torch.tensor(list(drives), dtype=bands_tc.dtype, device=bands_tc.device).view(1, 1, n_d)
    out = torch.empty((l_seq, c_b * n_d * 2), dtype=bands_tc.dtype, device=bands_tc.device)
    half_cols = c_b * n_d
    for b0 in range(0, c_b, 8):
        b1 = min(c_b, b0 + 8)
        w_b = b1 - b0
        u = torch.clamp((2.0 / 3.0) * bands_tc[:, b0:b1].unsqueeze(-1) * d_t, -1.0, 1.0).reshape(
            l_seq, w_b * n_d
        )
        u2 = u * u
        g_odd = u * (1.5 - 0.5 * u2)
        del u
        g_even = u2 * (1.0 - 0.5 * u2)
        del u2
        c_start = b0 * n_d
        c_end = b1 * n_d
        out[:, c_start:c_end] = g_odd
        del g_odd
        out[:, half_cols + c_start : half_cols + c_end] = g_even - fast_1pole_lpf(
            g_even, 5.0, sr=sr
        )
        del g_even
    return out


def fast_wavefolder_inst_batch(
    bands_tc: torch.Tensor,
    drives: Sequence[float],
    sr: float = 48_000.0,
) -> torch.Tensor:
    """Division-free $C^2$ compactly-supported wavefolder B-spline basis:
    $w_{\\text{odd}}(u) = u(1 - u^2)^2$ and AC-coupled $w_{\\text{even}}(u) = u^2(1 - u^2)^2$
    for $u = \\operatorname{clip}(\\frac{2}{3} d x, -1, 1)$. Zero `FSQRT` / `FDIV` / `libm`.
    """
    l_seq, c_b = bands_tc.shape
    n_d = len(drives)
    d_t = torch.tensor(list(drives), dtype=bands_tc.dtype, device=bands_tc.device).view(1, 1, n_d)
    out = torch.empty((l_seq, c_b * n_d * 2), dtype=bands_tc.dtype, device=bands_tc.device)
    half_cols = c_b * n_d
    for b0 in range(0, c_b, 8):
        b1 = min(c_b, b0 + 8)
        w_b = b1 - b0
        u = torch.clamp((2.0 / 3.0) * bands_tc[:, b0:b1].unsqueeze(-1) * d_t, -1.0, 1.0).reshape(
            l_seq, w_b * n_d
        )
        u2 = u * u
        one_m_u2 = 1.0 - u2
        lobe = one_m_u2 * one_m_u2
        w_odd = u * lobe
        w_even = u2 * lobe
        del u, u2, one_m_u2, lobe
        c_start = b0 * n_d
        c_end = b1 * n_d
        out[:, c_start:c_end] = w_odd
        del w_odd
        out[:, half_cols + c_start : half_cols + c_end] = w_even - fast_1pole_lpf(
            w_even, 5.0, sr=sr
        )
        del w_even
    return out


@torch.no_grad()
def project_orthonormal_basis(mat: torch.Tensor, max_cols: int) -> torch.Tensor:
    """Projects `(L, C_in)` tensor onto its top `max_cols` orthonormal principal components
    using on-device `ExactGramAccumulator` + host `float64` `eigh`.
    """
    return HostDenseSolver().project_orthonormal_basis(mat, max_cols)


@torch.no_grad()
def measure_and_shift_usb_jitter_tensor(
    x_t: torch.Tensor,
    y_t: torch.Tensor,
    *,
    max_jitter: int = 32,
    min_len: int = 48_000,
    cached_lag: float | None = None,
) -> tuple[torch.Tensor, float, float]:
    """On-device 32-bit (`float32` / `complex64`) sub-sample USB full-duplex stream startup jitter
    alignment using high-frequency pre-emphasized GCC (`x[n] - 0.95 x[n-1]`), with clipped fuzz
    proxy fallback when linear correlation `< 0.35`.
    """
    x_f32 = x_t.reshape(-1).to(dtype=torch.float32)
    y_f32 = y_t.reshape(-1).to(device=x_f32.device, dtype=torch.float32)
    n = x_f32.shape[0]
    if n < min_len:
        return y_f32.clone(), 0.0, 1.0
    if cached_lag is not None:
        lag_val = float(cached_lag)
        if abs(lag_val) < 1.0e-4:
            return y_f32.clone(), 0.0, 1.0
        return shift_signal_fft_1d(y_f32, lag_val), lag_val, 1.0

    dev = x_f32.device
    x_pre = x_f32[1:] - 0.95 * x_f32[:-1]
    y_pre = y_f32[1:] - 0.95 * y_f32[:-1]
    n_fft = 1 << (2 * n).bit_length()
    x_spec = torch.fft.rfft(x_pre, n=n_fft)
    y_spec = torch.fft.rfft(y_pre, n=n_fft)
    cc = torch.fft.irfft(y_spec * torch.conj(x_spec), n=n_fft)
    denom = (
        float(torch.linalg.vector_norm(x_pre).item() * torch.linalg.vector_norm(y_pre).item())
        + 1.0e-12
    )
    lags_i32 = torch.arange(-max_jitter, max_jitter + 1, device=dev, dtype=torch.int32)
    lag_mod_i32 = torch.remainder(lags_i32, n_fft)
    vals_t = cc[lag_mod_i32] / denom
    best_idx = int(argmax_i32(vals_t.abs()).item())
    best_lag = int(lags_i32[best_idx].item())
    best_corr = float(vals_t[best_idx].item())

    if abs(best_corr) < 0.35:
        x_hp = x_f32 - fast_1pole_lpf(x_f32, 80.0, sr=48_000.0)
        proxy = fast_1pole_lpf(torch.tanh(10.0 * x_hp), 800.0, sr=48_000.0)
        p_fuzz = torch.fft.rfft(proxy, n=n_fft)
        y_raw = torch.fft.rfft(y_f32, n=n_fft)
        cc_f = torch.fft.irfft(y_raw * torch.conj(p_fuzz), n=n_fft)
        denom_f = (
            float(torch.linalg.vector_norm(proxy).item() * torch.linalg.vector_norm(y_f32).item())
            + 1.0e-12
        )
        vals_f = cc_f[lag_mod_i32] / denom_f
        idx_f = int(argmax_i32(vals_f.abs()).item())
        corr_f = float(vals_f[idx_f].item())
        if abs(corr_f) >= 0.45:
            vals_t = vals_f
            best_idx = idx_f
            best_lag = int(lags_i32[best_idx].item())
            best_corr = corr_f
        else:
            return y_f32.clone(), 0.0, best_corr

    frac = 0.0
    n_lags = int(vals_t.shape[0])
    if 0 < best_idx < n_lags - 1:
        y0 = abs(float(vals_t[best_idx - 1].item()))
        y1 = abs(float(vals_t[best_idx].item()))
        y2 = abs(float(vals_t[best_idx + 1].item()))
        d = y0 - 2.0 * y1 + y2
        if abs(d) > 1.0e-9:
            frac = max(-0.5, min(0.5, 0.5 * (y0 - y2) / d))

    total_lag = float(best_lag) + frac
    if abs(total_lag) < 1.0e-4:
        return y_f32.clone(), 0.0, best_corr
    y_aligned = shift_signal_fft_1d(y_f32, total_lag)
    return y_aligned, total_lag, best_corr


def measure_and_shift_usb_jitter(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    max_jitter: int = 32,
    min_len: int = 48_000,
    device: torch.device | str = "cpu",
    cached_lag: float | None = None,
) -> tuple[NDArray[np.float64], float, float]:
    """Sub-sample USB full-duplex stream startup jitter alignment using high-frequency
    pre-emphasized GCC (`x[n] - 0.95 x[n-1]`), with clipped fuzz proxy fallback when linear
    correlation `< 0.35`.
    """
    if len(x) < min_len:
        return y.copy(), 0.0, 1.0
    x_t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device=device)
    y_t = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)).to(device=device)
    y_al_t, total_lag, best_corr = measure_and_shift_usb_jitter_tensor(
        x_t,
        y_t,
        max_jitter=max_jitter,
        min_len=min_len,
        cached_lag=cached_lag,
    )
    if abs(total_lag) < 1.0e-4:
        return y.copy(), 0.0, best_corr
    y_al_np = (
        y_al_t.detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .numpy()
        .astype(np.float64)
    )
    return y_al_np, total_lag, best_corr


def shift_signal_fft_1d(sig: torch.Tensor, shift_samples: float) -> torch.Tensor:
    """Applies an exact fractional sample shift (`+shift_samples` advances `sig`) via 32-bit FFT
    phase rotation (`float32` / `complex64`).
    """
    if abs(shift_samples) < 1e-4 or sig.shape[0] < 8:
        return sig
    n = sig.shape[0]
    n_fft = 1 << (2 * n).bit_length()
    spec = torch.fft.rfft(sig.to(torch.float32), n=n_fft)
    freqs = torch.arange(spec.shape[0], device=sig.device, dtype=torch.float32)
    angle = (2.0 * math.pi * float(shift_samples) / float(n_fft)) * freqs
    phase = torch.complex(torch.cos(angle), torch.sin(angle))
    if n_fft % 2 == 0:
        phase[-1] = torch.complex(phase[-1].real, torch.zeros_like(phase[-1].real))
    shifted = torch.fft.irfft(spec * phase, n=n_fft)[:n]
    return shifted.to(sig.dtype)


def build_causal_bspline_window_weights(
    x: torch.Tensor,
    win_bounds: Sequence[tuple[int, int]],
    *,
    overlap: int = 256,
    sr: float = 48_000.0,
) -> torch.Tensor:
    """Computes strictly monotonic causal 1-pole state
    $e_{\\text{cum}}[n] = \\operatorname{LPF}_{0.008\\text{ Hz}}(1 + |x[n]|)$ and evaluates the
    $C^0$ B-spline trapezoidal partition-of-unity basis $\\Lambda_w(e_{\\text{cum}}[n])$
    supported strictly inside $[k_0 - \\text{overlap}, k_1 + \\text{overlap}]$.
    """
    return CausalBsplineWindowSchedule.from_signal(
        x,
        win_bounds,
        overlap=overlap,
        sr=sr,
    ).to_dense()


def build_fast_refinement_gates(x: torch.Tensor, sr: float = 48_000.0) -> list[torch.Tensor]:
    """15 causal state-space envelope and spectral co-occurrence gates."""
    x_low = fast_1pole_lpf(x, 150.0, sr=sr)
    x_mid = fast_1pole_lpf(x - x_low, 1200.0, sr=sr)
    x_hi = x - fast_1pole_lpf(x, 1200.0, sr=sr)
    e_fast = fast_1pole_lpf(torch.abs(x), 80.0, sr=sr)
    e_med = fast_1pole_lpf(torch.abs(x), 15.0, sr=sr)
    e_slow = fast_1pole_lpf(torch.abs(x), 3.0, sr=sr)
    e_vslow = fast_1pole_lpf(torch.abs(x), 0.7, sr=sr)
    e_low = fast_1pole_lpf(torch.abs(x_low), 10.0, sr=sr)
    e_mid = fast_1pole_lpf(torch.abs(x_mid), 15.0, sr=sr)
    e_hi = fast_1pole_lpf(torch.abs(x_hi), 25.0, sr=sr)
    g_att = torch.relu(e_slow - e_vslow)
    g_rel = torch.relu(e_vslow - e_slow)
    return [
        torch.ones_like(e_fast),
        e_fast,
        e_med,
        e_slow,
        e_vslow,
        e_low,
        e_mid,
        e_hi,
        g_att,
        g_rel,
        e_low * e_mid,
        e_low * e_hi,
        e_mid * e_hi,
        e_fast * e_fast,
        e_slow * e_slow,
    ]


@torch.no_grad()
def solve_lti_cabinet_and_space_stage_with_ir(
    x: torch.Tensor,
    y7_full: torch.Tensor,
    *,
    max_ir_len: int = 8192,
    sr: float = 48_000.0,
    cached_ir: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """On-device 32-bit (`float32` / `complex64`) acausal-calibrated causal Wiener LTI impulse
    response identification for Speaker Cabinets (`IR`s), Parametric EQs, and Acoustic Spaces.

    Processes one target column at a time so multi-stem 30 s FFT buffers remain `<= 34 MB`.
    Also returns the learned `(k_lti, M)` impulse response tensor for Arena 4 caching.
    """
    x_f32 = x.reshape(-1).to(dtype=torch.float32)
    dev = x_f32.device
    was_1d = y7_full.ndim == 1
    y_2d = (y7_full.unsqueeze(-1) if was_1d else y7_full).to(device=dev, dtype=torch.float32)
    l_tot, m_cols = y_2d.shape
    k_lti = min(max_ir_len, max(64, l_tot // 2))
    n_fft = 1 << (2 * l_tot - 1).bit_length()

    x_spec = torch.fft.rfft(x_f32, n=n_fft)
    s_xx = x_spec.real * x_spec.real + x_spec.imag * x_spec.imag
    mean_sxx = max(1.0e-18, float(torch.mean(s_xx).item()))
    inv_denom = torch.conj(x_spec) / (s_xx + 1.0e-6 * mean_sxx)

    ac_len = min(512, max(32, n_fft // 8))
    ac_end = min(16, ac_len // 2)
    fade_len = max(8, k_lti // 12)
    w_fade = torch.ones(k_lti, device=dev, dtype=torch.float32)
    t_fade = torch.linspace(0.0, math.pi, fade_len, device=dev, dtype=torch.float32)
    w_fade[k_lti - fade_len :] = 0.5 * (1.0 + torch.cos(t_fade))

    z_lti_full = torch.empty((l_tot, m_cols), device=dev, dtype=torch.float32)
    h_lti_out = torch.empty((k_lti, m_cols), device=dev, dtype=torch.float32)

    have_cached = (
        cached_ir is not None and cached_ir.ndim == 2 and cached_ir.shape == (k_lti, m_cols)
    )

    for col in range(m_cols):
        if have_cached:
            assert cached_ir is not None
            h_col = cached_ir[:, col].to(device=dev, dtype=torch.float32)
            h_lti_out[:, col] = h_col
            h_spec = torch.fft.rfft(h_col, n=n_fft)
            z_lti_full[:, col] = torch.fft.irfft(h_spec * x_spec, n=n_fft)[:l_tot]
            continue

        y_col = y_2d[:, col]
        y_spec = torch.fft.rfft(y_col, n=n_fft)
        h_full = torch.fft.irfft(inv_denom * y_spec, n=n_fft)
        del y_spec

        h_acausal = h_full[n_fft - ac_len : n_fft - ac_end]
        sigma_ac_sq = torch.mean(h_acausal * h_acausal)
        h_causal = h_full[:k_lti].clone()
        del h_full

        p_tap = h_causal * h_causal
        p_pad = F.pad(p_tap.view(1, 1, k_lti), (4, 4), mode="replicate")
        p_smooth = F.avg_pool1d(p_pad, kernel_size=9, stride=1).view(k_lti)
        thresh = 2.0 * sigma_ac_sq
        w_shrink = torch.clamp(p_smooth - thresh, min=0.0) / (p_smooth + thresh + 1.0e-20)
        w_mask = w_shrink * w_fade

        h_col = h_causal * w_mask
        h_spec = torch.fft.rfft(h_col, n=n_fft)
        z_col = torch.fft.irfft(h_spec * x_spec, n=n_fft)[:l_tot]

        for _ in range(5):
            r_col = y_col - z_col
            r_spec = torch.fft.rfft(r_col, n=n_fft)
            dh = torch.fft.irfft(inv_denom * r_spec, n=n_fft)[:k_lti] * w_mask
            dh_spec = torch.fft.rfft(dh, n=n_fft)
            dz = torch.fft.irfft(dh_spec * x_spec, n=n_fft)[:l_tot]
            alpha = torch.clamp(
                torch.sum(r_col * dz) / (torch.sum(dz * dz) + 1.0e-20),
                0.0,
                2.0,
            )
            h_col = h_col + alpha * dh
            z_col = z_col + alpha * dz

        z_lti_full[:, col] = z_col
        h_lti_out[:, col] = h_col

    del x_spec, s_xx, inv_denom
    z_lti_lf, z_lti_mf, z_lti_hf = split_bezout_3band(z_lti_full, sr=sr)
    if was_1d:
        return (
            z_lti_full.squeeze(-1),
            z_lti_lf.squeeze(-1),
            z_lti_mf.squeeze(-1),
            z_lti_hf.squeeze(-1),
            h_lti_out,
        )
    return z_lti_full, z_lti_lf, z_lti_mf, z_lti_hf, h_lti_out


@torch.no_grad()
def solve_lti_cabinet_and_space_stage(
    x: torch.Tensor,
    y7_full: torch.Tensor,
    *,
    max_ir_len: int = 8192,
    sr: float = 48_000.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Acausal-calibrated causal Wiener LTI impulse response identification for Speaker Cabinets
    (`IR`s), Parametric EQs, and Acoustic Spaces (Early Reflections + Reverberant Decay).
    """
    z_full, z_lf, z_mf, z_hf, _ = solve_lti_cabinet_and_space_stage_with_ir(
        x,
        y7_full,
        max_ir_len=max_ir_len,
        sr=sr,
    )
    return z_full, z_lf, z_mf, z_hf


def compute_tier32_modulator_vectors(
    u_ctrl_3: Sequence[float],
    c_knots: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Computes the two 32-element control modulator vectors `(f0, f1)` for a 32-channel tier base
    block (`32 * 2 = 64` channels per tier).
    """
    u0 = float(u_ctrl_3[0]) if len(u_ctrl_3) > 0 else 0.0
    u1 = float(u_ctrl_3[1]) if len(u_ctrl_3) > 1 else 0.0
    u2 = float(u_ctrl_3[2]) if len(u_ctrl_3) > 2 else 0.0
    f0 = np.ones(32, dtype=np.float64)
    if c_knots is not None and c_knots.shape == (32, 3):
        u_vec = np.array([u0, u1, u2], dtype=np.float64)[None, :]
        dist_sq = np.sum((c_knots - u_vec) ** 2, axis=1)
        f1 = np.exp(-2000.0 * dist_sq)
    else:
        j_idx = np.arange(1, 33, dtype=np.float64)
        f1 = np.cos(j_idx * u0 + 0.7 * j_idx * u1 + 0.3 * j_idx * u2)
    return f0, f1


def modulate_tier_blocks_with_controls(
    b_lite_16: NDArray[np.float32],
    b_full_16: NDArray[np.float32],
    b_ultra_16: NDArray[np.float32],
    u_ctrl_3: Sequence[float],
    *,
    c_knots: NDArray[np.float64] | None = None,
) -> NDArray[np.float32]:
    """Forms the 192-D bilinear control-modulated feature tensor `(T, 192)` from 16-channel or
    32-channel tier base blocks.
    """
    if b_lite_16.shape[1] == 32:
        f0, f1 = compute_tier32_modulator_vectors(u_ctrl_3, c_knots)
        f0_32 = f0.astype(np.float32)[None, :]
        f1_32 = f1.astype(np.float32)[None, :]
        f_lite = np.concatenate([b_lite_16 * f0_32, b_lite_16 * f1_32], axis=-1).astype(np.float32)
        f_full = np.concatenate([b_full_16 * f0_32, b_full_16 * f1_32], axis=-1).astype(np.float32)
        f_ultra = np.concatenate([b_ultra_16 * f0_32, b_ultra_16 * f1_32], axis=-1).astype(
            np.float32
        )
        return np.concatenate([f_lite, f_full, f_ultra], axis=-1)

    u0 = float(u_ctrl_3[0]) if len(u_ctrl_3) > 0 else 0.0
    u1 = float(u_ctrl_3[1]) if len(u_ctrl_3) > 1 else 0.0
    u2 = float(u_ctrl_3[2]) if len(u_ctrl_3) > 2 else 0.0
    factors = (1.0, u0, u1, u2)
    f_lite = np.concatenate([m * b_lite_16 for m in factors], axis=-1).astype(np.float32)
    f_full = np.concatenate([m * b_full_16 for m in factors], axis=-1).astype(np.float32)
    f_ultra = np.concatenate([m * b_ultra_16 for m in factors], axis=-1).astype(np.float32)
    return np.concatenate([f_lite, f_full, f_ultra], axis=-1)
