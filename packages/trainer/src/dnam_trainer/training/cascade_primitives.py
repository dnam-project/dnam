"""Core DSP primitives, nonlinear B-spline/wavefolder bases, causal B-spline windows, sub-sample
USB jitter alignment, causal Wiener LTI identification, and control modulation helpers for the
selective cascade solver.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray


def fast_1pole_lpf(
    sig_tc: torch.Tensor,
    f_hz: float,
    sr: float = 48_000.0,
    *,
    bezout_order4: bool = False,
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
    k_len = min(l_seq, max(64, math.ceil(-16.0 / math.log(max(r, 1e-6)))))
    t_idx = torch.arange(k_len, device=sig_tc.device, dtype=torch.float32)
    ir = (1.0 - r) * torch.pow(r, t_idx)
    n_fft = 1 << math.ceil(math.log2(l_seq + (7 * k_len if bezout_order4 else k_len)))
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
    """Solves `(gram + reg * I) W = rhs` in `float64` via Cholesky factorization with `lstsq`
    fallback.
    """
    diag_mean = float(torch.diagonal(gram).mean().item())
    reg = max(min_ridge, ridge_scale * diag_mean)
    eye = torch.eye(gram.shape[0], dtype=torch.float64, device=gram.device)
    try:
        l_mat = torch.linalg.cholesky(gram + reg * eye)
        return torch.linalg.solve(l_mat.T, torch.linalg.solve(l_mat, rhs)).to(out_dtype)
    except RuntimeError:
        return torch.linalg.lstsq(gram + reg * eye, rhs).solution.to(out_dtype)


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
    """Projects `(L, C_in)` tensor onto its top `max_cols` orthonormal principal components."""
    l_seq, c_in = mat.shape
    if c_in <= max_cols:
        return mat
    stride = max(1, l_seq // 16_384)
    sub = mat[::stride].to(torch.float64)
    col_std = torch.sqrt(torch.mean(sub * sub, dim=0, keepdim=True) + 1e-18)
    sub_n = sub / col_std
    gram = sub_n.T @ sub_n
    _, eigvecs = torch.linalg.eigh(gram)
    top_v = eigvecs[:, -max_cols:]
    proj = (top_v / col_std.T).to(mat.dtype)
    del sub, sub_n, gram, eigvecs, top_v
    return mat @ proj


def measure_and_shift_usb_jitter(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    max_jitter: int = 32,
    min_len: int = 48_000,
) -> tuple[NDArray[np.float64], float, float]:
    """Sub-sample USB full-duplex stream startup jitter alignment using high-frequency
    pre-emphasized GCC (`x[n] - 0.95 x[n-1]`), with clipped fuzz proxy fallback when linear
    correlation `< 0.35`.
    """
    n = len(x)
    if n < min_len:
        return y.copy(), 0.0, 1.0
    x_pre = x[1:] - 0.95 * x[:-1]
    y_pre = y[1:] - 0.95 * y[:-1]
    n_fft = 1 << (2 * n).bit_length()
    X = np.fft.rfft(x_pre, n=n_fft)
    Y = np.fft.rfft(y_pre, n=n_fft)
    cc = np.fft.irfft(Y * np.conj(X), n=n_fft)
    denom = float(np.linalg.norm(x_pre) * np.linalg.norm(y_pre)) + 1e-12
    lags = np.arange(-max_jitter, max_jitter + 1)
    vals = np.array([cc[lag % n_fft] / denom for lag in lags], dtype=np.float64)
    best_idx = int(np.argmax(np.abs(vals)))
    best_lag = int(lags[best_idx])
    best_corr = float(vals[best_idx])
    if abs(best_corr) < 0.35:
        x_t = torch.as_tensor(x, dtype=torch.float32)
        x_hp = x_t - fast_1pole_lpf(x_t, 80.0, sr=48_000.0)
        proxy = (
            fast_1pole_lpf(torch.tanh(10.0 * x_hp), 800.0, sr=48_000.0).numpy().astype(np.float64)
        )
        P_fuzz = np.fft.rfft(proxy, n=n_fft)
        Y_raw = np.fft.rfft(y, n=n_fft)
        cc_f = np.fft.irfft(Y_raw * np.conj(P_fuzz), n=n_fft)
        denom_f = float(np.linalg.norm(proxy) * np.linalg.norm(y)) + 1e-12
        vals_f = np.array([cc_f[lag % n_fft] / denom_f for lag in lags], dtype=np.float64)
        idx_f = int(np.argmax(np.abs(vals_f)))
        if abs(float(vals_f[idx_f])) >= 0.45:
            vals = vals_f
            best_idx = idx_f
            best_lag = int(lags[best_idx])
            best_corr = float(vals[best_idx])
        else:
            return y.copy(), 0.0, best_corr

    frac = 0.0
    if 0 < best_idx < len(vals) - 1:
        y0 = abs(vals[best_idx - 1])
        y1 = abs(vals[best_idx])
        y2 = abs(vals[best_idx + 1])
        d = y0 - 2.0 * y1 + y2
        if abs(d) > 1e-9:
            frac = float(np.clip(0.5 * (y0 - y2) / d, -0.5, 0.5))

    total_lag = float(best_lag) + frac
    if abs(total_lag) < 1e-4:
        return y.copy(), 0.0, best_corr
    Y_full = np.fft.rfft(y, n=n_fft)
    freqs = np.arange(len(Y_full), dtype=np.float64)
    phase = np.exp(1j * 2.0 * np.pi * freqs * total_lag / float(n_fft))
    if n_fft % 2 == 0:
        phase[-1] = np.real(phase[-1])
    y_aligned = np.fft.irfft(Y_full * phase, n=n_fft)[:n]
    return y_aligned, total_lag, best_corr


def shift_signal_fft_1d(sig: torch.Tensor, shift_samples: float) -> torch.Tensor:
    """Applies an exact fractional sample shift (`+shift_samples` advances `sig`) via FFT phase
    rotation.
    """
    if abs(shift_samples) < 1e-4 or sig.shape[0] < 8:
        return sig
    n = sig.shape[0]
    n_fft = 1 << (2 * n).bit_length()
    spec = torch.fft.rfft(sig.to(torch.float64), n=n_fft)
    freqs = torch.arange(spec.shape[0], device=sig.device, dtype=torch.float64)
    angle = 2.0 * math.pi * freqs * float(shift_samples) / float(n_fft)
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
    n_wins = len(win_bounds)
    l_tot = x.shape[0]
    if n_wins <= 1:
        return torch.ones((l_tot, 1), dtype=torch.float32, device=x.device)

    drv = 1.0 + torch.abs(x)
    e_cum = fast_1pole_lpf(drv, 0.008, sr=sr)
    weights = torch.zeros((l_tot, n_wins), dtype=torch.float32, device=x.device)
    for w, (k0, k1) in enumerate(win_bounds):
        if w == 0:
            left_ramp = torch.ones_like(e_cum)
        else:
            ka = e_cum[max(0, k0 - overlap)]
            kb = e_cum[min(l_tot - 1, k0 + overlap)]
            left_ramp = torch.clamp((e_cum - ka) / (kb - ka + 1e-12), 0.0, 1.0)
        if w == n_wins - 1:
            right_ramp = torch.ones_like(e_cum)
        else:
            kc = e_cum[max(0, k1 - overlap)]
            kd = e_cum[min(l_tot - 1, k1 + overlap)]
            right_ramp = torch.clamp((kd - e_cum) / (kd - kc + 1e-12), 0.0, 1.0)
        weights[:, w] = torch.minimum(left_ramp, right_ramp)
    w_sum = torch.sum(weights, dim=-1, keepdim=True).clamp(min=1e-12)
    return weights / w_sum


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
    l_tot = x.shape[0]
    k_lti = min(max_ir_len, max(64, l_tot // 2))
    n_fft = 1 << (2 * l_tot - 1).bit_length()

    x64 = x.to(torch.float64)
    X = torch.fft.rfft(x64, n=n_fft)
    Y = torch.fft.rfft(y7_full.to(torch.float64), n=n_fft, dim=0)
    S_xx = X.real * X.real + X.imag * X.imag
    mean_sxx = max(1e-18, float(torch.mean(S_xx).item()))

    inv_denom = torch.conj(X).unsqueeze(-1) / (S_xx.unsqueeze(-1) + 1e-6 * mean_sxx)
    H_raw = inv_denom * Y
    h_full = torch.fft.irfft(H_raw, n=n_fft, dim=0)
    del Y, H_raw

    ac_len = min(512, max(32, n_fft // 8))
    ac_end = min(16, ac_len // 2)
    h_acausal = h_full[n_fft - ac_len : n_fft - ac_end]
    sigma_ac_sq = torch.mean(h_acausal * h_acausal, dim=0, keepdim=True)

    h_causal = h_full[:k_lti].clone()
    del h_full

    p_tap = h_causal * h_causal
    p_pad = F.pad(p_tap.T.unsqueeze(1), (4, 4), mode="replicate")
    p_smooth = F.avg_pool1d(p_pad, kernel_size=9, stride=1).squeeze(1).T
    thresh = 2.0 * sigma_ac_sq
    w_shrink = torch.clamp(p_smooth - thresh, min=0.0) / (p_smooth + thresh + 1e-24)

    fade_len = max(8, k_lti // 12)
    w_fade = torch.ones((k_lti, 1), dtype=torch.float64)
    t_fade = torch.linspace(0.0, math.pi, fade_len, dtype=torch.float64).unsqueeze(-1)
    w_fade[k_lti - fade_len :] = 0.5 * (1.0 + torch.cos(t_fade))
    w_mask = w_shrink * w_fade

    y_f64 = y7_full.to(torch.float64)
    h_lti = h_causal * w_mask
    H_lti = torch.fft.rfft(h_lti, n=n_fft, dim=0)
    z_cur = torch.fft.irfft(H_lti * X.unsqueeze(-1), n=n_fft, dim=0)[:l_tot]
    for _ in range(5):
        r_cur = y_f64 - z_cur
        R_fft = torch.fft.rfft(r_cur, n=n_fft, dim=0)
        dh = torch.fft.irfft(inv_denom * R_fft, n=n_fft, dim=0)[:k_lti] * w_mask
        DH_fft = torch.fft.rfft(dh, n=n_fft, dim=0)
        dz = torch.fft.irfft(DH_fft * X.unsqueeze(-1), n=n_fft, dim=0)[:l_tot]
        alpha = torch.clamp(
            torch.sum(r_cur * dz, dim=0, keepdim=True)
            / (torch.sum(dz * dz, dim=0, keepdim=True) + 1e-24),
            0.0,
            2.0,
        )
        h_lti = h_lti + alpha * dh
        z_cur = z_cur + alpha * dz

    z_lti_full = z_cur.to(torch.float32)
    del X, inv_denom, h_causal, h_lti, z_cur, y_f64
    z_lti_lf, z_lti_mf, z_lti_hf = split_bezout_3band(z_lti_full, sr=sr)
    return z_lti_full, z_lti_lf, z_lti_mf, z_lti_hf


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
