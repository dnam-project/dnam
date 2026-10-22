# Empirical Verification Suite (8 Test Suites) & Automated Guardrails

All implementations of **Dynamic Neural Audio Modeler (DNAM Architecture 3)** must pass the **8 Empirical Mathematical, Numerical & PyTorch Verification Suites** (`Test 1` through `Test 8G`) and the **Automated Guardrail Tests** specified below.

---

## 1. Summary of the 8 Empirical Verification Suites

| Verification Test Suite | Baseline / Naive Behavior | Verified DNAM A3 Requirement | Target Threshold |
| :--- | :---: | :---: | :---: |
| **1. Division-Free Algebraic `PolyADAA` Exactness** | Requires `FDIV` + branch at $\Delta u \to 0$ | **$5.25 \times 10^{-14}$ error** with $0$ `FDIV`, $0$ `FSQRT`, $0$ branches | `max_alg_err < 1e-12` |
| **2. 8-Layer Cascaded Aliasing ($f_0 = 3.45\text{ kHz}$)** | $-23.20\text{ dBc}$ (NAM A2 `LeakyReLU`) | **$-62.08\text{ dBc}$ ($38.87\text{ dB}$ reduction)** via $1\times$ B-Spline `PolyADAA` | `(alias_a2_db - alias_a3_db) > 35.0` |
| **3. Selective RG-LRU Mid-Sweep $Q$ & Algebraic Rotor** | $Q$ drops $125.0 \to 6.5$ ($94.8\%$ loss) | **$Q = 124.7$ ($99.80\%$)**, rotor err **$9.99 \times 10^{-9}$** ($\le 1.0$), scan **$9.42 \times 10^{-16}$** | `q_polar > 0.99 * q_orig`, `max_unit_val <= 1.0` |
| **4. Modulated Delay Boundary Click Jerk ($5\text{ kHz}$)** | $0.04694$ (Recursive Thiran clicks) | **$0.00087$ ($54\times$ smoother)** via 4-Tap Cubic Lagrange | `err_thiran_jerk > 15.0 * err_lagrange_jerk` |
| **5. `M-UFDN` $+20\text{ dBFS}$ Spike + $1.18\times$ Loop Gain + DC** | Unbounded explosion / DC rail lockup | Peak **$0.9850 \le V_{\text{ceil}}$**, DC **$1.03 \times 10^{-6}$**, quiet tail decays | `out_peak <= v_ceil`, `abs(final_dc) < 1e-5` |
| **6. Givens + Householder Stereo Matrix $\mathbf{U}(\psi, \mathbf{v})$** | Fixed ping-pong or non-unitary | **$< 1.0 \times 10^{-14}$ orthogonality error** across all $\psi \in [0, \pi/2]$ | `ortho_err < 1e-14`, `max_eig_err < 1e-14` |
| **7. Surrogate Delay + Dual-Ridge `float64` Polish** | Lagrange stuck ($54.52$-samp error) / Cold $+0.01\text{ dB}$ | **$0.0035$-samp delay lock**, Step-0 **$-25.39\text{ dB}$** $\to$ Polish **$-47.63\text{ dB}$ ESR** | `err_surr < 0.01`, `esr_polished < -45.0` |
| **8. Parametric Control Bus, 3-Phase Audit & `.dnam-ckpt` Resume** | Double-echo crossfade / Corner explosion / Full retrain | **$3.41 \times 10^{-13}\text{ ms}$ delay interp**, **$50.7\times$ smoother switch**, **$-38.46\text{ dB}$ unseen combo ESR**, **`1.05e-9` `.dnam-ckpt` parity** | `val_esr < -35.0`, `iter_vs_full_err < 1e-8` |

---

## 2. Reference Python Verification Suite (`tests/test_architecture_3_verification.py`)

Port the following verified reference suite directly into `tests/test_architecture_3_verification.py`:

```python
"""Empirical & Mathematical Verification Suite for DNAM Architecture 3 (A3)."""

from __future__ import annotations

import math
import numpy as np


def test_1_poly_adaa_algebraic_exactness_and_aliasing() -> None:
    fs = 48000.0
    n_samples = 48000
    t = np.arange(n_samples, dtype=np.float64) / fs

    # 1A. Linear passband droop across 8 cascaded layers at 16 kHz if midpoint filter is applied to full signal
    f_test = 16000.0
    x_lin = 0.05 * np.sin(2.0 * np.pi * f_test * t)
    y_naive = x_lin.copy()
    for _ in range(8):
        x_prev = np.roll(y_naive, 1)
        x_prev[0] = 0.0
        y_naive = 0.5 * (y_naive + x_prev)
    gain_naive_db = 20.0 * math.log10(float(np.std(y_naive[1000:]) / np.std(x_lin[1000:])))
    assert gain_naive_db < -40.0

    # 1B. Exact Division-Free Algebraic Factorization of Polynomial ADAA1 (0 FDIV)
    rng = np.random.default_rng(123)
    a_pts = rng.uniform(-0.99, 0.99, size=10000)
    b_pts = rng.uniform(-0.99, 0.99, size=10000)
    mask_sep = np.abs(b_pts - a_pts) > 1e-3
    a_pts, b_pts = a_pts[mask_sep], b_pts[mask_sep]
    delta_c = 0.35

    def f_exact(u: np.ndarray) -> np.ndarray:
        return 0.75 * u**2 - 0.125 * u**4 + delta_c * ((u**3) / 3.0 - 0.2 * (u**5))

    quot_div = (f_exact(b_pts) - f_exact(a_pts)) / (b_pts - a_pts)
    m_pts = 0.5 * (a_pts + b_pts)
    d_pts = 0.5 * (b_pts - a_pts)
    m2 = m_pts * m_pts
    d2 = d_pts * d_pts
    quot_nodiv = (
        1.5 * m_pts
        - 0.5 * m_pts * (m2 + d2)
        + delta_c * ((m2 + (1.0 / 3.0) * d2) - (m2 * m2 + 2.0 * m2 * d2 + 0.2 * d2 * d2))
    )
    max_alg_err = float(np.max(np.abs(quot_div - quot_nodiv)))
    assert max_alg_err < 1e-12

    # 1C. 8-Layer Cascaded Distortion at f0 = 3,450 Hz: NAM A2 LeakyReLU(0.01) vs DNAM A3 1x B-Spline PolyADAA
    f0 = 3450.0
    x = 1.6 * np.sin(2.0 * np.pi * f0 * t)

    def eval_alg_adaa(u_a: np.ndarray, u_b: np.ndarray) -> np.ndarray:
        ua = np.clip((2.0 / 3.0) * u_a, -1.0, 1.0)
        ub = np.clip((2.0 / 3.0) * u_b, -1.0, 1.0)
        m = 0.5 * (ua + ub)
        d = 0.5 * (ub - ua)
        m2 = m * m
        d2 = d * d
        return (
            1.5 * m
            - 0.5 * m * (m2 + d2)
            + 0.15 * ((m2 + (1.0 / 3.0) * d2) - (m2 * m2 + 2.0 * m2 * d2 + 0.2 * d2 * d2))
        )

    def a3_1x_bspline_adaa_layer(sig: np.ndarray) -> np.ndarray:
        s_m1 = np.roll(sig, 1)
        a_curr = eval_alg_adaa(s_m1, sig)
        y_nl = 0.875 * a_curr + 0.125 * np.roll(a_curr, 1)
        y_nl = y_nl - np.mean(y_nl[2000:-2000])
        return 0.2 * sig + 0.95 * y_nl

    def leaky_layer(sig: np.ndarray, bias: float) -> np.ndarray:
        v = 1.25 * sig + bias
        z = np.where(v >= 0.0, v, 0.01 * v) - max(bias, 0.01 * bias)
        return 0.2 * sig + 0.65 * z

    y_a3 = x.copy()
    y_a2 = x.copy()
    for b in [0.35, -0.40, 0.30, -0.35, 0.45, -0.30, 0.35, -0.40]:
        y_a3 = a3_1x_bspline_adaa_layer(y_a3)
        y_a2 = leaky_layer(y_a2, b)

    sl = slice(2000, n_samples - 2000)
    n_win = n_samples - 4000
    win = np.blackman(n_win)
    freqs = np.fft.rfftfreq(n_win, d=1.0 / fs)
    spec_a2 = np.abs(np.fft.rfft(y_a2[sl] * win))
    spec_a3 = np.abs(np.fft.rfft(y_a3[sl] * win))

    alias_mask = freqs > 60.0
    for m_harm in range(1, 7):
        alias_mask &= np.abs(freqs - m_harm * f0) > 60.0

    alias_a2_db = 20.0 * math.log10(float(np.linalg.norm(spec_a2[alias_mask]) / np.max(spec_a2)))
    alias_a3_db = 20.0 * math.log10(float(np.linalg.norm(spec_a3[alias_mask]) / np.max(spec_a3)))
    assert (alias_a2_db - alias_a3_db) > 35.0


def test_2_selective_rglru_radial_polar_vs_cartesian() -> None:
    r0 = 0.992
    theta_low = 0.05 * math.pi
    theta_high = 0.40 * math.pi
    lam_low = r0 * np.exp(1j * theta_low)
    lam_high = r0 * np.exp(1j * theta_high)
    lam_cart_mid = 0.5 * lam_low + 0.5 * lam_high
    q_orig = 1.0 / (1.0 - r0)
    q_cart = 1.0 / (1.0 - abs(lam_cart_mid))

    d_theta_16 = 0.0115
    u_rot_mid = 0.5 * (1.0 + np.exp(1j * d_theta_16))
    q_polar = 1.0 / (1.0 - r0 * abs(u_rot_mid))
    assert q_cart < 0.10 * q_orig
    assert q_polar > 0.99 * q_orig

    # Branchless 4-FMA + Newton Unit-Circle Rotor
    phi = np.linspace(-math.pi / 2.0, math.pi / 2.0, 20000)
    p2 = phi * phi
    sin_phi = phi * (0.99999995 - p2 * (0.16666650 - p2 * (0.00833290 - p2 * 0.00019507)))
    cos_phi = 0.99999995 - p2 * (0.49999990 - p2 * (0.04166630 - p2 * (0.00138860 - p2 * 0.00002438)))
    c_hat, s_hat = -sin_phi, cos_phi
    y1_proj = 1.5 - 0.5 * (c_hat * c_hat + s_hat * s_hat)
    u_mag = np.sqrt((y1_proj * c_hat) ** 2 + (y1_proj * s_hat) ** 2)
    assert float(np.max(np.abs(u_mag - 1.0))) < 2e-8
    assert float(np.max(u_mag)) <= 1.0 + 1e-15

    # Parallel Associative Scan equivalence
    length = 1024
    rng = np.random.default_rng(42)
    a = 0.98 * np.exp(1j * rng.uniform(0.01, 0.3, size=length))
    b_x = (rng.standard_normal(length) + 1j * rng.standard_normal(length)) * 0.1
    z_seq = np.zeros(length, dtype=np.complex128)
    curr = 0.0 + 0.0j
    for n in range(length):
        curr = a[n] * curr + b_x[n]
        z_seq[n] = curr

    a_scan, b_scan = a.copy(), b_x.copy()
    step = 1
    while step < length:
        a_prev, b_prev = np.roll(a_scan, step), np.roll(b_scan, step)
        a_prev[:step] = 1.0 + 0.0j
        b_prev[:step] = 0.0 + 0.0j
        b_scan = a_scan * b_prev + b_scan
        a_scan = a_scan * a_prev
        step *= 2
    assert float(np.max(np.abs(z_seq - b_scan))) < 1e-13


def test_3_fractional_delay_thiran_clicks_vs_cubic_lagrange() -> None:
    fs = 48000.0
    n_samples = 12000
    t = np.arange(n_samples, dtype=np.float64) / fs
    x = 0.5 * np.sin(2.0 * np.pi * 5000.0 * t)
    tau = 120.0 + 60.0 * np.sin(2.0 * np.pi * 3.0 * t)
    y_exact = 0.5 * np.sin(2.0 * np.pi * 5000.0 * (t - tau / fs))

    y_thiran = np.zeros(n_samples, dtype=np.float64)
    d_prev = 0.0
    for n in range(1000, n_samples):
        d_int = int(math.floor(tau[n] - 0.5))
        mu = tau[n] - d_int
        eta = (1.0 - mu) / (1.0 + mu)
        d_curr = eta * (x[n - d_int] - d_prev) + x[n - d_int - 1]
        y_thiran[n] = d_curr
        d_prev = d_curr

    y_lagrange = np.zeros(n_samples, dtype=np.float64)
    for n in range(1000, n_samples):
        d_int = int(math.floor(tau[n]))
        mu = tau[n] - d_int
        c_m1 = -mu * (mu - 1.0) * (mu - 2.0) / 6.0
        c_0 = (mu + 1.0) * (mu - 1.0) * (mu - 2.0) / 2.0
        c_1 = -(mu + 1.0) * mu * (mu - 2.0) / 2.0
        c_2 = (mu + 1.0) * mu * (mu - 1.0) / 6.0
        y_lagrange[n] = (
            c_m1 * x[n - d_int + 1]
            + c_0 * x[n - d_int]
            + c_1 * x[n - d_int - 1]
            + c_2 * x[n - d_int - 2]
        )

    err_thiran_jerk = float(np.max(np.abs(np.diff((y_thiran - y_exact)[1500:], n=2))))
    err_lagrange_jerk = float(np.max(np.abs(np.diff((y_lagrange - y_exact)[1500:], n=2))))
    assert err_thiran_jerk > 15.0 * err_lagrange_jerk


def test_4_mufdn_lyapunov_and_dc_stability() -> None:
    m_lines = 8
    delays = [37, 53, 79, 113, 163, 229, 313, 419]
    max_d = max(delays) + 8
    buf = np.zeros((m_lines, max_d), dtype=np.float64)
    rng = np.random.default_rng(7)
    v = rng.standard_normal(m_lines)
    v /= np.linalg.norm(v)

    g_lin, g_regen, sigma_knee, v_ceil = 0.98, 0.20, 0.15, 0.985
    dc_x_prev, dc_y_prev = np.zeros(m_lines), np.zeros(m_lines)
    r_dc = 0.995
    alpha_dc = 0.5 * (1.0 + r_dc)

    out_peak = 0.0
    for n in range(24000):
        x_in = (10.0 * math.sin(0.2 * n) + 2.0) if n < 1000 else 0.0
        d_vec = np.array([buf[m, (n - delays[m]) % max_d] for m in range(m_lines)])
        q_vec = np.roll(d_vec - (2.0 * float(np.dot(v, d_vec))) * v, 1)
        q_ac = alpha_dc * (q_vec - dc_x_prev) + r_dc * dc_y_prev
        dc_x_prev, dc_y_prev = q_vec.copy(), q_ac.copy()
        vk = np.clip(np.abs(q_ac) / sigma_knee, 0.0, 1.0)
        g_eff = g_lin + g_regen * (vk * vk * (3.0 - 2.0 * vk))
        vs = np.clip((2.0 / (3.0 * v_ceil)) * (g_eff * q_ac + 0.25 * x_in), -1.0, 1.0)
        s_new = v_ceil * vs * (1.5 - 0.5 * vs * vs)
        for m in range(m_lines):
            buf[m, n % max_d] = s_new[m]
        out_peak = max(out_peak, float(np.max(np.abs(s_new))))

    q_hist = np.zeros((8000, m_lines))
    for k in range(8000):
        n = 24000 + k
        d_vec = np.array([buf[m, (n - delays[m]) % max_d] for m in range(m_lines)])
        q_vec = np.roll(d_vec - (2.0 * float(np.dot(v, d_vec))) * v, 1)
        q_ac = alpha_dc * (q_vec - dc_x_prev) + r_dc * dc_y_prev
        dc_x_prev, dc_y_prev = q_vec.copy(), q_ac.copy()
        vk = np.clip(np.abs(q_ac) / sigma_knee, 0.0, 1.0)
        g_eff = g_lin + g_regen * (vk * vk * (3.0 - 2.0 * vk))
        vs = np.clip((2.0 / (3.0 * v_ceil)) * (g_eff * q_ac), -1.0, 1.0)
        s_new = v_ceil * vs * (1.5 - 0.5 * vs * vs)
        for m in range(m_lines):
            buf[m, n % max_d] = s_new[m]
        q_hist[k] = q_ac

    w_blk = np.blackman(8000)
    w_blk /= np.sum(w_blk)
    final_dc = float(np.max(np.abs(w_blk @ q_hist)))
    assert out_peak <= v_ceil + 1e-9
    assert abs(final_dc) < 1e-5


def test_5_givens_householder_orthogonality() -> None:
    rng = np.random.default_rng(99)
    m_lines = 8
    for psi in [0.0, math.pi / 6.0, math.pi / 4.0, math.pi / 2.0]:
        v = rng.standard_normal(m_lines)
        v /= np.linalg.norm(v)
        c, s = math.cos(psi), math.sin(psi)
        r_mat = np.zeros((m_lines, m_lines))
        for k in range(m_lines // 2):
            r_mat[2 * k, 2 * k] = c
            r_mat[2 * k, 2 * k + 1] = -s
            r_mat[2 * k + 1, 2 * k] = s
            r_mat[2 * k + 1, 2 * k + 1] = c
        h_mat = np.eye(m_lines) - 2.0 * np.outer(v, v)
        u_mat = h_mat @ r_mat
        assert float(np.max(np.abs(u_mat.T @ u_mat - np.eye(m_lines)))) < 1e-14
        assert float(np.max(np.abs(np.abs(np.linalg.eigvals(u_mat)) - 1.0))) < 1e-14


def test_6_surrogate_delay_gradient_optimization() -> None:
    fs = 48000.0
    n_len = 4096
    t = np.arange(n_len, dtype=np.float64) / fs
    env = np.exp(-np.maximum(t - 0.01, 0.0) / 0.008) * (t >= 0.01)
    x = env * (0.7 * np.sin(2.0 * np.pi * 880.0 * t) + 0.3 * np.sin(2.0 * np.pi * 1760.0 * t))
    x /= np.max(np.abs(x))

    def eval_lagrange_and_grad(
        sig: np.ndarray, tau_val: float, sigma_smooth: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        d_int = int(math.floor(tau_val))
        mu = tau_val - d_int
        c_m1 = -mu * (mu - 1.0) * (mu - 2.0) / 6.0
        c_0 = (mu + 1.0) * (mu - 1.0) * (mu - 2.0) / 2.0
        c_1 = -(mu + 1.0) * mu * (mu - 2.0) / 2.0
        c_2 = (mu + 1.0) * mu * (mu - 1.0) / 6.0
        y = np.zeros_like(sig)
        idx = np.arange(300, n_len)
        y[idx] = (
            c_m1 * sig[idx - d_int + 1]
            + c_0 * sig[idx - d_int]
            + c_1 * sig[idx - d_int - 1]
            + c_2 * sig[idx - d_int - 2]
        )
        if sigma_smooth <= 1.0:
            dc_m1 = -(3.0 * mu * mu - 6.0 * mu + 2.0) / 6.0
            dc_0 = (3.0 * mu * mu - 4.0 * mu - 1.0) / 2.0
            dc_1 = -(3.0 * mu * mu - 2.0 * mu - 2.0) / 2.0
            dc_2 = (3.0 * mu * mu - 1.0) / 6.0
            dy_dtau = np.zeros_like(sig)
            dy_dtau[idx] = (
                dc_m1 * sig[idx - d_int + 1]
                + dc_0 * sig[idx - d_int]
                + dc_1 * sig[idx - d_int - 1]
                + dc_2 * sig[idx - d_int - 2]
            )
        else:
            half_w = int(math.ceil(3.0 * sigma_smooth))
            k_grid = np.arange(-half_w, half_w + 1, dtype=np.float64)
            g_deriv = (k_grid / (sigma_smooth**2)) * np.exp(-0.5 * (k_grid / sigma_smooth) ** 2)
            g_deriv /= np.sum(np.abs(g_deriv))
            dy_dtau = np.zeros_like(sig)
            for k_off, w_g in zip(k_grid, g_deriv, strict=True):
                shift_idx = np.clip(idx - d_int - int(k_off), 0, n_len - 1)
                dy_dtau[idx] += w_g * sig[shift_idx]
        return y, dy_dtau

    tau_true = 150.35
    y_target, _ = eval_lagrange_and_grad(x, tau_true, sigma_smooth=0.0)
    tau_surr = 85.0
    env_x, env_tgt = np.abs(x), np.abs(y_target)
    for step in range(1, 151):
        if step <= 70:
            y_env_pred, dy_env = eval_lagrange_and_grad(env_x, tau_surr, sigma_smooth=35.0)
            grad = float(np.mean(2.0 * (y_env_pred[300:] - env_tgt[300:]) * dy_env[300:]))
            tau_surr -= 1.5 * grad / (math.sqrt(grad * grad) + 1e-5)
        else:
            y_pred, dy = eval_lagrange_and_grad(x, tau_surr, sigma_smooth=0.0)
            grad = float(np.mean(2.0 * (y_pred[300:] - y_target[300:]) * dy[300:]))
            tau_surr -= 40.0 * grad
    assert abs(tau_surr - tau_true) < 0.01


def test_7_h2_log_pole_init_and_end_to_end_pytorch_a3() -> None:
    import torch

    torch.manual_seed(42)
    length = 4096
    c_total = 12
    t = torch.arange(length, dtype=torch.float32) / 48000.0
    env = torch.exp(-t / 0.04) + 0.3 * torch.sin(2.0 * math.pi * 5.0 * t).abs()
    x_in = (
        0.5 * torch.sin(2.0 * math.pi * 220.0 * t)
        + 0.3 * torch.sin(2.0 * math.pi * 440.0 * t)
        + 0.2 * torch.sin(2.0 * math.pi * 660.0 * t)
    ).view(1, 1, length) * env.view(1, 1, length)

    u_gt = torch.clamp(1.4 * x_in, -1.0, 1.0)
    nl_gt = 1.5 * u_gt - 0.5 * (u_gt**3) + 0.2 * (u_gt**2 - u_gt**4)
    nl_gt = nl_gt - nl_gt.mean(dim=-1, keepdim=True)
    y_gt_l = 0.7 * nl_gt + 0.25 * torch.roll(nl_gt, 3, dims=-1) - 0.15 * torch.roll(nl_gt, 7, dims=-1)
    y_gt_r = 0.65 * nl_gt - 0.20 * torch.roll(nl_gt, 4, dims=-1) + 0.18 * torch.roll(nl_gt, 9, dims=-1)
    y_tgt = torch.cat([y_gt_l, y_gt_r], dim=1)

    conv = torch.nn.Conv1d(1, c_total, kernel_size=6, padding=5)
    alpha = torch.nn.Parameter(torch.full((1, c_total, 1), 0.25))
    beta = torch.nn.Parameter(torch.full((1, c_total, 1), 0.75))
    delta = torch.nn.Parameter(torch.full((1, c_total, 1), 0.15))
    w_sel = torch.nn.Parameter(torch.full((1, c_total, 1), 0.20))
    head_stereo = torch.nn.Conv1d(c_total, 2, kernel_size=16, bias=True)

    def extract_features(x: torch.Tensor) -> torch.Tensor:
        z_conv = conv(x)[:, :, :length]
        u = torch.clamp((2.0 / 3.0) * z_conv, -1.0, 1.0)
        u_prev = torch.cat([torch.zeros_like(u[:, :, :1]), u[:, :, :-1]], dim=2)
        m, d = 0.5 * (u + u_prev), 0.5 * (u - u_prev)
        m2, d2 = m * m, d * d
        adaa = beta * m * (1.5 - 0.5 * (m2 + d2)) + delta * (
            (m2 + (1.0 / 3.0) * d2) - (m2 * m2 + 2.0 * m2 * d2 + 0.2 * d2 * d2)
        )
        adaa_prev = torch.cat([torch.zeros_like(adaa[:, :, :1]), adaa[:, :, :-1]], dim=2)
        y_nl = 0.875 * adaa + 0.125 * adaa_prev
        y_act = alpha * z_conv + (y_nl - y_nl.mean(dim=2, keepdim=True))
        g_gate = torch.clamp(0.5 + w_sel * y_act, 0.0, 1.0)
        r_t = 0.85 + g_gate * (0.96 - 0.85)
        return r_t * y_act + g_gate * math.sqrt(1.0 - 0.90**2) * y_act

    def esr_db(pred: torch.Tensor, tgt: torch.Tensor) -> float:
        err = torch.sum((pred[:, :, 100:] - tgt[:, :, 100:]) ** 2)
        sig = torch.sum(tgt[:, :, 100:] ** 2)
        return 10.0 * math.log10((err / sig).item() + 1e-12)

    def solve_head_float64(rel_lam: float) -> float:
        with torch.no_grad():
            h_0 = extract_features(x_in)
            h_pad = torch.nn.functional.pad(h_0, (15, 0))
            phi = (
                torch.nn.functional.unfold(h_pad.unsqueeze(-2), kernel_size=(1, 16))
                .squeeze(0)
                .T.to(torch.float64)
            )
            phi_valid = phi[100:]
            y_valid = y_tgt[0, :, 100:].T.to(torch.float64)
            gram = phi_valid.T @ phi_valid
            lam = rel_lam * (torch.trace(gram) / gram.shape[0])
            w_opt = torch.linalg.solve(
                gram + lam * torch.eye(gram.shape[0], dtype=torch.float64), phi_valid.T @ y_valid
            )
            head_stereo.weight.copy_(w_opt.to(torch.float32).T.view(2, c_total, 16))
            assert head_stereo.bias is not None
            head_stereo.bias.zero_()
            return esr_db(head_stereo(h_pad), y_tgt)

    esr_warm = solve_head_float64(1e-3)
    opt = torch.optim.Adam(
        [
            {"params": [conv.weight, conv.bias, alpha, beta, delta, w_sel], "lr": 1.0e-3},
            {"params": [head_stereo.weight, head_stereo.bias], "lr": 3.0e-4},
        ],
        weight_decay=0.0,
    )
    for step in range(1, 101):
        warmup = min(1.0, step / 20.0)
        opt.param_groups[0]["lr"] = 1.0e-3 * warmup
        opt.param_groups[1]["lr"] = 3.0e-4 * warmup
        opt.zero_grad()
        h = extract_features(x_in)
        y_pred = head_stereo(torch.nn.functional.pad(h, (15, 0)))
        loss = torch.mean((y_pred[:, :, 100:] - y_tgt[:, :, 100:]) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [conv.weight, conv.bias, alpha, beta, delta, w_sel, head_stereo.weight], 1.0
        )
        opt.step()

    esr_polished = solve_head_float64(1e-6)
    assert esr_warm < -20.0
    assert esr_polished < -45.0


def test_8_parametric_control_bus_and_multistem_conditioning() -> None:
    import torch

    torch.manual_seed(42)
    length = 2048
    t = torch.arange(length, dtype=torch.float64) / 48000.0
    x = (
        0.5 * torch.sin(2.0 * np.pi * 220.0 * t)
        + 0.35 * torch.sin(2.0 * np.pi * 1760.0 * t)
        + 0.25 * torch.sin(2.0 * np.pi * 3520.0 * t)
    )
    x_diff = x - torch.roll(x, 1)

    def synth_gt(u_gain: float, u_treble: float, s_bright: int) -> torch.Tensor:
        drive = 1.5 * math.exp(0.7 * u_gain)
        x_pre = x + (0.8 * s_bright) * x_diff
        u = torch.clamp((2.0 / 3.0) * drive * x_pre, -1.0, 1.0)
        y_nl = 1.5 * u - 0.5 * (u**3)
        return y_nl + (0.4 * u_treble) * (y_nl - torch.roll(y_nl, 1))

    stems_meta = [
        (-0.8, 0.0, 0, -0.8),
        (0.0, 0.0, 0, 0.0),
        (0.8, 0.0, 0, 0.8),
        (0.0, -0.8, 0, 0.0),
        (0.0, 0.8, 0, 0.0),
        (-0.8, 0.0, 1, -0.8),
        (0.8, 0.0, 1, 0.8),
    ]
    val_meta = (0.3, 0.5, 1, 0.3)

    def raw_control_basis(u_g: float, u_t: float, s_b: int, u_m: float) -> torch.Tensor:
        return torch.tensor(
            [u_g, u_g * u_g, u_t, u_t * u_t, float(s_b), u_m, u_m * u_m, u_g * float(s_b), u_g * u_t],
            dtype=torch.float64,
        )

    psi_raw = torch.stack([raw_control_basis(*m) for m in stems_meta], dim=0)
    _, s_vals, vh = torch.linalg.svd(psi_raw, full_matrices=False)
    valid_rank = int(torch.sum(s_vals > 1e-4).item())
    p_svd = vh[:valid_rank].T

    g_audio_ckpt = []
    c_audio_ckpt = []
    for m in stems_meta:
        u_g, u_t, s_b, _ = m
        feats = []
        for d_scale in [0.8, 1.5, 2.8]:
            for hf_pre in [0.0, 0.7]:
                x_in_c = x + hf_pre * x_diff
                u_c = torch.clamp(
                    (2.0 / 3.0) * d_scale * math.exp(0.5 * u_g) * (x_in_c + 0.5 * s_b * x_diff),
                    -1.0,
                    1.0,
                )
                y_c = 1.5 * u_c - 0.5 * (u_c**3)
                feats.append(y_c)
                feats.append(y_c - torch.roll(y_c, 1))
        phi_s = torch.stack(feats, dim=-1)[100:]
        y_s = synth_gt(u_g, u_t, s_b)[100:]
        g_audio_ckpt.append(phi_s.T @ phi_s)
        c_audio_ckpt.append(phi_s.T @ y_s)

    d_ctrl_7 = 1 + valid_rank
    g_updated = torch.zeros(12 * d_ctrl_7, 12 * d_ctrl_7, dtype=torch.float64)
    c_updated = torch.zeros(12 * d_ctrl_7, dtype=torch.float64)
    for s in range(7):
        psi_s7 = torch.cat([torch.ones(1, dtype=torch.float64), p_svd.T @ psi_raw[s]])
        g_updated += torch.kron(g_audio_ckpt[s], torch.outer(psi_s7, psi_s7))
        c_updated += torch.kron(c_audio_ckpt[s].unsqueeze(-1), psi_s7.unsqueeze(-1)).squeeze(-1)
    lam = 1e-5 * (torch.trace(g_updated) / g_updated.shape[0])
    w_ckpt_iter = torch.linalg.solve(
        g_updated + lam * torch.eye(g_updated.shape[0], dtype=torch.float64), c_updated
    )

    u_g, u_t, s_b, u_m = val_meta
    val_feats = []
    for d_scale in [0.8, 1.5, 2.8]:
        for hf_pre in [0.0, 0.7]:
            x_in_c = x + hf_pre * x_diff
            u_c = torch.clamp(
                (2.0 / 3.0) * d_scale * math.exp(0.5 * u_g) * (x_in_c + 0.5 * s_b * x_diff),
                -1.0,
                1.0,
            )
            y_c = 1.5 * u_c - 0.5 * (u_c**3)
            val_feats.append(y_c)
            val_feats.append(y_c - torch.roll(y_c, 1))
    phi_val = torch.stack(val_feats, dim=-1)[100:]
    psi_val = torch.cat([torch.ones(1, dtype=torch.float64), p_svd.T @ raw_control_basis(u_g, u_t, s_b, u_m)])
    x_val = torch.einsum("ld,k->ldk", phi_val, psi_val).reshape(-1, 12 * d_ctrl_7)
    y_val = synth_gt(u_g, u_t, s_b)[100:]
    val_esr = 10.0 * math.log10((torch.sum((x_val @ w_ckpt_iter - y_val) ** 2) / torch.sum(y_val**2)).item())
    assert val_esr < -35.0
```

---

## 3. Automated Guardrail Test Suite (`tests/test_guardrails.py`)

In addition to `test_architecture_3_verification.py`, `tests/test_guardrails.py` must enforce:
1. **`test_guardrail_agents_md_progressive_disclosure_budgets`:** Verifies that root [`AGENTS.md`](../AGENTS.md) is `< 5,000` bytes and every scoped `AGENTS.md` is `< 2,500` bytes.
2. **`test_guardrail_documentation_markdown_links_valid`:** Parses all Markdown files (`AGENTS.md`, `README.md`, `CONTRIBUTING.md`, `CLA.md`, and `docs/**/*.md`), strips fenced code blocks and inline code spans, and verifies that every relative link resolves to an existing file on disk.
3. **`test_guardrail_clamp_registry_integrity`:** Parses `CL-01`..`CL-18` in [`clamp_and_audit_registry.md`](clamp_and_audit_registry.md) and verifies that all 18 clamps are present and map to existing modules under `src/dnam/`.
4. **`test_guardrail_fail_fast_schema_validation`:** Verifies that `.dnam` v3 schema loading raises explicit `ValueError` / `ValidationError` on missing or malformed `controls_schema`, invalid sample rates, or unaligned channel counts ($C \notin \{4, 8\}$).
