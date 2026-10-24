"""Core 1A: Division-Free Algebraic `1x B-Spline PolyADAA` waveshaper (`CL-01`, `CL-02`),
3-state analytical DC resting equilibrium, Two-Level Chunked $5\\text{ Hz}$ AC coupling scan,
and `dnam_engine.dsp.PolyAdaa` reference bridge."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from dnam_engine.dsp import PolyAdaa
from dnam_engine.schema import PolyAdaaParams, PolyAdaaState


def clamp_pre_activation_torch(x: Tensor) -> Tensor:
    """Enforces `CL-01`: `u_n = clamp((2/3) * x_c[n], -1.0, +1.0)`."""
    return torch.clamp((2.0 / 3.0) * x, -1.0, 1.0)


def poly_adaa_waveshaper_torch(u: Tensor, beta: Tensor | float, delta: Tensor | float) -> Tensor:
    """Evaluates instantaneous `PolyADAA` waveshaper `g_nl(u) = beta*(1.5*u - 0.5*u^3) + delta*(u^2
    - u^4)`."""
    u2 = u * u
    return beta * u * (1.5 - 0.5 * u2) + delta * (u2 - u2 * u2)


def poly_adaa_algebraic_torch(
    u_prev: Tensor,
    u_curr: Tensor,
    beta: Tensor | float,
    delta: Tensor | float,
) -> Tensor:
    """Division-free algebraic `PolyADAA` segment average `A_c[n]` (`CL-01`, `0 FDIV`, `0 FSQRT`).

    Uses the midpoint/half-difference factorization `m_n = 0.5*(u_n + u_{n-1})`,
    `d_n = 0.5*(u_n - u_{n-1})` so `(F(u_n) - F(u_{n-1})) / (u_n - u_{n-1})` has zero division.
    """
    m = 0.5 * (u_curr + u_prev)
    d = 0.5 * (u_curr - u_prev)
    m2 = m * m
    d2 = d * d
    odd = beta * m * (1.5 - 0.5 * (m2 + d2))
    even = delta * ((m2 + (1.0 / 3.0) * d2) - (m2 * m2 + 2.0 * m2 * d2 + 0.2 * d2 * d2))
    return odd + even


def resting_dc_equilibrium(
    b_conv: Tensor,
    beta: Tensor,
    delta: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Computes the exact 3-state DC resting equilibrium `(u_rest, a_rest, m_dc_rest)` (`CL-01`,
    `CL-02`).

    Guarantees `0.000000` Sample-0 and Sample-1 transient on silence even when `b_conv != 0`.
    """
    u_rest = clamp_pre_activation_torch(b_conv)
    g_rest = poly_adaa_waveshaper_torch(u_rest, beta, delta)
    return u_rest, g_rest, g_rest


def _scan_real_1pole_ac_filter(
    y_nl: Tensor,
    r_ac: float,
    m_dc_init: Tensor,
    chunk_size: int = 256,
) -> tuple[Tensor, Tensor]:
    """Two-Level Chunked Associative Scan for `m_dc[n] = r_ac * m_dc[n-1] + (1 - r_ac) * y_nl[n]`
    (`CL-02`).

    Input `y_nl` has shape `(..., L)`, `m_dc_init` has shape `(..., 1)`.
    Returns `(m_dc_seq, m_dc_final)` matching sequential recurrence to `< 1e-13` in `float64`.
    """
    orig_shape = y_nl.shape
    seq_len = orig_shape[-1]
    batch_flat = y_nl.reshape(-1, seq_len)
    init_flat = m_dc_init.reshape(-1, 1)
    # Subtract equilibrium init_flat before scanning so constant/silent inputs have 0.0 drift
    b_scaled = (1.0 - r_ac) * (batch_flat - init_flat)

    # Pad to a multiple of chunk_size for static-shape chunked scan
    c_len = min(chunk_size, max(1, seq_len))
    rem = seq_len % c_len
    pad_len = (c_len - rem) % c_len
    b_padded = torch.nn.functional.pad(b_scaled, (0, pad_len)) if pad_len > 0 else b_scaled

    num_chunks = b_padded.shape[-1] // c_len
    chunks_b = b_padded.reshape(-1, num_chunks, c_len)
    chunks_a = torch.full_like(chunks_b, r_ac)

    # Level 1: Intra-chunk associative scan (log2(c_len) steps)
    step = 1
    while step < c_len:
        a_hi = chunks_a[:, :, step:]
        b_hi = chunks_b[:, :, step:]
        a_lo = chunks_a[:, :, :-step]
        b_lo = chunks_b[:, :, :-step]
        new_a = a_hi * a_lo
        new_b = a_hi * b_lo + b_hi
        chunks_a = torch.cat([chunks_a[:, :, :step], new_a], dim=-1)
        chunks_b = torch.cat([chunks_b[:, :, :step], new_b], dim=-1)
        step *= 2

    # Level 2: Inter-chunk boundary scan over chunk endpoints (relative to 0.0 deviation state)
    if num_chunks > 1:
        end_a = chunks_a[:, :, -1]
        end_b = chunks_b[:, :, -1]
        b_step = 1
        while b_step < num_chunks:
            ea_hi = end_a[:, b_step:]
            eb_hi = end_b[:, b_step:]
            ea_lo = end_a[:, :-b_step]
            eb_lo = end_b[:, :-b_step]
            end_a = torch.cat([end_a[:, :b_step], ea_hi * ea_lo], dim=-1)
            end_b = torch.cat([end_b[:, :b_step], ea_hi * eb_lo + eb_hi], dim=-1)
            b_step *= 2
        zero_col = torch.zeros_like(init_flat)
        chunk_inits = torch.cat([zero_col, end_b[:, :-1]], dim=-1).unsqueeze(-1)
    else:
        chunk_inits = torch.zeros_like(init_flat).unsqueeze(-1)

    out_padded = (chunks_a * chunk_inits + chunks_b).reshape(-1, num_chunks * c_len)
    out_seq = out_padded[:, :seq_len] + init_flat
    m_final = out_seq[:, -1:]
    return out_seq.reshape(orig_shape), m_final.reshape(m_dc_init.shape)


class DifferentiablePolyAdaa(nn.Module):
    """Differentiable Core 1A `1x B-Spline PolyADAA` layer (`CL-01`, `CL-02`).

    Implements:
    - Un-delayed linear path `alpha_c * x_c[n]` (`0.00 dB` passband flatness),
    - Pre-activation clamp `u_c[n] = clamp((2/3)*(x_c[n] + b_conv_c), -1.0, +1.0)` (`CL-01`),
    - Division-free algebraic antiderivative average `A_c[n]` (`CL-01`),
    - `1x` passive B-spline anti-aliasing blend `0.875 * A_c[n] + 0.125 * A_c[n-1]` (`CL-02`),
    - `5 Hz` 1-pole AC coupling filter (`R_ac = 0.9993`, `CL-02`) via Two-Level Chunked Scan,
    - Exact 3-state DC resting equilibrium (`0.000000` Sample-0 transient).
    """

    def __init__(
        self,
        num_channels: int = 12,
        *,
        r_ac: float = 0.9993,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if num_channels < 1:
            raise ValueError(f"num_channels must be >= 1, got {num_channels}")
        if not (0.990 <= r_ac <= 0.9999):
            raise ValueError(f"r_ac ({r_ac}) must be in [0.990, 0.9999] per CL-02")

        self.num_channels = num_channels
        self.r_ac = float(r_ac)

        self.b_conv = nn.Parameter(torch.zeros(num_channels, dtype=dtype))
        self.alpha = nn.Parameter(torch.full((num_channels,), 0.85, dtype=dtype))
        self.beta = nn.Parameter(torch.full((num_channels,), 0.45, dtype=dtype))
        self.delta = nn.Parameter(torch.full((num_channels,), 0.10, dtype=dtype))
        self._last_rail_lockup_fraction: float = 0.0

    @property
    def last_rail_lockup_fraction(self) -> float:
        """Fraction of samples hitting hard `|u_c[n]| >= 0.999` saturation (`[CLAMP-RAIL]`)."""
        return self._last_rail_lockup_fraction

    def resting_state(self, batch_size: int = 1) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns `(u_prev, a_prev, m_dc)` initialized to the analytical DC equilibrium (`CL-01`).
        """
        u_r, a_r, m_r = resting_dc_equilibrium(self.b_conv, self.beta, self.delta)
        shape = (batch_size, self.num_channels, 1)
        return (
            u_r.view(1, self.num_channels, 1).expand(shape).clone(),
            a_r.view(1, self.num_channels, 1).expand(shape).clone(),
            m_r.view(1, self.num_channels, 1).expand(shape).clone(),
        )

    def forward(
        self,
        x: Tensor,
        state: tuple[Tensor, Tensor, Tensor] | None = None,
        *,
        delta_b_conv: Tensor | None = None,
        delta_beta: Tensor | None = None,
        delta_delta: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        """
        Processes `x` of shape `(B, C, L)` or `(C, L)` in at least `float32` (`CL-01`, `CL-02`).
        """
        squeeze_batch = x.ndim == 2
        inp = x.unsqueeze(0) if squeeze_batch else x
        if inp.ndim != 3 or inp.shape[1] != self.num_channels:
            raise ValueError(
                f"Expected input shape (B, {self.num_channels}, L), got {tuple(x.shape)}"
            )

        work_dtype = torch.float64 if inp.dtype == torch.float64 else torch.float32
        dev_type = inp.device.type if inp.device.type in ("cuda", "cpu", "mps") else "cpu"

        with torch.autocast(device_type=dev_type, enabled=False):
            x_f = inp.to(dtype=work_dtype)
            b_c = self.b_conv.to(dtype=work_dtype).view(1, self.num_channels, 1)
            alp = self.alpha.to(dtype=work_dtype).view(1, self.num_channels, 1)
            bet = self.beta.to(dtype=work_dtype).view(1, self.num_channels, 1)
            dlt = self.delta.to(dtype=work_dtype).view(1, self.num_channels, 1)

            if delta_b_conv is not None:
                b_c = b_c + delta_b_conv.to(dtype=work_dtype)
            if delta_beta is not None:
                bet = bet + delta_beta.to(dtype=work_dtype)
            if delta_delta is not None:
                dlt = dlt + delta_delta.to(dtype=work_dtype)

            if state is None:
                u_r = clamp_pre_activation_torch(b_c)
                g_r = poly_adaa_waveshaper_torch(u_r, bet, dlt)
                bsz = x_f.shape[0]
                u_init = u_r.expand(bsz, self.num_channels, 1)
                a_init = g_r.expand(bsz, self.num_channels, 1)
                m_init = g_r.expand(bsz, self.num_channels, 1)
            else:
                u_init = state[0].to(dtype=work_dtype)
                a_init = state[1].to(dtype=work_dtype)
                m_init = state[2].to(dtype=work_dtype)

            # CL-01: Pre-activation clamp u_n = clamp((2/3)*(x + b_conv), -1.0, +1.0)
            u_curr = clamp_pre_activation_torch(x_f + b_c)
            with torch.no_grad():
                self._last_rail_lockup_fraction = float(
                    (u_curr.abs() >= 0.999).to(torch.float32).mean().item()
                )

            u_prev = torch.cat([u_init, u_curr[:, :, :-1]], dim=-1)
            a_curr = poly_adaa_algebraic_torch(u_prev, u_curr, bet, dlt)

            # CL-02: 1x passive B-spline anti-aliasing blend (0.875 * A[n] + 0.125 * A[n-1])
            a_prev = torch.cat([a_init, a_curr[:, :, :-1]], dim=-1)
            y_nl = 0.875 * a_curr + 0.125 * a_prev

            # CL-02: 5 Hz AC coupling filter (R_ac = 0.9993)
            m_dc_seq, m_dc_final = _scan_real_1pole_ac_filter(y_nl, self.r_ac, m_init)
            y_out = alp * x_f + (y_nl - m_dc_seq)

            next_state = (
                u_curr[:, :, -1:].detach(),
                a_curr[:, :, -1:].detach(),
                m_dc_final.detach(),
            )

        out = y_out.to(dtype=inp.dtype)
        return (out.squeeze(0) if squeeze_batch else out), next_state

    def to_reference_dsp(self, channel: int = 0) -> PolyAdaa:
        """Exports `channel` into a validated `dnam_engine.dsp.PolyAdaa` reference processor."""
        if not (0 <= channel < self.num_channels):
            raise IndexError(f"channel {channel} out of bounds for {self.num_channels} channels")
        params = PolyAdaaParams(
            b_conv=float(self.b_conv[channel].detach().cpu().item()),
            alpha=float(self.alpha[channel].detach().cpu().item()),
            beta=float(self.beta[channel].detach().cpu().item()),
            delta=float(self.delta[channel].detach().cpu().item()),
            r_ac=self.r_ac,
        )
        return PolyAdaa(params)


__all__ = [
    "DifferentiablePolyAdaa",
    "PolyAdaaState",
    "clamp_pre_activation_torch",
    "poly_adaa_algebraic_torch",
    "poly_adaa_waveshaper_torch",
    "resting_dc_equilibrium",
]
