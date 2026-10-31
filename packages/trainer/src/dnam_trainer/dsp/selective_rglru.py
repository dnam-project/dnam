"""Core 1B: Radial-Polar Selective Complex RG-LRU (`CL-03`, `CL-04`, `CL-05`),
$3\\text{ kHz}$ branchless 4-FMA + Algebraic Newton unit-circle rotor,
$\\mathcal{H}_2$-normalized log-octave pole initialization, Two-Level Chunked Associative Scan
($L_{\\text{chunk}} = 256$), and `dnam_engine.dsp.SelectiveRgLru` reference bridge."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from dnam_engine.dsp import SelectiveRgLru
from dnam_engine.schema import SelectiveRgLruParams


def log_octave_pole_init(
    num_channels: int = 12,
    *,
    sample_rate_hz: float = 48_000.0,
    f_min_hz: float = 25.0,
    f_max_hz: float = 16_000.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Computes log-octave initial pole frequencies `f_{0,c}` and angles `theta_{0,c}` (`Test 7`).
    """
    if num_channels < 1:
        raise ValueError(f"num_channels must be >= 1, got {num_channels}")
    if num_channels == 1:
        freqs_list = [math.sqrt(f_min_hz * f_max_hz)]
    else:
        ratio = f_max_hz / f_min_hz
        freqs_list = [
            f_min_hz * (ratio ** (i / float(num_channels - 1))) for i in range(num_channels)
        ]
    thetas_list = [2.0 * math.pi * f / float(sample_rate_hz) for f in freqs_list]
    freqs = torch.tensor(freqs_list, dtype=dtype, device=device)
    thetas = torch.tensor(thetas_list, dtype=dtype, device=device)
    return freqs, thetas


def algebraic_unit_rotor_torch(theta_rad: Tensor) -> tuple[Tensor, Tensor]:
    """Branchless $3\\text{ kHz}$ 4-FMA + Algebraic Newton unit-circle rotor (`CL-03`).

    Evaluates the minimax polynomial on `phi = theta - pi/2` followed by one division-free
    Algebraic Newton step `y_1 = 1.5 - 0.5*(c_hat^2 + s_hat^2)`, guaranteeing
    `|u_rot,c| <= 1.000000000000` with `0 FSQRT`, `0 FDIV`, and `0` branches.
    """
    eps = 1.0e-4 if theta_rad.dtype == torch.float32 else 0.0
    theta = torch.clamp(theta_rad, eps, math.pi - eps)
    phi = theta - (0.5 * math.pi)
    p2 = phi * phi
    sin_phi = phi * (0.999_999_95 - p2 * (0.166_666_5 - p2 * (0.008_332_9 - p2 * 0.000_195_07)))
    cos_phi = 0.999_999_95 - p2 * (
        0.499_999_9 - p2 * (0.041_666_3 - p2 * (0.001_388_6 - p2 * 0.000_024_38))
    )
    c_hat = -sin_phi
    s_hat = cos_phi
    y1 = 1.5 - 0.5 * (c_hat * c_hat + s_hat * s_hat)
    return y1 * c_hat, y1 * s_hat


def h2_input_norm_gain(r_pole: Tensor) -> Tensor:
    """Enforces `CL-04`: `gamma_{norm,c} = sqrt(1 - r_c^2)` for unit impulse energy."""
    r_clamped = torch.clamp(r_pole, 0.25, 0.9995)
    return torch.sqrt(torch.clamp(1.0 - r_clamped * r_clamped, min=1.0e-7))


def parallel_associative_scan_complex(
    a_complex: Tensor,
    b_x_complex: Tensor,
    z0: Tensor | None = None,
    *,
    chunk_size: int = 256,
) -> tuple[Tensor, Tensor]:
    """Two-Level Chunked Associative Scan ($L_{\\text{chunk}} = 256$) for `z[n] = a[n]*z[n-1] +
    b_x[n]`.

    Executes intra-chunk scans in SRAM/L2 (`log2(256) = 8` steps) + inter-chunk boundary scans,
    matching sequential complex recurrence to `< 1e-13` in `complex128` (`< 1e-6` in `complex64`).
    """
    if a_complex.shape != b_x_complex.shape:
        raise ValueError(
            f"a_complex shape {tuple(a_complex.shape)} != "
            f"b_x_complex shape {tuple(b_x_complex.shape)}"
        )

    orig_shape = a_complex.shape
    seq_len = orig_shape[-1]
    a_flat = a_complex.reshape(-1, seq_len)
    b_flat = b_x_complex.reshape(-1, seq_len)

    if z0 is None:
        z_init = torch.zeros((a_flat.shape[0], 1), dtype=a_complex.dtype, device=a_complex.device)
    else:
        z_init = z0.reshape(-1, 1).to(dtype=a_complex.dtype, device=a_complex.device)

    c_len = min(chunk_size, max(1, seq_len))
    rem = seq_len % c_len
    pad_len = (c_len - rem) % c_len
    if pad_len > 0:
        a_pad = torch.cat(
            [
                a_flat,
                torch.ones(
                    (a_flat.shape[0], pad_len), dtype=a_complex.dtype, device=a_complex.device
                ),
            ],
            dim=-1,
        )
        b_pad = torch.cat(
            [
                b_flat,
                torch.zeros(
                    (b_flat.shape[0], pad_len), dtype=b_x_complex.dtype, device=b_x_complex.device
                ),
            ],
            dim=-1,
        )
    else:
        a_pad = a_flat
        b_pad = b_flat

    num_chunks = a_pad.shape[-1] // c_len
    chunks_a = a_pad.reshape(-1, num_chunks, c_len)
    chunks_b = b_pad.reshape(-1, num_chunks, c_len)

    # Level 1: Intra-chunk associative scan
    step = 1
    while step < c_len:
        a_hi = chunks_a[:, :, step:]
        b_hi = chunks_b[:, :, step:]
        a_lo = chunks_a[:, :, :-step]
        b_lo = chunks_b[:, :, :-step]
        chunks_a = torch.cat([chunks_a[:, :, :step], a_hi * a_lo], dim=-1)
        chunks_b = torch.cat([chunks_b[:, :, :step], a_hi * b_lo + b_hi], dim=-1)
        step *= 2

    # Level 2: Inter-chunk boundary scan
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
        chunk_ends = end_a * z_init + end_b
        chunk_inits = torch.cat([z_init, chunk_ends[:, :-1]], dim=-1).unsqueeze(-1)
    else:
        chunk_inits = z_init.unsqueeze(-1)

    z_padded = (chunks_a * chunk_inits + chunks_b).reshape(-1, num_chunks * c_len)
    z_seq = z_padded[:, :seq_len]
    z_final = z_seq[:, -1:]
    return z_seq.reshape(orig_shape), z_final.reshape((*orig_shape[:-1], 1))


class DifferentiableSelectiveRgLru(nn.Module):
    """Differentiable Core 1B Radial-Polar Selective Complex RG-LRU layer (`CL-03`..`CL-05`).

    Enforces:
    - **`CL-03`:** $3\\text{ kHz}$ 4-FMA + Algebraic Newton unit-circle rotor
      (`|u_rot,c| <= 1.000000000000`),
    - **`CL-04`:** Pole radii `r_low, r_high in [0.25, 0.9995]` and $\\mathcal{H}_2$ input scaling
      `gamma_norm = sqrt(1 - r_bar^2)`,
    - **`CL-05`:** $48\\text{ kHz}$ input-dependent radial selectivity gate `g_c[n] in [0.0, 1.0]`.
    """

    def __init__(
        self,
        num_channels: int = 12,
        *,
        sample_rate_hz: float = 48_000.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if num_channels < 1:
            raise ValueError(f"num_channels must be >= 1, got {num_channels}")

        self.num_channels = num_channels
        self.sample_rate_hz = float(sample_rate_hz)

        _, thetas = log_octave_pole_init(
            num_channels,
            sample_rate_hz=sample_rate_hz,
            dtype=dtype,
        )
        # Parameterize radius in unconstrained logit space mapped into [0.25, 0.9995] (`CL-04`)
        self.r_low_raw = nn.Parameter(torch.full((num_channels,), 2.2, dtype=dtype))
        self.dr_span_raw = nn.Parameter(torch.full((num_channels,), -2.5, dtype=dtype))
        self.theta_rad = nn.Parameter(thetas)
        self.w_sel = nn.Parameter(torch.full((num_channels,), 0.15, dtype=dtype))
        self.b_in_gain = nn.Parameter(torch.ones(num_channels, dtype=dtype))
        self.c_real = nn.Parameter(torch.ones(num_channels, dtype=dtype))
        self.c_imag = nn.Parameter(torch.zeros(num_channels, dtype=dtype))

    def clamped_radii(self) -> tuple[Tensor, Tensor]:
        """Returns `(r_low, dr_span)` guaranteed to satisfy `r_low + dr_span in [0.25, 0.9995]`
        (`CL-04`)."""
        r_low = 0.25 + (0.9995 - 0.25) * torch.sigmoid(self.r_low_raw)
        max_span = 0.9995 - r_low
        dr_span = max_span * torch.sigmoid(self.dr_span_raw)
        return r_low, dr_span

    def forward(
        self,
        x: Tensor,
        z_init: Tensor | None = None,
        *,
        delta_r_low: Tensor | None = None,
        delta_theta_rad: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Processes `x` of shape `(B, C, L)` or `(C, L)` and returns `(y_out, z_final)`."""
        squeeze_batch = x.ndim == 2
        inp = x.unsqueeze(0) if squeeze_batch else x
        if inp.ndim != 3 or inp.shape[1] != self.num_channels:
            raise ValueError(
                f"Expected input shape (B, {self.num_channels}, L), got {tuple(x.shape)}"
            )

        work_dtype = torch.float32
        complex_dtype = torch.complex64
        dev_type = (
            inp.device.type if inp.device.type in ("cuda", "cpu", "mps", "xla", "hpu") else "cpu"
        )

        with torch.autocast(device_type=dev_type, enabled=False):
            x_f = inp.to(dtype=work_dtype)
            r_low_base, dr_span_base = self.clamped_radii()
            r_low = r_low_base.to(dtype=work_dtype).view(1, self.num_channels, 1)
            dr_span = dr_span_base.to(dtype=work_dtype).view(1, self.num_channels, 1)
            theta = self.theta_rad.to(dtype=work_dtype).view(1, self.num_channels, 1)

            if delta_r_low is not None:
                r_low = torch.clamp(r_low + delta_r_low.to(dtype=work_dtype), 0.25, 0.9995)
                dr_span = torch.minimum(dr_span, 0.9995 - r_low)
            if delta_theta_rad is not None:
                theta = torch.clamp(
                    theta + delta_theta_rad.to(dtype=work_dtype),
                    1.0e-4,
                    math.pi - 1.0e-4,
                )

            # CL-03: Branchless 4-FMA + Algebraic Newton unit-circle rotor
            u_cos, u_sin = algebraic_unit_rotor_torch(theta)

            # CL-05: 48 kHz radial selectivity gate (smoothstep / clamped envelope)
            w_s = torch.clamp(self.w_sel.to(dtype=work_dtype), 0.0, 4.0).view(
                1, self.num_channels, 1
            )
            q = torch.clamp(w_s * x_f.abs(), 0.0, 1.0)
            g_sel = q * q * (3.0 - 2.0 * q)

            # CL-04: Radial-polar pole radius r[n] in [0.25, 0.9995] & H2 norm scaling
            r_n = torch.clamp(r_low + dr_span * (1.0 - g_sel), 0.25, 0.9995)
            r_bar = torch.clamp(r_low + 0.5 * dr_span, 0.25, 0.9995)
            gamma_norm = h2_input_norm_gain(r_bar)
            b_gamma = self.b_in_gain.to(dtype=work_dtype).view(1, self.num_channels, 1) * gamma_norm

            a_re = r_n * u_cos
            a_im = r_n * u_sin
            a_cplx = torch.complex(a_re, a_im).to(dtype=complex_dtype)
            b_x_cplx = torch.complex(b_gamma * x_f, torch.zeros_like(x_f)).to(dtype=complex_dtype)

            z_seq, z_final = parallel_associative_scan_complex(a_cplx, b_x_cplx, z0=z_init)

            c_r = self.c_real.to(dtype=work_dtype).view(1, self.num_channels, 1)
            c_i = self.c_imag.to(dtype=work_dtype).view(1, self.num_channels, 1)
            y_out = c_r * z_seq.real - c_i * z_seq.imag

        out = y_out.to(dtype=inp.dtype)
        return (out.squeeze(0) if squeeze_batch else out), z_final.detach()

    def forward_complex_features(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Processes `x` of shape `(B, C, L)` and returns `(z_real, z_imag)` of shape `(B, C, L)`
        under `CL-03`..`CL-05` radial-polar selectivity and $\\mathcal{H}_2$ input scaling.
        """
        inp = x.unsqueeze(0) if x.ndim == 2 else x
        work_dtype = torch.float32
        complex_dtype = torch.complex64
        dev_type = (
            inp.device.type if inp.device.type in ("cuda", "cpu", "mps", "xla", "hpu") else "cpu"
        )

        with torch.autocast(device_type=dev_type, enabled=False):
            x_f = inp.to(dtype=work_dtype)
            r_low_base, dr_span_base = self.clamped_radii()
            r_low = r_low_base.to(dtype=work_dtype).view(1, self.num_channels, 1)
            dr_span = dr_span_base.to(dtype=work_dtype).view(1, self.num_channels, 1)
            theta = torch.clamp(
                self.theta_rad.to(dtype=work_dtype).view(1, self.num_channels, 1),
                1.0e-4,
                math.pi - 1.0e-4,
            )
            u_cos, u_sin = algebraic_unit_rotor_torch(theta)
            w_s = torch.clamp(self.w_sel.to(dtype=work_dtype), 0.0, 4.0).view(
                1, self.num_channels, 1
            )
            q = torch.clamp(w_s * x_f.abs(), 0.0, 1.0)
            g_sel = q * q * (3.0 - 2.0 * q)

            r_n = torch.clamp(r_low + dr_span * (1.0 - g_sel), 0.25, 0.9995)
            r_bar = torch.clamp(r_low + 0.5 * dr_span, 0.25, 0.9995)
            gamma_norm = h2_input_norm_gain(r_bar)
            b_gamma = self.b_in_gain.to(dtype=work_dtype).view(1, self.num_channels, 1) * gamma_norm

            a_cplx = torch.complex(r_n * u_cos, r_n * u_sin).to(dtype=complex_dtype)
            b_x_cplx = torch.complex(b_gamma * x_f, torch.zeros_like(x_f)).to(dtype=complex_dtype)
            z_seq, _ = parallel_associative_scan_complex(a_cplx, b_x_cplx)
            return z_seq.real.to(dtype=inp.dtype), z_seq.imag.to(dtype=inp.dtype)

    def to_reference_dsp(self, channel: int = 0) -> SelectiveRgLru:
        """Exports `channel` into a validated `dnam_engine.dsp.SelectiveRgLru` processor."""
        if not (0 <= channel < self.num_channels):
            raise IndexError(f"channel {channel} out of bounds for {self.num_channels} channels")
        with torch.no_grad():
            r_low_t, dr_span_t = self.clamped_radii()
            r_low_v = float(r_low_t[channel].cpu().item())
            dr_span_v = float(dr_span_t[channel].cpu().item())
            theta_t = self.theta_rad[channel].to(dtype=torch.float32, device="cpu")
            u_c, u_s = algebraic_unit_rotor_torch(theta_t)
            r_bar = min(0.9995, max(0.25, r_low_v + 0.5 * dr_span_v))
            gamma_v = math.sqrt(max(1e-7, 1.0 - r_bar * r_bar))
            b_gamma_v = float(self.b_in_gain[channel].cpu().item()) * gamma_v
            w_sel_v = float(torch.clamp(self.w_sel[channel], 0.0, 4.0).cpu().item())

            params = SelectiveRgLruParams(
                r_low=r_low_v,
                dr_span=dr_span_v,
                w_sel=w_sel_v,
                b_gamma_norm=b_gamma_v,
                c_real=float(self.c_real[channel].cpu().item()),
                c_imag=float(self.c_imag[channel].cpu().item()),
                u_rot_cos=float(u_c.item()),
                u_rot_sin=float(u_s.item()),
            )
            return SelectiveRgLru(params)


__all__ = [
    "DifferentiableSelectiveRgLru",
    "algebraic_unit_rotor_torch",
    "h2_input_norm_gain",
    "log_octave_pole_init",
    "parallel_associative_scan_complex",
]
