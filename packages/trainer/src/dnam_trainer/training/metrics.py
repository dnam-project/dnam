"""Universal Cross-Instrument Psychoacoustic & Multi-Band Audio Modeling Metrics and Losses:
- Per-stem noise-floor measurement (`measure_stem_band_noise_floor`) and silent/active stem
  classification (Design §7),
- Floor-Regularized Target-Adaptive Band-Balanced ESR (`bESR`) across Low/Sub (`20-250 Hz`),
  Mid/Body (`250 Hz - 2.5 kHz`), and High/Air (`2.5 - 18 kHz`), using each stem's measured
  per-band noise floor (or fallback `floor_dbfs`),
- Noise-limited band pass rule (residual `<= 1 dB` above the stem's band noise floor when band
  SNR is below the tier target),
- Band-Split ESR (`ESR_LF`, `ESR_Mid`, `ESR_HF`),
- Floor-Clamped Multi-Resolution Log-STFT Distance (`MR-STFT`, `N_fft in {256..4096}`),
- 1/3-Octave Spectral Tilt (`LTAS`), and
- Segmental Frame-Wise ESR (`SegESR`),
with both 32-bit on-device PyTorch evaluators (for in-training evaluation) and host NumPy
evaluators (for final audit reporting).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor

from dnam_trainer.training.cascade_primitives import split_bezout_3band

DEFAULT_MR_STFT_FFT_SIZES: tuple[int, ...] = (256, 512, 1024, 2048, 4096)
DEFAULT_NOISE_FLOOR_DBFS: float = -60.0
UNFLOORED_DBFS: float = -120.0


@dataclass(frozen=True, slots=True)
class StemNoiseProfile:
    """Measured per-stem per-band noise floor, band energies, and silent/active classification."""

    noise_floor_dbfs_broadband: float
    noise_floor_dbfs_lf: float
    noise_floor_dbfs_mf: float
    noise_floor_dbfs_hf: float
    body_rms_dbfs: float
    band_energy_broadband: float
    band_energy_lf: float
    band_energy_mf: float
    band_energy_hf: float
    is_silent: bool
    source: str  # "measured_silent_frames" | "request_noise_floor" | "unfloored"

    @property
    def band_floors_dbfs(self) -> tuple[float, float, float]:
        return (
            self.noise_floor_dbfs_lf,
            self.noise_floor_dbfs_mf,
            self.noise_floor_dbfs_hf,
        )


@dataclass(frozen=True, slots=True)
class PsychoacousticMetricsReport:
    """Complete cross-instrument psychoacoustic evaluation report for a stem or dataset."""

    besr_db: float
    esr_lf_db: float
    esr_mid_db: float
    esr_hf_db: float
    mr_stft_db: float
    ltas_rms_tilt_db: float
    ltas_hf_tilt_db: float
    seg_esr_db: float
    residual_dbfs_broadband: float = -120.0
    residual_dbfs_lf: float = -120.0
    residual_dbfs_mid: float = -120.0
    residual_dbfs_hf: float = -120.0
    target_dbfs_lf: float = -120.0
    target_dbfs_mid: float = -120.0
    target_dbfs_hf: float = -120.0
    floor_dbfs_lf: float = DEFAULT_NOISE_FLOOR_DBFS
    floor_dbfs_mid: float = DEFAULT_NOISE_FLOOR_DBFS
    floor_dbfs_hf: float = DEFAULT_NOISE_FLOOR_DBFS
    is_silent_stem: bool = False


def is_band_goal_satisfied(
    esr_band_db: float,
    goal_band_db: float,
    *,
    target_band_dbfs: float | None = None,
    residual_band_dbfs: float | None = None,
    noise_floor_band_dbfs: float | None = None,
    noise_limited_margin_db: float = 1.0,
) -> bool:
    """Checks whether a band meets its `goal_band_db` (e.g. `-25.0 dB` for Ultra), applying
    the Design §7 noise-limited rule: if the target band SNR
    (`target_band_dbfs - noise_floor_band_dbfs`) is smaller than `-goal_band_db`, the band is
    noise-limited and passes if
    `residual_band_dbfs <= noise_floor_band_dbfs + noise_limited_margin_db`.
    """
    if esr_band_db <= goal_band_db:
        return True
    if (
        target_band_dbfs is not None
        and residual_band_dbfs is not None
        and noise_floor_band_dbfs is not None
    ):
        band_snr_db = float(target_band_dbfs) - float(noise_floor_band_dbfs)
        required_snr_db = -float(goal_band_db)
        if band_snr_db < required_snr_db:
            return float(residual_band_dbfs) <= (
                float(noise_floor_band_dbfs) + float(noise_limited_margin_db)
            )
    return False


def causal_1pole_lpf_np(
    sig_tc: NDArray[np.floating[Any]],
    cutoff_hz: float,
    *,
    sample_rate_hz: float = 48_000.0,
    bezout_order4: bool = False,
) -> NDArray[np.float64]:
    """Computes an exact causal low-pass filter along axis 0 (`time`) using FFT convolution in
    pure NumPy. When `bezout_order4=True`, evaluates the 4th-order-flat Bezout complementary
    crossover $\\mathcal{L}(z) = H_1(z)^4 (1 + 4(1-H_1) + 10(1-H_1)^2 + 20(1-H_1)^3)$ so that both
    $\\mathcal{L}(z)$ and $1 - \\mathcal{L}(z)$ have $-80\\text{ dB/dec}$ (`-24 dB/oct`) stopbands.
    """
    arr = np.asarray(sig_tc, dtype=np.float64)
    was_1d = arr.ndim == 1
    if was_1d:
        arr = arr[:, None]
    l_seq, c_ch = arr.shape
    if l_seq <= 1:
        return arr[:, 0] if was_1d else arr.copy()

    r = float(math.exp(-2.0 * math.pi * float(cutoff_hz) / float(sample_rate_hz)))
    k_len = min(l_seq, max(64, math.ceil(-16.0 / math.log(max(r, 1e-6)))))
    t_idx = np.arange(k_len, dtype=np.float64)
    ir = (1.0 - r) * np.power(r, t_idx)
    n_fft = 1 << math.ceil(math.log2(l_seq + (7 * k_len if bezout_order4 else k_len)))
    h_1 = np.fft.rfft(ir, n=n_fft)
    if bezout_order4:
        q = 1.0 - h_1
        h_1 = (h_1**4) * (1.0 + q * (4.0 + q * (10.0 + 20.0 * q)))
    h_spec = h_1[:, None]

    out = np.empty_like(arr)
    for c0 in range(0, c_ch, 16):
        c1 = min(c_ch, c0 + 16)
        x_spec = np.fft.rfft(arr[:, c0:c1], n=n_fft, axis=0)
        out[:, c0:c1] = np.fft.irfft(x_spec * h_spec, n=n_fft, axis=0)[:l_seq, :]
    return out[:, 0] if was_1d else out


def split_three_bands_np(
    sig_tc: NDArray[np.floating[Any]],
    *,
    sample_rate_hz: float = 48_000.0,
    lf_cutoff_hz: float = 250.0,
    hf_cutoff_hz: float = 2_500.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Splits `sig_tc` into phase-coherent 4th-order-flat complementary bands
    `(sig_lf, sig_mf, sig_hf)` where `sig_lf + sig_mf + sig_hf == sig_tc` identically.
    """
    arr = np.asarray(sig_tc, dtype=np.float64)
    sig_lf = causal_1pole_lpf_np(
        arr, lf_cutoff_hz, sample_rate_hz=sample_rate_hz, bezout_order4=True
    )
    lpf_hf = causal_1pole_lpf_np(
        arr, hf_cutoff_hz, sample_rate_hz=sample_rate_hz, bezout_order4=True
    )
    sig_mf = lpf_hf - sig_lf
    sig_hf = arr - lpf_hf
    return sig_lf, sig_mf, sig_hf


@torch.no_grad()
def measure_stem_band_noise_floor(
    dry_input: NDArray[np.floating[Any]] | Tensor,
    target: NDArray[np.floating[Any]] | Tensor,
    *,
    sample_rate_hz: int = 48_000,
    request_noise_floor_dbfs: float | None = None,
    min_silent_seconds: float = 0.5,
    dry_silent_dbfs: float = -70.0,
    device: torch.device | str = "cpu",
) -> StemNoiseProfile:
    """Measures a stem's per-band noise floor (`LF`, `MF`, `HF`) and classifies whether the stem
    is `silent` (body energy within 6 dB of its noise floor) or `active` per Design §7.

    1. Identifies 20 ms frames where `dry_input` RMS is below `dry_silent_dbfs` (`-70 dBFS`).
       If at least `min_silent_seconds` (`0.5 s`) of silent-input frames exist, measures the
       target's steady-state per-band noise floor over the quietest half of those frames (avoiding
       reverb/capacitor ring-down immediately following active excitation).
    2. Otherwise falls back to `request_noise_floor_dbfs` if provided.
    3. Otherwise uses `UNFLOORED_DBFS` (`-120 dBFS`).
    """
    dev = torch.device(device)
    if isinstance(dry_input, Tensor):
        x_t = dry_input.reshape(-1).to(device=dev, dtype=torch.float32)
    else:
        x_np = np.ascontiguousarray(np.asarray(dry_input, dtype=np.float32).ravel())
        x_t = torch.from_numpy(x_np).to(device=dev, dtype=torch.float32)

    if isinstance(target, Tensor):
        y_t = target.to(device=dev, dtype=torch.float32)
    else:
        y_np = np.ascontiguousarray(np.asarray(target, dtype=np.float32))
        y_t = torch.from_numpy(y_np).to(device=dev, dtype=torch.float32)

    if y_t.ndim == 1:
        y_t = y_t.unsqueeze(-1)
    l_tot = min(x_t.shape[0], y_t.shape[0])
    x_t = x_t[:l_tot]
    y_t = y_t[:l_tot]

    y_lf, y_mf, y_hf = split_bezout_3band(y_t, sr=float(sample_rate_hz))
    e_bb = float(torch.sum(y_t * y_t).item())
    e_lf = float(torch.sum(y_lf * y_lf).item())
    e_mf = float(torch.sum(y_mf * y_mf).item())
    e_hf = float(torch.sum(y_hf * y_hf).item())

    frame_len = max(16, round(float(sample_rate_hz) * 0.020))
    n_frames = l_tot // frame_len
    min_silent_frames = max(1, math.ceil(float(min_silent_seconds) / 0.020))
    dry_thresh_sq = float(10.0 ** (float(dry_silent_dbfs) / 10.0))

    if n_frames >= 2:
        n_trim = n_frames * frame_len
        x_f = x_t[:n_trim].view(n_frames, frame_len)
        x_ms = torch.mean(x_f * x_f, dim=1)
        silent_mask = x_ms <= dry_thresh_sq
        active_mask = ~silent_mask

        y_f = y_t[:n_trim].view(n_frames, frame_len, -1)
        y_ms_frames = torch.mean(y_f * y_f, dim=(1, 2))
        if bool(torch.any(active_mask).item()):
            body_ms = float(torch.mean(y_ms_frames[active_mask]).item())
        else:
            body_ms = float(torch.mean(y_ms_frames).item())
        body_rms_dbfs = float(10.0 * math.log10(max(body_ms, 1.0e-18)))

        n_silent = int(torch.sum(silent_mask.to(torch.int32)).item())
        if n_silent >= min_silent_frames:
            ylf_f = torch.mean(y_lf[:n_trim].view(n_frames, frame_len, -1) ** 2, dim=(1, 2))
            ymf_f = torch.mean(y_mf[:n_trim].view(n_frames, frame_len, -1) ** 2, dim=(1, 2))
            yhf_f = torch.mean(y_hf[:n_trim].view(n_frames, frame_len, -1) ** 2, dim=(1, 2))

            s_bb_np = (
                y_ms_frames[silent_mask].detach().to(device="cpu", dtype=torch.float32).numpy()
            )
            s_lf_np = ylf_f[silent_mask].detach().to(device="cpu", dtype=torch.float32).numpy()
            s_mf_np = ymf_f[silent_mask].detach().to(device="cpu", dtype=torch.float32).numpy()
            s_hf_np = yhf_f[silent_mask].detach().to(device="cpu", dtype=torch.float32).numpy()

            # Exclude up to 5 frames (100 ms) immediately following any active frame to avoid
            # amplifier/reverb ring-down while keeping an unbiased estimate on stationary noise.
            silent_np = silent_mask.detach().to(device="cpu").numpy().astype(bool)
            steady_silent = silent_np.copy()
            for f_idx in range(len(silent_np)):
                if not silent_np[f_idx]:
                    steady_silent[f_idx : min(len(silent_np), f_idx + 6)] = False
            if int(np.sum(steady_silent)) >= min_silent_frames:
                use_mask_np = steady_silent[silent_np]
            else:
                use_mask_np = np.ones(len(silent_np[silent_np]), dtype=bool)

            s_bb_sub = s_bb_np[use_mask_np]
            s_lf_sub = s_lf_np[use_mask_np]
            s_mf_sub = s_mf_np[use_mask_np]
            s_hf_sub = s_hf_np[use_mask_np]

            # Guard against isolated transient spikes (> 6 dB above median silent frame)
            med_bb = float(np.median(s_bb_sub))
            inlier = s_bb_sub <= max(med_bb * 4.0, 1.0e-18)
            if not np.any(inlier):
                inlier = np.ones_like(s_bb_sub, dtype=bool)

            nf_bb = float(10.0 * math.log10(max(float(np.mean(s_bb_sub[inlier])), 1.0e-18)))
            nf_lf = float(10.0 * math.log10(max(float(np.mean(s_lf_sub[inlier])), 1.0e-18)))
            nf_mf = float(10.0 * math.log10(max(float(np.mean(s_mf_sub[inlier])), 1.0e-18)))
            nf_hf = float(10.0 * math.log10(max(float(np.mean(s_hf_sub[inlier])), 1.0e-18)))
            is_silent = body_rms_dbfs <= (nf_bb + 6.0)
            return StemNoiseProfile(
                noise_floor_dbfs_broadband=nf_bb,
                noise_floor_dbfs_lf=nf_lf,
                noise_floor_dbfs_mf=nf_mf,
                noise_floor_dbfs_hf=nf_hf,
                body_rms_dbfs=body_rms_dbfs,
                band_energy_broadband=e_bb,
                band_energy_lf=e_lf,
                band_energy_mf=e_mf,
                band_energy_hf=e_hf,
                is_silent=is_silent,
                source="measured_silent_frames",
            )
    else:
        body_ms = float(torch.mean(y_t * y_t).item())
        body_rms_dbfs = float(10.0 * math.log10(max(body_ms, 1.0e-18)))

    if request_noise_floor_dbfs is not None:
        nf_req = float(request_noise_floor_dbfs)
        band_nf = nf_req - 4.771212547196624  # 10 * log10(1/3) per band for flat floor
        is_silent = body_rms_dbfs <= (nf_req + 6.0)
        return StemNoiseProfile(
            noise_floor_dbfs_broadband=nf_req,
            noise_floor_dbfs_lf=band_nf,
            noise_floor_dbfs_mf=band_nf,
            noise_floor_dbfs_hf=band_nf,
            body_rms_dbfs=body_rms_dbfs,
            band_energy_broadband=e_bb,
            band_energy_lf=e_lf,
            band_energy_mf=e_mf,
            band_energy_hf=e_hf,
            is_silent=is_silent,
            source="request_noise_floor",
        )

    is_silent = body_rms_dbfs <= (UNFLOORED_DBFS + 6.0)
    return StemNoiseProfile(
        noise_floor_dbfs_broadband=UNFLOORED_DBFS,
        noise_floor_dbfs_lf=UNFLOORED_DBFS,
        noise_floor_dbfs_mf=UNFLOORED_DBFS,
        noise_floor_dbfs_hf=UNFLOORED_DBFS,
        body_rms_dbfs=body_rms_dbfs,
        band_energy_broadband=e_bb,
        band_energy_lf=e_lf,
        band_energy_mf=e_mf,
        band_energy_hf=e_hf,
        is_silent=is_silent,
        source="unfloored",
    )


def compute_band_split_esr_db(
    pred: NDArray[np.floating[Any]],
    target: NDArray[np.floating[Any]],
    *,
    sample_rate_hz: int = 48_000,
    lf_max_hz: float = 250.0,
    mid_max_hz: float = 2_500.0,
    hf_max_hz: float = 18_000.0,
    floor_dbfs: float = DEFAULT_NOISE_FLOOR_DBFS,
    band_floors_dbfs: tuple[float, float, float] | None = None,
) -> tuple[float, float, float]:
    """Computes Band-Split ESR (`ESR_LF`, `ESR_Mid`, `ESR_HF`) in dB across:
    - Low (`< 250 Hz`),
    - Mid (`250 Hz - 2.5 kHz`),
    - High (`2.5 kHz - 18 kHz`),
    using the phase-coherent complementary 3-band crossover `split_three_bands_np`, each normalized
    by that band's own floor-clamped target energy
    $\\|H_b(z)y\\|_2^2 + T \\cdot \\epsilon_{\\text{floor}, b}^2$.
    """
    _ = hf_max_hz
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.ndim == 1:
        p = p[:, None]
    if t.ndim == 1:
        t = t[:, None]
    if p.shape[1] != t.shape[1]:
        if t.shape[1] == 1:
            t = np.repeat(t, p.shape[1], axis=1)
        elif p.shape[1] == 1:
            p = np.repeat(p, t.shape[1], axis=1)
    n_frames = p.shape[0]
    if n_frames < 8:
        return 0.0, 0.0, 0.0

    nf_lf, nf_mf, nf_hf = (
        band_floors_dbfs
        if band_floors_dbfs is not None
        else (float(floor_dbfs), float(floor_dbfs), float(floor_dbfs))
    )
    floor_e_lf = float(p.size) * (10.0 ** (float(nf_lf) / 10.0))
    floor_e_mf = float(p.size) * (10.0 ** (float(nf_mf) / 10.0))
    floor_e_hf = float(p.size) * (10.0 ** (float(nf_hf) / 10.0))

    diff_lf, diff_mf, diff_hf = split_three_bands_np(
        p - t,
        sample_rate_hz=float(sample_rate_hz),
        lf_cutoff_hz=lf_max_hz,
        hf_cutoff_hz=mid_max_hz,
    )
    tgt_lf, tgt_mf, tgt_hf = split_three_bands_np(
        t,
        sample_rate_hz=float(sample_rate_hz),
        lf_cutoff_hz=lf_max_hz,
        hf_cutoff_hz=mid_max_hz,
    )

    if band_floors_dbfs is not None:
        hiss_margin = 10.0 ** (1.0 / 10.0)
        err_lf = max(0.0, float(np.sum(diff_lf**2)) - hiss_margin * floor_e_lf)
        err_mf = max(0.0, float(np.sum(diff_mf**2)) - hiss_margin * floor_e_mf)
        err_hf = max(0.0, float(np.sum(diff_hf**2)) - hiss_margin * floor_e_hf)
        r_lf = err_lf / max(float(np.sum(tgt_lf**2)), floor_e_lf, 1e-18)
        r_mf = err_mf / max(float(np.sum(tgt_mf**2)), floor_e_mf, 1e-18)
        r_hf = err_hf / max(float(np.sum(tgt_hf**2)), floor_e_hf, 1e-18)
    else:
        r_lf = float(np.sum(diff_lf**2)) / (float(np.sum(tgt_lf**2)) + floor_e_lf)
        r_mf = float(np.sum(diff_mf**2)) / (float(np.sum(tgt_mf**2)) + floor_e_mf)
        r_hf = float(np.sum(diff_hf**2)) / (float(np.sum(tgt_hf**2)) + floor_e_hf)
    return (
        float(10.0 * math.log10(max(r_lf, 1e-18))),
        float(10.0 * math.log10(max(r_mf, 1e-18))),
        float(10.0 * math.log10(max(r_hf, 1e-18))),
    )


def compute_floor_regularized_besr_db(
    pred: NDArray[np.floating[Any]],
    target: NDArray[np.floating[Any]],
    *,
    sample_rate_hz: int = 48_000,
    floor_dbfs: float = DEFAULT_NOISE_FLOOR_DBFS,
    band_floors_dbfs: tuple[float, float, float] | None = None,
    band_weights: tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
) -> float:
    """Computes Universal Floor-Regularized Target-Adaptive Band-Balanced ESR (`bESR`) in dB:
    weighted average of per-band error-to-signal ratios across Low, Mid, and High bands.
    """
    p = np.asarray(pred, dtype=np.float64)
    if p.size < 1:
        return 0.0
    esr_lf_db, esr_mf_db, esr_hf_db = compute_band_split_esr_db(
        pred,
        target,
        sample_rate_hz=sample_rate_hz,
        floor_dbfs=floor_dbfs,
        band_floors_dbfs=band_floors_dbfs,
    )
    if esr_lf_db == 0.0 and esr_mf_db == 0.0 and esr_hf_db == 0.0 and p.shape[0] < 8:
        t = np.asarray(target, dtype=np.float64)
        eps_floor_sq = 10.0 ** (float(floor_dbfs) / 10.0)
        r = float(np.sum((p - t) ** 2)) / (float(np.sum(t**2)) + float(p.size) * eps_floor_sq)
        return float(10.0 * math.log10(max(r, 1e-18)))

    w_lf, w_mf, w_hf = band_weights
    r_lf = 10.0 ** (esr_lf_db / 10.0)
    r_mf = 10.0 ** (esr_mf_db / 10.0)
    r_hf = 10.0 ** (esr_hf_db / 10.0)
    besr_lin = w_lf * r_lf + w_mf * r_mf + w_hf * r_hf
    return float(10.0 * math.log10(max(besr_lin, 1e-18)))


def compute_mr_stft_loss_db(
    pred: NDArray[np.floating[Any]],
    target: NDArray[np.floating[Any]],
    *,
    fft_sizes: Sequence[int] = DEFAULT_MR_STFT_FFT_SIZES,
    floor_db: float = DEFAULT_NOISE_FLOOR_DBFS,
) -> float:
    """Computes Floor-Clamped Multi-Resolution Log-STFT Magnitude Distance (`MR-STFT`) in dB across
    $N_{\\text{fft}} \\in \\{256, 512, 1024, 2048, 4096\\}$ using
    $\\log(|Y| + \\epsilon_{\\text{floor}})$.
    """
    p = np.asarray(pred, dtype=np.float32)
    t = np.asarray(target, dtype=np.float32)
    if p.ndim == 2:
        p = np.mean(p, axis=1)
    if t.ndim == 2:
        t = np.mean(t, axis=1)
    n_samples = int(p.shape[0])
    if n_samples < 64:
        return 0.0

    p_t = torch.as_tensor(p, dtype=torch.float32)
    t_t = torch.as_tensor(t, dtype=torch.float32)
    scale_errors: list[float] = []

    for n_fft in fft_sizes:
        eff_fft = min(n_fft, max(64, 1 << math.floor(math.log2(n_samples))))
        hop = max(16, eff_fft // 4)
        win = torch.hann_window(eff_fft, dtype=torch.float32)
        sp = torch.stft(p_t, n_fft=eff_fft, hop_length=hop, window=win, return_complex=True)
        st = torch.stft(t_t, n_fft=eff_fft, hop_length=hop, window=win, return_complex=True)
        mag_p = torch.abs(sp)
        mag_t = torch.abs(st)
        peak_t = float(torch.amax(mag_t).item())
        if peak_t <= 1e-9:
            continue
        eps_floor = peak_t * (10.0 ** (floor_db / 20.0))
        db_p = 20.0 * torch.log10(mag_p + eps_floor)
        db_t = 20.0 * torch.log10(mag_t + eps_floor)
        active_mask = mag_t > eps_floor
        if bool(torch.any(active_mask).item()):
            err_db = float(torch.mean(torch.abs(db_p[active_mask] - db_t[active_mask])).item())
        else:
            err_db = float(torch.mean(torch.abs(db_p - db_t)).item())
        scale_errors.append(err_db)

    return float(np.mean(scale_errors)) if scale_errors else 0.0


def compute_third_octave_ltas_tilt_db(
    pred: NDArray[np.floating[Any]],
    target: NDArray[np.floating[Any]],
    *,
    sample_rate_hz: int = 48_000,
    min_band_dbfs: float = -65.0,
) -> tuple[float, float, float]:
    """Computes 1/3-Octave Long-Term Average Spectrum (`LTAS`) tilt error (`\\Delta EQ_dB`)."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.ndim == 2:
        p = np.mean(p, axis=1)
    if t.ndim == 2:
        t = np.mean(t, axis=1)
    n_frames = p.shape[0]
    if n_frames < 256:
        return 0.0, 0.0, 0.0

    spec_p = np.abs(np.fft.rfft(p)) ** 2 / float(n_frames)
    spec_t = np.abs(np.fft.rfft(t)) ** 2 / float(n_frames)
    freqs = np.fft.rfftfreq(n_frames, d=1.0 / float(sample_rate_hz))

    centers_hz = (
        50.0,
        63.0,
        80.0,
        100.0,
        125.0,
        160.0,
        200.0,
        250.0,
        315.0,
        400.0,
        500.0,
        630.0,
        800.0,
        1000.0,
        1250.0,
        1600.0,
        2000.0,
        2500.0,
        3150.0,
        4000.0,
        5000.0,
        6300.0,
        8000.0,
        10000.0,
        12500.0,
    )
    two_1_6 = 2.0 ** (1.0 / 6.0)
    band_diffs: list[float] = []
    hf_diffs: list[float] = []

    peak_band_e = 1e-18
    band_energies_t: list[tuple[float, float, float]] = []
    for fc in centers_hz:
        f_lo = fc / two_1_6
        f_hi = fc * two_1_6
        mask = (freqs >= f_lo) & (freqs < f_hi)
        if not np.any(mask):
            continue
        ep = float(np.sum(spec_p[mask])) / float(n_frames)
        et = float(np.sum(spec_t[mask])) / float(n_frames)
        peak_band_e = max(peak_band_e, et)
        band_energies_t.append((fc, ep, et))

    floor_e = peak_band_e * (10.0 ** (min_band_dbfs / 10.0))
    for fc, ep, et in band_energies_t:
        if et <= floor_e:
            continue
        diff_db = 10.0 * math.log10(max(ep, 1e-18) / max(et, 1e-18))
        band_diffs.append(diff_db)
        if fc >= 2500.0:
            hf_diffs.append(diff_db)

    if not band_diffs:
        return 0.0, 0.0, 0.0
    arr_d = np.asarray(band_diffs, dtype=np.float64)
    rms_tilt = float(np.sqrt(np.mean(arr_d**2)))
    max_tilt = float(np.max(np.abs(arr_d)))
    hf_tilt = float(np.mean(hf_diffs)) if hf_diffs else 0.0
    return rms_tilt, max_tilt, hf_tilt


def compute_segmental_esr_db(
    pred: NDArray[np.floating[Any]],
    target: NDArray[np.floating[Any]],
    *,
    sample_rate_hz: int = 48_000,
    frame_ms: float = 20.0,
    gate_dbfs: float = DEFAULT_NOISE_FLOOR_DBFS,
) -> float:
    """Computes Segmental / Frame-Wise ESR (`SegESR`) in dB over active `frame_ms` (`20 ms`)
    windows whose target RMS exceeds `gate_dbfs`.
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.ndim == 1:
        p = p[:, None]
    if t.ndim == 1:
        t = t[:, None]
    frame_len = max(16, round(float(sample_rate_hz) * frame_ms / 1000.0))
    n_frames = p.shape[0] // frame_len
    if n_frames < 1:
        err = float(np.sum((p - t) ** 2))
        sig = float(np.sum(t**2)) + 1e-12
        return float(10.0 * math.log10(max(err / sig, 1e-18)))

    p_trim = p[: n_frames * frame_len].reshape(n_frames, frame_len, -1)
    t_trim = t[: n_frames * frame_len].reshape(n_frames, frame_len, -1)
    sig_frame = np.mean(t_trim**2, axis=(1, 2))
    gate_power = 10.0 ** (gate_dbfs / 10.0)
    active = sig_frame > gate_power
    if not np.any(active):
        return float(gate_dbfs)
    err_frame = np.mean((p_trim[active] - t_trim[active]) ** 2, axis=(1, 2))
    esr_frames_db = 10.0 * np.log10(np.maximum(err_frame / (sig_frame[active] + 1e-18), 1e-18))
    return float(np.mean(esr_frames_db))


def evaluate_psychoacoustic_metrics(
    pred: NDArray[np.floating[Any]],
    target: NDArray[np.floating[Any]],
    *,
    sample_rate_hz: int = 48_000,
    floor_dbfs: float = DEFAULT_NOISE_FLOOR_DBFS,
    band_floors_dbfs: tuple[float, float, float] | None = None,
    is_silent_stem: bool = False,
    is_silent: bool | None = None,
) -> PsychoacousticMetricsReport:
    """Computes the complete cross-instrument psychoacoustic evaluation report (`bESR`,
    `ESR_LF`, `ESR_Mid`, `ESR_HF`, `MR-STFT`, `LTAS`, and `SegESR`), supporting per-band
    measured noise floors (`band_floors_dbfs`) and silent-stem reporting per Design §7.
    """
    if is_silent is not None:
        is_silent_stem = is_silent
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.ndim == 1:
        p = p[:, None]
    if t.ndim == 1:
        t = t[:, None]
    if p.shape[1] != t.shape[1]:
        if t.shape[1] == 1:
            t = np.repeat(t, p.shape[1], axis=1)
        elif p.shape[1] == 1:
            p = np.repeat(p, t.shape[1], axis=1)

    nf_lf, nf_mf, nf_hf = (
        band_floors_dbfs
        if band_floors_dbfs is not None
        else (float(floor_dbfs), float(floor_dbfs), float(floor_dbfs))
    )
    eff_floor_bb = (
        max(float(floor_dbfs), nf_lf, nf_mf, nf_hf)
        if band_floors_dbfs is not None
        else float(floor_dbfs)
    )

    sig = float(np.sum(t**2))
    if is_silent_stem or sig <= 1e-12:
        rms_dbfs = float(10.0 * math.log10(max(float(np.mean(p**2)), 1e-18)))
        return PsychoacousticMetricsReport(
            besr_db=rms_dbfs,
            esr_lf_db=rms_dbfs,
            esr_mid_db=rms_dbfs,
            esr_hf_db=rms_dbfs,
            mr_stft_db=0.0,
            ltas_rms_tilt_db=0.0,
            ltas_hf_tilt_db=0.0,
            seg_esr_db=rms_dbfs,
            residual_dbfs_broadband=rms_dbfs,
            residual_dbfs_lf=rms_dbfs,
            residual_dbfs_mid=rms_dbfs,
            residual_dbfs_hf=rms_dbfs,
            floor_dbfs_lf=nf_lf,
            floor_dbfs_mid=nf_mf,
            floor_dbfs_hf=nf_hf,
            is_silent_stem=True,
        )

    diff = p - t
    diff_lf, diff_mf, diff_hf = split_three_bands_np(diff, sample_rate_hz=float(sample_rate_hz))
    tgt_lf, tgt_mf, tgt_hf = split_three_bands_np(t, sample_rate_hz=float(sample_rate_hz))
    denom_sz = float(max(1, p.size))

    res_bb_dbfs = float(10.0 * math.log10(max(float(np.sum(diff**2)) / denom_sz, 1e-18)))
    res_lf_dbfs = float(10.0 * math.log10(max(float(np.sum(diff_lf**2)) / denom_sz, 1e-18)))
    res_mf_dbfs = float(10.0 * math.log10(max(float(np.sum(diff_mf**2)) / denom_sz, 1e-18)))
    res_hf_dbfs = float(10.0 * math.log10(max(float(np.sum(diff_hf**2)) / denom_sz, 1e-18)))

    tgt_lf_dbfs = float(10.0 * math.log10(max(float(np.sum(tgt_lf**2)) / denom_sz, 1e-18)))
    tgt_mf_dbfs = float(10.0 * math.log10(max(float(np.sum(tgt_mf**2)) / denom_sz, 1e-18)))
    tgt_hf_dbfs = float(10.0 * math.log10(max(float(np.sum(tgt_hf**2)) / denom_sz, 1e-18)))

    floor_e_lf = denom_sz * (10.0 ** (float(nf_lf) / 10.0))
    floor_e_mf = denom_sz * (10.0 ** (float(nf_mf) / 10.0))
    floor_e_hf = denom_sz * (10.0 ** (float(nf_hf) / 10.0))
    if band_floors_dbfs is not None:
        hiss_margin = 10.0 ** (1.0 / 10.0)
        err_lf = max(0.0, float(np.sum(diff_lf**2)) - hiss_margin * floor_e_lf)
        err_mf = max(0.0, float(np.sum(diff_mf**2)) - hiss_margin * floor_e_mf)
        err_hf = max(0.0, float(np.sum(diff_hf**2)) - hiss_margin * floor_e_hf)
        r_lf = err_lf / max(float(np.sum(tgt_lf**2)), floor_e_lf, 1e-18)
        r_mf = err_mf / max(float(np.sum(tgt_mf**2)), floor_e_mf, 1e-18)
        r_hf = err_hf / max(float(np.sum(tgt_hf**2)), floor_e_hf, 1e-18)
    else:
        r_lf = float(np.sum(diff_lf**2)) / (float(np.sum(tgt_lf**2)) + floor_e_lf)
        r_mf = float(np.sum(diff_mf**2)) / (float(np.sum(tgt_mf**2)) + floor_e_mf)
        r_hf = float(np.sum(diff_hf**2)) / (float(np.sum(tgt_hf**2)) + floor_e_hf)
    esr_lf_db = float(10.0 * math.log10(max(r_lf, 1e-18)))
    esr_mid_db = float(10.0 * math.log10(max(r_mf, 1e-18)))
    esr_hf_db = float(10.0 * math.log10(max(r_hf, 1e-18)))

    besr_lin = (r_lf + r_mf + r_hf) / 3.0
    besr_db = float(10.0 * math.log10(max(besr_lin, 1e-18)))
    mr_stft_db = compute_mr_stft_loss_db(p, t, floor_db=eff_floor_bb)
    ltas_rms, _ltas_max, ltas_hf = compute_third_octave_ltas_tilt_db(
        p, t, sample_rate_hz=sample_rate_hz
    )
    seg_esr = compute_segmental_esr_db(
        p, t, sample_rate_hz=sample_rate_hz, gate_dbfs=min(eff_floor_bb + 6.0, -55.0)
    )
    return PsychoacousticMetricsReport(
        besr_db=besr_db,
        esr_lf_db=esr_lf_db,
        esr_mid_db=esr_mid_db,
        esr_hf_db=esr_hf_db,
        mr_stft_db=mr_stft_db,
        ltas_rms_tilt_db=ltas_rms,
        ltas_hf_tilt_db=ltas_hf,
        seg_esr_db=seg_esr,
        residual_dbfs_broadband=res_bb_dbfs,
        residual_dbfs_lf=res_lf_dbfs,
        residual_dbfs_mid=res_mf_dbfs,
        residual_dbfs_hf=res_hf_dbfs,
        target_dbfs_lf=tgt_lf_dbfs,
        target_dbfs_mid=tgt_mf_dbfs,
        target_dbfs_hf=tgt_hf_dbfs,
        floor_dbfs_lf=nf_lf,
        floor_dbfs_mid=nf_mf,
        floor_dbfs_hf=nf_hf,
        is_silent_stem=False,
    )


@torch.no_grad()
def compute_band_split_esr_db_torch(
    pred_t: Tensor,
    target_t: Tensor,
    *,
    sample_rate_hz: int = 48_000,
    floor_dbfs: float = DEFAULT_NOISE_FLOOR_DBFS,
    band_floors_dbfs: tuple[float, float, float] | None = None,
) -> tuple[float, float, float, float]:
    """On-device 32-bit (`float32` / `complex64`) evaluation of `(besr_db, esr_lf_db, esr_mid_db,
    esr_hf_db)` without host transfers or 64-bit tensors.
    """
    p = pred_t.to(dtype=torch.float32)
    t = target_t.to(device=p.device, dtype=torch.float32)
    if p.ndim == 1:
        p = p.unsqueeze(-1)
    if t.ndim == 1:
        t = t.unsqueeze(-1)
    if p.shape[1] != t.shape[1]:
        if t.shape[1] == 1:
            t = t.expand(-1, p.shape[1])
        elif p.shape[1] == 1:
            p = p.expand(-1, t.shape[1])
    if p.shape[0] < 8:
        return 0.0, 0.0, 0.0, 0.0

    nf_lf, nf_mf, nf_hf = (
        band_floors_dbfs
        if band_floors_dbfs is not None
        else (float(floor_dbfs), float(floor_dbfs), float(floor_dbfs))
    )
    denom_sz = float(max(1, p.numel()))
    fl_lf = denom_sz * (10.0 ** (float(nf_lf) / 10.0))
    fl_mf = denom_sz * (10.0 ** (float(nf_mf) / 10.0))
    fl_hf = denom_sz * (10.0 ** (float(nf_hf) / 10.0))

    sr_f = float(sample_rate_hz)
    diff_lf, diff_mf, diff_hf = split_bezout_3band(p - t, sr=sr_f)
    tgt_lf, tgt_mf, tgt_hf = split_bezout_3band(t, sr=sr_f)

    if band_floors_dbfs is not None:
        hiss_margin = 10.0 ** (1.0 / 10.0)
        err_lf = max(0.0, float(torch.sum(diff_lf * diff_lf).item()) - hiss_margin * fl_lf)
        err_mf = max(0.0, float(torch.sum(diff_mf * diff_mf).item()) - hiss_margin * fl_mf)
        err_hf = max(0.0, float(torch.sum(diff_hf * diff_hf).item()) - hiss_margin * fl_hf)
        r_lf = err_lf / max(float(torch.sum(tgt_lf * tgt_lf).item()), fl_lf, 1.0e-18)
        r_mf = err_mf / max(float(torch.sum(tgt_mf * tgt_mf).item()), fl_mf, 1.0e-18)
        r_hf = err_hf / max(float(torch.sum(tgt_hf * tgt_hf).item()), fl_hf, 1.0e-18)
    else:
        r_lf = float(torch.sum(diff_lf * diff_lf).item()) / (
            float(torch.sum(tgt_lf * tgt_lf).item()) + fl_lf
        )
        r_mf = float(torch.sum(diff_mf * diff_mf).item()) / (
            float(torch.sum(tgt_mf * tgt_mf).item()) + fl_mf
        )
        r_hf = float(torch.sum(diff_hf * diff_hf).item()) / (
            float(torch.sum(tgt_hf * tgt_hf).item()) + fl_hf
        )
    besr_lin = (r_lf + r_mf + r_hf) / 3.0
    return (
        float(10.0 * math.log10(max(besr_lin, 1.0e-18))),
        float(10.0 * math.log10(max(r_lf, 1.0e-18))),
        float(10.0 * math.log10(max(r_mf, 1.0e-18))),
        float(10.0 * math.log10(max(r_hf, 1.0e-18))),
    )


def _torch_causal_1pole_lpf_btc(
    sig_btc: Tensor,
    cutoff_hz: float,
    sample_rate_hz: float = 48_000.0,
) -> Tensor:
    _bsz, l_seq, _c_ch = sig_btc.shape
    if l_seq <= 1:
        return sig_btc
    r = float(math.exp(-2.0 * math.pi * float(cutoff_hz) / float(sample_rate_hz)))
    k_len = min(l_seq, max(64, math.ceil(-16.0 / math.log(max(r, 1e-6)))))
    t_idx = torch.arange(k_len, device=sig_btc.device, dtype=torch.float32)
    ir = (1.0 - r) * torch.pow(r, t_idx)
    n_fft = 1 << math.ceil(math.log2(l_seq + k_len))
    h_spec = torch.fft.rfft(ir, n=n_fft).view(1, -1, 1)
    x_spec = torch.fft.rfft(sig_btc.to(torch.float32), n=n_fft, dim=1)
    out = torch.fft.irfft(x_spec * h_spec, n=n_fft, dim=1)[:, :l_seq, :]
    return out.to(sig_btc.dtype)


def torch_floor_regularized_besr_loss(
    pred_btc: Tensor,
    target_btc: Tensor,
    *,
    sample_rate_hz: float = 48_000.0,
    floor_dbfs: float = DEFAULT_NOISE_FLOOR_DBFS,
    band_floors_dbfs: tuple[float, float, float] | None = None,
) -> Tensor:
    """Differentiable Universal Floor-Regularized Target-Adaptive Band-Balanced ESR (`bESR`) loss
    for Stage-3 `Adam` optimization.
    """
    diff = pred_btc - target_btc
    nf_lf, nf_mf, nf_hf = (
        band_floors_dbfs
        if band_floors_dbfs is not None
        else (float(floor_dbfs), float(floor_dbfs), float(floor_dbfs))
    )
    eps_lf = float(10.0 ** (float(nf_lf) / 10.0))
    eps_mf = float(10.0 ** (float(nf_mf) / 10.0))
    eps_hf = float(10.0 ** (float(nf_hf) / 10.0))

    diff_lf = _torch_causal_1pole_lpf_btc(diff, 250.0, sample_rate_hz)
    diff_lpf_hf = _torch_causal_1pole_lpf_btc(diff, 2500.0, sample_rate_hz)
    diff_mf = diff_lpf_hf - diff_lf
    diff_hf = diff - diff_lpf_hf

    with torch.no_grad():
        tgt_lf = _torch_causal_1pole_lpf_btc(target_btc, 250.0, sample_rate_hz)
        tgt_lpf_hf = _torch_causal_1pole_lpf_btc(target_btc, 2500.0, sample_rate_hz)
        tgt_mf = tgt_lpf_hf - tgt_lf
        tgt_hf = target_btc - tgt_lpf_hf
        den_lf = torch.mean(tgt_lf * tgt_lf) + eps_lf
        den_mf = torch.mean(tgt_mf * tgt_mf) + eps_mf
        den_hf = torch.mean(tgt_hf * tgt_hf) + eps_hf

    loss_lf = torch.mean(diff_lf * diff_lf) / den_lf
    loss_mf = torch.mean(diff_mf * diff_mf) / den_mf
    loss_hf = torch.mean(diff_hf * diff_hf) / den_hf
    return (loss_lf + loss_mf + loss_hf) / 3.0


def torch_mr_stft_loss(
    pred_btc: Tensor,
    target_btc: Tensor,
    *,
    fft_sizes: Sequence[int] = (256, 512, 1024, 2048),
    floor_dbfs: float = DEFAULT_NOISE_FLOOR_DBFS,
) -> Tensor:
    """Differentiable Floor-Clamped Multi-Resolution Log-STFT Magnitude + Spectral Convergence
    Loss (`MR-STFT`) for Stage-3 `Adam` optimization.
    """
    bsz, t_len, ch = pred_btc.shape
    if t_len < 128:
        return torch.zeros((), device=pred_btc.device, dtype=pred_btc.dtype)
    p_flat = pred_btc.transpose(1, 2).reshape(bsz * ch, t_len)
    t_flat = target_btc.transpose(1, 2).reshape(bsz * ch, t_len)
    eps_floor = float(10.0 ** (float(floor_dbfs) / 20.0))

    total = torch.zeros((), device=pred_btc.device, dtype=pred_btc.dtype)
    count = 0
    for n_fft in fft_sizes:
        if t_len < n_fft:
            continue
        hop = n_fft // 4
        win = torch.hann_window(n_fft, device=pred_btc.device, dtype=pred_btc.dtype)
        sp = torch.stft(p_flat, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
        st = torch.stft(t_flat, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
        mag_p = torch.abs(sp) + eps_floor
        mag_t = torch.abs(st) + eps_floor
        log_l1 = F.l1_loss(torch.log(mag_p), torch.log(mag_t))
        sc = torch.linalg.norm(mag_t - mag_p) / torch.linalg.norm(mag_t).clamp(min=eps_floor)
        total = total + (log_l1 + sc)
        count += 1
    if count == 0:
        return total
    return total / float(count)


__all__ = [
    "DEFAULT_MR_STFT_FFT_SIZES",
    "DEFAULT_NOISE_FLOOR_DBFS",
    "UNFLOORED_DBFS",
    "PsychoacousticMetricsReport",
    "StemNoiseProfile",
    "causal_1pole_lpf_np",
    "compute_band_split_esr_db",
    "compute_band_split_esr_db_torch",
    "compute_floor_regularized_besr_db",
    "compute_mr_stft_loss_db",
    "compute_segmental_esr_db",
    "compute_third_octave_ltas_tilt_db",
    "evaluate_psychoacoustic_metrics",
    "is_band_goal_satisfied",
    "measure_stem_band_noise_floor",
    "split_three_bands_np",
    "torch_floor_regularized_besr_loss",
    "torch_mr_stft_loss",
]
