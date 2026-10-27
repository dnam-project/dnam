"""Universal Cross-Instrument Psychoacoustic & Multi-Band Audio Modeling Metrics and Losses:
- Floor-Regularized Target-Adaptive Band-Balanced ESR (`bESR`) across Low/Sub (`20-250 Hz`),
  Mid/Body (`250 Hz - 2.5 kHz`), and High/Air (`2.5 - 18 kHz`), guarded by `-60 dBFS` noise floor,
- Band-Split ESR (`ESR_LF`, `ESR_Mid`, `ESR_HF`),
- Floor-Clamped Multi-Resolution Log-STFT Distance (`MR-STFT`, `N_fft in {256..4096}`),
- 1/3-Octave Spectral Tilt (`LTAS`), and
- Segmental Frame-Wise ESR (`SegESR`).
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

DEFAULT_MR_STFT_FFT_SIZES: tuple[int, ...] = (256, 512, 1024, 2048, 4096)
DEFAULT_NOISE_FLOOR_DBFS: float = -60.0


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


def compute_band_split_esr_db(
    pred: NDArray[np.floating[Any]],
    target: NDArray[np.floating[Any]],
    *,
    sample_rate_hz: int = 48_000,
    lf_max_hz: float = 250.0,
    mid_max_hz: float = 2_500.0,
    hf_max_hz: float = 18_000.0,
    floor_dbfs: float = DEFAULT_NOISE_FLOOR_DBFS,
) -> tuple[float, float, float]:
    """Computes Band-Split ESR (`ESR_LF`, `ESR_Mid`, `ESR_HF`) in dB across:
    - Low (`< 250 Hz`),
    - Mid (`250 Hz - 2.5 kHz`),
    - High (`2.5 kHz - 18 kHz`),
    using the phase-coherent complementary 3-band crossover `split_three_bands_np`, each normalized
    by that band's own floor-clamped target energy
    $\\|H_b(z)y\\|_2^2 + T \\cdot \\epsilon_{\\text{floor}}^2$.
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

    eps_floor_sq = 10.0 ** (float(floor_dbfs) / 10.0)
    floor_energy = float(p.size) * eps_floor_sq

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

    r_lf = float(np.sum(diff_lf**2)) / (float(np.sum(tgt_lf**2)) + floor_energy)
    r_mf = float(np.sum(diff_mf**2)) / (float(np.sum(tgt_mf**2)) + floor_energy)
    r_hf = float(np.sum(diff_hf**2)) / (float(np.sum(tgt_hf**2)) + floor_energy)
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
    band_weights: tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
) -> float:
    """Computes Universal Floor-Regularized Target-Adaptive Band-Balanced ESR (`bESR`) in dB:
    weighted average of per-band error-to-signal ratios across Low, Mid, and High bands.
    Automatically adapts across bass/808s, electric/acoustic guitar, drums/cymbals/synths, and
    clean/low-passed sources without amplifying analog thermal hiss below `floor_dbfs`.
    """
    p = np.asarray(pred, dtype=np.float64)
    if p.size < 1:
        return 0.0
    esr_lf_db, esr_mf_db, esr_hf_db = compute_band_split_esr_db(
        pred,
        target,
        sample_rate_hz=sample_rate_hz,
        floor_dbfs=floor_dbfs,
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
        peak_t = float(torch.max(mag_t).item())
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
    windows whose target RMS exceeds `gate_dbfs` (`-60 dBFS`).
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
        return -60.0
    err_frame = np.mean((p_trim[active] - t_trim[active]) ** 2, axis=(1, 2))
    esr_frames_db = 10.0 * np.log10(np.maximum(err_frame / (sig_frame[active] + 1e-18), 1e-18))
    return float(np.mean(esr_frames_db))


def evaluate_psychoacoustic_metrics(
    pred: NDArray[np.floating[Any]],
    target: NDArray[np.floating[Any]],
    *,
    sample_rate_hz: int = 48_000,
    floor_dbfs: float = DEFAULT_NOISE_FLOOR_DBFS,
) -> PsychoacousticMetricsReport:
    """Computes the complete cross-instrument psychoacoustic evaluation report (`bESR`,
    `ESR_LF`, `ESR_Mid`, `ESR_HF`, `MR-STFT`, `LTAS`, and `SegESR`).
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    sig = float(np.sum(t**2))
    if sig <= 1e-12:
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
        )

    esr_lf_db, esr_mid_db, esr_hf_db = compute_band_split_esr_db(
        p, t, sample_rate_hz=sample_rate_hz, floor_dbfs=floor_dbfs
    )
    besr_lin = (
        10.0 ** (esr_lf_db / 10.0) + 10.0 ** (esr_mid_db / 10.0) + 10.0 ** (esr_hf_db / 10.0)
    ) / 3.0
    besr_db = float(10.0 * math.log10(max(besr_lin, 1e-18)))
    mr_stft_db = compute_mr_stft_loss_db(p, t, floor_db=floor_dbfs)
    ltas_rms, _ltas_max, ltas_hf = compute_third_octave_ltas_tilt_db(
        p, t, sample_rate_hz=sample_rate_hz
    )
    seg_esr = compute_segmental_esr_db(p, t, sample_rate_hz=sample_rate_hz, gate_dbfs=floor_dbfs)
    return PsychoacousticMetricsReport(
        besr_db=besr_db,
        esr_lf_db=esr_lf_db,
        esr_mid_db=esr_mid_db,
        esr_hf_db=esr_hf_db,
        mr_stft_db=mr_stft_db,
        ltas_rms_tilt_db=ltas_rms,
        ltas_hf_tilt_db=ltas_hf,
        seg_esr_db=seg_esr,
    )


def _torch_causal_1pole_lpf_btc(
    sig_btc: torch.Tensor,
    cutoff_hz: float,
    sample_rate_hz: float = 48_000.0,
) -> torch.Tensor:
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
    pred_btc: torch.Tensor,
    target_btc: torch.Tensor,
    *,
    sample_rate_hz: float = 48_000.0,
    floor_dbfs: float = DEFAULT_NOISE_FLOOR_DBFS,
) -> torch.Tensor:
    """Differentiable Universal Floor-Regularized Target-Adaptive Band-Balanced ESR (`bESR`) loss
    for Stage-3 `Adam` optimization.
    """
    diff = pred_btc - target_btc
    eps_floor_sq = float(10.0 ** (float(floor_dbfs) / 10.0))

    diff_lf = _torch_causal_1pole_lpf_btc(diff, 250.0, sample_rate_hz)
    diff_lpf_hf = _torch_causal_1pole_lpf_btc(diff, 2500.0, sample_rate_hz)
    diff_mf = diff_lpf_hf - diff_lf
    diff_hf = diff - diff_lpf_hf

    with torch.no_grad():
        tgt_lf = _torch_causal_1pole_lpf_btc(target_btc, 250.0, sample_rate_hz)
        tgt_lpf_hf = _torch_causal_1pole_lpf_btc(target_btc, 2500.0, sample_rate_hz)
        tgt_mf = tgt_lpf_hf - tgt_lf
        tgt_hf = target_btc - tgt_lpf_hf
        # Guard silent/mute stems in a batch from dividing by `1e-6` while preserving per-band
        # target-adaptive normalization across active stems:
        den_lf = torch.mean(tgt_lf**2) + eps_floor_sq
        den_mf = torch.mean(tgt_mf**2) + eps_floor_sq
        den_hf = torch.mean(tgt_hf**2) + eps_floor_sq

    loss_lf = torch.mean(diff_lf**2) / den_lf
    loss_mf = torch.mean(diff_mf**2) / den_mf
    loss_hf = torch.mean(diff_hf**2) / den_hf
    return (loss_lf + loss_mf + loss_hf) / 3.0


def torch_mr_stft_loss(
    pred_btc: torch.Tensor,
    target_btc: torch.Tensor,
    *,
    fft_sizes: Sequence[int] = (256, 512, 1024, 2048),
    floor_dbfs: float = DEFAULT_NOISE_FLOOR_DBFS,
) -> torch.Tensor:
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
    "PsychoacousticMetricsReport",
    "causal_1pole_lpf_np",
    "compute_band_split_esr_db",
    "compute_floor_regularized_besr_db",
    "compute_mr_stft_loss_db",
    "compute_segmental_esr_db",
    "compute_third_octave_ltas_tilt_db",
    "evaluate_psychoacoustic_metrics",
    "split_three_bands_np",
    "torch_floor_regularized_besr_loss",
    "torch_mr_stft_loss",
]
