"""2-stage coarse-to-fine hardware latency alignment (`align_hardware_latency`),
silent-window noise floor telemetry (`measure_silent_window_noise_floor`), and
engaged hot-corner nonlinearity analysis (`analyze_hot_corner_nonlinearity`)."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import ConfigDict, Field

from dnam_trainer.capture.excitation import ALIGNMENT_HEADER_TOTAL_SEC, TRAILING_NOISE_WINDOW_SEC
from dnam_trainer.schema import StrictModel, coerce_audio_f32


class LatencyAlignmentResult(StrictModel):
    """Result of 2-stage coarse-to-fine hardware latency alignment."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        arbitrary_types_allowed=True,
    )

    latency_offset_samples: int
    peak_correlation: float
    aligned_audio: NDArray[np.float32]


class HotCornerAnalysis(StrictModel):
    """Telemetry from Step-2 Engaged Hot-Corner Verification (0.0 dB send preserved)."""

    measured_return_peak_dbfs: float
    measured_return_rms_dbfs: float
    net_gain_over_unity_db: float
    compression_ratio: float = Field(ge=0.1)
    clipped_sample_fraction: float = Field(ge=0.0, le=1.0)


def _to_mono_reference(arr: NDArray[np.float32]) -> NDArray[np.float64]:
    if arr.ndim == 1:
        return arr.astype(np.float64)
    if float(np.max(np.abs(arr[0]))) > 1e-6:
        return arr[0].astype(np.float64)
    return arr[1].astype(np.float64)


def _shift_and_match_length(
    wet: NDArray[np.float32],
    offset_samples: int,
    target_len: int,
) -> NDArray[np.float32]:
    if wet.ndim == 1:
        out = np.zeros(target_len, dtype=np.float32)
        if offset_samples >= 0:
            src_start = offset_samples
            copy_len = max(0, min(target_len, int(wet.shape[0]) - src_start))
            if copy_len > 0:
                out[:copy_len] = wet[src_start : src_start + copy_len]
        else:
            dst_start = -offset_samples
            copy_len = max(0, min(int(wet.shape[0]), target_len - dst_start))
            if copy_len > 0:
                out[dst_start : dst_start + copy_len] = wet[:copy_len]
        return out

    out2 = np.zeros((2, target_len), dtype=np.float32)
    if offset_samples >= 0:
        src_start = offset_samples
        copy_len = max(0, min(target_len, int(wet.shape[1]) - src_start))
        if copy_len > 0:
            out2[:, :copy_len] = wet[:, src_start : src_start + copy_len]
    else:
        dst_start = -offset_samples
        copy_len = max(0, min(int(wet.shape[1]), target_len - dst_start))
        if copy_len > 0:
            out2[:, dst_start : dst_start + copy_len] = wet[:, :copy_len]
    return out2


def align_hardware_latency(
    dry: ArrayLike,
    wet: ArrayLike,
    *,
    sample_rate_hz: int = 48_000,
    max_latency_sec: float = 0.25,
    causal_margin_samples: int = 0,
    trim_alignment_header: bool = False,
    fixed_latency_samples: int | None = None,
) -> LatencyAlignmentResult:
    """Aligns `wet` to `dry` using a 2-stage coarse-to-fine cross-correlation pipeline,
    or locks to `fixed_latency_samples` when calibrated via Step-1 Bypass Reference.

    - **Calibrated Lock (`fixed_latency_samples is not None`):** Uses the exact hardware
      round-trip latency measured during Step-1 Bypass Calibration so high-`Mix` (`> 50%`) or
      `100%` Wet (`Kill-Dry`) delay/reverb stems never lock onto the first echo repeat
      ($L_{\\text{hw}} + \\tau$) and strip the pedal's delay time $\\tau(\\mathbf{u})$.
    - **Stage 1 (Coarse):** Global FFT cross-correlation of high-passed transient envelopes
      `|x[n] - x[n-1]|` up to `max_latency_sec`.
    - **Stage 2 (Fine):** Time-domain signed cross-correlation in a `+/- 25 ms` window around
      the Stage-1 peak using the leading `0.45s` 3-pulse alignment burst (falls back to signed
      correlation if the signal has no alignment burst).
    - Preserves stereo L/R phase coherence by locking both channels to a single sample offset.
    """
    dry_f32 = coerce_audio_f32(dry, allow_stereo=False)
    wet_f32 = coerce_audio_f32(wet, allow_stereo=True)
    dry_len = int(dry_f32.shape[0])
    wet_mono = _to_mono_reference(wet_f32)
    dry_mono = dry_f32.astype(np.float64)

    probe_len = min(dry_len, max(256, round(0.45 * sample_rate_hz)))
    dry_probe = dry_mono[:probe_len]
    if float(np.max(np.abs(dry_probe))) < 1e-5:
        dry_probe = dry_mono
        probe_len = dry_len

    if fixed_latency_samples is not None:
        best_lag = max(0, fixed_latency_samples)
        seg_end = min(int(wet_mono.shape[0]), best_lag + probe_len)
        seg_len = max(0, seg_end - best_lag)
        best_corr = 0.0
        if seg_len >= 16:
            w_seg = wet_mono[best_lag:seg_end]
            d_seg = dry_probe[:seg_len]
            denom = float(np.linalg.norm(w_seg) * np.linalg.norm(d_seg))
            if denom > 1e-12:
                best_corr = abs(float(np.dot(w_seg, d_seg)) / denom)
    else:
        max_lag = min(
            max(1, round(max_latency_sec * sample_rate_hz)),
            max(1, int(wet_mono.shape[0]) - 1),
        )

        # Stage 1: Coarse FFT cross-correlation of transient envelopes |dx| and signed signals
        dry_env = np.abs(np.diff(dry_mono, prepend=0.0))
        wet_env = np.abs(np.diff(wet_mono, prepend=0.0))
        n_fft = 1 << math.ceil(math.log2(max(dry_len + int(wet_mono.shape[0]), 256)))

        # Combine signed correlation and envelope correlation so polarity-inverted amps still lock
        # in Stage 1
        d_spec_sig = np.fft.rfft(dry_mono, n=n_fft)
        w_spec_sig = np.fft.rfft(wet_mono, n=n_fft)
        xcorr_sig = np.fft.irfft(w_spec_sig * np.conj(d_spec_sig), n=n_fft)[: max_lag + 1]

        d_spec_env = np.fft.rfft(dry_env, n=n_fft)
        w_spec_env = np.fft.rfft(wet_env, n=n_fft)
        xcorr_env = np.fft.irfft(w_spec_env * np.conj(d_spec_env), n=n_fft)[: max_lag + 1]

        sig_norm = max(float(np.max(np.abs(xcorr_sig))), 1e-12)
        env_norm = max(float(np.max(np.abs(xcorr_env))), 1e-12)
        combined_coarse = 0.65 * (np.abs(xcorr_sig) / sig_norm) + 0.35 * (xcorr_env / env_norm)
        coarse_lag = int(np.argmax(combined_coarse))

        # Stage 2: Fine time-domain cross-correlation around coarse_lag (+/- 25 ms)
        half_win = max(16, round(0.025 * sample_rate_hz))
        lag_min = max(0, coarse_lag - half_win)
        lag_max = min(max_lag, coarse_lag + half_win)

        dry_probe_norm = float(np.linalg.norm(dry_probe))
        best_lag = coarse_lag
        best_corr = 0.0

        if dry_probe_norm > 1e-12:
            for lag in range(lag_min, lag_max + 1):
                seg_end = min(int(wet_mono.shape[0]), lag + probe_len)
                seg_len = seg_end - lag
                if seg_len < 16:
                    continue
                w_seg = wet_mono[lag:seg_end]
                d_seg = dry_probe[:seg_len]
                denom = float(np.linalg.norm(w_seg) * np.linalg.norm(d_seg))
                if denom > 1e-12:
                    corr = abs(float(np.dot(w_seg, d_seg)) / denom)
                    if corr > best_corr:
                        best_corr = corr
                        best_lag = lag

    final_lag = max(0, best_lag - causal_margin_samples)
    aligned = _shift_and_match_length(wet_f32, final_lag, dry_len)

    if trim_alignment_header:
        header_samples = round(ALIGNMENT_HEADER_TOTAL_SEC * sample_rate_hz)
        if header_samples < dry_len:
            aligned = aligned[header_samples:] if aligned.ndim == 1 else aligned[:, header_samples:]

    return LatencyAlignmentResult(
        latency_offset_samples=final_lag,
        peak_correlation=float(min(1.0, max(0.0, best_corr))),
        aligned_audio=np.ascontiguousarray(aligned, dtype=np.float32),
    )


def measure_silent_window_noise_floor(
    aligned_wet: ArrayLike,
    *,
    sample_rate_hz: int = 48_000,
    tail_window_sec: float = TRAILING_NOISE_WINDOW_SEC * 0.8,
    baseline_noise_floor_dbfs: float | None = None,
) -> float:
    """Measures the RMS `noise_floor_dbfs` from the trailing silent window of `aligned_wet`.

    On delay/reverb stems where echo repeats ring into the trailing silent window, scans
    sliding `50 ms` sub-windows across the tail (and bounds against `baseline_noise_floor_dbfs`
    when calibrated) so ringing echo repeats are not mistaken for broadband noise floor.
    """
    arr = coerce_audio_f32(aligned_wet, allow_stereo=True)
    n_samples = int(arr.shape[-1])
    win_samples = min(max(16, round(tail_window_sec * sample_rate_hz)), max(16, n_samples // 6))
    tail = arr[-win_samples:] if arr.ndim == 1 else arr[:, -win_samples:]
    tail_f64 = tail.astype(np.float64)

    sub_win = max(16, min(win_samples, round(0.050 * sample_rate_hz)))
    if win_samples >= 2 * sub_win:
        step = max(8, sub_win // 2)
        min_rms = float("inf")
        for start in range(0, win_samples - sub_win + 1, step):
            seg = (
                tail_f64[start : start + sub_win]
                if tail_f64.ndim == 1
                else tail_f64[:, start : start + sub_win]
            )
            seg_rms = float(np.sqrt(np.mean(seg * seg)))
            if seg_rms < min_rms:
                min_rms = seg_rms
        rms = min_rms
    else:
        rms = float(np.sqrt(np.mean(tail_f64 * tail_f64)))

    dbfs = -120.0 if rms <= 1e-6 else float(max(-120.0, min(0.0, 20.0 * math.log10(rms))))

    if baseline_noise_floor_dbfs is not None and baseline_noise_floor_dbfs < -20.0:
        dbfs = min(dbfs, float(baseline_noise_floor_dbfs) + 12.0)
    return dbfs


def analyze_hot_corner_nonlinearity(
    dry_probe: ArrayLike,
    wet_return: ArrayLike,
    *,
    unity_ref_dbfs: float = 0.0,
    clip_threshold_dbfs: float = -0.05,
) -> HotCornerAnalysis:
    """Analyzes Step-2 engaged hot-corner return signal without reducing `send_gain_db`."""
    dry_arr = coerce_audio_f32(dry_probe, allow_stereo=False).astype(np.float64)
    wet_arr = coerce_audio_f32(wet_return, allow_stereo=True).astype(np.float64)

    wet_abs = np.abs(wet_arr)
    peak_lin = max(float(np.max(wet_abs)), 1e-9)
    rms_lin = max(float(np.sqrt(np.mean(wet_arr**2))), 1e-9)
    peak_dbfs = float(max(-120.0, min(6.0, 20.0 * math.log10(peak_lin))))
    rms_dbfs = float(max(-120.0, min(6.0, 20.0 * math.log10(rms_lin))))

    dry_rms = max(float(np.sqrt(np.mean(dry_arr**2))), 1e-9)
    dry_rms_dbfs = 20.0 * math.log10(dry_rms)
    net_gain_over_unity = float((rms_dbfs - dry_rms_dbfs) - unity_ref_dbfs)

    dry_crest = max(float(np.max(np.abs(dry_arr))) / dry_rms, 1.0)
    wet_crest = max(peak_lin / rms_lin, 1.0)
    comp_ratio = max(0.1, dry_crest / max(wet_crest, 1e-6))

    clip_lin = 10.0 ** (clip_threshold_dbfs / 20.0)
    clipped_frac = float(np.mean(wet_abs >= clip_lin))

    return HotCornerAnalysis(
        measured_return_peak_dbfs=peak_dbfs,
        measured_return_rms_dbfs=rms_dbfs,
        net_gain_over_unity_db=net_gain_over_unity,
        compression_ratio=float(comp_ratio),
        clipped_sample_fraction=clipped_frac,
    )


__all__ = [
    "HotCornerAnalysis",
    "LatencyAlignmentResult",
    "align_hardware_latency",
    "analyze_hot_corner_nonlinearity",
    "measure_silent_window_noise_floor",
]
