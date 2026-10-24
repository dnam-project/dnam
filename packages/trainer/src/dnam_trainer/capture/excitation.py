"""24-bit PCM WAV I/O, 3-pulse asymmetric alignment burst synthesis, and 60s
system-identification dry excitation generation (`zero scipy/soundfile dependencies`)."""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dnam_trainer.schema import TargetExcitationProfile, coerce_audio_f32

ALIGNMENT_PULSE_DURATION_SEC: float = 0.35
ALIGNMENT_SETTLE_SILENCE_SEC: float = 0.10
ALIGNMENT_HEADER_TOTAL_SEC: float = ALIGNMENT_PULSE_DURATION_SEC + ALIGNMENT_SETTLE_SILENCE_SEC
TRAILING_NOISE_WINDOW_SEC: float = 0.50


def read_wav(path: str | Path) -> tuple[NDArray[np.float32], int]:
    """Reads a 16-bit, 24-bit, or 32-bit PCM WAV file into `(L,)` or `(2, L)` `float32` in `[-1,
    1]`."""
    wav_path = Path(path)
    if not wav_path.is_file():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate_hz = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    if n_channels not in (1, 2):
        raise ValueError(
            f"Unsupported WAV channel count {n_channels} in {wav_path} (expected 1 or 2)"
        )
    if n_frames == 0:
        raise ValueError(f"WAV file {wav_path} contains 0 frames")

    if sampwidth == 2:
        ints = np.frombuffer(raw_bytes, dtype="<i2").astype(np.float32)
        floats = ints / 32768.0
    elif sampwidth == 3:
        raw_u8 = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, 3)
        b0 = raw_u8[:, 0].astype(np.int32)
        b1 = raw_u8[:, 1].astype(np.int32) << 8
        b2 = raw_u8[:, 2].astype(np.int8).astype(np.int32) << 16
        ints24 = b0 | b1 | b2
        floats = ints24.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        ints32 = np.frombuffer(raw_bytes, dtype="<i4").astype(np.float32)
        floats = ints32 / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width {sampwidth} bytes in {wav_path}")

    if n_channels == 1:
        return np.ascontiguousarray(floats, dtype=np.float32), sample_rate_hz

    interleaved = floats.reshape(-1, 2)
    stereo = np.ascontiguousarray(interleaved.T, dtype=np.float32)
    return stereo, sample_rate_hz


def write_wav_24bit(
    path: str | Path,
    audio: ArrayLike,
    sample_rate_hz: int = 48_000,
) -> Path:
    """Writes a mono `(L,)` or stereo `(2, L)` array as a 24-bit signed little-endian PCM WAV."""
    if sample_rate_hz < 8_000 or sample_rate_hz > 192_000:
        raise ValueError(f"Invalid sample_rate_hz {sample_rate_hz}")

    arr = coerce_audio_f32(audio, allow_stereo=True)
    if arr.ndim == 1:
        n_channels = 1
        n_frames = int(arr.shape[0])
        interleaved = arr
    else:
        n_channels = 2
        n_frames = int(arr.shape[1])
        interleaved = np.ascontiguousarray(arr.T.reshape(-1), dtype=np.float32)

    clamped = np.clip(interleaved, -1.0, 8388607.0 / 8388608.0)
    scaled = np.rint(clamped * 8388608.0).astype(np.int32)
    u32 = scaled & 0xFFFFFF
    out_u8 = np.empty((u32.shape[0], 3), dtype=np.uint8)
    out_u8[:, 0] = (u32 & 0xFF).astype(np.uint8)
    out_u8[:, 1] = ((u32 >> 8) & 0xFF).astype(np.uint8)
    out_u8[:, 2] = ((u32 >> 16) & 0xFF).astype(np.uint8)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(3)
        wf.setframerate(sample_rate_hz)
        wf.writeframes(out_u8.tobytes())
    _ = n_frames
    return out_path


def synth_alignment_pulses(sample_rate_hz: int = 48_000) -> NDArray[np.float32]:
    """Synthesizes the 0.45s 3-pulse asymmetric alignment burst (`0.35s` chirps + `0.10s` silence).

    Uses 3 Hann-tapered log chirps (`800 Hz -> 6 kHz`, `12 ms` each) at asymmetric offsets
    `t = 50 ms, 115 ms, 210 ms` with polarities `+0.90, -0.90, +0.675`, followed by `100 ms`
    of exact silence so downstream nonlinear stages settle to zero before program audio begins.
    """
    if sample_rate_hz < 8_000:
        raise ValueError(f"Invalid sample_rate_hz: {sample_rate_hz}")

    n_total = round(ALIGNMENT_HEADER_TOTAL_SEC * sample_rate_hz)
    out = np.zeros(n_total, dtype=np.float64)
    chirp_len = max(16, round(0.012 * sample_rate_hz))
    t_chirp = np.arange(chirp_len, dtype=np.float64) / float(sample_rate_hz)
    f0 = 800.0
    f1 = min(6000.0, 0.4 * sample_rate_hz)
    dur_c = float(chirp_len) / float(sample_rate_hz)
    k_rate = math.log(f1 / f0) / max(dur_c, 1e-6)
    phase = 2.0 * math.pi * f0 * ((np.exp(k_rate * t_chirp) - 1.0) / k_rate)
    window = np.hanning(chirp_len)
    chirp_wave = np.sin(phase) * window

    pulses = [
        (0.050, +0.90),
        (0.115, -0.90),
        (0.210, +0.675),
    ]
    for onset_sec, gain in pulses:
        idx0 = round(onset_sec * sample_rate_hz)
        idx1 = min(n_total, idx0 + chirp_len)
        if idx1 > idx0:
            out[idx0:idx1] += gain * chirp_wave[: idx1 - idx0]

    return np.ascontiguousarray(out, dtype=np.float32)


def synth_calibration_probe(
    *,
    duration_sec: float = 2.4,
    sample_rate_hz: int = 48_000,
    seed: int = 101,
) -> NDArray[np.float32]:
    """Synthesizes the diagnostic calibration probe used in Step 1 (Bypass) and Step 2 (Hot Corner).

    Contains:
    1. Leading `0.45s` 3-pulse asymmetric alignment burst (`synth_alignment_pulses`).
    2. Log-amplitude nonlinearity ramp (`-40 dBFS -> 0 dBFS`) to measure `knee_amplitude_lin` and
       `saturation_ceiling_lin`.
    3. Steady harmonic chord carrier + fast log sweep to measure `detected_lfo_hz` and
       `dominant_resonance_hz`.
    4. Sharp plosive transient burst followed by an unmasked silent listening window to measure
       `has_delay_or_reverb_tail`, `estimated_max_delay_ms`, and `estimated_tail_t60_ms`.
    """
    if duration_sec < 1.5:
        raise ValueError(f"Probe duration_sec must be >= 1.5s, got {duration_sec}")
    if sample_rate_hz < 8_000:
        raise ValueError(f"Invalid sample_rate_hz: {sample_rate_hz}")

    _ = seed
    n_total = round(duration_sec * sample_rate_hz)
    header = synth_alignment_pulses(sample_rate_hz)
    n_header = int(header.shape[0])
    n_tail = min(round(TRAILING_NOISE_WINDOW_SEC * sample_rate_hz), max(64, n_total // 4))
    n_body = max(64, n_total - n_header - n_tail)

    # Part A (0..40% of body): Log-amplitude ramp (-40 dBFS -> 0 dBFS) on 220 + 440 Hz dyad
    a_end = int(0.40 * n_body)
    ta = np.arange(a_end, dtype=np.float64) / float(sample_rate_hz)
    amp_ramp = np.logspace(-2.0, 0.0, num=max(1, a_end))
    part_a = amp_ramp * (
        0.70 * np.sin(2.0 * math.pi * 220.0 * ta) + 0.30 * np.sin(2.0 * math.pi * 440.0 * ta)
    )

    # Part B (40..80% of body): Steady harmonic carrier + fast log sweep (60 Hz -> 10 kHz)
    b_end = int(0.80 * n_body)
    nb = max(1, b_end - a_end)
    tb = np.arange(nb, dtype=np.float64) / float(sample_rate_hz)
    dur_b = max(float(nb) / float(sample_rate_hz), 1e-4)
    f0, f1 = 60.0, min(10_000.0, 0.42 * sample_rate_hz)
    k_b = math.log(f1 / f0) / dur_b
    phi_b = 2.0 * math.pi * f0 * ((np.exp(k_b * tb) - 1.0) / k_b)
    steady_carrier = 0.35 * (
        np.sin(2.0 * math.pi * 330.0 * tb) + 0.5 * np.sin(2.0 * math.pi * 660.0 * tb)
    )
    part_b = steady_carrier + 0.35 * np.sin(phi_b)

    # Part C (80..100% of body): Sharp 15 ms plosive transient + silent echo listening gap
    nc = max(1, n_body - b_end)
    part_c = np.zeros(nc, dtype=np.float64)
    burst_samples = min(nc, max(16, round(0.015 * sample_rate_hz)))
    t_burst = np.arange(burst_samples, dtype=np.float64) / float(sample_rate_hz)
    part_c[:burst_samples] = (
        0.92
        * np.hanning(burst_samples)
        * (
            0.5 * np.sin(2.0 * math.pi * 500.0 * t_burst)
            + 0.5 * np.sin(2.0 * math.pi * 1800.0 * t_burst)
        )
    )

    body = np.concatenate([part_a, part_b, part_c])
    probe = np.zeros(n_total, dtype=np.float32)
    probe[:n_header] = header[: min(n_header, n_total)]
    end_body = min(n_total, n_header + n_body)
    probe[n_header:end_body] = body[: end_body - n_header].astype(np.float32)
    return probe


def analyze_probe_for_excitation(
    dry_probe: ArrayLike,
    aligned_wet: ArrayLike,
    *,
    sample_rate_hz: int = 48_000,
) -> TargetExcitationProfile:
    """Analyzes the Step-2 engaged probe return to extract `TargetExcitationProfile`."""
    dry_f32 = coerce_audio_f32(dry_probe, allow_stereo=False)
    wet_f32 = coerce_audio_f32(aligned_wet, allow_stereo=True)
    dry_mono = dry_f32.astype(np.float64)
    wet_mono = (
        wet_f32.astype(np.float64)
        if wet_f32.ndim == 1
        else 0.5 * (wet_f32[0].astype(np.float64) + wet_f32[1].astype(np.float64))
    )
    n_samples = min(int(dry_mono.shape[0]), int(wet_mono.shape[0]))
    dry_mono = dry_mono[:n_samples]
    wet_mono = wet_mono[:n_samples]

    # 1. Nonlinearity knee & saturation ceiling from windowed RMS envelope curve
    win = max(1, min(n_samples, max(32, round(0.010 * sample_rate_hz))))
    n_wins = max(1, n_samples // win)
    d_wins = dry_mono[: n_wins * win].reshape(n_wins, win)
    w_wins = wet_mono[: n_wins * win].reshape(n_wins, win)
    d_env = np.sqrt(np.mean(d_wins * d_wins, axis=1))
    w_env = np.sqrt(np.mean(w_wins * w_wins, axis=1))

    active_mask = d_env > 0.015
    knee_lin = 0.15
    sat_lin = 0.85
    if int(np.sum(active_mask)) >= 4:
        d_act = d_env[active_mask]
        w_act = w_env[active_mask]
        order = np.argsort(d_act)
        d_sorted = d_act[order]
        w_sorted = w_act[order]
        small_sig_gain = float(
            np.median(
                w_sorted[: max(1, len(w_sorted) // 4)] / d_sorted[: max(1, len(d_sorted) // 4)]
            )
        )
        inst_gains = w_sorted / np.maximum(d_sorted, 1e-6)
        compressed_idx = np.flatnonzero(inst_gains < 0.85 * small_sig_gain)
        if compressed_idx.size > 0:
            knee_lin = float(np.clip(d_sorted[int(compressed_idx[0])] * math.sqrt(2.0), 0.02, 0.90))
        w_max = max(float(np.max(w_sorted)), 1e-6)
        sat_idx = np.flatnonzero(w_sorted >= 0.92 * w_max)
        if sat_idx.size > 0:
            sat_lin = float(
                np.clip(d_sorted[int(sat_idx[0])] * math.sqrt(2.0), max(knee_lin + 0.05, 0.10), 1.0)
            )

    # 2. Echo / reverb tail detection in the post-burst silent listening window
    active_dry_indices = np.flatnonzero(np.abs(dry_mono) >= 1e-3)
    listen_start = (
        min(n_samples, int(active_dry_indices[-1]) + round(0.005 * sample_rate_hz))
        if active_dry_indices.size > 0
        else int(0.65 * n_samples)
    )
    quiet_wet = wet_mono[listen_start:]
    has_tail = False
    est_delay_ms = 0.0
    est_t60_ms = 0.0
    if quiet_wet.size >= 64:
        tail_rms = float(np.sqrt(np.mean(quiet_wet * quiet_wet)))
        tail_peak = float(np.max(np.abs(quiet_wet)))
        if tail_rms > 2.5e-3 or tail_peak > 0.02:
            has_tail = True
            peak_offset = int(np.argmax(np.abs(quiet_wet)))
            est_delay_ms = float(
                np.clip(
                    1000.0 * max(peak_offset, round(0.015 * sample_rate_hz)) / sample_rate_hz,
                    15.0,
                    2000.0,
                )
            )
            half_len = max(16, len(quiet_wet) // 2)
            rms_early = max(float(np.sqrt(np.mean(quiet_wet[:half_len] ** 2))), 1e-6)
            rms_late = max(float(np.sqrt(np.mean(quiet_wet[half_len:] ** 2))), 1e-7)
            dt_ms = 1000.0 * float(half_len) / float(sample_rate_hz)
            decay_db = max(1.5, 20.0 * math.log10(max(rms_early / rms_late, 1.05)))
            est_t60_ms = float(np.clip(dt_ms * (60.0 / decay_db), 80.0, 5000.0))

    # 3. Autonomous LFO modulation rate on active envelope
    detected_lfo = 0.0
    if n_wins >= 16:
        env_ac = w_env - np.mean(w_env)
        env_dc = max(float(np.mean(w_env)), 1e-6)
        if float(np.std(env_ac)) / env_dc > 0.08:
            env_fs = float(sample_rate_hz) / float(win)
            spec_e = np.abs(np.fft.rfft(env_ac * np.hanning(n_wins)))
            freqs_e = np.fft.rfftfreq(n_wins, d=1.0 / env_fs)
            lfo_mask = (freqs_e >= 0.25) & (freqs_e <= 15.0)
            if np.any(lfo_mask):
                valid_f = freqs_e[lfo_mask]
                valid_s = spec_e[lfo_mask]
                detected_lfo = float(np.clip(valid_f[int(np.argmax(valid_s))], 0.0, 20.0))

    # 4. Dominant spectral resonance frequency
    spec_d = np.abs(np.fft.rfft(dry_mono))
    spec_w = np.abs(np.fft.rfft(wet_mono))
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / float(sample_rate_hz))
    band_mask = (freqs >= 60.0) & (freqs <= min(12_000.0, 0.42 * sample_rate_hz))
    dom_res_hz = 800.0
    if np.any(band_mask):
        tf_ratio = spec_w[band_mask] / (spec_d[band_mask] + 1e-2 * max(float(np.max(spec_d)), 1e-6))
        dom_res_hz = float(np.clip(freqs[band_mask][int(np.argmax(tf_ratio))], 40.0, 16_000.0))

    return TargetExcitationProfile(
        knee_amplitude_lin=knee_lin,
        saturation_ceiling_lin=sat_lin,
        has_delay_or_reverb_tail=has_tail,
        estimated_max_delay_ms=est_delay_ms,
        estimated_tail_t60_ms=est_t60_ms,
        detected_lfo_hz=detected_lfo,
        dominant_resonance_hz=dom_res_hz,
    )


def generate_dry_excitation(
    output_path: str | Path | None = None,
    *,
    duration_sec: float = 60.0,
    sample_rate_hz: int = 48_000,
    seed: int = 42,
    profile: TargetExcitationProfile | None = None,
) -> NDArray[np.float32]:
    """Synthesizes the DNAM system-identification dry excitation signal (adapted to `profile` when
    provided).

    Structure:
    1. Leading `0.45s` 3-pulse asymmetric alignment burst + settling silence
    (`synth_alignment_pulses`).
    2. Core excitation body adapted to `profile`:
       - Logarithmic frequency sweeps (`25 Hz -> 16 kHz`) warped toward `dominant_resonance_hz`
         + steady LFO carrier window when `detected_lfo_hz > 0`,
       - Transient velocity ladders & palm-mute bursts clustered around
         `[knee_amplitude_lin, saturation_ceiling_lin]`, with unmasked silent echo-isolation
         windows when `has_delay_or_reverb_tail` is True,
       - Dynamic amplitude swells + resonance-centered CCIF two-tone + Schroeder multitone.
    3. Trailing silent window (`TRAILING_NOISE_WINDOW_SEC`) for automatic per-stem
       `noise_floor_dbfs` and tail decay measurement.
    """
    if duration_sec < 1.5:
        raise ValueError(f"duration_sec must be >= 1.5 seconds, got {duration_sec}")
    if sample_rate_hz < 8_000:
        raise ValueError(f"Invalid sample_rate_hz: {sample_rate_hz}")

    prof = profile or TargetExcitationProfile()
    rng = np.random.default_rng(seed)
    n_total = round(duration_sec * sample_rate_hz)
    header = synth_alignment_pulses(sample_rate_hz)
    n_header = int(header.shape[0])
    tail_sec = (
        max(TRAILING_NOISE_WINDOW_SEC, min(1.2, prof.estimated_tail_t60_ms / 1000.0 * 0.5))
        if prof.has_delay_or_reverb_tail
        else TRAILING_NOISE_WINDOW_SEC
    )
    n_tail = min(round(tail_sec * sample_rate_hz), max(64, n_total // 4))
    n_body = max(64, n_total - n_header - n_tail)

    t = np.arange(n_body, dtype=np.float64) / float(sample_rate_hz)
    body_dur = float(n_body) / float(sample_rate_hz)

    # Section 1 (0..35%): Stepped logarithmic sweeps + optional steady LFO phase-locking carrier
    s1_end = int(0.35 * n_body)
    t1 = t[:s1_end]
    d1 = max(float(s1_end) / sample_rate_hz, 1e-4)
    f_start, f_stop = 25.0, min(14000.0, 0.42 * sample_rate_hz)
    k_sw = math.log(f_stop / f_start) / d1
    phi_sw = 2.0 * math.pi * f_start * ((np.exp(k_sw * t1) - 1.0) / k_sw)
    env_sw = prof.knee_amplitude_lin + (prof.saturation_ceiling_lin - prof.knee_amplitude_lin) * (
        0.5 - 0.5 * np.cos(2.0 * math.pi * 1.5 * t1 / d1)
    )
    sec1 = env_sw * np.sin(phi_sw)
    if prof.detected_lfo_hz > 0.15 and s1_end > 32:
        half_s1 = s1_end // 2
        tl = t1[half_s1:]
        f_res = min(max(prof.dominant_resonance_hz, 110.0), 1200.0)
        sec1[half_s1:] = prof.knee_amplitude_lin * (
            0.65 * np.sin(2.0 * math.pi * f_res * tl)
            + 0.35 * np.sin(2.0 * math.pi * (1.5 * f_res) * tl)
        )

    # Section 2 (35..70%): Transient velocity ladder, dyads/fifths & echo-isolation gaps
    s2_end = int(0.70 * n_body)
    n2 = s2_end - s1_end
    notes_hz = [55.0, 82.41, 110.0, 146.83, 196.0, 246.94, 329.63, 440.0]
    sec2 = np.zeros(n2, dtype=np.float64)
    burst_len = max(32, min(round(0.18 * sample_rate_hz), max(32, n2 // 8)))
    stride = burst_len * 2 if prof.has_delay_or_reverb_tail else burst_len
    vel_lo = max(0.05, 0.45 * prof.knee_amplitude_lin)
    vel_span = max(0.15, min(0.95, 1.15 * prof.saturation_ceiling_lin) - vel_lo)
    for b_idx, start_idx in enumerate(range(0, n2, max(1, stride))):
        end_idx = min(n2, start_idx + burst_len)
        seg_len = end_idx - start_idx
        tb = np.arange(seg_len, dtype=np.float64) / float(sample_rate_hz)
        f_root = notes_hz[b_idx % len(notes_hz)]
        f_fifth = f_root * 1.4983070768766815
        vel = vel_lo + vel_span * ((b_idx % 6) / 5.0)
        decay_tau = (
            0.030 if prof.has_delay_or_reverb_tail else (0.035 if (b_idx % 3 == 0) else 0.12)
        )
        env_b = (1.0 - np.exp(-tb / 0.002)) * np.exp(-tb / decay_tau)
        wave_b = (
            0.60 * np.sin(2.0 * math.pi * f_root * tb)
            + 0.30 * np.sin(2.0 * math.pi * f_fifth * tb + 0.3)
            + 0.10 * np.sin(2.0 * math.pi * (2.0 * f_root) * tb + 0.7)
        )
        sec2[start_idx:end_idx] = vel * env_b * wave_b

    # Section 3 (70..100%): Dynamic swell + resonance-centered CCIF two-tone + Schroeder multitone
    n3 = n_body - s2_end
    t3 = np.arange(n3, dtype=np.float64) / float(sample_rate_hz)
    d3 = max(float(n3) / sample_rate_hz, 1e-4)
    swell_env = np.sin(math.pi * t3 / d3) ** 2
    f_ccif = min(max(prof.dominant_resonance_hz, 250.0), 3500.0)
    ccif = 0.45 * (
        np.sin(2.0 * math.pi * f_ccif * t3) + np.sin(2.0 * math.pi * (f_ccif + 80.0) * t3)
    )
    multitone = np.zeros(n3, dtype=np.float64)
    n_tones = 16
    for k_tone in range(1, n_tones + 1):
        fk = 40.0 * (1.38 ** (k_tone - 1))
        if fk >= 0.45 * sample_rate_hz:
            break
        schroeder_phase = math.pi * k_tone * (k_tone - 1) / float(n_tones)
        multitone += np.cos(2.0 * math.pi * fk * t3 + schroeder_phase)
    multitone /= max(float(np.max(np.abs(multitone))), 1e-6)
    pink_burst = rng.standard_normal(n3) * 0.04
    sec3 = swell_env * (0.55 * multitone + 0.40 * ccif) + pink_burst * swell_env

    body = np.concatenate([sec1, sec2, sec3])
    # Smooth 5 ms fade-in and fade-out on body so boundary transitions are C^1 click-free
    fade_len = min(max(8, round(0.005 * sample_rate_hz)), n_body // 4)
    fade_win = 0.5 * (1.0 - np.cos(math.pi * np.arange(fade_len, dtype=np.float64) / fade_len))
    body[:fade_len] *= fade_win
    body[-fade_len:] *= fade_win[::-1]
    peak_body = float(np.max(np.abs(body)))
    if peak_body > 0.0:
        body = body * (0.92 / peak_body)

    full_signal = np.zeros(n_total, dtype=np.float32)
    full_signal[:n_header] = header[: min(n_header, n_total)]
    if n_header < n_total:
        end_body = min(n_total, n_header + n_body)
        full_signal[n_header:end_body] = body[: end_body - n_header].astype(np.float32)
    _ = body_dur

    if output_path is not None:
        write_wav_24bit(output_path, full_signal, sample_rate_hz=sample_rate_hz)

    return full_signal


__all__ = [
    "ALIGNMENT_HEADER_TOTAL_SEC",
    "ALIGNMENT_PULSE_DURATION_SEC",
    "ALIGNMENT_SETTLE_SILENCE_SEC",
    "TRAILING_NOISE_WINDOW_SEC",
    "analyze_probe_for_excitation",
    "generate_dry_excitation",
    "read_wav",
    "synth_alignment_pulses",
    "synth_calibration_probe",
    "write_wav_24bit",
]
