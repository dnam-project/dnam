"""16/24/32-bit PCM WAV reader and 24-bit WAV writer for `LocalWavPathRef` stem resolution."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dnam_trainer.schema import coerce_audio_f32


def read_wav(path: str | Path) -> tuple[NDArray[np.float32], int]:
    """Reads a 16-bit, 24-bit, or 32-bit PCM WAV file into `(L,)` or `(2, L)` `float32`."""
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
        interleaved = arr
    else:
        n_channels = 2
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
    return out_path


__all__ = ["read_wav", "write_wav_24bit"]
