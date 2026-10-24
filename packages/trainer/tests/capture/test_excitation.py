"""Unit tests for `dnam_trainer.capture.excitation` (24-bit WAV I/O, 3-pulse burst, dry
excitation)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from dnam_trainer.capture.excitation import (
    analyze_probe_for_excitation,
    generate_dry_excitation,
    read_wav,
    synth_alignment_pulses,
    synth_calibration_probe,
    write_wav_24bit,
)


def test_wav_24bit_mono_and_stereo_roundtrip_plus_16bit_32bit(tmp_path: Path) -> None:
    mono = np.sin(np.linspace(0.0, 20.0, 1024, dtype=np.float32)) * 0.75
    mono_path = write_wav_24bit(tmp_path / "mono24.wav", mono, sample_rate_hz=48_000)
    read_m, sr_m = read_wav(mono_path)
    assert sr_m == 48_000
    assert read_m.shape == (1024,)
    assert float(np.max(np.abs(read_m - mono))) < 1e-5

    stereo = np.stack([mono, -0.5 * mono], axis=0)
    st_path = write_wav_24bit(tmp_path / "stereo24.wav", stereo, sample_rate_hz=48_000)
    read_s, sr_s = read_wav(st_path)
    assert sr_s == 48_000
    assert read_s.shape == (2, 1024)
    assert float(np.max(np.abs(read_s - stereo))) < 1e-5

    # 16-bit WAV read
    w16 = tmp_path / "m16.wav"
    ints16 = np.rint(mono * 32767.0).astype("<i2")
    with wave.open(str(w16), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(48_000)
        wf.writeframes(ints16.tobytes())
    r16, _ = read_wav(w16)
    assert float(np.max(np.abs(r16 - mono))) < 1e-3

    # 32-bit int WAV read
    w32 = tmp_path / "m32.wav"
    ints32 = np.rint(mono * 2147483647.0).astype("<i4")
    with wave.open(str(w32), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(4)
        wf.setframerate(48_000)
        wf.writeframes(ints32.tobytes())
    r32, _ = read_wav(w32)
    assert float(np.max(np.abs(r32 - mono))) < 1e-6

    with pytest.raises(FileNotFoundError):
        read_wav(tmp_path / "missing.wav")
    with pytest.raises(ValueError, match="Invalid sample_rate_hz"):
        write_wav_24bit(tmp_path / "bad.wav", mono, sample_rate_hz=100)

    # Error branches in read_wav: 4 channels, 0 frames, 1-byte sample width
    w4ch = tmp_path / "4ch.wav"
    with wave.open(str(w4ch), "wb") as wf:
        wf.setnchannels(4)
        wf.setsampwidth(2)
        wf.setframerate(48_000)
        wf.writeframes(np.zeros(16, dtype="<i2").tobytes())
    with pytest.raises(ValueError, match="Unsupported WAV channel count"):
        read_wav(w4ch)

    w0f = tmp_path / "0f.wav"
    with wave.open(str(w0f), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(48_000)
        wf.writeframes(b"")
    with pytest.raises(ValueError, match="contains 0 frames"):
        read_wav(w0f)

    w8b = tmp_path / "8bit.wav"
    with wave.open(str(w8b), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(48_000)
        wf.writeframes(b"\x80\x80\x80\x80")
    with pytest.raises(ValueError, match="Unsupported WAV sample width"):
        read_wav(w8b)


def test_synth_alignment_pulses_and_dry_excitation(tmp_path: Path) -> None:
    pulses = synth_alignment_pulses(48_000)
    assert pulses.shape == (round(0.45 * 48_000),)
    # Trailing 100 ms of the alignment header must be exact silence
    assert float(np.max(np.abs(pulses[-round(0.08 * 48_000) :]))) == 0.0

    with pytest.raises(ValueError, match="Invalid sample_rate_hz"):
        synth_alignment_pulses(100)
    with pytest.raises(ValueError, match=r"Probe duration_sec must be >= 1\.5s"):
        synth_calibration_probe(duration_sec=0.5)
    with pytest.raises(ValueError, match="Invalid sample_rate_hz"):
        synth_calibration_probe(sample_rate_hz=100)

    out_wav = tmp_path / "dry.wav"
    sig = generate_dry_excitation(out_wav, duration_sec=2.0, sample_rate_hz=48_000, seed=7)
    assert out_wav.is_file()
    assert sig.shape == (96_000,)
    # Trailing silent window is exact zero
    assert float(np.max(np.abs(sig[-8000:]))) == 0.0

    with pytest.raises(ValueError, match=r"duration_sec must be >= 1\.5"):
        generate_dry_excitation(duration_sec=0.5)
    with pytest.raises(ValueError, match="Invalid sample_rate_hz"):
        generate_dry_excitation(sample_rate_hz=100)

    # Low sample-rate (8 kHz) excitation exercises the Nyquist multitone cutoff guard
    sig_8k = generate_dry_excitation(duration_sec=1.6, sample_rate_hz=8_000, seed=9)
    assert sig_8k.shape == (round(1.6 * 8_000),)

    # Fallback profile extraction on silent / ultra-short probe
    silent_prof = analyze_probe_for_excitation(
        np.zeros(256, dtype=np.float32),
        np.zeros((2, 256), dtype=np.float32),
        sample_rate_hz=48_000,
    )
    assert silent_prof.has_delay_or_reverb_tail is False
    assert silent_prof.detected_lfo_hz == 0.0

    # Probe synthesis & target profile extraction (delay + LFO + soft-clip saturation)
    probe = synth_calibration_probe(duration_sec=2.4, sample_rate_hz=48_000)
    assert probe.shape == (round(2.4 * 48_000),)

    t_p = np.arange(probe.shape[0], dtype=np.float64) / 48_000.0
    lfo_mod = 1.0 + 0.35 * np.cos(2.0 * np.pi * 3.0 * t_p)
    driven = np.tanh(probe.astype(np.float64) * 4.5) * lfo_mod
    # Add a 60 ms echo tap into the silent listening window
    echo_samples = round(0.060 * 48_000)
    wet_delay = driven.copy()
    wet_delay[echo_samples:] += 0.45 * driven[:-echo_samples]

    prof = analyze_probe_for_excitation(
        probe,
        wet_delay.astype(np.float32),
        sample_rate_hz=48_000,
    )
    assert prof.has_delay_or_reverb_tail is True
    assert prof.estimated_max_delay_ms >= 15.0
    assert prof.estimated_tail_t60_ms >= 80.0
    assert prof.detected_lfo_hz > 0.0
    assert 0.01 <= prof.knee_amplitude_lin < prof.saturation_ceiling_lin <= 1.0

    # Adaptive excitation synthesis from extracted profile
    tailored = generate_dry_excitation(
        tmp_path / "tailored_dry.wav",
        duration_sec=2.5,
        sample_rate_hz=48_000,
        profile=prof,
    )
    assert tailored.shape == (round(2.5 * 48_000),)
    assert not np.allclose(tailored[:96_000], sig)
