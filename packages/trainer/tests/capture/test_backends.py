"""Unit tests for `dnam_trainer.capture.backends` (`DarkglassAnagramBackend`,
`UsbAudioInterfaceBackend`,
`SoftwareDspBackend`, 2-step Reference Unity & Hot-Corner Calibration, and `record_single_stem`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dnam_engine import DnamContainer
from dnam_trainer.capture.backends import (
    DarkglassAnagramBackend,
    SoftwareDspBackend,
    UsbAudioInterfaceBackend,
    get_capture_backend,
    list_capture_backends,
    record_single_stem,
    register_capture_backend,
)
from dnam_trainer.capture.excitation import generate_dry_excitation
from dnam_trainer.schema import CaptureBackendConfig


class _FakeSoundDevice:
    """Deterministic fake `sounddevice` module for testing `UsbAudioInterfaceBackend` and
    `DarkglassAnagramBackend`."""

    def query_devices(self) -> list[dict[str, Any]]:
        return [
            {
                "index": 0,
                "name": "Built-in Audio",
                "max_input_channels": 2,
                "max_output_channels": 2,
                "default_samplerate": 48_000.0,
            },
            {
                "index": 1,
                "name": "Darkglass Anagram USB-C Audio",
                "max_input_channels": 8,
                "max_output_channels": 10,
                "default_samplerate": 48_000.0,
            },
        ]

    def playrec(
        self,
        out_buf: np.ndarray,
        *,
        samplerate: int,
        input_mapping: list[int],
        device: int | str | None,
        dtype: str,
        blocking: bool,
    ) -> np.ndarray:
        _ = (samplerate, device, dtype, blocking)
        out_buf.shape[0]
        send_sig = np.sum(out_buf, axis=1)
        delayed = np.zeros_like(send_sig)
        delayed[32:] = send_sig[:-32] * 0.95
        if len(input_mapping) == 1:
            return delayed[:, None].astype(np.float32)
        return np.column_stack([delayed, delayed * 0.98]).astype(np.float32)


def test_software_dsp_backend_calibration_and_record_single_stem(tmp_path: Path) -> None:
    backends = list_capture_backends()
    ids = [b[0] for b in backends]
    assert {"anagram", "usb", "software"}.issubset(ids)

    cfg = CaptureBackendConfig(backend_type="software", send_gain_db=0.0)
    sw = get_capture_backend(cfg)
    assert isinstance(sw, SoftwareDspBackend)
    assert "loopback" in sw.setup_instructions().lower()
    assert len(sw.list_devices()) == 1

    cal_wav = tmp_path / "calibrated_input.wav"
    cal = sw.calibrate_reference_and_hot_corner(
        hot_corner_controls={"drive": 9.0, "attack": "boost"},
        output_dry_wav_path=cal_wav,
        dry_duration_sec=1.6,
    )
    assert cal.send_gain_db == 0.0
    assert cal.calibrated_latency_samples == 48
    assert cal.hot_corner_net_gain_db > 0.0
    assert cal_wav.is_file()

    dry = generate_dry_excitation(duration_sec=1.6, sample_rate_hz=48_000, seed=5)
    wet_out = tmp_path / "stems" / "s0.wav"
    rec = record_single_stem(
        sw,
        model_id="m1",
        stem_id="s0",
        dry_signal=dry,
        output_wet_wav_path=wet_out,
        relative_wet_wav="stems/s0.wav",
        controls_meta={"drive": 6.0},
    )
    assert wet_out.is_file()
    assert rec.entry.latency_offset_samples == 48

    # Also test SoftwareDspBackend backed by a DnamContainer and mono/custom dsp_fn
    ref_c = DnamContainer.reference(core2_active=False)
    sw_engine = SoftwareDspBackend(dnam_container=ref_c, simulated_latency_samples=0)
    ctrl_ids = [c.id for c in ref_c.header.controls_schema]
    sample_ctrls: dict[str, float | str] = {ctrl_ids[0]: 5.0} if ctrl_ids else {"drive": 5.0}
    out_eng = sw_engine.play_and_record(dry[:2048], controls_meta=sample_ctrls)
    assert out_eng.shape == (2, 2048)

    mono_cfg = CaptureBackendConfig(backend_type="software", return_channels=[1])
    sw_mono = SoftwareDspBackend(mono_cfg, simulated_latency_samples=16)
    out_mono = sw_mono.play_and_record(dry[:2048], controls_meta={"drive": 5.0})
    assert out_mono.shape == (2048,)

    sw_custom = SoftwareDspBackend(
        mono_cfg,
        dsp_fn=lambda x, _c, _fs: (x * 0.5).astype(np.float32),
        simulated_latency_samples=8,
    )
    out_cust = sw_custom.play_and_record(dry[:1024], controls_meta={"drive": 3.0})
    assert out_cust.shape == (1024,)


def test_anagram_and_usb_backends_with_mock_sounddevice() -> None:
    fake_sd = _FakeSoundDevice()
    anagram_cfg = CaptureBackendConfig(
        backend_type="anagram",
        send_channel=9,
        return_channels=[1, 2],
        anagram_preset="B7K Ultra",
    )
    anagram = DarkglassAnagramBackend(anagram_cfg, sounddevice_module=fake_sd)
    assert "Darkglass Anagram" in anagram.setup_instructions()
    devs = anagram.list_devices()
    assert len(devs) == 2
    assert anagram._resolve_device_index() == 1

    dry = generate_dry_excitation(duration_sec=1.6, sample_rate_hz=48_000, seed=3)
    rec_st = anagram.play_and_record(dry[:4096])
    assert rec_st.shape == (2, 4096)

    # Mono return channel on USB backend
    usb_cfg = CaptureBackendConfig(
        backend_type="usb",
        device=0,
        send_channel=3,
        return_channels=[1],
    )
    usb = UsbAudioInterfaceBackend(usb_cfg, sounddevice_module=fake_sd)
    assert "Output 3" in usb.setup_instructions()
    rec_mono = usb.play_and_record(dry[:4096])
    assert rec_mono.shape == (4096,)

    register_capture_backend("custom_sw", SoftwareDspBackend)
    with pytest.raises(ValueError, match="non-empty"):
        register_capture_backend("", SoftwareDspBackend)
    bad_kwargs: dict[str, Any] = {"backend_type": "unknown"}
    with pytest.raises(KeyError, match="Unknown capture backend"):
        get_capture_backend(CaptureBackendConfig.model_construct(**bad_kwargs))
