"""Pluggable hardware and software audio capture backends (`CaptureBackend`,
`DarkglassAnagramBackend`, `UsbAudioInterfaceBackend`, `SoftwareDspBackend`) and
2-step Reference Unity & Hot-Corner Calibration."""

from __future__ import annotations

import importlib
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import ConfigDict

from dnam_engine import DnamContainer, DnamEngine
from dnam_trainer.capture.alignment import (
    align_hardware_latency,
    analyze_hot_corner_nonlinearity,
    measure_silent_window_noise_floor,
)
from dnam_trainer.capture.excitation import (
    analyze_probe_for_excitation,
    generate_dry_excitation,
    read_wav,
    synth_calibration_probe,
    write_wav_24bit,
)
from dnam_trainer.events import (
    CalibrationStepEvent,
    CaptureObserver,
    NullObserver,
    StemCaptureEvent,
)
from dnam_trainer.schema import (
    CaptureBackendConfig,
    ControlCoords,
    ProjectStemEntry,
    StrictModel,
    TargetExcitationProfile,
    coerce_audio_f32,
    compute_pcm_sha256,
)

type SoftwareDspCallable = Callable[[NDArray[np.float32], ControlCoords, int], NDArray[np.float32]]


class AudioDeviceInfo(StrictModel):
    """Metadata for a discovered hardware or virtual audio device."""

    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float


class CalibrationResult(StrictModel):
    """Result of the 2-step Reference Unity & Hot-Corner Calibration workflow.

    Note: `send_gain_db` is strictly preserved across Step 1 and Step 2 without backoff so
    100% of device gain, compression, and `0.0 dBFS` rail clipping are captured intact.
    """

    backend_id: str
    send_gain_db: float
    unity_ref_dbfs: float
    calibrated_latency_samples: int
    baseline_noise_floor_dbfs: float
    hot_corner_peak_dbfs: float
    hot_corner_rms_dbfs: float
    hot_corner_net_gain_db: float
    hot_corner_compression_ratio: float
    hot_corner_clipped_fraction: float
    excitation_profile: TargetExcitationProfile = TargetExcitationProfile()


class CapturedStemRecord(StrictModel):
    """In-memory + on-disk result of capturing and auto-aligning a single stem."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        arbitrary_types_allowed=True,
    )

    entry: ProjectStemEntry
    aligned_audio: NDArray[np.float32]


def _load_sounddevice(override_module: Any | None = None) -> Any:
    if override_module is not None:
        return override_module
    try:
        return importlib.import_module("sounddevice")
    except ImportError as exc:
        raise RuntimeError(
            "Hardware audio capture requires the 'sounddevice' package. "
            "Install it via `uv sync --extra audio`."
        ) from exc


class CaptureBackend(ABC):
    """Abstract base class for hardware and software reamping/capture backends."""

    backend_id: str
    display_name: str
    description: str

    def __init__(self, config: CaptureBackendConfig) -> None:
        self.config = config

    @abstractmethod
    def setup_instructions(self) -> str:
        """Returns human-readable cabling/routing setup instructions for this backend."""

    @abstractmethod
    def list_devices(self) -> list[AudioDeviceInfo]:
        """Lists available audio devices for this backend."""

    @abstractmethod
    def play_and_record(
        self,
        send_signal: ArrayLike,
        *,
        send_gain_db: float | None = None,
        controls_meta: ControlCoords | None = None,
    ) -> NDArray[np.float32]:
        """Plays `send_signal` (scaled by `send_gain_db`) and returns recorded `(L,)` or `(2, L)`
        `float32`."""

    def calibrate_reference_and_hot_corner(
        self,
        *,
        bypass_probe: ArrayLike | None = None,
        hot_corner_probe: ArrayLike | None = None,
        hot_corner_controls: ControlCoords | None = None,
        observer: CaptureObserver | None = None,
        step1_prompt_fn: Callable[[], None] | None = None,
        step2_prompt_fn: Callable[[CalibrationStepEvent], None] | None = None,
        output_dry_wav_path: str | Path | None = None,
        dry_duration_sec: float | None = None,
    ) -> CalibrationResult:
        """Executes the 2-step Reference Unity, Hot-Corner & Probe-Informed Excitation Calibration.

        1. **Step 1 (Bypassed / Unity Reference):** Plays `synth_calibration_probe` at
           `config.send_gain_db` (default `0.0 dB` Unity) to lock `calibrated_latency_samples`,
           `unity_ref_dbfs`, and `baseline_noise_floor_dbfs`.
        2. **Step 2 (Engaged Hot-Corner & Target Excitation Probe):** Plays the diagnostic probe at
           the **exact same locked `send_gain_db`**, measures net boost/cut (`+dB`), compression
           ratio, and clipping telemetry, extracts `TargetExcitationProfile` (`knee`, `saturation`,
           `delay/reverb tail`, `LFO`, `resonance`), and synthesizes the target-optimal `input.wav`
           when `output_dry_wav_path` is provided.
        """
        obs = observer or NullObserver()
        fs = self.config.sample_rate_hz
        locked_send_db = float(self.config.send_gain_db)

        if step1_prompt_fn is not None:
            step1_prompt_fn()

        probe_1 = (
            coerce_audio_f32(bypass_probe, allow_stereo=False)
            if bypass_probe is not None
            else synth_calibration_probe(duration_sec=2.4, sample_rate_hz=fs, seed=101)
        )
        wet_1 = self.play_and_record(probe_1, send_gain_db=locked_send_db, controls_meta={})
        align_1 = align_hardware_latency(probe_1, wet_1, sample_rate_hz=fs)
        noise_1 = measure_silent_window_noise_floor(align_1.aligned_audio, sample_rate_hz=fs)

        dry_rms = max(float(np.sqrt(np.mean(probe_1.astype(np.float64) ** 2))), 1e-9)
        wet1_rms = max(
            float(np.sqrt(np.mean(align_1.aligned_audio.astype(np.float64) ** 2))),
            1e-9,
        )
        wet1_peak = max(
            float(np.max(np.abs(align_1.aligned_audio.astype(np.float64)))),
            1e-9,
        )
        unity_ref_dbfs = float(20.0 * math.log10(wet1_rms / dry_rms))
        step1_peak_dbfs = float(max(-120.0, min(6.0, 20.0 * math.log10(wet1_peak))))
        step1_rms_dbfs = float(max(-120.0, min(6.0, 20.0 * math.log10(wet1_rms))))

        step1_event = CalibrationStepEvent(
            step=1,
            backend_id=self.backend_id,
            send_gain_db=locked_send_db,
            unity_ref_dbfs=unity_ref_dbfs,
            measured_peak_dbfs=step1_peak_dbfs,
            measured_rms_dbfs=step1_rms_dbfs,
            calibrated_latency_samples=align_1.latency_offset_samples,
            noise_floor_dbfs=noise_1,
        )
        obs.on_capture_event(step1_event)

        if step2_prompt_fn is not None:
            step2_prompt_fn(step1_event)

        probe_2 = (
            coerce_audio_f32(hot_corner_probe, allow_stereo=False)
            if hot_corner_probe is not None
            else probe_1
        )
        wet_2 = self.play_and_record(
            probe_2,
            send_gain_db=locked_send_db,
            controls_meta=hot_corner_controls or {"drive": 10.0, "blend": 10.0, "attack": "boost"},
        )
        align_2 = align_hardware_latency(
            probe_2,
            wet_2,
            sample_rate_hz=fs,
            fixed_latency_samples=align_1.latency_offset_samples,
        )
        hot_stats = analyze_hot_corner_nonlinearity(
            probe_2,
            align_2.aligned_audio,
            unity_ref_dbfs=unity_ref_dbfs,
        )
        exc_profile = analyze_probe_for_excitation(
            probe_2,
            align_2.aligned_audio,
            sample_rate_hz=fs,
        )

        if output_dry_wav_path is not None:
            out_p = Path(output_dry_wav_path)
            if dry_duration_sec is not None:
                target_dur = max(1.5, float(dry_duration_sec))
            elif out_p.is_file():
                existing_pcm, existing_fs = read_wav(out_p)
                target_dur = max(1.5, float(existing_pcm.shape[-1]) / float(existing_fs))
            else:
                target_dur = 60.0
            generate_dry_excitation(
                out_p,
                duration_sec=target_dur,
                sample_rate_hz=fs,
                profile=exc_profile,
            )

        obs.on_capture_event(
            CalibrationStepEvent(
                step=2,
                backend_id=self.backend_id,
                send_gain_db=locked_send_db,
                unity_ref_dbfs=unity_ref_dbfs,
                measured_peak_dbfs=hot_stats.measured_return_peak_dbfs,
                measured_rms_dbfs=hot_stats.measured_return_rms_dbfs,
                net_gain_over_unity_db=hot_stats.net_gain_over_unity_db,
                compression_ratio=hot_stats.compression_ratio,
                clipped_sample_fraction=hot_stats.clipped_sample_fraction,
                calibrated_latency_samples=align_1.latency_offset_samples,
                noise_floor_dbfs=noise_1,
                excitation_profile=exc_profile,
            )
        )

        self.config = self.config.model_copy(
            update={
                "send_gain_db": locked_send_db,
                "unity_ref_dbfs": unity_ref_dbfs,
                "calibrated_latency_samples": align_1.latency_offset_samples,
                "baseline_noise_floor_dbfs": noise_1,
                "excitation_profile": exc_profile,
            }
        )

        return CalibrationResult(
            backend_id=self.backend_id,
            send_gain_db=locked_send_db,
            unity_ref_dbfs=unity_ref_dbfs,
            calibrated_latency_samples=align_1.latency_offset_samples,
            baseline_noise_floor_dbfs=noise_1,
            hot_corner_peak_dbfs=hot_stats.measured_return_peak_dbfs,
            hot_corner_rms_dbfs=hot_stats.measured_return_rms_dbfs,
            hot_corner_net_gain_db=hot_stats.net_gain_over_unity_db,
            hot_corner_compression_ratio=hot_stats.compression_ratio,
            hot_corner_clipped_fraction=hot_stats.clipped_sample_fraction,
            excitation_profile=exc_profile,
        )


class UsbAudioInterfaceBackend(CaptureBackend):
    """Multi-channel USB/Thunderbolt audio interface capture backend (`sounddevice`)."""

    backend_id = "usb"
    display_name = "USB / Thunderbolt Audio Interface"
    description = (
        "Configurable multi-channel hardware reamping backend supporting mono or stereo "
        "return channels at 44.1-96 kHz with bypass loopback unity reference calibration."
    )

    def __init__(
        self,
        config: CaptureBackendConfig,
        *,
        sounddevice_module: Any | None = None,
    ) -> None:
        super().__init__(config)
        self._sd_override = sounddevice_module

    def setup_instructions(self) -> str:
        ret_str = ", ".join(str(c) for c in self.config.return_channels)
        sr = self.config.sample_rate_hz
        gain = self.config.send_gain_db
        send_ch = self.config.send_channel
        return (
            f"[{self.display_name}]\n"
            f"  1. Connect Audio Interface Output {send_ch} -> Reamp / Device In.\n"
            f"  2. Connect Device Output(s) -> Audio Interface Input(s) [{ret_str}].\n"
            f"  3. Lock Sample Rate to {sr} Hz (Send Gain = {gain:+.1f} dB)."
        )

    def list_devices(self) -> list[AudioDeviceInfo]:
        sd = _load_sounddevice(self._sd_override)
        raw_devs = sd.query_devices()
        out: list[AudioDeviceInfo] = []
        for idx, dev in enumerate(raw_devs):
            out.append(
                AudioDeviceInfo(
                    index=int(dev.get("index", idx)),
                    name=str(dev.get("name", f"Device {idx}")),
                    max_input_channels=int(dev.get("max_input_channels", 0)),
                    max_output_channels=int(dev.get("max_output_channels", 0)),
                    default_samplerate=float(dev.get("default_samplerate", 48_000.0)),
                )
            )
        return out

    def _resolve_device_index(self) -> int | str | None:
        if self.config.device is not None:
            return self.config.device
        return None

    def play_and_record(
        self,
        send_signal: ArrayLike,
        *,
        send_gain_db: float | None = None,
        controls_meta: ControlCoords | None = None,
    ) -> NDArray[np.float32]:
        _ = controls_meta
        sd = _load_sounddevice(self._sd_override)
        dry = coerce_audio_f32(send_signal, allow_stereo=False)
        gain_db = self.config.send_gain_db if send_gain_db is None else float(send_gain_db)
        gain_lin = float(10.0 ** (gain_db / 20.0))
        scaled_send = (dry * gain_lin).astype(np.float32)

        send_ch = max(1, self.config.send_channel)
        ret_chs = [max(1, c) for c in self.config.return_channels]
        max_out_ch = send_ch
        max_in_ch = max(ret_chs)

        out_buf = np.zeros((dry.shape[0], max_out_ch), dtype=np.float32)
        out_buf[:, send_ch - 1] = scaled_send

        dev_arg = self._resolve_device_index()
        recorded = sd.playrec(
            out_buf,
            samplerate=self.config.sample_rate_hz,
            input_mapping=ret_chs,
            device=dev_arg,
            dtype="float32",
            blocking=True,
        )
        rec_arr = np.asarray(recorded, dtype=np.float32)
        if rec_arr.ndim == 1:
            return np.ascontiguousarray(rec_arr, dtype=np.float32)
        if rec_arr.shape[1] == 1:
            return np.ascontiguousarray(rec_arr[:, 0], dtype=np.float32)
        if rec_arr.shape[1] >= 2 and len(ret_chs) == 2:
            if rec_arr.shape[1] == max_in_ch and max_in_ch > 2:
                selected = rec_arr[:, [c - 1 for c in ret_chs]]
            else:
                selected = rec_arr[:, :2]
            return np.ascontiguousarray(selected.T, dtype=np.float32)
        return np.ascontiguousarray(rec_arr[:, 0], dtype=np.float32)


class DarkglassAnagramBackend(UsbAudioInterfaceBackend):
    """Pre-configured Darkglass Anagram 48 kHz 24-bit USB-C Reamp Capture Backend (`anagram`).

    Routes USB Output 9 (`send_channel=9`) -> Input 1 (`[1]`) or True-Stereo Inputs 1 & 2
    (`[1, 2]`) at Unity `send_gain_db = 0.0 dB`, preserving 100% of device gain, compression,
    and clipping without send-gain backoff.
    """

    backend_id = "anagram"
    display_name = "Darkglass Anagram (USB-C Digital Reamp)"
    description = (
        "Darkglass Anagram 48 kHz 24-bit USB-C reamping: USB Out 9 Reamp Send -> "
        "In 1 (Mono) or In 1 & 2 (True-Stereo Return) at 0.0 dB Unity."
    )

    def setup_instructions(self) -> str:
        ret_str = (
            "Inputs 1 & 2 (True Stereo)"
            if len(self.config.return_channels) == 2
            else "Input 1 (Mono)"
        )
        preset_note = (
            f" | Preset: {self.config.anagram_preset}" if self.config.anagram_preset else ""
        )
        send_ch = self.config.send_channel
        gain = self.config.send_gain_db
        return (
            f"[{self.display_name}{preset_note}]\n"
            f"  1. Connect Darkglass Anagram via USB-C (48 kHz / 24-bit).\n"
            f"  2. Route USB Output {send_ch} (Reamp Send) -> Anagram Block Chain Input.\n"
            f"  3. Capture Return on USB {ret_str} at Unity Send Gain ({gain:+.1f} dB).\n"
            f"  4. Step-1 measures Bypass Unity (0.0 dBFS); Step-2 preserves engaged gain."
        )

    def _resolve_device_index(self) -> int | str | None:
        if self.config.device is not None:
            return self.config.device
        for dev in self.list_devices():
            if "anagram" in dev.name.lower() or "darkglass" in dev.name.lower():
                return dev.index
        return None


def _default_software_dsp(
    dry: NDArray[np.float32],
    controls: ControlCoords,
    sample_rate_hz: int,
) -> NDArray[np.float32]:
    """Default deterministic Tri-Physics software synth callable for hardware-free testing."""
    _ = sample_rate_hz
    drive_val = float(controls.get("drive", controls.get("gain", 5.0)))
    blend_val = float(controls.get("blend", controls.get("mix", 7.0)))
    mode_val = str(controls.get("attack", controls.get("mode", "flat")))

    drive_gain = 1.0 + 0.35 * max(0.0, drive_val)
    blend_alpha = min(1.0, max(0.0, blend_val / 10.0)) if blend_val > 1.0 else float(blend_val)
    mode_boost = (
        1.25 if mode_val in ("boost", "high", "ultra") else (0.80 if mode_val == "cut" else 1.0)
    )

    x = dry.astype(np.float64) * drive_gain * mode_boost
    # Smooth cubic soft-clipper
    u = np.clip((2.0 / 3.0) * x, -1.0, 1.0)
    nl = 1.5 * u - 0.5 * (u**3)
    wet_mono = ((1.0 - blend_alpha) * dry.astype(np.float64) + blend_alpha * nl).astype(np.float32)

    # Slight stereo decorrelation on R channel
    wet_r = np.roll(wet_mono, 3)
    wet_r[:3] = 0.0
    right_ch = (0.96 * wet_r + 0.04 * wet_mono).astype(np.float32)
    return np.stack([wet_mono, right_ch], axis=0)


class SoftwareDspBackend(CaptureBackend):
    """Deterministic software capture backend wrapping a Python callable or `DnamEngine`."""

    backend_id = "software"
    display_name = "Software DSP / DnamEngine Loopback"
    description = (
        "Deterministic in-memory software reamping backend for hardware-free capture, "
        "teacher-student distillation, and automated CI testing."
    )

    def __init__(
        self,
        config: CaptureBackendConfig | None = None,
        *,
        dsp_fn: SoftwareDspCallable | None = None,
        dnam_container: DnamContainer | None = None,
        simulated_latency_samples: int = 48,
        simulated_noise_dbfs: float = -96.0,
    ) -> None:
        cfg = config or CaptureBackendConfig(
            backend_type="software",
            send_channel=1,
            return_channels=[1, 2],
            send_gain_db=0.0,
            unity_ref_dbfs=0.0,
        )
        super().__init__(cfg)
        self._dsp_fn = dsp_fn
        self._container = dnam_container
        self._sim_latency = max(0, simulated_latency_samples)
        self._sim_noise_dbfs = float(simulated_noise_dbfs)

    def setup_instructions(self) -> str:
        gain = self.config.send_gain_db
        return (
            f"[{self.display_name}]\n"
            f"  In-memory deterministic loopback @ {self.config.sample_rate_hz} Hz "
            f"(Latency = {self._sim_latency} samples, Unity Send = {gain:+.1f} dB)."
        )

    def list_devices(self) -> list[AudioDeviceInfo]:
        return [
            AudioDeviceInfo(
                index=0,
                name="DNAM Virtual Software Loopback (Stereo 48-96 kHz)",
                max_input_channels=2,
                max_output_channels=2,
                default_samplerate=float(self.config.sample_rate_hz),
            )
        ]

    def play_and_record(
        self,
        send_signal: ArrayLike,
        *,
        send_gain_db: float | None = None,
        controls_meta: ControlCoords | None = None,
    ) -> NDArray[np.float32]:
        dry = coerce_audio_f32(send_signal, allow_stereo=False)
        gain_db = self.config.send_gain_db if send_gain_db is None else float(send_gain_db)
        gain_lin = float(10.0 ** (gain_db / 20.0))
        scaled = (dry * gain_lin).astype(np.float32)
        ctrls = dict(controls_meta or {})

        if self._container is not None:
            engine = DnamEngine(
                self._container,
                sample_rate=float(self.config.sample_rate_hz),
            )
            known_ids = {c.id for c in self._container.header.controls_schema}
            valid_ctrls = {k: v for k, v in ctrls.items() if k in known_ids}
            if valid_ctrls:
                engine.controls.snap(valid_ctrls)
            wet = engine(scaled)
        elif self._dsp_fn is not None:
            wet = coerce_audio_f32(
                self._dsp_fn(scaled, ctrls, self.config.sample_rate_hz),
                allow_stereo=True,
            )
        elif not ctrls:
            # Step-1 bypass unity probe when no controls are active
            wet = (
                np.stack([scaled, scaled], axis=0)
                if len(self.config.return_channels) == 2
                else scaled.copy()
            )
        else:
            wet = _default_software_dsp(scaled, ctrls, self.config.sample_rate_hz)
            if len(self.config.return_channels) == 1 and wet.ndim == 2:
                wet = wet[0]

        # Apply simulated hardware latency offset
        n_samples = int(dry.shape[0])
        if self._sim_latency > 0:
            if wet.ndim == 1:
                padded = np.zeros(n_samples + self._sim_latency, dtype=np.float32)
                padded[self._sim_latency : self._sim_latency + wet.shape[0]] = wet
                wet = padded[:n_samples]
            else:
                padded2 = np.zeros((2, n_samples + self._sim_latency), dtype=np.float32)
                padded2[:, self._sim_latency : self._sim_latency + wet.shape[1]] = wet
                wet = padded2[:, :n_samples]

        # Add tiny deterministic noise floor in the silent window only if above -118 dBFS
        if self._sim_noise_dbfs > -118.0:
            rng = np.random.default_rng(12345)
            noise_std = float(10.0 ** (self._sim_noise_dbfs / 20.0))
            wet = (wet + rng.standard_normal(wet.shape).astype(np.float32) * noise_std).astype(
                np.float32
            )

        return np.ascontiguousarray(wet, dtype=np.float32)


_BACKEND_REGISTRY: dict[str, type[CaptureBackend]] = {
    "anagram": DarkglassAnagramBackend,
    "usb": UsbAudioInterfaceBackend,
    "software": SoftwareDspBackend,
}


def register_capture_backend(backend_id: str, backend_cls: type[CaptureBackend]) -> None:
    """Registers a custom `CaptureBackend` subclass."""
    if not backend_id:
        raise ValueError("backend_id must be non-empty")
    _BACKEND_REGISTRY[backend_id] = backend_cls


def list_capture_backends() -> list[tuple[str, str, str]]:
    """Returns `(backend_id, display_name, description)` for all registered capture backends."""
    return [
        (b_id, cls.display_name, cls.description) for b_id, cls in sorted(_BACKEND_REGISTRY.items())
    ]


def get_capture_backend(
    config: CaptureBackendConfig,
    **kwargs: Any,
) -> CaptureBackend:
    """Instantiates the registered `CaptureBackend` for `config.backend_type`."""
    cls = _BACKEND_REGISTRY.get(config.backend_type)
    if cls is None:
        raise KeyError(
            f"Unknown capture backend '{config.backend_type}'. "
            f"Available: {sorted(_BACKEND_REGISTRY.keys())}"
        )
    return cls(config, **kwargs)


def record_single_stem(
    backend: CaptureBackend,
    *,
    model_id: str,
    stem_id: str,
    dry_signal: ArrayLike,
    output_wet_wav_path: str | Path,
    relative_wet_wav: str,
    controls_meta: ControlCoords | None = None,
    stem_index: int = 1,
    total_stems: int = 1,
    observer: CaptureObserver | None = None,
) -> CapturedStemRecord:
    """Plays `dry_signal` through `backend`, auto-aligns latency, measures telemetry, and saves
    WAV."""
    obs = observer or NullObserver()
    ctrls = dict(controls_meta or {})
    dry_f32 = coerce_audio_f32(dry_signal, allow_stereo=False)
    fs = backend.config.sample_rate_hz

    obs.on_capture_event(
        StemCaptureEvent(
            status="starting",
            model_id=model_id,
            stem_id=stem_id,
            stem_index=stem_index,
            total_stems=total_stems,
            controls=ctrls,
        )
    )

    raw_wet = backend.play_and_record(
        dry_f32,
        send_gain_db=backend.config.send_gain_db,
        controls_meta=ctrls,
    )
    cal_lat = backend.config.calibrated_latency_samples
    fixed_lat = cal_lat if cal_lat is not None and cal_lat > 0 else None
    aligned_res = align_hardware_latency(
        dry_f32,
        raw_wet,
        sample_rate_hz=fs,
        fixed_latency_samples=fixed_lat,
    )
    aligned_wet = aligned_res.aligned_audio

    # Scale relative to calibrated unity_ref_dbfs if non-zero while preserving clipping telemetry
    hot_stats = analyze_hot_corner_nonlinearity(
        dry_f32,
        aligned_wet,
        unity_ref_dbfs=backend.config.unity_ref_dbfs,
    )
    cal_noise = backend.config.baseline_noise_floor_dbfs
    base_noise = cal_noise if cal_noise is not None and cal_noise < -20.0 else None
    noise_db = measure_silent_window_noise_floor(
        aligned_wet,
        sample_rate_hz=fs,
        baseline_noise_floor_dbfs=base_noise,
    )

    write_wav_24bit(output_wet_wav_path, aligned_wet, sample_rate_hz=fs)

    entry = ProjectStemEntry(
        stem_id=stem_id,
        wet_wav=relative_wet_wav,
        controls=ctrls,
        send_gain_db=backend.config.send_gain_db,
        unity_ref_dbfs=backend.config.unity_ref_dbfs,
        measured_return_peak_dbfs=hot_stats.measured_return_peak_dbfs,
        measured_return_rms_dbfs=hot_stats.measured_return_rms_dbfs,
        net_gain_over_unity_db=hot_stats.net_gain_over_unity_db,
        noise_floor_dbfs=noise_db,
        clipped_sample_fraction=hot_stats.clipped_sample_fraction,
        latency_offset_samples=aligned_res.latency_offset_samples,
        input_pcm_sha256=compute_pcm_sha256(dry_f32),
        target_pcm_sha256=compute_pcm_sha256(aligned_wet),
        captured_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    obs.on_capture_event(
        StemCaptureEvent(
            status="completed",
            model_id=model_id,
            stem_id=stem_id,
            stem_index=stem_index,
            total_stems=total_stems,
            controls=ctrls,
            measured_peak_dbfs=entry.measured_return_peak_dbfs,
            noise_floor_dbfs=entry.noise_floor_dbfs,
            clipped_sample_fraction=entry.clipped_sample_fraction,
            latency_offset_samples=entry.latency_offset_samples,
        )
    )

    return CapturedStemRecord(entry=entry, aligned_audio=aligned_wet)


__all__ = [
    "AudioDeviceInfo",
    "CalibrationResult",
    "CaptureBackend",
    "CapturedStemRecord",
    "DarkglassAnagramBackend",
    "SoftwareDspBackend",
    "SoftwareDspCallable",
    "UsbAudioInterfaceBackend",
    "get_capture_backend",
    "list_capture_backends",
    "record_single_stem",
    "register_capture_backend",
]
