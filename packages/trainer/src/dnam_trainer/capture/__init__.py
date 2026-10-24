"""Pluggable hardware and software audio capture pipeline (`dnam_trainer.capture`)."""

from __future__ import annotations

from dnam_trainer.capture.alignment import (
    HotCornerAnalysis,
    LatencyAlignmentResult,
    align_hardware_latency,
    analyze_hot_corner_nonlinearity,
    measure_silent_window_noise_floor,
)
from dnam_trainer.capture.backends import (
    AudioDeviceInfo,
    CalibrationResult,
    CaptureBackend,
    CapturedStemRecord,
    DarkglassAnagramBackend,
    SoftwareDspBackend,
    SoftwareDspCallable,
    UsbAudioInterfaceBackend,
    get_capture_backend,
    list_capture_backends,
    record_single_stem,
    register_capture_backend,
)
from dnam_trainer.capture.excitation import (
    ALIGNMENT_HEADER_TOTAL_SEC,
    ALIGNMENT_PULSE_DURATION_SEC,
    ALIGNMENT_SETTLE_SILENCE_SEC,
    TRAILING_NOISE_WINDOW_SEC,
    generate_dry_excitation,
    read_wav,
    synth_alignment_pulses,
    write_wav_24bit,
)

__all__ = [
    "ALIGNMENT_HEADER_TOTAL_SEC",
    "ALIGNMENT_PULSE_DURATION_SEC",
    "ALIGNMENT_SETTLE_SILENCE_SEC",
    "TRAILING_NOISE_WINDOW_SEC",
    "AudioDeviceInfo",
    "CalibrationResult",
    "CaptureBackend",
    "CapturedStemRecord",
    "DarkglassAnagramBackend",
    "HotCornerAnalysis",
    "LatencyAlignmentResult",
    "SoftwareDspBackend",
    "SoftwareDspCallable",
    "UsbAudioInterfaceBackend",
    "align_hardware_latency",
    "analyze_hot_corner_nonlinearity",
    "generate_dry_excitation",
    "get_capture_backend",
    "list_capture_backends",
    "measure_silent_window_noise_floor",
    "read_wav",
    "record_single_stem",
    "register_capture_backend",
    "synth_alignment_pulses",
    "write_wav_24bit",
]
