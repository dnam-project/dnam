# `dnam_trainer/capture/` — Pluggable Hardware & Software Audio Capture Pipeline

**`dnam_trainer.capture`** implements dry system-identification excitation synthesis, 24-bit PCM WAV I/O, 2-stage coarse-to-fine hardware latency alignment, 2-step Reference Unity & Hot-Corner Calibration, and extensible `CaptureBackend` drivers (`DarkglassAnagramBackend`, `UsbAudioInterfaceBackend`, `SoftwareDspBackend`).

## 1. Module Routing Map

| Module | Key Exports & Responsibilities | Canonical Documentation (SSOT) |
| :--- | :--- | :--- |
| [`excitation.py`](excitation.py) | `read_wav`, `write_wav_24bit`, `synth_alignment_pulses` (3-pulse asymmetric burst + silent settling window), `generate_dry_excitation` | [`training_and_iterative_resume.md`](../../../../docs/training_and_iterative_resume.md) |
| [`alignment.py`](alignment.py) | `align_hardware_latency` (2-stage FFT envelope + fine time-domain correlation), `measure_silent_window_noise_floor`, `analyze_hot_corner_nonlinearity` | [`training_and_iterative_resume.md`](../../../../docs/training_and_iterative_resume.md) |
| [`backends.py`](backends.py) | `CaptureBackend`, `DarkglassAnagramBackend` (`"anagram"`), `UsbAudioInterfaceBackend` (`"usb"`), `SoftwareDspBackend` (`"software"`), `record_model_stem`, `record_planned_stems` | [`training_and_iterative_resume.md`](../../../../docs/training_and_iterative_resume.md) |

## 2. Invariants

- **Zero `torch` Imports:** Never import `torch` in `dnam_trainer.capture` so CLI discovery and capture commands start in `< 20 ms`.
- **Unity Reference Calibration (`0.0 dB` Send):** Never reduce `send_gain_db` when a device compresses or clips during Step-2 hot-corner verification; preserve 100% of device gain, compression, and clipping relative to Step-1 `unity_ref_dbfs`.
