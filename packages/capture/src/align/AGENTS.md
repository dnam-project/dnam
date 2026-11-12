# `packages/capture/src/align/` — 4-Regime Alignment & Loopback Calibration Guide

Read [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md) before editing.

| File | Core Responsibilities | Canonical Specification (SSOT) |
| :--- | :--- | :--- |
| [`../align.rs`](../align.rs) | 5-bit-depth WAV PCM codec (`Int12`..`Float32`), `<f4` SHA-256, and coarse parabolic lag. | [`capture_and_alignment.md`](../../../../docs/specs/training/capture_and_alignment.md) |
| [`regimes.rs`](regimes.rs) | 4-regime sub-sample/clock-skew alignment (`CleanLinear`, `PolarityInverted`, `DispersiveAllPass`, `NonlinearEnvelopeFallback`) and `64`-tap Kaiser-sinc shift. | [`capture_and_alignment.md`](../../../../docs/specs/training/capture_and_alignment.md) |
| [`calibration.rs`](calibration.rs) | Phase-0 `0.5s` `CALB` loopback calibration, `400 ms` mic-pad ratio, `< 0.05 dB` null homing, `<= 0.15 dB` thermal gate, and live decay-tail monitor. | [`capture_and_alignment.md`](../../../../docs/specs/training/capture_and_alignment.md) |
