# `packages/capture/src/stimulus/` — Closed-Loop Excitation & Dual-Mode Pipeline Guide

Read [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md) before editing.

| File | Core Responsibilities | Canonical Specification (SSOT) |
| :--- | :--- | :--- |
| [`../stimulus.rs`](../stimulus.rs) | `AdaptiveStimulusController` (`next_frame` `< 1.5 ms`), telemetry, and `INPC`/`STMC` sealing. | [`capture_and_alignment.md`](../../../../docs/specs/training/capture_and_alignment.md) |
| [`fft.rs`](fft.rs) | Radix-2 `f64` FFT/IFFT, disjoint-comb scheduler, Schroeder phases (`<= 4.5 dB`), `512 + 4,096` cyclic frames. | [`capture_and_alignment.md`](../../../../docs/specs/training/capture_and_alignment.md) |
| [`waveforms.rs`](waveforms.rs) | `A1..A6` primitives, `Theorem 4.7` active primitives, `Theorem 4.8` Bloom telemetry, `fp16`/PSD encoders. | [`capture_and_alignment.md`](../../../../docs/specs/training/capture_and_alignment.md) |
| [`safeguards.rs`](safeguards.rs) | `Laws 0.A–0.F` (`LTV` LFO/`EnvelopeSvf`, echo hold, squeal, USB dropout, `-1.5 dBFS` governor, `OC-2`, capsule clip). | [`capture_and_alignment.md`](../../../../docs/specs/training/capture_and_alignment.md) |
| [`laws.rs`](laws.rs) | `Laws 1–3`, `Theorem 4.5` multi-port disambiguation, `Theorem 4.6` endpoint probing & Hadamard diagonals, 4-Gate stop. | [`capture_and_alignment.md`](../../../../docs/specs/training/capture_and_alignment.md) |
| [`pipeline.rs`](pipeline.rs) | Mode 1 closed-loop streaming (`train_adaptive_target`), endpoint probing, `FreezeLatch` transition capture. | [`capture_and_alignment.md`](../../../../docs/specs/training/capture_and_alignment.md) |
| [`null_test.rs`](null_test.rs) | `1.0s` Out-of-Sample Live Null-Test and `< 5 ms` sufficient-statistic `STAT` auto-patch. | [`capture_and_alignment.md`](../../../../docs/specs/training/capture_and_alignment.md) |
| [`sealing.rs`](sealing.rs) | `INPC`/`STMC` coreset sealing, aperiodic stem alignment, `FreezeLatch` synthesis, `recapture_clipped_stem`. | [`capture_and_alignment.md`](../../../../docs/specs/training/capture_and_alignment.md) |
| [`batch.rs`](batch.rs) | Mode 2 1-or-2-pass sans-IO batch (`generate_stimulus_batch`, `generate_refinement_stimulus_batch`, `solve_from_responses`). | [`capture_and_alignment.md`](../../../../docs/specs/training/capture_and_alignment.md) |
