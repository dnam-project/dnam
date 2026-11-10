# `packages/trainer/src/` — Crate Root, CLI & Submodules Guide

Read [`../CONTRIBUTING.md`](../CONTRIBUTING.md) before editing.

| File / Submodule | Core Responsibilities | Canonical Specification (SSOT) |
| :--- | :--- | :--- |
| [`lib.rs`](lib.rs) | `TrainerCli` (`check`, `dnwb-serve`), `DNAM-GRAM-INPUT-v1`, and RNE `fp16` helpers. | [`checkpoint_and_cache_format.md`](../../../docs/specs/containers/checkpoint_and_cache_format.md) |
| [`ckpt.rs`](ckpt.rs) & [`ckpt/`](ckpt/AGENTS.md) | `.dnam-ckpt` (`DNCK`) view/builder, `DCA1`, `Arena 4` (`uplo='U'`), `Arena 5` (`DNCA`), and `.dnam` export. | [`checkpoint_and_cache_format.md`](../../../docs/specs/containers/checkpoint_and_cache_format.md) |
| [`audit.rs`](audit.rs) | 3-Phase Control-Space Audit (`audit_phase1_clamp_range`, `audit_phase1_clamp_switch`, `verify_dynamic_stability_excursion`). | [`control_space_audit.md`](../../../docs/specs/training/control_space_audit.md) |
| [`metrics.rs`](metrics.rs) & [`metrics/`](metrics/AGENTS.md) | 4-metric perceptual suite (`bESR`, 5-scale `MR-IIR`, 33-band `LTAS`, `SegESR`) and 4-tier non-regression. | [`psychoacoustic_quality_and_tiers.md`](../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`targets.rs`](targets.rs) | `DistillationTarget`, `CaptureBitDepth` (`Int12`..`Int32`, `Float32`), `StemStimulus`, and `StemResponse`. | [`four_stage_pipeline.md`](../../../docs/specs/training/four_stage_pipeline.md) |
| [`resampler.rs`](resampler.rs) | Zero-phase Kaiser-sinc `8..192 kHz <-> 48 kHz` audio and trajectory resampler. | [`capture_and_alignment.md`](../../../docs/specs/training/capture_and_alignment.md) |
| [`whitebox.rs`](whitebox.rs) & [`whitebox/`](whitebox/AGENTS.md) | `64 B` `DnwbHeader64B` pipe IPC, `FaustRunner`, and `SpiceRunner`. | [`ADR-0018`](../../../docs/adr/0018-signal-chain-fixtures-crate-and-whitebox-distillation.md) |
| [`bin/dnam_train.rs`](bin/dnam_train.rs) | Thin CLI binary entrypoint delegating to `run_cli_from_args`. | [`four_stage_pipeline.md`](../../../docs/specs/training/four_stage_pipeline.md) |
| [`solver.rs`](solver.rs) & [`solver/`](solver/AGENTS.md) | `uplo='U'` Gram accumulation, Cholesky solver, degree-7 Bézout crossover, `bESR`, and 3-Phase Audit bounds. | [`four_stage_pipeline.md`](../../../docs/specs/training/four_stage_pipeline.md) |
