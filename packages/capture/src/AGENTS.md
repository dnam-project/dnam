# `packages/capture/src/` — Crate Root & Submodules Guide

Read [`../CONTRIBUTING.md`](../CONTRIBUTING.md) before editing.

| File / Submodule | Core Responsibilities | Canonical Spec |
| :--- | :--- | :--- |
| [`lib.rs`](lib.rs) | Pure IO-buffer root, `DNAM-GRAM-INPUT-v2`, and RNE `fp16` helpers. | [`project_container_format.md`](../../../docs/specs/containers/project_container_format.md) |
| [`align.rs`](align.rs) & [`align/`](align/AGENTS.md) | WAV codec, `CALB`, 4-regime Kaiser-sinc alignment, and null/pad gates. | [`capture_and_alignment.md`](../../../docs/specs/training/capture_and_alignment.md) |
| [`stimulus.rs`](stimulus.rs) & [`stimulus/`](stimulus/AGENTS.md) | `AdaptiveStimulusController` (`Laws 0–3`) and Mode-1/2 distillation. | [`capture_and_alignment.md`](../../../docs/specs/training/capture_and_alignment.md) |
| [`proj.rs`](proj.rs) & [`proj/`](proj/AGENTS.md) | `.dnam-proj` (`DNPJ`, `v2.0`) container, capsules, vault, and `.dnam` export. | [`project_container_format.md`](../../../docs/specs/containers/project_container_format.md) |
| [`audit.rs`](audit.rs) | 3-Phase Control-Space Audit (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]`). | [`control_space_audit.md`](../../../docs/specs/training/control_space_audit.md) |
| [`metrics.rs`](metrics.rs) & [`metrics/`](metrics/AGENTS.md) | `bESR`, 5-scale `MR-IIR`, 33-band `LTAS`, `SegESR`, and 4-tier matrix. | [`psychoacoustic_quality_and_tiers.md`](../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`targets.rs`](targets.rs) & [`resampler.rs`](resampler.rs) | `DistillationTarget`, `CaptureBitDepth`, and Kaiser-sinc `8..192 kHz` resampler. | [`capture_and_alignment.md`](../../../docs/specs/training/capture_and_alignment.md) |
| [`whitebox.rs`](whitebox.rs) & [`whitebox/`](whitebox/AGENTS.md) | `64 B` `DnwbHeader64B` IPC, `FaustRunner`, and `SpiceRunner`. | [`ADR-0018`](../../../docs/adr/0018-signal-chain-fixtures-crate-and-whitebox-distillation.md) |
| [`solver.rs`](solver.rs) & [`solver/`](solver/AGENTS.md) | `uplo='U'` Gram Cholesky solver, Bézout crossover, and audit bounds. | [`four_stage_pipeline.md`](../../../docs/specs/training/four_stage_pipeline.md) |
