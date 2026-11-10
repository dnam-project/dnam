# `packages/trainer/` — Pure-Rust Analytical `VarPro` Two-Sweep Trainer Guide

**`dnam-trainer`** (`packages/trainer/`) is the pure-Rust (`#![forbid(unsafe_code)]`) Analytical Variable-Projection (`VarPro`) training crate for **DNAM**, sharing DSP operators (`Op-T`, `Op-M`, `Op-N`) and `.dnam` primitives with [`dnam-engine`](../engine/README.md).

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`README.md`](README.md) before editing.

## Submodule & Documentation Routing Map

| Submodule / Directory | Scoped Guide & Files | Canonical Specification (SSOT) |
| :--- | :--- | :--- |
| **Root, Targets, Resampler & Audit (`src/`)** | [`src/AGENTS.md`](src/AGENTS.md), [`src/lib.rs`](src/lib.rs), [`src/targets.rs`](src/targets.rs), [`src/resampler.rs`](src/resampler.rs), [`src/audit.rs`](src/audit.rs) | [`four_stage_pipeline.md`](../../docs/specs/training/four_stage_pipeline.md), [`control_space_audit.md`](../../docs/specs/training/control_space_audit.md) |
| **Metrics & White-Box (`src/metrics/`, `src/whitebox/`)** | [`src/metrics/AGENTS.md`](src/metrics/AGENTS.md), [`src/whitebox/AGENTS.md`](src/whitebox/AGENTS.md) | [`psychoacoustic_quality_and_tiers.md`](../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md), [`ADR-0018`](../../docs/adr/0018-signal-chain-fixtures-crate-and-whitebox-distillation.md) |
| **Checkpoint & `DNCA` (`src/ckpt/`)** | [`src/ckpt/AGENTS.md`](src/ckpt/AGENTS.md), [`src/ckpt.rs`](src/ckpt.rs) | [`checkpoint_and_cache_format.md`](../../docs/specs/containers/checkpoint_and_cache_format.md), [`iterative_resume_and_caching.md`](../../docs/specs/training/iterative_resume_and_caching.md) |
| **Analytical Solver (`src/solver/`)** | [`src/solver/AGENTS.md`](src/solver/AGENTS.md), [`src/solver.rs`](src/solver.rs) | [`four_stage_pipeline.md`](../../docs/specs/training/four_stage_pipeline.md), [`control_space_audit.md`](../../docs/specs/training/control_space_audit.md) |
| **Integration Tests (`tests/`)** | [`tests/AGENTS.md`](tests/AGENTS.md) | [`psychoacoustic_quality_and_tiers.md`](../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md), [`capture_and_alignment.md`](../../docs/specs/training/capture_and_alignment.md) |
