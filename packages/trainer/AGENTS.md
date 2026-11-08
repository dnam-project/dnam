# `packages/trainer/` — Pure-Rust Analytical `VarPro` Two-Sweep Trainer Guide

**`dnam-trainer`** (`packages/trainer/`) is the 100% pure-Rust (`#![forbid(unsafe_code)]`) Analytical Variable-Projection (`VarPro`) Two-Sweep Block-Coordinate Cholesky training crate for **DNAM**, sharing the exact `#![no_std]` + `alloc` DSP operators (`Op-T`, `Op-M`, `Op-N`), `3 kHz` Parametric Control Bus, `uplo='U'` Gram packer, and `.dnam`/`.dnam-ckpt` container format with [`dnam-engine`](../engine/README.md).

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`README.md`](README.md) before editing.

## 1. Commands (`moon`)

```bash
moon check trainer            # Run format-check, lint (clippy), typecheck, build, and test
moon run trainer:test         # Run Rust unit, proptest, and integration suites
moon run trainer:coverage     # Enforce >= 95% line coverage via cargo llvm-cov
moon run trainer:bench        # Run Criterion solver throughput benchmarks
moon run trainer:cli -- --help
```

## 2. Submodule & Documentation Routing Map

| Submodule / Directory | Scoped Guide & Files | Canonical Specification (SSOT) |
| :--- | :--- | :--- |
| **Crate Root & CLI (`src/`)** | [`src/AGENTS.md`](src/AGENTS.md), [`src/lib.rs`](src/lib.rs), [`src/bin/dnam_train.rs`](src/bin/dnam_train.rs) | [`four_stage_pipeline.md`](../../docs/specs/training/four_stage_pipeline.md), [`iterative_resume_and_caching.md`](../../docs/specs/training/iterative_resume_and_caching.md) |
| **Analytical Solver & Stages (`src/solver/`)** | [`src/solver/AGENTS.md`](src/solver/AGENTS.md), [`src/solver.rs`](src/solver.rs), [`src/solver/stages.rs`](src/solver/stages.rs) | [`control_space_audit.md`](../../docs/specs/training/control_space_audit.md), [`checkpoint_and_cache_format.md`](../../docs/specs/containers/checkpoint_and_cache_format.md) |
| **Integration & Spec Tests (`tests/`)** | [`tests/AGENTS.md`](tests/AGENTS.md), [`tests/trainer_spec_verification.rs`](tests/trainer_spec_verification.rs) | [`psychoacoustic_quality_and_tiers.md`](../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md), [`capture_and_alignment.md`](../../docs/specs/training/capture_and_alignment.md) |
