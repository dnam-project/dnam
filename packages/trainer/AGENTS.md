# `packages/trainer/` — Pure-Rust Analytical `VarPro` Two-Sweep Trainer Guide

**`dnam-trainer`** (`packages/trainer/`) is the 100% pure-Rust (`#![forbid(unsafe_code)]`) Analytical Variable-Projection (`VarPro`) training crate for **DNAM**, sharing DSP operators (`Op-T`, `Op-M`, `Op-N`) and `.dnam` container primitives with [`dnam-engine`](../engine/README.md).

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
| **Crate Root, Targets & Audit (`src/`)** | [`src/AGENTS.md`](src/AGENTS.md), [`src/lib.rs`](src/lib.rs), [`src/targets.rs`](src/targets.rs), [`src/audit.rs`](src/audit.rs) | [`four_stage_pipeline.md`](../../docs/specs/training/four_stage_pipeline.md), [`control_space_audit.md`](../../docs/specs/training/control_space_audit.md) |
| **Checkpoint & `DNCA` (`src/ckpt/`)** | [`src/ckpt/AGENTS.md`](src/ckpt/AGENTS.md), [`src/ckpt.rs`](src/ckpt.rs) | [`checkpoint_and_cache_format.md`](../../docs/specs/containers/checkpoint_and_cache_format.md), [`iterative_resume_and_caching.md`](../../docs/specs/training/iterative_resume_and_caching.md) |
| **Analytical Solver (`src/solver/`)** | [`src/solver/AGENTS.md`](src/solver/AGENTS.md), [`src/solver.rs`](src/solver.rs) | [`four_stage_pipeline.md`](../../docs/specs/training/four_stage_pipeline.md), [`control_space_audit.md`](../../docs/specs/training/control_space_audit.md) |
| **Integration Tests (`tests/`)** | [`tests/AGENTS.md`](tests/AGENTS.md), [`tests/trainer_spec_verification.rs`](tests/trainer_spec_verification.rs) | [`psychoacoustic_quality_and_tiers.md`](../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md), [`capture_and_alignment.md`](../../docs/specs/training/capture_and_alignment.md) |
