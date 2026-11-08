# `packages/trainer/src/` — Crate Root, CLI & Container Primitives Guide

**`packages/trainer/src/`** contains the `#![forbid(unsafe_code)]` library root ([`lib.rs`](lib.rs)), the `dnam-train` CLI binary ([`bin/dnam_train.rs`](bin/dnam_train.rs)), and the analytical Two-Sweep solver modules ([`solver.rs`](solver.rs), [`solver/AGENTS.md`](solver/AGENTS.md)).

## 1. Module Responsibilities

| File / Submodule | Core Responsibilities | Canonical Specification (SSOT) |
| :--- | :--- | :--- |
| [`lib.rs`](lib.rs) | `TrainerCli` (`clap`), `run_cli_from_args`, `DNAM-GRAM-INPUT-v1` SHA-256 fingerprint (`compute_gram_input_fingerprint`), and `fp16` coreset conversion (`f32_to_fp16_bits`, `fp16_bits_to_f32`). | [`checkpoint_and_cache_format.md`](../../../docs/specs/containers/checkpoint_and_cache_format.md), [`iterative_resume_and_caching.md`](../../../docs/specs/training/iterative_resume_and_caching.md) |
| [`bin/dnam_train.rs`](bin/dnam_train.rs) | Thin CLI binary entrypoint delegating to `dnam_trainer::run_cli_from_args`. | [`four_stage_pipeline.md`](../../../docs/specs/training/four_stage_pipeline.md) |
| [`solver.rs`](solver.rs) & [`solver/`](solver/AGENTS.md) | Column-major `uplo='U'` Gram accumulation, regularized Cholesky solver, Bézout crossover weights, Noise-Subtracted `B-ESR`, and 3-Phase Control-Space Audit bounds. | [`four_stage_pipeline.md`](../../../docs/specs/training/four_stage_pipeline.md), [`control_space_audit.md`](../../../docs/specs/training/control_space_audit.md) |

## 2. Invariants

- **100% Safe Rust (`#![forbid(unsafe_code)]`):** Zero `unsafe` blocks or `#[allow(unsafe_code)]`.
- **Engine Isomorphism:** Never duplicate real-time DSP recurrences (`Op-T`, `Op-M`, `Op-N`) or `.dnam` wire framing; import them from [`dnam-engine`](../../engine/README.md).
- **Zero `mod.rs`:** Follow Rust 2024 module conventions (`solver.rs` + `solver/stages.rs`).
