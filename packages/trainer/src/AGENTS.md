# `packages/trainer/src/` — Crate Root, CLI & Container Primitives Guide

**`packages/trainer/src/`** contains the `#![forbid(unsafe_code)]` library root ([`lib.rs`](lib.rs)), the `dnam-train` CLI ([`bin/dnam_train.rs`](bin/dnam_train.rs)), `.dnam-ckpt` framing ([`ckpt/`](ckpt/AGENTS.md)), and the analytical Two-Sweep solver ([`solver/`](solver/AGENTS.md)).

## 1. Module Responsibilities

| File / Submodule | Core Responsibilities | Canonical Specification (SSOT) |
| :--- | :--- | :--- |
| [`lib.rs`](lib.rs) | `TrainerCli` (`clap`), `run_cli_from_args`, `DNAM-GRAM-INPUT-v1` fingerprint, and `fp16` coreset helpers. | [`checkpoint_and_cache_format.md`](../../../docs/specs/containers/checkpoint_and_cache_format.md), [`iterative_resume_and_caching.md`](../../../docs/specs/training/iterative_resume_and_caching.md) |
| [`ckpt.rs`](ckpt.rs) & [`ckpt/`](ckpt/AGENTS.md) | `.dnam-ckpt` (`DNCK`) view/builder, `DCA1`, `Arena 4` (`uplo='U'`), `Arena 5` (`DNCA`), and `.dnam` export. | [`checkpoint_and_cache_format.md`](../../../docs/specs/containers/checkpoint_and_cache_format.md) |
| [`audit.rs`](audit.rs) | 3-Phase Control-Space Audit helpers (`audit_phase1_clamp_range`, `audit_phase1_clamp_switch`, `verify_dynamic_stability_excursion`). | [`control_space_audit.md`](../../../docs/specs/training/control_space_audit.md) |
| [`targets.rs`](targets.rs) | `DistillationTarget` (`render_stem`, `render_frame`), `StemStimulus`, `StemResponse`, and `StimulusFrame`. | [`four_stage_pipeline.md`](../../../docs/specs/training/four_stage_pipeline.md) |
| [`bin/dnam_train.rs`](bin/dnam_train.rs) | Thin CLI binary entrypoint delegating to `dnam_trainer::run_cli_from_args`. | [`four_stage_pipeline.md`](../../../docs/specs/training/four_stage_pipeline.md) |
| [`solver.rs`](solver.rs) & [`solver/`](solver/AGENTS.md) | `uplo='U'` Gram accumulation, Cholesky solver, Bézout crossover, `B-ESR`, and 3-Phase Audit bounds. | [`four_stage_pipeline.md`](../../../docs/specs/training/four_stage_pipeline.md), [`control_space_audit.md`](../../../docs/specs/training/control_space_audit.md) |

## 2. Invariants

- **100% Safe Rust (`#![forbid(unsafe_code)]`):** Zero `unsafe` blocks or `#[allow(unsafe_code)]`.
- **Engine Isomorphism:** Import DSP recurrences (`Op-T`, `Op-M`, `Op-N`) and `.dnam` framing from [`dnam-engine`](../../engine/README.md).
- **Zero `mod.rs`:** Follow Rust 2024 module conventions (`solver.rs` + `solver/stages.rs`).
