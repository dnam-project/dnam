# `packages/trainer/tests/` — Specification Verification & Integration Suite Guide

**`packages/trainer/tests/`** contains cross-cutting Rust integration and `proptest!` suites verifying the Two-Sweep system-identification pipeline, `uplo='U'` `.dnam-ckpt` / `DNCA` `fp16` coreset invariants, Non-Stop 3-Phase Control-Space Audit bounds, and `Noise-Subtracted B-ESR` quality metrics.

## 1. Test Suite Map

| Test File | Verified Specification Domains (`verify[...]`) | Canonical Specification (SSOT) |
| :--- | :--- | :--- |
| [`trainer_spec_verification.rs`](trainer_spec_verification.rs) | `arch.isomorphism.*`, `qual.besr.*`, `qual.eval.*`, `qual.companion.*`, `qual.tiers.*`, `container.ckpt.*`, `container.dnca.*`, `dsp.op-m.*`, `dsp.op-n.*`, `dsp.ctrl.*`, `train.capture.*`, `train.audit.*`, `train.pipeline.*`, `train.gram.*`, `train.resume.*`, `train.cache.*`, `train.cloud.*`. | [`four_stage_pipeline.md`](../../../docs/specs/training/four_stage_pipeline.md), [`control_space_audit.md`](../../../docs/specs/training/control_space_audit.md), [`psychoacoustic_quality_and_tiers.md`](../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md), [`checkpoint_and_cache_format.md`](../../../docs/specs/containers/checkpoint_and_cache_format.md) |

## 2. Testing Rules

- **Zero `docs/` Markdown Parsing (`ref[repo.docs.spec-rules-and-adrs]`):** Never read or parse Markdown files from `docs/` in unit or integration tests; cite verified rules via `// verify[<id>]` comments.
- **Property-Based Testing:** Combine deterministic boundary/vector assertions with `proptest!` sweeps across parameter and signal domains.
