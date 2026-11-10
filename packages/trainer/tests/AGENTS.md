# `packages/trainer/tests/` — Specification Verification & Integration Suite Guide

**`packages/trainer/tests/`** contains cross-cutting Rust integration and `proptest!` suites verifying the Two-Sweep pipeline, `BlackBoxDistillationAdapter` (`Chain-01..12`), `.dnam-ckpt` / `DNCA` security, 3-Phase Audit bounds, and `4-Metric` (`bESR`, `MR-IIR`, `LTAS`, `SegESR`) gates.

## 1. Test Suite Map

| Test File | Verified Domains (`verify[...]`) | Canonical Specification (SSOT) |
| :--- | :--- | :--- |
| [`trainer_spec_verification.rs`](trainer_spec_verification.rs) | `arch.isomorphism.*`, `qual.*`, `container.ckpt.*`, `container.dnca.*`, `dsp.*`, `train.*`. | [`four_stage_pipeline.md`](../../../docs/specs/training/four_stage_pipeline.md) |
| [`trainer_multirate_bitdepth_fixtures.rs`](trainer_multirate_bitdepth_fixtures.rs) | Multi-rate (`32`..`192 kHz`), 5-bit-depth (`Int12`..`Float32`), and `DNWB` pipe integration. | [`ADR-0018`](../../../docs/adr/0018-signal-chain-fixtures-crate-and-whitebox-distillation.md) |
| [`trainer_ckpt_security.rs`](trainer_ckpt_security.rs) | `DNCK` / `DCA1` / `DNCA` framing, `PCHIP` overflow safety, and `DnwbWorkerSession` hardening. | [`checkpoint_and_cache_format.md`](../../../docs/specs/containers/checkpoint_and_cache_format.md) |
| [`trainer_ckpt_dau1_security.rs`](trainer_ckpt_dau1_security.rs) | `DAU1` / `DMU1` signature parity, `DistributionPolicy` export, and anti-spoofing guards. | [`watermarking_and_marketplace.md`](../../../docs/specs/security/watermarking_and_marketplace.md) |

## 2. Testing Rules

- **Zero `docs/` Markdown Parsing (`ref[repo.docs.spec-rules-and-adrs]`):** Cite verified rules via `// verify[<id>]` comments only.
- **Property-Based Testing:** Combine deterministic boundary vectors with `proptest!` sweeps.
