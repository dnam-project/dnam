# `packages/trainer/src/ckpt/` — `.dnam-ckpt` (`DNCK`), `Arena 4` & `DNCA` Calibration Guide

Implements `#![forbid(unsafe_code)]` `.dnam-ckpt` (`DNCK`) checkpoint container framing, `uplo='U'` `Arena 4` sufficient statistics, `Arena 5` (`DNCA` `INPC`/`STMC`) `fp16` coresets, `DCA1` checkpoint signatures, and zero-copy `.dnam` runtime export.

## 1. Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`arena.rs`](arena.rs) | `pack_uplo_u_f64`, `unpack_uplo_u_f64`, `extract_tier_uplo_u_submatrix`, `DNCA` (`INPC`/`STMC`), `S1C1` | [`checkpoint_and_cache_format.md`](../../../../docs/specs/containers/checkpoint_and_cache_format.md) |
| [`schema.rs`](schema.rs) | `DnamCkptHeader`, `StemRegistryEntry`, `AuditReport`, `TensorSliceRef` | [`checkpoint_and_cache_format.md`](../../../../docs/specs/containers/checkpoint_and_cache_format.md) |
| [`view.rs`](view.rs) | `DnamCkptView`, `build_dnam_ckpt_container`, `resolve_ckpt_stems_end` | [`checkpoint_and_cache_format.md`](../../../../docs/specs/containers/checkpoint_and_cache_format.md) |
| [`bundle.rs`](bundle.rs) | `build_v2_dnck_checkpoint`, `validate_v2_dnck_checkpoint`, `DCA1`, `export_runtime_dnam_from_ckpt` | [`checkpoint_and_cache_format.md`](../../../../docs/specs/containers/checkpoint_and_cache_format.md) |

See [`packages/trainer/CONTRIBUTING.md`](../../CONTRIBUTING.md) for testing and coverage rules.
