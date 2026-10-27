# `src/container` — `.dnam` & `.dnam-ckpt` Binary Framing, ZOH & `uplo='U'` Packing

Implements the hybrid binary container parser/builder (`#[forbid(unsafe_code)]`), Castagnoli `CRC32C` prefix verification, 4 KB page-aligned in-place JSON header editor, Continuous-Time Zero-Order Hold (ZOH) sample-rate rescaling ($44.1\text{–}96\text{ kHz}$), and `Arena 3` `uplo='U'` sufficient statistics packing.

## 1. Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`preamble.rs`](preamble.rs) | 64-byte `DnamPreamble` (`DNAM` / `DNCK` magic, flags, offsets, `CRC32C` fields) | [`dnam_container_schema.md`](../../../../docs/dnam_container_schema.md) |
| [`view.rs`](view.rs) | `DnamContainerView`, `build_dnam_container`, 4 KB in-place `update_header_in_place` | [`dnam_container_schema.md`](../../../../docs/dnam_container_schema.md) |
| [`zoh.rs`](zoh.rs) | `ZohRescaledParams` (44.1–96 kHz poles, delays, and LFO invariance) + `proptest` | [`dnam_container_schema.md`](../../../../docs/dnam_container_schema.md) |
| [`ckpt.rs`](ckpt.rs) | `pack_uplo_u_f64`, `unpack_uplo_u_f64`, prefix submatrix extraction + `proptest` | [`training_and_iterative_resume.md`](../../../../docs/training_and_iterative_resume.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for incremental `CRC32C` verification rules.
