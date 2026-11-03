# `src/container` — `.dnam` & `.dnam-ckpt` Binary Framing, ZOH & `uplo='U'` Packing

Routes the current parser, builder, `CRC32C`, sample-rate rescaling, and checkpoint packing code. The selected new framing and authenticated-prefix contracts are not established by the current implementation.

## 1. Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`preamble.rs`](preamble.rs) | 64-byte `DnamPreamble` (`DNAM` / `DNCK` magic, flags, offsets, `CRC32C` fields) | [`dnam_binary_format.md`](../../../../docs/containers/dnam_binary_format.md) |
| [`view.rs`](view.rs) | `DnamContainerView`, `build_dnam_container`, 4 KB in-place `update_header_in_place` | [`dnam_binary_format.md`](../../../../docs/containers/dnam_binary_format.md) |
| [`zoh.rs`](zoh.rs) | `ZohRescaledParams` (44.1–192 kHz poles, delays, LFO invariance, FIR resampling) + `proptest` | [`router_and_stages_topology.md`](../../../../docs/dsp/router_and_stages_topology.md) |
| [`ckpt.rs`](ckpt.rs) | `pack_uplo_u_f64`, `unpack_uplo_u_f64`, prefix submatrix extraction + `proptest` | [`checkpoint_and_cache_format.md`](../../../../docs/containers/checkpoint_and_cache_format.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for container verification guidance.
