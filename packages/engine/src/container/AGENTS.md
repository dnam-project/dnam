# `src/container` — `.dnam` Runtime Binary Framing & Continuous-Time ZOH

Implements the `#![no_std]` + `alloc` runtime `.dnam` parser, builder, `CRC32C`, `DMU1`/`DMUE` mutable suffix framing, and Continuous-Time ZOH sample-rate rescaling (`#[forbid(unsafe_code)]`). Training checkpoint (`.dnam-ckpt` / `DNCK`, `Arena 4`, `Arena 5`) types live in `packages/trainer/src/ckpt/`.

## 1. Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`preamble.rs`](preamble.rs), [`wire.rs`](wire.rs) ([`layout.rs`](wire/layout.rs), [`jcs.rs`](wire/jcs.rs), [`slices_and_cp1.rs`](wire/slices_and_cp1.rs), [`conditional.rs`](wire/conditional.rs)) | 64B preamble, 128B directory, JCS header projection, `DMU1`/`DMUE`, `CP1`, `delay_histories` | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md) |
| [`view.rs`](view.rs), [`strip.rs`](strip.rs) | `DnamContainerView`, `build_dnam_container`, 4 KB in-place `update_header_in_place`, tier stripping | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md) |
| [`zoh.rs`](zoh.rs) | `ZohRescaledParams` (44.1–192 kHz poles, delays, LFO invariance, FIR resampling) + `proptest` | [`router_and_stages_topology.md`](../../../../docs/specs/dsp/router_and_stages_topology.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for container verification guidance.
