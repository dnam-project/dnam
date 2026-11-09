# `src/schema` — `#![no_std]` + `alloc` JSON Header, Control Schema & Error Types

Implements `#![no_std]` + `alloc` `serde` / `serde_json` parsing and validation (`#[forbid(unsafe_code)]`) for runtime `.dnam` JSON headers, UI power-skew gesture mapping, and compilation into 64-byte [`ControlClampDescriptor`](../control/descriptor.rs) cache lines. (`.dnam-ckpt` schema types live in `packages/trainer/src/ckpt/schema.rs`.)

## 1. Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`header.rs`](header.rs) | `DnamJsonHeader`, `TierProfile`, TORA `StageConfigs`, `compile_clamp_descriptors` | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md) |
| [`controls.rs`](controls.rs) | `ControlSchemaDef`, `ControlUiHints`, midpoint power-skew (`norm_to_physical` / `physical_to_norm`) + `proptest` | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md) |
| [`signatures.rs`](signatures.rs) | `ModelSignatures`, `CreatorSignature`, `MarketplaceReceipt`, `AuthorizedDevice`, `RoyaltySplit` | [`provenance_and_enclave_drm.md`](../../../../docs/specs/security/provenance_and_enclave_drm.md) |
| [`error.rs`](error.rs) | `DnamError` (`core::error::Error`) and cold-path `f64` math wrappers | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for `#![no_std]` + `alloc` `serde` rules.
