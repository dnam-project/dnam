# `src/schema` — `#![no_std]` + `alloc` JSON Header, Control Schema & Error Types

Implements `#![no_std]` + `alloc` `serde` / `serde_json` parsing and validation (`#[forbid(unsafe_code)]`) for `.dnam` and `.dnam-ckpt` JSON headers, UI power-skew gesture mapping, and compilation into 64-byte [`ControlClampDescriptor`](../control/descriptor.rs) cache lines.

## 1. Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`header.rs`](header.rs) | `DnamJsonHeader`, `TierProfile`, `Core1Config`, `Core2Config`, `compile_clamp_descriptors` | [`dnam_container_schema.md`](../../../../docs/dnam_container_schema.md) |
| [`controls.rs`](controls.rs) | `ControlSchemaDef`, `ControlUiHints`, midpoint power-skew (`norm_to_physical` / `physical_to_norm`) + `proptest` | [`dnam_container_schema.md`](../../../../docs/dnam_container_schema.md) |
| [`signatures.rs`](signatures.rs) | `ModelSignatures`, `CreatorSignature`, `MarketplaceReceipt`, `AuthorizedDevice`, `RoyaltySplit` | [`digital_marketplace_and_provenance.md`](../../../../docs/digital_marketplace_and_provenance.md) |
| [`error.rs`](error.rs) | `DnamError` (`core::error::Error`) and cold-path `f64` math wrappers | [`dnam_container_schema.md`](../../../../docs/dnam_container_schema.md) |

## 2. Invariants

- Always use `serde` (`default-features = false, features = ["alloc", "derive"]`) and `serde_json` (`default-features = false, features = ["alloc"]`).
