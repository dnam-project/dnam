# `packages/engine` — Rust `#![no_std]` + `alloc` SIMD Engine Routing Guide

**`dnam-engine`** is the `#![no_std]` + `alloc`, zero-audio-allocation, 16B/64B SIMD DSP and container core for **DNAM ("Tri-Physics")**.

## 1. Submodule Discovery Map

| Subsystem / Module | Scoped `AGENTS.md` | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| **Root & Engine** (`lib.rs`, `types.rs`, `engine.rs`, `dynamic.rs`, `memory.rs`, `ffi.rs`) | [`AGENTS.md`](AGENTS.md) | [`monorepo_implementation_guide.md`](../../docs/monorepo_implementation_guide.md) |
| **Hardware SIMD & Math** (`src/math.rs`, `src/math/*`) | [`src/math/AGENTS.md`](src/math/AGENTS.md) | [`core1_hybrid_ssm_and_polyadaa.md`](../../docs/core1_hybrid_ssm_and_polyadaa.md) |
| **Core 1: `PolyADAA` & RG-LRU** (`src/core1.rs`, `src/core1/*`) | [`src/core1/AGENTS.md`](src/core1/AGENTS.md) | [`core1_hybrid_ssm_and_polyadaa.md`](../../docs/core1_hybrid_ssm_and_polyadaa.md) |
| **Core 2: Diffusive `M-UFDN`** (`src/core2.rs`, `src/core2/*`) | [`src/core2/AGENTS.md`](src/core2/AGENTS.md) | [`core2_modulated_ufdn.md`](../../docs/core2_modulated_ufdn.md) |
| **Parametric Control Bus** (`src/control.rs`, `src/control/*`) | [`src/control/AGENTS.md`](src/control/AGENTS.md) | [`parametric_control_bus.md`](../../docs/parametric_control_bus.md) |
| **`.dnam` Container & ZOH** (`src/container.rs`, `src/container/*`) | [`src/container/AGENTS.md`](src/container/AGENTS.md) | [`dnam_container_schema.md`](../../docs/dnam_container_schema.md) |
| **Marketplace & Provenance** (`src/provenance.rs`, `src/provenance/*`) | [`src/provenance/AGENTS.md`](src/provenance/AGENTS.md) | [`digital_marketplace_and_provenance.md`](../../docs/digital_marketplace_and_provenance.md) |
| **JSON Schema & Controls** (`src/schema.rs`, `src/schema/*`) | [`src/schema/AGENTS.md`](src/schema/AGENTS.md) | [`dnam_container_schema.md`](../../docs/dnam_container_schema.md) |

## 2. Guides & Verification

1. [`CONTRIBUTING.md`](CONTRIBUTING.md) — Crate layout, `#[forbid(unsafe_code)]` isolation, submodule invariants, zero-allocation rules, and `moon check engine` verification commands.
2. [`README.md`](README.md) — Public Rust API, I/O topologies, and embedded scratch usage.
