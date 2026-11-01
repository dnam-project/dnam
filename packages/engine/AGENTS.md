# `packages/engine` — Rust `#![no_std]` + `alloc` SIMD Engine Routing Guide

**`dnam-engine`** is the `#![no_std]` + `alloc`, zero-audio-allocation, 16B/64B SIMD DSP and container core for **DNAM ("Tri-Physics")**.

## 1. Submodule Discovery Map

| Subsystem / Module | Scoped `AGENTS.md` | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| **Root & Engine** (`lib.rs`, `types.rs`, `engine.rs`, `dynamic.rs`, `memory.rs`, `ffi.rs`) | [`AGENTS.md`](AGENTS.md) | [`tri_physics_architecture.md`](../../docs/overview/tri_physics_architecture.md) |
| **Hardware SIMD & Math** (`src/math.rs`, `src/math/*`) | [`src/math/AGENTS.md`](src/math/AGENTS.md) | [`core1_polyadaa_and_rglru.md`](../../docs/dsp/core1_polyadaa_and_rglru.md) |
| **Core 1: `PolyADAA` & RG-LRU** (`src/core1.rs`, `src/core1/*`) | [`src/core1/AGENTS.md`](src/core1/AGENTS.md) | [`core1_polyadaa_and_rglru.md`](../../docs/dsp/core1_polyadaa_and_rglru.md) |
| **Core 2: Diffusive `M-UFDN`** (`src/core2.rs`, `src/core2/*`) | [`src/core2/AGENTS.md`](src/core2/AGENTS.md) | [`core2_modulated_ufdn.md`](../../docs/dsp/core2_modulated_ufdn.md) |
| **Parametric Control Bus** (`src/control.rs`, `src/control/*`) | [`src/control/AGENTS.md`](src/control/AGENTS.md) | [`parametric_control_bus.md`](../../docs/dsp/parametric_control_bus.md) |
| **`.dnam` Container & ZOH** (`src/container.rs`, `src/container/*`) | [`src/container/AGENTS.md`](src/container/AGENTS.md) | [`dnam_binary_format.md`](../../docs/containers/dnam_binary_format.md) |
| **Marketplace & Provenance** (`src/provenance.rs`, `src/provenance/*`) | [`src/provenance/AGENTS.md`](src/provenance/AGENTS.md) | [`provenance_and_enclave_drm.md`](../../docs/security/provenance_and_enclave_drm.md) |
| **JSON Schema & Controls** (`src/schema.rs`, `src/schema/*`) | [`src/schema/AGENTS.md`](src/schema/AGENTS.md) | [`dnam_binary_format.md`](../../docs/containers/dnam_binary_format.md) |

## 2. Guides & Verification

1. [`CONTRIBUTING.md`](CONTRIBUTING.md) — Crate layout, `#[forbid(unsafe_code)]` isolation, submodule invariants, zero-allocation rules, and `moon check engine` verification commands.
2. [`README.md`](README.md) — Public Rust API, I/O topologies, and embedded scratch usage.
