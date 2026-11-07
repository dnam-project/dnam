# `packages/engine` — Rust `#![no_std]` + `alloc` SIMD Engine Routing Guide

**`dnam-engine`** is the `#![no_std]` + `alloc`, zero-audio-allocation, 16B/64B SIMD DSP and container core for **DNAM**.

## 1. Submodule Discovery Map

| Subsystem / Module | Scoped `AGENTS.md` | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| **Engine Runtime & Dispatch** (`src/engine/*`, `lib.rs`, `types.rs`) | [`src/engine/AGENTS.md`](src/engine/AGENTS.md) | [`tora_architecture.md`](../../docs/specs/architecture/tora_architecture.md) |
| **Real-Time Host & SPSC** (`src/host.rs`, `src/host/*`) | [`src/host/AGENTS.md`](src/host/AGENTS.md) | [`parametric_control_bus.md`](../../docs/specs/dsp/parametric_control_bus.md) |
| **Operators: Op-T, Op-M, Op-N** (`src/operators/*`) | [`src/operators/AGENTS.md`](src/operators/AGENTS.md) | [`op_t_transport.md`](../../docs/specs/dsp/op_t_transport.md), [`op_m_modal_kautz.md`](../../docs/specs/dsp/op_m_modal_kautz.md), [`op_n_commuted_cascade.md`](../../docs/specs/dsp/op_n_commuted_cascade.md) |
| **Stages & Matrix Router** (`src/stages/*`) | [`src/stages/AGENTS.md`](src/stages/AGENTS.md) | [`router_and_stages_topology.md`](../../docs/specs/dsp/router_and_stages_topology.md) |
| **Hardware SIMD & Math** (`src/math.rs`, `src/math/*`) | [`src/math/AGENTS.md`](src/math/AGENTS.md) | [`op_n_commuted_cascade.md`](../../docs/specs/dsp/op_n_commuted_cascade.md) |
| **Parametric Control Bus** (`src/control.rs`, `src/control/*`) | [`src/control/AGENTS.md`](src/control/AGENTS.md) | [`parametric_control_bus.md`](../../docs/specs/dsp/parametric_control_bus.md) |
| **`.dnam` Container & ZOH** (`src/container.rs`, `src/container/*`) | [`src/container/AGENTS.md`](src/container/AGENTS.md) | [`dnam_binary_format.md`](../../docs/specs/containers/dnam_binary_format.md) |
| **Marketplace & Provenance** (`src/provenance.rs`, `src/provenance/*`) | [`src/provenance/AGENTS.md`](src/provenance/AGENTS.md) | [`provenance_and_enclave_drm.md`](../../docs/specs/security/provenance_and_enclave_drm.md) |
| **JSON Schema & Controls** (`src/schema.rs`, `src/schema/*`) | [`src/schema/AGENTS.md`](src/schema/AGENTS.md) | [`dnam_binary_format.md`](../../docs/specs/containers/dnam_binary_format.md) |

## 2. Guides & Verification

1. [`spec_scope.toml`](spec_scope.toml) — Rule scope manifest.
2. [`CONTRIBUTING.md`](CONTRIBUTING.md) — Crate layout, co-located tests, and `engine:coverage`.
3. [`README.md`](README.md) — Public Rust API and I/O topologies.
