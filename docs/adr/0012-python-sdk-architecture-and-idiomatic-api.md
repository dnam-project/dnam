---
id: ADR-0012
title: "Idiomatic Python SDK Architecture, Domain-Free Naming, Thin Native Shim & Schema Parity"
status: superseded
date: "2026-11-04"
supersedes: null
superseded_by: ADR-0018
specs:
  - "../specs/architecture/project_and_documentation_quality.md"
  - "../specs/architecture/tora_architecture.md"
  - "../specs/containers/dnam_binary_format.md"
requirements:
  - REQ-4.1
  - REQ-5.2
  - REQ-6.1
  - REQ-7.1
  - REQ-8.8
---

# ADR-0012: Idiomatic Python SDK Architecture, Domain-Free Naming, Thin Native Shim & Schema Parity

> **Superseded by [`ADR-0018`](0018-signal-chain-fixtures-crate-and-whitebox-distillation.md) and [`ADR-0017`](0017-pure-rust-analytical-varpro-solver-and-planar-decoupling.md):** DNAM has unified training and inference in 100% pure Rust (`packages/engine`, `packages/trainer`, `packages/studio`) and TypeScript (`packages/js`), retiring `packages/python`, `PyO3`, `sdk.python.*`, and `docs/specs/sdks/python_sdk.md` (retired by `ADR-0018`).

This record documents the historical architectural and ergonomic design decisions for the former Python SDK wrapping `packages/engine/`, superseded by [`ADR-0018`](0018-signal-chain-fixtures-crate-and-whitebox-distillation.md).

---

## 1. Context & Problem Statement

Following the TORA v2.0 engine implementation ([`ADR-0002`](0002-tri-operator-recurrent-architecture.md), [`ADR-0011`](0011-end-to-end-stage-wiring-and-arena-slice-topology.md)), the initial Python package (`dnam_engine`) exhibited five structural and maintenance liabilities:

1. **Redundant Namespace & Prefixing:** Repeating `dnam` in both the module import path and every class identifier obscured the domain separation between a deployable `.dnam` artifact (`Model`), a stateful real-time processor (`Engine`), and an iterative `.dnam-ckpt` training artifact (`Checkpoint`).
2. **Mixed-Responsibility Flat Modules:** Placing live engine control racks, offline control-space clamping, and low-level Lobatto/FM kernels inside a single flat module blurred the boundary between container inspection, real-time state mutation, and reference DSP evaluation.
3. **Duplicated DSP / ZOH Logic in the FFI Layer (`ref[dsp.zoh.ctrl-stride]`):** Recomputing `ZohRescaledParams` and control-decimation strides inside FFI shims duplicated invariants already owned by `DnamEngine`, causing compile-time and behavioral drift whenever the Rust engine's internal rescaling contract evolved.
4. **Silent Schema Drift:** Maintaining hand-written wrapper models without automated round-trip verification risked retaining superseded fields while missing TORA Stage 0–IV configurations.
5. **Incomplete Runtime Dispatch on `DynamicDnamEngine`:** Forwarding methods for `supports_freeze`, `set_bypass`, `set_stereo_width`, `configure_host_mix_and_leveling`, `sync_tempo`, `register_tap_tempo`, `apply_preset`, and `process_block_in_place` were required on `DynamicDnamEngine` so language SDKs would never reimplement engine state transitions.

---

## 2. Considered Options & Rejected Candidates

| Candidate Approach | Why It Fails | Quantified Failure / Maintenance Penalty |
| :--- | :--- | :--- |
| **Thin C-Style Binding Mirror (`import dnam_engine` with Flat 1:1 Function Exports)** | Exposes raw FFI functions and index-addressed control setters (`set_control_value(idx, val)`) directly to callers instead of structured facets. | Forces every caller to manually map control names, restore bypass/freeze states on exceptions, and coerce memory layouts. |
| **Duplicating ZOH, Preset, or Clamping Math Inside SDK Shims (`ref[dsp.zoh.ctrl-stride]`)** | Maintains a second copy of sample-rate rescaling, preset default restoration, or switch crossfade logic in the SDK wrapper crate. | Guarantees behavioral divergence from `packages/engine` and breaks bit-identical parity with native Rust `DynamicDnamEngine::process_block`. |
| **Untested or Dict-Only JSON Header Wrappers (`ref[repo.code.type-safety-and-imports]`)** | Returning untyped dictionaries from `model.header` sacrifices static typing, whereas unverified models drift from Rust `DnamJsonHeader`. | Violates the zero-`any` quality gate or causes runtime deserialization failures when loading valid `.dnam` v2 containers. |

---

## 3. Historical Decision Outcome & Supersedure

1. **Complete `DynamicDnamEngine` Forwarding:** Forwarded all runtime UX and in-place processing methods on `dnam_engine::DynamicDnamEngine` so SDK wrappers delegate 100% of DSP, ZOH rescaling, preset batching, and tempo synchronization to `packages/engine`.
2. **Supersedure by Pure-Rust Trainer & TypeScript SDK (`ADR-0017`, `ADR-0018`):** With the migration of `packages/trainer` to 100% pure Rust (`dnam-trainer`) in [`ADR-0017`](0017-pure-rust-analytical-varpro-solver-and-planar-decoupling.md), the Python SDK was retired in [`ADR-0018`](0018-signal-chain-fixtures-crate-and-whitebox-distillation.md), leaving the TypeScript / WebAudio / N-API SDK ([`../specs/sdks/typescript_sdk.md`](../specs/sdks/typescript_sdk.md)) as the canonical high-level language SDK alongside `dnam-engine`.
