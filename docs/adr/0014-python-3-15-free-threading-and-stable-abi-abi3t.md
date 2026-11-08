---
id: ADR-0014
title: "Python 3.15 Free-Threaded Dev Runtime, PEP 803 Dual-ABI (abi3.abi3t) Wheels & PyO3 Concurrency"
status: superseded
date: "2026-11-06"
supersedes: []
superseded_by: ADR-0018
specs:
  - "../specs/architecture/project_and_documentation_quality.md"
  - "../specs/architecture/tora_architecture.md"
  - "../specs/sdks/typescript_sdk.md"
requirements:
  - REQ-4.1
  - REQ-6.1
  - REQ-6.3
---

# ADR-0014: Python 3.15 Free-Threaded Dev Runtime, PEP 803 Dual-ABI (`abi3.abi3t`) Wheels & PyO3 Concurrency

> **Superseded by [`ADR-0018`](0018-signal-chain-fixtures-crate-and-whitebox-distillation.md) and [`ADR-0017`](0017-pure-rust-analytical-varpro-solver-and-planar-decoupling.md):** DNAM has unified training and inference in 100% pure Rust (`packages/engine`, `packages/trainer`, `packages/studio`) and TypeScript (`packages/js`), retiring Python/`PyO3`, `sdk.python.*`, and `docs/specs/sdks/python_sdk.md` (retired by `ADR-0018`).

This record documents the historical architectural decision to target Python `3.15t` free-threaded execution and PEP 803 `abi3t` wheels prior to the pure-Rust trainer unification in [`ADR-0017`](0017-pure-rust-analytical-varpro-solver-and-planar-decoupling.md) and [`ADR-0018`](0018-signal-chain-fixtures-crate-and-whitebox-distillation.md).

---

## 1. Context & Problem Statement

Following [`ADR-0012`](0012-python-sdk-architecture-and-idiomatic-api.md), four runtime and packaging constraints motivated free-threaded concurrency hardening before the Python layer was ultimately superseded by pure Rust:

| Previous Formulation | Failure Mechanism | Runtime or Packaging Consequence |
| :--- | :--- | :--- |
| **`abi3-py314` Stable ABI Without `abi3t`** | Python 3.14's legacy `abi3` stable ABI does not support free-threaded (`Py_GIL_DISABLED`) interpreters; PEP 803 introduces `abi3t` starting at Python `3.15`. | Loading an `abi3`-only extension on a free-threaded interpreter either fails at import time or forces a per-minor-version wheel matrix. |
| **Implicit GIL Re-Enablement on Import** | Any C/Rust extension that omits `Py_mod_gil = Py_MOD_GIL_NOT_USED` causes the free-threaded interpreter to re-enable the Global Interpreter Lock globally on `import`. | Silently degrades multi-threaded audio batch rendering back to serialized execution. |
| **Unchecked Slice Borrows Under Concurrent Threads** | Extracting raw slice pointers without borrow tracking allows a second thread to mutate or resize the same buffer concurrently during native DSP execution. | Creates data races reachable from concurrent callers. |

---

## 2. Historical Decision & Supersedure

1. **Lock-Free Engine-Core SPSC Control Ingress (`ref[dsp.ctrl.op-m.spsc-ingress]`):**
   - Motivated the pure-Rust `dnam_engine::host` concurrency layer ([`ADR-0015`](0015-realtime-host-concurrency-spsc-and-telemetry-seqlock.md)) providing wait-free `TelemetryCell` seqlock reads and 3-lane `SpscControlRing` ingress.
2. **Supersedure by Pure-Rust Workspace (`ADR-0017`, `ADR-0018`):**
   - Eliminating Python/PyTorch in favor of `dnam-trainer` (`packages/trainer`) removed the `uv` / `PyO3` / `maturin` build surface entirely while preserving `dnam_engine::host` across native Rust and TypeScript (`N-API v10` / `WASM`, [`../specs/sdks/typescript_sdk.md`](../specs/sdks/typescript_sdk.md)).
