---
id: ADR-0015
title: "Real-Time Host Concurrency: 3-Lane SPSC Control Ingress, AcceptedStateShadow, Wait-Free Telemetry Seqlock & Safe Engine Swap"
status: accepted
date: "2026-11-06"
supersedes: []
superseded_by: null
specs:
  - "../specs/architecture/tora_architecture.md"
  - "../specs/dsp/parametric_control_bus.md"
  - "../specs/sdks/python_sdk.md"
  - "../specs/sdks/typescript_sdk.md"
requirements:
  - REQ-2.1
  - REQ-6.1
  - REQ-6.2
  - REQ-6.3
---

# ADR-0015: Real-Time Host Concurrency: 3-Lane SPSC Control Ingress, `AcceptedStateShadow`, Wait-Free Telemetry Seqlock & Safe Engine Swap

This record documents the architectural decision to introduce `dnam_engine::host` (`packages/engine/src/host.rs`) as a `#![no_std]` + `alloc`, 100% safe Rust (`#[forbid(unsafe_code)]`) real-time concurrency layer providing a **3-Lane SPSC Control Ingress (`SpscControlRing`)**, **synchronous producer-side validation and state shadowing (`AcceptedStateShadow`)**, a **wait-free `AtomicU64` seqlock (`TelemetryCell`)**, a **ping-pong non-blocking hot swap (`RealtimeEngineSwap`)**, and a unified **`DnamHostHandle`** shared across the Rust engine, Python 3.15 (`3.15t`) SDK (`packages/python/`), and TypeScript 7 (`N-API v10` / `WASM`) SDK (`packages/js/`), specified normatively in [`../specs/architecture/tora_architecture.md`](../specs/architecture/tora_architecture.md), [`../specs/dsp/parametric_control_bus.md`](../specs/dsp/parametric_control_bus.md), [`../specs/sdks/python_sdk.md`](../specs/sdks/python_sdk.md), and [`../specs/sdks/typescript_sdk.md`](../specs/sdks/typescript_sdk.md).

---

## 1. Context & Problem Statement

Following [`ADR-0014`](0014-python-3-15-free-threading-and-stable-abi-abi3t.md), wrapping `DynamicDnamEngine<'static>` inside a single coarse-grained `Mutex<DynamicDnamEngine<'static>>` guaranteed memory safety under free-threaded Python `3.15t` (`Py_GIL_DISABLED = 1`), but left four real-time host concurrency bottlenecks across Python, Node.js (`processAsync`), and plugin hosts:

| Previous Formulation | Failure Mechanism | Real-Time or Host Concurrency Consequence |
| :--- | :--- | :--- |
| **Single Coarse `Mutex<DynamicDnamEngine>` Shared by Audio, Control & Telemetry** | Any multi-block `process` or `bounce` render holds the engine `Mutex` for the duration of the render (milliseconds to tens of milliseconds). | Concurrent UI/MIDI threads calling `set_control_value`, `apply_preset`, `sync_tempo`, or reading `engine.telemetry` block until the entire audio render finishes, violating real-time responsiveness (`REQ-6.1`, `REQ-6.2`). |
| **Single Fixed-Capacity FIFO Ring for All Control Updates** | High-frequency continuous parameter automation (MIDI CC bursts, UI knob drags, or DAW curves) can emit thousands of updates between audio callbacks. | A simple bounded FIFO either drops commands when full or forces blocking/allocation on the producer or audio consumer. |
| **Asynchronous Ring Enqueue Without Producer-Side State Shadow** | Enqueuing control or preset commands into an SPSC ring defers `DynamicDnamEngine` mutation until the next `process_*` audio block drains the ring. | Synchronous host inspection (`engine.controls[id]`, `engine.controls.clamped`, `engine.active_tier`, `engine.is_bypassed`, `engine.tempo.bpm`) immediately after a setter call observes stale pre-command state when no audio thread is actively ticking. |
| **`UnsafeCell` Seqlock & Raw-Pointer Engine Swap Under `#![deny(unsafe_code)]`** | Traditional seqlocks (`UnsafeCell<T>` with speculative non-atomic reads) trigger Rust data-race undefined behavior (`UB`) when a writer mutates `T` concurrently with a reader, and raw `AtomicPtr` swaps require `unsafe` blocks. | Violates the crate-wide `#![deny(unsafe_code)]` invariant in `dnam-engine` (`packages/engine/src/lib.rs`). |

---

## 2. Decision & Counterexample Matrix

| Mechanism & Spec Rule | Canonical Selection (`dnam_engine::host`) | Counterexample to Naive Formulation |
| :--- | :--- | :--- |
| **3-Lane Lossless SPSC Control Ingress (`SpscControlRing`, `ref[dsp.ctrl.op-m.spsc-ingress]`)** | Combine three cache-line-aligned (`#[repr(align(64))]`) pure-atomic lanes: (1) **Lane 1** (`[AtomicU64; 32]`) coalesces per-control `Slewed`/`Immediate` updates in `O(1)` by packing `(epoch_u29, mode_u2, kind_u1, payload_u32)` into a single `u64`; (2) **Lane 2** (`64`-slot SPSC command ring of `AtomicU64` words) queues discrete structural transitions (`ApplyPreset`, `SyncTempo`, `TapTempo`, `SetBypass`, `SetFreeze`, `SetTier`, `ResetEngine`, etc.); (3) **Lane 3** (`FallbackLatch` of `AtomicU64` words) absorbs structural state if Lane 2 fills before the next audio tick, drained in deterministic causal epoch order `(Lane 3 -> Lane 2 -> Lane 1)`. | A single uncoalesced FIFO ring overflows under high-rate knob sweeps before the next audio callback; dropping overflowed structural commands (`SetBypass`, `ApplyPreset`, `ResetEngine`) causes permanent state divergence between host and DSP engine. |
| **Synchronous Producer Validation & `AcceptedStateShadow` (`ref[dsp.ctrl.op-m.spsc-ingress]`)** | Validate control indices, switch option indices, preset indices, freeze capability, and sample-rate bounds synchronously on the producer side before publishing to the SPSC lanes, and immediately update `AcceptedStateShadow` (evaluating `ControlClampDescriptor` + conditional clamp rules in `O(P)`). | Deferring validation to the audio consumer prevents synchronous error reporting (`ControlError`, `SampleRateError`) on the calling thread and causes `get_control()` / `clamped_controls()` to return stale values until the next `process_*` call. |
| **100% Safe Wait-Free `AtomicU64` Seqlock (`TelemetryCell`, `ref[arch.telemetry.seqlock-publication]`)** | Pack `DnamTelemetrySnapshot` (`RawTelemetryRecord`) into a cache-line-aligned (`#[repr(align(64))]`) array of `29` `AtomicU64` words guarded by `seq: AtomicU64` (`seq` odd -> relaxed atomic word stores -> `Release` store `seq + 1`; reader checks `Acquire` `seq1` even -> relaxed atomic word loads -> `Acquire` fence -> `seq1 == seq2`, bounded to `64` retries with last-good fallback). | Using `UnsafeCell<DnamTelemetrySnapshot>` with non-atomic struct copies is undefined behavior (`UB`) under the Rust memory model whenever a reader overlaps a writer, even if the reader discards the torn read. |
| **Safe Ping-Pong `RealtimeEngineSwap` & Sub-Block Chunking (`DnamHostHandle`)** | Implement `RealtimeEngineSwap` using two `Mutex<Option<DynamicDnamEngine<'static>>>` slots indexed by `active_slot: AtomicUsize` (`try_lock` on the audio path, old-engine `Drop` on the controller thread), and chunk large `DnamHostHandle::process_*` buffers into `64`-frame real-time sub-blocks that drain `ControlConsumer` at sub-block entry and publish `TelemetryCell` at sub-block exit. | Holding the audio lock across an unchunked multi-second buffer delays mid-render SPSC control actuation and telemetry publication until the entire buffer completes. |

---

## 3. Consequences & Verification

1. **Zero-`unsafe` Real-Time Host Layer (`packages/engine/src/host.rs`):**
   - `packages/engine/src/lib.rs` declares `#[forbid(unsafe_code)] pub mod host;`, and every submodule (`host/telemetry.rs`, `host/spsc.rs`, `host/swap.rs`, `host/handle.rs`) compiles with `#![forbid(unsafe_code)]` on both native targets and `wasm32-unknown-unknown`.
2. **Non-Blocking Control & Telemetry Across Python (`3.15t`) and TypeScript (`N-API v10` / `WASM`):**
   - In `packages/python/rust/engine.rs` and `packages/js/rust/core/engine.rs`, `PyEngine` and `EngineCore` wrap `DnamHostHandle`. Control mutations (`set_control_value`, `apply_preset`, `sync_tempo`, `set_bypass`, `set_freeze`, `set_tier`, etc.) acquire only the lightweight `Mutex<ControlProducer>` (nanoseconds), and `telemetry` reads `TelemetryReader` wait-free (`0` locks), completing in `< 5 ms` even while another thread executes a long multi-block `process` or `bounce` render.
