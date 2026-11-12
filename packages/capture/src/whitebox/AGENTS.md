# `packages/capture/src/whitebox/` — Dual-Mode `DNWB` White-Box Distillation Guide

Implements the `64 B` `DNWB` binary pipe protocol, `FaustRunner`, and `SpiceRunner` (`PCHIP` `4x` + `309`-tap Kaiser-sinc `4x -> 1x` decimation).

## Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`../whitebox.rs`](../whitebox.rs) | `DnwbHeader64B`, `DnwbCommand`, `DnwbError`, and structured `DNWB` stimulus request serialization (`write_dnwb_stimulus_request_with_switches`) | [`ADR-0018`](../../../../docs/adr/0018-signal-chain-fixtures-crate-and-whitebox-distillation.md) |
| [`runners.rs`](runners.rs) | `SpiceRunner` (`PCHIP` variable-step `4x` resampling + `309`-tap Kaiser-sinc `4x -> 1x` causal/zero-phase decimation) | [`ADR-0018`](../../../../docs/adr/0018-signal-chain-fixtures-crate-and-whitebox-distillation.md) |
| [`faust.rs`](faust.rs) | `FaustControlSpec`, `parse_faust_json_controls`, and `FaustRunner` dual-mode in-process / `DNWB` subprocess adapter | [`ADR-0018`](../../../../docs/adr/0018-signal-chain-fixtures-crate-and-whitebox-distillation.md) |
| [`ipc.rs`](ipc.rs) | Concurrent non-deadlocking `DNWB` stdin/stdout pipe transport (`execute_dnwb_subprocess_request`, `serve_dnwb_single_request_with_switches`) | [`ADR-0018`](../../../../docs/adr/0018-signal-chain-fixtures-crate-and-whitebox-distillation.md) |
| [`session.rs`](session.rs) | Persistent stateful `DnwbWorkerSession` and multi-packet stream server (`serve_dnwb_stream_with_switches`) | [`ADR-0018`](../../../../docs/adr/0018-signal-chain-fixtures-crate-and-whitebox-distillation.md) |
