# `packages/engine` — Rust `#![no_std]` SIMD Engine Guide

**`dnam-engine`** is the `#![no_std]` + `alloc`, 0-audio-alloc, 16B/64B SIMD DSP and container core for **DNAM ("Tri-Physics")**.

## 1. Subsystem Discovery Map

| Module | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`src/lib.rs`](src/lib.rs) | `DnamEngine<B>`, `DynamicDnamEngine`, `Tier`, `FtzDazGuard`, telemetry | [`monorepo_implementation_guide.md`](../../docs/monorepo_implementation_guide.md) |
| [`src/memory.rs`](src/memory.rs) | `CacheLine64`, 64B `AlignedBuffer`, `ArenaStorage`, `DelayScratchStorage` | [`dnam_container_schema.md`](../../docs/dnam_container_schema.md) |
| [`src/math.rs`](src/math.rs), `src/math/*` | `SimdVec4`, `MathBackend` (`Scalar`, `NEON`, `ARM SME`, `x86 AVX2`, `WASM128`) | [`core1_hybrid_ssm_and_polyadaa.md`](../../docs/core1_hybrid_ssm_and_polyadaa.md) |
| [`src/schema.rs`](src/schema.rs) | `#![no_std]` `serde` + `serde_json` `.dnam`/`.dnam-ckpt` header & skew | [`dnam_container_schema.md`](../../docs/dnam_container_schema.md) |
| [`src/provenance.rs`](src/provenance.rs) | `CryptoProvider`, `RustCryptoProvider`, `ChaCha20` 64B wrapping, watermarks | [`digital_marketplace_and_provenance.md`](../../docs/digital_marketplace_and_provenance.md) |
| [`src/container.rs`](src/container.rs) | `DnamPreamble`, `Arena 0..3` (`uplo='U'`), 4 KB in-place editor & ZOH | [`dnam_container_schema.md`](../../docs/dnam_container_schema.md) |
| [`src/control.rs`](src/control.rs) | $3\text{ kHz}$ `ControlBus<B>`, `ControlClampDescriptor`, anti-zipper (`CL-12..16`) | [`parametric_control_bus.md`](../../docs/parametric_control_bus.md) |
| [`src/core1.rs`](src/core1.rs) | 12-Layer Hybrid Dilated-Conv (`PolyADAA`) + `SelectiveRgLru` (`CL-01..05`) | [`core1_hybrid_ssm_and_polyadaa.md`](../../docs/core1_hybrid_ssm_and_polyadaa.md) |
| [`src/core2.rs`](src/core2.rs) | Stereo Diffusive `ModulatedUfdn<B>` & Lyapunov loop (`CL-06..11`) | [`core2_modulated_ufdn.md`](../../docs/core2_modulated_ufdn.md) |
| [`src/ffi.rs`](src/ffi.rs) | Zero-allocation `extern "C"` ABI for WASM and embedded C/C++ hosts | [`monorepo_implementation_guide.md`](../../docs/monorepo_implementation_guide.md) |
| `tests/`, `benches/` | Empirical suites, backend parity, 0-alloc `#[global_allocator]` audit | [`verification_and_guardrails.md`](../../docs/verification_and_guardrails.md) |

## 2. Invariants

- **Zero `FSQRT` / `FDIV` / Allocs** at $48\text{ kHz}$ and $3\text{ kHz}$. Keep `< 2,500` B.
