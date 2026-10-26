# `packages/js` — Node.js N-API, WASM SIMD128 & WebAudio SDK Routing Guide

**`@dnam-project/engine`** (`dnam-js` crate) wraps `packages/engine` in a dual-target Rust crate (`rust/` -> native Node.js N-API v8 addon + `wasm32-unknown-unknown` SIMD128 binary) and an idiomatic TypeScript / WebAudio `AudioWorklet` SDK (`src/`).

## 1. Module Routing Map

- [`rust/`](rust/lib.rs): Shared safe Rust core (`engine.rs`, `container.rs`, `dsp.rs`, `provenance.rs`) + conditional N-API (`napi_bindings.rs`) and WASM FFI (`wasm_ffi.rs`).
- [`src/drivers/`](src/drivers/index.ts), [`src/node.ts`](src/node.ts), [`src/browser.ts`](src/browser.ts): Dual-target N-API v8 and WASM SIMD128 driver routing.
- [`src/engine.ts`](src/engine.ts), [`src/container.ts`](src/container.ts), [`src/control.ts`](src/control.ts), [`src/dsp.ts`](src/dsp.ts): `DnamEngine<TControls>`, `.dnam` container/ZOH/Gram utilities, 3-Phase control bus, and Reference DSP.
- [`src/provenance.ts`](src/provenance.ts), [`src/schema/`](src/schema/index.ts), [`src/errors.ts`](src/errors.ts): Cryptographic provenance/watermarking, runtime schemas, and `DnamError` hierarchy.
- [`src/worklet/`](src/worklet/index.ts): Lock-free `SharedArrayBuffer` (`Atomics`) + `MessagePort` WebAudio `AudioWorklet` (`shared_memory.ts`, `processor.ts`, `controller.ts`).
- [`tests/`](tests/) & `src/**/*.test.ts`: Co-located unit tests, benchmarks ([`src/engine.bench.ts`](src/engine.bench.ts)), and differential N-API vs WASM integration suites.

## 2. Guides & Verification

1. [`CONTRIBUTING.md`](CONTRIBUTING.md) — Dual-target Rust build (`moon run js:build`), Biome runtime boundary rules, Vitest workflow (`moon check js`), and coverage/benchmarks.
2. [`README.md`](README.md) & [`monorepo_implementation_guide.md`](../../docs/monorepo_implementation_guide.md) — TypeScript/WebAudio API reference.
