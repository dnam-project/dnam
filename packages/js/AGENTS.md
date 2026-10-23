# `packages/js` — Node.js N-API, WASM SIMD128 & WebAudio SDK Routing Guide

**`@dnam-project/engine`** (`dnam-js` crate) wraps `packages/engine` in a dual-target Rust crate (`rust/` -> native Node.js N-API v8 addon + `wasm32-unknown-unknown` SIMD128 binary) and an idiomatic TypeScript / WebAudio `AudioWorklet` SDK (`src/`).

## 1. Key Modules

- [`rust/`](rust/lib.rs): Shared Rust core ([`engine.rs`](rust/engine.rs), [`container.rs`](rust/container.rs), [`dsp.rs`](rust/dsp.rs), [`provenance.rs`](rust/provenance.rs)) + conditional N-API ([`napi_bindings.rs`](rust/napi_bindings.rs)) and WASM FFI ([`wasm_ffi.rs`](rust/wasm_ffi.rs)).
- [`src/drivers/`](src/drivers/index.ts): Driver abstraction ([`types.ts`](src/drivers/types.ts), [`native.ts`](src/drivers/native.ts), [`wasm.ts`](src/drivers/wasm.ts)).
- [`src/engine.ts`](src/engine.ts): `DnamEngine<TControls>` (`DnamProcessor`) real-time processor (`processStereo`, `processMonoToStereo`, `processMono`, `processPlanar`, `processAudioBuffer`, `processAsync`, `reload`, `telemetry`).
- [`src/container.ts`](src/container.ts): `DnamContainer`, `GramMatrix` (`uplo='U'`), ZOH rescaling, and `crc32c`.
- [`src/control.ts`](src/control.ts): `ControlSpace`, `ControlRack<TControls>`, `LobattoBasis`, `Rank4Fm`, `chebyshevLfo`, and `trustRegionSoftclip`.
- [`src/dsp.ts`](src/dsp.ts): Reference DSP (`PolyAdaa`, `SelectiveRgLru`, `ModulatedUfdn`, `unitRotor`, `lagrangeWeights`, `givensHouseholderMix`).
- [`src/provenance.ts`](src/provenance.ts), [`src/schema/`](src/schema/index.ts) & [`src/errors.ts`](src/errors.ts): Provenance/watermarking, runtime schemas, and `DnamError` hierarchy.
- [`src/worklet/`](src/worklet/index.ts): Lock-free `SharedArrayBuffer` (`Atomics`) + `MessagePort` fallback WebAudio `AudioWorklet` ([`shared_memory.ts`](src/worklet/shared_memory.ts), [`processor.ts`](src/worklet/processor.ts), [`controller.ts`](src/worklet/controller.ts)).
- [`test/`](test/): Vitest v5 unit, `fast-check` property, differential N-API vs WASM, and benchmark suites.

## 2. Required Reading

1. [`CONTRIBUTING.md`](CONTRIBUTING.md) — Dual-target Rust build, Biome runtime boundaries, and Vitest workflow.
2. [`README.md`](README.md) & [`monorepo_implementation_guide.md`](../../docs/monorepo_implementation_guide.md) — TypeScript & WebAudio API reference.

## 3. Verification

```bash
pnpm --filter @dnam-project/engine lint && pnpm --filter @dnam-project/engine typecheck && pnpm --filter @dnam-project/engine coverage
```
