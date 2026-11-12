# `packages/js` — TypeScript 7, Node.js N-API v10 & WebAudio SDK Routing Guide

**`@dnam-project/dnam`** (`dnam-js` crate) wraps `packages/engine` in a unified dual-target Rust crate (`rust/` -> native Node.js 26 N-API v10 addon + `wasm32-unknown-unknown` SIMD128 binary) and a zero-dependency TypeScript 7 / WebAudio `AudioWorklet` SDK (`src/`), specified in [`docs/specs/sdks/typescript_sdk.md`](../../docs/specs/sdks/typescript_sdk.md) and [`ADR-0013`](../../docs/adr/0013-typescript-sdk-architecture-and-dual-target-runtime.md).

## 1. Module Routing Map

- [`rust/`](rust/lib.rs): Target-agnostic safe Rust core (`rust/core/`) + conditional N-API v10 (`rust/napi/`) and WASM SIMD128 FFI (`rust/wasm/`).
- [`src/index.ts`](src/index.ts), [`src/node.ts`](src/node.ts), [`src/browser.ts`](src/browser.ts), [`src/internal/native/`](src/internal/native/driver.ts): Dual-target `NapiDriver` and `WasmDriver` routing.
- [`src/model/`](src/model/index.ts): `Model<TParams>`, `GramMatrix` (`uplo='U'`), and `crc32c`.
- [`src/engine/`](src/engine/index.ts): Stateful `Engine<TParams>` (`process`, `processAsync`, `render`, `bounce`, `stream`, `bypass`/`freeze` scopes, `telemetry`).
- [`src/parameters/`](src/parameters/index.ts), [`src/tempo/`](src/tempo/index.ts): WebAudio `AudioParam`-style `Parameter`, `ParameterMap`, `ControlSpace`, `clampParameters`, `defineModelTypes`, and `TempoController`.
- [`src/schema/`](src/schema/index.ts), [`src/errors.ts`](src/errors.ts): Native Rust `serde`-backed `decodeHeader`/`encodeHeader`, RFC 8785 JCS `headerToJcs`, and coded `DnamError`.
- [`src/provenance/`](src/provenance/index.ts), [`src/dsp/`](src/dsp/index.ts): `verify`, `signCreatorHeader`, `signMarketplaceReceipt`, `seal`/`unseal`, watermarking, and reference TORA DSP kernels.
- [`src/web-audio/`](src/web-audio/index.ts): `DnamAudioNode<TParams>`, `DnamAudioWorkletProcessor`, and lock-free `WorkletSharedState` (`SharedArrayBuffer` + `Atomics`).

## 2. Guides & Verification

1. [`CONTRIBUTING.md`](CONTRIBUTING.md) — Zero-duplication rule, dual-target Rust build (`moon run js:build`), `fast-check` testing, and `moon check js`.
2. [`README.md`](README.md) & [`docs/specs/sdks/typescript_sdk.md`](../../docs/specs/sdks/typescript_sdk.md) — API reference, subpath exports, and benchmarks.

