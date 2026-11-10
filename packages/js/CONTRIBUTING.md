# Contributing to `@dnam-project/dnam` (`packages/js/`)

This guide covers engineering, dual-target Rust compilation (Node.js 26 LTS `N-API v10` + WebAssembly `SIMD128`), strict Biome linting, TypeScript 7.0 typing, WebAudio `AudioWorklet` runtime boundary rules, testing, coverage, and profiling/benchmarking standards for **`@dnam-project/dnam`** ([`README.md`](README.md), specified in [`docs/specs/sdks/typescript_sdk.md`](../../docs/specs/sdks/typescript_sdk.md) and [`ADR-0013`](../../docs/adr/0013-typescript-sdk-architecture-and-dual-target-runtime.md)).

All contributions must also comply with the root [`CONTRIBUTING.md`](../../CONTRIBUTING.md) and [`CLA.md`](../../CLA.md) (`git commit -s`).

---

## 1. Unified Dual-Target Architecture & Zero-Duplication Rule

Consult [`AGENTS.md`](AGENTS.md) for the module-to-spec discovery table:

1. **Zero Duplicated Engine Logic (`ref[sdk.ts.runtime.dual-target-drivers]`):**
   - Never reimplement DSP, ZOH sample-rate rescaling, 3-Phase control clamping, Lobatto calibration inversion, JSON schema validation, RFC 8785 JCS canonicalization, or provenance signature/receipt construction in TypeScript or in `packages/js/rust/`.
   - Delegate 100% of domain logic to `dnam-engine` via [`rust/core/`](rust/core.rs). Keep `src/` strictly focused on idiomatic TypeScript ergonomics (`Model`, `Checkpoint`, `Engine`, `Parameter`, `DnamAudioNode`), snake_case $\leftrightarrow$ camelCase key translation (`src/schema/case-codec.ts`), and native handle/buffer management (`0` runtime npm dependencies).
2. **Unified Dual-Target Rust Crate (`rust/` -> `dnam-js`):**
   - Uses the modern Rust 2024 `foo.rs` + `foo/bar.rs` module layout (zero `mod.rs` files), enforces `#![deny(unsafe_code)]` at the crate root (`rust/lib.rs`, for `#[napi]` proc-macro expansion and Rust 2024 `#[unsafe(no_mangle)]` C-ABI symbol exports), and enforces `#[forbid(unsafe_code)]` on `core`, `error`, and `wasm::abi` (`ref[sdk.ts.runtime.zero-unsafe]`, `ref[repo.code.type-safety-and-imports]`). Never use `unsafe { ... }` blocks, `unsafe impl Send`, or raw pointer slice aliases anywhere in `rust/`; use `&mut [f32]` / `&mut [f64]` N-API slice parameters and owned `Vec<f32>` snapshots for background `AsyncTask` workers.
   - [`rust/core.rs`](rust/core.rs) + `rust/core/{engine,container,control,provenance,operators}.rs` implement the target-agnostic safe Rust core (`#[forbid(unsafe_code)]`).
   - **Host Target (`#[cfg(not(target_arch = "wasm32"))]` in [`rust/napi.rs`](rust/napi.rs)):** Compiles via `napi` (`napi10`) into `dist/dnam.<platform>-<arch>.node` with zero-copy `Float32Array`, `Float64Array`, and `Uint8Array` borrows plus libuv worker-pool `AsyncTask` support (`processAsync`).
   - **WASM Target (`#[cfg(target_arch = "wasm32")]` in [`rust/wasm.rs`](rust/wasm.rs)):** Compiles for `wasm32-unknown-unknown` (`--no-default-features`, omitting `testing`) with `.cargo/config.toml` flags (`-C target-feature=+simd128 -C link-arg=-zstack-size=786432 -C link-arg=--import-memory -C link-arg=--export-memory -C link-arg=--max-memory=268435456`), `--profile wasm-release` (`opt-level = 2`, `lto = "thin"`, `codegen-units = 16`, `panic = "abort"`, `strip = "symbols"`), and Binaryen `wasm-opt -O1` into `dist/dnam.wasm` (or `--profile profiling` into `dist/dnam.profiling.wasm`).
3. **Feature-Oriented TypeScript 7 & WebAudio `AudioWorklet` SDK (`src/`):**
   - Conditional `exports` in [`package.json`](package.json) route Node.js 26 LTS (`"node"` -> [`src/node.ts`](src/node.ts)) to `NapiDriver` ([`src/internal/native/napi.ts`](src/internal/native/napi.ts), with synchronous WASM fallback) and Browser/Bundler imports (`"browser"` -> [`src/browser.ts`](src/browser.ts)) to `WasmDriver` ([`src/internal/native/wasm.ts`](src/internal/native/wasm.ts)).
   - Feature directories: `src/model/`, `src/checkpoint/`, `src/engine/`, `src/parameters/`, `src/tempo/`, `src/schema/`, `src/provenance/`, `src/dsp/`, `src/web-audio/`, and private `src/internal/`.

---

## 2. Build, Profiling & Development Commands

Run all commands via **`moon`**:

```bash
# 1. Compile both the native Node.js 26 N-API v10 addon (.node) and WASM SIMD128 binary (.wasm)
moon run js:build-rust

# 2. Full build (Rust N-API + WASM + TypeScript ESM bundles + web-audio-processor.js + .d.ts)
moon run js:build

# 3. Build symbolicated profiling WASM (--profile profiling) and capture V8 --cpu-prof profiles
moon run js:profile
```

---

## 3. Strict Biome & Runtime Boundary Rules (`ref[sdk.ts.layout.dependency-direction]`)

Root [`biome.json`](../../biome.json) and [`tsconfig.json`](tsconfig.json) enforce strict typing and runtime isolation:

1. **Browser, Schema & Worklet Isolation:** `src/browser.ts`, `src/internal/native/wasm.ts`, `src/schema/**/*.ts`, and `src/web-audio/**/*.ts` must never import `node:*` built-in modules (`noRestrictedImports` in `biome.json`).
2. **Zero-Dependency Package & `AudioWorkletGlobalScope`:** `@dnam-project/dnam` has zero runtime npm dependencies (`"dependencies"` is omitted), and `src/web-audio/processor.ts` bundles into a standalone `dist/web-audio-processor.js` module.
3. **Strict TypeScript 7 (`tsc --noEmit`):** Maintain `strict: true`, `noUncheckedIndexedAccess: true`, `exactOptionalPropertyTypes: true`, and `verbatimModuleSyntax: true` with zero `any` or non-null assertions (`!`).

---

## 4. Testing, Coverage (≥ 95%) & Benchmarks (`vitest` v5 + `@vitest/browser-playwright` + `fast-check`)

Use **`fast-check`** (`fc.assert(fc.property(...))`) wherever possible across all Node (`src/**/*.test.ts`, `tests/integration/**/*.test.ts`) and headless Chromium (`src/**/*.browser.test.ts` via `@vitest/browser-playwright` with COOP/COEP `crossOriginIsolated` headers) test suites to verify snake_case $\leftrightarrow$ camelCase codec roundtrips, container header/arena integrity, 3-Phase parameter clamping, ZOH sample-rate rescaling, audio block processing stability, provenance signatures, WebAudio `OfflineAudioContext` + `AudioWorkletNode` shared-ring state transitions, and N-API vs WASM parity over arbitrary generated inputs.

```bash
# 1. Run all js quality gates (format-check, lint, typecheck, build, test)
moon check js

# 2. Run unit, fast-check property, and differential N-API vs WASM tests with >= 95% v8 coverage
moon run js:coverage

# 3. Run statistical Vitest 5 benchmarks
moon run js:bench
```
