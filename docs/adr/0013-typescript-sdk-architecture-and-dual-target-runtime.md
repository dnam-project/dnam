---
id: ADR-0013
title: "Idiomatic TypeScript 7 SDK Architecture, Dual-Target N-API + WASM SIMD128 Runtime & WebAudio Parameter Semantics"
status: accepted
date: "2026-11-04"
supersedes: null
superseded_by: null
specs:
  - "../specs/sdks/typescript_sdk.md"
  - "../specs/architecture/tora_architecture.md"
  - "../specs/containers/dnam_binary_format.md"
requirements:
  - REQ-2.4
  - REQ-2.5
  - REQ-5.5
  - REQ-6.1
  - REQ-7.1
  - REQ-8.1
  - REQ-8.7
  - REQ-8.8
---

# ADR-0013: Idiomatic TypeScript 7 SDK Architecture, Dual-Target N-API + WASM SIMD128 Runtime & WebAudio Parameter Semantics

This record documents the architectural and ergonomic design decisions for **`@dnam-project/dnam`** (`packages/js/`), the idiomatic TypeScript 7 SDK wrapping `packages/engine/` across Node.js 26 LTS (`N-API v10`) and WebAssembly `SIMD128` / WebAudio `AudioWorklet`, specified normatively in [`../specs/sdks/typescript_sdk.md`](../specs/sdks/typescript_sdk.md).

---

## 1. Context & Problem Statement

Following the TORA v2.0 engine implementation ([`ADR-0002`](0002-tri-operator-recurrent-architecture.md), [`ADR-0011`](0011-end-to-end-stage-wiring-and-arena-slice-topology.md)), the initial JavaScript/TypeScript package (`@dnam-project/engine`) exhibited six structural, ergonomic, and performance liabilities:

1. **Non-Idiomatic API Surface & Pythonic/C-Style Leakage (`ref[sdk.ts.api.resources-and-data]`, `ref[sdk.ts.parameters.audio-param-semantics]`):**
   Exposing redundant `Dnam*` class prefixes (`DnamEngine`, `DnamContainer`), positional boolean flags (`setBypass(true, true)`), snake_case metadata fields (`max_tier`, `controls_schema`), and untyped string control maps ignored standard TypeScript and Web platform conventions (`AudioParam`, `Disposable` / `using`, `ReadonlyMap`, `TransformStream`, `EventTarget`, and `AbortSignal`).
2. **Monolithic, Unmaintainable Driver & FFI Files (`ref[sdk.ts.layout.feature-modules]`, `ref[sdk.ts.layout.dependency-direction]`):**
   Concentration of all FFI glue into `rust/wasm_ffi.rs` (`1,548` lines), `rust/napi_bindings.rs` (`1,291` lines), `src/drivers/wasm.ts` (`2,159` lines), and `src/drivers/native.ts` (`1,250` lines) coupled container inspection, real-time audio processing, provenance cryptography, and DSP reference kernels into single files.
3. **Duplicated DSP, ZOH, Schema, JCS & Signing Logic in the Shim or Host Language (`ref[sdk.ts.runtime.dual-target-drivers]`):**
   Re-implementing `ZohRescaledParams`, preset application, control clamping, Lobatto calibration inversion, UI skew, RFC 8785 JCS canonicalization, or tier-prefix signature assembly inside `packages/js/rust/` or TypeScript (`src/`) caused compile-time breakage (`21` compiler errors) and behavioral divergence when `packages/engine` upgraded to TORA v2.0.
4. **Redundant Runtime JS Schema Libraries & Legacy Tri-Physics Surface (`ref[sdk.ts.schema.camel-case-codec-parity]`):**
   Maintaining a second copy of schema validation rules in JavaScript (`zod`) duplicated `dnam_engine::schema::DnamJsonHeader`, retained superseded Tri-Physics definitions (`SelectiveRgLru`, `ModulatedUfdn`, `Core1Config`, `Core2Config`), and added ~41 KB of unnecessary JS bundle weight.
5. **WASM Binding Overhead & `AudioWorkletGlobalScope` Constraints (`ref[sdk.ts.web-audio.realtime-bridge]`, `ref[sdk.ts.perf.profiling-budgets]`):**
   High-frequency `128`-frame WebAudio render quanta (`375 blocks/s` at `48 kHz`) require zero per-block heap allocations and zero JS-to-WASM copy overhead inside `AudioWorkletGlobalScope`, where DOM APIs (`TextDecoder`, `fetch`, `URL`) are unavailable.
6. **Stale TypeScript 5.x & Node.js 24 Toolchain Baseline:**
   Pinning TypeScript `5.8.x` and Node.js `24` missed the native TypeScript `7.0` compiler (`@typescript/typescript-*`) and Node.js `26 LTS` native `using` / `Symbol.dispose`, `Float16Array`, and `N-API v10` capabilities.

---

## 2. Considered Options & Rejected Candidates

| Candidate Approach | Why It Fails | Quantified Failure / Maintenance Penalty |
| :--- | :--- | :--- |
| **Mirroring the Python SDK API (`import dnam` Snake-Case Facets & `__call__` Conventions)** | Forces Pythonic idioms (mutable mapping indexing, generator context managers, snake_case metadata properties) onto TypeScript consumers instead of leveraging `AudioParam`, `using` / `Symbol.dispose`, camelCase types, and `TransformStream`. | Breaks ergonomic consistency with WebAudio (`AudioWorkletNode`, `AudioParam`) and standard TypeScript libraries; requires callers to convert casing at every UI boundary. |
| **`wasm-bindgen` Codegen for the WebAssembly Target (`ref[sdk.ts.runtime.dual-target-drivers]`)** | Emits JS glue that relies on `TextDecoder`/`TextEncoder` and generated wrapper objects, complicating standalone `AudioWorkletGlobalScope` bundling and adding per-call marshaling overhead on the real-time audio path. | Increases bundle complexity in `AudioWorkletGlobalScope` and prevents direct zero-copy in-place processing over pre-allocated linear-memory scratch regions. |
| **WASM-Only Runtime Across Node.js & Browsers (Dropping Native `N-API v10`)** | Abandons native `ARM64 NEON` / `Apple SME` and `x86_64 AVX2+FMA` hardware execution in Node.js 26 LTS, CLI batch tools, and Tauri desktop hosts. | Sacrifices `2x–4x` native SIMD throughput and prevents true multi-threaded libuv worker-pool execution (`processAsync`). |
| **Re-Implementing Engine Math, Schema Validation, JCS, Calibration, or Signing in TypeScript (`ref[sdk.ts.runtime.dual-target-drivers]`)** | Duplicates `dnam-engine` logic (such as Zod/Valibot/TypeBox schema rules, RFC 8785 JCS canonicalization, Lobatto calibration inversion, UI power-skew, or tier-prefix SHA-256 signing) in JS/TS when both N-API and WASM already link `dnam-engine`. | Guarantees behavioral drift from `packages/engine` and bloats the JS bundle with redundant validation and math code. |
| **Exposing Raw Snake-Case Wire JSON Directly as TypeScript Types (`ref[sdk.ts.schema.camel-case-codec-parity]`)** | Leaks Rust/JSON wire naming (`controls_schema`, `safe_clamped_range`, `stage3_cabinet_kautz`) throughout TypeScript application code and React props. | Violates idiomatic TypeScript camelCase conventions (`ref[repo.code.type-safety-and-imports]`) across `@dnam-project/ui` and `@dnam-project/studio`. |

---

## 3. Decision Outcome

We establish the **`@dnam-project/dnam` v2 TypeScript 7 SDK architecture** ([`../specs/sdks/typescript_sdk.md`](../specs/sdks/typescript_sdk.md)):

1. **Idiomatic TypeScript-First Design (`ref[sdk.ts.api.resources-and-data]`, `ref[sdk.ts.api.literal-unions-and-options]`, `ref[sdk.ts.api.disposable-scopes]`):**
   - Separate **plain immutable data** (deeply frozen camelCase objects validated by native Rust `serde` in `dnam-engine`: `ModelHeader`, `ParameterDescriptor`, `Preset`, `TelemetrySnapshot`) from **stateful native resources** (`Model`, `Checkpoint`, `Engine`, `Parameter`) that expose both explicit `.dispose()` and `[Symbol.dispose]()` (`Disposable`) with a `FinalizationRegistry` safety net.
   - Emit `/// <reference lib="esnext.disposable" />` in entry declaration files and provide both callback helpers (`engine.withBypass(opts, fn)`, `engine.withFreeze(opts, fn)`), `.restore()` handles, and `using` scopes so downstream consumers on `ES2024`/`ES2025` need zero `tsconfig.json` changes even with `skipLibCheck: false`.
   - Use string-literal unions (`Tier = "lite" | "full" | "pro" | "ultra"`, `Backend`, `Subdivision`, `ErrorCode`) with companion readonly tuple arrays (`tiers`, `backends`, `subdivisions`) instead of TypeScript `enum`, and accept options objects rather than positional boolean flags.
   - Use a single coded error class `DnamError` (`code: ErrorCode` such as `"ERR_DNAM_CONTAINER"`, `"ERR_DNAM_PARAMETER"`, `"ERR_DNAM_PROVENANCE"`, `"ERR_DNAM_BUSY"`) with a discriminated `detail` union and `isDnamError(err, code?)` type guard (`ref[sdk.ts.errors.coded-error]`).
2. **WebAudio `AudioParam`-Aligned Parameter API & Compile-Time Model Inference (`ref[sdk.ts.parameters.audio-param-semantics]`, `ref[sdk.ts.api.typed-model-inference]`):**
   - Expose live parameters on both `Engine` and `DnamAudioNode` via `engine.parameters.get(id)`, returning a typed `Parameter<TId, TValue>` object with `.value` (3 kHz smoothed), `.setValue(value, { immediate })`, `.reset()`, `.minValue`, `.maxValue`, `.defaultValue`, `.unit`, `.options`, and `.clamped`, plus batch updates via `engine.parameters.set(values, { immediate })`.
   - Support compile-time parameter ID and switch-option narrowing via `defineModelTypes<const T>()` and `Model.from<TParams>(bytes)` so invalid parameter names or switch values fail at compile time under TypeScript 7.
3. **Dual-Target `N-API v10` + Safe `extern "C"` `WASM SIMD128` Behind One Private `NativeDriver` Seam (`ref[sdk.ts.runtime.dual-target-drivers]`, `ref[sdk.ts.runtime.zero-unsafe]`, `ref[sdk.ts.api.sync-init]`):**
   - Compile `packages/js/rust/` (`dnam-js`) under `#![deny(unsafe_code)]` (with `#[forbid(unsafe_code)]` on `core`, `error`, and `wasm::abi` and zero `unsafe { ... }` blocks anywhere in the crate) to both a native Node.js 26 `N-API v10` addon (`dist/dnam.<platform>-<arch>.node`) and a `wasm32-unknown-unknown` `+simd128` module (`dist/dnam.wasm`) sharing `100%` of target-agnostic Rust core logic in `rust/core/` (`DynamicDnamEngine<'static>`).
   - Provide a **uniform synchronous API** across Node.js and browsers: `src/node.ts` loads the N-API addon synchronously (with WASM fallback), while `src/browser.ts` instantiates the WASM module via top-level `await` (with optional `initWasm({ module | bytes | url | memory | initialPages | maximumPages | shared })` pre-initialization).
   - Link `dist/dnam.wasm` with `-C link-arg=-zstack-size=786432 -C link-arg=--import-memory -C link-arg=--export-memory -C link-arg=--max-memory=268435456` and instantiate it with `new WebAssembly.Memory({ initial: 18, maximum: 4096, shared })` via `{ env: { memory } }` (`1.125 MiB` initial above the `15`-page module minimum from the `768 KiB` shadow stack + static data, `256 MiB` default maximum for V8 compile-time bounds-check elision and in-place virtual page commits, plus optional `shared: true` `SharedArrayBuffer` linear memory verified in headless Chromium via `@vitest/browser-playwright`), keeping the WASM ABI a zero-dependency `extern "C"` handle ABI (`wasm::abi` under `#[forbid(unsafe_code)]`) with persistent pre-allocated 64-byte-aligned linear-memory audio scratch buffers (`CacheLine64` + `zerocopy`) and cached `Float32Array` views (`ref[sdk.ts.engine.zero-copy-and-async]`, `ref[sdk.ts.perf.profiling-budgets]`).
4. **Feature-Oriented Package Layout & Enforced Dependency Direction (`ref[sdk.ts.layout.feature-modules]`, `ref[sdk.ts.layout.dependency-direction]`):**
   - Organize `packages/js/src/` into focused kebab-case feature directories (`model/`, `checkpoint/`, `engine/`, `parameters/`, `tempo/`, `schema/`, `provenance/`, `dsp/`, `web-audio/`, and private `internal/`), enforcing one-way imports via Biome `noRestrictedImports` rules and locking public subpath exports via snapshot tests (`ref[sdk.ts.layout.curated-exports]`).
5. **Zero-Dependency Bijective CamelCase/Snake-Case Schema Codec Backed by Native Rust `serde` & Verified by RFC 8785 JCS Round-Trip Parity (`ref[sdk.ts.schema.camel-case-codec-parity]`):**
   - Delegate all JSON schema validation, default population, and invariant checks directly to compiled Rust `serde` (`dnam_engine::schema::DnamJsonHeader`) across both `N-API v10` and `WASM SIMD128`, eliminating runtime JS validation libraries (`0` npm runtime dependencies, `0 KB` JS validator bundle, `100%` CSP-safe, and impossible to drift from `packages/engine/src/schema/`).
   - Translate between the snake_case `.dnam` wire JSON format and frozen camelCase TypeScript `ModelHeader` interfaces via a deterministic codec (`decodeHeader` / `encodeHeader`) verified against Rust-emitted reference and fully populated headers for byte-identical RFC 8785 JCS output and identical `immutable_header_sha256` digests.
6. **Symbolicated WASM Profiling (`--profile profiling`) & 4-Lane `WasmSimd128Backend` Vectorization (`ref[sdk.ts.perf.profiling-budgets]`, `ref[arch.fp.callback-env]`, `ref[arch.simd.lane-packing]`):**
   - Building `dnam-js` with `--profiling` (`moon run js:profile`) compiles `wasm32-unknown-unknown` under Cargo's `[profile.profiling]` (`debug = "line-tables-only"`, `lto = "thin"`, `codegen-units = 16`, `strip = "none"`) into `dist/dnam.profiling.wasm` without overwriting the stripped release `dist/dnam.wasm` (`[profile.wasm-release]`, `opt-level = 2`, `lto = "thin"`, `codegen-units = 16`, `panic = "abort"`, `strip = "symbols"` + Binaryen `wasm-opt -O1`), preserving the WebAssembly `"name"` custom section so V8 `--cpu-prof` (or `wasmtime` + `samply`) attributes samples directly to mangled Rust symbols inside `dnam_engine`.
   - Profiling `DnamEngine<WasmSimd128Backend>::process_block` identified scalar bottlenecks on `wasm32` where LLVM could not auto-vectorize scalar `f32::clamp` loops, scalar `fp16 <-> f32` delay conversions, or per-lane software subnormal flushes (`flush_recursive_state_f32`). Vectorizing `WasmSimd128Backend::f32x4_to_fp16x4`, `fp16x4_to_f32x4`, and `conv6_taps_4groups`, plus 4-lane `SimdVec4` execution in `MultiDrivePolyAdaaChannel`, `ToneBasisFilters`, `Stage3CabinetModals::step_fir_stereo_with_delayed_tap`, `KautzModalBank::step_4lane_stereo_simd_splatted`, `Stage4SpatialTank::coercive_saturate_4ch` / DC blockers, `ControlBus::step_micro_ramp`, and L1i-resident `#[inline(never)]` `step_frame_tora` kernels, achieves `84×–54×` real-time throughput in WASM `SIMD128` (`123×–78×` in Node.js N-API):

| Target | Tier | Backend | Mean (`us`) | P50 (`us`) | P99 (`us`) | Real-Time Speedup (`2,666.7 us` Budget) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| Node.js 26 N-API v10 | `lite` (`64-D`) | `neon` | 21.73 | 21.33 | 30.04 | 122.7× |
| Node.js 26 N-API v10 | `full` (`128-D`) | `neon` | 25.40 | 24.92 | 35.67 | 105.0× |
| Node.js 26 N-API v10 | `pro` (`192-D`) | `neon` | 28.03 | 27.71 | 35.46 | 95.1× |
| Node.js 26 N-API v10 | `ultra` (`256-D`) | `neon` | 34.32 | 33.37 | 54.04 | 77.7× |
| WASM SIMD128 | `lite` (`64-D`) | `wasm-simd128` | 31.55 | 30.50 | 56.79 | 84.5× |
| WASM SIMD128 | `full` (`128-D`) | `wasm-simd128` | 37.66 | 36.42 | 62.17 | 70.8× |
| WASM SIMD128 | `pro` (`192-D`) | `wasm-simd128` | 42.87 | 41.46 | 68.04 | 62.2× |
| WASM SIMD128 | `ultra` (`256-D`) | `wasm-simd128` | 49.05 | 48.00 | 68.92 | 54.4× |

