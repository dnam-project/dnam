# Contributing to `@dnam-project/engine` (`packages/js/`)

This guide covers engineering, dual-target Rust compilation (Node.js 24/26 LTS N-API v8 + WebAssembly SIMD128), strict Biome linting, TypeScript 5.8 typing, WebAudio `AudioWorklet` runtime boundary rules, testing, coverage, and performance benchmarking standards for **`@dnam-project/engine`** ([`README.md`](README.md)).

All contributions must also comply with the root [`CONTRIBUTING.md`](../../CONTRIBUTING.md) and [`CLA.md`](../../CLA.md) (`git commit -s`).

---

## 1. Unified Dual-Target Architecture

Consult [`AGENTS.md`](AGENTS.md) for the module-to-spec discovery table:

1. **Unified Dual-Target Rust Core (`rust/` -> `dnam-js` crate):**
   - [`rust/engine.rs`](rust/engine.rs), [`rust/container.rs`](rust/container.rs), [`rust/dsp.rs`](rust/dsp.rs), and [`rust/provenance.rs`](rust/provenance.rs) implement the shared safe Rust core (`#![forbid(unsafe_code)]`) used by both Node.js and WebAssembly.
   - **Host Target (`#[cfg(not(target_arch = "wasm32"))]` in [`rust/napi_bindings.rs`](rust/napi_bindings.rs)):** Compiles via `napi = "3.15"` + `napi-derive = "3.6"` into `dist/dnam_js.<platform>-<arch>.node` with zero-copy `Float32Array`, `Float64Array`, and `Uint8Array` views plus libuv worker-pool `AsyncTask` support (`processAsync`).
   - **WASM Target (`#[cfg(target_arch = "wasm32")]` in [`rust/wasm_ffi.rs`](rust/wasm_ffi.rs)):** Compiles with `RUSTFLAGS="-C target-feature=+simd128"` into `dist/dnam_engine.wasm`.
2. **TypeScript & WebAudio `AudioWorklet` SDK (`src/`):**
   - Conditional `exports` in [`package.json`](package.json) automatically route Node.js/Bun/Deno imports (`"node"` condition -> [`src/node.ts`](src/node.ts)) to the native N-API v8 driver (with synchronous WASM fallback) and Browser/Bundler imports (`"browser"` / `"default"` condition -> [`src/browser.ts`](src/browser.ts)) to the pure WebAssembly SIMD128 driver.
   - [`src/worklet/`](src/worklet/index.ts) provides `DnamWorkletController<TControls>` and `DnamAudioWorkletProcessor`, using a 256-byte lock-free `SharedArrayBuffer` (`Atomics`) state/telemetry bridge when `crossOriginIsolated === true` and automatically falling back to `MessagePort` when `crossOriginIsolated === false`.

---

## 2. Build & Development Commands

Always use **`pnpm`** from the workspace root or scoped to `@dnam-project/engine`:

```bash
# 1. Compile both the native Node.js N-API v8 addon (.node) and WASM SIMD128 binary (.wasm)
pnpm --filter @dnam-project/engine build:rust

# 2. Bundle ESM entry points, self-contained worklet-processor.js, and .d.ts declarations
pnpm --filter @dnam-project/engine build:js

# 3. Full build (Rust + TypeScript)
pnpm --filter @dnam-project/engine build
```

---

## 3. Strict Agentic Biome & Runtime Boundary Rules

Root [`biome.json`](../../biome.json) and [`tsconfig.json`](tsconfig.json) enforce strict typing and runtime isolation:

1. **Browser & Schema Isolation:** `src/browser.ts`, `src/drivers/wasm.ts`, `src/schema/**/*.ts`, and `src/worklet/**/*.ts` must never import `node:*` built-in modules (`noRestrictedImports` in `biome.json`).
2. **Zero-Dependency `AudioWorkletGlobalScope`:** `src/worklet/processor.ts` and `src/worklet/shared_memory.ts` are bundled into the standalone `dist/worklet-processor.js` loaded via `audioContext.audioWorklet.addModule(...)` and must never import `"zod"` or `node:*`.
3. **Strict TypeScript (`tsc --noEmit`):** Maintain `strict: true`, `noUncheckedIndexedAccess: true`, `exactOptionalPropertyTypes: true`, and `verbatimModuleSyntax: true` with zero `any` or non-null assertions (`!`).

---

## 4. Testing, Coverage ($\ge 95\%$) & Benchmarks (`vitest` v5 + `fast-check`)

```bash
# 1. Run Biome linter and formatter checks
pnpm --filter @dnam-project/engine lint
pnpm --filter @dnam-project/engine format:check

# 2. Run strict TypeScript typechecker
pnpm --filter @dnam-project/engine typecheck

# 3. Run unit, fast-check property, and differential N-API vs WASM tests with >= 95% v8 coverage
pnpm --filter @dnam-project/engine coverage

# 4. Run statistical Vitest 5 benchmarks
pnpm --filter @dnam-project/engine bench
```
