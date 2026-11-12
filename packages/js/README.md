# `@dnam-project/dnam` — Idiomatic TypeScript 7, Node.js 26 N-API v10 & WebAudio SDK

**`@dnam-project/dnam`** is the official TypeScript 7.0, Node.js 26 LTS (`N-API v10`), and Browser WebAssembly `SIMD128` / WebAudio `AudioWorklet` SDK for the **Dynamic Neural Audio Modeler (DNAM / TORA v2.0)** architecture ([`docs/specs/sdks/typescript_sdk.md`](../../docs/specs/sdks/typescript_sdk.md), [`ADR-0013`](../../docs/adr/0013-typescript-sdk-architecture-and-dual-target-runtime.md)):

1. **Zero Runtime Dependencies (`0 KB` JS validator overhead):** 100% of DSP, ZOH sample-rate rescaling, 3-Phase control clamping, Lobatto calibration inversion, `.dnam` schema validation, RFC 8785 JCS canonicalization, `uplo='U'` Gram packing, and `Ed25519`/`X25519`/`ChaCha20-Poly1305` provenance logic delegates directly to compiled Rust (`dnam-engine` and `dnam-capture`).
2. **Unified Dual-Target Rust Core (`dnam-js`):**
   - **Node.js 26 LTS (`N-API v10`):** Zero-copy `Float32Array`, `Float64Array`, and `Uint8Array` slices backed by hardware `ARM64 NEON` / `x86_64 AVX2+FMA` SIMD plus libuv worker-pool `processAsync(...)`.
   - **Browser / Edge (`wasm32-unknown-unknown` `+simd128`):** Synchronous and streaming WASM instantiation (`initWasm`) with persistent pre-allocated linear-memory scratch rings (`0` per-block allocations).
3. **Idiomatic TypeScript-First Ergonomics:**
   - Deeply frozen `camelCase` plain objects (`model.header.ioConfig.outputMode`, `model.presets`) paired with explicit `Disposable` (`using` / `.dispose()` + `FinalizationRegistry`) native handles (`Model`, `Engine`, `ControlSpace`, `DnamAudioNode`).
   - WebAudio `AudioParam`-style `Parameter` controllers (`engine.parameters.get("gain")`) and `TempoController` (`engine.tempo.sync({ bpm: 120, subdivision: "1/8." })`).
   - Compile-time typed parameter maps via `defineModelTypes(...)`.
4. **Real-Time WebAudio `AudioWorklet` (`@dnam-project/dnam/web-audio`):**
   - `DnamAudioNode<TParams>` and standalone `dist/web-audio-processor.js` using a lock-free `SharedArrayBuffer` (`Atomics`) parameter & seqlock telemetry ring when `crossOriginIsolated === true` and automatic `MessagePort` fallback otherwise.

---

## 1. Quick Start

### Node.js / Server-Side Real-Time & Offline Rendering

```typescript
import { defineModelTypes, Model } from "@dnam-project/dnam";

const RigSchema = defineModelTypes([
  { id: "gain", type: "range" },
  { id: "boost", type: "switch", declaredOptions: ["off", "on", "ultra"] },
  { id: "delay_ms", type: "range" },
] as const);

using model = Model.reference<typeof RigSchema.$infer>({ supportsFreeze: true });
using engine = model.createEngine({ sampleRate: 48_000, tier: "full" });

// AudioParam-style parameter updates with automatic 3-Phase clamping
const gain = engine.parameters.get("gain");
gain.setValue(7.5, { immediate: true });
engine.tempo.sync({ bpm: 120, subdivision: "1/8." });

// Zero-allocation 128-frame stereo processing
const leftIn = new Float32Array(128).fill(0.25);
const rightIn = new Float32Array(128).fill(-0.25);
const leftOut = new Float32Array(128);
const rightOut = new Float32Array(128);
engine.process([leftIn, rightIn], { output: [leftOut, rightOut] });

// Scoped Stage IV hardware freeze (`using` restores state on scope exit)
{
  using _freeze = engine.freeze({ amount: 0.85 });
  engine.process([leftIn, rightIn], { output: [leftOut, rightOut] });
}

// Asynchronous libuv worker-pool processing & 4x oversampled studio bounce
const [asyncL, asyncR] = await engine.processAsync([leftIn, rightIn]);
const [bounceL, bounceR] = engine.bounce([leftIn, rightIn], { maxTier: true });
```

### Browser WebAudio `AudioWorklet`

```typescript
import { initWasm, Model } from "@dnam-project/dnam";
import { DnamAudioNode } from "@dnam-project/dnam/web-audio";

await initWasm(); // Optional pre-initialization in browsers
const model = await Model.load("/models/studio_rig.dnam");

const audioCtx = new AudioContext({ sampleRate: 48_000 });
using dnamNode = await DnamAudioNode.create(audioCtx, model, { tier: "full" });

dnamNode.connect(audioCtx.destination);
dnamNode.parameters.get("gain").setValue(6.5);
dnamNode.tempo.set({ bpm: 128, subdivision: "1/8" });
```

---

## 2. Subpath Exports

| Subpath | Environment | Description |
| :--- | :--- | :--- |
| `@dnam-project/dnam` | Node.js & Browser | `Model`, `GramMatrix`, `uploUPackedLength`, `Engine`, `Parameter`, `ParameterMap`, `ControlSpace`, `clampParameters`, `defineModelTypes`, `TempoController`, `decodeHeader`/`encodeHeader`/`headerToJcs`, `DnamError`, `initWasm` |
| `@dnam-project/dnam/parameters` | Universal | `Parameter`, `ParameterMap`, `ControlSpace`, `clampParameters`, `defineModelTypes` |
| `@dnam-project/dnam/tempo` | Universal | `TempoController`, `subdivisions`, `computeZoh`, `controlStride`, `controlTickRatio`, `subdivisionPhase`, `evalCalibration3Node`, `invertTempoCalibration` |
| `@dnam-project/dnam/schema` | Universal | `ModelHeader`, `decodeHeader`, `encodeHeader`, `headerToJcs`, `canonicalJson`, `parseSchemaVersion`, and stage/parameter TypeScript interfaces |
| `@dnam-project/dnam/provenance` | Universal | `verify`, `enforcePolicy`, `signCreatorHeader`, `signMarketplaceReceipt`, `seal`, `unseal`, `encodeDau1`/`decodeDau1`, `applyGaugeWatermark`, `applyTardosWatermark`, `scoreTardosWatermark` |
| `@dnam-project/dnam/dsp` | Universal | Reference TORA DSP kernels (`catmullRomHermiteSlice`, `tankSlice`, `modalSlice`, `unitRotorSlice`, `polyAdaaSlice`, `lobattoBasisSlice`, `rank4FmSlice`, `chebyshevLfoSlice`, `trustRegionSoftclipSlice`, `givensHouseholder8chSlice`, `givensHouseholder16chSlice`) |
| `@dnam-project/dnam/web-audio` | Browser / Universal | `DnamAudioNode`, `DnamAudioWorkletProcessor`, `WorkletSharedState`, `registerDnamProcessor`, `isSharedMemorySupported` |
| `@dnam-project/dnam/web-audio/processor` | `AudioWorkletGlobalScope` | Standalone zero-dependency worklet processor bundle (`dist/web-audio-processor.js`) |
| `@dnam-project/dnam/dnam.wasm` | Universal | Compiled `wasm32-unknown-unknown` `+simd128` release binary (`dist/wasm/release/dnam.wasm`) |

---

## 3. Measured Real-Time Performance (48 kHz, 128-Frame Stereo Block = `2,666.7 us` Budget)

| Target | Tier | Backend | Mean (`us`) | P50 (`us`) | P99 (`us`) | Real-Time Speedup |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| Node.js 26 N-API v10 | `lite` (64-D) | `neon` | 21.73 | 21.33 | 30.04 | 122.7× |
| Node.js 26 N-API v10 | `full` (128-D) | `neon` | 25.40 | 24.92 | 35.67 | 105.0× |
| Node.js 26 N-API v10 | `pro` (192-D) | `neon` | 28.03 | 27.71 | 35.46 | 95.1× |
| Node.js 26 N-API v10 | `ultra` (256-D) | `neon` | 34.32 | 33.37 | 54.04 | 77.7× |
| WASM SIMD128 | `lite` (64-D) | `wasm-simd128` | 31.55 | 30.50 | 56.79 | 84.5× |
| WASM SIMD128 | `full` (128-D) | `wasm-simd128` | 37.66 | 36.42 | 62.17 | 70.8× |
| WASM SIMD128 | `pro` (192-D) | `wasm-simd128` | 42.87 | 41.46 | 68.04 | 62.2× |
| WASM SIMD128 | `ultra` (256-D) | `wasm-simd128` | 49.05 | 48.00 | 68.92 | 54.4× |

---

## 4. Development, Profiling & Verification

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`examples/`](examples/) for full build, test, coverage, and profiling instructions:

```bash
moon check js           # format-check, lint, typecheck, build, test
moon run js:coverage    # >= 95% v8 coverage (unit + fast-check + N-API/WASM parity)
moon run js:bench       # Vitest benchmarks across tiers and drivers
moon run js:profile     # Symbolicated WASM (--profile profiling) + V8 --cpu-prof
```

