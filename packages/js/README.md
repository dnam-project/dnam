# `@dnam-project/engine` — Node.js N-API v8, WebAssembly SIMD128 & WebAudio SDK

**`@dnam-project/engine`** is the official TypeScript, Node.js, and Browser WebAudio SDK for the **Dynamic Neural Audio Modeler (DNAM / "Tri-Physics")** architecture, wrapping the native Rust SIMD engine ([`packages/engine/`](../engine/README.md)) in a unified dual-target package:

1. **Node.js / Bun / Deno Native N-API v8 (`napi-rs` v3):** Zero-copy `Float32Array`, `Float64Array`, and `Uint8Array` slices backed by hardware `ARM64 NEON` / `x86_64 AVX2+FMA` SIMD and libuv worker-pool `processAsync(...)`.
2. **Browser / Edge WebAssembly SIMD128 (`dnam_engine.wasm`):** Zero-dependency `wasm32-unknown-unknown` `+simd128` execution with persistent pre-allocated linear-memory audio arenas and `FinalizationRegistry` + `Symbol.dispose` deterministic cleanup.
3. **WebAudio `AudioWorklet` Integration (`@dnam-project/engine/worklet`):** Real-time 128-sample quantum processing via `DnamAudioWorkletProcessor` and `DnamWorkletController<TControls>`, using a lock-free 256-byte `SharedArrayBuffer` (`Atomics`) control/telemetry bridge when `crossOriginIsolated === true` and automatic `MessagePort` fallback when `crossOriginIsolated === false`.
4. **Zod v4 + Standard Schema v1 (`@dnam-project/engine/schema`):** Bidirectional `.dnam` wire `snake_case` $\leftrightarrow$ idiomatic TypeScript `camelCase` validation implementing `@standard-schema/spec` (including `ModelCapabilities` `supportsFreeze`/`freezeMutesDry`, `ControlCalibrationRole` granular/freeze roles, and `.dnam-ckpt` `ExtractedPhysics` grain/pitch telemetry).

---

## 1. Quick Start

### Node.js / Server-Side Batch & Real-Time Processing

```typescript
import { DnamContainer, DnamEngine } from "@dnam-project/engine";

interface AmpControls {
  gain: number;
  boost: "off" | "on" | "ultra";
  [key: string]: number | string;
}

const container = DnamContainer.createReference(true);
using engine = new DnamEngine<AmpControls>(container, {
  tier: "full",
  sampleRateHz: 48_000,
});

// Update parametric controls with 3-Phase clamping
const clampSnap = engine.setControls({ gain: 7.5, boost: "on" });

// Synchronous zero-copy stereo processing
const leftIn = new Float32Array(128);
const rightIn = new Float32Array(128);
const [leftOut, rightOut] = engine.processStereo(leftIn, rightIn);

// Asynchronous libuv worker-pool processing
const [asyncL, asyncR] = await engine.processAsync(leftIn, rightIn);
```

### Browser WebAudio `AudioWorklet`

```typescript
import { DnamWorkletController } from "@dnam-project/engine";

const audioCtx = new AudioContext({ sampleRate: 48_000 });
const response = await fetch("/models/darkglass_b7k.dnam");
const modelBytes = new Uint8Array(await response.arrayBuffer());

const controller = await DnamWorkletController.createNode(audioCtx, modelBytes, {
  tier: "full",
  workletModuleUrl: new URL("@dnam-project/engine/worklet-processor", import.meta.url),
});

controller.node.connect(audioCtx.destination);
controller.setControls({ gain: 6.5, boost: "on" });
const telemetry = controller.telemetry();
```

---

## 2. Subpath Exports

| Subpath | Environment | Description |
| :--- | :--- | :--- |
| `@dnam-project/engine` | Node.js & Browser | Conditional entry point (`node` → N-API v8 + WASM; `browser` → WASM SIMD128) |
| `@dnam-project/engine/node` | Node.js / Bun / Deno | Explicit Node.js N-API v8 + synchronous WASM driver |
| `@dnam-project/engine/browser` | Browser / Edge | Pure browser entry point (zero `node:*` imports) |
| `@dnam-project/engine/schema` | Universal | Zod v4 + `@standard-schema/spec` v1 schemas & wire serializers |
| `@dnam-project/engine/worklet` | Universal | `DnamWorkletController`, `DnamSharedBridge`, and `DnamAudioWorkletProcessor` |
| `@dnam-project/engine/worklet-processor` | `AudioWorkletGlobalScope` | Self-contained zero-dependency bundle for `audioWorklet.addModule(...)` |
| `@dnam-project/engine/wasm` | Universal | Raw `dnam_engine.wasm` (`wasm32-unknown-unknown` `+simd128`) binary asset |

---

## 3. Development & Verification

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`AGENTS.md`](AGENTS.md) for full build, lint, test, and benchmark instructions:

```bash
pnpm --filter @dnam-project/engine build
pnpm --filter @dnam-project/engine lint
pnpm --filter @dnam-project/engine typecheck
pnpm --filter @dnam-project/engine coverage
pnpm --filter @dnam-project/engine bench
```
