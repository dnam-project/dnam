/**
 * Headless Chromium browser benchmark suite (`@vitest/browser-playwright`) for WebAssembly
 * `SIMD128` (`--import-memory`, bounded & `shared: true` `WebAssembly.Memory`)
 * (`ref[sdk.ts.perf.profiling-budgets]`).
 */

import { describe, test } from "vitest";
import { initWasm, Model } from "../browser.js";
import { instantiateWasmDriver } from "../internal/native/wasm.js";
import { createWasmMemory } from "./index.js";

const unsharedMod = await initWasm({ initialPages: 32, maximumPages: 1024, shared: false });
const sharedMem = createWasmMemory({ initialPages: 32, maximumPages: 1024, shared: true });
const sharedMod = await initWasm({ memory: sharedMem, shared: true });

const refModel = Model.reference();
const refBytes = refModel.toBytes();

const inL = new Float32Array(128).fill(0.15);
const inR = new Float32Array(128).fill(-0.15);
const outL = new Float32Array(128);
const outR = new Float32Array(128);

const unsharedDriver = instantiateWasmDriver(unsharedMod, {
  initialPages: 32,
  maximumPages: 1024,
});
const sharedDriver = instantiateWasmDriver(sharedMod, {
  memory: sharedMem,
});

const unsharedLite = unsharedDriver.createEngine({
  containerBytes: refBytes,
  tier: 0,
  sampleRateHz: 48_000,
  backend: "auto",
});
const unsharedFull = unsharedDriver.createEngine({
  containerBytes: refBytes,
  tier: 1,
  sampleRateHz: 48_000,
  backend: "auto",
});
const unsharedPro = unsharedDriver.createEngine({
  containerBytes: refBytes,
  tier: 2,
  sampleRateHz: 48_000,
  backend: "auto",
});
const unsharedUltra = unsharedDriver.createEngine({
  containerBytes: refBytes,
  tier: 3,
  sampleRateHz: 48_000,
  backend: "auto",
});

const sharedLite = sharedDriver.createEngine({
  containerBytes: refBytes,
  tier: 0,
  sampleRateHz: 48_000,
  backend: "auto",
});
const sharedUltra = sharedDriver.createEngine({
  containerBytes: refBytes,
  tier: 3,
  sampleRateHz: 48_000,
  backend: "auto",
});

describe("Chromium Browser WASM SIMD128 128-frame stereo block processing (48 kHz)", () => {
  test("Bounded vs Shared WebAssembly.Memory across Lite..Ultra tiers", async ({ bench }) => {
    await bench.compare(
      bench("Chromium WASM SIMD128 Lite (64-D, bounded memory)", () => {
        unsharedLite.processStereo(inL, inR, outL, outR);
      }),
      bench("Chromium WASM SIMD128 Lite (64-D, shared memory)", () => {
        sharedLite.processStereo(inL, inR, outL, outR);
      }),
      bench("Chromium WASM SIMD128 Full (128-D, bounded memory)", () => {
        unsharedFull.processStereo(inL, inR, outL, outR);
      }),
      bench("Chromium WASM SIMD128 Pro (192-D, bounded memory)", () => {
        unsharedPro.processStereo(inL, inR, outL, outR);
      }),
      bench("Chromium WASM SIMD128 Ultra (256-D, bounded memory)", () => {
        unsharedUltra.processStereo(inL, inR, outL, outR);
      }),
      bench("Chromium WASM SIMD128 Ultra (256-D, shared memory)", () => {
        sharedUltra.processStereo(inL, inR, outL, outR);
      }),
      { time: 50 },
    );
  });
});
