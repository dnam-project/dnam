#!/usr/bin/env node
/**
 * End-to-end performance & memory profiler for `@dnam-project/dnam`
 * (`ref[sdk.ts.perf.profiling-budgets]`).
 *
 * Run with V8 CPU & heap profiling via:
 *   moon run js:profile
 */

import { readFileSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";
import * as nodeSdk from "../dist/node.js";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const pkgRoot = resolve(scriptDir, "..");

const refModel = nodeSdk.Model.reference();
const refBytes = refModel.toBytes();
const inL = new Float32Array(128).fill(0.2);
const inR = new Float32Array(128).fill(-0.2);
const outL = new Float32Array(128);
const outR = new Float32Array(128);

const BLOCK_DURATION_US = (128 / 48_000) * 1_000_000; // 2666.67 us

function measureBlockLatency(sdk, tier, iterations = 2_000) {
  using model = sdk.Model.from(refBytes);
  using engine = model.createEngine({ tier, sampleRate: 48_000 });
  for (let i = 0; i < 2_000; i++) {
    engine.process([inL, inR], { output: [outL, outR] });
  }
  const samplesUs = new Float64Array(iterations);
  for (let i = 0; i < iterations; i++) {
    const t0 = performance.now();
    engine.process([inL, inR], { output: [outL, outR] });
    samplesUs[i] = (performance.now() - t0) * 1000;
  }
  samplesUs.sort();
  const sum = samplesUs.reduce((a, b) => a + b, 0);
  const meanUs = sum / iterations;
  const p50Us = samplesUs[Math.floor(iterations * 0.5)] ?? 0;
  const p99Us = samplesUs[Math.floor(iterations * 0.99)] ?? 0;
  const realtimeFactor = BLOCK_DURATION_US / meanUs;
  return { backend: engine.backend, meanUs, p50Us, p99Us, realtimeFactor };
}

console.log("=== @dnam-project/dnam Performance & Profiling Summary (48 kHz, 128 frames) ===");
console.log("| Target | Tier | Backend | Mean (us) | P50 (us) | P99 (us) | Real-Time Speedup |");
console.log("| :--- | :--- | :--- | ---: | ---: | ---: | ---: |");

for (const tier of nodeSdk.tiers) {
  const res = measureBlockLatency(nodeSdk, tier);
  console.log(
    `| Node.js 26 N-API v10 | \`${tier}\` | \`${res.backend}\` | ${res.meanUs.toFixed(2)} | ${res.p50Us.toFixed(2)} | ${res.p99Us.toFixed(2)} | ${res.realtimeFactor.toFixed(1)}x |`,
  );
}

// Measure ControlSpace clamp & Header JCS throughput on N-API
using space = nodeSdk.ControlSpace.from(refModel);
const rawVec = new Float32Array([8.5, 1.0, 350.0]);
const clampedVec = new Float32Array(3);
const clampIters = 20_000;
const tClamp0 = performance.now();
for (let i = 0; i < clampIters; i++) {
  space.clampNumericInto(rawVec, clampedVec);
}
const clampMeanUs = ((performance.now() - tClamp0) * 1000) / clampIters;

const jcsIters = 1_000;
const tJcs0 = performance.now();
for (let i = 0; i < jcsIters; i++) {
  const hdr = nodeSdk.decodeHeader(refModel.rawHeaderJson);
  nodeSdk.encodeHeader(hdr);
  nodeSdk.headerToJcs(hdr);
}
const jcsMeanUs = ((performance.now() - tJcs0) * 1000) / jcsIters;

// Now switch active driver to WasmDriver via initWasm and measure WASM SIMD128
const wasmBytes = readFileSync(resolve(pkgRoot, "dist/dnam.wasm"));
await nodeSdk.initWasm({ bytes: wasmBytes });
for (const tier of nodeSdk.tiers) {
  const res = measureBlockLatency(nodeSdk, tier);
  console.log(
    `| WASM SIMD128 | \`${tier}\` | \`${res.backend}\` | ${res.meanUs.toFixed(2)} | ${res.p50Us.toFixed(2)} | ${res.p99Us.toFixed(2)} | ${res.realtimeFactor.toFixed(1)}x |`,
  );
}

const wasmSize = statSync(resolve(pkgRoot, "dist/dnam.wasm")).size;
console.log("\n=== Auxiliary Operations & Artifact Sizes ===");
console.log(`- ControlSpace.clampNumericInto (3-Phase): ${clampMeanUs.toFixed(3)} us/call`);
console.log(`- decodeHeader + encodeHeader + headerToJcs: ${jcsMeanUs.toFixed(2)} us/roundtrip`);
console.log(`- dist/dnam.wasm binary size: ${(wasmSize / 1024).toFixed(1)} KiB`);
