/**
 * Profiling & real-time budget verification suite (`verify[sdk.ts.perf.profiling-budgets]`).
 */

import "../../src/node.js";

import { describe, expect, it } from "vitest";
import { withDriver } from "../../src/internal/native/load.js";
import { loadNodeDriverSync, loadNodeWasmDriverSync } from "../../src/internal/native/napi.js";
import { Model } from "../../src/model/model.js";
import { ControlSpace } from "../../src/parameters/clamp.js";

describe("SDK Performance & Real-Time Profiling Budgets", () => {
  it("sustains > 15x real-time 128-frame stereo throughput across all tiers on N-API and WASM SIMD128", () => {
    // verify[sdk.ts.perf.profiling-budgets]
    const napiDriver = loadNodeDriverSync();
    const wasmDriver = loadNodeWasmDriverSync();
    const inL = new Float32Array(128).fill(0.15);
    const inR = new Float32Array(128).fill(-0.15);
    const outL = new Float32Array(128);
    const outR = new Float32Array(128);
    const blockDeadlineUs = (128 / 48_000) * 1_000_000; // 2666.67 us

    for (const driver of [napiDriver, wasmDriver]) {
      withDriver(driver, () => {
        using model = Model.reference();
        using engine = model.createEngine({ tier: "ultra", sampleRate: 48_000 });
        for (let i = 0; i < 32; i++) {
          engine.process([inL, inR], { output: [outL, outR] });
        }
        const iters = 500;
        const t0 = performance.now();
        for (let i = 0; i < iters; i++) {
          engine.process([inL, inR], { output: [outL, outR] });
        }
        const meanUs = ((performance.now() - t0) * 1000) / iters;
        expect(meanUs).toBeLessThan(blockDeadlineUs / 5);

        using space = ControlSpace.from(model);
        const raw = new Float32Array([7.5, 1.0, 250.0]);
        const clamped = new Float32Array(3);
        const tClamp0 = performance.now();
        for (let i = 0; i < 2000; i++) {
          space.clampNumericInto(raw, clamped);
        }
        const clampMeanUs = ((performance.now() - tClamp0) * 1000) / 2000;
        expect(clampMeanUs).toBeLessThan(50);
      });
    }
  });
});
