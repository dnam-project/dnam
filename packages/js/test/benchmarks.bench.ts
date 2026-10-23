import { beforeAll, describe, test } from "vitest";
import {
  ControlSpace,
  DnamContainer,
  DnamEngine,
  extractStructuralHash160d,
  ModulatedUfdn,
  PolyAdaa,
  SelectiveRgLru,
} from "../src/index.js";
import { ensureWasmReady } from "./helpers.js";

describe("DNAM Engine & Kernel Benchmarks", () => {
  beforeAll(async () => {
    await ensureWasmReady();
  });

  test("DnamEngine 128-sample WebAudio Quantum (48 kHz = 2.667 ms budget)", async ({ bench }) => {
    const refContainer = DnamContainer.createReference(true, "native");
    const nativeEngineFull = new DnamEngine(refContainer, {
      tier: "full",
      sampleRateHz: 48_000,
      backend: "native",
    });
    const wasmEngineFull = new DnamEngine(refContainer.bytes, {
      tier: "full",
      sampleRateHz: 48_000,
      backend: "wasm",
    });

    const block128L = new Float32Array(128).fill(0.15);
    const block128R = new Float32Array(128).fill(-0.15);
    const out128L = new Float32Array(128);
    const out128R = new Float32Array(128);

    await bench.compare(
      bench("DnamEngine.processStereo (native N-API, Full Tier + M-UFDN)", () => {
        nativeEngineFull.processStereo(block128L, block128R, {
          leftOut: out128L,
          rightOut: out128R,
        });
      }),
      bench("DnamEngine.processStereo (WASM SIMD128, Full Tier + M-UFDN)", () => {
        wasmEngineFull.processStereo(block128L, block128R, {
          leftOut: out128L,
          rightOut: out128R,
        });
      }),
      { time: 50 },
    );

    nativeEngineFull.dispose();
    wasmEngineFull.dispose();
  });

  test("3-Phase Control Clamping & Reference DSP Kernels", async ({ bench }) => {
    const refContainer = DnamContainer.createReference(true, "native");
    const nativeSpace = new ControlSpace(refContainer.header, "native");
    const wasmSpace = new ControlSpace(refContainer.header, "wasm");

    const f64Block256 = new Float64Array(256).fill(0.2);
    const f64Out256 = new Float64Array(256);
    const adaaNative = new PolyAdaa({ bConv: 0.1 }, "native");
    const rglruNative = new SelectiveRgLru({ rLow: 0.9 }, "native");
    const ufdnNative = new ModulatedUfdn({ delaySamples: 32 }, "native");

    await bench.compare(
      bench("ControlSpace.clamp (native N-API)", () => {
        nativeSpace.clamp({ gain: 7.5, boost: "on" });
      }),
      bench("ControlSpace.clamp (WASM SIMD128)", () => {
        wasmSpace.clamp({ gain: 7.5, boost: "on" });
      }),
      bench("PolyAdaa.process (256 samples f64)", () => {
        adaaNative.process(f64Block256, { out: f64Out256 });
      }),
      bench("SelectiveRgLru.process (256 samples f64)", () => {
        rglruNative.process(f64Block256, { out: f64Out256 });
      }),
      bench("ModulatedUfdn.process (256 samples f64)", () => {
        ufdnNative.process(f64Block256, { out: f64Out256 });
      }),
      bench("extractStructuralHash160d (native N-API)", () => {
        extractStructuralHash160d(refContainer.arena, "native");
      }),
      { time: 50 },
    );

    nativeSpace.dispose();
    wasmSpace.dispose();
  });
});
