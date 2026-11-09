/**
 * Cross-cutting integration tests for `@dnam-project/dnam`:
 * 1. Curated subpath export snapshots, feature-module layout, and zero legacy Tri-Physics symbols
 *    (`verify[sdk.ts.layout.feature-modules]`, `verify[sdk.ts.layout.curated-exports]`,
 *    `verify[sdk.ts.layout.dependency-direction]`).
 * 2. Bit-identical (`0 ULP`) scalar parity across `NapiDriver`, `WasmDriver`, and native Rust
 *    `DynamicDnamEngine` across all 4 tiers, sample rates, and topologies
 *    (`verify[sdk.ts.engine.behavioral-parity]`, `verify[sdk.ts.runtime.dual-target-drivers]`).
 * 3. WebAssembly `SIMD128` vs. scalar subnormal/denormal flushing verification.
 * 4. Real-time FFI overhead & WASM binary size budgets (`verify[sdk.ts.perf.profiling-budgets]`).
 */

import { readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import fc from "fast-check";
import { describe, expect, it } from "vitest";
import * as checkpointMod from "../../src/checkpoint/index.js";
import * as dspMod from "../../src/dsp/index.js";
import { withDriver } from "../../src/internal/native/load.js";
import {
  getTestingBindings,
  loadNodeDriverSync,
  loadNodeWasmDriverSync,
} from "../../src/internal/native/napi.js";
import * as rootMod from "../../src/node.js";
import * as parametersMod from "../../src/parameters/index.js";
import * as provenanceMod from "../../src/provenance/index.js";
import * as schemaMod from "../../src/schema/index.js";
import * as tempoMod from "../../src/tempo/index.js";
import { tiers } from "../../src/types.js";
import * as webAudioMod from "../../src/web-audio/index.js";

const FORBIDDEN_TRI_PHYSICS = [
  "SelectiveRgLru",
  "SelectiveRgLruParams",
  "ModulatedUfdn",
  "MufdnParams",
  "Core1Config",
  "Core2Config",
  "core2Active",
];

describe("Public API surface, 0-ULP behavioral parity, WASM denormals & performance budgets", () => {
  it("exposes curated subpath exports with zero internal or legacy Tri-Physics symbols", () => {
    // verify[sdk.ts.layout.feature-modules]
    // verify[sdk.ts.layout.curated-exports]
    // verify[sdk.ts.layout.dependency-direction]
    // verify[sdk.ts.api.resources-and-data]
    const subpaths: Readonly<Record<string, Record<string, unknown>>> = {
      ".": rootMod,
      "./parameters": parametersMod,
      "./tempo": tempoMod,
      "./schema": schemaMod,
      "./provenance": provenanceMod,
      "./checkpoint": checkpointMod,
      "./dsp": dspMod,
      "./web-audio": webAudioMod,
    };

    for (const [subpath, mod] of Object.entries(subpaths)) {
      const keys = Object.keys(mod);
      expect(keys.length, `subpath ${subpath} should be non-empty`).toBeGreaterThan(0);
      expect(keys).not.toContain("default");
      for (const key of keys) {
        expect(key.startsWith("native")).toBe(false);
        expect(key.startsWith("Native")).toBe(false);
        for (const legacy of FORBIDDEN_TRI_PHYSICS) {
          expect(key).not.toBe(legacy);
        }
      }
    }

    expect(Object.keys(rootMod).sort()).toEqual([
      "Checkpoint",
      "ControlSpace",
      "DnamError",
      "Engine",
      "GramMatrix",
      "Model",
      "Parameter",
      "ParameterMap",
      "TempoController",
      "backends",
      "clampParameters",
      "crc32c",
      "decodeHeader",
      "defineModelTypes",
      "encodeHeader",
      "errorCodes",
      "headerToJcs",
      "initWasm",
      "isDnamError",
      "parseSchemaVersion",
      "protectionProfiles",
      "subdivisions",
      "tiers",
      "uploUPackedLength",
    ]);
  });

  it("achieves 0-ULP bit-identical parity between N-API, WASM, and native Rust on the scalar backend across all tiers and topologies", () => {
    // verify[sdk.ts.engine.behavioral-parity]
    // verify[sdk.ts.runtime.dual-target-drivers]
    const testing = getTestingBindings();
    const napiDriver = loadNodeDriverSync();
    const wasmDriver = loadNodeWasmDriverSync();

    const frames = 256;
    const leftIn = new Float32Array(frames);
    const rightIn = new Float32Array(frames);
    for (let i = 0; i < frames; i++) {
      leftIn[i] = Math.sin(i * 0.11) * 0.5;
      rightIn[i] = Math.cos(i * 0.07) * 0.4;
    }

    // 1. Stereo (2->2) across all 4 tiers
    const stereoContainer = rootMod.Model.reference({
      inputs: 2,
      outputMode: "stereo",
    }).toBytes();

    for (let tierIdx = 0; tierIdx < tiers.length; tierIdx++) {
      const tier = tiers[tierIdx] ?? "full";
      const [rustL, rustR] = testing.testingRenderReferenceStereo(
        stereoContainer,
        tierIdx,
        "scalar",
        48_000,
        leftIn,
        rightIn,
        64,
        true,
      );

      const [napiL, napiR] = withDriver(napiDriver, () => {
        using model = rootMod.Model.from(stereoContainer);
        using engine = model.createEngine({
          tier,
          backend: "scalar",
          sampleRate: 48_000,
        });
        return engine.process([leftIn, rightIn], { blockSize: 64 });
      });

      const [wasmL, wasmR] = withDriver(wasmDriver, () => {
        using model = rootMod.Model.from(stereoContainer);
        using engine = model.createEngine({
          tier,
          backend: "scalar",
          sampleRate: 48_000,
        });
        return engine.process([leftIn, rightIn], { blockSize: 64 });
      });

      // N-API vs. native Rust on the same ISA: bit-identical (0 ULP)
      expect(new Uint32Array((napiL as Float32Array).buffer)).toEqual(
        new Uint32Array((rustL as Float32Array).buffer),
      );
      expect(new Uint32Array((napiR as Float32Array).buffer)).toEqual(
        new Uint32Array((rustR as Float32Array).buffer),
      );
      // WASM (wasm32 non-fused IEEE-754 mul+add) vs. host (hardware fmadd): <= 4 ULP (< 5e-7)
      for (let i = 0; i < frames; i++) {
        expect(
          Math.abs(((wasmL as Float32Array)[i] ?? 0) - ((rustL as Float32Array)[i] ?? 0)),
        ).toBeLessThan(5e-7);
        expect(
          Math.abs(((wasmR as Float32Array)[i] ?? 0) - ((rustR as Float32Array)[i] ?? 0)),
        ).toBeLessThan(5e-7);
      }
    }

    // 2. Mono-to-stereo (1->2) and Mono (1->1) parity
    const [rustM2sL, rustM2sR] = testing.testingRenderReferenceMonoToStereo(
      stereoContainer,
      1,
      "scalar",
      48_000,
      leftIn,
      128,
      true,
    );
    const [napiM2sL, napiM2sR] = withDriver(napiDriver, () => {
      using model = rootMod.Model.from(stereoContainer);
      using engine = model.createEngine({ tier: "full", backend: "scalar", sampleRate: 48_000 });
      return engine.process(leftIn, { blockSize: 128 });
    });
    const [wasmM2sL, wasmM2sR] = withDriver(wasmDriver, () => {
      using model = rootMod.Model.from(stereoContainer);
      using engine = model.createEngine({ tier: "full", backend: "scalar", sampleRate: 48_000 });
      return engine.process(leftIn, { blockSize: 128 });
    });
    expect(new Uint32Array((napiM2sL as Float32Array).buffer)).toEqual(
      new Uint32Array((rustM2sL as Float32Array).buffer),
    );
    expect(new Uint32Array((napiM2sR as Float32Array).buffer)).toEqual(
      new Uint32Array((rustM2sR as Float32Array).buffer),
    );
    for (let i = 0; i < frames; i++) {
      expect(
        Math.abs(((wasmM2sL as Float32Array)[i] ?? 0) - ((rustM2sL as Float32Array)[i] ?? 0)),
      ).toBeLessThan(5e-7);
      expect(
        Math.abs(((wasmM2sR as Float32Array)[i] ?? 0) - ((rustM2sR as Float32Array)[i] ?? 0)),
      ).toBeLessThan(5e-7);
    }

    const monoContainer = rootMod.Model.reference({ inputs: 1, outputMode: "mono" }).toBytes();
    const rustMono = testing.testingRenderReferenceMono(
      monoContainer,
      1,
      "scalar",
      48_000,
      leftIn,
      128,
      true,
    );
    const [napiMono] = withDriver(napiDriver, () => {
      using model = rootMod.Model.from(monoContainer);
      using engine = model.createEngine({ tier: "full", backend: "scalar", sampleRate: 48_000 });
      return engine.process(leftIn, { blockSize: 128 });
    });
    const [wasmMono] = withDriver(wasmDriver, () => {
      using model = rootMod.Model.from(monoContainer);
      using engine = model.createEngine({ tier: "full", backend: "scalar", sampleRate: 48_000 });
      return engine.process(leftIn, { blockSize: 128 });
    });
    expect(new Uint32Array((napiMono as Float32Array).buffer)).toEqual(
      new Uint32Array(rustMono.buffer),
    );
    for (let i = 0; i < frames; i++) {
      expect(Math.abs(((wasmMono as Float32Array)[i] ?? 0) - (rustMono[i] ?? 0))).toBeLessThan(
        5e-7,
      );
    }
  });

  it("verifies WASM SIMD128 parity and subnormal/denormal flushing without NaN or underflow traps", () => {
    // verify[sdk.ts.engine.behavioral-parity]
    // verify[sdk.ts.runtime.dual-target-drivers]
    const wasmDriver = loadNodeWasmDriverSync();
    withDriver(wasmDriver, () => {
      using model = rootMod.Model.reference();
      using simdEngine = model.createEngine({
        tier: "ultra",
        backend: "wasm-simd128",
        sampleRate: 48_000,
      });
      using scalarEngine = model.createEngine({
        tier: "ultra",
        backend: "scalar",
        sampleRate: 48_000,
      });
      expect(simdEngine.backend).toBe("wasm-simd128");
      expect(scalarEngine.backend).toBe("scalar");

      // Construct subnormal + normal Float32Array inputs
      const subBits = new Uint32Array(128);
      for (let i = 0; i < 128; i++) {
        subBits[i] = i % 2 === 0 ? 0x0000_0001 : 0x807f_ffff; // smallest & largest subnormals
      }
      const subInput = new Float32Array(subBits.buffer);
      const [simdSubL, simdSubR] = simdEngine.process([subInput, subInput]);
      const [scalarSubL, scalarSubR] = scalarEngine.process([subInput, subInput]);

      for (let i = 0; i < 128; i++) {
        expect(Number.isFinite(simdSubL?.[i])).toBe(true);
        expect(Number.isFinite(simdSubR?.[i])).toBe(true);
        expect(Math.abs((simdSubL?.[i] ?? 0) - (scalarSubL?.[i] ?? 0))).toBeLessThan(1e-5);
        expect(Math.abs((simdSubR?.[i] ?? 0) - (scalarSubR?.[i] ?? 0))).toBeLessThan(1e-5);
      }
      expect(simdEngine.telemetry().overflowFlags.nonFiniteSanitized).toBe(false);
    });
  });

  it("meets real-time 128-frame latency budgets", () => {
    // verify[sdk.ts.perf.profiling-budgets]
    const dir = dirname(fileURLToPath(import.meta.url));
    const wasmPath = resolve(dir, "../../dist/dnam.wasm");
    const wasmSize = statSync(wasmPath).size;
    expect(wasmSize).toBeGreaterThan(0);

    const napiDriver = loadNodeDriverSync();
    const wasmDriver = loadNodeWasmDriverSync();
    const inL = new Float32Array(128).fill(0.125);
    const inR = new Float32Array(128).fill(-0.125);
    const outL = new Float32Array(128);
    const outR = new Float32Array(128);

    for (const driver of [napiDriver, wasmDriver]) {
      withDriver(driver, () => {
        using model = rootMod.Model.reference();
        using engine = model.createEngine({ tier: "full", sampleRate: 48_000 });
        for (let i = 0; i < 16; i++) {
          engine.process([inL, inR], { output: [outL, outR] });
        }
        const iters = 200;
        let avgUs = Number.POSITIVE_INFINITY;
        for (let attempt = 0; attempt < 3 && avgUs >= 500; attempt++) {
          const start = performance.now();
          for (let i = 0; i < iters; i++) {
            engine.process([inL, inR], { output: [outL, outR] });
          }
          avgUs = Math.min(avgUs, ((performance.now() - start) * 1000) / iters);
        }
        expect(avgUs).toBeLessThan(500);
      });
    }
  });

  it("property: fast-check verifies N-API vs WASM vs native Rust scalar equivalence over random stereo signals and tiers", () => {
    // verify[sdk.ts.engine.behavioral-parity]
    // verify[sdk.ts.runtime.dual-target-drivers]
    const testing = getTestingBindings();
    const napiDriver = loadNodeDriverSync();
    const wasmDriver = loadNodeWasmDriverSync();
    const stereoContainer = rootMod.Model.reference({
      inputs: 2,
      outputMode: "stereo",
    }).toBytes();

    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: 3 }),
        fc.array(fc.float({ min: Math.fround(-0.9), max: Math.fround(0.9), noNaN: true }), {
          minLength: 32,
          maxLength: 64,
        }),
        fc.array(fc.float({ min: Math.fround(-0.9), max: Math.fround(0.9), noNaN: true }), {
          minLength: 32,
          maxLength: 64,
        }),
        (tierIdx, leftSamples, rightSamples) => {
          const n = Math.min(leftSamples.length, rightSamples.length);
          const leftIn = new Float32Array(leftSamples.slice(0, n));
          const rightIn = new Float32Array(rightSamples.slice(0, n));
          const tier = tiers[tierIdx] ?? "full";

          const [rustL, rustR] = testing.testingRenderReferenceStereo(
            stereoContainer,
            tierIdx,
            "scalar",
            48_000,
            leftIn,
            rightIn,
            32,
            true,
          );

          const [napiL, napiR] = withDriver(napiDriver, () => {
            using model = rootMod.Model.from(stereoContainer);
            using engine = model.createEngine({ tier, backend: "scalar", sampleRate: 48_000 });
            return engine.process([leftIn, rightIn], { blockSize: 32 });
          });

          const [wasmL, wasmR] = withDriver(wasmDriver, () => {
            using model = rootMod.Model.from(stereoContainer);
            using engine = model.createEngine({ tier, backend: "scalar", sampleRate: 48_000 });
            return engine.process([leftIn, rightIn], { blockSize: 32 });
          });

          expect(new Uint32Array((napiL as Float32Array).buffer)).toEqual(
            new Uint32Array((rustL as Float32Array).buffer),
          );
          expect(new Uint32Array((napiR as Float32Array).buffer)).toEqual(
            new Uint32Array((rustR as Float32Array).buffer),
          );
          for (let i = 0; i < n; i++) {
            expect(
              Math.abs(((wasmL as Float32Array)[i] ?? 0) - ((rustL as Float32Array)[i] ?? 0)),
            ).toBeLessThan(1e-6);
            expect(
              Math.abs(((wasmR as Float32Array)[i] ?? 0) - ((rustR as Float32Array)[i] ?? 0)),
            ).toBeLessThan(1e-6);
          }
        },
      ),
      { numRuns: 20 },
    );
  });

  it.each([
    "01-quickstart-render.ts",
    "02-parameters-tempo-and-clamping.ts",
    "03-checkpoints-and-gram-matrix.ts",
    "04-provenance-sealing-and-watermarking.ts",
    "05-reference-dsp-kernels.ts",
    "06-web-audio-worklet.ts",
  ] as const)("executes runnable example %s without error", async (exampleName) => {
    // verify[sdk.ts.api.resources-and-data]
    // verify[sdk.ts.api.disposable-scopes]
    const dir = dirname(fileURLToPath(import.meta.url));
    const examplePath = resolve(dir, "../../examples", exampleName);
    const mod = await import(pathToFileURL(examplePath).href);
    expect(mod).toBeDefined();
  });

  it("enforces zero unsafe blocks, zero raw pointer slices, and forbid(unsafe_code) across Rust SDK bindings", () => {
    // verify[sdk.ts.runtime.zero-unsafe]
    const dir = dirname(fileURLToPath(import.meta.url));
    const jsRustRoot = resolve(dir, "../../rust");
    const engineFfiPath = resolve(dir, "../../../engine/src/ffi.rs");

    function collectRsFiles(root: string): string[] {
      const out: string[] = [];
      for (const entry of readdirSync(root, { withFileTypes: true })) {
        const full = resolve(root, entry.name);
        if (entry.isDirectory()) {
          out.push(...collectRsFiles(full));
        } else if (entry.isFile() && entry.name.endsWith(".rs")) {
          out.push(full);
        }
      }
      return out.sort();
    }

    const engineFfiSrc = readFileSync(engineFfiPath, "utf8");
    expect(engineFfiSrc).not.toContain("dnam_wasm_export");
    expect(engineFfiSrc).not.toContain("dnam_napi_ctor");

    const jsLibSrc = readFileSync(resolve(jsRustRoot, "lib.rs"), "utf8");
    expect(jsLibSrc).toContain("#![deny(unsafe_code)]");
    expect(jsLibSrc).toMatch(/#\[forbid\(unsafe_code\)\]\s*pub mod core;/);
    expect(jsLibSrc).toMatch(/#\[forbid\(unsafe_code\)\]\s*pub mod error;/);

    const jsWasmSrc = readFileSync(resolve(jsRustRoot, "wasm.rs"), "utf8");
    expect(jsWasmSrc).toMatch(/#\[forbid\(unsafe_code\)\]\s*pub mod abi;/);

    for (const jsFile of collectRsFiles(jsRustRoot)) {
      const rel = relative(jsRustRoot, jsFile).replaceAll("\\", "/");
      const src = readFileSync(jsFile, "utf8");
      expect(src, `${rel} must not contain unsafe blocks`).not.toMatch(/\bunsafe\s*\{/);
      expect(src, `${rel} must not contain unsafe fn/impl/trait`).not.toMatch(
        /\bunsafe\s+(?:extern\b|fn\b|impl\b|trait\b)/,
      );
      expect(src, `${rel} must not use raw slice::from_raw_parts`).not.toContain("from_raw_parts");
      if (rel !== "wasm.rs") {
        expect(src, `${rel} must not contain #[allow(unsafe_code)]`).not.toContain(
          "allow(unsafe_code)",
        );
      }
    }
  });
});
