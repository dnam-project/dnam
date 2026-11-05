/**
 * Vitest benchmark suite for `@dnam-project/dnam` (`ref[sdk.ts.perf.profiling-budgets]`):
 * 1. Real-time 128-frame stereo processing across `Lite`, `Full`, `Pro`, and `Ultra` tiers
 *    on both `NapiDriver` and `WasmDriver`.
 * 2. 3-Phase `ControlSpace.clamp` and `Parameter.setValue` throughput.
 * 3. `ModelHeader` `decodeHeader` / `encodeHeader` and RFC 8785 JCS canonicalization throughput.
 * 4. `.dnam` container parsing and CRC-32C validation throughput.
 */

import "../node.js";

import { describe, test } from "vitest";
import { withDriver } from "../internal/native/load.js";
import { loadNodeDriverSync, loadNodeWasmDriverSync } from "../internal/native/napi.js";
import { crc32c, Model } from "../model/model.js";
import { ControlSpace } from "../parameters/clamp.js";
import { decodeHeader, encodeHeader, headerToJcs } from "../schema/header.js";

const napiDriver = loadNodeDriverSync();
const wasmDriver = loadNodeWasmDriverSync();

const refModel = Model.reference();
const refBytes = refModel.toBytes();
const refHeaderJson = refModel.rawHeaderJson;
const refHeader = refModel.header;

const inL = new Float32Array(128).fill(0.15);
const inR = new Float32Array(128).fill(-0.15);
const outL = new Float32Array(128);
const outR = new Float32Array(128);

const napiLiteEngine = withDriver(napiDriver, () =>
  Model.from(refBytes).createEngine({ tier: "lite", sampleRate: 48_000 }),
);
const napiFullEngine = withDriver(napiDriver, () =>
  Model.from(refBytes).createEngine({ tier: "full", sampleRate: 48_000 }),
);
const napiProEngine = withDriver(napiDriver, () =>
  Model.from(refBytes).createEngine({ tier: "pro", sampleRate: 48_000 }),
);
const napiUltraEngine = withDriver(napiDriver, () =>
  Model.from(refBytes).createEngine({ tier: "ultra", sampleRate: 48_000 }),
);

const wasmLiteEngine = withDriver(wasmDriver, () =>
  Model.from(refBytes).createEngine({ tier: "lite", sampleRate: 48_000 }),
);
const wasmFullEngine = withDriver(wasmDriver, () =>
  Model.from(refBytes).createEngine({ tier: "full", sampleRate: 48_000 }),
);
const wasmProEngine = withDriver(wasmDriver, () =>
  Model.from(refBytes).createEngine({ tier: "pro", sampleRate: 48_000 }),
);
const wasmUltraEngine = withDriver(wasmDriver, () =>
  Model.from(refBytes).createEngine({ tier: "ultra", sampleRate: 48_000 }),
);

const controlSpace = ControlSpace.from(refModel);
const rawParams = new Float32Array([7.5, 1.0, 350.0]);
const clampedParams = new Float32Array(3);

describe("Engine 128-frame stereo block processing (48 kHz, zero-allocation)", () => {
  test("N-API v10 vs WASM SIMD128 across Lite, Full, Pro, and Ultra tiers", async ({ bench }) => {
    await bench.compare(
      bench("N-API Lite (64-D)", () => {
        napiLiteEngine.process([inL, inR], { output: [outL, outR] });
      }),
      bench("N-API Full (128-D)", () => {
        napiFullEngine.process([inL, inR], { output: [outL, outR] });
      }),
      bench("N-API Pro (192-D)", () => {
        napiProEngine.process([inL, inR], { output: [outL, outR] });
      }),
      bench("N-API Ultra (256-D)", () => {
        napiUltraEngine.process([inL, inR], { output: [outL, outR] });
      }),
      bench("WASM SIMD128 Lite (64-D)", () => {
        wasmLiteEngine.process([inL, inR], { output: [outL, outR] });
      }),
      bench("WASM SIMD128 Full (128-D)", () => {
        wasmFullEngine.process([inL, inR], { output: [outL, outR] });
      }),
      bench("WASM SIMD128 Pro (192-D)", () => {
        wasmProEngine.process([inL, inR], { output: [outL, outR] });
      }),
      bench("WASM SIMD128 Ultra (256-D)", () => {
        wasmUltraEngine.process([inL, inR], { output: [outL, outR] });
      }),
      { time: 50 },
    );
  });

  test("ControlSpace 3-Phase clamping, Schema codec & Container CRC-32C", async ({ bench }) => {
    await bench.compare(
      bench("ControlSpace.clampNumericInto (zero-allocation)", () => {
        controlSpace.clampNumericInto(rawParams, clampedParams);
      }),
      bench("decodeHeader + encodeHeader + headerToJcs", () => {
        const hdr = decodeHeader(refHeaderJson);
        encodeHeader(hdr);
        headerToJcs(refHeader);
      }),
      bench("crc32c (64 KB arena)", () => {
        crc32c(refModel.arenaBytes);
      }),
      { time: 50 },
    );
  });
});
