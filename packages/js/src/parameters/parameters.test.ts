/**
 * Tests for `Parameter`, `ParameterMap`, `ControlSpace`, `clampParameters`, `defineModelTypes`,
 * and `TempoController` / ZOH / tempo calibration inversion.
 */

import "../node.js";

import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { DnamError } from "../errors.js";
import { withDriver } from "../internal/native/load.js";
import { loadNodeDriverSync, loadNodeWasmDriverSync } from "../internal/native/napi.js";
import { Model } from "../model/model.js";
import {
  computeZoh,
  controlStride,
  controlTickRatio,
  evalCalibration3Node,
  invertTempoCalibration,
  subdivisionPhase,
} from "../tempo/index.js";
import { ControlSpace, clampParameters, defineModelTypes } from "./index.js";

describe("Parameters, ControlSpace & Tempo", () => {
  it.each(["napi", "wasm"] as const)(
    "evaluates 3-Phase clamping and AudioParam-style Parameter updates on %s driver",
    (driverKind) => {
      // verify[sdk.ts.parameters.audio-param-semantics]
      // verify[sdk.ts.api.typed-model-inference]
      // verify[sdk.ts.api.literal-unions-and-options]
      const driver = driverKind === "napi" ? loadNodeDriverSync() : loadNodeWasmDriverSync();
      withDriver(driver, () => {
        const AmpSpec = defineModelTypes([
          { id: "gain", type: "range" },
          { id: "boost", type: "switch", declaredOptions: ["off", "on", "ultra"] },
          { id: "delay_ms", type: "range" },
        ] as const);
        expect(AmpSpec.entries.length).toBe(3);

        const model = Model.reference<typeof AmpSpec.$infer>();
        using space = ControlSpace.from(model);
        expect(space.size).toBe(3);
        expect(space.descriptors.length).toBe(3);
        using spaceFromHdr = ControlSpace.from(model.header);
        expect(spaceFromHdr.size).toBe(3);
        using spaceFromStr = ControlSpace.from(model.rawHeaderJson);
        expect(spaceFromStr.size).toBe(3);

        // Phase-1 [CLAMP-RANGE] and [CLAMP-SWITCH] + conditional max (gain <= 8.0 when boost=on)
        const report = clampParameters(space, {
          gain: 9.2,
          boost: "ultra",
          delay_ms: 250,
        });
        expect(report.anyClamped).toBe(true);
        expect(report.values.boost).toBe("off"); // 'ultra' unobserved -> falls back to default 'off'
        expect(report.clampedIds).toContain("boost");

        const boostOnReport = clampParameters(model, {
          gain: 9.2,
          boost: "on",
        });
        expect(boostOnReport.values.gain).toBeCloseTo(8.0, 5);
        expect(boostOnReport.clampedIds).toContain("gain");

        // Live Engine parameters
        using engine = model.createEngine({ sampleRate: 48_000, tier: "full" });
        expect(engine.parameters.size).toBe(3);
        expect(engine.parameters.has("gain")).toBe(true);
        expect(engine.parameters.has("nonexistent")).toBe(false);
        expect(() => engine.parameters.get("nonexistent" as "gain")).toThrow(DnamError);

        const gain = engine.parameters.get("gain");
        expect(gain.name).toBe("Drive");
        expect(gain.kind).toBe("range");
        expect(gain.unit).toBe("dB");
        expect(gain.minValue).toBe(0);
        expect(gain.maxValue).toBe(10);
        expect(gain.safeRange).toEqual([0, 9.5]);
        expect(gain.defaultValue).toBe(5);
        expect(gain.uiHints.widget).toBe("knob");
        expect(gain.clampWarning).toContain("[CLAMP-RANGE]");

        gain.setValue(7.5, { immediate: true });
        expect(gain.value).toBeCloseTo(7.5, 5);
        gain.value = 6.0;
        gain.reset();
        expect(gain.value).toBeCloseTo(5.0, 5);

        const normU = gain.toNormalized(5.0);
        expect(normU).toBeCloseTo(0.0, 5);
        expect(gain.toPhysical(normU)).toBeCloseTo(5.0, 4);

        const boost = engine.parameters.get("boost");
        expect(boost.kind).toBe("switch");
        expect(boost.options).toEqual(["off", "on", "ultra"]);
        expect(boost.activeOptions).toEqual(["off", "on"]);
        boost.setValue("on", { immediate: true });
        expect(boost.value).toBe("on");

        // Batch immediate set triggers conditional clamp on gain
        engine.parameters.set({ boost: "on", gain: 9.0 }, { immediate: true });
        expect(gain.value).toBeCloseTo(8.0, 5);
        expect(gain.clamped).toBe(true);
        expect(engine.parameters.clampedReport().anyClamped).toBe(true);

        const entries = [...engine.parameters];
        expect(entries.length).toBe(3);
        expect([...engine.parameters.keys()]).toEqual(["gain", "boost", "delay_ms"]);
        expect([...engine.parameters.values()].length).toBe(3);
        expect(engine.parameters.toObject().boost).toBe("on");
        engine.parameters.set({ gain: 6.0 }, { immediate: false });

        engine.parameters.resetAll();
        expect(gain.value).toBeCloseTo(5.0, 5);
        expect(boost.value).toBe("off");
      });
    },
  );

  it.each(["napi", "wasm"] as const)(
    "computes ZOH rescaling, subdivision phases, and 32-step tempo calibration inversion on %s driver",
    (driverKind) => {
      // verify[sdk.ts.model.containers-and-checkpoints]
      const driver = driverKind === "napi" ? loadNodeDriverSync() : loadNodeWasmDriverSync();
      withDriver(driver, () => {
        expect(controlStride(48_000)).toBe(16);
        expect(controlStride(96_000)).toBe(32);
        expect(controlTickRatio(48_000)).toBeCloseTo(1.0, 6);

        const zoh = computeZoh({
          sampleRateTrainHz: 48_000,
          sampleRateHostHz: 96_000,
        });
        expect(zoh.sampleRateHostHz).toBe(96_000);
        expect(zoh.rhoSr).toBeCloseTo(0.5, 6);
        expect(zoh.controlDecimationHost).toBe(32);

        const phase = subdivisionPhase({
          beatMod12: 0,
          withinBeatQ32: 0,
          subdivision: "1/8.",
        });
        expect(phase.phaseQ32).toBe(0);
        expect(phase.cosPhi).toBeCloseTo(1.0, 5);
        expect(phase.sinPhi).toBeCloseTo(0.0, 5);

        const anchors = [20.0, 350.0, 1000.0] as const;
        expect(evalCalibration3Node(anchors, 0.0)).toBeCloseTo(350.0, 6);

        const inv = invertTempoCalibration({
          declaredRange: [20.0, 1000.0],
          safeRange: [20.0, 1000.0],
          anchors,
          target: 375.0,
        });
        expect(inv.searchSteps).toBeGreaterThanOrEqual(20);
        expect(inv.searchSteps).toBeLessThanOrEqual(32);
        expect(Math.abs(inv.realizedValue - 375.0)).toBeLessThan(1e-2);

        using model = Model.reference();
        using engine = model.createEngine({ sampleRate: 48_000 });
        const updated = engine.tempo.sync({ bpm: 120, subdivision: "1/8" });
        expect(updated).toBeGreaterThanOrEqual(1);
        expect(engine.tempo.bpm).toBeCloseTo(120, 4);
        const delayParam = engine.parameters.get("delay_ms");
        expect(delayParam.value).toBeCloseTo(402.2654, 2);

        const tapRes = engine.tempo.tap({ subdivision: "1/4" });
        expect(tapRes.bpm).toBeGreaterThan(0);
      });
    },
  );

  it("property: fast-check verifies 3-Phase clamping bounds, ZOH invariants, and tempo calibration inversion across N-API & WASM", () => {
    // verify[sdk.ts.parameters.audio-param-semantics]
    // verify[sdk.ts.model.containers-and-checkpoints]
    const napiDriver = loadNodeDriverSync();
    const wasmDriver = loadNodeWasmDriverSync();
    using model = Model.reference();

    fc.assert(
      fc.property(
        fc.double({ min: -50, max: 50, noNaN: true }),
        fc.constantFrom("off", "on", "ultra"),
        fc.double({ min: -100, max: 2000, noNaN: true }),
        (gain, boost, delayMs) => {
          const napiReport = withDriver(napiDriver, () =>
            clampParameters(model.header, { gain, boost, delay_ms: delayMs }),
          );
          const wasmReport = withDriver(wasmDriver, () =>
            clampParameters(model.header, { gain, boost, delay_ms: delayMs }),
          );
          expect(wasmReport).toEqual(napiReport);

          const clampedBoost = napiReport.values.boost;
          expect(["off", "on"]).toContain(clampedBoost);

          const clampedGain = Number(napiReport.values.gain);
          const maxGain = clampedBoost === "on" ? 8.0 : 9.5;
          expect(clampedGain).toBeGreaterThanOrEqual(0.0 - 1e-6);
          expect(clampedGain).toBeLessThanOrEqual(maxGain + 1e-6);

          const clampedDelay = Number(napiReport.values.delay_ms);
          expect(clampedDelay).toBeGreaterThanOrEqual(20.0 - 1e-4);
          expect(clampedDelay).toBeLessThanOrEqual(1000.0 + 1e-4);
        },
      ),
      { numRuns: 50 },
    );

    const anchors = [20.0, 350.0, 1000.0] as const;
    fc.assert(
      fc.property(
        fc.double({ min: 20.0, max: 1000.0, noNaN: true }),
        fc.constantFrom(44_100, 48_000, 88_200, 96_000, 192_000),
        fc.integer({ min: 0, max: 11 }),
        fc.constantFrom("1/4", "1/8", "1/8.", "1/8T", "1/16"),
        (targetMs, hostSr, beatMod12, sub) => {
          const zoh = computeZoh({
            sampleRateTrainHz: 48_000,
            sampleRateHostHz: hostSr,
          });
          expect(zoh.rhoSr).toBeCloseTo(48_000 / hostSr, 5);
          expect(zoh.rAcHost).toBeGreaterThan(0);
          expect(zoh.rAcHost).toBeLessThan(1);

          const ph = subdivisionPhase({
            beatMod12,
            withinBeatQ32: 0x4000_0000,
            subdivision: sub,
          });
          expect(ph.cosPhi * ph.cosPhi + ph.sinPhi * ph.sinPhi).toBeCloseTo(1.0, 5);

          const inv = invertTempoCalibration({
            declaredRange: [20.0, 1000.0],
            safeRange: [20.0, 900.0],
            anchors,
            target: targetMs,
          });
          expect(inv.normalizedF32).toBeGreaterThanOrEqual(-1.0 - 1e-6);
          expect(inv.normalizedF32).toBeLessThanOrEqual(1.0 + 1e-6);
          expect(inv.physicalF32).toBeGreaterThanOrEqual(20.0 - 1e-4);
          expect(inv.physicalF32).toBeLessThanOrEqual(900.0 + 1e-4);
        },
      ),
      { numRuns: 40 },
    );
  });
});
