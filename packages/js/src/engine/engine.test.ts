/**
 * Tests for `Engine` audio processing, topologies, `processAsync`, `render`, `bounce`,
 * `stream` (`TransformStream`), and `bypass`/`freeze` state scopes.
 */

import "../node.js";

import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { DnamError, isDnamError } from "../errors.js";
import { withDriver } from "../internal/native/load.js";
import { loadNodeDriverSync, loadNodeWasmDriverSync } from "../internal/native/napi.js";
import { Model } from "../model/model.js";

describe("Engine real-time processing, scopes, async & streaming", () => {
  it.each(["napi", "wasm"] as const)(
    "processes all 4 I/O topologies, in-place buffers, presets, bypass/freeze scopes, and telemetry on %s",
    (driverKind) => {
      // verify[sdk.ts.engine.zero-copy-and-async]
      // verify[sdk.ts.engine.streaming-and-render]
      // verify[sdk.ts.api.disposable-scopes]
      const driver = driverKind === "napi" ? loadNodeDriverSync() : loadNodeWasmDriverSync();
      withDriver(driver, () => {
        // 1. 2->2 stereo model
        using stereoModel = Model.reference({
          supportsFreeze: true,
          inputs: 2,
          outputMode: "stereo",
        });
        using engine = stereoModel.createEngine({ sampleRate: 48_000, tier: "full" });
        expect(engine.model).toBe(stereoModel);
        expect(engine.presets.has("warm_slapback")).toBe(true);
        expect(engine.backend.length).toBeGreaterThan(0);
        expect(engine.tier).toBe("full");
        expect(engine.effectiveTier).toBe("full");
        expect(engine.maxAvailableTier).toBe("ultra");
        expect(engine.inputChannels).toBe(2);
        expect(engine.outputChannels).toBe(2);
        expect(engine.latencySamples).toBe(0);

        engine.offlineBounce = true;
        expect(engine.offlineBounce).toBe(true);
        expect(engine.latencySamples).toBe(12);
        engine.offlineBounce = false;
        expect(engine.offlineBounce).toBe(false);

        engine.maximizeTierOnBounce = false;
        expect(engine.maximizeTierOnBounce).toBe(false);
        engine.maximizeTierOnBounce = true;
        expect(engine.maximizeTierOnBounce).toBe(true);

        engine.maxSupportedTier = "lite";
        expect(engine.maxSupportedTier).toBe("lite");
        expect(engine.effectiveTier).toBe("lite");
        engine.maxSupportedTier = "ultra";
        engine.tier = "full";
        expect(engine.effectiveTier).toBe("full");
        engine.sampleRate = 44_100;
        expect(engine.sampleRate).toBe(44_100);
        engine.sampleRate = 48_000;
        engine.reset();

        const inL = new Float32Array(128).fill(0.25);
        const inR = new Float32Array(128).fill(-0.25);
        const [outL, outR] = engine.process([inL, inR]);
        expect(outL?.length).toBe(128);
        expect(outR?.length).toBe(128);

        // Pre-allocated output buffer (zero-allocation path) & validation checks
        const preL = new Float32Array(128);
        const preR = new Float32Array(128);
        const returned = engine.process([inL, inR], { output: [preL, preR] });
        expect(returned[0]).toBe(preL);
        expect(returned[1]).toBe(preR);
        expect(engine.process(new Float32Array(0))[0]?.length).toBe(0);
        expect(() => engine.process([inL, inR], { output: [preL] })).toThrow(DnamError);
        expect(() =>
          engine.process([inL, inR], { output: [new Float32Array(4), new Float32Array(4)] }),
        ).toThrow(DnamError);

        // In-place stereo processing (`leftOut === leftIn && rightOut === rightIn`)
        const inplaceL = inL.slice();
        const inplaceR = inR.slice();
        engine.process([inplaceL, inplaceR], { output: [inplaceL, inplaceR] });

        // Mono input to stereo model (`1->2`)
        const [m2sL, m2sR] = engine.process(inL);
        expect(m2sL?.length).toBe(128);
        expect(m2sR?.length).toBe(128);

        // Presets, host mix, width
        engine.applyPreset("warm_slapback");
        expect(engine.parameters.get("boost").value).toBe("on");
        expect(() => engine.applyPreset("unknown_preset")).toThrow(DnamError);
        engine.configureHostMix({ blendWet: 0.8, outputGainDb: 0.0, applyUnityLeveling: true });
        engine.stereoWidth = 0.5;
        expect(engine.stereoWidth).toBeCloseTo(0.5, 5);

        // Scoped bypass (`using` + `withBypass`)
        expect(engine.isBypassed).toBe(false);
        {
          using _scope = engine.bypass({ trails: false });
          expect(engine.isBypassed).toBe(true);
          expect(engine.trailsEnabled).toBe(false);
        }
        expect(engine.isBypassed).toBe(false);

        engine.withBypass({ trails: true }, () => {
          expect(engine.isBypassed).toBe(true);
          expect(engine.trailsEnabled).toBe(true);
        });
        expect(engine.isBypassed).toBe(false);

        // Scoped freeze (`using` + `withFreeze`)
        expect(engine.supportsFreeze).toBe(true);
        expect(engine.isFrozen).toBe(false);
        {
          using _fScope = engine.freeze({ amount: 0.75 });
          expect(engine.isFrozen).toBe(true);
        }
        expect(engine.isFrozen).toBe(false);

        engine.withFreeze(() => {
          expect(engine.isFrozen).toBe(true);
        });
        expect(engine.isFrozen).toBe(false);

        // Unsupported freeze throws ERR_DNAM_PARAMETER
        using noFreezeModel = Model.reference({ supportsFreeze: false });
        using noFreezeEngine = noFreezeModel.createEngine();
        expect(noFreezeEngine.supportsFreeze).toBe(false);
        expect(() => noFreezeEngine.freeze()).toThrow(DnamError);

        // 1->1 and 2->1 mono output topologies
        using monoModel = Model.reference({ inputs: 1, outputMode: "mono" });
        using monoEngine = monoModel.createEngine();
        expect(monoEngine.outputChannels).toBe(1);
        const [monoOut1] = monoEngine.process(inL);
        expect(monoOut1?.length).toBe(128);
        const [monoOut2] = monoEngine.process([inL, inR]);
        expect(monoOut2?.length).toBe(128);

        // Offline render & 4x bounce
        const bounced = engine.bounce([inL, inR], { maxTier: true });
        expect(bounced.length).toBe(2);
        expect(bounced[0]?.length).toBe(128);
        const renderedOversampled = engine.render([inL, inR], { oversample: 4, maxTier: false });
        expect(renderedOversampled[0]?.length).toBe(128);

        engine.setBypass(true, { trails: true });
        expect(engine.isBypassed).toBe(true);
        engine.setBypass(false);
        expect(engine.isBypassed).toBe(false);

        engine.setFreeze(true, { amount: 0.5 });
        expect(engine.isFrozen).toBe(true);
        engine.setFreeze(false);
        expect(engine.isFrozen).toBe(false);

        const rendered = engine.render([inL, inR], {
          sampleRate: 96_000,
          blockSize: 64,
          reset: true,
        });
        expect(rendered[0]?.length).toBe(128);
        expect(engine.sampleRate).toBe(96_000);

        // Telemetry snapshot
        const snap = engine.telemetry();
        expect(Object.isFrozen(snap)).toBe(true);
        expect(snap.sampleRateHz).toBe(96_000);
        expect(snap.rhoSr).toBeCloseTo(0.5, 5);

        // Reload in place
        engine.reload(stereoModel, { tier: "lite", sampleRate: 48_000 });
        expect(engine.tier).toBe("lite");
        expect(engine.sampleRate).toBe(48_000);
      });
    },
  );

  it("executes processAsync on the worker pool, enforces ERR_DNAM_BUSY on concurrent calls, and streams via TransformStream", async () => {
    // verify[sdk.ts.engine.zero-copy-and-async]
    // verify[sdk.ts.engine.streaming-and-render]
    using model = Model.reference();
    using engine = model.createEngine({ sampleRate: 48_000, tier: "ultra" });
    const inL = new Float32Array(32_768).fill(0.1);
    const inR = new Float32Array(32_768).fill(-0.1);

    const promise = engine.processAsync([inL, inR], { blockSize: 256 });
    let busyErr: unknown;
    try {
      engine.process([inL, inR]);
    } catch (err) {
      busyErr = err;
    }
    expect(isDnamError(busyErr, "ERR_DNAM_BUSY")).toBe(true);
    const [asyncL, asyncR] = await promise;
    expect(asyncL?.length).toBe(32_768);
    expect(asyncR?.length).toBe(32_768);

    // Also test processAsync with output buffers & mono model on WASM driver
    const wasmDriver = loadNodeWasmDriverSync();
    await withDriver(wasmDriver, async () => {
      using wasmModel = Model.reference();
      using wasmEngine = wasmModel.createEngine({ sampleRate: 48_000, tier: "lite" });
      const outA = new Float32Array(128);
      const outB = new Float32Array(128);
      const [wL, wR] = await wasmEngine.processAsync(
        [new Float32Array(128).fill(0.1), new Float32Array(128).fill(-0.1)],
        { output: [outA, outB] },
      );
      expect(wL).toBe(outA);
      expect(wR).toBe(outB);

      using monoWasmModel = Model.reference({ inputs: 1, outputMode: "mono" });
      using monoWasmEngine = monoWasmModel.createEngine();
      const [monoAsync] = await monoWasmEngine.processAsync(new Float32Array(64).fill(0.1));
      expect(monoAsync?.length).toBe(64);
    });

    // AbortSignal support
    const abortCtrl = new AbortController();
    abortCtrl.abort();
    await expect(engine.processAsync([inL, inR], { signal: abortCtrl.signal })).rejects.toThrow();

    // TransformStream
    const chunksIn: Float32Array[][] = [
      [new Float32Array(64).fill(0.1), new Float32Array(64).fill(0.1)],
      [new Float32Array(64).fill(0.2), new Float32Array(64).fill(0.2)],
    ];
    const chunksOut: Float32Array[][] = [];
    const readable = new ReadableStream<Float32Array[]>({
      start(controller) {
        for (const c of chunksIn) {
          controller.enqueue(c);
        }
        controller.close();
      },
    });
    const writable = new WritableStream<Float32Array[]>({
      write(chunk) {
        chunksOut.push(chunk);
      },
    });
    await readable.pipeThrough(engine.stream({ blockSize: 32 })).pipeTo(writable);
    expect(chunksOut.length).toBe(2);
    expect(chunksOut[0]?.[0]?.length).toBe(64);
  });

  it("property: fast-check verifies bounded stereo processing stability, zero-allocation output parity, and telemetry across arbitrary inputs", () => {
    // verify[sdk.ts.engine.zero-copy-and-async]
    // verify[sdk.ts.engine.behavioral-parity]
    using model = Model.reference({ supportsFreeze: true, inputs: 2, outputMode: "stereo" });
    using engine = model.createEngine({ sampleRate: 48_000, tier: "lite" });

    fc.assert(
      fc.property(
        fc.array(fc.float({ min: -1.0, max: 1.0, noNaN: true }), { minLength: 16, maxLength: 64 }),
        fc.constantFrom("lite", "full", "pro", "ultra"),
        fc.double({ min: 0.0, max: 1.0, noNaN: true }),
        fc.double({ min: 0.0, max: 9.5, noNaN: true }),
        (samples, tier, width, gain) => {
          const n = samples.length;
          const inL = new Float32Array(samples);
          const inR = new Float32Array(n);
          for (let i = 0; i < n; i++) {
            inR[i] = -(inL[i] ?? 0) * 0.8;
          }

          engine.reset();
          engine.tier = tier;
          engine.stereoWidth = width;
          engine.parameters.get("gain").setValue(gain, { immediate: true });
          const [allocL, allocR] = engine.process([inL, inR]);

          engine.reset();
          engine.tier = tier;
          engine.stereoWidth = width;
          engine.parameters.get("gain").setValue(gain, { immediate: true });
          const preL = new Float32Array(n);
          const preR = new Float32Array(n);
          engine.process([inL, inR], { output: [preL, preR] });

          expect(preL).toEqual(allocL);
          expect(preR).toEqual(allocR);
          for (let i = 0; i < n; i++) {
            expect(Number.isFinite(preL[i])).toBe(true);
            expect(Number.isFinite(preR[i])).toBe(true);
          }
          expect(engine.telemetry().overflowFlags.nonFiniteSanitized).toBe(false);
        },
      ),
      { numRuns: 25 },
    );
  });
});
