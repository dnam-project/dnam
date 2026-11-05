/**
 * Tests for `@dnam-project/dnam/web-audio` (`WorkletSharedState`, `DnamAudioWorkletProcessor`,
 * and `DnamAudioNode`) across both `SharedArrayBuffer` (`Atomics`) and `MessagePort` fallback modes.
 */

import "../node.js";

import fc from "fast-check";
import { describe, expect, it } from "vitest";
import type { TelemetrySnapshot } from "../engine/telemetry.js";
import { DnamError } from "../errors.js";
import {
  coerceAudioChannels,
  coerceBackend,
  coerceBytes,
  coerceSubdivision,
  coerceTier,
} from "../internal/coerce.js";
import { initWasm } from "../internal/native/load.js";
import { readDistWasmBytes } from "../internal/native/napi.js";
import { getCompiledWasmModule } from "../internal/native/wasm.js";
import { Model } from "../model/model.js";
import {
  CMD_BYPASS_CHANGE,
  CMD_BYPASSED,
  CMD_FREEZE_CHANGE,
  CMD_FROZEN,
  CMD_RESET,
  CMD_RESET_CONTROLS,
  CMD_TEMPO_CHANGE,
  CMD_TIER_CHANGE,
  CMD_TRAILS,
  CMD_WIDTH_CHANGE,
  DnamAudioNode,
  DnamAudioWorkletProcessor,
  isSharedMemorySupported,
  WorkletSharedState,
} from "./index.js";

function loadCompiledWasmModule(): WebAssembly.Module {
  return new WebAssembly.Module(readDistWasmBytes());
}

describe("WebAudio AudioWorklet Bridge (SAB + Atomics & MessagePort)", () => {
  it("synchronizes parameters, commands, and seqlock telemetry via WorkletSharedState & DnamAudioWorkletProcessor", () => {
    // verify[sdk.ts.web-audio.realtime-bridge]
    expect(isSharedMemorySupported()).toBe(true);
    const wasmModule = loadCompiledWasmModule();
    using model = Model.reference();

    const shared = new WorkletSharedState();
    shared.initDefaults(new Float32Array([5.0, 0.0, 250.0]), 1, 48_000);

    const processor = new DnamAudioWorkletProcessor({
      processorOptions: {
        wasmModule,
        containerBytes: model.toBytes(),
        tier: 1,
        sampleRate: 48_000,
        sharedBuffer: shared.buffer,
        telemetryIntervalBlocks: 1,
      },
    });

    // Write parameter update via SAB + Atomics
    shared.writeParameter(0, 7.5, true);
    const inL = new Float32Array(128).fill(0.2);
    const inR = new Float32Array(128).fill(-0.2);
    const outL = new Float32Array(128);
    const outR = new Float32Array(128);

    const keepAlive = processor.process([[inL, inR]], [[outL, outR]]);
    expect(keepAlive).toBe(true);
    expect(shared.readParameter(0)).toBeCloseTo(7.5, 4);

    // Exercise all SAB command bits & batch parameter updates in drainUpdatesIntoEngine
    shared.writeParametersBatch([0, 2], [6.0, 300.0], false);
    shared.writeCommand(
      CMD_RESET_CONTROLS |
        CMD_RESET |
        CMD_BYPASS_CHANGE |
        CMD_BYPASSED |
        CMD_TRAILS |
        CMD_FREEZE_CHANGE |
        CMD_FROZEN |
        CMD_TIER_CHANGE |
        CMD_WIDTH_CHANGE |
        CMD_TEMPO_CHANGE,
      0,
      { tier: 0, freezeAmount: 0.9, stereoWidth: 0.75, bpm: 130.0 },
    );
    processor.process([[inL]], [[outL]]); // mono 1->1 path
    processor.process([[inL]], [[outL, outR]]); // mono-to-stereo 1->2 path
    expect(processor.process([], [])).toBe(true);
    expect(processor.process([[]], [[]])).toBe(true);

    const telBuf = new Float64Array(56);
    shared.readTelemetryInto(telBuf);
    expect(telBuf[2]).toBe(48_000);

    // MessagePort commands on processor
    const onMessage = processor.port.onmessage;
    expect(onMessage).not.toBeNull();
    onMessage?.({ data: { type: "set-param", index: 0, value: 6.0, immediate: true } });
    onMessage?.({
      data: { type: "set-params-batch", indices: [0, 1], values: [8.5, 1], immediate: true },
    });
    expect(shared.readActiveClampMask()).toBeGreaterThan(0);
    onMessage?.({ data: { type: "apply-preset", presetId: "warm_slapback" } });
    onMessage?.({ data: { type: "apply-preset", presetId: "unknown_preset" } });
    onMessage?.({ data: { type: "set-bypass", bypassed: true, trails: true } });
    onMessage?.({ data: { type: "set-freeze", frozen: true, amount: 0.8 } });
    onMessage?.({ data: { type: "set-tier", tier: 0 } });
    onMessage?.({ data: { type: "set-max-supported-tier", tier: 2 } });
    onMessage?.({ data: { type: "set-width", width: 0.5 } });
    onMessage?.({
      data: { type: "sync-tempo", bpm: 120, subdivision: "1/8", foldOctave: false },
    });
    onMessage?.({ data: { type: "tap-tempo", subdivision: "1/4", foldOctave: false } });
    onMessage?.({ data: { type: "reset-controls" } });
    onMessage?.({ data: { type: "reset" } });
    onMessage?.({
      data: { type: "reload", containerBytes: model.toBytes(), tier: 1, sampleRate: 48_000 },
    });
  });

  it.each([true, false] as const)(
    "drives DnamAudioNode with useSharedMemory=%s",
    async (useSharedMemory) => {
      // verify[sdk.ts.web-audio.realtime-bridge]
      // verify[sdk.ts.parameters.audio-param-semantics]
      const wasmModule = loadCompiledWasmModule();
      using model = Model.reference();
      let registeredUrl: string | URL | undefined;
      const mockContext = {
        sampleRate: 48_000,
        audioWorklet: {
          async addModule(url: string | URL) {
            registeredUrl = url;
          },
        },
      };

      using node = await DnamAudioNode.create(mockContext, model, {
        wasmModule,
        tier: "full",
        useSharedMemory,
      });
      expect(registeredUrl).toBeDefined();
      expect(node.usesSharedMemory).toBe(useSharedMemory);
      expect(node.workletNode).toBeUndefined();
      expect(node.model).toBe(model);
      expect(node.presets.has("warm_slapback")).toBe(true);
      expect(node.tier).toBe("full");

      const gain = node.parameters.get("gain");
      gain.setValue(7.5, { immediate: true });
      expect(gain.value).toBeCloseTo(7.5, 4);
      expect(gain.toPhysical(gain.toNormalized(5.0))).toBeCloseTo(5.0, 4);

      node.parameters.set({ boost: "on", gain: 9.0 }, { immediate: true });
      expect(gain.value).toBeCloseTo(8.0, 4);
      expect(gain.clamped).toBe(true);
      expect(node.parameters.clampedReport().anyClamped).toBe(true);

      node.applyPreset("warm_slapback");
      expect(() => node.applyPreset("invalid")).toThrow(DnamError);

      node.tier = "lite";
      expect(node.tier).toBe("lite");
      node.stereoWidth = 0.75;
      expect(node.stereoWidth).toBeCloseTo(0.75, 5);

      {
        using _b = node.bypass({ trails: true });
        expect(node.isBypassed).toBe(true);
      }
      expect(node.isBypassed).toBe(false);

      {
        using _f = node.freeze({ amount: 0.9 });
        expect(node.isFrozen).toBe(true);
      }
      expect(node.isFrozen).toBe(false);

      node.tempo.set({ bpm: 125 });
      node.tempo.set({ bpm: 130, subdivision: "1/8." });
      expect(node.tempo.bpm).toBeCloseTo(130, 4);
      node.tempo.tap({ subdivision: "1/4" });

      node.parameters.resetAll();
      node.reset();
      node.reload(model, { tier: "pro" });
      expect(node.tier).toBe("pro");

      const snap: TelemetrySnapshot = node.telemetry();
      expect(snap.sampleRateHz).toBe(48_000);
      node.disconnect();
      expect(() => node.connect({} as AudioNode)).toThrow(DnamError);
    },
  );

  it("wires AudioWorkletNode and CustomEvent telemetry when BaseAudioContext & AudioWorkletNode exist", async () => {
    // verify[sdk.ts.web-audio.realtime-bridge]
    const wasmModule = loadCompiledWasmModule();
    using model = Model.reference();

    class MockBaseAudioContext {
      readonly sampleRate = 48_000;
      readonly audioWorklet = {
        async addModule(_url: string | URL): Promise<void> {},
      };
    }

    let lastCreatedWorkletNode: MockAudioWorkletNode | undefined;
    class MockAudioWorkletNode {
      readonly port = {
        onmessage: null as ((ev: { readonly data: unknown }) => void) | null,
        postMessage(_msg: unknown): void {},
      };
      connectedTo: unknown = undefined;
      disconnected = false;
      constructor(
        public readonly context: unknown,
        public readonly name: string,
        public readonly options: unknown,
      ) {
        lastCreatedWorkletNode = this;
      }
      connect(dest: AudioNode): AudioNode {
        this.connectedTo = dest;
        return dest;
      }
      disconnect(): void {
        this.disconnected = true;
      }
    }

    const g = globalThis as Record<string, unknown>;
    const prevBaseCtx = g["BaseAudioContext"];
    const prevWorkletNode = g["AudioWorkletNode"];
    g["BaseAudioContext"] = MockBaseAudioContext;
    g["AudioWorkletNode"] = MockAudioWorkletNode;
    try {
      const ctx = new MockBaseAudioContext();
      using node = await DnamAudioNode.create(ctx, model, {
        wasmModule,
        useSharedMemory: false,
      });
      expect(node.workletNode).toBeDefined();

      let receivedEvent: TelemetrySnapshot | undefined;
      node.addEventListener("telemetry", (ev) => {
        receivedEvent = (ev as CustomEvent<TelemetrySnapshot>).detail;
      });
      const telPayload = new Float64Array(56);
      telPayload[2] = 48_000;
      telPayload[32] = 6.25;
      lastCreatedWorkletNode?.port.onmessage?.({
        data: { type: "telemetry", data: telPayload },
      });
      expect(receivedEvent?.sampleRateHz).toBe(48_000);
      expect(node.parameters.get("gain").value).toBeCloseTo(6.25, 4);

      const dest = {} as AudioNode;
      expect(node.connect(dest)).toBe(dest);
      node.disconnect();
      expect(lastCreatedWorkletNode?.disconnected).toBe(true);
    } finally {
      g["BaseAudioContext"] = prevBaseCtx;
      g["AudioWorkletNode"] = prevWorkletNode;
    }
  });

  it("supports initWasm({ module | bytes }) and validates input coercion helpers", async () => {
    // verify[sdk.ts.api.sync-init]
    // verify[sdk.ts.api.literal-unions-and-options]
    const wasmBytes = readDistWasmBytes();
    const modFromBytes = await initWasm({ bytes: wasmBytes });
    expect(modFromBytes).toBeInstanceOf(WebAssembly.Module);
    expect(getCompiledWasmModule()).toBe(modFromBytes);

    const modFromModule = await initWasm({ module: modFromBytes });
    expect(modFromModule).toBe(modFromBytes);

    expect(coerceTier(0)).toBe("lite");
    expect(coerceTier("ULTRA")).toBe("ultra");
    expect(() => coerceTier(99)).toThrow(DnamError);
    expect(() => coerceTier("invalid")).toThrow(DnamError);
    expect(coerceBackend("arm_neon")).toBe("neon");
    expect(() => coerceBackend("invalid")).toThrow(DnamError);
    expect(() => coerceSubdivision("1/99")).toThrow(DnamError);
    expect(coerceBytes(new DataView(new ArrayBuffer(4)))).toHaveLength(4);
    expect(() => coerceBytes(123 as unknown as Uint8Array)).toThrow(DnamError);
    expect(() => coerceAudioChannels([] as unknown as Float32Array[])).toThrow(DnamError);
    expect(() => coerceAudioChannels([new Float32Array(4), new Float32Array(8)])).toThrow(
      DnamError,
    );
  });

  it("property: fast-check verifies WorkletSharedState lock-free parameter ring and seqlock telemetry roundtrips", () => {
    // verify[sdk.ts.web-audio.realtime-bridge]
    const shared = new WorkletSharedState();

    fc.assert(
      fc.property(
        fc.array(fc.float({ min: -10.0, max: 10.0, noNaN: true }), { minLength: 3, maxLength: 8 }),
        fc.integer({ min: 0, max: 7 }),
        fc.float({ min: -10.0, max: 10.0, noNaN: true }),
        fc.boolean(),
        fc.array(fc.double({ min: -1000.0, max: 1000.0, noNaN: true }), {
          minLength: 56,
          maxLength: 56,
        }),
        (defaults, idxRaw, val, immediate, telValues) => {
          const defF32 = new Float32Array(defaults);
          shared.initDefaults(defF32, 1, 48_000);
          const idx = idxRaw % defF32.length;
          shared.writeParameter(idx, val, immediate);
          expect(shared.readParameter(idx)).toBeCloseTo(Math.fround(val), 6);

          const telIn = new Float64Array(telValues);
          shared.publishTelemetry(telIn, defF32.length);
          const telOut = new Float64Array(56);
          shared.readTelemetryInto(telOut);
          expect(telOut).toEqual(telIn);
        },
      ),
      { numRuns: 50 },
    );
  });
});
