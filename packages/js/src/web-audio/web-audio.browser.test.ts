/**
 * Real headless Chromium browser tests (`@vitest/browser-playwright`) for WebAssembly `SIMD128`
 * (`--import-memory`, bounded & `shared: true` `WebAssembly.Memory`) and WebAudio `AudioWorklet`
 * (`OfflineAudioContext` + `AudioWorkletNode` + `DnamAudioWorkletProcessor`).
 */

import { describe, expect, it } from "vitest";
import { DnamError, initWasm, Model } from "../browser.js";
import { getCompiledWasmModule, instantiateWasmDriver } from "../internal/native/wasm.js";
import {
  createWasmMemory,
  DnamAudioNode,
  isSharedMemorySupported,
  isSharedWasmMemorySupported,
} from "./index.js";

describe("Chromium Browser WASM (--import-memory / SharedArrayBuffer) & WebAudio AudioWorklet", () => {
  it("runs in a crossOriginIsolated Chromium context with SharedArrayBuffer and shared WebAssembly.Memory", async () => {
    // verify[sdk.ts.api.sync-init]
    // verify[sdk.ts.engine.zero-copy-and-async]
    expect(globalThis.crossOriginIsolated).toBe(true);
    expect(isSharedMemorySupported()).toBe(true);
    expect(isSharedWasmMemorySupported()).toBe(true);

    const unsharedMod = await initWasm({ initialPages: 24, maximumPages: 1024 });
    expect(unsharedMod).toBeInstanceOf(WebAssembly.Module);
    expect(getCompiledWasmModule(false)).toBe(unsharedMod);
    expect(WebAssembly.Module.imports(unsharedMod)).toContainEqual({
      module: "env",
      name: "memory",
      kind: "memory",
    });

    const sharedMem = createWasmMemory({ initialPages: 32, maximumPages: 1024, shared: true });
    expect(sharedMem.buffer).toBeInstanceOf(SharedArrayBuffer);

    const sharedMod = await initWasm({ memory: sharedMem, shared: true });
    expect(sharedMod).toBeInstanceOf(WebAssembly.Module);
    expect(getCompiledWasmModule(true)).toBe(sharedMod);

    using model = Model.reference({ supportsFreeze: true });
    expect(model.name.length).toBeGreaterThan(0);

    using engine = model.createEngine({ tier: "full", sampleRate: 48_000 });
    const inL = new Float32Array(128).fill(0.25);
    const inR = new Float32Array(128).fill(-0.25);
    const outL = new Float32Array(128);
    const outR = new Float32Array(128);
    engine.process([inL, inR], { output: [outL, outR] });
    expect(outL.some((s) => s !== 0 && Number.isFinite(s))).toBe(true);
    expect(outR.some((s) => s !== 0 && Number.isFinite(s))).toBe(true);

    // Verify bounded WebAssembly.Memory OOM throws structured DnamError without trapping WASM
    const tinyMem = new WebAssembly.Memory({ initial: 18, maximum: 22 });
    const tinyDriver = instantiateWasmDriver(unsharedMod, { memory: tinyMem });
    expect(() => tinyDriver.validateContainer(new Uint8Array(512 * 1024))).toThrow(DnamError);
    expect(() =>
      instantiateWasmDriver(unsharedMod, {
        memory: new WebAssembly.Memory({ initial: 18, maximum: 20 }),
      }),
    ).toThrow(DnamError);
  });

  it("executes DnamAudioNode + DnamAudioWorkletProcessor inside real Chromium OfflineAudioContext with shared and unshared WebAssembly.Memory", async () => {
    // verify[sdk.ts.web-audio.realtime-bridge]
    await initWasm();
    using model = Model.reference({ supportsFreeze: true });

    // 1. Cross-thread shared: true WebAssembly.Memory + SharedArrayBuffer lock-free WorkletSharedState
    const offlineShared = new OfflineAudioContext(2, 128 * 12, 48_000);
    using sharedNode = await DnamAudioNode.create(offlineShared, model, {
      tier: "lite",
      sharedWasmMemory: true,
      initialMemoryPages: 32,
      maximumMemoryPages: 512,
      telemetryIntervalBlocks: 1,
    });
    expect(sharedNode.usesSharedMemory).toBe(true);
    expect(sharedNode.workletNode).toBeInstanceOf(AudioWorkletNode);
    expect(sharedNode.wasmMemory?.buffer).toBeInstanceOf(SharedArrayBuffer);

    sharedNode.parameters.get("gain").setValue(7.0, { immediate: true });
    sharedNode.stereoWidth = 0.85;

    const osc1 = new OscillatorNode(offlineShared, { frequency: 220, type: "sawtooth" });
    if (sharedNode.workletNode) {
      osc1.connect(sharedNode.workletNode);
    }
    sharedNode.connect(offlineShared.destination);
    osc1.start(0);

    const renderedShared = await offlineShared.startRendering();
    const ch0Shared = renderedShared.getChannelData(0);
    const ch1Shared = renderedShared.getChannelData(1);
    expect(ch0Shared.some((x) => Math.abs(x) > 1e-5 && Number.isFinite(x))).toBe(true);
    expect(ch1Shared.some((x) => Math.abs(x) > 1e-5 && Number.isFinite(x))).toBe(true);
    expect(sharedNode.telemetry().sampleRateHz).toBe(48_000);

    // 2. Unshared WebAssembly.Memory + MessagePort fallback (verifies no DataCloneError on AudioWorkletNode)
    const offlineUnshared = new OfflineAudioContext(2, 128 * 12, 48_000);
    const unsharedMem = new WebAssembly.Memory({ initial: 32, maximum: 256 });
    using unsharedNode = await DnamAudioNode.create(offlineUnshared, model, {
      tier: "lite",
      wasmMemory: unsharedMem,
      useSharedMemory: false,
      telemetryIntervalBlocks: 1,
    });
    expect(unsharedNode.usesSharedMemory).toBe(false);
    expect(unsharedNode.workletNode).toBeInstanceOf(AudioWorkletNode);
    expect(unsharedNode.wasmMemory).toBe(unsharedMem);

    const osc2 = new OscillatorNode(offlineUnshared, { frequency: 330, type: "sine" });
    if (unsharedNode.workletNode) {
      osc2.connect(unsharedNode.workletNode);
    }
    unsharedNode.connect(offlineUnshared.destination);
    osc2.start(0);

    const renderedUnshared = await offlineUnshared.startRendering();
    const ch0Unshared = renderedUnshared.getChannelData(0);
    expect(ch0Unshared.some((x) => Math.abs(x) > 1e-5 && Number.isFinite(x))).toBe(true);
  });
});
