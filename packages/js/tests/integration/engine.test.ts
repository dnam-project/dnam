import { beforeAll, describe, expect, it } from "vitest";
import {
  type AudioBufferLike,
  DnamContainer,
  DnamEngine,
  DnamError,
  DnamProcessor,
  DnamValidationError,
  initDnamWasm,
} from "../../src/index.js";
import { BACKENDS, ensureWasmReady, getWasmBytes } from "../helpers.js";

interface RefControls {
  gain: number;
  boost: "off" | "on" | "ultra";
  [key: string]: number | string;
}

describe("DnamEngine & DnamProcessor Real-Time Inference", () => {
  beforeAll(async () => {
    await ensureWasmReady();
  });

  describe.each(BACKENDS)("DnamEngine (%s)", (backend) => {
    it("guarantees exact 0.00000000 silence on zero input at Sample 0 (CL-02)", () => {
      const container = DnamContainer.createReference(true, backend);
      const engine = new DnamEngine<RefControls>(container, {
        tier: "full",
        sampleRateHz: 48_000,
        backend,
      });
      expect(engine.backend).toBeTruthy();
      if (backend === "wasm") {
        expect(engine.backend).toBe("wasm_simd128");
      }
      expect(engine.activeTier).toBe(1);
      expect(engine.activeTierName).toBe("full");
      expect(engine.sampleRateHz).toBe(48_000);
      expect(engine.preamble.core2Active).toBe(true);
      expect(engine.header.metadata.name).toBeTruthy();

      const zerosL = new Float32Array(128);
      const zerosR = new Float32Array(128);
      const [outL, outR] = engine.processStereo(zerosL, zerosR);
      expect(outL[0]).toBe(0.0);
      expect(outR[0]).toBe(0.0);
      for (let i = 0; i < 128; i++) {
        expect(Math.abs(outL[i] ?? 0)).toBeLessThan(1e-7);
        expect(Math.abs(outR[i] ?? 0)).toBeLessThan(1e-7);
      }
      engine.dispose();
      expect(() => engine.backend).toThrow(DnamError);
    });

    it("processes stereo, mono-to-stereo, mono, planar, AudioBuffer, and async blocks", async () => {
      const bytes = DnamContainer.createReference(true, backend).bytes;
      const engine = new DnamProcessor<RefControls>(bytes, {
        tier: "full",
        sampleRateHz: 48_000,
        backend,
      });

      const n = 256;
      const sigL = new Float32Array(n);
      const sigR = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        sigL[i] = 0.25 * Math.sin((2 * Math.PI * 440 * i) / 48_000);
        sigR[i] = 0.25 * Math.cos((2 * Math.PI * 440 * i) / 48_000);
      }

      const [stL, stR] = engine.process(sigL, sigR, {
        bufferSize: 64,
        controls: { gain: 6.0, boost: "on" },
        immediateControls: true,
      });
      expect(stL).toHaveLength(n);
      expect(stR).toHaveLength(n);

      // In-place aliasing protection
      const inPlaceL = sigL.slice();
      const inPlaceR = sigR.slice();
      engine.reset();
      engine.setControls({ gain: 6.0, boost: "on" }, { immediate: true });
      engine.processStereo(inPlaceL, inPlaceR, {
        leftOut: inPlaceL,
        rightOut: inPlaceR,
        bufferSize: 64,
      });
      for (let i = 0; i < n; i++) {
        expect(inPlaceL[i]).toBeCloseTo(stL[i] ?? 0, 5);
        expect(inPlaceR[i]).toBeCloseTo(stR[i] ?? 0, 5);
      }

      // Mono-to-stereo and Mono
      const [m2sL, m2sR] = engine.process(sigL);
      expect(m2sL).toHaveLength(n);
      expect(m2sR).toHaveLength(n);

      const monoOut = engine.processMono(sigL);
      expect(monoOut).toHaveLength(n);

      // Planar (stereo, mono-to-stereo, mono, and empty guard)
      const planarOutL = new Float32Array(n);
      const planarOutR = new Float32Array(n);
      engine.processPlanar([sigL, sigR], [planarOutL, planarOutR]);
      engine.processPlanar([sigL], [planarOutL, planarOutR]);
      engine.processPlanar([sigL], [planarOutL]);
      engine.processPlanar([], []);

      // AudioBufferLike
      const mockBuffer: AudioBufferLike = {
        numberOfChannels: 2,
        length: n,
        sampleRate: 44_100,
        getChannelData: (c) => (c === 0 ? sigL : sigR),
      };
      const outAudioBuffer = engine.processAudioBuffer(mockBuffer);
      expect(outAudioBuffer.numberOfChannels).toBe(2);
      expect(outAudioBuffer.length).toBe(n);
      expect(engine.sampleRateHz).toBe(44_100);
      expect(() => outAudioBuffer.getChannelData(5)).toThrow(DnamValidationError);

      // Async processing on worker pool / promise
      engine.reset();
      const [asyncL, asyncR] = await engine.processAsync(sigL, sigR, { bufferSize: 128 });
      expect(asyncL).toHaveLength(n);
      expect(asyncR).toHaveLength(n);

      const [asyncMonoL, asyncMonoR] = await engine.processAsync(sigL);
      expect(asyncMonoL).toHaveLength(n);
      expect(asyncMonoR).toHaveLength(n);

      engine[Symbol.dispose]();
    });

    it("supports tier switching, sample-rate ZOH rescaling, control slewing, presets, reload, and telemetry", () => {
      const c1 = DnamContainer.createReference(false, backend);
      const c2 = DnamContainer.createReference(true, backend);
      const engine = new DnamEngine<RefControls>(c1, {
        tier: "lite",
        sampleRateHz: 48_000,
        backend,
      });

      expect(engine.activeTier).toBe(0);
      expect(engine.activeTierName).toBe("lite");

      engine.setTier("ultra");
      expect(engine.activeTier).toBe(2);
      expect(engine.activeTierName).toBe("ultra");

      engine.setSampleRate(96_000);
      expect(engine.sampleRateHz).toBe(96_000);

      // Numeric and string control updates
      engine.setControlValue("gain", 7.5);
      engine.setControlImmediate("boost", "on");
      engine.setControlValue(0, 6.0);
      engine.setControlImmediate(0, 5.0);

      expect(() => engine.setControlValue(20, 1.0)).toThrow(DnamValidationError);
      expect(() => engine.setControlValue(0, "invalid")).toThrow(DnamValidationError);
      expect(() => engine.setControlImmediate(-1, 1.0)).toThrow(DnamValidationError);
      expect(() => engine.setControlImmediate(0, Number.NaN)).toThrow(DnamValidationError);

      // Apply preset if available or verify error on unknown preset
      if (engine.controls.presets.length > 0) {
        const firstPreset = engine.controls.presets[0]?.id;
        if (firstPreset) {
          const snap = engine.applyPreset(firstPreset);
          expect(snap.clampedControls).toBeDefined();
        }
      }

      // Hot-swap reload with Core 2 container
      engine.reload(c2.bytes, { tier: "full", sampleRateHz: 48_000 });
      expect(engine.activeTierName).toBe("full");
      expect(engine.sampleRateHz).toBe(48_000);
      expect(engine.preamble.core2Active).toBe(true);

      const telem = engine.telemetry();
      expect(telem.activeTier).toBe(1);
      expect(telem.sampleRateHz).toBe(48_000);
      expect(telem.zHead).toHaveLength(4);
      expect(telem.envStates).toHaveLength(4);
      expect(telem.lfoHarmonics).toHaveLength(8);
      expect(telem.clampedControls).toHaveLength(16);

      engine.dispose();
    });
  });

  it("maintains strict numerical parity (maxAbsDiff < 1e-5) between native N-API and WASM SIMD128", () => {
    const container = DnamContainer.createReference(true, "native");
    const nativeEngine = new DnamEngine<RefControls>(container, {
      tier: "full",
      sampleRateHz: 48_000,
      backend: "native",
    });
    const wasmEngine = new DnamEngine<RefControls>(container.bytes, {
      tier: "full",
      sampleRateHz: 48_000,
      backend: "wasm",
    });

    nativeEngine.setControls({ gain: 7.0, boost: "on" }, { immediate: true });
    wasmEngine.setControls({ gain: 7.0, boost: "on" }, { immediate: true });

    const n = 512;
    const leftIn = new Float32Array(n);
    const rightIn = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      leftIn[i] = 0.35 * Math.sin((2 * Math.PI * 220 * i) / 48_000);
      rightIn[i] = 0.35 * Math.cos((2 * Math.PI * 330 * i) / 48_000);
    }

    const [natL, natR] = nativeEngine.processStereo(leftIn, rightIn);
    const [wasmL, wasmR] = wasmEngine.processStereo(leftIn, rightIn);

    let maxDiff = 0;
    for (let i = 0; i < n; i++) {
      const diffL = Math.abs((natL[i] ?? 0) - (wasmL[i] ?? 0));
      const diffR = Math.abs((natR[i] ?? 0) - (wasmR[i] ?? 0));
      if (diffL > maxDiff) maxDiff = diffL;
      if (diffR > maxDiff) maxDiff = diffR;
    }
    expect(maxDiff).toBeLessThan(1e-5);

    // Exercise dynamic scratch arena growth (> 4096 samples) on WASM
    const bigL = new Float32Array(4608);
    const bigR = new Float32Array(4608);
    const [bigOutL, bigOutR] = wasmEngine.processStereo(bigL, bigR);
    expect(bigOutL).toHaveLength(4608);
    expect(bigOutR).toHaveLength(4608);

    nativeEngine.dispose();
    wasmEngine.dispose();
  });

  it("initializes WasmDriver from WebAssembly.Module, Response, Promise<Response>, and cached state", async () => {
    const wasmBytes = getWasmBytes();
    const wasmSlice = wasmBytes.slice().buffer;
    const mod = new WebAssembly.Module(wasmSlice);
    expect((await initDnamWasm(mod)).kind).toBe("wasm");
    expect((await initDnamWasm()).kind).toBe("wasm");
    expect((await initDnamWasm(new Response(wasmSlice, { status: 200 }))).kind).toBe("wasm");
    expect(
      (await initDnamWasm(Promise.resolve(new Response(wasmSlice, { status: 200 })))).kind,
    ).toBe("wasm");
    await expect(
      initDnamWasm(new Response(null, { status: 404, statusText: "Not Found" })),
    ).rejects.toThrow(DnamError);
    // Restore cached binary buffer
    await initDnamWasm(wasmSlice);
  });
});
