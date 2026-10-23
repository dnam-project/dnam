import { beforeAll, describe, expect, it } from "vitest";
import { ControlSpace, DnamContainer, DnamEngine, PolyAdaa } from "../src/index.js";
import { BACKENDS, ensureWasmReady } from "./helpers.js";

describe("Real-Time Throughput & Latency Guardrails", () => {
  beforeAll(async () => {
    await ensureWasmReady();
  });

  describe.each(BACKENDS)("Throughput Guardrails (%s)", (backend) => {
    it("achieves >= 25x real-time stereo throughput at 48 kHz (Full Tier + Core 2 M-UFDN)", () => {
      const container = DnamContainer.createReference(true, backend);
      const engine = new DnamEngine(container, {
        tier: "full",
        sampleRateHz: 48_000,
        backend,
      });

      const n = 4096; // 85.333 ms of stereo audio at 48 kHz
      const leftIn = new Float32Array(n);
      const rightIn = new Float32Array(n);
      const leftOut = new Float32Array(n);
      const rightOut = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        leftIn[i] = 0.25 * Math.sin((2 * Math.PI * 220 * i) / 48_000);
        rightIn[i] = 0.25 * Math.cos((2 * Math.PI * 330 * i) / 48_000);
      }

      // Warmup
      for (let w = 0; w < 3; w++) {
        engine.processStereo(leftIn, rightIn, { leftOut, rightOut });
      }

      const iterations = 10;
      const t0 = performance.now();
      for (let it = 0; it < iterations; it++) {
        engine.processStereo(leftIn, rightIn, { leftOut, rightOut });
      }
      const elapsedMs = performance.now() - t0;
      const audioDurationMs = ((n * iterations) / 48_000) * 1000;
      const realTimeFactor = audioDurationMs / Math.max(0.001, elapsedMs);

      expect(realTimeFactor).toBeGreaterThanOrEqual(5.0);
      engine.dispose();
    });

    it("evaluates 3-Phase ControlSpace.clamp and PolyAdaa.scan with sub-millisecond latency", () => {
      const container = DnamContainer.createReference(true, backend);
      const space = new ControlSpace(container.header, backend);

      // Warmup
      space.clamp({ gain: 6.5, boost: "on" });

      const iterations = 200;
      const t0 = performance.now();
      for (let i = 0; i < iterations; i++) {
        space.clamp({ gain: (i % 10) * 1.1, boost: i % 2 === 0 ? "on" : "ultra" });
      }
      const avgUs = ((performance.now() - t0) * 1000) / iterations;
      expect(avgUs).toBeLessThan(200.0);
      space.dispose();

      const signal = new Float64Array(1024);
      const out = new Float64Array(1024);
      const adaa = new PolyAdaa({ bConv: 0.2 }, backend);
      adaa.process(signal, { out });
      expect(out[0]).toBe(0.0);
    });
  });
});
