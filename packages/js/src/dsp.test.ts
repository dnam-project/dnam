import * as fc from "fast-check";
import { beforeAll, describe, expect, it } from "vitest";
import { BACKENDS, ensureWasmReady } from "../tests/helpers.js";
import {
  givensHouseholderMix,
  lagrangeWeights,
  ModulatedUfdn,
  PolyAdaa,
  SelectiveRgLru,
  unitRotor,
} from "./index.js";

describe("DSP Reference Kernels (Core 1 & Core 2)", () => {
  beforeAll(async () => {
    await ensureWasmReady();
  });

  describe.each(BACKENDS)("PolyAdaa (%s)", (backend) => {
    it("guarantees exact 0.00000000 DC resting equilibrium at Sample 0 (CL-02)", () => {
      const adaa = new PolyAdaa({ bConv: 0.45, alpha: 1.0, beta: 0.5, delta: 0.1 }, backend);
      expect(adaa.step(0.0)).toBe(0.0);

      const zeros = new Float64Array(64);
      const out = adaa.process(zeros, { reset: true });
      for (const sample of out) {
        expect(Math.abs(sample)).toBeLessThan(1e-15);
      }
    });

    it("evaluates division-free algebraic segment average (CL-01) smoothly across uPrev === uCurr", () => {
      fc.assert(
        fc.property(
          fc.double({ min: -1.0, max: 1.0, noNaN: true }),
          fc.double({ min: -1e-7, max: 1e-7, noNaN: true }),
          (u0, eps) => {
            const u1 = Math.min(1.0, Math.max(-1.0, u0 + eps));
            const exact = PolyAdaa.algebraic(u0, u0, 0.5, 0.1, backend);
            const nearby = PolyAdaa.algebraic(u0, u1, 0.5, 0.1, backend);
            expect(Number.isFinite(exact)).toBe(true);
            expect(Math.abs(nearby - exact)).toBeLessThan(1e-5);
          },
        ),
        { numRuns: 40 },
      );
    });

    it("matches stateful step() and vectorized scan()", () => {
      const signal = new Float64Array(32);
      for (let i = 0; i < signal.length; i++) {
        signal[i] = 0.5 * Math.sin(0.2 * i);
      }
      const scanned = PolyAdaa.scan(
        signal,
        { bConv: 0.1, alpha: 1.2, beta: 0.5, delta: 0.1 },
        { backend },
      );
      const adaa = new PolyAdaa({ bConv: 0.1, alpha: 1.2, beta: 0.5, delta: 0.1 }, backend);
      for (let i = 0; i < signal.length; i++) {
        expect(adaa.step(signal[i] ?? 0)).toBeCloseTo(scanned[i] ?? 0, 12);
      }
    });
  });

  it("produces identical PolyAdaa scan outputs across native and wasm", () => {
    const signal = new Float64Array(64);
    for (let i = 0; i < signal.length; i++) {
      signal[i] = 0.8 * Math.cos(0.15 * i);
    }
    const nativeOut = PolyAdaa.scan(signal, { bConv: 0.25 }, { backend: "native" });
    const wasmOut = PolyAdaa.scan(signal, { bConv: 0.25 }, { backend: "wasm" });
    for (let i = 0; i < signal.length; i++) {
      expect(nativeOut[i]).toBeCloseTo(wasmOut[i] ?? 0, 12);
    }
  });

  describe.each(BACKENDS)("SelectiveRgLru (%s)", (backend) => {
    it("matches step(), process(), and static scan() and remains BIBO stable (CL-03)", () => {
      const input = new Float64Array(48);
      for (let i = 0; i < input.length; i++) {
        input[i] = i === 0 ? 1.0 : 0.2 * Math.sin(0.3 * i);
      }
      const scanned = SelectiveRgLru.scan(input, { rLow: 0.9, drSpan: 0.05 }, { backend });
      const rglru = new SelectiveRgLru({ rLow: 0.9, drSpan: 0.05 }, backend);
      for (let i = 0; i < input.length; i++) {
        const y = rglru.step(input[i] ?? 0);
        expect(y).toBeCloseTo(scanned[i] ?? 0, 12);
        expect(Math.abs(y)).toBeLessThan(2.0);
      }
      rglru.reset();
      expect(rglru.zRe).toBe(0);
      expect(rglru.zIm).toBe(0);
    });
  });

  describe.each(BACKENDS)("ModulatedUfdn (%s)", (backend) => {
    it("enforces coercive saturation ceiling (CL-09..CL-11) on stepCoercive and scan", () => {
      const ufdn = new ModulatedUfdn(
        { delaySamples: 16, gLin: 0.85, sigmaKnee: 0.25, vCeil: 0.985, rDc: 0.995 },
        backend,
      );
      for (let i = 0; i < 32; i++) {
        const s = ufdn.stepCoercive(10.0 * (i % 2 === 0 ? 1 : -1));
        expect(Number.isFinite(s)).toBe(true);
        expect(Math.abs(s)).toBeLessThan(2.0);
      }
      ufdn.reset();
      expect(ufdn.dcXPrev).toBe(0);
      expect(ufdn.dcYPrev).toBe(0);

      const impulse = new Float64Array(64);
      impulse[0] = 1.0;
      const tail = ModulatedUfdn.scan(impulse, { delaySamples: 16, gLin: 0.85 }, { backend });
      expect(tail[16]).not.toBe(0);
    });
  });

  describe.each(BACKENDS)("unitRotor, lagrangeWeights & givensHouseholderMix (%s)", (backend) => {
    it("maintains unit norm on unitRotor scalar and slice calls", () => {
      fc.assert(
        fc.property(fc.double({ min: -0.5, max: 0.5, noNaN: true }), (phi) => {
          const [c, s] = unitRotor(phi, backend);
          expect(c * c + s * s).toBeCloseTo(1.0, 5);
        }),
        { numRuns: 30 },
      );

      const phis = new Float64Array([-0.25, 0.0, 0.125, 0.5]);
      const vecOut = unitRotor(phis, undefined, backend);
      expect(vecOut.length).toBe(8);
      for (let i = 0; i < phis.length; i++) {
        const c = vecOut[i] ?? 0;
        const s = vecOut[phis.length + i] ?? 0;
        expect(c * c + s * s).toBeCloseTo(1.0, 5);
      }

      const prealloc = new Float64Array(8);
      const returned = unitRotor(phis, prealloc, backend);
      expect(returned).toBe(prealloc);
    });

    it("satisfies partition of unity on 4-tap Cubic Lagrange weights (CL-06)", () => {
      fc.assert(
        fc.property(fc.double({ min: 0.0, max: 1.0, noNaN: true }), (mu) => {
          const [w0, w1, w2, w3] = lagrangeWeights(mu, backend);
          expect(w0 + w1 + w2 + w3).toBeCloseTo(1.0, 12);
        }),
        { numRuns: 40 },
      );
    });

    it("preserves exact L2 energy on 8-channel Givens + Householder mixing (CL-08)", () => {
      fc.assert(
        fc.property(
          fc.array(fc.double({ min: -2.0, max: 2.0, noNaN: true }), {
            minLength: 8,
            maxLength: 8,
          }),
          fc.double({ min: -Math.PI, max: Math.PI, noNaN: true }),
          (arr, psi) => {
            const cosPsi = Math.cos(psi);
            const sinPsi = Math.sin(psi);
            const y = givensHouseholderMix(arr, cosPsi, sinPsi, backend);
            let eIn = 0;
            let eOut = 0;
            for (let i = 0; i < 8; i++) {
              eIn += (arr[i] ?? 0) ** 2;
              eOut += (y[i] ?? 0) ** 2;
            }
            expect(eOut).toBeCloseTo(eIn, 10);
          },
        ),
        { numRuns: 40 },
      );
    });
  });
});
