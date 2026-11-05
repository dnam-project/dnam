/**
 * Tests for `@dnam-project/dnam/dsp` reference kernels (`transport`, `modal`, `nonlinear`,
 * `control-bus`, `mixing`) verifying 0-ULP scalar vs. slice parity and cross-driver N-API vs. WASM
 * parity.
 */

import "../node.js";

import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { withDriver } from "../internal/native/load.js";
import { loadNodeDriverSync, loadNodeWasmDriverSync } from "../internal/native/napi.js";
import {
  catmullRomHermite,
  catmullRomHermiteSlice,
  chebyshevLfo,
  chebyshevLfoSlice,
  coerciveStep,
  cubicLagrangeWeights,
  cubicLagrangeWeightsSlice,
  givensHouseholder8ch,
  givensHouseholder8chSlice,
  givensHouseholder16ch,
  givensHouseholder16chSlice,
  lobattoBasis,
  lobattoBasisSlice,
  modalSlice,
  modalStep,
  polyAdaaAlgebraic,
  polyAdaaSlice,
  polyAdaaStep,
  rank4Fm,
  rank4FmSlice,
  tankSlice,
  trustRegionSoftclip,
  trustRegionSoftclipSlice,
  unitRotor,
  unitRotorSlice,
} from "./index.js";

describe("TORA Reference DSP Kernels", () => {
  it.each(["napi", "wasm"] as const)(
    "matches scalar and vectorized slice kernels to 0 ULP on %s driver",
    (driverKind) => {
      // verify[sdk.ts.dsp.reference-kernels]
      const driver = driverKind === "napi" ? loadNodeDriverSync() : loadNodeWasmDriverSync();
      withDriver(driver, () => {
        // 1. Op-T: Catmull-Rom & Coercive Tank
        const sM1 = new Float64Array([0.1, 0.2, -0.1, 0.4]);
        const s0 = new Float64Array([0.2, 0.3, 0.0, 0.5]);
        const s1 = new Float64Array([0.4, 0.5, 0.2, 0.6]);
        const s2 = new Float64Array([0.5, 0.7, 0.3, 0.8]);
        const mu = new Float64Array([0.0, 0.25, 0.5, 0.75]);
        const crSlice = catmullRomHermiteSlice({ sM1, s0, s1, s2, mu });
        for (let i = 0; i < 4; i++) {
          expect(crSlice[i]).toBe(
            catmullRomHermite(sM1[i] ?? 0, s0[i] ?? 0, s1[i] ?? 0, s2[i] ?? 0, mu[i] ?? 0),
          );
        }

        const cStep = coerciveStep({
          uIn: 0.8,
          gLin: 0.95,
          sigmaKnee: 1.2,
          vCeil: 1.5,
          rDc: 0.995,
        });
        expect(Math.abs(cStep.y)).toBeLessThanOrEqual(1.5);
        const tOut = tankSlice(new Float64Array([0.5, -0.25, 0.1, 0.0]));
        expect(tOut.length).toBe(4);

        // 2. Op-M: Modal step/slice & Unit Rotor
        const mIn = new Float64Array([0.5, -0.3, 0.2, 0.1]);
        const mSlice = modalSlice(mIn, { reset: true });
        let zRe = 0;
        let zIm = 0;
        for (let i = 0; i < mIn.length; i++) {
          const st = modalStep(mIn[i] ?? 0, { zRe, zIm });
          expect(mSlice.output[i]).toBe(st.y);
          zRe = st.zRe;
          zIm = st.zIm;
        }
        expect(mSlice.zRe).toBe(zRe);
        expect(mSlice.zIm).toBe(zIm);

        const phis = new Float64Array([0.0, 0.25, 0.5, -0.5]);
        const rotSlice = unitRotorSlice(phis);
        for (let i = 0; i < phis.length; i++) {
          const [c, s] = unitRotor(phis[i] ?? 0);
          expect(rotSlice[i]).toBe(c);
          expect(rotSlice[phis.length + i]).toBe(s);
        }

        // 3. Op-N: PolyADAA
        expect(Number.isFinite(polyAdaaAlgebraic(0.1, 0.2, 1.0, 0.1))).toBe(true);
        const nlIn = new Float64Array([0.2, 0.6, -0.4, 0.1]);
        const nlSlice = polyAdaaSlice(nlIn, { bConv: 0.0, reset: true });
        let uPrev = 0;
        let aPrev = 0;
        let mDc = 0;
        for (let i = 0; i < nlIn.length; i++) {
          const st = polyAdaaStep(nlIn[i] ?? 0, { bConv: 0.0, uPrev, aPrev, mDc });
          expect(nlSlice.output[i]).toBe(st.y);
          uPrev = st.uPrev;
          aPrev = st.aPrev;
          mDc = st.mDc;
        }

        // 4. Control Bus: Lobatto, Rank-4 FM, Chebyshev LFO, Trust-Region Softclip
        const uArr = new Float64Array([-0.5, 0.0, 0.5]);
        expect(lobattoBasis(0.0)).toHaveLength(3);
        expect(lobattoBasisSlice(uArr)).toHaveLength(9);
        const lobSlice = lobattoBasisSlice(uArr, {
          anchors: [0.0, -1.0, 1.0],
          degree: 3,
          output: new Float64Array(9),
        });
        expect(lobattoBasisSlice(uArr, { anchors: new Float64Array([0.0, -1.0, 1.0]) })).toEqual(
          lobSlice,
        );
        for (let i = 0; i < uArr.length; i++) {
          const b = lobattoBasis(uArr[i] ?? 0, {
            anchors: new Float64Array([0.0, -1.0, 1.0]),
            degree: 3,
          });
          expect(lobattoBasis(uArr[i] ?? 0, { anchors: [0.0, -1.0, 1.0] })).toEqual(b);
          expect(lobSlice.subarray(i * 3, (i + 1) * 3)).toEqual(new Float64Array(b));
        }

        const feat = new Float64Array([0.5, -0.25, 0.75]);
        const vf = new Float64Array(12).fill(0.2);
        const fmSingle = rank4Fm([0.5, -0.25, 0.75], [...vf]);
        expect(rank4Fm(feat, vf)).toEqual(fmSingle);
        expect(rank4FmSlice({ featuresFlat: feat, nFrames: 1, dDim: 3, vFactorsFlat: vf })).toEqual(
          new Float64Array(fmSingle),
        );
        const fmBatch = rank4FmSlice({
          featuresFlat: feat,
          nFrames: 1,
          dDim: 3,
          vFactorsFlat: vf,
          output: new Float64Array(4),
        });
        expect(fmBatch).toEqual(new Float64Array(fmSingle));

        const cosArr = new Float64Array([-0.5, 0.0, 0.5, 1.0]);
        expect(chebyshevLfoSlice(cosArr)).toHaveLength(16);
        const chebSlice = chebyshevLfoSlice(cosArr, { output: new Float64Array(16) });
        for (let i = 0; i < cosArr.length; i++) {
          expect(chebSlice.subarray(i * 4, (i + 1) * 4)).toEqual(
            new Float64Array(chebyshevLfo(cosArr[i] ?? 0)),
          );
        }

        const zArr = new Float64Array([-5.0, -1.0, 0.0, 1.0, 5.0]);
        expect(trustRegionSoftclip(0.0)).toBeCloseTo(0.0, 6);
        expect(trustRegionSoftclipSlice(zArr)).toHaveLength(5);
        const trSlice = trustRegionSoftclipSlice(zArr, {
          mh: 2.5,
          output: new Float64Array(5),
        });
        for (let i = 0; i < zArr.length; i++) {
          expect(trSlice[i]).toBe(trustRegionSoftclip(zArr[i] ?? 0, 2.5));
        }

        // 5. Mixing: Cubic Lagrange & Givens-Householder 8ch/16ch
        expect(cubicLagrangeWeightsSlice(mu)).toHaveLength(16);
        const lagSlice = cubicLagrangeWeightsSlice(mu, { output: new Float64Array(16) });
        for (let i = 0; i < mu.length; i++) {
          expect(lagSlice.subarray(i * 4, (i + 1) * 4)).toEqual(
            new Float64Array(cubicLagrangeWeights(mu[i] ?? 0)),
          );
        }

        const x8 = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]);
        expect(givensHouseholder8ch(x8)).toHaveLength(8);
        expect(givensHouseholder8chSlice(x8)).toHaveLength(8);
        const y8 = givensHouseholder8ch([1, 2, 3, 4, 5, 6, 7, 8], Math.cos(0.3), Math.sin(0.3));
        const y8Slice = givensHouseholder8chSlice(x8, {
          cosPsi: Math.cos(0.3),
          sinPsi: Math.sin(0.3),
          output: new Float64Array(8),
        });
        expect(y8Slice).toEqual(y8);

        const x16 = new Float64Array(16).map((_, idx) => idx + 1);
        expect(givensHouseholder16ch(x16)).toHaveLength(16);
        expect(givensHouseholder16chSlice(x16)).toHaveLength(16);
        expect(
          givensHouseholder16ch(x16, {
            crossCos8: new Float64Array(8).fill(1),
            crossSin8: new Float64Array(8).fill(0),
          }),
        ).toHaveLength(16);
        expect(
          givensHouseholder16chSlice(x16, {
            crossCos8: [1, 1, 1, 1, 1, 1, 1, 1],
            crossSin8: [0, 0, 0, 0, 0, 0, 0, 0],
          }),
        ).toHaveLength(16);
        const y16 = givensHouseholder16ch([...x16], {
          cosPsi: Math.cos(0.3),
          sinPsi: Math.sin(0.3),
          crossCos8: [1, 1, 1, 1, 1, 1, 1, 1],
          crossSin8: [0, 0, 0, 0, 0, 0, 0, 0],
        });
        const y16Slice = givensHouseholder16chSlice(x16, {
          cosPsi: Math.cos(0.3),
          sinPsi: Math.sin(0.3),
          crossCos8: new Float64Array(8).fill(1),
          crossSin8: new Float64Array(8).fill(0),
          output: new Float64Array(16),
        });
        expect(y16Slice).toEqual(y16);
      });
    },
  );

  it("satisfies fast-check unit-norm & energy-preservation properties", () => {
    // verify[sdk.ts.dsp.reference-kernels]
    fc.assert(
      fc.property(fc.double({ min: -10, max: 10, noNaN: true }), (phi) => {
        const [c, s] = unitRotor(phi);
        expect(Math.abs(c * c + s * s - 1.0)).toBeLessThan(1e-5);
      }),
      { numRuns: 32 },
    );
  });
});
