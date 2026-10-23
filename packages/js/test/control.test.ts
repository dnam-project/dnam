import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  ControlSpace,
  chebyshevLfo,
  clampControlValues,
  DnamContainer,
  DnamControlError,
  DnamEngine,
  DnamError,
  DnamValidationError,
  LobattoBasis,
  Rank4Fm,
  trustRegionSoftclip,
} from "../src/index.js";
import { BACKENDS, maxAbsDiff } from "./helpers.js";

describe.each(BACKENDS)("Parametric Control Bus & Basis Suite (%s)", (backend) => {
  it("evaluates 3-Phase control clamping ([CLAMP-RANGE], [CLAMP-SWITCH], conditional bounds)", () => {
    const container = DnamContainer.buildReference({ core2Active: false, backend });
    using space = new ControlSpace(container.header, backend);

    // 1. Default values (unclamped)
    const snap0 = space.clamp({ gain: 5.0, boost: "off" });
    expect(snap0.clampedControls["gain"]).toBeCloseTo(5.0, 5);
    expect(snap0.clampedControls["boost"]).toBe("off");
    expect(snap0.wasClamped["gain"]).toBe(false);
    expect(snap0.wasClamped["boost"]).toBe(false);

    // 2. [CLAMP-RANGE] when boost=off (safe_clamped_range = [0.0, 9.5])
    const snapHigh = space.clamp({ gain: 10.0, boost: "off" });
    expect(snapHigh.clampedControls["gain"]).toBeCloseTo(9.5, 5);
    expect(snapHigh.wasClamped["gain"]).toBe(true);
    expect(snapHigh.warnings["gain"]).toContain("[CLAMP-RANGE]");

    // 3. Switch-conditioned range clamp when boost=on (conditional_max = 8.0)
    const snapCond = space.clamp({ gain: 9.0, boost: "on" });
    expect(snapCond.clampedControls["gain"]).toBeCloseTo(8.0, 5);
    expect(snapCond.wasClamped["gain"]).toBe(true);

    // 4. [CLAMP-SWITCH] unobserved option "ultra" -> falls back to active option
    const snapSw = clampControlValues(container.header, { gain: 5.0, boost: "ultra" }, backend);
    expect(snapSw.wasClamped["boost"]).toBe(true);
    expect(snapSw.clampedControls["boost"]).not.toBe("ultra");
    expect(snapSw.warnings["boost"]).toContain("[CLAMP-SWITCH]");

    // 5. Boolean & numeric switch coercion + error paths
    expect(space.clamp({ boost: true }).clampedControls["boost"]).toBe("on");
    expect(space.clamp({ boost: 0 }).clampedControls["boost"]).toBe("off");
    expect(() => space.clamp({ unknownCtrl: 1.0 })).toThrow(DnamControlError);
    expect(() => space.clamp({ boost: "nonexistent" })).toThrow(DnamControlError);
    expect(() => space.clamp({ gain: "not_a_number" })).toThrow(DnamControlError);
    expect(() => space.clamp({ boost: Number.NaN })).toThrow(DnamControlError);

    space.dispose();
    expect(() => space.clamp({})).toThrow(DnamError);
  });

  it("supports ControlRack<TControls> gesture skew, presets, and reactive updates", () => {
    interface TestControls {
      gain: number;
      boost: "off" | "on" | "ultra";
      [key: string]: number | string | boolean;
    }

    const container = DnamContainer.buildReference({ core2Active: false, backend });
    using engine = new DnamEngine<TestControls>(container, { backend });
    const rack = engine.controls;

    expect(rack.schema).toHaveLength(2);
    expect(rack.presets).toHaveLength(1);
    expect(rack.get("gain")).toBeCloseTo(5.0, 5);

    rack.set("gain", 7.5, { immediate: true });
    expect(rack.get("gain")).toBeCloseTo(7.5, 5);

    rack.setAll({ gain: 9.2, boost: "on" }, { immediate: true });
    // Clamped to 8.0 when boost=on!
    expect(rack.get("gain")).toBeCloseTo(8.0, 5);
    expect(rack.get("boost")).toBe("on");

    const presetSnap = rack.applyPreset("warm_slapback");
    expect(presetSnap.clampedControls["gain"]).toBeCloseTo(6.5, 5);
    expect(presetSnap.clampedControls["boost"]).toBe("on");
    expect(() => rack.applyPreset("missing_preset")).toThrow(DnamControlError);
    expect(() => rack.get("missing" as "gain")).toThrow(DnamControlError);

    // Midpoint power-skew gesture helpers on ControlRack
    const gamma = rack.skewExponent("gain");
    expect(gamma).not.toBe(1.0);
    const midPhys = rack.normalizedToPhysical("gain", 0.5);
    expect(midPhys).toBeCloseTo(4.0, 3);
    expect(rack.physicalToNormalized("gain", midPhys)).toBeCloseTo(0.5, 3);
  });

  it("evaluates Node-Anchored LobattoBasis (CL-13) with zero-flop default preset anchor", () => {
    // At u = u_1 = 0.0, all 3 basis functions are identically 0.0
    const atDefault = LobattoBasis.eval(0.0, {
      anchors: [0.0, -1.0, 1.0],
      degree: 3,
      backend,
    });
    expect(atDefault[0]).toBeCloseTo(0.0, 12);
    expect(atDefault[1]).toBeCloseTo(0.0, 12);
    expect(atDefault[2]).toBeCloseTo(0.0, 12);

    const uGrid = new Float64Array([0.0, -0.5, 0.5, 1.0]);
    const sliceOut = LobattoBasis.evalSlice(uGrid, undefined, {
      anchors: [0.0, -1.0, 1.0],
      degree: 3,
      backend,
    });
    expect(sliceOut.length).toBe(12);
    for (let i = 0; i < uGrid.length; i++) {
      const single = LobattoBasis.eval(uGrid[i] ?? 0, {
        anchors: [0.0, -1.0, 1.0],
        degree: 3,
        backend,
      });
      expect(sliceOut[i * 3]).toBeCloseTo(single[0], 12);
      expect(sliceOut[i * 3 + 1]).toBeCloseTo(single[1], 12);
      expect(sliceOut[i * 3 + 2]).toBeCloseTo(single[2], 12);
    }

    expect(() => LobattoBasis.evalSlice(uGrid, new Float64Array(5), { backend })).toThrow(
      DnamValidationError,
    );
  });

  it("evaluates Rank-4 Factorization Machine (CL-14), Chebyshev LFO (CL-16), and Trust-Region Soft-Clipper (CL-15)", () => {
    const features = new Float64Array([0.5, -0.25, 0.75]);
    const vFactors = new Float64Array([
      1.0, 0.5, -0.5, 0.25, 0.5, -1.0, 0.25, 0.5, -0.25, 0.5, 1.0, -0.5,
    ]);
    const fm = Rank4Fm.eval(features, vFactors, backend);
    expect(fm).toHaveLength(4);
    // Verify against exact O(D*R) pairwise interaction formula
    for (let r = 0; r < 4; r++) {
      let pairSum = 0.0;
      for (let i = 0; i < 3; i++) {
        for (let j = i + 1; j < 3; j++) {
          pairSum +=
            (vFactors[i * 4 + r] ?? 0) *
            (vFactors[j * 4 + r] ?? 0) *
            (features[i] ?? 0) *
            (features[j] ?? 0);
        }
      }
      expect(fm[r]).toBeCloseTo(pairSum, 12);
    }
    expect(() => Rank4Fm.eval(features, new Float64Array(5), backend)).toThrow(DnamValidationError);

    // 4-Harmonic Chebyshev LFO
    const theta = 0.42;
    const c = Math.cos(theta);
    const cheb = chebyshevLfo(c, backend);
    expect(cheb[0]).toBeCloseTo(Math.cos(theta), 12);
    expect(cheb[1]).toBeCloseTo(Math.cos(2 * theta), 12);
    expect(cheb[2]).toBeCloseTo(Math.cos(3 * theta), 12);
    expect(cheb[3]).toBeCloseTo(Math.cos(4 * theta), 12);

    const cVec = new Float64Array([Math.cos(0.1), Math.cos(0.7)]);
    const chebVec = chebyshevLfo(cVec, undefined, backend);
    expect(chebVec.length).toBe(8);
    expect(maxAbsDiff(chebVec.subarray(0, 4), chebyshevLfo(Math.cos(0.1), backend))).toBeLessThan(
      1e-12,
    );

    // Trust-Region Soft-Clipper (CL-15)
    expect(trustRegionSoftclip(0.0, 2.5, backend)).toBe(0.0);
    expect(trustRegionSoftclip(100.0, 2.5, backend)).toBeCloseTo(2.5, 12);
    expect(trustRegionSoftclip(-100.0, 2.5, backend)).toBeCloseTo(-2.5, 12);

    const zArr = new Float64Array([-10.0, -1.0, 0.0, 1.0, 10.0]);
    const clippedArr = trustRegionSoftclip(zArr, 2.5, undefined, backend);
    expect(clippedArr.length).toBe(5);

    const largeArr = new Float64Array(36_000);
    largeArr[0] = 100.0;
    const largeClipped = trustRegionSoftclip(largeArr, 2.5, undefined, backend);
    expect(largeClipped[0]).toBeCloseTo(2.5, 12);

    fc.assert(
      fc.property(
        fc.double({ min: -20, max: 20, noNaN: true }),
        fc.double({ min: 0.1, max: 5, noNaN: true }),
        (z, delta) => {
          const y1 = trustRegionSoftclip(z, 2.5, backend);
          const y2 = trustRegionSoftclip(z + delta, 2.5, backend);
          return Math.abs(y1) <= 2.5 + 1e-12 && y2 >= y1 - 1e-12;
        },
      ),
      { numRuns: 30 },
    );
  });
});
