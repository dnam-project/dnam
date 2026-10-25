import { describe, expect, it, vi } from "vitest";
import { page, userEvent } from "vitest/browser";
import { render } from "vitest-browser-react";
import { expectZeroAxeViolations, setRangeInputValue } from "../../../tests/setup.js";
import {
  ClampAuditVisualizer,
  ControlBusVisualizer,
  computeLobattoDegree,
  computePolyAdaaMetrics,
  computeRgLruInterpolation,
  computeSurrogateDelayBasin,
  computeTrustRegionDamping,
  computeUfdnCrossfadeWeights,
  computeZohPoleRescaling,
  KroneckerResumeVisualizer,
  ModulatedUfdnVisualizer,
  PolyAdaaVisualizer,
  RgLruPoleVisualizer,
  ZohRescalerVisualizer,
} from "./index.js";

describe("Tri-Physics visualizer mathematical models", () => {
  it("computes PolyADAA, RG-LRU, M-UFDN, Lobatto, Kronecker, Trust-Region, and ZOH metrics", () => {
    const adaa = computePolyAdaaMetrics(4.0, 2000);
    expect(adaa.samplePairWeights).toEqual([0.875, 0.125]);
    expect(adaa.bsplineSuppressionDb).toBeGreaterThan(30);

    const rglru = computeRgLruInterpolation(0.995, 1.2, 0.5);
    expect(rglru.radialPolarRadius).toBeCloseTo(0.995, 5);
    expect(rglru.cartesianRadius).toBeLessThan(rglru.radialPolarRadius);
    expect(rglru.cartesianGainDropDb).toBeLessThan(0);

    const rglruZero = computeRgLruInterpolation(0.0, 0.0, 0.5);
    expect(rglruZero.cartesianGainDropDb).toBe(0);

    const ufdnStart = computeUfdnCrossfadeWeights(0);
    const ufdnEnd = computeUfdnCrossfadeWeights(128);
    expect(ufdnStart.fadeOutWeight).toBeCloseTo(1.0, 6);
    expect(ufdnEnd.fadeInWeight).toBeCloseTo(1.0, 6);

    expect(computeLobattoDegree(1).degreeKp).toBe(0);
    expect(computeLobattoDegree(3).degreeKp).toBe(2);
    expect(computeLobattoDegree(6).degreeKp).toBe(3);

    const basin = computeSurrogateDelayBasin(0);
    expect(basin.envelopeLoss).toBe(0);
    expect(basin.oldStemDriftDb).toBe(0);

    const trustIn = computeTrustRegionDamping(1.0, 4.0);
    const trustOut = computeTrustRegionDamping(8.0, 4.0);
    expect(trustIn.isOutOfDistribution).toBe(false);
    expect(trustOut.isOutOfDistribution).toBe(true);

    const zoh96 = computeZohPoleRescaling(0.99, 1000, 96_000);
    expect(zoh96.rescaledRadius).toBeCloseTo(Math.sqrt(0.99), 6);
  });
});

describe("Tri-Physics interactive visualizer components", () => {
  it("renders PolyAdaaVisualizer andRgLruPoleVisualizer and handles slider input", async () => {
    const onDrive = vi.fn();
    const onFreq = vi.fn();
    const onBlend = vi.fn();

    const screen = await render(
      <div>
        <PolyAdaaVisualizer onDriveChange={onDrive} onFreqChange={onFreq} />
        <RgLruPoleVisualizer onBlendChange={onBlend} />
      </div>,
    );

    setRangeInputValue(
      screen.container.querySelector('input[aria-label="PolyADAA Drive"]') as HTMLInputElement,
      6.5,
    );
    setRangeInputValue(
      screen.container.querySelector('input[aria-label="PolyADAA Frequency"]') as HTMLInputElement,
      3200,
    );
    setRangeInputValue(
      screen.container.querySelector('input[aria-label="RG-LRU Blend"]') as HTMLInputElement,
      0.75,
    );

    expect(onDrive).toHaveBeenCalledWith(6.5);
    expect(onFreq).toHaveBeenCalledWith(3200);
    expect(onBlend).toHaveBeenCalledWith(0.75);
    await expectZeroAxeViolations(screen.container);
  });

  it("renders ModulatedUfdnVisualizer, ControlBusVisualizer, and KroneckerResumeVisualizer", async () => {
    const onSample = vi.fn();
    const onNodes = vi.fn();
    const onLag = vi.fn();

    const screen = await render(
      <div>
        <ModulatedUfdnVisualizer onSampleIndexChange={onSample} />
        <ControlBusVisualizer onNodesChange={onNodes} />
        <KroneckerResumeVisualizer onLagErrorChange={onLag} />
      </div>,
    );

    setRangeInputValue(
      screen.container.querySelector(
        'input[aria-label="UFDN Crossfade Sample"]',
      ) as HTMLInputElement,
      32,
    );
    setRangeInputValue(
      screen.container.querySelector(
        'input[aria-label="Lobatto Unique Nodes"]',
      ) as HTMLInputElement,
      5,
    );
    setRangeInputValue(
      screen.container.querySelector('input[aria-label="Delay Lag Error"]') as HTMLInputElement,
      8,
    );

    expect(onSample).toHaveBeenCalledWith(32);
    expect(onNodes).toHaveBeenCalledWith(5);
    expect(onLag).toHaveBeenCalledWith(8);
    await expectZeroAxeViolations(screen.container);
  });

  it("renders ClampAuditVisualizer and ZohRescalerVisualizer and switches sample rates", async () => {
    const onR2 = vi.fn();
    const onRate = vi.fn();

    const screen = await render(
      <div>
        <ClampAuditVisualizer onR2Change={onR2} r2={6.0} />
        <ClampAuditVisualizer r2={1.5} />
        <ZohRescalerVisualizer onSampleRateChange={onRate} />
      </div>,
    );

    const r2Slider = screen.container.querySelector(
      'input[aria-label="Mahalanobis R2"]',
    ) as HTMLInputElement;
    setRangeInputValue(r2Slider, 4.4);
    expect(onR2).toHaveBeenCalledWith(4.4);

    await userEvent.click(page.getByRole("button", { name: "48.0 kHz" }));
    expect(onRate).toHaveBeenCalledWith(48_000);
    await expectZeroAxeViolations(screen.container);
  });
});
