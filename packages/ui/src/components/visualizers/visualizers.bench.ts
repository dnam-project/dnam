import { describe, test } from "vitest";
import { describeRotaryArc } from "../controls/geometry.js";
import { dbToMeterPercent } from "../rack/TriPhysicsTelemetry.js";
import {
  computeLobattoDegree,
  computePolyAdaaMetrics,
  computeRgLruInterpolation,
  computeSurrogateDelayBasin,
  computeTrustRegionDamping,
  computeUfdnCrossfadeWeights,
  computeZohPoleRescaling,
} from "./models.js";

describe("@dnam-project/ui 60fps geometry and visualizer math benchmarks", () => {
  test("Rotary SVG geometry and Tri-Physics visualizer mathematical models", async ({ bench }) => {
    await bench.compare(
      bench("describeRotaryArc + polarToCartesian SVG dial path", () => {
        describeRotaryArc(32, 32, 24, 0.1, 0.85);
        dbToMeterPercent(-12.4);
      }),
      bench("7 Tri-Physics visualizer mathematical models", () => {
        computePolyAdaaMetrics(5.5, 2400);
        computeRgLruInterpolation(0.994, 1.35, 0.5);
        computeUfdnCrossfadeWeights(64);
        computeLobattoDegree(4);
        computeSurrogateDelayBasin(6.0);
        computeTrustRegionDamping(5.2, 4.0);
        computeZohPoleRescaling(0.992, 1200, 96_000);
      }),
      { time: 50 },
    );
  });
});
