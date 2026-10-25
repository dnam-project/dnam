import { describe, expect, it, vi } from "vitest";
import { page, userEvent } from "vitest/browser";
import { render } from "vitest-browser-react";
import { expectZeroAxeViolations, setRangeInputValue } from "../../../tests/setup.js";
import {
  AccelerationBadge,
  DiscreteSwitch,
  describeRotaryArc,
  formatBackendLabel,
  normalizedToAngleDeg,
  polarToCartesian,
  RotaryKnob,
  StabilityAuditBadge,
  valueToNormalized,
} from "./index.js";

describe("rotary SVG geometry", () => {
  it("computes rotary arcs and handles degenerate ranges", () => {
    expect(valueToNormalized(5, 10, 10)).toBe(0);
    expect(valueToNormalized(5, 0, 10)).toBe(0.5);
    expect(normalizedToAngleDeg(0)).toBe(-135);
    expect(normalizedToAngleDeg(1)).toBe(135);
    const pt = polarToCartesian(32, 32, 20, 0);
    expect(pt.x).toBeCloseTo(32, 3);
    expect(describeRotaryArc(32, 32, 24, 0.5, 0.5)).toContain("M ");
    expect(describeRotaryArc(32, 32, 24, 0, 1)).toContain("A 24.00 24.00 0 1 1");
    expect(describeRotaryArc(32, 32, 24, 0.2, 0.4)).toContain("A 24.00 24.00 0 0 1");
  });
});

describe("RotaryKnob, DiscreteSwitch, and StatusBadges", () => {
  it("renders RotaryKnob with CL-12 observed bounds and fires slider updates", async () => {
    const onKnobChange = vi.fn();
    const screen = await render(
      <div>
        <RotaryKnob
          id="gain"
          label="Gain"
          value={6.0}
          min={0}
          max={10}
          observedMin={2}
          observedMax={8}
          unit="dB"
          onChange={onKnobChange}
        />
        <RotaryKnob id="tonestack" label="Tone" value={5.0} />
      </div>,
    );

    const gainSlider = screen.container.querySelector(
      'input[aria-label="Gain"]',
    ) as HTMLInputElement;
    setRangeInputValue(gainSlider, 5.5);
    expect(onKnobChange).toHaveBeenCalledWith(5.5);
    await expectZeroAxeViolations(screen.container);
  });

  it("renders DiscreteSwitch with CL-13 clamping, matches ARIA snapshot, and fires selection", async () => {
    const onSwitchChange = vi.fn();
    const screen = await render(
      <DiscreteSwitch
        id="bright"
        label="Bright Switch"
        activeIndex={0}
        onChange={onSwitchChange}
        options={[
          { index: 0, label: "Off", observed: true },
          { index: 1, label: "Warm", observed: true },
          { index: 2, label: "Scorch", observed: false },
        ]}
      />,
    );

    await expect
      .element(page.getByRole("group", { name: "Bright Switch" }))
      .toMatchAriaInlineSnapshot(`
      - group "Bright Switch":
        - button "Off" [pressed]
        - button "Warm"
        - button "Scorch" [disabled]
    `);

    await userEvent.click(page.getByRole("button", { name: "Warm" }));
    expect(onSwitchChange).toHaveBeenCalledWith(1);
    await expectZeroAxeViolations(screen.container);
  });

  it("renders AccelerationBadge and StabilityAuditBadge variants", async () => {
    expect(formatBackendLabel("arm-sme")).toContain("ARM SME");
    expect(formatBackendLabel("arm64-neon")).toContain("ARM64 NEON");
    expect(formatBackendLabel("x86-avx2-fma")).toContain("AVX2+FMA");
    expect(formatBackendLabel("wasm-simd128")).toContain("WASM SIMD128");
    expect(formatBackendLabel("scalar-reference")).toContain("Scalar Reference");

    const screen = await render(
      <div>
        <AccelerationBadge backend="arm-sme" />
        <AccelerationBadge backend="wasm-simd128" />
        <StabilityAuditBadge maxRgLruRadius={0.991} ufdnLyapunovMargin={0.04} mahalanobisR2={0.8} />
        <StabilityAuditBadge maxRgLruRadius={0.991} ufdnLyapunovMargin={0.04} mahalanobisR2={6.5} />
      </div>,
    );

    await expect.element(page.getByText(/CL-01\.\.CL-20 Verified/)).toBeVisible();
    await expect.element(page.getByText(/Trust Region Active/)).toBeVisible();
    await expectZeroAxeViolations(screen.container);
  });
});
