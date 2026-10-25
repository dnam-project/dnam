import { describe, expect, it } from "vitest";
import { clampNumber, cn, formatDecibels, formatParameterValue } from "./utils.js";

describe("UI formatting and class utilities", () => {
  it("merges Tailwind classes via clsx and tailwind-merge", () => {
    expect(cn("px-2", "px-4", false && "hidden")).toBe("px-4");
  });

  it("clamps numeric values and guards against NaN", () => {
    expect(clampNumber(Number.NaN, 1, 5)).toBe(1);
    expect(clampNumber(-3, 0, 10)).toBe(0);
    expect(clampNumber(15, 0, 10)).toBe(10);
    expect(clampNumber(4, 0, 10)).toBe(4);
  });

  it("formats parameter values and signed decibels", () => {
    expect(formatParameterValue(3.456, 2, "dB")).toBe("3.46 dB");
    expect(formatParameterValue(3.456, 1)).toBe("3.5");
    expect(formatDecibels(2.5, 1)).toBe("+2.5 dB");
    expect(formatDecibels(-12.34, 1)).toBe("-12.3 dB");
  });
});
