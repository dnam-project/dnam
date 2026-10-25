import { describe, expect, it } from "vitest";
import { DNAM_THEME_TOKENS, domainAccentHex } from "./tokens.js";

describe("DNAM_THEME_TOKENS and domainAccentHex", () => {
  it("resolves surface palette and Tri-Physics domain accent hex codes", () => {
    expect(DNAM_THEME_TOKENS.colors.surface950).toBe("#090a0f");
    expect(domainAccentHex("core1")).toBe(DNAM_THEME_TOKENS.colors.core1);
    expect(domainAccentHex("core2")).toBe(DNAM_THEME_TOKENS.colors.core2);
    expect(domainAccentHex("control")).toBe(DNAM_THEME_TOKENS.colors.control);
  });
});
