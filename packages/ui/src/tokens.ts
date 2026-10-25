/**
 * Unified Design Tokens for DNAM Desktop Studio (`packages/studio`)
 * and DNAM Web Studio / Astro Starlight (`packages/web`).
 */

export interface DnamThemeTokens {
  readonly colors: {
    readonly surface950: string;
    readonly surface900: string;
    readonly surface800: string;
    readonly surface700: string;
    readonly border: string;
    readonly core1: string;
    readonly core2: string;
    readonly control: string;
    readonly safe: string;
    readonly warn: string;
    readonly danger: string;
  };
  readonly fonts: {
    readonly sans: string;
    readonly mono: string;
  };
  readonly radii: {
    readonly sm: string;
    readonly md: string;
    readonly lg: string;
  };
}

export const DNAM_THEME_TOKENS: DnamThemeTokens = {
  colors: {
    surface950: "#090a0f",
    surface900: "#11131c",
    surface800: "#1a1d2a",
    surface700: "#252a3d",
    border: "#2c3248",
    core1: "#f59e0b",
    core2: "#06b6d4",
    control: "#8b5cf6",
    safe: "#10b981",
    warn: "#f59e0b",
    danger: "#ef4444",
  },
  fonts: {
    sans: '"Inter", "SF Pro Display", system-ui, -apple-system, sans-serif',
    mono: '"JetBrains Mono", "SF Mono", ui-monospace, monospace',
  },
  radii: {
    sm: "0.375rem",
    md: "0.5rem",
    lg: "0.75rem",
  },
} as const;

export type TriPhysicsDomain = "core1" | "core2" | "control";

export function domainAccentHex(domain: TriPhysicsDomain): string {
  switch (domain) {
    case "core1":
      return DNAM_THEME_TOKENS.colors.core1;
    case "core2":
      return DNAM_THEME_TOKENS.colors.core2;
    default:
      return DNAM_THEME_TOKENS.colors.control;
  }
}
