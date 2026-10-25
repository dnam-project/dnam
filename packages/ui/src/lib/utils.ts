import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Merges Tailwind CSS v4 utility classes deterministically using `clsx` + `tailwind-merge`.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/**
 * Clamps a numeric value to `[min, max]`.
 */
export function clampNumber(value: number, min: number, max: number): number {
  if (Number.isNaN(value)) {
    return min;
  }
  return Math.min(max, Math.max(min, value));
}

/**
 * Formats a parameter value with fixed precision and optional engineering unit.
 */
export function formatParameterValue(
  value: number,
  precision = 2,
  unit?: string | undefined,
): string {
  const digits = clampNumber(Math.trunc(precision), 0, 6);
  const formatted = value.toFixed(digits);
  return unit && unit.length > 0 ? `${formatted} ${unit}` : formatted;
}

/**
 * Formats a decibel value with explicit sign (`+3.2 dB` or `-14.8 dB`).
 */
export function formatDecibels(db: number, precision = 1): string {
  const digits = clampNumber(Math.trunc(precision), 0, 4);
  const prefix = db > 0 ? "+" : "";
  return `${prefix}${db.toFixed(digits)} dB`;
}
