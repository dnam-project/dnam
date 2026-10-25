import { clampNumber } from "../../lib/utils.js";

export const ROTARY_START_ANGLE_DEG = -135;
export const ROTARY_END_ANGLE_DEG = 135;
export const ROTARY_SWEEP_DEG = ROTARY_END_ANGLE_DEG - ROTARY_START_ANGLE_DEG;

export function valueToNormalized(value: number, min: number, max: number): number {
  const span = max - min;
  if (span <= 0) {
    return 0;
  }
  return clampNumber((value - min) / span, 0, 1);
}

export function normalizedToAngleDeg(normalized: number): number {
  const u = clampNumber(normalized, 0, 1);
  return ROTARY_START_ANGLE_DEG + u * ROTARY_SWEEP_DEG;
}

export function polarToCartesian(
  cx: number,
  cy: number,
  radius: number,
  angleDeg: number,
): { readonly x: number; readonly y: number } {
  const angleRad = ((angleDeg - 90) * Math.PI) / 180.0;
  return {
    x: cx + radius * Math.cos(angleRad),
    y: cy + radius * Math.sin(angleRad),
  };
}

export function describeRotaryArc(
  cx: number,
  cy: number,
  radius: number,
  startNorm: number,
  endNorm: number,
): string {
  const u0 = clampNumber(startNorm, 0, 1);
  const u1 = clampNumber(endNorm, 0, 1);
  if (Math.abs(u1 - u0) < 1e-6) {
    const pt = polarToCartesian(cx, cy, radius, normalizedToAngleDeg(u0));
    return `M ${pt.x.toFixed(2)} ${pt.y.toFixed(2)}`;
  }
  const a0 = normalizedToAngleDeg(Math.min(u0, u1));
  const a1 = normalizedToAngleDeg(Math.max(u0, u1));
  const start = polarToCartesian(cx, cy, radius, a0);
  const end = polarToCartesian(cx, cy, radius, a1);
  const largeArcFlag = a1 - a0 > 180 ? 1 : 0;
  return `M ${start.x.toFixed(2)} ${start.y.toFixed(2)} A ${radius.toFixed(2)} ${radius.toFixed(2)} 0 ${largeArcFlag} 1 ${end.x.toFixed(2)} ${end.y.toFixed(2)}`;
}
