/**
 * Interactive `.dnam-prev` (`DNPR`) preview single-control window & switch option validators
 * (`ref[sdk.ts.provenance.v2-security]`, `ref[sec.preview.containment]`).
 *
 * @module
 */

import { getActiveDriver } from "../internal/native/load.js";

/**
 * Options for `validatePreviewSwitchOptions(options)`.
 */
export interface PreviewSwitchSpec {
  readonly preset: string;
  readonly alternative: string;
  readonly observed: readonly string[];
  readonly active: readonly string[];
  readonly safe: readonly string[];
}

/**
 * Computes the bounded single-control preview window `[start, end]` around `anchor`
 * (`ref[sec.preview.containment]`).
 */
export function computePreviewWindow(options: {
  readonly anchor: number;
  readonly delta: number;
  readonly safeBounds: readonly [number, number];
}): readonly [number, number] {
  const [start, end] = getActiveDriver().computePreviewWindow(
    options.anchor,
    options.delta,
    options.safeBounds[0],
    options.safeBounds[1],
  );
  return Object.freeze([start, end] as const);
}

/**
 * Validates that a single-switch preview exposes two distinct options that are observed, active,
 * and safe (`ref[sec.preview.containment]`).
 */
export function validatePreviewSwitchOptions(spec: PreviewSwitchSpec): readonly [string, string] {
  const [p, a] = getActiveDriver().validatePreviewSwitchOptions(
    spec.preset,
    spec.alternative,
    spec.observed,
    spec.active,
    spec.safe,
  );
  return Object.freeze([p, a] as const);
}
