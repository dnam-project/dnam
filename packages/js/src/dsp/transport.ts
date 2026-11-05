/**
 * Op-T reference DSP kernels (`catmullRomHermite`, `catmullRomHermiteSlice`, `coerciveStep`,
 * `tankSlice`) (`ref[sdk.ts.dsp.reference-kernels]`, `ref[dsp.op-t.reader.catmull-rom]`).
 *
 * @module
 */

import { getActiveDriver } from "../internal/native/load.js";

/**
 * 4-tap Catmull-Rom cubic Hermite fractional delay interpolation (`ref[dsp.op-t.reader.catmull-rom]`).
 */
export function catmullRomHermite(
  sM1: number,
  s0: number,
  s1: number,
  s2: number,
  mu: number,
): number {
  return getActiveDriver().catmullRomHermite(sM1, s0, s1, s2, mu);
}

/**
 * Vectorized 4-tap Catmull-Rom cubic Hermite evaluation over `Float64Array` slices.
 */
export function catmullRomHermiteSlice(options: {
  readonly sM1: Float64Array;
  readonly s0: Float64Array;
  readonly s1: Float64Array;
  readonly s2: Float64Array;
  readonly mu: Float64Array;
  readonly output?: Float64Array;
}): Float64Array {
  const out = options.output ?? new Float64Array(options.s0.length);
  getActiveDriver().catmullRomHermiteSlice(
    options.sM1,
    options.s0,
    options.s1,
    options.s2,
    options.mu,
    out,
  );
  return out;
}

/**
 * Single-sample Op-T coercive-saturating Lyapunov loop step (`ref[dsp.stage4.loop-saturator]`).
 */
export function coerciveStep(options: {
  readonly uIn: number;
  readonly gLin?: number;
  readonly sigmaKnee?: number;
  readonly vCeil?: number;
  readonly rDc?: number;
  readonly dcXPrev?: number;
  readonly dcYPrev?: number;
}): {
  readonly y: number;
  readonly dcXPrev: number;
  readonly dcYPrev: number;
} {
  const [y, dcXPrev, dcYPrev] = getActiveDriver().opTCoerciveStep(
    options.uIn,
    options.gLin ?? 0.95,
    options.sigmaKnee ?? 1.2,
    options.vCeil ?? 1.5,
    options.rDc ?? 0.995,
    options.dcXPrev ?? 0.0,
    options.dcYPrev ?? 0.0,
  );
  return Object.freeze({ y, dcXPrev, dcYPrev });
}

/**
 * Vectorized Op-T coercive tank loop evaluation over a `Float64Array` slice.
 */
export function tankSlice(
  input: Float64Array,
  options?: {
    readonly gLin?: number;
    readonly sigmaKnee?: number;
    readonly vCeil?: number;
    readonly rDc?: number;
    readonly output?: Float64Array;
  },
): Float64Array {
  const out = options?.output ?? new Float64Array(input.length);
  getActiveDriver().opTTankSlice(
    input,
    out,
    options?.gLin ?? 0.95,
    options?.sigmaKnee ?? 1.2,
    options?.vCeil ?? 1.5,
    options?.rDc ?? 0.995,
  );
  return out;
}
