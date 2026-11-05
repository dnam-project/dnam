/**
 * Op-N division-free algebraic `PolyADAA` reference DSP kernels (`polyAdaaAlgebraic`,
 * `polyAdaaStep`, `polyAdaaSlice`) (`ref[sdk.ts.dsp.reference-kernels]`,
 * `ref[dsp.op-n.polyadaa.midpoint]`).
 *
 * @module
 */

import { getActiveDriver } from "../internal/native/load.js";

/**
 * Parameters for `polyAdaaStep` and `polyAdaaSlice`.
 */
export interface PolyAdaaParams {
  readonly bConv?: number;
  readonly alpha?: number;
  readonly beta?: number;
  readonly delta?: number;
  readonly rAc?: number;
  readonly uPrev?: number;
  readonly aPrev?: number;
  readonly mDc?: number;
}

/**
 * Division-free algebraic `PolyADAA` segment average (`ref[dsp.op-n.polyadaa.midpoint]`).
 */
export function polyAdaaAlgebraic(uPrev: number, uCurr: number, beta = 1.0, delta = 0.0): number {
  return getActiveDriver().polyAdaaAlgebraic(uPrev, uCurr, beta, delta);
}

/**
 * Single-sample `PolyADAA` step returning `{ y, uPrev, aPrev, mDc }`.
 */
export function polyAdaaStep(
  x: number,
  params?: PolyAdaaParams,
): {
  readonly y: number;
  readonly uPrev: number;
  readonly aPrev: number;
  readonly mDc: number;
} {
  const [y, uPrev, aPrev, mDc] = getActiveDriver().polyAdaaStep(
    x,
    params?.bConv ?? 0.0,
    params?.alpha ?? 0.5,
    params?.beta ?? 0.5,
    params?.delta ?? 0.0,
    params?.rAc ?? 0.9993,
    params?.uPrev ?? 0.0,
    params?.aPrev ?? 0.0,
    params?.mDc ?? 0.0,
  );
  return Object.freeze({ y, uPrev, aPrev, mDc });
}

/**
 * Vectorized `PolyADAA` evaluation over a `Float64Array` slice.
 */
export function polyAdaaSlice(
  input: Float64Array,
  options?: PolyAdaaParams & {
    readonly reset?: boolean;
    readonly output?: Float64Array;
  },
): {
  readonly output: Float64Array;
  readonly uPrev: number;
  readonly aPrev: number;
  readonly mDc: number;
} {
  const output = options?.output ?? new Float64Array(input.length);
  const [uPrev, aPrev, mDc] = getActiveDriver().polyAdaaSlice(
    input,
    output,
    options?.bConv ?? 0.0,
    options?.alpha ?? 0.5,
    options?.beta ?? 0.5,
    options?.delta ?? 0.0,
    options?.rAc ?? 0.9993,
    options?.uPrev ?? 0.0,
    options?.aPrev ?? 0.0,
    options?.mDc ?? 0.0,
    options?.reset ?? true,
  );
  return Object.freeze({ output, uPrev, aPrev, mDc });
}
