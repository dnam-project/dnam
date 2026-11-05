/**
 * Mixing and interpolation reference DSP kernels (`cubicLagrangeWeights`,
 * `cubicLagrangeWeightsSlice`, `givensHouseholder8ch`, `givensHouseholder8chSlice`,
 * `givensHouseholder16ch`, `givensHouseholder16chSlice`) (`ref[sdk.ts.dsp.reference-kernels]`,
 * `ref[dsp.stage4.spatial-tank]`).
 *
 * @module
 */

import { getActiveDriver } from "../internal/native/load.js";

/**
 * 4-tap Cubic Lagrange fractional delay weights (`ref[dsp.op-t.reader.catmull-rom]`).
 */
export function cubicLagrangeWeights(mu: number): readonly [number, number, number, number] {
  return Object.freeze(getActiveDriver().cubicLagrangeWeights(mu));
}

/**
 * Vectorized 4-tap Cubic Lagrange fractional delay weights over a `Float64Array` slice into `4 * N`.
 */
export function cubicLagrangeWeightsSlice(
  mu: Float64Array,
  options?: { readonly output?: Float64Array },
): Float64Array {
  const out = options?.output ?? new Float64Array(mu.length * 4);
  getActiveDriver().cubicLagrangeWeightsSlice(mu, out);
  return out;
}

/**
 * 8-channel Orthogonal Givens + Householder mixing matrix (`ref[dsp.stage4.spatial-tank]`).
 */
export function givensHouseholder8ch(
  x8: Float64Array | readonly number[],
  cosPsi = 1.0,
  sinPsi = 0.0,
): Float64Array {
  const arr = x8 instanceof Float64Array ? x8 : new Float64Array(x8);
  return getActiveDriver().givensHouseholder8ch(arr, cosPsi, sinPsi);
}

/**
 * Vectorized 8-channel Givens + Householder mixing over a `Float64Array` slice (`8 * N`).
 */
export function givensHouseholder8chSlice(
  xFlat: Float64Array,
  options?: {
    readonly cosPsi?: number;
    readonly sinPsi?: number;
    readonly output?: Float64Array;
  },
): Float64Array {
  const out = options?.output ?? new Float64Array(xFlat.length);
  getActiveDriver().givensHouseholder8chSlice(
    xFlat,
    out,
    options?.cosPsi ?? 1.0,
    options?.sinPsi ?? 0.0,
  );
  return out;
}

const DEFAULT_CROSS_COS_8 = new Float64Array(8).fill(1.0);
const DEFAULT_CROSS_SIN_8 = new Float64Array(8).fill(0.0);

/**
 * 16-channel Orthogonal Givens + Householder mixing matrix (`ref[dsp.stage4.spatial-tank]`).
 */
export function givensHouseholder16ch(
  x16: Float64Array | readonly number[],
  options?: {
    readonly cosPsi?: number;
    readonly sinPsi?: number;
    readonly crossCos8?: Float64Array | readonly number[];
    readonly crossSin8?: Float64Array | readonly number[];
  },
): Float64Array {
  const arr = x16 instanceof Float64Array ? x16 : new Float64Array(x16);
  const cc =
    options?.crossCos8 === undefined
      ? DEFAULT_CROSS_COS_8
      : options.crossCos8 instanceof Float64Array
        ? options.crossCos8
        : new Float64Array(options.crossCos8);
  const cs =
    options?.crossSin8 === undefined
      ? DEFAULT_CROSS_SIN_8
      : options.crossSin8 instanceof Float64Array
        ? options.crossSin8
        : new Float64Array(options.crossSin8);
  return getActiveDriver().givensHouseholder16ch(
    arr,
    options?.cosPsi ?? 1.0,
    options?.sinPsi ?? 0.0,
    cc,
    cs,
  );
}

/**
 * Vectorized 16-channel Givens + Householder mixing over a `Float64Array` slice (`16 * N`).
 */
export function givensHouseholder16chSlice(
  xFlat: Float64Array,
  options?: {
    readonly cosPsi?: number;
    readonly sinPsi?: number;
    readonly crossCos8?: Float64Array | readonly number[];
    readonly crossSin8?: Float64Array | readonly number[];
    readonly output?: Float64Array;
  },
): Float64Array {
  const out = options?.output ?? new Float64Array(xFlat.length);
  const cc =
    options?.crossCos8 === undefined
      ? DEFAULT_CROSS_COS_8
      : options.crossCos8 instanceof Float64Array
        ? options.crossCos8
        : new Float64Array(options.crossCos8);
  const cs =
    options?.crossSin8 === undefined
      ? DEFAULT_CROSS_SIN_8
      : options.crossSin8 instanceof Float64Array
        ? options.crossSin8
        : new Float64Array(options.crossSin8);
  getActiveDriver().givensHouseholder16chSlice(
    xFlat,
    out,
    options?.cosPsi ?? 1.0,
    options?.sinPsi ?? 0.0,
    cc,
    cs,
  );
  return out;
}
