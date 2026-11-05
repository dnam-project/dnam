/**
 * Control-Bus reference DSP kernels (`lobattoBasis`, `lobattoBasisSlice`, `rank4Fm`,
 * `rank4FmSlice`, `chebyshevLfo`, `chebyshevLfoSlice`, `trustRegionSoftclip`,
 * `trustRegionSoftclipSlice`) (`ref[sdk.ts.dsp.reference-kernels]`,
 * `ref[dsp.ctrl.basis.lobatto]`, `ref[dsp.ctrl.op-n.trust-clip]`).
 *
 * @module
 */

import { getActiveDriver } from "../internal/native/load.js";

const DEFAULT_LOBATTO_ANCHORS = new Float64Array([0.0, -1.0, 1.0]);

/**
 * Node-Anchored Newton/Lobatto basis evaluation (`ref[dsp.ctrl.basis.lobatto]`).
 */
export function lobattoBasis(
  u: number,
  options?: {
    readonly anchors?: readonly [number, number, number] | Float64Array;
    readonly degree?: number;
  },
): readonly [number, number, number] {
  const anc =
    options?.anchors === undefined
      ? DEFAULT_LOBATTO_ANCHORS
      : options.anchors instanceof Float64Array
        ? options.anchors
        : new Float64Array(options.anchors);
  return Object.freeze(getActiveDriver().evalLobattoBasis(u, anc, options?.degree ?? 3));
}

/**
 * Vectorized Node-Anchored Newton/Lobatto basis evaluation over a `Float64Array` slice into `3 * N`.
 */
export function lobattoBasisSlice(
  u: Float64Array,
  options?: {
    readonly anchors?: readonly [number, number, number] | Float64Array;
    readonly degree?: number;
    readonly output?: Float64Array;
  },
): Float64Array {
  const anc =
    options?.anchors === undefined
      ? DEFAULT_LOBATTO_ANCHORS
      : options.anchors instanceof Float64Array
        ? options.anchors
        : new Float64Array(options.anchors);
  const out = options?.output ?? new Float64Array(u.length * 3);
  getActiveDriver().evalLobattoBasisSlice(u, out, anc, options?.degree ?? 3);
  return out;
}

/**
 * Rank-4 Factorization Machine interaction evaluation (`ref[dsp.ctrl.accum.anova-fm]`).
 */
export function rank4Fm(
  features: Float64Array | readonly number[],
  vFactorsFlat: Float64Array | readonly number[],
): readonly [number, number, number, number] {
  const feat = features instanceof Float64Array ? features : new Float64Array(features);
  const vf = vFactorsFlat instanceof Float64Array ? vFactorsFlat : new Float64Array(vFactorsFlat);
  return Object.freeze(getActiveDriver().evalRank4Fm(feat, vf));
}

/**
 * Vectorized Rank-4 Factorization Machine evaluation over `nFrames` frames of `dDim` features.
 */
export function rank4FmSlice(options: {
  readonly featuresFlat: Float64Array;
  readonly nFrames: number;
  readonly dDim: number;
  readonly vFactorsFlat: Float64Array;
  readonly output?: Float64Array;
}): Float64Array {
  const out = options.output ?? new Float64Array(options.nFrames * 4);
  getActiveDriver().evalRank4FmSlice(
    options.featuresFlat,
    options.nFrames,
    options.dDim,
    options.vFactorsFlat,
    out,
  );
  return out;
}

/**
 * 4-Harmonic Chebyshev LFO polynomial evaluation (`ref[dsp.op-n.ctrl.chebyshev-lfo]`).
 */
export function chebyshevLfo(cosPhi: number): readonly [number, number, number, number] {
  return Object.freeze(getActiveDriver().evalChebyshevLfo(cosPhi));
}

/**
 * Vectorized 4-Harmonic Chebyshev LFO evaluation over a `Float64Array` slice into `4 * N`.
 */
export function chebyshevLfoSlice(
  cosPhi: Float64Array,
  options?: { readonly output?: Float64Array },
): Float64Array {
  const out = options?.output ?? new Float64Array(cosPhi.length * 4);
  getActiveDriver().evalChebyshevLfoSlice(cosPhi, out);
  return out;
}

/**
 * Cubic trust-region soft-clipper (`ref[dsp.ctrl.op-n.trust-clip]`).
 */
export function trustRegionSoftclip(z: number, mh = 2.5): number {
  return getActiveDriver().evalTrustRegionSoftclip(z, mh);
}

/**
 * Vectorized cubic trust-region soft-clipper over a `Float64Array` slice.
 */
export function trustRegionSoftclipSlice(
  z: Float64Array,
  options?: { readonly mh?: number; readonly output?: Float64Array },
): Float64Array {
  const out = options?.output ?? new Float64Array(z.length);
  getActiveDriver().evalTrustRegionSoftclipSlice(z, out, options?.mh ?? 2.5);
  return out;
}
