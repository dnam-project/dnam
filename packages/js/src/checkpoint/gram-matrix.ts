/**
 * Column-major upper-triangular (`LAPACK uplo='U'`) symmetric Gram matrix packing and unpacking
 * (`ref[sdk.ts.model.containers-and-checkpoints]`, `ref[container.ckpt.uplo-u]`).
 *
 * @module
 */

import { DnamError } from "../errors.js";
import { getActiveDriver } from "../internal/native/load.js";

/**
 * Returns the packed element count $N(N+1)/2$ for an $N \times N$ symmetric `uplo='U'` matrix.
 */
export function uploUPackedLength(dimension: number): number {
  if (!Number.isInteger(dimension) || dimension < 0) {
    throw new DnamError("ERR_DNAM_VALIDATION", `Invalid matrix dimension: ${String(dimension)}`);
  }
  return getActiveDriver().uploUPackedLen(dimension);
}

/**
 * Packed `LAPACK uplo='U'` column-major upper-triangular `Float64Array` Gram matrix
 * (`ref[container.ckpt.uplo-u]`).
 *
 * @example
 * ```ts
 * const gram = GramMatrix.fromDense(new Float64Array([1, 2, 2, 4]), 2);
 * const dense = gram.toDense();
 * ```
 */
export class GramMatrix {
  readonly dimension: number;
  readonly #packed: Float64Array;

  constructor(packed: Float64Array, dimension: number) {
    if (!Number.isInteger(dimension) || dimension <= 0) {
      throw new DnamError(
        "ERR_DNAM_VALIDATION",
        `GramMatrix dimension must be a positive integer, got ${String(dimension)}`,
      );
    }
    const expectedLen = uploUPackedLength(dimension);
    if (packed.length < expectedLen) {
      throw new DnamError(
        "ERR_DNAM_VALIDATION",
        `Packed buffer too short for dimension=${String(dimension)}: need ${String(expectedLen)}, got ${String(packed.length)}`,
      );
    }
    this.dimension = dimension;
    this.#packed = packed.slice(0, expectedLen);
  }

  /**
   * Packs a flattened row-major $N \times N$ `Float64Array` into a `uplo='U'` `GramMatrix`.
   */
  static fromDense(
    denseFlat: Float64Array,
    dimension: number,
    options?: { readonly output?: Float64Array },
  ): GramMatrix {
    if (!Number.isInteger(dimension) || dimension <= 0) {
      throw new DnamError(
        "ERR_DNAM_VALIDATION",
        `dimension must be a positive integer, got ${String(dimension)}`,
      );
    }
    const packedLen = uploUPackedLength(dimension);
    const out = options?.output ?? new Float64Array(packedLen);
    getActiveDriver().packUploUF64Into(denseFlat, dimension, out);
    return new GramMatrix(out, dimension);
  }

  /**
   * Number of packed upper-triangular `f64` elements ($N(N+1)/2$).
   */
  get packedLength(): number {
    return this.#packed.length;
  }

  /**
   * Returns a copy of the packed `uplo='U'` `Float64Array`.
   */
  get packed(): Float64Array {
    return this.#packed.slice();
  }

  /**
   * Unpacks the `uplo='U'` matrix into a flattened $N_{\text{tier}} \times N_{\text{tier}}$
   * `Float64Array` (`tierDimension <= this.dimension`).
   */
  toDense(options?: {
    readonly tierDimension?: number;
    readonly output?: Float64Array;
  }): Float64Array {
    const nTier = options?.tierDimension ?? this.dimension;
    if (!Number.isInteger(nTier) || nTier <= 0 || nTier > this.dimension) {
      throw new DnamError(
        "ERR_DNAM_VALIDATION",
        `tierDimension must be in 1..${String(this.dimension)}, got ${String(nTier)}`,
      );
    }
    const out = options?.output ?? new Float64Array(nTier * nTier);
    getActiveDriver().unpackUploUF64Into(this.#packed, nTier, out);
    return out;
  }
}
