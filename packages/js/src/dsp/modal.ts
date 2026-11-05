/**
 * Op-M reference DSP kernels (`modalStep`, `modalSlice`, `unitRotor`, `unitRotorSlice`)
 * (`ref[sdk.ts.dsp.reference-kernels]`, `ref[dsp.op-m.recurrence.symplectic]`,
 * `ref[dsp.op-m.rotor.contractive]`).
 *
 * @module
 */

import { getActiveDriver } from "../internal/native/load.js";

/**
 * Parameters for `modalStep` and `modalSlice`.
 */
export interface ModalParams {
  readonly rLow?: number;
  readonly drSpan?: number;
  readonly wSel?: number;
  readonly bGammaNorm?: number;
  readonly cReal?: number;
  readonly cImag?: number;
  readonly uRotCos?: number;
  readonly uRotSin?: number;
  readonly zRe?: number;
  readonly zIm?: number;
}

/**
 * Single-sample Radial-Polar `Op-M` symplectic modal step (`ref[dsp.op-m.recurrence.symplectic]`).
 */
export function modalStep(
  x: number,
  params?: ModalParams,
): {
  readonly y: number;
  readonly zRe: number;
  readonly zIm: number;
} {
  const [y, zRe, zIm] = getActiveDriver().opMModalStep(
    x,
    params?.rLow ?? 0.9,
    params?.drSpan ?? 0.08,
    params?.wSel ?? 0.0,
    params?.bGammaNorm ?? 0.25,
    params?.cReal ?? 1.0,
    params?.cImag ?? 0.0,
    params?.uRotCos ?? 0.98,
    params?.uRotSin ?? 0.198997,
    params?.zRe ?? 0.0,
    params?.zIm ?? 0.0,
  );
  return Object.freeze({ y, zRe, zIm });
}

/**
 * Vectorized `Op-M` modal resonator evaluation over a `Float64Array` slice.
 */
export function modalSlice(
  input: Float64Array,
  options?: ModalParams & {
    readonly reset?: boolean;
    readonly output?: Float64Array;
  },
): {
  readonly output: Float64Array;
  readonly zRe: number;
  readonly zIm: number;
} {
  const output = options?.output ?? new Float64Array(input.length);
  const [zRe, zIm] = getActiveDriver().opMModalSlice(
    input,
    output,
    options?.rLow ?? 0.9,
    options?.drSpan ?? 0.08,
    options?.wSel ?? 0.0,
    options?.bGammaNorm ?? 0.25,
    options?.cReal ?? 1.0,
    options?.cImag ?? 0.0,
    options?.uRotCos ?? 0.98,
    options?.uRotSin ?? 0.198997,
    options?.zRe ?? 0.0,
    options?.zIm ?? 0.0,
    options?.reset ?? true,
  );
  return Object.freeze({ output, zRe, zIm });
}

/**
 * Quadrant-reduced contractive unit-circle rotor returning `[cosTheta, sinTheta]`
 * (`ref[dsp.op-m.rotor.contractive]`).
 */
export function unitRotor(phi: number): readonly [number, number] {
  return Object.freeze(getActiveDriver().unitRotor(phi));
}

/**
 * Vectorized unit-circle rotor evaluation over a `Float64Array` slice into a `2 * N` buffer
 * (`cos` row followed by `sin` row).
 */
export function unitRotorSlice(
  phi: Float64Array,
  options?: { readonly output?: Float64Array },
): Float64Array {
  const out = options?.output ?? new Float64Array(phi.length * 2);
  getActiveDriver().unitRotorSlice(phi, out);
  return out;
}
