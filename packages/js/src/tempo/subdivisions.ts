/**
 * Musical tempo subdivision types, ZOH rescaling parameters, and calibration inversion results
 * (`ref[sdk.ts.model.containers-and-checkpoints]`, `ref[dsp.tempo.physical-inversion]`).
 *
 * @module
 */

export { type Subdivision, subdivisions } from "../types.js";

/**
 * Continuous-Time Zero-Order Hold (ZOH) sample-rate rescaled parameters (`ref[dsp.zoh.ctrl-stride]`).
 */
export interface ZohParams {
  readonly sampleRateHostHz: number;
  readonly rhoSr: number;
  readonly controlDecimationHost: number;
  readonly rhoCtrl: number;
  readonly alphaRampHost: number;
  readonly rAcHost: number;
  readonly rDcHost: number;
  readonly tauRefSamplesHost: number;
  readonly switchXfadeSamplesHost: number;
}

/**
 * Result of `subdivisionPhase(...)` (`Q0.32` phase and unit phasor `[cosPhi, sinPhi]`).
 */
export interface SubdivisionPhaseResult {
  readonly phaseQ32: number;
  readonly cosPhi: number;
  readonly sinPhi: number;
}

/**
 * Result of 32-step binary-search physical tempo calibration inversion (`invertTempoCalibration`).
 */
export interface TempoCalibrationResult {
  readonly physicalF32: number;
  readonly normalizedF32: number;
  readonly realizedValue: number;
  readonly residual: number;
  readonly searchSteps: number;
}
