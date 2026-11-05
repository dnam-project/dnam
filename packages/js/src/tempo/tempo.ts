/**
 * Live `TempoController` (`engine.tempo`, `node.tempo`) and offline ZOH / tempo calibration
 * utilities (`ref[sdk.ts.model.containers-and-checkpoints]`, `ref[dsp.tempo.physical-inversion]`).
 *
 * @module
 */

import { coerceSubdivision } from "../internal/coerce.js";
import { getActiveDriver } from "../internal/native/load.js";
import type { Subdivision } from "../types.js";
import type { SubdivisionPhaseResult, TempoCalibrationResult, ZohParams } from "./subdivisions.js";

/**
 * Options for `tempo.set(options)` / `tempo.sync(options)`.
 */
export interface TempoSyncOptions {
  readonly bpm: number;
  readonly subdivision?: Subdivision;
  readonly foldOctave?: boolean;
}

/**
 * Options for `tempo.tap(options)`.
 */
export interface TempoTapOptions {
  readonly subdivision?: Subdivision;
  readonly foldOctave?: boolean;
}

/**
 * Result returned by `tempo.tap(options)`.
 */
export interface TapTempoResult {
  readonly bpm: number;
  readonly updatedParameters: number;
}

/**
 * Internal backend interface for `TempoController`.
 */
export interface TempoBackend {
  currentBpm(): number;
  syncTempo(bpm: number, subdivision: Subdivision | undefined, foldOctave: boolean): number;
  registerTapTempo(subdivision: Subdivision | undefined, foldOctave: boolean): TapTempoResult;
}

/**
 * Live tempo synchronization controller exposed as `engine.tempo` and `node.tempo`
 * (`REQ-2.5`, `ref[dsp.tempo.physical-inversion]`).
 *
 * @example
 * ```ts
 * engine.tempo.set({ bpm: 120, subdivision: "1/8." });
 * engine.tempo.tap({ subdivision: "1/4" });
 * ```
 */
export class TempoController {
  readonly #backend: TempoBackend;

  constructor(backend: TempoBackend) {
    this.#backend = backend;
  }

  /**
   * Current active host/tap tempo in beats per minute (`BPM`).
   */
  get bpm(): number {
    return this.#backend.currentBpm();
  }

  /**
   * Synchronizes all tempo-capable parameters (`delay_time_ms`, `grain_size_ms`, `lfo_rate_hz`)
   * to `options.bpm` and optional `options.subdivision`, returning the number of updated parameters.
   */
  set(options: TempoSyncOptions): number {
    const sub =
      options.subdivision !== undefined ? coerceSubdivision(options.subdivision) : undefined;
    return this.#backend.syncTempo(options.bpm, sub, options.foldOctave ?? false);
  }

  /**
   * Alias for `tempo.set(options)`.
   */
  sync(options: TempoSyncOptions): number {
    return this.set(options);
  }

  /**
   * Registers a live tap-tempo timestamp in the engine's outlier-rejecting tap tracker and
   * updates tempo-synced parameters.
   */
  tap(options?: TempoTapOptions): TapTempoResult {
    const sub =
      options?.subdivision !== undefined ? coerceSubdivision(options.subdivision) : undefined;
    return this.#backend.registerTapTempo(sub, options?.foldOctave ?? false);
  }
}

/**
 * Computes Continuous-Time Zero-Order Hold (ZOH) rescaled parameters (`ref[dsp.zoh.ctrl-stride]`).
 */
export function computeZoh(options: {
  readonly sampleRateTrainHz: number;
  readonly sampleRateHostHz: number;
  readonly rAcTrain?: number;
  readonly rDcTrain?: number;
  readonly tauRefTrain?: number;
  readonly switchXfadeTrain?: number;
}): ZohParams {
  const arr = getActiveDriver().computeZoh(
    options.sampleRateTrainHz,
    options.sampleRateHostHz,
    options.rAcTrain ?? 0.9993,
    options.rDcTrain ?? 0.995,
    options.tauRefTrain ?? 64.0,
    options.switchXfadeTrain ?? 128,
  );
  return Object.freeze({
    sampleRateHostHz: Number(arr[0]),
    rhoSr: Number(arr[1]),
    controlDecimationHost: Number(arr[2]),
    rhoCtrl: Number(arr[3]),
    alphaRampHost: Number(arr[4]),
    rAcHost: Number(arr[5]),
    rDcHost: Number(arr[6]),
    tauRefSamplesHost: Number(arr[7]),
    switchXfadeSamplesHost: Number(arr[8]),
  });
}

/**
 * Returns the control-bus decimation stride $K_{\text{ctrl}}(f_s) = \max(8, \lfloor(f_s + 1500)/3000\rfloor)$
 * (`ref[dsp.zoh.ctrl-stride]`).
 */
export function controlStride(sampleRateHz: number): number {
  return getActiveDriver().controlStrideForRate(sampleRateHz);
}

/**
 * Returns the control-tick ratio $\rho_{\text{ctrl}}(f_s)$.
 */
export function controlTickRatio(sampleRateHz: number): number {
  return getActiveDriver().controlTickRatio(sampleRateHz);
}

/**
 * Evaluates the exact `Q0.32` musical subdivision phase and unit phasor `[cosPhi, sinPhi]`
 * (`ref[dsp.tempo.physical-inversion]`).
 */
export function subdivisionPhase(options: {
  readonly beatMod12: number;
  readonly withinBeatQ32: number;
  readonly subdivision: Subdivision | string;
  readonly offsetQ32?: number;
}): SubdivisionPhaseResult {
  const sub = coerceSubdivision(options.subdivision);
  const [phaseQ32, cosPhi, sinPhi] = getActiveDriver().evalSubdivisionPhase(
    options.beatMod12,
    options.withinBeatQ32 >>> 0,
    sub,
    (options.offsetQ32 ?? 0) >>> 0,
  );
  return Object.freeze({
    phaseQ32: phaseQ32 >>> 0,
    cosPhi,
    sinPhi,
  });
}

/**
 * Evaluates a 3-node physical calibration curve $v(u)$ at $u \in [-1, 1]$.
 */
export function evalCalibration3Node(
  anchors: readonly [number, number, number] | Float64Array,
  u: number,
): number {
  const f64 = anchors instanceof Float64Array ? anchors : new Float64Array(anchors);
  return getActiveDriver().eval3NodeCalibration(f64, u);
}

/**
 * Inverts a monotone 3-node physical calibration curve using the 32-step `f32` rank binary search
 * (`ref[dsp.tempo.physical-inversion]`).
 */
export function invertTempoCalibration(options: {
  readonly declaredRange: readonly [number, number];
  readonly safeRange: readonly [number, number];
  readonly anchors: readonly [number, number, number] | Float64Array;
  readonly target: number;
  readonly flushSubnormals?: boolean;
}): TempoCalibrationResult {
  const f64 =
    options.anchors instanceof Float64Array ? options.anchors : new Float64Array(options.anchors);
  const res = getActiveDriver().invertTempoCalibration(
    options.declaredRange[0],
    options.declaredRange[1],
    options.safeRange[0],
    options.safeRange[1],
    f64,
    options.target,
    options.flushSubnormals ?? true,
  );
  return Object.freeze({
    physicalF32: Number(res[0]),
    normalizedF32: Number(res[1]),
    realizedValue: Number(res[2]),
    residual: Number(res[3]),
    searchSteps: Number(res[4]),
  });
}
