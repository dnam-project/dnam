/**
 * Lock-free `TelemetrySnapshot` decoder from the 56-slot `Float64Array` engine buffer
 * (`ref[sdk.ts.api.resources-and-data]`, `ref[arch.fp.range-saturation]`).
 *
 * @module
 */

import { backendFromCode, tierFromIndex } from "../internal/coerce.js";
import type { ActiveBackend, Tier } from "../types.js";

/**
 * Decoded overflow or saturation bitmask flags (`overflowOrSaturationFlags`).
 */
export interface OverflowFlags {
  readonly rawBits: number;
  readonly stage1Saturated: boolean;
  readonly stage2Saturated: boolean;
  readonly stage3Saturated: boolean;
  readonly stage4Saturated: boolean;
  readonly headSaturated: boolean;
  readonly nonFiniteSanitized: boolean;
  readonly anyActive: boolean;
}

/**
 * Frozen plain telemetry & ZOH snapshot returned by `engine.telemetry()`
 * (`ref[sdk.ts.api.resources-and-data]`).
 */
export interface TelemetrySnapshot {
  readonly activeTier: Tier;
  readonly backend: ActiveBackend;
  readonly sampleRateHz: number;
  readonly stereoOut: boolean;
  readonly bypassed: boolean;
  readonly trailsEnabled: boolean;
  readonly isFrozen: boolean;
  readonly supportsFreeze: boolean;
  readonly stereoWidth: number;
  readonly currentBpm: number;
  readonly activeClampMask: number;
  readonly controlDecimationHost: number;
  readonly maxOpNDrive: number;
  readonly maxPoleRadius: number;
  readonly maxEffectiveLoopGain: number;
  readonly tailTimeMs: number;
  readonly zHead: readonly [number, number, number, number];
  readonly envStates: readonly [number, number, number, number];
  readonly lfoHarmonics: readonly [number, number, number, number, number, number, number, number];
  readonly clampedControls: readonly number[];
  readonly overflowFlags: OverflowFlags;
  readonly rhoSr: number;
  readonly rhoCtrl: number;
  readonly alphaRampHost: number;
  readonly rAcHost: number;
  readonly rDcHost: number;
  readonly tauRefSamplesHost: number;
  readonly switchXfadeSamplesHost: number;
}

/**
 * Decodes the 56-element `Float64Array` telemetry buffer into a deeply frozen `TelemetrySnapshot`.
 */
export function decodeTelemetryF64(buf: Float64Array, numControls = 16): TelemetrySnapshot {
  const tierIdx = Math.min(3, Math.max(0, Math.round(Number(buf[0]))));
  const backendCode = Math.round(Number(buf[1]));
  const rawFlags = Number(buf[48]) >>> 0;
  const overflowFlags: OverflowFlags = Object.freeze({
    rawBits: rawFlags,
    stage1Saturated: (rawFlags & 1) !== 0,
    stage2Saturated: (rawFlags & 2) !== 0,
    stage3Saturated: (rawFlags & 4) !== 0,
    stage4Saturated: (rawFlags & 8) !== 0,
    headSaturated: (rawFlags & 16) !== 0,
    nonFiniteSanitized: (rawFlags & 32) !== 0,
    anyActive: rawFlags !== 0,
  });

  const zHead = Object.freeze([
    Number(buf[16]),
    Number(buf[17]),
    Number(buf[18]),
    Number(buf[19]),
  ] as const);

  const envStates = Object.freeze([
    Number(buf[20]),
    Number(buf[21]),
    Number(buf[22]),
    Number(buf[23]),
  ] as const);

  const lfoHarmonics = Object.freeze([
    Number(buf[24]),
    Number(buf[25]),
    Number(buf[26]),
    Number(buf[27]),
    Number(buf[28]),
    Number(buf[29]),
    Number(buf[30]),
    Number(buf[31]),
  ] as const);

  const ctrlCount = Math.min(16, Math.max(0, numControls));
  const clampedControls: number[] = [];
  for (let i = 0; i < ctrlCount; i++) {
    clampedControls.push(Number(buf[32 + i]));
  }

  return Object.freeze({
    activeTier: tierFromIndex(tierIdx),
    backend: backendFromCode(backendCode),
    sampleRateHz: Math.round(Number(buf[2])),
    stereoOut: Number(buf[3]) !== 0,
    bypassed: Number(buf[4]) !== 0,
    trailsEnabled: Number(buf[5]) !== 0,
    isFrozen: Number(buf[6]) !== 0,
    supportsFreeze: Number(buf[7]) !== 0,
    stereoWidth: Number(buf[8]),
    currentBpm: Number(buf[9]),
    activeClampMask: Number(buf[10]) >>> 0,
    controlDecimationHost: Math.round(Number(buf[11])),
    maxOpNDrive: Number(buf[12]),
    maxPoleRadius: Number(buf[13]),
    maxEffectiveLoopGain: Number(buf[14]),
    tailTimeMs: Number(buf[15]),
    zHead,
    envStates,
    lfoHarmonics,
    clampedControls: Object.freeze(clampedControls),
    overflowFlags,
    rhoSr: Number(buf[49]),
    rhoCtrl: Number(buf[50]),
    alphaRampHost: Number(buf[51]),
    rAcHost: Number(buf[52]),
    rDcHost: Number(buf[53]),
    tauRefSamplesHost: Number(buf[54]),
    switchXfadeSamplesHost: Math.round(Number(buf[55])),
  });
}
