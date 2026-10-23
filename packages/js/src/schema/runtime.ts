/**
 * Runtime telemetry, preamble, ZOH, 3-Phase clamp result, and DSP parameter schemas
 * using idiomatic TypeScript `camelCase`.
 */

import { z } from "zod";
import type { ControlValue } from "./base.js";

export interface PreambleSnapshot {
  magic: string;
  versionMajor: number;
  versionMinor: number;
  flagsBits: number;
  numControls: number;
  ringBufferPow2Log2: number;
  sampleRateTrainHz: number;
  jsonHeaderByteLen: number;
  tensorArenaOffset: number;
  tensorArenaByteLen: number;
  arenaOffsetLiteEnd: number;
  arenaOffsetFullEnd: number;
  arenaOffsetUltraEnd: number;
  arenaCrc32cLite: number;
  arenaCrc32cFull: number;
  arenaCrc32cUltra: number;
  numCkptStems: number;
  controlDecimation: number;
  headSvdRank: number;
  tailTimeMs: number;
  headerCrc32c: number;
  core2Active: boolean;
  lfoEnvActive: boolean;
  stereoOut: boolean;
}

export const PreambleSnapshotSchema: z.ZodType<PreambleSnapshot> = z
  .object({
    magic: z.string(),
    versionMajor: z.number().int().nonnegative(),
    versionMinor: z.number().int().nonnegative(),
    flagsBits: z.number().int().nonnegative(),
    numControls: z.number().int().nonnegative(),
    ringBufferPow2Log2: z.number().int().nonnegative(),
    sampleRateTrainHz: z.number().int().positive(),
    jsonHeaderByteLen: z.number().int().nonnegative(),
    tensorArenaOffset: z.number().int().nonnegative(),
    tensorArenaByteLen: z.number().int().nonnegative(),
    arenaOffsetLiteEnd: z.number().int().nonnegative(),
    arenaOffsetFullEnd: z.number().int().nonnegative(),
    arenaOffsetUltraEnd: z.number().int().nonnegative(),
    arenaCrc32cLite: z.number().int().nonnegative(),
    arenaCrc32cFull: z.number().int().nonnegative(),
    arenaCrc32cUltra: z.number().int().nonnegative(),
    numCkptStems: z.number().int().nonnegative(),
    controlDecimation: z.number().int().positive(),
    headSvdRank: z.number().int().nonnegative(),
    tailTimeMs: z.number().nonnegative(),
    headerCrc32c: z.number().int().nonnegative(),
    core2Active: z.boolean(),
    lfoEnvActive: z.boolean(),
    stereoOut: z.boolean(),
  })
  .strict();

export interface ZohParamsSnapshot {
  sampleRateHostHz: number;
  rhoSr: number;
  controlDecimationHost: number;
  rhoCtrl: number;
  rAcHost: number;
  rDcHost: number;
  tauRefSamplesHost: number;
  switchXfadeSamplesHost: number;
}

export const ZohParamsSnapshotSchema: z.ZodType<ZohParamsSnapshot> = z
  .object({
    sampleRateHostHz: z.number().int().min(44_100).max(96_000),
    rhoSr: z.number().positive(),
    controlDecimationHost: z.number().int().positive(),
    rhoCtrl: z.number().positive(),
    rAcHost: z.number(),
    rDcHost: z.number(),
    tauRefSamplesHost: z.number().positive(),
    switchXfadeSamplesHost: z.number().int().positive(),
  })
  .strict();

export interface TelemetrySnapshot {
  activeTier: number;
  backend: string;
  sampleRateHz: number;
  core2Active: boolean;
  lfoEnvActive: boolean;
  stereoOut: boolean;
  activeClampMask: number;
  controlDecimationHost: number;
  maxAdaaDrive: number;
  maxPoleRadius: number;
  maxEffectiveLoopGain: number;
  tailTimeMs: number;
  zHead: [number, number, number, number];
  envStates: [number, number, number, number];
  lfoHarmonics: [number, number, number, number, number, number, number, number];
  clampedControls: number[];
  rhoSr: number;
  rhoCtrl: number;
  rAcHost: number;
  rDcHost: number;
  tauRefSamplesHost: number;
  switchXfadeSamplesHost: number;
}

export const TelemetrySnapshotSchema: z.ZodType<TelemetrySnapshot> = z
  .object({
    activeTier: z.number().int().min(0).max(2),
    backend: z.string(),
    sampleRateHz: z.number().int().min(44_100).max(96_000),
    core2Active: z.boolean(),
    lfoEnvActive: z.boolean(),
    stereoOut: z.boolean(),
    activeClampMask: z.number().int().nonnegative(),
    controlDecimationHost: z.number().int().positive(),
    maxAdaaDrive: z.number(),
    maxPoleRadius: z.number(),
    maxEffectiveLoopGain: z.number(),
    tailTimeMs: z.number().nonnegative(),
    zHead: z.tuple([z.number(), z.number(), z.number(), z.number()]),
    envStates: z.tuple([z.number(), z.number(), z.number(), z.number()]),
    lfoHarmonics: z.tuple([
      z.number(),
      z.number(),
      z.number(),
      z.number(),
      z.number(),
      z.number(),
      z.number(),
      z.number(),
    ]),
    clampedControls: z.array(z.number()).length(16),
    rhoSr: z.number().positive(),
    rhoCtrl: z.number().positive(),
    rAcHost: z.number(),
    rDcHost: z.number(),
    tauRefSamplesHost: z.number().positive(),
    switchXfadeSamplesHost: z.number().int().positive(),
  })
  .strict();

export interface ClampResultSnapshot {
  clampedControls: Record<string, ControlValue>;
  normalizedU: Record<string, number>;
  wasClamped: Record<string, boolean>;
  warnings: Record<string, string>;
}

export const ClampResultSnapshotSchema: z.ZodType<ClampResultSnapshot> = z
  .object({
    clampedControls: z.record(z.string(), z.union([z.number(), z.string(), z.boolean()])),
    normalizedU: z.record(z.string(), z.number()),
    wasClamped: z.record(z.string(), z.boolean()),
    warnings: z.record(z.string(), z.string()),
  })
  .strict();

export interface PolyAdaaParams {
  bConv?: number | undefined;
  alpha?: number | undefined;
  beta?: number | undefined;
  delta?: number | undefined;
  rAc?: number | undefined;
}

export const PolyAdaaParamsSchema: z.ZodType<PolyAdaaParams> = z
  .object({
    bConv: z.number().optional(),
    alpha: z.number().optional(),
    beta: z.number().min(-0.985).max(0.985).optional(),
    delta: z.number().min(-0.245).max(0.245).optional(),
    rAc: z.number().min(0.0).max(0.99999).optional(),
  })
  .strict();

export interface SelectiveRgLruParams {
  rLow?: number | undefined;
  drSpan?: number | undefined;
  wSel?: number | undefined;
  bGammaNorm?: number | undefined;
  cReal?: number | undefined;
  cImag?: number | undefined;
  uRotCos?: number | undefined;
  uRotSin?: number | undefined;
}

export const SelectiveRgLruParamsSchema: z.ZodType<SelectiveRgLruParams> = z
  .object({
    rLow: z.number().min(0.25).max(0.9995).optional(),
    drSpan: z.number().min(0.0).max(0.75).optional(),
    wSel: z.number().min(-4.0).max(4.0).optional(),
    bGammaNorm: z.number().optional(),
    cReal: z.number().optional(),
    cImag: z.number().optional(),
    uRotCos: z.number().min(-1.001).max(1.001).optional(),
    uRotSin: z.number().min(-1.001).max(1.001).optional(),
  })
  .strict()
  .superRefine((val, ctx) => {
    const rLow = val.rLow ?? 0.9;
    const drSpan = val.drSpan ?? 0.05;
    if (rLow + drSpan > 0.9995 + 1e-6) {
      ctx.addIssue({
        code: "custom",
        message: `CL-03 violation: rLow (${rLow}) + drSpan (${drSpan}) must be <= 0.9995`,
      });
    }
  });

export interface ModulatedUfdnParams {
  delaySamples?: number | undefined;
  gLin?: number | undefined;
  sigmaKnee?: number | undefined;
  vCeil?: number | undefined;
  rDc?: number | undefined;
}

export const ModulatedUfdnParamsSchema: z.ZodType<ModulatedUfdnParams> = z
  .object({
    delaySamples: z.number().int().min(4).optional(),
    gLin: z.number().min(0.0).max(1.05).optional(),
    sigmaKnee: z.number().min(0.0).max(2.0).optional(),
    vCeil: z.number().min(0.01).max(0.985).optional(),
    rDc: z.number().min(0.98).max(0.9995).optional(),
  })
  .strict();
