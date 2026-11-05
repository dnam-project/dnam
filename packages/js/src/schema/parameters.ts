/**
 * Parametric control descriptors (`ParameterDescriptor`), UI hints (`ParameterUiHints`),
 * physical calibration (`ParameterCalibration`), and presets (`Preset`)
 * (`ref[sdk.ts.schema.camel-case-codec-parity]`, `ref[sdk.ts.parameters.audio-param-semantics]`).
 *
 * Zero runtime JS dependencies; validated by native Rust `serde` (`dnam_engine::schema`).
 *
 * @module
 */

import type { Subdivision } from "../types.js";

/**
 * Physical calibration roles (`ControlCalibrationRole` in Rust).
 */
export const calibrationRoles = [
  "delay_time_ms",
  "lfo_rate_hz",
  "feedback",
  "mix",
  "crossover_hz",
  "stereo_mode",
  "stereo_width",
  "freeze_latch",
  "grain_size_ms",
  "grain_pitch_semitones",
  "grain_spray",
] as const;

/**
 * Physical calibration role literal union.
 */
export type CalibrationRole = (typeof calibrationRoles)[number];

/**
 * Physical calibration and MIDI tempo-sync metadata for a parameter (`controlsSchema[*].calibration`).
 */
export interface ParameterCalibration {
  readonly role: CalibrationRole;
  readonly lobattoPhysicalAnchors: readonly [number, number, number] | null;
  readonly conditionalPhysicalAnchors: Readonly<Record<string, readonly [number, number, number]>>;
  readonly tempoSyncDefaultSubdivision: Subdivision | null;
}

/**
 * UI rendering and interaction hints for a parameter (`controlsSchema[*].uiHints`).
 */
export interface ParameterUiHints {
  readonly section: string | null;
  readonly order: number;
  readonly widget: string | null;
  readonly size: string | null;
  readonly polarity: string | null;
  readonly midpoint: number | null;
  readonly detents: readonly number[];
  readonly step: number | null;
  readonly decimals: number | null;
  readonly scaleLabels: Readonly<Record<string, string>>;
  readonly optionLabels: Readonly<Record<string, string>>;
  readonly colorHex: string | null;
}

/**
 * Parametric control kinds (`"range" | "switch"`).
 */
export const parameterKinds = ["range", "switch"] as const;

/**
 * Parametric control kind literal union.
 */
export type ParameterKind = (typeof parameterKinds)[number];

/**
 * Validated, frozen parameter schema descriptor (`header.controlsSchema[*]`).
 */
export interface ParameterDescriptor {
  readonly id: string;
  readonly name: string;
  readonly type: ParameterKind;
  readonly unit: string | null;
  readonly lobattoDegree: number | null;
  readonly anchorNodesNorm: readonly number[] | null;
  readonly declaredRange: readonly [number, number] | null;
  readonly safeClampedRange: readonly [number, number] | null;
  readonly default: number | string;
  readonly conditionalMin: Readonly<Record<string, number>>;
  readonly conditionalMax: Readonly<Record<string, number>>;
  readonly declaredOptions: readonly string[] | null;
  readonly activeOptions: readonly string[] | null;
  readonly clampWarning: string | null;
  readonly calibration: ParameterCalibration | null;
  readonly uiHints: ParameterUiHints;
  readonly userMetadata: Readonly<Record<string, unknown>>;
}

/**
 * Creator factory preset snapshot (`header.presets[*]`).
 */
export interface Preset {
  readonly id: string;
  readonly name: string;
  readonly description: string | null;
  readonly controls: Readonly<Record<string, number | string>>;
  readonly userMetadata: Readonly<Record<string, unknown>>;
}
