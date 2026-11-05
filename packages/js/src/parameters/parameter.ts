/**
 * WebAudio `AudioParam`-aligned `Parameter<TId, TValue>` controller used by both `Engine` and
 * `DnamAudioNode` (`ref[sdk.ts.parameters.audio-param-semantics]`).
 *
 * @module
 */

import type { ParameterDescriptor, ParameterKind, ParameterUiHints } from "../schema/parameters.js";
import { numericToParameterValue, parameterValueToNumeric } from "./clamp.js";

/**
 * Options for `parameter.setValue(value, options)` and `parameters.set(values, options)`.
 */
export interface SetParameterOptions {
  /**
   * When `true`, snaps immediately to `value` without 3 kHz smoothing (`set_control_immediate`).
   */
  readonly immediate?: boolean;
}

/**
 * Internal bridge connecting `Parameter` / `ParameterMap` to either a synchronous `Engine`
 * or an `AudioWorklet` `DnamAudioNode`.
 */
export interface ParameterBackend {
  getClampedNumericValue(index: number): number;
  getRawNumericValues(): Float32Array;
  setNumericValue(index: number, numericValue: number, immediate: boolean): void;
  setNumericValuesBatch(
    indices: readonly number[],
    numericValues: readonly number[],
    immediate: boolean,
  ): void;
  resetAllToDefaults(): void;
  getActiveClampMask(): number;
  toPhysical(index: number, normalizedU: number): number;
  toNormalized(index: number, physicalValue: number): number;
}

/**
 * Live typed parameter controller modeled on WebAudio's `AudioParam`
 * (`ref[sdk.ts.parameters.audio-param-semantics]`).
 *
 * @example
 * ```ts
 * const gain = engine.parameters.get("gain");
 * gain.value = 7.5; // 3 kHz smoothed
 * gain.setValue(7.5, { immediate: true });
 * ```
 */
export class Parameter<
  TId extends string = string,
  TValue extends number | string = number | string,
> {
  readonly id: TId;
  readonly index: number;
  readonly descriptor: ParameterDescriptor;
  readonly #backend: ParameterBackend;

  constructor(id: TId, index: number, descriptor: ParameterDescriptor, backend: ParameterBackend) {
    this.id = id;
    this.index = index;
    this.descriptor = descriptor;
    this.#backend = backend;
  }

  /**
   * Human-readable display label (`descriptor.name`).
   */
  get name(): string {
    return this.descriptor.name;
  }

  /**
   * Parameter kind (`"range" | "switch"`).
   */
  get kind(): ParameterKind {
    return this.descriptor.type;
  }

  /**
   * Physical unit label (e.g. `"dB"`, `"ms"`), or `undefined` for unitless/switch parameters.
   */
  get unit(): string | undefined {
    return this.descriptor.unit ?? undefined;
  }

  /**
   * Default parameter value (`number` for `"range"`, option `string` for `"switch"`).
   */
  get defaultValue(): TValue {
    return this.descriptor.default as TValue;
  }

  /**
   * Declared minimum physical value (`"range"` only).
   */
  get minValue(): number | undefined {
    return this.descriptor.declaredRange?.[0];
  }

  /**
   * Declared maximum physical value (`"range"` only).
   */
  get maxValue(): number | undefined {
    return this.descriptor.declaredRange?.[1];
  }

  /**
   * Base safe clamped physical interval `[min, max]` (`"range"` only).
   */
  get safeRange(): readonly [number, number] | undefined {
    return this.descriptor.safeClampedRange ?? undefined;
  }

  /**
   * Declared switch option IDs (`"switch"` only).
   */
  get options(): readonly string[] | undefined {
    return this.descriptor.declaredOptions ?? undefined;
  }

  /**
   * Observed/active switch option IDs (`"switch"` only).
   */
  get activeOptions(): readonly string[] | undefined {
    return this.descriptor.activeOptions ?? undefined;
  }

  /**
   * UI interaction and rendering hints (`descriptor.uiHints`).
   */
  get uiHints(): ParameterUiHints {
    return this.descriptor.uiHints;
  }

  /**
   * Current live clamped value (`number` for `"range"`, option `string` for `"switch"`).
   */
  get value(): TValue {
    const num = this.#backend.getClampedNumericValue(this.index);
    return numericToParameterValue(this.descriptor, num) as TValue;
  }

  /**
   * Schedules a 3 kHz smoothed update to `newValue` (`set_control_value`).
   */
  set value(newValue: TValue) {
    this.setValue(newValue, { immediate: false });
  }

  /**
   * Updates the parameter value, optionally snapping immediately (`options.immediate === true`).
   */
  setValue(newValue: TValue, options?: SetParameterOptions): this {
    const num = parameterValueToNumeric(this.descriptor, newValue);
    this.#backend.setNumericValue(this.index, num, options?.immediate ?? false);
    return this;
  }

  /**
   * Resets this parameter to `this.defaultValue` immediately.
   */
  reset(): this {
    return this.setValue(this.defaultValue, { immediate: true });
  }

  /**
   * Whether this parameter is currently clamped by the 3-Phase control audit (`activeClampMask`).
   */
  get clamped(): boolean {
    if (this.index >= 16) {
      return false;
    }
    return (this.#backend.getActiveClampMask() & (1 << this.index)) !== 0;
  }

  /**
   * Human-readable clamp warning message if declared on the parameter descriptor.
   */
  get clampWarning(): string | undefined {
    return this.descriptor.clampWarning ?? undefined;
  }

  /**
   * Maps a normalized coordinate $u \in [-1, 1]$ to this parameter's physical range.
   */
  toPhysical(normalizedU: number): number {
    return this.#backend.toPhysical(this.index, normalizedU);
  }

  /**
   * Maps a physical value to its clamped normalized coordinate $u \in [-1, 1]$.
   */
  toNormalized(physicalValue: number): number {
    return this.#backend.toNormalized(this.index, physicalValue);
  }
}
