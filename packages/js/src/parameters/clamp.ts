/**
 * Pre-compiled 3-Phase control space (`ControlSpace`) and `clampParameters(...)`
 * (`ref[sdk.ts.parameters.audio-param-semantics]`, `ref[sdk.ts.model.containers-and-checkpoints]`).
 *
 * @module
 */

import { DnamError } from "../errors.js";
import { deepFreeze } from "../internal/freeze.js";
import { registerDisposable } from "../internal/handle.js";
import type { NativeControlSpaceHandle } from "../internal/native/driver.js";
import { getActiveDriver } from "../internal/native/load.js";
import { Model } from "../model/model.js";
import { decodeHeader, encodeHeader, type ModelHeader } from "../schema/header.js";
import type { ParameterDescriptor } from "../schema/parameters.js";
import type { ParameterValueMap } from "./typed-model.js";

/**
 * Per-parameter result inside a `ClampReport`.
 */
export interface ClampedParameterEntry {
  readonly id: string;
  readonly kind: "range" | "switch";
  readonly requested: number | string;
  readonly effective: number | string;
  readonly normalizedU: number;
  readonly wasClamped: boolean;
  readonly warning: string | undefined;
}

/**
 * Complete 3-Phase control clamping report returned by `controlSpace.clamp(...)` and `clampParameters(...)`.
 */
export interface ClampReport<TParams extends ParameterValueMap = Record<string, number | string>> {
  readonly values: Readonly<TParams>;
  readonly normalizedU: Readonly<Record<keyof TParams & string, number>>;
  readonly entries: Readonly<Record<keyof TParams & string, ClampedParameterEntry>>;
  readonly clampedIds: readonly (keyof TParams & string)[];
  readonly activeClampMask: number;
  readonly anyClamped: boolean;
}

/**
 * Converts a user-supplied physical number or switch option string into its numeric control value.
 */
export function parameterValueToNumeric(
  descriptor: ParameterDescriptor,
  value: number | string | undefined,
): number {
  if (value === undefined) {
    return defaultNumericValue(descriptor);
  }
  if (descriptor.type === "switch") {
    const opts = descriptor.declaredOptions ?? [];
    if (typeof value === "string") {
      const idx = opts.indexOf(value);
      if (idx === -1) {
        throw new DnamError(
          "ERR_DNAM_PARAMETER",
          `Invalid option '${value}' for switch parameter '${descriptor.id}' (declared: ${opts.join(", ")})`,
          { parameterId: descriptor.id },
        );
      }
      return idx;
    }
    if (typeof value === "number" && Number.isInteger(value) && value >= 0 && value < opts.length) {
      return value;
    }
    throw new DnamError(
      "ERR_DNAM_PARAMETER",
      `Invalid value '${String(value)}' for switch parameter '${descriptor.id}'`,
      { parameterId: descriptor.id },
    );
  }
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new DnamError(
      "ERR_DNAM_PARAMETER",
      `Range parameter '${descriptor.id}' requires a finite number, got '${String(value)}'`,
      { parameterId: descriptor.id },
    );
  }
  return value;
}

/**
 * Converts a numeric control value back into its canonical physical number (`"range"`)
 * or option string (`"switch"`).
 */
export function numericToParameterValue(
  descriptor: ParameterDescriptor,
  numericValue: number,
): number | string {
  if (descriptor.type === "switch") {
    const opts = descriptor.declaredOptions ?? [];
    const idx = Math.round(numericValue);
    return opts[idx] ?? String(descriptor.default);
  }
  return numericValue;
}

/**
 * Returns the default numeric value for `descriptor`.
 */
export function defaultNumericValue(descriptor: ParameterDescriptor): number {
  if (descriptor.type === "switch") {
    const opts = descriptor.declaredOptions ?? [];
    const idx = opts.indexOf(String(descriptor.default));
    return Math.max(idx, 0);
  }
  return typeof descriptor.default === "number" ? descriptor.default : 0;
}

/**
 * Pre-compiled 3-Phase control clamping space (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]`)
 * backed by Rust `ControlClampDescriptor` records (`ref[sdk.ts.parameters.audio-param-semantics]`).
 */
export class ControlSpace<TParams extends ParameterValueMap = Record<string, number | string>>
  implements Disposable
{
  readonly #handle: NativeControlSpaceHandle;
  readonly #descriptors: readonly ParameterDescriptor[];
  readonly #indexById: ReadonlyMap<string, number>;
  readonly #rawScratch: Float32Array;
  readonly #clampedScratch: Float32Array;
  readonly #normScratch: Float32Array;
  readonly #release: () => void;
  #disposed = false;

  private constructor(
    handle: NativeControlSpaceHandle,
    descriptors: readonly ParameterDescriptor[],
  ) {
    this.#handle = handle;
    this.#descriptors = descriptors;
    const map = new Map<string, number>();
    for (let i = 0; i < descriptors.length; i++) {
      const d = descriptors[i];
      if (d) {
        map.set(d.id, i);
      }
    }
    this.#indexById = map;
    const n = descriptors.length;
    this.#rawScratch = new Float32Array(n);
    this.#clampedScratch = new Float32Array(n);
    this.#normScratch = new Float32Array(n);
    this.#release = registerDisposable(this, () => {
      this.#disposed = true;
      this.#handle.dispose();
    });
  }

  /**
   * Compiles a `ControlSpace` from a `Model`, `ModelHeader`, or raw wire JSON string.
   */
  static from<TParams extends ParameterValueMap = Record<string, number | string>>(
    modelOrHeader: Model<TParams> | ModelHeader | string,
  ): ControlSpace<TParams> {
    let headerJson: string;
    let descriptors: readonly ParameterDescriptor[];
    if (modelOrHeader instanceof Model) {
      headerJson = modelOrHeader.rawHeaderJson;
      descriptors = modelOrHeader.parameters;
    } else if (typeof modelOrHeader === "string") {
      headerJson = modelOrHeader;
      descriptors = decodeHeader(modelOrHeader).controlsSchema;
    } else {
      headerJson = encodeHeader(modelOrHeader);
      descriptors = modelOrHeader.controlsSchema;
    }
    const handle = getActiveDriver().createControlSpace(headerJson);
    return new ControlSpace<TParams>(handle, descriptors);
  }

  #ensureAlive(): void {
    if (this.#disposed) {
      throw new DnamError("ERR_DNAM_DISPOSED", "ControlSpace has been disposed");
    }
  }

  /**
   * Parameter descriptors in declaration order.
   */
  get descriptors(): readonly ParameterDescriptor[] {
    this.#ensureAlive();
    return this.#descriptors;
  }

  /**
   * Number of parameters in this control space.
   */
  get size(): number {
    this.#ensureAlive();
    return this.#descriptors.length;
  }

  /**
   * Looks up the 0-based index of parameter `id`, throwing `DnamError` (`"ERR_DNAM_PARAMETER"`) if unknown.
   */
  indexOf(id: string): number {
    this.#ensureAlive();
    const idx = this.#indexById.get(id);
    if (idx === undefined) {
      throw new DnamError("ERR_DNAM_PARAMETER", `Unknown parameter '${id}'`, {
        parameterId: id,
      });
    }
    return idx;
  }

  /**
   * Evaluates 3-Phase clamping directly over raw numeric `Float32Array` values in compiled Rust
   * (`0` JS object allocations), returning the 32-bit active clamp bitmask.
   */
  clampNumericInto(
    rawNumeric: Float32Array,
    clampedOut: Float32Array,
    normOut?: Float32Array,
  ): number {
    this.#ensureAlive();
    return this.#handle.clampInto(rawNumeric, clampedOut, normOut ?? this.#normScratch) >>> 0;
  }

  /**
   * Evaluates 3-Phase clamping over `values` in compiled Rust and returns a frozen `ClampReport`.
   */
  clamp(
    values?: Partial<TParams> | Readonly<Record<string, number | string>>,
  ): ClampReport<TParams> {
    this.#ensureAlive();
    const inputRec = (values ?? {}) as Readonly<Record<string, number | string | undefined>>;
    for (const key of Object.keys(inputRec)) {
      if (!this.#indexById.has(key)) {
        throw new DnamError("ERR_DNAM_PARAMETER", `Unknown parameter '${key}'`, {
          parameterId: key,
        });
      }
    }
    const n = this.#descriptors.length;
    for (let i = 0; i < n; i++) {
      const desc = this.#descriptors[i];
      if (!desc) {
        continue;
      }
      this.#rawScratch[i] = parameterValueToNumeric(desc, inputRec[desc.id]);
    }
    const mask =
      this.#handle.clampInto(this.#rawScratch, this.#clampedScratch, this.#normScratch) >>> 0;

    const outValues: Record<string, number | string> = {};
    const outNorm: Record<string, number> = {};
    const outEntries: Record<string, ClampedParameterEntry> = {};
    const clampedIds: (keyof TParams & string)[] = [];

    for (let i = 0; i < n; i++) {
      const desc = this.#descriptors[i];
      if (!desc) {
        continue;
      }
      const requested =
        inputRec[desc.id] !== undefined
          ? (inputRec[desc.id] as number | string)
          : (desc.default as number | string);
      const clampedNum = this.#clampedScratch[i] ?? 0;
      const normU = this.#normScratch[i] ?? 0;
      const effective = numericToParameterValue(desc, clampedNum);
      const wasClamped = (mask & (1 << i)) !== 0;
      outValues[desc.id] = effective;
      outNorm[desc.id] = normU;
      outEntries[desc.id] = {
        id: desc.id,
        kind: desc.type,
        requested,
        effective,
        normalizedU: normU,
        wasClamped,
        warning: wasClamped ? (desc.clampWarning ?? undefined) : undefined,
      };
      if (wasClamped) {
        clampedIds.push(desc.id as keyof TParams & string);
      }
    }

    return deepFreeze({
      values: outValues as TParams,
      normalizedU: outNorm as Record<keyof TParams & string, number>,
      entries: outEntries as Record<keyof TParams & string, ClampedParameterEntry>,
      clampedIds,
      activeClampMask: mask & 0xffff,
      anyClamped: clampedIds.length > 0,
    });
  }

  /**
   * Maps normalized coordinate $u \in [-1, 1]$ to physical value for parameter `id`.
   */
  toPhysical(id: keyof TParams & string, normalizedU: number): number {
    const idx = this.indexOf(id);
    return this.#handle.toPhysical(idx, normalizedU);
  }

  /**
   * Maps a physical value to clamped normalized coordinate $u \in [-1, 1]$ for parameter `id`.
   */
  toNormalized(id: keyof TParams & string, physicalValue: number): number {
    const idx = this.indexOf(id);
    return this.#handle.toNormalized(idx, physicalValue);
  }

  /**
   * Releases the underlying native control space handle idempotently.
   */
  dispose(): void {
    this.#release();
  }

  /**
   * Disposes the control space (`using` protocol).
   */
  [Symbol.dispose](): void {
    this.dispose();
  }
}

/**
 * Evaluates 3-Phase control clamping over `values` against a `Model`, `ModelHeader`, or `ControlSpace`
 * (`ref[sdk.ts.parameters.audio-param-semantics]`).
 *
 * @example
 * ```ts
 * const report = clampParameters(model, { gain: 12, boost: "on" });
 * console.log(report.anyClamped, report.values.gain);
 * ```
 */
export function clampParameters<
  TParams extends ParameterValueMap = Record<string, number | string>,
>(
  source: ControlSpace<TParams> | Model<TParams> | ModelHeader | string,
  values?: Partial<TParams> | Readonly<Record<string, number | string>>,
): ClampReport<TParams> {
  if (source instanceof ControlSpace) {
    return source.clamp(values);
  }
  const space = ControlSpace.from<TParams>(source);
  try {
    return space.clamp(values);
  } finally {
    space.dispose();
  }
}
