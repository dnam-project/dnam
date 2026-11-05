/**
 * Iterable `ParameterMap<TParams>` collection exposed as `engine.parameters` and `node.parameters`
 * (`ref[sdk.ts.parameters.audio-param-semantics]`).
 *
 * @module
 */

import { DnamError } from "../errors.js";
import { deepFreeze } from "../internal/freeze.js";
import type { ParameterDescriptor } from "../schema/parameters.js";
import { type ClampReport, type ControlSpace, parameterValueToNumeric } from "./clamp.js";
import { Parameter, type ParameterBackend, type SetParameterOptions } from "./parameter.js";
import type { ParameterValueMap } from "./typed-model.js";

/**
 * Typed collection of `Parameter` controllers on a running `Engine` or `DnamAudioNode`
 * (`ref[sdk.ts.parameters.audio-param-semantics]`).
 *
 * @example
 * ```ts
 * engine.parameters.get("gain").value = 7.5;
 * engine.parameters.set({ gain: 7.5, boost: "on" }, { immediate: true });
 * ```
 */
export class ParameterMap<TParams extends ParameterValueMap = Record<string, number | string>>
  implements Iterable<readonly [keyof TParams & string, Parameter<keyof TParams & string>]>
{
  readonly #byId: ReadonlyMap<string, Parameter>;
  readonly #ordered: readonly Parameter[];
  readonly #backend: ParameterBackend;
  readonly #controlSpace: ControlSpace<TParams>;

  constructor(
    descriptors: readonly ParameterDescriptor[],
    backend: ParameterBackend,
    controlSpace: ControlSpace<TParams>,
  ) {
    this.#backend = backend;
    this.#controlSpace = controlSpace;
    const byId = new Map<string, Parameter>();
    const ordered: Parameter[] = [];
    for (let i = 0; i < descriptors.length; i++) {
      const desc = descriptors[i];
      if (!desc) {
        continue;
      }
      const param = new Parameter(desc.id, i, desc, backend);
      byId.set(desc.id, param);
      ordered.push(param);
    }
    this.#byId = byId;
    this.#ordered = Object.freeze(ordered);
  }

  /**
   * Number of exposed parameters on the model.
   */
  get size(): number {
    return this.#ordered.length;
  }

  /**
   * Returns `true` if a parameter with `id` exists on the model.
   */
  has(id: string): id is keyof TParams & string {
    return this.#byId.has(id);
  }

  /**
   * Returns the typed `Parameter<K, TParams[K]>` for `id`, throwing `DnamError`
   * (`code: "ERR_DNAM_PARAMETER"`) if `id` is not declared on the model.
   */
  get<K extends keyof TParams & string>(id: K): Parameter<K, TParams[K]> {
    const param = this.#byId.get(id);
    if (!param) {
      throw new DnamError("ERR_DNAM_PARAMETER", `Unknown parameter '${id}'`, {
        parameterId: id,
      });
    }
    return param as unknown as Parameter<K, TParams[K]>;
  }

  /**
   * Updates multiple parameters at once, evaluating switch-conditioned 3-Phase clamping
   * atomically when `options.immediate === true`.
   */
  set(values: Partial<TParams>, options?: SetParameterOptions): this {
    const entries = Object.entries(values as Record<string, number | string | undefined>);
    if (entries.length === 0) {
      return this;
    }
    const indices: number[] = [];
    const nums: number[] = [];
    for (const [id, val] of entries) {
      if (val === undefined) {
        continue;
      }
      const param = this.#byId.get(id);
      if (!param) {
        throw new DnamError("ERR_DNAM_PARAMETER", `Unknown parameter '${id}'`, {
          parameterId: id,
        });
      }
      indices.push(param.index);
      nums.push(parameterValueToNumeric(param.descriptor, val));
    }
    if (indices.length > 0) {
      this.#backend.setNumericValuesBatch(indices, nums, options?.immediate ?? false);
    }
    return this;
  }

  /**
   * Resets all parameters to their declared defaults (`reset_controls_to_defaults`).
   */
  resetAll(): this {
    this.#backend.resetAllToDefaults();
    return this;
  }

  /**
   * Returns a frozen plain object of all current live parameter values (`Readonly<TParams>`).
   */
  toObject(): Readonly<TParams> {
    const out: Record<string, number | string> = {};
    for (const param of this.#ordered) {
      out[param.id] = param.value;
    }
    return deepFreeze(out as TParams);
  }

  /**
   * Returns a full 3-Phase `ClampReport` for the current requested parameter values.
   */
  clampedReport(): ClampReport<TParams> {
    const raw = this.#backend.getRawNumericValues();
    const rawObj: Record<string, number | string> = {};
    for (const param of this.#ordered) {
      const rVal = raw[param.index] ?? 0;
      if (param.descriptor.type === "switch") {
        const opts = param.descriptor.declaredOptions ?? [];
        rawObj[param.id] = opts[Math.round(rVal)] ?? String(param.descriptor.default);
      } else {
        rawObj[param.id] = rVal;
      }
    }
    return this.#controlSpace.clamp(rawObj as Partial<TParams>);
  }

  /**
   * Iterates over all `Parameter` instances in declaration order.
   */
  values(): IterableIterator<Parameter> {
    return this.#ordered[Symbol.iterator]();
  }

  /**
   * Iterates over all parameter IDs in declaration order.
   */
  *keys(): IterableIterator<keyof TParams & string> {
    for (const param of this.#ordered) {
      yield param.id as keyof TParams & string;
    }
  }

  /**
   * Iterates over `[id, parameter]` pairs in declaration order.
   */
  *[Symbol.iterator](): IterableIterator<
    readonly [keyof TParams & string, Parameter<keyof TParams & string>]
  > {
    for (const param of this.#ordered) {
      yield [
        param.id as keyof TParams & string,
        param as Parameter<keyof TParams & string>,
      ] as const;
    }
  }
}
