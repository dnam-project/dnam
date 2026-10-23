/**
 * Parametric Control Bus (`ControlSpace`, `ControlRack<TControls>`, `clampControlValues`),
 * Node-Anchored Lobatto Basis (`LobattoBasis`), Rank-4 Factorization Machine (`Rank4Fm`),
 * 4-Harmonic Chebyshev LFO (`chebyshevLfo`), and Trust-Region Soft-Clipper (`trustRegionSoftclip`).
 */

import { type BackendChoice, type DnamControlSpaceHandle, resolveDriver } from "./drivers/index.js";
import { DnamControlError } from "./errors.js";
import { type ControlMap, type ControlValue, validateStandardSchema } from "./schema/base.js";
import {
  type ControlSchemaDef,
  computeSkewExponent,
  normalizedToPhysical as hintsNormalizedToPhysical,
  physicalToNormalized as hintsPhysicalToNormalized,
  type PresetDef,
} from "./schema/controls.js";
import {
  type DnamJsonHeader,
  DnamJsonHeaderSchema,
  dnamJsonHeaderToWire,
} from "./schema/header.js";
import type { ClampResultSnapshot } from "./schema/runtime.js";

function coerceControlValueToNumeric(
  def: ControlSchemaDef,
  rawVal: ControlValue | undefined,
): number {
  if (def.type === "switch") {
    const declOpts = def.declaredOptions ?? [];
    if (rawVal === undefined) {
      const defIdx = typeof def.default === "string" ? declOpts.indexOf(def.default) : 0;
      return Math.max(defIdx, 0);
    }
    if (typeof rawVal === "boolean") {
      return rawVal ? 1 : 0;
    }
    if (typeof rawVal === "number") {
      if (!Number.isFinite(rawVal)) {
        throw new DnamControlError(`Non-finite numeric value for switch control '${def.id}'`);
      }
      return Math.round(rawVal);
    }
    const idx = declOpts.indexOf(rawVal);
    if (idx < 0) {
      throw new DnamControlError(
        `Unknown option '${rawVal}' for switch control '${def.id}' (declared: ${declOpts.join(", ")})`,
      );
    }
    return idx;
  }

  // range control
  if (rawVal === undefined) {
    return typeof def.default === "number" ? def.default : 0;
  }
  if (typeof rawVal !== "number" || !Number.isFinite(rawVal)) {
    throw new DnamControlError(
      `Range control '${def.id}' requires a finite number, got '${String(rawVal)}'`,
    );
  }
  return rawVal;
}

/**
 * Pre-compiled 3-Phase control clamping space (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]`)
 * evaluating all 16 controls in `< 100 ns` without per-call JSON parsing.
 */
export class ControlSpace implements Disposable {
  public readonly header: DnamJsonHeader;
  private readonly handle: DnamControlSpaceHandle;
  private readonly controlById: Map<string, { index: number; def: ControlSchemaDef }>;
  private readonly rawNumericBuf: Float32Array;

  public constructor(
    headerInput: DnamJsonHeader | Record<string, unknown> | string,
    backend?: BackendChoice,
  ) {
    const parsed: unknown = typeof headerInput === "string" ? JSON.parse(headerInput) : headerInput;
    this.header = validateStandardSchema(DnamJsonHeaderSchema, parsed, "DnamJsonHeader");
    const wireJson = JSON.stringify(dnamJsonHeaderToWire(this.header));
    const driver = resolveDriver(backend);
    this.handle = driver.createControlSpace(wireJson);
    this.controlById = new Map();
    const schema = this.header.controlsSchema;
    this.rawNumericBuf = new Float32Array(schema.length);
    schema.forEach((def, index) => {
      this.controlById.set(def.id, { index, def });
    });
  }

  /**
   * Evaluates 3-Phase control clamping over a pre-coerced `Float32Array` of numeric control values.
   */
  public clampNumeric(rawNumeric: Float32Array): ClampResultSnapshot {
    const schema = this.header.controlsSchema;
    const rawResult = this.handle.clamp(rawNumeric);
    const clampedControls: Record<string, ControlValue> = {};
    const normalizedU: Record<string, number> = {};
    const wasClamped: Record<string, boolean> = {};
    const warnings: Record<string, string> = {};

    for (let i = 0; i < schema.length; i++) {
      const def = schema[i];
      if (!def) {
        continue;
      }
      const numVal = rawResult.clampedNumeric[i] ?? 0;
      const uVal = rawResult.normalizedU[i] ?? 0;
      const clampedFlag = rawResult.wasClamped[i] ?? false;

      if (def.type === "switch") {
        const declOpts = def.declaredOptions ?? [];
        const optIdx = Math.round(numVal);
        clampedControls[def.id] = declOpts[optIdx] ?? String(optIdx);
      } else {
        clampedControls[def.id] = numVal;
      }
      normalizedU[def.id] = uVal;
      wasClamped[def.id] = clampedFlag;
      if (clampedFlag && def.clampWarning) {
        warnings[def.id] = def.clampWarning;
      }
    }

    return {
      clampedControls,
      normalizedU,
      wasClamped,
      warnings,
    };
  }

  /**
   * Evaluates 3-Phase control clamping over `controls` (`Record<string, number | string | boolean>`),
   * returning clamped physical values, normalized coordinates $u_p \in [-1, +1]$, boolean clamp flags,
   * and active clamp warnings.
   */
  public clamp(controls: ControlMap = {}): ClampResultSnapshot {
    for (const key in controls) {
      if (Object.hasOwn(controls, key) && !this.controlById.has(key)) {
        throw new DnamControlError(`Unknown control ID '${key}'`);
      }
    }

    const schema = this.header.controlsSchema;
    const rawNumeric = this.rawNumericBuf;
    for (let i = 0; i < schema.length; i++) {
      const def = schema[i];
      if (!def) {
        continue;
      }
      rawNumeric[i] = coerceControlValueToNumeric(def, controls[def.id]);
    }

    return this.clampNumeric(rawNumeric);
  }

  /**
   * Resolves a control ID and raw value into `(ctrlIdx, numericValue)` for `DnamEngine`.
   */
  public resolveNumericControl(
    id: string,
    value: ControlValue,
  ): { index: number; numericValue: number; def: ControlSchemaDef } {
    const entry = this.controlById.get(id);
    if (!entry) {
      throw new DnamControlError(`Unknown control ID '${id}'`);
    }
    const numericValue = coerceControlValueToNumeric(entry.def, value);
    return { index: entry.index, numericValue, def: entry.def };
  }

  public dispose(): void {
    this.handle.dispose();
  }

  public [Symbol.dispose](): void {
    this.dispose();
  }
}

/**
 * Convenience function to evaluate 3-Phase control clamping on a `DnamJsonHeader` and control map.
 */
export function clampControlValues(
  header: DnamJsonHeader | Record<string, unknown> | string,
  controls: ControlMap,
  backend?: BackendChoice,
): ClampResultSnapshot {
  using space = new ControlSpace(header, backend);
  return space.clamp(controls);
}

export interface SetControlOptions {
  immediate?: boolean | undefined;
}

export interface ControlSink {
  setNumericControl(ctrlIdx: number, numericValue: number, immediate: boolean): void;
}

/**
 * Type-safe parametric control rack bound to a `DnamEngine<TControls>`, supporting
 * physical & normalized UI gesture mapping (`midpoint` power-skew), preset loading,
 * and 3-Phase control clamping.
 */
export class ControlRack<TControls extends ControlMap = ControlMap> {
  private readonly space: ControlSpace;
  private readonly sink: ControlSink;
  private readonly presetsById: Map<string, PresetDef>;
  private readonly currentValues: Map<string, ControlValue>;
  private readonly currentNumeric: Float32Array;

  public constructor(space: ControlSpace, sink: ControlSink) {
    this.space = space;
    this.sink = sink;
    this.presetsById = new Map();
    for (const preset of space.header.presets) {
      this.presetsById.set(preset.id, preset);
    }
    this.currentValues = new Map();
    const schema = space.header.controlsSchema;
    this.currentNumeric = new Float32Array(schema.length);
    for (let i = 0; i < schema.length; i++) {
      const def = schema[i];
      if (!def) {
        continue;
      }
      this.currentValues.set(def.id, def.default);
      this.currentNumeric[i] = coerceControlValueToNumeric(def, def.default);
    }
  }

  public get schema(): readonly ControlSchemaDef[] {
    return this.space.header.controlsSchema;
  }

  public get presets(): readonly PresetDef[] {
    return this.space.header.presets;
  }

  /**
   * Sets a single control by ID (`"gain"`, `"boost"`, etc.).
   * By default, smooths transitions over the $3\text{ kHz}$ control bus (`immediate: false`).
   */
  public set<K extends keyof TControls & string>(
    id: K,
    value: TControls[K],
    options?: SetControlOptions,
  ): ClampResultSnapshot {
    const immediate = options?.immediate ?? false;
    const { index, numericValue } = this.space.resolveNumericControl(id, value);
    this.sink.setNumericControl(index, numericValue, immediate);
    this.currentValues.set(id, value);
    this.currentNumeric[index] = numericValue;
    return this.space.clampNumeric(this.currentNumeric);
  }

  /**
   * Updates multiple controls atomically and returns the resulting 3-Phase `ClampResultSnapshot`.
   */
  public setAll(controls: Partial<TControls>, options?: SetControlOptions): ClampResultSnapshot {
    const immediate = options?.immediate ?? false;
    for (const id in controls) {
      if (Object.hasOwn(controls, id)) {
        const val = controls[id];
        if (val !== undefined) {
          const { index, numericValue } = this.space.resolveNumericControl(id, val);
          this.sink.setNumericControl(index, numericValue, immediate);
          this.currentValues.set(id, val);
          this.currentNumeric[index] = numericValue;
        }
      }
    }
    return this.space.clampNumeric(this.currentNumeric);
  }

  /**
   * Applies a built-in preset by `presetId` (e.g. `"warm_slapback"`).
   */
  public applyPreset(presetId: string, options?: SetControlOptions): ClampResultSnapshot {
    const preset = this.presetsById.get(presetId);
    if (!preset) {
      throw new DnamControlError(`Unknown preset ID '${presetId}'`);
    }
    const immediate = options?.immediate ?? true;
    for (const [id, val] of Object.entries(preset.controls)) {
      const { index, numericValue } = this.space.resolveNumericControl(id, val);
      this.sink.setNumericControl(index, numericValue, immediate);
      this.currentValues.set(id, val);
      this.currentNumeric[index] = numericValue;
    }
    return this.space.clampNumeric(this.currentNumeric);
  }

  /**
   * Returns the current clamped value of control `id`.
   */
  public get<K extends keyof TControls & string>(id: K): TControls[K] {
    const snap = this.space.clampNumeric(this.currentNumeric);
    if (!(id in snap.clampedControls)) {
      throw new DnamControlError(`Unknown control ID '${id}'`);
    }
    return snap.clampedControls[id] as TControls[K];
  }

  /**
   * Returns the full 3-Phase `ClampResultSnapshot` for the current control rack state.
   */
  public clampSnapshot(): ClampResultSnapshot {
    return this.space.clampNumeric(this.currentNumeric);
  }

  /**
   * Computes the power-skew exponent $\gamma_{\text{skew}}$ for a `range` control `id`.
   */
  public skewExponent<K extends keyof TControls & string>(id: K): number {
    const { def } = this.space.resolveNumericControl(id, this.currentValues.get(id) ?? 0);
    const range = def.declaredRange ?? [0, 1];
    return computeSkewExponent(def.uiHints, range[0], range[1]);
  }

  /**
   * Maps a normalized UI gesture coordinate $t \in [0, 1]$ to physical range value for control `id`.
   */
  public normalizedToPhysical<K extends keyof TControls & string>(id: K, tNorm: number): number {
    const { def } = this.space.resolveNumericControl(id, this.currentValues.get(id) ?? 0);
    const range = def.declaredRange ?? [0, 1];
    return hintsNormalizedToPhysical(def.uiHints, tNorm, range[0], range[1]);
  }

  /**
   * Maps a physical range value to normalized UI gesture coordinate $t \in [0, 1]$ for control `id`.
   */
  public physicalToNormalized<K extends keyof TControls & string>(id: K, xPhys: number): number {
    const { def } = this.space.resolveNumericControl(id, this.currentValues.get(id) ?? 0);
    const range = def.declaredRange ?? [0, 1];
    return hintsPhysicalToNormalized(def.uiHints, xPhys, range[0], range[1]);
  }
}

export interface LobattoEvalOptions {
  anchors?: readonly [number, number, number] | Float64Array | undefined;
  degree?: number | undefined;
  backend?: BackendChoice | undefined;
}

const DEFAULT_LOBATTO_ANCHORS = new Float64Array([0.0, -1.0, 1.0]);

function normalizeLobattoAnchors(
  anchors?: readonly [number, number, number] | Float64Array,
): Float64Array {
  if (anchors === undefined) {
    return DEFAULT_LOBATTO_ANCHORS;
  }
  if (anchors instanceof Float64Array) {
    return anchors;
  }
  return new Float64Array(anchors);
}

/**
 * Node-Anchored Newton/Lobatto polynomial basis (`CL-13`) with exact zero-flop default-preset
 * anchoring ($\psi_{p,k}(u_{p,1}) \equiv 0$).
 */
export const LobattoBasis = {
  /**
   * Evaluates $[\psi_1(u), \psi_2(u), \psi_3(u)]$ at a scalar normalized coordinate $u \in [-1, +1]$.
   */
  eval(u: number, options: LobattoEvalOptions = {}): [number, number, number] {
    const driver = resolveDriver(options.backend);
    const anc = normalizeLobattoAnchors(options.anchors);
    return driver.evalLobattoBasis(u, anc, options.degree ?? 3);
  },

  /**
   * Evaluates $[\psi_1, \psi_2, \psi_3]$ over a 1D `Float64Array` slice `u` of length $N$
   * into a flattened `Float64Array` of length $3N$.
   */
  evalSlice(u: Float64Array, out?: Float64Array, options: LobattoEvalOptions = {}): Float64Array {
    const driver = resolveDriver(options.backend);
    const anc = normalizeLobattoAnchors(options.anchors);
    const target = out ?? new Float64Array(u.length * 3);
    driver.evalLobattoBasisSlice(u, target, anc, options.degree ?? 3);
    return target;
  },
} as const;

/**
 * Rank-4 Factorization Machine (`CL-14`) cross-control interaction kernel:
 * $$\mathbf{e}_{\text{FM},r} = \frac{1}{2}\left[\left(\sum_{j=1}^D \mathbf{V}_{j,r}\phi_j\right)^2 - \sum_{j=1}^D \mathbf{V}_{j,r}^2\phi_j^2\right]$$
 */
export const Rank4Fm = {
  eval(
    features: Float64Array | readonly number[],
    vFactorsFlat: Float64Array | readonly number[],
    backend?: BackendChoice,
  ): [number, number, number, number] {
    const driver = resolveDriver(backend);
    const featF64 = features instanceof Float64Array ? features : new Float64Array(features);
    const vF64 =
      vFactorsFlat instanceof Float64Array ? vFactorsFlat : new Float64Array(vFactorsFlat);
    return driver.evalRank4Fm(featF64, vF64);
  },
} as const;

/**
 * Evaluates the 4-Harmonic Chebyshev LFO basis $[T_1(c), T_2(c), T_3(c), T_4(c)]$ (`CL-16`)
 * for a scalar `number` or vectorized `Float64Array`.
 */
export function chebyshevLfo(
  cosPhi: number,
  backend?: BackendChoice,
): [number, number, number, number];
export function chebyshevLfo(
  cosPhi: Float64Array,
  out?: Float64Array,
  backend?: BackendChoice,
): Float64Array;
export function chebyshevLfo(
  cosPhi: number | Float64Array,
  outOrBackend?: Float64Array | BackendChoice,
  maybeBackend?: BackendChoice,
): [number, number, number, number] | Float64Array {
  if (typeof cosPhi === "number") {
    const backend = typeof outOrBackend === "string" ? outOrBackend : undefined;
    const driver = resolveDriver(backend);
    return driver.evalChebyshevLfo(cosPhi);
  }
  const out =
    outOrBackend instanceof Float64Array ? outOrBackend : new Float64Array(cosPhi.length * 4);
  const backend = typeof outOrBackend === "string" ? outOrBackend : maybeBackend;
  const driver = resolveDriver(backend);
  driver.evalChebyshevLfoSlice(cosPhi, out);
  return out;
}

/**
 * Evaluates the $C^1$-smooth cubic trust-region soft-clipper (`CL-15`) for a scalar `number`
 * or vectorized `Float64Array`.
 */
export function trustRegionSoftclip(z: number, mh?: number, backend?: BackendChoice): number;
export function trustRegionSoftclip(
  z: Float64Array,
  mh?: number,
  out?: Float64Array,
  backend?: BackendChoice,
): Float64Array;
export function trustRegionSoftclip(
  z: number | Float64Array,
  mh = 2.5,
  outOrBackend?: Float64Array | BackendChoice,
  maybeBackend?: BackendChoice,
): number | Float64Array {
  if (typeof z === "number") {
    const backend = typeof outOrBackend === "string" ? outOrBackend : undefined;
    const driver = resolveDriver(backend);
    return driver.evalTrustRegionSoftclip(z, mh);
  }
  const out = outOrBackend instanceof Float64Array ? outOrBackend : new Float64Array(z.length);
  const backend = typeof outOrBackend === "string" ? outOrBackend : maybeBackend;
  const driver = resolveDriver(backend);
  driver.evalTrustRegionSoftclipSlice(z, out, mh);
  return out;
}
