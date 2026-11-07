/**
 * Stateful real-time SIMD processor (`Engine<TParams>`) wrapping `DynamicDnamEngine<'static>`
 * (`ref[sdk.ts.api.resources-and-data]`, `ref[sdk.ts.engine.zero-copy-and-async]`,
 * `ref[sdk.ts.engine.streaming-and-render]`).
 *
 * @module
 */

import { DnamError } from "../errors.js";
import {
  coerceAudioChannels,
  coerceBackend,
  coerceTierIndex,
  tierFromIndex,
} from "../internal/coerce.js";
import { registerDisposable } from "../internal/handle.js";
import { type NativeEngineHandle, TELEMETRY_F64_LEN } from "../internal/native/driver.js";
import { getActiveDriver } from "../internal/native/load.js";
import { type Model, registerModelEngineFactory } from "../model/model.js";
import { ControlSpace } from "../parameters/clamp.js";
import type { ParameterBackend } from "../parameters/parameter.js";
import { ParameterMap } from "../parameters/parameter-map.js";
import type { ParameterValueMap } from "../parameters/typed-model.js";
import type { Preset } from "../schema/parameters.js";
import { TempoController } from "../tempo/tempo.js";
import type {
  ActiveBackend,
  AudioInput,
  Backend,
  StateScope,
  Subdivision,
  Tier,
  TierInput,
} from "../types.js";
import { type BypassOptions, createStateScope, type FreezeOptions } from "./state-scopes.js";
import { createEngineTransformStream, type StreamOptions } from "./stream.js";
import { decodeTelemetryF64, type TelemetrySnapshot } from "./telemetry.js";

/**
 * Options for `new Engine(model, options)` and `model.createEngine(options)`.
 */
export interface EngineOptions {
  readonly sampleRate?: number;
  readonly tier?: TierInput | string;
  readonly backend?: Backend | string;
  readonly maxSupportedTier?: TierInput | string;
  readonly offlineBounce?: boolean;
  readonly maximizeTierOnBounce?: boolean;
}

/**
 * Options for `engine.process(input, options)`.
 */
export interface ProcessOptions {
  readonly output?: Float32Array[];
  readonly blockSize?: number;
}

/**
 * Options for `engine.processAsync(input, options)` (`ref[sdk.ts.engine.zero-copy-and-async]`,
 * `ref[sdk.ts.runtime.zero-unsafe]`).
 */
export interface ProcessAsyncOptions {
  readonly output?: Float32Array[];
  readonly blockSize?: number;
  readonly signal?: AbortSignal;
  readonly transfer?: boolean;
}

function detachInputBuffers(
  channels: readonly Float32Array[],
  out: readonly Float32Array[] | undefined,
): void {
  const seen = new Set<ArrayBufferLike>();
  if (out !== undefined) {
    for (const outCh of out) {
      seen.add(outCh.buffer);
    }
  }
  for (const ch of channels) {
    const buf = ch.buffer;
    if (seen.has(buf)) {
      continue;
    }
    seen.add(buf);
    if (buf instanceof ArrayBuffer && !buf.detached && typeof buf.transfer === "function") {
      buf.transfer();
    }
  }
}

/**
 * Options for one-shot `engine.render(input, options)`.
 */
export interface RenderOptions {
  readonly sampleRate?: number;
  readonly blockSize?: number;
  readonly reset?: boolean;
  readonly oversample?: 1 | 4;
  readonly maxTier?: boolean;
  readonly output?: Float32Array[];
}

/**
 * Options for `engine.bounce(input, options)`.
 */
export interface BounceOptions {
  readonly maxTier?: boolean;
  readonly output?: Float32Array[];
}

/**
 * Options for `engine.reload(model, options)`.
 */
export interface ReloadOptions {
  readonly tier?: TierInput | string;
  readonly sampleRate?: number;
}

/**
 * Options for `engine.configureHostMix(options)`.
 */
export interface HostMixOptions {
  readonly blendWet?: number;
  readonly outputGainDb?: number;
  readonly applyUnityLeveling?: boolean;
}

/**
 * Stateful real-time TORA SIMD audio processor (`ref[sdk.ts.api.resources-and-data]`,
 * `ref[sdk.ts.engine.zero-copy-and-async]`).
 *
 * @example
 * ```ts
 * using engine = model.createEngine({ sampleRate: 48_000, tier: "full" });
 * engine.parameters.get("gain").value = 7.5;
 * const [left, right] = engine.process([inL, inR]);
 * ```
 */
export class Engine<TParams extends ParameterValueMap = Record<string, number | string>>
  implements Disposable
{
  #model: Model<TParams>;
  readonly #handle: NativeEngineHandle;
  #controlSpace: ControlSpace<TParams>;
  #parameters: ParameterMap<TParams>;
  readonly #tempo: TempoController;
  readonly #telemetryScratch = new Float64Array(TELEMETRY_F64_LEN);
  #monoScratchLeft = new Float32Array(0);
  #monoScratchRight = new Float32Array(0);
  readonly #release: () => void;
  #disposed = false;

  constructor(model: Model<TParams>, options?: EngineOptions) {
    this.#model = model;
    const tierIdx = options?.tier !== undefined ? coerceTierIndex(options.tier) : 1;
    const maxTierIdx =
      options?.maxSupportedTier !== undefined
        ? coerceTierIndex(options.maxSupportedTier)
        : undefined;
    const sampleRateHz = Math.round(options?.sampleRate ?? 48_000);
    const backendStr = coerceBackend(options?.backend ?? "auto");

    const driver = getActiveDriver();
    const handle = driver.createEngine({
      containerBytes: model.toBytes(),
      tier: tierIdx,
      sampleRateHz,
      backend: backendStr,
      ...(maxTierIdx !== undefined ? { maxSupportedTier: maxTierIdx } : {}),
      ...(options?.offlineBounce !== undefined ? { offlineBounce: options.offlineBounce } : {}),
      ...(options?.maximizeTierOnBounce !== undefined
        ? { maximizeTierOnBounce: options.maximizeTierOnBounce }
        : {}),
    });

    this.#handle = handle;
    this.#controlSpace = ControlSpace.from<TParams>(model);
    this.#parameters = new ParameterMap<TParams>(
      model.parameters,
      this.#createParameterBackend(),
      this.#controlSpace,
    );
    this.#tempo = new TempoController({
      currentBpm: () => this.#ensureHandle().currentBpm(),
      syncTempo: (bpm: number, subdivision: Subdivision | undefined, foldOctave: boolean) =>
        this.#ensureHandle().syncTempo(bpm, subdivision, foldOctave),
      registerTapTempo: (subdivision: Subdivision | undefined, foldOctave: boolean) => {
        const [bpm, updatedParameters] = this.#ensureHandle().registerTapTempo(
          subdivision,
          foldOctave,
        );
        return { bpm, updatedParameters };
      },
    });

    this.#release = registerDisposable(this, () => {
      this.#disposed = true;
      this.#controlSpace.dispose();
      this.#handle.dispose();
    });
  }

  #ensureHandle(): NativeEngineHandle {
    if (this.#disposed) {
      throw new DnamError("ERR_DNAM_DISPOSED", "Engine has been disposed");
    }
    return this.#handle;
  }

  #createParameterBackend(): ParameterBackend {
    return {
      getClampedNumericValue: (index: number) => this.#ensureHandle().clampedControlValue(index),
      getRawNumericValues: () => this.#ensureHandle().rawControlValues(),
      setNumericValue: (index: number, numericValue: number, immediate: boolean) => {
        const h = this.#ensureHandle();
        if (immediate) {
          h.setControlImmediate(index, numericValue);
        } else {
          h.setControlValue(index, numericValue);
        }
      },
      setNumericValuesBatch: (
        indices: readonly number[],
        numericValues: readonly number[],
        immediate: boolean,
      ) => {
        const h = this.#ensureHandle();
        if (immediate) {
          h.setControlsImmediateBatch(indices, numericValues);
        } else {
          for (let i = 0; i < indices.length; i++) {
            const idx = indices[i];
            const val = numericValues[i];
            if (idx !== undefined && val !== undefined) {
              h.setControlValue(idx, val);
            }
          }
        }
      },
      resetAllToDefaults: () => {
        this.#ensureHandle().resetControlsToDefaults();
      },
      getActiveClampMask: () => {
        this.#ensureHandle().telemetryInto(this.#telemetryScratch);
        return (this.#telemetryScratch[10] ?? 0) >>> 0;
      },
      toPhysical: (index: number, normalizedU: number) => {
        const desc = this.#model.parameters[index];
        if (!desc) {
          throw new DnamError("ERR_DNAM_PARAMETER", `Invalid parameter index ${String(index)}`);
        }
        return this.#controlSpace.toPhysical(desc.id as keyof TParams & string, normalizedU);
      },
      toNormalized: (index: number, physicalValue: number) => {
        const desc = this.#model.parameters[index];
        if (!desc) {
          throw new DnamError("ERR_DNAM_PARAMETER", `Invalid parameter index ${String(index)}`);
        }
        return this.#controlSpace.toNormalized(desc.id as keyof TParams & string, physicalValue);
      },
    };
  }

  /**
   * Currently loaded `Model`.
   */
  get model(): Model<TParams> {
    this.#ensureHandle();
    return this.#model;
  }

  /**
   * Live `ParameterMap` collection (`engine.parameters.get(id)`).
   */
  get parameters(): ParameterMap<TParams> {
    this.#ensureHandle();
    return this.#parameters;
  }

  /**
   * Read-only map of factory presets on the loaded model (`model.presets`).
   */
  get presets(): ReadonlyMap<string, Preset> {
    this.#ensureHandle();
    return this.#model.presets;
  }

  /**
   * Live tempo synchronization controller (`engine.tempo.set(...)`, `engine.tempo.tap(...)`).
   */
  get tempo(): TempoController {
    this.#ensureHandle();
    return this.#tempo;
  }

  /**
   * Active hardware SIMD backend (`"scalar" | "neon" | "arm-sme" | "x86-simd" | "wasm-simd128"`).
   */
  get backend(): ActiveBackend {
    return this.#ensureHandle().backend();
  }

  /**
   * Requested tier (`"lite" | "full" | "pro" | "ultra"`).
   */
  get tier(): Tier {
    return tierFromIndex(this.#ensureHandle().tier());
  }

  set tier(value: TierInput | string) {
    this.#ensureHandle().setTier(coerceTierIndex(value));
  }

  /**
   * Host maximum supported tier ceiling (`REQ-5.5`).
   */
  get maxSupportedTier(): Tier {
    return tierFromIndex(this.#ensureHandle().maxSupportedTier());
  }

  set maxSupportedTier(value: TierInput | string) {
    this.#ensureHandle().setMaxSupportedTier(coerceTierIndex(value));
  }

  /**
   * Maximum tier present in the loaded `.dnam` container.
   */
  get maxAvailableTier(): Tier {
    return tierFromIndex(this.#ensureHandle().maxAvailableTier());
  }

  /**
   * Effective tier currently running (`min(requested, maxSupported, maxAvailable)`, or promoted on bounce).
   */
  get effectiveTier(): Tier {
    return tierFromIndex(this.#ensureHandle().effectiveTier());
  }

  /**
   * Host sample rate in Hz (`44,100..192,000`).
   */
  get sampleRate(): number {
    return this.#ensureHandle().sampleRate();
  }

  set sampleRate(value: number) {
    this.#ensureHandle().setSampleRate(Math.round(value));
  }

  /**
   * Algorithmic lookahead latency in samples (`0` in real-time mode, `12` in `4x` bounce mode).
   */
  get latencySamples(): number {
    return this.#ensureHandle().latencySamples();
  }

  /**
   * Whether `4x` polyphase offline bounce mode is enabled.
   */
  get offlineBounce(): boolean {
    return this.#ensureHandle().offlineBounce();
  }

  set offlineBounce(enabled: boolean) {
    this.#ensureHandle().setOfflineBounce(enabled);
  }

  /**
   * Whether offline bounce automatically promotes to `maxAvailableTier`.
   */
  get maximizeTierOnBounce(): boolean {
    return this.#ensureHandle().maximizeTierOnBounce();
  }

  set maximizeTierOnBounce(enabled: boolean) {
    this.#ensureHandle().setMaximizeTierOnBounce(enabled);
  }

  /**
   * Whether the loaded model supports Stage IV captured hardware freeze.
   */
  get supportsFreeze(): boolean {
    return this.#ensureHandle().supportsFreeze();
  }

  /**
   * Whether Stage IV hardware freeze is currently engaged.
   */
  get isFrozen(): boolean {
    return this.#ensureHandle().isFrozen();
  }

  /**
   * Whether the engine is currently bypassed.
   */
  get isBypassed(): boolean {
    return this.#ensureHandle().bypassState()[0];
  }

  /**
   * Whether spillover trails are enabled during bypass.
   */
  get trailsEnabled(): boolean {
    return this.#ensureHandle().bypassState()[1];
  }

  /**
   * Continuous stereo width multiplier (`0.0 = Mono` to `1.0 = Full Stereo`).
   */
  get stereoWidth(): number {
    return this.#ensureHandle().stereoWidth();
  }

  set stereoWidth(width: number) {
    this.#ensureHandle().setStereoWidth(width);
  }

  /**
   * Number of input channels declared by the model (`1` or `2`).
   */
  get inputChannels(): 1 | 2 {
    return (this.#model.header.ioConfig.inputs === 1 ? 1 : 2) as 1 | 2;
  }

  /**
   * Number of output channels declared by the model (`1` for `"mono"`, `2` for `"stereo"`).
   */
  get outputChannels(): 1 | 2 {
    return this.#model.header.ioConfig.outputMode === "mono" ? 1 : 2;
  }

  /**
   * Applies a factory preset by ID (`ref[dsp.ctrl.switch.conditional-batch]`).
   */
  applyPreset(presetId: string): this {
    this.#ensureHandle().applyPreset(presetId);
    return this;
  }

  /**
   * Configures host dry/wet parallel blend, output gain trim (dB), and auto-unity leveling.
   */
  configureHostMix(options: HostMixOptions): this {
    const leveling = this.#model.header.ioConfig.leveling;
    const cap = this.#model.header.ioConfig.capabilities;
    this.#ensureHandle().configureHostMix(
      options.blendWet ?? cap.defaultBlend,
      options.outputGainDb ?? leveling.targetGainDb,
      options.applyUnityLeveling ?? leveling.autoUnityEnabled,
    );
    return this;
  }

  /**
   * Sets the engine bypass state directly (`setBypass(true, { trails: true })`).
   */
  setBypass(bypassed: boolean, options?: BypassOptions): this {
    this.#ensureHandle().setBypass(bypassed, options?.trails ?? true);
    return this;
  }

  /**
   * Enters bypass and returns a `StateScope` that restores the prior `(bypassed, trails)` state
   * when `.restore()` or `using` (`[Symbol.dispose]()`) completes (`ref[sdk.ts.api.disposable-scopes]`).
   */
  bypass(options?: BypassOptions): StateScope {
    const h = this.#ensureHandle();
    const [prevBypassed, prevTrails] = h.bypassState();
    h.setBypass(true, options?.trails ?? true);
    return createStateScope(() => {
      if (!this.#disposed) {
        this.#handle.setBypass(prevBypassed, prevTrails);
      }
    });
  }

  /**
   * Executes `fn` while bypassed and restores the prior bypass state in a `finally` block.
   */
  withBypass<T>(optionsOrFn: BypassOptions | (() => T), maybeFn?: () => T): T {
    const options = typeof optionsOrFn === "function" ? undefined : optionsOrFn;
    const fn = typeof optionsOrFn === "function" ? optionsOrFn : maybeFn;
    if (!fn) {
      throw new DnamError("ERR_DNAM_VALIDATION", "withBypass requires a callback function");
    }
    const scope = this.bypass(options);
    try {
      return fn();
    } finally {
      scope.restore();
    }
  }

  /**
   * Sets the Stage IV captured hardware freeze state directly.
   */
  setFreeze(frozen: boolean, options?: FreezeOptions): this {
    const h = this.#ensureHandle();
    if (frozen && !h.supportsFreeze()) {
      throw new DnamError(
        "ERR_DNAM_PARAMETER",
        `Model '${this.#model.name}' does not support Stage IV hardware freeze`,
      );
    }
    if (options?.amount !== undefined) {
      h.setFreezeAmount(options.amount);
    }
    h.setFreeze(frozen);
    return this;
  }

  /**
   * Engages Stage IV captured hardware freeze and returns a `StateScope` that restores the prior
   * freeze state when `.restore()` or `using` (`[Symbol.dispose]()`) completes
   * (`ref[sdk.ts.api.disposable-scopes]`).
   */
  freeze(options?: FreezeOptions): StateScope {
    const h = this.#ensureHandle();
    if (!h.supportsFreeze()) {
      throw new DnamError(
        "ERR_DNAM_PARAMETER",
        `Model '${this.#model.name}' does not support Stage IV hardware freeze`,
      );
    }
    const prevFrozen = h.isFrozen();
    h.setFreezeAmount(options?.amount ?? 1.0);
    h.setFreeze(true);
    return createStateScope(() => {
      if (!this.#disposed) {
        this.#handle.setFreeze(prevFrozen);
      }
    });
  }

  /**
   * Executes `fn` while Stage IV hardware freeze is engaged and restores the prior freeze state
   * in a `finally` block.
   */
  withFreeze<T>(optionsOrFn: FreezeOptions | (() => T), maybeFn?: () => T): T {
    const options = typeof optionsOrFn === "function" ? undefined : optionsOrFn;
    const fn = typeof optionsOrFn === "function" ? optionsOrFn : maybeFn;
    if (!fn) {
      throw new DnamError("ERR_DNAM_VALIDATION", "withFreeze requires a callback function");
    }
    const scope = this.freeze(options);
    try {
      return fn();
    } finally {
      scope.restore();
    }
  }

  /**
   * Hot-swaps a new `Model` in-place (`reload_model_in_place_owned`).
   */
  reload(model: Model<TParams>, options?: ReloadOptions): this {
    const h = this.#ensureHandle();
    const tierIdx = options?.tier !== undefined ? coerceTierIndex(options.tier) : undefined;
    const sr = options?.sampleRate !== undefined ? Math.round(options.sampleRate) : undefined;
    h.reload(model.toBytes(), tierIdx, sr);
    this.#controlSpace.dispose();
    this.#model = model;
    this.#controlSpace = ControlSpace.from<TParams>(model);
    this.#parameters = new ParameterMap<TParams>(
      model.parameters,
      this.#createParameterBackend(),
      this.#controlSpace,
    );
    return this;
  }

  /**
   * Resets all filter, delay, and operator states to analytical resting DC equilibrium
   * (`ref[dsp.op-n.commuted.dc-equilibrium]`).
   */
  reset(): this {
    this.#ensureHandle().reset();
    return this;
  }

  /**
   * Processes an `AudioInput` (`Float32Array` or `[left, right]`) synchronously across all four
   * model topologies (`1->1`, `2->1`, `1->2`, `2->2`, `ref[sdk.ts.engine.zero-copy-and-async]`).
   *
   * When `options.output` is provided with matching channel count and frame length, writes
   * directly into `options.output` with zero output allocation.
   */
  process(input: AudioInput, options?: ProcessOptions): Float32Array[] {
    const h = this.#ensureHandle();
    const { channels, frames } = coerceAudioChannels(input);
    const outChannelsCount = this.outputChannels;
    const blockSize = options?.blockSize;

    let out: Float32Array[];
    if (options?.output !== undefined) {
      if (options.output.length !== outChannelsCount) {
        throw new DnamError(
          "ERR_DNAM_VALIDATION",
          `options.output must have ${String(outChannelsCount)} channel(s), got ${String(options.output.length)}`,
        );
      }
      for (let c = 0; c < outChannelsCount; c++) {
        if (options.output[c]?.length !== frames) {
          throw new DnamError(
            "ERR_DNAM_VALIDATION",
            `options.output[${String(c)}] length (${String(options.output[c]?.length)}) must match input frames (${String(frames)})`,
          );
        }
      }
      out = options.output;
    } else {
      out =
        outChannelsCount === 2
          ? [new Float32Array(frames), new Float32Array(frames)]
          : [new Float32Array(frames)];
    }

    if (frames === 0) {
      return out;
    }

    const ch0In = channels[0] as Float32Array;
    if (outChannelsCount === 2) {
      const leftOut = out[0] as Float32Array;
      const rightOut = out[1] as Float32Array;
      if (channels.length === 2) {
        const ch1In = channels[1] as Float32Array;
        if (leftOut === ch0In && rightOut === ch1In && blockSize === undefined) {
          h.processStereoInPlace(leftOut, rightOut);
        } else {
          h.processStereo(ch0In, ch1In, leftOut, rightOut, blockSize);
        }
      } else {
        h.processMonoToStereo(ch0In, leftOut, rightOut, blockSize);
      }
      return out;
    }

    // Mono output (`outChannelsCount === 1`)
    const monoOut = out[0] as Float32Array;
    if (channels.length === 1) {
      h.processMono(ch0In, monoOut, blockSize);
      return out;
    }

    // `2 -> 1` topology: process stereo into reusable scratch buffers and average to mono
    const ch1In = channels[1] as Float32Array;
    if (this.#monoScratchLeft.length < frames) {
      this.#monoScratchLeft = new Float32Array(frames);
      this.#monoScratchRight = new Float32Array(frames);
    }
    const scratchL = this.#monoScratchLeft.subarray(0, frames);
    const scratchR = this.#monoScratchRight.subarray(0, frames);
    h.processStereo(ch0In, ch1In, scratchL, scratchR, blockSize);
    for (let i = 0; i < frames; i++) {
      monoOut[i] = 0.5 * ((scratchL[i] ?? 0) + (scratchR[i] ?? 0));
    }
    return out;
  }

  /**
   * Processes an `AudioInput` asynchronously on the Node.js libuv worker pool (or microtask in
   * browsers), honoring `options.signal` (`AbortSignal`), optional caller-supplied `out` buffers,
   * and `{ transfer: true }` input `ArrayBuffer` ownership transfer, while rejecting concurrent
   * audio processing calls with `DnamError` (`"ERR_DNAM_BUSY"`,
   * `ref[sdk.ts.engine.zero-copy-and-async]`, `ref[sdk.ts.runtime.zero-unsafe]`).
   */
  processAsync(input: AudioInput, out?: Float32Array[]): Promise<Float32Array[]>;
  processAsync(input: AudioInput, options?: ProcessAsyncOptions): Promise<Float32Array[]>;
  async processAsync(
    input: AudioInput,
    optionsOrOut?: ProcessAsyncOptions | Float32Array[],
  ): Promise<Float32Array[]> {
    const h = this.#ensureHandle();
    const isPositionalOut = Array.isArray(optionsOrOut);
    const options = isPositionalOut ? undefined : optionsOrOut;
    const out = isPositionalOut ? optionsOrOut : options?.output;
    if (options?.signal?.aborted) {
      throw options.signal.reason ?? new DOMException("The operation was aborted.", "AbortError");
    }
    const { channels, frames } = coerceAudioChannels(input);
    const outChannelsCount = this.outputChannels;
    if (out !== undefined) {
      if (out.length !== outChannelsCount) {
        throw new DnamError(
          "ERR_DNAM_VALIDATION",
          `output must have ${String(outChannelsCount)} channel(s), got ${String(out.length)}`,
        );
      }
      for (let c = 0; c < outChannelsCount; c++) {
        if (out[c]?.length !== frames) {
          throw new DnamError(
            "ERR_DNAM_VALIDATION",
            `output[${String(c)}] length (${String(out[c]?.length)}) must match input frames (${String(frames)})`,
          );
        }
      }
    }
    const leftIn = channels[0] as Float32Array;
    const rightIn = (channels[1] ?? channels[0]) as Float32Array;
    const pending = h.processStereoAsync(leftIn, rightIn, options?.blockSize);
    if (options?.transfer) {
      detachInputBuffers(channels, out);
    }
    const [leftOut, rightOut] = await pending;

    if (outChannelsCount === 1) {
      const monoOut = out !== undefined ? (out[0] as Float32Array) : leftOut;
      if (channels.length === 2) {
        for (let i = 0; i < frames; i++) {
          monoOut[i] = 0.5 * ((leftOut[i] ?? 0) + (rightOut[i] ?? 0));
        }
      } else if (out !== undefined) {
        monoOut.set(leftOut);
      }
      return out !== undefined ? out : [monoOut];
    }
    if (out !== undefined) {
      out[0]?.set(leftOut);
      out[1]?.set(rightOut);
      return out;
    }
    return [leftOut, rightOut];
  }

  /**
   * One-shot offline audio rendering with optional `sampleRate` update, DC equilibrium `reset`
   * (`default = true`), chunked `blockSize`, and `4x` linear-phase polyphase oversampling
   * (`oversample: 4`, `ref[sdk.ts.engine.streaming-and-render]`).
   */
  render(input: AudioInput, options?: RenderOptions): Float32Array[] {
    const h = this.#ensureHandle();
    if (options?.sampleRate !== undefined && Math.round(options.sampleRate) !== h.sampleRate()) {
      h.setSampleRate(Math.round(options.sampleRate));
    }
    if (options?.reset ?? true) {
      h.reset();
    }
    if (options?.oversample === 4) {
      return this.bounce(input, {
        ...(options.maxTier !== undefined ? { maxTier: options.maxTier } : {}),
        ...(options.output !== undefined ? { output: options.output } : {}),
      });
    }
    return this.process(input, {
      blockSize: options?.blockSize ?? 8192,
      ...(options?.output !== undefined ? { output: options.output } : {}),
    });
  }

  /**
   * Executes `4x` linear-phase polyphase oversampled offline bounce with automatic `12`-sample
   * PDC lead-in stripping (`ref[arch.bounce.oversampling]`, `ref[sdk.ts.engine.streaming-and-render]`).
   */
  bounce(input: AudioInput, options?: BounceOptions): Float32Array[] {
    const h = this.#ensureHandle();
    const { channels, frames } = coerceAudioChannels(input);
    const prevMaxTier = h.maximizeTierOnBounce();
    if (options?.maxTier !== undefined) {
      h.setMaximizeTierOnBounce(options.maxTier);
    }
    try {
      const leftIn = channels[0] as Float32Array;
      const rightIn = (channels[1] ?? channels[0]) as Float32Array;
      const leftOut = options?.output?.[0] ?? new Float32Array(frames);
      const rightOut = options?.output?.[1] ?? new Float32Array(frames);
      h.renderOfflineBounce(leftIn, rightIn, leftOut, rightOut);
      if (this.outputChannels === 1) {
        return [leftOut];
      }
      return [leftOut, rightOut];
    } finally {
      if (options?.maxTier !== undefined && !this.#disposed) {
        h.setMaximizeTierOnBounce(prevMaxTier);
      }
    }
  }

  /**
   * Returns a WHATWG `TransformStream<AudioInput, Float32Array[]>` for streaming audio through
   * this engine (`ref[sdk.ts.engine.streaming-and-render]`).
   */
  stream(options?: StreamOptions): TransformStream<AudioInput, Float32Array[]> {
    this.#ensureHandle();
    return createEngineTransformStream(
      (chunk, blockSize) =>
        this.process(chunk, blockSize !== undefined ? { blockSize } : undefined),
      options,
    );
  }

  /**
   * Captures a frozen `TelemetrySnapshot` of the engine's real-time state, ZOH parameters,
   * and saturation/overflow flags (`ref[sdk.ts.api.resources-and-data]`).
   */
  telemetry(): TelemetrySnapshot {
    const h = this.#ensureHandle();
    h.telemetryInto(this.#telemetryScratch);
    return decodeTelemetryF64(this.#telemetryScratch, this.#model.parameters.length);
  }

  /**
   * Releases the engine's native resources idempotently.
   */
  dispose(): void {
    this.#release();
  }

  /**
   * Disposes the engine (`using` protocol).
   */
  [Symbol.dispose](): void {
    this.dispose();
  }
}

registerModelEngineFactory((model, options) => new Engine(model, options));
