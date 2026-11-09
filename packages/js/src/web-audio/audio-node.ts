/**
 * Main-thread WebAudio `DnamAudioNode<TParams>` controller exposing the exact same `ParameterMap`
 * (`node.parameters.get(id)`), `TempoController`, `bypass`/`freeze` scopes, and typed
 * `"telemetry"` events as `Engine<TParams>` (`ref[sdk.ts.web-audio.realtime-bridge]`).
 *
 * @module
 */

import {
  type BypassOptions,
  createStateScope,
  type FreezeOptions,
} from "../engine/state-scopes.js";
import { decodeTelemetryF64, type TelemetrySnapshot } from "../engine/telemetry.js";
import { DnamError } from "../errors.js";
import {
  coerceBackend,
  coerceSubdivision,
  coerceTierIndex,
  tierFromIndex,
} from "../internal/coerce.js";
import { TELEMETRY_F64_LEN } from "../internal/native/driver.js";
import {
  compileWasmModule,
  createWasmMemory,
  isSharedWasmMemory,
  WASM_MIN_INITIAL_PAGES,
} from "../internal/native/wasm.js";
import type { Model } from "../model/model.js";
import { ControlSpace, defaultNumericValue } from "../parameters/clamp.js";
import type { ParameterBackend } from "../parameters/parameter.js";
import { ParameterMap } from "../parameters/parameter-map.js";
import type { ParameterValueMap } from "../parameters/typed-model.js";
import type { Preset } from "../schema/parameters.js";
import { TempoController } from "../tempo/tempo.js";
import type { Backend, StateScope, Subdivision, Tier, TierInput } from "../types.js";
import { DNAM_WORKLET_PROCESSOR_NAME } from "./processor.js";
import type {
  DnamWorkletProcessorOptions,
  WorkletInboundMessage,
  WorkletOutboundMessage,
} from "./protocol.js";
import {
  CMD_BYPASS_CHANGE,
  CMD_BYPASSED,
  CMD_FREEZE_CHANGE,
  CMD_FROZEN,
  CMD_RESET,
  CMD_RESET_CONTROLS,
  CMD_TEMPO_CHANGE,
  CMD_TIER_CHANGE,
  CMD_TRAILS,
  CMD_WIDTH_CHANGE,
  isSharedMemorySupported,
  WorkletSharedState,
} from "./shared-state.js";

/**
 * Minimal WebAudio `BaseAudioContext` structural interface so `DnamAudioNode` works with both
 * browser `AudioContext` / `OfflineAudioContext` and headless test mocks.
 */
export interface AudioContextLike {
  readonly sampleRate: number;
  readonly audioWorklet?: {
    addModule(moduleUrl: string | URL): Promise<void>;
  };
}

/**
 * Options for `DnamAudioNode.create(context, model, options)`.
 */
export interface DnamAudioNodeOptions {
  readonly tier?: TierInput | string;
  readonly maxSupportedTier?: TierInput | string;
  readonly backend?: Backend | string;
  readonly processorUrl?: string | URL;
  readonly wasmModule?: WebAssembly.Module;
  readonly wasmMemory?: WebAssembly.Memory;
  readonly initialMemoryPages?: number;
  readonly maximumMemoryPages?: number;
  readonly sharedWasmMemory?: boolean;
  readonly useSharedMemory?: boolean;
  readonly telemetryIntervalBlocks?: number;
}

function resolveDefaultProcessorUrl(): URL {
  const metaUrl = import.meta.url;
  if (metaUrl.includes("/src/web-audio/")) {
    return new URL(/* @vite-ignore */ "../../dist/web-audio-processor.js", metaUrl);
  }
  if (metaUrl.includes("/web-audio/")) {
    return new URL(/* @vite-ignore */ "../web-audio-processor.js", metaUrl);
  }
  return new URL(/* @vite-ignore */ "./web-audio-processor.js", metaUrl);
}

/**
 * Registers the standalone `web-audio-processor.js` module on `context.audioWorklet`.
 */
export async function registerDnamProcessor(
  context: AudioContextLike,
  processorUrl?: string | URL,
): Promise<void> {
  if (!context.audioWorklet) {
    return;
  }
  const url = processorUrl ?? resolveDefaultProcessorUrl();
  await context.audioWorklet.addModule(url);
}

/**
 * WebAudio `AudioWorklet` controller node (`ref[sdk.ts.web-audio.realtime-bridge]`).
 *
 * Uses a lock-free `SharedArrayBuffer` (`Atomics`) bridge when `crossOriginIsolated === true`
 * and transparently falls back to `MessagePort` when `crossOriginIsolated === false`.
 *
 * @example
 * ```ts
 * const node = await DnamAudioNode.create(audioContext, model, { tier: "lite" });
 * node.parameters.get("gain").value = 6.5;
 * node.addEventListener("telemetry", (e) => console.log(e.detail.maxOpNDrive));
 * ```
 */
export class DnamAudioNode<TParams extends ParameterValueMap = Record<string, number | string>>
  extends EventTarget
  implements Disposable
{
  #model: Model<TParams>;
  readonly #context: AudioContextLike;
  readonly #workletNode: AudioWorkletNode | undefined;
  readonly #wasmMemory: WebAssembly.Memory | undefined;
  readonly #port: {
    postMessage(msg: WorkletInboundMessage): void;
    onmessage: ((ev: { readonly data: WorkletOutboundMessage }) => void) | null;
  };
  readonly #shared: WorkletSharedState | undefined;
  #controlSpace: ControlSpace<TParams>;
  #parameters: ParameterMap<TParams>;
  readonly #tempo: TempoController;
  readonly #rawParamValues: Float32Array;
  readonly #fallbackParamValues: Float32Array;
  readonly #telemetryScratch = new Float64Array(TELEMETRY_F64_LEN);
  #tierIdx: number;
  #maxSupportedTierIdx: number;
  #bypassed = false;
  #trails = true;
  #frozen = false;
  #stereoWidth = 1.0;
  #bpm = 120.0;
  #disposed = false;

  private constructor(
    context: AudioContextLike,
    model: Model<TParams>,
    wasmModule: WebAssembly.Module,
    options?: DnamAudioNodeOptions,
  ) {
    super();
    this.#context = context;
    this.#model = model;
    this.#tierIdx = options?.tier !== undefined ? coerceTierIndex(options.tier) : 1;
    this.#maxSupportedTierIdx =
      options?.maxSupportedTier !== undefined ? coerceTierIndex(options.maxSupportedTier) : 3;

    const useSab = (options?.useSharedMemory ?? true) && isSharedMemorySupported();
    const defaultNumerics = new Float32Array(
      model.parameters.map((desc) => defaultNumericValue(desc)),
    );
    this.#rawParamValues = defaultNumerics.slice();
    this.#fallbackParamValues = defaultNumerics.slice();

    if (useSab) {
      this.#shared = new WorkletSharedState();
      this.#shared.initDefaults(defaultNumerics, this.#tierIdx, context.sampleRate);
    }

    const wantSharedWasm = (options?.sharedWasmMemory ?? false) && isSharedMemorySupported();
    if (options?.wasmMemory) {
      this.#wasmMemory = options.wasmMemory;
    } else if (wantSharedWasm) {
      this.#wasmMemory = createWasmMemory({
        ...(options?.initialMemoryPages !== undefined
          ? { initialPages: options.initialMemoryPages }
          : {}),
        ...(options?.maximumMemoryPages !== undefined
          ? { maximumPages: options.maximumMemoryPages }
          : {}),
        shared: true,
      });
    }

    this.#telemetryScratch[0] = this.#tierIdx;
    this.#telemetryScratch[1] = 4; // wasm-simd128
    this.#telemetryScratch[2] = context.sampleRate;
    this.#telemetryScratch[3] = model.header.ioConfig.outputMode === "stereo" ? 1 : 0;
    this.#telemetryScratch[7] = model.header.ioConfig.capabilities.supportsFreeze ? 1 : 0;
    this.#telemetryScratch[8] = 1.0;
    this.#telemetryScratch[9] = 120.0;
    for (let i = 0; i < Math.min(16, defaultNumerics.length); i++) {
      this.#telemetryScratch[32 + i] = defaultNumerics[i] ?? 0;
    }

    const isCloneableSharedWasmMem =
      this.#wasmMemory !== undefined && isSharedWasmMemory(this.#wasmMemory);
    const derivedInitialPages =
      options?.initialMemoryPages ??
      (this.#wasmMemory !== undefined
        ? Math.max(WASM_MIN_INITIAL_PAGES, Math.ceil(this.#wasmMemory.buffer.byteLength / 65536))
        : undefined);

    const processorOptions: DnamWorkletProcessorOptions = {
      wasmModule,
      containerBytes: model.toBytes(),
      tier: this.#tierIdx,
      sampleRate: context.sampleRate,
      backend: coerceBackend(options?.backend ?? "auto"),
      maxSupportedTier: this.#maxSupportedTierIdx,
      ...(this.#shared ? { sharedBuffer: this.#shared.buffer } : {}),
      ...(isCloneableSharedWasmMem && this.#wasmMemory ? { wasmMemory: this.#wasmMemory } : {}),
      ...(derivedInitialPages !== undefined ? { initialMemoryPages: derivedInitialPages } : {}),
      ...(options?.maximumMemoryPages !== undefined
        ? { maximumMemoryPages: options.maximumMemoryPages }
        : {}),
      ...(wantSharedWasm || isCloneableSharedWasmMem ? { sharedWasmMemory: true } : {}),
      ...(options?.telemetryIntervalBlocks !== undefined
        ? { telemetryIntervalBlocks: options.telemetryIntervalBlocks }
        : {}),
    };

    if (typeof AudioWorkletNode !== "undefined" && context instanceof BaseAudioContext) {
      this.#workletNode = new AudioWorkletNode(context, DNAM_WORKLET_PROCESSOR_NAME, {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [model.header.ioConfig.outputMode === "mono" ? 1 : 2],
        processorOptions,
      });
      this.#port = this.#workletNode.port as unknown as {
        postMessage(msg: WorkletInboundMessage): void;
        onmessage: ((ev: { readonly data: WorkletOutboundMessage }) => void) | null;
      };
    } else {
      this.#port = {
        onmessage: null,
        postMessage: () => {},
      };
    }

    this.#port.onmessage = (event: { readonly data: WorkletOutboundMessage }) => {
      const msg = event.data;
      if (msg.type === "telemetry") {
        this.#telemetryScratch.set(msg.data.subarray(0, TELEMETRY_F64_LEN));
        for (let i = 0; i < Math.min(16, this.#fallbackParamValues.length); i++) {
          this.#fallbackParamValues[i] = this.#telemetryScratch[32 + i] ?? 0;
        }
        const snap = this.telemetry();
        this.dispatchEvent(new CustomEvent<TelemetrySnapshot>("telemetry", { detail: snap }));
      }
    };

    this.#controlSpace = ControlSpace.from<TParams>(model);
    this.#parameters = new ParameterMap<TParams>(
      model.parameters,
      this.#createParameterBackend(),
      this.#controlSpace,
    );
    this.#tempo = new TempoController({
      currentBpm: () => this.#bpm,
      syncTempo: (bpm: number, subdivision: Subdivision | undefined, foldOctave: boolean) => {
        this.#bpm = Math.max(20, Math.min(400, bpm));
        if (this.#shared && subdivision === undefined && !foldOctave) {
          this.#shared.writeCommand(CMD_TEMPO_CHANGE, 0, { bpm: this.#bpm });
        } else {
          this.#port.postMessage({
            type: "sync-tempo",
            bpm: this.#bpm,
            ...(subdivision !== undefined ? { subdivision: coerceSubdivision(subdivision) } : {}),
            foldOctave,
          });
        }
        return 1;
      },
      registerTapTempo: (subdivision: Subdivision | undefined, foldOctave: boolean) => {
        this.#port.postMessage({
          type: "tap-tempo",
          ...(subdivision !== undefined ? { subdivision: coerceSubdivision(subdivision) } : {}),
          foldOctave,
        });
        return { bpm: this.#bpm, updatedParameters: 1 };
      },
    });
  }

  /**
   * Registers the worklet processor (when `context.audioWorklet` is present), compiles the WASM
   * module if needed, and creates a `DnamAudioNode<TParams>`.
   */
  static async create<TParams extends ParameterValueMap = Record<string, number | string>>(
    context: AudioContextLike,
    model: Model<TParams>,
    options?: DnamAudioNodeOptions,
  ): Promise<DnamAudioNode<TParams>> {
    await registerDnamProcessor(context, options?.processorUrl);
    const wantSharedWasm =
      options?.wasmMemory !== undefined
        ? isSharedWasmMemory(options.wasmMemory)
        : (options?.sharedWasmMemory ?? false) && isSharedMemorySupported();
    const wasmModule = options?.wasmModule ?? (await compileWasmModule({ shared: wantSharedWasm }));
    return new DnamAudioNode<TParams>(context, model, wasmModule, options);
  }

  #ensureAlive(): void {
    if (this.#disposed) {
      throw new DnamError("ERR_DNAM_DISPOSED", "DnamAudioNode has been disposed");
    }
  }

  #createParameterBackend(): ParameterBackend {
    return {
      getClampedNumericValue: (index: number) => {
        this.#ensureAlive();
        if (this.#shared) {
          return this.#shared.readParameter(index);
        }
        return this.#fallbackParamValues[index] ?? 0;
      },
      getRawNumericValues: () => {
        this.#ensureAlive();
        return this.#rawParamValues.slice();
      },
      setNumericValue: (index: number, numericValue: number, immediate: boolean) => {
        this.#ensureAlive();
        this.#rawParamValues[index] = numericValue;
        const clampedOut = new Float32Array(this.#rawParamValues.length);
        const mask = this.#controlSpace.clampNumericInto(this.#rawParamValues, clampedOut);
        this.#fallbackParamValues.set(clampedOut);
        const clampedNumeric = Number(clampedOut[index]);
        this.#telemetryScratch[10] = mask;
        if (this.#shared) {
          this.#shared.writeParameter(index, clampedNumeric, immediate);
        } else {
          this.#port.postMessage({
            type: "set-param",
            index,
            value: clampedNumeric,
            immediate,
          });
        }
      },
      setNumericValuesBatch: (
        indices: readonly number[],
        numericValues: readonly number[],
        immediate: boolean,
      ) => {
        this.#ensureAlive();
        for (let i = 0; i < indices.length; i++) {
          const idx = indices[i];
          const val = numericValues[i];
          if (idx !== undefined && val !== undefined) {
            this.#rawParamValues[idx] = val;
          }
        }
        const clampedOut = new Float32Array(this.#rawParamValues.length);
        const mask = this.#controlSpace.clampNumericInto(this.#rawParamValues, clampedOut);
        this.#fallbackParamValues.set(clampedOut);
        this.#telemetryScratch[10] = mask;
        const clampedVals = indices.map((idx) => Number(this.#fallbackParamValues[idx]));
        if (this.#shared) {
          this.#shared.writeParametersBatch(indices, clampedVals, immediate);
        } else {
          this.#port.postMessage({
            type: "set-params-batch",
            indices,
            values: clampedVals,
            immediate,
          });
        }
      },
      resetAllToDefaults: () => {
        this.#ensureAlive();
        for (let i = 0; i < this.#model.parameters.length; i++) {
          const desc = this.#model.parameters[i];
          if (desc) {
            const defVal = defaultNumericValue(desc);
            this.#rawParamValues[i] = defVal;
            this.#fallbackParamValues[i] = defVal;
          }
        }
        this.#telemetryScratch[10] = 0;
        if (this.#shared) {
          this.#shared.writeCommand(CMD_RESET_CONTROLS);
        } else {
          this.#port.postMessage({ type: "reset-controls" });
        }
      },
      getActiveClampMask: () => {
        return (
          ((this.#shared ? this.#shared.readActiveClampMask() : 0) |
            ((this.#telemetryScratch[10] ?? 0) >>> 0)) >>>
          0
        );
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
   * Underlying WebAudio `AudioWorkletNode` (when created with a browser `BaseAudioContext`).
   */
  get workletNode(): AudioWorkletNode | undefined {
    return this.#workletNode;
  }

  /**
   * Shared or pre-allocated `WebAssembly.Memory` passed to the worklet processor, when configured.
   */
  get wasmMemory(): WebAssembly.Memory | undefined {
    return this.#wasmMemory;
  }

  /**
   * Whether this node is communicating with the audio thread via lock-free `SharedArrayBuffer` (`Atomics`).
   */
  get usesSharedMemory(): boolean {
    return this.#shared !== undefined;
  }

  /**
   * Loaded `Model`.
   */
  get model(): Model<TParams> {
    this.#ensureAlive();
    return this.#model;
  }

  /**
   * Live `ParameterMap` collection (`node.parameters.get(id)`).
   */
  get parameters(): ParameterMap<TParams> {
    this.#ensureAlive();
    return this.#parameters;
  }

  /**
   * Read-only map of factory presets on the loaded model (`model.presets`).
   */
  get presets(): ReadonlyMap<string, Preset> {
    this.#ensureAlive();
    return this.#model.presets;
  }

  /**
   * Live tempo synchronization controller (`node.tempo`).
   */
  get tempo(): TempoController {
    this.#ensureAlive();
    return this.#tempo;
  }

  /**
   * Requested tier (`"lite" | "full" | "pro" | "ultra"`).
   */
  get tier(): Tier {
    return tierFromIndex(this.#tierIdx);
  }

  set tier(value: TierInput | string) {
    this.#ensureAlive();
    this.#tierIdx = coerceTierIndex(value);
    if (this.#shared) {
      this.#shared.writeCommand(CMD_TIER_CHANGE, 0, { tier: this.#tierIdx });
    } else {
      this.#port.postMessage({ type: "set-tier", tier: this.#tierIdx });
    }
  }

  /**
   * Stereo width multiplier (`0.0 = Mono` to `1.0 = Full Stereo`).
   */
  get stereoWidth(): number {
    return this.#stereoWidth;
  }

  set stereoWidth(width: number) {
    this.#ensureAlive();
    this.#stereoWidth = Math.max(0, Math.min(1, width));
    if (this.#shared) {
      this.#shared.writeCommand(CMD_WIDTH_CHANGE, 0, { stereoWidth: this.#stereoWidth });
    } else {
      this.#port.postMessage({ type: "set-width", width: this.#stereoWidth });
    }
  }

  /**
   * Whether the node is currently bypassed.
   */
  get isBypassed(): boolean {
    return this.#bypassed;
  }

  /**
   * Whether Stage IV hardware freeze is currently engaged.
   */
  get isFrozen(): boolean {
    return this.#frozen;
  }

  /**
   * Applies a factory preset by ID.
   */
  applyPreset(presetId: string): this {
    this.#ensureAlive();
    const preset = this.#model.presets.get(presetId);
    if (!preset) {
      throw new DnamError("ERR_DNAM_PARAMETER", `Unknown preset '${presetId}'`);
    }
    this.#parameters.set(preset.controls as Partial<TParams>, { immediate: true });
    this.#port.postMessage({ type: "apply-preset", presetId });
    return this;
  }

  /**
   * Sets the bypass state directly.
   */
  setBypass(bypassed: boolean, options?: BypassOptions): this {
    this.#ensureAlive();
    this.#bypassed = bypassed;
    this.#trails = options?.trails ?? true;
    if (this.#shared) {
      const setBits =
        CMD_BYPASS_CHANGE | (this.#bypassed ? CMD_BYPASSED : 0) | (this.#trails ? CMD_TRAILS : 0);
      const clearBits = (!this.#bypassed ? CMD_BYPASSED : 0) | (!this.#trails ? CMD_TRAILS : 0);
      this.#shared.writeCommand(setBits, clearBits);
    } else {
      this.#port.postMessage({
        type: "set-bypass",
        bypassed: this.#bypassed,
        trails: this.#trails,
      });
    }
    return this;
  }

  /**
   * Enters bypass and returns a `StateScope` (`restore()` + `[Symbol.dispose]()`).
   */
  bypass(options?: BypassOptions): StateScope {
    const prevBypassed = this.#bypassed;
    const prevTrails = this.#trails;
    this.setBypass(true, options);
    return createStateScope(() => {
      if (!this.#disposed) {
        this.setBypass(prevBypassed, { trails: prevTrails });
      }
    });
  }

  /**
   * Sets the Stage IV captured hardware freeze state directly.
   */
  setFreeze(frozen: boolean, options?: FreezeOptions): this {
    this.#ensureAlive();
    if (frozen && !this.#model.header.ioConfig.capabilities.supportsFreeze) {
      throw new DnamError(
        "ERR_DNAM_PARAMETER",
        `Model '${this.#model.name}' does not support Stage IV hardware freeze`,
      );
    }
    this.#frozen = frozen;
    const amount = options?.amount ?? 1.0;
    if (this.#shared) {
      const setBits = CMD_FREEZE_CHANGE | (frozen ? CMD_FROZEN : 0);
      const clearBits = !frozen ? CMD_FROZEN : 0;
      this.#shared.writeCommand(setBits, clearBits, { freezeAmount: amount });
    } else {
      this.#port.postMessage({ type: "set-freeze", frozen, amount });
    }
    return this;
  }

  /**
   * Engages Stage IV hardware freeze and returns a `StateScope` (`restore()` + `[Symbol.dispose]()`).
   */
  freeze(options?: FreezeOptions): StateScope {
    const prevFrozen = this.#frozen;
    this.setFreeze(true, options);
    return createStateScope(() => {
      if (!this.#disposed) {
        this.setFreeze(prevFrozen);
      }
    });
  }

  /**
   * Resets the audio worklet's internal filter/delay states to DC equilibrium.
   */
  reset(): this {
    this.#ensureAlive();
    if (this.#shared) {
      this.#shared.writeCommand(CMD_RESET);
    } else {
      this.#port.postMessage({ type: "reset" });
    }
    return this;
  }

  /**
   * Hot-swaps a new `Model` into the running worklet processor.
   */
  reload(model: Model<TParams>, options?: { readonly tier?: TierInput | string }): this {
    this.#ensureAlive();
    const nextTier = options?.tier !== undefined ? coerceTierIndex(options.tier) : this.#tierIdx;
    this.#port.postMessage({
      type: "reload",
      containerBytes: model.toBytes(),
      tier: nextTier,
      sampleRate: this.#context.sampleRate,
    });
    this.#controlSpace.dispose();
    this.#model = model;
    this.#tierIdx = nextTier;
    this.#controlSpace = ControlSpace.from<TParams>(model);
    this.#parameters = new ParameterMap<TParams>(
      model.parameters,
      this.#createParameterBackend(),
      this.#controlSpace,
    );
    return this;
  }

  /**
   * Reads the latest lock-free `TelemetrySnapshot` from the worklet processor.
   */
  telemetry(): TelemetrySnapshot {
    this.#ensureAlive();
    if (this.#shared) {
      this.#shared.readTelemetryInto(this.#telemetryScratch);
    }
    return decodeTelemetryF64(this.#telemetryScratch, this.#model.parameters.length);
  }

  /**
   * Connects the underlying `AudioWorkletNode` to a destination `AudioNode`.
   */
  connect(destination: AudioNode, output?: number, input?: number): AudioNode {
    this.#ensureAlive();
    if (!this.#workletNode) {
      throw new DnamError(
        "ERR_DNAM_VALIDATION",
        "connect() requires a browser AudioContext with AudioWorkletNode",
      );
    }
    return this.#workletNode.connect(destination, output, input);
  }

  /**
   * Disconnects the underlying `AudioWorkletNode`.
   */
  disconnect(): void {
    this.#workletNode?.disconnect();
  }

  /**
   * Releases the node and disconnects its underlying `AudioWorkletNode`.
   */
  dispose(): void {
    if (this.#disposed) {
      return;
    }
    this.#disposed = true;
    this.#workletNode?.disconnect();
    this.#controlSpace.dispose();
  }

  /**
   * Disposes the node (`using` protocol).
   */
  [Symbol.dispose](): void {
    this.dispose();
  }
}
