/**
 * Zero-allocation WebAudio `AudioWorkletProcessor` (`DnamAudioWorkletProcessor`) running
 * `WasmDriver` inside `AudioWorkletGlobalScope` (`ref[sdk.ts.web-audio.realtime-bridge]`,
 * `ref[sdk.ts.layout.dependency-direction]`).
 *
 * Strictly zero `"node:*"` imports so it bundles into a self-contained
 * `dist/web-audio-processor.js` module.
 *
 * @module
 */

import { type NativeEngineHandle, TELEMETRY_F64_LEN } from "../internal/native/driver.js";
import { instantiateWasmDriver } from "../internal/native/wasm.js";
import type {
  DnamWorkletProcessorOptions,
  WorkletInboundMessage,
  WorkletOutboundMessage,
} from "./protocol.js";
import { WorkletSharedState } from "./shared-state.js";

/**
 * Canonical WebAudio processor name registered with `audioWorklet.addModule(...)`.
 */
export const DNAM_WORKLET_PROCESSOR_NAME = "dnam-audio-worklet-processor";

interface AudioWorkletPortLike {
  onmessage: ((event: { readonly data: WorkletInboundMessage }) => void) | null;
  postMessage(message: WorkletOutboundMessage): void;
}

const BaseWorkletProcessor: {
  new (options?: {
    readonly processorOptions?: DnamWorkletProcessorOptions;
  }): {
    readonly port: AudioWorkletPortLike;
  };
} =
  typeof globalThis !== "undefined" && "AudioWorkletProcessor" in globalThis
    ? (
        globalThis as unknown as {
          AudioWorkletProcessor: {
            new (options?: {
              readonly processorOptions?: DnamWorkletProcessorOptions;
            }): {
              readonly port: AudioWorkletPortLike;
            };
          };
        }
      ).AudioWorkletProcessor
    : class FallbackAudioWorkletProcessor {
        readonly port: AudioWorkletPortLike = {
          onmessage: null,
          postMessage() {},
        };
      };

/**
 * Real-time `AudioWorkletProcessor` executing `WasmDriver` with zero per-quantum allocations
 * (`ref[sdk.ts.web-audio.realtime-bridge]`).
 */
export class DnamAudioWorkletProcessor extends BaseWorkletProcessor {
  readonly #engine: NativeEngineHandle;
  readonly #wasmMemory: WebAssembly.Memory;
  readonly #shared: WorkletSharedState | undefined;
  readonly #telemetryScratch = new Float64Array(TELEMETRY_F64_LEN);
  readonly #silenceScratch = new Float32Array(128);
  readonly #telemetryIntervalBlocks: number;
  #blockCounter = 0;

  constructor(options?: { readonly processorOptions?: DnamWorkletProcessorOptions }) {
    super(options);
    const procOpts = options?.processorOptions;
    if (!procOpts?.wasmModule || !procOpts.containerBytes) {
      throw new Error(
        "DnamAudioWorkletProcessor requires processorOptions.{wasmModule, containerBytes}",
      );
    }
    const globalSampleRate =
      typeof globalThis !== "undefined" && "sampleRate" in globalThis
        ? Number((globalThis as unknown as { sampleRate: number }).sampleRate)
        : 48_000;
    const driver = instantiateWasmDriver(procOpts.wasmModule, {
      ...(procOpts.wasmMemory !== undefined ? { memory: procOpts.wasmMemory } : {}),
      ...(procOpts.initialMemoryPages !== undefined
        ? { initialPages: procOpts.initialMemoryPages }
        : {}),
      ...(procOpts.maximumMemoryPages !== undefined
        ? { maximumPages: procOpts.maximumMemoryPages }
        : {}),
      ...(procOpts.sharedWasmMemory !== undefined ? { shared: procOpts.sharedWasmMemory } : {}),
    });
    this.#wasmMemory = driver.memory;
    this.#engine = driver.createEngine({
      containerBytes: procOpts.containerBytes,
      tier: procOpts.tier ?? 1,
      sampleRateHz: procOpts.sampleRate ?? globalSampleRate,
      backend: procOpts.backend ?? "auto",
      ...(procOpts.maxSupportedTier !== undefined
        ? { maxSupportedTier: procOpts.maxSupportedTier }
        : {}),
    });
    this.#telemetryIntervalBlocks = Math.max(1, procOpts.telemetryIntervalBlocks ?? 8);
    if (procOpts.sharedBuffer) {
      this.#shared = new WorkletSharedState(procOpts.sharedBuffer);
    }
    this.#publishTelemetry(true);

    this.port.onmessage = (event: { readonly data: WorkletInboundMessage }) => {
      this.#handleMessage(event.data);
    };
  }

  /**
   * Linear `WebAssembly.Memory` instance imported into this processor's `WasmDriver`.
   */
  get wasmMemory(): WebAssembly.Memory {
    return this.#wasmMemory;
  }

  #handleMessage(msg: WorkletInboundMessage): void {
    try {
      switch (msg.type) {
        case "set-param":
          if (msg.immediate) {
            this.#engine.setControlImmediate(msg.index, msg.value);
          } else {
            this.#engine.setControlValue(msg.index, msg.value);
          }
          break;
        case "set-params-batch":
          if (msg.immediate) {
            this.#engine.setControlsImmediateBatch(msg.indices, msg.values);
          } else {
            for (let i = 0; i < msg.indices.length; i++) {
              const idx = msg.indices[i];
              const val = msg.values[i];
              if (idx !== undefined && val !== undefined) {
                this.#engine.setControlValue(idx, val);
              }
            }
          }
          break;
        case "reset-controls":
          this.#engine.resetControlsToDefaults();
          break;
        case "reset":
          this.#engine.reset();
          break;
        case "apply-preset":
          this.#engine.applyPreset(msg.presetId);
          break;
        case "set-bypass":
          this.#engine.setBypass(msg.bypassed, msg.trails);
          break;
        case "set-freeze":
          if (msg.amount !== undefined) {
            this.#engine.setFreezeAmount(msg.amount);
          }
          this.#engine.setFreeze(msg.frozen);
          break;
        case "set-tier":
          this.#engine.setTier(msg.tier);
          break;
        case "set-max-supported-tier":
          this.#engine.setMaxSupportedTier(msg.tier);
          break;
        case "set-width":
          this.#engine.setStereoWidth(msg.width);
          break;
        case "sync-tempo":
          this.#engine.syncTempo(msg.bpm, msg.subdivision, msg.foldOctave);
          break;
        case "tap-tempo":
          this.#engine.registerTapTempo(msg.subdivision, msg.foldOctave);
          break;
        case "reload":
          this.#engine.reload(msg.containerBytes, msg.tier, msg.sampleRate);
          break;
        default:
          break;
      }
      this.#publishTelemetry(true);
    } catch (err) {
      this.port.postMessage({
        type: "error",
        message: err instanceof Error ? err.message : String(err),
      });
    }
  }

  #publishTelemetry(forcePostMessage: boolean): void {
    this.#engine.telemetryInto(this.#telemetryScratch);
    if (this.#shared) {
      this.#shared.publishTelemetry(this.#telemetryScratch, this.#engine.numControls());
    }
    if (!this.#shared || forcePostMessage) {
      this.port.postMessage({
        type: "telemetry",
        data: this.#telemetryScratch.slice(),
      });
    }
  }

  /**
   * Real-time WebAudio `process(inputs, outputs)` callback (`128` frames, zero allocations).
   */
  process(
    inputs: readonly (readonly Float32Array[])[],
    outputs: readonly (readonly Float32Array[])[],
  ): boolean {
    if (this.#shared) {
      this.#shared.drainUpdatesIntoEngine(this.#engine);
    }

    const outBus = outputs[0];
    if (!outBus || outBus.length === 0) {
      return true;
    }
    const outL = outBus[0];
    if (!outL) {
      return true;
    }
    const frames = outL.length;
    const outR = outBus[1] ?? outL;
    const inBus = inputs[0];
    const inCh0 = inBus?.[0];
    const inCh1 = inBus?.[1];
    const inL = inCh0 && inCh0.length === frames ? inCh0 : this.#silenceScratch.subarray(0, frames);
    const inR = inCh1 && inCh1.length === frames ? inCh1 : inL;

    if (outBus.length >= 2) {
      if (inCh0 && inCh0.length === frames && inCh1 && inCh1.length === frames) {
        this.#engine.processStereo(inL, inR, outL, outR);
      } else {
        this.#engine.processMonoToStereo(inL, outL, outR);
      }
    } else {
      this.#engine.processMono(inL, outL);
    }

    this.#blockCounter++;
    if (this.#blockCounter >= this.#telemetryIntervalBlocks) {
      this.#blockCounter = 0;
      this.#publishTelemetry(false);
    }
    return true;
  }
}

if (
  typeof globalThis !== "undefined" &&
  "registerProcessor" in globalThis &&
  typeof (globalThis as unknown as { registerProcessor: unknown }).registerProcessor === "function"
) {
  (
    globalThis as unknown as {
      registerProcessor(name: string, ctor: typeof DnamAudioWorkletProcessor): void;
    }
  ).registerProcessor(DNAM_WORKLET_PROCESSOR_NAME, DnamAudioWorkletProcessor);
}
