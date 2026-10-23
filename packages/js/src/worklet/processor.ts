/**
 * WebAudio `AudioWorkletProcessor` (`DnamAudioWorkletProcessor`) for `AudioWorkletGlobalScope`.
 *
 * Self-contained and zero-dependency (`zod` and `node:*` are forbidden by `biome.json`).
 * Supports lock-free 256-byte `SharedArrayBuffer` (`Atomics`) synchronization with automatic
 * fallback to `MessagePort` when `crossOriginIsolated === false`.
 */

import type { DnamEngineHandle } from "../drivers/types.js";
import { initDnamWasmSync, type WasmDriver } from "../drivers/wasm.js";
import {
  DnamSharedBridge,
  type WorkletInboundMessage,
  type WorkletOutboundMessage,
  type WorkletTelemetryDigest,
} from "./shared_memory.js";

export const DNAM_WORKLET_PROCESSOR_NAME = "dnam-audio-worklet-processor";

export interface DnamWorkletProcessorOptions {
  wasmBinary: BufferSource | WebAssembly.Module;
  containerBytes: Uint8Array | ArrayBuffer;
  tier?: number | undefined;
  sampleRateHz?: number | undefined;
  sharedBuffer?: SharedArrayBuffer | ArrayBuffer | undefined;
  telemetryIntervalBlocks?: number | undefined;
}

export interface AudioWorkletInitOptions {
  processorOptions?: DnamWorkletProcessorOptions | undefined;
}

export interface WorkletPortLike {
  onmessage: ((event: { data: WorkletInboundMessage }) => void) | null;
  postMessage(message: WorkletOutboundMessage): void;
}

class FallbackWorkletBase {
  public readonly port: WorkletPortLike;

  public constructor() {
    this.port = {
      onmessage: null,
      postMessage: () => {
        // No-op when not attached to a real MessagePort
      },
    };
  }
}

const GlobalWorkletBase = (
  typeof globalThis !== "undefined" &&
  "AudioWorkletProcessor" in globalThis &&
  typeof (globalThis as Record<string, unknown>)["AudioWorkletProcessor"] === "function"
    ? (globalThis as unknown as { AudioWorkletProcessor: typeof FallbackWorkletBase })
        .AudioWorkletProcessor
    : FallbackWorkletBase
) as typeof FallbackWorkletBase;

function normalizeBytes(input: Uint8Array | ArrayBuffer): Uint8Array {
  return input instanceof Uint8Array ? input : new Uint8Array(input);
}

/**
 * Real-time WebAudio `AudioWorkletProcessor` running `dnam_engine.wasm` SIMD128 in `AudioWorkletGlobalScope`.
 */
export class DnamAudioWorkletProcessor extends GlobalWorkletBase {
  private readonly driver: WasmDriver;
  private readonly engine: DnamEngineHandle;
  private readonly sharedBridge: DnamSharedBridge | null;
  private readonly numControls: number;
  private readonly telemetryIntervalBlocks: number;
  private blockCounter = 0;
  private lastCtrlSeq = 0;
  private lastResetSeq = 0;
  private readonly appliedControls = new Float32Array(16);

  public constructor(options?: AudioWorkletInitOptions) {
    super();
    const procOpts = options?.processorOptions;
    if (!procOpts?.wasmBinary || !procOpts.containerBytes) {
      throw new Error(
        "DnamAudioWorkletProcessor requires processorOptions.wasmBinary and processorOptions.containerBytes",
      );
    }

    this.driver = initDnamWasmSync(procOpts.wasmBinary);
    const containerBytes = normalizeBytes(procOpts.containerBytes);
    const globalSampleRate =
      typeof globalThis !== "undefined" &&
      "sampleRate" in globalThis &&
      typeof (globalThis as Record<string, unknown>)["sampleRate"] === "number"
        ? ((globalThis as Record<string, unknown>)["sampleRate"] as number)
        : 48_000;
    const sampleRateHz = procOpts.sampleRateHz ?? globalSampleRate;
    const initialTier = procOpts.tier ?? 1;

    this.numControls = containerBytes.byteLength >= 64 ? (containerBytes[10] ?? 0) : 0;
    this.engine = this.driver.createEngine(
      containerBytes,
      initialTier,
      sampleRateHz,
      "wasm_simd128",
    );
    this.telemetryIntervalBlocks = Math.max(1, procOpts.telemetryIntervalBlocks ?? 8);

    if (procOpts.sharedBuffer) {
      this.sharedBridge = new DnamSharedBridge(procOpts.sharedBuffer);
      this.syncFromSharedMemory(true);
      this.publishTelemetryToSharedMemory();
    } else {
      this.sharedBridge = null;
    }

    this.port.onmessage = (event: { data: WorkletInboundMessage }) => {
      this.handleMessage(event.data);
    };
  }

  private handleMessage(msg: WorkletInboundMessage): void {
    try {
      switch (msg.type) {
        case "setControl": {
          if (msg.immediate) {
            this.engine.setControlImmediate(msg.index, msg.value);
          } else {
            this.engine.setControlValue(msg.index, msg.value);
          }
          if (msg.index >= 0 && msg.index < 16) {
            this.appliedControls[msg.index] = msg.value;
          }
          break;
        }
        case "setControls": {
          for (const entry of msg.entries) {
            if (msg.immediate) {
              this.engine.setControlImmediate(entry.index, entry.value);
            } else {
              this.engine.setControlValue(entry.index, entry.value);
            }
            if (entry.index >= 0 && entry.index < 16) {
              this.appliedControls[entry.index] = entry.value;
            }
          }
          break;
        }
        case "setTier": {
          this.engine.activeTier = msg.tier;
          break;
        }
        case "reset": {
          this.engine.reset();
          break;
        }
        case "reload": {
          this.engine.reload(msg.containerBytes, msg.tier, msg.sampleRateHz);
          this.port.postMessage({
            type: "reloaded",
            activeTier: this.engine.activeTier,
          });
          break;
        }
        case "requestTelemetry": {
          this.port.postMessage({
            type: "telemetry",
            telemetry: this.extractDigest(),
          });
          break;
        }
        default: {
          break;
        }
      }
    } catch (err) {
      this.port.postMessage({
        type: "error",
        message: err instanceof Error ? err.message : String(err),
      });
    }
  }

  private syncFromSharedMemory(forceImmediate = false): void {
    if (!this.sharedBridge) {
      return;
    }
    const ctrlSeq = this.sharedBridge.getCtrlSeq();
    if (!forceImmediate && ctrlSeq === this.lastCtrlSeq) {
      return;
    }
    this.lastCtrlSeq = ctrlSeq;

    const resetSeq = this.sharedBridge.getResetSeq();
    if (resetSeq !== this.lastResetSeq) {
      this.lastResetSeq = resetSeq;
      this.engine.reset();
    }

    const targetTier = this.sharedBridge.getTargetTier();
    if (targetTier >= 0 && targetTier <= 2 && targetTier !== this.engine.activeTier) {
      this.engine.activeTier = targetTier;
    }

    const immediateMask = this.sharedBridge.consumeImmediateMask();
    const limit = Math.max(this.numControls, 16);
    for (let i = 0; i < limit; i++) {
      const targetVal = this.sharedBridge.readTargetControl(i);
      const isImm = forceImmediate || (immediateMask & (1 << i)) !== 0;
      if (isImm || targetVal !== this.appliedControls[i]) {
        if (isImm) {
          this.engine.setControlImmediate(i, targetVal);
        } else {
          this.engine.setControlValue(i, targetVal);
        }
        this.appliedControls[i] = targetVal;
      }
    }
  }

  private extractDigest(): WorkletTelemetryDigest {
    const t = this.engine.telemetry();
    return {
      activeTier: t.activeTier,
      activeClampMask: t.activeClampMask,
      controlDecimationHost: t.controlDecimationHost,
      maxAdaaDrive: t.maxAdaaDrive,
      maxPoleRadius: t.maxPoleRadius,
      maxEffectiveLoopGain: t.maxEffectiveLoopGain,
      tailTimeMs: t.tailTimeMs,
      zHead: t.zHead,
      envStates: t.envStates,
      lfoHarmonics: t.lfoHarmonics,
      clampedControls: t.clampedControls,
    };
  }

  private publishTelemetryToSharedMemory(): void {
    if (this.sharedBridge) {
      this.sharedBridge.publishTelemetry(this.extractDigest());
    }
  }

  /**
   * Standard WebAudio 128-sample quantum callback (`process(inputs, outputs, parameters)`).
   */
  public process(
    inputs: readonly (readonly Float32Array[])[],
    outputs: readonly (readonly Float32Array[])[],
  ): boolean {
    this.syncFromSharedMemory(false);

    const outBus = outputs[0];
    if (!outBus || outBus.length === 0) {
      return true;
    }
    const outL = outBus[0];
    if (!outL || outL.length === 0) {
      return true;
    }
    const n = outL.length;
    const outR = outBus[1];
    const inBus = inputs[0];
    const inL = inBus?.[0] ?? new Float32Array(n);
    const inR = inBus?.[1];

    if (outR !== undefined) {
      if (inR !== undefined) {
        this.engine.processStereo(inL, inR, outL, outR);
      } else {
        this.engine.processMonoToStereo(inL, outL, outR);
      }
    } else {
      this.engine.processMono(inL, outL);
    }

    this.blockCounter++;
    if (this.blockCounter >= this.telemetryIntervalBlocks) {
      this.blockCounter = 0;
      this.publishTelemetryToSharedMemory();
    }

    return true;
  }

  public dispose(): void {
    this.engine.dispose();
  }
}

// Auto-register inside AudioWorkletGlobalScope when `registerProcessor` is present
if (
  typeof globalThis !== "undefined" &&
  "registerProcessor" in globalThis &&
  typeof (globalThis as Record<string, unknown>)["registerProcessor"] === "function"
) {
  const reg = (
    globalThis as unknown as {
      registerProcessor: (name: string, ctor: typeof DnamAudioWorkletProcessor) => void;
    }
  ).registerProcessor;
  reg(DNAM_WORKLET_PROCESSOR_NAME, DnamAudioWorkletProcessor);
}
