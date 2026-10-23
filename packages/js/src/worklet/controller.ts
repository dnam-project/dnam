/**
 * Main-thread WebAudio `AudioWorkletNode` controller (`DnamWorkletController<TControls>`)
 * pairing lock-free 256-byte `SharedArrayBuffer` (`Atomics`) synchronization with automatic
 * `MessagePort` fallback when `crossOriginIsolated === false`.
 */

import { DnamContainer, toUint8Array } from "../container.js";
import { ControlRack, type ControlSink, ControlSpace, type SetControlOptions } from "../control.js";
import { type BackendChoice, getCachedWasmBinary, initDnamWasm } from "../drivers/index.js";
import { DnamError } from "../errors.js";
import {
  type ControlMap,
  parseTier,
  type TierCode,
  type TierInput,
  type TierName,
  tierToName,
} from "../schema/base.js";
import type { ClampResultSnapshot } from "../schema/runtime.js";
import { DNAM_WORKLET_PROCESSOR_NAME, type DnamWorkletProcessorOptions } from "./processor.js";
import {
  createDnamSharedBuffer,
  DnamSharedBridge,
  isSharedArrayBufferSupported,
  type WorkletInboundMessage,
  type WorkletOutboundMessage,
  type WorkletTelemetryDigest,
} from "./shared_memory.js";

export interface WorkletNodePortLike {
  onmessage: ((event: { data: WorkletOutboundMessage }) => void) | null;
  postMessage(message: WorkletInboundMessage): void;
}

export interface AudioWorkletNodeLike {
  readonly port: WorkletNodePortLike;
  disconnect?(): void;
}

export interface AudioWorkletContextLike {
  readonly sampleRate: number;
  readonly audioWorklet?: {
    addModule(moduleUrl: string | URL): Promise<void>;
  };
}

export type AudioWorkletNodeConstructorLike = new (
  context: AudioWorkletContextLike,
  name: string,
  options?: {
    numberOfInputs?: number;
    numberOfOutputs?: number;
    outputChannelCount?: number[];
    processorOptions?: DnamWorkletProcessorOptions;
  },
) => AudioWorkletNodeLike;

export interface CreateDnamWorkletNodeOptions {
  tier?: TierInput | undefined;
  wasmSource?:
    | BufferSource
    | WebAssembly.Module
    | Response
    | Promise<Response>
    | string
    | URL
    | undefined;
  workletModuleUrl?: string | URL | undefined;
  processorName?: string | undefined;
  useSharedArrayBuffer?: boolean | undefined;
  telemetryIntervalBlocks?: number | undefined;
  backend?: BackendChoice | undefined;
  NodeConstructor?: AudioWorkletNodeConstructorLike | undefined;
}

/**
 * Main-thread controller wrapping a WebAudio `AudioWorkletNode` and exposing the same
 * ergonomic `ControlRack<TControls>`, preset, hot-swap, and telemetry API as `DnamEngine<TControls>`.
 */
export class DnamWorkletController<TControls extends ControlMap = ControlMap>
  implements Disposable
{
  public readonly node: AudioWorkletNodeLike;
  private currentContainer: DnamContainer;
  private currentSpace: ControlSpace;
  private currentRack: ControlRack<TControls>;
  private readonly sharedBridge: DnamSharedBridge | null;
  private readonly backendChoice?: BackendChoice | undefined;
  private currentTier: TierCode;
  private lastMessageTelemetry: WorkletTelemetryDigest;

  public constructor(
    node: AudioWorkletNodeLike,
    containerInput: DnamContainer | Uint8Array | ArrayBuffer | ArrayBufferView,
    options: {
      tier?: TierInput | undefined;
      sharedBuffer?: SharedArrayBuffer | ArrayBuffer | null | undefined;
      backend?: BackendChoice | undefined;
    } = {},
  ) {
    this.node = node;
    this.backendChoice = options.backend;
    this.currentContainer =
      containerInput instanceof DnamContainer
        ? containerInput
        : DnamContainer.fromBytes(containerInput, options.backend);
    this.currentTier = parseTier(options.tier ?? "full");
    this.currentSpace = new ControlSpace(this.currentContainer.header, options.backend);
    this.sharedBridge = options.sharedBuffer ? new DnamSharedBridge(options.sharedBuffer) : null;

    this.lastMessageTelemetry = {
      activeTier: this.currentTier,
      activeClampMask: 0,
      controlDecimationHost: this.currentContainer.preamble.controlDecimation,
      maxAdaaDrive: 0,
      maxPoleRadius: 0,
      maxEffectiveLoopGain: 0,
      tailTimeMs: this.currentContainer.preamble.tailTimeMs,
      zHead: [0, 0, 0, 0],
      envStates: [0, 0, 0, 0],
      lfoHarmonics: [0, 0, 0, 0, 0, 0, 0, 0],
      clampedControls: new Array<number>(16).fill(0),
    };

    this.node.port.onmessage = (event: { data: WorkletOutboundMessage }) => {
      const msg = event.data;
      if (msg.type === "telemetry") {
        this.lastMessageTelemetry = msg.telemetry;
        this.currentTier = parseTier(msg.telemetry.activeTier);
      } else if (msg.type === "reloaded") {
        this.currentTier = parseTier(msg.activeTier);
      }
    };

    const sink: ControlSink = {
      setNumericControl: (ctrlIdx, numericValue, immediate) => {
        if (this.sharedBridge) {
          this.sharedBridge.writeControl(ctrlIdx, numericValue, immediate);
        } else {
          this.node.port.postMessage({
            type: "setControl",
            index: ctrlIdx,
            value: numericValue,
            immediate,
          });
        }
      },
    };
    this.currentRack = new ControlRack<TControls>(this.currentSpace, sink);
  }

  /**
   * Async factory that initializes `dnam_engine.wasm`, optionally loads the worklet processor module,
   * creates a lock-free 256-byte `SharedArrayBuffer` (when supported), instantiates the `AudioWorkletNode`,
   * and returns a connected `DnamWorkletController<TControls>`.
   */
  public static async createNode<TControls extends ControlMap = ControlMap>(
    audioContext: AudioWorkletContextLike,
    containerInput: DnamContainer | Uint8Array | ArrayBuffer | ArrayBufferView,
    options: CreateDnamWorkletNodeOptions = {},
  ): Promise<DnamWorkletController<TControls>> {
    await initDnamWasm(options.wasmSource);
    const wasmBinary = getCachedWasmBinary();
    if (!wasmBinary) {
      throw new DnamError(
        "WASM binary bytes not available in cache; pass a BufferSource or URL to DnamWorkletController.createNode",
      );
    }

    if (options.workletModuleUrl !== undefined && audioContext.audioWorklet !== undefined) {
      await audioContext.audioWorklet.addModule(options.workletModuleUrl);
    }

    const backend = options.backend ?? "wasm";
    const container =
      containerInput instanceof DnamContainer
        ? containerInput
        : DnamContainer.fromBytes(containerInput, backend);
    const tierCode = parseTier(options.tier ?? "full");

    const initialControls: number[] = [];
    for (const def of container.header.controlsSchema) {
      if (def.type === "switch") {
        const opts = def.declaredOptions ?? [];
        const idx = typeof def.default === "string" ? opts.indexOf(def.default) : 0;
        initialControls.push(Math.max(idx, 0));
      } else {
        initialControls.push(typeof def.default === "number" ? def.default : 0);
      }
    }

    const useSab = (options.useSharedArrayBuffer ?? true) && isSharedArrayBufferSupported();
    const sharedBuffer = useSab ? createDnamSharedBuffer(tierCode, initialControls) : undefined;

    const NodeCtor =
      options.NodeConstructor ??
      ((globalThis as Record<string, unknown>)["AudioWorkletNode"] as
        | AudioWorkletNodeConstructorLike
        | undefined);
    if (!NodeCtor) {
      throw new DnamError(
        "AudioWorkletNode is not defined in globalThis; pass options.NodeConstructor in non-browser environments",
      );
    }

    const node = new NodeCtor(audioContext, options.processorName ?? DNAM_WORKLET_PROCESSOR_NAME, {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [2],
      processorOptions: {
        wasmBinary,
        containerBytes: container.bytes,
        tier: tierCode,
        sampleRateHz: audioContext.sampleRate,
        sharedBuffer,
        telemetryIntervalBlocks: options.telemetryIntervalBlocks,
      },
    });

    return new DnamWorkletController<TControls>(node, container, {
      tier: tierCode,
      sharedBuffer,
      backend,
    });
  }

  public get container(): DnamContainer {
    return this.currentContainer;
  }

  public get controls(): ControlRack<TControls> {
    return this.currentRack;
  }

  public get usesSharedMemory(): boolean {
    return this.sharedBridge !== null;
  }

  public get activeTier(): TierCode {
    if (this.sharedBridge) {
      return parseTier(this.sharedBridge.readTelemetry().activeTier);
    }
    return this.currentTier;
  }

  public get activeTierName(): TierName {
    return tierToName(this.activeTier);
  }

  public setTier(tier: TierInput): void {
    const code = parseTier(tier);
    this.currentTier = code;
    if (this.sharedBridge) {
      this.sharedBridge.writeTier(code);
    } else {
      this.node.port.postMessage({ type: "setTier", tier: code });
    }
  }

  public setControls(
    controls: Partial<TControls>,
    options?: SetControlOptions,
  ): ClampResultSnapshot {
    return this.currentRack.setAll(controls, options);
  }

  public applyPreset(presetId: string, options?: SetControlOptions): ClampResultSnapshot {
    return this.currentRack.applyPreset(presetId, options);
  }

  public reset(): void {
    if (this.sharedBridge) {
      this.sharedBridge.triggerReset();
    } else {
      this.node.port.postMessage({ type: "reset" });
    }
  }

  public reload(
    containerInput: DnamContainer | Uint8Array | ArrayBuffer | ArrayBufferView,
    options: { tier?: TierInput | undefined; sampleRateHz?: number | undefined } = {},
  ): void {
    const nextContainer =
      containerInput instanceof DnamContainer
        ? containerInput
        : DnamContainer.fromBytes(toUint8Array(containerInput), this.backendChoice);
    const nextTier = options.tier !== undefined ? parseTier(options.tier) : undefined;
    if (nextTier !== undefined) {
      this.currentTier = nextTier;
      if (this.sharedBridge) {
        this.sharedBridge.writeTier(nextTier);
      }
    }
    this.node.port.postMessage({
      type: "reload",
      containerBytes: nextContainer.bytes,
      tier: nextTier,
      sampleRateHz: options.sampleRateHz,
    });
    this.currentSpace.dispose();
    this.currentContainer = nextContainer;
    this.currentSpace = new ControlSpace(nextContainer.header, this.backendChoice);
    const sink: ControlSink = {
      setNumericControl: (ctrlIdx, numericValue, immediate) => {
        if (this.sharedBridge) {
          this.sharedBridge.writeControl(ctrlIdx, numericValue, immediate);
        } else {
          this.node.port.postMessage({
            type: "setControl",
            index: ctrlIdx,
            value: numericValue,
            immediate,
          });
        }
      },
    };
    this.currentRack = new ControlRack<TControls>(this.currentSpace, sink);
  }

  /**
   * Reads the latest lock-free `WorkletTelemetryDigest` from `SharedArrayBuffer` (or requests
   * an async update via `MessagePort` when running without `SharedArrayBuffer`).
   */
  public telemetry(): WorkletTelemetryDigest {
    if (this.sharedBridge) {
      return this.sharedBridge.readTelemetry();
    }
    this.node.port.postMessage({ type: "requestTelemetry" });
    return this.lastMessageTelemetry;
  }

  public dispose(): void {
    this.currentSpace.dispose();
    this.node.disconnect?.();
  }

  public [Symbol.dispose](): void {
    this.dispose();
  }
}
