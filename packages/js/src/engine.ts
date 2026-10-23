/**
 * Primary Stateful Real-Time Inference Engine (`DnamEngine<TControls>` and `DnamProcessor` alias).
 */

import { DnamContainer, toUint8Array } from "./container.js";
import { ControlRack, type ControlSink, ControlSpace, type SetControlOptions } from "./control.js";
import { type BackendChoice, type DnamEngineHandle, resolveDriver } from "./drivers/index.js";
import { DnamValidationError } from "./errors.js";
import {
  type ControlMap,
  type ControlValue,
  parseTier,
  type TierCode,
  type TierInput,
  type TierName,
  tierToName,
} from "./schema/base.js";
import type { DnamJsonHeader } from "./schema/header.js";
import type { ClampResultSnapshot, PreambleSnapshot, TelemetrySnapshot } from "./schema/runtime.js";

export interface DnamEngineOptions {
  tier?: TierInput | undefined;
  sampleRateHz?: number | undefined;
  backend?: BackendChoice | undefined;
}

export interface ProcessPlanarOptions {
  bufferSize?: number | undefined;
  controls?: ControlMap | undefined;
  immediateControls?: boolean | undefined;
}

export interface ProcessStereoOptions extends ProcessPlanarOptions {
  leftOut?: Float32Array | undefined;
  rightOut?: Float32Array | undefined;
}

export interface ProcessMonoOptions extends ProcessPlanarOptions {
  monoOut?: Float32Array | undefined;
}

/**
 * WebAudio `AudioBuffer`-compatible duck-typed interface usable in both Browser and Node.js.
 */
export interface AudioBufferLike {
  readonly numberOfChannels: number;
  readonly length: number;
  readonly sampleRate: number;
  getChannelData(channel: number): Float32Array;
}

class MemoryAudioBuffer implements AudioBufferLike {
  public readonly numberOfChannels: number;
  public readonly length: number;
  public readonly sampleRate: number;
  private readonly channels: Float32Array[];

  public constructor(channels: Float32Array[], sampleRate: number) {
    this.channels = channels;
    this.numberOfChannels = channels.length;
    this.length = channels[0]?.length ?? 0;
    this.sampleRate = sampleRate;
  }

  public getChannelData(channel: number): Float32Array {
    const ch = this.channels[channel];
    if (!ch) {
      throw new DnamValidationError(
        `Channel index ${String(channel)} out of bounds (numberOfChannels=${String(this.numberOfChannels)})`,
      );
    }
    return ch;
  }
}

/**
 * Stateful real-time DNAM ("Tri-Physics") inference engine generic over the model's
 * parametric control map `TControls`.
 */
export class DnamEngine<TControls extends ControlMap = ControlMap> implements Disposable {
  private currentContainer: DnamContainer;
  private currentSpace: ControlSpace;
  private currentRack: ControlRack<TControls>;
  private readonly handle: DnamEngineHandle;
  private readonly backendChoice?: BackendChoice | undefined;

  public constructor(
    containerInput: DnamContainer | Uint8Array | ArrayBuffer | ArrayBufferView,
    options: DnamEngineOptions = {},
  ) {
    this.backendChoice = options.backend;
    this.currentContainer =
      containerInput instanceof DnamContainer
        ? containerInput
        : DnamContainer.fromBytes(containerInput, options.backend);

    const tierCode = parseTier(options.tier ?? "full");
    const sampleRateHz = options.sampleRateHz ?? 48_000;
    const driver = resolveDriver(options.backend);
    this.handle = driver.createEngine(
      this.currentContainer.bytes,
      tierCode,
      sampleRateHz,
      options.backend,
    );
    this.currentSpace = new ControlSpace(this.currentContainer.header, options.backend);
    const sink: ControlSink = {
      setNumericControl: (ctrlIdx, numericValue, immediate) => {
        if (immediate) {
          this.handle.setControlImmediate(ctrlIdx, numericValue);
        } else {
          this.handle.setControlValue(ctrlIdx, numericValue);
        }
      },
    };
    this.currentRack = new ControlRack<TControls>(this.currentSpace, sink);
  }

  public get container(): DnamContainer {
    return this.currentContainer;
  }

  public get header(): DnamJsonHeader {
    return this.currentContainer.header;
  }

  public get preamble(): PreambleSnapshot {
    return this.currentContainer.preamble;
  }

  public get controls(): ControlRack<TControls> {
    return this.currentRack;
  }

  public get backend(): string {
    return this.handle.backend;
  }

  public get activeTier(): TierCode {
    return parseTier(this.handle.activeTier);
  }

  public get activeTierName(): TierName {
    return tierToName(this.handle.activeTier);
  }

  public setTier(tier: TierInput): void {
    this.handle.activeTier = parseTier(tier);
  }

  public get sampleRateHz(): number {
    return this.handle.sampleRateHz;
  }

  public setSampleRate(sampleRateHz: number): void {
    this.handle.sampleRateHz = sampleRateHz;
  }

  /**
   * Hot-swaps a new `.dnam` or `.dnam-ckpt` container in-place.
   */
  public reload(
    containerInput: DnamContainer | Uint8Array | ArrayBuffer | ArrayBufferView,
    options: { tier?: TierInput | undefined; sampleRateHz?: number | undefined } = {},
  ): void {
    const nextContainer =
      containerInput instanceof DnamContainer
        ? containerInput
        : DnamContainer.fromBytes(toUint8Array(containerInput), this.backendChoice);
    const nextTier = options.tier !== undefined ? parseTier(options.tier) : undefined;
    this.handle.reload(nextContainer.bytes, nextTier, options.sampleRateHz);
    this.currentSpace.dispose();
    this.currentContainer = nextContainer;
    this.currentSpace = new ControlSpace(nextContainer.header, this.backendChoice);
    const sink: ControlSink = {
      setNumericControl: (ctrlIdx, numericValue, immediate) => {
        if (immediate) {
          this.handle.setControlImmediate(ctrlIdx, numericValue);
        } else {
          this.handle.setControlValue(ctrlIdx, numericValue);
        }
      },
    };
    this.currentRack = new ControlRack<TControls>(this.currentSpace, sink);
  }

  /**
   * Sets a single control by string ID (`"gain"`, `"boost"`) or numeric index (`0..15`).
   */
  public setControlValue(
    idOrIndex: (keyof TControls & string) | number,
    value: ControlValue,
  ): void {
    if (typeof idOrIndex === "number") {
      if (!Number.isInteger(idOrIndex) || idOrIndex < 0 || idOrIndex >= 16) {
        throw new DnamValidationError(`Control index must be in 0..15, got ${String(idOrIndex)}`);
      }
      if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new DnamValidationError("Numeric control index requires a finite number value");
      }
      this.handle.setControlValue(idOrIndex, value);
      return;
    }
    this.currentRack.set(idOrIndex, value as TControls[typeof idOrIndex], { immediate: false });
  }

  /**
   * Immediately snaps a control by string ID (`"gain"`, `"boost"`) or numeric index (`0..15`)
   * without $3\text{ kHz}$ slewing.
   */
  public setControlImmediate(
    idOrIndex: (keyof TControls & string) | number,
    value: ControlValue,
  ): void {
    if (typeof idOrIndex === "number") {
      if (!Number.isInteger(idOrIndex) || idOrIndex < 0 || idOrIndex >= 16) {
        throw new DnamValidationError(`Control index must be in 0..15, got ${String(idOrIndex)}`);
      }
      if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new DnamValidationError("Numeric control index requires a finite number value");
      }
      this.handle.setControlImmediate(idOrIndex, value);
      return;
    }
    this.currentRack.set(idOrIndex, value as TControls[typeof idOrIndex], { immediate: true });
  }

  /**
   * Updates multiple controls by ID and returns the 3-Phase `ClampResultSnapshot`.
   */
  public setControls(
    controls: Partial<TControls>,
    options?: SetControlOptions,
  ): ClampResultSnapshot {
    return this.currentRack.setAll(controls, options);
  }

  /**
   * Applies a built-in preset by `presetId` and returns the 3-Phase `ClampResultSnapshot`.
   */
  public applyPreset(presetId: string, options?: SetControlOptions): ClampResultSnapshot {
    return this.currentRack.applyPreset(presetId, options);
  }

  /**
   * Resets all Core 1 (`PolyADAA` + `SelectiveRgLru`), Core 2 (`M-UFDN`), and $3\text{ kHz}$
   * Control Bus states to analytical DC resting equilibrium (`CL-02`).
   */
  public reset(): void {
    this.handle.reset();
  }

  private applyOptionalControls(options?: ProcessPlanarOptions): void {
    if (options?.controls !== undefined) {
      this.currentRack.setAll(options.controls as Partial<TControls>, {
        immediate: options.immediateControls ?? false,
      });
    }
  }

  /**
   * Zero-allocation planar `Float32Array[]` block processor directly compatible with WebAudio
   * `AudioWorkletProcessor.process(inputs[0], outputs[0])`.
   */
  public processPlanar(
    inputs: readonly Float32Array[],
    outputs: readonly Float32Array[],
    options?: ProcessPlanarOptions,
  ): void {
    this.applyOptionalControls(options);
    const out0 = outputs[0];
    if (!out0 || out0.length === 0) {
      return;
    }
    const n = out0.length;
    const out1 = outputs[1];
    const in0 = inputs[0] ?? new Float32Array(n);
    const in1 = inputs[1];
    const bufferSize = options?.bufferSize;

    // Guard against in-place aliasing when caller passes the same Float32Array view as input and output
    const safeIn0 =
      in0.buffer === out0.buffer && in0.byteOffset === out0.byteOffset ? in0.slice() : in0;

    if (out1 !== undefined) {
      if (in1 !== undefined) {
        const safeIn1 =
          (in1.buffer === out0.buffer && in1.byteOffset === out0.byteOffset) ||
          (in1.buffer === out1.buffer && in1.byteOffset === out1.byteOffset)
            ? in1.slice()
            : in1;
        this.handle.processStereo(safeIn0, safeIn1, out0, out1, bufferSize);
      } else {
        this.handle.processMonoToStereo(safeIn0, out0, out1, bufferSize);
      }
    } else {
      this.handle.processMono(safeIn0, out0, bufferSize);
    }
  }

  /**
   * Processes a WebAudio `AudioBuffer` (or `AudioBufferLike`) into `outputBuffer` (or a newly
   * allocated `AudioBufferLike` if `outputBuffer` is omitted).
   */
  public processAudioBuffer(
    inputBuffer: AudioBufferLike,
    outputBuffer?: AudioBufferLike,
    options?: ProcessPlanarOptions,
  ): AudioBufferLike {
    if (inputBuffer.sampleRate !== this.sampleRateHz) {
      this.setSampleRate(inputBuffer.sampleRate);
    }
    const inChannels: Float32Array[] = [];
    const numIn = Math.min(2, inputBuffer.numberOfChannels);
    for (let c = 0; c < numIn; c++) {
      inChannels.push(inputBuffer.getChannelData(c));
    }

    const target =
      outputBuffer ??
      new MemoryAudioBuffer(
        [new Float32Array(inputBuffer.length), new Float32Array(inputBuffer.length)],
        inputBuffer.sampleRate,
      );
    const outChannels: Float32Array[] = [];
    const numOut = Math.min(2, target.numberOfChannels);
    for (let c = 0; c < numOut; c++) {
      outChannels.push(target.getChannelData(c));
    }

    this.processPlanar(inChannels, outChannels, options);
    return target;
  }

  /**
   * Processes stereo `leftIn` and `rightIn` `Float32Array` buffers into `[leftOut, rightOut]`.
   */
  public processStereo(
    leftIn: Float32Array,
    rightIn: Float32Array,
    options?: ProcessStereoOptions,
  ): [Float32Array, Float32Array] {
    this.applyOptionalControls(options);
    const n = leftIn.length;
    const leftOut = options?.leftOut ?? new Float32Array(n);
    const rightOut = options?.rightOut ?? new Float32Array(n);
    const safeLeftIn =
      leftIn.buffer === leftOut.buffer && leftIn.byteOffset === leftOut.byteOffset
        ? leftIn.slice()
        : leftIn;
    const safeRightIn =
      rightIn.buffer === rightOut.buffer && rightIn.byteOffset === rightOut.byteOffset
        ? rightIn.slice()
        : rightIn;
    this.handle.processStereo(safeLeftIn, safeRightIn, leftOut, rightOut, options?.bufferSize);
    return [leftOut, rightOut];
  }

  /**
   * Processes a 1-channel `monoIn` `Float32Array` into true-stereo `[leftOut, rightOut]`.
   */
  public processMonoToStereo(
    monoIn: Float32Array,
    options?: ProcessStereoOptions,
  ): [Float32Array, Float32Array] {
    this.applyOptionalControls(options);
    const n = monoIn.length;
    const leftOut = options?.leftOut ?? new Float32Array(n);
    const rightOut = options?.rightOut ?? new Float32Array(n);
    const safeMonoIn =
      (monoIn.buffer === leftOut.buffer && monoIn.byteOffset === leftOut.byteOffset) ||
      (monoIn.buffer === rightOut.buffer && monoIn.byteOffset === rightOut.byteOffset)
        ? monoIn.slice()
        : monoIn;
    this.handle.processMonoToStereo(safeMonoIn, leftOut, rightOut, options?.bufferSize);
    return [leftOut, rightOut];
  }

  /**
   * Processes a 1-channel `monoIn` `Float32Array` into a 1-channel `monoOut` `Float32Array`.
   */
  public processMono(monoIn: Float32Array, options?: ProcessMonoOptions): Float32Array {
    this.applyOptionalControls(options);
    const n = monoIn.length;
    const monoOut = options?.monoOut ?? new Float32Array(n);
    const safeMonoIn =
      monoIn.buffer === monoOut.buffer && monoIn.byteOffset === monoOut.byteOffset
        ? monoIn.slice()
        : monoIn;
    this.handle.processMono(safeMonoIn, monoOut, options?.bufferSize);
    return monoOut;
  }

  /**
   * Convenience dispatcher: processes `(leftIn, rightIn)` as stereo or `(monoIn)` as mono-to-stereo.
   */
  public process(
    leftOrMonoIn: Float32Array,
    rightIn?: Float32Array,
    options?: ProcessStereoOptions,
  ): [Float32Array, Float32Array] {
    if (rightIn !== undefined) {
      return this.processStereo(leftOrMonoIn, rightIn, options);
    }
    return this.processMonoToStereo(leftOrMonoIn, options);
  }

  /**
   * Asynchronously processes audio on the libuv worker pool (Native N-API) or async promise (WASM)
   * without blocking the Node.js event loop.
   */
  public async processAsync(
    leftOrMonoIn: Float32Array,
    rightIn?: Float32Array,
    options?: ProcessPlanarOptions,
  ): Promise<[Float32Array, Float32Array]> {
    this.applyOptionalControls(options);
    return this.handle.processAsync(leftOrMonoIn, rightIn, options?.bufferSize);
  }

  /**
   * Extracts a lock-free `TelemetrySnapshot` combined with ZOH sample-rate rescaling factors.
   */
  public telemetry(): TelemetrySnapshot {
    return this.handle.telemetry();
  }

  public dispose(): void {
    this.currentSpace.dispose();
    this.handle.dispose();
  }

  public [Symbol.dispose](): void {
    this.dispose();
  }
}

export const DnamProcessor = DnamEngine;
export type DnamProcessor<TControls extends ControlMap = ControlMap> = DnamEngine<TControls>;
