/**
 * Lock-free `SharedArrayBuffer` (`Atomics`) state and telemetry bridge for WebAudio `AudioWorklet`
 * (`ref[sdk.ts.web-audio.realtime-bridge]`).
 *
 * Zero external dependencies (`no node:*`) so it can be bundled into standalone
 * `AudioWorkletGlobalScope` modules.
 *
 * @module
 */

import { TELEMETRY_F64_LEN } from "../internal/native/driver.js";

/**
 * Total byte size of the shared worklet state buffer (`1,024 B`, 64B cache-line aligned).
 */
export const WORKLET_SAB_BYTE_LENGTH = 1024;

// Int32 header indices (bytes 0..64)
const IDX_PARAM_SEQ = 0;
const IDX_DIRTY_PARAM_MASK = 1;
const IDX_IMMEDIATE_PARAM_MASK = 2;
const IDX_CMD_FLAGS = 3;
const IDX_REQUESTED_TIER = 4;
const IDX_TELEMETRY_SEQ = 5;
const IDX_ACTIVE_CLAMP_MASK = 6;

// Command flag bits in `IDX_CMD_FLAGS`
export const CMD_RESET = 1 << 0;
export const CMD_BYPASS_CHANGE = 1 << 1;
export const CMD_BYPASSED = 1 << 2;
export const CMD_TRAILS = 1 << 3;
export const CMD_FREEZE_CHANGE = 1 << 4;
export const CMD_FROZEN = 1 << 5;
export const CMD_TIER_CHANGE = 1 << 6;
export const CMD_WIDTH_CHANGE = 1 << 7;
export const CMD_TEMPO_CHANGE = 1 << 8;
export const CMD_RESET_CONTROLS = 1 << 9;

// Float32 parameter & scalar slots (indices 16..48, bytes 64..192)
const F32_PARAM_OFFSET = 16; // 16 slots: 16..32
const F32_FREEZE_AMOUNT_IDX = 32;
const F32_STEREO_WIDTH_IDX = 33;
const F32_BPM_IDX = 34;

// Float64 telemetry slots (indices 32..88, bytes 256..704)
const F64_TELEMETRY_OFFSET = 32;

/**
 * Returns `true` if `SharedArrayBuffer` and `Atomics` are available and permitted by the
 * current environment (`crossOriginIsolated !== false`).
 */
export function isSharedMemorySupported(): boolean {
  if (typeof SharedArrayBuffer === "undefined" || typeof Atomics === "undefined") {
    return false;
  }
  if (
    typeof globalThis !== "undefined" &&
    "crossOriginIsolated" in globalThis &&
    (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === false
  ) {
    return false;
  }
  try {
    new SharedArrayBuffer(64);
    return true;
  } catch {
    return false;
  }
}

/**
 * Lock-free `SharedArrayBuffer` state bridge shared between `DnamAudioNode` (main thread) and
 * `DnamAudioWorkletProcessor` (audio rendering thread).
 */
export class WorkletSharedState {
  readonly buffer: SharedArrayBuffer;
  readonly #i32: Int32Array;
  readonly #f32: Float32Array;
  readonly #f64: Float64Array;
  #lastReadParamSeq = 0;

  constructor(buffer?: SharedArrayBuffer) {
    this.buffer = buffer ?? new SharedArrayBuffer(WORKLET_SAB_BYTE_LENGTH);
    this.#i32 = new Int32Array(this.buffer, 0, 16);
    this.#f32 = new Float32Array(this.buffer, 0, 64);
    this.#f64 = new Float64Array(this.buffer, 0, 128);
  }

  /**
   * Initializes the default parameter values in the shared buffer before processor startup.
   */
  initDefaults(defaultParams: Float32Array, initialTier: number, initialSampleRate: number): void {
    const n = Math.min(16, defaultParams.length);
    for (let i = 0; i < n; i++) {
      this.#f32[F32_PARAM_OFFSET + i] = defaultParams[i] ?? 0;
    }
    this.#f32[F32_FREEZE_AMOUNT_IDX] = 1.0;
    this.#f32[F32_STEREO_WIDTH_IDX] = 1.0;
    this.#f32[F32_BPM_IDX] = 120.0;
    this.#f64[F64_TELEMETRY_OFFSET + 0] = initialTier;
    this.#f64[F64_TELEMETRY_OFFSET + 2] = initialSampleRate;
    this.#f64[F64_TELEMETRY_OFFSET + 8] = 1.0;
    this.#f64[F64_TELEMETRY_OFFSET + 9] = 120.0;
    Atomics.store(this.#i32, IDX_REQUESTED_TIER, initialTier);
  }

  /**
   * Writer (main thread): schedules a single parameter update.
   */
  writeParameter(index: number, value: number, immediate: boolean): void {
    if (index < 0 || index >= 16) {
      return;
    }
    this.#f32[F32_PARAM_OFFSET + index] = value;
    const bit = 1 << index;
    if (immediate) {
      Atomics.or(this.#i32, IDX_IMMEDIATE_PARAM_MASK, bit);
    }
    Atomics.or(this.#i32, IDX_DIRTY_PARAM_MASK, bit);
    Atomics.add(this.#i32, IDX_PARAM_SEQ, 1);
  }

  /**
   * Writer (main thread): schedules a batch of parameter updates.
   */
  writeParametersBatch(
    indices: readonly number[],
    values: readonly number[],
    immediate: boolean,
  ): void {
    let mask = 0;
    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i] ?? -1;
      if (idx >= 0 && idx < 16) {
        this.#f32[F32_PARAM_OFFSET + idx] = values[i] ?? 0;
        mask |= 1 << idx;
      }
    }
    if (mask === 0) {
      return;
    }
    if (immediate) {
      Atomics.or(this.#i32, IDX_IMMEDIATE_PARAM_MASK, mask);
    }
    Atomics.or(this.#i32, IDX_DIRTY_PARAM_MASK, mask);
    Atomics.add(this.#i32, IDX_PARAM_SEQ, 1);
  }

  /**
   * Reads a parameter's latest clamped value from telemetry (or falls back to the written value).
   */
  readParameter(index: number): number {
    if (index < 0 || index >= 16) {
      return 0;
    }
    return this.#f32[F32_PARAM_OFFSET + index] ?? 0;
  }

  /**
   * Returns the latest 16-bit active clamp bitmask reported by the audio processor.
   */
  readActiveClampMask(): number {
    return Atomics.load(this.#i32, IDX_ACTIVE_CLAMP_MASK) >>> 0;
  }

  /**
   * Writer (main thread): posts a command bitmask to the audio thread.
   */
  writeCommand(
    flagsToSet: number,
    flagsToClear = 0,
    scalars?: {
      readonly tier?: number;
      readonly freezeAmount?: number;
      readonly stereoWidth?: number;
      readonly bpm?: number;
    },
  ): void {
    if (scalars?.tier !== undefined) {
      Atomics.store(this.#i32, IDX_REQUESTED_TIER, scalars.tier);
    }
    if (scalars?.freezeAmount !== undefined) {
      this.#f32[F32_FREEZE_AMOUNT_IDX] = scalars.freezeAmount;
    }
    if (scalars?.stereoWidth !== undefined) {
      this.#f32[F32_STEREO_WIDTH_IDX] = scalars.stereoWidth;
    }
    if (scalars?.bpm !== undefined) {
      this.#f32[F32_BPM_IDX] = scalars.bpm;
    }
    if (flagsToClear !== 0) {
      Atomics.and(this.#i32, IDX_CMD_FLAGS, ~flagsToClear);
    }
    Atomics.or(this.#i32, IDX_CMD_FLAGS, flagsToSet);
    Atomics.add(this.#i32, IDX_PARAM_SEQ, 1);
  }

  /**
   * Reader (audio thread): drains any pending parameter or command updates in O(1) with zero allocations.
   */
  drainUpdatesIntoEngine(target: {
    setControlValue(idx: number, val: number): void;
    setControlImmediate(idx: number, val: number): void;
    resetControlsToDefaults(): void;
    reset(): void;
    setBypass(bypassed: boolean, trails: boolean): void;
    setFreeze(frozen: boolean): void;
    setFreezeAmount(amount: number): void;
    setTier(tier: number): void;
    setStereoWidth(width: number): void;
    syncTempo(bpm: number, sub: undefined, fold: boolean): number;
  }): boolean {
    const seq = Atomics.load(this.#i32, IDX_PARAM_SEQ);
    if (seq === this.#lastReadParamSeq) {
      return false;
    }
    this.#lastReadParamSeq = seq;

    const dirtyMask = Atomics.exchange(this.#i32, IDX_DIRTY_PARAM_MASK, 0);
    const immMask = Atomics.exchange(this.#i32, IDX_IMMEDIATE_PARAM_MASK, 0);
    if (dirtyMask !== 0) {
      for (let i = 0; i < 16; i++) {
        const bit = 1 << i;
        if ((dirtyMask & bit) !== 0) {
          const val = this.#f32[F32_PARAM_OFFSET + i] ?? 0;
          if ((immMask & bit) !== 0) {
            target.setControlImmediate(i, val);
          } else {
            target.setControlValue(i, val);
          }
        }
      }
    }

    const cmd = Atomics.exchange(this.#i32, IDX_CMD_FLAGS, 0);
    if (cmd !== 0) {
      if ((cmd & CMD_RESET_CONTROLS) !== 0) {
        target.resetControlsToDefaults();
      }
      if ((cmd & CMD_RESET) !== 0) {
        target.reset();
      }
      if ((cmd & CMD_BYPASS_CHANGE) !== 0) {
        target.setBypass((cmd & CMD_BYPASSED) !== 0, (cmd & CMD_TRAILS) !== 0);
      }
      if ((cmd & CMD_FREEZE_CHANGE) !== 0) {
        target.setFreezeAmount(this.#f32[F32_FREEZE_AMOUNT_IDX] ?? 1.0);
        target.setFreeze((cmd & CMD_FROZEN) !== 0);
      }
      if ((cmd & CMD_TIER_CHANGE) !== 0) {
        target.setTier(Atomics.load(this.#i32, IDX_REQUESTED_TIER));
      }
      if ((cmd & CMD_WIDTH_CHANGE) !== 0) {
        target.setStereoWidth(this.#f32[F32_STEREO_WIDTH_IDX] ?? 1.0);
      }
      if ((cmd & CMD_TEMPO_CHANGE) !== 0) {
        target.syncTempo(this.#f32[F32_BPM_IDX] ?? 120.0, undefined, false);
      }
    }
    return true;
  }

  /**
   * Writer (audio thread): publishes a 56-slot `f64` telemetry snapshot using a seqlock.
   */
  publishTelemetry(telemetry56: Float64Array, numControls: number): void {
    Atomics.add(this.#i32, IDX_TELEMETRY_SEQ, 1); // odd = writing
    this.#f64.set(telemetry56.subarray(0, TELEMETRY_F64_LEN), F64_TELEMETRY_OFFSET);
    const n = Math.min(16, numControls);
    for (let i = 0; i < n; i++) {
      this.#f32[F32_PARAM_OFFSET + i] = telemetry56[32 + i] ?? 0;
    }
    Atomics.store(this.#i32, IDX_ACTIVE_CLAMP_MASK, (telemetry56[10] ?? 0) >>> 0);
    Atomics.add(this.#i32, IDX_TELEMETRY_SEQ, 1); // even = consistent
  }

  /**
   * Reader (main thread): reads a consistent 56-slot `f64` telemetry snapshot into `out56`.
   */
  readTelemetryInto(out56: Float64Array): void {
    for (let attempt = 0; attempt < 4; attempt++) {
      const seq1 = Atomics.load(this.#i32, IDX_TELEMETRY_SEQ);
      if ((seq1 & 1) !== 0) {
        continue;
      }
      out56.set(this.#f64.subarray(F64_TELEMETRY_OFFSET, F64_TELEMETRY_OFFSET + TELEMETRY_F64_LEN));
      const seq2 = Atomics.load(this.#i32, IDX_TELEMETRY_SEQ);
      if (seq1 === seq2) {
        return;
      }
    }
    out56.set(this.#f64.subarray(F64_TELEMETRY_OFFSET, F64_TELEMETRY_OFFSET + TELEMETRY_F64_LEN));
  }
}
