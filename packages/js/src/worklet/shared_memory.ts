/**
 * Lock-free 256-byte `SharedArrayBuffer` (`Atomics`) ring & state bridge between the main UI thread
 * (`DnamWorkletController`) and the real-time WebAudio thread (`DnamAudioWorkletProcessor`).
 *
 * Zero external dependencies (`zod` and `node:*` are strictly forbidden in `AudioWorkletGlobalScope`).
 */

export const DNAM_SAB_BYTE_LENGTH = 256;

export const SAB_I32_CTRL_SEQ = 0;
export const SAB_I32_TARGET_TIER = 1;
export const SAB_I32_RESET_SEQ = 2;
export const SAB_I32_IMMEDIATE_MASK = 3;
export const SAB_I32_TELEMETRY_SEQ = 4;
export const SAB_I32_ACTIVE_TIER = 5;
export const SAB_I32_ACTIVE_CLAMP_MASK = 6;
export const SAB_I32_CTRL_DECIMATION_HOST = 7;

export const SAB_F32_TARGET_CONTROLS_OFFSET = 8; // 8..24 (16 f32s)
export const SAB_F32_CLAMPED_CONTROLS_OFFSET = 24; // 24..40 (16 f32s)
export const SAB_F32_Z_HEAD_OFFSET = 40; // 40..44 (4 f32s)
export const SAB_F32_ENV_STATES_OFFSET = 44; // 44..48 (4 f32s)
export const SAB_F32_LFO_HARMONICS_OFFSET = 48; // 48..56 (8 f32s)
export const SAB_F32_MAX_ADAA_DRIVE = 56;
export const SAB_F32_MAX_POLE_RADIUS = 57;
export const SAB_F32_MAX_EFFECTIVE_LOOP_GAIN = 58;
export const SAB_F32_TAIL_TIME_MS = 59;

export interface WorkletTelemetryDigest {
  activeTier: number;
  activeClampMask: number;
  controlDecimationHost: number;
  maxAdaaDrive: number;
  maxPoleRadius: number;
  maxEffectiveLoopGain: number;
  tailTimeMs: number;
  zHead: [number, number, number, number];
  envStates: [number, number, number, number];
  lfoHarmonics: [number, number, number, number, number, number, number, number];
  clampedControls: number[];
}

export type WorkletInboundMessage =
  | {
      type: "setControl";
      index: number;
      value: number;
      immediate?: boolean | undefined;
    }
  | {
      type: "setControls";
      entries: { index: number; value: number }[];
      immediate?: boolean | undefined;
    }
  | {
      type: "setTier";
      tier: number;
    }
  | {
      type: "reset";
    }
  | {
      type: "reload";
      containerBytes: Uint8Array;
      tier?: number | undefined;
      sampleRateHz?: number | undefined;
    }
  | {
      type: "requestTelemetry";
    };

export type WorkletOutboundMessage =
  | {
      type: "telemetry";
      telemetry: WorkletTelemetryDigest;
    }
  | {
      type: "reloaded";
      activeTier: number;
    }
  | {
      type: "error";
      message: string;
    };

/**
 * Checks whether `SharedArrayBuffer` with `Atomics` is usable in the current environment.
 */
export function isSharedArrayBufferSupported(): boolean {
  if (typeof SharedArrayBuffer === "undefined" || typeof Atomics === "undefined") {
    return false;
  }
  if (typeof globalThis.crossOriginIsolated === "boolean") {
    return globalThis.crossOriginIsolated;
  }
  return true;
}

/**
 * Allocates a 256-byte `SharedArrayBuffer` (or standard `ArrayBuffer` when `allowFallbackBuffer` is true).
 */
export function createDnamSharedBuffer(
  initialTier = 1,
  initialControls?: ArrayLike<number>,
  allowFallbackBuffer = false,
): SharedArrayBuffer | ArrayBuffer {
  const buf =
    typeof SharedArrayBuffer !== "undefined" &&
    (isSharedArrayBufferSupported() || allowFallbackBuffer)
      ? new SharedArrayBuffer(DNAM_SAB_BYTE_LENGTH)
      : new ArrayBuffer(DNAM_SAB_BYTE_LENGTH);
  const bridge = new DnamSharedBridge(buf);
  bridge.writeTier(initialTier);
  if (initialControls) {
    for (let i = 0; i < Math.min(16, initialControls.length); i++) {
      bridge.writeControl(i, initialControls[i] ?? 0, true);
    }
  }
  return buf;
}

/**
 * Lock-free reader/writer over the 256-byte `SharedArrayBuffer` control & telemetry block.
 */
export class DnamSharedBridge {
  public readonly buffer: SharedArrayBuffer | ArrayBuffer;
  private readonly i32: Int32Array;
  private readonly f32: Float32Array;
  private readonly useAtomics: boolean;

  public constructor(buffer: SharedArrayBuffer | ArrayBuffer) {
    if (buffer.byteLength < DNAM_SAB_BYTE_LENGTH) {
      throw new Error(
        `DnamSharedBridge buffer must be at least ${String(DNAM_SAB_BYTE_LENGTH)} bytes`,
      );
    }
    this.buffer = buffer;
    this.i32 = new Int32Array(buffer, 0, 64);
    this.f32 = new Float32Array(buffer, 0, 64);
    this.useAtomics =
      typeof SharedArrayBuffer !== "undefined" &&
      buffer instanceof SharedArrayBuffer &&
      typeof Atomics !== "undefined";
  }

  private loadI32(slot: number): number {
    if (this.useAtomics) {
      return Atomics.load(this.i32, slot);
    }
    return this.i32[slot] ?? 0;
  }

  private storeI32(slot: number, value: number): void {
    if (this.useAtomics) {
      Atomics.store(this.i32, slot, value);
    } else {
      this.i32[slot] = value;
    }
  }

  private addI32(slot: number, delta: number): number {
    if (this.useAtomics) {
      return Atomics.add(this.i32, slot, delta);
    }
    const prev = this.i32[slot] ?? 0;
    this.i32[slot] = (prev + delta) | 0;
    return prev;
  }

  private orI32(slot: number, mask: number): number {
    if (this.useAtomics) {
      return Atomics.or(this.i32, slot, mask);
    }
    const prev = this.i32[slot] ?? 0;
    this.i32[slot] = prev | mask;
    return prev;
  }

  private exchangeI32(slot: number, value: number): number {
    if (this.useAtomics) {
      return Atomics.exchange(this.i32, slot, value);
    }
    const prev = this.i32[slot] ?? 0;
    this.i32[slot] = value;
    return prev;
  }

  public writeControl(ctrlIdx: number, value: number, immediate = false): void {
    if (ctrlIdx < 0 || ctrlIdx >= 16) {
      return;
    }
    this.f32[SAB_F32_TARGET_CONTROLS_OFFSET + ctrlIdx] = value;
    if (immediate) {
      this.orI32(SAB_I32_IMMEDIATE_MASK, 1 << ctrlIdx);
    }
    this.addI32(SAB_I32_CTRL_SEQ, 1);
  }

  public writeTier(tier: number): void {
    this.storeI32(SAB_I32_TARGET_TIER, tier);
    this.addI32(SAB_I32_CTRL_SEQ, 1);
  }

  public triggerReset(): void {
    this.addI32(SAB_I32_RESET_SEQ, 1);
    this.addI32(SAB_I32_CTRL_SEQ, 1);
  }

  public getCtrlSeq(): number {
    return this.loadI32(SAB_I32_CTRL_SEQ);
  }

  public getResetSeq(): number {
    return this.loadI32(SAB_I32_RESET_SEQ);
  }

  public getTargetTier(): number {
    return this.loadI32(SAB_I32_TARGET_TIER);
  }

  public consumeImmediateMask(): number {
    return this.exchangeI32(SAB_I32_IMMEDIATE_MASK, 0);
  }

  public readTargetControl(ctrlIdx: number): number {
    return this.f32[SAB_F32_TARGET_CONTROLS_OFFSET + ctrlIdx] ?? 0;
  }

  public publishTelemetry(digest: WorkletTelemetryDigest): void {
    this.storeI32(SAB_I32_ACTIVE_TIER, digest.activeTier);
    this.storeI32(SAB_I32_ACTIVE_CLAMP_MASK, digest.activeClampMask);
    this.storeI32(SAB_I32_CTRL_DECIMATION_HOST, digest.controlDecimationHost);
    this.f32[SAB_F32_MAX_ADAA_DRIVE] = digest.maxAdaaDrive;
    this.f32[SAB_F32_MAX_POLE_RADIUS] = digest.maxPoleRadius;
    this.f32[SAB_F32_MAX_EFFECTIVE_LOOP_GAIN] = digest.maxEffectiveLoopGain;
    this.f32[SAB_F32_TAIL_TIME_MS] = digest.tailTimeMs;
    for (let i = 0; i < 4; i++) {
      this.f32[SAB_F32_Z_HEAD_OFFSET + i] = digest.zHead[i] ?? 0;
      this.f32[SAB_F32_ENV_STATES_OFFSET + i] = digest.envStates[i] ?? 0;
    }
    for (let i = 0; i < 8; i++) {
      this.f32[SAB_F32_LFO_HARMONICS_OFFSET + i] = digest.lfoHarmonics[i] ?? 0;
    }
    for (let i = 0; i < 16; i++) {
      this.f32[SAB_F32_CLAMPED_CONTROLS_OFFSET + i] = digest.clampedControls[i] ?? 0;
    }
    this.addI32(SAB_I32_TELEMETRY_SEQ, 1);
  }

  public readTelemetry(): WorkletTelemetryDigest {
    const activeTier = this.loadI32(SAB_I32_ACTIVE_TIER);
    const activeClampMask = this.loadI32(SAB_I32_ACTIVE_CLAMP_MASK);
    const controlDecimationHost = this.loadI32(SAB_I32_CTRL_DECIMATION_HOST);
    const maxAdaaDrive = this.f32[SAB_F32_MAX_ADAA_DRIVE] ?? 0;
    const maxPoleRadius = this.f32[SAB_F32_MAX_POLE_RADIUS] ?? 0;
    const maxEffectiveLoopGain = this.f32[SAB_F32_MAX_EFFECTIVE_LOOP_GAIN] ?? 0;
    const tailTimeMs = this.f32[SAB_F32_TAIL_TIME_MS] ?? 0;
    const zHead: [number, number, number, number] = [
      this.f32[SAB_F32_Z_HEAD_OFFSET] ?? 0,
      this.f32[SAB_F32_Z_HEAD_OFFSET + 1] ?? 0,
      this.f32[SAB_F32_Z_HEAD_OFFSET + 2] ?? 0,
      this.f32[SAB_F32_Z_HEAD_OFFSET + 3] ?? 0,
    ];
    const envStates: [number, number, number, number] = [
      this.f32[SAB_F32_ENV_STATES_OFFSET] ?? 0,
      this.f32[SAB_F32_ENV_STATES_OFFSET + 1] ?? 0,
      this.f32[SAB_F32_ENV_STATES_OFFSET + 2] ?? 0,
      this.f32[SAB_F32_ENV_STATES_OFFSET + 3] ?? 0,
    ];
    const lfoHarmonics: [number, number, number, number, number, number, number, number] = [
      this.f32[SAB_F32_LFO_HARMONICS_OFFSET] ?? 0,
      this.f32[SAB_F32_LFO_HARMONICS_OFFSET + 1] ?? 0,
      this.f32[SAB_F32_LFO_HARMONICS_OFFSET + 2] ?? 0,
      this.f32[SAB_F32_LFO_HARMONICS_OFFSET + 3] ?? 0,
      this.f32[SAB_F32_LFO_HARMONICS_OFFSET + 4] ?? 0,
      this.f32[SAB_F32_LFO_HARMONICS_OFFSET + 5] ?? 0,
      this.f32[SAB_F32_LFO_HARMONICS_OFFSET + 6] ?? 0,
      this.f32[SAB_F32_LFO_HARMONICS_OFFSET + 7] ?? 0,
    ];
    const clampedControls: number[] = [];
    for (let i = 0; i < 16; i++) {
      clampedControls.push(this.f32[SAB_F32_CLAMPED_CONTROLS_OFFSET + i] ?? 0);
    }
    return {
      activeTier,
      activeClampMask,
      controlDecimationHost,
      maxAdaaDrive,
      maxPoleRadius,
      maxEffectiveLoopGain,
      tailTimeMs,
      zHead,
      envStates,
      lfoHarmonics,
      clampedControls,
    };
  }
}
