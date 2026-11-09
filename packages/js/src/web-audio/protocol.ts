/**
 * Zero-dependency `MessagePort` command & telemetry protocol for WebAudio `AudioWorklet`
 * (`ref[sdk.ts.web-audio.realtime-bridge]`).
 *
 * @module
 */

/**
 * Initialization options passed via `AudioWorkletNodeOptions.processorOptions`.
 */
export interface DnamWorkletProcessorOptions {
  readonly wasmModule: WebAssembly.Module;
  readonly containerBytes: Uint8Array;
  readonly tier?: number;
  readonly sampleRate?: number;
  readonly backend?: string;
  readonly maxSupportedTier?: number;
  readonly sharedBuffer?: SharedArrayBuffer;
  readonly wasmMemory?: WebAssembly.Memory;
  readonly initialMemoryPages?: number;
  readonly maximumMemoryPages?: number;
  readonly sharedWasmMemory?: boolean;
  readonly telemetryIntervalBlocks?: number;
}

/**
 * Commands sent from `DnamAudioNode` (main thread) to `DnamAudioWorkletProcessor` over `MessagePort`.
 */
export type WorkletInboundMessage =
  | {
      readonly type: "set-param";
      readonly index: number;
      readonly value: number;
      readonly immediate: boolean;
    }
  | {
      readonly type: "set-params-batch";
      readonly indices: readonly number[];
      readonly values: readonly number[];
      readonly immediate: boolean;
    }
  | { readonly type: "reset-controls" }
  | { readonly type: "reset" }
  | { readonly type: "apply-preset"; readonly presetId: string }
  | { readonly type: "set-bypass"; readonly bypassed: boolean; readonly trails: boolean }
  | { readonly type: "set-freeze"; readonly frozen: boolean; readonly amount?: number }
  | { readonly type: "set-tier"; readonly tier: number }
  | { readonly type: "set-max-supported-tier"; readonly tier: number }
  | { readonly type: "set-width"; readonly width: number }
  | {
      readonly type: "sync-tempo";
      readonly bpm: number;
      readonly subdivision?: string;
      readonly foldOctave: boolean;
    }
  | { readonly type: "tap-tempo"; readonly subdivision?: string; readonly foldOctave: boolean }
  | {
      readonly type: "reload";
      readonly containerBytes: Uint8Array;
      readonly tier?: number;
      readonly sampleRate?: number;
    };

/**
 * Messages sent from `DnamAudioWorkletProcessor` (audio thread) to `DnamAudioNode` over `MessagePort`.
 */
export type WorkletOutboundMessage =
  | { readonly type: "telemetry"; readonly data: Float64Array }
  | { readonly type: "error"; readonly message: string };
