/**
 * Curated exports for `@dnam-project/dnam/web-audio` (`ref[sdk.ts.layout.curated-exports]`,
 * `ref[sdk.ts.web-audio.realtime-bridge]`).
 *
 * @module
 */

export {
  createWasmMemory,
  isSharedWasmMemorySupported,
  WASM_DEFAULT_INITIAL_PAGES,
  WASM_DEFAULT_MAXIMUM_PAGES,
  WASM_MAX_PAGES,
  WASM_MIN_INITIAL_PAGES,
  type WasmMemoryOptions,
} from "../internal/native/wasm.js";
export {
  type AudioContextLike,
  DnamAudioNode,
  type DnamAudioNodeOptions,
  registerDnamProcessor,
} from "./audio-node.js";
export {
  DNAM_WORKLET_PROCESSOR_NAME,
  DnamAudioWorkletProcessor,
} from "./processor.js";
export type {
  DnamWorkletProcessorOptions,
  WorkletInboundMessage,
  WorkletOutboundMessage,
} from "./protocol.js";
export {
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
  WORKLET_SAB_BYTE_LENGTH,
  WorkletSharedState,
} from "./shared-state.js";
