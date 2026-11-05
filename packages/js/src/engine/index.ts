/**
 * Curated exports for `src/engine/` (`ref[sdk.ts.layout.curated-exports]`).
 *
 * @module
 */

export {
  type BounceOptions,
  Engine,
  type EngineOptions,
  type HostMixOptions,
  type ProcessAsyncOptions,
  type ProcessOptions,
  type ReloadOptions,
  type RenderOptions,
} from "./engine.js";
export {
  type BypassOptions,
  createStateScope,
  type FreezeOptions,
} from "./state-scopes.js";
export { createEngineTransformStream, type StreamOptions } from "./stream.js";
export {
  decodeTelemetryF64,
  type OverflowFlags,
  type TelemetrySnapshot,
} from "./telemetry.js";
