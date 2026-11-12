/// <reference lib="esnext.disposable" />

/**
 * Browser / WebAssembly `SIMD128` entrypoint for `@dnam-project/dnam/browser`
 * (`ref[sdk.ts.api.sync-init]`, `ref[sdk.ts.layout.dependency-direction]`).
 *
 * Automatically initializes the WebAssembly SIMD128 driver via top-level `await` when
 * `dnam.wasm` is reachable via `fetch`, while allowing explicit override via `await initWasm(...)`.
 *
 * @module
 */

import { initWasm } from "./internal/native/load.js";

try {
  await initWasm();
} catch {
  // In custom hosting or test environments without a relative ./dnam.wasm asset,
  // callers initialize explicitly via `await initWasm({ module | bytes | url })`.
}

export {
  type ActiveBackend,
  type AudioBlock,
  type AudioInput,
  type Backend,
  type BounceOptions,
  type BuildModelOptions,
  type BypassOptions,
  backends,
  type ClampedParameterEntry,
  type ClampReport,
  ControlSpace,
  clampParameters,
  crc32c,
  DnamError,
  type DnamErrorOptions,
  decodeHeader,
  defineModelTypes,
  Engine,
  type EngineOptions,
  type ErrorCode,
  type ErrorDetail,
  encodeHeader,
  errorCodes,
  type FreezeOptions,
  GramMatrix,
  type HostMixOptions,
  headerToJcs,
  type InferParameterMap,
  type InitWasmOptions,
  initWasm,
  isDnamError,
  Model,
  type ModelHeader,
  type ModelLoadOptions,
  type ModelSource,
  type OverflowFlags,
  Parameter,
  type ParameterCalibration,
  type ParameterDescriptor,
  type ParameterKind,
  ParameterMap,
  type ParameterUiHints,
  type ParameterValueMap,
  type PreambleInfo,
  type Preset,
  type ProcessAsyncOptions,
  type ProcessOptions,
  type ProtectionProfile,
  parseSchemaVersion,
  protectionProfiles,
  type ReferenceModelOptions,
  type ReloadOptions,
  type RenderOptions,
  type SchemaVersion,
  type SetParameterOptions,
  type StateScope,
  type StreamOptions,
  type Subdivision,
  subdivisions,
  type TapTempoResult,
  type TelemetrySnapshot,
  TempoController,
  type TempoSyncOptions,
  type TempoTapOptions,
  type Tier,
  type TierArenas,
  type TierInput,
  type TypedParameterEntry,
  tiers,
  uploUPackedLength,
  type V2OpenLayoutInfo,
  type WasmMemoryOptions,
} from "./public-exports.js";
