/// <reference lib="esnext.disposable" />

/**
 * Shared curated top-level exports for `@dnam-project/dnam` (`ref[sdk.ts.layout.curated-exports]`,
 * `ref[sdk.ts.api.resources-and-data]`).
 *
 * @module
 */

export {
  Checkpoint,
  type DnckLayoutInfo,
  type ReferenceCheckpointOptions,
} from "./checkpoint/checkpoint.js";
export { GramMatrix, uploUPackedLength } from "./checkpoint/gram-matrix.js";
export {
  type BounceOptions,
  Engine,
  type EngineOptions,
  type HostMixOptions,
  type ProcessAsyncOptions,
  type ProcessOptions,
  type ReloadOptions,
  type RenderOptions,
} from "./engine/engine.js";
export type { BypassOptions, FreezeOptions } from "./engine/state-scopes.js";
export type { StreamOptions } from "./engine/stream.js";
export type { OverflowFlags, TelemetrySnapshot } from "./engine/telemetry.js";
export {
  DnamError,
  type DnamErrorOptions,
  type ErrorCode,
  type ErrorDetail,
  errorCodes,
  isDnamError,
} from "./errors.js";
export { initWasm } from "./internal/native/load.js";
export type { InitWasmOptions } from "./internal/native/wasm.js";
export type { ModelSource } from "./model/load.js";
export {
  type BuildModelOptions,
  crc32c,
  Model,
  type ModelLoadOptions,
  type ReferenceModelOptions,
  type V2OpenLayoutInfo,
} from "./model/model.js";
export type { PreambleInfo, TierArenas } from "./model/preamble.js";
export {
  type ClampedParameterEntry,
  type ClampReport,
  ControlSpace,
  clampParameters,
} from "./parameters/clamp.js";
export { Parameter, type SetParameterOptions } from "./parameters/parameter.js";
export { ParameterMap } from "./parameters/parameter-map.js";
export {
  defineModelTypes,
  type InferParameterMap,
  type ParameterValueMap,
  type TypedParameterEntry,
} from "./parameters/typed-model.js";
export {
  decodeHeader,
  encodeHeader,
  headerToJcs,
  type ModelHeader,
  parseSchemaVersion,
  type SchemaVersion,
} from "./schema/header.js";
export type {
  ParameterCalibration,
  ParameterDescriptor,
  ParameterKind,
  ParameterUiHints,
  Preset,
} from "./schema/parameters.js";
export {
  type TapTempoResult,
  TempoController,
  type TempoSyncOptions,
  type TempoTapOptions,
} from "./tempo/tempo.js";
export {
  type ActiveBackend,
  type AudioBlock,
  type AudioInput,
  type Backend,
  backends,
  type ProtectionProfile,
  protectionProfiles,
  type StateScope,
  type Subdivision,
  subdivisions,
  type Tier,
  type TierInput,
  tiers,
} from "./types.js";
