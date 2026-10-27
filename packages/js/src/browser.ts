/**
 * Browser & Edge entry point for `@dnam-project/engine` (`@dnam-project/engine/browser`).
 * Zero `node:*` imports.
 */

export {
  type BuildContainerOptions,
  type BuildReferenceOptions,
  buildDnamContainer,
  buildReferenceContainer,
  computeZohRescaledParams,
  crc32c,
  DnamContainer,
  type FromBytesOptions,
  GramMatrix,
  type SealContainerOptions,
  toUint8Array,
  updateHeaderInPlace,
  validateDnamModel,
  type ZohComputeOptions,
} from "./container.js";
export {
  ControlRack,
  type ControlSink,
  ControlSpace,
  chebyshevLfo,
  clampControlValues,
  LobattoBasis,
  type LobattoEvalOptions,
  Rank4Fm,
  type SetControlOptions,
  trustRegionSoftclip,
} from "./control.js";
export {
  type BackendChoice,
  type DnamControlSpaceHandle,
  type DnamDriver,
  type DnamEngineHandle,
  getCachedWasmBinary,
  getWasmDriver,
  hasWasmDriver,
  initDnamWasm,
  initDnamWasmSync,
  type RawClampResult,
  resolveDriver,
  type ValidatedContainerRaw,
} from "./drivers/index.js";
export {
  type DspScanOptions,
  givensHouseholderMix,
  lagrangeWeights,
  ModulatedUfdn,
  PolyAdaa,
  SelectiveRgLru,
  unitRotor,
} from "./dsp.js";
export {
  type AudioBufferLike,
  DnamEngine,
  type DnamEngineOptions,
  DnamProcessor,
  type ProcessMonoOptions,
  type ProcessPlanarOptions,
  type ProcessStereoOptions,
} from "./engine.js";
export {
  DnamContainerError,
  DnamControlError,
  DnamError,
  DnamSampleRateError,
  DnamValidationError,
  mapTaggedError,
  rethrowDnamError,
} from "./errors.js";
export {
  applyIsometricGaugeWatermark,
  applyTardosWatermark,
  buildCreatorSigningMessage,
  buildMarketplaceSigningMessage,
  chacha20XorArena,
  computeImmutableHeaderSha256,
  ed25519PubkeyHex,
  ed25519SignHex,
  extractStructuralHash160d,
  generateTardosCode,
  type SealDnamContainerOptions,
  type SignOemDeviceCertificateOptions,
  scoreTardosWatermark,
  sealDnamContainer,
  sha256Hex,
  signOemDeviceCertificate,
  structuralCosineSimilarity,
  type UnsealDnamContainerOptions,
  unsealDnamContainer,
  verifyOemDeviceCertificate,
  wrapDeviceCekHex,
  x25519PubkeyHex,
} from "./provenance.js";
export * from "./schema/index.js";
export * from "./worklet/index.js";
