/**
 * Curated exports for `@dnam-project/dnam/schema` (`ref[sdk.ts.layout.curated-exports]`,
 * `ref[sdk.ts.schema.camel-case-codec-parity]`).
 *
 * @module
 */

export {
  type CanonicalJsonOptions,
  canonicalJson,
} from "./canonical-json.js";
export {
  camelToSnakeKey,
  camelToWireObject,
  pruneRustSkipSerializingIf,
  snakeToCamelKey,
  wireToCamelObject,
} from "./case-codec.js";
export type {
  AuditReport,
  ExtractedPhysics,
  StemProvenance,
  StemRegistryEntry,
} from "./checkpoint.js";
export {
  decodeHeader,
  encodeHeader,
  headerToJcs,
  type ModelHeader,
  parseSchemaVersion,
  type SchemaVersion,
} from "./header.js";
export {
  type BlendMode,
  blendModes,
  type IoConfig,
  type LevelingConfig,
  type LevelingMetric,
  levelingMetrics,
  type ModelCapabilities,
  type SlimmableTiersConfig,
  type TierProfile,
} from "./io.js";
export {
  type CalibrationRole,
  calibrationRoles,
  type ParameterCalibration,
  type ParameterDescriptor,
  type ParameterKind,
  type ParameterUiHints,
  type Preset,
  parameterKinds,
} from "./parameters.js";
export type {
  AuthorizedDevice,
  CreatorSignature,
  DatasetProvenance,
  DistributionPolicy,
  MarketplaceReceipt,
  ModelMetadata,
  ModelSignatures,
  OemDeviceCertificate,
  RoyaltySplit,
} from "./signatures.js";
export {
  type ControlBusConfig,
  DEFAULT_ALLPASS_DELAYS,
  type RouterConfig,
  type Stage1PreFxConfig,
  type Stage2CascadeConfig,
  type Stage3CabinetKautzConfig,
  type Stage4SpatialTankConfig,
  type TensorSliceEntry,
} from "./stages.js";
