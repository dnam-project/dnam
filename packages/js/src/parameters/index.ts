/**
 * Curated exports for `@dnam-project/dnam/parameters` (`ref[sdk.ts.layout.curated-exports]`,
 * `ref[sdk.ts.parameters.audio-param-semantics]`).
 *
 * @module
 */

export type {
  ParameterCalibration,
  ParameterDescriptor,
  ParameterKind,
  ParameterUiHints,
} from "../schema/parameters.js";
export {
  type ClampedParameterEntry,
  type ClampReport,
  ControlSpace,
  clampParameters,
  defaultNumericValue,
  numericToParameterValue,
  parameterValueToNumeric,
} from "./clamp.js";
export {
  Parameter,
  type ParameterBackend,
  type SetParameterOptions,
} from "./parameter.js";
export { ParameterMap } from "./parameter-map.js";
export {
  defineModelTypes,
  type InferParameterMap,
  type ParameterValueMap,
  type TypedParameterEntry,
} from "./typed-model.js";
