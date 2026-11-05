/**
 * Curated exports for `@dnam-project/dnam/tempo` (`ref[sdk.ts.layout.curated-exports]`).
 *
 * @module
 */

export {
  type Subdivision,
  type SubdivisionPhaseResult,
  subdivisions,
  type TempoCalibrationResult,
  type ZohParams,
} from "./subdivisions.js";
export {
  computeZoh,
  controlStride,
  controlTickRatio,
  evalCalibration3Node,
  invertTempoCalibration,
  subdivisionPhase,
  type TapTempoResult,
  type TempoBackend,
  TempoController,
  type TempoSyncOptions,
  type TempoTapOptions,
} from "./tempo.js";
