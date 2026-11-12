/**
 * Curated exports for `src/model/` (`ref[sdk.ts.layout.curated-exports]`).
 *
 * @module
 */

export { GramMatrix, uploUPackedLength } from "./gram-matrix.js";
export { type ModelSource, readModelSourceBytes } from "./load.js";
export {
  type BuildModelOptions,
  crc32c,
  Model,
  type ModelLoadOptions,
  type ReferenceModelOptions,
  type V2OpenLayoutInfo,
} from "./model.js";
export {
  extractPreambleAndArenas,
  type PreambleInfo,
  type TierArenas,
} from "./preamble.js";
