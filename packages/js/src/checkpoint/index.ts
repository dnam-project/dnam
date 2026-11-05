/**
 * Curated exports for `@dnam-project/dnam/checkpoint` (`ref[sdk.ts.layout.curated-exports]`,
 * `ref[sdk.ts.model.containers-and-checkpoints]`).
 *
 * @module
 */

export type {
  AuditReport,
  ExtractedPhysics,
  StemProvenance,
  StemRegistryEntry,
} from "../schema/checkpoint.js";
export {
  Checkpoint,
  type DnckLayoutInfo,
  type ReferenceCheckpointOptions,
} from "./checkpoint.js";
export { GramMatrix, uploUPackedLength } from "./gram-matrix.js";
