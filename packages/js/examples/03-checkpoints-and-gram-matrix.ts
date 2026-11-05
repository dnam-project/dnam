/**
 * Example 03 — Training Checkpoints (`.dnam-ckpt`) & `uplo='U'` Packed `GramMatrix`.
 *
 * Run via:
 *   node packages/js/examples/03-checkpoints-and-gram-matrix.ts
 */

import { Checkpoint, GramMatrix } from "../dist/node.js";

using ckpt = Checkpoint.reference({ includeCalibStats: true });
console.log(
  `Loaded checkpoint: magic=${ckpt.preamble.magic}, stems=${String(ckpt.stems.length)}, phase1Clamps=${String(ckpt.auditReport?.phase1Clamps.join(","))}`,
);

// Verify Arena 4 stem CRC-32C digests and DNCA calibration statistics header
ckpt.verifyStems();
const numPoints = ckpt.verifyCalibrationStats(68_096);
console.log(`Verified stem arenas and DNCA calibration header (${String(numPoints)} points)`);

// Pack and unpack a symmetric 4x4 Gram matrix in upper-triangular column-major (`uplo='U'`) format
const dense4x4 = new Float64Array([4, 1, 0, 0, 1, 3, 1, 0, 0, 1, 2, 1, 0, 0, 1, 2]);
const gram = GramMatrix.fromDense(dense4x4, 4);
console.log(
  `GramMatrix D=4: packedLength=${String(gram.packedLength)}, 2x2 principal submatrix=[${gram.toDense({ tierDimension: 2 }).join(", ")}]`,
);

// Export runtime `.dnam` Model with training-only stems stripped
using runtimeModel = ckpt.toModel();
console.log(
  `Exported runtime Model: magic=${runtimeModel.preamble.magic}, maxTier=${runtimeModel.maxTier}, bytes=${String(runtimeModel.toBytes().byteLength)}`,
);
