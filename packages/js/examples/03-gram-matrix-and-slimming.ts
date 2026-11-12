/**
 * Example 03 — Tier Slimming, In-Place `DMU1` Header Editing & `uplo='U'` Packed `GramMatrix`.
 *
 * Run via:
 *   node packages/js/examples/03-gram-matrix-and-slimming.ts
 */

import { GramMatrix, Model } from "../dist/node.js";

using ultraModel = Model.reference({ supportsFreeze: true });
using liteModel = ultraModel.withMaxTier("lite");
console.log(
  `Slimmed runtime Model: magic=${liteModel.preamble.magic}, maxTier=${liteModel.maxTier}, bytes=${String(liteModel.toBytes().byteLength)}`,
);

// Pack and unpack a symmetric 4x4 Gram matrix in upper-triangular column-major (`uplo='U'`) format
const dense4x4 = new Float64Array([4, 1, 0, 0, 1, 3, 1, 0, 0, 1, 2, 1, 0, 0, 1, 2]);
const gram = GramMatrix.fromDense(dense4x4, 4);
console.log(
  `GramMatrix D=4: packedLength=${String(gram.packedLength)}, 2x2 principal submatrix=[${gram.toDense({ tierDimension: 2 }).join(", ")}]`,
);
