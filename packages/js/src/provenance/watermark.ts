/**
 * Dual-mechanism buyer watermarking (Isometric Gauge + Coupled Tardos-7680), 224-D Gauge-Invariant
 * Structural Hash, Stage III tier-local fingerprint, and HMAC-SHA256 seed derivation
 * (`ref[sdk.ts.provenance.v2-security]`, `ref[sec.watermark.tardos-code]`,
 * `ref[sec.anticlone.stage3-fingerprint]`).
 *
 * @module
 */

import { coerceBytes } from "../internal/coerce.js";
import { getActiveDriver } from "../internal/native/load.js";
import { Model } from "../model/model.js";
import type { ParameterValueMap } from "../parameters/typed-model.js";
import { hexToBytes } from "./keys.js";

/**
 * Options for `deriveWatermarkSeed(options)`.
 */
export interface WatermarkSeedSpec {
  readonly key32: Uint8Array | string;
  readonly marketplacePubkey32: Uint8Array | string;
  readonly watermarkKeyId16: Uint8Array | string;
  readonly creatorSignedManifestSha25632: Uint8Array | string;
  readonly modelId16: Uint8Array | string;
  readonly watermarkProfileSha25632: Uint8Array | string;
  readonly owningTier: number;
  readonly licenseId16?: Uint8Array | string;
  readonly buyerId16?: Uint8Array | string;
}

function toBytes(v: Uint8Array | string): Uint8Array {
  return typeof v === "string" ? hexToBytes(v) : v;
}

/**
 * Applies Mechanism A (64-channel modal/Kautz reciprocal scaling + SO(4) SVD head rotation)
 * to `arenaBytes` (`ref[sec.watermark.reciprocal-gauge]`).
 */
export function applyGaugeWatermark(options: {
  readonly arenaBytes: Uint8Array | ArrayBuffer | ArrayBufferView;
  readonly kappa64: Float32Array | readonly number[];
  readonly qBuyer4x4: Float32Array | readonly number[];
}): Uint8Array {
  const kappa =
    options.kappa64 instanceof Float32Array ? options.kappa64 : new Float32Array(options.kappa64);
  const q =
    options.qBuyer4x4 instanceof Float32Array
      ? options.qBuyer4x4
      : new Float32Array(options.qBuyer4x4);
  return getActiveDriver().applyIsometricGaugeWatermark(coerceBytes(options.arenaBytes), kappa, q);
}

/**
 * Generates a 7,680-chip Tardos binary spread-spectrum watermark code `{-1.0, +1.0}^7680`
 * (`ref[sec.watermark.tardos-code]`).
 */
export function generateTardosCode(options: {
  readonly marketplaceSecret: Uint8Array | string;
  readonly modelId: string;
  readonly licenseId: string;
}): Float32Array {
  return getActiveDriver().generateTardosCode(
    toBytes(options.marketplaceSecret),
    options.modelId,
    options.licenseId,
  );
}

/**
 * Embeds Mechanism B (7,680-weight Tardos spread-spectrum watermark) into `arenaBytes`.
 */
export function applyTardosWatermark(options: {
  readonly arenaBytes: Uint8Array | ArrayBuffer | ArrayBufferView;
  readonly code7680: Float32Array;
  readonly epsilon?: number;
}): Uint8Array {
  return getActiveDriver().applyTardosWatermark(
    coerceBytes(options.arenaBytes),
    options.code7680,
    options.epsilon ?? 1e-4,
  );
}

/**
 * Computes the Tardos matched-filter Z-score (`sigma`) of `suspectArena` against `masterArena`
 * and `code7680`.
 */
export function scoreTardosWatermark(options: {
  readonly masterArena: Uint8Array | ArrayBuffer | ArrayBufferView;
  readonly suspectArena: Uint8Array | ArrayBuffer | ArrayBufferView;
  readonly code7680: Float32Array;
}): number {
  return getActiveDriver().scoreTardosWatermark(
    coerceBytes(options.masterArena),
    coerceBytes(options.suspectArena),
    options.code7680,
  );
}

/**
 * Extracts the prefix-composable `4 x 56-D = 224-D` Gauge-Invariant Structural Hash vector from
 * a `Model` or `arenaBytes` (`ref[sec.anticlone.stage3-fingerprint]`).
 */
export function extractStructuralHash<
  TParams extends ParameterValueMap = Record<string, number | string>,
>(modelOrArenaBytes: Model<TParams> | Uint8Array | ArrayBuffer | ArrayBufferView): Float64Array {
  const bytes =
    modelOrArenaBytes instanceof Model
      ? modelOrArenaBytes.arenaBytes
      : coerceBytes(modelOrArenaBytes);
  return getActiveDriver().extractStructuralHash(bytes);
}

/**
 * Computes the cosine similarity `cos(a, b)` in `[-1, +1]` between two structural hash vectors.
 */
export function structuralCosineSimilarity(
  a: Float64Array | readonly number[],
  b: Float64Array | readonly number[],
): number {
  const fa = a instanceof Float64Array ? a : new Float64Array(a);
  const fb = b instanceof Float64Array ? b : new Float64Array(b);
  return getActiveDriver().structuralCosineSimilarity(fa, fb);
}

/**
 * Derives the license-scoped or shared HMAC-SHA256 watermark seed (64-char hex).
 */
export function deriveWatermarkSeed(spec: WatermarkSeedSpec): string {
  return getActiveDriver().deriveWatermarkSeedHex(
    toBytes(spec.key32),
    toBytes(spec.marketplacePubkey32),
    toBytes(spec.watermarkKeyId16),
    toBytes(spec.creatorSignedManifestSha25632),
    toBytes(spec.modelId16),
    toBytes(spec.watermarkProfileSha25632),
    spec.owningTier,
    spec.licenseId16 !== undefined ? toBytes(spec.licenseId16) : undefined,
    spec.buyerId16 !== undefined ? toBytes(spec.buyerId16) : undefined,
  );
}

/**
 * Extracts the Stage III tier-local fingerprint component from `earlyFir128b` and `kautzModalBytes`.
 */
export function extractStage3Fingerprint(options: {
  readonly earlyFir128b: Uint8Array | ArrayBuffer | ArrayBufferView;
  readonly kautzModalBytes: Uint8Array | ArrayBuffer | ArrayBufferView;
  readonly addedModes: number;
}): Float64Array {
  return getActiveDriver().extractStage3Fingerprint(
    coerceBytes(options.earlyFir128b),
    coerceBytes(options.kautzModalBytes),
    options.addedModes,
  );
}
