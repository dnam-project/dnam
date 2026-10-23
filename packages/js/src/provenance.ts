/**
 * Cryptographic Provenance, RFC 8785 JCS SHA-256, `Ed25519` Signing Messages,
 * `ChaCha20` 64B Cache-Line Arena Encryption, Dual-Mechanism Buyer Watermarking (`Test 8H`),
 * and 160-D Gauge-Invariant Structural Hash (`f_struct in R^160`).
 */

import { toUint8Array } from "./container.js";
import { type BackendChoice, resolveDriver } from "./drivers/index.js";
import { validateStandardSchema } from "./schema/base.js";
import {
  type DnamJsonHeader,
  DnamJsonHeaderSchema,
  dnamJsonHeaderToWire,
} from "./schema/header.js";
import {
  type MarketplaceReceipt,
  MarketplaceReceiptSchema,
  marketplaceReceiptToWire,
} from "./schema/signatures.js";

/**
 * Computes the preset-safe RFC 8785 JCS `immutable_header_sha256` hex digest over `header`.
 */
export function computeImmutableHeaderSha256(
  header: DnamJsonHeader | Record<string, unknown> | string,
  backend?: BackendChoice,
): string {
  const driver = resolveDriver(backend);
  const parsed: unknown = typeof header === "string" ? JSON.parse(header) : header;
  const validated = validateStandardSchema(DnamJsonHeaderSchema, parsed, "DnamJsonHeader");
  const wireJson = JSON.stringify(dnamJsonHeaderToWire(validated));
  return driver.immutableHeaderSha256(wireJson);
}

/**
 * Computes the lowercase hex SHA-256 digest of `data`.
 */
export function sha256Hex(
  data: Uint8Array | ArrayBuffer | ArrayBufferView,
  backend?: BackendChoice,
): string {
  const driver = resolveDriver(backend);
  return driver.sha256Hex(toUint8Array(data));
}

/**
 * Builds the canonical `Ed25519` creator signing message for `(immutableHeaderSha256Hex, masterArenaSha256Hex)`.
 */
export function buildCreatorSigningMessage(
  immutableHeaderSha256Hex: string,
  masterArenaSha256Hex: string,
  backend?: BackendChoice,
): Uint8Array {
  const driver = resolveDriver(backend);
  return driver.buildCreatorSigningMessage(immutableHeaderSha256Hex, masterArenaSha256Hex);
}

/**
 * Builds the canonical `Ed25519` marketplace receipt signing message.
 */
export function buildMarketplaceSigningMessage(
  immutableHeaderSha256Hex: string,
  masterArenaSha256Hex: string,
  receipt: MarketplaceReceipt | Record<string, unknown> | string,
  backend?: BackendChoice,
): Uint8Array {
  const driver = resolveDriver(backend);
  const parsed: unknown = typeof receipt === "string" ? JSON.parse(receipt) : receipt;
  const validated = validateStandardSchema(MarketplaceReceiptSchema, parsed, "MarketplaceReceipt");
  const receiptJson = JSON.stringify(marketplaceReceiptToWire(validated));
  return driver.buildMarketplaceSigningMessage(
    immutableHeaderSha256Hex,
    masterArenaSha256Hex,
    receiptJson,
  );
}

/**
 * Applies RFC 8439 `ChaCha20` 64-byte cache-line XOR encryption/decryption over `arenaBytes`.
 */
export function chacha20XorArena(
  arenaBytes: Uint8Array | ArrayBuffer | ArrayBufferView,
  key32: Uint8Array | ArrayBuffer | ArrayBufferView,
  nonce12: Uint8Array | ArrayBuffer | ArrayBufferView,
  startBlockCounter = 0,
  backend?: BackendChoice,
): Uint8Array {
  const driver = resolveDriver(backend);
  return driver.chacha20XorArena(
    toUint8Array(arenaBytes),
    toUint8Array(key32),
    toUint8Array(nonce12),
    startBlockCounter,
  );
}

/**
 * Applies Mechanism A (48-channel RG-LRU reciprocal scaling + SO(4) SVD head rotation) to `arenaBytes`
 * with exact zero audio difference (`Test 8H`).
 */
export function applyIsometricGaugeWatermark(
  arenaBytes: Uint8Array | ArrayBuffer | ArrayBufferView,
  kappa48: Float32Array | readonly number[],
  qBuyer4x4: Float32Array | readonly number[],
  backend?: BackendChoice,
): Uint8Array {
  const driver = resolveDriver(backend);
  const k32 = kappa48 instanceof Float32Array ? kappa48 : new Float32Array(kappa48);
  const q32 = qBuyer4x4 instanceof Float32Array ? qBuyer4x4 : new Float32Array(qBuyer4x4);
  return driver.applyIsometricGaugeWatermark(toUint8Array(arenaBytes), k32, q32);
}

/**
 * Generates a 4,608-chip Tardos binary spread-spectrum watermark code $\{-1.0, +1.0\}^{4608}$.
 */
export function generateTardosCode(
  marketplaceSecret32: Uint8Array | ArrayBuffer | ArrayBufferView,
  modelId: string,
  licenseId: string,
  backend?: BackendChoice,
): Float32Array {
  const driver = resolveDriver(backend);
  return driver.generateTardosCode(toUint8Array(marketplaceSecret32), modelId, licenseId);
}

/**
 * Embeds Mechanism B (4,608-weight Tardos spread-spectrum watermark) into `arenaBytes`.
 */
export function applyTardosWatermark(
  arenaBytes: Uint8Array | ArrayBuffer | ArrayBufferView,
  code4608: Float32Array | readonly number[],
  epsilon = 5e-4,
  backend?: BackendChoice,
): Uint8Array {
  const driver = resolveDriver(backend);
  const c32 = code4608 instanceof Float32Array ? code4608 : new Float32Array(code4608);
  return driver.applyTardosWatermark(toUint8Array(arenaBytes), c32, epsilon);
}

/**
 * Computes the Tardos matched-filter Z-score ($\sigma$) of `suspectArena` against `masterArena` and `code4608`.
 */
export function scoreTardosWatermark(
  masterArena: Uint8Array | ArrayBuffer | ArrayBufferView,
  suspectArena: Uint8Array | ArrayBuffer | ArrayBufferView,
  code4608: Float32Array | readonly number[],
  backend?: BackendChoice,
): number {
  const driver = resolveDriver(backend);
  const c32 = code4608 instanceof Float32Array ? code4608 : new Float32Array(code4608);
  return driver.scoreTardosWatermark(toUint8Array(masterArena), toUint8Array(suspectArena), c32);
}

/**
 * Extracts the 160-Dimensional Gauge-Invariant Structural Hash vector $\mathbf{f}_{\text{struct}} \in \mathbb{R}^{160}$
 * from `arenaBytes`.
 */
export function extractStructuralHash160d(
  arenaBytes: Uint8Array | ArrayBuffer | ArrayBufferView,
  backend?: BackendChoice,
): Float64Array {
  const driver = resolveDriver(backend);
  return driver.extractStructuralHash(toUint8Array(arenaBytes));
}

/**
 * Computes the cosine similarity $\cos(\mathbf{a}, \mathbf{b}) \in [-1, +1]$ between two 160-D structural hash vectors.
 */
export function structuralCosineSimilarity(
  a: Float64Array | readonly number[],
  b: Float64Array | readonly number[],
  backend?: BackendChoice,
): number {
  const driver = resolveDriver(backend);
  const a64 = a instanceof Float64Array ? a : new Float64Array(a);
  const b64 = b instanceof Float64Array ? b : new Float64Array(b);
  return driver.structuralCosineSimilarity(a64, b64);
}
