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
  type OemDeviceCertificate,
  OemDeviceCertificateSchema,
  oemDeviceCertificateToWire,
} from "./schema/signatures.js";

export interface SignOemDeviceCertificateOptions {
  oemSeed32: Uint8Array | ArrayBuffer | ArrayBufferView;
  oemVendor: string;
  modelCode: string;
  deviceIdHash: string;
  pubkeyX25519_32: Uint8Array | ArrayBuffer | ArrayBufferView;
  backend?: BackendChoice | undefined;
}

export interface SealDnamContainerOptions {
  innerDnamBytes: Uint8Array | ArrayBuffer | ArrayBufferView;
  recipientCerts: readonly (OemDeviceCertificate | Record<string, unknown>)[];
  trustedOemPubkeys: Readonly<Record<string, string>>;
  marketplaceSeed32: Uint8Array | ArrayBuffer | ArrayBufferView;
  ephemeralSecret32: Uint8Array | ArrayBuffer | ArrayBufferView;
  cek32: Uint8Array | ArrayBuffer | ArrayBufferView;
  nonce12: Uint8Array | ArrayBuffer | ArrayBufferView;
  backend?: BackendChoice | undefined;
}

export interface UnsealDnamContainerOptions {
  deviceIdHash?: string | undefined;
  deviceSecretX25519?: Uint8Array | ArrayBuffer | ArrayBufferView | undefined;
  trustedMarketplacePubkeyHex?: string | undefined;
  backend?: BackendChoice | undefined;
}

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

/**
 * Derives a 64-character lowercase hex `Ed25519` public key from a 32-byte seed.
 */
export function ed25519PubkeyHex(
  seed32: Uint8Array | ArrayBuffer | ArrayBufferView,
  backend?: BackendChoice,
): string {
  const driver = resolveDriver(backend);
  return driver.ed25519PubkeyHex(toUint8Array(seed32));
}

/**
 * Signs `message` with a 32-byte `Ed25519` seed and returns the 128-character lowercase hex signature.
 */
export function ed25519SignHex(
  seed32: Uint8Array | ArrayBuffer | ArrayBufferView,
  message: Uint8Array | ArrayBuffer | ArrayBufferView,
  backend?: BackendChoice,
): string {
  const driver = resolveDriver(backend);
  return driver.ed25519SignHex(toUint8Array(seed32), toUint8Array(message));
}

/**
 * Derives a 64-character lowercase hex `X25519` Montgomery public key from a 32-byte secret key.
 */
export function x25519PubkeyHex(
  secret32: Uint8Array | ArrayBuffer | ArrayBufferView,
  backend?: BackendChoice,
): string {
  const driver = resolveDriver(backend);
  return driver.x25519PubkeyHex(toUint8Array(secret32));
}

/**
 * Wraps a 32-byte Content Encryption Key (`cek32`) for an OEM device (`devicePubkey32`, `deviceIdHash`)
 * via `X25519` ECDH + `SHA-256` KDF + `ChaCha20-Poly1305` AEAD, returning the 96-character lowercase hex
 * `wrapped_cek` (`32B` ciphertext + `16B` tag).
 */
export function wrapDeviceCekHex(
  ephemeralSecret32: Uint8Array | ArrayBuffer | ArrayBufferView,
  devicePubkey32: Uint8Array | ArrayBuffer | ArrayBufferView,
  deviceIdHash: string,
  cek32: Uint8Array | ArrayBuffer | ArrayBufferView,
  backend?: BackendChoice,
): string {
  const driver = resolveDriver(backend);
  return driver.wrapDeviceCekHex(
    toUint8Array(ephemeralSecret32),
    toUint8Array(devicePubkey32),
    deviceIdHash,
    toUint8Array(cek32),
  );
}

/**
 * Signs an OEM factory device attestation certificate (`OemDeviceCertificate`) over
 * `b"DNAM-OEM-CERT-V1:" || oemVendor || b"|" || modelCode || b"|" || deviceIdHash || b"|" || pubkeyX25519_32`.
 */
export function signOemDeviceCertificate(
  options: SignOemDeviceCertificateOptions,
): OemDeviceCertificate {
  const driver = resolveDriver(options.backend);
  const certJson = driver.signOemDeviceCert(
    toUint8Array(options.oemSeed32),
    options.oemVendor,
    options.modelCode,
    options.deviceIdHash,
    toUint8Array(options.pubkeyX25519_32),
  );
  const parsed: unknown = JSON.parse(certJson);
  return validateStandardSchema(OemDeviceCertificateSchema, parsed, "OemDeviceCertificate");
}

/**
 * Verifies an `OemDeviceCertificate` against `trustedOemPubkeyHex` and returns the verified
 * 64-character lowercase hex `X25519` device public key.
 */
export function verifyOemDeviceCertificate(
  cert: OemDeviceCertificate | Record<string, unknown> | string,
  trustedOemPubkeyHex: string,
  backend?: BackendChoice,
): string {
  const driver = resolveDriver(backend);
  const parsed: unknown = typeof cert === "string" ? JSON.parse(cert) : cert;
  const validated = validateStandardSchema(
    OemDeviceCertificateSchema,
    parsed,
    "OemDeviceCertificate",
  );
  const certJson = JSON.stringify(oemDeviceCertificateToWire(validated));
  return driver.verifyOemDeviceCert(certJson, trustedOemPubkeyHex);
}

/**
 * Seals an inner `.dnam` container into a `.dnam-seal` (`b"DNSL"`) 64-byte aligned outer binary envelope
 * with nested double encryption (`cek_outer` + `cek_inner`), verifying each `OemDeviceCertificate`
 * against `trustedOemPubkeys` and enforcing the creator's `metadata.distribution_policy`.
 */
export function sealDnamContainer(options: SealDnamContainerOptions): Uint8Array {
  const driver = resolveDriver(options.backend);
  const wireCerts = options.recipientCerts.map((c) => {
    const validated = validateStandardSchema(OemDeviceCertificateSchema, c, "OemDeviceCertificate");
    return oemDeviceCertificateToWire(validated);
  });
  return driver.sealContainer(
    toUint8Array(options.innerDnamBytes),
    JSON.stringify(wireCerts),
    JSON.stringify(options.trustedOemPubkeys),
    toUint8Array(options.marketplaceSeed32),
    toUint8Array(options.ephemeralSecret32),
    toUint8Array(options.cek32),
    toUint8Array(options.nonce12),
  );
}

/**
 * Unseals a `.dnam-seal` (`b"DNSL"`) outer binary envelope using an authorized device's `X25519` secret key,
 * returning the decrypted inner `.dnam` byte buffer.
 */
export function unsealDnamContainer(
  sealedBytes: Uint8Array | ArrayBuffer | ArrayBufferView,
  options: UnsealDnamContainerOptions = {},
): Uint8Array {
  const driver = resolveDriver(options.backend);
  return driver.unsealContainer(
    toUint8Array(sealedBytes),
    options.deviceIdHash,
    options.deviceSecretX25519 !== undefined ? toUint8Array(options.deviceSecretX25519) : undefined,
    options.trustedMarketplacePubkeyHex,
  );
}
