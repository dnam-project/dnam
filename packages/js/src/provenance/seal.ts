/**
 * `.dnam-seal` (`DNSL`) hardware envelope sealing (`seal`), unsealing (`unseal`), `X25519` CEK
 * wrapping, and 64B cache-line `ChaCha20` arena encryption (`ref[sdk.ts.provenance.v2-security]`,
 * `ref[sec.drm.dnsl-public-framing]`).
 *
 * @module
 */

import { coerceBytes } from "../internal/coerce.js";
import { getActiveDriver } from "../internal/native/load.js";
import { Model } from "../model/model.js";
import type { ParameterValueMap } from "../parameters/typed-model.js";
import { camelToWireObject } from "../schema/case-codec.js";
import type { OemDeviceCertificate } from "../schema/signatures.js";
import { hexToBytes } from "./keys.js";

/**
 * Options for `seal(modelOrBytes, options)`.
 */
export interface SealOptions {
  readonly recipientCertificates: readonly OemDeviceCertificate[];
  readonly trustedOemPubkeysHex: Readonly<Record<string, string>>;
  readonly marketplaceSeed: Uint8Array | string;
  readonly ephemeralSecret: Uint8Array | string;
  readonly cek: Uint8Array | string;
  readonly nonce12: Uint8Array | string;
}

/**
 * Options for `unseal(sealedBytes, options)`.
 */
export interface UnsealOptions {
  readonly deviceIdHash?: string;
  readonly deviceSecret?: Uint8Array | string;
  readonly trustedMarketplacePubkeyHex?: string;
}

function toBytes32Or12(val: Uint8Array | string): Uint8Array {
  return typeof val === "string" ? hexToBytes(val) : val;
}

/**
 * Seals an inner `.dnam` `Model` (or raw container bytes) into a 64-byte aligned `.dnam-seal`
 * (`b"DNSL"`) hardware-bound envelope (`ref[sec.drm.dnsl-public-framing]`).
 */
export function seal<TParams extends ParameterValueMap = Record<string, number | string>>(
  modelOrBytes: Model<TParams> | Uint8Array | ArrayBuffer | ArrayBufferView,
  options: SealOptions,
): Uint8Array {
  const innerBytes =
    modelOrBytes instanceof Model ? modelOrBytes.toBytes() : coerceBytes(modelOrBytes);
  const certsJson = JSON.stringify(camelToWireObject(options.recipientCertificates));
  const oemPksJson = JSON.stringify(options.trustedOemPubkeysHex);
  return getActiveDriver().sealContainer(
    innerBytes,
    certsJson,
    oemPksJson,
    toBytes32Or12(options.marketplaceSeed),
    toBytes32Or12(options.ephemeralSecret),
    toBytes32Or12(options.cek),
    toBytes32Or12(options.nonce12),
  );
}

/**
 * Unseals a `.dnam-seal` (`b"DNSL"`) hardware-bound envelope using the device's `X25519` secret
 * key, returning the decrypted inner `Model` (`ref[sec.drm.dnsl-public-framing]`).
 */
export function unseal(
  sealedBytes: Uint8Array | ArrayBuffer | ArrayBufferView,
  options?: UnsealOptions,
): Model {
  const devSec =
    options?.deviceSecret !== undefined ? toBytes32Or12(options.deviceSecret) : undefined;
  const innerBytes = getActiveDriver().unsealContainer(
    coerceBytes(sealedBytes),
    options?.deviceIdHash,
    devSec,
    options?.trustedMarketplacePubkeyHex,
  );
  return Model.from(innerBytes);
}

/**
 * Wraps a 32-byte Content Encryption Key (`cek`) for an OEM device (`X25519` + `HKDF-SHA256` +
 * `ChaCha20-Poly1305`), returning the 96-character lowercase hex `wrappedCek`.
 */
export function wrapDeviceCek(options: {
  readonly ephemeralSecret: Uint8Array | string;
  readonly devicePubkey: Uint8Array | string;
  readonly deviceIdHash: string;
  readonly cek: Uint8Array | string;
}): string {
  return getActiveDriver().wrapDeviceCekHex(
    toBytes32Or12(options.ephemeralSecret),
    toBytes32Or12(options.devicePubkey),
    options.deviceIdHash,
    toBytes32Or12(options.cek),
  );
}

/**
 * Applies RFC 8439 `ChaCha20` 64-byte cache-line XOR encryption/decryption over `arenaBytes`.
 */
export function chacha20XorArena(options: {
  readonly arenaBytes: Uint8Array | ArrayBuffer | ArrayBufferView;
  readonly key: Uint8Array | string;
  readonly nonce12: Uint8Array | string;
  readonly startBlockCounter?: number;
}): Uint8Array {
  return getActiveDriver().chacha20XorArena(
    coerceBytes(options.arenaBytes),
    toBytes32Or12(options.key),
    toBytes32Or12(options.nonce12),
    options.startBlockCounter ?? 0,
  );
}
