/**
 * `Ed25519` / `X25519` key derivation, signing, SHA-256, HMAC-SHA256, and hex conversion
 * utilities (`ref[sdk.ts.provenance.v2-security]`).
 *
 * @module
 */

import { DnamError } from "../errors.js";
import { coerceBytes } from "../internal/coerce.js";
import { getActiveDriver } from "../internal/native/load.js";

/**
 * Converts a lowercase/uppercase hex string into a `Uint8Array`.
 */
export function hexToBytes(hex: string): Uint8Array {
  const clean = hex.trim();
  if (clean.length % 2 !== 0) {
    throw new DnamError("ERR_DNAM_VALIDATION", "Hex string must have an even number of characters");
  }
  const out = new Uint8Array(clean.length / 2);
  for (let i = 0; i < out.length; i++) {
    const byte = Number.parseInt(clean.slice(i * 2, i * 2 + 2), 16);
    if (Number.isNaN(byte)) {
      throw new DnamError("ERR_DNAM_VALIDATION", `Invalid hex character at index ${String(i * 2)}`);
    }
    out[i] = byte;
  }
  return out;
}

/**
 * Converts a `Uint8Array` into a lowercase hex string.
 */
export function bytesToHex(bytes: Uint8Array | ArrayBuffer | ArrayBufferView): string {
  const u8 = coerceBytes(bytes);
  let out = "";
  for (let i = 0; i < u8.length; i++) {
    out += (u8[i] ?? 0).toString(16).padStart(2, "0");
  }
  return out;
}

/**
 * Computes the lowercase hex SHA-256 digest of `data`.
 */
export function sha256Hex(data: Uint8Array | ArrayBuffer | ArrayBufferView): string {
  return getActiveDriver().sha256Hex(coerceBytes(data));
}

/**
 * Computes the lowercase hex HMAC-SHA256 digest of `message` under `key32`.
 */
export function hmacSha256Hex(
  key32: Uint8Array | string,
  message: Uint8Array | ArrayBuffer | ArrayBufferView,
): string {
  const key = typeof key32 === "string" ? hexToBytes(key32) : key32;
  return getActiveDriver().hmacSha256Hex(key, coerceBytes(message));
}

/**
 * Derives the 64-character lowercase hex `Ed25519` public key from a 32-byte secret key seed.
 */
export function ed25519PublicKey(seed32: Uint8Array | string): string {
  const seed = typeof seed32 === "string" ? hexToBytes(seed32) : seed32;
  return getActiveDriver().ed25519PubkeyHex(seed);
}

/**
 * Signs `message` using a 32-byte `Ed25519` secret key seed, returning the 128-character hex signature.
 */
export function ed25519Sign(
  seed32: Uint8Array | string,
  message: Uint8Array | ArrayBuffer | ArrayBufferView,
): string {
  const seed = typeof seed32 === "string" ? hexToBytes(seed32) : seed32;
  return getActiveDriver().ed25519SignHex(seed, coerceBytes(message));
}

/**
 * Derives the 64-character lowercase hex `X25519` public key from a 32-byte secret key.
 */
export function x25519PublicKey(secret32: Uint8Array | string): string {
  const sec = typeof secret32 === "string" ? hexToBytes(secret32) : secret32;
  return getActiveDriver().x25519PubkeyHex(sec);
}
