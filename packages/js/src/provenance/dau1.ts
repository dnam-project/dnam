/**
 * Marketplace v2 authentication (`DAU1`), encrypted mutable trailers (`DME1`), plaintext mutable
 * trailers (`DMU1`), HPKE recipient slots, and record directories (`ref[sdk.ts.provenance.v2-security]`,
 * `ref[sec.sig.dau1-envelope]`).
 *
 * @module
 */

import { coerceBytes } from "../internal/coerce.js";
import { deepFreeze } from "../internal/freeze.js";
import { getActiveDriver } from "../internal/native/load.js";
import { bytesToHex } from "./keys.js";

/**
 * Decoded or input specification for a `DAU1` authentication envelope (`ref[sec.sig.dau1-envelope]`).
 */
export interface Dau1EnvelopeSpec {
  readonly authKind: 0 | 1 | 2;
  readonly creatorPubkeyHex: string;
  readonly creatorMaxTier: number;
  readonly masterPrefixesHex: readonly string[];
  readonly creatorSignatureHex: string;
  readonly marketplacePubkeyHex?: string | null;
  readonly policyBytes?: Uint8Array | string;
  readonly buyerMaxTier?: number | null;
  readonly buyerPrefixesHex?: readonly string[];
  readonly marketplaceSignatureHex?: string | null;
}

/**
 * Decoded `DAU1` authentication envelope.
 */
export interface DecodedDau1Envelope {
  readonly authKind: 0 | 1 | 2;
  readonly creatorPubkeyHex: string;
  readonly creatorMaxTier: number;
  readonly masterPrefixesHex: readonly string[];
  readonly creatorSignatureHex: string;
  readonly marketplacePubkeyHex: string | null;
  readonly policyBytesHex: string;
  readonly buyerMaxTier: number | null;
  readonly buyerPrefixesHex: readonly string[];
  readonly marketplaceSignatureHex: string | null;
}

/**
 * Serializes a `DAU1` authentication envelope into wire bytes (`ref[sec.sig.dau1-envelope]`).
 */
export function encodeDau1(spec: Dau1EnvelopeSpec): Uint8Array {
  const policyHex =
    spec.policyBytes === undefined
      ? ""
      : typeof spec.policyBytes === "string"
        ? spec.policyBytes
        : bytesToHex(spec.policyBytes);
  const payload = JSON.stringify({
    authKind: spec.authKind,
    creatorPubkeyHex: spec.creatorPubkeyHex,
    creatorMaxTier: spec.creatorMaxTier,
    masterPrefixesHex: spec.masterPrefixesHex,
    creatorSignatureHex: spec.creatorSignatureHex,
    marketplacePubkeyHex: spec.marketplacePubkeyHex ?? null,
    policyBytesHex: policyHex,
    buyerMaxTier: spec.buyerMaxTier ?? null,
    buyerPrefixesHex: spec.buyerPrefixesHex ?? [],
    marketplaceSignatureHex: spec.marketplaceSignatureHex ?? null,
  });
  return getActiveDriver().encodeDau1Envelope(payload);
}

/**
 * Parses and validates a `DAU1` authentication envelope from wire bytes.
 */
export function decodeDau1(
  wireBytes: Uint8Array | ArrayBuffer | ArrayBufferView,
): DecodedDau1Envelope {
  const jsonStr = getActiveDriver().decodeDau1Envelope(coerceBytes(wireBytes));
  return deepFreeze(JSON.parse(jsonStr) as DecodedDau1Envelope);
}

/**
 * Builds a 20-byte `DME1` encrypted mutable frame header (`ref[sec.drm.dme1-mutable-suffix]`).
 */
export function buildDme1Header(options: {
  readonly profile: number;
  readonly recipientCount: number;
  readonly ciphertextLength: number;
}): Uint8Array {
  return getActiveDriver().buildDme1Header(
    options.profile,
    options.recipientCount,
    options.ciphertextLength,
  );
}

/**
 * Validates and parses decrypted `DME1` plaintext against recipient SHA-256 key digests (hex).
 */
export function parseDme1Plaintext(
  plaintext: Uint8Array | ArrayBuffer | ArrayBufferView,
  recipientKeyDigestsHex: readonly string[],
): Record<string, unknown> {
  const jsonStr = getActiveDriver().parseDme1Plaintext(
    coerceBytes(plaintext),
    recipientKeyDigestsHex,
  );
  return deepFreeze(JSON.parse(jsonStr) as Record<string, unknown>);
}

/**
 * Builds a 64B-aligned `DMU1` mutable trailer frame from `mutableJson` and `capacity`.
 */
export function buildDmu1Frame(
  mutableJson: string | Record<string, unknown> | Uint8Array,
  capacity: number,
): Uint8Array {
  let bytes: Uint8Array;
  if (mutableJson instanceof Uint8Array) {
    bytes = mutableJson;
  } else {
    const str = typeof mutableJson === "string" ? mutableJson : JSON.stringify(mutableJson);
    bytes = new TextEncoder().encode(str);
  }
  return getActiveDriver().buildDmu1Frame(bytes, capacity);
}

/**
 * Parses and validates a 64B-aligned `DMU1` mutable trailer frame.
 */
export function parseDmu1Frame(
  frameBytes: Uint8Array | ArrayBuffer | ArrayBufferView,
): Record<string, unknown> {
  const jsonStr = getActiveDriver().parseDmu1Frame(coerceBytes(frameBytes));
  return deepFreeze(JSON.parse(jsonStr) as Record<string, unknown>);
}

/**
 * Encodes HPKE recipient key SHA-256 digests (`R = u32le(N) || digest[0..N]`).
 */
export function encodeRecipientKeyIds(keyDigestsHex: readonly string[]): Uint8Array {
  return getActiveDriver().encodeRecipientKeyIds(keyDigestsHex);
}

/**
 * Parses HPKE recipient key SHA-256 digests (`R = u32le(N) || digest[0..N]`).
 */
export function parseRecipientKeyIds(
  data: Uint8Array | ArrayBuffer | ArrayBufferView,
  maxRecipients = 16,
): readonly string[] {
  return Object.freeze(getActiveDriver().parseRecipientKeyIds(coerceBytes(data), maxRecipients));
}

/**
 * Encodes an HPKE slot vector (`S = u32le(N) || (enc[32] || ct_and_tag[48])[0..N]`).
 */
export function encodeHpkeSlots(slotsHex: readonly (readonly [string, string])[]): Uint8Array {
  return getActiveDriver().encodeHpkeSlots(JSON.stringify(slotsHex));
}

/**
 * Parses an HPKE slot vector (`S = u32le(N) || (enc[32] || ct_and_tag[48])[0..N]`).
 */
export function parseHpkeSlots(
  data: Uint8Array | ArrayBuffer | ArrayBufferView,
  expectedCount: number,
  maxRecipients = 16,
): readonly (readonly [string, string])[] {
  const jsonStr = getActiveDriver().parseHpkeSlots(coerceBytes(data), expectedCount, maxRecipients);
  return deepFreeze(JSON.parse(jsonStr) as (readonly [string, string])[]);
}

/**
 * Encodes a record directory (`D = u32le(C) || (u64le(offset) || u64le(length))[0..C]`).
 */
export function encodeRecordDirectory(
  maximumTier: number,
  records: readonly (readonly [number, number])[],
): Uint8Array {
  return getActiveDriver().encodeRecordDirectory(maximumTier, JSON.stringify(records));
}

/**
 * Parses a record directory (`D = u32le(C) || (u64le(offset) || u64le(length))[0..C]`).
 */
export function parseRecordDirectory(
  data: Uint8Array | ArrayBuffer | ArrayBufferView,
  maximumTier: number,
): readonly (readonly [number, number])[] {
  const jsonStr = getActiveDriver().parseRecordDirectory(coerceBytes(data), maximumTier);
  return deepFreeze(JSON.parse(jsonStr) as (readonly [number, number])[]);
}
