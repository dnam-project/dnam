/**
 * Creator header signing (`signCreatorHeader`), marketplace receipt signing
 * (`signMarketplaceReceipt`), and `OemDeviceCertificate` V1/V2 attestation
 * (`ref[sdk.ts.provenance.v2-security]`, `ref[sec.sig.prefix-delegation]`,
 * `ref[sec.drm.oem-certificate]`).
 *
 * @module
 */

import { deepFreeze } from "../internal/freeze.js";
import { getActiveDriver } from "../internal/native/load.js";
import { camelToWireObject, wireToCamelObject } from "../schema/case-codec.js";
import { decodeHeader, encodeHeader, type ModelHeader } from "../schema/header.js";
import type {
  AuthorizedDevice,
  MarketplaceReceipt,
  OemDeviceCertificate,
  RoyaltySplit,
} from "../schema/signatures.js";
import { hexToBytes } from "./keys.js";

/**
 * Builds the canonical `Ed25519` creator signing message (`immutable_header_sha256 || master_arena_sha256`).
 */
export function buildCreatorSigningMessage(
  immutableHeaderSha256Hex: string,
  masterArenaSha256Hex: string,
): Uint8Array {
  return getActiveDriver().buildCreatorSigningMessage(
    immutableHeaderSha256Hex,
    masterArenaSha256Hex,
  );
}

/**
 * Builds the canonical `Ed25519` marketplace receipt signing message.
 */
export function buildMarketplaceSigningMessage(
  immutableHeaderSha256Hex: string,
  masterArenaSha256Hex: string,
  receipt: MarketplaceReceipt,
): Uint8Array {
  const wireReceipt = JSON.stringify(camelToWireObject(receipt));
  return getActiveDriver().buildMarketplaceSigningMessage(
    immutableHeaderSha256Hex,
    masterArenaSha256Hex,
    wireReceipt,
  );
}

/**
 * Signs a `ModelHeader` and `masterArena` using the creator's 32-byte `Ed25519` seed in Rust,
 * returning a new frozen `ModelHeader` with `metadata.signatures` populated
 * (`ref[sec.sig.prefix-delegation]`).
 */
export function signCreatorHeader(options: {
  readonly header: ModelHeader | string;
  readonly masterArena: Uint8Array;
  readonly creatorSeed: Uint8Array | string;
  readonly signerId: string;
  readonly includeTierPrefixes?: boolean;
}): ModelHeader {
  const headerJson =
    typeof options.header === "string" ? options.header : encodeHeader(options.header);
  const seed =
    typeof options.creatorSeed === "string" ? hexToBytes(options.creatorSeed) : options.creatorSeed;
  const signedWireJson = getActiveDriver().signCreatorHeader(
    headerJson,
    options.masterArena,
    seed,
    options.signerId,
    options.includeTierPrefixes ?? false,
  );
  return decodeHeader(signedWireJson);
}

/**
 * Signs a `MarketplaceReceipt` and attaches it to a creator-signed `ModelHeader` in Rust
 * (`ref[sec.sig.prefix-delegation]`).
 */
export function signMarketplaceReceipt(options: {
  readonly header: ModelHeader | string;
  readonly buyerArena: Uint8Array;
  readonly marketplaceSeed: Uint8Array | string;
  readonly marketplaceId: string;
  readonly licenseId: string;
  readonly buyerHash: string;
  readonly licenseTier: string;
  readonly issuedAt?: string;
  readonly encryptionMode?: string;
  readonly arenaNonce96?: string | null;
  readonly ephemeralPubkeyX25519?: string | null;
  readonly authorizedDevices?: readonly AuthorizedDevice[];
  readonly royaltySplits?: readonly RoyaltySplit[];
  readonly includeBuyerTierPrefixes?: boolean;
}): ModelHeader {
  const headerJson =
    typeof options.header === "string" ? options.header : encodeHeader(options.header);
  const seed =
    typeof options.marketplaceSeed === "string"
      ? hexToBytes(options.marketplaceSeed)
      : options.marketplaceSeed;
  const specJson = JSON.stringify({
    marketplaceId: options.marketplaceId,
    licenseId: options.licenseId,
    buyerHash: options.buyerHash,
    licenseTier: options.licenseTier,
    issuedAt: options.issuedAt ?? null,
    encryptionMode: options.encryptionMode ?? null,
    arenaNonce96: options.arenaNonce96 ?? null,
    ephemeralPubkeyX25519: options.ephemeralPubkeyX25519 ?? null,
    authorizedDevices: camelToWireObject(options.authorizedDevices ?? []),
    royaltySplits: camelToWireObject(options.royaltySplits ?? []),
    includeBuyerTierPrefixes: options.includeBuyerTierPrefixes ?? false,
  });
  const signedWireJson = getActiveDriver().signMarketplaceReceipt(
    headerJson,
    options.buyerArena,
    seed,
    specJson,
  );
  return decodeHeader(signedWireJson);
}

/**
 * Signs an `OemDeviceCertificate` using the OEM vendor's 32-byte `Ed25519` factory seed
 * (`ref[sec.drm.oem-certificate]`).
 */
export function signDeviceCertificate(options: {
  readonly oemSeed: Uint8Array | string;
  readonly oemVendor: string;
  readonly modelCode: string;
  readonly deviceIdHash: string;
  readonly pubkeyX25519: Uint8Array | string;
}): OemDeviceCertificate {
  const seed = typeof options.oemSeed === "string" ? hexToBytes(options.oemSeed) : options.oemSeed;
  const pk =
    typeof options.pubkeyX25519 === "string"
      ? hexToBytes(options.pubkeyX25519)
      : options.pubkeyX25519;
  const wireJson = getActiveDriver().signOemDeviceCert(
    seed,
    options.oemVendor,
    options.modelCode,
    options.deviceIdHash,
    pk,
  );
  const camelObj = wireToCamelObject(JSON.parse(wireJson)) as OemDeviceCertificate;
  return deepFreeze(camelObj);
}

/**
 * Verifies an `OemDeviceCertificate` against a trusted 64-character hex OEM `Ed25519` public key,
 * returning the verified 64-character lowercase hex `pubkeyX25519`.
 */
export function verifyDeviceCertificate(
  certificate: OemDeviceCertificate | string,
  trustedOemPubkeyHex: string,
): string {
  const wireJson =
    typeof certificate === "string" ? certificate : JSON.stringify(camelToWireObject(certificate));
  return getActiveDriver().verifyOemDeviceCert(wireJson, trustedOemPubkeyHex);
}

/**
 * Builds the domain-separated V2 OEM device certificate signing message (`"DNAM-OEM-CERT-V2"`).
 */
export function buildOemCertV2SigningMessage(options: {
  readonly oemVendor: string;
  readonly modelCode: string;
  readonly deviceIdHash16: Uint8Array | string;
  readonly pubkeyX2551932: Uint8Array | string;
  readonly issuerPubkeyEd2551932: Uint8Array | string;
  readonly assurance?: number;
  readonly attestationPolicySha25632?: Uint8Array | string;
  readonly attestationEvidenceSha25632?: Uint8Array | string;
}): Uint8Array {
  const toBytes = (v: Uint8Array | string | undefined): Uint8Array | undefined =>
    v === undefined ? undefined : typeof v === "string" ? hexToBytes(v) : v;
  return getActiveDriver().buildOemCertV2SigningMessage(
    options.oemVendor,
    options.modelCode,
    toBytes(options.deviceIdHash16) as Uint8Array,
    toBytes(options.pubkeyX2551932) as Uint8Array,
    toBytes(options.issuerPubkeyEd2551932) as Uint8Array,
    options.assurance ?? 1,
    toBytes(options.attestationPolicySha25632),
    toBytes(options.attestationEvidenceSha25632),
  );
}

/**
 * Validates `licenseTier` and that `royaltySplits` sums to `10,000` bps (`100%`).
 */
export function validateRoyaltySplits(
  licenseTier: string,
  royaltySplits: readonly RoyaltySplit[],
): void {
  const wireJson = JSON.stringify(camelToWireObject(royaltySplits));
  getActiveDriver().validateRoyaltySplits(licenseTier, wireJson);
}
