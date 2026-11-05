/**
 * Model metadata, dataset provenance, distribution policy, `Ed25519` signatures, marketplace
 * receipts, and OEM device certificates (`ref[sdk.ts.schema.camel-case-codec-parity]`,
 * `ref[sdk.ts.provenance.v2-security]`).
 *
 * Zero runtime JS dependencies; validated by native Rust `serde` (`dnam_engine::schema`).
 *
 * @module
 */

import type { ProtectionProfile } from "../types.js";

/**
 * Creator-signed distribution and OEM hardware exclusivity policy (`metadata.distributionPolicy`).
 */
export interface DistributionPolicy {
  readonly allowedProtectionProfiles: readonly ProtectionProfile[];
  readonly allowedOemVendors: readonly string[];
}

/**
 * OEM hardware factory attestation certificate (`OemDeviceCertificate`).
 */
export interface OemDeviceCertificate {
  readonly oemVendor: string;
  readonly modelCode: string;
  readonly deviceIdHash: string;
  readonly pubkeyX25519: string;
  readonly oemSigEd25519: string;
}

/**
 * Content-addressable dataset and recipe provenance (`metadata.provenance`).
 */
export interface DatasetProvenance {
  readonly datasetManifestSha256: string;
  readonly configRecipeSha256: string | null;
  readonly epochsTrained: number;
}

/**
 * Creator `Ed25519` signature block (`signatures.creatorSignature`).
 */
export interface CreatorSignature {
  readonly signerId: string;
  readonly pubkeyEd25519: string;
  readonly sigEd25519: string;
}

/**
 * Authorized OEM hardware device entry inside `MarketplaceReceipt`.
 */
export interface AuthorizedDevice {
  readonly oemVendor: string;
  readonly modelCode: string;
  readonly deviceIdHash: string;
  readonly wrappedCek?: string | null;
}

/**
 * Multi-creator royalty split entry (`10,000 bps = 100%`).
 */
export interface RoyaltySplit {
  readonly recipientId: string;
  readonly role: string;
  readonly shareBps: number;
}

/**
 * Marketplace download receipt (`MarketplaceReceipt`).
 */
export interface MarketplaceReceipt {
  readonly marketplaceId: string;
  readonly licenseId: string;
  readonly buyerHash: string;
  readonly licenseTier: string;
  readonly issuedAt: string;
  readonly buyerArenaSha256: string;
  readonly buyerTierArenaSha256: readonly [string, string, string, string] | null;
  readonly encryptionMode: string;
  readonly arenaNonce96: string | null;
  readonly ephemeralPubkeyX25519: string | null;
  readonly authorizedDevices: readonly AuthorizedDevice[];
  readonly royaltySplits: readonly RoyaltySplit[];
  readonly receiptSigEd25519: string;
}

/**
 * Split-payload `Ed25519` creator signature and optional marketplace receipt (`metadata.signatures`).
 */
export interface ModelSignatures {
  readonly immutableHeaderSha256: string;
  readonly masterArenaSha256: string;
  readonly tierArenaSha256: readonly [string, string, string, string] | null;
  readonly creatorSignature: CreatorSignature;
  readonly marketplaceReceipt: MarketplaceReceipt | null;
}

/**
 * Model identity, calibration, dataset provenance, and cryptographic signatures (`header.metadata`).
 */
export interface ModelMetadata {
  readonly name: string;
  readonly author: string | null;
  readonly description: string | null;
  readonly license: string | null;
  readonly createdAt: string;
  readonly inputReferenceDbfs: number;
  readonly provenance: DatasetProvenance;
  readonly distributionPolicy: DistributionPolicy | null;
  readonly signatures: ModelSignatures | null;
}
