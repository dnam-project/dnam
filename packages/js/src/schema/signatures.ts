/**
 * Split-payload `Ed25519` signature envelope (`ModelSignatures`), creator signature (`CreatorSignature`),
 * marketplace download receipt (`MarketplaceReceipt`), OEM hardware device allowlist (`AuthorizedDevice`),
 * and multi-creator royalty split records (`RoyaltySplit`).
 */

import { z } from "zod";

export type DnamProtectionProfile = "open" | "device_allowlisted" | "enclave_sealed";

export const DnamProtectionProfileSchema = z.enum(["open", "device_allowlisted", "enclave_sealed"]);

export interface DistributionPolicy {
  allowedProtectionProfiles: DnamProtectionProfile[];
  allowedOemVendors: string[];
}

const RawDistributionPolicySchema = z
  .object({
    allowed_protection_profiles: z.array(DnamProtectionProfileSchema).optional(),
    allowedProtectionProfiles: z.array(DnamProtectionProfileSchema).optional(),
    allowed_oem_vendors: z.array(z.string()).optional(),
    allowedOemVendors: z.array(z.string()).optional(),
  })
  .strict();

export const DistributionPolicySchema: z.ZodType<DistributionPolicy, unknown> =
  RawDistributionPolicySchema.transform(
    (raw): DistributionPolicy => ({
      allowedProtectionProfiles: raw.allowedProtectionProfiles ??
        raw.allowed_protection_profiles ?? ["open", "device_allowlisted", "enclave_sealed"],
      allowedOemVendors: raw.allowedOemVendors ?? raw.allowed_oem_vendors ?? [],
    }),
  );

export interface OemDeviceCertificate {
  oemVendor: string;
  modelCode: string;
  deviceIdHash: string;
  pubkeyX25519: string;
  oemSigEd25519: string;
}

const RawOemDeviceCertificateSchema = z
  .object({
    oem_vendor: z.string().optional(),
    oemVendor: z.string().optional(),
    model_code: z.string().optional(),
    modelCode: z.string().optional(),
    device_id_hash: z.string().optional(),
    deviceIdHash: z.string().optional(),
    pubkey_x25519: z.string().optional(),
    pubkeyX25519: z.string().optional(),
    oem_sig_ed25519: z.string().optional(),
    oemSigEd25519: z.string().optional(),
  })
  .strict()
  .superRefine((val, ctx) => {
    const vendor = val.oemVendor ?? val.oem_vendor;
    const model = val.modelCode ?? val.model_code;
    const devId = val.deviceIdHash ?? val.device_id_hash;
    const pk = val.pubkeyX25519 ?? val.pubkey_x25519;
    const sig = val.oemSigEd25519 ?? val.oem_sig_ed25519;
    if (!vendor || !model || !devId || !pk || !sig) {
      ctx.addIssue({
        code: "custom",
        message:
          "OemDeviceCertificate requires oemVendor, modelCode, deviceIdHash, pubkeyX25519, and oemSigEd25519",
      });
    }
  });

export const OemDeviceCertificateSchema: z.ZodType<OemDeviceCertificate, unknown> =
  RawOemDeviceCertificateSchema.transform(
    (raw): OemDeviceCertificate => ({
      oemVendor: raw.oemVendor ?? raw.oem_vendor ?? "",
      modelCode: raw.modelCode ?? raw.model_code ?? "",
      deviceIdHash: raw.deviceIdHash ?? raw.device_id_hash ?? "",
      pubkeyX25519: raw.pubkeyX25519 ?? raw.pubkey_x25519 ?? "",
      oemSigEd25519: raw.oemSigEd25519 ?? raw.oem_sig_ed25519 ?? "",
    }),
  );

export function distributionPolicyToWire(policy: DistributionPolicy): Record<string, unknown> {
  return {
    allowed_protection_profiles: policy.allowedProtectionProfiles,
    allowed_oem_vendors: policy.allowedOemVendors,
  };
}

export function oemDeviceCertificateToWire(cert: OemDeviceCertificate): Record<string, unknown> {
  return {
    oem_vendor: cert.oemVendor,
    model_code: cert.modelCode,
    device_id_hash: cert.deviceIdHash,
    pubkey_x25519: cert.pubkeyX25519,
    oem_sig_ed25519: cert.oemSigEd25519,
  };
}

export interface DatasetProvenance {
  datasetManifestSha256: string;
  configRecipeSha256?: string | undefined;
  epochsTrained: number;
}

const RawDatasetProvenanceSchema = z
  .object({
    dataset_manifest_sha256: z.string().optional(),
    datasetManifestSha256: z.string().optional(),
    config_recipe_sha256: z.string().nullable().optional(),
    configRecipeSha256: z.string().nullable().optional(),
    epochs_trained: z.number().int().nonnegative().optional(),
    epochsTrained: z.number().int().nonnegative().optional(),
  })
  .strict();

export const DatasetProvenanceSchema: z.ZodType<DatasetProvenance, unknown> =
  RawDatasetProvenanceSchema.transform(
    (raw): DatasetProvenance => ({
      datasetManifestSha256: raw.datasetManifestSha256 ?? raw.dataset_manifest_sha256 ?? "",
      configRecipeSha256: raw.configRecipeSha256 ?? raw.config_recipe_sha256 ?? undefined,
      epochsTrained: raw.epochsTrained ?? raw.epochs_trained ?? 0,
    }),
  );

export interface CreatorSignature {
  signerId: string;
  pubkeyEd25519: string;
  sigEd25519: string;
}

const RawCreatorSignatureSchema = z
  .object({
    signer_id: z.string().optional(),
    signerId: z.string().optional(),
    pubkey_ed25519: z.string().optional(),
    pubkeyEd25519: z.string().optional(),
    sig_ed25519: z.string().optional(),
    sigEd25519: z.string().optional(),
  })
  .strict()
  .superRefine((val, ctx) => {
    const signerId = val.signerId ?? val.signer_id;
    const pubkey = val.pubkeyEd25519 ?? val.pubkey_ed25519;
    const sig = val.sigEd25519 ?? val.sig_ed25519;
    if (!signerId || !pubkey || !sig) {
      ctx.addIssue({
        code: "custom",
        message: "CreatorSignature requires signerId, pubkeyEd25519, and sigEd25519",
      });
    }
  });

export const CreatorSignatureSchema: z.ZodType<CreatorSignature, unknown> =
  RawCreatorSignatureSchema.transform(
    (raw): CreatorSignature => ({
      signerId: raw.signerId ?? raw.signer_id ?? "",
      pubkeyEd25519: raw.pubkeyEd25519 ?? raw.pubkey_ed25519 ?? "",
      sigEd25519: raw.sigEd25519 ?? raw.sig_ed25519 ?? "",
    }),
  );

export interface AuthorizedDevice {
  oemVendor: string;
  modelCode: string;
  deviceIdHash: string;
  wrappedCek?: string | undefined;
}

const RawAuthorizedDeviceSchema = z
  .object({
    oem_vendor: z.string().optional(),
    oemVendor: z.string().optional(),
    model_code: z.string().optional(),
    modelCode: z.string().optional(),
    device_id_hash: z.string().optional(),
    deviceIdHash: z.string().optional(),
    wrapped_cek: z.string().nullable().optional(),
    wrappedCek: z.string().nullable().optional(),
  })
  .strict();

export const AuthorizedDeviceSchema: z.ZodType<AuthorizedDevice, unknown> =
  RawAuthorizedDeviceSchema.transform(
    (raw): AuthorizedDevice => ({
      oemVendor: raw.oemVendor ?? raw.oem_vendor ?? "",
      modelCode: raw.modelCode ?? raw.model_code ?? "",
      deviceIdHash: raw.deviceIdHash ?? raw.device_id_hash ?? "",
      wrappedCek: raw.wrappedCek ?? raw.wrapped_cek ?? undefined,
    }),
  );

export interface RoyaltySplit {
  recipientId: string;
  role: string;
  shareBps: number;
}

const RawRoyaltySplitSchema = z
  .object({
    recipient_id: z.string().optional(),
    recipientId: z.string().optional(),
    role: z.string(),
    share_bps: z.number().int().min(0).max(10000).optional(),
    shareBps: z.number().int().min(0).max(10000).optional(),
  })
  .strict();

export const RoyaltySplitSchema: z.ZodType<RoyaltySplit, unknown> = RawRoyaltySplitSchema.transform(
  (raw): RoyaltySplit => ({
    recipientId: raw.recipientId ?? raw.recipient_id ?? "",
    role: raw.role,
    shareBps: raw.shareBps ?? raw.share_bps ?? 0,
  }),
);

export interface MarketplaceReceipt {
  marketplaceId: string;
  licenseId: string;
  buyerHash: string;
  licenseTier: string;
  issuedAt: string;
  buyerArenaSha256: string;
  encryptionMode: string;
  arenaNonce96?: string | undefined;
  ephemeralPubkeyX25519?: string | undefined;
  authorizedDevices: AuthorizedDevice[];
  royaltySplits: RoyaltySplit[];
  receiptSigEd25519: string;
}

const RawMarketplaceReceiptSchema = z
  .object({
    marketplace_id: z.string().optional(),
    marketplaceId: z.string().optional(),
    license_id: z.string().optional(),
    licenseId: z.string().optional(),
    buyer_hash: z.string().optional(),
    buyerHash: z.string().optional(),
    license_tier: z.string().optional(),
    licenseTier: z.string().optional(),
    issued_at: z.string().optional(),
    issuedAt: z.string().optional(),
    buyer_arena_sha256: z.string().optional(),
    buyerArenaSha256: z.string().optional(),
    encryption_mode: z.string().optional(),
    encryptionMode: z.string().optional(),
    arena_nonce_96: z.string().nullable().optional(),
    arenaNonce96: z.string().nullable().optional(),
    ephemeral_pubkey_x25519: z.string().nullable().optional(),
    ephemeralPubkeyX25519: z.string().nullable().optional(),
    authorized_devices: z.array(AuthorizedDeviceSchema).optional(),
    authorizedDevices: z.array(AuthorizedDeviceSchema).optional(),
    royalty_splits: z.array(RoyaltySplitSchema).optional(),
    royaltySplits: z.array(RoyaltySplitSchema).optional(),
    receipt_sig_ed25519: z.string().optional(),
    receiptSigEd25519: z.string().optional(),
  })
  .strict();

export const MarketplaceReceiptSchema: z.ZodType<MarketplaceReceipt, unknown> =
  RawMarketplaceReceiptSchema.transform(
    (raw): MarketplaceReceipt => ({
      marketplaceId: raw.marketplaceId ?? raw.marketplace_id ?? "",
      licenseId: raw.licenseId ?? raw.license_id ?? "",
      buyerHash: raw.buyerHash ?? raw.buyer_hash ?? "",
      licenseTier: raw.licenseTier ?? raw.license_tier ?? "",
      issuedAt: raw.issuedAt ?? raw.issued_at ?? "",
      buyerArenaSha256: raw.buyerArenaSha256 ?? raw.buyer_arena_sha256 ?? "",
      encryptionMode: raw.encryptionMode ?? raw.encryption_mode ?? "none",
      arenaNonce96: raw.arenaNonce96 ?? raw.arena_nonce_96 ?? undefined,
      ephemeralPubkeyX25519: raw.ephemeralPubkeyX25519 ?? raw.ephemeral_pubkey_x25519 ?? undefined,
      authorizedDevices: raw.authorizedDevices ?? raw.authorized_devices ?? [],
      royaltySplits: raw.royaltySplits ?? raw.royalty_splits ?? [],
      receiptSigEd25519: raw.receiptSigEd25519 ?? raw.receipt_sig_ed25519 ?? "",
    }),
  );

export interface ModelSignatures {
  immutableHeaderSha256: string;
  masterArenaSha256: string;
  creatorSignature: CreatorSignature;
  marketplaceReceipt?: MarketplaceReceipt | undefined;
}

const RawModelSignaturesSchema = z
  .object({
    immutable_header_sha256: z.string().optional(),
    immutableHeaderSha256: z.string().optional(),
    master_arena_sha256: z.string().optional(),
    masterArenaSha256: z.string().optional(),
    creator_signature: CreatorSignatureSchema.optional(),
    creatorSignature: CreatorSignatureSchema.optional(),
    marketplace_receipt: MarketplaceReceiptSchema.nullable().optional(),
    marketplaceReceipt: MarketplaceReceiptSchema.nullable().optional(),
  })
  .strict()
  .superRefine((val, ctx) => {
    const creator = val.creatorSignature ?? val.creator_signature;
    if (!creator) {
      ctx.addIssue({
        code: "custom",
        message: "ModelSignatures requires creatorSignature",
      });
    }
  });

export const ModelSignaturesSchema: z.ZodType<ModelSignatures, unknown> =
  RawModelSignaturesSchema.transform((raw): ModelSignatures => {
    const rawCreator = (raw.creatorSignature ?? raw.creator_signature) as CreatorSignature;
    const rawReceipt = raw.marketplaceReceipt ?? raw.marketplace_receipt;
    return {
      immutableHeaderSha256: raw.immutableHeaderSha256 ?? raw.immutable_header_sha256 ?? "",
      masterArenaSha256: raw.masterArenaSha256 ?? raw.master_arena_sha256 ?? "",
      creatorSignature: rawCreator,
      marketplaceReceipt: rawReceipt ?? undefined,
    };
  });

export function datasetProvenanceToWire(prov: DatasetProvenance): Record<string, unknown> {
  return {
    dataset_manifest_sha256: prov.datasetManifestSha256,
    config_recipe_sha256: prov.configRecipeSha256 ?? null,
    epochs_trained: prov.epochsTrained,
  };
}

export function creatorSignatureToWire(sig: CreatorSignature): Record<string, unknown> {
  return {
    signer_id: sig.signerId,
    pubkey_ed25519: sig.pubkeyEd25519,
    sig_ed25519: sig.sigEd25519,
  };
}

export function authorizedDeviceToWire(dev: AuthorizedDevice): Record<string, unknown> {
  return {
    oem_vendor: dev.oemVendor,
    model_code: dev.modelCode,
    device_id_hash: dev.deviceIdHash,
    wrapped_cek: dev.wrappedCek ?? null,
  };
}

export function royaltySplitToWire(split: RoyaltySplit): Record<string, unknown> {
  return {
    recipient_id: split.recipientId,
    role: split.role,
    share_bps: split.shareBps,
  };
}

export function marketplaceReceiptToWire(receipt: MarketplaceReceipt): Record<string, unknown> {
  return {
    marketplace_id: receipt.marketplaceId,
    license_id: receipt.licenseId,
    buyer_hash: receipt.buyerHash,
    license_tier: receipt.licenseTier,
    issued_at: receipt.issuedAt,
    buyer_arena_sha256: receipt.buyerArenaSha256,
    encryption_mode: receipt.encryptionMode,
    arena_nonce_96: receipt.arenaNonce96 ?? null,
    ephemeral_pubkey_x25519: receipt.ephemeralPubkeyX25519 ?? null,
    authorized_devices: receipt.authorizedDevices.map(authorizedDeviceToWire),
    royalty_splits: receipt.royaltySplits.map(royaltySplitToWire),
    receipt_sig_ed25519: receipt.receiptSigEd25519,
  };
}

export function modelSignaturesToWire(sigs: ModelSignatures): Record<string, unknown> {
  return {
    immutable_header_sha256: sigs.immutableHeaderSha256,
    master_arena_sha256: sigs.masterArenaSha256,
    creator_signature: creatorSignatureToWire(sigs.creatorSignature),
    marketplace_receipt: sigs.marketplaceReceipt
      ? marketplaceReceiptToWire(sigs.marketplaceReceipt)
      : null,
  };
}
