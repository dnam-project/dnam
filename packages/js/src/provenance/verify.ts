/**
 * Container provenance verification (`verify`), distribution policy enforcement (`enforcePolicy`),
 * anti-cloning gate evaluation, and preset-safe RFC 8785 JCS `immutableHeaderSha256`
 * (`ref[sdk.ts.provenance.v2-security]`, `ref[sec.jcs.immutable-header-digest]`).
 *
 * @module
 */

import { getActiveDriver } from "../internal/native/load.js";
import { Model } from "../model/model.js";
import type { ParameterValueMap } from "../parameters/typed-model.js";
import { encodeHeader, type ModelHeader } from "../schema/header.js";
import type { ProtectionProfile } from "../types.js";

/**
 * Discriminated provenance verification result returned by `verify(model, options)`.
 */
export type VerificationResult =
  | { readonly status: "unsigned" }
  | {
      readonly status: "creator_verified";
      readonly signerId: string;
      readonly pubkeyEd25519: string;
    }
  | {
      readonly status: "marketplace_verified";
      readonly signerId: string;
      readonly marketplaceId: string;
      readonly licenseId: string;
      readonly licenseTier: string;
    };

/**
 * Options for `verify(model, options)`.
 */
export interface VerifyOptions {
  readonly hasSignaturesFlag?: boolean;
  readonly deviceIdHash?: string;
  readonly deviceSecret?: Uint8Array;
  readonly trustedMarketplacePubkeyHex?: string;
}

/**
 * 3-stage anti-cloning revocation policy action (`ref[sec.anticlone.three-stage-gate]`).
 */
export type AnticloneAction =
  | "no_match_proceed"
  | "flag_license_hold_for_review"
  | "revoke_identified_device_future_licensing_only";

/**
 * Verifies a `Model` (or `(header, arenaBytes)`) against its embedded `Ed25519` creator
 * signature and optional marketplace receipt (`ref[sdk.ts.provenance.v2-security]`).
 */
export function verify<TParams extends ParameterValueMap = Record<string, number | string>>(
  modelOrHeader: Model<TParams> | ModelHeader | string,
  options?: VerifyOptions & { readonly arenaBytes?: Uint8Array },
): VerificationResult {
  let headerJson: string;
  let arenaBytes: Uint8Array;
  let hasSig: boolean;

  if (modelOrHeader instanceof Model) {
    headerJson = modelOrHeader.rawHeaderJson;
    arenaBytes = options?.arenaBytes ?? modelOrHeader.arenaBytes;
    hasSig =
      options?.hasSignaturesFlag ??
      (modelOrHeader.preamble.hasSignatures || modelOrHeader.header.metadata.signatures !== null);
  } else {
    headerJson = typeof modelOrHeader === "string" ? modelOrHeader : encodeHeader(modelOrHeader);
    arenaBytes = options?.arenaBytes ?? new Uint8Array(64_000);
    hasSig = options?.hasSignaturesFlag ?? true;
  }

  const resJson = getActiveDriver().verifyContainerProvenance(
    headerJson,
    arenaBytes,
    hasSig,
    options?.deviceIdHash,
    options?.deviceSecret,
    options?.trustedMarketplacePubkeyHex,
  );
  const raw = JSON.parse(resJson) as {
    status: string;
    signerId: string | null;
    pubkeyEd25519: string | null;
    marketplaceId: string | null;
    licenseId: string | null;
    licenseTier: string | null;
  };
  if (raw.status === "creator_verified") {
    return Object.freeze({
      status: "creator_verified",
      signerId: raw.signerId ?? "",
      pubkeyEd25519: raw.pubkeyEd25519 ?? "",
    });
  }
  if (raw.status === "marketplace_verified") {
    return Object.freeze({
      status: "marketplace_verified",
      signerId: raw.signerId ?? "",
      marketplaceId: raw.marketplaceId ?? "",
      licenseId: raw.licenseId ?? "",
      licenseTier: raw.licenseTier ?? "",
    });
  }
  return Object.freeze({ status: "unsigned" });
}

/**
 * Enforces the creator-signed `DistributionPolicy` in `modelOrHeader` against `activeProfile`.
 */
export function enforcePolicy<TParams extends ParameterValueMap = Record<string, number | string>>(
  modelOrHeader: Model<TParams> | ModelHeader | string,
  activeProfile: ProtectionProfile,
): void {
  const headerJson =
    modelOrHeader instanceof Model
      ? modelOrHeader.rawHeaderJson
      : typeof modelOrHeader === "string"
        ? modelOrHeader
        : encodeHeader(modelOrHeader);
  getActiveDriver().enforceDistributionPolicy(headerJson, activeProfile);
}

/**
 * Evaluates the 3-stage anti-cloning revocation policy (`ref[sec.anticlone.three-stage-gate]`).
 */
export function evaluateAnticlonePolicy(options: {
  readonly fingerprintMatched: boolean;
  readonly buyerWatermarkIdentified: boolean;
  readonly independentDeviceSourceEvidence: boolean;
}): AnticloneAction {
  return getActiveDriver().evaluateAnticlonePolicy(
    options.fingerprintMatched,
    options.buyerWatermarkIdentified,
    options.independentDeviceSourceEvidence,
  ) as AnticloneAction;
}

/**
 * Computes the preset-safe RFC 8785 JCS `immutable_header_sha256` hex digest over `modelOrHeader`
 * (`ref[sec.jcs.immutable-header-digest]`).
 */
export function immutableHeaderSha256(modelOrHeader: Model | ModelHeader | string): string {
  const headerJson =
    modelOrHeader instanceof Model
      ? modelOrHeader.rawHeaderJson
      : typeof modelOrHeader === "string"
        ? modelOrHeader
        : encodeHeader(modelOrHeader);
  return getActiveDriver().immutableHeaderSha256(headerJson);
}
