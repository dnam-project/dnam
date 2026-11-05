/**
 * Curated exports for `@dnam-project/dnam/provenance` (`ref[sdk.ts.layout.curated-exports]`,
 * `ref[sdk.ts.provenance.v2-security]`).
 *
 * @module
 */

export {
  buildDme1Header,
  buildDmu1Frame,
  type Dau1EnvelopeSpec,
  type DecodedDau1Envelope,
  decodeDau1,
  encodeDau1,
  encodeHpkeSlots,
  encodeRecipientKeyIds,
  encodeRecordDirectory,
  parseDme1Plaintext,
  parseDmu1Frame,
  parseHpkeSlots,
  parseRecipientKeyIds,
  parseRecordDirectory,
} from "./dau1.js";
export {
  bytesToHex,
  ed25519PublicKey,
  ed25519Sign,
  hexToBytes,
  hmacSha256Hex,
  sha256Hex,
  x25519PublicKey,
} from "./keys.js";
export {
  computePreviewWindow,
  type PreviewSwitchSpec,
  validatePreviewSwitchOptions,
} from "./preview.js";
export {
  chacha20XorArena,
  type SealOptions,
  seal,
  type UnsealOptions,
  unseal,
  wrapDeviceCek,
} from "./seal.js";
export {
  buildCreatorSigningMessage,
  buildMarketplaceSigningMessage,
  buildOemCertV2SigningMessage,
  signCreatorHeader,
  signDeviceCertificate,
  signMarketplaceReceipt,
  validateRoyaltySplits,
  verifyDeviceCertificate,
} from "./sign.js";
export {
  type AnticloneAction,
  enforcePolicy,
  evaluateAnticlonePolicy,
  immutableHeaderSha256,
  type VerificationResult,
  type VerifyOptions,
  verify,
} from "./verify.js";
export {
  applyGaugeWatermark,
  applyTardosWatermark,
  deriveWatermarkSeed,
  extractStage3Fingerprint,
  extractStructuralHash,
  generateTardosCode,
  scoreTardosWatermark,
  structuralCosineSimilarity,
  type WatermarkSeedSpec,
} from "./watermark.js";
