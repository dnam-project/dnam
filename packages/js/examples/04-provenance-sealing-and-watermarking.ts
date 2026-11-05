/**
 * Example 04 — Provenance Signing, Dual-Mechanism Buyer Watermarking & `.dnam-seal` Envelopes.
 *
 * Run via:
 *   node packages/js/examples/04-provenance-sealing-and-watermarking.ts
 */

import { Model } from "../dist/node.js";
import {
  applyGaugeWatermark,
  applyTardosWatermark,
  ed25519PublicKey,
  extractStructuralHash,
  generateTardosCode,
  hexToBytes,
  scoreTardosWatermark,
  seal,
  signCreatorHeader,
  signDeviceCertificate,
  signMarketplaceReceipt,
  structuralCosineSimilarity,
  unseal,
  verify,
  x25519PublicKey,
} from "../dist/provenance/index.js";

using baseModel = Model.reference();
const creatorSeed = new Uint8Array(32).fill(0x11);
const mktSeed = new Uint8Array(32).fill(0x22);
const oemSeed = new Uint8Array(32).fill(0x33);
const deviceSecret = new Uint8Array(32).fill(0x44);

// 1. Creator Ed25519 signature
const creatorHdr = signCreatorHeader({
  header: baseModel.header,
  masterArena: baseModel.arenaBytes,
  creatorSeed,
  signerId: "did:web:dnam.audio:creators:studio-a",
});

// 2. Dual-mechanism buyer watermark (Stage 0 Gauge + Stage II Tardos)
const gaugeArena = applyGaugeWatermark({
  arenaBytes: baseModel.arenaBytes,
  kappa64: new Float32Array(64).fill(1.0),
  qBuyer4x4: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
});
const tardosCode = generateTardosCode({
  marketplaceSecret: mktSeed,
  modelId: "rig-001",
  licenseId: "lic-001",
});
const buyerArena = applyTardosWatermark({
  arenaBytes: gaugeArena,
  code7680: tardosCode,
  epsilon: 1e-3,
});
const zScore = scoreTardosWatermark({
  masterArena: baseModel.arenaBytes,
  suspectArena: buyerArena,
  code7680: tardosCode,
});
const cosSim = structuralCosineSimilarity(
  extractStructuralHash(baseModel),
  extractStructuralHash(buyerArena),
);
console.log(
  `Buyer watermark embedded: Tardos Z=${zScore.toFixed(2)}, structural cosine=${cosSim.toFixed(6)}`,
);

// 3. Marketplace receipt & hardware .dnam-seal (DNSL) envelope
const mktHdr = signMarketplaceReceipt({
  header: creatorHdr,
  buyerArena,
  marketplaceSeed: mktSeed,
  marketplaceId: "dnam.store",
  licenseId: "0192f6c0-0000-7000-8000-000000000001",
  buyerHash: "bb".repeat(32),
  licenseTier: "commercial_studio",
  royaltySplits: [{ recipientId: "studio-a", role: "amp_capture", shareBps: 10_000 }],
});
using mktModel = Model.build(mktHdr, buyerArena, { hasSignatures: true });

const deviceIdHash = "aa".repeat(16);
const cert = signDeviceCertificate({
  oemSeed,
  oemVendor: "darkglass",
  modelCode: "anagram_rk3582",
  deviceIdHash,
  pubkeyX25519: hexToBytes(x25519PublicKey(deviceSecret)),
});
const sealedBytes = seal(mktModel, {
  recipientCertificates: [cert],
  trustedOemPubkeysHex: { darkglass: ed25519PublicKey(oemSeed) },
  marketplaceSeed: mktSeed,
  ephemeralSecret: new Uint8Array(32).fill(0x55),
  cek: new Uint8Array(32).fill(0x66),
  nonce12: new Uint8Array(12).fill(0x77),
});
using unsealed = unseal(sealedBytes, {
  deviceIdHash,
  deviceSecret,
  trustedMarketplacePubkeyHex: ed25519PublicKey(mktSeed),
});
console.log(
  "Unsealed hardware model verification:",
  verify(unsealed, {
    trustedMarketplacePubkeyHex: ed25519PublicKey(mktSeed),
  }),
);
