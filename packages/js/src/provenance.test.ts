import { beforeAll, describe, expect, it } from "vitest";
import { BACKENDS, ensureWasmReady } from "../tests/helpers.js";
import {
  applyIsometricGaugeWatermark,
  applyTardosWatermark,
  buildCreatorSigningMessage,
  buildDnamContainer,
  buildMarketplaceSigningMessage,
  chacha20XorArena,
  computeImmutableHeaderSha256,
  DnamContainer,
  DnamValidationError,
  ed25519PubkeyHex,
  ed25519SignHex,
  extractStructuralHash160d,
  generateTardosCode,
  scoreTardosWatermark,
  sealDnamContainer,
  sha256Hex,
  signOemDeviceCertificate,
  structuralCosineSimilarity,
  unsealDnamContainer,
  verifyOemDeviceCertificate,
  wrapDeviceCekHex,
  x25519PubkeyHex,
} from "./index.js";

describe("Cryptographic Provenance, ChaCha20, Dual Watermarking (Test 8H) & 160-D Structural Hash", () => {
  beforeAll(async () => {
    await ensureWasmReady();
  });

  describe.each(BACKENDS)("provenance (%s)", (backend) => {
    it("computes preset-safe RFC 8785 JCS immutableHeaderSha256", () => {
      const container = DnamContainer.createReference(true, backend);
      const shaOrig = container.immutableHeaderSha256();
      expect(shaOrig).toHaveLength(64);
      expect(computeImmutableHeaderSha256(container.header, backend)).toBe(shaOrig);
      expect(computeImmutableHeaderSha256(JSON.stringify(container.header), backend)).toBe(shaOrig);

      const modifiedHeader = {
        ...container.header,
        presets: [
          ...container.header.presets,
          { id: "user_custom", name: "User Custom", controls: { gain: 8.5 } },
        ],
        userMetadata: { favorite: true, rating: 5 },
      };
      expect(computeImmutableHeaderSha256(modifiedHeader, backend)).toBe(shaOrig);
    });

    it("builds Ed25519 creator and marketplace signing messages and ChaCha20 64B cache-line XOR", () => {
      const container = DnamContainer.createReference(true, backend);
      const hdrSha = container.immutableHeaderSha256();
      const arenaBytes = container.arena;
      const arenaSha = sha256Hex(arenaBytes, backend);
      expect(arenaSha).toHaveLength(64);

      const creatorMsg = buildCreatorSigningMessage(hdrSha, arenaSha, backend);
      expect(creatorMsg.byteLength).toBe(64);

      const receipt = {
        marketplaceId: "dnam.store",
        licenseId: "0192b8e0-1234-7000-8000-000000000001",
        buyerHash: "a".repeat(64),
        licenseTier: "personal" as const,
        issuedAt: "2026-10-23T00:00:00Z",
        buyerArenaSha256: arenaSha,
        receiptSigEd25519: "b".repeat(128),
      };
      const mpMsgObj = buildMarketplaceSigningMessage(hdrSha, arenaSha, receipt, backend);
      const mpMsgJson = buildMarketplaceSigningMessage(
        hdrSha,
        arenaSha,
        JSON.stringify(receipt),
        backend,
      );
      expect(mpMsgObj).toEqual(mpMsgJson);

      const key = new Uint8Array(32).fill(0x42);
      const nonce = new Uint8Array(12).fill(0x07);
      const encrypted = chacha20XorArena(arenaBytes, key, nonce, 0, backend);
      expect(encrypted).not.toEqual(arenaBytes);
      const decrypted = chacha20XorArena(encrypted, key, nonce, 0, backend);
      expect(decrypted).toEqual(arenaBytes);

      // Random-access block offset decryption (cache line 2 = byte offset 128)
      const sliceCipher = encrypted.subarray(128, 256);
      const slicePlain = chacha20XorArena(sliceCipher, key, nonce, 2, backend);
      expect(slicePlain).toEqual(arenaBytes.subarray(128, 256));

      expect(() => chacha20XorArena(arenaBytes, new Uint8Array(16), nonce, 0, backend)).toThrow(
        DnamValidationError,
      );
      expect(() => chacha20XorArena(arenaBytes, key, new Uint8Array(8), 0, backend)).toThrow(
        DnamValidationError,
      );
    });

    it("embeds Dual-Mechanism Buyer Watermarking (Test 8H) and preserves 160-D structural hash", () => {
      const container = DnamContainer.createReference(true, backend);
      const fMaster = container.structuralHash();
      expect(fMaster).toHaveLength(160);
      expect(extractStructuralHash160d(container.arena, backend)).toEqual(fMaster);

      const secret = new Uint8Array(32).fill(0x99);
      const code = generateTardosCode(secret, "darkglass_b7k", "lic_001", backend);
      expect(code).toHaveLength(4608);

      const kappa48 = Array.from({ length: 48 }, (_, i) => 1.0 + 0.01 * ((i % 3) - 1));
      const watermarked = container.withWatermark({
        kappa48,
        tardosCode4608: Array.from(code),
        tardosEpsilon: 1e-4,
      });

      const fBuyer = watermarked.structuralHash();
      const sim = structuralCosineSimilarity(fMaster, fBuyer, backend);
      expect(sim).toBeGreaterThan(0.9999);

      const zScore = scoreTardosWatermark(container.arena, watermarked.arena, code, backend);
      expect(zScore).toBeGreaterThan(60.0);

      // Standalone Mechanism A + B functions
      const qEye = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
      const arenaA = applyIsometricGaugeWatermark(container.arena, kappa48, qEye, backend);
      const fGauge = extractStructuralHash160d(arenaA, backend);
      expect(structuralCosineSimilarity(fMaster, fGauge, backend)).toBeGreaterThan(0.999999);

      const arenaAB = applyTardosWatermark(arenaA, code, 1e-4, backend);
      expect(arenaAB.byteLength).toBe(container.arena.byteLength);

      expect(() => structuralCosineSimilarity(new Float64Array(80), fMaster, backend)).toThrow(
        DnamValidationError,
      );
    });

    it("signs and verifies OEM device certificates, enforces distributionPolicy, and seals/unseals .dnam-seal (DNSL)", () => {
      const base = DnamContainer.createReference(true, backend);
      expect(base.protectionProfile).toBe("open");

      const oemSeed = new Uint8Array(32).fill(0x51);
      const oemPubHex = ed25519PubkeyHex(oemSeed, backend);
      expect(oemPubHex).toHaveLength(64);

      const devSecret = new Uint8Array(32).fill(0x61);
      const devPubHex = x25519PubkeyHex(devSecret, backend);
      expect(devPubHex).toHaveLength(64);
      const devPubBytes = Uint8Array.from(
        Array.from({ length: 32 }, (_, i) =>
          Number.parseInt(devPubHex.slice(i * 2, i * 2 + 2), 16),
        ),
      );
      const devIdHash = "d1".repeat(16);

      const cert = signOemDeviceCertificate({
        oemSeed32: oemSeed,
        oemVendor: "darkglass",
        modelCode: "anagram_rk3582",
        deviceIdHash: devIdHash,
        pubkeyX25519_32: devPubBytes,
        backend,
      });
      expect(cert.oemVendor).toBe("darkglass");
      expect(verifyOemDeviceCertificate(cert, oemPubHex, backend)).toBe(devPubHex);
      expect(verifyOemDeviceCertificate(JSON.stringify(cert), oemPubHex, backend)).toBe(devPubHex);

      // Wrong OEM pubkey fails verification
      const wrongOemPub = ed25519PubkeyHex(new Uint8Array(32).fill(0x52), backend);
      expect(() => verifyOemDeviceCertificate(cert, wrongOemPub, backend)).toThrow(
        DnamValidationError,
      );

      // Wrap device CEK
      const ephSec = new Uint8Array(32).fill(0x71);
      const cek = new Uint8Array(32).fill(0x81);
      const wrappedHex = wrapDeviceCekHex(ephSec, devPubBytes, devIdHash, cek, backend);
      expect(wrappedHex).toHaveLength(96);

      // Sign a test message with ed25519SignHex
      const sigHex = ed25519SignHex(oemSeed, new Uint8Array([1, 2, 3, 4]), backend);
      expect(sigHex).toHaveLength(128);

      // Seal and unseal a container with distributionPolicy restricting to enclave_sealed + darkglass:anagram_rk3582
      const mktSeed = new Uint8Array(32).fill(0x91);
      const mktPubHex = ed25519PubkeyHex(mktSeed, backend);
      const nonce = new Uint8Array(12).fill(0xa1);

      const restrictedHeader = {
        ...base.header,
        metadata: {
          ...base.header.metadata,
          distributionPolicy: {
            allowedProtectionProfiles: ["enclave_sealed" as const],
            allowedOemVendors: ["darkglass:anagram_rk3582"],
          },
        },
      };
      // Open container with enclave_sealed-only distributionPolicy should fail open validation
      const rawOpenRestrictedBytes = buildDnamContainer({
        header: restrictedHeader,
        arenaBytes: base.arena,
        backend,
      });
      expect(() => DnamContainer.fromBytes(rawOpenRestrictedBytes, backend)).toThrow(
        /distribution_policy/,
      );

      // Seal the restricted inner container into a .dnam-seal (DNSL) envelope
      const sealedBytes = sealDnamContainer({
        innerDnamBytes: rawOpenRestrictedBytes,
        recipientCerts: [cert],
        trustedOemPubkeys: { darkglass: oemPubHex },
        marketplaceSeed32: mktSeed,
        ephemeralSecret32: ephSec,
        cek32: cek,
        nonce12: nonce,
        backend,
      });
      expect(new TextDecoder().decode(sealedBytes.subarray(0, 4))).toBe("DNSL");
      expect(sealedBytes.byteLength % 64).toBe(0);

      // Also verify container.seal() on a DnamContainer instance
      const sealedFromInstance = base.seal({
        recipientCerts: [cert],
        trustedOemPubkeys: { darkglass: oemPubHex },
        marketplaceSeed32: mktSeed,
        ephemeralSecret32: ephSec,
        cek32: cek,
        nonce12: nonce,
        backend,
      });
      expect(new TextDecoder().decode(sealedFromInstance.subarray(0, 4))).toBe("DNSL");

      // Unseal into raw inner .dnam bytes
      const unsealedInner = unsealDnamContainer(sealedBytes, {
        deviceIdHash: devIdHash,
        deviceSecretX25519: devSecret,
        trustedMarketplacePubkeyHex: mktPubHex,
        backend,
      });
      expect(unsealedInner).toEqual(rawOpenRestrictedBytes);

      // Validate/unseal directly via DnamContainer.unseal / DnamContainer.fromBytes
      const unsealedContainer = DnamContainer.unseal(sealedBytes, {
        deviceIdHash: devIdHash,
        deviceSecretX25519: devSecret,
        trustedMarketplacePubkeyHex: mktPubHex,
        backend,
      });
      expect(unsealedContainer.protectionProfile).toBe("enclave_sealed");
      expect(unsealedContainer.arena).toEqual(base.arena);

      // Disallowed OEM vendor in distributionPolicy fails sealDnamContainer
      const badCert = signOemDeviceCertificate({
        oemSeed32: oemSeed,
        oemVendor: "other_oem",
        modelCode: "pedal_x",
        deviceIdHash: devIdHash,
        pubkeyX25519_32: devPubBytes,
        backend,
      });
      expect(() =>
        sealDnamContainer({
          innerDnamBytes: rawOpenRestrictedBytes,
          recipientCerts: [badCert],
          trustedOemPubkeys: { other_oem: oemPubHex },
          marketplaceSeed32: mktSeed,
          ephemeralSecret32: ephSec,
          cek32: cek,
          nonce12: nonce,
          backend,
        }),
      ).toThrow(/distribution_policy/);
    });
  });
});
