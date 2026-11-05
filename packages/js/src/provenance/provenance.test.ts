/**
 * Tests for `@dnam-project/dnam/provenance` (`verify`, `signCreatorHeader`,
 * `signMarketplaceReceipt`, `seal`, `unseal`, `encodeDau1`/`decodeDau1`, `DME1`, `DMU1`,
 * preview containment, and dual-mechanism buyer watermarking) across both `NapiDriver` and `WasmDriver`.
 */

import "../node.js";

import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { DnamError } from "../errors.js";
import { withDriver } from "../internal/native/load.js";
import { loadNodeDriverSync, loadNodeWasmDriverSync } from "../internal/native/napi.js";
import { Model } from "../model/model.js";
import {
  applyGaugeWatermark,
  applyTardosWatermark,
  buildCreatorSigningMessage,
  buildDme1Header,
  buildDmu1Frame,
  buildMarketplaceSigningMessage,
  buildOemCertV2SigningMessage,
  bytesToHex,
  chacha20XorArena,
  computePreviewWindow,
  decodeDau1,
  deriveWatermarkSeed,
  ed25519PublicKey,
  ed25519Sign,
  encodeDau1,
  encodeHpkeSlots,
  encodeRecipientKeyIds,
  encodeRecordDirectory,
  enforcePolicy,
  evaluateAnticlonePolicy,
  extractStage3Fingerprint,
  extractStructuralHash,
  generateTardosCode,
  hexToBytes,
  hmacSha256Hex,
  immutableHeaderSha256,
  parseDme1Plaintext,
  parseDmu1Frame,
  parseHpkeSlots,
  parseRecipientKeyIds,
  parseRecordDirectory,
  scoreTardosWatermark,
  seal,
  sha256Hex,
  signCreatorHeader,
  signDeviceCertificate,
  signMarketplaceReceipt,
  structuralCosineSimilarity,
  unseal,
  validatePreviewSwitchOptions,
  validateRoyaltySplits,
  verify,
  verifyDeviceCertificate,
  wrapDeviceCek,
  x25519PublicKey,
} from "./index.js";

describe("Provenance, Sealing, Envelopes & Watermarking", () => {
  it.each(["napi", "wasm"] as const)(
    "signs, verifies, seals/unseals, and watermarks models identically on %s driver",
    (driverKind) => {
      // verify[sdk.ts.provenance.v2-security]
      const driver = driverKind === "napi" ? loadNodeDriverSync() : loadNodeWasmDriverSync();
      withDriver(driver, () => {
        using baseModel = Model.reference();
        const unsignedStatus = verify(baseModel);
        expect(unsignedStatus.status).toBe("unsigned");

        const creatorSeed = new Uint8Array(32).fill(0x11);
        const mktSeed = new Uint8Array(32).fill(0x22);
        const oemSeed = new Uint8Array(32).fill(0x33);
        const deviceSecret = new Uint8Array(32).fill(0x44);
        const ephSecret = new Uint8Array(32).fill(0x55);
        const cek = new Uint8Array(32).fill(0x66);
        const nonce12 = new Uint8Array(12).fill(0x77);

        expect(bytesToHex(hexToBytes("deadbeef"))).toBe("deadbeef");
        expect(sha256Hex(creatorSeed)).toHaveLength(64);
        expect(hmacSha256Hex(creatorSeed, mktSeed)).toHaveLength(64);
        expect(ed25519Sign(creatorSeed, new Uint8Array([1, 2, 3]))).toHaveLength(128);

        const mktPubkeyHex = ed25519PublicKey(mktSeed);
        const oemPubkeyHex = ed25519PublicKey(oemSeed);
        const devicePubkeyHex = x25519PublicKey(deviceSecret);
        const devicePubkeyBytes = hexToBytes(devicePubkeyHex);
        const deviceIdHash = "aa".repeat(16);

        // 1. Creator signing & verification
        const creatorSignedHdr = signCreatorHeader({
          header: baseModel.header,
          masterArena: baseModel.arenaBytes,
          creatorSeed,
          signerId: "did:web:dnam.audio:creators:alice",
          includeTierPrefixes: true,
        });
        expect(immutableHeaderSha256(creatorSignedHdr)).toBe(
          immutableHeaderSha256(baseModel.header),
        );
        using creatorSignedModel = Model.build(creatorSignedHdr, baseModel.arenaBytes, {
          hasSignatures: true,
        });
        const creatorRes = verify(creatorSignedModel);
        expect(creatorRes.status).toBe("creator_verified");
        expect(verify(creatorSignedHdr, { arenaBytes: baseModel.arenaBytes }).status).toBe(
          "creator_verified",
        );

        // 2. Dual-mechanism buyer watermarking + structural hash
        const kappa64 = new Float32Array(64).fill(1.0);
        const qIdentity4x4 = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
        const gaugeArena = applyGaugeWatermark({
          arenaBytes: baseModel.arenaBytes,
          kappa64,
          qBuyer4x4: qIdentity4x4,
        });
        const tardosCode = generateTardosCode({
          marketplaceSecret: mktSeed,
          modelId: "model-01",
          licenseId: "lic-01",
        });
        expect(tardosCode.length).toBe(7680);
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
        expect(zScore).toBeGreaterThan(10.0);

        const hashMaster = extractStructuralHash(baseModel);
        const hashGauge = extractStructuralHash(gaugeArena);
        expect(hashMaster.length).toBe(224);
        expect(structuralCosineSimilarity(hashMaster, hashGauge)).toBeCloseTo(1.0, 6);

        const stage3Fp = extractStage3Fingerprint({
          earlyFir128b: new Uint8Array(128),
          kautzModalBytes: new Uint8Array(256),
          addedModes: 8,
        });
        expect(stage3Fp.length).toBe(88);

        const wmSeedHex = deriveWatermarkSeed({
          key32: mktSeed,
          marketplacePubkey32: hexToBytes(mktPubkeyHex),
          watermarkKeyId16: new Uint8Array(16).fill(1),
          creatorSignedManifestSha25632: new Uint8Array(32).fill(2),
          modelId16: new Uint8Array(16).fill(3),
          watermarkProfileSha25632: new Uint8Array(32).fill(4),
          owningTier: 1,
          licenseId16: new Uint8Array(16).fill(5),
          buyerId16: new Uint8Array(16).fill(6),
        });
        expect(wmSeedHex).toHaveLength(64);

        // 3. Marketplace receipt signing & verification
        const splits = [
          { recipientId: "alice", role: "amp_capture", shareBps: 7_000 },
          { recipientId: "bob", role: "ir_cabinet", shareBps: 3_000 },
        ] as const;
        validateRoyaltySplits("commercial_studio", splits);

        const mktHdr = signMarketplaceReceipt({
          header: creatorSignedHdr,
          buyerArena,
          marketplaceSeed: mktSeed,
          marketplaceId: "dnam.store",
          licenseId: "0192f6c0-0000-7000-8000-000000000001",
          buyerHash: "bb".repeat(32),
          licenseTier: "commercial_studio",
          royaltySplits: splits,
          includeBuyerTierPrefixes: true,
        });
        using mktModel = Model.build(mktHdr, buyerArena, { hasSignatures: true });
        const mktRes = verify(mktModel, {
          trustedMarketplacePubkeyHex: mktPubkeyHex,
        });
        expect(mktRes.status).toBe("marketplace_verified");

        const creatorMsg = buildCreatorSigningMessage("11".repeat(32), "22".repeat(32));
        expect(creatorMsg.byteLength).toBe(64);
        const mktReceipt = mktHdr.metadata.signatures?.marketplaceReceipt;
        expect(mktReceipt).toBeDefined();
        if (mktReceipt) {
          const mktMsg = buildMarketplaceSigningMessage(
            "11".repeat(32),
            "22".repeat(32),
            mktReceipt,
          );
          expect(mktMsg.byteLength).toBeGreaterThan(64);
        }

        // 4. OEM Device Certificate + .dnam-seal (DNSL) hardware envelope
        const cert = signDeviceCertificate({
          oemSeed,
          oemVendor: "darkglass",
          modelCode: "anagram_rk3582",
          deviceIdHash,
          pubkeyX25519: devicePubkeyBytes,
        });
        expect(verifyDeviceCertificate(cert, oemPubkeyHex)).toBe(devicePubkeyHex);

        const certV2Msg = buildOemCertV2SigningMessage({
          oemVendor: "darkglass",
          modelCode: "anagram_rk3582",
          deviceIdHash16: hexToBytes(deviceIdHash),
          pubkeyX2551932: devicePubkeyBytes,
          issuerPubkeyEd2551932: hexToBytes(oemPubkeyHex),
        });
        expect(certV2Msg.byteLength).toBeGreaterThan(64);

        const wrappedCek = wrapDeviceCek({
          ephemeralSecret: ephSecret,
          devicePubkey: devicePubkeyBytes,
          deviceIdHash,
          cek,
        });
        expect(wrappedCek).toHaveLength(96);

        const xored = chacha20XorArena({
          arenaBytes: new Uint8Array(128).fill(0x42),
          key: cek,
          nonce12,
        });
        const unxored = chacha20XorArena({
          arenaBytes: xored,
          key: cek,
          nonce12,
        });
        expect(unxored).toEqual(new Uint8Array(128).fill(0x42));

        const sealedBytes = seal(mktModel, {
          recipientCertificates: [cert],
          trustedOemPubkeysHex: { darkglass: oemPubkeyHex },
          marketplaceSeed: mktSeed,
          ephemeralSecret: ephSecret,
          cek,
          nonce12,
        });
        expect(new TextDecoder().decode(sealedBytes.subarray(0, 4))).toBe("DNSL");

        using unsealedModel = unseal(sealedBytes, {
          deviceIdHash,
          deviceSecret,
          trustedMarketplacePubkeyHex: mktPubkeyHex,
        });
        expect(unsealedModel.name).toBe(mktModel.name);

        // 5. Policy, Anti-cloning gate, DAU1/DME1/DMU1, HPKE slots, Record Dir, Preview containment
        enforcePolicy(baseModel, "open");
        expect(
          evaluateAnticlonePolicy({
            fingerprintMatched: true,
            buyerWatermarkIdentified: true,
            independentDeviceSourceEvidence: true,
          }),
        ).toBe("revoke_identified_device_future_licensing_only");

        const dau1Wire = encodeDau1({
          authKind: 0,
          creatorPubkeyHex: ed25519PublicKey(creatorSeed),
          creatorMaxTier: 3,
          masterPrefixesHex: ["11".repeat(32), "22".repeat(32), "33".repeat(32), "44".repeat(32)],
          creatorSignatureHex: "55".repeat(64),
        });
        const dau1Decoded = decodeDau1(dau1Wire);
        expect(dau1Decoded.authKind).toBe(0);
        expect(dau1Decoded.creatorMaxTier).toBe(3);

        const dme1Hdr = buildDme1Header({
          profile: 1,
          recipientCount: 1,
          ciphertextLength: 128,
        });
        expect(dme1Hdr.byteLength).toBe(20);

        const dmu1Frame = buildDmu1Frame({ presets: [] }, 256);
        expect(parseDmu1Frame(dmu1Frame)).toEqual({ presets: [] });

        const keyIds = encodeRecipientKeyIds(["aa".repeat(32)]);
        expect(parseRecipientKeyIds(keyIds)).toEqual(["aa".repeat(32)]);

        const slots = encodeHpkeSlots([["bb".repeat(32), "cc".repeat(48)]]);
        expect(parseHpkeSlots(slots, 1)).toEqual([["bb".repeat(32), "cc".repeat(48)]]);

        const recDir = encodeRecordDirectory(0, [
          [4096, 10752],
          [14848, 4096],
        ]);
        expect(parseRecordDirectory(recDir, 0)).toEqual([
          [4096, 10752],
          [14848, 4096],
        ]);

        const [wLo, wHi] = computePreviewWindow({
          anchor: 0.5,
          delta: 0.1,
          safeBounds: [0.0, 1.0],
        });
        expect(wLo).toBeCloseTo(0.4, 6);
        expect(wHi).toBeCloseTo(0.6, 6);

        expect(
          validatePreviewSwitchOptions({
            preset: "off",
            alternative: "on",
            observed: ["off", "on"],
            active: ["off", "on"],
            safe: ["off", "on"],
          }),
        ).toEqual(["off", "on"]);

        // Rejects .dnam-prev (DNPR) container with ERR_DNAM_PROVENANCE
        const dnprBytes = new Uint8Array(64);
        dnprBytes.set([0x44, 0x4e, 0x50, 0x52], 0); // "DNPR"
        expect(() => Model.from(dnprBytes)).toThrow(DnamError);

        // parseDme1Plaintext validation
        expect(() => parseDme1Plaintext(new Uint8Array([1, 2, 3]), ["aa".repeat(32)])).toThrow(
          DnamError,
        );
      });
    },
  );

  it("property: fast-check verifies ChaCha20 arena involution, DAU1/RecordDirectory wire roundtrips, and preview window containment", () => {
    // verify[sdk.ts.provenance.v2-security]
    const creatorPubHex = ed25519PublicKey(new Uint8Array(32).fill(0x11));

    fc.assert(
      fc.property(
        fc.uint8Array({ minLength: 64, maxLength: 64 }),
        fc.uint8Array({ minLength: 32, maxLength: 32 }),
        fc.uint8Array({ minLength: 12, maxLength: 12 }),
        (arena, key, nonce) => {
          const hex = bytesToHex(arena);
          expect(hexToBytes(hex)).toEqual(arena);

          const cipher = chacha20XorArena({ arenaBytes: arena, key, nonce12: nonce });
          const plain = chacha20XorArena({ arenaBytes: cipher, key, nonce12: nonce });
          expect(plain).toEqual(arena);
        },
      ),
      { numRuns: 40 },
    );

    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: 3 }),
        fc.integer({ min: 1, max: 64 }),
        (tierIdx, trailerBlocks) => {
          const allPrefixes = ["11".repeat(32), "22".repeat(32), "33".repeat(32), "44".repeat(32)];
          const dau1 = encodeDau1({
            authKind: 0,
            creatorPubkeyHex: creatorPubHex,
            creatorMaxTier: tierIdx,
            masterPrefixesHex: allPrefixes.slice(0, tierIdx + 1),
            creatorSignatureHex: "55".repeat(64),
          });
          const decoded = decodeDau1(dau1);
          expect(decoded.creatorMaxTier).toBe(tierIdx);
          expect(decoded.creatorPubkeyHex).toBe(creatorPubHex);

          const trailerLen = trailerBlocks * 64;
          const records: readonly (readonly [number, number])[] = [
            [4096, 10752],
            [14848, trailerLen],
          ];
          const dirBytes = encodeRecordDirectory(0, records);
          expect(parseRecordDirectory(dirBytes, 0)).toEqual(records);
        },
      ),
      { numRuns: 30 },
    );

    fc.assert(
      fc.property(
        fc.double({ min: 0.0, max: 1.0, noNaN: true }),
        fc.double({ min: 0.01, max: 0.15, noNaN: true }),
        (anchor, delta) => {
          const safeBounds = [0.0, 1.0] as const;
          const [lo, hi] = computePreviewWindow({ anchor, delta, safeBounds });
          expect(lo).toBeGreaterThanOrEqual(safeBounds[0] - 1e-6);
          expect(hi).toBeLessThanOrEqual(safeBounds[1] + 1e-6);
          expect(lo).toBeLessThan(hi);
        },
      ),
      { numRuns: 50 },
    );
  });
});
