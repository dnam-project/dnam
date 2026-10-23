import type { StandardSchemaV1 } from "@standard-schema/spec";
import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  AuditReportSchema,
  AuthorizedDeviceSchema,
  ClampResultSnapshotSchema,
  ControlSchemaDefSchema,
  ControlUiHintsSchema,
  CreatorSignatureSchema,
  computeSkewExponent,
  controlSchemaDefToWire,
  DatasetProvenanceSchema,
  DnamContainer,
  DnamContainerError,
  DnamControlError,
  DnamError,
  DnamJsonHeaderSchema,
  DnamSampleRateError,
  DnamValidationError,
  dnamJsonHeaderToWire,
  ExtractedPhysicsSchema,
  MarketplaceReceiptSchema,
  ModelSignaturesSchema,
  ModulatedUfdnParamsSchema,
  mapTaggedError,
  normalizedToPhysical,
  PolyAdaaParamsSchema,
  PreambleSnapshotSchema,
  PresetDefSchema,
  parseTier,
  physicalToNormalized,
  RoyaltySplitSchema,
  rethrowDnamError,
  SelectiveRgLruParamsSchema,
  StemProvenanceSchema,
  StemRegistryEntrySchema,
  TelemetrySnapshotSchema,
  TierSchema,
  tierToName,
  validateStandardSchema,
  ZohParamsSnapshotSchema,
} from "../src/index.js";

describe("Schema & Error Hierarchy Suite", () => {
  it("maps tagged Rust/FFI errors to exact DnamError subclasses", () => {
    const cErr = mapTaggedError("[DNAM-CONTAINER] Bad magic");
    expect(cErr).toBeInstanceOf(DnamContainerError);
    expect(cErr.code).toBe("DNAM_CONTAINER_ERROR");
    expect(cErr.message).toBe("Bad magic");

    const sErr = mapTaggedError("[DNAM-SAMPLERATE] Unsupported 12000 Hz");
    expect(sErr).toBeInstanceOf(DnamSampleRateError);

    const ctrlErr = mapTaggedError("[DNAM-CONTROL] Invalid control");
    expect(ctrlErr).toBeInstanceOf(DnamControlError);

    const valErr = mapTaggedError("[DNAM-VALIDATION] Unaligned offset");
    expect(valErr).toBeInstanceOf(DnamValidationError);

    const genErr = mapTaggedError("Untagged error");
    expect(genErr).toBeInstanceOf(DnamError);

    expect(() => rethrowDnamError(cErr)).toThrow(DnamContainerError);
    expect(() => rethrowDnamError(new Error("[DNAM-SAMPLERATE] 8000"))).toThrow(
      DnamSampleRateError,
    );
    expect(() => rethrowDnamError("[DNAM-CONTROL] bad")).toThrow(DnamControlError);
  });

  it("parses and formats slimmable tiers", () => {
    expect(parseTier("lite")).toBe(0);
    expect(parseTier("Full")).toBe(1);
    expect(parseTier("ULTRA")).toBe(2);
    expect(parseTier(0)).toBe(0);
    expect(parseTier(1)).toBe(1);
    expect(parseTier(2)).toBe(2);
    expect(() => parseTier(9)).toThrow(DnamValidationError);
    expect(() => parseTier("invalid" as "lite")).toThrow(DnamValidationError);

    expect(tierToName(0)).toBe("lite");
    expect(tierToName(1)).toBe("full");
    expect(tierToName(2)).toBe("ultra");
    expect(() => tierToName(5)).toThrow(DnamValidationError);
    expect(TierSchema.parse("full")).toBe("full");
  });

  it("validates via StandardSchemaV1 (~standard) and handles sync/async error paths", () => {
    const validHints = validateStandardSchema(ControlUiHintsSchema, {
      section: "Preamp",
      order: 1,
      midpoint: 4.0,
    });
    expect(validHints.section).toBe("Preamp");
    expect(validHints.midpoint).toBe(4.0);

    expect(() =>
      validateStandardSchema(ControlSchemaDefSchema, { id: "x", name: "X", default: 1.0 }),
    ).toThrow(DnamValidationError);

    const fakeAsyncSchema: StandardSchemaV1<string, string> = {
      "~standard": {
        version: 1,
        vendor: "test",
        validate: () => Promise.resolve({ value: "ok" }),
      },
    };
    expect(() => validateStandardSchema(fakeAsyncSchema, "x")).toThrow(DnamValidationError);
  });

  it("satisfies midpoint power-skew round-trip property across [0, 1]", () => {
    expect(computeSkewExponent({ midpoint: 5.0 }, 5.0, 5.0)).toBe(1.0);
    expect(computeSkewExponent({ midpoint: 5.0 }, 0.0, 10.0)).toBe(1.0);
    expect(normalizedToPhysical({ midpoint: 2.5 }, 0.0, 0.0, 10.0)).toBe(0.0);
    expect(normalizedToPhysical({ midpoint: 2.5 }, 1.0, 0.0, 10.0)).toBe(10.0);
    expect(physicalToNormalized({ midpoint: 2.5 }, 0.0, 5.0, 5.0)).toBe(0.0);
    expect(physicalToNormalized({ midpoint: 2.5 }, 0.0, 0.0, 10.0)).toBe(0.0);
    expect(physicalToNormalized({ midpoint: 2.5 }, 10.0, 0.0, 10.0)).toBe(1.0);

    fc.assert(
      fc.property(
        fc.double({ min: 1.0, max: 9.0, noNaN: true }),
        fc.double({ min: 0.01, max: 0.99, noNaN: true }),
        (midpoint, tNorm) => {
          const hints = { midpoint };
          const xPhys = normalizedToPhysical(hints, tNorm, 0.0, 10.0);
          const tBack = physicalToNormalized(hints, xPhys, 0.0, 10.0);
          return Math.abs(tBack - tNorm) < 1e-6;
        },
      ),
    );
  });

  it("parses and round-trips DnamJsonHeader, signatures, stems, and runtime schemas", () => {
    const refContainer = DnamContainer.buildReference({ core2Active: true, isCheckpoint: true });
    const header = refContainer.header;
    expect(header.containerKind).toBe("dnam-ckpt");
    expect(header.core2Mufdn.active).toBe(true);

    const wire = dnamJsonHeaderToWire(header);
    const reParsed = DnamJsonHeaderSchema.parse(wire);
    expect(reParsed).toEqual(header);

    const firstCtrl = header.controlsSchema[0];
    if (firstCtrl) {
      const ctrlWire = controlSchemaDefToWire(firstCtrl);
      expect(ControlSchemaDefSchema.parse(ctrlWire)).toEqual(firstCtrl);
    }

    const preset = PresetDefSchema.parse({
      id: "clean",
      name: "Clean",
      controls: { gain: 2.0 },
    });
    expect(preset.id).toBe("clean");

    const prov = DatasetProvenanceSchema.parse({
      datasetManifestSha256: "a".repeat(64),
      epochsTrained: 100,
    });
    expect(prov.epochsTrained).toBe(100);

    const dev = AuthorizedDeviceSchema.parse({
      oemVendor: "darkglass",
      modelCode: "anagram",
      deviceIdHash: "b".repeat(64),
    });
    expect(dev.oemVendor).toBe("darkglass");

    const split = RoyaltySplitSchema.parse({
      recipientId: "did:web:creator",
      role: "amp_capture",
      shareBps: 10000,
    });
    expect(split.shareBps).toBe(10000);

    const creatorSig = CreatorSignatureSchema.parse({
      signerId: "did:web:creator",
      pubkeyEd25519: "c".repeat(64),
      sigEd25519: "d".repeat(128),
    });
    expect(() => CreatorSignatureSchema.parse({ signerId: "only" })).toThrow();

    const receipt = MarketplaceReceiptSchema.parse({
      marketplaceId: "dnam.store",
      licenseId: "0192b5",
      buyerHash: "e".repeat(64),
      licenseTier: "personal",
      issuedAt: "2026-10-23T00:00:00Z",
      buyerArenaSha256: "f".repeat(64),
      authorizedDevices: [dev],
      royaltySplits: [split],
      receiptSigEd25519: "1".repeat(128),
    });

    const sigs = ModelSignaturesSchema.parse({
      immutableHeaderSha256: "a".repeat(64),
      masterArenaSha256: "b".repeat(64),
      creatorSignature: creatorSig,
      marketplaceReceipt: receipt,
    });
    expect(sigs.marketplaceReceipt?.marketplaceId).toBe("dnam.store");
    expect(() => ModelSignaturesSchema.parse({ immutableHeaderSha256: "a" })).toThrow();

    const stemProv = StemProvenanceSchema.parse({
      inputPcmSha256: "1".repeat(64),
      targetPcmSha256: "2".repeat(64),
      numFrames: 48000,
    });
    const phys = ExtractedPhysicsSchema.parse({
      delaysMs: [12.5],
      lfoFreqsHz: [1.2],
      decayGainsLin: [0.8],
    });
    const stemEntry = StemRegistryEntrySchema.parse({
      stemId: "stem_0",
      controlsMeta: { gain: 5.0 },
      provenance: stemProv,
      stemArena3Crc32c: 123,
      extractedPhysics: phys,
      gAudioUploUF64: {
        arena: 3,
        dtype: "f64",
        packing: "uplo_upper_col_major_tier_ordered",
        featureDim: 64,
        packedLen: 2080,
        byteOffset: 0,
        byteLength: 16640,
      },
      cAudioTierF64: {
        arena: 3,
        dtype: "f64",
        shape: [64, 2],
        byteOffset: 16640,
        byteLength: 1024,
      },
    });
    const audit = AuditReportSchema.parse({
      phase1Clamps: ["CLAMP-RANGE"],
      prescriptions: ["Add max stem"],
    });

    const fullHeader = DnamJsonHeaderSchema.parse({
      ...header,
      metadata: {
        ...header.metadata,
        signatures: sigs,
      },
      tensorDirectory: {
        core1_weights: {
          arena: 0,
          dtype: "f32",
          shape: [64],
          byteOffset: 0,
          byteLength: 256,
        },
      },
      stemsRegistry: [stemEntry],
      auditReport: audit,
    });
    const fullWire = dnamJsonHeaderToWire(fullHeader);
    expect(DnamJsonHeaderSchema.parse(fullWire)).toEqual(fullHeader);
    expect(() => DnamJsonHeaderSchema.parse({ metadata: header.metadata })).toThrow();

    expect(PreambleSnapshotSchema.parse(refContainer.preamble)).toEqual(refContainer.preamble);
    const zoh = refContainer.computeZoh(48_000);
    expect(ZohParamsSnapshotSchema.parse(zoh)).toEqual(zoh);
    expect(
      ClampResultSnapshotSchema.parse({
        clampedControls: { gain: 5.0 },
        normalizedU: { gain: 0.0 },
        wasClamped: { gain: false },
        warnings: {},
      }),
    ).toBeDefined();
    expect(PolyAdaaParamsSchema.parse({ beta: 0.5, delta: 0.1 })).toBeDefined();
    expect(SelectiveRgLruParamsSchema.parse({ rLow: 0.9, drSpan: 0.05 })).toBeDefined();
    expect(() => SelectiveRgLruParamsSchema.parse({ rLow: 0.98, drSpan: 0.05 })).toThrow();
    expect(ModulatedUfdnParamsSchema.parse({ gLin: 0.85, vCeil: 0.985 })).toBeDefined();
    expect(TelemetrySnapshotSchema).toBeDefined();
  });
});
