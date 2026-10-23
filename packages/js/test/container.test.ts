import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  buildDnamContainer,
  buildReferenceContainer,
  computeZohRescaledParams,
  crc32c,
  DnamContainer,
  DnamContainerError,
  DnamSampleRateError,
  DnamValidationError,
  GramMatrix,
  toUint8Array,
  updateHeaderInPlace,
  validateDnamModel,
} from "../src/index.js";
import { BACKENDS, maxAbsDiff } from "./helpers.js";

describe.each(BACKENDS)("Container, ZOH & GramMatrix Suite (%s)", (backend) => {
  it("builds, validates, and round-trips .dnam and .dnam-ckpt containers", () => {
    const refBytes = buildReferenceContainer(false, false, backend);
    const container = validateDnamModel(refBytes, backend);
    expect(container.preamble.magic).toBe("DNAM");
    expect(container.preamble.tensorArenaOffset).toBe(4096);
    expect(container.preamble.tensorArenaOffset % 64).toBe(0);
    expect(container.verificationStatus).toBe("Unsigned");
    expect(container.header.controlsSchema).toHaveLength(2);

    // ArrayBuffer & DataView input conversions
    expect(toUint8Array(refBytes.buffer).byteLength).toBeGreaterThanOrEqual(refBytes.byteLength);
    const view = new DataView(refBytes.buffer, refBytes.byteOffset, refBytes.byteLength);
    expect(toUint8Array(view)).toEqual(refBytes);

    // Build from header object, JSON string, and with plaintextArenaCrcSource
    const rebuiltBytes = buildDnamContainer({
      header: container.header,
      arenaBytes: container.arenaBytes,
      plaintextArenaCrcSource: container.arenaBytes,
      backend,
    });
    const rebuilt = DnamContainer.fromBytes(rebuiltBytes, backend);
    expect(rebuilt.preamble.arenaCrc32cUltra).toBe(container.preamble.arenaCrc32cUltra);

    const fromJsonStr = DnamContainer.build({
      header: container.headerJson,
      arenaBytes: container.arenaBytes,
      backend,
    });
    expect(fromJsonStr.header.metadata.name).toBe(container.header.metadata.name);

    // Checkpoint container
    const ckpt = DnamContainer.buildReference({ core2Active: true, isCheckpoint: true, backend });
    expect(ckpt.preamble.magic).toBe("DNCK");
    expect(ckpt.preamble.core2Active).toBe(true);
  });

  it("updates JSON header in-place within the 4 KB page boundary without touching arenas", () => {
    const container = DnamContainer.buildReference({ core2Active: false, backend });
    const updatedHeader = {
      ...container.header,
      userMetadata: { artist: "Jaco", rating: 5 },
    };
    const updatedContainer = container.updateHeaderInPlace(updatedHeader, backend);
    expect(updatedContainer.header.userMetadata["artist"]).toBe("Jaco");
    expect(updatedContainer.preamble.arenaCrc32cUltra).toBe(container.preamble.arenaCrc32cUltra);

    const rawUpdated = updateHeaderInPlace(container.bytes, updatedHeader, backend);
    expect(DnamContainer.fromBytes(rawUpdated, backend).header.userMetadata["rating"]).toBe(5);
  });

  it("rejects corrupted magic, truncated buffers, and tampered CRC32C checksums", () => {
    expect(() => DnamContainer.fromBytes(new Uint8Array(12), backend)).toThrow(DnamContainerError);

    const badMagic = buildReferenceContainer(false, false, backend).slice();
    badMagic[0] = 0x00;
    expect(() => DnamContainer.fromBytes(badMagic, backend)).toThrow(DnamContainerError);

    const badArenaCrc = buildReferenceContainer(false, false, backend).slice();
    const lastIdx = badArenaCrc.length - 1;
    badArenaCrc[lastIdx] = (badArenaCrc[lastIdx] ?? 0) ^ 0xff;
    expect(() => DnamContainer.fromBytes(badArenaCrc, backend)).toThrow(DnamContainerError);
  });

  it("computes ZOH sample-rate rescaling across 44.1 / 48 / 88.2 / 96 kHz", () => {
    const container = DnamContainer.buildReference({ core2Active: true, backend });
    for (const fsHost of [44_100, 48_000, 88_200, 96_000]) {
      const zoh = container.computeZoh(fsHost, backend);
      expect(zoh.sampleRateHostHz).toBe(fsHost);
      expect(zoh.rhoSr).toBeCloseTo(48_000 / fsHost, 5);
      // Verify LFO tick frequency invariance
      const fLfoTrain = 2.5;
      const dphiTrain = (2 * Math.PI * fLfoTrain * 16) / 48_000;
      const dphiHost = dphiTrain * zoh.rhoCtrl;
      const fLfoHost = (dphiHost * (fsHost / zoh.controlDecimationHost)) / (2 * Math.PI);
      expect(fLfoHost).toBeCloseTo(fLfoTrain, 10);
    }

    expect(() => computeZohRescaledParams({ sampleRateHostHz: 12_000, backend })).toThrow(
      DnamSampleRateError,
    );
  });

  it("packs and unpacks LAPACK uplo='U' Gram matrices and leading principal minors", () => {
    expect(GramMatrix.packedLen(4)).toBe(10);
    expect(() => GramMatrix.packedLen(0)).toThrow(DnamValidationError);

    fc.assert(
      fc.property(
        fc.array(fc.double({ min: -10, max: 10, noNaN: true }), { minLength: 16, maxLength: 16 }),
        (rawVals) => {
          const n = 4;
          const sym = new Float64Array(n * n);
          for (let r = 0; r < n; r++) {
            for (let c = r; c < n; c++) {
              const v = rawVals[r * n + c] ?? 0;
              sym[r * n + c] = v;
              sym[c * n + r] = v;
            }
          }
          const packed = GramMatrix.packUploU(sym, n, backend);
          expect(packed.length).toBe(10);
          const unpackedFull = GramMatrix.unpackUploU(packed, n, backend);
          expect(maxAbsDiff(sym, unpackedFull)).toBeLessThan(1e-12);

          // Leading 2x2 principal minor
          const unpacked2 = GramMatrix.unpackUploU(packed, 2, backend);
          expect(unpacked2[0]).toBeCloseTo(sym[0] ?? 0, 12);
          expect(unpacked2[1]).toBeCloseTo(sym[1] ?? 0, 12);
          expect(unpacked2[2]).toBeCloseTo(sym[4] ?? 0, 12);
          expect(unpacked2[3]).toBeCloseTo(sym[5] ?? 0, 12);
        },
      ),
      { numRuns: 20 },
    );

    expect(() => GramMatrix.packUploU(new Float64Array(3), 2, backend)).toThrow(
      DnamValidationError,
    );
    expect(() => GramMatrix.unpackUploU(new Float64Array(2), 4, backend)).toThrow(
      DnamValidationError,
    );
  });

  it("computes hardware Castagnoli CRC32C checksums identically across backends", () => {
    const payload = new TextEncoder().encode("123456789");
    // Standard Castagnoli CRC32C of "123456789" is 0xE3069283
    expect(crc32c(payload, backend)).toBe(0xe3069283);
  });
});
