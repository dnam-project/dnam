/**
 * Tests for `@dnam-project/dnam/schema` camelCase codec, RFC 8785 JCS round-trip parity against
 * Rust `DnamJsonHeader`, and exhaustive `DnamError` variant mapping.
 */

import "../node.js";

import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { DnamError, type ErrorCode, fromNativeError, isDnamError } from "../errors.js";
import { withDriver } from "../internal/native/load.js";
import {
  getTestingBindings,
  loadNodeDriverSync,
  loadNodeWasmDriverSync,
} from "../internal/native/napi.js";
import { immutableHeaderSha256 } from "../provenance/verify.js";
import { camelToWireObject, wireToCamelObject } from "./case-codec.js";
import {
  canonicalJson,
  DEFAULT_ALLPASS_DELAYS,
  decodeHeader,
  encodeHeader,
  headerToJcs,
  parseSchemaVersion,
} from "./index.js";

describe("schema & error parity", () => {
  it.each(["default", "full"] as const)(
    "round-trips Rust %s DnamJsonHeader through camelCase ModelHeader to byte-identical JCS",
    (kind) => {
      // verify[sdk.ts.schema.camel-case-codec-parity]
      // verify[sdk.ts.api.resources-and-data]
      const testing = getTestingBindings();
      const rustJson = testing.testingReferenceHeaderJson(kind);
      const header = decodeHeader(rustJson);

      expect(Object.isFrozen(header)).toBe(true);
      expect(Object.isFrozen(header.metadata)).toBe(true);
      expect(Object.isFrozen(header.ioConfig)).toBe(true);

      const reencodedJson = encodeHeader(header);
      const rustJcs = canonicalJson(rustJson);
      const tsJcs = headerToJcs(header);
      expect(tsJcs).toBe(rustJcs);

      const rustImmutableJcs = canonicalJson(rustJson, { projectImmutable: true });
      const tsImmutableJcs = headerToJcs(header, { projectImmutable: true });
      expect(tsImmutableJcs).toBe(rustImmutableJcs);

      expect(immutableHeaderSha256(header)).toBe(immutableHeaderSha256(rustJson));
      expect(decodeHeader(reencodedJson)).toEqual(header);

      // Also verify JCS canonicalization via WASM driver matches
      const wasmDriver = loadNodeWasmDriverSync();
      withDriver(wasmDriver, () => {
        expect(wasmDriver.canonicalizeJcs(reencodedJson, false, false)).toBe(rustJcs);
        expect(wasmDriver.canonicalizeJcs(reencodedJson, true, false)).toBe(rustImmutableJcs);
        expect(immutableHeaderSha256(header)).toBe(immutableHeaderSha256(rustJson));
      });
    },
  );

  it("maps every Rust DnamError variant exhaustively to its exact ErrorCode", () => {
    // verify[sdk.ts.errors.coded-error]
    const testing = getTestingBindings();
    const expectedCodes: Readonly<Record<string, ErrorCode>> = {
      BufferTooSmall: "ERR_DNAM_CONTAINER",
      InvalidMagic: "ERR_DNAM_CONTAINER",
      UnsupportedVersion: "ERR_DNAM_CONTAINER",
      HeaderCapacityExceeded: "ERR_DNAM_CONTAINER",
      Crc32cMismatch: "ERR_DNAM_CONTAINER",
      InvalidSampleRate: "ERR_DNAM_SAMPLE_RATE",
      InvalidControlIndex: "ERR_DNAM_PARAMETER",
      CryptoVerificationFailed: "ERR_DNAM_PROVENANCE",
      EncryptedArenaRequiresDeviceKey: "ERR_DNAM_PROVENANCE",
      PreviewContainerRejected: "ERR_DNAM_PROVENANCE",
      UnalignedOffset: "ERR_DNAM_VALIDATION",
      InvalidJsonHeader: "ERR_DNAM_VALIDATION",
      SchemaValidationError: "ERR_DNAM_VALIDATION",
    };

    const variantNames = testing.testingErrorVariantNames();
    expect(new Set(variantNames)).toEqual(new Set(Object.keys(expectedCodes)));

    for (const name of variantNames) {
      const expectedCode = expectedCodes[name];
      let caught: unknown;
      try {
        testing.testingRaiseErrorVariant(name);
      } catch (err) {
        caught = fromNativeError(err);
      }
      expect(isDnamError(caught, expectedCode)).toBe(true);
      if (isDnamError(caught)) {
        expect(caught.message.startsWith("[DNAM-")).toBe(false);
        expect(Object.isFrozen(caught.detail)).toBe(true);
      }
    }

    expect(isDnamError(new Error("plain"))).toBe(false);
    const paramErr = new DnamError("ERR_DNAM_PARAMETER", "bad knob", { parameterId: "gain" });
    expect(isDnamError(paramErr, "ERR_DNAM_PARAMETER")).toBe(true);
    expect(isDnamError(paramErr, "ERR_DNAM_CONTAINER")).toBe(false);
    expect(fromNativeError(paramErr)).toBe(paramErr);
  });

  it("validates SchemaVersion and canonical JCS via native Rust serde on both N-API and WASM drivers", () => {
    // verify[sdk.ts.schema.camel-case-codec-parity]
    // verify[sdk.ts.api.literal-unions-and-options]
    for (const driver of [loadNodeDriverSync(), loadNodeWasmDriverSync()]) {
      withDriver(driver, () => {
        const v20 = parseSchemaVersion("2.0.0");
        expect(v20.semver).toBe("2.0.0");
        expect(v20.isSupported).toBe(true);
        expect(parseSchemaVersion("2.0").semver).toBe("2.0.0");
        expect(parseSchemaVersion("1.1.0").isSupported).toBe(false);
        expect(parseSchemaVersion("3.0.0").isSupported).toBe(false);
        expect(() => parseSchemaVersion("bad")).toThrow(DnamError);
        expect(() => parseSchemaVersion("1.x.0")).toThrow(DnamError);

        expect(() => decodeHeader("{bad json")).toThrow(DnamError);
        expect(() => decodeHeader({ unknownField: 123 })).toThrow(DnamError);

        // Canonical JCS edge cases & DEFAULT_ALLPASS_DELAYS
        expect(canonicalJson({ b: -0, a: [true, false, null] })).toBe(
          '{"a":[true,false,null],"b":0}',
        );
        expect(() => canonicalJson(Number.NaN)).toThrow(DnamError);
        expect(DEFAULT_ALLPASS_DELAYS).toHaveLength(16);
      });
    }
  });

  it("property: fast-check verifies camelCase/snakeCase codec, SchemaVersion, and JCS header roundtrips", () => {
    // verify[sdk.ts.schema.camel-case-codec-parity]
    const baseHeader = decodeHeader(getTestingBindings().testingReferenceHeaderJson("default"));

    fc.assert(
      fc.property(
        fc.string({ minLength: 1, maxLength: 32 }),
        fc.string({ minLength: 1, maxLength: 24 }),
        fc.constantFrom(44_100, 48_000, 88_200, 96_000),
        (name, author, sampleRateTrainHz) => {
          const modified: typeof baseHeader = {
            ...baseHeader,
            sampleRateTrainHz,
            metadata: {
              ...baseHeader.metadata,
              name,
              author,
            },
          };
          const wireJson = encodeHeader(modified);
          const roundtripped = decodeHeader(wireJson);
          expect(roundtripped).toEqual(modified);
          expect(headerToJcs(roundtripped)).toBe(canonicalJson(wireJson));
          expect(immutableHeaderSha256(roundtripped)).toBe(immutableHeaderSha256(wireJson));
        },
      ),
      { numRuns: 30 },
    );

    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: 5 }),
        fc.integer({ min: 0, max: 5 }),
        fc.integer({ min: 0, max: 20 }),
        (major, minor, patch) => {
          const v = parseSchemaVersion(`${String(major)}.${String(minor)}.${String(patch)}`);
          expect(v.major).toBe(major);
          expect(v.minor).toBe(minor);
          expect(v.patch).toBe(patch);
          expect(v.isSupported).toBe(major === 2 && minor === 0);
        },
      ),
      { numRuns: 40 },
    );

    fc.assert(
      fc.property(
        fc.dictionary(
          fc.constantFrom("gain_db", "delay_ms", "user_key"),
          fc.double({ min: -100, max: 100, noNaN: true }),
        ),
        (controlsDict) => {
          const raw = {
            sample_rate_train_hz: 48_000,
            presets: [{ preset_id: "p1", name: "P1", controls: controlsDict }],
          };
          const camel = wireToCamelObject(raw);
          const backToSnake = camelToWireObject(camel);
          expect(backToSnake).toEqual(raw);
        },
      ),
      { numRuns: 30 },
    );
  });
});
