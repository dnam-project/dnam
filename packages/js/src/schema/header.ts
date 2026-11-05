/**
 * Top-level `.dnam` / `.dnam-ckpt` `ModelHeader` interface, `SchemaVersion`, and bijective
 * `decodeHeader` / `encodeHeader` functions backed by native Rust `serde` validation
 * (`ref[sdk.ts.schema.camel-case-codec-parity]`).
 *
 * Zero runtime JS dependencies (`0 KB` validator bundle).
 *
 * @module
 */

import { DnamError } from "../errors.js";
import { deepFreeze } from "../internal/freeze.js";
import { getActiveDriver } from "../internal/native/load.js";
import { canonicalJson } from "./canonical-json.js";
import { camelToWireObject, pruneRustSkipSerializingIf, wireToCamelObject } from "./case-codec.js";
import type { AuditReport, StemRegistryEntry } from "./checkpoint.js";
import type { IoConfig, SlimmableTiersConfig } from "./io.js";
import type { ParameterDescriptor, Preset } from "./parameters.js";
import type { ModelMetadata } from "./signatures.js";
import type {
  ControlBusConfig,
  RouterConfig,
  Stage1PreFxConfig,
  Stage2CascadeConfig,
  Stage3CabinetKautzConfig,
  Stage4SpatialTankConfig,
  TensorSliceEntry,
} from "./stages.js";

/**
 * Parsed semantic version (`major.minor.patch`) of the `.dnam` / `.dnam-ckpt` container schema.
 */
export interface SchemaVersion {
  readonly major: number;
  readonly minor: number;
  readonly patch: number;
  readonly semver: string;
  readonly isSupported: boolean;
}

/**
 * Parses a semantic version string (`"major.minor.patch"` or `"major.minor"`) by delegating
 * directly to `dnam_engine::schema::SchemaVersion::parse` in Rust.
 */
export function parseSchemaVersion(versionStr: string): SchemaVersion {
  const [major, minor, patch, isSupported] = getActiveDriver().parseSchemaVersion(versionStr);
  return Object.freeze({
    major,
    minor,
    patch,
    semver: `${String(major)}.${String(minor)}.${String(patch)}`,
    isSupported,
  });
}

/**
 * Validated, deeply frozen camelCase `.dnam` / `.dnam-ckpt` JSON header (`ModelHeader`).
 */
export interface ModelHeader {
  readonly version: string;
  readonly containerKind: "dnam" | "dnam-ckpt" | "preview" | string;
  readonly architecture: string;
  readonly sampleRateTrainHz: number;
  readonly metadata: ModelMetadata;
  readonly userMetadata: Readonly<Record<string, unknown>>;
  readonly ioConfig: IoConfig;
  readonly slimmableTiers: SlimmableTiersConfig;
  readonly controlsSchema: readonly ParameterDescriptor[];
  readonly presets: readonly Preset[];
  readonly controlBus: ControlBusConfig;
  readonly stage0Router: RouterConfig;
  readonly stage1Pre: Stage1PreFxConfig;
  readonly stage2Cascade: Stage2CascadeConfig;
  readonly stage3Cabinet: Stage3CabinetKautzConfig;
  readonly stage4Tank: Stage4SpatialTankConfig;
  readonly tensorDirectory: Readonly<Record<string, TensorSliceEntry>>;
  readonly stemsRegistry: readonly StemRegistryEntry[] | null;
  readonly auditReport: AuditReport | null;
}

function normalizeOptionalNullableFields(camel: Record<string, unknown>): void {
  if (!("stemsRegistry" in camel)) {
    camel["stemsRegistry"] = null;
  } else if (Array.isArray(camel["stemsRegistry"])) {
    for (const stem of camel["stemsRegistry"]) {
      if (stem !== null && typeof stem === "object") {
        const sRec = stem as Record<string, unknown>;
        if (!("extractedPhysics" in sRec)) {
          sRec["extractedPhysics"] = null;
        }
        for (const sk of ["gAudioUploUF64", "cAudioTierF64"]) {
          const sl = sRec[sk];
          if (sl !== null && typeof sl === "object") {
            const slRec = sl as Record<string, unknown>;
            for (const ok of ["packing", "featureDim", "packedLen", "shape"]) {
              if (!(ok in slRec)) {
                slRec[ok] = null;
              }
            }
          }
        }
      }
    }
  }
  if (!("auditReport" in camel)) {
    camel["auditReport"] = null;
  }
  const meta = camel["metadata"];
  if (meta !== null && typeof meta === "object") {
    const mRec = meta as Record<string, unknown>;
    if (!("distributionPolicy" in mRec)) {
      mRec["distributionPolicy"] = null;
    }
    const sigs = mRec["signatures"];
    if (sigs !== null && typeof sigs === "object") {
      const sigsRec = sigs as Record<string, unknown>;
      if (!("tierArenaSha256" in sigsRec)) {
        sigsRec["tierArenaSha256"] = null;
      }
      const rcpt = sigsRec["marketplaceReceipt"];
      if (rcpt !== null && typeof rcpt === "object") {
        const rRec = rcpt as Record<string, unknown>;
        if (!("buyerTierArenaSha256" in rRec)) {
          rRec["buyerTierArenaSha256"] = null;
        }
      }
    }
  }
  const ctrls = camel["controlsSchema"];
  if (Array.isArray(ctrls)) {
    for (const ctrl of ctrls) {
      if (ctrl !== null && typeof ctrl === "object") {
        const cRec = ctrl as Record<string, unknown>;
        if (!("calibration" in cRec) || cRec["calibration"] === null) {
          cRec["calibration"] = null;
        } else if (typeof cRec["calibration"] === "object") {
          const calRec = cRec["calibration"] as Record<string, unknown>;
          if (!("lobattoPhysicalAnchors" in calRec)) {
            calRec["lobattoPhysicalAnchors"] = null;
          }
          if (!("conditionalPhysicalAnchors" in calRec)) {
            calRec["conditionalPhysicalAnchors"] = {};
          }
          if (!("tempoSyncDefaultSubdivision" in calRec)) {
            calRec["tempoSyncDefaultSubdivision"] = null;
          }
        }
      }
    }
  }
  const tdir = camel["tensorDirectory"];
  if (tdir !== null && typeof tdir === "object") {
    for (const sl of Object.values(tdir as Record<string, unknown>)) {
      if (sl !== null && typeof sl === "object") {
        const slRec = sl as Record<string, unknown>;
        for (const ok of ["packing", "featureDim", "packedLen", "shape"]) {
          if (!(ok in slRec)) {
            slRec[ok] = null;
          }
        }
      }
    }
  }
}

/**
 * Decodes a snake_case `.dnam` wire JSON string (or camelCase/snake_case object) into a
 * native-`serde`-validated, deeply frozen camelCase `ModelHeader`
 * (`ref[sdk.ts.schema.camel-case-codec-parity]`).
 *
 * @example
 * ```ts
 * const header = decodeHeader(wireJsonString);
 * console.log(header.metadata.name, header.ioConfig.outputMode);
 * ```
 */
export function decodeHeader(wireJsonOrObject: string | Record<string, unknown>): ModelHeader {
  let wireJson: string;
  if (typeof wireJsonOrObject === "string") {
    wireJson = wireJsonOrObject;
  } else if (wireJsonOrObject !== null && typeof wireJsonOrObject === "object") {
    const wireObj = camelToWireObject(wireJsonOrObject) as Record<string, unknown>;
    pruneRustSkipSerializingIf(wireObj);
    wireJson = JSON.stringify(wireObj);
  } else {
    throw new DnamError("ERR_DNAM_VALIDATION", "Expected JSON string or object for decodeHeader");
  }

  const normalizedWireJson = getActiveDriver().validateAndNormalizeHeader(wireJson);
  const parsedWire = JSON.parse(normalizedWireJson) as Record<string, unknown>;
  const camelObj = wireToCamelObject(parsedWire) as Record<string, unknown>;
  normalizeOptionalNullableFields(camelObj);
  return deepFreeze(camelObj as unknown as ModelHeader);
}

/**
 * Validates and encodes a camelCase `ModelHeader` into its canonical snake_case `.dnam` wire
 * JSON string via native Rust `serde` (`ref[sdk.ts.schema.camel-case-codec-parity]`).
 */
export function encodeHeader(header: ModelHeader): string {
  const wireObj = camelToWireObject(header) as Record<string, unknown>;
  pruneRustSkipSerializingIf(wireObj);
  return getActiveDriver().validateAndNormalizeHeader(JSON.stringify(wireObj));
}

/**
 * Serializes a `ModelHeader` (or wire JSON string) to canonical RFC 8785 JCS UTF-8 text
 * (`ref[sec.jcs.immutable-header-digest]`).
 */
export function headerToJcs(
  headerOrWireJson: ModelHeader | string,
  options?: { readonly projectImmutable?: boolean },
): string {
  const wireJson =
    typeof headerOrWireJson === "string" ? headerOrWireJson : encodeHeader(headerOrWireJson);
  return canonicalJson(wireJson, options);
}
