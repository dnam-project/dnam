/**
 * Bijective snake_case wire JSON <-> camelCase TypeScript codec (`ref[sdk.ts.schema.camel-case-codec-parity]`).
 *
 * Preserves user-supplied dictionary keys (`userMetadata`, `controls`, `controlsMeta`,
 * `tensorDirectory`, `conditionalMin`, `conditionalMax`, `conditionalPhysicalAnchors`,
 * `scaleLabels`, `optionLabels`) verbatim while translating all schema structural keys
 * bijectively and enforcing Rust `serde` `skip_serializing_if` rules.
 *
 * @module
 */

const SPECIAL_SNAKE_TO_CAMEL: Readonly<Record<string, string>> = {
  arena_nonce_96: "arenaNonce96",
  g_audio_uplo_u_f64: "gAudioUploUF64",
};

const SPECIAL_CAMEL_TO_SNAKE: Readonly<Record<string, string>> = {
  arenaNonce96: "arena_nonce_96",
  gAudioUploUF64: "g_audio_uplo_u_f64",
};

/**
 * Schema fields whose values are open `Record<string, V>` maps where keys are user/model data
 * rather than schema field names (and therefore must never have their keys case-converted).
 */
const USER_KEY_MAP_FIELDS = new Set<string>([
  "user_metadata",
  "userMetadata",
  "controls",
  "controls_meta",
  "controlsMeta",
  "conditional_min",
  "conditionalMin",
  "conditional_max",
  "conditionalMax",
  "conditional_physical_anchors",
  "conditionalPhysicalAnchors",
  "scale_labels",
  "scaleLabels",
  "option_labels",
  "optionLabels",
]);

/**
 * Schema fields whose keys are user/model identifiers (`tensor_directory`), while each value
 * is a nested schema struct (`TensorSliceEntry`) whose own keys ARE case-converted.
 */
const STRUCT_VALUE_MAP_FIELDS = new Set<string>(["tensor_directory", "tensorDirectory"]);

/**
 * Converts a snake_case schema key into its canonical camelCase TypeScript property name.
 */
export function snakeToCamelKey(key: string): string {
  const special = SPECIAL_SNAKE_TO_CAMEL[key];
  if (special !== undefined) {
    return special;
  }
  return key.replace(/_([a-z])/g, (_, ch: string) => ch.toUpperCase());
}

/**
 * Converts a camelCase TypeScript property name into its canonical snake_case wire JSON key.
 */
export function camelToSnakeKey(key: string): string {
  const special = SPECIAL_CAMEL_TO_SNAKE[key];
  if (special !== undefined) {
    return special;
  }
  return key.replace(/[A-Z]/g, (ch) => `_${ch.toLowerCase()}`);
}

/**
 * Recursively transforms a parsed wire JSON value from snake_case schema keys to camelCase.
 */
export function wireToCamelObject(value: unknown, parentField?: string): unknown {
  if (value === null || typeof value !== "object") {
    return value;
  }
  if (Array.isArray(value)) {
    return value.map((item) => wireToCamelObject(item));
  }
  const record = value as Record<string, unknown>;
  if (parentField !== undefined && USER_KEY_MAP_FIELDS.has(parentField)) {
    const preserved: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(record)) {
      preserved[k] = v;
    }
    return preserved;
  }
  if (parentField !== undefined && STRUCT_VALUE_MAP_FIELDS.has(parentField)) {
    const mapped: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(record)) {
      mapped[k] = wireToCamelObject(v);
    }
    return mapped;
  }
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(record)) {
    const camelKey = snakeToCamelKey(k);
    out[camelKey] = wireToCamelObject(v, camelKey);
  }
  return out;
}

/**
 * Recursively transforms a camelCase TypeScript schema object into snake_case wire JSON keys.
 */
export function camelToWireObject(value: unknown, parentField?: string): unknown {
  if (value === null || typeof value !== "object") {
    return value;
  }
  if (Array.isArray(value)) {
    return value.map((item) => camelToWireObject(item));
  }
  const record = value as Record<string, unknown>;
  if (parentField !== undefined && USER_KEY_MAP_FIELDS.has(parentField)) {
    const preserved: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(record)) {
      preserved[k] = v;
    }
    return preserved;
  }
  if (parentField !== undefined && STRUCT_VALUE_MAP_FIELDS.has(parentField)) {
    const mapped: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(record)) {
      mapped[k] = camelToWireObject(v);
    }
    return mapped;
  }
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(record)) {
    const snakeKey = camelToSnakeKey(k);
    out[snakeKey] = camelToWireObject(v, snakeKey);
  }
  return out;
}

function pruneTensorSliceEntry(sl: Record<string, unknown>): void {
  for (const optKey of ["packing", "feature_dim", "packed_len", "shape"]) {
    if (sl[optKey] === null || sl[optKey] === undefined) {
      sl[optKey] = undefined;
    }
  }
}

/**
 * Applies the exact Rust `serde` `skip_serializing_if` rules to a snake_case wire header object
 * so `encodeHeader` produces byte-identical JSON to `serde_json::to_string(&DnamJsonHeader)`.
 */
export function pruneRustSkipSerializingIf(raw: Record<string, unknown>): Record<string, unknown> {
  if (raw["stems_registry"] === null || raw["stems_registry"] === undefined) {
    raw["stems_registry"] = undefined;
  } else if (Array.isArray(raw["stems_registry"])) {
    for (const stem of raw["stems_registry"]) {
      if (stem !== null && typeof stem === "object") {
        const stemRec = stem as Record<string, unknown>;
        if (stemRec["extracted_physics"] === null || stemRec["extracted_physics"] === undefined) {
          stemRec["extracted_physics"] = undefined;
        }
        for (const sliceKey of ["g_audio_uplo_u_f64", "c_audio_tier_f64"]) {
          const sl = stemRec[sliceKey];
          if (sl !== null && typeof sl === "object") {
            pruneTensorSliceEntry(sl as Record<string, unknown>);
          }
        }
      }
    }
  }

  if (raw["audit_report"] === null || raw["audit_report"] === undefined) {
    raw["audit_report"] = undefined;
  }

  const meta = raw["metadata"];
  if (meta !== null && typeof meta === "object") {
    const metaRec = meta as Record<string, unknown>;
    if (metaRec["distribution_policy"] === null || metaRec["distribution_policy"] === undefined) {
      metaRec["distribution_policy"] = undefined;
    }
    const sigs = metaRec["signatures"];
    if (sigs !== null && typeof sigs === "object") {
      const sigsRec = sigs as Record<string, unknown>;
      if (sigsRec["tier_arena_sha256"] === null || sigsRec["tier_arena_sha256"] === undefined) {
        sigsRec["tier_arena_sha256"] = undefined;
      }
      const rcpt = sigsRec["marketplace_receipt"];
      if (rcpt !== null && typeof rcpt === "object") {
        const rcptRec = rcpt as Record<string, unknown>;
        if (
          rcptRec["buyer_tier_arena_sha256"] === null ||
          rcptRec["buyer_tier_arena_sha256"] === undefined
        ) {
          rcptRec["buyer_tier_arena_sha256"] = undefined;
        }
      }
    }
  }

  const ctrls = raw["controls_schema"];
  if (Array.isArray(ctrls)) {
    for (const ctrl of ctrls) {
      if (ctrl !== null && typeof ctrl === "object") {
        const ctrlRec = ctrl as Record<string, unknown>;
        const cal = ctrlRec["calibration"];
        if (cal === null || cal === undefined) {
          ctrlRec["calibration"] = undefined;
        } else if (typeof cal === "object") {
          const calRec = cal as Record<string, unknown>;
          if (
            calRec["lobatto_physical_anchors"] === null ||
            calRec["lobatto_physical_anchors"] === undefined
          ) {
            calRec["lobatto_physical_anchors"] = undefined;
          }
          const condAnchors = calRec["conditional_physical_anchors"];
          if (
            condAnchors === null ||
            condAnchors === undefined ||
            (typeof condAnchors === "object" &&
              Object.keys(condAnchors as Record<string, unknown>).length === 0)
          ) {
            calRec["conditional_physical_anchors"] = undefined;
          }
          if (
            calRec["tempo_sync_default_subdivision"] === null ||
            calRec["tempo_sync_default_subdivision"] === undefined
          ) {
            calRec["tempo_sync_default_subdivision"] = undefined;
          }
        }
      }
    }
  }

  const tdir = raw["tensor_directory"];
  if (tdir !== null && typeof tdir === "object") {
    for (const sl of Object.values(tdir as Record<string, unknown>)) {
      if (sl !== null && typeof sl === "object") {
        pruneTensorSliceEntry(sl as Record<string, unknown>);
      }
    }
  }

  return raw;
}
