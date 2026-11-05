/**
 * RFC 8785 JSON Canonicalization Scheme (JCS) serializer delegating to native Rust
 * `dnam_engine::container::serialize_jcs_value` and `project_immutable_header_value`
 * (`ref[sec.jcs.immutable-header-digest]`, `ref[sdk.ts.schema.camel-case-codec-parity]`).
 *
 * @module
 */

import { DnamError } from "../errors.js";
import { getActiveDriver } from "../internal/native/load.js";

/**
 * Options for `canonicalJson(...)`.
 */
export interface CanonicalJsonOptions {
  /**
   * When `true`, projects out mutable `.dnam` header fields (`presets`, `user_metadata`,
   * `slimmable_tiers.*.prefix_crc32c`, `controls_schema[*].user_metadata`,
   * `stems_registry[*].user_metadata`, `metadata.signatures`) in Rust per
   * `ref[sec.jcs.immutable-header-digest]`.
   */
  readonly projectImmutable?: boolean;
  /**
   * When `true`, rejects floating-point numbers outside the integer JCS subset.
   */
  readonly rejectFloats?: boolean;
}

/**
 * Serializes a JSON string or plain value into canonical RFC 8785 JCS UTF-8 text by delegating
 * directly to `dnam_engine::container::serialize_jcs_value` in Rust
 * (`ref[sec.jcs.immutable-header-digest]`).
 *
 * @example
 * ```ts
 * const jcs = canonicalJson({ b: 1, a: [true, null] });
 * // => '{"a":[true,null],"b":1}'
 * ```
 */
export function canonicalJson(valueOrJson: unknown, options?: CanonicalJsonOptions): string {
  let jsonStr: string;
  if (typeof valueOrJson === "string") {
    jsonStr = valueOrJson;
  } else {
    if (typeof valueOrJson === "number" && !Number.isFinite(valueOrJson)) {
      throw new DnamError("ERR_DNAM_VALIDATION", "RFC 8785 JCS forbids NaN and Infinity");
    }
    jsonStr = JSON.stringify(valueOrJson);
  }
  return getActiveDriver().canonicalizeJcs(
    jsonStr,
    options?.projectImmutable ?? false,
    options?.rejectFloats ?? false,
  );
}
