/**
 * Base Zod v4 & Standard Schema v1 (`@standard-schema/spec`) primitives, tier parsing,
 * and `.dnam` wire `snake_case` <-> idiomatic TypeScript `camelCase` helpers.
 */

import type { StandardSchemaV1 } from "@standard-schema/spec";
import { z } from "zod";
import { DnamValidationError } from "../errors.js";

export type TierName = "lite" | "full" | "pro" | "ultra";
export type TierCode = 0 | 1 | 2 | 3;
export type TierInput = TierName | "Lite" | "Full" | "Pro" | "Ultra" | TierCode | number;

export const TierSchema = z.union([
  z.enum(["lite", "full", "pro", "ultra", "Lite", "Full", "Pro", "Ultra"]),
  z.literal(0),
  z.literal(1),
  z.literal(2),
  z.literal(3),
]);

/**
 * Normalizes a tier name (`"lite"`, `"full"`, `"pro"`, `"ultra"`) or code (`0`, `1`, `2`, `3`) into `0 | 1 | 2 | 3`.
 */
export function parseTier(tier: TierInput): TierCode {
  if (typeof tier === "number") {
    if (tier === 0 || tier === 1 || tier === 2 || tier === 3) {
      return tier;
    }
    throw new DnamValidationError(`Invalid tier number: ${String(tier)} (expected 0, 1, 2, or 3)`);
  }
  const lower = tier.toLowerCase();
  if (lower === "lite") {
    return 0;
  }
  if (lower === "full") {
    return 1;
  }
  if (lower === "pro") {
    return 2;
  }
  if (lower === "ultra") {
    return 3;
  }
  throw new DnamValidationError(
    `Invalid tier name: '${tier}' (expected 'lite', 'full', 'pro', or 'ultra')`,
  );
}

/**
 * Converts a numeric tier code (`0`, `1`, `2`, `3`) into its canonical lowercase name (`"lite"`, `"full"`, `"pro"`, `"ultra"`).
 */
export function tierToName(tier: number): TierName {
  if (tier === 0) {
    return "lite";
  }
  if (tier === 1) {
    return "full";
  }
  if (tier === 2) {
    return "pro";
  }
  if (tier === 3) {
    return "ultra";
  }
  throw new DnamValidationError(`Invalid tier code: ${String(tier)}`);
}

export type ControlValue = number | string | boolean;
export type ControlMap = Record<string, ControlValue>;

export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

export const JsonValueSchema: z.ZodType<JsonValue> = z.lazy(() =>
  z.union([
    z.string(),
    z.number(),
    z.boolean(),
    z.null(),
    z.array(JsonValueSchema),
    z.record(z.string(), JsonValueSchema),
  ]),
);

/**
 * Validates any value synchronously against a `@standard-schema/spec` `StandardSchemaV1`
 * (including all `zod` v4 schemas via `schema["~standard"]`), throwing `DnamValidationError` on failure.
 */
export function validateStandardSchema<TInput, TOutput = TInput>(
  schema: StandardSchemaV1<TInput, TOutput>,
  input: unknown,
  label = "Schema validation",
): TOutput {
  const result = schema["~standard"].validate(input);
  if (result instanceof Promise) {
    throw new DnamValidationError(`${label} returned an unexpected async validation Promise`);
  }
  if (result.issues) {
    const details = result.issues
      .map((issue) => {
        const pathStr = issue.path
          ? issue.path.map((p) => (typeof p === "object" ? String(p.key) : String(p))).join(".")
          : "";
        return pathStr ? `${pathStr}: ${issue.message}` : issue.message;
      })
      .join("; ");
    throw new DnamValidationError(`${label} failed: ${details}`);
  }
  return result.value;
}
