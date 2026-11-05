/**
 * Compile-time model parameter type inference (`ParameterValueMap`, `defineModelTypes`)
 * (`ref[sdk.ts.api.typed-model-inference]`).
 *
 * @module
 */

/**
 * Mapping of parameter IDs to their TypeScript runtime types (`number` for `"range"`,
 * string-literal union for `"switch"`).
 */
export type ParameterValueMap = Readonly<Record<string, number | string>>;

/**
 * Compile-time descriptor entry passed to `defineModelTypes(...)`.
 */
export type TypedParameterEntry =
  | { readonly id: string; readonly type: "range" }
  | {
      readonly id: string;
      readonly type: "switch";
      readonly declaredOptions: readonly string[];
    };

/**
 * Infers a `ParameterValueMap` from a `readonly TypedParameterEntry[]` tuple.
 */
export type InferParameterMap<TEntries extends readonly TypedParameterEntry[]> = {
  readonly [E in TEntries[number] as E["id"]]: E extends {
    readonly type: "switch";
    readonly declaredOptions: readonly (infer TOpt extends string)[];
  }
    ? TOpt
    : number;
};

/**
 * Declares compile-time parameter types from a literal parameter descriptor tuple
 * (`ref[sdk.ts.api.typed-model-inference]`).
 *
 * @example
 * ```ts
 * const MyAmpParams = defineModelTypes([
 *   { id: "gain", type: "range" },
 *   { id: "boost", type: "switch", declaredOptions: ["off", "on"] },
 * ] as const);
 * type MyAmpMap = typeof MyAmpParams.$infer;
 * ```
 */
export function defineModelTypes<const TEntries extends readonly TypedParameterEntry[]>(
  entries: TEntries,
): {
  readonly entries: TEntries;
  readonly $infer: InferParameterMap<TEntries>;
} {
  return Object.freeze({
    entries,
    $infer: {} as InferParameterMap<TEntries>,
  });
}
