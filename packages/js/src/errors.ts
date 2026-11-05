/**
 * Coded `DnamError` class, `ErrorCode` literal union, discriminated `ErrorDetail`, and
 * exhaustive native tag parser (`ref[sdk.ts.errors.coded-error]`).
 *
 * @module
 */

/**
 * Canonical error codes emitted by `@dnam-project/dnam` (`ref[sdk.ts.errors.coded-error]`).
 */
export const errorCodes = [
  "ERR_DNAM_CONTAINER",
  "ERR_DNAM_SAMPLE_RATE",
  "ERR_DNAM_PARAMETER",
  "ERR_DNAM_PROVENANCE",
  "ERR_DNAM_VALIDATION",
  "ERR_DNAM_BUSY",
  "ERR_DNAM_DISPOSED",
] as const;

/**
 * Canonical `DnamError.code` string-literal union.
 */
export type ErrorCode = (typeof errorCodes)[number];

/**
 * Discriminated structured detail attached to a `DnamError`.
 */
export type ErrorDetail =
  | { readonly kind: "container"; readonly rawMessage: string }
  | { readonly kind: "sampleRate"; readonly rawMessage: string }
  | { readonly kind: "parameter"; readonly rawMessage: string; readonly parameterId?: string }
  | { readonly kind: "provenance"; readonly rawMessage: string }
  | { readonly kind: "validation"; readonly rawMessage: string }
  | { readonly kind: "busy"; readonly rawMessage: string }
  | { readonly kind: "disposed"; readonly rawMessage: string };

const DETAIL_KIND_BY_CODE = {
  ERR_DNAM_CONTAINER: "container",
  ERR_DNAM_SAMPLE_RATE: "sampleRate",
  ERR_DNAM_PARAMETER: "parameter",
  ERR_DNAM_PROVENANCE: "provenance",
  ERR_DNAM_VALIDATION: "validation",
  ERR_DNAM_BUSY: "busy",
  ERR_DNAM_DISPOSED: "disposed",
} as const satisfies Record<ErrorCode, ErrorDetail["kind"]>;

/**
 * Options for constructing a `DnamError`.
 */
export interface DnamErrorOptions {
  readonly cause?: unknown;
  readonly parameterId?: string;
}

/**
 * Unified coded error thrown by all `@dnam-project/dnam` operations (`ref[sdk.ts.errors.coded-error]`).
 *
 * @example
 * ```ts
 * try {
 *   engine.parameters.get("unknown");
 * } catch (err) {
 *   if (isDnamError(err, "ERR_DNAM_PARAMETER")) {
 *     console.error(err.code, err.detail);
 *   }
 * }
 * ```
 */
export class DnamError<TCode extends ErrorCode = ErrorCode> extends Error {
  readonly code: TCode;
  readonly detail: ErrorDetail;

  constructor(code: TCode, message: string, options?: DnamErrorOptions) {
    super(message, options?.cause !== undefined ? { cause: options.cause } : undefined);
    this.name = "DnamError";
    this.code = code;
    const kind = DETAIL_KIND_BY_CODE[code];
    const baseDetail =
      kind === "parameter" && options?.parameterId !== undefined
        ? { kind, rawMessage: message, parameterId: options.parameterId }
        : { kind, rawMessage: message };
    this.detail = Object.freeze(baseDetail) as ErrorDetail;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * Type guard narrowing `err` to `DnamError` (and to a specific `code` when provided).
 *
 * @example
 * ```ts
 * if (isDnamError(err, "ERR_DNAM_CONTAINER")) {
 *   console.log(err.detail.rawMessage);
 * }
 * ```
 */
export function isDnamError<TCode extends ErrorCode = ErrorCode>(
  err: unknown,
  code?: TCode,
): err is DnamError<TCode> {
  if (!(err instanceof DnamError)) {
    return false;
  }
  if (code !== undefined && err.code !== code) {
    return false;
  }
  return true;
}

const NATIVE_TAG_TO_CODE: Readonly<Record<string, ErrorCode>> = {
  "[DNAM-CONTAINER]": "ERR_DNAM_CONTAINER",
  "[DNAM-SAMPLERATE]": "ERR_DNAM_SAMPLE_RATE",
  "[DNAM-CONTROL]": "ERR_DNAM_PARAMETER",
  "[DNAM-PROVENANCE]": "ERR_DNAM_PROVENANCE",
  "[DNAM-VALIDATION]": "ERR_DNAM_VALIDATION",
  "[DNAM-BUSY]": "ERR_DNAM_BUSY",
  "[DNAM-DISPOSED]": "ERR_DNAM_DISPOSED",
};

/**
 * Maps a tagged native N-API or WASM error into a typed `DnamError`.
 */
export function fromNativeError(err: unknown): DnamError {
  if (err instanceof DnamError) {
    return err;
  }
  const raw = err instanceof Error ? err.message : String(err);
  for (const [prefix, code] of Object.entries(NATIVE_TAG_TO_CODE)) {
    if (raw.startsWith(prefix)) {
      const cleanMessage = raw.slice(prefix.length).trim();
      return new DnamError(code, cleanMessage, { cause: err });
    }
  }
  return new DnamError("ERR_DNAM_VALIDATION", raw, { cause: err });
}
