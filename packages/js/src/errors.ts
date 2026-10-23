/**
 * Structured error hierarchy for `@dnam-project/engine` mirroring `dnam-engine` Rust errors
 * and Python `DnamError` subclasses.
 */

export class DnamError extends Error {
  public readonly code: string;

  public constructor(message: string, code = "DNAM_ERROR", options?: ErrorOptions) {
    super(message, options);
    this.name = "DnamError";
    this.code = code;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

export class DnamContainerError extends DnamError {
  public constructor(message: string, options?: ErrorOptions) {
    super(message, "DNAM_CONTAINER_ERROR", options);
    this.name = "DnamContainerError";
  }
}

export class DnamValidationError extends DnamError {
  public constructor(message: string, options?: ErrorOptions) {
    super(message, "DNAM_VALIDATION_ERROR", options);
    this.name = "DnamValidationError";
  }
}

export class DnamControlError extends DnamError {
  public constructor(message: string, options?: ErrorOptions) {
    super(message, "DNAM_CONTROL_ERROR", options);
    this.name = "DnamControlError";
  }
}

export class DnamSampleRateError extends DnamError {
  public constructor(message: string, options?: ErrorOptions) {
    super(message, "DNAM_SAMPLE_RATE_ERROR", options);
    this.name = "DnamSampleRateError";
  }
}

/**
 * Maps a raw Rust / N-API / WASM error message (prefixed with `[DNAM-*]`) to the exact
 * `DnamError` subclass.
 */
export function mapTaggedError(rawMessage: string, cause?: unknown): DnamError {
  const options: ErrorOptions | undefined = cause !== undefined ? { cause } : undefined;
  if (rawMessage.startsWith("[DNAM-CONTAINER]")) {
    const clean = rawMessage.slice("[DNAM-CONTAINER]".length).trim();
    return new DnamContainerError(clean, options);
  }
  if (rawMessage.startsWith("[DNAM-SAMPLERATE]")) {
    const clean = rawMessage.slice("[DNAM-SAMPLERATE]".length).trim();
    return new DnamSampleRateError(clean, options);
  }
  if (rawMessage.startsWith("[DNAM-CONTROL]")) {
    const clean = rawMessage.slice("[DNAM-CONTROL]".length).trim();
    return new DnamControlError(clean, options);
  }
  if (rawMessage.startsWith("[DNAM-VALIDATION]")) {
    const clean = rawMessage.slice("[DNAM-VALIDATION]".length).trim();
    return new DnamValidationError(clean, options);
  }
  return new DnamError(rawMessage, "DNAM_ERROR", options);
}

/**
 * Rethrows any error caught from the native N-API addon or WASM linear-memory bridge
 * as a typed `DnamError` subclass.
 */
export function rethrowDnamError(err: unknown): never {
  if (err instanceof DnamError) {
    throw err;
  }
  if (err instanceof Error) {
    throw mapTaggedError(err.message, err);
  }
  throw mapTaggedError(String(err), err);
}
