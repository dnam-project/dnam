/**
 * Recursive deep-freeze utility for immutable plain metadata objects
 * (`ref[sdk.ts.api.resources-and-data]`).
 *
 * @module
 */

/**
 * Recursively freezes a plain object or array in-place (skipping TypedArrays and already-frozen objects).
 */
export function deepFreeze<T>(value: T): Readonly<T> {
  if (value === null || typeof value !== "object") {
    return value;
  }
  if (ArrayBuffer.isView(value) || value instanceof ArrayBuffer) {
    return value;
  }
  if (Object.isFrozen(value)) {
    return value;
  }
  if (Array.isArray(value)) {
    for (let i = 0; i < value.length; i++) {
      deepFreeze(value[i]);
    }
  } else {
    for (const key of Object.keys(value)) {
      deepFreeze((value as Record<string, unknown>)[key]);
    }
  }
  return Object.freeze(value);
}
