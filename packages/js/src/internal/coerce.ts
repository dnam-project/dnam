/**
 * Input coercion helpers for tiers, backends, subdivisions, binary buffers, and planar audio
 * (`ref[sdk.ts.api.literal-unions-and-options]`).
 *
 * @module
 */

import { DnamError } from "../errors.js";
import {
  type ActiveBackend,
  type AudioInput,
  type Backend,
  backends,
  type Subdivision,
  subdivisions,
  type Tier,
  type TierInput,
  tiers,
} from "../types.js";

/**
 * Coerces a `TierInput` (`"lite" | "full" | "pro" | "ultra" | 0 | 1 | 2 | 3`) to its numeric index `0..3`.
 */
export function coerceTierIndex(tier: TierInput | string | number): 0 | 1 | 2 | 3 {
  if (typeof tier === "number") {
    if (Number.isInteger(tier) && tier >= 0 && tier <= 3) {
      return tier as 0 | 1 | 2 | 3;
    }
    throw new DnamError(
      "ERR_DNAM_VALIDATION",
      `Invalid tier index: ${String(tier)} (expected 0..3)`,
    );
  }
  const normalized = tier.trim().toLowerCase();
  const idx = tiers.indexOf(normalized as Tier);
  if (idx === -1) {
    throw new DnamError(
      "ERR_DNAM_VALIDATION",
      `Unknown tier '${tier}' (expected one of: ${tiers.join(", ")})`,
    );
  }
  return idx as 0 | 1 | 2 | 3;
}

/**
 * Converts a numeric tier index `0..3` to its canonical `Tier` string literal.
 */
export function tierFromIndex(index: number): Tier {
  const tier = tiers[index];
  if (tier === undefined) {
    throw new DnamError("ERR_DNAM_VALIDATION", `Invalid tier index: ${String(index)}`);
  }
  return tier;
}

/**
 * Coerces a `TierInput` to its canonical `Tier` string literal.
 */
export function coerceTier(tier: TierInput | string | number): Tier {
  return tierFromIndex(coerceTierIndex(tier));
}

const BACKEND_ALIASES: Readonly<Record<string, Backend>> = {
  auto: "auto",
  native: "auto",
  scalar: "scalar",
  neon: "neon",
  arm_neon: "neon",
  "arm-neon": "neon",
  arm64_neon: "neon",
  arm_sve2: "neon",
  "arm-sve2": "neon",
  armsve2: "neon",
  sve2: "neon",
  arm_sme: "arm-sme",
  "arm-sme": "arm-sme",
  armsme: "arm-sme",
  sme: "arm-sme",
  x86_simd: "x86-simd",
  "x86-simd": "x86-simd",
  x86_sse2: "x86-simd",
  "x86-sse2": "x86-simd",
  sse2: "x86-simd",
  x86_sse42: "x86-simd",
  "x86-sse42": "x86-simd",
  x86sse42: "x86-simd",
  sse42: "x86-simd",
  "sse4.2": "x86-simd",
  x86_avx2_fma: "x86-simd",
  "x86-avx2-fma": "x86-simd",
  x86avx2fma: "x86-simd",
  avx2: "x86-simd",
  wasm_simd128: "wasm-simd128",
  "wasm-simd128": "wasm-simd128",
  wasm: "wasm-simd128",
};

const BACKEND_CODES: Readonly<Record<Backend, number>> = {
  scalar: 0,
  neon: 1,
  "arm-sme": 2,
  "x86-simd": 3,
  "wasm-simd128": 4,
  auto: 255,
};

const ACTIVE_BACKEND_FROM_CODE: readonly ActiveBackend[] = [
  "scalar",
  "neon",
  "arm-sme",
  "x86-simd",
  "wasm-simd128",
];

/**
 * Coerces a backend string to its canonical `Backend` literal.
 */
export function coerceBackend(backend: Backend | string = "auto"): Backend {
  const mapped = BACKEND_ALIASES[backend.trim().toLowerCase()];
  if (mapped === undefined) {
    throw new DnamError(
      "ERR_DNAM_VALIDATION",
      `Unknown backend '${backend}' (expected one of: ${backends.join(", ")})`,
    );
  }
  return mapped;
}

/**
 * Converts a `Backend` literal to its compact numeric C-ABI code (`0..4` or `255` for `"auto"`).
 */
export function backendToCode(backend: Backend | string = "auto"): number {
  return BACKEND_CODES[coerceBackend(backend)];
}

/**
 * Converts a compact numeric C-ABI backend code (`0..4`) to its canonical `ActiveBackend` literal.
 */
export function backendFromCode(code: number): ActiveBackend {
  return ACTIVE_BACKEND_FROM_CODE[code] ?? "scalar";
}

/**
 * Validates and returns a canonical `Subdivision` literal.
 */
export function coerceSubdivision(subdivision: Subdivision | string): Subdivision {
  const trimmed = subdivision.trim() as Subdivision;
  if (!subdivisions.includes(trimmed)) {
    throw new DnamError(
      "ERR_DNAM_VALIDATION",
      `Unknown tempo subdivision '${subdivision}' (expected one of: ${subdivisions.join(", ")})`,
    );
  }
  return trimmed;
}

/**
 * Coerces a `Uint8Array | ArrayBuffer | ArrayBufferView` into a `Uint8Array`.
 */
export function coerceBytes(input: Uint8Array | ArrayBuffer | ArrayBufferView): Uint8Array {
  if (input instanceof Uint8Array) {
    return input;
  }
  if (input instanceof ArrayBuffer) {
    return new Uint8Array(input);
  }
  if (ArrayBuffer.isView(input)) {
    return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  }
  throw new DnamError("ERR_DNAM_VALIDATION", "Expected Uint8Array or ArrayBuffer");
}

/**
 * Normalizes an `AudioInput` (`Float32Array` or `readonly Float32Array[]`) into planar channels
 * (`[mono]` or `[left, right]`) and validates uniform frame length.
 */
export function coerceAudioChannels(input: AudioInput): {
  readonly channels: readonly Float32Array[];
  readonly frames: number;
} {
  if (input instanceof Float32Array) {
    return { channels: [input], frames: input.length };
  }
  if (!Array.isArray(input) || input.length === 0 || input.length > 2) {
    throw new DnamError(
      "ERR_DNAM_VALIDATION",
      "Audio input must be a Float32Array or a 1- or 2-channel Float32Array[]",
    );
  }
  const ch0 = input[0];
  if (!(ch0 instanceof Float32Array)) {
    throw new DnamError("ERR_DNAM_VALIDATION", "Audio channel 0 must be a Float32Array");
  }
  const frames = ch0.length;
  if (input.length === 2) {
    const ch1 = input[1];
    if (!(ch1 instanceof Float32Array)) {
      throw new DnamError("ERR_DNAM_VALIDATION", "Audio channel 1 must be a Float32Array");
    }
    if (ch1.length !== frames) {
      throw new DnamError(
        "ERR_DNAM_VALIDATION",
        `Stereo channel length mismatch: left=${String(frames)}, right=${String(ch1.length)}`,
      );
    }
    return { channels: [ch0, ch1], frames };
  }
  return { channels: [ch0], frames };
}
