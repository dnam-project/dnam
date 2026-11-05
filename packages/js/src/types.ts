/// <reference lib="esnext.disposable" />

/**
 * Core domain literal unions, companion tuples, and shared resource scope interfaces
 * (`ref[sdk.ts.api.literal-unions-and-options]`, `ref[sdk.ts.api.disposable-scopes]`).
 *
 * @module
 */

/**
 * Canonical 4-tier slimmable model hierarchy (`ref[arch.tiers.arena-hierarchy]`).
 */
export const tiers = ["lite", "full", "pro", "ultra"] as const;

/**
 * Canonical 4-tier slimmable model identifier (`"lite" | "full" | "pro" | "ultra"`).
 *
 * @example
 * ```ts
 * const tier: Tier = "full";
 * ```
 */
export type Tier = (typeof tiers)[number];

/**
 * Flexible tier input accepted by SDK methods (`Tier` string literal or `0..3` numeric index).
 */
export type TierInput = Tier | 0 | 1 | 2 | 3;

/**
 * Canonical SIMD execution backend identifiers (`ref[arch.topology.signal-flow]`).
 */
export const backends = ["auto", "scalar", "neon", "arm-sme", "x86-simd", "wasm-simd128"] as const;

/**
 * SIMD execution backend identifier.
 */
export type Backend = (typeof backends)[number];

/**
 * Active resolved hardware SIMD backend identifier (excludes `"auto"`).
 */
export type ActiveBackend = Exclude<Backend, "auto">;

/**
 * Canonical 12 musical tempo subdivisions (`REQ-2.5`, `ref[dsp.tempo.physical-inversion]`).
 */
export const subdivisions = [
  "1/2.",
  "1/2",
  "1/2T",
  "1/4.",
  "1/4",
  "1/4T",
  "1/8.",
  "1/8",
  "1/8T",
  "1/16.",
  "1/16",
  "1/16T",
] as const;

/**
 * Musical tempo subdivision literal (`"1/2."` through `"1/16T"`).
 */
export type Subdivision = (typeof subdivisions)[number];

/**
 * Graduated container protection profiles (`REQ-8.3`, `ref[sec.drm.protection-profiles]`).
 */
export const protectionProfiles = ["open", "device_allowlisted", "enclave_sealed"] as const;

/**
 * Container protection profile literal.
 */
export type ProtectionProfile = (typeof protectionProfiles)[number];

/**
 * Planar multi-channel `Float32Array` audio block (`[mono]` or `[left, right]`).
 */
export type AudioBlock = readonly Float32Array[];

/**
 * Flexible audio input accepted by `engine.process`, `engine.render`, and `engine.bounce`:
 * either a single mono `Float32Array` or planar channel arrays (`readonly Float32Array[]`).
 */
export type AudioInput = Float32Array | readonly Float32Array[];

/**
 * Scoped state handle returned by `engine.bypass(...)` and `engine.freeze(...)`
 * (`ref[sdk.ts.api.disposable-scopes]`).
 *
 * Supports both explicit `.restore()` and `using` (`[Symbol.dispose]()`).
 */
export interface StateScope extends Disposable {
  /**
   * Restores the prior engine state idempotently.
   */
  restore(): void;
  /**
   * Disposes the scope and restores the prior engine state (`using` protocol).
   */
  [Symbol.dispose](): void;
}
