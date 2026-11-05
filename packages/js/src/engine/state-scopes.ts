/**
 * Dual-mode `StateScope` (`restore()` + `[Symbol.dispose]()`) for temporary bypass and freeze
 * (`ref[sdk.ts.api.disposable-scopes]`).
 *
 * @module
 */

import type { StateScope } from "../types.js";

/**
 * Options for `engine.bypass(options)` and `engine.withBypass(options, fn)`.
 */
export interface BypassOptions {
  /**
   * When `true` (default), enables spillover trails (`SpilloverTrails`); when `false`,
   * engages hard bypass (`HardBypass`, `ref[arch.topology.spillover-trails]`).
   */
  readonly trails?: boolean;
}

/**
 * Options for `engine.freeze(options)` and `engine.withFreeze(options, fn)`.
 */
export interface FreezeOptions {
  /**
   * Continuous Stage IV freeze blend amount $\alpha \in [0, 1]$ (`default = 1.0`).
   */
  readonly amount?: number;
}

/**
 * Creates an idempotent `StateScope` implementing both `.restore()` and `[Symbol.dispose]()`.
 */
export function createStateScope(onRestore: () => void): StateScope {
  let restored = false;
  const restore = (): void => {
    if (restored) {
      return;
    }
    restored = true;
    onRestore();
  };
  return Object.freeze({
    restore,
    [Symbol.dispose]: restore,
  });
}
