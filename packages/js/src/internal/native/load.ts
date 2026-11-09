/**
 * Active `NativeDriver` registry (`getActiveDriver`, `setActiveDriver`, `withDriver`, `initWasm`)
 * (`ref[sdk.ts.runtime.dual-target-drivers]`, `ref[sdk.ts.api.sync-init]`).
 *
 * @module
 */

import { DnamError } from "../../errors.js";
import type { NativeDriver } from "./driver.js";
import { compileWasmModule, type InitWasmOptions, instantiateWasmDriver } from "./wasm.js";

let activeDriver: NativeDriver | undefined;

/**
 * Returns the currently active `NativeDriver`, throwing `DnamError` if no runtime entrypoint
 * (`@dnam-project/dnam` / `initWasm`) has initialized a driver yet.
 */
export function getActiveDriver(): NativeDriver {
  if (!activeDriver) {
    throw new DnamError(
      "ERR_DNAM_CONTAINER",
      "Native DNAM driver is not initialized. Import '@dnam-project/dnam' or call await initWasm().",
    );
  }
  return activeDriver;
}

/**
 * Sets the active `NativeDriver` instance.
 */
export function setActiveDriver(driver: NativeDriver): void {
  activeDriver = driver;
}

/**
 * Executes `fn` with `driver` temporarily installed as the active driver (supporting both
 * synchronous callbacks and `Promise`-returning async callbacks).
 */
export function withDriver<T>(driver: NativeDriver, fn: () => T): T {
  const prev = activeDriver;
  activeDriver = driver;
  let isAsync = false;
  try {
    const result = fn();
    if (result instanceof Promise) {
      isAsync = true;
      return result.finally(() => {
        activeDriver = prev;
      }) as T;
    }
    return result;
  } finally {
    if (!isAsync) {
      activeDriver = prev;
    }
  }
}

/**
 * Initializes (or re-initializes) the WebAssembly `SIMD128` driver from a custom URL,
 * `BufferSource`, or pre-compiled `WebAssembly.Module` (`ref[sdk.ts.api.sync-init]`).
 *
 * @example
 * ```ts
 * const wasmModule = await initWasm({ url: "/assets/dnam.wasm" });
 * ```
 */
export async function initWasm(options?: InitWasmOptions): Promise<WebAssembly.Module> {
  const mod = await compileWasmModule(options);
  setActiveDriver(instantiateWasmDriver(mod, options));
  return mod;
}
