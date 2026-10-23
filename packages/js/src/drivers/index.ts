/**
 * Driver registry and resolution (`resolveDriver`) for `@dnam-project/engine`.
 */

import { DnamError } from "../errors.js";
import type { BackendChoice, DnamDriver } from "./types.js";
import {
  getCachedWasmBinary,
  getWasmDriver,
  hasWasmDriver,
  initDnamWasm,
  initDnamWasmSync,
} from "./wasm.js";

let defaultNativeDriver: DnamDriver | null = null;
let nodeWasmLoader: (() => DnamDriver) | null = null;

export function registerNativeDriver(driver: DnamDriver): void {
  defaultNativeDriver = driver;
}

export function registerNodeWasmAutoLoader(loader: () => DnamDriver): void {
  nodeWasmLoader = loader;
}

export function resolveDriver(backend?: BackendChoice | string): DnamDriver {
  const b = (backend ?? "auto").toLowerCase();
  if (b === "wasm" || b === "wasm_simd128") {
    if (hasWasmDriver()) {
      return getWasmDriver();
    }
    if (nodeWasmLoader !== null) {
      return nodeWasmLoader();
    }
    return getWasmDriver();
  }
  if (defaultNativeDriver !== null) {
    return defaultNativeDriver;
  }
  if (hasWasmDriver()) {
    return getWasmDriver();
  }
  if (nodeWasmLoader !== null) {
    return nodeWasmLoader();
  }
  throw new DnamError(
    "No DNAM driver initialized. In Browser environments, call `await initDnamWasm()` before creating a DnamEngine.",
  );
}

export type {
  BackendChoice,
  DnamControlSpaceHandle,
  DnamDriver,
  DnamEngineHandle,
  RawClampResult,
  ValidatedContainerRaw,
} from "./types.js";
export { getCachedWasmBinary, getWasmDriver, hasWasmDriver, initDnamWasm, initDnamWasmSync };
