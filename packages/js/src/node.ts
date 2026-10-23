/**
 * Node.js entry point for `@dnam-project/engine` (`@dnam-project/engine/node`).
 * Automatically registers the zero-copy N-API v8 `NativeDriver` and synchronous Node `WasmDriver`.
 */

import "./drivers/native.js";

export * from "./browser.js";
export { ensureNodeWasmDriver, getNativeDriver, NativeDriver } from "./drivers/native.js";
