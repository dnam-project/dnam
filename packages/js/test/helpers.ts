/**
 * Shared test helpers and deterministic signal generators for `@dnam-project/engine` test suites.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { ensureNodeWasmDriver } from "../src/index.js";

export const BACKENDS = ["native", "wasm"] as const;

let cachedWasmBytes: Uint8Array | null = null;

export function getWasmBytes(): Uint8Array {
  if (cachedWasmBytes !== null) {
    return cachedWasmBytes;
  }
  const currentDir = dirname(fileURLToPath(import.meta.url));
  const wasmPath = resolve(currentDir, "..", "dist", "dnam_engine.wasm");
  cachedWasmBytes = new Uint8Array(readFileSync(wasmPath));
  return cachedWasmBytes;
}

export async function ensureWasmReady(): Promise<void> {
  ensureNodeWasmDriver();
}

export function makeSineSignalF32(
  length: number,
  freqHz = 440.0,
  sampleRateHz = 48_000,
  amplitude = 0.5,
): Float32Array {
  const out = new Float32Array(length);
  const omega = (2.0 * Math.PI * freqHz) / sampleRateHz;
  for (let n = 0; n < length; n++) {
    out[n] = amplitude * Math.sin(omega * n);
  }
  return out;
}

export function makeSineSignalF64(
  length: number,
  freqHz = 440.0,
  sampleRateHz = 48_000,
  amplitude = 0.5,
): Float64Array {
  const out = new Float64Array(length);
  const omega = (2.0 * Math.PI * freqHz) / sampleRateHz;
  for (let n = 0; n < length; n++) {
    out[n] = amplitude * Math.sin(omega * n);
  }
  return out;
}

export function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = Math.min(a.length, b.length);
  let maxDiff = 0.0;
  for (let i = 0; i < n; i++) {
    const diff = Math.abs((a[i] ?? 0) - (b[i] ?? 0));
    if (diff > maxDiff) {
      maxDiff = diff;
    }
  }
  return maxDiff;
}

export function rms(arr: ArrayLike<number>): number {
  if (arr.length === 0) {
    return 0.0;
  }
  let sumSq = 0.0;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i] ?? 0;
    sumSq += v * v;
  }
  return Math.sqrt(sumSq / arr.length);
}
