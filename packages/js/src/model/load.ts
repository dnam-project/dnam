/**
 * Asynchronous and synchronous binary container source loaders (`ModelSource`)
 * (`ref[sdk.ts.model.containers-and-checkpoints]`).
 *
 * @module
 */

import { DnamError } from "../errors.js";
import { coerceBytes } from "../internal/coerce.js";

/**
 * Input sources accepted by `await Model.load(source)`.
 */
export type ModelSource =
  | string
  | URL
  | Uint8Array
  | ArrayBuffer
  | ArrayBufferView
  | Blob
  | Response
  | ReadableStream<Uint8Array>;

function isReadableStream(val: unknown): val is ReadableStream<Uint8Array> {
  return (
    val !== null &&
    typeof val === "object" &&
    "getReader" in val &&
    typeof (val as ReadableStream<Uint8Array>).getReader === "function"
  );
}

async function readStreamToBytes(stream: ReadableStream<Uint8Array>): Promise<Uint8Array> {
  const reader = stream.getReader();
  const chunks: Uint8Array[] = [];
  let totalLength = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      if (value) {
        chunks.push(value);
        totalLength += value.byteLength;
      }
    }
  } finally {
    reader.releaseLock();
  }
  const combined = new Uint8Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    combined.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return combined;
}

/**
 * Reads any `ModelSource` (`string` path/URL, `URL`, `Blob`, `Response`, `ReadableStream`,
 * `Uint8Array`, or `ArrayBuffer`) into a `Uint8Array`.
 */
export async function readModelSourceBytes(source: ModelSource): Promise<Uint8Array> {
  if (source instanceof Uint8Array || source instanceof ArrayBuffer || ArrayBuffer.isView(source)) {
    return coerceBytes(source);
  }
  if (typeof Blob !== "undefined" && source instanceof Blob) {
    return new Uint8Array(await source.arrayBuffer());
  }
  if (typeof Response !== "undefined" && source instanceof Response) {
    if (!source.ok) {
      throw new DnamError(
        "ERR_DNAM_CONTAINER",
        `Failed to fetch container: HTTP ${String(source.status)}`,
      );
    }
    return new Uint8Array(await source.arrayBuffer());
  }
  if (isReadableStream(source)) {
    return readStreamToBytes(source);
  }
  if (source instanceof URL || typeof source === "string") {
    const str = source instanceof URL ? source.href : source;
    const isHttpOrBlobUrl =
      str.startsWith("http://") ||
      str.startsWith("https://") ||
      str.startsWith("blob:") ||
      str.startsWith("data:");
    const isNodeEnv =
      typeof process !== "undefined" &&
      typeof process.versions === "object" &&
      Boolean(process.versions?.node);

    if (!isHttpOrBlobUrl && isNodeEnv) {
      try {
        const fsModuleName = "node:fs/promises";
        const fs = (await import(/* @vite-ignore */ fsModuleName)) as {
          readFile(path: string | URL): Promise<Uint8Array>;
        };
        const filePath = source instanceof URL && source.protocol === "file:" ? source : str;
        const buf = await fs.readFile(filePath);
        return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
      } catch (err) {
        throw new DnamError(
          "ERR_DNAM_CONTAINER",
          `Failed to read container file '${str}': ${String(err)}`,
          { cause: err },
        );
      }
    }

    if (typeof fetch === "function") {
      const resp = await fetch(source);
      if (!resp.ok) {
        throw new DnamError(
          "ERR_DNAM_CONTAINER",
          `Failed to fetch container from '${str}': HTTP ${String(resp.status)}`,
        );
      }
      return new Uint8Array(await resp.arrayBuffer());
    }
  }
  throw new DnamError("ERR_DNAM_VALIDATION", "Unsupported ModelSource input");
}
