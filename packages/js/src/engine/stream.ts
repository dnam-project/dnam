/**
 * WHATWG `TransformStream` wrapper for streaming audio blocks through an `Engine`
 * (`ref[sdk.ts.engine.streaming-and-render]`).
 *
 * @module
 */

import type { AudioInput } from "../types.js";

/**
 * Options for `engine.stream(options)`.
 */
export interface StreamOptions {
  readonly blockSize?: number;
}

/**
 * Creates a WHATWG `TransformStream<AudioInput, Float32Array[]>` that processes incoming
 * planar or mono audio chunks through `processChunk`.
 */
export function createEngineTransformStream(
  processChunk: (chunk: AudioInput, blockSize?: number) => Float32Array[],
  options?: StreamOptions,
): TransformStream<AudioInput, Float32Array[]> {
  const blockSize = options?.blockSize;
  return new TransformStream<AudioInput, Float32Array[]>({
    transform(chunk, controller) {
      const out = processChunk(chunk, blockSize);
      controller.enqueue(out);
    },
  });
}
