/**
 * Example 06 — Browser / WebAudio `AudioWorklet` Real-Time Node (`@dnam-project/dnam/web-audio`).
 *
 * Demonstrates creating a `DnamAudioNode` with lock-free `SharedArrayBuffer` (`Atomics`)
 * parameter synchronization and seqlock telemetry events.
 *
 * Run via:
 *   node packages/js/examples/06-web-audio-worklet.ts
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineModelTypes, Model } from "../dist/node.js";
import { DnamAudioNode } from "../dist/web-audio/index.js";

const RigSchema = defineModelTypes([
  { id: "gain", type: "range" },
  { id: "boost", type: "switch", declaredOptions: ["off", "on", "ultra"] },
  { id: "delay_ms", type: "range" },
] as const);

const dir = dirname(fileURLToPath(import.meta.url));
const wasmModule = new WebAssembly.Module(readFileSync(resolve(dir, "../dist/dnam.wasm")));

using model = Model.reference<typeof RigSchema.$infer>();
const mockAudioContext = {
  sampleRate: 48_000,
  audioWorklet: {
    async addModule(_url: string | URL) {
      // In a browser AudioContext, this registers `dist/web-audio-processor.js`
    },
  },
};

using node = await DnamAudioNode.create(mockAudioContext, model, {
  wasmModule,
  tier: "full",
  useSharedMemory: true,
});

node.parameters.get("gain").setValue(7.2, { immediate: true });
node.tempo.set({ bpm: 128, subdivision: "1/8." });

const telemetry = node.telemetry();
console.log(
  `DnamAudioNode ready: usesSharedMemory=${String(node.usesSharedMemory)}, tier=${node.tier}, bpm=${String(node.tempo.bpm)}, sr=${String(telemetry.sampleRateHz)} Hz`,
);
