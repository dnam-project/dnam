/**
 * Example 01 — Quickstart: Load a `.dnam` model, inspect frozen camelCase metadata,
 * process real-time 128-frame stereo audio, and render a 4x oversampled offline bounce.
 *
 * Run via:
 *   node packages/js/examples/01-quickstart-render.ts
 */

import { Model } from "../dist/node.js";

using model = Model.reference({ supportsFreeze: true });
console.log(
  `Loaded model: ${model.name} (maxTier=${model.maxTier}, sr=${String(model.sampleRateTrainHz)} Hz)`,
);
console.log(
  "Parameters:",
  model.parameters.map((p) => `${p.id} (${p.type}, default=${String(p.default)})`).join(", "),
);

using engine = model.createEngine({ sampleRate: 48_000, tier: "full" });
console.log(`Engine active on backend=${engine.backend}, effectiveTier=${engine.effectiveTier}`);

// Process a 128-frame stereo block with zero-allocation output buffers
const leftIn = new Float32Array(128).fill(0.25);
const rightIn = new Float32Array(128).fill(-0.25);
const leftOut = new Float32Array(128);
const rightOut = new Float32Array(128);
engine.process([leftIn, rightIn], { output: [leftOut, rightOut] });

// Scoped Stage IV hardware freeze (`using` automatically restores state on exit)
{
  using _freeze = engine.freeze({ amount: 0.85 });
  engine.process([leftIn, rightIn], { output: [leftOut, rightOut] });
}

// High-precision 4x oversampled offline studio bounce at Ultra tier
const [bounceL, bounceR] = engine.bounce([leftIn, rightIn], { maxTier: true });
console.log(
  `Rendered ${String(bounceL?.length ?? 0)} frames (peak L=${Math.max(...(bounceL ?? [])).toFixed(4)}, R=${Math.max(...(bounceR ?? [])).toFixed(4)})`,
);
