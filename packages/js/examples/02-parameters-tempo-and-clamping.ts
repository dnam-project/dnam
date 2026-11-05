/**
 * Example 02 — Typed Parameters, 3-Phase Control Clamping & Tempo Synchronization.
 *
 * Run via:
 *   node packages/js/examples/02-parameters-tempo-and-clamping.ts
 */

import { clampParameters, defineModelTypes, Model } from "../dist/node.js";

const RigSchema = defineModelTypes([
  { id: "gain", type: "range" },
  { id: "boost", type: "switch", declaredOptions: ["off", "on", "ultra"] },
  { id: "delay_ms", type: "range" },
] as const);

using model = Model.reference<typeof RigSchema.$infer>();

// 1. Evaluate 3-Phase Control Clamping before audio processing
const report = clampParameters(model, {
  gain: 9.4,
  boost: "on",
  delay_ms: 350,
});
console.log("Clamp report:", {
  anyClamped: report.anyClamped,
  clampedIds: report.clampedIds,
  values: report.values,
});

// 2. Drive AudioParam-style Parameter handles and TempoController
using engine = model.createEngine({ sampleRate: 48_000, tier: "full" });
const gain = engine.parameters.get("gain");
const boost = engine.parameters.get("boost");

boost.setValue("on", { immediate: true });
gain.setValue(9.0, { immediate: true }); // Conditionally clamped to 8.0 when boost="on"
console.log(`gain.value=${String(gain.value)} dB (clamped=${String(gain.clamped)})`);

// 3. Synchronize tempo-enabled parameters to 120 BPM dotted-eighth ("1/8.")
const updatedCount = engine.tempo.sync({ bpm: 120, subdivision: "1/8." });
console.log(
  `Synced ${String(updatedCount)} tempo parameter(s) at ${String(engine.tempo.bpm)} BPM -> delay_ms=${String(engine.parameters.get("delay_ms").value.toFixed(2))} ms`,
);
