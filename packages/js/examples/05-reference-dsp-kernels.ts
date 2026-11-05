/**
 * Example 05 — Reference TORA DSP Kernels (`@dnam-project/dnam/dsp`).
 *
 * Run via:
 *   node packages/js/examples/05-reference-dsp-kernels.ts
 */

import "../dist/node.js";
import {
  givensHouseholder8ch,
  lobattoBasisSlice,
  modalSlice,
  polyAdaaSlice,
  rank4Fm,
  tankSlice,
} from "../dist/dsp/index.js";

const frames = 64;
const rawAudio = new Float64Array(frames).map((_, i) => Math.sin(i * 0.2) * 0.8);

// 1. Op-T Coercive Delay Tank + DC-blocked write
const opTOut = tankSlice(rawAudio, { rDc: 0.995, gLin: 0.7 });
console.log(`Op-T tank slice: y[last]=${String(opTOut[frames - 1]?.toFixed(5))}`);

// 2. Op-M Symplectic Modal resonator
const opMRes = modalSlice(opTOut, {
  rLow: 0.92,
  drSpan: 0.06,
  uRotCos: 0.98,
  uRotSin: 0.198,
});
console.log(`Op-M modal response: y[last]=${String(opMRes.output[frames - 1]?.toFixed(5))}`);

// 3. Op-N Division-free 1x Passive B-Spline PolyADAA
const opNRes = polyAdaaSlice(opMRes.output, { beta: 1.5, delta: 0.1 });
console.log(`Op-N PolyADAA output: y[last]=${String(opNRes.output[frames - 1]?.toFixed(5))}`);

// 4. Control Bus & 8-channel Givens-Householder mixing
const lobatto = lobattoBasisSlice(new Float64Array([-0.5, 0.0, 0.5]));
const fm4 = rank4Fm([0.5, -0.25, 0.75], new Float64Array(12).fill(0.2));
const mixed8 = givensHouseholder8ch([1, 2, 3, 4, 5, 6, 7, 8], Math.cos(0.4), Math.sin(0.4));
console.log(
  `Lobatto len=${String(lobatto.length)}, Rank-4 FM=[${fm4.map((v) => v.toFixed(4)).join(", ")}], Mixed8[0]=${String(mixed8[0]?.toFixed(4))}`,
);
