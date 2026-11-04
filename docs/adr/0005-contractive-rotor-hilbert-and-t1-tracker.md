---
id: ADR-0005
title: "Quadrant-Reduced Contractive Rotor, H1 Hilbert Lattice, T1 Tracker & Stage III ZOH"
status: accepted
supersedes: []
superseded_by: null
specs:
  - docs/specs/dsp/op_m_modal_kautz.md
requirements:
  - REQ-1.2
  - REQ-1.3
---

# ADR-0005: Quadrant-Reduced Contractive Rotor, `H1` Hilbert Lattice, `T1` Tracker & Stage III ZOH

This record justifies the modal, Hilbert, and tracker formulations in [`../specs/dsp/op_m_modal_kautz.md`](../specs/dsp/op_m_modal_kautz.md) fulfilling [`REQ-1.2`, `REQ-1.3`](../requirements/system_goals_and_requirements.md).

## 1. Decision & Counterexample Matrix

| Mechanism | Canonical Selection | Counterexample to Naive Formulation |
| :--- | :--- | :--- |
| **Contractive Rotor & LFO Phase (`ref[dsp.op-m.rotor.contractive]`, `ref[dsp.op-m.rotor.q0-192-lfo]`)** | Quadrant reduction to `[0, π/4]`, degree-13/14 Horner `f64` evaluation with adaptive 1–2 ULP `f32` norm guard (`‖u_rot‖₂ < 1`), exact zero-increment hold, and `Q0.192` fixed-point LFO phase accumulator. | Unreduced `[-π/2, π/2]` polynomial suffers cancellation near `π/2`; `f32` rounded rotor state update stalls into a fixed point for slow LFO increments (`Δθ = 2^-30 rad`). |
| **Family 2 `H1` Hilbert Pair (`ref[dsp.op-m.hilbert.h1-anchors]`, `ref[dsp.op-m.hilbert.ties-to-even]`, `ref[dsp.op-m.hilbert.rate-remap]`)** | 6-section all-pass pair with 15-anchor rate table (`HILBERT-H1-ANCHORS`, `≤ 0.34°` quadrature error across `100 Hz–10 kHz`), exact ties-to-even interpolation, and history-consistent two-sample section state initialization on rate transitions. | Naive `f64` linear interpolation rounds halfway rates to the wrong `f32` neighbor; copying transposed state cells across rate changes breaks constant-input equilibrium. |
| **Family 1 `T1` Sub-Octave Tracker (`ref[dsp.op-m.tracker.t1-dc-blocker]`, `ref[dsp.op-m.tracker.toggle-sheet]`)** | Tracker-only `5 Hz` TPT DC blocker (`T1-COEFFICIENT-VECTORS`), scaled exponent-split normalization, independent `prev_i_nonpositive` crossing detector, smooth half-angle quadratures (`cos`, `sin`) + divider squares (`f/2`, `f/4`) in Lite slots `4..11`. | Testing crossings on the normalized sheet phasor misses or invents toggles when `I` flushes under FTZ; pausing tracker state during local mute loses divider phase parity. |
| **Stage III Sample-Rate Conversion (`ref[dsp.op-m.stage3.rate-convert]`, `ref[dsp.zoh.stage3-fir-onset]`)** | Integer multiples (`48, 96, 192, 384 kHz`) use exact sparse delay `H_ref(z^q)`; live `44.1–48 kHz` uses the 64-tap anchored difference family; `4×` (`176.4–384 kHz`) uses the 6-point Lagrange difference mapping; other non-integer rates fail fast. | 4-tap Catmull-Rom at `44.1 kHz` attenuates `20 kHz` by `4.51 dB`; forward `f32` summation of unit-DC taps yields `0x3F800002` without difference-form anchoring. |

## 2. Evaluated R&D Candidates & LFO Detection Profiles

1. **Quarter-Wave Fixed-Point Rotor Table:** 513-entry `u32` table $T[m] = \lfloor \sin(m\pi/1024) \cdot 2^{30} + 0.5 \rfloor$ with linear interpolation $S(u) = (1 - 2s)\big(((2^{16}-f)T[idx] + f T[idx+1] + 2^{15}) \gg 16\big)2^{-30}$ and unit-circle correction $\delta = \big((3 \cdot 2^{29} - (\tilde{c}^2 + \tilde{s}^2)2^{-31}) \gg 1\big)2^{-30}$.
2. **Repeated-Window Cubic-Nuisance & Helmert LFO Detection:** Evaluates harmonic regression over $M$ repeated windows against a cubic drift baseline ($1, \tau, \tau^2, \tau^3$) via exact $F(2, \nu)$ variance ratios or orthonormal Helmert contrasts $z_k[n] = \big(\sum_{j=1}^k y^{(j)}[n] - k y^{(k+1)}[n]\big) / \sqrt{k(k+1)}$.
