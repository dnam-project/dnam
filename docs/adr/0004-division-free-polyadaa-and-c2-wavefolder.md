---
id: ADR-0004
title: "Unclamped 6-Step PolyADAA Rail Crossing, C² Wavefolder & Fixed-Hertz ZOH Bitcrusher"
status: accepted
supersedes: []
superseded_by: null
specs:
  - docs/specs/dsp/op_n_commuted_cascade.md
requirements:
  - REQ-1.1
  - REQ-6.2
---

# ADR-0004: Unclamped 6-Step `PolyADAA` Rail Crossing, `C²` Wavefolder & Fixed-Hertz ZOH Bitcrusher

This record justifies the nonlinear operator formulations in [`../specs/dsp/op_n_commuted_cascade.md`](../specs/dsp/op_n_commuted_cascade.md) fulfilling [`REQ-1.1`, `REQ-6.2`](../requirements/system_goals_and_requirements.md).

## 1. Context & Mathematical Counterexamples

In `Op-N` nonlinear waveshaping, wavefolding, and digital quantization, naive endpoint clamping and float-to-integer truncation introduce severe numerical and acoustic artifacts.

## 2. Decision & Counterexample Matrix

| Mechanism | Canonical Selection | Counterexample to Naive Formulation |
| :--- | :--- | :--- |
| **PolyADAA Rail Crossing (`ref[dsp.op-n.polyadaa.rail-crossing]`)** | Retain unclamped prior input history and evaluate the exact piecewise antiderivative across `±1` rail crossings using a 6-step division-free `f64` Newton reciprocal (`< 2^-44` error before and `< 2^-24 + 2^-44` after `f32` conversion). | Clamping endpoints to `[-1, 1]` before ADAA loses overshoot information (`0 → 1` and `0 → 2` yield identical clamped endpoints `(5/8, 2/15)` instead of `(13/16, 1/15)`). |
| **Smooth-Rail Wavefolder (`ref[dsp.op-n.wavefolder.cubic-c2]`)** | Cubed-envelope wavefolders `w_odd,3(u) = ū(1 - ū²)³` and `w_even,3(u) = ū²(1 - ū²)³` (`C²` at `±1` rails) with exact degree-7/8 midpoint-difference `PolyADAA` (`Q_1..Q_8`) and direct-linear split. | Squared-envelope folds `u(1 - u²)²` and `u²(1 - u²)²` have discontinuous second derivatives (`±8`) at the rails (`C¹`, not `C²`). |
| **Family 5 ZOH Bitcrusher (`ref[dsp.op-n.bitcrusher.quantizer]`, `ref[dsp.op-n.bitcrusher.zoh-phase]`, `ref[dsp.op-n.bitcrusher.rate-transfer]`)** | Nearest-step quantizer `Q_B(x) = floor(x 2^(B-1) + 0.5) 2^-(B-1)` (halfway ties toward `+∞`) + integer fixed-hertz phase (`D = round(2^32 f_s,eff)`, `I = round(2^32 f_target)`) with pre-reduction carry detection and division-free `u128` reciprocal rate transfer. | Adding `0.5f32` before truncating `f32 -> i32` rounds negative values wrong and rounds `0.5 - 2^-25` to `1.0`; modulo reduction before carry check prevents full-rate (`I = D`) latching. |
