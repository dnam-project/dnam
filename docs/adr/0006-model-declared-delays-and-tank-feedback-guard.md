---
id: ADR-0006
title: "Model-Declared fp16 Delay Histories, Centered Granular Pitch & Stage IV Feedback Guard"
status: accepted
supersedes: []
superseded_by: null
specs:
  - docs/specs/dsp/op_t_transport.md
  - docs/specs/training/control_space_audit.md
requirements:
  - REQ-1.4
---

# ADR-0006: Model-Declared `fp16` Delay Histories, Centered Granular Pitch & Stage IV Feedback Guard

This record justifies the delay history, pitch windowing, and feedback stability rules in [`../specs/dsp/op_t_transport.md`](../specs/dsp/op_t_transport.md) and [`../specs/training/control_space_audit.md`](../specs/training/control_space_audit.md) fulfilling [`REQ-1.4`](../requirements/system_goals_and_requirements.md).

## 1. Decision & Counterexample Matrix

| Mechanism | Canonical Selection | Counterexample to Naive Formulation |
| :--- | :--- | :--- |
| **Model-Declared Delay Histories (`ref[dsp.op-t.ring.capacity]`, `ref[dsp.op-t.ring.q0-32-pointer]`)** | Cold-allocated 64B-aligned `fp16`/`f32` histories with arbitrary-length branchless wrap, integer-plus-`Q0.32` delay pointers, inward representable `f32` reader bounds, and host memory-ceiling admission. | Fixed power-of-two ring assumption wastes up to `50%` RAM or truncates `20 BPM` dotted-half (`9 s`) delays; unguarded `f32` fraction rounding can carry across fractional safe bounds. |
| **Granular Pitch Shifter (`±24` st, `ref[dsp.op-t.window.m2-pitch]`)** | Scope the `0.25` sample/sample slew limit to continuous base-delay changes; use the centered grain-duration construction (`D_center ≥ 7,204` samples for a `100 ms` grain at `48 kHz` and `+24` st) with per-head causal floor `τ_m ≥ 4.0`. | Applying `0.25` sample/sample slew to grain-head phase warp caps pitch shift well below `+24` semitones (`3.0` samples/sample warp rate). |
| **Stage IV Blended Feedback Knee & History Guard (`ref[dsp.op-t.guard.bounded-history]`, `ref[dsp.stage4.quiet-feedback-knee]`)** | Per-line blended feedback knee `K(u) = g_lin u ((1 - α) + α / (1 + (u/σ)²))`, input + non-recirculating readout DC conditioning, and grouped max-tree bounded-history feedback guard `clamp(v_i[n], -M_n, M_n)`. | Even within the `0.25` slew limit at `3 kHz` control rate, a time-varying Catmull-Rom loop at `g = 0.999` has a period multiplier in `(1.02, 1.03)` without the bounded-history feedback guard. |

## 2. Evaluated R&D Candidates

1. **Bounded Hermite Reader & Passive Projected-State Delay:**
   - *Bounded Hermite Slope Clamp:* For $d_0 = x_1 - x_0$ and raw centered differences $m_0 = (x_1 - x_{-1})/2$, $m_1 = (x_2 - x_0)/2$, the bounded slope clamp $T(a, b) = \frac{1}{2}\big(\mathrm{clamp}(a + b, -3|d_0|, 3|d_0|) + \mathrm{clamp}(a - b, -|d_0|, |d_0|)\big)$ enforces $p(\mu) \in [\min(x_0, x_1), \max(x_0, x_1)]$.
   - *Passive Projected-State Update:* Projects history into orthogonal Legendre modes $z_m[n] = \langle \mathbf{x}_n, \mathbf{q}_m(\mu_{n-1}) \rangle$ so time-varying delay sweeps remain contractive ($\Vert \mathbf{z}[n] \Vert_2 \le \Vert \mathbf{x}_n \Vert_2$).
2. **Octave/Fractional Fixed-Point Pitch Table:** Decomposes $p_{\text{st}} = 12 q + r$ with 257-entry table $P[idx] = \lfloor 2^{idx/256} \cdot 2^{30} + 0.5 \rfloor$ and cubic correction, achieving $< 0.002$ cents error without runtime `exp2f`.
