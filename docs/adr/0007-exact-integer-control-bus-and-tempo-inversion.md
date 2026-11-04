---
id: ADR-0007
title: "Centered ANOVA + Rank-R FM Control Bus, Exact-Integer Accumulator & Physical Tempo Inversion"
status: accepted
supersedes: []
superseded_by: null
specs:
  - docs/specs/dsp/parametric_control_bus.md
requirements:
  - REQ-2.4
  - REQ-2.5
---

# ADR-0007: Centered ANOVA + Rank-`R` FM Control Bus, Exact-Integer Accumulator & Physical Tempo Inversion

This record justifies the control-bus and tempo-sync formulations in [`../specs/dsp/parametric_control_bus.md`](../specs/dsp/parametric_control_bus.md) fulfilling [`REQ-2.4`, `REQ-2.5`](../requirements/system_goals_and_requirements.md).

## 1. Context: Why Naive Conditioning Architectures & $P \le 16$ Control Buses Fail

| Naive Architecture | Structural Failure Mode |
| :--- | :--- |
| **1. Audio-Rate (`48 kHz`) Input Concatenation / FiLM** | Re-evaluates knob embeddings on every audio sample (`+1,536 MACs/sample`), wasting `> 40%` of the embedded Cortex-A76 cycle budget even when knobs are stationary. |
| **2. Unconstrained Dense MLP Hypernetwork** | Overfits sparse `Star`/`Corner` stem grids (`15–30` stems) and hallucinates unstable pole radii (`r > 1.0`) or negative delays at unvisited multi-knob corners. |
| **3. High-Degree Uniform Polynomials** | Fitting degree-3 polynomials to a control captured at only `1` or `2` settings triggers severe Runge oscillation across interior knob positions. |
| **4. Unfactored Pairwise Tensor Products** | Full pairwise interaction tables scale as `O(P²)` in memory and `O(P)` per control update, hitting a hard `P ≤ 16` wall on multi-channel studio rigs (`P = 20..64` controls). |

## 2. Decision & Counterexample Matrix

| Mechanism | Canonical Selection | Counterexample to Naive Formulation |
| :--- | :--- | :--- |
| **Automation Timing & Tie Precedence (`ref[dsp.ctrl.op-m.timing]`)** | Apply host parameter updates at the `3 kHz` control boundary (`n mod K_ctrl == 0`), coalescing in timestamp order. On exact-timestamp ties: DAW automation wins over live Studio/MIDI; between live Studio and MIDI, last-received wins. | Sample-rate re-projection of `W_ctrl` on every host event or arbitrary array-order tie breaking. |
| **Exact-Integer Sufficient-Statistic Accumulator (`ref[dsp.ctrl.accum.exact-int]`)** | Accumulate materialized finite `f32` main effects and Rank-`R` interaction pair statistics exactly in fixed-width integers (updating pair sums via two shifted 24-bit significand products) and feed the exact-real oracle into trust-region clipping before narrowing to `f32`. | Naive `f32` `(∑ v_p z_p)² - ∑ v_p² z_p²` suffers catastrophic cancellation and drifts by `> 0.001` after `10,000` closed three-step control cycles. |
| **Outward-Checked Trust-Region Clip (`ref[dsp.ctrl.op-n.trust-clip]`)** | Bit-preserving `f32 -> f64` widening, outward-rounded `f64` norm/output verification with 6-step Newton fast path and 32-step bisection fallback, bounding normal-radius vector error below `1.185e-6 * radius`. | Naive `f32` `rsqrt` trust-region scaling can round outside the exact ball `‖z‖₂ ≤ ρ` or flush subnormal coordinates under FTZ/DAZ. |
| **Tempo Range, Transport Lock & Reachable Physical Inversion (`REQ-2.5`, `ref[dsp.tempo.q64-q32-phase]`, `ref[dsp.tempo.physical-inversion]`)** | Support `20–400 BPM` inclusive across host, MIDI (`24 PPQN`), and tap sources; invert calibrated Lobatto polynomials over reachable `f32` physical knob coordinates (with inward `f32` rank intersection of `f64` safety bounds), optional octave folding, and `Q64`/`Q32` phase tracking. | Solving quadratic roots in normalized real coordinates can select unreachable `f32` values or wrong branches; averaging raw frame counts across sample-rate changes corrupts BPM (`120 -> 160 BPM`). |

## 3. Evaluated R&D Candidate: Fixed-Tree Control Accumulator

Maintains a balanced binary tree over $P \le 32$ controls where internal nodes store $S_v = S_L + S_R$ and $I_v = I_L + I_R + S_L \odot S_R$, updating in $\mathcal{O}(\log_2 P)$ steps per control move without subtraction cancellation (superseded by the $\mathcal{O}(1)$ exact-integer sufficient-statistic accumulator supporting $P \in [0, 255]$).
