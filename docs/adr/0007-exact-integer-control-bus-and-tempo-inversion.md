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
| **Canonical `v2.0` `CP1` Arbitrary-`P` Control Projections (`ref[arch.pillars.cp1-projections]`, `ref[container.dnam.cp1-profile]`, `ref[dsp.op-n.ctrl.incremental-update]`)** | Retire the legacy `v1.1` fixed-slot `basis_16`/`basis_32` loop and unify on variable-length `CP1` contributor records (`Cp1ContributorSlot` + `LatentAccumulator64B` + Region 4 `ControlClampDescriptor` and overflow `CP1` stream), supporting arbitrary `P` controls in `O(1)` per moved control with zero real-time heap allocations on `reload_model_in_place`. | Fixed 16-slot in-arena basis arrays hit a hard control-count ceiling (`P ≤ 8..16`) or alias slots modulo 16 on multi-channel studio rigs. |
| **Switch `default_val` Zero-Baseline & Hadamard FM Identification (`ref[dsp.ctrl.switch.option-contributors]`, [`ADR-0016`](0016-codesigned-disjoint-comb-stimulus-and-null-homing-capture.md))** | Anchor each discrete `switch` at its signed `default_val` (`StarCenter` emits `0.0`) so only non-default captured options emit `1`-lane `CP1` records, and identify Rank-4/8 FM interaction factors (`U_int^base[8, 4]` and `3 × U_int,Delta[8, 4]`) from 2 orthogonal `3.0 s` Hadamard diagonal sweeps with automatic 1D commutator interaction pruning (`Theorem 3.5`). | Non-zero switch baselines shift `StarCenter` mode-0 audio; full `P(P-1)/2` pairwise captures explode capture time quadratically in `P`. |
| **Automation Timing & Tie Precedence (`ref[dsp.ctrl.op-m.timing]`)** | Apply host parameter updates at the `3 kHz` control boundary (`n mod K_ctrl == 0`), coalescing in timestamp order. On exact-timestamp ties: DAW automation wins over live Studio/MIDI; between live Studio and MIDI, last-received wins. | Sample-rate re-projection of `W_ctrl` on every host event or arbitrary array-order tie breaking. |
| **Exact-Integer Sufficient-Statistic Accumulator (`ref[dsp.ctrl.accum.cp1-scaling]`, `ref[dsp.ctrl.accum.exact-int]`)** | Accumulate materialized finite `f32` main effects and Rank-`R` interaction pair statistics (`ExactIntFactorAccumulator`) exactly in fixed-width integers (updating pair sums via two shifted 24-bit significand products) and feed the exact-real oracle into trust-region clipping before narrowing to `f32`. | Naive `f32` `(∑ v_p z_p)² - ∑ v_p² z_p²` suffers catastrophic cancellation and drifts by `> 0.001` after `10,000` closed three-step control cycles. |
| **4-D Groupwise Outward-Checked Trust-Region Clip & Disjoint `W_ctrl` Stage Dispatch (`ref[dsp.ctrl.op-n.trust-clip]`, [`ADR-0011`](0011-end-to-end-stage-wiring-and-arena-slice-topology.md))** | Apply outward-checked `f64` trust-region clipping independently to each 4-D block (`z_head[0..4]`, `z_stage[0..4]`, `z_head[4..8]`, `z_stage[4..8]`) and drive `theta_0..15` via `16 × 16` `W_ctrl^base` (`Columns 4..8` for `z_stage[0..4]`, `Columns 8..16` for `m_dyn`) + `16 × 4` `W_ctrl,Delta^Ultra` (`z_stage[4..8]`). | Joint 8-D/16-D norm clipping couples head and stage coordinates and breaks Mechanism A `Q_buyer` gauge invariance; `8 × 32` tank-only dispatch ignores Stage I–III modulation. |
| **Tempo Range, Transport Lock & Reachable Physical Inversion (`REQ-2.5`, `ref[dsp.tempo.q64-q32-phase]`, `ref[dsp.tempo.physical-inversion]`)** | Support `20–400 BPM` inclusive across host, MIDI (`24 PPQN`), and tap sources; invert calibrated Lobatto polynomials over reachable `f32` physical knob coordinates (with inward `f32` rank intersection of `f64` safety bounds), optional octave folding, and `Q64`/`Q32` phase tracking. | Solving quadratic roots in normalized real coordinates can select unreachable `f32` values or wrong branches; averaging raw frame counts across sample-rate changes corrupts BPM (`120 -> 160 BPM`). |

## 3. Evaluated R&D Candidate: Fixed-Tree Control Accumulator

Maintains a balanced binary tree over $P \le 32$ controls where internal nodes store $S_v = S_L + S_R$ and $I_v = I_L + I_R + S_L \odot S_R$, updating in $\mathcal{O}(\log_2 P)$ steps per control move without subtraction cancellation (superseded by the $\mathcal{O}(1)$ `CP1` + `ExactIntFactorAccumulator` architecture supporting arbitrary $P \in [0, 65535]$).
