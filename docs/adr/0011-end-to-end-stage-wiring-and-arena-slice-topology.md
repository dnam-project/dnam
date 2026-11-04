---
id: ADR-0011
title: "End-to-End Stage Wiring, 256-D Feature Bus Slot Map, Stage Modulation Laws & Arena Slice Directory"
status: accepted
date: "2026-11-04"
supersedes: null
superseded_by: null
specs:
  - "../specs/dsp/router_and_stages_topology.md"
  - "../specs/dsp/parametric_control_bus.md"
  - "../specs/containers/dnam_binary_format.md"
requirements:
  - REQ-1.5
  - REQ-5.1
  - REQ-5.2
  - REQ-6.1
  - REQ-7.4
---

# ADR-0011: End-to-End Stage Wiring, 256-D Feature Bus Slot Map, Stage Modulation Laws & Arena Slice Directory

This record resolves the end-to-end integration ("glue") specifications connecting the leaf TORA operators (`Op-T`, `Op-M`, `Op-N`), the $3\text{ kHz}$ parametric control bus, the 256-D 4-tier Kronecker output head, and the 64-byte-aligned `.dnam` tier arenas across [`../specs/dsp/router_and_stages_topology.md`](../specs/dsp/router_and_stages_topology.md), [`../specs/dsp/parametric_control_bus.md`](../specs/dsp/parametric_control_bus.md), and [`../specs/containers/dnam_binary_format.md`](../specs/containers/dnam_binary_format.md).

---

## 1. Context & Problem Statement

While implementing and verifying the Rust engine (`packages/engine/`) against the leaf operator, control-bus, container, and security specifications (`ADR-0002` through `ADR-0010`), four integration-boundary gaps surfaced between otherwise bit-exact leaf specifications:

1. **Unassigned `"Other-Stage Slots"` in the 256-D Feature Bus (`ref[dsp.head.feature-bus-slots]`):**
   `STAGE-III-TIER-SLOTS` assigned Stage III modal groups (`0..3`, `64..67`, `128..135`, `192..207`), Stage I `T1` sub-harmonic tracker features (`4..11`), and Stage III early FIR outputs (`62..63`, `126..127`, `190..191`, `254..255`), but left the remaining 208 slots (`12..61`, `68..125`, `136..189`, `208..253`) labeled only as `"Other-stage slots"` without a deterministic producer-to-slot map.
2. **Implicit Stage Modulation Laws for $\boldsymbol{\theta}_{\text{stage}}[n] \in \mathbb{R}^{16}$ (`ref[dsp.ctrl.dispatch.matrix]`, `ref[dsp.ctrl.dispatch.stage-laws]`):**
   The 16 rows of $\mathbf{W}_{\text{ctrl}}$ declared physical roles (`g_send,L/R`, `tau_pre`, `nu_pitch`, `pre_drive`, `v1a_drive`, `tone_bal`, `pwr_drive`, `sag_depth`, `kautz_freq`, `kautz_radial_gate`, `fir_gain`, `tau_tank`, `g_loop`, `a_diff`, `alpha_width`), but only $\theta_{10}$ (`kautz_radial_gate`) had an explicit algebraic mapping (`clamp(0.5 + theta_10, 0, 1)`).
3. **Prose-Only Inter-Substage Signal Flow in Stages I & II (`ref[dsp.stage1.pre-bus]`, `ref[dsp.stage2.commuted-cascade]`):**
   Unlike Stage III and Stage IV, Stages I and II lacked explicit per-sample equations specifying how the 4-bit `topology_selector` orders `Op-T[Pre]`, `Op-M[Pre]`, and `Op-N[Pre]`, and how Sub-Stage 2.1 (`V1A` + 4-filter tone basis), Sub-Stage 2.2 (`V1B` + degree-7/8 ADAA $C^2$ wavefolder), and Sub-Stage 2.3 (Power Amp + 1-pole B+ sag + 2D quadrature hysteresis) chain together.
4. **Incomplete `tensor_directory` Slice Registry Across `Arena 0..3` (`ref[container.dnam.tensor-directory]`):**
   `dnam_binary_format.md` defined exact 64B-aligned slices for `stage3_early_fir`, `stage3_kautz_modals`, and `CP1` control projections, but omitted canonical slice keys and shapes for Stages 0, I, II, IV, and the 256-D Kronecker head.

---

## 2. Considered Options & Rejected Candidates

| Candidate Approach | Why It Fails | Quantified Failure / Maintenance Penalty |
| :--- | :--- | :--- |
| **Ad-Hoc / Implementation-Defined Feature Bus Slot Ordering (`ref[dsp.head.feature-bus-slots]`)** | Allows `packages/trainer` and `packages/engine` to pack Stage I, II, and IV features in different slot orders within `12..61`, `68..125`, `136..189`, and `208..253`. | Breaks trainer-to-engine isomorphism (`REQ-4.1`, `REQ-5.1`) and corrupts Kronecker head dot-products (`W_head * Phi[n]`) even when all leaf operators match to 0 ULP. |
| **Unclamped Additive Dispatch of `theta_0..theta_15` (`ref[dsp.ctrl.dispatch.stage-laws]`)** | Applying raw micro-ramped offsets `theta_i[n]` without per-target domain clamps allows transient modulator overshoots to push `g_loop > 0.9990`, `\|a_diff\| > 0.75`, or `tau_pre < 4.0`. | Violates unconditional BIBO/Lyapunov stability (`REQ-7.2`) and the 4-sample Catmull-Rom causal margin (`ref[dsp.op-t.ring.capacity]`). |
| **Hardcoded Magic Byte Offsets Instead of 64B-Aligned `tensor_directory` Slices (`ref[container.dnam.tensor-directory]`)** | Couples engine loaders to fixed prototype offsets (e.g. `1024`, `4160`, `4928`) that cannot accommodate variable contributor counts in `CP1`. | Prevents zero-copy validation of model-declared stage slices (`ref[container.dnam.load-validation]`). |

---

## 3. Decision Outcome

We codify four normative integration specifications across `router_and_stages_topology.md`, `parametric_control_bus.md`, and `dnam_binary_format.md`:

1. **Deterministic 256-D Feature Bus Slot Map (`ref[dsp.head.feature-bus-slots]`):**
   Every slot across `Lite` (`0..63`), `Full` (`64..127`), `Pro` (`128..191`), and `Ultra` (`192..255`) is assigned to a fixed stage producer (`Stage III Kautz`, `Stage I T1`, `Stage I Pre-FX`, `Stage II V1A + Tone Basis`, `Stage II V1B + Wavefolder + Power/Sag/Hyst`, `Stage IV Tank + Passthrough`, `Stage 0 Clean Sub`, and `Stage III Early FIR`), guaranteeing bit-identical feature-bus ordering between trainer and engine.
2. **Closed-Form Per-Sample Stage Modulation Laws (`ref[dsp.ctrl.dispatch.stage-laws]`):**
   Each micro-ramped parameter $\theta_0[n] \dots \theta_{15}[n]$ maps to its target stage parameter through an explicit division-free clamping law enforcing all registered stability bounds (`tau >= 4.0`, `p in [-24, 24]` st, `g_kautz in [0, 1]`, `g_lin <= 0x3F7FBE76`, `|a_diff| <= 0.75`, `alpha_width in [0, 2]`).
3. **Explicit Stage I & Stage II Sample-by-Sample Recurrences (`ref[dsp.stage1.pre-bus]`, `ref[dsp.stage2.commuted-cascade]`):**
   Stage I executes the 4-bit serial permutation selector (`N->M->T`, `M->N->T`, `T->N->M`, `T->M->N`) alongside the parallel `T1` sub-harmonic tracker (`slots 4..11`); Stage II executes the commuted `V1A -> 4-Filter Tone Stack -> V1B + Degree-7/8 ADAA C² Wavefolder -> Power Amp + 1-Pole Sag + 2D Hysteresis` chain.
4. **Canonical 64B-Aligned `tensor_directory` Slice Table (`ref[container.dnam.tensor-directory]`):**
   Each tier arena (`Arena 0..3`) organizes its stage and head tensors into named, 64-byte-aligned slices relative to `tier_start` (`stage0_router`, `stage1_pre`, `stage2_commuted`, `stage3_early_fir`, `stage3_kautz_modals`, `stage4_tank`, `head_kronecker`, plus the `CP1` control slices in `Arena 0` and `Arena 3`).
5. **Runtime Preset/Bypass Transitions, Over-Envelope Telemetry, Captured-Freeze Gating & Tempo-Clamp Indicators:**
   - **Preset / Bypass Transitions (`ref[dsp.ctrl.switch.conditional-batch]`, `ref[dsp.ctrl.dispatch.micro-ramp]`, `ref[dsp.op-t.transition.dual-mode]`, `ref[arch.topology.spillover-trails]`):** Loading a partial preset restores signed defaults for omitted controls, applies `switch` controls with $L_{\text{xfade}} = \max(16, \mathrm{round}(128 / \rho_{\text{sr}}))$ delay-pointer crossfades (using a two-path output crossfade only on disjoint-interval conditional switch transitions), and slews continuous `range` controls at the $3\text{ kHz}$ anti-zipper rate + per-sample micro-ramp. Bypass transitions between `Active` and `SpilloverTrails` (`trails = true`, muting tank injection while decaying tank state alongside dry passthrough) or `HardBypass` (`trails = false`).
   - **Over-Envelope Input Telemetry (`ref[arch.fp.range-saturation]`, `ref[train.audit.phase3.numeric-envelope]`):** Inputs exceeding the certified $\ge \pm 100.0$ (`+40 dBFS`) envelope set `OUTSIDE_CERTIFIED_INPUT` (`bit 4` of `overflow_or_saturation_flags`) for the host/Studio clip LED without engaging an unmodeled soft input limiter; hard clamping occurs only at the representational safety guards (`±65,504.0` `FP16_SATURATED` and `±1e9` `INPUT/INTERNAL/OUTPUT_SATURATED`).
   - **Captured-Only Hardware Freeze (`ref[dsp.stage4.captured-freeze]`, `ref[train.audit.phase1.clamp-switch]`):** When `supports_freeze == false` (and no control declares `role == "freeze_latch"`), the host hides/disables the freeze footswitch and the engine ignores synthetic freeze requests.
   - **Tempo-Sync Out-of-Range Subdivision Clamping (`ref[dsp.ctrl.range.f32-rank-bounds]`, `ref[dsp.tempo.physical-inversion]`):** When a tempo-synced target falls outside the active option's calibrated `[v_min, v_max]` (after optional octave folding), the engine clamps to the nearest reachable `f32` physical rank bound (`L` or `U - 1`) and sets the control's bit in `active_clamp_mask` to surface a UI clamp indicator badge.
