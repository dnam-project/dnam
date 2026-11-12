---
id: template.spec
title: "Normative Specification Template (docs/specs/<domain>/)"
status: template
version: "1.0.0"
domain: example
rule_prefixes:
  - "r[example."
requirements:
  - REQ-1.1
adrs:
  - ADR-0002
---

# Normative Specification Template (`docs/specs/<domain>/`)

Use this template when authoring or extending a normative technical specification under `docs/specs/`. Every normative specification file must start with YAML frontmatter declaring its `id`, `title`, `status`, `version`, `domain`, `rule_prefixes`, `requirements`, and active `adrs` (never superseded ADRs), and every normative rule must carry a unique Rust Reference-style `r[<domain>.<subdomain>.<rule>]` identifier so implementation code and tests can cite an exact specification rule. See [`../CONTRIBUTING.md`](../CONTRIBUTING.md) for the complete authoring guide.

This specification defines **how** the `<Subsystem Name>` operates (`packages/engine/` and `packages/capture/`, fulfilling [`REQ-1.1`](../requirements/system_goals_and_requirements.md)). Design rationales, rejected R&D alternatives, and counterexamples belong in [`../adr/0002-tri-operator-recurrent-architecture.md`](../adr/0002-tri-operator-recurrent-architecture.md)—never as narrative essays inside `docs/specs/`.

---

## 1. `<Subsystem Overview & Signal Flow>`

`r[example.subsystem.signal-flow]`

```mermaid
flowchart LR
    In["Input Signal x[n]"] --> Kernel["Zero-FDIV SIMD Kernel"] --> Out["Output y[n]"]
```

```text
+---------------------------------------------------------------------------------------------------+
| 64-BYTE CACHE-LINE STRUCT OR ARENA LAYOUT                                                         |
+---------------------------------------------------------------------------------------------------+
|  [0..15]:  f32x4 state_lane_0  |  [16..31]: f32x4 state_lane_1  |  [32..63]: 32 B alignment pad   |
+---------------------------------------------------------------------------------------------------+
```

### 1.1 `<Section-Level Normative Rule>`

`r[example.subsystem.recurrence]`

State the exact mathematical recurrence, bit-level wire layout, or state transition law:
$$y[n] = \alpha x[n] + (1 - \alpha) y[n-1], \qquad \alpha \in [0, 1]$$

1. `r[example.subsystem.cold-path-prep]` **Cold-Path Coefficient Preparation:** Specify exact rounding modes (`RNE`), bit widths (`f32`, `f64`, `u128`), and admission checks executed outside the real-time loop.
2. `r[example.subsystem.realtime-loop]` **Real-Time Loop Execution (`0 FDIV`, `0 FSQRT`, `0 libm`):** Specify the branchless per-sample (`48 kHz`) or control-tick (`3 kHz`) operations and clamped state bounds.

| Parameter / Field | Type & Domain | Normative Invariant |
| :--- | :---: | :--- |
| `alpha` | `f32 ∈ [0.0, 1.0]` | Precomputed on cold path; zero real-time division (`ref[repo.docs.markdown-table-math]`: no raw dollar signs in tables). |
