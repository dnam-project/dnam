# `src/stages` — TORA Multi-Stage Rig Topology & Matrix Router

Implements the 4-stage rig pipeline and matrix routing core of TORA (`#[forbid(unsafe_code)]`):
- **Stage 0:** 8-Port Matrix Router and return alignment.
- **Stage I:** Pre-Dynamics & input shaping (Op-T + Op-N).
- **Stage II:** Commuted Nonlinear Cascade with 4-basis decomposition (Op-N + Op-M).
- **Stage III:** Kautz Modal Cabinets & acoustic dispersion (Op-M).
- **Stage IV:** Spatial Diffusive Tank & stereo reverb/delay (Op-T + Op-M).

## 1. Submodule Map

| Stage / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`router.rs`](router.rs) | 8-Port Matrix Router, return aligners, and current clean crossover | [`router_and_stages_topology.md`](../../../../docs/specs/dsp/router_and_stages_topology.md) |
| [`stage1_pre.rs`](stage1_pre.rs) | Stage I Pre-Dynamics and envelope-governed input conditioning | [`router_and_stages_topology.md`](../../../../docs/specs/dsp/router_and_stages_topology.md) |
| [`stage2_cascade.rs`](stage2_cascade.rs) | Stage II Commuted 3-stage nonlinear cascade and 4-filter tone basis | [`op_n_commuted_cascade.md`](../../../../docs/specs/dsp/op_n_commuted_cascade.md) |
| [`stage3_cabinet.rs`](stage3_cabinet.rs) | Stage III early FIR and modal cabinet synthesis | [`op_m_modal_kautz.md`](../../../../docs/specs/dsp/op_m_modal_kautz.md) |
| [`stage4_tank.rs`](stage4_tank.rs) | Stage IV Unitary Spatial Tank (`Givens+Householder`, diffusers, DC blocker & loop saturator) | [`router_and_stages_topology.md`](../../../../docs/specs/dsp/router_and_stages_topology.md) |
| [`head.rs`](head.rs) | Unified 256-D Feature Bus & Canonical Mid/Side Kronecker Output Head | [`router_and_stages_topology.md`](../../../../docs/specs/dsp/router_and_stages_topology.md) |
| [`weights.rs`](weights.rs) | 64-byte cache-line aligned `Arena 0..3` weight directory view | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for stage chaining and memory layout.
