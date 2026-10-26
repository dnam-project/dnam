# `dnam_trainer/control/` — Parametric Control Bus & Non-Stop 3-Phase Audit (`CL-12`–`CL-17`)

**`dnam_trainer.control`** implements the Node-Anchored Lobatto HDMR + Rank-4 Factorization Machine (`FM`) Control Bus, Step-0 Order-Statistic Optimal-Transport delay matching, Step-0 Physical Variance Gating, and the Non-Stop 3-Phase Control-Space Audit pipeline.

## 1. Module Routing Map

| Module | Enforced Clamps | Key Exports & Responsibilities | Canonical Spec (SSOT) |
| :--- | :--- | :--- | :--- |
| [`basis.py`](basis.py) | `CL-12`, `CL-13`, `CL-14` | `normalize_and_clamp_range`, `compute_lobatto_degree_and_anchors`, `eval_node_anchored_newton_basis`, `fit_order_statistic_delay_transport`, `compute_step0_variance_gates`, `ParametricControlHyperBus` | [`parametric_control_bus.md`](../../../../docs/parametric_control_bus.md) |
| [`audit.py`](audit.py) | `CL-15`, `CL-16`, `CL-17` | `run_phase1_preflight_audit`, `compute_phase2_corner_barrier_loss`, `run_phase2_rail_lockup_audit`, `run_phase3_export_certification`, `format_consolidated_prescription_report` | [`clamp_and_audit_registry.md`](../../../../docs/clamp_and_audit_registry.md) |

See [`packages/trainer/CONTRIBUTING.md`](../../../CONTRIBUTING.md) for `CL-13` and `CL-16` implementation invariants.
