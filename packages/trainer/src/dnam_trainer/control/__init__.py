"""Parametric Control Bus & Non-Stop 3-Phase Control-Space Audit (`CL-12`..`CL-17`)."""

from __future__ import annotations

from dnam_trainer.control.audit import (
    Phase1PreflightResult,
    compute_phase2_corner_barrier_loss,
    format_consolidated_prescription_report,
    run_phase1_preflight_audit,
    run_phase2_rail_lockup_audit,
    run_phase3_export_certification,
)
from dnam_trainer.control.basis import (
    ParametricControlHyperBus,
    compute_lobatto_degree_and_anchors,
    compute_step0_variance_gates,
    eval_node_anchored_newton_basis,
    extract_blind_lfo_phase_trajectory,
    fit_order_statistic_delay_transport,
    normalize_and_clamp_range,
)

__all__ = [
    "ParametricControlHyperBus",
    "Phase1PreflightResult",
    "compute_lobatto_degree_and_anchors",
    "compute_phase2_corner_barrier_loss",
    "compute_step0_variance_gates",
    "eval_node_anchored_newton_basis",
    "extract_blind_lfo_phase_trajectory",
    "fit_order_statistic_delay_transport",
    "format_consolidated_prescription_report",
    "normalize_and_clamp_range",
    "run_phase1_preflight_audit",
    "run_phase2_rail_lockup_audit",
    "run_phase3_export_certification",
]
