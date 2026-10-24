"""Multi-model DNAM project workspace, capture planner, and container runner
(`dnam_trainer.project`)."""

from __future__ import annotations

from dnam_trainer.project.planner import (
    PlannedStemCapture,
    compute_control_deltas,
    plan_model_capture_schedule,
)
from dnam_trainer.project.runner import (
    ContainerInspectionSummary,
    inspect_dnam_container,
    render_with_dnam_engine,
)
from dnam_trainer.project.workspace import (
    DnamProject,
    atomic_write_toml,
    default_user_config_dir,
    dumps_toml,
    ensure_creator_ed25519_keypair,
    parse_compact_range_spec,
    parse_compact_switch_spec,
)

__all__ = [
    "ContainerInspectionSummary",
    "DnamProject",
    "PlannedStemCapture",
    "atomic_write_toml",
    "compute_control_deltas",
    "default_user_config_dir",
    "dumps_toml",
    "ensure_creator_ed25519_keypair",
    "inspect_dnam_container",
    "parse_compact_range_spec",
    "parse_compact_switch_spec",
    "plan_model_capture_schedule",
    "render_with_dnam_engine",
]
