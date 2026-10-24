"""`cyclopts` CLI application (`dnam` and `dnam-train`) for multi-model DNAM workspaces."""

from __future__ import annotations

from collections.abc import Sequence

import cyclopts

from dnam_trainer.cli.capture import register_capture_commands
from dnam_trainer.cli.train import register_train_command, train_project_model
from dnam_trainer.cli.workspace import (
    control_app,
    model_app,
    parse_key_value_controls,
    preset_app,
    register_workspace_commands,
    stem_app,
)

app = cyclopts.App(
    name="dnam",
    help=(
        "Dynamic Neural Audio Modeler (DNAM / Tri-Physics) — "
        "Multi-model project workspace, reamp capture, and 4-Stage neural trainer."
    ),
)

register_workspace_commands(app)
register_capture_commands(app)
register_train_command(app)


def main(argv: Sequence[str] | None = None) -> None:
    """Entrypoint for the `dnam` and `dnam-train` console scripts."""
    if argv is not None:
        app(list(argv), result_action="return_value")
    else:
        app()


__all__ = [
    "app",
    "control_app",
    "main",
    "model_app",
    "parse_key_value_controls",
    "preset_app",
    "stem_app",
    "train_project_model",
]
