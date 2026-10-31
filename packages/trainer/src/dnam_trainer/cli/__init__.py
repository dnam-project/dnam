"""Headless `dnam-train` worker CLI invoked by `packages/studio` (`dnam train` & Tauri Studio)."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import cyclopts

from dnam_trainer.cli.observers import (
    EventFormat,
    JsonlTrainingObserver,
    RichTrainingObserver,
    SseTrainingObserver,
    create_training_observer,
)
from dnam_trainer.cli.worker import get_trainer_status_payload, run_worker_job
from dnam_trainer.schema import AcceleratorDevice, PrecisionPolicy, TopologyMode

app = cyclopts.App(
    name="dnam-train",
    help=(
        "Dynamic Neural Audio Modeler (DNAM / Tri-Physics) — "
        "Headless 4-Stage PyTorch Trainer & `.dnam-ckpt` Kronecker Worker."
    ),
)


@app.default
def train_cmd(
    *,
    request_json: Path | None = None,
    checkpoint_in: Path | None = None,
    output_dnam: Path | None = None,
    output_ckpt: Path | None = None,
    reports_dir: Path | None = None,
    result_json: Path | None = None,
    from_scratch: bool = False,
    audit_only: bool = False,
    force_tier2: bool = False,
    epochs: int | None = None,
    topology: TopologyMode | None = None,
    backend: Literal["local", "remote"] = "local",
    device: AcceleratorDevice | None = None,
    precision: PrecisionPolicy | None = None,
    unsigned: bool = False,
    creator_seed_hex: str | None = None,
    no_cache: bool = False,
    cache_max_gb: float | None = None,
    event_format: EventFormat = "jsonl",
    status_json: bool = False,
) -> None:
    """Execute a file-backed training/resume job or emit environment status JSON."""
    if status_json:
        payload = get_trainer_status_payload(
            device=device or "auto",
            precision=precision or "auto",
        )
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()
        return

    if request_json is None:
        raise ValueError("Provide `--request-json <path>` or `--status-json`")

    run_worker_job(
        request_json,
        checkpoint_in=checkpoint_in,
        output_dnam=output_dnam,
        output_ckpt=output_ckpt,
        reports_dir=reports_dir,
        result_json=result_json,
        from_scratch=from_scratch,
        audit_only=audit_only,
        force_tier2=force_tier2,
        epochs=epochs,
        topology=topology,
        backend_name=backend,
        device=device,
        precision=precision,
        unsigned=unsigned,
        creator_seed_hex=creator_seed_hex,
        no_cache=no_cache,
        cache_max_gb=cache_max_gb,
        event_format=event_format,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Entrypoint for the `dnam-train` console script."""
    if argv is not None:
        app(list(argv), result_action="return_value")
    else:
        if moon_cwd := os.environ.get("MOON_WORKING_DIR"):
            os.chdir(moon_cwd)
        app()


__all__ = [
    "EventFormat",
    "JsonlTrainingObserver",
    "RichTrainingObserver",
    "SseTrainingObserver",
    "app",
    "create_training_observer",
    "get_trainer_status_payload",
    "main",
    "run_worker_job",
]
