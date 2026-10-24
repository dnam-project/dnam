"""CLI `dnam train` command with smart `.dnam-ckpt` auto-resume, `--from-scratch`,
`--audit-only`, local/remote backend dispatch, and Altair/Parquet report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cyclopts
from rich.console import Console

from dnam_trainer.cli.console import CliTrainingObserver
from dnam_trainer.control.audit import format_consolidated_prescription_report
from dnam_trainer.project.workspace import DnamProject, ensure_creator_ed25519_keypair
from dnam_trainer.schema import (
    AcceleratorDevice,
    PrecisionPolicy,
    ResumeJobRequest,
    TopologyMode,
    TrainingBackendConfig,
    TrainingJobRequest,
    TrainingJobResult,
)
from dnam_trainer.training.backends import get_training_backend
from dnam_trainer.training.telemetry import (
    build_training_telemetry_df,
    save_telemetry_html_report,
    save_telemetry_parquet,
)


def train_project_model(
    proj: DnamProject,
    model_id: str,
    *,
    from_scratch: bool = False,
    audit_only: bool = False,
    force_tier2: bool = False,
    epochs: int | None = None,
    topology: TopologyMode | None = None,
    backend_name: Literal["local", "remote"] = "local",
    device: AcceleratorDevice = "auto",
    precision: PrecisionPolicy = "auto",
    unsigned: bool = False,
    console: Console | None = None,
) -> TrainingJobResult:
    """Trains or auto-resumes `model_id` inside `proj` and writes `.dnam`, `.dnam-ckpt`, and
    reports."""
    con = console or Console()
    spec = proj.get_model(model_id)
    stems = proj.load_training_stems(model_id)

    base_cfg = spec.training_config or proj.manifest.training
    backend_cfg = TrainingBackendConfig(
        backend_type=backend_name,
        device=device,
        precision=precision,
    )
    train_cfg = base_cfg.model_copy(
        update={
            "sample_rate_train_hz": proj.manifest.sample_rate_hz,
            "output_mode": spec.output_mode,
            "topology": topology if topology is not None else spec.topology,
            "epochs": epochs if epochs is not None else base_cfg.epochs,
            "backend": backend_cfg,
        }
    )

    creator_seed_hex: str | None = None
    if not unsigned:
        _, seed_bytes, _ = ensure_creator_ed25519_keypair(proj.manifest.signing_key_path)
        creator_seed_hex = seed_bytes.hex()

    observer = CliTrainingObserver(console=con)
    backend = get_training_backend(backend_name)
    ckpt_path = proj.model_checkpoint_path(model_id)

    if ckpt_path.is_file() and not from_scratch and not audit_only:
        con.print(
            f"[cyan]↻ Found existing checkpoint[/cyan] [bold]{ckpt_path.name}[/bold] — "
            f"running Two-Tier `.dnam-ckpt` resume (pass `--from-scratch` for full retrain)."
        )
        resume_req = ResumeJobRequest(
            checkpoint_bytes=ckpt_path.read_bytes(),
            new_stem_descriptors=[s.to_wire_descriptor() for s in stems],
            new_stems=stems,
            config=train_cfg,
            force_tier2=force_tier2,
        )
        result = backend.resume(
            resume_req,
            observer=observer,
            creator_seed_hex=creator_seed_hex,
        )
    else:
        job_req = TrainingJobRequest(
            model_id=spec.model_id,
            model_name=spec.name,
            author=proj.manifest.author,
            description=spec.description,
            controls_schema=list(spec.controls),
            presets=list(spec.presets),
            config=train_cfg,
            stem_descriptors=[s.to_wire_descriptor() for s in stems],
            stems=stems,
            user_metadata={},
        )
        result = backend.train(
            job_req,
            observer=observer,
            creator_seed_hex=creator_seed_hex,
            audit_only=audit_only,
        )

    # Write deployment .dnam to dist/<model_id>.dnam and checkpoint to
    # models/<model_id>/checkpoint.dnam-ckpt
    dist_path = proj.dist_dnam_path(model_id)
    dist_path.parent.mkdir(parents=True, exist_ok=True)
    dist_path.write_bytes(result.dnam_bytes)

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_bytes(result.checkpoint_bytes)

    # Write telemetry.parquet, telemetry.html, and audit_report.json
    reports_dir = proj.model_reports_dir(model_id)
    reports_dir.mkdir(parents=True, exist_ok=True)
    df = build_training_telemetry_df(result.telemetry)
    save_telemetry_parquet(df, reports_dir / "telemetry.parquet")
    save_telemetry_html_report(df, reports_dir / "telemetry.html", model_id=model_id)
    (reports_dir / "audit_report.json").write_text(
        result.audit_report.model_dump_json(indent=2),
        encoding="utf-8",
    )

    report_text = format_consolidated_prescription_report(
        str(dist_path),
        str(ckpt_path),
        result.audit_report,
        spec.controls,
    )
    con.print(report_text)
    return result


def register_train_command(app: cyclopts.App) -> None:
    """Registers the top-level `train` command on `app`."""

    @app.command(name="train")
    def cmd_train(
        model_id: str | None = None,
        *,
        all_models: bool = False,
        from_scratch: bool = False,
        audit_only: bool = False,
        force_tier2: bool = False,
        epochs: int | None = None,
        topology: TopologyMode | None = None,
        backend: Literal["local", "remote"] = "local",
        device: AcceleratorDevice = "auto",
        precision: PrecisionPolicy = "auto",
        unsigned: bool = False,
        project_dir: Path = Path("."),
    ) -> None:
        """Train or iteratively resume a model (`dist/<model_id>.dnam` + `checkpoint.dnam-ckpt`)."""
        proj = DnamProject.load(project_dir)
        con = Console()
        if all_models:
            target_ids = [m.model_id for m in proj.list_models() if m.stems]
        elif model_id is not None:
            target_ids = [model_id]
        else:
            models = proj.list_models()
            if len(models) == 1:
                target_ids = [models[0].model_id]
            else:
                raise ValueError("Specify `model_id` or pass `--all-models`")

        for mid in target_ids:
            train_project_model(
                proj,
                mid,
                from_scratch=from_scratch,
                audit_only=audit_only,
                force_tier2=force_tier2,
                epochs=epochs,
                topology=topology,
                backend_name=backend,
                device=device,
                precision=precision,
                unsigned=unsigned,
                console=con,
            )
