"""File-backed JSON job worker (`run_worker_job`) and hardware status probe
(`get_trainer_status_payload`) for `dnam-train`."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

import torch
from rich.console import Console

from dnam_trainer.cli.observers import EventFormat, create_training_observer
from dnam_trainer.control.audit import format_consolidated_prescription_report
from dnam_trainer.schema import (
    AcceleratorDevice,
    PrecisionPolicy,
    ResumeJobRequest,
    TopologyMode,
    TrainingJobRequest,
    TrainingJobResult,
)
from dnam_trainer.training.backends import get_training_backend
from dnam_trainer.training.hardware import resolve_hardware_profile
from dnam_trainer.training.telemetry import (
    build_training_telemetry_df,
    save_telemetry_html_report,
    save_telemetry_parquet,
)


def get_trainer_status_payload(
    *,
    device: AcceleratorDevice = "auto",
    precision: PrecisionPolicy = "auto",
) -> dict[str, Any]:
    """Returns a JSON-serializable dict describing the Python trainer environment & hardware."""
    from dnam_trainer import __version__

    prof = resolve_hardware_profile(device, precision=precision)
    return {
        "installed": True,
        "trainer_version": __version__,
        "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__),
        "resolved_device": prof.device_type,
        "torch_device": str(prof.torch_device),
        "contraction_dtype": str(prof.contraction_dtype),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    }


def _resolve_creator_seed_hex(explicit_seed_hex: str | None, unsigned: bool) -> str | None:
    if unsigned:
        return None
    if explicit_seed_hex:
        return explicit_seed_hex.strip()
    env_seed = os.environ.get("DNAM_CREATOR_SEED_HEX", "").strip()
    if env_seed:
        return env_seed
    env_key_path = os.environ.get("DNAM_SIGNING_KEY", "").strip()
    if env_key_path and Path(env_key_path).is_file():
        return Path(env_key_path).read_text(encoding="utf-8").strip()
    return None


def run_worker_job(
    request_json: Path,
    *,
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
    backend_name: Literal["local", "remote"] = "local",
    device: AcceleratorDevice | None = None,
    precision: PrecisionPolicy | None = None,
    unsigned: bool = False,
    creator_seed_hex: str | None = None,
    event_format: EventFormat = "jsonl",
    console: Console | None = None,
) -> TrainingJobResult:
    """Executes a file-backed `TrainingJobRequest` or `ResumeJobRequest` and writes outputs."""
    raw_text = Path(request_json).read_text(encoding="utf-8")
    raw_obj = json.loads(raw_text)

    observer = create_training_observer(event_format, console=console)
    backend = get_training_backend(backend_name)
    resolved_seed = _resolve_creator_seed_hex(creator_seed_hex, unsigned)

    is_resume_payload = "checkpoint_bytes" in raw_obj or "new_stem_descriptors" in raw_obj
    has_ckpt_file = (
        checkpoint_in is not None
        and Path(checkpoint_in).is_file()
        and not from_scratch
        and not audit_only
    )

    if (is_resume_payload or has_ckpt_file) and not from_scratch and not audit_only:
        if is_resume_payload:
            resume_req = ResumeJobRequest.model_validate(raw_obj)
        else:
            train_req = TrainingJobRequest.model_validate(raw_obj)
            assert checkpoint_in is not None
            resume_req = ResumeJobRequest(
                checkpoint_bytes=Path(checkpoint_in).read_bytes(),
                new_stem_descriptors=list(train_req.stem_descriptors),
                config=train_req.config,
                force_tier2=force_tier2,
            )

        cfg = resume_req.config
        if epochs is not None:
            cfg = cfg.model_copy(update={"epochs": epochs})
        if topology is not None:
            cfg = cfg.model_copy(update={"topology": topology})
        if device is not None or precision is not None:
            b_cfg = cfg.backend.model_copy(
                update={
                    "device": device if device is not None else cfg.backend.device,
                    "precision": precision if precision is not None else cfg.backend.precision,
                }
            )
            cfg = cfg.model_copy(update={"backend": b_cfg})
        resume_req = resume_req.model_copy(
            update={"config": cfg, "force_tier2": force_tier2 or resume_req.force_tier2}
        )
        result = backend.resume(
            resume_req,
            observer=observer,
            creator_seed_hex=resolved_seed,
        )
        controls_for_report = (
            TrainingJobRequest.model_validate(raw_obj).controls_schema
            if "controls_schema" in raw_obj
            else []
        )
    else:
        job_req = TrainingJobRequest.model_validate(raw_obj)
        cfg = job_req.config
        if epochs is not None:
            cfg = cfg.model_copy(update={"epochs": epochs})
        if topology is not None:
            cfg = cfg.model_copy(update={"topology": topology})
        if device is not None or precision is not None:
            b_cfg = cfg.backend.model_copy(
                update={
                    "device": device if device is not None else cfg.backend.device,
                    "precision": precision if precision is not None else cfg.backend.precision,
                }
            )
            cfg = cfg.model_copy(update={"backend": b_cfg})
        job_req = job_req.model_copy(update={"config": cfg})
        result = backend.train(
            job_req,
            observer=observer,
            creator_seed_hex=resolved_seed,
            audit_only=audit_only,
        )
        controls_for_report = list(job_req.controls_schema)

    if output_dnam is not None:
        out_d = Path(output_dnam)
        out_d.parent.mkdir(parents=True, exist_ok=True)
        out_d.write_bytes(result.dnam_bytes)

    if output_ckpt is not None:
        out_c = Path(output_ckpt)
        out_c.parent.mkdir(parents=True, exist_ok=True)
        out_c.write_bytes(result.checkpoint_bytes)

    if reports_dir is not None:
        rep_dir = Path(reports_dir)
        rep_dir.mkdir(parents=True, exist_ok=True)
        df = build_training_telemetry_df(result.telemetry)
        save_telemetry_parquet(df, rep_dir / "telemetry.parquet")
        save_telemetry_html_report(df, rep_dir / "telemetry.html", model_id=result.model_id)
        (rep_dir / "audit_report.json").write_text(
            result.audit_report.model_dump_json(indent=2),
            encoding="utf-8",
        )

    if result_json is not None:
        res_p = Path(result_json)
        res_p.parent.mkdir(parents=True, exist_ok=True)
        summary_dict = {
            "model_id": result.model_id,
            "esr_lite_db": result.esr_lite_db,
            "esr_full_db": result.esr_full_db,
            "esr_ultra_db": result.esr_ultra_db,
            "resume_tier_executed": result.resume_tier_executed,
            "core2_active": result.core2_active,
            "lfo_env_active": result.lfo_env_active,
            "stem_esr_ultra_db": result.stem_esr_ultra_db,
            "audit_report": result.audit_report.model_dump(mode="json"),
        }
        res_p.write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")

    if event_format == "rich":
        con = console or Console()
        report_text = format_consolidated_prescription_report(
            str(output_dnam or "dist/model.dnam"),
            str(output_ckpt or "checkpoint.dnam-ckpt"),
            result.audit_report,
            controls_for_report,
        )
        con.print(report_text)

    return result


__all__ = [
    "get_trainer_status_payload",
    "run_worker_job",
]
