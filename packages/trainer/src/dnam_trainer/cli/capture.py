"""CLI commands for hardware/software audio capture (`dnam capture backends`, `devices`,
`generate`, `calibrate`, `plan`, `record` / `reamp`)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cyclopts
from rich.console import Console
from rich.table import Table

from dnam_engine import DnamContainer
from dnam_trainer.capture.backends import (
    get_capture_backend,
    list_capture_backends,
    record_single_stem,
)
from dnam_trainer.capture.excitation import generate_dry_excitation, read_wav
from dnam_trainer.cli.console import (
    CliCaptureObserver,
    prompt_calibration_step1,
    prompt_calibration_step2,
    prompt_next_stem_action,
    render_capture_plan_table,
)
from dnam_trainer.project.planner import plan_model_capture_schedule
from dnam_trainer.project.workspace import DnamProject
from dnam_trainer.schema import AuditReport, CaptureBackendConfig, ControlCoords

capture_app = cyclopts.App(
    name="capture",
    help="Hardware & software audio capture, 2-step unity calibration, and stem recording.",
)


@capture_app.command(name="backends")
def capture_backends() -> None:
    """List all registered hardware and software capture backends."""
    con = Console()
    table = Table(title="Registered Capture Backends", header_style="bold cyan")
    table.add_column("Backend ID", style="bold")
    table.add_column("Display Name")
    table.add_column("Description")
    for b_id, name, desc in list_capture_backends():
        table.add_row(b_id, name, desc)
    con.print(table)


@capture_app.command(name="devices")
def capture_devices(
    *,
    backend: Literal["anagram", "usb", "software"] = "software",
) -> None:
    """List audio devices discovered by `backend`."""
    con = Console()
    cap = get_capture_backend(CaptureBackendConfig(backend_type=backend))
    devs = cap.list_devices()
    table = Table(title=f"Audio Devices ({backend})", header_style="bold cyan")
    table.add_column("Index", justify="right")
    table.add_column("Name", style="bold")
    table.add_column("Max In", justify="right")
    table.add_column("Max Out", justify="right")
    table.add_column("Default SR", justify="right")
    for d in devs:
        table.add_row(
            str(d.index),
            d.name,
            str(d.max_input_channels),
            str(d.max_output_channels),
            f"{d.default_samplerate:.0f} Hz",
        )
    con.print(table)


@capture_app.command(name="generate")
def capture_generate(
    output_wav: Path = Path("audio/input.wav"),
    *,
    duration_sec: float = 4.0,
    sample_rate_hz: int = 48_000,
    seed: int = 42,
) -> None:
    """Synthesize the 24-bit system-identification dry excitation WAV (`audio/input.wav`)."""
    generate_dry_excitation(
        output_wav,
        duration_sec=duration_sec,
        sample_rate_hz=sample_rate_hz,
        seed=seed,
    )
    Console().print(
        f"[green]✓ Generated 24-bit dry excitation WAV[/green] at [bold]{output_wav}[/bold] "
        f"({duration_sec:.1f}s @ {sample_rate_hz} Hz)."
    )


@capture_app.command(name="calibrate")
def capture_calibrate(
    *,
    backend: Literal["anagram", "usb", "software"] | None = None,
    send_gain_db: float | None = None,
    yes: bool = False,
    project_dir: Path = Path("."),
) -> None:
    """Run 2-Step Reference Unity & Hot-Corner Calibration and persist to `dnam.toml`."""
    proj = DnamProject.load(project_dir)
    cfg = proj.manifest.capture
    if backend is not None:
        cfg = cfg.model_copy(update={"backend_type": backend})
    if send_gain_db is not None:
        cfg = cfg.model_copy(update={"send_gain_db": float(send_gain_db)})

    cap_backend = get_capture_backend(cfg)
    con = Console()
    obs = CliCaptureObserver(console=con)
    auto_yes = yes or cfg.backend_type == "software"

    res = cap_backend.calibrate_reference_and_hot_corner(
        observer=obs,
        step1_prompt_fn=lambda: prompt_calibration_step1(
            cap_backend.setup_instructions(),
            auto_yes=auto_yes,
            console=con,
        ),
        step2_prompt_fn=lambda ev1: prompt_calibration_step2(
            ev1,
            auto_yes=auto_yes,
            console=con,
        ),
        output_dry_wav_path=proj.dry_input_wav_path,
    )

    if res.hot_corner_clipped_fraction > 0.001 and cfg.backend_type != "anagram":
        clip_pct = res.hot_corner_clipped_fraction * 100.0
        con.print(
            f"[yellow]⚠ Hot-corner return clipped ({clip_pct:.2f}% samples >= -0.05 dBFS). "
            f"If your audio interface A/D input is clipping, lower the interface hardware "
            f"Input Trim/Pad (keep DNAM Send Gain locked at {res.send_gain_db:+.1f} dB).[/yellow]"
        )

    con.print(
        f"[green]✓ Synthesized probe-optimized dry excitation[/green] at "
        f"[bold]{proj.dry_input_wav_path}[/bold]."
    )

    updated_cap = cfg.model_copy(
        update={
            "send_gain_db": res.send_gain_db,
            "unity_ref_dbfs": res.unity_ref_dbfs,
            "calibrated_latency_samples": res.calibrated_latency_samples,
            "baseline_noise_floor_dbfs": res.baseline_noise_floor_dbfs,
            "excitation_profile": res.excitation_profile,
        }
    )
    proj.manifest = proj.manifest.model_copy(update={"capture": updated_cap})
    proj.save_manifest()


@capture_app.command(name="plan")
def capture_plan(
    model_id: str,
    *,
    include_corners: bool = True,
    project_dir: Path = Path("."),
) -> None:
    """Display the Star + Switch + Extreme Corner + `.dnam-ckpt` Prescription capture plan for
    `model_id`."""
    proj = DnamProject.load(project_dir)
    spec = proj.get_model(model_id)
    ckpt_audit: AuditReport | None = None
    ckpt_path = proj.model_checkpoint_path(model_id)
    if ckpt_path.is_file():
        ckpt_audit = DnamContainer.from_file(ckpt_path).header.audit_report

    plan = plan_model_capture_schedule(
        spec,
        ckpt_audit_report=ckpt_audit,
        include_corners=include_corners,
    )
    render_capture_plan_table(model_id, plan)


@capture_app.command(name=["record", "reamp"])
def capture_record(
    model_id: str,
    *,
    backend: Literal["anagram", "usb", "software"] | None = None,
    yes: bool = False,
    countdown_sec: float = 0.0,
    include_corners: bool = True,
    only_unrecorded: bool = True,
    project_dir: Path = Path("."),
) -> None:
    """Execute the interactive (or automated `--yes`) stem capture schedule for `model_id`."""
    proj = DnamProject.load(project_dir)
    spec = proj.get_model(model_id)
    cap_cfg = spec.capture_config or proj.manifest.capture
    if backend is not None:
        cap_cfg = cap_cfg.model_copy(update={"backend_type": backend})

    cap_backend = get_capture_backend(cap_cfg)
    ckpt_audit: AuditReport | None = None
    ckpt_path = proj.model_checkpoint_path(model_id)
    if ckpt_path.is_file():
        ckpt_audit = DnamContainer.from_file(ckpt_path).header.audit_report

    full_plan = plan_model_capture_schedule(
        spec,
        ckpt_audit_report=ckpt_audit,
        include_corners=include_corners,
    )
    pending = [p for p in full_plan if not p.is_recorded] if only_unrecorded else full_plan

    con = Console()
    if not pending:
        con.print(
            f"[green]✓ All {len(full_plan)} planned stems for "
            f"'{model_id}' are already recorded.[/green]"
        )
        return

    obs = CliCaptureObserver(console=con)
    auto_yes = yes or cap_cfg.backend_type == "software"

    # Always ensure target probe calibration & probe-tailored dry stem synthesis have run
    if cap_cfg.excitation_profile is None:
        cal_res = cap_backend.calibrate_reference_and_hot_corner(
            observer=obs,
            step1_prompt_fn=lambda: prompt_calibration_step1(
                cap_backend.setup_instructions(),
                auto_yes=auto_yes,
                console=con,
            ),
            step2_prompt_fn=lambda ev1: prompt_calibration_step2(
                ev1,
                auto_yes=auto_yes,
                console=con,
            ),
            output_dry_wav_path=proj.dry_input_wav_path,
        )
        cap_cfg = cap_backend.config
        proj.manifest = proj.manifest.model_copy(update={"capture": cap_cfg})
        proj.save_manifest()
        _ = cal_res

    dry_audio, _ = read_wav(proj.dry_input_wav_path)

    prev_controls: ControlCoords | None = None
    idx = 0
    total = len(pending)
    while idx < total:
        planned = pending[idx]
        action = prompt_next_stem_action(
            planned,
            stem_index=idx + 1,
            total_stems=total,
            prev_controls=prev_controls,
            auto_yes=auto_yes,
            countdown_sec=countdown_sec,
            console=con,
        )
        if action == "skip":
            idx += 1
            continue
        if action == "retake" and idx > 0:
            idx -= 1
            continue

        wet_wav_path = proj.model_stems_dir(model_id) / f"{planned.stem_id}.wav"
        rec = record_single_stem(
            cap_backend,
            model_id=model_id,
            stem_id=planned.stem_id,
            dry_signal=dry_audio,
            output_wet_wav_path=wet_wav_path,
            relative_wet_wav=f"stems/{planned.stem_id}.wav",
            controls_meta=planned.controls,
            stem_index=idx + 1,
            total_stems=total,
            observer=obs,
        )
        proj.upsert_stem_entry(model_id, rec.entry)
        prev_controls = dict(planned.controls)
        idx += 1


def register_capture_commands(app: cyclopts.App) -> None:
    """Registers the `capture` sub-application on `app`."""
    app.command(capture_app)
