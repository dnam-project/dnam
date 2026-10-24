"""Rich console observers (`CliCaptureObserver`, `CliTrainingObserver`), interactive
hardware capture prompts (`prompt_next_stem_action`), and structured table formatters."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Literal

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from dnam_trainer.events import (
    AuditPhaseEvent,
    CalibrationStepEvent,
    CaptureEvent,
    StemCaptureEvent,
    TrainingCompleteEvent,
    TrainingEvent,
    TrainingStageEvent,
    TrainingStepEvent,
)
from dnam_trainer.project.planner import PlannedStemCapture, compute_control_deltas
from dnam_trainer.schema import (
    ControlCoords,
    ControlKind,
    ControlSchemaDef,
    PresetDef,
    ProjectModelSpec,
    ProjectStemEntry,
)

StemPromptAction = Literal["record", "retake", "skip"]


class CliCaptureObserver:
    """Rich terminal observer for hardware/software calibration and stem capture events."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def on_capture_event(self, event: CaptureEvent) -> None:
        if isinstance(event, CalibrationStepEvent):
            if event.step == 1:
                self.console.print(
                    f"[bold cyan][Calibrate Step 1/2 — Bypass Unity][/bold cyan] "
                    f"backend={event.backend_id} | send={event.send_gain_db:+.1f} dB | "
                    f"unity_ref={event.unity_ref_dbfs:+.2f} dBFS | "
                    f"latency={event.calibrated_latency_samples} samples | "
                    f"noise={event.noise_floor_dbfs:.1f} dBFS"
                )
            else:
                prof_str = ""
                if event.excitation_profile is not None:
                    ep = event.excitation_profile
                    tail_tag = (
                        f"Tail({ep.estimated_max_delay_ms:.0f}ms)"
                        if ep.has_delay_or_reverb_tail
                        else "NoTail"
                    )
                    prof_str = (
                        f" | knee={ep.knee_amplitude_lin:.2f} | sat={ep.saturation_ceiling_lin:.2f}"
                        f" | {tail_tag} | res={ep.dominant_resonance_hz:.0f}Hz"
                    )
                self.console.print(
                    f"[bold green][Calibrate Step 2/2 — Engaged Hot Corner & Probe][/bold green] "
                    f"send={event.send_gain_db:+.1f} dB (locked) | "
                    f"net_gain={event.net_gain_over_unity_db:+.2f} dB | "
                    f"comp_ratio={event.compression_ratio:.2f}:1 | "
                    f"clipped={event.clipped_sample_fraction * 100.0:.2f}%{prof_str}"
                )
        elif isinstance(event, StemCaptureEvent):
            ctrl_str = ", ".join(f"{k}={v}" for k, v in event.controls.items()) or "default"
            if event.status == "starting":
                self.console.print(
                    f"[cyan]▶ Recording [{event.stem_index}/{event.total_stems}][/cyan] "
                    f"[bold]{event.stem_id}[/bold] ({ctrl_str})..."
                )
            elif event.status == "completed":
                self.console.print(
                    f"[green]✓ Captured [{event.stem_index}/{event.total_stems}][/green] "
                    f"[bold]{event.stem_id}[/bold] | peak={event.measured_peak_dbfs:+.2f} dBFS | "
                    f"noise={event.noise_floor_dbfs:.1f} dBFS | "
                    f"latency={event.latency_offset_samples} samples"
                )
            elif event.status == "skipped":
                self.console.print(
                    f"[yellow]↷ Skipped [{event.stem_index}/{event.total_stems}][/yellow] "
                    f"[bold]{event.stem_id}[/bold]"
                )


class CliTrainingObserver:
    """Rich terminal observer for 4-Stage DNAM training and 3-Phase control-space audits."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def on_training_event(self, event: TrainingEvent) -> None:
        if isinstance(event, TrainingStageEvent):
            self.console.print(
                f"[bold magenta][Stage {event.stage}/4 — {event.stage_name}][/bold magenta] "
                f"{event.message}"
            )
        elif isinstance(event, TrainingStepEvent):
            self.console.print(
                f"  [dim]Epoch {event.epoch:>3}/{event.total_epochs}[/dim] | "
                f"Lite={event.esr_lite_db:+.2f} dB | "
                f"Full={event.esr_full_db:+.2f} dB | "
                f"[bold]Ultra={event.esr_ultra_db:+.2f} dB[/bold] | "
                f"barrier={event.barrier_loss:.4f} | t={event.elapsed_sec:.2f}s"
            )
        elif isinstance(event, AuditPhaseEvent):
            rep = event.audit_report
            clamps = (
                rep.phase1_clamps
                if event.phase == 1
                else (rep.phase2_rail_warnings if event.phase == 2 else rep.phase3_clamps)
            )
            if clamps:
                for c_msg in clamps:
                    self.console.print(f"  [yellow]⚠ {c_msg}[/yellow]")
        elif isinstance(event, TrainingCompleteEvent):
            mode_label = (
                f"Resume Tier {event.resume_tier_executed}"
                if event.resume_tier_executed > 0
                else "Full 4-Stage Train"
            )
            c2_state = "ON" if event.core2_active else "BYPASS"
            self.console.print(
                f"[bold green]✓ {mode_label} Complete ({event.model_id})[/bold green] | "
                f"Lite={event.esr_lite_db:+.2f} dB | Full={event.esr_full_db:+.2f} dB | "
                f"Ultra={event.esr_ultra_db:+.2f} dB | Core2={c2_state} | "
                f"{event.elapsed_sec:.2f}s"
            )


def prompt_next_stem_action(
    planned: PlannedStemCapture,
    *,
    stem_index: int,
    total_stems: int,
    prev_controls: ControlCoords | None = None,
    auto_yes: bool = False,
    countdown_sec: float = 0.0,
    console: Console | None = None,
    input_fn: Callable[[str], str] | None = None,
) -> StemPromptAction:
    """Renders a Rich knob/switch delta panel for `planned` and prompts for `[Enter]`/`r`/`s`."""
    con = console or Console()
    deltas = compute_control_deltas(prev_controls, planned.controls)

    delta_table = Table(show_header=True, header_style="bold cyan", box=None)
    delta_table.add_column("Control", style="bold")
    delta_table.add_column("Previous")
    delta_table.add_column("Target Setting")
    delta_table.add_column("Action")

    if not deltas:
        delta_table.add_row("(fixed)", "—", "default", "Hold steady")
    else:
        for ctrl_id, (prev_v, next_v) in deltas.items():
            prev_s = "—" if prev_v is None else str(prev_v)
            next_s = str(next_v)
            changed = prev_v is None or str(prev_v) != str(next_v)
            action_s = (
                "[bold yellow]TURN / FLIP[/bold yellow]" if changed else "[dim]Unchanged[/dim]"
            )
            delta_table.add_row(ctrl_id, prev_s, f"[bold green]{next_s}[/bold green]", action_s)

    panel = Panel(
        delta_table,
        title=(
            f"[bold]Stem [{stem_index}/{total_stems}]: {planned.stem_id}[/bold] "
            f"([cyan]{planned.category}[/cyan])"
        ),
        subtitle=f"[dim]{planned.reason}[/dim]",
        expand=False,
    )
    con.print(panel)

    if countdown_sec > 0.0:
        time.sleep(min(countdown_sec, 0.05))
        return "record"

    if auto_yes:
        return "record"

    prompt_msg = "Press [Enter] to record, [r] to retake previous, or [s] to skip"
    raw_choice = (
        input_fn(prompt_msg)
        if input_fn is not None
        else Prompt.ask(prompt_msg, console=con, default="")
    )
    cleaned = raw_choice.strip().lower()
    if cleaned.startswith("s"):
        return "skip"
    if cleaned.startswith("r"):
        return "retake"
    return "record"


def prompt_calibration_step1(
    setup_instructions: str,
    *,
    auto_yes: bool = False,
    console: Console | None = None,
    input_fn: Callable[[str], str] | None = None,
) -> None:
    """Displays Step-1 Bypass / Dry-Through calibration guidance and waits for `[Enter]`."""
    con = console or Console()
    body = (
        f"{setup_instructions}\n\n"
        "[bold cyan]Step 1/2 — Bypass / Dry-Through Unity Reference[/bold cyan]\n"
        "• Put the pedal/amp in [bold]Bypass[/bold] (or for [bold]Delay / Reverb / Echo[/bold] "
        "pedals without hardware bypass, set [bold]Mix = 0%[/bold] dry-only and "
        "[bold]Feedback = 0%[/bold] so there are zero echo repeats).\n"
        "• Locks interface round-trip latency ([bold]calibrated_latency_samples[/bold]), "
        "unity reference level ([bold]unity_ref_dbfs[/bold]), and baseline noise floor."
    )
    con.print(Panel(body, title="DNAM Capture Calibration — Step 1/2", border_style="cyan"))
    if not auto_yes:
        msg = "Press [Enter] when device is in Bypass / 0% Mix Dry-Through"
        _ = input_fn(msg) if input_fn is not None else Prompt.ask(msg, console=con, default="")


def prompt_calibration_step2(
    step1_event: CalibrationStepEvent,
    *,
    auto_yes: bool = False,
    console: Console | None = None,
    input_fn: Callable[[str], str] | None = None,
) -> None:
    """Displays Step-2 Engaged Hot-Corner / Max-Repeat calibration guidance and prompts."""
    con = console or Console()
    lat = step1_event.calibrated_latency_samples
    body = (
        f"[green]✓ Step 1 Locked:[/green] latency={lat} samples | "
        f"unity_ref={step1_event.unity_ref_dbfs:+.2f} dBFS | "
        f"noise={step1_event.noise_floor_dbfs:.1f} dBFS\n\n"
        "[bold yellow]Step 2/2 — Engaged Hot-Corner & Headroom Verification[/bold yellow]\n"
        "• [bold]Engage[/bold] the pedal/device at its hottest setting (at locked "
        f"[bold]{step1_event.send_gain_db:+.1f} dB[/bold] Unity Send Gain):\n"
        "  - [bold]Drive / Preamp / Amp:[/bold] Set [bold]Drive / Gain / Level[/bold] to maximum.\n"
        "  - [bold]Delay / Echo / Reverb:[/bold] Set [bold]Feedback / Repeats[/bold] and "
        "[bold]Mix[/bold] to maximum (including self-oscillation / dub runaway) to verify that "
        "stacked repeats saturate inside the pedal without clipping your interface A/D input."
    )
    con.print(Panel(body, title="DNAM Capture Calibration — Step 2/2", border_style="yellow"))
    if not auto_yes:
        msg = "Press [Enter] when device is engaged at its Hot Corner"
        _ = input_fn(msg) if input_fn is not None else Prompt.ask(msg, console=con, default="")


def render_models_table(
    models: Sequence[ProjectModelSpec], console: Console | None = None
) -> Table:
    """Renders a Rich summary table of models in the project workspace."""
    con = console or Console()
    table = Table(title="DNAM Project Models", header_style="bold cyan")
    table.add_column("Model ID", style="bold")
    table.add_column("Name")
    table.add_column("Topology")
    table.add_column("Output")
    table.add_column("Controls", justify="right")
    table.add_column("Presets", justify="right")
    table.add_column("Stems", justify="right")

    for m in models:
        table.add_row(
            m.model_id,
            m.name,
            m.topology,
            m.output_mode,
            str(len(m.controls)),
            str(len(m.presets)),
            str(len(m.stems)),
        )
    con.print(table)
    return table


def render_controls_table(
    model_id: str,
    controls: Sequence[ControlSchemaDef],
    console: Console | None = None,
) -> Table:
    """Renders a Rich table of controls for `model_id`."""
    con = console or Console()
    table = Table(title=f"Controls — {model_id}", header_style="bold cyan")
    table.add_column("ID", style="bold")
    table.add_column("Label")
    table.add_column("Kind")
    table.add_column("Declared Range / Options")
    table.add_column("Default")

    for c in controls:
        spec_str = (
            f"{c.declared_range[0]:g} .. {c.declared_range[1]:g}"
            if c.kind == ControlKind.RANGE and c.declared_range
            else ", ".join(c.declared_options or [])
        )
        table.add_row(c.id, c.name, str(c.kind.value), spec_str, str(c.default))
    con.print(table)
    return table


def render_presets_table(
    model_id: str,
    presets: Sequence[PresetDef],
    console: Console | None = None,
) -> Table:
    """Renders a Rich table of presets for `model_id`."""
    con = console or Console()
    table = Table(title=f"Presets — {model_id}", header_style="bold cyan")
    table.add_column("Preset ID", style="bold")
    table.add_column("Name")
    table.add_column("Controls")
    for p in presets:
        c_str = ", ".join(f"{k}={v}" for k, v in p.controls.items())
        table.add_row(p.id, p.name, c_str)
    con.print(table)
    return table


def render_stems_table(
    model_id: str,
    stems: Sequence[ProjectStemEntry],
    console: Console | None = None,
) -> Table:
    """Renders a Rich table of recorded stems for `model_id`."""
    con = console or Console()
    table = Table(title=f"Recorded Stems — {model_id}", header_style="bold cyan")
    table.add_column("Stem ID", style="bold")
    table.add_column("Wet WAV")
    table.add_column("Controls")
    table.add_column("Peak (dBFS)", justify="right")
    table.add_column("Noise (dBFS)", justify="right")
    table.add_column("Latency", justify="right")

    for s in stems:
        c_str = ", ".join(f"{k}={v}" for k, v in s.controls.items()) or "default"
        table.add_row(
            s.stem_id,
            s.wet_wav,
            c_str,
            f"{s.measured_return_peak_dbfs:+.2f}",
            f"{s.noise_floor_dbfs:.1f}",
            f"{s.latency_offset_samples}",
        )
    con.print(table)
    return table


def render_capture_plan_table(
    model_id: str,
    plan: Sequence[PlannedStemCapture],
    console: Console | None = None,
) -> Table:
    """Renders the Star + Switch + Corner + Prescription capture schedule table."""
    con = console or Console()
    table = Table(title=f"Capture Plan — {model_id}", header_style="bold cyan")
    table.add_column("#", justify="right")
    table.add_column("Stem ID", style="bold")
    table.add_column("Category")
    table.add_column("Target Controls")
    table.add_column("Status")
    table.add_column("Reason")

    for idx, item in enumerate(plan, start=1):
        c_str = ", ".join(f"{k}={v}" for k, v in item.controls.items()) or "default"
        status_s = "[green]RECORDED[/green]" if item.is_recorded else "[yellow]PENDING[/yellow]"
        table.add_row(str(idx), item.stem_id, item.category, c_str, status_s, item.reason)
    con.print(table)
    return table


__all__ = [
    "CliCaptureObserver",
    "CliTrainingObserver",
    "StemPromptAction",
    "prompt_calibration_step1",
    "prompt_calibration_step2",
    "prompt_next_stem_action",
    "render_capture_plan_table",
    "render_controls_table",
    "render_models_table",
    "render_presets_table",
    "render_stems_table",
]
