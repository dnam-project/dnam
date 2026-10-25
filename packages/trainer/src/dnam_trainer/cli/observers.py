"""Training event observers (`RichTrainingObserver`, `JsonlTrainingObserver`,
`SseTrainingObserver`) for terminal, `packages/studio` subprocess IPC, and cloud SSE streams."""

from __future__ import annotations

import sys
from typing import Literal, TextIO

from rich.console import Console

from dnam_trainer.events import (
    AuditPhaseEvent,
    TrainingCompleteEvent,
    TrainingEvent,
    TrainingObserver,
    TrainingStageEvent,
    TrainingStepEvent,
    format_training_event_sse,
)

type EventFormat = Literal["rich", "jsonl", "sse"]


class RichTrainingObserver:
    """Rich terminal observer for interactive `dnam-train` sessions."""

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


class JsonlTrainingObserver:
    """Line-delimited JSON (`JSONL`) observer consumed by `packages/studio` (`dnam train`)."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stdout

    def on_training_event(self, event: TrainingEvent) -> None:
        self._stream.write(event.model_dump_json() + "\n")
        self._stream.flush()


class SseTrainingObserver:
    """Server-Sent Events (`SSE`) observer for remote GPU workers."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stdout

    def on_training_event(self, event: TrainingEvent) -> None:
        self._stream.write(format_training_event_sse(event))
        self._stream.flush()


def create_training_observer(
    event_format: EventFormat = "jsonl",
    *,
    console: Console | None = None,
    stream: TextIO | None = None,
) -> TrainingObserver:
    """Instantiates the requested [`TrainingObserver`] implementation."""
    if event_format == "jsonl":
        return JsonlTrainingObserver(stream=stream)
    if event_format == "sse":
        return SseTrainingObserver(stream=stream)
    return RichTrainingObserver(console=console)


__all__ = [
    "EventFormat",
    "JsonlTrainingObserver",
    "RichTrainingObserver",
    "SseTrainingObserver",
    "create_training_observer",
]
