"""Training event observers (`RichTrainingObserver`, `JsonlTrainingObserver`,
`SseTrainingObserver`) for terminal, `packages/studio` subprocess IPC, and cloud SSE streams."""

from __future__ import annotations

import sys
from typing import Literal, TextIO

from rich.console import Console

from dnam_trainer.events import (
    AuditPhaseEvent,
    CacheEvent,
    PlannerEvent,
    RuntimeEvent,
    SolverEvent,
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
            bands_str = ""
            if (
                event.esr_lf_db is not None
                and event.esr_mid_db is not None
                and event.esr_hf_db is not None
            ):
                bands_str = (
                    f" | Bands(LF/MF/HF)={event.esr_lf_db:+.2f}/"
                    f"{event.esr_mid_db:+.2f}/{event.esr_hf_db:+.2f} dB"
                )
            self.console.print(
                f"  [dim]Epoch {event.epoch:>3}/{event.total_epochs}[/dim] | "
                f"bESR: Lite={event.esr_lite_db:+.2f} dB | "
                f"Full={event.esr_full_db:+.2f} dB | "
                f"[bold]Ultra={event.esr_ultra_db:+.2f} dB[/bold]"
                f"{bands_str} | "
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
            psycho_str = ""
            if (
                event.esr_lf_db is not None
                and event.esr_mid_db is not None
                and event.esr_hf_db is not None
            ):
                psycho_str += (
                    f" | Bands(LF/MF/HF)={event.esr_lf_db:+.2f}/"
                    f"{event.esr_mid_db:+.2f}/{event.esr_hf_db:+.2f} dB"
                )
            if event.mr_stft_db is not None:
                psycho_str += f" | MR-STFT={event.mr_stft_db:.2f} dB"
            if event.ltas_rms_tilt_db is not None and event.ltas_hf_tilt_db is not None:
                psycho_str += (
                    f" | Tilt(RMS/HF)={event.ltas_rms_tilt_db:.2f}/{event.ltas_hf_tilt_db:+.2f} dB"
                )
            if event.seg_esr_db is not None:
                psycho_str += f" | SegESR={event.seg_esr_db:+.2f} dB"
            self.console.print(
                f"[bold green]✓ {mode_label} Complete ({event.model_id})[/bold green] | "
                f"Final bESR: Lite={event.esr_lite_db:+.2f} dB | "
                f"Full={event.esr_full_db:+.2f} dB | "
                f"Ultra={event.esr_ultra_db:+.2f} dB{psycho_str} | "
                f"Core2={c2_state} | {event.elapsed_sec:.2f}s"
            )
        elif isinstance(event, RuntimeEvent):
            budget_gib = event.budget_bytes / (1024**3)
            cap_gib = event.max_tensor_bytes / (1024**3)
            unver = " [unverified]" if event.unverified_hardware else ""
            self.console.print(
                f"[bold cyan][Runtime][/bold cyan] device={event.device_kind} "
                f"({event.torch_device}){unver} | contraction={event.contraction_dtype} | "
                f"budget={budget_gib:.2f} GiB (max_tensor={cap_gib:.2f} GiB)"
            )
            if event.warning:
                self.console.print(f"  [yellow]⚠ {event.warning}[/yellow]")
        elif isinstance(event, PlannerEvent):
            budget_gib = event.budget_bytes / (1024**3)
            peak_gib = event.estimated_peak_bytes / (1024**3)
            self.console.print(
                f"[bold cyan][Planner][/bold cyan] mode={event.mode} | "
                f"block_rows={event.block_rows} | stem_group={event.stem_group_size} | "
                f"est_peak={peak_gib:.2f}/{budget_gib:.2f} GiB | regimes={len(event.regimes)}"
            )
        elif isinstance(event, CacheEvent):
            mb = event.bytes_count / (1024**2)
            self.console.print(
                f"  [dim][Cache {event.tier}:{event.action}] {event.key} "
                f"({mb:.2f} MiB, {event.elapsed_ms:.1f} ms) {event.message}[/dim]"
            )
        elif isinstance(event, SolverEvent):
            fb = f" | lstsq_fallbacks={event.lstsq_fallbacks}" if event.lstsq_fallbacks > 0 else ""
            self.console.print(
                f"  [dim][HostSolve {event.pass_name}] {event.num_solves}x "
                f"({event.matrix_dim}x{event.matrix_dim}, rhs={event.rhs_cols}) | "
                f"transfer={event.transfer_ms:.1f} ms | solve={event.solve_ms:.1f} ms{fb}[/dim]"
            )
            if event.warning:
                self.console.print(f"  [yellow]⚠ {event.warning}[/yellow]")


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
