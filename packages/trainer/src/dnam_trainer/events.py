"""UI-agnostic Event and Observer protocols (`CaptureEvent`, `TrainingEvent`,
`CaptureObserver`, `TrainingObserver`) shared by the `cyclopts`/`rich` CLI,
future Textual TUI, and remote Server-Sent Events (SSE) training streams."""

from __future__ import annotations

from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import Field, TypeAdapter

from dnam_trainer.schema import AuditReport, ControlCoords, StrictModel, TargetExcitationProfile


class CalibrationStepEvent(StrictModel):
    """Emitted after Step 1 (bypass unity reference) or Step 2 (hot-corner verification)."""

    event_type: Literal["calibration_step"] = "calibration_step"
    step: Literal[1, 2]
    backend_id: str
    send_gain_db: float
    unity_ref_dbfs: float
    measured_peak_dbfs: float
    measured_rms_dbfs: float
    net_gain_over_unity_db: float = 0.0
    compression_ratio: float = 1.0
    clipped_sample_fraction: float = 0.0
    calibrated_latency_samples: int = 0
    noise_floor_dbfs: float = -96.0
    excitation_profile: TargetExcitationProfile | None = None


class StemCaptureEvent(StrictModel):
    """Emitted when a stem capture starts, finishes, or is skipped."""

    event_type: Literal["stem_capture"] = "stem_capture"
    status: Literal["starting", "completed", "skipped", "retake"]
    model_id: str
    stem_id: str
    stem_index: int = Field(ge=1)
    total_stems: int = Field(ge=1)
    controls: ControlCoords = Field(default_factory=dict)
    measured_peak_dbfs: float = 0.0
    noise_floor_dbfs: float = -96.0
    clipped_sample_fraction: float = 0.0
    latency_offset_samples: int = 0


type CaptureEvent = Annotated[
    CalibrationStepEvent | StemCaptureEvent,
    Field(discriminator="event_type"),
]

_CAPTURE_EVENT_ADAPTER: TypeAdapter[CaptureEvent] = TypeAdapter(CaptureEvent)


def parse_capture_event_json(raw_json: str | bytes) -> CaptureEvent:
    """Parses a discriminated `CaptureEvent` from JSON."""
    return _CAPTURE_EVENT_ADAPTER.validate_json(raw_json)


class TrainingStageEvent(StrictModel):
    """Emitted when entering or completing one of the 4 training stages."""

    event_type: Literal["training_stage"] = "training_stage"
    stage: Literal[1, 2, 3, 4]
    stage_name: str
    message: str


class TrainingStepEvent(StrictModel):
    """Emitted at the end of an epoch or closed-form solve step."""

    event_type: Literal["training_step"] = "training_step"
    stage: Literal[1, 2, 3, 4]
    epoch: int = Field(ge=0)
    total_epochs: int = Field(ge=1)
    esr_lite_db: float
    esr_full_db: float
    esr_ultra_db: float
    esr_lf_db: float | None = None
    esr_mid_db: float | None = None
    esr_hf_db: float | None = None
    barrier_loss: float = 0.0
    group_lasso_loss: float = 0.0
    surrogate_sigma_samples: float = 0.0
    elapsed_sec: float = 0.0


class AuditPhaseEvent(StrictModel):
    """Emitted when Phase-1, Phase-2, or Phase-3 control-space audit completes."""

    event_type: Literal["audit_phase"] = "audit_phase"
    phase: Literal[1, 2, 3]
    audit_report: AuditReport
    rail_lockup_clamped_corners: int = 0


class TrainingCompleteEvent(StrictModel):
    """Emitted when full 4-Stage training or `.dnam-ckpt` iterative resume finishes."""

    event_type: Literal["training_complete"] = "training_complete"
    model_id: str
    esr_lite_db: float
    esr_full_db: float
    esr_ultra_db: float
    esr_lf_db: float | None = None
    esr_mid_db: float | None = None
    esr_hf_db: float | None = None
    mr_stft_db: float | None = None
    ltas_rms_tilt_db: float | None = None
    ltas_hf_tilt_db: float | None = None
    seg_esr_db: float | None = None
    core2_active: bool
    lfo_env_active: bool
    resume_tier_executed: Literal[0, 1, 2] = 0
    elapsed_sec: float = 0.0


type TrainingEvent = Annotated[
    TrainingStageEvent | TrainingStepEvent | AuditPhaseEvent | TrainingCompleteEvent,
    Field(discriminator="event_type"),
]

_TRAINING_EVENT_ADAPTER: TypeAdapter[TrainingEvent] = TypeAdapter(TrainingEvent)


def parse_training_event_json(raw_json: str | bytes) -> TrainingEvent:
    """Parses a discriminated `TrainingEvent` from JSON or an SSE `data:` frame."""
    text = raw_json.decode("utf-8") if isinstance(raw_json, bytes) else raw_json
    stripped = text.strip()
    if stripped.startswith("data:"):
        stripped = stripped.removeprefix("data:").strip()
    return _TRAINING_EVENT_ADAPTER.validate_json(stripped)


def format_training_event_sse(event: TrainingEvent) -> str:
    """Serializes a `TrainingEvent` as a Server-Sent Events (`SSE`) frame."""
    return f"event: {event.event_type}\ndata: {event.model_dump_json()}\n\n"


@runtime_checkable
class CaptureObserver(Protocol):
    """Observer protocol receiving hardware/software capture events."""

    def on_capture_event(self, event: CaptureEvent) -> None:
        """Handles a `CaptureEvent`."""


@runtime_checkable
class TrainingObserver(Protocol):
    """Observer protocol receiving 4-Stage training and 3-Phase audit events."""

    def on_training_event(self, event: TrainingEvent) -> None:
        """Handles a `TrainingEvent`."""


class NullObserver:
    """No-op observer implementing both `CaptureObserver` and `TrainingObserver`."""

    def on_capture_event(self, event: CaptureEvent) -> None:
        _ = event

    def on_training_event(self, event: TrainingEvent) -> None:
        _ = event


__all__ = [
    "AuditPhaseEvent",
    "CalibrationStepEvent",
    "CaptureEvent",
    "CaptureObserver",
    "NullObserver",
    "StemCaptureEvent",
    "TrainingCompleteEvent",
    "TrainingEvent",
    "TrainingObserver",
    "TrainingStageEvent",
    "TrainingStepEvent",
    "format_training_event_sse",
    "parse_capture_event_json",
    "parse_training_event_json",
]
