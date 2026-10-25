"""Dynamic Neural Audio Modeler (`dnam_trainer`) — Differentiable Tri-Physics DSP,
Non-Stop 3-Phase Control-Space Audit, 4-Stage Deterministic PyTorch Trainer (`CL-01`..`CL-18`),
and Headless `dnam-train` Worker CLI."""

from __future__ import annotations

from dnam_trainer import cli, control, dsp, events, schema, training
from dnam_trainer.schema import (
    ResumeJobRequest,
    TrainingConfig,
    TrainingJobRequest,
    TrainingJobResult,
    TrainingStem,
    parse_compact_range_spec,
    parse_compact_switch_spec,
)
from dnam_trainer.training.backends import (
    LocalTrainingBackend,
    RemoteTrainingBackend,
    TrainingBackend,
    get_training_backend,
    list_training_backends,
    register_training_backend,
)
from dnam_trainer.training.pipeline import DnamTrainer, DnamTriPhysicsModel

__version__ = "0.1.0"

__all__ = [
    "DnamTrainer",
    "DnamTriPhysicsModel",
    "LocalTrainingBackend",
    "RemoteTrainingBackend",
    "ResumeJobRequest",
    "TrainingBackend",
    "TrainingConfig",
    "TrainingJobRequest",
    "TrainingJobResult",
    "TrainingStem",
    "__version__",
    "cli",
    "control",
    "dsp",
    "events",
    "get_training_backend",
    "list_training_backends",
    "parse_compact_range_spec",
    "parse_compact_switch_spec",
    "register_training_backend",
    "schema",
    "training",
]
