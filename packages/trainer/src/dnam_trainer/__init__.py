"""Dynamic Neural Audio Modeler (`dnam_trainer`) — Multi-Model Project Workspace,
Hardware/Software Capture Pipeline, Differentiable Tri-Physics DSP, Non-Stop 3-Phase
Control-Space Audit, and 4-Stage Deterministic PyTorch Trainer (`CL-01`..`CL-18`)."""

from __future__ import annotations

from dnam_trainer import capture, cli, control, dsp, events, project, schema, training
from dnam_trainer.project.workspace import DnamProject
from dnam_trainer.schema import (
    ResumeJobRequest,
    TrainingConfig,
    TrainingJobRequest,
    TrainingJobResult,
    TrainingStem,
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
    "DnamProject",
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
    "capture",
    "cli",
    "control",
    "dsp",
    "events",
    "get_training_backend",
    "list_training_backends",
    "project",
    "register_training_backend",
    "schema",
    "training",
]
