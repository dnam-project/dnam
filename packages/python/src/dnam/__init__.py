"""`dnam` — Idiomatic Python 3.15 SDK & Zero-Copy NumPy SIMD Engine for Dynamic Neural Audio
Modeler (`ref[sdk.python.layout.package-namespace]`, `ref[sdk.python.abi.abi3-abi3t]`).
"""

from dnam import controls, dsp, engine, errors, model, provenance, schema
from dnam.controls import ClampedControl, ClampResult, ControlSpace, TempoSubdivision
from dnam.engine import (
    Backend,
    Controls,
    Engine,
    OverflowFlags,
    Presets,
    Telemetry,
    Tempo,
    Tier,
)
from dnam.errors import (
    ContainerError,
    ControlError,
    DnamError,
    ProvenanceError,
    SampleRateError,
    ValidationError,
)
from dnam.model import Checkpoint, GramMatrix, Model, Preamble, crc32c
from dnam.schema import DnamJsonHeader, DnamProtectionProfile, VerificationStatus

__version__ = "0.2.0"

__all__ = [
    "Backend",
    "Checkpoint",
    "ClampResult",
    "ClampedControl",
    "ContainerError",
    "ControlError",
    "ControlSpace",
    "Controls",
    "DnamError",
    "DnamJsonHeader",
    "DnamProtectionProfile",
    "Engine",
    "GramMatrix",
    "Model",
    "OverflowFlags",
    "Preamble",
    "Presets",
    "ProvenanceError",
    "SampleRateError",
    "Telemetry",
    "Tempo",
    "TempoSubdivision",
    "Tier",
    "ValidationError",
    "VerificationStatus",
    "__version__",
    "controls",
    "crc32c",
    "dsp",
    "engine",
    "errors",
    "model",
    "provenance",
    "schema",
]
