"""`dnam_engine` — Idiomatic Python 3.14 Interface & Zero-Copy NumPy SIMD Engine for DNAM."""

from dnam_engine import control, dsp, provenance, schema
from dnam_engine.container import DnamContainer, GramMatrix, crc32c
from dnam_engine.control import ControlRack, ControlSpace
from dnam_engine.engine import DnamEngine
from dnam_engine.exceptions import (
    DnamContainerError,
    DnamControlError,
    DnamError,
    DnamSampleRateError,
    DnamValidationError,
)
from dnam_engine.schema import (
    BackendKind,
    ControlKind,
    DnamEngineSettings,
    Tier,
    VerificationStatus,
)

__version__ = "0.1.0"

__all__ = [
    "BackendKind",
    "ControlKind",
    "ControlRack",
    "ControlSpace",
    "DnamContainer",
    "DnamContainerError",
    "DnamControlError",
    "DnamEngine",
    "DnamEngineSettings",
    "DnamError",
    "DnamSampleRateError",
    "DnamValidationError",
    "GramMatrix",
    "Tier",
    "VerificationStatus",
    "__version__",
    "control",
    "crc32c",
    "dsp",
    "provenance",
    "schema",
]
