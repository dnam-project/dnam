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
    DistributionPolicy,
    DnamEngineSettings,
    DnamProtectionProfile,
    OemDeviceCertificate,
    Tier,
    VerificationStatus,
)

__version__ = "0.1.0"

__all__ = [
    "BackendKind",
    "ControlKind",
    "ControlRack",
    "ControlSpace",
    "DistributionPolicy",
    "DnamContainer",
    "DnamContainerError",
    "DnamControlError",
    "DnamEngine",
    "DnamEngineSettings",
    "DnamError",
    "DnamProtectionProfile",
    "DnamSampleRateError",
    "DnamValidationError",
    "GramMatrix",
    "OemDeviceCertificate",
    "Tier",
    "VerificationStatus",
    "__version__",
    "control",
    "crc32c",
    "dsp",
    "provenance",
    "schema",
]
