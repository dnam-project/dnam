"""Pydantic v2 `StrictModel` schema mirror of `dnam-engine::schema`
(`ref[sdk.python.schema.rust-parity]`).
"""

from dnam.schema._base import Backend, StrictModel, Tier, VerificationStatus
from dnam.schema.checkpoint import (
    AuditReport,
    ExtractedPhysics,
    StemProvenance,
    StemRegistryEntry,
)
from dnam.schema.controls import (
    ConditionalClampRule,
    ControlCalibration,
    ControlCalibrationRole,
    ControlKind,
    ControlSchemaDef,
    PresetDef,
)
from dnam.schema.header import DnamJsonHeader, ModelMetadata, SchemaVersion
from dnam.schema.io_and_tiers import (
    BlendMode,
    IoConfig,
    LevelingConfig,
    LevelingMetric,
    ModelCapabilities,
    SlimmableTiersConfig,
    TierProfile,
)
from dnam.schema.signatures import (
    AuthorizedDevice,
    CreatorSignature,
    DatasetProvenance,
    DistributionPolicy,
    DnamProtectionProfile,
    MarketplaceReceipt,
    ModelSignatures,
    OemDeviceCertificate,
    RoyaltySplit,
    WatermarkParams,
)
from dnam.schema.stages import (
    ControlBusConfig,
    RouterConfig,
    Stage1PreFxConfig,
    Stage2CascadeConfig,
    Stage3CabinetKautzConfig,
    Stage4SpatialTankConfig,
    TensorSliceEntry,
)
from dnam.schema.tempo import (
    TempoInversionResult,
    TempoSubdivision,
    TempoSyncResolution,
)
from dnam.schema.ui_hints import ControlUiHints

__all__ = [
    "AuditReport",
    "AuthorizedDevice",
    "Backend",
    "BlendMode",
    "ConditionalClampRule",
    "ControlBusConfig",
    "ControlCalibration",
    "ControlCalibrationRole",
    "ControlKind",
    "ControlSchemaDef",
    "ControlUiHints",
    "CreatorSignature",
    "DatasetProvenance",
    "DistributionPolicy",
    "DnamJsonHeader",
    "DnamProtectionProfile",
    "ExtractedPhysics",
    "IoConfig",
    "LevelingConfig",
    "LevelingMetric",
    "MarketplaceReceipt",
    "ModelCapabilities",
    "ModelMetadata",
    "ModelSignatures",
    "OemDeviceCertificate",
    "PresetDef",
    "RouterConfig",
    "RoyaltySplit",
    "SchemaVersion",
    "SlimmableTiersConfig",
    "Stage1PreFxConfig",
    "Stage2CascadeConfig",
    "Stage3CabinetKautzConfig",
    "Stage4SpatialTankConfig",
    "StemProvenance",
    "StemRegistryEntry",
    "StrictModel",
    "TempoInversionResult",
    "TempoSubdivision",
    "TempoSyncResolution",
    "TensorSliceEntry",
    "Tier",
    "TierProfile",
    "VerificationStatus",
    "WatermarkParams",
]
