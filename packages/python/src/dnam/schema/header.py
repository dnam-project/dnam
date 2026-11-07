"""Canonical `.dnam` and `.dnam-ckpt` UTF-8 JSON Header (`DnamJsonHeader`) and `SchemaVersion`
(`ref[sdk.python.schema.rust-parity]`).
"""

from __future__ import annotations

import json
from dataclasses import field
from typing import Literal, Self, cast

from dnam._native import (
    native_parse_schema_version,
    native_serialize_jcs,
    native_validate_and_normalize_header,
)
from dnam.errors import ValidationError
from dnam.schema._base import StrictModel, strict_field
from dnam.schema.checkpoint import AuditReport, StemRegistryEntry
from dnam.schema.controls import ControlKind, ControlSchemaDef, JsonValue, PresetDef
from dnam.schema.io_and_tiers import IoConfig, SlimmableTiersConfig
from dnam.schema.signatures import (
    DatasetProvenance,
    DistributionPolicy,
    ModelSignatures,
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


class SchemaVersion(StrictModel):
    """Semantic version (`major.minor.patch`) of the `.dnam` and `.dnam-ckpt` container schema."""

    major: int = strict_field(default=1, ge=0, le=65535)
    minor: int = strict_field(default=0, ge=0, le=65535)
    patch: int = strict_field(default=0, ge=0, le=65535)

    @classmethod
    def parse(cls, version_str: str) -> SchemaVersion:
        """Parses a semantic version string (`'major.minor.patch'` or `'major.minor'`)."""
        major, minor, patch, _ = native_parse_schema_version(version_str)
        return cls._construct_trusted(major=major, minor=minor, patch=patch)

    @property
    def semver(self) -> str:
        """Formats the schema version as `'major.minor.patch'`."""
        return f"{self.major}.{self.minor}.{self.patch}"

    @property
    def is_supported(self) -> bool:
        """Returns `True` if this schema version is supported by the engine (`1.0`..`1.1`)."""
        _, _, _, supported = native_parse_schema_version(self.semver)
        return supported


class ModelMetadata(StrictModel):
    """Essential model identity, calibration, dataset provenance, and cryptographic signatures."""

    name: str
    author: str | None = None
    description: str | None = None
    license: str | None = None
    created_at: str
    input_reference_dbfs: float = -20.5
    provenance: DatasetProvenance = strict_field(default_factory=DatasetProvenance)
    distribution_policy: DistributionPolicy | None = None
    signatures: ModelSignatures | None = None


class DnamJsonHeader(StrictModel):
    """Top-level UTF-8 JSON header located at byte offset `0x40` of a `.dnam` or `.dnam-ckpt`
    container.
    """

    version: str = "1.0.0"
    container_kind: Literal["dnam", "dnam-ckpt", "preview"] = "dnam"
    architecture: str = "DNAM_TORA"
    sample_rate_train_hz: int = strict_field(default=48_000, ge=44_100, le=96_000)
    metadata: ModelMetadata
    user_metadata: dict[str, JsonValue] = strict_field(default_factory=dict)
    io_config: IoConfig = strict_field(default_factory=IoConfig)
    slimmable_tiers: SlimmableTiersConfig
    controls_schema: list[ControlSchemaDef] = strict_field(default_factory=list)
    presets: list[PresetDef] = strict_field(default_factory=list)
    control_bus: ControlBusConfig = strict_field(default_factory=ControlBusConfig)
    stage0_router: RouterConfig = strict_field(default_factory=RouterConfig)
    stage1_pre: Stage1PreFxConfig = strict_field(default_factory=Stage1PreFxConfig)
    stage2_cascade: Stage2CascadeConfig = strict_field(default_factory=Stage2CascadeConfig)
    stage3_cabinet: Stage3CabinetKautzConfig = strict_field(
        default_factory=Stage3CabinetKautzConfig
    )
    stage4_tank: Stage4SpatialTankConfig = strict_field(default_factory=Stage4SpatialTankConfig)
    tensor_directory: dict[str, TensorSliceEntry] = strict_field(default_factory=dict)
    stems_registry: list[StemRegistryEntry] | None = None
    audit_report: AuditReport | None = None

    _cached_json: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        range_ids = {c.id for c in self.controls_schema if c.kind == ControlKind.RANGE}
        if range_ids:
            for preset in self.presets:
                for k, v in preset.controls.items():
                    if k in range_ids and type(v) is int:
                        preset.controls[k] = float(v)
            if self.stems_registry is not None:
                for stem in self.stems_registry:
                    for k, v in stem.controls_meta.items():
                        if k in range_ids and type(v) is int:
                            stem.controls_meta[k] = float(v)
        if self._cached_json is None:
            raw_json = super().to_json(by_alias=True)
            canonical = native_validate_and_normalize_header(raw_json)
            object.__setattr__(self, "_cached_json", canonical)

    @property
    def schema_version(self) -> SchemaVersion:
        """Parsed `SchemaVersion` of this header."""
        return SchemaVersion.parse(self.version)

    @classmethod
    def from_json(cls, json_data: str | bytes | bytearray | memoryview) -> Self:
        """Parses and validates a UTF-8 JSON header string or byte buffer."""
        try:
            raw_str = json_data if isinstance(json_data, str) else bytes(json_data).decode("utf-8")
        except Exception as exc:
            raise ValidationError(f"Invalid DnamJsonHeader JSON: {exc}") from None
        canonical = native_validate_and_normalize_header(raw_str)
        raw_dict: dict[str, object] = json.loads(canonical)
        raw_dict.setdefault("stems_registry", None)
        raw_dict.setdefault("audit_report", None)
        m = cls._construct_trusted_dict(raw_dict)
        object.__setattr__(m, "_cached_json", canonical)
        StrictModel.__post_init__(m)
        return m

    def to_rust_dict(self) -> dict[str, object]:
        """Dumps the header into a dictionary matching Rust `serde_json` field names and
        `skip_serializing_if` rules.
        """
        return cast(dict[str, object], json.loads(self.to_json(by_alias=True)))

    def to_json(self, *, by_alias: bool = True) -> str:
        """Serializes the header to compact JSON matching Rust `DnamJsonHeader` wire
        representation.
        """
        if by_alias and self._cached_json is not None:
            return self._cached_json
        return super().to_json(by_alias=by_alias)

    def to_jcs(self, *, project_immutable: bool = False) -> str:
        """Serializes the header via RFC 8785 JCS (`ref[sdk.python.schema.rust-parity]`)."""
        return native_serialize_jcs(
            self.to_json(),
            project_immutable=project_immutable,
            reject_floats=False,
        )
