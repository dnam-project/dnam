"""Canonical `.dnam` and `.dnam-ckpt` UTF-8 JSON Header (`DnamJsonHeader`) and `SchemaVersion`
(`ref[sdk.python.schema.rust-parity]`).
"""

from __future__ import annotations

import json
from typing import Literal, Self

from pydantic import Field

from dnam._native import native_serialize_jcs
from dnam.errors import ValidationError
from dnam.schema._base import StrictModel
from dnam.schema.checkpoint import AuditReport, StemRegistryEntry
from dnam.schema.controls import ControlSchemaDef, JsonValue, PresetDef
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

    major: int = Field(default=1, ge=0, le=65535)
    minor: int = Field(default=0, ge=0, le=65535)
    patch: int = Field(default=0, ge=0, le=65535)

    @classmethod
    def parse(cls, version_str: str) -> SchemaVersion:
        """Parses a semantic version string (`'major.minor.patch'` or `'major.minor'`)."""
        parts = version_str.strip().split(".")
        if len(parts) not in (2, 3):
            raise ValidationError(
                f"Invalid schema version '{version_str}': expected 'major.minor.patch'"
            )
        try:
            major = int(parts[0])
            minor = int(parts[1])
            patch = int(parts[2]) if len(parts) == 3 else 0
        except ValueError:
            raise ValidationError(
                f"Invalid numeric segments in schema version '{version_str}'"
            ) from None
        return cls(major=major, minor=minor, patch=patch)

    @property
    def semver(self) -> str:
        """Formats the schema version as `'major.minor.patch'`."""
        return f"{self.major}.{self.minor}.{self.patch}"

    @property
    def is_supported(self) -> bool:
        """Returns `True` if this schema version is supported by the engine (`1.0`..`1.1`)."""
        return self.major == 1 and self.minor <= 1


class ModelMetadata(StrictModel):
    """Essential model identity, calibration, dataset provenance, and cryptographic signatures."""

    name: str
    author: str | None = None
    description: str | None = None
    license: str | None = None
    created_at: str
    input_reference_dbfs: float = -20.5
    provenance: DatasetProvenance = Field(default_factory=DatasetProvenance)
    distribution_policy: DistributionPolicy | None = None
    signatures: ModelSignatures | None = None


def _prune_rust_skip_serializing_if(raw: dict[str, object]) -> dict[str, object]:
    """Removes optional keys that Rust `serde` marks with `skip_serializing_if` when `None` or
    empty.
    """
    if raw.get("stems_registry") is None:
        raw.pop("stems_registry", None)
    else:
        stems = raw["stems_registry"]
        if isinstance(stems, list):
            for stem in stems:
                if isinstance(stem, dict):
                    if stem.get("extracted_physics") is None:
                        stem.pop("extracted_physics", None)
                    for slice_key in ("g_audio_uplo_u_f64", "c_audio_tier_f64"):
                        sl = stem.get(slice_key)
                        if isinstance(sl, dict):
                            for opt_k in ("packing", "feature_dim", "packed_len", "shape"):
                                if sl.get(opt_k) is None:
                                    sl.pop(opt_k, None)

    if raw.get("audit_report") is None:
        raw.pop("audit_report", None)

    meta = raw.get("metadata")
    if isinstance(meta, dict):
        if meta.get("distribution_policy") is None:
            meta.pop("distribution_policy", None)
        sigs = meta.get("signatures")
        if isinstance(sigs, dict):
            if sigs.get("tier_arena_sha256") is None:
                sigs.pop("tier_arena_sha256", None)
            rcpt = sigs.get("marketplace_receipt")
            if isinstance(rcpt, dict) and rcpt.get("buyer_tier_arena_sha256") is None:
                rcpt.pop("buyer_tier_arena_sha256", None)

    ctrls = raw.get("controls_schema")
    if isinstance(ctrls, list):
        for ctrl in ctrls:
            if isinstance(ctrl, dict):
                cal = ctrl.get("calibration")
                if cal is None:
                    ctrl.pop("calibration", None)
                elif isinstance(cal, dict):
                    if cal.get("lobatto_physical_anchors") is None:
                        cal.pop("lobatto_physical_anchors", None)
                    if not cal.get("conditional_physical_anchors"):
                        cal.pop("conditional_physical_anchors", None)
                    if cal.get("tempo_sync_default_subdivision") is None:
                        cal.pop("tempo_sync_default_subdivision", None)

    tdir = raw.get("tensor_directory")
    if isinstance(tdir, dict):
        for sl in tdir.values():
            if isinstance(sl, dict):
                for opt_k in ("packing", "feature_dim", "packed_len", "shape"):
                    if sl.get(opt_k) is None:
                        sl.pop(opt_k, None)

    return raw


class DnamJsonHeader(StrictModel):
    """Top-level UTF-8 JSON header located at byte offset `0x40` of a `.dnam` or `.dnam-ckpt`
    container.
    """

    version: str = "1.0.0"
    container_kind: Literal["dnam", "dnam-ckpt", "preview"] = "dnam"
    architecture: str = "DNAM_TORA"
    sample_rate_train_hz: int = Field(default=48_000, ge=44_100, le=192_000)
    metadata: ModelMetadata
    user_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    io_config: IoConfig = Field(default_factory=IoConfig)
    slimmable_tiers: SlimmableTiersConfig
    controls_schema: list[ControlSchemaDef] = Field(default_factory=list)
    presets: list[PresetDef] = Field(default_factory=list)
    control_bus: ControlBusConfig = Field(default_factory=ControlBusConfig)
    stage0_router: RouterConfig = Field(default_factory=RouterConfig)
    stage1_pre: Stage1PreFxConfig = Field(default_factory=Stage1PreFxConfig)
    stage2_cascade: Stage2CascadeConfig = Field(default_factory=Stage2CascadeConfig)
    stage3_cabinet: Stage3CabinetKautzConfig = Field(default_factory=Stage3CabinetKautzConfig)
    stage4_tank: Stage4SpatialTankConfig = Field(default_factory=Stage4SpatialTankConfig)
    tensor_directory: dict[str, TensorSliceEntry] = Field(default_factory=dict)
    stems_registry: list[StemRegistryEntry] | None = None
    audit_report: AuditReport | None = None

    @property
    def schema_version(self) -> SchemaVersion:
        """Parsed `SchemaVersion` of this header."""
        return SchemaVersion.parse(self.version)

    @classmethod
    def from_json(cls, json_str: str | bytes) -> Self:
        """Parses and validates a UTF-8 JSON header string or byte buffer."""
        try:
            return cls.model_validate_json(json_str)
        except Exception as exc:
            raise ValidationError(f"Invalid DnamJsonHeader JSON: {exc}") from None

    def to_rust_dict(self) -> dict[str, object]:
        """Dumps the header into a dictionary matching Rust `serde_json` field names and
        `skip_serializing_if` rules.
        """
        dumped = self.model_dump(mode="json", by_alias=True)
        return _prune_rust_skip_serializing_if(dumped)

    def to_json(self) -> str:
        """Serializes the header to compact JSON matching Rust `DnamJsonHeader` wire
        representation.
        """
        return json.dumps(self.to_rust_dict(), separators=(",", ":"), ensure_ascii=False)

    def to_jcs(self, *, project_immutable: bool = False) -> str:
        """Serializes the header via RFC 8785 JCS (`ref[sdk.python.schema.rust-parity]`)."""
        return native_serialize_jcs(
            self.to_json(),
            project_immutable=project_immutable,
            reject_floats=False,
        )
