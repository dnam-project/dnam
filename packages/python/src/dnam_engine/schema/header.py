"""Canonical `.dnam` and `.dnam-ckpt` UTF-8 JSON header (`DnamJsonHeader`) and subsystem schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from dnam_engine.exceptions import DnamSampleRateError, DnamValidationError
from dnam_engine.schema.base import StrictModel
from dnam_engine.schema.controls import ControlSchemaDef, PresetDef
from dnam_engine.schema.signatures import DatasetProvenance, DistributionPolicy, ModelSignatures


class ModelMetadata(StrictModel):
    """Essential model identity, calibration, dataset provenance, and cryptographic signatures."""

    name: str = Field(min_length=1)
    author: str | None = None
    description: str | None = None
    license: str | None = None
    created_at: str = Field(min_length=1)
    input_reference_dbfs: float = -20.5
    provenance: DatasetProvenance = Field(default_factory=DatasetProvenance)
    distribution_policy: DistributionPolicy | None = None
    signatures: ModelSignatures | None = None


class ModelCapabilities(StrictModel):
    """Model-level host integration capabilities (`io_config.capabilities`)."""

    blend_mode: Literal["none", "host_parallel", "modeled_control"] = "none"
    default_blend: float = Field(default=1.0, ge=0.0, le=1.0)
    wet_polarity_inverted: bool = False
    supports_trails: bool = False
    tempo_sync_capable: bool = False
    supports_freeze: bool = False
    freeze_mutes_dry: bool = False


class LevelingConfig(StrictModel):
    """Auto unity gain-staging and loudness calibration metadata (`io_config.leveling`).

    Evaluated strictly at default control settings (`controls_schema[*].default`, normalized
    `u = 0`), never `presets[0]`. Defaults to gated unweighted broadband RMS (`rms_dbfs`) for
    electrical Vrms pedal-to-pedal gain staging.
    """

    auto_unity_enabled: bool = True
    leveling_metric: Literal["rms_dbfs", "k_weighted_lufs"] = "rms_dbfs"
    reference_input_level_db: float = -20.5
    measured_default_output_level_db: float = -20.5
    unity_gain_offset_db: float = 0.0
    target_gain_db: float = 0.0
    calibrated_trim_db: float = 0.0


class IoConfig(StrictModel):
    """Audio I/O topology, host capabilities, and default-controls unity leveling configuration."""

    inputs: int = Field(default=2, ge=1, le=2)
    output_mode: Literal["stereo", "mono"] = "stereo"
    latency_samples: int = Field(default=0, ge=0)
    tail_time_ms: float = Field(default=0.0, ge=0.0)
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    leveling: LevelingConfig = Field(default_factory=LevelingConfig)


class TierProfile(StrictModel):
    """Metadata and prefix checksum for an individual slimmable tier (`lite`, `full`, `ultra`)."""

    channels: int
    fdn_lines: int = Field(ge=0, le=8)
    early_taps: int = Field(default=0, ge=0, le=8)
    head_svd_rank: int = Field(default=4, ge=1, le=8)
    loudness_lufs: float = -20.5
    output_trim_db: float = 0.0
    validation_esr_db: float = 0.0
    arena_end_byte_offset: int = Field(ge=64)
    prefix_crc32c: int = Field(default=0, ge=0)

    @field_validator("channels")
    @classmethod
    def _validate_channels(cls, v: int) -> int:
        if v not in (4, 8, 12):
            raise DnamValidationError(f"TierProfile.channels must be 4, 8, or 12; got {v}")
        return v

    @field_validator("arena_end_byte_offset")
    @classmethod
    def _validate_alignment(cls, v: int) -> int:
        if (v & 0x3F) != 0:
            raise DnamValidationError(
                f"TierProfile.arena_end_byte_offset ({v}) must be 64-byte aligned"
            )
        return v


class SlimmableTiersConfig(StrictModel):
    """Slimmable tier profiles (`lite`, `full`, `ultra`)."""

    lite: TierProfile
    full: TierProfile
    ultra: TierProfile


class ControlBusConfig(StrictModel):
    """Control-Bus configuration (`control_bus`)."""

    decimation: int = Field(default=16, ge=8, le=64)
    envelopes: int = Field(default=3, ge=0, le=8)
    lfo_oscillators: int = Field(default=2, ge=0, le=8)
    lfo_harmonics: int = Field(default=4, ge=1, le=8)
    fm_interaction_rank: int = Field(default=4, ge=1, le=8)
    head_trust_region_radius: float = Field(default=2.5, gt=0.0, le=16.0)


class Core1Config(StrictModel):
    """Core 1 Hybrid SSM configuration (`core1_hybrid_ssm`)."""

    blocks: int = Field(default=4, ge=1, le=8)
    conv_kernel: int = Field(default=6, ge=1, le=16)
    dilations: tuple[int, int, int, int] = (1, 3, 7, 19)
    adaa_mode: str = "1x_bspline_algebraic"
    ac_coupling_pole: float = Field(default=0.9993, gt=0.9, lt=1.0)
    weight_layout: str = "tier_delta_exec_order_soa_align64"


class Core2Config(StrictModel):
    """Core 2 Diffusive Modulated Unitary FDN configuration (`core2_mufdn`)."""

    active: bool = False
    delay_storage: str = "fp16"
    ring_buffer_pow2_log2: int = Field(default=0, ge=0, le=18)
    interp: str = "cubic_lagrange_4tap"
    tzf_reference_tap_samples: float = Field(default=64.0, ge=0.0)
    switch_xfade_samples: int = Field(default=128, ge=0)
    allpass_delays_samples: tuple[int, int, int, int, int, int, int, int] = (
        19,
        31,
        47,
        73,
        103,
        139,
        179,
        223,
    )
    dc_blocker_pole: float = Field(default=0.995, gt=0.9, lt=1.0)
    loop_ceiling: float = Field(default=0.985, gt=0.0, le=2.0)


class TensorSliceEntry(StrictModel):
    """Tensor directory entry describing a 64-byte aligned slice in `Arena 0..3`."""

    arena: int = Field(ge=0, le=3)
    dtype: Literal["f32", "f64", "fp16"] = "f32"
    packing: str | None = None
    feature_dim: int | None = None
    packed_len: int | None = None
    shape: list[int] | None = None
    byte_offset: int = Field(ge=0)
    byte_length: int = Field(ge=64)

    @model_validator(mode="after")
    def _validate_alignment(self) -> TensorSliceEntry:
        if (self.byte_offset & 0x3F) != 0:
            raise DnamValidationError(
                f"TensorSliceEntry.byte_offset ({self.byte_offset}) must be 64-byte aligned"
            )
        if (self.byte_length & 0x3F) != 0:
            raise DnamValidationError(
                f"TensorSliceEntry.byte_length ({self.byte_length}) must be 64-byte aligned"
            )
        return self


class StemProvenance(StrictModel):
    """Platform-agnostic IEEE-754 PCM SHA-256 provenance for a training stem."""

    input_pcm_sha256: str = ""
    target_pcm_sha256: str = ""
    num_frames: int = Field(default=0, ge=0)


class ExtractedPhysics(StrictModel):
    """Extracted Stage-1 physical system-identification anchors for a stem."""

    delays_ms: list[float] = Field(default_factory=list)
    lfo_freqs_hz: list[float] = Field(default_factory=list)
    decay_gains_lin: list[float] = Field(default_factory=list)
    detected_grain_size_ms: float = 0.0
    detected_pitch_shift_semitones: float = 0.0


class StemRegistryEntry(StrictModel):
    """`.dnam-ckpt` per-stem registry record (`stems_registry[*]`)."""

    stem_id: str = Field(min_length=1)
    controls_meta: dict[str, Any] = Field(default_factory=dict)
    provenance: StemProvenance = Field(default_factory=StemProvenance)
    stem_arena3_crc32c: int = Field(default=0, ge=0)
    user_metadata: dict[str, Any] = Field(default_factory=dict)
    extracted_physics: ExtractedPhysics | None = None
    g_audio_uplo_u_f64: TensorSliceEntry
    c_audio_tier_f64: TensorSliceEntry


class AuditReport(StrictModel):
    """`.dnam-ckpt` 3-phase control clamp audit report and actionable stem prescriptions."""

    phase1_clamps: list[str] = Field(default_factory=list)
    phase2_rail_warnings: list[str] = Field(default_factory=list)
    phase3_clamps: list[str] = Field(default_factory=list)
    prescriptions: list[str] = Field(default_factory=list)


CURRENT_SCHEMA_VERSION: str = "1.0.0"
SUPPORTED_SCHEMA_MAJOR: int = 1
MAX_SUPPORTED_SCHEMA_MINOR: int = 1


def parse_schema_version(version_str: str) -> tuple[int, int, int]:
    """Parses and validates a `'major.minor.patch'` container schema version."""
    parts = version_str.strip().split(".")
    if len(parts) not in (2, 3):
        raise DnamValidationError(
            f"Invalid schema version '{version_str}': expected 'major.minor.patch'"
        )
    try:
        major = int(parts[0])
        minor = int(parts[1])
        patch = int(parts[2]) if len(parts) == 3 else 0
    except ValueError as err:
        raise DnamValidationError(
            f"Invalid numeric segments in schema version '{version_str}'"
        ) from err
    if major != SUPPORTED_SCHEMA_MAJOR or minor > MAX_SUPPORTED_SCHEMA_MINOR or patch < 0:
        max_ver = f"{SUPPORTED_SCHEMA_MAJOR}.{MAX_SUPPORTED_SCHEMA_MINOR}"
        raise DnamValidationError(
            f"Unsupported schema version '{version_str}' "
            f"(supported: {SUPPORTED_SCHEMA_MAJOR}.0..{max_ver})"
        )
    return (major, minor, patch)


class DnamJsonHeader(StrictModel):
    """Top-level UTF-8 JSON header at byte offset `0x40` of a `.dnam` or `.dnam-ckpt` container."""

    version: str = CURRENT_SCHEMA_VERSION
    container_kind: Literal["dnam", "dnam-ckpt"] = "dnam"
    architecture: str = "DNAM_TriPhysics"
    sample_rate_train_hz: int = 48_000
    metadata: ModelMetadata
    user_metadata: dict[str, Any] = Field(default_factory=dict)
    io_config: IoConfig = Field(default_factory=IoConfig)
    slimmable_tiers: SlimmableTiersConfig
    controls_schema: list[ControlSchemaDef] = Field(default_factory=list)
    presets: list[PresetDef] = Field(default_factory=list)
    control_bus: ControlBusConfig = Field(default_factory=ControlBusConfig)
    core1_hybrid_ssm: Core1Config = Field(default_factory=Core1Config)
    core2_mufdn: Core2Config = Field(default_factory=Core2Config)
    tensor_directory: dict[str, TensorSliceEntry] = Field(default_factory=dict)
    stems_registry: list[StemRegistryEntry] | None = None
    audit_report: AuditReport | None = None

    @field_validator("version")
    @classmethod
    def _validate_version(cls, v: str) -> str:
        parse_schema_version(v)
        return v

    @field_validator("sample_rate_train_hz")
    @classmethod
    def _validate_sample_rate(cls, v: int) -> int:
        if not (44_100 <= v <= 96_000):
            raise DnamSampleRateError(
                f"sample_rate_train_hz ({v} Hz) must lie in [44100, 96000] Hz"
            )
        return v

    @field_validator("controls_schema")
    @classmethod
    def _validate_controls_count(cls, v: list[ControlSchemaDef]) -> list[ControlSchemaDef]:
        if len(v) > 16:
            raise DnamValidationError(
                f"controls_schema exceeds maximum of 16 controls (got {len(v)})"
            )
        seen: set[str] = set()
        for c in v:
            if c.id in seen:
                raise DnamValidationError(f"Duplicate control id '{c.id}' in controls_schema")
            seen.add(c.id)
        return v

    @classmethod
    def default_reference(cls, *, core2_active: bool = False) -> DnamJsonHeader:
        """Constructs a canonical reference `DnamJsonHeader`."""
        return cls(
            metadata=ModelMetadata(
                name="Reference DNAM Model",
                created_at="2026-10-23T00:00:00Z",
            ),
            slimmable_tiers=SlimmableTiersConfig(
                lite=TierProfile(
                    channels=4,
                    fdn_lines=4 if core2_active else 0,
                    early_taps=4,
                    arena_end_byte_offset=10_752,
                ),
                full=TierProfile(
                    channels=8,
                    fdn_lines=8 if core2_active else 0,
                    early_taps=8,
                    arena_end_byte_offset=22_336,
                ),
                ultra=TierProfile(
                    channels=12,
                    fdn_lines=8 if core2_active else 0,
                    early_taps=8,
                    arena_end_byte_offset=36_224,
                ),
            ),
            core2_mufdn=Core2Config(
                active=core2_active,
                ring_buffer_pow2_log2=10 if core2_active else 0,
            ),
        )
