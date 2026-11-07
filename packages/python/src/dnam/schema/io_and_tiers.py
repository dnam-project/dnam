"""Audio I/O topologies, leveling configurations, and slimmable tier profiles
(`ref[sdk.python.schema.rust-parity]`).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from dnam.schema._base import StrictModel, strict_field


class BlendMode(StrEnum):
    """Host dry/wet blend capability mode (`IoConfig.capabilities.blend_mode`)."""

    NONE = "none"
    HOST_PARALLEL = "host_parallel"
    MODELED_CONTROL = "modeled_control"


class ModelCapabilities(StrictModel):
    """Model-level physical capabilities (`IoConfig.capabilities`, covered by
    `immutable_header_sha256`).
    """

    blend_mode: BlendMode = BlendMode.NONE
    default_blend: float = strict_field(default=1.0, ge=0.0, le=1.0)
    wet_polarity_inverted: bool = False
    supports_trails: bool = False
    tempo_sync_capable: bool = False
    supports_freeze: bool = False
    freeze_mutes_dry: bool = False


class LevelingMetric(StrEnum):
    """
    Loudness measurement metric used for auto-unity leveling (`LevelingConfig.leveling_metric`).
    """

    RMS_DBFS = "rms_dbfs"
    K_WEIGHTED_LUFS = "k_weighted_lufs"


class LevelingConfig(StrictModel):
    """Auto-unity gain leveling calibration anchored at the default control settings."""

    auto_unity_enabled: bool = True
    leveling_metric: LevelingMetric = LevelingMetric.RMS_DBFS
    reference_input_level_db: float = -20.5
    measured_default_output_level_db: float = -20.5
    unity_gain_offset_db: float = 0.0
    target_gain_db: float = 0.0
    calibrated_trim_db: float = 0.0


class IoConfig(StrictModel):
    """Audio I/O topology, host PDC / tail configuration, model capabilities, and auto-unity
    leveling.
    """

    inputs: Literal[1, 2] = 2
    output_mode: Literal["stereo", "mono"] = "stereo"
    latency_samples: int = strict_field(default=0, ge=0)
    tail_time_ms: float = strict_field(default=0.0, ge=0.0)
    capabilities: ModelCapabilities = strict_field(default_factory=ModelCapabilities)
    leveling: LevelingConfig = strict_field(default_factory=LevelingConfig)


class TierProfile(StrictModel):
    """Metadata and prefix checksum for an individual slimmable tier (`Lite`, `Full`, `Pro`,
    `Ultra`).
    """

    feature_bus_dim: int = strict_field(default=0, ge=0)
    channels: int = strict_field(ge=1, le=64)
    early_fir_taps: int = strict_field(default=0, ge=0)
    kautz_modes: int = strict_field(default=0, ge=0)
    tank_lines: int = strict_field(ge=0)
    head_svd_rank: int = strict_field(default=4, ge=1, le=16)
    loudness_lufs: float = -20.5
    output_trim_db: float = 0.0
    validation_esr_db: float = 0.0
    arena_end_byte_offset: int = strict_field(ge=0)
    prefix_crc32c: int = strict_field(default=0, ge=0)


def _default_pro_tier_profile() -> TierProfile:
    return TierProfile(
        feature_bus_dim=192,
        channels=12,
        early_fir_taps=48,
        kautz_modes=16,
        tank_lines=8,
        head_svd_rank=4,
        loudness_lufs=-20.5,
        output_trim_db=0.0,
        validation_esr_db=-48.0,
        arena_end_byte_offset=36_224,
        prefix_crc32c=0,
    )


class SlimmableTiersConfig(StrictModel):
    """Slimmable tier profiles (`lite`, `full`, `pro`, `ultra`)."""

    lite: TierProfile
    full: TierProfile
    pro: TierProfile = strict_field(default_factory=_default_pro_tier_profile)
    ultra: TierProfile
