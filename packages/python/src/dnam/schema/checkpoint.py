"""`.dnam-ckpt` stem registry, training provenance, and clamp audit report schemas
(`ref[sdk.python.schema.rust-parity]`).
"""

from __future__ import annotations

from dnam.schema._base import StrictModel, strict_field
from dnam.schema.controls import JsonValue
from dnam.schema.stages import TensorSliceEntry


class StemProvenance(StrictModel):
    """Platform-agnostic IEEE-754 PCM SHA-256 provenance for a training stem."""

    input_pcm_sha256: str = ""
    target_pcm_sha256: str = ""
    num_frames: int = strict_field(default=0, ge=0)


class ExtractedPhysics(StrictModel):
    """Extracted Stage-1 physical system-identification anchors for a stem."""

    delays_ms: list[float] = strict_field(default_factory=list)
    lfo_freqs_hz: list[float] = strict_field(default_factory=list)
    decay_gains_lin: list[float] = strict_field(default_factory=list)
    detected_grain_size_ms: float = 0.0
    detected_pitch_shift_semitones: float = 0.0


class StemRegistryEntry(StrictModel):
    """`.dnam-ckpt` per-stem registry record (`stems_registry[*]`)."""

    stem_id: str
    controls_meta: dict[str, JsonValue] = strict_field(default_factory=dict)
    provenance: StemProvenance = strict_field(default_factory=StemProvenance)
    stem_arena4_crc32c: int = strict_field(default=0, ge=0)
    user_metadata: dict[str, JsonValue] = strict_field(default_factory=dict)
    extracted_physics: ExtractedPhysics | None = None
    g_audio_uplo_u_f64: TensorSliceEntry
    c_audio_tier_f64: TensorSliceEntry


class AuditReport(StrictModel):
    """`.dnam-ckpt` 3-phase control clamp audit report and actionable stem prescriptions."""

    phase1_clamps: list[str] = strict_field(default_factory=list)
    phase2_rail_warnings: list[str] = strict_field(default_factory=list)
    phase3_clamps: list[str] = strict_field(default_factory=list)
    prescriptions: list[str] = strict_field(default_factory=list)
