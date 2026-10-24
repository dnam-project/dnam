"""Container inspector (`inspect_dnam_container`) and offline `DnamEngine` WAV renderer."""

from __future__ import annotations

from pathlib import Path

from dnam_engine import DnamContainer, DnamEngine
from dnam_trainer.capture.excitation import read_wav, write_wav_24bit
from dnam_trainer.schema import (
    AuditReport,
    ControlCoords,
    ControlSchemaDef,
    PresetDef,
    SlimmableTiersConfig,
    StrictModel,
    Tier,
)


class ContainerInspectionSummary(StrictModel):
    """Structured inspection summary of a `.dnam` or `.dnam-ckpt` container."""

    path: str
    magic: str
    is_checkpoint: bool
    model_name: str
    author: str
    sample_rate_train_hz: int
    verification_status: str
    core2_active: bool
    lfo_env_active: bool
    num_controls: int
    num_ckpt_stems: int
    head_svd_rank: int
    tail_time_ms: float
    tensor_arena_byte_len: int
    slimmable_tiers: SlimmableTiersConfig
    controls: list[ControlSchemaDef]
    presets: list[PresetDef]
    audit_report: AuditReport | None = None


def inspect_dnam_container(container_path: str | Path) -> ContainerInspectionSummary:
    """
    Loads and validates a `.dnam` or `.dnam-ckpt` container and extracts its inspection summary.
    """
    p = Path(container_path).resolve()
    container = DnamContainer.from_file(p)
    pre = container.preamble
    hdr = container.header
    flags = pre.flags
    core2_active = bool(flags & (1 << 6)) or hdr.core2_mufdn.active
    lfo_env_active = bool(flags & (1 << 7)) or (
        hdr.control_bus.envelopes > 0 or hdr.control_bus.lfo_oscillators > 0
    )

    return ContainerInspectionSummary(
        path=str(p),
        magic=pre.magic,
        is_checkpoint=container.is_checkpoint,
        model_name=hdr.metadata.name,
        author=hdr.metadata.author or "",
        sample_rate_train_hz=pre.sample_rate_train_hz,
        verification_status=str(container.verification),
        core2_active=core2_active,
        lfo_env_active=lfo_env_active,
        num_controls=pre.num_controls,
        num_ckpt_stems=pre.num_ckpt_stems,
        head_svd_rank=pre.head_svd_rank,
        tail_time_ms=pre.tail_time_ms,
        tensor_arena_byte_len=pre.tensor_arena_byte_len,
        slimmable_tiers=hdr.slimmable_tiers,
        controls=list(hdr.controls_schema),
        presets=list(hdr.presets),
        audit_report=hdr.audit_report,
    )


def render_with_dnam_engine(
    container_path: str | Path,
    input_wav_path: str | Path,
    output_wav_path: str | Path,
    *,
    tier: Tier | str = Tier.ULTRA,
    preset: str | None = None,
    controls: ControlCoords | None = None,
) -> Path:
    """Renders `input_wav_path` through the Rust SIMD `DnamEngine` and writes a 24-bit WAV."""
    container = DnamContainer.from_file(container_path)
    dry_audio, sample_rate_hz = read_wav(input_wav_path)
    if isinstance(tier, str):
        tier_map = {"lite": Tier.LITE, "full": Tier.FULL, "ultra": Tier.ULTRA}
        resolved_tier = tier_map.get(tier.lower(), Tier.ULTRA)
    else:
        resolved_tier = tier

    engine = DnamEngine(
        container,
        tier=resolved_tier,
        sample_rate=float(sample_rate_hz),
    )
    if preset is not None:
        engine.controls.apply_preset(preset, immediate=True)
    if controls:
        engine.controls.snap(controls)

    rendered = engine(dry_audio)
    return write_wav_24bit(output_wav_path, rendered, sample_rate_hz=sample_rate_hz)


__all__ = [
    "ContainerInspectionSummary",
    "inspect_dnam_container",
    "render_with_dnam_engine",
]
