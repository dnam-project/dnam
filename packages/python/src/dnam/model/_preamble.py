"""64-byte binary `.dnam` / `.dnam-ckpt` preamble model
(`ref[sdk.python.model.containers-and-checkpoints]`).
"""

from __future__ import annotations

from dnam.schema._base import StrictModel, Tier


class Preamble(StrictModel):
    """Parsed 64-byte binary preamble of a `.dnam` or `.dnam-ckpt` container."""

    magic: str
    version_major: int
    version_minor: int
    flags: int
    num_controls: int
    sample_rate_train_hz: int
    json_header_byte_len: int
    tensor_arena_offset: int
    tensor_arena_byte_len: int
    arena_lite_end_cl: int
    arena_full_end_cl: int
    arena_pro_end_cl: int
    arena_ultra_end_cl: int
    arena_offset_lite_end: int
    arena_offset_full_end: int
    arena_offset_pro_end: int
    arena_offset_ultra_end: int
    arena_crc32c_lite: int
    arena_crc32c_full: int
    arena_crc32c_pro: int
    arena_crc32c_ultra: int
    num_ckpt_stems: int
    tail_time_ms: float
    header_crc32c: int
    highest_tier: Tier
    is_stereo_out: bool
    has_signatures: bool
    is_encrypted_arena: bool
    has_calib_stats: bool
