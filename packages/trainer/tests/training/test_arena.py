"""Unit tests for `dnam_trainer.training.arena` (`ArenaWeightLayout` 64B packing,
`Ed25519` signing, and `.dnam` / `.dnam-ckpt` container building)."""

from __future__ import annotations

import numpy as np
import pytest

from dnam_engine import provenance
from dnam_engine.schema import DnamJsonHeader, VerificationStatus, WatermarkParams
from dnam_trainer.training.arena import (
    ARENA_OFFSET_ULTRA_END,
    ArenaWeightLayout,
    build_signed_dnam_containers,
)
from dnam_trainer.training.kronecker_head import compute_stem_sufficient_statistics


def test_arena_weight_layout_roundtrip_and_signed_containers() -> None:
    rng = np.random.default_rng(77)
    layout = ArenaWeightLayout()
    layout.conv_weights_4608 = rng.standard_normal(4608).astype(np.float32) * 0.1
    layout.w_head_tiers = rng.standard_normal((3, 5, 2, 64)).astype(np.float32) * 0.05
    layout.b_head = rng.standard_normal((5, 2)).astype(np.float32) * 0.01
    layout.p_svd_16x4 = rng.standard_normal((16, 4)).astype(np.float32) * 0.2
    layout.mufdn_grain_len_8 = np.full(8, 2400.0, dtype=np.float32)
    layout.mufdn_grain_warp_8 = np.full(8, 0.5, dtype=np.float32)
    layout.mufdn_grain_meta_0 = np.array([0.8, 4.0, 0.25, 0.5], dtype=np.float32)
    layout.mufdn_grain_meta_1 = np.array([0.0, 0.2, 0.0, 0.0], dtype=np.float32)
    layout.w_ctrl_core2_8x32 = rng.standard_normal((8, 32)).astype(np.float32) * 0.05

    raw_bytes = layout.to_arena012_bytes()
    assert len(raw_bytes) == ARENA_OFFSET_ULTRA_END

    unpacked = ArenaWeightLayout.from_arena012_bytes(raw_bytes)
    assert np.allclose(unpacked.conv_weights_4608, layout.conv_weights_4608)
    assert np.allclose(unpacked.w_head_tiers, layout.w_head_tiers)
    assert np.allclose(unpacked.b_head, layout.b_head)
    assert np.allclose(unpacked.p_svd_16x4, layout.p_svd_16x4)
    assert np.allclose(unpacked.mufdn_delays_8, layout.mufdn_delays_8)
    assert np.allclose(unpacked.mufdn_grain_len_8, layout.mufdn_grain_len_8)
    assert np.allclose(unpacked.mufdn_grain_warp_8, layout.mufdn_grain_warp_8)
    assert np.allclose(unpacked.mufdn_grain_meta_0, layout.mufdn_grain_meta_0)
    assert np.allclose(unpacked.mufdn_grain_meta_1, layout.mufdn_grain_meta_1)
    assert np.allclose(unpacked.w_ctrl_core2_8x32, layout.w_ctrl_core2_8x32)

    with pytest.raises(ValueError, match="Arena buffer too small"):
        ArenaWeightLayout.from_arena012_bytes(b"\x00" * 1024)

    # Build signed .dnam and .dnam-ckpt containers and verify Ed25519 provenance + watermarking
    stat = compute_stem_sufficient_statistics(
        np.ones((256, 192), dtype=np.float32) * 0.1,
        np.ones((256, 2), dtype=np.float32) * 0.05,
        stem_id="stem_0",
        controls_meta={"drive": 5.0},
        input_pcm_sha256="aa" * 32,
        target_pcm_sha256="bb" * 32,
    )
    seed_hex = "42" * 32
    base_hdr = DnamJsonHeader.default_reference(core2_active=True)

    dnam_c, ckpt_c = build_signed_dnam_containers(
        base_hdr,
        layout,
        [stat],
        creator_seed_hex=seed_hex,
        signer_id="creator-pytest",
    )
    assert not dnam_c.is_checkpoint
    assert ckpt_c.is_checkpoint
    assert dnam_c.verification == VerificationStatus.CREATOR_VERIFIED
    assert ckpt_c.verification == VerificationStatus.CREATOR_VERIFIED
    assert ckpt_c.preamble.num_ckpt_stems == 1

    # Verify structural hash and dual watermark compatibility on built container
    f_master = dnam_c.structural_hash()
    assert f_master.shape == (160,)
    wm_c = dnam_c.with_watermark(WatermarkParams())
    assert provenance.structural_cosine_similarity(f_master, wm_c.structural_hash()) > 0.999999999
