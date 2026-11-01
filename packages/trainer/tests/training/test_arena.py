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
    layout.conv_weights_ultra_3072 = rng.standard_normal(3072).astype(np.float32) * 0.08
    layout.w_head_tiers = rng.standard_normal((4, 5, 2, 64)).astype(np.float32) * 0.05
    layout.w_head_ultra_modes5to8 = rng.standard_normal((4, 4, 2, 64)).astype(np.float32) * 0.03
    layout.b_head = rng.standard_normal((5, 2)).astype(np.float32) * 0.01
    layout.b_head_ultra_modes5to8 = rng.standard_normal((4, 2)).astype(np.float32) * 0.01
    layout.tier_gain_trim_lin = np.array([0.85, 0.92, 0.98, 1.05], dtype=np.float32)
    layout.p_svd_16x4 = rng.standard_normal((16, 4)).astype(np.float32) * 0.2
    layout.p_svd_ultra_16x4 = rng.standard_normal((16, 4)).astype(np.float32) * 0.15
    layout.mufdn_grain_len_8 = np.full(8, 2400.0, dtype=np.float32)
    layout.mufdn_grain_warp_8 = np.full(8, 0.5, dtype=np.float32)
    layout.mufdn_grain_meta_0 = np.array([0.8, 4.0, 0.25, 0.5], dtype=np.float32)
    layout.mufdn_grain_meta_1 = np.array([0.0, 0.2, 0.0, 0.0], dtype=np.float32)
    layout.w_ctrl_core2_8x32 = rng.standard_normal((8, 32)).astype(np.float32) * 0.05

    raw_bytes = layout.to_arena012_bytes()
    assert len(raw_bytes) == ARENA_OFFSET_ULTRA_END

    unpacked = ArenaWeightLayout.from_arena012_bytes(raw_bytes)
    assert np.allclose(unpacked.conv_weights_4608, layout.conv_weights_4608)
    assert np.allclose(unpacked.conv_weights_ultra_3072, layout.conv_weights_ultra_3072)
    assert np.allclose(unpacked.w_head_tiers, layout.w_head_tiers)
    assert np.allclose(unpacked.w_head_ultra_modes5to8, layout.w_head_ultra_modes5to8)
    assert np.allclose(unpacked.b_head, layout.b_head)
    assert np.allclose(unpacked.b_head_ultra_modes5to8, layout.b_head_ultra_modes5to8)
    assert np.allclose(unpacked.tier_gain_trim_lin, layout.tier_gain_trim_lin)
    assert np.allclose(unpacked.p_svd_16x4, layout.p_svd_16x4)
    assert np.allclose(unpacked.p_svd_ultra_16x4, layout.p_svd_ultra_16x4)
    assert np.allclose(unpacked.mufdn_delays_8, layout.mufdn_delays_8)
    assert np.allclose(unpacked.mufdn_delays_ultra_8, layout.mufdn_delays_ultra_8)
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
    layout_default_trim = ArenaWeightLayout.from_arena012_bytes(raw_bytes)
    layout_default_trim.tier_gain_trim_lin = np.ones(4, dtype=np.float32)
    base_hdr_trims = base_hdr.model_copy(
        update={
            "slimmable_tiers": base_hdr.slimmable_tiers.model_copy(
                update={
                    "lite": base_hdr.slimmable_tiers.lite.model_copy(
                        update={"output_trim_db": -1.5}
                    ),
                    "full": base_hdr.slimmable_tiers.full.model_copy(
                        update={"output_trim_db": -0.5}
                    ),
                    "pro": base_hdr.slimmable_tiers.pro.model_copy(update={"output_trim_db": 0.25}),
                    "ultra": base_hdr.slimmable_tiers.ultra.model_copy(
                        update={"output_trim_db": 0.0}
                    ),
                }
            )
        }
    )

    dnam_c, ckpt_c = build_signed_dnam_containers(
        base_hdr_trims,
        layout_default_trim,
        [stat],
        creator_seed_hex=seed_hex,
        signer_id="creator-pytest",
    )
    unpacked_built = ArenaWeightLayout.from_arena012_bytes(bytes(dnam_c.arena))
    assert np.allclose(
        unpacked_built.tier_gain_trim_lin,
        np.array(
            [10.0 ** (-1.5 / 20.0), 10.0 ** (-0.5 / 20.0), 10.0 ** (0.25 / 20.0), 1.0],
            dtype=np.float32,
        ),
    )
    assert not dnam_c.is_checkpoint
    assert ckpt_c.is_checkpoint
    assert dnam_c.verification == VerificationStatus.CREATOR_VERIFIED
    assert ckpt_c.verification == VerificationStatus.CREATOR_VERIFIED
    assert ckpt_c.preamble.num_ckpt_stems == 1

    # Verify structural hash & dual watermark compatibility (64-ch kappa + 7680-chip Tardos)
    f_master = dnam_c.structural_hash()
    assert f_master.shape == (224,)
    kappa_64 = tuple(1.0 + 0.005 * ((i % 5) - 2) for i in range(64))
    tardos_7680 = provenance.generate_tardos_code(
        b"\x5a" * 32, model_id="arena-test", license_id="lic-01"
    )
    wm_c = dnam_c.with_watermark(
        WatermarkParams(
            kappa_64=kappa_64,
            tardos_code_7680=tuple(float(x) for x in tardos_7680),
            tardos_epsilon=1e-4,
        )
    )
    assert provenance.structural_cosine_similarity(f_master, wm_c.structural_hash()) > 0.9999
    assert (
        provenance.score_tardos_watermark(
            bytes(dnam_c.arena), bytes(wm_c.arena), code_7680=tardos_7680
        )
        > 60.0
    )
    stripped_pro = dnam_c.strip_to_tier(2)
    assert stripped_pro.verification == VerificationStatus.CREATOR_VERIFIED
    assert (
        provenance.structural_cosine_similarity(f_master, stripped_pro.structural_hash())
        > 0.999999999
    )
