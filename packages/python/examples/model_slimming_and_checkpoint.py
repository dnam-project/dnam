"""Example 3: `.dnam` Tier Slimming, In-Place 4 KB Header Editing, `.dnam-ckpt` & `GramMatrix`
(`examples/model_slimming_and_checkpoint.py`).

Run with:
  uv run --no-sync python packages/python/examples/model_slimming_and_checkpoint.py
"""

from __future__ import annotations

import numpy as np

import dnam
from dnam.schema import PresetDef


def main() -> None:
    """Demonstrates `Model.with_tier_limit`, `with_header`, `Checkpoint`, and `GramMatrix`."""
    model = dnam.Model.reference()

    # 1. Strip an Ultra (64,000 B arena) .dnam container down to Pro, Full, and Lite
    for tier in (dnam.Tier.ULTRA, dnam.Tier.PRO, dnam.Tier.FULL, dnam.Tier.LITE):
        slimmed = model.with_tier_limit(tier)
        print(
            f"Stripped Tier {tier.label:<5} -> File: {len(slimmed):>6} B "
            f"| Arena: {len(slimmed.arena_bytes):>6} B | Max Tier: {slimmed.max_tier.label}"
        )

    # 2. Edit presets and user_metadata in place within the 4 KB page boundary
    new_preset = PresetDef(
        id="studio_clean",
        name="Studio Clean",
        description="Added in-place without touching tensor arena",
        controls={"gain": 4.0, "boost": "off"},
    )
    updated_header = model.header.model_copy(
        update={
            "presets": [*model.header.presets, new_preset],
            "user_metadata": {"favorite": True},
        }
    )
    edited_model = model.with_header(updated_header)
    print(
        f"In-place 4 KB header edit | Presets: {list(edited_model.presets)} "
        f"| favorite={edited_model.header.user_metadata['favorite']}"
    )

    # 3. Validate a .dnam-ckpt Checkpoint and export a deployable runtime Model
    ckpt = dnam.Checkpoint.reference(include_calib_stats=True)
    verified_stems = ckpt.verify_stems()
    arena5_bytes = ckpt.verify_calib_stats()
    exported_lite = ckpt.export_model().with_tier_limit("lite")
    print(
        f"Checkpoint verified ({verified_stems} stem, {arena5_bytes} B DNCA) -> "
        f"Exported runtime {exported_lite!r}"
    )

    # 4. Pack/unpack an Ultra (256x256) GramMatrix in uplo='U' and extract Lite (64x64) submatrix
    idx = np.arange(1, 257, dtype=np.float64)
    dense_ultra = np.outer(idx, idx)
    gram = dnam.GramMatrix.from_dense(dense_ultra)
    sub_lite = gram.to_dense(submatrix_dim=64)
    assert np.array_equal(sub_lite, dense_ultra[:64, :64])
    print(f"GramMatrix uplo='U' Prefix Theorem verified: extracted {sub_lite.shape} from {gram!r}")


if __name__ == "__main__":
    main()
