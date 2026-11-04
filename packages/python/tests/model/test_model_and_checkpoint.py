"""Tests for `dnam.model` (`Model`, `Checkpoint`, `Preamble`, `GramMatrix`, `crc32c`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from numpy.typing import NDArray

import dnam
from dnam.errors import ContainerError, ProvenanceError, ValidationError
from dnam.schema import DnamProtectionProfile, PresetDef, Tier, VerificationStatus
from strategies import symmetric_matrix_strategy, tier_strategy


def test_model_load_save_header_edit_and_tier_stripping(
    reference_model: dnam.Model,
    tmp_path: Path,
) -> None:
    """verify[sdk.python.model.containers-and-checkpoints] verify[sdk.python.api.core-types]
    Verifies Model load/save round-trip, in-place 4 KB header editing, tier prefix stripping,
    and informative reprs.
    """
    assert reference_model.name == "Reference DNAM Model"
    assert reference_model.max_tier == Tier.ULTRA
    assert reference_model.verification == VerificationStatus.UNSIGNED
    assert reference_model.protection_profile == DnamProtectionProfile.OPEN
    assert len(reference_model.controls) == 3
    assert "gain" in reference_model.controls
    assert "warm_slapback" in reference_model.presets
    assert "Reference DNAM Model" in repr(reference_model)
    assert bytes(reference_model) == reference_model.to_bytes()
    assert len(reference_model) == len(reference_model.to_bytes())

    # Save and load round-trip
    out_file = tmp_path / "test_model.dnam"
    saved_path = reference_model.save(out_file)
    assert saved_path == out_file
    reloaded = dnam.Model.load(out_file)
    assert reloaded.to_bytes() == reference_model.to_bytes()

    # Tier arena views
    for t in (Tier.LITE, Tier.FULL, Tier.PRO, Tier.ULTRA):
        view = reference_model.tier_arena(t)
        assert len(view) == t.arena_end_byte_offset

    # Strip down to each tier and verify
    for t in (Tier.PRO, Tier.FULL, Tier.LITE):
        stripped = reference_model.with_tier_limit(t)
        assert stripped.max_tier == t
        assert len(stripped.arena_bytes) == t.arena_end_byte_offset
        with pytest.raises(ValidationError):
            stripped.tier_arena(Tier.ULTRA)

    # In-place header edit within 4 KB page
    new_preset = PresetDef(
        id="crunch_lead",
        name="Crunch Lead",
        description="Added in-place",
        controls={"gain": 8.0, "boost": "on"},
    )
    updated_hdr = reference_model.header.model_copy(
        update={
            "presets": [*reference_model.header.presets, new_preset],
            "user_metadata": {"author_tag": "studio"},
        }
    )
    edited = reference_model.with_header(updated_hdr)
    assert "crunch_lead" in edited.presets
    assert edited.header.user_metadata["author_tag"] == "studio"

    # Oversized header (> 4 KB) fails with ContainerError
    huge_hdr = reference_model.header.model_copy(update={"user_metadata": {"blob": "x" * 6000}})
    with pytest.raises(ContainerError):
        reference_model.with_header(huge_hdr)


def test_model_corruption_and_preview_rejection(reference_model: dnam.Model) -> None:
    """verify[sdk.python.model.containers-and-checkpoints] verify[sdk.python.errors.typed-hierarchy]
    Verifies container corruption detection (short buffer, bad magic, CRC32C mismatch) and
    preview container (`DNPR`) rejection via `ProvenanceError`.
    """
    with pytest.raises(ContainerError):
        dnam.Model.from_bytes(b"short")

    bad_magic = bytearray(reference_model.to_bytes())
    bad_magic[0:4] = b"BAD!"
    with pytest.raises(ContainerError):
        dnam.Model.from_bytes(bad_magic)

    bad_crc = bytearray(reference_model.to_bytes())
    bad_crc[5000] ^= 0xFF
    with pytest.raises(ContainerError):
        dnam.Model.from_bytes(bad_crc)

    # Preview container magic 'DNPR' or header kind 'preview' raises ProvenanceError
    dnpr = bytearray(reference_model.to_bytes())
    dnpr[0:4] = b"DNPR"
    with pytest.raises((ContainerError, ProvenanceError)):
        dnam.Model.from_bytes(dnpr)

    # Model.build validation
    built = dnam.Model.build(reference_model.header, reference_model.arena_bytes)
    assert built.name == reference_model.name
    ckpt_hdr = reference_model.header.model_copy(update={"container_kind": "dnam-ckpt"})
    with pytest.raises(ValidationError):
        dnam.Model.build(ckpt_hdr, reference_model.arena_bytes)

    # DMU1 mutable frame round-trip
    frame = dnam.Model.build_dmu1_frame('{"presets":[],"user_metadata":{"k":"v"}}', 128)
    assert len(frame) == 128
    parsed = dnam.Model.parse_dmu1_frame(frame)
    assert '"k":"v"' in parsed


def test_checkpoint_validation_stems_calib_and_export(
    reference_checkpoint: dnam.Checkpoint,
    reference_model: dnam.Model,
    tmp_path: Path,
) -> None:
    """verify[sdk.python.model.containers-and-checkpoints] verify[sdk.python.api.core-types]
    Verifies Checkpoint loading, stem Arena 4 CRC32C verification, Arena 5 DNCA verification,
    and runtime Model export.
    """
    assert reference_checkpoint.preamble.magic == "DNCK"
    assert reference_checkpoint.preamble.has_calib_stats
    assert len(reference_checkpoint.stems) == 1
    assert reference_checkpoint.audit_report is not None
    assert reference_checkpoint.arena4_bytes is not None
    assert "Checkpoint" in repr(reference_checkpoint)
    assert bytes(reference_checkpoint) == reference_checkpoint.to_bytes()
    assert len(reference_checkpoint) == len(reference_checkpoint.to_bytes())

    # Loading a checkpoint with Model.from_bytes or vice-versa raises ContainerError
    with pytest.raises(ContainerError):
        dnam.Model.from_bytes(reference_checkpoint.to_bytes())
    with pytest.raises(ContainerError):
        dnam.Checkpoint.from_bytes(reference_model.to_bytes())

    assert reference_checkpoint.verify_stems() == 1
    assert reference_checkpoint.verify_calib_stats() == 64

    # Save & reload checkpoint
    ckpt_path = tmp_path / "run.dnam-ckpt"
    reference_checkpoint.save(ckpt_path)
    reloaded = dnam.Checkpoint.load(ckpt_path)
    assert reloaded.to_bytes() == reference_checkpoint.to_bytes()

    # Export runtime Model and strip to Lite
    exported = reference_checkpoint.export_model()
    assert isinstance(exported, dnam.Model)
    assert exported.preamble.magic == "DNAM"
    lite = exported.with_tier_limit("lite")
    assert lite.max_tier == Tier.LITE

    # Checkpoint.build
    rebuilt = dnam.Checkpoint.build(reference_checkpoint.header, reference_checkpoint.arena_bytes)
    assert len(rebuilt.stems) == 1
    with pytest.raises(ValidationError):
        dnam.Checkpoint.build(reference_model.header, reference_model.arena_bytes)


@given(mat=symmetric_matrix_strategy(dim=8))
@settings(max_examples=25, deadline=None)
def test_gram_matrix_uplo_u_roundtrip_and_prefix_submatrix(
    mat: NDArray[np.float64],
) -> None:
    """verify[sdk.python.model.containers-and-checkpoints]
    Verifies GramMatrix `uplo='U'` packing/unpacking and the Nested Principal Submatrix Prefix
    Theorem.
    """
    gram = dnam.GramMatrix.from_dense(mat)
    assert gram.dim == 8
    assert gram.packed.shape == (dnam.GramMatrix.packed_length(8),)
    assert "GramMatrix" in repr(gram)
    assert len(gram.to_bytes()) == 36 * 8

    unpacked = gram.to_dense()
    np.testing.assert_allclose(unpacked, mat, rtol=0.0, atol=0.0)

    # Leading 4x4 principal submatrix extracted from prefix
    sub4 = gram.to_dense(submatrix_dim=4)
    np.testing.assert_allclose(sub4, mat[:4, :4], rtol=0.0, atol=0.0)

    from_packed = dnam.GramMatrix.from_packed(gram.packed, 8)
    np.testing.assert_allclose(from_packed.to_dense(), mat, rtol=0.0, atol=0.0)


def test_gram_matrix_and_crc32c_validation() -> None:
    """verify[sdk.python.model.containers-and-checkpoints]
    Verifies GramMatrix validation errors and hardware CRC32C determinism.
    """
    c1 = dnam.crc32c(b"123456789")
    assert c1 == 0xE3069283
    with pytest.raises(ValidationError):
        dnam.GramMatrix.packed_length(-1)
    with pytest.raises(ValidationError):
        dnam.GramMatrix.from_dense(np.zeros((3, 4)))
    with pytest.raises(ValidationError):
        dnam.GramMatrix.from_packed(np.zeros(5), 4)
    g = dnam.GramMatrix.from_dense(np.eye(4))
    with pytest.raises(ValidationError):
        g.to_dense(submatrix_dim=5)


@given(tier=tier_strategy)
@settings(max_examples=8, deadline=None)
def test_tier_properties_and_parsing(tier: Tier) -> None:
    """verify[sdk.python.api.flexible-in-precise-out]
    Verifies Tier.parse accepts enum, int, and string labels and exposes channel/bus dimensions.
    """
    assert Tier.parse(tier) == tier
    assert Tier.parse(int(tier)) == tier
    assert Tier.parse(tier.label) == tier
    assert Tier.parse(tier.label.upper()) == tier
    assert tier.channels in (4, 8, 12, 16)
    assert tier.feature_bus_dim in (64, 128, 192, 256)
    assert tier.arena_end_byte_offset in (10_752, 22_336, 36_224, 64_000)
