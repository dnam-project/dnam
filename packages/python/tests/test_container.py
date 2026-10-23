"""Unit and property-based tests for `DnamContainer`, `GramMatrix`, ZOH rescaling, and `crc32c`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from numpy.typing import NDArray

from dnam_engine import (
    DnamContainer,
    DnamContainerError,
    DnamSampleRateError,
    DnamValidationError,
    GramMatrix,
    VerificationStatus,
    crc32c,
)
from dnam_engine.schema import DnamJsonHeader, PresetDef
from strategies import st_sample_rates, st_symmetric_gram_matrix


def test_container_reference_and_file_roundtrip(
    tmp_path: Path,
    ref_container_core1: DnamContainer,
    ref_container_core2: DnamContainer,
    ref_checkpoint_container: DnamContainer,
) -> None:
    assert not ref_container_core1.is_checkpoint
    assert ref_container_core1.verification == VerificationStatus.UNSIGNED
    assert ref_container_core1.preamble.header_capacity_bytes >= 3000
    assert len(ref_container_core1.arena) == 36_224
    assert "gain" in ref_container_core1.controls
    assert "boost" in ref_container_core1.controls
    assert "warm_slapback" in ref_container_core1.presets
    assert ref_checkpoint_container.is_checkpoint
    assert ref_container_core2.header.core2_mufdn.active

    out_path = ref_container_core2.save(tmp_path / "model.dnam")
    loaded = DnamContainer.from_file(out_path)
    assert bytes(loaded) == ref_container_core2.to_bytes()
    assert DnamContainer.from_bytes(bytes(loaded)).header == ref_container_core2.header

    built = DnamContainer.build(DnamJsonHeader.default_reference(core2_active=True))
    assert built.header.core2_mufdn.active


def test_container_in_place_header_update_and_corruption(
    ref_container_core1: DnamContainer,
) -> None:
    new_preset = PresetDef(id="crunch", name="Crunch", controls={"gain": 8.0, "boost": "on"})
    new_hdr = ref_container_core1.header.model_copy(
        update={"presets": [*ref_container_core1.header.presets, new_preset]}
    )
    updated = ref_container_core1.with_header(new_hdr)
    assert "crunch" in updated.presets
    assert updated.arena == ref_container_core1.arena
    clamped = updated.clamp_controls({"gain": 8.0})
    assert not clamped.any_clamped

    bad_bytes = bytearray(ref_container_core1.to_bytes())
    bad_bytes[0:4] = b"XXXX"
    with pytest.raises(DnamContainerError):
        DnamContainer.from_bytes(bad_bytes)


@given(sr=st_sample_rates)
def test_container_zoh_rescaling_property(
    ref_container_core2: DnamContainer,
    sr: int,
) -> None:
    zoh = ref_container_core2.zoh_params(float(sr))
    assert zoh.sample_rate_host_hz == sr
    assert abs(zoh.rho_sr - (48_000.0 / sr)) < 1e-4
    assert 0.9 < zoh.r_ac_host < 1.0
    assert 0.9 < zoh.r_dc_host < 1.0

    with pytest.raises(DnamSampleRateError):
        ref_container_core2.zoh_params(12_000.0)


@given(mat=st_symmetric_gram_matrix(dim=16))
def test_gram_matrix_pack_unpack_and_prefix_theorem(mat: NDArray[np.float64]) -> None:
    assert GramMatrix.packed_len(16) == 136
    packed = GramMatrix.pack(mat)
    assert packed.shape == (136,)
    unpacked = GramMatrix.unpack(packed, dim=16)
    np.testing.assert_allclose(unpacked, mat, atol=1e-12)

    # Nested Principal Submatrix Prefix Theorem: leading 8x8 block from prefix
    sub_8 = GramMatrix.unpack(packed, dim=8)
    np.testing.assert_allclose(sub_8, mat[:8, :8], atol=1e-12)


def test_gram_matrix_validation_and_crc32c() -> None:
    with pytest.raises(DnamValidationError):
        GramMatrix.packed_len(0)
    with pytest.raises(DnamValidationError):
        GramMatrix.pack(np.zeros((4, 5), dtype=np.float64))
    with pytest.raises(DnamValidationError):
        GramMatrix.unpack(np.zeros(10, dtype=np.float64), dim=0)
    with pytest.raises(DnamValidationError):
        GramMatrix.unpack(np.zeros((2, 2), dtype=np.float64), dim=2)
    with pytest.raises(DnamValidationError):
        GramMatrix.unpack(np.zeros(3, dtype=np.float64), dim=4)

    # Non-contiguous float32 packed array coercion path
    packed = GramMatrix.pack(np.eye(4, dtype=np.float64))
    unpacked = GramMatrix.unpack(packed.astype(np.float32), dim=4)
    np.testing.assert_allclose(unpacked, np.eye(4), atol=1e-6)

    c1 = crc32c(b"DNAM-TriPhysics-Test")
    c2 = crc32c(b"DNAM-TriPhysics-Test")
    assert c1 == c2
    assert c1 > 0
