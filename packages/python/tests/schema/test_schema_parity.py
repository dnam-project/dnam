"""Tests for `dnam.schema` Rust-to-Python JCS round-trip parity and `dnam.errors` exhaustive
mapping.
"""

from __future__ import annotations

import pytest

from dnam._native import _testing
from dnam.errors import (
    ContainerError,
    ControlError,
    DnamError,
    ProvenanceError,
    SampleRateError,
    ValidationError,
)
from dnam.provenance import immutable_header_sha256
from dnam.schema import (
    Backend,
    BlendMode,
    ControlCalibration,
    ControlCalibrationRole,
    ControlUiHints,
    DistributionPolicy,
    DnamJsonHeader,
    DnamProtectionProfile,
    SchemaVersion,
    Stage4SpatialTankConfig,
    TempoSubdivision,
    Tier,
)


@pytest.mark.parametrize("kind", ["default", "full"])
def test_rust_to_python_header_jcs_roundtrip_parity(kind: str) -> None:
    """verify[sdk.python.schema.rust-parity]
    Verifies that Rust-emitted DnamJsonHeader (both default_reference and fully populated)
    round-trips through Python DnamJsonHeader to byte-identical RFC 8785 JCS and identical
    immutable_header_sha256 digests.
    """
    rust_json = _testing.reference_header_json(kind)
    parsed = DnamJsonHeader.from_json(rust_json)

    # Full JCS and projected immutable JCS must match byte-for-byte
    rust_jcs = DnamJsonHeader.from_json(parsed.to_json()).to_jcs()
    assert parsed.to_jcs() == rust_jcs
    assert parsed.to_jcs(project_immutable=True) == DnamJsonHeader.from_json(rust_json).to_jcs(
        project_immutable=True
    )
    assert immutable_header_sha256(parsed) == immutable_header_sha256(rust_json)


def test_exhaustive_dnam_error_variant_mapping() -> None:
    """verify[sdk.python.errors.typed-hierarchy]
    Iterates over every DnamError variant name emitted by Rust and asserts it maps to the
    exact dnam.errors exception subclass.
    """
    expected_classes: dict[str, type[DnamError]] = {
        "BufferTooSmall": ContainerError,
        "InvalidMagic": ContainerError,
        "UnsupportedVersion": ContainerError,
        "HeaderCapacityExceeded": ContainerError,
        "Crc32cMismatch": ContainerError,
        "InvalidSampleRate": SampleRateError,
        "InvalidControlIndex": ControlError,
        "CryptoVerificationFailed": ProvenanceError,
        "EncryptedArenaRequiresDeviceKey": ProvenanceError,
        "PreviewContainerRejected": ProvenanceError,
        "UnalignedOffset": ValidationError,
        "InvalidJsonHeader": ValidationError,
        "SchemaValidationError": ValidationError,
    }
    variant_names = _testing.error_variant_names()
    assert set(variant_names) == set(expected_classes)

    for name in variant_names:
        expected_cls = expected_classes[name]
        with pytest.raises(expected_cls) as exc_info:
            _testing.raise_error_variant(name)
        assert isinstance(exc_info.value, DnamError)
        assert not str(exc_info.value).startswith("[DNAM-")


def test_strict_model_immutability_and_schema_helpers() -> None:
    """verify[sdk.python.schema.rust-parity] verify[sdk.python.api.flexible-in-precise-out]
    Verifies StrictModel frozen/extra='forbid' enforcement, SchemaVersion, ControlUiHints skew,
    ControlCalibration Lobatto inversion, DistributionPolicy, and Stage4 allpass validation.
    """
    ver = SchemaVersion.parse("1.1.0")
    assert ver.semver == "1.1.0"
    assert ver.is_supported
    assert not SchemaVersion.parse("2.0.0").is_supported
    with pytest.raises(ValidationError):
        SchemaVersion.parse("invalid")
    with pytest.raises(ValidationError):
        SchemaVersion.parse("1.x.0")

    # Extra fields forbidden & frozen enforced
    with pytest.raises(ValidationError):
        SchemaVersion.model_validate({"major": 1, "minor": 0, "unknown": 1})
    with pytest.raises(ValidationError):
        DnamJsonHeader.from_json("{bad json")

    # UI hints power-skew round-trip
    hints = ControlUiHints(midpoint=2.5)
    phys = hints.normalized_to_physical(0.5, 0.0, 10.0)
    assert abs(phys - 2.5) < 1e-4
    norm = hints.physical_to_normalized(phys, 0.0, 10.0)
    assert abs(norm - 0.5) < 1e-4
    assert hints.normalized_to_physical(0.0, 0.0, 10.0) == 0.0
    assert hints.normalized_to_physical(1.0, 0.0, 10.0) == 10.0
    assert hints.physical_to_normalized(0.0, 5.0, 5.0) == 0.0

    # ControlCalibration Lobatto evaluation & tempo sync resolution
    cal = ControlCalibration(
        role=ControlCalibrationRole.DELAY_TIME_MS,
        lobatto_physical_anchors=(20.0, 350.0, 1000.0),
        tempo_sync_default_subdivision=TempoSubdivision.EIGHTH,
    )
    mid_v = cal.evaluate_lobatto_anchors(0.0)
    assert mid_v == 350.0
    res = cal.resolve_tempo_sync(120.0, "1/8")
    assert res is not None
    assert not res.was_clamped
    ev_back = cal.evaluate_lobatto_anchors(res.normalized_u)
    assert ev_back is not None
    assert abs(ev_back - 250.0) < 1e-2

    # Conditional anchors resolution
    cond_cal = ControlCalibration(
        role=ControlCalibrationRole.DELAY_TIME_MS,
        conditional_physical_anchors={
            "range=short": (20.0, 100.0, 200.0),
            "range=long": (200.0, 600.0, 1000.0),
        },
    )
    cond_res = cond_cal.resolve_tempo_sync(120.0, "1/4")
    assert cond_res is not None
    assert cond_res.switch_selection == ("range", "long")

    # Stage4 allpass delays accepts 8 or 16 elements
    s4 = Stage4SpatialTankConfig(allpass_delays_samples=[10, 20, 30, 40, 50, 60, 70, 80])
    assert len(s4.allpass_delays_samples) == 16
    with pytest.raises(ValidationError):
        Stage4SpatialTankConfig(allpass_delays_samples=[1, 2, 3])
    with pytest.raises(ValidationError):
        Stage4SpatialTankConfig.model_validate({"allpass_delays_samples": 123})

    # Extra ControlCalibration branches (octave folding, descending/linear anchors, LFO role)
    assert ControlCalibration.invert_lobatto_anchors((10.0, 10.0, 10.0), 10.0) == (
        0.0,
        True,
    )
    u_fold_hi, _ = ControlCalibration.invert_lobatto_anchors(
        (20.0, 100.0, 200.0), 400.0, fold_octave=True
    )
    assert abs(u_fold_hi - 1.0) < 1e-5
    u_fold_lo, _ = ControlCalibration.invert_lobatto_anchors(
        (20.0, 100.0, 200.0), 10.0, fold_octave=True
    )
    assert abs(u_fold_lo - (-1.0)) < 1e-5
    u_desc_lo, clamped_lo = ControlCalibration.invert_lobatto_anchors((200.0, 100.0, 20.0), 5.0)
    assert u_desc_lo == 1.0
    assert clamped_lo
    u_desc_hi, clamped_hi = ControlCalibration.invert_lobatto_anchors((200.0, 100.0, 20.0), 300.0)
    assert u_desc_hi == -1.0
    assert clamped_hi
    u_lin, clamped_lin = ControlCalibration.invert_lobatto_anchors((0.0, 50.0, 100.0), 75.0)
    assert abs(u_lin - 0.5) < 1e-5
    assert not clamped_lin

    lfo_cal = ControlCalibration(
        role=ControlCalibrationRole.LFO_RATE_HZ,
        lobatto_physical_anchors=(0.5, 2.0, 8.0),
    )
    assert lfo_cal.resolve_tempo_sync(120.0, "1/4") is not None
    fb_cal = ControlCalibration(role=ControlCalibrationRole.FEEDBACK)
    assert fb_cal.evaluate_lobatto_anchors(0.0) is None
    assert fb_cal.resolve_tempo_sync(120.0) is None

    # UI hints edge cases
    assert ControlUiHints().skew_exponent(10.0, 10.0) == 1.0
    assert ControlUiHints().physical_to_normalized(10.0, 0.0, 10.0) == 1.0
    assert ControlUiHints().physical_to_normalized(-1.0, 0.0, 10.0) == 0.0

    # DistributionPolicy checks
    pol = DistributionPolicy(
        allowed_protection_profiles=[DnamProtectionProfile.DEVICE_ALLOWLISTED],
        allowed_oem_vendors=["darkglass:anagram_rk3582", "neural"],
    )
    assert pol.is_profile_allowed("device_allowlisted")
    assert not pol.is_profile_allowed("open")
    assert pol.is_oem_device_allowed("darkglass", "anagram_rk3582")
    assert not pol.is_oem_device_allowed("darkglass", "other")
    assert pol.is_oem_device_allowed("neural", "quad_cortex")
    assert not pol.is_oem_device_allowed("other", "model")

    # Enum parsing
    assert Backend.parse("arm_neon") == Backend.NEON
    assert Backend.parse("avx2") == Backend.X86_SIMD
    with pytest.raises(ValidationError):
        Backend.parse("invalid_backend")
    with pytest.raises(ValidationError):
        Tier.parse(True)
    with pytest.raises(ValidationError):
        Tier.parse(99)
    with pytest.raises(ValidationError):
        Tier.parse("invalid_tier")
    assert BlendMode.HOST_PARALLEL.value == "host_parallel"
