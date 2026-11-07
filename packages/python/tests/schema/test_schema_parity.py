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

    # Extra fields forbidden & frozen/slots enforced
    assert not hasattr(ver, "__dict__")
    assert not hasattr(ver, "model_copy")
    assert not hasattr(ver, "model_dump")
    assert not hasattr(SchemaVersion, "model_validate")
    with pytest.raises(AttributeError):
        ver.__setattr__("major", 2)
    assert ver.replace() is ver
    ver_patched = ver.replace(patch=3)
    assert ver_patched.semver == "1.1.3"
    with pytest.raises(ValidationError):
        ver.replace(unknown_field=1)
    assert SchemaVersion.from_json(ver_patched.to_json()) == ver_patched
    assert SchemaVersion.from_dict(ver_patched.to_dict()) == ver_patched

    with pytest.raises(ValidationError):
        SchemaVersion.from_dict({"major": 1, "minor": 0, "unknown": 1})
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
        Stage4SpatialTankConfig.from_dict({"allpass_delays_samples": 123})

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


def test_strict_model_validation_and_bounds_branches() -> None:
    """Exercises all StrictModel type coercion, numeric/length bounds, and JSON/dict error paths."""
    from typing import Any, cast

    from dnam.provenance import (
        build_creator_signing_message,
        build_marketplace_signing_message,
        ed25519_sign,
    )
    from dnam.schema import (
        ControlBusConfig,
        ControlSchemaDef,
        ModelCapabilities,
        ModelMetadata,
        PresetDef,
        Stage2CascadeConfig,
        VerificationStatus,
        WatermarkParams,
    )

    hdr = DnamJsonHeader.from_json(_testing.reference_header_json("default"))
    assert hdr.schema_version.semver == "1.0.0"
    assert isinstance(hdr.to_rust_dict(), dict)
    assert isinstance(hdr.to_json(by_alias=False), str)
    with pytest.raises(ValidationError):
        DnamJsonHeader.from_json(b"\xff\xfe")

    # StrictModel.from_dict / from_json error branches
    with pytest.raises(ValidationError):
        SchemaVersion.from_dict(cast(Any, 123))
    with pytest.raises(ValidationError):
        ModelMetadata.from_dict({"name": "OnlyName"})
    with pytest.raises(ValidationError):
        SchemaVersion.from_json("{invalid json")
    with pytest.raises(ValidationError):
        SchemaVersion.from_json("[1, 2, 3]")

    # Type coercion error branches
    with pytest.raises(ValidationError):
        SchemaVersion(major=True)
    with pytest.raises(ValidationError):
        SchemaVersion(major=cast(Any, "not_an_int"))
    assert SchemaVersion(major=cast(Any, 1.0)).major == 1
    with pytest.raises(ValidationError):
        ModelCapabilities(supports_trails=cast(Any, "yes"))
    with pytest.raises(ValidationError):
        ControlBusConfig(head_trust_region_radius=True)
    with pytest.raises(ValidationError):
        ControlBusConfig(head_trust_region_radius=float("nan"))
    with pytest.raises(ValidationError):
        ModelMetadata(name=cast(Any, 123), created_at="2026-01-01T00:00:00Z")
    with pytest.raises(ValidationError):
        ControlSchemaDef(id="g", name="G", type=cast(Any, True), default=0.0)
    with pytest.raises(ValidationError):
        ControlSchemaDef(id="g", name="G", type=cast(Any, "invalid_kind"), default=0.0)
    with pytest.raises(ValidationError):
        ControlSchemaDef(id="g", name="G", type=cast(Any, "range"), default=cast(Any, [1, 2]))
    with pytest.raises(ValidationError):
        ControlSchemaDef(
            id="g",
            name="G",
            type=cast(Any, "range"),
            default=0.0,
            safe_clamped_range=cast(Any, "not_tuple"),
        )
    with pytest.raises(ValidationError):
        ControlSchemaDef(
            id="g",
            name="G",
            type=cast(Any, "range"),
            default=0.0,
            safe_clamped_range=cast(Any, (1.0,)),
        )
    with pytest.raises(ValidationError):
        PresetDef(id="p", name="P", controls=cast(Any, "not_a_dict"))
    with pytest.raises(ValidationError):
        hdr.replace(metadata=123)

    # Numeric & length bounds (`ge`, `le`, `gt`, `lt`, `min_length`, `max_length`)
    with pytest.raises(ValidationError):
        SchemaVersion(major=-1)
    with pytest.raises(ValidationError):
        ControlBusConfig(envelopes=99)
    with pytest.raises(ValidationError):
        Stage2CascadeConfig(ac_coupling_pole=0.0)
    with pytest.raises(ValidationError):
        Stage2CascadeConfig(ac_coupling_pole=1.0)
    with pytest.raises(ValidationError):
        WatermarkParams(kappa_64=(1.0,))
    with pytest.raises(ValidationError):
        WatermarkParams(kappa_64=(1.0,) * 65)
    with pytest.raises(SampleRateError):
        hdr.replace(sample_rate_train_hz=100)

    # JsonValue validation in user_metadata
    p_ext = PresetDef(
        id="p",
        name="P",
        controls={},
        user_metadata={"nested": [1, True, None, 2.5, {"k": "v"}]},
    )
    assert p_ext.user_metadata
    with pytest.raises(ValidationError):
        PresetDef(id="p", name="P", controls={}, user_metadata={"bad_float": float("inf")})
    with pytest.raises(ValidationError):
        PresetDef(id="p", name="P", controls={}, user_metadata=cast(Any, {"bad_obj": object()}))

    # VerificationStatus & low-level signing helpers
    assert (
        VerificationStatus.from_rust_debug("CreatorVerified") == VerificationStatus.CREATOR_VERIFIED
    )
    assert (
        VerificationStatus.from_rust_debug("MarketplaceVerified")
        == VerificationStatus.MARKETPLACE_VERIFIED
    )
    seed = bytes(range(32))
    sig_hex = ed25519_sign(seed, b"hello")
    assert len(sig_hex) == 128
    h_sha = "11" * 32
    a_sha = "22" * 32
    assert len(build_creator_signing_message(h_sha, a_sha)) == 64
    from dnam.schema import MarketplaceReceipt

    rcpt = MarketplaceReceipt(
        marketplace_id="m",
        license_id="l",
        buyer_hash="33" * 32,
        license_tier="pro",
        issued_at="2026-01-01T00:00:00Z",
        buyer_arena_sha256="44" * 32,
        receipt_sig_ed25519="66" * 64,
    )
    assert len(build_marketplace_signing_message(h_sha, a_sha, rcpt)) > 64


def test_zero_pydantic_runtime_dependency_and_review_regressions() -> None:
    """verify[sdk.python.schema.rust-parity]
    Verifies zero pydantic dependency in dnam package metadata, conditional tempo-sync octave
    folding parity, alias collision rejection, union int preservation, and range normalization.
    """
    import importlib.metadata
    from typing import Any, cast

    from dnam.schema import (
        ControlKind,
        ControlSchemaDef,
        PresetDef,
        Stage3CabinetKautzConfig,
        StemRegistryEntry,
        TensorSliceEntry,
        TierProfile,
    )

    # 1. Zero pydantic dependency in dnam metadata
    reqs = importlib.metadata.requires("dnam") or []
    assert not any("pydantic" in r.lower() for r in reqs)

    # 2. Conditional tempo-sync octave folding (fold_octave=True vs False on conditional anchors)
    cond_cal = ControlCalibration(
        role=ControlCalibrationRole.DELAY_TIME_MS,
        conditional_physical_anchors={
            "range=short": (20.0, 60.0, 100.0),
            "range=long": (100.0, 200.0, 300.0),
        },
    )
    # 120 BPM quarter note = 500 ms (> 300 ms max of "range=long")
    unfolded = cond_cal.resolve_tempo_sync(120.0, "1/4", fold_octave=False)
    folded = cond_cal.resolve_tempo_sync(120.0, "1/4", fold_octave=True)
    assert unfolded is not None
    assert folded is not None
    assert unfolded.was_clamped
    assert unfolded.normalized_u == 1.0
    assert not folded.was_clamped
    assert folded.switch_selection == ("range", "long")
    assert abs(folded.normalized_u - 0.5) < 1e-4

    # 3. Alias / field-name collision rejection across __init__, from_dict, and replace
    ctrl = ControlSchemaDef(
        id="Gain",
        name="Gain",
        type=ControlKind.RANGE,
        declared_range=(0.0, 10.0),
        default=5,
    )
    assert type(ctrl.default) is float
    assert ctrl.default == 5.0
    with pytest.raises(ValidationError, match="Duplicate field 'kind'"):
        ControlSchemaDef(
            id="Gain",
            name="Gain",
            type=ControlKind.SWITCH,
            default=5.0,
            **cast(Any, {"kind": ControlKind.RANGE}),
        )
    with pytest.raises(ValidationError, match="Duplicate field 'kind'"):
        ControlSchemaDef.from_dict(
            {"id": "Gain", "name": "Gain", "type": "range", "kind": "switch", "default": 5.0}
        )
    with pytest.raises(ValidationError, match="Duplicate field 'kind'"):
        ctrl.replace(type=ControlKind.RANGE, kind=ControlKind.SWITCH)

    # 4. Union exact-type preservation (int stays int unless normalized as range control in header)
    preset = PresetDef(id="p1", name="P1", controls={"Step": 2, "Gain": 5, "Mode": "clean"})
    assert type(preset.controls["Step"]) is int
    with pytest.raises(ValidationError, match="JSON object keys must be str"):
        PresetDef(
            id="p1", name="P1", controls={}, user_metadata=cast(Any, {"nested": {123: "bad"}})
        )

    # 5. TierProfile default kautz_modes=0 and Stage3 fir_taps <= 255
    tp = TierProfile.from_dict(
        {"channels": 4, "tank_lines": 4, "arena_end_byte_offset": 10752, "prefix_crc32c": 0}
    )
    assert tp.kautz_modes == 0
    with pytest.raises(ValidationError):
        Stage3CabinetKautzConfig(fir_taps=256)

    # 6. DnamJsonHeader.__post_init__ normalizes int range control values in presets/stems
    base_hdr = DnamJsonHeader.from_json(_testing.reference_header_json("default"))
    dummy_slice = TensorSliceEntry(arena=4, dtype="f64", byte_offset=0, byte_length=64)
    norm_hdr = base_hdr.replace(
        controls_schema=[ctrl],
        presets=[PresetDef(id="p1", name="P1", controls={"Gain": 7})],
        stems_registry=[
            StemRegistryEntry(
                stem_id="s1",
                controls_meta={"Gain": 3},
                g_audio_uplo_u_f64=dummy_slice,
                c_audio_tier_f64=dummy_slice,
            )
        ],
    )
    assert type(norm_hdr.presets[0].controls["Gain"]) is float
    assert norm_hdr.stems_registry is not None
    assert type(norm_hdr.stems_registry[0].controls_meta["Gain"]) is float

    # 7. Misplaced skip_serializing_if field names rejected by Rust header validator
    raw_default = _testing.reference_header_json("default")
    bad_misplaced = raw_default.replace('"stage1_pre":{', '"stage1_pre":{"shape":"misplaced",')
    with pytest.raises(ValidationError, match="Unknown schema field 'shape'"):
        DnamJsonHeader.from_json(bad_misplaced)
