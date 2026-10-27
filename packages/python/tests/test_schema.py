"""Unit, `hypothesis-jsonschema` fuzzing, and `hypothesis[crosshair]` symbolic Z3 tests
for `dnam_engine.schema` and `dnam_engine.exceptions`."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis_jsonschema import from_schema

from dnam_engine import (
    BackendKind,
    ControlKind,
    DnamContainerError,
    DnamControlError,
    DnamEngineSettings,
    DnamError,
    DnamSampleRateError,
    DnamValidationError,
    Tier,
    VerificationStatus,
)
from dnam_engine.exceptions import translate_native_error
from dnam_engine.schema import (
    TEMPO_SUBDIVISION_MULTIPLIERS,
    ControlCalibration,
    ControlSchemaDef,
    ControlUiHints,
    DnamJsonHeader,
    LevelingConfig,
    LobattoParams,
    ModelCapabilities,
    MufdnParams,
    OpenMetadataModel,
    PolyAdaaParams,
    PolyAdaaState,
    PresetDef,
    SelectiveRgLruParams,
    TensorSliceEntry,
    TierProfile,
    WatermarkParams,
    tempo_subdivision_delay_ms,
    tempo_subdivision_freq_hz,
)


def test_exception_hierarchy_and_translation() -> None:
    assert issubclass(DnamValidationError, DnamError)
    assert issubclass(DnamContainerError, DnamError)
    assert issubclass(DnamControlError, DnamError)
    assert issubclass(DnamControlError, KeyError)
    assert issubclass(DnamSampleRateError, DnamError)

    assert isinstance(
        translate_native_error(ValueError("[DNAM-CONTAINER] bad magic")), DnamContainerError
    )
    assert isinstance(
        translate_native_error(ValueError("[DNAM-SAMPLERATE] bad sr")), DnamSampleRateError
    )
    assert isinstance(
        translate_native_error(ValueError("[DNAM-VALIDATION] bad field")), DnamValidationError
    )
    assert isinstance(translate_native_error(ValueError("generic error")), DnamError)


def test_enums_and_verification_status() -> None:
    assert Tier.LITE.channels == 4
    assert Tier.FULL.channels == 8
    assert Tier.ULTRA.channels == 12
    assert Tier.LITE.fdn_lines == 4
    assert Tier.FULL.fdn_lines == 8
    assert Tier.ULTRA.fdn_lines == 8
    assert BackendKind.AUTO == "auto"
    assert ControlKind.RANGE == "range"
    assert VerificationStatus.from_rust_debug("Unsigned") == VerificationStatus.UNSIGNED
    assert (
        VerificationStatus.from_rust_debug("CreatorVerified { signer_id: ... }")
        == VerificationStatus.CREATOR_VERIFIED
    )
    assert (
        VerificationStatus.from_rust_debug("MarketplaceVerified { ... }")
        == VerificationStatus.MARKETPLACE_VERIFIED
    )

    open_m = OpenMetadataModel.model_validate({"custom_tag": 42})
    assert open_m.model_dump()["custom_tag"] == 42


def test_schema_validation_errors() -> None:
    with pytest.raises(DnamSampleRateError):
        DnamEngineSettings(sample_rate_hz=22_050.0)

    with pytest.raises(DnamValidationError):
        ControlSchemaDef(id="g", name="G", type=ControlKind.RANGE, default=5.0)

    with pytest.raises(DnamValidationError):
        ControlSchemaDef(
            id="g",
            name="G",
            type=ControlKind.RANGE,
            declared_range=(5.0, 5.0),
            default=5.0,
        )

    with pytest.raises(DnamValidationError):
        ControlSchemaDef(
            id="g",
            name="G",
            type=ControlKind.RANGE,
            declared_range=(0.0, 10.0),
            default="invalid",
        )

    with pytest.raises(DnamValidationError):
        ControlSchemaDef(id="s", name="S", type=ControlKind.SWITCH, default="off")

    with pytest.raises(DnamValidationError):
        ControlSchemaDef(
            id="s",
            name="S",
            type=ControlKind.SWITCH,
            declared_options=["off", "on"],
            default="invalid",
        )

    with pytest.raises(DnamValidationError):
        TierProfile(channels=6, fdn_lines=4, arena_end_byte_offset=10_752)

    with pytest.raises(DnamValidationError):
        TierProfile(channels=4, fdn_lines=4, arena_end_byte_offset=10_750)

    with pytest.raises(DnamValidationError):
        TensorSliceEntry(arena=0, byte_offset=10, byte_length=64)

    with pytest.raises(DnamValidationError):
        TensorSliceEntry(arena=0, byte_offset=64, byte_length=100)

    with pytest.raises(DnamValidationError):
        TensorSliceEntry(arena=3, byte_offset=64, byte_length=70)

    with pytest.raises(DnamValidationError):
        PresetDef.model_validate_json("{not valid json")

    hdr = DnamJsonHeader.default_reference()
    with pytest.raises(DnamSampleRateError):
        DnamJsonHeader.model_validate({**hdr.model_dump(), "sample_rate_train_hz": 16_000})

    dup_ctrl = ControlSchemaDef(
        id="gain", name="Gain", type=ControlKind.RANGE, declared_range=(0.0, 10.0), default=5.0
    )
    with pytest.raises(DnamValidationError):
        DnamJsonHeader.model_validate(
            {**hdr.model_dump(), "controls_schema": [dup_ctrl.model_dump(), dup_ctrl.model_dump()]}
        )

    too_many = [
        ControlSchemaDef(
            id=f"c_{i}",
            name=f"C{i}",
            type=ControlKind.RANGE,
            declared_range=(0.0, 10.0),
            default=5.0,
        ).model_dump()
        for i in range(17)
    ]
    with pytest.raises(DnamValidationError):
        DnamJsonHeader.model_validate({**hdr.model_dump(), "controls_schema": too_many})

    with pytest.raises(DnamValidationError):
        LobattoParams(anchors=(0.0, -1.5, 1.0))

    with pytest.raises(DnamValidationError):
        SelectiveRgLruParams(r_low=0.95, dr_span=0.10)

    with pytest.raises(DnamValidationError):
        WatermarkParams(kappa_48=(1.0,) * 12)

    with pytest.raises(DnamValidationError):
        WatermarkParams(kappa_48=(1.5,) * 48)

    with pytest.raises(DnamValidationError):
        WatermarkParams(tardos_code_4608=(1.0,) * 10)


@given(
    t=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    midpoint=st.one_of(
        st.none(),
        st.floats(min_value=1.0, max_value=9.0, allow_nan=False, allow_infinity=False),
    ),
)
def test_ui_hints_power_skew_bijection(t: float, midpoint: float | None) -> None:
    hints = ControlUiHints(midpoint=midpoint)
    phys = hints.normalized_to_physical(t, 0.0, 10.0)
    assert 0.0 <= phys <= 10.0
    t_back = hints.physical_to_normalized(phys, 0.0, 10.0)
    assert abs(t_back - t) < 1e-5
    if midpoint is not None:
        assert abs(hints.normalized_to_physical(0.5, 0.0, 10.0) - midpoint) < 1e-5
    assert hints.skew_exponent(5.0, 5.0) == 1.0
    assert hints.physical_to_normalized(5.0, 5.0, 5.0) == 0.0
    assert hints.physical_to_normalized(0.0, 0.0, 10.0) == 0.0
    assert hints.physical_to_normalized(10.0, 0.0, 10.0) == 1.0


@settings(max_examples=15)
@given(payload=from_schema(PresetDef.model_json_schema()))
def test_preset_def_jsonschema_fuzzing(payload: dict[str, object]) -> None:
    preset = PresetDef.model_validate(payload)
    roundtrip = PresetDef.model_validate_json(preset.model_dump_json())
    assert roundtrip == preset


@settings(settings.get_profile("smt"))
@given(
    b_conv=st.floats(min_value=-1.5, max_value=1.5, allow_nan=False, allow_infinity=False),
    beta=st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
    delta=st.floats(min_value=-0.4, max_value=0.4, allow_nan=False, allow_infinity=False),
)
def test_symbolic_crosshair_poly_adaa_resting_equilibrium(
    b_conv: float, beta: float, delta: float
) -> None:
    params = PolyAdaaParams(b_conv=b_conv, beta=beta, delta=delta)
    eq = PolyAdaaState.resting_equilibrium(params)
    assert -1.0 <= eq.u_prev <= 1.0
    assert eq.a_prev == eq.m_dc

    mufdn = MufdnParams(delay_samples=64, g_lin=0.85, sigma_knee=0.35, v_ceil=0.985)
    assert 0.0 <= mufdn.g_lin <= 0.9990


def test_control_calibration_tempo_subdivisions_capabilities_and_leveling() -> None:
    assert len(TEMPO_SUBDIVISION_MULTIPLIERS) == 12
    assert abs(tempo_subdivision_delay_ms(120.0, "1/2.") - 1500.0) < 1e-4
    assert abs(tempo_subdivision_delay_ms(120.0, "1/4") - 500.0) < 1e-4
    assert abs(tempo_subdivision_delay_ms(120.0, "1/8.") - 375.0) < 1e-4
    assert abs(tempo_subdivision_delay_ms(120.0, "1/16T") - 83.3333) < 1e-2
    assert abs(tempo_subdivision_freq_hz(120.0, "1/4") - 2.0) < 1e-4

    cal = ControlCalibration(
        role="delay_time_ms",
        conditional_physical_anchors={
            "range=0.01-0.1s": (10.0, 55.0, 100.0),
            "range=0.1-1s": (100.0, 550.0, 1000.0),
            "range=1-2s": (1000.0, 1500.0, 2000.0),
        },
        tempo_sync_default_subdivision="1/8.",
    )
    res = cal.resolve_tempo_sync(120.0, "1/8.", fold_octave=False)
    assert res is not None
    assert res.switch_selection == ("range", "0.1-1s")
    assert abs(res.target_physical - 375.0) < 1e-3
    assert not res.was_clamped

    stereo_cal = ControlCalibration(role="stereo_mode")
    assert stereo_cal.stereo_width_for_option("Mono") == 0.0
    assert stereo_cal.stereo_width_for_option("Stereo") == 1.0
    assert stereo_cal.stereo_width_for_option("Unknown") is None

    caps = ModelCapabilities(
        blend_mode="host_parallel",
        default_blend=0.5,
        wet_polarity_inverted=True,
        supports_trails=True,
        tempo_sync_capable=True,
    )
    lvl = LevelingConfig(
        auto_unity_enabled=True,
        leveling_metric="rms_dbfs",
        reference_input_level_db=-20.5,
        measured_default_output_level_db=-14.5,
        unity_gain_offset_db=-6.0,
        target_gain_db=1.5,
        calibrated_trim_db=-4.5,
    )
    hdr = DnamJsonHeader.default_reference(core2_active=True)
    assert hdr.io_config.leveling.leveling_metric == "rms_dbfs"
    assert caps.wet_polarity_inverted is True
    assert abs(lvl.calibrated_trim_db - (-4.5)) < 1e-6

    with pytest.raises(DnamValidationError):
        ControlCalibration(role="delay_time_ms", lobatto_physical_anchors=(10.0, 10.0, 10.0))
    with pytest.raises(DnamValidationError):
        ControlCalibration(
            role="delay_time_ms",
            conditional_physical_anchors={"bad_key": (10.0, 50.0, 100.0)},
        )
