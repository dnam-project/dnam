"""Unit and property-based tests for `ControlSpace`, `ControlRack`, `LobattoBasis`, `Rank4Fm`,
`chebyshev_lfo`, and `trust_region_softclip`."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from dnam_engine import (
    ControlKind,
    ControlSpace,
    DnamContainer,
    DnamControlError,
    DnamEngine,
    DnamValidationError,
    control,
)
from dnam_engine.schema import ControlSchemaDef, LobattoParams


def test_control_space_clamping_and_gesture_mapping(
    ref_container_core1: DnamContainer,
) -> None:
    space = ref_container_core1.control_space
    assert "gain" in space.schema
    assert "warm_slapback" in space.presets
    assert space.index_of("gain") == 0
    assert space.index_of("boost") == 1

    # [CLAMP-RANGE]: gain declared [0, 10], safe [0, 9.5]
    res_range = space.clamp({"gain": 12.0, "boost": "off"})
    assert res_range.any_clamped
    assert res_range.controls["gain"].was_clamped
    assert abs(float(res_range.controls["gain"].clamped_physical) - 9.5) < 1e-5
    assert res_range.physical_values["boost"] == "off"

    # [CLAMP-STABILITY]: conditional_max boost=on -> gain <= 8.0
    res_cond = space.clamp({"gain": 9.0, "boost": "on"})
    assert res_cond.controls["gain"].was_clamped
    assert abs(float(res_cond.controls["gain"].clamped_physical) - 8.0) < 1e-5

    # [CLAMP-SWITCH]: unobserved option 'ultra' -> falls back to default 'off'
    res_sw = space.clamp({"gain": 5.0, "boost": "ultra"})
    assert res_sw.controls["boost"].was_clamped
    assert res_sw.controls["boost"].clamped_physical == "off"

    # Switch by integer option index
    res_idx = space.clamp({"boost": 1})
    assert res_idx.controls["boost"].clamped_physical == "on"

    # UI gesture power-skew mapping
    phys_mid = space.to_physical("gain", 0.5)
    assert abs(phys_mid - 4.0) < 1e-5
    norm_mid = space.to_normalized("gain", phys_mid)
    assert abs(norm_mid - 0.5) < 1e-5

    # Standalone ControlSpace from Sequence[ControlSchemaDef]
    custom_space = ControlSpace(list(space.schema.values()), list(space.presets.values()))
    assert custom_space.clamp({"gain": 5.0}).physical_values["gain"] == 5.0

    with pytest.raises(DnamControlError):
        space.index_of("nonexistent")
    with pytest.raises(DnamControlError):
        space.resolve_numeric_value("nonexistent", 1.0)
    with pytest.raises(DnamControlError):
        space.resolve_numeric_value("boost", "invalid_opt")
    with pytest.raises(DnamControlError):
        space.resolve_numeric_value("boost", 99)
    with pytest.raises(DnamControlError):
        space.resolve_numeric_value("gain", "not_a_float")
    with pytest.raises(DnamControlError):
        space.clamp({"unknown_ctrl": 1.0})
    with pytest.raises(DnamControlError):
        space.to_physical("unknown_ctrl", 0.5)
    with pytest.raises(DnamControlError):
        space.to_physical("boost", 0.5)
    with pytest.raises(DnamControlError):
        space.to_normalized("unknown_ctrl", 5.0)
    with pytest.raises(DnamControlError):
        space.to_normalized("boost", 1.0)


def test_control_rack_mutable_mapping(engine_core2: DnamEngine) -> None:
    rack = engine_core2.controls
    assert len(rack) == 2
    assert list(rack) == ["gain", "boost"]
    assert "gain" in rack
    assert rack.schema["gain"].kind == ControlKind.RANGE

    rack["gain"] = 7.5
    assert abs(float(rack["gain"]) - 7.5) < 1e-5
    del rack["gain"]
    assert abs(float(rack["gain"]) - 5.0) < 1e-5

    snapped = rack.snap({"gain": 6.0}, boost="on")
    assert snapped.physical_values["boost"] == "on"

    preset_res = rack.apply_preset("warm_slapback", immediate=True)
    assert abs(float(preset_res.physical_values["gain"]) - 6.5) < 1e-5
    preset_slew = rack.apply_preset("warm_slapback", immediate=False)
    assert preset_slew.physical_values["boost"] == "on"

    reset_res = rack.reset_to_defaults()
    assert reset_res.physical_values["boost"] == "off"

    with pytest.raises(DnamControlError):
        _ = rack["missing"]
    with pytest.raises(DnamControlError):
        del rack["missing"]
    with pytest.raises(DnamControlError):
        rack.apply_preset("missing_preset")


@given(u=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_lobatto_basis_and_rank4_fm_properties(
    ref_container_core1: DnamContainer,
    u: float,
) -> None:
    lobatto = control.LobattoBasis.from_schema(ref_container_core1.controls["gain"])
    assert lobatto.params.degree == 2
    # Anchor 0 is at u0 = 0.0 -> basis vanishes at default anchor!
    at_zero = lobatto(0.0)
    np.testing.assert_allclose(at_zero, np.zeros(3), atol=1e-12)

    val_1d = lobatto(u)
    assert val_1d.shape == (3,)
    val_vec = lobatto(np.array([0.0, u], dtype=np.float64))
    assert val_vec.shape == (2, 3)
    np.testing.assert_allclose(val_vec[1], val_1d, atol=1e-12)

    # Fallback anchor calculation when anchor_nodes_norm is None
    ctrl_no_anchors = ControlSchemaDef(
        id="tone",
        name="Tone",
        type=ControlKind.RANGE,
        declared_range=(0.0, 10.0),
        default=5.0,
    )
    lob2 = control.LobattoBasis.from_schema(ctrl_no_anchors)
    np.testing.assert_allclose(lob2(0.0), np.zeros(3), atol=1e-12)

    with pytest.raises(DnamValidationError):
        control.LobattoBasis(cast(Any, "not_params"))
    with pytest.raises(DnamValidationError):
        control.LobattoBasis.from_schema(ref_container_core1.controls["boost"])
    with pytest.raises(DnamValidationError):
        lobatto(np.zeros((2, 2)))

    # Rank-4 FM exact agreement with explicit pairwise sum
    v_mat = np.array(
        [
            [0.2, -0.1, 0.4, 0.3],
            [-0.3, 0.5, 0.1, -0.2],
            [0.4, 0.2, -0.5, 0.1],
            [0.1, -0.4, 0.3, 0.5],
        ],
        dtype=np.float64,
    )
    fm = control.Rank4Fm(v_mat)
    assert fm.num_features == 4
    assert fm.v_factors.shape == (4, 4)
    x = np.array([u, -0.5 * u, 0.25, -0.75], dtype=np.float64)
    fm_out = fm(x)
    assert fm_out.shape == (4,)
    # Coerce from list input
    np.testing.assert_allclose(fm([u, -0.5 * u, 0.25, -0.75]), fm_out, atol=1e-12)

    with pytest.raises(DnamValidationError):
        control.Rank4Fm(np.zeros((4, 3)))
    with pytest.raises(DnamValidationError):
        fm(np.zeros(3))


@given(cos_phi=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_chebyshev_lfo_and_trust_region_softclip(cos_phi: float) -> None:
    cheb_scalar = control.chebyshev_lfo(cos_phi)
    assert cheb_scalar.shape == (4,)
    assert abs(cheb_scalar[0] - cos_phi) < 1e-12
    assert abs(cheb_scalar[1] - (2.0 * cos_phi * cos_phi - 1.0)) < 1e-12

    cheb_arr = control.chebyshev_lfo(np.array([cos_phi, 0.0]))
    assert cheb_arr.shape == (2, 4)
    with pytest.raises(DnamValidationError):
        control.chebyshev_lfo(np.zeros((2, 2)))

    clipped_s = control.trust_region_softclip(10.0, mh=2.5)
    assert isinstance(clipped_s, float)
    assert abs(clipped_s - 2.5) < 1e-12
    clipped_0d = control.trust_region_softclip(np.array(10.0), mh=2.5)
    assert isinstance(clipped_0d, float)
    assert abs(clipped_0d - 2.5) < 1e-12
    clipped_arr = control.trust_region_softclip(np.array([-10.0, 0.0, 10.0]), mh=2.5)
    assert isinstance(clipped_arr, np.ndarray)
    np.testing.assert_allclose(clipped_arr, np.array([-2.5, 0.0, 2.5]), atol=1e-12)

    with pytest.raises(DnamValidationError):
        control.trust_region_softclip(1.0, mh=0.0)
    assert LobattoParams().degree == 2
