"""Tests for `dnam.controls` (`ControlSpace`, `compute_zoh`, `invert_tempo_calibration`,
`subdivision_phase_q32`).
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import dnam
from dnam.controls import (
    compute_zoh,
    control_stride_for_rate,
    control_tick_ratio,
    eval_3node_calibration,
    invert_tempo_calibration,
    subdivision_phase_q32,
)
from dnam.errors import ControlError, SampleRateError, ValidationError
from dnam.schema import TempoSubdivision
from strategies import bpm_strategy, sample_rate_strategy, subdivision_strategy


@given(
    gain=st.floats(min_value=-20.0, max_value=30.0, allow_nan=False, allow_infinity=False),
    boost=st.sampled_from(["off", "on", "ultra"]),
)
@settings(max_examples=30, deadline=None)
def test_control_space_3phase_clamping_properties(
    reference_model: dnam.Model,
    gain: float,
    boost: str,
) -> None:
    """verify[sdk.python.controls.space-and-tempo]
    Verifies 3-Phase control clamping idempotence, conditional range bounds (`boost=on` -> max 8.0),
    and unobserved switch clamping (`boost=ultra` -> clamped to active option).
    """
    space = reference_model.control_space
    assert len(space) == 3
    assert "gain" in space
    assert "ControlSpace" in repr(space)

    res = space.clamp({"gain": gain, "boost": boost})
    clamped_gain = float(res["gain"].value)
    clamped_boost = str(res["boost"].value)

    assert clamped_boost in ("off", "on")
    if boost == "ultra":
        assert res["boost"].was_clamped

    upper_bound = 8.0 if clamped_boost == "on" else 9.5
    assert 0.0 <= clamped_gain <= upper_bound + 1e-6

    # Clamping is idempotent: clamping already-clamped values changes nothing
    res2 = space.clamp({"gain": clamped_gain, "boost": clamped_boost})
    assert not res2.any_clamped
    assert abs(float(res2["gain"].value) - clamped_gain) < 1e-6
    assert res2["boost"].value == clamped_boost


def test_control_space_error_handling(reference_model: dnam.Model) -> None:
    """verify[sdk.python.controls.space-and-tempo] verify[sdk.python.errors.typed-hierarchy]
    Verifies ControlError and ValidationError on invalid control IDs, options, and non-finite
    numbers.
    """
    space = dnam.ControlSpace.from_header(reference_model.header)
    assert len(space.default_numeric_values()) == 3
    assert space.resolve_numeric_value("boost", 1) == 1.0
    assert space.resolve_numeric_value("gain", 5) == 5.0
    assert not space.clamp().any_clamped
    with pytest.raises(ControlError):
        space.index_of("nonexistent")
    with pytest.raises(ControlError):
        space.resolve_numeric_value("nonexistent", 1.0)
    with pytest.raises(ControlError):
        space.resolve_numeric_value("boost", "invalid_option")
    with pytest.raises(ControlError):
        space.resolve_numeric_value("boost", 99)
    with pytest.raises(ControlError):
        space.resolve_numeric_value("boost", True)
    with pytest.raises(ControlError):
        space.resolve_numeric_value("gain", "not_a_number")
    with pytest.raises(ValidationError):
        space.resolve_numeric_value("gain", float("nan"))
    with pytest.raises(ControlError):
        space.format_value("unknown", 1.0)
    with pytest.raises(ControlError):
        space.to_physical("unknown", 0.5)
    with pytest.raises(ControlError):
        space.to_normalized("unknown", 5.0)
    with pytest.raises(ControlError):
        space.clamp({"unknown": 1.0})
    with pytest.raises(ControlError):
        _ = space.clamp()["unknown"]


@given(sr=sample_rate_strategy)
@settings(max_examples=10, deadline=None)
def test_zoh_and_control_stride_invariants(sr: int) -> None:
    """verify[sdk.python.controls.space-and-tempo]
    Verifies ZOH parameter computation, control decimation stride, and tick ratio invariants.
    """
    zoh = compute_zoh(48_000, sr)
    assert zoh.sample_rate_host_hz == sr
    stride = control_stride_for_rate(sr)
    assert stride >= 8
    assert zoh.control_decimation_host == stride
    ratio = control_tick_ratio(sr)
    assert abs(zoh.rho_ctrl - ratio) < 1e-9

    with pytest.raises(SampleRateError):
        compute_zoh(48_000, 12_000)


@given(
    sub=subdivision_strategy,
    bpm=bpm_strategy,
    beat_mod12=st.integers(min_value=0, max_value=11),
    within_q32=st.integers(min_value=0, max_value=0xFFFF_FFFF),
)
@settings(max_examples=25, deadline=None)
def test_subdivision_phase_and_tempo_physical_inversion(
    sub: TempoSubdivision,
    bpm: float,
    beat_mod12: int,
    within_q32: int,
) -> None:
    """verify[sdk.python.controls.space-and-tempo]
    Verifies Q64/Q32 subdivision phase unit-phasor norm and 32-step binary-search physical
    calibration inversion.
    """
    assert sub.delay_ms_at_bpm(bpm) > 0.0
    assert sub.lfo_hz_at_bpm(bpm) > 0.0

    q32, cos_phi, sin_phi = subdivision_phase_q32(beat_mod12, within_q32, sub)
    assert 0 <= q32 <= 0xFFFF_FFFF
    assert abs(math.hypot(cos_phi, sin_phi) - 1.0) < 1e-5

    anchors = (20.0, 350.0, 1000.0)
    target = min(max(sub.delay_ms_at_bpm(bpm), 20.0), 1000.0)
    inv = invert_tempo_calibration((0.0, 10.0), (0.0, 10.0), anchors, target)
    assert 0.0 <= inv.physical_f32 <= 10.0
    assert -1.0 <= inv.normalized_f32 <= 1.0
    assert inv.search_steps[0] <= 32
    assert inv.search_steps[1] <= 32
    ev = eval_3node_calibration(anchors, inv.normalized_f32)
    assert abs(ev - inv.realized_value) < 1e-3
