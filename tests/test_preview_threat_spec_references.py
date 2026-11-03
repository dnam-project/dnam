"""Observability limits for audio-only preview guards, not an engine test."""

from __future__ import annotations

from fractions import Fraction
from itertools import product

import pytest


def _single_control_window(
    anchor: Fraction, delta: Fraction, safe: tuple[Fraction, Fraction]
) -> tuple[Fraction, Fraction]:
    lower, upper = safe
    if not 0 <= lower <= anchor <= upper <= 1 or not 0 < delta <= Fraction(3, 20):
        raise ValueError("invalid preview bounds")
    start = max(lower, anchor - delta)
    end = min(upper, anchor + delta)
    if start == end:
        raise ValueError("no responsive preview interval")
    return start, end


def _single_switch_options(
    preset: str, alternative: str, observed: set[str], active: set[str], safe: set[str]
) -> tuple[str, str]:
    selected = {preset, alternative}
    if len(selected) != 2 or not selected <= observed & active & safe:
        raise ValueError("preview options are not distinct, observed, active and safe")
    return preset, alternative


def test_audio_only_origin_guard_cannot_distinguish_identical_replays() -> None:
    natural_audio = (
        (0.0, 0.25, 0.0, -0.125),
        (0.5, 0.1, 0.0, 0.0),
        (0.0, -0.2, 0.2, 0.0),
        (0.1, 0.1, 0.1, 0.1),
    )
    replayed_audio = tuple(tuple(frame) for frame in natural_audio)
    assert replayed_audio == natural_audio

    for classifications in product((False, True), repeat=len(natural_audio)):
        guard = dict(zip(natural_audio, classifications, strict=True))
        poisoned_natural = sum(guard[frame] for frame in natural_audio)
        poisoned_replay = sum(guard[frame] for frame in replayed_audio)
        assert poisoned_natural == poisoned_replay
        assert not (poisoned_natural == 0 and poisoned_replay == len(replayed_audio))


def test_strict_rank_deficiency_cannot_cover_zero_control_dimensions() -> None:
    full_rank = 0
    available_preview_ranks = (0,)
    assert not any(preview_rank < full_rank for preview_rank in available_preview_ranks)


def test_one_dimensional_linear_preview_cannot_be_both_responsive_and_rank_deficient() -> None:
    full_outputs = (0, 2)
    responsive_outputs = (1, 2)
    locked_outputs = (1, 1)
    full_rank = int(full_outputs[1] != full_outputs[0])
    responsive_rank = int(responsive_outputs[1] != responsive_outputs[0])
    locked_rank = int(locked_outputs[1] != locked_outputs[0])
    assert full_rank == responsive_rank == 1
    assert locked_rank == 0 < full_rank


def test_individually_rank_deficient_preview_directions_can_span_full_space() -> None:
    first = (1, 0)
    second = (0, 1)
    assert first != (0, 0)
    assert second != (0, 0)
    determinant = first[0] * second[1] - first[1] * second[0]
    assert determinant == 1


def test_common_tangent_does_not_constrain_different_preset_anchors() -> None:
    tangent = (1, 0)
    anchor_difference = (0, 1)
    determinant = tangent[0] * anchor_difference[1] - tangent[1] * anchor_difference[0]
    assert determinant == 1


def test_common_affine_subspace_contains_all_preview_points() -> None:
    anchor = (Fraction(1, 4), Fraction(1, 2))
    direction = (Fraction(1), Fraction(2))
    parameters = (Fraction(-1, 8), Fraction(0), Fraction(1, 8))
    points = tuple((anchor[0] + direction[0] * t, anchor[1] + direction[1] * t) for t in parameters)
    first = (points[1][0] - points[0][0], points[1][1] - points[0][1])
    second = (points[2][0] - points[0][0], points[2][1] - points[0][1])
    assert first[0] * second[1] - first[1] * second[0] == 0


def test_independent_safety_clamping_can_leave_common_affine_subspace() -> None:
    requested = (Fraction(1), Fraction(1))
    effective = (min(requested[0], Fraction(1, 2)), requested[1])
    assert requested[0] == requested[1]
    assert effective[0] != effective[1]


def test_single_control_window_respects_preset_and_safe_bounds() -> None:
    delta = Fraction(3, 20)
    assert _single_control_window(Fraction(0), delta, (Fraction(0), Fraction(1))) == (0, delta)
    assert _single_control_window(Fraction(1), delta, (Fraction(0), Fraction(1))) == (1 - delta, 1)
    assert _single_control_window(Fraction(1, 2), delta, (Fraction(2, 5), Fraction(3, 5))) == (
        Fraction(2, 5),
        Fraction(3, 5),
    )


@pytest.mark.parametrize(
    ("anchor", "delta", "safe"),
    [
        (Fraction(1, 2), Fraction(0), (Fraction(0), Fraction(1))),
        (Fraction(1, 2), Fraction(4, 20), (Fraction(0), Fraction(1))),
        (Fraction(1, 2), Fraction(1, 10), (Fraction(3, 5), Fraction(1))),
    ],
)
def test_single_control_window_rejects_invalid_domain(
    anchor: Fraction, delta: Fraction, safe: tuple[Fraction, Fraction]
) -> None:
    with pytest.raises(ValueError, match="invalid preview bounds"):
        _single_control_window(anchor, delta, safe)


def test_single_control_window_requires_a_nonempty_interval() -> None:
    with pytest.raises(ValueError, match="no responsive preview interval"):
        _single_control_window(Fraction(1, 2), Fraction(1, 10), (Fraction(1, 2), Fraction(1, 2)))


def test_single_control_can_be_responsive_when_preset_tangent_is_constant() -> None:
    anchor = Fraction(0)
    another_setting = Fraction(1, 10)
    target_at_anchor = anchor**2
    target_at_other = another_setting**2
    linear_tangent_at_other = target_at_anchor + 2 * anchor * (another_setting - anchor)
    assert target_at_other != target_at_anchor
    assert linear_tangent_at_other == target_at_anchor


def test_single_switch_preview_uses_only_two_valid_choices() -> None:
    assert _single_switch_options("a", "b", {"a", "b", "c"}, {"a", "b"}, {"a", "b"}) == (
        "a",
        "b",
    )
    with pytest.raises(ValueError, match="not distinct, observed, active and safe"):
        _single_switch_options("a", "c", {"a", "b", "c"}, {"a", "c"}, {"a", "b"})
    with pytest.raises(ValueError, match="not distinct, observed, active and safe"):
        _single_switch_options("a", "a", {"a"}, {"a"}, {"a"})
