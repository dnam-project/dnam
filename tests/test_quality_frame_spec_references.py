"""Deterministic frame-grid examples for the segmental quality contract.

verify[qual.eval.metric-scope]
verify[qual.besr.per-output]
verify[qual.besr.finite-timeline]
verify[qual.besr.joint-intervals]
verify[qual.companion.suite]
verify[qual.companion.mr-stft]
verify[qual.companion.seg-esr]
verify[qual.tiers.benchmark-matrix]
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest


def _frame_length(sample_rate: float, *, max_length: int = 2**64 - 1) -> int:
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample rate must be finite and positive")
    length = round(Fraction.from_float(sample_rate) / 50)
    if length <= 0 or length > max_length:
        raise ValueError("frame length is not representable")
    return length


def _selected_frames(
    length: int, selected: tuple[int, ...], frame_length: int
) -> tuple[tuple[int, ...], ...]:
    if (
        length <= 0
        or frame_length <= 0
        or tuple(sorted(set(selected))) != selected
        or any(index < 0 or index >= length for index in selected)
    ):
        raise ValueError("invalid timeline selection")
    return tuple(
        frame
        for start in range(0, length, frame_length)
        if (frame := tuple(index for index in selected if start <= index < start + frame_length))
    )


def _duration_weighted_mean(
    frames: tuple[tuple[int, float, Fraction | float], ...],
) -> Fraction | float:
    if not frames:
        raise ValueError("no eligible frames")
    total_weight = Fraction()
    weighted_sum = Fraction()
    infinite = False
    for samples, rate, ratio in frames:
        if samples <= 0 or samples > _frame_length(rate):
            raise ValueError("invalid observed sample count")
        if isinstance(ratio, float):
            if math.isnan(ratio) or ratio < 0:
                raise ValueError("invalid frame ratio")
            if math.isinf(ratio):
                infinite = True
            else:
                ratio = Fraction.from_float(ratio)
        if ratio < 0:
            raise ValueError("invalid frame ratio")
        weight = Fraction(samples) / Fraction.from_float(rate)
        total_weight += weight
        if not isinstance(ratio, float):
            weighted_sum += weight * ratio
    return math.inf if infinite else weighted_sum / total_weight


def _rate_frames(
    chunks: tuple[tuple[int, float], ...], selected: tuple[int, ...]
) -> tuple[tuple[float, tuple[tuple[int, ...], ...]], ...]:
    if not chunks or any(length <= 0 for length, _ in chunks):
        raise ValueError("invalid timeline chunks")
    total = sum(length for length, _ in chunks)
    if not selected or tuple(sorted(set(selected))) != selected or selected[0] < 0:
        raise ValueError("invalid timeline selection")
    if selected[-1] >= total:
        raise ValueError("invalid timeline selection")
    runs: list[tuple[int, int, float]] = []
    start = 0
    for length, rate in chunks:
        _frame_length(rate)
        if runs and runs[-1][2] == rate:
            previous_start, _, _ = runs[-1]
            runs[-1] = (previous_start, start + length, rate)
        else:
            runs.append((start, start + length, rate))
        start += length
    output = []
    for start, end, rate in runs:
        local = tuple(index - start for index in selected if start <= index < end)
        frames = _selected_frames(end - start, local, _frame_length(rate))
        output.append((rate, tuple(tuple(index + start for index in frame) for frame in frames)))
    return tuple(output)


@pytest.mark.parametrize(
    ("sample_rate", "length"),
    [
        (44_100.0, 882),
        (48_000.0, 960),
        (96_000.0, 1920),
        (384_000.0, 7680),
        (44_100.25, 882),
        (44_124.9, 882),
        (44_125.1, 883),
    ],
)
def test_frame_length_uses_the_encoded_rate(sample_rate: float, length: int) -> None:
    assert _frame_length(sample_rate) == length


@pytest.mark.parametrize(
    ("tie_rate", "below", "at", "above"),
    [(48_025.0, 960, 960, 961), (48_075.0, 961, 962, 962)],
)
def test_frame_length_uses_even_ties_and_exact_neighbor_rates(
    tie_rate: float, below: int, at: int, above: int
) -> None:
    assert _frame_length(math.nextafter(tie_rate, -math.inf)) == below
    assert _frame_length(tie_rate) == at
    assert _frame_length(math.nextafter(tie_rate, math.inf)) == above


@pytest.mark.parametrize("rate", [0.0, -1.0, 25.0, math.inf, -math.inf, math.nan])
def test_invalid_or_zero_frame_length_is_rejected(rate: float) -> None:
    with pytest.raises(ValueError, match=r"sample rate|frame length"):
        _frame_length(rate)


def test_frame_length_must_fit_the_selected_representation() -> None:
    assert _frame_length(48_025.0, max_length=960) == 960
    with pytest.raises(ValueError, match="not representable"):
        _frame_length(48_075.0, max_length=960)


def test_disjoint_selection_retains_the_original_grid_and_partial_tail() -> None:
    frames = _selected_frames(1_925, (0, 2, 959, 960, 1919, 1920, 1924), 960)
    assert frames == ((0, 2, 959), (960, 1919), (1920, 1924))
    weights = tuple(Fraction(len(frame), 960) for frame in frames)
    ratios = (Fraction(0), Fraction(1), Fraction(2))
    mean = sum((weight * ratio for weight, ratio in zip(weights, ratios, strict=True)), Fraction())
    mean /= sum(weights, Fraction())
    assert mean == Fraction(6, 7)


def test_multirate_frames_use_elapsed_duration_not_raw_sample_count() -> None:
    frames = ((882, 44_100.0, Fraction(1)), (1_920, 96_000.0, Fraction(0)))
    assert _duration_weighted_mean(frames) == Fraction(1, 2)
    sample_count_mean = Fraction(882, 882 + 1_920)
    assert sample_count_mean < Fraction(1, 3)


@pytest.mark.parametrize("rate", [44_125.0, 44_124.9])
def test_rounded_full_frames_use_actual_encoded_duration(rate: float) -> None:
    frames = ((882, rate, Fraction(1)), (960, 48_000.0, Fraction(0)))
    duration_a = Fraction(882) / Fraction.from_float(rate)
    duration_b = Fraction(960, 48_000)
    assert _frame_length(rate) == 882
    assert _duration_weighted_mean(frames) == duration_a / (duration_a + duration_b)
    assert _duration_weighted_mean(frames) < Fraction(1, 2)


def test_rate_runs_preserve_grid_across_host_blocks_and_split_at_rate_change() -> None:
    selected = (0, 399, 400, 959, 960, 999, 1000, 1001, 2099)
    blocked = ((400, 48_000.0), (600, 48_000.0), (100, 96_000.0), (1000, 96_000.0))
    combined = ((1000, 48_000.0), (1100, 96_000.0))
    expected = (
        (48_000.0, ((0, 399, 400, 959), (960, 999))),
        (96_000.0, ((1000, 1001, 2099),)),
    )
    assert _rate_frames(blocked, selected) == expected
    assert _rate_frames(combined, selected) == expected


def test_run_with_no_selected_samples_does_not_invent_a_frame() -> None:
    runs = _rate_frames(((960, 48_000.0), (960, 96_000.0)), (0, 959))
    assert runs == ((48_000.0, ((0, 959),)), (96_000.0, ()))


def test_duration_uses_exact_encoded_rate_not_a_rounded_sample_rate() -> None:
    rate = math.nextafter(48_000.0, math.inf)
    frames = ((960, rate, Fraction(1)), (960, 48_000.0, Fraction(0)))
    duration_a = Fraction(960) / Fraction.from_float(rate)
    duration_b = Fraction(960, 48_000)
    assert _duration_weighted_mean(frames) == duration_a / (duration_a + duration_b)
    assert _duration_weighted_mean(frames) < Fraction(1, 2)


def test_multirate_partial_frames_contribute_selected_duration() -> None:
    frames = (
        (960, 48_000.0, Fraction(0)),
        (96, 96_000.0, Fraction(1)),
        (441, 44_100.0, Fraction(2)),
    )
    expected = (Fraction(1, 1_000) + Fraction(2, 100)) / (
        Fraction(1, 50) + Fraction(1, 1_000) + Fraction(1, 100)
    )
    assert _duration_weighted_mean(frames) == expected


def test_single_rate_duration_weights_preserve_existing_frame_mean() -> None:
    counts = (3, 2, 2)
    ratios = (Fraction(0), Fraction(1), Fraction(2))
    frames = tuple(zip(counts, (48_000.0,) * 3, ratios, strict=True))
    assert _duration_weighted_mean(frames) == Fraction(6, 7)


def test_infinite_frame_and_invalid_frame_remain_distinct() -> None:
    assert _duration_weighted_mean(((1, 48_000.0, math.inf),)) == math.inf
    with pytest.raises(ValueError, match="invalid frame ratio"):
        _duration_weighted_mean(((1, 48_000.0, math.inf), (1, 48_000.0, math.nan)))


@pytest.mark.parametrize(
    ("frames", "message"),
    [
        ((), "no eligible frames"),
        (((0, 48_000.0, Fraction(0)),), "invalid observed sample count"),
        (((961, 48_000.0, Fraction(0)),), "invalid observed sample count"),
        (((1, math.nan, Fraction(0)),), "sample rate"),
        (((1, 48_000.0, Fraction(-1)),), "invalid frame ratio"),
        (((1, 48_000.0, -math.inf),), "invalid frame ratio"),
    ],
)
def test_invalid_duration_inputs_are_rejected(
    frames: tuple[tuple[int, float, Fraction | float], ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _duration_weighted_mean(frames)


@pytest.mark.parametrize(
    ("chunks", "selected"),
    [
        ((), (0,)),
        (((0, 48_000.0),), (0,)),
        (((10, 48_000.0),), ()),
        (((10, 48_000.0),), (2, 2)),
        (((10, 48_000.0),), (10,)),
    ],
)
def test_invalid_rate_runs_or_selections_are_rejected(
    chunks: tuple[tuple[int, float], ...], selected: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match="invalid timeline"):
        _rate_frames(chunks, selected)


@pytest.mark.parametrize("selected", [(-1,), (1, 1), (2, 1), (10,)])
def test_invalid_sample_selections_are_rejected(selected: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="invalid timeline selection"):
        _selected_frames(10, selected, 4)
