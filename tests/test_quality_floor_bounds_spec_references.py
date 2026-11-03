"""Exact-arithmetic examples for bounded bESR noise-floor decisions."""

from __future__ import annotations

import math
from fractions import Fraction
from math import inf
from random import Random
from typing import Literal

import pytest

_HISS_LOWER = Fraction(1_258_925_411_794_167, 10**15)
_HISS_UPPER = Fraction(1_258_925_411_794_168, 10**15)
_Ratio = Fraction | float
_Decision = Literal["pass", "fail", "unavailable"]


def _combine_required_decisions(decisions: tuple[_Decision, ...]) -> _Decision:
    if "fail" in decisions:
        return "fail"
    if not decisions or "unavailable" in decisions:
        return "unavailable"
    return "pass"


def _band(
    target: Fraction,
    residual: Fraction,
    floor: Fraction,
    goal: Fraction,
    margin: Fraction | None = None,
) -> tuple[_Ratio, bool]:
    if min(target, residual, floor) < 0 or goal <= 0 or (margin is not None and margin <= 0):
        raise ValueError("invalid energy or threshold")
    if residual <= floor:
        ratio: _Ratio = Fraction(0)
    elif target + floor == 0:
        ratio = inf
    else:
        ratio = (residual - floor) / (target + floor)
    within_margin = residual**10 <= 10 * floor**10 if margin is None else residual <= margin * floor
    qualified = floor > 0 and target * goal < floor and within_margin
    return ratio, qualified


def _score(
    targets: tuple[Fraction, ...],
    residuals: tuple[Fraction, ...],
    floors: tuple[Fraction, ...],
    goals: tuple[Fraction, ...],
    composite_goal: Fraction,
) -> tuple[_Ratio, tuple[bool, bool, bool], bool]:
    if any(len(values) != 3 for values in (targets, residuals, floors, goals)):
        raise ValueError("three bands are required")
    if composite_goal <= 0:
        raise ValueError("invalid composite threshold")
    bands = tuple(
        _band(target, residual, floor, goal)
        for target, residual, floor, goal in zip(targets, residuals, floors, goals, strict=True)
    )
    qualified = (bands[0][1], bands[1][1], bands[2][1])
    contributions = tuple(Fraction(0) if match else ratio for ratio, match in bands)
    composite = sum(contributions) / 3
    band_pass = all(
        match or ratio <= goal for (ratio, match), goal in zip(bands, goals, strict=True)
    )
    return composite, qualified, band_pass and composite <= composite_goal


def _decision(
    targets: tuple[Fraction, Fraction, Fraction],
    residuals: tuple[Fraction, Fraction, Fraction],
    lower: tuple[Fraction, Fraction, Fraction],
    upper: tuple[Fraction, Fraction, Fraction],
    goals: tuple[Fraction, Fraction, Fraction],
    composite_goal: Fraction,
) -> str:
    if any(low < 0 or low > high for low, high in zip(lower, upper, strict=True)):
        raise ValueError("invalid floor bounds")
    worst = _score(targets, residuals, lower, goals, composite_goal)[2]
    best = _score(targets, residuals, upper, goals, composite_goal)[2]
    return "pass" if worst else "unavailable" if best else "fail"


def _rectangle_band_bounds(
    target: tuple[Fraction, Fraction],
    residual: tuple[Fraction, Fraction],
    floor: tuple[Fraction, Fraction],
    goal: Fraction,
) -> tuple[_Ratio, _Ratio, _Ratio, _Ratio, str]:
    if goal <= 0 or any(low < 0 or low > high for low, high in (target, residual, floor)):
        raise ValueError("invalid energy bounds or threshold")
    target_low, target_high = target
    residual_low, residual_high = residual
    floor_low, floor_high = floor

    def contribution(y: Fraction, r: Fraction, f: Fraction) -> _Ratio:
        ratio, qualified = _band(y, r, f, goal)
        return Fraction(0) if qualified else ratio

    raw_low = _band(target_high, residual_low, floor_high, goal)[0]
    raw_high = _band(target_low, residual_high, floor_low, goal)[0]
    if (
        floor_high > 0
        and target_low * goal < floor_high
        and residual_low**10 <= 10 * floor_high**10
    ):
        acceptance_low: _Ratio = Fraction(0)
    else:
        acceptance_low = contribution(target_high, residual_low, floor_high)

    if floor_low > 0 and residual_high**10 <= 10 * floor_low**10 and target_high * goal < floor_low:
        acceptance_high: _Ratio = Fraction(0)
    elif floor_low > 0 and residual_high**10 <= 10 * floor_low**10:
        target_at_worst = max(target_low, floor_low / goal)
        acceptance_high = contribution(target_at_worst, residual_high, floor_low)
    else:
        acceptance_high = contribution(target_low, residual_high, floor_low)

    if floor_low > 0 and target_high * goal < floor_low and residual_high**10 <= 10 * floor_low**10:
        qualification = "guaranteed"
    elif (
        floor_high == 0 or target_low * goal >= floor_high or residual_low**10 > 10 * floor_high**10
    ):
        qualification = "impossible"
    else:
        qualification = "uncertain"
    return raw_low, raw_high, acceptance_low, acceptance_high, qualification


def _conditional_gaussian_floor_interval(
    lower_energies: tuple[float, ...],
    alpha: float,
    *,
    upper_energies: tuple[float, ...] | None = None,
) -> tuple[float, float]:
    if not lower_energies or not 0 < alpha < 1:
        raise ValueError("invalid calibration")
    upper_energies = lower_energies if upper_energies is None else upper_energies
    if len(upper_energies) != len(lower_energies) or any(
        not math.isfinite(low) or not math.isfinite(high) or low < 0 or low > high
        for low, high in zip(lower_energies, upper_energies, strict=True)
    ):
        raise ValueError("invalid energy")
    lower_average = math.fsum(lower_energies) / len(lower_energies)
    upper_average = math.fsum(upper_energies) / len(upper_energies)
    if upper_average == 0:
        raise ValueError("zero observations do not establish an exact zero floor")
    t = math.log(2) - math.log(alpha)
    lower_deviation = 2 * math.sqrt(t / len(lower_energies))
    if lower_deviation >= 1:
        raise ValueError("finite upper bound unavailable")
    upper_deviation = lower_deviation + 2 * t / len(lower_energies)
    return lower_average / (1 + upper_deviation), upper_average / (1 - lower_deviation)


def _pcm_floor_interval(
    ideal_lower: float, ideal_upper: float, error_rms: float
) -> tuple[float, float]:
    if (
        not all(math.isfinite(value) for value in (ideal_lower, ideal_upper, error_rms))
        or ideal_lower < 0
        or ideal_lower > ideal_upper
        or error_rms < 0
    ):
        raise ValueError("invalid population bound")
    return (
        max(0.0, math.sqrt(ideal_lower) - error_rms) ** 2,
        (math.sqrt(ideal_upper) + error_rms) ** 2,
    )


def _unrailed_ideal_energy_bounds(
    samples: tuple[Fraction, ...],
    impulse: tuple[Fraction, ...],
    selected: tuple[int, ...],
    step: Fraction,
    rail: Fraction,
) -> tuple[Fraction, Fraction]:
    if (
        step <= 0
        or rail <= 0
        or (rail / step).denominator != 1
        or not selected
        or tuple(sorted(set(selected))) != selected
        or selected[0] < 0
        or selected[-1] >= len(samples)
        or not impulse
    ):
        raise ValueError("invalid quantized timeline or filter")
    context = samples[: selected[-1] + 1]
    if any((value / step).denominator != 1 or abs(value) >= rail for value in context):
        raise ValueError("rail or non-lattice sample has no unrailed bound")
    lower = upper = Fraction()
    for index in selected:
        taps = impulse[: index + 1]
        value = sum(
            coefficient * samples[index - offset] for offset, coefficient in enumerate(taps)
        )
        uncertainty = step / 2 * sum(map(abs, taps))
        lower += max(Fraction(), abs(value) - uncertainty) ** 2
        upper += (abs(value) + uncertainty) ** 2
    return lower, upper


def _gaussian_quantizer_error_rms_bound(sigma: float, rail: float, step: float) -> float:
    if (
        not all(math.isfinite(value) for value in (sigma, rail, step))
        or sigma < 0
        or rail <= 0
        or step <= 0
        or step / 2 == 0
    ):
        raise ValueError("invalid quantizer bound")
    if sigma == 0:
        return step / 2
    ratio = rail / sigma
    if not math.isfinite(ratio) or ratio == 0:
        raise ValueError("Gaussian tail needs a non-underflowing bound")
    density = math.exp(-ratio * ratio / 2) / math.sqrt(2 * math.pi)
    if density == 0 or 2 * density / ratio == 0:
        raise ValueError("Gaussian tail needs a non-underflowing bound")
    tail = sigma * min(1.0, math.sqrt(2 * density / ratio))
    if tail == 0:
        raise ValueError("Gaussian tail needs a non-underflowing bound")
    return step / 2 + tail


def _gaussian_unrailed_sigma_upper(
    timelines: tuple[tuple[Fraction, ...], ...],
    step: Fraction,
    rail: Fraction,
    failure_probability: float,
) -> float:
    if (
        not timelines
        or not timelines[0]
        or any(len(timeline) != len(timelines[0]) for timeline in timelines)
        or step <= 0
        or rail <= 0
        or (rail / step).denominator != 1
        or not 0 < failure_probability < 1
    ):
        raise ValueError("invalid variance calibration")
    if any(
        (sample / step).denominator != 1 or abs(sample) >= rail
        for timeline in timelines
        for sample in timeline
    ):
        raise ValueError("rail or non-lattice sample has no unrailed bound")
    count = len(timelines)
    sample_count = len(timelines[0])
    t = math.log(sample_count) - math.log(failure_probability)
    denominator = count - 2 * math.sqrt(count * t)
    if denominator <= 0:
        raise ValueError("insufficient independent timelines")
    square_sum_upper = max(
        sum((abs(timeline[index]) + step / 2) ** 2 for timeline in timelines)
        for index in range(sample_count)
    )
    return math.sqrt(float(square_sum_upper) / denominator)


def _bounded_pcm_floor_interval(
    lower_energies: tuple[float, ...],
    upper_energies: tuple[float, ...],
    maximum_energy: float,
    alpha: float,
) -> tuple[float, float]:
    if (
        not lower_energies
        or len(lower_energies) != len(upper_energies)
        or not math.isfinite(maximum_energy)
        or maximum_energy <= 0
        or not 0 < alpha < 1
        or any(
            not math.isfinite(low)
            or not math.isfinite(high)
            or low < 0
            or low > high
            or high > maximum_energy
            for low, high in zip(lower_energies, upper_energies, strict=True)
        )
    ):
        raise ValueError("invalid bounded PCM calibration")
    lower_average = math.fsum(lower_energies) / len(lower_energies)
    upper_average = math.fsum(upper_energies) / len(upper_energies)
    radius = maximum_energy * math.sqrt((math.log(2) - math.log(alpha)) / (2 * len(lower_energies)))
    return max(0.0, lower_average - radius), min(maximum_energy, upper_average + radius)


def _empirical_pcm_floor_interval(
    lower_energies: tuple[float, ...],
    upper_energies: tuple[float, ...],
    maximum_energy: float,
    alpha: float,
) -> tuple[float, float]:
    if len(lower_energies) < 2:
        raise ValueError("at least two energy observations are required")
    _bounded_pcm_floor_interval(lower_energies, upper_energies, maximum_energy, alpha)
    lows = tuple(map(Fraction.from_float, lower_energies))
    highs = tuple(map(Fraction.from_float, upper_energies))
    count = len(lows)
    sum_low = sum(lows)
    variance_upper = (sum(value**2 for value in highs) - sum_low**2 / count) / (count - 1)
    t = math.log(4) - math.log(alpha)
    radius = math.sqrt(2 * float(variance_upper) * t / count)
    radius += 7 * maximum_energy * t / (3 * (count - 1))
    return (
        max(0.0, float(sum_low / count) - radius),
        min(maximum_energy, float(sum(highs) / count) + radius),
    )


def test_hiss_margin_bracket_is_exact() -> None:
    assert _HISS_LOWER**10 < 10 < _HISS_UPPER**10
    assert _band(Fraction(0), _HISS_LOWER, Fraction(1), Fraction(1, 1000))[1]
    assert not _band(Fraction(0), _HISS_UPPER, Fraction(1), Fraction(1, 1000))[1]


@pytest.mark.parametrize(
    ("decisions", "expected"),
    [
        ((), "unavailable"),
        (("pass", "pass"), "pass"),
        (("pass", "unavailable"), "unavailable"),
        (("unavailable", "fail"), "fail"),
        (("fail", "unavailable"), "fail"),
    ],
)
def test_required_results_preserve_failure_and_unavailable(
    decisions: tuple[_Decision, ...], expected: _Decision
) -> None:
    assert _combine_required_decisions(decisions) == expected


def test_failed_channel_is_not_hidden_by_unavailable_other_channel() -> None:
    right_channel = _combine_required_decisions(("pass", "unavailable"))
    left_channel = _combine_required_decisions(("pass", "fail"))
    assert right_channel == "unavailable"
    assert left_channel == "fail"
    assert _combine_required_decisions((right_channel, left_channel)) == "fail"


def test_optimistic_floor_cannot_resolve_an_ambiguous_decision() -> None:
    targets = (Fraction(100),) * 3
    residuals = (Fraction(5, 4),) * 3
    goals = (Fraction(1, 1000),) * 3
    lower = (Fraction(4, 5),) * 3
    point = (Fraction(21, 20),) * 3
    upper = (Fraction(11, 10),) * 3
    assert _band(targets[0], residuals[0], lower[0], goals[0]) == (Fraction(1, 224), False)
    assert _score(targets, residuals, point, goals, goals[0])[2]
    assert _decision(targets, residuals, lower, upper, goals, goals[0]) == "unavailable"
    assert _decision(targets, residuals, lower, (Fraction(9, 10),) * 3, goals, goals[0]) == "fail"
    assert _decision(targets, residuals, point, upper, goals, goals[0]) == "pass"


def test_qualification_can_be_uncertain_even_when_requirements_pass() -> None:
    targets = (Fraction(100),) * 3
    residuals = (Fraction(5, 4),) * 3
    goals = (Fraction(1, 100),) * 3
    lower = (Fraction(4, 5),) * 3
    upper = (Fraction(11, 10),) * 3
    assert _score(targets, residuals, lower, goals, goals[0])[1] == (False,) * 3
    assert _score(targets, residuals, upper, goals, goals[0])[1] == (True,) * 3
    assert _decision(targets, residuals, lower, upper, goals, goals[0]) == "pass"


def test_floor_endpoints_bound_sampled_raw_and_qualified_scores() -> None:
    rng = Random(2026)
    goals = (Fraction(1, 100), Fraction(1, 200), Fraction(1, 1000))
    for _ in range(500):
        targets = tuple(Fraction(rng.randrange(201)) for _ in range(3))
        residuals = tuple(Fraction(rng.randrange(201), 10) for _ in range(3))
        lower = tuple(Fraction(rng.randrange(10), 10) for _ in range(3))
        upper = tuple(value + Fraction(rng.randrange(11), 10) for value in lower)
        middle = tuple((low + high) / 2 for low, high in zip(lower, upper, strict=True))
        low_score, low_qualified, low_pass = _score(targets, residuals, lower, goals, goals[0])
        mid_score, mid_qualified, mid_pass = _score(targets, residuals, middle, goals, goals[0])
        high_score, high_qualified, high_pass = _score(targets, residuals, upper, goals, goals[0])
        assert low_score >= mid_score >= high_score
        assert all(
            int(low) <= int(mid) <= int(high)
            for low, mid, high in zip(low_qualified, mid_qualified, high_qualified, strict=True)
        )
        assert int(low_pass) <= int(mid_pass) <= int(high_pass)
        for target, residual, low, mid, high, goal in zip(
            targets, residuals, lower, middle, upper, goals, strict=True
        ):
            assert _band(target, residual, low, goal)[0] >= _band(target, residual, mid, goal)[0]
            assert _band(target, residual, mid, goal)[0] >= _band(target, residual, high, goal)[0]


def test_zero_floor_and_composite_only_failure() -> None:
    zero = Fraction(0)
    goal = Fraction(1, 1000)
    assert _band(zero, zero, zero, goal) == (zero, False)
    assert _band(zero, Fraction(1), zero, goal) == (inf, False)
    targets = (Fraction(100),) * 3
    residuals = (Fraction(3, 20),) * 3
    floors = (zero,) * 3
    goals = (Fraction(1, 500),) * 3
    score, _, passed = _score(targets, residuals, floors, goals, goal)
    assert score == Fraction(3, 2000)
    assert not passed
    assert _decision(targets, residuals, floors, floors, goals, goal) == "fail"


def test_strict_snr_and_inclusive_residual_qualification() -> None:
    goal = Fraction(1, 10)
    assert not _band(Fraction(1), Fraction(0), Fraction(1, 10), goal)[1]
    assert _band(Fraction(1), Fraction(0), Fraction(101, 1000), goal)[1]
    margin = Fraction(5, 4)
    assert _band(Fraction(0), margin, Fraction(1), goal, margin)[1]
    assert not _band(Fraction(0), margin + Fraction(1, 10**9), Fraction(1), goal, margin)[1]


def test_qualified_score_is_not_monotone_in_target_energy() -> None:
    floor = Fraction(1)
    residual = Fraction(5, 4)
    goal = Fraction(1, 1000)
    below, below_qualified = _band(Fraction(0), residual, floor, goal)
    boundary, boundary_qualified = _band(Fraction(1000), residual, floor, goal)
    above, above_qualified = _band(Fraction(10_000), residual, floor, goal)
    below_contribution = Fraction(0) if below_qualified else below
    boundary_contribution = Fraction(0) if boundary_qualified else boundary
    above_contribution = Fraction(0) if above_qualified else above
    assert below_contribution == 0
    assert boundary_contribution == Fraction(1, 4004)
    assert above_contribution == Fraction(1, 40_004)
    assert below_contribution < boundary_contribution > above_contribution


def test_target_uncertainty_can_hide_an_interior_composite_failure() -> None:
    target = (Fraction(0), Fraction(2_000))
    residual = (Fraction(5, 4),) * 2
    floor = (Fraction(1),) * 2
    goal = Fraction(1, 1_000)
    composite_goal = Fraction(1, 20_000)
    _, _, best, worst, qualification = _rectangle_band_bounds(target, residual, floor, goal)
    assert best == 0
    assert worst == Fraction(1, 4_004)
    assert qualification == "uncertain"

    def passes(first_target: Fraction) -> bool:
        return _score(
            (first_target, Fraction(0), Fraction(0)),
            (residual[0], Fraction(0), Fraction(0)),
            (floor[0],) * 3,
            (goal,) * 3,
            composite_goal,
        )[2]

    assert passes(target[0])
    assert passes(target[1])
    assert not passes(Fraction(1_000))


def test_rectangle_bounds_cover_target_residual_and_floor_interiors() -> None:
    rng = Random(20260609)
    for _ in range(500):
        bounds = []
        for _ in range(3):
            low = Fraction(rng.randrange(21), 4)
            high = low + Fraction(rng.randrange(21), 4)
            bounds.append((low, high))
        target, residual, floor = bounds
        goal = Fraction(rng.randrange(1, 5), 10)
        raw_low, raw_high, accept_low, accept_high, qualification = _rectangle_band_bounds(
            target, residual, floor, goal
        )
        endpoint_targets = {target[0], target[1]}
        for endpoint_floor in floor:
            threshold = endpoint_floor / goal
            if target[0] <= threshold <= target[1]:
                endpoint_targets.add(threshold)
        endpoint_bands = [
            _band(y, r, f, goal) for y in endpoint_targets for r in residual for f in floor
        ]
        endpoint_raw = [ratio for ratio, _ in endpoint_bands]
        endpoint_acceptance = [
            Fraction(0) if qualified else ratio for ratio, qualified in endpoint_bands
        ]
        assert (min(endpoint_raw), max(endpoint_raw)) == (raw_low, raw_high)
        assert (min(endpoint_acceptance), max(endpoint_acceptance)) == (accept_low, accept_high)
        qualifiers = [qualified for _, qualified in endpoint_bands]
        assert qualification == (
            "guaranteed" if all(qualifiers) else "uncertain" if any(qualifiers) else "impossible"
        )
        for f in (floor[0], (floor[0] + floor[1]) / 2, floor[1]):
            targets = {target[0], (target[0] + target[1]) / 2, target[1]}
            if target[0] <= f / goal <= target[1]:
                targets.add(f / goal)
            residuals = {residual[0], (residual[0] + residual[1]) / 2, residual[1]}
            for margin in (_HISS_LOWER, _HISS_UPPER):
                if residual[0] <= margin * f <= residual[1]:
                    residuals.add(margin * f)
            for y in targets:
                for r in residuals:
                    ratio, qualified = _band(y, r, f, goal)
                    contribution = Fraction(0) if qualified else ratio
                    assert raw_low <= ratio <= raw_high
                    assert accept_low <= contribution <= accept_high
                    if qualification == "guaranteed":
                        assert qualified
                    elif qualification == "impossible":
                        assert not qualified


def test_rectangle_bounds_handle_zero_floor_and_invalid_inputs() -> None:
    zero = Fraction(0)
    one = Fraction(1)
    assert _rectangle_band_bounds((zero, one), (one, one), (zero, zero), one) == (
        one,
        inf,
        one,
        inf,
        "impossible",
    )
    assert _rectangle_band_bounds((zero, zero), (zero, zero), (zero, zero), one) == (
        zero,
        zero,
        zero,
        zero,
        "impossible",
    )
    for target, residual, floor, goal in (
        ((one, zero), (zero, one), (zero, one), one),
        ((zero, one), (-one, one), (zero, one), one),
        ((zero, one), (zero, one), (zero, one), zero),
    ):
        with pytest.raises(ValueError, match="invalid"):
            _rectangle_band_bounds(target, residual, floor, goal)


def test_joint_coverage_does_not_assume_independence() -> None:
    outcomes = set(range(10))
    missed = ({0}, {1}, {2})
    marginal = Fraction(len(outcomes - missed[0]), len(outcomes))
    simultaneous = Fraction(len(outcomes - set.union(*missed)), len(outcomes))
    union_bound = 1 - sum(1 - marginal for _ in missed)
    assert simultaneous == union_bound == Fraction(7, 10)
    assert simultaneous < marginal**3


def test_exact_ratio_detects_underflow_hidden_by_binary64() -> None:
    tiny = Fraction(1, 10**500)
    ratio, qualified = _band(Fraction(1), tiny, Fraction(0), Fraction(1, 1000))
    assert ratio == tiny
    assert not qualified
    assert float(tiny) == 0.0


def test_conditional_gaussian_interval_requires_sufficient_nonzero_evidence() -> None:
    with pytest.raises(ValueError, match="finite upper"):
        _conditional_gaussian_floor_interval((1.0,) * 14, 0.05)
    with pytest.raises(ValueError, match="exact zero"):
        _conditional_gaussian_floor_interval((0.0,) * 100, 0.05)
    for energies, alpha in [
        ((), 0.05),
        ((1.0,) * 100, 0.0),
        ((1.0,) * 100, 1.0),
        ((float("nan"),) * 100, 0.05),
        ((float("inf"),) * 100, 0.05),
        ((-1.0,) * 100, 0.05),
    ]:
        with pytest.raises(ValueError, match="invalid"):
            _conditional_gaussian_floor_interval(energies, alpha)
    lower, upper = _conditional_gaussian_floor_interval((1.0,) * 100, 0.05)
    assert 0.68 < lower < 0.69
    assert 1.62 < upper < 1.63
    widened_lower, widened_upper = _conditional_gaussian_floor_interval(
        (0.9,) * 100, 0.05, upper_energies=(1.1,) * 100
    )
    assert math.isclose(widened_lower, 0.9 * lower)
    assert math.isclose(widened_upper, 1.1 * upper)
    with pytest.raises(ValueError, match="invalid energy"):
        _conditional_gaussian_floor_interval((1.0,) * 100, 0.05, upper_energies=(0.9,) * 100)


def test_conditional_gaussian_interval_on_seeded_covariance_examples() -> None:
    rng = Random(87)
    misses = [0, 0]
    repetitions = 600
    for _ in range(repetitions):
        for case, weights in enumerate(((1.0,), (0.25, 0.25, 0.5))):
            energies = tuple(
                sum(weight * rng.gauss(0, 1) ** 2 for weight in weights) for _ in range(64)
            )
            lower, upper = _conditional_gaussian_floor_interval(energies, 0.05)
            misses[case] += not lower <= 1.0 <= upper
    assert all(miss < repetitions * 0.05 for miss in misses)


def test_population_rms_error_bounds_actual_pcm_energy() -> None:
    lower, upper = _pcm_floor_interval(1.0, 1.0, 0.1)
    assert math.isclose(lower, 0.9**2)
    assert math.isclose(upper, 1.1**2)
    assert _pcm_floor_interval(1.0, 1.0, 2.0) == (0.0, 9.0)
    assert _pcm_floor_interval(1.0, 4.0, 0.0) == (1.0, 4.0)
    for lower, upper, error in [
        (-1.0, 1.0, 0.1),
        (2.0, 1.0, 0.1),
        (0.0, 1.0, -0.1),
        (0.0, float("inf"), 0.1),
    ]:
        with pytest.raises(ValueError, match="invalid population"):
            _pcm_floor_interval(lower, upper, error)


def test_unrailed_pcm_windows_bracket_ideal_filtered_energy() -> None:
    step = Fraction(1, 4)
    rail = Fraction(2)
    raw = (Fraction(2, 5), Fraction(-2, 5), Fraction(3, 10), Fraction(-3, 10), Fraction(0))
    pcm = (Fraction(1, 2), Fraction(-1, 2), Fraction(1, 4), Fraction(-1, 4), Fraction(0))
    impulse = (Fraction(1), Fraction(-1, 2), Fraction(1, 4))
    selected = (0, 1, 2, 4)
    assert tuple(step * round(value / step) for value in raw) == pcm
    assert all(abs(q - x) <= step / 2 for x, q in zip(raw, pcm, strict=True))
    ideal_energy = sum(
        sum(
            coefficient * raw[index - offset]
            for offset, coefficient in enumerate(impulse[: index + 1])
        )
        ** 2
        for index in selected
    )
    lower, upper = _unrailed_ideal_energy_bounds(pcm, impulse, selected, step, rail)
    assert lower < ideal_energy < upper
    assert _unrailed_ideal_energy_bounds((Fraction(0), rail), impulse, (0,), step, rail) == (
        Fraction(0),
        (step / 2) ** 2,
    )
    with pytest.raises(ValueError, match="rail"):
        _unrailed_ideal_energy_bounds((rail,), impulse, (0,), step, rail)
    assert tuple(step * min(round(value / step), int(rail / step)) for value in (rail, 100)) == (
        rail,
        rail,
    )


def test_gaussian_quantizer_rms_bridge_covers_correlated_filter_errors() -> None:
    sigma = 1.0
    rail = 2.0
    step = 0.25
    ratio = rail / sigma
    density = math.exp(-ratio * ratio / 2) / math.sqrt(2 * math.pi)
    tail_probability = math.erfc(ratio / math.sqrt(2)) / 2
    exact_tail_energy = 2 * sigma**2 * ((1 + ratio**2) * tail_probability - ratio * density)
    assert 0 < exact_tail_energy < 2 * sigma**2 * density / ratio
    sample_bound = _gaussian_quantizer_error_rms_bound(sigma, rail, step)

    error_energy = 0.0
    tail_energy = 0.0
    pcm_energy = 0.0
    count = 32_768
    width = 16 / count
    for index in range(count):
        value = -8 + (index + 0.5) * width
        probability = math.exp(-(value**2) / 2) / math.sqrt(2 * math.pi) * width
        quantized = step * max(-8, min(8, round(value / step)))
        error_energy += (quantized - value) ** 2 * probability
        tail_energy += max(abs(value) - rail, 0.0) ** 2 * probability
        pcm_energy += quantized**2 * probability
    assert math.isclose(tail_energy, exact_tail_energy, rel_tol=1e-5)
    assert math.sqrt(error_energy) < sample_bound

    row_sums_squared = 1.0**2 + 1.5**2 + 1.75**2 + 1.75**2
    independent_squares = 1.0 + 1.25 + 1.3125 + 1.3125
    correlated_error = error_energy * row_sums_squared
    assert correlated_error > 1.9 * error_energy * independent_squares
    bound = sample_bound * math.sqrt(row_sums_squared)
    lower, upper = _pcm_floor_interval(row_sums_squared, row_sums_squared, bound)
    assert lower < pcm_energy * row_sums_squared < upper
    assert _gaussian_quantizer_error_rms_bound(0.0, rail, step) == step / 2
    with pytest.raises(ValueError, match="underflowing"):
        _gaussian_quantizer_error_rms_bound(1.0, 100.0, step)


def test_noise_only_quantizer_floor_does_not_bound_signal_conditioned_error() -> None:
    sigma = 0.125
    floor = sum(
        (2 * level - 1) * math.erfc((level - 0.5) / (sigma * math.sqrt(2))) for level in range(1, 5)
    )
    assert 6.3e-5 < floor < 6.4e-5
    assert round(0.5) == 0
    assert round(0.5 - 0.0625) == 0
    assert round(0.5 + 0.0625) == 1
    assert round(-0.0625) == round(0.0625) == 0
    inside = math.erf(1 / (sigma * math.sqrt(2)))
    assert inside > 0.999999999
    assert 0.25 * inside > 3_900 * floor


def test_unrailed_gaussian_variance_bound_covers_correlated_timelines() -> None:
    step = Fraction(1, 16)
    rail = Fraction(8)
    count = 128
    beta = 0.05
    rng = Random(839)
    misses = 0
    for _ in range(200):
        samples = []
        for _ in range(count):
            quantized = Fraction(round(16 * rng.gauss(0, 1)), 16)
            samples.append((quantized, -quantized, quantized, -quantized))
        upper = _gaussian_unrailed_sigma_upper(tuple(samples), step, rail, beta)
        misses += upper < 1.0
    assert misses < 200 * beta

    zero = ((Fraction(0),) * 4,) * count
    assert _gaussian_unrailed_sigma_upper(zero, step, rail, beta) > 0
    with pytest.raises(ValueError, match="insufficient"):
        _gaussian_unrailed_sigma_upper(zero[:8], step, rail, beta)
    with pytest.raises(ValueError, match="rail"):
        _gaussian_unrailed_sigma_upper(((rail,),) * count, step, rail, beta)


def test_bounded_pcm_interval_needs_a_prior_bound_and_keeps_zero_uncertain() -> None:
    low, high = _bounded_pcm_floor_interval((0.5,) * 100, (0.5,) * 100, 1.0, 0.05)
    assert 0.364 < low < 0.365
    assert 0.635 < high < 0.636
    widened_low, widened_high = _bounded_pcm_floor_interval((0.4,) * 100, (0.6,) * 100, 1.0, 0.05)
    assert math.isclose(widened_low, low - 0.1)
    assert math.isclose(widened_high, high + 0.1)
    zero_low, zero_high = _bounded_pcm_floor_interval((0.0,) * 100, (0.0,) * 100, 1.0, 0.05)
    assert zero_low == 0.0
    assert zero_high > 0.0
    for lower, upper, bound, alpha in [
        ((), (), 1.0, 0.05),
        ((0.0,), (), 1.0, 0.05),
        ((0.0,), (1.0,), 0.0, 0.05),
        ((0.0,), (2.0,), 1.0, 0.05),
        ((0.5,), (0.4,), 1.0, 0.05),
        ((0.0,), (1.0,), 1.0, 1.0),
    ]:
        with pytest.raises(ValueError, match="invalid bounded"):
            _bounded_pcm_floor_interval(lower, upper, bound, alpha)


def test_bounded_pcm_interval_on_seeded_non_gaussian_noise() -> None:
    rng = Random(299)
    failures = 0
    repetitions = 1_000
    for _ in range(repetitions):
        samples = tuple(float(rng.random() < 0.2) for _ in range(128))
        low, high = _bounded_pcm_floor_interval(samples, samples, 1.0, 0.05)
        failures += not low <= 0.2 <= high
    assert failures < repetitions * 0.05


def test_worst_case_pcm_bound_can_require_impractical_sample_counts() -> None:
    alpha = 0.05
    average_energy = 1e-8
    desired_radius = average_energy / 2
    sample_count = (math.log(2) - math.log(alpha)) / (2 * desired_radius**2)
    assert 7e16 < sample_count < 8e16


def test_empirical_pcm_variance_bound_covers_all_interval_corners() -> None:
    lows = (Fraction(1, 10), Fraction(2, 10), Fraction(3, 10))
    highs = (Fraction(3, 10), Fraction(6, 10), Fraction(9, 10))
    count = len(lows)
    upper = (sum(value**2 for value in highs) - sum(lows) ** 2 / count) / (count - 1)
    for first in (lows[0], highs[0]):
        for second in (lows[1], highs[1]):
            for third in (lows[2], highs[2]):
                values = (first, second, third)
                mean = sum(values) / count
                variance = sum((value - mean) ** 2 for value in values) / (count - 1)
                assert variance <= upper


def test_empirical_pcm_interval_is_variance_sensitive_but_never_infers_zero() -> None:
    energies = (1e-8,) * 1_000
    empirical = _empirical_pcm_floor_interval(energies, energies, 1.0, 0.05)
    hoeffding = _bounded_pcm_floor_interval(energies, energies, 1.0, 0.05)
    assert empirical[1] < hoeffding[1]
    zero = _empirical_pcm_floor_interval((0.0,) * 1_000, (0.0,) * 1_000, 1.0, 0.05)
    assert zero[0] == 0.0
    assert zero[1] > 0.0
    required = 1 + 7 * math.log(4 / 0.05) / (3 * 5e-9)
    assert 2e9 < required < 2.1e9
    with pytest.raises(ValueError, match="at least two"):
        _empirical_pcm_floor_interval((0.0,), (0.0,), 1.0, 0.05)


def test_empirical_pcm_interval_on_seeded_non_gaussian_noise() -> None:
    rng = Random(501)
    failures = 0
    repetitions = 500
    for _ in range(repetitions):
        samples = tuple(float(rng.random() < 0.2) for _ in range(64))
        low, high = _empirical_pcm_floor_interval(samples, samples, 1.0, 0.05)
        failures += not low <= 0.2 <= high
    assert failures < repetitions * 0.05
