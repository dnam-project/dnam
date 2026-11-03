"""Composed option-selection references, not the production tempo-control implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import pytest

from test_tempo_physical_inversion_spec_references import _domain, _invert


@dataclass(frozen=True)
class _Option:
    name: str
    anchors: tuple[float, float, float]
    low: float = -1.0
    high: float = 1.0
    observed: bool = True
    active: bool = True
    tolerance: float | None = None
    calibration_error: float | None = None
    forward_error: float | None = None
    declared: tuple[float, float] = (-1.0, 1.0)


@dataclass(frozen=True)
class _Candidate:
    option: str
    coordinate: float
    physical: float
    shifted_target: float
    shift: int
    residual: Fraction
    midpoint_distance: Fraction
    evidence: str
    index: int


@dataclass(frozen=True)
class _Result:
    kind: str
    candidate: _Candidate | None
    rejected: tuple[str, ...]


def _validate_option(option: _Option, *, flush_subnormals: bool) -> bool:
    evidence = (option.tolerance, option.calibration_error, option.forward_error)
    if any(value is not None and (not math.isfinite(value) or value < 0) for value in evidence):
        return False
    try:
        _domain(
            option.declared,
            (option.low, option.high),
            option.anchors,
            flush_physical=flush_subnormals,
            flush_normalized=flush_subnormals,
        )
    except ValueError:
        return False
    return True


def _evidence(option: _Option, residual: Fraction) -> str:
    if option.tolerance is None or option.calibration_error is None or option.forward_error is None:
        return "uncertified"
    tolerance = Fraction.from_float(option.tolerance)
    error = Fraction.from_float(option.calibration_error) + Fraction.from_float(
        option.forward_error
    )
    if residual + error <= tolerance:
        return "certified"
    if max(Fraction(0), residual - error) > tolerance:
        return "outside-tolerance"
    return "uncertified"


def _scaled_exact(target: float, shift: int) -> float | None:
    try:
        result = math.ldexp(target, shift)
    except OverflowError:
        return None
    if not math.isfinite(result) or result <= 0:
        return None
    exact = Fraction.from_float(target)
    exact = exact * (1 << shift) if shift >= 0 else exact / (1 << -shift)
    return result if Fraction.from_float(result) == exact else None


def _candidate(
    option: _Option,
    index: int,
    target: float,
    shift: int,
    *,
    flush_subnormals: bool,
) -> _Candidate | None:
    wanted = Fraction.from_float(target)
    domain = _domain(
        option.declared,
        (option.low, option.high),
        option.anchors,
        flush_physical=flush_subnormals,
        flush_normalized=flush_subnormals,
    )
    result = _invert(domain, target)
    if result is None:
        return None
    endpoints = domain.continuous_values
    return _Candidate(
        option.name,
        result.normalized,
        result.physical,
        target,
        shift,
        result.residual,
        abs(wanted - (endpoints[0] + endpoints[1]) / 2),
        _evidence(option, result.residual),
        index,
    )


def _select(
    target: float,
    options: tuple[_Option, ...],
    current: str | None,
    *,
    fold: bool,
    max_shift: int,
    flush_subnormals: bool = False,
    initial_default: str | None = None,
) -> _Result:
    if not math.isfinite(target) or target <= 0 or not 0 <= max_shift <= 2097:
        raise ValueError("invalid requested tempo target")
    if len({option.name for option in options}) != len(options):
        raise ValueError("duplicate option names")
    active = [
        (index, option) for index, option in enumerate(options) if option.observed and option.active
    ]
    rejected = tuple(
        option.name
        for _, option in active
        if not _validate_option(option, flush_subnormals=flush_subnormals)
    )
    eligible = [
        (index, option)
        for index, option in active
        if _validate_option(option, flush_subnormals=flush_subnormals)
    ]
    for magnitude in range((max_shift if fold else 0) + 1):
        shifts = (0,) if magnitude == 0 else (-magnitude, magnitude)
        candidates = []
        for shift in shifts:
            shifted_target = _scaled_exact(target, shift)
            if shifted_target is None:
                continue
            for index, option in eligible:
                candidate = _candidate(
                    option, index, shifted_target, shift, flush_subnormals=flush_subnormals
                )
                if candidate is not None:
                    candidates.append(candidate)
        if candidates:
            selected = min(
                candidates,
                key=lambda candidate: (
                    candidate.evidence != "certified",
                    candidate.residual,
                    candidate.midpoint_distance,
                    candidate.option != current,
                    candidate.index,
                    candidate.shift,
                ),
            )
            return _Result("folded" if selected.shift else "unmodified", selected, rejected)
    fallback_name = current if current is not None else initial_default
    selected_option = next((option for _, option in active if option.name == fallback_name), None)
    if selected_option is None and active and fallback_name is not None:
        selected_option = active[0][1]
    if selected_option is None or not _validate_option(
        selected_option, flush_subnormals=flush_subnormals
    ):
        return _Result("unavailable", None, rejected)
    domain = _domain(
        selected_option.declared,
        (selected_option.low, selected_option.high),
        selected_option.anchors,
        flush_physical=flush_subnormals,
        flush_normalized=flush_subnormals,
    )
    low_value, high_value = domain.continuous_values
    wanted = Fraction.from_float(target)
    if min(low_value, high_value) <= wanted <= max(low_value, high_value):
        return _Result("unavailable", None, rejected)
    result = _invert(domain, target, require_inside=False)
    if result is None:
        raise ValueError("valid fallback has no representable coordinate")
    fallback = _Candidate(
        selected_option.name,
        result.normalized,
        result.physical,
        target,
        0,
        result.residual,
        abs(wanted - (low_value + high_value) / 2),
        _evidence(selected_option, result.residual),
        next(index for index, option in eligible if option.name == selected_option.name),
    )
    return _Result("clamped", fallback, rejected)


def test_exact_representable_tie_chooses_lower_coordinate() -> None:
    option = _Option("main", (1.0, 2.0, 3.0), tolerance=1.0, calibration_error=0, forward_error=0)
    result = _select(3.0 - 2**-25, (option,), None, fold=False, max_shift=0)
    assert result.kind == "unmodified"
    assert result.candidate is not None
    assert result.candidate.coordinate == 1.0 - 2**-24
    assert result.candidate.residual == Fraction(1, 2**25)
    assert result.candidate.evidence == "certified"


def test_current_then_declared_option_breaks_exact_ties() -> None:
    options = (
        _Option("first", (1.0, 2.0, 3.0)),
        _Option("second", (1.0, 2.0, 3.0)),
    )
    retained = _select(2.0, options, "second", fold=False, max_shift=0)
    initial = _select(2.0, options, None, fold=False, max_shift=0)
    assert retained.candidate is not None
    assert retained.candidate.option == "second"
    assert initial.candidate is not None
    assert initial.candidate.option == "first"


def test_certified_option_is_preferred_over_uncertified_current_option() -> None:
    options = (
        _Option("current", (1.0, 2.0, 3.0)),
        _Option("certified", (1.0, 2.0, 3.0), tolerance=0, calibration_error=0, forward_error=0),
    )
    result = _select(2.0, options, "current", fold=False, max_shift=0)
    assert result.candidate is not None
    assert result.candidate.option == "certified"
    assert result.candidate.evidence == "certified"


def test_option_ranking_uses_reachable_physical_coordinates() -> None:
    lower = float(2**30)
    coarse = _Option(
        "coarse",
        (0.0, 1.0, 2.0),
        low=lower,
        high=lower + 128.0,
        declared=(lower, lower + 128.0),
        tolerance=0.1,
        calibration_error=0,
        forward_error=0,
    )
    fine = _Option(
        "fine",
        (0.0, 1.0, 2.0),
        tolerance=0.1,
        calibration_error=0,
        forward_error=0,
    )
    coarse_only = _select(1.0, (coarse,), None, fold=False, max_shift=0)
    assert coarse_only.candidate is not None
    assert coarse_only.candidate.coordinate == -1.0
    assert coarse_only.candidate.physical == lower
    assert coarse_only.candidate.residual == 1
    assert coarse_only.candidate.evidence == "outside-tolerance"
    selected = _select(1.0, (coarse, fine), "coarse", fold=False, max_shift=0)
    assert selected.candidate is not None
    assert selected.candidate.option == "fine"
    assert selected.candidate.physical == 0.0
    assert selected.candidate.evidence == "certified"


def test_missing_or_insufficient_error_evidence_is_not_certification() -> None:
    missing = _Option("missing", (1.0, 2.0, 3.0), tolerance=0.1)
    bounded = _Option(
        "bounded", (1.0, 2.0, 3.0), tolerance=0.1, calibration_error=0.2, forward_error=0
    )
    for option in (missing, bounded):
        result = _select(2.0, (option,), None, fold=False, max_shift=0)
        assert result.candidate is not None
        assert result.candidate.evidence == "uncertified"


def test_error_evidence_uses_inclusive_upper_and_strict_lower_bounds() -> None:
    option = _Option(
        "bounded", (1.0, 2.0, 3.0), tolerance=1.0, calibration_error=0.25, forward_error=0.75
    )
    assert _evidence(option, Fraction(0)) == "certified"
    assert _evidence(option, Fraction(2)) == "uncertified"
    assert _evidence(option, Fraction(3)) == "outside-tolerance"


def test_ftz_changes_the_safe_coordinate_and_can_prove_tolerance_failure() -> None:
    scale = float(2**126)
    option = _Option(
        "rate",
        (-scale, 0.0, scale),
        low=0.0,
        high=1.0,
        tolerance=0.01,
        calibration_error=0.0,
        forward_error=0.0,
    )
    gradual = _select(0.125, (option,), None, fold=False, max_shift=0)
    flushed = _select(0.125, (option,), None, fold=False, max_shift=0, flush_subnormals=True)
    assert gradual.candidate is not None
    assert gradual.candidate.coordinate == 2**-129
    assert gradual.candidate.evidence == "certified"
    assert flushed.candidate is not None
    assert flushed.candidate.coordinate == 0
    assert flushed.candidate.evidence == "outside-tolerance"


def test_folding_preserves_unmodified_targets_and_uses_deterministic_ties() -> None:
    low = _Option("low", (4.0, 5.0, 6.0))
    high = _Option("high", (19.0, 20.0, 21.0))
    direct = _Option("direct", (8.0, 10.0, 12.0))
    unmodified = _select(10.0, (low, high, direct), None, fold=True, max_shift=1)
    assert unmodified.kind == "unmodified"
    assert unmodified.candidate is not None
    assert unmodified.candidate.option == "direct"
    retained = _select(10.0, (low, high), "high", fold=True, max_shift=1)
    initial = _select(10.0, (low, high), None, fold=True, max_shift=1)
    assert retained.candidate is not None
    assert retained.candidate.shift == 1
    assert initial.candidate is not None
    assert initial.candidate.shift == -1
    assert retained.kind == initial.kind == "folded"


def test_octave_scaling_rejects_overflow_and_inexact_underflow() -> None:
    minimum = float.fromhex("0x0.0000000000001p-1022")
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    assert _scaled_exact(minimum, -1) is None
    assert _scaled_exact(maximum, 1) is None
    assert _scaled_exact(3 * minimum, -1) is None
    assert _scaled_exact(4 * minimum, -1) == 2 * minimum
    assert _scaled_exact(minimum, 2097) == 2**1023


def test_unobserved_invalid_and_inactive_options_never_bypass_safe_selection() -> None:
    options = (
        _Option("unobserved", (1.0, 2.0, 3.0), observed=False),
        _Option("inactive", (1.0, 2.0, 3.0), active=False),
        _Option("ambiguous", (0.0, 100.0, 10.0)),
        _Option("descending", (3.0, 2.0, 1.0)),
    )
    result = _select(2.0, options, "unobserved", fold=False, max_shift=0)
    assert result.candidate is not None
    assert result.candidate.option == "descending"
    assert result.rejected == ("ambiguous",)
    assert result.candidate.coordinate == 0


def test_clamping_and_empty_ftz_safe_range_report_distinct_outcomes() -> None:
    option = _Option("range", (10.0, 20.0, 30.0))
    clamped = _select(35.0, (option,), "range", fold=False, max_shift=0)
    assert clamped.kind == "clamped"
    assert clamped.candidate is not None
    assert clamped.candidate.coordinate == 1.0
    tiny = _Option("tiny", (-1.0, 0.0, 1.0), low=2**-130, high=2**-128)
    unavailable = _select(2**-129, (tiny,), "tiny", fold=False, max_shift=0, flush_subnormals=True)
    assert unavailable.kind == "unavailable"
    assert unavailable.candidate is None


def test_singleton_safe_interval_has_only_its_observed_coordinate() -> None:
    option = _Option("fixed", (1.0, 2.0, 3.0), low=0.25, high=0.25)
    exact = _select(2.25, (option,), None, fold=False, max_shift=0)
    clamped = _select(2.5, (option,), None, fold=False, max_shift=0, initial_default="fixed")
    assert exact.kind == "unmodified"
    assert exact.candidate is not None
    assert exact.candidate.coordinate == 0.25
    assert clamped.kind == "clamped"
    assert clamped.candidate is not None
    assert clamped.candidate.coordinate == 0.25


def test_fallback_keeps_the_current_option_or_uses_the_active_signed_default() -> None:
    invalid_current = _Option("current", (0.0, 100.0, 10.0))
    safe = _Option("safe", (10.0, 20.0, 30.0))
    unavailable = _select(40.0, (invalid_current, safe), "current", fold=False, max_shift=0)
    assert unavailable.kind == "unavailable"
    initial = _select(
        40.0, (invalid_current, safe), None, fold=False, max_shift=0, initial_default="safe"
    )
    assert initial.kind == "clamped"
    assert initial.candidate is not None
    assert initial.candidate.option == "safe"


@pytest.mark.parametrize("target", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_tempo_targets_are_not_silently_clamped(target: float) -> None:
    with pytest.raises(ValueError, match="invalid requested tempo target"):
        _select(target, (_Option("range", (0.0, 1.0, 2.0)),), None, fold=False, max_shift=0)
