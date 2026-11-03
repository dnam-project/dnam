"""Conditional-selector examples, not production loader or control-bus tests."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import pytest

_Selector = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _Switch:
    control_id: str
    options: tuple[str, ...]
    active: tuple[str, ...]


@dataclass(frozen=True)
class _Bound:
    when: _Selector
    minimum: float | None
    maximum: float | None


@dataclass(frozen=True)
class _Curve:
    when: _Selector
    anchors: tuple[float, float, float]


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("invalid number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("invalid number") from error
    if not math.isfinite(result):
        raise ValueError("invalid number")
    return result


def _selector(value: object, switches: tuple[_Switch, ...]) -> _Selector:
    if not isinstance(value, list) or not value:
        raise ValueError("invalid conjunction")
    result: list[tuple[str, str]] = []
    previous = -1
    for item in value:
        if not isinstance(item, dict) or set(item) != {"control_id", "option_id"}:
            raise ValueError("invalid selector object")
        control_id, option_id = item["control_id"], item["option_id"]
        if not isinstance(control_id, str) or not control_id:
            raise ValueError("invalid control identity")
        if not isinstance(option_id, str) or not option_id:
            raise ValueError("invalid option identity")
        index = next(
            (index for index, switch in enumerate(switches) if switch.control_id == control_id),
            -1,
        )
        if index < 0:
            raise ValueError("unknown switch")
        if index <= previous:
            raise ValueError("duplicate or unordered switch")
        if option_id not in switches[index].options:
            raise ValueError("unknown option")
        result.append((control_id, option_id))
        previous = index
    return tuple(result)


def _bounds(value: object, switches: tuple[_Switch, ...]) -> tuple[_Bound, ...]:
    if not isinstance(value, list):
        raise ValueError("invalid bounds array")
    result: list[_Bound] = []
    seen: set[_Selector] = set()
    for item in value:
        if (
            not isinstance(item, dict)
            or not {"when"} <= set(item) <= {"when", "min", "max"}
            or not ({"min", "max"} & set(item))
        ):
            raise ValueError("invalid bound object")
        when = _selector(item["when"], switches)
        if when in seen:
            raise ValueError("duplicate conjunction")
        seen.add(when)
        minimum = _number(item["min"]) if "min" in item else None
        maximum = _number(item["max"]) if "max" in item else None
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("reversed bound")
        result.append(_Bound(when, minimum, maximum))
    return tuple(result)


def _curves(value: object, switches: tuple[_Switch, ...]) -> tuple[_Curve, ...]:
    if not isinstance(value, list):
        raise ValueError("invalid calibration array")
    result: list[_Curve] = []
    seen: set[_Selector] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {"when", "anchors"}:
            raise ValueError("invalid calibration object")
        when = _selector(item["when"], switches)
        if when in seen:
            raise ValueError("duplicate conjunction")
        seen.add(when)
        anchors = item["anchors"]
        if not isinstance(anchors, list) or len(anchors) != 3:
            raise ValueError("invalid anchors")
        result.append(_Curve(when, (_number(anchors[0]), _number(anchors[1]), _number(anchors[2]))))
    return tuple(result)


def _matches(
    selectors: tuple[_Selector, ...],
    switches: tuple[_Switch, ...],
    effective: dict[str, str],
) -> tuple[bool, ...]:
    by_id = {switch.control_id: switch for switch in switches}
    for control_id in {control for when in selectors for control, _ in when}:
        if control_id not in effective:
            raise ValueError("missing effective switch state")
        if effective[control_id] not in by_id[control_id].active:
            raise ValueError("inactive effective option")
    return tuple(
        all(effective[control_id] == option_id for control_id, option_id in when)
        for when in selectors
    )


def _safe_interval(
    base: tuple[float, float],
    rules: tuple[_Bound, ...],
    switches: tuple[_Switch, ...],
    effective: dict[str, str],
) -> tuple[float, float]:
    lower, upper = base
    matched = _matches(tuple(rule.when for rule in rules), switches, effective)
    for rule, selected in zip(rules, matched, strict=True):
        if selected and rule.minimum is not None:
            lower = max(lower, rule.minimum)
        if selected and rule.maximum is not None:
            upper = min(upper, rule.maximum)
    if lower > upper:
        raise ValueError("empty safe intersection")
    return lower, upper


def _select_curve(
    rules: tuple[_Curve, ...],
    switches: tuple[_Switch, ...],
    effective: dict[str, str],
    default: tuple[float, float, float] | None = None,
) -> tuple[float, float, float]:
    matched = _matches(tuple(rule.when for rule in rules), switches, effective)
    applicable = [rule for rule, selected in zip(rules, matched, strict=True) if selected]
    if not applicable:
        if default is None:
            raise ValueError("calibration unavailable")
        return default
    size = max(len(rule.when) for rule in applicable)
    best = [rule for rule in applicable if len(rule.when) == size]
    if len(best) != 1:
        raise ValueError("ambiguous calibration")
    return best[0].anchors


def _when(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"control_id": control, "option_id": option} for control, option in pairs]


_SWITCHES = (
    _Switch("mode=variant", ("x", "off"), ("x", "off")),
    _Switch("mode", ("variant=x", "off", "masked"), ("variant=x", "off")),
)
_EFFECTIVE = {"mode=variant": "x", "mode": "variant=x"}


def test_identifiers_with_equals_remain_distinct_and_bounds_intersect() -> None:
    left = ("mode=variant", "x")
    right = ("mode", "variant=x")
    assert "=".join(left) == "=".join(right)
    rules = _bounds(
        [
            {"when": _when(left), "max": 8.0},
            {"when": _when(right), "min": 2.0},
            {"when": _when(left, right), "min": 4.0, "max": 6.0},
        ],
        _SWITCHES,
    )
    assert _safe_interval((0.0, 10.0), rules, _SWITCHES, _EFFECTIVE) == (4.0, 6.0)
    assert _safe_interval((0.0, 10.0), rules[::-1], _SWITCHES, _EFFECTIVE) == (4.0, 6.0)
    assert _safe_interval((0.0, 10.0), rules, _SWITCHES, {"mode=variant": "x", "mode": "off"}) == (
        0.0,
        8.0,
    )


def test_exact_unicode_and_masked_options_do_not_alias() -> None:
    switches = (_Switch("tone", ("é", "e\u0301"), ("é",)),)
    rules = _bounds([{"when": _when(("tone", "e\u0301")), "min": 0.8}], switches)
    assert _safe_interval((0.0, 1.0), rules, switches, {"tone": "é"}) == (0.0, 1.0)
    with pytest.raises(ValueError, match="inactive effective option"):
        _safe_interval((0.0, 1.0), rules, switches, {"tone": "e\u0301"})
    with pytest.raises(ValueError, match="unknown option"):
        _bounds([{"when": _when(("tone", "É")), "min": 0.8}], switches)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (None, "invalid bounds array"),
        ([{"when": [], "min": 1.0}], "invalid conjunction"),
        ([{"when": [{"control_id": "mode"}], "min": 1.0}], "invalid selector object"),
        ([{"when": _when(("missing", "x")), "min": 1.0}], "unknown switch"),
        ([{"when": _when(("mode", "missing")), "min": 1.0}], "unknown option"),
        (
            [{"when": _when(("mode", "off"), ("mode", "variant=x")), "min": 1}],
            "duplicate or unordered",
        ),
        (
            [{"when": _when(("mode", "off"), ("mode=variant", "x")), "min": 1}],
            "duplicate or unordered",
        ),
        ([{"when": _when(("mode", "off"))}], "invalid bound object"),
        ([{"when": _when(("mode", "off")), "min": True}], "invalid number"),
        ([{"when": _when(("mode", "off")), "max": math.inf}], "invalid number"),
        ([{"when": _when(("mode", "off")), "min": 2, "max": 1}], "reversed bound"),
        ([{"when": _when(("mode", "off")), "min": 1, "extra": 0}], "invalid bound object"),
        (
            [
                {"when": _when(("mode", "off")), "min": 1},
                {"when": _when(("mode", "off")), "max": 2},
            ],
            "duplicate conjunction",
        ),
    ],
)
def test_malformed_bounds_are_rejected(payload: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _bounds(payload, _SWITCHES)


def test_missing_state_or_empty_intersection_never_publishes_candidate() -> None:
    rules = _bounds(
        [
            {"when": _when(("mode=variant", "x")), "min": 7.0},
            {"when": _when(("mode", "variant=x")), "max": 3.0},
        ],
        _SWITCHES,
    )
    current = {"mode=variant": "off", "mode": "off"}
    with pytest.raises(ValueError, match="empty safe intersection"):
        _safe_interval((0.0, 10.0), rules, _SWITCHES, _EFFECTIVE)
    with pytest.raises(ValueError, match="missing effective switch state"):
        _safe_interval((0.0, 10.0), rules, _SWITCHES, {"mode=variant": "off"})
    assert current == {"mode=variant": "off", "mode": "off"}


def test_f32_nearest_rounding_can_cross_a_signed_safety_bound() -> None:
    rounded = struct.unpack("<f", struct.pack("<f", 0.1))[0]
    bits = struct.unpack("<I", struct.pack("<f", rounded))[0]
    below = struct.unpack("<f", struct.pack("<I", bits - 1))[0]
    assert below < 0.1 < rounded
    # The adjacent values bracket the entire singleton interval [0.1, 0.1].
    assert bits - struct.unpack("<I", struct.pack("<f", below))[0] == 1


def test_calibration_specificity_ambiguity_and_missing_fallback() -> None:
    left, right = ("mode=variant", "x"), ("mode", "variant=x")
    rows = [
        {"when": _when(left), "anchors": [1.0, 2.0, 3.0]},
        {"when": _when(right), "anchors": [10.0, 20.0, 30.0]},
    ]
    rules = _curves(rows, _SWITCHES)
    with pytest.raises(ValueError, match="ambiguous calibration"):
        _select_curve(rules, _SWITCHES, _EFFECTIVE, (0.0, 1.0, 2.0))
    with pytest.raises(ValueError, match="ambiguous calibration"):
        _select_curve(rules[::-1], _SWITCHES, _EFFECTIVE)
    specific = _curves([*rows, {"when": _when(left, right), "anchors": [9.0, 0.0, 9.0]}], _SWITCHES)
    assert _select_curve(specific, _SWITCHES, _EFFECTIVE) == (9.0, 0.0, 9.0)
    other = {"mode=variant": "off", "mode": "off"}
    assert _select_curve(rules, _SWITCHES, other, (0.0, 1.0, 2.0)) == (0.0, 1.0, 2.0)
    with pytest.raises(ValueError, match="calibration unavailable"):
        _select_curve(rules, _SWITCHES, other)


def test_malformed_calibration_is_rejected() -> None:
    row = {"when": _when(("mode", "off")), "anchors": [1.0, 2.0, 3.0]}
    with pytest.raises(ValueError, match="invalid calibration array"):
        _curves({"mode=off": [1.0, 2.0, 3.0]}, _SWITCHES)
    with pytest.raises(ValueError, match="duplicate conjunction"):
        _curves([row, row], _SWITCHES)
    with pytest.raises(ValueError, match="invalid anchors"):
        _curves([{"when": row["when"], "anchors": [1.0, 2.0]}], _SWITCHES)
    with pytest.raises(ValueError, match="invalid number"):
        _curves([{"when": row["when"], "anchors": [1.0, math.nan, 3.0]}], _SWITCHES)
