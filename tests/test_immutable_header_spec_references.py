"""Immutable-header projection examples; the small serializer is not full JCS."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Never

import pytest

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None

_MAX_SAFE_INTEGER = 2**53 - 1
_TIERS = ("lite", "full", "pro", "ultra")


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate object member")
        result[key] = value
    return result


def _reject_constant(_value: str) -> Never:
    raise ValueError("non-finite number")


def _valid_text(text: str) -> bool:
    return all(
        not (
            0xD800 <= ord(char) <= 0xDFFF
            or 0xFDD0 <= ord(char) <= 0xFDEF
            or ord(char) & 0xFFFF in (0xFFFE, 0xFFFF)
        )
        for char in text
    )


def _validate_json(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite number")
        return value
    if isinstance(value, str):
        if not _valid_text(value):
            raise ValueError("invalid Unicode")
        return value
    if isinstance(value, list):
        return [_validate_json(item) for item in value]
    if isinstance(value, dict):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not _valid_text(key):
                raise ValueError("invalid Unicode")
            result[key] = _validate_json(item)
        return result
    raise ValueError("invalid JSON value")


def _parse_object(data: bytes) -> dict[str, JsonValue]:
    try:
        parsed: object = json.loads(
            data.decode("utf-8"), object_pairs_hook=_unique_pairs, parse_constant=_reject_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid JSON") from error
    value = _validate_json(parsed)
    if not isinstance(value, dict):
        raise ValueError("header must be an object")
    return value


def _object(value: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError("expected object")
    return value


def _array(value: JsonValue) -> list[JsonValue]:
    if not isinstance(value, list):
        raise ValueError("expected array")
    return value


def _project(header: dict[str, JsonValue]) -> dict[str, JsonValue]:
    result = copy.deepcopy(header)
    result.pop("presets", None)
    result.pop("user_metadata", None)
    _object(result.get("metadata")).pop("signatures", None)
    for control in _array(result.get("controls_schema")):
        _object(control).pop("user_metadata", None)
    stems = result.get("stems_registry")
    if stems is not None:
        for stem in _array(stems):
            _object(stem).pop("user_metadata", None)
    tiers = _object(result.get("slimmable_tiers"))
    for name in _TIERS:
        if name in tiers:
            _object(tiers[name]).pop("prefix_crc32c", None)
    return result


def _jcs_subset(value: JsonValue) -> str:
    """Serialize strings, literals, and safe integers; reject floating-point values."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        if abs(value) > _MAX_SAFE_INTEGER:
            raise ValueError("integer outside reference subset")
        return str(value)
    if isinstance(value, str):
        if not _valid_text(value):
            raise ValueError("invalid Unicode")
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ",".join(_jcs_subset(item) for item in value) + "]"
    if isinstance(value, dict):
        keys = sorted(value, key=lambda key: key.encode("utf-16be"))
        return (
            "{" + ",".join(_jcs_subset(key) + ":" + _jcs_subset(value[key]) for key in keys) + "}"
        )
    raise ValueError("float outside reference subset")


def _canonical_header(data: bytes) -> bytes:
    result = _jcs_subset(_project(_parse_object(data))).encode("utf-8")
    if data != result:
        raise ValueError("noncanonical or mutable header")
    return result


def _fixture() -> dict[str, JsonValue]:
    return _parse_object(
        b"""{
          "version":"future-format",
          "metadata":{"name":"amp","signatures":{"sig":"old"}},
          "user_metadata":{"stars":1},
          "presets":[{"name":"old"}],
          "controls_schema":[{"id":"gain","user_metadata":{"tag":"a"},
            "calibration":{"user_metadata":"signed"}}],
          "stems_registry":[{"stem_id":"a","user_metadata":{"tag":"b"}}],
          "slimmable_tiers":{
            "lite":{"channels":4,"prefix_crc32c":12},
            "future":{"prefix_crc32c":13}}
        }"""
    )


def test_projection_removes_only_designated_mutable_paths() -> None:
    original = _fixture()
    changed = copy.deepcopy(original)
    changed["presets"] = [{"name": "new"}]
    changed["user_metadata"] = {"stars": 5}
    _object(changed["metadata"])["signatures"] = {"sig": "new"}
    _object(_array(changed["controls_schema"])[0])["user_metadata"] = {"tag": "new"}
    _object(_array(changed["stems_registry"])[0])["user_metadata"] = {"tag": "new"}
    _object(_object(changed["slimmable_tiers"])["lite"])["prefix_crc32c"] = 99
    projected = _project(original)
    assert _project(changed) == projected

    controls = _array(projected["controls_schema"])
    assert _object(_object(controls[0])["calibration"])["user_metadata"] == "signed"
    assert _object(_object(projected["slimmable_tiers"])["future"])["prefix_crc32c"] == 13
    _object(_object(controls[0])["calibration"])["user_metadata"] = "changed"
    assert (
        hashlib.sha256(_jcs_subset(projected).encode()).digest()
        != hashlib.sha256(_jcs_subset(_project(original)).encode()).digest()
    )


def test_control_selectors_and_projection_remain_in_signed_header() -> None:
    original = _fixture()
    control = _object(_array(original["controls_schema"])[0])
    control["conditional_bounds"] = [
        {"when": [{"control_id": "mode", "option_id": "on"}], "max": 8}
    ]
    control["calibration"] = {
        "conditional_physical_anchors": [
            {"when": [{"control_id": "mode", "option_id": "on"}], "anchors": [1, 2, 3]}
        ]
    }
    original["control_projection"] = {
        "profile": "CP1",
        "represented_switch_options": [{"control_id": "mode", "option_id": "on"}],
    }
    digest = hashlib.sha256(_jcs_subset(_project(original)).encode()).digest()

    changed_bound = copy.deepcopy(original)
    changed_control = _object(_array(changed_bound["controls_schema"])[0])
    _object(_array(changed_control["conditional_bounds"])[0])["max"] = 9

    changed_selector = copy.deepcopy(original)
    changed_control = _object(_array(changed_selector["controls_schema"])[0])
    bound = _object(_array(changed_control["conditional_bounds"])[0])
    _object(_array(bound["when"])[0])["option_id"] = "off"

    changed_curve = copy.deepcopy(original)
    changed_control = _object(_array(changed_curve["controls_schema"])[0])
    calibration = _object(changed_control["calibration"])
    _array(_object(_array(calibration["conditional_physical_anchors"])[0])["anchors"])[1] = 5

    changed_projection = copy.deepcopy(original)
    projection = _object(changed_projection["control_projection"])
    _object(_array(projection["represented_switch_options"])[0])["option_id"] = "other"

    for changed in (changed_bound, changed_selector, changed_curve, changed_projection):
        assert hashlib.sha256(_jcs_subset(_project(changed)).encode()).digest() != digest


def test_invalid_or_duplicate_json_is_rejected_before_projection() -> None:
    for data, error in (
        (b'{"a":1,"\\u0061":2}', "duplicate object member"),
        (b'{"name":"\\ud800"}', "invalid Unicode"),
        (b'{"name":"\\ufdd0"}', "invalid Unicode"),
        (b'{"name":"\\uffff"}', "invalid Unicode"),
        (b'{"name":"\xff"}', "invalid JSON"),
        (b'{"number":1e999}', "non-finite number"),
        (b'{"number":NaN}', "non-finite number"),
        (b"[]", "header must be an object"),
    ):
        with pytest.raises(ValueError, match=error):
            _parse_object(data)


def test_reference_subset_uses_utf16_order_and_does_not_normalize_unicode() -> None:
    keys: dict[str, JsonValue] = {"\ue000": 2, "\U00010000": 1}
    assert _jcs_subset(keys) == '{"\U00010000":1,"\ue000":2}'
    assert sorted(keys) == ["\ue000", "\U00010000"]
    assert _jcs_subset({"name": "\u00e9"}) != _jcs_subset({"name": "e\u0301"})


def test_projected_wire_bytes_are_canonical_and_contain_no_mutable_members() -> None:
    projected = _project(_fixture())
    canonical = _jcs_subset(projected).encode("utf-8")
    assert _canonical_header(canonical) == canonical
    with pytest.raises(ValueError, match="noncanonical"):
        _canonical_header(b" " + canonical)
    with pytest.raises(ValueError, match="noncanonical"):
        _canonical_header(_jcs_subset(_fixture()).encode("utf-8"))
    with pytest.raises(ValueError, match="noncanonical"):
        _canonical_header(canonical.replace(b'"channels":4', b'"channels":-0'))


def test_reference_subset_rejects_out_of_range_integers_and_floats() -> None:
    assert _jcs_subset(_MAX_SAFE_INTEGER) == "9007199254740991"
    with pytest.raises(ValueError, match="integer outside"):
        _jcs_subset(_MAX_SAFE_INTEGER + 1)
    with pytest.raises(ValueError, match="float outside"):
        _jcs_subset(1.0)
