"""Initial mutable JSON examples, not a production schema or editor."""

from __future__ import annotations

import struct
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from test_open_layout_spec_references import _frame_for_test, _read_mutable_frame

_TOP_LEVEL = {"version", "presets", "user_metadata", "control_user_metadata", "stem_user_metadata"}
_PRESET = {"id", "name", "description", "controls", "user_metadata"}


def _validate_shape(value: dict[str, object]) -> None:
    if value.keys() - _TOP_LEVEL:
        raise ValueError("unknown mutable member")
    if "version" in value and (not isinstance(value["version"], Decimal) or value["version"] != 1):
        raise ValueError("unsupported version")
    for key in ("user_metadata", "control_user_metadata", "stem_user_metadata"):
        if key in value and not isinstance(value[key], dict):
            raise ValueError("invalid metadata object")
    for key in ("control_user_metadata", "stem_user_metadata"):
        objects = value.get(key, {})
        if not isinstance(objects, dict) or any(
            not isinstance(item, dict) for item in objects.values()
        ):
            raise ValueError("invalid keyed metadata")
    presets = value.get("presets", [])
    if not isinstance(presets, list):
        raise ValueError("invalid presets")
    ids: set[str] = set()
    for preset in presets:
        if not isinstance(preset, dict) or preset.keys() - _PRESET:
            raise ValueError("invalid preset")
        identifier = preset.get("id")
        name = preset.get("name")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("invalid preset id")
        ids.add(identifier)
        if not isinstance(name, str) or not name:
            raise ValueError("invalid preset name")
        if (
            "description" in preset
            and preset["description"] is not None
            and not isinstance(preset["description"], str)
        ):
            raise ValueError("invalid description")
        if "user_metadata" in preset and not isinstance(preset["user_metadata"], dict):
            raise ValueError("invalid preset metadata")
        controls = preset.get("controls")
        if not isinstance(controls, dict):
            raise ValueError("invalid control map")
        for control_value in controls.values():
            if isinstance(control_value, Decimal) and control_value.is_finite():
                continue
            if not isinstance(control_value, str):
                raise ValueError("invalid control value")


def _read_document(payload: bytes) -> dict[str, object]:
    document = _read_mutable_frame(_frame_for_test(payload, 16 + len(payload)))
    _validate_shape(document)
    return document


def test_empty_and_open_user_metadata_preserve_exact_values() -> None:
    assert _read_document(b"{}") == {}
    payload = (
        b'{"version":1,"presets":[{"id":"warm","name":"Warm","controls":{"gain":0.4},'
        b'"user_metadata":{"note":"test"}}],"user_metadata":{"tags":["amp"],'
        b'"large":9007199254740993},"control_user_metadata":{"gain":{"color":"red"},'
        b'"unbound":{"note":"keep"}},"stem_user_metadata":{"stem-1":{"tag":"keep"}}}'
    )
    document = _read_document(payload)
    assert document["user_metadata"] == {
        "tags": ["amp"],
        "large": Decimal("9007199254740993"),
    }
    assert document["control_user_metadata"] == {
        "gain": {"color": "red"},
        "unbound": {"note": "keep"},
    }
    assert document["stem_user_metadata"] == {"stem-1": {"tag": "keep"}}


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        (b'{"version":true}', "unsupported version"),
        (b'{"version":2}', "unsupported version"),
        (b'{"unknown":{}}', "unknown mutable member"),
        (b'{"presets":{}}', "invalid presets"),
        (b'{"control_user_metadata":{"gain":2}}', "invalid keyed metadata"),
        (b'{"presets":[{"id":"p","name":"P","controls":{},"unknown":0}]}', "invalid preset"),
        (
            b'{"presets":[{"id":"p","name":"P","controls":{}},{"id":"p","name":"Q","controls":{}}]}',
            "invalid preset id",
        ),
        (b'{"presets":[{"id":"p","name":"P","controls":{"gain":true}}]}', "invalid control value"),
        (b'{"presets":[{"id":"p","name":"P"}]}', "invalid control map"),
    ],
)
def test_invalid_mutable_member_shapes_are_rejected(payload: bytes, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        _read_document(payload)


def test_preset_decimal_can_require_direct_binary32_rounding() -> None:
    with localcontext() as context:
        context.prec = 100
        value = Decimal(1) + Decimal(2) ** -24 + Decimal(2) ** -55
    payload = (
        b'{"presets":[{"id":"p","name":"P","controls":{"gain":'
        + str(value).encode("ascii")
        + b"}}]}"
    )
    document = _read_document(payload)
    presets = document["presets"]
    assert isinstance(presets, list)
    controls = presets[0]["controls"]
    assert isinstance(controls, dict)
    decoded = controls["gain"]
    assert isinstance(decoded, Decimal)
    exact = Fraction(decoded)
    lower = Fraction(1)
    upper = lower + Fraction(1, 2**23)
    assert upper - exact < exact - lower
    assert struct.unpack("<f", struct.pack("<f", float(decoded)))[0] == 1.0
