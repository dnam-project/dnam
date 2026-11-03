"""Open-container physical layout examples, not a production verifier."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from decimal import Decimal
from typing import Never

import pytest

_U64_MAX = 2**64 - 1
_PREAMBLE_SIZE = 64
_DIRECTORY_SIZE = 128
_HEADER_OFFSET = _PREAMBLE_SIZE + _DIRECTORY_SIZE
_MUTABLE_HEADER_SIZE = 16
_MUTABLE_FOOTER_SIZE = 16
_REFERENCE_MAX_MUTABLE_CAPACITY = 4_096


@dataclass(frozen=True)
class _Layout:
    maximum_tier: int
    header: tuple[int, int]
    shared: tuple[int, int]
    authentication: tuple[int, int]
    mutable: tuple[int, int]
    tier_start: int
    tier_ends: tuple[int, int, int, int]


def _checked_end(start: int, length: int) -> int:
    if start < 0 or length < 0 or start + length > _U64_MAX:
        raise ValueError("invalid extent")
    return start + length


def _align64(value: int) -> int:
    if value > _U64_MAX - 63:
        raise ValueError("invalid alignment")
    return (value + 63) & ~63


def _example_file(
    header: bytes,
    shared: bytes,
    authentication: bytes,
    mutable: bytes,
    tiers: tuple[bytes, ...],
    *,
    auth_kind: int = 0,
) -> bytes:
    if not header or not authentication or not 1 <= len(tiers) <= 4 or auth_kind not in (0, 1):
        raise ValueError("invalid inputs")
    if any(not tier or len(tier) % 64 for tier in tiers):
        raise ValueError("invalid tier")
    header_offset = _HEADER_OFFSET
    shared_offset = _align64(header_offset + len(header))
    authentication_offset = _align64(shared_offset + len(shared))
    tier_start = _align64(authentication_offset + len(authentication))
    ends: list[int] = []
    end = tier_start
    for tier in tiers:
        end += len(tier)
        ends.append(end)
    ends.extend([0] * (4 - len(ends)))

    preamble = struct.pack(
        "<4sHHBBBBI48s", b"DNAM", 2, 0, 0, auth_kind, len(tiers) - 1, 0, 48_000, bytes(48)
    )
    directory = struct.pack(
        "<13Q24s",
        header_offset,
        len(header),
        shared_offset,
        len(shared),
        authentication_offset,
        len(authentication),
        0,
        0,
        tier_start,
        *ends,
        bytes(24),
    )
    output = bytearray(end)
    output[:_PREAMBLE_SIZE] = preamble
    output[_PREAMBLE_SIZE:_HEADER_OFFSET] = directory
    output[header_offset : header_offset + len(header)] = header
    output[shared_offset : shared_offset + len(shared)] = shared
    output[authentication_offset : authentication_offset + len(authentication)] = authentication
    offset = tier_start
    for tier in tiers:
        output[offset : offset + len(tier)] = tier
        offset += len(tier)
    return _append_mutable(bytes(output), mutable or _frame_for_test(b"{}", 18))


def _append_mutable(prefix: bytes, mutable: bytes) -> bytes:
    return prefix + mutable + struct.pack("<4sQI", b"DMUE", len(mutable), 0)


def _validate_layout(data: bytes) -> tuple[_Layout, int]:
    if len(data) < _HEADER_OFFSET:
        raise ValueError("short structure")
    magic, major, minor, profile, auth_kind, maximum, reserved, rate, padding = struct.unpack(
        "<4sHHBBBBI48s", data[:_PREAMBLE_SIZE]
    )
    if (
        magic != b"DNAM"
        or (major, minor) != (2, 0)
        or profile != 0
        or auth_kind not in (0, 1)
        or maximum > 3
        or reserved
        or rate != 48_000
        or any(padding)
    ):
        raise ValueError("invalid preamble")
    fields = struct.unpack("<13Q24s", data[_PREAMBLE_SIZE:_HEADER_OFFSET])
    (
        header_offset,
        header_length,
        shared_offset,
        shared_length,
        auth_offset,
        auth_length,
        reserved_offset,
        reserved_capacity,
        tier_start,
        lite_end,
        full_end,
        pro_end,
        ultra_end,
        directory_padding,
    ) = fields
    ends = (lite_end, full_end, pro_end, ultra_end)
    if (
        any(directory_padding)
        or reserved_offset
        or reserved_capacity
        or header_offset != _HEADER_OFFSET
        or not header_length
    ):
        raise ValueError("invalid directory")
    header_end = _checked_end(header_offset, header_length)
    if shared_offset != _align64(header_end):
        raise ValueError("invalid directory")
    shared_end = _checked_end(shared_offset, shared_length)
    if auth_offset != _align64(shared_end) or not auth_length:
        raise ValueError("invalid directory")
    auth_end = _checked_end(auth_offset, auth_length)
    if tier_start != _align64(auth_end):
        raise ValueError("invalid tier start")
    previous = tier_start
    for index, end in enumerate(ends):
        if index <= maximum:
            if end <= previous or end % 64:
                raise ValueError("invalid tier end")
            previous = end
        elif end:
            raise ValueError("unexpected tier end")
    if len(data) < _HEADER_OFFSET + _MUTABLE_FOOTER_SIZE:
        raise ValueError("invalid mutable trailer")
    magic, mutable_capacity, reserved = struct.unpack("<4sQI", data[-_MUTABLE_FOOTER_SIZE:])
    if (
        magic != b"DMUE"
        or reserved
        or mutable_capacity < _MUTABLE_HEADER_SIZE + 2
        or mutable_capacity > _REFERENCE_MAX_MUTABLE_CAPACITY
        or mutable_capacity > len(data) - _MUTABLE_FOOTER_SIZE
    ):
        raise ValueError("invalid mutable trailer")
    mutable_offset = len(data) - _MUTABLE_FOOTER_SIZE - mutable_capacity
    if mutable_offset not in ends[: maximum + 1]:
        raise ValueError("invalid retained end")
    retained = ends.index(mutable_offset)
    _read_mutable_frame(data[mutable_offset:-_MUTABLE_FOOTER_SIZE])
    padding_ranges = [
        (header_end, shared_offset),
        (shared_end, auth_offset),
        (auth_end, tier_start),
    ]
    if any(any(data[start:end]) for start, end in padding_ranges):
        raise ValueError("nonzero padding")
    return (
        _Layout(
            maximum,
            (header_offset, header_length),
            (shared_offset, shared_length),
            (auth_offset, auth_length),
            (mutable_offset, mutable_capacity),
            tier_start,
            ends,
        ),
        retained,
    )


def _strip(data: bytes, tier: int) -> bytes:
    layout, retained = _validate_layout(data)
    if not 0 <= tier <= retained:
        raise ValueError("invalid strip tier")
    mutable_offset, _ = layout.mutable
    return data[: layout.tier_ends[tier]] + data[mutable_offset:]


def _field(label: str, value: bytes) -> bytes:
    name = label.encode("ascii")
    return struct.pack("<I", len(name)) + name + struct.pack("<Q", len(value)) + value


def _retained_prefix(data: bytes, domain: bytes) -> bytes:
    layout, retained = _validate_layout(data)
    h_offset, h_length = layout.header
    s_offset, s_length = layout.shared
    shared = (
        _field("domain", b"DNAM-PLAINTEXT-SHARED-V1")
        + _field("preamble", data[:_PREAMBLE_SIZE])
        + _field("directory", data[_PREAMBLE_SIZE:_HEADER_OFFSET])
        + _field("shared_model_binary", data[s_offset : s_offset + s_length])
    )
    state = hashlib.sha256()
    state.update(_field("domain", domain))
    state.update(
        _field(
            "immutable_header_sha256", hashlib.sha256(data[h_offset : h_offset + h_length]).digest()
        )
    )
    state.update(_field("shared_model_data", shared))
    start = layout.tier_start
    for label, end in zip(
        ("tier_lite", "tier_full", "tier_pro", "tier_ultra"),
        layout.tier_ends[: retained + 1],
        strict=False,
    ):
        state.update(_field(label, data[start:end]))
        start = end
    return state.digest()


def _crc32c(data: bytes) -> int:
    crc = 0xFFFF_FFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F6_3B78 if crc & 1 else 0)
    return crc ^ 0xFFFF_FFFF


def _reject_constant(token: str) -> Never:
    raise ValueError(f"invalid JSON constant: {token}")


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON name")
        result[key] = value
    return result


def _check_scalar_strings(value: object) -> None:
    if isinstance(value, str):
        if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
            raise ValueError("unpaired surrogate")
    elif isinstance(value, list):
        for item in value:
            _check_scalar_strings(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_scalar_strings(key)
            _check_scalar_strings(item)


def _read_mutable_frame(region: bytes) -> dict[str, object]:
    if len(region) < _MUTABLE_HEADER_SIZE + 2:
        raise ValueError("short mutable frame")
    magic, checksum, length = struct.unpack("<4sIQ", region[:_MUTABLE_HEADER_SIZE])
    if magic != b"DMU1" or not 2 <= length <= len(region) - _MUTABLE_HEADER_SIZE:
        raise ValueError("invalid mutable frame")
    payload_end = _MUTABLE_HEADER_SIZE + length
    payload = region[_MUTABLE_HEADER_SIZE:payload_end]
    if checksum != _crc32c(magic + struct.pack("<Q", length) + payload):
        raise ValueError("mutable checksum mismatch")
    if any(region[payload_end:]):
        raise ValueError("nonzero mutable fill")
    text = payload.decode("utf-8")
    if text.startswith("\ufeff"):
        raise ValueError("UTF-8 BOM")
    value: object = json.loads(
        text,
        object_pairs_hook=_unique_pairs,
        parse_int=Decimal,
        parse_float=Decimal,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("mutable JSON must be an object")
    _check_scalar_strings(value)
    return value


def _frame_for_test(payload: bytes, capacity: int) -> bytes:
    if (
        capacity < _MUTABLE_HEADER_SIZE + 2
        or capacity > _REFERENCE_MAX_MUTABLE_CAPACITY
        or len(payload) > capacity - _MUTABLE_HEADER_SIZE
    ):
        raise ValueError("unsupported mutable capacity")
    length = struct.pack("<Q", len(payload))
    checksum = _crc32c(b"DMU1" + length + payload)
    return (
        b"DMU1"
        + struct.pack("<I", checksum)
        + length
        + payload
        + bytes(capacity - _MUTABLE_HEADER_SIZE - len(payload))
    )


def test_mutable_frame_has_fixed_checksum_and_minimum_bytes() -> None:
    assert _crc32c(b"123456789") == 0xE306_9283
    assert _frame_for_test(b"{}", 18) == (b"DMU1\x8f\xb4\xcc\xa1\x02\x00\x00\x00\x00\x00\x00\x00{}")
    assert _read_mutable_frame(_frame_for_test(b"{}", 18)) == {}
    with pytest.raises(ValueError, match="unsupported mutable capacity"):
        _frame_for_test(b"{}", 17)
    with pytest.raises(ValueError, match="unsupported mutable capacity"):
        _frame_for_test(b'{"x":0}', 18)


def test_mutable_frame_preserves_numbers_and_decodes_valid_unicode() -> None:
    payload = b'{"value":9007199254740993,"large":1e400,"note":"\\ud834\\udd1e"}'
    assert _read_mutable_frame(_frame_for_test(payload, 128)) == {
        "value": Decimal("9007199254740993"),
        "large": Decimal("1e400"),
        "note": "\U0001d11e",
    }


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        (b'{"x":1,"\\u0078":2}', "duplicate JSON name"),
        (b'{"nested":[{"a":1,"a":2}]}', "duplicate JSON name"),
        (b'{"value":NaN}', "invalid JSON constant"),
        (b'{"value":"\\ud800"}', "unpaired surrogate"),
        (b'{"value":"\xff"}', "invalid start byte"),
        (b"\xef\xbb\xbf{}", "UTF-8 BOM"),
        (b"[]", "mutable JSON must be an object"),
        (b"{}{}", "Extra data"),
        (b"{", "invalid mutable frame"),
    ],
)
def test_mutable_frame_rejects_ambiguous_or_invalid_json(payload: bytes, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        _read_mutable_frame(_frame_for_test(payload, 128))


def test_mutable_frame_rejects_bad_lengths_checksum_and_fill() -> None:
    original = _frame_for_test(b'{"x":1}', 64)
    for changed, error in (
        (original[:17], "short mutable frame"),
        (b"WRNG" + original[4:], "invalid mutable frame"),
        (original[:8] + struct.pack("<Q", 0) + original[16:], "invalid mutable frame"),
        (original[:8] + struct.pack("<Q", _U64_MAX) + original[16:], "invalid mutable frame"),
        (original[:16] + b"[" + original[17:], "mutable checksum mismatch"),
        (original[:-1] + b"\x01", "nonzero mutable fill"),
    ):
        with pytest.raises(ValueError, match=error):
            _read_mutable_frame(changed)


def test_open_physical_layout_has_fixed_preamble_and_directory_bytes() -> None:
    data = _example_file(b"{}", b"ctrl", b"signature", _frame_for_test(b"{}", 65), (b"L" * 64,) * 4)
    assert data[:16] == b"DNAM\x02\x00\x00\x00\x00\x00\x03\x00\x80\xbb\x00\x00"
    assert struct.unpack("<13Q24s", data[64:192]) == (
        192,
        2,
        256,
        4,
        320,
        9,
        0,
        0,
        384,
        448,
        512,
        576,
        640,
        bytes(24),
    )
    layout, retained = _validate_layout(data)
    assert retained == 3
    assert layout.mutable == (640, 65)


def test_stripping_and_mutable_edits_preserve_retained_prefix() -> None:
    data = _example_file(
        b"{}", b"ctrl", b"signature", _frame_for_test(b'{"tag":"A"}', 65), (b"L" * 64,) * 4
    )
    layout, _ = _validate_layout(data)
    for tier, end in enumerate(layout.tier_ends):
        stripped = _strip(data, tier)
        assert _validate_layout(stripped)[1] == tier
        original_digest = _retained_prefix(stripped, b"DNAM-MASTER-PREFIX-V1")
        edited = bytearray(_append_mutable(stripped[:end], _frame_for_test(b'{"tag":"B"}', 65)))
        assert _read_mutable_frame(bytes(edited[end:-_MUTABLE_FOOTER_SIZE])) == {"tag": "B"}
        assert _retained_prefix(bytes(edited), b"DNAM-MASTER-PREFIX-V1") == original_digest
        edited[256] ^= 1
        assert _retained_prefix(bytes(edited), b"DNAM-MASTER-PREFIX-V1") != original_digest


@pytest.mark.parametrize(
    ("auth_kind", "domain"),
    [(0, b"DNAM-MASTER-PREFIX-V1"), (1, b"DNAM-BUYER-PREFIX-V1")],
)
def test_growing_the_mutable_suffix_preserves_signed_prefixes(
    auth_kind: int, domain: bytes
) -> None:
    preset = b'{"presets":[{"id":"p","name":"Preset","controls":{"gain":0.5}}]}'
    with pytest.raises(ValueError, match="unsupported mutable capacity"):
        _frame_for_test(preset, 65)
    tiers = (b"A" * 64, b"B" * 64, b"C" * 64, b"D" * 64)
    original = _example_file(
        b"{}", b"ctrl", b"signature", _frame_for_test(b"{}", 65), tiers, auth_kind=auth_kind
    )
    grown = _example_file(
        b"{}", b"ctrl", b"signature", _frame_for_test(preset, 128), tiers, auth_kind=auth_kind
    )
    old_layout, _ = _validate_layout(original)
    new_layout, _ = _validate_layout(grown)
    assert old_layout.tier_start == new_layout.tier_start
    assert old_layout.tier_ends == new_layout.tier_ends
    assert (
        original[old_layout.tier_start : old_layout.tier_ends[3]]
        == grown[new_layout.tier_start : new_layout.tier_ends[3]]
    )
    assert _read_mutable_frame(grown[new_layout.mutable[0] : new_layout.mutable[0] + 128])[
        "presets"
    ]
    for tier in range(4):
        assert _retained_prefix(_strip(original, tier), domain) == _retained_prefix(
            _strip(grown, tier), domain
        )


def test_empty_mutable_suffix_follows_the_retained_tier() -> None:
    data = _example_file(b"{}", b"", b"signature", b"", (b"L" * 64,))
    layout, retained = _validate_layout(data)
    assert retained == 0
    assert layout.mutable == (384, 18)
    assert layout.tier_start == 320
    assert layout.tier_ends == (384, 0, 0, 0)


def test_lower_tier_buyer_can_keep_header_with_different_physical_placement() -> None:
    header, shared = b"{}", b"ctrl"
    master_tiers = (b"A" * 64, b"B" * 64, b"C" * 64, b"D" * 64)
    buyer_tiers = (b"X" * 64, b"Y" * 64)
    master = _example_file(header, shared, b"s" * 64, b"", master_tiers)
    buyer = _example_file(header, shared, b"b" * 200, b"", buyer_tiers, auth_kind=1)
    master_layout, master_retained = _validate_layout(master)
    buyer_layout, buyer_retained = _validate_layout(buyer)
    assert (master_retained, buyer_retained) == (3, 1)
    assert master_layout.tier_start != buyer_layout.tier_start
    assert master[192:194] == buyer[192:194] == header
    assert tuple(end - master_layout.tier_start for end in master_layout.tier_ends[:2]) == (64, 128)
    assert tuple(end - buyer_layout.tier_start for end in buyer_layout.tier_ends[:2]) == (64, 128)
    assert buyer_layout.tier_ends[2:] == (0, 0)

    buyer_lite = _strip(buyer, 0)
    digest = _retained_prefix(buyer_lite, b"DNAM-BUYER-PREFIX-V1")
    relocated = _example_file(
        header, shared, b"b" * 200, _frame_for_test(b"{}", 18), buyer_tiers, auth_kind=1
    )
    relocated_layout, _ = _validate_layout(relocated)
    assert relocated_layout.tier_start == buyer_layout.tier_start
    assert _retained_prefix(_strip(relocated, 0), b"DNAM-BUYER-PREFIX-V1") == digest


def test_layout_rejects_bad_boundaries_overflow_padding_and_suffixes() -> None:
    data = _example_file(b"{}", b"ctrl", b"signature", _frame_for_test(b"{}", 65), (b"L" * 64,) * 4)
    for offset, value, error in (
        (64, 256, "invalid directory"),
        (72, 0, "invalid directory"),
        (80, 320, "invalid directory"),
        (112, 320, "invalid directory"),
        (120, 17, "invalid directory"),
        (120, _U64_MAX, "invalid directory"),
        (136, 512, "invalid tier end"),
    ):
        changed = bytearray(data)
        struct.pack_into("<Q", changed, offset, value)
        with pytest.raises(ValueError, match=error):
            _validate_layout(bytes(changed))
    for index, error in (
        (11, "invalid preamble"),
        (16, "invalid preamble"),
        (168, "invalid directory"),
        (194, "nonzero padding"),
        (260, "nonzero padding"),
        (329, "nonzero padding"),
    ):
        changed = bytearray(data)
        changed[index] = 1
        with pytest.raises(ValueError, match=error):
            _validate_layout(bytes(changed))
    for invalid in (data[:-1], data + b"\x00", data[:192]):
        with pytest.raises(ValueError, match="invalid mutable trailer"):
            _validate_layout(invalid)

    layout, _ = _validate_layout(data)
    for invalid in (
        _append_mutable(data[: layout.tier_ends[3] - 1], _frame_for_test(b"{}", 18)),
        _append_mutable(data[: layout.tier_ends[3] + 1], _frame_for_test(b"{}", 18)),
    ):
        with pytest.raises(ValueError, match="invalid retained end"):
            _validate_layout(invalid)
