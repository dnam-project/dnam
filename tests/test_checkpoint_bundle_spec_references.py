"""Toy tier-bundle examples, not a checkpoint wire format or signature verifier."""

from __future__ import annotations

import hashlib
import struct

import pytest

from test_open_layout_spec_references import _append_mutable, _frame_for_test, _read_mutable_frame

_BLOCK = 64
_BUNDLE = 3 * _BLOCK


def _example_checkpoint(payload: bytes) -> tuple[bytes, tuple[int, ...], tuple[bytes, ...]]:
    ends = tuple(_BLOCK + _BUNDLE * (tier + 1) for tier in range(4))
    shared = b"DNCK" + bytes(4) + struct.pack("<4Q", *ends) + b"T" * 24
    bundles = tuple(
        bytes([0x41 + tier]) * _BLOCK
        + bytes([0x61 + tier]) * _BLOCK
        + bytes([0x31 + tier]) * _BLOCK
        for tier in range(4)
    )
    protected = shared + b"".join(bundles)
    prefixes = tuple(hashlib.sha256(protected[:end]).digest() for end in ends)
    frame = _frame_for_test(payload, 16 + len(payload))
    return _append_mutable(protected, frame), ends, prefixes


def _read_example(data: bytes, trusted_prefixes: tuple[bytes, ...]) -> int:
    if len(data) < _BLOCK + 16 or data[:4] != b"DNCK":
        raise ValueError("invalid checkpoint")
    ends = struct.unpack("<4Q", data[8:40])
    if ends != tuple(_BLOCK + _BUNDLE * (tier + 1) for tier in range(4)):
        raise ValueError("invalid checkpoint directory")
    magic, frame_length, reserved = struct.unpack("<4sQI", data[-16:])
    if magic != b"DMUE" or reserved or frame_length > len(data) - 16:
        raise ValueError("invalid mutable footer")
    boundary = len(data) - 16 - frame_length
    if boundary not in ends:
        raise ValueError("invalid bundle boundary")
    tier = ends.index(boundary)
    if hashlib.sha256(data[:boundary]).digest() != trusted_prefixes[tier]:
        raise ValueError("checkpoint prefix mismatch")
    _read_mutable_frame(data[boundary:-16])
    return tier


@pytest.mark.parametrize("tier", range(4))
def test_stripping_a_bundle_prefix_retains_each_lower_tier_record(tier: int) -> None:
    original, ends, prefixes = _example_checkpoint(b'{"preset":"clean"}')
    suffix = original[ends[-1] :]
    stripped = original[: ends[tier]] + suffix
    assert _read_example(stripped, prefixes) == tier
    assert stripped[:_BLOCK] == original[:_BLOCK]
    assert stripped[ends[tier] :] == suffix
    for retained in range(tier + 1):
        start = _BLOCK + retained * _BUNDLE
        assert stripped[start : start + _BLOCK] == bytes([0x41 + retained]) * _BLOCK
        assert stripped[start + _BLOCK : start + 2 * _BLOCK] == bytes([0x61 + retained]) * _BLOCK
        assert stripped[start + 2 * _BLOCK : start + _BUNDLE] == bytes([0x31 + retained]) * _BLOCK


def test_metadata_growth_and_higher_tier_tampering_leave_lite_prefix_unchanged() -> None:
    original, ends, prefixes = _example_checkpoint(b"{}")
    grown, grown_ends, grown_prefixes = _example_checkpoint(b'{"presets":["clean","drive"]}')
    assert grown_ends == ends
    assert grown_prefixes == prefixes
    assert original[: ends[-1]] == grown[: ends[-1]]
    assert _read_example(grown, prefixes) == 3

    changed = bytearray(original)
    changed[ends[0] + _BLOCK] ^= 1
    with pytest.raises(ValueError, match="checkpoint prefix mismatch"):
        _read_example(bytes(changed), prefixes)
    suffix = original[ends[-1] :]
    assert _read_example(bytes(changed[: ends[0]]) + suffix, prefixes) == 0


def test_each_bundle_component_and_shared_structure_are_authenticated() -> None:
    original, _, prefixes = _example_checkpoint(b"{}")
    for offset in (40, _BLOCK, 2 * _BLOCK, 3 * _BLOCK):
        changed = bytearray(original)
        changed[offset] ^= 1
        with pytest.raises(ValueError, match="checkpoint prefix mismatch"):
            _read_example(bytes(changed), prefixes)


def test_partial_bundle_and_mutable_corruption_are_rejected() -> None:
    original, ends, prefixes = _example_checkpoint(b"{}")
    suffix = original[ends[-1] :]
    with pytest.raises(ValueError, match="invalid bundle boundary"):
        _read_example(original[: ends[1] - _BLOCK] + suffix, prefixes)
    corrupt = bytearray(original)
    corrupt[-17] ^= 1
    with pytest.raises(ValueError, match="mutable checksum mismatch"):
        _read_example(bytes(corrupt), prefixes)
