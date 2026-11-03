"""Sealed public-layout references; signature and encryption are placeholders."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import pytest

from test_encrypted_context_spec_references import (
    _parse_prewrap_plan,
    _parse_recipient_vector,
    _parse_record_directory,
    _parse_slot_vector,
    _prewrap_plan,
    _record_context_digest,
    _slot_vector,
    _structural_metadata_digest,
    _wrap_plan_digest,
)
from test_encrypted_mutable_spec_references import _example_keys, _example_suffix, _parse_suffix
from test_signed_prefix_spec_references import (
    _ciphertext_prefixes,
    _outer_message,
    _outer_public_metadata_digest,
    _parse_prefix_list,
    _prefix_list,
)

_U64_MAX = (1 << 64) - 1
_SIGNATURE = b"\x5a" * 64


def _align64(value: int) -> int:
    if not 0 <= value <= _U64_MAX - 63:
        raise ValueError("invalid aligned extent")
    return (value + 63) & ~63


def _records(start: int, lengths: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    records = []
    for length in lengths:
        records.append((start, length))
        start = _align64(start + length)
    return tuple(records)


def _fields(plan: bytes) -> dict[str, bytes]:
    position = 0
    fields = {}
    while position < len(plan):
        label_length = struct.unpack_from("<I", plan, position)[0]
        position += 4
        label = plan[position : position + label_length].decode("ascii")
        position += label_length
        value_length = struct.unpack_from("<Q", plan, position)[0]
        position += 8
        fields[label] = plan[position : position + value_length]
        position += value_length
    return fields


@dataclass(frozen=True)
class _Fixture:
    data: bytes
    signed_message: bytes
    boundaries: tuple[int, ...]
    manifest_offset: int
    shared_offset: int


def _example(maximum: int) -> _Fixture:
    keys = _example_keys()
    suffix, recipients, _ = _example_suffix(2, keys, b"{}")
    slots = _slot_vector(((b"a" * 32, b"b" * 48), (b"c" * 32, b"d" * 48)))
    lengths = (48, 17, 100, 64, 33)[: maximum + 2]
    placeholder = _prewrap_plan(
        b"i" * 16, 1, 2, maximum, bytes(32), recipients, _records(64, lengths)
    )
    plan_offset = 192
    slots_offset = _align64(plan_offset + len(placeholder))
    manifest_offset = _align64(slots_offset + len(slots))
    manifest_length = 105 + 33 * (maximum + 1)
    shared_offset = _align64(manifest_offset + manifest_length)
    records = _records(shared_offset, lengths)
    boundaries = tuple(offset + _align64(length) for offset, length in records[1:])
    padded_ends = boundaries + (0,) * (4 - len(boundaries))
    preamble = struct.pack("<4sHHBB54s", b"DNSL", 2, 0, 2, maximum, bytes(54))
    directory = struct.pack(
        "<12Q32s",
        plan_offset,
        len(placeholder),
        slots_offset,
        len(slots),
        manifest_offset,
        manifest_length,
        shared_offset,
        lengths[0],
        *padded_ends,
        bytes(32),
    )
    structure = _structural_metadata_digest(1, preamble, directory)
    plan = _prewrap_plan(b"i" * 16, 1, 2, maximum, structure, recipients, records)
    assert len(plan) == len(placeholder)
    plan_region = plan + bytes(slots_offset - plan_offset - len(plan))
    slot_region = slots + bytes(manifest_offset - slots_offset - len(slots))
    manifest_padding = bytes(shared_offset - manifest_offset - manifest_length)
    auxiliary = plan_region + slot_region + manifest_padding
    context = _record_context_digest(_wrap_plan_digest(plan), slots)
    ciphertexts = tuple(
        bytes([index + 1]) * length + bytes(_align64(length) - length)
        for index, length in enumerate(lengths)
    )
    prefixes = _ciphertext_prefixes(context, ciphertexts[0], ciphertexts[1:])
    signer = b"k" * 32
    metadata = _outer_public_metadata_digest(preamble, directory, auxiliary)
    message = _outer_message(signer, metadata, context, prefixes)
    manifest = b"DSA1" + bytes([maximum]) + bytes(3) + signer + _prefix_list(prefixes) + _SIGNATURE
    assert len(manifest) == manifest_length
    data = (
        preamble
        + directory
        + plan_region
        + slot_region
        + manifest
        + manifest_padding
        + b"".join(ciphertexts)
        + suffix
    )
    return _Fixture(data, message, boundaries, manifest_offset, shared_offset)


def _parse_example(data: bytes, trusted_message: bytes) -> int:
    if len(data) < 192:
        raise ValueError("short sealed wrapper")
    preamble, directory = data[:64], data[64:192]
    magic, major, minor, profile, maximum, reserved = struct.unpack("<4sHHBB54s", preamble)
    if (magic, major, minor, profile) != (b"DNSL", 2, 0, 2) or maximum > 3 or any(reserved):
        raise ValueError("invalid sealed preamble")
    values = struct.unpack("<12Q32s", directory)
    (
        plan_offset,
        plan_length,
        slots_offset,
        slots_length,
        manifest_offset,
        manifest_length,
        shared_offset,
        shared_length,
    ) = values[:8]
    ends = values[8:12]
    if (
        any(values[12])
        or plan_offset != 192
        or not 0 < plan_length <= 4_096
        or slots_offset != _align64(plan_offset + plan_length)
        or manifest_offset != _align64(slots_offset + slots_length)
        or manifest_length != 105 + 33 * (maximum + 1)
        or shared_offset != _align64(manifest_offset + manifest_length)
        or shared_offset > len(data)
        or any(ends[maximum + 1 :])
    ):
        raise ValueError("invalid sealed directory")

    plan_end, slots_end = plan_offset + plan_length, slots_offset + slots_length
    manifest_end = manifest_offset + manifest_length
    if any(
        data[plan_end:slots_offset]
        + data[slots_end:manifest_offset]
        + data[manifest_end:shared_offset]
    ):
        raise ValueError("nonzero public padding")
    plan = data[plan_offset:plan_end]
    _parse_prewrap_plan(plan, 16)
    fields = _fields(plan)
    if (
        fields["layer"] != b"\x01"
        or fields["protection_profile"] != b"\x02"
        or fields["issued_max_tier"] != bytes([maximum])
        or fields["structural_metadata_sha256"]
        != _structural_metadata_digest(1, preamble, directory)
    ):
        raise ValueError("mismatched sealed plan")
    recipients = _parse_recipient_vector(fields["recipient_key_ids"], 16)
    slots = data[slots_offset:slots_end]
    _parse_slot_vector(slots, len(recipients), 16)
    records = _parse_record_directory(fields["record_descriptors"], maximum)
    if records[0] != (shared_offset, shared_length):
        raise ValueError("mismatched shared record")
    previous_end = shared_offset + _align64(shared_length)
    for tier, (offset, length) in enumerate(records[1:]):
        end = offset + _align64(length)
        if offset != previous_end or end > _U64_MAX or ends[tier] != end:
            raise ValueError("mismatched tier extent")
        previous_end = end

    manifest = data[manifest_offset:manifest_end]
    if manifest[:8] != b"DSA1" + bytes([maximum]) + bytes(3) or manifest[-64:] != _SIGNATURE:
        raise ValueError("invalid manifest")
    signer = manifest[8:40]
    prefixes = _parse_prefix_list(manifest[40:-64], maximum)
    auxiliary = data[plan_offset:manifest_offset] + data[manifest_end:shared_offset]
    context = _record_context_digest(_wrap_plan_digest(plan), slots)
    metadata = _outer_public_metadata_digest(preamble, directory, auxiliary)
    if _outer_message(signer, metadata, context, prefixes) != trusted_message:
        raise ValueError("outer message differs")

    retained_end, _, _ = _parse_suffix(data, tuple(ends[: maximum + 1]), 2, recipients)
    retained = ends.index(retained_end)
    chunks = []
    for offset, length in records[: retained + 2]:
        end = offset + _align64(length)
        if end > retained_end or any(data[offset + length : end]):
            raise ValueError("invalid retained ciphertext")
        chunks.append(data[offset:end])
    if _ciphertext_prefixes(context, chunks[0], tuple(chunks[1:]))[-1] != prefixes[retained]:
        raise ValueError("invalid retained commitment")
    return retained


@pytest.mark.parametrize("maximum", range(4))
def test_sealed_public_layout_retains_signed_message_after_each_strip(maximum: int) -> None:
    fixture = _example(maximum)
    suffix = fixture.data[fixture.boundaries[-1] :]
    for tier, boundary in enumerate(fixture.boundaries):
        stripped = fixture.data[:boundary] + suffix
        assert _parse_example(stripped, fixture.signed_message) == tier
        assert (
            stripped[fixture.manifest_offset : fixture.shared_offset]
            == fixture.data[fixture.manifest_offset : fixture.shared_offset]
        )


@pytest.mark.parametrize(
    ("region", "error"),
    [
        ("plan", "pre-wrap plan"),
        ("slots", "outer message differs"),
        ("manifest", "outer message differs"),
        ("padding", "nonzero public padding"),
        ("shared", "invalid retained commitment"),
        ("tier", "invalid retained commitment"),
    ],
)
def test_sealed_public_layout_rejects_mutation_of_committed_regions(
    region: str, error: str
) -> None:
    fixture = _example(3)
    directory = struct.unpack("<12Q32s", fixture.data[64:192])
    positions = {
        "plan": directory[0] + 4,
        "slots": directory[2] + 4,
        "manifest": fixture.manifest_offset + 42,
        "padding": fixture.manifest_offset + directory[5],
        "shared": fixture.shared_offset,
        "tier": fixture.boundaries[0] - 64,
    }
    data = bytearray(fixture.data)
    data[positions[region]] ^= 1
    with pytest.raises(ValueError, match=error):
        _parse_example(bytes(data), fixture.signed_message)


def test_sealed_public_layout_rejects_a_suffix_inside_a_record() -> None:
    fixture = _example(3)
    suffix = fixture.data[fixture.boundaries[-1] :]
    broken = fixture.data[: fixture.boundaries[0] - 1] + suffix
    with pytest.raises(ValueError, match="retained boundary"):
        _parse_example(broken, fixture.signed_message)


def test_stripped_file_still_authenticates_the_omitted_tier_commitments() -> None:
    fixture = _example(3)
    suffix = fixture.data[fixture.boundaries[-1] :]
    stripped = bytearray(fixture.data[: fixture.boundaries[0]] + suffix)
    stripped[fixture.manifest_offset + 42 + 3 * 33] ^= 1
    with pytest.raises(ValueError, match="outer message differs"):
        _parse_example(bytes(stripped), fixture.signed_message)


def test_sealed_directory_rejects_overflow_and_nonzero_reserved_bytes() -> None:
    fixture = _example(3)
    overflow = bytearray(fixture.data)
    struct.pack_into("<Q", overflow, 64 + 3 * 8, _U64_MAX)
    with pytest.raises(ValueError, match="invalid aligned extent"):
        _parse_example(bytes(overflow), fixture.signed_message)
    reserved = bytearray(fixture.data)
    reserved[64 + 96] = 1
    with pytest.raises(ValueError, match="invalid sealed directory"):
        _parse_example(bytes(reserved), fixture.signed_message)
