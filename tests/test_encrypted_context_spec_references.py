"""Framing references for encrypted issuance, not an HPKE implementation."""

from __future__ import annotations

import hashlib
import struct
from itertools import pairwise

import pytest

_U64_MAX = 2**64 - 1
_MAX_RECORD_CIPHERTEXT = (2**32 - 1) * 64 + 16
_HPKE_SUITE = b"\x00\x20\x00\x01\x00\x03"
_EXAMPLE_STRUCTURE = b"s" * 32


def _field(label: str, value: bytes) -> bytes:
    encoded = label.encode("ascii")
    return struct.pack("<I", len(encoded)) + encoded + struct.pack("<Q", len(value)) + value


def _structural_metadata_digest(layer: int, preamble: bytes, directory: bytes) -> bytes:
    if layer not in (0, 1) or not preamble or not directory:
        raise ValueError("invalid encrypted structure")
    return hashlib.sha256(
        _field("domain", b"DNAM-ENCRYPTED-STRUCTURE-V1")
        + _field("layer", bytes([layer]))
        + _field("preamble", preamble)
        + _field("directory", directory)
    ).digest()


def _wrap_plan_digest(plan: bytes) -> bytes:
    return hashlib.sha256(_field("domain", b"DNAM-WRAP-PLAN-V1") + _field("plan", plan)).digest()


def _record_context_digest(plan_digest: bytes, slots: bytes) -> bytes:
    if len(plan_digest) != 32:
        raise ValueError("invalid plan digest")
    return hashlib.sha256(
        _field("domain", b"DNAM-RECORD-CONTEXT-V1")
        + _field("wrap_plan_sha256", plan_digest)
        + _field("slots", slots)
    ).digest()


def _hpke_binding(domain: bytes, plan_digest: bytes, index: int, key_digest: bytes) -> bytes:
    if (
        domain not in (b"DNAM-HPKE-INFO-V1", b"DNAM-HPKE-AAD-V1")
        or len(plan_digest) != 32
        or len(key_digest) != 32
        or not 0 <= index < 2**32
    ):
        raise ValueError("invalid HPKE binding")
    return (
        _field("domain", domain)
        + _field("wrap_plan_sha256", plan_digest)
        + _field("slot_index", struct.pack("<I", index))
        + _field("recipient_key_sha256", key_digest)
    )


def _recipient_vector(key_digests: tuple[bytes, ...]) -> bytes:
    if (
        not key_digests
        or len(key_digests) > 2**32 - 1
        or any(len(digest) != 32 for digest in key_digests)
        or any(left >= right for left, right in pairwise(key_digests))
    ):
        raise ValueError("invalid recipient vector")
    return struct.pack("<I", len(key_digests)) + b"".join(key_digests)


def _parse_recipient_vector(data: bytes, maximum: int) -> tuple[bytes, ...]:
    if len(data) < 4:
        raise ValueError("invalid recipient vector")
    count = struct.unpack("<I", data[:4])[0]
    if not 0 < count <= maximum or len(data) != 4 + 32 * count:
        raise ValueError("invalid recipient vector")
    identities = tuple(data[4 + 32 * i : 4 + 32 * (i + 1)] for i in range(count))
    if _recipient_vector(identities) != data:
        raise ValueError("invalid recipient vector")
    return identities


def _slot_vector(slots: tuple[tuple[bytes, bytes], ...]) -> bytes:
    if (
        not slots
        or len(slots) > 2**32 - 1
        or any(len(enc) != 32 or len(ciphertext) != 48 for enc, ciphertext in slots)
    ):
        raise ValueError("invalid slot vector")
    return struct.pack("<I", len(slots)) + b"".join(enc + ciphertext for enc, ciphertext in slots)


def _parse_slot_vector(
    data: bytes, expected_count: int, maximum: int
) -> tuple[tuple[bytes, bytes], ...]:
    if len(data) < 4:
        raise ValueError("invalid slot vector")
    count = struct.unpack("<I", data[:4])[0]
    if not 0 < count <= maximum or count != expected_count or len(data) != 4 + 80 * count:
        raise ValueError("invalid slot vector")
    return tuple(
        (data[4 + 80 * i : 36 + 80 * i], data[36 + 80 * i : 84 + 80 * i]) for i in range(count)
    )


def _record_directory(maximum_tier: int, records: tuple[tuple[int, int], ...]) -> bytes:
    if not 0 <= maximum_tier <= 3 or len(records) != maximum_tier + 2:
        raise ValueError("invalid record directory")
    previous_end = 0
    for index, (offset, length) in enumerate(records):
        if not 0 <= offset <= _U64_MAX or offset % 64 or not 16 <= length <= _MAX_RECORD_CIPHERTEXT:
            raise ValueError("invalid record directory")
        padded_length = (length + 63) // 64 * 64
        end = offset + padded_length
        if end > _U64_MAX or (index == 1 and offset < previous_end):
            raise ValueError("invalid record directory")
        if index > 1 and offset != previous_end:
            raise ValueError("invalid record directory")
        previous_end = end
    return struct.pack("<I", len(records)) + b"".join(
        struct.pack("<QQ", offset, length) for offset, length in records
    )


def _parse_record_directory(data: bytes, maximum_tier: int) -> tuple[tuple[int, int], ...]:
    if not 0 <= maximum_tier <= 3 or len(data) < 4:
        raise ValueError("invalid record directory")
    count = struct.unpack("<I", data[:4])[0]
    if count != maximum_tier + 2 or len(data) != 4 + 16 * count:
        raise ValueError("invalid record directory")
    records = tuple(
        struct.unpack("<QQ", data[4 + 16 * index : 20 + 16 * index]) for index in range(count)
    )
    if _record_directory(maximum_tier, records) != data:
        raise ValueError("invalid record directory")
    return records


def _validate_retained_record_extents(
    directory: bytes, maximum_tier: int, retained_tier: int, layer_data: bytes
) -> None:
    records = _parse_record_directory(directory, maximum_tier)
    if not 0 <= retained_tier <= maximum_tier:
        raise ValueError("invalid retained tier")
    retained = records[: retained_tier + 2]
    for offset, length in retained:
        end = offset + (length + 63) // 64 * 64
        if end > len(layer_data) or any(layer_data[offset + length : end]):
            raise ValueError("invalid record extent")
    offset, length = retained[-1]
    if len(layer_data) != offset + (length + 63) // 64 * 64:
        raise ValueError("invalid retained end")


def _prewrap_plan(
    issuance_id: bytes,
    layer: int,
    profile: int,
    maximum_tier: int,
    structural_metadata_digest: bytes,
    recipients: tuple[bytes, ...],
    records: tuple[tuple[int, int], ...],
    mutable_region: tuple[int, int] = (0, 0),
) -> bytes:
    if (
        len(issuance_id) != 16
        or layer not in (0, 1)
        or profile not in (1, 2)
        or (profile == 1 and layer != 0)
        or not 0 <= maximum_tier <= 3
        or len(structural_metadata_digest) != 32
    ):
        raise ValueError("invalid pre-wrap plan")
    directory = _record_directory(maximum_tier, records)
    mutable_offset, mutable_capacity = mutable_region
    if mutable_offset != 0 or mutable_capacity != 0:
        raise ValueError("invalid mutable region")
    return (
        _field("domain", b"DNAM-PRE-WRAP-PLAN-V1")
        + _field("issuance_id", issuance_id)
        + _field("layer", bytes([layer]))
        + _field("protection_profile", bytes([profile]))
        + _field("issued_max_tier", bytes([maximum_tier]))
        + _field("structural_metadata_sha256", structural_metadata_digest)
        + _field("mutable_offset", struct.pack("<Q", mutable_offset))
        + _field("mutable_capacity", struct.pack("<Q", mutable_capacity))
        + _field("hpke_suite", _HPKE_SUITE)
        + _field("recipient_key_ids", _recipient_vector(recipients))
        + _field("record_descriptors", directory)
    )


def _parse_prewrap_plan(data: bytes, maximum_recipients: int) -> bytes:
    if maximum_recipients < 1:
        raise ValueError("invalid recipient limit")
    view = memoryview(data)
    position = 0

    def read_field(label: bytes, maximum_length: int) -> bytes:
        nonlocal position
        if len(view) - position < 4:
            raise ValueError("invalid pre-wrap plan")
        label_len = struct.unpack("<I", view[position : position + 4])[0]
        position += 4
        if label_len != len(label) or len(view) - position < label_len + 8:
            raise ValueError("invalid pre-wrap plan")
        if view[position : position + label_len] != label:
            raise ValueError("invalid pre-wrap plan")
        position += label_len
        value_len = struct.unpack("<Q", view[position : position + 8])[0]
        position += 8
        if value_len > maximum_length or value_len > len(view) - position:
            raise ValueError("invalid pre-wrap plan")
        value = bytes(view[position : position + value_len])
        position += value_len
        return value

    domain = read_field(b"domain", 21)
    issuance_id = read_field(b"issuance_id", 16)
    layer = read_field(b"layer", 1)
    profile = read_field(b"protection_profile", 1)
    maximum_tier = read_field(b"issued_max_tier", 1)
    structural_digest = read_field(b"structural_metadata_sha256", 32)
    mutable_offset = read_field(b"mutable_offset", 8)
    mutable_capacity = read_field(b"mutable_capacity", 8)
    suite = read_field(b"hpke_suite", len(_HPKE_SUITE))
    recipients = read_field(b"recipient_key_ids", 4 + 32 * maximum_recipients)
    records = read_field(b"record_descriptors", 4 + 16 * 5)
    if (
        position != len(view)
        or domain != b"DNAM-PRE-WRAP-PLAN-V1"
        or len(issuance_id) != 16
        or len(layer) != 1
        or len(profile) != 1
        or len(maximum_tier) != 1
        or len(structural_digest) != 32
        or len(mutable_offset) != 8
        or len(mutable_capacity) != 8
        or suite != _HPKE_SUITE
    ):
        raise ValueError("invalid pre-wrap plan")
    keys = _parse_recipient_vector(recipients, maximum_recipients)
    entries = _parse_record_directory(records, maximum_tier[0])
    mutable = (struct.unpack("<Q", mutable_offset)[0], struct.unpack("<Q", mutable_capacity)[0])
    encoded = _prewrap_plan(
        issuance_id,
        layer[0],
        profile[0],
        maximum_tier[0],
        structural_digest,
        keys,
        entries,
        mutable,
    )
    if encoded != data:
        raise ValueError("invalid pre-wrap plan")
    return encoded


def test_prewrap_hash_uses_versioned_length_framing() -> None:
    expected = hashlib.sha256(
        b"\x06\x00\x00\x00domain\x11\x00\x00\x00\x00\x00\x00\x00DNAM-WRAP-PLAN-V1"
        b"\x04\x00\x00\x00plan\x05\x00\x00\x00\x00\x00\x00\x00known"
    ).digest()
    assert _wrap_plan_digest(b"known") == expected
    assert _wrap_plan_digest(b"known") != _wrap_plan_digest(b"known\x00")


def test_structural_metadata_binds_exact_layer_preamble_and_directory_before_wrapping() -> None:
    expected = hashlib.sha256(
        b"\x06\x00\x00\x00domain\x1b\x00\x00\x00\x00\x00\x00\x00"
        b"DNAM-ENCRYPTED-STRUCTURE-V1"
        b"\x05\x00\x00\x00layer\x01\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\x08\x00\x00\x00preamble\x01\x00\x00\x00\x00\x00\x00\x00p"
        b"\x09\x00\x00\x00directory\x02\x00\x00\x00\x00\x00\x00\x00dd"
    ).digest()
    structure = _structural_metadata_digest(0, b"p", b"dd")
    assert structure == expected
    alternatives = (
        _structural_metadata_digest(1, b"p", b"dd"),
        _structural_metadata_digest(0, b"q", b"dd"),
        _structural_metadata_digest(0, b"p", b"de"),
        _structural_metadata_digest(0, b"pd", b"d"),
    )
    assert all(value != structure for value in alternatives)
    for layer, preamble, directory in ((2, b"p", b"d"), (0, b"", b"d"), (0, b"p", b"")):
        with pytest.raises(ValueError, match="invalid encrypted structure"):
            _structural_metadata_digest(layer, preamble, directory)

    records = ((64, 17), (128, 32))
    key_id = b"a" * 32
    plan = _prewrap_plan(b"i" * 16, 0, 1, 0, structure, (key_id,), records)
    plan_hash = _wrap_plan_digest(plan)
    changed_plan = _prewrap_plan(b"i" * 16, 0, 1, 0, alternatives[1], (key_id,), records)
    changed_hash = _wrap_plan_digest(changed_plan)
    assert plan_hash != changed_hash
    assert _hpke_binding(b"DNAM-HPKE-INFO-V1", plan_hash, 0, key_id) != _hpke_binding(
        b"DNAM-HPKE-INFO-V1", changed_hash, 0, key_id
    )
    assert _record_context_digest(plan_hash, b"slots") != _record_context_digest(
        changed_hash, b"slots"
    )


def test_key_wrap_binding_includes_plan_recipient_slot_and_purpose() -> None:
    plan = _wrap_plan_digest(b"inner issuance and recipient plan")
    other_plan = _wrap_plan_digest(b"outer issuance and recipient plan")
    recipient = hashlib.sha256(bytes(range(32))).digest()
    other_recipient = hashlib.sha256(bytes(reversed(range(32)))).digest()
    info = _hpke_binding(b"DNAM-HPKE-INFO-V1", plan, 0, recipient)
    assert info != _hpke_binding(b"DNAM-HPKE-AAD-V1", plan, 0, recipient)
    assert info != _hpke_binding(b"DNAM-HPKE-INFO-V1", other_plan, 0, recipient)
    assert info != _hpke_binding(b"DNAM-HPKE-INFO-V1", plan, 1, recipient)
    assert info != _hpke_binding(b"DNAM-HPKE-INFO-V1", plan, 0, other_recipient)
    for domain, digest, index, key in [
        (b"wrong", plan, 0, recipient),
        (b"DNAM-HPKE-INFO-V1", plan[:-1], 0, recipient),
        (b"DNAM-HPKE-INFO-V1", plan, 0, recipient[:-1]),
        (b"DNAM-HPKE-INFO-V1", plan, -1, recipient),
        (b"DNAM-HPKE-INFO-V1", plan, 2**32, recipient),
    ]:
        with pytest.raises(ValueError, match="invalid HPKE binding"):
            _hpke_binding(domain, digest, index, key)


def test_record_context_commits_to_slots_without_changing_wrap_plan() -> None:
    plan = _wrap_plan_digest(b"precomputed plan with recipient descriptors")
    recipient = hashlib.sha256(bytes(range(32))).digest()
    info = _hpke_binding(b"DNAM-HPKE-INFO-V1", plan, 0, recipient)
    slots = bytes(range(32)) + bytes(range(48))
    changed_slots = slots[:-1] + bytes([slots[-1] ^ 1])
    assert _record_context_digest(plan, slots) != _record_context_digest(plan, changed_slots)
    assert _hpke_binding(b"DNAM-HPKE-INFO-V1", plan, 0, recipient) == info
    assert _record_context_digest(plan, slots) != _record_context_digest(
        _wrap_plan_digest(b"another plan"), slots
    )
    with pytest.raises(ValueError, match="invalid plan digest"):
        _record_context_digest(plan[:-1], slots)


def test_hashing_finished_slots_into_wrap_plan_creates_a_dependency_cycle() -> None:
    base_plan = b"precomputed issuance"
    recipient = hashlib.sha256(bytes(range(32))).digest()
    before = _wrap_plan_digest(base_plan)
    info = _hpke_binding(b"DNAM-HPKE-INFO-V1", before, 0, recipient)
    mock_slot = hashlib.sha256(info + b"mock slot output").digest()
    after = _wrap_plan_digest(base_plan + mock_slot)
    assert after != before
    assert _hpke_binding(b"DNAM-HPKE-INFO-V1", after, 0, recipient) != info


def test_recipient_and_slot_vectors_have_canonical_lengths_and_order() -> None:
    keys = (bytes(range(32)), bytes(reversed(range(32))))
    identities = tuple(sorted(hashlib.sha256(key).digest() for key in keys))
    recipients = _recipient_vector(identities)
    slots = ((b"a" * 32, b"A" * 48), (b"b" * 32, b"B" * 48))
    encoded_slots = _slot_vector(slots)
    assert recipients == b"\x02\x00\x00\x00" + identities[0] + identities[1]
    assert encoded_slots == b"\x02\x00\x00\x00" + b"a" * 32 + b"A" * 48 + b"b" * 32 + b"B" * 48
    assert len(recipients) == 4 + 32 * len(slots)
    assert len(encoded_slots) == 4 + 80 * len(slots)
    assert _parse_recipient_vector(recipients, 2) == identities
    assert _parse_slot_vector(encoded_slots, len(identities), 2) == slots
    plan = _wrap_plan_digest(_field("recipient_key_ids", recipients))
    context = _record_context_digest(plan, encoded_slots)
    changed = _slot_vector((slots[1], slots[0]))
    assert _record_context_digest(plan, changed) != context
    replaced = tuple(sorted((identities[0], hashlib.sha256(b"replacement key").digest())))
    replacement_plan = _wrap_plan_digest(_field("recipient_key_ids", _recipient_vector(replaced)))
    assert replacement_plan != plan
    assert _hpke_binding(b"DNAM-HPKE-INFO-V1", replacement_plan, 0, replaced[0]) != _hpke_binding(
        b"DNAM-HPKE-INFO-V1", plan, 0, identities[0]
    )
    assert _hpke_binding(b"DNAM-HPKE-INFO-V1", plan, 0, identities[0]) != _hpke_binding(
        b"DNAM-HPKE-INFO-V1", plan, 1, identities[0]
    )


def test_recipient_vectors_reject_noncanonical_or_oversized_inputs() -> None:
    first, second = b"a" * 32, b"b" * 32
    for identities in [(), (first, first), (second, first), (first[:-1], second)]:
        with pytest.raises(ValueError, match="invalid recipient vector"):
            _recipient_vector(identities)
    valid = _recipient_vector((first, second))
    for data, maximum in [
        (b"", 2),
        (b"\x00\x00\x00\x00", 2),
        (valid[:-1], 2),
        (valid + b"\x00", 2),
        (valid, 1),
        (b"\xff\xff\xff\xff", 10),
        (b"\xff\xff\xff\xff", 2**32 - 1),
        (b"\x02\x00\x00\x00" + second + first, 2),
        (b"\x02\x00\x00\x00" + first + first, 2),
    ]:
        with pytest.raises(ValueError, match="invalid recipient vector"):
            _parse_recipient_vector(data, maximum)


def test_slot_vectors_reject_bad_lengths_counts_and_resource_limits() -> None:
    for slots in [(), ((bytes(31), bytes(48)),), ((bytes(32), bytes(47)),)]:
        with pytest.raises(ValueError, match="invalid slot vector"):
            _slot_vector(slots)
    valid = _slot_vector(((b"a" * 32, b"A" * 48),))
    for data, expected, maximum in [
        (b"", 1, 1),
        (b"\x00\x00\x00\x00", 0, 1),
        (valid[:-1], 1, 1),
        (valid + b"\x00", 1, 1),
        (valid, 2, 2),
        (valid, 1, 0),
        (b"\xff\xff\xff\xff", 2**32 - 1, 10),
        (b"\xff\xff\xff\xff", 2**32 - 1, 2**32 - 1),
    ]:
        with pytest.raises(ValueError, match="invalid slot vector"):
            _parse_slot_vector(data, expected, maximum)


def test_x25519_byte_hashes_alone_do_not_prove_distinct_coordinates() -> None:
    key = b"\x09" + bytes(31)
    alias = key[:-1] + b"\x80"
    mask = 2**255 - 1
    prime = 2**255 - 19
    key_coordinate = (int.from_bytes(key, "little") & mask) % prime
    alias_coordinate = (int.from_bytes(alias, "little") & mask) % prime
    assert key != alias
    assert key_coordinate == alias_coordinate
    key_id = hashlib.sha256(key).digest()
    alias_id = hashlib.sha256(alias).digest()
    assert key_id != alias_id
    plan = _wrap_plan_digest(b"example plan")
    assert _hpke_binding(b"DNAM-HPKE-INFO-V1", plan, 0, key_id) != _hpke_binding(
        b"DNAM-HPKE-INFO-V1", plan, 0, alias_id
    )


def test_prewrap_plan_has_exact_ordered_fields_and_layer_relative_records() -> None:
    issuance_id = bytes(range(16))
    key_id = bytes(range(32))
    records = ((64, 17), (128, 32))
    encoded = _prewrap_plan(issuance_id, 0, 1, 0, _EXAMPLE_STRUCTURE, (key_id,), records)
    expected = (
        b"\x06\x00\x00\x00domain\x15\x00\x00\x00\x00\x00\x00\x00DNAM-PRE-WRAP-PLAN-V1"
        b"\x0b\x00\x00\x00issuance_id\x10\x00\x00\x00\x00\x00\x00\x00"
        + issuance_id
        + b"\x05\x00\x00\x00layer\x01\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\x12\x00\x00\x00protection_profile\x01\x00\x00\x00\x00\x00\x00\x00\x01"
        b"\x0f\x00\x00\x00issued_max_tier\x01\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\x1a\x00\x00\x00structural_metadata_sha256\x20\x00\x00\x00\x00\x00\x00\x00"
        + _EXAMPLE_STRUCTURE
        + b"\x0e\x00\x00\x00mutable_offset\x08\x00\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\x10\x00\x00\x00mutable_capacity\x08\x00\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\x0a\x00\x00\x00hpke_suite\x06\x00\x00\x00\x00\x00\x00\x00\x00\x20\x00\x01\x00\x03"
        b"\x11\x00\x00\x00recipient_key_ids\x24\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00"
        + key_id
        + b"\x12\x00\x00\x00record_descriptors\x24\x00\x00\x00\x00\x00\x00\x00"
        b"\x02\x00\x00\x00"
        b"\x40\x00\x00\x00\x00\x00\x00\x00\x11\x00\x00\x00\x00\x00\x00\x00"
        b"\x80\x00\x00\x00\x00\x00\x00\x00\x20\x00\x00\x00\x00\x00\x00\x00"
    )
    assert encoded == expected
    assert _parse_prewrap_plan(encoded, maximum_recipients=1) == encoded
    assert _parse_record_directory(_record_directory(0, records), 0) == records


def test_plan_binds_issuance_profile_layer_tier_keys_and_boundaries() -> None:
    first_key, second_key = b"a" * 32, b"b" * 32
    base = _prewrap_plan(
        b"i" * 16, 0, 2, 1, _EXAMPLE_STRUCTURE, (first_key,), ((64, 17), (128, 32), (192, 16))
    )
    alternatives = [
        _prewrap_plan(
            b"j" * 16, 0, 2, 1, _EXAMPLE_STRUCTURE, (first_key,), ((64, 17), (128, 32), (192, 16))
        ),
        _prewrap_plan(
            b"i" * 16, 1, 2, 1, _EXAMPLE_STRUCTURE, (first_key,), ((64, 17), (128, 32), (192, 16))
        ),
        _prewrap_plan(
            b"i" * 16, 0, 1, 1, _EXAMPLE_STRUCTURE, (first_key,), ((64, 17), (128, 32), (192, 16))
        ),
        _prewrap_plan(b"i" * 16, 0, 2, 0, _EXAMPLE_STRUCTURE, (first_key,), ((64, 17), (128, 32))),
        _prewrap_plan(
            b"i" * 16, 0, 2, 1, _EXAMPLE_STRUCTURE, (second_key,), ((64, 17), (128, 32), (192, 16))
        ),
        _prewrap_plan(
            b"i" * 16, 0, 2, 1, _EXAMPLE_STRUCTURE, (first_key,), ((64, 17), (192, 32), (256, 16))
        ),
        _prewrap_plan(
            b"i" * 16, 0, 2, 1, _EXAMPLE_STRUCTURE, (first_key,), ((64, 17), (128, 33), (192, 16))
        ),
        _prewrap_plan(
            b"i" * 16, 0, 2, 1, b"x" * 32, (first_key,), ((64, 17), (128, 32), (192, 16))
        ),
    ]
    assert all(_wrap_plan_digest(base) != _wrap_plan_digest(value) for value in alternatives)
    assert all(_parse_prewrap_plan(value, 1) == value for value in [base, *alternatives])


def test_plan_reserves_zero_mutable_fields_for_growable_suffix() -> None:
    records = ((128, 17), (256, 32))
    empty = _prewrap_plan(b"i" * 16, 0, 1, 0, _EXAMPLE_STRUCTURE, (b"a" * 32,), records)
    assert _parse_prewrap_plan(empty, 1) == empty
    for mutable in (
        (0, 1),
        (64, 0),
        (64, 32),
        (65, 1),
        (128, 1),
        (192, 64),
        (192, 65),
        (256, 64),
        (_U64_MAX - 63, 128),
    ):
        with pytest.raises(ValueError, match="invalid mutable region"):
            _prewrap_plan(b"i" * 16, 0, 1, 0, _EXAMPLE_STRUCTURE, (b"a" * 32,), records, mutable)


def test_record_directory_rejects_bad_counts_lengths_offsets_and_gaps() -> None:
    largest = ((0, 16), (64, _MAX_RECORD_CIPHERTEXT))
    assert _parse_record_directory(_record_directory(0, largest), 0) == largest
    for maximum_tier, records in [
        (-1, ()),
        (4, ()),
        (0, ((64, 16),)),
        (0, ((1, 16), (128, 16))),
        (0, ((64, 15), (128, 16))),
        (0, ((64, _MAX_RECORD_CIPHERTEXT + 1), (128, 16))),
        (0, ((64, 65), (128, 16))),
        (1, ((64, 16), (128, 16), (256, 16))),
        (0, ((_U64_MAX - 63, 16), (_U64_MAX - 63, 16))),
    ]:
        with pytest.raises(ValueError, match="invalid record directory"):
            _record_directory(maximum_tier, records)

    valid = _record_directory(0, ((64, 16), (128, 16)))
    for data in [
        b"",
        b"\xff\xff\xff\xff",
        valid[:-1],
        valid + b"\x00",
        b"\x01\x00\x00\x00" + valid[4:],
    ]:
        with pytest.raises(ValueError, match="invalid record directory"):
            _parse_record_directory(data, 0)


def test_retained_prefixes_allow_omitted_suffixes_and_check_zero_padding() -> None:
    records = ((64, 20), (128, 18), (192, 17), (256, 65), (384, 80))
    directory = _record_directory(3, records)
    plan = _prewrap_plan(b"i" * 16, 0, 2, 3, _EXAMPLE_STRUCTURE, (b"a" * 32,), records)
    data = bytearray(512)
    for offset, length in records:
        data[offset : offset + length] = b"x" * length
    for retained_tier in range(4):
        offset, length = records[retained_tier + 1]
        end = offset + (length + 63) // 64 * 64
        assert _parse_prewrap_plan(plan, 1) == plan
        _validate_retained_record_extents(directory, 3, retained_tier, bytes(data[:end]))
        with pytest.raises(ValueError, match="invalid record extent"):
            _validate_retained_record_extents(directory, 3, retained_tier, bytes(data[: end - 1]))
        with pytest.raises(ValueError, match="invalid retained end"):
            _validate_retained_record_extents(
                directory, 3, retained_tier, bytes(data[:end]) + b"\x00"
            )
    data[148] = 1
    with pytest.raises(ValueError, match="invalid record extent"):
        _validate_retained_record_extents(directory, 3, 0, bytes(data[:192]))


def test_plan_parser_rejects_noncanonical_order_lengths_and_limits() -> None:
    key_id = b"a" * 32
    records = ((64, 16), (128, 16))
    encoded = _prewrap_plan(b"i" * 16, 0, 1, 0, _EXAMPLE_STRUCTURE, (key_id,), records)
    layer_field = _field("layer", b"\x00")
    profile_field = _field("protection_profile", b"\x01")
    offset_field = _field("mutable_offset", struct.pack("<Q", 0))
    capacity_field = _field("mutable_capacity", struct.pack("<Q", 0))
    recipient_field = _field("recipient_key_ids", _recipient_vector((key_id,)))
    enormous_recipient_length = (
        struct.pack("<I", len("recipient_key_ids"))
        + b"recipient_key_ids"
        + struct.pack("<Q", _U64_MAX)
    )
    for invalid in [
        encoded[:-1],
        encoded + b"\x00",
        encoded + layer_field,
        encoded.replace(layer_field + profile_field, profile_field + layer_field),
        encoded.replace(_HPKE_SUITE, b"\x00\x20\x00\x01\x00\x02"),
        encoded.replace(_field("issuance_id", b"i" * 16), _field("issuance_id", b"i" * 15)),
        encoded.replace(_field("layer", b"\x00"), _field("layer", b"\x01")),
        encoded.replace(
            _field("structural_metadata_sha256", _EXAMPLE_STRUCTURE),
            _field("structural_metadata_sha256", _EXAMPLE_STRUCTURE[:-1]),
        ),
        encoded.replace(offset_field, _field("mutable_offset", b"\x00")),
        encoded.replace(offset_field, _field("mutable_offset", struct.pack("<Q", 64))).replace(
            capacity_field, _field("mutable_capacity", struct.pack("<Q", 1))
        ),
        encoded.replace(recipient_field, _field("recipient_key_ids", b"\xff\xff\xff\xff")),
        encoded.replace(recipient_field, enormous_recipient_length),
    ]:
        with pytest.raises(ValueError, match="invalid"):
            _parse_prewrap_plan(invalid, 1)
    with pytest.raises(ValueError, match="invalid recipient limit"):
        _parse_prewrap_plan(encoded, 0)
