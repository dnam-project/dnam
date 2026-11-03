"""Protected-layout references; the signatures and record cipher are placeholders."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass

import pytest

from test_encrypted_context_spec_references import (
    _field,
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
from test_sealed_public_framing_spec_references import (
    _SIGNATURE,
    _align64,
    _fields,
    _parse_example,
    _records,
)
from test_signed_prefix_spec_references import (
    _buyer_message,
    _ciphertext_prefixes,
    _creator_message,
    _creator_signed_identity,
    _license_policy_digest,
    _outer_message,
    _outer_public_metadata_digest,
    _parse_prefix_list,
    _plaintext_shared,
    _prefix_list,
    _prefixes,
)

_ISSUANCE = b"i" * 16
_CREATOR = b"c" * 32
_MARKETPLACE = b"m" * 32
_POLICY = b"example-policy"
_HEADER = (
    b'{"name":"Example rig","slimmable_tiers":{'
    b'"full":{"arena_end_byte_offset":192},'
    b'"lite":{"arena_end_byte_offset":64},'
    b'"pro":{"arena_end_byte_offset":256},'
    b'"ultra":{"arena_end_byte_offset":448}}}'
)
_SHARED = b"shared model data"
_TIERS = (b"L" * 64, b"F" * 128, b"P" * 64, b"U" * 192)
_SLOTS = _slot_vector(((b"a" * 32, b"b" * 48), (b"c" * 32, b"d" * 48)))


def _signature(key: bytes, message: bytes) -> bytes:
    # This deterministic stand-in is deliberately not a digital signature.
    return hashlib.sha512(key + message).digest()


def _encrypt(plaintext: bytes, context: bytes, index: int) -> bytes:
    # Reversible fixture encoding and digest, not encryption or an AEAD.
    body = bytes(value ^ 0xA5 for value in plaintext)
    tag = hashlib.sha256(context + struct.pack("<I", index) + body).digest()[:16]
    return body + tag


def _decrypt(record: bytes, context: bytes, index: int) -> bytes:
    if len(record) < 16:
        raise ValueError("short fixture record")
    body, tag = record[:-16], record[-16:]
    expected = hashlib.sha256(context + struct.pack("<I", index) + body).digest()[:16]
    if tag != expected:
        raise ValueError("invalid fixture tag")
    return bytes(value ^ 0xA5 for value in body)


def _logical_digest(prefix: bytes) -> bytes:
    return hashlib.sha256(
        _field("domain", b"DNAM-LOGICAL-METADATA-V1") + _field("logical_prefix", prefix)
    ).digest()


def _padded(data: bytes) -> bytes:
    return data + bytes(_align64(len(data)) - len(data))


@dataclass(frozen=True)
class _Logical:
    preamble: bytes
    directory: bytes
    prefix: bytes
    tier_start: int
    ends: tuple[int, ...]


def _make_logical(profile: int, maximum: int) -> _Logical:
    preamble = struct.pack("<4sHHBBHI48s", b"DNPL", 2, 0, profile, maximum, 0, 48_000, bytes(48))
    shared_offset = _align64(192 + len(_HEADER))
    tier_start = _align64(shared_offset + len(_SHARED))
    ends = []
    position = tier_start
    for payload in _TIERS[: maximum + 1]:
        position += len(payload)
        ends.append(position)
    directory = struct.pack(
        "<9Q56s",
        192,
        len(_HEADER),
        shared_offset,
        len(_SHARED),
        tier_start,
        *(tuple(ends) + (0,) * (4 - len(ends))),
        bytes(56),
    )
    prefix = preamble + directory + _HEADER
    prefix += bytes(shared_offset - len(prefix))
    return _Logical(preamble, directory, prefix, tier_start, tuple(ends))


def _parse_logical(prefix: bytes, profile: int, maximum: int) -> _Logical:
    if len(prefix) < 192:
        raise ValueError("short logical metadata")
    preamble, directory = prefix[:64], prefix[64:192]
    magic, major, minor, actual_profile, actual_max, reserved, rate, tail = struct.unpack(
        "<4sHHBBHI48s", preamble
    )
    if (magic, major, minor, actual_profile, actual_max, reserved, rate) != (
        b"DNPL",
        2,
        0,
        profile,
        maximum,
        0,
        48_000,
    ) or any(tail):
        raise ValueError("invalid logical preamble")
    values = struct.unpack("<9Q56s", directory)
    header_offset, header_length, shared_offset, shared_length, tier_start = values[:5]
    ends = values[5:9]
    if (
        any(values[9])
        or header_offset != 192
        or not 0 < header_length <= len(prefix) - 192
        or shared_offset != _align64(header_offset + header_length)
        or len(prefix) != shared_offset
        or tier_start != _align64(shared_offset + shared_length)
        or any(prefix[header_offset + header_length : shared_offset])
        or any(ends[maximum + 1 :])
    ):
        raise ValueError("invalid logical directory")
    previous = tier_start
    for end in ends[: maximum + 1]:
        if end % 64 or end <= previous:
            raise ValueError("invalid logical tier boundary")
        previous = end
    return _Logical(preamble, directory, prefix, tier_start, tuple(ends[: maximum + 1]))


def _check_signed_tier_ends(header: bytes, logical: _Logical) -> None:
    document: object = json.loads(header)
    if not isinstance(document, dict):
        raise ValueError("invalid fixture header")
    tiers: object = document.get("slimmable_tiers")
    if not isinstance(tiers, dict):
        raise ValueError("invalid fixture header")
    for name, end in zip(("lite", "full", "pro", "ultra"), logical.ends, strict=False):
        tier: object = tiers.get(name)
        if not isinstance(tier, dict):
            raise ValueError("invalid fixture header")
        signed_end: object = tier.get("arena_end_byte_offset")
        if isinstance(signed_end, bool) or not isinstance(signed_end, int):
            raise ValueError("invalid fixture header")
        if signed_end != end - logical.tier_start:
            raise ValueError("signed tier end differs")


@dataclass(frozen=True)
class _Envelope:
    creator_maximum: int
    creator_prefixes: tuple[bytes, ...]
    creator_signature: bytes
    buyer_prefixes: tuple[bytes, ...]
    cipher_prefixes: tuple[bytes, ...]
    buyer_signature: bytes


def _auth_bytes(envelope: _Envelope, maximum: int) -> bytes:
    creator = _prefix_list(envelope.creator_prefixes)
    buyer = _prefix_list(envelope.buyer_prefixes)
    encrypted = _prefix_list(envelope.cipher_prefixes)
    return (
        b"DAI1"
        + bytes(4)
        + _CREATOR
        + bytes([envelope.creator_maximum])
        + creator
        + envelope.creator_signature
        + _MARKETPLACE
        + struct.pack("<I", len(_POLICY))
        + _POLICY
        + bytes([maximum])
        + buyer
        + encrypted
        + envelope.buyer_signature
    )


def _parse_auth(data: bytes, maximum: int) -> _Envelope:
    position = 0

    def take(length: int) -> bytes:
        nonlocal position
        if length < 0 or position + length > len(data):
            raise ValueError("invalid protected authentication envelope")
        result = data[position : position + length]
        position += length
        return result

    if take(8) != b"DAI1" + bytes(4) or take(32) != _CREATOR:
        raise ValueError("invalid protected authentication envelope")
    creator_maximum = take(1)[0]
    if creator_maximum > 3 or creator_maximum < maximum:
        raise ValueError("invalid protected authentication envelope")
    creator_prefixes = _parse_prefix_list(take(1 + 33 * (creator_maximum + 1)), creator_maximum)
    creator_signature = take(64)
    if take(32) != _MARKETPLACE:
        raise ValueError("invalid protected authentication envelope")
    policy_length = struct.unpack("<I", take(4))[0]
    if not 0 < policy_length <= 1_024 or take(policy_length) != _POLICY:
        raise ValueError("invalid protected authentication envelope")
    if take(1) != bytes([maximum]):
        raise ValueError("invalid protected authentication envelope")
    buyer_prefixes = _parse_prefix_list(take(1 + 33 * (maximum + 1)), maximum)
    cipher_prefixes = _parse_prefix_list(take(1 + 33 * (maximum + 1)), maximum)
    buyer_signature = take(64)
    if position != len(data):
        raise ValueError("invalid protected authentication envelope")
    return _Envelope(
        creator_maximum,
        creator_prefixes,
        creator_signature,
        buyer_prefixes,
        cipher_prefixes,
        buyer_signature,
    )


@dataclass(frozen=True)
class _Inner:
    data: bytes
    profile: int
    maximum: int
    ends: tuple[int, ...]
    records: tuple[tuple[int, int], ...]
    auth_offset: int


def _make_inner(profile: int, maximum: int, creator_maximum: int = 3) -> _Inner:
    logical = _make_logical(profile, maximum)
    header_hash = hashlib.sha256(_HEADER).digest()
    plaintext_shared = _plaintext_shared(logical.preamble, logical.directory, _SHARED)
    buyer_prefixes = _prefixes(
        b"DNAM-BUYER-PREFIX-V1", header_hash, plaintext_shared, _TIERS[: maximum + 1]
    )
    creator_prefixes = tuple(bytes([index + 10]) * 32 for index in range(creator_maximum + 1))
    creator_message = _creator_message(_CREATOR, header_hash, creator_prefixes)
    creator_signature = _signature(_CREATOR, creator_message)
    suffix, recipients, _ = _example_suffix(profile, _example_keys(), b"{}")
    lengths = (len(_SHARED) + 16, *(len(tier) + 16 for tier in _TIERS[: maximum + 1]))
    placeholder = _prewrap_plan(
        _ISSUANCE, 0, profile, maximum, bytes(32), recipients, _records(64, lengths)
    )
    plan_offset = 192 + len(logical.prefix)
    slots_offset = _align64(plan_offset + len(placeholder))
    auth_offset = _align64(slots_offset + len(_SLOTS))
    auth_length = 209 + 33 * (creator_maximum + 1) + 66 * (maximum + 1) + len(_POLICY)
    shared_offset = _align64(auth_offset + auth_length)
    records = _records(shared_offset, lengths)
    ends = tuple(offset + _align64(length) for offset, length in records[1:])
    preamble = struct.pack(
        "<4sHHBBBBI48s", b"DNAM", 2, 0, profile, 1, maximum, 0, 48_000, bytes(48)
    )
    directory = struct.pack(
        "<12Q32s",
        plan_offset,
        len(placeholder),
        slots_offset,
        len(_SLOTS),
        auth_offset,
        auth_length,
        shared_offset,
        lengths[0],
        *(ends + (0,) * (4 - len(ends))),
        _logical_digest(logical.prefix),
    )
    structure = _structural_metadata_digest(0, preamble, directory)
    plan = _prewrap_plan(_ISSUANCE, 0, profile, maximum, structure, recipients, records)
    context = _record_context_digest(_wrap_plan_digest(plan), _SLOTS)
    ciphertexts = tuple(
        _padded(_encrypt(plaintext, context, index))
        for index, plaintext in enumerate((_SHARED, *_TIERS[: maximum + 1]))
    )
    cipher_prefixes = _ciphertext_prefixes(context, ciphertexts[0], ciphertexts[1:])
    buyer_message = _buyer_message(
        _MARKETPLACE,
        _creator_signed_identity(creator_message, creator_signature),
        _license_policy_digest(_POLICY),
        profile,
        buyer_prefixes,
        context,
        cipher_prefixes,
    )
    envelope = _Envelope(
        creator_maximum,
        creator_prefixes,
        creator_signature,
        buyer_prefixes,
        cipher_prefixes,
        _signature(_MARKETPLACE, buyer_message),
    )
    auth = _auth_bytes(envelope, maximum)
    assert len(auth) == auth_length
    data = (
        preamble
        + directory
        + logical.prefix
        + _padded(plan)
        + _padded(_SLOTS)
        + _padded(auth)
        + b"".join(ciphertexts)
    )
    assert len(data) == ends[-1]
    if profile == 1:
        data += suffix
    return _Inner(data, profile, maximum, ends, records, auth_offset)


@dataclass(frozen=True)
class _ParsedInner:
    retained: int
    header: bytes
    context: bytes
    issuance_id: bytes
    maximum: int


def _parse_inner(
    data: bytes, outer: tuple[bytes, int, int] | None = None, *, decrypt: bool = True
) -> _ParsedInner:
    if len(data) < 192:
        raise ValueError("short protected inner layer")
    preamble, directory = data[:64], data[64:192]
    magic, major, minor, profile, kind, maximum, reserved, rate, tail = struct.unpack(
        "<4sHHBBBBI48s", preamble
    )
    if (
        (magic, major, minor, kind, reserved, rate) != (b"DNAM", 2, 0, 1, 0, 48_000)
        or profile not in (1, 2)
        or maximum > 3
        or any(tail)
        or (profile == 2 and outer is None)
        or (profile == 1 and outer is not None)
    ):
        raise ValueError("invalid protected preamble")
    values = struct.unpack("<12Q32s", directory)
    plan_offset, plan_length, slots_offset, slots_length, auth_offset, auth_length = values[:6]
    shared_offset, shared_length = values[6:8]
    ends = values[8:12]
    if (
        plan_offset < 384
        or plan_offset % 64
        or plan_offset > len(data)
        or not 0 < plan_length <= 4_096
        or slots_offset != _align64(plan_offset + plan_length)
        or auth_offset != _align64(slots_offset + slots_length)
        or shared_offset != _align64(auth_offset + auth_length)
        or shared_offset > len(data)
        or any(ends[maximum + 1 :])
    ):
        raise ValueError("invalid protected directory")
    logical_prefix = data[192:plan_offset]
    if _logical_digest(logical_prefix) != values[12]:
        raise ValueError("invalid logical metadata digest")
    logical = _parse_logical(logical_prefix, profile, maximum)
    header_length = struct.unpack_from("<Q", logical.directory, 8)[0]
    header = logical.prefix[192 : 192 + header_length]
    plan_end, slots_end, auth_end = (
        plan_offset + plan_length,
        slots_offset + slots_length,
        auth_offset + auth_length,
    )
    if any(
        data[plan_end:slots_offset] + data[slots_end:auth_offset] + data[auth_end:shared_offset]
    ):
        raise ValueError("invalid protected padding")
    plan = data[plan_offset:plan_end]
    _parse_prewrap_plan(plan, 16)
    fields = _fields(plan)
    if (
        fields["layer"] != b"\x00"
        or fields["protection_profile"] != bytes([profile])
        or fields["issued_max_tier"] != bytes([maximum])
        or fields["structural_metadata_sha256"]
        != _structural_metadata_digest(0, preamble, directory)
    ):
        raise ValueError("invalid inner plan")
    recipients = _parse_recipient_vector(fields["recipient_key_ids"], 16)
    slots = data[slots_offset:slots_end]
    _parse_slot_vector(slots, len(recipients), 16)
    records = _parse_record_directory(fields["record_descriptors"], maximum)
    logical_shared_length = struct.unpack_from("<Q", logical.directory, 24)[0]
    if records[0] != (shared_offset, shared_length) or shared_length != logical_shared_length + 16:
        raise ValueError("invalid shared record")
    previous_physical = shared_offset + _align64(shared_length)
    previous_logical = logical.tier_start
    for tier, (offset, length) in enumerate(records[1:]):
        if (
            offset != previous_physical
            or length != logical.ends[tier] - previous_logical + 16
            or ends[tier] != offset + _align64(length)
        ):
            raise ValueError("invalid protected tier record")
        previous_physical = ends[tier]
        previous_logical = logical.ends[tier]
    if profile == 1:
        retained_end, _, _ = _parse_suffix(data, tuple(ends[: maximum + 1]), 1, recipients)
    else:
        retained_end = len(data)
    if retained_end not in ends[: maximum + 1]:
        raise ValueError("invalid inner retained boundary")
    retained = ends.index(retained_end)
    if outer is not None and (fields["issuance_id"], maximum, retained) != outer:
        raise ValueError("invalid sealed layer link")
    envelope = _parse_auth(data[auth_offset:auth_end], maximum)
    expected_auth_length = (
        209 + 33 * (envelope.creator_maximum + 1) + 66 * (maximum + 1) + len(_POLICY)
    )
    if auth_length != expected_auth_length:
        raise ValueError("invalid protected authentication length")
    creator_message = _creator_message(
        _CREATOR, hashlib.sha256(header).digest(), envelope.creator_prefixes
    )
    if envelope.creator_signature != _signature(_CREATOR, creator_message):
        raise ValueError("invalid creator signature")
    context = _record_context_digest(_wrap_plan_digest(plan), slots)
    buyer_message = _buyer_message(
        _MARKETPLACE,
        _creator_signed_identity(creator_message, envelope.creator_signature),
        _license_policy_digest(_POLICY),
        profile,
        envelope.buyer_prefixes,
        context,
        envelope.cipher_prefixes,
    )
    if envelope.buyer_signature != _signature(_MARKETPLACE, buyer_message):
        raise ValueError("invalid buyer signature")
    _check_signed_tier_ends(header, logical)
    chunks = []
    for offset, length in records[: retained + 2]:
        end = offset + _align64(length)
        if end > retained_end or any(data[offset + length : end]):
            raise ValueError("invalid inner record padding")
        chunks.append(data[offset:end])
    if (
        _ciphertext_prefixes(context, chunks[0], tuple(chunks[1:]))[-1]
        != envelope.cipher_prefixes[retained]
    ):
        raise ValueError("invalid inner ciphertext prefix")
    if decrypt:
        plaintexts = tuple(
            _decrypt(data[offset : offset + length], context, index)
            for index, (offset, length) in enumerate(records[: retained + 2])
        )
        shared = _plaintext_shared(logical.preamble, logical.directory, plaintexts[0])
        prefixes = _prefixes(
            b"DNAM-BUYER-PREFIX-V1", hashlib.sha256(header).digest(), shared, plaintexts[1:]
        )
        if prefixes[-1] != envelope.buyer_prefixes[retained]:
            raise ValueError("invalid buyer plaintext prefix")
    return _ParsedInner(retained, header, context, fields["issuance_id"], maximum)


@dataclass(frozen=True)
class _Outer:
    data: bytes
    message: bytes
    ends: tuple[int, ...]
    maximum: int


def _make_outer(inner: _Inner) -> _Outer:
    assert inner.profile == 2
    keys = _example_keys()
    suffix, recipients, _ = _example_suffix(2, keys, b"{}")
    first_tier = inner.records[1][0]
    plaintexts = (
        inner.data[:first_tier],
        *(inner.data[offset : offset + _align64(length)] for offset, length in inner.records[1:]),
    )
    lengths = tuple(len(plaintext) + 16 for plaintext in plaintexts)
    placeholder = _prewrap_plan(
        _ISSUANCE, 1, 2, inner.maximum, bytes(32), recipients, _records(64, lengths)
    )
    plan_offset = 192
    slots_offset = _align64(plan_offset + len(placeholder))
    manifest_offset = _align64(slots_offset + len(_SLOTS))
    manifest_length = 105 + 33 * (inner.maximum + 1)
    shared_offset = _align64(manifest_offset + manifest_length)
    records = _records(shared_offset, lengths)
    ends = tuple(offset + _align64(length) for offset, length in records[1:])
    preamble = struct.pack("<4sHHBB54s", b"DNSL", 2, 0, 2, inner.maximum, bytes(54))
    directory = struct.pack(
        "<12Q32s",
        plan_offset,
        len(placeholder),
        slots_offset,
        len(_SLOTS),
        manifest_offset,
        manifest_length,
        shared_offset,
        lengths[0],
        *(ends + (0,) * (4 - len(ends))),
        bytes(32),
    )
    structure = _structural_metadata_digest(1, preamble, directory)
    plan = _prewrap_plan(_ISSUANCE, 1, 2, inner.maximum, structure, recipients, records)
    context = _record_context_digest(_wrap_plan_digest(plan), _SLOTS)
    ciphertexts = tuple(
        _padded(_encrypt(plaintext, context, index)) for index, plaintext in enumerate(plaintexts)
    )
    prefixes = _ciphertext_prefixes(context, ciphertexts[0], ciphertexts[1:])
    manifest = b"DSA1" + bytes([inner.maximum]) + bytes(3) + b"k" * 32
    manifest += _prefix_list(prefixes) + _SIGNATURE
    auxiliary = (
        _padded(plan) + _padded(_SLOTS) + bytes(shared_offset - manifest_offset - len(manifest))
    )
    message = _outer_message(
        b"k" * 32, _outer_public_metadata_digest(preamble, directory, auxiliary), context, prefixes
    )
    data = (
        preamble
        + directory
        + _padded(plan)
        + _padded(_SLOTS)
        + _padded(manifest)
        + b"".join(ciphertexts)
        + suffix
    )
    return _Outer(data, message, ends, inner.maximum)


def _parse_outer(fixture: _Outer, data: bytes) -> _ParsedInner:
    retained = _parse_example(data, fixture.message)
    values = struct.unpack("<12Q32s", data[64:192])
    plan = data[values[0] : values[0] + values[1]]
    fields = _fields(plan)
    slots = data[values[2] : values[2] + values[3]]
    context = _record_context_digest(_wrap_plan_digest(plan), slots)
    records = _parse_record_directory(fields["record_descriptors"], fixture.maximum)
    plaintexts = tuple(
        _decrypt(data[offset : offset + length], context, index)
        for index, (offset, length) in enumerate(records[: retained + 2])
    )
    inner = b"".join(plaintexts)
    return _parse_inner(inner, (fields["issuance_id"], fixture.maximum, retained))


@pytest.mark.parametrize("maximum", range(4))
def test_allowlisted_header_and_all_stripping_boundaries(maximum: int) -> None:
    fixture = _make_inner(1, maximum)
    suffix = fixture.data[fixture.ends[-1] :]
    assert _HEADER in fixture.data
    for tier, end in enumerate(fixture.ends):
        stripped = fixture.data[:end] + suffix
        indexed = _parse_inner(stripped, decrypt=False)
        assert indexed.header == _HEADER
        assert indexed.retained == tier
        assert _parse_inner(stripped).retained == tier
        assert (
            stripped[fixture.auth_offset : fixture.records[0][0]]
            == fixture.data[fixture.auth_offset : fixture.records[0][0]]
        )


@pytest.mark.parametrize("maximum", range(4))
def test_sealed_outer_reconstructs_each_matching_inner_prefix(maximum: int) -> None:
    inner = _make_inner(2, maximum)
    outer = _make_outer(inner)
    suffix = outer.data[outer.ends[-1] :]
    assert _HEADER not in outer.data
    with pytest.raises(ValueError, match="invalid protected preamble"):
        _parse_inner(inner.data)
    for tier, end in enumerate(outer.ends):
        parsed = _parse_outer(outer, outer.data[:end] + suffix)
        assert parsed.retained == tier
        assert parsed.header == _HEADER
        assert parsed.maximum == maximum
        assert parsed.issuance_id == _ISSUANCE


def test_logical_metadata_is_bound_to_the_inner_plan_and_receipt() -> None:
    fixture = _make_inner(1, 3)
    original_directory = fixture.data[64:192]
    changed = bytearray(fixture.data)
    header_index = 192 + 192
    changed[header_index + 9] ^= 1
    with pytest.raises(ValueError, match="invalid logical metadata digest"):
        _parse_inner(bytes(changed))

    plan_offset = struct.unpack_from("<Q", original_directory)[0]
    changed[64 + 96 : 64 + 128] = _logical_digest(bytes(changed[192:plan_offset]))
    with pytest.raises(ValueError, match="invalid inner plan"):
        _parse_inner(bytes(changed))


def test_protected_envelope_rejects_omitted_tier_commitment_changes() -> None:
    fixture = _make_inner(1, 3)
    suffix = fixture.data[fixture.ends[-1] :]
    stripped = bytearray(fixture.data[: fixture.ends[0]] + suffix)
    auth = fixture.auth_offset
    creator_list_length = 1 + 33 * 4
    buyer_list = auth + 8 + 32 + 1 + creator_list_length + 64 + 32 + 4 + len(_POLICY) + 1
    stripped[buyer_list + 2 + 3 * 33] ^= 1
    with pytest.raises(ValueError, match="invalid buyer signature"):
        _parse_inner(bytes(stripped))


def test_logical_layout_rejects_bad_boundaries_and_padding() -> None:
    logical = _make_logical(1, 3)
    for offset, replacement in ((64 + 72, b"\x01"), (len(logical.prefix) - 1, b"\x01")):
        changed = bytearray(logical.prefix)
        changed[offset : offset + 1] = replacement
        with pytest.raises(ValueError, match="invalid logical directory"):
            _parse_logical(bytes(changed), 1, 3)
    changed = bytearray(logical.prefix)
    struct.pack_into("<Q", changed, 64 + 40, logical.tier_start)
    with pytest.raises(ValueError, match="invalid logical tier boundary"):
        _parse_logical(bytes(changed), 1, 3)
    changed = bytearray(logical.prefix)
    struct.pack_into("<Q", changed, 64 + 24, (1 << 64) - 1)
    with pytest.raises(ValueError, match="invalid aligned extent"):
        _parse_logical(bytes(changed), 1, 3)


def test_logical_tier_lengths_must_match_the_creator_signed_header() -> None:
    logical = _make_logical(1, 3)
    _check_signed_tier_ends(_HEADER, logical)
    changed = _HEADER.replace(b'"arena_end_byte_offset":192', b'"arena_end_byte_offset":128')
    with pytest.raises(ValueError, match="signed tier end differs"):
        _check_signed_tier_ends(changed, logical)


def test_inner_and_outer_must_agree_on_issuance_and_retained_tier() -> None:
    inner = _make_inner(2, 3)
    lite = inner.data[: inner.ends[0]]
    for link in ((b"x" * 16, 3, 0), (_ISSUANCE, 2, 0), (_ISSUANCE, 3, 1)):
        with pytest.raises(ValueError, match="invalid sealed layer link"):
            _parse_inner(lite, link)
    assert _parse_inner(lite, (_ISSUANCE, 3, 0)).retained == 0


def test_protected_authentication_length_and_trailing_bytes_are_checked() -> None:
    fixture = _make_inner(1, 0, creator_maximum=2)
    auth_length = struct.unpack_from("<Q", fixture.data, 64 + 40)[0]
    auth = fixture.data[fixture.auth_offset : fixture.auth_offset + auth_length]
    assert len(auth) == 209 + 33 * 3 + 66 + len(_POLICY)
    assert _parse_auth(auth, 0).creator_maximum == 2
    for invalid in (auth[:-1], auth + b"\x00", b"DAU1" + auth[4:]):
        with pytest.raises(ValueError, match="invalid protected authentication envelope"):
            _parse_auth(invalid, 0)
