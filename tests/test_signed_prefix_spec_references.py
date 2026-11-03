"""Prefix and manifest byte references, not a production container verifier."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import pytest

_TIER_FIELDS = ("tier_lite", "tier_full", "tier_pro", "tier_ultra")
_PREFIX_DOMAINS = (b"DNAM-MASTER-PREFIX-V1", b"DNAM-BUYER-PREFIX-V1")


def _field(label: str, value: bytes) -> bytes:
    name = label.encode("ascii")
    return struct.pack("<I", len(name)) + name + struct.pack("<Q", len(value)) + value


def _plaintext_shared(preamble: bytes, directory: bytes, shared_binary: bytes) -> bytes:
    if not preamble or not directory:
        raise ValueError("invalid plaintext structure")
    return (
        _field("domain", b"DNAM-PLAINTEXT-SHARED-V1")
        + _field("preamble", preamble)
        + _field("directory", directory)
        + _field("shared_model_binary", shared_binary)
    )


def _prefixes(
    domain: bytes, header_digest: bytes, shared: bytes, tiers: tuple[bytes, ...]
) -> tuple[bytes, ...]:
    if domain not in _PREFIX_DOMAINS or len(header_digest) != 32 or not 1 <= len(tiers) <= 4:
        raise ValueError("invalid prefix inputs")
    state = hashlib.sha256()
    state.update(_field("domain", domain))
    state.update(_field("immutable_header_sha256", header_digest))
    state.update(_field("shared_model_data", shared))
    results = []
    for label, payload in zip(_TIER_FIELDS, tiers, strict=False):
        state.update(_field(label, payload))
        results.append(state.copy().digest())
    return tuple(results)


def test_plaintext_shared_framing_binds_open_structure_and_tier_ends() -> None:
    shared = _plaintext_shared(b"p", b"dd", b"xyz")
    assert shared == (
        b"\x06\x00\x00\x00domain\x18\x00\x00\x00\x00\x00\x00\x00DNAM-PLAINTEXT-SHARED-V1"
        b"\x08\x00\x00\x00preamble\x01\x00\x00\x00\x00\x00\x00\x00p"
        b"\x09\x00\x00\x00directory\x02\x00\x00\x00\x00\x00\x00\x00dd"
        b"\x13\x00\x00\x00shared_model_binary\x03\x00\x00\x00\x00\x00\x00\x00xyz"
    )
    with pytest.raises(ValueError, match="invalid plaintext structure"):
        _plaintext_shared(b"", b"dd", b"xyz")
    with pytest.raises(ValueError, match="invalid plaintext structure"):
        _plaintext_shared(b"p", b"", b"xyz")

    header = b"h" * 32
    preamble = b"original preamble"
    ends = struct.pack("<QQ", 384, 448)
    directory = b"original directory" + ends
    original = _plaintext_shared(preamble, directory, b"control bytes")
    prefix = _prefixes(_PREFIX_DOMAINS[0], header, original, (b"lite",))
    for changed in (
        _plaintext_shared(b"changed preamble", directory, b"control bytes"),
        _plaintext_shared(preamble, b"changed directory" + ends, b"control bytes"),
        _plaintext_shared(
            preamble, b"original directory" + struct.pack("<QQ", 384, 512), b"control bytes"
        ),
        _plaintext_shared(preamble, directory, b"changed control bytes"),
    ):
        assert _prefixes(_PREFIX_DOMAINS[0], header, changed, (b"lite",)) != prefix

    mutable_contents = (b"preset A", b"preset B")
    assert mutable_contents[0] != mutable_contents[1]
    assert _prefixes(_PREFIX_DOMAINS[0], header, original, (b"lite",)) == prefix


def _prefix_list(digests: tuple[bytes, ...]) -> bytes:
    if not 1 <= len(digests) <= 4 or any(len(digest) != 32 for digest in digests):
        raise ValueError("invalid prefix list")
    return bytes([len(digests)]) + b"".join(
        bytes([tier]) + digest for tier, digest in enumerate(digests)
    )


def _parse_prefix_list(data: bytes, maximum_tier: int) -> tuple[bytes, ...]:
    if not 0 <= maximum_tier <= 3 or not data:
        raise ValueError("invalid prefix list")
    count = data[0]
    if count != maximum_tier + 1 or len(data) != 1 + 33 * count:
        raise ValueError("invalid prefix list")
    if any(data[1 + 33 * tier] != tier for tier in range(count)):
        raise ValueError("invalid prefix list")
    return tuple(data[2 + 33 * tier : 34 + 33 * tier] for tier in range(count))


@dataclass(frozen=True)
class _CreatorEnvelope:
    key: bytes
    prefixes: tuple[bytes, ...]
    signature: bytes


@dataclass(frozen=True)
class _BuyerEnvelope:
    key: bytes
    policy: bytes
    prefixes: tuple[bytes, ...]
    signature: bytes


def _encode_open_envelope(creator: _CreatorEnvelope, buyer: _BuyerEnvelope | None) -> bytes:
    if len(creator.key) != 32 or len(creator.signature) != 64:
        raise ValueError("invalid authentication envelope")
    creator_list = _prefix_list(creator.prefixes)
    encoded = (
        b"DAU1"
        + bytes((int(buyer is not None), 0, 0, 0))
        + creator.key
        + bytes((len(creator.prefixes) - 1,))
        + creator_list
        + creator.signature
    )
    if buyer is not None:
        if (
            len(buyer.key) != 32
            or buyer.key == creator.key
            or not buyer.policy
            or len(buyer.policy) >= 2**32
            or len(buyer.signature) != 64
            or not 1 <= len(buyer.prefixes) <= len(creator.prefixes)
        ):
            raise ValueError("invalid authentication envelope")
        encoded += (
            buyer.key
            + struct.pack("<I", len(buyer.policy))
            + buyer.policy
            + bytes((len(buyer.prefixes) - 1,))
            + _prefix_list(buyer.prefixes)
            + buyer.signature
        )
    return encoded


def _parse_open_envelope(
    data: bytes, preamble_kind: int, preamble_maximum: int, maximum_policy_bytes: int
) -> tuple[_CreatorEnvelope, _BuyerEnvelope | None]:
    if preamble_kind not in (0, 1) or not 0 <= preamble_maximum <= 3:
        raise ValueError("invalid authentication envelope")
    cursor = 0

    def take(length: int) -> bytes:
        nonlocal cursor
        end = cursor + length
        if length < 0 or end > len(data):
            raise ValueError("invalid authentication envelope")
        value = data[cursor:end]
        cursor = end
        return value

    if take(4) != b"DAU1" or take(1) != bytes((preamble_kind,)) or take(3) != bytes(3):
        raise ValueError("invalid authentication envelope")
    creator_key = take(32)
    creator_maximum = take(1)[0]
    if creator_maximum > 3:
        raise ValueError("invalid authentication envelope")
    creator_list = _parse_prefix_list(take(1 + 33 * (creator_maximum + 1)), creator_maximum)
    creator = _CreatorEnvelope(creator_key, creator_list, take(64))
    buyer = None
    if preamble_kind == 1:
        buyer_key = take(32)
        policy_length = struct.unpack("<I", take(4))[0]
        if buyer_key == creator_key or not 0 < policy_length <= maximum_policy_bytes:
            raise ValueError("invalid authentication envelope")
        policy = take(policy_length)
        buyer_maximum = take(1)[0]
        if buyer_maximum != preamble_maximum or buyer_maximum > creator_maximum:
            raise ValueError("invalid authentication envelope")
        buyer_list = _parse_prefix_list(take(1 + 33 * (buyer_maximum + 1)), buyer_maximum)
        buyer = _BuyerEnvelope(buyer_key, policy, buyer_list, take(64))
    elif creator_maximum != preamble_maximum:
        raise ValueError("invalid authentication envelope")
    if cursor != len(data):
        raise ValueError("invalid authentication envelope")
    return creator, buyer


def test_open_authentication_envelope_has_canonical_bytes_and_lengths() -> None:
    creator = _CreatorEnvelope(b"c" * 32, (b"a" * 32, b"b" * 32), b"s" * 64)
    creator_only = _encode_open_envelope(creator, None)
    assert creator_only == (
        b"DAU1\x00\x00\x00\x00"
        + b"c" * 32
        + b"\x01\x02\x00"
        + b"a" * 32
        + b"\x01"
        + b"b" * 32
        + b"s" * 64
    )
    assert len(creator_only) == 106 + 33 * 2
    assert _parse_open_envelope(creator_only, 0, 1, 1024) == (creator, None)

    buyer = _BuyerEnvelope(b"m" * 32, b"policy", (b"x" * 32,), b"t" * 64)
    commercial = _encode_open_envelope(creator, buyer)
    assert commercial == (
        b"DAU1\x01\x00\x00\x00"
        + creator_only[8:]
        + b"m" * 32
        + b"\x06\x00\x00\x00policy"
        + b"\x00\x01\x00"
        + b"x" * 32
        + b"t" * 64
    )
    assert len(commercial) == len(creator_only) + 102 + len(buyer.policy) + 33
    assert _parse_open_envelope(commercial, 1, 0, 1024) == (creator, buyer)


def test_open_authentication_envelope_rejects_noncanonical_and_mismatched_data() -> None:
    creator = _CreatorEnvelope(b"c" * 32, (b"a" * 32, b"b" * 32), b"s" * 64)
    buyer = _BuyerEnvelope(b"m" * 32, b"policy", (b"x" * 32,), b"t" * 64)
    data = _encode_open_envelope(creator, buyer)
    for index, value, message in (
        (0, 0, "invalid authentication envelope"),
        (4, 0, "invalid authentication envelope"),
        (5, 1, "invalid authentication envelope"),
        (40, 4, "invalid authentication envelope"),
        (41, 1, "invalid prefix list"),
        (42, 1, "invalid prefix list"),
    ):
        changed = bytearray(data)
        changed[index] = value
        with pytest.raises(ValueError, match=message):
            _parse_open_envelope(bytes(changed), 1, 0, 1024)
    for invalid in (b"", data[:-1], data + b"\x00"):
        with pytest.raises(ValueError, match="invalid authentication envelope"):
            _parse_open_envelope(invalid, 1, 0, 1024)
    with pytest.raises(ValueError, match="invalid authentication envelope"):
        _parse_open_envelope(data, 1, 1, 1024)
    with pytest.raises(ValueError, match="invalid authentication envelope"):
        _parse_open_envelope(data, 1, 0, 5)
    buyer_offset = len(_encode_open_envelope(creator, None))
    for changed in (
        data[:buyer_offset] + creator.key + data[buyer_offset + 32 :],
        data[: buyer_offset + 32] + bytes(4) + data[buyer_offset + 36 :],
        data[: buyer_offset + 32] + b"\xff" * 4 + data[buyer_offset + 36 :],
        data[: buyer_offset + 42] + b"\x02" + data[buyer_offset + 43 :],
    ):
        with pytest.raises(ValueError, match="invalid authentication envelope"):
            _parse_open_envelope(changed, 1, 0, 1024)
    with pytest.raises(ValueError, match="invalid authentication envelope"):
        _parse_open_envelope(_encode_open_envelope(creator, None), 0, 0, 1024)
    with pytest.raises(ValueError, match="invalid authentication envelope"):
        _encode_open_envelope(
            creator, _BuyerEnvelope(creator.key, b"policy", buyer.prefixes, buyer.signature)
        )
    with pytest.raises(ValueError, match="invalid authentication envelope"):
        _encode_open_envelope(
            creator, _BuyerEnvelope(buyer.key, b"", buyer.prefixes, buyer.signature)
        )
    with pytest.raises(ValueError, match="invalid authentication envelope"):
        _encode_open_envelope(creator, _BuyerEnvelope(buyer.key, buyer.policy, (), buyer.signature))


def _creator_message(public_key: bytes, header_digest: bytes, digests: tuple[bytes, ...]) -> bytes:
    if len(public_key) != 32 or len(header_digest) != 32:
        raise ValueError("invalid creator message")
    prefix_list = _prefix_list(digests)
    return (
        _field("domain", b"DNAM-CREATOR-MANIFEST-V1")
        + _field("creator_pubkey_ed25519", public_key)
        + _field("immutable_header_sha256", header_digest)
        + _field("issued_max_tier", bytes([len(digests) - 1]))
        + _field("master_prefix_sha256", prefix_list)
    )


def _creator_signed_identity(message: bytes, signature: bytes) -> bytes:
    if len(signature) != 64:
        raise ValueError("invalid creator signature")
    return hashlib.sha256(
        _field("domain", b"DNAM-CREATOR-SIGNED-V1")
        + _field("creator_manifest", message)
        + _field("creator_signature", signature)
    ).digest()


def _ciphertext_prefixes(
    context: bytes, shared: bytes, tiers: tuple[bytes, ...]
) -> tuple[bytes, ...]:
    if len(context) != 32 or not 1 <= len(tiers) <= 4:
        raise ValueError("invalid ciphertext prefix inputs")
    state = hashlib.sha256()
    state.update(_field("domain", b"DNAM-CIPHERTEXT-PREFIX-V1"))
    state.update(_field("record_context_sha256", context))
    state.update(_field("shared", shared))
    results = []
    for label, payload in zip(_TIER_FIELDS, tiers, strict=False):
        state.update(_field(label, payload))
        results.append(state.copy().digest())
    return tuple(results)


def _license_policy_digest(policy: bytes) -> bytes:
    if not policy:
        raise ValueError("invalid license policy")
    return hashlib.sha256(
        _field("domain", b"DNAM-LICENSE-POLICY-V1") + _field("policy", policy)
    ).digest()


def _buyer_message(
    marketplace_key: bytes,
    creator_identity: bytes,
    policy_digest: bytes,
    profile: int,
    buyer_prefixes: tuple[bytes, ...],
    inner_context: bytes,
    ciphertext_prefixes: tuple[bytes, ...],
) -> bytes:
    if (
        len(marketplace_key) != 32
        or len(creator_identity) != 32
        or len(policy_digest) != 32
        or profile not in (0, 1, 2)
    ):
        raise ValueError("invalid buyer message")
    buyer_list = _prefix_list(buyer_prefixes)
    if profile == 0:
        if inner_context or ciphertext_prefixes:
            raise ValueError("invalid buyer message")
        ciphertext_list = b""
    else:
        if len(inner_context) != 32 or len(ciphertext_prefixes) != len(buyer_prefixes):
            raise ValueError("invalid buyer message")
        ciphertext_list = _prefix_list(ciphertext_prefixes)
    return (
        _field("domain", b"DNAM-BUYER-RECEIPT-V1")
        + _field("marketplace_pubkey_ed25519", marketplace_key)
        + _field("creator_signed_manifest_sha256", creator_identity)
        + _field("license_policy_sha256", policy_digest)
        + _field("protection_profile", bytes([profile]))
        + _field("issued_max_tier", bytes([len(buyer_prefixes) - 1]))
        + _field("buyer_prefix_sha256", buyer_list)
        + _field("inner_record_context_sha256", inner_context)
        + _field("inner_ciphertext_prefix_sha256", ciphertext_list)
    )


def _validate_issued_tiers(
    creator_tier: int, buyer_tier: int, profile: int, inner_plan_tier: int | None
) -> None:
    if not 0 <= buyer_tier <= creator_tier <= 3 or profile not in (0, 1, 2):
        raise ValueError("invalid issued tier")
    if (profile == 0 and inner_plan_tier is not None) or (
        profile != 0 and inner_plan_tier != buyer_tier
    ):
        raise ValueError("invalid issued tier")


def _outer_public_metadata_digest(preamble: bytes, directory: bytes, auxiliary: bytes) -> bytes:
    return hashlib.sha256(
        _field("domain", b"DNAM-OUTER-PUBLIC-METADATA-V1")
        + _field("preamble", preamble)
        + _field("directory", directory)
        + _field("public_auxiliary", auxiliary)
    ).digest()


def _outer_message(
    signer_key: bytes, metadata_digest: bytes, record_context: bytes, prefixes: tuple[bytes, ...]
) -> bytes:
    if len(signer_key) != 32 or len(metadata_digest) != 32 or len(record_context) != 32:
        raise ValueError("invalid outer message")
    prefix_list = _prefix_list(prefixes)
    return (
        _field("domain", b"DNAM-OUTER-MANIFEST-V1")
        + _field("outer_signer_pubkey_ed25519", signer_key)
        + _field("outer_public_metadata_sha256", metadata_digest)
        + _field("outer_record_context_sha256", record_context)
        + _field("issued_max_tier", bytes([len(prefixes) - 1]))
        + _field("outer_ciphertext_prefix_sha256", prefix_list)
    )


def _validate_sealed_layer_link(
    inner: tuple[bytes, int, int, int], outer: tuple[bytes, int, int, int], receipt_tier: int
) -> None:
    inner_id, inner_layer, inner_profile, inner_tier = inner
    outer_id, outer_layer, outer_profile, outer_tier = outer
    if (
        len(inner_id) != 16
        or inner_id != outer_id
        or inner_layer != 0
        or outer_layer != 1
        or inner_profile != 2
        or outer_profile != 2
        or not 0 <= receipt_tier <= 3
        or inner_tier != receipt_tier
        or outer_tier != receipt_tier
    ):
        raise ValueError("invalid sealed layer link")


def test_plaintext_prefix_has_explicit_domain_lengths_and_tier_label() -> None:
    header = bytes(range(32))
    shared = b"\x01\x02"
    lite = b"abc"
    encoded = (
        b"\x06\x00\x00\x00domain\x15\x00\x00\x00\x00\x00\x00\x00DNAM-MASTER-PREFIX-V1"
        b"\x17\x00\x00\x00immutable_header_sha256\x20\x00\x00\x00\x00\x00\x00\x00"
        + header
        + b"\x11\x00\x00\x00shared_model_data\x02\x00\x00\x00\x00\x00\x00\x00\x01\x02"
        b"\x09\x00\x00\x00tier_lite\x03\x00\x00\x00\x00\x00\x00\x00abc"
    )
    expected = hashlib.sha256(encoded).digest()
    assert _prefixes(_PREFIX_DOMAINS[0], header, shared, (lite,)) == (expected,)
    assert _prefixes(_PREFIX_DOMAINS[1], header, shared, (lite,)) != (expected,)
    assert _prefixes(_PREFIX_DOMAINS[0], header, b"A", (b"BC",)) != _prefixes(
        _PREFIX_DOMAINS[0], header, b"AB", (b"C",)
    )


def test_recomputed_retained_prefixes_match_original_and_bind_shared_bytes() -> None:
    header = b"h" * 32
    shared = b"control descriptors and other immutable shared data"
    tiers = (b"lite", b"full", b"pro", b"ultra")
    originals = _prefixes(_PREFIX_DOMAINS[0], header, shared, tiers)
    encoded_list = _prefix_list(originals)
    assert _parse_prefix_list(encoded_list, 3) == originals
    for retained_count in range(1, 5):
        retained = _prefixes(_PREFIX_DOMAINS[0], header, shared, tiers[:retained_count])
        assert retained[-1] == originals[retained_count - 1]
    assert all(
        left != right
        for left, right in zip(
            originals, _prefixes(_PREFIX_DOMAINS[0], header, shared + b"x", tiers), strict=True
        )
    )
    changed_full = _prefixes(_PREFIX_DOMAINS[0], header, shared, (b"lite", b"FULL", *tiers[2:]))
    assert changed_full[0] == originals[0]
    assert all(left != right for left, right in zip(originals[1:], changed_full[1:], strict=True))
    reordered = _prefixes(_PREFIX_DOMAINS[0], header, shared, (b"full", b"lite", *tiers[2:]))
    assert reordered != originals
    assert _prefixes(_PREFIX_DOMAINS[0], b"x" * 32, shared, tiers) != originals


def test_prefix_list_is_tier_labelled_complete_and_canonical() -> None:
    digests = (b"a" * 32, b"b" * 32)
    encoded = _prefix_list(digests)
    assert encoded == b"\x02\x00" + digests[0] + b"\x01" + digests[1]
    assert _parse_prefix_list(encoded, 1) == digests
    for invalid in [
        b"",
        b"\x00",
        b"\x05",
        encoded[:-1],
        encoded + b"\x00",
        b"\x02\x01" + digests[0] + b"\x00" + digests[1],
        b"\x02\x00" + digests[0] + b"\x00" + digests[1],
    ]:
        with pytest.raises(ValueError, match="invalid prefix list"):
            _parse_prefix_list(invalid, 1)
    with pytest.raises(ValueError, match="invalid prefix list"):
        _parse_prefix_list(encoded, 0)
    for invalid_digests in [(), (b"a" * 31,), (b"a" * 32,) * 5]:
        with pytest.raises(ValueError, match="invalid prefix list"):
            _prefix_list(invalid_digests)


def test_creator_message_and_signed_identity_bind_manifest_and_signature() -> None:
    public_key = b"k" * 32
    header = b"h" * 32
    prefixes = _prefixes(_PREFIX_DOMAINS[0], header, b"shared", (b"lite", b"full"))
    message = _creator_message(public_key, header, prefixes)
    assert message == (
        _field("domain", b"DNAM-CREATOR-MANIFEST-V1")
        + _field("creator_pubkey_ed25519", public_key)
        + _field("immutable_header_sha256", header)
        + _field("issued_max_tier", b"\x01")
        + _field("master_prefix_sha256", _prefix_list(prefixes))
    )
    for other in [
        _creator_message(b"z" * 32, header, prefixes),
        _creator_message(public_key, b"x" * 32, prefixes),
        _creator_message(public_key, header, prefixes[:1]),
        _creator_message(public_key, header, (prefixes[0], b"x" * 32)),
    ]:
        assert message != other
        assert _creator_signed_identity(message, b"s" * 64) != _creator_signed_identity(
            other, b"s" * 64
        )
    assert _creator_signed_identity(message, b"s" * 64) != _creator_signed_identity(
        message, b"t" * 64
    )
    with pytest.raises(ValueError, match="invalid creator signature"):
        _creator_signed_identity(message, b"s" * 63)


def test_ciphertext_prefixes_bind_context_shared_tiers_and_padding() -> None:
    context = b"c" * 32
    shared = b"shared ciphertext and tag" + bytes(3)
    tiers = (b"lite ciphertext" + bytes(2), b"full ciphertext" + bytes(2))
    originals = _ciphertext_prefixes(context, shared, tiers)
    assert _ciphertext_prefixes(context, shared, tiers[:1]) == originals[:1]
    assert _ciphertext_prefixes(b"x" * 32, shared, tiers) != originals
    assert _ciphertext_prefixes(context, shared[:-1] + b"x", tiers) != originals
    assert _ciphertext_prefixes(context, shared, (tiers[1], tiers[0])) != originals
    changed_full = _ciphertext_prefixes(context, shared, (tiers[0], tiers[1] + b"x"))
    assert changed_full[0] == originals[0]
    assert changed_full[1] != originals[1]
    expected_lite = hashlib.sha256(
        _field("domain", b"DNAM-CIPHERTEXT-PREFIX-V1")
        + _field("record_context_sha256", context)
        + _field("shared", shared)
        + _field("tier_lite", tiers[0])
    ).digest()
    assert originals[0] == expected_lite


def test_buyer_message_binds_creator_policy_and_both_prefix_lists() -> None:
    marketplace_key = b"m" * 32
    creator_identity = b"c" * 32
    policy = _license_policy_digest(b"versioned policy for buyer A")
    buyer_prefixes = (b"l" * 32, b"f" * 32)
    context = b"i" * 32
    ciphertext_prefixes = (b"x" * 32, b"y" * 32)
    message = _buyer_message(
        marketplace_key, creator_identity, policy, 1, buyer_prefixes, context, ciphertext_prefixes
    )
    assert message == (
        _field("domain", b"DNAM-BUYER-RECEIPT-V1")
        + _field("marketplace_pubkey_ed25519", marketplace_key)
        + _field("creator_signed_manifest_sha256", creator_identity)
        + _field("license_policy_sha256", policy)
        + _field("protection_profile", b"\x01")
        + _field("issued_max_tier", b"\x01")
        + _field("buyer_prefix_sha256", _prefix_list(buyer_prefixes))
        + _field("inner_record_context_sha256", context)
        + _field("inner_ciphertext_prefix_sha256", _prefix_list(ciphertext_prefixes))
    )
    alternatives = [
        _buyer_message(
            b"z" * 32, creator_identity, policy, 1, buyer_prefixes, context, ciphertext_prefixes
        ),
        _buyer_message(
            marketplace_key, b"d" * 32, policy, 1, buyer_prefixes, context, ciphertext_prefixes
        ),
        _buyer_message(
            marketplace_key,
            creator_identity,
            _license_policy_digest(b"other"),
            1,
            buyer_prefixes,
            context,
            ciphertext_prefixes,
        ),
        _buyer_message(
            marketplace_key,
            creator_identity,
            policy,
            2,
            buyer_prefixes,
            context,
            ciphertext_prefixes,
        ),
        _buyer_message(
            marketplace_key,
            creator_identity,
            policy,
            1,
            buyer_prefixes[:1],
            context,
            ciphertext_prefixes[:1],
        ),
        _buyer_message(
            marketplace_key,
            creator_identity,
            policy,
            1,
            (b"q" * 32, buyer_prefixes[1]),
            context,
            ciphertext_prefixes,
        ),
        _buyer_message(
            marketplace_key,
            creator_identity,
            policy,
            1,
            buyer_prefixes,
            b"j" * 32,
            ciphertext_prefixes,
        ),
        _buyer_message(
            marketplace_key,
            creator_identity,
            policy,
            1,
            buyer_prefixes,
            context,
            (ciphertext_prefixes[0], b"z" * 32),
        ),
    ]
    assert all(value != message for value in alternatives)
    assert _buyer_message(marketplace_key, creator_identity, policy, 0, buyer_prefixes, b"", ())
    for profile, candidate_context, candidate_prefixes in [
        (0, context, ()),
        (0, b"", ciphertext_prefixes),
        (1, b"", ciphertext_prefixes),
        (1, context, ciphertext_prefixes[:1]),
        (3, context, ciphertext_prefixes),
    ]:
        with pytest.raises(ValueError, match="invalid buyer message"):
            _buyer_message(
                marketplace_key,
                creator_identity,
                policy,
                profile,
                buyer_prefixes,
                candidate_context,
                candidate_prefixes,
            )
    with pytest.raises(ValueError, match="invalid license policy"):
        _license_policy_digest(b"")


def test_issued_tier_cannot_exceed_creator_or_disagree_with_inner_plan() -> None:
    _validate_issued_tiers(3, 1, 1, 1)
    _validate_issued_tiers(2, 2, 2, 2)
    _validate_issued_tiers(3, 0, 0, None)
    for creator, buyer, profile, inner in [
        (1, 2, 1, 2),
        (3, 1, 1, 3),
        (3, 1, 2, None),
        (3, 1, 0, 1),
        (4, 1, 1, 1),
    ]:
        with pytest.raises(ValueError, match="invalid issued tier"):
            _validate_issued_tiers(creator, buyer, profile, inner)


def test_public_outer_manifest_binds_structural_bytes_context_and_prefixes() -> None:
    key = b"k" * 32
    context = b"c" * 32
    prefixes = (b"a" * 32, b"b" * 32)
    metadata = _outer_public_metadata_digest(b"preamble", b"directory", b"plan and slots")
    assert (
        metadata
        == hashlib.sha256(
            b"\x06\x00\x00\x00domain\x1d\x00\x00\x00\x00\x00\x00\x00DNAM-OUTER-PUBLIC-METADATA-V1"
            b"\x08\x00\x00\x00preamble\x08\x00\x00\x00\x00\x00\x00\x00preamble"
            b"\x09\x00\x00\x00directory\x09\x00\x00\x00\x00\x00\x00\x00directory"
            b"\x10\x00\x00\x00public_auxiliary\x0e\x00\x00\x00\x00\x00\x00\x00plan and slots"
        ).digest()
    )
    message = _outer_message(key, metadata, context, prefixes)
    assert message == (
        _field("domain", b"DNAM-OUTER-MANIFEST-V1")
        + _field("outer_signer_pubkey_ed25519", key)
        + _field("outer_public_metadata_sha256", metadata)
        + _field("outer_record_context_sha256", context)
        + _field("issued_max_tier", b"\x01")
        + _field("outer_ciphertext_prefix_sha256", _prefix_list(prefixes))
    )
    alternatives = [
        _outer_message(b"z" * 32, metadata, context, prefixes),
        _outer_message(key, metadata, b"d" * 32, prefixes),
        _outer_message(key, metadata, context, prefixes[:1]),
        _outer_message(key, metadata, context, (prefixes[0], b"x" * 32)),
    ]
    for preamble, directory, auxiliary in [
        (b"other", b"directory", b"plan and slots"),
        (b"preamble", b"other", b"plan and slots"),
        (b"preamble", b"directory", b"other"),
    ]:
        digest = _outer_public_metadata_digest(preamble, directory, auxiliary)
        alternatives.append(_outer_message(key, digest, context, prefixes))
    assert all(value != message for value in alternatives)
    assert _outer_public_metadata_digest(b"A", b"BC", b"") != _outer_public_metadata_digest(
        b"AB", b"C", b""
    )
    for bad_key, bad_metadata, bad_context in [
        (key[:-1], metadata, context),
        (key, metadata[:-1], context),
        (key, metadata, context[:-1]),
    ]:
        with pytest.raises(ValueError, match="invalid outer message"):
            _outer_message(bad_key, bad_metadata, bad_context, prefixes)
    with pytest.raises(ValueError, match="invalid prefix list"):
        _outer_message(key, metadata, context, ())


def test_sealed_layer_links_require_same_issuance_profile_and_original_tier() -> None:
    inner = (b"i" * 16, 0, 2, 1)
    outer = (b"i" * 16, 1, 2, 1)
    _validate_sealed_layer_link(inner, outer, 1)
    for changed_inner, changed_outer, receipt_tier in [
        ((b"x" * 16, 0, 2, 1), outer, 1),
        ((b"i" * 15, 0, 2, 1), outer, 1),
        ((b"i" * 16, 1, 2, 1), outer, 1),
        ((b"i" * 16, 0, 1, 1), outer, 1),
        (inner, (b"i" * 16, 0, 2, 1), 1),
        (inner, (b"i" * 16, 1, 1, 1), 1),
        (inner, (b"i" * 16, 1, 2, 2), 1),
        (inner, outer, 2),
    ]:
        with pytest.raises(ValueError, match="invalid sealed layer link"):
            _validate_sealed_layer_link(changed_inner, changed_outer, receipt_tier)
