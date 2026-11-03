"""Independent watermark and fingerprint examples, not production security tests."""

from __future__ import annotations

import hashlib
import hmac
import math
import random
import struct
from collections.abc import Sequence
from dataclasses import dataclass, replace
from uuid import UUID

import pytest

_TIER_MODES = (4, 4, 8, 16)
_F32 = struct.Struct("<f")
_F32X32 = struct.Struct("<32f")
_F16 = struct.Struct("<e")


def _legacy_seed(secret: bytes, model_id: str, license_id: str) -> bytes:
    data = b"DNAM-TARDOS-V1:" + secret + model_id.encode() + b":" + license_id.encode()
    return hashlib.sha256(data).digest()


def _field(label: bytes, value: bytes) -> bytes:
    return struct.pack("<I", len(label)) + label + struct.pack("<Q", len(value)) + value


@dataclass(frozen=True)
class _SeedScope:
    marketplace: bytes
    key_id: bytes
    creator_manifest: bytes
    model_id: bytes
    profile_digest: bytes
    tier: int


def _seed_message(scope: _SeedScope, buyer: tuple[bytes, bytes] | None = None) -> bytes:
    fields = (
        (b"marketplace_pubkey_ed25519", scope.marketplace, 32),
        (b"watermark_key_id", scope.key_id, 16),
        (b"creator_signed_manifest_sha256", scope.creator_manifest, 32),
        (b"model_id", scope.model_id, 16),
        (b"watermark_profile_sha256", scope.profile_digest, 32),
    )
    if any(len(value) != size for _, value, size in fields):
        raise ValueError("invalid scope field length")
    if not 0 <= scope.tier <= 3:
        raise ValueError("unsupported owning tier")
    if buyer is not None and any(len(value) != 16 for value in buyer):
        raise ValueError("invalid buyer field length")
    domain = b"DNAM-WM-SHARED-SEED-V1" if buyer is None else b"DNAM-WM-BUYER-SEED-V1"
    message = _field(b"domain", domain)
    message += b"".join(_field(label, value) for label, value, _ in fields)
    message += _field(b"owning_tier", bytes((scope.tier,)))
    if buyer is not None:
        license_id, buyer_id = buyer
        message += _field(b"license_id", license_id) + _field(b"buyer_id", buyer_id)
    return message


def _seed(key: bytes, scope: _SeedScope, buyer: tuple[bytes, bytes] | None = None) -> bytes:
    if len(key) != 32:
        raise ValueError("invalid watermark key length")
    return hmac.digest(key, _seed_message(scope, buyer), "sha256")


def _profile_digest(profile: bytes) -> bytes:
    message = _field(b"domain", b"DNAM-WM-PROFILE-V1") + _field(b"profile", profile)
    return hashlib.sha256(message).digest()


def _seed_fixture() -> tuple[bytes, _SeedScope, tuple[bytes, bytes]]:
    key = bytes(range(32))
    scope = _SeedScope(
        marketplace=bytes(range(32, 64)),
        key_id=bytes(range(64, 80)),
        creator_manifest=bytes(range(80, 112)),
        model_id=bytes(range(112, 128)),
        profile_digest=bytes(range(128, 160)),
        tier=2,
    )
    buyer = (
        UUID("01966c44-47e5-7abc-8123-456789abcdef").bytes,
        UUID("01966c44-47e5-7def-b012-3456789abcde").bytes,
    )
    return key, scope, buyer


def _pack_modes(modes: Sequence[Sequence[float]]) -> bytes:
    if len(modes) % 4 or any(len(mode) != 8 for mode in modes):
        raise ValueError("four complete eight-field modes are required per block")
    blocks = []
    for start in range(0, len(modes), 4):
        values = [modes[start + lane][field] for field in range(8) for lane in range(4)]
        blocks.append(_F32X32.pack(*values))
    return b"".join(blocks)


def _stage3_component(fir: bytes, modal: bytes, added_modes: int) -> tuple[float, ...]:
    if added_modes not in _TIER_MODES:
        raise ValueError("unsupported initial modal increment")
    if len(fir) != _F32X32.size or len(modal) != added_modes * 32:
        raise ValueError("incorrect tier-local slice length")
    values = list(_F32X32.unpack(fir))
    for offset in range(0, len(modal), _F32X32.size):
        block = _F32X32.unpack_from(modal, offset)
        if not all(math.isfinite(value) for value in block):
            raise ValueError("nonfinite modal value")
        for lane in range(4):
            b_gamma = block[12 + lane]
            values.extend(
                (
                    block[lane],
                    block[4 + lane],
                    block[8 + lane],
                    b_gamma * block[16 + lane],
                    b_gamma * block[20 + lane],
                    b_gamma * block[24 + lane],
                    b_gamma * block[28 + lane],
                )
            )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite FIR or fingerprint value")
    return tuple(0.0 if value == 0.0 else value for value in values)


def _tier_fixture(tier: int) -> tuple[bytes, list[list[float]]]:
    fir = _F32X32.pack(*(float(32 * tier + i) / 128 for i in range(32)))
    modes = [
        [
            0.5,
            0.0,
            (tier * 16 + mode + 1) / 1024,
            0.5,
            (mode + 1) / 16,
            -(mode + 1) / 32,
            (mode + 3) / 64,
            -(mode + 5) / 128,
        ]
        for mode in range(_TIER_MODES[tier])
    ]
    return fir, modes


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("zero-norm comparison is inconclusive")
    return dot / (left_norm * right_norm)


def test_legacy_seed_ambiguity_when_both_ids_allow_delimiters() -> None:
    secret = bytes([0x5A]) * 32
    assert _legacy_seed(secret, "a:b", "c") == _legacy_seed(secret, "a", "b:c")

    left = _field(b"model_id", b"a:b") + _field(b"license_id", b"c")
    right = _field(b"model_id", b"a") + _field(b"license_id", b"b:c")
    assert left != right


def test_seed_messages_match_independent_openssl_vectors() -> None:
    key, scope, buyer = _seed_fixture()
    shared_message = _seed_message(scope)
    buyer_message = _seed_message(scope, buyer)

    assert len(shared_message) == 356
    assert hashlib.sha256(shared_message).hexdigest() == (
        "13d2bee542d18355e05b8e75faabbd9e64c1d245f485bc7e30c0057cc0239477"
    )
    assert _seed(key, scope).hex() == (
        "3c29ae4f95ccf1db51d722da34801dec3813809b6ffe3cdf9af4de9b0270cf4c"
    )
    assert len(buyer_message) == 429
    assert hashlib.sha256(buyer_message).hexdigest() == (
        "a31c8b7430f9ba153c0a7c838ed9ff788351f9bb8c597e75da737dd18927c118"
    )
    assert _seed(key, scope, buyer).hex() == (
        "04a710a6112a44082e84290642093fe72df7644c168b1552d8ab9e4d22a575b3"
    )
    assert _profile_digest(b"profile\x00v1\xff").hex() == (
        "ada5b074a45838ba487d6ac9bc2c11d6b781872a07d78aa3e689841338895186"
    )


def test_seed_scope_separates_issuers_buyers_profiles_and_tiers() -> None:
    key, scope, (license_id, buyer_id) = _seed_fixture()
    buyer = (license_id, buyer_id)
    shared = _seed(key, scope)
    specific = _seed(key, scope, buyer)
    assert shared != specific
    assert UUID(bytes=license_id).version == 7
    assert UUID(bytes=buyer_id).version == 7
    assert UUID(bytes=license_id).bytes_le != license_id

    def flip(value: bytes) -> bytes:
        return bytes((value[0] ^ 1,)) + value[1:]

    changes = (
        replace(scope, marketplace=flip(scope.marketplace)),
        replace(scope, key_id=flip(scope.key_id)),
        replace(scope, creator_manifest=flip(scope.creator_manifest)),
        replace(scope, model_id=flip(scope.model_id)),
        replace(scope, profile_digest=flip(scope.profile_digest)),
        replace(scope, tier=3),
    )
    for changed in changes:
        assert _seed(key, changed) != shared
        assert _seed(key, changed, buyer) != specific
    assert _seed(key, scope, (flip(license_id), buyer_id)) != specific
    assert _seed(key, scope, (license_id, flip(buyer_id))) != specific
    assert _seed(flip(key), scope, buyer) != specific


def test_owner_tier_seed_prefix_is_independent_of_issued_maximum() -> None:
    key, scope, buyer = _seed_fixture()

    def issue(max_tier: int) -> list[tuple[bytes, bytes]]:
        return [
            (_seed(key, replace(scope, tier=tier)), _seed(key, replace(scope, tier=tier), buyer))
            for tier in range(max_tier + 1)
        ]

    full_issue = issue(3)
    assert len(set(full_issue)) == 4
    for maximum in range(4):
        assert issue(maximum) == full_issue[: maximum + 1]
        for retained in range(maximum + 1):
            assert issue(maximum)[: retained + 1] == issue(retained)


def test_seed_references_reject_malformed_lengths_and_tiers() -> None:
    key, scope, (license_id, buyer_id) = _seed_fixture()
    with pytest.raises(ValueError, match="key length"):
        _seed(key[:-1], scope)
    for changed in (
        replace(scope, marketplace=scope.marketplace[:-1]),
        replace(scope, key_id=scope.key_id[:-1]),
        replace(scope, creator_manifest=scope.creator_manifest[:-1]),
        replace(scope, model_id=scope.model_id[:-1]),
        replace(scope, profile_digest=scope.profile_digest[:-1]),
    ):
        with pytest.raises(ValueError, match="scope field length"):
            _seed(key, changed)
    for tier in (-1, 4):
        with pytest.raises(ValueError, match="owning tier"):
            _seed(key, replace(scope, tier=tier))
    for buyer in ((license_id[:-1], buyer_id), (license_id, buyer_id[:-1])):
        with pytest.raises(ValueError, match="buyer field length"):
            _seed(key, scope, buyer)


def test_fp16_can_erase_a_small_additive_mark() -> None:
    masters = [1.0 + (index % 64) / 256 for index in range(768)]
    epsilon = 2.0**-14
    chips = [1.0 if index % 2 else -1.0 for index in range(len(masters))]
    marked = [master + epsilon * chip for master, chip in zip(masters, chips, strict=True)]

    assert all(marked_value != master for marked_value, master in zip(marked, masters, strict=True))
    assert all(_F32.unpack(_F32.pack(value))[0] == value for value in marked)
    quantized = [_F16.unpack(_F16.pack(value))[0] for value in marked]
    assert quantized == masters


def test_coordinate_midpoints_can_remove_additive_coalition_marks() -> None:
    rng = random.Random(20260601)
    count = 768
    masters = [1.0 + (index % 64) / 256 for index in range(count)]
    codes = [[1.0 if rng.getrandbits(1) else -1.0 for _ in range(count)] for _ in range(32)]
    epsilon = 2.0**-10
    copies = [
        [master + epsilon * chip for master, chip in zip(masters, code, strict=True)]
        for code in codes
    ]

    assert all(len({code[index] for code in codes}) == 2 for index in range(count))
    assert all(_F32.unpack(_F32.pack(value))[0] == value for copy in copies for value in copy)
    recovered = [
        (min(copy[index] for copy in copies) + max(copy[index] for copy in copies)) * 0.5
        for index in range(count)
    ]
    assert recovered == masters
    assert all(
        sum(
            chip * (value - master)
            for chip, value, master in zip(code, recovered, masters, strict=True)
        )
        == 0.0
        for code in codes
    )


def test_stage3_component_respects_tiers_and_both_readouts() -> None:
    components = []
    for tier, mode_count in enumerate(_TIER_MODES):
        fir, modes = _tier_fixture(tier)
        component = _stage3_component(fir, _pack_modes(modes), mode_count)
        assert len(component) == (60, 60, 88, 144)[tier]
        assert component[32:39] == (
            modes[0][0],
            modes[0][1],
            modes[0][2],
            modes[0][3] * modes[0][4],
            modes[0][3] * modes[0][5],
            modes[0][3] * modes[0][6],
            modes[0][3] * modes[0][7],
        )
        components.append(component)

        modes[0][6] += 0.25
        changed = _stage3_component(fir, _pack_modes(modes), mode_count)
        differences = [
            index for index, (a, b) in enumerate(zip(component, changed, strict=True)) if a != b
        ]
        assert differences == [37]

    full = tuple(value for component in components for value in component)
    for retained, cumulative in enumerate((60, 120, 208, 352), start=1):
        assert full[:cumulative] == tuple(
            value for component in components[:retained] for value in component
        )


def test_exact_reciprocal_gauge_preserves_both_channel_products() -> None:
    for tier, mode_count in enumerate(_TIER_MODES):
        fir, modes = _tier_fixture(tier)
        baseline = _stage3_component(fir, _pack_modes(modes), mode_count)
        for mode in modes:
            mode[3] *= 2.0
            for index in range(4, 8):
                mode[index] *= 0.5
        assert _stage3_component(fir, _pack_modes(modes), mode_count) == baseline


def test_tier_presence_cannot_be_inferred_from_zero_filled_features() -> None:
    lite_padded = (1.0, 0.0, 0.0, 0.0)
    full_with_zero_increment = (1.0, 0.0, 0.0, 0.0)
    another_full = (1.0, 0.0, 1.0, 0.0)
    assert lite_padded == full_with_zero_increment
    assert _cosine(lite_padded[:2], another_full[:2]) == 1.0
    assert math.isclose(_cosine(full_with_zero_increment, another_full), 1 / math.sqrt(2))
    with pytest.raises(ValueError, match="zero-norm"):
        _cosine((0.0, 0.0), (0.0, 0.0))


def test_stage3_candidate_rejects_bad_lengths_and_nonfinite_values() -> None:
    fir, modes = _tier_fixture(0)
    modal = _pack_modes(modes)
    with pytest.raises(ValueError, match="length"):
        _stage3_component(fir[:-1], modal, 4)
    with pytest.raises(ValueError, match="length"):
        _stage3_component(fir, modal[:-1], 4)
    with pytest.raises(ValueError, match="increment"):
        _stage3_component(fir, modal, 12)
    modes[0][6] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        _stage3_component(fir, _pack_modes(modes), 4)
