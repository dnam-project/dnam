"""Certificate-framing examples, not production signing or hardware attestation tests."""

from __future__ import annotations

import hashlib
import re
import struct

import pytest


def _field(label: str, value: bytes) -> bytes:
    name = label.encode("ascii")
    return struct.pack("<I", len(name)) + name + struct.pack("<Q", len(value)) + value


def _legacy_message(vendor: str, model: str, device_id: bytes, public_key: bytes) -> bytes:
    return (
        b"DNAM-OEM-CERT-V1:"
        + vendor.encode("utf-8")
        + b"|"
        + model.encode("utf-8")
        + b"|"
        + device_id.hex().encode("ascii")
        + b"|"
        + public_key
    )


def _certificate_message(
    vendor: str,
    model: str,
    device_id: bytes,
    public_key: bytes,
    issuer_public_key: bytes,
    assurance: int,
    policy_digest: bytes,
    evidence_digest: bytes,
) -> bytes:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,31}", vendor):
        raise ValueError("invalid vendor")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", model):
        raise ValueError("invalid model")
    if (
        len(device_id) != 16
        or len(public_key) != 32
        or len(issuer_public_key) != 32
        or len(policy_digest) != 32
        or len(evidence_digest) != 32
        or assurance not in (1, 2)
    ):
        raise ValueError("invalid certificate field")
    return b"".join(
        (
            _field("domain", b"DNAM-OEM-CERT-V2"),
            _field("oem_vendor", vendor.encode("ascii")),
            _field("model_code", model.encode("ascii")),
            _field("device_id_hash", device_id),
            _field("public_key_x25519", public_key),
            _field("issuer_key_sha256", hashlib.sha256(issuer_public_key).digest()),
            _field("assurance", bytes([assurance])),
            _field("attestation_policy_sha256", policy_digest),
            _field("attestation_evidence_sha256", evidence_digest),
        )
    )


def test_legacy_variable_length_identifiers_have_an_ambiguous_signing_message() -> None:
    device_id = bytes(range(16))
    public_key = bytes(range(32))
    first = ("acme|pedal", "mk2")
    second = ("acme", "pedal|mk2")
    assert first != second
    assert _legacy_message(*first, device_id, public_key) == _legacy_message(
        *second, device_id, public_key
    )
    assert _field("oem_vendor", first[0].encode()) + _field(
        "model_code", first[1].encode()
    ) != _field("oem_vendor", second[0].encode()) + _field("model_code", second[1].encode())
    with pytest.raises(ValueError, match="invalid vendor"):
        _certificate_message(*first, device_id, public_key, bytes(32), 1, bytes(32), bytes(32))
    with pytest.raises(ValueError, match="invalid model"):
        _certificate_message(*second, device_id, public_key, bytes(32), 1, bytes(32), bytes(32))


def test_v2_certificate_message_matches_fixed_field_and_length_vector() -> None:
    message = _certificate_message(
        "acme", "pedal-v1", bytes(range(16)), bytes(range(32)), bytes(32), 1, b"P" * 32, b"E" * 32
    )
    issuer_digest = bytes.fromhex(
        "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925"
    )
    expected = (
        b"\x06\x00\x00\x00domain\x10\x00\x00\x00\x00\x00\x00\x00DNAM-OEM-CERT-V2"
        b"\x0a\x00\x00\x00oem_vendor\x04\x00\x00\x00\x00\x00\x00\x00acme"
        b"\x0a\x00\x00\x00model_code\x08\x00\x00\x00\x00\x00\x00\x00pedal-v1"
        b"\x0e\x00\x00\x00device_id_hash\x10\x00\x00\x00\x00\x00\x00\x00"
        + bytes(range(16))
        + b"\x11\x00\x00\x00public_key_x25519\x20\x00\x00\x00\x00\x00\x00\x00"
        + bytes(range(32))
        + b"\x11\x00\x00\x00issuer_key_sha256\x20\x00\x00\x00\x00\x00\x00\x00"
        + issuer_digest
        + b"\x09\x00\x00\x00assurance\x01\x00\x00\x00\x00\x00\x00\x00\x01"
        + b"\x19\x00\x00\x00attestation_policy_sha256\x20\x00\x00\x00\x00\x00\x00\x00"
        + b"P" * 32
        + b"\x1b\x00\x00\x00attestation_evidence_sha256\x20\x00\x00\x00\x00\x00\x00\x00"
        + b"E" * 32
    )
    assert message == expected


@pytest.mark.parametrize(
    ("vendor", "model"),
    [
        ("", "pedal"),
        ("Acme", "pedal"),
        ("acme|pedal", "mk2"),
        ("a" * 33, "pedal"),
        ("acme", ""),
        ("acme", "Pedal"),
        ("acme", "pedal|mk2"),
        ("acme", "p" * 65),
        ("acme", "pédal"),
    ],
)
def test_v2_rejects_noncanonical_identifiers(vendor: str, model: str) -> None:
    with pytest.raises(ValueError, match=r"invalid (vendor|model)"):
        _certificate_message(
            vendor, model, bytes(16), bytes(32), bytes(32), 1, bytes(32), bytes(32)
        )


@pytest.mark.parametrize(
    ("device_id", "public_key", "issuer_key", "assurance", "policy", "evidence"),
    [
        (bytes(15), bytes(32), bytes(32), 1, bytes(32), bytes(32)),
        (bytes(16), bytes(31), bytes(32), 1, bytes(32), bytes(32)),
        (bytes(16), bytes(32), bytes(31), 1, bytes(32), bytes(32)),
        (bytes(16), bytes(32), bytes(32), 0, bytes(32), bytes(32)),
        (bytes(16), bytes(32), bytes(32), 3, bytes(32), bytes(32)),
        (bytes(16), bytes(32), bytes(32), 1, bytes(31), bytes(32)),
        (bytes(16), bytes(32), bytes(32), 1, bytes(32), bytes(31)),
    ],
)
def test_v2_rejects_wrong_field_sizes_and_unknown_assurance(
    device_id: bytes,
    public_key: bytes,
    issuer_key: bytes,
    assurance: int,
    policy: bytes,
    evidence: bytes,
) -> None:
    with pytest.raises(ValueError, match="invalid certificate field"):
        _certificate_message(
            "acme", "pedal", device_id, public_key, issuer_key, assurance, policy, evidence
        )


def test_v2_binds_identity_key_assurance_and_evidence() -> None:
    common = ("acme", "pedal", bytes(16), bytes(32), bytes(32), 1, bytes(32), bytes(32))
    baseline = _certificate_message(*common)
    changes = (
        ("other", *common[1:]),
        (common[0], "other", *common[2:]),
        (*common[:2], b"\x01" + bytes(15), *common[3:]),
        (*common[:3], b"\x01" + bytes(31), *common[4:]),
        (*common[:4], b"\x01" + bytes(31), *common[5:]),
        (*common[:5], 2, *common[6:]),
        (*common[:6], b"\x01" + bytes(31), *common[7:]),
        (*common[:7], b"\x01" + bytes(31)),
    )
    assert all(_certificate_message(*change) != baseline for change in changes)
