"""Encrypted mutable-suffix framing examples, not an HPKE implementation."""

from __future__ import annotations

import hashlib
import struct
from itertools import pairwise

import pytest

from test_encrypted_context_spec_references import (
    _MAX_RECORD_CIPHERTEXT,
    _field,
    _recipient_vector,
    _slot_vector,
)
from test_open_layout_spec_references import _frame_for_test, _read_mutable_frame

_HEADER_SIZE = 20
_FOOTER_SIZE = 16
_MAX_RECIPIENTS = 16
_MAX_CIPHERTEXT = 4_096


def _header(profile: int, count: int, ciphertext_length: int) -> bytes:
    minimum_ciphertext = 4 + 32 * count + 18 + 16
    if (
        profile not in (1, 2)
        or not 1 <= count <= _MAX_RECIPIENTS
        or not minimum_ciphertext <= ciphertext_length <= _MAX_CIPHERTEXT
        or ciphertext_length > _MAX_RECORD_CIPHERTEXT
    ):
        raise ValueError("invalid encrypted mutable header")
    return struct.pack("<4sB3sIQ", b"DME1", profile, bytes(3), count, ciphertext_length)


def _wrap_digest(context: bytes, header: bytes, recipients: tuple[bytes, ...]) -> bytes:
    if len(context) != 32 or len(header) != _HEADER_SIZE:
        raise ValueError("invalid mutable wrap input")
    return hashlib.sha256(
        _field("domain", b"DNAM-MUTABLE-WRAP-V1")
        + _field("record_context_sha256", context)
        + _field("header", header)
        + _field("recipient_key_ids", _recipient_vector(recipients))
    ).digest()


def _mutable_hpke_binding(domain: bytes, digest: bytes, index: int, recipient: bytes) -> bytes:
    if (
        domain not in (b"DNAM-MUTABLE-HPKE-INFO-V1", b"DNAM-MUTABLE-HPKE-AAD-V1")
        or len(digest) != 32
        or not 0 <= index < 2**32
        or len(recipient) != 32
    ):
        raise ValueError("invalid mutable HPKE binding")
    return (
        _field("domain", domain)
        + _field("mutable_wrap_sha256", digest)
        + _field("slot_index", struct.pack("<I", index))
        + _field("recipient_key_sha256", recipient)
    )


def _content_aad(digest: bytes, slots: tuple[tuple[bytes, bytes], ...]) -> bytes:
    if len(digest) != 32:
        raise ValueError("invalid mutable wrap digest")
    return (
        _field("domain", b"DNAM-MUTABLE-CONTENT-V1")
        + _field("mutable_wrap_sha256", digest)
        + _field("slots", _slot_vector(slots))
    )


def _read_plaintext(data: bytes, recipients: tuple[bytes, ...]) -> dict[str, object]:
    if len(data) < 4:
        raise ValueError("invalid mutable plaintext")
    count = struct.unpack("<I", data[:4])[0]
    key_end = 4 + 32 * count
    if count != len(recipients) or not 1 <= count <= _MAX_RECIPIENTS or key_end > len(data):
        raise ValueError("invalid mutable plaintext")
    keys = tuple(data[4 + 32 * i : 36 + 32 * i] for i in range(count))
    digests = tuple(hashlib.sha256(key).digest() for key in keys)
    if digests != recipients or any(left >= right for left, right in pairwise(digests)):
        raise ValueError("recipient key mismatch")
    return _read_mutable_frame(data[key_end:])


def _example_suffix(
    profile: int, keys: tuple[bytes, ...], payload: bytes
) -> tuple[bytes, tuple[bytes, ...], bytes]:
    recipients = tuple(hashlib.sha256(key).digest() for key in keys)
    _recipient_vector(recipients)
    inner_frame = _frame_for_test(payload, 16 + len(payload))
    plaintext = struct.pack("<I", len(keys)) + b"".join(keys) + inner_frame
    header = _header(profile, len(keys), len(plaintext) + 16)
    slots = tuple((bytes([index]) * 32, bytes([index + 10]) * 48) for index in range(len(keys)))
    frame = header + _slot_vector(slots)[4:] + bytes(len(plaintext) + 16)
    return frame + struct.pack("<4sQI", b"DMUE", len(frame), 0), recipients, plaintext


def _parse_suffix(
    data: bytes, boundaries: tuple[int, ...], profile: int, recipients: tuple[bytes, ...]
) -> tuple[int, bytes, tuple[tuple[bytes, bytes], ...]]:
    if len(data) < _HEADER_SIZE + _FOOTER_SIZE:
        raise ValueError("short encrypted suffix")
    magic, frame_length, reserved = struct.unpack("<4sQI", data[-_FOOTER_SIZE:])
    if magic != b"DMUE" or reserved or frame_length > len(data) - _FOOTER_SIZE:
        raise ValueError("invalid encrypted footer")
    start = len(data) - _FOOTER_SIZE - frame_length
    if start not in boundaries or frame_length < _HEADER_SIZE:
        raise ValueError("invalid retained boundary")
    frame = data[start:-_FOOTER_SIZE]
    magic, actual_profile, reserved_bytes, count, ciphertext_length = struct.unpack(
        "<4sB3sIQ", frame[:_HEADER_SIZE]
    )
    if (
        magic != b"DME1"
        or actual_profile != profile
        or reserved_bytes != bytes(3)
        or count != len(recipients)
        or _header(actual_profile, count, ciphertext_length) != frame[:_HEADER_SIZE]
        or len(frame) != _HEADER_SIZE + 80 * count + ciphertext_length
    ):
        raise ValueError("invalid encrypted mutable frame")
    slots = tuple(
        (
            frame[_HEADER_SIZE + 80 * i : _HEADER_SIZE + 80 * i + 32],
            frame[_HEADER_SIZE + 80 * i + 32 : _HEADER_SIZE + 80 * (i + 1)],
        )
        for i in range(count)
    )
    return start, frame[:_HEADER_SIZE], slots


def _example_keys() -> tuple[bytes, ...]:
    return tuple(
        sorted(
            (b"\x09" + bytes(31), b"\x0a" + bytes(31)),
            key=lambda key: hashlib.sha256(key).digest(),
        )
    )


def test_mutable_encrypted_frame_has_exact_fields_and_keeps_keys_in_plaintext() -> None:
    keys = _example_keys()
    suffix, recipients, plaintext = _example_suffix(2, keys, b'{"preset":"clean"}')
    prefix = b"L" * 64 + b"F" * 64
    start, header, slots = _parse_suffix(prefix + suffix, (64, 128), 2, recipients)
    ciphertext_length = len(plaintext) + 16
    assert start == 128
    assert header == b"DME1\x02\x00\x00\x00\x02\x00\x00\x00" + struct.pack("<Q", ciphertext_length)
    assert len(slots) == 2
    assert _read_plaintext(plaintext, recipients) == {"preset": "clean"}
    assert all(key not in suffix for key in keys)
    assert struct.unpack("<4sQI", suffix[-16:]) == (b"DMUE", len(suffix) - 16, 0)


def test_mutable_context_binds_profile_issuance_recipients_and_slots() -> None:
    keys = _example_keys()
    suffix, recipients, _ = _example_suffix(1, keys, b"{}")
    _, header, slots = _parse_suffix(b"L" * 64 + suffix, (64,), 1, recipients)
    context = b"C" * 32
    digest = _wrap_digest(context, header, recipients)
    assert digest != _wrap_digest(b"D" * 32, header, recipients)
    assert digest != _wrap_digest(context, _header(2, 2, 102), recipients)
    replacement = tuple(sorted((recipients[0], hashlib.sha256(b"replacement").digest())))
    assert digest != _wrap_digest(context, header, replacement)
    info = _mutable_hpke_binding(b"DNAM-MUTABLE-HPKE-INFO-V1", digest, 0, recipients[0])
    assert info != _mutable_hpke_binding(b"DNAM-MUTABLE-HPKE-AAD-V1", digest, 0, recipients[0])
    assert info != _mutable_hpke_binding(b"DNAM-MUTABLE-HPKE-INFO-V1", digest, 1, recipients[0])
    assert info != _mutable_hpke_binding(b"DNAM-MUTABLE-HPKE-INFO-V1", digest, 0, recipients[1])
    assert _content_aad(digest, slots) != _content_aad(digest, tuple(reversed(slots)))


def test_stripping_preserves_the_suffix_and_growth_preserves_signed_prefix() -> None:
    keys = _example_keys()
    suffix, recipients, _ = _example_suffix(1, keys, b"{}")
    grown, _, _ = _example_suffix(1, keys, b'{"presets":["clean","drive"]}')
    lite, full = b"L" * 64, b"F" * 64
    original = lite + full + suffix
    stripped = lite + suffix
    updated = lite + full + grown
    _, original_header, original_slots = _parse_suffix(original, (64, 128), 1, recipients)
    _, stripped_header, stripped_slots = _parse_suffix(stripped, (64, 128), 1, recipients)
    _, grown_header, _ = _parse_suffix(updated, (64, 128), 1, recipients)
    assert stripped[64:] == original[128:]
    assert updated[:128] == original[:128]
    assert (original_header, original_slots) == (stripped_header, stripped_slots)
    context = b"C" * 32
    assert _wrap_digest(context, original_header, recipients) == _wrap_digest(
        context, stripped_header, recipients
    )
    assert _wrap_digest(context, original_header, recipients) != _wrap_digest(
        context, grown_header, recipients
    )


def test_decrypted_key_vector_must_match_signed_recipients() -> None:
    keys = _example_keys()
    _, recipients, plaintext = _example_suffix(2, keys, b"{}")
    assert _read_plaintext(plaintext, recipients) == {}
    with pytest.raises(ValueError, match="recipient key mismatch"):
        _read_plaintext(plaintext[:4] + b"".join(reversed(keys)) + plaintext[68:], recipients)
    with pytest.raises(ValueError, match="invalid mutable plaintext"):
        _read_plaintext(b"\x03\x00\x00\x00" + plaintext[4:], recipients)
    corrupt = bytearray(plaintext)
    corrupt[-1] ^= 1
    with pytest.raises(ValueError, match="mutable checksum mismatch"):
        _read_plaintext(bytes(corrupt), recipients)


def test_encrypted_suffix_rejects_bad_bounds_counts_and_reserved_fields() -> None:
    keys = _example_keys()
    suffix, recipients, _ = _example_suffix(2, keys, b"{}")
    prefix = b"L" * 64
    for offset, value in ((0, ord("X")), (4, 1), (5, 1), (8, 3)):
        corrupt = bytearray(suffix)
        corrupt[offset] = value
        with pytest.raises(ValueError, match="invalid encrypted mutable frame"):
            _parse_suffix(prefix + bytes(corrupt), (64,), 2, recipients)
    with pytest.raises(ValueError, match="invalid retained boundary"):
        _parse_suffix(prefix + suffix, (128,), 2, recipients)
    with pytest.raises(ValueError, match="invalid encrypted footer"):
        _parse_suffix(prefix + suffix[:-1] + b"\x01", (64,), 2, recipients)
    with pytest.raises(ValueError, match="invalid encrypted mutable header"):
        _header(2, _MAX_RECIPIENTS + 1, 100)
    with pytest.raises(ValueError, match="invalid encrypted mutable header"):
        _header(2, 1, 69)
    with pytest.raises(ValueError, match="invalid encrypted mutable header"):
        _header(2, 1, _MAX_CIPHERTEXT + 1)
