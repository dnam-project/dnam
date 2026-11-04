"""Marketplace v2 envelopes (`DAU1`, `DME1`, HPKE slots), record directories, royalty splits, and
preview containment (`ref[sdk.python.provenance.v2-security]`).
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from pydantic import Field

from dnam._native import (
    native_build_dme1_header,
    native_compute_preview_window,
    native_decode_dau1_envelope,
    native_encode_dau1_envelope,
    native_encode_hpke_slots,
    native_encode_recipient_key_ids,
    native_encode_record_directory,
    native_parse_dme1_plaintext,
    native_parse_hpke_slots,
    native_parse_recipient_key_ids,
    native_parse_record_directory,
    native_validate_preview_switch_options,
    native_validate_royalty_splits,
)
from dnam.schema._base import StrictModel
from dnam.schema.signatures import RoyaltySplit


class Dau1Manifest(StrictModel):
    """Open `DAU1` Authentication Envelope (`ref[sec.sig.dau1-envelope]`)."""

    auth_kind: int = Field(ge=0, le=1)
    creator_pubkey_hex: str
    creator_max_tier: int = Field(ge=0, le=3)
    master_prefixes_hex: list[str]
    creator_signature_hex: str
    marketplace_pubkey_hex: str | None = None
    policy_bytes: bytes = b""
    buyer_max_tier: int | None = Field(default=None, ge=0, le=3)
    buyer_prefixes_hex: list[str] = Field(default_factory=list)
    marketplace_signature_hex: str | None = None

    def to_bytes(self) -> bytes:
        """Serializes this `Dau1Manifest` into canonical `DAU1` wire bytes."""
        return native_encode_dau1_envelope(
            self.auth_kind,
            self.creator_pubkey_hex,
            self.creator_max_tier,
            list(self.master_prefixes_hex),
            self.creator_signature_hex,
            self.marketplace_pubkey_hex,
            self.policy_bytes,
            self.buyer_max_tier,
            list(self.buyer_prefixes_hex),
            self.marketplace_signature_hex,
        )

    @classmethod
    def from_bytes(cls, wire_bytes: bytes | bytearray | memoryview) -> Dau1Manifest:
        """Parses and validates a `Dau1Manifest` from canonical `DAU1` wire bytes."""
        raw = native_decode_dau1_envelope(bytes(wire_bytes))
        return cls._construct_trusted(
            auth_kind=raw["auth_kind"],
            creator_pubkey_hex=raw["creator_pubkey_hex"],
            creator_max_tier=raw["creator_max_tier"],
            master_prefixes_hex=raw["master_prefixes_hex"],
            creator_signature_hex=raw["creator_signature_hex"],
            marketplace_pubkey_hex=raw["marketplace_pubkey_hex"],
            policy_bytes=raw["policy_bytes"],
            buyer_max_tier=raw["buyer_max_tier"],
            buyer_prefixes_hex=raw["buyer_prefixes_hex"],
            marketplace_signature_hex=raw["marketplace_signature_hex"],
        )


def build_dme1_header(
    profile: int,
    recipient_count: int,
    ciphertext_length: int,
) -> bytes:
    """Builds the 20-byte `DME1` encrypted mutable frame header
    (`ref[sec.drm.dme1-mutable-suffix]`).
    """
    return native_build_dme1_header(
        profile,
        recipient_count,
        ciphertext_length,
    )


def parse_dme1_plaintext(
    plaintext: bytes | bytearray | memoryview,
    recipient_key_digests_hex: Sequence[str],
) -> str:
    """Validates and parses decrypted `DME1` plaintext into canonical mutable JSON."""
    return native_parse_dme1_plaintext(
        bytes(plaintext),
        list(recipient_key_digests_hex),
    )


def encode_recipient_key_ids(key_digests_hex: Sequence[str]) -> bytes:
    """
    Encodes strictly sorted HPKE recipient key SHA-256 digests (`R = u32le(N) || digest[0..N]`).
    """
    return native_encode_recipient_key_ids(list(key_digests_hex))


def parse_recipient_key_ids(
    data: bytes | bytearray | memoryview,
    *,
    max_recipients: int = 16,
) -> list[str]:
    """Parses and validates HPKE recipient key SHA-256 digests."""
    return native_parse_recipient_key_ids(bytes(data), max_recipients)


def encode_hpke_slots(slots_hex: Sequence[tuple[str, str]]) -> bytes:
    """Encodes an HPKE slot vector (`S = u32le(N) || (enc[32] || ct_and_tag[48])[0..N]`)."""
    return native_encode_hpke_slots(list(slots_hex))


def parse_hpke_slots(
    data: bytes | bytearray | memoryview,
    expected_count: int,
    *,
    max_recipients: int = 16,
) -> list[tuple[str, str]]:
    """Parses and validates an HPKE slot vector."""
    return native_parse_hpke_slots(
        bytes(data),
        expected_count,
        max_recipients,
    )


def encode_record_directory(
    maximum_tier: int,
    records: Sequence[tuple[int, int]],
) -> bytes:
    """Encodes a 64B-aligned record directory (`D = u32le(C) || (offset, length)[0..C]`)."""
    return native_encode_record_directory(
        maximum_tier,
        list(records),
    )


def parse_record_directory(
    data: bytes | bytearray | memoryview,
    maximum_tier: int,
) -> list[tuple[int, int]]:
    """Parses and validates a 64B-aligned record directory."""
    return native_parse_record_directory(bytes(data), maximum_tier)


def compute_preview_window(
    anchor: float,
    delta: float,
    safe_bounds: tuple[float, float],
) -> tuple[float, float]:
    """Computes the bounded single-control preview window `[start, end]`
    (`ref[sec.preview.containment]`).
    """
    lo, hi = native_compute_preview_window(
        float(anchor),
        float(delta),
        (float(safe_bounds[0]), float(safe_bounds[1])),
    )
    return float(lo), float(hi)


def validate_preview_switch_options(
    preset: str,
    alternative: str,
    *,
    observed: Sequence[str],
    active: Sequence[str],
    safe: Sequence[str],
) -> tuple[str, str]:
    """Validates that a single-switch preview exposes two distinct options that are observed,
    active, and safe.
    """
    return native_validate_preview_switch_options(
        preset,
        alternative,
        list(observed),
        list(active),
        list(safe),
    )


def validate_royalty_splits(
    license_tier: str,
    splits: Sequence[RoyaltySplit],
) -> None:
    """Validates `license_tier` and that `splits` sum to `10,000` bps (`100%`)."""
    splits_json = json.dumps([s.model_dump(mode="json") for s in splits])
    native_validate_royalty_splits(license_tier, splits_json)
