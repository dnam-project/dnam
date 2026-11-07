"""`.dnam-seal` (`DNSL`) hardware envelope sealing/unsealing, `X25519` CEK wrapping, and `ChaCha20`
arena encryption (`ref[sdk.python.provenance.v2-security]`).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from dnam._native import (
    native_chacha20_xor_arena,
    native_seal_container,
    native_unseal_container,
    native_wrap_device_cek_hex,
)
from dnam.model._model import Model
from dnam.schema.signatures import OemDeviceCertificate


def chacha20_xor_arena(
    arena_bytes: bytes | bytearray | memoryview,
    key_32: bytes,
    nonce_12: bytes,
    *,
    start_block_counter: int = 0,
) -> bytes:
    """
    Applies RFC 8439 `ChaCha20` 64-byte cache-line XOR encryption/decryption over `arena_bytes`.
    """
    return native_chacha20_xor_arena(
        bytes(arena_bytes),
        key_32,
        nonce_12,
        start_block_counter,
    )


def wrap_device_cek(
    ephemeral_secret_32: bytes,
    device_pubkey_32: bytes,
    device_id_hash: str,
    cek_32: bytes,
) -> str:
    """Wraps a 32-byte Content Encryption Key (`cek_32`) for an OEM device (`X25519` +
    `HKDF-SHA256` + `ChaCha20`).
    """
    return native_wrap_device_cek_hex(
        ephemeral_secret_32,
        device_pubkey_32,
        device_id_hash,
        cek_32,
    )


def seal_model(
    inner_model: Model | bytes,
    *,
    recipient_certificates: Sequence[OemDeviceCertificate],
    trusted_oem_pubkeys: Mapping[str, str],
    marketplace_seed_32: bytes,
    ephemeral_secret_32: bytes,
    cek_32: bytes,
    nonce_12: bytes,
) -> bytes:
    """Seals an inner `.dnam` `Model` into a 64-byte aligned `.dnam-seal` (`DNSL`) hardware-bound
    envelope.
    """
    inner_bytes = inner_model.to_bytes() if isinstance(inner_model, Model) else inner_model
    certs_json = json.dumps([c.to_dict() for c in recipient_certificates])
    oem_json = json.dumps(dict(trusted_oem_pubkeys))
    return native_seal_container(
        inner_bytes,
        certs_json,
        oem_json,
        marketplace_seed_32,
        ephemeral_secret_32,
        cek_32,
        nonce_12,
    )


def unseal_model(
    sealed_bytes: bytes | bytearray | memoryview,
    *,
    device_id_hash: str,
    device_secret_32: bytes,
    trusted_marketplace_pubkey_hex: str | None = None,
) -> Model:
    """Unseals a `.dnam-seal` (`DNSL`) hardware envelope and returns the decrypted inner `Model`."""
    inner_bytes = native_unseal_container(
        bytes(sealed_bytes),
        device_id_hash,
        device_secret_32,
        trusted_marketplace_pubkey_hex,
    )
    return Model.from_bytes(
        inner_bytes,
        device_id_hash=device_id_hash,
        device_secret=device_secret_32,
        trusted_marketplace_pubkey_hex=trusted_marketplace_pubkey_hex,
    )
