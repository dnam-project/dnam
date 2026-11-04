"""`Ed25519` / `X25519` keys, RFC 8785 JCS header hashing, creator/marketplace signing, and OEM
attestation (`ref[sdk.python.provenance.v2-security]`).
"""

from __future__ import annotations

from dnam._native import (
    native_build_creator_signing_message,
    native_build_marketplace_signing_message,
    native_build_oem_cert_v2_signing_message,
    native_ed25519_pubkey_hex,
    native_ed25519_sign_hex,
    native_hmac_sha256_hex,
    native_immutable_header_sha256,
    native_sha256_hex,
    native_sign_oem_device_cert,
    native_verify_oem_device_cert,
    native_x25519_pubkey_hex,
)
from dnam.schema.header import DnamJsonHeader
from dnam.schema.signatures import (
    AuthorizedDevice,
    CreatorSignature,
    MarketplaceReceipt,
    ModelSignatures,
    OemDeviceCertificate,
    RoyaltySplit,
)


def immutable_header_sha256(header: DnamJsonHeader | str) -> str:
    """Computes the preset-safe RFC 8785 JCS `immutable_header_sha256` hex digest."""
    raw = header.to_json() if isinstance(header, DnamJsonHeader) else header
    return native_immutable_header_sha256(raw)


def sha256_hex(data: bytes | bytearray | memoryview) -> str:
    """Computes the lowercase 64-character hex SHA-256 digest of `data`."""
    return native_sha256_hex(bytes(data))


def hmac_sha256_hex(key_32: bytes, message: bytes) -> str:
    """Computes the lowercase 64-character hex HMAC-SHA256 digest of `message` under `key_32`."""
    return native_hmac_sha256_hex(key_32, message)


def ed25519_public_key(seed_32: bytes) -> str:
    """
    Derives the 64-character lowercase hex `Ed25519` public key from a 32-byte secret key seed.
    """
    return native_ed25519_pubkey_hex(seed_32)


def ed25519_sign(seed_32: bytes, message: bytes) -> str:
    """Signs `message` with a 32-byte `Ed25519` secret key seed, returning a 128-char hex
    signature.
    """
    return native_ed25519_sign_hex(seed_32, message)


def x25519_public_key(secret_32: bytes) -> str:
    """Derives the 64-character lowercase hex `X25519` public key from a 32-byte secret key."""
    return native_x25519_pubkey_hex(secret_32)


def build_creator_signing_message(
    immutable_header_sha256_hex: str,
    master_arena_sha256_hex: str,
) -> bytes:
    """Builds the canonical 64-byte `Ed25519` creator signing message (`immutable_header_sha256 ||
    master_arena_sha256`).
    """
    return native_build_creator_signing_message(
        immutable_header_sha256_hex,
        master_arena_sha256_hex,
    )


def build_marketplace_signing_message(
    immutable_header_sha256_hex: str,
    master_arena_sha256_hex: str,
    receipt: MarketplaceReceipt | str,
) -> bytes:
    """Builds the canonical `Ed25519` marketplace receipt signing message."""
    rcpt_json = receipt.model_dump_json() if isinstance(receipt, MarketplaceReceipt) else receipt
    return native_build_marketplace_signing_message(
        immutable_header_sha256_hex,
        master_arena_sha256_hex,
        rcpt_json,
    )


def _compute_tier_arena_digests(
    arena_bytes: bytes,
) -> tuple[str, str, str, str] | None:
    offsets = (10_752, 22_336, 36_224, 64_000)
    if len(arena_bytes) < 64_000:
        return None
    d0 = sha256_hex(arena_bytes[: offsets[0]])
    d1 = sha256_hex(arena_bytes[: offsets[1]])
    d2 = sha256_hex(arena_bytes[: offsets[2]])
    d3 = sha256_hex(arena_bytes[: offsets[3]])
    return (d0, d1, d2, d3)


def sign_creator_header(
    header: DnamJsonHeader,
    master_arena_bytes: bytes,
    *,
    creator_seed_32: bytes,
    signer_id: str,
) -> DnamJsonHeader:
    """Signs `header` and `master_arena_bytes` with `creator_seed_32`, returning an updated
    `DnamJsonHeader`.
    """
    hdr_sha = immutable_header_sha256(header)
    master_sha = sha256_hex(master_arena_bytes)
    tier_shas = _compute_tier_arena_digests(master_arena_bytes)
    pubkey_hex = ed25519_public_key(creator_seed_32)
    msg = build_creator_signing_message(hdr_sha, master_sha)
    sig_hex = ed25519_sign(creator_seed_32, msg)

    sigs = ModelSignatures(
        immutable_header_sha256=hdr_sha,
        master_arena_sha256=master_sha,
        tier_arena_sha256=tier_shas,
        creator_signature=CreatorSignature(
            signer_id=signer_id,
            pubkey_ed25519=pubkey_hex,
            sig_ed25519=sig_hex,
        ),
        marketplace_receipt=None,
    )
    updated_meta = header.metadata.model_copy(update={"signatures": sigs})
    return header.model_copy(update={"metadata": updated_meta})


def sign_marketplace_receipt(
    signed_header: DnamJsonHeader,
    buyer_arena_bytes: bytes,
    *,
    marketplace_seed_32: bytes,
    marketplace_id: str,
    license_id: str,
    buyer_hash: str,
    license_tier: str = "personal",
    issued_at: str = "2026-11-04T00:00:00Z",
    encryption_mode: str = "none",
    arena_nonce_96: str | None = None,
    ephemeral_pubkey_x25519: str | None = None,
    authorized_devices: list[AuthorizedDevice] | None = None,
    royalty_splits: list[RoyaltySplit] | None = None,
) -> DnamJsonHeader:
    """Counter-signs `signed_header` and `buyer_arena_bytes` with a `MarketplaceReceipt`."""
    sigs = signed_header.metadata.signatures
    if sigs is None:
        hdr_sha = immutable_header_sha256(signed_header)
        master_sha = sha256_hex(buyer_arena_bytes)
    else:
        hdr_sha = sigs.immutable_header_sha256
        master_sha = sigs.master_arena_sha256

    buyer_sha = sha256_hex(buyer_arena_bytes)
    buyer_tier_shas = _compute_tier_arena_digests(buyer_arena_bytes)

    draft_receipt = MarketplaceReceipt(
        marketplace_id=marketplace_id,
        license_id=license_id,
        buyer_hash=buyer_hash,
        license_tier=license_tier,
        issued_at=issued_at,
        buyer_arena_sha256=buyer_sha,
        buyer_tier_arena_sha256=buyer_tier_shas,
        encryption_mode=encryption_mode,
        arena_nonce_96=arena_nonce_96,
        ephemeral_pubkey_x25519=ephemeral_pubkey_x25519,
        authorized_devices=authorized_devices or [],
        royalty_splits=royalty_splits or [],
        receipt_sig_ed25519="00" * 64,
    )
    msg = build_marketplace_signing_message(hdr_sha, master_sha, draft_receipt)
    receipt_sig = ed25519_sign(marketplace_seed_32, msg)
    final_receipt = draft_receipt.model_copy(update={"receipt_sig_ed25519": receipt_sig})

    if sigs is None:
        raise ValueError(
            "signed_header.metadata.signatures must be present before marketplace counter-signing"
        )
    updated_sigs = sigs.model_copy(update={"marketplace_receipt": final_receipt})
    updated_meta = signed_header.metadata.model_copy(update={"signatures": updated_sigs})
    return signed_header.model_copy(update={"metadata": updated_meta})


def sign_oem_certificate(
    oem_seed_32: bytes,
    *,
    oem_vendor: str,
    model_code: str,
    device_id_hash: str,
    pubkey_x25519_32: bytes,
) -> OemDeviceCertificate:
    """Signs a factory `OemDeviceCertificate` using `oem_seed_32`."""
    cert_json = native_sign_oem_device_cert(
        oem_seed_32,
        oem_vendor,
        model_code,
        device_id_hash,
        pubkey_x25519_32,
    )
    return OemDeviceCertificate.model_validate_json(cert_json)


def verify_oem_certificate(
    certificate: OemDeviceCertificate | str,
    trusted_oem_pubkey_hex: str,
) -> str:
    """Verifies `certificate` against `trusted_oem_pubkey_hex`, returning `pubkey_x25519` hex."""
    cert_json = (
        certificate.model_dump_json()
        if isinstance(certificate, OemDeviceCertificate)
        else certificate
    )
    return native_verify_oem_device_cert(
        cert_json,
        trusted_oem_pubkey_hex,
    )


def build_oem_cert_v2_signing_message(
    *,
    oem_vendor: str,
    model_code: str,
    device_id_hash_16: bytes,
    pubkey_x25519_32: bytes,
    issuer_pubkey_ed25519_32: bytes,
    assurance: int = 1,
    attestation_policy_sha256_32: bytes | None = None,
    attestation_evidence_sha256_32: bytes | None = None,
) -> bytes:
    """Builds the domain-separated V2 `OemDeviceCertificate` signing message."""
    return native_build_oem_cert_v2_signing_message(
        oem_vendor,
        model_code,
        device_id_hash_16,
        pubkey_x25519_32,
        issuer_pubkey_ed25519_32,
        assurance,
        attestation_policy_sha256_32,
        attestation_evidence_sha256_32,
    )
