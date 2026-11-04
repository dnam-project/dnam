"""Provenance verification, distribution policy enforcement, and anti-cloning gate evaluation
(`ref[sdk.python.provenance.v2-security]`).
"""

from __future__ import annotations

from enum import StrEnum

from dnam._native import (
    native_enforce_distribution_policy,
    native_evaluate_anticlone_policy,
    native_verify_container_provenance,
)
from dnam.model._model import Model
from dnam.schema._base import StrictModel, VerificationStatus
from dnam.schema.header import DnamJsonHeader
from dnam.schema.signatures import DnamProtectionProfile


class AnticloneAction(StrEnum):
    """Decision outcome of the Stage B anti-cloning ingestion gate
    (`ref[sec.anticlone.three-stage-gate]`).
    """

    NO_MATCH_PROCEED = "no_match_proceed"
    FLAG_LICENSE_HOLD_FOR_REVIEW = "flag_license_hold_for_review"
    REVOKE_IDENTIFIED_DEVICE_FUTURE_LICENSING_ONLY = (
        "revoke_identified_device_future_licensing_only"
    )


class VerificationReport(StrictModel):
    """Structured provenance verification result returned by `verify_model`."""

    status: VerificationStatus
    signer_id: str | None = None
    pubkey_ed25519: str | None = None
    marketplace_id: str | None = None
    license_id: str | None = None
    license_tier: str | None = None

    @property
    def is_verified(self) -> bool:
        """`True` if either creator or marketplace signature was verified."""
        return self.status != VerificationStatus.UNSIGNED


def verify_model(
    model: Model | DnamJsonHeader,
    *,
    plaintext_arena: bytes | None = None,
    has_signatures_flag: bool | None = None,
    device_id_hash: str | None = None,
    device_secret: bytes | None = None,
    trusted_marketplace_pubkey_hex: str | None = None,
) -> VerificationReport:
    """Verifies cryptographic provenance (`immutable_header_sha256`, `creator_signature`,
    `marketplace_receipt`).
    """
    if isinstance(model, Model):
        hdr_json = model.header.to_json()
        arena = plaintext_arena if plaintext_arena is not None else model.arena_bytes
        sig_flag = (
            has_signatures_flag
            if has_signatures_flag is not None
            else model.preamble.has_signatures
        )
    else:
        hdr_json = model.to_json()
        arena = plaintext_arena if plaintext_arena is not None else b""
        sig_flag = (
            has_signatures_flag
            if has_signatures_flag is not None
            else (model.metadata.signatures is not None)
        )

    raw = native_verify_container_provenance(
        hdr_json,
        arena,
        sig_flag,
        device_id_hash,
        device_secret,
        trusted_marketplace_pubkey_hex,
    )
    return VerificationReport._construct_trusted(
        status=VerificationStatus(raw["status"]),
        signer_id=raw["signer_id"],
        pubkey_ed25519=raw["pubkey_ed25519"],
        marketplace_id=raw["marketplace_id"],
        license_id=raw["license_id"],
        license_tier=raw["license_tier"],
    )


def enforce_distribution_policy(
    header: DnamJsonHeader | Model,
    active_profile: DnamProtectionProfile | str,
) -> None:
    """Enforces the creator-signed `DistributionPolicy` against `active_profile`."""
    hdr = header.header if isinstance(header, Model) else header
    prof = DnamProtectionProfile(active_profile)
    native_enforce_distribution_policy(hdr.to_json(), prof.value)


def evaluate_anticlone_policy(
    *,
    fingerprint_matched: bool,
    buyer_watermark_identified: bool,
    independent_device_source_evidence: bool,
) -> AnticloneAction:
    """Evaluates the 3-stage anti-cloning revocation policy."""
    res = native_evaluate_anticlone_policy(
        fingerprint_matched,
        buyer_watermark_identified,
        independent_device_source_evidence,
    )
    return AnticloneAction(res)
