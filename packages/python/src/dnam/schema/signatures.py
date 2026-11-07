"""Cryptographic provenance, protection profiles, OEM attestation, marketplace receipts, and
watermarking schemas (`ref[sdk.python.schema.rust-parity]`).
"""

from __future__ import annotations

from dataclasses import field
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from dnam._native import (
    native_is_oem_device_allowed,
    native_is_profile_allowed,
)
from dnam.schema._base import StrictModel, strict_field


def _empty_f32() -> NDArray[np.float32]:
    return np.empty(0, dtype=np.float32)


class DnamProtectionProfile(StrEnum):
    """Canonical container cryptographic protection profile (`'open'`, `'device_allowlisted'`,
    `'enclave_sealed'`).
    """

    OPEN = "open"
    DEVICE_ALLOWLISTED = "device_allowlisted"
    ENCLAVE_SEALED = "enclave_sealed"


class DistributionPolicy(StrictModel):
    """Creator-signed distribution and hardware exclusivity policy
    (`metadata.distribution_policy`).
    """

    allowed_protection_profiles: list[DnamProtectionProfile] = strict_field(default_factory=list)
    allowed_oem_vendors: list[str] = strict_field(default_factory=list)

    def is_profile_allowed(self, profile: DnamProtectionProfile | str) -> bool:
        """Returns `True` if `profile` is permitted by `allowed_protection_profiles`."""
        prof = DnamProtectionProfile(profile)
        return native_is_profile_allowed(
            [p.value for p in self.allowed_protection_profiles],
            prof.value,
        )

    def is_oem_device_allowed(self, oem_vendor: str, model_code: str) -> bool:
        """Returns `True` if `(oem_vendor, model_code)` is permitted by `allowed_oem_vendors`."""
        return native_is_oem_device_allowed(
            list(self.allowed_oem_vendors),
            oem_vendor,
            model_code,
        )


class OemDeviceCertificate(StrictModel):
    """Factory-provisioned OEM hardware attestation certificate."""

    oem_vendor: str
    model_code: str
    device_id_hash: str
    pubkey_x25519: str
    oem_sig_ed25519: str


class DatasetProvenance(StrictModel):
    """Content-addressable dataset and recipe provenance."""

    dataset_manifest_sha256: str = ""
    config_recipe_sha256: str | None = None
    epochs_trained: int = strict_field(default=0, ge=0)


class CreatorSignature(StrictModel):
    """Creator's `Ed25519` signature block."""

    signer_id: str
    pubkey_ed25519: str
    sig_ed25519: str


class AuthorizedDevice(StrictModel):
    """Authorized hardware device record inside `marketplace_receipt.authorized_devices`."""

    oem_vendor: str
    model_code: str
    device_id_hash: str
    wrapped_cek: str | None = None


class RoyaltySplit(StrictModel):
    """Multi-creator royalty split entry (`10000 bps = 100.00%`)."""

    recipient_id: str
    role: str
    share_bps: int = strict_field(ge=1, le=10000)


class MarketplaceReceipt(StrictModel):
    """
    Marketplace download receipt (`< 0.1 ms` edge personalization & OEM hardware authorization).
    """

    marketplace_id: str
    license_id: str
    buyer_hash: str
    license_tier: str
    issued_at: str
    buyer_arena_sha256: str
    buyer_tier_arena_sha256: tuple[str, str, str, str] | None = None
    encryption_mode: str = "none"
    arena_nonce_96: str | None = None
    ephemeral_pubkey_x25519: str | None = None
    authorized_devices: list[AuthorizedDevice] = strict_field(default_factory=list)
    royalty_splits: list[RoyaltySplit] = strict_field(default_factory=list)
    receipt_sig_ed25519: str


class ModelSignatures(StrictModel):
    """Split-payload `Ed25519` signature envelope (`HAS_SIGNATURES` bit 10)."""

    immutable_header_sha256: str
    master_arena_sha256: str
    tier_arena_sha256: tuple[str, str, str, str] | None = None
    creator_signature: CreatorSignature
    marketplace_receipt: MarketplaceReceipt | None = None


class WatermarkParams(StrictModel):
    """Parameters for Dual-Mechanism Buyer Watermarking (Isometric Gauge + Coupled Tardos-7680)."""

    kappa_64: tuple[float, ...] = strict_field(
        default_factory=lambda: tuple(1.0 for _ in range(64)),
        min_length=64,
        max_length=64,
    )
    q_buyer_4x4: tuple[tuple[float, float, float, float], ...] = strict_field(
        default=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        min_length=4,
        max_length=4,
    )
    tardos_epsilon: float = strict_field(default=1.5e-3, gt=0.0, le=0.05)

    _kappa_arr: NDArray[np.float32] = field(
        default_factory=_empty_f32, init=False, repr=False, compare=False
    )
    _q_flat_arr: NDArray[np.float32] = field(
        default_factory=_empty_f32, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self,
            "_kappa_arr",
            np.asarray(self.kappa_64, dtype=np.float32, order="C"),
        )
        object.__setattr__(
            self,
            "_q_flat_arr",
            np.asarray(self.q_buyer_4x4, dtype=np.float32, order="C").reshape(16),
        )

    def cached_f32_arrays(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Returns pre-allocated C-contiguous `(kappa_64, q_buyer_flat_16)` `float32` arrays."""
        return self._kappa_arr, self._q_flat_arr
