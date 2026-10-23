"""Cryptographic provenance (`DatasetProvenance`), `Ed25519` signatures (`ModelSignatures`),
marketplace receipts (`MarketplaceReceipt`), and buyer watermarking (`WatermarkParams`)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, PrivateAttr, field_validator

from dnam_engine.exceptions import DnamValidationError
from dnam_engine.schema.base import StrictModel


class DatasetProvenance(StrictModel):
    """Content-addressable training dataset and recipe provenance."""

    dataset_manifest_sha256: str = ""
    config_recipe_sha256: str | None = None
    epochs_trained: int = Field(default=0, ge=0)


class CreatorSignature(StrictModel):
    """Creator's offline `Ed25519` signature block."""

    signer_id: str = Field(min_length=1)
    pubkey_ed25519: str = Field(min_length=64, max_length=64)
    sig_ed25519: str = Field(min_length=128, max_length=128)


class AuthorizedDevice(StrictModel):
    """Authorized OEM hardware device record inside `marketplace_receipt.authorized_devices`."""

    oem_vendor: str = Field(min_length=1)
    model_code: str = Field(min_length=1)
    device_id_hash: str = Field(min_length=1)
    wrapped_cek: str | None = None


class RoyaltySplit(StrictModel):
    """Multi-creator royalty split entry (`10000 bps = 100.00%`)."""

    recipient_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    share_bps: int = Field(ge=0, le=10_000)


class MarketplaceReceipt(StrictModel):
    """Marketplace download receipt (`< 0.1 ms` edge personalization & OEM authorization)."""

    marketplace_id: str = Field(min_length=1)
    license_id: str = Field(min_length=1)
    buyer_hash: str = Field(min_length=64, max_length=64)
    license_tier: str = Field(min_length=1)
    issued_at: str = Field(min_length=1)
    buyer_arena_sha256: str = Field(min_length=64, max_length=64)
    encryption_mode: str = "none"
    arena_nonce_96: str | None = None
    ephemeral_pubkey_x25519: str | None = None
    authorized_devices: list[AuthorizedDevice] = Field(default_factory=list)
    royalty_splits: list[RoyaltySplit] = Field(default_factory=list)
    receipt_sig_ed25519: str = Field(min_length=128, max_length=128)


class ModelSignatures(StrictModel):
    """Split-payload `Ed25519` signature envelope (`HAS_SIGNATURES` bit 9)."""

    immutable_header_sha256: str = Field(min_length=64, max_length=64)
    master_arena_sha256: str = Field(min_length=64, max_length=64)
    creator_signature: CreatorSignature
    marketplace_receipt: MarketplaceReceipt | None = None


class WatermarkParams(StrictModel):
    """Validated parameters for Dual-Mechanism Buyer Watermarking (`Test 8H`)."""

    kappa_48: tuple[float, ...] = Field(default_factory=lambda: (1.0,) * 48)
    q_buyer_4x4: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ] = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    tardos_code_4608: tuple[float, ...] | None = None
    tardos_epsilon: float = Field(default=1e-4, ge=0.0, le=1e-2)

    _f32_bundle: tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32] | None] = (
        PrivateAttr()
    )

    def model_post_init(self, __context: Any) -> None:
        kappa_f32 = np.ascontiguousarray(self.kappa_48, dtype=np.float32).ravel()
        q_flat_f32 = np.ascontiguousarray(self.q_buyer_4x4, dtype=np.float32).ravel()
        tardos_f32 = (
            np.ascontiguousarray(self.tardos_code_4608, dtype=np.float32).ravel()
            if self.tardos_code_4608 is not None
            else None
        )
        self._f32_bundle = (kappa_f32, q_flat_f32, tardos_f32)

    def cached_f32_arrays(
        self,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32] | None]:
        """Returns cached C-contiguous `float32` `(kappa_48, q_buyer_16, tardos_4608)` arrays."""
        priv: dict[str, Any] = object.__getattribute__(self, "__pydantic_private__")
        return priv["_f32_bundle"]

    @field_validator("kappa_48")
    @classmethod
    def _validate_kappa(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        if len(v) != 48:
            raise DnamValidationError(f"kappa_48 must have 48 elements, got {len(v)}")
        for idx, k in enumerate(v):
            if not (0.985 <= k <= 1.015):
                raise DnamValidationError(f"kappa_48[{idx}] = {k} must lie within [0.985, 1.015]")
        return v

    @field_validator("tardos_code_4608")
    @classmethod
    def _validate_tardos_code(cls, v: tuple[float, ...] | None) -> tuple[float, ...] | None:
        if v is not None and len(v) != 4608:
            raise DnamValidationError(f"tardos_code_4608 must have 4608 elements, got {len(v)}")
        return v
