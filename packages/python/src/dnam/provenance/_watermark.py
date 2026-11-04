"""Dual-Mechanism Buyer Watermarking (Isometric Gauge + Coupled Tardos-7680) and 224-D Structural
Hashing (`ref[sdk.python.provenance.v2-security]`).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dnam._arrays import coerce_f64_1d
from dnam._native import (
    native_apply_isometric_gauge_watermark,
    native_apply_tardos_watermark,
    native_derive_watermark_seed_hex,
    native_extract_stage3_fingerprint,
    native_extract_structural_hash,
    native_generate_tardos_code,
    native_score_tardos_watermark,
    native_structural_cosine_similarity,
)
from dnam.schema.signatures import WatermarkParams


def apply_gauge_watermark(
    arena_bytes: bytes | bytearray | memoryview,
    params: WatermarkParams | None = None,
    *,
    kappa_64: ArrayLike | None = None,
    q_buyer_4x4: ArrayLike | None = None,
) -> bytes:
    """Applies Mechanism A (64-channel modal/Kautz reciprocal scaling + SO(4) SVD head rotation) to
    `arena_bytes`.
    """
    if params is not None:
        k_arr, q_flat = params.cached_f32_arrays()
    else:
        k_arr = (
            np.asarray(kappa_64, dtype=np.float32, order="C").reshape(-1)
            if kappa_64 is not None
            else np.ones(64, dtype=np.float32)
        )
        q_flat = (
            np.asarray(q_buyer_4x4, dtype=np.float32, order="C").reshape(-1)
            if q_buyer_4x4 is not None
            else np.eye(4, dtype=np.float32).reshape(16)
        )
    return native_apply_isometric_gauge_watermark(
        bytes(arena_bytes),
        k_arr,
        q_flat,
    )


def generate_tardos_code(
    marketplace_secret_32: bytes,
    model_id: str,
    license_id: str,
) -> NDArray[np.float32]:
    """Generates a 7,680-chip Tardos binary spread-spectrum watermark code `{-1.0, +1.0}^7680`."""
    return native_generate_tardos_code(
        marketplace_secret_32,
        model_id,
        license_id,
    )


def apply_tardos_watermark(
    arena_bytes: bytes | bytearray | memoryview,
    code_7680: ArrayLike,
    *,
    epsilon: float = 1.5e-3,
) -> bytes:
    """Embeds Mechanism B (7,680-weight Tardos spread-spectrum watermark) into `arena_bytes`."""
    code_arr = np.asarray(code_7680, dtype=np.float32, order="C").reshape(-1)
    return native_apply_tardos_watermark(
        bytes(arena_bytes),
        code_arr,
        float(epsilon),
    )


def score_tardos_watermark(
    master_arena: bytes | bytearray | memoryview,
    suspect_arena: bytes | bytearray | memoryview,
    code_7680: ArrayLike,
) -> float:
    """Computes the Tardos matched-filter Z-score (`sigma`) of `suspect_arena` against
    `master_arena` and `code_7680`.
    """
    code_arr = np.asarray(code_7680, dtype=np.float32, order="C").reshape(-1)
    return float(
        native_score_tardos_watermark(
            bytes(master_arena),
            bytes(suspect_arena),
            code_arr,
        )
    )


def extract_structural_hash(
    arena_bytes: bytes | bytearray | memoryview,
) -> NDArray[np.float64]:
    """Extracts the prefix-composable `4 x 56-D = 224-D` Gauge-Invariant Structural Hash vector
    from `arena_bytes`.
    """
    return native_extract_structural_hash(bytes(arena_bytes))


def structural_cosine_similarity(
    hash_a: ArrayLike,
    hash_b: ArrayLike,
) -> float:
    """
    Computes the cosine similarity `cos(a, b) in [-1, +1]` between two structural hash vectors.
    """
    a_arr = coerce_f64_1d(hash_a, name="hash_a")
    b_arr = coerce_f64_1d(hash_b, name="hash_b")
    return float(native_structural_cosine_similarity(a_arr, b_arr))


def derive_watermark_seed(
    key_32: bytes,
    *,
    marketplace_pubkey_32: bytes,
    watermark_key_id_16: bytes,
    creator_signed_manifest_sha256_32: bytes,
    model_id_16: bytes,
    watermark_profile_sha256_32: bytes,
    owning_tier: int,
    license_id_16: bytes | None = None,
    buyer_id_16: bytes | None = None,
) -> str:
    """Derives the 64-character hex shared or buyer-scoped HMAC-SHA256 watermark seed."""
    return native_derive_watermark_seed_hex(
        key_32,
        marketplace_pubkey_32,
        watermark_key_id_16,
        creator_signed_manifest_sha256_32,
        model_id_16,
        watermark_profile_sha256_32,
        owning_tier,
        license_id_16,
        buyer_id_16,
    )


def extract_stage3_fingerprint(
    early_fir_128b: bytes | bytearray | memoryview,
    kautz_modal_bytes: bytes | bytearray | memoryview,
    added_modes: int,
) -> NDArray[np.float64]:
    """Extracts the gauge-invariant Stage III tier-local fingerprint component."""
    return native_extract_stage3_fingerprint(
        bytes(early_fir_128b),
        bytes(kautz_modal_bytes),
        added_modes,
    )
