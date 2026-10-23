"""Cryptographic provenance, RFC 8785 JCS SHA-256, `Ed25519` signing messages,
RFC 8439 `ChaCha20` 64-byte cache-line arena encryption, Dual-Mechanism Buyer Watermarking,
and 160-D Gauge-Invariant Structural Hashing."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dnam_engine import _native
from dnam_engine.exceptions import DnamValidationError, call_native
from dnam_engine.schema.header import DnamJsonHeader
from dnam_engine.schema.signatures import MarketplaceReceipt, WatermarkParams


def sha256_hex(data: bytes) -> str:
    """Computes the lowercase hex SHA-256 digest of `data`."""
    return call_native(_native.native_sha256_hex, data)


def immutable_header_sha256(header: DnamJsonHeader | str) -> str:
    """Computes the preset-safe RFC 8785 JCS `immutable_header_sha256` hex digest over `header`."""
    header_json = header if isinstance(header, str) else header.model_dump_json(by_alias=True)
    return call_native(_native.native_immutable_header_sha256, header_json)


def build_creator_signing_message(
    immutable_header_sha256_hex: str,
    master_arena_sha256_hex: str,
) -> bytes:
    """Builds the canonical `Ed25519` creator signing message for the header & arena SHA-256."""
    return call_native(
        _native.native_build_creator_signing_message,
        immutable_header_sha256_hex,
        master_arena_sha256_hex,
    )


def build_marketplace_signing_message(
    immutable_header_sha256_hex: str,
    master_arena_sha256_hex: str,
    receipt: MarketplaceReceipt | str,
) -> bytes:
    """Builds the canonical `Ed25519` marketplace receipt signing message."""
    receipt_json = receipt if isinstance(receipt, str) else receipt.model_dump_json()
    return call_native(
        _native.native_build_marketplace_signing_message,
        immutable_header_sha256_hex,
        master_arena_sha256_hex,
        receipt_json,
    )


def chacha20_xor_arena(
    arena_bytes: bytes,
    *,
    key_32: bytes,
    nonce_12: bytes,
    start_block_counter: int = 0,
) -> bytes:
    """Applies RFC 8439 `ChaCha20` 64-byte cache-line XOR encryption over `arena_bytes`."""
    return call_native(
        _native.native_chacha20_xor_arena,
        arena_bytes,
        key_32,
        nonce_12,
        start_block_counter,
    )


def _coerce_f32_flat(arr: ArrayLike) -> NDArray[np.float32]:
    if isinstance(arr, np.ndarray) and arr.dtype == np.float32 and arr.flags.c_contiguous:
        return arr.ravel()
    return np.ascontiguousarray(arr, dtype=np.float32).ravel()


def apply_isometric_gauge_watermark(
    arena_bytes: bytes,
    *,
    kappa_48: ArrayLike,
    q_buyer_4x4: ArrayLike,
) -> bytes:
    """Applies Mechanism A (48-channel RG-LRU scaling + SO(4) head rotation) to `arena_bytes`."""
    kappa_arr = _coerce_f32_flat(kappa_48)
    q_arr = _coerce_f32_flat(q_buyer_4x4)
    return call_native(
        _native.native_apply_isometric_gauge_watermark,
        arena_bytes,
        kappa_arr,
        q_arr,
    )


def generate_tardos_code(
    marketplace_secret_32: bytes,
    *,
    model_id: str,
    license_id: str,
) -> NDArray[np.float32]:
    """Deterministically generates a 4,608-chip Tardos binary watermark code `{-1.0, +1.0}^4608`."""
    return call_native(
        _native.native_generate_tardos_code,
        marketplace_secret_32,
        model_id,
        license_id,
    )


def apply_tardos_watermark(
    arena_bytes: bytes,
    *,
    code_4608: ArrayLike,
    epsilon: float = 1e-4,
) -> bytes:
    """Embeds Mechanism B (4,608-weight Tardos spread-spectrum watermark) into `arena_bytes`."""
    code_arr = _coerce_f32_flat(code_4608)
    return call_native(
        _native.native_apply_tardos_watermark,
        arena_bytes,
        code_arr,
        float(epsilon),
    )


def apply_dual_watermark(arena_bytes: bytes, params: WatermarkParams) -> bytes:
    """Applies Mechanism A (Isometric Gauge) and optional Mechanism B (Tardos) watermarks."""
    kappa_arr, q_flat_arr, tardos_arr = params.cached_f32_arrays()
    out = call_native(
        _native.native_apply_isometric_gauge_watermark,
        arena_bytes,
        kappa_arr,
        q_flat_arr,
    )
    if tardos_arr is not None:
        out = call_native(
            _native.native_apply_tardos_watermark,
            out,
            tardos_arr,
            params.tardos_epsilon,
        )
    return out


def score_tardos_watermark(
    master_arena: bytes,
    suspect_arena: bytes,
    *,
    code_4608: ArrayLike,
) -> float:
    """Computes the Tardos matched-filter Z-score (`sigma`) of `suspect_arena` vs `master_arena`."""
    code_arr = _coerce_f32_flat(code_4608)
    return float(
        call_native(
            _native.native_score_tardos_watermark,
            master_arena,
            suspect_arena,
            code_arr,
        )
    )


def extract_structural_hash(arena_bytes: bytes) -> NDArray[np.float64]:
    """Extracts the 160-D Gauge-Invariant Structural Hash `f_struct in R^160` from `arena_bytes`."""
    return call_native(_native.native_extract_structural_hash, arena_bytes)


def structural_cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """Computes cosine similarity `cos(a, b) in [-1, +1]` between two 160-D structural hashes."""
    a_arr = np.ascontiguousarray(a, dtype=np.float64)
    b_arr = np.ascontiguousarray(b, dtype=np.float64)
    if a_arr.ndim != 1 or b_arr.ndim != 1:
        raise DnamValidationError("Structural hash vectors must be 1D arrays of length 160")
    return float(call_native(_native.native_structural_cosine_similarity, a_arr, b_arr))
