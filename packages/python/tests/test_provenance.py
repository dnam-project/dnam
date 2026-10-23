"""Unit and property-based tests for `dnam_engine.provenance` (JCS SHA-256, `Ed25519` signing
messages, `ChaCha20` 64B cache-line arena encryption, Dual-Mechanism Buyer Watermarking `Test 8H`,
and 160-D structural hashing)."""

from __future__ import annotations

import numpy as np
import pytest

from dnam_engine import DnamContainer, DnamValidationError, provenance
from dnam_engine.schema import MarketplaceReceipt, PresetDef, WatermarkParams


def test_jcs_immutable_header_sha256_preset_invariance(
    ref_container_core2: DnamContainer,
) -> None:
    sha_orig = ref_container_core2.immutable_header_sha256()
    assert len(sha_orig) == 64

    # Adding a user preset or user_metadata must NOT change immutable_header_sha256
    hdr_modified = ref_container_core2.header.model_copy(
        update={
            "presets": [
                *ref_container_core2.header.presets,
                PresetDef(id="user_fav", name="User Fav", controls={"gain": 7.0}),
            ],
            "user_metadata": {"favorite": True},
        }
    )
    assert provenance.immutable_header_sha256(hdr_modified) == sha_orig
    assert (
        provenance.immutable_header_sha256(hdr_modified.model_dump_json(by_alias=True)) == sha_orig
    )


def test_signing_messages_and_chacha20_arena_encryption(
    ref_container_core2: DnamContainer,
) -> None:
    hdr_sha = ref_container_core2.immutable_header_sha256()
    arena_bytes = bytes(ref_container_core2.arena)
    arena_sha = provenance.sha256_hex(arena_bytes)
    assert len(arena_sha) == 64

    creator_msg = provenance.build_creator_signing_message(hdr_sha, arena_sha)
    assert len(creator_msg) == 64

    receipt = MarketplaceReceipt(
        marketplace_id="dnam.store",
        license_id="0192b8e0-1234-7000-8000-000000000001",
        buyer_hash="a" * 64,
        license_tier="personal",
        issued_at="2026-10-23T00:00:00Z",
        buyer_arena_sha256=arena_sha,
        receipt_sig_ed25519="b" * 128,
    )
    mp_msg = provenance.build_marketplace_signing_message(hdr_sha, arena_sha, receipt)
    mp_msg_str = provenance.build_marketplace_signing_message(
        hdr_sha, arena_sha, receipt.model_dump_json()
    )
    assert mp_msg == mp_msg_str

    key = b"\x42" * 32
    nonce = b"\x07" * 12
    encrypted = provenance.chacha20_xor_arena(arena_bytes, key_32=key, nonce_12=nonce)
    assert encrypted != arena_bytes
    decrypted = provenance.chacha20_xor_arena(encrypted, key_32=key, nonce_12=nonce)
    assert decrypted == arena_bytes

    with pytest.raises(DnamValidationError):
        provenance.chacha20_xor_arena(arena_bytes, key_32=b"short", nonce_12=nonce)
    with pytest.raises(DnamValidationError):
        provenance.chacha20_xor_arena(arena_bytes, key_32=key, nonce_12=b"short")


def test_dual_mechanism_buyer_watermarking_and_structural_hash(
    ref_container_core2: DnamContainer,
) -> None:
    f_master = ref_container_core2.structural_hash()
    assert f_master.shape == (160,)

    secret = b"\x99" * 32
    code = provenance.generate_tardos_code(secret, model_id="darkglass_b7k", license_id="lic_001")
    assert code.shape == (4608,)

    wm_params = WatermarkParams(
        kappa_48=tuple(1.0 + 0.01 * ((i % 3) - 1) for i in range(48)),
        tardos_code_4608=tuple(float(v) for v in code),
        tardos_epsilon=1e-4,
    )
    watermarked = ref_container_core2.with_watermark(wm_params)
    f_buyer = watermarked.structural_hash()

    # Gauge-invariant structural hash must remain invariant under Mechanism A + B
    sim = provenance.structural_cosine_similarity(f_master, f_buyer)
    assert sim > 0.9999

    # Tardos matched-filter Z-score must strongly identify the buyer
    z_score = provenance.score_tardos_watermark(
        bytes(ref_container_core2.arena),
        bytes(watermarked.arena),
        code_4608=code,
    )
    assert z_score > 60.0

    # Mechanism A only (tardos_code_4608=None)
    wm_mech_a_only = WatermarkParams(
        kappa_48=tuple(1.0 + 0.005 * ((i % 3) - 1) for i in range(48)),
    )
    assert ref_container_core2.with_watermark(wm_mech_a_only).structural_hash().shape == (160,)

    # Standalone Mechanism A and Mechanism B functions
    arena_a = provenance.apply_isometric_gauge_watermark(
        bytes(ref_container_core2.arena),
        kappa_48=[1.0 + 0.005 * ((i % 3) - 1) for i in range(48)],
        q_buyer_4x4=np.eye(4, dtype=np.float32),
    )
    arena_ab = provenance.apply_tardos_watermark(arena_a, code_4608=code, epsilon=1e-4)
    assert len(arena_ab) == len(ref_container_core2.arena)

    with pytest.raises(DnamValidationError):
        provenance.structural_cosine_similarity(np.zeros((2, 80)), f_master)
