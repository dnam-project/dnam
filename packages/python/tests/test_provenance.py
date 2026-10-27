"""Unit and property-based tests for `dnam_engine.provenance` (JCS SHA-256, `Ed25519` signing
messages, `ChaCha20` 64B cache-line arena encryption, Dual-Mechanism Buyer Watermarking `Test 8H`,
and 160-D structural hashing)."""

from __future__ import annotations

import numpy as np
import pytest

from dnam_engine import (
    DistributionPolicy,
    DnamContainer,
    DnamProtectionProfile,
    DnamValidationError,
    VerificationStatus,
    provenance,
)
from dnam_engine.schema import (
    AuthorizedDevice,
    CreatorSignature,
    MarketplaceReceipt,
    ModelSignatures,
    PresetDef,
    WatermarkParams,
)


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


def test_ed25519_pubkey_and_sign_hex(
    ref_container_core2: DnamContainer,
) -> None:
    seed = b"\x33" * 32
    pubkey_hex = provenance.ed25519_pubkey_hex(seed)
    assert len(pubkey_hex) == 64

    hdr_sha = ref_container_core2.immutable_header_sha256()
    arena_sha = provenance.sha256_hex(bytes(ref_container_core2.arena))
    msg = provenance.build_creator_signing_message(hdr_sha, arena_sha)
    sig_hex = provenance.ed25519_sign_hex(seed, msg)
    assert len(sig_hex) == 128

    with pytest.raises(DnamValidationError):
        provenance.ed25519_pubkey_hex(b"short")
    with pytest.raises(DnamValidationError):
        provenance.ed25519_sign_hex(b"short", msg)


def test_oem_attestation_distribution_policy_and_dnam_seal(
    ref_container_core2: DnamContainer,
) -> None:
    assert ref_container_core2.protection_profile == DnamProtectionProfile.OPEN

    oem_seed = b"\x51" * 32
    oem_pubkey_hex = provenance.ed25519_pubkey_hex(oem_seed)
    device_secret = b"\x61" * 32
    device_pubkey_hex = provenance.x25519_pubkey_hex(device_secret)
    device_pubkey_bytes = bytes.fromhex(device_pubkey_hex)
    device_id_hash = "d1" * 16

    cert = provenance.sign_oem_device_certificate(
        oem_seed,
        oem_vendor="darkglass",
        model_code="anagram_rk3582",
        device_id_hash=device_id_hash,
        pubkey_x25519_32=device_pubkey_bytes,
    )
    assert cert.oem_vendor == "darkglass"
    assert cert.model_code == "anagram_rk3582"
    verified_pk = provenance.verify_oem_device_certificate(cert, oem_pubkey_hex)
    assert verified_pk == device_pubkey_hex

    # Creator signs model restricting to EnclaveSealed + darkglass:anagram_rk3582
    creator_seed = b"\x11" * 32
    creator_pk_hex = provenance.ed25519_pubkey_hex(creator_seed)
    mkt_seed = b"\x22" * 32
    mkt_pk_hex = provenance.ed25519_pubkey_hex(mkt_seed)

    policy = DistributionPolicy(
        allowed_protection_profiles=[DnamProtectionProfile.ENCLAVE_SEALED],
        allowed_oem_vendors=["darkglass:anagram_rk3582"],
    )
    hdr_with_policy = ref_container_core2.header.model_copy(
        update={
            "metadata": ref_container_core2.header.metadata.model_copy(
                update={"distribution_policy": policy}
            )
        }
    )
    hdr_sha = provenance.immutable_header_sha256(hdr_with_policy)
    arena_plain = bytes(ref_container_core2.arena)
    master_arena_sha = provenance.sha256_hex(arena_plain)
    creator_msg = provenance.build_creator_signing_message(hdr_sha, master_arena_sha)
    creator_sig_hex = provenance.ed25519_sign_hex(creator_seed, creator_msg)

    # Inner arena encryption (ENCRYPTED_ARENA = 1) + marketplace receipt
    eph_inner_sec = b"\x71" * 32
    eph_inner_pk_hex = provenance.x25519_pubkey_hex(eph_inner_sec)
    cek_inner_arena = b"\x81" * 32
    nonce_inner_arena = b"\x91" * 12
    wrapped_cek_hex = provenance.wrap_device_cek_hex(
        eph_inner_sec, device_pubkey_bytes, device_id_hash, cek_inner_arena
    )

    receipt_unsigned = MarketplaceReceipt(
        marketplace_id="dnam.store",
        license_id="0192b8e0-1234-7000-8000-000000000099",
        buyer_hash="aa" * 32,
        license_tier="oem_hardware",
        issued_at="2026-10-27T00:00:00Z",
        buyer_arena_sha256=master_arena_sha,
        encryption_mode="chacha20_x25519_v1",
        arena_nonce_96=nonce_inner_arena.hex(),
        ephemeral_pubkey_x25519=eph_inner_pk_hex,
        authorized_devices=[
            AuthorizedDevice(
                oem_vendor="darkglass",
                model_code="anagram_rk3582",
                device_id_hash=device_id_hash,
                wrapped_cek=wrapped_cek_hex,
            )
        ],
        receipt_sig_ed25519="00" * 64,
    )
    mkt_msg = provenance.build_marketplace_signing_message(
        hdr_sha, master_arena_sha, receipt_unsigned
    )
    mkt_sig_hex = provenance.ed25519_sign_hex(mkt_seed, mkt_msg)
    receipt = receipt_unsigned.model_copy(update={"receipt_sig_ed25519": mkt_sig_hex})

    hdr_signed = hdr_with_policy.model_copy(
        update={
            "metadata": hdr_with_policy.metadata.model_copy(
                update={
                    "signatures": ModelSignatures(
                        immutable_header_sha256=hdr_sha,
                        master_arena_sha256=master_arena_sha,
                        creator_signature=CreatorSignature(
                            signer_id="creator_01",
                            pubkey_ed25519=creator_pk_hex,
                            sig_ed25519=creator_sig_hex,
                        ),
                        marketplace_receipt=receipt,
                    )
                }
            )
        }
    )

    encrypted_arena = provenance.chacha20_xor_arena(
        arena_plain, key_32=cek_inner_arena, nonce_12=nonce_inner_arena
    )
    # Loading inner container directly as DeviceAllowlisted fails due to DistributionPolicy
    from dnam_engine import _native

    inner_bytes = _native.native_build_container(
        hdr_signed.model_dump_json(by_alias=True),
        encrypted_arena,
        arena_plain,
        True,
    )
    with pytest.raises(DnamValidationError, match="distribution_policy"):
        DnamContainer.from_bytes(
            inner_bytes,
            device_id_hash=device_id_hash,
            device_secret_x25519=device_secret,
            trusted_marketplace_pubkey_hex=mkt_pk_hex,
        )

    # Seal into .dnam-seal (b"DNSL") with EnclaveSealed nested double encryption
    sealed_bytes = provenance.seal_container(
        inner_bytes,
        recipient_certs=[cert],
        trusted_oem_pubkeys={"darkglass": oem_pubkey_hex},
        marketplace_seed_32=mkt_seed,
        ephemeral_secret_32=b"\xa1" * 32,
        cek_32=b"\xb1" * 32,
        nonce_12=b"\xc1" * 12,
    )
    assert sealed_bytes[:4] == b"DNSL"
    assert len(sealed_bytes) % 64 == 0

    unsealed_container = DnamContainer.unseal(
        sealed_bytes,
        device_id_hash=device_id_hash,
        device_secret_x25519=device_secret,
        trusted_marketplace_pubkey_hex=mkt_pk_hex,
    )
    assert unsealed_container.protection_profile == DnamProtectionProfile.ENCLAVE_SEALED
    assert unsealed_container.verification == VerificationStatus.MARKETPLACE_VERIFIED
    assert bytes(unsealed_container.arena) == arena_plain

    # Unsealing with raw inner bytes returns encrypted_arena until inner unwrap
    raw_inner = provenance.unseal_container(
        sealed_bytes,
        device_id_hash=device_id_hash,
        device_secret_32=device_secret,
        trusted_marketplace_pubkey_hex=mkt_pk_hex,
    )
    assert raw_inner == inner_bytes
