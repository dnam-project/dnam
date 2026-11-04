"""Tests for `dnam.provenance` (signing, verification, sealing, DAU1/DME1/HPKE, preview,
watermarking).
"""

from __future__ import annotations

import numpy as np
import pytest

import dnam
from dnam.errors import ContainerError, ProvenanceError, ValidationError
from dnam.provenance import (
    AnticloneAction,
    Dau1Manifest,
    apply_gauge_watermark,
    apply_tardos_watermark,
    build_dme1_header,
    build_oem_cert_v2_signing_message,
    chacha20_xor_arena,
    compute_preview_window,
    derive_watermark_seed,
    ed25519_public_key,
    encode_hpke_slots,
    encode_recipient_key_ids,
    encode_record_directory,
    enforce_distribution_policy,
    evaluate_anticlone_policy,
    extract_stage3_fingerprint,
    extract_structural_hash,
    generate_tardos_code,
    hmac_sha256_hex,
    immutable_header_sha256,
    parse_dme1_plaintext,
    parse_hpke_slots,
    parse_recipient_key_ids,
    parse_record_directory,
    score_tardos_watermark,
    seal_model,
    sha256_hex,
    sign_creator_header,
    sign_marketplace_receipt,
    sign_oem_certificate,
    structural_cosine_similarity,
    unseal_model,
    validate_preview_switch_options,
    validate_royalty_splits,
    verify_model,
    verify_oem_certificate,
    wrap_device_cek,
    x25519_public_key,
)
from dnam.schema import (
    DistributionPolicy,
    DnamProtectionProfile,
    PresetDef,
    RoyaltySplit,
    VerificationStatus,
    WatermarkParams,
)


def test_creator_and_marketplace_signing_and_tamper_detection(
    reference_model: dnam.Model,
) -> None:
    """verify[sdk.python.provenance.v2-security]
    Verifies Ed25519 creator signing, preset-safe JCS digest invariance, marketplace
    counter-signing, tier prefix stripping without re-signing, and tamper rejection.
    """
    creator_seed = bytes([0x11] * 32)
    mkt_seed = bytes([0x22] * 32)
    mkt_pk = ed25519_public_key(mkt_seed)

    signed_hdr = sign_creator_header(
        reference_model.header,
        reference_model.arena_bytes,
        creator_seed_32=creator_seed,
        signer_id="did:web:dnam.audio:creators:alice",
    )
    signed_model = dnam.Model.build(signed_hdr, reference_model.arena_bytes)
    assert signed_model.verification == VerificationStatus.CREATOR_VERIFIED
    report = verify_model(signed_model)
    assert report.is_verified
    assert report.status == VerificationStatus.CREATOR_VERIFIED
    assert report.signer_id == "did:web:dnam.audio:creators:alice"

    # Adding a user preset in-place preserves immutable_header_sha256 and CREATOR_VERIFIED!
    added_preset = PresetDef(
        id="user_fav",
        name="User Fav",
        controls={"gain": 6.0},
    )
    edited_hdr = signed_model.header.model_copy(
        update={"presets": [*signed_model.header.presets, added_preset]}
    )
    assert immutable_header_sha256(edited_hdr) == immutable_header_sha256(signed_hdr)
    edited_model = signed_model.with_header(edited_hdr)
    assert edited_model.verification == VerificationStatus.CREATOR_VERIFIED

    # Stripping to Lite preserves CREATOR_VERIFIED via tier_arena_sha256
    lite_signed = signed_model.with_tier_limit("lite")
    assert lite_signed.verification == VerificationStatus.CREATOR_VERIFIED

    # Marketplace receipt counter-signing
    mkt_hdr = sign_marketplace_receipt(
        signed_hdr,
        reference_model.arena_bytes,
        marketplace_seed_32=mkt_seed,
        marketplace_id="dnam.store",
        license_id="lic-001",
        buyer_hash="aa" * 32,
        license_tier="commercial_studio",
        royalty_splits=[RoyaltySplit(recipient_id="alice", role="amp", share_bps=10_000)],
    )
    mkt_model = dnam.Model.build(mkt_hdr, reference_model.arena_bytes)
    mkt_report = verify_model(mkt_model, trusted_marketplace_pubkey_hex=mkt_pk)
    assert mkt_report.status == VerificationStatus.MARKETPLACE_VERIFIED
    assert mkt_report.license_id == "lic-001"

    # Tampering with immutable header field (name) fails verification with ProvenanceError
    tampered_meta = mkt_hdr.metadata.model_copy(update={"name": "Tampered"})
    tampered_hdr = mkt_hdr.model_copy(update={"metadata": tampered_meta})
    with pytest.raises(ProvenanceError):
        verify_model(
            tampered_hdr,
            plaintext_arena=reference_model.arena_bytes,
            has_signatures_flag=True,
        )


def test_oem_certificate_and_dnsl_seal_unseal_roundtrip(
    reference_model: dnam.Model,
) -> None:
    """verify[sdk.python.provenance.v2-security]
    Verifies OemDeviceCertificate signing/verification, V2 signing message, ChaCha20 arena XOR,
    and .dnam-seal (DNSL) hardware sealing/unsealing.
    """
    oem_seed = bytes([0x33] * 32)
    oem_pk_hex = ed25519_public_key(oem_seed)
    dev_sec = bytes([0x44] * 32)
    dev_pk_hex = x25519_public_key(dev_sec)
    dev_pk_bytes = bytes.fromhex(dev_pk_hex)
    dev_id_hash = "ab" * 16

    cert = sign_oem_certificate(
        oem_seed,
        oem_vendor="darkglass",
        model_code="anagram_rk3582",
        device_id_hash=dev_id_hash,
        pubkey_x25519_32=dev_pk_bytes,
    )
    verified_pk = verify_oem_certificate(cert, oem_pk_hex)
    assert verified_pk == dev_pk_hex

    v2_msg = build_oem_cert_v2_signing_message(
        oem_vendor="darkglass",
        model_code="anagram_rk3582",
        device_id_hash_16=bytes.fromhex(dev_id_hash),
        pubkey_x25519_32=dev_pk_bytes,
        issuer_pubkey_ed25519_32=bytes.fromhex(oem_pk_hex),
    )
    assert v2_msg.startswith(b"\x06\x00\x00\x00domain")

    # ChaCha20 XOR round-trip & CEK wrapping
    key = bytes([0x55] * 32)
    nonce = bytes([0x66] * 12)
    ct = chacha20_xor_arena(b"\x00" * 128, key, nonce)
    pt = chacha20_xor_arena(ct, key, nonce)
    assert pt == b"\x00" * 128
    wrapped_cek = wrap_device_cek(bytes([0x77] * 32), dev_pk_bytes, dev_id_hash, key)
    assert len(wrapped_cek) == 96

    # Seal & unseal .dnam-seal (DNSL)
    mkt_seed = bytes([0x88] * 32)
    mkt_pk_hex = ed25519_public_key(mkt_seed)
    sealed = seal_model(
        reference_model,
        recipient_certificates=[cert],
        trusted_oem_pubkeys={"darkglass": oem_pk_hex},
        marketplace_seed_32=mkt_seed,
        ephemeral_secret_32=bytes([0x99] * 32),
        cek_32=key,
        nonce_12=nonce,
    )
    assert sealed[:4] == b"DNSL"
    loaded_sealed = dnam.Model.from_bytes(
        sealed,
        device_id_hash=dev_id_hash,
        device_secret=dev_sec,
        trusted_marketplace_pubkey_hex=mkt_pk_hex,
    )
    assert loaded_sealed.protection_profile == DnamProtectionProfile.ENCLAVE_SEALED
    unsealed = unseal_model(
        sealed,
        device_id_hash=dev_id_hash,
        device_secret_32=dev_sec,
        trusted_marketplace_pubkey_hex=mkt_pk_hex,
    )
    assert unsealed.name == reference_model.name
    assert unsealed.protection_profile == DnamProtectionProfile.OPEN

    # Wrong device secret fails to unseal
    with pytest.raises((ContainerError, ProvenanceError)):
        unseal_model(
            sealed,
            device_id_hash=dev_id_hash,
            device_secret_32=bytes([0xFE] * 32),
            trusted_marketplace_pubkey_hex=mkt_pk_hex,
        )


def test_dau1_dme1_hpke_preview_and_distribution_policy(
    reference_model: dnam.Model,
) -> None:
    """verify[sdk.python.provenance.v2-security]
    Verifies DAU1 manifest encode/decode, DME1 header/plaintext, HPKE slots & record directory,
    preview window/switch containment, royalty splits, and distribution policy enforcement.
    """
    dau = Dau1Manifest(
        auth_kind=0,
        creator_pubkey_hex="11" * 32,
        creator_max_tier=1,
        master_prefixes_hex=["22" * 32, "33" * 32],
        creator_signature_hex="44" * 64,
    )
    wire = dau.to_bytes()
    assert wire[:4] == b"DAU1"
    decoded = Dau1Manifest.from_bytes(wire)
    assert decoded == dau

    dme_hdr = build_dme1_header(1, 1, 128)
    assert dme_hdr[:4] == b"DME1"
    rec_pk = bytes([0xAA] * 32)
    rec_digest = sha256_hex(rec_pk)
    dmu1 = dnam.Model.build_dmu1_frame('{"presets":[],"user_metadata":{}}', 64)
    dme_pt = (1).to_bytes(4, "little") + rec_pk + dmu1
    parsed_json = parse_dme1_plaintext(dme_pt, [rec_digest])
    assert '"presets":[]' in parsed_json

    # HPKE recipient key IDs, slots, and record directory round-trips
    r_bytes = encode_recipient_key_ids(["11" * 32, "22" * 32])
    assert parse_recipient_key_ids(r_bytes) == ["11" * 32, "22" * 32]
    s_bytes = encode_hpke_slots([("aa" * 32, "bb" * 48)])
    assert parse_hpke_slots(s_bytes, 1) == [("aa" * 32, "bb" * 48)]
    d_bytes = encode_record_directory(0, [(0, 64), (64, 64)])
    assert parse_record_directory(d_bytes, 0) == [(0, 64), (64, 64)]

    # Preview single-control window & switch options
    lo, hi = compute_preview_window(0.5, 0.1, (0.0, 1.0))
    assert abs(lo - 0.4) < 1e-9
    assert abs(hi - 0.6) < 1e-9
    with pytest.raises(ValidationError):
        compute_preview_window(0.5, 0.5, (0.0, 1.0))
    p_opt, a_opt = validate_preview_switch_options(
        "off",
        "on",
        observed=["off", "on"],
        active=["off", "on"],
        safe=["off", "on"],
    )
    assert (p_opt, a_opt) == ("off", "on")
    with pytest.raises(ValidationError):
        validate_preview_switch_options(
            "off", "off", observed=["off"], active=["off"], safe=["off"]
        )

    # Royalty splits & distribution policy
    validate_royalty_splits(
        "personal",
        [RoyaltySplit(recipient_id="a", role="amp", share_bps=10_000)],
    )
    with pytest.raises(ValidationError):
        validate_royalty_splits(
            "personal",
            [RoyaltySplit(recipient_id="a", role="amp", share_bps=5_000)],
        )

    restricted_meta = reference_model.header.metadata.model_copy(
        update={
            "distribution_policy": DistributionPolicy(
                allowed_protection_profiles=[DnamProtectionProfile.ENCLAVE_SEALED]
            )
        }
    )
    restricted_hdr = reference_model.header.model_copy(update={"metadata": restricted_meta})
    with pytest.raises(ProvenanceError):
        enforce_distribution_policy(restricted_hdr, "open")
    enforce_distribution_policy(restricted_hdr, "enclave_sealed")

    assert (
        evaluate_anticlone_policy(
            fingerprint_matched=True,
            buyer_watermark_identified=True,
            independent_device_source_evidence=False,
        )
        == AnticloneAction.FLAG_LICENSE_HOLD_FOR_REVIEW
    )
    assert (
        evaluate_anticlone_policy(
            fingerprint_matched=True,
            buyer_watermark_identified=True,
            independent_device_source_evidence=True,
        )
        == AnticloneAction.REVOKE_IDENTIFIED_DEVICE_FUTURE_LICENSING_ONLY
    )
    assert len(hmac_sha256_hex(b"\x01" * 32, b"msg")) == 64


def test_dual_mechanism_buyer_watermarking_and_structural_hash() -> None:
    """verify[sdk.python.provenance.v2-security]
    Verifies Isometric Gauge Watermarking (preserving the 224-D structural hash with cosine
    similarity == 1.0) and Coupled Tardos-7680 traitor tracing.
    """
    rng = np.random.default_rng(42)
    master = bytearray(64_000)
    floats = rng.uniform(0.02, 0.08, size=(63_000 - 128) // 4).astype(np.float32)
    master[128:63_000] = floats.tobytes()
    master_bytes = bytes(master)

    hash_master = extract_structural_hash(master_bytes)
    assert hash_master.shape == (224,)

    theta = 0.25
    ct, st = float(np.cos(theta)), float(np.sin(theta))
    params = WatermarkParams(
        kappa_64=tuple(1.03 if i % 2 == 0 else 0.97 for i in range(64)),
        q_buyer_4x4=(
            (ct, -st, 0.0, 0.0),
            (st, ct, 0.0, 0.0),
            (0.0, 0.0, ct, -st),
            (0.0, 0.0, st, ct),
        ),
    )
    gauged = apply_gauge_watermark(master_bytes, params)
    gauged_kw = apply_gauge_watermark(
        master_bytes,
        kappa_64=np.ones(64, dtype=np.float32),
        q_buyer_4x4=np.eye(4, dtype=np.float32),
    )
    assert len(gauged_kw) == 64_000
    hash_gauged = extract_structural_hash(gauged)
    cos_sim = structural_cosine_similarity(hash_master, hash_gauged)
    assert abs(cos_sim - 1.0) < 1e-6

    secret = bytes([0x42] * 32)
    code_buyer1 = generate_tardos_code(secret, "model_01", "buyer_1")
    code_buyer2 = generate_tardos_code(secret, "model_01", "buyer_2")
    watermarked = apply_tardos_watermark(master_bytes, code_buyer1, epsilon=1.5e-3)

    score_guilty = score_tardos_watermark(master_bytes, watermarked, code_buyer1)
    score_innocent = score_tardos_watermark(master_bytes, watermarked, code_buyer2)
    assert score_guilty > 20.0
    assert abs(score_innocent) < 5.0

    seed_hex = derive_watermark_seed(
        secret,
        marketplace_pubkey_32=bytes([0x11] * 32),
        watermark_key_id_16=bytes([0x22] * 16),
        creator_signed_manifest_sha256_32=bytes([0x33] * 32),
        model_id_16=bytes([0x44] * 16),
        watermark_profile_sha256_32=bytes([0x55] * 32),
        owning_tier=1,
        license_id_16=bytes([0xAA] * 16),
        buyer_id_16=bytes([0xBB] * 16),
    )
    assert len(seed_hex) == 64

    fp = extract_stage3_fingerprint(master_bytes[128:256], master_bytes[256:512], 8)
    assert fp.shape == (88,)
