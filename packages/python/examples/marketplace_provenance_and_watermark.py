"""Example 4: Marketplace Provenance, `.dnam-seal` (`DNSL`) HPKE & Dual-Mechanism Watermarking
(`examples/marketplace_provenance_and_watermark.py`).

Run with:
  uv run --no-sync python packages/python/examples/marketplace_provenance_and_watermark.py
"""

from __future__ import annotations

import numpy as np

import dnam
from dnam.provenance import (
    apply_gauge_watermark,
    apply_tardos_watermark,
    ed25519_public_key,
    extract_structural_hash,
    generate_tardos_code,
    score_tardos_watermark,
    seal_model,
    sign_creator_header,
    sign_marketplace_receipt,
    sign_oem_certificate,
    structural_cosine_similarity,
    unseal_model,
    verify_model,
    x25519_public_key,
)
from dnam.schema import RoyaltySplit, WatermarkParams


def main() -> None:
    """Demonstrates creator/marketplace signing, DNSL hardware sealing, and buyer watermarking."""
    model = dnam.Model.reference()
    creator_seed = bytes([0x11] * 32)
    mkt_seed = bytes([0x22] * 32)
    mkt_pubkey_hex = ed25519_public_key(mkt_seed)

    # 1. Sign creator header + counter-sign marketplace receipt
    creator_hdr = sign_creator_header(
        model.header,
        model.arena_bytes,
        creator_seed_32=creator_seed,
        signer_id="did:web:dnam.audio:creators:alice",
    )
    mkt_hdr = sign_marketplace_receipt(
        creator_hdr,
        model.arena_bytes,
        marketplace_seed_32=mkt_seed,
        marketplace_id="dnam.store",
        license_id="0192f6c0-0000-7000-8000-000000000001",
        buyer_hash="aa" * 32,
        license_tier="commercial_studio",
        royalty_splits=[RoyaltySplit(recipient_id="alice", role="amp_capture", share_bps=10_000)],
    )
    signed_model = dnam.Model.build(mkt_hdr, model.arena_bytes)
    report = verify_model(signed_model, trusted_marketplace_pubkey_hex=mkt_pubkey_hex)
    print(
        f"Provenance verified: status={report.status.value} "
        f"signer={report.signer_id} license={report.license_id}"
    )

    # 2. Seal & unseal a hardware-bound .dnam-seal (DNSL) envelope for an OEM pedalboard
    oem_seed = bytes([0x33] * 32)
    oem_pubkey_hex = ed25519_public_key(oem_seed)
    device_secret = bytes([0x44] * 32)
    device_pubkey_bytes = bytes.fromhex(x25519_public_key(device_secret))
    device_id_hash = "ab" * 16

    oem_cert = sign_oem_certificate(
        oem_seed,
        oem_vendor="darkglass",
        model_code="anagram_rk3582",
        device_id_hash=device_id_hash,
        pubkey_x25519_32=device_pubkey_bytes,
    )
    sealed_bytes = seal_model(
        signed_model,
        recipient_certificates=[oem_cert],
        trusted_oem_pubkeys={"darkglass": oem_pubkey_hex},
        marketplace_seed_32=mkt_seed,
        ephemeral_secret_32=bytes([0x55] * 32),
        cek_32=bytes([0x66] * 32),
        nonce_12=bytes([0x77] * 12),
    )
    unsealed_model = unseal_model(
        sealed_bytes,
        device_id_hash=device_id_hash,
        device_secret_32=device_secret,
        trusted_marketplace_pubkey_hex=mkt_pubkey_hex,
    )
    print(
        f"Sealed DNSL envelope ({len(sealed_bytes)} B) -> "
        f"Unsealed {unsealed_model.name!r} ({unsealed_model.verification.value})"
    )

    # 3. Dual-Mechanism Buyer Watermarking (Isometric Gauge + Coupled Tardos-7680)
    rng = np.random.default_rng(42)
    master = bytearray(64_000)
    master[128:63_000] = (
        rng.uniform(0.02, 0.08, size=(63_000 - 128) // 4).astype(np.float32).tobytes()
    )
    master_bytes = bytes(master)

    theta = 0.25
    ct, st = float(np.cos(theta)), float(np.sin(theta))
    wm_params = WatermarkParams(
        kappa_64=tuple(1.03 if i % 2 == 0 else 0.97 for i in range(64)),
        q_buyer_4x4=(
            (ct, -st, 0.0, 0.0),
            (st, ct, 0.0, 0.0),
            (0.0, 0.0, ct, -st),
            (0.0, 0.0, st, ct),
        ),
    )
    buyer_arena = apply_gauge_watermark(master_bytes, wm_params)
    cos_sim = structural_cosine_similarity(
        extract_structural_hash(master_bytes),
        extract_structural_hash(buyer_arena),
    )

    secret = bytes([0x42] * 32)
    code_buyer1 = generate_tardos_code(secret, "model_001", "buyer_1001")
    code_buyer2 = generate_tardos_code(secret, "model_001", "buyer_1002")
    buyer_arena = apply_tardos_watermark(buyer_arena, code_buyer1, epsilon=1.5e-3)

    z_guilty = score_tardos_watermark(master_bytes, buyer_arena, code_buyer1)
    z_innocent = score_tardos_watermark(master_bytes, buyer_arena, code_buyer2)
    print(
        f"224-D Structural Hash Cosine Similarity: {cos_sim:.6f} | "
        f"Tardos Z-score: buyer_1001={z_guilty:+.2f}, buyer_1002={z_innocent:+.2f}"
    )


if __name__ == "__main__":
    main()
