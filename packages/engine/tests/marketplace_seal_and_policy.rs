//! `.dnam-seal` (`DNSL`) Hardware Enclave Envelope, `OemDeviceCertificate`, and `DistributionPolicy`
//! Verification Suite (`tests/marketplace_seal_and_policy.rs`).
//!
//! Canonical Reference: `docs/security/provenance_and_enclave_drm.md`.

use dnam_engine::container::{DnamContainerView, build_dnam_container};
use dnam_engine::provenance::{
    DNSL_MAGIC, DnamSealEnvelopePreamble, DnamSealRecipientSlot, RustCryptoProvider,
    build_creator_signing_message, build_marketplace_receipt_signing_message,
    chacha20_xor_arena_blocks, compute_immutable_header_sha256, encode_hex, seal_dnam_container,
    sha256_digest, unseal_dnam_container, verify_oem_device_certificate,
};
use dnam_engine::schema::{
    AuthorizedDevice, CreatorSignature, DistributionPolicy, DnamJsonHeader, DnamProtectionProfile,
    MarketplaceReceipt, ModelSignatures,
};
use dnam_engine::{DynamicDnamEngine, Tier};

#[test]
fn test_dnam_seal_dnsl_envelope_oem_certificate_and_distribution_policy() {
    let mut header = DnamJsonHeader::default_reference();
    let mut master_arena = vec![0u8; 64_000];
    master_arena[124..128].copy_from_slice(&0.9993_f32.to_le_bytes());
    for i in 0..640 {
        let val = 0.03_f32 * (((i * 7 + 3) % 11) as f32 - 5.0);
        master_arena[4_928 + i * 4..4_928 + (i + 1) * 4].copy_from_slice(&val.to_le_bytes());
    }

    // 1. Configure Creator-Signed DistributionPolicy restricting model to EnclaveSealed on Darkglass Anagram
    header.metadata.distribution_policy = Some(DistributionPolicy {
        allowed_protection_profiles: vec![DnamProtectionProfile::EnclaveSealed],
        allowed_oem_vendors: vec!["darkglass:anagram_rk3582".to_string()],
    });

    let creator_seed = [21u8; 32];
    let creator_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&creator_seed);
    let mp_seed = [33u8; 32];
    let mp_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&mp_seed);

    let oem_darkglass_seed = [55u8; 32];
    let oem_darkglass_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&oem_darkglass_seed);
    let oem_other_seed = [66u8; 32];
    let oem_other_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&oem_other_seed);

    let device_id_hash = "9c1185a5c5e9fc54612808977ee8f548";
    let dev_secret_x25519 = [44u8; 32];
    let dev_pubkey_x25519 = RustCryptoProvider::x25519_pubkey_from_secret(&dev_secret_x25519);

    let cert_anagram = RustCryptoProvider::sign_oem_device_certificate(
        &oem_darkglass_seed,
        "darkglass",
        "anagram_rk3582",
        device_id_hash,
        &dev_pubkey_x25519,
    )
    .unwrap();
    let verified_x25519 =
        verify_oem_device_certificate(&cert_anagram, &oem_darkglass_pk, &RustCryptoProvider::new())
            .unwrap();
    assert_eq!(verified_x25519, dev_pubkey_x25519);

    assert!(
        verify_oem_device_certificate(&cert_anagram, &oem_other_pk, &RustCryptoProvider::new())
            .is_err()
    );

    // 2. Sign inner .dnam with creator signature + marketplace receipt (ENCRYPTED_ARENA = 1)
    let eph_inner_sk = [77u8; 32];
    let eph_inner_pk = RustCryptoProvider::x25519_pubkey_from_secret(&eph_inner_sk);
    let cek_arena = [0x3C_u8; 32];
    let nonce_arena = [0x22_u8; 12];
    let wrapped_cek_arena = RustCryptoProvider::wrap_device_cek(
        &eph_inner_sk,
        &dev_pubkey_x25519,
        device_id_hash,
        &cek_arena,
    )
    .unwrap();

    let imm_sha = compute_immutable_header_sha256(&header).unwrap();
    let master_sha = sha256_digest(&master_arena);
    let creator_msg = build_creator_signing_message(&imm_sha, &master_sha);
    let creator_sig = RustCryptoProvider::ed25519_sign(&creator_seed, &creator_msg);

    let mut receipt = MarketplaceReceipt {
        marketplace_id: "dnam.store".to_string(),
        license_id: "0192b6c0-7e4a-7000-8000-998877665544".to_string(),
        buyer_hash: encode_hex(&[0xCD; 32]),
        license_tier: "oem_bundle".to_string(),
        issued_at: "2026-10-27T00:00:00Z".to_string(),
        buyer_arena_sha256: encode_hex(&master_sha),
        buyer_tier_arena_sha256: None,
        encryption_mode: "x25519_chacha20_arena".to_string(),
        arena_nonce_96: Some(encode_hex(&nonce_arena)),
        ephemeral_pubkey_x25519: Some(encode_hex(&eph_inner_pk)),
        authorized_devices: vec![AuthorizedDevice {
            oem_vendor: "darkglass".to_string(),
            model_code: "anagram_rk3582".to_string(),
            device_id_hash: device_id_hash.to_string(),
            wrapped_cek: Some(encode_hex(&wrapped_cek_arena)),
        }],
        royalty_splits: Vec::new(),
        receipt_sig_ed25519: String::new(),
    };
    let receipt_msg =
        build_marketplace_receipt_signing_message(&imm_sha, &master_sha, &receipt).unwrap();
    receipt.receipt_sig_ed25519 =
        encode_hex(&RustCryptoProvider::ed25519_sign(&mp_seed, &receipt_msg));

    header.metadata.signatures = Some(ModelSignatures {
        immutable_header_sha256: encode_hex(&imm_sha),
        master_arena_sha256: encode_hex(&master_sha),
        tier_arena_sha256: None,
        creator_signature: CreatorSignature {
            signer_id: "did:web:darkglass.com".to_string(),
            pubkey_ed25519: encode_hex(&creator_pk),
            sig_ed25519: encode_hex(&creator_sig),
        },
        marketplace_receipt: Some(receipt),
    });

    let mut encrypted_arena = master_arena.clone();
    chacha20_xor_arena_blocks(&mut encrypted_arena, &cek_arena, &nonce_arena, 0).unwrap();
    let inner_dnam_bytes =
        build_dnam_container(&mut header, &master_arena, &encrypted_arena, true).unwrap();

    // 3. Loading the unsealed inner .dnam directly must FAIL because DistributionPolicy requires EnclaveSealed!
    let crypto = RustCryptoProvider::new()
        .with_device_key(device_id_hash, dev_secret_x25519)
        .with_marketplace_pubkey(mp_pk);
    assert!(DnamContainerView::from_bytes(&inner_dnam_bytes, &crypto).is_err());

    // 4. Attempting to seal for a different model ("darkglass:other_pedal") must FAIL DistributionPolicy!
    let cert_wrong_model = RustCryptoProvider::sign_oem_device_certificate(
        &oem_darkglass_seed,
        "darkglass",
        "other_pedal",
        device_id_hash,
        &dev_pubkey_x25519,
    )
    .unwrap();
    let trusted_oems = [("darkglass", oem_darkglass_pk), ("other", oem_other_pk)];
    let eph_outer_sk = [88u8; 32];
    let cek_outer = [0xA5_u8; 32];
    let nonce_outer = [0x77_u8; 12];
    assert!(
        seal_dnam_container(
            &inner_dnam_bytes,
            &[cert_wrong_model],
            &trusted_oems,
            &mp_seed,
            &eph_outer_sk,
            &cek_outer,
            &nonce_outer,
        )
        .is_err()
    );

    // 5. Sealing for the authorized "darkglass:anagram_rk3582" certificate succeeds and produces 64B-aligned DNSL
    let sealed_bytes = seal_dnam_container(
        &inner_dnam_bytes,
        std::slice::from_ref(&cert_anagram),
        &trusted_oems,
        &mp_seed,
        &eph_outer_sk,
        &cek_outer,
        &nonce_outer,
    )
    .unwrap();
    assert_eq!(&sealed_bytes[0..4], &DNSL_MAGIC);
    assert_eq!(sealed_bytes.len() % 64, 0);
    assert_eq!(sealed_bytes.len(), 3 * 64 + inner_dnam_bytes.len());

    // 6. Unsealing recovers the exact inner .dnam bytes
    let unsealed_inner = unseal_dnam_container(&sealed_bytes, &crypto).unwrap();
    assert_eq!(unsealed_inner, inner_dnam_bytes);

    // 7. DnamContainerView::from_bytes & DynamicDnamEngine load .dnam-seal directly and report EnclaveSealed
    let sealed_view = DnamContainerView::from_bytes(&sealed_bytes, &crypto).unwrap();
    assert_eq!(
        sealed_view.protection_profile,
        DnamProtectionProfile::EnclaveSealed
    );
    assert_eq!(
        sealed_view.arena_storage.as_bytes(),
        master_arena.as_slice()
    );

    let mut dyn_eng = DynamicDnamEngine::from_bytes_owned_with_crypto(
        &sealed_bytes,
        Tier::Ultra,
        48_000,
        &crypto,
    )
    .unwrap();
    assert_eq!(
        dyn_eng.protection_profile(),
        DnamProtectionProfile::EnclaveSealed
    );
    let in_l = [0.25_f32; 64];
    let in_r = [0.25_f32; 64];
    let mut out_l = [0.0_f32; 64];
    let mut out_r = [0.0_f32; 64];
    dyn_eng.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
    assert!(out_l.iter().any(|&v| v.abs() > 1e-6));

    // 8. Verify seal error branches (preamble/slot parsing, unaligned inner, empty recipients, unknown OEM, tampered seal)
    assert!(DnamSealEnvelopePreamble::from_bytes(&[0u8; 32]).is_err());
    assert!(DnamSealRecipientSlot::from_bytes(&[0u8; 32]).is_err());
    assert!(
        seal_dnam_container(
            &inner_dnam_bytes[..100],
            std::slice::from_ref(&cert_anagram),
            &trusted_oems,
            &mp_seed,
            &eph_outer_sk,
            &cek_outer,
            &nonce_outer,
        )
        .is_err()
    );
    assert!(
        seal_dnam_container(
            &inner_dnam_bytes,
            &[],
            &trusted_oems,
            &mp_seed,
            &eph_outer_sk,
            &cek_outer,
            &nonce_outer,
        )
        .is_err()
    );
    assert!(
        seal_dnam_container(
            &inner_dnam_bytes,
            &[cert_anagram],
            &[],
            &mp_seed,
            &eph_outer_sk,
            &cek_outer,
            &nonce_outer,
        )
        .is_err()
    );

    let wrong_crypto = RustCryptoProvider::new()
        .with_device_key(device_id_hash, [99u8; 32])
        .with_marketplace_pubkey(mp_pk);
    assert!(DnamContainerView::from_bytes(&sealed_bytes, &wrong_crypto).is_err());

    let missing_dev_crypto = RustCryptoProvider::new()
        .with_device_key("00000000000000000000000000000000", dev_secret_x25519)
        .with_marketplace_pubkey(mp_pk);
    assert!(unseal_dnam_container(&sealed_bytes, &missing_dev_crypto).is_err());

    let mut tampered_seal = sealed_bytes;
    tampered_seal[200] ^= 0x01;
    assert!(DnamContainerView::from_bytes(&tampered_seal, &crypto).is_err());
}
