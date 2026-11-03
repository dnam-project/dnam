#![cfg(feature = "crypto")]

use chacha20poly1305::aead::{AeadInPlace, KeyInit};
use chacha20poly1305::{ChaCha20Poly1305, Nonce};
use dnam_engine::provenance::{CryptoProvider, RustCryptoProvider, encode_hex};
use dnam_engine::schema::DnamError;
use hkdf::Hkdf;
use sha2::Sha256;

const NON_CONTRIBUTORY: DnamError =
    DnamError::CryptoVerificationFailed("non-contributory X25519 shared secret");

#[test]
fn legacy_wrap_and_unwrap_reject_low_order_x25519_points() {
    let zero = [0u8; 32];
    let mut one = [0u8; 32];
    one[0] = 1;
    let mut zero_with_high_bit = [0u8; 32];
    zero_with_high_bit[31] = 0x80;
    let device_id = [0u8; 16];
    let device_id_hex = encode_hex(&device_id);
    let provider = RustCryptoProvider::new().with_device_key(&device_id_hex, [9u8; 32]);

    for public_key in [zero, one, zero_with_high_bit] {
        assert_eq!(
            RustCryptoProvider::wrap_device_cek(
                &[7u8; 32],
                &public_key,
                &device_id_hex,
                &[42u8; 32]
            ),
            Err(NON_CONTRIBUTORY)
        );
        assert_eq!(
            provider.unwrap_seal_slot_cek(&public_key, &device_id, &[0u8; 48]),
            Err(NON_CONTRIBUTORY)
        );
    }
}

#[test]
fn low_order_ephemeral_key_rejects_a_publicly_forgeable_slot() {
    let device_id = [3u8; 16];
    let device_id_hex = encode_hex(&device_id);
    let hkdf = Hkdf::<Sha256>::new(Some(device_id_hex.as_bytes()), &[0u8; 32]);
    let mut key_and_nonce = [0u8; 44];
    hkdf.expand(b"dnam-hpke-cek-v1", &mut key_and_nonce)
        .unwrap();
    let cipher = ChaCha20Poly1305::new_from_slice(&key_and_nonce[..32]).unwrap();
    let mut content_key = [42u8; 32];
    let tag = cipher
        .encrypt_in_place_detached(
            Nonce::from_slice(&key_and_nonce[32..]),
            device_id_hex.as_bytes(),
            &mut content_key,
        )
        .unwrap();
    let mut wrapped = [0u8; 48];
    wrapped[..32].copy_from_slice(&content_key);
    wrapped[32..].copy_from_slice(&tag);
    let provider = RustCryptoProvider::new().with_device_key(&device_id_hex, [9u8; 32]);
    assert_eq!(
        provider.unwrap_seal_slot_cek(&[0u8; 32], &device_id, &wrapped),
        Err(NON_CONTRIBUTORY)
    );
}

#[test]
fn contributory_legacy_keys_still_round_trip_and_reject_tampering() {
    let device_secret = [9u8; 32];
    let ephemeral_secret = [7u8; 32];
    let device_id = [3u8; 16];
    let device_id_hex = encode_hex(&device_id);
    let content_key = [42u8; 32];
    let device_public = RustCryptoProvider::x25519_pubkey_from_secret(&device_secret);
    let ephemeral_public = RustCryptoProvider::x25519_pubkey_from_secret(&ephemeral_secret);
    let mut wrapped = RustCryptoProvider::wrap_device_cek(
        &ephemeral_secret,
        &device_public,
        &device_id_hex,
        &content_key,
    )
    .unwrap();
    let provider = RustCryptoProvider::new().with_device_key(&device_id_hex, device_secret);
    let recovered = provider
        .unwrap_seal_slot_cek(&ephemeral_public, &device_id, &wrapped)
        .unwrap();
    assert_eq!(*recovered, content_key);

    wrapped[0] ^= 1;
    assert!(matches!(
        provider.unwrap_seal_slot_cek(&ephemeral_public, &device_id, &wrapped),
        Err(DnamError::CryptoVerificationFailed(_))
    ));
}
