//! Pluggable cryptographic and hardware-enclave provider traits and implementations.

use alloc::string::{String, ToString};
use zeroize::Zeroizing;

use crate::schema::{AuthorizedDevice, DnamError, MarketplaceReceipt, OemDeviceCertificate};

#[cfg(feature = "crypto")]
use super::sha256::{decode_hex_array, encode_hex};

/// Provenance verification status returned when loading a `.dnam` or `.dnam-ckpt` container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationStatus {
    /// Local / community model without signatures (`HAS_SIGNATURES == 0`).
    Unsigned,
    /// Creator-signed model verified against `creator_signature` (`HAS_SIGNATURES == 1`, no `marketplace_receipt`).
    CreatorVerified {
        /// Creator identifier (`signer_id`).
        signer_id: String,
        /// Verified 32-byte `Ed25519` creator public key.
        pubkey_ed25519: [u8; 32],
    },
    /// Commercial marketplace model verified against both `creator_signature` and `marketplace_receipt`.
    MarketplaceVerified {
        /// Creator identifier (`signer_id`).
        signer_id: String,
        /// Marketplace domain (`marketplace_id`).
        marketplace_id: String,
        /// Unique license UUID (`license_id`).
        license_id: String,
        /// License tier (`"free"`, `"personal"`, `"commercial_studio"`, `"oem_bundle"`).
        license_tier: String,
    },
}

/// Pluggable cryptographic and hardware-enclave provider trait (`#![no_std]`).
///
/// Unifies:
/// - Built-in `#![no_std]` software verification (`RustCryptoProvider`),
/// - Embedded Rockchip RK3582 `OTP` + ARM TrustZone `OP-TEE` (`Darkglass Anagram`),
/// - Desktop/mobile DAW plugins (`Apple Secure Enclave` `SEP` & Windows/Linux `TPM 2.0`).
pub trait CryptoProvider {
    /// Verifies an `Ed25519` signature over `message` using the 32-byte `pubkey`.
    fn ed25519_verify(
        &self,
        pubkey: &[u8; 32],
        message: &[u8],
        signature: &[u8; 64],
    ) -> Result<(), DnamError>;

    /// Unwraps the 32-byte Content Encryption Key (`CEK`) for an authorized hardware device
    /// inside the platform's secure enclave (`OP-TEE`, `SEP`, `TPM 2.0`, or software `X25519`),
    /// returning a [`Zeroizing`] buffer that automatically wipes its stack memory on drop.
    fn unwrap_device_cek(
        &self,
        receipt: &MarketplaceReceipt,
        device: &AuthorizedDevice,
    ) -> Result<Zeroizing<[u8; 32]>, DnamError>;

    /// Unwraps a 48-byte `wrapped_cek` from a `.dnam-seal` (`b"DNSL"`) binary recipient slot
    /// using the 32-byte `ephemeral_pubkey_x25519` and 16-byte raw `device_id_hash`.
    fn unwrap_seal_slot_cek(
        &self,
        ephemeral_pubkey_x25519: &[u8; 32],
        device_id_hash_16: &[u8; 16],
        wrapped_cek_48: &[u8; 48],
    ) -> Result<Zeroizing<[u8; 32]>, DnamError> {
        let epk_hex = super::sha256::encode_hex(ephemeral_pubkey_x25519);
        let dev_id_hex = super::sha256::encode_hex(device_id_hash_16);
        let wrapped_hex = super::sha256::encode_hex(wrapped_cek_48);
        let dummy_dev = AuthorizedDevice {
            oem_vendor: String::new(),
            model_code: String::new(),
            device_id_hash: dev_id_hex,
            wrapped_cek: Some(wrapped_hex),
        };
        let dummy_receipt = MarketplaceReceipt {
            marketplace_id: String::new(),
            license_id: String::new(),
            buyer_hash: String::new(),
            license_tier: String::new(),
            issued_at: String::new(),
            buyer_arena_sha256: String::new(),
            encryption_mode: String::from("x25519_chacha20_arena"),
            arena_nonce_96: None,
            ephemeral_pubkey_x25519: Some(epk_hex),
            authorized_devices: alloc::vec![],
            royalty_splits: alloc::vec![],
            receipt_sig_ed25519: String::new(),
        };
        self.unwrap_device_cek(&dummy_receipt, &dummy_dev)
    }

    /// Optional local hardware `device_id_hash` for automatic selection in `receipt.authorized_devices`.
    fn local_device_id_hash(&self) -> Option<&str> {
        None
    }

    /// Optional pinned marketplace `Ed25519` public key (`32 B`).
    fn marketplace_pubkey(&self) -> Option<&[u8; 32]> {
        None
    }
}

/// Minimal fallback provider for `--no-default-features` builds that only load open (`ENCRYPTED_ARENA = 0`, `Unsigned`) models.
#[derive(Debug, Default, Clone, Copy)]
pub struct OpenArenaCryptoProvider;

impl CryptoProvider for OpenArenaCryptoProvider {
    fn ed25519_verify(
        &self,
        _pubkey: &[u8; 32],
        _message: &[u8],
        _signature: &[u8; 64],
    ) -> Result<(), DnamError> {
        Err(DnamError::CryptoVerificationFailed(
            "Ed25519 verification requires the 'crypto' feature or a custom CryptoProvider",
        ))
    }

    fn unwrap_device_cek(
        &self,
        _receipt: &MarketplaceReceipt,
        _device: &AuthorizedDevice,
    ) -> Result<Zeroizing<[u8; 32]>, DnamError> {
        Err(DnamError::EncryptedArenaRequiresDeviceKey)
    }
}

/// Built-in `#![no_std]` RustCrypto provider (`ed25519-dalek` + `x25519-dalek` + `chacha20poly1305` + `hkdf` + `sha2`).
#[cfg(feature = "crypto")]
#[derive(Clone, Default)]
pub struct RustCryptoProvider {
    /// Optional local device identifier hash (`16 B` = 32 hex chars).
    device_id_hash: Option<String>,
    /// Optional local `X25519` device private key (zeroized on drop).
    device_secret_x25519: Option<Zeroizing<[u8; 32]>>,
    /// Optional pinned `Ed25519` public key for `marketplace_receipt` verification.
    trusted_marketplace_pubkey: Option<[u8; 32]>,
}

#[cfg(feature = "crypto")]
impl core::fmt::Debug for RustCryptoProvider {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RustCryptoProvider")
            .field("device_id_hash", &self.device_id_hash)
            .field(
                "has_device_secret_x25519",
                &self.device_secret_x25519.is_some(),
            )
            .field(
                "trusted_marketplace_pubkey",
                &self.trusted_marketplace_pubkey,
            )
            .finish()
    }
}

#[cfg(feature = "crypto")]
impl RustCryptoProvider {
    /// Creates a default `RustCryptoProvider` for open (`ENCRYPTED_ARENA = 0`) and signed `.dnam` models.
    #[inline]
    pub const fn new() -> Self {
        Self {
            device_id_hash: None,
            device_secret_x25519: None,
            trusted_marketplace_pubkey: None,
        }
    }

    /// Configures a local `X25519` device key and `device_id_hash` for unwrapping `ENCRYPTED_ARENA = 1` models.
    #[inline]
    pub fn with_device_key(
        mut self,
        device_id_hash: impl Into<String>,
        secret_x25519: [u8; 32],
    ) -> Self {
        self.device_id_hash = Some(device_id_hash.into());
        self.device_secret_x25519 = Some(Zeroizing::new(secret_x25519));
        self
    }

    /// Configures a pinned marketplace `Ed25519` public key for verifying `marketplace_receipt`.
    #[inline]
    pub const fn with_marketplace_pubkey(mut self, pubkey: [u8; 32]) -> Self {
        self.trusted_marketplace_pubkey = Some(pubkey);
        self
    }

    /// Derives the 32-byte `Ed25519` public key from a 32-byte secret key seed.
    #[inline]
    pub fn ed25519_pubkey_from_seed(seed_32: &[u8; 32]) -> [u8; 32] {
        use ed25519_dalek::SigningKey;
        let sk = SigningKey::from_bytes(seed_32);
        sk.verifying_key().to_bytes()
    }

    /// Signs `message` using a 32-byte `Ed25519` secret key seed, returning the 64-byte signature.
    #[inline]
    pub fn ed25519_sign(seed_32: &[u8; 32], message: &[u8]) -> [u8; 64] {
        use ed25519_dalek::{Signer, SigningKey};
        let sk = SigningKey::from_bytes(seed_32);
        sk.sign(message).to_bytes()
    }

    /// Derives the 32-byte `X25519` public key from a 32-byte static secret key.
    #[inline]
    pub fn x25519_pubkey_from_secret(secret_32: &[u8; 32]) -> [u8; 32] {
        use x25519_dalek::{PublicKey, StaticSecret};
        let sk = StaticSecret::from(*secret_32);
        PublicKey::from(&sk).to_bytes()
    }

    /// Signs a factory-provisioned [`OemDeviceCertificate`] binding a hardware device's
    /// `(oem_vendor, model_code, device_id_hash, pubkey_x25519)` to the OEM's root `Ed25519` key.
    pub fn sign_oem_device_certificate(
        oem_seed_ed25519: &[u8; 32],
        oem_vendor: &str,
        model_code: &str,
        device_id_hash: &str,
        pubkey_x25519: &[u8; 32],
    ) -> Result<OemDeviceCertificate, DnamError> {
        let raw_id = decode_hex_array::<16>(device_id_hash)?;
        let normalized_id = encode_hex(&raw_id);
        let msg = super::verify::build_oem_device_cert_signing_message(
            oem_vendor,
            model_code,
            &normalized_id,
            pubkey_x25519,
        );
        let sig = Self::ed25519_sign(oem_seed_ed25519, &msg);
        Ok(OemDeviceCertificate {
            oem_vendor: oem_vendor.to_string(),
            model_code: model_code.to_string(),
            device_id_hash: normalized_id,
            pubkey_x25519: encode_hex(pubkey_x25519),
            oem_sig_ed25519: encode_hex(&sig),
        })
    }

    /// Wraps a 32-byte Content Encryption Key (`CEK`) for a target device's `X25519` public key
    /// using RFC 9180 HPKE (`X25519-HKDF-SHA256` + `ChaCha20-Poly1305`), returning the 48-byte
    /// `wrapped_cek` (`32 B` ciphertext + `16 B` Poly1305 tag).
    pub fn wrap_device_cek(
        ephemeral_secret_x25519: &[u8; 32],
        device_pubkey_x25519: &[u8; 32],
        device_id_hash: &str,
        cek: &[u8; 32],
    ) -> Result<[u8; 48], DnamError> {
        use chacha20poly1305::aead::AeadInPlace;
        use chacha20poly1305::{ChaCha20Poly1305, KeyInit, Nonce};
        use hkdf::Hkdf;
        use sha2::Sha256;
        use x25519_dalek::{PublicKey, StaticSecret};

        let esk = StaticSecret::from(*ephemeral_secret_x25519);
        let dpk = PublicKey::from(*device_pubkey_x25519);
        let shared = esk.diffie_hellman(&dpk);

        let hk = Hkdf::<Sha256>::new(Some(device_id_hash.as_bytes()), shared.as_bytes());
        let mut okm = Zeroizing::new([0u8; 44]); // 32B key + 12B nonce
        hk.expand(b"dnam-hpke-cek-v1", &mut *okm).map_err(|_| {
            DnamError::CryptoVerificationFailed("HKDF-SHA256 expand failed during CEK wrap")
        })?;

        let mut key_bytes = Zeroizing::new([0u8; 32]);
        key_bytes.copy_from_slice(&okm[0..32]);
        let nonce = Nonce::from_slice(&okm[32..44]);

        let cipher = ChaCha20Poly1305::new((&*key_bytes).into());
        let mut buffer = [0u8; 32];
        buffer.copy_from_slice(cek);
        let tag = cipher
            .encrypt_in_place_detached(nonce, device_id_hash.as_bytes(), &mut buffer)
            .map_err(|_| {
                DnamError::CryptoVerificationFailed("ChaCha20-Poly1305 seal failed during CEK wrap")
            })?;

        let mut out = [0u8; 48];
        out[0..32].copy_from_slice(&buffer);
        out[32..48].copy_from_slice(tag.as_slice());
        Ok(out)
    }
}

#[cfg(feature = "crypto")]
impl CryptoProvider for RustCryptoProvider {
    fn ed25519_verify(
        &self,
        pubkey: &[u8; 32],
        message: &[u8],
        signature: &[u8; 64],
    ) -> Result<(), DnamError> {
        use ed25519_dalek::{Signature, Verifier, VerifyingKey};

        let vk = VerifyingKey::from_bytes(pubkey)
            .map_err(|_| DnamError::CryptoVerificationFailed("invalid Ed25519 public key bytes"))?;
        let sig = Signature::from_bytes(signature);
        vk.verify(message, &sig).map_err(|_| {
            DnamError::CryptoVerificationFailed("Ed25519 signature verification failed")
        })
    }

    fn unwrap_device_cek(
        &self,
        receipt: &MarketplaceReceipt,
        device: &AuthorizedDevice,
    ) -> Result<Zeroizing<[u8; 32]>, DnamError> {
        use chacha20poly1305::aead::AeadInPlace;
        use chacha20poly1305::{ChaCha20Poly1305, KeyInit, Nonce, Tag};
        use hkdf::Hkdf;
        use sha2::Sha256;
        use x25519_dalek::{PublicKey, StaticSecret};

        let dsk_bytes = self
            .device_secret_x25519
            .as_ref()
            .ok_or(DnamError::EncryptedArenaRequiresDeviceKey)?;
        let epk_hex =
            receipt
                .ephemeral_pubkey_x25519
                .as_ref()
                .ok_or(DnamError::CryptoVerificationFailed(
                    "missing ephemeral_pubkey_x25519 in receipt",
                ))?;
        let epk_bytes = decode_hex_array::<32>(epk_hex)?;
        let wrapped_hex =
            device
                .wrapped_cek
                .as_ref()
                .ok_or(DnamError::CryptoVerificationFailed(
                    "missing wrapped_cek for authorized device",
                ))?;
        let wrapped_bytes = decode_hex_array::<48>(wrapped_hex)?;

        let dsk = StaticSecret::from(**dsk_bytes);
        let epk = PublicKey::from(epk_bytes);
        let shared = dsk.diffie_hellman(&epk);

        let hk = Hkdf::<Sha256>::new(Some(device.device_id_hash.as_bytes()), shared.as_bytes());
        let mut okm = Zeroizing::new([0u8; 44]);
        hk.expand(b"dnam-hpke-cek-v1", &mut *okm).map_err(|_| {
            DnamError::CryptoVerificationFailed("HKDF-SHA256 expand failed during CEK unwrap")
        })?;

        let mut key_bytes = Zeroizing::new([0u8; 32]);
        key_bytes.copy_from_slice(&okm[0..32]);
        let nonce = Nonce::from_slice(&okm[32..44]);

        let cipher = ChaCha20Poly1305::new((&*key_bytes).into());
        let mut cek = Zeroizing::new([0u8; 32]);
        cek.copy_from_slice(&wrapped_bytes[0..32]);
        let tag = Tag::from_slice(&wrapped_bytes[32..48]);

        cipher
            .decrypt_in_place_detached(nonce, device.device_id_hash.as_bytes(), &mut *cek, tag)
            .map_err(|_| {
                DnamError::CryptoVerificationFailed(
                    "ChaCha20-Poly1305 authentication tag mismatch when unwrapping device CEK",
                )
            })?;

        Ok(cek)
    }

    fn local_device_id_hash(&self) -> Option<&str> {
        self.device_id_hash.as_deref()
    }

    fn marketplace_pubkey(&self) -> Option<&[u8; 32]> {
        self.trusted_marketplace_pubkey.as_ref()
    }
}

/// Default cryptographic provider type alias (`RustCryptoProvider` when `crypto` is enabled,
/// or `OpenArenaCryptoProvider` when built with `--no-default-features`).
#[cfg(feature = "crypto")]
pub type DefaultCryptoProvider = RustCryptoProvider;

/// Default cryptographic provider type alias when `crypto` is disabled.
#[cfg(not(feature = "crypto"))]
pub type DefaultCryptoProvider = OpenArenaCryptoProvider;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_arena_crypto_provider_and_debug() {
        let open = OpenArenaCryptoProvider;
        assert!(open.local_device_id_hash().is_none());
        assert!(open.marketplace_pubkey().is_none());
        assert!(open.ed25519_verify(&[0u8; 32], b"msg", &[0u8; 64]).is_err());

        let dummy_dev = AuthorizedDevice {
            oem_vendor: String::from("darkglass"),
            model_code: String::from("anagram_v1"),
            device_id_hash: String::from("0011"),
            wrapped_cek: None,
        };
        let dummy_rcpt = MarketplaceReceipt {
            marketplace_id: String::from("tone.dnam.audio"),
            license_id: String::from("lic-1"),
            buyer_hash: String::from("00"),
            license_tier: String::from("personal"),
            issued_at: String::from("2026-10-23T00:00:00Z"),
            buyer_arena_sha256: String::from("00"),
            encryption_mode: String::from("none"),
            arena_nonce_96: None,
            ephemeral_pubkey_x25519: None,
            authorized_devices: alloc::vec![],
            royalty_splits: alloc::vec![],
            receipt_sig_ed25519: String::from("00"),
        };
        assert_eq!(
            open.unwrap_device_cek(&dummy_rcpt, &dummy_dev),
            Err(DnamError::EncryptedArenaRequiresDeviceKey)
        );
        assert_eq!(
            open.unwrap_seal_slot_cek(&[0u8; 32], &[0u8; 16], &[0u8; 48]),
            Err(DnamError::EncryptedArenaRequiresDeviceKey)
        );

        #[cfg(feature = "crypto")]
        {
            let prov = RustCryptoProvider::new()
                .with_device_key("dev1", [7u8; 32])
                .with_marketplace_pubkey([9u8; 32]);
            let dbg = alloc::format!("{prov:?}");
            assert!(dbg.contains("RustCryptoProvider"));
            assert_eq!(prov.local_device_id_hash(), Some("dev1"));
            assert_eq!(prov.marketplace_pubkey(), Some(&[9u8; 32]));

            // Verify Ed25519 key derivation, signing, and verification roundtrip
            let seed = [0x5Au8; 32];
            let pubkey = RustCryptoProvider::ed25519_pubkey_from_seed(&seed);
            let message = b"dnam-creator-signature-test-payload";
            let sig = RustCryptoProvider::ed25519_sign(&seed, message);
            assert!(prov.ed25519_verify(&pubkey, message, &sig).is_ok());

            let mut tampered_sig = sig;
            tampered_sig[0] ^= 0xFF;
            assert!(
                prov.ed25519_verify(&pubkey, message, &tampered_sig)
                    .is_err()
            );
        }
    }
}
