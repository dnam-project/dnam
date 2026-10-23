//! Marketplace Provenance, `Ed25519` Signatures, Hardware Enclaves & Traitor Tracing (`provenance.rs`).
//!
//! Implements the 5-Layer Defense-in-Depth architecture from
//! [`docs/digital_marketplace_and_provenance.md`](../../../docs/digital_marketplace_and_provenance.md):
//! 1. **Preset-Safe RFC 8785 JCS `immutable_header_sha256` & 2-Certificate `Ed25519` Chain**
//!    ([`VerificationStatus`], [`verify_container_provenance`]).
//! 2. **Hardware-Agnostic [`CryptoProvider`] Trait** supporting the built-in `#![no_std]`
//!    [`RustCryptoProvider`] (`ENCRYPTED_ARENA = 0` & software-keyed `ENCRYPTED_ARENA = 1`),
//!    Rockchip RK3582 `OP-TEE` TrustZone, Apple Secure Enclave (`SEP`), and Windows/Linux `TPM 2.0`.
//! 3. **RFC 8439 `ChaCha20` 64-Byte Cache-Line Arena Stream Encryption** (`Lite=168`, `Full=349`,
//!    `Ultra=566` blocks, `0` bytes padding overhead).
//! 4. **Dual-Mechanism Buyer Watermarking (`Test 8H`):**
//!    - **Mechanism A:** 48-channel RG-LRU reciprocal scaling ($\kappa_{b,c}$) + $SO(4)$ SVD head rotation ($\mathbf{Q}_{\text{buyer}}$).
//!    - **Mechanism B:** 4,608-weight Coupled Spread-Spectrum Tardos code surviving 4-buyer collusion + `fp16` truncation.
//! 5. **160-Dimensional Gauge-Invariant Structural Hash** ($\mathbf{f}_{\text{struct}} \in \mathbb{R}^{160}$).

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use zeroize::Zeroizing;

use crate::schema::{AuthorizedDevice, DnamError, DnamJsonHeader, MarketplaceReceipt};

/// Number of nonlinear dilated convolution weights across `Arena 0..2` (`768 + 1536 + 2304 = 4608`).
pub const TARDOS_CONV_WEIGHTS: usize = 4608;

/// Number of RG-LRU complex resonator channels across `Arena 0..2` (`4 blocks × 12 channels = 48`).
pub const RGLRU_TOTAL_CHANNELS: usize = 48;

/// Dimension of the gauge-invariant structural hash vector $\mathbf{f}_{\text{struct}} \in \mathbb{R}^{160}$.
pub const STRUCTURAL_HASH_DIM: usize = 160;

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

// ============================================================================
// Pure `#![no_std]` FIPS 180-4 SHA-256 & Hex Utilities
// ============================================================================

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

#[inline(always)]
const fn sha256_compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    let mut i = 0;
    while i < 16 {
        let j = i * 4;
        w[i] = u32::from_be_bytes([block[j], block[j + 1], block[j + 2], block[j + 3]]);
        i += 1;
    }
    while i < 64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
        i += 1;
    }

    let mut a = state[0];
    let mut b = state[1];
    let mut c = state[2];
    let mut d = state[3];
    let mut e = state[4];
    let mut f = state[5];
    let mut g = state[6];
    let mut h = state[7];

    let mut t = 0;
    while t < 64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let temp1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(SHA256_K[t])
            .wrapping_add(w[t]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);

        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
        t += 1;
    }

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

/// Computes the FIPS 180-4 SHA-256 digest of `data` in `#![no_std]`.
pub fn sha256_digest(data: &[u8]) -> [u8; 32] {
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let (chunks, rem) = data.as_chunks::<64>();
    for blk in chunks {
        sha256_compress(&mut state, blk);
    }
    let mut pad = [0u8; 128];
    pad[..rem.len()].copy_from_slice(rem);
    pad[rem.len()] = 0x80;
    let bit_len = (data.len() as u64) * 8;
    if rem.len() < 56 {
        pad[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let mut blk = [0u8; 64];
        blk.copy_from_slice(&pad[0..64]);
        sha256_compress(&mut state, &blk);
    } else {
        pad[120..128].copy_from_slice(&bit_len.to_be_bytes());
        let mut blk0 = [0u8; 64];
        let mut blk1 = [0u8; 64];
        blk0.copy_from_slice(&pad[0..64]);
        blk1.copy_from_slice(&pad[64..128]);
        sha256_compress(&mut state, &blk0);
        sha256_compress(&mut state, &blk1);
    }

    let mut out = [0u8; 32];
    let mut i = 0;
    while i < 8 {
        let b = state[i].to_be_bytes();
        out[i * 4] = b[0];
        out[i * 4 + 1] = b[1];
        out[i * 4 + 2] = b[2];
        out[i * 4 + 3] = b[3];
        i += 1;
    }
    out
}

/// Encodes a byte slice as lowercase hexadecimal.
pub fn encode_hex(bytes: &[u8]) -> String {
    const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX_CHARS[(b >> 4) as usize] as char);
        out.push(HEX_CHARS[(b & 0x0F) as usize] as char);
    }
    out
}

/// Decodes a fixed-size hexadecimal string into `[u8; N]`.
pub fn decode_hex_array<const N: usize>(hex: &str) -> Result<[u8; N], DnamError> {
    if hex.len() != N * 2 {
        return Err(DnamError::CryptoVerificationFailed(
            "invalid hex string length",
        ));
    }
    let bytes = hex.as_bytes();
    let mut out = [0u8; N];
    let mut i = 0;
    while i < N {
        let hi = hex_nibble(bytes[i * 2])?;
        let lo = hex_nibble(bytes[i * 2 + 1])?;
        out[i] = (hi << 4) | lo;
        i += 1;
    }
    Ok(out)
}

#[inline(always)]
const fn hex_nibble(b: u8) -> Result<u8, DnamError> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(DnamError::CryptoVerificationFailed(
            "invalid hex character in cryptographic field",
        )),
    }
}

// ============================================================================
// RFC 8785 JCS `immutable_header_sha256` & 2-Certificate `Ed25519` Verification
// ============================================================================

/// Recursively canonicalizes a `serde_json::Value` per RFC 8785 JSON Canonicalization Scheme (JCS).
fn canonicalize_jcs_value(val: &serde_json::Value, out: &mut String) {
    match val {
        serde_json::Value::Null => out.push_str("null"),
        serde_json::Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        serde_json::Value::Number(n) => {
            if let Some(f) = n.as_f64()
                && f == 0.0
            {
                out.push('0');
                return;
            }
            out.push_str(&serde_json::to_string(n).unwrap_or_default());
        }
        serde_json::Value::String(s) => {
            out.push_str(&serde_json::to_string(s).unwrap_or_default());
        }
        serde_json::Value::Array(arr) => {
            out.push('[');
            for (i, item) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                canonicalize_jcs_value(item, out);
            }
            out.push(']');
        }
        serde_json::Value::Object(map) => {
            // Collect into BTreeMap for strict UTF-16/UTF-8 lexicographical key sorting
            let mut sorted: BTreeMap<&str, &serde_json::Value> = BTreeMap::new();
            for (k, v) in map {
                sorted.insert(k.as_str(), v);
            }
            out.push('{');
            for (i, (k, v)) in sorted.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(k).unwrap_or_default());
                out.push(':');
                canonicalize_jcs_value(v, out);
            }
            out.push('}');
        }
    }
}

/// Computes the preset-safe RFC 8785 JCS `immutable_header_sha256` digest over a [`DnamJsonHeader`].
///
/// Excludes mutable fields (`presets`, `user_metadata`, `metadata.signatures`,
/// `slimmable_tiers.*.prefix_crc32c`, `controls_schema[*].user_metadata`, and
/// `stems_registry[*].user_metadata`) so users can save DAW presets and tags in-place
/// (`0x40..0x1000`) without invalidating `creator_signature` or `marketplace_receipt`.
pub fn compute_immutable_header_sha256(header: &DnamJsonHeader) -> Result<[u8; 32], DnamError> {
    let mut val = serde_json::to_value(header)
        .map_err(|e| DnamError::InvalidJsonHeader(alloc::format!("{e}")))?;

    if let Some(obj) = val.as_object_mut() {
        obj.remove("presets");
        obj.remove("user_metadata");

        if let Some(meta) = obj.get_mut("metadata").and_then(|m| m.as_object_mut()) {
            meta.remove("signatures");
        }
        if let Some(tiers) = obj
            .get_mut("slimmable_tiers")
            .and_then(|t| t.as_object_mut())
        {
            for tier_key in ["lite", "full", "ultra"] {
                if let Some(t_obj) = tiers.get_mut(tier_key).and_then(|t| t.as_object_mut()) {
                    t_obj.remove("prefix_crc32c");
                }
            }
        }
        if let Some(ctrls) = obj
            .get_mut("controls_schema")
            .and_then(|c| c.as_array_mut())
        {
            for c_item in ctrls {
                if let Some(c_obj) = c_item.as_object_mut() {
                    c_obj.remove("user_metadata");
                }
            }
        }
        if let Some(stems) = obj.get_mut("stems_registry").and_then(|s| s.as_array_mut()) {
            for s_item in stems {
                if let Some(s_obj) = s_item.as_object_mut() {
                    s_obj.remove("user_metadata");
                }
            }
        }
    }

    let mut canonical = String::with_capacity(2048);
    canonicalize_jcs_value(&val, &mut canonical);
    Ok(sha256_digest(canonical.as_bytes()))
}

/// Builds the canonical 64-byte message signed by the creator's `Ed25519` key:
/// `immutable_header_sha256 (32 B) || master_arena_sha256 (32 B)`.
#[inline]
pub fn build_creator_signing_message(
    immutable_header_sha256: &[u8; 32],
    master_arena_sha256: &[u8; 32],
) -> [u8; 64] {
    let mut msg = [0u8; 64];
    msg[0..32].copy_from_slice(immutable_header_sha256);
    msg[32..64].copy_from_slice(master_arena_sha256);
    msg
}

/// Builds the canonical byte payload signed by the marketplace's `Ed25519` key for a [`MarketplaceReceipt`].
pub fn build_marketplace_receipt_signing_message(
    immutable_header_sha256: &[u8; 32],
    master_arena_sha256: &[u8; 32],
    receipt: &MarketplaceReceipt,
) -> Result<Vec<u8>, DnamError> {
    let buyer_arena_sha256 = decode_hex_array::<32>(&receipt.buyer_arena_sha256)?;
    let mut msg = Vec::with_capacity(256);
    msg.extend_from_slice(b"DNAM-RECEIPT-V1:");
    msg.extend_from_slice(immutable_header_sha256);
    msg.extend_from_slice(master_arena_sha256);
    msg.extend_from_slice(&buyer_arena_sha256);
    msg.extend_from_slice(receipt.marketplace_id.as_bytes());
    msg.push(b'|');
    msg.extend_from_slice(receipt.license_id.as_bytes());
    msg.push(b'|');
    msg.extend_from_slice(receipt.buyer_hash.as_bytes());
    msg.push(b'|');
    msg.extend_from_slice(receipt.license_tier.as_bytes());
    msg.push(b'|');
    msg.extend_from_slice(receipt.encryption_mode.as_bytes());
    for dev in &receipt.authorized_devices {
        msg.push(b'|');
        msg.extend_from_slice(dev.oem_vendor.as_bytes());
        msg.push(b':');
        msg.extend_from_slice(dev.model_code.as_bytes());
        msg.push(b':');
        msg.extend_from_slice(dev.device_id_hash.as_bytes());
    }
    Ok(msg)
}

/// Verifies the split-payload `Ed25519` signature chain (`creator_signature` and optional
/// `marketplace_receipt`) and plaintext `Arena 0..2` SHA-256 integrity.
pub fn verify_container_provenance<C: CryptoProvider>(
    header: &DnamJsonHeader,
    has_signatures_flag: bool,
    plaintext_arena: &[u8],
    crypto: &C,
) -> Result<VerificationStatus, DnamError> {
    let Some(ref sigs) = header.metadata.signatures else {
        if has_signatures_flag {
            return Err(DnamError::CryptoVerificationFailed(
                "HAS_SIGNATURES flag is set in preamble, but metadata.signatures is missing",
            ));
        }
        return Ok(VerificationStatus::Unsigned);
    };

    // 1. Verify immutable_header_sha256
    let expected_header_sha = decode_hex_array::<32>(&sigs.immutable_header_sha256)?;
    let computed_header_sha = compute_immutable_header_sha256(header)?;
    if expected_header_sha != computed_header_sha {
        return Err(DnamError::CryptoVerificationFailed(
            "immutable_header_sha256 mismatch (header was tampered with)",
        ));
    }

    // 2. Verify creator_signature over immutable_header_sha256 || master_arena_sha256
    let master_arena_sha = decode_hex_array::<32>(&sigs.master_arena_sha256)?;
    let creator_pubkey = decode_hex_array::<32>(&sigs.creator_signature.pubkey_ed25519)?;
    let creator_sig = decode_hex_array::<64>(&sigs.creator_signature.sig_ed25519)?;
    let creator_msg = build_creator_signing_message(&expected_header_sha, &master_arena_sha);
    crypto.ed25519_verify(&creator_pubkey, &creator_msg, &creator_sig)?;

    // 3. Verify either master_arena_sha256 (direct creator release) or marketplace_receipt
    let computed_arena_sha = sha256_digest(plaintext_arena);

    if let Some(ref receipt) = sigs.marketplace_receipt {
        let buyer_arena_sha = decode_hex_array::<32>(&receipt.buyer_arena_sha256)?;
        if computed_arena_sha != buyer_arena_sha {
            return Err(DnamError::CryptoVerificationFailed(
                "buyer_arena_sha256 mismatch against plaintext tensor arena",
            ));
        }
        // Check authorized_devices allowlist if local_device_id_hash is configured and list is non-empty
        if !receipt.authorized_devices.is_empty()
            && let Some(local_id) = crypto.local_device_id_hash()
            && !receipt
                .authorized_devices
                .iter()
                .any(|d| d.device_id_hash == local_id)
        {
            return Err(DnamError::CryptoVerificationFailed(
                "local hardware device_id_hash is not in receipt.authorized_devices",
            ));
        }
        if let Some(mp_pubkey) = crypto.marketplace_pubkey() {
            let receipt_sig = decode_hex_array::<64>(&receipt.receipt_sig_ed25519)?;
            let receipt_msg = build_marketplace_receipt_signing_message(
                &expected_header_sha,
                &master_arena_sha,
                receipt,
            )?;
            crypto.ed25519_verify(mp_pubkey, &receipt_msg, &receipt_sig)?;
        }
        Ok(VerificationStatus::MarketplaceVerified {
            signer_id: sigs.creator_signature.signer_id.clone(),
            marketplace_id: receipt.marketplace_id.clone(),
            license_id: receipt.license_id.clone(),
            license_tier: receipt.license_tier.clone(),
        })
    } else {
        if computed_arena_sha != master_arena_sha {
            return Err(DnamError::CryptoVerificationFailed(
                "master_arena_sha256 mismatch against plaintext tensor arena",
            ));
        }
        Ok(VerificationStatus::CreatorVerified {
            signer_id: sigs.creator_signature.signer_id.clone(),
            pubkey_ed25519: creator_pubkey,
        })
    }
}

// ============================================================================
// RFC 8439 `ChaCha20` 64-Byte Cache-Line Arena Stream Encryption
// ============================================================================

#[inline(always)]
const fn chacha20_quarter_round(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    state[a] = state[a].wrapping_add(state[b]);
    state[d] ^= state[a];
    state[d] = state[d].rotate_left(16);

    state[c] = state[c].wrapping_add(state[d]);
    state[b] ^= state[c];
    state[b] = state[b].rotate_left(12);

    state[a] = state[a].wrapping_add(state[b]);
    state[d] ^= state[a];
    state[d] = state[d].rotate_left(8);

    state[c] = state[c].wrapping_add(state[d]);
    state[b] ^= state[c];
    state[b] = state[b].rotate_left(7);
}

/// Generates a single 64-byte RFC 8439 `ChaCha20` keystream block for `(cek, nonce_96, block_counter)`.
#[inline]
pub const fn chacha20_block_64b(
    cek: &[u8; 32],
    nonce_96: &[u8; 12],
    block_counter: u32,
) -> [u8; 64] {
    let mut initial = [0u32; 16];
    // "expand 32-byte k"
    initial[0] = 0x6170_7865;
    initial[1] = 0x3320_646e;
    initial[2] = 0x7962_2d32;
    initial[3] = 0x6b20_6574;

    let mut i = 0;
    while i < 8 {
        let j = i * 4;
        initial[4 + i] = u32::from_le_bytes([cek[j], cek[j + 1], cek[j + 2], cek[j + 3]]);
        i += 1;
    }
    initial[12] = block_counter;
    initial[13] = u32::from_le_bytes([nonce_96[0], nonce_96[1], nonce_96[2], nonce_96[3]]);
    initial[14] = u32::from_le_bytes([nonce_96[4], nonce_96[5], nonce_96[6], nonce_96[7]]);
    initial[15] = u32::from_le_bytes([nonce_96[8], nonce_96[9], nonce_96[10], nonce_96[11]]);

    let mut working = initial;
    let mut r = 0;
    while r < 10 {
        // Column rounds
        chacha20_quarter_round(&mut working, 0, 4, 8, 12);
        chacha20_quarter_round(&mut working, 1, 5, 9, 13);
        chacha20_quarter_round(&mut working, 2, 6, 10, 14);
        chacha20_quarter_round(&mut working, 3, 7, 11, 15);
        // Diagonal rounds
        chacha20_quarter_round(&mut working, 0, 5, 10, 15);
        chacha20_quarter_round(&mut working, 1, 6, 11, 12);
        chacha20_quarter_round(&mut working, 2, 7, 8, 13);
        chacha20_quarter_round(&mut working, 3, 4, 9, 14);
        r += 1;
    }

    let mut out = [0u8; 64];
    let mut k = 0;
    while k < 16 {
        let word = working[k].wrapping_add(initial[k]).to_le_bytes();
        let base = k * 4;
        out[base] = word[0];
        out[base + 1] = word[1];
        out[base + 2] = word[2];
        out[base + 3] = word[3];
        k += 1;
    }
    out
}

/// Encrypts or decrypts a 64-byte aligned tensor arena prefix in-place using RFC 8439 `ChaCha20`
/// (`Lite=168`, `Full=349`, `Ultra=566` blocks, `0` bytes padding overhead).
#[inline]
pub fn chacha20_xor_arena_blocks(
    arena: &mut [u8],
    cek: &[u8; 32],
    nonce_96: &[u8; 12],
    start_block: u32,
) -> Result<(), DnamError> {
    if (arena.len() & 0x3F) != 0 {
        return Err(DnamError::UnalignedOffset {
            field: "chacha20_xor_arena_blocks.len",
            offset: arena.len() as u32,
        });
    }
    for (j, chunk) in arena.as_chunks_mut::<64>().0.iter_mut().enumerate() {
        let ks = Zeroizing::new(chacha20_block_64b(
            cek,
            nonce_96,
            start_block.wrapping_add(j as u32),
        ));
        for i in 0..64 {
            chunk[i] ^= ks[i];
        }
    }
    Ok(())
}

// ============================================================================
// Dual-Mechanism Buyer Watermarking & 160-D Structural Hash (`Test 8H`)
// ============================================================================

#[inline]
fn read_f32_le(arena: &[u8], byte_offset: usize) -> f32 {
    f32::from_le_bytes([
        arena[byte_offset],
        arena[byte_offset + 1],
        arena[byte_offset + 2],
        arena[byte_offset + 3],
    ])
}

#[inline]
fn write_f32_le(arena: &mut [u8], byte_offset: usize, val: f32) {
    let b = val.to_le_bytes();
    arena[byte_offset..byte_offset + 4].copy_from_slice(&b);
}

/// Returns the byte offset of the 128-byte `SelectiveRgLru` SoA struct for `(tier, block_idx)`
/// inside `Arena 0..2`:
/// - `tier == 0` (`Lite`, `ch 0..3`): `128 + block_idx * 1024 + 896`
/// - `tier == 1` (`Full`, `ch 4..7`): `10816 + block_idx * 1792 + 1664`
/// - `tier == 2` (`Ultra`, `ch 8..11`): `22400 + block_idx * 2560 + 2432`
#[inline]
pub const fn rglru_soa_byte_offset(tier: usize, block_idx: usize) -> usize {
    match tier {
        0 => 128 + block_idx * 1024 + 896,
        1 => 10816 + block_idx * 1792 + 1664,
        _ => 22400 + block_idx * 2560 + 2432,
    }
}

/// Applies **Mechanism A (Exact Isometric Gauge Watermark)** in-place on a `36,224 B` `Arena 0..2` buffer:
/// 1. Scales $b_{\gamma,b,c}$ by $\kappa_{b,c} \in [0.985, 1.015]$ and $(c_{\text{real},b,c}, c_{\text{imag},b,c})$
///    by $\kappa_{b,c}^{-1}$ across all 48 RG-LRU channels (`0.0000 dB` audio distortion).
/// 2. Rotates the SVD stereo output head basis $\mathbf{W}_{\text{head},1..4}$ by $\mathbf{Q}_{\text{buyer}}^\top \in SO(4)$
///    and the SVD control subspace projection matrix $\mathbf{P}_{\text{svd}}$ by $\mathbf{Q}_{\text{buyer}}$.
pub fn apply_isometric_gauge_watermark(
    arena: &mut [u8],
    kappa_48: &[f32; RGLRU_TOTAL_CHANNELS],
    q_buyer_4x4: &[[f32; 4]; 4],
) -> Result<(), DnamError> {
    if arena.len() < 36_224 {
        return Err(DnamError::BufferTooSmall {
            expected: 36_224,
            actual: arena.len(),
        });
    }

    // 1. 48-channel RG-LRU reciprocal scaling gauge (b_gamma * kappa, c_real / kappa, c_imag / kappa)
    for tier in 0..3 {
        for blk in 0..4 {
            let base = rglru_soa_byte_offset(tier, blk);
            for lane in 0..4 {
                let ch_global = tier * 16 + blk * 4 + lane;
                let kappa = kappa_48[ch_global];
                let inv_kappa = 1.0 / kappa;

                // In SelectiveRgLru SoA (32 floats = 128 B):
                // [0..4]: r_low, [4..8]: dr_span, [8..12]: theta0_rad, [12..16]: b_gamma_norm,
                // [16..20]: w_sel, [20..24]: c_real, [24..28]: c_imag, [28..32]: u_rot_init
                let b_gamma_off = base + (12 + lane) * 4;
                let c_real_off = base + (20 + lane) * 4;
                let c_imag_off = base + (24 + lane) * 4;

                let b_gamma = read_f32_le(arena, b_gamma_off);
                let c_real = read_f32_le(arena, c_real_off);
                let c_imag = read_f32_le(arena, c_imag_off);

                write_f32_le(arena, b_gamma_off, b_gamma * kappa);
                write_f32_le(arena, c_real_off, c_real * inv_kappa);
                write_f32_le(arena, c_imag_off, c_imag * inv_kappa);
            }
        }
    }

    // 2. Rotate SVD stereo head basis W_head[1..5] (each mode has 2*16*4 = 128 floats = 512 B)
    // across Arena 0 (4224..6784), Arena 1 (17984..20544), and Arena 2 (32640..35200):
    // W_head_buyer[k, i] = sum_{m=0..3} Q_buyer[m][k] * W_head_master[m, i]  (i.e. Q^T @ W_head)
    let head_arena_offsets = [4224_usize, 17984_usize, 32640_usize];
    for &head_base in &head_arena_offsets {
        // Mode 0 (base head W_head,0) is at head_base + 0..512; Modes 1..=4 are at head_base + 512..2560
        for elem_idx in 0..128 {
            let mut old_vals = [0.0_f32; 4];
            for m in 0..4 {
                let off = head_base + (m + 1) * 512 + elem_idx * 4;
                old_vals[m] = read_f32_le(arena, off);
            }
            for k in 0..4 {
                let mut rotated = 0.0_f32;
                for m in 0..4 {
                    rotated += q_buyer_4x4[m][k] * old_vals[m];
                }
                let off = head_base + (k + 1) * 512 + elem_idx * 4;
                write_f32_le(arena, off, rotated);
            }
        }
    }

    // Also rotate b_head[1..5] in Arena 0 CL1 (bytes 64..104: 5 modes * 2 stereo channels):
    for lr in 0..2 {
        let mut old_b = [0.0_f32; 4];
        for m in 0..4 {
            let off = 64 + ((m + 1) * 2 + lr) * 4;
            old_b[m] = read_f32_le(arena, off);
        }
        for k in 0..4 {
            let mut rotated = 0.0_f32;
            for m in 0..4 {
                rotated += q_buyer_4x4[m][k] * old_b[m];
            }
            let off = 64 + ((k + 1) * 2 + lr) * 4;
            write_f32_le(arena, off, rotated);
        }
    }

    // 3. Rotate P_svd (16 rows x 4 cols = 64 floats = 256 B) in Arena 0 control_bus_and_clamp_table
    // Global control bus weights start at 7040 + 1024 = 8064; P_svd occupies 8064 + 896 .. 8064 + 1152 (256 B):
    let p_svd_base = 8960_usize;
    for row in 0..16 {
        let mut old_row = [0.0_f32; 4];
        for m in 0..4 {
            old_row[m] = read_f32_le(arena, p_svd_base + (row * 4 + m) * 4);
        }
        for k in 0..4 {
            let mut rotated = 0.0_f32;
            for m in 0..4 {
                rotated += old_row[m] * q_buyer_4x4[m][k];
            }
            write_f32_le(arena, p_svd_base + (row * 4 + k) * 4, rotated);
        }
    }

    Ok(())
}

/// Extracts the **160-Dimensional Gauge-Invariant Structural Vector** $\mathbf{f}_{\text{struct}} \in \mathbb{R}^{160}$
/// directly from a `36,224 B` `Arena 0..2` buffer:
/// - `[0..48]`: 48 RG-LRU invariant composite pole signatures $r_{\text{low},b,c} + 0.5\,\Delta r_{b,c} + 0.1\,\theta_{0,b,c}$,
/// - `[48..96]`: 48 gauge-invariant real products $I_{\text{real},b,c} = b_{\gamma,b,c} \cdot c_{\text{real},b,c}$,
/// - `[96..144]`: 48 gauge-invariant imaginary products $I_{\text{imag},b,c} = b_{\gamma,b,c} \cdot c_{\text{imag},b,c}$,
/// - `[144..152]`: 8 Core 2 `M-UFDN` base delay anchors $\tau_{0,m}$,
/// - `[152..160]`: 8 base stereo output head $\mathbf{W}_{\text{head},0}$ block energy invariants.
pub fn extract_structural_hash_160d(arena: &[u8]) -> Result<[f64; STRUCTURAL_HASH_DIM], DnamError> {
    if arena.len() < 36_224 {
        return Err(DnamError::BufferTooSmall {
            expected: 36_224,
            actual: arena.len(),
        });
    }
    let mut vec160 = [0.0_f64; STRUCTURAL_HASH_DIM];

    for tier in 0..3 {
        for blk in 0..4 {
            let base = rglru_soa_byte_offset(tier, blk);
            for lane in 0..4 {
                let idx = tier * 16 + blk * 4 + lane;
                let r_low = read_f32_le(arena, base + lane * 4) as f64;
                let dr = read_f32_le(arena, base + (4 + lane) * 4) as f64;
                let theta = read_f32_le(arena, base + (8 + lane) * 4) as f64;
                let b_gamma = read_f32_le(arena, base + (12 + lane) * 4) as f64;
                let c_real = read_f32_le(arena, base + (20 + lane) * 4) as f64;
                let c_imag = read_f32_le(arena, base + (24 + lane) * 4) as f64;

                vec160[idx] = r_low + 0.5 * dr + 0.1 * theta;
                vec160[48 + idx] = b_gamma * c_real;
                vec160[96 + idx] = b_gamma * c_imag;
            }
        }
    }

    // 8 M-UFDN base delays (4 in Arena 0 at 6784, 4 in Arena 1 at 20544)
    for lane in 0..4 {
        vec160[144 + lane] = read_f32_le(arena, 6784 + lane * 4) as f64;
        vec160[148 + lane] = read_f32_le(arena, 20544 + lane * 4) as f64;
    }

    // 8 Base stereo head W_head,0 energy norms (unaffected by Q_buyer rotation on modes 1..4)
    for seg in 0..8 {
        let mut energy = 0.0_f64;
        for i in 0..16 {
            let v = read_f32_le(arena, 4224 + (seg * 16 + i) * 4) as f64;
            energy += v * v;
        }
        vec160[152 + seg] = energy;
    }

    Ok(vec160)
}

/// Computes the cosine similarity $\cos(\mathbf{f}_1, \mathbf{f}_2) \in [-1, +1]$ between two 160-D structural hashes.
pub fn structural_cosine_similarity(
    a: &[f64; STRUCTURAL_HASH_DIM],
    b: &[f64; STRUCTURAL_HASH_DIM],
) -> f64 {
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    let mut i = 0;
    while i < STRUCTURAL_HASH_DIM {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
        i += 1;
    }
    let denom = crate::schema::cold_sqrt_f64(na * nb);
    if denom <= 1e-30 { 0.0 } else { dot / denom }
}

/// Visits the byte offsets of all $N_{\text{conv}} = 4,608$ nonlinear dilated convolution `f32` weights
/// across `Arena 0` (`768` floats), `Arena 1` (`1,536` floats), and `Arena 2` (`2,304` floats).
fn for_each_conv_weight_offset<F: FnMut(usize, usize)>(mut f: F) {
    let mut idx = 0_usize;
    // Arena 0 (Lite): 4 blocks * 1024 B/block starting at 128; Conv0 is 0..384, Conv1 is 448..832
    for blk in 0..4 {
        let base = 128 + blk * 1024;
        for w in 0..96 {
            f(idx, base + w * 4);
            idx += 1;
        }
        for w in 0..96 {
            f(idx, base + 448 + w * 4);
            idx += 1;
        }
    }
    // Arena 1 (Full Delta): 4 blocks * 1792 B/block starting at 10816; Conv0 is 0..768, Conv1 is 832..1600
    for blk in 0..4 {
        let base = 10816 + blk * 1792;
        for w in 0..192 {
            f(idx, base + w * 4);
            idx += 1;
        }
        for w in 0..192 {
            f(idx, base + 832 + w * 4);
            idx += 1;
        }
    }
    // Arena 2 (Ultra Delta): 4 blocks * 2560 B/block starting at 22400; Conv0 is 0..1152, Conv1 is 1216..2368
    for blk in 0..4 {
        let base = 22400 + blk * 2560;
        for w in 0..288 {
            f(idx, base + w * 4);
            idx += 1;
        }
        for w in 0..288 {
            f(idx, base + 1216 + w * 4);
            idx += 1;
        }
    }
}

/// Deterministically generates a 4,608-chip Tardos binary spread-spectrum watermark code
/// $\mathbf{c}^{(\text{buyer})} \in \{-1.0, +1.0\}^{4608}$ from `(marketplace_secret, model_id, license_id)`.
pub fn generate_tardos_code_4608(
    marketplace_secret: &[u8; 32],
    model_id: &str,
    license_id: &str,
) -> [f32; TARDOS_CONV_WEIGHTS] {
    let mut seed_input = Vec::with_capacity(64 + model_id.len() + license_id.len());
    seed_input.extend_from_slice(b"DNAM-TARDOS-V1:");
    seed_input.extend_from_slice(marketplace_secret);
    seed_input.extend_from_slice(model_id.as_bytes());
    seed_input.push(b':');
    seed_input.extend_from_slice(license_id.as_bytes());
    let buyer_seed = sha256_digest(&seed_input);
    let nonce = [0u8; 12];

    let mut code = [1.0_f32; TARDOS_CONV_WEIGHTS];
    let mut bit_idx = 0_usize;
    let mut blk_counter = 0_u32;

    while bit_idx < TARDOS_CONV_WEIGHTS {
        let stream = chacha20_block_64b(&buyer_seed, &nonce, blk_counter);
        blk_counter += 1;
        for &byte in &stream {
            for b in 0..8 {
                if bit_idx < TARDOS_CONV_WEIGHTS {
                    code[bit_idx] = if ((byte >> b) & 1) == 1 { 1.0 } else { -1.0 };
                    bit_idx += 1;
                }
            }
        }
    }
    code
}

/// Embeds **Mechanism B (4,608-Weight Spread-Spectrum Tardos Watermark)** into `Arena 0..2`:
/// $$W_{\text{conv},i}^{(\text{buyer})} = W_{\text{conv},i}^{(\text{master})} + \epsilon \, c_i^{(\text{buyer})}$$
pub fn apply_tardos_watermark_4608(
    arena: &mut [u8],
    code: &[f32; TARDOS_CONV_WEIGHTS],
    epsilon: f32,
) -> Result<(), DnamError> {
    if arena.len() < 36_224 {
        return Err(DnamError::BufferTooSmall {
            expected: 36_224,
            actual: arena.len(),
        });
    }
    for_each_conv_weight_offset(|idx, byte_off| {
        let w = read_f32_le(arena, byte_off);
        write_f32_le(arena, byte_off, w + epsilon * code[idx]);
    });
    Ok(())
}

/// Computes the Tardos matched-filter Z-score ($\sigma$) of a suspect `Arena 0..2` against
/// `master_arena` and a candidate buyer's `code`:
/// $$Z = \frac{1}{\sqrt{4608}} \sum_{i=0}^{4607} c_i^{(\text{buyer})} \operatorname{sign}\!\big(W_{\text{conv},i}^{(\text{suspect})} - W_{\text{conv},i}^{(\text{master})}\big)$$
pub fn score_tardos_watermark_4608(
    master_arena: &[u8],
    suspect_arena: &[u8],
    code: &[f32; TARDOS_CONV_WEIGHTS],
) -> Result<f64, DnamError> {
    if master_arena.len() < 36_224 || suspect_arena.len() < 36_224 {
        return Err(DnamError::BufferTooSmall {
            expected: 36_224,
            actual: master_arena.len().min(suspect_arena.len()),
        });
    }
    let mut corr = 0.0_f64;
    for_each_conv_weight_offset(|idx, byte_off| {
        let w_m = read_f32_le(master_arena, byte_off);
        let w_s = read_f32_le(suspect_arena, byte_off);
        let diff = w_s - w_m;
        let sign = if diff > 0.0 {
            1.0_f64
        } else if diff < 0.0 {
            -1.0_f64
        } else {
            0.0_f64
        };
        corr += (code[idx] as f64) * sign;
    });
    // 1.0 / sqrt(4608.0) = 1.0 / 67.88225099390857
    Ok(corr * (1.0 / 67.882_250_993_908_57))
}
