//! RFC 8785 JCS `immutable_header_sha256` & 2-Certificate `Ed25519` Verification.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::container::{DNAM_MAGIC, DNCK_MAGIC, DnamPreamble, crc32c_update};
use crate::schema::{
    DnamError, DnamJsonHeader, DnamProtectionProfile, MarketplaceReceipt, OemDeviceCertificate,
    SchemaVersion,
};

use super::chacha20::chacha20_xor_arena_blocks;
use super::provider::{CryptoProvider, VerificationStatus};
use super::sha256::{decode_hex_array, encode_hex, sha256_digest};

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

/// Projects a serialized [`DnamJsonHeader`] `serde_json::Value` onto the canonical immutable
/// field set for `schema_version`, stripping mutable DAW/user fields and any additive fields
/// introduced in minor schema versions newer than `schema_version`.
pub fn project_immutable_header_for_schema_version(
    val: &mut serde_json::Value,
    schema_version: SchemaVersion,
) -> Result<(), DnamError> {
    schema_version.ensure_supported()?;

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

        // Future minor-version projection hooks (e.g. `if schema_version.minor < 1 { ... }`)
        // strip additive `#[serde(default)]` fields introduced after `schema_version` so
        // containers signed under an earlier `1.x` minor version retain their exact JCS digest.
    }

    Ok(())
}

/// Computes the preset-safe RFC 8785 JCS `immutable_header_sha256` digest over a [`DnamJsonHeader`]
/// projected to a specific [`SchemaVersion`].
pub fn compute_immutable_header_sha256_for_version(
    header: &DnamJsonHeader,
    schema_version: SchemaVersion,
) -> Result<[u8; 32], DnamError> {
    let mut val = serde_json::to_value(header)
        .map_err(|e| DnamError::InvalidJsonHeader(alloc::format!("{e}")))?;
    project_immutable_header_for_schema_version(&mut val, schema_version)?;

    let mut canonical = String::with_capacity(2048);
    canonicalize_jcs_value(&val, &mut canonical);
    Ok(sha256_digest(canonical.as_bytes()))
}

/// Computes the preset-safe RFC 8785 JCS `immutable_header_sha256` digest over a [`DnamJsonHeader`]
/// using the header's declared [`SchemaVersion`].
///
/// Excludes mutable fields (`presets`, `user_metadata`, `metadata.signatures`,
/// `slimmable_tiers.*.prefix_crc32c`, `controls_schema[*].user_metadata`, and
/// `stems_registry[*].user_metadata`) so users can save DAW presets and tags in-place
/// (`0x40..0x1000`) without invalidating `creator_signature` or `marketplace_receipt`.
pub fn compute_immutable_header_sha256(header: &DnamJsonHeader) -> Result<[u8; 32], DnamError> {
    let schema_version = header.schema_version()?;
    compute_immutable_header_sha_inner(header, schema_version)
}

#[inline]
fn compute_immutable_header_sha_inner(
    header: &DnamJsonHeader,
    schema_version: SchemaVersion,
) -> Result<[u8; 32], DnamError> {
    compute_immutable_header_sha256_for_version(header, schema_version)
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

/// Builds the canonical signing message for a factory-provisioned [`OemDeviceCertificate`]:
/// `b"DNAM-OEM-CERT-V1:" || oem_vendor || b"|" || model_code || b"|" || device_id_hash || b"|" || pubkey_x25519_bytes`.
pub fn build_oem_device_cert_signing_message(
    oem_vendor: &str,
    model_code: &str,
    device_id_hash: &str,
    pubkey_x25519_bytes: &[u8; 32],
) -> Vec<u8> {
    let mut msg = Vec::with_capacity(
        17 + oem_vendor.len() + 1 + model_code.len() + 1 + device_id_hash.len() + 1 + 32,
    );
    msg.extend_from_slice(b"DNAM-OEM-CERT-V1:");
    msg.extend_from_slice(oem_vendor.as_bytes());
    msg.push(b'|');
    msg.extend_from_slice(model_code.as_bytes());
    msg.push(b'|');
    msg.extend_from_slice(device_id_hash.as_bytes());
    msg.push(b'|');
    msg.extend_from_slice(pubkey_x25519_bytes);
    msg
}

/// Verifies a factory-provisioned [`OemDeviceCertificate`] against the manufacturer's pinned
/// 32-byte `Ed25519` public key, returning the verified 32-byte TrustZone `X25519` public key.
pub fn verify_oem_device_certificate<C: CryptoProvider>(
    cert: &OemDeviceCertificate,
    trusted_oem_pubkey_ed25519: &[u8; 32],
    crypto: &C,
) -> Result<[u8; 32], DnamError> {
    let raw_id = decode_hex_array::<16>(&cert.device_id_hash)?;
    let normalized_id = encode_hex(&raw_id);
    let pubkey_x25519 = decode_hex_array::<32>(&cert.pubkey_x25519)?;
    let oem_sig = decode_hex_array::<64>(&cert.oem_sig_ed25519)?;
    let msg = build_oem_device_cert_signing_message(
        &cert.oem_vendor,
        &cert.model_code,
        &normalized_id,
        &pubkey_x25519,
    );
    crypto.ed25519_verify(trusted_oem_pubkey_ed25519, &msg, &oem_sig)?;
    Ok(pubkey_x25519)
}

/// Enforces the creator-signed [`crate::schema::DistributionPolicy`] (`metadata.distribution_policy`)
/// against the container's active [`DnamProtectionProfile`] and `marketplace_receipt.authorized_devices`.
pub fn enforce_distribution_policy(
    header: &DnamJsonHeader,
    active_profile: DnamProtectionProfile,
) -> Result<(), DnamError> {
    let Some(ref policy) = header.metadata.distribution_policy else {
        return Ok(());
    };

    if !policy.is_profile_allowed(active_profile) {
        return Err(DnamError::CryptoVerificationFailed(
            "container protection profile is not permitted by creator distribution_policy",
        ));
    }

    if !policy.allowed_oem_vendors.is_empty() {
        if active_profile == DnamProtectionProfile::Open {
            return Err(DnamError::CryptoVerificationFailed(
                "creator distribution_policy restricts model to specific OEM hardware vendors",
            ));
        }

        let maybe_receipt = header
            .metadata
            .signatures
            .as_ref()
            .and_then(|s| s.marketplace_receipt.as_ref());

        if let Some(receipt) = maybe_receipt {
            if receipt.authorized_devices.is_empty()
                && active_profile == DnamProtectionProfile::DeviceAllowlisted
            {
                return Err(DnamError::CryptoVerificationFailed(
                    "creator distribution_policy requires authorized OEM devices",
                ));
            }
            for dev in &receipt.authorized_devices {
                if !policy.is_oem_device_allowed(&dev.oem_vendor, &dev.model_code) {
                    return Err(DnamError::CryptoVerificationFailed(
                        "authorized device OEM vendor/model is not permitted by creator distribution_policy",
                    ));
                }
            }
        } else if active_profile == DnamProtectionProfile::DeviceAllowlisted {
            return Err(DnamError::CryptoVerificationFailed(
                "creator distribution_policy requires authorized OEM devices",
            ));
        }
    }

    Ok(())
}

/// Magic bytes for a `.dnam-seal` OEM hardware envelope (`0x4C534E44` little-endian).
pub const DNSL_MAGIC: [u8; 4] = *b"DNSL";

/// Flag bit 0 in [`DnamSealEnvelopePreamble::flags`]: full inner `.dnam` container + nested `Arena 0..2` encrypted with `ChaCha20`.
pub const FLAG_SEAL_CHACHA20_FULL: u16 = 1 << 0;

/// Maximum number of hardware recipient slots in a `.dnam-seal` (`b"DNSL"`) envelope (`16`).
pub const MAX_SEAL_RECIPIENTS: usize = 16;

/// Cache Line 0 (`64 B`, `#[repr(C, align(64))]`) of a `.dnam-seal` (`b"DNSL"`) hardware envelope.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DnamSealEnvelopePreamble {
    /// Magic bytes (`b"DNSL"` / `0x4C534E44`).
    pub magic: [u8; 4],
    /// Envelope major version (`1`).
    pub version_major: u16,
    /// Envelope minor version (`0`).
    pub version_minor: u16,
    /// Envelope flags (`FLAG_SEAL_CHACHA20_FULL`).
    pub flags: u16,
    /// Number of [`DnamSealRecipientSlot`] cache lines following Cache Line 0 (`1..=16`).
    pub num_recipients: u8,
    /// Reserved byte (`0`).
    pub reserved_0: u8,
    /// Byte length of the inner `.dnam` container (`% 64 == 0`, e.g., `40,320 B`).
    pub inner_dnam_byte_len: u32,
    /// Raw 32-byte ephemeral `X25519` public key for `HPKE` key agreement.
    pub ephemeral_pubkey_x25519: [u8; 32],
    /// Raw 12-byte (`96-bit`) outer `ChaCha20` nonce.
    pub chacha20_nonce_96: [u8; 12],
    /// Castagnoli `CRC32C` over `CL 0..=N+1` (`[0..(2 + N) * 64]`) with `[0x3C..0x40]` zeroed.
    pub envelope_crc32c: u32,
}

impl DnamSealEnvelopePreamble {
    /// Serializes this 64-byte preamble into little-endian bytes.
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        out[0..4].copy_from_slice(&self.magic);
        out[4..6].copy_from_slice(&self.version_major.to_le_bytes());
        out[6..8].copy_from_slice(&self.version_minor.to_le_bytes());
        out[8..10].copy_from_slice(&self.flags.to_le_bytes());
        out[10] = self.num_recipients;
        out[11] = self.reserved_0;
        out[12..16].copy_from_slice(&self.inner_dnam_byte_len.to_le_bytes());
        out[16..48].copy_from_slice(&self.ephemeral_pubkey_x25519);
        out[48..60].copy_from_slice(&self.chacha20_nonce_96);
        out[60..64].copy_from_slice(&self.envelope_crc32c.to_le_bytes());
        out
    }

    /// Parses and validates a 64-byte `.dnam-seal` (`b"DNSL"`) preamble from `bytes`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DnamError> {
        if bytes.len() < 64 {
            return Err(DnamError::BufferTooSmall {
                expected: 64,
                actual: bytes.len(),
            });
        }
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&bytes[0..4]);
        if magic != DNSL_MAGIC {
            return Err(DnamError::InvalidMagic(magic));
        }
        let version_major = u16::from_le_bytes([bytes[4], bytes[5]]);
        let version_minor = u16::from_le_bytes([bytes[6], bytes[7]]);
        if version_major != 1 {
            return Err(DnamError::UnsupportedVersion {
                major: version_major,
                minor: version_minor,
            });
        }
        let flags = u16::from_le_bytes([bytes[8], bytes[9]]);
        let num_recipients = bytes[10];
        let reserved_0 = bytes[11];
        let inner_dnam_byte_len = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        let mut ephemeral_pubkey_x25519 = [0u8; 32];
        ephemeral_pubkey_x25519.copy_from_slice(&bytes[16..48]);
        let mut chacha20_nonce_96 = [0u8; 12];
        chacha20_nonce_96.copy_from_slice(&bytes[48..60]);
        let envelope_crc32c = u32::from_le_bytes([bytes[60], bytes[61], bytes[62], bytes[63]]);

        Ok(Self {
            magic,
            version_major,
            version_minor,
            flags,
            num_recipients,
            reserved_0,
            inner_dnam_byte_len,
            ephemeral_pubkey_x25519,
            chacha20_nonce_96,
            envelope_crc32c,
        })
    }
}

/// Cache Line `1..=N` (`64 B` per registered hardware device, `#[repr(C, align(64))]`) in a `.dnam-seal` envelope.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DnamSealRecipientSlot {
    /// Raw 16-byte (`128-bit`) binary `device_id_hash`.
    pub device_id_hash: [u8; 16],
    /// Raw 48-byte `X25519-HKDF-SHA256` + `ChaCha20-Poly1305` wrapped `CEK` (`32 B` ciphertext + `16 B` tag).
    pub wrapped_cek: [u8; 48],
}

impl DnamSealRecipientSlot {
    /// Serializes this 64-byte recipient slot.
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        out[0..16].copy_from_slice(&self.device_id_hash);
        out[16..64].copy_from_slice(&self.wrapped_cek);
        out
    }

    /// Parses a 64-byte recipient slot from `bytes`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DnamError> {
        if bytes.len() < 64 {
            return Err(DnamError::BufferTooSmall {
                expected: 64,
                actual: bytes.len(),
            });
        }
        let mut device_id_hash = [0u8; 16];
        device_id_hash.copy_from_slice(&bytes[0..16]);
        let mut wrapped_cek = [0u8; 48];
        wrapped_cek.copy_from_slice(&bytes[16..64]);
        Ok(Self {
            device_id_hash,
            wrapped_cek,
        })
    }
}

fn compute_seal_envelope_crc32c(header_region: &[u8]) -> u32 {
    let mut crc = crc32c_update(0, &header_region[0..60]);
    crc = crc32c_update(crc, &[0u8; 4]);
    crc32c_update(crc, &header_region[64..])
}

fn derive_nested_inner_arena_key_and_nonce(
    cek_outer: &[u8; 32],
    nonce_outer: &[u8; 12],
) -> ([u8; 32], [u8; 12]) {
    let mut key_msg = [0u8; 25 + 32];
    key_msg[0..25].copy_from_slice(b"DNAM-SEAL-INNER-ARENA-V1:");
    key_msg[25..57].copy_from_slice(cek_outer);
    let cek_inner = sha256_digest(&key_msg);

    let mut nonce_msg = [0u8; 25 + 12];
    nonce_msg[0..25].copy_from_slice(b"DNAM-SEAL-INNER-NONCE-V1:");
    nonce_msg[25..37].copy_from_slice(nonce_outer);
    let nonce_digest = sha256_digest(&nonce_msg);
    let mut nonce_inner = [0u8; 12];
    nonce_inner.copy_from_slice(&nonce_digest[0..12]);
    (cek_inner, nonce_inner)
}

fn build_seal_envelope_signing_message(
    preamble_zero_crc_bytes: &[u8; 64],
    slots_bytes: &[u8],
    encrypted_payload: &[u8],
) -> Vec<u8> {
    let payload_sha = sha256_digest(encrypted_payload);
    let mut msg = Vec::with_capacity(13 + 64 + slots_bytes.len() + 32);
    msg.extend_from_slice(b"DNAM-SEAL-V1:");
    msg.extend_from_slice(preamble_zero_crc_bytes);
    msg.extend_from_slice(slots_bytes);
    msg.extend_from_slice(&payload_sha);
    msg
}

/// Seals a 64-byte aligned `.dnam` container into a `.dnam-seal` (`b"DNSL"`) hardware envelope
/// (`DnamProtectionProfile::EnclaveSealed`), verifying the creator's [`crate::schema::DistributionPolicy`]
/// and every recipient's factory [`OemDeviceCertificate`] before encrypting both `Arena 0..2`
/// (nested inner arena pass) and the entire `.dnam` container (outer full-container pass).
#[cfg(feature = "crypto")]
pub fn seal_dnam_container(
    inner_dnam_bytes: &[u8],
    recipient_certs: &[OemDeviceCertificate],
    trusted_oem_pubkeys: &[(&str, [u8; 32])],
    marketplace_seed_ed25519: &[u8; 32],
    ephemeral_secret_x25519: &[u8; 32],
    cek_32: &[u8; 32],
    chacha20_nonce_96: &[u8; 12],
) -> Result<Vec<u8>, DnamError> {
    use super::provider::RustCryptoProvider;

    if inner_dnam_bytes.len() < 4096 || (inner_dnam_bytes.len() & 0x3F) != 0 {
        return Err(DnamError::UnalignedOffset {
            field: "inner_dnam_bytes.len",
            offset: inner_dnam_bytes.len() as u32,
        });
    }
    if recipient_certs.is_empty() || recipient_certs.len() > MAX_SEAL_RECIPIENTS {
        return Err(DnamError::CryptoVerificationFailed(
            "seal_dnam_container requires 1..=16 recipient OemDeviceCertificates",
        ));
    }

    // 1. Parse and validate the inner .dnam preamble & JSON header
    let inner_preamble = DnamPreamble::from_bytes(inner_dnam_bytes)?;
    let arena_off = inner_preamble.tensor_arena_offset as usize;
    let arena_len = inner_preamble.tensor_arena_byte_len as usize;
    let json_len = inner_preamble.json_header_byte_len as usize;
    if arena_off < 64 + json_len || inner_dnam_bytes.len() < arena_off + arena_len {
        return Err(DnamError::BufferTooSmall {
            expected: arena_off + arena_len,
            actual: inner_dnam_bytes.len(),
        });
    }
    let inner_header: DnamJsonHeader = serde_json::from_slice(&inner_dnam_bytes[64..64 + json_len])
        .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?;
    inner_header.validate()?;

    // 2. Enforce creator-signed DistributionPolicy for EnclaveSealed and every recipient certificate
    enforce_distribution_policy(&inner_header, DnamProtectionProfile::EnclaveSealed)?;
    if let Some(ref policy) = inner_header.metadata.distribution_policy {
        for cert in recipient_certs {
            if !policy.is_oem_device_allowed(&cert.oem_vendor, &cert.model_code) {
                return Err(DnamError::CryptoVerificationFailed(
                    "recipient OemDeviceCertificate vendor/model is not permitted by creator distribution_policy",
                ));
            }
        }
    }

    // 3. Verify each recipient's factory OemDeviceCertificate and wrap cek_32 into a 64B slot
    let verify_provider = RustCryptoProvider::new();
    let mut slots_bytes = Vec::with_capacity(recipient_certs.len() * 64);

    for cert in recipient_certs {
        let (_, oem_pk) = trusted_oem_pubkeys
            .iter()
            .find(|(vendor, _)| vendor.eq_ignore_ascii_case(&cert.oem_vendor))
            .ok_or(DnamError::CryptoVerificationFailed(
                "unknown or untrusted OEM vendor in OemDeviceCertificate",
            ))?;
        let device_pk_x25519 = verify_oem_device_certificate(cert, oem_pk, &verify_provider)?;
        let raw_id_16 = decode_hex_array::<16>(&cert.device_id_hash)?;
        let normalized_id = encode_hex(&raw_id_16);
        let wrapped_cek = RustCryptoProvider::wrap_device_cek(
            ephemeral_secret_x25519,
            &device_pk_x25519,
            &normalized_id,
            cek_32,
        )?;
        let slot = DnamSealRecipientSlot {
            device_id_hash: raw_id_16,
            wrapped_cek,
        };
        slots_bytes.extend_from_slice(&slot.to_bytes());
    }

    // 4. Nested Double Encryption:
    //    Pass A: Encrypt Arena 0..2 with domain-separated inner arena key (cek_inner, nonce_inner)
    //    Pass B: Encrypt entire inner .dnam container (preamble + JSON + Arena 0..2) with (cek_32, chacha20_nonce_96)
    let mut encrypted_payload = inner_dnam_bytes.to_vec();
    let ultra_end = (inner_preamble.arena_offset_ultra_end as usize).min(arena_len);
    let (cek_inner, nonce_inner) =
        derive_nested_inner_arena_key_and_nonce(cek_32, chacha20_nonce_96);
    chacha20_xor_arena_blocks(
        &mut encrypted_payload[arena_off..arena_off + ultra_end],
        &cek_inner,
        &nonce_inner,
        0,
    )?;
    chacha20_xor_arena_blocks(&mut encrypted_payload, cek_32, chacha20_nonce_96, 0)?;

    // 5. Build preamble, sign envelope, and compute Castagnoli CRC32C
    let epk_x25519 = RustCryptoProvider::x25519_pubkey_from_secret(ephemeral_secret_x25519);
    let mut preamble = DnamSealEnvelopePreamble {
        magic: DNSL_MAGIC,
        version_major: 1,
        version_minor: 0,
        flags: FLAG_SEAL_CHACHA20_FULL | crate::container::FLAG_ENCRYPTED_ARENA,
        num_recipients: recipient_certs.len() as u8,
        reserved_0: 0,
        inner_dnam_byte_len: inner_dnam_bytes.len() as u32,
        ephemeral_pubkey_x25519: epk_x25519,
        chacha20_nonce_96: *chacha20_nonce_96,
        envelope_crc32c: 0,
    };
    let preamble_zero_crc = preamble.to_bytes();
    let sign_msg =
        build_seal_envelope_signing_message(&preamble_zero_crc, &slots_bytes, &encrypted_payload);
    let marketplace_sig = RustCryptoProvider::ed25519_sign(marketplace_seed_ed25519, &sign_msg);

    let header_total_len = (2 + recipient_certs.len()) * 64;
    let mut out = Vec::with_capacity(header_total_len + encrypted_payload.len());
    out.extend_from_slice(&preamble_zero_crc);
    out.extend_from_slice(&slots_bytes);
    out.extend_from_slice(&marketplace_sig);
    out.extend_from_slice(&encrypted_payload);

    preamble.envelope_crc32c = compute_seal_envelope_crc32c(&out[0..header_total_len]);
    out[60..64].copy_from_slice(&preamble.envelope_crc32c.to_le_bytes());
    Ok(out)
}

/// Unseals a `.dnam-seal` (`b"DNSL"`) hardware envelope using `crypto`, verifying `envelope_crc32c`
/// and `marketplace_sig_ed25519` before unwrapping the recipient slot `CEK` and recovering the inner `.dnam` bytes.
pub fn unseal_dnam_container<C: CryptoProvider>(
    sealed_bytes: &[u8],
    crypto: &C,
) -> Result<Vec<u8>, DnamError> {
    let preamble = DnamSealEnvelopePreamble::from_bytes(sealed_bytes)?;
    let n = preamble.num_recipients as usize;
    if n == 0 || n > MAX_SEAL_RECIPIENTS {
        return Err(DnamError::CryptoVerificationFailed(
            "invalid num_recipients in .dnam-seal preamble",
        ));
    }
    let payload_len = preamble.inner_dnam_byte_len as usize;
    if payload_len < 4096 || (payload_len & 0x3F) != 0 {
        return Err(DnamError::UnalignedOffset {
            field: "inner_dnam_byte_len",
            offset: preamble.inner_dnam_byte_len,
        });
    }
    let header_total_len = (2 + n) * 64;
    let expected_total = header_total_len + payload_len;
    if sealed_bytes.len() < expected_total {
        return Err(DnamError::BufferTooSmall {
            expected: expected_total,
            actual: sealed_bytes.len(),
        });
    }
    if sealed_bytes.len() != expected_total {
        return Err(DnamError::UnalignedOffset {
            field: "sealed_bytes.len",
            offset: sealed_bytes.len() as u32,
        });
    }

    // 1. Verify Castagnoli CRC32C over CL 0..=N+1
    let computed_crc = compute_seal_envelope_crc32c(&sealed_bytes[0..header_total_len]);
    if computed_crc != preamble.envelope_crc32c {
        return Err(DnamError::Crc32cMismatch {
            region: "dnam_seal_envelope",
            expected: preamble.envelope_crc32c,
            computed: computed_crc,
        });
    }

    // 2. Verify outer marketplace_sig_ed25519 if marketplace_pubkey is pinned
    let slots_bytes = &sealed_bytes[64..(1 + n) * 64];
    let sig_slice = &sealed_bytes[(1 + n) * 64..header_total_len];
    let encrypted_payload = &sealed_bytes[header_total_len..expected_total];

    if let Some(mp_pubkey) = crypto.marketplace_pubkey() {
        let mut preamble_zero_crc = [0u8; 64];
        preamble_zero_crc.copy_from_slice(&sealed_bytes[0..64]);
        preamble_zero_crc[60..64].fill(0);
        let mut sig_64 = [0u8; 64];
        sig_64.copy_from_slice(sig_slice);
        let sign_msg =
            build_seal_envelope_signing_message(&preamble_zero_crc, slots_bytes, encrypted_payload);
        crypto.ed25519_verify(mp_pubkey, &sign_msg, &sig_64)?;
    }

    // 3. Locate matching DnamSealRecipientSlot
    let target_slot = if let Some(local_id_hex) = crypto.local_device_id_hash() {
        let local_id_16 = decode_hex_array::<16>(local_id_hex)?;
        let mut found = None;
        for i in 0..n {
            let slot = DnamSealRecipientSlot::from_bytes(&slots_bytes[i * 64..(i + 1) * 64])?;
            if slot.device_id_hash == local_id_16 {
                found = Some(slot);
                break;
            }
        }
        found.ok_or(DnamError::CryptoVerificationFailed(
            "local hardware device_id_hash is not in .dnam-seal recipient slots",
        ))?
    } else {
        DnamSealRecipientSlot::from_bytes(&slots_bytes[0..64])?
    };

    // 4. Unwrap outer CEK and decrypt both outer full-container and nested inner arena layers
    let cek_outer = crypto.unwrap_seal_slot_cek(
        &preamble.ephemeral_pubkey_x25519,
        &target_slot.device_id_hash,
        &target_slot.wrapped_cek,
    )?;

    let mut plain_dnam = encrypted_payload.to_vec();
    chacha20_xor_arena_blocks(&mut plain_dnam, &cek_outer, &preamble.chacha20_nonce_96, 0)?;

    let inner_magic = [plain_dnam[0], plain_dnam[1], plain_dnam[2], plain_dnam[3]];
    if inner_magic != DNAM_MAGIC && inner_magic != DNCK_MAGIC {
        return Err(DnamError::CryptoVerificationFailed(
            "decrypted .dnam-seal payload does not begin with valid DNAM/DNCK magic",
        ));
    }

    if (preamble.flags & crate::container::FLAG_ENCRYPTED_ARENA) != 0 {
        let inner_preamble = DnamPreamble::from_bytes(&plain_dnam)?;
        let arena_off = inner_preamble.tensor_arena_offset as usize;
        let arena_len = inner_preamble.tensor_arena_byte_len as usize;
        let ultra_end = (inner_preamble.arena_offset_ultra_end as usize).min(arena_len);
        if plain_dnam.len() >= arena_off + ultra_end {
            let (cek_inner, nonce_inner) =
                derive_nested_inner_arena_key_and_nonce(&cek_outer, &preamble.chacha20_nonce_96);
            chacha20_xor_arena_blocks(
                &mut plain_dnam[arena_off..arena_off + ultra_end],
                &cek_inner,
                &nonce_inner,
                0,
            )?;
        }
    }

    Ok(plain_dnam)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::DefaultCryptoProvider;
    use crate::schema::PresetDef;

    #[test]
    fn test_immutable_header_jcs_excludes_mutable_presets_and_user_metadata() {
        let mut h1 = DnamJsonHeader::default_reference(false);
        let sha1 = compute_immutable_header_sha256(&h1).unwrap();

        h1.presets.push(PresetDef {
            id: String::from("user_custom"),
            name: String::from("User Custom Preset"),
            description: Some(String::from("Player")),
            controls: BTreeMap::from([(String::from("gain"), serde_json::json!(9.5))]),
            user_metadata: BTreeMap::new(),
        });
        h1.user_metadata
            .insert(String::from("favorite"), serde_json::json!(true));
        h1.user_metadata
            .insert(String::from("null_val"), serde_json::json!(null));
        let sha2 = compute_immutable_header_sha256(&h1).unwrap();
        assert_eq!(sha1, sha2);

        // Missing signatures when HAS_SIGNATURES is set must fail
        let crypto = DefaultCryptoProvider::default();
        assert!(verify_container_provenance(&h1, true, &[0u8; 64], &crypto).is_err());
        assert_eq!(
            verify_container_provenance(&h1, false, &[0u8; 64], &crypto).unwrap(),
            VerificationStatus::Unsigned
        );

        #[cfg(feature = "crypto")]
        {
            use crate::provenance::RustCryptoProvider;
            use crate::provenance::sha256::encode_hex;
            use crate::schema::{
                CreatorSignature, ModelSignatures, StemProvenance, StemRegistryEntry,
                TensorSliceEntry,
            };

            let dummy_slice = TensorSliceEntry {
                arena: 3,
                dtype: String::from("f64"),
                packing: None,
                feature_dim: None,
                packed_len: None,
                shape: None,
                byte_offset: 0,
                byte_length: 64,
            };
            h1.stems_registry = Some(alloc::vec![StemRegistryEntry {
                stem_id: String::from("stem0"),
                controls_meta: BTreeMap::new(),
                provenance: StemProvenance::default(),
                stem_arena3_crc32c: 0,
                extracted_physics: None,
                g_audio_uplo_u_f64: dummy_slice.clone(),
                c_audio_tier_f64: dummy_slice,
                user_metadata: BTreeMap::from([(String::from("tag"), serde_json::json!("val"))]),
            }]);
            let imm_sha = compute_immutable_header_sha256(&h1).unwrap();
            let arena = [0x42u8; 64];
            let arena_sha = sha256_digest(&arena);
            let seed = [11u8; 32];
            let pk = RustCryptoProvider::ed25519_pubkey_from_seed(&seed);
            let msg = build_creator_signing_message(&imm_sha, &arena_sha);
            let sig = RustCryptoProvider::ed25519_sign(&seed, &msg);

            h1.metadata.signatures = Some(ModelSignatures {
                immutable_header_sha256: encode_hex(&imm_sha),
                master_arena_sha256: encode_hex(&arena_sha),
                creator_signature: CreatorSignature {
                    signer_id: String::from("creator-1"),
                    pubkey_ed25519: encode_hex(&pk),
                    sig_ed25519: encode_hex(&sig),
                },
                marketplace_receipt: None,
            });
            let status = verify_container_provenance(&h1, true, &arena, &crypto).unwrap();
            assert_eq!(
                status,
                VerificationStatus::CreatorVerified {
                    signer_id: String::from("creator-1"),
                    pubkey_ed25519: pk,
                }
            );

            // Tampered arena must fail
            assert!(verify_container_provenance(&h1, true, &[0u8; 64], &crypto).is_err());

            // Tampered header must fail
            h1.metadata.name = String::from("Tampered");
            assert!(verify_container_provenance(&h1, true, &arena, &crypto).is_err());

            // Unsupported future schema version must fail validation and JCS projection
            let mut h_future = DnamJsonHeader::default_reference(false);
            h_future.version = String::from("1.99.0");
            assert!(h_future.validate().is_err());
            assert!(compute_immutable_header_sha256(&h_future).is_err());
            assert!(
                compute_immutable_header_sha256_for_version(&h1, SchemaVersion::new(2, 0, 0))
                    .is_err()
            );

            // SchemaVersion parsing and preamble matching
            let v1 = SchemaVersion::parse("1.0.0").unwrap();
            assert_eq!(v1, SchemaVersion::V1_0_0);
            assert!(v1.is_supported());
            assert!(v1.ensure_matches_preamble(1, 0).is_ok());
            assert!(v1.ensure_matches_preamble(1, 1).is_err());
            assert!(SchemaVersion::parse("invalid").is_err());
            assert!(SchemaVersion::parse("1.0.0.1").is_err());
        }
    }
}
