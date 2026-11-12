//! `.dnam-seal` OEM hardware container envelope, recipient slot crypto, and distribution policy enforcement.
//!
//! Canonical Reference: `docs/security/provenance_and_enclave_drm.md`.

use alloc::string::ToString;
use alloc::vec::Vec;

use crate::container::{
    DNAM_MAGIC, crc32c_update, parse_and_validate_dmu1_frame, validate_mutable_json_schema,
    validate_v2_dnam_layout_with_max_profile, validate_v2_dnam_layout_with_max_profile_no_alloc,
};
use crate::provenance::chacha20::chacha20_xor_arena_blocks;
use crate::provenance::provider::CryptoProvider;
use crate::provenance::sha256::{decode_hex_array, encode_hex, sha256_digest};
use crate::schema::{DnamError, DnamJsonHeader, DnamProtectionProfile, OemDeviceCertificate};

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
    pub num_recipients: u16,
    /// Byte length of the inner `.dnam` container (`% 64 == 0`, e.g., `68,096 B`).
    pub inner_dnam_byte_len: u32,
    /// Raw 32-byte ephemeral `X25519` public key for `HPKE` key agreement.
    pub ephemeral_pubkey_x25519: [u8; 32],
    /// Raw 12-byte (`96-bit`) outer `ChaCha20` nonce.
    pub chacha20_nonce_96: [u8; 12],
    /// Castagnoli `CRC32C` over `CL 0..=N+1` (`[0..(2 + N) * 64]`) with `[0x3C..0x40]` zeroed.
    pub envelope_crc32c: u32,
}

const _: () = assert!(core::mem::size_of::<DnamSealEnvelopePreamble>() == 64);
const _: () = assert!(core::mem::align_of::<DnamSealEnvelopePreamble>() == 64);

impl DnamSealEnvelopePreamble {
    /// Serializes this 64-byte preamble into little-endian bytes.
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        out[0..4].copy_from_slice(&self.magic);
        out[4..6].copy_from_slice(&self.version_major.to_le_bytes());
        out[6..8].copy_from_slice(&self.version_minor.to_le_bytes());
        out[8..10].copy_from_slice(&self.flags.to_le_bytes());
        out[10..12].copy_from_slice(&self.num_recipients.to_le_bytes());
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
        let num_recipients = u16::from_le_bytes([bytes[10], bytes[11]]);
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

pub fn compute_seal_envelope_crc32c(header_region: &[u8]) -> u32 {
    let mut crc = crc32c_update(0, &header_region[0..60]);
    crc = crc32c_update(crc, &[0u8; 4]);
    crc32c_update(crc, &header_region[64..])
}

pub fn derive_nested_inner_arena_key_and_nonce(
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

pub fn build_seal_envelope_signing_message(
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
    use super::verify::verify_oem_device_certificate;

    if inner_dnam_bytes.len() < 256 || (inner_dnam_bytes.len() & 0x3F) != 0 {
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

    // 1. Parse and validate the inner canonical v2.0 .dnam layout, JCS header, and DMU1 trailer
    let layout = validate_v2_dnam_layout_with_max_profile(inner_dnam_bytes, 2)?;
    let json_slice =
        &inner_dnam_bytes[layout.header_extent.0..layout.header_extent.0 + layout.header_extent.1];
    let mut inner_header = DnamJsonHeader::from_json_slice(json_slice)
        .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?;
    let mutable_slice = &inner_dnam_bytes
        [layout.mutable_extent.0..layout.mutable_extent.0 + layout.mutable_extent.1];
    let dmu_val = parse_and_validate_dmu1_frame(mutable_slice)?;
    validate_mutable_json_schema(&dmu_val)?;
    if let Some(presets_val) = dmu_val.get("presets") {
        inner_header.presets = serde_json::from_value(presets_val.clone())
            .map_err(|e| DnamError::SchemaValidationError(e.to_string()))?;
    }
    if let Some(um_val) = dmu_val.get("user_metadata") {
        inner_header.user_metadata = serde_json::from_value(um_val.clone())
            .map_err(|e| DnamError::SchemaValidationError(e.to_string()))?;
    }
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
    //    Pass A: Encrypt retained tier arena slices with domain-separated inner arena key (cek_inner, nonce_inner)
    //    Pass B: Encrypt entire inner .dnam container with (cek_32, chacha20_nonce_96)
    let mut encrypted_payload = inner_dnam_bytes.to_vec();
    let arena_off = layout.tier_start;
    let retained_end = layout.tier_ends[layout.retained_tier as usize];
    let (cek_inner, nonce_inner) =
        derive_nested_inner_arena_key_and_nonce(cek_32, chacha20_nonce_96);
    chacha20_xor_arena_blocks(
        &mut encrypted_payload[arena_off..retained_end],
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
        num_recipients: recipient_certs.len() as u16,
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
    if payload_len < 256 || (payload_len & 0x3F) != 0 {
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
    if inner_magic != DNAM_MAGIC {
        return Err(DnamError::CryptoVerificationFailed(
            "decrypted .dnam-seal payload does not begin with valid DNAM magic",
        ));
    }

    if (preamble.flags & crate::container::FLAG_ENCRYPTED_ARENA) != 0 {
        let layout = validate_v2_dnam_layout_with_max_profile_no_alloc(&plain_dnam, 2)?;
        let arena_off = layout.tier_start;
        let retained_end = layout.tier_ends[layout.retained_tier as usize];
        let (cek_inner, nonce_inner) =
            derive_nested_inner_arena_key_and_nonce(&cek_outer, &preamble.chacha20_nonce_96);
        chacha20_xor_arena_blocks(
            &mut plain_dnam[arena_off..retained_end],
            &cek_inner,
            &nonce_inner,
            0,
        )?;
    }

    Ok(plain_dnam)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{
        AuthorizedDevice, DistributionPolicy, DnamJsonHeader, MarketplaceReceipt, ModelSignatures,
    };

    #[test]
    fn test_distribution_policy_enforcement_branches() {
        let mut header = DnamJsonHeader::default_reference();

        // 1. No distribution policy: always Ok
        header.metadata.distribution_policy = None;
        assert!(enforce_distribution_policy(&header, DnamProtectionProfile::Open).is_ok());

        // 2. Profile not allowed
        header.metadata.distribution_policy = Some(DistributionPolicy {
            allowed_protection_profiles: alloc::vec![DnamProtectionProfile::EnclaveSealed],
            allowed_oem_vendors: alloc::vec![],
        });
        assert!(enforce_distribution_policy(&header, DnamProtectionProfile::Open).is_err());

        // 3. Vendor restricted + Open profile -> Err
        header.metadata.distribution_policy = Some(DistributionPolicy {
            allowed_protection_profiles: alloc::vec![
                DnamProtectionProfile::Open,
                DnamProtectionProfile::DeviceAllowlisted
            ],
            allowed_oem_vendors: alloc::vec!["darkglass".to_string()],
        });
        assert!(enforce_distribution_policy(&header, DnamProtectionProfile::Open).is_err());

        // 4. DeviceAllowlisted without receipt -> Err
        header.metadata.signatures = None;
        assert!(
            enforce_distribution_policy(&header, DnamProtectionProfile::DeviceAllowlisted).is_err()
        );

        // 5. DeviceAllowlisted with empty authorized_devices -> Err
        header.metadata.signatures = Some(ModelSignatures {
            immutable_header_sha256: "0".repeat(64),
            master_arena_sha256: "0".repeat(64),
            tier_arena_sha256: None,
            creator_signature: crate::schema::CreatorSignature {
                signer_id: "creator".to_string(),
                pubkey_ed25519: "0".repeat(64),
                sig_ed25519: "0".repeat(128),
            },
            marketplace_receipt: Some(MarketplaceReceipt {
                marketplace_id: "mkt".to_string(),
                license_id: "lic".to_string(),
                buyer_hash: "0".repeat(64),
                license_tier: "oem".to_string(),
                issued_at: "2026-01-01T00:00:00Z".to_string(),
                buyer_arena_sha256: "0".repeat(64),
                buyer_tier_arena_sha256: None,
                encryption_mode: "none".to_string(),
                arena_nonce_96: None,
                ephemeral_pubkey_x25519: None,
                authorized_devices: alloc::vec![],
                royalty_splits: alloc::vec![],
                receipt_sig_ed25519: "0".repeat(128),
            }),
        });
        assert!(
            enforce_distribution_policy(&header, DnamProtectionProfile::DeviceAllowlisted).is_err()
        );

        // 6. DeviceAllowlisted with matching authorized device -> Ok
        if let Some(ref mut sigs) = header.metadata.signatures
            && let Some(ref mut rcpt) = sigs.marketplace_receipt
        {
            rcpt.authorized_devices.push(AuthorizedDevice {
                oem_vendor: "darkglass".to_string(),
                model_code: "anagram".to_string(),
                device_id_hash: "0".repeat(32),
                wrapped_cek: None,
            });
        }
        assert!(
            enforce_distribution_policy(&header, DnamProtectionProfile::DeviceAllowlisted).is_ok()
        );

        // 7. Disallowed device vendor -> Err
        if let Some(ref mut sigs) = header.metadata.signatures
            && let Some(ref mut rcpt) = sigs.marketplace_receipt
        {
            rcpt.authorized_devices.push(AuthorizedDevice {
                oem_vendor: "disallowed".to_string(),
                model_code: "other".to_string(),
                device_id_hash: "0".repeat(32),
                wrapped_cek: None,
            });
        }
        assert!(
            enforce_distribution_policy(&header, DnamProtectionProfile::DeviceAllowlisted).is_err()
        );
    }

    #[cfg(feature = "crypto")]
    #[test]
    fn test_seal_and_unseal_error_branches() {
        use crate::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
        use crate::provenance::{RustCryptoProvider, encode_hex};
        use crate::schema::OemDeviceCertificate;
        use alloc::string::String;

        assert!(DnamSealEnvelopePreamble::from_bytes(&[0u8; 16]).is_err());
        let mut bad_magic = [0u8; 64];
        bad_magic[0..4].copy_from_slice(b"NOPE");
        assert!(DnamSealEnvelopePreamble::from_bytes(&bad_magic).is_err());
        bad_magic[0..4].copy_from_slice(b"DNSL");
        bad_magic[4..6].copy_from_slice(&99u16.to_le_bytes());
        assert!(DnamSealEnvelopePreamble::from_bytes(&bad_magic).is_err());
        assert!(DnamSealRecipientSlot::from_bytes(&[0u8; 16]).is_err());

        let mut header = DnamJsonHeader::default_reference();
        let arena = alloc::vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let dnam = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

        let oem_seed = [21u8; 32];
        let oem_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&oem_seed);
        let dev_sk = [22u8; 32];
        let dev_pk = RustCryptoProvider::x25519_pubkey_from_secret(&dev_sk);
        let dev_id = "0102030405060708090a0b0c0d0e0f10";
        let cert_msg = crate::provenance::build_oem_device_cert_signing_message(
            "darkglass",
            "anagram",
            dev_id,
            &dev_pk,
        );
        let cert_sig = RustCryptoProvider::ed25519_sign(&oem_seed, &cert_msg);
        let cert = OemDeviceCertificate {
            oem_vendor: String::from("darkglass"),
            model_code: String::from("anagram"),
            device_id_hash: String::from(dev_id),
            pubkey_x25519: encode_hex(&dev_pk),
            oem_sig_ed25519: encode_hex(&cert_sig),
        };
        let trusted = [("darkglass", oem_pk)];
        let crypto = RustCryptoProvider::new().with_device_key(dev_id, dev_sk);

        // Invalid seal inputs
        assert!(
            seal_dnam_container(
                &[0u8; 4095],
                core::slice::from_ref(&cert),
                &trusted,
                &[1u8; 32],
                &[2u8; 32],
                &[4u8; 32],
                &[3u8; 12],
            )
            .is_err()
        );

        let sealed = seal_dnam_container(
            &dnam,
            &[cert],
            &trusted,
            &[1u8; 32],
            &[2u8; 32],
            &[4u8; 32],
            &[3u8; 12],
        )
        .unwrap();

        // Corrupt num_recipients = 0
        let mut bad_n = sealed.clone();
        bad_n[10..12].copy_from_slice(&0u16.to_le_bytes());
        assert!(unseal_dnam_container(&bad_n, &crypto).is_err());

        // Corrupt inner_dnam_byte_len = 100
        let mut bad_len = sealed.clone();
        bad_len[12..16].copy_from_slice(&100u32.to_le_bytes());
        assert!(unseal_dnam_container(&bad_len, &crypto).is_err());

        // Truncated and extra trailing bytes
        assert!(unseal_dnam_container(&sealed[..256], &crypto).is_err());
        let mut extra = sealed.clone();
        extra.extend_from_slice(&[0u8; 64]);
        assert!(unseal_dnam_container(&extra, &crypto).is_err());

        // Corrupt envelope_crc32c
        let mut bad_crc = sealed.clone();
        bad_crc[60] ^= 0xFF;
        assert!(unseal_dnam_container(&bad_crc, &crypto).is_err());

        // Corrupt first ciphertext block so decrypted inner magic != DNAM
        let mut bad_payload = sealed;
        bad_payload[192] ^= 0xFF;
        assert!(unseal_dnam_container(&bad_payload, &crypto).is_err());
    }
}
