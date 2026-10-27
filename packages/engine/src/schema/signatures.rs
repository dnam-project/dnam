//! Split-payload `Ed25519` signature envelope (`ModelSignatures`), creator signature (`CreatorSignature`),
//! creator distribution policy (`DistributionPolicy`), container protection profile (`DnamProtectionProfile`),
//! OEM hardware factory attestation certificate (`OemDeviceCertificate`), marketplace download receipt
//! (`MarketplaceReceipt`), OEM hardware device allowlist (`AuthorizedDevice`), and multi-creator
//! royalty split records (`RoyaltySplit`).

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use super::error::DnamError;

/// Canonical container cryptographic protection profile (`"open"`, `"device_allowlisted"`, or `"enclave_sealed"`).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum DnamProtectionProfile {
    /// Open `.dnam` container (`ENCRYPTED_ARENA = 0`); runs on all DAWs, browsers, and hardware pedals.
    #[default]
    Open,
    /// Device-allowlisted `.dnam` container (`ENCRYPTED_ARENA = 1` or signed `authorized_devices`);
    /// keeps the 4 KB JSON header in plaintext while binding `Arena 0..2` to `authorized_devices`.
    DeviceAllowlisted,
    /// Enclave-sealed `.dnam-seal` (`b"DNSL"`) 64-byte aligned binary wrapper with nested `ENCRYPTED_ARENA = 1`.
    EnclaveSealed,
}

impl DnamProtectionProfile {
    /// Returns the canonical snake_case string identifier (`"open"`, `"device_allowlisted"`, `"enclave_sealed"`).
    #[inline]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::DeviceAllowlisted => "device_allowlisted",
            Self::EnclaveSealed => "enclave_sealed",
        }
    }

    /// Parses a snake_case protection profile identifier.
    pub fn from_str_name(s: &str) -> Result<Self, DnamError> {
        match s {
            "open" => Ok(Self::Open),
            "device_allowlisted" => Ok(Self::DeviceAllowlisted),
            "enclave_sealed" => Ok(Self::EnclaveSealed),
            _ => Err(DnamError::SchemaValidationError(alloc::format!(
                "invalid protection profile '{s}': expected 'open', 'device_allowlisted', or 'enclave_sealed'"
            ))),
        }
    }
}

/// Creator-signed distribution and hardware exclusivity policy (`metadata.distribution_policy`,
/// covered by RFC 8785 JCS `immutable_header_sha256` and signed by `creator_signature`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct DistributionPolicy {
    /// Permitted container protection profiles (`"open"`, `"device_allowlisted"`, `"enclave_sealed"`).
    /// When empty, all protection profiles are permitted.
    #[serde(default)]
    pub allowed_protection_profiles: Vec<DnamProtectionProfile>,
    /// Permitted OEM hardware vendor or `vendor:model_code` patterns (e.g., `"darkglass"`, `"darkglass:anagram_rk3582"`).
    /// When empty, all OEM vendors and hardware models are permitted.
    #[serde(default)]
    pub allowed_oem_vendors: Vec<String>,
}

impl DistributionPolicy {
    /// Returns `true` if `profile` is permitted by `allowed_protection_profiles` (empty means all profiles allowed).
    #[inline]
    pub fn is_profile_allowed(&self, profile: DnamProtectionProfile) -> bool {
        self.allowed_protection_profiles.is_empty()
            || self.allowed_protection_profiles.contains(&profile)
    }

    /// Returns `true` if `(oem_vendor, model_code)` is permitted by `allowed_oem_vendors`
    /// (empty means all OEM vendors and models are allowed).
    /// Supports both vendor-wide (`"darkglass"`) and model-specific (`"darkglass:anagram_rk3582"`) entries.
    pub fn is_oem_device_allowed(&self, oem_vendor: &str, model_code: &str) -> bool {
        if self.allowed_oem_vendors.is_empty() {
            return true;
        }
        for pattern in &self.allowed_oem_vendors {
            if let Some((pat_vendor, pat_model)) = pattern.split_once(':') {
                if pat_vendor.eq_ignore_ascii_case(oem_vendor)
                    && pat_model.eq_ignore_ascii_case(model_code)
                {
                    return true;
                }
            } else if pattern.eq_ignore_ascii_case(oem_vendor) {
                return true;
            }
        }
        false
    }
}

/// Factory-provisioned OEM hardware attestation certificate binding a device's non-exportable
/// TrustZone `X25519` public key (`pubkey_x25519`) to the manufacturer's root `Ed25519` key (`oem_sig_ed25519`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OemDeviceCertificate {
    /// OEM vendor identifier (e.g., `"darkglass"`).
    pub oem_vendor: String,
    /// Hardware model code (e.g., `"anagram_rk3582"`).
    pub model_code: String,
    /// 16-byte (`32` lowercase hex chars) hardware `device_id_hash`.
    pub device_id_hash: String,
    /// 32-byte (`64` lowercase hex chars) TrustZone `X25519` public key.
    pub pubkey_x25519: String,
    /// 64-byte (`128` lowercase hex chars) OEM factory `Ed25519` attestation signature.
    pub oem_sig_ed25519: String,
}

/// Content-addressable dataset and recipe provenance.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct DatasetProvenance {
    /// SHA-256 hex digest over sorted `(stem_id, input_pcm_sha256, target_pcm_sha256, controls_meta)`.
    #[serde(default)]
    pub dataset_manifest_sha256: String,
    /// Optional SHA-256 hex digest of the training recipe.
    #[serde(default)]
    pub config_recipe_sha256: Option<String>,
    /// Total training epochs completed.
    #[serde(default)]
    pub epochs_trained: u32,
}

/// Split-payload `Ed25519` signature envelope (`HAS_SIGNATURES` bit 9).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelSignatures {
    /// Hex SHA-256 digest of the RFC 8785 JCS canonical JSON header excluding mutable fields.
    pub immutable_header_sha256: String,
    /// Hex SHA-256 digest of the creator's un-watermarked master `Arena 0..2` (`36,224 B`).
    pub master_arena_sha256: String,
    /// Creator's offline `Ed25519` signature over `immutable_header_sha256 || master_arena_sha256`.
    pub creator_signature: CreatorSignature,
    /// Optional marketplace counter-signature binding `master_arena_sha256 -> buyer_arena_sha256`.
    #[serde(default)]
    pub marketplace_receipt: Option<MarketplaceReceipt>,
}

/// Creator's `Ed25519` signature block.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CreatorSignature {
    /// Creator identifier or `did:web` URI.
    pub signer_id: String,
    /// 32-byte `Ed25519` public key encoded as 64 lowercase hex characters.
    pub pubkey_ed25519: String,
    /// 64-byte `Ed25519` signature encoded as 128 lowercase hex characters.
    pub sig_ed25519: String,
}

/// Marketplace download receipt (`< 0.1 ms` edge personalization & OEM hardware authorization).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MarketplaceReceipt {
    /// Marketplace domain identifier (e.g., `"dnam.store"`).
    pub marketplace_id: String,
    /// Unique UUIDv7 license identifier.
    pub license_id: String,
    /// Hex SHA-256 digest of the buyer account identity.
    pub buyer_hash: String,
    /// License tier (e.g., `"personal"`, `"commercial"`, `"oem_bundle"`).
    pub license_tier: String,
    /// ISO-8601 UTC timestamp of receipt issuance.
    pub issued_at: String,
    /// Hex SHA-256 digest of the personalized buyer `Arena 0..2` (`36,224 B`, pre-encryption).
    pub buyer_arena_sha256: String,
    /// Arena encryption mode: `"none"` (`ENCRYPTED_ARENA = 0`) or `"x25519_chacha20_arena"` (`ENCRYPTED_ARENA = 1`).
    #[serde(default = "default_encryption_mode")]
    pub encryption_mode: String,
    /// 96-bit (`12 B`, 24 hex chars) ChaCha20 nonce when `encryption_mode == "x25519_chacha20_arena"`.
    #[serde(default)]
    pub arena_nonce_96: Option<String>,
    /// 32-byte (`64` hex chars) ephemeral `X25519` public key when `encryption_mode == "x25519_chacha20_arena"`.
    #[serde(default)]
    pub ephemeral_pubkey_x25519: Option<String>,
    /// Authorized OEM hardware devices and optional wrapped Content Encryption Keys (`wrapped_cek`).
    #[serde(default)]
    pub authorized_devices: Vec<AuthorizedDevice>,
    /// Optional multi-creator royalty split records (`sum(share_bps) == 10000`).
    #[serde(default)]
    pub royalty_splits: Vec<RoyaltySplit>,
    /// 64-byte `Ed25519` marketplace signature encoded as 128 lowercase hex characters.
    pub receipt_sig_ed25519: String,
}

fn default_encryption_mode() -> String {
    "none".to_string()
}

/// Authorized hardware device record inside `marketplace_receipt.authorized_devices`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuthorizedDevice {
    /// OEM vendor identifier (e.g., `"darkglass"`, `"apple"`, `"microsoft"`).
    pub oem_vendor: String,
    /// Hardware model code (e.g., `"anagram_rk3582"`, `"mac_secure_enclave"`, `"tpm2_pcr07"`).
    pub model_code: String,
    /// Hex digest of the hardware device unique identifier.
    pub device_id_hash: String,
    /// Optional 48-byte (`96` hex chars) `ChaCha20-Poly1305` wrapped CEK (`32 B` ciphertext + `16 B` tag).
    #[serde(default)]
    pub wrapped_cek: Option<String>,
}

/// Multi-creator royalty split entry (`10000 bps = 100.00%`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RoyaltySplit {
    /// Recipient identifier or `did:web` URI.
    pub recipient_id: String,
    /// Creator role (e.g., `"amp_capture"`, `"ir_cabinet"`, `"preset_designer"`).
    pub role: String,
    /// Share in basis points (`0..=10000`).
    pub share_bps: u16,
}
