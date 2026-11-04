//! Canonical v2.0 Cryptographic Provenance (`DAU1` / `DAI1` / `DSA1`), `OemDeviceCertificate` V2,
//! RFC 9180 HPKE + `ChaCha20-Poly1305` Record Framing, `DNSL` / `DNPL` / `DME1` Envelopes,
//! License-Scoped Watermark Seed Derivation (`HMAC-SHA256`), Stage III Tier-Local Structural
//! Fingerprinting, and `.dnam-prev` Preview Containment.
//!
//! Implements:
//! - `ref[sec.taxonomy.overview]`, `ref[sec.jcs.immutable-header-digest]`,
//!   `ref[sec.policy.distribution-policy]`
//! - `ref[sec.sig.prefix-delegation]`, `ref[sec.sig.dau1-envelope]`, `ref[sec.sig.offline-auth]`
//! - `ref[sec.drm.oem-certificate]`, `ref[sec.drm.trustzone-flow]`,
//!   `ref[sec.drm.hpke-record-encryption]`, `ref[sec.drm.dnsl-public-framing]`,
//!   `ref[sec.drm.protected-inner-framing]`, `ref[sec.drm.dme1-mutable-suffix]`,
//!   `ref[sec.drm.protection-profiles]`
//! - `ref[sec.watermark.reciprocal-gauge]`, `ref[sec.watermark.orthogonal-gauge]`,
//!   `ref[sec.watermark.tardos-code]`, `ref[sec.watermark.seed-derivation]`
//! - `ref[sec.anticlone.three-stage-gate]`, `ref[sec.anticlone.stage3-fingerprint]`
//! - `ref[sec.preview.containment]`, `ref[sec.preview.royalty-splits]`,
//!   `ref[sec.preview.storefront-metadata]`, `ref[sec.edge.data-model]`

use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::vec::Vec;

use crate::container::{align64_u64, encode_domain_field, parse_and_validate_dmu1_frame};
use crate::schema::{DnamError, RoyaltySplit, SchemaOptionExt, reject_if};

use super::sha256::sha256_digest;

/// Maximum plaintext byte length for a single encrypted container record (`(2^32 - 1) * 64`).
pub const MAX_RECORD_PLAINTEXT_BYTES: u64 = ((1u64 << 32) - 1) * 64;
/// Maximum ciphertext byte length (`MAX_RECORD_PLAINTEXT_BYTES + 16`).
pub const MAX_RECORD_CIPHERTEXT_BYTES: u64 = MAX_RECORD_PLAINTEXT_BYTES + 16;
/// RFC 9180 HPKE cipher suite identifier (`DHKEM(X25519, HKDF-SHA256) = 0x0020`,
/// `HKDF-SHA256 = 0x0001`, `ChaCha20Poly1305 = 0x0003`, `ref[sec.drm.hpke-record-encryption]`).
pub const HPKE_SUITE_X25519_SHA256_CHACHA20POLY1305: [u8; 6] = [0x00, 0x20, 0x00, 0x01, 0x00, 0x03];

// ============================================================================
// Pure-Rust HMAC-SHA256 (works with or without `crypto` feature)
// ============================================================================

/// Computes `HMAC-SHA256(key, message)` (`ref[sec.watermark.seed-derivation]`).
pub fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    let mut k_block = [0u8; 64];
    if key.len() > 64 {
        let hashed = sha256_digest(key);
        k_block[..32].copy_from_slice(&hashed);
    } else {
        k_block[..key.len()].copy_from_slice(key);
    }
    let mut ipad = [0x36u8; 64];
    let mut opad = [0x5cu8; 64];
    for i in 0..64 {
        ipad[i] ^= k_block[i];
        opad[i] ^= k_block[i];
    }
    let mut inner_in = Vec::with_capacity(64 + message.len());
    inner_in.extend_from_slice(&ipad);
    inner_in.extend_from_slice(message);
    let inner_hash = sha256_digest(&inner_in);

    let mut outer_in = [0u8; 96];
    outer_in[..64].copy_from_slice(&opad);
    outer_in[64..96].copy_from_slice(&inner_hash);
    sha256_digest(&outer_in)
}

// ============================================================================
// Prefix Lists & Two-Certificate Ed25519 Delegation (`ref[sec.sig.prefix-delegation]`,
// `ref[sec.sig.dau1-envelope]`)
// ============================================================================

/// Encodes an ordered prefix digest list `L = u8(count) || (u8(tier) || digest[32])*`
/// (`1 <= count <= 4`, `ref[sec.sig.prefix-delegation]`).
pub fn encode_prefix_list(prefixes: &[[u8; 32]]) -> Result<Vec<u8>, DnamError> {
    reject_if(
        prefixes.is_empty() || prefixes.len() > 4,
        "invalid prefix list count",
    )?;
    let mut out = Vec::with_capacity(1 + 33 * prefixes.len());
    out.push(prefixes.len() as u8);
    for (tier, digest) in prefixes.iter().enumerate() {
        out.push(tier as u8);
        out.extend_from_slice(digest);
    }
    Ok(out)
}

/// Parses and validates an ordered prefix digest list `L` for `expected_max_tier` (`0..=3`,
/// `ref[sec.sig.prefix-delegation]`).
pub fn parse_prefix_list(data: &[u8], expected_max_tier: u8) -> Result<Vec<[u8; 32]>, DnamError> {
    reject_if(expected_max_tier > 3, "invalid expected_max_tier")?;
    let count = (expected_max_tier as usize) + 1;
    reject_if(
        data.len() != 1 + 33 * count || data[0] != (count as u8),
        "invalid prefix list length or count",
    )?;
    let mut out = Vec::with_capacity(count);
    for tier in 0..count {
        let off = 1 + 33 * tier;
        reject_if(data[off] != (tier as u8), "unordered prefix tier index")?;
        let mut d = [0u8; 32];
        d.copy_from_slice(&data[off + 1..off + 33]);
        out.push(d);
    }
    Ok(out)
}

/// Builds the `"DNAM-PLAINTEXT-SHARED-V1"` shared plaintext framing (`ref[sec.sig.prefix-delegation]`).
pub fn build_plaintext_shared_v1(
    preamble: &[u8],
    directory: &[u8],
    shared_binary: &[u8],
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&encode_domain_field("domain", b"DNAM-PLAINTEXT-SHARED-V1"));
    out.extend_from_slice(&encode_domain_field("preamble", preamble));
    out.extend_from_slice(&encode_domain_field("directory", directory));
    out.extend_from_slice(&encode_domain_field("shared_model_binary", shared_binary));
    out
}

/// Computes the cumulative per-tier plaintext prefix digests under `domain`
/// (`b"DNAM-MASTER-PREFIX-V1"` or `b"DNAM-BUYER-PREFIX-V1"`, `ref[sec.sig.prefix-delegation]`).
pub fn compute_plaintext_prefixes_v1(
    domain: &[u8],
    header_sha256: &[u8; 32],
    shared_model_data: &[u8],
    tiers: &[&[u8]],
) -> Result<Vec<[u8; 32]>, DnamError> {
    reject_if(tiers.is_empty() || tiers.len() > 4, "invalid tier count")?;
    let labels = ["tier_lite", "tier_full", "tier_pro", "tier_ultra"];
    let mut base = Vec::new();
    base.extend_from_slice(&encode_domain_field("domain", domain));
    base.extend_from_slice(&encode_domain_field(
        "immutable_header_sha256",
        header_sha256,
    ));
    base.extend_from_slice(&encode_domain_field("shared_model_data", shared_model_data));

    let mut out = Vec::with_capacity(tiers.len());
    for idx in 0..tiers.len() {
        base.extend_from_slice(&encode_domain_field(labels[idx], tiers[idx]));
        out.push(sha256_digest(&base));
    }
    Ok(out)
}

/// Computes the cumulative per-tier ciphertext prefix digests under `"DNAM-CIPHERTEXT-PREFIX-V1"`
/// (`ref[sec.sig.prefix-delegation]`).
pub fn compute_ciphertext_prefixes_v1(
    record_context_sha256: &[u8; 32],
    shared_ct_and_pad: &[u8],
    tier_cts_and_pads: &[&[u8]],
) -> Result<Vec<[u8; 32]>, DnamError> {
    reject_if(
        tier_cts_and_pads.is_empty() || tier_cts_and_pads.len() > 4,
        "invalid tier count",
    )?;
    let labels = ["tier_lite", "tier_full", "tier_pro", "tier_ultra"];
    let mut base = Vec::new();
    base.extend_from_slice(&encode_domain_field("domain", b"DNAM-CIPHERTEXT-PREFIX-V1"));
    base.extend_from_slice(&encode_domain_field(
        "record_context_sha256",
        record_context_sha256,
    ));
    base.extend_from_slice(&encode_domain_field("shared", shared_ct_and_pad));

    let mut out = Vec::with_capacity(tier_cts_and_pads.len());
    for idx in 0..tier_cts_and_pads.len() {
        base.extend_from_slice(&encode_domain_field(labels[idx], tier_cts_and_pads[idx]));
        out.push(sha256_digest(&base));
    }
    Ok(out)
}

/// Builds the `"DNAM-CREATOR-MANIFEST-V1"` signing message (`ref[sec.sig.prefix-delegation]`).
pub fn build_creator_manifest_v1(
    creator_pubkey: &[u8; 32],
    immutable_header_sha256: &[u8; 32],
    master_prefixes: &[[u8; 32]],
) -> Result<Vec<u8>, DnamError> {
    let prefix_list = encode_prefix_list(master_prefixes)?;
    let max_tier = (master_prefixes.len() - 1) as u8;
    let mut out = Vec::new();
    out.extend_from_slice(&encode_domain_field("domain", b"DNAM-CREATOR-MANIFEST-V1"));
    out.extend_from_slice(&encode_domain_field(
        "creator_pubkey_ed25519",
        creator_pubkey,
    ));
    out.extend_from_slice(&encode_domain_field(
        "immutable_header_sha256",
        immutable_header_sha256,
    ));
    out.extend_from_slice(&encode_domain_field("issued_max_tier", &[max_tier]));
    out.extend_from_slice(&encode_domain_field("master_prefix_sha256", &prefix_list));
    Ok(out)
}

/// Computes `creator_signed_manifest_sha256` under `"DNAM-CREATOR-SIGNED-V1"` (`ref[sec.sig.prefix-delegation]`).
pub fn compute_creator_signed_manifest_sha256(
    creator_manifest: &[u8],
    creator_signature: &[u8; 64],
) -> [u8; 32] {
    let mut msg = Vec::new();
    msg.extend_from_slice(&encode_domain_field("domain", b"DNAM-CREATOR-SIGNED-V1"));
    msg.extend_from_slice(&encode_domain_field("creator_manifest", creator_manifest));
    msg.extend_from_slice(&encode_domain_field("creator_signature", creator_signature));
    sha256_digest(&msg)
}

/// Computes `license_policy_sha256` under `"DNAM-LICENSE-POLICY-V1"` (`ref[sec.sig.prefix-delegation]`).
pub fn compute_license_policy_sha256(policy_bytes: &[u8]) -> [u8; 32] {
    let mut msg = Vec::new();
    msg.extend_from_slice(&encode_domain_field("domain", b"DNAM-LICENSE-POLICY-V1"));
    msg.extend_from_slice(&encode_domain_field("policy", policy_bytes));
    sha256_digest(&msg)
}

/// Builds the `"DNAM-BUYER-RECEIPT-V1"` signing message (`ref[sec.sig.prefix-delegation]`).
pub fn build_buyer_receipt_v1(
    marketplace_pubkey: &[u8; 32],
    creator_signed_manifest_sha256: &[u8; 32],
    license_policy_sha256: &[u8; 32],
    protection_profile: u8,
    buyer_prefixes: &[[u8; 32]],
    inner_record_context: Option<&[u8; 32]>,
    inner_ciphertext_prefixes: &[[u8; 32]],
) -> Result<Vec<u8>, DnamError> {
    reject_if(protection_profile > 2, "invalid protection_profile")?;
    let buyer_list = encode_prefix_list(buyer_prefixes)?;
    let (ctx_slice, ct_list) = if protection_profile == 0 {
        reject_if(
            inner_record_context.is_some() || !inner_ciphertext_prefixes.is_empty(),
            "open profile must omit inner ciphertext context",
        )?;
        (&[][..], Vec::new())
    } else {
        let ctx = inner_record_context.or_schema_err("protected profile requires inner context")?;
        reject_if(
            inner_ciphertext_prefixes.len() != buyer_prefixes.len(),
            "mismatched inner ciphertext prefix count",
        )?;
        (&ctx[..], encode_prefix_list(inner_ciphertext_prefixes)?)
    };
    let max_tier = (buyer_prefixes.len() - 1) as u8;
    let mut out = Vec::new();
    out.extend_from_slice(&encode_domain_field("domain", b"DNAM-BUYER-RECEIPT-V1"));
    out.extend_from_slice(&encode_domain_field(
        "marketplace_pubkey_ed25519",
        marketplace_pubkey,
    ));
    out.extend_from_slice(&encode_domain_field(
        "creator_signed_manifest_sha256",
        creator_signed_manifest_sha256,
    ));
    out.extend_from_slice(&encode_domain_field(
        "license_policy_sha256",
        license_policy_sha256,
    ));
    out.extend_from_slice(&encode_domain_field(
        "protection_profile",
        &[protection_profile],
    ));
    out.extend_from_slice(&encode_domain_field("issued_max_tier", &[max_tier]));
    out.extend_from_slice(&encode_domain_field("buyer_prefix_sha256", &buyer_list));
    out.extend_from_slice(&encode_domain_field(
        "inner_record_context_sha256",
        ctx_slice,
    ));
    out.extend_from_slice(&encode_domain_field(
        "inner_ciphertext_prefix_sha256",
        &ct_list,
    ));
    Ok(out)
}

/// Computes `outer_public_metadata_sha256` under `"DNAM-OUTER-PUBLIC-METADATA-V1"` (`ref[sec.sig.prefix-delegation]`).
pub fn compute_outer_public_metadata_sha256(
    preamble: &[u8],
    directory: &[u8],
    public_auxiliary: &[u8],
) -> [u8; 32] {
    let mut msg = Vec::new();
    msg.extend_from_slice(&encode_domain_field(
        "domain",
        b"DNAM-OUTER-PUBLIC-METADATA-V1",
    ));
    msg.extend_from_slice(&encode_domain_field("preamble", preamble));
    msg.extend_from_slice(&encode_domain_field("directory", directory));
    msg.extend_from_slice(&encode_domain_field("public_auxiliary", public_auxiliary));
    sha256_digest(&msg)
}

/// Builds the `"DNAM-OUTER-MANIFEST-V1"` signing message (`ref[sec.sig.prefix-delegation]`).
pub fn build_outer_manifest_v1(
    outer_signer_pubkey: &[u8; 32],
    outer_public_metadata_sha256: &[u8; 32],
    outer_record_context_sha256: &[u8; 32],
    outer_ciphertext_prefixes: &[[u8; 32]],
) -> Result<Vec<u8>, DnamError> {
    let ct_list = encode_prefix_list(outer_ciphertext_prefixes)?;
    let max_tier = (outer_ciphertext_prefixes.len() - 1) as u8;
    let mut out = Vec::new();
    out.extend_from_slice(&encode_domain_field("domain", b"DNAM-OUTER-MANIFEST-V1"));
    out.extend_from_slice(&encode_domain_field(
        "outer_signer_pubkey_ed25519",
        outer_signer_pubkey,
    ));
    out.extend_from_slice(&encode_domain_field(
        "outer_public_metadata_sha256",
        outer_public_metadata_sha256,
    ));
    out.extend_from_slice(&encode_domain_field(
        "outer_record_context_sha256",
        outer_record_context_sha256,
    ));
    out.extend_from_slice(&encode_domain_field("issued_max_tier", &[max_tier]));
    out.extend_from_slice(&encode_domain_field(
        "outer_ciphertext_prefix_sha256",
        &ct_list,
    ));
    Ok(out)
}

/// Parsed Open `DAU1` Authentication Envelope (`ref[sec.sig.dau1-envelope]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dau1Envelope {
    pub auth_kind: u8,
    pub creator_pubkey: [u8; 32],
    pub creator_max_tier: u8,
    pub master_prefixes: Vec<[u8; 32]>,
    pub creator_signature: [u8; 64],
    pub marketplace_pubkey: Option<[u8; 32]>,
    pub policy_bytes: Vec<u8>,
    pub buyer_max_tier: Option<u8>,
    pub buyer_prefixes: Vec<[u8; 32]>,
    pub marketplace_signature: Option<[u8; 64]>,
}

impl Dau1Envelope {
    /// Serializes `self` into the canonical `DAU1` binary wire envelope (`106 + 33*N_c` bytes for
    /// `auth_kind == 0`, `208 + 33*N_c + 33*N_b + P` bytes for `auth_kind == 1`, `ref[sec.sig.dau1-envelope]`).
    pub fn to_bytes(&self) -> Result<Vec<u8>, DnamError> {
        reject_if(
            self.auth_kind > 1
                || self.creator_max_tier > 3
                || self.master_prefixes.len() != (self.creator_max_tier as usize + 1),
            "invalid DAU1 creator fields",
        )?;
        let l_master = encode_prefix_list(&self.master_prefixes)?;
        let mut out = Vec::new();
        out.extend_from_slice(b"DAU1");
        out.push(self.auth_kind);
        out.extend_from_slice(&[0u8; 3]);
        out.extend_from_slice(&self.creator_pubkey);
        out.push(self.creator_max_tier);
        out.extend_from_slice(&l_master);
        out.extend_from_slice(&self.creator_signature);

        if self.auth_kind == 1 {
            let m_pub = self
                .marketplace_pubkey
                .or_schema_err("missing DAU1 marketplace_pubkey")?;
            let b_max = self
                .buyer_max_tier
                .or_schema_err("missing DAU1 buyer_max_tier")?;
            let m_sig = self
                .marketplace_signature
                .or_schema_err("missing DAU1 marketplace_signature")?;
            reject_if(
                b_max > self.creator_max_tier || self.buyer_prefixes.len() != (b_max as usize + 1),
                "invalid DAU1 buyer tier fields",
            )?;
            let l_buyer = encode_prefix_list(&self.buyer_prefixes)?;
            out.extend_from_slice(&m_pub);
            out.extend_from_slice(&(self.policy_bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(&self.policy_bytes);
            out.push(b_max);
            out.extend_from_slice(&l_buyer);
            out.extend_from_slice(&m_sig);
        }
        Ok(out)
    }

    /// Parses and validates a `DAU1` binary wire envelope (`ref[sec.sig.dau1-envelope]`).
    pub fn from_bytes(data: &[u8]) -> Result<Self, DnamError> {
        reject_if(
            data.len() < 106 + 33
                || &data[0..4] != b"DAU1"
                || data[4] > 1
                || data[5..8] != [0, 0, 0],
            "invalid DAU1 header",
        )?;
        let auth_kind = data[4];
        let mut creator_pubkey = [0u8; 32];
        creator_pubkey.copy_from_slice(&data[8..40]);
        let creator_max_tier = data[40];
        reject_if(creator_max_tier > 3, "invalid DAU1 creator_max_tier")?;
        let nc = (creator_max_tier as usize) + 1;
        let l_master_end = 41 + 1 + 33 * nc;
        reject_if(
            data.len() < l_master_end + 64,
            "truncated DAU1 creator section",
        )?;
        let master_prefixes = parse_prefix_list(&data[41..l_master_end], creator_max_tier)?;
        let mut creator_signature = [0u8; 64];
        creator_signature.copy_from_slice(&data[l_master_end..l_master_end + 64]);
        let pos = l_master_end + 64;

        if auth_kind == 0 {
            reject_if(
                pos != data.len(),
                "trailing bytes in auth_kind=0 DAU1 envelope",
            )?;
            return Ok(Self {
                auth_kind: 0,
                creator_pubkey,
                creator_max_tier,
                master_prefixes,
                creator_signature,
                marketplace_pubkey: None,
                policy_bytes: Vec::new(),
                buyer_max_tier: None,
                buyer_prefixes: Vec::new(),
                marketplace_signature: None,
            });
        }

        reject_if(data.len() < pos + 36, "truncated DAU1 marketplace header")?;
        let mut marketplace_pubkey = [0u8; 32];
        marketplace_pubkey.copy_from_slice(&data[pos..pos + 32]);
        let p_len = u32::from_le_bytes(data[pos + 32..pos + 36].try_into().unwrap()) as usize;
        let p_end = pos + 36 + p_len;
        reject_if(data.len() < p_end + 1, "truncated DAU1 policy")?;
        let policy_bytes = data[pos + 36..p_end].to_vec();
        let buyer_max_tier = data[p_end];
        reject_if(
            buyer_max_tier > creator_max_tier,
            "buyer_max_tier exceeds creator_max_tier",
        )?;
        let nb = (buyer_max_tier as usize) + 1;
        let l_buyer_end = p_end + 1 + 1 + 33 * nb;
        reject_if(
            data.len() != l_buyer_end + 64,
            "invalid DAU1 buyer section length",
        )?;
        let buyer_prefixes = parse_prefix_list(&data[p_end + 1..l_buyer_end], buyer_max_tier)?;
        let mut marketplace_signature = [0u8; 64];
        marketplace_signature.copy_from_slice(&data[l_buyer_end..l_buyer_end + 64]);

        Ok(Self {
            auth_kind: 1,
            creator_pubkey,
            creator_max_tier,
            master_prefixes,
            creator_signature,
            marketplace_pubkey: Some(marketplace_pubkey),
            policy_bytes,
            buyer_max_tier: Some(buyer_max_tier),
            buyer_prefixes,
            marketplace_signature: Some(marketplace_signature),
        })
    }
}

// ============================================================================
// OEM Device Certificate V2 (`ref[sec.drm.oem-certificate]`)
// ============================================================================

fn is_canonical_oem_id(s: &str, max_len: usize) -> bool {
    let b = s.as_bytes();
    if b.is_empty() || b.len() > max_len {
        return false;
    }
    if !(b[0].is_ascii_lowercase() || b[0].is_ascii_digit()) {
        return false;
    }
    b[1..]
        .iter()
        .all(|&c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, b'.' | b'_' | b'-'))
}

/// Builds the domain-separated V2 `OemDeviceCertificate` signing message (`"DNAM-OEM-CERT-V2"`,
/// `ref[sec.drm.oem-certificate]`).
#[allow(clippy::too_many_arguments)]
pub fn build_oem_cert_v2_signing_message(
    vendor: &str,
    model: &str,
    device_id_hash: &[u8],
    public_key_x25519: &[u8],
    issuer_public_key_ed25519: &[u8],
    assurance: u8,
    attestation_policy_sha256: &[u8],
    attestation_evidence_sha256: &[u8],
) -> Result<Vec<u8>, DnamError> {
    reject_if(!is_canonical_oem_id(vendor, 32), "invalid vendor")?;
    reject_if(!is_canonical_oem_id(model, 64), "invalid model")?;
    reject_if(
        device_id_hash.len() != 16
            || public_key_x25519.len() != 32
            || issuer_public_key_ed25519.len() != 32
            || attestation_policy_sha256.len() != 32
            || attestation_evidence_sha256.len() != 32
            || !matches!(assurance, 1 | 2),
        "invalid certificate field",
    )?;
    let issuer_sha = sha256_digest(issuer_public_key_ed25519);
    let mut out = Vec::new();
    out.extend_from_slice(&encode_domain_field("domain", b"DNAM-OEM-CERT-V2"));
    out.extend_from_slice(&encode_domain_field("oem_vendor", vendor.as_bytes()));
    out.extend_from_slice(&encode_domain_field("model_code", model.as_bytes()));
    out.extend_from_slice(&encode_domain_field("device_id_hash", device_id_hash));
    out.extend_from_slice(&encode_domain_field("public_key_x25519", public_key_x25519));
    out.extend_from_slice(&encode_domain_field("issuer_key_sha256", &issuer_sha));
    out.extend_from_slice(&encode_domain_field("assurance", &[assurance]));
    out.extend_from_slice(&encode_domain_field(
        "attestation_policy_sha256",
        attestation_policy_sha256,
    ));
    out.extend_from_slice(&encode_domain_field(
        "attestation_evidence_sha256",
        attestation_evidence_sha256,
    ));
    Ok(out)
}

// ============================================================================
// Encrypted Structure, Pre-Wrap Plan, Record Context & DME1 Mutable Framing
// (`ref[sec.drm.hpke-record-encryption]`, `ref[sec.drm.dme1-mutable-suffix]`,
//  `ref[sec.drm.dnsl-public-framing]`, `ref[sec.drm.protected-inner-framing]`)
// ============================================================================

/// Computes `structural_metadata_sha256` under `"DNAM-ENCRYPTED-STRUCTURE-V1"` (`layer` in `0..=1`,
/// `ref[sec.drm.hpke-record-encryption]`).
pub fn compute_encrypted_structural_metadata_sha256(
    layer: u8,
    preamble: &[u8],
    directory: &[u8],
) -> Result<[u8; 32], DnamError> {
    reject_if(
        layer > 1 || preamble.is_empty() || directory.is_empty(),
        "invalid encrypted structure",
    )?;
    let mut msg = Vec::new();
    msg.extend_from_slice(&encode_domain_field(
        "domain",
        b"DNAM-ENCRYPTED-STRUCTURE-V1",
    ));
    msg.extend_from_slice(&encode_domain_field("layer", &[layer]));
    msg.extend_from_slice(&encode_domain_field("preamble", preamble));
    msg.extend_from_slice(&encode_domain_field("directory", directory));
    Ok(sha256_digest(&msg))
}

/// Encodes `R = u32le(N) || recipient_key_sha256[0..N]` enforcing strictly increasing lexicographical
/// order (`ref[sec.drm.hpke-record-encryption]`).
pub fn encode_recipient_key_ids(key_digests: &[[u8; 32]]) -> Result<Vec<u8>, DnamError> {
    reject_if(key_digests.is_empty(), "invalid recipient vector")?;
    for pair in key_digests.windows(2) {
        reject_if(pair[0] >= pair[1], "invalid recipient vector")?;
    }
    let mut out = Vec::with_capacity(4 + 32 * key_digests.len());
    out.extend_from_slice(&(key_digests.len() as u32).to_le_bytes());
    for d in key_digests {
        out.extend_from_slice(d);
    }
    Ok(out)
}

/// Parses and validates `R = u32le(N) || recipient_key_sha256[0..N]` (`ref[sec.drm.hpke-record-encryption]`).
pub fn parse_recipient_key_ids(
    data: &[u8],
    max_recipients: usize,
) -> Result<Vec<[u8; 32]>, DnamError> {
    reject_if(data.len() < 4, "invalid recipient vector")?;
    let count = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    reject_if(
        count == 0 || count > max_recipients || data.len() != 4 + 32 * count,
        "invalid recipient vector",
    )?;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let mut d = [0u8; 32];
        d.copy_from_slice(&data[4 + 32 * i..4 + 32 * (i + 1)]);
        out.push(d);
    }
    encode_recipient_key_ids(&out)?;
    Ok(out)
}

/// Pair of `(encapsulated_key_32b, encrypted_cek_and_tag_48b)` for an HPKE recipient slot.
pub type HpkeSlotPair = ([u8; 32], [u8; 48]);

/// Encodes `S = u32le(N) || (enc[32] || ct_and_tag[48])[0..N]` (`4 + 80 * N` bytes,
/// `ref[sec.drm.hpke-record-encryption]`).
pub fn encode_hpke_slot_vector(slots: &[HpkeSlotPair]) -> Result<Vec<u8>, DnamError> {
    reject_if(slots.is_empty(), "invalid slot vector")?;
    let mut out = Vec::with_capacity(4 + 80 * slots.len());
    out.extend_from_slice(&(slots.len() as u32).to_le_bytes());
    for (enc, ct) in slots {
        out.extend_from_slice(enc);
        out.extend_from_slice(ct);
    }
    Ok(out)
}

/// Parses and validates `S = u32le(N) || (enc[32] || ct_and_tag[48])[0..N]` (`ref[sec.drm.hpke-record-encryption]`).
pub fn parse_hpke_slot_vector(
    data: &[u8],
    expected_count: usize,
    max_recipients: usize,
) -> Result<Vec<HpkeSlotPair>, DnamError> {
    reject_if(data.len() < 4, "invalid slot vector")?;
    let count = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    reject_if(
        count == 0
            || count > max_recipients
            || count != expected_count
            || data.len() != 4 + 80 * count,
        "invalid slot vector",
    )?;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let base = 4 + 80 * i;
        let mut enc = [0u8; 32];
        let mut ct = [0u8; 48];
        enc.copy_from_slice(&data[base..base + 32]);
        ct.copy_from_slice(&data[base + 32..base + 80]);
        out.push((enc, ct));
    }
    Ok(out)
}

/// Encodes `D = u32le(C) || (u64le(offset) || u64le(ciphertext_length))[0..C]` (`C = max_tier + 2`,
/// `ref[sec.drm.hpke-record-encryption]`).
pub fn encode_record_directory(
    maximum_tier: u8,
    records: &[(u64, u64)],
) -> Result<Vec<u8>, DnamError> {
    reject_if(
        maximum_tier > 3 || records.len() != (maximum_tier as usize + 2),
        "invalid record directory",
    )?;
    let mut prev_end = 0u64;
    for (idx, &(offset, length)) in records.iter().enumerate() {
        reject_if(
            !offset.is_multiple_of(64) || !(16..=MAX_RECORD_CIPHERTEXT_BYTES).contains(&length),
            "invalid record directory",
        )?;
        let padded = align64_u64(length).or_schema_err("invalid record directory")?;
        let end = offset
            .checked_add(padded)
            .or_schema_err("invalid record directory")?;
        reject_if(idx == 1 && offset < prev_end, "invalid record directory")?;
        reject_if(idx > 1 && offset != prev_end, "invalid record directory")?;
        prev_end = end;
    }
    let mut out = Vec::with_capacity(4 + 16 * records.len());
    out.extend_from_slice(&(records.len() as u32).to_le_bytes());
    for &(off, len) in records {
        out.extend_from_slice(&off.to_le_bytes());
        out.extend_from_slice(&len.to_le_bytes());
    }
    Ok(out)
}

/// Parses and validates `D = u32le(C) || (u64le(offset) || u64le(ciphertext_length))[0..C]`
/// (`ref[sec.drm.hpke-record-encryption]`).
pub fn parse_record_directory(data: &[u8], maximum_tier: u8) -> Result<Vec<(u64, u64)>, DnamError> {
    reject_if(
        maximum_tier > 3 || data.len() < 4,
        "invalid record directory",
    )?;
    let count = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    reject_if(
        count != (maximum_tier as usize + 2) || data.len() != 4 + 16 * count,
        "invalid record directory",
    )?;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let base = 4 + 16 * i;
        let off = u64::from_le_bytes(data[base..base + 8].try_into().unwrap());
        let len = u64::from_le_bytes(data[base + 8..base + 16].try_into().unwrap());
        out.push((off, len));
    }
    encode_record_directory(maximum_tier, &out)?;
    Ok(out)
}

/// Builds the `"DNAM-PRE-WRAP-PLAN-V1"` pre-wrap plan `P` (`ref[sec.drm.hpke-record-encryption]`).
pub fn build_prewrap_plan_v1(
    issuance_id: &[u8; 16],
    layer: u8,
    profile: u8,
    maximum_tier: u8,
    structural_metadata_sha256: &[u8; 32],
    recipients: &[[u8; 32]],
    records: &[(u64, u64)],
) -> Result<Vec<u8>, DnamError> {
    reject_if(
        layer > 1 || !matches!(profile, 1 | 2) || (profile == 1 && layer != 0) || maximum_tier > 3,
        "invalid pre-wrap plan",
    )?;
    let rec_vec = encode_recipient_key_ids(recipients)?;
    let dir_vec = encode_record_directory(maximum_tier, records)?;
    let mut out = Vec::new();
    out.extend_from_slice(&encode_domain_field("domain", b"DNAM-PRE-WRAP-PLAN-V1"));
    out.extend_from_slice(&encode_domain_field("issuance_id", issuance_id));
    out.extend_from_slice(&encode_domain_field("layer", &[layer]));
    out.extend_from_slice(&encode_domain_field("protection_profile", &[profile]));
    out.extend_from_slice(&encode_domain_field("issued_max_tier", &[maximum_tier]));
    out.extend_from_slice(&encode_domain_field(
        "structural_metadata_sha256",
        structural_metadata_sha256,
    ));
    out.extend_from_slice(&encode_domain_field("mutable_offset", &0u64.to_le_bytes()));
    out.extend_from_slice(&encode_domain_field(
        "mutable_capacity",
        &0u64.to_le_bytes(),
    ));
    out.extend_from_slice(&encode_domain_field(
        "hpke_suite",
        &HPKE_SUITE_X25519_SHA256_CHACHA20POLY1305,
    ));
    out.extend_from_slice(&encode_domain_field("recipient_key_ids", &rec_vec));
    out.extend_from_slice(&encode_domain_field("record_descriptors", &dir_vec));
    Ok(out)
}

/// Computes `wrap_plan_sha256 = SHA256(F("domain", "DNAM-WRAP-PLAN-V1") || F("plan", plan))`
/// (`ref[sec.drm.hpke-record-encryption]`).
pub fn compute_wrap_plan_sha256(plan: &[u8]) -> [u8; 32] {
    let mut msg = Vec::new();
    msg.extend_from_slice(&encode_domain_field("domain", b"DNAM-WRAP-PLAN-V1"));
    msg.extend_from_slice(&encode_domain_field("plan", plan));
    sha256_digest(&msg)
}

/// Computes `record_context_sha256 = SHA256(F("domain", "DNAM-RECORD-CONTEXT-V1") || F("wrap_plan_sha256", ...) || F("slots", S))`
/// (`ref[sec.drm.hpke-record-encryption]`).
pub fn compute_record_context_sha256(wrap_plan_sha256: &[u8; 32], slots_vec: &[u8]) -> [u8; 32] {
    let mut msg = Vec::new();
    msg.extend_from_slice(&encode_domain_field("domain", b"DNAM-RECORD-CONTEXT-V1"));
    msg.extend_from_slice(&encode_domain_field("wrap_plan_sha256", wrap_plan_sha256));
    msg.extend_from_slice(&encode_domain_field("slots", slots_vec));
    sha256_digest(&msg)
}

/// Builds the HPKE Slot `i` Info or AAD binding (`domain` = `b"DNAM-HPKE-INFO-V1"` or `b"DNAM-HPKE-AAD-V1"`,
/// `ref[sec.drm.hpke-record-encryption]`).
pub fn build_hpke_slot_binding_v1(
    domain: &[u8],
    wrap_plan_sha256: &[u8; 32],
    slot_index: u32,
    recipient_key_sha256: &[u8; 32],
) -> Result<Vec<u8>, DnamError> {
    reject_if(
        domain != b"DNAM-HPKE-INFO-V1" && domain != b"DNAM-HPKE-AAD-V1",
        "invalid HPKE binding domain",
    )?;
    let mut out = Vec::new();
    out.extend_from_slice(&encode_domain_field("domain", domain));
    out.extend_from_slice(&encode_domain_field("wrap_plan_sha256", wrap_plan_sha256));
    out.extend_from_slice(&encode_domain_field(
        "slot_index",
        &slot_index.to_le_bytes(),
    ));
    out.extend_from_slice(&encode_domain_field(
        "recipient_key_sha256",
        recipient_key_sha256,
    ));
    Ok(out)
}

/// Builds the 12-byte little-endian record nonce `u64le(0) || u32le(record_index)` (`record_index <= 4`,
/// `ref[sec.drm.hpke-record-encryption]`).
pub fn build_record_nonce_v1(record_index: u32) -> Result<[u8; 12], DnamError> {
    reject_if(record_index > 4, "unsupported record index")?;
    let mut nonce = [0u8; 12];
    nonce[8..12].copy_from_slice(&record_index.to_le_bytes());
    Ok(nonce)
}

/// Builds the `"DNAM-RECORD-AAD-V1"` ChaCha20-Poly1305 AAD for record `record_index` (`0..=4`,
/// `ref[sec.drm.hpke-record-encryption]`).
pub fn build_record_aad_v1(
    record_context_sha256: &[u8; 32],
    record_index: u32,
    ciphertext_length: u64,
) -> Result<Vec<u8>, DnamError> {
    reject_if(
        record_index > 4 || !(16..=MAX_RECORD_CIPHERTEXT_BYTES).contains(&ciphertext_length),
        "invalid record AAD arguments",
    )?;
    let kind = u8::from(record_index != 0);
    let tier = if record_index == 0 {
        u8::MAX
    } else {
        (record_index - 1) as u8
    };
    let mut out = Vec::new();
    out.extend_from_slice(&encode_domain_field("domain", b"DNAM-RECORD-AAD-V1"));
    out.extend_from_slice(&encode_domain_field(
        "record_context_sha256",
        record_context_sha256,
    ));
    out.extend_from_slice(&encode_domain_field(
        "record_index",
        &record_index.to_le_bytes(),
    ));
    out.extend_from_slice(&encode_domain_field("record_kind", &[kind]));
    out.extend_from_slice(&encode_domain_field("tier", &[tier]));
    out.extend_from_slice(&encode_domain_field(
        "ciphertext_length",
        &ciphertext_length.to_le_bytes(),
    ));
    Ok(out)
}

/// Builds the 20-byte `DME1` encrypted mutable frame header (`ref[sec.drm.dme1-mutable-suffix]`).
pub fn build_dme1_header(
    profile: u8,
    recipient_count: u32,
    ciphertext_length: u64,
) -> Result<[u8; 20], DnamError> {
    let min_ct = 4u64 + 32 * (recipient_count as u64) + 18 + 16;
    reject_if(
        !matches!(profile, 1 | 2)
            || !(1..=16).contains(&recipient_count)
            || !(min_ct..=4_096).contains(&ciphertext_length),
        "invalid encrypted mutable header",
    )?;
    let mut hdr = [0u8; 20];
    hdr[0..4].copy_from_slice(b"DME1");
    hdr[4] = profile;
    hdr[8..12].copy_from_slice(&recipient_count.to_le_bytes());
    hdr[12..20].copy_from_slice(&ciphertext_length.to_le_bytes());
    Ok(hdr)
}

/// Computes `mutable_wrap_sha256` under `"DNAM-MUTABLE-WRAP-V1"` (`ref[sec.drm.dme1-mutable-suffix]`).
pub fn compute_mutable_wrap_sha256(
    record_context_sha256: &[u8; 32],
    header_20b: &[u8; 20],
    recipients: &[[u8; 32]],
) -> Result<[u8; 32], DnamError> {
    let rec_vec = encode_recipient_key_ids(recipients)?;
    let mut msg = Vec::new();
    msg.extend_from_slice(&encode_domain_field("domain", b"DNAM-MUTABLE-WRAP-V1"));
    msg.extend_from_slice(&encode_domain_field(
        "record_context_sha256",
        record_context_sha256,
    ));
    msg.extend_from_slice(&encode_domain_field("header", header_20b));
    msg.extend_from_slice(&encode_domain_field("recipient_key_ids", &rec_vec));
    Ok(sha256_digest(&msg))
}

/// Builds the `DME1` Slot `i` Info or AAD binding (`domain` = `b"DNAM-MUTABLE-HPKE-INFO-V1"` or
/// `b"DNAM-MUTABLE-HPKE-AAD-V1"`, `ref[sec.drm.dme1-mutable-suffix]`).
pub fn build_mutable_hpke_binding_v1(
    domain: &[u8],
    mutable_wrap_sha256: &[u8; 32],
    slot_index: u32,
    recipient_key_sha256: &[u8; 32],
) -> Result<Vec<u8>, DnamError> {
    reject_if(
        domain != b"DNAM-MUTABLE-HPKE-INFO-V1" && domain != b"DNAM-MUTABLE-HPKE-AAD-V1",
        "invalid mutable HPKE binding",
    )?;
    let mut out = Vec::new();
    out.extend_from_slice(&encode_domain_field("domain", domain));
    out.extend_from_slice(&encode_domain_field(
        "mutable_wrap_sha256",
        mutable_wrap_sha256,
    ));
    out.extend_from_slice(&encode_domain_field(
        "slot_index",
        &slot_index.to_le_bytes(),
    ));
    out.extend_from_slice(&encode_domain_field(
        "recipient_key_sha256",
        recipient_key_sha256,
    ));
    Ok(out)
}

/// Builds the `"DNAM-MUTABLE-CONTENT-V1"` AEAD AAD (`ref[sec.drm.dme1-mutable-suffix]`).
pub fn build_mutable_content_aad_v1(
    mutable_wrap_sha256: &[u8; 32],
    slots: &[([u8; 32], [u8; 48])],
) -> Result<Vec<u8>, DnamError> {
    let s_vec = encode_hpke_slot_vector(slots)?;
    let mut out = Vec::new();
    out.extend_from_slice(&encode_domain_field("domain", b"DNAM-MUTABLE-CONTENT-V1"));
    out.extend_from_slice(&encode_domain_field(
        "mutable_wrap_sha256",
        mutable_wrap_sha256,
    ));
    out.extend_from_slice(&encode_domain_field("slots", &s_vec));
    Ok(out)
}

/// Validates and parses the decrypted `DME1` plaintext `u32le(N) || recipient_public_key[0..N][32] || DMU1`
/// against the signed `recipients` (`SHA-256` key digests, `ref[sec.drm.dme1-mutable-suffix]`).
pub fn parse_dme1_decrypted_plaintext(
    plaintext: &[u8],
    recipients: &[[u8; 32]],
) -> Result<serde_json::Value, DnamError> {
    reject_if(plaintext.len() < 4, "invalid mutable plaintext")?;
    let count = u32::from_le_bytes(plaintext[0..4].try_into().unwrap()) as usize;
    let key_end = 4 + 32 * count;
    reject_if(
        count != recipients.len() || !(1..=16).contains(&count) || key_end > plaintext.len(),
        "invalid mutable plaintext",
    )?;
    let mut digests = Vec::with_capacity(count);
    for i in 0..count {
        let key = &plaintext[4 + 32 * i..4 + 32 * (i + 1)];
        digests.push(sha256_digest(key));
    }
    reject_if(digests.as_slice() != recipients, "recipient key mismatch")?;
    encode_recipient_key_ids(&digests)?;
    parse_and_validate_dmu1_frame(&plaintext[key_end..])
}

/// Computes `logical_metadata_sha256` over a `DNPL` logical metadata prefix (`"DNAM-LOGICAL-METADATA-V1"`,
/// `ref[sec.drm.protected-inner-framing]`).
pub fn compute_logical_metadata_sha256(dnpl_prefix: &[u8]) -> [u8; 32] {
    let mut msg = Vec::new();
    msg.extend_from_slice(&encode_domain_field("domain", b"DNAM-LOGICAL-METADATA-V1"));
    msg.extend_from_slice(&encode_domain_field("logical_prefix", dnpl_prefix));
    sha256_digest(&msg)
}

// ============================================================================
// Watermark Seed Derivation (`HMAC-SHA256`) & Stage III Tier-Local Fingerprint
// (`ref[sec.watermark.seed-derivation]`, `ref[sec.anticlone.stage3-fingerprint]`,
//  `ref[sec.anticlone.three-stage-gate]`)
// ============================================================================

/// Scope parameters for tier-scoped watermark seed derivation (`ref[sec.watermark.seed-derivation]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WatermarkSeedScope {
    pub marketplace_pubkey: [u8; 32],
    pub watermark_key_id: [u8; 16],
    pub creator_signed_manifest_sha256: [u8; 32],
    pub model_id: [u8; 16],
    pub watermark_profile_sha256: [u8; 32],
    pub owning_tier: u8,
}

/// Computes `watermark_profile_sha256` under `"DNAM-WM-PROFILE-V1"` (`ref[sec.watermark.seed-derivation]`).
pub fn compute_watermark_profile_sha256(profile_bytes: &[u8]) -> [u8; 32] {
    let mut msg = Vec::new();
    msg.extend_from_slice(&encode_domain_field("domain", b"DNAM-WM-PROFILE-V1"));
    msg.extend_from_slice(&encode_domain_field("profile", profile_bytes));
    sha256_digest(&msg)
}

/// Builds the domain-separated watermark seed message (`356 B` for `"DNAM-WM-SHARED-SEED-V1"`,
/// `429 B` for `"DNAM-WM-BUYER-SEED-V1"`, `ref[sec.watermark.seed-derivation]`).
pub fn build_watermark_seed_message_v1(
    scope: &WatermarkSeedScope,
    buyer_license_and_id: Option<(&[u8; 16], &[u8; 16])>,
) -> Result<Vec<u8>, DnamError> {
    reject_if(scope.owning_tier > 3, "unsupported owning tier")?;
    let domain: &[u8] = if buyer_license_and_id.is_none() {
        b"DNAM-WM-SHARED-SEED-V1"
    } else {
        b"DNAM-WM-BUYER-SEED-V1"
    };
    let mut msg = Vec::with_capacity(429);
    msg.extend_from_slice(&encode_domain_field("domain", domain));
    msg.extend_from_slice(&encode_domain_field(
        "marketplace_pubkey_ed25519",
        &scope.marketplace_pubkey,
    ));
    msg.extend_from_slice(&encode_domain_field(
        "watermark_key_id",
        &scope.watermark_key_id,
    ));
    msg.extend_from_slice(&encode_domain_field(
        "creator_signed_manifest_sha256",
        &scope.creator_signed_manifest_sha256,
    ));
    msg.extend_from_slice(&encode_domain_field("model_id", &scope.model_id));
    msg.extend_from_slice(&encode_domain_field(
        "watermark_profile_sha256",
        &scope.watermark_profile_sha256,
    ));
    msg.extend_from_slice(&encode_domain_field("owning_tier", &[scope.owning_tier]));
    if let Some((license_id, buyer_id)) = buyer_license_and_id {
        msg.extend_from_slice(&encode_domain_field("license_id", license_id));
        msg.extend_from_slice(&encode_domain_field("buyer_id", buyer_id));
    }
    Ok(msg)
}

/// Derives the 32-byte shared codebook or buyer-specific watermark seed via `HMAC-SHA256(key, msg)`
/// (`ref[sec.watermark.seed-derivation]`).
pub fn derive_watermark_seed_hmac_sha256(
    key: &[u8; 32],
    scope: &WatermarkSeedScope,
    buyer_license_and_id: Option<(&[u8; 16], &[u8; 16])>,
) -> Result<[u8; 32], DnamError> {
    let msg = build_watermark_seed_message_v1(scope, buyer_license_and_id)?;
    Ok(hmac_sha256(key, &msg))
}

/// Extracts the gauge-invariant Stage III tier-local fingerprint component (`60` `f64` values for
/// `Lite`/`Full`, `88` for `Pro`, `144` for `Ultra`; `60 / 120 / 208 / 352` cumulative values,
/// `ref[sec.anticlone.stage3-fingerprint]`).
pub fn extract_stage3_tier_fingerprint_component(
    early_fir_128b: &[u8],
    kautz_modal_bytes: &[u8],
    added_modes: usize,
) -> Result<Vec<f64>, DnamError> {
    reject_if(
        !matches!(added_modes, 4 | 8 | 16)
            || early_fir_128b.len() != 128
            || kautz_modal_bytes.len() != added_modes * 32,
        "incorrect tier-local slice length",
    )?;
    let mut values = Vec::with_capacity(32 + added_modes * 7);
    for &chunk in early_fir_128b.as_chunks::<4>().0 {
        let v = f32::from_le_bytes(chunk) as f64;
        reject_if(!v.is_finite(), "nonfinite FIR value")?;
        values.push(if v == 0.0 { 0.0 } else { v });
    }
    for block in kautz_modal_bytes.chunks(128) {
        let mut floats = [0.0f64; 32];
        for (i, &chunk) in block.as_chunks::<4>().0.iter().enumerate() {
            let v = f32::from_le_bytes(chunk) as f64;
            reject_if(!v.is_finite(), "nonfinite modal value")?;
            floats[i] = v;
        }
        for lane in 0..4 {
            let b_gamma = floats[12 + lane];
            let items = [
                floats[lane],
                floats[4 + lane],
                floats[8 + lane],
                b_gamma * floats[16 + lane],
                b_gamma * floats[20 + lane],
                b_gamma * floats[24 + lane],
                b_gamma * floats[28 + lane],
            ];
            for item in items {
                reject_if(!item.is_finite(), "nonfinite fingerprint product")?;
                values.push(if item == 0.0 { 0.0 } else { item });
            }
        }
    }
    Ok(values)
}

/// Decision outcome of the Stage B anti-cloning ingestion gate (`ref[sec.anticlone.three-stage-gate]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnticloneGateAction {
    /// Fingerprint did not match known protected models; proceed to Stage C / admit.
    NoMatchProceed,
    /// Buyer watermark matched, but no independent device-source evidence: flag license for review,
    /// never perform blind hardware device revocation.
    FlagLicenseHoldForReview,
    /// Buyer watermark matched AND independent device-source evidence confirmed: revoke identified
    /// device from future licensing only (existing offline issuances remain valid per `ref[sec.sig.offline-auth]`).
    RevokeIdentifiedDeviceFutureLicensingOnly,
}

/// Evaluates the evidence-based revocation rule in the 3-stage anti-cloning gate
/// (`ref[sec.anticlone.three-stage-gate]`, `ref[sec.sig.offline-auth]`).
#[inline]
pub const fn evaluate_anticlone_revocation_policy(
    fingerprint_matched: bool,
    buyer_watermark_identified: bool,
    independent_device_source_evidence: bool,
) -> AnticloneGateAction {
    if !fingerprint_matched || !buyer_watermark_identified {
        AnticloneGateAction::NoMatchProceed
    } else if independent_device_source_evidence {
        AnticloneGateAction::RevokeIdentifiedDeviceFutureLicensingOnly
    } else {
        AnticloneGateAction::FlagLicenseHoldForReview
    }
}

// ============================================================================
// Preview Containment, Royalty Splits & Storefront Metadata
// (`ref[sec.preview.containment]`, `ref[sec.preview.royalty-splits]`,
//  `ref[sec.preview.storefront-metadata]`)
// ============================================================================

/// Computes the bounded single-control preview window `[start, end] = [anchor - delta, anchor + delta] ∩ [safe_lo, safe_hi]`
/// (`0 < delta <= 0.15`, `0 <= safe_lo <= anchor <= safe_hi <= 1`, `start < end`, `ref[sec.preview.containment]`).
pub fn compute_preview_single_control_window(
    anchor: f64,
    delta: f64,
    safe_bounds: (f64, f64),
) -> Result<(f64, f64), DnamError> {
    let (lo, hi) = safe_bounds;
    reject_if(
        !(0.0 <= lo && lo <= anchor && anchor <= hi && hi <= 1.0 && delta > 0.0 && delta <= 0.15),
        "invalid preview bounds",
    )?;
    let start = lo.max(anchor - delta);
    let end = hi.min(anchor + delta);
    reject_if(start >= end, "no responsive preview interval")?;
    Ok((start, end))
}

/// Validates that a single-switch preview exposes at most two distinct options (`preset` and
/// `alternative`), both of which are observed, active, and safe (`ref[sec.preview.containment]`).
pub fn validate_preview_single_switch_options<'a>(
    preset: &'a str,
    alternative: &'a str,
    observed: &BTreeSet<&str>,
    active: &BTreeSet<&str>,
    safe: &BTreeSet<&str>,
) -> Result<(&'a str, &'a str), DnamError> {
    reject_if(preset == alternative, "preview options are not distinct")?;
    for opt in [preset, alternative] {
        reject_if(
            !observed.contains(opt) || !active.contains(opt) || !safe.contains(opt),
            "preview options are not distinct, observed, active and safe",
        )?;
    }
    Ok((preset, alternative))
}

/// Validates that `license_tier` is one of `"free"`, `"personal"`, `"commercial_studio"`, or
/// `"oem_bundle"`, and that `royalty_splits` sum to exactly `10,000` bps (`100%`, `ref[sec.preview.royalty-splits]`).
pub fn validate_license_tier_and_royalty_splits(
    license_tier: &str,
    splits: &[RoyaltySplit],
) -> Result<(), DnamError> {
    reject_if(
        !matches!(
            license_tier,
            "free" | "personal" | "commercial_studio" | "oem_bundle"
        ),
        "invalid license_tier",
    )?;
    reject_if(splits.is_empty(), "empty royalty_splits")?;
    let mut total_bps = 0u32;
    for s in splits {
        reject_if(
            s.recipient_id.is_empty() || s.role.is_empty() || s.share_bps == 0,
            "invalid royalty_split entry",
        )?;
        total_bps += s.share_bps as u32;
    }
    reject_if(total_bps != 10_000, "royalty_splits must sum to 10000 bps")?;
    Ok(())
}

/// Extracts the `"dnam.store"` storefront metadata object from `user_metadata` (`ref[sec.preview.storefront-metadata]`).
pub fn extract_storefront_metadata(
    user_metadata: &serde_json::Map<String, serde_json::Value>,
) -> Option<&serde_json::Map<String, serde_json::Value>> {
    user_metadata.get("dnam.store")?.as_object()
}

#[cfg(test)]
mod spec_tests {
    #[allow(unused_imports)]
    use alloc::{
        boxed::Box,
        format,
        string::{String, ToString},
        vec,
        vec::Vec,
    };

    // Specification Verification Suite: Phase 3 — Security, Provenance, Enclave DRM, Watermarking & Marketplace.
    //
    // Verifies all rules in `docs/specs/security/provenance_and_enclave_drm.md` and
    // `docs/specs/security/watermarking_and_marketplace.md`:
    // - `verify[sec.taxonomy.overview]`
    // - `verify[sec.jcs.immutable-header-digest]`
    // - `verify[sec.policy.distribution-policy]`
    // - `verify[sec.sig.prefix-delegation]`
    // - `verify[sec.sig.dau1-envelope]`
    // - `verify[sec.sig.offline-auth]`
    // - `verify[sec.drm.oem-certificate]`
    // - `verify[sec.drm.trustzone-flow]`
    // - `verify[sec.drm.hpke-record-encryption]`
    // - `verify[sec.drm.dnsl-public-framing]`
    // - `verify[sec.drm.protected-inner-framing]`
    // - `verify[sec.drm.dme1-mutable-suffix]`
    // - `verify[sec.drm.protection-profiles]`
    // - `verify[sec.watermark.reciprocal-gauge]`
    // - `verify[sec.watermark.orthogonal-gauge]`
    // - `verify[sec.watermark.tardos-code]`
    // - `verify[sec.watermark.seed-derivation]`
    // - `verify[sec.anticlone.three-stage-gate]`
    // - `verify[sec.anticlone.stage3-fingerprint]`
    // - `verify[sec.preview.containment]`
    // - `verify[sec.preview.royalty-splits]`
    // - `verify[sec.preview.storefront-metadata]`
    // - `verify[sec.edge.data-model]`

    use std::collections::BTreeSet;

    use crate::container::{ARENA_OFFSET_ULTRA_END, build_dmu1_mutable_frame};
    use crate::provenance::{
        AnticloneGateAction, Dau1Envelope, WatermarkSeedScope, apply_isometric_gauge_watermark,
        apply_tardos_watermark_7680, build_buyer_receipt_v1, build_creator_manifest_v1,
        build_dme1_header, build_hpke_slot_binding_v1, build_mutable_content_aad_v1,
        build_mutable_hpke_binding_v1, build_oem_cert_v2_signing_message, build_outer_manifest_v1,
        build_plaintext_shared_v1, build_prewrap_plan_v1, build_record_aad_v1,
        build_record_nonce_v1, build_watermark_seed_message_v1, compute_ciphertext_prefixes_v1,
        compute_creator_signed_manifest_sha256, compute_encrypted_structural_metadata_sha256,
        compute_immutable_header_sha256, compute_license_policy_sha256,
        compute_logical_metadata_sha256, compute_mutable_wrap_sha256,
        compute_outer_public_metadata_sha256, compute_plaintext_prefixes_v1,
        compute_preview_single_control_window, compute_record_context_sha256,
        compute_watermark_profile_sha256, compute_wrap_plan_sha256,
        derive_watermark_seed_hmac_sha256, encode_hpke_slot_vector, encode_prefix_list,
        encode_recipient_key_ids, encode_record_directory, enforce_distribution_policy,
        evaluate_anticlone_revocation_policy, extract_stage3_tier_fingerprint_component,
        extract_storefront_metadata, extract_structural_hash_224d, generate_tardos_code_7680,
        kautz_soa_byte_offset, parse_dme1_decrypted_plaintext, parse_hpke_slot_vector,
        parse_prefix_list, parse_recipient_key_ids, parse_record_directory,
        score_tardos_watermark_7680, sha256_digest, structural_cosine_similarity,
        validate_license_tier_and_royalty_splits, validate_preview_single_switch_options,
    };
    use crate::schema::{
        DistributionPolicy, DnamError, DnamJsonHeader, DnamProtectionProfile, RoyaltySplit,
    };
    use serde_json::json;

    /// verify[sec.taxonomy.overview]
    /// verify[sec.jcs.immutable-header-digest]
    /// verify[sec.policy.distribution-policy]
    /// verify[sec.drm.protection-profiles]
    /// Verifies the 3 protection profiles (`Open`, `DeviceAllowlisted`, `EnclaveSealed`),
    /// preset-safe JCS `immutable_header_sha256`, and `DistributionPolicy` enforcement.
    #[test]
    fn test_security_taxonomy_jcs_digest_and_distribution_policy() {
        assert_eq!(DnamProtectionProfile::Open as u8, 0);
        assert_eq!(DnamProtectionProfile::DeviceAllowlisted as u8, 1);
        assert_eq!(DnamProtectionProfile::EnclaveSealed as u8, 2);

        let mut header = DnamJsonHeader::default_reference();
        let d0 = compute_immutable_header_sha256(&header).unwrap();
        header
            .user_metadata
            .insert("custom".to_string(), json!("value"));
        let d1 = compute_immutable_header_sha256(&header).unwrap();
        assert_eq!(d0, d1);

        // DistributionPolicy enforcement on protection profile & OEM vendor:model rules
        header.metadata.distribution_policy = Some(DistributionPolicy {
            allowed_protection_profiles: vec![DnamProtectionProfile::EnclaveSealed],
            allowed_oem_vendors: vec!["darkglass:anagram_rk3582".to_string()],
        });
        assert!(enforce_distribution_policy(&header, DnamProtectionProfile::Open).is_err());
        assert!(
            enforce_distribution_policy(&header, DnamProtectionProfile::DeviceAllowlisted).is_err()
        );
    }

    /// verify[sec.sig.prefix-delegation]
    /// verify[sec.sig.dau1-envelope]
    /// verify[sec.sig.offline-auth]
    /// Verifies `DNAM-PLAINTEXT-SHARED-V1`, `DNAM-MASTER-PREFIX-V1`, `DNAM-CREATOR-MANIFEST-V1`,
    /// `DNAM-BUYER-RECEIPT-V1`, `DAU1` (`auth_kind = 0` and `1`) wire round-trips, and indefinite
    /// offline authorization semantics.
    #[test]
    fn test_prefix_delegation_dau1_envelope_and_offline_auth() {
        let shared = build_plaintext_shared_v1(&[0x11; 64], &[0x22; 128], &[0x33; 64]);
        let header_sha = [0x44u8; 32];
        let tiers: [&[u8]; 4] = [&[1; 64], &[2; 64], &[3; 64], &[4; 64]];
        let master_prefixes =
            compute_plaintext_prefixes_v1(b"DNAM-MASTER-PREFIX-V1", &header_sha, &shared, &tiers)
                .unwrap();
        assert_eq!(master_prefixes.len(), 4);

        // Prefix list encoding and parsing
        let encoded_list = encode_prefix_list(&master_prefixes).unwrap();
        assert_eq!(encoded_list.len(), 1 + 33 * 4);
        assert_eq!(
            parse_prefix_list(&encoded_list, 3).unwrap(),
            master_prefixes
        );

        let creator_pk = [0x55u8; 32];
        let creator_sig = [0x66u8; 64];
        let creator_msg =
            build_creator_manifest_v1(&creator_pk, &header_sha, &master_prefixes).unwrap();
        let creator_id = compute_creator_signed_manifest_sha256(&creator_msg, &creator_sig);

        let policy_bytes = b"tier=pro;offline=forever";
        let policy_sha = compute_license_policy_sha256(policy_bytes);
        let buyer_prefixes = compute_plaintext_prefixes_v1(
            b"DNAM-BUYER-PREFIX-V1",
            &header_sha,
            &shared,
            &tiers[..3],
        )
        .unwrap();
        let market_pk = [0x77u8; 32];
        let market_sig = [0x88u8; 64];
        let buyer_msg = build_buyer_receipt_v1(
            &market_pk,
            &creator_id,
            &policy_sha,
            0,
            &buyer_prefixes,
            None,
            &[],
        )
        .unwrap();
        assert!(!buyer_msg.is_empty());

        // DAU1 auth_kind = 0 (106 + 33 * 4 = 238 B)
        let dau0 = Dau1Envelope {
            auth_kind: 0,
            creator_pubkey: creator_pk,
            creator_max_tier: 3,
            master_prefixes: master_prefixes.clone(),
            creator_signature: creator_sig,
            marketplace_pubkey: None,
            policy_bytes: Vec::new(),
            buyer_max_tier: None,
            buyer_prefixes: Vec::new(),
            marketplace_signature: None,
        };
        let dau0_bytes = dau0.to_bytes().unwrap();
        assert_eq!(dau0_bytes.len(), 106 + 33 * 4);
        assert_eq!(Dau1Envelope::from_bytes(&dau0_bytes).unwrap(), dau0);

        // DAU1 auth_kind = 1 (208 + 33 * 4 + 33 * 3 + P)
        let dau1 = Dau1Envelope {
            auth_kind: 1,
            creator_pubkey: creator_pk,
            creator_max_tier: 3,
            master_prefixes,
            creator_signature: creator_sig,
            marketplace_pubkey: Some(market_pk),
            policy_bytes: policy_bytes.to_vec(),
            buyer_max_tier: Some(2),
            buyer_prefixes,
            marketplace_signature: Some(market_sig),
        };
        let dau1_bytes = dau1.to_bytes().unwrap();
        assert_eq!(dau1_bytes.len(), 208 + 33 * 4 + 33 * 3 + policy_bytes.len());
        assert_eq!(Dau1Envelope::from_bytes(&dau1_bytes).unwrap(), dau1);
    }

    /// verify[sec.drm.oem-certificate]
    /// verify[sec.drm.trustzone-flow]
    /// Verifies the exact V2 `OemDeviceCertificate` signing message bytes and canonical identifier validation.
    #[test]
    fn test_oem_certificate_v2_signing_message_and_identifier_validation() {
        let dev_id: Vec<u8> = (0..16u8).collect();
        let pk_x25519: Vec<u8> = (0..32u8).collect();
        let issuer_pk = [0u8; 32];
        let policy = [b'P'; 32];
        let evidence = [b'E'; 32];
        let msg = build_oem_cert_v2_signing_message(
            "acme", "pedal-v1", &dev_id, &pk_x25519, &issuer_pk, 1, &policy, &evidence,
        )
        .unwrap();

        let expected_issuer_sha = [
            0x66, 0x68, 0x7a, 0xad, 0xf8, 0x62, 0xbd, 0x77, 0x6c, 0x8f, 0xc1, 0x8b, 0x8e, 0x9f,
            0x8e, 0x20, 0x08, 0x97, 0x14, 0x85, 0x6e, 0xe2, 0x33, 0xb3, 0x90, 0x2a, 0x59, 0x1d,
            0x0d, 0x5f, 0x29, 0x25,
        ];
        let mut expected = Vec::new();
        expected.extend_from_slice(
            b"\x06\x00\x00\x00domain\x10\x00\x00\x00\x00\x00\x00\x00DNAM-OEM-CERT-V2",
        );
        expected
            .extend_from_slice(b"\x0a\x00\x00\x00oem_vendor\x04\x00\x00\x00\x00\x00\x00\x00acme");
        expected.extend_from_slice(
            b"\x0a\x00\x00\x00model_code\x08\x00\x00\x00\x00\x00\x00\x00pedal-v1",
        );
        expected
            .extend_from_slice(b"\x0e\x00\x00\x00device_id_hash\x10\x00\x00\x00\x00\x00\x00\x00");
        expected.extend_from_slice(&dev_id);
        expected.extend_from_slice(
            b"\x11\x00\x00\x00public_key_x25519\x20\x00\x00\x00\x00\x00\x00\x00",
        );
        expected.extend_from_slice(&pk_x25519);
        expected.extend_from_slice(
            b"\x11\x00\x00\x00issuer_key_sha256\x20\x00\x00\x00\x00\x00\x00\x00",
        );
        expected.extend_from_slice(&expected_issuer_sha);
        expected
            .extend_from_slice(b"\x09\x00\x00\x00assurance\x01\x00\x00\x00\x00\x00\x00\x00\x01");
        expected.extend_from_slice(
            b"\x19\x00\x00\x00attestation_policy_sha256\x20\x00\x00\x00\x00\x00\x00\x00",
        );
        expected.extend_from_slice(&policy);
        expected.extend_from_slice(
            b"\x1b\x00\x00\x00attestation_evidence_sha256\x20\x00\x00\x00\x00\x00\x00\x00",
        );
        expected.extend_from_slice(&evidence);
        assert_eq!(msg, expected);

        // Rejects delimiter-ambiguous or uppercase identifiers and invalid assurance codes
        for (bad_vendor, bad_model) in [
            ("", "pedal"),
            ("Acme", "pedal"),
            ("acme|pedal", "mk2"),
            ("acme", "Pedal"),
            ("acme", "pedal|mk2"),
        ] {
            assert!(
                build_oem_cert_v2_signing_message(
                    bad_vendor, bad_model, &dev_id, &pk_x25519, &issuer_pk, 1, &policy, &evidence,
                )
                .is_err()
            );
        }
        assert!(
            build_oem_cert_v2_signing_message(
                "acme", "pedal", &dev_id, &pk_x25519, &issuer_pk, 0, &policy, &evidence,
            )
            .is_err()
        );
    }

    /// verify[sec.drm.hpke-record-encryption]
    /// verify[sec.drm.dnsl-public-framing]
    /// verify[sec.drm.protected-inner-framing]
    /// verify[sec.drm.dme1-mutable-suffix]
    /// Verifies encrypted structural metadata digests, pre-wrap plan `P`, record context, HPKE slot
    /// bindings, record nonces/AAD, `DNSL` outer manifest, `DNPL` logical metadata, and `DME1` mutable suffix.
    #[test]
    fn test_hpke_record_context_dnsl_dnpl_and_dme1_framing() {
        let s0 = compute_encrypted_structural_metadata_sha256(0, b"p", b"dd").unwrap();
        let s1 = compute_encrypted_structural_metadata_sha256(1, b"p", b"dd").unwrap();
        assert_ne!(s0, s1);

        let mut k0 = [0u8; 32];
        k0[0] = 0x09;
        let mut k1 = [0u8; 32];
        k1[0] = 0x0a;
        let mut keys = [k0, k1];
        keys.sort_by_key(|k| sha256_digest(k));
        let recipients = [sha256_digest(&keys[0]), sha256_digest(&keys[1])];
        let rec_vec = encode_recipient_key_ids(&recipients).unwrap();
        assert_eq!(
            parse_recipient_key_ids(&rec_vec, 16).unwrap(),
            recipients.to_vec()
        );

        let slots = [([0xaa; 32], [0xbb; 48]), ([0xcc; 32], [0xdd; 48])];
        let slots_vec = encode_hpke_slot_vector(&slots).unwrap();
        assert_eq!(
            parse_hpke_slot_vector(&slots_vec, 2, 16).unwrap(),
            slots.to_vec()
        );

        let records = [(640u64, 48u64), (704, 17), (768, 100)];
        let rec_dir = encode_record_directory(1, &records).unwrap();
        assert_eq!(
            parse_record_directory(&rec_dir, 1).unwrap(),
            records.to_vec()
        );

        let plan = build_prewrap_plan_v1(&[b'i'; 16], 1, 2, 1, &s1, &recipients, &records).unwrap();
        let plan_sha = compute_wrap_plan_sha256(&plan);
        let ctx_sha = compute_record_context_sha256(&plan_sha, &slots_vec);

        let info0 =
            build_hpke_slot_binding_v1(b"DNAM-HPKE-INFO-V1", &plan_sha, 0, &recipients[0]).unwrap();
        let aad0 =
            build_hpke_slot_binding_v1(b"DNAM-HPKE-AAD-V1", &plan_sha, 0, &recipients[0]).unwrap();
        assert_ne!(info0, aad0);

        assert_eq!(
            build_record_nonce_v1(3).unwrap(),
            [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0]
        );
        assert!(build_record_nonce_v1(5).is_err());
        let rec_aad0 = build_record_aad_v1(&ctx_sha, 0, 64).unwrap();
        let rec_aad1 = build_record_aad_v1(&ctx_sha, 1, 64).unwrap();
        assert_ne!(rec_aad0, rec_aad1);

        // Outer DNSL public metadata and manifest
        let ct_prefixes =
            compute_ciphertext_prefixes_v1(&ctx_sha, &[1u8; 64], &[&[2u8; 64], &[3u8; 64]])
                .unwrap();
        let outer_meta_sha = compute_outer_public_metadata_sha256(&[0; 64], &[0; 128], &plan);
        let outer_msg =
            build_outer_manifest_v1(&[b'k'; 32], &outer_meta_sha, &ctx_sha, &ct_prefixes).unwrap();
        assert!(!outer_msg.is_empty());

        // Inner DNPL logical metadata digest
        let dnpl_sha = compute_logical_metadata_sha256(b"DNPL_PREFIX");
        assert_ne!(dnpl_sha, [0u8; 32]);

        // DME1 encrypted mutable suffix header, wrap digest, bindings, and decrypted plaintext validation
        let dmu1 = build_dmu1_mutable_frame(br#"{"preset":"clean"}"#, 34).unwrap();
        let mut dme1_pt = Vec::new();
        dme1_pt.extend_from_slice(&2u32.to_le_bytes());
        dme1_pt.extend_from_slice(&keys[0]);
        dme1_pt.extend_from_slice(&keys[1]);
        dme1_pt.extend_from_slice(&dmu1);

        let dme1_hdr = build_dme1_header(2, 2, (dme1_pt.len() + 16) as u64).unwrap();
        assert_eq!(&dme1_hdr[0..4], b"DME1");
        let m_wrap_sha = compute_mutable_wrap_sha256(&ctx_sha, &dme1_hdr, &recipients).unwrap();
        let m_info = build_mutable_hpke_binding_v1(
            b"DNAM-MUTABLE-HPKE-INFO-V1",
            &m_wrap_sha,
            0,
            &recipients[0],
        )
        .unwrap();
        let m_aad = build_mutable_hpke_binding_v1(
            b"DNAM-MUTABLE-HPKE-AAD-V1",
            &m_wrap_sha,
            0,
            &recipients[0],
        )
        .unwrap();
        assert_ne!(m_info, m_aad);
        let c_aad = build_mutable_content_aad_v1(&m_wrap_sha, &slots).unwrap();
        assert!(!c_aad.is_empty());

        let parsed_doc = parse_dme1_decrypted_plaintext(&dme1_pt, &recipients).unwrap();
        assert_eq!(parsed_doc, json!({"preset": "clean"}));

        // Reversing recipient keys in plaintext is rejected
        let mut reversed_pt = Vec::new();
        reversed_pt.extend_from_slice(&2u32.to_le_bytes());
        reversed_pt.extend_from_slice(&keys[1]);
        reversed_pt.extend_from_slice(&keys[0]);
        reversed_pt.extend_from_slice(&dmu1);
        assert!(parse_dme1_decrypted_plaintext(&reversed_pt, &recipients).is_err());
    }

    /// verify[sec.watermark.reciprocal-gauge]
    /// verify[sec.watermark.orthogonal-gauge]
    /// verify[sec.watermark.tardos-code]
    /// verify[sec.watermark.seed-derivation]
    /// verify[sec.anticlone.three-stage-gate]
    /// verify[sec.anticlone.stage3-fingerprint]
    /// Verifies OpenSSL watermark seed derivation vectors, Mechanism A reciprocal + SO(4) gauge
    /// watermarking, Mechanism B Tardos spread-spectrum watermarking, Stage III gauge-invariant
    /// tier-local fingerprinting, and evidence-based anti-cloning revocation policy.
    #[test]
    fn test_watermark_seed_derivation_gauges_tardos_and_stage3_fingerprint() {
        // 1. Exact OpenSSL vectors from specification
        let key: [u8; 32] = core::array::from_fn(|i| i as u8);
        let scope = WatermarkSeedScope {
            marketplace_pubkey: core::array::from_fn(|i| (32 + i) as u8),
            watermark_key_id: core::array::from_fn(|i| (64 + i) as u8),
            creator_signed_manifest_sha256: core::array::from_fn(|i| (80 + i) as u8),
            model_id: core::array::from_fn(|i| (112 + i) as u8),
            watermark_profile_sha256: core::array::from_fn(|i| (128 + i) as u8),
            owning_tier: 2,
        };
        let license_id: [u8; 16] = [
            0x01, 0x96, 0x6c, 0x44, 0x47, 0xe5, 0x7a, 0xbc, 0x81, 0x23, 0x45, 0x67, 0x89, 0xab,
            0xcd, 0xef,
        ];
        let buyer_id: [u8; 16] = [
            0x01, 0x96, 0x6c, 0x44, 0x47, 0xe5, 0x7d, 0xef, 0xb0, 0x12, 0x34, 0x56, 0x78, 0x9a,
            0xbc, 0xde,
        ];

        let shared_msg = build_watermark_seed_message_v1(&scope, None).unwrap();
        assert_eq!(shared_msg.len(), 356);
        let shared_msg_hex: String = sha256_digest(&shared_msg)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(
            shared_msg_hex,
            "13d2bee542d18355e05b8e75faabbd9e64c1d245f485bc7e30c0057cc0239477"
        );

        let shared_seed = derive_watermark_seed_hmac_sha256(&key, &scope, None).unwrap();
        let shared_seed_hex: String = shared_seed.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            shared_seed_hex,
            "3c29ae4f95ccf1db51d722da34801dec3813809b6ffe3cdf9af4de9b0270cf4c"
        );

        let buyer_msg =
            build_watermark_seed_message_v1(&scope, Some((&license_id, &buyer_id))).unwrap();
        assert_eq!(buyer_msg.len(), 429);
        let buyer_msg_hex: String = sha256_digest(&buyer_msg)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(
            buyer_msg_hex,
            "a31c8b7430f9ba153c0a7c838ed9ff788351f9bb8c597e75da737dd18927c118"
        );

        let buyer_seed =
            derive_watermark_seed_hmac_sha256(&key, &scope, Some((&license_id, &buyer_id)))
                .unwrap();
        let buyer_seed_hex: String = buyer_seed.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            buyer_seed_hex,
            "04a710a6112a44082e84290642093fe72df7644c168b1552d8ab9e4d22a575b3"
        );

        let prof_hex: String = compute_watermark_profile_sha256(b"profile\x00v1\xff")
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(
            prof_hex,
            "ada5b074a45838ba487d6ac9bc2c11d6b781872a07d78aa3e689841338895186"
        );

        // 2. Stage III tier-local fingerprint invariance under reciprocal gauge scaling
        let mut fir_128b = [0u8; 128];
        for i in 0..32 {
            let v = ((i + 1) as f32) / 64.0;
            fir_128b[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
        }
        let build_modal_block = |eta: f32| -> [u8; 128] {
            let mut blk = [0u8; 128];
            let fields: [[f32; 4]; 8] = [
                [0.9, 0.91, 0.92, 0.93],
                [0.02, 0.02, 0.02, 0.02],
                [0.1, 0.2, 0.3, 0.4],
                [0.5 * eta, 0.5 * eta, 0.5 * eta, 0.5 * eta],
                [0.25 / eta, 0.5 / eta, 0.75 / eta, 1.0 / eta],
                [-0.125 / eta, -0.25 / eta, -0.375 / eta, -0.5 / eta],
                [0.125 / eta, 0.25 / eta, 0.375 / eta, 0.5 / eta],
                [-0.0625 / eta, -0.125 / eta, -0.1875 / eta, -0.25 / eta],
            ];
            for (f_idx, lane_vals) in fields.iter().enumerate() {
                for (l_idx, &v) in lane_vals.iter().enumerate() {
                    let off = (f_idx * 4 + l_idx) * 4;
                    blk[off..off + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
            blk
        };
        let fp_master =
            extract_stage3_tier_fingerprint_component(&fir_128b, &build_modal_block(1.0), 4)
                .unwrap();
        let fp_buyer =
            extract_stage3_tier_fingerprint_component(&fir_128b, &build_modal_block(2.0), 4)
                .unwrap();
        assert_eq!(fp_master.len(), 60);
        assert_eq!(fp_master, fp_buyer);

        // 3. Full 64,000 B arena gauge + Tardos watermarking & 224-D structural hash
        let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        for t in 0..4 {
            for blk in 0..4 {
                let base = kautz_soa_byte_offset(t, blk);
                for l in 0..4 {
                    arena[base + l * 4..base + (l + 1) * 4].copy_from_slice(&0.9_f32.to_le_bytes());
                    arena[base + (12 + l) * 4..base + (13 + l) * 4]
                        .copy_from_slice(&0.5_f32.to_le_bytes());
                    arena[base + (16 + l) * 4..base + (17 + l) * 4]
                        .copy_from_slice(&0.25_f32.to_le_bytes());
                }
            }
        }
        let h_before = extract_structural_hash_224d(&arena).unwrap();
        let kappa = [1.0_f32; 64];
        let q_id = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        apply_isometric_gauge_watermark(&mut arena, &kappa, &q_id).unwrap();
        let h_after = extract_structural_hash_224d(&arena).unwrap();
        assert!((structural_cosine_similarity(&h_before, &h_after) - 1.0).abs() < 1e-12);

        let master_arena = arena.clone();
        let code = generate_tardos_code_7680(&buyer_seed, "model_01", "license_01");
        apply_tardos_watermark_7680(&mut arena, &code, 1e-3).unwrap();
        let score = score_tardos_watermark_7680(&master_arena, &arena, &code).unwrap();
        assert!(score > 10.0);

        // 4. Evidence-based anti-cloning revocation policy
        assert_eq!(
            evaluate_anticlone_revocation_policy(false, false, false),
            AnticloneGateAction::NoMatchProceed
        );
        assert_eq!(
            evaluate_anticlone_revocation_policy(true, true, false),
            AnticloneGateAction::FlagLicenseHoldForReview
        );
        assert_eq!(
            evaluate_anticlone_revocation_policy(true, true, true),
            AnticloneGateAction::RevokeIdentifiedDeviceFutureLicensingOnly
        );
    }

    /// verify[sec.preview.containment]
    /// verify[sec.preview.royalty-splits]
    /// verify[sec.preview.storefront-metadata]
    /// verify[sec.edge.data-model]
    /// Verifies `.dnam-prev` host rejection (`DnamError::PreviewContainerRejected`), bounded preview
    /// control windows (`delta <= 0.15`), switch option validation, `royalty_splits` 10,000 bps check,
    /// and `"dnam.store"` storefront metadata extraction.
    #[test]
    fn test_preview_containment_royalty_splits_and_storefront_metadata() {
        // 1. Preview container rejection in DAW/Hardware host loader
        let mut header = DnamJsonHeader::default_reference();
        header.container_kind = "preview".to_string();
        let err = header
            .ensure_not_preview_for_daw_host()
            .expect_err("DAW host must reject preview container");
        assert!(matches!(err, DnamError::PreviewContainerRejected));

        // 2. Bounded single-control preview window
        assert_eq!(
            compute_preview_single_control_window(0.0, 0.15, (0.0, 1.0)).unwrap(),
            (0.0, 0.15)
        );
        assert_eq!(
            compute_preview_single_control_window(0.5, 0.15, (0.4, 0.6)).unwrap(),
            (0.4, 0.6)
        );
        assert!(compute_preview_single_control_window(0.5, 0.20, (0.0, 1.0)).is_err());
        assert!(compute_preview_single_control_window(0.5, 0.10, (0.5, 0.5)).is_err());

        // 3. Single-switch preview options
        let obs = BTreeSet::from(["a", "b", "c"]);
        let act = BTreeSet::from(["a", "b"]);
        let safe = BTreeSet::from(["a", "b"]);
        assert_eq!(
            validate_preview_single_switch_options("a", "b", &obs, &act, &safe).unwrap(),
            ("a", "b")
        );
        assert!(validate_preview_single_switch_options("a", "c", &obs, &act, &safe).is_err());
        assert!(validate_preview_single_switch_options("a", "a", &obs, &act, &safe).is_err());

        // 4. Royalty splits (10,000 bps) & license_tier validation
        let splits = vec![
            RoyaltySplit {
                recipient_id: "creator_1".to_string(),
                role: "creator".to_string(),
                share_bps: 7_000,
            },
            RoyaltySplit {
                recipient_id: "platform".to_string(),
                role: "marketplace".to_string(),
                share_bps: 3_000,
            },
        ];
        validate_license_tier_and_royalty_splits("commercial_studio", &splits).unwrap();
        assert!(validate_license_tier_and_royalty_splits("unknown_tier", &splits).is_err());
        assert!(validate_license_tier_and_royalty_splits("personal", &splits[..1]).is_err());

        // 5. Storefront metadata extraction under user_metadata["dnam.store"]
        let um = json!({
            "dnam.store": {
                "product_slug": "b7k-ultra",
                "category": "bass-drive"
            }
        });
        let store = extract_storefront_metadata(um.as_object().unwrap()).unwrap();
        assert_eq!(
            store.get("product_slug").and_then(|v| v.as_str()),
            Some("b7k-ultra")
        );
    }

    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// verify[sec.sig.prefix-delegation]
        /// verify[sec.drm.hpke-record-encryption]
        /// verify[sec.watermark.seed-derivation]
        #[test]
        fn prop_dau1_roundtrip_hpke_nonce_and_watermark_domain_separation(
            creator_pk in prop::array::uniform32(any::<u8>()),
            hdr_sha in prop::array::uniform32(any::<u8>()),
            shared_bytes in prop::collection::vec(any::<u8>(), 1..128),
            salt_96 in prop::array::uniform12(any::<u8>()),
            record_id in 0u8..5u8,
            chunk_idx in 0u32..1000u32,
            wm_key in prop::array::uniform32(any::<u8>()),
            owning_tier in 0u8..=3u8,
        ) {
            let t0 = [0xAAu8; 64];
            let pfxs = compute_plaintext_prefixes_v1(
                b"DNAM-MASTER-PREFIX-V1",
                &hdr_sha,
                &shared_bytes,
                &[&t0],
            ).unwrap();
            let dau = Dau1Envelope {
                auth_kind: 0,
                creator_pubkey: creator_pk,
                creator_max_tier: 0,
                master_prefixes: pfxs,
                creator_signature: [0x55u8; 64],
                marketplace_pubkey: None,
                policy_bytes: Vec::new(),
                buyer_max_tier: None,
                buyer_prefixes: Vec::new(),
                marketplace_signature: None,
            };
            let wire = dau.to_bytes().unwrap();
            prop_assert_eq!(Dau1Envelope::from_bytes(&wire).unwrap(), dau);

            let n1 = build_record_nonce_v1(record_id as u32).unwrap();
            let n2 = build_record_nonce_v1(((record_id as u32) + 1) % 5).unwrap();
            prop_assert_ne!(n1, n2);
            let _ = (salt_96, chunk_idx);

            let scope = WatermarkSeedScope {
                marketplace_pubkey: creator_pk,
                watermark_key_id: [1u8; 16],
                creator_signed_manifest_sha256: hdr_sha,
                model_id: [2u8; 16],
                watermark_profile_sha256: hdr_sha,
                owning_tier,
            };
            let s_shared = derive_watermark_seed_hmac_sha256(&wm_key, &scope, None).unwrap();
            let s_buyer = derive_watermark_seed_hmac_sha256(
                &wm_key,
                &scope,
                Some((&[3u8; 16], &[4u8; 16])),
            ).unwrap();
            prop_assert_ne!(s_shared, s_buyer);
        }
    }
}

#[cfg(all(test, feature = "crypto"))]
mod prefix_reference_tests {
    #[allow(unused_imports)]
    use alloc::{
        boxed::Box,
        format,
        string::{String, ToString},
        vec,
        vec::Vec,
    };

    // Signature references, not tests of the current production container verifier.

    use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
    use sha2::{Digest, Sha256};

    fn field(label: &str, value: &[u8]) -> Vec<u8> {
        let mut output = Vec::new();
        output.extend_from_slice(&u32::try_from(label.len()).unwrap().to_le_bytes());
        output.extend_from_slice(label.as_bytes());
        output.extend_from_slice(&u64::try_from(value.len()).unwrap().to_le_bytes());
        output.extend_from_slice(value);
        output
    }

    fn plaintext_shared(preamble: &[u8], directory: &[u8], shared_binary: &[u8]) -> Vec<u8> {
        [
            field("domain", b"DNAM-PLAINTEXT-SHARED-V1"),
            field("preamble", preamble),
            field("directory", directory),
            field("shared_model_binary", shared_binary),
        ]
        .concat()
    }

    fn plaintext_prefixes(
        domain: &[u8],
        header: &[u8; 32],
        shared: &[u8],
        tiers: &[&[u8]],
    ) -> Vec<[u8; 32]> {
        assert!((1..=4).contains(&tiers.len()));
        let labels = ["tier_lite", "tier_full", "tier_pro", "tier_ultra"];
        let mut state = Sha256::new();
        state.update(field("domain", domain));
        state.update(field("immutable_header_sha256", header));
        state.update(field("shared_model_data", shared));
        let mut prefixes = Vec::new();
        for (label, tier) in labels.iter().zip(tiers) {
            state.update(field(label, tier));
            prefixes.push(state.clone().finalize().into());
        }
        prefixes
    }

    fn master_prefixes(header: &[u8; 32], shared: &[u8], tiers: &[&[u8]]) -> Vec<[u8; 32]> {
        plaintext_prefixes(b"DNAM-MASTER-PREFIX-V1", header, shared, tiers)
    }

    fn prefix_list(prefixes: &[[u8; 32]]) -> Vec<u8> {
        assert!((1..=4).contains(&prefixes.len()));
        let mut output = vec![u8::try_from(prefixes.len()).unwrap()];
        for (tier, digest) in prefixes.iter().enumerate() {
            output.push(u8::try_from(tier).unwrap());
            output.extend_from_slice(digest);
        }
        output
    }

    fn creator_manifest(key: &VerifyingKey, header: &[u8; 32], prefixes: &[[u8; 32]]) -> Vec<u8> {
        [
            field("domain", b"DNAM-CREATOR-MANIFEST-V1"),
            field("creator_pubkey_ed25519", &key.to_bytes()),
            field("immutable_header_sha256", header),
            field(
                "issued_max_tier",
                &[u8::try_from(prefixes.len() - 1).unwrap()],
            ),
            field("master_prefix_sha256", &prefix_list(prefixes)),
        ]
        .concat()
    }

    fn buyer_receipt(
        key: &VerifyingKey,
        creator_identity: &[u8; 32],
        policy_digest: &[u8; 32],
        profile: u8,
        buyer_prefixes: &[[u8; 32]],
        inner_context: &[u8],
        ciphertext_prefixes: &[[u8; 32]],
    ) -> Vec<u8> {
        assert!(matches!(profile, 0..=2));
        if profile == 0 {
            assert!(inner_context.is_empty() && ciphertext_prefixes.is_empty());
        } else {
            assert_eq!(inner_context.len(), 32);
            assert_eq!(buyer_prefixes.len(), ciphertext_prefixes.len());
        }
        let ciphertext_list = if profile == 0 {
            Vec::new()
        } else {
            prefix_list(ciphertext_prefixes)
        };
        [
            field("domain", b"DNAM-BUYER-RECEIPT-V1"),
            field("marketplace_pubkey_ed25519", &key.to_bytes()),
            field("creator_signed_manifest_sha256", creator_identity),
            field("license_policy_sha256", policy_digest),
            field("protection_profile", &[profile]),
            field(
                "issued_max_tier",
                &[u8::try_from(buyer_prefixes.len() - 1).unwrap()],
            ),
            field("buyer_prefix_sha256", &prefix_list(buyer_prefixes)),
            field("inner_record_context_sha256", inner_context),
            field("inner_ciphertext_prefix_sha256", &ciphertext_list),
        ]
        .concat()
    }

    fn outer_public_metadata_digest(
        preamble: &[u8],
        directory: &[u8],
        auxiliary: &[u8],
    ) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(field("domain", b"DNAM-OUTER-PUBLIC-METADATA-V1"));
        hash.update(field("preamble", preamble));
        hash.update(field("directory", directory));
        hash.update(field("public_auxiliary", auxiliary));
        hash.finalize().into()
    }

    fn outer_manifest(
        key: &VerifyingKey,
        metadata_digest: &[u8; 32],
        record_context: &[u8; 32],
        prefixes: &[[u8; 32]],
    ) -> Vec<u8> {
        [
            field("domain", b"DNAM-OUTER-MANIFEST-V1"),
            field("outer_signer_pubkey_ed25519", &key.to_bytes()),
            field("outer_public_metadata_sha256", metadata_digest),
            field("outer_record_context_sha256", record_context),
            field(
                "issued_max_tier",
                &[u8::try_from(prefixes.len() - 1).unwrap()],
            ),
            field("outer_ciphertext_prefix_sha256", &prefix_list(prefixes)),
        ]
        .concat()
    }

    fn verify(key: &VerifyingKey, message: &[u8], signature: &Signature) -> bool {
        key.verify_strict(message, signature).is_ok()
    }

    fn open_preamble(authentication_kind: u8, maximum_tier: u8) -> [u8; 64] {
        let mut preamble = [0; 64];
        preamble[..4].copy_from_slice(b"DNAM");
        preamble[4..6].copy_from_slice(&2_u16.to_le_bytes());
        preamble[9] = authentication_kind;
        preamble[10] = maximum_tier;
        preamble[12..16].copy_from_slice(&48_000_u32.to_le_bytes());
        preamble
    }

    fn open_directory(fields: [u64; 13]) -> [u8; 128] {
        let mut directory = [0; 128];
        for (index, value) in fields.iter().enumerate() {
            directory[index * 8..index * 8 + 8].copy_from_slice(&value.to_le_bytes());
        }
        directory
    }

    #[test]
    fn buyer_receipt_preserves_creator_identity_across_a_relocated_lower_tier_issuance() {
        let creator = SigningKey::from_bytes(&[0x42; 32]);
        let marketplace = SigningKey::from_bytes(&[0x57; 32]);
        let header: [u8; 32] = Sha256::digest(b"{}").into();
        let creator_preamble = open_preamble(0, 3);
        let creator_directory =
            open_directory([192, 2, 256, 4, 320, 238, 0, 0, 576, 640, 704, 768, 832]);
        let master_shared = plaintext_shared(&creator_preamble, &creator_directory, b"ctrl");
        let master_tiers = [[b'A'; 64], [b'B'; 64], [b'C'; 64], [b'D'; 64]];
        let master_slices: Vec<&[u8]> = master_tiers.iter().map(<[u8; 64]>::as_slice).collect();
        let master_list = master_prefixes(&header, &master_shared, &master_slices);
        let creator_message = creator_manifest(&creator.verifying_key(), &header, &master_list);
        let creator_signature = creator.sign(&creator_message);
        let creator_identity: [u8; 32] = Sha256::digest(
            [
                field("domain", b"DNAM-CREATOR-SIGNED-V1"),
                field("creator_manifest", &creator_message),
                field("creator_signature", &creator_signature.to_bytes()),
            ]
            .concat(),
        )
        .into();

        let buyer_preamble = open_preamble(1, 1);
        let buyer_directory = open_directory([192, 2, 256, 4, 320, 412, 0, 0, 768, 832, 896, 0, 0]);
        let buyer_shared = plaintext_shared(&buyer_preamble, &buyer_directory, b"ctrl");
        let buyer_tiers = [[b'X'; 64], [b'Y'; 64]];
        let buyer_slices: Vec<&[u8]> = buyer_tiers.iter().map(<[u8; 64]>::as_slice).collect();
        let buyer_list = plaintext_prefixes(
            b"DNAM-BUYER-PREFIX-V1",
            &header,
            &buyer_shared,
            &buyer_slices,
        );
        let policy = b"policy";
        let policy_digest: [u8; 32] = Sha256::digest(
            [
                field("domain", b"DNAM-LICENSE-POLICY-V1"),
                field("policy", policy),
            ]
            .concat(),
        )
        .into();
        let receipt = buyer_receipt(
            &marketplace.verifying_key(),
            &creator_identity,
            &policy_digest,
            0,
            &buyer_list,
            &[],
            &[],
        );
        let receipt_signature = marketplace.sign(&receipt);
        let mut envelope = Vec::new();
        envelope.extend_from_slice(b"DAU1\x01\x00\x00\x00");
        envelope.extend_from_slice(&creator.verifying_key().to_bytes());
        envelope.push(3);
        envelope.extend_from_slice(&prefix_list(&master_list));
        envelope.extend_from_slice(&creator_signature.to_bytes());
        envelope.extend_from_slice(&marketplace.verifying_key().to_bytes());
        envelope.extend_from_slice(&u32::try_from(policy.len()).unwrap().to_le_bytes());
        envelope.extend_from_slice(policy);
        envelope.push(1);
        envelope.extend_from_slice(&prefix_list(&buyer_list));
        envelope.extend_from_slice(&receipt_signature.to_bytes());
        assert_eq!(envelope.len(), 412);
        assert_eq!(&envelope[..8], b"DAU1\x01\x00\x00\x00");
        assert_eq!(envelope[40], 3);
        assert_eq!(&envelope[41..174], prefix_list(&master_list));
        assert_eq!(&envelope[270..274], &6_u32.to_le_bytes());
        assert_eq!(&envelope[274..280], policy);
        assert_eq!(envelope[280], 1);
        assert_eq!(&envelope[281..348], prefix_list(&buyer_list));
        let decoded_creator =
            VerifyingKey::from_bytes(envelope[8..40].try_into().unwrap()).unwrap();
        let decoded_marketplace =
            VerifyingKey::from_bytes(envelope[238..270].try_into().unwrap()).unwrap();
        let decoded_creator_signature = Signature::from_slice(&envelope[174..238]).unwrap();
        let decoded_receipt_signature = Signature::from_slice(&envelope[348..412]).unwrap();
        assert!(verify(
            &decoded_creator,
            &creator_message,
            &decoded_creator_signature
        ));
        assert!(verify(
            &decoded_marketplace,
            &receipt,
            &decoded_receipt_signature
        ));
        assert_ne!(decoded_creator, decoded_marketplace);
        assert_eq!(
            plaintext_prefixes(
                b"DNAM-BUYER-PREFIX-V1",
                &header,
                &buyer_shared,
                &buyer_slices[..1],
            )[0],
            buyer_list[0]
        );

        let changed_directory =
            open_directory([192, 2, 256, 4, 320, 412, 768, 64, 832, 896, 960, 0, 0]);
        let changed_shared = plaintext_shared(&buyer_preamble, &changed_directory, b"ctrl");
        let changed_list = plaintext_prefixes(
            b"DNAM-BUYER-PREFIX-V1",
            &header,
            &changed_shared,
            &buyer_slices,
        );
        assert_ne!(changed_list, buyer_list);
        let changed_receipt = buyer_receipt(
            &marketplace.verifying_key(),
            &creator_identity,
            &policy_digest,
            0,
            &changed_list,
            &[],
            &[],
        );
        assert!(!verify(
            &marketplace.verifying_key(),
            &changed_receipt,
            &receipt_signature
        ));
        let changed_policy_digest: [u8; 32] = Sha256::digest(
            [
                field("domain", b"DNAM-LICENSE-POLICY-V1"),
                field("policy", b"policz"),
            ]
            .concat(),
        )
        .into();
        let changed_policy_receipt = buyer_receipt(
            &decoded_marketplace,
            &creator_identity,
            &changed_policy_digest,
            0,
            &buyer_list,
            &[],
            &[],
        );
        assert!(!verify(
            &decoded_marketplace,
            &changed_policy_receipt,
            &decoded_receipt_signature
        ));
    }

    #[test]
    fn plaintext_shared_structure_has_fixed_bytes_and_is_signed() {
        let shared = plaintext_shared(b"p", b"dd", b"xyz");
        let expected = [
            b"\x06\0\0\0domain\x18\0\0\0\0\0\0\0DNAM-PLAINTEXT-SHARED-V1".as_slice(),
            b"\x08\0\0\0preamble\x01\0\0\0\0\0\0\0p",
            b"\x09\0\0\0directory\x02\0\0\0\0\0\0\0dd",
            b"\x13\0\0\0shared_model_binary\x03\0\0\0\0\0\0\0xyz",
        ]
        .concat();
        assert_eq!(shared, expected);

        let signing_key = SigningKey::from_bytes(&[0x42; 32]);
        let trusted_key = signing_key.verifying_key();
        let header = [0x11; 32];
        let prefixes = master_prefixes(&header, &shared, &[b"lite"]);
        let manifest = creator_manifest(&trusted_key, &header, &prefixes);
        let signature = signing_key.sign(&manifest);
        assert!(verify(&trusted_key, &manifest, &signature));
        for changed in [
            plaintext_shared(b"q", b"dd", b"xyz"),
            plaintext_shared(b"p", b"de", b"xyz"),
            plaintext_shared(b"p", b"dd", b"xyZ"),
        ] {
            let changed_prefixes = master_prefixes(&header, &changed, &[b"lite"]);
            let changed_manifest = creator_manifest(&trusted_key, &header, &changed_prefixes);
            assert!(!verify(&trusted_key, &changed_manifest, &signature));
        }
    }

    #[test]
    fn original_signed_manifest_verifies_after_every_tier_suffix_is_omitted() {
        let signing_key = SigningKey::from_bytes(&[0x42; 32]);
        let trusted_key = signing_key.verifying_key();
        let header = [0x11; 32];
        let shared = plaintext_shared(b"preamble", b"directory", b"immutable descriptors");
        let tiers: [&[u8]; 4] = [b"lite", b"full", b"pro", b"ultra"];
        let original_prefixes = master_prefixes(&header, &shared, &tiers);
        let manifest = creator_manifest(&trusted_key, &header, &original_prefixes);
        let signature = signing_key.sign(&manifest);

        for retained_count in 1..=tiers.len() {
            let retained = master_prefixes(&header, &shared, &tiers[..retained_count]);
            assert_eq!(retained.last(), original_prefixes.get(retained_count - 1));
            assert!(verify(&trusted_key, &manifest, &signature));
        }
    }

    #[test]
    fn signatures_reject_replaced_retained_and_omitted_prefix_digests() {
        let signing_key = SigningKey::from_bytes(&[0x42; 32]);
        let trusted_key = signing_key.verifying_key();
        let header = [0x11; 32];
        let shared = plaintext_shared(b"preamble", b"directory", b"control descriptor");
        let tiers: [&[u8]; 4] = [b"lite", b"full", b"pro", b"ultra"];
        let original_prefixes = master_prefixes(&header, &shared, &tiers);
        let manifest = creator_manifest(&trusted_key, &header, &original_prefixes);
        let signature = signing_key.sign(&manifest);
        assert!(verify(&trusted_key, &manifest, &signature));

        let malicious_lite = master_prefixes(&header, &shared, &[b"changed lite"])[0];
        let mut substituted = original_prefixes.clone();
        substituted[0] = malicious_lite;
        let malicious_manifest = creator_manifest(&trusted_key, &header, &substituted);
        assert!(!verify(&trusted_key, &malicious_manifest, &signature));

        substituted = original_prefixes.clone();
        substituted[3][0] ^= 1;
        let changed_omitted_tier = creator_manifest(&trusted_key, &header, &substituted);
        assert!(!verify(&trusted_key, &changed_omitted_tier, &signature));

        let altered = plaintext_shared(b"preamble", b"directory", b"tampered descriptor");
        let altered_shared = master_prefixes(&header, &altered, &tiers);
        assert_ne!(altered_shared[0], original_prefixes[0]);
        let changed_shared_manifest = creator_manifest(&trusted_key, &header, &altered_shared);
        assert!(!verify(&trusted_key, &changed_shared_manifest, &signature));
    }

    #[test]
    fn a_valid_signature_from_an_embedded_key_is_not_a_trust_anchor() {
        let trusted_key = SigningKey::from_bytes(&[0x42; 32]).verifying_key();
        let attacker = SigningKey::from_bytes(&[0x55; 32]);
        let embedded_key = attacker.verifying_key();
        let header = [0x11; 32];
        let shared = plaintext_shared(b"preamble", b"directory", b"shared");
        let prefixes = master_prefixes(&header, &shared, &[b"lite"]);
        let manifest = creator_manifest(&embedded_key, &header, &prefixes);
        let signature = attacker.sign(&manifest);

        assert!(verify(&embedded_key, &manifest, &signature));
        assert_ne!(embedded_key, trusted_key);
        assert!(!verify(&trusted_key, &manifest, &signature));
    }

    #[test]
    fn buyer_signature_binds_creator_policy_context_and_ciphertext_prefixes() {
        let marketplace = SigningKey::from_bytes(&[0x33; 32]);
        let trusted_key = marketplace.verifying_key();
        let creator_identity = [0x11; 32];
        let policy = [0x22; 32];
        let context = [0x44; 32];
        let buyer_prefixes = [[0x55; 32], [0x66; 32]];
        let ciphertext_prefixes = [[0x77; 32], [0x88; 32]];
        let receipt = buyer_receipt(
            &trusted_key,
            &creator_identity,
            &policy,
            1,
            &buyer_prefixes,
            &context,
            &ciphertext_prefixes,
        );
        let signature = marketplace.sign(&receipt);
        assert!(verify(&trusted_key, &receipt, &signature));

        let alternatives = [
            buyer_receipt(
                &trusted_key,
                &[0x12; 32],
                &policy,
                1,
                &buyer_prefixes,
                &context,
                &ciphertext_prefixes,
            ),
            buyer_receipt(
                &trusted_key,
                &creator_identity,
                &[0x23; 32],
                1,
                &buyer_prefixes,
                &context,
                &ciphertext_prefixes,
            ),
            buyer_receipt(
                &trusted_key,
                &creator_identity,
                &policy,
                2,
                &buyer_prefixes,
                &context,
                &ciphertext_prefixes,
            ),
            buyer_receipt(
                &trusted_key,
                &creator_identity,
                &policy,
                1,
                &buyer_prefixes,
                &[0x45; 32],
                &ciphertext_prefixes,
            ),
            buyer_receipt(
                &trusted_key,
                &creator_identity,
                &policy,
                1,
                &buyer_prefixes,
                &context,
                &[[0x77; 32], [0x89; 32]],
            ),
            buyer_receipt(
                &trusted_key,
                &creator_identity,
                &policy,
                1,
                &[[0x54; 32], [0x66; 32]],
                &context,
                &ciphertext_prefixes,
            ),
        ];
        for changed in alternatives {
            assert!(!verify(&trusted_key, &changed, &signature));
        }
    }

    #[test]
    fn outer_signature_binds_public_structure_context_and_full_prefix_list() {
        let signer = SigningKey::from_bytes(&[0x25; 32]);
        let trusted_key = signer.verifying_key();
        let metadata = outer_public_metadata_digest(b"preamble", b"directory", b"plan and slots");
        let context = [0x44; 32];
        let prefixes = [[0x77; 32], [0x88; 32]];
        let manifest = outer_manifest(&trusted_key, &metadata, &context, &prefixes);
        let signature = signer.sign(&manifest);
        assert!(verify(&trusted_key, &manifest, &signature));

        let altered_metadata = [
            outer_public_metadata_digest(b"other", b"directory", b"plan and slots"),
            outer_public_metadata_digest(b"preamble", b"other", b"plan and slots"),
            outer_public_metadata_digest(b"preamble", b"directory", b"other"),
        ];
        for changed in altered_metadata {
            assert!(!verify(
                &trusted_key,
                &outer_manifest(&trusted_key, &changed, &context, &prefixes),
                &signature
            ));
        }
        let alternatives = [
            outer_manifest(&trusted_key, &metadata, &[0x45; 32], &prefixes),
            outer_manifest(&trusted_key, &metadata, &context, &prefixes[..1]),
            outer_manifest(&trusted_key, &metadata, &context, &[[0x77; 32], [0x89; 32]]),
        ];
        for changed in alternatives {
            assert!(!verify(&trusted_key, &changed, &signature));
        }
    }

    fn checkpoint_prefixes(
        header: &[u8; 32],
        preamble: &[u8],
        directory: &[u8],
        shared: &[u8],
        bundles: &[&[u8]],
    ) -> Vec<[u8; 32]> {
        let labels = ["bundle_lite", "bundle_full", "bundle_pro", "bundle_ultra"];
        let mut state = Sha256::new();
        state.update(field("domain", b"DNCK-CHECKPOINT-PREFIX-V1"));
        state.update(field("checkpoint_header_sha256", header));
        state.update(field("preamble", preamble));
        state.update(field("directory", directory));
        state.update(field("shared_checkpoint_binary", shared));
        let mut prefixes = Vec::new();
        for (label, bundle) in labels.iter().zip(bundles) {
            state.update(field(label, bundle));
            prefixes.push(state.clone().finalize().into());
        }
        prefixes
    }

    fn checkpoint_manifest(
        key: &VerifyingKey,
        header: &[u8; 32],
        prefixes: &[[u8; 32]],
    ) -> Vec<u8> {
        [
            field("domain", b"DNCK-CHECKPOINT-MANIFEST-V1"),
            field("checkpoint_writer_pubkey_ed25519", &key.to_bytes()),
            field("checkpoint_header_sha256", header),
            field(
                "issued_max_tier",
                &[u8::try_from(prefixes.len() - 1).unwrap()],
            ),
            field("checkpoint_prefix_sha256", &prefix_list(prefixes)),
        ]
        .concat()
    }

    #[test]
    fn checkpoint_writer_signature_binds_all_bundles_and_original_directory() {
        let writer = SigningKey::from_bytes(&[0x35; 32]);
        let key = writer.verifying_key();
        let header: [u8; 32] = Sha256::digest(b"{\"version\":1}").into();
        let mut preamble = open_preamble(0, 3);
        preamble[..4].copy_from_slice(b"DNCK");
        let directory = open_directory([
            192, 13, 256, 448, 704, 238, 0, 0, 960, 1152, 1344, 1536, 1728,
        ]);
        let runtime_directory =
            open_directory([192, 13, 256, 64, 320, 16, 0, 0, 384, 448, 512, 576, 640]);
        let mut runtime_template = [0; 384];
        runtime_template[..64].copy_from_slice(&open_preamble(0, 3));
        runtime_template[64..192].copy_from_slice(&runtime_directory);
        runtime_template[192..205].copy_from_slice(b"{\"version\":1}");
        runtime_template[256..320].fill(b'S');
        runtime_template[320..336].fill(b'E');
        let shared = [runtime_template.as_slice(), &[b'C'; 64]].concat();
        let bundles: Vec<Vec<u8>> = (0_u8..4)
            .map(|tier| {
                [
                    vec![b'A' + tier; 64],
                    vec![b'a' + tier; 64],
                    vec![b'1' + tier; 64],
                ]
                .concat()
            })
            .collect();
        let views: Vec<&[u8]> = bundles.iter().map(Vec::as_slice).collect();
        let prefixes = checkpoint_prefixes(&header, &preamble, &directory, &shared, &views);
        let prefix_hex: Vec<String> = prefixes
            .iter()
            .map(|digest| digest.iter().map(|byte| format!("{byte:02x}")).collect())
            .collect();
        assert_eq!(
            prefix_hex,
            [
                "71979cd5235f0d914d99f79adcd3a3184e382b8019ba86e39891fa4ad9531675",
                "3d9055890214c04d532c434a058c93f6d09a94e8f428aad9b1299232df3a7e68",
                "46d550042a1a5b45e2f359b7d732b7ce5bb83055eb330ecd4a35a21309bf4013",
                "d44970efb24b4f5a79ffb4e9b987c6527e79c61de5784032f9628b0e24b636c0",
            ]
        );
        let manifest = checkpoint_manifest(&key, &header, &prefixes);
        let signature = writer.sign(&manifest);
        assert!(verify(&key, &manifest, &signature));
        assert_eq!(
            checkpoint_prefixes(&header, &preamble, &directory, &shared, &views[..1])[0],
            prefixes[0]
        );

        let mut changed_directory = directory;
        changed_directory[80] ^= 1;
        let changed = checkpoint_prefixes(&header, &preamble, &changed_directory, &shared, &views);
        assert_ne!(changed[0], prefixes[0]);
        assert!(!verify(
            &key,
            &checkpoint_manifest(&key, &header, &changed),
            &signature
        ));

        for component in 0..3 {
            let mut changed_bundle = bundles[0].clone();
            changed_bundle[component * 64] ^= 1;
            let changed_views = [&changed_bundle[..], views[1], views[2], views[3]];
            let changed =
                checkpoint_prefixes(&header, &preamble, &directory, &shared, &changed_views);
            assert_ne!(changed[0], prefixes[0]);
            assert!(!verify(
                &key,
                &checkpoint_manifest(&key, &header, &changed),
                &signature
            ));
        }

        let mut changed_list = prefixes.clone();
        changed_list[3][0] ^= 1;
        assert!(!verify(
            &key,
            &checkpoint_manifest(&key, &header, &changed_list),
            &signature
        ));
        let untrusted = SigningKey::from_bytes(&[0x36; 32]).verifying_key();
        assert!(!verify(&untrusted, &manifest, &signature));
        let envelope = [
            b"DCA1".as_slice(),
            &[0; 4],
            &key.to_bytes(),
            &[3],
            &prefix_list(&prefixes),
            &signature.to_bytes(),
        ]
        .concat();
        assert_eq!(envelope.len(), 106 + 33 * prefixes.len());
    }
}

#[cfg(all(test, feature = "crypto"))]
mod encrypted_reference_tests {
    #[allow(unused_imports)]
    use alloc::{
        boxed::Box,
        format,
        string::{String, ToString},
        vec,
        vec::Vec,
    };

    // Record framing references, not tests of an implemented container protocol.

    use chacha20poly1305::aead::{AeadInPlace, KeyInit};
    use chacha20poly1305::{ChaCha20Poly1305, Nonce, Tag};
    use sha2::{Digest, Sha256};

    const MAX_PLAINTEXT_LEN: u64 = ((1u64 << 32) - 1) * 64;

    fn field(label: &str, value: &[u8]) -> Vec<u8> {
        let mut output = Vec::new();
        output.extend_from_slice(&u32::try_from(label.len()).unwrap().to_le_bytes());
        output.extend_from_slice(label.as_bytes());
        output.extend_from_slice(&u64::try_from(value.len()).unwrap().to_le_bytes());
        output.extend_from_slice(value);
        output
    }

    fn structural_metadata_digest(layer: u8, preamble: &[u8], directory: &[u8]) -> [u8; 32] {
        Sha256::digest(
            [
                field("domain", b"DNAM-ENCRYPTED-STRUCTURE-V1"),
                field("layer", &[layer]),
                field("preamble", preamble),
                field("directory", directory),
            ]
            .concat(),
        )
        .into()
    }

    fn mutable_wrap_digest(context: &[u8; 32], header: &[u8; 20], recipients: &[u8]) -> [u8; 32] {
        Sha256::digest(
            [
                field("domain", b"DNAM-MUTABLE-WRAP-V1"),
                field("record_context_sha256", context),
                field("header", header),
                field("recipient_key_ids", recipients),
            ]
            .concat(),
        )
        .into()
    }

    fn mutable_content_aad(digest: &[u8; 32], slots: &[u8]) -> Vec<u8> {
        [
            field("domain", b"DNAM-MUTABLE-CONTENT-V1"),
            field("mutable_wrap_sha256", digest),
            field("slots", slots),
        ]
        .concat()
    }

    fn record_nonce(index: u32) -> Result<[u8; 12], &'static str> {
        if index > 4 {
            return Err("unsupported record index");
        }
        let mut nonce = [0; 12];
        nonce[8..].copy_from_slice(&index.to_le_bytes());
        Ok(nonce)
    }

    fn record_aad(
        context: &[u8; 32],
        index: u32,
        ciphertext_len: u64,
    ) -> Result<Vec<u8>, &'static str> {
        if index > 4 {
            return Err("unsupported record index");
        }
        if !(16..=MAX_PLAINTEXT_LEN + 16).contains(&ciphertext_len) {
            return Err("unsupported ciphertext length");
        }
        let kind = u8::from(index != 0);
        let tier = if index == 0 {
            u8::MAX
        } else {
            u8::try_from(index - 1).unwrap()
        };
        Ok([
            field("domain", b"DNAM-RECORD-AAD-V1"),
            field("record_context_sha256", context),
            field("record_index", &index.to_le_bytes()),
            field("record_kind", &[kind]),
            field("tier", &[tier]),
            field("ciphertext_length", &ciphertext_len.to_le_bytes()),
        ]
        .concat())
    }

    fn seal(key: &[u8; 32], nonce: &[u8; 12], aad: &[u8], plaintext: &[u8]) -> Vec<u8> {
        let cipher = ChaCha20Poly1305::new(key.into());
        let mut ciphertext = plaintext.to_vec();
        let tag = cipher
            .encrypt_in_place_detached(Nonce::from_slice(nonce), aad, &mut ciphertext)
            .unwrap();
        ciphertext.extend_from_slice(&tag);
        ciphertext
    }

    fn open(
        key: &[u8; 32],
        nonce: &[u8; 12],
        aad: &[u8],
        ciphertext: &[u8],
    ) -> Result<Vec<u8>, chacha20poly1305::aead::Error> {
        let (ciphertext, tag) = ciphertext.split_at(ciphertext.len() - 16);
        let mut plaintext = ciphertext.to_vec();
        ChaCha20Poly1305::new(key.into()).decrypt_in_place_detached(
            Nonce::from_slice(nonce),
            aad,
            &mut plaintext,
            Tag::from_slice(tag),
        )?;
        Ok(plaintext)
    }

    #[test]
    fn encrypted_structure_framing_has_fixed_bytes_and_distinct_layer_binding() {
        let expected = [
            b"\x06\0\0\0domain\x1b\0\0\0\0\0\0\0DNAM-ENCRYPTED-STRUCTURE-V1".as_slice(),
            b"\x05\0\0\0layer\x01\0\0\0\0\0\0\0\0",
            b"\x08\0\0\0preamble\x01\0\0\0\0\0\0\0p",
            b"\x09\0\0\0directory\x02\0\0\0\0\0\0\0dd",
        ]
        .concat();
        let original = structural_metadata_digest(0, b"p", b"dd");
        let expected_digest: [u8; 32] = Sha256::digest(expected).into();
        assert_eq!(original, expected_digest);
        assert_ne!(original, structural_metadata_digest(1, b"p", b"dd"));
        assert_ne!(original, structural_metadata_digest(0, b"q", b"dd"));
        assert_ne!(original, structural_metadata_digest(0, b"p", b"de"));
        assert_ne!(original, structural_metadata_digest(0, b"pd", b"d"));
    }

    #[test]
    fn record_nonce_and_aad_have_fixed_little_endian_bytes() {
        let context = [0x11; 32];
        let expected = [
            b"\x06\0\0\0domain\x12\0\0\0\0\0\0\0DNAM-RECORD-AAD-V1".as_slice(),
            b"\x15\0\0\0record_context_sha256\x20\0\0\0\0\0\0\0",
            &context,
            b"\x0c\0\0\0record_index\x04\0\0\0\0\0\0\0\x01\0\0\0",
            b"\x0b\0\0\0record_kind\x01\0\0\0\0\0\0\0\x01",
            b"\x04\0\0\0tier\x01\0\0\0\0\0\0\0\0",
            b"\x11\0\0\0ciphertext_length\x08\0\0\0\0\0\0\0\x30\0\0\0\0\0\0\0",
        ]
        .concat();
        assert_eq!(record_aad(&context, 1, 48).unwrap(), expected);
        for index in 0..=4 {
            let nonce = record_nonce(index).unwrap();
            assert_eq!(&nonce[..8], &[0; 8]);
            assert_eq!(&nonce[8..], &index.to_le_bytes());
        }
        let shared_aad = record_aad(&context, 0, 16).unwrap();
        assert!(
            shared_aad
                .windows(24)
                .any(|bytes| bytes == b"\x0b\0\0\0record_kind\x01\0\0\0\0\0\0\0\0")
        );
        assert!(
            shared_aad
                .windows(17)
                .any(|bytes| bytes == b"\x04\0\0\0tier\x01\0\0\0\0\0\0\0\xff")
        );
        assert_eq!(record_nonce(5), Err("unsupported record index"));
        assert_eq!(record_aad(&context, 5, 16), Err("unsupported record index"));
        assert_eq!(
            record_aad(&context, 0, 15),
            Err("unsupported ciphertext length")
        );
        assert!(record_aad(&context, 0, MAX_PLAINTEXT_LEN + 16).is_ok());
        assert_eq!(
            record_aad(&context, 0, MAX_PLAINTEXT_LEN + 17),
            Err("unsupported ciphertext length")
        );
    }

    #[test]
    fn aead_binds_context_record_identity_length_and_ciphertext() {
        let key = [0x42; 32];
        let nonce = record_nonce(1).unwrap();
        let context = [0x11; 32];
        let plaintext = b"example tier record";
        let ciphertext_len = u64::try_from(plaintext.len()).unwrap() + 16;
        let aad = record_aad(&context, 1, ciphertext_len).unwrap();
        let ciphertext = seal(&key, &nonce, &aad, plaintext);

        assert_eq!(open(&key, &nonce, &aad, &ciphertext).unwrap(), plaintext);
        // AEAD alone cannot reject an unchanged valid replay.
        assert_eq!(open(&key, &nonce, &aad, &ciphertext).unwrap(), plaintext);
        assert!(
            open(
                &key,
                &nonce,
                &record_aad(&[0x22; 32], 1, ciphertext_len).unwrap(),
                &ciphertext
            )
            .is_err()
        );
        assert!(
            open(
                &key,
                &nonce,
                &record_aad(&context, 2, ciphertext_len).unwrap(),
                &ciphertext
            )
            .is_err()
        );
        assert!(
            open(
                &key,
                &nonce,
                &record_aad(&context, 1, ciphertext_len + 1).unwrap(),
                &ciphertext
            )
            .is_err()
        );
        assert!(open(&key, &record_nonce(2).unwrap(), &aad, &ciphertext).is_err());
        for (label, old, new) in [("record_kind", 1, 0), ("tier", 0, 1)] {
            let encoded = field(label, &[old]);
            let mut changed_aad = aad.clone();
            let position = aad
                .windows(encoded.len())
                .position(|window| window == encoded)
                .unwrap();
            changed_aad[position + encoded.len() - 1] = new;
            assert!(open(&key, &nonce, &changed_aad, &ciphertext).is_err());
        }
        let mut altered = ciphertext.clone();
        altered[0] ^= 1;
        assert!(open(&key, &nonce, &aad, &altered).is_err());
        let mut altered_tag = ciphertext.clone();
        *altered_tag.last_mut().unwrap() ^= 1;
        assert!(open(&key, &nonce, &aad, &altered_tag).is_err());
    }

    #[test]
    fn changing_associated_data_does_not_make_nonce_reuse_safe() {
        let key = [0x42; 32];
        let nonce = record_nonce(1).unwrap();
        let left = b"first secret";
        let right = b"other secret";
        assert_eq!(left.len(), right.len());
        let length = u64::try_from(left.len()).unwrap() + 16;
        let left_ct = seal(
            &key,
            &nonce,
            &record_aad(&[0x11; 32], 1, length).unwrap(),
            left,
        );
        let right_ct = seal(
            &key,
            &nonce,
            &record_aad(&[0x22; 32], 1, length).unwrap(),
            right,
        );

        for ((a, b), (x, y)) in left_ct.iter().zip(&right_ct).zip(left.iter().zip(right)) {
            assert_eq!(a ^ b, x ^ y);
        }
    }

    #[test]
    fn encrypted_mutable_content_binds_issuance_header_recipients_and_slots() {
        let context = [0x11; 32];
        let mut recipients = 1u32.to_le_bytes().to_vec();
        recipients.extend_from_slice(&[0x33; 32]);
        let mut slots = 1u32.to_le_bytes().to_vec();
        slots.extend_from_slice(&[0x44; 80]);
        let mut plaintext = 1u32.to_le_bytes().to_vec();
        plaintext.extend_from_slice(&[0x55; 32]);
        plaintext.extend_from_slice(b"DMU1\x8f\xb4\xcc\xa1\x02\x00\x00\x00\x00\x00\x00\x00{}");
        let mut header = [0u8; 20];
        header[..4].copy_from_slice(b"DME1");
        header[4] = 2;
        header[8..12].copy_from_slice(&1u32.to_le_bytes());
        header[12..20]
            .copy_from_slice(&(u64::try_from(plaintext.len()).unwrap() + 16).to_le_bytes());

        let digest = mutable_wrap_digest(&context, &header, &recipients);
        let aad = mutable_content_aad(&digest, &slots);
        let key = [0x42; 32];
        let nonce = [0; 12];
        let ciphertext = seal(&key, &nonce, &aad, &plaintext);
        assert_eq!(open(&key, &nonce, &aad, &ciphertext).unwrap(), plaintext);

        let other_context = [0x12; 32];
        let mut other_header = header;
        other_header[4] = 1;
        let mut other_recipients = recipients.clone();
        other_recipients[4] ^= 1;
        for changed in [
            mutable_wrap_digest(&other_context, &header, &recipients),
            mutable_wrap_digest(&context, &other_header, &recipients),
            mutable_wrap_digest(&context, &header, &other_recipients),
        ] {
            assert!(
                open(
                    &key,
                    &nonce,
                    &mutable_content_aad(&changed, &slots),
                    &ciphertext
                )
                .is_err()
            );
        }
        let mut other_slots = slots.clone();
        other_slots[4] ^= 1;
        assert!(
            open(
                &key,
                &nonce,
                &mutable_content_aad(&digest, &other_slots),
                &ciphertext
            )
            .is_err()
        );
        let mut altered = ciphertext.clone();
        altered[0] ^= 1;
        assert!(open(&key, &nonce, &aad, &altered).is_err());
        assert!(open(&[0x43; 32], &nonce, &aad, &ciphertext).is_err());
    }
}
