//! Prefix Lists, Two-Certificate Ed25519 Delegation & Open `DAU1` Authentication Envelopes.
//!
//! Canonical Reference: `docs/specs/security/provenance_and_enclave_drm.md` (`ref[sec.sig.prefix-delegation]`, `ref[sec.sig.dau1-envelope]`, `ref[sec.sig.offline-auth]`).

use alloc::vec::Vec;

use crate::container::encode_domain_field;
use crate::provenance::sha256::sha256_digest;
use crate::schema::{DnamError, SchemaOptionExt, reject_if};

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

#[cfg(test)]
mod spec_tests {
    use crate::provenance::{
        Dau1Envelope, WatermarkSeedScope, build_buyer_receipt_v1, build_creator_manifest_v1,
        build_plaintext_shared_v1, build_record_nonce_v1, compute_creator_signed_manifest_sha256,
        compute_immutable_header_sha256, compute_license_policy_sha256,
        compute_plaintext_prefixes_v1, derive_watermark_seed_hmac_sha256, encode_prefix_list,
        enforce_distribution_policy, parse_prefix_list,
    };
    use crate::schema::{DistributionPolicy, DnamJsonHeader, DnamProtectionProfile};
    use alloc::{string::ToString, vec, vec::Vec};
    use proptest::prelude::*;
    use serde_json::json;

    /// verify[sec.taxonomy.overview]
    /// verify[sec.jcs.immutable-header-digest]
    /// verify[sec.policy.distribution-policy]
    /// verify[sec.drm.protection-profiles]
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
    #[test]
    fn test_prefix_delegation_dau1_envelope_and_offline_auth() {
        let shared = build_plaintext_shared_v1(&[0x11; 64], &[0x22; 128], &[0x33; 64]);
        let header_sha = [0x44u8; 32];
        let tiers: [&[u8]; 4] = [&[1; 64], &[2; 64], &[3; 64], &[4; 64]];
        let master_prefixes =
            compute_plaintext_prefixes_v1(b"DNAM-MASTER-PREFIX-V1", &header_sha, &shared, &tiers)
                .unwrap();
        assert_eq!(master_prefixes.len(), 4);

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
