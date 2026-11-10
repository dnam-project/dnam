//! Target-agnostic v2 cryptographic provenance, `Ed25519`/`X25519` signing, `.dnam-seal` (`DNSL`)
//! HPKE sealing, `DAU1`/`DME1` marketplace framing, preview containment, and dual-mechanism
//! buyer watermarking (`ref[sdk.ts.provenance.v2-security]`).

use std::collections::{BTreeMap, BTreeSet};

use dnam_engine::container::{
    ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_PRO_END, ARENA_OFFSET_ULTRA_END,
    seal_dnam_container, unseal_dnam_container,
};
use dnam_engine::provenance::{
    AnticloneGateAction, Dau1Envelope, KAUTZ_TOTAL_CHANNELS, RustCryptoProvider,
    STRUCTURAL_HASH_DIM, STRUCTURAL_HASH_TIER_DIM, TARDOS_CASCADE_WEIGHTS, VerificationStatus,
    WatermarkSeedScope, apply_isometric_gauge_watermark, apply_tardos_watermark_7680,
    build_creator_signing_message, build_dme1_header, build_marketplace_receipt_signing_message,
    build_oem_cert_v2_signing_message, chacha20_xor_arena_blocks, compute_immutable_header_sha256,
    compute_preview_single_control_window, decode_hex_array, derive_watermark_seed_hmac_sha256,
    encode_hex, encode_hpke_slot_vector, encode_recipient_key_ids, encode_record_directory,
    enforce_distribution_policy, evaluate_anticlone_revocation_policy,
    extract_stage3_tier_fingerprint_component, extract_structural_hash_224d,
    generate_tardos_code_7680, hmac_sha256, parse_dme1_decrypted_plaintext, parse_hpke_slot_vector,
    parse_recipient_key_ids, parse_record_directory, score_tardos_watermark_7680, sha256_digest,
    structural_cosine_similarity, validate_license_tier_and_royalty_splits,
    validate_preview_single_switch_options, verify_container_provenance,
    verify_oem_device_certificate,
};
use dnam_engine::schema::{
    DnamJsonHeader, DnamProtectionProfile, MarketplaceReceipt, OemDeviceCertificate, RoyaltySplit,
};

use crate::error::{format_tagged_dnam_error, validation_err};

fn check_len<const N: usize>(bytes: &[u8], name: &str) -> Result<[u8; N], String> {
    bytes
        .try_into()
        .map_err(|_| validation_err(format!("{name} must be {N} bytes, got {}", bytes.len())))
}

fn decode_hex_prefixes(hex_list: &[String]) -> Result<Vec<[u8; 32]>, String> {
    let mut out = Vec::with_capacity(hex_list.len());
    for h in hex_list {
        out.push(decode_hex_array::<32>(h).map_err(|e| format_tagged_dnam_error(&e))?);
    }
    Ok(out)
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProvenanceVerifyInfo {
    pub status: String,
    pub signer_id: Option<String>,
    pub pubkey_ed25519: Option<String>,
    pub marketplace_id: Option<String>,
    pub license_id: Option<String>,
    pub license_tier: Option<String>,
}

pub fn verify_container_provenance_core(
    header_json: &str,
    plaintext_arena: &[u8],
    has_signatures_flag: bool,
    device_id_hash: Option<&str>,
    device_secret_32: Option<&[u8]>,
    trusted_marketplace_pubkey_hex: Option<&str>,
) -> Result<ProvenanceVerifyInfo, String> {
    let header: DnamJsonHeader =
        DnamJsonHeader::from_json_str(header_json).map_err(|e| validation_err(e.to_string()))?;
    let mut crypto = RustCryptoProvider::new();
    if let (Some(id), Some(sec)) = (device_id_hash, device_secret_32)
        && !id.is_empty()
        && !sec.is_empty()
    {
        let sec_arr = check_len::<32>(sec, "deviceSecret")?;
        crypto = crypto.with_device_key(id, sec_arr);
    }
    if let Some(mkt_pk_hex) = trusted_marketplace_pubkey_hex
        && !mkt_pk_hex.is_empty()
    {
        let mkt_pk =
            decode_hex_array::<32>(mkt_pk_hex).map_err(|e| format_tagged_dnam_error(&e))?;
        crypto = crypto.with_marketplace_pubkey(mkt_pk);
    }
    let status =
        verify_container_provenance(&header, has_signatures_flag, plaintext_arena, &crypto)
            .map_err(|e| format_tagged_dnam_error(&e))?;
    Ok(match status {
        VerificationStatus::Unsigned => ProvenanceVerifyInfo {
            status: "unsigned".to_string(),
            signer_id: None,
            pubkey_ed25519: None,
            marketplace_id: None,
            license_id: None,
            license_tier: None,
        },
        VerificationStatus::CreatorVerified {
            signer_id,
            pubkey_ed25519,
        } => ProvenanceVerifyInfo {
            status: "creator_verified".to_string(),
            signer_id: Some(signer_id),
            pubkey_ed25519: Some(encode_hex(&pubkey_ed25519)),
            marketplace_id: None,
            license_id: None,
            license_tier: None,
        },
        VerificationStatus::MarketplaceVerified {
            signer_id,
            marketplace_id,
            license_id,
            license_tier,
        } => ProvenanceVerifyInfo {
            status: "marketplace_verified".to_string(),
            signer_id: Some(signer_id),
            pubkey_ed25519: None,
            marketplace_id: Some(marketplace_id),
            license_id: Some(license_id),
            license_tier: Some(license_tier),
        },
    })
}

pub fn enforce_distribution_policy_core(
    header_json: &str,
    active_profile: &str,
) -> Result<(), String> {
    let header: DnamJsonHeader =
        DnamJsonHeader::from_json_str(header_json).map_err(|e| validation_err(e.to_string()))?;
    let profile = DnamProtectionProfile::from_str_name(active_profile)
        .map_err(|e| format_tagged_dnam_error(&e))?;
    enforce_distribution_policy(&header, profile).map_err(|e| format_tagged_dnam_error(&e))
}

pub fn evaluate_anticlone_policy_core(
    fingerprint_matched: bool,
    buyer_watermark_identified: bool,
    independent_device_source_evidence: bool,
) -> &'static str {
    match evaluate_anticlone_revocation_policy(
        fingerprint_matched,
        buyer_watermark_identified,
        independent_device_source_evidence,
    ) {
        AnticloneGateAction::NoMatchProceed => "no_match_proceed",
        AnticloneGateAction::FlagLicenseHoldForReview => "flag_license_hold_for_review",
        AnticloneGateAction::RevokeIdentifiedDeviceFutureLicensingOnly => {
            "revoke_identified_device_future_licensing_only"
        }
    }
}

pub fn immutable_header_sha256_core(header_json: &str) -> Result<String, String> {
    let header: DnamJsonHeader =
        DnamJsonHeader::from_json_str(header_json).map_err(|e| validation_err(e.to_string()))?;
    let digest =
        compute_immutable_header_sha256(&header).map_err(|e| format_tagged_dnam_error(&e))?;
    Ok(encode_hex(&digest))
}

#[inline]
pub fn sha256_hex_core(data: &[u8]) -> String {
    encode_hex(&sha256_digest(data))
}

pub fn hmac_sha256_hex_core(key_32: &[u8], message: &[u8]) -> Result<String, String> {
    let key = check_len::<32>(key_32, "key")?;
    Ok(encode_hex(&hmac_sha256(&key, message)))
}

pub fn ed25519_pubkey_hex_core(seed_32: &[u8]) -> Result<String, String> {
    let seed = check_len::<32>(seed_32, "seed")?;
    Ok(encode_hex(&RustCryptoProvider::ed25519_pubkey_from_seed(
        &seed,
    )))
}

pub fn ed25519_sign_hex_core(seed_32: &[u8], message: &[u8]) -> Result<String, String> {
    let seed = check_len::<32>(seed_32, "seed")?;
    Ok(encode_hex(&RustCryptoProvider::ed25519_sign(
        &seed, message,
    )))
}

pub fn x25519_pubkey_hex_core(secret_32: &[u8]) -> Result<String, String> {
    let secret = check_len::<32>(secret_32, "secret")?;
    Ok(encode_hex(&RustCryptoProvider::x25519_pubkey_from_secret(
        &secret,
    )))
}

pub fn build_creator_signing_message_core(
    immutable_header_sha256_hex: &str,
    master_arena_sha256_hex: &str,
) -> Result<Vec<u8>, String> {
    let hdr_sha = decode_hex_array::<32>(immutable_header_sha256_hex)
        .map_err(|e| format_tagged_dnam_error(&e))?;
    let arena_sha = decode_hex_array::<32>(master_arena_sha256_hex)
        .map_err(|e| format_tagged_dnam_error(&e))?;
    Ok(build_creator_signing_message(&hdr_sha, &arena_sha).to_vec())
}

pub fn build_marketplace_signing_message_core(
    immutable_header_sha256_hex: &str,
    master_arena_sha256_hex: &str,
    receipt_json: &str,
) -> Result<Vec<u8>, String> {
    let hdr_sha = decode_hex_array::<32>(immutable_header_sha256_hex)
        .map_err(|e| format_tagged_dnam_error(&e))?;
    let arena_sha = decode_hex_array::<32>(master_arena_sha256_hex)
        .map_err(|e| format_tagged_dnam_error(&e))?;
    let receipt: MarketplaceReceipt =
        serde_json::from_str(receipt_json).map_err(|e| validation_err(e.to_string()))?;
    build_marketplace_receipt_signing_message(&hdr_sha, &arena_sha, &receipt)
        .map_err(|e| format_tagged_dnam_error(&e))
}

/// Signs a `DnamJsonHeader` and `master_arena` using the creator's 32-byte `Ed25519` seed in Rust,
/// returning the updated canonical wire JSON header string (`ref[sec.sig.prefix-delegation]`).
pub fn sign_creator_header_core(
    header_json: &str,
    master_arena: &[u8],
    creator_seed_32: &[u8],
    signer_id: &str,
    include_tier_prefixes: bool,
) -> Result<String, String> {
    let seed = check_len::<32>(creator_seed_32, "creatorSeed")?;
    dnam_engine::provenance::sign_creator_header_json(
        header_json,
        master_arena,
        &seed,
        signer_id,
        include_tier_prefixes,
    )
    .map_err(|e| format_tagged_dnam_error(&e))
}

/// Signs a `MarketplaceReceipt` and attaches it to a creator-signed `DnamJsonHeader` in Rust,
/// returning the updated canonical wire JSON header string (`ref[sec.sig.prefix-delegation]`).
pub fn sign_marketplace_receipt_core(
    header_json: &str,
    buyer_arena: &[u8],
    marketplace_seed_32: &[u8],
    spec_json: &str,
) -> Result<String, String> {
    let seed = check_len::<32>(marketplace_seed_32, "marketplaceSeed")?;
    let spec: dnam_engine::provenance::MarketplaceReceiptSigningSpec =
        serde_json::from_str(spec_json).map_err(|e| validation_err(e.to_string()))?;
    dnam_engine::provenance::sign_marketplace_receipt_json(header_json, buyer_arena, &seed, spec)
        .map_err(|e| format_tagged_dnam_error(&e))
}

pub fn sign_oem_device_cert_core(
    oem_seed_32: &[u8],
    oem_vendor: &str,
    model_code: &str,
    device_id_hash: &str,
    pubkey_x25519_32: &[u8],
) -> Result<String, String> {
    let seed = check_len::<32>(oem_seed_32, "oemSeed")?;
    let pk = check_len::<32>(pubkey_x25519_32, "pubkeyX25519")?;
    let cert = RustCryptoProvider::sign_oem_device_certificate(
        &seed,
        oem_vendor,
        model_code,
        device_id_hash,
        &pk,
    )
    .map_err(|e| format_tagged_dnam_error(&e))?;
    serde_json::to_string(&cert).map_err(|e| validation_err(e.to_string()))
}

pub fn verify_oem_device_cert_core(
    cert_json: &str,
    trusted_oem_pubkey_hex: &str,
) -> Result<String, String> {
    let cert: OemDeviceCertificate =
        serde_json::from_str(cert_json).map_err(|e| validation_err(e.to_string()))?;
    let oem_pk =
        decode_hex_array::<32>(trusted_oem_pubkey_hex).map_err(|e| format_tagged_dnam_error(&e))?;
    let crypto = RustCryptoProvider::default();
    let pk_x25519 = verify_oem_device_certificate(&cert, &oem_pk, &crypto)
        .map_err(|e| format_tagged_dnam_error(&e))?;
    Ok(encode_hex(&pk_x25519))
}

#[allow(clippy::too_many_arguments)]
pub fn build_oem_cert_v2_signing_message_core(
    oem_vendor: &str,
    model_code: &str,
    device_id_hash_16: &[u8],
    pubkey_x25519_32: &[u8],
    issuer_pubkey_ed25519_32: &[u8],
    assurance: u8,
    attestation_policy_sha256_32: Option<&[u8]>,
    attestation_evidence_sha256_32: Option<&[u8]>,
) -> Result<Vec<u8>, String> {
    let dev_id = check_len::<16>(device_id_hash_16, "deviceIdHash")?;
    let pk = check_len::<32>(pubkey_x25519_32, "pubkeyX25519")?;
    let iss_pk = check_len::<32>(issuer_pubkey_ed25519_32, "issuerPubkeyEd25519")?;
    let pol = match attestation_policy_sha256_32 {
        Some(p) if !p.is_empty() => check_len::<32>(p, "attestationPolicySha256")?,
        _ => [0u8; 32],
    };
    let ev = match attestation_evidence_sha256_32 {
        Some(e) if !e.is_empty() => check_len::<32>(e, "attestationEvidenceSha256")?,
        _ => [0u8; 32],
    };
    build_oem_cert_v2_signing_message(
        oem_vendor, model_code, &dev_id, &pk, &iss_pk, assurance, &pol, &ev,
    )
    .map_err(|e| format_tagged_dnam_error(&e))
}

pub fn chacha20_xor_arena_core(
    arena_bytes: &[u8],
    key_32: &[u8],
    nonce_12: &[u8],
    start_block_counter: u32,
) -> Result<Vec<u8>, String> {
    let key = check_len::<32>(key_32, "key")?;
    let nonce = check_len::<12>(nonce_12, "nonce")?;
    let mut buf = arena_bytes.to_vec();
    chacha20_xor_arena_blocks(&mut buf, &key, &nonce, start_block_counter)
        .map_err(|e| format_tagged_dnam_error(&e))?;
    Ok(buf)
}

pub fn wrap_device_cek_hex_core(
    ephemeral_secret_32: &[u8],
    device_pubkey_32: &[u8],
    device_id_hash: &str,
    cek_32: &[u8],
) -> Result<String, String> {
    let eph = check_len::<32>(ephemeral_secret_32, "ephemeralSecret")?;
    let dev_pk = check_len::<32>(device_pubkey_32, "devicePubkey")?;
    let cek = check_len::<32>(cek_32, "cek")?;
    let wrapped = RustCryptoProvider::wrap_device_cek(&eph, &dev_pk, device_id_hash, &cek)
        .map_err(|e| format_tagged_dnam_error(&e))?;
    Ok(encode_hex(&wrapped))
}

pub fn seal_container_core(
    inner_dnam_bytes: &[u8],
    recipient_certs_json: &str,
    trusted_oem_pubkeys_json: &str,
    marketplace_seed_32: &[u8],
    ephemeral_secret_32: &[u8],
    cek_32: &[u8],
    nonce_12: &[u8],
) -> Result<Vec<u8>, String> {
    let certs: Vec<OemDeviceCertificate> =
        serde_json::from_str(recipient_certs_json).map_err(|e| validation_err(e.to_string()))?;
    let oem_map: BTreeMap<String, String> = serde_json::from_str(trusted_oem_pubkeys_json)
        .map_err(|e| validation_err(e.to_string()))?;
    let mut trusted_owned: Vec<(String, [u8; 32])> = Vec::with_capacity(oem_map.len());
    for (vendor, pk_hex) in oem_map {
        let pk = decode_hex_array::<32>(&pk_hex).map_err(|e| format_tagged_dnam_error(&e))?;
        trusted_owned.push((vendor, pk));
    }
    let trusted_refs: Vec<(&str, [u8; 32])> = trusted_owned
        .iter()
        .map(|(v, pk)| (v.as_str(), *pk))
        .collect();

    let mkt_seed = check_len::<32>(marketplace_seed_32, "marketplaceSeed")?;
    let eph_sec = check_len::<32>(ephemeral_secret_32, "ephemeralSecret")?;
    let cek = check_len::<32>(cek_32, "cek")?;
    let nonce = check_len::<12>(nonce_12, "nonce")?;

    seal_dnam_container(
        inner_dnam_bytes,
        &certs,
        &trusted_refs,
        &mkt_seed,
        &eph_sec,
        &cek,
        &nonce,
    )
    .map_err(|e| format_tagged_dnam_error(&e))
}

pub fn unseal_container_core(
    sealed_bytes: &[u8],
    device_id_hash: Option<&str>,
    device_secret_32: Option<&[u8]>,
    trusted_marketplace_pubkey_hex: Option<&str>,
) -> Result<Vec<u8>, String> {
    let mut crypto = RustCryptoProvider::new();
    if let (Some(id), Some(sec)) = (device_id_hash, device_secret_32)
        && !id.is_empty()
        && !sec.is_empty()
    {
        let sec_arr = check_len::<32>(sec, "deviceSecret")?;
        crypto = crypto.with_device_key(id, sec_arr);
    }
    if let Some(mkt_pk_hex) = trusted_marketplace_pubkey_hex
        && !mkt_pk_hex.is_empty()
    {
        let mkt_pk =
            decode_hex_array::<32>(mkt_pk_hex).map_err(|e| format_tagged_dnam_error(&e))?;
        crypto = crypto.with_marketplace_pubkey(mkt_pk);
    }
    unseal_dnam_container(sealed_bytes, &crypto).map_err(|e| format_tagged_dnam_error(&e))
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Dau1EnvelopeJson {
    pub auth_kind: u8,
    pub creator_pubkey_hex: String,
    pub creator_max_tier: u8,
    pub master_prefixes_hex: Vec<String>,
    pub creator_signature_hex: String,
    pub marketplace_pubkey_hex: Option<String>,
    pub policy_bytes_hex: String,
    pub buyer_max_tier: Option<u8>,
    pub buyer_prefixes_hex: Vec<String>,
    pub marketplace_signature_hex: Option<String>,
}

pub fn encode_dau1_envelope_core(spec_json: &str) -> Result<Vec<u8>, String> {
    let spec: Dau1EnvelopeJson =
        serde_json::from_str(spec_json).map_err(|e| validation_err(e.to_string()))?;
    let creator_pubkey = decode_hex_array::<32>(&spec.creator_pubkey_hex)
        .map_err(|e| format_tagged_dnam_error(&e))?;
    let master_prefixes = decode_hex_prefixes(&spec.master_prefixes_hex)?;
    let creator_signature = decode_hex_array::<64>(&spec.creator_signature_hex)
        .map_err(|e| format_tagged_dnam_error(&e))?;
    let marketplace_pubkey = match spec.marketplace_pubkey_hex.as_deref() {
        Some(h) if !h.is_empty() => {
            Some(decode_hex_array::<32>(h).map_err(|e| format_tagged_dnam_error(&e))?)
        }
        _ => None,
    };
    let policy_bytes = decode_hex_bytes(&spec.policy_bytes_hex)?;
    let buyer_prefixes = decode_hex_prefixes(&spec.buyer_prefixes_hex)?;
    let marketplace_signature = match spec.marketplace_signature_hex.as_deref() {
        Some(h) if !h.is_empty() => {
            Some(decode_hex_array::<64>(h).map_err(|e| format_tagged_dnam_error(&e))?)
        }
        _ => None,
    };
    let env = Dau1Envelope {
        auth_kind: spec.auth_kind,
        creator_pubkey,
        creator_max_tier: spec.creator_max_tier,
        master_prefixes,
        creator_signature,
        marketplace_pubkey,
        policy_bytes,
        buyer_max_tier: spec.buyer_max_tier,
        buyer_prefixes,
        marketplace_signature,
    };
    env.to_bytes().map_err(|e| format_tagged_dnam_error(&e))
}

pub fn decode_dau1_envelope_core(wire_bytes: &[u8]) -> Result<String, String> {
    let env = Dau1Envelope::from_bytes(wire_bytes).map_err(|e| format_tagged_dnam_error(&e))?;
    let out = Dau1EnvelopeJson {
        auth_kind: env.auth_kind,
        creator_pubkey_hex: encode_hex(&env.creator_pubkey),
        creator_max_tier: env.creator_max_tier,
        master_prefixes_hex: env.master_prefixes.iter().map(|d| encode_hex(d)).collect(),
        creator_signature_hex: encode_hex(&env.creator_signature),
        marketplace_pubkey_hex: env.marketplace_pubkey.as_ref().map(|d| encode_hex(d)),
        policy_bytes_hex: encode_hex(&env.policy_bytes),
        buyer_max_tier: env.buyer_max_tier,
        buyer_prefixes_hex: env.buyer_prefixes.iter().map(|d| encode_hex(d)).collect(),
        marketplace_signature_hex: env.marketplace_signature.as_ref().map(|d| encode_hex(d)),
    };
    serde_json::to_string(&out).map_err(|e| validation_err(e.to_string()))
}

fn decode_hex_bytes(hex_str: &str) -> Result<Vec<u8>, String> {
    if hex_str.is_empty() {
        return Ok(Vec::new());
    }
    if !hex_str.len().is_multiple_of(2) {
        return Err(validation_err("hex string must have even length"));
    }
    let mut out = Vec::with_capacity(hex_str.len() / 2);
    for chunk in hex_str.as_bytes().as_chunks::<2>().0 {
        let s = core::str::from_utf8(chunk).map_err(|e| validation_err(e.to_string()))?;
        let b = u8::from_str_radix(s, 16).map_err(|e| validation_err(e.to_string()))?;
        out.push(b);
    }
    Ok(out)
}

pub fn build_dme1_header_core(
    profile: u8,
    recipient_count: u32,
    ciphertext_length: u64,
) -> Result<Vec<u8>, String> {
    build_dme1_header(profile, recipient_count, ciphertext_length)
        .map(|h| h.to_vec())
        .map_err(|e| format_tagged_dnam_error(&e))
}

pub fn parse_dme1_plaintext_core(
    plaintext: &[u8],
    recipient_key_digests_hex: &[String],
) -> Result<String, String> {
    let digests = decode_hex_prefixes(recipient_key_digests_hex)?;
    let val = parse_dme1_decrypted_plaintext(plaintext, &digests)
        .map_err(|e| format_tagged_dnam_error(&e))?;
    serde_json::to_string(&val).map_err(|e| validation_err(e.to_string()))
}

pub fn encode_recipient_key_ids_core(key_digests_hex: &[String]) -> Result<Vec<u8>, String> {
    let digests = decode_hex_prefixes(key_digests_hex)?;
    encode_recipient_key_ids(&digests).map_err(|e| format_tagged_dnam_error(&e))
}

pub fn parse_recipient_key_ids_core(
    data: &[u8],
    max_recipients: usize,
) -> Result<Vec<String>, String> {
    let digests =
        parse_recipient_key_ids(data, max_recipients).map_err(|e| format_tagged_dnam_error(&e))?;
    Ok(digests.iter().map(|d| encode_hex(d)).collect())
}

pub fn encode_hpke_slots_core(slots_hex: &[(String, String)]) -> Result<Vec<u8>, String> {
    let mut slots = Vec::with_capacity(slots_hex.len());
    for (enc_hex, ct_hex) in slots_hex {
        let enc = decode_hex_array::<32>(enc_hex).map_err(|e| format_tagged_dnam_error(&e))?;
        let ct = decode_hex_array::<48>(ct_hex).map_err(|e| format_tagged_dnam_error(&e))?;
        slots.push((enc, ct));
    }
    encode_hpke_slot_vector(&slots).map_err(|e| format_tagged_dnam_error(&e))
}

pub fn parse_hpke_slots_core(
    data: &[u8],
    expected_count: usize,
    max_recipients: usize,
) -> Result<Vec<(String, String)>, String> {
    let slots = parse_hpke_slot_vector(data, expected_count, max_recipients)
        .map_err(|e| format_tagged_dnam_error(&e))?;
    Ok(slots
        .iter()
        .map(|(e, c)| (encode_hex(e), encode_hex(c)))
        .collect())
}

pub fn encode_record_directory_core(
    maximum_tier: u8,
    records: &[(u64, u64)],
) -> Result<Vec<u8>, String> {
    encode_record_directory(maximum_tier, records).map_err(|e| format_tagged_dnam_error(&e))
}

pub fn parse_record_directory_core(
    data: &[u8],
    maximum_tier: u8,
) -> Result<Vec<(u64, u64)>, String> {
    parse_record_directory(data, maximum_tier).map_err(|e| format_tagged_dnam_error(&e))
}

pub fn compute_preview_window_core(
    anchor: f64,
    delta: f64,
    safe_bounds: (f64, f64),
) -> Result<[f64; 2], String> {
    let (s, e) = compute_preview_single_control_window(anchor, delta, safe_bounds)
        .map_err(|e| format_tagged_dnam_error(&e))?;
    Ok([s, e])
}

pub fn validate_preview_switch_options_core(
    preset: &str,
    alternative: &str,
    observed: &[String],
    active: &[String],
    safe: &[String],
) -> Result<(String, String), String> {
    let obs_set: BTreeSet<&str> = observed.iter().map(String::as_str).collect();
    let act_set: BTreeSet<&str> = active.iter().map(String::as_str).collect();
    let safe_set: BTreeSet<&str> = safe.iter().map(String::as_str).collect();
    let (p, a) =
        validate_preview_single_switch_options(preset, alternative, &obs_set, &act_set, &safe_set)
            .map_err(|e| format_tagged_dnam_error(&e))?;
    Ok((p.to_string(), a.to_string()))
}

pub fn validate_royalty_splits_core(
    license_tier: &str,
    royalty_splits_json: &str,
) -> Result<(), String> {
    let splits: Vec<RoyaltySplit> =
        serde_json::from_str(royalty_splits_json).map_err(|e| validation_err(e.to_string()))?;
    validate_license_tier_and_royalty_splits(license_tier, &splits)
        .map_err(|e| format_tagged_dnam_error(&e))
}

pub fn apply_isometric_gauge_watermark_core(
    arena_bytes: &[u8],
    kappa: &[f32],
    q_buyer_flat_16: &[f32],
) -> Result<Vec<u8>, String> {
    if kappa.len() != KAUTZ_TOTAL_CHANNELS {
        return Err(validation_err(format!(
            "kappa must have {KAUTZ_TOTAL_CHANNELS} elements, got {}",
            kappa.len()
        )));
    }
    if q_buyer_flat_16.len() != 16 {
        return Err(validation_err(format!(
            "qBuyer4x4 must have 16 elements, got {}",
            q_buyer_flat_16.len()
        )));
    }
    let mut kappa_arr = [1.0_f32; KAUTZ_TOTAL_CHANNELS];
    kappa_arr.copy_from_slice(kappa);
    let mut q_arr = [[0.0_f32; 4]; 4];
    for r in 0..4 {
        q_arr[r].copy_from_slice(&q_buyer_flat_16[r * 4..(r + 1) * 4]);
    }
    let mut buf = arena_bytes.to_vec();
    apply_isometric_gauge_watermark(&mut buf, &kappa_arr, &q_arr)
        .map_err(|e| format_tagged_dnam_error(&e))?;
    Ok(buf)
}

pub fn generate_tardos_code_core(
    marketplace_secret_32: &[u8],
    model_id: &str,
    license_id: &str,
) -> Result<Vec<f32>, String> {
    let secret = check_len::<32>(marketplace_secret_32, "marketplaceSecret")?;
    Ok(generate_tardos_code_7680(&secret, model_id, license_id).to_vec())
}

pub fn apply_tardos_watermark_core(
    arena_bytes: &[u8],
    code: &[f32],
    epsilon: f32,
) -> Result<Vec<u8>, String> {
    if code.len() != TARDOS_CASCADE_WEIGHTS {
        return Err(validation_err(format!(
            "Tardos code must have {TARDOS_CASCADE_WEIGHTS} elements, got {}",
            code.len()
        )));
    }
    let mut code_arr = [1.0_f32; TARDOS_CASCADE_WEIGHTS];
    code_arr.copy_from_slice(code);
    let mut buf = arena_bytes.to_vec();
    apply_tardos_watermark_7680(&mut buf, &code_arr, epsilon)
        .map_err(|e| format_tagged_dnam_error(&e))?;
    Ok(buf)
}

pub fn score_tardos_watermark_core(
    master_arena: &[u8],
    suspect_arena: &[u8],
    code: &[f32],
) -> Result<f64, String> {
    if code.len() != TARDOS_CASCADE_WEIGHTS {
        return Err(validation_err(format!(
            "Tardos code must have {TARDOS_CASCADE_WEIGHTS} elements, got {}",
            code.len()
        )));
    }
    let mut code_arr = [1.0_f32; TARDOS_CASCADE_WEIGHTS];
    code_arr.copy_from_slice(code);
    score_tardos_watermark_7680(master_arena, suspect_arena, &code_arr)
        .map_err(|e| format_tagged_dnam_error(&e))
}

pub fn extract_structural_hash_core(arena_bytes: &[u8]) -> Result<Vec<f64>, String> {
    if arena_bytes.len() < ARENA_OFFSET_LITE_END as usize {
        return Err(validation_err(format!(
            "arenaBytes length ({}) < ARENA_OFFSET_LITE_END ({ARENA_OFFSET_LITE_END})",
            arena_bytes.len()
        )));
    }
    let vec224 =
        extract_structural_hash_224d(arena_bytes).map_err(|e| format_tagged_dnam_error(&e))?;
    let tiers = if arena_bytes.len() >= ARENA_OFFSET_ULTRA_END as usize {
        4
    } else if arena_bytes.len() >= ARENA_OFFSET_PRO_END as usize {
        3
    } else if arena_bytes.len() >= ARENA_OFFSET_FULL_END as usize {
        2
    } else {
        1
    };
    Ok(vec224[..tiers * STRUCTURAL_HASH_TIER_DIM].to_vec())
}

pub fn structural_cosine_similarity_core(a: &[f64], b: &[f64]) -> Result<f64, String> {
    let min_len = a.len().min(b.len());
    if min_len == 0
        || min_len > STRUCTURAL_HASH_DIM
        || !min_len.is_multiple_of(STRUCTURAL_HASH_TIER_DIM)
    {
        return Err(validation_err(format!(
            "Structural hash vectors must have prefix length 56, 112, 168, or {STRUCTURAL_HASH_DIM}"
        )));
    }
    let mut a_arr = [0.0_f64; STRUCTURAL_HASH_DIM];
    let mut b_arr = [0.0_f64; STRUCTURAL_HASH_DIM];
    a_arr[..min_len].copy_from_slice(&a[..min_len]);
    b_arr[..min_len].copy_from_slice(&b[..min_len]);
    Ok(structural_cosine_similarity(&a_arr, &b_arr))
}

#[allow(clippy::too_many_arguments)]
pub fn derive_watermark_seed_hex_core(
    key_32: &[u8],
    marketplace_pubkey_32: &[u8],
    watermark_key_id_16: &[u8],
    creator_signed_manifest_sha256_32: &[u8],
    model_id_16: &[u8],
    watermark_profile_sha256_32: &[u8],
    owning_tier: u8,
    license_id_16: Option<&[u8]>,
    buyer_id_16: Option<&[u8]>,
) -> Result<String, String> {
    let key = check_len::<32>(key_32, "key")?;
    let scope = WatermarkSeedScope {
        marketplace_pubkey: check_len::<32>(marketplace_pubkey_32, "marketplacePubkey")?,
        watermark_key_id: check_len::<16>(watermark_key_id_16, "watermarkKeyId")?,
        creator_signed_manifest_sha256: check_len::<32>(
            creator_signed_manifest_sha256_32,
            "creatorSignedManifestSha256",
        )?,
        model_id: check_len::<16>(model_id_16, "modelId")?,
        watermark_profile_sha256: check_len::<32>(
            watermark_profile_sha256_32,
            "watermarkProfileSha256",
        )?,
        owning_tier,
    };
    let buyer_pair = match (license_id_16, buyer_id_16) {
        (Some(lic), Some(buy)) if !lic.is_empty() && !buy.is_empty() => Some((
            check_len::<16>(lic, "licenseId")?,
            check_len::<16>(buy, "buyerId")?,
        )),
        (None, None) | (Some(&[]), Some(&[])) => None,
        _ => {
            return Err(validation_err(
                "licenseId and buyerId must both be provided or both omitted",
            ));
        }
    };
    let seed =
        derive_watermark_seed_hmac_sha256(&key, &scope, buyer_pair.as_ref().map(|(l, b)| (l, b)))
            .map_err(|e| format_tagged_dnam_error(&e))?;
    Ok(encode_hex(&seed))
}

pub fn extract_stage3_fingerprint_core(
    early_fir_128b: &[u8],
    kautz_modal_bytes: &[u8],
    added_modes: usize,
) -> Result<Vec<f64>, String> {
    extract_stage3_tier_fingerprint_component(early_fir_128b, kautz_modal_bytes, added_modes)
        .map_err(|e| format_tagged_dnam_error(&e))
}
