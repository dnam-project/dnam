//! Node-API v10 v2 provenance, signing, sealing, `DAU1`/`DME1`, preview containment,
//! and buyer watermarking bindings (`ref[sdk.ts.provenance.v2-security]`).

use napi::bindgen_prelude::{Float32Array, Float64Array, Result, Uint8Array};
use napi_derive::napi;

use crate::core::provenance::{
    apply_isometric_gauge_watermark_core, apply_tardos_watermark_core,
    build_creator_signing_message_core, build_dme1_header_core,
    build_marketplace_signing_message_core, build_oem_cert_v2_signing_message_core,
    chacha20_xor_arena_core, compute_preview_window_core, decode_dau1_envelope_core,
    derive_watermark_seed_hex_core, ed25519_pubkey_hex_core, ed25519_sign_hex_core,
    encode_dau1_envelope_core, encode_hpke_slots_core, encode_recipient_key_ids_core,
    encode_record_directory_core, enforce_distribution_policy_core, evaluate_anticlone_policy_core,
    extract_stage3_fingerprint_core, extract_structural_hash_core, generate_tardos_code_core,
    hmac_sha256_hex_core, immutable_header_sha256_core, parse_dme1_plaintext_core,
    parse_hpke_slots_core, parse_recipient_key_ids_core, parse_record_directory_core,
    score_tardos_watermark_core, seal_container_core, sha256_hex_core, sign_creator_header_core,
    sign_marketplace_receipt_core, sign_oem_device_cert_core, structural_cosine_similarity_core,
    unseal_container_core, validate_preview_switch_options_core, validate_royalty_splits_core,
    verify_container_provenance_core, verify_oem_device_cert_core, wrap_device_cek_hex_core,
    x25519_pubkey_hex_core,
};
use crate::error::validation_err;
use crate::napi::napi_err;

#[napi]
pub fn verify_container_provenance_native(
    header_json: String,
    plaintext_arena: Uint8Array,
    has_signatures_flag: Option<bool>,
    device_id_hash: Option<String>,
    device_secret_32: Option<Uint8Array>,
    trusted_marketplace_pubkey_hex: Option<String>,
) -> Result<String> {
    let info = verify_container_provenance_core(
        &header_json,
        &plaintext_arena,
        has_signatures_flag.unwrap_or(false),
        device_id_hash.as_deref(),
        device_secret_32.as_deref(),
        trusted_marketplace_pubkey_hex.as_deref(),
    )
    .map_err(napi_err)?;
    serde_json::to_string(&info).map_err(|e| napi_err(validation_err(e.to_string())))
}

#[napi]
pub fn enforce_distribution_policy_native(
    header_json: String,
    active_profile: String,
) -> Result<()> {
    enforce_distribution_policy_core(&header_json, &active_profile).map_err(napi_err)
}

#[napi]
pub fn evaluate_anticlone_policy_native(
    fingerprint_matched: bool,
    buyer_watermark_identified: bool,
    independent_device_source_evidence: bool,
) -> String {
    evaluate_anticlone_policy_core(
        fingerprint_matched,
        buyer_watermark_identified,
        independent_device_source_evidence,
    )
    .to_string()
}

#[napi]
pub fn immutable_header_sha256_native(header_json: String) -> Result<String> {
    immutable_header_sha256_core(&header_json).map_err(napi_err)
}

#[napi]
pub fn sha256_hex_native(data: Uint8Array) -> String {
    sha256_hex_core(&data)
}

#[napi]
pub fn hmac_sha256_hex_native(key_32: Uint8Array, message: Uint8Array) -> Result<String> {
    hmac_sha256_hex_core(&key_32, &message).map_err(napi_err)
}

#[napi]
pub fn ed25519_pubkey_hex_native(seed_32: Uint8Array) -> Result<String> {
    ed25519_pubkey_hex_core(&seed_32).map_err(napi_err)
}

#[napi]
pub fn ed25519_sign_hex_native(seed_32: Uint8Array, message: Uint8Array) -> Result<String> {
    ed25519_sign_hex_core(&seed_32, &message).map_err(napi_err)
}

#[napi]
pub fn x25519_pubkey_hex_native(secret_32: Uint8Array) -> Result<String> {
    x25519_pubkey_hex_core(&secret_32).map_err(napi_err)
}

#[napi]
pub fn build_creator_signing_message_native(
    immutable_header_sha256_hex: String,
    master_arena_sha256_hex: String,
) -> Result<Uint8Array> {
    build_creator_signing_message_core(&immutable_header_sha256_hex, &master_arena_sha256_hex)
        .map(Uint8Array::new)
        .map_err(napi_err)
}

#[napi]
pub fn build_marketplace_signing_message_native(
    immutable_header_sha256_hex: String,
    master_arena_sha256_hex: String,
    receipt_json: String,
) -> Result<Uint8Array> {
    build_marketplace_signing_message_core(
        &immutable_header_sha256_hex,
        &master_arena_sha256_hex,
        &receipt_json,
    )
    .map(Uint8Array::new)
    .map_err(napi_err)
}

#[napi]
pub fn sign_creator_header_native(
    header_json: String,
    master_arena: Uint8Array,
    creator_seed_32: Uint8Array,
    signer_id: String,
    include_tier_prefixes: Option<bool>,
) -> Result<String> {
    sign_creator_header_core(
        &header_json,
        &master_arena,
        &creator_seed_32,
        &signer_id,
        include_tier_prefixes.unwrap_or(false),
    )
    .map_err(napi_err)
}

#[napi]
pub fn sign_marketplace_receipt_native(
    header_json: String,
    buyer_arena: Uint8Array,
    marketplace_seed_32: Uint8Array,
    spec_json: String,
) -> Result<String> {
    sign_marketplace_receipt_core(&header_json, &buyer_arena, &marketplace_seed_32, &spec_json)
        .map_err(napi_err)
}

#[napi]
pub fn sign_oem_device_cert_native(
    oem_seed_32: Uint8Array,
    oem_vendor: String,
    model_code: String,
    device_id_hash: String,
    pubkey_x25519_32: Uint8Array,
) -> Result<String> {
    sign_oem_device_cert_core(
        &oem_seed_32,
        &oem_vendor,
        &model_code,
        &device_id_hash,
        &pubkey_x25519_32,
    )
    .map_err(napi_err)
}

#[napi]
pub fn verify_oem_device_cert_native(
    cert_json: String,
    trusted_oem_pubkey_hex: String,
) -> Result<String> {
    verify_oem_device_cert_core(&cert_json, &trusted_oem_pubkey_hex).map_err(napi_err)
}

#[napi]
#[allow(clippy::too_many_arguments)]
pub fn build_oem_cert_v2_signing_message_native(
    oem_vendor: String,
    model_code: String,
    device_id_hash_16: Uint8Array,
    pubkey_x25519_32: Uint8Array,
    issuer_pubkey_ed25519_32: Uint8Array,
    assurance: Option<u8>,
    attestation_policy_sha256_32: Option<Uint8Array>,
    attestation_evidence_sha256_32: Option<Uint8Array>,
) -> Result<Uint8Array> {
    build_oem_cert_v2_signing_message_core(
        &oem_vendor,
        &model_code,
        &device_id_hash_16,
        &pubkey_x25519_32,
        &issuer_pubkey_ed25519_32,
        assurance.unwrap_or(1),
        attestation_policy_sha256_32.as_deref(),
        attestation_evidence_sha256_32.as_deref(),
    )
    .map(Uint8Array::new)
    .map_err(napi_err)
}

#[napi]
pub fn chacha20_xor_arena_native(
    arena_bytes: Uint8Array,
    key_32: Uint8Array,
    nonce_12: Uint8Array,
    start_block_counter: Option<u32>,
) -> Result<Uint8Array> {
    chacha20_xor_arena_core(
        &arena_bytes,
        &key_32,
        &nonce_12,
        start_block_counter.unwrap_or(0),
    )
    .map(Uint8Array::new)
    .map_err(napi_err)
}

#[napi]
pub fn wrap_device_cek_hex_native(
    ephemeral_secret_32: Uint8Array,
    device_pubkey_32: Uint8Array,
    device_id_hash: String,
    cek_32: Uint8Array,
) -> Result<String> {
    wrap_device_cek_hex_core(
        &ephemeral_secret_32,
        &device_pubkey_32,
        &device_id_hash,
        &cek_32,
    )
    .map_err(napi_err)
}

#[napi]
#[allow(clippy::too_many_arguments)]
pub fn seal_container_native(
    inner_dnam_bytes: Uint8Array,
    recipient_certs_json: String,
    trusted_oem_pubkeys_json: String,
    marketplace_seed_32: Uint8Array,
    ephemeral_secret_32: Uint8Array,
    cek_32: Uint8Array,
    nonce_12: Uint8Array,
) -> Result<Uint8Array> {
    seal_container_core(
        &inner_dnam_bytes,
        &recipient_certs_json,
        &trusted_oem_pubkeys_json,
        &marketplace_seed_32,
        &ephemeral_secret_32,
        &cek_32,
        &nonce_12,
    )
    .map(Uint8Array::new)
    .map_err(napi_err)
}

#[napi]
pub fn unseal_container_native(
    sealed_bytes: Uint8Array,
    device_id_hash: Option<String>,
    device_secret_32: Option<Uint8Array>,
    trusted_marketplace_pubkey_hex: Option<String>,
) -> Result<Uint8Array> {
    unseal_container_core(
        &sealed_bytes,
        device_id_hash.as_deref(),
        device_secret_32.as_deref(),
        trusted_marketplace_pubkey_hex.as_deref(),
    )
    .map(Uint8Array::new)
    .map_err(napi_err)
}

#[napi]
pub fn encode_dau1_envelope_native(spec_json: String) -> Result<Uint8Array> {
    encode_dau1_envelope_core(&spec_json)
        .map(Uint8Array::new)
        .map_err(napi_err)
}

#[napi]
pub fn decode_dau1_envelope_native(wire_bytes: Uint8Array) -> Result<String> {
    decode_dau1_envelope_core(&wire_bytes).map_err(napi_err)
}

#[napi]
pub fn build_dme1_header_native(
    profile: u8,
    recipient_count: u32,
    ciphertext_length: f64,
) -> Result<Uint8Array> {
    build_dme1_header_core(profile, recipient_count, ciphertext_length as u64)
        .map(Uint8Array::new)
        .map_err(napi_err)
}

#[napi]
pub fn parse_dme1_plaintext_native(
    plaintext: Uint8Array,
    recipient_key_digests_hex: Vec<String>,
) -> Result<String> {
    parse_dme1_plaintext_core(&plaintext, &recipient_key_digests_hex).map_err(napi_err)
}

#[napi]
pub fn encode_recipient_key_ids_native(key_digests_hex: Vec<String>) -> Result<Uint8Array> {
    encode_recipient_key_ids_core(&key_digests_hex)
        .map(Uint8Array::new)
        .map_err(napi_err)
}

#[napi]
pub fn parse_recipient_key_ids_native(
    data: Uint8Array,
    max_recipients: Option<u32>,
) -> Result<Vec<String>> {
    parse_recipient_key_ids_core(&data, max_recipients.unwrap_or(16) as usize).map_err(napi_err)
}

#[napi]
pub fn encode_hpke_slots_native(slots_json: String) -> Result<Uint8Array> {
    let pairs: Vec<(String, String)> =
        serde_json::from_str(&slots_json).map_err(|e| napi_err(validation_err(e.to_string())))?;
    encode_hpke_slots_core(&pairs)
        .map(Uint8Array::new)
        .map_err(napi_err)
}

#[napi]
pub fn parse_hpke_slots_native(
    data: Uint8Array,
    expected_count: u32,
    max_recipients: Option<u32>,
) -> Result<String> {
    let slots = parse_hpke_slots_core(
        &data,
        expected_count as usize,
        max_recipients.unwrap_or(16) as usize,
    )
    .map_err(napi_err)?;
    serde_json::to_string(&slots).map_err(|e| napi_err(validation_err(e.to_string())))
}

#[napi]
pub fn encode_record_directory_native(
    maximum_tier: u8,
    records_json: String,
) -> Result<Uint8Array> {
    let recs: Vec<(u64, u64)> =
        serde_json::from_str(&records_json).map_err(|e| napi_err(validation_err(e.to_string())))?;
    encode_record_directory_core(maximum_tier, &recs)
        .map(Uint8Array::new)
        .map_err(napi_err)
}

#[napi]
pub fn parse_record_directory_native(data: Uint8Array, maximum_tier: u8) -> Result<String> {
    let recs = parse_record_directory_core(&data, maximum_tier).map_err(napi_err)?;
    serde_json::to_string(&recs).map_err(|e| napi_err(validation_err(e.to_string())))
}

#[napi]
pub fn compute_preview_window_native(
    anchor: f64,
    delta: f64,
    safe_min: f64,
    safe_max: f64,
) -> Result<Float64Array> {
    let res = compute_preview_window_core(anchor, delta, (safe_min, safe_max)).map_err(napi_err)?;
    Ok(Float64Array::new(res.to_vec()))
}

#[napi]
pub fn validate_preview_switch_options_native(
    preset: String,
    alternative: String,
    observed: Vec<String>,
    active: Vec<String>,
    safe: Vec<String>,
) -> Result<Vec<String>> {
    let (p, a) =
        validate_preview_switch_options_core(&preset, &alternative, &observed, &active, &safe)
            .map_err(napi_err)?;
    Ok(vec![p, a])
}

#[napi]
pub fn validate_royalty_splits_native(
    license_tier: String,
    royalty_splits_json: String,
) -> Result<()> {
    validate_royalty_splits_core(&license_tier, &royalty_splits_json).map_err(napi_err)
}

#[napi]
pub fn apply_isometric_gauge_watermark_native(
    arena_bytes: Uint8Array,
    kappa: Float32Array,
    q_buyer_flat_16: Float32Array,
) -> Result<Uint8Array> {
    apply_isometric_gauge_watermark_core(&arena_bytes, &kappa, &q_buyer_flat_16)
        .map(Uint8Array::new)
        .map_err(napi_err)
}

#[napi]
pub fn generate_tardos_code_native(
    marketplace_secret_32: Uint8Array,
    model_id: String,
    license_id: String,
) -> Result<Float32Array> {
    generate_tardos_code_core(&marketplace_secret_32, &model_id, &license_id)
        .map(Float32Array::new)
        .map_err(napi_err)
}

#[napi]
pub fn apply_tardos_watermark_native(
    arena_bytes: Uint8Array,
    code: Float32Array,
    epsilon: f64,
) -> Result<Uint8Array> {
    apply_tardos_watermark_core(&arena_bytes, &code, epsilon as f32)
        .map(Uint8Array::new)
        .map_err(napi_err)
}

#[napi]
pub fn score_tardos_watermark_native(
    master_arena: Uint8Array,
    suspect_arena: Uint8Array,
    code: Float32Array,
) -> Result<f64> {
    score_tardos_watermark_core(&master_arena, &suspect_arena, &code).map_err(napi_err)
}

#[napi]
pub fn extract_structural_hash_native(arena_bytes: Uint8Array) -> Result<Float64Array> {
    extract_structural_hash_core(&arena_bytes)
        .map(Float64Array::new)
        .map_err(napi_err)
}

#[napi]
pub fn structural_cosine_similarity_native(a: Float64Array, b: Float64Array) -> Result<f64> {
    structural_cosine_similarity_core(&a, &b).map_err(napi_err)
}

#[napi]
#[allow(clippy::too_many_arguments)]
pub fn derive_watermark_seed_hex_native(
    key_32: Uint8Array,
    marketplace_pubkey_32: Uint8Array,
    watermark_key_id_16: Uint8Array,
    creator_signed_manifest_sha256_32: Uint8Array,
    model_id_16: Uint8Array,
    watermark_profile_sha256_32: Uint8Array,
    owning_tier: u8,
    license_id_16: Option<Uint8Array>,
    buyer_id_16: Option<Uint8Array>,
) -> Result<String> {
    derive_watermark_seed_hex_core(
        &key_32,
        &marketplace_pubkey_32,
        &watermark_key_id_16,
        &creator_signed_manifest_sha256_32,
        &model_id_16,
        &watermark_profile_sha256_32,
        owning_tier,
        license_id_16.as_deref(),
        buyer_id_16.as_deref(),
    )
    .map_err(napi_err)
}

#[napi]
pub fn extract_stage3_fingerprint_native(
    early_fir_128b: Uint8Array,
    kautz_modal_bytes: Uint8Array,
    added_modes: u32,
) -> Result<Float64Array> {
    extract_stage3_fingerprint_core(&early_fir_128b, &kautz_modal_bytes, added_modes as usize)
        .map(Float64Array::new)
        .map_err(napi_err)
}
