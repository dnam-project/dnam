//! OEM Device Certificate V2 & RFC 9180 HPKE + `ChaCha20-Poly1305` Record Framing.
//!
//! Canonical Reference: `docs/specs/security/provenance_and_enclave_drm.md` (`ref[sec.drm.oem-certificate]`, `ref[sec.drm.trustzone-flow]`, `ref[sec.drm.hpke-record-encryption]`, `ref[sec.drm.dnsl-public-framing]`, `ref[sec.drm.protected-inner-framing]`, `ref[sec.drm.protection-profiles]`).

use alloc::vec::Vec;

use crate::container::{align64_u64, encode_domain_field};
use crate::provenance::sha256::sha256_digest;
use crate::schema::{DnamError, SchemaOptionExt, reject_if};

/// Maximum plaintext byte length for a single encrypted container record (`(2^32 - 1) * 64`).
pub const MAX_RECORD_PLAINTEXT_BYTES: u64 = ((1u64 << 32) - 1) * 64;
/// Maximum ciphertext byte length (`MAX_RECORD_PLAINTEXT_BYTES + 16`).
pub const MAX_RECORD_CIPHERTEXT_BYTES: u64 = MAX_RECORD_PLAINTEXT_BYTES + 16;
/// RFC 9180 HPKE cipher suite identifier (`DHKEM(X25519, HKDF-SHA256) = 0x0020`,
/// `HKDF-SHA256 = 0x0001`, `ChaCha20Poly1305 = 0x0003`, `ref[sec.drm.hpke-record-encryption]`).
pub const HPKE_SUITE_X25519_SHA256_CHACHA20POLY1305: [u8; 6] = [0x00, 0x20, 0x00, 0x01, 0x00, 0x03];

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

#[cfg(test)]
mod spec_tests {
    use crate::container::build_dmu1_mutable_frame;
    use crate::provenance::{
        build_dme1_header, build_hpke_slot_binding_v1, build_mutable_content_aad_v1,
        build_mutable_hpke_binding_v1, build_oem_cert_v2_signing_message, build_outer_manifest_v1,
        build_prewrap_plan_v1, build_record_aad_v1, build_record_nonce_v1,
        compute_ciphertext_prefixes_v1, compute_encrypted_structural_metadata_sha256,
        compute_logical_metadata_sha256, compute_mutable_wrap_sha256,
        compute_outer_public_metadata_sha256, compute_record_context_sha256,
        compute_wrap_plan_sha256, encode_hpke_slot_vector, encode_recipient_key_ids,
        encode_record_directory, parse_dme1_decrypted_plaintext, parse_hpke_slot_vector,
        parse_recipient_key_ids, parse_record_directory, sha256_digest,
    };
    use alloc::vec::Vec;
    use serde_json::json;

    /// verify[sec.drm.oem-certificate]
    /// verify[sec.drm.trustzone-flow]
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

        let ct_prefixes =
            compute_ciphertext_prefixes_v1(&ctx_sha, &[1u8; 64], &[&[2u8; 64], &[3u8; 64]])
                .unwrap();
        let outer_meta_sha = compute_outer_public_metadata_sha256(&[0; 64], &[0; 128], &plan);
        let outer_msg =
            build_outer_manifest_v1(&[b'k'; 32], &outer_meta_sha, &ctx_sha, &ct_prefixes).unwrap();
        assert!(!outer_msg.is_empty());

        let dnpl_sha = compute_logical_metadata_sha256(b"DNPL_PREFIX");
        assert_ne!(dnpl_sha, [0u8; 32]);

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

        let mut reversed_pt = Vec::new();
        reversed_pt.extend_from_slice(&2u32.to_le_bytes());
        reversed_pt.extend_from_slice(&keys[1]);
        reversed_pt.extend_from_slice(&keys[0]);
        reversed_pt.extend_from_slice(&dmu1);
        assert!(parse_dme1_decrypted_plaintext(&reversed_pt, &recipients).is_err());
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
