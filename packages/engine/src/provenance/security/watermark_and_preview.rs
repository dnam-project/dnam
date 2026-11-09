//! `DME1` Encrypted Mutable Suffix, Watermark Seed Derivation (`HMAC-SHA256`), Stage III Tier-Local
//! Structural Fingerprinting & Preview Containment.
//!
//! Canonical Reference: `docs/specs/security/provenance_and_enclave_drm.md` & `docs/specs/security/watermarking_and_marketplace.md` (`ref[sec.drm.dme1-mutable-suffix]`, `ref[sec.drm.protected-inner-framing]`, `ref[sec.watermark.seed-derivation]`, `ref[sec.anticlone.three-stage-gate]`, `ref[sec.anticlone.stage3-fingerprint]`, `ref[sec.preview.containment]`, `ref[sec.preview.royalty-splits]`, `ref[sec.preview.storefront-metadata]`, `ref[sec.edge.data-model]`).

use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::vec::Vec;

use super::oem_and_drm::{encode_hpke_slot_vector, encode_recipient_key_ids};
use crate::container::{encode_domain_field, parse_and_validate_dmu1_frame};
use crate::provenance::security::hmac_sha256;
use crate::provenance::sha256::sha256_digest;
use crate::schema::{DnamError, RoyaltySplit, reject_if};

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
    use crate::container::ARENA_OFFSET_ULTRA_END;
    use crate::provenance::{
        AnticloneGateAction, WatermarkSeedScope, apply_isometric_gauge_watermark,
        apply_tardos_watermark_7680, build_watermark_seed_message_v1,
        compute_preview_single_control_window, compute_watermark_profile_sha256,
        derive_watermark_seed_hmac_sha256, evaluate_anticlone_revocation_policy,
        extract_stage3_tier_fingerprint_component, extract_storefront_metadata,
        extract_structural_hash_224d, generate_tardos_code_7680, kautz_soa_byte_offset,
        score_tardos_watermark_7680, sha256_digest, structural_cosine_similarity,
        validate_license_tier_and_royalty_splits, validate_preview_single_switch_options,
    };
    use crate::schema::{DnamError, DnamJsonHeader, RoyaltySplit};
    use alloc::{
        collections::BTreeSet,
        format,
        string::{String, ToString},
        vec,
    };
    use serde_json::json;

    /// verify[sec.watermark.reciprocal-gauge]
    /// verify[sec.watermark.orthogonal-gauge]
    /// verify[sec.watermark.tardos-code]
    /// verify[sec.watermark.seed-derivation]
    /// verify[sec.anticlone.three-stage-gate]
    /// verify[sec.anticlone.stage3-fingerprint]
    #[test]
    fn test_watermark_seed_derivation_gauges_tardos_and_stage3_fingerprint() {
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

        let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        for t in 0..4 {
            for blk in 0..crate::provenance::STAGE3_KAUTZ_BLOCKS_PER_TIER[t] {
                let base = kautz_soa_byte_offset(t, blk);
                for l in 0..4 {
                    arena[base + l * 4..base + (l + 1) * 4].copy_from_slice(&0.9_f32.to_le_bytes());
                    arena[base + (12 + l) * 4..base + (13 + l) * 4]
                        .copy_from_slice(&0.5_f32.to_le_bytes());
                    arena[base + (16 + l) * 4..base + (17 + l) * 4]
                        .copy_from_slice(&0.25_f32.to_le_bytes());
                    arena[base + (24 + l) * 4..base + (25 + l) * 4]
                        .copy_from_slice(&0.15_f32.to_le_bytes());
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
    #[test]
    fn test_preview_containment_royalty_splits_and_storefront_metadata() {
        let mut header = DnamJsonHeader::default_reference();
        header.container_kind = "preview".to_string();
        let err = header
            .ensure_not_preview_for_daw_host()
            .expect_err("DAW host must reject preview container");
        assert!(matches!(err, DnamError::PreviewContainerRejected));

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

        let obs = BTreeSet::from(["a", "b", "c"]);
        let act = BTreeSet::from(["a", "b"]);
        let safe = BTreeSet::from(["a", "b"]);
        assert_eq!(
            validate_preview_single_switch_options("a", "b", &obs, &act, &safe).unwrap(),
            ("a", "b")
        );
        assert!(validate_preview_single_switch_options("a", "c", &obs, &act, &safe).is_err());
        assert!(validate_preview_single_switch_options("a", "a", &obs, &act, &safe).is_err());

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
}
