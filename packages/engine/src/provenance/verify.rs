//! RFC 8785 JCS `immutable_header_sha256` & 2-Certificate `Ed25519` Verification.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use crate::schema::{DnamError, DnamJsonHeader, MarketplaceReceipt};

use super::provider::{CryptoProvider, VerificationStatus};
use super::sha256::{decode_hex_array, sha256_digest};

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
        }
    }
}
