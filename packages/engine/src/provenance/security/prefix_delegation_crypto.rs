//! Two-Certificate Ed25519 Prefix Delegation Manifests, Buyer Receipts, Outer Manifests &
//! Cryptographic Reference Suite (`ref[sec.sig.prefix-delegation]`, `ref[sec.sig.offline-auth]`).

use alloc::vec::Vec;

use super::prefix_and_dau1::encode_prefix_list;
use crate::container::encode_domain_field;
use crate::provenance::sha256::sha256_digest;
use crate::schema::{DnamError, SchemaOptionExt, reject_if};

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
