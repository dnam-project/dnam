//! Canonical v2.0 Cryptographic Provenance (`DAU1` / `DAI1` / `DSA1`), `OemDeviceCertificate` V2,
//! RFC 9180 HPKE + `ChaCha20-Poly1305` Record Framing, `DNSL` / `DNPL` / `DME1` Envelopes,
//! License-Scoped Watermark Seed Derivation (`HMAC-SHA256`), Stage III Tier-Local Structural
//! Fingerprinting, and `.dnam-prev` Preview Containment.
//!
//! Implements:
//! - `ref[sec.taxonomy.overview]`, `ref[sec.jcs.immutable-header-digest]`,
//!   `ref[sec.policy.distribution-policy]`
//! - `ref[sec.sig.prefix-delegation]`, `ref[sec.sig.dau1-envelope]`,
//!   `ref[sec.sig.offline-auth]`
//! - `ref[sec.drm.oem-certificate]`, `ref[sec.drm.trustzone-flow]`,
//!   `ref[sec.drm.hpke-record-encryption]`, `ref[sec.drm.dnsl-public-framing]`,
//!   `ref[sec.drm.protected-inner-framing]`, `ref[sec.drm.dme1-mutable-suffix]`,
//!   `ref[sec.drm.protection-profiles]`
//! - `ref[sec.watermark.reciprocal-gauge]`, `ref[sec.watermark.orthogonal-gauge]`,
//!   `ref[sec.watermark.tardos-code]`, `ref[sec.watermark.seed-derivation]`
//! - `ref[sec.anticlone.three-stage-gate]`, `ref[sec.anticlone.stage3-fingerprint]`
//! - `ref[sec.preview.containment]`, `ref[sec.preview.royalty-splits]`,
//!   `ref[sec.preview.storefront-metadata]`, `ref[sec.edge.data-model]`

pub mod oem_and_drm;
pub mod prefix_and_dau1;
pub mod prefix_delegation_crypto;
pub mod watermark_and_preview;

pub use oem_and_drm::{
    HPKE_SUITE_X25519_SHA256_CHACHA20POLY1305, HpkeSlotPair, MAX_RECORD_CIPHERTEXT_BYTES,
    MAX_RECORD_PLAINTEXT_BYTES, build_hpke_slot_binding_v1, build_oem_cert_v2_signing_message,
    build_prewrap_plan_v1, build_record_aad_v1, build_record_nonce_v1,
    compute_encrypted_structural_metadata_sha256, compute_record_context_sha256,
    compute_wrap_plan_sha256, encode_hpke_slot_vector, encode_recipient_key_ids,
    encode_record_directory, parse_hpke_slot_vector, parse_recipient_key_ids,
    parse_record_directory,
};
pub use prefix_and_dau1::{
    Dau1Envelope, build_plaintext_shared_v1, compute_ciphertext_prefixes_v1,
    compute_plaintext_prefixes_v1, encode_prefix_list, hmac_sha256, parse_prefix_list,
};
pub use prefix_delegation_crypto::{
    build_buyer_receipt_v1, build_creator_manifest_v1, build_outer_manifest_v1,
    compute_creator_signed_manifest_sha256, compute_license_policy_sha256,
    compute_outer_public_metadata_sha256,
};
pub use watermark_and_preview::{
    AnticloneGateAction, WatermarkSeedScope, build_dme1_header, build_mutable_content_aad_v1,
    build_mutable_hpke_binding_v1, build_watermark_seed_message_v1,
    compute_logical_metadata_sha256, compute_mutable_wrap_sha256,
    compute_preview_single_control_window, compute_watermark_profile_sha256,
    derive_watermark_seed_hmac_sha256, evaluate_anticlone_revocation_policy,
    extract_stage3_tier_fingerprint_component, extract_storefront_metadata,
    parse_dme1_decrypted_plaintext, validate_license_tier_and_royalty_splits,
    validate_preview_single_switch_options,
};
