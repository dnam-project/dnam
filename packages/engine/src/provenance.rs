//! Marketplace Provenance, `Ed25519` Signatures, Hardware Enclaves & Traitor Tracing (`provenance.rs`).
//!
//! Implements the 5-Layer Defense-in-Depth architecture from
//! [`docs/security/provenance_and_enclave_drm.md`](../../../docs/security/provenance_and_enclave_drm.md)
//! and [`docs/security/watermarking_and_marketplace.md`](../../../docs/security/watermarking_and_marketplace.md):
//! 1. **Preset-Safe RFC 8785 JCS `immutable_header_sha256` & 2-Certificate `Ed25519` Chain**
//!    ([`VerificationStatus`], [`verify_container_provenance`]).
//! 2. **Hardware-Agnostic [`CryptoProvider`] Trait** supporting the built-in `#![no_std]`
//!    [`RustCryptoProvider`] (`ENCRYPTED_ARENA = 0` & software-keyed `ENCRYPTED_ARENA = 1`),
//!    Rockchip RK3582 `OP-TEE` TrustZone, Apple Secure Enclave (`SEP`), and Windows/Linux `TPM 2.0`.
//! 3. **RFC 8439 `ChaCha20` 64-Byte Cache-Line Arena Stream Encryption** (`Lite=168`, `Full=349`,
//!    `Pro=566`, `Ultra=1000` blocks, `0` bytes padding overhead).
//! 4. **Dual-Mechanism Buyer Watermarking (`Test 8H`):**
//!    - **Mechanism A:** 64-channel modal/Kautz reciprocal scaling ($\kappa_{b,c}$) + $SO(4)$ SVD head rotation ($\mathbf{Q}_{\text{buyer}}$).
//!    - **Mechanism B:** 7,680-weight Coupled Spread-Spectrum Tardos code surviving 4-buyer collusion + `fp16` truncation.
//! 5. **224-Dimensional Gauge-Invariant Structural Hash** ($\mathbf{f}_{\text{struct}} \in \mathbb{R}^{224}$).

mod chacha20;
pub mod fingerprint;
mod provider;
pub mod seal;
pub mod security;
mod sha256;
mod verify;
pub(crate) use verify::canonicalize_jcs_value;
pub(crate) mod watermark;

pub use security::{
    AnticloneGateAction, Dau1Envelope, HPKE_SUITE_X25519_SHA256_CHACHA20POLY1305,
    MAX_RECORD_CIPHERTEXT_BYTES, MAX_RECORD_PLAINTEXT_BYTES, WatermarkSeedScope,
    build_buyer_receipt_v1, build_creator_manifest_v1, build_dme1_header,
    build_hpke_slot_binding_v1, build_mutable_content_aad_v1, build_mutable_hpke_binding_v1,
    build_oem_cert_v2_signing_message, build_outer_manifest_v1, build_plaintext_shared_v1,
    build_prewrap_plan_v1, build_record_aad_v1, build_record_nonce_v1,
    build_watermark_seed_message_v1, compute_ciphertext_prefixes_v1,
    compute_creator_signed_manifest_sha256, compute_encrypted_structural_metadata_sha256,
    compute_license_policy_sha256, compute_logical_metadata_sha256, compute_mutable_wrap_sha256,
    compute_outer_public_metadata_sha256, compute_plaintext_prefixes_v1,
    compute_preview_single_control_window, compute_record_context_sha256,
    compute_watermark_profile_sha256, compute_wrap_plan_sha256, derive_watermark_seed_hmac_sha256,
    encode_hpke_slot_vector, encode_prefix_list, encode_recipient_key_ids, encode_record_directory,
    evaluate_anticlone_revocation_policy, extract_stage3_tier_fingerprint_component,
    extract_storefront_metadata, hmac_sha256, parse_dme1_decrypted_plaintext,
    parse_hpke_slot_vector, parse_prefix_list, parse_recipient_key_ids, parse_record_directory,
    validate_license_tier_and_royalty_splits, validate_preview_single_switch_options,
};

pub use chacha20::{chacha20_block_64b, chacha20_xor_arena_blocks};
#[cfg(feature = "crypto")]
pub use provider::RustCryptoProvider;
pub use provider::{
    CryptoProvider, DefaultCryptoProvider, OpenArenaCryptoProvider, VerificationStatus,
};
pub use sha256::{decode_hex_array, encode_hex, sha256_digest, sha256_software_digest};
pub use verify::{
    DNSL_MAGIC, DnamSealEnvelopePreamble, DnamSealRecipientSlot, FLAG_SEAL_CHACHA20_FULL,
    MAX_SEAL_RECIPIENTS, build_creator_signing_message, build_marketplace_receipt_signing_message,
    build_oem_device_cert_signing_message, compute_immutable_header_sha256,
    compute_immutable_header_sha256_for_version, enforce_distribution_policy,
    project_immutable_header_for_schema_version, unseal_dnam_container,
    verify_container_provenance, verify_oem_device_certificate,
};
#[cfg(feature = "crypto")]
pub use verify::{
    MarketplaceReceiptSigningSpec, seal_dnam_container, sign_creator_header_json,
    sign_marketplace_receipt_json,
};
pub use watermark::{
    PHASE_CODING_MAX_RAD, STAGE3_EARLY_FIR_BASES, STAGE3_KAUTZ_BASES, STAGE3_KAUTZ_BLOCKS_PER_TIER,
    STAGE3_KAUTZ_TIER_CHANNEL_BASE, TARDOS_CASCADE_RANGES, TardosCodebook7680,
    apply_cascade_head_gauge, apply_isometric_gauge_watermark, apply_phase_coding_watermark,
    apply_tardos_watermark_7680, apply_tardos_watermark_7680_envelopes,
    compute_emr_masked_tardos_envelopes, extract_structural_hash_224d, generate_tardos_code_7680,
    generate_tardos_codebook_7680, is_role_a_cascade_index, kautz_soa_byte_offset,
    role_b_head_slot_for_index, score_phase_coding_arena, score_phase_coding_blackbox,
    score_tardos_watermark_7680, structural_cosine_similarity,
};

/// Number of Stage II commuted nonlinear cascade projection weights across `Arena 0..3` (`768 + 1536 + 2304 + 3072 = 7680`).
pub const TARDOS_CASCADE_WEIGHTS: usize = 7680;

/// Number of modal/Kautz complex resonator channels across `Arena 0..3` (`4 blocks × 16 channels = 64`).
pub const KAUTZ_TOTAL_CHANNELS: usize = 64;

/// Per-tier dimension of the prefix-composable gauge-invariant structural hash (`56-D` per tier).
pub const STRUCTURAL_HASH_TIER_DIM: usize = 56;

/// Dimension of the 4-tier prefix-composable gauge-invariant structural hash vector $\mathbf{f}_{\text{struct}} \in \mathbb{R}^{224}$ (`4 × 56-D`).
pub const STRUCTURAL_HASH_DIM: usize = 224;
