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
mod sha256;
mod verify;
pub(crate) mod watermark;

pub use chacha20::{chacha20_block_64b, chacha20_xor_arena_blocks};
#[cfg(feature = "crypto")]
pub use provider::RustCryptoProvider;
pub use provider::{
    CryptoProvider, DefaultCryptoProvider, OpenArenaCryptoProvider, VerificationStatus,
};
pub use sha256::{decode_hex_array, encode_hex, sha256_digest};
#[cfg(feature = "crypto")]
pub use verify::seal_dnam_container;
pub use verify::{
    DNSL_MAGIC, DnamSealEnvelopePreamble, DnamSealRecipientSlot, FLAG_SEAL_CHACHA20_FULL,
    MAX_SEAL_RECIPIENTS, build_creator_signing_message, build_marketplace_receipt_signing_message,
    build_oem_device_cert_signing_message, compute_immutable_header_sha256,
    compute_immutable_header_sha256_for_version, enforce_distribution_policy,
    project_immutable_header_for_schema_version, unseal_dnam_container,
    verify_container_provenance, verify_oem_device_certificate,
};
pub use watermark::{
    apply_isometric_gauge_watermark, apply_tardos_watermark_7680, extract_structural_hash_224d,
    generate_tardos_code_7680, kautz_soa_byte_offset, score_tardos_watermark_7680,
    structural_cosine_similarity,
};

/// Number of Stage II commuted nonlinear cascade projection weights across `Arena 0..3` (`768 + 1536 + 2304 + 3072 = 7680`).
pub const TARDOS_CASCADE_WEIGHTS: usize = 7680;

/// Number of modal/Kautz complex resonator channels across `Arena 0..3` (`4 blocks × 16 channels = 64`).
pub const KAUTZ_TOTAL_CHANNELS: usize = 64;

/// Per-tier dimension of the prefix-composable gauge-invariant structural hash (`56-D` per tier).
pub const STRUCTURAL_HASH_TIER_DIM: usize = 56;

/// Dimension of the 4-tier prefix-composable gauge-invariant structural hash vector $\mathbf{f}_{\text{struct}} \in \mathbb{R}^{224}$ (`4 × 56-D`).
pub const STRUCTURAL_HASH_DIM: usize = 224;
