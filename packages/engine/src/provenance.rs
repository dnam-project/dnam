//! Marketplace Provenance, `Ed25519` Signatures, Hardware Enclaves & Traitor Tracing (`provenance.rs`).
//!
//! Implements the 5-Layer Defense-in-Depth architecture from
//! [`docs/digital_marketplace_and_provenance.md`](../../../docs/digital_marketplace_and_provenance.md):
//! 1. **Preset-Safe RFC 8785 JCS `immutable_header_sha256` & 2-Certificate `Ed25519` Chain**
//!    ([`VerificationStatus`], [`verify_container_provenance`]).
//! 2. **Hardware-Agnostic [`CryptoProvider`] Trait** supporting the built-in `#![no_std]`
//!    [`RustCryptoProvider`] (`ENCRYPTED_ARENA = 0` & software-keyed `ENCRYPTED_ARENA = 1`),
//!    Rockchip RK3582 `OP-TEE` TrustZone, Apple Secure Enclave (`SEP`), and Windows/Linux `TPM 2.0`.
//! 3. **RFC 8439 `ChaCha20` 64-Byte Cache-Line Arena Stream Encryption** (`Lite=168`, `Full=349`,
//!    `Ultra=566` blocks, `0` bytes padding overhead).
//! 4. **Dual-Mechanism Buyer Watermarking (`Test 8H`):**
//!    - **Mechanism A:** 48-channel RG-LRU reciprocal scaling ($\kappa_{b,c}$) + $SO(4)$ SVD head rotation ($\mathbf{Q}_{\text{buyer}}$).
//!    - **Mechanism B:** 4,608-weight Coupled Spread-Spectrum Tardos code surviving 4-buyer collusion + `fp16` truncation.
//! 5. **160-Dimensional Gauge-Invariant Structural Hash** ($\mathbf{f}_{\text{struct}} \in \mathbb{R}^{160}$).

mod chacha20;
mod provider;
mod sha256;
mod verify;
mod watermark;

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
    apply_isometric_gauge_watermark, apply_tardos_watermark_4608, extract_structural_hash_160d,
    generate_tardos_code_4608, rglru_soa_byte_offset, score_tardos_watermark_4608,
    structural_cosine_similarity,
};

/// Number of nonlinear dilated convolution weights across `Arena 0..2` (`768 + 1536 + 2304 = 4608`).
pub const TARDOS_CONV_WEIGHTS: usize = 4608;

/// Number of RG-LRU complex resonator channels across `Arena 0..2` (`4 blocks × 12 channels = 48`).
pub const RGLRU_TOTAL_CHANNELS: usize = 48;

/// Dimension of the gauge-invariant structural hash vector $\mathbf{f}_{\text{struct}} \in \mathbb{R}^{160}$.
pub const STRUCTURAL_HASH_DIM: usize = 160;
