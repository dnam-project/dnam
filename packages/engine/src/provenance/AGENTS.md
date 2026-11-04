# `src/provenance` — Marketplace Signatures, Hardware Enclaves & Traitor Tracing

Implements cryptographic provenance, hardware enclave DRM, and forensic watermarking (`#[forbid(unsafe_code)]`).

## 1. Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`security.rs`](security.rs) | `DAU1` envelope, prefix commitments, `OemDeviceCertificate`, RFC 9180 HPKE, `DME1`, watermark seeds & `.dnam-prev` containment | [`provenance_and_enclave_drm.md`](../../../../docs/specs/security/provenance_and_enclave_drm.md), [`watermarking_and_marketplace.md`](../../../../docs/specs/security/watermarking_and_marketplace.md) |
| [`provider.rs`](provider.rs) | `CryptoProvider` trait, `OpenArenaCryptoProvider`, `RustCryptoProvider`, `VerificationStatus` | [`provenance_and_enclave_drm.md`](../../../../docs/specs/security/provenance_and_enclave_drm.md) |
| [`verify.rs`](verify.rs) | Current header serialization and `verify_container_provenance` | [`provenance_and_enclave_drm.md`](../../../../docs/specs/security/provenance_and_enclave_drm.md) |
| [`seal.rs`](seal.rs) | `.dnam-seal` (`DNSL`) outer envelope encryption, unsealing, and OEM device certificates | [`provenance_and_enclave_drm.md`](../../../../docs/specs/security/provenance_and_enclave_drm.md) |
| [`chacha20.rs`](chacha20.rs) | `#![no_std]` RFC 8439 `ChaCha20` 64-byte cache-line block stream cipher + `proptest` | [`provenance_and_enclave_drm.md`](../../../../docs/specs/security/provenance_and_enclave_drm.md) |
| [`sha256.rs`](sha256.rs) | `#![no_std]` FIPS 180-4 `SHA-256` & constant-time hex encoding/decoding + `proptest` | [`provenance_and_enclave_drm.md`](../../../../docs/specs/security/provenance_and_enclave_drm.md) |
| [`watermark.rs`](watermark.rs) | Isometric Gauge Watermark & 7,680-chip Tardos traitor-tracing codes + `proptest` | [`watermarking_and_marketplace.md`](../../../../docs/specs/security/watermarking_and_marketplace.md) |
| [`fingerprint.rs`](fingerprint.rs) | Prefix-composable 224-D gauge-invariant structural hash (`4 × 56-D`) & cosine similarity | [`watermarking_and_marketplace.md`](../../../../docs/specs/security/watermarking_and_marketplace.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for JCS canonicalization and preset-invariance rules.
