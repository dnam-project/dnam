# `src/provenance` — Marketplace Signatures, Hardware Enclaves & Traitor Tracing

Implements the 5-Layer Digital Marketplace & Provenance architecture (`#[forbid(unsafe_code)]`): RFC 8785 JCS `immutable_header_sha256`, 2-certificate `Ed25519` verification, `X25519-HKDF-ChaCha20` 64-byte block-aligned arena encryption, Isometric Gauge Watermarking, 4,608-chip Coupled Tardos Traitor-Tracing codes, and 160-D Structural Hashing.

## 1. Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`provider.rs`](provider.rs) | `CryptoProvider` trait, `OpenArenaCryptoProvider`, `RustCryptoProvider`, `VerificationStatus` | [`digital_marketplace_and_provenance.md`](../../../../docs/digital_marketplace_and_provenance.md) |
| [`verify.rs`](verify.rs) | RFC 8785 JCS canonicalization (`presets` & `user_metadata` excluded) & `verify_container_provenance` | [`digital_marketplace_and_provenance.md`](../../../../docs/digital_marketplace_and_provenance.md) |
| [`chacha20.rs`](chacha20.rs) | `#![no_std]` RFC 8439 `ChaCha20` 64-byte cache-line block stream cipher + `proptest` | [`digital_marketplace_and_provenance.md`](../../../../docs/digital_marketplace_and_provenance.md) |
| [`sha256.rs`](sha256.rs) | `#![no_std]` FIPS 180-4 `SHA-256` & constant-time hex encoding/decoding + `proptest` | [`digital_marketplace_and_provenance.md`](../../../../docs/digital_marketplace_and_provenance.md) |
| [`watermark.rs`](watermark.rs) | Isometric Gauge Watermark, 4,608-chip Tardos code, 160-D structural hash + `proptest` | [`digital_marketplace_and_provenance.md`](../../../../docs/digital_marketplace_and_provenance.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for JCS canonicalization and preset-invariance rules.
