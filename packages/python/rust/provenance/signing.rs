//! RFC 8785 JCS SHA-256, `Ed25519` / `X25519` keys, creator & marketplace signing messages,
//! and `OemDeviceCertificate` V1/V2 attestation (`ref[sdk.python.provenance.v2-security]`).

use dnam_engine::provenance::{
    RustCryptoProvider, build_creator_signing_message, build_marketplace_receipt_signing_message,
    build_oem_cert_v2_signing_message, compute_immutable_header_sha256, decode_hex_array,
    encode_hex, hmac_sha256, sha256_digest, verify_oem_device_certificate,
};
use dnam_engine::schema::{DnamJsonHeader, MarketplaceReceipt, OemDeviceCertificate};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::{dnam_err_to_py, validation_err};

fn check_len<const N: usize>(bytes: &[u8], name: &str) -> PyResult<[u8; N]> {
    bytes
        .try_into()
        .map_err(|_| validation_err(format!("{name} must be {N} bytes, got {}", bytes.len())))
}

/// Computes the preset-safe RFC 8785 JCS `immutable_header_sha256` hex digest over `header_json`.
#[pyfunction]
pub fn native_immutable_header_sha256(py: Python<'_>, header_json: &str) -> PyResult<String> {
    py.detach(|| {
        let header: DnamJsonHeader = serde_json::from_str(header_json).map_err(validation_err)?;
        let digest = compute_immutable_header_sha256(&header).map_err(dnam_err_to_py)?;
        Ok(encode_hex(&digest))
    })
}

/// Computes the lowercase hex SHA-256 digest of `data`.
#[pyfunction]
pub fn native_sha256_hex(py: Python<'_>, data: &[u8]) -> String {
    py.detach(|| encode_hex(&sha256_digest(data)))
}

/// Computes the lowercase hex HMAC-SHA256 digest of `message` under `key_32`.
#[pyfunction]
pub fn native_hmac_sha256_hex(py: Python<'_>, key_32: &[u8], message: &[u8]) -> PyResult<String> {
    let key = check_len::<32>(key_32, "key_32")?;
    Ok(py.detach(|| encode_hex(&hmac_sha256(&key, message))))
}

/// Derives the 64-character lowercase hex `Ed25519` public key from a 32-byte secret key seed.
#[pyfunction]
pub fn native_ed25519_pubkey_hex(py: Python<'_>, seed_32: &[u8]) -> PyResult<String> {
    let seed = check_len::<32>(seed_32, "seed_32")?;
    Ok(py.detach(|| {
        let pk = RustCryptoProvider::ed25519_pubkey_from_seed(&seed);
        encode_hex(&pk)
    }))
}

/// Signs `message` using a 32-byte `Ed25519` secret key seed, returning the 128-character hex signature.
#[pyfunction]
pub fn native_ed25519_sign_hex(py: Python<'_>, seed_32: &[u8], message: &[u8]) -> PyResult<String> {
    let seed = check_len::<32>(seed_32, "seed_32")?;
    Ok(py.detach(|| {
        let sig = RustCryptoProvider::ed25519_sign(&seed, message);
        encode_hex(&sig)
    }))
}

/// Derives the 64-character lowercase hex `X25519` public key from a 32-byte secret key.
#[pyfunction]
pub fn native_x25519_pubkey_hex(py: Python<'_>, secret_32: &[u8]) -> PyResult<String> {
    let secret = check_len::<32>(secret_32, "secret_32")?;
    Ok(py.detach(|| {
        let pk = RustCryptoProvider::x25519_pubkey_from_secret(&secret);
        encode_hex(&pk)
    }))
}

/// Builds the canonical `Ed25519` creator signing message (`immutable_header_sha256 || master_arena_sha256`).
#[pyfunction]
pub fn native_build_creator_signing_message<'py>(
    py: Python<'py>,
    immutable_header_sha256_hex: &str,
    master_arena_sha256_hex: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let msg = py.detach(|| -> PyResult<_> {
        let hdr_sha =
            decode_hex_array::<32>(immutable_header_sha256_hex).map_err(dnam_err_to_py)?;
        let arena_sha = decode_hex_array::<32>(master_arena_sha256_hex).map_err(dnam_err_to_py)?;
        Ok(build_creator_signing_message(&hdr_sha, &arena_sha))
    })?;
    Ok(PyBytes::new(py, &msg))
}

/// Builds the canonical `Ed25519` marketplace receipt signing message.
#[pyfunction]
pub fn native_build_marketplace_signing_message<'py>(
    py: Python<'py>,
    immutable_header_sha256_hex: &str,
    master_arena_sha256_hex: &str,
    receipt_json: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let msg = py.detach(|| -> PyResult<Vec<u8>> {
        let hdr_sha =
            decode_hex_array::<32>(immutable_header_sha256_hex).map_err(dnam_err_to_py)?;
        let arena_sha = decode_hex_array::<32>(master_arena_sha256_hex).map_err(dnam_err_to_py)?;
        let receipt: MarketplaceReceipt =
            serde_json::from_str(receipt_json).map_err(validation_err)?;
        build_marketplace_receipt_signing_message(&hdr_sha, &arena_sha, &receipt)
            .map_err(dnam_err_to_py)
    })?;
    Ok(PyBytes::new(py, &msg))
}

/// Signs an `OemDeviceCertificate` using the OEM vendor's 32-byte `Ed25519` factory seed,
/// returning the serialized JSON string of `OemDeviceCertificate`.
#[pyfunction]
pub fn native_sign_oem_device_cert(
    py: Python<'_>,
    oem_seed_32: &[u8],
    oem_vendor: &str,
    model_code: &str,
    device_id_hash: &str,
    pubkey_x25519_32: &[u8],
) -> PyResult<String> {
    let seed = check_len::<32>(oem_seed_32, "oem_seed_32")?;
    let pk_x25519 = check_len::<32>(pubkey_x25519_32, "pubkey_x25519_32")?;
    py.detach(|| {
        let cert = RustCryptoProvider::sign_oem_device_certificate(
            &seed,
            oem_vendor,
            model_code,
            device_id_hash,
            &pk_x25519,
        )
        .map_err(dnam_err_to_py)?;
        serde_json::to_string(&cert).map_err(validation_err)
    })
}

/// Verifies an `OemDeviceCertificate` JSON against a trusted 64-character hex OEM `Ed25519` public key,
/// returning the verified 64-character lowercase hex `pubkey_x25519`.
#[pyfunction]
pub fn native_verify_oem_device_cert(
    py: Python<'_>,
    cert_json: &str,
    trusted_oem_pubkey_hex: &str,
) -> PyResult<String> {
    py.detach(|| {
        let cert: OemDeviceCertificate = serde_json::from_str(cert_json).map_err(validation_err)?;
        let oem_pk = decode_hex_array::<32>(trusted_oem_pubkey_hex).map_err(dnam_err_to_py)?;
        let crypto = RustCryptoProvider::default();
        let pk_x25519 =
            verify_oem_device_certificate(&cert, &oem_pk, &crypto).map_err(dnam_err_to_py)?;
        Ok(encode_hex(&pk_x25519))
    })
}

/// Builds the domain-separated V2 OEM device certificate signing message (`"DNAM-OEM-CERT-V2"`).
#[pyfunction]
#[pyo3(signature = (oem_vendor, model_code, device_id_hash_16, pubkey_x25519_32, issuer_pubkey_ed25519_32, assurance = 1, attestation_policy_sha256_32 = None, attestation_evidence_sha256_32 = None))]
#[allow(clippy::too_many_arguments)]
pub fn native_build_oem_cert_v2_signing_message<'py>(
    py: Python<'py>,
    oem_vendor: &str,
    model_code: &str,
    device_id_hash_16: &[u8],
    pubkey_x25519_32: &[u8],
    issuer_pubkey_ed25519_32: &[u8],
    assurance: u8,
    attestation_policy_sha256_32: Option<&[u8]>,
    attestation_evidence_sha256_32: Option<&[u8]>,
) -> PyResult<Bound<'py, PyBytes>> {
    let dev_id = check_len::<16>(device_id_hash_16, "device_id_hash_16")?;
    let pk = check_len::<32>(pubkey_x25519_32, "pubkey_x25519_32")?;
    let iss_pk = check_len::<32>(issuer_pubkey_ed25519_32, "issuer_pubkey_ed25519_32")?;
    let pol = match attestation_policy_sha256_32 {
        Some(p) => check_len::<32>(p, "attestation_policy_sha256_32")?,
        None => [0u8; 32],
    };
    let ev = match attestation_evidence_sha256_32 {
        Some(e) => check_len::<32>(e, "attestation_evidence_sha256_32")?,
        None => [0u8; 32],
    };
    let msg = py
        .detach(|| {
            build_oem_cert_v2_signing_message(
                oem_vendor, model_code, &dev_id, &pk, &iss_pk, assurance, &pol, &ev,
            )
        })
        .map_err(dnam_err_to_py)?;
    Ok(PyBytes::new(py, &msg))
}

/// Signs a `DnamJsonHeader` and `master_arena` using the creator's 32-byte `Ed25519` seed in Rust,
/// returning the updated canonical wire JSON header string (`ref[sec.sig.prefix-delegation]`).
#[pyfunction]
#[pyo3(signature = (header_json, master_arena, creator_seed_32, signer_id, include_tier_prefixes = true))]
pub fn native_sign_creator_header(
    py: Python<'_>,
    header_json: &str,
    master_arena: &[u8],
    creator_seed_32: &[u8],
    signer_id: &str,
    include_tier_prefixes: bool,
) -> PyResult<String> {
    let seed = check_len::<32>(creator_seed_32, "creator_seed_32")?;
    py.detach(|| {
        dnam_engine::provenance::sign_creator_header_json(
            header_json,
            master_arena,
            &seed,
            signer_id,
            include_tier_prefixes,
        )
        .map_err(dnam_err_to_py)
    })
}

/// Signs a `MarketplaceReceipt` and attaches it to a creator-signed `DnamJsonHeader` in Rust,
/// returning the updated canonical wire JSON header string (`ref[sec.sig.prefix-delegation]`).
#[pyfunction]
pub fn native_sign_marketplace_receipt(
    py: Python<'_>,
    header_json: &str,
    buyer_arena: &[u8],
    marketplace_seed_32: &[u8],
    spec_json: &str,
) -> PyResult<String> {
    let seed = check_len::<32>(marketplace_seed_32, "marketplace_seed_32")?;
    py.detach(|| {
        let spec: dnam_engine::provenance::MarketplaceReceiptSigningSpec =
            serde_json::from_str(spec_json).map_err(validation_err)?;
        dnam_engine::provenance::sign_marketplace_receipt_json(
            header_json,
            buyer_arena,
            &seed,
            spec,
        )
        .map_err(dnam_err_to_py)
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(native_immutable_header_sha256, m)?)?;
    m.add_function(wrap_pyfunction!(native_sha256_hex, m)?)?;
    m.add_function(wrap_pyfunction!(native_hmac_sha256_hex, m)?)?;
    m.add_function(wrap_pyfunction!(native_ed25519_pubkey_hex, m)?)?;
    m.add_function(wrap_pyfunction!(native_ed25519_sign_hex, m)?)?;
    m.add_function(wrap_pyfunction!(native_x25519_pubkey_hex, m)?)?;
    m.add_function(wrap_pyfunction!(native_build_creator_signing_message, m)?)?;
    m.add_function(wrap_pyfunction!(
        native_build_marketplace_signing_message,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(native_sign_creator_header, m)?)?;
    m.add_function(wrap_pyfunction!(native_sign_marketplace_receipt, m)?)?;
    m.add_function(wrap_pyfunction!(native_sign_oem_device_cert, m)?)?;
    m.add_function(wrap_pyfunction!(native_verify_oem_device_cert, m)?)?;
    m.add_function(wrap_pyfunction!(
        native_build_oem_cert_v2_signing_message,
        m
    )?)?;
    Ok(())
}
