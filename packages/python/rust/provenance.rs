//! Provenance, RFC 8785 JCS SHA-256, `Ed25519` signing messages, `ChaCha20` 64B cache-line
//! arena encryption, Dual-Mechanism Buyer Watermarking (`Test 8H`), and 160-D structural hash FFI.

use dnam_engine::container::{seal_dnam_container, unseal_dnam_container};
use dnam_engine::provenance::{
    RGLRU_TOTAL_CHANNELS, RustCryptoProvider, STRUCTURAL_HASH_DIM, TARDOS_CONV_WEIGHTS,
    apply_isometric_gauge_watermark, apply_tardos_watermark_4608, build_creator_signing_message,
    build_marketplace_receipt_signing_message, chacha20_xor_arena_blocks,
    compute_immutable_header_sha256, decode_hex_array, encode_hex, extract_structural_hash_160d,
    generate_tardos_code_4608, score_tardos_watermark_4608, sha256_digest,
    structural_cosine_similarity, verify_oem_device_certificate,
};
use dnam_engine::schema::{DnamJsonHeader, MarketplaceReceipt, OemDeviceCertificate};
use numpy::{PyArray1, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::dnam_err_to_py;

/// Computes the preset-safe RFC 8785 JCS `immutable_header_sha256` hex digest over `header_json`.
#[pyfunction]
pub fn native_immutable_header_sha256(header_json: &str) -> PyResult<String> {
    let header: DnamJsonHeader = serde_json::from_str(header_json)
        .map_err(|e| PyValueError::new_err(format!("[DNAM-VALIDATION] {e}")))?;
    let digest = compute_immutable_header_sha256(&header).map_err(dnam_err_to_py)?;
    Ok(encode_hex(&digest))
}

/// Computes the lowercase hex SHA-256 digest of `data`.
#[pyfunction]
pub fn native_sha256_hex(data: &[u8]) -> String {
    let digest = sha256_digest(data);
    encode_hex(&digest)
}

/// Builds the canonical `Ed25519` creator signing message for `(immutable_header_sha256_hex, master_arena_sha256_hex)`.
#[pyfunction]
pub fn native_build_creator_signing_message<'py>(
    py: Python<'py>,
    immutable_header_sha256_hex: &str,
    master_arena_sha256_hex: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let hdr_sha = decode_hex_array::<32>(immutable_header_sha256_hex).map_err(dnam_err_to_py)?;
    let arena_sha = decode_hex_array::<32>(master_arena_sha256_hex).map_err(dnam_err_to_py)?;
    let msg = build_creator_signing_message(&hdr_sha, &arena_sha);
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
    let hdr_sha = decode_hex_array::<32>(immutable_header_sha256_hex).map_err(dnam_err_to_py)?;
    let arena_sha = decode_hex_array::<32>(master_arena_sha256_hex).map_err(dnam_err_to_py)?;
    let receipt: MarketplaceReceipt = serde_json::from_str(receipt_json)
        .map_err(|e| PyValueError::new_err(format!("[DNAM-VALIDATION] {e}")))?;
    let msg = build_marketplace_receipt_signing_message(&hdr_sha, &arena_sha, &receipt)
        .map_err(dnam_err_to_py)?;
    Ok(PyBytes::new(py, &msg))
}

/// Applies RFC 8439 `ChaCha20` 64-byte cache-line XOR encryption/decryption over `arena_bytes`.
#[pyfunction]
#[pyo3(signature = (arena_bytes, key_32, nonce_12, start_block_counter = 0))]
pub fn native_chacha20_xor_arena<'py>(
    py: Python<'py>,
    arena_bytes: &[u8],
    key_32: &[u8],
    nonce_12: &[u8],
    start_block_counter: u32,
) -> PyResult<Bound<'py, PyBytes>> {
    let key: &[u8; 32] = key_32.try_into().map_err(|_| {
        PyValueError::new_err(format!(
            "[DNAM-VALIDATION] key_32 must be 32 bytes, got {}",
            key_32.len()
        ))
    })?;
    let nonce: &[u8; 12] = nonce_12.try_into().map_err(|_| {
        PyValueError::new_err(format!(
            "[DNAM-VALIDATION] nonce_12 must be 12 bytes, got {}",
            nonce_12.len()
        ))
    })?;
    let mut buf = arena_bytes.to_vec();
    chacha20_xor_arena_blocks(&mut buf, key, nonce, start_block_counter).map_err(dnam_err_to_py)?;
    Ok(PyBytes::new(py, &buf))
}

/// Applies Mechanism A (48-channel RG-LRU reciprocal scaling + SO(4) SVD head rotation) to `arena_bytes`.
#[pyfunction]
pub fn native_apply_isometric_gauge_watermark<'py>(
    py: Python<'py>,
    arena_bytes: &[u8],
    kappa_48: PyReadonlyArray1<'py, f32>,
    q_buyer_flat_16: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyBytes>> {
    if kappa_48.len() != RGLRU_TOTAL_CHANNELS {
        return Err(PyValueError::new_err(format!(
            "[DNAM-VALIDATION] kappa_48 must have {RGLRU_TOTAL_CHANNELS} elements, got {}",
            kappa_48.len()
        )));
    }
    if q_buyer_flat_16.len() != 16 {
        return Err(PyValueError::new_err(format!(
            "[DNAM-VALIDATION] q_buyer_4x4 must have 16 elements, got {}",
            q_buyer_flat_16.len()
        )));
    }
    let k_slice = kappa_48
        .as_slice()
        .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] kappa_48 must be contiguous"))?;
    let q_slice = q_buyer_flat_16
        .as_slice()
        .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] q_buyer_4x4 must be contiguous"))?;

    let mut kappa_arr = [1.0_f32; RGLRU_TOTAL_CHANNELS];
    kappa_arr.copy_from_slice(k_slice);
    let mut q_arr = [[0.0_f32; 4]; 4];
    for r in 0..4 {
        q_arr[r].copy_from_slice(&q_slice[r * 4..(r + 1) * 4]);
    }

    let mut buf = arena_bytes.to_vec();
    apply_isometric_gauge_watermark(&mut buf, &kappa_arr, &q_arr).map_err(dnam_err_to_py)?;
    Ok(PyBytes::new(py, &buf))
}

/// Generates a 4,608-chip Tardos binary spread-spectrum watermark code `{-1.0, +1.0}^4608`.
#[pyfunction]
pub fn native_generate_tardos_code<'py>(
    py: Python<'py>,
    marketplace_secret_32: &[u8],
    model_id: &str,
    license_id: &str,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let secret: &[u8; 32] = marketplace_secret_32.try_into().map_err(|_| {
        PyValueError::new_err(format!(
            "[DNAM-VALIDATION] marketplace_secret must be 32 bytes, got {}",
            marketplace_secret_32.len()
        ))
    })?;
    let code = generate_tardos_code_4608(secret, model_id, license_id);
    Ok(PyArray1::from_slice(py, &code))
}

/// Embeds Mechanism B (4,608-weight Tardos spread-spectrum watermark) into `arena_bytes`.
#[pyfunction]
pub fn native_apply_tardos_watermark<'py>(
    py: Python<'py>,
    arena_bytes: &[u8],
    code_4608: PyReadonlyArray1<'py, f32>,
    epsilon: f32,
) -> PyResult<Bound<'py, PyBytes>> {
    if code_4608.len() != TARDOS_CONV_WEIGHTS {
        return Err(PyValueError::new_err(format!(
            "[DNAM-VALIDATION] Tardos code must have {TARDOS_CONV_WEIGHTS} elements, got {}",
            code_4608.len()
        )));
    }
    let c_slice = code_4608
        .as_slice()
        .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] code_4608 must be contiguous"))?;
    let mut code_arr = [1.0_f32; TARDOS_CONV_WEIGHTS];
    code_arr.copy_from_slice(c_slice);

    let mut buf = arena_bytes.to_vec();
    apply_tardos_watermark_4608(&mut buf, &code_arr, epsilon).map_err(dnam_err_to_py)?;
    Ok(PyBytes::new(py, &buf))
}

/// Computes the Tardos matched-filter Z-score (`sigma`) of `suspect_arena` against `master_arena` and `code_4608`.
#[pyfunction]
pub fn native_score_tardos_watermark<'py>(
    master_arena: &[u8],
    suspect_arena: &[u8],
    code_4608: PyReadonlyArray1<'py, f32>,
) -> PyResult<f64> {
    if code_4608.len() != TARDOS_CONV_WEIGHTS {
        return Err(PyValueError::new_err(format!(
            "[DNAM-VALIDATION] Tardos code must have {TARDOS_CONV_WEIGHTS} elements, got {}",
            code_4608.len()
        )));
    }
    let c_slice = code_4608
        .as_slice()
        .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] code_4608 must be contiguous"))?;
    let mut code_arr = [1.0_f32; TARDOS_CONV_WEIGHTS];
    code_arr.copy_from_slice(c_slice);

    score_tardos_watermark_4608(master_arena, suspect_arena, &code_arr).map_err(dnam_err_to_py)
}

/// Extracts the 160-Dimensional Gauge-Invariant Structural Hash vector `f_struct in R^160` from `arena_bytes`.
#[pyfunction]
pub fn native_extract_structural_hash<'py>(
    py: Python<'py>,
    arena_bytes: &[u8],
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let vec160 = extract_structural_hash_160d(arena_bytes).map_err(dnam_err_to_py)?;
    Ok(PyArray1::from_slice(py, &vec160))
}

/// Computes the cosine similarity `cos(a, b) in [-1, +1]` between two 160-D structural hash vectors.
#[pyfunction]
pub fn native_structural_cosine_similarity<'py>(
    a: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
) -> PyResult<f64> {
    if a.len() != STRUCTURAL_HASH_DIM || b.len() != STRUCTURAL_HASH_DIM {
        return Err(PyValueError::new_err(format!(
            "[DNAM-VALIDATION] Structural hash vectors must have length {STRUCTURAL_HASH_DIM}"
        )));
    }
    let a_slice = a
        .as_slice()
        .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] vector a must be contiguous"))?;
    let b_slice = b
        .as_slice()
        .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] vector b must be contiguous"))?;
    let mut a_arr = [0.0_f64; STRUCTURAL_HASH_DIM];
    let mut b_arr = [0.0_f64; STRUCTURAL_HASH_DIM];
    a_arr.copy_from_slice(a_slice);
    b_arr.copy_from_slice(b_slice);
    Ok(structural_cosine_similarity(&a_arr, &b_arr))
}

/// Derives the 64-character lowercase hex `Ed25519` public key from a 32-byte secret key seed.
#[pyfunction]
pub fn native_ed25519_pubkey_hex(seed_32: &[u8]) -> PyResult<String> {
    let seed: &[u8; 32] = seed_32.try_into().map_err(|_| {
        PyValueError::new_err(format!(
            "[DNAM-VALIDATION] Ed25519 seed_32 must be 32 bytes, got {}",
            seed_32.len()
        ))
    })?;
    let pk = RustCryptoProvider::ed25519_pubkey_from_seed(seed);
    Ok(encode_hex(&pk))
}

/// Signs `message` using a 32-byte `Ed25519` secret key seed, returning the 128-character hex signature.
#[pyfunction]
pub fn native_ed25519_sign_hex(seed_32: &[u8], message: &[u8]) -> PyResult<String> {
    let seed: &[u8; 32] = seed_32.try_into().map_err(|_| {
        PyValueError::new_err(format!(
            "[DNAM-VALIDATION] Ed25519 seed_32 must be 32 bytes, got {}",
            seed_32.len()
        ))
    })?;
    let sig = RustCryptoProvider::ed25519_sign(seed, message);
    Ok(encode_hex(&sig))
}

/// Derives the 64-character lowercase hex `X25519` public key from a 32-byte secret key.
#[pyfunction]
pub fn native_x25519_pubkey_hex(secret_32: &[u8]) -> PyResult<String> {
    let secret: &[u8; 32] = secret_32.try_into().map_err(|_| {
        PyValueError::new_err(format!(
            "[DNAM-VALIDATION] X25519 secret_32 must be 32 bytes, got {}",
            secret_32.len()
        ))
    })?;
    let pk = RustCryptoProvider::x25519_pubkey_from_secret(secret);
    Ok(encode_hex(&pk))
}

/// Wraps a 32-byte Content Encryption Key (`cek_32`) for an OEM device (`X25519` + `HKDF-SHA256` + `ChaCha20`),
/// returning the 64-character lowercase hex `wrapped_cek`.
#[pyfunction]
pub fn native_wrap_device_cek_hex(
    ephemeral_secret_32: &[u8],
    device_pubkey_32: &[u8],
    device_id_hash: &str,
    cek_32: &[u8],
) -> PyResult<String> {
    let eph: &[u8; 32] = ephemeral_secret_32.try_into().map_err(|_| {
        PyValueError::new_err(format!(
            "[DNAM-VALIDATION] ephemeral_secret_32 must be 32 bytes, got {}",
            ephemeral_secret_32.len()
        ))
    })?;
    let dev_pk: &[u8; 32] = device_pubkey_32.try_into().map_err(|_| {
        PyValueError::new_err(format!(
            "[DNAM-VALIDATION] device_pubkey_32 must be 32 bytes, got {}",
            device_pubkey_32.len()
        ))
    })?;
    let cek: &[u8; 32] = cek_32.try_into().map_err(|_| {
        PyValueError::new_err(format!(
            "[DNAM-VALIDATION] cek_32 must be 32 bytes, got {}",
            cek_32.len()
        ))
    })?;
    let wrapped = RustCryptoProvider::wrap_device_cek(eph, dev_pk, device_id_hash, cek)
        .map_err(dnam_err_to_py)?;
    Ok(encode_hex(&wrapped))
}

/// Signs an `OemDeviceCertificate` using the OEM vendor's 32-byte `Ed25519` factory seed,
/// returning the serialized JSON string of `OemDeviceCertificate`.
#[pyfunction]
pub fn native_sign_oem_device_cert(
    oem_seed_32: &[u8],
    oem_vendor: &str,
    model_code: &str,
    device_id_hash: &str,
    pubkey_x25519_32: &[u8],
) -> PyResult<String> {
    let seed: &[u8; 32] = oem_seed_32.try_into().map_err(|_| {
        PyValueError::new_err(format!(
            "[DNAM-VALIDATION] oem_seed_32 must be 32 bytes, got {}",
            oem_seed_32.len()
        ))
    })?;
    let pk_x25519: &[u8; 32] = pubkey_x25519_32.try_into().map_err(|_| {
        PyValueError::new_err(format!(
            "[DNAM-VALIDATION] pubkey_x25519_32 must be 32 bytes, got {}",
            pubkey_x25519_32.len()
        ))
    })?;
    let cert = RustCryptoProvider::sign_oem_device_certificate(
        seed,
        oem_vendor,
        model_code,
        device_id_hash,
        pk_x25519,
    )
    .map_err(dnam_err_to_py)?;
    serde_json::to_string(&cert)
        .map_err(|e| PyValueError::new_err(format!("[DNAM-VALIDATION] {e}")))
}

/// Verifies an `OemDeviceCertificate` JSON against a trusted 64-character hex OEM `Ed25519` public key,
/// returning the verified 64-character lowercase hex `pubkey_x25519`.
#[pyfunction]
pub fn native_verify_oem_device_cert(
    cert_json: &str,
    trusted_oem_pubkey_hex: &str,
) -> PyResult<String> {
    let cert: OemDeviceCertificate = serde_json::from_str(cert_json)
        .map_err(|e| PyValueError::new_err(format!("[DNAM-VALIDATION] {e}")))?;
    let oem_pk = decode_hex_array::<32>(trusted_oem_pubkey_hex).map_err(dnam_err_to_py)?;
    let crypto = RustCryptoProvider::default();
    let pk_x25519 =
        verify_oem_device_certificate(&cert, &oem_pk, &crypto).map_err(dnam_err_to_py)?;
    Ok(encode_hex(&pk_x25519))
}

/// Seals an inner `.dnam` container into a 64-byte aligned `.dnam-seal` (`b"DNSL"`) hardware-bound envelope.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn native_seal_container<'py>(
    py: Python<'py>,
    inner_dnam_bytes: &[u8],
    recipient_certs_json: &str,
    trusted_oem_pubkeys_json: &str,
    marketplace_seed_32: &[u8],
    ephemeral_secret_32: &[u8],
    cek_32: &[u8],
    nonce_12: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    let certs: Vec<OemDeviceCertificate> = serde_json::from_str(recipient_certs_json)
        .map_err(|e| PyValueError::new_err(format!("[DNAM-VALIDATION] {e}")))?;
    let oem_map: std::collections::BTreeMap<String, String> =
        serde_json::from_str(trusted_oem_pubkeys_json)
            .map_err(|e| PyValueError::new_err(format!("[DNAM-VALIDATION] {e}")))?;
    let mut trusted_owned: Vec<(String, [u8; 32])> = Vec::with_capacity(oem_map.len());
    for (vendor, pk_hex) in oem_map {
        let pk = decode_hex_array::<32>(&pk_hex).map_err(dnam_err_to_py)?;
        trusted_owned.push((vendor, pk));
    }
    let trusted_refs: Vec<(&str, [u8; 32])> = trusted_owned
        .iter()
        .map(|(v, pk)| (v.as_str(), *pk))
        .collect();

    let mkt_seed: &[u8; 32] = marketplace_seed_32.try_into().map_err(|_| {
        PyValueError::new_err(format!(
            "[DNAM-VALIDATION] marketplace_seed_32 must be 32 bytes, got {}",
            marketplace_seed_32.len()
        ))
    })?;
    let eph_sec: &[u8; 32] = ephemeral_secret_32.try_into().map_err(|_| {
        PyValueError::new_err(format!(
            "[DNAM-VALIDATION] ephemeral_secret_32 must be 32 bytes, got {}",
            ephemeral_secret_32.len()
        ))
    })?;
    let cek: &[u8; 32] = cek_32.try_into().map_err(|_| {
        PyValueError::new_err(format!(
            "[DNAM-VALIDATION] cek_32 must be 32 bytes, got {}",
            cek_32.len()
        ))
    })?;
    let nonce: &[u8; 12] = nonce_12.try_into().map_err(|_| {
        PyValueError::new_err(format!(
            "[DNAM-VALIDATION] nonce_12 must be 12 bytes, got {}",
            nonce_12.len()
        ))
    })?;

    let sealed = seal_dnam_container(
        inner_dnam_bytes,
        &certs,
        &trusted_refs,
        mkt_seed,
        eph_sec,
        cek,
        nonce,
    )
    .map_err(dnam_err_to_py)?;
    Ok(PyBytes::new(py, &sealed))
}

/// Unseals a `.dnam-seal` (`b"DNSL"`) hardware-bound envelope using the device's `X25519` secret key,
/// returning the decrypted inner `.dnam` container bytes.
#[pyfunction]
#[pyo3(signature = (sealed_bytes, device_id_hash = None, device_secret_32 = None, trusted_marketplace_pubkey_hex = None))]
pub fn native_unseal_container<'py>(
    py: Python<'py>,
    sealed_bytes: &[u8],
    device_id_hash: Option<&str>,
    device_secret_32: Option<&[u8]>,
    trusted_marketplace_pubkey_hex: Option<&str>,
) -> PyResult<Bound<'py, PyBytes>> {
    let mut crypto = RustCryptoProvider::new();
    if let (Some(id), Some(sec)) = (device_id_hash, device_secret_32) {
        let sec_arr: &[u8; 32] = sec.try_into().map_err(|_| {
            PyValueError::new_err(format!(
                "[DNAM-VALIDATION] device_secret_32 must be 32 bytes, got {}",
                sec.len()
            ))
        })?;
        crypto = crypto.with_device_key(id, *sec_arr);
    }
    if let Some(mkt_pk_hex) = trusted_marketplace_pubkey_hex {
        let mkt_pk = decode_hex_array::<32>(mkt_pk_hex).map_err(dnam_err_to_py)?;
        crypto = crypto.with_marketplace_pubkey(mkt_pk);
    }
    let unsealed = unseal_dnam_container(sealed_bytes, &crypto).map_err(dnam_err_to_py)?;
    Ok(PyBytes::new(py, &unsealed))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(native_immutable_header_sha256, m)?)?;
    m.add_function(wrap_pyfunction!(native_sha256_hex, m)?)?;
    m.add_function(wrap_pyfunction!(native_build_creator_signing_message, m)?)?;
    m.add_function(wrap_pyfunction!(
        native_build_marketplace_signing_message,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(native_chacha20_xor_arena, m)?)?;
    m.add_function(wrap_pyfunction!(native_apply_isometric_gauge_watermark, m)?)?;
    m.add_function(wrap_pyfunction!(native_generate_tardos_code, m)?)?;
    m.add_function(wrap_pyfunction!(native_apply_tardos_watermark, m)?)?;
    m.add_function(wrap_pyfunction!(native_score_tardos_watermark, m)?)?;
    m.add_function(wrap_pyfunction!(native_extract_structural_hash, m)?)?;
    m.add_function(wrap_pyfunction!(native_structural_cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(native_ed25519_pubkey_hex, m)?)?;
    m.add_function(wrap_pyfunction!(native_ed25519_sign_hex, m)?)?;
    m.add_function(wrap_pyfunction!(native_x25519_pubkey_hex, m)?)?;
    m.add_function(wrap_pyfunction!(native_wrap_device_cek_hex, m)?)?;
    m.add_function(wrap_pyfunction!(native_sign_oem_device_cert, m)?)?;
    m.add_function(wrap_pyfunction!(native_verify_oem_device_cert, m)?)?;
    m.add_function(wrap_pyfunction!(native_seal_container, m)?)?;
    m.add_function(wrap_pyfunction!(native_unseal_container, m)?)?;
    Ok(())
}
