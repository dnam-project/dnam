//! `.dnam-seal` (`DNSL`) hardware envelope sealing/unsealing, `X25519` CEK wrapping,
//! and 64-byte cache-line `ChaCha20` arena encryption (`ref[sdk.python.provenance.v2-security]`).

use std::collections::BTreeMap;

use dnam_engine::container::{seal_dnam_container, unseal_dnam_container};
use dnam_engine::provenance::{
    RustCryptoProvider, chacha20_xor_arena_blocks, decode_hex_array, encode_hex,
};
use dnam_engine::schema::OemDeviceCertificate;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::{dnam_err_to_py, validation_err};

fn check_len<const N: usize>(bytes: &[u8], name: &str) -> PyResult<[u8; N]> {
    bytes
        .try_into()
        .map_err(|_| validation_err(format!("{name} must be {N} bytes, got {}", bytes.len())))
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
    let key = check_len::<32>(key_32, "key_32")?;
    let nonce = check_len::<12>(nonce_12, "nonce_12")?;
    let mut buf = arena_bytes.to_vec();
    chacha20_xor_arena_blocks(&mut buf, &key, &nonce, start_block_counter)
        .map_err(dnam_err_to_py)?;
    Ok(PyBytes::new(py, &buf))
}

/// Wraps a 32-byte Content Encryption Key (`cek_32`) for an OEM device (`X25519` + `HKDF-SHA256` + `ChaCha20`),
/// returning the lowercase hex `wrapped_cek`.
#[pyfunction]
pub fn native_wrap_device_cek_hex(
    ephemeral_secret_32: &[u8],
    device_pubkey_32: &[u8],
    device_id_hash: &str,
    cek_32: &[u8],
) -> PyResult<String> {
    let eph = check_len::<32>(ephemeral_secret_32, "ephemeral_secret_32")?;
    let dev_pk = check_len::<32>(device_pubkey_32, "device_pubkey_32")?;
    let cek = check_len::<32>(cek_32, "cek_32")?;
    let wrapped = RustCryptoProvider::wrap_device_cek(&eph, &dev_pk, device_id_hash, &cek)
        .map_err(dnam_err_to_py)?;
    Ok(encode_hex(&wrapped))
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
    let certs: Vec<OemDeviceCertificate> =
        serde_json::from_str(recipient_certs_json).map_err(validation_err)?;
    let oem_map: BTreeMap<String, String> =
        serde_json::from_str(trusted_oem_pubkeys_json).map_err(validation_err)?;
    let mut trusted_owned: Vec<(String, [u8; 32])> = Vec::with_capacity(oem_map.len());
    for (vendor, pk_hex) in oem_map {
        let pk = decode_hex_array::<32>(&pk_hex).map_err(dnam_err_to_py)?;
        trusted_owned.push((vendor, pk));
    }
    let trusted_refs: Vec<(&str, [u8; 32])> = trusted_owned
        .iter()
        .map(|(v, pk)| (v.as_str(), *pk))
        .collect();

    let mkt_seed = check_len::<32>(marketplace_seed_32, "marketplace_seed_32")?;
    let eph_sec = check_len::<32>(ephemeral_secret_32, "ephemeral_secret_32")?;
    let cek = check_len::<32>(cek_32, "cek_32")?;
    let nonce = check_len::<12>(nonce_12, "nonce_12")?;

    let sealed = seal_dnam_container(
        inner_dnam_bytes,
        &certs,
        &trusted_refs,
        &mkt_seed,
        &eph_sec,
        &cek,
        &nonce,
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
        let sec_arr = check_len::<32>(sec, "device_secret_32")?;
        crypto = crypto.with_device_key(id, sec_arr);
    }
    if let Some(mkt_pk_hex) = trusted_marketplace_pubkey_hex {
        let mkt_pk = decode_hex_array::<32>(mkt_pk_hex).map_err(dnam_err_to_py)?;
        crypto = crypto.with_marketplace_pubkey(mkt_pk);
    }
    let unsealed = unseal_dnam_container(sealed_bytes, &crypto).map_err(dnam_err_to_py)?;
    Ok(PyBytes::new(py, &unsealed))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(native_chacha20_xor_arena, m)?)?;
    m.add_function(wrap_pyfunction!(native_wrap_device_cek_hex, m)?)?;
    m.add_function(wrap_pyfunction!(native_seal_container, m)?)?;
    m.add_function(wrap_pyfunction!(native_unseal_container, m)?)?;
    Ok(())
}
