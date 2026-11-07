//! Marketplace v2 envelopes (`DAU1`, `DME1`, HPKE slots), prefix commitments, royalty splits,
//! and preview containment (`ref[sdk.python.provenance.v2-security]`).

use std::collections::BTreeSet;

use dnam_engine::provenance::{
    Dau1Envelope, build_dme1_header, compute_preview_single_control_window, decode_hex_array,
    encode_hex, encode_hpke_slot_vector, encode_recipient_key_ids, encode_record_directory,
    parse_dme1_decrypted_plaintext, parse_hpke_slot_vector, parse_recipient_key_ids,
    parse_record_directory, validate_license_tier_and_royalty_splits,
    validate_preview_single_switch_options,
};
use dnam_engine::schema::RoyaltySplit;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::{dnam_err_to_py, validation_err};

fn decode_hex_prefixes(hex_list: &[String]) -> PyResult<Vec<[u8; 32]>> {
    let mut out = Vec::with_capacity(hex_list.len());
    for h in hex_list {
        out.push(decode_hex_array::<32>(h).map_err(dnam_err_to_py)?);
    }
    Ok(out)
}

/// Serializes a `DAU1` authentication envelope into wire bytes (`ref[sec.sig.dau1-envelope]`).
#[pyfunction]
#[pyo3(signature = (auth_kind, creator_pubkey_hex, creator_max_tier, master_prefixes_hex, creator_signature_hex, marketplace_pubkey_hex = None, policy_bytes = None, buyer_max_tier = None, buyer_prefixes_hex = Vec::new(), marketplace_signature_hex = None))]
#[allow(clippy::too_many_arguments)]
pub fn native_encode_dau1_envelope<'py>(
    py: Python<'py>,
    auth_kind: u8,
    creator_pubkey_hex: &str,
    creator_max_tier: u8,
    master_prefixes_hex: Vec<String>,
    creator_signature_hex: &str,
    marketplace_pubkey_hex: Option<&str>,
    policy_bytes: Option<&[u8]>,
    buyer_max_tier: Option<u8>,
    buyer_prefixes_hex: Vec<String>,
    marketplace_signature_hex: Option<&str>,
) -> PyResult<Bound<'py, PyBytes>> {
    let wire = py.detach(|| -> PyResult<Vec<u8>> {
        let creator_pubkey = decode_hex_array::<32>(creator_pubkey_hex).map_err(dnam_err_to_py)?;
        let master_prefixes = decode_hex_prefixes(&master_prefixes_hex)?;
        let creator_signature =
            decode_hex_array::<64>(creator_signature_hex).map_err(dnam_err_to_py)?;
        let marketplace_pubkey = match marketplace_pubkey_hex {
            Some(h) => Some(decode_hex_array::<32>(h).map_err(dnam_err_to_py)?),
            None => None,
        };
        let buyer_prefixes = decode_hex_prefixes(&buyer_prefixes_hex)?;
        let marketplace_signature = match marketplace_signature_hex {
            Some(h) => Some(decode_hex_array::<64>(h).map_err(dnam_err_to_py)?),
            None => None,
        };
        let env = Dau1Envelope {
            auth_kind,
            creator_pubkey,
            creator_max_tier,
            master_prefixes,
            creator_signature,
            marketplace_pubkey,
            policy_bytes: policy_bytes.unwrap_or(&[]).to_vec(),
            buyer_max_tier,
            buyer_prefixes,
            marketplace_signature,
        };
        env.to_bytes().map_err(dnam_err_to_py)
    })?;
    Ok(PyBytes::new(py, &wire))
}

/// Parses and validates a `DAU1` authentication envelope from wire bytes.
#[pyfunction]
pub fn native_decode_dau1_envelope<'py>(
    py: Python<'py>,
    wire_bytes: &[u8],
) -> PyResult<Bound<'py, PyDict>> {
    let env = py
        .detach(|| Dau1Envelope::from_bytes(wire_bytes))
        .map_err(dnam_err_to_py)?;
    let dict = PyDict::new(py);
    dict.set_item("auth_kind", env.auth_kind)?;
    dict.set_item("creator_pubkey_hex", encode_hex(&env.creator_pubkey))?;
    dict.set_item("creator_max_tier", env.creator_max_tier)?;
    let master_hex: Vec<String> = env.master_prefixes.iter().map(|d| encode_hex(d)).collect();
    dict.set_item("master_prefixes_hex", master_hex)?;
    dict.set_item("creator_signature_hex", encode_hex(&env.creator_signature))?;
    dict.set_item(
        "marketplace_pubkey_hex",
        env.marketplace_pubkey.as_ref().map(|d| encode_hex(d)),
    )?;
    dict.set_item("policy_bytes", PyBytes::new(py, &env.policy_bytes))?;
    dict.set_item("buyer_max_tier", env.buyer_max_tier)?;
    let buyer_hex: Vec<String> = env.buyer_prefixes.iter().map(|d| encode_hex(d)).collect();
    dict.set_item("buyer_prefixes_hex", buyer_hex)?;
    dict.set_item(
        "marketplace_signature_hex",
        env.marketplace_signature.as_ref().map(|d| encode_hex(d)),
    )?;
    Ok(dict)
}

/// Builds a 20-byte `DME1` encrypted mutable frame header (`ref[sec.drm.dme1-mutable-suffix]`).
#[pyfunction]
pub fn native_build_dme1_header<'py>(
    py: Python<'py>,
    profile: u8,
    recipient_count: u32,
    ciphertext_length: u64,
) -> PyResult<Bound<'py, PyBytes>> {
    let hdr = py
        .detach(|| build_dme1_header(profile, recipient_count, ciphertext_length))
        .map_err(dnam_err_to_py)?;
    Ok(PyBytes::new(py, &hdr))
}

/// Validates and parses decrypted `DME1` plaintext against recipient SHA-256 key digests (hex).
#[pyfunction]
pub fn native_parse_dme1_plaintext(
    py: Python<'_>,
    plaintext: &[u8],
    recipient_key_digests_hex: Vec<String>,
) -> PyResult<String> {
    py.detach(|| {
        let digests = decode_hex_prefixes(&recipient_key_digests_hex)?;
        let val = parse_dme1_decrypted_plaintext(plaintext, &digests).map_err(dnam_err_to_py)?;
        serde_json::to_string(&val).map_err(validation_err)
    })
}

/// Encodes HPKE recipient key SHA-256 digests (`R = u32le(N) || digest[0..N]`).
#[pyfunction]
pub fn native_encode_recipient_key_ids<'py>(
    py: Python<'py>,
    key_digests_hex: Vec<String>,
) -> PyResult<Bound<'py, PyBytes>> {
    let out = py.detach(|| -> PyResult<Vec<u8>> {
        let digests = decode_hex_prefixes(&key_digests_hex)?;
        encode_recipient_key_ids(&digests).map_err(dnam_err_to_py)
    })?;
    Ok(PyBytes::new(py, &out))
}

/// Parses HPKE recipient key SHA-256 digests (`R = u32le(N) || digest[0..N]`).
#[pyfunction]
#[pyo3(signature = (data, max_recipients = 16))]
pub fn native_parse_recipient_key_ids(
    py: Python<'_>,
    data: &[u8],
    max_recipients: usize,
) -> PyResult<Vec<String>> {
    py.detach(|| {
        let digests = parse_recipient_key_ids(data, max_recipients).map_err(dnam_err_to_py)?;
        Ok(digests.iter().map(|d| encode_hex(d)).collect())
    })
}

/// Encodes an HPKE slot vector (`S = u32le(N) || (enc[32] || ct_and_tag[48])[0..N]`).
#[pyfunction]
pub fn native_encode_hpke_slots<'py>(
    py: Python<'py>,
    slots_hex: Vec<(String, String)>,
) -> PyResult<Bound<'py, PyBytes>> {
    let out = py.detach(|| -> PyResult<Vec<u8>> {
        let mut slots = Vec::with_capacity(slots_hex.len());
        for (enc_hex, ct_hex) in slots_hex {
            let enc = decode_hex_array::<32>(&enc_hex).map_err(dnam_err_to_py)?;
            let ct = decode_hex_array::<48>(&ct_hex).map_err(dnam_err_to_py)?;
            slots.push((enc, ct));
        }
        encode_hpke_slot_vector(&slots).map_err(dnam_err_to_py)
    })?;
    Ok(PyBytes::new(py, &out))
}

/// Parses an HPKE slot vector (`S = u32le(N) || (enc[32] || ct_and_tag[48])[0..N]`).
#[pyfunction]
#[pyo3(signature = (data, expected_count, max_recipients = 16))]
pub fn native_parse_hpke_slots(
    py: Python<'_>,
    data: &[u8],
    expected_count: usize,
    max_recipients: usize,
) -> PyResult<Vec<(String, String)>> {
    py.detach(|| {
        let slots =
            parse_hpke_slot_vector(data, expected_count, max_recipients).map_err(dnam_err_to_py)?;
        Ok(slots
            .iter()
            .map(|(e, c)| (encode_hex(e), encode_hex(c)))
            .collect())
    })
}

/// Encodes a record directory (`D = u32le(C) || (u64le(offset) || u64le(length))[0..C]`).
#[pyfunction]
pub fn native_encode_record_directory<'py>(
    py: Python<'py>,
    maximum_tier: u8,
    records: Vec<(u64, u64)>,
) -> PyResult<Bound<'py, PyBytes>> {
    let out = py
        .detach(|| encode_record_directory(maximum_tier, &records))
        .map_err(dnam_err_to_py)?;
    Ok(PyBytes::new(py, &out))
}

/// Parses a record directory (`D = u32le(C) || (u64le(offset) || u64le(length))[0..C]`).
#[pyfunction]
pub fn native_parse_record_directory(
    py: Python<'_>,
    data: &[u8],
    maximum_tier: u8,
) -> PyResult<Vec<(u64, u64)>> {
    py.detach(|| parse_record_directory(data, maximum_tier))
        .map_err(dnam_err_to_py)
}

/// Computes the bounded single-control preview window `[start, end]` (`ref[sec.preview.containment]`).
#[pyfunction]
pub fn native_compute_preview_window(
    anchor: f64,
    delta: f64,
    safe_bounds: (f64, f64),
) -> PyResult<(f64, f64)> {
    compute_preview_single_control_window(anchor, delta, safe_bounds).map_err(dnam_err_to_py)
}

/// Validates that a single-switch preview exposes two distinct options that are observed, active, and safe.
#[pyfunction]
pub fn native_validate_preview_switch_options(
    py: Python<'_>,
    preset: &str,
    alternative: &str,
    observed: Vec<String>,
    active: Vec<String>,
    safe: Vec<String>,
) -> PyResult<(String, String)> {
    py.detach(|| {
        let obs_set: BTreeSet<&str> = observed.iter().map(String::as_str).collect();
        let act_set: BTreeSet<&str> = active.iter().map(String::as_str).collect();
        let safe_set: BTreeSet<&str> = safe.iter().map(String::as_str).collect();
        let (p, a) = validate_preview_single_switch_options(
            preset,
            alternative,
            &obs_set,
            &act_set,
            &safe_set,
        )
        .map_err(dnam_err_to_py)?;
        Ok((p.to_string(), a.to_string()))
    })
}

/// Validates `license_tier` and that `royalty_splits_json` sums to `10,000` bps (`100%`).
#[pyfunction]
pub fn native_validate_royalty_splits(
    py: Python<'_>,
    license_tier: &str,
    royalty_splits_json: &str,
) -> PyResult<()> {
    py.detach(|| {
        let splits: Vec<RoyaltySplit> =
            serde_json::from_str(royalty_splits_json).map_err(validation_err)?;
        validate_license_tier_and_royalty_splits(license_tier, &splits).map_err(dnam_err_to_py)
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(native_encode_dau1_envelope, m)?)?;
    m.add_function(wrap_pyfunction!(native_decode_dau1_envelope, m)?)?;
    m.add_function(wrap_pyfunction!(native_build_dme1_header, m)?)?;
    m.add_function(wrap_pyfunction!(native_parse_dme1_plaintext, m)?)?;
    m.add_function(wrap_pyfunction!(native_encode_recipient_key_ids, m)?)?;
    m.add_function(wrap_pyfunction!(native_parse_recipient_key_ids, m)?)?;
    m.add_function(wrap_pyfunction!(native_encode_hpke_slots, m)?)?;
    m.add_function(wrap_pyfunction!(native_parse_hpke_slots, m)?)?;
    m.add_function(wrap_pyfunction!(native_encode_record_directory, m)?)?;
    m.add_function(wrap_pyfunction!(native_parse_record_directory, m)?)?;
    m.add_function(wrap_pyfunction!(native_compute_preview_window, m)?)?;
    m.add_function(wrap_pyfunction!(native_validate_preview_switch_options, m)?)?;
    m.add_function(wrap_pyfunction!(native_validate_royalty_splits, m)?)?;
    Ok(())
}
