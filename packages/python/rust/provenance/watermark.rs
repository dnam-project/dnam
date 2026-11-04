//! Dual-Mechanism Buyer Watermarking (Isometric Gauge + Coupled Tardos-7680), 224-D Structural
//! Hash, Stage III Tier-Local Fingerprint, and HMAC-SHA256 Seed Derivation
//! (`ref[sdk.python.provenance.v2-security]`).

use dnam_engine::provenance::{
    KAUTZ_TOTAL_CHANNELS, STRUCTURAL_HASH_DIM, STRUCTURAL_HASH_TIER_DIM, TARDOS_CASCADE_WEIGHTS,
    WatermarkSeedScope, apply_isometric_gauge_watermark, apply_tardos_watermark_7680,
    derive_watermark_seed_hmac_sha256, encode_hex, extract_stage3_tier_fingerprint_component,
    extract_structural_hash_224d, generate_tardos_code_7680, score_tardos_watermark_7680,
    structural_cosine_similarity,
};
use numpy::{PyArray1, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::arrays::slice_1d;
use crate::{dnam_err_to_py, validation_err};

fn check_len<const N: usize>(bytes: &[u8], name: &str) -> PyResult<[u8; N]> {
    bytes
        .try_into()
        .map_err(|_| validation_err(format!("{name} must be {N} bytes, got {}", bytes.len())))
}

/// Applies Mechanism A (64-channel modal/Kautz reciprocal scaling + SO(4) SVD head rotation) to `arena_bytes`.
#[pyfunction]
pub fn native_apply_isometric_gauge_watermark<'py>(
    py: Python<'py>,
    arena_bytes: &[u8],
    kappa_64: PyReadonlyArray1<'py, f32>,
    q_buyer_flat_16: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyBytes>> {
    if kappa_64.len() != KAUTZ_TOTAL_CHANNELS {
        return Err(validation_err(format!(
            "kappa_64 must have {KAUTZ_TOTAL_CHANNELS} elements, got {}",
            kappa_64.len()
        )));
    }
    if q_buyer_flat_16.len() != 16 {
        return Err(validation_err(format!(
            "q_buyer_4x4 must have 16 elements, got {}",
            q_buyer_flat_16.len()
        )));
    }
    let k_slice = slice_1d(&kappa_64, "kappa_64")?;
    let q_slice = slice_1d(&q_buyer_flat_16, "q_buyer_4x4")?;

    let mut kappa_arr = [1.0_f32; KAUTZ_TOTAL_CHANNELS];
    kappa_arr.copy_from_slice(k_slice);
    let mut q_arr = [[0.0_f32; 4]; 4];
    for r in 0..4 {
        q_arr[r].copy_from_slice(&q_slice[r * 4..(r + 1) * 4]);
    }

    let mut buf = arena_bytes.to_vec();
    apply_isometric_gauge_watermark(&mut buf, &kappa_arr, &q_arr).map_err(dnam_err_to_py)?;
    Ok(PyBytes::new(py, &buf))
}

/// Generates a 7,680-chip Tardos binary spread-spectrum watermark code `{-1.0, +1.0}^7680`.
#[pyfunction]
pub fn native_generate_tardos_code<'py>(
    py: Python<'py>,
    marketplace_secret_32: &[u8],
    model_id: &str,
    license_id: &str,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let secret = check_len::<32>(marketplace_secret_32, "marketplace_secret")?;
    let code = generate_tardos_code_7680(&secret, model_id, license_id);
    Ok(PyArray1::from_slice(py, &code))
}

/// Embeds Mechanism B (7,680-weight Tardos spread-spectrum watermark) into `arena_bytes`.
#[pyfunction]
pub fn native_apply_tardos_watermark<'py>(
    py: Python<'py>,
    arena_bytes: &[u8],
    code_7680: PyReadonlyArray1<'py, f32>,
    epsilon: f32,
) -> PyResult<Bound<'py, PyBytes>> {
    if code_7680.len() != TARDOS_CASCADE_WEIGHTS {
        return Err(validation_err(format!(
            "Tardos code must have {TARDOS_CASCADE_WEIGHTS} elements, got {}",
            code_7680.len()
        )));
    }
    let c_slice = slice_1d(&code_7680, "code_7680")?;
    let mut code_arr = [1.0_f32; TARDOS_CASCADE_WEIGHTS];
    code_arr.copy_from_slice(c_slice);

    let mut buf = arena_bytes.to_vec();
    apply_tardos_watermark_7680(&mut buf, &code_arr, epsilon).map_err(dnam_err_to_py)?;
    Ok(PyBytes::new(py, &buf))
}

/// Computes the Tardos matched-filter Z-score (`sigma`) of `suspect_arena` against `master_arena` and `code_7680`.
#[pyfunction]
pub fn native_score_tardos_watermark<'py>(
    master_arena: &[u8],
    suspect_arena: &[u8],
    code_7680: PyReadonlyArray1<'py, f32>,
) -> PyResult<f64> {
    if code_7680.len() != TARDOS_CASCADE_WEIGHTS {
        return Err(validation_err(format!(
            "Tardos code must have {TARDOS_CASCADE_WEIGHTS} elements, got {}",
            code_7680.len()
        )));
    }
    let c_slice = slice_1d(&code_7680, "code_7680")?;
    let mut code_arr = [1.0_f32; TARDOS_CASCADE_WEIGHTS];
    code_arr.copy_from_slice(c_slice);

    score_tardos_watermark_7680(master_arena, suspect_arena, &code_arr).map_err(dnam_err_to_py)
}

/// Extracts the prefix-composable `4 × 56-D = 224-D` Gauge-Invariant Structural Hash vector from `arena_bytes`.
#[pyfunction]
pub fn native_extract_structural_hash<'py>(
    py: Python<'py>,
    arena_bytes: &[u8],
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let vec224 = extract_structural_hash_224d(arena_bytes).map_err(dnam_err_to_py)?;
    let tiers = if arena_bytes.len() >= 64_000 {
        4
    } else if arena_bytes.len() >= 36_224 {
        3
    } else if arena_bytes.len() >= 22_336 {
        2
    } else {
        1
    };
    Ok(PyArray1::from_slice(
        py,
        &vec224[..tiers * STRUCTURAL_HASH_TIER_DIM],
    ))
}

/// Computes the cosine similarity `cos(a, b) in [-1, +1]` between two structural hash vectors.
#[pyfunction]
pub fn native_structural_cosine_similarity<'py>(
    a: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
) -> PyResult<f64> {
    let min_len = a.len().min(b.len());
    if min_len == 0
        || min_len > STRUCTURAL_HASH_DIM
        || !min_len.is_multiple_of(STRUCTURAL_HASH_TIER_DIM)
    {
        return Err(validation_err(format!(
            "Structural hash vectors must have prefix length 56, 112, 168, or {STRUCTURAL_HASH_DIM} (got {} and {})",
            a.len(),
            b.len()
        )));
    }
    let a_slice = slice_1d(&a, "a")?;
    let b_slice = slice_1d(&b, "b")?;
    let mut a_arr = [0.0_f64; STRUCTURAL_HASH_DIM];
    let mut b_arr = [0.0_f64; STRUCTURAL_HASH_DIM];
    a_arr[..min_len].copy_from_slice(&a_slice[..min_len]);
    b_arr[..min_len].copy_from_slice(&b_slice[..min_len]);
    Ok(structural_cosine_similarity(&a_arr, &b_arr))
}

/// Derives the license-scoped or shared HMAC-SHA256 watermark seed (64-char hex).
#[pyfunction]
#[pyo3(signature = (key_32, marketplace_pubkey_32, watermark_key_id_16, creator_signed_manifest_sha256_32, model_id_16, watermark_profile_sha256_32, owning_tier, license_id_16 = None, buyer_id_16 = None))]
#[allow(clippy::too_many_arguments)]
pub fn native_derive_watermark_seed_hex(
    key_32: &[u8],
    marketplace_pubkey_32: &[u8],
    watermark_key_id_16: &[u8],
    creator_signed_manifest_sha256_32: &[u8],
    model_id_16: &[u8],
    watermark_profile_sha256_32: &[u8],
    owning_tier: u8,
    license_id_16: Option<&[u8]>,
    buyer_id_16: Option<&[u8]>,
) -> PyResult<String> {
    let key = check_len::<32>(key_32, "key_32")?;
    let scope = WatermarkSeedScope {
        marketplace_pubkey: check_len::<32>(marketplace_pubkey_32, "marketplace_pubkey_32")?,
        watermark_key_id: check_len::<16>(watermark_key_id_16, "watermark_key_id_16")?,
        creator_signed_manifest_sha256: check_len::<32>(
            creator_signed_manifest_sha256_32,
            "creator_signed_manifest_sha256_32",
        )?,
        model_id: check_len::<16>(model_id_16, "model_id_16")?,
        watermark_profile_sha256: check_len::<32>(
            watermark_profile_sha256_32,
            "watermark_profile_sha256_32",
        )?,
        owning_tier,
    };
    let buyer_pair = match (license_id_16, buyer_id_16) {
        (Some(lic), Some(buy)) => Some((
            check_len::<16>(lic, "license_id_16")?,
            check_len::<16>(buy, "buyer_id_16")?,
        )),
        (None, None) => None,
        _ => {
            return Err(validation_err(
                "license_id_16 and buyer_id_16 must both be provided or both omitted",
            ));
        }
    };
    let seed =
        derive_watermark_seed_hmac_sha256(&key, &scope, buyer_pair.as_ref().map(|(l, b)| (l, b)))
            .map_err(dnam_err_to_py)?;
    Ok(encode_hex(&seed))
}

/// Extracts the Stage III tier-local fingerprint component from `early_fir_128b` and `kautz_modal_bytes`.
#[pyfunction]
pub fn native_extract_stage3_fingerprint<'py>(
    py: Python<'py>,
    early_fir_128b: &[u8],
    kautz_modal_bytes: &[u8],
    added_modes: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let fp =
        extract_stage3_tier_fingerprint_component(early_fir_128b, kautz_modal_bytes, added_modes)
            .map_err(dnam_err_to_py)?;
    Ok(PyArray1::from_vec(py, fp))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(native_apply_isometric_gauge_watermark, m)?)?;
    m.add_function(wrap_pyfunction!(native_generate_tardos_code, m)?)?;
    m.add_function(wrap_pyfunction!(native_apply_tardos_watermark, m)?)?;
    m.add_function(wrap_pyfunction!(native_score_tardos_watermark, m)?)?;
    m.add_function(wrap_pyfunction!(native_extract_structural_hash, m)?)?;
    m.add_function(wrap_pyfunction!(native_structural_cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(native_derive_watermark_seed_hex, m)?)?;
    m.add_function(wrap_pyfunction!(native_extract_stage3_fingerprint, m)?)?;
    Ok(())
}
