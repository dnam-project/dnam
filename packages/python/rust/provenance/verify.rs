//! Provenance verification, distribution policy enforcement & anti-cloning gate evaluation
//! (`ref[sdk.python.provenance.v2-security]`).

use dnam_engine::provenance::{
    AnticloneGateAction, RustCryptoProvider, VerificationStatus, decode_hex_array, encode_hex,
    enforce_distribution_policy, evaluate_anticlone_revocation_policy, verify_container_provenance,
};
use dnam_engine::schema::{DnamJsonHeader, DnamProtectionProfile};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::{dnam_err_to_py, validation_err};

/// Verifies `header_json` and `plaintext_arena` provenance against `has_signatures_flag`,
/// optional `device_id_hash`, `device_secret_32`, and `trusted_marketplace_pubkey_hex`.
#[pyfunction]
#[pyo3(signature = (header_json, plaintext_arena, has_signatures_flag = false, device_id_hash = None, device_secret_32 = None, trusted_marketplace_pubkey_hex = None))]
pub fn native_verify_container_provenance<'py>(
    py: Python<'py>,
    header_json: &str,
    plaintext_arena: &[u8],
    has_signatures_flag: bool,
    device_id_hash: Option<&str>,
    device_secret_32: Option<&[u8]>,
    trusted_marketplace_pubkey_hex: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let status = py.detach(|| -> PyResult<VerificationStatus> {
        let header: DnamJsonHeader = serde_json::from_str(header_json).map_err(validation_err)?;
        let mut crypto = RustCryptoProvider::new();
        if let (Some(id), Some(sec)) = (device_id_hash, device_secret_32) {
            let sec_arr: &[u8; 32] = sec.try_into().map_err(|_| {
                validation_err(format!(
                    "device_secret_32 must be 32 bytes, got {}",
                    sec.len()
                ))
            })?;
            crypto = crypto.with_device_key(id, *sec_arr);
        }
        if let Some(mkt_pk_hex) = trusted_marketplace_pubkey_hex {
            let mkt_pk = decode_hex_array::<32>(mkt_pk_hex).map_err(dnam_err_to_py)?;
            crypto = crypto.with_marketplace_pubkey(mkt_pk);
        }
        verify_container_provenance(&header, has_signatures_flag, plaintext_arena, &crypto)
            .map_err(dnam_err_to_py)
    })?;

    let dict = PyDict::new(py);
    match status {
        VerificationStatus::Unsigned => {
            dict.set_item("status", "unsigned")?;
            dict.set_item("signer_id", py.None())?;
            dict.set_item("pubkey_ed25519", py.None())?;
            dict.set_item("marketplace_id", py.None())?;
            dict.set_item("license_id", py.None())?;
            dict.set_item("license_tier", py.None())?;
        }
        VerificationStatus::CreatorVerified {
            signer_id,
            pubkey_ed25519,
        } => {
            dict.set_item("status", "creator_verified")?;
            dict.set_item("signer_id", signer_id)?;
            dict.set_item("pubkey_ed25519", encode_hex(&pubkey_ed25519))?;
            dict.set_item("marketplace_id", py.None())?;
            dict.set_item("license_id", py.None())?;
            dict.set_item("license_tier", py.None())?;
        }
        VerificationStatus::MarketplaceVerified {
            signer_id,
            marketplace_id,
            license_id,
            license_tier,
        } => {
            dict.set_item("status", "marketplace_verified")?;
            dict.set_item("signer_id", signer_id)?;
            dict.set_item("pubkey_ed25519", py.None())?;
            dict.set_item("marketplace_id", marketplace_id)?;
            dict.set_item("license_id", license_id)?;
            dict.set_item("license_tier", license_tier)?;
        }
    }
    Ok(dict)
}

/// Enforces the creator-signed `DistributionPolicy` in `header_json` against `active_profile`.
#[pyfunction]
pub fn native_enforce_distribution_policy(
    py: Python<'_>,
    header_json: &str,
    active_profile: &str,
) -> PyResult<()> {
    py.detach(|| {
        let header: DnamJsonHeader = serde_json::from_str(header_json).map_err(validation_err)?;
        let profile =
            DnamProtectionProfile::from_str_name(active_profile).map_err(dnam_err_to_py)?;
        enforce_distribution_policy(&header, profile).map_err(dnam_err_to_py)
    })
}

/// Evaluates the 3-stage anti-cloning revocation policy (`ref[sec.anticlone.three-stage-gate]`).
#[pyfunction]
pub fn native_evaluate_anticlone_policy(
    fingerprint_matched: bool,
    buyer_watermark_identified: bool,
    independent_device_source_evidence: bool,
) -> &'static str {
    match evaluate_anticlone_revocation_policy(
        fingerprint_matched,
        buyer_watermark_identified,
        independent_device_source_evidence,
    ) {
        AnticloneGateAction::NoMatchProceed => "no_match_proceed",
        AnticloneGateAction::FlagLicenseHoldForReview => "flag_license_hold_for_review",
        AnticloneGateAction::RevokeIdentifiedDeviceFutureLicensingOnly => {
            "revoke_identified_device_future_licensing_only"
        }
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(native_verify_container_provenance, m)?)?;
    m.add_function(wrap_pyfunction!(native_enforce_distribution_policy, m)?)?;
    m.add_function(wrap_pyfunction!(native_evaluate_anticlone_policy, m)?)?;
    Ok(())
}
