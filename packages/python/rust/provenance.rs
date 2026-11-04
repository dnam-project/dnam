//! Provenance verification, `Ed25519` signing, `.dnam-seal` (`DNSL`) HPKE sealing,
//! `DAU1`/`DME1` marketplace framing, preview containment, and dual-mechanism buyer watermarking
//! (`ref[sdk.python.provenance.v2-security]`).

mod marketplace;
mod sealing;
mod signing;
mod verify;
mod watermark;

use pyo3::prelude::*;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    verify::register(m)?;
    signing::register(m)?;
    sealing::register(m)?;
    marketplace::register(m)?;
    watermark::register(m)?;
    Ok(())
}
