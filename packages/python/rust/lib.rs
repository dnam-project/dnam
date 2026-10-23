//! `dnam_engine._native` — Modular Zero-Copy `PyO3` (`abi3-py314`) + `NumPy` FFI Layer for `dnam-engine`.

mod container;
mod dsp;
mod engine;
mod provenance;

use dnam_engine::schema::DnamError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Maps a [`DnamError`] into a Python `ValueError` with a structured `[DNAM-*]` tag prefix
/// so the Python interface (`dnam_engine.exceptions`) can classify it into the exact
/// `DnamError` subclass (`DnamContainerError`, `DnamValidationError`, `DnamSampleRateError`).
pub(crate) fn dnam_err_to_py(err: DnamError) -> PyErr {
    let tagged = match &err {
        DnamError::InvalidMagic(_)
        | DnamError::UnsupportedVersion { .. }
        | DnamError::BufferTooSmall { .. }
        | DnamError::Crc32cMismatch { .. }
        | DnamError::HeaderCapacityExceeded { .. } => format!("[DNAM-CONTAINER] {err}"),
        DnamError::InvalidSampleRate(_) => format!("[DNAM-SAMPLERATE] {err}"),
        DnamError::UnalignedOffset { .. }
        | DnamError::InvalidJsonHeader(_)
        | DnamError::SchemaValidationError(_)
        | DnamError::InvalidControlIndex(_)
        | DnamError::CryptoVerificationFailed(_)
        | DnamError::EncryptedArenaRequiresDeviceKey => format!("[DNAM-VALIDATION] {err}"),
    };
    PyValueError::new_err(tagged)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    engine::register(m)?;
    container::register(m)?;
    dsp::register(m)?;
    provenance::register(m)?;
    Ok(())
}
