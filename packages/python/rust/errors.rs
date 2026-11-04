//! Exhaustive `DnamError` -> `dnam.errors.*` exception mapping (`ref[sdk.python.errors.typed-hierarchy]`).

use dnam_engine::schema::DnamError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn raise_sdk_exception(cls_name: &str, tag: &str, msg: String) -> PyErr {
    Python::attach(|py| {
        if let Ok(mod_err) = py.import("dnam.errors")
            && let Ok(exc_cls) = mod_err.getattr(cls_name)
            && let Ok(inst) = exc_cls.call1((&msg,))
        {
            return PyErr::from_value(inst);
        }
        PyValueError::new_err(format!("{tag} {msg}"))
    })
}

/// Maps a [`DnamError`] directly into the corresponding `dnam.errors` Python exception subclass
/// (`ContainerError`, `ValidationError`, `SampleRateError`, `ControlError`, `ProvenanceError`)
/// on the cold error path (`ref[sdk.python.errors.typed-hierarchy]`).
///
/// Matches `DnamError` exhaustively without a wildcard arm.
pub(crate) fn dnam_err_to_py(err: DnamError) -> PyErr {
    let (cls_name, tag) = match &err {
        DnamError::InvalidMagic(_)
        | DnamError::UnsupportedVersion { .. }
        | DnamError::BufferTooSmall { .. }
        | DnamError::Crc32cMismatch { .. }
        | DnamError::HeaderCapacityExceeded { .. } => ("ContainerError", "[DNAM-CONTAINER]"),
        DnamError::InvalidSampleRate(_) => ("SampleRateError", "[DNAM-SAMPLERATE]"),
        DnamError::InvalidControlIndex(_) => ("ControlError", "[DNAM-CONTROL]"),
        DnamError::CryptoVerificationFailed(_)
        | DnamError::EncryptedArenaRequiresDeviceKey
        | DnamError::PreviewContainerRejected => ("ProvenanceError", "[DNAM-PROVENANCE]"),
        DnamError::UnalignedOffset { .. }
        | DnamError::InvalidJsonHeader(_)
        | DnamError::SchemaValidationError(_) => ("ValidationError", "[DNAM-VALIDATION]"),
    };
    raise_sdk_exception(cls_name, tag, err.to_string())
}

#[inline(never)]
pub(crate) fn validation_err(msg: impl core::fmt::Display) -> PyErr {
    raise_sdk_exception("ValidationError", "[DNAM-VALIDATION]", msg.to_string())
}

#[inline(never)]
pub(crate) fn control_err(msg: impl core::fmt::Display) -> PyErr {
    raise_sdk_exception("ControlError", "[DNAM-CONTROL]", msg.to_string())
}
