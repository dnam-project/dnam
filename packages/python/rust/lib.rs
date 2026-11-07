//! `dnam._native` — Modular Zero-Copy `PyO3` (`abi3-py315` + `abi3t-py315`, PEP 803 `cp315-abi3.abi3t`)
//! and `NumPy` FFI Shim for `dnam-engine` (`ref[sdk.python.layout.package-namespace]`,
//! `ref[sdk.python.abi.abi3-abi3t]`, `ref[sdk.python.engine.zero-copy-and-gil]`).

#![forbid(unsafe_code)]

mod arrays;
mod controls;
mod dsp;
mod engine;
mod errors;
mod model;
mod provenance;
#[cfg(feature = "testing")]
mod testing;

use pyo3::prelude::*;

pub(crate) use errors::{control_err, dnam_err_to_py, validation_err};

#[pymodule(gil_used = false)]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    engine::register(m)?;
    model::register(m)?;
    controls::register(m)?;
    provenance::register(m)?;
    dsp::register(m)?;
    #[cfg(feature = "testing")]
    testing::register(m)?;
    Ok(())
}
