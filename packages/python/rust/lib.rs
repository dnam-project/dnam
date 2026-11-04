//! `dnam._native` — Modular Zero-Copy `PyO3` (`abi3-py314`) + `NumPy` FFI Shim for `dnam-engine`
//! (`ref[sdk.python.layout.package-namespace]`, `ref[sdk.python.engine.zero-copy-and-gil]`).

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

#[pymodule]
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
