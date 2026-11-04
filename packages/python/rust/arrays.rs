//! Shared zero-copy NumPy slice extraction & shape validation helpers (`ref[sdk.python.engine.zero-copy-and-gil]`).

use numpy::{Element, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2};
use pyo3::prelude::*;

use crate::validation_err;

#[inline]
pub(crate) fn slice_1d<'a, 'py, T: Element>(
    arr: &'a PyReadonlyArray1<'py, T>,
    name: &str,
) -> PyResult<&'a [T]> {
    arr.as_slice()
        .map_err(|_| validation_err(format!("{name} must be C-contiguous")))
}

#[inline]
pub(crate) fn slice_1d_mut<'a, 'py, T: Element>(
    arr: &'a mut PyReadwriteArray1<'py, T>,
    name: &str,
) -> PyResult<&'a mut [T]> {
    arr.as_slice_mut()
        .map_err(|_| validation_err(format!("{name} must be C-contiguous")))
}

#[inline]
pub(crate) fn slice_2d<'a, 'py, T: Element>(
    arr: &'a PyReadonlyArray2<'py, T>,
    name: &str,
) -> PyResult<&'a [T]> {
    arr.as_slice()
        .map_err(|_| validation_err(format!("{name} must be C-contiguous")))
}

#[inline]
pub(crate) fn slice_2d_mut<'a, 'py, T: Element>(
    arr: &'a mut PyReadwriteArray2<'py, T>,
    name: &str,
) -> PyResult<&'a mut [T]> {
    arr.as_slice_mut()
        .map_err(|_| validation_err(format!("{name} must be C-contiguous")))
}
