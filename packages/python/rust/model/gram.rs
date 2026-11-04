//! Column-Major Upper-Triangular (`LAPACK uplo='U'`) Gram Matrix packing and unpacking
//! (`ref[sdk.python.model.containers-and-checkpoints]`, `ref[container.ckpt.uplo-u]`).

use dnam_engine::container::{pack_uplo_u_f64, unpack_uplo_u_f64, uplo_u_packed_len};
use numpy::{
    PyArray1, PyReadonlyArray1, PyReadwriteArray1, PyReadwriteArray2, PyUntypedArrayMethods,
};
use pyo3::prelude::*;

use crate::arrays::{slice_1d, slice_1d_mut, slice_2d_mut};
use crate::validation_err;

/// Returns the packed element count $N(N+1)/2$ for an $N \times N$ symmetric `uplo='U'` matrix.
#[pyfunction]
pub fn native_uplo_u_packed_len(n_dim: usize) -> usize {
    uplo_u_packed_len(n_dim)
}

/// Packs a flattened row-major $N \times N$ symmetric matrix into `LAPACK uplo='U'` column-major upper-triangular format.
#[pyfunction]
pub fn native_pack_uplo_u_f64<'py>(
    py: Python<'py>,
    dense_flat: PyReadonlyArray1<'py, f64>,
    n_dim: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let expected = n_dim * n_dim;
    if dense_flat.len() != expected {
        return Err(validation_err(format!(
            "Expected {expected} elements for {n_dim}x{n_dim} matrix, got {}",
            dense_flat.len()
        )));
    }
    let slice = slice_1d(&dense_flat, "dense_flat")?;
    let packed_len = uplo_u_packed_len(n_dim);
    let mut out = vec![0.0_f64; packed_len];
    pack_uplo_u_f64(slice, n_dim, &mut out);
    Ok(PyArray1::from_vec(py, out))
}

/// Zero-copy in-place packing of a flattened row-major $N \times N$ symmetric matrix into `out_packed`.
#[pyfunction]
pub fn native_pack_uplo_u_f64_into<'py>(
    dense_flat: PyReadonlyArray1<'py, f64>,
    n_dim: usize,
    mut out_packed: PyReadwriteArray1<'py, f64>,
) -> PyResult<()> {
    let expected = n_dim * n_dim;
    if dense_flat.len() != expected {
        return Err(validation_err(format!(
            "Expected {expected} elements for {n_dim}x{n_dim} matrix, got {}",
            dense_flat.len()
        )));
    }
    let packed_len = uplo_u_packed_len(n_dim);
    if out_packed.len() != packed_len {
        return Err(validation_err(format!(
            "Expected out_packed length {packed_len}, got {}",
            out_packed.len()
        )));
    }
    let slice = slice_1d(&dense_flat, "dense_flat")?;
    let out_slice = slice_1d_mut(&mut out_packed, "out_packed")?;
    pack_uplo_u_f64(slice, n_dim, out_slice);
    Ok(())
}

/// Unpacks a `LAPACK uplo='U'` column-major upper-triangular buffer into a flattened $N_{\text{tier}} \times N_{\text{tier}}$ matrix.
#[pyfunction]
pub fn native_unpack_uplo_u_f64<'py>(
    py: Python<'py>,
    packed_u: PyReadonlyArray1<'py, f64>,
    n_tier: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let req_len = uplo_u_packed_len(n_tier);
    if packed_u.len() < req_len {
        return Err(validation_err(format!(
            "Packed buffer too short for n_tier={n_tier}: need {req_len}, got {}",
            packed_u.len()
        )));
    }
    let slice = slice_1d(&packed_u, "packed_u")?;
    let mut out = vec![0.0_f64; n_tier * n_tier];
    unpack_uplo_u_f64(slice, n_tier, &mut out);
    Ok(PyArray1::from_vec(py, out))
}

/// Zero-copy unpacking of a `LAPACK uplo='U'` buffer directly into a 2D `(n_tier, n_tier)` NumPy array.
#[pyfunction]
pub fn native_unpack_uplo_u_f64_into<'py>(
    packed_u: PyReadonlyArray1<'py, f64>,
    n_tier: usize,
    mut out_dense: PyReadwriteArray2<'py, f64>,
) -> PyResult<()> {
    let req_len = uplo_u_packed_len(n_tier);
    if packed_u.len() < req_len {
        return Err(validation_err(format!(
            "Packed buffer too short for n_tier={n_tier}: need {req_len}, got {}",
            packed_u.len()
        )));
    }
    if out_dense.shape() != [n_tier, n_tier] {
        return Err(validation_err(format!(
            "out_dense must have shape ({n_tier}, {n_tier}), got {:?}",
            out_dense.shape()
        )));
    }
    let slice = slice_1d(&packed_u, "packed_u")?;
    let out_slice = slice_2d_mut(&mut out_dense, "out_dense")?;
    unpack_uplo_u_f64(slice, n_tier, out_slice);
    Ok(())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(native_uplo_u_packed_len, m)?)?;
    m.add_function(wrap_pyfunction!(native_pack_uplo_u_f64, m)?)?;
    m.add_function(wrap_pyfunction!(native_pack_uplo_u_f64_into, m)?)?;
    m.add_function(wrap_pyfunction!(native_unpack_uplo_u_f64, m)?)?;
    m.add_function(wrap_pyfunction!(native_unpack_uplo_u_f64_into, m)?)?;
    Ok(())
}
