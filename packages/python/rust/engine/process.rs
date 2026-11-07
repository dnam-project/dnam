//! Zero-copy, GIL-releasing block audio processing helpers for [`PyEngine`]
//! (`ref[sdk.python.engine.zero-copy-and-gil]`, `ref[sdk.python.engine.pedalboard-call]`).

use numpy::{
    PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2, PyUntypedArrayMethods,
};
use pyo3::prelude::*;

use super::PyEngine;
use crate::arrays::{slice_1d, slice_1d_mut, slice_2d, slice_2d_mut};
use crate::validation_err;

impl PyEngine {
    pub(crate) fn render_offline_bounce_impl<'py>(
        &self,
        py: Python<'py>,
        left_in: PyReadonlyArray1<'py, f32>,
        right_in: Option<PyReadonlyArray1<'py, f32>>,
        mut stereo_out: PyReadwriteArray2<'py, f32>,
        maximize_tier: Option<bool>,
    ) -> PyResult<()> {
        let n = left_in.len();
        if let Some(ref r_arr) = right_in
            && r_arr.len() != n
        {
            return Err(validation_err(
                "left_in and right_in must have identical length",
            ));
        }
        let out_shape = stereo_out.shape();
        if out_shape.len() != 2 || out_shape[0] != 2 || out_shape[1] != n {
            return Err(validation_err("stereo_out must have shape (2, N)"));
        }
        let l_in = slice_1d(&left_in, "left_in")?;
        let r_in = match right_in {
            Some(ref r_arr) => slice_1d(r_arr, "right_in")?,
            None => l_in,
        };
        let out_slice = slice_2d_mut(&mut stereo_out, "stereo_out")?;
        let (l_out, r_out) = out_slice.split_at_mut(n);
        py.detach(|| {
            self.host
                .render_offline_bounce_with_options(l_in, r_in, l_out, r_out, maximize_tier);
        });
        Ok(())
    }

    pub(crate) fn process_stereo_impl<'py>(
        &self,
        py: Python<'py>,
        left_in: PyReadonlyArray1<'py, f32>,
        right_in: PyReadonlyArray1<'py, f32>,
        mut left_out: PyReadwriteArray1<'py, f32>,
        mut right_out: PyReadwriteArray1<'py, f32>,
        buffer_size: Option<usize>,
    ) -> PyResult<()> {
        let n = left_in.len();
        if right_in.len() != n || left_out.len() != n || right_out.len() != n {
            return Err(validation_err(
                "left_in, right_in, left_out, and right_out must have identical length",
            ));
        }
        let l_in = slice_1d(&left_in, "left_in")?;
        let r_in = slice_1d(&right_in, "right_in")?;
        let l_out = slice_1d_mut(&mut left_out, "left_out")?;
        let r_out = slice_1d_mut(&mut right_out, "right_out")?;

        py.detach(|| {
            self.host
                .process_block_chunked(l_in, r_in, l_out, r_out, buffer_size);
        });
        Ok(())
    }

    pub(crate) fn process_stereo_in_place_impl<'py>(
        &self,
        py: Python<'py>,
        mut left_io: PyReadwriteArray1<'py, f32>,
        mut right_io: PyReadwriteArray1<'py, f32>,
        buffer_size: Option<usize>,
    ) -> PyResult<()> {
        let n = left_io.len();
        if right_io.len() != n {
            return Err(validation_err(
                "left_io and right_io must have identical length",
            ));
        }
        let l_io = slice_1d_mut(&mut left_io, "left_io")?;
        let r_io = slice_1d_mut(&mut right_io, "right_io")?;
        py.detach(|| {
            self.host
                .process_block_in_place_chunked(l_io, r_io, buffer_size);
        });
        Ok(())
    }

    pub(crate) fn process_stereo_2d_impl<'py>(
        &self,
        py: Python<'py>,
        stereo_in: PyReadonlyArray2<'py, f32>,
        mut stereo_out: PyReadwriteArray2<'py, f32>,
        buffer_size: Option<usize>,
    ) -> PyResult<()> {
        let in_shape = stereo_in.shape();
        let out_shape = stereo_out.shape();
        if in_shape.len() != 2 || in_shape[0] != 2 || out_shape != in_shape {
            return Err(validation_err(
                "stereo_in and stereo_out must have matching (2, N) shape",
            ));
        }
        let n = in_shape[1];
        let in_slice = slice_2d(&stereo_in, "stereo_in")?;
        let out_slice = slice_2d_mut(&mut stereo_out, "stereo_out")?;
        let (l_in, r_in) = in_slice.split_at(n);
        let (l_out, r_out) = out_slice.split_at_mut(n);

        py.detach(|| {
            self.host
                .process_block_chunked(l_in, r_in, l_out, r_out, buffer_size);
        });
        Ok(())
    }

    pub(crate) fn process_stereo_2d_in_place_impl<'py>(
        &self,
        py: Python<'py>,
        mut stereo_io: PyReadwriteArray2<'py, f32>,
        buffer_size: Option<usize>,
    ) -> PyResult<()> {
        let shape = stereo_io.shape();
        if shape.len() != 2 || shape[0] != 2 {
            return Err(validation_err("stereo_io must have shape (2, N)"));
        }
        let n = shape[1];
        let slice = slice_2d_mut(&mut stereo_io, "stereo_io")?;
        let (l_io, r_io) = slice.split_at_mut(n);
        py.detach(|| {
            self.host
                .process_block_in_place_chunked(l_io, r_io, buffer_size);
        });
        Ok(())
    }

    pub(crate) fn process_mono_to_stereo_impl<'py>(
        &self,
        py: Python<'py>,
        mono_in: PyReadonlyArray1<'py, f32>,
        mut left_out: PyReadwriteArray1<'py, f32>,
        mut right_out: PyReadwriteArray1<'py, f32>,
        buffer_size: Option<usize>,
    ) -> PyResult<()> {
        let n = mono_in.len();
        if left_out.len() != n || right_out.len() != n {
            return Err(validation_err(
                "mono_in, left_out, and right_out must have identical length",
            ));
        }
        let m_in = slice_1d(&mono_in, "mono_in")?;
        let l_out = slice_1d_mut(&mut left_out, "left_out")?;
        let r_out = slice_1d_mut(&mut right_out, "right_out")?;

        py.detach(|| {
            self.host
                .process_mono_to_stereo_chunked(m_in, l_out, r_out, buffer_size);
        });
        Ok(())
    }

    pub(crate) fn process_mono_to_stereo_2d_impl<'py>(
        &self,
        py: Python<'py>,
        mono_in: PyReadonlyArray1<'py, f32>,
        mut stereo_out: PyReadwriteArray2<'py, f32>,
        buffer_size: Option<usize>,
    ) -> PyResult<()> {
        let n = mono_in.len();
        let out_shape = stereo_out.shape();
        if out_shape.len() != 2 || out_shape[0] != 2 || out_shape[1] != n {
            return Err(validation_err("stereo_out must have shape (2, N)"));
        }
        let m_in = slice_1d(&mono_in, "mono_in")?;
        let out_slice = slice_2d_mut(&mut stereo_out, "stereo_out")?;
        let (l_out, r_out) = out_slice.split_at_mut(n);

        py.detach(|| {
            self.host
                .process_mono_to_stereo_chunked(m_in, l_out, r_out, buffer_size);
        });
        Ok(())
    }

    pub(crate) fn process_mono_impl<'py>(
        &self,
        py: Python<'py>,
        mono_in: PyReadonlyArray1<'py, f32>,
        mut mono_out: PyReadwriteArray1<'py, f32>,
        buffer_size: Option<usize>,
    ) -> PyResult<()> {
        let n = mono_in.len();
        if mono_out.len() != n {
            return Err(validation_err(
                "mono_in and mono_out must have identical length",
            ));
        }
        let m_in = slice_1d(&mono_in, "mono_in")?;
        let m_out = slice_1d_mut(&mut mono_out, "mono_out")?;

        py.detach(|| {
            self.host.process_mono_chunked(m_in, m_out, buffer_size);
        });
        Ok(())
    }

    pub(crate) fn process_mono_in_place_impl<'py>(
        &self,
        py: Python<'py>,
        mut mono_io: PyReadwriteArray1<'py, f32>,
        buffer_size: Option<usize>,
    ) -> PyResult<()> {
        let m_io = slice_1d_mut(&mut mono_io, "mono_io")?;
        py.detach(|| {
            self.host.process_mono_in_place_chunked(m_io, buffer_size);
        });
        Ok(())
    }
}
