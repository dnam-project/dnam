//! Zero-copy `NativeDnamEngine` `PyO3` class wrapping [`DynamicDnamEngine<'static>`].

use dnam_engine::container::{DnamPreamble, ZohRescaledParams};
use dnam_engine::math::{BackendKind, detect_best_backend};
use dnam_engine::{DynamicDnamEngine, Tier};
use numpy::{
    PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::dnam_err_to_py;

fn parse_backend_kind(backend: &str) -> PyResult<BackendKind> {
    match backend.to_ascii_lowercase().as_str() {
        "auto" => Ok(detect_best_backend()),
        "scalar" => Ok(BackendKind::Scalar),
        "neon" | "arm_neon" | "arm64_neon" => Ok(BackendKind::Neon),
        "arm_sme" | "armsme" | "sme" => Ok(BackendKind::ArmSme),
        "x86_simd" | "x86_avx2_fma" | "avx2" | "x86_sse42" => Ok(BackendKind::X86Simd),
        "wasm_simd128" | "wasm" => Ok(BackendKind::WasmSimd128),
        other => Err(PyValueError::new_err(format!(
            "[DNAM-VALIDATION] Unknown backend kind: '{other}'"
        ))),
    }
}

fn backend_kind_str(kind: BackendKind) -> &'static str {
    match kind {
        BackendKind::Scalar => "scalar",
        BackendKind::Neon | BackendKind::ArmNeon => "arm_neon",
        BackendKind::ArmSme => "arm_sme",
        BackendKind::X86Simd | BackendKind::X86Avx2Fma | BackendKind::X86Sse42 => "x86_simd",
        BackendKind::WasmSimd128 => "wasm_simd128",
    }
}

/// Native Rust SIMD engine wrapper exposed to `dnam_engine._native`.
#[pyclass]
#[derive(Debug)]
pub struct NativeDnamEngine {
    inner: DynamicDnamEngine<'static>,
    sample_rate_hz: u32,
    sample_rate_train_hz: u32,
    control_decimation_train: u8,
    zoh: ZohRescaledParams,
}

#[pymethods]
impl NativeDnamEngine {
    #[new]
    #[pyo3(signature = (container_bytes, tier = 1, sample_rate_hz = 48_000, backend = "auto"))]
    pub fn new(
        container_bytes: &[u8],
        tier: u8,
        sample_rate_hz: u32,
        backend: &str,
    ) -> PyResult<Self> {
        let preamble = DnamPreamble::from_bytes(container_bytes).map_err(dnam_err_to_py)?;
        let sample_rate_train_hz = preamble.sample_rate_train_hz;
        let control_decimation_train = preamble.control_decimation.max(8);

        let backend_enum = parse_backend_kind(backend)?;
        let tier_enum = Tier::from_u8(tier);
        let inner = DynamicDnamEngine::from_bytes_owned_with_backend(
            container_bytes,
            tier_enum,
            sample_rate_hz,
            backend_enum,
        )
        .map_err(dnam_err_to_py)?;

        let zoh = ZohRescaledParams::compute(
            sample_rate_train_hz,
            sample_rate_hz,
            control_decimation_train,
            0.9993,
            0.995,
            64.0,
            128,
        )
        .map_err(dnam_err_to_py)?;

        Ok(Self {
            inner,
            sample_rate_hz,
            sample_rate_train_hz,
            control_decimation_train,
            zoh,
        })
    }

    /// Hot-swaps a new `.dnam` or `.dnam-ckpt` container in-place.
    #[pyo3(signature = (container_bytes, tier = None, sample_rate_hz = None))]
    pub fn reload(
        &mut self,
        container_bytes: &[u8],
        tier: Option<u8>,
        sample_rate_hz: Option<u32>,
    ) -> PyResult<()> {
        let preamble = DnamPreamble::from_bytes(container_bytes).map_err(dnam_err_to_py)?;
        let target_tier = tier.map_or_else(
            || Tier::from_u8(self.inner.telemetry_snapshot().active_tier),
            Tier::from_u8,
        );
        let target_sr = sample_rate_hz.unwrap_or(self.sample_rate_hz);

        self.inner
            .reload_model_in_place_owned(container_bytes, target_tier, target_sr)
            .map_err(dnam_err_to_py)?;

        let sample_rate_train_hz = preamble.sample_rate_train_hz;
        let control_decimation_train = preamble.control_decimation.max(8);
        self.zoh = ZohRescaledParams::compute(
            sample_rate_train_hz,
            target_sr,
            control_decimation_train,
            0.9993,
            0.995,
            64.0,
            128,
        )
        .map_err(dnam_err_to_py)?;
        self.sample_rate_hz = target_sr;
        self.sample_rate_train_hz = sample_rate_train_hz;
        self.control_decimation_train = control_decimation_train;
        Ok(())
    }

    pub fn backend_kind(&self) -> &'static str {
        backend_kind_str(self.inner.backend_kind())
    }

    pub fn active_tier(&self) -> u8 {
        self.inner.telemetry_snapshot().active_tier
    }

    pub fn set_tier(&mut self, tier: u8) {
        self.inner.set_tier(Tier::from_u8(tier));
    }

    pub fn sample_rate_hz(&self) -> u32 {
        self.sample_rate_hz
    }

    pub fn set_sample_rate(&mut self, sample_rate_hz: u32) -> PyResult<()> {
        self.inner
            .set_sample_rate(sample_rate_hz)
            .map_err(dnam_err_to_py)?;
        self.zoh = ZohRescaledParams::compute(
            self.sample_rate_train_hz,
            sample_rate_hz,
            self.control_decimation_train,
            0.9993,
            0.995,
            64.0,
            128,
        )
        .map_err(dnam_err_to_py)?;
        self.sample_rate_hz = sample_rate_hz;
        Ok(())
    }

    pub fn set_control_value(&mut self, ctrl_idx: usize, value: f32) {
        self.inner.set_control_value(ctrl_idx, value);
    }

    pub fn set_control_immediate(&mut self, ctrl_idx: usize, value: f32) {
        self.inner.set_control_immediate(ctrl_idx, value);
    }

    pub fn reset(&mut self) {
        self.inner.reset_to_dc_equilibrium();
    }

    /// Processes stereo planar `float32` slices in-place into `left_out` and `right_out`,
    /// optionally chunked into `buffer_size` blocks while releasing the Python GIL.
    #[pyo3(signature = (left_in, right_in, left_out, right_out, buffer_size = None))]
    pub fn process_stereo<'py>(
        &mut self,
        py: Python<'py>,
        left_in: PyReadonlyArray1<'py, f32>,
        right_in: PyReadonlyArray1<'py, f32>,
        mut left_out: PyReadwriteArray1<'py, f32>,
        mut right_out: PyReadwriteArray1<'py, f32>,
        buffer_size: Option<usize>,
    ) -> PyResult<()> {
        let n = left_in.len();
        if right_in.len() != n || left_out.len() != n || right_out.len() != n {
            return Err(PyValueError::new_err(
                "[DNAM-VALIDATION] left_in, right_in, left_out, and right_out must have identical length",
            ));
        }
        let l_in = left_in
            .as_slice()
            .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] left_in must be contiguous"))?;
        let r_in = right_in
            .as_slice()
            .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] right_in must be contiguous"))?;
        let l_out = left_out
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] left_out must be contiguous"))?;
        let r_out = right_out
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] right_out must be contiguous"))?;

        let chunk = buffer_size.unwrap_or(n).max(1);
        py.detach(|| {
            if chunk >= n {
                self.inner.process_block(l_in, r_in, l_out, r_out);
            } else {
                for (((li, ri), lo), ro) in l_in
                    .chunks(chunk)
                    .zip(r_in.chunks(chunk))
                    .zip(l_out.chunks_mut(chunk))
                    .zip(r_out.chunks_mut(chunk))
                {
                    self.inner.process_block(li, ri, lo, ro);
                }
            }
        });
        Ok(())
    }

    /// Processes a 2D C-contiguous `(2, N)` stereo `float32` array directly into a `(2, N)` output array
    /// without creating intermediate 1D NumPy row slice objects in Python.
    #[pyo3(signature = (stereo_in, stereo_out, buffer_size = None))]
    pub fn process_stereo_2d<'py>(
        &mut self,
        py: Python<'py>,
        stereo_in: PyReadonlyArray2<'py, f32>,
        mut stereo_out: PyReadwriteArray2<'py, f32>,
        buffer_size: Option<usize>,
    ) -> PyResult<()> {
        let in_shape = stereo_in.shape();
        let out_shape = stereo_out.shape();
        if in_shape.len() != 2 || in_shape[0] != 2 || out_shape != in_shape {
            return Err(PyValueError::new_err(
                "[DNAM-VALIDATION] stereo_in and stereo_out must have matching (2, N) shape",
            ));
        }
        let n = in_shape[1];
        let in_slice = stereo_in
            .as_slice()
            .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] stereo_in must be contiguous"))?;
        let out_slice = stereo_out.as_slice_mut().map_err(|_| {
            PyValueError::new_err("[DNAM-VALIDATION] stereo_out must be contiguous")
        })?;
        let (l_in, r_in) = in_slice.split_at(n);
        let (l_out, r_out) = out_slice.split_at_mut(n);

        let chunk = buffer_size.unwrap_or(n).max(1);
        py.detach(|| {
            if chunk >= n {
                self.inner.process_block(l_in, r_in, l_out, r_out);
            } else {
                for (((li, ri), lo), ro) in l_in
                    .chunks(chunk)
                    .zip(r_in.chunks(chunk))
                    .zip(l_out.chunks_mut(chunk))
                    .zip(r_out.chunks_mut(chunk))
                {
                    self.inner.process_block(li, ri, lo, ro);
                }
            }
        });
        Ok(())
    }

    /// Processes a mono `float32` input slice into true-stereo `left_out` and `right_out`.
    #[pyo3(signature = (mono_in, left_out, right_out, buffer_size = None))]
    pub fn process_mono_to_stereo<'py>(
        &mut self,
        py: Python<'py>,
        mono_in: PyReadonlyArray1<'py, f32>,
        mut left_out: PyReadwriteArray1<'py, f32>,
        mut right_out: PyReadwriteArray1<'py, f32>,
        buffer_size: Option<usize>,
    ) -> PyResult<()> {
        let n = mono_in.len();
        if left_out.len() != n || right_out.len() != n {
            return Err(PyValueError::new_err(
                "[DNAM-VALIDATION] mono_in, left_out, and right_out must have identical length",
            ));
        }
        let m_in = mono_in
            .as_slice()
            .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] mono_in must be contiguous"))?;
        let l_out = left_out
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] left_out must be contiguous"))?;
        let r_out = right_out
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] right_out must be contiguous"))?;

        let chunk = buffer_size.unwrap_or(n).max(1);
        py.detach(|| {
            if chunk >= n {
                self.inner.process_mono_to_stereo(m_in, l_out, r_out);
            } else {
                for ((mi, lo), ro) in m_in
                    .chunks(chunk)
                    .zip(l_out.chunks_mut(chunk))
                    .zip(r_out.chunks_mut(chunk))
                {
                    self.inner.process_mono_to_stereo(mi, lo, ro);
                }
            }
        });
        Ok(())
    }

    /// Processes a 1D mono `float32` input slice directly into a 2D `(2, N)` stereo output array.
    #[pyo3(signature = (mono_in, stereo_out, buffer_size = None))]
    pub fn process_mono_to_stereo_2d<'py>(
        &mut self,
        py: Python<'py>,
        mono_in: PyReadonlyArray1<'py, f32>,
        mut stereo_out: PyReadwriteArray2<'py, f32>,
        buffer_size: Option<usize>,
    ) -> PyResult<()> {
        let n = mono_in.len();
        let out_shape = stereo_out.shape();
        if out_shape.len() != 2 || out_shape[0] != 2 || out_shape[1] != n {
            return Err(PyValueError::new_err(
                "[DNAM-VALIDATION] stereo_out must have shape (2, N)",
            ));
        }
        let m_in = mono_in
            .as_slice()
            .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] mono_in must be contiguous"))?;
        let out_slice = stereo_out.as_slice_mut().map_err(|_| {
            PyValueError::new_err("[DNAM-VALIDATION] stereo_out must be contiguous")
        })?;
        let (l_out, r_out) = out_slice.split_at_mut(n);

        let chunk = buffer_size.unwrap_or(n).max(1);
        py.detach(|| {
            if chunk >= n {
                self.inner.process_mono_to_stereo(m_in, l_out, r_out);
            } else {
                for ((mi, lo), ro) in m_in
                    .chunks(chunk)
                    .zip(l_out.chunks_mut(chunk))
                    .zip(r_out.chunks_mut(chunk))
                {
                    self.inner.process_mono_to_stereo(mi, lo, ro);
                }
            }
        });
        Ok(())
    }

    /// Processes a mono `float32` input slice into a 1-channel `mono_out` slice.
    #[pyo3(signature = (mono_in, mono_out, buffer_size = None))]
    pub fn process_mono<'py>(
        &mut self,
        py: Python<'py>,
        mono_in: PyReadonlyArray1<'py, f32>,
        mut mono_out: PyReadwriteArray1<'py, f32>,
        buffer_size: Option<usize>,
    ) -> PyResult<()> {
        let n = mono_in.len();
        if mono_out.len() != n {
            return Err(PyValueError::new_err(
                "[DNAM-VALIDATION] mono_in and mono_out must have identical length",
            ));
        }
        let m_in = mono_in
            .as_slice()
            .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] mono_in must be contiguous"))?;
        let m_out = mono_out
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] mono_out must be contiguous"))?;

        let chunk = buffer_size.unwrap_or(n).max(1);
        py.detach(|| {
            if chunk >= n {
                self.inner.process_mono(m_in, m_out);
            } else {
                for (mi, mo) in m_in.chunks(chunk).zip(m_out.chunks_mut(chunk)) {
                    self.inner.process_mono(mi, mo);
                }
            }
        });
        Ok(())
    }

    /// Returns a dictionary snapshot of `DnamTelemetrySnapshot` and `ZohRescaledParams`.
    pub fn telemetry<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let snap = self.inner.telemetry_snapshot();
        let zoh = &self.zoh;

        let dict = PyDict::new(py);
        dict.set_item("active_tier", snap.active_tier)?;
        dict.set_item("backend", backend_kind_str(self.inner.backend_kind()))?;
        dict.set_item("sample_rate_hz", self.sample_rate_hz)?;
        dict.set_item("core2_active", snap.core2_active != 0)?;
        dict.set_item("lfo_env_active", snap.lfo_env_active != 0)?;
        dict.set_item("stereo_out", snap.stereo_out != 0)?;
        dict.set_item("active_clamp_mask", snap.active_clamp_mask)?;
        dict.set_item("control_decimation_host", snap.control_decimation_host)?;
        dict.set_item("max_adaa_drive", snap.max_adaa_drive)?;
        dict.set_item("max_pole_radius", snap.max_pole_radius)?;
        dict.set_item("max_effective_loop_gain", snap.max_effective_loop_gain)?;
        dict.set_item("tail_time_ms", snap.tail_time_ms)?;
        dict.set_item(
            "z_head",
            (
                snap.z_head[0],
                snap.z_head[1],
                snap.z_head[2],
                snap.z_head[3],
            ),
        )?;
        dict.set_item(
            "env_states",
            (
                snap.env_states[0],
                snap.env_states[1],
                snap.env_states[2],
                snap.env_states[3],
            ),
        )?;
        dict.set_item(
            "lfo_harmonics",
            (
                snap.lfo_harmonics[0],
                snap.lfo_harmonics[1],
                snap.lfo_harmonics[2],
                snap.lfo_harmonics[3],
                snap.lfo_harmonics[4],
                snap.lfo_harmonics[5],
                snap.lfo_harmonics[6],
                snap.lfo_harmonics[7],
            ),
        )?;
        dict.set_item("clamped_controls", snap.clamped_controls.to_vec())?;
        dict.set_item("rho_sr", zoh.rho_sr)?;
        dict.set_item("rho_ctrl", zoh.rho_ctrl)?;
        dict.set_item("r_ac_host", zoh.r_ac_host)?;
        dict.set_item("r_dc_host", zoh.r_dc_host)?;
        dict.set_item("tau_ref_samples_host", zoh.tau_ref_samples_host)?;
        dict.set_item("switch_xfade_samples_host", zoh.switch_xfade_samples_host)?;
        Ok(dict)
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeDnamEngine>()?;
    Ok(())
}
