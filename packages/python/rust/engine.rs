//! Zero-copy `PyEngine` (`dnam._native.PyEngine`) wrapping [`DynamicDnamEngine<'static>`]
//! (`ref[sdk.python.engine.zero-copy-and-gil]`, `ref[sdk.python.engine.pedalboard-call]`).

mod process;
mod runtime;
mod telemetry;

use std::sync::{Mutex, MutexGuard, PoisonError};

use dnam_engine::math::{BackendKind, detect_best_backend};
use dnam_engine::{DynamicDnamEngine, Tier};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::{dnam_err_to_py, validation_err};

pub(crate) fn parse_backend_kind(backend: &str) -> PyResult<BackendKind> {
    match backend.to_ascii_lowercase().as_str() {
        "auto" => Ok(detect_best_backend()),
        "scalar" => Ok(BackendKind::Scalar),
        "neon" | "arm_neon" | "arm64_neon" => Ok(BackendKind::Neon),
        "arm_sme" | "armsme" | "sme" => Ok(BackendKind::ArmSme),
        "x86_simd" | "x86_avx2_fma" | "avx2" | "x86_sse42" => Ok(BackendKind::X86Simd),
        "wasm_simd128" | "wasm" => Ok(BackendKind::WasmSimd128),
        other => Err(validation_err(format!("Unknown backend kind: '{other}'"))),
    }
}

pub(crate) const fn backend_kind_str(kind: BackendKind) -> &'static str {
    match kind {
        BackendKind::Scalar => "scalar",
        BackendKind::Neon | BackendKind::ArmNeon => "neon",
        BackendKind::ArmSme => "arm_sme",
        BackendKind::X86Simd | BackendKind::X86Avx2Fma | BackendKind::X86Sse42 => "x86_simd",
        BackendKind::WasmSimd128 => "wasm_simd128",
    }
}

/// Native Rust SIMD engine wrapper exposed as `dnam._native.PyEngine`.
///
/// Delegates 100% of DSP, ZOH rescaling, control clamping, preset loading, and tempo sync
/// to [`DynamicDnamEngine<'static>`] (`ref[sdk.python.engine.zero-copy-and-gil]`,
/// `ref[sdk.python.engine.free-threading]`).
#[pyclass(module = "dnam._native", frozen)]
#[derive(Debug)]
pub struct PyEngine {
    pub(crate) inner: Mutex<DynamicDnamEngine<'static>>,
}

impl PyEngine {
    #[inline]
    pub(crate) fn lock_inner(&self) -> MutexGuard<'_, DynamicDnamEngine<'static>> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[pymethods]
impl PyEngine {
    #[new]
    #[pyo3(signature = (container_bytes, tier = 1, sample_rate_hz = 48_000, backend = "auto", max_supported_tier = None, offline_bounce = false, maximize_tier_on_bounce = true))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        py: Python<'_>,
        container_bytes: &[u8],
        tier: u8,
        sample_rate_hz: u32,
        backend: &str,
        max_supported_tier: Option<u8>,
        offline_bounce: bool,
        maximize_tier_on_bounce: bool,
    ) -> PyResult<Self> {
        let backend_enum = parse_backend_kind(backend)?;
        let tier_enum = Tier::from_u8(tier);
        let max_tier_enum = max_supported_tier.map_or(Tier::Ultra, Tier::from_u8);
        let inner = py
            .detach(|| {
                let mut eng = DynamicDnamEngine::from_bytes_owned_with_backend(
                    container_bytes,
                    tier_enum,
                    sample_rate_hz,
                    backend_enum,
                )?;
                if max_tier_enum < Tier::Ultra {
                    eng.set_max_supported_tier(max_tier_enum);
                }
                eng.set_maximize_tier_on_bounce(maximize_tier_on_bounce);
                if offline_bounce {
                    eng.set_offline_bounce(true);
                }
                Ok(eng)
            })
            .map_err(dnam_err_to_py)?;
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    #[pyo3(signature = (container_bytes, tier = None, sample_rate_hz = None))]
    pub fn reload(
        &self,
        py: Python<'_>,
        container_bytes: &[u8],
        tier: Option<u8>,
        sample_rate_hz: Option<u32>,
    ) -> PyResult<()> {
        self.reload_impl(py, container_bytes, tier, sample_rate_hz)
    }

    pub fn backend_kind(&self, py: Python<'_>) -> &'static str {
        py.detach(|| backend_kind_str(self.lock_inner().backend_kind()))
    }

    pub fn active_tier(&self, py: Python<'_>) -> u8 {
        py.detach(|| self.lock_inner().tier() as u8)
    }

    pub fn set_tier(&self, py: Python<'_>, tier: u8) {
        let parsed = Tier::from_u8(tier);
        py.detach(|| self.lock_inner().set_tier(parsed));
    }

    pub fn max_available_tier(&self, py: Python<'_>) -> u8 {
        py.detach(|| self.lock_inner().max_available_tier() as u8)
    }

    pub fn max_supported_tier(&self, py: Python<'_>) -> u8 {
        py.detach(|| self.lock_inner().max_supported_tier() as u8)
    }

    pub fn set_max_supported_tier(&self, py: Python<'_>, tier: u8) {
        let parsed = Tier::from_u8(tier);
        py.detach(|| self.lock_inner().set_max_supported_tier(parsed));
    }

    pub fn effective_tier(&self, py: Python<'_>) -> u8 {
        py.detach(|| self.lock_inner().effective_tier() as u8)
    }

    pub fn offline_bounce(&self, py: Python<'_>) -> bool {
        py.detach(|| self.lock_inner().offline_bounce())
    }

    pub fn set_offline_bounce(&self, py: Python<'_>, enabled: bool) {
        py.detach(|| self.lock_inner().set_offline_bounce(enabled));
    }

    pub fn maximize_tier_on_bounce(&self, py: Python<'_>) -> bool {
        py.detach(|| self.lock_inner().maximize_tier_on_bounce())
    }

    pub fn set_maximize_tier_on_bounce(&self, py: Python<'_>, enabled: bool) {
        py.detach(|| self.lock_inner().set_maximize_tier_on_bounce(enabled));
    }

    pub fn latency_samples(&self, py: Python<'_>) -> usize {
        py.detach(|| self.lock_inner().latency_samples())
    }

    pub fn sample_rate_hz(&self, py: Python<'_>) -> u32 {
        py.detach(|| self.lock_inner().sample_rate_hz())
    }

    pub fn set_sample_rate(&self, py: Python<'_>, sample_rate_hz: u32) -> PyResult<()> {
        self.set_sample_rate_impl(py, sample_rate_hz)
    }

    pub fn num_controls(&self, py: Python<'_>) -> usize {
        py.detach(|| self.lock_inner().num_controls())
    }

    pub fn raw_control_values(&self, py: Python<'_>) -> Vec<f32> {
        py.detach(|| self.lock_inner().raw_control_values().to_vec())
    }

    pub fn clamped_control_values(&self, py: Python<'_>) -> Vec<f32> {
        py.detach(|| self.lock_inner().clamped_control_values().to_vec())
    }

    pub fn clamped_control_value(&self, py: Python<'_>, ctrl_idx: usize) -> PyResult<f32> {
        self.clamped_control_value_impl(py, ctrl_idx)
    }

    pub fn set_control_value(&self, py: Python<'_>, ctrl_idx: usize, value: f32) -> PyResult<()> {
        self.set_control_value_impl(py, ctrl_idx, value)
    }

    pub fn set_control_immediate(
        &self,
        py: Python<'_>,
        ctrl_idx: usize,
        value: f32,
    ) -> PyResult<()> {
        self.set_control_immediate_impl(py, ctrl_idx, value)
    }

    pub fn set_controls_immediate_batch(
        &self,
        py: Python<'_>,
        updates: Vec<(usize, f32)>,
    ) -> PyResult<()> {
        self.set_controls_immediate_batch_impl(py, &updates)
    }

    pub fn reset_controls(&self, py: Python<'_>) {
        py.detach(|| self.lock_inner().reset_controls_to_defaults());
    }

    pub fn reset(&self, py: Python<'_>) {
        py.detach(|| self.lock_inner().reset_to_dc_equilibrium());
    }

    pub fn supports_freeze(&self, py: Python<'_>) -> bool {
        py.detach(|| self.lock_inner().supports_freeze())
    }

    pub fn set_freeze(&self, py: Python<'_>, freeze: bool) {
        py.detach(|| self.lock_inner().set_freeze(freeze));
    }

    pub fn set_freeze_amount(&self, py: Python<'_>, alpha: f32) {
        py.detach(|| self.lock_inner().set_freeze_amount(alpha));
    }

    pub fn is_frozen(&self, py: Python<'_>) -> bool {
        py.detach(|| self.lock_inner().is_frozen())
    }

    pub fn set_bypass(&self, py: Python<'_>, bypassed: bool, trails: bool) {
        py.detach(|| self.lock_inner().set_bypass(bypassed, trails));
    }

    pub fn bypass_state(&self, py: Python<'_>) -> (bool, bool) {
        py.detach(|| self.lock_inner().bypass_state())
    }

    pub fn set_stereo_width(&self, py: Python<'_>, width: f32) {
        py.detach(|| self.lock_inner().set_stereo_width(width));
    }

    pub fn stereo_width(&self, py: Python<'_>) -> f32 {
        py.detach(|| self.lock_inner().stereo_width())
    }

    #[pyo3(signature = (blend_wet = 1.0, output_gain_db = 0.0, apply_unity_leveling = true))]
    pub fn configure_host_mix_and_leveling(
        &self,
        py: Python<'_>,
        blend_wet: f32,
        output_gain_db: f32,
        apply_unity_leveling: bool,
    ) {
        py.detach(|| {
            self.lock_inner().configure_host_mix_and_leveling(
                blend_wet,
                output_gain_db,
                apply_unity_leveling,
            );
        });
    }

    pub fn current_bpm(&self, py: Python<'_>) -> f32 {
        py.detach(|| self.lock_inner().current_bpm())
    }

    #[pyo3(signature = (bpm, subdivision = None, fold_octave = false))]
    pub fn sync_tempo(
        &self,
        py: Python<'_>,
        bpm: f32,
        subdivision: Option<&str>,
        fold_octave: bool,
    ) -> PyResult<usize> {
        self.sync_tempo_impl(py, bpm, subdivision, fold_octave)
    }

    #[pyo3(signature = (subdivision = None, fold_octave = false))]
    pub fn register_tap_tempo(
        &self,
        py: Python<'_>,
        subdivision: Option<&str>,
        fold_octave: bool,
    ) -> PyResult<(f32, usize)> {
        self.register_tap_tempo_impl(py, subdivision, fold_octave)
    }

    pub fn apply_preset(&self, py: Python<'_>, preset_id: &str) -> PyResult<()> {
        self.apply_preset_impl(py, preset_id)
    }

    #[pyo3(signature = (left_in, right_in, stereo_out, maximize_tier = None))]
    pub fn render_offline_bounce<'py>(
        &self,
        py: Python<'py>,
        left_in: PyReadonlyArray1<'py, f32>,
        right_in: Option<PyReadonlyArray1<'py, f32>>,
        stereo_out: PyReadwriteArray2<'py, f32>,
        maximize_tier: Option<bool>,
    ) -> PyResult<()> {
        self.render_offline_bounce_impl(py, left_in, right_in, stereo_out, maximize_tier)
    }

    #[pyo3(signature = (left_in, right_in, left_out, right_out, buffer_size = None))]
    pub fn process_stereo<'py>(
        &self,
        py: Python<'py>,
        left_in: PyReadonlyArray1<'py, f32>,
        right_in: PyReadonlyArray1<'py, f32>,
        left_out: PyReadwriteArray1<'py, f32>,
        right_out: PyReadwriteArray1<'py, f32>,
        buffer_size: Option<usize>,
    ) -> PyResult<()> {
        self.process_stereo_impl(py, left_in, right_in, left_out, right_out, buffer_size)
    }

    #[pyo3(signature = (left_io, right_io, buffer_size = None))]
    pub fn process_stereo_in_place<'py>(
        &self,
        py: Python<'py>,
        left_io: PyReadwriteArray1<'py, f32>,
        right_io: PyReadwriteArray1<'py, f32>,
        buffer_size: Option<usize>,
    ) -> PyResult<()> {
        self.process_stereo_in_place_impl(py, left_io, right_io, buffer_size)
    }

    #[pyo3(signature = (stereo_in, stereo_out, buffer_size = None))]
    pub fn process_stereo_2d<'py>(
        &self,
        py: Python<'py>,
        stereo_in: PyReadonlyArray2<'py, f32>,
        stereo_out: PyReadwriteArray2<'py, f32>,
        buffer_size: Option<usize>,
    ) -> PyResult<()> {
        self.process_stereo_2d_impl(py, stereo_in, stereo_out, buffer_size)
    }

    #[pyo3(signature = (stereo_io, buffer_size = None))]
    pub fn process_stereo_2d_in_place<'py>(
        &self,
        py: Python<'py>,
        stereo_io: PyReadwriteArray2<'py, f32>,
        buffer_size: Option<usize>,
    ) -> PyResult<()> {
        self.process_stereo_2d_in_place_impl(py, stereo_io, buffer_size)
    }

    #[pyo3(signature = (mono_in, left_out, right_out, buffer_size = None))]
    pub fn process_mono_to_stereo<'py>(
        &self,
        py: Python<'py>,
        mono_in: PyReadonlyArray1<'py, f32>,
        left_out: PyReadwriteArray1<'py, f32>,
        right_out: PyReadwriteArray1<'py, f32>,
        buffer_size: Option<usize>,
    ) -> PyResult<()> {
        self.process_mono_to_stereo_impl(py, mono_in, left_out, right_out, buffer_size)
    }

    #[pyo3(signature = (mono_in, stereo_out, buffer_size = None))]
    pub fn process_mono_to_stereo_2d<'py>(
        &self,
        py: Python<'py>,
        mono_in: PyReadonlyArray1<'py, f32>,
        stereo_out: PyReadwriteArray2<'py, f32>,
        buffer_size: Option<usize>,
    ) -> PyResult<()> {
        self.process_mono_to_stereo_2d_impl(py, mono_in, stereo_out, buffer_size)
    }

    #[pyo3(signature = (mono_in, mono_out, buffer_size = None))]
    pub fn process_mono<'py>(
        &self,
        py: Python<'py>,
        mono_in: PyReadonlyArray1<'py, f32>,
        mono_out: PyReadwriteArray1<'py, f32>,
        buffer_size: Option<usize>,
    ) -> PyResult<()> {
        self.process_mono_impl(py, mono_in, mono_out, buffer_size)
    }

    #[pyo3(signature = (mono_io, buffer_size = None))]
    pub fn process_mono_in_place<'py>(
        &self,
        py: Python<'py>,
        mono_io: PyReadwriteArray1<'py, f32>,
        buffer_size: Option<usize>,
    ) -> PyResult<()> {
        self.process_mono_in_place_impl(py, mono_io, buffer_size)
    }

    pub fn telemetry<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.telemetry_impl(py)
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    Ok(())
}
