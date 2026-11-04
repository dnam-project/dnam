//! Zero-copy `PyEngine` (`dnam._native.PyEngine`) wrapping [`DynamicDnamEngine<'static>`]
//! (`ref[sdk.python.engine.zero-copy-and-gil]`, `ref[sdk.python.engine.pedalboard-call]`).

mod process;
mod runtime;
mod telemetry;

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
/// to [`DynamicDnamEngine<'static>`] (`ref[sdk.python.engine.zero-copy-and-gil]`).
#[pyclass(module = "dnam._native")]
#[derive(Debug)]
pub struct PyEngine {
    pub(crate) inner: DynamicDnamEngine<'static>,
}

#[pymethods]
impl PyEngine {
    #[new]
    #[pyo3(signature = (container_bytes, tier = 1, sample_rate_hz = 48_000, backend = "auto", max_supported_tier = None, offline_bounce = false, maximize_tier_on_bounce = true))]
    pub fn new(
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
        let mut inner = DynamicDnamEngine::from_bytes_owned_with_backend(
            container_bytes,
            tier_enum,
            sample_rate_hz,
            backend_enum,
        )
        .map_err(dnam_err_to_py)?;
        if max_tier_enum < Tier::Ultra {
            inner.set_max_supported_tier(max_tier_enum);
        }
        inner.set_maximize_tier_on_bounce(maximize_tier_on_bounce);
        if offline_bounce {
            inner.set_offline_bounce(true);
        }
        Ok(Self { inner })
    }

    #[pyo3(signature = (container_bytes, tier = None, sample_rate_hz = None))]
    pub fn reload(
        &mut self,
        container_bytes: &[u8],
        tier: Option<u8>,
        sample_rate_hz: Option<u32>,
    ) -> PyResult<()> {
        self.reload_impl(container_bytes, tier, sample_rate_hz)
    }

    pub fn backend_kind(&self) -> &'static str {
        backend_kind_str(self.inner.backend_kind())
    }

    pub fn active_tier(&self) -> u8 {
        self.inner.tier() as u8
    }

    pub fn set_tier(&mut self, tier: u8) {
        self.inner.set_tier(Tier::from_u8(tier));
    }

    pub fn max_available_tier(&self) -> u8 {
        self.inner.max_available_tier() as u8
    }

    pub fn max_supported_tier(&self) -> u8 {
        self.inner.max_supported_tier() as u8
    }

    pub fn set_max_supported_tier(&mut self, tier: u8) {
        self.inner.set_max_supported_tier(Tier::from_u8(tier));
    }

    pub fn effective_tier(&self) -> u8 {
        self.inner.effective_tier() as u8
    }

    pub fn offline_bounce(&self) -> bool {
        self.inner.offline_bounce()
    }

    pub fn set_offline_bounce(&mut self, enabled: bool) {
        self.inner.set_offline_bounce(enabled);
    }

    pub fn maximize_tier_on_bounce(&self) -> bool {
        self.inner.maximize_tier_on_bounce()
    }

    pub fn set_maximize_tier_on_bounce(&mut self, enabled: bool) {
        self.inner.set_maximize_tier_on_bounce(enabled);
    }

    pub fn latency_samples(&self) -> usize {
        self.inner.latency_samples()
    }

    pub fn sample_rate_hz(&self) -> u32 {
        self.inner.sample_rate_hz()
    }

    pub fn set_sample_rate(&mut self, sample_rate_hz: u32) -> PyResult<()> {
        self.set_sample_rate_impl(sample_rate_hz)
    }

    pub fn num_controls(&self) -> usize {
        self.inner.num_controls()
    }

    pub fn raw_control_values(&self) -> Vec<f32> {
        self.inner.raw_control_values().to_vec()
    }

    pub fn clamped_control_values(&self) -> Vec<f32> {
        self.inner.clamped_control_values().to_vec()
    }

    pub fn clamped_control_value(&self, ctrl_idx: usize) -> PyResult<f32> {
        self.clamped_control_value_impl(ctrl_idx)
    }

    pub fn set_control_value(&mut self, ctrl_idx: usize, value: f32) -> PyResult<()> {
        self.set_control_value_impl(ctrl_idx, value)
    }

    pub fn set_control_immediate(&mut self, ctrl_idx: usize, value: f32) -> PyResult<()> {
        self.set_control_immediate_impl(ctrl_idx, value)
    }

    pub fn set_controls_immediate_batch(&mut self, updates: Vec<(usize, f32)>) -> PyResult<()> {
        self.set_controls_immediate_batch_impl(&updates)
    }

    pub fn reset_controls(&mut self) {
        self.inner.reset_controls_to_defaults();
    }

    pub fn reset(&mut self) {
        self.inner.reset_to_dc_equilibrium();
    }

    pub fn supports_freeze(&self) -> bool {
        self.inner.supports_freeze()
    }

    pub fn set_freeze(&mut self, freeze: bool) {
        self.inner.set_freeze(freeze);
    }

    pub fn set_freeze_amount(&mut self, alpha: f32) {
        self.inner.set_freeze_amount(alpha);
    }

    pub fn is_frozen(&self) -> bool {
        self.inner.is_frozen()
    }

    pub fn set_bypass(&mut self, bypassed: bool, trails: bool) {
        self.inner.set_bypass(bypassed, trails);
    }

    pub fn bypass_state(&self) -> (bool, bool) {
        self.inner.bypass_state()
    }

    pub fn set_stereo_width(&mut self, width: f32) {
        self.inner.set_stereo_width(width);
    }

    pub fn stereo_width(&self) -> f32 {
        self.inner.stereo_width()
    }

    #[pyo3(signature = (blend_wet = 1.0, output_gain_db = 0.0, apply_unity_leveling = true))]
    pub fn configure_host_mix_and_leveling(
        &mut self,
        blend_wet: f32,
        output_gain_db: f32,
        apply_unity_leveling: bool,
    ) {
        self.inner
            .configure_host_mix_and_leveling(blend_wet, output_gain_db, apply_unity_leveling);
    }

    pub fn current_bpm(&self) -> f32 {
        self.inner.current_bpm()
    }

    #[pyo3(signature = (bpm, subdivision = None, fold_octave = false))]
    pub fn sync_tempo(
        &mut self,
        bpm: f32,
        subdivision: Option<&str>,
        fold_octave: bool,
    ) -> PyResult<usize> {
        self.sync_tempo_impl(bpm, subdivision, fold_octave)
    }

    #[pyo3(signature = (subdivision = None, fold_octave = false))]
    pub fn register_tap_tempo(
        &mut self,
        subdivision: Option<&str>,
        fold_octave: bool,
    ) -> PyResult<(f32, usize)> {
        self.register_tap_tempo_impl(subdivision, fold_octave)
    }

    pub fn apply_preset(&mut self, preset_id: &str) -> PyResult<()> {
        self.apply_preset_impl(preset_id)
    }

    pub fn render_offline_bounce<'py>(
        &mut self,
        py: Python<'py>,
        left_in: PyReadonlyArray1<'py, f32>,
        right_in: PyReadonlyArray1<'py, f32>,
        left_out: PyReadwriteArray1<'py, f32>,
        right_out: PyReadwriteArray1<'py, f32>,
    ) -> PyResult<()> {
        self.render_offline_bounce_impl(py, left_in, right_in, left_out, right_out)
    }

    #[pyo3(signature = (left_in, right_in, left_out, right_out, buffer_size = None))]
    pub fn process_stereo<'py>(
        &mut self,
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
        &mut self,
        py: Python<'py>,
        left_io: PyReadwriteArray1<'py, f32>,
        right_io: PyReadwriteArray1<'py, f32>,
        buffer_size: Option<usize>,
    ) -> PyResult<()> {
        self.process_stereo_in_place_impl(py, left_io, right_io, buffer_size)
    }

    #[pyo3(signature = (stereo_in, stereo_out, buffer_size = None))]
    pub fn process_stereo_2d<'py>(
        &mut self,
        py: Python<'py>,
        stereo_in: PyReadonlyArray2<'py, f32>,
        stereo_out: PyReadwriteArray2<'py, f32>,
        buffer_size: Option<usize>,
    ) -> PyResult<()> {
        self.process_stereo_2d_impl(py, stereo_in, stereo_out, buffer_size)
    }

    #[pyo3(signature = (stereo_io, buffer_size = None))]
    pub fn process_stereo_2d_in_place<'py>(
        &mut self,
        py: Python<'py>,
        stereo_io: PyReadwriteArray2<'py, f32>,
        buffer_size: Option<usize>,
    ) -> PyResult<()> {
        self.process_stereo_2d_in_place_impl(py, stereo_io, buffer_size)
    }

    #[pyo3(signature = (mono_in, left_out, right_out, buffer_size = None))]
    pub fn process_mono_to_stereo<'py>(
        &mut self,
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
        &mut self,
        py: Python<'py>,
        mono_in: PyReadonlyArray1<'py, f32>,
        stereo_out: PyReadwriteArray2<'py, f32>,
        buffer_size: Option<usize>,
    ) -> PyResult<()> {
        self.process_mono_to_stereo_2d_impl(py, mono_in, stereo_out, buffer_size)
    }

    #[pyo3(signature = (mono_in, mono_out, buffer_size = None))]
    pub fn process_mono<'py>(
        &mut self,
        py: Python<'py>,
        mono_in: PyReadonlyArray1<'py, f32>,
        mono_out: PyReadwriteArray1<'py, f32>,
        buffer_size: Option<usize>,
    ) -> PyResult<()> {
        self.process_mono_impl(py, mono_in, mono_out, buffer_size)
    }

    pub fn telemetry<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.telemetry_impl(py)
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    Ok(())
}
