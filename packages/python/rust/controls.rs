//! Vectorized 3-Phase control clamping (`PyControlSpace`), ZOH rescaling, and tempo calibration
//! inversion bindings (`ref[sdk.python.controls.space-and-tempo]`).

use dnam_engine::container::ZohRescaledParams;
use dnam_engine::control::{
    CANONICAL_SUBDIVISION_RATIOS, ControlClampDescriptor, control_stride_for_rate,
    control_tick_ratio, eval_3node_calibration_f64, eval_q32_subdivision_phase,
    invert_tempo_physical_calibration, subdivision_q32_to_phasor,
};
use dnam_engine::schema::{
    ControlCalibration, ControlCalibrationRole, ControlUiHints, DnamJsonHeader, TempoSubdivision,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::{dnam_err_to_py, validation_err};

/// Pre-compiled 3-Phase control clamping space (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]`)
/// avoiding per-call JSON header parsing (`ref[sdk.python.controls.space-and-tempo]`).
#[pyclass(module = "dnam._native", frozen)]
#[derive(Debug, Clone)]
pub struct PyControlSpace {
    descriptors: Vec<ControlClampDescriptor>,
    ui_ranges: Vec<(ControlUiHints, f32, f32)>,
    num_controls: usize,
}

#[pymethods]
impl PyControlSpace {
    #[new]
    pub fn new(py: Python<'_>, header_json: &str) -> PyResult<Self> {
        py.detach(|| {
            let header: DnamJsonHeader =
                serde_json::from_str(header_json).map_err(validation_err)?;
            let descriptors = header.compile_clamp_descriptors().map_err(dnam_err_to_py)?;
            let ui_ranges = header
                .controls_schema
                .iter()
                .map(|c| {
                    let [lo, hi] = c.declared_range.unwrap_or([0.0, 10.0]);
                    (c.ui_hints.clone(), lo, hi)
                })
                .collect();
            let num_controls = descriptors.len();
            Ok(Self {
                descriptors,
                ui_ranges,
                num_controls,
            })
        })
    }

    pub fn num_controls(&self) -> usize {
        self.num_controls
    }

    /// Evaluates 3-Phase control clamping over `raw_numeric_values` in `< 100 ns` without JSON parsing.
    /// Returns `(clamped_numeric, normalized_u, was_clamped, active_clamp_mask)`.
    pub fn clamp(
        &self,
        py: Python<'_>,
        raw_numeric_values: Vec<f32>,
    ) -> (Vec<f32>, Vec<f32>, Vec<bool>, u16) {
        py.detach(|| {
            let n = self.num_controls;
            let mut raw_buf = vec![0.0_f32; n];
            for p in 0..n {
                raw_buf[p] = raw_numeric_values
                    .get(p)
                    .copied()
                    .unwrap_or(self.descriptors[p].default_val);
            }
            let mut active_switches = vec![0_u8; n];
            let mut clamped_out = vec![0.0_f32; n];
            let mut norm_out = vec![0.0_f32; n];
            let clamp_mask = ControlClampDescriptor::clamp_and_normalize_all(
                &self.descriptors,
                n,
                &raw_buf,
                &mut clamped_out,
                &mut norm_out,
                &mut active_switches,
            );
            let mut was_clamped_out = vec![false; n];
            for p in 0..n {
                was_clamped_out[p] = if p < 16 {
                    (clamp_mask & (1_u16 << p)) != 0
                } else {
                    (clamped_out[p] - raw_buf[p]).abs() > 1e-6
                };
            }
            (clamped_out, norm_out, was_clamped_out, clamp_mask)
        })
    }

    /// Maps a normalized UI gesture `t in [0.0, 1.0]` to physical value for control `idx`.
    pub fn to_physical(&self, idx: usize, normalized_gesture: f32) -> PyResult<f32> {
        let (hints, lo, hi) = self
            .ui_ranges
            .get(idx)
            .ok_or_else(|| validation_err(format!("control index {idx} out of bounds")))?;
        Ok(hints.normalized_to_physical(normalized_gesture, *lo, *hi))
    }

    /// Maps a physical value `x in [x_min, x_max]` to normalized UI gesture `t in [0.0, 1.0]` for control `idx`.
    pub fn to_normalized(&self, idx: usize, physical_value: f32) -> PyResult<f32> {
        let (hints, lo, hi) = self
            .ui_ranges
            .get(idx)
            .ok_or_else(|| validation_err(format!("control index {idx} out of bounds")))?;
        Ok(hints.physical_to_normalized(physical_value, *lo, *hi))
    }
}

/// Computes the UI power-skew exponent $\gamma = \ln(0.5)/\ln((x_{\text{mid}}-x_{\min})/(x_{\max}-x_{\min}))$ in Rust.
#[pyfunction]
#[pyo3(signature = (midpoint, x_min, x_max))]
pub fn native_ui_skew_exponent(midpoint: Option<f32>, x_min: f32, x_max: f32) -> f64 {
    let hints = ControlUiHints {
        midpoint,
        ..Default::default()
    };
    hints.skew_exponent(x_min, x_max)
}

/// Maps normalized UI gesture $t \in [0.0, 1.0]$ to physical value $x \in [x_{\min}, x_{\max}]$ in Rust.
#[pyfunction]
#[pyo3(signature = (midpoint, t, x_min, x_max))]
pub fn native_ui_normalized_to_physical(
    midpoint: Option<f32>,
    t: f32,
    x_min: f32,
    x_max: f32,
) -> f32 {
    let hints = ControlUiHints {
        midpoint,
        ..Default::default()
    };
    hints.normalized_to_physical(t, x_min, x_max)
}

/// Maps physical value $x \in [x_{\min}, x_{\max}]$ to normalized UI gesture $t \in [0.0, 1.0]$ in Rust.
#[pyfunction]
#[pyo3(signature = (midpoint, x_val, x_min, x_max))]
pub fn native_ui_physical_to_normalized(
    midpoint: Option<f32>,
    x_val: f32,
    x_min: f32,
    x_max: f32,
) -> f32 {
    let hints = ControlUiHints {
        midpoint,
        ..Default::default()
    };
    hints.physical_to_normalized(x_val, x_min, x_max)
}

/// Inverts a 3-point Lobatto quadratic $v(u) = v_{\text{mid}} + c_1 u + c_2 u^2$ ($u \in [-1, 1]$) in Rust.
#[pyfunction]
#[pyo3(signature = (anchors, target_val, fold_octave = true))]
pub fn native_invert_lobatto_anchors(
    anchors: [f32; 3],
    target_val: f32,
    fold_octave: bool,
) -> (f32, bool) {
    ControlCalibration::invert_lobatto_anchors(&anchors, target_val, fold_octave)
}

/// Evaluates the 3-point Lobatto quadratic $v(u) = v_{\text{mid}} + c_1 u + c_2 u^2$ ($u \in [-1, 1]$) in Rust.
#[pyfunction]
pub fn native_evaluate_lobatto_anchors(anchors: [f32; 3], u: f32) -> f32 {
    let cal = ControlCalibration {
        role: ControlCalibrationRole::DelayTimeMs,
        lobatto_physical_anchors: Some(anchors),
        conditional_physical_anchors: std::collections::BTreeMap::new(),
        tempo_sync_default_subdivision: None,
    };
    cal.evaluate_lobatto_anchors(u).unwrap_or(0.0)
}

/// `(switch_selection, normalized_u, was_clamped)` tuple returned by [`native_resolve_tempo_sync`].
pub type TempoSyncTuple = (Option<(String, String)>, f32, bool);

/// Resolves a MIDI/Host `(bpm, subdivision)` into `(switch_selection, normalized_u, was_clamped)` in Rust.
#[pyfunction]
#[pyo3(signature = (calibration_json, bpm, subdivision = None, fold_octave = true))]
pub fn native_resolve_tempo_sync(
    py: Python<'_>,
    calibration_json: &str,
    bpm: f32,
    subdivision: Option<&str>,
    fold_octave: bool,
) -> PyResult<Option<TempoSyncTuple>> {
    py.detach(|| {
        let cal: ControlCalibration =
            serde_json::from_str(calibration_json).map_err(validation_err)?;
        let sub = match subdivision {
            Some(s) => Some(
                TempoSubdivision::from_str_label(s)
                    .ok_or_else(|| validation_err(format!("Unknown tempo subdivision '{s}'")))?,
            ),
            None => None,
        };
        Ok(cal
            .resolve_tempo_sync(bpm, sub, fold_octave)
            .map(|r| (r.switch_selection, r.normalized_u, r.was_clamped)))
    })
}

/// Returns the quarter-note beat multiplier $m_{\text{sub}}$ for `subdivision` in Rust.
#[pyfunction]
pub fn native_subdivision_beat_multiplier(subdivision: &str) -> PyResult<f32> {
    let sub = TempoSubdivision::from_str_label(subdivision)
        .ok_or_else(|| validation_err(format!("Unknown tempo subdivision '{subdivision}'")))?;
    Ok(sub.beat_multiplier())
}

/// Returns the target delay / grain duration in milliseconds at `bpm` for `subdivision` in Rust.
#[pyfunction]
pub fn native_subdivision_delay_ms_at_bpm(subdivision: &str, bpm: f32) -> PyResult<f32> {
    let sub = TempoSubdivision::from_str_label(subdivision)
        .ok_or_else(|| validation_err(format!("Unknown tempo subdivision '{subdivision}'")))?;
    Ok(sub.delay_ms_at_bpm(bpm))
}

/// Returns the target LFO frequency in Hz at `bpm` for `subdivision` in Rust.
#[pyfunction]
pub fn native_subdivision_lfo_hz_at_bpm(subdivision: &str, bpm: f32) -> PyResult<f32> {
    let sub = TempoSubdivision::from_str_label(subdivision)
        .ok_or_else(|| validation_err(format!("Unknown tempo subdivision '{subdivision}'")))?;
    Ok(sub.lfo_hz_at_bpm(bpm))
}

/// Computes Continuous-Time Zero-Order Hold (ZOH) rescaled parameters for `sample_rate_host_hz`
/// (`ref[dsp.zoh.ctrl-stride]`).
#[pyfunction]
#[pyo3(signature = (sample_rate_train_hz, sample_rate_host_hz, r_ac_train = 0.9993, r_dc_train = 0.995, tau_ref_train = 64.0, switch_xfade_train = 128))]
pub fn native_compute_zoh<'py>(
    py: Python<'py>,
    sample_rate_train_hz: u32,
    sample_rate_host_hz: u32,
    r_ac_train: f32,
    r_dc_train: f32,
    tau_ref_train: f32,
    switch_xfade_train: u32,
) -> PyResult<Bound<'py, PyDict>> {
    let zoh = py
        .detach(|| {
            ZohRescaledParams::compute(
                sample_rate_train_hz,
                sample_rate_host_hz,
                r_ac_train,
                r_dc_train,
                tau_ref_train,
                switch_xfade_train,
            )
        })
        .map_err(dnam_err_to_py)?;

    let dict = PyDict::new(py);
    dict.set_item("sample_rate_host_hz", zoh.sample_rate_host_hz)?;
    dict.set_item("rho_sr", zoh.rho_sr)?;
    dict.set_item("control_decimation_host", zoh.control_decimation_host)?;
    dict.set_item("rho_ctrl", zoh.rho_ctrl)?;
    dict.set_item("alpha_ramp_host", zoh.alpha_ramp_host)?;
    dict.set_item("r_ac_host", zoh.r_ac_host)?;
    dict.set_item("r_dc_host", zoh.r_dc_host)?;
    dict.set_item("tau_ref_samples_host", zoh.tau_ref_samples_host)?;
    dict.set_item("switch_xfade_samples_host", zoh.switch_xfade_samples_host)?;
    Ok(dict)
}

/// Returns the control-bus decimation stride $K_{\text{ctrl}}(f_s) = \max(8, \lfloor(f_s + 1500)/3000\rfloor)$.
#[pyfunction]
pub fn native_control_stride_for_rate(sample_rate_hz: u32) -> u32 {
    control_stride_for_rate(sample_rate_hz)
}

/// Returns the control-tick ratio $\rho_{\text{ctrl}}(f_s)$.
#[pyfunction]
pub fn native_control_tick_ratio(sample_rate_hz: u32) -> f64 {
    control_tick_ratio(sample_rate_hz)
}

/// Evaluates the `Q0.32` subdivision phase and unit phasor `(phase_q32, cos_phi, sin_phi)`
/// for `(beat_mod12, within_beat_q32, subdivision_str, offset_q32)`.
#[pyfunction]
#[pyo3(signature = (beat_mod12, within_beat_q32, subdivision, offset_q32 = 0))]
pub fn native_eval_subdivision_phase(
    beat_mod12: u8,
    within_beat_q32: u32,
    subdivision: &str,
    offset_q32: u32,
) -> PyResult<(u32, f32, f32)> {
    let sub = TempoSubdivision::from_str_label(subdivision)
        .ok_or_else(|| validation_err(format!("Unknown tempo subdivision '{subdivision}'")))?;
    let idx = TempoSubdivision::ALL
        .iter()
        .position(|&s| s == sub)
        .unwrap_or(4);
    let ratio = CANONICAL_SUBDIVISION_RATIOS[idx];
    let q32 = eval_q32_subdivision_phase(beat_mod12, within_beat_q32, ratio, offset_q32)
        .ok_or_else(|| {
            validation_err(format!("Invalid beat_mod12={beat_mod12} (expected 0..12)"))
        })?;
    let (cos_phi, sin_phi) = subdivision_q32_to_phasor(q32);
    Ok((q32, cos_phi, sin_phi))
}

/// Evaluates the 3-node physical calibration curve $v(u)$ at $u \in [-1, 1]$.
#[pyfunction]
pub fn native_eval_3node_calibration(anchors: [f64; 3], u: f64) -> f64 {
    eval_3node_calibration_f64(anchors, u)
}

/// Inverts a monotone 3-node physical calibration curve `anchors = [v_min, v_mid, v_max]` over the
/// admissible `f32` physical rank interval for `declared_range` and `safe_range` (`ref[dsp.tempo.physical-inversion]`).
#[pyfunction]
#[pyo3(signature = (declared_range, safe_range, anchors, target, flush_subnormals = true))]
pub fn native_invert_tempo_calibration<'py>(
    py: Python<'py>,
    declared_range: (f32, f32),
    safe_range: (f32, f32),
    anchors: [f64; 3],
    target: f64,
    flush_subnormals: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let res = py
        .detach(|| {
            invert_tempo_physical_calibration(
                declared_range,
                safe_range,
                anchors,
                target,
                flush_subnormals,
            )
        })
        .ok_or_else(|| validation_err("invalid calibration range or non-finite target"))?;
    let dict = PyDict::new(py);
    dict.set_item("physical_f32", res.physical_f32)?;
    dict.set_item("normalized_f32", res.normalized_f32)?;
    dict.set_item("realized_value", res.realized_value)?;
    dict.set_item("residual", res.residual)?;
    dict.set_item("search_steps", res.search_steps)?;
    Ok(dict)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyControlSpace>()?;
    m.add_function(wrap_pyfunction!(native_ui_skew_exponent, m)?)?;
    m.add_function(wrap_pyfunction!(native_ui_normalized_to_physical, m)?)?;
    m.add_function(wrap_pyfunction!(native_ui_physical_to_normalized, m)?)?;
    m.add_function(wrap_pyfunction!(native_invert_lobatto_anchors, m)?)?;
    m.add_function(wrap_pyfunction!(native_evaluate_lobatto_anchors, m)?)?;
    m.add_function(wrap_pyfunction!(native_resolve_tempo_sync, m)?)?;
    m.add_function(wrap_pyfunction!(native_subdivision_beat_multiplier, m)?)?;
    m.add_function(wrap_pyfunction!(native_subdivision_delay_ms_at_bpm, m)?)?;
    m.add_function(wrap_pyfunction!(native_subdivision_lfo_hz_at_bpm, m)?)?;
    m.add_function(wrap_pyfunction!(native_compute_zoh, m)?)?;
    m.add_function(wrap_pyfunction!(native_control_stride_for_rate, m)?)?;
    m.add_function(wrap_pyfunction!(native_control_tick_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(native_eval_subdivision_phase, m)?)?;
    m.add_function(wrap_pyfunction!(native_eval_3node_calibration, m)?)?;
    m.add_function(wrap_pyfunction!(native_invert_tempo_calibration, m)?)?;
    Ok(())
}
