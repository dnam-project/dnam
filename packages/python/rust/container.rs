//! Container validation, building, 4 KB in-place header update, 3-Phase control clamping,
//! ZOH rescaling, `uplo='U'` Gram packing, and hardware Castagnoli `CRC32C` FFI bindings.

use std::collections::BTreeMap;

use dnam_engine::container::{
    ARENA_OFFSET_ULTRA_END, DnamContainerView, ZohRescaledParams, build_dnam_container,
    crc32c_compute, pack_uplo_u_f64, unpack_uplo_u_f64, update_header_in_place, uplo_u_packed_len,
};
use dnam_engine::control::{ControlClampDescriptor, MAX_CONTROLS};
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::schema::{
    ControlKind, ControlSchemaDef, ControlUiHints, DnamJsonHeader, PresetDef,
};
use numpy::{PyArray1, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use crate::dnam_err_to_py;

/// Pre-compiled 3-Phase control clamping space (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]`)
/// avoiding per-call JSON header parsing.
#[pyclass(module = "dnam_engine._native", frozen)]
#[derive(Debug, Clone)]
pub struct NativeControlSpace {
    descriptors: [ControlClampDescriptor; MAX_CONTROLS],
    num_controls: usize,
}

#[pymethods]
impl NativeControlSpace {
    #[new]
    pub fn new(header_json: &str) -> PyResult<Self> {
        let header: DnamJsonHeader = serde_json::from_str(header_json)
            .map_err(|e| PyValueError::new_err(format!("[DNAM-VALIDATION] {e}")))?;
        let descriptors = header.compile_clamp_descriptors().map_err(dnam_err_to_py)?;
        let num_controls = header.controls_schema.len().min(MAX_CONTROLS);
        Ok(Self {
            descriptors,
            num_controls,
        })
    }

    /// Evaluates 3-Phase control clamping over `raw_numeric_values` in `< 100 ns` without JSON parsing.
    /// Returns `(clamped_numeric, normalized_u, was_clamped)`.
    pub fn clamp(&self, raw_numeric_values: Vec<f32>) -> (Vec<f32>, Vec<f32>, Vec<bool>) {
        let n = self.num_controls;
        let mut active_switches = [0_u8; MAX_CONTROLS];
        let mut clamped_out = vec![0.0_f32; n];
        let mut norm_out = vec![0.0_f32; n];
        let mut was_clamped_out = vec![false; n];

        // Pass 1: Switch controls first so conditional range rules see active switch states
        for p in 0..n {
            let desc = &self.descriptors[p];
            if desc.ctrl_kind == 1 {
                let raw_v = raw_numeric_values
                    .get(p)
                    .copied()
                    .unwrap_or(desc.default_val);
                let (clamped, norm, was_clamped) =
                    desc.clamp_and_normalize(raw_v, &active_switches);
                active_switches[p] = (clamped + 0.5) as u8;
                clamped_out[p] = clamped;
                norm_out[p] = norm;
                was_clamped_out[p] = was_clamped;
            }
        }
        // Pass 2: Range controls with switch-conditioned rules active
        for p in 0..n {
            let desc = &self.descriptors[p];
            if desc.ctrl_kind == 0 {
                let raw_v = raw_numeric_values
                    .get(p)
                    .copied()
                    .unwrap_or(desc.default_val);
                let (clamped, norm, was_clamped) =
                    desc.clamp_and_normalize(raw_v, &active_switches);
                clamped_out[p] = clamped;
                norm_out[p] = norm;
                was_clamped_out[p] = was_clamped;
            }
        }
        (clamped_out, norm_out, was_clamped_out)
    }
}

/// Parses and validates a `.dnam` or `.dnam-ckpt` container, returning preamble fields,
/// canonical JSON header string, verification status, and arena byte offset/length.
#[pyfunction]
pub fn native_validate_container<'py>(
    py: Python<'py>,
    container_bytes: &[u8],
) -> PyResult<Bound<'py, PyDict>> {
    let crypto = DefaultCryptoProvider::default();
    let view = DnamContainerView::from_bytes(container_bytes, &crypto).map_err(dnam_err_to_py)?;
    let p = &view.preamble;
    let header_json = serde_json::to_string(&view.header)
        .map_err(|e| PyValueError::new_err(format!("[DNAM-VALIDATION] {e}")))?;

    let dict = PyDict::new(py);
    let magic_str = core::str::from_utf8(&p.magic).unwrap_or("DNAM");
    dict.set_item("magic", magic_str)?;
    dict.set_item("version_major", p.version_major)?;
    dict.set_item("version_minor", p.version_minor)?;
    dict.set_item("flags", p.flags)?;
    dict.set_item("num_controls", p.num_controls)?;
    dict.set_item("ring_buffer_pow2_log2", p.ring_buffer_pow2_log2)?;
    dict.set_item("sample_rate_train_hz", p.sample_rate_train_hz)?;
    dict.set_item("json_header_byte_len", p.json_header_byte_len)?;
    dict.set_item("tensor_arena_offset", p.tensor_arena_offset)?;
    dict.set_item("tensor_arena_byte_len", p.tensor_arena_byte_len)?;
    dict.set_item("arena_offset_lite_end", p.arena_offset_lite_end)?;
    dict.set_item("arena_offset_full_end", p.arena_offset_full_end)?;
    dict.set_item("arena_offset_ultra_end", p.arena_offset_ultra_end)?;
    dict.set_item("arena_crc32c_lite", p.arena_crc32c_lite)?;
    dict.set_item("arena_crc32c_full", p.arena_crc32c_full)?;
    dict.set_item("arena_crc32c_ultra", p.arena_crc32c_ultra)?;
    dict.set_item("num_ckpt_stems", p.num_ckpt_stems)?;
    dict.set_item("control_decimation", p.control_decimation)?;
    dict.set_item("head_svd_rank", p.head_svd_rank)?;
    dict.set_item("tail_time_ms", p.tail_time_ms)?;
    dict.set_item("header_crc32c", p.header_crc32c)?;
    dict.set_item(
        "verification_status",
        format!("{:?}", view.verification_status),
    )?;
    dict.set_item("header_json", header_json)?;
    dict.set_item(
        "arena_bytes",
        PyBytes::new(py, view.arena_storage.as_bytes()),
    )?;
    Ok(dict)
}

/// Builds a 4 KB page-aligned `.dnam` or `.dnam-ckpt` container from `header_json` and `arena_bytes`.
#[pyfunction]
#[pyo3(signature = (header_json, arena_bytes, plaintext_arena_crc_source = None, encrypted_arena = false))]
pub fn native_build_container<'py>(
    py: Python<'py>,
    header_json: &str,
    arena_bytes: &[u8],
    plaintext_arena_crc_source: Option<&[u8]>,
    encrypted_arena: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    let mut header: DnamJsonHeader = serde_json::from_str(header_json)
        .map_err(|e| PyValueError::new_err(format!("[DNAM-VALIDATION] {e}")))?;
    header.validate().map_err(dnam_err_to_py)?;

    // Compile binary clamp descriptors into arena[7040..8064] if not encrypted and currently unpopulated
    let mut stored_arena = arena_bytes.to_vec();
    if !encrypted_arena
        && stored_arena.len() >= 8064
        && !header.controls_schema.is_empty()
        && stored_arena[7040..7044] == [0, 0, 0, 0]
        && let Ok(descriptors) = header.compile_clamp_descriptors()
    {
        for (idx, desc) in descriptors.iter().enumerate() {
            let off = 7040 + idx * 64;
            stored_arena[off..off + 64].copy_from_slice(&desc.to_bytes());
        }
    }

    let crc_src = plaintext_arena_crc_source.unwrap_or(&stored_arena);
    let container = build_dnam_container(&mut header, crc_src, &stored_arena, encrypted_arena)
        .map_err(dnam_err_to_py)?;
    Ok(PyBytes::new(py, &container))
}

/// Synthesizes a canonical reference `.dnam` (or `.dnam-ckpt`) container with built-in `"gain"` (`range`)
/// and `"boost"` (`switch`) controls, a `"warm_slapback"` preset, and calibrated dry/wet pass-through weights.
#[pyfunction]
#[pyo3(signature = (core2_active = false, is_checkpoint = false))]
pub fn native_build_reference_container<'py>(
    py: Python<'py>,
    core2_active: bool,
    is_checkpoint: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    let mut header = DnamJsonHeader::default_reference(core2_active);
    if is_checkpoint {
        header.container_kind = "dnam-ckpt".to_string();
    }

    let mut cond_max = BTreeMap::new();
    cond_max.insert("boost=on".to_string(), 8.0_f32);

    header.controls_schema = vec![
        ControlSchemaDef {
            id: "gain".to_string(),
            name: "Drive".to_string(),
            kind: ControlKind::Range,
            unit: Some("dB".to_string()),
            lobatto_degree: Some(2),
            anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
            declared_range: Some([0.0, 10.0]),
            safe_clamped_range: Some([0.0, 9.5]),
            default: serde_json::json!(5.0),
            conditional_min: BTreeMap::new(),
            conditional_max: cond_max,
            declared_options: None,
            active_options: None,
            clamp_warning: Some(
                "[CLAMP-RANGE] Clamped at 9.5 dB (8.0 dB when boost=on)".to_string(),
            ),
            ui_hints: ControlUiHints {
                section: Some("Preamp".to_string()),
                order: 0,
                widget: Some("knob".to_string()),
                size: Some("lg".to_string()),
                polarity: Some("unipolar".to_string()),
                midpoint: Some(4.0),
                detents: vec![5.0],
                step: Some(0.1),
                decimals: Some(1),
                scale_labels: BTreeMap::new(),
                option_labels: BTreeMap::new(),
                color_hex: Some("#F59E0B".to_string()),
            },
            calibration: None,
            user_metadata: BTreeMap::new(),
        },
        ControlSchemaDef {
            id: "boost".to_string(),
            name: "Boost".to_string(),
            kind: ControlKind::Switch,
            unit: None,
            lobatto_degree: None,
            anchor_nodes_norm: None,
            declared_range: None,
            safe_clamped_range: None,
            default: serde_json::json!("off"),
            conditional_min: BTreeMap::new(),
            conditional_max: BTreeMap::new(),
            declared_options: Some(vec![
                "off".to_string(),
                "on".to_string(),
                "ultra".to_string(),
            ]),
            active_options: Some(vec!["off".to_string(), "on".to_string()]),
            clamp_warning: Some("[CLAMP-SWITCH] Option 'ultra' unobserved".to_string()),
            ui_hints: ControlUiHints {
                section: Some("Preamp".to_string()),
                order: 1,
                widget: Some("toggle".to_string()),
                ..Default::default()
            },
            calibration: None,
            user_metadata: BTreeMap::new(),
        },
    ];

    let mut preset_controls = BTreeMap::new();
    preset_controls.insert("gain".to_string(), serde_json::json!(6.5));
    preset_controls.insert("boost".to_string(), serde_json::json!("on"));
    header.presets = vec![PresetDef {
        id: "warm_slapback".to_string(),
        name: "Warm Slapback".to_string(),
        description: Some("DNAM Studio preset".to_string()),
        controls: preset_controls,
        user_metadata: BTreeMap::new(),
    }];

    let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    // Set dry/wet mix gain in Arena 0 CL1 (offset 116 = mix_dry_gain, 120 = mix_wet_gain, 124 = r_ac_train)
    arena[116..120].copy_from_slice(&0.5_f32.to_le_bytes());
    arena[120..124].copy_from_slice(&0.5_f32.to_le_bytes());
    arena[124..128].copy_from_slice(&0.9993_f32.to_le_bytes());
    // Set non-zero input projection & output head tap so wet path is active
    arena[0..4].copy_from_slice(&1.0_f32.to_le_bytes());
    arena[20..24].copy_from_slice(&1.0_f32.to_le_bytes());
    arena[4224..4228].copy_from_slice(&0.25_f32.to_le_bytes());
    arena[4480..4484].copy_from_slice(&0.25_f32.to_le_bytes());

    let descriptors = header.compile_clamp_descriptors().map_err(dnam_err_to_py)?;
    for (idx, desc) in descriptors.iter().enumerate() {
        let off = 7040 + idx * 64;
        arena[off..off + 64].copy_from_slice(&desc.to_bytes());
    }

    let container =
        build_dnam_container(&mut header, &arena, &arena, false).map_err(dnam_err_to_py)?;
    Ok(PyBytes::new(py, &container))
}

/// Updates the JSON header in-place inside the 4 KB page-aligned region (`0x40..tensor_arena_offset`).
#[pyfunction]
pub fn native_update_header_in_place<'py>(
    py: Python<'py>,
    container_bytes: &[u8],
    updated_header_json: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let updated_header: DnamJsonHeader = serde_json::from_str(updated_header_json)
        .map_err(|e| PyValueError::new_err(format!("[DNAM-VALIDATION] {e}")))?;
    updated_header.validate().map_err(dnam_err_to_py)?;
    let mut buf = container_bytes.to_vec();
    update_header_in_place(&mut buf, &updated_header).map_err(dnam_err_to_py)?;
    Ok(PyBytes::new(py, &buf))
}

/// Evaluates 3-Phase control clamping (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]`)
/// for a `controls_schema` JSON array and a list of `(control_id, raw_float_or_switch_idx)` pairs.
#[pyfunction]
pub fn native_clamp_controls<'py>(
    py: Python<'py>,
    header_json: &str,
    raw_numeric_values: Vec<f32>,
) -> PyResult<Bound<'py, PyList>> {
    let header: DnamJsonHeader = serde_json::from_str(header_json)
        .map_err(|e| PyValueError::new_err(format!("[DNAM-VALIDATION] {e}")))?;
    let descriptors = header.compile_clamp_descriptors().map_err(dnam_err_to_py)?;
    let num_controls = header.controls_schema.len();

    // Pass 1: Clamp switch controls first so conditional range bounds see active switch states
    let mut active_switches = [0_u8; MAX_CONTROLS];
    let mut intermediate = [(0.0_f32, 0.0_f32, false); MAX_CONTROLS];
    for p in 0..num_controls {
        let raw_v = raw_numeric_values
            .get(p)
            .copied()
            .unwrap_or(descriptors[p].default_val);
        if descriptors[p].ctrl_kind == 1 {
            let (clamped, norm, was_clamped) =
                descriptors[p].clamp_and_normalize(raw_v, &active_switches);
            active_switches[p] = (clamped + 0.5) as u8;
            intermediate[p] = (clamped, norm, was_clamped);
        }
    }
    // Pass 2: Clamp range controls with switch-conditioned rules active
    for p in 0..num_controls {
        if descriptors[p].ctrl_kind == 0 {
            let raw_v = raw_numeric_values
                .get(p)
                .copied()
                .unwrap_or(descriptors[p].default_val);
            intermediate[p] = descriptors[p].clamp_and_normalize(raw_v, &active_switches);
        }
    }

    let list = PyList::empty(py);
    for (p, ctrl) in header.controls_schema.iter().enumerate() {
        let (clamped_f, u_norm, was_clamped) = intermediate[p];
        let entry = PyDict::new(py);
        entry.set_item("id", &ctrl.id)?;
        entry.set_item("clamped_numeric", clamped_f)?;
        entry.set_item("normalized_u", u_norm)?;
        entry.set_item("was_clamped", was_clamped)?;
        entry.set_item("clamp_warning", ctrl.clamp_warning.as_deref())?;
        list.append(entry)?;
    }
    Ok(list)
}

/// Computes Continuous-Time Zero-Order Hold (ZOH) rescaled parameters for `sample_rate_host_hz`.
#[pyfunction]
#[pyo3(signature = (sample_rate_train_hz, sample_rate_host_hz, control_decimation_train = 16, r_ac_train = 0.9993, r_dc_train = 0.995, tau_ref_train = 64.0, switch_xfade_train = 128))]
#[allow(clippy::too_many_arguments)]
pub fn native_compute_zoh<'py>(
    py: Python<'py>,
    sample_rate_train_hz: u32,
    sample_rate_host_hz: u32,
    control_decimation_train: u8,
    r_ac_train: f32,
    r_dc_train: f32,
    tau_ref_train: f32,
    switch_xfade_train: u32,
) -> PyResult<Bound<'py, PyDict>> {
    let zoh = ZohRescaledParams::compute(
        sample_rate_train_hz,
        sample_rate_host_hz,
        control_decimation_train,
        r_ac_train,
        r_dc_train,
        tau_ref_train,
        switch_xfade_train,
    )
    .map_err(dnam_err_to_py)?;

    let dict = PyDict::new(py);
    dict.set_item("sample_rate_host_hz", zoh.sample_rate_host_hz)?;
    dict.set_item("rho_sr", zoh.rho_sr)?;
    dict.set_item("control_decimation_host", zoh.control_decimation_host)?;
    dict.set_item("rho_ctrl", zoh.rho_ctrl)?;
    dict.set_item("r_ac_host", zoh.r_ac_host)?;
    dict.set_item("r_dc_host", zoh.r_dc_host)?;
    dict.set_item("tau_ref_samples_host", zoh.tau_ref_samples_host)?;
    dict.set_item("switch_xfade_samples_host", zoh.switch_xfade_samples_host)?;
    Ok(dict)
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
        return Err(PyValueError::new_err(format!(
            "[DNAM-VALIDATION] Expected {expected} elements for {n_dim}x{n_dim} matrix, got {}",
            dense_flat.len()
        )));
    }
    let slice = dense_flat
        .as_slice()
        .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] dense_flat must be contiguous"))?;
    let packed_len = uplo_u_packed_len(n_dim);
    let mut out = vec![0.0_f64; packed_len];
    pack_uplo_u_f64(slice, n_dim, &mut out);
    Ok(PyArray1::from_vec(py, out))
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
        return Err(PyValueError::new_err(format!(
            "[DNAM-VALIDATION] Packed buffer too short for n_tier={n_tier}: need {req_len}, got {}",
            packed_u.len()
        )));
    }
    let slice = packed_u
        .as_slice()
        .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] packed_u must be contiguous"))?;
    let mut out = vec![0.0_f64; n_tier * n_tier];
    unpack_uplo_u_f64(slice, n_tier, &mut out);
    Ok(PyArray1::from_vec(py, out))
}

/// Computes the Castagnoli `CRC32C` checksum over `data` using hardware `CRC32C` instructions when available.
#[pyfunction]
pub fn native_crc32c(data: &[u8]) -> u32 {
    crc32c_compute(data)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeControlSpace>()?;
    m.add_function(wrap_pyfunction!(native_validate_container, m)?)?;
    m.add_function(wrap_pyfunction!(native_build_container, m)?)?;
    m.add_function(wrap_pyfunction!(native_build_reference_container, m)?)?;
    m.add_function(wrap_pyfunction!(native_update_header_in_place, m)?)?;
    m.add_function(wrap_pyfunction!(native_clamp_controls, m)?)?;
    m.add_function(wrap_pyfunction!(native_compute_zoh, m)?)?;
    m.add_function(wrap_pyfunction!(native_pack_uplo_u_f64, m)?)?;
    m.add_function(wrap_pyfunction!(native_unpack_uplo_u_f64, m)?)?;
    m.add_function(wrap_pyfunction!(native_crc32c, m)?)?;
    Ok(())
}
