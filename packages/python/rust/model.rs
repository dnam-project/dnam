//! `.dnam` model container validation, building, 4 KB in-place header updates, tier prefix
//! stripping, JCS canonicalization, `DMU1` mutable trailers, and hardware `CRC32C`
//! (`ref[sdk.python.model.containers-and-checkpoints]`).

mod checkpoint;
mod gram;

use std::collections::BTreeMap;

use dnam_engine::Tier;
use dnam_engine::container::{
    ARENA_OFFSET_ULTRA_END, DnamContainerView, build_dmu1_mutable_frame, build_dnam_container,
    crc32c_compute, parse_and_validate_dmu1_frame, project_immutable_header_value,
    serialize_jcs_value, strip_container_to_tier, strip_v2_open_dnam_container,
    update_header_in_place, validate_v2_open_dnam_layout,
};
use dnam_engine::provenance::{RustCryptoProvider, decode_hex_array};
use dnam_engine::schema::{
    ControlCalibration, ControlCalibrationRole, ControlKind, ControlSchemaDef, ControlUiHints,
    DnamJsonHeader, PresetDef, TempoSubdivision,
};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::{dnam_err_to_py, validation_err};

pub(crate) fn populate_reference_controls_and_arena(
    header: &mut DnamJsonHeader,
    supports_freeze: bool,
) -> PyResult<Vec<u8>> {
    header.io_config.capabilities.supports_freeze = supports_freeze;
    header.io_config.capabilities.supports_trails = true;
    header.io_config.capabilities.tempo_sync_capable = true;

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
        ControlSchemaDef {
            id: "delay_ms".to_string(),
            name: "Delay Time".to_string(),
            kind: ControlKind::Range,
            unit: Some("ms".to_string()),
            lobatto_degree: Some(2),
            anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
            declared_range: Some([20.0, 1000.0]),
            safe_clamped_range: Some([20.0, 1000.0]),
            default: serde_json::json!(250.0),
            conditional_min: BTreeMap::new(),
            conditional_max: BTreeMap::new(),
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            ui_hints: ControlUiHints {
                section: Some("Spatial Tank".to_string()),
                order: 2,
                widget: Some("knob".to_string()),
                ..Default::default()
            },
            calibration: Some(ControlCalibration {
                role: ControlCalibrationRole::DelayTimeMs,
                lobatto_physical_anchors: Some([20.0, 350.0, 1000.0]),
                conditional_physical_anchors: BTreeMap::new(),
                tempo_sync_default_subdivision: Some(TempoSubdivision::Eighth),
            }),
            user_metadata: BTreeMap::new(),
        },
    ];

    let mut preset_controls = BTreeMap::new();
    preset_controls.insert("gain".to_string(), serde_json::json!(6.5));
    preset_controls.insert("boost".to_string(), serde_json::json!("on"));
    header.presets = vec![PresetDef {
        id: "warm_slapback".to_string(),
        name: "Warm Slapback".to_string(),
        description: Some("DNAM Studio reference preset".to_string()),
        controls: preset_controls,
        user_metadata: BTreeMap::new(),
    }];

    let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    arena[116..120].copy_from_slice(&0.5_f32.to_le_bytes());
    arena[120..124].copy_from_slice(&0.5_f32.to_le_bytes());
    arena[124..128].copy_from_slice(&0.9993_f32.to_le_bytes());
    arena[0..4].copy_from_slice(&1.0_f32.to_le_bytes());
    arena[20..24].copy_from_slice(&1.0_f32.to_le_bytes());
    arena[4224..4228].copy_from_slice(&0.25_f32.to_le_bytes());
    arena[4480..4484].copy_from_slice(&0.25_f32.to_le_bytes());

    let descriptors = header.compile_clamp_descriptors().map_err(dnam_err_to_py)?;
    for (idx, desc) in descriptors.iter().enumerate() {
        let off = 8_512 + idx * 64;
        if off + 64 <= arena.len() {
            arena[off..off + 64].copy_from_slice(&desc.to_bytes());
        }
    }
    Ok(arena)
}

/// Parses and validates a `.dnam`, `.dnam-ckpt`, or `.dnam-seal` (`b"DNSL"`) container, returning
/// preamble fields, canonical JSON header string, verification status, protection profile, and arena bytes.
#[pyfunction]
#[pyo3(signature = (container_bytes, device_id_hash = None, device_secret_32 = None, trusted_marketplace_pubkey_hex = None))]
pub fn native_validate_container<'py>(
    py: Python<'py>,
    container_bytes: &[u8],
    device_id_hash: Option<&str>,
    device_secret_32: Option<&[u8]>,
    trusted_marketplace_pubkey_hex: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut crypto = RustCryptoProvider::new();
    if let (Some(id), Some(sec)) = (device_id_hash, device_secret_32) {
        let sec_arr: &[u8; 32] = sec.try_into().map_err(|_| {
            validation_err(format!(
                "device_secret_32 must be 32 bytes, got {}",
                sec.len()
            ))
        })?;
        crypto = crypto.with_device_key(id, *sec_arr);
    }
    if let Some(mkt_pk_hex) = trusted_marketplace_pubkey_hex {
        let mkt_pk = decode_hex_array::<32>(mkt_pk_hex).map_err(dnam_err_to_py)?;
        crypto = crypto.with_marketplace_pubkey(mkt_pk);
    }
    let view = DnamContainerView::from_bytes(container_bytes, &crypto).map_err(dnam_err_to_py)?;
    let p = &view.preamble;
    let is_unsealed = container_bytes.len() >= 64
        && (container_bytes[0..4] == *b"DNAM" || container_bytes[0..4] == *b"DNCK");
    let header_end = 64 + (p.json_header_byte_len as usize);
    let header_json_cow = if is_unsealed && header_end <= container_bytes.len() {
        match core::str::from_utf8(&container_bytes[64..header_end]) {
            Ok(s) => std::borrow::Cow::Borrowed(s),
            Err(_) => std::borrow::Cow::Owned(
                serde_json::to_string(&view.header).map_err(validation_err)?,
            ),
        }
    } else {
        std::borrow::Cow::Owned(serde_json::to_string(&view.header).map_err(validation_err)?)
    };

    let dict = PyDict::new(py);
    let magic_str = core::str::from_utf8(&p.magic).unwrap_or("DNAM");
    dict.set_item(intern!(py, "magic"), magic_str)?;
    dict.set_item(intern!(py, "version_major"), p.version_major)?;
    dict.set_item(intern!(py, "version_minor"), p.version_minor)?;
    dict.set_item(intern!(py, "flags"), p.flags)?;
    dict.set_item(intern!(py, "num_controls"), p.num_controls)?;
    dict.set_item(intern!(py, "sample_rate_train_hz"), p.sample_rate_train_hz)?;
    dict.set_item(intern!(py, "json_header_byte_len"), p.json_header_byte_len)?;
    dict.set_item(intern!(py, "tensor_arena_offset"), p.tensor_arena_offset)?;
    dict.set_item(
        intern!(py, "tensor_arena_byte_len"),
        p.tensor_arena_byte_len,
    )?;
    dict.set_item(intern!(py, "arena_lite_end_cl"), p.arena_lite_end_cl)?;
    dict.set_item(intern!(py, "arena_full_end_cl"), p.arena_full_end_cl)?;
    dict.set_item(intern!(py, "arena_pro_end_cl"), p.arena_pro_end_cl)?;
    dict.set_item(intern!(py, "arena_ultra_end_cl"), p.arena_ultra_end_cl)?;
    dict.set_item(
        intern!(py, "arena_offset_lite_end"),
        p.arena_offset_lite_end(),
    )?;
    dict.set_item(
        intern!(py, "arena_offset_full_end"),
        p.arena_offset_full_end(),
    )?;
    dict.set_item(
        intern!(py, "arena_offset_pro_end"),
        p.arena_offset_pro_end(),
    )?;
    dict.set_item(
        intern!(py, "arena_offset_ultra_end"),
        p.arena_offset_ultra_end(),
    )?;
    dict.set_item(intern!(py, "arena_crc32c_lite"), p.arena_crc32c_lite)?;
    dict.set_item(intern!(py, "arena_crc32c_full"), p.arena_crc32c_full)?;
    dict.set_item(intern!(py, "arena_crc32c_pro"), p.arena_crc32c_pro)?;
    dict.set_item(intern!(py, "arena_crc32c_ultra"), p.arena_crc32c_ultra)?;
    dict.set_item(intern!(py, "num_ckpt_stems"), p.num_ckpt_stems)?;
    dict.set_item(intern!(py, "tail_time_ms"), p.tail_time_ms)?;
    dict.set_item(intern!(py, "header_crc32c"), p.header_crc32c)?;
    dict.set_item(intern!(py, "highest_tier"), p.highest_tier() as u8)?;
    dict.set_item(intern!(py, "is_stereo_out"), p.is_stereo_out())?;
    dict.set_item(intern!(py, "has_signatures"), p.has_signatures())?;
    dict.set_item(intern!(py, "is_encrypted_arena"), p.is_encrypted_arena())?;
    dict.set_item(intern!(py, "has_calib_stats"), p.has_calib_stats())?;
    dict.set_item(
        intern!(py, "verification_status"),
        format!("{:?}", view.verification_status),
    )?;
    dict.set_item(
        intern!(py, "protection_profile"),
        view.protection_profile.as_str(),
    )?;
    dict.set_item(intern!(py, "header_json"), header_json_cow.as_ref())?;
    if !is_unsealed || p.is_encrypted_arena() {
        dict.set_item(
            intern!(py, "arena_bytes"),
            PyBytes::new(py, view.arena_storage.as_bytes()),
        )?;
    } else {
        dict.set_item(intern!(py, "arena_bytes"), py.None())?;
    }
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
    let mut header: DnamJsonHeader = serde_json::from_str(header_json).map_err(validation_err)?;
    header.validate().map_err(dnam_err_to_py)?;

    let mut stored_arena = arena_bytes.to_vec();
    if !encrypted_arena
        && stored_arena.len() >= 9_024
        && !header.controls_schema.is_empty()
        && stored_arena[8_512..8_516] == [0, 0, 0, 0]
        && let Ok(descriptors) = header.compile_clamp_descriptors()
    {
        for (idx, desc) in descriptors.iter().take(8).enumerate() {
            let off = 8_512 + idx * 64;
            stored_arena[off..off + 64].copy_from_slice(&desc.to_bytes());
        }
    }

    let crc_src = plaintext_arena_crc_source.unwrap_or(&stored_arena);
    let container = build_dnam_container(&mut header, crc_src, &stored_arena, encrypted_arena)
        .map_err(dnam_err_to_py)?;
    Ok(PyBytes::new(py, &container))
}

/// Synthesizes a canonical reference `.dnam` (or `.dnam-ckpt`) container with `"gain"`, `"boost"`,
/// and `"delay_ms"` controls, a `"warm_slapback"` preset, and calibrated dry/wet weights.
#[pyfunction]
#[pyo3(signature = (supports_freeze = true, is_checkpoint = false))]
pub fn native_build_reference_container<'py>(
    py: Python<'py>,
    supports_freeze: bool,
    is_checkpoint: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    let mut header = DnamJsonHeader::default_reference();
    if is_checkpoint {
        header.container_kind = "dnam-ckpt".to_string();
    }
    let arena = populate_reference_controls_and_arena(&mut header, supports_freeze)?;
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
    let updated_header: DnamJsonHeader =
        serde_json::from_str(updated_header_json).map_err(validation_err)?;
    updated_header.validate().map_err(dnam_err_to_py)?;
    let mut buf = container_bytes.to_vec();
    update_header_in_place(&mut buf, &updated_header).map_err(dnam_err_to_py)?;
    Ok(PyBytes::new(py, &buf))
}

/// Fast in-place header update returning `(updated_container_bytes, json_header_byte_len, header_crc32c)`.
#[pyfunction]
pub fn native_update_header_in_place_fast<'py>(
    py: Python<'py>,
    container_bytes: &[u8],
    updated_header_json: &str,
) -> PyResult<(Bound<'py, PyBytes>, u32, u32)> {
    let updated_header: DnamJsonHeader =
        serde_json::from_str(updated_header_json).map_err(validation_err)?;
    updated_header.validate().map_err(dnam_err_to_py)?;
    let mut buf = container_bytes.to_vec();
    update_header_in_place(&mut buf, &updated_header).map_err(dnam_err_to_py)?;
    let json_len = u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]);
    let hdr_crc = u32::from_le_bytes([buf[60], buf[61], buf[62], buf[63]]);
    Ok((PyBytes::new(py, &buf), json_len, hdr_crc))
}

/// Strips a `.dnam` container down to `max_tier` (`0 = Lite`, `1 = Full`, `2 = Pro`, `3 = Ultra`)
/// without modifying the immutable JSON header.
#[pyfunction]
pub fn native_strip_container_to_tier<'py>(
    py: Python<'py>,
    container_bytes: &[u8],
    max_tier: u8,
) -> PyResult<Bound<'py, PyBytes>> {
    if container_bytes.len() >= 8
        && &container_bytes[0..4] == b"DNAM"
        && u16::from_le_bytes([container_bytes[4], container_bytes[5]]) == 2
    {
        let stripped =
            strip_v2_open_dnam_container(container_bytes, max_tier).map_err(dnam_err_to_py)?;
        return Ok(PyBytes::new(py, &stripped));
    }
    let tier = Tier::from_u8(max_tier);
    let stripped = strip_container_to_tier(container_bytes, tier).map_err(dnam_err_to_py)?;
    Ok(PyBytes::new(py, &stripped))
}

/// Validates a v2.0 Open `.dnam` container layout and returns its parsed directory metadata.
#[pyfunction]
pub fn native_validate_v2_open_layout<'py>(
    py: Python<'py>,
    container_bytes: &[u8],
) -> PyResult<Bound<'py, PyDict>> {
    let layout = validate_v2_open_dnam_layout(container_bytes).map_err(dnam_err_to_py)?;
    let dict = PyDict::new(py);
    dict.set_item("protection_profile", layout.protection_profile)?;
    dict.set_item("auth_kind", layout.auth_kind)?;
    dict.set_item("maximum_tier", layout.maximum_tier)?;
    dict.set_item("retained_tier", layout.retained_tier)?;
    dict.set_item("sample_rate_train_hz", layout.sample_rate_train_hz)?;
    dict.set_item("header_extent", layout.header_extent)?;
    dict.set_item("shared_extent", layout.shared_extent)?;
    dict.set_item("auth_extent", layout.auth_extent)?;
    dict.set_item("tier_start", layout.tier_start)?;
    dict.set_item("tier_ends", layout.tier_ends.to_vec())?;
    dict.set_item("mutable_extent", layout.mutable_extent)?;
    Ok(dict)
}

/// Serializes any JSON string using RFC 8785 JSON Canonicalization Scheme (JCS), optionally
/// projecting out mutable `.dnam` header fields first (`project_immutable = true`).
#[pyfunction]
#[pyo3(signature = (json_str, project_immutable = false, reject_floats = false))]
pub fn native_serialize_jcs(
    json_str: &str,
    project_immutable: bool,
    reject_floats: bool,
) -> PyResult<String> {
    let parsed: serde_json::Value = serde_json::from_str(json_str).map_err(validation_err)?;
    let val = if project_immutable {
        project_immutable_header_value(&parsed).map_err(dnam_err_to_py)?
    } else {
        parsed
    };
    serialize_jcs_value(&val, reject_floats).map_err(dnam_err_to_py)
}

/// Builds a 64B-aligned `DMU1` mutable trailer frame from `mutable_json_bytes` and `capacity`.
#[pyfunction]
pub fn native_build_dmu1_frame<'py>(
    py: Python<'py>,
    mutable_json_bytes: &[u8],
    capacity: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let frame = build_dmu1_mutable_frame(mutable_json_bytes, capacity).map_err(dnam_err_to_py)?;
    Ok(PyBytes::new(py, &frame))
}

/// Validates and parses a 64B-aligned `DMU1` mutable trailer frame, returning its canonical JSON string.
#[pyfunction]
pub fn native_parse_dmu1_frame(frame_bytes: &[u8]) -> PyResult<String> {
    let val = parse_and_validate_dmu1_frame(frame_bytes).map_err(dnam_err_to_py)?;
    serde_json::to_string(&val).map_err(validation_err)
}

/// Computes the Castagnoli `CRC32C` checksum over `data` using hardware `CRC32C` instructions when available.
#[pyfunction]
pub fn native_crc32c(data: &[u8]) -> u32 {
    crc32c_compute(data)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(native_validate_container, m)?)?;
    m.add_function(wrap_pyfunction!(native_build_container, m)?)?;
    m.add_function(wrap_pyfunction!(native_build_reference_container, m)?)?;
    m.add_function(wrap_pyfunction!(native_update_header_in_place, m)?)?;
    m.add_function(wrap_pyfunction!(native_update_header_in_place_fast, m)?)?;
    m.add_function(wrap_pyfunction!(native_strip_container_to_tier, m)?)?;
    m.add_function(wrap_pyfunction!(native_validate_v2_open_layout, m)?)?;
    m.add_function(wrap_pyfunction!(native_serialize_jcs, m)?)?;
    m.add_function(wrap_pyfunction!(native_build_dmu1_frame, m)?)?;
    m.add_function(wrap_pyfunction!(native_parse_dmu1_frame, m)?)?;
    m.add_function(wrap_pyfunction!(native_crc32c, m)?)?;
    checkpoint::register(m)?;
    gram::register(m)?;
    Ok(())
}
