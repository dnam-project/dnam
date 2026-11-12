//! Target-agnostic `.dnam` model, `DMU1` mutable frame, JCS, `CRC32C`,
//! and `uplo='U'` Gram matrix core functions (`ref[sdk.ts.model.containers-and-checkpoints]`).

use std::collections::BTreeMap;

use dnam_capture::proj::{pack_uplo_u_f64, unpack_uplo_u_f64, uplo_u_packed_len};
use dnam_engine::Tier;
use dnam_engine::container::{
    ARENA_OFFSET_ULTRA_END, DnamContainerView, DnamPreamble, build_dmu1_mutable_frame,
    build_dnam_container, build_dnam_container_with_cp1, crc32c_compute,
    parse_and_validate_dmu1_frame, project_immutable_header_value, serialize_jcs_value,
    strip_container_to_tier, strip_v2_open_dnam_container, update_header_in_place,
    validate_v2_open_dnam_layout,
};
use dnam_engine::provenance::{RustCryptoProvider, decode_hex_array};
use dnam_engine::schema::{
    ControlCalibration, ControlCalibrationRole, ControlKind, ControlSchemaDef, ControlUiHints,
    DnamJsonHeader, PresetDef, SchemaVersion, TempoSubdivision,
};

use crate::error::{format_tagged_dnam_error, validation_err};

/// Validated container metadata extracted by [`validate_container_with_key_core`].
#[derive(Debug, Clone)]
pub struct ValidatedContainerInfo {
    pub preamble: DnamPreamble,
    pub magic_str: String,
    pub verification_status: String,
    pub protection_profile: String,
    pub header_json: String,
    pub arena_bytes: Vec<u8>,
}

pub fn validate_container_with_key_core(
    container_bytes: &[u8],
    device_id_hash: Option<&str>,
    device_secret_32: Option<&[u8]>,
    trusted_marketplace_pubkey_hex: Option<&str>,
) -> Result<ValidatedContainerInfo, String> {
    let mut crypto = RustCryptoProvider::new();
    if let (Some(id), Some(sec)) = (device_id_hash, device_secret_32)
        && !id.is_empty()
        && !sec.is_empty()
    {
        let sec_arr: &[u8; 32] = sec.try_into().map_err(|_| {
            validation_err(format!("deviceSecret must be 32 bytes, got {}", sec.len()))
        })?;
        crypto = crypto.with_device_key(id, *sec_arr);
    }
    if let Some(mkt_pk_hex) = trusted_marketplace_pubkey_hex
        && !mkt_pk_hex.is_empty()
    {
        let mkt_pk =
            decode_hex_array::<32>(mkt_pk_hex).map_err(|e| format_tagged_dnam_error(&e))?;
        crypto = crypto.with_marketplace_pubkey(mkt_pk);
    }
    let view = DnamContainerView::from_bytes(container_bytes, &crypto)
        .map_err(|e| format_tagged_dnam_error(&e))?;
    let header_json = view
        .header
        .to_json_string()
        .map_err(|e| validation_err(e.to_string()))?;

    Ok(ValidatedContainerInfo {
        preamble: view.preamble,
        magic_str: String::from_utf8_lossy(&view.preamble.magic).into_owned(),
        verification_status: format!("{:?}", view.verification_status),
        protection_profile: view.protection_profile.as_str().to_string(),
        header_json,
        arena_bytes: view.arena_storage.as_bytes().to_vec(),
    })
}

pub(crate) fn populate_reference_controls_and_arena(
    header: &mut DnamJsonHeader,
    supports_freeze: bool,
) -> Result<Vec<u8>, String> {
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
    arena[4416..4420].copy_from_slice(&0.25_f32.to_le_bytes());
    arena[4480..4484].copy_from_slice(&0.25_f32.to_le_bytes());

    let descriptors = header
        .compile_clamp_descriptors()
        .map_err(|e| format_tagged_dnam_error(&e))?;
    for (idx, desc) in descriptors.iter().enumerate() {
        let off = 8_512 + idx * 64;
        if off + 64 <= arena.len() {
            arena[off..off + 64].copy_from_slice(&desc.to_bytes());
        }
    }
    Ok(arena)
}

pub fn build_reference_container_core(
    supports_freeze: bool,
    inputs: u8,
    output_mode: &str,
) -> Result<Vec<u8>, String> {
    if inputs != 1 && inputs != 2 {
        return Err(validation_err(format!(
            "inputs must be 1 or 2, got {inputs}"
        )));
    }
    if output_mode != "mono" && output_mode != "stereo" {
        return Err(validation_err(format!(
            "outputMode must be 'mono' or 'stereo', got '{output_mode}'"
        )));
    }
    let mut header = DnamJsonHeader::default_reference();
    header.io_config.inputs = inputs;
    header.io_config.output_mode = output_mode.to_string();
    let arena = populate_reference_controls_and_arena(&mut header, supports_freeze)?;
    build_dnam_container(&mut header, &arena, &arena, false)
        .map_err(|e| format_tagged_dnam_error(&e))
}

pub fn build_dnam_container_core(
    header_json: &str,
    arena_bytes: &[u8],
    encrypted_arena_bytes: Option<&[u8]>,
    _has_signatures_flag: bool,
) -> Result<Vec<u8>, String> {
    let mut header: DnamJsonHeader =
        DnamJsonHeader::from_json_str(header_json).map_err(|e| validation_err(e.to_string()))?;
    header
        .validate()
        .map_err(|e| format_tagged_dnam_error(&e))?;

    let is_encrypted = encrypted_arena_bytes.is_some();
    let mut stored_arena = encrypted_arena_bytes.unwrap_or(arena_bytes).to_vec();
    if !is_encrypted
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

    let crc_src = if is_encrypted {
        arena_bytes
    } else {
        &stored_arena
    };
    build_dnam_container(&mut header, crc_src, &stored_arena, is_encrypted)
        .map_err(|e| format_tagged_dnam_error(&e))
}

pub fn update_header_in_place_core(
    container_bytes: &[u8],
    new_header_json: &str,
) -> Result<Vec<u8>, String> {
    let mut new_header: DnamJsonHeader = DnamJsonHeader::from_json_str(new_header_json)
        .map_err(|e| validation_err(e.to_string()))?;
    let mut buf = container_bytes.to_vec();
    match update_header_in_place(&mut buf, &new_header) {
        Ok(()) => Ok(buf),
        Err(in_place_err) => {
            if container_bytes.starts_with(b"DNSL") {
                return Err(format_tagged_dnam_error(&in_place_err));
            }
            let Ok(layout) = validate_v2_open_dnam_layout(container_bytes) else {
                return Err(format_tagged_dnam_error(&in_place_err));
            };
            let auth_slice =
                &container_bytes[layout.auth_extent.0..layout.auth_extent.0 + layout.auth_extent.1];
            if layout.protection_profile != 0
                || layout.auth_kind != 0
                || auth_slice.starts_with(b"DAU1")
            {
                return Err(format_tagged_dnam_error(&in_place_err));
            }
            let crypto = RustCryptoProvider::new();
            let existing_view = DnamContainerView::from_bytes(container_bytes, &crypto)
                .map_err(|e| format_tagged_dnam_error(&e))?;
            if existing_view.header.metadata.signatures.is_some() {
                return Err(format_tagged_dnam_error(&in_place_err));
            }
            let arena = existing_view.arena_storage.as_bytes();
            build_dnam_container_with_cp1(
                &mut new_header,
                arena,
                arena,
                false,
                &existing_view.cp1_contributors,
            )
            .map_err(|e| format_tagged_dnam_error(&e))
        }
    }
}

pub fn strip_container_to_tier_core(
    container_bytes: &[u8],
    max_tier: u8,
) -> Result<Vec<u8>, String> {
    if max_tier > 3 {
        return Err(validation_err(format!("Invalid maxTier: {max_tier}")));
    }
    strip_container_to_tier(container_bytes, Tier::from_u8(max_tier))
        .map_err(|e| format_tagged_dnam_error(&e))
}

pub fn parse_schema_version_core(version_str: &str) -> Result<[u32; 4], String> {
    let v = SchemaVersion::parse(version_str).map_err(|e| format_tagged_dnam_error(&e))?;
    Ok([
        u32::from(v.major),
        u32::from(v.minor),
        u32::from(v.patch),
        u32::from(v.is_supported()),
    ])
}

#[inline]
pub fn crc32c_compute_core(data: &[u8]) -> u32 {
    crc32c_compute(data)
}

pub fn canonicalize_jcs_core(
    json_str: &str,
    strip_mutable: bool,
    reject_floats: bool,
) -> Result<String, String> {
    let val: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| validation_err(e.to_string()))?;
    let projected = if strip_mutable {
        project_immutable_header_value(&val).map_err(|e| format_tagged_dnam_error(&e))?
    } else {
        val
    };
    serialize_jcs_value(&projected, reject_floats).map_err(|e| format_tagged_dnam_error(&e))
}

/// Validates and normalizes a snake_case `DnamJsonHeader` JSON string using
/// Rust `serde` and `.validate()`, rejecting unknown fields and returning canonical wire JSON.
pub fn validate_and_normalize_header_core(header_json: &str) -> Result<String, String> {
    DnamJsonHeader::validate_and_normalize_json_str(header_json)
        .map_err(|e| format_tagged_dnam_error(&e))
}

pub fn build_dmu1_frame_core(
    mutable_json_bytes: &[u8],
    capacity: usize,
) -> Result<Vec<u8>, String> {
    build_dmu1_mutable_frame(mutable_json_bytes, capacity).map_err(|e| format_tagged_dnam_error(&e))
}

pub fn parse_dmu1_frame_core(frame_bytes: &[u8]) -> Result<String, String> {
    let val =
        parse_and_validate_dmu1_frame(frame_bytes).map_err(|e| format_tagged_dnam_error(&e))?;
    serde_json::to_string(&val).map_err(|e| validation_err(e.to_string()))
}

pub fn validate_v2_open_dnam_layout_core(container_bytes: &[u8]) -> Result<String, String> {
    let layout =
        validate_v2_open_dnam_layout(container_bytes).map_err(|e| format_tagged_dnam_error(&e))?;
    let val = serde_json::json!({
        "protectionProfile": layout.protection_profile,
        "authKind": layout.auth_kind,
        "maximumTier": layout.maximum_tier,
        "retainedTier": layout.retained_tier,
        "sampleRateTrainHz": layout.sample_rate_train_hz,
        "headerExtent": [layout.header_extent.0, layout.header_extent.1],
        "sharedExtent": [layout.shared_extent.0, layout.shared_extent.1],
        "authExtent": [layout.auth_extent.0, layout.auth_extent.1],
        "tierStart": layout.tier_start,
        "tierEnds": layout.tier_ends,
        "mutableExtent": [layout.mutable_extent.0, layout.mutable_extent.1],
    });
    serde_json::to_string(&val).map_err(|e| validation_err(e.to_string()))
}

pub fn strip_v2_open_dnam_container_core(
    container_bytes: &[u8],
    target_tier: u8,
) -> Result<Vec<u8>, String> {
    strip_v2_open_dnam_container(container_bytes, target_tier)
        .map_err(|e| format_tagged_dnam_error(&e))
}

// ============================================================================
// uplo='U' Gram Matrix Packing (`ref[container.proj.stat-uplo-u]`)
// ============================================================================

#[inline]
pub fn uplo_u_packed_len_core(n_dim: usize) -> usize {
    uplo_u_packed_len(n_dim)
}

pub fn pack_uplo_u_f64_into_core(
    dense_flat: &[f64],
    n_dim: usize,
    out_packed: &mut [f64],
) -> Result<(), String> {
    let expected = n_dim * n_dim;
    if dense_flat.len() != expected {
        return Err(validation_err(format!(
            "Expected {expected} elements for {n_dim}x{n_dim} matrix, got {}",
            dense_flat.len()
        )));
    }
    let packed_len = uplo_u_packed_len(n_dim);
    if out_packed.len() < packed_len {
        return Err(validation_err(format!(
            "Expected outPacked length >= {packed_len}, got {}",
            out_packed.len()
        )));
    }
    pack_uplo_u_f64(dense_flat, n_dim, &mut out_packed[..packed_len]);
    Ok(())
}

pub fn unpack_uplo_u_f64_into_core(
    packed_u: &[f64],
    n_tier: usize,
    out_dense: &mut [f64],
) -> Result<(), String> {
    let req_len = uplo_u_packed_len(n_tier);
    if packed_u.len() < req_len {
        return Err(validation_err(format!(
            "Packed buffer too short for nTier={n_tier}: need {req_len}, got {}",
            packed_u.len()
        )));
    }
    let expected_dense = n_tier * n_tier;
    if out_dense.len() < expected_dense {
        return Err(validation_err(format!(
            "Expected outDense length >= {expected_dense}, got {}",
            out_dense.len()
        )));
    }
    unpack_uplo_u_f64(packed_u, n_tier, &mut out_dense[..expected_dense]);
    Ok(())
}
