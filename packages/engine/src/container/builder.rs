//! Canonical v2.0 `.dnam` container builder (`build_dnam_container`) and `DMU1` in-place
//! mutable suffix editor (`update_header_in_place`).
//!
//! Canonical Reference: `docs/specs/containers/dnam_binary_format.md`.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::control::Cp1ContributorSlot;
use crate::provenance::canonicalize_jcs_value;
use crate::schema::{DnamError, DnamJsonHeader, SchemaVersion, reject_if};

use super::wire::{
    DCRC_RECORD_SIZE, DEFAULT_DMU1_TRAILER_CAPACITY, DMU1_HEADER_SIZE, DcrcAuthRecord,
    build_dmu1_mutable_frame, build_v2_dnam_container_with_profile, encode_cp1_shared_stream,
    parse_strict_json_object_bytes, validate_mutable_json_schema,
    validate_v2_dnam_layout_with_max_profile,
};
use super::{
    ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_PRO_END, ARENA_OFFSET_ULTRA_END,
    FLAG_ENCRYPTED_ARENA, FLAG_HAS_SIGNATURES, FLAG_STEREO_OUT, crc32c_update,
};

fn build_mutable_dmu1_json_bytes(header: &DnamJsonHeader) -> Result<Vec<u8>, DnamError> {
    let mut dmu_obj = serde_json::Map::new();
    if !header.presets.is_empty() {
        dmu_obj.insert(
            "presets".to_string(),
            serde_json::to_value(&header.presets)
                .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?,
        );
    }
    if !header.user_metadata.is_empty() {
        dmu_obj.insert(
            "user_metadata".to_string(),
            serde_json::to_value(&header.user_metadata)
                .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?,
        );
    }
    let mut ctrl_um = serde_json::Map::new();
    for ctrl in &header.controls_schema {
        if !ctrl.user_metadata.is_empty() {
            ctrl_um.insert(
                ctrl.id.clone(),
                serde_json::to_value(&ctrl.user_metadata)
                    .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?,
            );
        }
    }
    if !ctrl_um.is_empty() {
        dmu_obj.insert(
            "control_user_metadata".to_string(),
            serde_json::Value::Object(ctrl_um),
        );
    }
    let doc = serde_json::Value::Object(dmu_obj);
    validate_mutable_json_schema(&doc)?;
    serde_json::to_vec(&doc).map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))
}

fn strip_dmu1_mutable_fields_from_header_value(val: &mut serde_json::Value) {
    if let Some(root) = val.as_object_mut() {
        root.remove("presets");
        root.remove("user_metadata");
        if let Some(ctrls) = root
            .get_mut("controls_schema")
            .and_then(serde_json::Value::as_array_mut)
        {
            for ctrl in ctrls {
                if let Some(c_obj) = ctrl.as_object_mut() {
                    c_obj.remove("user_metadata");
                }
            }
        }
    }
}

fn strip_mutable_and_crc_fields_for_in_place_comparison(val: &mut serde_json::Value) {
    strip_dmu1_mutable_fields_from_header_value(val);
    if let Some(tiers) = val
        .get_mut("slimmable_tiers")
        .and_then(serde_json::Value::as_object_mut)
    {
        for name in ["lite", "full", "pro", "ultra"] {
            if let Some(t_obj) = tiers
                .get_mut(name)
                .and_then(serde_json::Value::as_object_mut)
            {
                t_obj.remove("prefix_crc32c");
            }
        }
    }
}

/// Builds a canonical `v2.0` 64B-aligned runtime `.dnam` container byte vector.
pub fn build_dnam_container(
    header: &mut DnamJsonHeader,
    plaintext_arena_crc_source: &[u8],
    stored_arena_bytes: &[u8],
    encrypted_arena: bool,
) -> Result<Vec<u8>, DnamError> {
    build_dnam_container_with_cp1(
        header,
        plaintext_arena_crc_source,
        stored_arena_bytes,
        encrypted_arena,
        &[],
    )
}

/// Builds a canonical `v2.0` 64B-aligned runtime `.dnam` container byte vector with optional
/// packed `CP1\0` control contributors in Region 4 `shared_extent`.
pub fn build_dnam_container_with_cp1(
    header: &mut DnamJsonHeader,
    plaintext_arena_crc_source: &[u8],
    stored_arena_bytes: &[u8],
    encrypted_arena: bool,
    cp1_slots: &[Cp1ContributorSlot],
) -> Result<Vec<u8>, DnamError> {
    let lite_end = ARENA_OFFSET_LITE_END as usize;
    let full_end = ARENA_OFFSET_FULL_END as usize;
    let pro_end = ARENA_OFFSET_PRO_END as usize;
    let ultra_end = ARENA_OFFSET_ULTRA_END as usize;

    if stored_arena_bytes.len() < lite_end || (stored_arena_bytes.len() & 0x3F) != 0 {
        return Err(DnamError::UnalignedOffset {
            field: "stored_arena_bytes.len",
            offset: stored_arena_bytes.len() as u32,
        });
    }
    let tier_slices: Vec<&[u8]> = match stored_arena_bytes.len() {
        len if len == lite_end => alloc::vec![&stored_arena_bytes[..lite_end]],
        len if len == full_end => alloc::vec![
            &stored_arena_bytes[..lite_end],
            &stored_arena_bytes[lite_end..full_end],
        ],
        len if len == pro_end => alloc::vec![
            &stored_arena_bytes[..lite_end],
            &stored_arena_bytes[lite_end..full_end],
            &stored_arena_bytes[full_end..pro_end],
        ],
        len if len == ultra_end => alloc::vec![
            &stored_arena_bytes[..lite_end],
            &stored_arena_bytes[lite_end..full_end],
            &stored_arena_bytes[full_end..pro_end],
            &stored_arena_bytes[pro_end..ultra_end],
        ],
        _ => {
            return Err(DnamError::SchemaValidationError(
                "runtime .dnam arena must equal canonical Lite (10,752 B), Full (22,336 B), Pro (36,224 B), or Ultra (64,000 B)".to_string(),
            ));
        }
    };

    if plaintext_arena_crc_source.len() < stored_arena_bytes.len() {
        return Err(DnamError::BufferTooSmall {
            expected: stored_arena_bytes.len(),
            actual: plaintext_arena_crc_source.len(),
        });
    }

    let schema_ver = SchemaVersion::parse(&header.version)?;
    schema_ver.ensure_supported()?;
    reject_if(
        header.sample_rate_train_hz != 48_000,
        "v2.0 .dnam container requires sample_rate_train_hz == 48000",
    )?;

    let weights_len = stored_arena_bytes.len();
    let crc_lite = crc32c_update(0, &plaintext_arena_crc_source[..lite_end]);
    let crc_full = if weights_len >= full_end {
        crc32c_update(crc_lite, &plaintext_arena_crc_source[lite_end..full_end])
    } else {
        crc_lite
    };
    let crc_pro = if weights_len >= pro_end {
        crc32c_update(crc_full, &plaintext_arena_crc_source[full_end..pro_end])
    } else {
        crc_full
    };
    let crc_ultra = if weights_len >= ultra_end {
        crc32c_update(crc_pro, &plaintext_arena_crc_source[pro_end..ultra_end])
    } else {
        crc_pro
    };

    header.slimmable_tiers.lite.prefix_crc32c = crc_lite;
    header.slimmable_tiers.full.prefix_crc32c = crc_full;
    header.slimmable_tiers.pro.prefix_crc32c = crc_pro;
    header.slimmable_tiers.ultra.prefix_crc32c = crc_ultra;

    let dmu_payload = build_mutable_dmu1_json_bytes(header)?;
    if DMU1_HEADER_SIZE + dmu_payload.len() > DEFAULT_DMU1_TRAILER_CAPACITY {
        return Err(DnamError::HeaderCapacityExceeded {
            capacity: DEFAULT_DMU1_TRAILER_CAPACITY,
            required: DMU1_HEADER_SIZE + dmu_payload.len(),
        });
    }
    let dmu1_frame = build_dmu1_mutable_frame(&dmu_payload, DEFAULT_DMU1_TRAILER_CAPACITY)?;

    let hdr_json_bytes = header
        .to_json_vec()
        .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?;
    let mut region3_val = parse_strict_json_object_bytes(&hdr_json_bytes)?;
    strip_dmu1_mutable_fields_from_header_value(&mut region3_val);
    let mut region3_str = String::with_capacity(hdr_json_bytes.len());
    canonicalize_jcs_value(&region3_val, &mut region3_str);
    let region3_bytes = region3_str.into_bytes();

    let compiled_descs = header.compile_clamp_descriptors()?;
    let cp1_bytes = encode_cp1_shared_stream(cp1_slots)?;
    let mut shared_bytes = Vec::with_capacity(compiled_descs.len() * 64 + cp1_bytes.len());
    for desc in &compiled_descs {
        shared_bytes.extend_from_slice(&desc.to_bytes());
    }
    shared_bytes.extend_from_slice(&cp1_bytes);

    let has_authorized_devices = header
        .metadata
        .signatures
        .as_ref()
        .and_then(|s| s.marketplace_receipt.as_ref())
        .is_some_and(|r| !r.authorized_devices.is_empty());
    let protection_profile = u8::from(encrypted_arena || has_authorized_devices);

    let mut flags_hint = 0u16;
    if header.io_config.output_mode != "mono" {
        flags_hint |= FLAG_STEREO_OUT;
    }
    if encrypted_arena {
        flags_hint |= FLAG_ENCRYPTED_ARENA;
    }
    if header.metadata.signatures.is_some() {
        flags_hint |= FLAG_HAS_SIGNATURES;
    }

    let placeholder_auth = [0u8; DCRC_RECORD_SIZE];
    let mut container = build_v2_dnam_container_with_profile(
        protection_profile,
        &region3_bytes,
        &shared_bytes,
        &placeholder_auth,
        &tier_slices,
        Some(&dmu1_frame),
        0,
    )?;

    let auth_offset =
        u64::from_le_bytes(container[64 + 4 * 8..64 + 5 * 8].try_into().unwrap()) as usize;
    let crc_prefix = crc32c_update(0, &container[..auth_offset]);
    let dcrc = DcrcAuthRecord {
        crc_prefix,
        crc_tiers: [crc_lite, crc_full, crc_pro, crc_ultra],
        num_controls: header.controls_schema.len().min(u16::MAX as usize) as u16,
        flags_hint,
        tail_time_ms_bits: header.io_config.tail_time_ms.to_bits(),
    };
    container[auth_offset..auth_offset + DCRC_RECORD_SIZE].copy_from_slice(&dcrc.to_bytes());

    Ok(container)
}

/// Updates the `DMU1` mutable suffix (`presets`, `user_metadata`, `control_user_metadata`) of an
/// existing canonical `v2.0` `.dnam` container in-place (`< 0.1 ms`) without altering the immutable
/// Region 3 JCS header, Region 4 control descriptors, or tensor arenas.
pub fn update_header_in_place(
    container_bytes: &mut [u8],
    new_header: &DnamJsonHeader,
) -> Result<(), DnamError> {
    let layout = validate_v2_dnam_layout_with_max_profile(container_bytes, 1)?;
    new_header.validate()?;

    let existing_json_slice =
        &container_bytes[layout.header_extent.0..layout.header_extent.0 + layout.header_extent.1];
    let mut existing_val = parse_strict_json_object_bytes(existing_json_slice)?;
    strip_mutable_and_crc_fields_for_in_place_comparison(&mut existing_val);
    let mut existing_jcs = String::with_capacity(existing_json_slice.len());
    canonicalize_jcs_value(&existing_val, &mut existing_jcs);

    let candidate_bytes = new_header
        .to_json_vec()
        .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?;
    let mut candidate_val = parse_strict_json_object_bytes(&candidate_bytes)?;
    strip_mutable_and_crc_fields_for_in_place_comparison(&mut candidate_val);
    let mut candidate_jcs = String::with_capacity(candidate_bytes.len());
    canonicalize_jcs_value(&candidate_val, &mut candidate_jcs);

    if existing_jcs != candidate_jcs {
        return Err(DnamError::SchemaValidationError(
            "update_header_in_place only permits editing mutable DMU1 fields (presets, user_metadata, control_user_metadata)".to_string(),
        ));
    }

    let payload_bytes = build_mutable_dmu1_json_bytes(new_header)?;
    let cap = layout.mutable_extent.1;
    if DMU1_HEADER_SIZE + payload_bytes.len() > cap {
        return Err(DnamError::HeaderCapacityExceeded {
            capacity: cap,
            required: DMU1_HEADER_SIZE + payload_bytes.len(),
        });
    }
    let dmu1 = build_dmu1_mutable_frame(&payload_bytes, cap)?;
    container_bytes[layout.mutable_extent.0..layout.mutable_extent.0 + cap].copy_from_slice(&dmu1);
    Ok(())
}
