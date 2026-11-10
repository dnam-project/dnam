//! Container builder (`build_dnam_container`) and in-place header editor (`update_header_in_place`).
//!
//! Canonical Reference: `docs/containers/dnam_binary_format.md`.

use alloc::string::ToString;
use alloc::vec::Vec;

use crate::schema::{DnamError, DnamJsonHeader};

use super::preamble::DnamPreamble;
use super::view::compute_header_crc32c;
use super::{
    ARENA_FULL_END_CL, ARENA_LITE_END_CL, ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END,
    ARENA_OFFSET_PRO_END, ARENA_OFFSET_ULTRA_END, ARENA_PRO_END_CL, ARENA_ULTRA_END_CL, DNAM_MAGIC,
    FLAG_ENCRYPTED_ARENA, FLAG_HAS_FULL_DELTA, FLAG_HAS_LITE, FLAG_HAS_PRO_DELTA,
    FLAG_HAS_SIGNATURES, FLAG_HAS_ULTRA_DELTA, FLAG_STEREO_OUT, crc32c_update,
};

/// Builds a complete 4 KB page-aligned runtime `.dnam` container byte vector.
pub fn build_dnam_container(
    header: &mut DnamJsonHeader,
    plaintext_arena_crc_source: &[u8],
    stored_arena_bytes: &[u8],
    encrypted_arena: bool,
) -> Result<Vec<u8>, DnamError> {
    if stored_arena_bytes.len() < ARENA_OFFSET_LITE_END as usize
        || (stored_arena_bytes.len() & 0x3F) != 0
    {
        return Err(DnamError::UnalignedOffset {
            field: "stored_arena_bytes.len",
            offset: stored_arena_bytes.len() as u32,
        });
    }

    if stored_arena_bytes.len() > ARENA_OFFSET_ULTRA_END as usize {
        return Err(DnamError::SchemaValidationError(
            "runtime .dnam arena cannot exceed ARENA_OFFSET_ULTRA_END (64,000 B)".to_string(),
        ));
    }

    let weights_capacity = stored_arena_bytes.len();
    let weights_end = if weights_capacity >= ARENA_OFFSET_ULTRA_END as usize {
        ARENA_OFFSET_ULTRA_END as usize
    } else if weights_capacity >= ARENA_OFFSET_PRO_END as usize {
        ARENA_OFFSET_PRO_END as usize
    } else if weights_capacity >= ARENA_OFFSET_FULL_END as usize {
        ARENA_OFFSET_FULL_END as usize
    } else {
        ARENA_OFFSET_LITE_END as usize
    };

    let crc_src_len = plaintext_arena_crc_source.len().min(weights_end);
    let lite_end = ARENA_OFFSET_LITE_END as usize;
    let full_end = (ARENA_OFFSET_FULL_END as usize).min(crc_src_len);
    let pro_end = (ARENA_OFFSET_PRO_END as usize).min(crc_src_len);
    let ultra_end = (ARENA_OFFSET_ULTRA_END as usize).min(crc_src_len);

    let crc_lite = crc32c_update(0, &plaintext_arena_crc_source[..lite_end]);
    let crc_full = if full_end >= ARENA_OFFSET_FULL_END as usize {
        crc32c_update(crc_lite, &plaintext_arena_crc_source[lite_end..full_end])
    } else {
        crc_lite
    };
    let crc_pro = if pro_end >= ARENA_OFFSET_PRO_END as usize {
        crc32c_update(crc_full, &plaintext_arena_crc_source[full_end..pro_end])
    } else {
        crc_full
    };
    let crc_ultra = if ultra_end >= ARENA_OFFSET_ULTRA_END as usize {
        crc32c_update(crc_pro, &plaintext_arena_crc_source[pro_end..ultra_end])
    } else {
        crc_pro
    };

    header.slimmable_tiers.lite.prefix_crc32c = crc_lite;
    header.slimmable_tiers.full.prefix_crc32c = crc_full;
    header.slimmable_tiers.pro.prefix_crc32c = crc_pro;
    header.slimmable_tiers.ultra.prefix_crc32c = crc_ultra;

    if header.metadata.signatures.is_none()
        && (header.version == "1.0.0" || header.version == "1.1.0")
    {
        header.version = crate::schema::SchemaVersion::V2_0_0.as_semver_string();
    }

    let json_bytes = header
        .to_json_vec()
        .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?;
    let p_ctrls = header.controls_schema.len();
    let min_header_region = 64 + json_bytes.len() + 128 * p_ctrls;
    let tensor_arena_offset = min_header_region.div_ceil(4096).max(1) * 4096;

    let mut flags = FLAG_HAS_LITE;
    if weights_end >= ARENA_OFFSET_FULL_END as usize {
        flags |= FLAG_HAS_FULL_DELTA;
    }
    if weights_end >= ARENA_OFFSET_PRO_END as usize {
        flags |= FLAG_HAS_PRO_DELTA;
    }
    if weights_end >= ARENA_OFFSET_ULTRA_END as usize {
        flags |= FLAG_HAS_ULTRA_DELTA;
    }
    if encrypted_arena {
        flags |= FLAG_ENCRYPTED_ARENA;
    }
    if header.metadata.signatures.is_some() {
        flags |= FLAG_HAS_SIGNATURES;
    }
    if header.io_config.output_mode != "mono" {
        flags |= FLAG_STEREO_OUT;
    }

    let mut container = alloc::vec![0u8; tensor_arena_offset + stored_arena_bytes.len()];
    let control_table_off = tensor_arena_offset.saturating_sub(128 * p_ctrls);
    container[64..64 + json_bytes.len()].copy_from_slice(&json_bytes);
    container[64 + json_bytes.len()..control_table_off].fill(0x20);
    // Populate CL 0 (ControlClampDescriptor) of each 128-byte ControlChannelRecord[0..P]
    if let Ok(compiled_descs) = header.compile_clamp_descriptors() {
        for (p, desc) in compiled_descs.iter().enumerate() {
            let rec_off = control_table_off + p * 128;
            if rec_off + 64 <= tensor_arena_offset {
                container[rec_off..rec_off + 64].copy_from_slice(&desc.to_bytes());
            }
        }
    }
    container[tensor_arena_offset..].copy_from_slice(stored_arena_bytes);

    let version = crate::schema::SchemaVersion::parse(&header.version)?;
    let preamble = DnamPreamble {
        magic: DNAM_MAGIC,
        version_major: version.major,
        version_minor: version.minor,
        flags,
        num_controls: header.controls_schema.len().min(u16::MAX as usize) as u16,
        sample_rate_train_hz: header.sample_rate_train_hz,
        json_header_byte_len: json_bytes.len() as u32,
        tensor_arena_offset: tensor_arena_offset as u32,
        tensor_arena_byte_len: stored_arena_bytes.len() as u32,
        arena_lite_end_cl: ARENA_LITE_END_CL,
        arena_full_end_cl: ARENA_FULL_END_CL,
        arena_pro_end_cl: ARENA_PRO_END_CL,
        arena_ultra_end_cl: ARENA_ULTRA_END_CL,
        arena_crc32c_lite: crc_lite,
        arena_crc32c_full: crc_full,
        arena_crc32c_pro: crc_pro,
        arena_crc32c_ultra: crc_ultra,
        num_ckpt_stems: 0,
        _reserved: 0,
        tail_time_ms: header.io_config.tail_time_ms,
        header_crc32c: 0,
    };

    let preamble_bytes = preamble.to_bytes();
    container[0..64].copy_from_slice(&preamble_bytes);

    let computed_header_crc = compute_header_crc32c(&container, tensor_arena_offset);
    let crc_bytes = computed_header_crc.to_le_bytes();
    container[0x3C..0x40].copy_from_slice(&crc_bytes);

    Ok(container)
}

/// Updates the JSON header or `DMU1` mutable frame of an existing container in-place (< 0.1 ms).
pub fn update_header_in_place(
    container_bytes: &mut [u8],
    new_header: &DnamJsonHeader,
) -> Result<(), DnamError> {
    if super::preamble::is_v2_wire_directory_container(container_bytes)
        && &container_bytes[0..4] == b"DNAM"
    {
        let layout = super::wire::validate_v2_open_dnam_layout(container_bytes)?;
        let mut dmu_obj = serde_json::Map::new();
        if !new_header.presets.is_empty() {
            dmu_obj.insert(
                "presets".to_string(),
                serde_json::to_value(&new_header.presets)
                    .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?,
            );
        }
        if !new_header.user_metadata.is_empty() {
            dmu_obj.insert(
                "user_metadata".to_string(),
                serde_json::to_value(&new_header.user_metadata)
                    .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?,
            );
        }
        let payload_bytes = serde_json::to_vec(&serde_json::Value::Object(dmu_obj))
            .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?;
        let cap = layout.mutable_extent.1;
        if super::wire::DMU1_HEADER_SIZE + payload_bytes.len() > cap {
            return Err(DnamError::HeaderCapacityExceeded {
                capacity: cap,
                required: super::wire::DMU1_HEADER_SIZE + payload_bytes.len(),
            });
        }
        let dmu1 = super::wire::build_dmu1_mutable_frame(&payload_bytes, cap)?;
        container_bytes[layout.mutable_extent.0..layout.mutable_extent.0 + cap]
            .copy_from_slice(&dmu1);
        return Ok(());
    }
    let mut preamble = DnamPreamble::from_bytes(container_bytes)?;
    let arena_off = preamble.tensor_arena_offset as usize;
    if container_bytes.len() < arena_off {
        return Err(DnamError::BufferTooSmall {
            expected: arena_off,
            actual: container_bytes.len(),
        });
    }
    let p_ctrls = new_header.controls_schema.len();
    let control_table_off = arena_off.saturating_sub(128 * p_ctrls);
    let max_header_len = control_table_off.saturating_sub(64);

    let new_json = new_header
        .to_json_vec()
        .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?;
    if new_json.len() > max_header_len {
        return Err(DnamError::HeaderCapacityExceeded {
            capacity: max_header_len,
            required: new_json.len(),
        });
    }

    container_bytes[64..64 + new_json.len()].copy_from_slice(&new_json);
    container_bytes[64 + new_json.len()..control_table_off].fill(0x20);

    // Re-compile and write CL 0 (ControlClampDescriptor) of each 128-byte ControlChannelRecord[0..P]
    if let Ok(compiled_descs) = new_header.compile_clamp_descriptors() {
        for (p, desc) in compiled_descs.iter().enumerate() {
            let rec_off = control_table_off + p * 128;
            if rec_off + 64 <= arena_off {
                container_bytes[rec_off..rec_off + 64].copy_from_slice(&desc.to_bytes());
            }
        }
    }

    preamble.json_header_byte_len = new_json.len() as u32;
    preamble.num_controls = p_ctrls.min(u16::MAX as usize) as u16;
    if new_header.metadata.signatures.is_some() {
        preamble.flags |= FLAG_HAS_SIGNATURES;
    } else {
        preamble.flags &= !FLAG_HAS_SIGNATURES;
    }
    container_bytes[0..64].copy_from_slice(&preamble.to_bytes());

    let computed_crc = compute_header_crc32c(container_bytes, arena_off);
    container_bytes[0x3C..0x40].copy_from_slice(&computed_crc.to_le_bytes());
    Ok(())
}
