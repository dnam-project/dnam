//! WASM `.dnam` container, `.dnam-ckpt` checkpoint & `uplo='U'` Gram matrix exports
//! (`ref[sdk.ts.model.containers-and-checkpoints]`, `ref[sdk.ts.runtime.zero-unsafe]`).

use crate::core::container::{
    build_dmu1_frame_core, build_dnam_container_core, build_reference_checkpoint_core,
    build_reference_container_core, canonicalize_jcs_core, crc32c_compute_core,
    export_runtime_dnam_from_ckpt_core, pack_uplo_u_f64_into_core, parse_dmu1_frame_core,
    parse_schema_version_core, strip_container_to_tier_core, strip_v2_open_dnam_container_core,
    unpack_uplo_u_f64_into_core, update_header_in_place_core, uplo_u_packed_len_core,
    validate_and_normalize_header_core, validate_container_with_key_core,
    validate_v2_dnck_checkpoint_core, validate_v2_open_dnam_layout_core,
    verify_arena5_calib_stats_core, verify_stem_arena4_crc32c_core,
};
use crate::error::validation_err;
use crate::wasm::abi::{
    read_bytes, read_f64_vec, read_opt_bytes, read_opt_str, read_str, set_arena, set_error,
    set_result, with_f64_mut, write_u32_slice,
};

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_validate_container_with_key(
    container_ptr: usize,
    container_len: usize,
    dev_id_ptr: usize,
    dev_id_len: usize,
    dev_sec_ptr: usize,
    dev_sec_len: usize,
    mkt_pk_ptr: usize,
    mkt_pk_len: usize,
) -> i32 {
    let container_bytes = read_bytes(container_ptr, container_len);
    let dev_id = match read_opt_str(dev_id_ptr, dev_id_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let dev_sec = read_opt_bytes(dev_sec_ptr, dev_sec_len);
    let mkt_pk = match read_opt_str(mkt_pk_ptr, mkt_pk_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match validate_container_with_key_core(
        &container_bytes,
        dev_id.as_deref(),
        dev_sec.as_deref(),
        mkt_pk.as_deref(),
    ) {
        Ok(info) => {
            let p = info.preamble;
            let meta = serde_json::json!({
                "magic": info.magic_str,
                "versionMajor": p.version_major,
                "versionMinor": p.version_minor,
                "flags": p.flags,
                "numControls": p.num_controls,
                "sampleRateTrainHz": p.sample_rate_train_hz,
                "jsonHeaderByteLen": p.json_header_byte_len,
                "tensorArenaOffset": p.tensor_arena_offset,
                "tensorArenaByteLen": p.tensor_arena_byte_len,
                "arenaLiteEndCl": p.arena_lite_end_cl,
                "arenaFullEndCl": p.arena_full_end_cl,
                "arenaProEndCl": p.arena_pro_end_cl,
                "arenaUltraEndCl": p.arena_ultra_end_cl,
                "arenaOffsetLiteEnd": p.arena_offset_lite_end(),
                "arenaOffsetFullEnd": p.arena_offset_full_end(),
                "arenaOffsetProEnd": p.arena_offset_pro_end(),
                "arenaOffsetUltraEnd": p.arena_offset_ultra_end(),
                "arenaCrc32cLite": p.arena_crc32c_lite,
                "arenaCrc32cFull": p.arena_crc32c_full,
                "arenaCrc32cPro": p.arena_crc32c_pro,
                "arenaCrc32cUltra": p.arena_crc32c_ultra,
                "numCkptStems": p.num_ckpt_stems,
                "tailTimeMs": p.tail_time_ms,
                "headerCrc32c": p.header_crc32c,
                "highestTier": p.highest_tier() as u8,
                "isStereoOut": p.is_stereo_out(),
                "hasSignatures": p.has_signatures(),
                "isEncryptedArena": p.is_encrypted_arena(),
                "hasCalibStats": p.has_calib_stats(),
                "verificationStatus": info.verification_status,
                "protectionProfile": info.protection_profile,
                "headerJson": info.header_json,
            });
            set_arena(info.arena_bytes);
            match serde_json::to_vec(&meta) {
                Ok(vec) => set_result(vec),
                Err(e) => set_error(validation_err(e.to_string())),
            }
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_build_reference_container(
    supports_freeze: u8,
    inputs: u8,
    out_mode_ptr: usize,
    out_mode_len: usize,
) -> i32 {
    let mode = match read_str(out_mode_ptr, out_mode_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match build_reference_container_core(supports_freeze != 0, inputs, &mode) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_build_dnam_container(
    hdr_ptr: usize,
    hdr_len: usize,
    arena_ptr: usize,
    arena_len: usize,
    enc_ptr: usize,
    enc_len: usize,
    has_signatures: u8,
) -> i32 {
    let hdr_str = match read_str(hdr_ptr, hdr_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let arena = read_bytes(arena_ptr, arena_len);
    let enc_opt = read_opt_bytes(enc_ptr, enc_len);
    match build_dnam_container_core(&hdr_str, &arena, enc_opt.as_deref(), has_signatures != 0) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_update_header_in_place(
    container_ptr: usize,
    container_len: usize,
    hdr_ptr: usize,
    hdr_len: usize,
) -> i32 {
    let container = read_bytes(container_ptr, container_len);
    let hdr_str = match read_str(hdr_ptr, hdr_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match update_header_in_place_core(&container, &hdr_str) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_strip_container_to_tier(
    container_ptr: usize,
    container_len: usize,
    max_tier: u8,
) -> i32 {
    let container = read_bytes(container_ptr, container_len);
    match strip_container_to_tier_core(&container, max_tier) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_parse_schema_version(
    ver_ptr: usize,
    ver_len: usize,
    out4_ptr: usize,
) -> i32 {
    if out4_ptr == 0 {
        return set_error("[DNAM-VALIDATION] null out4_ptr".to_string());
    }
    let ver_str = match read_str(ver_ptr, ver_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match parse_schema_version_core(&ver_str) {
        Ok(arr) => {
            if !write_u32_slice(out4_ptr, &arr) {
                return set_error("[DNAM-VALIDATION] invalid out4_ptr".to_string());
            }
            0
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_crc32c(data_ptr: usize, data_len: usize) -> u32 {
    let data = read_bytes(data_ptr, data_len);
    crc32c_compute_core(&data)
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_canonicalize_jcs(
    json_ptr: usize,
    json_len: usize,
    strip_mutable: u8,
    reject_floats: u8,
) -> i32 {
    let json_str = match read_str(json_ptr, json_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match canonicalize_jcs_core(&json_str, strip_mutable != 0, reject_floats != 0) {
        Ok(s) => set_result(s.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_validate_and_normalize_header(hdr_ptr: usize, hdr_len: usize) -> i32 {
    let hdr_str = match read_str(hdr_ptr, hdr_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match validate_and_normalize_header_core(&hdr_str) {
        Ok(s) => set_result(s.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_build_dmu1_frame(
    json_ptr: usize,
    json_len: usize,
    capacity: usize,
) -> i32 {
    let json_bytes = read_bytes(json_ptr, json_len);
    match build_dmu1_frame_core(&json_bytes, capacity) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_parse_dmu1_frame(frame_ptr: usize, frame_len: usize) -> i32 {
    let frame = read_bytes(frame_ptr, frame_len);
    match parse_dmu1_frame_core(&frame) {
        Ok(s) => set_result(s.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_validate_v2_open_dnam_layout(
    container_ptr: usize,
    container_len: usize,
) -> i32 {
    let container = read_bytes(container_ptr, container_len);
    match validate_v2_open_dnam_layout_core(&container) {
        Ok(s) => set_result(s.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_strip_v2_open_dnam_container(
    container_ptr: usize,
    container_len: usize,
    target_tier: u8,
) -> i32 {
    let container = read_bytes(container_ptr, container_len);
    match strip_v2_open_dnam_container_core(&container, target_tier) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_build_reference_checkpoint(include_calib_stats: u8) -> i32 {
    match build_reference_checkpoint_core(include_calib_stats != 0) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_verify_stem_arena4_crc32c(
    arena_ptr: usize,
    arena_len: usize,
    stem_offset: usize,
    stem_len: usize,
    expected_crc32c: u32,
) -> i32 {
    let arena = read_bytes(arena_ptr, arena_len);
    match verify_stem_arena4_crc32c_core(&arena, stem_offset, stem_len, expected_crc32c) {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_verify_arena5_calib_stats(
    arena_ptr: usize,
    arena_len: usize,
    arena5_offset: usize,
) -> i32 {
    let arena = read_bytes(arena_ptr, arena_len);
    match verify_arena5_calib_stats_core(&arena, arena5_offset) {
        Ok(len) => len as i32,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_validate_v2_dnck_checkpoint(ckpt_ptr: usize, ckpt_len: usize) -> i32 {
    let ckpt = read_bytes(ckpt_ptr, ckpt_len);
    match validate_v2_dnck_checkpoint_core(&ckpt) {
        Ok((tier, ends, boundary)) => {
            let val = serde_json::json!({
                "retainedTier": tier,
                "bundleEnds": ends,
                "mutableBoundary": boundary,
            });
            match serde_json::to_vec(&val) {
                Ok(vec) => set_result(vec),
                Err(e) => set_error(validation_err(e.to_string())),
            }
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_export_runtime_dnam_from_ckpt(ckpt_ptr: usize, ckpt_len: usize) -> i32 {
    let ckpt = read_bytes(ckpt_ptr, ckpt_len);
    match export_runtime_dnam_from_ckpt_core(&ckpt) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_uplo_u_packed_len(n_dim: usize) -> usize {
    uplo_u_packed_len_core(n_dim)
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_pack_uplo_u_f64_into(
    dense_ptr: usize,
    dense_len: usize,
    n_dim: usize,
    out_ptr: usize,
    out_len: usize,
) -> i32 {
    let dense = read_f64_vec(dense_ptr, dense_len);
    let Some(res) = with_f64_mut(out_ptr, out_len, |out| {
        pack_uplo_u_f64_into_core(&dense, n_dim, out)
    }) else {
        return set_error("[DNAM-VALIDATION] invalid out_ptr".to_string());
    };
    match res {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_unpack_uplo_u_f64_into(
    packed_ptr: usize,
    packed_len: usize,
    n_tier: usize,
    out_ptr: usize,
    out_len: usize,
) -> i32 {
    let packed = read_f64_vec(packed_ptr, packed_len);
    let Some(res) = with_f64_mut(out_ptr, out_len, |out| {
        unpack_uplo_u_f64_into_core(&packed, n_tier, out)
    }) else {
        return set_error("[DNAM-VALIDATION] invalid out_ptr".to_string());
    };
    match res {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}
