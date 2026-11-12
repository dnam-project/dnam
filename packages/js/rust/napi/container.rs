//! Node-API v10 `.dnam` container & `uplo='U'` Gram matrix bindings
//! (`ref[sdk.ts.model.containers-and-checkpoints]`).

use napi::bindgen_prelude::{Float64Array, Result, Uint8Array};
use napi_derive::napi;

use crate::core::container::{
    build_dmu1_frame_core, build_dnam_container_core, build_reference_container_core,
    canonicalize_jcs_core, crc32c_compute_core, pack_uplo_u_f64_into_core, parse_dmu1_frame_core,
    parse_schema_version_core, strip_container_to_tier_core, strip_v2_open_dnam_container_core,
    unpack_uplo_u_f64_into_core, update_header_in_place_core, uplo_u_packed_len_core,
    validate_and_normalize_header_core, validate_container_with_key_core,
    validate_v2_open_dnam_layout_core,
};
use crate::napi::{MutF64Slice, napi_err};

#[napi(object)]
pub struct NativeValidatedContainer {
    pub magic: String,
    pub version_major: u32,
    pub version_minor: u32,
    pub flags: u32,
    pub num_controls: u32,
    pub sample_rate_train_hz: u32,
    pub json_header_byte_len: u32,
    pub tensor_arena_offset: u32,
    pub tensor_arena_byte_len: u32,
    pub arena_lite_end_cl: u32,
    pub arena_full_end_cl: u32,
    pub arena_pro_end_cl: u32,
    pub arena_ultra_end_cl: u32,
    pub arena_offset_lite_end: u32,
    pub arena_offset_full_end: u32,
    pub arena_offset_pro_end: u32,
    pub arena_offset_ultra_end: u32,
    pub arena_crc32c_lite: u32,
    pub arena_crc32c_full: u32,
    pub arena_crc32c_pro: u32,
    pub arena_crc32c_ultra: u32,
    pub tail_time_ms: f64,
    pub header_crc32c: u32,
    pub highest_tier: u32,
    pub is_stereo_out: bool,
    pub has_signatures: bool,
    pub is_encrypted_arena: bool,
    pub verification_status: String,
    pub protection_profile: String,
    pub header_json: String,
    pub arena_bytes: Uint8Array,
}

impl std::fmt::Debug for NativeValidatedContainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeValidatedContainer")
            .field("magic", &self.magic)
            .field("highest_tier", &self.highest_tier)
            .finish()
    }
}

#[napi]
pub fn validate_container_with_key_native(
    container_bytes: Uint8Array,
    device_id_hash: Option<String>,
    device_secret_32: Option<Uint8Array>,
    trusted_marketplace_pubkey_hex: Option<String>,
) -> Result<NativeValidatedContainer> {
    let info = validate_container_with_key_core(
        &container_bytes,
        device_id_hash.as_deref(),
        device_secret_32.as_deref(),
        trusted_marketplace_pubkey_hex.as_deref(),
    )
    .map_err(napi_err)?;
    let p = info.preamble;
    Ok(NativeValidatedContainer {
        magic: info.magic_str,
        version_major: u32::from(p.version_major),
        version_minor: u32::from(p.version_minor),
        flags: u32::from(p.flags),
        num_controls: u32::from(p.num_controls),
        sample_rate_train_hz: p.sample_rate_train_hz,
        json_header_byte_len: p.json_header_byte_len,
        tensor_arena_offset: p.tensor_arena_offset,
        tensor_arena_byte_len: p.tensor_arena_byte_len,
        arena_lite_end_cl: u32::from(p.arena_lite_end_cl),
        arena_full_end_cl: u32::from(p.arena_full_end_cl),
        arena_pro_end_cl: u32::from(p.arena_pro_end_cl),
        arena_ultra_end_cl: u32::from(p.arena_ultra_end_cl),
        arena_offset_lite_end: p.arena_offset_lite_end(),
        arena_offset_full_end: p.arena_offset_full_end(),
        arena_offset_pro_end: p.arena_offset_pro_end(),
        arena_offset_ultra_end: p.arena_offset_ultra_end(),
        arena_crc32c_lite: p.arena_crc32c_lite,
        arena_crc32c_full: p.arena_crc32c_full,
        arena_crc32c_pro: p.arena_crc32c_pro,
        arena_crc32c_ultra: p.arena_crc32c_ultra,
        tail_time_ms: f64::from(p.tail_time_ms),
        header_crc32c: p.header_crc32c,
        highest_tier: u32::from(p.highest_tier() as u8),
        is_stereo_out: p.is_stereo_out(),
        has_signatures: p.has_signatures(),
        is_encrypted_arena: p.is_encrypted_arena(),
        verification_status: info.verification_status,
        protection_profile: info.protection_profile,
        header_json: info.header_json,
        arena_bytes: Uint8Array::new(info.arena_bytes),
    })
}

#[napi]
pub fn build_reference_container_native(
    supports_freeze: Option<bool>,
    inputs: Option<u8>,
    output_mode: Option<String>,
) -> Result<Uint8Array> {
    let bytes = build_reference_container_core(
        supports_freeze.unwrap_or(true),
        inputs.unwrap_or(2),
        output_mode.as_deref().unwrap_or("stereo"),
    )
    .map_err(napi_err)?;
    Ok(Uint8Array::new(bytes))
}

#[napi]
pub fn build_dnam_container_native(
    header_json: String,
    arena_bytes: Uint8Array,
    encrypted_arena_bytes: Option<Uint8Array>,
    has_signatures_flag: Option<bool>,
) -> Result<Uint8Array> {
    let bytes = build_dnam_container_core(
        &header_json,
        &arena_bytes,
        encrypted_arena_bytes.as_deref(),
        has_signatures_flag.unwrap_or(false),
    )
    .map_err(napi_err)?;
    Ok(Uint8Array::new(bytes))
}

#[napi]
pub fn update_header_in_place_native(
    container_bytes: Uint8Array,
    new_header_json: String,
) -> Result<Uint8Array> {
    let out = update_header_in_place_core(&container_bytes, &new_header_json).map_err(napi_err)?;
    Ok(Uint8Array::new(out))
}

#[napi]
pub fn strip_container_to_tier_native(
    container_bytes: Uint8Array,
    max_tier: u8,
) -> Result<Uint8Array> {
    let out = strip_container_to_tier_core(&container_bytes, max_tier).map_err(napi_err)?;
    Ok(Uint8Array::new(out))
}

#[napi]
pub fn parse_schema_version_native(version_str: String) -> Result<Vec<u32>> {
    parse_schema_version_core(&version_str)
        .map(|arr| arr.to_vec())
        .map_err(napi_err)
}

#[napi]
pub fn crc32c_compute_native(data: Uint8Array) -> u32 {
    crc32c_compute_core(&data)
}

#[napi]
pub fn validate_and_normalize_header_native(header_json: String) -> Result<String> {
    validate_and_normalize_header_core(&header_json).map_err(napi_err)
}

#[napi]
pub fn canonicalize_jcs_native(
    json_str: String,
    strip_mutable: Option<bool>,
    reject_floats: Option<bool>,
) -> Result<String> {
    canonicalize_jcs_core(
        &json_str,
        strip_mutable.unwrap_or(false),
        reject_floats.unwrap_or(false),
    )
    .map_err(napi_err)
}

#[napi]
pub fn build_dmu1_frame_native(
    mutable_json_bytes: Uint8Array,
    capacity: u32,
) -> Result<Uint8Array> {
    build_dmu1_frame_core(&mutable_json_bytes, capacity as usize)
        .map(Uint8Array::new)
        .map_err(napi_err)
}

#[napi]
pub fn parse_dmu1_frame_native(frame_bytes: Uint8Array) -> Result<String> {
    parse_dmu1_frame_core(&frame_bytes).map_err(napi_err)
}

#[napi]
pub fn validate_v2_open_dnam_layout_native(container_bytes: Uint8Array) -> Result<String> {
    validate_v2_open_dnam_layout_core(&container_bytes).map_err(napi_err)
}

#[napi]
pub fn strip_v2_open_dnam_container_native(
    container_bytes: Uint8Array,
    target_tier: u8,
) -> Result<Uint8Array> {
    strip_v2_open_dnam_container_core(&container_bytes, target_tier)
        .map(Uint8Array::new)
        .map_err(napi_err)
}

#[napi]
pub fn uplo_u_packed_len_native(n_dim: u32) -> u32 {
    uplo_u_packed_len_core(n_dim as usize) as u32
}

#[napi]
pub fn pack_uplo_u_f64_into_native(
    dense_flat: Float64Array,
    n_dim: u32,
    out_packed: MutF64Slice<'_>,
) -> Result<()> {
    pack_uplo_u_f64_into_core(&dense_flat, n_dim as usize, out_packed).map_err(napi_err)
}

#[napi]
pub fn unpack_uplo_u_f64_into_native(
    packed_u: Float64Array,
    n_tier: u32,
    out_dense: MutF64Slice<'_>,
) -> Result<()> {
    unpack_uplo_u_f64_into_core(&packed_u, n_tier as usize, out_dense).map_err(napi_err)
}
