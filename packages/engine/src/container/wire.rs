//! Canonical v2.0 `.dnam` (`b"DNAM"`) & `.dnam-ckpt` (`b"DNCK"`) Hybrid Binary Container Wire Framing,
//! RFC 8785 JCS Header Projection, `DMU1`/`DMUE` Mutable Suffix, `CP1` Control Projections,
//! Structured Conditional Selectors, Delay History Declarations, and Zero-Copy Checkpoint Export.
//!
//! Implements:
//! - `ref[container.dnam.formats]`, `ref[container.dnam.wire-framing]`,
//!   `ref[container.dnam.preamble-directory]`, `ref[container.dnam.open-layout]`
//! - `ref[container.dnam.mutable-frame]`, `ref[container.dnam.mutable-json]`
//! - `ref[container.dnam.tier-prefix]`, `ref[container.dnam.delay-histories]`,
//!   `ref[container.dnam.load-validation]`, `ref[container.dnam.stage3-slices]`,
//!   `ref[container.dnam.cp1-profile]`, `ref[container.dnam.conditional-rules]`
//! - `ref[container.dnam.metadata-ui-hints]`, `ref[container.dnam.sample-rate-anchor]`,
//!   `ref[container.dnam.jcs-header-schema]`, `ref[container.dnam.security-crossref]`
//! - `ref[container.ckpt.bundle-framing]`, `ref[container.ckpt.dca1-signature]`,
//!   `ref[container.ckpt.export-template]`, `ref[container.ckpt.declarative-no-pickle]`,
//!   `ref[container.ckpt.uplo-u]`

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use serde_json::Value;

use crate::math::crc32c;
use crate::provenance::sha256_digest;
use crate::schema::{DnamError, SchemaOptionExt, cold_ln_f64, cold_pow_f64, reject_if};

/// Fixed 64-byte preamble size (`0..64`, `ref[container.dnam.preamble-directory]`).
pub const V2_PREAMBLE_SIZE: usize = 64;
/// Fixed 128-byte directory size (`64..192`, `ref[container.dnam.preamble-directory]`).
pub const V2_DIRECTORY_SIZE: usize = 128;
/// Fixed offset where the immutable header begins (`192`, `ref[container.dnam.preamble-directory]`).
pub const V2_HEADER_OFFSET: usize = V2_PREAMBLE_SIZE + V2_DIRECTORY_SIZE;
/// Fixed 16-byte `DMU1` mutable frame header size (`ref[container.dnam.mutable-frame]`).
pub const DMU1_HEADER_SIZE: usize = 16;
/// Fixed 16-byte `DMUE` mutable footer size (`ref[container.dnam.mutable-frame]`).
pub const DMUE_FOOTER_SIZE: usize = 16;
/// Maximum default mutable frame capacity for bounded structural validation (`4,096 B`).
pub const DEFAULT_MAX_MUTABLE_CAPACITY: usize = 4_096;

/// Domain-separated binary field encoder `F(label, value) = u32le(len(label)) || label || u64le(len(value)) || value`
/// (`ref[container.dnam.wire-framing]`, `ref[sec.sig.prefix-delegation]`).
pub fn encode_domain_field(label: &str, value: &[u8]) -> Vec<u8> {
    let label_bytes = label.as_bytes();
    let mut out = Vec::with_capacity(4 + label_bytes.len() + 8 + value.len());
    out.extend_from_slice(&(label_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(label_bytes);
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value);
    out
}

/// Aligns `value` up to a 64-byte boundary with `u64` overflow protection (`ref[container.dnam.preamble-directory]`).
#[inline]
pub const fn align64_u64(value: u64) -> Option<u64> {
    if value > u64::MAX - 63 {
        None
    } else {
        Some((value + 63) & !63)
    }
}

/// Validated v2.0 Open `.dnam` or `.dnam-ckpt` layout (`ref[container.dnam.open-layout]`, `ref[container.ckpt.bundle-framing]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct V2ContainerLayout {
    pub magic: [u8; 4],
    pub protection_profile: u8,
    pub auth_kind: u8,
    pub maximum_tier: u8,
    pub retained_tier: u8,
    pub sample_rate_train_hz: u32,
    pub header_extent: (usize, usize),
    pub shared_extent: (usize, usize),
    pub auth_extent: (usize, usize),
    pub tier_start: usize,
    pub tier_ends: [usize; 4],
    pub mutable_extent: (usize, usize),
}

/// Builds a canonical v2.0 Open `.dnam` container (`magic = b"DNAM"`, `version = (2, 0)`,
/// `ref[container.dnam.open-layout]`).
pub fn build_v2_open_dnam_container(
    header_bytes: &[u8],
    shared_bytes: &[u8],
    auth_bytes: &[u8],
    tier_slices: &[&[u8]],
    mutable_frame_bytes: Option<&[u8]>,
    auth_kind: u8,
) -> Result<Vec<u8>, DnamError> {
    reject_if(
        header_bytes.is_empty()
            || auth_bytes.is_empty()
            || tier_slices.is_empty()
            || tier_slices.len() > 4
            || auth_kind > 1,
        "invalid v2 open container inputs",
    )?;
    for slice in tier_slices {
        reject_if(
            slice.is_empty() || !slice.len().is_multiple_of(64),
            "tier slice must be non-empty and 64-byte aligned",
        )?;
    }
    let header_offset = V2_HEADER_OFFSET as u64;
    let header_len = header_bytes.len() as u64;
    let shared_offset = align64_u64(header_offset + header_len).unwrap();
    let shared_len = shared_bytes.len() as u64;
    let auth_offset = align64_u64(shared_offset + shared_len).unwrap();
    let auth_len = auth_bytes.len() as u64;
    let tier_start = align64_u64(auth_offset + auth_len).unwrap();

    let mut ends = [0u64; 4];
    let mut cur_end = tier_start;
    for (i, slice) in tier_slices.iter().enumerate() {
        cur_end = cur_end.checked_add(slice.len() as u64).unwrap();
        ends[i] = cur_end;
    }

    let mut out = vec![0u8; cur_end as usize];
    // 1. 64B Preamble
    out[0..4].copy_from_slice(b"DNAM");
    out[4..6].copy_from_slice(&2u16.to_le_bytes());
    out[6..8].copy_from_slice(&0u16.to_le_bytes());
    out[8] = 0; // protection_profile = Open
    out[9] = auth_kind;
    out[10] = (tier_slices.len() - 1) as u8;
    out[11] = 0;
    out[12..16].copy_from_slice(&48_000u32.to_le_bytes());

    // 2. 128B Directory
    let dir_fields: [u64; 13] = [
        header_offset,
        header_len,
        shared_offset,
        shared_len,
        auth_offset,
        auth_len,
        0,
        0,
        tier_start,
        ends[0],
        ends[1],
        ends[2],
        ends[3],
    ];
    for (idx, val) in dir_fields.iter().enumerate() {
        let off = V2_PREAMBLE_SIZE + idx * 8;
        out[off..off + 8].copy_from_slice(&val.to_le_bytes());
    }

    // 3. Regions
    let h_off = header_offset as usize;
    out[h_off..h_off + header_bytes.len()].copy_from_slice(header_bytes);
    let s_off = shared_offset as usize;
    out[s_off..s_off + shared_bytes.len()].copy_from_slice(shared_bytes);
    let a_off = auth_offset as usize;
    out[a_off..a_off + auth_bytes.len()].copy_from_slice(auth_bytes);

    let mut t_off = tier_start as usize;
    for slice in tier_slices {
        out[t_off..t_off + slice.len()].copy_from_slice(slice);
        t_off += slice.len();
    }

    let default_dmu1;
    let m_bytes = match mutable_frame_bytes {
        Some(b) if !b.is_empty() => b,
        _ => {
            default_dmu1 = build_dmu1_mutable_frame(b"{}", 18)?;
            &default_dmu1
        }
    };
    append_dmue_mutable_suffix(&mut out, m_bytes);
    Ok(out)
}

/// Appends `mutable_frame` and the 16-byte `DMUE` footer to `container_prefix` (`ref[container.dnam.mutable-frame]`).
pub fn append_dmue_mutable_suffix(container_prefix: &mut Vec<u8>, mutable_frame: &[u8]) {
    container_prefix.extend_from_slice(mutable_frame);
    container_prefix.extend_from_slice(b"DMUE");
    container_prefix.extend_from_slice(&(mutable_frame.len() as u64).to_le_bytes());
    container_prefix.extend_from_slice(&0u32.to_le_bytes());
}

/// Builds a `DMU1` open mutable frame of total capacity `capacity >= 16 + payload.len()`
/// (`ref[container.dnam.mutable-frame]`).
pub fn build_dmu1_mutable_frame(payload: &[u8], capacity: usize) -> Result<Vec<u8>, DnamError> {
    reject_if(
        payload.len() < 2 || capacity < DMU1_HEADER_SIZE + payload.len(),
        "invalid DMU1 payload or capacity",
    )?;
    let len_le = (payload.len() as u64).to_le_bytes();
    let mut crc_input = Vec::with_capacity(12 + payload.len());
    crc_input.extend_from_slice(b"DMU1");
    crc_input.extend_from_slice(&len_le);
    crc_input.extend_from_slice(payload);
    let checksum = crc32c(&crc_input);

    let mut frame = vec![0u8; capacity];
    frame[0..4].copy_from_slice(b"DMU1");
    frame[4..8].copy_from_slice(&checksum.to_le_bytes());
    frame[8..16].copy_from_slice(&len_le);
    frame[16..16 + payload.len()].copy_from_slice(payload);
    Ok(frame)
}

/// Validates a `DMU1` mutable frame (CRC32C, zero padding, and strict RFC 8259 UTF-8 JSON object without
/// BOM or duplicate keys) and returns the parsed JSON `Value` (`ref[container.dnam.mutable-frame]`).
pub fn parse_and_validate_dmu1_frame(frame: &[u8]) -> Result<Value, DnamError> {
    reject_if(
        frame.len() < DMU1_HEADER_SIZE + 2 || &frame[0..4] != b"DMU1",
        "invalid DMU1 magic or short frame",
    )?;
    let stored_crc = u32::from_le_bytes([frame[4], frame[5], frame[6], frame[7]]);
    let n_len = u64::from_le_bytes(frame[8..16].try_into().unwrap()) as usize;
    reject_if(
        n_len < 2 || n_len > frame.len() - DMU1_HEADER_SIZE,
        "invalid DMU1 JSON payload length",
    )?;
    let payload = &frame[16..16 + n_len];
    reject_if(
        frame[16 + n_len..].iter().any(|&b| b != 0),
        "non-zero DMU1 padding bytes",
    )?;
    let mut crc_input = Vec::with_capacity(12 + n_len);
    crc_input.extend_from_slice(b"DMU1");
    crc_input.extend_from_slice(&frame[8..16]);
    crc_input.extend_from_slice(payload);
    let calc_crc = crc32c(&crc_input);
    if stored_crc != calc_crc {
        return Err(DnamError::Crc32cMismatch {
            region: "DMU1",
            expected: stored_crc,
            computed: calc_crc,
        });
    }
    parse_strict_json_object_bytes(payload)
}

/// Validates a mutable JSON document against the normative schema (`version`, `presets`,
/// `user_metadata`, `control_user_metadata`, `stem_user_metadata`, `ref[container.dnam.mutable-json]`).
pub fn validate_mutable_json_schema(doc: &Value) -> Result<(), DnamError> {
    let obj = doc
        .as_object()
        .or_schema_err("mutable JSON must be an object")?;
    for key in obj.keys() {
        if !matches!(
            key.as_str(),
            "version"
                | "presets"
                | "user_metadata"
                | "control_user_metadata"
                | "stem_user_metadata"
        ) {
            return Err(DnamError::SchemaValidationError(alloc::format!(
                "unknown mutable member: {key}"
            )));
        }
    }
    if let Some(ver) = obj.get("version")
        && ver.as_i64() != Some(1)
        && ver.as_u64() != Some(1)
    {
        return Err(DnamError::SchemaValidationError(
            "unsupported version".to_string(),
        ));
    }
    for meta_key in [
        "user_metadata",
        "control_user_metadata",
        "stem_user_metadata",
    ] {
        if let Some(v) = obj.get(meta_key) {
            let map = v.as_object().or_schema_err("invalid metadata object")?;
            if meta_key != "user_metadata" {
                for item in map.values() {
                    reject_if(!item.is_object(), "invalid keyed metadata")?;
                }
            }
        }
    }
    if let Some(presets_val) = obj.get("presets") {
        let presets = presets_val.as_array().or_schema_err("invalid presets")?;
        let mut seen_ids = BTreeSet::new();
        for preset_val in presets {
            let p_obj = preset_val.as_object().or_schema_err("invalid preset")?;
            for k in p_obj.keys() {
                reject_if(
                    !matches!(
                        k.as_str(),
                        "id" | "name" | "description" | "controls" | "user_metadata"
                    ),
                    "invalid preset",
                )?;
            }
            let id = p_obj.get("id").and_then(Value::as_str).unwrap_or("");
            reject_if(
                id.is_empty() || !seen_ids.insert(id.to_string()),
                "invalid preset id",
            )?;
            let name = p_obj.get("name").and_then(Value::as_str).unwrap_or("");
            reject_if(name.is_empty(), "invalid preset name")?;
            if let Some(desc) = p_obj.get("description")
                && !desc.is_null()
                && !desc.is_string()
            {
                return Err(DnamError::SchemaValidationError(
                    "invalid description".to_string(),
                ));
            }
            if let Some(um) = p_obj.get("user_metadata")
                && !um.is_object()
            {
                return Err(DnamError::SchemaValidationError(
                    "invalid preset metadata".to_string(),
                ));
            }
            let controls = p_obj
                .get("controls")
                .and_then(Value::as_object)
                .or_schema_err("invalid control map")?;
            for cv in controls.values() {
                let valid_val =
                    cv.is_string() || (cv.is_number() && cv.as_f64().is_some_and(f64::is_finite));
                reject_if(!valid_val, "invalid control value")?;
            }
        }
    }
    Ok(())
}

/// Validates a v2.0 Open `.dnam` container's physical layout (`ref[container.dnam.open-layout]`,
/// `ref[container.dnam.preamble-directory]`, `ref[container.dnam.mutable-frame]`).
pub fn validate_v2_open_dnam_layout(data: &[u8]) -> Result<V2ContainerLayout, DnamError> {
    reject_if(data.len() < V2_HEADER_OFFSET, "short structure")?;
    let magic: [u8; 4] = data[0..4].try_into().unwrap();
    let major = u16::from_le_bytes([data[4], data[5]]);
    let minor = u16::from_le_bytes([data[6], data[7]]);
    let profile = data[8];
    let auth_kind = data[9];
    let maximum = data[10];
    let reserved = data[11];
    let rate = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
    reject_if(
        magic != *b"DNAM"
            || (major, minor) != (2, 0)
            || profile != 0
            || auth_kind > 1
            || maximum > 3
            || reserved != 0
            || rate != 48_000
            || data[16..64].iter().any(|&b| b != 0),
        "invalid preamble",
    )?;

    let read_u64 = |idx: usize| -> u64 {
        let off = V2_PREAMBLE_SIZE + idx * 8;
        u64::from_le_bytes(data[off..off + 8].try_into().unwrap())
    };
    let header_offset = read_u64(0);
    let header_length = read_u64(1);
    let shared_offset = read_u64(2);
    let shared_length = read_u64(3);
    let auth_offset = read_u64(4);
    let auth_length = read_u64(5);
    let reserved_offset = read_u64(6);
    let reserved_capacity = read_u64(7);
    let tier_start = read_u64(8);
    let ends_u64 = [read_u64(9), read_u64(10), read_u64(11), read_u64(12)];

    reject_if(
        data[V2_PREAMBLE_SIZE + 13 * 8..V2_HEADER_OFFSET]
            .iter()
            .any(|&b| b != 0)
            || reserved_offset != 0
            || reserved_capacity != 0
            || header_offset != (V2_HEADER_OFFSET as u64)
            || header_length == 0,
        "invalid directory",
    )?;

    let header_end = header_offset
        .checked_add(header_length)
        .or_schema_err("header extent overflow")?;
    reject_if(
        Some(shared_offset) != align64_u64(header_end),
        "invalid directory",
    )?;
    let shared_end = shared_offset
        .checked_add(shared_length)
        .or_schema_err("shared extent overflow")?;
    reject_if(
        Some(auth_offset) != align64_u64(shared_end) || auth_length == 0,
        "invalid directory",
    )?;
    let auth_end = auth_offset
        .checked_add(auth_length)
        .or_schema_err("auth extent overflow")?;
    reject_if(
        Some(tier_start) != align64_u64(auth_end),
        "invalid tier start",
    )?;

    let mut previous = tier_start;
    for (index, &end) in ends_u64.iter().enumerate() {
        if index <= (maximum as usize) {
            reject_if(
                end <= previous || !end.is_multiple_of(64),
                "invalid tier end",
            )?;
            previous = end;
        } else {
            reject_if(end != 0, "unexpected tier end")?;
        }
    }

    reject_if(
        data.len() < V2_HEADER_OFFSET + DMUE_FOOTER_SIZE,
        "invalid mutable trailer",
    )?;
    let footer = &data[data.len() - DMUE_FOOTER_SIZE..];
    let footer_magic = &footer[0..4];
    let mutable_capacity = u64::from_le_bytes(footer[4..12].try_into().unwrap());
    let footer_reserved = u32::from_le_bytes(footer[12..16].try_into().unwrap());
    reject_if(
        footer_magic != b"DMUE"
            || footer_reserved != 0
            || mutable_capacity < (DMU1_HEADER_SIZE as u64 + 2)
            || mutable_capacity > (DEFAULT_MAX_MUTABLE_CAPACITY as u64)
            || mutable_capacity > ((data.len() - DMUE_FOOTER_SIZE) as u64),
        "invalid mutable trailer",
    )?;
    let mutable_offset = data.len() - DMUE_FOOTER_SIZE - (mutable_capacity as usize);
    let mut retained_opt = None;
    for (idx, &end) in ends_u64[..=(maximum as usize)].iter().enumerate() {
        if end == (mutable_offset as u64) {
            retained_opt = Some(idx as u8);
            break;
        }
    }
    let retained_tier = retained_opt.or_schema_err("invalid retained end")?;

    parse_and_validate_dmu1_frame(&data[mutable_offset..data.len() - DMUE_FOOTER_SIZE])?;

    for (start, end) in [
        (header_end as usize, shared_offset as usize),
        (shared_end as usize, auth_offset as usize),
        (auth_end as usize, tier_start as usize),
    ] {
        reject_if(data[start..end].iter().any(|&b| b != 0), "nonzero padding")?;
    }

    Ok(V2ContainerLayout {
        magic,
        protection_profile: profile,
        auth_kind,
        maximum_tier: maximum,
        retained_tier,
        sample_rate_train_hz: rate,
        header_extent: (header_offset as usize, header_length as usize),
        shared_extent: (shared_offset as usize, shared_length as usize),
        auth_extent: (auth_offset as usize, auth_length as usize),
        tier_start: tier_start as usize,
        tier_ends: [
            ends_u64[0] as usize,
            ends_u64[1] as usize,
            ends_u64[2] as usize,
            ends_u64[3] as usize,
        ],
        mutable_extent: (mutable_offset, mutable_capacity as usize),
    })
}

/// Strips a validated v2.0 Open `.dnam` container to `target_tier <= retained_tier` without re-signing
/// (`ref[container.dnam.open-layout]`, `ref[container.dnam.tier-prefix]`).
pub fn strip_v2_open_dnam_container(data: &[u8], target_tier: u8) -> Result<Vec<u8>, DnamError> {
    let layout = validate_v2_open_dnam_layout(data)?;
    reject_if(target_tier > layout.retained_tier, "invalid strip tier")?;
    let tier_end = layout.tier_ends[target_tier as usize];
    let mutable_offset = layout.mutable_extent.0;
    let mut out = Vec::with_capacity(tier_end + (data.len() - mutable_offset));
    out.extend_from_slice(&data[..tier_end]);
    out.extend_from_slice(&data[mutable_offset..]);
    Ok(out)
}

/// Computes the cumulative SHA-256 prefix digest of the retained tier in a v2.0 Open `.dnam` container
/// under `domain` (e.g. `b"DNAM-MASTER-PREFIX-V1"`, `ref[container.dnam.open-layout]`, `ref[sec.sig.prefix-delegation]`).
pub fn compute_v2_open_retained_prefix_sha256(
    data: &[u8],
    domain: &[u8],
) -> Result<[u8; 32], DnamError> {
    let layout = validate_v2_open_dnam_layout(data)?;
    let (h_off, h_len) = layout.header_extent;
    let (s_off, s_len) = layout.shared_extent;
    let header_sha = sha256_digest(&data[h_off..h_off + h_len]);

    let mut shared = Vec::new();
    shared.extend_from_slice(&encode_domain_field("domain", b"DNAM-PLAINTEXT-SHARED-V1"));
    shared.extend_from_slice(&encode_domain_field("preamble", &data[..V2_PREAMBLE_SIZE]));
    shared.extend_from_slice(&encode_domain_field(
        "directory",
        &data[V2_PREAMBLE_SIZE..V2_HEADER_OFFSET],
    ));
    shared.extend_from_slice(&encode_domain_field(
        "shared_model_binary",
        &data[s_off..s_off + s_len],
    ));

    let mut transcript = Vec::new();
    transcript.extend_from_slice(&encode_domain_field("domain", domain));
    transcript.extend_from_slice(&encode_domain_field("immutable_header_sha256", &header_sha));
    transcript.extend_from_slice(&encode_domain_field("shared_model_data", &shared));

    let labels = ["tier_lite", "tier_full", "tier_pro", "tier_ultra"];
    let mut start = layout.tier_start;
    for t in 0..=(layout.retained_tier as usize) {
        let end = layout.tier_ends[t];
        transcript.extend_from_slice(&encode_domain_field(labels[t], &data[start..end]));
        start = end;
    }
    Ok(sha256_digest(&transcript))
}

// ============================================================================
// Strict JSON Parsing & RFC 8785 JCS Header Projection (`ref[container.dnam.jcs-header-schema]`)
// ============================================================================

/// Returns `true` if `s` contains no Unicode noncharacters (`0xFDD0..=0xFDEF`, `..0xFFFE/0xFFFF`).
pub fn is_valid_jcs_unicode_text(s: &str) -> bool {
    for ch in s.chars() {
        let code = ch as u32;
        if (0xFDD0..=0xFDEF).contains(&code) || (code & 0xFFFF) >= 0xFFFE {
            return false;
        }
    }
    true
}

fn validate_json_value_unicode_and_finite(v: &Value) -> Result<(), DnamError> {
    match v {
        Value::Null | Value::Bool(_) => Ok(()),
        Value::Number(n) => {
            reject_if(
                n.as_f64().is_none_or(|f| !f.is_finite()),
                "non-finite number",
            )?;
            Ok(())
        }
        Value::String(s) => {
            reject_if(!is_valid_jcs_unicode_text(s), "invalid Unicode")?;
            Ok(())
        }
        Value::Array(arr) => {
            for item in arr {
                validate_json_value_unicode_and_finite(item)?;
            }
            Ok(())
        }
        Value::Object(map) => {
            for (k, val) in map {
                reject_if(!is_valid_jcs_unicode_text(k), "invalid Unicode")?;
                validate_json_value_unicode_and_finite(val)?;
            }
            Ok(())
        }
    }
}

/// Checks raw JSON UTF-8 bytes for duplicate object keys (including `\uXXXX` escapes), BOM, and invalid
/// Unicode, and parses a top-level JSON object (`ref[container.dnam.jcs-header-schema]`).
pub fn parse_strict_json_object_bytes(data: &[u8]) -> Result<Value, DnamError> {
    reject_if(
        data.starts_with(&[0xEF, 0xBB, 0xBF]),
        "UTF-8 BOM is forbidden",
    )?;
    let text = core::str::from_utf8(data)
        .map_err(|_| DnamError::SchemaValidationError("invalid UTF-8 JSON".to_string()))?;
    check_no_duplicate_json_keys(text)?;
    let val: Value = serde_json::from_str(text)
        .map_err(|e| DnamError::SchemaValidationError(alloc::format!("invalid JSON: {e}")))?;
    validate_json_value_unicode_and_finite(&val)?;
    reject_if(!val.is_object(), "header must be an object")?;
    Ok(val)
}

fn check_no_duplicate_json_keys(text: &str) -> Result<(), DnamError> {
    let bytes = text.as_bytes();
    let mut i = 0;
    let mut stack: Vec<(bool, BTreeSet<String>, bool)> = Vec::new(); // (is_object, seen_keys, expect_key)

    while i < bytes.len() {
        let b = bytes[i];
        match b {
            b' ' | b'\t' | b'\r' | b'\n' => {
                i += 1;
            }
            b'{' => {
                stack.push((true, BTreeSet::new(), true));
                i += 1;
            }
            b'}' => {
                stack.pop();
                if let Some((true, _, expect_key)) = stack.last_mut() {
                    *expect_key = false;
                }
                i += 1;
            }
            b'[' => {
                stack.push((false, BTreeSet::new(), false));
                i += 1;
            }
            b']' => {
                stack.pop();
                if let Some((true, _, expect_key)) = stack.last_mut() {
                    *expect_key = false;
                }
                i += 1;
            }
            b',' => {
                if let Some((is_obj, _, expect_key)) = stack.last_mut()
                    && *is_obj
                {
                    *expect_key = true;
                }
                i += 1;
            }
            b':' => {
                if let Some((is_obj, _, expect_key)) = stack.last_mut()
                    && *is_obj
                {
                    *expect_key = false;
                }
                i += 1;
            }
            b'"' => {
                let start = i;
                i += 1;
                while i < bytes.len() {
                    if bytes[i] == b'\\' {
                        i += 2;
                    } else if bytes[i] == b'"' {
                        i += 1;
                        break;
                    } else {
                        i += 1;
                    }
                }
                let raw_str = &text[start..i];
                if let Some((true, seen, expect_key)) = stack.last_mut()
                    && *expect_key
                {
                    let parsed_key: String = serde_json::from_str(raw_str).map_err(|_| {
                        DnamError::SchemaValidationError("invalid Unicode".to_string())
                    })?;
                    reject_if(!seen.insert(parsed_key), "duplicate object member")?;
                }
            }
            _ => {
                i += 1;
            }
        }
    }
    Ok(())
}

/// Projects out the designated mutable members (`presets`, `user_metadata`, `metadata.signatures`,
/// `controls_schema[*].user_metadata`, `stems_registry[*].user_metadata`, and
/// `slimmable_tiers.{lite,full,pro,ultra}.prefix_crc32c`) from a parsed JSON header object
/// (`ref[container.dnam.jcs-header-schema]`, `ref[sec.jcs.immutable-header-digest]`).
pub fn project_immutable_header_value(header: &Value) -> Result<Value, DnamError> {
    let mut result = header.clone();
    let root = result
        .as_object_mut()
        .or_schema_err("header must be an object")?;
    root.remove("presets");
    root.remove("user_metadata");

    if let Some(meta) = root.get_mut("metadata") {
        let meta_obj = meta
            .as_object_mut()
            .or_schema_err("metadata must be an object")?;
        meta_obj.remove("signatures");
    }
    if let Some(ctrls) = root.get_mut("controls_schema") {
        let arr = ctrls
            .as_array_mut()
            .or_schema_err("controls_schema must be an array")?;
        for ctrl in arr {
            let c_obj = ctrl
                .as_object_mut()
                .or_schema_err("control must be an object")?;
            c_obj.remove("user_metadata");
        }
    }
    if let Some(stems) = root.get_mut("stems_registry")
        && !stems.is_null()
    {
        let arr = stems
            .as_array_mut()
            .or_schema_err("stems_registry must be an array")?;
        for stem in arr {
            let s_obj = stem
                .as_object_mut()
                .or_schema_err("stem must be an object")?;
            s_obj.remove("user_metadata");
        }
    }
    if let Some(tiers) = root.get_mut("slimmable_tiers") {
        let t_obj = tiers
            .as_object_mut()
            .or_schema_err("slimmable_tiers must be an object")?;
        for name in ["lite", "full", "pro", "ultra"] {
            if let Some(tier_entry) = t_obj.get_mut(name) {
                let te_obj = tier_entry
                    .as_object_mut()
                    .or_schema_err("tier entry must be an object")?;
                te_obj.remove("prefix_crc32c");
            }
        }
    }
    Ok(result)
}

/// Serializes a JSON `Value` using RFC 8785 JCS UTF-16 code-unit key ordering (`ref[container.dnam.jcs-header-schema]`,
/// `ref[sec.jcs.immutable-header-digest]`). When `reject_floats` is `true`, enforces the integer/string
/// JCS subset (integers within `±(2^53 - 1)`).
pub fn serialize_jcs_value(value: &Value, reject_floats: bool) -> Result<String, DnamError> {
    const MAX_SAFE_INT: i64 = (1i64 << 53) - 1;
    validate_json_value_unicode_and_finite(value)?;
    match value {
        Value::Null => Ok("null".to_string()),
        Value::Bool(true) => Ok("true".to_string()),
        Value::Bool(false) => Ok("false".to_string()),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                reject_if(i.abs() > MAX_SAFE_INT, "integer outside reference subset")?;
                Ok(i.to_string())
            } else if let Some(u) = n.as_u64() {
                reject_if(
                    u > (MAX_SAFE_INT as u64),
                    "integer outside reference subset",
                )?;
                Ok(u.to_string())
            } else if reject_floats {
                Err(DnamError::SchemaValidationError(
                    "float outside reference subset".to_string(),
                ))
            } else {
                let f = n.as_f64().or_schema_err("non-finite number")?;
                if f.fract() == 0.0 && f.abs() <= (MAX_SAFE_INT as f64) {
                    Ok((f as i64).to_string())
                } else {
                    serde_json::to_string(&f).map_err(|e| {
                        DnamError::SchemaValidationError(alloc::format!("JCS float error: {e}"))
                    })
                }
            }
        }
        Value::String(s) => serde_json::to_string(s)
            .map_err(|e| DnamError::SchemaValidationError(alloc::format!("JCS string error: {e}"))),
        Value::Array(arr) => {
            let mut parts = Vec::with_capacity(arr.len());
            for item in arr {
                parts.push(serialize_jcs_value(item, reject_floats)?);
            }
            Ok(alloc::format!("[{}]", parts.join(",")))
        }
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|(a, _), (b, _)| a.encode_utf16().cmp(b.encode_utf16()));
            let mut parts = Vec::with_capacity(entries.len());
            for (k, v) in entries {
                let ks = serde_json::to_string(k).unwrap();
                let vs = serialize_jcs_value(v, reject_floats)?;
                parts.push(alloc::format!("{ks}:{vs}"));
            }
            Ok(alloc::format!("{{{}}}", parts.join(",")))
        }
    }
}

/// Validates that `wire_bytes` are already canonical projected JCS bytes without mutable members
/// (`ref[container.dnam.jcs-header-schema]`).
pub fn validate_canonical_jcs_header_bytes(
    wire_bytes: &[u8],
    reject_floats: bool,
) -> Result<Vec<u8>, DnamError> {
    let parsed = parse_strict_json_object_bytes(wire_bytes)?;
    let projected = project_immutable_header_value(&parsed)?;
    let canonical = serialize_jcs_value(&projected, reject_floats)?.into_bytes();
    reject_if(
        wire_bytes != canonical.as_slice(),
        "noncanonical or mutable header",
    )?;
    Ok(canonical)
}

// ============================================================================
// Model-Declared Delay Histories (`ref[container.dnam.delay-histories]`)
// ============================================================================

/// Validated `delay_histories` declaration (`ref[container.dnam.delay-histories]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeclaredDelayHistory {
    pub id: String,
    pub stage: String,
    pub kind: String,
    pub introduced_tier: u8,
    pub sample_format: String,
    /// Per-tier reference sample counts at `48 kHz` (`0` for tiers below `introduced_tier`).
    pub reference_samples_by_tier: [u64; 4],
}

/// Parses a canonical decimal string (`[1-9][0-9]*`) as a non-zero `u64` (`ref[container.dnam.delay-histories]`).
pub fn parse_canonical_reference_samples(text: &str) -> Result<u64, DnamError> {
    let bytes = text.as_bytes();
    reject_if(
        bytes.is_empty() || !bytes[0].is_ascii_digit() || bytes[0] == b'0',
        "noncanonical unsigned count",
    )?;
    reject_if(
        bytes.iter().any(|b| !b.is_ascii_digit()),
        "noncanonical unsigned count",
    )?;
    text.parse::<u64>()
        .map_err(|_| DnamError::SchemaValidationError("count exceeds u64".to_string()))
}

fn tier_name_to_index(name: &str) -> Option<u8> {
    match name {
        "lite" => Some(0),
        "full" => Some(1),
        "pro" => Some(2),
        "ultra" => Some(3),
        _ => None,
    }
}

/// Validates and parses the `delay_histories` array for a container with `max_tier` (`0..=3`,
/// `ref[container.dnam.delay-histories]`, `ref[dsp.op-t.ring.declaration]`).
pub fn validate_delay_histories(
    entries: &[Value],
    max_tier: u8,
) -> Result<Vec<DeclaredDelayHistory>, DnamError> {
    reject_if(max_tier > 3, "invalid max_tier")?;
    let tier_names = ["lite", "full", "pro", "ultra"];
    let mut seen_ids = BTreeSet::new();
    let mut out = Vec::with_capacity(entries.len());

    for item in entries {
        let obj = item
            .as_object()
            .or_schema_err("delay history must be an object")?;
        reject_if(obj.len() != 6, "invalid delay history fields")?;
        let id = obj.get("id").and_then(Value::as_str).unwrap_or("");
        reject_if(
            id.is_empty() || !seen_ids.insert(id.to_string()),
            "duplicate or empty delay history id",
        )?;
        let stage = obj.get("stage").and_then(Value::as_str).unwrap_or("");
        reject_if(
            !matches!(stage, "stage1_pre" | "stage4_tank"),
            "invalid delay history stage",
        )?;
        let kind = obj.get("kind").and_then(Value::as_str).unwrap_or("");
        reject_if(
            !matches!(kind, "fractional" | "allpass"),
            "invalid delay history kind",
        )?;
        let intro_str = obj
            .get("introduced_tier")
            .and_then(Value::as_str)
            .unwrap_or("");
        let introduced_tier =
            tier_name_to_index(intro_str).or_schema_err("invalid introduced_tier")?;
        reject_if(
            introduced_tier > max_tier,
            "introduced_tier exceeds max_tier",
        )?;
        let sample_format = obj
            .get("sample_format")
            .and_then(Value::as_str)
            .unwrap_or("");
        reject_if(sample_format != "fp16", "invalid sample_format")?;
        let ref_map = obj
            .get("reference_samples")
            .and_then(Value::as_object)
            .or_schema_err("invalid reference_samples")?;

        let expected_tier_count = (max_tier - introduced_tier + 1) as usize;
        reject_if(
            ref_map.len() != expected_tier_count,
            "mismatched reference_samples tier keys",
        )?;
        let mut ref_by_tier = [0u64; 4];
        for t in (introduced_tier as usize)..=(max_tier as usize) {
            let s = ref_map
                .get(tier_names[t])
                .and_then(Value::as_str)
                .or_schema_err("missing tier sample count")?;
            let count = parse_canonical_reference_samples(s)?;
            reject_if(
                kind == "fractional" && count < 4,
                "fractional delay reference_samples must be >= 4",
            )?;
            ref_by_tier[t] = count;
        }
        out.push(DeclaredDelayHistory {
            id: id.to_string(),
            stage: stage.to_string(),
            kind: kind.to_string(),
            introduced_tier,
            sample_format: sample_format.to_string(),
            reference_samples_by_tier: ref_by_tier,
        });
    }
    Ok(out)
}

// ============================================================================
// Stage III Separate 64B-Aligned Slices & CP1 Packed Control Projection
// (`ref[container.dnam.stage3-slices]`, `ref[container.dnam.cp1-profile]`)
// ============================================================================

/// Validates that a Stage III early FIR slice (`128 B` per tier) and Kautz modal slice
/// (`128 B * {1, 1, 2, 4}` blocks per tier) are 64-byte aligned, non-overlapping, within their
/// tier bounds, and contain only finite `f32` values (`ref[container.dnam.stage3-slices]`,
/// `ref[container.dnam.load-validation]`).
pub fn validate_stage3_tier_slices(
    tier_index: u8,
    tier_bytes: &[u8],
    early_fir_offset_in_tier: usize,
    early_fir_len: usize,
    kautz_offset_in_tier: usize,
    kautz_len: usize,
) -> Result<(), DnamError> {
    let expected_kautz_blocks = match tier_index {
        0 => 1usize,
        1 => 1,
        2 => 2,
        3 => 4,
        _ => {
            return Err(DnamError::SchemaValidationError(
                "invalid tier_index".to_string(),
            ));
        }
    };
    let expected_kautz_len = expected_kautz_blocks * 128;
    reject_if(
        early_fir_len != 128 || kautz_len != expected_kautz_len,
        "invalid Stage III slice byte length",
    )?;
    reject_if(
        !early_fir_offset_in_tier.is_multiple_of(64) || !kautz_offset_in_tier.is_multiple_of(64),
        "misaligned Stage III slice offset",
    )?;
    let fir_end = early_fir_offset_in_tier + early_fir_len;
    let kautz_end = kautz_offset_in_tier + kautz_len;
    reject_if(
        fir_end > tier_bytes.len() || kautz_end > tier_bytes.len(),
        "Stage III slice outside tier",
    )?;
    reject_if(
        early_fir_offset_in_tier < kautz_end && kautz_offset_in_tier < fir_end,
        "overlapping Stage III slices",
    )?;
    for slice in [
        &tier_bytes[early_fir_offset_in_tier..fir_end],
        &tier_bytes[kautz_offset_in_tier..kautz_end],
    ] {
        for &chunk in slice.as_chunks::<4>().0 {
            let v = f32::from_le_bytes(chunk);
            reject_if(!v.is_finite(), "non-finite Stage III coefficient")?;
        }
    }
    Ok(())
}

/// Returns the byte length of a single `CP1` contributor record with `lanes` (`1..=3`):
/// `4 * (9 * lanes + 4)` in `Arena 0` (`ultra == false`) or `4 * (8 * lanes + 4)` in `Arena 3`
/// (`ultra == true`, `ref[container.dnam.cp1-profile]`).
#[inline]
pub const fn cp1_contributor_record_bytes(lanes: usize, ultra_delta: bool) -> Option<usize> {
    if lanes < 1 || lanes > 3 {
        None
    } else if ultra_delta {
        Some(4 * (8 * lanes + 4))
    } else {
        Some(4 * (9 * lanes + 4))
    }
}

/// Decoded `CP1` contributor record (`ref[container.dnam.cp1-profile]`).
#[derive(Clone, Debug, PartialEq)]
pub struct Cp1DecodedRecord {
    pub lanes: usize,
    pub base_main: Vec<f32>,
    pub embedding: Vec<f32>,
    pub base_factors: [f32; 4],
    pub extra_main: Vec<f32>,
    pub extra_factors: Option<[f32; 4]>,
}

/// Decodes a stream of packed `CP1` contributor records from `base_bytes` (`Arena 0`) and optional
/// `ultra_delta_bytes` (`Arena 3`) for the given sequence of contributor lane counts (`ref[container.dnam.cp1-profile]`).
pub fn decode_cp1_contributor_stream(
    base_bytes: &[u8],
    ultra_delta_bytes: Option<&[u8]>,
    contributor_lanes: &[usize],
) -> Result<Vec<Cp1DecodedRecord>, DnamError> {
    let mut expected_base = 0usize;
    let mut expected_ultra = 0usize;
    for &lanes in contributor_lanes {
        expected_base += cp1_contributor_record_bytes(lanes, false)
            .or_schema_err("invalid contributor lane count")?;
        expected_ultra += cp1_contributor_record_bytes(lanes, true).unwrap();
    }
    reject_if(
        base_bytes.len() != expected_base,
        "incorrect base tensor length",
    )?;
    if let Some(ub) = ultra_delta_bytes
        && ub.len() != expected_ultra
    {
        return Err(DnamError::SchemaValidationError(
            "incorrect delta tensor length".to_string(),
        ));
    }

    let read_f32_vec = |bytes: &[u8]| -> Result<Vec<f32>, DnamError> {
        let mut out = Vec::with_capacity(bytes.len() / 4);
        for &chunk in bytes.as_chunks::<4>().0 {
            let v = f32::from_le_bytes(chunk);
            reject_if(!v.is_finite(), "nonfinite coefficient")?;
            out.push(v);
        }
        Ok(out)
    };

    let mut records = Vec::with_capacity(contributor_lanes.len());
    let mut b_off = 0usize;
    let mut u_off = 0usize;
    for &lanes in contributor_lanes {
        let b_len = cp1_contributor_record_bytes(lanes, false).unwrap();
        let b_vals = read_f32_vec(&base_bytes[b_off..b_off + b_len])?;
        b_off += b_len;
        let main_end = 8 * lanes;
        let base_main = b_vals[..main_end].to_vec();
        let embedding = b_vals[main_end..main_end + lanes].to_vec();
        let base_factors = [
            b_vals[main_end + lanes],
            b_vals[main_end + lanes + 1],
            b_vals[main_end + lanes + 2],
            b_vals[main_end + lanes + 3],
        ];
        let (extra_main, extra_factors) = if let Some(ub) = ultra_delta_bytes {
            let u_len = cp1_contributor_record_bytes(lanes, true).unwrap();
            let u_vals = read_f32_vec(&ub[u_off..u_off + u_len])?;
            u_off += u_len;
            (
                u_vals[..main_end].to_vec(),
                Some([
                    u_vals[main_end],
                    u_vals[main_end + 1],
                    u_vals[main_end + 2],
                    u_vals[main_end + 3],
                ]),
            )
        } else {
            (Vec::new(), None)
        };
        records.push(Cp1DecodedRecord {
            lanes,
            base_main,
            embedding,
            base_factors,
            extra_main,
            extra_factors,
        });
    }
    Ok(records)
}

// ============================================================================
// Structured Conditional Control Rules (`ref[container.dnam.conditional-rules]`)
// ============================================================================

/// Schema definition of a discrete switch control for selector validation (`ref[container.dnam.conditional-rules]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SwitchControlSpec {
    pub control_id: String,
    pub options: Vec<String>,
    pub active_options: Vec<String>,
}

/// Validated conditional bound rule (`ref[container.dnam.conditional-rules]`).
#[derive(Clone, Debug, PartialEq)]
pub struct StructuredConditionalBound {
    pub when: Vec<(String, String)>,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

/// Validated conditional calibration curve rule (`ref[container.dnam.conditional-rules]`).
#[derive(Clone, Debug, PartialEq)]
pub struct StructuredConditionalCurve {
    pub when: Vec<(String, String)>,
    pub anchors: [f64; 3],
}

fn parse_structured_selector(
    when_val: &Value,
    switches: &[SwitchControlSpec],
) -> Result<Vec<(String, String)>, DnamError> {
    let arr = when_val.as_array().or_schema_err("invalid conjunction")?;
    reject_if(arr.is_empty(), "invalid conjunction")?;
    let mut out = Vec::with_capacity(arr.len());
    let mut prev_idx: Option<usize> = None;
    for item in arr {
        let obj = item.as_object().or_schema_err("invalid selector object")?;
        reject_if(
            obj.len() != 2 || !obj.contains_key("control_id") || !obj.contains_key("option_id"),
            "invalid selector object",
        )?;
        let cid = obj.get("control_id").and_then(Value::as_str).unwrap_or("");
        let oid = obj.get("option_id").and_then(Value::as_str).unwrap_or("");
        reject_if(
            cid.is_empty() || oid.is_empty(),
            "invalid control or option identity",
        )?;
        let sw_idx = switches
            .iter()
            .position(|s| s.control_id == cid)
            .or_schema_err("unknown switch")?;
        if let Some(prev) = prev_idx
            && sw_idx <= prev
        {
            return Err(DnamError::SchemaValidationError(
                "duplicate or unordered switch".to_string(),
            ));
        }
        reject_if(
            !switches[sw_idx].options.iter().any(|o| o == oid),
            "unknown option",
        )?;
        out.push((cid.to_string(), oid.to_string()));
        prev_idx = Some(sw_idx);
    }
    Ok(out)
}

/// Parses and validates a `conditional_bounds` array (`ref[container.dnam.conditional-rules]`).
pub fn parse_structured_conditional_bounds(
    value: &Value,
    switches: &[SwitchControlSpec],
) -> Result<Vec<StructuredConditionalBound>, DnamError> {
    let arr = value.as_array().or_schema_err("invalid bounds array")?;
    let mut seen = BTreeSet::new();
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let obj = item.as_object().or_schema_err("invalid bound object")?;
        for k in obj.keys() {
            reject_if(
                !matches!(k.as_str(), "when" | "min" | "max"),
                "invalid bound object",
            )?;
        }
        let when_val = obj.get("when").or_schema_err("invalid bound object")?;
        let has_min = obj.contains_key("min");
        let has_max = obj.contains_key("max");
        reject_if(!has_min && !has_max, "invalid bound object")?;
        let when = parse_structured_selector(when_val, switches)?;
        reject_if(!seen.insert(when.clone()), "duplicate conjunction")?;
        let min = if let Some(m) = obj.get("min") {
            let v = m.as_f64().or_schema_err("invalid number")?;
            reject_if(!v.is_finite(), "invalid number")?;
            Some(v)
        } else {
            None
        };
        let max = if let Some(m) = obj.get("max") {
            let v = m.as_f64().or_schema_err("invalid number")?;
            reject_if(!v.is_finite(), "invalid number")?;
            Some(v)
        } else {
            None
        };
        if let (Some(lo), Some(hi)) = (min, max)
            && lo > hi
        {
            return Err(DnamError::SchemaValidationError(
                "reversed bound".to_string(),
            ));
        }
        out.push(StructuredConditionalBound { when, min, max });
    }
    Ok(out)
}

/// Parses and validates a `conditional_physical_anchors` array (`ref[container.dnam.conditional-rules]`).
pub fn parse_structured_conditional_curves(
    value: &Value,
    switches: &[SwitchControlSpec],
) -> Result<Vec<StructuredConditionalCurve>, DnamError> {
    let arr = value
        .as_array()
        .or_schema_err("invalid calibration array")?;
    let mut seen = BTreeSet::new();
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let obj = item
            .as_object()
            .or_schema_err("invalid calibration object")?;
        reject_if(
            obj.len() != 2 || !obj.contains_key("when") || !obj.contains_key("anchors"),
            "invalid calibration object",
        )?;
        let when = parse_structured_selector(obj.get("when").unwrap(), switches)?;
        reject_if(!seen.insert(when.clone()), "duplicate conjunction")?;
        let anchors_arr = obj
            .get("anchors")
            .and_then(Value::as_array)
            .or_schema_err("invalid anchors")?;
        reject_if(anchors_arr.len() != 3, "invalid anchors")?;
        let mut anchors = [0.0f64; 3];
        for (i, a) in anchors_arr.iter().enumerate() {
            let v = a.as_f64().or_schema_err("invalid number")?;
            reject_if(!v.is_finite(), "invalid number")?;
            anchors[i] = v;
        }
        out.push(StructuredConditionalCurve { when, anchors });
    }
    Ok(out)
}

fn evaluate_selectors(
    selectors: &[&[(String, String)]],
    switches: &[SwitchControlSpec],
    effective: &BTreeMap<String, String>,
) -> Result<Vec<bool>, DnamError> {
    for when in selectors {
        for (cid, _) in *when {
            let eff_opt = effective
                .get(cid)
                .or_schema_err("missing effective switch state")?;
            let sw = switches
                .iter()
                .find(|s| &s.control_id == cid)
                .or_schema_err("unknown switch")?;
            reject_if(
                !sw.active_options.iter().any(|o| o == eff_opt),
                "inactive effective option",
            )?;
        }
    }
    Ok(selectors
        .iter()
        .map(|when| {
            when.iter()
                .all(|(cid, oid)| effective.get(cid).is_some_and(|v| v == oid))
        })
        .collect())
}

/// Evaluates the intersected safe interval `(lower, upper)` across all matching `conditional_bounds`
/// (`ref[container.dnam.conditional-rules]`).
pub fn resolve_conditional_safe_interval(
    base: (f64, f64),
    rules: &[StructuredConditionalBound],
    switches: &[SwitchControlSpec],
    effective: &BTreeMap<String, String>,
) -> Result<(f64, f64), DnamError> {
    let selectors: Vec<&[(String, String)]> = rules.iter().map(|r| r.when.as_slice()).collect();
    let matched = evaluate_selectors(&selectors, switches, effective)?;
    let (mut lower, mut upper) = base;
    for (rule, &sel) in rules.iter().zip(matched.iter()) {
        if sel {
            if let Some(lo) = rule.min {
                lower = lower.max(lo);
            }
            if let Some(hi) = rule.max {
                upper = upper.min(hi);
            }
        }
    }
    reject_if(lower > upper, "empty safe intersection")?;
    Ok((lower, upper))
}

/// Selects the active calibration anchors with greatest selector specificity (`ref[container.dnam.conditional-rules]`),
/// returning an error on a specificity tie.
pub fn select_conditional_calibration_curve(
    rules: &[StructuredConditionalCurve],
    switches: &[SwitchControlSpec],
    effective: &BTreeMap<String, String>,
    default_anchors: Option<[f64; 3]>,
) -> Result<[f64; 3], DnamError> {
    let selectors: Vec<&[(String, String)]> = rules.iter().map(|r| r.when.as_slice()).collect();
    let matched = evaluate_selectors(&selectors, switches, effective)?;
    let applicable: Vec<&StructuredConditionalCurve> = rules
        .iter()
        .zip(matched.iter())
        .filter_map(|(r, &m)| m.then_some(r))
        .collect();
    if applicable.is_empty() {
        return default_anchors.ok_or_else(|| {
            DnamError::SchemaValidationError("calibration unavailable".to_string())
        });
    }
    let max_spec = applicable.iter().map(|r| r.when.len()).max().unwrap();
    let best: Vec<&&StructuredConditionalCurve> = applicable
        .iter()
        .filter(|r| r.when.len() == max_spec)
        .collect();
    reject_if(best.len() != 1, "ambiguous calibration")?;
    Ok(best[0].anchors)
}

/// Evaluates the midpoint-anchored UI power-skew exponent $\gamma_{\mathrm{skew}}$, forward map $x(t)$,
/// and inverse map $t(x)$ (`ref[container.dnam.metadata-ui-hints]`).
pub fn eval_ui_midpoint_power_skew(
    x_min: f64,
    x_mid: f64,
    x_max: f64,
    t_norm_0_to_1: f64,
) -> Option<(f64, f64, f64)> {
    if !(x_min < x_mid && x_mid < x_max && (0.0..=1.0).contains(&t_norm_0_to_1)) {
        return None;
    }
    let ratio = (x_mid - x_min) / (x_max - x_min);
    let gamma = cold_ln_f64(0.5) / cold_ln_f64(ratio);
    let x_val = if t_norm_0_to_1 <= 0.0 {
        x_min
    } else if t_norm_0_to_1 >= 1.0 {
        x_max
    } else {
        x_min + (x_max - x_min) * cold_pow_f64(t_norm_0_to_1, 1.0 / gamma)
    };
    let t_roundtrip = if x_val <= x_min {
        0.0
    } else if x_val >= x_max {
        1.0
    } else {
        cold_pow_f64((x_val - x_min) / (x_max - x_min), gamma)
    };
    Some((gamma, x_val, t_roundtrip))
}

// ============================================================================
// .dnam-ckpt v2.0 Bundle Framing, DCA1 Signature & Runtime Export
// (`ref[container.ckpt.bundle-framing]`, `ref[container.ckpt.dca1-signature]`,
//  `ref[container.ckpt.export-template]`)
// ============================================================================

/// Computes the per-tier SHA-256 checkpoint prefix commitments under `"DNCK-CHECKPOINT-PREFIX-V1"`
/// (`ref[container.ckpt.dca1-signature]`).
pub fn compute_dnck_bundle_prefixes(
    preamble: &[u8],
    directory: &[u8],
    header: &[u8],
    shared: &[u8],
    bundles: &[&[u8]],
) -> Vec<[u8; 32]> {
    let header_sha = sha256_digest(header);
    let mut base = Vec::new();
    base.extend_from_slice(&encode_domain_field("domain", b"DNCK-CHECKPOINT-PREFIX-V1"));
    base.extend_from_slice(&encode_domain_field(
        "checkpoint_header_sha256",
        &header_sha,
    ));
    base.extend_from_slice(&encode_domain_field("preamble", preamble));
    base.extend_from_slice(&encode_domain_field("directory", directory));
    base.extend_from_slice(&encode_domain_field("shared_checkpoint_binary", shared));

    let labels = ["bundle_lite", "bundle_full", "bundle_pro", "bundle_ultra"];
    let mut out = Vec::with_capacity(bundles.len());
    for idx in 0..bundles.len() {
        let mut msg = base.clone();
        for t in 0..=idx {
            msg.extend_from_slice(&encode_domain_field(labels[t], bundles[t]));
        }
        out.push(sha256_digest(&msg));
    }
    out
}

/// Builds the `"DNCK-CHECKPOINT-MANIFEST-V1"` signing message for `DCA1` (`ref[container.ckpt.dca1-signature]`).
pub fn build_dnck_writer_manifest_message(
    writer_pubkey: &[u8; 32],
    checkpoint_header_sha256: &[u8; 32],
    issued_max_tier: u8,
    prefixes: &[[u8; 32]],
) -> Vec<u8> {
    let mut prefix_list = Vec::with_capacity(1 + 33 * prefixes.len());
    prefix_list.push(prefixes.len() as u8);
    for (t, d) in prefixes.iter().enumerate() {
        prefix_list.push(t as u8);
        prefix_list.extend_from_slice(d);
    }
    let mut msg = Vec::new();
    msg.extend_from_slice(&encode_domain_field(
        "domain",
        b"DNCK-CHECKPOINT-MANIFEST-V1",
    ));
    msg.extend_from_slice(&encode_domain_field(
        "checkpoint_writer_pubkey_ed25519",
        writer_pubkey,
    ));
    msg.extend_from_slice(&encode_domain_field(
        "checkpoint_header_sha256",
        checkpoint_header_sha256,
    ));
    msg.extend_from_slice(&encode_domain_field("issued_max_tier", &[issued_max_tier]));
    msg.extend_from_slice(&encode_domain_field(
        "checkpoint_prefix_sha256",
        &prefix_list,
    ));
    msg
}

/// Validates a v2.0 `.dnam-ckpt` (`b"DNCK"`) container, its `DCA1` envelope, and its retained
/// tier bundle prefix commitment (`ref[container.ckpt.bundle-framing]`, `ref[container.ckpt.dca1-signature]`).
pub fn validate_v2_dnck_checkpoint(data: &[u8]) -> Result<(u8, Vec<usize>, usize), DnamError> {
    reject_if(data.len() < 208, "short checkpoint")?;
    let magic = &data[0..4];
    let major = u16::from_le_bytes([data[4], data[5]]);
    let minor = u16::from_le_bytes([data[6], data[7]]);
    let profile = data[8];
    let auth_kind = data[9];
    let maximum = data[10];
    let reserved = data[11];
    let rate = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
    reject_if(
        magic != b"DNCK"
            || (major, minor, profile, auth_kind, reserved, rate) != (2, 0, 0, 0, 0, 48_000)
            || maximum > 3
            || data[16..64].iter().any(|&b| b != 0),
        "invalid preamble",
    )?;

    let read_u64 = |idx: usize| -> u64 {
        let off = V2_PREAMBLE_SIZE + idx * 8;
        u64::from_le_bytes(data[off..off + 8].try_into().unwrap())
    };
    let header_offset = read_u64(0);
    let header_length = read_u64(1);
    let shared_offset = read_u64(2);
    let shared_length = read_u64(3);
    let auth_offset = read_u64(4);
    let auth_length = read_u64(5);
    let res0 = read_u64(6);
    let res1 = read_u64(7);
    let bundle_start = read_u64(8);
    let ends = [read_u64(9), read_u64(10), read_u64(11), read_u64(12)];
    let count = (maximum as usize) + 1;
    let expected_auth_len = (106 + 33 * count) as u64;

    reject_if(
        res0 != 0
            || res1 != 0
            || data[V2_PREAMBLE_SIZE + 13 * 8..V2_HEADER_OFFSET]
                .iter()
                .any(|&b| b != 0)
            || header_offset != 192
            || header_length == 0
            || Some(shared_offset) != align64_u64(header_offset + header_length)
            || Some(auth_offset) != align64_u64(shared_offset + shared_length)
            || auth_length != expected_auth_len
            || Some(bundle_start) != align64_u64(auth_offset + auth_length)
            || ends[count..].iter().any(|&e| e != 0),
        "invalid directory",
    )?;

    let mut prev = bundle_start;
    let mut issued_ends = Vec::with_capacity(count);
    for &end in &ends[..count] {
        reject_if(end <= prev || !end.is_multiple_of(64), "invalid directory")?;
        issued_ends.push(end as usize);
        prev = end;
    }

    for (s, e) in [
        (
            (header_offset + header_length) as usize,
            shared_offset as usize,
        ),
        (
            (shared_offset + shared_length) as usize,
            auth_offset as usize,
        ),
        ((auth_offset + auth_length) as usize, bundle_start as usize),
    ] {
        reject_if(
            data[s..e].iter().any(|&b| b != 0),
            "nonzero alignment padding",
        )?;
    }

    let footer = &data[data.len() - DMUE_FOOTER_SIZE..];
    let frame_length = u64::from_le_bytes(footer[4..12].try_into().unwrap()) as usize;
    let footer_res = u32::from_le_bytes(footer[12..16].try_into().unwrap());
    reject_if(
        &footer[0..4] != b"DMUE" || footer_res != 0 || frame_length > data.len() - DMUE_FOOTER_SIZE,
        "invalid footer",
    )?;
    let boundary = data.len() - DMUE_FOOTER_SIZE - frame_length;
    let tier = issued_ends
        .iter()
        .position(|&e| e == boundary)
        .or_schema_err("invalid retained boundary")?;

    parse_and_validate_dmu1_frame(&data[boundary..data.len() - DMUE_FOOTER_SIZE])?;

    let a_off = auth_offset as usize;
    let a_len = auth_length as usize;
    let envelope = &data[a_off..a_off + a_len];
    reject_if(
        &envelope[0..8] != b"DCA1\0\0\0\0"
            || envelope[40] != maximum
            || envelope[41] != (count as u8)
            || (0..count).any(|idx| envelope[42 + 33 * idx] != (idx as u8)),
        "invalid envelope",
    )?;

    let mut bundles = Vec::with_capacity(tier + 1);
    let mut b_start = bundle_start as usize;
    for &b_end in &issued_ends[..=tier] {
        bundles.push(&data[b_start..b_end]);
        b_start = b_end;
    }
    let h_off = header_offset as usize;
    let h_len = header_length as usize;
    let s_off = shared_offset as usize;
    let s_len = shared_length as usize;
    let computed = compute_dnck_bundle_prefixes(
        &data[..64],
        &data[64..192],
        &data[h_off..h_off + h_len],
        &data[s_off..s_off + s_len],
        &bundles,
    );
    let expected_prefix = &envelope[43 + 33 * tier..75 + 33 * tier];
    reject_if(
        computed[tier].as_slice() != expected_prefix,
        "checkpoint prefix mismatch",
    )?;

    Ok((tier as u8, issued_ends, boundary))
}

/// Exports the retained tier of a validated v2.0 `.dnam-ckpt` container into a standalone v2.0
/// `.dnam` runtime container using the zero-copy template in `shared_checkpoint_binary`
/// (`ref[container.ckpt.export-template]`, `ref[container.ckpt.declarative-no-pickle]`).
pub fn export_runtime_dnam_from_ckpt(data: &[u8]) -> Result<Vec<u8>, DnamError> {
    let (tier, ends, boundary) = validate_v2_dnck_checkpoint(data)?;
    let shared_offset = u64::from_le_bytes(data[64 + 16..64 + 24].try_into().unwrap()) as usize;
    let shared_length = u64::from_le_bytes(data[64 + 24..64 + 32].try_into().unwrap()) as usize;
    let bundle_start = u64::from_le_bytes(data[64 + 64..64 + 72].try_into().unwrap()) as usize;
    reject_if(shared_length < V2_HEADER_OFFSET, "short runtime template")?;
    let template = &data[shared_offset..shared_offset + shared_length];
    let runtime_start = u64::from_le_bytes(template[64 + 64..64 + 72].try_into().unwrap()) as usize;
    let runtime_ends: [usize; 4] = core::array::from_fn(|i| {
        let off = 64 + 72 + i * 8;
        u64::from_le_bytes(template[off..off + 8].try_into().unwrap()) as usize
    });
    reject_if(runtime_start > template.len(), "invalid runtime template")?;

    let mut output = Vec::new();
    output.extend_from_slice(&template[..runtime_start]);
    let mut prev_runtime = runtime_start;
    let mut ckpt_start = bundle_start;
    for idx in 0..=(tier as usize) {
        let runtime_len = runtime_ends[idx].saturating_sub(prev_runtime);
        let ckpt_len = ends[idx].saturating_sub(ckpt_start);
        reject_if(
            runtime_len == 0 || !runtime_len.is_multiple_of(64) || runtime_len > ckpt_len,
            "invalid runtime slice",
        )?;
        output.extend_from_slice(&data[ckpt_start..ckpt_start + runtime_len]);
        prev_runtime = runtime_ends[idx];
        ckpt_start = ends[idx];
    }
    output.extend_from_slice(&data[boundary..]);
    let exported_layout = validate_v2_open_dnam_layout(&output)?;
    reject_if(
        exported_layout.retained_tier != tier,
        "runtime tier mismatch",
    )?;
    Ok(output)
}

#[cfg(test)]
mod spec_tests {
    #[allow(unused_imports)]
    use alloc::{
        boxed::Box,
        format,
        string::{String, ToString},
        vec,
        vec::Vec,
    };

    // Specification Verification Suite: Phase 2 — `.dnam` & `.dnam-ckpt` Containers & Schema.
    //
    // Verifies all engine-scope rules in `docs/specs/containers/dnam_binary_format.md` and
    // `docs/specs/containers/checkpoint_and_cache_format.md`:
    // - `verify[container.dnam.formats]`
    // - `verify[container.dnam.wire-framing]`
    // - `verify[container.dnam.preamble-directory]`
    // - `verify[container.dnam.open-layout]`
    // - `verify[container.dnam.mutable-frame]`
    // - `verify[container.dnam.mutable-json]`
    // - `verify[container.dnam.tier-prefix]`
    // - `verify[container.dnam.delay-histories]`
    // - `verify[container.dnam.load-validation]`
    // - `verify[container.dnam.stage3-slices]`
    // - `verify[container.dnam.cp1-profile]`
    // - `verify[container.dnam.conditional-rules]`
    // - `verify[container.dnam.metadata-ui-hints]`
    // - `verify[container.dnam.sample-rate-anchor]`
    // - `verify[container.dnam.jcs-header-schema]`
    // - `verify[container.dnam.security-crossref]`
    // - `verify[container.ckpt.bundle-framing]`
    // - `verify[container.ckpt.dca1-signature]`
    // - `verify[container.ckpt.export-template]`
    // - `verify[container.ckpt.declarative-no-pickle]`
    // - `verify[container.ckpt.uplo-u]`

    use std::collections::BTreeMap;

    use crate::container::{
        ARENA4_STEM_BYTES_FULL, ARENA4_STEM_BYTES_LITE, ARENA4_STEM_BYTES_PRO,
        ARENA4_STEM_BYTES_ULTRA, DNAM_MAGIC, DNCK_MAGIC, DNSL_MAGIC, SwitchControlSpec,
        align64_u64, append_dmue_mutable_suffix, build_dmu1_mutable_frame,
        build_dnck_writer_manifest_message, build_v2_open_dnam_container,
        compute_dnck_bundle_prefixes, compute_v2_open_retained_prefix_sha256,
        cp1_contributor_record_bytes, decode_cp1_contributor_stream, encode_domain_field,
        eval_ui_midpoint_power_skew, export_runtime_dnam_from_ckpt, pack_uplo_u_f64,
        parse_and_validate_dmu1_frame, parse_canonical_reference_samples,
        parse_strict_json_object_bytes, parse_structured_conditional_bounds,
        parse_structured_conditional_curves, project_immutable_header_value,
        resolve_conditional_safe_interval, select_conditional_calibration_curve,
        serialize_jcs_value, strip_v2_open_dnam_container, unpack_uplo_u_f64, uplo_u_packed_len,
        validate_canonical_jcs_header_bytes, validate_delay_histories,
        validate_mutable_json_schema, validate_stage3_tier_slices, validate_v2_dnck_checkpoint,
        validate_v2_open_dnam_layout,
    };
    use crate::math::crc32c;
    use serde_json::json;

    /// verify[container.dnam.formats]
    /// verify[container.dnam.wire-framing]
    /// verify[container.dnam.preamble-directory]
    /// verify[container.dnam.open-layout]
    /// verify[container.dnam.tier-prefix]
    /// verify[container.dnam.sample-rate-anchor]
    /// Verifies `.dnam` v2.0 64B preamble, 128B directory, 64B-aligned regions, tier stripping across
    /// `Lite..Ultra`, and prefix commitment invariance under mutable suffix edits.
    #[test]
    fn test_v2_open_dnam_framing_preamble_directory_and_tier_stripping() {
        assert_eq!(&DNAM_MAGIC, b"DNAM");
        assert_eq!(&DNCK_MAGIC, b"DNCK");
        assert_eq!(&DNSL_MAGIC, b"DNSL");

        let header = br#"{"version":1}"#;
        let shared = [b'S'; 64];
        let auth = [b'A'; 32];
        let tiers: [&[u8]; 4] = [&[0x41; 64], &[0x42; 64], &[0x43; 64], &[0x44; 64]];
        let dmu1 = build_dmu1_mutable_frame(br#"{"user_metadata":{"tag":"v1"}}"#, 64).unwrap();

        let container =
            build_v2_open_dnam_container(header, &shared, &auth, &tiers, Some(&dmu1), 0).unwrap();
        let layout = validate_v2_open_dnam_layout(&container).unwrap();
        assert_eq!(layout.magic, *b"DNAM");
        assert_eq!(layout.protection_profile, 0);
        assert_eq!(layout.auth_kind, 0);
        assert_eq!(layout.maximum_tier, 3);
        assert_eq!(layout.retained_tier, 3);
        assert_eq!(layout.sample_rate_train_hz, 48_000);
        assert_eq!(layout.header_extent, (192, header.len()));
        assert_eq!(layout.shared_extent.0 % 64, 0);
        assert_eq!(layout.auth_extent.0 % 64, 0);
        assert_eq!(layout.tier_start % 64, 0);

        // Stripping to each tier 0..=3 preserves the original preamble, directory, and tier prefix digest
        for target_tier in 0..=3u8 {
            let stripped = strip_v2_open_dnam_container(&container, target_tier).unwrap();
            let s_layout = validate_v2_open_dnam_layout(&stripped).unwrap();
            assert_eq!(s_layout.maximum_tier, 3);
            assert_eq!(s_layout.retained_tier, target_tier);
            assert_eq!(s_layout.tier_ends, layout.tier_ends);

            // Replacing mutable suffix does not alter the retained prefix digest
            let mut with_new_mutable =
                stripped[..s_layout.tier_ends[target_tier as usize]].to_vec();
            let dmu2 = build_dmu1_mutable_frame(br#"{"user_metadata":{"tag":"v2"}}"#, 64).unwrap();
            append_dmue_mutable_suffix(&mut with_new_mutable, &dmu2);
            let d1 = compute_v2_open_retained_prefix_sha256(&stripped, b"DNAM-MASTER-PREFIX-V1")
                .unwrap();
            let d2 =
                compute_v2_open_retained_prefix_sha256(&with_new_mutable, b"DNAM-MASTER-PREFIX-V1")
                    .unwrap();
            assert_eq!(d1, d2);
        }

        // Tampering with preamble padding, sample rate, or alignment padding is rejected
        let mut bad_rate = container.clone();
        bad_rate[12..16].copy_from_slice(&44_100u32.to_le_bytes());
        assert!(validate_v2_open_dnam_layout(&bad_rate).is_err());

        let mut bad_pad = container.clone();
        bad_pad[192 + header.len()] = 1;
        assert!(validate_v2_open_dnam_layout(&bad_pad).is_err());
    }

    /// verify[container.dnam.mutable-frame]
    /// verify[container.dnam.mutable-json]
    /// Verifies `DMU1` wire framing, Castagnoli CRC-32C vectors (`0xE3069283`, `0xA1CCB48F`),
    /// `DMUE` footer validation, and mutable JSON member schema validation.
    #[test]
    fn test_dmu1_mutable_frame_and_mutable_json_schema() {
        assert_eq!(crc32c(b"123456789"), 0xe306_9283);
        let empty_frame = build_dmu1_mutable_frame(b"{}", 18).unwrap();
        assert_eq!(empty_frame, b"DMU1\x8f\xb4\xcc\xa1\x02\0\0\0\0\0\0\0{}");
        let doc = parse_and_validate_dmu1_frame(&empty_frame).unwrap();
        validate_mutable_json_schema(&doc).unwrap();

        let rich_payload = br#"{"version":1,"presets":[{"id":"warm","name":"Warm","controls":{"gain":0.4,"mode":"lead"},"user_metadata":{"note":"ok"}}],"user_metadata":{"tags":["amp"]},"control_user_metadata":{"gain":{"color":"red"}},"stem_user_metadata":{"stem-1":{"tag":"keep"}}}"#;
        let rich_frame =
            build_dmu1_mutable_frame(rich_payload, 16 + rich_payload.len() + 16).unwrap();
        let rich_doc = parse_and_validate_dmu1_frame(&rich_frame).unwrap();
        validate_mutable_json_schema(&rich_doc).unwrap();

        // Non-zero padding or corrupted CRC is rejected
        let mut bad_pad = rich_frame.clone();
        *bad_pad.last_mut().unwrap() = 1;
        assert!(parse_and_validate_dmu1_frame(&bad_pad).is_err());

        let mut bad_crc = empty_frame.clone();
        bad_crc[4] ^= 0xff;
        assert!(parse_and_validate_dmu1_frame(&bad_crc).is_err());

        // Invalid mutable member shapes are rejected
        for bad_json in [
        br#"{"version":2}"#.as_slice(),
        br#"{"unknown":{}}"#.as_slice(),
        br#"{"presets":{}}"#.as_slice(),
        br#"{"control_user_metadata":{"gain":2}}"#.as_slice(),
        br#"{"presets":[{"id":"p","name":"P","controls":{},"unknown":0}]}"#.as_slice(),
        br#"{"presets":[{"id":"p","name":"P","controls":{}},{"id":"p","name":"Q","controls":{}}]}"#
            .as_slice(),
        br#"{"presets":[{"id":"p","name":"P","controls":{"gain":true}}]}"#.as_slice(),
        br#"{"presets":[{"id":"p","name":"P"}]}"#.as_slice(),
    ] {
        let f = build_dmu1_mutable_frame(bad_json, 16 + bad_json.len()).unwrap();
        let v = parse_and_validate_dmu1_frame(&f).unwrap();
        assert!(validate_mutable_json_schema(&v).is_err());
    }
    }

    /// verify[container.dnam.jcs-header-schema]
    /// verify[container.dnam.security-crossref]
    /// Verifies RFC 8785 JCS UTF-16 code-unit key ordering, strict duplicate-key & noncharacter rejection,
    /// and exact mutable-member projection.
    #[test]
    fn test_jcs_immutable_header_projection_and_utf16_ordering() {
        let fixture = br#"{
      "version":"future-format",
      "metadata":{"name":"amp","signatures":{"sig":"old"}},
      "user_metadata":{"stars":1},
      "presets":[{"name":"old"}],
      "controls_schema":[{"id":"gain","user_metadata":{"tag":"a"},"calibration":{"user_metadata":"signed"}}],
      "stems_registry":[{"stem_id":"a","user_metadata":{"tag":"b"}}],
      "slimmable_tiers":{"lite":{"channels":4,"prefix_crc32c":12},"future":{"prefix_crc32c":13}}
    }"#;
        let orig_val = parse_strict_json_object_bytes(fixture).unwrap();
        let proj_orig = project_immutable_header_value(&orig_val).unwrap();
        let jcs_orig = serialize_jcs_value(&proj_orig, true).unwrap();

        // Mutating only designated mutable paths produces the exact same projected JCS string
        let modified = br#"{
      "version":"future-format",
      "metadata":{"name":"amp","signatures":{"sig":"new"}},
      "user_metadata":{"stars":5},
      "presets":[{"name":"new"}],
      "controls_schema":[{"id":"gain","user_metadata":{"tag":"b"},"calibration":{"user_metadata":"signed"}}],
      "stems_registry":[{"stem_id":"a","user_metadata":{"tag":"c"}}],
      "slimmable_tiers":{"lite":{"channels":4,"prefix_crc32c":99},"future":{"prefix_crc32c":13}}
    }"#;
        let mod_val = parse_strict_json_object_bytes(modified).unwrap();
        let proj_mod = project_immutable_header_value(&mod_val).unwrap();
        assert_eq!(serialize_jcs_value(&proj_mod, true).unwrap(), jcs_orig);

        // Canonical wire validation succeeds on canonical bytes and rejects non-canonical or unprojected bytes
        assert_eq!(
            validate_canonical_jcs_header_bytes(jcs_orig.as_bytes(), true).unwrap(),
            jcs_orig.as_bytes()
        );
        assert!(validate_canonical_jcs_header_bytes(fixture, true).is_err());

        // UTF-16 code-unit order puts supplementary U+10000 (0xD800 0xDC00) before U+E000 (0xE000)
        let utf16_obj = json!({ "\u{e000}": 2, "\u{10000}": 1 });
        assert_eq!(
            serialize_jcs_value(&utf16_obj, true).unwrap(),
            "{\"\u{10000}\":1,\"\u{e000}\":2}"
        );

        // Duplicate keys (including escaped \u0061 == 'a'), surrogates, and noncharacters are rejected
        for bad in [
            br#"{"a":1,"\u0061":2}"#.as_slice(),
            br#"{"name":"\ud800"}"#.as_slice(),
            br#"{"name":"\ufdd0"}"#.as_slice(),
            br#"{"name":"\uffff"}"#.as_slice(),
            b"[]".as_slice(),
        ] {
            assert!(parse_strict_json_object_bytes(bad).is_err());
        }
    }

    /// verify[container.dnam.delay-histories]
    /// Verifies canonical decimal parsing (`u64` without leading zeros) and `delay_histories` validation.
    #[test]
    fn test_model_declared_delay_histories_validation() {
        assert_eq!(
            parse_canonical_reference_samples("432000").unwrap(),
            432_000
        );
        assert_eq!(
            parse_canonical_reference_samples("18446744073709551615").unwrap(),
            u64::MAX
        );
        for bad in ["", "0", "04", "+4", "4.0", "18446744073709551616"] {
            assert!(parse_canonical_reference_samples(bad).is_err());
        }

        let valid_entries = vec![
            json!({
                "id": "pre.left",
                "stage": "stage1_pre",
                "kind": "fractional",
                "introduced_tier": "lite",
                "sample_format": "fp16",
                "reference_samples": {
                    "lite": "480",
                    "full": "480",
                    "pro": "960",
                    "ultra": "960"
                }
            }),
            json!({
                "id": "tank.ap.0",
                "stage": "stage4_tank",
                "kind": "allpass",
                "introduced_tier": "pro",
                "sample_format": "fp16",
                "reference_samples": {
                    "pro": "149",
                    "ultra": "149"
                }
            }),
        ];
        let parsed = validate_delay_histories(&valid_entries, 3).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].reference_samples_by_tier, [480, 480, 960, 960]);
        assert_eq!(parsed[1].reference_samples_by_tier, [0, 0, 149, 149]);

        // Fractional delay < 4 is rejected
        let too_short = vec![json!({
            "id": "pre.left",
            "stage": "stage1_pre",
            "kind": "fractional",
            "introduced_tier": "lite",
            "sample_format": "fp16",
            "reference_samples": { "lite": "3" }
        })];
        assert!(validate_delay_histories(&too_short, 0).is_err());
    }

    /// verify[container.dnam.load-validation]
    /// verify[container.dnam.stage3-slices]
    /// verify[container.dnam.cp1-profile]
    /// Verifies Stage III separate 64B-aligned slices and `CP1` packed control projection records.
    #[test]
    fn test_stage3_slices_and_cp1_packed_projection_validation() {
        // 1. Stage III slices across Lite (1 block), Full (1 block), Pro (2 blocks), Ultra (4 blocks)
        for (tier, blocks) in [(0u8, 1usize), (1, 1), (2, 2), (3, 4)] {
            let kautz_len = blocks * 128;
            let arena_bytes = vec![0u8; 128 + kautz_len + 64];
            validate_stage3_tier_slices(tier, &arena_bytes, 0, 128, 128, kautz_len).unwrap();
            // Overlapping slices rejected
            assert!(
                validate_stage3_tier_slices(tier, &arena_bytes, 0, 128, 64, kautz_len).is_err()
            );
            // Unaligned offset rejected
            assert!(
                validate_stage3_tier_slices(tier, &arena_bytes, 32, 128, 192, kautz_len).is_err()
            );
        }

        // 2. CP1 contributor sizes: 4*(9k+4) B in Arena 0, 4*(8k+4) B in Arena 3
        assert_eq!(cp1_contributor_record_bytes(1, false), Some(52));
        assert_eq!(cp1_contributor_record_bytes(2, false), Some(88));
        assert_eq!(cp1_contributor_record_bytes(3, false), Some(124));
        assert_eq!(cp1_contributor_record_bytes(1, true), Some(48));
        assert_eq!(cp1_contributor_record_bytes(2, true), Some(80));
        assert_eq!(cp1_contributor_record_bytes(3, true), Some(112));
        assert_eq!(cp1_contributor_record_bytes(0, false), None);
        assert_eq!(cp1_contributor_record_bytes(4, false), None);

        let lanes = [3usize, 1, 2];
        let base_total: usize = lanes
            .iter()
            .map(|&k| cp1_contributor_record_bytes(k, false).unwrap())
            .sum();
        let ultra_total: usize = lanes
            .iter()
            .map(|&k| cp1_contributor_record_bytes(k, true).unwrap())
            .sum();
        let base_bytes = vec![0u8; base_total];
        let ultra_bytes = vec![0u8; ultra_total];
        let decoded =
            decode_cp1_contributor_stream(&base_bytes, Some(&ultra_bytes), &lanes).unwrap();
        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0].lanes, 3);
        assert_eq!(decoded[0].base_main.len(), 24);
        assert_eq!(decoded[0].embedding.len(), 3);
        assert!(decoded[0].extra_factors.is_some());
    }

    /// verify[container.dnam.conditional-rules]
    /// verify[container.dnam.metadata-ui-hints]
    /// Verifies structured switch conjunctions (`when: [{"control_id", "option_id"}]`), safe-interval
    /// intersection, specificity-based calibration curve selection, and midpoint power-skew UI hints.
    #[test]
    fn test_structured_conditional_rules_and_ui_power_skew() {
        let switches = vec![
            SwitchControlSpec {
                control_id: "mode=variant".to_string(),
                options: vec!["x".to_string(), "off".to_string()],
                active_options: vec!["x".to_string(), "off".to_string()],
            },
            SwitchControlSpec {
                control_id: "mode".to_string(),
                options: vec![
                    "variant=x".to_string(),
                    "off".to_string(),
                    "masked".to_string(),
                ],
                active_options: vec!["variant=x".to_string(), "off".to_string()],
            },
        ];

        let bounds_json = json!([
            { "when": [{"control_id": "mode=variant", "option_id": "x"}], "max": 8.0 },
            { "when": [{"control_id": "mode", "option_id": "variant=x"}], "min": 2.0 },
            {
                "when": [
                    {"control_id": "mode=variant", "option_id": "x"},
                    {"control_id": "mode", "option_id": "variant=x"}
                ],
                "min": 4.0,
                "max": 6.0
            }
        ]);
        let rules = parse_structured_conditional_bounds(&bounds_json, &switches).unwrap();
        let eff = BTreeMap::from([
            ("mode=variant".to_string(), "x".to_string()),
            ("mode".to_string(), "variant=x".to_string()),
        ]);
        assert_eq!(
            resolve_conditional_safe_interval((0.0, 10.0), &rules, &switches, &eff).unwrap(),
            (4.0, 6.0)
        );

        // Specificity selection and ambiguity detection on curves
        let curves_json = json!([
            { "when": [{"control_id": "mode=variant", "option_id": "x"}], "anchors": [1.0, 2.0, 3.0] },
            {
                "when": [
                    {"control_id": "mode=variant", "option_id": "x"},
                    {"control_id": "mode", "option_id": "variant=x"}
                ],
                "anchors": [10.0, 20.0, 30.0]
            }
        ]);
        let curves = parse_structured_conditional_curves(&curves_json, &switches).unwrap();
        assert_eq!(
            select_conditional_calibration_curve(&curves, &switches, &eff, None).unwrap(),
            [10.0, 20.0, 30.0]
        );

        let tie_json = json!([
            { "when": [{"control_id": "mode=variant", "option_id": "x"}], "anchors": [1.0, 2.0, 3.0] },
            { "when": [{"control_id": "mode", "option_id": "variant=x"}], "anchors": [4.0, 5.0, 6.0] }
        ]);
        let tie_curves = parse_structured_conditional_curves(&tie_json, &switches).unwrap();
        assert!(select_conditional_calibration_curve(&tie_curves, &switches, &eff, None).is_err());

        // Midpoint-anchored UI power-skew law: x(0.5) == x_mid
        let (gamma, x_at_half, t_back) =
            eval_ui_midpoint_power_skew(20.0, 200.0, 2020.0, 0.5).unwrap();
        assert!(gamma > 0.0);
        assert!((x_at_half - 200.0).abs() < 1e-9);
        assert!((t_back - 0.5).abs() < 1e-12);
    }

    /// verify[container.ckpt.bundle-framing]
    /// verify[container.ckpt.dca1-signature]
    /// verify[container.ckpt.export-template]
    /// verify[container.ckpt.declarative-no-pickle]
    /// verify[container.ckpt.uplo-u]
    /// Verifies `.dnam-ckpt` v2.0 bundle framing, exact `DNCK-CHECKPOINT-PREFIX-V1` SHA-256 test vectors,
    /// `DCA1` envelope validation, signature-preserving bundle stripping, zero-copy `.dnam` runtime export,
    /// and `LAPACK uplo='U'` column-major upper-triangular packing.
    #[test]
    fn test_v2_dnck_checkpoint_bundle_framing_dca1_export_and_uplo_u() {
        // 1. Exact golden vectors from checkpoint wire specification
        let header = br#"{"version":1}"#;
        let bundles_owned: Vec<Vec<u8>> = (0..4u8)
            .map(|tier| {
                let mut b = Vec::with_capacity(192);
                b.extend_from_slice(&[0x41 + tier; 64]);
                b.extend_from_slice(&[0x61 + tier; 64]);
                b.extend_from_slice(&[0x31 + tier; 64]);
                b
            })
            .collect();
        let runtime_slices: Vec<&[u8]> = bundles_owned.iter().map(|b| &b[..64]).collect();
        let runtime = build_v2_open_dnam_container(
            header,
            &[b'S'; 64],
            &[b'E'; 16],
            &runtime_slices,
            None,
            0,
        )
        .unwrap();
        let runtime_layout = validate_v2_open_dnam_layout(&runtime).unwrap();
        let mut shared = runtime[..runtime_layout.tier_start].to_vec();
        shared.extend_from_slice(&[b'C'; 64]);

        let count = 4usize;
        let shared_offset = align64_u64(192 + header.len() as u64).unwrap();
        let auth_offset = align64_u64(shared_offset + shared.len() as u64).unwrap();
        let auth_length = (106 + 33 * count) as u64;
        let bundle_start = align64_u64(auth_offset + auth_length).unwrap();
        let ends: [u64; 4] = core::array::from_fn(|t| bundle_start + 192 * ((t as u64) + 1));

        let mut preamble = [0u8; 64];
        preamble[0..4].copy_from_slice(b"DNCK");
        preamble[4..6].copy_from_slice(&2u16.to_le_bytes());
        preamble[10] = 3;
        preamble[12..16].copy_from_slice(&48_000u32.to_le_bytes());

        let mut directory = [0u8; 128];
        let dir_fields: [u64; 13] = [
            192,
            header.len() as u64,
            shared_offset,
            shared.len() as u64,
            auth_offset,
            auth_length,
            0,
            0,
            bundle_start,
            ends[0],
            ends[1],
            ends[2],
            ends[3],
        ];
        for (idx, val) in dir_fields.iter().enumerate() {
            directory[idx * 8..(idx + 1) * 8].copy_from_slice(&val.to_le_bytes());
        }

        let bundle_refs: Vec<&[u8]> = bundles_owned.iter().map(Vec::as_slice).collect();
        let digests =
            compute_dnck_bundle_prefixes(&preamble, &directory, header, &shared, &bundle_refs);

        let expected_hex = [
            "71979cd5235f0d914d99f79adcd3a3184e382b8019ba86e39891fa4ad9531675",
            "3d9055890214c04d532c434a058c93f6d09a94e8f428aad9b1299232df3a7e68",
            "46d550042a1a5b45e2f359b7d732b7ce5bb83055eb330ecd4a35a21309bf4013",
            "d44970efb24b4f5a79ffb4e9b987c6527e79c61de5784032f9628b0e24b636c0",
        ];
        for (d, exp_h) in digests.iter().zip(expected_hex.iter()) {
            let hex_str: String = d.iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(hex_str, *exp_h);
        }

        let mut envelope = Vec::with_capacity(auth_length as usize);
        envelope.extend_from_slice(b"DCA1\0\0\0\0");
        envelope.extend_from_slice(&[b'K'; 32]);
        envelope.push(3);
        envelope.push(4);
        for (t, d) in digests.iter().enumerate() {
            envelope.push(t as u8);
            envelope.extend_from_slice(d);
        }
        envelope.extend_from_slice(&[b'X'; 64]);

        let writer_msg = build_dnck_writer_manifest_message(&[b'K'; 32], &[0x11; 32], 3, &digests);
        assert!(writer_msg.starts_with(&encode_domain_field(
            "domain",
            b"DNCK-CHECKPOINT-MANIFEST-V1"
        )));

        let mut ckpt = vec![0u8; ends[3] as usize];
        ckpt[..64].copy_from_slice(&preamble);
        ckpt[64..192].copy_from_slice(&directory);
        ckpt[192..192 + header.len()].copy_from_slice(header);
        let s_off = shared_offset as usize;
        ckpt[s_off..s_off + shared.len()].copy_from_slice(&shared);
        let a_off = auth_offset as usize;
        ckpt[a_off..a_off + envelope.len()].copy_from_slice(&envelope);
        let mut b_off = bundle_start as usize;
        for b in &bundles_owned {
            ckpt[b_off..b_off + b.len()].copy_from_slice(b);
            b_off += b.len();
        }
        let ckpt_frame =
            build_dmu1_mutable_frame(br#"{"user_metadata":{"tag":"amp"}}"#, 47).unwrap();
        append_dmue_mutable_suffix(&mut ckpt, &ckpt_frame);
        let suffix = ckpt[ends[3] as usize..].to_vec();

        // 2. Validate stripping to each tier 0..=3 and zero-copy runtime .dnam export
        for tier in 0..4usize {
            let mut stripped_ckpt = ckpt[..ends[tier] as usize].to_vec();
            stripped_ckpt.extend_from_slice(&suffix);
            let (ret_tier, _, boundary) = validate_v2_dnck_checkpoint(&stripped_ckpt).unwrap();
            assert_eq!(ret_tier, tier as u8);
            assert_eq!(boundary, ends[tier] as usize);

            let exported_dnam = export_runtime_dnam_from_ckpt(&stripped_ckpt).unwrap();
            let mut expected_dnam = runtime[..runtime_layout.tier_ends[tier]].to_vec();
            expected_dnam.extend_from_slice(&suffix);
            assert_eq!(exported_dnam, expected_dnam);
        }

        // 3. Verify uplo='U' packing and Nested Principal Submatrix Prefix Theorem across 64/128/192/256
        assert_eq!(ARENA4_STEM_BYTES_LITE, 17_664);
        assert_eq!(ARENA4_STEM_BYTES_FULL, 68_096);
        assert_eq!(ARENA4_STEM_BYTES_PRO, 151_296);
        assert_eq!(ARENA4_STEM_BYTES_ULTRA, 267_264);
        assert_eq!(uplo_u_packed_len(64), 2_080);
        assert_eq!(uplo_u_packed_len(128), 8_256);
        assert_eq!(uplo_u_packed_len(192), 18_528);
        assert_eq!(uplo_u_packed_len(256), 32_896);

        let n_ultra = 256usize;
        let mut dense = vec![0.0_f64; n_ultra * n_ultra];
        for i in 0..n_ultra {
            for j in i..n_ultra {
                let val = ((i + 1) * 1000 + (j + 1)) as f64;
                dense[i * n_ultra + j] = val;
                dense[j * n_ultra + i] = val;
            }
        }
        let mut packed = vec![0.0_f64; uplo_u_packed_len(n_ultra)];
        pack_uplo_u_f64(&dense, n_ultra, &mut packed);

        for n_tier in [64usize, 128, 192, 256] {
            let mut unpacked = vec![0.0_f64; n_tier * n_tier];
            unpack_uplo_u_f64(&packed[..uplo_u_packed_len(n_tier)], n_tier, &mut unpacked);
            for i in 0..n_tier {
                for j in 0..n_tier {
                    assert_eq!(unpacked[i * n_tier + j], dense[i * n_ultra + j]);
                }
            }
        }
    }

    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// verify[container.dnam.open-layout]
        /// verify[container.dnam.mutable-frame]
        /// verify[container.dnam.metadata-ui-hints]
        #[test]
        fn prop_v2_container_stripping_dmu1_and_ui_power_skew_bijection(
            cls in prop::array::uniform4(1usize..=4usize),
            target_tier in 0u8..=3u8,
            x_min in -100.0_f64..0.0_f64,
            span in 10.0_f64..500.0_f64,
            mid_frac in 0.1_f64..0.9_f64,
            t_norm in 0.0_f64..=1.0_f64,
        ) {
            let s0 = vec![0x11u8; cls[0] * 64];
            let s1 = vec![0x22u8; cls[1] * 64];
            let s2 = vec![0x33u8; cls[2] * 64];
            let s3 = vec![0x44u8; cls[3] * 64];
            let container = build_v2_open_dnam_container(
                b"{\"schema_version\":\"2.0\"}",
                &[0u8; 64],
                b"DAU1-TEST",
                &[&s0, &s1, &s2, &s3],
                None,
                0,
            ).unwrap();
            let stripped = strip_v2_open_dnam_container(&container, target_tier).unwrap();
            let layout = validate_v2_open_dnam_layout(&stripped).unwrap();
            prop_assert_eq!(layout.retained_tier, target_tier);

            let x_max = x_min + span;
            let x_mid = x_min + mid_frac * span;
            let (_, phys, back) = eval_ui_midpoint_power_skew(x_min, x_mid, x_max, t_norm).unwrap();
            prop_assert!(phys >= x_min && phys <= x_max);
            prop_assert!((back - t_norm).abs() < 1e-11);
        }
    }
}

#[cfg(test)]
mod jcs_reference_tests {
    #[allow(unused_imports)]
    use alloc::{
        boxed::Box,
        format,
        string::{String, ToString},
        vec,
        vec::Vec,
    };

    use serde_json::json;

    #[test]
    fn utf8_string_order_is_not_jcs_utf16_key_order() {
        let bmp = "\u{e000}";
        let supplementary = "\u{10000}";
        assert!(bmp < supplementary);
        assert!(supplementary.encode_utf16().cmp(bmp.encode_utf16()).is_lt());
        assert_eq!(
            serde_json::to_string(&json!({bmp: 1, supplementary: 2})).unwrap(),
            format!("{{\"{bmp}\":1,\"{supplementary}\":2}}")
        );
    }
}

#[cfg(test)]
mod mutable_reference_tests {
    #[allow(unused_imports)]
    use alloc::{
        boxed::Box,
        format,
        string::{String, ToString},
        vec,
        vec::Vec,
    };

    // Byte-level mutable-frame examples, not a production container parser.

    use crate::math::crc32c;

    #[test]
    fn open_mutable_empty_object_has_a_fixed_crc32c_and_wire_frame() {
        let payload = b"{}";
        let length = u64::try_from(payload.len()).unwrap().to_le_bytes();
        let checksum_input = [b"DMU1".as_slice(), &length, payload].concat();
        assert_eq!(crc32c(b"123456789"), 0xe306_9283);
        assert_eq!(crc32c(&checksum_input), 0xa1cc_b48f);

        let frame = [
            b"DMU1".as_slice(),
            &crc32c(&checksum_input).to_le_bytes(),
            &length,
            payload,
        ]
        .concat();
        assert_eq!(frame, b"DMU1\x8f\xb4\xcc\xa1\x02\0\0\0\0\0\0\0{}");
    }

    #[test]
    fn recomputing_mutable_crc_does_not_authenticate_the_content() {
        let first = b"DMU1\x07\0\0\0\0\0\0\0{\"x\":0}";
        let mut frame = [
            b"DMU1".as_slice(),
            &crc32c(first).to_le_bytes(),
            &first[4..],
        ]
        .concat();
        frame[21] = b'1';
        let altered_checksum = crc32c(&[&frame[..4], &frame[8..16], &frame[16..]].concat());
        let stored_checksum = u32::from_le_bytes(frame[4..8].try_into().unwrap());
        assert_ne!(stored_checksum, altered_checksum);
        frame[4..8].copy_from_slice(&altered_checksum.to_le_bytes());
        assert_eq!(
            u32::from_le_bytes(frame[4..8].try_into().unwrap()),
            crc32c(&[&frame[..4], &frame[8..16], &frame[16..]].concat())
        );
        assert_eq!(&frame[16..], br#"{"x":1}"#);
    }
}
