//! Canonical v2.0 `.dnam` (`b"DNAM"`) Open Layout, 64B Preamble, 128B Directory,
//! and `DMU1` / `DMUE` Mutable Suffix Framing (`ref[container.dnam.formats]`,
//! `ref[container.dnam.wire-framing]`, `ref[container.dnam.preamble-directory]`,
//! `ref[container.dnam.open-layout]`, `ref[container.dnam.mutable-frame]`,
//! `ref[container.dnam.mutable-json]`, `ref[container.dnam.tier-prefix]`,
//! `ref[container.dnam.sample-rate-anchor]`).

use alloc::collections::BTreeSet;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

use serde_json::Value;

use super::jcs::parse_strict_json_object_bytes;
use crate::math::crc32c;
use crate::provenance::sha256_digest;
use crate::schema::{DnamError, SchemaOptionExt, reject_if};

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
/// Default `DMU1` mutable frame capacity (`4,080 B`) so `DMU1 (4,080 B) + DMUE (16 B) == 4,096 B` (`% 64 == 0`).
pub const DEFAULT_DMU1_TRAILER_CAPACITY: usize = DEFAULT_MAX_MUTABLE_CAPACITY - DMUE_FOOTER_SIZE;
/// Magic bytes for a 64-byte local Castagnoli `CRC32C` integrity record in an unsigned v2 `auth_extent`.
pub const DCRC_MAGIC: [u8; 4] = *b"DCRC";
/// Fixed byte size of a `DCRC` local integrity record (`64 B`).
pub const DCRC_RECORD_SIZE: usize = 64;

/// Local 64-byte Castagnoli `CRC32C` integrity record stored in `auth_extent` for non-`DAU1` v2 `.dnam` containers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DcrcAuthRecord {
    pub crc_prefix: u32,
    pub crc_tiers: [u32; 4],
    pub num_controls: u16,
    pub flags_hint: u16,
    pub tail_time_ms_bits: u32,
}

impl DcrcAuthRecord {
    /// Serializes this 64-byte `DCRC` record into little-endian bytes.
    #[inline]
    pub fn to_bytes(&self) -> [u8; DCRC_RECORD_SIZE] {
        let mut out = [0u8; DCRC_RECORD_SIZE];
        out[0..4].copy_from_slice(&DCRC_MAGIC);
        out[4..8].copy_from_slice(&self.crc_prefix.to_le_bytes());
        out[8..12].copy_from_slice(&self.crc_tiers[0].to_le_bytes());
        out[12..16].copy_from_slice(&self.crc_tiers[1].to_le_bytes());
        out[16..20].copy_from_slice(&self.crc_tiers[2].to_le_bytes());
        out[20..24].copy_from_slice(&self.crc_tiers[3].to_le_bytes());
        out[24..26].copy_from_slice(&self.num_controls.to_le_bytes());
        out[26..28].copy_from_slice(&self.flags_hint.to_le_bytes());
        out[28..32].copy_from_slice(&self.tail_time_ms_bits.to_le_bytes());
        out
    }
}

/// Parses a `DCRC` local integrity record if `auth_slice` starts with `b"DCRC"`.
pub fn parse_dcrc_record(auth_slice: &[u8]) -> Result<Option<DcrcAuthRecord>, DnamError> {
    if auth_slice.len() < 4 || auth_slice[0..4] != DCRC_MAGIC {
        return Ok(None);
    }
    reject_if(
        auth_slice.len() != DCRC_RECORD_SIZE || auth_slice[32..64].iter().any(|&b| b != 0),
        "invalid DCRC auth record",
    )?;
    let u32_at = |off: usize| -> u32 {
        u32::from_le_bytes([
            auth_slice[off],
            auth_slice[off + 1],
            auth_slice[off + 2],
            auth_slice[off + 3],
        ])
    };
    Ok(Some(DcrcAuthRecord {
        crc_prefix: u32_at(4),
        crc_tiers: [u32_at(8), u32_at(12), u32_at(16), u32_at(20)],
        num_controls: u16::from_le_bytes([auth_slice[24], auth_slice[25]]),
        flags_hint: u16::from_le_bytes([auth_slice[26], auth_slice[27]]),
        tail_time_ms_bits: u32_at(28),
    }))
}

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

/// Validated v2.0 Open `.dnam` layout (`ref[container.dnam.open-layout]`).
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

/// Builds a canonical v2.0 `.dnam` container (`magic = b"DNAM"`, `version = (2, 0)`) with an
/// explicit `protection_profile` (`0 = Open`, `1 = DeviceAllowlisted`, `2 = EnclaveSealed`).
pub fn build_v2_dnam_container_with_profile(
    protection_profile: u8,
    header_bytes: &[u8],
    shared_bytes: &[u8],
    auth_bytes: &[u8],
    tier_slices: &[&[u8]],
    mutable_frame_bytes: Option<&[u8]>,
    auth_kind: u8,
) -> Result<Vec<u8>, DnamError> {
    reject_if(
        protection_profile > 2
            || header_bytes.is_empty()
            || auth_bytes.is_empty()
            || tier_slices.is_empty()
            || tier_slices.len() > 4
            || auth_kind > 1,
        "invalid v2 container inputs",
    )?;
    for slice in tier_slices {
        reject_if(
            slice.is_empty() || !slice.len().is_multiple_of(64),
            "tier slice must be non-empty and 64-byte aligned",
        )?;
    }
    let header_offset = V2_HEADER_OFFSET as u64;
    let header_len = header_bytes.len() as u64;
    let shared_offset = header_offset
        .checked_add(header_len)
        .and_then(align64_u64)
        .or_schema_err("header extent overflow")?;
    let shared_len = shared_bytes.len() as u64;
    let auth_offset = shared_offset
        .checked_add(shared_len)
        .and_then(align64_u64)
        .or_schema_err("shared extent overflow")?;
    let auth_len = auth_bytes.len() as u64;
    let tier_start = auth_offset
        .checked_add(auth_len)
        .and_then(align64_u64)
        .or_schema_err("auth extent overflow")?;

    let mut ends = [0u64; 4];
    let mut cur_end = tier_start;
    for (i, slice) in tier_slices.iter().enumerate() {
        cur_end = cur_end
            .checked_add(slice.len() as u64)
            .or_schema_err("tier extent overflow")?;
        ends[i] = cur_end;
    }
    reject_if(cur_end > (u32::MAX as u64), "container exceeds u32::MAX")?;

    let mut out = vec![0u8; cur_end as usize];
    // 1. 64B Preamble
    out[0..4].copy_from_slice(b"DNAM");
    out[4..6].copy_from_slice(&2u16.to_le_bytes());
    out[6..8].copy_from_slice(&0u16.to_le_bytes());
    out[8] = protection_profile;
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
            default_dmu1 = build_dmu1_mutable_frame(b"{}", 48)?;
            &default_dmu1
        }
    };
    append_dmue_mutable_suffix(&mut out, m_bytes);
    Ok(out)
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
    build_v2_dnam_container_with_profile(
        0,
        header_bytes,
        shared_bytes,
        auth_bytes,
        tier_slices,
        mutable_frame_bytes,
        auth_kind,
    )
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

/// Validates a `DMU1` mutable frame (CRC32C, zero padding, and UTF-8 `{...}` framing) without heap allocation.
pub fn validate_dmu1_frame_no_alloc(frame: &[u8]) -> Result<(), DnamError> {
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
    let c0 = crate::math::crc32c_slice(0, b"DMU1");
    let c1 = crate::math::crc32c_slice(c0, &frame[8..16]);
    let calc_crc = crate::math::crc32c_slice(c1, payload);
    if stored_crc != calc_crc {
        return Err(DnamError::Crc32cMismatch {
            region: "DMU1",
            expected: stored_crc,
            computed: calc_crc,
        });
    }
    reject_if(
        payload.starts_with(&[0xEF, 0xBB, 0xBF]),
        "UTF-8 BOM is forbidden",
    )?;
    let text = core::str::from_utf8(payload)
        .map_err(|_| DnamError::SchemaValidationError("invalid UTF-8 JSON".to_string()))?;
    let trimmed = text.trim_ascii();
    reject_if(
        !trimmed.starts_with('{') || !trimmed.ends_with('}'),
        "DMU1 payload must be a JSON object",
    )?;
    Ok(())
}

/// Validates a `DMU1` mutable frame (CRC32C, zero padding, and strict RFC 8259 UTF-8 JSON object without
/// BOM or duplicate keys) and returns the parsed JSON `Value` (`ref[container.dnam.mutable-frame]`).
pub fn parse_and_validate_dmu1_frame(frame: &[u8]) -> Result<Value, DnamError> {
    validate_dmu1_frame_no_alloc(frame)?;
    let n_len = u64::from_le_bytes(frame[8..16].try_into().unwrap()) as usize;
    let payload = &frame[16..16 + n_len];
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

/// Validates a v2.0 `.dnam` container's physical layout up to `max_profile` (`0..=2`) without heap allocation.
pub fn validate_v2_dnam_layout_with_max_profile_no_alloc(
    data: &[u8],
    max_profile: u8,
) -> Result<V2ContainerLayout, DnamError> {
    reject_if(max_profile > 2, "invalid max_profile")?;
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
            || profile > max_profile
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
        header_end > (u32::MAX as u64) || Some(shared_offset) != align64_u64(header_end),
        "invalid directory",
    )?;
    let shared_end = shared_offset
        .checked_add(shared_length)
        .or_schema_err("shared extent overflow")?;
    reject_if(
        shared_end > (u32::MAX as u64)
            || Some(auth_offset) != align64_u64(shared_end)
            || auth_length == 0,
        "invalid directory",
    )?;
    let auth_end = auth_offset
        .checked_add(auth_length)
        .or_schema_err("auth extent overflow")?;
    reject_if(
        auth_end > (u32::MAX as u64)
            || Some(tier_start) != align64_u64(auth_end)
            || tier_start > (u32::MAX as u64),
        "invalid tier start",
    )?;

    let mut previous = tier_start;
    for (index, &end) in ends_u64.iter().enumerate() {
        if index <= (maximum as usize) {
            reject_if(
                end <= previous || !end.is_multiple_of(64) || end > (u32::MAX as u64),
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

    validate_dmu1_frame_no_alloc(&data[mutable_offset..data.len() - DMUE_FOOTER_SIZE])?;

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

/// Validates a v2.0 Open `.dnam` container's physical layout without heap allocation (`ref[container.dnam.open-layout]`,
/// `ref[container.dnam.preamble-directory]`, `ref[container.dnam.mutable-frame]`).
pub fn validate_v2_dnam_layout_no_alloc(data: &[u8]) -> Result<V2ContainerLayout, DnamError> {
    validate_v2_dnam_layout_with_max_profile_no_alloc(data, 0)
}

/// Validates a v2.0 `.dnam` container's physical layout and strict RFC 8259 `DMU1` mutable JSON object up to `max_profile`.
pub fn validate_v2_dnam_layout_with_max_profile(
    data: &[u8],
    max_profile: u8,
) -> Result<V2ContainerLayout, DnamError> {
    let layout = validate_v2_dnam_layout_with_max_profile_no_alloc(data, max_profile)?;
    parse_and_validate_dmu1_frame(&data[layout.mutable_extent.0..data.len() - DMUE_FOOTER_SIZE])?;
    Ok(layout)
}

/// Validates a v2.0 Open `.dnam` container's physical layout (`ref[container.dnam.open-layout]`,
/// `ref[container.dnam.preamble-directory]`, `ref[container.dnam.mutable-frame]`).
pub fn validate_v2_open_dnam_layout(data: &[u8]) -> Result<V2ContainerLayout, DnamError> {
    validate_v2_dnam_layout_with_max_profile(data, 0)
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

#[cfg(test)]
mod spec_tests {
    use super::*;
    use crate::container::{DNAM_MAGIC, DNSL_MAGIC};

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
}

#[cfg(test)]
mod mutable_reference_tests {
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
