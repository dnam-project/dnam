#![forbid(unsafe_code)]
//! Canonical `v2.0` `.dnam-ckpt` View ([`DnamCkptView`]) and `DCA1` Bundle Builder
//! ([`build_dnam_ckpt_container`], [`build_dnam_ckpt_container_with_cp1`]).

use dnam_engine::container::{
    ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_PRO_END, ARENA_OFFSET_ULTRA_END,
    DEFAULT_DMU1_TRAILER_CAPACITY, DMU1_HEADER_SIZE, DMUE_FOOTER_SIZE, DNCK_MAGIC,
    DnamContainerView, DnamPreamble, FLAG_HAS_CALIB_STATS, FLAG_HAS_CKPT_STEMS,
    FLAG_HAS_FULL_DELTA, FLAG_HAS_PRO_DELTA, FLAG_HAS_SIGNATURES, FLAG_HAS_ULTRA_DELTA,
    build_dmu1_mutable_frame, build_dnam_container_with_cp1, parse_and_validate_dmu1_frame,
    parse_strict_json_object_bytes, validate_mutable_json_schema,
};
use dnam_engine::control::{ControlChannelRecord, ControlClampDescriptor, Cp1ContributorSlot};
use dnam_engine::memory::ArenaStorage;
use dnam_engine::provenance::{
    CryptoProvider, RustCryptoProvider, VerificationStatus, compute_immutable_header_sha256,
    sha256_digest,
};
use dnam_engine::schema::{DnamError, DnamProtectionProfile, SchemaVersion};

use super::arena::{
    ARENA4_STEM_BYTES_FULL, ARENA4_STEM_BYTES_LITE, ARENA4_STEM_BYTES_PRO, ARENA4_STEM_BYTES_ULTRA,
    DNCA_MAGIC, assemble_and_validate_v2_ckpt_arena, validate_ckpt_arena4_and_stems_registry,
    verify_ckpt_arena5_calib_stats,
};
use super::bundle::{
    build_dnck_writer_manifest_message, build_v2_dnck_checkpoint,
    export_runtime_dnam_from_ckpt_with_crypto, validate_v2_dnck_checkpoint,
    verify_v2_dnck_dca1_signature,
};
use super::schema::DnamCkptHeader;

const DEFAULT_CKPT_WRITER_SEED: [u8; 32] = [0x4B; 32];

/// Returns `true` if an `Arena 4` stem slice byte length is valid for `num_stems` and `stem_bytes`:
/// accepts both single-matrix $S_{\text{total}}$-summed $\mathbf{G}_{\text{audio}}, \mathbf{C}_{\text{audio}}$
/// (`a4_len == stem_bytes` when `num_stems >= 1`) and per-stem (`a4_len == num_stems * stem_bytes`).
#[inline]
#[must_use]
pub const fn is_valid_arena4_stem_slice_len(
    a4_len: usize,
    num_stems: usize,
    stem_bytes: usize,
) -> bool {
    if num_stems == 0 {
        a4_len == 0
    } else if a4_len == stem_bytes {
        true
    } else {
        match num_stems.checked_mul(stem_bytes) {
            Some(prod) => a4_len == prod,
            None => false,
        }
    }
}

#[inline]
const fn tier_ordered_stem_sizes(weights_end: usize) -> [usize; 4] {
    const U: usize = ARENA4_STEM_BYTES_ULTRA;
    const P: usize = ARENA4_STEM_BYTES_PRO;
    const F: usize = ARENA4_STEM_BYTES_FULL;
    const L: usize = ARENA4_STEM_BYTES_LITE;
    if weights_end >= ARENA_OFFSET_ULTRA_END as usize {
        [U, P, F, L]
    } else if weights_end >= ARENA_OFFSET_PRO_END as usize {
        [P, U, F, L]
    } else if weights_end >= ARENA_OFFSET_FULL_END as usize {
        [F, P, U, L]
    } else {
        [L, F, P, U]
    }
}

fn registry_stems_end(
    header: &DnamCkptHeader,
    weights_end: usize,
    total_arena_len: usize,
) -> Option<usize> {
    header.stems_registry.as_ref().and_then(|stems| {
        stems
            .iter()
            .filter_map(|s| {
                let g_end = (s.g_audio_uplo_u_f64.byte_offset as usize)
                    .checked_add(s.g_audio_uplo_u_f64.byte_length as usize)?;
                let c_end = (s.c_audio_tier_f64.byte_offset as usize)
                    .checked_add(s.c_audio_tier_f64.byte_length as usize)?;
                Some(g_end.max(c_end))
            })
            .max()
            .filter(|&end| end >= weights_end && end <= total_arena_len)
    })
}

/// Resolves the byte end of `Arena 4` within a checkpoint arena buffer, accepting both
/// single-matrix (`a4.len() == stem_bytes` when `stems_registry.len() >= 1`) and per-stem
/// (`num_stems * stem_bytes`) layouts without overflowing on 32-bit `usize` targets.
#[must_use]
pub fn resolve_ckpt_stems_end(
    header: &DnamCkptHeader,
    weights_end: usize,
    num_ckpt_stems: u16,
    total_arena_len: usize,
) -> usize {
    if let Some(end) = registry_stems_end(header, weights_end, total_arena_len) {
        return end;
    }
    let n_stems = num_ckpt_stems as usize;
    if n_stems == 0 {
        return weights_end;
    }
    let sizes = [
        ARENA4_STEM_BYTES_ULTRA,
        ARENA4_STEM_BYTES_PRO,
        ARENA4_STEM_BYTES_FULL,
        ARENA4_STEM_BYTES_LITE,
    ];
    let rem = total_arena_len.saturating_sub(weights_end);
    for stem_bytes in sizes {
        if let Some(multi) = n_stems.checked_mul(stem_bytes)
            && rem >= multi
            && let Some(end) = weights_end.checked_add(multi)
        {
            return end;
        }
        if rem >= stem_bytes
            && let Some(end) = weights_end.checked_add(stem_bytes)
        {
            return end;
        }
    }
    weights_end.saturating_add(n_stems.saturating_mul(sizes[0]))
}

fn resolve_ckpt_stems_end_in_arena(
    header: &DnamCkptHeader,
    weights_end: usize,
    num_ckpt_stems: u16,
    arena_bytes: &[u8],
) -> usize {
    let total_len = arena_bytes.len();
    if let Some(end) = registry_stems_end(header, weights_end, total_len) {
        return end;
    }
    let n_stems = num_ckpt_stems as usize;
    if n_stems == 0 {
        return weights_end;
    }
    let sizes = tier_ordered_stem_sizes(weights_end);
    for stem_bytes in sizes {
        for cand_opt in [
            weights_end.checked_add(stem_bytes),
            n_stems
                .checked_mul(stem_bytes)
                .and_then(|m| weights_end.checked_add(m)),
        ] {
            if let Some(cand) = cand_opt
                && cand < total_len
                && verify_ckpt_arena5_calib_stats(arena_bytes, cand).is_ok()
            {
                return cand;
            }
        }
    }
    for stem_bytes in sizes {
        for cand_opt in [
            n_stems
                .checked_mul(stem_bytes)
                .and_then(|m| weights_end.checked_add(m)),
            weights_end.checked_add(stem_bytes),
        ] {
            if let Some(cand) = cand_opt
                && (cand == total_len || arena_bytes.get(cand..cand + 4) == Some(&DNCA_MAGIC))
            {
                return cand;
            }
        }
    }
    resolve_ckpt_stems_end(header, weights_end, num_ckpt_stems, total_len)
}

/// Validated canonical `v2.0` view over a `.dnam-ckpt` (`b"DNCK"`) training checkpoint container.
#[derive(Debug)]
pub struct DnamCkptView<'a> {
    /// Logical 64-byte binary preamble summary.
    pub preamble: DnamPreamble,
    /// Parsed UTF-8 JSON checkpoint header (`DnamCkptHeader`) with `DMU1` mutable fields merged.
    pub header: DnamCkptHeader,
    /// Cryptographic provenance status (`CreatorVerified` or `MarketplaceVerified`).
    pub verification_status: VerificationStatus,
    /// Resolved container protection profile (`Open`).
    pub protection_profile: DnamProtectionProfile,
    /// Compiled 64-byte `ControlClampDescriptor[0..P]` table.
    pub clamp_descriptors: Vec<ControlClampDescriptor>,
    /// Decoded `CP1\0` variable-length contributor slots (if present).
    pub cp1_contributors: Vec<Cp1ContributorSlot>,
    /// Unified 128-byte `ControlChannelRecord[0..P]` table.
    pub control_records: Vec<ControlChannelRecord>,
    /// 64-byte aligned tensor arena storage (`Arena 0..3` + `Arena 4` + optional `Arena 5`).
    pub arena_storage: ArenaStorage<'a>,
}

impl<'a> DnamCkptView<'a> {
    /// Loads and validates a canonical `v2.0` `.dnam-ckpt` (`b"DNCK"`) container from `bytes`.
    pub fn from_bytes<C: CryptoProvider>(bytes: &'a [u8], crypto: &C) -> Result<Self, DnamError> {
        if bytes.len() < 64 {
            return Err(DnamError::BufferTooSmall {
                expected: 64,
                actual: bytes.len(),
            });
        }
        if &bytes[0..4] != b"DNCK" {
            return Err(DnamError::InvalidMagic([
                bytes[0], bytes[1], bytes[2], bytes[3],
            ]));
        }
        let major = u16::from_le_bytes([bytes[4], bytes[5]]);
        let minor = u16::from_le_bytes([bytes[6], bytes[7]]);
        if (major, minor) != (2, 0) {
            return Err(DnamError::UnsupportedVersion { major, minor });
        }
        let (tier, ends, _) = validate_v2_dnck_checkpoint(bytes)?;
        let writer_pubkey = verify_v2_dnck_dca1_signature(bytes, crypto)?;
        let exported = export_runtime_dnam_from_ckpt_with_crypto(bytes, crypto)?;
        let runtime_view = DnamContainerView::from_bytes(&exported, crypto)?;
        build_v2_ckpt_view(bytes, tier, &ends, writer_pubkey, runtime_view)
    }
}

fn build_v2_ckpt_view<'a>(
    bytes: &'a [u8],
    tier: u8,
    ends: &[usize],
    writer_pubkey: [u8; 32],
    runtime_view: DnamContainerView<'_>,
) -> Result<DnamCkptView<'a>, DnamError> {
    let header_offset = u64::from_le_bytes(bytes[64..72].try_into().unwrap()) as usize;
    let header_length = u64::from_le_bytes(bytes[72..80].try_into().unwrap()) as usize;
    let json_slice = &bytes[header_offset..header_offset + header_length];
    let _ = parse_strict_json_object_bytes(json_slice)?;
    let mut header = DnamCkptHeader::from_json_slice(json_slice)
        .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?;

    if header.base.controls_schema.len() != runtime_view.header.controls_schema.len() {
        return Err(DnamError::CryptoVerificationFailed(
            "DNCK checkpoint controls_schema count mismatch against embedded runtime .dnam header",
        ));
    }
    header.base.presets = runtime_view.header.presets.clone();
    header.base.user_metadata = runtime_view.header.user_metadata.clone();
    for (dst_ctrl, src_ctrl) in header
        .base
        .controls_schema
        .iter_mut()
        .zip(&runtime_view.header.controls_schema)
    {
        dst_ctrl.user_metadata = src_ctrl.user_metadata.clone();
    }
    let dmu_slice = &bytes[ends[tier as usize]..bytes.len() - DMUE_FOOTER_SIZE];
    let dmu_val = parse_and_validate_dmu1_frame(dmu_slice)?;
    validate_mutable_json_schema(&dmu_val)?;
    if let Some(stem_um_obj) = dmu_val
        .get("stem_user_metadata")
        .and_then(|v| v.as_object())
        && let Some(stems) = header.stems_registry.as_mut()
    {
        for stem in stems {
            if let Some(entry_val) = stem_um_obj.get(&stem.stem_id) {
                stem.user_metadata = serde_json::from_value(entry_val.clone())
                    .map_err(|e| DnamError::SchemaValidationError(e.to_string()))?;
            }
        }
    }
    header.validate()?;

    // Enforce cryptographic parity between the DNCK wrapper's base header and the embedded .dnam header
    let mut ckpt_base_norm = header.base.clone();
    ckpt_base_norm.container_kind = "dnam".to_string();
    let mut rt_hdr_norm = runtime_view.header;
    rt_hdr_norm.container_kind = "dnam".to_string();
    if tier < bytes[10] {
        ckpt_base_norm
            .tensor_directory
            .retain(|_, e| e.arena <= tier);
        rt_hdr_norm.tensor_directory.retain(|_, e| e.arena <= tier);
    }
    let ckpt_imm_sha = compute_immutable_header_sha256(&ckpt_base_norm)?;
    let rt_imm_sha = compute_immutable_header_sha256(&rt_hdr_norm)?;
    if ckpt_imm_sha != rt_imm_sha
        || ckpt_base_norm.metadata.signatures != rt_hdr_norm.metadata.signatures
    {
        return Err(DnamError::CryptoVerificationFailed(
            "DNCK checkpoint base header immutable digest mismatch against embedded runtime .dnam header",
        ));
    }

    let shared_offset = u64::from_le_bytes(bytes[64 + 16..64 + 24].try_into().unwrap()) as usize;
    let shared_length = u64::from_le_bytes(bytes[64 + 24..64 + 32].try_into().unwrap()) as usize;
    let bundle_start = u64::from_le_bytes(bytes[64 + 64..64 + 72].try_into().unwrap()) as usize;
    let template = &bytes[shared_offset..shared_offset + shared_length];
    let runtime_start = u64::from_le_bytes(template[64 + 64..64 + 72].try_into().unwrap()) as usize;
    let runtime_ends: [usize; 4] = core::array::from_fn(|i| {
        let off = 64 + 72 + i * 8;
        u64::from_le_bytes(template[off..off + 8].try_into().unwrap()) as usize
    });

    let (ckpt_arena, has_tails, has_calib) = assemble_and_validate_v2_ckpt_arena(
        bytes,
        tier,
        bytes[10],
        ends,
        bundle_start,
        runtime_start,
        &runtime_ends,
        runtime_view.arena_storage.as_bytes(),
        &template[runtime_start..],
        &mut header,
        runtime_view.control_records.len(),
    )?;

    let mut preamble = runtime_view.preamble;
    preamble.magic = DNCK_MAGIC;
    let num_stems = match header.stems_registry.as_ref() {
        Some(s) => u16::try_from(s.len()).map_err(|_| {
            DnamError::SchemaValidationError("too many ckpt stems (> 65535)".to_string())
        })?,
        None => 0,
    };
    preamble.num_ckpt_stems = num_stems;
    for (cond, flag) in [
        (has_tails || num_stems > 0, FLAG_HAS_CKPT_STEMS),
        (has_calib, FLAG_HAS_CALIB_STATS),
        (tier >= 1, FLAG_HAS_FULL_DELTA),
        (tier >= 2, FLAG_HAS_PRO_DELTA),
        (tier >= 3, FLAG_HAS_ULTRA_DELTA),
    ] {
        if cond {
            preamble.flags |= flag;
        }
    }
    preamble.tensor_arena_byte_len = u32::try_from(ckpt_arena.len()).map_err(|_| {
        DnamError::SchemaValidationError("ckpt arena exceeds u32::MAX bytes".to_string())
    })?;
    let verification_status = match runtime_view.verification_status {
        VerificationStatus::Unsigned => VerificationStatus::CreatorVerified {
            signer_id: header.base.metadata.author.clone().unwrap_or_default(),
            pubkey_ed25519: writer_pubkey,
        },
        other => other,
    };
    if !matches!(verification_status, VerificationStatus::Unsigned) {
        preamble.flags |= FLAG_HAS_SIGNATURES;
    }
    Ok(DnamCkptView {
        preamble,
        header,
        verification_status,
        protection_profile: runtime_view.protection_profile,
        clamp_descriptors: runtime_view.clamp_descriptors,
        cp1_contributors: runtime_view.cp1_contributors,
        control_records: runtime_view.control_records,
        arena_storage: ArenaStorage::from_slice_owned(&ckpt_arena),
    })
}

fn build_ckpt_dmu1_payload(header: &DnamCkptHeader) -> Result<Vec<u8>, DnamError> {
    let mut obj = serde_json::Map::new();
    if !header.base.presets.is_empty() {
        obj.insert(
            "presets".to_string(),
            serde_json::to_value(&header.base.presets)
                .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?,
        );
    }
    if !header.base.user_metadata.is_empty() {
        obj.insert(
            "user_metadata".to_string(),
            serde_json::to_value(&header.base.user_metadata)
                .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?,
        );
    }
    let mut ctrl_um = serde_json::Map::new();
    for ctrl in &header.base.controls_schema {
        if !ctrl.user_metadata.is_empty() {
            ctrl_um.insert(
                ctrl.id.clone(),
                serde_json::to_value(&ctrl.user_metadata)
                    .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?,
            );
        }
    }
    if !ctrl_um.is_empty() {
        obj.insert(
            "control_user_metadata".to_string(),
            serde_json::Value::Object(ctrl_um),
        );
    }
    if let Some(stems) = header.stems_registry.as_ref() {
        let mut stem_um = serde_json::Map::new();
        for stem in stems {
            if !stem.user_metadata.is_empty() {
                stem_um.insert(
                    stem.stem_id.clone(),
                    serde_json::to_value(&stem.user_metadata)
                        .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?,
                );
            }
        }
        if !stem_um.is_empty() {
            obj.insert(
                "stem_user_metadata".to_string(),
                serde_json::Value::Object(stem_um),
            );
        }
    }
    let doc = serde_json::Value::Object(obj);
    validate_mutable_json_schema(&doc)?;
    serde_json::to_vec(&doc).map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))
}

fn strip_ckpt_dmu1_mutable_fields(val: &mut serde_json::Value) {
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
        if let Some(stems) = root
            .get_mut("stems_registry")
            .and_then(serde_json::Value::as_array_mut)
        {
            for stem in stems {
                if let Some(s_obj) = stem.as_object_mut() {
                    s_obj.remove("user_metadata");
                }
            }
        }
    }
}

/// Builds a canonical `v2.0` 64B-aligned `.dnam-ckpt` (`b"DNCK"`) container byte vector.
pub fn build_dnam_ckpt_container(
    header: &mut DnamCkptHeader,
    plaintext_arena_crc_source: &[u8],
    stored_arena_bytes: &[u8],
    encrypted_arena: bool,
) -> Result<Vec<u8>, DnamError> {
    build_dnam_ckpt_container_with_cp1(
        header,
        plaintext_arena_crc_source,
        stored_arena_bytes,
        encrypted_arena,
        &[],
    )
}

/// Builds a canonical `v2.0` 64B-aligned `.dnam-ckpt` (`b"DNCK"`) container byte vector with
/// optional packed `CP1\0` control contributors in the embedded runtime template.
pub fn build_dnam_ckpt_container_with_cp1(
    header: &mut DnamCkptHeader,
    plaintext_arena_crc_source: &[u8],
    stored_arena_bytes: &[u8],
    encrypted_arena: bool,
    cp1_slots: &[Cp1ContributorSlot],
) -> Result<Vec<u8>, DnamError> {
    if encrypted_arena {
        return Err(DnamError::CryptoVerificationFailed(
            "ENCRYPTED_ARENA=1 is not supported in .dnam-ckpt training checkpoints",
        ));
    }
    let arena_byte_len = u32::try_from(stored_arena_bytes.len()).map_err(|_| {
        DnamError::SchemaValidationError("stored_arena_bytes exceeds u32::MAX".to_string())
    })?;
    if stored_arena_bytes.len() < ARENA_OFFSET_LITE_END as usize
        || (stored_arena_bytes.len() & 0x3F) != 0
    {
        return Err(DnamError::UnalignedOffset {
            field: "stored_arena_bytes.len",
            offset: arena_byte_len,
        });
    }
    if plaintext_arena_crc_source.len() < ARENA_OFFSET_LITE_END as usize {
        return Err(DnamError::BufferTooSmall {
            expected: ARENA_OFFSET_LITE_END as usize,
            actual: plaintext_arena_crc_source.len(),
        });
    }

    header.base.container_kind = "dnam-ckpt".to_string();
    SchemaVersion::parse(&header.base.version)?.ensure_supported()?;
    header.validate()?;
    let num_ckpt_stems = match header.stems_registry.as_ref() {
        Some(s) => u16::try_from(s.len()).map_err(|_| {
            DnamError::SchemaValidationError("stems_registry exceeds u16::MAX".to_string())
        })?,
        None => 0,
    };

    let first_stem_off = header
        .stems_registry
        .as_ref()
        .and_then(|stems| {
            stems
                .iter()
                .map(|s| {
                    s.g_audio_uplo_u_f64
                        .byte_offset
                        .min(s.c_audio_tier_f64.byte_offset) as usize
                })
                .min()
                .filter(|&off| off >= ARENA_OFFSET_LITE_END as usize)
        })
        .or_else(|| {
            [
                ARENA_OFFSET_LITE_END as usize,
                ARENA_OFFSET_FULL_END as usize,
                ARENA_OFFSET_PRO_END as usize,
                ARENA_OFFSET_ULTRA_END as usize,
            ]
            .into_iter()
            .find(|&off| verify_ckpt_arena5_calib_stats(stored_arena_bytes, off).is_ok())
        });
    let weights_capacity = first_stem_off.map_or(stored_arena_bytes.len(), |off| {
        stored_arena_bytes.len().min(off)
    });
    let weights_end = if weights_capacity >= ARENA_OFFSET_ULTRA_END as usize {
        ARENA_OFFSET_ULTRA_END as usize
    } else if weights_capacity >= ARENA_OFFSET_PRO_END as usize {
        ARENA_OFFSET_PRO_END as usize
    } else if weights_capacity >= ARENA_OFFSET_FULL_END as usize {
        ARENA_OFFSET_FULL_END as usize
    } else {
        ARENA_OFFSET_LITE_END as usize
    };
    if plaintext_arena_crc_source.len() < weights_end {
        return Err(DnamError::BufferTooSmall {
            expected: weights_end,
            actual: plaintext_arena_crc_source.len(),
        });
    }

    let stems_end =
        resolve_ckpt_stems_end_in_arena(header, weights_end, num_ckpt_stems, stored_arena_bytes);
    let has_calib_stats = stored_arena_bytes.len() > stems_end;
    validate_ckpt_arena4_and_stems_registry(
        header,
        weights_end,
        stems_end,
        stored_arena_bytes,
        num_ckpt_stems,
        has_calib_stats,
    )?;
    if has_calib_stats {
        verify_ckpt_arena5_calib_stats(stored_arena_bytes, stems_end)?;
    }

    let a4_slice = &stored_arena_bytes[weights_end..stems_end];
    if (a4_slice.len() & 0x3F) != 0 {
        return Err(DnamError::UnalignedOffset {
            field: "arena4_stem_slice.len",
            offset: a4_slice.len() as u32,
        });
    }
    let dnca_slice = &stored_arena_bytes[stems_end..];

    let mut rt_hdr = header.base.clone();
    rt_hdr.container_kind = "dnam".to_string();
    let runtime_dnam = build_dnam_container_with_cp1(
        &mut rt_hdr,
        &plaintext_arena_crc_source[..weights_end],
        &stored_arena_bytes[..weights_end],
        false,
        cp1_slots,
    )?;
    header.base.slimmable_tiers = rt_hdr.slimmable_tiers;

    let dmu_payload = build_ckpt_dmu1_payload(header)?;
    if DMU1_HEADER_SIZE + dmu_payload.len() > DEFAULT_DMU1_TRAILER_CAPACITY {
        return Err(DnamError::HeaderCapacityExceeded {
            capacity: DEFAULT_DMU1_TRAILER_CAPACITY,
            required: DMU1_HEADER_SIZE + dmu_payload.len(),
        });
    }
    let dmu1_frame = build_dmu1_mutable_frame(&dmu_payload, DEFAULT_DMU1_TRAILER_CAPACITY)?;

    let raw_hdr_bytes = header
        .to_json_vec()
        .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?;
    let mut ckpt_hdr_val = parse_strict_json_object_bytes(&raw_hdr_bytes)?;
    strip_ckpt_dmu1_mutable_fields(&mut ckpt_hdr_val);
    let ckpt_hdr_bytes = serde_json::to_vec(&ckpt_hdr_val)
        .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?;

    let lite_end = ARENA_OFFSET_LITE_END as usize;
    let full_end = ARENA_OFFSET_FULL_END as usize;
    let pro_end = ARENA_OFFSET_PRO_END as usize;
    let ultra_end = ARENA_OFFSET_ULTRA_END as usize;

    let mut bundle0 = Vec::with_capacity(lite_end + a4_slice.len());
    bundle0.extend_from_slice(&stored_arena_bytes[..lite_end]);
    bundle0.extend_from_slice(a4_slice);

    let mut bundle_refs: Vec<&[u8]> = vec![bundle0.as_slice()];
    if weights_end >= full_end {
        bundle_refs.push(&stored_arena_bytes[lite_end..full_end]);
    }
    if weights_end >= pro_end {
        bundle_refs.push(&stored_arena_bytes[full_end..pro_end]);
    }
    if weights_end >= ultra_end {
        bundle_refs.push(&stored_arena_bytes[pro_end..ultra_end]);
    }

    let writer_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&DEFAULT_CKPT_WRITER_SEED);
    let mut ckpt = build_v2_dnck_checkpoint(
        &ckpt_hdr_bytes,
        &runtime_dnam,
        dnca_slice,
        &bundle_refs,
        &writer_pk,
        &[0u8; 64],
        Some(&dmu1_frame),
    )?;
    let auth_off = u64::from_le_bytes(ckpt[64 + 32..64 + 40].try_into().unwrap()) as usize;
    let auth_len = u64::from_le_bytes(ckpt[64 + 40..64 + 48].try_into().unwrap()) as usize;
    let count = bundle_refs.len();
    let prefixes: Vec<[u8; 32]> = (0..count)
        .map(|i| {
            ckpt[auth_off + 43 + 33 * i..auth_off + 75 + 33 * i]
                .try_into()
                .unwrap()
        })
        .collect();
    let hdr_sha = sha256_digest(&ckpt_hdr_bytes);
    let manifest_msg =
        build_dnck_writer_manifest_message(&writer_pk, &hdr_sha, (count - 1) as u8, &prefixes);
    let writer_sig = RustCryptoProvider::ed25519_sign(&DEFAULT_CKPT_WRITER_SEED, &manifest_msg);
    let sig_start = auth_off + auth_len - 64;
    ckpt[sig_start..sig_start + 64].copy_from_slice(&writer_sig);

    Ok(ckpt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ckpt::{ExtractedPhysics, StemProvenance, StemRegistryEntry};
    use dnam_engine::container::{build_v2_open_dnam_container, crc32c_compute};
    use dnam_engine::provenance::DefaultCryptoProvider;
    use dnam_engine::schema::TensorSliceEntry;
    use std::collections::BTreeMap;

    #[test]
    fn test_arena4_slice_len_and_resolve_ckpt_stems_end() {
        let l = ARENA4_STEM_BYTES_LITE;
        for (len, n, ok) in [(0, 0, true), (64, 0, false), (l, 3, true), (3 * l, 3, true)] {
            assert_eq!(is_valid_arena4_stem_slice_len(len, n, l), ok);
        }
        let hdr = DnamCkptHeader::default_reference();
        let w = ARENA_OFFSET_LITE_END as usize;
        let u2 = 2 * ARENA4_STEM_BYTES_ULTRA;
        assert_eq!(resolve_ckpt_stems_end(&hdr, w, 0, w), w);
        assert_eq!(resolve_ckpt_stems_end(&hdr, w, 2, w + u2), w + u2);
        assert_eq!(
            resolve_ckpt_stems_end(&hdr, w, 2, w + ARENA4_STEM_BYTES_PRO),
            w + ARENA4_STEM_BYTES_PRO
        );
        assert_eq!(resolve_ckpt_stems_end(&hdr, w, 2, w + 64), w + u2);
        for (te, sz) in [
            (ARENA_OFFSET_LITE_END as usize, l),
            (ARENA_OFFSET_FULL_END as usize, ARENA4_STEM_BYTES_FULL),
            (ARENA_OFFSET_PRO_END as usize, ARENA4_STEM_BYTES_PRO),
            (ARENA_OFFSET_ULTRA_END as usize, ARENA4_STEM_BYTES_ULTRA),
        ] {
            assert_eq!(
                resolve_ckpt_stems_end_in_arena(&hdr, te, 2, &vec![0; te + sz]),
                te + sz
            );
        }
    }

    #[test]
    fn test_build_and_view_dnam_ckpt_container_tiers_stems_and_calib_stats() {
        let crypto = DefaultCryptoProvider::default();
        let mut hdr = DnamCkptHeader::default_reference();

        // Unaligned or too small arena rejected, and v1.x schema rejected
        assert!(build_dnam_ckpt_container(&mut hdr, &[0u8; 64], &[0u8; 64], false).is_err());
        let mut bad_ver = hdr.clone();
        bad_ver.base.version = "1.1.0".to_string();
        let lite_arena = vec![0u8; ARENA_OFFSET_LITE_END as usize];
        assert!(matches!(
            build_dnam_ckpt_container(&mut bad_ver, &lite_arena, &lite_arena, false),
            Err(DnamError::UnsupportedVersion { major: 1, minor: 1 })
        ));

        // Lite, Full, Pro, Ultra canonical v2.0 builds
        for &len in &[
            ARENA_OFFSET_LITE_END as usize,
            ARENA_OFFSET_FULL_END as usize,
            ARENA_OFFSET_PRO_END as usize,
            ARENA_OFFSET_ULTRA_END as usize,
        ] {
            let arena = vec![0u8; len];
            let bytes = build_dnam_ckpt_container(&mut hdr, &arena, &arena, false).unwrap();
            assert_eq!(&bytes[0..8], b"DNCK\x02\x00\x00\x00");
            let view = DnamCkptView::from_bytes(&bytes, &crypto).unwrap();
            assert_eq!(view.preamble.magic, DNCK_MAGIC);
            assert_eq!(view.arena_storage.as_bytes().len(), len);
        }

        // Stem registry + Arena 5 DNCA calibration stats
        let dummy_slice = TensorSliceEntry {
            arena: 4,
            dtype: "f64".to_string(),
            packing: Some("uplo_upper_col_major_tier_ordered".to_string()),
            feature_dim: Some(64),
            packed_len: Some(2080),
            shape: None,
            byte_offset: ARENA_OFFSET_LITE_END,
            byte_length: 64,
        };
        hdr.stems_registry = Some(vec![StemRegistryEntry {
            stem_id: "stem0".to_string(),
            enabled: true,
            controls_meta: BTreeMap::new(),
            provenance: StemProvenance {
                input_pcm_sha256: "00".repeat(32),
                target_pcm_sha256: "11".repeat(32),
                num_frames: 48_000,
            },
            stem_arena4_crc32c: crc32c_compute(&[0u8; 64]),
            user_metadata: BTreeMap::from([("note".to_string(), serde_json::json!("ok"))]),
            extracted_physics: Some(ExtractedPhysics::default()),
            g_audio_uplo_u_f64: dummy_slice.clone(),
            c_audio_tier_f64: dummy_slice,
        }]);
        let stems_end = (ARENA_OFFSET_LITE_END as usize) + 64;
        let mut ckpt_arena = vec![0u8; stems_end + 384];
        let a5 = &mut ckpt_arena[stems_end..];
        a5[0..4].copy_from_slice(b"DNCA");
        a5[4..6].copy_from_slice(&1u16.to_le_bytes());
        a5[8..10].copy_from_slice(&1u16.to_le_bytes());
        a5[10..12].copy_from_slice(&1u16.to_le_bytes());
        a5[12..16].copy_from_slice(&384u32.to_le_bytes());
        a5[64..68].copy_from_slice(b"INPC");
        a5[64 + 56..64 + 60].copy_from_slice(&128u32.to_le_bytes());
        let inpc_crc = crc32c_compute(&a5[128..192]);
        a5[64 + 60..64 + 64].copy_from_slice(&inpc_crc.to_le_bytes());
        a5[192..196].copy_from_slice(b"STMC");
        a5[192 + 116..192 + 120].copy_from_slice(&192u32.to_le_bytes());
        let stmc_crc = crc32c_compute(&a5[320..384]);
        a5[192 + 120..192 + 124].copy_from_slice(&stmc_crc.to_le_bytes());
        let body_crc = crc32c_compute(&a5[64..]);
        a5[16..20].copy_from_slice(&body_crc.to_le_bytes());

        let ckpt_with_a5 =
            build_dnam_ckpt_container(&mut hdr, &ckpt_arena, &ckpt_arena, false).unwrap();
        let view_a5 = DnamCkptView::from_bytes(&ckpt_with_a5, &crypto).unwrap();
        assert_eq!(view_a5.preamble.num_ckpt_stems, 1);
        assert!(view_a5.preamble.has_calib_stats());
        assert_eq!(
            view_a5.header.stems_registry.as_ref().unwrap()[0]
                .user_metadata
                .get("note"),
            Some(&serde_json::json!("ok"))
        );

        // Encrypted arena is rejected fail-fast by build_dnam_ckpt_container
        assert!(matches!(
            build_dnam_ckpt_container(&mut hdr, &ckpt_arena, &ckpt_arena, true),
            Err(DnamError::CryptoVerificationFailed(_))
        ));

        // Error branches: legacy v1 version, truncated buffer, and directory tampering
        let mut legacy_v1 = ckpt_with_a5.clone();
        legacy_v1[4..8].copy_from_slice(&[1, 0, 1, 0]);
        assert!(matches!(
            DnamCkptView::from_bytes(&legacy_v1, &crypto),
            Err(DnamError::UnsupportedVersion { major: 1, minor: 1 })
        ));
        assert!(DnamCkptView::from_bytes(&ckpt_with_a5[..128], &crypto).is_err());
        let mut bad_dir = ckpt_with_a5.clone();
        bad_dir[65] ^= 0xFF;
        assert!(DnamCkptView::from_bytes(&bad_dir, &crypto).is_err());

        // Direct build_v2_dnck_checkpoint (4-tier + DNCA extra_shared + verified Ed25519 DCA1 signature)
        let dnca_block = ckpt_arena[stems_end..].to_vec();
        let bundles_owned: Vec<Vec<u8>> = (0..4u8).map(|t| vec![0x10 + t; 192]).collect();
        let mut v2_a4 = Vec::with_capacity(512);
        for b in &bundles_owned {
            v2_a4.extend_from_slice(&b[64..]);
        }
        let mut v2_hdr = hdr.clone();
        v2_hdr.stems_registry.as_mut().unwrap()[0]
            .user_metadata
            .clear();
        if let Some(s) = v2_hdr.stems_registry.as_mut().and_then(|v| v.first_mut()) {
            s.g_audio_uplo_u_f64.byte_offset = 256;
            s.g_audio_uplo_u_f64.byte_length = 256;
            s.c_audio_tier_f64.byte_offset = 512;
            s.c_audio_tier_f64.byte_length = 256;
            s.stem_arena4_crc32c = crc32c_compute(&v2_a4);
        }
        let hdr_json = serde_json::to_vec(&v2_hdr).unwrap();
        let runtime_slices: Vec<&[u8]> = bundles_owned.iter().map(|b| &b[..64]).collect();
        let runtime_dnam = build_v2_open_dnam_container(
            &hdr_json,
            &[0u8; 64],
            &[0u8; 16],
            &runtime_slices,
            None,
            0,
        )
        .unwrap();
        let bundle_refs: Vec<&[u8]> = bundles_owned.iter().map(Vec::as_slice).collect();
        let seed = [0x22_u8; 32];
        let writer_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&seed);
        let unsigned_v2 = build_v2_dnck_checkpoint(
            &hdr_json,
            &runtime_dnam,
            &dnca_block,
            &bundle_refs,
            &writer_pk,
            &[0u8; 64],
            None,
        )
        .unwrap();
        assert!(DnamCkptView::from_bytes(&unsigned_v2, &crypto).is_err());
        let a_off = u64::from_le_bytes(unsigned_v2[64 + 32..64 + 40].try_into().unwrap()) as usize;
        let pfxs: Vec<[u8; 32]> = (0..4)
            .map(|i| {
                unsigned_v2[a_off + 43 + 33 * i..a_off + 75 + 33 * i]
                    .try_into()
                    .unwrap()
            })
            .collect();
        let hdr_sha = sha256_digest(&hdr_json);
        let manifest_msg = build_dnck_writer_manifest_message(&writer_pk, &hdr_sha, 3, &pfxs);
        let writer_sig = RustCryptoProvider::ed25519_sign(&seed, &manifest_msg);
        let v2_ckpt = build_v2_dnck_checkpoint(
            &hdr_json,
            &runtime_dnam,
            &dnca_block,
            &bundle_refs,
            &writer_pk,
            &writer_sig,
            None,
        )
        .unwrap();
        let v2_view = DnamCkptView::from_bytes(&v2_ckpt, &crypto).unwrap();
        assert_eq!(v2_view.preamble.magic, DNCK_MAGIC);
        assert!((v2_view.preamble.flags & FLAG_HAS_CKPT_STEMS) != 0);
        assert!((v2_view.preamble.flags & FLAG_HAS_CALIB_STATS) != 0);
        assert!((v2_view.preamble.flags & FLAG_HAS_ULTRA_DELTA) != 0);
        assert!(v2_view.preamble.has_signatures());

        assert!(!is_valid_arena4_stem_slice_len(64, usize::MAX, 267_264));
        let mut no_reg_hdr = DnamCkptHeader::default_reference();
        no_reg_hdr.stems_registry = None;
        let lite_stems_end = (ARENA_OFFSET_LITE_END as usize) + ARENA4_STEM_BYTES_LITE;
        let mut lite_no_reg = vec![0u8; lite_stems_end + 70_016];
        let a5_l = &mut lite_no_reg[lite_stems_end..];
        a5_l[0..4].copy_from_slice(b"DNCA");
        a5_l[4..6].copy_from_slice(&1u16.to_le_bytes());
        a5_l[12..16].copy_from_slice(&70_016u32.to_le_bytes());
        let l_crc = crc32c_compute(&a5_l[64..]);
        a5_l[16..20].copy_from_slice(&l_crc.to_le_bytes());
        assert_eq!(
            resolve_ckpt_stems_end_in_arena(&no_reg_hdr, 10_752, 1, &lite_no_reg),
            lite_stems_end
        );
        let _ = resolve_ckpt_stems_end(&no_reg_hdr, 64_000, u16::MAX, lite_no_reg.len());
        let mut bad_hdr = hdr;
        bad_hdr.stems_registry.as_mut().unwrap()[0]
            .g_audio_uplo_u_f64
            .byte_offset = 10_753;
        assert!(build_dnam_ckpt_container(&mut bad_hdr, &ckpt_arena, &ckpt_arena, false).is_err());
    }
}
