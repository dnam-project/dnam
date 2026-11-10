#![forbid(unsafe_code)]
//! Zero-Copy or Decrypted `.dnam-ckpt` View ([`DnamCkptView`]) and 4 KB Page-Aligned Builder
//! ([`build_dnam_ckpt_container`]).

use dnam_engine::container::{
    ARENA_FULL_END_CL, ARENA_LITE_END_CL, ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END,
    ARENA_OFFSET_PRO_END, ARENA_OFFSET_ULTRA_END, ARENA_PRO_END_CL, ARENA_ULTRA_END_CL, DNCK_MAGIC,
    DnamContainerView, DnamPreamble, FLAG_ENCRYPTED_ARENA, FLAG_HAS_CALIB_STATS,
    FLAG_HAS_CKPT_STEMS, FLAG_HAS_FULL_DELTA, FLAG_HAS_LITE, FLAG_HAS_PRO_DELTA,
    FLAG_HAS_SIGNATURES, FLAG_HAS_ULTRA_DELTA, FLAG_STEREO_OUT, compute_header_crc32c,
    crc32c_update, decode_shared_controls_extent, is_v2_wire_directory_container,
};
use dnam_engine::control::{ControlChannelRecord, ControlClampDescriptor, Cp1ContributorSlot};
use dnam_engine::memory::ArenaStorage;
use dnam_engine::provenance::{
    CryptoProvider, VerificationStatus, enforce_distribution_policy, verify_container_provenance,
};
use dnam_engine::schema::{DnamError, DnamProtectionProfile, SchemaVersion};

use super::arena::{
    ARENA4_STEM_BYTES_FULL, ARENA4_STEM_BYTES_LITE, ARENA4_STEM_BYTES_PRO, ARENA4_STEM_BYTES_ULTRA,
    verify_ckpt_arena5_calib_stats,
};
use super::bundle::{export_runtime_dnam_from_ckpt, validate_v2_dnck_checkpoint};
use super::schema::DnamCkptHeader;

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
    } else {
        a4_len == stem_bytes || a4_len == num_stems * stem_bytes
    }
}

/// Resolves the byte end of `Arena 4` within a checkpoint arena buffer, accepting both
/// single-matrix (`a4.len() == stem_bytes` when `stems_registry.len() >= 1`) and per-stem
/// (`num_stems * stem_bytes`) layouts.
#[must_use]
pub fn resolve_ckpt_stems_end(
    header: &DnamCkptHeader,
    weights_end: usize,
    num_ckpt_stems: u16,
    total_arena_len: usize,
) -> usize {
    if let Some(end) = header.stems_registry.as_ref().and_then(|stems| {
        stems
            .iter()
            .map(|s| (s.c_audio_tier_f64.byte_offset + s.c_audio_tier_f64.byte_length) as usize)
            .max()
            .filter(|&end| end >= weights_end)
    }) {
        return end;
    }
    let n_stems = num_ckpt_stems as usize;
    if n_stems == 0 {
        return weights_end;
    }
    let rem = total_arena_len.saturating_sub(weights_end);
    for &stem_bytes in &[
        ARENA4_STEM_BYTES_ULTRA,
        ARENA4_STEM_BYTES_PRO,
        ARENA4_STEM_BYTES_FULL,
        ARENA4_STEM_BYTES_LITE,
    ] {
        if rem >= n_stems * stem_bytes {
            return weights_end + n_stems * stem_bytes;
        }
        if rem >= stem_bytes {
            return weights_end + stem_bytes;
        }
    }
    weights_end + n_stems * ARENA4_STEM_BYTES_ULTRA
}

/// Zero-copy or decrypted view over a validated `.dnam-ckpt` (`b"DNCK"`) training checkpoint container.
#[derive(Debug)]
pub struct DnamCkptView<'a> {
    /// 64-byte binary preamble.
    pub preamble: DnamPreamble,
    /// Parsed UTF-8 JSON checkpoint header (`DnamCkptHeader`).
    pub header: DnamCkptHeader,
    /// Cryptographic provenance status (`Unsigned`, `CreatorVerified`, or `MarketplaceVerified`).
    pub verification_status: VerificationStatus,
    /// Resolved container protection profile (`Open`, `DeviceAllowlisted`, or `EnclaveSealed`).
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
    /// Loads and validates a `.dnam-ckpt` (`b"DNCK"`) container from `bytes`.
    pub fn from_bytes<C: CryptoProvider>(bytes: &'a [u8], crypto: &C) -> Result<Self, DnamError> {
        if is_v2_wire_directory_container(bytes) && &bytes[0..4] == b"DNCK" {
            let (tier, _ends, _boundary) = validate_v2_dnck_checkpoint(bytes)?;
            let header_offset = u64::from_le_bytes(bytes[64..72].try_into().unwrap()) as usize;
            let header_length = u64::from_le_bytes(bytes[72..80].try_into().unwrap()) as usize;
            let json_slice = &bytes[header_offset..header_offset + header_length];
            let header: DnamCkptHeader = DnamCkptHeader::from_json_slice(json_slice)
                .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?;
            let exported = export_runtime_dnam_from_ckpt(bytes)?;
            let runtime_view = DnamContainerView::from_bytes(&exported, crypto)?;
            let mut preamble = runtime_view.preamble;
            preamble.magic = DNCK_MAGIC;
            preamble.flags |= FLAG_HAS_CKPT_STEMS | FLAG_HAS_CALIB_STATS;
            if tier >= 1 {
                preamble.flags |= FLAG_HAS_FULL_DELTA;
            }
            let (clamp_descriptors, cp1_contributors, control_records) =
                decode_shared_controls_extent(&[], &header.base)?;
            return Ok(DnamCkptView {
                preamble,
                header,
                verification_status: runtime_view.verification_status,
                protection_profile: runtime_view.protection_profile,
                clamp_descriptors,
                cp1_contributors,
                control_records,
                arena_storage: ArenaStorage::from_slice_owned(
                    runtime_view.arena_storage.as_bytes(),
                ),
            });
        }

        let preamble = DnamPreamble::from_bytes(bytes)?;
        let arena_off = preamble.tensor_arena_offset as usize;
        let arena_len = preamble.tensor_arena_byte_len as usize;
        let json_len = preamble.json_header_byte_len as usize;

        if arena_off < 64 + json_len || bytes.len() < arena_off + arena_len {
            return Err(DnamError::BufferTooSmall {
                expected: arena_off + arena_len,
                actual: bytes.len(),
            });
        }

        let computed_hdr_crc = compute_header_crc32c(bytes, arena_off);
        if computed_hdr_crc != preamble.header_crc32c {
            return Err(DnamError::Crc32cMismatch {
                region: "header",
                expected: preamble.header_crc32c,
                computed: computed_hdr_crc,
            });
        }

        let json_slice = &bytes[64..64 + json_len];
        let header: DnamCkptHeader = DnamCkptHeader::from_json_slice(json_slice)
            .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?;
        header.validate()?;
        let schema_ver = header.base.schema_version()?;
        schema_ver.ensure_matches_preamble(preamble.version_major, preamble.version_minor)?;

        let p_ctrls = header.base.controls_schema.len();
        if arena_off < 64 + json_len + 128 * p_ctrls {
            return Err(DnamError::BufferTooSmall {
                expected: 64 + json_len + 128 * p_ctrls,
                actual: arena_off,
            });
        }
        let control_table_off = arena_off - 128 * p_ctrls;
        let mut control_records = Vec::with_capacity(p_ctrls);
        for p in 0..p_ctrls {
            let rec_off = control_table_off + p * 128;
            let mut rec_bytes = [0u8; 128];
            rec_bytes.copy_from_slice(&bytes[rec_off..rec_off + 128]);
            control_records.push(ControlChannelRecord::from_bytes(&rec_bytes));
        }

        if preamble.is_encrypted_arena() {
            return Err(DnamError::CryptoVerificationFailed(
                "ENCRYPTED_ARENA=1 is not supported in .dnam-ckpt training checkpoints",
            ));
        }
        let raw_arena = &bytes[arena_off..arena_off + arena_len];
        let arena_storage = ArenaStorage::from_slice_or_copy(raw_arena);
        let protection_profile = DnamProtectionProfile::Open;
        enforce_distribution_policy(&header.base, protection_profile)?;

        let plain_bytes = arena_storage.as_bytes();
        DnamContainerView::verify_arena_prefix_crcs(&preamble, plain_bytes)?;

        if preamble.magic == DNCK_MAGIC && preamble.has_calib_stats() {
            let weights_end = preamble.arena_end_for_tier(preamble.highest_tier());
            let stems_end = resolve_ckpt_stems_end(
                &header,
                weights_end,
                preamble.num_ckpt_stems,
                plain_bytes.len(),
            );
            verify_ckpt_arena5_calib_stats(plain_bytes, stems_end)?;
        }

        let model_weights_end = preamble
            .arena_end_for_tier(preamble.highest_tier())
            .min(plain_bytes.len());
        let verification_status = verify_container_provenance(
            &header.base,
            preamble.has_signatures(),
            &plain_bytes[..model_weights_end],
            crypto,
        )?;
        let clamp_descriptors = control_records.iter().map(|r| r.clamp).collect();

        Ok(DnamCkptView {
            preamble,
            header,
            verification_status,
            protection_profile,
            clamp_descriptors,
            cp1_contributors: Vec::new(),
            control_records,
            arena_storage,
        })
    }
}

/// Builds a 4 KB page-aligned `.dnam-ckpt` (`b"DNCK"`) container byte vector.
pub fn build_dnam_ckpt_container(
    header: &mut DnamCkptHeader,
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

    header.base.container_kind = "dnam-ckpt".to_string();
    let num_ckpt_stems = header.stems_registry.as_ref().map_or(0, |s| s.len() as u16);

    let first_stem_off = header.stems_registry.as_ref().and_then(|stems| {
        stems
            .iter()
            .map(|s| s.g_audio_uplo_u_f64.byte_offset as usize)
            .min()
            .filter(|&off| off >= ARENA_OFFSET_LITE_END as usize)
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

    header.base.slimmable_tiers.lite.prefix_crc32c = crc_lite;
    header.base.slimmable_tiers.full.prefix_crc32c = crc_full;
    header.base.slimmable_tiers.pro.prefix_crc32c = crc_pro;
    header.base.slimmable_tiers.ultra.prefix_crc32c = crc_ultra;

    let stems_end = resolve_ckpt_stems_end(
        header,
        weights_end,
        num_ckpt_stems,
        stored_arena_bytes.len(),
    );
    let has_calib_stats = stored_arena_bytes.len() > stems_end;
    if has_calib_stats {
        verify_ckpt_arena5_calib_stats(stored_arena_bytes, stems_end)?;
    }
    if header.base.metadata.signatures.is_none()
        && (header.base.version == "1.0.0" || header.base.version == "1.1.0")
    {
        header.base.version = SchemaVersion::V2_0_0.as_semver_string();
    }

    let json_bytes = header
        .to_json_vec()
        .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?;
    let p_ctrls = header.base.controls_schema.len();
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
    if header.base.metadata.signatures.is_some() {
        flags |= FLAG_HAS_SIGNATURES;
    }
    if header.base.io_config.output_mode != "mono" {
        flags |= FLAG_STEREO_OUT;
    }
    if num_ckpt_stems > 0 {
        flags |= FLAG_HAS_CKPT_STEMS;
    }
    if has_calib_stats {
        flags |= FLAG_HAS_CALIB_STATS;
    }

    let mut container = vec![0u8; tensor_arena_offset + stored_arena_bytes.len()];
    let control_table_off = tensor_arena_offset.saturating_sub(128 * p_ctrls);
    container[64..64 + json_bytes.len()].copy_from_slice(&json_bytes);
    container[64 + json_bytes.len()..control_table_off].fill(0x20);
    if let Ok(compiled_descs) = header.base.compile_clamp_descriptors() {
        for (p, desc) in compiled_descs.iter().enumerate() {
            let rec_off = control_table_off + p * 128;
            if rec_off + 64 <= tensor_arena_offset {
                container[rec_off..rec_off + 64].copy_from_slice(&desc.to_bytes());
            }
        }
    }
    container[tensor_arena_offset..].copy_from_slice(stored_arena_bytes);

    let version = SchemaVersion::parse(&header.base.version)?;
    let preamble = DnamPreamble {
        magic: DNCK_MAGIC,
        version_major: version.major,
        version_minor: version.minor,
        flags,
        num_controls: header.base.controls_schema.len().min(u16::MAX as usize) as u16,
        sample_rate_train_hz: header.base.sample_rate_train_hz,
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
        num_ckpt_stems,
        _reserved: 0,
        tail_time_ms: header.base.io_config.tail_time_ms,
        header_crc32c: 0,
    };

    container[0..64].copy_from_slice(&preamble.to_bytes());
    let computed_header_crc = compute_header_crc32c(&container, tensor_arena_offset);
    container[0x3C..0x40].copy_from_slice(&computed_header_crc.to_le_bytes());

    Ok(container)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ckpt::{
        ExtractedPhysics, StemProvenance, StemRegistryEntry, build_v2_dnck_checkpoint,
    };
    use dnam_engine::container::{
        ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_PRO_END, ARENA_OFFSET_ULTRA_END,
        build_v2_open_dnam_container, crc32c_compute,
    };
    use dnam_engine::provenance::DefaultCryptoProvider;
    use dnam_engine::schema::TensorSliceEntry;
    use std::collections::BTreeMap;

    #[test]
    fn test_arena4_slice_len_and_resolve_ckpt_stems_end() {
        assert!(is_valid_arena4_stem_slice_len(0, 0, ARENA4_STEM_BYTES_LITE));
        assert!(!is_valid_arena4_stem_slice_len(
            64,
            0,
            ARENA4_STEM_BYTES_LITE
        ));
        assert!(is_valid_arena4_stem_slice_len(
            ARENA4_STEM_BYTES_LITE,
            3,
            ARENA4_STEM_BYTES_LITE
        ));
        assert!(is_valid_arena4_stem_slice_len(
            3 * ARENA4_STEM_BYTES_LITE,
            3,
            ARENA4_STEM_BYTES_LITE
        ));

        let hdr = DnamCkptHeader::default_reference();
        let w_end = ARENA_OFFSET_LITE_END as usize;
        assert_eq!(resolve_ckpt_stems_end(&hdr, w_end, 0, w_end), w_end);
        assert_eq!(
            resolve_ckpt_stems_end(&hdr, w_end, 2, w_end + 2 * ARENA4_STEM_BYTES_ULTRA),
            w_end + 2 * ARENA4_STEM_BYTES_ULTRA
        );
        assert_eq!(
            resolve_ckpt_stems_end(&hdr, w_end, 2, w_end + ARENA4_STEM_BYTES_PRO),
            w_end + ARENA4_STEM_BYTES_PRO
        );
        assert_eq!(
            resolve_ckpt_stems_end(&hdr, w_end, 2, w_end + 64),
            w_end + 2 * ARENA4_STEM_BYTES_ULTRA
        );
    }

    #[test]
    fn test_build_and_view_dnam_ckpt_container_tiers_stems_and_calib_stats() {
        let crypto = DefaultCryptoProvider::default();
        let mut hdr = DnamCkptHeader::default_reference();

        // Unaligned or too small arena rejected
        assert!(build_dnam_ckpt_container(&mut hdr, &[0u8; 64], &[0u8; 64], false).is_err());

        // Lite, Full, Pro, Ultra builds
        for &len in &[
            ARENA_OFFSET_LITE_END as usize,
            ARENA_OFFSET_FULL_END as usize,
            ARENA_OFFSET_PRO_END as usize,
            ARENA_OFFSET_ULTRA_END as usize,
        ] {
            let arena = vec![0u8; len];
            let bytes = build_dnam_ckpt_container(&mut hdr, &arena, &arena, false).unwrap();
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
            user_metadata: BTreeMap::new(),
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

        // Error branches: corrupted header CRC and truncated buffer
        assert!(DnamCkptView::from_bytes(&ckpt_with_a5[..128], &crypto).is_err());
        let mut bad_crc = ckpt_with_a5.clone();
        bad_crc[65] ^= 0xFF;
        assert!(DnamCkptView::from_bytes(&bad_crc, &crypto).is_err());

        // v2 wire-directory DNCK container view
        let hdr_json = serde_json::to_vec(&hdr).unwrap();
        let bundles_owned: Vec<Vec<u8>> = (0..2u8).map(|t| vec![0x10 + t; 192]).collect();
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
        let v2_ckpt = build_v2_dnck_checkpoint(
            &hdr_json,
            &runtime_dnam,
            &[0u8; 64],
            &bundle_refs,
            &[0x22; 32],
            &[0x33; 64],
            None,
        )
        .unwrap();
        let v2_view = DnamCkptView::from_bytes(&v2_ckpt, &crypto).unwrap();
        assert_eq!(v2_view.preamble.magic, DNCK_MAGIC);
    }
}
