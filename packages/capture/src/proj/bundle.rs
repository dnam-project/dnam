#![forbid(unsafe_code)]
//! `.dnam-proj` (`DNPJ`) Container View, Builder, `< 1 ms` Capsule Compaction,
//! and `< 0.2 ms` Runtime `.dnam` Exporter (`proj::bundle`).
//!
//! Canonical Reference: `docs/specs/containers/project_container_format.md`
//! (`ref[container.proj.lifecycle-states]`, `ref[container.proj.wght-and-export]`).

use dnam_engine::Tier;
use dnam_engine::container::{
    ARENA_OFFSET_LITE_END, ARENA_OFFSET_ULTRA_END, DEFAULT_DMU1_TRAILER_CAPACITY, DMU1_HEADER_SIZE,
    DnamContainerView, align64_u64, append_dmue_mutable_suffix, build_dmu1_mutable_frame,
    build_dnam_container_with_cp1, crc32c_compute, strip_container_to_tier, update_header_in_place,
    validate_v2_dnam_layout_with_max_profile,
};
use dnam_engine::control::Cp1ContributorSlot;
use dnam_engine::provenance::{
    CryptoProvider, DefaultCryptoProvider, enforce_distribution_policy, sha256_digest,
};
use dnam_engine::schema::DnamError;

use super::directory::{
    CalbChunk64B, DnamProjPreamble, ProjAuthRecord, ProjChunkDirEntry, ProjChunkKind,
    build_proj_auth_signing_message, compute_proj_header_crc32c, parse_proj_manifest_range,
    verify_proj_auth_chunk, verify_proj_inpc_capsule, verify_proj_stat_uplo_u,
    verify_proj_stmc_capsule,
};
use super::schema::DnamProjManifest;
use super::{
    DNPJ_MAGIC, PROJ_FLAG_COMPACTED_CAPSULES, PROJ_FLAG_HAS_AUTH, PROJ_FLAG_HAS_CALIBRATION,
    PROJ_FLAG_HAS_CAPSULES, PROJ_FLAG_HAS_RAW_STEMS, PROJ_FLAG_HAS_STAT, PROJ_FLAG_HAS_WGHT,
    PROJ_FLAG_STEREO_OUT, PROJ_PREAMBLE_SIZE, PROJ_VERSION_MAJOR, PROJ_VERSION_MINOR,
};

/// Input chunk specification passed to [`build_dnam_proj_container`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjChunkInput {
    /// Typed FourCC chunk kind.
    pub kind: ProjChunkKind,
    /// Chunk ID (`0` for singletons, `stem.chunk_id` for `SPCM` and `STMC`).
    pub chunk_id: u32,
    /// Unpadded chunk payload bytes.
    pub payload: Vec<u8>,
}

/// Pre-verified chunk input used internally by `proj::vault` to avoid redundant `SHA-256` / `CRC32C`
/// recomputation on unmodified capsules.
pub(crate) struct VaultChunkInput {
    pub kind: ProjChunkKind,
    pub chunk_id: u32,
    pub payload: Vec<u8>,
    pub cached_digest: Option<(u32, [u8; 32])>,
}

type ValidatedWghtPair<'a> = (&'a [u8], DnamContainerView<'a>);
/// Enforces `b_head = 0.0` (`[STAB-ZERO-DC]`) in `Arena 0..3` at `9_024..9_064` (`Lite..Pro` modes)
/// and `63_424..63_456` (`Ultra` delta modes 5..8).
pub fn zero_arena_b_head_dc_bias(arena_bytes: &mut [u8]) {
    if arena_bytes.len() >= 9_064 {
        arena_bytes[9_024..9_064].fill(0);
    }
    if arena_bytes.len() >= 63_456 {
        arena_bytes[63_424..63_456].fill(0);
    }
}

/// Verifies that `b_head == 0.0` (`[STAB-ZERO-DC]`) holds across all retained modes in `arena_bytes`.
pub fn verify_arena_b_head_zero_dc(arena_bytes: &[u8]) -> Result<(), DnamError> {
    if arena_bytes.len() < ARENA_OFFSET_LITE_END as usize {
        return Err(DnamError::BufferTooSmall {
            expected: ARENA_OFFSET_LITE_END as usize,
            actual: arena_bytes.len(),
        });
    }
    if arena_bytes[9_024..9_064].iter().any(|&b| b != 0) {
        return Err(DnamError::SchemaValidationError(
            "WGHT Arena 0 violates [STAB-ZERO-DC]: non-zero b_head in bytes 9024..9064".to_string(),
        ));
    }
    if arena_bytes.len() >= ARENA_OFFSET_ULTRA_END as usize
        && arena_bytes[63_424..63_456].iter().any(|&b| b != 0)
    {
        return Err(DnamError::SchemaValidationError(
            "WGHT Arena 3 violates [STAB-ZERO-DC]: non-zero b_head in bytes 63424..63456"
                .to_string(),
        ));
    }
    Ok(())
}

/// Zero-copy validated view over a canonical `v2.0` `.dnam-proj` (`DNPJ`) container (`r[container.proj.framing]`).
#[derive(Debug, Clone, PartialEq)]
pub struct DnamProjView<'a> {
    /// Validated `64 B` preamble.
    pub preamble: DnamProjPreamble,
    /// Validated Region 1 chunk directory entries.
    pub directory: Vec<ProjChunkDirEntry>,
    /// Validated Region 2 JCS project manifest.
    pub manifest: DnamProjManifest,
    /// Zero-copy borrowed slices for every chunk in `directory` (same index order).
    pub chunk_slices: Vec<&'a [u8]>,
    /// Verified `Ed25519` creator provenance record when an `AUTH` chunk is present.
    pub auth_record: Option<ProjAuthRecord>,
}

impl<'a> DnamProjView<'a> {
    /// Parses and cryptographically validates a `.dnam-proj` (`DNPJ`) container buffer.
    pub fn from_bytes(bytes: &'a [u8], crypto: &dyn CryptoProvider) -> Result<Self, DnamError> {
        Self::from_bytes_inner(bytes, crypto, true)
    }

    fn from_bytes_inner(
        bytes: &'a [u8],
        crypto: &dyn CryptoProvider,
        verify_chunk_digests: bool,
    ) -> Result<Self, DnamError> {
        Self::from_bytes_with_wght_view(bytes, crypto, verify_chunk_digests).map(|(v, _)| v)
    }

    pub(crate) fn from_bytes_with_wght_view(
        bytes: &'a [u8],
        crypto: &dyn CryptoProvider,
        verify_chunk_digests: bool,
    ) -> Result<(Self, Option<ValidatedWghtPair<'a>>), DnamError> {
        let range = parse_proj_manifest_range(bytes)?;
        let expected_file_end = match range.directory.last() {
            None => range.payload_region_offset as u64,
            Some(last) => last
                .offset
                .checked_add(last.byte_len)
                .and_then(align64_u64)
                .ok_or_else(|| {
                    DnamError::SchemaValidationError("chunk end overflow".to_string())
                })?,
        };
        if (bytes.len() as u64) < expected_file_end {
            return Err(DnamError::BufferTooSmall {
                expected: expected_file_end as usize,
                actual: bytes.len(),
            });
        }
        if (bytes.len() as u64) != expected_file_end {
            return Err(DnamError::SchemaValidationError(format!(
                "trailing bytes after .dnam-proj Region 3: expected {expected_file_end} B, got {} B",
                bytes.len()
            )));
        }

        let mut chunk_slices = Vec::with_capacity(range.directory.len());
        let mut has_calb = false;
        let mut has_spcm = false;
        let mut has_inpc = false;
        let mut has_stmc = false;
        let mut has_stat = false;
        let mut wght_view_opt: Option<ValidatedWghtPair<'a>> = None;
        let mut auth_bytes_opt: Option<&'a [u8]> = None;
        let mut stmc_flags_by_id = std::collections::BTreeMap::new();

        for entry in &range.directory {
            let start = entry.offset as usize;
            let unpadded_end = start + (entry.byte_len as usize);
            let padded_end = align64_u64(unpadded_end as u64).unwrap() as usize;
            let payload = &bytes[start..unpadded_end];
            if bytes[unpadded_end..padded_end].iter().any(|&b| b != 0) {
                return Err(DnamError::SchemaValidationError(format!(
                    "non-zero 64B alignment padding after chunk {:?}",
                    entry.kind()?
                )));
            }
            if verify_chunk_digests {
                let calc_crc = crc32c_compute(payload);
                if calc_crc != entry.crc32c {
                    return Err(DnamError::Crc32cMismatch {
                        region: "chunk",
                        expected: entry.crc32c,
                        computed: calc_crc,
                    });
                }
                let calc_sha = sha256_digest(payload);
                if calc_sha != entry.sha256 {
                    return Err(DnamError::CryptoVerificationFailed(
                        "chunk directory SHA-256 mismatch",
                    ));
                }
            }

            match entry.kind()? {
                ProjChunkKind::Calb => {
                    let _ = CalbChunk64B::from_bytes(payload)?;
                    has_calb = true;
                }
                ProjChunkKind::Spcm => {
                    if payload.len() < 64 || payload[0..4] != *b"SPCM" {
                        return Err(DnamError::SchemaValidationError(
                            "invalid SPCM header".to_string(),
                        ));
                    }
                    has_spcm = true;
                }
                ProjChunkKind::Inpc => {
                    let _ = verify_proj_inpc_capsule(payload)?;
                    has_inpc = true;
                }
                ProjChunkKind::Stmc => {
                    let stmc = verify_proj_stmc_capsule(payload)?;
                    if u32::from(stmc.stem_index) != entry.chunk_id {
                        return Err(DnamError::SchemaValidationError(format!(
                            "STMC stem_index ({}) != chunk directory chunk_id ({})",
                            stmc.stem_index, entry.chunk_id
                        )));
                    }
                    stmc_flags_by_id
                        .insert(entry.chunk_id, (stmc.is_enabled(), stmc.is_star_center()));
                    has_stmc = true;
                }
                ProjChunkKind::Stat => {
                    let _ = verify_proj_stat_uplo_u(payload)?;
                    has_stat = true;
                }
                ProjChunkKind::Wght => {
                    let rt_view = DnamContainerView::from_bytes(payload, &crypto)?;
                    if !rt_view.preamble.is_encrypted_arena() {
                        verify_arena_b_head_zero_dc(rt_view.arena_storage.as_bytes())?;
                    }
                    wght_view_opt = Some((payload, rt_view));
                }
                ProjChunkKind::Auth => {
                    auth_bytes_opt = Some(payload);
                }
            }
            chunk_slices.push(payload);
        }

        let has_wght = wght_view_opt.is_some();
        // Verify lifecycle_flags consistency
        let fl = range.preamble.lifecycle_flags;
        let expected_flags_ok = ((fl & PROJ_FLAG_HAS_CALIBRATION != 0) == has_calb)
            && ((fl & PROJ_FLAG_HAS_RAW_STEMS != 0) == has_spcm)
            && ((fl & PROJ_FLAG_HAS_CAPSULES != 0) == (has_inpc && has_stmc))
            && ((fl & PROJ_FLAG_HAS_STAT != 0) == has_stat)
            && ((fl & PROJ_FLAG_HAS_WGHT != 0) == has_wght)
            && ((fl & PROJ_FLAG_HAS_AUTH != 0) == auth_bytes_opt.is_some())
            && !(has_spcm && (fl & PROJ_FLAG_COMPACTED_CAPSULES != 0));
        if !expected_flags_ok {
            return Err(DnamError::SchemaValidationError(
                "preamble lifecycle_flags inconsistent with Region 1 chunk directory".to_string(),
            ));
        }

        if !range.manifest.stems_registry.is_empty() && !has_stmc && !has_spcm {
            return Err(DnamError::SchemaValidationError(
                "manifest.stems_registry is non-empty without SPCM or STMC chunks".to_string(),
            ));
        }

        // Verify STMC <-> manifest.stems_registry bijection whenever STMC capsules exist
        if has_stmc {
            if !has_inpc {
                return Err(DnamError::SchemaValidationError(
                    "STMC capsules require a shared INPC dry stimulus capsule".to_string(),
                ));
            }
            if stmc_flags_by_id.len() != range.manifest.stems_registry.len() {
                return Err(DnamError::SchemaValidationError(
                    "STMC chunk count does not match manifest.stems_registry length".to_string(),
                ));
            }
            for stem in &range.manifest.stems_registry {
                let Some(&(stmc_en, stmc_star)) = stmc_flags_by_id.get(&stem.chunk_id) else {
                    return Err(DnamError::SchemaValidationError(format!(
                        "missing STMC chunk for stem '{}' (chunk_id {})",
                        stem.stem_id, stem.chunk_id
                    )));
                };
                if stmc_en != stem.enabled {
                    return Err(DnamError::SchemaValidationError(format!(
                        "STMC enabled bit ({stmc_en}) != manifest.stems_registry.enabled ({}) on '{}'",
                        stem.enabled, stem.stem_id
                    )));
                }
                if stmc_star != stem.is_star_center {
                    return Err(DnamError::SchemaValidationError(format!(
                        "STMC star_center bit ({stmc_star}) != manifest.stems_registry.is_star_center ({}) on '{}'",
                        stem.is_star_center, stem.stem_id
                    )));
                }
            }
        }

        let auth_record = if let Some(auth_bytes) = auth_bytes_opt {
            Some(verify_proj_auth_chunk(
                auth_bytes,
                &range.manifest,
                &range.directory,
                crypto,
            )?)
        } else {
            None
        };

        Ok((
            Self {
                preamble: range.preamble,
                directory: range.directory,
                manifest: range.manifest,
                chunk_slices,
                auth_record,
            },
            wght_view_opt,
        ))
    }

    /// Looks up a chunk slice by `(kind, chunk_id)`.
    #[must_use]
    pub fn chunk_slice(&self, kind: ProjChunkKind, chunk_id: u32) -> Option<&'a [u8]> {
        let target_fcc = kind.fourcc();
        for (entry, &slice) in self.directory.iter().zip(&self.chunk_slices) {
            if entry.fourcc == target_fcc && entry.chunk_id == chunk_id {
                return Some(slice);
            }
        }
        None
    }
}

fn assemble_proj_bytes(
    manifest: &DnamProjManifest,
    chunks: &[ProjChunkInput],
    compacted_flag: bool,
) -> Result<Vec<u8>, DnamError> {
    let vault_chunks: Vec<VaultChunkInput> = chunks
        .iter()
        .map(|c| VaultChunkInput {
            kind: c.kind,
            chunk_id: c.chunk_id,
            payload: c.payload.clone(),
            cached_digest: None,
        })
        .collect();
    assemble_proj_bytes_with_digests(manifest, &vault_chunks, compacted_flag)
}

fn assemble_proj_bytes_with_digests(
    manifest: &DnamProjManifest,
    chunks: &[VaultChunkInput],
    compacted_flag: bool,
) -> Result<Vec<u8>, DnamError> {
    if chunks.len() > super::MAX_PROJ_CHUNKS {
        return Err(DnamError::SchemaValidationError(
            "too many chunks for .dnam-proj".to_string(),
        ));
    }
    let manifest_jcs = manifest.to_jcs_bytes()?;
    let manifest_offset = (PROJ_PREAMBLE_SIZE + chunks.len() * 64) as u64;
    let manifest_len = manifest_jcs.len() as u64;
    let payload_start = align64_u64(manifest_offset + manifest_len).ok_or_else(|| {
        DnamError::SchemaValidationError("manifest padded end overflow".to_string())
    })?;

    let mut lifecycle_flags = 0u32;
    if manifest.base.io_config.output_mode == "stereo" {
        lifecycle_flags |= PROJ_FLAG_STEREO_OUT;
    }
    if compacted_flag {
        lifecycle_flags |= PROJ_FLAG_COMPACTED_CAPSULES;
    }

    let mut has_inpc = false;
    let mut has_stmc = false;
    let mut dir_entries = Vec::with_capacity(chunks.len());
    let mut cur_offset = payload_start;

    for ch in chunks {
        if ch.payload.is_empty() {
            return Err(DnamError::SchemaValidationError(format!(
                "empty chunk payload for {:?}",
                ch.kind
            )));
        }
        match ch.kind {
            ProjChunkKind::Calb => lifecycle_flags |= PROJ_FLAG_HAS_CALIBRATION,
            ProjChunkKind::Spcm => {
                lifecycle_flags |= PROJ_FLAG_HAS_RAW_STEMS;
                lifecycle_flags &= !PROJ_FLAG_COMPACTED_CAPSULES;
            }
            ProjChunkKind::Inpc => has_inpc = true,
            ProjChunkKind::Stmc => has_stmc = true,
            ProjChunkKind::Stat => lifecycle_flags |= PROJ_FLAG_HAS_STAT,
            ProjChunkKind::Wght => lifecycle_flags |= PROJ_FLAG_HAS_WGHT,
            ProjChunkKind::Auth => lifecycle_flags |= PROJ_FLAG_HAS_AUTH,
        }
        let (crc, sha) = match ch.cached_digest {
            Some(pair) => pair,
            None => (crc32c_compute(&ch.payload), sha256_digest(&ch.payload)),
        };
        let byte_len = ch.payload.len() as u64;
        dir_entries.push(ProjChunkDirEntry {
            fourcc: ch.kind.fourcc(),
            chunk_id: ch.chunk_id,
            flags: 0,
            crc32c: crc,
            offset: cur_offset,
            byte_len,
            sha256: sha,
        });
        cur_offset = cur_offset
            .checked_add(byte_len)
            .and_then(align64_u64)
            .ok_or_else(|| DnamError::SchemaValidationError("chunk offset overflow".to_string()))?;
    }
    if has_inpc && has_stmc {
        lifecycle_flags |= PROJ_FLAG_HAS_CAPSULES;
    }

    let mut dir_bytes = Vec::with_capacity(dir_entries.len() * 64);
    for entry in &dir_entries {
        dir_bytes.extend_from_slice(&entry.to_bytes());
    }

    let mut preamble = DnamProjPreamble {
        magic: DNPJ_MAGIC,
        version_major: PROJ_VERSION_MAJOR,
        version_minor: PROJ_VERSION_MINOR,
        lifecycle_flags,
        chunk_count: chunks.len() as u32,
        chunk_dir_offset: PROJ_PREAMBLE_SIZE as u64,
        manifest_offset,
        manifest_len,
        header_crc32c: 0,
        reserved: [0u8; 20],
    };
    let tmp_pre = preamble.to_bytes();
    preamble.header_crc32c = compute_proj_header_crc32c(&tmp_pre[0..40], &dir_bytes, &manifest_jcs);

    let total_len = usize::try_from(cur_offset)
        .map_err(|_| DnamError::SchemaValidationError("container exceeds usize".to_string()))?;
    let mut out = vec![0u8; total_len];
    out[0..64].copy_from_slice(&preamble.to_bytes());
    out[64..64 + dir_bytes.len()].copy_from_slice(&dir_bytes);
    let m_off = manifest_offset as usize;
    out[m_off..m_off + manifest_jcs.len()].copy_from_slice(&manifest_jcs);
    for (entry, ch) in dir_entries.iter().zip(chunks) {
        let start = entry.offset as usize;
        out[start..start + ch.payload.len()].copy_from_slice(&ch.payload);
    }
    Ok(out)
}

/// Builds and validates a canonical `v2.0` `.dnam-proj` (`DNPJ`) container from `manifest` and `chunks`.
pub fn build_dnam_proj_container(
    manifest: &DnamProjManifest,
    chunks: &[ProjChunkInput],
) -> Result<Vec<u8>, DnamError> {
    build_dnam_proj_container_with_crypto(manifest, chunks, &DefaultCryptoProvider::default())
}

/// Builds and validates a canonical `v2.0` `.dnam-proj` (`DNPJ`) container using a custom `crypto` provider.
pub fn build_dnam_proj_container_with_crypto(
    manifest: &DnamProjManifest,
    chunks: &[ProjChunkInput],
    crypto: &dyn CryptoProvider,
) -> Result<Vec<u8>, DnamError> {
    let bytes = assemble_proj_bytes(manifest, chunks, false)?;
    let _ = DnamProjView::from_bytes(&bytes, crypto)?;
    Ok(bytes)
}

/// Builds a canonical `v2.0` `.dnam-proj` (`DNPJ`) container, automatically compiling `arena_0_to_3`
/// and `cp1_contributors` into a `[STAB-ZERO-DC]`-verified `WGHT` chunk.
pub fn build_dnam_proj_container_with_cp1(
    manifest: &mut DnamProjManifest,
    non_wght_chunks: &[ProjChunkInput],
    arena_0_to_3: &[u8],
    cp1_contributors: &[Cp1ContributorSlot],
) -> Result<Vec<u8>, DnamError> {
    build_dnam_proj_container_with_cp1_and_compacted(
        manifest,
        non_wght_chunks,
        arena_0_to_3,
        cp1_contributors,
        false,
    )
}

fn grow_open_dnam_dmu1_suffix(
    out_dnam: &mut Vec<u8>,
    merged_hdr: &dnam_engine::schema::DnamJsonHeader,
    crypto: &dyn CryptoProvider,
) -> Result<(), DnamError> {
    merged_hdr.validate()?;
    let layout = validate_v2_dnam_layout_with_max_profile(out_dnam, 1)?;
    let mut dmu_obj = serde_json::Map::new();
    if !merged_hdr.presets.is_empty() {
        dmu_obj.insert(
            "presets".to_string(),
            serde_json::to_value(&merged_hdr.presets)
                .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?,
        );
    }
    if !merged_hdr.user_metadata.is_empty() {
        dmu_obj.insert(
            "user_metadata".to_string(),
            serde_json::to_value(&merged_hdr.user_metadata)
                .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?,
        );
    }
    let mut ctrl_um = serde_json::Map::new();
    for ctrl in &merged_hdr.controls_schema {
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
    let payload_bytes = serde_json::to_vec(&serde_json::Value::Object(dmu_obj))
        .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?;
    let required = DMU1_HEADER_SIZE + payload_bytes.len();
    if required > DEFAULT_DMU1_TRAILER_CAPACITY {
        return Err(DnamError::HeaderCapacityExceeded {
            capacity: DEFAULT_DMU1_TRAILER_CAPACITY,
            required,
        });
    }
    let grown_cap = align64_u64((required + 16) as u64)
        .and_then(|n| usize::try_from(n).ok())
        .map(|n| (n - 16).min(DEFAULT_DMU1_TRAILER_CAPACITY))
        .ok_or_else(|| {
            DnamError::SchemaValidationError("DMU1 grown capacity overflow".to_string())
        })?;
    let dmu1 = build_dmu1_mutable_frame(&payload_bytes, grown_cap)?;
    out_dnam.truncate(layout.mutable_extent.0);
    append_dmue_mutable_suffix(out_dnam, &dmu1);
    let _ = DnamContainerView::from_bytes(out_dnam, &crypto)?;
    Ok(())
}

pub(crate) fn build_dnam_proj_container_from_vault(
    manifest: &mut DnamProjManifest,
    mut vault_chunks: Vec<VaultChunkInput>,
    arena_0_to_3: &[u8],
    cp1_contributors: &[Cp1ContributorSlot],
    compacted_flag: bool,
) -> Result<Vec<u8>, DnamError> {
    enforce_distribution_policy(
        &manifest.base,
        dnam_engine::schema::DnamProtectionProfile::Open,
    )?;
    let mut clean_arena = arena_0_to_3.to_vec();
    zero_arena_b_head_dc_bias(&mut clean_arena);
    let descriptors = manifest.base.compile_clamp_descriptors()?;
    if clean_arena.len() >= 9_024 {
        clean_arena[8_512..9_024].fill(0);
        for (idx, desc) in descriptors.iter().take(8).enumerate() {
            let off = 8_512 + idx * 64;
            clean_arena[off..off + 64].copy_from_slice(&desc.to_bytes());
        }
    }
    let wght_dnam = build_dnam_container_with_cp1(
        &mut manifest.base,
        &clean_arena,
        &clean_arena,
        false,
        cp1_contributors,
    )?;
    vault_chunks.retain(|c| c.kind != ProjChunkKind::Wght);
    vault_chunks.push(VaultChunkInput {
        kind: ProjChunkKind::Wght,
        chunk_id: 0,
        payload: wght_dnam,
        cached_digest: None,
    });
    assemble_proj_bytes_with_digests(manifest, &vault_chunks, compacted_flag)
}

pub(crate) fn build_dnam_proj_container_with_cp1_and_compacted(
    manifest: &mut DnamProjManifest,
    non_wght_chunks: &[ProjChunkInput],
    arena_0_to_3: &[u8],
    cp1_contributors: &[Cp1ContributorSlot],
    compacted_flag: bool,
) -> Result<Vec<u8>, DnamError> {
    let mut clean_arena = arena_0_to_3.to_vec();
    zero_arena_b_head_dc_bias(&mut clean_arena);
    let descriptors = manifest.base.compile_clamp_descriptors()?;
    if clean_arena.len() >= 9_024 {
        clean_arena[8_512..9_024].fill(0);
        for (idx, desc) in descriptors.iter().take(8).enumerate() {
            let off = 8_512 + idx * 64;
            clean_arena[off..off + 64].copy_from_slice(&desc.to_bytes());
        }
    }
    let wght_dnam = build_dnam_container_with_cp1(
        &mut manifest.base,
        &clean_arena,
        &clean_arena,
        false,
        cp1_contributors,
    )?;
    let mut all_chunks: Vec<ProjChunkInput> = non_wght_chunks
        .iter()
        .filter(|c| c.kind != ProjChunkKind::Wght)
        .cloned()
        .collect();
    all_chunks.push(ProjChunkInput {
        kind: ProjChunkKind::Wght,
        chunk_id: 0,
        payload: wght_dnam,
    });
    let crypto = DefaultCryptoProvider::default();
    let bytes = assemble_proj_bytes(manifest, &all_chunks, compacted_flag)?;
    let _ = DnamProjView::from_bytes_inner(&bytes, &crypto, false)?;
    Ok(bytes)
}

/// Signs a `.dnam-proj` container with an `AUTH` (`"DNPJ-PROJECT-MANIFEST-V2"`) `Ed25519` chunk.
pub fn attach_proj_auth_signature<F>(
    proj_bytes: &[u8],
    signer_pubkey_ed25519: [u8; 32],
    sign_fn: F,
    crypto: &dyn CryptoProvider,
) -> Result<Vec<u8>, DnamError>
where
    F: FnOnce(&[u8]) -> [u8; 64],
{
    let view = DnamProjView::from_bytes(proj_bytes, crypto)?;
    let mut non_auth_inputs = Vec::new();
    let mut non_auth_entries = Vec::new();
    for (entry, &slice) in view.directory.iter().zip(&view.chunk_slices) {
        let kind = entry.kind()?;
        if kind != ProjChunkKind::Auth {
            non_auth_entries.push(*entry);
            non_auth_inputs.push(ProjChunkInput {
                kind,
                chunk_id: entry.chunk_id,
                payload: slice.to_vec(),
            });
        }
    }
    let msg =
        build_proj_auth_signing_message(&view.manifest, &signer_pubkey_ed25519, &non_auth_entries)?;
    let sig = sign_fn(&msg);
    let auth_rec = ProjAuthRecord {
        signer_pubkey_ed25519,
        signature_ed25519: sig,
    };
    non_auth_inputs.push(ProjChunkInput {
        kind: ProjChunkKind::Auth,
        chunk_id: 0,
        payload: auth_rec.to_bytes().to_vec(),
    });
    let compacted = (view.preamble.lifecycle_flags & PROJ_FLAG_COMPACTED_CAPSULES) != 0;
    let signed_bytes = assemble_proj_bytes(&view.manifest, &non_auth_inputs, compacted)?;
    let _ = DnamProjView::from_bytes(&signed_bytes, crypto)?;
    Ok(signed_bytes)
}

/// Strips all optional `SPCM` raw audio chunks in `< 1 ms` while preserving `CALB`, `INPC`,
/// `STMC`, `STAT`, and `WGHT` capsules (`r[container.proj.compact]`).
pub fn compact_to_capsules(proj_bytes: &[u8]) -> Result<Vec<u8>, DnamError> {
    let crypto = DefaultCryptoProvider::default();
    let view = DnamProjView::from_bytes(proj_bytes, &crypto)?;
    let mut had_spcm = false;
    let mut has_inpc = false;
    let mut has_stmc = false;
    let mut retained = Vec::with_capacity(view.directory.len());
    for (entry, &slice) in view.directory.iter().zip(&view.chunk_slices) {
        let kind = entry.kind()?;
        if kind == ProjChunkKind::Spcm {
            had_spcm = true;
            continue;
        }
        if kind == ProjChunkKind::Inpc {
            has_inpc = true;
        } else if kind == ProjChunkKind::Stmc {
            has_stmc = true;
        }
        // If SPCM chunks were stripped, drop any stale AUTH signature covering the uncompacted directory.
        if kind == ProjChunkKind::Auth && had_spcm {
            continue;
        }
        retained.push(ProjChunkInput {
            kind,
            chunk_id: entry.chunk_id,
            payload: slice.to_vec(),
        });
    }
    if !view.manifest.stems_registry.is_empty() && (!has_inpc || !has_stmc) {
        return Err(DnamError::SchemaValidationError(
            "cannot compact_to_capsules when stems_registry is non-empty without INPC and STMC capsules".to_string(),
        ));
    }
    if !had_spcm && view.auth_record.is_some() {
        // Re-verify and return exact compacted bytes without invalidating existing AUTH
        let out = assemble_proj_bytes(&view.manifest, &retained, true)?;
        let _ = DnamProjView::from_bytes(&out, &crypto)?;
        return Ok(out);
    }
    let retained_no_auth: Vec<ProjChunkInput> = retained
        .into_iter()
        .filter(|c| c.kind != ProjChunkKind::Auth)
        .collect();
    let out = assemble_proj_bytes(&view.manifest, &retained_no_auth, true)?;
    let _ = DnamProjView::from_bytes(&out, &crypto)?;
    Ok(out)
}

/// Exports a validated runtime `.dnam` (`DNAM`) container from the project's `WGHT` chunk in `< 0.2 ms`
/// (`r[container.proj.wght]`).
pub fn export_runtime_dnam(
    proj_bytes: &[u8],
    target_tier: Option<Tier>,
) -> Result<Vec<u8>, DnamError> {
    export_runtime_dnam_with_crypto(proj_bytes, target_tier, &DefaultCryptoProvider::default())
}

/// Exports a validated runtime `.dnam` (`DNAM`) container from `WGHT` using `crypto`, enforcing
/// `[STAB-ZERO-DC]`, `DistributionPolicy`, and `DAU1` buyer tier-slimming rules (`r[container.proj.wght]`).
pub fn export_runtime_dnam_with_crypto(
    proj_bytes: &[u8],
    target_tier: Option<Tier>,
    crypto: &dyn CryptoProvider,
) -> Result<Vec<u8>, DnamError> {
    let (view, wght_pair_opt) = DnamProjView::from_bytes_with_wght_view(proj_bytes, crypto, true)?;
    let Some((wght_bytes, rt_view)) = wght_pair_opt else {
        return Err(DnamError::SchemaValidationError(
            "cannot export runtime .dnam: project has no solved WGHT chunk".to_string(),
        ));
    };

    enforce_distribution_policy(&rt_view.header, rt_view.protection_profile)?;

    let mut out_dnam = wght_bytes.to_vec();
    let ctrl_meta_differs = rt_view.header.controls_schema.len()
        == view.manifest.base.controls_schema.len()
        && rt_view
            .header
            .controls_schema
            .iter()
            .zip(&view.manifest.base.controls_schema)
            .any(|(a, b)| a.id == b.id && a.user_metadata != b.user_metadata);
    if rt_view.header.presets != view.manifest.base.presets
        || rt_view.header.user_metadata != view.manifest.base.user_metadata
        || ctrl_meta_differs
    {
        let mut merged_hdr = rt_view.header.clone();
        merged_hdr.presets = view.manifest.base.presets.clone();
        merged_hdr.user_metadata = view.manifest.base.user_metadata.clone();
        if merged_hdr.controls_schema.len() == view.manifest.base.controls_schema.len() {
            for (dst, src) in merged_hdr
                .controls_schema
                .iter_mut()
                .zip(&view.manifest.base.controls_schema)
            {
                if dst.id == src.id {
                    dst.user_metadata = src.user_metadata.clone();
                }
            }
        }
        match update_header_in_place(&mut out_dnam, &merged_hdr) {
            Ok(()) => {}
            Err(DnamError::HeaderCapacityExceeded { .. })
                if !rt_view.preamble.is_encrypted_arena() =>
            {
                grow_open_dnam_dmu1_suffix(&mut out_dnam, &merged_hdr, crypto)?;
            }
            Err(err) => return Err(err),
        }
    }

    if let Some(max_tier) = target_tier
        && (max_tier as u8) < (rt_view.preamble.highest_tier() as u8)
    {
        if let Some(ref sigs) = rt_view.header.metadata.signatures
            && let Some(ref rcpt) = sigs.marketplace_receipt
            && rcpt.buyer_tier_arena_sha256.is_none()
        {
            return Err(DnamError::CryptoVerificationFailed(
                "cannot strip buyer-signed WGHT without buyer_tier_arena_sha256",
            ));
        }
        out_dnam = strip_container_to_tier(&out_dnam, max_tier)?;
        let _ = DnamContainerView::from_bytes(&out_dnam, &crypto)?;
    }

    Ok(out_dnam)
}
