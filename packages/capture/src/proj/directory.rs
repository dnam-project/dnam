#![forbid(unsafe_code)]
//! `64 B` `.dnam-proj` (`DNPJ`) Preamble, `64 B` Chunk Directory, Front-Loaded `HTTP Range`
//! Manifest Parser, and `AUTH` Provenance Verifier (`proj::directory`).
//!
//! Canonical Reference: `docs/specs/containers/project_container_format.md`
//! (`ref[container.proj.preamble-and-directory]`, `ref[container.proj.auth-signature]`).

use dnam_engine::container::{
    align64_u64, crc32c_update, encode_domain_field, parse_strict_json_object_bytes,
    serialize_jcs_value,
};
use dnam_engine::provenance::CryptoProvider;
use dnam_engine::schema::{DnamError, check_no_unknown_header_keys};

pub use super::capsules::{
    CalbChunk64B, InpcCapsuleView, StmcCapsuleView, StmcTrajectoryWindow, build_proj_inpc_capsule,
    build_proj_stmc_capsule, set_stmc_capsule_enabled_flag, set_stmc_capsule_star_center_flag,
    verify_proj_inpc_capsule, verify_proj_stat_uplo_u, verify_proj_stmc_capsule,
};
use super::schema::DnamProjManifest;
use super::{
    DNPJ_MAGIC, MAX_PROJ_CHUNKS, PROJ_AUTH_RECORD_BYTES, PROJ_FLAG_KNOWN_MASK, PROJ_PREAMBLE_SIZE,
    PROJ_VERSION_MAJOR, PROJ_VERSION_MINOR,
};

/// Enumerated FourCC chunk kinds permitted in Region 1 of a `.dnam-proj` (`DNPJ`) container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ProjChunkKind {
    /// `b"CALB"`: `64 B` loopback latency, signed gain, polarity, and noise-floor summary.
    Calb,
    /// `b"SPCM"`: Optional aligned raw stem PCM audio (stripped in `< 1 ms` by `compact_to_capsules`).
    Spcm,
    /// `b"INPC"`: Deterministic dry stimulus seed + `12,000`-frame `fp16` dry coreset (`24,640 B` / `48,640 B`).
    Inpc,
    /// `b"STMC"`: `StemCaptureRecord` capsule (`28,800 B` mono / `52,800 B` stereo).
    Stmc,
    /// `b"STAT"`: `StarCenter` `uplo='U'` `G_0` + `C_0` sufficient statistics (`267,264 B` at `Ultra`).
    Stat,
    /// `b"WGHT"`: Latest solved `v2.0` runtime `.dnam` weights (`< 0.2 ms` direct export).
    Wght,
    /// `b"AUTH"`: `128 B` `Ed25519` creator provenance & chunk-directory `sha256` attestation.
    Auth,
}

impl ProjChunkKind {
    /// Returns the 4-byte ASCII FourCC tag for this chunk kind.
    #[must_use]
    pub const fn fourcc(self) -> [u8; 4] {
        match self {
            Self::Calb => *b"CALB",
            Self::Spcm => *b"SPCM",
            Self::Inpc => *b"INPC",
            Self::Stmc => *b"STMC",
            Self::Stat => *b"STAT",
            Self::Wght => *b"WGHT",
            Self::Auth => *b"AUTH",
        }
    }

    /// Parses a 4-byte ASCII FourCC tag into a [`ProjChunkKind`], failing closed on unknown FourCCs.
    pub fn from_fourcc(fourcc: [u8; 4]) -> Result<Self, DnamError> {
        match &fourcc {
            b"CALB" => Ok(Self::Calb),
            b"SPCM" => Ok(Self::Spcm),
            b"INPC" => Ok(Self::Inpc),
            b"STMC" => Ok(Self::Stmc),
            b"STAT" => Ok(Self::Stat),
            b"WGHT" => Ok(Self::Wght),
            b"AUTH" => Ok(Self::Auth),
            _ => Err(DnamError::SchemaValidationError(format!(
                "unknown .dnam-proj chunk FourCC {:?}",
                core::str::from_utf8(&fourcc).unwrap_or("????")
            ))),
        }
    }

    /// Returns `true` if at most one chunk of this kind may appear in a `.dnam-proj` container.
    #[must_use]
    pub const fn is_singleton(self) -> bool {
        matches!(
            self,
            Self::Calb | Self::Inpc | Self::Stat | Self::Wght | Self::Auth
        )
    }
}

/// Region 0 (`0x0000..0x0040`, `64 B`) binary preamble of a `.dnam-proj` (`DNPJ`) container.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DnamProjPreamble {
    /// Magic bytes (`*b"DNPJ"`).
    pub magic: [u8; 4],
    /// Major wire version (`2`).
    pub version_major: u16,
    /// Minor wire version (`0`).
    pub version_minor: u16,
    /// Bitmask of `PROJ_FLAG_*` lifecycle flags.
    pub lifecycle_flags: u32,
    /// Number of `64 B` [`ProjChunkDirEntry`] records in Region 1 (`0..=4096`).
    pub chunk_count: u32,
    /// Byte offset of Region 1 Chunk Directory (`64`).
    pub chunk_dir_offset: u64,
    /// Byte offset of Region 2 RFC 8785 JCS Project Manifest (`64 + 64 * chunk_count`).
    pub manifest_offset: u64,
    /// Unpadded UTF-8 byte length of Region 2 JCS Project Manifest (`> 0`).
    pub manifest_len: u64,
    /// Castagnoli `CRC32C` over `preamble[0..40]` + Region 1 Chunk Directory + Region 2 Manifest JCS.
    pub header_crc32c: u32,
    /// Reserved alignment tail (`0`, `20 B`).
    pub reserved: [u8; 20],
}

const _: () = assert!(core::mem::size_of::<DnamProjPreamble>() == 64);

impl DnamProjPreamble {
    /// Serializes this preamble into an exact `64 B` little-endian array.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        out[0..4].copy_from_slice(&self.magic);
        out[4..6].copy_from_slice(&self.version_major.to_le_bytes());
        out[6..8].copy_from_slice(&self.version_minor.to_le_bytes());
        out[8..12].copy_from_slice(&self.lifecycle_flags.to_le_bytes());
        out[12..16].copy_from_slice(&self.chunk_count.to_le_bytes());
        out[16..24].copy_from_slice(&self.chunk_dir_offset.to_le_bytes());
        out[24..32].copy_from_slice(&self.manifest_offset.to_le_bytes());
        out[32..40].copy_from_slice(&self.manifest_len.to_le_bytes());
        out[40..44].copy_from_slice(&self.header_crc32c.to_le_bytes());
        out[44..64].copy_from_slice(&self.reserved);
        out
    }

    /// Parses and validates a `64 B` [`DnamProjPreamble`] from `bytes[0..64]`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DnamError> {
        if bytes.len() < PROJ_PREAMBLE_SIZE {
            return Err(DnamError::BufferTooSmall {
                expected: PROJ_PREAMBLE_SIZE,
                actual: bytes.len(),
            });
        }
        let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if magic != DNPJ_MAGIC {
            return Err(DnamError::InvalidMagic(magic));
        }
        let version_major = u16::from_le_bytes([bytes[4], bytes[5]]);
        let version_minor = u16::from_le_bytes([bytes[6], bytes[7]]);
        if version_major != PROJ_VERSION_MAJOR || version_minor != PROJ_VERSION_MINOR {
            return Err(DnamError::UnsupportedVersion {
                major: version_major,
                minor: version_minor,
            });
        }
        let lifecycle_flags = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        if (lifecycle_flags & !PROJ_FLAG_KNOWN_MASK) != 0 {
            return Err(DnamError::SchemaValidationError(format!(
                "unknown .dnam-proj lifecycle_flags {lifecycle_flags:#010x}"
            )));
        }
        let chunk_count = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        if (chunk_count as usize) > MAX_PROJ_CHUNKS {
            return Err(DnamError::SchemaValidationError(format!(
                "chunk_count {chunk_count} exceeds MAX_PROJ_CHUNKS ({MAX_PROJ_CHUNKS})"
            )));
        }
        let chunk_dir_offset = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let manifest_offset = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
        let manifest_len = u64::from_le_bytes(bytes[32..40].try_into().unwrap());
        let header_crc32c = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]);
        let mut reserved = [0u8; 20];
        reserved.copy_from_slice(&bytes[44..64]);
        if reserved.iter().any(|&b| b != 0) {
            return Err(DnamError::SchemaValidationError(
                "non-zero reserved bytes in DnamProjPreamble[44..64]".to_string(),
            ));
        }
        if chunk_dir_offset != PROJ_PREAMBLE_SIZE as u64 {
            return Err(DnamError::UnalignedOffset {
                field: "chunk_dir_offset",
                offset: chunk_dir_offset as u32,
            });
        }
        let expected_manifest_off =
            (PROJ_PREAMBLE_SIZE as u64).saturating_add(u64::from(chunk_count) * 64);
        if manifest_offset != expected_manifest_off {
            return Err(DnamError::UnalignedOffset {
                field: "manifest_offset",
                offset: manifest_offset as u32,
            });
        }
        if manifest_len == 0 || manifest_len > 16 * 1024 * 1024 {
            return Err(DnamError::SchemaValidationError(format!(
                "invalid .dnam-proj manifest_len {manifest_len}"
            )));
        }
        Ok(Self {
            magic,
            version_major,
            version_minor,
            lifecycle_flags,
            chunk_count,
            chunk_dir_offset,
            manifest_offset,
            manifest_len,
            header_crc32c,
            reserved,
        })
    }

    /// Returns the `64 B`-aligned file offset where Region 3 payload chunks begin.
    pub fn payload_region_offset(&self) -> Result<usize, DnamError> {
        let raw_end = self
            .manifest_offset
            .checked_add(self.manifest_len)
            .and_then(align64_u64)
            .ok_or_else(|| {
                DnamError::SchemaValidationError("manifest extent overflow".to_string())
            })?;
        usize::try_from(raw_end)
            .map_err(|_| DnamError::SchemaValidationError("manifest offset overflow".to_string()))
    }
}

/// Single `64 B` entry in Region 1 Chunk Directory of a `.dnam-proj` (`DNPJ`) container.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProjChunkDirEntry {
    /// FourCC chunk tag (`CALB`, `SPCM`, `INPC`, `STMC`, `STAT`, `WGHT`, `AUTH`).
    pub fourcc: [u8; 4],
    /// Unique chunk identifier within `fourcc` (`0` for singletons, `stem_index` for `STMC`/`SPCM`).
    pub chunk_id: u32,
    /// Chunk-specific flags (`0` unless defined by the chunk spec).
    pub flags: u32,
    /// Castagnoli `CRC32C` over `bytes[offset..offset + byte_len]`.
    pub crc32c: u32,
    /// `64 B`-aligned file byte offset in Region 3.
    pub offset: u64,
    /// Unpadded chunk payload length in bytes (`padded_end = align64(offset + byte_len)`).
    pub byte_len: u64,
    /// Content-addressed SHA-256 digest of `bytes[offset..offset + byte_len]`.
    pub sha256: [u8; 32],
}

const _: () = assert!(core::mem::size_of::<ProjChunkDirEntry>() == 64);

impl ProjChunkDirEntry {
    /// Returns the typed [`ProjChunkKind`] for this directory entry.
    pub fn kind(&self) -> Result<ProjChunkKind, DnamError> {
        ProjChunkKind::from_fourcc(self.fourcc)
    }

    /// Serializes this entry into an exact `64 B` little-endian array.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        out[0..4].copy_from_slice(&self.fourcc);
        out[4..8].copy_from_slice(&self.chunk_id.to_le_bytes());
        out[8..12].copy_from_slice(&self.flags.to_le_bytes());
        out[12..16].copy_from_slice(&self.crc32c.to_le_bytes());
        out[16..24].copy_from_slice(&self.offset.to_le_bytes());
        out[24..32].copy_from_slice(&self.byte_len.to_le_bytes());
        out[32..64].copy_from_slice(&self.sha256);
        out
    }

    /// Parses a `64 B` directory entry and validates FourCC & `64 B` offset alignment.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DnamError> {
        if bytes.len() < 64 {
            return Err(DnamError::BufferTooSmall {
                expected: 64,
                actual: bytes.len(),
            });
        }
        let fourcc = [bytes[0], bytes[1], bytes[2], bytes[3]];
        let kind = ProjChunkKind::from_fourcc(fourcc)?;
        let chunk_id = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        if kind.is_singleton() && chunk_id != 0 {
            return Err(DnamError::SchemaValidationError(format!(
                "singleton chunk {:?} must have chunk_id == 0, got {chunk_id}",
                kind
            )));
        }
        let flags = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        if flags != 0 {
            return Err(DnamError::SchemaValidationError(format!(
                "non-zero reserved directory entry flags {flags:#x} on {:?}",
                kind
            )));
        }
        let crc32c = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        let offset = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let byte_len = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
        if (offset & 0x3F) != 0 {
            return Err(DnamError::UnalignedOffset {
                field: "chunk_dir.offset",
                offset: offset as u32,
            });
        }
        if byte_len == 0 {
            return Err(DnamError::SchemaValidationError(format!(
                "chunk {:?} has zero byte_len",
                kind
            )));
        }
        let mut sha256 = [0u8; 32];
        sha256.copy_from_slice(&bytes[32..64]);
        Ok(Self {
            fourcc,
            chunk_id,
            flags,
            crc32c,
            offset,
            byte_len,
            sha256,
        })
    }
}

/// Computes the `header_crc32c` over `preamble_bytes[0..40]` + `dir_bytes` + `manifest_jcs_bytes`.
#[must_use]
pub fn compute_proj_header_crc32c(
    preamble_prefix_40: &[u8],
    dir_bytes: &[u8],
    manifest_jcs_bytes: &[u8],
) -> u32 {
    let crc = crc32c_update(0, &preamble_prefix_40[..40]);
    let crc = crc32c_update(crc, dir_bytes);
    crc32c_update(crc, manifest_jcs_bytes)
}

/// Result of front-loaded `HTTP Range` header inspection (`parse_proj_manifest_range`).
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedProjManifestRange {
    /// Validated `64 B` preamble.
    pub preamble: DnamProjPreamble,
    /// Validated Region 1 chunk directory entries (monotonic, non-overlapping, `64 B`-aligned).
    pub directory: Vec<ProjChunkDirEntry>,
    /// Validated Region 2 JCS [`DnamProjManifest`].
    pub manifest: DnamProjManifest,
    /// `64 B`-aligned file offset where Region 3 payload chunks begin.
    pub payload_region_offset: usize,
}

/// Parses and verifies Region 0 (`DnamProjPreamble`), Region 1 (`Vec<ProjChunkDirEntry>`), and
/// Region 2 (`DnamProjManifest` RFC 8785 JCS + `header_crc32c`) from a front-loaded byte range
/// (e.g., `HTTP Range: bytes=0-65535`) **without** requiring Region 3 chunk payloads (`r[container.proj.range-inspect]`).
pub fn parse_proj_manifest_range(range_bytes: &[u8]) -> Result<ParsedProjManifestRange, DnamError> {
    let preamble = DnamProjPreamble::from_bytes(range_bytes)?;
    let payload_region_offset = preamble.payload_region_offset()?;
    if range_bytes.len() < payload_region_offset {
        return Err(DnamError::BufferTooSmall {
            expected: payload_region_offset,
            actual: range_bytes.len(),
        });
    }

    let dir_start = preamble.chunk_dir_offset as usize;
    let manifest_start = preamble.manifest_offset as usize;
    let manifest_end = manifest_start + (preamble.manifest_len as usize);
    let dir_bytes = &range_bytes[dir_start..manifest_start];
    let manifest_bytes = &range_bytes[manifest_start..manifest_end];
    if range_bytes[manifest_end..payload_region_offset]
        .iter()
        .any(|&b| b != 0)
    {
        return Err(DnamError::SchemaValidationError(
            "non-zero padding bytes after .dnam-proj manifest JCS".to_string(),
        ));
    }

    let calc_crc = compute_proj_header_crc32c(&range_bytes[0..40], dir_bytes, manifest_bytes);
    if calc_crc != preamble.header_crc32c {
        return Err(DnamError::Crc32cMismatch {
            region: "header",
            expected: preamble.header_crc32c,
            computed: calc_crc,
        });
    }

    let raw_val = parse_strict_json_object_bytes(manifest_bytes)?;
    let canonical_jcs = serialize_jcs_value(&raw_val, false)?;
    if manifest_bytes != canonical_jcs.as_bytes() {
        return Err(DnamError::InvalidJsonHeader(
            "noncanonical RFC 8785 JCS manifest in .dnam-proj".to_string(),
        ));
    }
    let manifest: DnamProjManifest = serde_json::from_slice(manifest_bytes)
        .map_err(|e| DnamError::SchemaValidationError(e.to_string()))?;
    manifest.validate()?;
    if let Some(in_base) = raw_val.get("base") {
        let norm_base = serde_json::to_value(&manifest.base)
            .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?;
        check_no_unknown_header_keys(in_base, &norm_base, None)?;
    }

    let mut directory = Vec::with_capacity(preamble.chunk_count as usize);
    let mut seen_keys = std::collections::BTreeSet::new();
    let mut expected_next_offset = payload_region_offset as u64;
    for idx in 0..(preamble.chunk_count as usize) {
        let entry = ProjChunkDirEntry::from_bytes(&dir_bytes[idx * 64..(idx + 1) * 64])?;
        if !seen_keys.insert((entry.fourcc, entry.chunk_id)) {
            return Err(DnamError::SchemaValidationError(format!(
                "duplicate chunk directory key ({:?}, {})",
                entry.kind()?,
                entry.chunk_id
            )));
        }
        if entry.offset != expected_next_offset {
            return Err(DnamError::UnalignedOffset {
                field: "chunk_dir.offset",
                offset: entry.offset as u32,
            });
        }
        expected_next_offset = entry
            .offset
            .checked_add(entry.byte_len)
            .and_then(align64_u64)
            .ok_or_else(|| DnamError::SchemaValidationError("chunk extent overflow".to_string()))?;
        directory.push(entry);
    }

    Ok(ParsedProjManifestRange {
        preamble,
        directory,
        manifest,
        payload_region_offset,
    })
}

/// `128 B` `AUTH` creator provenance record (`r[container.proj.auth]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProjAuthRecord {
    /// Verified `32 B` `Ed25519` creator public key.
    pub signer_pubkey_ed25519: [u8; 32],
    /// `64 B` `Ed25519` signature over [`build_proj_auth_signing_message`].
    pub signature_ed25519: [u8; 64],
}

impl ProjAuthRecord {
    /// Serializes this `AUTH` record into an exact `128 B` (`2 × 64 B`) little-endian array.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; PROJ_AUTH_RECORD_BYTES] {
        let mut out = [0u8; PROJ_AUTH_RECORD_BYTES];
        out[0..4].copy_from_slice(b"AUTH");
        out[4..6].copy_from_slice(&2_u16.to_le_bytes());
        out[32..64].copy_from_slice(&self.signer_pubkey_ed25519);
        out[64..128].copy_from_slice(&self.signature_ed25519);
        out
    }

    /// Parses a `128 B` `AUTH` chunk.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DnamError> {
        if bytes.len() != PROJ_AUTH_RECORD_BYTES {
            return Err(DnamError::BufferTooSmall {
                expected: PROJ_AUTH_RECORD_BYTES,
                actual: bytes.len(),
            });
        }
        if bytes[0..4] != *b"AUTH" {
            return Err(DnamError::InvalidMagic([
                bytes[0], bytes[1], bytes[2], bytes[3],
            ]));
        }
        let ver = u16::from_le_bytes([bytes[4], bytes[5]]);
        if ver != 2 {
            return Err(DnamError::UnsupportedVersion {
                major: ver,
                minor: 0,
            });
        }
        if bytes[6..32].iter().any(|&b| b != 0) {
            return Err(DnamError::SchemaValidationError(
                "non-zero reserved bytes in AUTH[6..32]".to_string(),
            ));
        }
        let mut signer_pubkey_ed25519 = [0u8; 32];
        signer_pubkey_ed25519.copy_from_slice(&bytes[32..64]);
        let mut signature_ed25519 = [0u8; 64];
        signature_ed25519.copy_from_slice(&bytes[64..128]);
        Ok(Self {
            signer_pubkey_ed25519,
            signature_ed25519,
        })
    }
}

/// Builds the domain-separated `"DNPJ-PROJECT-MANIFEST-V2"` signing message binding the
/// immutable JCS manifest SHA-256, creator `Ed25519` public key, and every non-`AUTH` chunk's
/// `(fourcc, chunk_id, flags, byte_len, sha256)` tuple in Region 1 order.
pub fn build_proj_auth_signing_message(
    manifest: &DnamProjManifest,
    signer_pubkey_ed25519: &[u8; 32],
    non_auth_entries: &[ProjChunkDirEntry],
) -> Result<Vec<u8>, DnamError> {
    let immutable_sha = manifest.immutable_jcs_sha256()?;
    let mut chunk_table = Vec::with_capacity(non_auth_entries.len() * 52);
    chunk_table.extend_from_slice(&(non_auth_entries.len() as u32).to_le_bytes());
    for entry in non_auth_entries {
        chunk_table.extend_from_slice(&entry.fourcc);
        chunk_table.extend_from_slice(&entry.chunk_id.to_le_bytes());
        chunk_table.extend_from_slice(&entry.flags.to_le_bytes());
        chunk_table.extend_from_slice(&entry.byte_len.to_le_bytes());
        chunk_table.extend_from_slice(&entry.sha256);
    }
    let mut msg = Vec::new();
    msg.extend_from_slice(&encode_domain_field("domain", b"DNPJ-PROJECT-MANIFEST-V2"));
    msg.extend_from_slice(&encode_domain_field(
        "signer_pubkey_ed25519",
        signer_pubkey_ed25519,
    ));
    msg.extend_from_slice(&encode_domain_field(
        "manifest_immutable_sha256",
        &immutable_sha,
    ));
    msg.extend_from_slice(&encode_domain_field("chunk_directory_v2", &chunk_table));
    Ok(msg)
}

/// Verifies a `128 B` `AUTH` chunk against `manifest`, `directory`, and `crypto`.
pub fn verify_proj_auth_chunk(
    auth_bytes: &[u8],
    manifest: &DnamProjManifest,
    directory: &[ProjChunkDirEntry],
    crypto: &dyn CryptoProvider,
) -> Result<ProjAuthRecord, DnamError> {
    let record = ProjAuthRecord::from_bytes(auth_bytes)?;
    let non_auth: Vec<ProjChunkDirEntry> = directory
        .iter()
        .copied()
        .filter(|e| e.fourcc != *b"AUTH")
        .collect();
    let msg = build_proj_auth_signing_message(manifest, &record.signer_pubkey_ed25519, &non_auth)?;
    crypto.ed25519_verify(
        &record.signer_pubkey_ed25519,
        &msg,
        &record.signature_ed25519,
    )?;
    Ok(record)
}
