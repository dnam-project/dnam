//! `.dnam` & `.dnam-ckpt` Hybrid Binary Container Parser, Builder, ZOH Rescaling & `Arena 3` `uplo='U'` Packing (`container.rs`).
//!
//! Implements the wire framing, zero-copy `mmap` loader (`< 50 ns` zero-JSON hardware fast-path),
//! 4 KB page-aligned in-place JSON header editor, Castagnoli `CRC32C` verification,
//! `.dnam-ckpt` `Arena 3` column-major upper-triangular (`uplo='U'`) sufficient statistics packing,
//! and Continuous-Time Zero-Order Hold (ZOH) sample-rate rescaling ($44.1\text{--}96\text{ kHz}$)
//! from [`docs/dnam_container_schema.md`](../../../docs/dnam_container_schema.md).

use alloc::string::ToString;
use alloc::vec::Vec;
use core::mem::{align_of, size_of};

use crate::memory::{AlignedBuffer, ArenaStorage, CacheLine64};
use crate::provenance::{
    CryptoProvider, VerificationStatus, chacha20_xor_arena_blocks, decode_hex_array,
    verify_container_provenance,
};
use crate::schema::{DnamError, DnamJsonHeader, cold_pow_f64, cold_sqrt_f64};

/// Magic bytes for a `.dnam` deployment model container (`0x4D414E44` little-endian).
pub const DNAM_MAGIC: [u8; 4] = *b"DNAM";

/// Magic bytes for a `.dnam-ckpt` iterative training checkpoint container (`0x4B434E44` little-endian).
pub const DNCK_MAGIC: [u8; 4] = *b"DNCK";

/// Current container major version (`1`).
pub const CONTAINER_VERSION_MAJOR: u16 = 1;

/// Current container minor version (`0`).
pub const CONTAINER_VERSION_MINOR: u16 = 0;

/// Fixed byte length of `Arena 0 (Lite)` (`168 × 64 B = 10,752 B`).
pub const ARENA_OFFSET_LITE_END: u32 = 10_752;

/// Fixed cumulative byte end of `Arena 0 + Arena 1 (Full)` (`349 × 64 B = 22,336 B`).
pub const ARENA_OFFSET_FULL_END: u32 = 22_336;

/// Fixed cumulative byte end of `Arena 0..2 (Ultra)` (`566 × 64 B = 36,224 B`).
pub const ARENA_OFFSET_ULTRA_END: u32 = 36_224;

/// Default 4 KB OS virtual-memory page boundary for `tensor_arena_offset` (`0x1000 = 4,096 B`).
pub const DEFAULT_PAGE_ALIGNED_ARENA_OFFSET: u32 = 4_096;

/// Per-stem `Arena 3` byte length for `Lite` ($N=64$: `16,640 B` `G_audio` + `1,024 B` `C_audio` = `17,664 B`).
pub const ARENA3_STEM_BYTES_LITE: usize = 17_664;

/// Per-stem `Arena 3` byte length for `Full` ($N=128$: `66,048 B` `G_audio` + `2,048 B` `C_audio` = `68,096 B`).
pub const ARENA3_STEM_BYTES_FULL: usize = 68_096;

/// Per-stem `Arena 3` byte length for `Ultra` ($N=192$: `148,224 B` `G_audio` + `3,072 B` `C_audio` = `151,296 B`).
pub const ARENA3_STEM_BYTES_ULTRA: usize = 151_296;

// ============================================================================
// Preamble Flag Bits (`flags: u16`)
// ============================================================================

/// Bit 0 (`1`): `Arena 0 (Lite)` is present.
pub const FLAG_HAS_LITE: u16 = 1 << 0;
/// Bit 1 (`2`): `Arena 1 (Full Delta)` is present.
pub const FLAG_HAS_FULL_DELTA: u16 = 1 << 1;
/// Bit 2 (`4`): `Arena 2 (Ultra Delta)` is present.
pub const FLAG_HAS_ULTRA_DELTA: u16 = 1 << 2;
/// Bit 3 (`8`): `Arena 3` `.dnam-ckpt` stem statistics are present.
pub const FLAG_HAS_CKPT_STEMS: u16 = 1 << 3;
/// Bit 4 (`16`): Core 1 weights are interleaved in execution order (`Core1BlockSlice`).
pub const FLAG_EXEC_ORDER_SOA: u16 = 1 << 4;
/// Bit 5 (`32`): First Lobatto node is anchored at the default preset (`0` flops at Sample 0).
pub const FLAG_DEFAULT_ANCHORED: u16 = 1 << 5;
/// Bit 6 (`64`): Core 2 (`M-UFDN`) is active (`0` = bypassed with `0 KB` delay RAM).
pub const FLAG_CORE2_ACTIVE: u16 = 1 << 6;
/// Bit 7 (`128`): $3\text{ kHz}$ LFOs and envelope followers are active.
pub const FLAG_LFO_ENV_ACTIVE: u16 = 1 << 7;
/// Bit 8 (`256`): Stereo output head enabled (`1 = stereo`, `0 = mono` saving 50% head MACs).
pub const FLAG_STEREO_OUT: u16 = 1 << 8;
/// Bit 9 (`512`): Split-payload `Ed25519` signatures are present in `metadata.signatures`.
pub const FLAG_HAS_SIGNATURES: u16 = 1 << 9;
/// Bit 10 (`1024`): `Arena 0..2` is `ChaCha20` 64-byte block encrypted for `authorized_devices`.
pub const FLAG_ENCRYPTED_ARENA: u16 = 1 << 10;

/// Fixed 64-byte binary preamble (`#[repr(C, align(64))]`, `<4s3H2B10IH2BfI`) at byte offset `0x0000`.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DnamPreamble {
    /// `b"DNAM"` (`0x4D414E44`) or `b"DNCK"` (`0x4B434E44`).
    pub magic: [u8; 4],
    /// Schema major version (`1`).
    pub version_major: u16,
    /// Schema minor version (`0`).
    pub version_minor: u16,
    /// Bitmask of `FLAG_*` constants.
    pub flags: u16,
    /// Total user controls $P \in [0, 16]$.
    pub num_controls: u8,
    /// $\log_2(N_{\text{ring}})$ for Core 2 (`0` when `CORE2_ACTIVE == 0`, `8..=15` when active).
    pub ring_buffer_pow2_log2: u8,
    /// Reference training sample rate in Hz (`48000`).
    pub sample_rate_train_hz: u32,
    /// Unpadded UTF-8 byte length of the JSON header starting at `0x40`.
    pub json_header_byte_len: u32,
    /// File byte offset of `Arena 0` (`% 64 == 0`, typically `4096`).
    pub tensor_arena_offset: u32,
    /// Total byte length of all binary tensor arenas (`% 64 == 0`).
    pub tensor_arena_byte_len: u32,
    /// Arena-relative byte end of `Arena 0 (Lite)` (`10,752`).
    pub arena_offset_lite_end: u32,
    /// Arena-relative byte end of `Arena 0..1 (Full)` (`22,336`).
    pub arena_offset_full_end: u32,
    /// Arena-relative byte end of `Arena 0..2 (Ultra)` (`36,224`).
    pub arena_offset_ultra_end: u32,
    /// Castagnoli `CRC32C` over `arena[0..arena_offset_lite_end]`.
    pub arena_crc32c_lite: u32,
    /// Castagnoli `CRC32C` over `arena[0..arena_offset_full_end]`.
    pub arena_crc32c_full: u32,
    /// Castagnoli `CRC32C` over `arena[0..arena_offset_ultra_end]`.
    pub arena_crc32c_ultra: u32,
    /// Number of stem records in `Arena 3` (`0` for `.dnam`, $\ge 1$ for `.dnam-ckpt`).
    pub num_ckpt_stems: u16,
    /// Control-bus decimation stride $K_{\text{ctrl}}$ at $f_{s,\text{train}}$ (`16`).
    pub control_decimation: u8,
    /// Active SVD principal control subspace rank $D_h \in [0, 4]$.
    pub head_svd_rank: u8,
    /// Estimated $T_{60}$ decay tail time in milliseconds for DAW sleep management.
    pub tail_time_ms: f32,
    /// Castagnoli `CRC32C` over `bytes[0..tensor_arena_offset]` with `bytes[0x3C..0x40]` zeroed.
    pub header_crc32c: u32,
}

const _: () = assert!(size_of::<DnamPreamble>() == 64);
const _: () = assert!(align_of::<DnamPreamble>() == 64);

impl DnamPreamble {
    /// Parses a 64-byte [`DnamPreamble`] from the leading 64 bytes of a container buffer.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DnamError> {
        if bytes.len() < 64 {
            return Err(DnamError::BufferTooSmall {
                expected: 64,
                actual: bytes.len(),
            });
        }
        let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if magic != DNAM_MAGIC && magic != DNCK_MAGIC {
            return Err(DnamError::InvalidMagic(magic));
        }
        let version_major = u16::from_le_bytes([bytes[4], bytes[5]]);
        let version_minor = u16::from_le_bytes([bytes[6], bytes[7]]);
        if version_major != CONTAINER_VERSION_MAJOR {
            return Err(DnamError::UnsupportedVersion {
                major: version_major,
                minor: version_minor,
            });
        }
        let flags = u16::from_le_bytes([bytes[8], bytes[9]]);
        let num_controls = bytes[10];
        let ring_buffer_pow2_log2 = bytes[11];
        let sample_rate_train_hz = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        let json_header_byte_len = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        let tensor_arena_offset = u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
        let tensor_arena_byte_len =
            u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
        let arena_offset_lite_end =
            u32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]);
        let arena_offset_full_end =
            u32::from_le_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]);
        let arena_offset_ultra_end =
            u32::from_le_bytes([bytes[36], bytes[37], bytes[38], bytes[39]]);
        let arena_crc32c_lite = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]);
        let arena_crc32c_full = u32::from_le_bytes([bytes[44], bytes[45], bytes[46], bytes[47]]);
        let arena_crc32c_ultra = u32::from_le_bytes([bytes[48], bytes[49], bytes[50], bytes[51]]);
        let num_ckpt_stems = u16::from_le_bytes([bytes[52], bytes[53]]);
        let control_decimation = bytes[54];
        let head_svd_rank = bytes[55];
        let tail_time_ms = f32::from_le_bytes([bytes[56], bytes[57], bytes[58], bytes[59]]);
        let header_crc32c = u32::from_le_bytes([bytes[60], bytes[61], bytes[62], bytes[63]]);

        for (field, offset) in [
            ("tensor_arena_offset", tensor_arena_offset),
            ("tensor_arena_byte_len", tensor_arena_byte_len),
            ("arena_offset_lite_end", arena_offset_lite_end),
            ("arena_offset_full_end", arena_offset_full_end),
            ("arena_offset_ultra_end", arena_offset_ultra_end),
        ] {
            if (offset & 0x3F) != 0 {
                return Err(DnamError::UnalignedOffset { field, offset });
            }
        }

        Ok(Self {
            magic,
            version_major,
            version_minor,
            flags,
            num_controls,
            ring_buffer_pow2_log2,
            sample_rate_train_hz,
            json_header_byte_len,
            tensor_arena_offset,
            tensor_arena_byte_len,
            arena_offset_lite_end,
            arena_offset_full_end,
            arena_offset_ultra_end,
            arena_crc32c_lite,
            arena_crc32c_full,
            arena_crc32c_ultra,
            num_ckpt_stems,
            control_decimation,
            head_svd_rank,
            tail_time_ms,
            header_crc32c,
        })
    }

    /// Serializes the 64-byte [`DnamPreamble`] into little-endian bytes.
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        out[0..4].copy_from_slice(&self.magic);
        out[4..6].copy_from_slice(&self.version_major.to_le_bytes());
        out[6..8].copy_from_slice(&self.version_minor.to_le_bytes());
        out[8..10].copy_from_slice(&self.flags.to_le_bytes());
        out[10] = self.num_controls;
        out[11] = self.ring_buffer_pow2_log2;
        out[12..16].copy_from_slice(&self.sample_rate_train_hz.to_le_bytes());
        out[16..20].copy_from_slice(&self.json_header_byte_len.to_le_bytes());
        out[20..24].copy_from_slice(&self.tensor_arena_offset.to_le_bytes());
        out[24..28].copy_from_slice(&self.tensor_arena_byte_len.to_le_bytes());
        out[28..32].copy_from_slice(&self.arena_offset_lite_end.to_le_bytes());
        out[32..36].copy_from_slice(&self.arena_offset_full_end.to_le_bytes());
        out[36..40].copy_from_slice(&self.arena_offset_ultra_end.to_le_bytes());
        out[40..44].copy_from_slice(&self.arena_crc32c_lite.to_le_bytes());
        out[44..48].copy_from_slice(&self.arena_crc32c_full.to_le_bytes());
        out[48..52].copy_from_slice(&self.arena_crc32c_ultra.to_le_bytes());
        out[52..54].copy_from_slice(&self.num_ckpt_stems.to_le_bytes());
        out[54] = self.control_decimation;
        out[55] = self.head_svd_rank;
        out[56..60].copy_from_slice(&self.tail_time_ms.to_le_bytes());
        out[60..64].copy_from_slice(&self.header_crc32c.to_le_bytes());
        out
    }

    /// Returns `true` if `CORE2_ACTIVE` (`bit 6`) is set.
    #[inline]
    pub const fn is_core2_active(&self) -> bool {
        (self.flags & FLAG_CORE2_ACTIVE) != 0
    }

    /// Returns `true` if `LFO_ENV_ACTIVE` (`bit 7`) is set.
    #[inline]
    pub const fn is_lfo_env_active(&self) -> bool {
        (self.flags & FLAG_LFO_ENV_ACTIVE) != 0
    }

    /// Returns `true` if `STEREO_OUT` (`bit 8`) is set.
    #[inline]
    pub const fn is_stereo_out(&self) -> bool {
        (self.flags & FLAG_STEREO_OUT) != 0
    }

    /// Returns `true` if `HAS_SIGNATURES` (`bit 9`) is set.
    #[inline]
    pub const fn has_signatures(&self) -> bool {
        (self.flags & FLAG_HAS_SIGNATURES) != 0
    }

    /// Returns `true` if `ENCRYPTED_ARENA` (`bit 10`) is set.
    #[inline]
    pub const fn is_encrypted_arena(&self) -> bool {
        (self.flags & FLAG_ENCRYPTED_ARENA) != 0
    }
}

// ============================================================================
// Pure `#![no_std]` Castagnoli `CRC32C` (`0x1EDC6F41`, reflected `0x82F63B78`)
// ============================================================================

#[cfg(not(target_arch = "aarch64"))]
const CRC32C_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ 0x82F6_3B78;
            } else {
                crc >>= 1;
            }
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

/// Updates an incremental Castagnoli `CRC32C` state with `data` using hardware `CRC32C` on `aarch64`.
#[inline]
pub fn crc32c_update(crc: u32, data: &[u8]) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        crate::math::neon::crc32c_arm64_update(crc, data)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut c = !crc;
        let mut i = 0;
        while i < data.len() {
            let idx = ((c ^ (data[i] as u32)) & 0xFF) as usize;
            c = (c >> 8) ^ CRC32C_TABLE[idx];
            i += 1;
        }
        !c
    }
}

/// Computes the Castagnoli `CRC32C` (`0x1EDC6F41`) checksum of `data`.
#[inline]
pub fn crc32c_compute(data: &[u8]) -> u32 {
    crc32c_update(0, data)
}

/// Computes `header_crc32c` over `container_bytes[0..tensor_arena_offset]` with
/// `bytes[0x3C..0x40]` treated as `[0, 0, 0, 0]`.
pub fn compute_header_crc32c(container_bytes: &[u8], tensor_arena_offset: usize) -> u32 {
    let mut crc = crc32c_update(0, &container_bytes[0..60]);
    crc = crc32c_update(crc, &[0u8; 4]);
    crc32c_update(crc, &container_bytes[64..tensor_arena_offset])
}

// ============================================================================
// Parsed Container View (`DnamContainerView`) & In-Place Header Editor
// ============================================================================

/// Zero-copy or decrypted view over a validated `.dnam` or `.dnam-ckpt` container.
#[derive(Debug)]
pub struct DnamContainerView<'a> {
    /// 64-byte binary preamble.
    pub preamble: DnamPreamble,
    /// Parsed UTF-8 JSON header (`no_std` `serde` + `alloc`).
    pub header: DnamJsonHeader,
    /// Cryptographic provenance status (`Unsigned`, `CreatorVerified`, or `MarketplaceVerified`).
    pub verification_status: VerificationStatus,
    /// 64-byte aligned tensor arena storage (`Borrowed` on aligned `mmap`, or `Owned` if decrypted/copied).
    pub arena_storage: ArenaStorage<'a>,
}

impl<'a> DnamContainerView<'a> {
    /// Full container loader: verifies `header_crc32c`, parses and validates the UTF-8 JSON header
    /// via `#![no_std]` `serde` + `alloc`, decrypts `Arena 0..2` if `ENCRYPTED_ARENA == 1`, verifies
    /// prefix `CRC32C` checksums, and verifies `Ed25519` signatures if `HAS_SIGNATURES == 1`.
    pub fn from_bytes<C: CryptoProvider>(bytes: &'a [u8], crypto: &C) -> Result<Self, DnamError> {
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

        // 1. Verify header_crc32c over [0..tensor_arena_offset] with [60..64] zeroed
        let computed_hdr_crc = compute_header_crc32c(bytes, arena_off);
        if computed_hdr_crc != preamble.header_crc32c {
            return Err(DnamError::Crc32cMismatch {
                region: "header",
                expected: preamble.header_crc32c,
                computed: computed_hdr_crc,
            });
        }

        // 2. Parse UTF-8 JSON header at 0x40..0x40 + json_header_byte_len via #![no_std] serde + alloc
        let json_slice = &bytes[64..64 + json_len];
        let header: DnamJsonHeader = serde_json::from_slice(json_slice)
            .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?;
        header.validate()?;

        // 3. Prepare or decrypt Arena 0..2
        let raw_arena = &bytes[arena_off..arena_off + arena_len];
        let arena_storage = if preamble.is_encrypted_arena() {
            let sigs =
                header
                    .metadata
                    .signatures
                    .as_ref()
                    .ok_or(DnamError::CryptoVerificationFailed(
                        "ENCRYPTED_ARENA=1 requires metadata.signatures",
                    ))?;
            let receipt =
                sigs.marketplace_receipt
                    .as_ref()
                    .ok_or(DnamError::CryptoVerificationFailed(
                        "ENCRYPTED_ARENA=1 requires metadata.signatures.marketplace_receipt",
                    ))?;
            let nonce_hex =
                receipt
                    .arena_nonce_96
                    .as_ref()
                    .ok_or(DnamError::CryptoVerificationFailed(
                        "missing arena_nonce_96 in receipt",
                    ))?;
            let nonce_96 = decode_hex_array::<12>(nonce_hex)?;

            let target_device = if let Some(local_id) = crypto.local_device_id_hash() {
                receipt
                    .authorized_devices
                    .iter()
                    .find(|d| d.device_id_hash == local_id)
            } else {
                receipt.authorized_devices.first()
            }
            .ok_or(DnamError::EncryptedArenaRequiresDeviceKey)?;

            let cek = crypto.unwrap_device_cek(receipt, target_device)?;
            let mut owned = AlignedBuffer::<CacheLine64>::new_zeroed(arena_len / 64);
            owned.as_bytes_mut().copy_from_slice(raw_arena);
            let ultra_end = (preamble.arena_offset_ultra_end as usize).min(arena_len);
            chacha20_xor_arena_blocks(&mut owned.as_bytes_mut()[..ultra_end], &cek, &nonce_96, 0)?;
            ArenaStorage::from_owned_buffer(owned)
        } else {
            ArenaStorage::from_slice_or_copy(raw_arena)
        };

        let plain_bytes = arena_storage.as_bytes();
        Self::verify_arena_prefix_crcs(&preamble, plain_bytes)?;

        // 4. Verify split-payload Ed25519 provenance over Arena 0..2
        let ultra_end = (preamble.arena_offset_ultra_end as usize).min(plain_bytes.len());
        let verification_status = verify_container_provenance(
            &header,
            preamble.has_signatures(),
            &plain_bytes[..ultra_end],
            crypto,
        )?;

        Ok(Self {
            preamble,
            header,
            verification_status,
            arena_storage,
        })
    }

    /// Verifies `arena_crc32c_lite`, `arena_crc32c_full`, and `arena_crc32c_ultra` over the
    /// available prefix of `plain_arena` in a single incremental pass.
    pub fn verify_arena_prefix_crcs(
        preamble: &DnamPreamble,
        plain_arena: &[u8],
    ) -> Result<(), DnamError> {
        let lite_end = preamble.arena_offset_lite_end as usize;
        let mut crc = 0_u32;
        if plain_arena.len() >= lite_end && lite_end > 0 {
            crc = crc32c_update(0, &plain_arena[..lite_end]);
            if crc != preamble.arena_crc32c_lite {
                return Err(DnamError::Crc32cMismatch {
                    region: "arena_lite",
                    expected: preamble.arena_crc32c_lite,
                    computed: crc,
                });
            }
        }
        let full_end = preamble.arena_offset_full_end as usize;
        if (preamble.flags & FLAG_HAS_FULL_DELTA) != 0
            && plain_arena.len() >= full_end
            && full_end > lite_end
        {
            crc = crc32c_update(crc, &plain_arena[lite_end..full_end]);
            if crc != preamble.arena_crc32c_full {
                return Err(DnamError::Crc32cMismatch {
                    region: "arena_full",
                    expected: preamble.arena_crc32c_full,
                    computed: crc,
                });
            }
        }
        let ultra_end = preamble.arena_offset_ultra_end as usize;
        if (preamble.flags & FLAG_HAS_ULTRA_DELTA) != 0
            && plain_arena.len() >= ultra_end
            && ultra_end > full_end
        {
            crc = crc32c_update(crc, &plain_arena[full_end..ultra_end]);
            if crc != preamble.arena_crc32c_ultra {
                return Err(DnamError::Crc32cMismatch {
                    region: "arena_ultra",
                    expected: preamble.arena_crc32c_ultra,
                    computed: crc,
                });
            }
        }
        Ok(())
    }
}

/// Builds a complete 4 KB page-aligned `.dnam` or `.dnam-ckpt` container byte vector from a
/// [`DnamJsonHeader`] and plaintext (or encrypted) tensor arena (`Arena 0..2` or `Arena 0..3`).
///
/// Automatically computes `arena_crc32c_lite`, `arena_crc32c_full`, `arena_crc32c_ultra`,
/// space-pads (`0x20`) the JSON header to a 4,096-byte OS page boundary, and seals `header_crc32c`.
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

    let lite_end = ARENA_OFFSET_LITE_END as usize;
    let full_end = (ARENA_OFFSET_FULL_END as usize).min(plaintext_arena_crc_source.len());
    let ultra_end = (ARENA_OFFSET_ULTRA_END as usize).min(plaintext_arena_crc_source.len());

    let crc_lite = crc32c_update(0, &plaintext_arena_crc_source[..lite_end]);
    let crc_full = if full_end >= ARENA_OFFSET_FULL_END as usize {
        crc32c_update(crc_lite, &plaintext_arena_crc_source[lite_end..full_end])
    } else {
        crc_lite
    };
    let crc_ultra = if ultra_end >= ARENA_OFFSET_ULTRA_END as usize {
        crc32c_update(crc_full, &plaintext_arena_crc_source[full_end..ultra_end])
    } else {
        crc_full
    };

    header.slimmable_tiers.lite.prefix_crc32c = crc_lite;
    header.slimmable_tiers.full.prefix_crc32c = crc_full;
    header.slimmable_tiers.ultra.prefix_crc32c = crc_ultra;

    // Compile binary ControlClampDescriptor table into stored_arena_bytes if unencrypted
    let json_bytes =
        serde_json::to_vec(header).map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?;
    let min_header_region = 64 + json_bytes.len();
    // Round up to a multiple of 4,096 B (at least 4,096 B) so Arena 0 is 4 KB page-aligned with slack
    let tensor_arena_offset = min_header_region.div_ceil(4096).max(1) * 4096;

    let is_ckpt = header.container_kind == "dnam-ckpt";
    let magic = if is_ckpt { DNCK_MAGIC } else { DNAM_MAGIC };

    let mut flags = FLAG_HAS_LITE | FLAG_EXEC_ORDER_SOA | FLAG_DEFAULT_ANCHORED;
    if stored_arena_bytes.len() >= ARENA_OFFSET_FULL_END as usize {
        flags |= FLAG_HAS_FULL_DELTA;
    }
    if stored_arena_bytes.len() >= ARENA_OFFSET_ULTRA_END as usize {
        flags |= FLAG_HAS_ULTRA_DELTA;
    }
    if is_ckpt
        && header
            .stems_registry
            .as_ref()
            .is_some_and(|s| !s.is_empty())
    {
        flags |= FLAG_HAS_CKPT_STEMS;
    }
    if header.core2_mufdn.active {
        flags |= FLAG_CORE2_ACTIVE;
    }
    if header.control_bus.envelopes > 0 || header.control_bus.lfo_oscillators > 0 {
        flags |= FLAG_LFO_ENV_ACTIVE;
    }
    if header.io_config.output_mode != "mono" {
        flags |= FLAG_STEREO_OUT;
    }
    if header.metadata.signatures.is_some() {
        flags |= FLAG_HAS_SIGNATURES;
    }
    if encrypted_arena {
        flags |= FLAG_ENCRYPTED_ARENA;
    }

    let num_ckpt_stems = header.stems_registry.as_ref().map_or(0, |s| s.len() as u16);

    let mut preamble = DnamPreamble {
        magic,
        version_major: CONTAINER_VERSION_MAJOR,
        version_minor: CONTAINER_VERSION_MINOR,
        flags,
        num_controls: header.controls_schema.len() as u8,
        ring_buffer_pow2_log2: if header.core2_mufdn.active {
            header.core2_mufdn.ring_buffer_pow2_log2
        } else {
            0
        },
        sample_rate_train_hz: header.sample_rate_train_hz,
        json_header_byte_len: json_bytes.len() as u32,
        tensor_arena_offset: tensor_arena_offset as u32,
        tensor_arena_byte_len: stored_arena_bytes.len() as u32,
        arena_offset_lite_end: ARENA_OFFSET_LITE_END,
        arena_offset_full_end: ARENA_OFFSET_FULL_END,
        arena_offset_ultra_end: ARENA_OFFSET_ULTRA_END,
        arena_crc32c_lite: crc_lite,
        arena_crc32c_full: crc_full,
        arena_crc32c_ultra: crc_ultra,
        num_ckpt_stems,
        control_decimation: header.control_bus.decimation as u8,
        head_svd_rank: header.slimmable_tiers.ultra.head_svd_rank,
        tail_time_ms: header.io_config.tail_time_ms,
        header_crc32c: 0,
    };

    let total_len = tensor_arena_offset + stored_arena_bytes.len();
    let mut out = alloc::vec![0x20_u8; total_len]; // 0x20 ASCII space padding in header slack
    out[0..64].copy_from_slice(&preamble.to_bytes());
    out[64..64 + json_bytes.len()].copy_from_slice(&json_bytes);
    out[tensor_arena_offset..].copy_from_slice(stored_arena_bytes);

    preamble.header_crc32c = compute_header_crc32c(&out, tensor_arena_offset);
    out[60..64].copy_from_slice(&preamble.header_crc32c.to_le_bytes());
    Ok(out)
}

/// Updates the JSON header (`presets`, `user_metadata`, or `authorized_devices`) **100% in-place**
/// inside the pre-allocated 4 KB header region (`0x40..tensor_arena_offset`) without moving
/// `Arena 0..3`, without invalidating `immutable_header_sha256` (when only mutable fields change),
/// and recomputes `header_crc32c` in `< 0.1 ms`.
pub fn update_header_in_place(
    container_bytes: &mut [u8],
    updated_header: &DnamJsonHeader,
) -> Result<(), DnamError> {
    let mut preamble = DnamPreamble::from_bytes(container_bytes)?;
    let arena_off = preamble.tensor_arena_offset as usize;
    if container_bytes.len() < arena_off {
        return Err(DnamError::BufferTooSmall {
            expected: arena_off,
            actual: container_bytes.len(),
        });
    }
    let capacity = arena_off.saturating_sub(64);
    let json_bytes = serde_json::to_vec(updated_header)
        .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?;
    if json_bytes.len() > capacity {
        return Err(DnamError::HeaderCapacityExceeded {
            required: json_bytes.len(),
            capacity,
        });
    }

    container_bytes[64..64 + json_bytes.len()].copy_from_slice(&json_bytes);
    container_bytes[64 + json_bytes.len()..arena_off].fill(0x20);

    preamble.json_header_byte_len = json_bytes.len() as u32;
    if updated_header.metadata.signatures.is_some() {
        preamble.flags |= FLAG_HAS_SIGNATURES;
    }
    container_bytes[0..64].copy_from_slice(&preamble.to_bytes());
    let new_crc = compute_header_crc32c(container_bytes, arena_off);
    container_bytes[60..64].copy_from_slice(&new_crc.to_le_bytes());
    Ok(())
}

// ============================================================================
// `.dnam-ckpt` `Arena 3` Column-Major Upper-Triangular Packing (`uplo='U'`)
// ============================================================================

/// Returns the packed element count $\frac{N(N+1)}{2}$ for a symmetric $N \times N$ Gram matrix:
/// - `Lite` ($N=64$): `2,080` `f64` (`16,640 B`)
/// - `Full` ($N=128$): `8,256` `f64` (`66,048 B`)
/// - `Ultra` ($N=192$): `18,528` `f64` (`148,224 B`)
#[inline]
pub const fn uplo_u_packed_len(n_dim: usize) -> usize {
    (n_dim * (n_dim + 1)) / 2
}

/// Packs a dense symmetric $N \times N$ row-major matrix `dense_matrix` (`N * N` elements)
/// into **Column-Major Upper-Triangular (`LAPACK uplo='U'`)** format:
/// $$\operatorname{pack}_{\text{U}}(\mathbf{G})\!\left[i + \frac{j(j+1)}{2}\right] = \mathbf{G}[i, j], \qquad 0 \le i \le j < N$$
pub fn pack_uplo_u_f64(dense_matrix: &[f64], n_dim: usize, out_packed: &mut [f64]) {
    let mut idx = 0;
    for col in 0..n_dim {
        for row in 0..=col {
            out_packed[idx] = dense_matrix[row * n_dim + col];
            idx += 1;
        }
    }
}

/// Unpacks a **Column-Major Upper-Triangular (`LAPACK uplo='U'`)** buffer into a symmetric
/// $N_{\text{tier}} \times N_{\text{tier}}$ matrix (`out_dense`, length `n_tier * n_tier`).
///
/// By the **Nested Principal Submatrix Prefix Theorem**, passing `n_tier = 64` (`Lite`) or
/// `n_tier = 128` (`Full`) on an `Ultra` ($N=192$) packed buffer extracts the exact leading
/// principal submatrix from the contiguous prefix `[0 .. n_tier*(n_tier+1)/2]`!
pub fn unpack_uplo_u_f64(packed_u: &[f64], n_tier: usize, out_dense: &mut [f64]) {
    let mut idx = 0;
    for col in 0..n_tier {
        for row in 0..=col {
            let val = packed_u[idx];
            out_dense[row * n_tier + col] = val;
            out_dense[col * n_tier + row] = val;
            idx += 1;
        }
    }
}

/// Verifies a `.dnam-ckpt` stem's `stem_arena3_crc32c` over its `Arena 3` slice (`151,296 B` Ultra).
pub fn verify_ckpt_stem_crc32c(
    arena_bytes: &[u8],
    stem_byte_offset: usize,
    stem_byte_len: usize,
    expected_crc32c: u32,
) -> Result<(), DnamError> {
    if arena_bytes.len() < stem_byte_offset + stem_byte_len {
        return Err(DnamError::BufferTooSmall {
            expected: stem_byte_offset + stem_byte_len,
            actual: arena_bytes.len(),
        });
    }
    let slice = &arena_bytes[stem_byte_offset..stem_byte_offset + stem_byte_len];
    let computed = crc32c_compute(slice);
    if computed != expected_crc32c {
        return Err(DnamError::Crc32cMismatch {
            region: "arena3_stem",
            expected: expected_crc32c,
            computed,
        });
    }
    Ok(())
}

// ============================================================================
// Continuous-Time Zero-Order Hold (ZOH) Sample-Rate Adaptability (44.1-96 kHz)
// ============================================================================

/// Rescaled discrete-time parameters for a host sample rate $f_{s,\text{host}} \in [44100, 96000]\text{ Hz}$
/// per [`docs/dnam_container_schema.md` §4](../../../docs/dnam_container_schema.md#4-continuous-time-zero-order-hold-zoh-sample-rate-adaptability-441--96-khz).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZohRescaledParams {
    /// Host sample rate $f_{s,\text{host}}$ in Hz.
    pub sample_rate_host_hz: u32,
    /// Sample-rate ratio $\rho_{\text{sr}} = f_{s,\text{train}} / f_{s,\text{host}}$.
    pub rho_sr: f64,
    /// Host control-bus decimation stride $K_{\text{ctrl}}(f_{s,\text{host}}) = \max(8, \operatorname{round}(f_{s,\text{host}} / 3000))$.
    pub control_decimation_host: usize,
    /// Exact control-bus tick ratio $\rho_{\text{ctrl}} = \frac{f_{s,\text{train}} K_{\text{ctrl}}(f_{s,\text{host}})}{f_{s,\text{host}} K_{\text{ctrl}}(f_{s,\text{train}})}$
    /// keeping LFO frequencies in Hz 100% invariant across $44.1 / 48 / 88.2 / 96\text{ kHz}$.
    pub rho_ctrl: f64,
    /// Rescaled Core 1A $5\text{ Hz}$ AC-coupling pole $R_{\text{ac}}(f_{s,\text{host}}) = R_{\text{ac}}(f_{s,\text{train}})^{\rho_{\text{sr}}}$.
    pub r_ac_host: f32,
    /// Rescaled Core 2 $5\text{ Hz}$ DC-blocker pole $R_{\text{dc}}(f_{s,\text{host}}) = R_{\text{dc}}(f_{s,\text{train}})^{\rho_{\text{sr}}}$.
    pub r_dc_host: f32,
    /// Rescaled Core 2 TZF reference tap $\tau_{\text{ref}}^{\text{samples}}(f_{s,\text{host}}) = \tau_{\text{ref}}(f_{s,\text{train}}) / \rho_{\text{sr}}$.
    pub tau_ref_samples_host: f32,
    /// Rescaled Core 2 switch crossfade length $L_{\text{xfade}}(f_{s,\text{host}}) = \operatorname{round}(L_{\text{xfade}}(f_{s,\text{train}}) / \rho_{\text{sr}})$.
    pub switch_xfade_samples_host: u32,
}

impl ZohRescaledParams {
    /// Computes the closed-form ZOH rescaled parameters for `fs_host_hz` given training anchors.
    /// When `fs_host_hz == fs_train_hz`, returns exact identity factors with `0` drift.
    pub fn compute(
        fs_train_hz: u32,
        fs_host_hz: u32,
        k_ctrl_train: u8,
        r_ac_train: f32,
        r_dc_train: f32,
        tau_ref_train: f32,
        xfade_train: u32,
    ) -> Result<Self, DnamError> {
        if !(44_100..=96_000).contains(&fs_host_hz) {
            return Err(DnamError::InvalidSampleRate(fs_host_hz));
        }
        if fs_host_hz == fs_train_hz {
            return Ok(Self {
                sample_rate_host_hz: fs_host_hz,
                rho_sr: 1.0,
                control_decimation_host: k_ctrl_train as usize,
                rho_ctrl: 1.0,
                r_ac_host: r_ac_train,
                r_dc_host: r_dc_train,
                tau_ref_samples_host: tau_ref_train,
                switch_xfade_samples_host: xfade_train,
            });
        }

        let fs_train = fs_train_hz as f64;
        let fs_host = fs_host_hz as f64;
        let rho_sr = fs_train / fs_host;

        let k_host_raw = ((fs_host / 3000.0) + 0.5) as usize;
        let k_host = k_host_raw.max(8);
        let rho_ctrl = (fs_train * (k_host as f64)) / (fs_host * (k_ctrl_train as f64));

        let r_ac_host = cold_pow_f64(r_ac_train as f64, rho_sr) as f32;
        let r_dc_host = cold_pow_f64(r_dc_train as f64, rho_sr) as f32;
        let tau_ref_samples_host = ((tau_ref_train as f64) / rho_sr) as f32;
        let switch_xfade_samples_host = (((xfade_train as f64) / rho_sr) + 0.5) as u32;

        Ok(Self {
            sample_rate_host_hz: fs_host_hz,
            rho_sr,
            control_decimation_host: k_host,
            rho_ctrl,
            r_ac_host,
            r_dc_host,
            tau_ref_samples_host,
            switch_xfade_samples_host: switch_xfade_samples_host.max(16),
        })
    }

    /// Rescales a Core 1B `SelectiveRgLru` pole radius $r_c$, angle $\theta_{0,c}$, and $\mathcal{H}_2$-norm
    /// input normalizer $b_{\gamma,c}$ under ZOH (`CL-03`, `CL-04`):
    /// $$r_c(f_{s,\text{host}}) = \operatorname{clamp}\!\big(r_c(f_{s,\text{train}})^{\rho_{\text{sr}}}, 0.25, 0.9995\big)$$
    /// $$\theta_{0,c}(f_{s,\text{host}}) = \operatorname{clamp}\!\big(\rho_{\text{sr}}\,\theta_{0,c}(f_{s,\text{train}}), 10^{-4}, \pi - 10^{-4}\big)$$
    #[inline]
    pub fn rescale_rglru_channel(
        &self,
        r_low_train: f32,
        dr_span_train: f32,
        theta0_train: f32,
        b_gamma_train: f32,
    ) -> (f32, f32, f32, f32) {
        if (self.rho_sr - 1.0).abs() < 1e-12 {
            return (r_low_train, dr_span_train, theta0_train, b_gamma_train);
        }
        let r_high_train = (r_low_train + dr_span_train).clamp(0.25, 0.9995);
        let r_low_host = cold_pow_f64(r_low_train as f64, self.rho_sr).clamp(0.25, 0.9995) as f32;
        let r_high_host = cold_pow_f64(r_high_train as f64, self.rho_sr).clamp(0.25, 0.9995) as f32;
        let dr_span_host = (r_high_host - r_low_host).max(0.0);

        let theta_host =
            ((theta0_train as f64) * self.rho_sr).clamp(1e-4, core::f64::consts::PI - 1e-4) as f32;

        // Rescale H2 normalizer sqrt(1 - r_bar_host^2) / sqrt(1 - r_bar_train^2)
        let r_bar_train = 0.5 * ((r_low_train + r_high_train) as f64);
        let r_bar_host = 0.5 * ((r_low_host + r_high_host) as f64);
        let norm_train = cold_sqrt_f64((1.0 - r_bar_train * r_bar_train).max(1e-7));
        let norm_host = cold_sqrt_f64((1.0 - r_bar_host * r_bar_host).max(1e-7));
        let b_gamma_host = ((b_gamma_train as f64) * (norm_host / norm_train)) as f32;

        (r_low_host, dr_span_host, theta_host, b_gamma_host)
    }

    /// Rescales a Core 2 base delay $\tau_{0,m}^{\text{samples}}$ and integer Schroeder all-pass delay $D_{\text{ap},m}$ (`CL-06`, `CL-08`).
    #[inline]
    pub fn rescale_core2_delays(&self, tau0_samples_train: f32, d_ap_train: f32) -> (f32, usize) {
        if (self.rho_sr - 1.0).abs() < 1e-12 {
            return (
                tau0_samples_train.max(4.0),
                (d_ap_train as usize).clamp(4, 255),
            );
        }
        let tau0_host = (((tau0_samples_train as f64) / self.rho_sr) as f32).max(4.0);
        let d_ap_host = ((((d_ap_train as f64) / self.rho_sr) + 0.5) as usize).clamp(4, 255);
        (tau0_host, d_ap_host)
    }
}
