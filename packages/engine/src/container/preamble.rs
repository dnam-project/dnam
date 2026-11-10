//! Canonical v2.0 64-byte binary preamble & 128-byte region directory view (`DnamPreamble`).

use core::mem::{align_of, size_of};

use crate::control::ControlClampDescriptor;
use crate::schema::{DnamError, SchemaOptionExt, reject_if};
use crate::types::Tier;

use super::wire::{
    CP1_MAGIC, DEFAULT_MAX_MUTABLE_CAPACITY, DMU1_HEADER_SIZE, DMUE_FOOTER_SIZE, V2_HEADER_OFFSET,
    V2_PREAMBLE_SIZE, align64_u64, parse_dcrc_record,
};
use super::{
    CONTAINER_VERSION_MAJOR, CONTAINER_VERSION_MINOR, DNAM_MAGIC, DNCK_MAGIC, FLAG_ENCRYPTED_ARENA,
    FLAG_HAS_CALIB_STATS, FLAG_HAS_CKPT_STEMS, FLAG_HAS_FULL_DELTA, FLAG_HAS_LITE,
    FLAG_HAS_PRO_DELTA, FLAG_HAS_SIGNATURES, FLAG_HAS_ULTRA_DELTA, FLAG_STEREO_OUT,
};

/// Minimum byte length of a valid v2.0 `.dnam` or `.dnam-ckpt` container (`192 + 18 + 16 = 226 B`).
const MIN_V2_CONTAINER_BYTES: usize = V2_HEADER_OFFSET + DMU1_HEADER_SIZE + 2 + DMUE_FOOTER_SIZE;

/// Fixed 64-byte logical preamble descriptor (`#[repr(C, align(64))]`) summarizing a validated
/// v2.0 `.dnam` or `.dnam-ckpt` container without heap allocation.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DnamPreamble {
    /// `b"DNAM"` (`0x4D414E44`) or `b"DNCK"` (`0x4B434E44`).
    pub magic: [u8; 4],
    /// Container major version (`2`).
    pub version_major: u16,
    /// Container minor version (`0`).
    pub version_minor: u16,
    /// Bitmask of `FLAG_*` constants.
    pub flags: u16,
    /// Total user controls $P \in [0, 65535]$.
    pub num_controls: u16,
    /// Reference training sample rate in Hz (`48000`).
    pub sample_rate_train_hz: u32,
    /// Unpadded UTF-8 byte length of Region 3 JCS JSON header starting at `0x00C0` (`192`).
    pub json_header_byte_len: u32,
    /// File byte offset of `tier_start` / `bundle_start` (`% 64 == 0`).
    pub tensor_arena_offset: u32,
    /// Total retained byte length of tier slices / checkpoint bundles (`% 64 == 0`).
    pub tensor_arena_byte_len: u32,
    /// Arena-relative end of `Lite` (`tier 0`) in 64B cache lines (`168` $\to$ `10,752 B`).
    pub arena_lite_end_cl: u16,
    /// Arena-relative end of `Full` (`tier 0..=1`) in 64B cache lines (`349` $\to$ `22,336 B`).
    pub arena_full_end_cl: u16,
    /// Arena-relative end of `Pro` (`tier 0..=2`) in 64B cache lines (`566` $\to$ `36,224 B`).
    pub arena_pro_end_cl: u16,
    /// Arena-relative end of `Ultra` (`tier 0..=3`) in 64B cache lines (`1,000` $\to$ `64,000 B`).
    pub arena_ultra_end_cl: u16,
    /// Castagnoli `CRC32C` over `arena[0..arena_offset_lite_end]`.
    pub arena_crc32c_lite: u32,
    /// Castagnoli `CRC32C` over `arena[0..arena_offset_full_end]`.
    pub arena_crc32c_full: u32,
    /// Castagnoli `CRC32C` over `arena[0..arena_offset_pro_end]`.
    pub arena_crc32c_pro: u32,
    /// Castagnoli `CRC32C` over `arena[0..arena_offset_ultra_end]`.
    pub arena_crc32c_ultra: u32,
    /// Number of stem records in `Arena 4` (`0` for `.dnam`, $\ge 1$ for `.dnam-ckpt`).
    pub num_ckpt_stems: u16,
    /// Reserved 16-bit alignment gap (`0`).
    pub _reserved: u16,
    /// Estimated $T_{60}$ decay tail time in milliseconds for DAW sleep management.
    pub tail_time_ms: f32,
    /// Castagnoli `CRC32C` over `bytes[0..auth_offset]` when a `DCRC` record is present.
    pub header_crc32c: u32,
}

const _: () = assert!(size_of::<DnamPreamble>() == 64);
const _: () = assert!(align_of::<DnamPreamble>() == 64);

fn scan_shared_control_count(shared_slice: &[u8]) -> u16 {
    if shared_slice.len() >= 4 && shared_slice[0..4] == CP1_MAGIC {
        return 0;
    }
    let mut count = 0usize;
    let mut off = 0usize;
    while off + 64 <= shared_slice.len() {
        if shared_slice[off..off + 4] == CP1_MAGIC {
            break;
        }
        let mut chunk = [0u8; 64];
        chunk.copy_from_slice(&shared_slice[off..off + 64]);
        let desc = ControlClampDescriptor::from_bytes(&chunk);
        let populated = desc.ctrl_kind <= 1
            && (desc.safe_max_base > desc.safe_min_base
                || desc.scale_norm != 0.0
                || desc.active_options_mask != 0
                || desc.lobatto_degree > 0);
        if !populated {
            break;
        }
        count += 1;
        off += 64;
    }
    count.min(u16::MAX as usize) as u16
}

impl DnamPreamble {
    /// Parses a [`DnamPreamble`] from a canonical `v2.0` `.dnam` or `.dnam-ckpt` container buffer
    /// without heap allocation.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DnamError> {
        if bytes.len() < V2_PREAMBLE_SIZE {
            return Err(DnamError::BufferTooSmall {
                expected: V2_PREAMBLE_SIZE,
                actual: bytes.len(),
            });
        }
        let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if magic != DNAM_MAGIC && magic != DNCK_MAGIC {
            return Err(DnamError::InvalidMagic(magic));
        }
        let version_major = u16::from_le_bytes([bytes[4], bytes[5]]);
        let version_minor = u16::from_le_bytes([bytes[6], bytes[7]]);
        if version_major != CONTAINER_VERSION_MAJOR || version_minor != CONTAINER_VERSION_MINOR {
            return Err(DnamError::UnsupportedVersion {
                major: version_major,
                minor: version_minor,
            });
        }
        if bytes.len() < MIN_V2_CONTAINER_BYTES {
            return Err(DnamError::BufferTooSmall {
                expected: MIN_V2_CONTAINER_BYTES,
                actual: bytes.len(),
            });
        }

        let profile = bytes[8];
        let auth_kind = bytes[9];
        let maximum = bytes[10];
        let reserved = bytes[11];
        let sample_rate_train_hz = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        let max_profile = if magic == DNAM_MAGIC { 2 } else { 0 };
        let max_auth_kind = if magic == DNAM_MAGIC { 1 } else { 0 };
        reject_if(
            profile > max_profile
                || auth_kind > max_auth_kind
                || maximum > 3
                || reserved != 0
                || sample_rate_train_hz != 48_000
                || bytes[16..64].iter().any(|&b| b != 0),
            "invalid v2 preamble",
        )?;

        let read_u64 = |idx: usize| -> u64 {
            let off = V2_PREAMBLE_SIZE + idx * 8;
            u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap())
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
            bytes[V2_PREAMBLE_SIZE + 13 * 8..V2_HEADER_OFFSET]
                .iter()
                .any(|&b| b != 0)
                || reserved_offset != 0
                || reserved_capacity != 0
                || header_offset != (V2_HEADER_OFFSET as u64)
                || header_length == 0
                || auth_length == 0,
            "invalid v2 directory",
        )?;

        let header_end = header_offset
            .checked_add(header_length)
            .or_schema_err("header extent overflow")?;
        let shared_end = shared_offset
            .checked_add(shared_length)
            .or_schema_err("shared extent overflow")?;
        let auth_end = auth_offset
            .checked_add(auth_length)
            .or_schema_err("auth extent overflow")?;
        reject_if(
            header_end > (u32::MAX as u64)
                || shared_end > (u32::MAX as u64)
                || auth_end > (u32::MAX as u64)
                || tier_start > (u32::MAX as u64)
                || Some(shared_offset) != align64_u64(header_end)
                || Some(auth_offset) != align64_u64(shared_end)
                || Some(tier_start) != align64_u64(auth_end),
            "invalid v2 directory alignment",
        )?;

        let mut prev_end = tier_start;
        for (idx, &end) in ends_u64.iter().enumerate() {
            if idx <= (maximum as usize) {
                if !end.is_multiple_of(64) {
                    return Err(DnamError::UnalignedOffset {
                        field: "tier_ends",
                        offset: end as u32,
                    });
                }
                reject_if(
                    end <= prev_end || end > (u32::MAX as u64),
                    "invalid v2 tier end",
                )?;
                prev_end = end;
            } else {
                reject_if(end != 0, "unexpected v2 tier end")?;
            }
        }

        let footer = &bytes[bytes.len() - DMUE_FOOTER_SIZE..];
        let mutable_capacity = u64::from_le_bytes(footer[4..12].try_into().unwrap());
        let footer_reserved = u32::from_le_bytes(footer[12..16].try_into().unwrap());
        reject_if(
            &footer[0..4] != b"DMUE"
                || footer_reserved != 0
                || mutable_capacity < (DMU1_HEADER_SIZE as u64 + 2)
                || mutable_capacity > (DEFAULT_MAX_MUTABLE_CAPACITY as u64)
                || mutable_capacity > ((bytes.len() - DMUE_FOOTER_SIZE) as u64),
            "invalid v2 DMUE footer",
        )?;
        let mutable_offset = bytes.len() - DMUE_FOOTER_SIZE - (mutable_capacity as usize);
        let retained_tier = ends_u64[..=(maximum as usize)]
            .iter()
            .position(|&end| end == (mutable_offset as u64))
            .or_schema_err("invalid v2 retained tier boundary")? as u8;

        let retained_end = ends_u64[retained_tier as usize];
        let tensor_arena_offset = tier_start as u32;
        let tensor_arena_byte_len = (retained_end - tier_start) as u32;

        let rel0 = (ends_u64[0] - tier_start) as u32;
        let rel1 = if maximum >= 1 {
            (ends_u64[1] - tier_start) as u32
        } else {
            rel0
        };
        let rel2 = if maximum >= 2 {
            (ends_u64[2] - tier_start) as u32
        } else {
            rel1
        };
        let rel3 = if maximum >= 3 {
            (ends_u64[3] - tier_start) as u32
        } else {
            rel2
        };

        let to_cl = |rel_bytes: u32| -> u16 { (rel_bytes / 64).min(u16::MAX as u32) as u16 };

        let mut flags = FLAG_HAS_LITE | FLAG_STEREO_OUT;
        if retained_tier >= 1 {
            flags |= FLAG_HAS_FULL_DELTA;
        }
        if retained_tier >= 2 {
            flags |= FLAG_HAS_PRO_DELTA;
        }
        if retained_tier >= 3 {
            flags |= FLAG_HAS_ULTRA_DELTA;
        }
        if profile > 0 {
            flags |= FLAG_ENCRYPTED_ARENA;
        }
        if auth_kind > 0 {
            flags |= FLAG_HAS_SIGNATURES;
        }

        let shared_slice = &bytes[shared_offset as usize..shared_end as usize];
        let auth_slice = &bytes[auth_offset as usize..auth_end as usize];

        let num_controls: u16;
        let mut num_ckpt_stems = 0u16;
        let mut tail_time_ms = 0.0f32;
        let mut header_crc32c = 0u32;
        let mut crc_tiers = [0u32; 4];

        if magic == DNAM_MAGIC {
            if auth_slice.starts_with(b"DAU1") {
                flags |= FLAG_HAS_SIGNATURES;
                num_controls = scan_shared_control_count(shared_slice);
            } else if let Some(dcrc) = parse_dcrc_record(auth_slice)? {
                let hint_mask = FLAG_STEREO_OUT
                    | FLAG_ENCRYPTED_ARENA
                    | FLAG_HAS_SIGNATURES
                    | FLAG_HAS_CALIB_STATS;
                flags = (flags & !FLAG_STEREO_OUT) | (dcrc.flags_hint & hint_mask);
                num_controls = dcrc.num_controls;
                tail_time_ms = f32::from_bits(dcrc.tail_time_ms_bits);
                header_crc32c = dcrc.crc_prefix;
                crc_tiers = dcrc.crc_tiers;
            } else {
                num_controls = scan_shared_control_count(shared_slice);
            }
        } else {
            flags |= FLAG_HAS_CKPT_STEMS | FLAG_HAS_SIGNATURES;
            num_ckpt_stems = 1;
            num_controls = scan_shared_control_count(shared_slice);
        }

        Ok(Self {
            magic,
            version_major,
            version_minor,
            flags,
            num_controls,
            sample_rate_train_hz,
            json_header_byte_len: header_length as u32,
            tensor_arena_offset,
            tensor_arena_byte_len,
            arena_lite_end_cl: to_cl(rel0),
            arena_full_end_cl: to_cl(rel1),
            arena_pro_end_cl: to_cl(rel2),
            arena_ultra_end_cl: to_cl(rel3),
            arena_crc32c_lite: crc_tiers[0],
            arena_crc32c_full: crc_tiers[1],
            arena_crc32c_pro: crc_tiers[2],
            arena_crc32c_ultra: crc_tiers[3],
            num_ckpt_stems,
            _reserved: 0,
            tail_time_ms,
            header_crc32c,
        })
    }

    /// Serializes the canonical 64-byte v2.0 wire preamble into little-endian bytes.
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        out[0..4].copy_from_slice(&self.magic);
        out[4..6].copy_from_slice(&self.version_major.to_le_bytes());
        out[6..8].copy_from_slice(&self.version_minor.to_le_bytes());
        out[8] = u8::from(self.is_encrypted_arena());
        out[9] = 0;
        out[10] = self.highest_tier() as u8;
        out[11] = 0;
        out[12..16].copy_from_slice(&self.sample_rate_train_hz.to_le_bytes());
        out
    }

    /// Arena-relative byte end of `Arena 0 (Lite)` (`arena_lite_end_cl * 64`).
    #[inline]
    pub const fn arena_offset_lite_end(&self) -> u32 {
        (self.arena_lite_end_cl as u32) * 64
    }

    /// Arena-relative byte end of `Arena 0..1 (Full)` (`arena_full_end_cl * 64`).
    #[inline]
    pub const fn arena_offset_full_end(&self) -> u32 {
        (self.arena_full_end_cl as u32) * 64
    }

    /// Arena-relative byte end of `Arena 0..2 (Pro)` (`arena_pro_end_cl * 64`).
    #[inline]
    pub const fn arena_offset_pro_end(&self) -> u32 {
        (self.arena_pro_end_cl as u32) * 64
    }

    /// Arena-relative byte end of `Arena 0..3 (Ultra)` (`arena_ultra_end_cl * 64`).
    #[inline]
    pub const fn arena_offset_ultra_end(&self) -> u32 {
        (self.arena_ultra_end_cl as u32) * 64
    }

    /// Returns the highest tier present in the container based on `flags` (`FLAG_HAS_*_DELTA`).
    #[inline]
    pub const fn highest_tier(&self) -> Tier {
        if (self.flags & FLAG_HAS_ULTRA_DELTA) != 0 {
            Tier::Ultra
        } else if (self.flags & FLAG_HAS_PRO_DELTA) != 0 {
            Tier::Pro
        } else if (self.flags & FLAG_HAS_FULL_DELTA) != 0 {
            Tier::Full
        } else {
            Tier::Lite
        }
    }

    /// Returns the cumulative byte end offset for `tier`.
    #[inline]
    pub const fn arena_end_for_tier(&self, tier: Tier) -> usize {
        match tier {
            Tier::Lite => self.arena_offset_lite_end() as usize,
            Tier::Full => self.arena_offset_full_end() as usize,
            Tier::Pro => self.arena_offset_pro_end() as usize,
            Tier::Ultra => self.arena_offset_ultra_end() as usize,
        }
    }

    /// Returns the cumulative Castagnoli `CRC32C` for `tier`.
    #[inline]
    pub const fn crc32c_for_tier(&self, tier: Tier) -> u32 {
        match tier {
            Tier::Lite => self.arena_crc32c_lite,
            Tier::Full => self.arena_crc32c_full,
            Tier::Pro => self.arena_crc32c_pro,
            Tier::Ultra => self.arena_crc32c_ultra,
        }
    }

    /// Returns `true` if `STEREO_OUT` (`bit 9`) is set.
    #[inline]
    pub const fn is_stereo_out(&self) -> bool {
        (self.flags & FLAG_STEREO_OUT) != 0
    }

    /// Returns `true` if `HAS_SIGNATURES` (`bit 10`) is set.
    #[inline]
    pub const fn has_signatures(&self) -> bool {
        (self.flags & FLAG_HAS_SIGNATURES) != 0
    }

    /// Returns `true` if `ENCRYPTED_ARENA` (`bit 11`) is set.
    #[inline]
    pub const fn is_encrypted_arena(&self) -> bool {
        (self.flags & FLAG_ENCRYPTED_ARENA) != 0
    }

    /// Returns `true` if `HAS_CALIB_STATS` (`bit 12`) is set.
    #[inline]
    pub const fn has_calib_stats(&self) -> bool {
        (self.flags & FLAG_HAS_CALIB_STATS) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
    use crate::schema::DnamJsonHeader;
    use alloc::vec;

    #[test]
    fn test_preamble_roundtrip_and_error_branches() {
        assert!(DnamPreamble::from_bytes(&[0u8; 10]).is_err());
        let mut bad_preamble = [0u8; 64];
        bad_preamble[0..4].copy_from_slice(b"NOPE");
        assert!(matches!(
            DnamPreamble::from_bytes(&bad_preamble),
            Err(DnamError::InvalidMagic(_))
        ));
        bad_preamble[0..4].copy_from_slice(b"DNAM");
        bad_preamble[4..6].copy_from_slice(&1_u16.to_le_bytes());
        assert!(matches!(
            DnamPreamble::from_bytes(&bad_preamble),
            Err(DnamError::UnsupportedVersion { major: 1, minor: 0 })
        ));
        bad_preamble[4..6].copy_from_slice(&CONTAINER_VERSION_MAJOR.to_le_bytes());
        assert!(matches!(
            DnamPreamble::from_bytes(&bad_preamble),
            Err(DnamError::BufferTooSmall { .. })
        ));

        let mut header = DnamJsonHeader::default_reference();
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let mut container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
        let mut p = DnamPreamble::from_bytes(&container).unwrap();
        assert_eq!(p.version_major, 2);
        assert_eq!(p.version_minor, 0);
        assert_eq!(p.highest_tier(), Tier::Ultra);
        assert_eq!(p.arena_lite_end_cl, 168);
        assert_eq!(p.arena_full_end_cl, 349);
        assert_eq!(p.arena_pro_end_cl, 566);
        assert_eq!(p.arena_ultra_end_cl, 1000);
        assert_eq!(p.to_bytes()[0..4], DNAM_MAGIC);

        p.arena_crc32c_lite = 11;
        p.arena_crc32c_full = 22;
        p.arena_crc32c_pro = 33;
        p.arena_crc32c_ultra = 44;
        assert_eq!(p.crc32c_for_tier(Tier::Lite), 11);
        assert_eq!(p.crc32c_for_tier(Tier::Full), 22);
        assert_eq!(p.crc32c_for_tier(Tier::Pro), 33);
        assert_eq!(p.crc32c_for_tier(Tier::Ultra), 44);

        // Unaligned tier_end in directory must fail with UnalignedOffset
        container[64 + 9 * 8] = 65;
        assert!(matches!(
            DnamPreamble::from_bytes(&container),
            Err(DnamError::UnalignedOffset { .. })
        ));
    }
}
