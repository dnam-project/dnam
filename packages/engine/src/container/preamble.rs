//! Fixed 64-byte binary preamble (`DnamPreamble`) at byte offset `0x0000` of a `.dnam` or `.dnam-ckpt` container.

use core::mem::{align_of, size_of};

use crate::schema::DnamError;

use super::{
    CONTAINER_VERSION_MAJOR, CONTAINER_VERSION_MINOR, DNAM_MAGIC, DNCK_MAGIC, FLAG_CORE2_ACTIVE,
    FLAG_ENCRYPTED_ARENA, FLAG_HAS_SIGNATURES, FLAG_LFO_ENV_ACTIVE, FLAG_STEREO_OUT,
};

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
        if version_major != CONTAINER_VERSION_MAJOR || version_minor > CONTAINER_VERSION_MINOR {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preamble_roundtrip_and_error_branches() {
        assert!(DnamPreamble::from_bytes(&[0u8; 10]).is_err());
        let mut bad_preamble = [0u8; 64];
        bad_preamble[0..4].copy_from_slice(b"NOPE");
        assert!(DnamPreamble::from_bytes(&bad_preamble).is_err());
        bad_preamble[0..4].copy_from_slice(b"DNAM");
        bad_preamble[4..6].copy_from_slice(&99_u16.to_le_bytes());
        assert!(DnamPreamble::from_bytes(&bad_preamble).is_err());
        bad_preamble[4..6].copy_from_slice(&1_u16.to_le_bytes());
        bad_preamble[20..24].copy_from_slice(&65_u32.to_le_bytes());
        assert!(DnamPreamble::from_bytes(&bad_preamble).is_err());
    }
}
