//! `.dnam` & `.dnam-ckpt` Hybrid Binary Container Parser, Builder, ZOH Rescaling & `Arena 3` `uplo='U'` Packing (`container`).
//!
//! Implements the wire framing, zero-copy `mmap` loader (`< 50 ns` zero-JSON hardware fast-path),
//! 4 KB page-aligned in-place JSON header editor, Castagnoli `CRC32C` verification,
//! `.dnam-ckpt` `Arena 3` column-major upper-triangular (`uplo='U'`) sufficient statistics packing,
//! and Continuous-Time Zero-Order Hold (ZOH) sample-rate rescaling ($44.1\text{--}96\text{ kHz}$)
//! from [`docs/dnam_container_schema.md`](../../../docs/dnam_container_schema.md).

pub mod ckpt;
pub mod preamble;
pub mod view;
pub mod zoh;

#[cfg(feature = "crypto")]
pub use crate::provenance::seal_dnam_container;
pub use crate::provenance::{
    DNSL_MAGIC, DnamSealEnvelopePreamble, DnamSealRecipientSlot, FLAG_SEAL_CHACHA20_FULL,
    MAX_SEAL_RECIPIENTS, unseal_dnam_container,
};
pub use ckpt::{
    ARENA4_HEADER_BYTES, DNCA_MAGIC, pack_uplo_u_f64, unpack_uplo_u_f64, uplo_u_packed_len,
    verify_ckpt_arena4_calib_stats, verify_ckpt_stem_crc32c,
};
pub use preamble::DnamPreamble;
pub use view::{
    DnamContainerView, build_dnam_container, compute_header_crc32c, update_header_in_place,
};
pub use zoh::ZohRescaledParams;

/// Magic bytes for a `.dnam` deployment model container (`0x4D414E44` little-endian).
pub const DNAM_MAGIC: [u8; 4] = *b"DNAM";

/// Magic bytes for a `.dnam-ckpt` iterative training checkpoint container (`0x4B434E44` little-endian).
pub const DNCK_MAGIC: [u8; 4] = *b"DNCK";

/// Current container major version (`1`).
pub const CONTAINER_VERSION_MAJOR: u16 = 1;

/// Current container minor version (`1`, backwards-compatible with `1.0.0` and `1.1.0`).
pub const CONTAINER_VERSION_MINOR: u16 = 1;

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
/// Bit 11 (`2048`): `Arena 4` `.dnam-ckpt` calibration statistics (`DNCA`) are present.
pub const FLAG_HAS_CALIB_STATS: u16 = 1 << 11;

/// Updates an incremental Castagnoli `CRC32C` state with `data` using hardware `CRC32C` on `aarch64`.
#[inline]
pub fn crc32c_update(crc: u32, data: &[u8]) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        crate::math::neon::crc32c_arm64_update(crc, data)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        crate::math::crc32c_slice(crc, data)
    }
}

/// Computes the Castagnoli `CRC32C` (`0x1EDC6F41`) checksum of `data`.
#[inline]
pub fn crc32c_compute(data: &[u8]) -> u32 {
    crc32c_update(0, data)
}
