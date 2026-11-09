//! `.dnam` & `.dnam-ckpt` Hybrid Binary Container Parser, Builder, ZOH Rescaling & `Arena 4` `uplo='U'` Packing (`container`).
//!
//! Implements the wire framing, zero-copy `mmap` loader (`< 50 ns` zero-JSON hardware fast-path),
//! 4 KB page-aligned in-place JSON header editor, Castagnoli `CRC32C` verification,
//! `.dnam-ckpt` `Arena 4` column-major upper-triangular (`uplo='U'`) sufficient statistics packing,
//! and Continuous-Time Zero-Order Hold (ZOH) sample-rate rescaling ($44.1\text{--}96\text{ kHz}$ live,
//! plus $4\times$ internal ZOH oversampling for offline bounce)
//! from [`docs/containers/dnam_binary_format.md`](../../../docs/containers/dnam_binary_format.md)
//! and [`docs/containers/checkpoint_and_cache_format.md`](../../../docs/containers/checkpoint_and_cache_format.md).

pub mod builder;
pub mod preamble;
pub mod strip;
pub mod view;
pub mod wire;
pub mod zoh;

pub use wire::{
    CP1_MAGIC, Cp1DecodedRecord, DEFAULT_MAX_MUTABLE_CAPACITY, DMU1_HEADER_SIZE, DMUE_FOOTER_SIZE,
    DeclaredDelayHistory, StructuredConditionalBound, StructuredConditionalCurve,
    SwitchControlSpec, V2_DIRECTORY_SIZE, V2_HEADER_OFFSET, V2_PREAMBLE_SIZE, V2ContainerLayout,
    align64_u64, append_dmue_mutable_suffix, build_dmu1_mutable_frame,
    build_v2_open_dnam_container, compute_v2_open_retained_prefix_sha256,
    cp1_contributor_record_bytes, decode_cp1_contributor_stream,
    decode_cp1_contributor_stream_into_slice, decode_cp1_shared_stream_into_slice,
    encode_domain_field, eval_ui_midpoint_power_skew, is_valid_jcs_unicode_text,
    parse_and_validate_dmu1_frame, parse_canonical_reference_samples,
    parse_strict_json_object_bytes, parse_structured_conditional_bounds,
    parse_structured_conditional_curves, project_immutable_header_value,
    resolve_conditional_safe_interval, select_conditional_calibration_curve, serialize_jcs_value,
    strip_v2_open_dnam_container, validate_canonical_jcs_header_bytes, validate_delay_histories,
    validate_dmu1_frame_no_alloc, validate_mutable_json_schema, validate_stage3_tier_slices,
    validate_v2_dnam_layout_no_alloc, validate_v2_open_dnam_layout,
};

#[cfg(feature = "crypto")]
pub use crate::provenance::seal_dnam_container;
pub use crate::provenance::{
    DNSL_MAGIC, DnamSealEnvelopePreamble, DnamSealRecipientSlot, FLAG_SEAL_CHACHA20_FULL,
    MAX_SEAL_RECIPIENTS, unseal_dnam_container,
};
pub use builder::{build_dnam_container, update_header_in_place};
pub use preamble::{DnamPreamble, is_v2_wire_directory_container};
pub use strip::strip_container_to_tier;
pub use view::{
    DnamContainerView, SharedControlsExtent, compute_header_crc32c, decode_shared_controls_extent,
};
pub use zoh::ZohRescaledParams;

/// Magic bytes for a `.dnam` deployment model container (`0x4D414E44` little-endian).
pub const DNAM_MAGIC: [u8; 4] = *b"DNAM";

/// Magic bytes for a `.dnam-ckpt` iterative training checkpoint container (`0x4B434E44` little-endian).
pub const DNCK_MAGIC: [u8; 4] = *b"DNCK";

/// Current container major version (`2`).
pub const CONTAINER_VERSION_MAJOR: u16 = 2;

/// Current container minor version (`0`).
pub const CONTAINER_VERSION_MINOR: u16 = 0;

/// Fixed byte length of `Arena 0 (Lite)` (`168 × 64 B = 10,752 B`).
pub const ARENA_OFFSET_LITE_END: u32 = 10_752;

/// Fixed cumulative byte end of `Arena 0..1 (Full)` (`349 × 64 B = 22,336 B`).
pub const ARENA_OFFSET_FULL_END: u32 = 22_336;

/// Fixed cumulative byte end of `Arena 0..2 (Pro)` (`566 × 64 B = 36,224 B`).
pub const ARENA_OFFSET_PRO_END: u32 = 36_224;

/// Fixed cumulative byte end of `Arena 0..3 (Ultra)` (`1,000 × 64 B = 64,000 B`).
pub const ARENA_OFFSET_ULTRA_END: u32 = 64_000;

/// End of `Arena 0 (Lite)` in 64-byte cache lines (`168`).
pub const ARENA_LITE_END_CL: u16 = 168;

/// End of `Arena 0..1 (Full)` in 64-byte cache lines (`349`).
pub const ARENA_FULL_END_CL: u16 = 349;

/// End of `Arena 0..2 (Pro)` in 64-byte cache lines (`566`).
pub const ARENA_PRO_END_CL: u16 = 566;

/// End of `Arena 0..3 (Ultra)` in 64-byte cache lines (`1,000`).
pub const ARENA_ULTRA_END_CL: u16 = 1_000;

/// Default 4 KB OS virtual-memory page boundary for `tensor_arena_offset` (`0x1000 = 4,096 B`).
pub const DEFAULT_PAGE_ALIGNED_ARENA_OFFSET: u32 = 4_096;

// ============================================================================
// Preamble Flag Bits (`flags: u16`)
// ============================================================================

/// Bit 0 (`1`): `Arena 0 (Lite)` is present.
pub const FLAG_HAS_LITE: u16 = 1 << 0;
/// Bit 1 (`2`): `Arena 1 (Full Delta)` is present.
pub const FLAG_HAS_FULL_DELTA: u16 = 1 << 1;
/// Bit 2 (`4`): `Arena 2 (Pro Delta)` is present.
pub const FLAG_HAS_PRO_DELTA: u16 = 1 << 2;
/// Bit 3 (`8`): `Arena 3 (Ultra Delta)` is present.
pub const FLAG_HAS_ULTRA_DELTA: u16 = 1 << 3;
/// Bit 4 (`16`): `.dnam-ckpt` stem statistics are present.
pub const FLAG_HAS_CKPT_STEMS: u16 = 1 << 4;
/// Bit 9 (`512`): Stereo output head enabled (`1 = stereo`, `0 = mono` saving 50% head MACs).
pub const FLAG_STEREO_OUT: u16 = 1 << 9;
/// Bit 10 (`1024`): Split-payload `Ed25519` signatures are present in `metadata.signatures`.
pub const FLAG_HAS_SIGNATURES: u16 = 1 << 10;
/// Bit 11 (`2048`): `Arena 0..3` is `ChaCha20` 64-byte block encrypted for `authorized_devices`.
pub const FLAG_ENCRYPTED_ARENA: u16 = 1 << 11;
/// Bit 12 (`4096`): `.dnam-ckpt` calibration statistics (`DNCA`) are present.
pub const FLAG_HAS_CALIB_STATS: u16 = 1 << 12;

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
