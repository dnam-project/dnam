//! Zero-re-signing canonical v2.0 `.dnam` container tier stripper (`strip_container_to_tier`).
//!
//! Canonical Reference: `docs/specs/containers/dnam_binary_format.md`.

use alloc::string::ToString;
use alloc::vec::Vec;

use crate::schema::DnamError;
use crate::types::Tier;

use super::wire::{V2_PREAMBLE_SIZE, parse_dcrc_record, validate_v2_dnam_layout_with_max_profile};
use super::{DNAM_MAGIC, crc32c_update};

/// Strips a canonical `v2.0` `.dnam` container down to `max_tier` (`Lite = 10,752 B`,
/// `Full = 22,336 B`, `Pro = 36,224 B`, or `Ultra = 64,000 B`) **without modifying the preamble,
/// directory, Region 3 JCS header, Region 4 shared extent, or `auth_extent`**, preserving
/// `DAU1` / `metadata.signatures` and `DMU1` mutable metadata with zero re-signing required.
pub fn strip_container_to_tier(
    container_bytes: &[u8],
    max_tier: Tier,
) -> Result<Vec<u8>, DnamError> {
    if container_bytes.len() < V2_PREAMBLE_SIZE {
        return Err(DnamError::BufferTooSmall {
            expected: V2_PREAMBLE_SIZE,
            actual: container_bytes.len(),
        });
    }
    if container_bytes[0..4] != DNAM_MAGIC {
        return Err(DnamError::SchemaValidationError(
            "strip_container_to_tier only applies to runtime .dnam containers".to_string(),
        ));
    }
    let layout = validate_v2_dnam_layout_with_max_profile(container_bytes, 1)?;
    let auth_slice =
        &container_bytes[layout.auth_extent.0..layout.auth_extent.0 + layout.auth_extent.1];
    if let Some(dcrc) = parse_dcrc_record(auth_slice)? {
        let calc_crc = crc32c_update(0, &container_bytes[..layout.auth_extent.0]);
        if calc_crc != dcrc.crc_prefix {
            return Err(DnamError::Crc32cMismatch {
                region: "header",
                expected: dcrc.crc_prefix,
                computed: calc_crc,
            });
        }
    }

    let effective_tier = (max_tier as u8).min(layout.retained_tier);
    let tier_end = layout.tier_ends[effective_tier as usize];
    let mutable_offset = layout.mutable_extent.0;
    let mut stripped = Vec::with_capacity(tier_end + (container_bytes.len() - mutable_offset));
    stripped.extend_from_slice(&container_bytes[..tier_end]);
    stripped.extend_from_slice(&container_bytes[mutable_offset..]);
    Ok(stripped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{
        ARENA_OFFSET_ULTRA_END, DNCK_MAGIC, DnamPreamble, build_dnam_container,
    };
    use crate::schema::DnamJsonHeader;
    use alloc::vec;

    #[test]
    fn test_strip_container_error_branches_and_tier_flags() {
        let mut hdr = DnamJsonHeader::default_reference();
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let container = build_dnam_container(&mut hdr, &arena, &arena, false).unwrap();

        // Truncated buffer must fail
        assert!(strip_container_to_tier(&container[..32], Tier::Lite).is_err());
        assert!(strip_container_to_tier(&container[..512], Tier::Lite).is_err());

        // Non-DNAM magic (e.g. DNCK checkpoint) must be rejected
        let mut ckpt_magic = container.clone();
        ckpt_magic[0..4].copy_from_slice(&DNCK_MAGIC);
        assert!(strip_container_to_tier(&ckpt_magic, Tier::Lite).is_err());

        // Stripping to Ultra preserves all flags and 8 SVD modes
        let ultra_stripped = strip_container_to_tier(&container, Tier::Ultra).unwrap();
        let pr_ultra = DnamPreamble::from_bytes(&ultra_stripped).unwrap();
        assert_eq!(pr_ultra.highest_tier(), Tier::Ultra);

        // Stripping to Lite clears Full/Pro/Ultra flags and sets highest_tier to Lite (4 SVD modes)
        let lite_stripped = strip_container_to_tier(&container, Tier::Lite).unwrap();
        let pr_lite = DnamPreamble::from_bytes(&lite_stripped).unwrap();
        assert_eq!(pr_lite.highest_tier(), Tier::Lite);
        assert_eq!(pr_lite.highest_tier().svd_control_modes(), 4);

        // Stripping an already-Lite container to Pro clamps cleanly to Lite
        let clamped_lite = strip_container_to_tier(&lite_stripped, Tier::Pro).unwrap();
        assert_eq!(clamped_lite, lite_stripped);
    }
}
