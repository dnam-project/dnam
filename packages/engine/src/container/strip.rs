//! Zero-re-signing container tier stripper (`strip_container_to_tier`).
//!
//! Canonical Reference: `docs/containers/dnam_binary_format.md`.

use alloc::string::ToString;
use alloc::vec::Vec;

use crate::schema::DnamError;
use crate::types::Tier;

use super::preamble::DnamPreamble;
use super::view::compute_header_crc32c;
use super::{DNAM_MAGIC, FLAG_HAS_FULL_DELTA, FLAG_HAS_PRO_DELTA, FLAG_HAS_ULTRA_DELTA};

/// Strips a `.dnam` container down to `max_tier` (`Lite = 10,752 B`, `Full = 22,336 B`,
/// `Pro = 36,224 B`, or `Ultra = 64,000 B`) **without modifying the UTF-8 JSON header**,
/// preserving `immutable_header_sha256`, `creator_signature`, and `marketplace_receipt`
/// with zero re-signing required.
pub fn strip_container_to_tier(
    container_bytes: &[u8],
    max_tier: Tier,
) -> Result<Vec<u8>, DnamError> {
    let mut preamble = DnamPreamble::from_bytes(container_bytes)?;
    if preamble.magic != DNAM_MAGIC {
        return Err(DnamError::SchemaValidationError(
            "strip_container_to_tier only applies to runtime .dnam containers".to_string(),
        ));
    }
    let arena_off = preamble.tensor_arena_offset as usize;
    let arena_len = preamble.tensor_arena_byte_len as usize;
    if container_bytes.len() < arena_off + arena_len {
        return Err(DnamError::BufferTooSmall {
            expected: arena_off + arena_len,
            actual: container_bytes.len(),
        });
    }
    let effective_tier = preamble.highest_tier().min(max_tier);
    let target_arena_len = preamble.arena_end_for_tier(effective_tier).min(arena_len);

    if effective_tier < Tier::Ultra {
        preamble.flags &= !FLAG_HAS_ULTRA_DELTA;
    }
    if effective_tier < Tier::Pro {
        preamble.flags &= !FLAG_HAS_PRO_DELTA;
    }
    if effective_tier < Tier::Full {
        preamble.flags &= !FLAG_HAS_FULL_DELTA;
    }
    preamble.tensor_arena_byte_len = target_arena_len as u32;

    let mut stripped = container_bytes[..arena_off + target_arena_len].to_vec();
    stripped[0..64].copy_from_slice(&preamble.to_bytes());
    let new_crc = compute_header_crc32c(&stripped, arena_off);
    stripped[60..64].copy_from_slice(&new_crc.to_le_bytes());
    Ok(stripped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{ARENA_OFFSET_ULTRA_END, DNCK_MAGIC, build_dnam_container};
    use crate::schema::DnamJsonHeader;
    use alloc::vec;

    #[test]
    fn test_strip_container_error_branches_and_tier_flags() {
        let mut hdr = DnamJsonHeader::default_reference();
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let container = build_dnam_container(&mut hdr, &arena, &arena, false).unwrap();

        // Truncated buffer must fail with BufferTooSmall
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
    }
}
