//! Zero-copy container view (`DnamContainerView`), 4 KB page-aligned builder (`build_dnam_container`),
//! and `< 0.1 ms` in-place JSON header editor (`update_header_in_place`).

use alloc::string::ToString;
use alloc::vec::Vec;

use crate::memory::{AlignedBuffer, ArenaStorage, CacheLine64};
use crate::provenance::{
    CryptoProvider, VerificationStatus, chacha20_xor_arena_blocks, decode_hex_array,
    verify_container_provenance,
};
use crate::schema::{DnamError, DnamJsonHeader};

use super::preamble::DnamPreamble;
use super::{
    ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_ULTRA_END, CONTAINER_VERSION_MAJOR,
    CONTAINER_VERSION_MINOR, DNAM_MAGIC, DNCK_MAGIC, FLAG_CORE2_ACTIVE, FLAG_DEFAULT_ANCHORED,
    FLAG_ENCRYPTED_ARENA, FLAG_EXEC_ORDER_SOA, FLAG_HAS_CKPT_STEMS, FLAG_HAS_FULL_DELTA,
    FLAG_HAS_LITE, FLAG_HAS_SIGNATURES, FLAG_HAS_ULTRA_DELTA, FLAG_LFO_ENV_ACTIVE, FLAG_STEREO_OUT,
    crc32c_update,
};

/// Computes `header_crc32c` over `container_bytes[0..tensor_arena_offset]` with
/// `bytes[0x3C..0x40]` treated as `[0, 0, 0, 0]`.
pub fn compute_header_crc32c(container_bytes: &[u8], tensor_arena_offset: usize) -> u32 {
    let mut crc = crc32c_update(0, &container_bytes[0..60]);
    crc = crc32c_update(crc, &[0u8; 4]);
    crc32c_update(crc, &container_bytes[64..tensor_arena_offset])
}

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

    let json_bytes =
        serde_json::to_vec(header).map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?;
    let min_header_region = 64 + json_bytes.len();
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
    let mut out = alloc::vec![0x20_u8; total_len];
    out[0..64].copy_from_slice(&preamble.to_bytes());
    out[64..64 + json_bytes.len()].copy_from_slice(&json_bytes);
    out[tensor_arena_offset..].copy_from_slice(stored_arena_bytes);

    preamble.header_crc32c = compute_header_crc32c(&out, tensor_arena_offset);
    out[60..64].copy_from_slice(&preamble.header_crc32c.to_le_bytes());
    Ok(out)
}

/// Updates the JSON header (`presets`, `user_metadata`, or `authorized_devices`) **100% in-place**
/// inside the pre-allocated 4 KB header region (`0x40..tensor_arena_offset`).
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::DefaultCryptoProvider;
    use crate::schema::{
        ControlBusConfig, Core1Config, Core2Config, DatasetProvenance, ExtractedPhysics, IoConfig,
        ModelMetadata, SlimmableTiersConfig, StemProvenance, StemRegistryEntry, TensorSliceEntry,
        TierProfile,
    };
    use alloc::collections::BTreeMap;
    use alloc::{format, vec};

    fn make_base_header(name: &str) -> DnamJsonHeader {
        let mk_tier = |ch: u8, fdn: u8, end_off: u32| TierProfile {
            channels: ch,
            fdn_lines: fdn,
            early_taps: fdn,
            head_svd_rank: 4,
            loudness_lufs: -20.5,
            output_trim_db: 0.0,
            validation_esr_db: -42.0,
            arena_end_byte_offset: end_off,
            prefix_crc32c: 0,
        };
        DnamJsonHeader {
            version: "1.0.0".to_string(),
            container_kind: "dnam".to_string(),
            architecture: "DNAM_TriPhysics".to_string(),
            sample_rate_train_hz: 48_000,
            metadata: ModelMetadata {
                name: name.to_string(),
                author: Some("UnitTest".to_string()),
                description: None,
                license: Some("MIT".to_string()),
                created_at: "2026-10-23T00:00:00Z".to_string(),
                input_reference_dbfs: -20.5,
                provenance: DatasetProvenance::default(),
                signatures: None,
            },
            user_metadata: BTreeMap::new(),
            io_config: IoConfig::default(),
            slimmable_tiers: SlimmableTiersConfig {
                lite: mk_tier(4, 4, ARENA_OFFSET_LITE_END),
                full: mk_tier(8, 8, ARENA_OFFSET_FULL_END),
                ultra: mk_tier(12, 8, ARENA_OFFSET_ULTRA_END),
            },
            controls_schema: Vec::new(),
            presets: Vec::new(),
            control_bus: ControlBusConfig::default(),
            core1_hybrid_ssm: Core1Config::default(),
            core2_mufdn: Core2Config::default(),
            tensor_directory: BTreeMap::new(),
            stems_registry: None,
            audit_report: None,
        }
    }

    #[test]
    fn test_container_build_view_and_in_place_update() {
        let mut hdr = make_base_header("Test");
        assert!(build_dnam_container(&mut hdr, &[0u8; 128], &[0u8; 128], false).is_err());

        let lite_arena = vec![0u8; ARENA_OFFSET_LITE_END as usize];
        let full_arena = vec![0u8; ARENA_OFFSET_FULL_END as usize];
        let ultra_arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];

        let lite_bytes = build_dnam_container(&mut hdr, &lite_arena, &lite_arena, false).unwrap();
        let full_bytes = build_dnam_container(&mut hdr, &full_arena, &full_arena, false).unwrap();
        let mut ultra_bytes =
            build_dnam_container(&mut hdr, &ultra_arena, &ultra_arena, false).unwrap();

        let crypto = DefaultCryptoProvider::default();
        assert!(DnamContainerView::from_bytes(&lite_bytes, &crypto).is_ok());
        assert!(DnamContainerView::from_bytes(&full_bytes, &crypto).is_ok());
        assert!(DnamContainerView::from_bytes(&ultra_bytes[..100], &crypto).is_err());

        ultra_bytes[60] ^= 0xFF;
        assert!(DnamContainerView::from_bytes(&ultra_bytes, &crypto).is_err());
        ultra_bytes[60] ^= 0xFF;

        let mut preamble = DnamPreamble::from_bytes(&ultra_bytes).unwrap();
        let arena_off = preamble.tensor_arena_offset as usize;
        let raw_arena = &ultra_bytes[arena_off..];
        preamble.arena_crc32c_lite ^= 1;
        assert!(DnamContainerView::verify_arena_prefix_crcs(&preamble, raw_arena).is_err());
        preamble.arena_crc32c_lite ^= 1;
        preamble.arena_crc32c_full ^= 1;
        assert!(DnamContainerView::verify_arena_prefix_crcs(&preamble, raw_arena).is_err());
        preamble.arena_crc32c_full ^= 1;
        preamble.arena_crc32c_ultra ^= 1;
        assert!(DnamContainerView::verify_arena_prefix_crcs(&preamble, raw_arena).is_err());

        // dnam-ckpt container with stems_registry
        hdr.container_kind = "dnam-ckpt".to_string();
        let dummy_slice = TensorSliceEntry {
            arena: 3,
            dtype: "f64".to_string(),
            packing: Some("uplo_upper_col_major_tier_ordered".to_string()),
            feature_dim: Some(64),
            packed_len: Some(2080),
            shape: None,
            byte_offset: 0,
            byte_length: 64,
        };
        hdr.stems_registry = Some(vec![StemRegistryEntry {
            stem_id: "stem0".to_string(),
            controls_meta: BTreeMap::new(),
            provenance: StemProvenance {
                input_pcm_sha256: "00".repeat(32),
                target_pcm_sha256: "11".repeat(32),
                num_frames: 48_000,
            },
            stem_arena3_crc32c: super::super::crc32c_compute(&[0u8; 64]),
            user_metadata: BTreeMap::new(),
            extracted_physics: Some(ExtractedPhysics {
                delays_ms: vec![10.0],
                lfo_freqs_hz: vec![0.5],
                decay_gains_lin: vec![0.7],
            }),
            g_audio_uplo_u_f64: dummy_slice.clone(),
            c_audio_tier_f64: dummy_slice,
        }]);
        let ckpt_bytes = build_dnam_container(&mut hdr, &lite_arena, &lite_arena, false).unwrap();
        let ckpt_view = DnamContainerView::from_bytes(&ckpt_bytes, &crypto).unwrap();
        assert_eq!(ckpt_view.preamble.num_ckpt_stems, 1);

        assert!(update_header_in_place(&mut ultra_bytes[..64], &hdr).is_err());
        let mut huge_hdr = make_base_header("Huge");
        for i in 0..200 {
            huge_hdr
                .user_metadata
                .insert(format!("key_{i}"), serde_json::json!("x".repeat(50)));
        }
        assert!(update_header_in_place(&mut ultra_bytes, &huge_hdr).is_err());
    }
}
