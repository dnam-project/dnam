//! Zero-copy container view (`DnamContainerView`), 4 KB page-aligned builder (`build_dnam_container`),
//! `< 0.1 ms` in-place JSON header editor (`update_header_in_place`), and zero-re-signing
//! container tier stripper (`strip_container_to_tier`).

use alloc::string::ToString;
use alloc::vec::Vec;

use crate::control::ControlChannelRecord;
use crate::memory::{AlignedBuffer, ArenaStorage, CacheLine64};
use crate::provenance::{
    CryptoProvider, DNSL_MAGIC, VerificationStatus, chacha20_xor_arena_blocks, decode_hex_array,
    enforce_distribution_policy, unseal_dnam_container, verify_container_provenance,
};
use crate::schema::{DnamError, DnamJsonHeader, DnamProtectionProfile};
use crate::types::Tier;

use super::preamble::DnamPreamble;
use super::{
    ARENA4_STEM_BYTES_PRO, ARENA4_STEM_BYTES_ULTRA, DNAM_MAGIC, DNCK_MAGIC, FLAG_HAS_FULL_DELTA,
    FLAG_HAS_PRO_DELTA, FLAG_HAS_ULTRA_DELTA, crc32c_update, verify_ckpt_arena5_calib_stats,
};

/// Computes `header_crc32c` over `container_bytes[0..tensor_arena_offset]` with
/// `bytes[0x3C..0x40]` treated as `[0, 0, 0, 0]`.
pub fn compute_header_crc32c(container_bytes: &[u8], tensor_arena_offset: usize) -> u32 {
    let mut crc = crc32c_update(0, &container_bytes[0..60]);
    crc = crc32c_update(crc, &[0u8; 4]);
    crc32c_update(crc, &container_bytes[64..tensor_arena_offset])
}

/// Zero-copy or decrypted view over a validated `.dnam`, `.dnam-ckpt`, or `.dnam-seal` (`b"DNSL"`) container.
#[derive(Debug)]
pub struct DnamContainerView<'a> {
    /// 64-byte binary preamble.
    pub preamble: DnamPreamble,
    /// Parsed UTF-8 JSON header (`no_std` `serde` + `alloc`).
    pub header: DnamJsonHeader,
    /// Cryptographic provenance status (`Unsigned`, `CreatorVerified`, or `MarketplaceVerified`).
    pub verification_status: VerificationStatus,
    /// Resolved container protection profile (`Open`, `DeviceAllowlisted`, or `EnclaveSealed`).
    pub protection_profile: DnamProtectionProfile,
    /// Unified 128-byte `ControlChannelRecord[0..P]` table parsed from Region 2B (`control_table_offset..tensor_arena_offset`).
    pub control_records: Vec<ControlChannelRecord>,
    /// 64-byte aligned tensor arena storage (`Borrowed` on aligned `mmap`, or `Owned` if decrypted/copied).
    pub arena_storage: ArenaStorage<'a>,
}

impl<'a> DnamContainerView<'a> {
    /// Full container loader: loads all tiers present in `bytes` (up to [`Tier::Ultra`]).
    #[inline]
    pub fn from_bytes<C: CryptoProvider>(bytes: &'a [u8], crypto: &C) -> Result<Self, DnamError> {
        Self::from_bytes_with_max_tier(bytes, crypto, Tier::Ultra)
    }

    /// Tier-capped container loader: loads, decrypts, and verifies only up to `max_supported_tier`
    /// (`Lite = 10,752 B`, `Full = 22,336 B`, `Pro = 36,224 B`, `Ultra = 64,000 B`) so plugins on
    /// constrained embedded hardware never allocate, decrypt, or touch higher-tier arena bytes.
    pub fn from_bytes_with_max_tier<C: CryptoProvider>(
        bytes: &'a [u8],
        crypto: &C,
        max_supported_tier: Tier,
    ) -> Result<Self, DnamError> {
        if bytes.len() >= 4 && bytes[0..4] == DNSL_MAGIC {
            let unsealed = unseal_dnam_container(bytes, crypto)?;
            let owned_view = DnamContainerView::from_inner_dnam_bytes(
                &unsealed,
                crypto,
                true,
                max_supported_tier,
            )?;
            return Ok(Self {
                preamble: owned_view.preamble,
                header: owned_view.header,
                verification_status: owned_view.verification_status,
                protection_profile: owned_view.protection_profile,
                control_records: owned_view.control_records,
                arena_storage: ArenaStorage::from_slice_owned(owned_view.arena_storage.as_bytes()),
            });
        }
        Self::from_inner_dnam_bytes(bytes, crypto, false, max_supported_tier)
    }

    fn from_inner_dnam_bytes<C: CryptoProvider>(
        bytes: &'a [u8],
        crypto: &C,
        is_enclave_sealed_wrapper: bool,
        max_supported_tier: Tier,
    ) -> Result<Self, DnamError> {
        let mut preamble = DnamPreamble::from_bytes(bytes)?;
        let arena_off = preamble.tensor_arena_offset as usize;
        let full_arena_len = preamble.tensor_arena_byte_len as usize;
        let json_len = preamble.json_header_byte_len as usize;

        let effective_max_tier = preamble.highest_tier().min(max_supported_tier);
        let arena_len = if preamble.magic == DNAM_MAGIC {
            preamble
                .arena_end_for_tier(effective_max_tier)
                .min(full_arena_len)
        } else {
            full_arena_len
        };

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
        let schema_ver = header.schema_version()?;
        schema_ver.ensure_matches_preamble(preamble.version_major, preamble.version_minor)?;

        let p_ctrls = header.controls_schema.len();
        if arena_off < 64 + json_len + 128 * p_ctrls {
            return Err(DnamError::BufferTooSmall {
                expected: 64 + json_len + 128 * p_ctrls,
                actual: arena_off,
            });
        }
        let control_table_off = arena_off - 128 * p_ctrls;
        let mut control_records = Vec::with_capacity(p_ctrls);
        for p in 0..p_ctrls {
            let rec_off = control_table_off + p * 128;
            let mut rec_bytes = [0u8; 128];
            rec_bytes.copy_from_slice(&bytes[rec_off..rec_off + 128]);
            control_records.push(ControlChannelRecord::from_bytes(&rec_bytes));
        }

        // If capped by max_supported_tier on a runtime .dnam, mask out higher tier flags in the view preamble
        if preamble.magic == DNAM_MAGIC && effective_max_tier < preamble.highest_tier() {
            if effective_max_tier < Tier::Ultra {
                preamble.flags &= !FLAG_HAS_ULTRA_DELTA;
            }
            if effective_max_tier < Tier::Pro {
                preamble.flags &= !FLAG_HAS_PRO_DELTA;
            }
            if effective_max_tier < Tier::Full {
                preamble.flags &= !FLAG_HAS_FULL_DELTA;
            }
            preamble.tensor_arena_byte_len = arena_len as u32;
        }

        // 3. Prepare or decrypt Arena prefix up to arena_len
        let raw_arena = &bytes[arena_off..arena_off + arena_len];
        let arena_storage =
            if preamble.is_encrypted_arena() {
                let sigs = header.metadata.signatures.as_ref().ok_or(
                    DnamError::CryptoVerificationFailed(
                        "ENCRYPTED_ARENA=1 requires metadata.signatures",
                    ),
                )?;
                let receipt = sigs.marketplace_receipt.as_ref().ok_or(
                    DnamError::CryptoVerificationFailed(
                        "ENCRYPTED_ARENA=1 requires metadata.signatures.marketplace_receipt",
                    ),
                )?;
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
                let decrypt_end = (preamble.arena_offset_ultra_end() as usize).min(arena_len);
                chacha20_xor_arena_blocks(
                    &mut owned.as_bytes_mut()[..decrypt_end],
                    &cek,
                    &nonce_96,
                    0,
                )?;
                ArenaStorage::from_owned_buffer(owned)
            } else if is_enclave_sealed_wrapper {
                ArenaStorage::from_slice_owned(raw_arena)
            } else {
                ArenaStorage::from_slice_or_copy(raw_arena)
            };

        // 4. Resolve protection_profile and enforce creator-signed DistributionPolicy
        let has_authorized_devices = header
            .metadata
            .signatures
            .as_ref()
            .and_then(|s| s.marketplace_receipt.as_ref())
            .is_some_and(|r| !r.authorized_devices.is_empty());
        let protection_profile = if is_enclave_sealed_wrapper {
            DnamProtectionProfile::EnclaveSealed
        } else if preamble.is_encrypted_arena() || has_authorized_devices {
            DnamProtectionProfile::DeviceAllowlisted
        } else {
            DnamProtectionProfile::Open
        };
        enforce_distribution_policy(&header, protection_profile)?;

        let plain_bytes = arena_storage.as_bytes();
        Self::verify_arena_prefix_crcs(&preamble, plain_bytes)?;

        if preamble.magic == DNAM_MAGIC && preamble.has_calib_stats() {
            return Err(DnamError::SchemaValidationError(
                "FLAG_HAS_CALIB_STATS (Arena 5) is forbidden in runtime .dnam containers"
                    .to_string(),
            ));
        }
        if preamble.magic == DNCK_MAGIC && preamble.has_calib_stats() {
            let weights_end = preamble.arena_end_for_tier(preamble.highest_tier());
            let stems_end = header
                .stems_registry
                .as_ref()
                .and_then(|stems| {
                    stems
                        .iter()
                        .map(|s| {
                            (s.c_audio_tier_f64.byte_offset + s.c_audio_tier_f64.byte_length)
                                as usize
                        })
                        .max()
                        .filter(|&end| end >= weights_end)
                })
                .unwrap_or_else(|| {
                    let num_stems = preamble.num_ckpt_stems as usize;
                    let stride =
                        if plain_bytes.len() >= weights_end + num_stems * ARENA4_STEM_BYTES_ULTRA {
                            ARENA4_STEM_BYTES_ULTRA
                        } else {
                            ARENA4_STEM_BYTES_PRO
                        };
                    weights_end + num_stems * stride
                });
            verify_ckpt_arena5_calib_stats(plain_bytes, stems_end)?;
        }

        // 5. Verify split-payload Ed25519 provenance over the loaded model weight arena prefix
        let model_weights_end = preamble
            .arena_end_for_tier(preamble.highest_tier())
            .min(plain_bytes.len());
        let verification_status = verify_container_provenance(
            &header,
            preamble.has_signatures(),
            &plain_bytes[..model_weights_end],
            crypto,
        )?;

        Ok(Self {
            preamble,
            header,
            verification_status,
            protection_profile,
            control_records,
            arena_storage,
        })
    }

    /// Verifies `arena_crc32c_lite`, `arena_crc32c_full`, `arena_crc32c_pro`, and `arena_crc32c_ultra`
    /// over the available prefix of `plain_arena` in a single incremental pass.
    pub fn verify_arena_prefix_crcs(
        preamble: &DnamPreamble,
        plain_arena: &[u8],
    ) -> Result<(), DnamError> {
        let lite_end = preamble.arena_offset_lite_end() as usize;
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
        let full_end = preamble.arena_offset_full_end() as usize;
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
        let pro_end = preamble.arena_offset_pro_end() as usize;
        if (preamble.flags & FLAG_HAS_PRO_DELTA) != 0
            && plain_arena.len() >= pro_end
            && pro_end > full_end
        {
            crc = crc32c_update(crc, &plain_arena[full_end..pro_end]);
            if crc != preamble.arena_crc32c_pro {
                return Err(DnamError::Crc32cMismatch {
                    region: "arena_pro",
                    expected: preamble.arena_crc32c_pro,
                    computed: crc,
                });
            }
        }
        let ultra_end = preamble.arena_offset_ultra_end() as usize;
        if (preamble.flags & FLAG_HAS_ULTRA_DELTA) != 0
            && plain_arena.len() >= ultra_end
            && ultra_end > pro_end
        {
            crc = crc32c_update(crc, &plain_arena[pro_end..ultra_end]);
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

    /// Returns the byte offset of the 128-byte `ControlChannelRecord[0..P]` table.
    /// Canonical Reference: `docs/dsp/parametric_control_bus.md` §5:
    /// `control_table_offset = tensor_arena_offset - 128 * P`.
    #[inline]
    pub fn control_table_offset(&self) -> usize {
        let p = self.header.controls_schema.len();
        (self.preamble.tensor_arena_offset as usize).saturating_sub(128 * p)
    }
}

pub use super::builder::{build_dnam_container, update_header_in_place};
pub use super::strip::strip_container_to_tier;

#[cfg(test)]
mod tests {
    use super::super::{
        ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_PRO_END, ARENA_OFFSET_ULTRA_END,
    };
    use super::*;
    use crate::provenance::DefaultCryptoProvider;
    use crate::schema::{
        ControlBusConfig, DatasetProvenance, ExtractedPhysics, IoConfig, ModelMetadata,
        RouterConfig, SlimmableTiersConfig, Stage1PreFxConfig, Stage2CascadeConfig,
        Stage3CabinetKautzConfig, Stage4SpatialTankConfig, StemProvenance, StemRegistryEntry,
        TensorSliceEntry, TierProfile,
    };
    use alloc::collections::BTreeMap;
    use alloc::vec::Vec;
    use alloc::{format, vec};

    fn make_base_header(name: &str) -> DnamJsonHeader {
        let mk_tier = |ch: u8, tank: u8, rank: u8, end_off: u32| TierProfile {
            feature_bus_dim: (ch as u16) * 16,
            channels: ch,
            early_fir_taps: tank * 4,
            kautz_modes: tank,
            tank_lines: tank,
            head_svd_rank: rank,
            loudness_lufs: -20.5,
            output_trim_db: 0.0,
            validation_esr_db: -42.0,
            arena_end_byte_offset: end_off,
            prefix_crc32c: 0,
        };
        DnamJsonHeader {
            version: "1.0.0".to_string(),
            container_kind: "dnam".to_string(),
            architecture: "DNAM_TORA".to_string(),
            sample_rate_train_hz: 48_000,
            metadata: ModelMetadata {
                name: name.to_string(),
                author: Some("UnitTest".to_string()),
                description: None,
                license: Some("MIT".to_string()),
                created_at: "2026-10-23T00:00:00Z".to_string(),
                input_reference_dbfs: -20.5,
                provenance: DatasetProvenance::default(),
                distribution_policy: None,
                signatures: None,
            },
            user_metadata: BTreeMap::new(),
            io_config: IoConfig::default(),
            slimmable_tiers: SlimmableTiersConfig {
                lite: mk_tier(4, 4, 4, ARENA_OFFSET_LITE_END),
                full: mk_tier(8, 8, 4, ARENA_OFFSET_FULL_END),
                pro: mk_tier(12, 8, 4, ARENA_OFFSET_PRO_END),
                ultra: mk_tier(16, 16, 8, ARENA_OFFSET_ULTRA_END),
            },
            controls_schema: Vec::new(),
            presets: Vec::new(),
            control_bus: ControlBusConfig::default(),
            stage0_router: RouterConfig::default(),
            stage1_pre: Stage1PreFxConfig::default(),
            stage2_cascade: Stage2CascadeConfig::default(),
            stage3_cabinet: Stage3CabinetKautzConfig::default(),
            stage4_tank: Stage4SpatialTankConfig::default(),
            tensor_directory: BTreeMap::new(),
            stems_registry: None,
            audit_report: None,
        }
    }

    #[test]
    fn test_container_build_view_strip_and_in_place_update() {
        let mut hdr = make_base_header("Test");
        assert!(build_dnam_container(&mut hdr, &[0u8; 128], &[0u8; 128], false).is_err());

        let lite_arena = vec![0u8; ARENA_OFFSET_LITE_END as usize];
        let full_arena = vec![0u8; ARENA_OFFSET_FULL_END as usize];
        let pro_arena = vec![0u8; ARENA_OFFSET_PRO_END as usize];
        let ultra_arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];

        let lite_bytes = build_dnam_container(&mut hdr, &lite_arena, &lite_arena, false).unwrap();
        let full_bytes = build_dnam_container(&mut hdr, &full_arena, &full_arena, false).unwrap();
        let pro_bytes = build_dnam_container(&mut hdr, &pro_arena, &pro_arena, false).unwrap();
        let mut ultra_bytes =
            build_dnam_container(&mut hdr, &ultra_arena, &ultra_arena, false).unwrap();

        let crypto = DefaultCryptoProvider::default();
        assert!(DnamContainerView::from_bytes(&lite_bytes, &crypto).is_ok());
        assert!(DnamContainerView::from_bytes(&full_bytes, &crypto).is_ok());
        assert!(DnamContainerView::from_bytes(&pro_bytes, &crypto).is_ok());
        assert!(DnamContainerView::from_bytes(&ultra_bytes, &crypto).is_ok());
        assert!(DnamContainerView::from_bytes(&ultra_bytes[..100], &crypto).is_err());

        // Test max_supported_tier prefix loading on unstripped Ultra container
        let capped_pro_view =
            DnamContainerView::from_bytes_with_max_tier(&ultra_bytes, &crypto, Tier::Pro).unwrap();
        assert_eq!(capped_pro_view.preamble.highest_tier(), Tier::Pro);
        assert_eq!(
            capped_pro_view.arena_storage.as_bytes().len(),
            ARENA_OFFSET_PRO_END as usize
        );

        // Test strip_container_to_tier
        let stripped_pro = strip_container_to_tier(&ultra_bytes, Tier::Pro).unwrap();
        let stripped_view = DnamContainerView::from_bytes(&stripped_pro, &crypto).unwrap();
        assert_eq!(stripped_view.preamble.highest_tier(), Tier::Pro);
        assert_eq!(
            stripped_view.arena_storage.as_bytes().len(),
            ARENA_OFFSET_PRO_END as usize
        );

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
        preamble.arena_crc32c_pro ^= 1;
        assert!(DnamContainerView::verify_arena_prefix_crcs(&preamble, raw_arena).is_err());
        preamble.arena_crc32c_pro ^= 1;
        preamble.arena_crc32c_ultra ^= 1;
        assert!(DnamContainerView::verify_arena_prefix_crcs(&preamble, raw_arena).is_err());

        // dnam-ckpt container with stems_registry
        hdr.container_kind = "dnam-ckpt".to_string();
        let dummy_slice = TensorSliceEntry {
            arena: 4,
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
            stem_arena4_crc32c: super::super::crc32c_compute(&[0u8; 64]),
            user_metadata: BTreeMap::new(),
            extracted_physics: Some(ExtractedPhysics {
                delays_ms: vec![10.0],
                lfo_freqs_hz: vec![0.5],
                decay_gains_lin: vec![0.7],
                detected_grain_size_ms: 0.0,
                detected_pitch_shift_semitones: 0.0,
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

        // Test stripping to Full and Lite
        let stripped_full = strip_container_to_tier(&ultra_bytes, Tier::Full).unwrap();
        let full_v = DnamContainerView::from_bytes(&stripped_full, &crypto).unwrap();
        assert_eq!(full_v.preamble.highest_tier(), Tier::Full);

        let stripped_lite = strip_container_to_tier(&ultra_bytes, Tier::Lite).unwrap();
        let lite_v = DnamContainerView::from_bytes(&stripped_lite, &crypto).unwrap();
        assert_eq!(lite_v.preamble.highest_tier(), Tier::Lite);

        // Test capped view at Full and Lite
        let capped_full =
            DnamContainerView::from_bytes_with_max_tier(&ultra_bytes, &crypto, Tier::Full).unwrap();
        assert_eq!(capped_full.preamble.highest_tier(), Tier::Full);
        let capped_lite =
            DnamContainerView::from_bytes_with_max_tier(&ultra_bytes, &crypto, Tier::Lite).unwrap();
        assert_eq!(capped_lite.preamble.highest_tier(), Tier::Lite);

        // Test valid in-place header update
        let mut mod_hdr = make_base_header("Modified");
        mod_hdr.metadata.name = "NewName".to_string();
        update_header_in_place(&mut ultra_bytes, &mod_hdr).unwrap();
        let updated_v = DnamContainerView::from_bytes(&ultra_bytes, &crypto).unwrap();
        assert_eq!(updated_v.header.metadata.name, "NewName");

        // Test forbidden calib stats on DNAM_MAGIC
        let flags_val = u16::from_le_bytes([ultra_bytes[8], ultra_bytes[9]])
            | super::super::FLAG_HAS_CALIB_STATS;
        ultra_bytes[8..10].copy_from_slice(&flags_val.to_le_bytes());
        // recompute header crc
        let pr = DnamPreamble::from_bytes(&ultra_bytes).unwrap();
        let off = pr.tensor_arena_offset as usize;
        let c = compute_header_crc32c(&ultra_bytes, off);
        ultra_bytes[60..64].copy_from_slice(&c.to_le_bytes());
        assert!(DnamContainerView::from_bytes(&ultra_bytes, &crypto).is_err());

        // Error branch tests:
        // 1. Buffer too small for tensor arena
        let truncated_len = ultra_bytes.len() - 100;
        assert!(DnamContainerView::from_bytes(&ultra_bytes[..truncated_len], &crypto).is_err());

        // 2. Header CRC mismatch
        let mut bad_crc = ultra_bytes.clone();
        bad_crc[60] ^= 0x55;
        assert!(DnamContainerView::from_bytes(&bad_crc, &crypto).is_err());

        // 3. Invalid JSON header
        let mut bad_json = ultra_bytes.clone();
        bad_json[64..72].copy_from_slice(b"invalid_");
        let c_bad_json = compute_header_crc32c(&bad_json, off);
        bad_json[60..64].copy_from_slice(&c_bad_json.to_le_bytes());
        assert!(DnamContainerView::from_bytes(&bad_json, &crypto).is_err());

        // 4. Schema version mismatch
        let mut ver_mismatch = ultra_bytes.clone();
        ver_mismatch[4] = 99; // corrupt version_major
        let c_ver = compute_header_crc32c(&ver_mismatch, off);
        ver_mismatch[60..64].copy_from_slice(&c_ver.to_le_bytes());
        assert!(DnamContainerView::from_bytes(&ver_mismatch, &crypto).is_err());

        // 5. DNSL sealed magic with corrupt container
        let mut corrupt_dnsl = [0u8; 128];
        corrupt_dnsl[0..4].copy_from_slice(&DNSL_MAGIC);
        assert!(DnamContainerView::from_bytes(&corrupt_dnsl, &crypto).is_err());
    }
}
