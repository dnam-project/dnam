//! Zero-copy canonical `v2.0` container view (`DnamContainerView`).

use alloc::string::ToString;
use alloc::vec::Vec;

use crate::control::{
    ControlChannelRecord, ControlClampDescriptor, ControlLatentProjection, Cp1ContributorSlot,
};
use crate::memory::{AlignedBuffer, ArenaStorage, CacheLine64};
use crate::provenance::{
    CryptoProvider, DNSL_MAGIC, VerificationStatus, chacha20_xor_arena_blocks, decode_hex_array,
    enforce_distribution_policy, unseal_dnam_container, verify_container_provenance,
};
use crate::schema::{DnamError, DnamJsonHeader, DnamProtectionProfile};
use crate::types::Tier;

use super::preamble::DnamPreamble;
use super::wire::{
    CP1_MAGIC, V2_PREAMBLE_SIZE, decode_cp1_shared_stream_into_slice,
    parse_and_validate_dmu1_frame, parse_dcrc_record, parse_strict_json_object_bytes,
    validate_mutable_json_schema, validate_v2_dnam_layout_with_max_profile_no_alloc,
};
use super::{
    CONTAINER_VERSION_MAJOR, CONTAINER_VERSION_MINOR, DNAM_MAGIC, FLAG_ENCRYPTED_ARENA,
    FLAG_HAS_FULL_DELTA, FLAG_HAS_PRO_DELTA, FLAG_HAS_SIGNATURES, FLAG_HAS_ULTRA_DELTA,
    crc32c_update,
};

const FORBIDDEN_DCRC_RESERVED_FLAG: u16 = 1 << 12;

/// Decoded shared control descriptors, `CP1\0` contributor slots, and unified 128-byte control records.
pub type SharedControlsExtent = (
    Vec<ControlClampDescriptor>,
    Vec<Cp1ContributorSlot>,
    Vec<ControlChannelRecord>,
);

/// Decodes the shared controls extent (`ControlClampDescriptor[0..P]` + optional `CP1\0` stream)
/// for a runtime `.dnam` header.
pub fn decode_shared_controls_extent(
    shared_slice: &[u8],
    header: &DnamJsonHeader,
) -> Result<SharedControlsExtent, DnamError> {
    let p_ctrls = header.controls_schema.len();
    let desc_bytes_len = p_ctrls.checked_mul(64).ok_or_else(|| {
        DnamError::SchemaValidationError("controls_schema count overflow".to_string())
    })?;
    let starts_with_cp1 = shared_slice.len() >= 4 && shared_slice[0..4] == CP1_MAGIC;
    let (clamp_descriptors, cp1_slice) =
        if p_ctrls > 0 && shared_slice.len() >= desc_bytes_len && !starts_with_cp1 {
            let mut descs = Vec::with_capacity(p_ctrls);
            for p in 0..p_ctrls {
                let off = p * 64;
                let mut desc_bytes = [0u8; 64];
                desc_bytes.copy_from_slice(&shared_slice[off..off + 64]);
                descs.push(ControlClampDescriptor::from_bytes(&desc_bytes));
            }
            (descs, &shared_slice[desc_bytes_len..])
        } else {
            (
                header.compile_clamp_descriptors().unwrap_or_default(),
                shared_slice,
            )
        };
    let mut cp1_contributors = Vec::new();
    if cp1_slice.len() >= 8 && cp1_slice[0..4] == CP1_MAGIC {
        let count = u16::from_le_bytes([cp1_slice[4], cp1_slice[5]]) as usize;
        cp1_contributors.resize(count, Cp1ContributorSlot::zero());
        decode_cp1_shared_stream_into_slice(cp1_slice, &mut cp1_contributors)?;
    }
    let control_records = clamp_descriptors
        .iter()
        .copied()
        .map(|clamp| ControlChannelRecord {
            clamp,
            projection: ControlLatentProjection::new(),
        })
        .collect();
    Ok((clamp_descriptors, cp1_contributors, control_records))
}

/// Zero-copy or decrypted view over a validated canonical `v2.0` `.dnam` or `.dnam-seal` (`b"DNSL"`) container.
#[derive(Debug)]
pub struct DnamContainerView<'a> {
    /// Logical 64-byte binary preamble summary.
    pub preamble: DnamPreamble,
    /// Parsed UTF-8 JSON header (`no_std` `serde` + `alloc`) with `DMU1` mutable fields merged.
    pub header: DnamJsonHeader,
    /// Cryptographic provenance status (`Unsigned`, `CreatorVerified`, or `MarketplaceVerified`).
    pub verification_status: VerificationStatus,
    /// Resolved container protection profile (`Open`, `DeviceAllowlisted`, or `EnclaveSealed`).
    pub protection_profile: DnamProtectionProfile,
    /// Compiled 64-byte `ControlClampDescriptor[0..P]` table from Region 4 `shared_extent`.
    pub clamp_descriptors: Vec<ControlClampDescriptor>,
    /// Decoded `CP1\0` variable-length contributor slots from Region 4 `shared_extent` (if present).
    pub cp1_contributors: Vec<Cp1ContributorSlot>,
    /// Unified 128-byte `ControlChannelRecord[0..P]` table derived from `clamp_descriptors`.
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
                clamp_descriptors: owned_view.clamp_descriptors,
                cp1_contributors: owned_view.cp1_contributors,
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
        if bytes.len() < V2_PREAMBLE_SIZE {
            return Err(DnamError::BufferTooSmall {
                expected: V2_PREAMBLE_SIZE,
                actual: bytes.len(),
            });
        }
        let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if magic != DNAM_MAGIC {
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

        let layout = validate_v2_dnam_layout_with_max_profile_no_alloc(bytes, 2)?;
        let auth_slice = &bytes[layout.auth_extent.0..layout.auth_extent.0 + layout.auth_extent.1];
        let maybe_dcrc = parse_dcrc_record(auth_slice)?;
        if let Some(dcrc) = maybe_dcrc {
            let calc_prefix_crc = crc32c_update(0, &bytes[..layout.auth_extent.0]);
            if calc_prefix_crc != dcrc.crc_prefix {
                return Err(DnamError::Crc32cMismatch {
                    region: "header",
                    expected: dcrc.crc_prefix,
                    computed: calc_prefix_crc,
                });
            }
            if (dcrc.flags_hint & FORBIDDEN_DCRC_RESERVED_FLAG) != 0 {
                return Err(DnamError::SchemaValidationError(
                    "reserved flags_hint bit 12 is forbidden in runtime .dnam containers"
                        .to_string(),
                ));
            }
        }

        let json_slice =
            &bytes[layout.header_extent.0..layout.header_extent.0 + layout.header_extent.1];
        let raw_val = parse_strict_json_object_bytes(json_slice)?;
        if raw_val
            .get("metadata")
            .and_then(|m| m.get("container_kind"))
            .and_then(|v| v.as_str())
            .is_some_and(|k| k != "runtime")
        {
            return Err(DnamError::PreviewContainerRejected);
        }
        let mut header: DnamJsonHeader = serde_json::from_value(raw_val)
            .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?;

        let mutable_slice =
            &bytes[layout.mutable_extent.0..layout.mutable_extent.0 + layout.mutable_extent.1];
        let dmu_val = parse_and_validate_dmu1_frame(mutable_slice)?;
        validate_mutable_json_schema(&dmu_val)?;
        if let Some(presets_val) = dmu_val.get("presets") {
            header.presets = serde_json::from_value(presets_val.clone())
                .map_err(|e| DnamError::SchemaValidationError(e.to_string()))?;
        }
        if let Some(um_val) = dmu_val.get("user_metadata") {
            header.user_metadata = serde_json::from_value(um_val.clone())
                .map_err(|e| DnamError::SchemaValidationError(e.to_string()))?;
        }
        if let Some(ctrl_um_obj) = dmu_val
            .get("control_user_metadata")
            .and_then(|v| v.as_object())
        {
            for ctrl in &mut header.controls_schema {
                if let Some(entry_val) = ctrl_um_obj.get(&ctrl.id) {
                    ctrl.user_metadata = serde_json::from_value(entry_val.clone())
                        .map_err(|e| DnamError::SchemaValidationError(e.to_string()))?;
                }
            }
        }

        header.validate()?;
        let schema_ver = header.schema_version()?;
        schema_ver.ensure_matches_preamble(version_major, version_minor)?;

        let mut preamble = DnamPreamble::from_bytes(bytes)?;
        let retained_tier = Tier::from_u8(layout.retained_tier).min(max_supported_tier);
        let tier_end = layout.tier_ends[retained_tier as usize];
        let raw_arena = &bytes[layout.tier_start..tier_end];

        if retained_tier < Tier::from_u8(layout.retained_tier) {
            if retained_tier < Tier::Ultra {
                preamble.flags &= !FLAG_HAS_ULTRA_DELTA;
            }
            if retained_tier < Tier::Pro {
                preamble.flags &= !FLAG_HAS_PRO_DELTA;
            }
            if retained_tier < Tier::Full {
                preamble.flags &= !FLAG_HAS_FULL_DELTA;
            }
            preamble.tensor_arena_byte_len = raw_arena.len() as u32;
        }

        let dcrc_encrypted = maybe_dcrc.is_some_and(|d| (d.flags_hint & FLAG_ENCRYPTED_ARENA) != 0);
        let receipt_has_nonce = header
            .metadata
            .signatures
            .as_ref()
            .and_then(|s| s.marketplace_receipt.as_ref())
            .is_some_and(|r| r.arena_nonce_96.is_some());
        let needs_arena_decrypt = dcrc_encrypted
            || (!is_enclave_sealed_wrapper && layout.protection_profile == 1 && receipt_has_nonce);

        let arena_storage =
            if needs_arena_decrypt {
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
                let mut owned = AlignedBuffer::<CacheLine64>::new_zeroed(raw_arena.len() / 64);
                owned.as_bytes_mut().copy_from_slice(raw_arena);
                chacha20_xor_arena_blocks(owned.as_bytes_mut(), &cek, &nonce_96, 0)?;
                ArenaStorage::from_owned_buffer(owned)
            } else if is_enclave_sealed_wrapper {
                ArenaStorage::from_slice_owned(raw_arena)
            } else {
                ArenaStorage::from_slice_or_copy(raw_arena)
            };

        let has_authorized_devices = header
            .metadata
            .signatures
            .as_ref()
            .and_then(|s| s.marketplace_receipt.as_ref())
            .is_some_and(|r| !r.authorized_devices.is_empty());
        let protection_profile = if is_enclave_sealed_wrapper {
            DnamProtectionProfile::EnclaveSealed
        } else {
            match layout.protection_profile {
                0 if dcrc_encrypted || has_authorized_devices => {
                    DnamProtectionProfile::DeviceAllowlisted
                }
                0 => DnamProtectionProfile::Open,
                1 => DnamProtectionProfile::DeviceAllowlisted,
                2 => DnamProtectionProfile::EnclaveSealed,
                _ => {
                    return Err(DnamError::SchemaValidationError(
                        "invalid v2 protection_profile".to_string(),
                    ));
                }
            }
        };
        enforce_distribution_policy(&header, protection_profile)?;

        let plain_bytes = arena_storage.as_bytes();
        if maybe_dcrc.is_some() {
            Self::verify_arena_prefix_crcs(&preamble, plain_bytes)?;
        }

        let shared_slice =
            &bytes[layout.shared_extent.0..layout.shared_extent.0 + layout.shared_extent.1];
        let (clamp_descriptors, cp1_contributors, control_records) =
            decode_shared_controls_extent(shared_slice, &header)?;

        let verification_status = if auth_slice.starts_with(b"DAU1") || layout.auth_kind > 0 {
            let env = crate::provenance::Dau1Envelope::from_bytes(auth_slice)?;
            if env.auth_kind != layout.auth_kind {
                return Err(DnamError::CryptoVerificationFailed(
                    "DAU1 auth_kind mismatch with preamble",
                ));
            }
            let shared_framing = crate::provenance::build_plaintext_shared_v1(
                &bytes[0..64],
                &bytes[64..192],
                shared_slice,
            );
            let hdr_sha = crate::provenance::sha256_digest(json_slice);
            let mut tier_slices_vec = Vec::with_capacity((retained_tier as usize) + 1);
            let mut prev_t = layout.tier_start;
            for t_idx in 0..=(retained_tier as usize) {
                let end_t = layout.tier_ends[t_idx];
                tier_slices_vec.push(&bytes[prev_t..end_t]);
                prev_t = end_t;
            }
            if env.auth_kind == 0 {
                if env.creator_max_tier != layout.maximum_tier {
                    return Err(DnamError::CryptoVerificationFailed(
                        "DAU1 creator_max_tier mismatch with preamble",
                    ));
                }
                let prefixes = crate::provenance::compute_plaintext_prefixes_v1(
                    b"DNAM-MASTER-PREFIX-V1",
                    &hdr_sha,
                    &shared_framing,
                    &tier_slices_vec,
                )?;
                for (t_idx, pfx) in prefixes.iter().enumerate() {
                    if env.master_prefixes.get(t_idx) != Some(pfx) {
                        return Err(DnamError::CryptoVerificationFailed(
                            "DAU1 master tier prefix digest mismatch",
                        ));
                    }
                }
            }
            let creator_manifest = crate::provenance::build_creator_manifest_v1(
                &env.creator_pubkey,
                &hdr_sha,
                &env.master_prefixes,
            )?;
            crypto.ed25519_verify(
                &env.creator_pubkey,
                &creator_manifest,
                &env.creator_signature,
            )?;
            if env.auth_kind == 1 {
                let m_pub = env
                    .marketplace_pubkey
                    .ok_or(DnamError::CryptoVerificationFailed(
                        "missing DAU1 marketplace_pubkey",
                    ))?;
                if let Some(pinned_mp) = crypto.marketplace_pubkey()
                    && &m_pub != pinned_mp
                {
                    return Err(DnamError::CryptoVerificationFailed(
                        "DAU1 marketplace_pubkey mismatch against pinned trusted marketplace key",
                    ));
                }
                let b_max = env
                    .buyer_max_tier
                    .ok_or(DnamError::CryptoVerificationFailed(
                        "missing DAU1 buyer_max_tier",
                    ))?;
                if b_max != layout.maximum_tier || b_max > env.creator_max_tier {
                    return Err(DnamError::CryptoVerificationFailed(
                        "DAU1 buyer_max_tier mismatch with preamble or creator_max_tier",
                    ));
                }
                let m_sig =
                    env.marketplace_signature
                        .ok_or(DnamError::CryptoVerificationFailed(
                            "missing DAU1 marketplace_signature",
                        ))?;
                let buyer_prefixes = crate::provenance::compute_plaintext_prefixes_v1(
                    b"DNAM-BUYER-PREFIX-V1",
                    &hdr_sha,
                    &shared_framing,
                    &tier_slices_vec,
                )?;
                for (t_idx, bpfx) in buyer_prefixes.iter().enumerate() {
                    if env.buyer_prefixes.get(t_idx) != Some(bpfx) {
                        return Err(DnamError::CryptoVerificationFailed(
                            "DAU1 buyer tier prefix digest mismatch",
                        ));
                    }
                }
                let creator_signed_sha = crate::provenance::compute_creator_signed_manifest_sha256(
                    &creator_manifest,
                    &env.creator_signature,
                );
                let policy_sha =
                    crate::provenance::compute_license_policy_sha256(&env.policy_bytes);
                let buyer_receipt = crate::provenance::build_buyer_receipt_v1(
                    &m_pub,
                    &creator_signed_sha,
                    &policy_sha,
                    layout.protection_profile,
                    &env.buyer_prefixes,
                    None,
                    &[],
                )?;
                crypto.ed25519_verify(&m_pub, &buyer_receipt, &m_sig)?;
                VerificationStatus::MarketplaceVerified {
                    signer_id: header.metadata.author.clone().unwrap_or_default(),
                    marketplace_id: alloc::string::String::from("dnam-marketplace"),
                    license_id: alloc::string::String::from("dau1"),
                    license_tier: Tier::from_u8(b_max).as_str().to_string(),
                }
            } else {
                VerificationStatus::CreatorVerified {
                    signer_id: header.metadata.author.clone().unwrap_or_default(),
                    pubkey_ed25519: env.creator_pubkey,
                }
            }
        } else {
            verify_container_provenance(&header, preamble.has_signatures(), plain_bytes, crypto)?
        };

        if !matches!(verification_status, VerificationStatus::Unsigned) {
            preamble.flags |= FLAG_HAS_SIGNATURES;
        }
        preamble.num_controls = header.controls_schema.len().min(u16::MAX as usize) as u16;
        preamble.tail_time_ms = header.io_config.tail_time_ms;

        Ok(Self {
            preamble,
            header,
            verification_status,
            protection_profile,
            clamp_descriptors,
            cp1_contributors,
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
}

pub use super::builder::{build_dnam_container, update_header_in_place};
pub use super::strip::strip_container_to_tier;

#[cfg(test)]
mod tests {
    use super::super::{
        ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_PRO_END, ARENA_OFFSET_ULTRA_END,
    };
    use super::*;
    use crate::container::validate_v2_open_dnam_layout;
    use crate::provenance::DefaultCryptoProvider;
    use crate::schema::{
        ControlBusConfig, DatasetProvenance, IoConfig, ModelMetadata, PresetDef, RouterConfig,
        SlimmableTiersConfig, Stage1PreFxConfig, Stage2CascadeConfig, Stage3CabinetKautzConfig,
        Stage4SpatialTankConfig, TierProfile,
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
            version: "2.0.0".to_string(),
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
        assert_eq!(ultra_bytes.len() % 64, 0);

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

        let layout = validate_v2_open_dnam_layout(&ultra_bytes).unwrap();
        ultra_bytes[layout.header_extent.0] ^= 0xFF;
        assert!(DnamContainerView::from_bytes(&ultra_bytes, &crypto).is_err());
        ultra_bytes[layout.header_extent.0] ^= 0xFF;

        let mut preamble = DnamPreamble::from_bytes(&ultra_bytes).unwrap();
        let arena_off = preamble.tensor_arena_offset as usize;
        let raw_arena = &ultra_bytes[arena_off..layout.tier_ends[3]];
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

        assert!(update_header_in_place(&mut ultra_bytes[..64], &hdr).is_err());
        let mut huge_hdr = hdr.clone();
        for i in 0..200 {
            huge_hdr
                .user_metadata
                .insert(format!("key_{i}"), serde_json::json!("x".repeat(50)));
        }
        assert!(update_header_in_place(&mut ultra_bytes, &huge_hdr).is_err());

        // Immutable field edits must be rejected by update_header_in_place
        let mut immut_edit = hdr.clone();
        immut_edit.metadata.name = "TamperedName".to_string();
        assert!(update_header_in_place(&mut ultra_bytes, &immut_edit).is_err());

        // Valid mutable DMU1 in-place edit (presets + user_metadata)
        let mut mut_hdr = hdr.clone();
        mut_hdr
            .user_metadata
            .insert("tag".to_string(), serde_json::json!("studio"));
        mut_hdr.presets.push(PresetDef {
            id: "clean".to_string(),
            name: "Clean".to_string(),
            description: None,
            controls: BTreeMap::new(),
            user_metadata: BTreeMap::new(),
        });
        update_header_in_place(&mut ultra_bytes, &mut_hdr).unwrap();
        let updated_v = DnamContainerView::from_bytes(&ultra_bytes, &crypto).unwrap();
        assert_eq!(
            updated_v.header.user_metadata.get("tag"),
            Some(&serde_json::json!("studio"))
        );
        assert_eq!(updated_v.header.presets.len(), 1);

        // Test stripping to Full and Lite preserves DMU1 mutable edits
        let stripped_full = strip_container_to_tier(&ultra_bytes, Tier::Full).unwrap();
        let full_v = DnamContainerView::from_bytes(&stripped_full, &crypto).unwrap();
        assert_eq!(full_v.preamble.highest_tier(), Tier::Full);
        assert_eq!(full_v.header.presets.len(), 1);

        let stripped_lite = strip_container_to_tier(&ultra_bytes, Tier::Lite).unwrap();
        let lite_v = DnamContainerView::from_bytes(&stripped_lite, &crypto).unwrap();
        assert_eq!(lite_v.preamble.highest_tier(), Tier::Lite);
        assert_eq!(lite_v.header.presets.len(), 1);

        // Test capped view at Full and Lite
        let capped_full =
            DnamContainerView::from_bytes_with_max_tier(&ultra_bytes, &crypto, Tier::Full).unwrap();
        assert_eq!(capped_full.preamble.highest_tier(), Tier::Full);
        let capped_lite =
            DnamContainerView::from_bytes_with_max_tier(&ultra_bytes, &crypto, Tier::Lite).unwrap();
        assert_eq!(capped_lite.preamble.highest_tier(), Tier::Lite);

        // Error branch tests:
        // 1. Forbidden reserved flag in DCRC
        let mut bad_calib = ultra_bytes.clone();
        let flags_off = layout.auth_extent.0 + 26;
        let cur_flags = u16::from_le_bytes([bad_calib[flags_off], bad_calib[flags_off + 1]])
            | FORBIDDEN_DCRC_RESERVED_FLAG;
        bad_calib[flags_off..flags_off + 2].copy_from_slice(&cur_flags.to_le_bytes());
        assert!(DnamContainerView::from_bytes(&bad_calib, &crypto).is_err());

        // 2. Truncated buffer
        let truncated_len = ultra_bytes.len() - 100;
        assert!(DnamContainerView::from_bytes(&ultra_bytes[..truncated_len], &crypto).is_err());

        // 3. DCRC prefix CRC mismatch
        let mut bad_crc = ultra_bytes.clone();
        bad_crc[layout.auth_extent.0 + 4] ^= 0x55;
        assert!(DnamContainerView::from_bytes(&bad_crc, &crypto).is_err());

        // 4. Invalid JSON header (with recomputed DCRC prefix CRC)
        let mut bad_json = ultra_bytes.clone();
        bad_json[layout.header_extent.0..layout.header_extent.0 + 8].copy_from_slice(b"invalid_");
        let recrc = crc32c_update(0, &bad_json[..layout.auth_extent.0]);
        bad_json[layout.auth_extent.0 + 4..layout.auth_extent.0 + 8]
            .copy_from_slice(&recrc.to_le_bytes());
        assert!(DnamContainerView::from_bytes(&bad_json, &crypto).is_err());

        // 5. Unsupported container version
        let mut ver_mismatch = ultra_bytes.clone();
        ver_mismatch[4] = 99;
        assert!(matches!(
            DnamContainerView::from_bytes(&ver_mismatch, &crypto),
            Err(DnamError::UnsupportedVersion {
                major: 99,
                minor: 0
            })
        ));

        // 6. DNSL sealed magic with corrupt container
        let mut corrupt_dnsl = [0u8; 128];
        corrupt_dnsl[0..4].copy_from_slice(&DNSL_MAGIC);
        assert!(DnamContainerView::from_bytes(&corrupt_dnsl, &crypto).is_err());

        // 7. v2 DAU1 auth_kind=1 enforces pinned marketplace_pubkey on CryptoProvider
        #[cfg(feature = "crypto")]
        {
            use crate::container::wire::build_v2_open_dnam_container;
            use crate::provenance::{
                Dau1Envelope, RustCryptoProvider, build_buyer_receipt_v1,
                build_creator_manifest_v1, build_plaintext_shared_v1,
                compute_creator_signed_manifest_sha256, compute_license_policy_sha256,
                compute_plaintext_prefixes_v1, sha256_digest,
            };

            let v2_hdr_json = serde_json::to_vec(&make_base_header("Dau1Market")).unwrap();
            let hdr_sha = sha256_digest(&v2_hdr_json);
            let t0 = [0u8; 10_752];
            let tiers: [&[u8]; 1] = [&t0];
            let c_seed = [0x31u8; 32];
            let m_seed = [0x42u8; 32];
            let c_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&c_seed);
            let m_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&m_seed);
            let policy = b"tier=lite;offline=1".to_vec();
            let dummy_dau = Dau1Envelope {
                auth_kind: 1,
                creator_pubkey: c_pk,
                creator_max_tier: 0,
                master_prefixes: alloc::vec![[0u8; 32]],
                creator_signature: [0u8; 64],
                marketplace_pubkey: Some(m_pk),
                policy_bytes: policy.clone(),
                buyer_max_tier: Some(0),
                buyer_prefixes: alloc::vec![[0u8; 32]],
                marketplace_signature: Some([0u8; 64]),
            }
            .to_bytes()
            .unwrap();
            let probe =
                build_v2_open_dnam_container(&v2_hdr_json, &[0u8; 64], &dummy_dau, &tiers, None, 1)
                    .unwrap();
            let shared_f = build_plaintext_shared_v1(&probe[0..64], &probe[64..192], &[0u8; 64]);
            let m_pfxs = compute_plaintext_prefixes_v1(
                b"DNAM-MASTER-PREFIX-V1",
                &hdr_sha,
                &shared_f,
                &tiers,
            )
            .unwrap();
            let b_pfxs =
                compute_plaintext_prefixes_v1(b"DNAM-BUYER-PREFIX-V1", &hdr_sha, &shared_f, &tiers)
                    .unwrap();
            let c_man = build_creator_manifest_v1(&c_pk, &hdr_sha, &m_pfxs).unwrap();
            let c_sig = RustCryptoProvider::ed25519_sign(&c_seed, &c_man);
            let c_id = compute_creator_signed_manifest_sha256(&c_man, &c_sig);
            let pol_sha = compute_license_policy_sha256(&policy);
            let b_rec =
                build_buyer_receipt_v1(&m_pk, &c_id, &pol_sha, 0, &b_pfxs, None, &[]).unwrap();
            let m_sig = RustCryptoProvider::ed25519_sign(&m_seed, &b_rec);
            let real_dau = Dau1Envelope {
                auth_kind: 1,
                creator_pubkey: c_pk,
                creator_max_tier: 0,
                master_prefixes: m_pfxs,
                creator_signature: c_sig,
                marketplace_pubkey: Some(m_pk),
                policy_bytes: policy,
                buyer_max_tier: Some(0),
                buyer_prefixes: b_pfxs,
                marketplace_signature: Some(m_sig),
            }
            .to_bytes()
            .unwrap();
            let signed_v2 =
                build_v2_open_dnam_container(&v2_hdr_json, &[0u8; 64], &real_dau, &tiers, None, 1)
                    .unwrap();
            let pinned_ok = RustCryptoProvider::new().with_marketplace_pubkey(m_pk);
            let pinned_bad = RustCryptoProvider::new().with_marketplace_pubkey([0x99u8; 32]);
            assert!(DnamContainerView::from_bytes(&signed_v2, &pinned_ok).is_ok());
            assert!(matches!(
                DnamContainerView::from_bytes(&signed_v2, &pinned_bad),
                Err(DnamError::CryptoVerificationFailed(_))
            ));
        }
    }
}
