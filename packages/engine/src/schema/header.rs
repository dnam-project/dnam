//! Canonical `.dnam` and `.dnam-ckpt` UTF-8 JSON Header (`DnamJsonHeader`), subsystem configurations,
//! tensor directory (`TensorSliceEntry`), stem registry (`StemRegistryEntry`), and binary clamp compiler.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use crate::control::{ConditionalClampRule, ControlClampDescriptor};

use super::controls::{ControlKind, ControlSchemaDef, PresetDef};
use super::error::DnamError;
use super::signatures::{DatasetProvenance, DistributionPolicy, ModelSignatures};

pub use super::io_and_tiers::{
    BlendMode, IoConfig, LevelingConfig, LevelingMetric, ModelCapabilities, SlimmableTiersConfig,
    TierProfile, default_blend_one, default_head_svd_rank, default_input_ref_dbfs, default_inputs,
    default_output_mode, default_pro_tier_profile, default_true,
};
pub use super::json_wire::TensorSliceEntry;
pub use super::stages::{
    ControlBusConfig, RouterConfig, Stage1PreFxConfig, Stage2CascadeConfig,
    Stage3CabinetKautzConfig, Stage4SpatialTankConfig, default_ac_pole, default_adaa_mode,
    default_allpass_delays, default_cascade_stages, default_dc_pole, default_delay_storage,
    default_envelopes, default_fir_taps, default_fm_rank, default_gate_threshold_db,
    default_interp, default_kautz_modals, default_lfo_harmonics, default_lfos,
    default_loop_ceiling, default_modulation_bus_dim, default_router_ports,
    default_sub_crossover_hz, default_switch_xfade, default_trust_radius, default_tzf_tap,
    deserialize_allpass_delays,
};

/// Semantic version (`major.minor.patch`) of the `.dnam` and `.dnam-ckpt` container schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SchemaVersion {
    /// Breaking container wire format or tensor layout version (`CONTAINER_VERSION_MAJOR`).
    pub major: u16,
    /// Backwards-compatible additive schema version (`CONTAINER_VERSION_MINOR`).
    pub minor: u16,
    /// Backwards-compatible patch version (`0`).
    pub patch: u16,
}

impl SchemaVersion {
    /// Initial `1.0.0` schema release.
    pub const V1_0_0: Self = Self::new(1, 0, 0);

    /// `1.1.0` schema release adding optional `.dnam-ckpt` Arena 4 (`FLAG_HAS_CALIB_STATS`).
    pub const V1_1_0: Self = Self::new(1, 1, 0);

    /// Canonical `2.0.0` schema release (`ref[container.dnam.formats]`).
    pub const V2_0_0: Self = Self::new(2, 0, 0);

    /// Current container schema version supported by this engine build.
    pub const CURRENT: Self = Self::new(
        crate::container::CONTAINER_VERSION_MAJOR,
        crate::container::CONTAINER_VERSION_MINOR,
        0,
    );

    /// Constructs a new [`SchemaVersion`].
    #[inline]
    pub const fn new(major: u16, minor: u16, patch: u16) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Parses a semantic version string (`"major.minor.patch"` or `"major.minor"`).
    pub fn parse(version_str: &str) -> Result<Self, DnamError> {
        let mut parts = version_str.trim().split('.');
        let major_str = parts.next().ok_or_else(|| {
            DnamError::SchemaValidationError(alloc::format!(
                "invalid schema version '{version_str}': expected 'major.minor.patch'"
            ))
        })?;
        let minor_str = parts.next().ok_or_else(|| {
            DnamError::SchemaValidationError(alloc::format!(
                "invalid schema version '{version_str}': expected 'major.minor.patch'"
            ))
        })?;
        let patch_str = parts.next().unwrap_or("0");
        if parts.next().is_some() {
            return Err(DnamError::SchemaValidationError(alloc::format!(
                "invalid schema version '{version_str}': too many dot-separated segments"
            )));
        }
        let major = major_str.parse::<u16>().map_err(|_| {
            DnamError::SchemaValidationError(alloc::format!(
                "invalid schema major version in '{version_str}'"
            ))
        })?;
        let minor = minor_str.parse::<u16>().map_err(|_| {
            DnamError::SchemaValidationError(alloc::format!(
                "invalid schema minor version in '{version_str}'"
            ))
        })?;
        let patch = patch_str.parse::<u16>().map_err(|_| {
            DnamError::SchemaValidationError(alloc::format!(
                "invalid schema patch version in '{version_str}'"
            ))
        })?;
        Ok(Self {
            major,
            minor,
            patch,
        })
    }

    /// Formats the schema version as `"major.minor.patch"`.
    pub fn as_semver_string(&self) -> String {
        alloc::format!("{}.{}.{}", self.major, self.minor, self.patch)
    }

    /// Returns `true` if this schema version is supported by the current engine build.
    #[inline]
    #[allow(clippy::absurd_extreme_comparisons)]
    pub const fn is_supported(&self) -> bool {
        self.major == crate::container::CONTAINER_VERSION_MAJOR
            && self.minor <= crate::container::CONTAINER_VERSION_MINOR
    }

    /// Fails with [`DnamError::UnsupportedVersion`] if this schema version is not supported.
    #[inline]
    pub fn ensure_supported(&self) -> Result<(), DnamError> {
        if !self.is_supported() {
            return Err(DnamError::UnsupportedVersion {
                major: self.major,
                minor: self.minor,
            });
        }
        Ok(())
    }

    /// Verifies that the 64-byte binary preamble `(version_major, version_minor)` matches this JSON header version.
    pub fn ensure_matches_preamble(
        &self,
        preamble_major: u16,
        preamble_minor: u16,
    ) -> Result<(), DnamError> {
        if self.major != preamble_major || self.minor != preamble_minor {
            return Err(DnamError::SchemaValidationError(alloc::format!(
                "JSON header version '{}.{}.{}' does not match binary preamble version '{}.{}'",
                self.major,
                self.minor,
                self.patch,
                preamble_major,
                preamble_minor
            )));
        }
        Ok(())
    }
}

fn default_version() -> String {
    SchemaVersion::V2_0_0.as_semver_string()
}

fn default_container_kind() -> String {
    "dnam".to_string()
}

fn default_architecture() -> String {
    "DNAM_TORA".to_string()
}

const fn default_sample_rate() -> u32 {
    48_000
}

/// Top-level UTF-8 JSON header located at byte offset `0x40` of a `.dnam` or `.dnam-ckpt` container.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DnamJsonHeader {
    /// Semantic version of the container schema (`"1.0.0"`).
    #[serde(default = "default_version")]
    pub version: String,
    /// Container kind: `"dnam"` or `"dnam-ckpt"`.
    #[serde(default = "default_container_kind")]
    pub container_kind: String,
    /// Target engine architecture tag (`"DNAM_TORA"`).
    #[serde(default = "default_architecture")]
    pub architecture: String,
    /// Sampling rate at which the neural model was trained (`48000`).
    #[serde(default = "default_sample_rate")]
    pub sample_rate_train_hz: u32,
    /// Essential model identity, calibration, dataset provenance, and cryptographic signatures.
    pub metadata: ModelMetadata,
    /// Open dictionary for arbitrary application/user-defined metadata.
    #[serde(default)]
    pub user_metadata: BTreeMap<String, serde_json::Value>,
    /// Audio I/O topology, PDC latency, estimated tail time, capabilities, and auto-unity leveling.
    pub io_config: IoConfig,
    /// Slimmable tier profiles (`lite`, `full`, `pro`, `ultra`).
    pub slimmable_tiers: SlimmableTiersConfig,
    /// Human-readable definitions of all exposed macro controls ($P \in [0, 16]$).
    #[serde(default)]
    pub controls_schema: Vec<ControlSchemaDef>,
    /// Creator factory presets ($K \in [0, 64]$).
    #[serde(default)]
    pub presets: Vec<PresetDef>,
    /// Control-Bus parameter definitions.
    #[serde(default)]
    pub control_bus: ControlBusConfig,
    /// Stage 0 8-Port Matrix Router configuration.
    #[serde(default)]
    pub stage0_router: RouterConfig,
    /// Stage I Pre-Dynamics and Tone Shaping configuration.
    #[serde(default)]
    pub stage1_pre: Stage1PreFxConfig,
    /// Stage II Commuted Cascade configuration.
    #[serde(default)]
    pub stage2_cascade: Stage2CascadeConfig,
    /// Stage III Cabinet Early FIR + Orthonormal Kautz Modal Bank configuration.
    #[serde(default)]
    pub stage3_cabinet: Stage3CabinetKautzConfig,
    /// Stage IV Spatial Tank configuration.
    #[serde(default)]
    pub stage4_tank: Stage4SpatialTankConfig,
    /// Tensor directory mapping symbolic tensor names to byte slices in `Arena 0..3`.
    #[serde(default)]
    pub tensor_directory: BTreeMap<String, TensorSliceEntry>,
}

/// Essential model identity, calibration, dataset provenance, and cryptographic signatures.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelMetadata {
    /// Human-readable model name.
    pub name: String,
    /// Optional creator / studio name.
    #[serde(default)]
    pub author: Option<String>,
    /// Optional description of the modeled hardware gear.
    #[serde(default)]
    pub description: Option<String>,
    /// Optional SPDX or custom license identifier.
    #[serde(default)]
    pub license: Option<String>,
    /// ISO-8601 UTC creation timestamp.
    pub created_at: String,
    /// Reference input calibration level in dBFS (`1 dBu = 0.869 Vrms = -20.5 dBFS` default).
    #[serde(default = "default_input_ref_dbfs")]
    pub input_reference_dbfs: f32,
    /// Content-addressable training dataset provenance.
    #[serde(default)]
    pub provenance: DatasetProvenance,
    /// Optional creator-signed distribution and OEM hardware exclusivity policy (covered by `immutable_header_sha256`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distribution_policy: Option<DistributionPolicy>,
    /// Optional split-payload `Ed25519` creator signature and marketplace receipt.
    #[serde(default)]
    pub signatures: Option<ModelSignatures>,
}

impl DnamJsonHeader {
    /// Constructs a canonical reference [`DnamJsonHeader`] (used by tests and container builders).
    pub fn default_reference() -> Self {
        Self {
            version: "2.0.0".to_string(),
            container_kind: "dnam".to_string(),
            architecture: "DNAM_TORA".to_string(),
            sample_rate_train_hz: 48_000,
            metadata: ModelMetadata {
                name: "Reference DNAM Model".to_string(),
                author: None,
                description: None,
                license: None,
                created_at: "2026-10-23T00:00:00Z".to_string(),
                input_reference_dbfs: -20.5,
                provenance: DatasetProvenance::default(),
                distribution_policy: None,
                signatures: None,
            },
            user_metadata: BTreeMap::new(),
            io_config: IoConfig::default(),
            slimmable_tiers: SlimmableTiersConfig {
                lite: TierProfile {
                    feature_bus_dim: 64,
                    channels: 4,
                    early_fir_taps: 16,
                    kautz_modes: 4,
                    tank_lines: 4,
                    head_svd_rank: 4,
                    loudness_lufs: -20.5,
                    output_trim_db: 0.0,
                    validation_esr_db: -40.0,
                    arena_end_byte_offset: 10_752,
                    prefix_crc32c: 0,
                },
                full: TierProfile {
                    feature_bus_dim: 128,
                    channels: 8,
                    early_fir_taps: 32,
                    kautz_modes: 8,
                    tank_lines: 8,
                    head_svd_rank: 4,
                    loudness_lufs: -20.5,
                    output_trim_db: 0.0,
                    validation_esr_db: -45.0,
                    arena_end_byte_offset: 22_336,
                    prefix_crc32c: 0,
                },
                pro: TierProfile {
                    feature_bus_dim: 192,
                    channels: 12,
                    early_fir_taps: 48,
                    kautz_modes: 16,
                    tank_lines: 8,
                    head_svd_rank: 4,
                    loudness_lufs: -20.5,
                    output_trim_db: 0.0,
                    validation_esr_db: -48.0,
                    arena_end_byte_offset: 36_224,
                    prefix_crc32c: 0,
                },
                ultra: TierProfile {
                    feature_bus_dim: 256,
                    channels: 16,
                    early_fir_taps: 64,
                    kautz_modes: 32,
                    tank_lines: 16,
                    head_svd_rank: 8,
                    loudness_lufs: -20.5,
                    output_trim_db: 0.0,
                    validation_esr_db: -52.0,
                    arena_end_byte_offset: 64_000,
                    prefix_crc32c: 0,
                },
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

    /// Parses and validates the header's semantic [`SchemaVersion`].
    pub fn schema_version(&self) -> Result<SchemaVersion, DnamError> {
        let ver = SchemaVersion::parse(&self.version)?;
        ver.ensure_supported()?;
        Ok(ver)
    }

    /// Rejects storefront `.dnam-prev` (`container_kind == "preview"`) containers when loaded on a
    /// DAW or hardware host (`ref[sec.preview.containment]`).
    pub fn ensure_not_preview_for_daw_host(&self) -> Result<(), DnamError> {
        if self.container_kind == "preview" {
            return Err(DnamError::PreviewContainerRejected);
        }
        Ok(())
    }

    /// Validates all structural, alignment, and safety invariants of the JSON header.
    pub fn validate(&self) -> Result<(), DnamError> {
        let _ = self.schema_version()?;
        if self.container_kind != "dnam"
            && self.container_kind != "dnam-ckpt"
            && self.container_kind != "preview"
        {
            return Err(DnamError::SchemaValidationError(
                "container_kind must be 'dnam', 'dnam-ckpt', or 'preview'".to_string(),
            ));
        }
        if self.sample_rate_train_hz < 44_100 || self.sample_rate_train_hz > 96_000 {
            return Err(DnamError::InvalidSampleRate(self.sample_rate_train_hz));
        }
        if self.controls_schema.len() > u8::MAX as usize {
            return Err(DnamError::SchemaValidationError(
                "controls_schema exceeds u8::MAX (255) controls in binary preamble".to_string(),
            ));
        }
        for entry in self.tensor_directory.values() {
            if (entry.byte_offset & 0x3F) != 0 {
                return Err(DnamError::UnalignedOffset {
                    field: "tensor_directory.byte_offset",
                    offset: entry.byte_offset,
                });
            }
            if (entry.byte_length & 0x3F) != 0 {
                return Err(DnamError::UnalignedOffset {
                    field: "tensor_directory.byte_length",
                    offset: entry.byte_length,
                });
            }
        }
        let _ = self.compile_clamp_descriptors()?;
        Ok(())
    }

    /// Compiles `controls_schema` ($P \in [0, 255]$) into a vector of 64-byte binary
    /// [`ControlClampDescriptor`] records (`CL 0` of each 128-byte `ControlChannelRecord[p]`)
    /// used at $3\text{ kHz}$ with zero string parsing and zero `FDIV`.
    pub fn compile_clamp_descriptors(&self) -> Result<Vec<ControlClampDescriptor>, DnamError> {
        if self.controls_schema.len() > u8::MAX as usize {
            return Err(DnamError::SchemaValidationError(
                "controls_schema exceeds u8::MAX (255) controls in binary preamble".to_string(),
            ));
        }
        let mut table = alloc::vec![ControlClampDescriptor::empty(); self.controls_schema.len()];

        for (p, ctrl) in self.controls_schema.iter().enumerate() {
            match ctrl.kind {
                ControlKind::Range => {
                    let decl = ctrl.declared_range.ok_or_else(|| {
                        DnamError::SchemaValidationError(
                            "range control missing declared_range".to_string(),
                        )
                    })?;
                    let safe = ctrl.safe_clamped_range.unwrap_or(decl);
                    let span = decl[1] - decl[0];
                    if span <= 1e-6 {
                        return Err(DnamError::SchemaValidationError(
                            "range control declared_range span must be > 0".to_string(),
                        ));
                    }
                    let scale_norm = 2.0 / span;
                    let offset_norm = -1.0 - decl[0] * scale_norm;
                    let default_val = ctrl.default.as_f64().ok_or_else(|| {
                        DnamError::SchemaValidationError(
                            "range control default must be numeric".to_string(),
                        )
                    })? as f32;
                    let degree = ctrl.lobatto_degree.unwrap_or(2).min(3);
                    let mut anchors = [0.0_f32; 3];
                    if let Some(ref nodes) = ctrl.anchor_nodes_norm {
                        for (i, &val) in nodes.iter().take(3).enumerate() {
                            anchors[i] = val.clamp(-1.0, 1.0);
                        }
                    } else {
                        anchors[0] = (default_val * scale_norm + offset_norm).clamp(-1.0, 1.0);
                        anchors[1] = -1.0;
                        anchors[2] = 1.0;
                    }

                    let mut cond_rules = [ConditionalClampRule::empty(); 3];
                    let mut rule_idx = 0;

                    for (cond_key, &max_val) in &ctrl.conditional_max {
                        if rule_idx < 3
                            && let Some((sw_idx, opt_idx)) = self.resolve_switch_condition(cond_key)
                        {
                            cond_rules[rule_idx] = ConditionalClampRule {
                                switch_ctrl_idx: sw_idx,
                                switch_opt_idx: opt_idx,
                                rule_op: 1,
                                bound_val: max_val,
                            };
                            rule_idx += 1;
                        }
                    }
                    for (cond_key, &min_val) in &ctrl.conditional_min {
                        if rule_idx < 3
                            && let Some((sw_idx, opt_idx)) = self.resolve_switch_condition(cond_key)
                        {
                            cond_rules[rule_idx] = ConditionalClampRule {
                                switch_ctrl_idx: sw_idx,
                                switch_opt_idx: opt_idx,
                                rule_op: 2,
                                bound_val: min_val,
                            };
                            rule_idx += 1;
                        }
                    }

                    table[p] = ControlClampDescriptor {
                        ctrl_kind: 0,
                        lobatto_degree: degree,
                        active_options_mask: 0x0001,
                        default_val: default_val.clamp(safe[0], safe[1]),
                        safe_min_base: safe[0].max(decl[0]),
                        safe_max_base: safe[1].min(decl[1]),
                        scale_norm,
                        offset_norm,
                        anchor_nodes_norm: anchors,
                        _reserved: 0,
                        cond_rules,
                    };
                }
                ControlKind::Switch => {
                    let decl_opts = ctrl.declared_options.as_ref().ok_or_else(|| {
                        DnamError::SchemaValidationError(
                            "switch control missing declared_options".to_string(),
                        )
                    })?;
                    if decl_opts.is_empty() || decl_opts.len() > 16 {
                        return Err(DnamError::SchemaValidationError(
                            "switch control must have 1..=16 declared_options".to_string(),
                        ));
                    }
                    let active_opts = ctrl.active_options.as_ref().unwrap_or(decl_opts);
                    let mut mask: u16 = 0;
                    for (idx, opt_id) in decl_opts.iter().enumerate() {
                        if active_opts.iter().any(|a| a == opt_id) {
                            mask |= 1_u16 << idx;
                        }
                    }
                    if mask == 0 {
                        mask = 1;
                    }
                    let default_str = ctrl.default.as_str().unwrap_or(decl_opts[0].as_str());
                    let default_idx = decl_opts.iter().position(|o| o == default_str).unwrap_or(0);

                    table[p] = ControlClampDescriptor {
                        ctrl_kind: 1,
                        lobatto_degree: (decl_opts.len() as u8).saturating_sub(1),
                        active_options_mask: mask,
                        default_val: default_idx as f32,
                        safe_min_base: 0.0,
                        safe_max_base: (decl_opts.len() - 1) as f32,
                        scale_norm: 1.0,
                        offset_norm: 0.0,
                        anchor_nodes_norm: [0.0; 3],
                        _reserved: 0,
                        cond_rules: [ConditionalClampRule::empty(); 3],
                    };
                }
            }
        }

        Ok(table)
    }

    fn resolve_switch_condition(&self, cond_key: &str) -> Option<(u16, u8)> {
        let (sw_id, opt_id) = cond_key.split_once('=')?;
        for (idx, ctrl) in self.controls_schema.iter().enumerate() {
            if ctrl.id == sw_id
                && ctrl.kind == ControlKind::Switch
                && let Some(ref opts) = ctrl.declared_options
                && let Some(opt_idx) = opts.iter().position(|o| o == opt_id)
            {
                return Some((idx as u16, opt_idx as u8));
            }
        }
        None
    }

    /// Deserializes a [`DnamJsonHeader`] from UTF-8 JSON bytes in a single non-generic compilation unit.
    #[inline(never)]
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }

    /// Deserializes a [`DnamJsonHeader`] from a UTF-8 JSON string (delegating to [`Self::from_json_slice`]).
    #[inline]
    pub fn from_json_str(s: &str) -> Result<Self, serde_json::Error> {
        Self::from_json_slice(s.as_bytes())
    }

    /// Serializes this [`DnamJsonHeader`] to a UTF-8 JSON byte vector in a single non-generic compilation unit.
    #[inline(never)]
    pub fn to_json_vec(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    /// Serializes this [`DnamJsonHeader`] to a UTF-8 JSON string (delegating to [`Self::to_json_vec`]).
    #[inline]
    pub fn to_json_string(&self) -> Result<String, serde_json::Error> {
        let bytes = self.to_json_vec()?;
        // SAFETY: `serde_json::to_vec` guarantees valid UTF-8 output.
        Ok(String::from_utf8(bytes).expect("serde_json::to_vec must emit valid UTF-8"))
    }

    /// Parses, normalizes numeric range defaults/presets/stems, validates structural invariants,
    /// rejects unknown schema fields, and serializes a [`DnamJsonHeader`] to canonical wire JSON.
    pub fn validate_and_normalize_json_str(header_json: &str) -> Result<String, DnamError> {
        let input_val: serde_json::Value = serde_json::from_slice(header_json.as_bytes())
            .map_err(|e| DnamError::InvalidJsonHeader(alloc::format!("{e}")))?;
        let mut header: Self = Self::from_json_slice(header_json.as_bytes())
            .map_err(|e| DnamError::SchemaValidationError(alloc::format!("{e}")))?;
        for ctrl in &mut header.controls_schema {
            if ctrl.kind == ControlKind::Range
                && let Some(f) = ctrl.default.as_f64()
            {
                ctrl.default = serde_json::Value::from(f);
            }
        }
        let range_ids: Vec<String> = header
            .controls_schema
            .iter()
            .filter(|c| c.kind == ControlKind::Range)
            .map(|c| c.id.clone())
            .collect();
        for preset in &mut header.presets {
            for (k, v) in &mut preset.controls {
                if range_ids.contains(k)
                    && let Some(f) = v.as_f64()
                {
                    *v = serde_json::Value::from(f);
                }
            }
        }
        header.validate()?;
        let out_bytes = header
            .to_json_vec()
            .map_err(|e| DnamError::InvalidJsonHeader(alloc::format!("{e}")))?;
        let norm_val: serde_json::Value = serde_json::from_slice(&out_bytes)
            .map_err(|e| DnamError::InvalidJsonHeader(alloc::format!("{e}")))?;
        check_no_unknown_header_keys(&input_val, &norm_val, None)?;
        String::from_utf8(out_bytes)
            .map_err(|e| DnamError::InvalidJsonHeader(alloc::format!("{e}")))
    }
}

const USER_DICT_KEYS: &[&str] = &[
    "user_metadata",
    "controls",
    "controls_meta",
    "conditional_min",
    "conditional_max",
    "conditional_physical_anchors",
    "scale_labels",
    "option_labels",
];

fn is_skipped_optional_field(
    parent_key: Option<&str>,
    key: &str,
    v_in: &serde_json::Value,
) -> bool {
    match (parent_key, key) {
        (None, "stems_registry" | "audit_report")
        | (Some("metadata"), "distribution_policy")
        | (Some("signatures"), "tier_arena_sha256")
        | (Some("marketplace_receipt"), "buyer_tier_arena_sha256")
        | (Some("controls_schema"), "calibration")
        | (Some("calibration"), "lobatto_physical_anchors" | "tempo_sync_default_subdivision")
        | (Some("stems_registry"), "extracted_physics")
        | (
            Some("tensor_directory_entry" | "g_audio_uplo_u_f64" | "c_audio_tier_f64"),
            "packing" | "feature_dim" | "packed_len" | "shape",
        ) => v_in.is_null(),
        (Some("calibration"), "conditional_physical_anchors") => {
            v_in.as_object().is_some_and(serde_json::Map::is_empty)
        }
        _ => false,
    }
}

/// Verifies that no unknown JSON schema fields exist in `input` relative to `normalized`.
pub fn check_no_unknown_header_keys(
    input: &serde_json::Value,
    normalized: &serde_json::Value,
    parent_key: Option<&str>,
) -> Result<(), DnamError> {
    if let Some(pk) = parent_key
        && USER_DICT_KEYS.contains(&pk)
    {
        return Ok(());
    }
    match (input, normalized) {
        (serde_json::Value::Object(in_map), serde_json::Value::Object(norm_map)) => {
            if parent_key == Some("tensor_directory") {
                for (k, v_in) in in_map {
                    if let Some(v_norm) = norm_map.get(k) {
                        check_no_unknown_header_keys(v_in, v_norm, Some("tensor_directory_entry"))?;
                    }
                }
                return Ok(());
            }
            for (k, v_in) in in_map {
                if let Some(v_norm) = norm_map.get(k) {
                    check_no_unknown_header_keys(v_in, v_norm, Some(k.as_str()))?;
                } else if !is_skipped_optional_field(parent_key, k.as_str(), v_in) {
                    return Err(DnamError::SchemaValidationError(alloc::format!(
                        "Unknown schema field '{k}'"
                    )));
                }
            }
            Ok(())
        }
        (serde_json::Value::Array(in_arr), serde_json::Value::Array(norm_arr)) => {
            for (v_in, v_norm) in in_arr.iter().zip(norm_arr.iter()) {
                check_no_unknown_header_keys(v_in, v_norm, parent_key)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::ControlUiHints;
    use alloc::vec;

    #[test]
    fn test_minimal_json_serde_defaults_and_validation_errors() {
        let minimal_json = r#"{
            "metadata": {
                "name": "Minimal",
                "created_at": "2026-10-23T00:00:00Z",
                "signatures": {
                    "immutable_header_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                    "master_arena_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                    "creator_signature": {
                        "signer_id": "did:web:test",
                        "pubkey_ed25519": "0000000000000000000000000000000000000000000000000000000000000000",
                        "sig_ed25519": "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                    },
                    "marketplace_receipt": {
                        "marketplace_id": "dnam.store",
                        "license_id": "lic1",
                        "buyer_hash": "0000000000000000000000000000000000000000000000000000000000000000",
                        "license_tier": "personal",
                        "issued_at": "2026-10-23T00:00:00Z",
                        "buyer_arena_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                        "receipt_sig_ed25519": "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                    }
                }
            },
            "io_config": {},
            "control_bus": {},
            "stage0_router": {},
            "stage1_pre": {},
            "stage2_cascade": {},
            "stage3_cabinet": {},
            "stage4_tank": {},
            "slimmable_tiers": {
                "lite": { "channels": 4, "tank_lines": 4, "arena_end_byte_offset": 10752, "prefix_crc32c": 0 },
                "full": { "channels": 8, "tank_lines": 8, "arena_end_byte_offset": 22336, "prefix_crc32c": 0 },
                "ultra": { "channels": 12, "tank_lines": 8, "arena_end_byte_offset": 36224, "prefix_crc32c": 0 }
            },
            "controls_schema": [
                {
                    "id": "Gain",
                    "name": "Gain",
                    "type": "range",
                    "declared_range": [0.0, 10.0],
                    "default": 5.0,
                    "conditional_min": { "Sw=on": 2.0, "UnknownSw=on": 1.0 },
                    "conditional_max": { "Sw=off": 8.0 }
                },
                {
                    "id": "Sw",
                    "name": "Sw",
                    "type": "switch",
                    "declared_options": ["off", "on"],
                    "active_options": ["nonexistent"],
                    "default": "off"
                }
            ]
        }"#;
        let mut h: DnamJsonHeader = serde_json::from_str(minimal_json).unwrap();
        assert_eq!(h.version, "2.0.0");
        assert_eq!(h.container_kind, "dnam");
        assert_eq!(h.architecture, "DNAM_TORA");
        assert_eq!(h.sample_rate_train_hz, 48_000);
        assert_eq!(h.metadata.input_reference_dbfs, -20.5);
        assert_eq!(h.io_config.inputs, 2);
        assert_eq!(h.slimmable_tiers.lite.head_svd_rank, 4);
        assert_eq!(h.control_bus.envelopes, 4);
        assert_eq!(h.stage2_cascade.num_stages, 3);
        assert_eq!(h.stage4_tank.dc_blocker_pole, 0.995);
        assert!(h.validate().is_ok());

        let compiled = h.compile_clamp_descriptors().unwrap();
        assert_eq!(compiled[0].cond_rules[0].rule_op, 1);
        assert_eq!(compiled[0].cond_rules[1].rule_op, 2);
        assert_eq!(compiled[1].active_options_mask, 1);

        // Validation error branches
        h.container_kind = "invalid".to_string();
        assert!(h.validate().is_err());
        h.container_kind = "preview".to_string();
        assert!(h.validate().is_ok());
        h.container_kind = "dnam".to_string();

        h.sample_rate_train_hz = 10_000;
        assert!(h.validate().is_err());
        h.sample_rate_train_hz = 48_000;

        let mut bad_slice = TensorSliceEntry {
            arena: 0,
            dtype: "f32".to_string(),
            packing: None,
            feature_dim: None,
            packed_len: None,
            shape: None,
            byte_offset: 3,
            byte_length: 64,
        };
        h.tensor_directory
            .insert("bad_off".to_string(), bad_slice.clone());
        assert!(h.validate().is_err());
        bad_slice.byte_offset = 64;
        bad_slice.byte_length = 63;
        h.tensor_directory.insert("bad_off".to_string(), bad_slice);
        assert!(h.validate().is_err());
        h.tensor_directory.clear();

        // Missing declared_range on range control
        h.controls_schema = vec![ControlSchemaDef {
            id: "Gain".to_string(),
            name: "Gain".to_string(),
            kind: ControlKind::Range,
            unit: None,
            lobatto_degree: None,
            anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
            declared_range: None,
            safe_clamped_range: None,
            default: serde_json::json!(5.0),
            conditional_min: BTreeMap::new(),
            conditional_max: BTreeMap::new(),
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            ui_hints: ControlUiHints::default(),
            calibration: None,
            user_metadata: BTreeMap::new(),
        }];
        assert!(h.validate().is_err());

        h.controls_schema[0].declared_range = Some([5.0, 5.0]);
        assert!(h.validate().is_err());

        h.controls_schema[0].declared_range = Some([0.0, 10.0]);
        h.controls_schema[0].default = serde_json::json!("bad");
        assert!(h.validate().is_err());

        h.controls_schema[0].kind = ControlKind::Switch;
        assert!(h.validate().is_err());

        h.controls_schema[0].declared_options = Some(vec![]);
        assert!(h.validate().is_err());

        h.controls_schema[0].declared_options = Some(vec!["off".to_string(), "on".to_string()]);
        h.controls_schema = vec![h.controls_schema[0].clone(); 256];
        assert!(h.validate().is_err());
        assert!(h.compile_clamp_descriptors().is_err());

        // Capabilities & Leveling defaults and validation
        assert_eq!(h.io_config.capabilities.blend_mode, BlendMode::None);
        assert_eq!(h.io_config.capabilities.default_blend, 1.0);
        assert!(!h.io_config.capabilities.wet_polarity_inverted);
        assert_eq!(
            h.io_config.leveling.leveling_metric,
            LevelingMetric::RmsDbfs
        );
        assert!(h.io_config.leveling.auto_unity_enabled);

        assert!(SchemaVersion::parse("a.0.0").is_err());
        assert!(SchemaVersion::parse("1.b.0").is_err());
        assert!(SchemaVersion::parse("1.0.c").is_err());
        assert_eq!(SchemaVersion::V2_0_0.as_semver_string(), "2.0.0");

        let norm_json = DnamJsonHeader::validate_and_normalize_json_str(minimal_json).unwrap();
        assert!(norm_json.contains("\"version\":\"2.0.0\""));
        assert!(DnamJsonHeader::validate_and_normalize_json_str("not json").is_err());
        let bad_unknown = minimal_json.replace(
            "\"metadata\": {",
            "\"unknown_root_field\": 123, \"metadata\": {",
        );
        assert!(DnamJsonHeader::validate_and_normalize_json_str(&bad_unknown).is_err());

        // Misplaced optional field ("shape" inside "stage1_pre" or "distribution_policy" at root) must fail
        let bad_misplaced =
            minimal_json.replace("\"stage1_pre\": {}", "\"stage1_pre\": {\"shape\": \"bad\"}");
        assert!(DnamJsonHeader::validate_and_normalize_json_str(&bad_misplaced).is_err());
        let bad_root_opt = minimal_json.replace(
            "\"metadata\": {",
            "\"distribution_policy\": null, \"metadata\": {",
        );
        assert!(DnamJsonHeader::validate_and_normalize_json_str(&bad_root_opt).is_err());

        // Explicit null on valid skip_serializing_if fields must succeed
        let ok_explicit_nulls = minimal_json.replace(
            "\"metadata\": {",
            "\"stems_registry\": null, \"audit_report\": null, \"metadata\": {\"distribution_policy\": null,",
        );
        assert!(DnamJsonHeader::validate_and_normalize_json_str(&ok_explicit_nulls).is_ok());
    }
}
