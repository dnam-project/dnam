//! Canonical `.dnam` and `.dnam-ckpt` UTF-8 JSON Header (`DnamJsonHeader`), subsystem configurations,
//! tensor directory (`TensorSliceEntry`), stem registry (`StemRegistryEntry`), and binary clamp compiler.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use crate::control::{ConditionalClampRule, ControlClampDescriptor, MAX_CONTROLS};

use super::controls::{ControlKind, ControlSchemaDef, PresetDef};
use super::error::DnamError;
use super::signatures::{DatasetProvenance, DistributionPolicy, ModelSignatures};

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
    SchemaVersion::V1_0_0.as_semver_string()
}

fn default_container_kind() -> String {
    "dnam".to_string()
}

fn default_architecture() -> String {
    "DNAM_TriPhysics".to_string()
}

const fn default_sample_rate() -> u32 {
    48_000
}

const fn default_input_ref_dbfs() -> f32 {
    -20.5
}

const fn default_inputs() -> u8 {
    2
}

fn default_output_mode() -> String {
    "stereo".to_string()
}

const fn default_head_svd_rank() -> u8 {
    4
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
    /// Architecture identifier (`"DNAM_TriPhysics"`).
    #[serde(default = "default_architecture")]
    pub architecture: String,
    /// Native training sample rate in Hz (typically `48000`).
    #[serde(default = "default_sample_rate")]
    pub sample_rate_train_hz: u32,
    /// Essential model identity, calibration, provenance, and split-payload signatures.
    pub metadata: ModelMetadata,
    /// Open dictionary for storefront/host/user metadata (excluded from `immutable_header_sha256`).
    #[serde(default)]
    pub user_metadata: BTreeMap<String, serde_json::Value>,
    /// Audio I/O topology, latency, and tail time configuration.
    #[serde(default)]
    pub io_config: IoConfig,
    /// Slimmable tier configurations (`lite`, `full`, `ultra`).
    pub slimmable_tiers: SlimmableTiersConfig,
    /// Parametric `range` and `switch` control definitions (`0..=16` controls).
    #[serde(default)]
    pub controls_schema: Vec<ControlSchemaDef>,
    /// Built-in parametric control snapshots (excluded from `immutable_header_sha256`).
    #[serde(default)]
    pub presets: Vec<PresetDef>,
    /// Control-bus hyperparameters.
    #[serde(default)]
    pub control_bus: ControlBusConfig,
    /// Core 1 Hybrid Dilated-Conv + `PolyADAA` + `SelectiveRgLru` architecture parameters.
    #[serde(default)]
    pub core1_hybrid_ssm: Core1Config,
    /// Core 2 Diffusive Modulated Unitary FDN (`M-UFDN`) parameters.
    #[serde(default)]
    pub core2_mufdn: Core2Config,
    /// 64-byte aligned tensor directory mapping named slices in `Arena 0..2`.
    #[serde(default)]
    pub tensor_directory: BTreeMap<String, TensorSliceEntry>,
    /// `.dnam-ckpt` only: per-stem provenance, extracted physics, and `Arena 3` `uplo='U'` records.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stems_registry: Option<Vec<StemRegistryEntry>>,
    /// `.dnam-ckpt` only: 3-phase control clamp audit report and actionable prescriptions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audit_report: Option<AuditReport>,
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

const fn default_true() -> bool {
    true
}

const fn default_blend_one() -> f32 {
    1.0
}

/// Host dry/wet blend capability mode (`IoConfig::capabilities.blend_mode`).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BlendMode {
    /// No separate host parallel dry/wet blend control should be exposed.
    #[default]
    None,
    /// Host/UI should expose a parallel Dry/Wet `Blend` knob driving `DnamEngine`'s `mix_dry_gain` / `mix_wet_gain`.
    HostParallel,
    /// Dry/Wet `Blend` or `Mix` is already modeled as a parametric control in `controls_schema`.
    ModeledControl,
}

/// Model-level physical capabilities (`IoConfig::capabilities`, covered by `immutable_header_sha256`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelCapabilities {
    /// Dry/Wet parallel blend configuration (`"none"`, `"host_parallel"`, or `"modeled_control"`).
    #[serde(default)]
    pub blend_mode: BlendMode,
    /// Default wet blend ratio $b \in [0.0, 1.0]$ when `blend_mode == "host_parallel"` (`1.0` = 100% wet).
    #[serde(default = "default_blend_one")]
    pub default_blend: f32,
    /// `true` if the modeled hardware inverts signal polarity (`-1`) relative to the dry input.
    /// When `true`, `DnamEngine::configure_host_mix_and_leveling` inverts `wet_g` (`wet_g *= -1.0`)
    /// so `0%` blend matches bypass polarity and `50%` blend sums constructively without phase cancellation.
    #[serde(default)]
    pub wet_polarity_inverted: bool,
    /// `true` if spillover bypass (`DnamEngine::set_bypass(true, true)`) is supported.
    #[serde(default)]
    pub supports_trails: bool,
    /// `true` if one or more controls in `controls_schema` carry `DelayTimeMs`, `GrainSizeMs`, or `LfoRateHz` calibration for MIDI tempo sync.
    #[serde(default)]
    pub tempo_sync_capable: bool,
    /// `true` if unitary spectral freeze (`DnamEngine::set_freeze(true)` or `FreezeLatch` control) is supported.
    #[serde(default)]
    pub supports_freeze: bool,
    /// `true` if engaging spectral freeze also mutes the live dry/Core 1 signal path (stutter/hold replacement);
    /// `false` (default) allows "Sound-on-Sound" playing over the frozen Core 2 pad.
    #[serde(default)]
    pub freeze_mutes_dry: bool,
}

impl Default for ModelCapabilities {
    fn default() -> Self {
        Self {
            blend_mode: BlendMode::None,
            default_blend: 1.0,
            wet_polarity_inverted: false,
            supports_trails: false,
            tempo_sync_capable: false,
            supports_freeze: false,
            freeze_mutes_dry: false,
        }
    }
}

/// Loudness measurement metric used for auto-unity leveling (`LevelingConfig::leveling_metric`).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LevelingMetric {
    /// Gated unweighted broadband RMS in dBFS (`"rms_dbfs"`, default for pedal-to-pedal `Vrms` gain staging).
    #[default]
    RmsDbfs,
    /// Gated ITU-R BS.1770-4 K-weighted LUFS (`"k_weighted_lufs"`, optional for full-rig + cab models).
    KWeightedLufs,
}

/// Auto-unity gain leveling calibration anchored at the default control settings (`controls_schema[*].default`, $\mathbf{u} = \mathbf{0}$).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LevelingConfig {
    /// Whether auto-unity leveling is enabled by default for this model.
    #[serde(default = "default_true")]
    pub auto_unity_enabled: bool,
    /// Loudness metric used to compute `unity_gain_offset_db` (`"rms_dbfs"` default or `"k_weighted_lufs"`).
    #[serde(default)]
    pub leveling_metric: LevelingMetric,
    /// Measured dry excitation reference level in dB (`dBFS` RMS or `LUFS`).
    #[serde(default = "default_input_ref_dbfs", alias = "dry_reference_db")]
    pub reference_input_level_db: f32,
    /// Measured model wet output level in dB at the default control settings ($\mathbf{u} = \mathbf{0}$).
    #[serde(default = "default_input_ref_dbfs", alias = "default_controls_wet_db")]
    pub measured_default_output_level_db: f32,
    /// Gain offset in dB required to match unity at default control settings (`reference_input_level_db - measured_default_output_level_db`).
    #[serde(default)]
    pub unity_gain_offset_db: f32,
    /// Creator-configured target gain offset in dB relative to unity (`0.0` = exact unity; e.g. `+3.0` for a boost).
    #[serde(default)]
    pub target_gain_db: f32,
    /// Total calibrated output trim in dB (`unity_gain_offset_db + target_gain_db` when `auto_unity_enabled`, else `target_gain_db`).
    #[serde(default)]
    pub calibrated_trim_db: f32,
}

impl Default for LevelingConfig {
    fn default() -> Self {
        Self {
            auto_unity_enabled: true,
            leveling_metric: LevelingMetric::RmsDbfs,
            reference_input_level_db: default_input_ref_dbfs(),
            measured_default_output_level_db: default_input_ref_dbfs(),
            unity_gain_offset_db: 0.0,
            target_gain_db: 0.0,
            calibrated_trim_db: 0.0,
        }
    }
}

/// Audio I/O topology, host PDC / tail configuration, model capabilities, and auto-unity leveling.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IoConfig {
    /// Number of input channels (`1` or `2`).
    #[serde(default = "default_inputs")]
    pub inputs: u8,
    /// Output topology mode (`"stereo"` or `"mono"`).
    #[serde(default = "default_output_mode")]
    pub output_mode: String,
    /// Algorithmic lookahead latency in samples (always `0` for DNAM).
    #[serde(default)]
    pub latency_samples: u32,
    /// Estimated $T_{60}$ reverb/delay tail time in milliseconds for DAW sleep management.
    #[serde(default)]
    pub tail_time_ms: f32,
    /// Model-level capabilities (parallel blend, polarity inversion, trails, tempo sync).
    #[serde(default)]
    pub capabilities: ModelCapabilities,
    /// Auto-unity leveling calibration anchored at default control settings ($\mathbf{u} = \mathbf{0}$).
    #[serde(default)]
    pub leveling: LevelingConfig,
}

impl Default for IoConfig {
    fn default() -> Self {
        Self {
            inputs: 2,
            output_mode: default_output_mode(),
            latency_samples: 0,
            tail_time_ms: 0.0,
            capabilities: ModelCapabilities::default(),
            leveling: LevelingConfig::default(),
        }
    }
}

/// Slimmable tier profiles (`lite`, `full`, `ultra`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SlimmableTiersConfig {
    /// `Lite` ($C=4$, $M=4$) tier profile.
    pub lite: TierProfile,
    /// `Full` ($C=8$, $M=8$) tier profile.
    pub full: TierProfile,
    /// `Ultra` ($C=12$, $M=8$) tier profile.
    pub ultra: TierProfile,
}

/// Metadata and prefix checksum for an individual slimmable tier.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TierProfile {
    /// Active Core 1 channels ($C \in \{4, 8, 12\}$).
    pub channels: u8,
    /// Active Core 2 FDN delay lines ($M \in \{0, 4, 8\}$).
    pub fdn_lines: u8,
    /// Active early-reflection taps ($M_{\text{early}} \in \{0, 4, 8\}$).
    #[serde(default)]
    pub early_taps: u8,
    /// SVD output head rank ($D_h = 4$).
    #[serde(default = "default_head_svd_rank")]
    pub head_svd_rank: u8,
    /// Integrated loudness in LUFS (`-20.5` target).
    #[serde(default = "default_input_ref_dbfs")]
    pub loudness_lufs: f32,
    /// Linear output loudness trim in dB (`0.0` default).
    #[serde(default)]
    pub output_trim_db: f32,
    /// Validation Error-to-Signal Ratio (ESR) in dB.
    #[serde(default)]
    pub validation_esr_db: f32,
    /// End byte offset of this tier's cumulative prefix relative to `arena_offset_start`.
    pub arena_end_byte_offset: u32,
    /// Castagnoli `CRC32C` over `arena[0 .. arena_end_byte_offset]`.
    pub prefix_crc32c: u32,
}

const fn default_decimation() -> u16 {
    16
}
const fn default_envelopes() -> u8 {
    3
}
const fn default_lfos() -> u8 {
    2
}
const fn default_lfo_harmonics() -> u8 {
    4
}
const fn default_fm_rank() -> u8 {
    4
}
const fn default_trust_radius() -> f32 {
    2.5
}

/// Control-Bus configuration (`control_bus`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ControlBusConfig {
    /// Control-bus decimation stride at $f_{s,\text{train}}$ (`16` at $48\text{ kHz}$ $\to 3\text{ kHz}$).
    #[serde(default = "default_decimation")]
    pub decimation: u16,
    /// Number of envelope follower bands (`3`).
    #[serde(default = "default_envelopes")]
    pub envelopes: u8,
    /// Number of LFO oscillators (`2`).
    #[serde(default = "default_lfos")]
    pub lfo_oscillators: u8,
    /// Number of Chebyshev harmonics per LFO (`4`).
    #[serde(default = "default_lfo_harmonics")]
    pub lfo_harmonics: u8,
    /// Factorization Machine (FM) interaction rank ($R = 4$).
    #[serde(default = "default_fm_rank")]
    pub fm_interaction_rank: u8,
    /// Trust-region soft-clip radius $M_h$ (`CL-15`, default `2.5`).
    #[serde(default = "default_trust_radius")]
    pub head_trust_region_radius: f32,
}

impl Default for ControlBusConfig {
    fn default() -> Self {
        Self {
            decimation: 16,
            envelopes: 3,
            lfo_oscillators: 2,
            lfo_harmonics: 4,
            fm_interaction_rank: 4,
            head_trust_region_radius: 2.5,
        }
    }
}

const fn default_blocks() -> u8 {
    4
}
const fn default_conv_kernel() -> u8 {
    6
}
const fn default_dilations() -> [u16; 4] {
    [1, 3, 7, 19]
}
fn default_adaa_mode() -> String {
    "1x_bspline_algebraic".to_string()
}
const fn default_ac_pole() -> f32 {
    0.9993
}
fn default_weight_layout() -> String {
    "tier_delta_exec_order_soa_align64".to_string()
}

/// Core 1 Hybrid SSM configuration (`core1_hybrid_ssm`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Core1Config {
    /// Number of interleaved Core 1 blocks (`4`).
    #[serde(default = "default_blocks")]
    pub blocks: u8,
    /// Causal dilated convolution kernel width ($K = 6$).
    #[serde(default = "default_conv_kernel")]
    pub conv_kernel: u8,
    /// Block dilation schedule (`[1, 3, 7, 19]`).
    #[serde(default = "default_dilations")]
    pub dilations: [u16; 4],
    /// Anti-aliasing mode (`"1x_bspline_algebraic"`).
    #[serde(default = "default_adaa_mode")]
    pub adaa_mode: String,
    /// $5\text{ Hz}$ AC-coupling pole at $f_{s,\text{train}}$ (`0.9993`).
    #[serde(default = "default_ac_pole")]
    pub ac_coupling_pole: f32,
    /// Memory layout tag (`"tier_delta_exec_order_soa_align64"`).
    #[serde(default = "default_weight_layout")]
    pub weight_layout: String,
}

impl Default for Core1Config {
    fn default() -> Self {
        Self {
            blocks: 4,
            conv_kernel: 6,
            dilations: [1, 3, 7, 19],
            adaa_mode: default_adaa_mode(),
            ac_coupling_pole: 0.9993,
            weight_layout: default_weight_layout(),
        }
    }
}

fn default_delay_storage() -> String {
    "fp16".to_string()
}
fn default_interp() -> String {
    "cubic_lagrange_4tap".to_string()
}
const fn default_tzf_tap() -> f32 {
    64.0
}
const fn default_switch_xfade() -> u16 {
    128
}
const fn default_allpass_delays() -> [u16; 8] {
    [19, 31, 47, 73, 103, 139, 179, 223]
}
const fn default_dc_pole() -> f32 {
    0.995
}
const fn default_loop_ceiling() -> f32 {
    0.985
}

/// Core 2 Diffusive Modulated Unitary FDN configuration (`core2_mufdn`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Core2Config {
    /// Whether Core 2 is active (`CORE2_ACTIVE` bit 6).
    #[serde(default)]
    pub active: bool,
    /// Delay buffer storage format (`"fp16"`).
    #[serde(default = "default_delay_storage")]
    pub delay_storage: String,
    /// $\log_2(N_{\text{ring}})$ per FDN delay line (`0` when inactive, `10..=15` when active).
    #[serde(default)]
    pub ring_buffer_pow2_log2: u8,
    /// Fractional delay interpolation kernel (`"cubic_lagrange_4tap"`).
    #[serde(default = "default_interp")]
    pub interp: String,
    /// Through-Zero Flanging (TZF) dry reference tap in samples (`64.0` at $48\text{ kHz}$).
    #[serde(default = "default_tzf_tap")]
    pub tzf_reference_tap_samples: f32,
    /// Raised-cosine switch crossfade window in samples (`128` at $48\text{ kHz}$).
    #[serde(default = "default_switch_xfade")]
    pub switch_xfade_samples: u16,
    /// Coprime Schroeder all-pass delay lengths in samples (`[19, 31, 47, 73, 103, 139, 179, 223]`).
    #[serde(default = "default_allpass_delays")]
    pub allpass_delays_samples: [u16; 8],
    /// In-loop $5\text{ Hz}$ DC blocker pole (`0.995` at $48\text{ kHz}$).
    #[serde(default = "default_dc_pole")]
    pub dc_blocker_pole: f32,
    /// Global loop ceiling (`0.985`).
    #[serde(default = "default_loop_ceiling")]
    pub loop_ceiling: f32,
}

impl Default for Core2Config {
    fn default() -> Self {
        Self {
            active: false,
            delay_storage: default_delay_storage(),
            ring_buffer_pow2_log2: 0,
            interp: default_interp(),
            tzf_reference_tap_samples: 64.0,
            switch_xfade_samples: 128,
            allpass_delays_samples: default_allpass_delays(),
            dc_blocker_pole: 0.995,
            loop_ceiling: 0.985,
        }
    }
}

/// Tensor directory entry describing a 64-byte aligned slice in `Arena 0..3`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorSliceEntry {
    /// Arena index (`0` = Lite, `1` = Full Delta, `2` = Ultra Delta, `3` = `.dnam-ckpt` Gram stats).
    pub arena: u8,
    /// Data type (`"f32"` or `"f64"`).
    pub dtype: String,
    /// Optional packing tag (`"uplo_upper_col_major_tier_ordered"` for `Arena 3` Gram matrices).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packing: Option<String>,
    /// Optional feature dimension $N \in \{64, 128, 192\}$ for `uplo='U'` packed Gram matrices.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_dim: Option<u32>,
    /// Optional packed element count $\frac{N(N+1)}{2}$.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packed_len: Option<u32>,
    /// Tensor dimensions (for standard dense tensors).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<Vec<usize>>,
    /// 64-byte aligned byte offset relative to `arena_offset_start`.
    pub byte_offset: u32,
    /// 64-byte aligned byte length.
    pub byte_length: u32,
}

/// `.dnam-ckpt` per-stem registry record (`stems_registry[*]`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StemRegistryEntry {
    /// Unique stem identifier (e.g., `"stem_00_noon"`).
    pub stem_id: String,
    /// Control settings captured for this stem.
    pub controls_meta: BTreeMap<String, serde_json::Value>,
    /// Platform-agnostic IEEE-754 PCM SHA-256 provenance digests.
    pub provenance: StemProvenance,
    /// Castagnoli `CRC32C` over this stem's $151,296\text{ B}$ `Arena 3` record (`G_audio + C_audio`).
    pub stem_arena3_crc32c: u32,
    /// Open dictionary for stem-specific user metadata.
    #[serde(default)]
    pub user_metadata: BTreeMap<String, serde_json::Value>,
    /// Extracted Stage-1 physical anchors (delays, LFO frequencies, decay gains).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extracted_physics: Option<ExtractedPhysics>,
    /// `Arena 3` slice descriptor for $\operatorname{pack}_{\text{U}}(\mathbf{G}_{\text{audio}}^{(s)})$ (`148,224 B`).
    pub g_audio_uplo_u_f64: TensorSliceEntry,
    /// `Arena 3` slice descriptor for tier-ordered $\mathbf{C}_{\text{audio}}^{(s)}$ (`3,072 B`).
    pub c_audio_tier_f64: TensorSliceEntry,
}

/// Platform-agnostic IEEE-754 PCM SHA-256 provenance for a training stem.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct StemProvenance {
    /// SHA-256 hex digest of raw little-endian `f32` DI input frames.
    pub input_pcm_sha256: String,
    /// SHA-256 hex digest of raw little-endian `f32` target output frames.
    pub target_pcm_sha256: String,
    /// Number of stereo/mono frames in the stem.
    pub num_frames: u64,
}

/// Extracted Stage-1 physical system-identification anchors for a stem.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ExtractedPhysics {
    /// Extracted FDN delay times in milliseconds.
    #[serde(default)]
    pub delays_ms: Vec<f32>,
    /// Extracted LFO frequencies in Hz.
    #[serde(default)]
    pub lfo_freqs_hz: Vec<f32>,
    /// Extracted per-line linear feedback decay gains.
    #[serde(default)]
    pub decay_gains_lin: Vec<f32>,
    /// Extracted grain window length in milliseconds (`0.0` when un-windowed).
    #[serde(default)]
    pub detected_grain_size_ms: f32,
    /// Extracted dominant pitch-shift interval in semitones (`0.0` when un-pitched).
    #[serde(default)]
    pub detected_pitch_shift_semitones: f32,
}

/// `.dnam-ckpt` 3-phase control clamp audit report and actionable stem prescriptions.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuditReport {
    /// Phase-1 dataset coverage clamps (`"CLAMP-RANGE"`, `"CLAMP-SWITCH"`, `"CLAMP-STABILITY"`).
    #[serde(default)]
    pub phase1_clamps: Vec<String>,
    /// Phase-2 interior rail saturation warnings.
    #[serde(default)]
    pub phase2_rail_warnings: Vec<String>,
    /// Phase-3 post-fit dynamic stability & head gain clamps (`"CLAMP-DYNAMIC"`, `"CLAMP-HEAD-GAIN"`).
    #[serde(default)]
    pub phase3_clamps: Vec<String>,
    /// Human-readable actionable stem prescriptions to lift active clamps.
    #[serde(default)]
    pub prescriptions: Vec<String>,
}

impl DnamJsonHeader {
    /// Constructs a canonical reference [`DnamJsonHeader`] (used by tests and container builders).
    pub fn default_reference(core2_active: bool) -> Self {
        Self {
            version: "1.0.0".to_string(),
            container_kind: "dnam".to_string(),
            architecture: "DNAM_TriPhysics".to_string(),
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
                    channels: 4,
                    fdn_lines: if core2_active { 4 } else { 0 },
                    early_taps: 4,
                    head_svd_rank: 4,
                    loudness_lufs: -20.5,
                    output_trim_db: 0.0,
                    validation_esr_db: -40.0,
                    arena_end_byte_offset: 10_752,
                    prefix_crc32c: 0,
                },
                full: TierProfile {
                    channels: 8,
                    fdn_lines: if core2_active { 8 } else { 0 },
                    early_taps: 8,
                    head_svd_rank: 4,
                    loudness_lufs: -20.5,
                    output_trim_db: 0.0,
                    validation_esr_db: -45.0,
                    arena_end_byte_offset: 22_336,
                    prefix_crc32c: 0,
                },
                ultra: TierProfile {
                    channels: 12,
                    fdn_lines: if core2_active { 8 } else { 0 },
                    early_taps: 8,
                    head_svd_rank: 4,
                    loudness_lufs: -20.5,
                    output_trim_db: 0.0,
                    validation_esr_db: -48.0,
                    arena_end_byte_offset: 36_224,
                    prefix_crc32c: 0,
                },
            },
            controls_schema: Vec::new(),
            presets: Vec::new(),
            control_bus: ControlBusConfig::default(),
            core1_hybrid_ssm: Core1Config::default(),
            core2_mufdn: Core2Config {
                active: core2_active,
                ring_buffer_pow2_log2: if core2_active { 10 } else { 0 },
                ..Default::default()
            },
            tensor_directory: BTreeMap::new(),
            stems_registry: None,
            audit_report: None,
        }
    }

    /// Parses and validates the header's semantic [`SchemaVersion`].
    pub fn schema_version(&self) -> Result<SchemaVersion, DnamError> {
        let ver = SchemaVersion::parse(&self.version)?;
        ver.ensure_supported()?;
        Ok(ver)
    }

    /// Validates all structural, alignment, and safety invariants of the JSON header.
    pub fn validate(&self) -> Result<(), DnamError> {
        let _ = self.schema_version()?;
        if self.container_kind != "dnam" && self.container_kind != "dnam-ckpt" {
            return Err(DnamError::SchemaValidationError(
                "container_kind must be 'dnam' or 'dnam-ckpt'".to_string(),
            ));
        }
        if self.sample_rate_train_hz < 44_100 || self.sample_rate_train_hz > 96_000 {
            return Err(DnamError::InvalidSampleRate(self.sample_rate_train_hz));
        }
        if self.controls_schema.len() > MAX_CONTROLS {
            return Err(DnamError::SchemaValidationError(
                "controls_schema exceeds maximum of 16 controls".to_string(),
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

    /// Compiles `controls_schema` into the 16-slot binary [`ControlClampDescriptor`] table
    /// (`1,024 B = 16 × 64 B` cache lines) used at $3\text{ kHz}$ with zero string parsing and zero `FDIV`.
    pub fn compile_clamp_descriptors(
        &self,
    ) -> Result<[ControlClampDescriptor; MAX_CONTROLS], DnamError> {
        if self.controls_schema.len() > MAX_CONTROLS {
            return Err(DnamError::SchemaValidationError(
                "controls_schema exceeds MAX_CONTROLS (16)".to_string(),
            ));
        }
        let mut table = [ControlClampDescriptor::empty(); MAX_CONTROLS];

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
                    let mut anchors = [0.0_f32; 4];
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
                                _pad: 0,
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
                                _pad: 0,
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
                        anchor_nodes_norm: [0.0; 4],
                        cond_rules: [ConditionalClampRule::empty(); 3],
                    };
                }
            }
        }

        Ok(table)
    }

    fn resolve_switch_condition(&self, cond_key: &str) -> Option<(u8, u8)> {
        let (sw_id, opt_id) = cond_key.split_once('=')?;
        for (idx, ctrl) in self.controls_schema.iter().enumerate() {
            if ctrl.id == sw_id
                && ctrl.kind == ControlKind::Switch
                && let Some(ref opts) = ctrl.declared_options
                && let Some(opt_idx) = opts.iter().position(|o| o == opt_id)
            {
                return Some((idx as u8, opt_idx as u8));
            }
        }
        None
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
            "core1_hybrid_ssm": {},
            "core2_mufdn": {},
            "slimmable_tiers": {
                "lite": { "channels": 4, "fdn_lines": 4, "arena_end_byte_offset": 10752, "prefix_crc32c": 0 },
                "full": { "channels": 8, "fdn_lines": 8, "arena_end_byte_offset": 22336, "prefix_crc32c": 0 },
                "ultra": { "channels": 12, "fdn_lines": 8, "arena_end_byte_offset": 36224, "prefix_crc32c": 0 }
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
        assert_eq!(h.version, "1.0.0");
        assert_eq!(h.container_kind, "dnam");
        assert_eq!(h.architecture, "DNAM_TriPhysics");
        assert_eq!(h.sample_rate_train_hz, 48_000);
        assert_eq!(h.metadata.input_reference_dbfs, -20.5);
        assert_eq!(h.io_config.inputs, 2);
        assert_eq!(h.slimmable_tiers.lite.head_svd_rank, 4);
        assert_eq!(h.control_bus.decimation, 16);
        assert_eq!(h.core1_hybrid_ssm.blocks, 4);
        assert_eq!(h.core2_mufdn.dc_blocker_pole, 0.995);
        assert!(h.validate().is_ok());

        let compiled = h.compile_clamp_descriptors().unwrap();
        assert_eq!(compiled[0].cond_rules[0].rule_op, 1);
        assert_eq!(compiled[0].cond_rules[1].rule_op, 2);
        assert_eq!(compiled[1].active_options_mask, 1);

        // Validation error branches
        h.container_kind = "invalid".to_string();
        assert!(h.validate().is_err());
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

        h.controls_schema = vec![h.controls_schema[0].clone(); 17];
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
    }
}
