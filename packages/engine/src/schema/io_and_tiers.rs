//! Audio I/O topologies, leveling configurations, and slimmable tier profiles.
//!
//! Canonical Reference: `docs/containers/dnam_binary_format.md` and `docs/overview/psychoacoustic_quality_and_tiers.md`.

use alloc::string::String;
use serde::{Deserialize, Serialize};

pub const fn default_true() -> bool {
    true
}

pub const fn default_blend_one() -> f32 {
    1.0
}

pub const fn default_input_ref_dbfs() -> f32 {
    -20.5
}

pub const fn default_inputs() -> u8 {
    2
}

pub fn default_output_mode() -> String {
    "stereo".into()
}

pub const fn default_head_svd_rank() -> u8 {
    4
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
    /// `true` if engaging spectral freeze also mutes the live dry/Stage I signal path (stutter/hold replacement);
    /// `false` (default) allows "Sound-on-Sound" playing over the frozen Stage IV pad.
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
    #[serde(default = "default_input_ref_dbfs")]
    pub reference_input_level_db: f32,
    /// Measured model wet output level in dB at the default control settings ($\mathbf{u} = \mathbf{0}$).
    #[serde(default = "default_input_ref_dbfs")]
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

pub fn default_pro_tier_profile() -> TierProfile {
    TierProfile {
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
    }
}

/// Slimmable tier profiles (`lite`, `full`, `pro`, `ultra`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SlimmableTiersConfig {
    /// `Lite` ($C=4$, $M=4$, $D_h=4$, `10,752 B`) tier profile.
    pub lite: TierProfile,
    /// `Full` ($C=8$, $M=8$, $D_h=4$, `22,336 B`) tier profile.
    pub full: TierProfile,
    /// `Pro` ($C=12$, $M=8$, $D_h=4$, `36,224 B`) tier profile.
    #[serde(default = "default_pro_tier_profile")]
    pub pro: TierProfile,
    /// `Ultra` ($C=16$, $M=16$, $D_h=8$, `64,000 B`) tier profile.
    pub ultra: TierProfile,
}

/// Metadata and prefix checksum for an individual slimmable tier.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TierProfile {
    /// Dimension of the shared physical feature bus ($D_{\text{feat}} \in \{64, 128, 192, 256\}$).
    #[serde(default)]
    pub feature_bus_dim: u16,
    /// Active channels ($C \in \{4, 8, 12, 16\}$).
    pub channels: u8,
    /// Active Stage III direct SIMD early-reflection FIR taps ($M_{\text{early}} \in \{16, 32, 48, 64\}$).
    #[serde(default)]
    pub early_fir_taps: u8,
    /// Active Stage III Orthonormal Kautz modal bank modes ($M_{\text{kautz}} \in \{4, 8, 16, 32\}$).
    #[serde(default)]
    pub kautz_modes: u8,
    /// Active Stage IV tank delay lines ($M \in \{4, 8, 8, 16\}$).
    pub tank_lines: u8,
    /// SVD output head rank ($D_h \in \{4, 8\}$).
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

impl TierProfile {
    /// Early reflection taps accessor (`early_fir_taps`).
    #[inline]
    pub const fn early_taps(&self) -> u8 {
        self.early_fir_taps
    }

    /// Tank delay lines accessor (`tank_lines`).
    #[inline]
    pub const fn fdn_lines(&self) -> u8 {
        self.tank_lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_io_and_tiers_defaults() {
        assert!(default_true());
        assert_eq!(default_blend_one(), 1.0);
        assert_eq!(default_input_ref_dbfs(), -20.5);
        assert_eq!(default_inputs(), 2);
        assert_eq!(default_output_mode(), "stereo");
        assert_eq!(default_head_svd_rank(), 4);

        let mc = ModelCapabilities::default();
        assert_eq!(mc.blend_mode, BlendMode::None);
        assert_eq!(mc.default_blend, 1.0);

        let lc = LevelingConfig::default();
        assert!(lc.auto_unity_enabled);
        assert_eq!(lc.leveling_metric, LevelingMetric::RmsDbfs);

        let io = IoConfig::default();
        assert_eq!(io.inputs, 2);
        assert_eq!(io.latency_samples, 0);

        let pro = default_pro_tier_profile();
        assert_eq!(pro.feature_bus_dim, 192);
        assert_eq!(pro.channels, 12);
        assert_eq!(pro.early_fir_taps, 48);
        assert_eq!(pro.early_taps(), 48);
        assert_eq!(pro.kautz_modes, 16);
        assert_eq!(pro.tank_lines, 8);
        assert_eq!(pro.fdn_lines(), 8);
    }
}
