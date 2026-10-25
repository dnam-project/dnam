//! Serde models for `dnam.toml` and `models/<id>/model.toml`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::capture::TargetExcitationProfile;

fn default_output_mode() -> String {
    "stereo".to_string()
}

fn default_topology() -> String {
    "auto".to_string()
}

fn default_noise_floor_dbfs() -> f32 {
    -96.0
}

/// Top-level `dnam.toml` workspace manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DnamProjectManifest {
    /// Schema version (`"1.0"`).
    pub schema_version: String,
    /// Workspace metadata.
    pub project: ProjectMetadata,
    /// Registered model IDs in `models/<id>/model.toml`.
    pub models: Vec<String>,
}

/// Workspace-level metadata in `dnam.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectMetadata {
    /// Human-readable workspace name.
    pub name: String,
    /// Creator or studio handle.
    pub author: String,
    /// Canonical sample rate in Hz (default `48000`).
    pub sample_rate_hz: u32,
    /// Public Ed25519 creator key in hex.
    pub creator_public_key_hex: String,
    /// Optional explicit path override for the creator signing key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signing_key_path: Option<String>,
}

/// Per-model hardware/software capture & calibration settings in `models/<id>/model.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptureSettings {
    /// Active backend (`"software"`, `"usb"`, or `"anagram"`).
    pub backend: String,
    /// Selected audio interface identifier.
    pub device_id: String,
    /// Round-trip calibrated latency in samples.
    pub calibrated_latency_samples: i32,
    /// Measured input noise floor in dBFS.
    pub noise_floor_dbfs: f32,
    /// Hardware send trim in dB (`0.0` default).
    #[serde(default)]
    pub send_gain_db: f32,
    /// Step-1 unity reference level in dBFS (`0.0` default).
    #[serde(default)]
    pub unity_ref_dbfs: f32,
    /// Probed target device excitation profile used to synthesize `models/<id>/stems/input.wav`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub excitation_profile: Option<TargetExcitationProfile>,
}

impl Default for CaptureSettings {
    fn default() -> Self {
        Self {
            backend: "software".to_string(),
            device_id: "loopback_default".to_string(),
            calibrated_latency_samples: 0,
            noise_floor_dbfs: -96.0,
            send_gain_db: 0.0,
            unity_ref_dbfs: 0.0,
            excitation_profile: None,
        }
    }
}

/// Continuous range parameter definition in `models/<id>/model.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControlRangeSpec {
    /// Unique identifier (`"gain"`, `"bass"`, etc.).
    pub id: String,
    /// Display label.
    pub label: String,
    /// Minimum dial value.
    pub min: f32,
    /// Maximum dial value.
    pub max: f32,
    /// Default dial value.
    pub default_value: f32,
    /// Optional engineering unit (`"dB"`, `"%"`, etc.).
    pub unit: Option<String>,
}

/// Single throw option of a discrete switch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControlSwitchOptionSpec {
    /// Zero-based throw index.
    pub index: usize,
    /// Human-readable label (`"Clean"`, `"Crunch"`, `"Lead"`).
    pub label: String,
}

/// Discrete switch definition in `models/<id>/model.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControlSwitchSpec {
    /// Unique identifier (`"voice"`, `"bright"`).
    pub id: String,
    /// Display label.
    pub label: String,
    /// Ordered throw options (`2..=8`).
    pub options: Vec<ControlSwitchOptionSpec>,
    /// Default zero-based option index.
    pub default_index: usize,
}

/// Named preset snapshot in `models/<id>/model.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresetSpec {
    /// Unique preset identifier.
    pub id: String,
    /// Display title.
    pub label: String,
    /// Continuous parameter values keyed by `ControlRangeSpec::id`.
    pub ranges: BTreeMap<String, f32>,
    /// Discrete switch indices keyed by `ControlSwitchSpec::id`.
    pub switches: BTreeMap<String, usize>,
}

/// Captured dry/wet stem entry in `models/<id>/model.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StemRecord {
    /// Unique stem identifier.
    pub stem_id: String,
    /// Relative path to dry excitation WAV.
    pub dry_wav_path: String,
    /// Relative path to wet captured WAV.
    pub wet_wav_path: String,
    /// Normalized `[0.0, 1.0]` continuous coordinates.
    pub normalized_ranges: BTreeMap<String, f32>,
    /// Zero-based discrete switch indices.
    pub switches: BTreeMap<String, usize>,
    /// IEEE-754 `<f4` SHA-256 hex digest of aligned wet PCM.
    pub pcm_sha256: String,
    /// IEEE-754 `<f4` SHA-256 hex digest of dry excitation PCM.
    #[serde(default)]
    pub dry_pcm_sha256: String,
    /// Sub-sample aligned latency in samples.
    pub aligned_lag_samples: f32,
    /// Signal-to-noise ratio in dB.
    pub snr_db: f32,
    /// Measured silent-window noise floor in dBFS.
    #[serde(default = "default_noise_floor_dbfs")]
    pub noise_floor_dbfs: f32,
}

/// Per-model manifest stored at `models/<id>/model.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Unique model identifier.
    pub model_id: String,
    /// Human-readable model title.
    pub title: String,
    /// Target tier (`"Lite"`, `"Full"`, or `"Ultra"`).
    pub tier: String,
    /// Optional model description.
    #[serde(default)]
    pub description: String,
    /// Output channel mode (`"stereo"`, `"mono"`, or `"dual_mono"`).
    #[serde(default = "default_output_mode")]
    pub output_mode: String,
    /// Architecture topology (`"auto"`, `"core1_only"`, or `"full"`).
    #[serde(default = "default_topology")]
    pub topology: String,
    /// Per-model capture & calibration settings.
    #[serde(default)]
    pub capture: CaptureSettings,
    /// Continuous range controls.
    pub ranges: Vec<ControlRangeSpec>,
    /// Discrete switch controls.
    pub switches: Vec<ControlSwitchSpec>,
    /// Saved presets.
    pub presets: Vec<PresetSpec>,
    /// Registered dry/wet stems.
    pub stems: Vec<StemRecord>,
}
