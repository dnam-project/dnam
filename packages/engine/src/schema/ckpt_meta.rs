//! `.dnam-ckpt` stem registry, training provenance, and clamp audit report structures.
//!
//! Canonical Reference: `docs/containers/checkpoint_and_cache_format.md` and `docs/training/clamp_and_audit_registry.md`.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use super::json_wire::TensorSliceEntry;

/// `.dnam-ckpt` per-stem registry record (`stems_registry[*]`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StemRegistryEntry {
    /// Unique stem identifier (e.g., `"stem_00_noon"`).
    pub stem_id: String,
    /// Control settings captured for this stem.
    pub controls_meta: BTreeMap<String, serde_json::Value>,
    /// Platform-agnostic IEEE-754 PCM SHA-256 provenance digests.
    pub provenance: StemProvenance,
    /// Castagnoli `CRC32C` over this stem's $267,264\text{ B}$ `Arena 4` record.
    #[serde(default)]
    pub stem_arena4_crc32c: u32,
    /// Open dictionary for stem-specific user metadata.
    #[serde(default)]
    pub user_metadata: BTreeMap<String, serde_json::Value>,
    /// Extracted Stage-1 physical anchors (delays, LFO frequencies, decay gains).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extracted_physics: Option<ExtractedPhysics>,
    /// `Arena 4` slice descriptor for $\operatorname{pack}_{\text{U}}(\mathbf{G}_{\text{audio}}^{(s)})$ (`263,168 B`).
    pub g_audio_uplo_u_f64: TensorSliceEntry,
    /// `Arena 4` slice descriptor for tier-ordered $\mathbf{C}_{\text{audio}}^{(s)}$ (`4,096 B`).
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
    /// Extracted Stage IV tank delay times in milliseconds.
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
