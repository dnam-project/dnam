#![forbid(unsafe_code)]
//! `.dnam-ckpt` stem registry, training provenance, clamp audit report, and [`DnamCkptHeader`]
//! structures (`ref[container.ckpt.stems-registry]`, `ref[container.ckpt.declarative-no-pickle]`).

use std::collections::BTreeMap;

use dnam_engine::schema::{
    ControlKind, DnamError, DnamJsonHeader, TensorSliceEntry, check_no_unknown_header_keys,
};
use serde::{Deserialize, Serialize};

const fn default_true() -> bool {
    true
}

/// `.dnam-ckpt` per-stem registry record (`stems_registry[*]`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StemRegistryEntry {
    /// Unique stem identifier (e.g., `"stem_00_noon"`).
    pub stem_id: String,
    /// Whether this stem is active for incremental re-fitting (`true` by default; `false` prunes stem without re-recording).
    #[serde(default = "default_true")]
    pub enabled: bool,
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

/// Top-level UTF-8 JSON header for a `.dnam-ckpt` (`DNCK`) training checkpoint container,
/// wrapping the runtime [`DnamJsonHeader`] with optional `stems_registry` and `audit_report`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DnamCkptHeader {
    /// Runtime `.dnam` header fields (`#[serde(flatten)]`).
    #[serde(flatten)]
    pub base: DnamJsonHeader,
    /// Optional `.dnam-ckpt` multi-stem registry mapping stem IDs to Stem Record slice offsets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stems_registry: Option<Vec<StemRegistryEntry>>,
    /// Optional `.dnam-ckpt` 3-phase clamp audit report.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audit_report: Option<AuditReport>,
}

impl DnamCkptHeader {
    /// Constructs a canonical reference [`DnamCkptHeader`].
    #[must_use]
    pub fn default_reference() -> Self {
        let mut base = DnamJsonHeader::default_reference();
        base.container_kind = "dnam-ckpt".to_string();
        Self {
            base,
            stems_registry: None,
            audit_report: None,
        }
    }

    /// Validates all structural, alignment, and safety invariants of the checkpoint header.
    pub fn validate(&self) -> Result<(), DnamError> {
        self.base.validate()
    }

    /// Parses, normalizes numeric range defaults/presets/stems, validates structural invariants,
    /// rejects unknown schema fields, and serializes a [`DnamCkptHeader`] to canonical wire JSON.
    pub fn validate_and_normalize_json_str(header_json: &str) -> Result<String, DnamError> {
        let input_val: serde_json::Value = serde_json::from_str(header_json)
            .map_err(|e| DnamError::InvalidJsonHeader(format!("{e}")))?;
        let mut header: Self = serde_json::from_value(input_val.clone())
            .map_err(|e| DnamError::SchemaValidationError(format!("{e}")))?;
        for ctrl in &mut header.base.controls_schema {
            if ctrl.kind == ControlKind::Range
                && let Some(f) = ctrl.default.as_f64()
            {
                ctrl.default = serde_json::Value::from(f);
            }
        }
        let range_ids: Vec<String> = header
            .base
            .controls_schema
            .iter()
            .filter(|c| c.kind == ControlKind::Range)
            .map(|c| c.id.clone())
            .collect();
        for preset in &mut header.base.presets {
            for (k, v) in &mut preset.controls {
                if range_ids.contains(k)
                    && let Some(f) = v.as_f64()
                {
                    *v = serde_json::Value::from(f);
                }
            }
        }
        if let Some(stems) = &mut header.stems_registry {
            for stem in stems {
                for (k, v) in &mut stem.controls_meta {
                    if range_ids.contains(k)
                        && let Some(f) = v.as_f64()
                    {
                        *v = serde_json::Value::from(f);
                    }
                }
            }
        }
        header.validate()?;
        let norm_val = serde_json::to_value(&header)
            .map_err(|e| DnamError::InvalidJsonHeader(format!("{e}")))?;
        check_no_unknown_header_keys(&input_val, &norm_val, None)?;
        serde_json::to_string(&header).map_err(|e| DnamError::InvalidJsonHeader(format!("{e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dnam_ckpt_header_validate_and_normalize() {
        let mut hdr = DnamCkptHeader::default_reference();
        assert_eq!(hdr.base.container_kind, "dnam-ckpt");
        hdr.validate().unwrap();

        let dummy_slice = TensorSliceEntry {
            arena: 4,
            dtype: "f64".to_string(),
            packing: Some("uplo_upper_col_major_tier_ordered".to_string()),
            feature_dim: Some(64),
            packed_len: Some(2080),
            shape: None,
            byte_offset: 10_240,
            byte_length: 64,
        };
        let mut controls_meta = BTreeMap::new();
        if let Some(first_ctrl) = hdr.base.controls_schema.first() {
            controls_meta.insert(first_ctrl.id.clone(), serde_json::json!(5));
        }
        hdr.stems_registry = Some(vec![StemRegistryEntry {
            stem_id: "stem_0".to_string(),
            enabled: default_true(),
            controls_meta,
            provenance: StemProvenance {
                input_pcm_sha256: "00".repeat(32),
                target_pcm_sha256: "11".repeat(32),
                num_frames: 48_000,
            },
            stem_arena4_crc32c: 0,
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
        hdr.audit_report = Some(AuditReport {
            phase1_clamps: vec!["CLAMP-RANGE".to_string()],
            phase2_rail_warnings: Vec::new(),
            phase3_clamps: Vec::new(),
            prescriptions: vec!["Record stem at gain=10".to_string()],
        });

        let raw_json = serde_json::to_string(&hdr).unwrap();
        let norm_json = DnamCkptHeader::validate_and_normalize_json_str(&raw_json).unwrap();
        let roundtrip: DnamCkptHeader = serde_json::from_str(&norm_json).unwrap();
        assert_eq!(roundtrip.stems_registry.as_ref().unwrap().len(), 1);
        assert!(roundtrip.stems_registry.as_ref().unwrap()[0].enabled);

        assert!(DnamCkptHeader::validate_and_normalize_json_str("{not json").is_err());
    }
}
