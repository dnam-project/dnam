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
        self.base.validate()?;
        self.base.ensure_not_preview_for_daw_host()?;
        if let Some(stems) = &self.stems_registry {
            if u16::try_from(stems.len()).is_err() {
                return Err(DnamError::SchemaValidationError(
                    "stems_registry exceeds u16::MAX (65535) entries".to_string(),
                ));
            }
            for stem in stems {
                if stem.stem_id.trim().is_empty() {
                    return Err(DnamError::SchemaValidationError(
                        "stems_registry entry stem_id must not be empty".to_string(),
                    ));
                }
                if let Some(phys) = &stem.extracted_physics
                    && (!phys.detected_grain_size_ms.is_finite()
                        || phys.detected_grain_size_ms < 0.0
                        || !phys.detected_pitch_shift_semitones.is_finite()
                        || phys
                            .delays_ms
                            .iter()
                            .chain(&phys.lfo_freqs_hz)
                            .any(|&v| !v.is_finite() || v < 0.0)
                        || phys.decay_gains_lin.iter().any(|&v| !v.is_finite()))
                {
                    return Err(DnamError::SchemaValidationError(
                        "stems_registry extracted_physics contains non-finite or negative value"
                            .to_string(),
                    ));
                }
                for (off_field, len_field, slice) in [
                    (
                        "stems_registry.g_audio_uplo_u_f64.byte_offset",
                        "stems_registry.g_audio_uplo_u_f64.byte_length",
                        &stem.g_audio_uplo_u_f64,
                    ),
                    (
                        "stems_registry.c_audio_tier_f64.byte_offset",
                        "stems_registry.c_audio_tier_f64.byte_length",
                        &stem.c_audio_tier_f64,
                    ),
                ] {
                    if slice.arena != 4 || slice.byte_length == 0 || slice.dtype != "f64" {
                        return Err(DnamError::SchemaValidationError(
                            "stems_registry slice must target arena 4 with non-zero f64 length"
                                .to_string(),
                        ));
                    }
                    if (slice.byte_offset & 0x3F) != 0 {
                        return Err(DnamError::UnalignedOffset {
                            field: off_field,
                            offset: slice.byte_offset,
                        });
                    }
                    if (slice.byte_length & 0x3F) != 0 {
                        return Err(DnamError::UnalignedOffset {
                            field: len_field,
                            offset: slice.byte_length,
                        });
                    }
                    if slice.byte_offset.checked_add(slice.byte_length).is_none() {
                        return Err(DnamError::SchemaValidationError(
                            "stems_registry slice offset + length overflow".to_string(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Deserializes a [`DnamCkptHeader`] from UTF-8 JSON bytes in a single non-generic compilation unit.
    #[inline(never)]
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }

    /// Deserializes a [`DnamCkptHeader`] from a UTF-8 JSON string (delegating to [`Self::from_json_slice`]).
    #[inline]
    pub fn from_json_str(s: &str) -> Result<Self, serde_json::Error> {
        Self::from_json_slice(s.as_bytes())
    }

    /// Serializes this [`DnamCkptHeader`] to a UTF-8 JSON byte vector in a single non-generic compilation unit.
    #[inline(never)]
    pub fn to_json_vec(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    /// Serializes this [`DnamCkptHeader`] to a UTF-8 JSON string (delegating to [`Self::to_json_vec`]).
    #[inline]
    pub fn to_json_string(&self) -> Result<String, serde_json::Error> {
        let bytes = self.to_json_vec()?;
        Ok(String::from_utf8(bytes).expect("serde_json::to_vec must emit valid UTF-8"))
    }

    /// Parses, normalizes numeric range defaults/presets/stems, validates structural invariants,
    /// rejects unknown schema fields, and serializes a [`DnamCkptHeader`] to canonical wire JSON.
    pub fn validate_and_normalize_json_str(header_json: &str) -> Result<String, DnamError> {
        let input_val: serde_json::Value = serde_json::from_slice(header_json.as_bytes())
            .map_err(|e| DnamError::InvalidJsonHeader(format!("{e}")))?;
        let mut header: Self = Self::from_json_slice(header_json.as_bytes())
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
        let out_bytes = header
            .to_json_vec()
            .map_err(|e| DnamError::InvalidJsonHeader(format!("{e}")))?;
        let norm_val: serde_json::Value = serde_json::from_slice(&out_bytes)
            .map_err(|e| DnamError::InvalidJsonHeader(format!("{e}")))?;
        check_no_unknown_header_keys(&input_val, &norm_val, None)?;
        String::from_utf8(out_bytes).map_err(|e| DnamError::InvalidJsonHeader(format!("{e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dnam_ckpt_header_validate_and_normalize() {
        let mut hdr = DnamCkptHeader::default_reference();
        assert_eq!(hdr.base.container_kind, "dnam-ckpt");
        hdr.base.controls_schema.push(
            serde_json::from_value(serde_json::json!({
                "id": "gain", "name": "Gain", "type": "range",
                "lobatto_degree": 0, "declared_range": [0.0, 10.0],
                "safe_clamped_range": [0.0, 10.0], "default": 5
            }))
            .unwrap(),
        );
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

        let mut preset_ctrls = BTreeMap::new();
        if let Some(first_ctrl) = hdr.base.controls_schema.first() {
            preset_ctrls.insert(first_ctrl.id.clone(), serde_json::json!(4));
        }
        hdr.base.presets = vec![dnam_engine::schema::PresetDef {
            id: "p0".to_string(),
            name: "Preset 0".to_string(),
            description: None,
            controls: preset_ctrls,
            user_metadata: BTreeMap::new(),
        }];

        let raw_json = hdr.to_json_string().unwrap();
        let parsed_str = DnamCkptHeader::from_json_str(&raw_json).unwrap();
        assert_eq!(parsed_str, hdr);
        let norm_json = DnamCkptHeader::validate_and_normalize_json_str(&raw_json).unwrap();
        let roundtrip: DnamCkptHeader = serde_json::from_str(&norm_json).unwrap();
        assert_eq!(roundtrip.stems_registry.as_ref().unwrap().len(), 1);
        assert!(roundtrip.stems_registry.as_ref().unwrap()[0].enabled);

        assert!(DnamCkptHeader::validate_and_normalize_json_str("{not json").is_err());
        assert!(DnamCkptHeader::validate_and_normalize_json_str(r#"{"unknown_field":1}"#).is_err());
        let mut with_unknown: serde_json::Value = serde_json::from_str(&raw_json).unwrap();
        with_unknown["unexpected_top_key"] = serde_json::json!(true);
        assert!(
            DnamCkptHeader::validate_and_normalize_json_str(&with_unknown.to_string()).is_err()
        );
        assert_eq!(StemProvenance::default().num_frames, 0);
        assert_eq!(ExtractedPhysics::default().detected_grain_size_ms, 0.0);
        assert!(AuditReport::default().phase1_clamps.is_empty());

        let mut bad_off = hdr.clone();
        bad_off.stems_registry.as_mut().unwrap()[0]
            .g_audio_uplo_u_f64
            .byte_offset = 10_241;
        assert!(bad_off.validate().is_err());

        let mut bad_len = hdr.clone();
        bad_len.stems_registry.as_mut().unwrap()[0]
            .c_audio_tier_f64
            .byte_length = 65;
        assert!(bad_len.validate().is_err());

        let mut bad_ovf = hdr.clone();
        bad_ovf.stems_registry.as_mut().unwrap()[0]
            .c_audio_tier_f64
            .byte_offset = !0x3F_u32;
        bad_ovf.stems_registry.as_mut().unwrap()[0]
            .c_audio_tier_f64
            .byte_length = 64;
        assert!(bad_ovf.validate().is_err());

        let mut preview_hdr = hdr.clone();
        preview_hdr.base.container_kind = "preview".to_string();
        assert!(preview_hdr.validate().is_err());

        let mut empty_id = hdr.clone();
        empty_id.stems_registry.as_mut().unwrap()[0].stem_id = "  ".to_string();
        assert!(empty_id.validate().is_err());

        let mut bad_dtype = hdr.clone();
        bad_dtype.stems_registry.as_mut().unwrap()[0]
            .g_audio_uplo_u_f64
            .dtype = "f32".to_string();
        assert!(bad_dtype.validate().is_err());

        let mut bad_phys = hdr;
        bad_phys.stems_registry.as_mut().unwrap()[0]
            .extracted_physics
            .as_mut()
            .unwrap()
            .delays_ms = vec![f32::NAN];
        assert!(bad_phys.validate().is_err());
    }
}
