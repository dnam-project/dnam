#![forbid(unsafe_code)]
//! Canonical `v2.0` `.dnam-proj` (`DNPJ`) RFC 8785 JCS Project Manifest Schema (`proj::schema`).
//!
//! Canonical Reference: `docs/specs/containers/project_container_format.md` (`ref[container.proj.jcs-manifest]`).

use std::collections::BTreeMap;

use dnam_engine::container::{parse_strict_json_object_bytes, serialize_jcs_value};
use dnam_engine::schema::{
    ControlKind, DnamError, DnamJsonHeader, SchemaVersion, check_no_unknown_header_keys,
};
use serde::{Deserialize, Serialize};

const fn default_true() -> bool {
    true
}

fn default_proj_version() -> String {
    "2.0.0".to_string()
}

fn default_proj_container_kind() -> String {
    "dnam-proj".to_string()
}

const fn default_device_sample_rate() -> u32 {
    48_000
}

fn default_bit_depth_tag() -> String {
    "int24".to_string()
}

const fn default_one_u8() -> u8 {
    1
}

/// Project-level authoring metadata inside [`DnamProjManifest`].
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProjectMeta {
    /// Stable project UUID or slug.
    #[serde(default)]
    pub project_id: String,
    /// Optional human-readable rig / reamp notes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rig_notes: Option<String>,
    /// ISO-8601 UTC timestamp of the last project mutation.
    #[serde(default)]
    pub updated_at: String,
}

/// Hardware / virtual audio I/O routing metadata recorded in [`DnamProjManifest`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RigRoutingConfig {
    /// Native capture interface sample rate in Hz (`32000..=192000`).
    #[serde(default = "default_device_sample_rate")]
    pub sample_rate_device_hz: u32,
    /// Native capture word format (`"int12"`, `"int16"`, `"int24"`, `"int32"`, or `"float32"`).
    #[serde(default = "default_bit_depth_tag")]
    pub bit_depth: String,
    /// Number of active dry send channels (`1..=4`).
    #[serde(default = "default_one_u8")]
    pub input_channels: u8,
    /// Number of active wet return channels (`1..=4`).
    #[serde(default = "default_one_u8")]
    pub output_channels: u8,
    /// 1-based hardware output channel index feeding the reamp send (`>= 1`).
    #[serde(default = "default_one_u8")]
    pub reamp_send_channel: u8,
    /// 1-based hardware input channel indices capturing the wet return.
    #[serde(default)]
    pub wet_return_channels: Vec<u8>,
}

impl Default for RigRoutingConfig {
    fn default() -> Self {
        Self {
            sample_rate_device_hz: 48_000,
            bit_depth: "int24".to_string(),
            input_channels: 1,
            output_channels: 2,
            reamp_send_channel: 1,
            wet_return_channels: vec![1, 2],
        }
    }
}

/// Loopback alignment, polarity, and reference level summary mirrored from `CALB`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CalibrationSummary {
    /// Sub-sample round-trip loopback latency in samples at `48 kHz`.
    pub loopback_lag_samples: f64,
    /// Signed linear loopback gain.
    pub signed_gain: f64,
    /// True if the wet return was detected polarity-inverted.
    pub polarity_inverted: bool,
    /// Calibrated input reference level in dBFS (`-20.5` default).
    pub input_reference_dbfs: f32,
    /// Measured quiescent hardware noise floor in dBFS.
    pub noise_floor_dbfs: f32,
}

/// SHA-256 content provenance of a captured stem's dry and wet PCM streams.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StemProvenance {
    /// Canonical `<f4` SHA-256 hex digest (`64` lowercase hex chars) of the dry stimulus.
    pub input_pcm_sha256: String,
    /// Canonical `<f4` SHA-256 hex digest (`64` lowercase hex chars) of the aligned wet return.
    pub target_pcm_sha256: String,
    /// Number of aligned frames at canonical `48 kHz`.
    pub num_frames: u64,
}

/// Extracted white-box / blind physical descriptors cached per stem in [`StemRegistryEntry`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ExtractedPhysics {
    /// Extracted delay-line times in milliseconds.
    #[serde(default)]
    pub delays_ms: Vec<f32>,
    /// Extracted LFO modulation rates in Hz.
    #[serde(default)]
    pub lfo_freqs_hz: Vec<f32>,
    /// Extracted feedback / decay loop gains.
    #[serde(default)]
    pub decay_gains_lin: Vec<f32>,
    /// Detected pitch/granular window length in milliseconds (`0.0` if none).
    #[serde(default)]
    pub detected_grain_size_ms: f32,
    /// Detected pitch shift interval in semitones (`0.0` if none).
    #[serde(default)]
    pub detected_pitch_shift_semitones: f32,
}

/// Per-capture entry in `manifest.stems_registry` (`r[container.proj.manifest]`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StemRegistryEntry {
    /// Unique human-readable stem identifier within the project.
    pub stem_id: String,
    /// Numeric `chunk_id` linking this stem to its `STMC` (and optional `SPCM`) chunk in Region 1.
    pub chunk_id: u32,
    /// Non-destructive Capture Vault mute/enable flag (`true` = included in active solve & audit).
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// True if this capture is the `StarCenter` (`s = 0`) anchor backing the `STAT` chunk.
    #[serde(default)]
    pub is_star_center: bool,
    /// Physical control coordinates (`control_id -> f64 | String`) active during this capture.
    #[serde(default)]
    pub controls_meta: BTreeMap<String, serde_json::Value>,
    /// Dry/wet PCM SHA-256 digests and frame count.
    pub provenance: StemProvenance,
    /// Open dictionary for creator tags/notes on this stem (excluded from immutable JCS digest).
    #[serde(default)]
    pub user_metadata: BTreeMap<String, serde_json::Value>,
    /// Optional extracted physical descriptors.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extracted_physics: Option<ExtractedPhysics>,
}

/// 3-Phase Control-Space Safety Audit report stored in [`DnamProjManifest`].
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AuditReport {
    /// Phase-1 coverage clamp tags (`"CLAMP-RANGE"`, `"CLAMP-SWITCH"`, `"CLAMP-CORNER"`).
    #[serde(default)]
    pub phase1_clamps: Vec<String>,
    /// Phase-2 nonlinear rail occupancy warnings.
    #[serde(default)]
    pub phase2_rail_warnings: Vec<String>,
    /// Phase-3 grid/corner stability clamp tags (`"CLAMP-STABILITY"`, `"CLAMP-DYNAMIC"`).
    #[serde(default)]
    pub phase3_clamps: Vec<String>,
    /// Actionable prescriptions telling the user which additional captures will widen clamps.
    #[serde(default)]
    pub prescriptions: Vec<String>,
}

/// Analytical `VarPro` solver timing and per-tier ESR telemetry stored in [`DnamProjManifest`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SolverTelemetrySummary {
    /// Count of enabled stems (`enabled == true`) included in the latest solve.
    pub active_stem_count: u32,
    /// Wall-clock solve duration in milliseconds.
    pub total_solve_ms: f32,
    /// Per-tier (`[Lite, Full, Pro, Ultra]`) validation ESR in dB.
    pub tier_esr_db: [f32; 4],
}

/// Canonical Region 2 UTF-8 JCS Project Manifest for `.dnam-proj` (`DNPJ`, `version == "2.0.0"`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DnamProjManifest {
    /// Semantic version of the `.dnam-proj` manifest schema (`"2.0.0"` only).
    #[serde(default = "default_proj_version")]
    pub version: String,
    /// Container discriminator (`"dnam-proj"` only).
    #[serde(default = "default_proj_container_kind")]
    pub container_kind: String,
    /// Embedded runtime [`DnamJsonHeader`] (`base.container_kind == "dnam"`, `base.version == "2.0.0"`).
    pub base: DnamJsonHeader,
    /// Project identity and authoring notes.
    #[serde(default)]
    pub project_meta: ProjectMeta,
    /// Hardware / virtual audio routing metadata.
    #[serde(default)]
    pub rig_routing: RigRoutingConfig,
    /// Optional loopback calibration summary (`CALB`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_summary: Option<CalibrationSummary>,
    /// Ordered Capture Vault stem registry (`STMC` / `SPCM` entries).
    #[serde(default)]
    pub stems_registry: Vec<StemRegistryEntry>,
    /// Optional 3-Phase control-space safety audit report.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audit_report: Option<AuditReport>,
    /// Optional solver timing and per-tier ESR telemetry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solver_telemetry: Option<SolverTelemetrySummary>,
}

fn is_valid_lower_hex_64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

impl DnamProjManifest {
    /// Constructs a canonical `v2.0` reference [`DnamProjManifest`] for tests and new projects.
    #[must_use]
    pub fn default_reference() -> Self {
        let mut base = DnamJsonHeader::default_reference();
        base.version = "2.0.0".to_string();
        base.container_kind = "dnam".to_string();
        Self {
            version: "2.0.0".to_string(),
            container_kind: "dnam-proj".to_string(),
            base,
            project_meta: ProjectMeta {
                project_id: "proj_reference_v2".to_string(),
                rig_notes: None,
                updated_at: "2026-11-11T00:00:00Z".to_string(),
            },
            rig_routing: RigRoutingConfig::default(),
            calibration_summary: None,
            stems_registry: Vec::new(),
            audit_report: None,
            solver_telemetry: None,
        }
    }

    /// Validates all `v2.0`-only version, schema, routing, and stem registry invariants.
    pub fn validate(&self) -> Result<(), DnamError> {
        let ver = SchemaVersion::parse(&self.version)?;
        if ver != SchemaVersion::V2_0_0 {
            return Err(DnamError::UnsupportedVersion {
                major: ver.major,
                minor: ver.minor,
            });
        }
        if self.container_kind != "dnam-proj" {
            return Err(DnamError::SchemaValidationError(
                "DnamProjManifest container_kind must be 'dnam-proj'".to_string(),
            ));
        }
        if self.base.container_kind != "dnam" {
            return Err(DnamError::SchemaValidationError(
                "DnamProjManifest base.container_kind must be 'dnam'".to_string(),
            ));
        }
        let base_ver = SchemaVersion::parse(&self.base.version)?;
        if base_ver != SchemaVersion::V2_0_0 {
            return Err(DnamError::UnsupportedVersion {
                major: base_ver.major,
                minor: base_ver.minor,
            });
        }
        self.base.validate()?;

        if !(8_000..=384_000).contains(&self.rig_routing.sample_rate_device_hz) {
            return Err(DnamError::InvalidSampleRate(
                self.rig_routing.sample_rate_device_hz,
            ));
        }
        if !matches!(
            self.rig_routing.bit_depth.as_str(),
            "int12" | "int16" | "int24" | "int32" | "float32"
        ) {
            return Err(DnamError::SchemaValidationError(format!(
                "unsupported rig_routing.bit_depth '{}'",
                self.rig_routing.bit_depth
            )));
        }
        if !(1..=4).contains(&self.rig_routing.input_channels)
            || !(1..=4).contains(&self.rig_routing.output_channels)
            || self.rig_routing.reamp_send_channel == 0
        {
            return Err(DnamError::SchemaValidationError(
                "rig_routing channels must be in 1..=4 and reamp_send_channel >= 1".to_string(),
            ));
        }

        if let Some(ref cal) = self.calibration_summary
            && (!cal.loopback_lag_samples.is_finite()
                || !cal.signed_gain.is_finite()
                || !cal.input_reference_dbfs.is_finite()
                || !cal.noise_floor_dbfs.is_finite())
        {
            return Err(DnamError::SchemaValidationError(
                "calibration_summary contains non-finite value".to_string(),
            ));
        }

        let mut seen_ids = std::collections::BTreeSet::new();
        let mut seen_chunks = std::collections::BTreeSet::new();
        let mut star_count = 0usize;
        for stem in &self.stems_registry {
            if stem.stem_id.trim().is_empty() {
                return Err(DnamError::SchemaValidationError(
                    "stems_registry.stem_id must not be empty".to_string(),
                ));
            }
            if !seen_ids.insert(stem.stem_id.as_str()) {
                return Err(DnamError::SchemaValidationError(format!(
                    "duplicate stem_id '{}' in stems_registry",
                    stem.stem_id
                )));
            }
            if !seen_chunks.insert(stem.chunk_id) {
                return Err(DnamError::SchemaValidationError(format!(
                    "duplicate chunk_id {} in stems_registry",
                    stem.chunk_id
                )));
            }
            if stem.is_star_center && stem.enabled {
                star_count += 1;
            }
            if !is_valid_lower_hex_64(&stem.provenance.input_pcm_sha256)
                || !is_valid_lower_hex_64(&stem.provenance.target_pcm_sha256)
                || stem.provenance.num_frames == 0
            {
                return Err(DnamError::SchemaValidationError(format!(
                    "invalid provenance SHA-256 or zero num_frames on stem '{}'",
                    stem.stem_id
                )));
            }
        }
        if star_count > 1 {
            return Err(DnamError::SchemaValidationError(
                "at most one enabled stem may be marked is_star_center".to_string(),
            ));
        }

        if let Some(ref tel) = self.solver_telemetry
            && (!tel.total_solve_ms.is_finite()
                || tel.total_solve_ms < 0.0
                || tel.tier_esr_db.iter().any(|v| !v.is_finite()))
        {
            return Err(DnamError::SchemaValidationError(
                "solver_telemetry contains non-finite or negative value".to_string(),
            ));
        }
        Ok(())
    }

    /// Serializes this manifest to canonical RFC 8785 JCS UTF-8 bytes after validating invariants.
    pub fn to_jcs_bytes(&self) -> Result<Vec<u8>, DnamError> {
        let mut normalized = self.clone();
        for ctrl in &mut normalized.base.controls_schema {
            if ctrl.kind == ControlKind::Range
                && let Some(f) = ctrl.default.as_f64()
            {
                ctrl.default = serde_json::Value::from(f);
            }
        }
        normalized.validate()?;
        let raw_bytes = serde_json::to_vec(&normalized)
            .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?;
        let val = parse_strict_json_object_bytes(&raw_bytes)?;
        let jcs = serialize_jcs_value(&val, false)?;
        Ok(jcs.into_bytes())
    }

    /// Parses, validates, rejects duplicate JSON keys and unknown fields inside `base`, and returns canonical JCS UTF-8 JSON.
    pub fn validate_and_normalize_json_str(json_str: &str) -> Result<String, DnamError> {
        let raw_val = parse_strict_json_object_bytes(json_str.as_bytes())?;
        let manifest: Self = serde_json::from_str(json_str)
            .map_err(|e| DnamError::SchemaValidationError(e.to_string()))?;
        manifest.validate()?;
        if let Some(in_base) = raw_val.get("base") {
            let norm_base = serde_json::to_value(&manifest.base)
                .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?;
            check_no_unknown_header_keys(in_base, &norm_base, None)?;
        }
        let jcs_bytes = manifest.to_jcs_bytes()?;
        String::from_utf8(jcs_bytes).map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))
    }

    /// Computes the SHA-256 digest of the immutable JCS projection of this manifest
    /// (stripping mutable `base.presets`, `base.user_metadata`, `controls_schema[*].user_metadata`,
    /// `stems_registry[*].user_metadata`, and `project_meta.rig_notes`).
    pub fn immutable_jcs_sha256(&self) -> Result<[u8; 32], DnamError> {
        use sha2::{Digest, Sha256};

        let mut clean = self.clone();
        clean.base.presets.clear();
        clean.base.user_metadata.clear();
        clean.base.metadata.signatures = None;
        for ctrl in &mut clean.base.controls_schema {
            ctrl.user_metadata.clear();
        }
        for stem in &mut clean.stems_registry {
            stem.user_metadata.clear();
        }
        clean.project_meta.rig_notes = None;
        clean.project_meta.updated_at.clear();

        let jcs_bytes = clean.to_jcs_bytes()?;
        Ok(Sha256::digest(&jcs_bytes).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serde_defaults_for_rig_routing_and_stem_entry() {
        let rig: RigRoutingConfig = serde_json::from_str("{}").expect("empty rig defaults");
        assert_eq!(rig.sample_rate_device_hz, 48_000);
        assert_eq!(rig.bit_depth, "int24");
        assert_eq!(rig.input_channels, 1);
        assert_eq!(rig.output_channels, 1);
        assert_eq!(rig.reamp_send_channel, 1);

        let stem: StemRegistryEntry = serde_json::from_value(serde_json::json!({
            "stem_id": "s0",
            "chunk_id": 0,
            "provenance": {
                "input_pcm_sha256": "0".repeat(64),
                "target_pcm_sha256": "1".repeat(64),
                "num_frames": 1024
            }
        }))
        .expect("minimal stem defaults");
        assert!(stem.enabled);
        assert!(!stem.is_star_center);
    }
}
