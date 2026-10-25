//! Workspace CRUD, atomic TOML persistence, Ed25519 key provisioning, and compact CLI parsers.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use ed25519_dalek::SigningKey;
use sha2::{Digest, Sha256};

use crate::error::{StudioError, StudioResult};
use crate::project::manifest::{
    CaptureSettings, ControlRangeSpec, ControlSwitchOptionSpec, ControlSwitchSpec,
    DnamProjectManifest, ModelManifest, PresetSpec, ProjectMetadata, StemRecord,
};

/// Parses a compact range specification `"id:Label:min..max@default[:unit]"`.
pub fn parse_compact_range_spec(spec: &str) -> StudioResult<ControlRangeSpec> {
    let parts: Vec<&str> = spec.split(':').collect();
    if parts.len() < 3 || parts.len() > 4 {
        return Err(StudioError::Validation(format!(
            "Invalid range spec '{spec}'; expected 'id:Label:min..max@default[:unit]'"
        )));
    }
    let id = parts[0].trim();
    let label = parts[1].trim();
    if id.is_empty() || label.is_empty() {
        return Err(StudioError::Validation(
            "Range id and label must be non-empty".to_string(),
        ));
    }
    let bounds_and_def: Vec<&str> = parts[2].split('@').collect();
    if bounds_and_def.len() != 2 {
        return Err(StudioError::Validation(format!(
            "Missing '@default' in range spec '{spec}'"
        )));
    }
    let min_max: Vec<&str> = bounds_and_def[0].split("..").collect();
    if min_max.len() != 2 {
        return Err(StudioError::Validation(format!(
            "Missing 'min..max' in range spec '{spec}'"
        )));
    }
    let min: f32 = min_max[0]
        .parse()
        .map_err(|_| StudioError::Validation(format!("Invalid min value in '{spec}'")))?;
    let max: f32 = min_max[1]
        .parse()
        .map_err(|_| StudioError::Validation(format!("Invalid max value in '{spec}'")))?;
    let default_value: f32 = bounds_and_def[1]
        .parse()
        .map_err(|_| StudioError::Validation(format!("Invalid default value in '{spec}'")))?;
    if max <= min || default_value < min || default_value > max {
        return Err(StudioError::Validation(format!(
            "Range bounds must satisfy min <= default <= max and min < max in '{spec}'"
        )));
    }
    let unit = parts
        .get(3)
        .map(|u| u.trim().to_string())
        .filter(|u| !u.is_empty());

    Ok(ControlRangeSpec {
        id: id.to_string(),
        label: label.to_string(),
        min,
        max,
        default_value,
        unit,
    })
}

/// Parses a compact switch specification `"id:Label:Opt0,Opt1,Opt2@default_idx"`.
pub fn parse_compact_switch_spec(spec: &str) -> StudioResult<ControlSwitchSpec> {
    let parts: Vec<&str> = spec.split(':').collect();
    if parts.len() != 3 {
        return Err(StudioError::Validation(format!(
            "Invalid switch spec '{spec}'; expected 'id:Label:Opt0,Opt1@default_idx'"
        )));
    }
    let id = parts[0].trim();
    let label = parts[1].trim();
    if id.is_empty() || label.is_empty() {
        return Err(StudioError::Validation(
            "Switch id and label must be non-empty".to_string(),
        ));
    }
    let opts_and_def: Vec<&str> = parts[2].split('@').collect();
    if opts_and_def.len() != 2 {
        return Err(StudioError::Validation(format!(
            "Missing '@default_idx' in switch spec '{spec}'"
        )));
    }
    let raw_labels: Vec<&str> = opts_and_def[0]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if raw_labels.len() < 2 {
        return Err(StudioError::Validation(format!(
            "Switch '{id}' must have at least 2 options"
        )));
    }
    let default_index: usize = opts_and_def[1]
        .parse()
        .map_err(|_| StudioError::Validation(format!("Invalid default index in '{spec}'")))?;
    if default_index >= raw_labels.len() {
        return Err(StudioError::Validation(format!(
            "Switch default index {default_index} out of bounds for {} options",
            raw_labels.len()
        )));
    }
    let options = raw_labels
        .into_iter()
        .enumerate()
        .map(|(index, opt_label)| ControlSwitchOptionSpec {
            index,
            label: opt_label.to_string(),
        })
        .collect();

    Ok(ControlSwitchSpec {
        id: id.to_string(),
        label: label.to_string(),
        options,
        default_index,
    })
}

/// Handle to a DNAM workspace directory containing `dnam.toml` and `models/<id>/model.toml`.
#[derive(Debug, Clone)]
pub struct DnamProject {
    root_dir: PathBuf,
    manifest: DnamProjectManifest,
}

impl DnamProject {
    /// Initializes a new DNAM workspace at `root_dir`, creating `dnam.toml`, `keys/`, and `models/`.
    pub fn init(root_dir: impl AsRef<Path>, name: &str, author: &str) -> StudioResult<Self> {
        let root = root_dir.as_ref().to_path_buf();
        fs::create_dir_all(root.join("models"))?;
        fs::create_dir_all(root.join("captures"))?;
        fs::create_dir_all(root.join("keys"))?;

        let mut hasher = Sha256::new();
        hasher.update(name.as_bytes());
        hasher.update(b":");
        hasher.update(author.as_bytes());
        let seed_bytes: [u8; 32] = hasher.finalize().into();
        let signing_key = SigningKey::from_bytes(&seed_bytes);
        let pub_hex = encode_hex(signing_key.verifying_key().as_bytes());
        let priv_hex = encode_hex(&signing_key.to_bytes());
        fs::write(root.join("keys/creator_ed25519.sec"), priv_hex)?;

        let manifest = DnamProjectManifest {
            schema_version: "1.0".to_string(),
            project: ProjectMetadata {
                name: name.to_string(),
                author: author.to_string(),
                sample_rate_hz: 48_000,
                creator_public_key_hex: pub_hex,
            },
            capture: CaptureSettings {
                backend: "software".to_string(),
                device_id: "loopback_default".to_string(),
                calibrated_latency_samples: 0,
                noise_floor_dbfs: -96.0,
            },
            models: Vec::new(),
        };

        let proj = Self {
            root_dir: root,
            manifest,
        };
        proj.save()?;
        Ok(proj)
    }

    /// Loads an existing DNAM workspace from `root_dir`.
    pub fn open(root_dir: impl AsRef<Path>) -> StudioResult<Self> {
        let root = root_dir.as_ref().to_path_buf();
        let manifest_path = root.join("dnam.toml");
        let content = fs::read_to_string(&manifest_path)?;
        let manifest: DnamProjectManifest =
            toml::from_str(&content).map_err(|e| StudioError::Toml(e.to_string()))?;
        Ok(Self {
            root_dir: root,
            manifest,
        })
    }

    /// Returns the workspace root directory.
    pub fn root_dir(&self) -> &Path {
        &self.root_dir
    }

    /// Returns the workspace manifest.
    pub fn manifest(&self) -> &DnamProjectManifest {
        &self.manifest
    }

    /// Updates capture calibration settings and saves `dnam.toml`.
    pub fn update_calibration(
        &mut self,
        backend: &str,
        device_id: &str,
        latency_samples: i32,
        noise_floor_dbfs: f32,
    ) -> StudioResult<()> {
        self.manifest.capture.backend = backend.to_string();
        self.manifest.capture.device_id = device_id.to_string();
        self.manifest.capture.calibrated_latency_samples = latency_samples;
        self.manifest.capture.noise_floor_dbfs = noise_floor_dbfs;
        self.save()
    }

    /// Atomically persists `dnam.toml`.
    pub fn save(&self) -> StudioResult<()> {
        let encoded =
            toml::to_string_pretty(&self.manifest).map_err(|e| StudioError::Toml(e.to_string()))?;
        atomic_write_text(&self.root_dir.join("dnam.toml"), &encoded)
    }

    /// Adds a new model to `models/<model_id>/model.toml`.
    pub fn add_model(
        &mut self,
        model_id: &str,
        title: &str,
        tier: &str,
    ) -> StudioResult<ModelManifest> {
        if model_id.trim().is_empty() {
            return Err(StudioError::Validation(
                "model_id must not be empty".to_string(),
            ));
        }
        let model_dir = self.root_dir.join("models").join(model_id);
        fs::create_dir_all(model_dir.join("stems"))?;
        fs::create_dir_all(model_dir.join("dist"))?;
        fs::create_dir_all(model_dir.join("reports"))?;

        let model = ModelManifest {
            model_id: model_id.to_string(),
            title: title.to_string(),
            tier: tier.to_string(),
            ranges: Vec::new(),
            switches: Vec::new(),
            presets: Vec::new(),
            stems: Vec::new(),
        };
        self.save_model(&model)?;
        if !self.manifest.models.iter().any(|m| m == model_id) {
            self.manifest.models.push(model_id.to_string());
            self.save()?;
        }
        Ok(model)
    }

    /// Loads `models/<model_id>/model.toml`.
    pub fn load_model(&self, model_id: &str) -> StudioResult<ModelManifest> {
        let path = self
            .root_dir
            .join("models")
            .join(model_id)
            .join("model.toml");
        let content = fs::read_to_string(&path)?;
        toml::from_str(&content).map_err(|e| StudioError::Toml(e.to_string()))
    }

    /// Atomically writes `models/<model_id>/model.toml`.
    pub fn save_model(&self, model: &ModelManifest) -> StudioResult<()> {
        let model_dir = self.root_dir.join("models").join(&model.model_id);
        fs::create_dir_all(&model_dir)?;
        let encoded =
            toml::to_string_pretty(model).map_err(|e| StudioError::Toml(e.to_string()))?;
        atomic_write_text(&model_dir.join("model.toml"), &encoded)
    }

    /// Adds a continuous range control to `model_id`.
    pub fn add_range_control(&self, model_id: &str, spec: ControlRangeSpec) -> StudioResult<()> {
        let mut model = self.load_model(model_id)?;
        model.ranges.retain(|r| r.id != spec.id);
        model.ranges.push(spec);
        self.save_model(&model)
    }

    /// Adds a discrete switch control to `model_id`.
    pub fn add_switch_control(&self, model_id: &str, spec: ControlSwitchSpec) -> StudioResult<()> {
        let mut model = self.load_model(model_id)?;
        model.switches.retain(|s| s.id != spec.id);
        model.switches.push(spec);
        self.save_model(&model)
    }

    /// Adds a named preset to `model_id`.
    pub fn add_preset(
        &self,
        model_id: &str,
        preset_id: &str,
        label: &str,
        ranges: BTreeMap<String, f32>,
        switches: BTreeMap<String, usize>,
    ) -> StudioResult<()> {
        let mut model = self.load_model(model_id)?;
        model.presets.retain(|p| p.id != preset_id);
        model.presets.push(PresetSpec {
            id: preset_id.to_string(),
            label: label.to_string(),
            ranges,
            switches,
        });
        self.save_model(&model)
    }

    /// Registers a captured dry/wet stem in `model_id`.
    pub fn register_stem(&self, model_id: &str, stem: StemRecord) -> StudioResult<()> {
        let mut model = self.load_model(model_id)?;
        model.stems.retain(|s| s.stem_id != stem.stem_id);
        model.stems.push(stem);
        self.save_model(&model)
    }
}

fn atomic_write_text(path: &Path, content: &str) -> StudioResult<()> {
    let tmp_path = path.with_extension("tmp");
    fs::write(&tmp_path, content)?;
    fs::rename(&tmp_path, path)?;
    Ok(())
}

fn encode_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}
