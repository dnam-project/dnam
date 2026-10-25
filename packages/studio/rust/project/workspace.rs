//! Workspace CRUD, atomic TOML persistence, OS-level `Ed25519` creator keypair provisioning
//! (`default_user_config_dir` / `ensure_creator_ed25519_keypair`), and compact CLI parsers.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use ed25519_dalek::SigningKey;
use sha2::{Digest, Sha256};

use crate::capture::{
    TargetExcitationProfile, align_hardware_latency, apply_subsample_shift_fft, compute_pcm_sha256,
    generate_excitation_signal_with_profile, read_wav, write_wav_24bit,
};
use crate::error::{StudioError, StudioResult};
use crate::project::manifest::{
    CaptureSettings, ControlRangeSpec, ControlSwitchOptionSpec, ControlSwitchSpec,
    DnamProjectManifest, ModelManifest, PresetSpec, ProjectMetadata, StemRecord,
};

const WORKSPACE_GITIGNORE: &str = "\
# DNAM Project Ignore Rules
*.dnam-ckpt
models/*/stems/input.wav
models/*/renders/
models/*/reports/*.parquet
models/*/reports/*.html
.cache/
";

/// Returns the OS-appropriate user configuration directory for `dnam`:
/// - Respects `DNAM_CONFIG_DIR` environment variable when set.
/// - macOS: `~/Library/Application Support/dnam`
/// - Windows: `%APPDATA%\dnam` (or `~/AppData/Roaming/dnam`)
/// - Linux / POSIX: `${XDG_CONFIG_HOME:-~/.config}/dnam`
pub fn default_user_config_dir() -> PathBuf {
    if let Ok(env_dir) = env::var("DNAM_CONFIG_DIR") {
        let trimmed = env_dir.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }

    let home = env::var("HOME")
        .or_else(|_| env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| env::temp_dir());

    if cfg!(target_os = "macos") {
        return home
            .join("Library")
            .join("Application Support")
            .join("dnam");
    }
    if cfg!(target_os = "windows") {
        if let Ok(appdata) = env::var("APPDATA") {
            let trimmed = appdata.trim();
            if !trimmed.is_empty() {
                return PathBuf::from(trimmed).join("dnam");
            }
        }
        return home.join("AppData").join("Roaming").join("dnam");
    }
    if let Ok(xdg) = env::var("XDG_CONFIG_HOME") {
        let trimmed = xdg.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed).join("dnam");
        }
    }
    home.join(".config").join("dnam")
}

/// Loads or generates the reusable 32-byte creator `Ed25519` keypair in the user config directory
/// (`<user_config_dir>/keys/creator_ed25519.key` or `DNAM_SIGNING_KEY`).
///
/// Returns `(resolved_key_path, seed_32_bytes, pubkey_ed25519_hex)`.
pub fn ensure_creator_ed25519_keypair(
    key_path: Option<&Path>,
) -> StudioResult<(PathBuf, [u8; 32], String)> {
    let resolved = if let Some(explicit) = key_path {
        explicit.to_path_buf()
    } else if let Ok(env_key) = env::var("DNAM_SIGNING_KEY")
        && !env_key.trim().is_empty()
    {
        PathBuf::from(env_key.trim())
    } else {
        default_user_config_dir()
            .join("keys")
            .join("creator_ed25519.key")
    };

    let seed_bytes: [u8; 32] = if resolved.is_file() {
        let raw_text = fs::read_to_string(&resolved)?;
        decode_hex_32(raw_text.trim()).ok_or_else(|| {
            StudioError::Validation(format!(
                "Invalid Ed25519 secret seed in {}: expected 64 hex chars (32 bytes)",
                resolved.display()
            ))
        })?
    } else {
        if let Some(parent) = resolved.parent() {
            fs::create_dir_all(parent)?;
        }
        let generated = generate_csprng_seed_32();
        let hex_line = format!("{}\n", encode_hex(&generated));
        fs::write(&resolved, hex_line)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let _ = fs::set_permissions(&resolved, fs::Permissions::from_mode(0o600));
        }
        generated
    };

    let signing_key = SigningKey::from_bytes(&seed_bytes);
    let pub_hex = encode_hex(signing_key.verifying_key().as_bytes());
    Ok((resolved, seed_bytes, pub_hex))
}

fn generate_csprng_seed_32() -> [u8; 32] {
    let mut buf = [0_u8; 32];
    if let Ok(mut urandom) = fs::File::open("/dev/urandom")
        && urandom.read_exact(&mut buf).is_ok()
    {
        return buf;
    }
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut hasher = Sha256::new();
    hasher.update(nanos.to_le_bytes());
    hasher.update(std::process::id().to_le_bytes());
    hasher.finalize().into()
}

fn humanize_id(id: &str) -> String {
    id.split('_')
        .filter(|s| !s.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<String>>()
        .join(" ")
}

/// Parses a compact range specification:
/// - `"id=min..max[:default]"` (e.g. `"drive=0..10:5"` or `"drive=0..10"`)
/// - `"id:Label:min..max@default[:unit]"` (e.g. `"gain:Preamp Gain:0..10@5:dB"`)
pub fn parse_compact_range_spec(spec: &str) -> StudioResult<ControlRangeSpec> {
    if let Some((lhs, rhs)) = spec.split_once('=') {
        let id = lhs.trim();
        if id.is_empty() {
            return Err(StudioError::Validation(format!(
                "Empty control id in '{spec}'"
            )));
        }
        let (bounds_part, def_opt) = if let Some((b, d)) = rhs.split_once(':') {
            (b.trim(), Some(d.trim()))
        } else {
            (rhs.trim(), None)
        };
        let (min_s, max_s) = bounds_part.split_once("..").ok_or_else(|| {
            StudioError::Validation(format!("Missing 'min..max' in compact range spec '{spec}'"))
        })?;
        let min: f32 = min_s
            .trim()
            .parse()
            .map_err(|_| StudioError::Validation(format!("Invalid min value in '{spec}'")))?;
        let max: f32 = max_s
            .trim()
            .parse()
            .map_err(|_| StudioError::Validation(format!("Invalid max value in '{spec}'")))?;
        if max <= min {
            return Err(StudioError::Validation(format!(
                "Range min ({min}) must be strictly less than max ({max}) in '{spec}'"
            )));
        }
        let default_value: f32 = match def_opt {
            Some(d) => d.parse().map_err(|_| {
                StudioError::Validation(format!("Invalid default value in '{spec}'"))
            })?,
            None => 0.5 * (min + max),
        };
        if default_value < min || default_value > max {
            return Err(StudioError::Validation(format!(
                "Default value {default_value} is outside declared range [{min}, {max}] in '{spec}'"
            )));
        }
        return Ok(ControlRangeSpec {
            id: id.to_string(),
            label: humanize_id(id),
            min,
            max,
            default_value,
            unit: None,
        });
    }

    let parts: Vec<&str> = spec.split(':').collect();
    if parts.len() < 3 || parts.len() > 4 {
        return Err(StudioError::Validation(format!(
            "Invalid range spec '{spec}'; expected 'id=min..max[:default]' or 'id:Label:min..max@default[:unit]'"
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

/// Parses a compact switch specification:
/// - `"id=opt0,opt1,...[:default]"` (e.g. `"attack=cut,flat,boost:flat"`)
/// - `"id:Label:Opt0,Opt1,Opt2@default_idx"` (e.g. `"voice:Voice:Clean,Crunch,Lead@1"`)
pub fn parse_compact_switch_spec(spec: &str) -> StudioResult<ControlSwitchSpec> {
    if let Some((lhs, rhs)) = spec.split_once('=') {
        let id = lhs.trim();
        if id.is_empty() {
            return Err(StudioError::Validation(format!(
                "Empty control id in '{spec}'"
            )));
        }
        let (opts_part, def_opt) = if let Some((o, d)) = rhs.split_once(':') {
            (o.trim(), Some(d.trim()))
        } else {
            (rhs.trim(), None)
        };
        let raw_labels: Vec<&str> = opts_part
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if raw_labels.len() < 2 {
            return Err(StudioError::Validation(format!(
                "Switch '{id}' must have at least 2 options in '{spec}'"
            )));
        }
        let default_index = match def_opt {
            Some(def_str) => {
                if let Some(pos) = raw_labels.iter().position(|&o| o == def_str) {
                    pos
                } else if let Ok(idx) = def_str.parse::<usize>()
                    && idx < raw_labels.len()
                {
                    idx
                } else {
                    return Err(StudioError::Validation(format!(
                        "Default option '{def_str}' is not in declared options for '{id}'"
                    )));
                }
            }
            None => 0,
        };
        let options = raw_labels
            .into_iter()
            .enumerate()
            .map(|(index, opt_label)| ControlSwitchOptionSpec {
                index,
                label: opt_label.to_string(),
            })
            .collect();
        return Ok(ControlSwitchSpec {
            id: id.to_string(),
            label: humanize_id(id),
            options,
            default_index,
        });
    }

    let parts: Vec<&str> = spec.split(':').collect();
    if parts.len() != 3 {
        return Err(StudioError::Validation(format!(
            "Invalid switch spec '{spec}'; expected 'id=opt0,opt1[:default]' or 'id:Label:Opt0,Opt1@default_idx'"
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
    /// Initializes a new DNAM workspace at `root_dir`, creating `dnam.toml`, `dist/`, `models/`,
    /// `.gitignore`, and provisioning the user's `Ed25519` creator keypair in
    /// `default_user_config_dir()`.
    pub fn init(root_dir: impl AsRef<Path>, name: &str, author: &str) -> StudioResult<Self> {
        let root = root_dir.as_ref().to_path_buf();
        fs::create_dir_all(root.join("dist"))?;
        fs::create_dir_all(root.join("models"))?;

        let gitignore_path = root.join(".gitignore");
        if !gitignore_path.exists() {
            fs::write(&gitignore_path, WORKSPACE_GITIGNORE)?;
        }

        let (_, _, pub_hex) = ensure_creator_ed25519_keypair(None)?;

        let manifest = DnamProjectManifest {
            schema_version: "1.0".to_string(),
            project: ProjectMetadata {
                name: name.to_string(),
                author: author.to_string(),
                sample_rate_hz: 48_000,
                creator_public_key_hex: pub_hex,
                signing_key_path: None,
            },
            capture: CaptureSettings {
                backend: "software".to_string(),
                device_id: "loopback_default".to_string(),
                calibrated_latency_samples: 0,
                noise_floor_dbfs: -96.0,
                send_gain_db: 0.0,
                unity_ref_dbfs: 0.0,
                excitation_profile: None,
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

    /// Returns the resolved path to a model's probe-optimized dry excitation WAV (`models/<model_id>/stems/input.wav`).
    pub fn model_dry_input_wav_path(&self, model_id: &str) -> PathBuf {
        self.root_dir
            .join("models")
            .join(model_id)
            .join("stems")
            .join("input.wav")
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

    /// Updates full 2-step calibration settings including `send_gain_db` and [`TargetExcitationProfile`] and saves `dnam.toml`.
    #[allow(clippy::too_many_arguments)]
    pub fn update_calibration_with_profile(
        &mut self,
        backend: &str,
        device_id: &str,
        send_gain_db: f32,
        latency_samples: i32,
        noise_floor_dbfs: f32,
        unity_ref_dbfs: f32,
        profile: TargetExcitationProfile,
    ) -> StudioResult<()> {
        self.manifest.capture.backend = backend.to_string();
        self.manifest.capture.device_id = device_id.to_string();
        self.manifest.capture.send_gain_db = send_gain_db;
        self.manifest.capture.calibrated_latency_samples = latency_samples;
        self.manifest.capture.noise_floor_dbfs = noise_floor_dbfs;
        self.manifest.capture.unity_ref_dbfs = unity_ref_dbfs;
        self.manifest.capture.excitation_profile = Some(profile);
        self.save()
    }

    /// Atomically persists `dnam.toml`.
    pub fn save(&self) -> StudioResult<()> {
        let encoded =
            toml::to_string_pretty(&self.manifest).map_err(|e| StudioError::Toml(e.to_string()))?;
        atomic_write_text(&self.root_dir.join("dnam.toml"), &encoded)
    }

    /// Lists all registered models in `dnam.toml`.
    pub fn list_models(&self) -> Vec<ModelManifest> {
        let mut out = Vec::new();
        for model_id in &self.manifest.models {
            if let Ok(m) = self.load_model(model_id) {
                out.push(m);
            }
        }
        out
    }

    /// Adds a new model to `models/<model_id>/model.toml`.
    pub fn add_model(
        &mut self,
        model_id: &str,
        title: &str,
        tier: &str,
    ) -> StudioResult<ModelManifest> {
        if model_id.trim().is_empty() || model_id.contains('/') || model_id.contains('\\') {
            return Err(StudioError::Validation(format!(
                "Invalid model_id '{model_id}'"
            )));
        }
        let model_dir = self.root_dir.join("models").join(model_id);
        fs::create_dir_all(model_dir.join("stems"))?;
        fs::create_dir_all(model_dir.join("dist"))?;
        fs::create_dir_all(model_dir.join("reports"))?;
        fs::create_dir_all(model_dir.join("renders"))?;

        let model = ModelManifest {
            model_id: model_id.to_string(),
            title: title.to_string(),
            tier: tier.to_string(),
            description: String::new(),
            output_mode: "stereo".to_string(),
            topology: "auto".to_string(),
            excitation_profile: None,
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

    /// Updates metadata on an existing model (`models/<model_id>/model.toml`).
    pub fn update_model(
        &self,
        model_id: &str,
        title: Option<&str>,
        tier: Option<&str>,
        description: Option<&str>,
        output_mode: Option<&str>,
        topology: Option<&str>,
    ) -> StudioResult<ModelManifest> {
        let mut model = self.load_model(model_id)?;
        if let Some(t) = title {
            model.title = t.to_string();
        }
        if let Some(tr) = tier {
            model.tier = tr.to_string();
        }
        if let Some(d) = description {
            model.description = d.to_string();
        }
        if let Some(o) = output_mode {
            model.output_mode = o.to_string();
        }
        if let Some(top) = topology {
            model.topology = top.to_string();
        }
        self.save_model(&model)?;
        Ok(model)
    }

    /// Removes `model_id` from `dnam.toml` and optionally deletes `models/<model_id>`.
    pub fn remove_model(&mut self, model_id: &str, delete_files: bool) -> StudioResult<()> {
        self.manifest.models.retain(|m| m != model_id);
        self.save()?;
        if delete_files {
            let model_dir = self.root_dir.join("models").join(model_id);
            if model_dir.exists() {
                fs::remove_dir_all(model_dir)?;
            }
        }
        Ok(())
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

    /// Updates an existing range or switch control on `model_id`.
    #[allow(clippy::too_many_arguments)]
    pub fn update_control(
        &self,
        model_id: &str,
        control_id: &str,
        label: Option<&str>,
        min: Option<f32>,
        max: Option<f32>,
        default_val: Option<&str>,
        options: Option<&str>,
    ) -> StudioResult<()> {
        let mut model = self.load_model(model_id)?;
        if let Some(r) = model.ranges.iter_mut().find(|r| r.id == control_id) {
            if let Some(lbl) = label {
                r.label = lbl.to_string();
            }
            if let Some(lo) = min {
                r.min = lo;
            }
            if let Some(hi) = max {
                r.max = hi;
            }
            if let Some(def_s) = default_val {
                r.default_value = def_s.parse().map_err(|_| {
                    StudioError::Validation(format!("Invalid range default '{def_s}'"))
                })?;
            }
            return self.save_model(&model);
        }
        if let Some(s) = model.switches.iter_mut().find(|s| s.id == control_id) {
            if let Some(lbl) = label {
                s.label = lbl.to_string();
            }
            if let Some(opts_csv) = options {
                let labels: Vec<&str> = opts_csv
                    .split(',')
                    .map(str::trim)
                    .filter(|o| !o.is_empty())
                    .collect();
                if labels.len() < 2 {
                    return Err(StudioError::Validation(
                        "Switch control requires at least 2 options".to_string(),
                    ));
                }
                s.options = labels
                    .into_iter()
                    .enumerate()
                    .map(|(index, opt)| ControlSwitchOptionSpec {
                        index,
                        label: opt.to_string(),
                    })
                    .collect();
            }
            if let Some(def_s) = default_val {
                if let Some(pos) = s.options.iter().position(|o| o.label == def_s) {
                    s.default_index = pos;
                } else if let Ok(idx) = def_s.parse::<usize>()
                    && idx < s.options.len()
                {
                    s.default_index = idx;
                } else {
                    return Err(StudioError::Validation(format!(
                        "Switch default '{def_s}' not found"
                    )));
                }
            }
            return self.save_model(&model);
        }
        Err(StudioError::Validation(format!(
            "Control '{control_id}' not found on model '{model_id}'"
        )))
    }

    /// Removes `control_id` from `model_id`.
    pub fn remove_control(&self, model_id: &str, control_id: &str) -> StudioResult<()> {
        let mut model = self.load_model(model_id)?;
        let before = model.ranges.len() + model.switches.len();
        model.ranges.retain(|r| r.id != control_id);
        model.switches.retain(|s| s.id != control_id);
        if model.ranges.len() + model.switches.len() == before {
            return Err(StudioError::Validation(format!(
                "Control '{control_id}' not found on model '{model_id}'"
            )));
        }
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

    /// Updates an existing preset on `model_id`.
    pub fn update_preset(
        &self,
        model_id: &str,
        preset_id: &str,
        label: Option<&str>,
        ranges: Option<BTreeMap<String, f32>>,
        switches: Option<BTreeMap<String, usize>>,
    ) -> StudioResult<()> {
        let mut model = self.load_model(model_id)?;
        let preset = model
            .presets
            .iter_mut()
            .find(|p| p.id == preset_id)
            .ok_or_else(|| {
                StudioError::Validation(format!(
                    "Preset '{preset_id}' not found on model '{model_id}'"
                ))
            })?;
        if let Some(l) = label {
            preset.label = l.to_string();
        }
        if let Some(r) = ranges {
            preset.ranges.extend(r);
        }
        if let Some(s) = switches {
            preset.switches.extend(s);
        }
        self.save_model(&model)
    }

    /// Removes `preset_id` from `model_id`.
    pub fn remove_preset(&self, model_id: &str, preset_id: &str) -> StudioResult<()> {
        let mut model = self.load_model(model_id)?;
        let before = model.presets.len();
        model.presets.retain(|p| p.id != preset_id);
        if model.presets.len() == before {
            return Err(StudioError::Validation(format!(
                "Preset '{preset_id}' not found on model '{model_id}'"
            )));
        }
        self.save_model(&model)
    }

    /// Registers a captured dry/wet stem in `model_id`.
    pub fn register_stem(&self, model_id: &str, stem: StemRecord) -> StudioResult<()> {
        let mut model = self.load_model(model_id)?;
        model.stems.retain(|s| s.stem_id != stem.stem_id);
        model.stems.push(stem);
        self.save_model(&model)
    }

    /// Registers an external wet WAV file as a stem on `model_id`, aligning it against the workspace dry excitation.
    pub fn register_external_stem(
        &self,
        model_id: &str,
        stem_id: &str,
        wet_wav_path: &Path,
        normalized_ranges: BTreeMap<String, f32>,
        switches: BTreeMap<String, usize>,
    ) -> StudioResult<StemRecord> {
        let dry_rel = format!("models/{model_id}/stems/input.wav");
        let dry_path = self.model_dry_input_wav_path(model_id);
        let wet_wav = read_wav(wet_wav_path)?;
        if !dry_path.is_file() {
            let model_prof = self
                .load_model(model_id)
                .ok()
                .and_then(|m| m.excitation_profile);
            let prof_ref =
                model_prof
                    .as_ref()
                    .or(self.manifest.capture.excitation_profile.as_ref());
            let synth = generate_excitation_signal_with_profile(
                self.manifest.project.sample_rate_hz,
                wet_wav.samples.len().max(2_048),
                prof_ref,
            );
            write_wav_24bit(&dry_path, &synth, self.manifest.project.sample_rate_hz, 1)?;
        }
        let dry_wav = read_wav(&dry_path)?;
        let min_len = dry_wav.samples.len().min(wet_wav.samples.len());
        if min_len == 0 {
            return Err(StudioError::Audio(
                "External wet WAV contains 0 samples".to_string(),
            ));
        }
        let locked_lag = if self.manifest.capture.calibrated_latency_samples != 0 {
            Some(self.manifest.capture.calibrated_latency_samples)
        } else {
            None
        };
        let align = align_hardware_latency(
            &dry_wav.samples[..min_len],
            &wet_wav.samples[..min_len],
            64,
            locked_lag,
            self.manifest.project.sample_rate_hz,
            Some(self.manifest.capture.noise_floor_dbfs),
        )?;
        let aligned_wet =
            apply_subsample_shift_fft(&wet_wav.samples[..min_len], align.subsample_lag_samples);
        let wet_rel = format!("models/{model_id}/stems/{stem_id}.wav");
        let wet_dest = self.root_dir.join(&wet_rel);
        write_wav_24bit(
            &wet_dest,
            &aligned_wet,
            wet_wav.sample_rate_hz,
            wet_wav.channels,
        )?;
        let rec = StemRecord {
            stem_id: stem_id.to_string(),
            dry_wav_path: dry_rel,
            wet_wav_path: wet_rel,
            normalized_ranges,
            switches,
            pcm_sha256: compute_pcm_sha256(&aligned_wet),
            dry_pcm_sha256: compute_pcm_sha256(&dry_wav.samples[..min_len]),
            aligned_lag_samples: align.subsample_lag_samples,
            snr_db: align.snr_db,
            noise_floor_dbfs: align.noise_floor_dbfs,
        };
        self.register_stem(model_id, rec.clone())?;
        Ok(rec)
    }

    /// Updates control coordinates on an existing stem.
    pub fn update_stem(
        &self,
        model_id: &str,
        stem_id: &str,
        normalized_ranges: BTreeMap<String, f32>,
        switches: BTreeMap<String, usize>,
    ) -> StudioResult<StemRecord> {
        let mut model = self.load_model(model_id)?;
        let stem = model
            .stems
            .iter_mut()
            .find(|s| s.stem_id == stem_id)
            .ok_or_else(|| {
                StudioError::Validation(format!("Stem '{stem_id}' not found on model '{model_id}'"))
            })?;
        stem.normalized_ranges.extend(normalized_ranges);
        stem.switches.extend(switches);
        let updated = stem.clone();
        self.save_model(&model)?;
        Ok(updated)
    }

    /// Removes `stem_id` from `model_id` and optionally deletes the wet WAV file.
    pub fn remove_stem(
        &self,
        model_id: &str,
        stem_id: &str,
        delete_files: bool,
    ) -> StudioResult<()> {
        let mut model = self.load_model(model_id)?;
        let pos = model
            .stems
            .iter()
            .position(|s| s.stem_id == stem_id)
            .ok_or_else(|| {
                StudioError::Validation(format!("Stem '{stem_id}' not found on model '{model_id}'"))
            })?;
        let removed = model.stems.remove(pos);
        self.save_model(&model)?;
        if delete_files {
            let p = self.root_dir.join(&removed.wet_wav_path);
            if p.is_file() {
                let _ = fs::remove_file(p);
            }
        }
        Ok(())
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

fn decode_hex_32(hex_str: &str) -> Option<[u8; 32]> {
    if hex_str.len() != 64 {
        return None;
    }
    let mut out = [0_u8; 32];
    let (chunks, _) = hex_str.as_bytes().as_chunks::<2>();
    for (idx, chunk) in chunks.iter().enumerate() {
        let s = std::str::from_utf8(chunk).ok()?;
        out[idx] = u8::from_str_radix(s, 16).ok()?;
    }
    Some(out)
}
