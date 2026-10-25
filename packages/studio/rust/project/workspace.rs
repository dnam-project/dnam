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
        let raw_bytes = fs::read(&resolved)?;
        if raw_bytes.len() == 32 {
            let mut arr = [0_u8; 32];
            arr.copy_from_slice(&raw_bytes);
            arr
        } else {
            let raw_text = String::from_utf8(raw_bytes).map_err(|_| {
                StudioError::Validation(format!(
                    "Invalid Ed25519 secret seed in {}: expected 64 hex chars or 32 raw bytes",
                    resolved.display()
                ))
            })?;
            decode_hex_32(raw_text.trim()).ok_or_else(|| {
                StudioError::Validation(format!(
                    "Invalid Ed25519 secret seed in {}: expected 64 hex chars (32 bytes)",
                    resolved.display()
                ))
            })?
        }
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
        Self::init_inner(root_dir.as_ref(), name, author)
    }

    fn init_inner(root_dir: &Path, name: &str, author: &str) -> StudioResult<Self> {
        let root = root_dir.to_path_buf();
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
        Self::open_inner(root_dir.as_ref())
    }

    fn open_inner(root_dir: &Path) -> StudioResult<Self> {
        let root = root_dir.to_path_buf();
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

    /// Updates capture calibration settings on `model_id` and saves `models/<model_id>/model.toml`.
    pub fn update_calibration(
        &self,
        model_id: &str,
        backend: &str,
        device_id: &str,
        latency_samples: i32,
        noise_floor_dbfs: f32,
    ) -> StudioResult<()> {
        let mut model = self.load_model(model_id)?;
        model.capture.backend = backend.to_string();
        model.capture.device_id = device_id.to_string();
        model.capture.calibrated_latency_samples = latency_samples;
        model.capture.noise_floor_dbfs = noise_floor_dbfs;
        self.save_model(&model)
    }

    /// Updates full 2-step calibration settings on `model_id` including `send_gain_db` and [`TargetExcitationProfile`] and saves `models/<model_id>/model.toml`.
    #[allow(clippy::too_many_arguments)]
    pub fn update_calibration_with_profile(
        &self,
        model_id: &str,
        backend: &str,
        device_id: &str,
        send_gain_db: f32,
        latency_samples: i32,
        noise_floor_dbfs: f32,
        unity_ref_dbfs: f32,
        profile: TargetExcitationProfile,
    ) -> StudioResult<()> {
        let mut model = self.load_model(model_id)?;
        model.capture.backend = backend.to_string();
        model.capture.device_id = device_id.to_string();
        model.capture.send_gain_db = send_gain_db;
        model.capture.calibrated_latency_samples = latency_samples;
        model.capture.noise_floor_dbfs = noise_floor_dbfs;
        model.capture.unity_ref_dbfs = unity_ref_dbfs;
        model.capture.excitation_profile = Some(profile);
        self.save_model(&model)
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
            capture: CaptureSettings::default(),
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
        let model = self.load_model(model_id)?;
        let dry_rel = format!("models/{model_id}/stems/input.wav");
        let dry_path = self.model_dry_input_wav_path(model_id);
        let wet_wav = read_wav(wet_wav_path)?;
        if !dry_path.is_file() {
            let synth = generate_excitation_signal_with_profile(
                self.manifest.project.sample_rate_hz,
                wet_wav.samples.len().max(2_048),
                model.capture.excitation_profile.as_ref(),
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
        let locked_lag = if model.capture.calibrated_latency_samples != 0 {
            Some(model.capture.calibrated_latency_samples)
        } else {
            None
        };
        let align = align_hardware_latency(
            &dry_wav.samples[..min_len],
            &wet_wav.samples[..min_len],
            64,
            locked_lag,
            self.manifest.project.sample_rate_hz,
            Some(model.capture.noise_floor_dbfs),
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

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use proptest::prelude::*;

    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_compact_range_and_switch_parsers_roundtrip(
            min_v in -24.0_f32..24.0_f32,
            span in 0.5_f32..24.0_f32,
            frac in 0.0_f32..=1.0_f32,
            n_opts in 2_usize..=5,
            def_idx_raw in 0_usize..10,
        ) {
            let max_v = min_v + span;
            let def_v = min_v + frac * span;
            let spec_eq = format!("knob_a={min_v:.3}..{max_v:.3}:{def_v:.3}");
            let parsed_eq = parse_compact_range_spec(&spec_eq).expect("parse =/: range");
            prop_assert_eq!(parsed_eq.id, "knob_a");
            prop_assert!(parsed_eq.min < parsed_eq.max);
            prop_assert!(parsed_eq.default_value >= parsed_eq.min - 1e-3);
            prop_assert!(parsed_eq.default_value <= parsed_eq.max + 1e-3);

            let spec_at = format!("knob_a:Knob A:{min_v:.3}..{max_v:.3}@{def_v:.3}");
            let parsed_at = parse_compact_range_spec(&spec_at).expect("parse :/@ range");
            prop_assert_eq!(parsed_at.id, "knob_a");
            prop_assert_eq!(parsed_at.label, "Knob A");

            let opts: Vec<String> = (0..n_opts).map(|i| format!("opt_{i}")).collect();
            let def_idx = def_idx_raw % n_opts;
            let sw_spec = format!("mode={}:{}", opts.join(","), opts[def_idx]);
            let parsed_sw = parse_compact_switch_spec(&sw_spec).expect("parse switch");
            prop_assert_eq!(parsed_sw.options.len(), n_opts);
            prop_assert_eq!(parsed_sw.default_index, def_idx);
        }
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = env::temp_dir().join(format!("dnam_ws_unit_{tag}_{nanos}"));
        let _ = fs::create_dir_all(&dir);
        dir
    }

    #[test]
    fn test_ed25519_keypair_and_compact_parsers() {
        let dir = temp_dir("keys_parsers");
        assert!(default_user_config_dir().ends_with("dnam"));

        let key_path = dir.join("nested_keys").join("creator.key");
        let (p1, seed1, pub1) =
            ensure_creator_ed25519_keypair(Some(&key_path)).expect("generate key");
        assert_eq!(p1, key_path);
        assert_eq!(pub1.len(), 64);

        let (_, seed2, pub2) =
            ensure_creator_ed25519_keypair(Some(&key_path)).expect("reuse existing key");
        assert_eq!(seed1, seed2);
        assert_eq!(pub1, pub2);

        // Raw 32-byte binary key file is also supported
        let raw_key_path = dir.join("raw32.key");
        fs::write(&raw_key_path, seed1).expect("write raw 32B key");
        let (_, seed3, pub3) =
            ensure_creator_ed25519_keypair(Some(&raw_key_path)).expect("load raw 32B key");
        assert_eq!(seed3, seed1);
        assert_eq!(pub3, pub1);

        // Invalid key length raises Validation error (both UTF-8 hex and non-UTF-8 binary)
        let bad_key = dir.join("bad.key");
        fs::write(&bad_key, "001122\n").expect("write bad key");
        assert!(ensure_creator_ed25519_keypair(Some(&bad_key)).is_err());
        let bad_bin_key = dir.join("bad_bin.key");
        fs::write(&bad_bin_key, [0xff_u8, 0xfe, 0xfd]).expect("write bad bin key");
        assert!(ensure_creator_ed25519_keypair(Some(&bad_bin_key)).is_err());

        // Compact range parser (=/: and :/@ formats + error branches)
        let r1 = parse_compact_range_spec("drive_gain=0..10:6.5").expect("r1");
        assert_eq!(r1.id, "drive_gain");
        assert_eq!(r1.label, "Drive Gain");
        assert!((r1.default_value - 6.5).abs() < 1e-6);

        let r2 = parse_compact_range_spec("blend=-12..12").expect("r2");
        assert!((r2.default_value - 0.0).abs() < 1e-6);

        assert!(parse_compact_range_spec("=0..10").is_err());
        assert!(parse_compact_range_spec("drive=10").is_err());
        assert!(parse_compact_range_spec("drive=10..0").is_err());
        assert!(parse_compact_range_spec("drive=0..10:15").is_err());
        assert!(parse_compact_range_spec("drive=0..10:not_a_num").is_err());
        assert!(parse_compact_range_spec("drive").is_err());
        assert!(parse_compact_range_spec(":Label:0..10@5").is_err());
        assert!(parse_compact_range_spec("drive:Label:0..10").is_err());
        assert!(parse_compact_range_spec("drive:Label:0@5").is_err());
        assert!(parse_compact_range_spec("drive:Drive:10..0@5").is_err());
        assert!(parse_compact_range_spec("drive:Drive:0..10@15").is_err());

        // Compact switch parser (=/: and :/@ formats + error branches)
        let s1 = parse_compact_switch_spec("attack=cut,flat,boost:boost").expect("s1");
        assert_eq!(s1.id, "attack");
        assert_eq!(s1.default_index, 2);

        let s2 = parse_compact_switch_spec("voice=clean,lead:1").expect("s2");
        assert_eq!(s2.default_index, 1);

        let s3 = parse_compact_switch_spec("voice=clean,lead").expect("s3");
        assert_eq!(s3.default_index, 0);

        assert!(parse_compact_switch_spec("=a,b").is_err());
        assert!(parse_compact_switch_spec("attack=only_one").is_err());
        assert!(parse_compact_switch_spec("attack=cut,boost:invalid").is_err());
        assert!(parse_compact_switch_spec("voice:OnlyTwoParts").is_err());
        assert!(parse_compact_switch_spec(":Label:a,b@0").is_err());
        assert!(parse_compact_switch_spec("voice:Label:a,b").is_err());
        assert!(parse_compact_switch_spec("voice:Voice:only_one@0").is_err());
        assert!(parse_compact_switch_spec("voice:Voice:a,b@5").is_err());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_dnam_project_crud_and_stem_lifecycle() {
        let dir = temp_dir("crud");
        let mut proj = DnamProject::init(&dir, "Test Pedal Pack", "peter").expect("init");
        assert_eq!(proj.root_dir(), dir.as_path());

        let reloaded = DnamProject::open(&dir).expect("open");
        assert_eq!(reloaded.manifest().project.name, "Test Pedal Pack");

        assert!(proj.add_model("", "Bad", "Full").is_err());
        let _ = proj
            .add_model("b7k_drive", "B7K Drive", "Full")
            .expect("add model");
        let _ = proj
            .add_model("alpha_omega", "Alpha Omega", "Lite")
            .expect("add second model");
        assert_eq!(proj.list_models().len(), 2);

        let up_m = proj
            .update_model(
                "alpha_omega",
                Some("Alpha Omega Ultra"),
                Some("Ultra"),
                Some("Bass distortion"),
                Some("dual_mono"),
                Some("core1_only"),
            )
            .expect("update model");
        assert_eq!(up_m.description, "Bass distortion");
        assert_eq!(up_m.output_mode, "dual_mono");
        assert_eq!(up_m.topology, "core1_only");

        proj.remove_model("alpha_omega", true)
            .expect("remove model");
        assert!(proj.load_model("alpha_omega").is_err());

        proj.update_calibration("b7k_drive", "usb", "cpal_0", 32, -92.0)
            .expect("update cal");
        assert_eq!(
            proj.load_model("b7k_drive")
                .expect("load")
                .capture
                .calibrated_latency_samples,
            32
        );

        proj.add_range_control(
            "b7k_drive",
            parse_compact_range_spec("drive=0..10:5").expect("drive"),
        )
        .expect("add range");
        proj.add_switch_control(
            "b7k_drive",
            parse_compact_switch_spec("attack=cut,flat,boost:flat").expect("attack"),
        )
        .expect("add switch");

        proj.update_control(
            "b7k_drive",
            "drive",
            Some("Drive Gain"),
            Some(0.0),
            Some(12.0),
            Some("6.0"),
            None,
        )
        .expect("update range");
        proj.update_control(
            "b7k_drive",
            "attack",
            Some("Attack Switch"),
            None,
            None,
            Some("boost"),
            Some("cut,flat,boost"),
        )
        .expect("update switch by label");
        proj.update_control("b7k_drive", "attack", None, None, None, Some("0"), None)
            .expect("update switch by index");
        assert!(
            proj.update_control(
                "b7k_drive",
                "attack",
                None,
                None,
                None,
                None,
                Some("only_one")
            )
            .is_err()
        );
        assert!(
            proj.update_control(
                "b7k_drive",
                "attack",
                None,
                None,
                None,
                Some("nonexistent"),
                None
            )
            .is_err()
        );
        assert!(
            proj.update_control("b7k_drive", "missing", Some("X"), None, None, None, None)
                .is_err()
        );
        assert!(proj.remove_control("b7k_drive", "missing").is_err());
        assert!(proj.remove_preset("b7k_drive", "missing").is_err());
        assert!(
            proj.update_preset("b7k_drive", "missing", Some("X"), None, None)
                .is_err()
        );

        // External stem registration, update, and removal
        let ext_wet = dir.join("ext_wet.wav");
        let sig: Vec<f32> = (0..2048).map(|i| ((i as f32) * 0.05).sin() * 0.3).collect();
        write_wav_24bit(&ext_wet, &sig, 48_000, 1).expect("write ext wet");

        let mut stem_ranges = BTreeMap::new();
        stem_ranges.insert("drive".to_string(), 0.5);
        let rec = proj
            .register_external_stem(
                "b7k_drive",
                "s_00",
                &ext_wet,
                stem_ranges.clone(),
                BTreeMap::new(),
            )
            .expect("register external stem");
        assert_eq!(rec.stem_id, "s_00");
        assert!(proj.model_dry_input_wav_path("b7k_drive").is_file());

        stem_ranges.insert("drive".to_string(), 0.8);
        let up_s = proj
            .update_stem("b7k_drive", "s_00", stem_ranges, BTreeMap::new())
            .expect("update stem");
        assert!((up_s.normalized_ranges["drive"] - 0.8).abs() < 1e-6);
        assert!(
            proj.update_stem("b7k_drive", "missing", BTreeMap::new(), BTreeMap::new())
                .is_err()
        );

        proj.remove_stem("b7k_drive", "s_00", true)
            .expect("remove stem");
        assert!(proj.remove_stem("b7k_drive", "s_00", false).is_err());

        let _ = fs::remove_dir_all(&dir);
    }
}
