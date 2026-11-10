//! Workspace CRUD, atomic TOML persistence, OS-level `Ed25519` creator keypair provisioning
//! (`default_user_config_dir` / `ensure_creator_ed25519_keypair`), and compact CLI parsers.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static ATOMIC_TMP_SEQ: AtomicU64 = AtomicU64::new(0);

use ed25519_dalek::SigningKey;

use crate::error::{StudioError, StudioResult};
use crate::project::manifest::{
    CaptureSettings, ControlRangeSpec, ControlSwitchOptionSpec, ControlSwitchSpec,
    DnamProjectManifest, ModelDistributionPolicy, ModelManifest, ModelProtectionProfile,
    PresetSpec, ProjectMetadata, StemRecord,
};

const WORKSPACE_GITIGNORE: &str = "\
# DNAM Project Ignore Rules
*.dnam-ckpt
models/*/stems/input.wav
models/*/renders/
models/*/reports/*.jsonl
models/*/reports/*.html
models/*/.dnam-cache/
.dnam-cache/
.cache/
";

fn ensure_gitignore_has_dnam_cache(root: &Path) {
    if ensure_no_symlink_components_under(root, Path::new(".gitignore")).is_err() {
        return;
    }
    let gitignore_path = root.join(".gitignore");
    if !gitignore_path.is_file() {
        return;
    }
    if let Ok(mut text) = fs::read_to_string(&gitignore_path) {
        let mut changed = false;
        for rule in ["models/*/.dnam-cache/", ".dnam-cache/"] {
            if !text.lines().any(|line| line.trim() == rule) {
                if !text.ends_with('\n') && !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(rule);
                text.push('\n');
                changed = true;
            }
        }
        if changed {
            let _ = atomic_write_text(&gitignore_path, &text);
        }
    }
}

fn resolve_os_config_dir_from_vars(
    home: PathBuf,
    appdata: Option<&str>,
    xdg: Option<&str>,
    target_os: &str,
) -> PathBuf {
    match target_os {
        "macos" => home
            .join("Library")
            .join("Application Support")
            .join("dnam"),
        "windows" => match appdata.map(str::trim).filter(|s| !s.is_empty()) {
            Some(dir) => PathBuf::from(dir).join("dnam"),
            None => home.join("AppData").join("Roaming").join("dnam"),
        },
        _ => match xdg.map(str::trim).filter(|s| !s.is_empty()) {
            Some(dir) => PathBuf::from(dir).join("dnam"),
            None => home.join(".config").join("dnam"),
        },
    }
}

fn try_default_user_config_dir() -> StudioResult<PathBuf> {
    if let Ok(env_dir) = env::var("DNAM_CONFIG_DIR") {
        let trimmed = env_dir.trim();
        if !trimmed.is_empty() {
            return Ok(PathBuf::from(trimmed));
        }
    }

    #[cfg(test)]
    {
        Ok(env::temp_dir()
            .join(format!("dnam_studio_unit_cfg_{}", std::process::id()))
            .join("dnam"))
    }

    #[cfg(not(test))]
    {
        let appdata = env::var("APPDATA").ok().filter(|s| !s.trim().is_empty());
        let xdg = env::var("XDG_CONFIG_HOME")
            .ok()
            .filter(|s| !s.trim().is_empty());
        let home = env::var("HOME")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .or_else(|| {
                env::var("USERPROFILE")
                    .ok()
                    .filter(|s| !s.trim().is_empty())
            });
        let os = std::env::consts::OS;
        if os == "windows"
            && let Some(ad) = appdata.as_deref()
        {
            return Ok(PathBuf::from(ad.trim()).join("dnam"));
        }
        if os != "macos"
            && os != "windows"
            && let Some(x) = xdg.as_deref()
        {
            return Ok(PathBuf::from(x.trim()).join("dnam"));
        }
        let Some(home_str) = home else {
            return Err(StudioError::Validation(
                "Cannot resolve user config directory: set HOME, USERPROFILE, XDG_CONFIG_HOME, APPDATA, or DNAM_CONFIG_DIR".to_string(),
            ));
        };
        Ok(resolve_os_config_dir_from_vars(
            PathBuf::from(home_str.trim()),
            appdata.as_deref(),
            xdg.as_deref(),
            os,
        ))
    }
}

/// Returns the OS-appropriate user configuration directory for `dnam`:
/// - Respects `DNAM_CONFIG_DIR` environment variable when set.
/// - macOS: `~/Library/Application Support/dnam`
/// - Windows: `%APPDATA%\dnam` (or `~/AppData/Roaming/dnam`)
/// - Linux / POSIX: `${XDG_CONFIG_HOME:-~/.config}/dnam`
pub fn default_user_config_dir() -> PathBuf {
    try_default_user_config_dir().unwrap_or_else(|_| PathBuf::from(".dnam-config"))
}

#[cfg(unix)]
const UNIX_O_NOFOLLOW: i32 = {
    #[cfg(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "netbsd",
        target_os = "openbsd"
    ))]
    {
        0x0100
    }
    #[cfg(all(
        target_os = "linux",
        any(target_arch = "arm", target_arch = "aarch64", target_arch = "m68k")
    ))]
    {
        0o100000
    }
    #[cfg(all(
        target_os = "linux",
        not(any(target_arch = "arm", target_arch = "aarch64", target_arch = "m68k"))
    ))]
    {
        0o400000
    }
};

fn resolve_dir_against_cwd(dir: &Path) -> StudioResult<PathBuf> {
    use std::path::Component;
    for comp in dir.components() {
        if matches!(comp, Component::ParentDir) {
            return Err(StudioError::Validation(format!(
                "Refusing parent-directory traversal in private path: {}",
                dir.display()
            )));
        }
    }
    if dir.is_absolute() {
        Ok(dir.to_path_buf())
    } else {
        let cwd = env::current_dir()?;
        if dir.as_os_str().is_empty() {
            Ok(cwd)
        } else {
            Ok(cwd.join(dir))
        }
    }
}

fn verify_dir_ancestors_no_symlinks(dir: &Path, require_exists: bool) -> StudioResult<()> {
    use std::path::Component;
    let target = resolve_dir_against_cwd(dir)?;
    let mut acc = PathBuf::new();
    let mut normal_depth = 0usize;
    for comp in target.components() {
        acc.push(comp);
        if matches!(
            comp,
            Component::Prefix(_) | Component::RootDir | Component::CurDir
        ) {
            continue;
        }
        if matches!(comp, Component::ParentDir) {
            return Err(StudioError::Validation(format!(
                "Refusing parent-directory traversal in private path: {}",
                target.display()
            )));
        }
        normal_depth += 1;
        if target.is_absolute()
            && normal_depth == 1
            && matches!(comp.as_os_str().to_str(), Some("tmp" | "var" | "etc"))
        {
            continue;
        }
        match fs::symlink_metadata(&acc) {
            Ok(meta) if meta.file_type().is_symlink() || !meta.is_dir() => {
                return Err(StudioError::Validation(format!(
                    "Refusing to traverse symlink or non-directory ancestor: {}",
                    acc.display()
                )));
            }
            Ok(meta) => {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt as _;
                    let mode = meta.permissions().mode();
                    if (mode & 0o002) != 0 && (mode & 0o1000) == 0 {
                        return Err(StudioError::Validation(format!(
                            "Refusing world-writable non-sticky ancestor directory: {}",
                            acc.display()
                        )));
                    }
                }
            }
            Err(err) if !require_exists && err.kind() == std::io::ErrorKind::NotFound => break,
            Err(err) => return Err(err.into()),
        }
    }
    Ok(())
}

fn ensure_private_dir_no_symlink_ancestors(dir: &Path) -> StudioResult<()> {
    let target = resolve_dir_against_cwd(dir)?;
    verify_dir_ancestors_no_symlinks(&target, false)?;
    fs::create_dir_all(&target)?;
    verify_dir_ancestors_no_symlinks(&target, true)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let is_implicit_cwd = dir
            .components()
            .all(|c| matches!(c, std::path::Component::CurDir));
        if is_implicit_cwd {
            let meta = fs::symlink_metadata(&target)?;
            if meta.file_type().is_symlink()
                || !meta.is_dir()
                || (meta.permissions().mode() & 0o022) != 0
            {
                return Err(StudioError::Validation(format!(
                    "Refusing group/world-writable working directory for private file: {}",
                    target.display()
                )));
            }
        } else {
            fs::set_permissions(&target, fs::Permissions::from_mode(0o700))?;
            let meta = fs::symlink_metadata(&target)?;
            if meta.file_type().is_symlink()
                || !meta.is_dir()
                || (meta.permissions().mode() & 0o077) != 0
            {
                return Err(StudioError::Validation(format!(
                    "Private directory must be owned by current user with 0700 permissions: {}",
                    target.display()
                )));
            }
        }
    }
    Ok(())
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
        try_default_user_config_dir()?
            .join("keys")
            .join("creator_ed25519.key")
    };

    if let Some(parent) = resolved.parent() {
        verify_dir_ancestors_no_symlinks(parent, false)?;
    }

    let exists_regular = match fs::symlink_metadata(&resolved) {
        Ok(meta) if meta.file_type().is_symlink() => {
            return Err(StudioError::Validation(format!(
                "Refusing to follow symlink for Ed25519 secret key: {}",
                resolved.display()
            )));
        }
        Ok(meta) if meta.is_file() => true,
        Ok(_) => {
            return Err(StudioError::Validation(format!(
                "Ed25519 secret key path is not a regular file: {}",
                resolved.display()
            )));
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => false,
        Err(err) => return Err(err.into()),
    };

    if exists_regular {
        if let Some(parent) = resolved.parent() {
            verify_dir_ancestors_no_symlinks(parent, true)?;
        }
    } else {
        if let Some(parent) = resolved.parent() {
            ensure_private_dir_no_symlink_ancestors(parent)?;
        }
        let generated = generate_csprng_seed_32()?;
        let hex_line = format!("{}\n", encode_hex(&generated));
        let seq = ATOMIC_TMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let tmp_path = resolved.with_extension(format!("key.tmp.{}_{}", std::process::id(), seq));
        {
            use std::io::Write as _;
            let mut opts = fs::OpenOptions::new();
            opts.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt as _;
                opts.mode(0o600);
            }
            let mut file = opts.open(&tmp_path)?;
            if let Err(err) = file.write_all(hex_line.as_bytes()) {
                let _ = fs::remove_file(&tmp_path);
                return Err(err.into());
            }
        }
        match fs::hard_link(&tmp_path, &resolved) {
            Ok(()) => {
                let _ = fs::remove_file(&tmp_path);
            }
            Err(_) if fs::symlink_metadata(&resolved).is_err() => {
                if let Err(err) = fs::rename(&tmp_path, &resolved) {
                    let _ = fs::remove_file(&tmp_path);
                    return Err(err.into());
                }
            }
            Err(_) => {
                let _ = fs::remove_file(&tmp_path);
            }
        }
    }

    let raw_bytes = {
        use std::io::Read as _;
        let mut opts = fs::OpenOptions::new();
        opts.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            opts.custom_flags(UNIX_O_NOFOLLOW);
        }
        let file = opts.open(&resolved)?;
        let fmeta = file.metadata()?;
        if !fmeta.is_file() {
            return Err(StudioError::Validation(format!(
                "Refusing to read non-regular Ed25519 secret key: {}",
                resolved.display()
            )));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            if (fmeta.permissions().mode() & 0o077) != 0 {
                return Err(StudioError::Validation(format!(
                    "Refusing to load Ed25519 secret key with group/world permissions (expected 0600): {}",
                    resolved.display()
                )));
            }
            if let Err(err) = file.set_permissions(fmeta.permissions())
                && err.kind() == std::io::ErrorKind::PermissionDenied
            {
                return Err(StudioError::Validation(format!(
                    "Refusing to load Ed25519 secret key not owned by current user: {}",
                    resolved.display()
                )));
            }
        }
        let mut buf = Vec::with_capacity(128);
        file.take(4096).read_to_end(&mut buf)?;
        buf
    };
    let seed_bytes: [u8; 32] = if raw_bytes.len() == 32 {
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
    };

    let signing_key = SigningKey::from_bytes(&seed_bytes);
    let pub_hex = encode_hex(signing_key.verifying_key().as_bytes());
    Ok((resolved, seed_bytes, pub_hex))
}

fn generate_csprng_seed_32() -> StudioResult<[u8; 32]> {
    let mut buf = [0_u8; 32];
    getrandom::fill(&mut buf)
        .map_err(|err| std::io::Error::other(format!("OS CSPRNG failure: {err}")))?;
    Ok(buf)
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
        let default_value: f32 = match def_opt {
            Some(d) => d.parse().map_err(|_| {
                StudioError::Validation(format!("Invalid default value in '{spec}'"))
            })?,
            None => 0.5 * (min + max),
        };
        validate_range_bounds(min, max, default_value, spec)?;
        return Ok(ControlRangeSpec {
            id: id.to_string(),
            label: humanize_id(id),
            min,
            max,
            default_value,
            unit: None,
            calibration: None,
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
    validate_range_bounds(min, max, default_value, spec)?;
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
        calibration: None,
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
            calibration: None,
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
        calibration: None,
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
        Self::init_inner(root_dir.as_ref(), name, author, None)
    }

    /// Initializes a new DNAM workspace at `root_dir`, provisioning the creator `Ed25519` keypair
    /// inside an explicit `config_dir` (`<config_dir>/keys/creator_ed25519.key`).
    pub fn init_with_config_dir(
        root_dir: impl AsRef<Path>,
        name: &str,
        author: &str,
        config_dir: impl AsRef<Path>,
    ) -> StudioResult<Self> {
        Self::init_inner(root_dir.as_ref(), name, author, Some(config_dir.as_ref()))
    }

    fn init_inner(
        root_dir: &Path,
        name: &str,
        author: &str,
        config_dir: Option<&Path>,
    ) -> StudioResult<Self> {
        let root = root_dir.to_path_buf();
        for rel in ["dist", "models", ".gitignore", "dnam.toml"] {
            ensure_no_symlink_components_under(&root, Path::new(rel))?;
        }
        if fs::symlink_metadata(root.join("dnam.toml")).is_ok() {
            return Err(StudioError::Validation(format!(
                "DNAM workspace already initialized at {}",
                root.display()
            )));
        }
        fs::create_dir_all(root.join("dist"))?;
        fs::create_dir_all(root.join("models"))?;

        let gitignore_path = root.join(".gitignore");
        if !gitignore_path.exists() {
            atomic_write_text(&gitignore_path, WORKSPACE_GITIGNORE)?;
        }

        let explicit_key = config_dir.map(|cfg| cfg.join("keys").join("creator_ed25519.key"));
        let (_, _, pub_hex) = ensure_creator_ed25519_keypair(explicit_key.as_deref())?;

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
        ensure_no_symlink_components_under(&root, Path::new("dnam.toml"))?;
        let manifest_path = root.join("dnam.toml");
        let content = fs::read_to_string(&manifest_path)?;
        let manifest: DnamProjectManifest =
            toml::from_str(&content).map_err(|e| StudioError::Toml(e.to_string()))?;
        ensure_gitignore_has_dnam_cache(&root);
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

    /// Atomically persists `dnam.toml`.
    pub fn save(&self) -> StudioResult<()> {
        ensure_no_symlink_components_under(&self.root_dir, Path::new("dnam.toml"))?;
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
        validate_model_id(model_id)?;
        let rel_model_dir = Path::new("models").join(model_id);
        ensure_no_symlink_components_under(&self.root_dir, &rel_model_dir.join("model.toml"))?;
        if self.manifest.models.iter().any(|m| m == model_id)
            || fs::symlink_metadata(self.root_dir.join(&rel_model_dir).join("model.toml")).is_ok()
        {
            return Err(StudioError::Validation(format!(
                "Model '{model_id}' already exists in workspace"
            )));
        }
        for sub in ["stems", "dist", "reports", "renders"] {
            ensure_no_symlink_components_under(&self.root_dir, &rel_model_dir.join(sub))?;
        }
        let model_dir = self.root_dir.join(&rel_model_dir);
        for sub in ["stems", "dist", "reports", "renders"] {
            fs::create_dir_all(model_dir.join(sub))?;
        }

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
            distribution_policy: None,
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

    /// Updates `distribution_policy` (`allowed_protection_profiles`, `allowed_oem_vendors`) on `model_id`.
    pub fn update_model_distribution_policy(
        &self,
        model_id: &str,
        allowed_profiles: &[String],
        allowed_oems: &[String],
    ) -> StudioResult<ModelManifest> {
        let mut model = self.load_model(model_id)?;
        let mut policy = model.distribution_policy.take().unwrap_or_default();
        if !allowed_profiles.is_empty() {
            let mut parsed = Vec::with_capacity(allowed_profiles.len());
            for p_str in allowed_profiles {
                let p = match p_str.trim().to_ascii_lowercase().as_str() {
                    "open" => ModelProtectionProfile::Open,
                    "device_allowlisted" => ModelProtectionProfile::DeviceAllowlisted,
                    "enclave_sealed" => ModelProtectionProfile::EnclaveSealed,
                    other => {
                        return Err(StudioError::Validation(format!(
                            "Unknown protection profile '{other}'; expected 'open', 'device_allowlisted', or 'enclave_sealed'"
                        )));
                    }
                };
                parsed.push(p);
            }
            policy.allowed_protection_profiles = parsed;
        }
        if !allowed_oems.is_empty() {
            policy.allowed_oem_vendors = allowed_oems
                .iter()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }
        model.distribution_policy = Some(ModelDistributionPolicy {
            allowed_protection_profiles: policy.allowed_protection_profiles,
            allowed_oem_vendors: policy.allowed_oem_vendors,
        });
        self.save_model(&model)?;
        Ok(model)
    }

    /// Removes `model_id` from `dnam.toml` and optionally deletes `models/<model_id>`.
    pub fn remove_model(&mut self, model_id: &str, delete_files: bool) -> StudioResult<()> {
        validate_model_id(model_id)?;
        if !self.manifest.models.iter().any(|m| m == model_id) {
            return Err(StudioError::Validation(format!(
                "Model '{model_id}' not found in workspace"
            )));
        }
        let rel_dir = Path::new("models").join(model_id);
        if delete_files {
            ensure_no_symlink_components_under(&self.root_dir, &rel_dir)?;
        }
        self.manifest.models.retain(|m| m != model_id);
        self.save()?;
        if delete_files {
            let model_dir = self.root_dir.join(&rel_dir);
            if model_dir.is_dir() {
                fs::remove_dir_all(model_dir)?;
            }
        }
        Ok(())
    }

    /// Loads `models/<model_id>/model.toml`.
    pub fn load_model(&self, model_id: &str) -> StudioResult<ModelManifest> {
        validate_model_id(model_id)?;
        let rel = Path::new("models").join(model_id).join("model.toml");
        ensure_no_symlink_components_under(&self.root_dir, &rel)?;
        let path = self.root_dir.join(&rel);
        let content = fs::read_to_string(&path)?;
        let parsed: ModelManifest =
            toml::from_str(&content).map_err(|e| StudioError::Toml(e.to_string()))?;
        if parsed.model_id != model_id {
            return Err(StudioError::Validation(format!(
                "Model manifest '{}' declares mismatched model_id '{}'",
                rel.display(),
                parsed.model_id
            )));
        }
        Ok(parsed)
    }

    /// Atomically writes `models/<model_id>/model.toml`.
    pub fn save_model(&self, model: &ModelManifest) -> StudioResult<()> {
        validate_model_id(&model.model_id)?;
        let rel_dir = Path::new("models").join(&model.model_id);
        let rel_file = rel_dir.join("model.toml");
        ensure_no_symlink_components_under(&self.root_dir, &rel_file)?;
        let model_dir = self.root_dir.join(&rel_dir);
        fs::create_dir_all(&model_dir)?;
        let encoded =
            toml::to_string_pretty(model).map_err(|e| StudioError::Toml(e.to_string()))?;
        atomic_write_text(&self.root_dir.join(&rel_file), &encoded)
    }

    /// Adds a continuous range control to `model_id`.
    pub fn add_range_control(&self, model_id: &str, spec: ControlRangeSpec) -> StudioResult<()> {
        let mut model = self.load_model(model_id)?;
        model.ranges.retain(|r| r.id != spec.id);
        model.switches.retain(|s| s.id != spec.id);
        model.ranges.push(spec);
        self.save_model(&model)
    }

    /// Adds a discrete switch control to `model_id`.
    pub fn add_switch_control(&self, model_id: &str, spec: ControlSwitchSpec) -> StudioResult<()> {
        let mut model = self.load_model(model_id)?;
        model.ranges.retain(|r| r.id != spec.id);
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
            validate_range_bounds(r.min, r.max, r.default_value, control_id)?;
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
            if s.default_index >= s.options.len() {
                return Err(StudioError::Validation(format!(
                    "Switch default index {} is out of bounds for {} options on '{control_id}'",
                    s.default_index,
                    s.options.len()
                )));
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
        if preset_id.is_empty()
            || preset_id != preset_id.trim()
            || preset_id.chars().any(char::is_control)
        {
            return Err(StudioError::Validation(format!(
                "Invalid preset_id '{preset_id}'"
            )));
        }
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
        validate_workspace_relative_path(&stem.dry_wav_path)?;
        if !Path::new(&stem.dry_wav_path)
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("wav"))
        {
            return Err(StudioError::Validation(format!(
                "Stem dry WAV path '{}' must end with '.wav'",
                stem.dry_wav_path
            )));
        }
        ensure_no_symlink_components_under(&self.root_dir, Path::new(&stem.dry_wav_path))?;
        validate_model_stem_wav_path(model_id, &stem.wet_wav_path)?;
        ensure_no_symlink_components_under(&self.root_dir, Path::new(&stem.wet_wav_path))?;
        let mut model = self.load_model(model_id)?;
        model.stems.retain(|s| s.stem_id != stem.stem_id);
        model.stems.push(stem);
        self.save_model(&model)
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
            let rel_wet = Path::new(&removed.wet_wav_path);
            if validate_model_stem_wav_path(model_id, &removed.wet_wav_path).is_ok()
                && ensure_no_symlink_components_under(&self.root_dir, rel_wet).is_ok()
            {
                let p = self.root_dir.join(rel_wet);
                if p.is_file() {
                    let _ = fs::remove_file(p);
                }
            }
        }
        Ok(())
    }
}

fn ensure_no_symlink_components_under(root: &Path, rel: &Path) -> StudioResult<()> {
    use std::path::Component;
    verify_dir_ancestors_no_symlinks(root, false)?;
    let mut cur = resolve_dir_against_cwd(root)?;
    for comp in rel.components() {
        if matches!(comp, Component::ParentDir) {
            return Err(StudioError::Validation(format!(
                "Refusing parent-directory traversal in workspace path '{}'",
                rel.display()
            )));
        }
        cur.push(comp);
        match fs::symlink_metadata(&cur) {
            Ok(meta) if meta.file_type().is_symlink() => {
                return Err(StudioError::Validation(format!(
                    "Refusing symlink path component '{}'",
                    cur.display()
                )));
            }
            Ok(_) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => break,
            Err(err) => return Err(err.into()),
        }
    }
    Ok(())
}

fn validate_model_stem_wav_path(model_id: &str, rel_path: &str) -> StudioResult<()> {
    validate_model_id(model_id)?;
    validate_workspace_relative_path(rel_path)?;
    let p = Path::new(rel_path);
    let prefix = Path::new("models").join(model_id).join("stems");
    let rest = p.strip_prefix(&prefix).map_err(|_| {
        StudioError::Validation(format!(
            "Wet stem WAV '{rel_path}' must reside under '{}'",
            prefix.display()
        ))
    })?;
    let has_wav_ext = p
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("wav"));
    if rest.as_os_str().is_empty() || !has_wav_ext {
        return Err(StudioError::Validation(format!(
            "Wet stem WAV '{rel_path}' must be a '.wav' file under '{}'",
            prefix.display()
        )));
    }
    Ok(())
}

fn validate_range_bounds(
    min: f32,
    max: f32,
    default_value: f32,
    context: &str,
) -> StudioResult<()> {
    if !min.is_finite()
        || !max.is_finite()
        || !default_value.is_finite()
        || min >= max
        || default_value < min
        || default_value > max
    {
        return Err(StudioError::Validation(format!(
            "Finite range bounds min <= default <= max with min < max required in '{context}'"
        )));
    }
    Ok(())
}

fn validate_model_id(model_id: &str) -> StudioResult<()> {
    use std::path::Component;
    if model_id.is_empty()
        || model_id != model_id.trim()
        || model_id == "."
        || model_id == ".."
        || model_id.contains(['/', '\\', '\0', ':'])
        || !matches!(
            Path::new(model_id)
                .components()
                .collect::<Vec<_>>()
                .as_slice(),
            [Component::Normal(_)]
        )
    {
        return Err(StudioError::Validation(format!(
            "Invalid model_id '{model_id}'"
        )));
    }
    Ok(())
}

fn validate_workspace_relative_path(rel_path: &str) -> StudioResult<()> {
    use std::path::Component;
    if rel_path.is_empty() || rel_path != rel_path.trim() || rel_path.contains(['\\', '\0', ':']) {
        return Err(StudioError::Validation(format!(
            "Invalid workspace-relative path '{rel_path}'"
        )));
    }
    let components: Vec<Component<'_>> = Path::new(rel_path).components().collect();
    if components.is_empty() || !components.iter().all(|c| matches!(c, Component::Normal(_))) {
        return Err(StudioError::Validation(format!(
            "Invalid workspace-relative path '{rel_path}'"
        )));
    }
    Ok(())
}

fn atomic_write_text(path: &Path, content: &str) -> StudioResult<()> {
    use std::io::Write as _;
    let seq = ATOMIC_TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp_path = path.with_extension(format!("tmp.{}_{}", std::process::id(), seq));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp_path)?;
    if let Err(err) = file.write_all(content.as_bytes()) {
        let _ = fs::remove_file(&tmp_path);
        return Err(err.into());
    }
    drop(file);
    if let Err(err) = fs::rename(&tmp_path, path) {
        let _ = fs::remove_file(&tmp_path);
        return Err(err.into());
    }
    Ok(())
}

/// Atomically writes an export or report file into a per-user private directory (`0o700`),
/// rejecting symlinks on both the directory and target file (`O_CREAT | O_EXCL`).
pub fn write_private_studio_export_file(
    export_dir: &Path,
    filename: &str,
    content: &[u8],
    overwrite: bool,
) -> StudioResult<PathBuf> {
    use std::io::Write as _;
    use std::path::Component;

    if filename.is_empty()
        || filename != filename.trim()
        || filename.contains(['/', '\\', '\0', ':'])
    {
        return Err(StudioError::Validation(format!(
            "Invalid export filename '{filename}'"
        )));
    }
    let comps: Vec<Component<'_>> = Path::new(filename).components().collect();
    if comps.len() != 1 || !matches!(comps[0], Component::Normal(_)) {
        return Err(StudioError::Validation(format!(
            "Invalid export filename '{filename}'"
        )));
    }

    ensure_private_dir_no_symlink_ancestors(export_dir)?;

    let target = export_dir.join(filename);
    match fs::symlink_metadata(&target) {
        Ok(meta) if meta.file_type().is_symlink() => {
            return Err(StudioError::Validation(format!(
                "Refusing to follow symlink at export path: {}",
                target.display()
            )));
        }
        Ok(meta) if !meta.is_file() => {
            return Err(StudioError::Validation(format!(
                "Export target is not a regular file: {}",
                target.display()
            )));
        }
        Ok(_) if !overwrite => return Ok(target),
        Ok(_) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => return Err(err.into()),
    }

    let seq = ATOMIC_TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp_path = target.with_extension(format!("tmp.{}_{}", std::process::id(), seq));
    let mut opts = fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        opts.mode(0o600);
    }
    let mut file = opts.open(&tmp_path)?;
    if let Err(err) = file.write_all(content) {
        let _ = fs::remove_file(&tmp_path);
        return Err(err.into());
    }
    drop(file);
    if let Err(err) = fs::rename(&tmp_path, &target) {
        let _ = fs::remove_file(&tmp_path);
        return Err(err.into());
    }
    let final_meta = fs::symlink_metadata(&target)?;
    if final_meta.file_type().is_symlink() || !final_meta.is_file() {
        return Err(StudioError::Validation(format!(
            "Export target violated regular-file invariant: {}",
            target.display()
        )));
    }
    Ok(target)
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
        assert_eq!(
            resolve_os_config_dir_from_vars(PathBuf::from("/u"), None, None, "macos"),
            PathBuf::from("/u/Library/Application Support/dnam")
        );
        assert_eq!(
            resolve_os_config_dir_from_vars(PathBuf::from("/u"), Some("/ad"), None, "windows"),
            PathBuf::from("/ad/dnam")
        );
        assert_eq!(
            resolve_os_config_dir_from_vars(PathBuf::from("/u"), None, None, "windows"),
            PathBuf::from("/u/AppData/Roaming/dnam")
        );
        assert_eq!(
            resolve_os_config_dir_from_vars(PathBuf::from("/u"), None, Some("/xdg"), "linux"),
            PathBuf::from("/xdg/dnam")
        );
        assert_eq!(
            resolve_os_config_dir_from_vars(PathBuf::from("/u"), None, None, "linux"),
            PathBuf::from("/u/.config/dnam")
        );

        let key_path = dir.join("nested_keys").join("creator.key");
        let (p1, seed1, pub1) =
            ensure_creator_ed25519_keypair(Some(&key_path)).expect("generate key");
        assert_eq!(p1, key_path);
        assert_eq!(pub1.len(), 64);

        let (_, seed2, pub2) =
            ensure_creator_ed25519_keypair(Some(&key_path)).expect("reuse existing key");
        assert_eq!(seed1, seed2);
        assert_eq!(pub1, pub2);

        // Raw 32-byte binary key file is also supported (when mode is 0600 on Unix)
        let raw_key_path = dir.join("raw32.key");
        fs::write(&raw_key_path, seed1).expect("write raw 32B key");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let world_readable_key = dir.join("world_readable.key");
            fs::write(&world_readable_key, seed1).expect("write 0644 key");
            fs::set_permissions(&world_readable_key, fs::Permissions::from_mode(0o644))
                .expect("chmod 0644");
            assert!(ensure_creator_ed25519_keypair(Some(&world_readable_key)).is_err());
            fs::set_permissions(&raw_key_path, fs::Permissions::from_mode(0o600))
                .expect("chmod 0600 raw32");
        }
        let (_, seed3, pub3) =
            ensure_creator_ed25519_keypair(Some(&raw_key_path)).expect("load raw 32B key");
        assert_eq!(seed3, seed1);
        assert_eq!(pub3, pub1);

        // Invalid key length raises Validation error (both UTF-8 hex and non-UTF-8 binary)
        let bad_key = dir.join("bad.key");
        fs::write(&bad_key, "001122\n").expect("write bad key");
        let bad_bin_key = dir.join("bad_bin.key");
        fs::write(&bad_bin_key, [0xff_u8, 0xfe, 0xfd]).expect("write bad bin key");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            fs::set_permissions(&bad_key, fs::Permissions::from_mode(0o600)).unwrap();
            fs::set_permissions(&bad_bin_key, fs::Permissions::from_mode(0o600)).unwrap();
        }
        assert!(ensure_creator_ed25519_keypair(Some(&bad_key)).is_err());
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
        assert!(parse_compact_range_spec("drive=NaN..10").is_err());
        assert!(parse_compact_range_spec("drive=-inf..inf").is_err());
        assert!(parse_compact_range_spec("drive:Drive:0..10@NaN").is_err());
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
        assert!(DnamProject::init(&dir, "Dup", "Author").is_err());

        // Legacy .gitignore without trailing newline gets .dnam-cache/ rules appended on open()
        fs::write(dir.join(".gitignore"), "# minimal").unwrap();
        let reloaded = DnamProject::open(&dir).expect("open");
        assert_eq!(reloaded.manifest().project.name, "Test Pedal Pack");
        let gi = fs::read_to_string(dir.join(".gitignore")).unwrap();
        assert!(gi.contains("models/*/.dnam-cache/") && gi.contains("\n.dnam-cache/\n"));

        // Default user config keypair & relative/implicit-cwd path helpers
        let (_, def_seed, def_pub) = ensure_creator_ed25519_keypair(None).expect("default key");
        assert_eq!(def_seed.len(), 32);
        assert_eq!(def_pub.len(), 64);
        assert!(
            resolve_dir_against_cwd(Path::new(""))
                .unwrap()
                .is_absolute()
        );
        assert!(
            resolve_dir_against_cwd(Path::new("."))
                .unwrap()
                .is_absolute()
        );
        assert!(
            resolve_dir_against_cwd(Path::new("rel_child"))
                .unwrap()
                .ends_with("rel_child")
        );
        assert!(resolve_dir_against_cwd(Path::new("../escape")).is_err());
        ensure_private_dir_no_symlink_ancestors(Path::new("")).unwrap();
        ensure_private_dir_no_symlink_ancestors(Path::new(".")).unwrap();

        let cfg_ws = dir.join("sub_ws_cfg");
        let cfg_dir = dir.join("custom_cfg");
        let _ = DnamProject::init_with_config_dir(&cfg_ws, "Cfg WS", "Auth", &cfg_dir)
            .expect("init_with_config_dir");
        let _ = fs::remove_dir_all(&cfg_ws);

        assert!(proj.add_model("", "Bad", "Full").is_err());
        for bad_id in [".", "..", "/abs", "a/b", "a\\b", "bad:id", " lead", "a\0b"] {
            assert!(proj.add_model(bad_id, "Bad", "Full").is_err());
            assert!(proj.remove_model(bad_id, true).is_err());
            assert!(proj.load_model(bad_id).is_err());
        }
        assert!(dir.join("dnam.toml").is_file() && dir.join("models").is_dir());
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
        let pol_m = proj
            .update_model_distribution_policy(
                "b7k_drive",
                &[
                    "open".to_string(),
                    "device_allowlisted".to_string(),
                    "enclave_sealed".to_string(),
                ],
                &["darkglass".to_string()],
            )
            .expect("update distribution policy");
        assert_eq!(
            pol_m
                .distribution_policy
                .as_ref()
                .unwrap()
                .allowed_oem_vendors,
            vec!["darkglass".to_string()]
        );
        assert!(
            proj.update_model_distribution_policy("b7k_drive", &["bogus".to_string()], &[])
                .is_err()
        );
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
        proj.update_control("b7k_drive", "attack", None, None, None, Some("2"), None)
            .expect("set switch default index 2");
        assert!(
            proj.update_control(
                "b7k_drive",
                "attack",
                None,
                None,
                None,
                None,
                Some("clean,lead")
            )
            .is_err()
        );
        assert!(
            proj.update_control(
                "b7k_drive",
                "drive",
                None,
                Some(10.0),
                Some(0.0),
                None,
                None
            )
            .is_err()
        );
        assert!(
            proj.update_control("b7k_drive", "drive", None, None, None, Some("NaN"), None)
                .is_err()
        );
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
        assert!(
            proj.add_preset("b7k_drive", "", "Bad", BTreeMap::new(), BTreeMap::new())
                .is_err()
        );
        assert!(proj.remove_preset("b7k_drive", "missing").is_err());
        assert!(
            proj.update_preset("b7k_drive", "missing", Some("X"), None, None)
                .is_err()
        );

        // Stem registration, update, and removal
        let mut stem_ranges = BTreeMap::new();
        stem_ranges.insert("drive".to_string(), 0.5);
        proj.register_stem(
            "b7k_drive",
            StemRecord {
                stem_id: "s_00".to_string(),
                dry_wav_path: "models/b7k_drive/stems/input.wav".to_string(),
                wet_wav_path: "models/b7k_drive/stems/s_00.wav".to_string(),
                normalized_ranges: stem_ranges.clone(),
                switches: BTreeMap::new(),
                pcm_sha256: "0".repeat(64),
                dry_pcm_sha256: "0".repeat(64),
                aligned_lag_samples: 0.0,
                snr_db: 80.0,
                noise_floor_dbfs: -96.0,
                clean_pad_db: 0.0,
                peak_dbfs: -3.0,
            },
        )
        .expect("register stem");

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
        for (bad_dry, bad_wet) in [
            (
                "models/b7k_drive/stems/input.txt",
                "models/b7k_drive/stems/s.wav",
            ),
            ("models/b7k_drive/stems/input.wav", "other/s.wav"),
            (
                "models/b7k_drive/stems/input.wav",
                "models/b7k_drive/stems/s.txt",
            ),
        ] {
            assert!(
                proj.register_stem(
                    "b7k_drive",
                    StemRecord {
                        stem_id: "bad_ext".to_string(),
                        dry_wav_path: bad_dry.to_string(),
                        wet_wav_path: bad_wet.to_string(),
                        normalized_ranges: BTreeMap::new(),
                        switches: BTreeMap::new(),
                        pcm_sha256: "0".repeat(64),
                        dry_pcm_sha256: "0".repeat(64),
                        aligned_lag_samples: 0.0,
                        snr_db: 80.0,
                        noise_floor_dbfs: -96.0,
                        clean_pad_db: 0.0,
                        peak_dbfs: -3.0,
                    },
                )
                .is_err()
            );
        }
        assert!(
            proj.register_stem(
                "b7k_drive",
                StemRecord {
                    stem_id: "bad_stem".to_string(),
                    dry_wav_path: "../escape.wav".to_string(),
                    wet_wav_path: "/abs/escape.wav".to_string(),
                    normalized_ranges: BTreeMap::new(),
                    switches: BTreeMap::new(),
                    pcm_sha256: "0".repeat(64),
                    dry_pcm_sha256: "0".repeat(64),
                    aligned_lag_samples: 0.0,
                    snr_db: 80.0,
                    noise_floor_dbfs: -96.0,
                    clean_pad_db: 0.0,
                    peak_dbfs: -3.0,
                }
            )
            .is_err()
        );

        for bad_wet in ["dnam.toml", "models/b7k_drive/stems/not_wav.toml"] {
            assert!(
                proj.register_stem(
                    "b7k_drive",
                    StemRecord {
                        stem_id: "bad_wet".to_string(),
                        dry_wav_path: "models/b7k_drive/stems/input.wav".to_string(),
                        wet_wav_path: bad_wet.to_string(),
                        normalized_ranges: BTreeMap::new(),
                        switches: BTreeMap::new(),
                        pcm_sha256: "0".repeat(64),
                        dry_pcm_sha256: "0".repeat(64),
                        aligned_lag_samples: 0.0,
                        snr_db: 80.0,
                        noise_floor_dbfs: -96.0,
                        clean_pad_db: 0.0,
                        peak_dbfs: -3.0,
                    }
                )
                .is_err()
            );
        }

        // Tampered model.toml pointing wet_wav_path at dnam.toml must NOT unlink dnam.toml
        let mut tampered = proj.load_model("b7k_drive").expect("load tampered");
        tampered.stems.push(StemRecord {
            stem_id: "tampered_manifest".to_string(),
            dry_wav_path: "models/b7k_drive/stems/input.wav".to_string(),
            wet_wav_path: "dnam.toml".to_string(),
            normalized_ranges: BTreeMap::new(),
            switches: BTreeMap::new(),
            pcm_sha256: "0".repeat(64),
            dry_pcm_sha256: "0".repeat(64),
            aligned_lag_samples: 0.0,
            snr_db: 80.0,
            noise_floor_dbfs: -96.0,
            clean_pad_db: 0.0,
            peak_dbfs: -3.0,
        });
        proj.save_model(&tampered).expect("save tampered");
        proj.remove_stem("b7k_drive", "tampered_manifest", true)
            .expect("remove tampered stem safely");
        assert!(dir.join("dnam.toml").is_file());

        // Intermediate directory symlink under models/<id>/stems must be rejected & never delete external files
        #[cfg(unix)]
        {
            let ext_dir = temp_dir("ext_symlink_target");
            let ext_wav = ext_dir.join("victim.wav");
            fs::write(&ext_wav, b"keep").expect("write victim");
            let stems_dir = dir.join("models").join("b7k_drive").join("stems");
            let _ = fs::remove_dir_all(&stems_dir);
            std::os::unix::fs::symlink(&ext_dir, &stems_dir).expect("symlink stems");
            assert!(
                proj.register_stem(
                    "b7k_drive",
                    StemRecord {
                        stem_id: "sym_stem".to_string(),
                        dry_wav_path: "models/b7k_drive/input.wav".to_string(),
                        wet_wav_path: "models/b7k_drive/stems/victim.wav".to_string(),
                        normalized_ranges: BTreeMap::new(),
                        switches: BTreeMap::new(),
                        pcm_sha256: "0".repeat(64),
                        dry_pcm_sha256: "0".repeat(64),
                        aligned_lag_samples: 0.0,
                        snr_db: 80.0,
                        noise_floor_dbfs: -96.0,
                        clean_pad_db: 0.0,
                        peak_dbfs: -3.0,
                    }
                )
                .is_err()
            );
            let mut m2 = proj.load_model("b7k_drive").expect("load m2");
            m2.stems.push(StemRecord {
                stem_id: "sym_stem".to_string(),
                dry_wav_path: "models/b7k_drive/input.wav".to_string(),
                wet_wav_path: "models/b7k_drive/stems/victim.wav".to_string(),
                normalized_ranges: BTreeMap::new(),
                switches: BTreeMap::new(),
                pcm_sha256: "0".repeat(64),
                dry_pcm_sha256: "0".repeat(64),
                aligned_lag_samples: 0.0,
                snr_db: 80.0,
                noise_floor_dbfs: -96.0,
                clean_pad_db: 0.0,
                peak_dbfs: -3.0,
            });
            proj.save_model(&m2).expect("save m2");
            proj.remove_stem("b7k_drive", "sym_stem", true)
                .expect("remove sym_stem without deleting external target");
            assert!(ext_wav.is_file());
            let _ = fs::remove_file(&stems_dir);
            let _ = fs::remove_dir_all(&ext_dir);
        }

        // Reject tampered model.toml declaring mismatched model_id & non-.wav dry_wav_path
        proj.add_model("victim_m", "Victim", "Lite")
            .expect("add victim_m");
        assert!(
            proj.register_stem(
                "victim_m",
                StemRecord {
                    stem_id: "bad_dry".to_string(),
                    dry_wav_path: "dnam.toml".to_string(),
                    wet_wav_path: "models/victim_m/stems/s0.wav".to_string(),
                    normalized_ranges: BTreeMap::new(),
                    switches: BTreeMap::new(),
                    pcm_sha256: "0".repeat(64),
                    dry_pcm_sha256: "0".repeat(64),
                    aligned_lag_samples: 0.0,
                    snr_db: 80.0,
                    noise_floor_dbfs: -96.0,
                    clean_pad_db: 0.0,
                    peak_dbfs: -3.0,
                }
            )
            .is_err()
        );
        let victim_toml = fs::read_to_string(dir.join("models/victim_m/model.toml")).unwrap();
        fs::write(dir.join("models/b7k_drive/model.toml"), &victim_toml).unwrap();
        assert!(proj.load_model("b7k_drive").is_err());

        // Reject re-initializing existing workspace, duplicate add_model, unregistered remove_model, and empty preset_id
        assert!(DnamProject::init(&dir, "Clobber", "attacker").is_err());
        assert!(proj.add_model("victim_m", "Duplicate", "Lite").is_err());
        assert!(proj.remove_model("unregistered_model", true).is_err());
        assert!(
            proj.add_preset("victim_m", "", "Empty", BTreeMap::new(), BTreeMap::new())
                .is_err()
        );
        assert!(
            proj.add_preset("victim_m", " pad ", "Pad", BTreeMap::new(), BTreeMap::new())
                .is_err()
        );

        // Private export writer + symlinked/non-regular key & export rejection
        let exp_dir = dir.join("private_exports");
        let exp_p1 = write_private_studio_export_file(&exp_dir, "ref.dnam", b"v1", false).unwrap();
        assert_eq!(fs::read(&exp_p1).unwrap(), b"v1");
        let exp_p2 = write_private_studio_export_file(&exp_dir, "ref.dnam", b"v2", false).unwrap();
        assert_eq!(fs::read(&exp_p2).unwrap(), b"v1");
        let exp_p3 = write_private_studio_export_file(&exp_dir, "ref.dnam", b"v2", true).unwrap();
        assert_eq!(fs::read(&exp_p3).unwrap(), b"v2");
        assert!(write_private_studio_export_file(&exp_dir, "../bad.dnam", b"x", true).is_err());
        assert!(write_private_studio_export_file(&exp_dir, "  ", b"x", true).is_err());
        assert!(write_private_studio_export_file(&exp_dir, ".", b"x", true).is_err());
        assert!(write_private_studio_export_file(&exp_p3, "nested.dnam", b"x", true).is_err());
        let sub_target = exp_dir.join("dir_target.dnam");
        fs::create_dir_all(&sub_target).unwrap();
        assert!(write_private_studio_export_file(&exp_dir, "dir_target.dnam", b"x", true).is_err());
        assert!(ensure_creator_ed25519_keypair(Some(&sub_target)).is_err());
        #[cfg(unix)]
        {
            let sym_key = dir.join("sym.key");
            std::os::unix::fs::symlink(dir.join("dnam.toml"), &sym_key).unwrap();
            assert!(ensure_creator_ed25519_keypair(Some(&sym_key)).is_err());

            let sym_key_dir = dir.join("sym_key_dir");
            std::os::unix::fs::symlink(&exp_dir, &sym_key_dir).unwrap();
            assert!(ensure_creator_ed25519_keypair(Some(&sym_key_dir.join("new.key"))).is_err());
            assert!(
                ensure_creator_ed25519_keypair(Some(
                    &sym_key_dir.join("sub_keys").join("nested.key")
                ))
                .is_err()
            );
            let existing_key_in_exp = exp_dir.join("existing.key");
            fs::write(&existing_key_in_exp, format!("{}\n", "42".repeat(32))).unwrap();
            {
                use std::os::unix::fs::PermissionsExt as _;
                fs::set_permissions(&existing_key_in_exp, fs::Permissions::from_mode(0o600))
                    .unwrap();
            }
            assert!(
                ensure_creator_ed25519_keypair(Some(&sym_key_dir.join("existing.key"))).is_err()
            );
            assert!(
                write_private_studio_export_file(
                    &sym_key_dir.join("sub_exports"),
                    "out.dnam",
                    b"x",
                    true
                )
                .is_err()
            );
            assert!(
                write_private_studio_export_file(
                    &exp_dir.join("..").join("private_exports"),
                    "out.dnam",
                    b"x",
                    true
                )
                .is_err()
            );

            let sym_exp_target = exp_dir.join("sym_report.html");
            std::os::unix::fs::symlink(dir.join("dnam.toml"), &sym_exp_target).unwrap();
            assert!(
                write_private_studio_export_file(&exp_dir, "sym_report.html", b"html", true)
                    .is_err()
            );

            // Symlinked workspace root & un-sticky world-writable ancestor directory are rejected
            let sym_ws = dir.join("sym_ws");
            std::os::unix::fs::symlink(&exp_dir, &sym_ws).unwrap();
            assert!(DnamProject::init(&sym_ws, "Sym", "Auth").is_err());
            assert!(DnamProject::open(&sym_ws).is_err());

            let ww_dir = dir.join("world_writable_ancestor");
            fs::create_dir_all(&ww_dir).unwrap();
            {
                use std::os::unix::fs::PermissionsExt as _;
                fs::set_permissions(&ww_dir, fs::Permissions::from_mode(0o777)).unwrap();
            }
            assert!(
                ensure_creator_ed25519_keypair(Some(&ww_dir.join("sub").join("k.key"))).is_err()
            );
            assert!(
                write_private_studio_export_file(&ww_dir.join("sub"), "out.dnam", b"x", true)
                    .is_err()
            );
            {
                use std::os::unix::fs::PermissionsExt as _;
                fs::set_permissions(&ww_dir, fs::Permissions::from_mode(0o700)).unwrap();
            }
        }

        let _ = fs::remove_dir_all(&dir);
    }
}
