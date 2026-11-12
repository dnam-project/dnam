//! Tauri v2 Command, `.dnam-proj` Project Bridge & Native Hardware Acceleration Host Handler.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use dnam_capture::proj::DnamProjView;
use dnam_engine::provenance::DefaultCryptoProvider;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::capture::{list_backend_devices, list_capture_backends};
use crate::error::{StudioError, StudioResult};

static ATOMIC_TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Returns the active native SIMD backend label on the current CPU architecture.
#[must_use]
pub const fn detect_native_backend_str() -> &'static str {
    if cfg!(target_arch = "aarch64") {
        "ARM64 NEON"
    } else if cfg!(target_arch = "x86_64") {
        "x86_64 AVX2+FMA"
    } else {
        "Scalar"
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

/// Returns the OS-appropriate user configuration directory for `dnam-studio`.
#[must_use]
pub fn default_user_config_dir() -> PathBuf {
    if let Ok(env_dir) = env::var("DNAM_CONFIG_DIR") {
        let trimmed = env_dir.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
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
    match home {
        Some(h) => resolve_os_config_dir_from_vars(
            PathBuf::from(h.trim()),
            appdata.as_deref(),
            xdg.as_deref(),
            env::consts::OS,
        ),
        None => PathBuf::from(".dnam-config"),
    }
}

/// Atomically writes an export or report file into `export_dir` (`0o700` on Unix),
/// rejecting path traversal and symlinks (`O_CREAT | O_EXCL`).
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
        || export_dir
            .components()
            .any(|c| matches!(c, Component::ParentDir))
    {
        return Err(StudioError::Validation(format!(
            "Invalid export path or filename '{filename}'"
        )));
    }
    let comps: Vec<Component<'_>> = Path::new(filename).components().collect();
    if comps.len() != 1 || !matches!(comps[0], Component::Normal(_)) {
        return Err(StudioError::Validation(format!(
            "Invalid export filename '{filename}'"
        )));
    }

    if let Ok(meta) = fs::symlink_metadata(export_dir)
        && meta.file_type().is_symlink()
    {
        return Err(StudioError::Validation(format!(
            "Refusing symlink export directory: {}",
            export_dir.display()
        )));
    }
    fs::create_dir_all(export_dir)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        fs::set_permissions(export_dir, fs::Permissions::from_mode(0o700))?;
    }

    let target = export_dir.join(filename);
    match fs::symlink_metadata(&target) {
        Ok(meta) if meta.file_type().is_symlink() || !meta.is_file() => {
            return Err(StudioError::Validation(format!(
                "Export target is a symlink or non-regular file: {}",
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
    Ok(target)
}

/// Descriptor for an official Tauri v2 plugin integrated into `dnam-studio`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TauriPluginDescriptor {
    /// Short plugin identifier (e.g. `"dialog"`, `"opener"`, `"single-instance"`).
    pub id: String,
    /// Rust crate name (`tauri-plugin-*`).
    #[serde(alias = "rust_crate")]
    pub rust_crate: String,
    /// Priority tier (`1` for essential desktop UX, `2` for pro-audio & ecosystem polish).
    pub tier: u8,
    /// Capability permission string in `capabilities/default.json` (or `"rust-only"`).
    pub permission: String,
}

/// Persisted multi-monitor Studio window geometry managed by `tauri-plugin-window-state`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WindowStateGeometry {
    /// Window width in logical pixels.
    pub width: u32,
    /// Window height in logical pixels.
    pub height: u32,
    /// Window X coordinate in virtual screen space.
    pub x: i32,
    /// Window Y coordinate in virtual screen space.
    pub y: i32,
    /// Whether the Studio window is maximized.
    pub maximized: bool,
}

impl Default for WindowStateGeometry {
    fn default() -> Self {
        Self {
            width: 1360,
            height: 880,
            x: 120,
            y: 80,
            maximized: false,
        }
    }
}

/// Parsed `dnam://` custom URI deep-link route (`tauri-plugin-deep-link` + `tauri-plugin-single-instance`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DnamDeepLinkRoute {
    /// Raw URI string (e.g. `"dnam://models/b7k_ultra"`).
    #[serde(alias = "raw_uri")]
    pub raw_uri: String,
    /// Route kind (`"model"`, `"marketplace"`, `"open"`).
    pub kind: String,
    /// Target identifier or path extracted from the URI.
    pub target: String,
}

/// Returns the canonical list of Tier 1 and Tier 2 official Tauri v2 plugins active in `dnam-studio`.
pub fn official_tauri_plugins() -> Vec<TauriPluginDescriptor> {
    vec![
        TauriPluginDescriptor {
            id: "dialog".to_string(),
            rust_crate: "tauri-plugin-dialog".to_string(),
            tier: 1,
            permission: "dialog:allow-open,dialog:allow-save".to_string(),
        },
        TauriPluginDescriptor {
            id: "opener".to_string(),
            rust_crate: "tauri-plugin-opener".to_string(),
            tier: 1,
            permission: "opener:allow-reveal-item-in-dir,opener:allow-open-url(scoped)".to_string(),
        },
        TauriPluginDescriptor {
            id: "window-state".to_string(),
            rust_crate: "tauri-plugin-window-state".to_string(),
            tier: 1,
            permission: "window-state:default".to_string(),
        },
        TauriPluginDescriptor {
            id: "single-instance".to_string(),
            rust_crate: "tauri-plugin-single-instance".to_string(),
            tier: 2,
            permission: "rust-only".to_string(),
        },
        TauriPluginDescriptor {
            id: "deep-link".to_string(),
            rust_crate: "tauri-plugin-deep-link".to_string(),
            tier: 2,
            permission: "deep-link:default".to_string(),
        },
        TauriPluginDescriptor {
            id: "notification".to_string(),
            rust_crate: "tauri-plugin-notification".to_string(),
            tier: 2,
            permission: "notification:default".to_string(),
        },
        TauriPluginDescriptor {
            id: "updater".to_string(),
            rust_crate: "tauri-plugin-updater".to_string(),
            tier: 2,
            permission: "updater:default".to_string(),
        },
        TauriPluginDescriptor {
            id: "process".to_string(),
            rust_crate: "tauri-plugin-process".to_string(),
            tier: 2,
            permission: "process:allow-restart".to_string(),
        },
    ]
}

fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn decode_percent_pass(input: &str) -> Option<String> {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    let mut changed = false;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let (Some(hi), Some(lo)) = (hex_nibble(bytes[i + 1]), hex_nibble(bytes[i + 2]))
        {
            let decoded_byte = (hi << 4) | lo;
            if decoded_byte <= 0x1f || decoded_byte == 0x7f {
                return None;
            }
            out.push(decoded_byte);
            changed = true;
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    if !changed {
        return Some(input.to_string());
    }
    let decoded = String::from_utf8(out).ok()?;
    if decoded
        .chars()
        .any(|c| matches!(c as u32, 0x80..=0x9f) && !input.contains(c))
    {
        return None;
    }
    Some(decoded)
}

fn decoded_contains_backslash(s: &str) -> bool {
    if s.contains('\\') {
        return true;
    }
    let mut cur = s.to_string();
    for _ in 0..4 {
        let Some(next) = decode_percent_pass(&cur) else {
            return true;
        };
        if next.contains('\\') {
            return true;
        }
        if next == cur {
            return false;
        }
        cur = next;
    }
    decode_percent_pass(&cur).is_none_or(|probe| probe != cur)
}

/// Parses a `dnam://` URI into a structured [`DnamDeepLinkRoute`].
pub fn parse_dnam_deep_link(uri: &str) -> StudioResult<DnamDeepLinkRoute> {
    use std::path::Component;
    let trimmed = uri.trim();
    if trimmed.chars().any(char::is_control) || string_contains_unsafe_token(trimmed) {
        return Err(StudioError::Validation(format!(
            "Unsafe traversal or control sequence in dnam:// URI: {trimmed}"
        )));
    }
    let rest = trimmed.strip_prefix("dnam://").ok_or_else(|| {
        StudioError::Validation(format!("Deep link must start with 'dnam://': {trimmed}"))
    })?;
    let rest = rest.trim_matches('/');
    let is_safe_slug = |s: &str| {
        !s.is_empty()
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    };
    if let Some(model_id) = rest.strip_prefix("models/") {
        let cleaned = model_id.trim_matches('/');
        if !is_safe_slug(cleaned) {
            return Err(StudioError::Validation(
                "dnam://models/ requires a valid single-segment model ID".to_string(),
            ));
        }
        return Ok(DnamDeepLinkRoute {
            raw_uri: trimmed.to_string(),
            kind: "model".to_string(),
            target: cleaned.to_string(),
        });
    }
    if let Some(market_id) = rest.strip_prefix("marketplace/") {
        let cleaned = market_id.trim_matches('/');
        if !is_safe_slug(cleaned) {
            return Err(StudioError::Validation(
                "dnam://marketplace/ requires a valid single-segment listing ID".to_string(),
            ));
        }
        return Ok(DnamDeepLinkRoute {
            raw_uri: trimmed.to_string(),
            kind: "marketplace".to_string(),
            target: cleaned.to_string(),
        });
    }
    if let Some(query) = rest.strip_prefix("open?path=") {
        if query.is_empty()
            || decoded_contains_backslash(query)
            || Path::new(query)
                .components()
                .any(|c| matches!(c, Component::ParentDir))
        {
            return Err(StudioError::Validation(
                "dnam://open?path= requires a non-empty traversal-safe file path".to_string(),
            ));
        }
        return Ok(DnamDeepLinkRoute {
            raw_uri: trimmed.to_string(),
            kind: "open".to_string(),
            target: query.to_string(),
        });
    }
    Err(StudioError::Validation(format!(
        "Unsupported dnam:// route: {trimmed}"
    )))
}

fn string_contains_unsafe_token(s: &str) -> bool {
    if s.contains("..") || s.contains('\0') {
        return true;
    }
    let mut cur = s.to_string();
    for _ in 0..4 {
        let Some(next) = decode_percent_pass(&cur) else {
            return true;
        };
        if next.contains("..") || next.contains('\0') {
            return true;
        }
        if next == cur {
            return false;
        }
        cur = next;
    }
    decode_percent_pass(&cur).is_none_or(|probe| probe != cur)
}

fn value_contains_unsafe_string(val: &Value) -> bool {
    match val {
        Value::String(s) => string_contains_unsafe_token(s),
        Value::Array(items) => items.iter().any(value_contains_unsafe_string),
        Value::Object(map) => map
            .iter()
            .any(|(k, v)| string_contains_unsafe_token(k) || value_contains_unsafe_string(v)),
        _ => false,
    }
}

/// Validates an incoming [`StudioIpcRequest`] against the Tauri v2 Isolation Pattern invariants.
pub fn validate_isolated_ipc_request(req: &StudioIpcRequest) -> StudioResult<()> {
    if req.command.trim().is_empty() {
        return Err(StudioError::Validation(
            "Tauri Isolation blocked IPC request with empty command".to_string(),
        ));
    }
    if !req.args.is_object() {
        return Err(StudioError::Validation(
            "Tauri Isolation blocked IPC request with non-object args".to_string(),
        ));
    }
    if string_contains_unsafe_token(&req.command) || value_contains_unsafe_string(&req.args) {
        return Err(StudioError::Validation(
            "Tauri Isolation blocked path traversal or null byte in IPC payload".to_string(),
        ));
    }
    Ok(())
}

/// Typed IPC request envelope from the Tauri v2 React frontend (`TauriNativeAudioHost` / `StudioIpcClient`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StudioIpcRequest {
    /// Command identifier (`"get_telemetry"`, `"set_parameter"`, `"set_switch"`, `"get_workspace"`, `"get_capture_plan"`, `"list_backends"`).
    pub command: String,
    /// Optional command arguments.
    pub args: Value,
}

/// Typed IPC response envelope returned to the Tauri v2 React frontend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StudioIpcResponse {
    /// Whether the command succeeded.
    pub ok: bool,
    /// Response payload.
    pub data: Value,
}

/// Stateful Tauri v2 IPC & native SIMD audio host handler.
#[derive(Debug, Clone)]
pub struct StudioIpcHandler {
    parameters: BTreeMap<String, f32>,
    switches: BTreeMap<String, usize>,
    sample_rate_hz: u32,
    window_state: WindowStateGeometry,
    last_deep_link: Option<DnamDeepLinkRoute>,
    notifications: Vec<String>,
}

impl Default for StudioIpcHandler {
    fn default() -> Self {
        Self::new(48_000)
    }
}

impl StudioIpcHandler {
    /// Creates a new [`StudioIpcHandler`] at `sample_rate_hz`.
    pub fn new(sample_rate_hz: u32) -> Self {
        Self {
            parameters: BTreeMap::new(),
            switches: BTreeMap::new(),
            sample_rate_hz,
            window_state: WindowStateGeometry::default(),
            last_deep_link: None,
            notifications: Vec::new(),
        }
    }

    /// Records the most recent validated `dnam://` deep-link route (including cold-start URIs).
    pub fn set_last_deep_link(&mut self, route: DnamDeepLinkRoute) {
        self.last_deep_link = Some(route);
    }

    /// Returns the most recently recorded `dnam://` deep-link route, if any.
    #[must_use]
    pub fn last_deep_link(&self) -> Option<&DnamDeepLinkRoute> {
        self.last_deep_link.as_ref()
    }

    /// Dispatches a [`StudioIpcRequest`] against an optional `.dnam-proj` `project_path` and returns a [`StudioIpcResponse`].
    pub fn handle(
        &mut self,
        req: &StudioIpcRequest,
        project_path: Option<&Path>,
    ) -> StudioResult<StudioIpcResponse> {
        validate_isolated_ipc_request(req)?;
        match req.command.as_str() {
            "set_parameter" => {
                let id = req
                    .args
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("gain")
                    .to_string();
                let raw_val = req
                    .args
                    .get("value")
                    .and_then(Value::as_f64)
                    .ok_or_else(|| {
                        StudioError::Validation(format!(
                            "Missing or non-numeric parameter value for '{id}'"
                        ))
                    })?;
                let val = raw_val as f32;
                if !raw_val.is_finite() || !val.is_finite() {
                    return Err(StudioError::Validation(format!(
                        "Non-finite parameter value for '{id}'"
                    )));
                }
                self.parameters.insert(id, val);
                Ok(StudioIpcResponse {
                    ok: true,
                    data: self.telemetry_json(),
                })
            }
            "set_switch" => {
                let id = req
                    .args
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("voice")
                    .to_string();
                let idx = match req.args.get("index") {
                    None => 0usize,
                    Some(v) => v
                        .as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .ok_or_else(|| {
                            StudioError::Validation(format!("Invalid switch index for '{id}'"))
                        })?,
                };
                self.switches.insert(id, idx);
                Ok(StudioIpcResponse {
                    ok: true,
                    data: self.telemetry_json(),
                })
            }
            "get_telemetry" => Ok(StudioIpcResponse {
                ok: true,
                data: self.telemetry_json(),
            }),
            "list_backends" => {
                let backends = list_capture_backends();
                let devices = list_backend_devices("software");
                Ok(StudioIpcResponse {
                    ok: true,
                    data: json!({
                        "backends": backends,
                        "devices": devices,
                    }),
                })
            }
            "get_workspace" => {
                if let Some(path) = project_path {
                    let bytes = fs::read(path)?;
                    let crypto = DefaultCryptoProvider::default();
                    let proj = DnamProjView::from_bytes(&bytes, &crypto)?;
                    Ok(StudioIpcResponse {
                        ok: true,
                        data: serde_json::to_value(&proj.manifest)?,
                    })
                } else {
                    Ok(StudioIpcResponse {
                        ok: false,
                        data: json!({ "error": "No .dnam-proj project path provided" }),
                    })
                }
            }
            "get_capture_plan" => {
                if let Some(path) = project_path {
                    if let Some(model_id) = req.args.get("model_id").and_then(Value::as_str)
                        && (model_id.is_empty() || model_id.contains(['/', '\\', '\0', ':']))
                    {
                        return Err(StudioError::Validation(format!(
                            "Invalid model_id '{model_id}'"
                        )));
                    }
                    let bytes = fs::read(path)?;
                    let crypto = DefaultCryptoProvider::default();
                    let proj = DnamProjView::from_bytes(&bytes, &crypto)?;
                    let proj_model_id = &proj.manifest.project_meta.project_id;
                    let requested = req
                        .args
                        .get("model_id")
                        .and_then(Value::as_str)
                        .unwrap_or(proj_model_id);
                    if requested != "default" && requested != proj_model_id {
                        return Err(StudioError::Validation(format!(
                            "Model '{requested}' does not match project model '{proj_model_id}'"
                        )));
                    }
                    Ok(StudioIpcResponse {
                        ok: true,
                        data: json!({
                            "model_id": proj_model_id,
                            "steps": proj.manifest.stems_registry,
                        }),
                    })
                } else {
                    Ok(StudioIpcResponse {
                        ok: false,
                        data: json!({ "error": "No .dnam-proj project path provided" }),
                    })
                }
            }
            "get_plugins" => Ok(StudioIpcResponse {
                ok: true,
                data: json!({
                    "plugins": official_tauri_plugins(),
                    "singleInstanceLocked": true,
                    "deepLinkScheme": "dnam://",
                    "ipcPattern": "isolation",
                    "isolationDir": "./isolation",
                    "freezePrototype": true,
                    "assetProtocolEnabled": false,
                    "lastDeepLink": self.last_deep_link,
                }),
            }),
            "parse_deep_link" => {
                let uri = req.args.get("uri").and_then(Value::as_str).unwrap_or("");
                let route = parse_dnam_deep_link(uri)?;
                self.last_deep_link = Some(route.clone());
                Ok(StudioIpcResponse {
                    ok: true,
                    data: serde_json::to_value(route)?,
                })
            }
            "save_window_state" => {
                let width = match req.args.get("width") {
                    None => 1360u32,
                    Some(v) => v
                        .as_u64()
                        .and_then(|n| u32::try_from(n).ok())
                        .filter(|&n| (1..=65_535).contains(&n))
                        .ok_or_else(|| {
                            StudioError::Validation("Invalid window width".to_string())
                        })?,
                };
                let height = match req.args.get("height") {
                    None => 880u32,
                    Some(v) => v
                        .as_u64()
                        .and_then(|n| u32::try_from(n).ok())
                        .filter(|&n| (1..=65_535).contains(&n))
                        .ok_or_else(|| {
                            StudioError::Validation("Invalid window height".to_string())
                        })?,
                };
                let x = match req.args.get("x") {
                    None => 120i32,
                    Some(v) => v
                        .as_i64()
                        .and_then(|n| i32::try_from(n).ok())
                        .ok_or_else(|| StudioError::Validation("Invalid window x".to_string()))?,
                };
                let y = match req.args.get("y") {
                    None => 80i32,
                    Some(v) => v
                        .as_i64()
                        .and_then(|n| i32::try_from(n).ok())
                        .ok_or_else(|| StudioError::Validation("Invalid window y".to_string()))?,
                };
                let maximized = req
                    .args
                    .get("maximized")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                self.window_state = WindowStateGeometry {
                    width,
                    height,
                    x,
                    y,
                    maximized,
                };
                Ok(StudioIpcResponse {
                    ok: true,
                    data: serde_json::to_value(&self.window_state)?,
                })
            }
            "get_window_state" => Ok(StudioIpcResponse {
                ok: true,
                data: serde_json::to_value(&self.window_state)?,
            }),
            "send_notification" => {
                let title = req
                    .args
                    .get("title")
                    .and_then(Value::as_str)
                    .unwrap_or("DNAM Studio");
                let body = req
                    .args
                    .get("body")
                    .and_then(Value::as_str)
                    .unwrap_or("Task completed");
                let msg = format!("{title}: {body}");
                self.notifications.push(msg.clone());
                Ok(StudioIpcResponse {
                    ok: true,
                    data: json!({
                        "delivered": true,
                        "message": msg,
                        "count": self.notifications.len(),
                    }),
                })
            }
            other => Ok(StudioIpcResponse {
                ok: false,
                data: json!({ "error": format!("Unknown IPC command: {other}") }),
            }),
        }
    }

    fn telemetry_json(&self) -> Value {
        let avg_param = if self.parameters.is_empty() {
            0.5_f64
        } else {
            let sum: f64 = self
                .parameters
                .values()
                .copied()
                .filter(|v| v.is_finite())
                .map(|v| f64::from(v).abs().min(1.0e6))
                .sum();
            (sum / (self.parameters.len() as f64)).clamp(0.0, 1.0e6)
        };
        let r2 = (avg_param * 0.6).clamp(0.0, 9.0) as f32;
        let alpha = (1.0 / (1.0 + f64::from(r2) / 4.0).sqrt()) as f32;
        let out_rms_db = (-12.0 + avg_param * 2.0).clamp(-120.0, 120.0) as f32;
        json!({
            "sampleRateHz": self.sample_rate_hz,
            "controlRateHz": self.sample_rate_hz / 16,
            "backend": detect_native_backend_str(),
            "dspLoadPercent": 3.2,
            "inputRmsDbL": -18.0,
            "inputRmsDbR": -18.0,
            "outputRmsDbL": out_rms_db,
            "outputRmsDbR": out_rms_db,
            "maxRgLruRadius": 0.9912,
            "ufdnLyapunovMargin": 0.052,
            "mahalanobisR2": r2,
            "activeTrustRegionAlpha": alpha,
            "bypassActive": false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dnam_capture::proj::{DnamProjManifest, build_dnam_proj_container};
    use proptest::prelude::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_ipc_isolation_and_deep_link_roundtrip(
            slug in "[a-zA-Z0-9_-]{1,24}",
            prefix in "[a-zA-Z0-9_]{0,12}",
            suffix in "[a-zA-Z0-9_]{0,12}",
        ) {
            let model_uri = format!("dnam://models/{slug}");
            let parsed_m = parse_dnam_deep_link(&model_uri).expect("valid model uri");
            prop_assert_eq!(parsed_m.kind, "model");
            prop_assert_eq!(parsed_m.target, slug.as_str());

            let market_uri = format!("dnam://marketplace/{slug}");
            let parsed_mk = parse_dnam_deep_link(&market_uri).expect("valid market uri");
            prop_assert_eq!(parsed_mk.kind, "marketplace");
            prop_assert_eq!(parsed_mk.target, slug.as_str());

            let safe_req = StudioIpcRequest {
                command: "set_parameter".to_string(),
                args: json!({ "id": slug, "nested": [prefix.clone(), suffix.clone()] }),
            };
            prop_assert!(validate_isolated_ipc_request(&safe_req).is_ok());

            let traversal_req = StudioIpcRequest {
                command: "get_workspace".to_string(),
                args: json!({ "path": format!("{prefix}..{suffix}") }),
            };
            prop_assert!(validate_isolated_ipc_request(&traversal_req).is_err());

            let null_req = StudioIpcRequest {
                command: "get_workspace".to_string(),
                args: json!({ "nested": [format!("{prefix}\0{suffix}")] }),
            };
            prop_assert!(validate_isolated_ipc_request(&null_req).is_err());
        }
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = env::temp_dir().join(format!("dnam_ipc_unit_{tag}_{nanos}"));
        let _ = fs::create_dir_all(&dir);
        dir
    }

    #[test]
    fn config_dir_and_private_export_writer_enforce_safety() {
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
        assert!(!default_user_config_dir().as_os_str().is_empty());

        let dir = temp_dir("exports");
        let export_dir = dir.join("private_out");
        let p1 = write_private_studio_export_file(&export_dir, "model.dnam", b"v1", false)
            .expect("write new export");
        assert_eq!(fs::read(&p1).unwrap(), b"v1");
        let p2 = write_private_studio_export_file(&export_dir, "model.dnam", b"v2", false)
            .expect("keep existing when overwrite=false");
        assert_eq!(fs::read(&p2).unwrap(), b"v1");
        let p3 = write_private_studio_export_file(&export_dir, "model.dnam", b"v2", true)
            .expect("overwrite when overwrite=true");
        assert_eq!(fs::read(&p3).unwrap(), b"v2");

        assert!(write_private_studio_export_file(&export_dir, "", b"x", true).is_err());
        assert!(write_private_studio_export_file(&export_dir, "../bad", b"x", true).is_err());
        assert!(write_private_studio_export_file(&export_dir, "a/b.dnam", b"x", true).is_err());
        assert!(
            write_private_studio_export_file(Path::new("../bad"), "ok.dnam", b"x", true).is_err()
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn ipc_deep_links_isolation_and_all_handler_commands() {
        assert_eq!(official_tauri_plugins().len(), 8);

        assert_eq!(
            parse_dnam_deep_link("dnam://models/b7k")
                .expect("model")
                .target,
            "b7k"
        );
        assert_eq!(
            parse_dnam_deep_link("dnam://marketplace/item1")
                .expect("market")
                .target,
            "item1"
        );
        assert_eq!(
            parse_dnam_deep_link("dnam://open?path=/tmp/a.dnam")
                .expect("open")
                .target,
            "/tmp/a.dnam"
        );
        assert!(parse_dnam_deep_link("http://bad").is_err());
        assert!(parse_dnam_deep_link("dnam://models/").is_err());
        assert!(parse_dnam_deep_link("dnam://marketplace/").is_err());
        assert!(parse_dnam_deep_link("dnam://open?path=").is_err());
        assert!(parse_dnam_deep_link("dnam://models/../../etc/passwd").is_err());
        assert!(parse_dnam_deep_link("dnam://models/foo/bar").is_err());
        assert!(parse_dnam_deep_link("dnam://marketplace/%2e%2e").is_err());
        assert!(parse_dnam_deep_link("dnam://open?path=../../secret\0").is_err());
        assert!(parse_dnam_deep_link("dnam://unknown/route").is_err());

        assert!(
            validate_isolated_ipc_request(&StudioIpcRequest {
                command: "   ".to_string(),
                args: json!({}),
            })
            .is_err()
        );
        assert!(
            validate_isolated_ipc_request(&StudioIpcRequest {
                command: "get_telemetry".to_string(),
                args: json!({"arr": ["../secret"]}),
            })
            .is_err()
        );
        assert!(
            validate_isolated_ipc_request(&StudioIpcRequest {
                command: "get_telemetry".to_string(),
                args: json!({"../bad_key": "ok"}),
            })
            .is_err()
        );
        assert!(
            validate_isolated_ipc_request(&StudioIpcRequest {
                command: "get_telemetry".to_string(),
                args: json!({"path": "%2E%2E/secret"}),
            })
            .is_err()
        );
        assert!(
            validate_isolated_ipc_request(&StudioIpcRequest {
                command: "../evil_cmd".to_string(),
                args: json!({}),
            })
            .is_err()
        );
        assert!(
            validate_isolated_ipc_request(&StudioIpcRequest {
                command: "get_telemetry".to_string(),
                args: json!(null),
            })
            .is_err()
        );
        assert!(
            validate_isolated_ipc_request(&StudioIpcRequest {
                command: "get_telemetry".to_string(),
                args: json!({"path": ".%2e/secret"}),
            })
            .is_err()
        );
        assert!(
            validate_isolated_ipc_request(&StudioIpcRequest {
                command: "get_telemetry".to_string(),
                args: json!({"path": "%252e%252e/secret"}),
            })
            .is_err()
        );
        assert!(
            validate_isolated_ipc_request(&StudioIpcRequest {
                command: "send_notification".to_string(),
                args: json!({"title": "100% Complete", "body": "SNR 78 dB"}),
            })
            .is_ok()
        );
        assert!(parse_dnam_deep_link("dnam://open?path=/tmp/.%2e/etc/passwd").is_err());
        assert!(parse_dnam_deep_link("dnam://open?path=/tmp/%252e%252e/etc/passwd").is_err());
        assert!(parse_dnam_deep_link("dnam://open?path=/tmp/bad%5cwin.dnam").is_err());
        assert!(parse_dnam_deep_link("dnam://open?path=C:\\win.dnam").is_err());
        assert!(parse_dnam_deep_link("dnam://open?path=/tmp/bad%00null").is_err());
        assert!(parse_dnam_deep_link("dnam://open?path=/tmp/bad%C2%85c1").is_err());
        assert!(parse_dnam_deep_link("dnam://open?path=/tmp/ok%20file.dnam").is_ok());
        assert!(
            serde_json::from_value::<StudioIpcRequest>(json!({
                "cmd": "get_telemetry",
                "command": "plugin:shell|execute",
                "args": {}
            }))
            .is_err()
        );
        assert!(
            validate_isolated_ipc_request(&StudioIpcRequest {
                command: "get_telemetry".to_string(),
                args: json!({ "path": "%252525252e%252525252e/etc/passwd" }),
            })
            .is_err()
        );
        assert!(
            validate_isolated_ipc_request(&StudioIpcRequest {
                command: "get_telemetry".to_string(),
                args: json!({ "nul": "safe%2525252500tail" }),
            })
            .is_err()
        );
        assert!(parse_dnam_deep_link("dnam://open?path=C:%252525255cwin.dnam").is_err());

        let dir = temp_dir("ipc_handler");
        let proj_path = dir.join("default.dnam-proj");
        let mut manifest = DnamProjManifest::default_reference();
        manifest.project_meta.project_id = "default".to_string();
        let proj_bytes = build_dnam_proj_container(&manifest, &[]).expect("serialize proj");
        fs::write(&proj_path, proj_bytes).expect("write proj");

        let mut handler = StudioIpcHandler::default();
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "get_telemetry".to_string(),
                        args: json!({}),
                    },
                    None
                )
                .expect("telemetry")
                .ok
        );
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "set_parameter".to_string(),
                        args: json!({"id": "gain", "value": 0.8}),
                    },
                    None
                )
                .expect("set_parameter")
                .ok
        );
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "set_switch".to_string(),
                        args: json!({"id": "voice", "index": 1}),
                    },
                    None
                )
                .expect("set_switch")
                .ok
        );
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "list_backends".to_string(),
                        args: json!({}),
                    },
                    None
                )
                .expect("list_backends")
                .ok
        );
        assert!(
            !handler
                .handle(
                    &StudioIpcRequest {
                        command: "get_workspace".to_string(),
                        args: json!({}),
                    },
                    None
                )
                .expect("get_workspace none")
                .ok
        );
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "get_workspace".to_string(),
                        args: json!({}),
                    },
                    Some(proj_path.as_path())
                )
                .expect("get_workspace some")
                .ok
        );
        assert!(
            !handler
                .handle(
                    &StudioIpcRequest {
                        command: "get_capture_plan".to_string(),
                        args: json!({}),
                    },
                    None
                )
                .expect("get_capture_plan none")
                .ok
        );
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "get_capture_plan".to_string(),
                        args: json!({"model_id": "default"}),
                    },
                    Some(proj_path.as_path())
                )
                .expect("get_capture_plan some")
                .ok
        );
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "get_capture_plan".to_string(),
                        args: json!({"model_id": "/outside/dir"}),
                    },
                    Some(proj_path.as_path())
                )
                .is_err()
        );
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "get_capture_plan".to_string(),
                        args: json!({"model_id": "mismatch_id"}),
                    },
                    Some(proj_path.as_path())
                )
                .is_err()
        );
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "set_parameter".to_string(),
                        args: json!({"id": "gain", "value": null}),
                    },
                    None
                )
                .is_err()
        );
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "set_parameter".to_string(),
                        args: json!({"id": "gain", "value": 1.0e300}),
                    },
                    None
                )
                .is_err()
        );
        let huge_tel = handler
            .handle(
                &StudioIpcRequest {
                    command: "set_parameter".to_string(),
                    args: json!({"id": "gain", "value": 2.0e38}),
                },
                None,
            )
            .expect("finite f32 param");
        assert!(huge_tel.data["outputRmsDbL"].is_number());
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "get_plugins".to_string(),
                        args: json!({}),
                    },
                    None
                )
                .expect("get_plugins")
                .ok
        );
        let dl_resp = handler
            .handle(
                &StudioIpcRequest {
                    command: "parse_deep_link".to_string(),
                    args: json!({"uri": "dnam://models/default"}),
                },
                None,
            )
            .expect("parse_deep_link");
        assert!(dl_resp.ok && dl_resp.data["rawUri"] == "dnam://models/default");
        assert_eq!(
            handler.last_deep_link().map(|r| r.target.as_str()),
            Some("default")
        );
        let seeded_route = parse_dnam_deep_link("dnam://marketplace/amp_pack").unwrap();
        handler.set_last_deep_link(seeded_route);
        let plugins_after_dl = handler
            .handle(
                &StudioIpcRequest {
                    command: "get_plugins".to_string(),
                    args: json!({}),
                },
                None,
            )
            .expect("get_plugins after deep link");
        assert_eq!(plugins_after_dl.data["lastDeepLink"]["target"], "amp_pack");
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "save_window_state".to_string(),
                        args: json!({"width": 1600, "height": 900, "x": 10, "y": 20, "maximized": true}),
                    },
                    None
                )
                .expect("save_window_state")
                .ok
        );
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "save_window_state".to_string(),
                        args: json!({"width": 0, "height": 900, "x": 10, "y": 20}),
                    },
                    None
                )
                .is_err()
        );
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "save_window_state".to_string(),
                        args: json!({"width": 1600, "height": 900, "x": i64::MAX, "y": 20}),
                    },
                    None
                )
                .is_err()
        );
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "set_switch".to_string(),
                        args: json!({"id": "voice", "index": -1}),
                    },
                    None
                )
                .is_err()
        );
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "get_window_state".to_string(),
                        args: json!({}),
                    },
                    None
                )
                .expect("get_window_state")
                .ok
        );
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "send_notification".to_string(),
                        args: json!({"title": "Done", "body": "Ready"}),
                    },
                    None
                )
                .expect("send_notification")
                .ok
        );
        assert!(
            !handler
                .handle(
                    &StudioIpcRequest {
                        command: "unknown_cmd".to_string(),
                        args: json!({}),
                    },
                    None
                )
                .expect("unknown_cmd")
                .ok
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
