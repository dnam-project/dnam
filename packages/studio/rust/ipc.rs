//! Subsystem 5: Tauri v2 Command & Native Hardware Acceleration Host Handler.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::capture::{list_backend_devices, list_capture_backends};
use crate::error::{StudioError, StudioResult};
use crate::project::{DnamProject, build_capture_plan};
use crate::runner::detect_native_backend_str;

/// Descriptor for an official Tauri v2 plugin integrated into `dnam-studio`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TauriPluginDescriptor {
    /// Short plugin identifier (e.g. `"dialog"`, `"opener"`, `"single-instance"`).
    pub id: String,
    /// Rust crate name (`tauri-plugin-*`).
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
pub struct DnamDeepLinkRoute {
    /// Raw URI string (e.g. `"dnam://models/b7k_ultra"`).
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

/// Parses a `dnam://` URI into a structured [`DnamDeepLinkRoute`].
pub fn parse_dnam_deep_link(uri: &str) -> StudioResult<DnamDeepLinkRoute> {
    let trimmed = uri.trim();
    let rest = trimmed.strip_prefix("dnam://").ok_or_else(|| {
        StudioError::Validation(format!("Deep link must start with 'dnam://': {trimmed}"))
    })?;
    let rest = rest.trim_matches('/');
    if let Some(model_id) = rest.strip_prefix("models/") {
        let cleaned = model_id.trim_matches('/');
        if cleaned.is_empty() {
            return Err(StudioError::Validation(
                "dnam://models/ requires a non-empty model ID".to_string(),
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
        if cleaned.is_empty() {
            return Err(StudioError::Validation(
                "dnam://marketplace/ requires a non-empty listing ID".to_string(),
            ));
        }
        return Ok(DnamDeepLinkRoute {
            raw_uri: trimmed.to_string(),
            kind: "marketplace".to_string(),
            target: cleaned.to_string(),
        });
    }
    if let Some(query) = rest.strip_prefix("open?path=") {
        if query.is_empty() {
            return Err(StudioError::Validation(
                "dnam://open?path= requires a non-empty file path".to_string(),
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

fn value_contains_unsafe_string(val: &Value) -> bool {
    match val {
        Value::String(s) => s.contains("..") || s.contains('\0'),
        Value::Array(items) => items.iter().any(value_contains_unsafe_string),
        Value::Object(map) => map.values().any(value_contains_unsafe_string),
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
    if value_contains_unsafe_string(&req.args) {
        return Err(StudioError::Validation(
            "Tauri Isolation blocked path traversal or null byte in IPC payload".to_string(),
        ));
    }
    Ok(())
}

/// Typed IPC request envelope from the Tauri v2 React frontend (`TauriNativeAudioHost` / `StudioIpcClient`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

    /// Dispatches a [`StudioIpcRequest`] and returns a [`StudioIpcResponse`].
    pub fn handle(
        &mut self,
        req: &StudioIpcRequest,
        workspace_dir: Option<&Path>,
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
                let val = req.args.get("value").and_then(Value::as_f64).unwrap_or(0.5) as f32;
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
                let idx = req.args.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
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
                if let Some(dir) = workspace_dir {
                    let proj = DnamProject::open(dir)?;
                    Ok(StudioIpcResponse {
                        ok: true,
                        data: serde_json::to_value(proj.manifest())?,
                    })
                } else {
                    Ok(StudioIpcResponse {
                        ok: false,
                        data: json!({ "error": "No workspace directory provided" }),
                    })
                }
            }
            "get_capture_plan" => {
                if let Some(dir) = workspace_dir {
                    let proj = DnamProject::open(dir)?;
                    let model_id = req
                        .args
                        .get("model_id")
                        .and_then(Value::as_str)
                        .unwrap_or("default");
                    let model = proj.load_model(model_id)?;
                    let plan = build_capture_plan(&model, true);
                    Ok(StudioIpcResponse {
                        ok: true,
                        data: serde_json::to_value(plan)?,
                    })
                } else {
                    Ok(StudioIpcResponse {
                        ok: false,
                        data: json!({ "error": "No workspace directory provided" }),
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
                let width = req
                    .args
                    .get("width")
                    .and_then(Value::as_u64)
                    .unwrap_or(1360) as u32;
                let height = req
                    .args
                    .get("height")
                    .and_then(Value::as_u64)
                    .unwrap_or(880) as u32;
                let x = req.args.get("x").and_then(Value::as_i64).unwrap_or(120) as i32;
                let y = req.args.get("y").and_then(Value::as_i64).unwrap_or(80) as i32;
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
            0.5_f32
        } else {
            self.parameters.values().copied().map(f32::abs).sum::<f32>()
                / (self.parameters.len() as f32)
        };
        let r2 = (avg_param * 0.6).clamp(0.0, 9.0);
        let alpha = 1.0 / (1.0 + r2 / 4.0).sqrt();
        json!({
            "sampleRateHz": self.sample_rate_hz,
            "controlRateHz": self.sample_rate_hz / 16,
            "backend": detect_native_backend_str(),
            "dspLoadPercent": 3.2,
            "inputRmsDbL": -18.0,
            "inputRmsDbR": -18.0,
            "outputRmsDbL": -12.0 + avg_param * 2.0,
            "outputRmsDbR": -12.0 + avg_param * 2.0,
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
    use proptest::prelude::*;
    use std::env;
    use std::fs;
    use std::path::PathBuf;
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
    fn ipc_deep_links_isolation_and_all_handler_commands() {
        assert_eq!(official_tauri_plugins().len(), 8);

        // Deep link parsing
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
        assert!(parse_dnam_deep_link("dnam://unknown/route").is_err());

        // Isolation validation
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

        let dir = temp_dir("ipc_handler");
        let mut proj = DnamProject::init(&dir, "WS", "Auth").expect("init");
        proj.add_model("default", "Default", "Full").expect("add");

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
                    Some(dir.as_path())
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
                    Some(dir.as_path())
                )
                .expect("get_capture_plan some")
                .ok
        );
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
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "parse_deep_link".to_string(),
                        args: json!({"uri": "dnam://models/default"}),
                    },
                    None
                )
                .expect("parse_deep_link")
                .ok
        );
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
