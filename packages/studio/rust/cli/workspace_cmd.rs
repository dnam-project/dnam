//! Workspace, model, control, and `dnam studio` subcommand handlers.

use std::path::Path;

use clap::Subcommand;
use serde_json::json;

use crate::cli::format;
use crate::error::StudioResult;
use crate::ipc::official_tauri_plugins;
use crate::project::{DnamProject, parse_compact_range_spec, parse_compact_switch_spec};
use crate::runner::detect_native_backend_str;

/// Subcommands under `dnam model`.
#[derive(Debug, Subcommand)]
pub enum ModelAction {
    /// Add a new model to `models/<id>/model.toml`.
    Add {
        /// Unique model identifier.
        #[arg(long)]
        id: String,
        /// Human-readable title.
        #[arg(long)]
        title: String,
        /// Target tier (`Lite`, `Full`, `Ultra`).
        #[arg(long, default_value = "Full")]
        tier: String,
    },
    /// Show details of a model.
    Show {
        /// Model identifier.
        #[arg(long)]
        id: String,
    },
}

/// Subcommands under `dnam control`.
#[derive(Debug, Subcommand)]
pub enum ControlAction {
    /// Add a continuous range control using compact spec `"id:Label:min..max@default[:unit]"`.
    AddRange {
        /// Target model identifier.
        #[arg(long)]
        model: String,
        /// Compact range specification.
        #[arg(long)]
        spec: String,
    },
    /// Add a discrete switch control using compact spec `"id:Label:Opt0,Opt1@default_idx"`.
    AddSwitch {
        /// Target model identifier.
        #[arg(long)]
        model: String,
        /// Compact switch specification.
        #[arg(long)]
        spec: String,
    },
}

/// Executes `dnam init`.
pub fn run_init(workspace: &Path, name: &str, author: &str) -> StudioResult<String> {
    let proj = DnamProject::init(workspace, name, author)?;
    Ok(format!(
        "Initialized DNAM workspace '{}' at {} (pubkey: {}...)",
        proj.manifest().project.name,
        proj.root_dir().display(),
        &proj.manifest().project.creator_public_key_hex[..16]
    ))
}

/// Executes `dnam status`.
pub fn run_status(workspace: &Path) -> StudioResult<String> {
    let proj = DnamProject::open(workspace)?;
    let mut models = Vec::new();
    for model_id in &proj.manifest().models {
        if let Ok(m) = proj.load_model(model_id) {
            models.push(m);
        }
    }
    Ok(format::format_workspace_status(proj.manifest(), &models))
}

/// Executes `dnam model`.
pub fn run_model(workspace: &Path, action: &ModelAction) -> StudioResult<String> {
    let mut proj = DnamProject::open(workspace)?;
    match action {
        ModelAction::Add { id, title, tier } => {
            let model = proj.add_model(id, title, tier)?;
            Ok(format!(
                "Created model '{}' ({}, tier={})",
                model.model_id, model.title, model.tier
            ))
        }
        ModelAction::Show { id } => {
            let model = proj.load_model(id)?;
            Ok(serde_json::to_string_pretty(&model)?)
        }
    }
}

/// Executes `dnam control`.
pub fn run_control(workspace: &Path, action: &ControlAction) -> StudioResult<String> {
    let proj = DnamProject::open(workspace)?;
    match action {
        ControlAction::AddRange { model, spec } => {
            let parsed = parse_compact_range_spec(spec)?;
            let id = parsed.id.clone();
            proj.add_range_control(model, parsed)?;
            Ok(format!("Added range control '{id}' to model '{model}'"))
        }
        ControlAction::AddSwitch { model, spec } => {
            let parsed = parse_compact_switch_spec(spec)?;
            let id = parsed.id.clone();
            proj.add_switch_control(model, parsed)?;
            Ok(format!("Added switch control '{id}' to model '{model}'"))
        }
    }
}

/// Executes `dnam studio`.
pub fn run_studio(workspace: &Path, headless_check: bool) -> StudioResult<String> {
    let plugins = official_tauri_plugins();
    let tier1: Vec<&str> = plugins
        .iter()
        .filter(|p| p.tier == 1)
        .map(|p| p.id.as_str())
        .collect();
    let tier2: Vec<&str> = plugins
        .iter()
        .filter(|p| p.tier == 2)
        .map(|p| p.id.as_str())
        .collect();
    let payload = json!({
        "app": "dnam-studio",
        "tauriVersion": "2.0",
        "workspace": workspace.display().to_string(),
        "nativeAudioHost": detect_native_backend_str(),
        "headlessCheck": headless_check,
        "singleInstanceLocked": true,
        "deepLinkScheme": "dnam://",
        "ipcPattern": "isolation",
        "isolationDir": "./isolation",
        "freezePrototype": true,
        "assetProtocolEnabled": false,
        "officialPlugins": {
            "tier1": tier1,
            "tier2": tier2,
        }
    });
    Ok(serde_json::to_string_pretty(&payload)?)
}
