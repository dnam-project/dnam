//! `dnam train` and `dnam trainer {status,install,uninstall}` subcommand handlers.

use std::fs;
use std::path::Path;

use clap::Subcommand;

use crate::error::StudioResult;
use crate::project::DnamProject;
use crate::trainer::{TrainerEnvManager, build_training_request_json};

/// Subcommands under `dnam trainer`.
#[derive(Debug, Subcommand)]
pub enum TrainerAction {
    /// Inspect `uv` discovery and `<config_dir>/trainer-venv` installation status.
    Status,
    /// Provision or update `<config_dir>/trainer-venv` via `uv`.
    Install {
        /// Simulate installation without downloading packages.
        #[arg(long)]
        dry_run: bool,
    },
    /// Remove the managed `<config_dir>/trainer-venv` directory.
    Uninstall,
}

/// Executes `dnam train`.
pub fn run_train(
    workspace: &Path,
    model_id: &str,
    epochs: Option<u32>,
    dry_run: bool,
) -> StudioResult<String> {
    let proj = DnamProject::open(workspace)?;
    let model = proj.load_model(model_id)?;
    let req_json = build_training_request_json(proj.manifest(), &model, workspace, epochs);
    let model_dir = workspace.join("models").join(model_id);
    let req_path = model_dir.join("reports").join("training_request.json");
    if let Some(parent) = req_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let encoded = serde_json::to_string_pretty(&req_json)?;
    fs::write(&req_path, &encoded)?;

    if dry_run {
        return Ok(format!(
            "Wrote training request JSON for '{model_id}' to {}",
            req_path.display()
        ));
    }

    Ok(format!(
        "Prepared training request JSON for '{model_id}' at {}",
        req_path.display()
    ))
}

/// Executes `dnam trainer {status,install,uninstall}`.
pub fn run_trainer_env(workspace: &Path, action: &TrainerAction) -> StudioResult<String> {
    let config_dir = workspace.join(".dnam-studio");
    let mgr = TrainerEnvManager::new(&config_dir, Some(workspace.to_path_buf()));
    match action {
        TrainerAction::Status => {
            let st = mgr.status();
            Ok(serde_json::to_string_pretty(&st)?)
        }
        TrainerAction::Install { dry_run } => {
            let st = mgr.install(*dry_run)?;
            Ok(serde_json::to_string_pretty(&st)?)
        }
        TrainerAction::Uninstall => {
            let st = mgr.uninstall()?;
            Ok(serde_json::to_string_pretty(&st)?)
        }
    }
}
